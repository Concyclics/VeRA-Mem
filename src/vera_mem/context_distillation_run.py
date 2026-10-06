"""Controlled context-teacher distillation through the writable VeRA interface.

The frozen teacher receives an observed fact as text; the student receives the
question and its actual sparse VeRA memory reads. Prefixes are exact token IDs.
Only the historical offline train/dev cache is selected. Development evaluation
never selects a checkpoint; all methods finish at the specified final update.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from . import augmentation_data as data
from .context_distillation import (
    ContextDistillationBackend, distillation_loss, hidden_alignment_loss,
)
from .data import Example
from .factcentric_losses import style_block_retrieval_loss
from .metrics import exact_match, save_examples
from .run import json_write, tensor_digest
from .scaling_run import evaluate, stats
from .stable_vector_vera import StableVectorVeRA
from .vector_store import PersistentVectorDB

REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
METHODS = ("ce", "off_kd", "off_kd_hidden", "on_kd", "on_kd_hidden")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_digest(model):
    """Hash frozen parameter bytes, supporting BF16 without lossy conversion."""
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def append_jsonl(path, rows):
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def select_offline_packet(packet):
    """Copy only explicitly allowed train/dev references, never test/control."""
    if packet["protocol"] != "generalization-v1-layer20" or packet["model_revision"] != REVISION:
        raise ValueError("Pinned feature-cache provenance required")
    if packet["data_fingerprint"] != data.protocol_fingerprint():
        raise ValueError("Feature-cache data fingerprint mismatch")
    selected = {name: packet[name] for name in ("protocol", "model_revision", "data_fingerprint")}
    for name in ("examples", "features", "template_ids"):
        selected[name] = {split: packet[name][split] for split in ("train", "dev")}
    return selected


def validate_packet(packet):
    for split, count in (("train", 4096), ("dev", 64)):
        if len(packet["examples"][split]) != count:
            raise ValueError("Unexpected offline fact count")
        for domain, views in (("q", 8 if split == "train" else 3),
                              ("s", 4 if split == "train" else 3)):
            features = packet["features"][split][domain]
            if tuple(features.shape) != (count, views, 9728) or not torch.isfinite(features).all():
                raise ValueError(f"Invalid {split}/{domain} features")
            if len(packet["template_ids"][split][domain]) != views:
                raise ValueError("Template count mismatch")


def sequence_mean(token_losses, batch_indices, counts):
    """Every fact gets equal weight, regardless of sampled trajectory length."""
    if token_losses.ndim != 1 or token_losses.shape != batch_indices.shape:
        raise ValueError("Expected equally sized per-token losses and sequence indices")
    if counts.ndim != 1 or counts.numel() == 0 or bool((counts <= 0).any()):
        raise ValueError("Each sequence must contain prediction tokens")
    if int(counts.sum()) != len(token_losses):
        raise ValueError("Sequence counts do not account for every token")
    sums = token_losses.new_zeros(len(counts)).index_add(0, batch_indices, token_losses)
    return (sums / counts).mean()


def gold_tokens(backend, answers):
    eos = backend.tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Gold supervision requires tokenizer EOS")
    return [list(backend.tokenizer.encode(answer, add_special_tokens=False)) + [eos] for answer in answers]


def clear_bank(backend):
    backend.vector_keys = backend.vector_values = None
    backend.vector_store = backend.vector_override = backend.oracle_values = None
    backend.prefill_retrieval = None
    backend.mode = "none"
    backend.trace_retrieval = False


def set_bank(backend, module, support, detach=False):
    """Never let rollout no_grad overwrite the differentiable training bank."""
    backend.mode, backend.vector_vera = "vector_vera", module
    backend.vector_store = backend.vector_override = backend.oracle_values = None
    backend.trace_retrieval = False
    backend.prefill_retrieval = None
    if detach:
        with torch.no_grad():
            backend.vector_keys = module.encode_key(support).detach()
            backend.vector_values = module.encode_value(support).detach()
    else:
        backend.vector_keys = module.encode_key(support)
        backend.vector_values = module.encode_value(support)
        if (backend.vector_store is not None or not backend.vector_keys.requires_grad
                or not backend.vector_values.requires_grad):
            raise AssertionError("Training must use the differentiable tensor bank")


def supervised_loss(branch, tokens):
    labels = torch.tensor([token for row in tokens for token in row],
                          dtype=torch.long, device=branch["logits"].device)
    loss = F.cross_entropy(branch["logits"], labels, reduction="none")
    return sequence_mean(loss, branch["batch_indices"], branch["counts"]), loss


def distill_branches(student, teacher, with_hidden):
    for field in ("batch_indices", "counts"):
        if not torch.equal(student[field], teacher[field]):
            raise AssertionError("Teacher/student continuation alignment mismatch")
    # Absolute positions deliberately differ because teacher alone has context.
    kl_tokens = distillation_loss(student["logits"], teacher["logits"],
                                  direction="reverse", temperature=1., reduction="none")
    hidden_tokens = hidden_alignment_loss(student["hidden"], teacher["hidden"], reduction="none")
    kl = sequence_mean(kl_tokens, student["batch_indices"], student["counts"])
    hidden = sequence_mean(hidden_tokens, student["batch_indices"], student["counts"])
    return kl + (0.1 * hidden if with_hidden else 0.), kl, hidden, kl_tokens, hidden_tokens


class EpisodeSampler:
    """Fact order, negative facts and expressions are independent RNG streams."""
    def __init__(self, size, batch_size, seed, negatives=64):
        if not 1 <= batch_size <= size:
            raise ValueError("Invalid target batch size")
        self.size, self.batch_size, self.negatives = size, batch_size, negatives
        self.fact_rng = random.Random(seed + 100)
        self.negative_rng = random.Random(seed + 110)
        self.view_rng = random.Random(seed + 120)
        self.order, self.cursor = [], 0

    def next(self, query_views=8, support_views=4):
        if self.cursor + self.batch_size > len(self.order):
            self.order = list(range(self.size))
            self.fact_rng.shuffle(self.order)
            self.cursor = 0
        targets = self.order[self.cursor:self.cursor+self.batch_size]
        self.cursor += len(targets)
        remaining = [i for i in range(self.size) if i not in set(targets)]
        episode = targets + self.negative_rng.sample(remaining, min(self.negatives, len(remaining)))
        self.negative_rng.shuffle(episode)
        qviews = [self.view_rng.randrange(query_views) for _ in episode]
        sviews = [self.view_rng.randrange(support_views) for _ in episode]
        return dict(targets=targets, episode=episode, qviews=qviews, sviews=sviews,
                    mapping=[episode.index(i) for i in targets])


def training_step(backend, module, query_views, support_views, questions, contexts,
                  answers, mapping, method, cfg, generator, replay_questions=None):
    """Build one loss graph. The caller owns zero_grad/backward/optimizer step."""
    golden = gold_tokens(backend, answers)
    rollout_input_tokens = 0
    rollout_seconds = 0.
    if method.startswith("on_"):
        set_bank(backend, module, support_views["chosen"], detach=True)
        start = time.perf_counter()
        tokens = backend.sample_student(questions, max_new_tokens=cfg["max_new_tokens"],
                                        temperature=cfg["sampling_temperature"], generator=generator)
        rollout_seconds = time.perf_counter() - start
        # The transparent sampler recomputes prompt+prefix at every token.
        rollout_input_tokens = sum(len(backend.prompt_ids(q))*len(t) + len(t)*(len(t)-1)//2
                                   for q,t in zip(questions,tokens))
    else:
        tokens = golden
    if len(tokens) != len(questions) or any(not row for row in tokens):
        raise ValueError("Every target requires an actual nonempty continuation")
    # Recompute with gradients after sampling; detached rollout values must not
    # accidentally become the training bank. Teacher runs first and cannot read
    # this memory because its backend scope disables the adapter.
    set_bank(backend, module, support_views["chosen"])
    teacher = None
    if method != "ce":
        with torch.no_grad():
            teacher = backend.forward_sequences(questions, tokens, contexts=contexts, teacher=True)
    student = backend.forward_sequences(questions, tokens)
    zero = student["logits"].new_zeros(())
    if method == "ce":
        main, main_tokens = supervised_loss(student, tokens)
        kl, hidden = zero, zero
        hidden_tokens = torch.zeros_like(main_tokens)
    else:
        main, kl, hidden, main_tokens, hidden_tokens = distill_branches(student, teacher, method.endswith("_hidden"))
    all_queries = module.encode_query(query_views)
    all_keys = module.encode_key(support_views["all"])
    address_views = style_block_retrieval_loss(all_queries, all_keys, temperature=.1)["retrieval"]
    target_columns = torch.as_tensor(mapping, device=student["logits"].device)
    predicted_query = module.encode_query(student["query_inputs"])
    address_tokens = F.cross_entropy(predicted_query @ backend.vector_keys.T / .1,
                                     target_columns[student["batch_indices"]], reduction="none")
    address_actual = sequence_mean(address_tokens, student["batch_indices"], student["counts"])
    # All actual prefix positions, including after wrong student tokens, retain
    # the current question's fact as address target. This is explicit auxiliary
    # supervision, not an inference-time entity parser or oracle read.
    total = main + .2 * address_views + .2 * address_actual
    replay = zero
    replay_target_tokens = replay_input_tokens = 0
    if replay_questions is not None:
        set_bank(backend, module, support_views["all"][:, 0])
        replay_branch = backend.forward_sequences(replay_questions, golden)
        replay, _ = supervised_loss(replay_branch, golden)
        total = total + .25 * replay
        replay_target_tokens = sum(map(len, golden))
        replay_input_tokens = sum(len(backend.prompt_ids(q))+len(t) for q,t in zip(replay_questions,golden))
    student_input_tokens = sum(len(backend.prompt_ids(q))+len(t) for q,t in zip(questions,tokens))
    teacher_input_tokens = (sum(len(backend.prompt_ids(q,c))+len(t) for q,c,t in zip(questions,contexts,tokens))
                            if teacher is not None else 0)
    metrics = dict(total_loss=float(total.detach()), main_loss=float(main.detach()),
                   reverse_kl=float(kl.detach()) if teacher is not None else None,
                   hidden_loss=float(hidden.detach()) if teacher is not None else None,
                   hidden_weight=.1 if method.endswith("_hidden") else 0.,
                   all_view_address_loss=float(address_views.detach()),
                   actual_prefix_address_loss=float(address_actual.detach()),
                   canonical_replay_ce=float(replay.detach()) if replay_questions is not None else None,
                   main_target_tokens=sum(map(len,tokens)), teacher_target_tokens=sum(map(len,tokens)) if teacher is not None else 0,
                   sampled_tokens=sum(map(len,tokens)) if method.startswith("on_") else 0,
                   student_input_tokens=student_input_tokens,teacher_input_tokens=teacher_input_tokens,
                   rollout_input_tokens=rollout_input_tokens,rollout_seconds=rollout_seconds,
                   replay_target_tokens=replay_target_tokens,replay_input_tokens=replay_input_tokens,
                   main_token_losses=main_tokens.detach().cpu().tolist(),
                   hidden_token_losses=hidden_tokens.detach().cpu().tolist(),
                   actual_address_token_losses=address_tokens.detach().cpu().tolist())
    return total, metrics, tokens, golden


def train(backend, module, cfg, packet, output):
    examples = [Example(**row) for row in packet["examples"]["train"]]
    qcache, scache = [packet["features"]["train"][name].to(backend.device) for name in ("q","s")]
    names = packet["template_ids"]["train"]
    sampler = EpisodeSampler(len(examples),cfg["batch_size"],cfg["seed"])
    generator = torch.Generator(device=backend.device).manual_seed(cfg["seed"]+500)
    module.requires_grad_(True).train()
    parameters = [list(module.Wv.parameters()),[module.b],list(module.Wq.parameters())+list(module.Wk.parameters())]
    optimizer = torch.optim.Adam([dict(params=parameters[0],lr=1e-4),
                                  dict(params=parameters[1],lr=.005),dict(params=parameters[2],lr=1e-5)])
    fact_hash, episode_hash = hashlib.sha256(), hashlib.sha256()
    started = time.perf_counter()
    totals = {name:0 for name in ("main_target_tokens","teacher_target_tokens","sampled_tokens",
                                  "student_input_tokens","teacher_input_tokens","rollout_input_tokens",
                                  "replay_target_tokens","replay_input_tokens")}
    last_step = 0
    try:
        for step in range(1,cfg["updates"]+1):
            batch = sampler.next(qcache.shape[1],scache.shape[1])
            targets,episode,mapping = [batch[name] for name in ("targets","episode","mapping")]
            index = torch.tensor(episode,device=backend.device)
            target_examples = [examples[i] for i in targets]
            questions = [data.render_question(ex,names["q"][batch["qviews"][column]])
                         for ex,column in zip(target_examples,mapping)]
            contexts = [data.render_support(ex,names["s"][batch["sviews"][column]])
                        for ex,column in zip(target_examples,mapping)]
            replay_questions = ([data.render_question(ex,names["q"][0]) for ex in target_examples]
                                if step%4==0 else None)
            support_views = dict(chosen=scache[index,batch["sviews"]],all=scache[index])
            optimizer.zero_grad(set_to_none=True)
            loss,metrics,tokens,golden = training_step(
                backend,module,qcache[index],support_views,questions,contexts,[ex.answer for ex in target_examples],
                mapping,cfg["method"],cfg,generator,replay_questions)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            for group in parameters:
                torch.nn.utils.clip_grad_norm_(group,1.)
            optimizer.step()
            last_step = step
            clear_bank(backend)
            ids = [ex.id for ex in target_examples]
            episode_ids = [examples[i].id for i in episode]
            fact_hash.update(json.dumps(ids,separators=(",",":")).encode())
            episode_record = dict(ids=episode_ids,qviews=batch["qviews"],sviews=batch["sviews"],targets=ids)
            episode_hash.update(json.dumps(episode_record,separators=(",",":")).encode())
            for name in totals:
                totals[name] += metrics[name]
            row = dict(step=step,finished=False,method=cfg["method"],target_ids=ids,episode_ids=episode_ids,
                       query_view_indices=batch["qviews"],support_view_indices=batch["sviews"],
                       target_to_episode=mapping,elapsed_seconds=time.perf_counter()-started,**metrics)
            append_jsonl(output/"training.jsonl",[row])
            rollout_rows = []
            for i,(ex,column) in enumerate(zip(target_examples,mapping)):
                rollout_rows.append(dict(step=step,id=ex.id,source="student_sample" if cfg["method"].startswith("on_") else "gold",
                    query_template=names["q"][batch["qviews"][column]],support_template=names["s"][batch["sviews"][column]],
                    question=questions[i],teacher_context=contexts[i],
                    student_prompt_token_ids=backend.prompt_ids(questions[i]),
                    teacher_prompt_token_ids=backend.prompt_ids(questions[i],contexts[i]) if cfg["method"]!="ce" else None,
                    continuation_token_ids=tokens[i],gold_token_ids=golden[i],
                    eos_appended_by_runner=not cfg["method"].startswith("on_"),
                    canonical_replay_prompt_token_ids=backend.prompt_ids(replay_questions[i]) if replay_questions else None,
                    canonical_replay_token_ids=golden[i] if replay_questions else None))
            append_jsonl(output/"rollouts.jsonl",rollout_rows)
            print(json.dumps({key:row[key] for key in ("step","finished","method","main_loss","reverse_kl",
                               "hidden_loss","main_target_tokens","sampled_tokens","canonical_replay_ce","elapsed_seconds")}),flush=True)
            if step%64==0 or step==cfg["updates"]:
                torch.save(dict(module=module.state_dict(),config=cfg,step=step),output/"last.pt")
                json_write(output/"training_status.json",dict(step=step,finished=False,token_totals=totals,
                           target_exposure_sha256=fact_hash.hexdigest(),episode_schedule_sha256=episode_hash.hexdigest()))
            del loss
    finally:
        clear_bank(backend)
    module.requires_grad_(False).eval()
    result = dict(step=last_step,finished=True,selected_step=last_step,selection="fixed final update; no development selection",
                  target_exposures=last_step*cfg["batch_size"],canonical_replay_exposures=(last_step//4)*cfg["batch_size"],
                  target_exposure_sha256=fact_hash.hexdigest(),episode_schedule_sha256=episode_hash.hexdigest(),
                  token_totals=totals,elapsed_seconds=time.perf_counter()-started,
                  rollout_sha256=sha256(output/"rollouts.jsonl"),training_log_sha256=sha256(output/"training.jsonl"))
    json_write(output/"training_status.json",result)
    print(json.dumps(dict(training=result,finished=True)),flush=True)
    return result


@torch.no_grad()
def evaluate_student(backend,module,db,examples,method,phase,output,max_new_tokens):
    # The historical evaluator fixes4; retain its audited scoring/retrieval
    # logic while honoring this runner's explicit generation budget.
    original = backend.generate
    def generate(question,context=None,max_new_tokens=None):
        return original(question,context=context,max_new_tokens=budget)
    budget = max_new_tokens
    backend.generate = generate
    try:
        return evaluate(backend,module,db,examples,method,phase,output)
    finally:
        backend.generate = original


@torch.no_grad()
def diagnostic_alignment(backend,module,db,examples,contexts,batch_size,phase,output):
    backend.mode,backend.vector_vera,backend.vector_store = "vector_vera",module,db
    backend.vector_override = backend.oracle_values = None
    backend.trace_retrieval = False
    rows = []
    for start in range(0,len(examples),batch_size):
        batch = examples[start:start+batch_size]
        questions = [ex.question for ex in batch]
        tokens = gold_tokens(backend,[ex.answer for ex in batch])
        teacher = backend.forward_sequences(questions,tokens,contexts=contexts[start:start+len(batch)],teacher=True)
        no_memory = backend.forward_sequences(questions,tokens,teacher=True)
        student = backend.forward_sequences(questions,tokens)
        _,_,_,kl,hidden = distill_branches(student,teacher,True)
        _,_,_,base_kl,base_hidden = distill_branches(no_memory,teacher,True)
        for index,ex in enumerate(batch):
            selected = student["batch_indices"]==index
            first = int(torch.where(selected)[0][0])
            rows.append(dict(id=ex.id,phase=phase,continuation_token_ids=tokens[index],tokens=len(tokens[index]),
                             reverse_kl=float(kl[selected].mean()),hidden_cosine=1-float(hidden[selected].mean()),
                             no_memory_reverse_kl=float(base_kl[selected].mean()),
                             no_memory_hidden_cosine=1-float(base_hidden[selected].mean()),
                             first_token_reverse_kl=float(kl[first]),first_token_hidden_cosine=1-float(hidden[first]),
                             first_token_no_memory_reverse_kl=float(base_kl[first]),
                             first_token_no_memory_hidden_cosine=1-float(base_hidden[first])))
    append_jsonl(output/"alignment_predictions.jsonl",rows)
    return dict(count=len(rows),tokens=sum(row["tokens"] for row in rows),
                reverse_kl=sum(row["reverse_kl"] for row in rows)/len(rows),
                hidden_cosine=sum(row["hidden_cosine"] for row in rows)/len(rows),
                no_memory_reverse_kl=sum(row["no_memory_reverse_kl"] for row in rows)/len(rows),
                no_memory_hidden_cosine=sum(row["no_memory_hidden_cosine"] for row in rows)/len(rows),
                first_token_reverse_kl=sum(row["first_token_reverse_kl"] for row in rows)/len(rows),
                first_token_hidden_cosine=sum(row["first_token_hidden_cosine"] for row in rows)/len(rows),
                first_token_no_memory_reverse_kl=sum(row["first_token_no_memory_reverse_kl"] for row in rows)/len(rows),
                first_token_no_memory_hidden_cosine=sum(row["first_token_no_memory_hidden_cosine"] for row in rows)/len(rows),
                aggregation="mean token metric per fact, then mean across facts; gold+EOS prefixes")


@torch.no_grad()
def teacher_acceptance(backend,examples,contexts,phase,output,max_new_tokens):
    rows = []
    for ex,context in zip(examples,contexts):
        with backend._teacher_forward():
            prediction,length,seconds = backend.generate(ex.question,context=context,max_new_tokens=max_new_tokens)
            score = backend.score(ex.question,ex.answer,context=context)
        rows.append(dict(id=ex.id,method="teacher",phase=phase,answer=ex.answer,prediction=prediction,
                         em=int(exact_match(prediction,ex.answer)),nll_sum=score.nll_sum,tokens=score.tokens,
                         generation_tokens=length,generation_seconds=seconds,teacher_context=context,
                         expected_in_bank=False))
    append_jsonl(output/"teacher_predictions.jsonl",rows)
    return stats(rows)


def evaluate_development(backend,module,cfg,packet,output):
    module.requires_grad_(False).eval()
    initial_hash = tensor_digest(module)
    examples = [Example(**row) for row in packet["examples"]["dev"][:cfg["eval_size"]]]
    features = packet["features"]["dev"]["s"][:len(examples)]
    names = packet["template_ids"]["dev"]
    results,writes = {},[]
    clear_bank(backend)
    backend.vector_vera = module
    for heldout_support in (False,True):
        bank_name = "heldout_support" if heldout_support else "canonical_support"
        db = PersistentVectorDB(module.key_dim,module.rank,top_k=module.top_k,temperature=module.temperature)
        db.save(output/(bank_name+"_initial_vdb.pt"))
        contexts = []
        for index,ex in enumerate(examples):
            view = 1+index%2 if heldout_support else 0
            support_text = data.render_support(ex,names["s"][view])
            with torch.no_grad():
                hidden = features[index,view].to(backend.device)
                db.write(ex.id,module.encode_key(hidden).cpu(),module.encode_value(hidden).cpu(),index)
            contexts.append(support_text)
            writes.append(dict(id=ex.id,bank=bank_name,timestamp=index,support_template=names["s"][view],support=support_text))
        db.save(output/(bank_name+"_vdb.pt"))
        bank_hash = db.hash()
        for heldout_query in (False,True):
            query_name = "heldout_query" if heldout_query else "canonical_query"
            phase = bank_name+"/"+query_name
            # Query and observation styles vary independently: every block of
            # four facts covers the four held-out q/s combinations once. Query
            # assignment stays fixed when switching between the two banks.
            rendered = [replace(ex,question=data.render_question(ex,names["q"][1+(index//2)%2 if heldout_query else 0]))
                        for index,ex in enumerate(examples)]
            scores = {method:stats(evaluate_student(backend,module,db,rendered,method,phase,output,cfg["max_new_tokens"]))
                      for method in ("real","oracle","shuffled","empty")}
            alignment = diagnostic_alignment(backend,module,db,rendered,contexts,cfg["batch_size"],phase,output)
            teacher = (None if cfg["skip_teacher_eval"] else
                       teacher_acceptance(backend,rendered,contexts,phase,output,cfg["max_new_tokens"]))
            results[phase] = dict(student=scores,alignment=alignment,teacher=teacher,
                                 records=len(db),vdb_sha256=bank_hash,numeric_vdb_bytes=db.resident_bytes())
            print(json.dumps(dict(evaluation=phase,finished=False,metrics=results[phase])),flush=True)
        if db.hash()!=bank_hash:
            raise AssertionError("Evaluation changed the frozen VDB")
    clear_bank(backend)
    if tensor_digest(module)!=initial_hash:
        raise AssertionError("Evaluation changed shared memory-module weights")
    result = dict(conditions=results,eval_facts=len(examples),evaluation_split="dev",
                  online_gradient_steps=0,shared_parameters_unchanged=True,vdb_unchanged=True,
                  model_id="Qwen/Qwen3-4B-Instruct-2507",module_sha256=initial_hash,
                  interpretation="Development diagnostics; no confirmation generalization claim or checkpoint selection")
    json_write(output/"writes.json",writes)
    json_write(output/"metrics.json",result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method",choices=METHODS,default="ce")
    parser.add_argument("--model",required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--updates",type=int,default=256)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--eval-size",type=int,default=64)
    parser.add_argument("--max-new-tokens",type=int,default=4)
    parser.add_argument("--sampling-temperature",type=float,default=1.)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--evaluate-only",action="store_true")
    parser.add_argument("--skip-teacher-eval",action="store_true")
    args = parser.parse_args(argv)
    if args.updates<1 or not 1<=args.batch_size<=4096 or not 1<=args.eval_size<=64 or args.max_new_tokens<1:
        parser.error("Invalid training/evaluation budget")
    if not math.isfinite(args.sampling_temperature) or args.sampling_temperature<=0:
        parser.error("Student sampling temperature must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(4)
    cfg = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    cfg.update(kl_direction="reverse",kl_temperature=1.,hidden_weight=.1,
               canonical_replay_frequency=4,canonical_replay_weight=.25,
               all_view_address_weight=.2,actual_prefix_address_weight=.2,
               wrong_prefix_address_policy="Every sampled prediction position targets the question's fact, even after a wrong token")
    args.run_dir.mkdir(parents=True,exist_ok=False)
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(),complete=False,
                    method=args.method,evaluate_only=args.evaluate_only,
                    source_files_sha256={path.name:sha256(path) for path in Path(__file__).parent.glob("*.py")})
    json_write(args.run_dir/"config.json",cfg)
    json_write(args.run_dir/"manifest.json",manifest)
    try:
        model_manifest = json.loads((Path(args.model).parent/"manifest.json").read_text())
        if model_manifest["revision"]!=REVISION:
            raise ValueError("Pinned Qwen model revision required")
        packet = select_offline_packet(torch.load(args.cache,map_location="cpu",weights_only=True))
        validate_packet(packet)
        checkpoint = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
        if checkpoint.get("method")!="anchored_all_view":
            raise ValueError("All conditions require the same anchored_all_view initialization")
        manifest.update(cache_sha256=sha256(args.cache),checkpoint_sha256=sha256(args.checkpoint),
                        model_revision=REVISION,model_manifest_sha256=sha256(Path(args.model).parent/"manifest.json"),
                        data_fingerprint=packet["data_fingerprint"],cache_splits_used=["train","dev"],
                        confirmation_used=False,dev_selection=False,initialization_step=checkpoint.get("step"))
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(4)
        backend = ContextDistillationBackend(args.model,layer=20)
        if any(parameter.requires_grad for parameter in backend.model.parameters()):
            raise AssertionError("The shared teacher/backbone must be frozen")
        module = StableVectorVeRA(backend.target.in_features,backend.target.out_features,
                                  rank=64,key_dim=64,top_k=4,temperature=.05,seed=args.seed).to(backend.device)
        module.load_state_dict(checkpoint["module"])
        backend.vector_vera = module
        module.requires_grad_(False).eval()
        teacher_hash = parameter_digest(backend.model)
        manifest.update(teacher_parameter_sha256_before=teacher_hash,module_sha256_before=tensor_digest(module),
                        gpu=torch.cuda.get_device_name(0),torch_version=str(torch.__version__))
        json_write(args.run_dir/"manifest.json",manifest)
        for split in ("train","dev"):
            save_examples(args.run_dir/(split+".jsonl"),[Example(**row) for row in packet["examples"][split]])
        training = None if args.evaluate_only else train(backend,module,cfg,packet,args.run_dir)
        result = evaluate_development(backend,module,cfg,packet,args.run_dir)
        after_teacher_hash = parameter_digest(backend.model)
        if after_teacher_hash!=teacher_hash or any(p.grad is not None for p in backend.model.parameters()):
            raise AssertionError("Frozen backbone parameters changed or received gradients")
        manifest.update(complete=True,finished=True,selected_step=0 if args.evaluate_only else args.updates,
                        training=training,teacher_parameter_sha256_after=after_teacher_hash,
                        teacher_parameters_unchanged=True,module_sha256_after=tensor_digest(module),
                        completed_at=datetime.now(timezone.utc).isoformat())
        json_write(args.run_dir/"manifest.json",manifest)
        print(json.dumps(dict(RUN_COMPLETE=result,finished=True)),flush=True)
    except BaseException as error:
        manifest.update(complete=False,finished=False,error_type=type(error).__name__,error=str(error),
                        failed_at=datetime.now(timezone.utc).isoformat())
        json_write(args.run_dir/"manifest.json",manifest)
        raise


if __name__=="__main__":
    main()
