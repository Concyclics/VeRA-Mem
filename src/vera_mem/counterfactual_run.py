"""Fixed-budget paired context interventions through the sparse VeRA writer.

Train and confirmation evaluation are separate commands and separate caches.
Every row has its own bank: worlds B/P replace exactly that row's target entry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from . import augmentation_data as expressions
from .context_distillation import distillation_loss
from .context_distillation_run import (
    REVISION, EpisodeSampler, append_jsonl, clear_bank, gold_tokens,
    parameter_digest, sequence_mean, sha256, supervised_loss,
)
from .counterfactual_backend import BatchedStableVectorVeRA, CounterfactualBackend
from .counterfactual_losses import hidden_delta_loss, pair_behavior_loss, paraphrase_invariance_loss
from .data import Example
from .factcentric_losses import style_block_retrieval_loss
from .run import json_write, tensor_digest

PROTOCOL = "counterfactual-context-v2"
METHODS = ("base", "behavior", "hidden", "mixed")


class CounterfactualSampler(EpisodeSampler):
    """Ensure same-answer wrong-entity negatives for both value worlds."""
    def __init__(self, pairs, batch_size, seed):
        super().__init__(len(pairs), batch_size, seed)
        self.pairs = pairs
        self.hard_rng = random.Random(seed + 130)
        self.by_answer = {}
        for index, pair in enumerate(pairs):
            self.by_answer.setdefault(pair["original"]["answer"], []).append(index)

    def next(self, query_views=8, support_views=4):
        sample = super().next(query_views, support_views)
        targets = sample["targets"]
        selected = set(targets)
        required = []
        for target in targets:
            for world in ("original", "alternative"):
                candidates = [i for i in self.by_answer[self.pairs[target][world]["answer"]]
                              if i != target]
                self.hard_rng.shuffle(candidates)
                for index in candidates[:2]:
                    if index not in selected:
                        required.append(index)
                        selected.add(index)
        size = len(sample["episode"])
        episode = targets + required
        for index in sample["episode"]:
            if index not in selected:
                episode.append(index)
                selected.add(index)
        episode = episode[:size]
        self.hard_rng.shuffle(episode)
        sample.update(episode=episode, mapping=[episode.index(i) for i in targets])
        return sample


def first_positions(branch):
    return torch.cat((branch["counts"].new_zeros(1), branch["counts"].cumsum(0)[:-1]))


def independent_banks(base, alternatives, paraphrases, columns):
    """A is shared content; B/P replace only the target for each batch row."""
    columns = torch.as_tensor(columns, device=base.device, dtype=torch.long)
    if base.ndim != 2 or alternatives.shape != paraphrases.shape or alternatives.shape != (len(columns), base.shape[1]):
        raise ValueError("Invalid base or per-target feature shapes")
    if len(columns) == 0 or bool(((columns < 0) | (columns >= len(base))).any()):
        raise ValueError("Invalid target bank columns")
    a = base.unsqueeze(0).expand(len(columns), -1, -1)
    b, p = a.clone(), a.clone()
    rows = torch.arange(len(columns), device=base.device)
    b[rows, columns] = alternatives
    p[rows, columns] = paraphrases
    return a, b, p


def activate(backend, module, support, detach=False):
    clear_bank(backend)
    backend.mode, backend.vector_vera = "vector_vera", module
    if detach:
        with torch.no_grad():
            backend.vector_keys = module.encode_key(support).detach()
            backend.vector_values = module.encode_value(support).detach()
    else:
        backend.vector_keys = module.encode_key(support)
        backend.vector_values = module.encode_value(support)
        if not backend.vector_keys.requires_grad or not backend.vector_values.requires_grad:
            raise AssertionError("Training bank lost its differentiable writer graph")


def address_loss(backend, module, branch, columns):
    queries = module.encode_query(branch["query_inputs"])
    banks = backend.vector_keys[branch["batch_indices"]]
    scores = torch.einsum("td,tnd->tn", queries, banks) / .1
    labels = torch.as_tensor(columns, device=scores.device)[branch["batch_indices"]]
    return sequence_mean(F.cross_entropy(scores, labels, reduction="none"),
                         branch["batch_indices"], branch["counts"])


def paired_step(backend, module, batch, method, on_policy, generator):
    """All worlds use the same weights until one joint optimizer step."""
    questions, contexts, answers = batch["questions"], batch["contexts"], batch["answers"]
    banks = batch["banks"]
    golden = [gold_tokens(backend, world) for world in answers]
    a_ids = torch.tensor([v[0] for v in golden[0]], device=backend.device)
    b_ids = torch.tensor([v[0] for v in golden[1]], device=backend.device)
    if bool((a_ids == b_ids).any()):
        raise ValueError("First-token counterfactual supervision requires distinct answer tokens")
    sampling = dict(forward_calls=0, unpadded_input_tokens=0, padded_input_tokens=0)
    selected_world = []
    if on_policy:
        selected_world = torch.randint(2, (len(questions),), generator=generator, device=backend.device).tolist()
        chosen = torch.stack([banks[world][row] for row, world in enumerate(selected_world)])
        activate(backend, module, chosen, detach=True)
        common = backend.sample_student(questions, max_new_tokens=4, temperature=1., generator=generator)
        sampling = dict(backend.last_sampling_cost)
        tokens = [common, common, common]
    else:
        tokens = golden
    # Teacher cannot consume any student bank, and is evaluated without autograd.
    teachers = [backend.forward_sequences(questions, t, contexts=c, teacher=True)
                for t, c in zip(tokens, contexts)]
    students, fkl, ce_first, address = [], [], [], []
    for support, target, teacher, correct in zip(banks, tokens, teachers, golden):
        activate(backend, module, support)
        branch = backend.forward_sequences(questions, target)
        students.append(branch)
        if not torch.equal(branch["counts"], teacher["counts"]):
            raise AssertionError("Teacher/student continuation mismatch")
        kl = distillation_loss(branch["logits"], teacher["logits"], direction="forward", reduction="none")
        fkl.append(sequence_mean(kl, branch["batch_indices"], branch["counts"]))
        labels = torch.tensor([v[0] for v in correct], device=backend.device)
        ce_first.append(F.cross_entropy(branch["logits"][first_positions(branch)], labels))
        address.append(address_loss(backend, module, branch, batch["columns"]))
    sl = [s["logits"][first_positions(s)] for s in students]
    sh = [s["hidden"][first_positions(s)] for s in students]
    tl = [t["logits"][first_positions(t)] for t in teachers]
    th = [t["hidden"][first_positions(t)] for t in teachers]
    behavior = pair_behavior_loss(sl[0], sl[1], tl[0], tl[1], a_ids, b_ids,
                                  target_clip=10., gold_weight=0.)
    hidden = hidden_delta_loss(sh[0], sh[1], th[0], th[1], min_teacher_norm=1e-3)
    invariance = paraphrase_invariance_loss(sl[0], sl[2], temperature=1.)
    # Both value worlds enter the common full-style identity objective. This is
    # auxiliary addressing, not the actual independently intervened read banks.
    all_queries = module.encode_query(batch["q_views"])
    style_losses = [style_block_retrieval_loss(all_queries, module.encode_key(view), temperature=.1)["retrieval"]
                    for view in batch["s_views"]]
    main = torch.stack(fkl).mean() + .5 * torch.stack(ce_first).mean()
    total = main + .2 * torch.stack(address).mean() + .2 * torch.stack(style_losses).mean() + .05 * invariance["loss"]
    if method in ("behavior", "hidden", "mixed"):
        total = total + .1 * behavior["loss"]
    if method in ("hidden", "mixed"):
        total = total + .1 * hidden["loss"]
    replay_loss = main.new_zeros(())
    replay_input = replay_target = 0
    if batch["replay_questions"] is not None:
        activate(backend, module, batch["replay_support"])
        replay = backend.forward_sequences(batch["replay_questions"], golden[0])
        replay_loss, _ = supervised_loss(replay, golden[0])
        total = total + .25 * replay_loss
        replay_target = sum(map(len, golden[0]))
        replay_input = sum(len(backend.prompt_ids(q)) + len(t) for q, t in zip(batch["replay_questions"], golden[0]))
    metrics = dict(main_loss=float(main.detach()), total_loss=float(total.detach()),
                   forward_kl=float(torch.stack(fkl).mean().detach()),
                   first_token_ce=float(torch.stack(ce_first).mean().detach()),
                   actual_address_loss=float(torch.stack(address).mean().detach()),
                   style_address_loss=float(torch.stack(style_losses).mean().detach()),
                   behavior_loss=float(behavior["loss"].detach()), hidden_loss=float(hidden["loss"].detach()),
                   paraphrase_loss=float(invariance["loss"].detach()), replay_loss=float(replay_loss.detach()),
                   behavior_valid_pairs=int(behavior["valid_mask"].sum()),
                   behavior_clipped_pairs=int(behavior["clipped_mask"].sum()),
                   teacher_behavior_delta_mean=float(behavior["teacher_delta_raw"].mean()),
                   student_behavior_delta_mean=float(behavior["student_delta"].mean()),
                   teacher_first_token_joint_correct=int(((tl[0].argmax(-1) == a_ids) & (tl[1].argmax(-1) == b_ids)).sum()),
                   student_first_token_joint_correct=int(((sl[0].argmax(-1) == a_ids) & (sl[1].argmax(-1) == b_ids)).sum()),
                   hidden_valid_pairs=int(hidden["valid_mask"].sum()),
                   teacher_delta_norm_mean=float(hidden["teacher_delta_norm"].mean()),
                   student_delta_norm_mean=float(hidden["student_delta_norm"].detach().mean()),
                   student_input_tokens=sum(len(backend.prompt_ids(q)) + len(t) for ts in tokens for q, t in zip(questions, ts)),
                   teacher_input_tokens=sum(len(backend.prompt_ids(q, c)) + len(t) for cs, ts in zip(contexts, tokens) for q, c, t in zip(questions, cs, ts)),
                   target_tokens=sum(len(t) for ts in tokens for t in ts),
                   sampled_tokens=sum(map(len, tokens[0])) if on_policy else 0,
                   rollout_input_tokens=sampling["unpadded_input_tokens"],
                   rollout_forward_calls=sampling["forward_calls"],
                   replay_input_tokens=replay_input, replay_target_tokens=replay_target)
    trajectories = [dict(question=q, contexts=[c[i] for c in contexts],
                         continuation_token_ids=[t[i] for t in tokens], gold_token_ids=[g[i] for g in golden],
                         sampled_world=selected_world[i] if on_policy else None)
                    for i, q in enumerate(questions)]
    return total, metrics, trajectories


def make_batch(packet, sample, step, device):
    rows = packet["pairs"]
    episode, targets, columns = [sample[k] for k in ("episode", "targets", "mapping")]
    qviews, sviews = sample["qviews"], sample["sviews"]
    q = packet["q"][episode].to(device)
    s = packet["s"][episode].to(device)  # [episode, A/B, style, feature]
    positions = torch.arange(len(episode), device=device)
    chosen = s[positions, 0, sviews]
    alternative = s[columns, 1, [sviews[c] for c in columns]]
    pviews = [(sviews[c] + 1) % 4 for c in columns]
    paraphrase = s[columns, 0, pviews]
    banks = independent_banks(chosen, alternative, paraphrase, columns)
    a = [Example(**rows[i]["original"]) for i in targets]
    b = [Example(**rows[i]["alternative"]) for i in targets]
    qids = packet["template_ids"]["q"]
    sids = packet["template_ids"]["s"]
    questions = [expressions.render_question(ex, qids[qviews[c]]) for ex, c in zip(a, columns)]
    contexts = [
        [expressions.render_support(ex, sids[sviews[c]]) for ex, c in zip(a, columns)],
        [expressions.render_support(ex, sids[sviews[c]]) for ex, c in zip(b, columns)],
        [expressions.render_support(ex, sids[v]) for ex, v in zip(a, pviews)],
    ]
    all_b = s[:, 0].clone()
    all_b[columns] = s[columns, 1]
    return dict(questions=questions, contexts=contexts,
                answers=[[x.answer for x in a], [x.answer for x in b], [x.answer for x in a]],
                banks=banks, columns=columns, q_views=q, s_views=[s[:, 0], all_b],
                replay_support=s[:, 0, 0],
                replay_questions=[expressions.render_question(ex, qids[0]) for ex in a] if step % 4 == 0 else None,
                target_ids=[x.id for x in a], episode_ids=[rows[i]["original"]["id"] for i in episode])


def train(backend, module, packet, checkpoint, cfg, output):
    if packet["protocol"] != PROTOCOL or packet["split"] != "train" or packet["model_revision"] != REVISION:
        raise ValueError("Only pinned train-only counterfactual cache may enter training")
    if packet["q"].shape != (len(packet["pairs"]), 8, 9728) or packet["s"].shape != (len(packet["pairs"]), 2, 4, 9728):
        raise ValueError("Unexpected training feature shapes")
    module.requires_grad_(True).train()
    groups = [list(module.Wv.parameters()), [module.b], list(module.Wq.parameters()) + list(module.Wk.parameters())]
    optimizer = torch.optim.Adam([dict(params=groups[0], lr=1e-4), dict(params=groups[1], lr=.005), dict(params=groups[2], lr=1e-5)])
    if cfg["start_step"]:
        if checkpoint.get("protocol") != PROTOCOL or checkpoint.get("step") != cfg["start_step"]:
            raise ValueError("Continuation must start from the fixed common warm checkpoint")
        optimizer.load_state_dict(checkpoint["optimizer"])
    sampler = CounterfactualSampler(packet["pairs"], cfg["batch_size"], cfg["seed"])
    for _ in range(cfg["start_step"]):
        sampler.next()
    generator = torch.Generator(device=backend.device).manual_seed(cfg["seed"] + 500)
    schedule = hashlib.sha256()
    totals = {}
    started = time.perf_counter()
    for local_step in range(1, cfg["updates"] + 1):
        step = cfg["start_step"] + local_step
        sample = sampler.next()
        batch = make_batch(packet, sample, step, backend.device)
        on = cfg["method"] == "mixed" and cfg["start_step"] > 0 and local_step % 4 == 0
        optimizer.zero_grad(set_to_none=True)
        loss, metrics, trajectories = paired_step(backend, module, batch, cfg["method"], on, generator)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite paired distillation loss")
        loss.backward()
        norms = [float(torch.nn.utils.clip_grad_norm_(g, 1.)) for g in groups]
        if not all(torch.isfinite(torch.tensor(norms))):
            raise FloatingPointError("Nonfinite paired distillation gradient")
        optimizer.step()
        clear_bank(backend)
        exposure = dict(target_ids=batch["target_ids"], episode_ids=batch["episode_ids"],
                        query_views=sample["qviews"], support_views=sample["sviews"], columns=sample["mapping"])
        schedule.update(json.dumps(exposure, sort_keys=True, separators=(",", ":")).encode())
        for name, value in metrics.items():
            if name.endswith("tokens") or name == "rollout_forward_calls":
                totals[name] = totals.get(name, 0) + value
        row = dict(step=step, local_step=local_step, on_policy=on, **exposure, **metrics,
                   gradient_norms=norms, elapsed_seconds=time.perf_counter()-started)
        append_jsonl(output/"training.jsonl", [row])
        append_jsonl(output/"trajectories.jsonl", [dict(step=step, id=identity, on_policy=on, **v)
                                                  for identity, v in zip(batch["target_ids"], trajectories)])
        if local_step == 1 or local_step % 16 == 0:
            print(json.dumps({k: row[k] for k in ("step", "local_step", "on_policy", "main_loss", "behavior_loss", "hidden_loss", "elapsed_seconds")}), flush=True)
        if local_step % 128 == 0 or local_step == cfg["updates"]:
            temporary = output/"last.pt.partial"
            torch.save(dict(protocol=PROTOCOL, module=module.state_dict(), optimizer=optimizer.state_dict(),
                            step=step, configuration=cfg), temporary)
            temporary.replace(output/"last.pt")
            json_write(output/"training_status.json", dict(complete=local_step == cfg["updates"], step=step,
                       updates=local_step, target_pair_exposures=local_step*cfg["batch_size"],
                       episode_schedule_sha256=schedule.hexdigest(), token_totals=totals,
                       elapsed_seconds=time.perf_counter()-started))
        del loss, batch, trajectories
    module.requires_grad_(False).eval()
    return json.loads((output/"training_status.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="base")
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--start-step", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--teacher", action="store_true")
    args = parser.parse_args()
    if min(args.updates, args.batch_size) < 1 or args.start_step < 0:
        parser.error("Invalid training budget")
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.run_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(protocol=PROTOCOL, complete=False, started_at=datetime.now(timezone.utc).isoformat(),
                    configuration=cfg, source_files_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")})
    json_write(args.run_dir/"manifest.json", manifest)
    try:
        if json.loads((Path(args.model).parent/"manifest.json").read_text())["revision"] != REVISION:
            raise ValueError("Pinned model revision required")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(4)
        packet = torch.load(args.cache, map_location="cpu", weights_only=True)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        backend = CounterfactualBackend(args.model, layer=20)
        module = BatchedStableVectorVeRA(9728, 2560, rank=64, key_dim=64, top_k=4, temperature=.05, seed=args.seed).to(backend.device)
        module.load_state_dict(checkpoint["module"])
        before_teacher = parameter_digest(backend.model)
        manifest.update(cache_sha256=sha256(args.cache), checkpoint_sha256=sha256(args.checkpoint),
                        model_revision=REVISION, teacher_parameter_sha256_before=before_teacher,
                        module_sha256_before=tensor_digest(module), cache_split=packet["split"])
        json_write(args.run_dir/"manifest.json", manifest)
        if args.evaluate_only:
            if packet["protocol"] != PROTOCOL or packet["split"] != "confirmation" or packet["model_revision"] != REVISION:
                raise ValueError("Evaluation requires separate pinned confirmation cache")
            from .counterfactual_eval import evaluate_counterfactual
            module.requires_grad_(False).eval()
            result = evaluate_counterfactual(backend, module, packet, args.run_dir, include_teacher=args.teacher, max_new_tokens=4)
            manifest["evaluation"] = result
        else:
            manifest["training"] = train(backend, module, packet, checkpoint, cfg, args.run_dir)
        after_teacher = parameter_digest(backend.model)
        if after_teacher != before_teacher or any(p.grad is not None for p in backend.model.parameters()):
            raise AssertionError("Frozen backbone changed or received gradients")
        manifest.update(complete=True, teacher_parameters_unchanged=True, teacher_parameter_sha256_after=after_teacher,
                        module_sha256_after=tensor_digest(module), finished_at=datetime.now(timezone.utc).isoformat())
        json_write(args.run_dir/"manifest.json", manifest)
        print(json.dumps(dict(complete=True, method=args.method, evaluate_only=args.evaluate_only)), flush=True)
    except BaseException as error:
        manifest.update(complete=False, error=repr(error), failed_at=datetime.now(timezone.utc).isoformat())
        json_write(args.run_dir/"manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
