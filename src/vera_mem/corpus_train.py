"""One complete, resumable WikiText pass for three VeRA readers and matched LoRA.

No result-based early stop: the 10M-token checkpoint is diagnostic only.
Frozen-prefix reuse saves compute while every arm receives the same windows
and episode order. Causal memory excludes every block from the query article.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .corpus_backend import CorpusBackbone, CorpusLoRA, adapter_delta, answer_batch, coarse_teacher_loss
from .corpus_data import Corpus, atomic_json, sha256
from .corpus_index import CandidateIndex
from .corpus_reader import CorpusReader, FeatureBank, MODES, sparse_cross_entropy


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def vocabulary_losses(backbone, hidden, labels, auxiliary, teacher=None):
    """Chunk the vocabulary projection, preserving a single exact loss mean."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    target = labels.flatten()
    positions = (target != -100).nonzero().flatten()
    if not len(positions):
        raise ValueError("An update has no supervised tokens")
    ce, sparse, kd = 0., 0., 0.
    teacher_ids, teacher_probs = teacher if teacher is not None else (None, None)
    for start in range(0, len(positions), 128):
        loc = positions[start:start + 128]
        logits = backbone.model.lm_head(flat[loc]).float()
        ce = ce + F.cross_entropy(logits, target[loc], reduction="sum")
        if "token_ids" in auxiliary:
            sparse = sparse + sparse_cross_entropy(logits, auxiliary["token_ids"][loc],
                                                   auxiliary["probabilities"][loc]).sum()
        if teacher is not None:
            kd = kd + coarse_teacher_loss(logits, teacher_ids[start:start + len(loc)],
                                           teacher_probs[start:start + len(loc)]).sum()
    total = len(positions)
    address = auxiliary["address"][positions].mean() if "address" in auxiliary else hidden.new_tensor(0.)
    route = auxiliary["route"][positions].mean() if "route" in auxiliary else hidden.new_tensor(0.)
    return {"ce": ce / total, "sparse_ce": sparse / total if torch.is_tensor(sparse) else 0.,
            "context_ce": kd / total if torch.is_tensor(kd) else 0.,
            "address": address, "route": route, "tokens": total}


@torch.no_grad()
def contextual_targets(backbone, episodes, style):
    ids, labels, mask = answer_batch(backbone.tokenizer, episodes, style=style, context=True)
    prefix = backbone.prefix(ids, mask)
    hidden = backbone.suffix(prefix)
    chosen = hidden[labels != -100]
    top_ids, probabilities = [], []
    correct, total = 0, len(chosen)
    gold = labels[labels != -100]
    for start in range(0, len(chosen), 128):
        logits = backbone.model.lm_head(chosen[start:start + 128]).float()
        logp = F.log_softmax(logits, -1)
        scores, indices = logp.topk(32, -1)
        top_ids.append(indices)
        probabilities.append(scores.exp())
        correct += int((logits.argmax(-1) == gold[start:start + 128]).sum())
    return (torch.cat(top_ids), torch.cat(probabilities)), {"teacher_forced_token_correct": correct,
                                                         "teacher_forced_tokens": total,
                                                         "teacher_input_positions": int(mask.sum())}


def source_manifest():
    root = Path(__file__).resolve().parent
    names = ("corpus_data.py", "corpus_features.py", "corpus_index.py", "corpus_reader.py",
             "corpus_backend.py", "corpus_episodes.py", "corpus_train.py")
    return {name: sha256(root / name) for name in names}


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(args.data, "train")
    table = corpus.window_table(1024, 1024)
    order = np.random.default_rng(args.seed).permutation(len(table))
    episodes = [json.loads(line) for line in (Path(args.episodes) / "train.jsonl").open()]
    ep_order = np.random.default_rng(args.seed + 1).permutation(len(episodes))
    config = {"schema": "vera-corpus-training-v1", "seed": args.seed, "batch_size": args.batch_size,
              "sequence_length": 1024, "full_pass_target_tokens": corpus.manifest["splits"]["train"]["lm_target_tokens"],
              "window_count": len(table), "episode_interval": 8, "episode_batch": args.batch_size * 8,
              "learning_rate": 1e-4, "warmup_lm_tokens": 2_000_000, "minimum_lr_ratio": .1,
              "lm_loss": "CE + 0.1 sparse-retrieval CE + 0.1 address CE + 0.1 route CE",
              "episode_loss": "CE + 0.5 contextual top32-and-other CE + 0.1 address CE + 0.1 route CE",
              "episode_candidate_rule": "Force observed positive for 3/4 of training episode updates; other 1/4 use natural ANN",
              "lora_rule": "Parameter-matched layer-20 LoRA; same CE/contextual teacher data; no VDB address losses",
              "memory_exclusion": "All training blocks from query article for causal LM",
              "stop_rule": "One complete training pass; 10M-token checkpoint is diagnostic, not an early stop",
              "data_manifest_sha256": sha256(Path(args.data) / "manifest.json"),
              "episode_manifest_sha256": sha256(Path(args.episodes) / "manifest.json"),
              "feature_manifest_sha256": sha256(Path(args.features) / "progress.json"),
              "index_manifest_sha256": sha256(Path(args.index) / "complete.json"),
              "source_sha256": source_manifest(), "dry_run_updates": args.max_updates,
              "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    registration = output / "registration.json"
    if registration.exists():
        previous = json.loads(registration.read_text())
        comparable = {k: v for k, v in config.items() if k != "created_at_utc"}
        if comparable != {k: v for k, v in previous.items() if k != "created_at_utc"}:
            raise ValueError("Resume configuration or source differs from registration")
        config = previous
    else:
        atomic_json(registration, config)
    backbone = CorpusBackbone(args.model)
    qualification = backbone.qualify_prefix()
    bank = FeatureBank(args.features)
    index = CandidateIndex.load(args.index, threads=8)
    modules = {mode: CorpusReader(mode=mode, seed=args.seed).to("cuda") for mode in MODES}
    for reader in modules.values():
        reader.set_statistics(bank)
    target_parameters = sum(p.numel() for p in modules["block_outer"].parameters() if p.requires_grad)
    lora_rank = round(target_parameters / (9728 + 2560))
    modules["lora"] = CorpusLoRA(rank=lora_rank, seed=args.seed).to("cuda")
    optimizers = {name: torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
                                         lr=1e-4, weight_decay=.01) for name, module in modules.items()}
    atomic_json(output / "qualification.json", dict(qualification, trainable_parameters={
        name: sum(p.numel() for p in module.parameters() if p.requires_grad) for name, module in modules.items()},
        index_blocks=index.index.ntotal, torch_version=torch.__version__))
    state = {"cursor": 0, "lm_tokens": 0, "episode_cursor": 0, "episode_updates": 0,
             "episode_tokens": 0, "updates": 0, "teacher_input_positions": 0,
             "teacher_forced_tokens": 0, "teacher_forced_token_correct": 0,
             "checkpoint_bin": 0, "complete": False}
    latest = output / "latest.json"
    if latest.exists():
        reference = json.loads(latest.read_text())
        checkpoint = output / reference["checkpoint"]
        if sha256(checkpoint) != reference["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
        state = saved["state"]
        for name, module in modules.items():
            module.load_state_dict(saved["modules"][name])
            optimizers[name].load_state_dict(saved["optimizers"][name])
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([t.cpu() for t in saved["cuda_rng"]])
        if state["complete"]:
            print("Training already complete", flush=True)
            return
    started = time.monotonic()
    start_tokens = state["lm_tokens"]
    metrics_path = output / "metrics.jsonl"

    def checkpoint(label):
        relative = f"checkpoints/{label}.pt"
        path = output / relative
        payload = {"state": dict(state), "modules": {n: m.state_dict() for n, m in modules.items()},
                   "configurations": {n: m.configuration for n, m in modules.items()},
                   "optimizers": {n: o.state_dict() for n, o in optimizers.items()},
                   "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                   "registration_sha256": sha256(registration)}
        save_checkpoint(path, payload)
        atomic_json(latest, {"checkpoint": relative, "sha256": sha256(path), **state})

    def learn(prefix, labels, *, doc_ids=None, positive=None, force=False, teacher=None):
        progress = state["lm_tokens"] / config["full_pass_target_tokens"]
        cosine = .1 + .9 * (1 + math.cos(math.pi * progress)) / 2
        warm = min(1., max(.01, state["lm_tokens"] / 2_000_000))
        lr = 1e-4 * cosine * warm
        reports = {}
        for name, module in modules.items():
            optimizer = optimizers[name]
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            t0 = time.monotonic()
            delta, aux = adapter_delta(module, prefix, bank, index, articles=doc_ids, positive=positive,
                                        force_positive=force, supervision=name != "lora")
            hidden = backbone.suffix(prefix, delta)
            losses = vocabulary_losses(backbone, hidden, labels, aux, teacher)
            loss = losses["ce"] + .1 * (losses["address"] + losses["route"])
            loss = loss + (.5 * losses["context_ce"] if teacher is not None else .1 * losses["sparse_ce"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite {name} objective at {state}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(module.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            reports[name] = {key: float(value.detach()) if torch.is_tensor(value) else value
                             for key, value in losses.items()}
            reports[name].update(objective=float(loss.detach()), gradient_norm=float(grad),
                                  seconds=time.monotonic() - t0, index_seconds=aux.get("index_seconds", 0))
            del loss, hidden, delta, aux, losses
        if any(p.grad is not None for p in backbone.model.parameters()):
            raise RuntimeError("Frozen backbone acquired gradients")
        return reports, lr

    for begin in range(state["cursor"], len(order), args.batch_size):
        end = min(len(order), begin + args.batch_size)
        records = table[order[begin:end]]
        ids, labels, mask = [torch.from_numpy(t).to("cuda") for t in corpus.batch(records, corpus.manifest["pad_token_id"])]
        prefix = backbone.prefix(ids, mask)
        reports, lr = learn(prefix, labels, doc_ids=corpus.article_groups[records[:, 0]])
        state.update(cursor=end, updates=state["updates"] + 1,
                     lm_tokens=state["lm_tokens"] + int((labels != -100).sum()))
        del prefix, ids, labels, mask
        if state["updates"] % 8 == 0:
            offset = state["episode_cursor"]
            batch = [episodes[ep_order[(offset + i) % len(ep_order)]] for i in range(config["episode_batch"])]
            style = state["episode_updates"] % 2
            teacher, teacher_stats = contextual_targets(backbone, batch, style)
            ids, labels, mask = answer_batch(backbone.tokenizer, batch, style=style)
            prefix = backbone.prefix(ids, mask)
            ep_reports, _ = learn(prefix, labels, positive=[e["chunk"] for e in batch],
                                   force=state["episode_updates"] % 4 != 3, teacher=teacher)
            state["episode_cursor"] += config["episode_batch"]
            state["episode_updates"] += 1
            state["episode_tokens"] += int((labels != -100).sum())
            for key, value in teacher_stats.items():
                state[key] += value
            reports["episodes"] = ep_reports
            del prefix, ids, labels, mask, teacher
        if state["updates"] % 10 == 0 or end == len(order) or args.max_updates:
            record = {"state": dict(state), "learning_rate": lr, "losses": reports,
                      "session_seconds": time.monotonic() - started,
                      "lm_tokens_per_second_all_arms": (state["lm_tokens"] - start_tokens) / (time.monotonic() - started),
                      "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            atomic_json(output / "progress.json", record)
            print(json.dumps(record), flush=True)
        new_bin = state["lm_tokens"] // 5_000_000
        if new_bin > state["checkpoint_bin"]:
            state["checkpoint_bin"] = new_bin
            checkpoint(f"tokens_{state['lm_tokens']:012d}")
        if args.max_updates and state["updates"] >= args.max_updates:
            checkpoint("engineering_pilot")
            atomic_json(output / "pilot_complete.json", state)
            return
        if args.pause_after_tokens and start_tokens < args.pause_after_tokens <= state["lm_tokens"]:
            checkpoint("development_10m")
            atomic_json(output / "development_pause.json", state)
            return
    if state["lm_tokens"] != config["full_pass_target_tokens"]:
        raise RuntimeError("Full-pass exposure count does not match the prepared corpus")
    state["complete"] = True
    checkpoint("final")
    atomic_json(output / "complete.json", dict(state, finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "episodes", "features", "index", "model", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-updates", type=int, default=0,
                        help="Engineering pilot only; nonzero never constitutes a full run")
    parser.add_argument("--pause-after-tokens", type=int, default=0,
                        help="Operational pause for development evaluation; resume completes the same pass")
    train(parser.parse_args())
