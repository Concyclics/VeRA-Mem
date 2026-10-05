"""Controlled offline-data scaling and causal online vector-memory experiments.

No text is retrieved into prompts. Actual layer inputs query sparse vector
values, which scale the fixed random VeRA branch. All online writes are derived
from an explicitly revealed observation; shared weights stay frozen online.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import Example, MEMORY_WORDS, synthetic_dataset
from .metrics import exact_match, save_examples
from .run import json_write, tensor_digest, digest_file
from .scaling_backend import ScalingQwenBackend
from .stable_vector_vera import StableVectorVeRA
from .vector_vera import VectorVeRA
from .vector_store import PersistentVectorDB
from .vector_run import support_input, shuffled_store


def datasets():
    # Generate one maximum-size pool, then take nested balanced prefixes. Merely
    # calling synthetic_dataset with different N changes the RNG/entity list.
    pool = synthetic_dataset(1042, 4096, 0)["stream"]
    groups = {w: [e for e in pool if e.answer == w] for w in MEMORY_WORDS}
    train = [groups[w][i] for i in range(256) for w in MEMORY_WORDS]
    evaluation = synthetic_dataset(7042, 128, 64)
    data = dict(train=train, dev=synthetic_dataset(2042, 64, 0)["stream"],
                stream=evaluation["stream"], control=evaluation["control"])
    identifiers = [e.id for rows in data.values() for e in rows]
    assert len(identifiers) == len(set(identifiers)), "Split entity overlap"
    return data


def prepare_features(backend, cache: Path):
    if cache.exists():
        packet = torch.load(cache, weights_only=True, map_location="cpu")
        if packet["protocol"] != "scaling-v1-final-prompt-layer20" or packet["model_revision"] != "cdbee75f17c01a7cc42f958dc650907174af0554":
            raise ValueError("Unexpected feature cache provenance")
        return packet
    data = datasets()
    packet = {"protocol": "scaling-v1-final-prompt-layer20",
              "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
              "examples": {name: [asdict(e) for e in rows] for name, rows in data.items()},
              "features": {}}
    for name, rows in data.items():
        print("FEATURES " + name, flush=True)
        packet["features"][name] = {
            "q": backend.layer_features([e.question for e in rows], batch_size=32),
            "s": backend.layer_features([support_input(e) for e in rows], batch_size=32),
        }
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(".partial")
    torch.save(packet, partial)
    partial.replace(cache)
    return packet


def make_module(backend, cfg):
    cls = StableVectorVeRA if cfg["variant"] == "stable" else VectorVeRA
    extra = {} if cls is StableVectorVeRA else {"value_centering": True}
    return cls(backend.target.in_features, backend.target.out_features,
               rank=64, key_dim=64, top_k=4,
               temperature=0.05 if cls is StableVectorVeRA else 0.2,
               seed=cfg["seed"], **extra).to("cuda")


@torch.no_grad()
def validation(backend, module, examples, q, s, batch_size=8, background_support=None):
    backend.mode, backend.vector_store, backend.vector_override = "vector_vera", None, None
    supports = torch.cat([s, background_support]) if background_support is not None else s
    backend.vector_keys, backend.vector_values = module.encode_key(supports), module.encode_value(supports)
    scores = module.encode_query(q) @ backend.vector_keys.T
    truth = torch.arange(len(examples), device=q.device)
    result = {"retrieval_at_1": float((scores.argmax(-1) == truth).float().mean()),
              "retrieval_at_4": float((scores.topk(min(4, len(examples)), -1).indices == truth[:, None]).any(-1).float().mean())}
    for kind in ("real", "oracle"):
        nll, tokens = 0., 0
        ratio_sum, ratio_count, ratio_max = 0., 0, 0.
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start+batch_size]
            backend.oracle_values = backend.vector_values[start:start+len(batch)] if kind == "oracle" else None
            backend.batched_loss([e.question for e in batch], [e.answer for e in batch], include_eos=False)
            nll += float(backend.last_answer_nll_sums.sum())
            tokens += int(backend.last_answer_token_counts.sum())
            ratios = backend.last_answer_residual_ratio
            ratio_sum += float(ratios.sum())
            ratio_count += ratios.numel()
            ratio_max = max(ratio_max, float(ratios.max()))
        result[kind + "_answer_token_nll"] = nll / tokens
        result[kind + "_post_addition_residual_to_base_rms_mean"] = ratio_sum / ratio_count
        result[kind + "_post_addition_residual_to_base_rms_max"] = ratio_max
    result["residual_ratio_note"] = (
        "Per-answer-prediction-position RMS(result-output)/RMS(output), "
        "after BF16 addition rounding; denominator clamped to 1e-12"
    )
    backend.oracle_values = None
    return result


def train(backend, cfg, packet, output):
    module = make_module(backend, cfg)
    backend.vector_vera = module
    n = cfg["train_size"]
    examples = [Example(**x) for x in packet["examples"]["train"][:n]]
    dev = [Example(**x) for x in packet["examples"]["dev"]]
    qt, st = [packet["features"]["train"][x][:n].to("cuda") for x in ("q", "s")]
    qd, sd = [packet["features"]["dev"][x].to("cuda") for x in ("q", "s")]
    if cfg["variant"] == "stable":
        module.fit_statistics(qt, st)
    else:
        module.fit_value_center(st)
    rng = random.Random(cfg["seed"])
    address_lr = 1e-4 if cfg["variant"] == "stable" else 1e-3
    optimizer = torch.optim.Adam([*module.Wq.parameters(), *module.Wk.parameters()], lr=address_lr)
    history = {"alignment": [], "training": []}
    for step in range(cfg["alignment_steps"]):
        indices = rng.sample(range(n), min(128, n))
        optimizer.zero_grad(set_to_none=True)
        loss = module.contrastive_loss(qt[indices], st[indices], temperature=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*module.Wq.parameters(), *module.Wk.parameters()], 1.)
        optimizer.step()
        if (step+1) % 100 == 0 or step+1 == cfg["alignment_steps"]:
            with torch.no_grad():
                scores = module.encode_query(qd) @ module.encode_key(sd).T
                hit = (scores.argmax(-1) == torch.arange(len(dev), device="cuda")).float().mean()
            row = dict(step=step+1, loss=float(loss.detach()), dev_retrieval_at_1=float(hit))
            history["alignment"].append(row)
            print(json.dumps({"alignment": row}), flush=True)
    module.zero_grad(set_to_none=True)
    stable = cfg["variant"] == "stable"
    optimizer = torch.optim.Adam([
        {"params": module.Wv.parameters(), "lr": 1e-4 if stable else 1e-3},
        {"params": [module.b], "lr": 0.005 if stable else 0.03},
        {"params": [*module.Wq.parameters(), *module.Wk.parameters()], "lr": 1e-5 if stable else 0.},
    ])
    order, cursor = [], 0
    best = float("inf")
    losses = []
    start_time = time.perf_counter()
    for step in range(cfg["updates"]):
        if cursor + cfg["batch_size"] > len(order):
            order = list(range(n))
            rng.shuffle(order)
            cursor = 0
        targets = order[cursor:cursor+cfg["batch_size"]]
        cursor += len(targets)
        # Deduplicate persistent background and current episode by record ID.
        cold = list(range(min(cfg["cold_bank_size"], n)))
        distractors = rng.sample(range(n), min(cfg["episode_size"], n))
        episode = list(dict.fromkeys(targets + distractors + cold))
        rng.shuffle(episode)
        phase = "oracle" if (not stable or step < cfg["oracle_updates"]) else "real"
        backend.mode, backend.vector_store, backend.vector_override = "vector_vera", None, None
        backend.vector_keys = module.encode_key(st[episode])
        backend.vector_values = module.encode_value(st[episode])
        backend.oracle_values = module.encode_value(st[targets]) if phase == "oracle" else None
        optimizer.zero_grad(set_to_none=True)
        batch = [examples[i] for i in targets]
        lm = backend.batched_loss([e.question for e in batch], [e.answer for e in batch])
        # Include answer-predicting token inputs, not only the final prompt token.
        # Teacher forcing uses previous gold tokens only, as ordinary LM training.
        address = lm.new_zeros(())
        if stable and phase == "real":
            token_q = module.encode_query(backend.last_answer_query_inputs)
            mapping = torch.tensor([episode.index(i) for i in targets], device="cuda")
            labels = mapping[backend.last_answer_batch_indices]
            address = F.cross_entropy(token_q @ backend.vector_keys.T / 0.1, labels)
            address = address + module.contrastive_loss(qt[episode], st[episode], temperature=0.1)
        total = lm + (0.2 * address)
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite training loss")
        total.backward()
        for parameters in ([module.b], list(module.Wv.parameters()),
                           [*module.Wq.parameters(), *module.Wk.parameters()]):
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        losses.append(float(lm.detach()))
        if (step+1) % cfg["validate_every"] == 0 or step+1 in (cfg["updates"], cfg["oracle_updates"]):
            background = st[:cfg["cold_bank_size"]] if cfg["cold_bank_size"] else None
            val = validation(backend, module, dev, qd, sd, background_support=background)
            row = dict(step=step+1, exposures=(step+1)*cfg["batch_size"], phase=phase,
                       train_lm_loss=float(np.mean(losses)), address_loss=float(address.detach()),
                       seconds=time.perf_counter()-start_time, b_abs_mean=float(module.b.detach().abs().mean()), **val)
            losses.clear()
            history["training"].append(row)
            print(json.dumps({"training": row}), flush=True)
            checkpoint = {"module": module.state_dict(), "config": cfg, "step": step+1}
            torch.save(checkpoint, output / "last.pt")
            # Select by real retrieval dev NLL; test never chooses a checkpoint.
            if row["real_answer_token_nll"] < best:
                best = row["real_answer_token_nll"]
                torch.save(checkpoint, output / "best.pt")
            if step+1 == cfg["oracle_updates"]:
                torch.save(checkpoint, output / "oracle_stage.pt")
            json_write(output / "training.json", history)
    checkpoint = torch.load(output / "best.pt", map_location="cuda", weights_only=True)
    module.load_state_dict(checkpoint["module"])
    module.requires_grad_(False).eval()
    backend.oracle_values = None
    backend.mode = "none"
    return module, checkpoint["step"]


def stats(rows):
    out = {"count": len(rows)}
    if not rows:
        return out
    out.update(em=sum(r["em"] for r in rows)/len(rows),
               answer_token_nll=sum(r["nll_sum"] for r in rows)/sum(r["tokens"] for r in rows))
    routed = [r for r in rows if r["expected_in_bank"] and r["method"] in ("real", "shuffled")]
    if routed:
        out.update(recall_at_1=sum(r["selected_ids"][:1] == [r["id"]] for r in routed)/len(routed),
                   recall_at_4=sum(r["id"] in r["selected_ids"] for r in routed)/len(routed))
    decoded = [r for r in routed if r.get("decode_correct_residency") is not None]
    if decoded:
        count = sum(r["decode_query_count"] for r in decoded)
        out.update(decode_correct_residency=sum(r["decode_correct_hits"] for r in decoded)/count,
                   decode_query_count=count, decode_examples=len(decoded))
    traced = [r for r in rows if r.get("switch_count") is not None]
    if traced:
        switches = sum(r["switch_count"] for r in traced)
        transitions = sum(r["retrieval_transition_count"] for r in traced)
        out.update(switch_count_mean=switches/len(traced), switch_count_total=switches,
                   retrieval_transition_count=transitions,
                   top1_switch_rate=switches/transitions if transitions else None)
    return out


def evaluate(backend, module, db, examples, method, phase, output, paraphrase=False):
    rows = []
    ids = list(db.ids)
    backend.mode, backend.vector_store = "vector_vera", shuffled_store(db) if method == "shuffled" else db
    backend.oracle_values = None
    for ex in examples:
        backend.vector_override = None
        if method == "empty":
            backend.vector_override = torch.zeros(module.rank, device="cuda")
        elif method == "oracle":
            backend.vector_override = db.values[ids.index(ex.id)].to("cuda") if ex.id in ids else torch.zeros(module.rank, device="cuda")
        backend.prefill_retrieval = None
        question = ex.paraphrase if paraphrase else ex.question
        backend.retrieval_trace = []
        backend.trace_retrieval = True
        try:
            prediction, length, seconds = backend.generate(question, max_new_tokens=4)
        finally:
            backend.trace_retrieval = False
        sequence = [dict(step=i, phase=entry["phase"],
                         selected_ids=[ids[j] for j in entry["indices"][0].tolist()])
                    for i, entry in enumerate(backend.retrieval_trace)]
        info = backend.prefill_retrieval
        selected = [ids[i] for i in info["indices"][0, -1].tolist()] if info and info["indices"].numel() else []
        eligible = ex.id in ids and method in ("real", "shuffled")
        decode = [entry for entry in sequence if entry["phase"] == "decode"]
        hits = sum(ex.id in entry["selected_ids"] for entry in decode) if eligible else 0
        # A switch is a change in top-1 address between successive generation
        # calls, including prefill->first decode. Residency excludes prefill,
        # and forced oracle/empty reads have no address metric.
        top1 = [entry["selected_ids"][:1] for entry in sequence]
        trace_eligible = method in ("real", "shuffled") and any(top1)
        switches = sum(a != b for a, b in zip(top1, top1[1:])) if trace_eligible else None
        score = backend.score(question, ex.answer)
        row = dict(id=ex.id, method=method, phase=phase, answer=ex.answer, prediction=prediction,
                   em=int(exact_match(prediction, ex.answer)), nll_sum=score.nll_sum, tokens=score.tokens,
                   selected_ids=selected, expected_in_bank=ex.id in ids,
                   generation_tokens=length, generation_seconds=seconds,
                   token_retrieval_sequence=sequence,
                   decode_correct_residency=hits/len(decode) if eligible and decode else None,
                   decode_correct_hits=hits if eligible else None,
                   decode_query_count=len(decode) if eligible else 0,
                   switch_count=switches,
                   retrieval_transition_count=max(0, len(sequence)-1) if trace_eligible else 0)
        rows.append(row)
    with (output / "predictions.jsonl").open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def online(backend, module, cfg, packet, output):
    initial_hash = tensor_digest(module)
    data = {name: [Example(**e) for e in rows] for name, rows in packet["examples"].items()}
    db = PersistentVectorDB(64, 64, top_k=module.top_k, temperature=module.temperature)
    backend.vector_vera = module
    writes = []

    @torch.no_grad()
    def write(example, support_hidden, phase):
        hidden = support_hidden.to("cuda")
        db.write(example.id, module.encode_key(hidden).cpu(), module.encode_value(hidden).cpu(), len(writes))
        writes.append(dict(id=example.id, phase=phase, timestamp=len(writes)))

    for i in range(cfg["cold_bank_size"]):
        write(data["train"][i], packet["features"]["train"]["s"][i], "offline_initialization")
    db.save(output / "initial_vdb.pt")
    before = evaluate(backend, module, db, data["control"], "real", "control_before", output)
    pre, immediate = [], []
    stream = data["stream"][:cfg["eval_size"]]
    for i, ex in enumerate(stream):
        # Support features may be cached, but neither K/V nor DB sees them before
        # this observation boundary. Query functions never receive support text.
        pre.extend(evaluate(backend, module, db, [ex], "real", "pre_write", output))
        write(ex, packet["features"]["stream"]["s"][i], "online_observation")
        immediate.extend(evaluate(backend, module, db, [ex], "real", "immediate", output))
        if (i+1) % 32 == 0:
            print(json.dumps({"online_writes": i+1, "pre": stats(pre), "immediate": stats(immediate)}), flush=True)
    db.save(output / "final_vdb.pt")
    db_hash = db.hash()
    final = {method: stats(evaluate(backend, module, db, stream, method, "final", output))
             for method in ("real", "oracle", "shuffled", "empty")}
    para = {method: stats(evaluate(backend, module, db, stream, method, "paraphrase", output, True))
            for method in ("real", "oracle")}
    after = evaluate(backend, module, db, data["control"], "real", "control_after", output)
    if tensor_digest(module) != initial_hash or db.hash() != db_hash:
        raise AssertionError("Evaluation mutated shared weights or VDB")
    values = db.values
    centered = values - values.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    prob = energy / energy.sum().clamp_min(1e-12)
    result = dict(pre_write=stats(pre), immediate=stats(immediate), final=final, paraphrase=para,
                  control_before=stats(before), control_after=stats(after), records=len(db.ids),
                  cold_records=cfg["cold_bank_size"], numeric_vdb_bytes=db.resident_bytes(),
                  value_std_mean=float(values.std(0).mean()),
                  value_effective_rank=float(torch.exp(-(prob*prob.clamp_min(1e-12).log()).sum())) if energy.sum() > 0 else 0.,
                  value_unique_rows=int(torch.unique(values, dim=0).shape[0]),
                  online_gradient_steps=0, shared_parameters_unchanged=True,
                  value_abs_max=float(values.abs().max()), model_id="Qwen/Qwen3-4B-Instruct-2507")
    json_write(output / "writes.json", writes)
    json_write(output / "metrics.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--checkpoint", type=Path, help="Evaluate a frozen checkpoint; permits changing only deployment cold-bank size")
    p.add_argument("--variant", choices=["raw", "stable"], default="stable")
    p.add_argument("--train-size", type=int, choices=[32, 128, 1024, 4096], default=4096)
    p.add_argument("--updates", type=int, default=512)
    p.add_argument("--alignment-steps", type=int, default=400)
    p.add_argument("--oracle-updates", type=int, default=256)
    p.add_argument("--cold-bank-size", type=int, default=0)
    p.add_argument("--eval-size", type=int, choices=[16, 32, 128], default=128)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    cfg = vars(a).copy()
    cfg.update(batch_size=8, episode_size=64, validate_every=min(128, a.updates))
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}
    if not 0 <= a.cold_bank_size <= a.train_size:
        raise ValueError("Cold bank must be a subset of offline training observations")
    if a.updates < 1 or a.alignment_steps < 1:
        raise ValueError("Training budgets must be positive")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    backend = ScalingQwenBackend(a.model, layer=20)
    model_manifest = json.loads((Path(a.model).parent / "manifest.json").read_text())
    if model_manifest["revision"] != "cdbee75f17c01a7cc42f958dc650907174af0554":
        raise ValueError("This controlled experiment requires the pinned Qwen revision")
    packet = prepare_features(backend, a.cache)
    if a.prepare_only:
        print("FEATURE_CACHE_COMPLETE", flush=True)
        return
    if a.train_size == 32:
        # Smoke uses its own online entities, so pipeline repairs cannot tune
        # the final 128-case evaluation set by inspecting smoke predictions.
        smoke = synthetic_dataset(8042, 32, 16)
        for name, rows in smoke.items():
            packet["examples"][name] = [asdict(e) for e in rows]
            packet["features"][name] = {
                "q": backend.layer_features([e.question for e in rows]),
                "s": backend.layer_features([support_input(e) for e in rows]),
            }
    if a.output is None:
        raise ValueError("--output required for a training run")
    a.output.mkdir(parents=True, exist_ok=False)
    json_write(a.output / "config.json", cfg)
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), complete=False,
                    cache_sha256=digest_file(a.cache),
                    model_revision=packet["model_revision"], gpu=torch.cuda.get_device_name(0),
                    source_files_sha256={f.name: digest_file(f) for f in Path(__file__).parent.glob("*.py")})
    json_write(a.output / "manifest.json", manifest)
    for name, rows in packet["examples"].items():
        selected = rows[:a.train_size] if name == "train" else rows
        save_examples(a.output / (name+".jsonl"), [Example(**e) for e in selected])
    if a.checkpoint:
        checkpoint = torch.load(a.checkpoint, weights_only=True, map_location="cuda")
        for name in ("variant", "train_size", "seed"):
            if checkpoint["config"][name] != cfg[name]:
                raise ValueError(f"Checkpoint {name} mismatch")
        module = make_module(backend, cfg)
        module.load_state_dict(checkpoint["module"])
        module.requires_grad_(False).eval()
        selected_step = checkpoint["step"]
        manifest["checkpoint_sha256"] = digest_file(a.checkpoint)
        manifest["training_cold_bank_size"] = checkpoint["config"]["cold_bank_size"]
        manifest["source_training_config"] = checkpoint["config"]
    else:
        module, selected_step = train(backend, cfg, packet, a.output)
    result = online(backend, module, cfg, packet, a.output)
    manifest.update(complete=True, selected_step=selected_step,
                    completed_at=datetime.now(timezone.utc).isoformat())
    json_write(a.output / "manifest.json", manifest)
    print(json.dumps({"RUN_COMPLETE": result}), flush=True)


if __name__ == "__main__":
    main()
