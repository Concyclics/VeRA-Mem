"""Frozen CPU-VDB evaluation of writer/readout interfaces and novel answers.

Phase letters are SUPPORT then QUERY: HC means held-out observation format and
canonical question. A/B differ in one target record only. The HC canonical-key
condition is an explicit diagnostic, not a deployment method or an upper bound.
Every student read still performs ordinary sparse mixing at the real layer.
"""
from __future__ import annotations

from collections import defaultdict
import io
import json
from pathlib import Path
import random

import torch

from .counterfactual_eval import _frozen_eval, _independent_update, _pair_summary, _read, _shuffle_values
from .metrics import normalize_answer
from .run import json_write, tensor_digest
from .vector_store import PersistentVectorDB


PROTOCOL = "interface-cpu-vdb-eval-v1"
QUERY_STYLE_SEED, SUPPORT_STYLE_SEED, CASE_SEED = 78043, 79043, 80043
PHASES = ("CC", "CH", "HC", "HH")


def _validate(packet, max_new_tokens, max_cases):
    if packet.get("split") not in {"dev", "confirm"}:
        raise ValueError("Evaluation accepts dev/confirm packets only, never the training split")
    rows = packet.get("rows", [])
    if len(rows) < 2:
        raise ValueError("At least two bank records are required")
    if type(max_new_tokens) is not int or max_new_tokens < 2:
        raise ValueError("max_new_tokens must be a positive answer-plus-EOS budget")
    if max_cases is not None and (type(max_cases) is not int or not 1 <= max_cases <= len(rows)):
        raise ValueError("max_cases must be between one and bank size")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate stable record IDs")
    qcount, scount = len(rows[0]["questions"]), len(rows[0]["supports"][0])
    if min(qcount, scount) < 2:
        raise ValueError("Canonical and held-out query/support views are required")
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k] for k in ("id", "entity", "relation", "a", "b")):
            raise ValueError("Missing fact identity/answer fields")
        if normalize_answer(row["a"]) == normalize_answer(row["b"]):
            raise ValueError("A/B answers must differ")
        if len(row["questions"]) != qcount or len(row["supports"]) != 2 or any(len(s) != scount for s in row["supports"]):
            raise ValueError("Inconsistent query/support view axes")
        if any(not isinstance(x, str) or not x for x in row["questions"] + [x for world in row["supports"] for x in world]):
            raise ValueError("Empty or non-text question/support")
        for label in ("a", "b"):
            tokens = row[label + "_tokens"]
            if not tokens or len(tokens) >= max_new_tokens:
                raise ValueError("Answer exceeds the generation budget including EOS")
    for name in ("last", "pool"):
        feature = packet[name]
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4 or feature.shape[:3] != (len(rows), 2, scount):
            raise ValueError("Support features must be [N, 2, S, D]")
        if not torch.isfinite(feature).all():
            raise ValueError("Nonfinite support features")
    if packet["last"].shape != packet["pool"].shape:
        raise ValueError("Pooled and last-token feature shapes differ")
    return qcount, scount


def view_assignments(rows, qcount, scount):
    """Answer-blind, independently balanced styles within each relation.

    Sorting by stable ID first makes assignments invariant to input row order.
    Query and support random streams are separate. Counts per relation/view
    differ by at most one; no answer label enters assignment or RNG seeding.
    """
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["relation"]].append(index)
    result = [{"id": row["id"], "entity": row["entity"], "relation": row["relation"]} for row in rows]
    for axis, count, seed in (("q", qcount, QUERY_STYLE_SEED), ("s", scount, SUPPORT_STYLE_SEED)):
        rng = random.Random(seed)
        for relation in sorted(groups):
            indices = sorted(groups[relation], key=lambda i: rows[i]["id"])
            rng.shuffle(indices)
            styles = list(range(1, count))
            rng.shuffle(styles)
            for order, index in enumerate(indices):
                result[index][axis] = styles[order % len(styles)]
    return result


def _summary(pairs):
    result = _pair_summary(pairs)
    if not pairs:
        return result
    for side, offset, label in (("a", 0, "A"), ("b", 1, "B")):
        rows = [pair[offset] for pair in pairs]
        result[side + "_answer_containment_rate"] = sum(r["answer_results"][label]["answer_containment"] for r in rows) / len(rows)
        result[side + "_budget_hit_rate"] = sum(r["budget_hit"] for r in rows) / len(rows)
        result[side + "_generation_tokens"] = sum(r["generation_tokens"] for r in rows)
        result[side + "_generation_seconds"] = sum(r["generation_seconds"] for r in rows)
    result["paired_answer_containment_rate"] = sum(
        a["answer_results"]["A"]["answer_containment"] and b["answer_results"]["B"]["answer_containment"]
        for a, b in pairs) / len(pairs)
    return result


def evaluate_interface(backend, module, packet, output, include_teacher=False,
                       max_new_tokens=32, max_cases=None):
    """Evaluate a full CPU bank, optionally querying a fixed random target subset.

    Raw traces, full A-bank snapshots, exact B-row patches and shuffle mappings
    are saved. Frozen shared weights and caller training flags are checked.
    Controls shuffled/empty are evaluated only in CC/HC; real and optional
    teacher cover all four phases. Teacher-qualified subsets supplement the
    complete denominators and require strict A/B EM, not substring containment.
    """
    qcount, scount = _validate(packet, max_new_tokens, max_cases)
    if not isinstance(include_teacher, bool):
        raise ValueError("include_teacher must be boolean")
    writer_mode = getattr(module, "writer_mode", "last_token")
    if writer_mode not in {"last_token", "masked_mean"}:
        raise ValueError("Unknown value-writer feature mode")
    rows, output = packet["rows"], Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("predictions.jsonl", "writes.jsonl", "banks")):
        raise FileExistsError("Refusing to overwrite an interface evaluation")
    (output / "banks").mkdir()
    assignments = view_assignments(rows, qcount, scount)
    selected = list(range(len(rows)))
    if max_cases is not None:
        candidates = sorted(selected, key=lambda i: rows[i]["id"])
        selected = sorted(random.Random(CASE_SEED).sample(candidates, max_cases))
    json_write(output / "assignments.json", dict(query_seed=QUERY_STYLE_SEED, support_seed=SUPPORT_STYLE_SEED,
        case_seed=CASE_SEED, assignments=assignments, selected_case_ids=[rows[i]["id"] for i in selected]))
    before = tensor_digest(module)
    saved_flags = {name: p.requires_grad for name, p in module.named_parameters()}
    previous_training = module.training
    model = getattr(backend, "model", None)
    model_training = model.training if model is not None else None
    metrics = dict(protocol=PROTOCOL, split=packet["split"], complete=False, writer_mode=writer_mode,
        bank_records=len(rows), evaluated_facts=len(selected), include_teacher=include_teacher,
        max_new_tokens=max_new_tokens, shared_weights_before=before,
        frozen_online=True, online_gradient_steps=0, phases={},
        generation_calls=0, generation_tokens=0, generation_seconds=0., answer_scoring_tokens=0,
        control_scope=dict(real=list(PHASES), teacher=list(PHASES) if include_teacher else [],
                           shuffled=["CC", "HC"], empty=["CC", "HC"], canonical_key=["HC"]),
        phase_order="support/query; index 0 canonical; held-out style balanced within relation",
        containment_note="Whole normalized answer phrase occurs in output; auxiliary, not strict format or semantic correctness",
        canonical_key_note="HC diagnostic: canonical keys with current writer's held-out values; actual sparse per-token mixing; not deployable or an upper bound",
        state_before=dict(module_training=previous_training, requires_grad=saved_flags, backbone_training=model_training),
        artifacts=dict(predictions="predictions.jsonl", writes="writes.jsonl", assignments="assignments.json", banks=[]))
    json_write(output / "metrics.json", metrics)
    try:
        if model is not None:
            model.eval()
        with _frozen_eval(backend, module), (output / "predictions.jsonl").open("x", buffering=1) as raw, \
                (output / "writes.jsonl").open("x", buffering=1) as writes:
            last = packet["last"].to(backend.device)
            value_features = packet["pool" if writer_mode == "masked_mean" else "last"].to(backend.device)
            keys = module.encode_key(last).detach().cpu()
            values = module.encode_value(value_features).detach().cpu()
            for phase in PHASES:
                sviews = [a["s"] if phase[0] == "H" else 0 for a in assignments]
                qviews = [a["q"] if phase[1] == "H" else 0 for a in assignments]
                bases = {}
                for bank_kind in (["real", "canonical_key"] if phase == "HC" else ["real"]):
                    bank = PersistentVectorDB(module.key_dim, module.rank, module.temperature, module.top_k)
                    for i, row in enumerate(rows):
                        ki = 0 if bank_kind == "canonical_key" else sviews[i]
                        bank.write(row["id"], keys[i, 0, ki], values[i, 0, sviews[i]], i)
                        writes.write(json.dumps(dict(phase=phase, bank_kind=bank_kind, world="A", id=row["id"],
                            timestamp=i, key_support_view=ki, value_support_view=sviews[i],
                            key_support=row["supports"][0][ki], value_support=row["supports"][0][sviews[i]])) + "\n")
                    bases[bank_kind] = bank
                hashes = {kind: bank.hash() for kind, bank in bases.items()}
                permutation = torch.arange(len(rows)).roll(1)
                reconstruction = dict(protocol=PROTOCOL, phase=phase,
                    bases={kind: bank.snapshot() for kind, bank in bases.items()}, base_hashes=hashes,
                    shuffle_permutation=permutation.tolist(), assignments=assignments, interventions=[])
                methods = ["real"] + (["shuffled", "empty"] if phase in {"CC", "HC"} else [])
                methods += (["canonical_key"] if phase == "HC" else []) + (["teacher"] if include_teacher else [])
                paired = {method: [] for method in methods}

                def read(bank, index, world, method, answers):
                    case = {"id": rows[index]["id"], "question": rows[index]["questions"][qviews[index]]}
                    actual_method = "real" if method == "canonical_key" else method
                    context = rows[index]["supports"][world == "B"][sviews[index]] if method == "teacher" else None
                    # The existing causal reader saves once; enrich its returned
                    # row before writing our authoritative JSONL, avoiding two
                    # competing records for the same generation.
                    event = _read(backend, module, bank, case, phase, world, actual_method, answers,
                                  io.StringIO(), max_new_tokens, context=context)
                    event.update(method=method, entity=rows[index]["entity"], relation=rows[index]["relation"],
                        query_view=qviews[index], support_view=sviews[index], budget_hit=event["generation_tokens"] >= max_new_tokens)
                    normalized_prediction = " " + normalize_answer(event["prediction"]) + " "
                    for answer in event["answer_results"].values():
                        needle = " " + normalize_answer(answer["answer"]) + " "
                        answer["answer_containment"] = int(needle in normalized_prediction)
                    raw.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                    for field in ("generation_tokens", "generation_seconds", "answer_scoring_tokens"):
                        metrics[field] += event[field]
                    metrics["generation_calls"] += 1
                    return event

                for index in selected:
                    row = rows[index]
                    worlds = {}
                    patch = dict(id=row["id"], index=index, banks={})
                    for kind, base in bases.items():
                        ki = 0 if kind == "canonical_key" else sviews[index]
                        timestamp = len(rows) + index
                        changed = _independent_update(base, row["id"], keys[index, 1, ki],
                                                      values[index, 1, sviews[index]], timestamp)
                        worlds[kind] = {"A": base, "B": changed}
                        patch["banks"][kind] = dict(write_key=keys[index, 1, ki].clone(),
                            key=changed.keys[index], value=changed.values[index], timestamp=timestamp,
                            bank_hash=changed.hash(), parent_hash=hashes[kind])
                        writes.write(json.dumps(dict(phase=phase, bank_kind=kind, world="B", id=row["id"],
                            timestamp=timestamp, parent_hash=hashes[kind], bank_hash=changed.hash(),
                            key_support_view=ki, value_support_view=sviews[index],
                            key_support=row["supports"][1][ki], value_support=row["supports"][1][sviews[index]])) + "\n")
                    reconstruction["interventions"].append(patch)
                    for method in methods:
                        kind = "canonical_key" if method == "canonical_key" else "real"
                        if method == "empty":
                            event = read(worlds[kind]["A"], index, "A_and_B", method, {"A": row["a"], "B": row["b"]})
                            paired[method].append((event, event))
                            continue
                        events = []
                        for world, answer_key in (("A", "a"), ("B", "b")):
                            bank = worlds[kind][world]
                            if method == "shuffled":
                                bank = _shuffle_values(bank, permutation)
                            events.append(read(bank, index, world, method, {world: row[answer_key]}))
                        paired[method].append(tuple(events))
                    if any(bank.hash() != hashes[kind] for kind, bank in bases.items()):
                        raise RuntimeError("Evaluation mutated an A bank across target interventions")
                teacher_ok = None
                if include_teacher:
                    teacher_ok = [bool(a["answer_results"]["A"]["em"] and b["answer_results"]["B"]["em"])
                                  for a, b in paired["teacher"]]
                bank_path = f"banks/{phase}.pt"
                torch.save(reconstruction, output / bank_path)
                metrics["artifacts"]["banks"].append(bank_path)
                metrics["phases"][phase] = dict(count=len(selected), bank_records=len(rows),
                    methods={method: _summary(pairs) for method, pairs in paired.items()},
                    teacher_acceptance=dict(count=len(selected), strict_pair_qualified=sum(teacher_ok),
                        qualified_ids=[rows[i]["id"] for i, good in zip(selected, teacher_ok) if good]) if teacher_ok is not None else None,
                    teacher_qualified_methods={method: _summary([pair for pair, good in zip(pairs, teacher_ok) if good])
                        for method, pairs in paired.items() if method != "teacher"} if teacher_ok is not None else None,
                    base_hashes_before=hashes, base_hashes_after={kind: bank.hash() for kind, bank in bases.items()},
                    read_only_verified=True, reconstruction=bank_path)
                if tensor_digest(module) != before:
                    raise RuntimeError("Shared module weights/buffers changed during evaluation")
                json_write(output / "metrics.json", metrics)
    finally:
        if model is not None:
            model.train(model_training)
    after_flags = {name: p.requires_grad for name, p in module.named_parameters()}
    metrics["shared_weights_after"] = tensor_digest(module)
    if metrics["shared_weights_after"] != before or module.training != previous_training or after_flags != saved_flags:
        raise RuntimeError("Evaluation failed to preserve module state and training flags")
    metrics.update(complete=True, state_after=dict(module_training=module.training, requires_grad=after_flags,
        backbone_training=model.training if model is not None else None))
    json_write(output / "metrics.json", metrics)
    return metrics
