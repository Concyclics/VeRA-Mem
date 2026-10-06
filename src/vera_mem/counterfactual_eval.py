"""Paired, frozen-weight counterfactual evaluation through the CPU vector DB.

Every B/P bank is independently copied from A: B updates one fact's value;
P paraphrases only that fact while preserving its answer. Stable record IDs
are audit labels, never retrieval filters or inputs to student generation.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from .metrics import exact_match, normalize_answer
from .run import json_write, tensor_digest
from .vector_store import PersistentVectorDB


PROTOCOL = "counterfactual-cpu-vdb-eval-v1"
_BACKEND_FIELDS = (
    "mode", "vector_vera", "vector_store", "vector_keys", "vector_values",
    "vector_override", "oracle_values", "prefill_retrieval", "last_retrieval",
    "trace_retrieval", "retrieval_trace", "capture_layer_input", "captured_input",
    "_answer_capture_batch_indices", "_answer_capture_positions",
    "last_answer_query_inputs", "last_answer_queries", "last_answer_batch_indices",
    "last_answer_token_counts", "last_answer_nll_sums", "last_residual_ratio",
    "last_answer_residual_ratio",
)


@contextmanager
def _frozen_eval(backend, module):
    missing = object()
    saved = {name: getattr(backend, name, missing) for name in _BACKEND_FIELDS}
    flags = [(p, p.requires_grad) for p in module.parameters()]
    training = module.training
    model = getattr(backend, "model", None)
    if model is not None and any(p.requires_grad for p in model.parameters()):
        raise ValueError("Counterfactual evaluation requires a frozen backbone")
    module.requires_grad_(False).eval()
    backend.vector_vera = module
    backend.vector_keys = backend.vector_values = None
    backend.vector_override = backend.oracle_values = None
    backend.capture_layer_input = backend.trace_retrieval = False
    backend._answer_capture_batch_indices = backend._answer_capture_positions = None
    try:
        with torch.no_grad():
            yield
    finally:
        for name, value in saved.items():
            if value is missing:
                if hasattr(backend, name):
                    delattr(backend, name)
            else:
                setattr(backend, name, value)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        module.train(training)


def _validate(evaluation):
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("phases"), dict):
        raise ValueError("evaluation must contain a phases mapping")
    if not evaluation["phases"]:
        raise ValueError("At least one phase is required")
    for phase, packet in evaluation["phases"].items():
        cases, features = packet["cases"], packet["supports"]
        if not isinstance(phase, str) or not phase:
            raise ValueError("Phase names must be nonempty strings")
        if len(cases) < 2:
            raise ValueError("Each phase needs at least two cases for a nonidentity shuffle")
        if not isinstance(features, torch.Tensor) or features.ndim != 3 or features.shape[:2] != (len(cases), 3):
            raise ValueError("supports must have shape [cases, 3, feature_dim]")
        if not torch.isfinite(features).all():
            raise ValueError("Support features must be finite")
        identifiers = []
        for case in cases:
            for name in ("id", "question", "answer_a", "answer_b", "support_a", "support_b", "support_p"):
                if not isinstance(case.get(name), str) or not case[name]:
                    raise ValueError(f"Case {name} must be a nonempty string")
            if exact_match(case["answer_a"], case["answer_b"]):
                raise ValueError("Paired answers must differ after exact-match normalization")
            identifiers.append(case["id"])
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Stable record IDs must be unique within a phase")
        if "evaluate_unrelated" in packet and not isinstance(packet["evaluate_unrelated"], bool):
            raise ValueError("evaluate_unrelated must be boolean")


def _independent_update(base, identifier, key, value, timestamp):
    changed = copy.deepcopy(base)
    changed.write(identifier, key, value, timestamp)
    if changed.ids != base.ids:
        raise RuntimeError("An intervention changed stable record order")
    index = base.ids.index(identifier)
    others = [i for i in range(len(base)) if i != index]
    if (not torch.equal(changed.keys[others], base.keys[others])
            or not torch.equal(changed.values[others], base.values[others])
            or any(changed.timestamps[i] != base.timestamps[i] for i in others)):
        raise RuntimeError("An intervention changed an unrelated record")
    return changed


def _shuffle_values(db, permutation):
    """Permute payloads only, preserving exact key/timestamp/ID bits."""
    result = copy.deepcopy(db)
    # Re-inserting via write() would renormalize keys and alter timestamps.
    # This isolated diagnostic copy deliberately replaces only its payload.
    result._values = db.values[permutation].contiguous().clone()
    if (result.ids != db.ids or result.timestamps != db.timestamps
            or not torch.equal(result.keys, db.keys)):
        raise RuntimeError("Shuffling changed the retrieval addresses")
    return result


def _read(backend, module, db, case, phase, world, method, answers, handle,
          max_new_tokens, context=None, target_id=None, unrelated=False):
    before = db.hash()
    backend.mode = "none" if method == "teacher" else "vector_vera"
    backend.vector_store = db
    backend.vector_keys = backend.vector_values = None
    backend.vector_override = backend.oracle_values = None
    if method == "oracle":
        backend.vector_override = db.values[list(db.ids).index(case["id"])].to(backend.device)
    elif method == "empty":
        backend.vector_override = torch.zeros(module.rank, device=backend.device)
    elif method not in ("real", "shuffled", "teacher"):
        raise ValueError("Unknown evaluation method")
    if (context is not None) != (method == "teacher"):
        raise ValueError("Only teacher reads receive a context")
    backend.prefill_retrieval = backend.last_retrieval = None
    backend.retrieval_trace = []
    backend.trace_retrieval = method in ("real", "shuffled")
    kwargs = {"context": context} if method == "teacher" else {}
    try:
        prediction, length, seconds = backend.generate(
            case["question"], max_new_tokens=max_new_tokens, **kwargs)
    finally:
        backend.trace_retrieval = False
    sequence = [dict(step=i, phase=entry["phase"],
                     selected_ids=[db.ids[j] for j in entry["indices"][0].tolist()])
                for i, entry in enumerate(backend.retrieval_trace)]
    # A trace contains only the last input position of each generation call,
    # not the prompt's earlier tokens. The first entry predicts the first token.
    selected = sequence[0]["selected_ids"] if sequence else []
    routed = method in ("real", "shuffled")
    decode = [entry for entry in sequence if entry["phase"] == "decode"]
    hits = sum(case["id"] in entry["selected_ids"] for entry in decode)
    scored = {}
    for label, answer in answers.items():
        score = backend.score(case["question"], answer, **kwargs)
        if score.tokens <= 0 or not math.isfinite(float(score.nll_sum)):
            raise ValueError("Answer scoring returned empty or nonfinite results")
        scored[label] = dict(answer=answer, em=int(exact_match(prediction, answer)),
                             nll_sum=float(score.nll_sum), tokens=int(score.tokens))
    after = db.hash()
    if before != after:
        raise RuntimeError("Vector database mutated during generation/scoring")
    row = dict(phase=phase, target_id=target_id or case["id"], query_id=case["id"],
               method=method, world=world, unrelated=unrelated, question=case["question"],
               prediction=prediction, answer_results=scored, selected_ids=selected,
               recall_at_1=int(selected[:1] == [case["id"]]) if routed else None,
               recall_at_4=int(case["id"] in selected[:4]) if routed else None,
               token_retrieval_sequence=sequence,
               decode_correct_hits=hits if routed else None,
               decode_query_count=len(decode) if routed else 0,
               bank_hash_before=before, bank_hash_after=after,
               generation_tokens=int(length), generation_seconds=float(seconds),
               answer_scoring_tokens=sum(item["tokens"] for item in scored.values()),
               teacher_context=context)
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return row


def _pair_summary(pairs, labels=("A", "B")):
    n = len(pairs)
    if not n:
        return {"count": 0}
    a = [left["answer_results"][labels[0]]["em"] for left, _ in pairs]
    b = [right["answer_results"][labels[1]]["em"] for _, right in pairs]
    joint = sum(x and y for x, y in zip(a, b))
    changed = [normalize_answer(left["prediction"]) != normalize_answer(right["prediction"])
               for left, right in pairs]
    result = dict(count=n, a_correct=sum(a), b_correct=sum(b), both_correct=joint,
                  a_em=sum(a)/n, b_em=sum(b)/n, paired_switch_em=joint/n,
                  changed_count=sum(changed), changed_rate=sum(changed)/n,
                  changed_but_not_both_correct_rate=sum(
                      change and not (x and y) for change, x, y in zip(changed, a, b))/n,
                  b_correct_given_a_correct=joint/sum(a) if sum(a) else None,
                  a_correct_given_b_correct=joint/sum(b) if sum(b) else None)
    for side, rows, label in (("a", [x for x, _ in pairs], labels[0]),
                              ("b", [y for _, y in pairs], labels[1])):
        tokens = sum(row["answer_results"][label]["tokens"] for row in rows)
        result[side + "_answer_token_nll"] = sum(
            row["answer_results"][label]["nll_sum"] for row in rows)/tokens
        for metric in ("recall_at_1", "recall_at_4"):
            scores = [row[metric] for row in rows if row[metric] is not None]
            result[side + "_" + metric] = sum(scores)/len(scores) if scores else None
        count = sum(row["decode_query_count"] for row in rows)
        result[side + "_decode_query_count"] = count
        result[side + "_decode_correct_residency"] = sum(
            row["decode_correct_hits"] or 0 for row in rows)/count if count else None
    return result


def evaluate_counterfactual(backend, module, evaluation: dict, output: Path,
                            include_teacher: bool = True, max_new_tokens: int = 4) -> dict:
    """Evaluate independently replaced CPU banks and persist reconstruction data.

    ``evaluation['phases'][name]`` contains ``cases``, support features of shape
    ``[N, 3, input_dim]`` in A/B/P order, and optional ``evaluate_unrelated``.
    Returns ``metrics['phases'][name]['methods'][method]`` paired summaries;
    ``paraphrase`` and ``unrelated`` report same-answer joint correctness.
    ``predictions.jsonl`` records every unique generation, including one empty
    generation scored against both answers. ``banks/phase_XX.pt`` stores A once
    and exact B/P row replacements, never N complete copies of the database.
    """
    _validate(evaluation)
    if not isinstance(include_teacher, bool):
        raise ValueError("include_teacher must be boolean")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "predictions.jsonl"
    if raw_path.exists():
        raise FileExistsError(f"Refusing to append a second evaluation to {raw_path}")
    (output / "banks").mkdir(exist_ok=True)
    before = tensor_digest(module)
    metrics = dict(protocol=PROTOCOL, complete=False, include_teacher=include_teacher,
                   max_new_tokens=max_new_tokens, shared_weights_before=before,
                   shared_weights_frozen=True, phases={},
                   artifacts=dict(predictions="predictions.jsonl", writes="writes.jsonl", banks=[]),
                   generation_calls=0, generation_tokens=0, answer_scoring_tokens=0)
    json_write(output / "metrics.json", metrics)
    with _frozen_eval(backend, module), raw_path.open("x", buffering=1) as raw, \
            (output / "writes.jsonl").open("x", buffering=1) as writes:
        for phase_index, (phase, packet) in enumerate(evaluation["phases"].items()):
            cases = packet["cases"]
            features = packet["supports"].to(backend.device)
            keys = module.encode_key(features).detach().cpu()
            values = module.encode_value(features).detach().cpu()
            base = PersistentVectorDB(module.key_dim, module.rank, module.temperature, module.top_k)
            for i, case in enumerate(cases):
                base.write(case["id"], keys[i, 0], values[i, 0], i)
                writes.write(json.dumps(dict(phase=phase, id=case["id"], world="A", timestamp=i,
                    support=case["support_a"], support_sha256=hashlib.sha256(case["support_a"].encode()).hexdigest())) + "\n")
            base_hash = base.hash()
            permutation = torch.arange(len(base)).roll(1)
            shuffled_a = _shuffle_values(base, permutation)
            bank_packet = dict(protocol=PROTOCOL, phase=phase, base=base.snapshot(),
                               base_hash=base_hash, shuffle_permutation=permutation.tolist(), interventions=[])
            methods = ["real", "shuffled", "oracle", "empty"] + (["teacher"] if include_teacher else [])
            pairs = {method: [] for method in methods}
            paraphrases, teacher_paraphrases, unrelated_pairs = [], [], []
            check_unrelated = packet.get("evaluate_unrelated", phase in (
                "canonical_support/canonical_query", "heldout_support/heldout_query"))

            def read(db, case, world, method, answers, **kwargs):
                row = _read(backend, module, db, case, phase, world, method, answers,
                            raw, max_new_tokens, **kwargs)
                metrics["generation_calls"] += 1
                metrics["generation_tokens"] += row["generation_tokens"]
                metrics["answer_scoring_tokens"] += row["answer_scoring_tokens"]
                return row

            for i, case in enumerate(cases):
                banks = {"A": base}
                replacement = {"id": case["id"], "index": i}
                for world, view in (("B", 1), ("P", 2)):
                    timestamp = len(cases) + i
                    bank = _independent_update(base, case["id"], keys[i, view], values[i, view], timestamp)
                    banks[world] = bank
                    replacement[world] = dict(key=bank.keys[i], write_key=keys[i, view].clone(),
                                               value=bank.values[i], timestamp=timestamp,
                                               bank_hash=bank.hash())
                    support = case["support_b" if world == "B" else "support_p"]
                    writes.write(json.dumps(dict(phase=phase, id=case["id"], world=world,
                        timestamp=timestamp, parent_bank_hash=base_hash, bank_hash=bank.hash(),
                        support=support, support_sha256=hashlib.sha256(support.encode()).hexdigest())) + "\n")
                bank_packet["interventions"].append(replacement)
                real = {world: read(banks[world], case, world, "real", {world: case[answer]})
                        for world, answer in (("A", "answer_a"), ("B", "answer_b"), ("P", "answer_a"))}
                pairs["real"].append((real["A"], real["B"]))
                paraphrases.append((real["A"], real["P"]))
                shuffled_b = _shuffle_values(banks["B"], permutation)
                pairs["shuffled"].append((
                    read(shuffled_a, case, "A", "shuffled", {"A": case["answer_a"]}),
                    read(shuffled_b, case, "B", "shuffled", {"B": case["answer_b"]})))
                pairs["oracle"].append((read(base, case, "A", "oracle", {"A": case["answer_a"]}),
                    read(banks["B"], case, "B", "oracle", {"B": case["answer_b"]})))
                empty = read(base, case, "A_and_B", "empty", {"A": case["answer_a"], "B": case["answer_b"]})
                pairs["empty"].append((empty, empty))
                if include_teacher:
                    teacher = {world: read(banks[world], case, world, "teacher", {world: case[answer]},
                                           context=case[support])
                               for world, answer, support in (("A", "answer_a", "support_a"),
                                   ("B", "answer_b", "support_b"), ("P", "answer_a", "support_p"))}
                    pairs["teacher"].append((teacher["A"], teacher["B"]))
                    teacher_paraphrases.append((teacher["A"], teacher["P"]))
                if check_unrelated:
                    other = cases[(i + 1) % len(cases)]
                    unrelated_pairs.append((
                        read(base, other, "A", "real", {"A": other["answer_a"]}, target_id=case["id"], unrelated=True),
                        read(banks["B"], other, "B", "real", {"B": other["answer_a"]}, target_id=case["id"], unrelated=True)))
                if base.hash() != base_hash:
                    raise RuntimeError("Base A database changed across independent interventions")

            def invariant_summary(rows, labels):
                result = _pair_summary(rows, labels)
                result["joint_em"] = result.pop("paired_switch_em")
                result["prediction_unchanged_rate"] = 1 - result["changed_rate"]
                return result

            bank_name = f"banks/phase_{phase_index:02d}.pt"
            torch.save(bank_packet, output / bank_name)
            metrics["artifacts"]["banks"].append(bank_name)
            metrics["phases"][phase] = dict(count=len(cases), bank_A_hash=base_hash,
                bank_A_hash_after=base.hash(), read_only_verified=True,
                methods={method: _pair_summary(rows) for method, rows in pairs.items()},
                paraphrase=invariant_summary(paraphrases, ("A", "P")),
                teacher_paraphrase=invariant_summary(teacher_paraphrases, ("A", "P")) if include_teacher else None,
                unrelated=invariant_summary(unrelated_pairs, ("A", "B")) if check_unrelated else None,
                reconstruction=bank_name)
            if tensor_digest(module) != before:
                raise RuntimeError("Shared module weights or buffers changed during evaluation")
            json_write(output / "metrics.json", metrics)
    metrics["shared_weights_after"] = tensor_digest(module)
    if metrics["shared_weights_after"] != before:
        raise RuntimeError("Shared module changed while leaving evaluation")
    metrics["complete"] = True
    json_write(output / "metrics.json", metrics)
    return metrics
