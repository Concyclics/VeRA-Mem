"""Strict, aggregate-only audit of the five-arm counterfactual confirmation.

Example: python scripts/summarize_counterfactual.py --initial EVAL_INITIAL
  --warm WARM --training base=TRAIN_BASE --evaluation base=EVAL_BASE
  [repeat --training/--evaluation for behavior, hidden, mixed] --output PUBLIC

All evidence is read-only. Prompts, token IDs, entity IDs and absolute paths are
excluded from the public JSON/Markdown. The initial arm precedes common warm-up.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from summarize_context_distillation import (
    assert_equal, digest_file, object_digest, read_json, read_jsonl,
    require_finite, require_hash, require_int,
)
from vera_mem.metrics import exact_match, normalize_answer


ARMS = ("base", "behavior", "hidden", "mixed")
EVAL_ARMS = ("initial", *ARMS)
PHASES = tuple(f"{support}_support/{query}_query" for support in ("canonical", "heldout")
               for query in ("canonical", "heldout"))
UNRELATED_PHASES = (PHASES[0], PHASES[-1])
METHODS = ("real", "shuffled", "oracle", "empty")
PROTOCOL = "counterfactual-context-v1"
EVALUATION_PROTOCOL = "counterfactual-cpu-vdb-eval-v1"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
CONFIRMATION_FACTS = 64
CORE_SOURCE_FILES = (
    "counterfactual_backend.py", "counterfactual_losses.py", "counterfactual_run.py",
    "counterfactual_eval.py", "counterfactual_data.py", "context_distillation.py",
    "context_distillation_run.py", "scaling_backend.py", "backend.py",
    "stable_vector_vera.py", "vector_vera.py", "vector_store.py", "factcentric_losses.py",
    "metrics.py", "augmentation_data.py", "data.py", "run.py",
)
CONFIG_FIELDS = ("method", "updates", "start_step", "batch_size", "seed", "evaluate_only", "teacher")
TOKEN_FIELDS = (
    "student_input_tokens", "teacher_input_tokens", "target_tokens", "sampled_tokens",
    "rollout_input_tokens", "rollout_forward_calls", "replay_input_tokens", "replay_target_tokens",
)
DIAGNOSTIC_COUNTS = (
    "behavior_valid_pairs", "behavior_clipped_pairs", "hidden_valid_pairs",
    "teacher_first_token_joint_correct", "student_first_token_joint_correct",
)
DIAGNOSTIC_MEANS = (
    "main_loss", "total_loss", "forward_kl", "first_token_ce", "actual_address_loss",
    "style_address_loss", "behavior_loss", "hidden_loss", "paraphrase_loss", "replay_loss",
    "teacher_behavior_delta_mean", "student_behavior_delta_mean", "teacher_delta_norm_mean",
    "student_delta_norm_mean",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _safe_relative(value):
    _require(isinstance(value, str) and value, "Missing relative artifact path")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts, "Unsafe artifact path")
    return path


def _identity(value):
    _require(isinstance(value, str) and bool(value), "Missing/invalid fact identifier")
    return value


def audit_manifest(directory, *, evaluation):
    directory = Path(directory)
    manifest = read_json(directory/"manifest.json")
    _require(manifest.get("protocol") == PROTOCOL, "Unexpected run protocol")
    _require(manifest.get("complete") is True, "Run manifest is incomplete")
    _require(manifest.get("teacher_parameters_unchanged") is True, "Frozen backbone verification missing")
    _require(manifest.get("model_revision") == MODEL_REVISION, "Pinned model revision mismatch")
    before = require_hash(manifest.get("teacher_parameter_sha256_before"), "teacher before")
    _require(manifest.get("teacher_parameter_sha256_after") == before, "Frozen backbone hashes differ")
    config = manifest.get("configuration")
    _require(isinstance(config, dict), "Missing run configuration")
    _require(config.get("evaluate_only") is evaluation, "Training/evaluation mode mismatch")
    _require(config.get("method") in ARMS, "Unknown training method")
    _require(isinstance(config.get("teacher"), bool), "Missing teacher-evaluation setting")
    for name in ("updates", "batch_size"):
        require_int(config.get(name), name, 1)
    for name in ("start_step", "seed"):
        require_int(config.get(name), name)
    _require(manifest.get("cache_split") == ("confirmation" if evaluation else "train"), "Wrong cache split")
    source = manifest.get("source_files_sha256")
    _require(isinstance(source, dict) and set(CORE_SOURCE_FILES).issubset(source), "Required runtime source hashes missing")
    snapshot = directory.parent/"source/src/vera_mem"
    for name, expected in source.items():
        _require(Path(name).name == name, "Runtime source names must be basenames")
        _require(require_hash(expected, "source hash") == digest_file(snapshot/name), "Immutable source snapshot hash mismatch")
    hashes = {name: require_hash(manifest.get(name), name) for name in (
        "cache_sha256", "checkpoint_sha256", "module_sha256_before", "module_sha256_after")}
    provenance = dict(**hashes, model_revision=MODEL_REVISION, teacher_parameter_sha256=before,
                      runtime_source_files_sha256=source, runtime_source_sha256=object_digest(source),
                      manifest_sha256=digest_file(directory/"manifest.json"))
    return manifest, {name: config[name] for name in CONFIG_FIELDS}, provenance


def _trajectory_tokens(row):
    tokens, golden = row.get("continuation_token_ids"), row.get("gold_token_ids")
    for collection in (tokens, golden):
        _require(isinstance(collection, list) and len(collection) == 3, "Trajectories require A/B/P continuations")
        for sequence in collection:
            _require(isinstance(sequence, list) and sequence and all(type(token) is int and token >= 0 for token in sequence),
                     "Invalid trajectory token sequence")
    _require(golden[0] == golden[2] and len(golden[0]) == len(golden[1]) and golden[0][0] != golden[1][0],
             "A/B equal-length distinct-first-token or A/P gold invariant failed")
    _require(golden[0][-1] == golden[1][-1], "A/B gold EOS token mismatch")
    if row.get("on_policy") is True:
        _require(tokens[0] == tokens[1] == tokens[2], "On-policy A/B/P must share the same sampled prefix")
        _require(type(row.get("sampled_world")) is int and row["sampled_world"] in (0, 1), "Invalid sampled world")
    else:
        _require(row.get("on_policy") is False and tokens == golden and row.get("sampled_world") is None,
                 "Off-policy trajectories must use gold prefixes")
    contexts = row.get("contexts")
    _require(isinstance(contexts, list) and len(contexts) == 3 and all(isinstance(c, str) and c for c in contexts),
             "Missing A/B/P teacher contexts")
    _require(contexts[0] != contexts[1] and contexts[0] != contexts[2], "Counterfactual or paraphrase teacher context is unchanged")
    _require(isinstance(row.get("question"), str) and row["question"], "Trajectory question missing")
    return tokens, golden


def audit_training(directory):
    directory = Path(directory)
    manifest, config, provenance = audit_manifest(directory, evaluation=False)
    status = read_json(directory/"training_status.json")
    _require(status.get("complete") is True, "Training status is incomplete")
    assert_equal(manifest.get("training"), status, "training manifest/status")
    _require("evaluation" not in manifest, "Training run unexpectedly contains confirmation evaluation")
    updates, batch, start = config["updates"], config["batch_size"], config["start_step"]
    rows = read_jsonl(directory/"training.jsonl")
    _require(len(rows) == updates, "Missing/duplicate training updates")
    trajectories = defaultdict(dict)
    for row in read_jsonl(directory/"trajectories.jsonl"):
        step, identity = require_int(row.get("step"), "trajectory step", 1), _identity(row.get("id"))
        _require(identity not in trajectories[step], "Duplicate trajectory fact/step")
        trajectories[step][identity] = row
    _require(set(trajectories) == set(range(start+1, start+updates+1)), "Missing/extra trajectory steps")
    schedule = hashlib.sha256()
    token_totals = {name: 0 for name in TOKEN_FIELDS}
    diagnostic_totals = {name: 0 for name in DIAGNOSTIC_COUNTS}
    diagnostic_means = {name: 0. for name in DIAGNOSTIC_MEANS}
    last_elapsed = 0.
    on_policy_updates = 0
    for local_step, row in enumerate(rows, 1):
        step = start+local_step
        _require(row.get("step") == step and row.get("local_step") == local_step, "Nonsequential training steps")
        on_policy = config["method"] == "mixed" and start > 0 and local_step % 4 == 0
        _require(row.get("on_policy") is on_policy, "Mixed/off-policy schedule mismatch")
        on_policy_updates += int(on_policy)
        target_ids, episode_ids = row.get("target_ids"), row.get("episode_ids")
        _require(isinstance(target_ids, list) and len(target_ids) == batch and len(set(target_ids)) == batch,
                 "Invalid training target facts")
        _require(isinstance(episode_ids, list) and len(set(episode_ids)) == len(episode_ids)
                 and set(target_ids).issubset(episode_ids), "Invalid episode facts")
        for identity in episode_ids:
            _identity(identity)
        _require(row.get("columns") == [episode_ids.index(identity) for identity in target_ids], "Episode target columns mismatch")
        for name, maximum in (("query_views", 8), ("support_views", 4)):
            view = row.get(name)
            _require(isinstance(view, list) and len(view) == len(episode_ids)
                     and all(type(value) is int and 0 <= value < maximum for value in view), "Invalid training view schedule")
        exposure = {name: row[name] for name in ("target_ids", "episode_ids", "query_views", "support_views", "columns")}
        schedule.update(json.dumps(exposure, sort_keys=True, separators=(",", ":")).encode())
        _require(set(trajectories[step]) == set(target_ids), "Trajectory/target fact mismatch")
        target_tokens = sampled_tokens = replay_targets = 0
        for identity in target_ids:
            trajectory = trajectories[step][identity]
            _require(trajectory.get("on_policy") is on_policy, "Trajectory on-policy flag mismatch")
            tokens, golden = _trajectory_tokens(trajectory)
            target_tokens += sum(map(len, tokens))
            sampled_tokens += len(tokens[0]) if on_policy else 0
            replay_targets += len(golden[0]) if step % 4 == 0 else 0
        for name, expected in (("target_tokens", target_tokens), ("sampled_tokens", sampled_tokens), ("replay_target_tokens", replay_targets)):
            _require(row.get(name) == expected, "Trajectory-derived training token count mismatch: "+name)
        if not on_policy:
            _require(row.get("rollout_input_tokens") == row.get("rollout_forward_calls") == 0, "Off-policy rollout cost must be zero")
        else:
            _require(row.get("rollout_input_tokens", 0) > 0 and row.get("rollout_forward_calls") == sampled_tokens,
                     "Independent per-question rollout cost mismatch")
        if step % 4:
            _require(row.get("replay_input_tokens") == 0, "Unexpected canonical replay cost")
        for name in TOKEN_FIELDS:
            token_totals[name] += require_int(row.get(name), name)
        for name in DIAGNOSTIC_COUNTS:
            count = require_int(row.get(name), name)
            _require(count <= batch, "Diagnostic pair count exceeds batch")
            diagnostic_totals[name] += count
        for name in DIAGNOSTIC_MEANS:
            diagnostic_means[name] += require_finite(row.get(name), name)/updates
        norms = row.get("gradient_norms")
        _require(isinstance(norms, list) and len(norms) == 3, "Missing optimizer gradient norms")
        for norm in norms:
            _require(require_finite(norm, "gradient norm") >= 0, "Negative gradient norm")
        elapsed = require_finite(row.get("elapsed_seconds"), "training elapsed")
        _require(elapsed >= last_elapsed, "Nonmonotonic training elapsed time")
        last_elapsed = elapsed
    expected = dict(step=start+updates, updates=updates, target_pair_exposures=updates*batch,
                    episode_schedule_sha256=schedule.hexdigest(), token_totals=token_totals)
    for name, value in expected.items():
        assert_equal(status.get(name), value, "training status."+name)
    elapsed = require_finite(status.get("elapsed_seconds"), "total training elapsed")
    _require(elapsed >= last_elapsed, "Final training elapsed time precedes last update")
    provenance.update(final_checkpoint_sha256=digest_file(directory/"last.pt"),
                      evidence_sha256={name: digest_file(directory/name) for name in (
                          "manifest.json", "training_status.json", "training.jsonl", "trajectories.jsonl", "last.pt")})
    return dict(configuration=config, provenance=provenance, training=dict(
        **expected, start_step=start, elapsed_seconds=elapsed, on_policy_updates=on_policy_updates,
        canonical_replay_pair_exposures=sum(step % 4 == 0 for step in range(start+1, start+updates+1))*batch,
        processed_input_tokens_total=sum(token_totals[name] for name in (
            "student_input_tokens", "teacher_input_tokens", "rollout_input_tokens", "replay_input_tokens")),
        diagnostic_pair_counts=diagnostic_totals,
        diagnostic_pair_rates={name: value/(updates*batch) for name, value in diagnostic_totals.items()},
        diagnostic_step_means=diagnostic_means,
    ))


def pair_summary(pairs, labels=("A", "B"), *, invariant=False):
    """Independently reconstruct the actual evaluator's complete pair schema."""
    count = len(pairs)
    _require(count > 0, "Cannot aggregate an empty pair group")
    a = [left["answer_results"][labels[0]]["em"] for left, _ in pairs]
    b = [right["answer_results"][labels[1]]["em"] for _, right in pairs]
    both = sum(x and y for x, y in zip(a, b))
    changed = [normalize_answer(left["prediction"]) != normalize_answer(right["prediction"]) for left, right in pairs]
    result = dict(count=count, a_correct=sum(a), b_correct=sum(b), both_correct=both,
                  a_em=sum(a)/count, b_em=sum(b)/count, paired_switch_em=both/count,
                  changed_count=sum(changed), changed_rate=sum(changed)/count,
                  changed_but_not_both_correct_rate=sum(change and not (x and y) for change, x, y in zip(changed, a, b))/count,
                  b_correct_given_a_correct=both/sum(a) if sum(a) else None,
                  a_correct_given_b_correct=both/sum(b) if sum(b) else None)
    for side, selected, label in (("a", [row[0] for row in pairs], labels[0]), ("b", [row[1] for row in pairs], labels[1])):
        tokens = sum(row["answer_results"][label]["tokens"] for row in selected)
        result[side+"_answer_token_nll"] = sum(row["answer_results"][label]["nll_sum"] for row in selected)/tokens
        for name in ("recall_at_1", "recall_at_4"):
            values = [row[name] for row in selected if row[name] is not None]
            result[side+"_"+name] = sum(values)/len(values) if values else None
        queries = sum(row["decode_query_count"] for row in selected)
        result[side+"_decode_query_count"] = queries
        result[side+"_decode_correct_residency"] = sum(row["decode_correct_hits"] or 0 for row in selected)/queries if queries else None
    if invariant:
        result["joint_em"] = result.pop("paired_switch_em")
        result["prediction_unchanged_rate"] = 1-result["changed_rate"]
    return result


def pair_diagnostics(pairs):
    swapped = wrong_wrong = stale_b = 0
    for a, b in pairs:
        score_a, score_b = a["answer_results"]["A"], b["answer_results"]["B"]
        swapped += bool(exact_match(a["prediction"], score_b["answer"]) and exact_match(b["prediction"], score_a["answer"]))
        wrong_wrong += not score_a["em"] and not score_b["em"]
        stale_b += bool(exact_match(b["prediction"], score_a["answer"]))
    return dict(swapped_answer_count=swapped, swapped_answer_rate=swapped/len(pairs),
                wrong_wrong_count=wrong_wrong, wrong_wrong_rate=wrong_wrong/len(pairs),
                b_outputs_old_a_count=stale_b, b_outputs_old_a_rate=stale_b/len(pairs))


def _validate_prediction(row, ids):
    method, world = row.get("method"), row.get("world")
    routed = method in ("real", "shuffled")
    for name in ("target_id", "query_id"):
        _require(_identity(row.get(name)) in ids, "Prediction ID outside phase bank")
    _require(type(row.get("unrelated")) is bool, "Missing unrelated flag")
    _require(isinstance(row.get("question"), str) and row["question"], "Missing question")
    _require(isinstance(row.get("prediction"), str), "Missing prediction text")
    _require(row.get("bank_hash_before") == row.get("bank_hash_after"), "Bank changed during read")
    require_hash(row.get("bank_hash_before"), "prediction bank hash")
    require_int(row.get("generation_tokens"), "generation tokens")
    _require(require_finite(row.get("generation_seconds"), "generation seconds") >= 0, "Negative generation time")
    expected_labels = {"A", "B"} if method == "empty" else {world}
    scores = row.get("answer_results")
    _require(isinstance(scores, dict) and set(scores) == expected_labels, "Answer score labels do not match world")
    for score in scores.values():
        _require(isinstance(score, dict) and set(score) == {"answer", "em", "nll_sum", "tokens"}, "Answer score schema mismatch")
        _require(isinstance(score["answer"], str) and score["answer"], "Missing answer label")
        _require(type(score["em"]) in (int, float) and score["em"] in (0, 1)
                 and score["em"] == exact_match(row["prediction"], score["answer"]), "Stored exact_match disagrees with prediction")
        require_finite(score["nll_sum"], "answer NLL")
        require_int(score["tokens"], "answer tokens", 1)
    _require(row.get("answer_scoring_tokens") == sum(score["tokens"] for score in scores.values()), "Answer scoring token sum mismatch")
    selected, trace = row.get("selected_ids"), row.get("token_retrieval_sequence")
    _require(isinstance(selected, list) and isinstance(trace, list), "Retrieval evidence missing")
    for entry in trace:
        keys = entry.get("selected_ids")
        _require(isinstance(keys, list) and 1 <= len(keys) <= 4 and len(set(keys)) == len(keys)
                 and set(keys).issubset(ids), "Invalid traced top-k addresses")
    if routed:
        _require(bool(trace) and trace[0].get("phase") == "prefill" and selected == trace[0]["selected_ids"], "Prefill retrieval trace mismatch")
        _require([entry.get("step") for entry in trace] == list(range(len(trace)))
                 and all(entry.get("phase") == "decode" for entry in trace[1:]), "Generation retrieval trace ordering mismatch")
        query = row["query_id"]
        _require(row.get("recall_at_1") == int(selected[:1] == [query])
                 and row.get("recall_at_4") == int(query in selected[:4]), "Stored recall disagrees with addresses")
        _require(row.get("decode_query_count") == len(trace)-1
                 and row.get("decode_correct_hits") == sum(query in entry["selected_ids"] for entry in trace[1:]),
                 "Decode retrieval count mismatch")
    else:
        _require(selected == trace == [] and row.get("recall_at_1") is None and row.get("recall_at_4") is None
                 and row.get("decode_correct_hits") is None and row.get("decode_query_count") == 0,
                 "Forced/teacher branch unexpectedly has retrieval metrics")
    _require((row.get("teacher_context") is not None) == (method == "teacher"), "Student receives context or teacher context missing")


def audit_bank_reconstruction(path, phase, ids, writes, expected_hash):
    """Replay only CPU VDB writes, verifying stored tensor payloads and hashes.

    This does not load a language model or run inference. The restricted torch
    reader accepts only tensor/primitive checkpoints; no evidence is rewritten.
    """
    import torch
    from vera_mem.vector_store import PersistentVectorDB

    def content_hash(store):
        # Same byte definition as PersistentVectorDB.hash, using contiguous
        # CPU buffers instead of allocating a Python integer per payload byte.
        # This matters when auditing thousands of independent bank copies.
        metadata = dict(format_version=store.FORMAT_VERSION,
                        config=dict(key_dim=store.key_dim, value_dim=store.value_dim,
                                    temperature=store.temperature, top_k=store.top_k),
                        ids=store._ids, timestamps=store._timestamps)
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, ensure_ascii=False,
                                          separators=(",", ":"), allow_nan=False).encode())
        for tensor in (store._keys, store._values):
            digest.update(tensor.contiguous().numpy().tobytes())
        return digest.hexdigest()

    packet = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(packet, dict) and packet.get("protocol") == EVALUATION_PROTOCOL
             and packet.get("phase") == phase, "Reconstruction bank protocol/phase mismatch")
    state = packet.get("base")
    _require(isinstance(state, dict) and set(state) == {"format_version", "config", "ids", "timestamps", "keys", "values"}, "Base bank snapshot schema mismatch")
    _require(state["format_version"] == PersistentVectorDB.FORMAT_VERSION, "Base bank format version mismatch")
    _require(state["config"] == dict(key_dim=64, value_dim=64, temperature=.05, top_k=4), "Formal bank dimensions/configuration mismatch")
    _require(state["ids"] == ids and state["timestamps"] == list(range(len(ids))), "Base bank IDs/timestamps differ from writes")
    for name in ("keys", "values"):
        tensor = state[name]
        _require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
                 and tuple(tensor.shape) == (len(ids), 64) and bool(torch.isfinite(tensor).all()), "Invalid base bank tensor")
    _require(bool(torch.allclose(state["keys"].norm(dim=-1), torch.ones(len(ids)), atol=1e-5, rtol=1e-5)), "Base bank keys are not normalized")
    # Preserve exact recorded key bits, just like PersistentVectorDB.load.
    base = PersistentVectorDB(**state["config"])
    base._ids, base._timestamps = list(state["ids"]), list(state["timestamps"])
    base._id_to_index = {identity: index for index, identity in enumerate(ids)}
    base._keys, base._values = state["keys"].clone(), state["values"].clone()
    _require(content_hash(base) == packet.get("base_hash") == expected_hash, "Reconstructed base bank hash mismatch")
    permutation = torch.arange(len(ids)).roll(1)
    _require(packet.get("shuffle_permutation") == permutation.tolist(), "Shuffle permutation is not the declared nonidentity rotation")

    def shuffled_hash(store):
        shuffled = copy.deepcopy(store)
        shuffled._values = store.values[permutation].contiguous().clone()
        return content_hash(shuffled)

    output = dict(shuffled_a=shuffled_hash(base), shuffled_b={})
    interventions = packet.get("interventions")
    _require(isinstance(interventions, list) and len(interventions) == len(ids), "Missing/extra bank interventions")
    for index, (identity, intervention) in enumerate(zip(ids, interventions)):
        _require(intervention.get("id") == identity and intervention.get("index") == index,
                 "Bank intervention ID/order mismatch")
        for world in ("B", "P"):
            replacement = intervention.get(world)
            _require(isinstance(replacement, dict) and set(replacement) == {"key", "write_key", "value", "timestamp", "bank_hash"}, "Bank replacement schema mismatch")
            for name in ("key", "write_key", "value"):
                tensor = replacement[name]
                _require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
                         and tuple(tensor.shape) == (64,) and bool(torch.isfinite(tensor).all()), "Invalid replacement tensor")
            _require(replacement["timestamp"] == writes[phase, identity, world]["timestamp"], "Replacement/write timestamp mismatch")
            changed = copy.deepcopy(base)
            changed.write(identity, replacement["write_key"], replacement["value"], replacement["timestamp"])
            _require(torch.equal(changed.keys[index], replacement["key"]), "Reconstructed normalized key differs from recorded replacement")
            _require(content_hash(changed) == replacement["bank_hash"] == writes[phase, identity, world]["bank_hash"], "Reconstructed intervention bank hash mismatch")
            if world == "B":
                output["shuffled_b"][identity] = shuffled_hash(changed)
    return output


def audit_evaluation(directory, *, include_teacher):
    directory = Path(directory)
    manifest, config, provenance = audit_manifest(directory, evaluation=True)
    _require(config["teacher"] is include_teacher, "Initial/other teacher evaluation policy mismatch")
    _require("training" not in manifest, "Evaluation unexpectedly includes training")
    _require(provenance["module_sha256_before"] == provenance["module_sha256_after"], "Evaluation module changed")
    metrics = read_json(directory/"metrics.json")
    assert_equal(manifest.get("evaluation"), metrics, "evaluation manifest/metrics")
    _require(metrics.get("protocol") == EVALUATION_PROTOCOL and metrics.get("complete") is True, "Evaluation metrics are incomplete or wrong protocol")
    _require(metrics.get("include_teacher") is include_teacher and metrics.get("shared_weights_frozen") is True, "Frozen evaluation policy missing")
    _require(metrics.get("shared_weights_before") == metrics.get("shared_weights_after") == provenance["module_sha256_before"], "Evaluation shared-weight hash mismatch")
    _require(set(metrics.get("phases", {})) == set(PHASES), "Four confirmation phases required")
    _require(metrics.get("max_new_tokens") == 4, "Unexpected confirmation generation budget")
    artifacts = metrics.get("artifacts")
    _require(isinstance(artifacts, dict) and artifacts.get("predictions") == "predictions.jsonl"
             and artifacts.get("writes") == "writes.jsonl", "Unexpected evaluation artifact names")
    bank_names = [f"banks/phase_{index:02d}.pt" for index in range(4)]
    _require(artifacts.get("banks") == bank_names, "Missing/extra reconstruction bank artifacts")
    writes = {}
    for row in read_jsonl(directory/"writes.jsonl"):
        phase, identity, world = row.get("phase"), _identity(row.get("id")), row.get("world")
        _require(phase in PHASES and world in ("A", "B", "P"), "Unexpected write phase/world")
        key = phase, identity, world
        _require(key not in writes, "Duplicate bank write")
        support = row.get("support")
        _require(isinstance(support, str) and bool(support), "Missing observation support")
        _require(row.get("support_sha256") == hashlib.sha256(support.encode()).hexdigest(), "Observation support SHA mismatch")
        require_int(row.get("timestamp"), "write timestamp")
        writes[key] = row
    ids_by_phase = {}
    for phase in PHASES:
        originals = [row for (name, _, world), row in writes.items() if name == phase and world == "A"]
        originals.sort(key=lambda row: row["timestamp"])
        _require(len(originals) == CONFIRMATION_FACTS and [row["timestamp"] for row in originals] == list(range(CONFIRMATION_FACTS)),
                 "Formal confirmation must contain 64 ordered facts; smoke evidence is forbidden")
        ids = [row["id"] for row in originals]
        ids_by_phase[phase] = ids
        base_hash = require_hash(metrics["phases"][phase].get("bank_A_hash"), "base bank hash")
        _require(metrics["phases"][phase].get("bank_A_hash_after") == base_hash
                 and metrics["phases"][phase].get("read_only_verified") is True, "Base bank immutability missing")
        for index, identity in enumerate(ids):
            for world in ("B", "P"):
                row = writes.get((phase, identity, world))
                _require(row is not None, "Missing target intervention write")
                _require(row.get("parent_bank_hash") == base_hash and row["timestamp"] == CONFIRMATION_FACTS+index,
                         "Intervention is not an independent update from base A")
                require_hash(row.get("bank_hash"), "intervention bank hash")
    _require(len(writes) == 4*CONFIRMATION_FACTS*3, "Extra bank write records")
    _require(all(ids == ids_by_phase[PHASES[0]] for ids in ids_by_phase.values()), "Fact/order sets differ across phases")
    predictions = read_jsonl(directory/"predictions.jsonl")
    indexed = {}
    for row in predictions:
        phase = row.get("phase")
        _require(phase in PHASES, "Unknown prediction phase")
        _validate_prediction(row, set(ids_by_phase[phase]))
        key = phase, row["target_id"], row["unrelated"], row.get("method"), row.get("world")
        _require(key not in indexed, "Duplicate prediction record")
        indexed[key] = row
    public_phases, private = {}, dict(phases={}, labels={}, questions={})
    expected_keys = set()
    for phase_index, phase in enumerate(PHASES):
        stored, ids = metrics["phases"][phase], ids_by_phase[phase]
        reconstructed = audit_bank_reconstruction(directory/bank_names[phase_index], phase, ids, writes, stored["bank_A_hash"])
        _require(stored.get("count") == CONFIRMATION_FACTS, "Phase fact count mismatch")
        methods = (*METHODS, "teacher") if include_teacher else METHODS
        _require(set(stored.get("methods", {})) == set(methods), "Missing/extra evaluation method")
        pairs, paraphrases, teacher_paraphrases, unrelated = {method: [] for method in methods}, [], [], []
        phase_private = {identity: {} for identity in ids}

        def get(identity, method, world, unrelated_query=False):
            key = phase, identity, unrelated_query, method, world
            expected_keys.add(key)
            _require(key in indexed, "Missing prediction record")
            return indexed[key]

        for index, identity in enumerate(ids):
            real = {world: get(identity, "real", world) for world in ("A", "B", "P")}
            answer_a = real["A"]["answer_results"]["A"]["answer"]
            answer_b = real["B"]["answer_results"]["B"]["answer"]
            _require(not exact_match(answer_a, answer_b), "Counterfactual answers must differ")
            _require(real["P"]["answer_results"]["P"]["answer"] == answer_a, "Paraphrase changed the answer")
            _require(real["A"]["answer_results"]["A"]["tokens"] == real["B"]["answer_results"]["B"]["tokens"], "Counterfactual answer token lengths differ")
            if identity in private["labels"]:
                _require(private["labels"][identity] == (answer_a, answer_b), "Counterfactual labels differ across phases")
            private["labels"][identity] = (answer_a, answer_b)
            private["questions"][phase, identity] = real["A"]["question"]
            support_a, support_b, support_p = [writes[phase, identity, world]["support"] for world in ("A", "B", "P")]
            _require(support_a != support_p, "Paraphrase support did not change")
            word = r"\b"+re.escape(answer_a)+r"\b"
            _require(len(re.findall(word, support_a)) == 1
                     and re.sub(word, lambda _: answer_b, support_a) == support_b,
                     "B support changes more than the target answer text")
            _require(len(re.findall(word, support_p)) == 1, "Paraphrase support does not retain answer A")
            for world in ("A", "B", "P"):
                row = real[world]
                _require(row["query_id"] == identity and row["question"] == real["A"]["question"], "Paired worlds change the entity/question")
            for method in methods:
                if method == "empty":
                    empty = get(identity, method, "A_and_B")
                    selected = {"A": empty, "B": empty}
                else:
                    selected = {world: get(identity, method, world) for world in ("A", "B")}
                for world, row in selected.items():
                    _require(row["query_id"] == identity and row["question"] == real["A"]["question"], "Control/teacher query differs")
                    expected_answer = answer_a if world == "A" else answer_b
                    _require(row["answer_results"][world]["answer"] == expected_answer, "Control/teacher answer label differs")
                    if method != "shuffled":
                        expected_hash = stored["bank_A_hash"] if world == "A" or method == "empty" else writes[phase, identity, "B"]["bank_hash"]
                        _require(row["bank_hash_before"] == expected_hash, "Prediction bank hash differs from its intervention")
                    else:
                        expected_hash = reconstructed["shuffled_a"] if world == "A" else reconstructed["shuffled_b"][identity]
                        _require(row["bank_hash_before"] == expected_hash, "Shuffled bank hash differs from exact payload permutation")
                    if method == "teacher":
                        _require(row["teacher_context"] == writes[phase, identity, world]["support"], "Teacher context differs from observation")
                pairs[method].append((selected["A"], selected["B"]))
                phase_private[identity][method] = dict(
                    paired_switch=int(selected["A"]["answer_results"]["A"]["em"] and selected["B"]["answer_results"]["B"]["em"]),
                    wrong_wrong=int(not selected["A"]["answer_results"]["A"]["em"] and not selected["B"]["answer_results"]["B"]["em"]),
                    swapped=int(exact_match(selected["A"]["prediction"], answer_b) and exact_match(selected["B"]["prediction"], answer_a)),
                )
            _require(real["P"]["bank_hash_before"] == writes[phase, identity, "P"]["bank_hash"], "Paraphrase bank hash mismatch")
            paraphrases.append((real["A"], real["P"]))
            phase_private[identity]["paraphrase_joint"] = int(real["A"]["answer_results"]["A"]["em"] and real["P"]["answer_results"]["P"]["em"])
            if include_teacher:
                teacher_a = get(identity, "teacher", "A")
                teacher_p = get(identity, "teacher", "P")
                _require(teacher_p["query_id"] == identity and teacher_p["question"] == real["A"]["question"]
                         and teacher_p["answer_results"]["P"]["answer"] == answer_a
                         and teacher_p["teacher_context"] == support_p
                         and teacher_p["bank_hash_before"] == writes[phase, identity, "P"]["bank_hash"], "Teacher paraphrase mismatch")
                teacher_paraphrases.append((teacher_a, teacher_p))
                phase_private[identity]["teacher_paraphrase_joint"] = int(teacher_a["answer_results"]["A"]["em"] and teacher_p["answer_results"]["P"]["em"])
                phase_private[identity]["teacher_abp_joint"] = int(phase_private[identity]["teacher"]["paired_switch"] and teacher_p["answer_results"]["P"]["em"])
            if phase in UNRELATED_PHASES:
                other = ids[(index+1) % len(ids)]
                a, b = [get(identity, "real", world, True) for world in ("A", "B")]
                other_primary = get(other, "real", "A")
                for world, row in (("A", a), ("B", b)):
                    _require(row["query_id"] == other and row["question"] == other_primary["question"]
                             and row["answer_results"][world]["answer"] == other_primary["answer_results"]["A"]["answer"],
                             "Unrelated query/label differs from declared neighbor")
                    expected_hash = stored["bank_A_hash"] if world == "A" else writes[phase, identity, "B"]["bank_hash"]
                    _require(row["bank_hash_before"] == expected_hash, "Unrelated read uses the wrong intervention bank")
                unrelated.append((a, b))
                phase_private[identity]["unrelated_joint"] = int(a["answer_results"]["A"]["em"] and b["answer_results"]["B"]["em"])
        methods_public = {}
        for method, records in pairs.items():
            computed = pair_summary(records)
            assert_equal(stored["methods"][method], computed, f"{phase}/{method}")
            methods_public[method] = {**computed, **pair_diagnostics(records)}
        _require(methods_public["empty"]["paired_switch_em"] == methods_public["empty"]["changed_rate"] == 0.,
                 "One empty-memory prediction cannot correctly switch between different answers")
        para = pair_summary(paraphrases, ("A", "P"), invariant=True)
        assert_equal(stored.get("paraphrase"), para, phase+"/paraphrase")
        teacher_para = pair_summary(teacher_paraphrases, ("A", "P"), invariant=True) if include_teacher else None
        assert_equal(stored.get("teacher_paraphrase"), teacher_para, phase+"/teacher_paraphrase")
        unrelated_summary = pair_summary(unrelated, invariant=True) if unrelated else None
        assert_equal(stored.get("unrelated"), unrelated_summary, phase+"/unrelated")
        _require(stored.get("reconstruction") == bank_names[phase_index], "Reconstruction phase path mismatch")
        public_phases[phase] = dict(count=CONFIRMATION_FACTS, methods=methods_public, paraphrase=para,
                                    teacher_paraphrase=teacher_para, unrelated=unrelated_summary,
                                    base_bank_sha256=stored["bank_A_hash"],
                                    reconstruction_file_sha256=digest_file(directory/_safe_relative(bank_names[phase_index])))
        private["phases"][phase] = phase_private
    _require(set(indexed) == expected_keys, "Extra/unexpected prediction records")
    for name, value in (("generation_calls", len(predictions)),
                        ("generation_tokens", sum(row["generation_tokens"] for row in predictions)),
                        ("answer_scoring_tokens", sum(row["answer_scoring_tokens"] for row in predictions))):
        _require(metrics.get(name) == value, "Evaluation total mismatch: "+name)
    evidence = ("manifest.json", "metrics.json", "predictions.jsonl", "writes.jsonl", *bank_names)
    provenance.update(evidence_sha256={name: digest_file(directory/_safe_relative(name)) for name in evidence},
                      fact_label_sha256=object_digest(sorted(private["labels"].items())))
    return dict(configuration=config, provenance=provenance, phases=public_phases,
                evaluation_cost={name: metrics[name] for name in ("generation_calls", "generation_tokens", "answer_scoring_tokens")}), private


def _percentile(values, quantile):
    position = (len(values)-1)*quantile
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def paired_cluster_bootstrap(first, second, *, samples=2000, seed=123):
    """Resample target facts with all their paired phase measurements together."""
    require_int(samples, "bootstrap samples", 1)
    _require(bool(first) and first.keys() == second.keys(), "Paired fact sets differ or are empty")
    counts = {len(value) for value in first.values()}
    _require(len(counts) == 1 and next(iter(counts)) > 0, "Unbalanced fact phase counts")
    differences = []
    for identity in sorted(first):
        a, b = first[identity], second[identity]
        _require(len(a) == len(b), "Paired phase counts differ")
        for value in [*a, *b]:
            require_finite(value, "bootstrap score")
        differences.append(sum(x-y for x, y in zip(a, b))/len(a))
    rng, count = random.Random(seed), len(differences)
    distribution = sorted(sum(rng.choices(differences, k=count))/count for _ in range(samples))
    return dict(n_paired_facts=count, phases_per_fact=next(iter(counts)), delta=sum(differences)/count,
                ci95=[_percentile(distribution, .025), _percentile(distribution, .975)],
                bootstrap_samples=samples, bootstrap_seed=seed,
                unit="target fact; all phases travel together", scope="Conditional fact uncertainty, excluding training-seed and expression-family uncertainty")


def _metric_vectors(private, phases, field, method="real"):
    ids = private["phases"][phases[0]].keys()
    return {identity: [private["phases"][phase][identity][method][field] if field in ("paired_switch", "swapped", "wrong_wrong")
                       else private["phases"][phase][identity][field] for phase in phases] for identity in ids}


def summarize(initial, warm, trainings, evaluations, *, samples=2000, seed=123):
    _require(set(trainings) == set(evaluations) == set(ARMS), "All four training/evaluation arms are required")
    paths = [Path(initial).resolve(), Path(warm).resolve(), *(Path(p).resolve() for p in trainings.values()), *(Path(p).resolve() for p in evaluations.values())]
    _require(len(paths) == len(set(paths)), "Run directories must be distinct")
    warmed = audit_training(warm)
    _require(warmed["configuration"]["method"] == "base" and warmed["configuration"]["start_step"] == 0, "Common warm-up must be base from step zero")
    initial_public, initial_private = audit_evaluation(initial, include_teacher=True)
    trained = {name: audit_training(trainings[name]) for name in ARMS}
    evaluated = {name: audit_evaluation(evaluations[name], include_teacher=False) for name in ARMS}
    all_evidence = [warmed, initial_public, *trained.values(), *(pair[0] for pair in evaluated.values())]
    for item in all_evidence:
        for field in ("runtime_source_sha256", "teacher_parameter_sha256", "model_revision"):
            _require(item["provenance"][field] == warmed["provenance"][field], "Runtime/backbone provenance differs: "+field)
        _require(item["configuration"]["seed"] == warmed["configuration"]["seed"], "Run seed mismatch")
    _require(initial_public["provenance"]["checkpoint_sha256"] == warmed["provenance"]["checkpoint_sha256"]
             and initial_public["provenance"]["module_sha256_before"] == warmed["provenance"]["module_sha256_before"],
             "Initial must evaluate the pre-warm anchored checkpoint")
    _require(initial_public["provenance"]["cache_sha256"] != warmed["provenance"]["cache_sha256"], "Train and confirmation must be separate caches")
    baseline_training = trained["base"]
    for name in ARMS:
        training, (evaluation, private) = trained[name], evaluated[name]
        _require(training["configuration"]["method"] == name, "Training arm name/method mismatch")
        _require(training["training"]["start_step"] == warmed["training"]["step"], "Continuation does not start at the common warm step")
        _require(training["configuration"]["batch_size"] == warmed["configuration"]["batch_size"], "Warm/continuation target batch size differs")
        _require(training["provenance"]["checkpoint_sha256"] == warmed["provenance"]["final_checkpoint_sha256"]
                 and training["provenance"]["module_sha256_before"] == warmed["provenance"]["module_sha256_after"],
                 "Continuation did not initialize from common warm checkpoint")
        _require(training["provenance"]["cache_sha256"] == warmed["provenance"]["cache_sha256"], "Training cache hashes differ")
        _require(evaluation["provenance"]["checkpoint_sha256"] == training["provenance"]["final_checkpoint_sha256"]
                 and evaluation["provenance"]["module_sha256_before"] == training["provenance"]["module_sha256_after"],
                 "Evaluation does not use the corresponding final training checkpoint")
        _require(evaluation["provenance"]["cache_sha256"] == initial_public["provenance"]["cache_sha256"], "Confirmation cache hashes differ")
        _require(private["labels"] == initial_private["labels"] and private["questions"] == initial_private["questions"], "Paired arm fact/label/question sets differ")
        for field in ("start_step", "step", "updates", "target_pair_exposures", "episode_schedule_sha256"):
            _require(training["training"][field] == baseline_training["training"][field], "Continuation schedules/budgets differ: "+field)
    all_runs = {"initial": (initial_public, initial_private), **evaluated}
    comparisons = {}
    for name, (public, private) in all_runs.items():
        public["method"] = name
        public["evaluation_configuration"] = public.pop("configuration")
        if name == "initial":
            public["configuration"] = dict(method="initial", evaluate_only=True, seed=warmed["configuration"]["seed"])
            public["training"] = dict(updates=0, start_step=0, step=0, target_pair_exposures=0,
                                      token_totals={field: 0 for field in TOKEN_FIELDS}, elapsed_seconds=0., processed_input_tokens_total=0)
        else:
            public["configuration"] = trained[name]["configuration"]
            public["training"] = trained[name]["training"]
            public["training_provenance"] = trained[name]["provenance"]
        for phase in PHASES:
            teacher = initial_private["phases"][phase]
            members = private["phases"][phase]
            eligible_ab = [identity for identity, row in teacher.items() if row["teacher"]["paired_switch"]]
            eligible_ap = [identity for identity, row in teacher.items() if row["teacher_paraphrase_joint"]]
            eligible_abp = [identity for identity, row in teacher.items() if row["teacher_abp_joint"]]
            numerator = sum(members[identity]["real"]["paired_switch"] for identity in eligible_ab)
            numerator_p = sum(members[identity]["paraphrase_joint"] for identity in eligible_ap)
            public["phases"][phase]["teacher_eligibility"] = dict(
                all_facts=len(members), teacher_ab_eligible_count=len(eligible_ab), teacher_ap_eligible_count=len(eligible_ap),
                teacher_abp_eligible_count=len(eligible_abp), teacher_ab_eligible_rate=len(eligible_ab)/len(members),
                student_real_switch_all_facts=public["phases"][phase]["methods"]["real"]["paired_switch_em"],
                student_real_switch_on_teacher_ab_eligible=numerator/len(eligible_ab) if eligible_ab else None,
                student_real_switch_on_teacher_ab_eligible_correct=numerator,
                student_paraphrase_joint_all_facts=public["phases"][phase]["paraphrase"]["joint_em"],
                student_paraphrase_joint_on_teacher_ap_eligible=numerator_p/len(eligible_ap) if eligible_ap else None,
                student_paraphrase_joint_on_teacher_ap_eligible_correct=numerator_p,
                definition="Teacher free-generation correctness; distinct from training margin/hidden-valid masks. All-fact scores remain primary.",
            )
            public["phases"][phase]["paired"] = dict(
                real_minus_shuffled_switch=paired_cluster_bootstrap(
                    _metric_vectors(private, (phase,), "paired_switch"), _metric_vectors(private, (phase,), "paired_switch", "shuffled"), samples=samples, seed=seed),
                real_minus_initial_switch=paired_cluster_bootstrap(
                    _metric_vectors(private, (phase,), "paired_switch"), _metric_vectors(initial_private, (phase,), "paired_switch"), samples=samples, seed=seed),
            )
        public["clustered_real_minus_shuffled_switch"] = paired_cluster_bootstrap(
            _metric_vectors(private, PHASES, "paired_switch"), _metric_vectors(private, PHASES, "paired_switch", "shuffled"), samples=samples, seed=seed)
        if name != "initial":
            comparison = {}
            for reference in ("initial", "base"):
                comparison["minus_"+reference] = {
                    field: paired_cluster_bootstrap(_metric_vectors(private, PHASES, field),
                                                    _metric_vectors(all_runs[reference][1], PHASES, field), samples=samples, seed=seed)
                    for field in ("paired_switch", "paraphrase_joint", "swapped", "wrong_wrong")
                }
                comparison["minus_"+reference]["unrelated_joint"] = paired_cluster_bootstrap(
                    _metric_vectors(private, UNRELATED_PHASES, "unrelated_joint"),
                    _metric_vectors(all_runs[reference][1], UNRELATED_PHASES, "unrelated_joint"), samples=samples, seed=seed)
            comparisons[name] = comparison
    cost_stages = [warmed, *trained.values()]
    return dict(protocol="counterfactual-public-audit-v1", complete=True,
                audited_at=datetime.now(timezone.utc).isoformat(), audit_numeric_tolerance=1e-7,
                evaluation=dict(facts=CONFIRMATION_FACTS, phases=list(PHASES), split="fresh confirmation",
                                fact_label_sha256=initial_public["provenance"]["fact_label_sha256"],
                                raw_prediction_count=sum(run[0]["evaluation_cost"]["generation_calls"] for run in all_runs.values())),
                warm=warmed, runs=[all_runs[name][0] for name in EVAL_ARMS], comparisons=comparisons,
                teacher_acceptance={phase: dict(counterfactual=initial_public["phases"][phase]["methods"]["teacher"],
                                                paraphrase=initial_public["phases"][phase]["teacher_paraphrase"]) for phase in PHASES},
                comparability=dict(common_initial_checkpoint_sha256=warmed["provenance"]["checkpoint_sha256"],
                                   common_warm_checkpoint_sha256=warmed["provenance"]["final_checkpoint_sha256"],
                                   train_cache_sha256=warmed["provenance"]["cache_sha256"],
                                   confirmation_cache_sha256=initial_public["provenance"]["cache_sha256"],
                                   continuation_episode_schedule_sha256=baseline_training["training"]["episode_schedule_sha256"],
                                   warm_updates=warmed["training"]["updates"], continuation_updates=baseline_training["training"]["updates"],
                                   source_sha256=warmed["provenance"]["runtime_source_sha256"], frozen_backbone_verified=True),
                physical_training_cost=dict(stages=5, shared_warm_counted_once=True,
                    target_pair_exposures=sum(item["training"]["target_pair_exposures"] for item in cost_stages),
                    processed_input_tokens_total=sum(item["training"]["processed_input_tokens_total"] for item in cost_stages),
                    token_totals={field: sum(item["training"]["token_totals"][field] for item in cost_stages) for field in TOKEN_FIELDS},
                    elapsed_stage_seconds_sum=sum(item["training"]["elapsed_seconds"] for item in cost_stages)),
                limitations=[
                    "64 fresh entities, a finite 16-word answer vocabulary, two reserved structure families and one training seed; broader semantic generalization is untested.",
                    "A/B must both be correct for paired_switch_em. Changed predictions alone, swapped wrong answers, and wrong/wrong changes are not successes.",
                    "Teacher eligibility conditions are reported alongside all-fact scores; no difficult teacher-ineligible facts are silently discarded.",
                    "A/B/P and all four phases share an entity. Bootstrap samples target-fact clusters, not individual world/phase rows; intervals exclude seed/family uncertainty.",
                    "Unrelated reads query the next entity under one target intervention; this limited preservation diagnostic is not a general forgetting benchmark.",
                    "The actual writer may change the target key and value together. These are single-record observation interventions, not fixed-key value-only interventions.",
                    "Oracle forces a value and changes the readout distribution; it is a diagnostic rather than a mathematical upper bound.",
                    "Equal continuation updates/target schedules do not imply equal compute; mixed rollouts add work. Token totals are unpadded positions, not FLOPs.",
                    "Physical training cost counts the common warm-up once and excludes training the inherited anchored initialization and cache preparation.",
                    "This audit verifies prediction/write records, reconstructs CPU bank and shuffle hashes, and checks source/checkpoint/cache lineage; it does not replay model inference.",
                ])


def report(summary):
    labels = {phase: label for phase, label in zip(PHASES, ("规范/规范", "规范/新问句", "新支持/规范", "新支持/新问句"))}
    percent = lambda value: "—" if value is None else f"{100*value:.1f}%"
    def interval(value):
        return f"{100*value['delta']:+.1f} [{100*value['ci95'][0]:+.1f}, {100*value['ci95'][1]:+.1f}]"
    lines = ["# 反事实记忆蒸馏：严格审计结果", "",
             "A/B为同一问题、不同目标事实；P只改目标观察的表达。主指标要求A、B两个世界同时正确，不能用输出发生变化代替成功。", "",
             f"共同warm {summary['comparability']['warm_updates']}步；四组各续训 {summary['comparability']['continuation_updates']}步，target/episode schedule SHA一致。Initial是warm前的原始anchored checkpoint。", "",
             "## 四条件的真实读出", "", "单元格：A/B成对正确率 / A与P同时正确率。每列64个事实。", "",
             "| 方法 | "+" | ".join(labels.values())+" |", "|---|"+"---|"*4]
    for run in summary["runs"]:
        cells = [percent(run["phases"][phase]["methods"]["real"]["paired_switch_em"])+" / "+percent(run["phases"][phase]["paraphrase"]["joint_em"]) for phase in PHASES]
        lines.append("| "+run["method"]+" | "+" | ".join(cells)+" |")
    lines += ["", "## 检索、错误切换与打乱对照", "", "R@1为A/B两世界；swapped指A答B且B答A。wrong/wrong包含两世界都错误的情况。Δ为real−shuffled paired switch，百分点及95%按事实成对bootstrap区间。", "",
              "| 方法 | 条件 | A/B R@1 | swapped | wrong/wrong | 变化但未同时正确 | Δreal−shuffled (pp, CI) |", "|---|---|---|---|---|---|---|"]
    for run in summary["runs"]:
        for phase in PHASES:
            row = run["phases"][phase]; score = row["methods"]["real"]
            lines.append(f"| {run['method']} | {labels[phase]} | {percent(score['a_recall_at_1'])}/{percent(score['b_recall_at_1'])} | {percent(score['swapped_answer_rate'])} | {percent(score['wrong_wrong_rate'])} | {percent(score['changed_but_not_both_correct_rate'])} | {interval(row['paired']['real_minus_shuffled_switch'])} |")
    lines += ["", "## Teacher可用性", "", "所有事实仍在主表分母中。条件分数仅作诊断；teacher正确性是自由生成结果，不等同训练loss的margin/hidden门控。", "",
              "| 条件 | Teacher A/B joint | Teacher A/P joint | Teacher A/B/P全对数量 |", "|---|---|---|---|"]
    for phase in PHASES:
        teacher = summary["teacher_acceptance"][phase]
        eligible = summary["runs"][0]["phases"][phase]["teacher_eligibility"]
        lines.append(f"| {labels[phase]} | {percent(teacher['counterfactual']['paired_switch_em'])} | {percent(teacher['paraphrase']['joint_em'])} | {eligible['teacher_abp_eligible_count']}/{eligible['all_facts']} |")
    lines += ["", "| 方法 | 条件 | 全部事实A/B joint | Teacher A/B均正确子集 | 子集分母 | A/P joint(全部/teacher合格) |", "|---|---|---|---|---|---|"]
    for run in summary["runs"]:
        for phase in PHASES:
            row = run["phases"][phase]["teacher_eligibility"]
            lines.append(f"| {run['method']} | {labels[phase]} | {percent(row['student_real_switch_all_facts'])} | {percent(row['student_real_switch_on_teacher_ab_eligible'])} | {row['teacher_ab_eligible_count']} | {percent(row['student_paraphrase_joint_all_facts'])}/{percent(row['student_paraphrase_joint_on_teacher_ap_eligible'])} |")
    lines += ["", "## 跨条件成对比较", "", "每个实体的四个条件一起重采样，不能把256行当作256个独立事实。以下为相对base的百分点差及95%区间。", "",
              "| 方法 | A/B joint | A/P joint | swapped | wrong/wrong | 无关实体joint(两个条件) |", "|---|---|---|---|---|---|"]
    for method in ARMS:
        pair = summary["comparisons"][method]["minus_base"]
        lines.append("| "+method+" | "+" | ".join(interval(pair[key]) for key in ("paired_switch", "paraphrase_joint", "swapped", "wrong_wrong", "unrelated_joint"))+" |")
    lines += ["", "## 实际训练成本与teacher训练覆盖率", "",
              "| 阶段 | 更新 | Pair exposures | sampled tokens | 总input tokens | 训练秒数 | Teacher首token A/B全对 | Behavior有效 | Hidden有效 |", "|---|---|---|---|---|---|---|---|---|"]
    stages = [("共同warm", summary["warm"]["training"])]+[(run["method"], run["training"]) for run in summary["runs"][1:]]
    for name, row in stages:
        rates = row["diagnostic_pair_rates"]
        lines.append(f"| {name} | {row['updates']} | {row['target_pair_exposures']} | {row['token_totals']['sampled_tokens']} | {row['processed_input_tokens_total']} | {row['elapsed_seconds']:.1f} | {percent(rates['teacher_first_token_joint_correct'])} | {percent(rates['behavior_valid_pairs'])} | {percent(rates['hidden_valid_pairs'])} |")
    lines += ["", "共同warm在物理成本总量中只计一次；每个续训方法的完整训练还需加上该共同warm。初始checkpoint的历史训练和数据缓存准备成本不计入此次成本。", "",
              "## 范围与限制", ""]+["- "+item for item in summary["limitations"]]
    lines += ["", "summary.json包含所有real/shuffled/oracle/empty、teacher、paraphrase、unrelated指标、成对区间、日志成本与SHA。原始问题、观察、token ID、实体ID和本地路径不进入公开输出。", ""]
    return "\n".join(lines)


def _assignments(values, kind):
    result = {}
    for value in values:
        _require("=" in value, f"{kind} requires method=directory")
        name, directory = value.split("=", 1)
        _require(name in ARMS and directory and name not in result, f"Unknown/duplicate {kind} arm")
        result[name] = Path(directory)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--training", action="append", required=True, metavar="METHOD=DIR")
    parser.add_argument("--evaluation", action="append", required=True, metavar="METHOD=DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    args = parser.parse_args(argv)
    summary = summarize(args.initial, args.warm, _assignments(args.training, "training"),
                        _assignments(args.evaluation, "evaluation"), samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, contents in (("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n"),
                           ("report.md", report(summary))):
        target = args.output/name
        partial = target.with_suffix(target.suffix+".partial")
        partial.write_text(contents, encoding="utf-8")
        partial.replace(target)
    print(json.dumps(dict(complete=True, arms=len(summary["runs"]), facts=summary["evaluation"]["facts"],
                         predictions=summary["evaluation"]["raw_prediction_count"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
