"""Audit completed context-distillation suites and publish aggregate evidence.

Requires initial plus all five training arms. Raw prompts, token IDs, fact IDs,
and local paths never enter the public summary. Inputs are strictly read-only.
Usage: python scripts/summarize_context_distillation.py --suite SUITE [...] --output DIR
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vera_mem.metrics import exact_match


ARMS = ("initial", "ce", "off_kd", "off_kd_hidden", "on_kd", "on_kd_hidden")
METHODS = ("real", "oracle", "shuffled", "empty")
PHASES = tuple(f"{s}_support/{q}_query" for s in ("canonical", "heldout")
               for q in ("canonical", "heldout"))
ALIGNMENT_FIELDS = (
    "reverse_kl", "hidden_cosine", "no_memory_reverse_kl", "no_memory_hidden_cosine",
    "first_token_reverse_kl", "first_token_hidden_cosine",
    "first_token_no_memory_reverse_kl", "first_token_no_memory_hidden_cosine",
)
TOKEN_FIELDS = (
    "main_target_tokens", "teacher_target_tokens", "sampled_tokens", "student_input_tokens",
    "teacher_input_tokens", "rollout_input_tokens", "replay_target_tokens", "replay_input_tokens",
)
CONFIG_FIELDS = (
    "method", "updates", "batch_size", "eval_size", "max_new_tokens", "sampling_temperature",
    "seed", "evaluate_only", "skip_teacher_eval", "kl_direction", "kl_temperature", "hidden_weight",
    "canonical_replay_frequency", "canonical_replay_weight", "all_view_address_weight",
    "actual_prefix_address_weight", "wrong_prefix_address_policy",
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
TOLERANCE = 1e-7
ALIGNMENT_AGGREGATION = "mean token metric per fact, then mean across facts; gold+EOS prefixes"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank evidence record at line {number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Evidence records must be JSON objects")
        rows.append(row)
    return rows


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require_hash(value, name):
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"Invalid SHA256: {name}")
    return value


def require_int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid integer: {name}")
    return value


def require_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Nonfinite/non-numeric value: {name}")
    return value


def assert_equal(actual, expected, name):
    """Strict keys/types, with only finite numeric values allowing 1e-7 error."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Metric keys mismatch: {name}")
        for key in expected:
            assert_equal(actual[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Metric list mismatch: {name}")
        for i, value in enumerate(expected):
            assert_equal(actual[i], value, f"{name}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require_finite(actual, name)
        if abs(actual - expected) > TOLERANCE:
            raise ValueError(f"Metric value mismatch: {name}")
    elif actual != expected or type(actual) is not type(expected):
        raise ValueError(f"Metric value mismatch: {name}")


def index_records(rows, expected_answers, name):
    indexed = {}
    for row in rows:
        identity = row.get("id")
        if identity in indexed:
            raise ValueError(f"Duplicate fact record: {name}")
        if identity not in expected_answers:
            raise ValueError(f"Unknown fact ID: {name}")
        if "answer" in row and row["answer"] != expected_answers[identity]:
            raise ValueError(f"Fact answer mismatch: {name}")
        indexed[identity] = row
    if indexed.keys() != expected_answers.keys():
        raise ValueError(f"Missing fact records: {name}")
    return indexed


def prediction_statistics(rows):
    """Reproduce the runner's full stats schema after per-record validation."""
    if not rows:
        raise ValueError("Empty prediction group")
    for row in rows:
        if not isinstance(row.get("prediction"), str) or not isinstance(row.get("answer"), str):
            raise ValueError("Prediction/answer must be strings")
        actual_em = exact_match(row["prediction"], row["answer"])
        if type(row.get("em")) not in (int, float) or row["em"] not in (0, 1) or row["em"] != actual_em:
            raise ValueError("Stored exact_match disagrees with prediction text")
        require_int(row.get("tokens"), "answer tokens", 1)
        require_finite(row.get("nll_sum"), "answer nll_sum")
        if not isinstance(row.get("expected_in_bank"), bool):
            raise ValueError("Missing expected_in_bank boolean")
    result = dict(count=len(rows), em=sum(r["em"] for r in rows)/len(rows),
                  answer_token_nll=sum(r["nll_sum"] for r in rows)/sum(r["tokens"] for r in rows))
    routed = [r for r in rows if r["expected_in_bank"] and r["method"] in ("real", "shuffled")]
    if routed:
        for row in routed:
            selected = row.get("selected_ids")
            if not isinstance(selected, list) or len(selected) > 4 or len(set(selected)) != len(selected):
                raise ValueError("Invalid sparse top-4 selected_ids")
        result.update(recall_at_1=sum(r["selected_ids"][:1] == [r["id"]] for r in routed)/len(routed),
                      recall_at_4=sum(r["id"] in r["selected_ids"] for r in routed)/len(routed))
    decoded = [r for r in routed if r.get("decode_correct_residency") is not None]
    if decoded:
        for row in decoded:
            count = require_int(row.get("decode_query_count"), "decode query count", 1)
            hits = require_int(row.get("decode_correct_hits"), "decode correct hits")
            if hits > count:
                raise ValueError("Decode hits exceed queries")
            assert_equal(row["decode_correct_residency"], hits/count, "decode residency")
        count = sum(r["decode_query_count"] for r in decoded)
        result.update(decode_correct_residency=sum(r["decode_correct_hits"] for r in decoded)/count,
                      decode_query_count=count, decode_examples=len(decoded))
    traced = [r for r in rows if r.get("switch_count") is not None]
    if traced:
        for row in traced:
            switches = require_int(row["switch_count"], "switch count")
            transitions = require_int(row.get("retrieval_transition_count"), "retrieval transition count")
            if switches > transitions:
                raise ValueError("Switch count exceeds transitions")
        switches = sum(r["switch_count"] for r in traced)
        transitions = sum(r["retrieval_transition_count"] for r in traced)
        result.update(switch_count_mean=switches/len(traced), switch_count_total=switches,
                      retrieval_transition_count=transitions,
                      top1_switch_rate=switches/transitions if transitions else None)
    return result


def alignment_statistics(rows):
    for row in rows:
        count = require_int(row.get("tokens"), "alignment tokens", 1)
        if not isinstance(row.get("continuation_token_ids"), list) or len(row["continuation_token_ids"]) != count:
            raise ValueError("Alignment continuation/token count mismatch")
        for field in ALIGNMENT_FIELDS:
            require_finite(row.get(field), field)
    return dict(count=len(rows), tokens=sum(row["tokens"] for row in rows),
                **{field: sum(row[field] for row in rows)/len(rows) for field in ALIGNMENT_FIELDS},
                aggregation=ALIGNMENT_AGGREGATION)


def _percentile(ordered, quantile):
    position = (len(ordered)-1)*quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


def bootstrap_difference(first, second, *, samples=5000, seed=123):
    """Bootstrap fact means; multiple phases are clustered within each fact.

    Each mapping is ``fact_id -> list[EM by phase]``. Both fact and phase sets
    must match. A fact is drawn once per bootstrap sample and all its phases
    travel together; no view/phase is treated as an independent observation.
    """
    require_int(samples, "bootstrap samples", 1)
    if not first or first.keys() != second.keys():
        raise ValueError("Paired fact sets differ or are empty")
    phase_counts = {len(first[identity]) for identity in first}
    if len(phase_counts) != 1 or next(iter(phase_counts)) == 0:
        raise ValueError("Paired phase counts differ")
    differences = []
    for identity in sorted(first):
        if len(first[identity]) != len(second[identity]):
            raise ValueError("Paired phase counts differ")
        for value in first[identity] + second[identity]:
            if value not in (0, 1):
                raise ValueError("Bootstrap inputs must be binary EM")
        differences.append(sum(a-b for a, b in zip(first[identity], second[identity]))/len(first[identity]))
    rng = random.Random(seed)
    count = len(differences)
    distribution = sorted(sum(rng.choices(differences, k=count))/count for _ in range(samples))
    return dict(n_paired_facts=count, phases_per_fact=next(iter(phase_counts)),
                delta_em=sum(differences)/count,
                ci95=[_percentile(distribution, .025), _percentile(distribution, .975)],
                bootstrap_samples=samples, bootstrap_seed=seed,
                unit="fact; all phases for one fact resampled together",
                scope="Conditional fact-sampling uncertainty, excludes training-seed and template-family uncertainty")


def audit_training(directory, config, manifest, initial):
    if initial:
        if manifest.get("training") is not None or manifest.get("selected_step") != 0:
            raise ValueError("Initial arm must have zero training updates")
        if any((directory/name).exists() for name in ("training.jsonl", "training_status.json", "rollouts.jsonl")):
            raise ValueError("Initial arm unexpectedly contains training evidence")
        return dict(step=0, selected_step=0, target_exposures=0, canonical_replay_exposures=0,
                    token_totals={field: 0 for field in TOKEN_FIELDS}, elapsed_seconds=0.,
                    processed_input_tokens_total=0, selection="Frozen initialization; no updates")
    training = manifest.get("training")
    if not isinstance(training, dict) or training.get("finished") is not True:
        raise ValueError("Training has not finished")
    assert_equal(read_json(directory/"training_status.json"), training, "training status/manifest")
    for field, filename in (("training_log_sha256", "training.jsonl"), ("rollout_sha256", "rollouts.jsonl")):
        if require_hash(training.get(field), field) != digest_file(directory/filename):
            raise ValueError(f"Training evidence hash mismatch: {field}")
    rows = read_jsonl(directory/"training.jsonl")
    updates, batch = config["updates"], config["batch_size"]
    if len(rows) != updates or [r.get("step") for r in rows] != list(range(1, updates+1)):
        raise ValueError("Missing/duplicate/nonsequential training steps")
    targets_digest, episode_digest = hashlib.sha256(), hashlib.sha256()
    totals = {field: 0 for field in TOKEN_FIELDS}
    for row in rows:
        if row.get("method") != config["method"]:
            raise ValueError("Training method mismatch")
        ids, episode = row.get("target_ids"), row.get("episode_ids")
        if (not isinstance(ids, list) or len(ids) != batch or len(set(ids)) != batch
                or not isinstance(episode, list) or len(set(episode)) != len(episode)
                or not set(ids).issubset(episode)):
            raise ValueError("Invalid target/episode facts")
        query_views, support_views = row.get("query_view_indices"), row.get("support_view_indices")
        for views, limit in ((query_views, 8), (support_views, 4)):
            if not isinstance(views, list) or len(views) != len(episode) or any(
                type(view) is not int or not 0 <= view < limit for view in views
            ):
                raise ValueError("Invalid episode style schedule")
        if row.get("target_to_episode") != [episode.index(identity) for identity in ids]:
            raise ValueError("Target/episode mapping mismatch")
        targets_digest.update(json.dumps(ids, separators=(",", ":")).encode())
        record = dict(ids=episode, qviews=query_views, sviews=support_views, targets=ids)
        episode_digest.update(json.dumps(record, separators=(",", ":")).encode())
        for field in TOKEN_FIELDS:
            totals[field] += require_int(row.get(field), field)
    expected = dict(step=updates, selected_step=updates, target_exposures=updates*batch,
                    canonical_replay_exposures=(updates//4)*batch,
                    target_exposure_sha256=targets_digest.hexdigest(),
                    episode_schedule_sha256=episode_digest.hexdigest(), token_totals=totals)
    for field, value in expected.items():
        assert_equal(training.get(field), value, f"training.{field}")
    if manifest.get("selected_step") != updates:
        raise ValueError("Final checkpoint update mismatch")
    elapsed = require_finite(training.get("elapsed_seconds"), "training elapsed seconds")
    if elapsed < 0:
        raise ValueError("Negative elapsed training time")
    if config["method"] == "ce" and totals["teacher_target_tokens"] != 0:
        raise ValueError("CE arm unexpectedly uses teacher targets")
    if not config["method"].startswith("on_") and totals["sampled_tokens"] != 0:
        raise ValueError("Off-policy/CE arm unexpectedly samples student tokens")
    if config["method"].startswith("on_") and totals["sampled_tokens"] != totals["main_target_tokens"]:
        raise ValueError("On-policy sampled and main target counts differ")
    return dict(**expected, elapsed_seconds=elapsed,
                rollout_sha256=training["rollout_sha256"], training_log_sha256=training["training_log_sha256"],
                processed_input_tokens_total=sum(totals[field] for field in (
                    "student_input_tokens", "teacher_input_tokens", "rollout_input_tokens", "replay_input_tokens")),
                selection=training.get("selection", "fixed final update"))


def audit_run(directory, name, suite):
    manifest, config, metrics = [read_json(directory/filename) for filename in ("manifest.json", "config.json", "metrics.json")]
    if manifest.get("complete") is not True or manifest.get("finished") is not True:
        raise ValueError("Run manifest is incomplete")
    if manifest.get("teacher_parameters_unchanged") is not True:
        raise ValueError("Frozen teacher verification missing")
    before = require_hash(manifest.get("teacher_parameter_sha256_before"), "teacher before")
    if manifest.get("teacher_parameter_sha256_after") != before:
        raise ValueError("Teacher parameter hashes differ")
    initial = name == "initial"
    if config.get("evaluate_only") is not initial or manifest.get("evaluate_only") is not initial:
        raise ValueError("Initial/training mode mismatch")
    method = "ce" if initial else name
    if config.get("method") != method or manifest.get("method") != method:
        raise ValueError("Arm/configuration method mismatch")
    for field in ("updates", "batch_size", "eval_size", "max_new_tokens"):
        require_int(config.get(field), field, 1)
    require_int(config.get("seed"), "seed")
    if metrics.get("evaluation_split") != "dev" or manifest.get("cache_splits_used") != ["train", "dev"]:
        raise ValueError("Only declared offline train/dev evidence is permitted")
    if manifest.get("confirmation_used") is not False or manifest.get("dev_selection") is not False:
        raise ValueError("Unexpected confirmation use or development checkpoint selection")
    for field in ("shared_parameters_unchanged", "vdb_unchanged"):
        if metrics.get(field) is not True:
            raise ValueError(f"Missing evaluation immutability: {field}")
    if metrics.get("online_gradient_steps") != 0:
        raise ValueError("Evaluation must not update shared parameters")
    if metrics.get("eval_facts") != config["eval_size"] or set(metrics.get("conditions", {})) != set(PHASES):
        raise ValueError("Evaluation size or four-condition coverage mismatch")
    source = manifest.get("source_files_sha256")
    if not isinstance(source, dict) or not source:
        raise ValueError("Runtime source hashes missing")
    for filename, digest in source.items():
        if Path(filename).name != filename or require_hash(digest, "runtime source") != suite["source_files_sha256"].get("src/vera_mem/"+filename):
            raise ValueError("Runtime/suite source hash mismatch")
    hashes = {field: require_hash(manifest.get(field), field) for field in (
        "cache_sha256", "checkpoint_sha256", "model_manifest_sha256", "data_fingerprint",
        "module_sha256_before", "module_sha256_after",
    )}
    for field in ("cache_sha256", "checkpoint_sha256"):
        if hashes[field] != suite.get(field):
            raise ValueError(f"Run/suite provenance mismatch: {field}")
    if metrics.get("module_sha256") != hashes["module_sha256_after"]:
        raise ValueError("Evaluation/module hash mismatch")
    examples = read_jsonl(directory/"dev.jsonl")[:config["eval_size"]]
    answers = {row["id"]: row["answer"] for row in examples}
    if len(answers) != config["eval_size"]:
        raise ValueError("Missing/duplicate development facts")
    if any(not isinstance(identity, str) or not identity or not isinstance(answer, str) for identity, answer in answers.items()):
        raise ValueError("Invalid development fact labels")
    groups = defaultdict(list)
    for row in read_jsonl(directory/"predictions.jsonl"):
        if row.get("phase") not in PHASES or row.get("method") not in METHODS:
            raise ValueError("Unexpected prediction phase/method")
        if row.get("expected_in_bank") is not True:
            raise ValueError("All four-condition facts must be present in the bank")
        if any(identity not in answers for identity in row.get("selected_ids", [])):
            raise ValueError("Retrieved ID is outside evaluation bank")
        groups[row["phase"], row["method"]].append(row)
    alignment_groups = defaultdict(list)
    for row in read_jsonl(directory/"alignment_predictions.jsonl"):
        if row.get("phase") not in PHASES:
            raise ValueError("Unexpected alignment phase")
        alignment_groups[row["phase"]].append(row)
    teacher_groups = defaultdict(list)
    teacher_file = directory/"teacher_predictions.jsonl"
    if teacher_file.exists():
        for row in read_jsonl(teacher_file):
            if row.get("phase") not in PHASES or row.get("method") != "teacher" or row.get("expected_in_bank") is not False:
                raise ValueError("Unexpected teacher prediction phase/method/bank flag")
            teacher_groups[row["phase"]].append(row)
    if initial and config.get("skip_teacher_eval") is not False:
        raise ValueError("Initial arm must include teacher acceptance evaluation")
    conditions = {}
    private = dict(answers=answers, student={}, teacher={})
    for phase in PHASES:
        stored = metrics["conditions"][phase]
        if set(stored.get("student", {})) != set(METHODS):
            raise ValueError("Missing student evaluation method")
        student = {}
        for evaluation_method in METHODS:
            rows = groups[phase, evaluation_method]
            indexed = index_records(rows, answers, f"{phase}/{evaluation_method}")
            student[evaluation_method] = prediction_statistics(rows)
            assert_equal(stored["student"][evaluation_method], student[evaluation_method], f"student {phase}/{evaluation_method}")
            private["student"][phase, evaluation_method] = indexed
        alignment_rows = alignment_groups[phase]
        index_records(alignment_rows, answers, f"alignment {phase}")
        alignment = alignment_statistics(alignment_rows)
        assert_equal(stored.get("alignment"), alignment, f"alignment {phase}")
        teacher = None
        if not config["skip_teacher_eval"]:
            rows = teacher_groups[phase]
            private["teacher"][phase] = index_records(rows, answers, f"teacher {phase}")
            teacher = prediction_statistics(rows)
        elif teacher_groups:
            raise ValueError("Teacher predictions exist despite skip_teacher_eval")
        assert_equal(stored.get("teacher"), teacher, f"teacher {phase}")
        if stored.get("records") != len(answers):
            raise ValueError("Bank record count mismatch")
        conditions[phase] = dict(student=student, alignment=alignment, teacher=teacher,
                                 records=stored["records"], numeric_vdb_bytes=require_int(stored.get("numeric_vdb_bytes"), "VDB bytes"),
                                 vdb_sha256=require_hash(stored.get("vdb_sha256"), "VDB hash"))
    training = audit_training(directory, config, manifest, initial)
    evidence_names = ["manifest.json", "config.json", "metrics.json", "dev.jsonl", "predictions.jsonl", "alignment_predictions.jsonl"]
    if teacher_file.exists():
        evidence_names.append("teacher_predictions.jsonl")
    if not initial:
        evidence_names += ["training.jsonl", "training_status.json", "rollouts.jsonl", "last.pt"]
    provenance = dict(**hashes, runtime_source_files_sha256=source,
                      runtime_source_sha256=object_digest(source), model_revision=manifest.get("model_revision"),
                      teacher_parameter_sha256=before,
                      evaluated_checkpoint_file_sha256=hashes["checkpoint_sha256"] if initial else digest_file(directory/"last.pt"),
                      evidence_sha256={filename: digest_file(directory/filename) for filename in evidence_names},
                      fact_label_sha256=object_digest(sorted(answers.items())))
    return dict(method=name, configuration={key: config[key] for key in CONFIG_FIELDS if key in config},
                initialization_step=manifest.get("initialization_step"), provenance=provenance,
                training=training, conditions=conditions), private


def audit_suite(path):
    directory = path.parent if path.name == "suite.json" else path
    suite = read_json(directory/"suite.json")
    if suite.get("complete") is not True or suite.get("status") != "complete":
        raise ValueError("Suite is incomplete")
    if suite.get("protocol") != "context-distillation-pilot-v1":
        raise ValueError("Unexpected suite protocol")
    source = suite.get("source_files_sha256")
    if not isinstance(source, dict) or not source:
        raise ValueError("Suite source manifest missing")
    for filename, expected in source.items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source manifest path")
        if require_hash(expected, "suite source") != digest_file(directory/"source"/relative):
            raise ValueError("Suite snapshot source hash mismatch")
    jobs = suite.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Suite jobs missing")
    names = [job.get("name") for job in jobs]
    if len(names) != len(set(names)) or any(name not in ARMS for name in names):
        raise ValueError("Unknown/duplicate suite arm")
    for job in jobs:
        if job.get("status") != "complete" or type(job.get("exit_code")) is not int or job["exit_code"] != 0:
            raise ValueError("Suite job did not complete successfully")
    return [(name, *audit_run(directory/name, name, suite)) for name in names], dict(
        suite_manifest_sha256=digest_file(directory/"suite.json"), source_snapshot_sha256=object_digest(source),
        methods=names,
    )


def summarize(suites, *, samples=5000, seed=123):
    audited, suites_public = {}, []
    for path in suites:
        runs, suite = audit_suite(Path(path))
        suites_public.append(suite)
        for name, public, private in runs:
            if name in audited:
                raise ValueError("Repeated arm across suites")
            audited[name] = (public, private)
    if set(audited) != set(ARMS):
        raise ValueError("Final audit requires initial and all five training arms")
    initial_public, initial_private = audited["initial"]
    for name in ARMS:
        public, private = audited[name]
        if private["answers"] != initial_private["answers"]:
            raise ValueError("Arm development fact/label sets differ")
        for field in ("cache_sha256", "checkpoint_sha256", "model_manifest_sha256", "data_fingerprint",
                      "module_sha256_before", "runtime_source_sha256", "model_revision", "teacher_parameter_sha256"):
            if public["provenance"][field] != initial_public["provenance"][field]:
                raise ValueError(f"Arm provenance differs: {field}")
        for field in ("seed", "batch_size", "eval_size", "max_new_tokens"):
            if public["configuration"][field] != initial_public["configuration"][field]:
                raise ValueError(f"Arm configuration differs: {field}")
    trained_reference = audited["ce"][0]["training"]
    for name in ARMS[1:]:
        training = audited[name][0]["training"]
        for field in ("step", "target_exposures", "canonical_replay_exposures", "target_exposure_sha256", "episode_schedule_sha256"):
            if training[field] != trained_reference[field]:
                raise ValueError(f"Training arms have different schedules/budgets: {field}")
    for name in ARMS:
        public, private = audited[name]
        for phase in PHASES:
            def values(owner, method):
                return {identity: [row["em"]] for identity, row in owner["student"][phase, method].items()}
            real = values(private, "real")
            teacher = {identity: [row["em"]] for identity, row in initial_private["teacher"][phase].items()}
            public["conditions"][phase]["paired"] = dict(
                real_minus_shuffled=bootstrap_difference(real, values(private, "shuffled"), samples=samples, seed=seed),
                real_minus_initial=bootstrap_difference(real, values(initial_private, "real"), samples=samples, seed=seed),
                teacher_minus_real=bootstrap_difference(teacher, real, samples=samples, seed=seed),
            )
        clustered = {
            method: {identity: [private["student"][phase, method][identity]["em"] for phase in PHASES]
                     for identity in private["answers"]}
            for method in ("real", "shuffled")
        }
        public["all_conditions_real_minus_shuffled"] = bootstrap_difference(
            clustered["real"], clustered["shuffled"], samples=samples, seed=seed,
        )
    return dict(
        protocol="context-distillation-public-audit-v1", audited_at=datetime.now(timezone.utc).isoformat(),
        complete=True, audit_tolerance=TOLERANCE, suites=suites_public,
        evaluation=dict(split="dev", facts=initial_public["configuration"]["eval_size"], conditions=list(PHASES),
                        fact_label_sha256=initial_public["provenance"]["fact_label_sha256"],
                        training_seeds=[initial_public["configuration"]["seed"]]),
        comparability=dict(shared_initialization=True, frozen_teacher_verified=True,
                           matching_target_exposure_sha256=trained_reference["target_exposure_sha256"],
                           matching_episode_schedule_sha256=trained_reference["episode_schedule_sha256"],
                           matched_updates=trained_reference["step"],
                           note="Five training arms match schedules and updates; sampled lengths, teacher/rollout work and wall time can differ. Initial performs zero updates."),
        teacher_acceptance={phase: initial_public["conditions"][phase]["teacher"] for phase in PHASES},
        runs=[audited[name][0] for name in ARMS],
        limitations=[
            "Exploratory development-set experiment: at most 64 facts, short finite-vocabulary answers, one training seed.",
            "No held-out confirmation claim, no checkpoint selection from development results, and no evidence of general continual-learning ability.",
            "Four conditions reuse facts; across-condition uncertainty bootstraps fact clusters, not individual fact/phase records.",
            "Gold+EOS prefix KL/hidden diagnostics use teacher forcing and are distinct from free-generation EM; first-token metrics expose the answer-free prefix.",
            "Cosine hidden alignment can be high for the no-memory backbone; compare changes against that baseline instead of absolute cosine alone.",
            "Forced-value oracle changes the readout distribution and is a diagnostic, not a guaranteed accuracy upper bound.",
            "Token cost totals count unpadded input positions and omit backward/FLOP costs; on-policy rollouts recompute prompt/prefix positions.",
            "Bootstrap intervals condition on this trained model and these template families; they do not estimate seed or unseen-family uncertainty.",
        ],
    )


def report(summary):
    short = {phase: label for phase, label in zip(PHASES, ("规范支持/规范问句", "规范支持/新问句", "新支持/规范问句", "新支持/新问句"))}
    pct = lambda x: f"{100*x:.1f}%"
    def paired(value):
        return f"{100*value['delta_em']:+.1f} [{100*value['ci95'][0]:+.1f}, {100*value['ci95'][1]:+.1f}]"
    lines = ["# Context teacher 蒸馏：已审计开发集结果", "",
             f"六组结果通过逐记录审计；五个训练组具有相同 target/episode schedule SHA，initial 为零更新。开发集 {summary['evaluation']['facts']} 个事实，同一事实在四个条件复用。", "",
             "## 自由生成与检索", "", "单元格：real EM / prefill R@1。Oracle 是强制 value 的诊断，不是数学上界。", "",
             "| 方法 | " + " | ".join(short.values()) + " |", "|---|" + "---|"*4]
    for run in summary["runs"]:
        cells = [f"{pct(run['conditions'][phase]['student']['real']['em'])} / {pct(run['conditions'][phase]['student']['real']['recall_at_1'])}" for phase in PHASES]
        lines.append("| "+run["method"]+" | "+" | ".join(cells)+" |")
    lines += ["", "## 记忆因果对照", "", "real − shuffled，单位百分点及 95% fact-paired bootstrap CI；最后一列先按事实平均四个条件，再抽样事实。", "",
              "| 方法 | "+" | ".join(short.values())+" | 四条件 fact-cluster 平均 |", "|---|"+"---|"*5]
    for run in summary["runs"]:
        cells = [paired(run["conditions"][phase]["paired"]["real_minus_shuffled"]) for phase in PHASES]
        lines.append("| "+run["method"]+" | "+" | ".join(cells+[paired(run["all_conditions_real_minus_shuffled"])])+" |")
    lines += ["", "## Teacher 验收与初始学生", "", "Teacher 接收文本事实；student 接收问题与 VDB。此对照用于确认 teacher 是否提供有效监督。", "",
              "| 条件 | Teacher EM | Initial real EM | Teacher − initial (pp, 95% CI) | Teacher answer NLL |", "|---|---|---|---|---|"]
    initial = summary["runs"][0]
    for phase in PHASES:
        condition, teacher = initial["conditions"][phase], summary["teacher_acceptance"][phase]
        lines.append(f"| {short[phase]} | {pct(teacher['em'])} | {pct(condition['student']['real']['em'])} | {paired(condition['paired']['teacher_minus_real'])} | {teacher['answer_token_nll']:.4f} |")
    lines += ["", "## Gold-prefix 对齐诊断", "", "每项先按事实内部 token 平均，再按事实平均。KL 为 reverse KL；hidden 为 cosine。括号内是相同问题的 no-memory 基线；首 token 尚未输入 gold answer。", "",
              "| 方法 | 条件 | KL (base) | 首 token KL (base) | Hidden cosine (base) | 首 token cosine (base) |", "|---|---|---|---|---|---|"]
    for run in summary["runs"]:
        for phase in PHASES:
            item = run["conditions"][phase]["alignment"]
            cells = [f"{item[a]:.4f} ({item[b]:.4f})" for a, b in (
                ("reverse_kl", "no_memory_reverse_kl"), ("first_token_reverse_kl", "first_token_no_memory_reverse_kl"),
                ("hidden_cosine", "no_memory_hidden_cosine"), ("first_token_hidden_cosine", "first_token_no_memory_hidden_cosine"))]
            lines.append("| "+run["method"]+" | "+short[phase]+" | "+" | ".join(cells)+" |")
    lines += ["", "## 实际训练成本", "", "相同步数不代表相同计算成本。Input totals 是未补齐的 student/teacher/rollout/replay 输入位置总数；不等于 FLOPs。", "",
              "| 方法 | 更新 | Target exposures | Main targets | Teacher targets | Sampled tokens | Replay targets | 总 input tokens | 训练秒数 |", "|---|---|---|---|---|---|---|---|---|"]
    for run in summary["runs"]:
        t = run["training"]
        tok = t["token_totals"]
        lines.append(f"| {run['method']} | {t['step']} | {t['target_exposures']} | {tok['main_target_tokens']} | {tok['teacher_target_tokens']} | {tok['sampled_tokens']} | {tok['replay_target_tokens']} | {t['processed_input_tokens_total']} | {t['elapsed_seconds']:.1f} |")
    lines += ["", "## 范围与限制", ""] + ["- "+item for item in summary["limitations"]]
    lines += ["", "summary.json 保留全部四象限的 real/oracle/shuffled/empty EM、NLL、R@1/R@4、decode 指标、成对差值及来源 SHA；原始提示、token ID、事实 ID 和本地路径均未发布。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    args = parser.parse_args(argv)
    summary = summarize(args.suite, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, contents in (("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n"),
                           ("report.md", report(summary))):
        target = args.output/name
        temporary = target.with_suffix(target.suffix+".partial")
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(target)
    print(json.dumps(dict(complete=True, arms=len(summary["runs"]), facts=summary["evaluation"]["facts"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
