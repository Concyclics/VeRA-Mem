"""Strict completion, independent aggregation, and fact-cluster audit checks."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "summarize_context_distillation", Path(__file__).resolve().parents[1]/"scripts/summarize_context_distillation.py")
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


def mutate_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)


def fixture_suite(tmp_path):
    suite = tmp_path/"private_experiment"
    source = suite/"source/src/vera_mem/context_distillation_run.py"
    source.parent.mkdir(parents=True)
    source.write_text("# immutable mock source\n")
    source_hash = SUMMARY.digest_file(source)
    snapshot = {"src/vera_mem/context_distillation_run.py": source_hash}
    jobs = []
    for arm in SUMMARY.ARMS:
        directory = suite/arm
        directory.mkdir()
        initial = arm == "initial"
        method = "ce" if initial else arm
        config = dict(method=method, model="/private/model", cache="/private/features.pt", checkpoint="/private/init.pt",
                      run_dir=str(directory), updates=4, batch_size=2, eval_size=2, max_new_tokens=4,
                      sampling_temperature=1., seed=42, evaluate_only=initial, skip_teacher_eval=not initial,
                      kl_direction="reverse", kl_temperature=1., hidden_weight=.1,
                      canonical_replay_frequency=4, canonical_replay_weight=.25,
                      all_view_address_weight=.2, actual_prefix_address_weight=.2)
        answers = {"PRIVATE_ENTITY_A": "PRIVATE_ANSWER_A", "PRIVATE_ENTITY_B": "PRIVATE_ANSWER_B"}
        write_jsonl(directory/"dev.jsonl", [dict(id=identity, answer=answer, question="PRIVATE_PROMPT") for identity, answer in answers.items()])
        predictions, alignment_predictions, teacher_predictions, conditions = [], [], [], {}
        for phase in SUMMARY.PHASES:
            scores = {}
            for evaluation_method in SUMMARY.METHODS:
                em = 1. if evaluation_method == "oracle" else .5 if evaluation_method == "real" and not initial else 0.
                routed = evaluation_method in ("real", "shuffled")
                for number, (identity, answer) in enumerate(answers.items()):
                    correct = evaluation_method == "oracle" or (evaluation_method == "real" and not initial and number == 0)
                    predictions.append(dict(
                        id=identity, answer=answer, prediction=answer if correct else "PRIVATE_WRONG",
                        phase=phase, method=evaluation_method, em=int(correct), nll_sum=float(number+1), tokens=number+1,
                        expected_in_bank=True, selected_ids=["PRIVATE_ENTITY_A"] if routed else [],
                        decode_correct_residency=float(number == 0) if routed else None,
                        decode_query_count=1 if routed else 0, decode_correct_hits=int(number == 0) if routed else None,
                        switch_count=0 if routed else None, retrieval_transition_count=1 if routed else 0,
                    ))
                scores[evaluation_method] = dict(count=2, em=em, answer_token_nll=1.)
                if routed:
                    scores[evaluation_method].update(
                        recall_at_1=.5, recall_at_4=.5, decode_correct_residency=.5,
                        decode_query_count=2, decode_examples=2, switch_count_mean=0.,
                        switch_count_total=0, retrieval_transition_count=2, top1_switch_rate=0.)
            alignment = dict(count=2, tokens=5, reverse_kl=2., hidden_cosine=.7,
                             no_memory_reverse_kl=3., no_memory_hidden_cosine=.5,
                             first_token_reverse_kl=2.5, first_token_hidden_cosine=.6,
                             first_token_no_memory_reverse_kl=4., first_token_no_memory_hidden_cosine=.4,
                             aggregation=SUMMARY.ALIGNMENT_AGGREGATION)
            for number, (identity, answer) in enumerate(answers.items()):
                row = dict(id=identity, phase=phase, tokens=number+2, continuation_token_ids=[101]*(number+2))
                for field in SUMMARY.ALIGNMENT_FIELDS:
                    offset = (-1 if number == 0 else 1) * (.1 if "cosine" in field else 1.)
                    row[field] = alignment[field]+offset
                alignment_predictions.append(row)
                if initial:
                    teacher_predictions.append(dict(id=identity, phase=phase, method="teacher", answer=answer,
                        prediction=answer, em=1, tokens=number+1, nll_sum=float(number+1),
                        expected_in_bank=False, teacher_context="PRIVATE_TEACHER_PROMPT"))
            conditions[phase] = dict(student=scores, alignment=alignment,
                                     teacher=dict(count=2, em=1., answer_token_nll=1.) if initial else None,
                                     records=2, numeric_vdb_bytes=1024, vdb_sha256="a"*64)
        write_jsonl(directory/"predictions.jsonl", predictions)
        write_jsonl(directory/"alignment_predictions.jsonl", alignment_predictions)
        if initial:
            write_jsonl(directory/"teacher_predictions.jsonl", teacher_predictions)
        module_after = "5"*64 if initial else "8"*64
        metrics = dict(conditions=conditions, eval_facts=2, evaluation_split="dev",
                       online_gradient_steps=0, shared_parameters_unchanged=True, vdb_unchanged=True,
                       model_id="Qwen/Qwen3-4B-Instruct-2507", module_sha256=module_after)
        manifest = dict(complete=True, finished=True, method=method, evaluate_only=initial,
                        teacher_parameters_unchanged=True, teacher_parameter_sha256_before="6"*64,
                        teacher_parameter_sha256_after="6"*64, module_sha256_before="5"*64,
                        module_sha256_after=module_after, cache_sha256="1"*64, checkpoint_sha256="2"*64,
                        model_manifest_sha256="3"*64, data_fingerprint="4"*64, initialization_step=1600,
                        model_revision="abcdef1234", source_files_sha256={source.name: source_hash},
                        cache_splits_used=["train", "dev"], confirmation_used=False, dev_selection=False,
                        selected_step=0 if initial else 4, training=None)
        if not initial:
            rows = []
            fact_digest, episode_digest = hashlib.sha256(), hashlib.sha256()
            totals = {field: 0 for field in SUMMARY.TOKEN_FIELDS}
            for step in range(1, 5):
                targets = ["PRIVATE_TRAIN_A", "PRIVATE_TRAIN_B"]
                episode = targets+["PRIVATE_NEGATIVE"]
                qviews, sviews = [step%8, 0, 3], [step%4, 1, 2]
                target_tokens = 4
                row = dict(step=step, method=method, target_ids=targets, episode_ids=episode,
                           query_view_indices=qviews, support_view_indices=sviews, target_to_episode=[0, 1],
                           main_target_tokens=target_tokens, teacher_target_tokens=0 if method == "ce" else target_tokens,
                           sampled_tokens=target_tokens if method.startswith("on_") else 0,
                           student_input_tokens=100, teacher_input_tokens=0 if method == "ce" else 150,
                           rollout_input_tokens=400 if method.startswith("on_") else 0,
                           replay_target_tokens=4 if step == 4 else 0, replay_input_tokens=100 if step == 4 else 0)
                rows.append(row)
                fact_digest.update(json.dumps(targets, separators=(",", ":")).encode())
                schedule = dict(ids=episode, qviews=qviews, sviews=sviews, targets=targets)
                episode_digest.update(json.dumps(schedule, separators=(",", ":")).encode())
                for field in totals:
                    totals[field] += row[field]
            write_jsonl(directory/"training.jsonl", rows)
            write_jsonl(directory/"rollouts.jsonl", [dict(question="PRIVATE_ROLLOUT", continuation_token_ids=[77, 88])])
            (directory/"last.pt").write_bytes(b"mock-checkpoint")
            training = dict(step=4, finished=True, selected_step=4, selection="fixed final update; no development selection",
                            target_exposures=8, canonical_replay_exposures=2, target_exposure_sha256=fact_digest.hexdigest(),
                            episode_schedule_sha256=episode_digest.hexdigest(), token_totals=totals, elapsed_seconds=10.,
                            rollout_sha256=SUMMARY.digest_file(directory/"rollouts.jsonl"),
                            training_log_sha256=SUMMARY.digest_file(directory/"training.jsonl"))
            manifest["training"] = training
            write_json(directory/"training_status.json", training)
        for filename, value in (("manifest.json", manifest), ("metrics.json", metrics), ("config.json", config)):
            write_json(directory/filename, value)
        jobs.append(dict(name=arm, status="complete", exit_code=0, command=["PRIVATE_COMMAND", str(directory)]))
    write_json(suite/"suite.json", dict(protocol="context-distillation-pilot-v1", complete=True, status="complete",
               jobs=jobs, source_files_sha256=snapshot, cache_sha256="1"*64, checkpoint_sha256="2"*64))
    return suite


def test_complete_audit_recomputes_metrics_costs_and_removes_private_fields(tmp_path):
    suite = fixture_suite(tmp_path)
    result = SUMMARY.summarize([suite], samples=40)
    assert result["complete"] and len(result["runs"]) == 6
    assert result["evaluation"]["facts"] == 2
    initial, ce, _, _, on, _ = result["runs"]
    assert initial["training"]["step"] == 0
    assert ce["training"]["target_exposures"] == 8
    assert ce["training"]["processed_input_tokens_total"] == 500
    assert on["training"]["processed_input_tokens_total"] == 2700
    condition = ce["conditions"][SUMMARY.PHASES[0]]
    assert condition["student"]["real"]["em"] == .5
    assert condition["student"]["real"]["answer_token_nll"] == 1.
    assert condition["student"]["real"]["recall_at_4"] == .5
    assert condition["paired"]["real_minus_shuffled"]["delta_em"] == .5
    assert ce["all_conditions_real_minus_shuffled"]["n_paired_facts"] == 2
    assert ce["all_conditions_real_minus_shuffled"]["phases_per_fact"] == 4
    assert result["teacher_acceptance"][SUMMARY.PHASES[0]]["em"] == 1.
    serialized = json.dumps(result)+SUMMARY.report(result)
    for private in ("PRIVATE_ENTITY", "PRIVATE_ANSWER", "PRIVATE_PROMPT", "PRIVATE_COMMAND", "PRIVATE_ROLLOUT", "/private/", str(tmp_path), "continuation_token_ids"):
        assert private not in serialized
    assert "首 token" in SUMMARY.report(result)
    assert "no-memory" in SUMMARY.report(result)


@pytest.mark.parametrize("mutation,match", [("missing", "Missing fact"), ("duplicate", "Duplicate fact"), ("em", "exact_match")])
def test_missing_duplicate_and_false_em_records_fail(tmp_path, mutation, match):
    suite = fixture_suite(tmp_path)
    path = suite/"ce/predictions.jsonl"
    rows = SUMMARY.read_jsonl(path)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["em"] = 1-rows[0]["em"]
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match=match):
        SUMMARY.summarize([suite], samples=10)


@pytest.mark.parametrize("field", ["em", "recall_at_1", "recall_at_4", "answer_token_nll", "count"])
def test_reported_prediction_metrics_must_match_raw_records(tmp_path, field):
    suite = fixture_suite(tmp_path)
    mutate_json(suite/"ce/metrics.json", lambda value: value["conditions"][SUMMARY.PHASES[0]]["student"]["real"].__setitem__(field, 9.))
    with pytest.raises(ValueError, match="Metric value mismatch"):
        SUMMARY.summarize([suite], samples=10)


@pytest.mark.parametrize("field", ["first_token_reverse_kl", "no_memory_hidden_cosine", "tokens"])
def test_alignment_first_token_baseline_and_token_aggregation_are_audited(tmp_path, field):
    suite = fixture_suite(tmp_path)
    mutate_json(suite/"ce/metrics.json", lambda value: value["conditions"][SUMMARY.PHASES[0]]["alignment"].__setitem__(field, 17.))
    with pytest.raises(ValueError, match="Metric value mismatch"):
        SUMMARY.summarize([suite], samples=10)


@pytest.mark.parametrize("location,field,value,match", [
    ("suite.json", "complete", False, "Suite is incomplete"),
    ("ce/manifest.json", "complete", False, "Run manifest is incomplete"),
    ("ce/manifest.json", "teacher_parameters_unchanged", False, "teacher verification"),
    ("ce/manifest.json", "teacher_parameter_sha256_after", "f"*64, "Teacher parameter hashes differ"),
    ("ce/manifest.json", "confirmation_used", True, "confirmation use"),
    ("initial/manifest.json", "selected_step", 4, "zero training"),
])
def test_completion_frozen_teacher_and_initial_requirements(tmp_path, location, field, value, match):
    suite = fixture_suite(tmp_path)
    mutate_json(suite/location, lambda item: item.__setitem__(field, value))
    with pytest.raises(ValueError, match=match):
        SUMMARY.summarize([suite], samples=10)


def test_nonzero_job_exit_and_incomplete_arm_set_fail(tmp_path):
    suite = fixture_suite(tmp_path)
    mutate_json(suite/"suite.json", lambda item: item["jobs"][0].__setitem__("exit_code", 1))
    with pytest.raises(ValueError, match="complete successfully"):
        SUMMARY.summarize([suite], samples=10)
    mutate_json(suite/"suite.json", lambda item: item.__setitem__("jobs", item["jobs"][1:]))
    with pytest.raises(ValueError, match="all five training arms"):
        SUMMARY.summarize([suite], samples=10)


def test_training_status_schedule_and_token_totals_are_recomputed(tmp_path):
    suite = fixture_suite(tmp_path)
    manifest_path, status_path = suite/"on_kd/manifest.json", suite/"on_kd/training_status.json"
    for path in (manifest_path, status_path):
        def change(item):
            training = item["training"] if "training" in item else item
            training["token_totals"]["sampled_tokens"] += 1
        mutate_json(path, change)
    with pytest.raises(ValueError, match="training.token_totals.sampled_tokens"):
        SUMMARY.summarize([suite], samples=10)


def test_different_but_internally_consistent_episode_schedule_is_rejected(tmp_path):
    suite = fixture_suite(tmp_path)
    directory = suite/"on_kd"
    rows = SUMMARY.read_jsonl(directory/"training.jsonl")
    rows[0]["query_view_indices"][0] = 7
    write_jsonl(directory/"training.jsonl", rows)
    digest = hashlib.sha256()
    for row in rows:
        record = dict(ids=row["episode_ids"], qviews=row["query_view_indices"], sviews=row["support_view_indices"], targets=row["target_ids"])
        digest.update(json.dumps(record, separators=(",", ":")).encode())
    for path in (directory/"manifest.json", directory/"training_status.json"):
        def change(item):
            training = item["training"] if "training" in item else item
            training["episode_schedule_sha256"] = digest.hexdigest()
            training["training_log_sha256"] = SUMMARY.digest_file(directory/"training.jsonl")
        mutate_json(path, change)
    with pytest.raises(ValueError, match="different schedules/budgets"):
        SUMMARY.summarize([suite], samples=10)


def test_bootstrap_uses_fact_clusters_is_reproducible_and_refuses_missing_pairs():
    first = {"a": [1, 0, 1, 0], "b": [1, 0, 1, 0]}
    second = {"b": [0, 0, 0, 0], "a": [0, 0, 0, 0]}
    result = SUMMARY.bootstrap_difference(first, second, samples=40, seed=7)
    assert result["delta_em"] == .5
    assert result["ci95"] == [.5, .5]
    assert result["n_paired_facts"] == 2  # Never counts eight fact/phase rows.
    assert result["phases_per_fact"] == 4
    assert result == SUMMARY.bootstrap_difference(dict(reversed(list(first.items()))), second, samples=40, seed=7)
    with pytest.raises(ValueError, match="fact sets"):
        SUMMARY.bootstrap_difference(first, {"a": [0, 0, 0, 0]})
    with pytest.raises(ValueError, match="phase counts"):
        SUMMARY.bootstrap_difference(first, {"a": [0], "b": [0]})


def test_cli_accepts_manifest_path_writes_only_aggregate_artifacts_and_keeps_evidence(tmp_path):
    suite = fixture_suite(tmp_path)
    before = {str(path.relative_to(suite)): SUMMARY.digest_file(path) for path in suite.rglob("*") if path.is_file()}
    output = tmp_path/"public"
    assert SUMMARY.main(["--suite", str(suite/"suite.json"), "--output", str(output), "--bootstrap-samples", "20"]) == 0
    assert set(path.name for path in output.iterdir()) == {"summary.json", "report.md"}
    after = {str(path.relative_to(suite)): SUMMARY.digest_file(path) for path in suite.rglob("*") if path.is_file()}
    assert before == after
