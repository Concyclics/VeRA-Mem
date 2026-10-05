"""Reject incomplete/malformed evidence and preserve entity-level uncertainty."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "summarize_generalization", Path(__file__).resolve().parents[1] / "scripts" / "summarize_generalization.py",
)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def create_run(root, condition="canonical"):
    suite = root / ("generalization_" + condition)
    run = suite / (condition + "4096")
    run.mkdir(parents=True)
    source = suite / "source" / "src" / "vera_mem"
    source.mkdir(parents=True)
    (source / "generalization_run.py").write_text("# frozen source fixture\n")
    config = dict(condition=condition, train_size=4096, updates=1536,
                  oracle_updates=768, alignment_steps=400, batch_size=8,
                  validate_every=256, eval_size=128, seed=42)
    write_json(run / "config.json", config)
    manifest = dict(complete=True, selected_step=1536, cache_sha256="a"*64,
                    data_fingerprint="b"*64, model_revision="pinned-model",
                    source_files_sha256={"generalization_run.py": summary.digest_file(source / "generalization_run.py")})
    write_json(run / "manifest.json", manifest)
    write_json(suite / "suite.json", dict(complete=True, status="complete", profile=condition,
                                         cache_sha256="a"*64,
                                         jobs=[dict(name=run.name, status="complete", exit_code=0)]))
    facts = [dict(id=f"fact_{index}", answer="apple", question="PRIVATE PROMPT MUST NOT APPEAR") for index in range(128)]
    controls = [dict(id=f"control_{index}", answer="apple") for index in range(32)]
    write_rows(run / "test.jsonl", facts)
    write_rows(run / "control.jsonl", controls)
    groups = {}
    def prediction(fact, phase, method):
        written = phase not in ("pre_write", "control_before", "control_after")
        correct = written and method in ("real", "oracle")
        return dict(id=fact["id"], answer=fact["answer"], phase=phase, method=method,
                    prediction="APPLE!" if correct else "banana", em=int(correct),
                    nll_sum=.5, tokens=1, expected_in_bank=written,
                    selected_ids=[fact["id"]] if written and method in ("real", "shuffled") else [],
                    decode_correct_residency=None, switch_count=None)
    for phase in ("pre_write", "immediate", "control_before", "control_after"):
        selected = controls if phase.startswith("control_") else facts
        groups[(phase, "real")] = [prediction(fact, phase, "real") for fact in selected]
    for bank in summary.BANKS:
        for template in summary.QUERIES:
            for method in summary.METHODS:
                phase = bank + "/" + template
                groups[(phase, method)] = [prediction(fact, phase, method) for fact in facts]
    metrics = dict(conditions={}, online_gradient_steps=0, shared_parameters_unchanged=True, eval_facts=128)
    for bank in summary.BANKS:
        metrics["conditions"][bank] = dict(records=128, scores={})
        for template in summary.QUERIES:
            metrics["conditions"][bank]["scores"][template] = {
                method: summary.summarize_phase(groups[(bank + "/" + template, method)])
                for method in summary.METHODS}
    for phase in ("pre_write", "immediate", "control_before", "control_after"):
        metrics["conditions"]["canonical_support"][phase] = summary.summarize_phase(groups[(phase, "real")])
    write_json(run / "metrics.json", metrics)
    write_rows(run / "predictions.jsonl", [row for rows in groups.values() for row in rows])
    writes = [dict(id=fact["id"], bank=bank, timestamp=index,
                   support_template="train_support_00" if bank == "canonical_support" else f"test_support_{index % 3:02d}")
              for bank in summary.BANKS for index, fact in enumerate(facts)]
    write_json(run / "writes.json", writes)
    write_json(run / "training.json", dict(training=[dict(step=1536, prompt_tokens_seen=123456, seconds=42.)]))
    return run


def test_complete_evidence_has_four_quadrants_and_no_private_payload(tmp_path):
    run = create_run(tmp_path)
    result, _ = summary.load_run(run, tmp_path)
    assert result["verified_prediction_count"] == 4416
    assert len(result["quadrants"]) == 4
    assert result["quadrants"]["heldout_query/heldout_support"]["real"] == dict(
        n_facts=128, n_templates=3, n_predictions=384, correct=384, em=1.,
        aggregation="equal-weight query-template macro average", recall_at_1=1., recall_at_4=1.)
    aggregate = summary.collect(tmp_path, samples=10)
    serialized = json.dumps(aggregate)
    assert "PRIVATE PROMPT" not in serialized and str(tmp_path) not in serialized
    assert "fact_0" not in serialized and "APPLE!" not in serialized
    assert aggregate["completed_run_count"] == 1
    assert "新查询+新观测" in summary.render(aggregate)


def test_false_em_cannot_pass_even_if_reported_phase_mean_is_unchanged(tmp_path):
    run = create_run(tmp_path)
    rows = summary.read_rows(run / "predictions.jsonl")
    target = next(row for row in rows if row["phase"] == "canonical_support/train_query_00" and row["method"] == "real")
    target["prediction"] = "banana"  # Leave claimed EM and aggregate metrics at 1.
    write_rows(run / "predictions.jsonl", rows)
    with pytest.raises(ValueError, match="Stored EM disagrees"):
        summary.load_run(run, tmp_path)
    aggregate = summary.collect(tmp_path, samples=10)
    assert aggregate["completed_run_count"] == 0
    assert len(aggregate["skipped_runs"]) == 1


def test_duplicate_fact_replacing_another_is_rejected_despite_equal_counts(tmp_path):
    run = create_run(tmp_path)
    rows = summary.read_rows(run / "predictions.jsonl")
    rows[1] = copy.deepcopy(rows[0])
    write_rows(run / "predictions.jsonl", rows)
    with pytest.raises(ValueError, match="Duplicate fact ID"):
        summary.load_run(run, tmp_path)


@pytest.mark.parametrize("mutation", ["missing_phase", "failed_suite", "metric_lie", "bad_write", "changed_source"])
def test_protocol_evidence_must_be_complete_and_consistent(tmp_path, mutation):
    run = create_run(tmp_path)
    if mutation == "missing_phase":
        rows = summary.read_rows(run / "predictions.jsonl")
        write_rows(run / "predictions.jsonl", [row for row in rows if row["phase"] != "control_before"])
    elif mutation == "failed_suite":
        suite = summary.read_json(run.parent / "suite.json")
        suite["jobs"][0]["exit_code"] = 1
        write_json(run.parent / "suite.json", suite)
    elif mutation == "metric_lie":
        metrics = summary.read_json(run / "metrics.json")
        metrics["conditions"]["heldout_support"]["scores"]["test_query_01"]["real"]["recall_at_4"] = 0
        write_json(run / "metrics.json", metrics)
    elif mutation == "bad_write":
        writes = summary.read_json(run / "writes.json")
        writes[-1]["support_template"] = "train_support_00"
        write_json(run / "writes.json", writes)
    else:
        (run.parent / "source" / "src" / "vera_mem" / "generalization_run.py").write_text("different source")
    with pytest.raises(ValueError):
        summary.load_run(run, tmp_path)


def test_bootstrap_clusters_all_three_query_views_within_entity():
    groups = {}
    scores = ((1, 0), (0, 1), (1, 0))
    for template, values in zip(summary.HELDOUT_QUERIES, scores):
        groups[("heldout_support/" + template, "real")] = [
            dict(id="a", answer="apple", em=values[0]), dict(id="b", answer="river", em=values[1])]
    first = summary.entity_scores(groups, "heldout_query/heldout_support")
    assert first == {"a": ("apple", 2/3), "b": ("river", 1/3)}
    second = {"a": ("apple", 0.), "b": ("river", 0.)}
    estimate = summary.paired_bootstrap(first, second, samples=1000)
    assert estimate["n_paired_facts"] == 2  # Not six independent examples.
    assert estimate["delta_em"] == .5
    assert estimate["ci95"] == pytest.approx([1/3, 2/3])
    with pytest.raises(ValueError, match="full fact ID equality"):
        summary.paired_bootstrap(first, {"a": ("apple", 0.)})


def test_matched_conditions_compare_but_changed_training_seed_refuses(tmp_path):
    create_run(tmp_path, "canonical")
    second = create_run(tmp_path, "augment")
    result = summary.collect(tmp_path, samples=10)
    assert len(result["paired_comparisons"]) == 4
    assert all(row["n_paired_facts"] == 128 and row["delta_em"] == 0 for row in result["paired_comparisons"])
    config = summary.read_json(second / "config.json")
    config["seed"] = 99
    write_json(second / "config.json", config)
    result = summary.collect(tmp_path, samples=10)
    assert not result["paired_comparisons"]
    assert len(result["refused_comparisons"]) == 1
