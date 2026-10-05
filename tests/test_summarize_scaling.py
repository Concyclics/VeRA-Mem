"""Pairing and evidence-admission tests for the aggregate-only scaling report."""
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "summarize_scaling", Path(__file__).resolve().parents[1] / "scripts/summarize_scaling.py")
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def score(identity, em, answer="secret-answer"):
    return dict(id=identity, answer=answer, em=em)


def test_paired_bootstrap_matches_ids_not_file_order_and_has_known_degenerate_interval():
    left = [score("one", 1), score("two", 1), score("three", 1)]
    right = [score("three", 0), score("one", 0), score("two", 0)]
    result = SUMMARY.paired_bootstrap(left, right, samples=50)
    assert result["n_paired_facts"] == 3
    assert result["delta_em"] == 1
    assert result["ci95"] == [1, 1]
    assert "secret-answer" not in json.dumps(result)
    reversed_result = SUMMARY.paired_bootstrap(list(reversed(left)), right, samples=50)
    assert result == reversed_result


@pytest.mark.parametrize("right, message", [
    ([score("one", 0)], "ID sets differ"),
    ([score("one", 0), score("two", 0, "changed")], "labels differ"),
    ([score("one", 0), score("one", 0)], "Duplicate"),
])
def test_pairing_rejects_missing_labels_or_duplicates_without_intersection_fallback(right, message):
    left = [score("one", 1), score("two", 0)]
    with pytest.raises(ValueError, match=message):
        SUMMARY.paired_bootstrap(left, right, samples=10)


def fixture_run(root, name="stable128", train_size=128, complete=True, cache="b"*64):
    directory = root / "scaling_stable" / name
    directory.mkdir(parents=True)
    config = dict(variant="stable", train_size=train_size, updates=512, batch_size=8,
                  eval_size=2, seed=42, cold_bank_size=0, alignment_steps=400,
                  oracle_updates=256, checkpoint=None)
    manifest = dict(complete=complete, cache_sha256=cache, model_revision="c"*40,
                    source_files_sha256={"scaling_run.py": "a"*64}, selected_step=256)
    rows = []
    metrics = dict(value_effective_rank=2., value_unique_rows=2, value_std_mean=.5,
                   value_abs_max=1., cold_records=0, records=2,
                   online_gradient_steps=0, shared_parameters_unchanged=True)
    for phase, method in SUMMARY.PHASES:
        group = []
        for number in range(2):
            identity = ("control" if phase.startswith("control_") else "fact") + str(number)
            group.append(dict(id=identity, answer="PRIVATE_ANSWER", prediction="PRIVATE_PREDICTION",
                              phase=phase, method=method, em=int(number == 0 and method != "empty"),
                              nll_sum=number + 1., tokens=1, expected_in_bank=phase in ("immediate", "final", "paraphrase"),
                              selected_ids=[identity] if number == 0 else ["wrong"]))
        computed = SUMMARY.phase_summary(group)
        reported = {"count": computed.pop("n"), **computed}
        if phase in ("final", "paraphrase"):
            metrics.setdefault(phase, {})[method] = reported
        else:
            metrics[phase] = reported
        rows.extend(group)
    for filename, value in (("manifest.json", manifest), ("config.json", config), ("metrics.json", metrics)):
        (directory / filename).write_text(json.dumps(value))
    (directory / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return directory


def test_collect_excludes_smoke_and_incomplete_runs_and_publishes_only_aggregates(tmp_path):
    fixture_run(tmp_path)
    fixture_run(tmp_path, "stable4096", train_size=4096)
    fixture_run(tmp_path, "stable32", train_size=32)
    fixture_run(tmp_path, "pending", complete=False)
    result = SUMMARY.collect(tmp_path, samples=30)
    assert result["completed_run_count"] == 2
    assert len(result["skipped_runs"]) == 2
    assert len(result["paired_comparisons"]) == 5  # two per run, one size contrast
    assert result["source_cache_group_count"] == 1
    serialized = json.dumps(result) + SUMMARY.render(result)
    for private in ("PRIVATE_ANSWER", "PRIVATE_PREDICTION", str(tmp_path)):
        assert private not in serialized
    assert "N4096_minus_N128_matched_updates" in serialized


def test_changed_source_cache_or_labels_refuses_cross_size_comparison(tmp_path):
    fixture_run(tmp_path)
    fixture_run(tmp_path, "stable4096", train_size=4096, cache="d"*64)
    result = SUMMARY.collect(tmp_path, samples=20)
    assert result["completed_run_count"] == 2
    assert result["source_cache_group_count"] == 2
    assert len(result["paired_comparisons"]) == 4
    assert len(result["refused_comparisons"]) == 1
    assert "source_cache_group_sha256" in result["refused_comparisons"][0]["reason"]
    # Even when cache provenance agrees, exact label equality remains required.
    left, right = result["runs"]
    right["source_cache_group_sha256"] = left["source_cache_group_sha256"]
    right["training_lm_updates"] += 1
    with pytest.raises(ValueError, match="training_lm_updates"):
        SUMMARY.cross_run_compatible(left, right)


def test_inconsistent_metrics_are_excluded_even_if_manifest_says_complete(tmp_path):
    run = fixture_run(tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    metrics["final"]["real"]["em"] = 1.
    (run / "metrics.json").write_text(json.dumps(metrics))
    result = SUMMARY.collect(tmp_path, samples=20)
    assert result["completed_run_count"] == 0
    assert "disagree" in result["skipped_runs"][0]["reason"]


def test_eval_only_uses_source_training_config_not_current_cli_budget(tmp_path):
    run = fixture_run(tmp_path, "reevaluation", train_size=4096)
    config = json.loads((run / "config.json").read_text())
    original = dict(config)
    config.update(checkpoint="/private/machine/adapter.pt", updates=999, cold_bank_size=128)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest.update(source_training_config=original, training_cold_bank_size=0,
                    checkpoint_sha256="e"*64)
    metrics = json.loads((run / "metrics.json").read_text())
    metrics["cold_records"] = 128
    for filename, value in (("config.json", config), ("manifest.json", manifest), ("metrics.json", metrics)):
        (run / filename).write_text(json.dumps(value))
    result = SUMMARY.collect(tmp_path, samples=20)
    row = result["runs"][0]
    assert row["new_lm_updates"] == 0
    assert row["training_lm_updates"] == 512
    assert row["training_cold_records"] == 0
    assert row["deployment_cold_records"] == 128
    assert "/private/machine" not in json.dumps(result)


def revise_run(directory, config_changes=None, manifest_changes=None, metrics_changes=None):
    for filename, changes in (("config.json", config_changes), ("manifest.json", manifest_changes),
                              ("metrics.json", metrics_changes)):
        if changes:
            current = json.loads((directory / filename).read_text())
            current.update(changes)
            (directory / filename).write_text(json.dumps(current))


def fixture_reevaluation(root, source, name, deploy_cold):
    original = json.loads((source / "config.json").read_text())
    destination = fixture_run(root, name, train_size=original["train_size"])
    config = dict(original, checkpoint="/private/source/best.pt", cold_bank_size=deploy_cold)
    (destination / "config.json").write_text(json.dumps(config))
    revise_run(destination, manifest_changes=dict(
        checkpoint_sha256=SUMMARY.digest_file(source / "best.pt"),
        source_training_config=original, training_cold_bank_size=original["cold_bank_size"]),
        metrics_changes=dict(cold_records=deploy_cold))
    return destination


def test_cold_deployment_pairs_each_training_condition_only_with_exact_same_checkpoint(tmp_path):
    standard = fixture_run(tmp_path, "standard", train_size=4096)
    standard.joinpath("best.pt").write_bytes(b"standard trained weights")
    cold = fixture_run(tmp_path, "coldtrained", train_size=4096)
    revise_run(cold, config_changes=dict(cold_bank_size=128), metrics_changes=dict(cold_records=128))
    cold.joinpath("best.pt").write_bytes(b"independently cold trained weights")
    fixture_reevaluation(tmp_path, standard, "standard_deploy128", deploy_cold=128)
    fixture_reevaluation(tmp_path, cold, "cold_deploy0", deploy_cold=0)
    result = SUMMARY.collect(tmp_path, samples=20)
    contrasts = [row for row in result["paired_comparisons"]
                 if row["contrast"] == "deployment_cold128_minus_0_same_checkpoint"]
    assert len(contrasts) == 2
    assert {row["training_cold_records"] for row in contrasts} == {0, 128}
    rows = {row["run"]: row for row in result["runs"]}
    for comparison in contrasts:
        first, second = rows[comparison["first_run"]], rows[comparison["second_run"]]
        assert first["deployment_cold_records"] == 128 and second["deployment_cold_records"] == 0
        assert first["checkpoint_sha256"] == second["checkpoint_sha256"]
        assert first["source_training_config_sha256"] == second["source_training_config_sha256"]
        assert first["training_cold_records"] == second["training_cold_records"]
    assert "不是等 FLOPs" in SUMMARY.render(result)
    assert "不是 top-1 比例" in SUMMARY.render(result)


@pytest.mark.parametrize("mismatch", ["checkpoint_sha256", "source_training_config", "cache_sha256", "labels"])
def test_cold_deployment_rejects_changed_weights_training_config_provenance_or_labels(tmp_path, mismatch):
    trained = fixture_run(tmp_path, "trained", train_size=4096)
    trained.joinpath("best.pt").write_bytes(b"verified checkpoint")
    evaluated = fixture_reevaluation(tmp_path, trained, "deployed128", deploy_cold=128)
    if mismatch == "source_training_config":
        changed = json.loads((trained / "config.json").read_text())
        changed["unrecorded_learning_rate"] = 0.123
        revise_run(evaluated, manifest_changes=dict(source_training_config=changed))
    elif mismatch == "labels":
        path = evaluated / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["answer"] = "A_DIFFERENT_PRIVATE_LABEL"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        revise_run(evaluated, manifest_changes={mismatch: "f"*64})
    result = SUMMARY.collect(tmp_path, samples=20)
    assert result["completed_run_count"] == 2
    assert not any(row["contrast"] == "deployment_cold128_minus_0_same_checkpoint"
                   for row in result["paired_comparisons"])
    assert len(result["refused_comparisons"]) == 1
    assert "PRIVATE_LABEL" not in json.dumps(result)


def test_long_budget_contrast_requires_proportional_schedule_and_does_not_enter_matched_size_result(tmp_path):
    fixture_run(tmp_path, "n128", train_size=128)
    fixture_run(tmp_path, "short", train_size=4096)
    long = fixture_run(tmp_path, "long", train_size=4096)
    revise_run(long, config_changes=dict(updates=1536, oracle_updates=768),
               manifest_changes=dict(selected_step=1024))
    result = SUMMARY.collect(tmp_path, samples=20)
    budgets = [row for row in result["paired_comparisons"]
               if row["contrast"] == "stable4096_LMschedule1536_minus_512"]
    assert len(budgets) == 1
    assert budgets[0]["lm_schedule_ratio"] == 3
    assert budgets[0]["first_run"].endswith("/long")
    assert budgets[0]["second_run"].endswith("/short")
    sizes = [row for row in result["paired_comparisons"]
             if row["contrast"] == "N4096_minus_N128_matched_updates"]
    assert len(sizes) == 1 and sizes[0]["first_run"].endswith("/short")
    assert any("training_lm_updates" in row["reason"] for row in result["refused_comparisons"])
    records = {row["run"].split("/")[-1]: row for row in result["runs"]}
    records["long"]["oracle_updates"] = 256
    with pytest.raises(ValueError, match="oracle fraction"):
        SUMMARY.long_budget_compatible(records["long"], records["short"])
    records["long"]["oracle_updates"] = 768
    records["long"]["evaluation_only"] = True
    with pytest.raises(ValueError, match="separately trained"):
        SUMMARY.long_budget_compatible(records["long"], records["short"])
