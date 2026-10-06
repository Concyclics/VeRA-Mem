"""Aggregate completed scaling runs without models, GPUs, or private examples.

Usage:
  python scripts/summarize_scaling.py --runs-root WORKSPACE/runs/xtrah100 --output REPORT_DIR

REPORT_DIR receives report.md and summary.json. Runs are discovered below
scaling_* suites; smoke runs and incomplete evidence are always excluded.
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


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
METHODS = ("real", "oracle", "shuffled", "empty")
PHASES = (("pre_write", "real"), ("immediate", "real"),
          *(("final", method) for method in METHODS),
          ("paraphrase", "real"), ("paraphrase", "oracle"),
          ("control_before", "real"), ("control_after", "real"))


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def index_by_id(rows):
    """Reject duplicate records rather than silently overriding a prediction."""
    indexed = {}
    for row in rows:
        identity, answer = row.get("id"), row.get("answer")
        if not isinstance(identity, str) or not identity or not isinstance(answer, str):
            raise ValueError("Prediction ID and answer must be strings")
        if identity in indexed:
            raise ValueError("Duplicate fact ID; pairing is ambiguous")
        if row.get("em") not in (0, 1, 0.0, 1.0):
            raise ValueError("Exact-match scores must be binary")
        indexed[identity] = row
    if not indexed:
        raise ValueError("Cannot compare an empty set of facts")
    return indexed


def matched_rows(first, second):
    """Require complete ID/label equality; never compare only an intersection."""
    left, right = index_by_id(first), index_by_id(second)
    if left.keys() != right.keys():
        raise ValueError("Fact ID sets differ; intersection-only pairing is forbidden")
    identities = sorted(left)
    if any(left[identity]["answer"] != right[identity]["answer"] for identity in identities):
        raise ValueError("Answer labels differ for matched fact IDs")
    return [left[identity] for identity in identities], [right[identity] for identity in identities]


def percentile(ordered, q):
    rank = (len(ordered) - 1) * q
    lower, upper = math.floor(rank), math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def paired_bootstrap(first, second, samples=10000, seed=123):
    """Fact-paired percentile CI for EM(first) - EM(second)."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("bootstrap samples must be a positive integer")
    left, right = matched_rows(first, second)
    differences = [a["em"] - b["em"] for a, b in zip(left, right)]
    rng, count = random.Random(seed), len(differences)
    distribution = sorted(sum(rng.choices(differences, k=count)) / count for _ in range(samples))
    identity_digest = hashlib.sha256(json.dumps(
        [[row["id"], row["answer"]] for row in left], ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()
    return dict(n_paired_facts=count, delta_em=sum(differences)/count,
                ci95=[percentile(distribution, .025), percentile(distribution, .975)],
                first_correct=sum(row["em"] for row in left),
                second_correct=sum(row["em"] for row in right),
                paired_id_label_sha256=identity_digest, bootstrap_samples=samples,
                bootstrap_seed=seed, unit="fact ID",
                scope="Conditional fact-sampling uncertainty; excludes training-seed uncertainty")


def phase_summary(rows):
    index_by_id(rows)
    if any(not finite(row.get("nll_sum")) or not isinstance(row.get("tokens"), int)
           or row["tokens"] <= 0 for row in rows):
        raise ValueError("Every prediction requires finite NLL and positive answer-token count")
    tokens = sum(row["tokens"] for row in rows)
    result = dict(n=len(rows), correct=sum(int(row["em"]) for row in rows),
                  em=sum(row["em"] for row in rows)/len(rows),
                  answer_token_nll=sum(row["nll_sum"] for row in rows)/tokens,
                  answer_tokens=tokens)
    routed = [row for row in rows if row.get("expected_in_bank") is True
              and row["method"] in ("real", "shuffled")]
    if routed:
        if any(not isinstance(row.get("selected_ids"), list) for row in routed):
            raise ValueError("Missing retrieved IDs in a routed prediction")
        result.update(retrieval_n=len(routed),
                      recall_at_1=sum(row["selected_ids"][:1] == [row["id"]] for row in routed)/len(routed),
                      recall_at_4=sum(row["id"] in row["selected_ids"][:4] for row in routed)/len(routed))
    decoded = [row for row in routed if row.get("decode_correct_residency") is not None]
    if decoded:
        if any(not isinstance(row.get("decode_query_count"), int) or row["decode_query_count"] <= 0
               or not isinstance(row.get("decode_correct_hits"), int)
               or not 0 <= row["decode_correct_hits"] <= row["decode_query_count"] for row in decoded):
            raise ValueError("Invalid decode retrieval counters")
        count = sum(row["decode_query_count"] for row in decoded)
        result.update(decode_correct_residency=sum(row["decode_correct_hits"] for row in decoded)/count,
                      decode_query_count=count)
    traced = [row for row in rows if row.get("switch_count") is not None]
    if traced:
        if any(not isinstance(row["switch_count"], int) or not isinstance(row.get("retrieval_transition_count"), int)
               or not 0 <= row["switch_count"] <= row["retrieval_transition_count"] for row in traced):
            raise ValueError("Invalid retrieval switch counters")
        switches = sum(row["switch_count"] for row in traced)
        transitions = sum(row["retrieval_transition_count"] for row in traced)
        result.update(switch_count_mean=switches/len(traced), switch_count_total=switches,
                      retrieval_transition_count=transitions,
                      top1_switch_rate=switches/transitions if transitions else None)
    return result


def metric_phase(metrics, phase, method):
    if phase in ("final", "paraphrase"):
        return metrics.get(phase, {}).get(method, {})
    return metrics.get(phase, {})


def provenance(manifest):
    cache = manifest.get("cache_sha256")
    source = manifest.get("source_files_sha256")
    model = manifest.get("model_revision")
    if not isinstance(cache, str) or not SHA256.fullmatch(cache):
        raise ValueError("Missing/invalid feature-cache SHA256")
    if not isinstance(source, dict) or not source or any(
        not isinstance(k, str) or not isinstance(v, str) or not SHA256.fullmatch(v)
        for k, v in source.items()
    ):
        raise ValueError("Missing/invalid runtime-source SHA256 mapping")
    if not isinstance(model, str) or not model:
        raise ValueError("Missing model revision")
    source_digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    combined = hashlib.sha256(json.dumps([cache, source_digest, model]).encode()).hexdigest()
    return dict(cache_sha256=cache, runtime_source_sha256=source_digest,
                model_revision=model, source_cache_group_sha256=combined)


def public_label(path, root):
    # Directory names only; never include machine paths, CLI commands, or examples.
    label = path.relative_to(root).as_posix()
    return label if re.fullmatch(r"[A-Za-z0-9_.\-/]+", label) else "run_" + hashlib.sha256(label.encode()).hexdigest()[:12]


def load_run(directory, root):
    manifest = json_read(directory / "manifest.json")
    if manifest.get("complete") is not True:
        raise ValueError("Run manifest is not complete")
    config = json_read(directory / "config.json")
    if config.get("train_size") == 32 or any("smoke" in path.name.casefold()
                                           for path in (directory, directory.parent)):
        raise ValueError("Smoke run excluded (independent evaluation entities, seed 8042)")
    if config.get("variant") not in ("raw", "stable"):
        raise ValueError("Unknown scaling variant")
    for key in ("train_size", "updates", "batch_size", "eval_size", "seed", "cold_bank_size",
                "alignment_steps", "oracle_updates"):
        if isinstance(config.get(key), bool) or not isinstance(config.get(key), int) or config[key] < 0:
            raise ValueError("Missing/invalid scaling configuration: " + key)
    if config["train_size"] not in (128, 1024, 4096) or config["eval_size"] < 1:
        raise ValueError("Not a main scaling experiment")
    metrics = json_read(directory / "metrics.json")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Missing completed metrics")
    groups = defaultdict(list)
    for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups[(row.get("phase"), row.get("method"))].append(row)
    summaries = {}
    reference = groups[("final", "real")]
    for phase, method in PHASES:
        rows = groups[(phase, method)]
        value = phase_summary(rows)
        reported = metric_phase(metrics, phase, method)
        if reported.get("count") != value["n"] or not finite(reported.get("em")) or not math.isclose(
            reported["em"], value["em"], abs_tol=1e-9
        ):
            raise ValueError(f"Predictions and reported metrics disagree for {phase}/{method}")
        for key in ("answer_token_nll", "recall_at_1", "recall_at_4", "decode_correct_residency", "switch_count_mean"):
            if key in value and (not finite(reported.get(key)) or not math.isclose(
                reported[key], value[key], rel_tol=1e-6, abs_tol=1e-8
            )):
                raise ValueError(f"Reported {key} does not match predictions for {phase}/{method}")
        if not phase.startswith("control_"):
            if value["n"] != config["eval_size"]:
                raise ValueError(f"Incomplete evaluation phase: {phase}/{method}")
            matched_rows(reference, rows)
        summaries[f"{phase}/{method}"] = value
    matched_rows(groups[("control_before", "real")], groups[("control_after", "real")])
    origin = provenance(manifest)
    suite_manifest_path = directory.parent / "suite.json"
    if suite_manifest_path.exists():
        suite = json_read(suite_manifest_path)
        if suite.get("profile") == "smoke":
            raise ValueError("Smoke suite excluded")
        if suite.get("cache_sha256") != origin["cache_sha256"]:
            raise ValueError("Suite/run cache hashes differ")
        jobs = [job for job in suite.get("jobs", []) if job.get("name") == directory.name]
        if jobs and any(job.get("status") != "complete" or job.get("exit_code") != 0 for job in jobs):
            raise ValueError("Suite job has not recorded successful completion")
    # When the immutable snapshot is locally available, verify its runtime
    # modules against the hashes rather than trusting only the manifest text.
    source_root = directory.parent / "source" / "src" / "vera_mem"
    if source_root.exists():
        for name, expected in manifest["source_files_sha256"].items():
            if Path(name).name != name:
                raise ValueError("Unexpected runtime-source filename")
            path = source_root / name
            if not path.is_file() or digest_file(path) != expected:
                raise ValueError("Runtime source snapshot hash mismatch")
    eval_only = bool(config.get("checkpoint"))
    checkpoint_hash = manifest.get("checkpoint_sha256") if eval_only else None
    if not eval_only and (directory / "best.pt").is_file():
        checkpoint_hash = digest_file(directory / "best.pt")
    selected_step = manifest.get("selected_step")
    if not isinstance(selected_step, int) or selected_step < 1:
        raise ValueError("Missing selected checkpoint step")
    source_training = manifest.get("source_training_config") if eval_only else config
    if source_training is not None and not isinstance(source_training, dict):
        raise ValueError("Invalid source training configuration")
    training_cold = (source_training.get("cold_bank_size") if source_training else
                     manifest.get("training_cold_bank_size")) if eval_only else config["cold_bank_size"]
    if not isinstance(training_cold, int):
        raise ValueError("Evaluation checkpoint training cold-bank size is unknown")
    for name in ("value_effective_rank", "value_std_mean", "value_abs_max"):
        if not finite(metrics.get(name)):
            raise ValueError("Missing/invalid vector diagnostic: " + name)
    if not isinstance(metrics.get("value_unique_rows"), int):
        raise ValueError("Missing value uniqueness diagnostic")
    if metrics.get("cold_records") != config["cold_bank_size"]:
        raise ValueError("Deployment cold-bank size differs from metrics")
    record = dict(run=public_label(directory, root), variant=config["variant"],
                  train_size=config["train_size"], training_seed=config["seed"],
                  selected_step=selected_step, training_cold_records=training_cold,
                  deployment_cold_records=config["cold_bank_size"],
                  evaluation_only=eval_only, new_lm_updates=0 if eval_only else config["updates"],
                  training_lm_updates=source_training.get("updates") if source_training else None,
                  batch_size=(source_training or config)["batch_size"],
                  alignment_steps=(source_training or config)["alignment_steps"],
                  oracle_updates=(source_training or config)["oracle_updates"], eval_size=config["eval_size"],
                  checkpoint_sha256=checkpoint_hash, phases=summaries, **origin,
                  source_training_config_sha256=hashlib.sha256(json.dumps(
                      source_training, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                  if source_training is not None else None,
                  vectors={key: metrics[key] for key in (
                      "value_effective_rank", "value_unique_rows", "value_std_mean", "value_abs_max")},
                  numeric_vdb_bytes=metrics.get("numeric_vdb_bytes"), records=metrics.get("records"),
                  online_gradient_steps=metrics.get("online_gradient_steps"),
                  shared_parameters_unchanged=metrics.get("shared_parameters_unchanged"),
                  source_snapshot_verified=source_root.exists(),
                  evidence_sha256={name: digest_file(directory / name) for name in (
                      "config.json", "manifest.json", "metrics.json", "predictions.jsonl")})
    return record, groups


def cross_run_compatible(first, second):
    """Permit size contrasts only with matched data, code, training and deployment."""
    for key in ("source_cache_group_sha256", "variant", "training_seed", "training_lm_updates",
                "batch_size", "alignment_steps", "oracle_updates", "training_cold_records",
                "deployment_cold_records", "eval_size"):
        if first.get(key) is None or first.get(key) != second.get(key):
            raise ValueError("Cross-run matching failed: " + key)
    if first.get("evaluation_only") or second.get("evaluation_only"):
        raise ValueError("Size contrasts require separately trained runs")


def cold_deployment_compatible(cold, empty):
    """Isolate changing deployment bank size while keeping exactly one model."""
    if cold.get("deployment_cold_records") != 128 or empty.get("deployment_cold_records") != 0:
        raise ValueError("Deployment contrast requires cold 128 minus cold 0")
    for key in ("checkpoint_sha256", "source_training_config_sha256"):
        identity = cold.get(key)
        if not isinstance(identity, str) or not SHA256.fullmatch(identity) or identity != empty.get(key):
            raise ValueError("Same-checkpoint deployment matching failed: " + key)
    for key in ("source_cache_group_sha256", "variant", "train_size", "training_seed",
                "training_lm_updates", "batch_size", "alignment_steps", "oracle_updates",
                "training_cold_records", "selected_step", "eval_size"):
        if cold.get(key) is None or cold[key] != empty.get(key):
            raise ValueError("Same-checkpoint deployment matching failed: " + key)


def long_budget_compatible(long, short):
    """Compare 3x LM schedules, not equal-FLOPs or a pure extra-step intervention."""
    if long.get("evaluation_only") or short.get("evaluation_only"):
        raise ValueError("Budget contrasts require separately trained runs")
    if long.get("variant") != "stable" or short.get("variant") != "stable" or any(
        row.get("train_size") != 4096 for row in (long, short)
    ):
        raise ValueError("Budget contrast is restricted to stable N4096")
    if long.get("training_lm_updates") != 1536 or short.get("training_lm_updates") != 512:
        raise ValueError("Budget contrast requires 1536 minus 512 LM updates")
    for key in ("source_cache_group_sha256", "training_seed", "batch_size", "alignment_steps",
                "training_cold_records", "deployment_cold_records", "eval_size"):
        if long.get(key) is None or long[key] != short.get(key):
            raise ValueError("Training-budget matching failed: " + key)
    if long["alignment_steps"] != 400:
        raise ValueError("Budget contrast requires 400 alignment steps in both runs")
    if (not isinstance(long.get("oracle_updates"), int)
            or not isinstance(short.get("oracle_updates"), int)
            or not 0 <= long["oracle_updates"] <= 1536
            or not 0 <= short["oracle_updates"] <= 512
            or long["oracle_updates"] * 512 != short["oracle_updates"] * 1536):
        raise ValueError("Training-budget matching failed: oracle fraction differs")


def collect(runs_root, samples=10000, seed=123):
    root = runs_root.resolve()
    records, predictions, skipped = [], {}, []
    if not root.is_dir():
        raise ValueError("runs-root must be an existing directory")
    suites = [root] if root.name.startswith("scaling_") else sorted(root.glob("scaling_*"))
    candidates = sorted({path.parent for suite in suites if suite.is_dir()
                         for path in suite.glob("*/config.json") if path.parent.name != "source"})
    for directory in candidates:
        try:
            record, groups = load_run(directory, root)
        except (ValueError, OSError, TypeError, KeyError) as error:
            # Error messages intentionally avoid raw predictions and fact IDs.
            reason = str(error) if isinstance(error, ValueError) else type(error).__name__
            skipped.append(dict(run=public_label(directory, root), reason=reason))
            continue
        records.append(record)
        predictions[record["run"]] = groups
    for record in records:
        if record["evaluation_only"]:
            parents = [other for other in records if not other["evaluation_only"]
                       and other["checkpoint_sha256"] and other["checkpoint_sha256"] == record["checkpoint_sha256"]]
            if len(parents) == 1:
                parent = parents[0]
                if record["training_lm_updates"] is None:
                    record["training_lm_updates"] = parent["training_lm_updates"]
                record["training_run"] = parent["run"]
                for key in ("variant", "train_size", "training_seed", "training_cold_records",
                            "selected_step", "training_lm_updates", "batch_size", "alignment_steps", "oracle_updates"):
                    if record[key] != parent[key]:
                        raise ValueError("Checkpoint source configuration conflicts with its matching training run")
                if record["source_training_config_sha256"] is None:
                    record["source_training_config_sha256"] = parent["source_training_config_sha256"]
    comparisons, refused = [], []
    for record in records:
        groups = predictions[record["run"]]
        for other in ("empty", "shuffled"):
            comparisons.append(dict(
                contrast="final_real_minus_" + other, first_run=record["run"],
                second_run=record["run"], phase="final", first_method="real", second_method=other,
                **paired_bootstrap(groups[("final", "real")], groups[("final", other)], samples, seed)))
    for large in records:
        if large["train_size"] != 4096 or large["evaluation_only"]:
            continue
        for small in records:
            if small["train_size"] != 128 or small["evaluation_only"] or small["variant"] != large["variant"]:
                continue
            try:
                cross_run_compatible(large, small)
                estimate = paired_bootstrap(predictions[large["run"]][("final", "real")],
                                            predictions[small["run"]][("final", "real")], samples, seed)
            except ValueError as error:
                refused.append(dict(first_run=large["run"], second_run=small["run"], reason=str(error)))
                continue
            comparisons.append(dict(contrast="N4096_minus_N128_matched_updates", first_run=large["run"],
                                    second_run=small["run"], phase="final", first_method="real",
                                    second_method="real", **estimate))
    for cold in records:
        if cold["deployment_cold_records"] != 128:
            continue
        for empty in records:
            if empty["deployment_cold_records"] != 0 or any(
                cold[key] != empty[key] for key in ("variant", "train_size", "training_seed", "training_cold_records")
            ):
                continue
            try:
                cold_deployment_compatible(cold, empty)
                estimate = paired_bootstrap(predictions[cold["run"]][("final", "real")],
                                            predictions[empty["run"]][("final", "real")], samples, seed)
            except ValueError as error:
                refused.append(dict(contrast="deployment_cold128_minus_0_same_checkpoint",
                                    first_run=cold["run"], second_run=empty["run"], reason=str(error)))
                continue
            comparisons.append(dict(contrast="deployment_cold128_minus_0_same_checkpoint",
                                    first_run=cold["run"], second_run=empty["run"], phase="final",
                                    first_method="real", second_method="real",
                                    training_cold_records=cold["training_cold_records"],
                                    checkpoint_sha256=cold["checkpoint_sha256"],
                                    interpretation="Deployment initialization only; same trained checkpoint and source training configuration",
                                    **estimate))
    for long in records:
        if (long["variant"] != "stable" or long["train_size"] != 4096
                or long["evaluation_only"] or long["training_lm_updates"] != 1536):
            continue
        for short in records:
            if (short["variant"] != "stable" or short["train_size"] != 4096
                    or short["evaluation_only"] or short["training_lm_updates"] != 512):
                continue
            try:
                long_budget_compatible(long, short)
                estimate = paired_bootstrap(predictions[long["run"]][("final", "real")],
                                            predictions[short["run"]][("final", "real")], samples, seed)
            except ValueError as error:
                refused.append(dict(contrast="stable4096_LMschedule1536_minus_512",
                                    first_run=long["run"], second_run=short["run"], reason=str(error)))
                continue
            comparisons.append(dict(contrast="stable4096_LMschedule1536_minus_512",
                                    first_run=long["run"], second_run=short["run"], phase="final",
                                    first_method="real", second_method="real",
                                    lm_schedule_ratio=3,
                                    interpretation="Threefold LM update/exposure schedule with the same oracle fraction; not equal FLOPs and not threefold total training compute",
                                    **estimate))
    provenance_groups = sorted({record["source_cache_group_sha256"] for record in records})
    for record in records:
        record["comparison_group"] = "G" + str(provenance_groups.index(record["source_cache_group_sha256"]) + 1)
    return dict(generated_at=datetime.now(timezone.utc).isoformat(), runs=records,
                paired_comparisons=comparisons, refused_comparisons=refused, skipped_runs=skipped,
                completed_run_count=len(records), source_cache_group_count=len(provenance_groups),
                bootstrap_samples=samples, bootstrap_seed=seed,
                limitations=[
                    "Only complete runs with consistent per-example evidence enter the tables.",
                    "Smoke runs use separate seed-8042 entities and are excluded.",
                    "Within-run controls share facts; cross-size comparisons require identical cache/source/model and matched update budgets.",
                    "Percentile CIs are conditional fact-sampling uncertainty; one training seed cannot estimate training randomness.",
                    "No multiplicity correction or population-level method-superiority claim is made.",
                    "Pre-write and immediate phases measure real retrieval only; other methods were not measured at those phases.",
                    "The empty condition means zero memory residual, not necessarily an empty allocated database.",
                    "Finite-answer-vocabulary shuffled values can accidentally retain the same answer category.",
                    "Oracle is a forced-correct-value diagnostic, not a mathematical upper bound: one correct value is injected at every token, changing the distribution learned under per-token sparse retrieval.",
                    "Primary memory evidence compares real retrieval with same-checkpoint shuffled and zero-residual controls; real EM may exceed forced-correct-value EM.",
                    "Training cold records and deployment cold records are distinct; evaluation-only rows perform zero new training steps.",
                    "Deployment cold128-minus-0 contrasts require identical checkpoint SHA and source training configuration; separately cold-trained models are not a same-checkpoint deployment intervention.",
                    "The 1536-minus-512 contrast scales the whole LM schedule, including oracle and real-retrieval stages, with 400 alignment steps fixed; it is not an equal-FLOPs comparison or a threefold-total-compute claim.",
                ])


def em_cell(row, phase, method="real"):
    value = row["phases"].get(f"{phase}/{method}")
    return "—" if value is None else f"{value['correct']}/{value['n']} ({value['em']:.1%})"


def render(summary):
    lines = ["# Scaling and cold-start experiment summary", "", "Generated at (UTC): " + summary["generated_at"], "",
             "List only completed runs that pass prediction-level verification. Incomplete runs or inconsistent evidence are excluded; independent smoke-test entities do not enter the main tables.", "",
             "EM tables show correct/total facts in controlled synthetic experiments. CIs describe fact-sampling variability conditional on a fixed model, not training randomness.", "",
             "The primary measure contrasts real per-token retrieval with shuffled and empty controls from the same checkpoint. "
             "The forced-correct-value diagnostic (data field oracle) injects one correct value at every token, changing the distribution established by real sparse reading. "
             "It is not a mathematical upper bound; real-retrieval EM can exceed it. A low diagnostic score alone cannot identify inadequate readout as the main bottleneck, "
             "or establish whether per-token retrieval errors have compensating effects; further ablations are needed.", "",
             "## Data scale, training budget, and memory performance", "",
             "LM steps gives the source training update budget; parentheses show new updates in the current run. Selected is the checkpoint step actually chosen on development data.", "",
             "| Run / group | Variant | N | LM steps (new) | selected | cold train/deploy | pre real | immediate real | final real | Forced-value diagnostic | shuffled | empty |",
             "| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary["runs"]:
        budget = row["training_lm_updates"]
        lines.append("| " + " | ".join(map(str, [
            row["run"] + " / " + row["comparison_group"], row["variant"], row["train_size"],
            f"{budget if budget is not None else 'unknown'} ({row['new_lm_updates']})", row["selected_step"],
            f"{row['training_cold_records']}/{row['deployment_cold_records']}",
            em_cell(row, "pre_write"), em_cell(row, "immediate"),
            *(em_cell(row, "final", method) for method in METHODS),
        ])) + " |")
    if not summary["runs"]:
        lines += ["", "No completed runs currently meet the inclusion criteria. No estimates or placeholder results are shown."]
    lines += ["", "pre/immediate did not measure Forced-value diagnostic, shuffled, or empty; missing measurements are not zeros.", "",
              "## Addressing, paraphrases, controls, and value distributions", "",
              "| Run | final R@1 / R@4 | decode residency / switches per example | paraphrase real / Forced-value diagnostic | control before / after | value effective rank | unique values / records |",
              "| --- | --- | --- | --- | --- | ---: | --- |"]
    for row in summary["runs"]:
        final = row["phases"]["final/real"]
        recall = " / ".join(f"{final[key]:.1%}" if key in final else "—" for key in ("recall_at_1", "recall_at_4"))
        lines.append("| " + " | ".join([
            row["run"], recall,
            (f"{final['decode_correct_residency']:.1%}" if "decode_correct_residency" in final else "—") + " / "
            + (f"{final['switch_count_mean']:.2f}" if "switch_count_mean" in final else "—"),
            em_cell(row, "paraphrase") + " / " + em_cell(row, "paraphrase", "oracle"),
            em_cell(row, "control_before") + " / " + em_cell(row, "control_after"),
            f"{row['vectors']['value_effective_rank']:.2f}",
            f"{row['vectors']['value_unique_rows']} / {row['records']}",
        ]) + " |")
    lines += ["", "R@k uses the final prompt token before actual generation. Decode residency is the fraction of decode queries whose top-4 includes the correct record, weighted by actual query counts; it is not a top-1 proportion or attention weight. Switches count top-1 changes per example, including prefill to first decode. Historical runs without traces show a dash. Controls are unwritten random facts; their EM is not abstention ability. Value effective rank and uniqueness measure numerical diversity, not memory usefulness by themselves.", "",
              "## Strictly fact-paired differences", "",
              "Differences are first minus second, in percentage points. ID sets and labels must match exactly; no intersection-only comparisons.", "",
              "`deployment_cold128_minus_0_same_checkpoint` changes only the initial deployment bank and requires identical checkpoint SHA and source training configuration, "
              "while keeping training cold=0/128 conditions separate. Models from different cold-training conditions are not treated as same-weight deployment controls. "
              "`stable4096_LMschedule1536_minus_512` compares a 3x LM schedule: oracle and real-retrieval stages scale together, while alignment remains at 400 updates. "
              "This is a combined training-budget comparison, not an equal-FLOPs comparison; total compute is not established to be exactly 3x.", "",
              "| Contrast | first | second | paired N | Δ EM (pp) | 95% percentile CI (pp) |",
              "| --- | --- | --- | ---: | ---: | --- |"]
    for row in summary["paired_comparisons"]:
        lines.append(f"| {row['contrast']} | {row['first_run']} | {row['second_run']} | {row['n_paired_facts']} | "
                     f"{100*row['delta_em']:+.2f} | [{100*row['ci95'][0]:+.2f}, {100*row['ci95'][1]:+.2f}] |")
    lines += ["", f"Bootstrap: {summary['bootstrap_samples']:,} resamples, seed {summary['bootstrap_seed']}. "
              "No multiple-comparison correction is applied; intervals from one training seed do not establish significance across training randomness.",
              "", "## Comparability and exclusions", "",
              "Matching G group IDs indicate the same feature cache, runtime source, and model revision; cross-group differences do not establish causal effects of data scale."]
    for group in sorted({row["comparison_group"] for row in summary["runs"]}):
        row = next(row for row in summary["runs"] if row["comparison_group"] == group)
        lines += ["", f"- {group}: cache `{row['cache_sha256']}`; source `{row['runtime_source_sha256']}`; model `{row['model_revision']}`."]
    for row in summary["refused_comparisons"]:
        lines += ["", f"- Not paired {row['first_run']} and {row['second_run']}: {row['reason']}."]
    for row in summary["skipped_runs"]:
        lines += ["", f"- Excluded {row['run']}: {row['reason']}."]
    unresolved = [row["run"] for row in summary["runs"] if row["evaluation_only"] and row["training_lm_updates"] is None]
    if unresolved:
        lines += ["", "For these evaluation-only runs, no local source checkpoint with matching SHA was found; the source training budget is unknown: " + ", ".join(unresolved) + "."]
    lines += ["", "This public JSON export contains aggregate results, evidence hashes, and control relationships, but no per-example answers/predictions, internal absolute paths, model weights, or connection settings.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output directory for report.md and summary.json")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    result = collect(args.runs_root, args.bootstrap_samples, args.bootstrap_seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render(result), encoding="utf-8")
    print(json.dumps(dict(completed_runs=result["completed_run_count"],
                          excluded_runs=len(result["skipped_runs"]),
                          paired_comparisons=len(result["paired_comparisons"]))))


if __name__ == "__main__":
    main()
