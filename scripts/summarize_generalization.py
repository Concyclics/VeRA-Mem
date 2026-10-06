"""Audit completed generalization runs and publish aggregates without raw examples.

Bootstrap units are entities: the three confirmation query views are retained
together in every resample. No model, GPU, or third-party package is required.
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
import unicodedata

METHODS = ("real", "oracle", "shuffled", "empty")
BANKS = ("canonical_support", "heldout_support")
CANONICAL_QUERY = "train_query_00"
HELDOUT_QUERIES = ("test_query_00", "test_query_01", "test_query_02")
QUERIES = (CANONICAL_QUERY, *HELDOUT_QUERIES)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
BUDGET_KEYS = ("train_size", "updates", "oracle_updates", "batch_size",
               "alignment_steps", "validate_every", "eval_size", "seed")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def normalize_answer(value):
    if not isinstance(value, str):
        raise ValueError("Prediction and answer must be strings")
    text = unicodedata.normalize("NFKC", value).casefold()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


def labels(rows):
    result = {}
    for row in rows:
        identity, answer = row.get("id"), row.get("answer")
        if not isinstance(identity, str) or not identity or not isinstance(answer, str):
            raise ValueError("Fact IDs and answers must be nonempty ID/string label pairs")
        if identity in result:
            raise ValueError("Duplicate fact ID within one evaluation phase")
        result[identity] = answer
    if not result:
        raise ValueError("Empty evaluation phase")
    return result


def summarize_phase(rows):
    labels(rows)
    for row in rows:
        if row.get("em") not in (0, 1) or isinstance(row.get("em"), bool):
            raise ValueError("Exact-match score must be numeric zero or one")
        if row["em"] != int(normalize_answer(row.get("prediction")) == normalize_answer(row["answer"])):
            raise ValueError("Stored EM disagrees with recomputed prediction/answer match")
        if (not finite(row.get("nll_sum")) or row["nll_sum"] < 0
                or not isinstance(row.get("tokens"), int) or isinstance(row["tokens"], bool) or row["tokens"] < 1):
            raise ValueError("Invalid answer NLL/token evidence")
        if not isinstance(row.get("expected_in_bank"), bool):
            raise ValueError("Missing explicit observation eligibility")
        if not isinstance(row.get("selected_ids"), list) or any(not isinstance(x, str) for x in row["selected_ids"]):
            raise ValueError("Invalid retrieval evidence")
        if len(row["selected_ids"]) > 4 or len(set(row["selected_ids"])) != len(row["selected_ids"]):
            raise ValueError("Retrieval must contain at most four distinct records")
    result = dict(count=len(rows), correct=sum(int(row["em"]) for row in rows),
                  em=sum(row["em"] for row in rows)/len(rows),
                  answer_token_nll=sum(row["nll_sum"] for row in rows)/sum(row["tokens"] for row in rows))
    routed = [row for row in rows if row["expected_in_bank"] and row["method"] in ("real", "shuffled")]
    if routed:
        result.update(recall_at_1=sum(row["selected_ids"][:1] == [row["id"]] for row in routed)/len(routed),
                      recall_at_4=sum(row["id"] in row["selected_ids"] for row in routed)/len(routed))
    decoded = [row for row in routed if row.get("decode_correct_residency") is not None]
    if decoded:
        for row in decoded:
            hits, count = row.get("decode_correct_hits"), row.get("decode_query_count")
            if (not isinstance(count, int) or not isinstance(hits, int) or count < 1
                    or not 0 <= hits <= count or not finite(row["decode_correct_residency"])
                    or not math.isclose(row["decode_correct_residency"], hits/count, abs_tol=1e-9)):
                raise ValueError("Invalid per-token retrieval counters")
        count = sum(row["decode_query_count"] for row in decoded)
        result.update(decode_correct_residency=sum(row["decode_correct_hits"] for row in decoded)/count,
                      decode_query_count=count, decode_examples=len(decoded))
    traced = [row for row in rows if row.get("switch_count") is not None]
    if traced:
        for row in traced:
            switches, count = row["switch_count"], row.get("retrieval_transition_count")
            if not isinstance(switches, int) or not isinstance(count, int) or not 0 <= switches <= count:
                raise ValueError("Invalid retrieval switch counters")
        switches = sum(row["switch_count"] for row in traced)
        count = sum(row["retrieval_transition_count"] for row in traced)
        result.update(switch_count_mean=switches/len(traced), switch_count_total=switches,
                      retrieval_transition_count=count, top1_switch_rate=switches/count if count else None)
    return result


def verify_reported(actual, reported):
    if not isinstance(reported, dict):
        raise ValueError("Missing phase metrics")
    for key, value in actual.items():
        if key == "correct":  # Runner records mean EM and count, not integer correct.
            continue
        observed = reported.get(key)
        if value is None:
            if key not in reported or observed is not None:
                raise ValueError("Reported phase metric does not match predictions: " + key)
        elif not finite(observed) or not math.isclose(observed, value, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Reported phase metric does not match predictions: " + key)


def public_label(directory, root):
    relative = directory.relative_to(root).as_posix()
    return relative if re.fullmatch(r"[A-Za-z0-9_.\-/]+", relative) else "run_" + object_digest(relative)[:12]


def quadrants(groups):
    result = {}
    for bank in BANKS:
        for kind, templates in (("canonical_query", (CANONICAL_QUERY,)), ("heldout_query", HELDOUT_QUERIES)):
            key = kind + "/" + bank
            result[key] = {}
            for method in METHODS:
                selected = [groups[(bank + "/" + template, method)] for template in templates]
                per_template = [summarize_phase(rows) for rows in selected]
                result[key][method] = dict(
                    n_facts=len(selected[0]), n_templates=len(templates),
                    n_predictions=sum(len(rows) for rows in selected),
                    correct=sum(row["correct"] for row in per_template),
                    em=sum(row["em"] for row in per_template)/len(templates),
                    # Equal-sized template sets make macro and micro EM identical.
                    aggregation="equal-weight query-template macro average",
                )
                for metric in ("recall_at_1", "recall_at_4", "decode_correct_residency"):
                    if all(metric in row for row in per_template):
                        result[key][method][metric] = sum(row[metric] for row in per_template)/len(templates)
    return result


def load_run(directory, root):
    config = read_json(directory / "config.json")
    if config.get("train_size") == 32 or config.get("smoke_uses_dev_only") or "smoke" in directory.parent.name.lower():
        raise ValueError("Smoke excluded: it uses development facts and templates")
    if config.get("condition") not in ("canonical", "augment", "invariant"):
        raise ValueError("Unknown generalization training condition")
    for key in BUDGET_KEYS:
        if isinstance(config.get(key), bool) or not isinstance(config.get(key), int) or config[key] < 0:
            raise ValueError("Missing/invalid training budget: " + key)
    if config["eval_size"] != 128 or config["train_size"] != 4096 or config["updates"] < 1:
        raise ValueError("Main confirmation requires 4096 training and 128 evaluation facts")
    manifest, suite = read_json(directory / "manifest.json"), read_json(directory.parent / "suite.json")
    if manifest.get("complete") is not True or suite.get("complete") is not True or suite.get("status") != "complete":
        raise ValueError("Both run and suite must record completion")
    jobs = [job for job in suite.get("jobs", []) if job.get("name") == directory.name]
    if len(jobs) != 1 or jobs[0].get("status") != "complete" or jobs[0].get("exit_code") != 0:
        raise ValueError("Suite must contain one successfully completed matching job")
    if suite.get("profile") == "smoke":
        raise ValueError("Smoke suite excluded")
    if not isinstance(manifest.get("selected_step"), int) or not 1 <= manifest["selected_step"] <= config["updates"]:
        raise ValueError("Selected checkpoint step is invalid")
    for field in ("cache_sha256", "data_fingerprint"):
        if not isinstance(manifest.get(field), str) or not SHA256.fullmatch(manifest[field]):
            raise ValueError("Missing/invalid provenance: " + field)
    if suite.get("cache_sha256") != manifest["cache_sha256"]:
        raise ValueError("Suite and run cache hashes disagree")
    source = manifest.get("source_files_sha256")
    if not isinstance(source, dict) or not source:
        raise ValueError("Missing runtime source hashes")
    for filename, digest in source.items():
        if (not isinstance(filename, str) or Path(filename).name != filename
                or not isinstance(digest, str) or not SHA256.fullmatch(digest)):
            raise ValueError("Invalid runtime source hash mapping")
        snapshot = directory.parent / "source" / "src" / "vera_mem" / filename
        if not snapshot.is_file() or digest_file(snapshot) != digest:
            raise ValueError("Runtime source snapshot differs from the run manifest")
    if not isinstance(manifest.get("model_revision"), str) or not manifest["model_revision"]:
        raise ValueError("Missing model revision")
    metrics = read_json(directory / "metrics.json")
    if (metrics.get("eval_facts") != 128 or metrics.get("online_gradient_steps") != 0
            or metrics.get("shared_parameters_unchanged") is not True):
        raise ValueError("Online protocol was not confirmed as frozen and complete")
    expected = {("pre_write", "real"), ("immediate", "real"),
                ("control_before", "real"), ("control_after", "real")}
    expected.update((bank + "/" + template, method) for bank in BANKS for template in QUERIES for method in METHODS)
    groups = defaultdict(list)
    for row in read_rows(directory / "predictions.jsonl"):
        if not isinstance(row, dict):
            raise ValueError("Prediction rows must be JSON objects")
        groups[(row.get("phase"), row.get("method"))].append(row)
    if set(groups) != expected:
        raise ValueError("Missing or unexpected phase/method in prediction evidence")
    test_rows = read_rows(directory / "test.jsonl")[:config["eval_size"]]
    reference = labels(test_rows)
    if len(reference) != config["eval_size"]:
        raise ValueError("Confirmation examples are incomplete")
    control_reference = labels(read_rows(directory / "control.jsonl"))
    if len(control_reference) != 32 or set(control_reference) & set(reference):
        raise ValueError("Control examples must contain 32 disjoint never-written facts")
    summaries = {}
    for phase, method in sorted(expected):
        rows = groups[(phase, method)]
        control = phase.startswith("control_")
        phase_reference = control_reference if control else reference
        if len(rows) != len(phase_reference) or labels(rows) != phase_reference:
            raise ValueError("Phase fact ID/label sets differ from confirmation examples")
        should_be_written = not control and phase != "pre_write"
        if any(row.get("expected_in_bank") is not should_be_written for row in rows):
            raise ValueError("Observation eligibility violates pre-write/post-write protocol")
        actual = summarize_phase(rows)
        if phase in ("pre_write", "immediate", "control_before", "control_after"):
            reported = metrics.get("conditions", {}).get("canonical_support", {}).get(phase)
        else:
            bank, template = phase.split("/")
            reported = metrics.get("conditions", {}).get(bank, {}).get("scores", {}).get(template, {}).get(method)
        verify_reported(actual, reported)
        summaries[phase + "/" + method] = actual
    writes = read_json(directory / "writes.json")
    if not isinstance(writes, list) or len(writes) != 2 * config["eval_size"]:
        raise ValueError("Observation write log is incomplete")
    for bank in BANKS:
        entries = [row for row in writes if row.get("bank") == bank]
        if len(entries) != 128 or [row.get("id") for row in entries] != [row["id"] for row in test_rows]:
            raise ValueError("Observation write order or fact coverage differs")
        for index, entry in enumerate(entries):
            expected_support = "train_support_00" if bank == "canonical_support" else f"test_support_{index % 3:02d}"
            if entry.get("timestamp") != index or entry.get("support_template") != expected_support:
                raise ValueError("Observation write template or timestamp differs from protocol")
        if metrics.get("conditions", {}).get(bank, {}).get("records") != 128:
            raise ValueError("Final vector bank record count differs")
    origin = {field: manifest[field] for field in ("cache_sha256", "data_fingerprint", "model_revision")}
    origin["runtime_source_sha256"] = object_digest(source)
    record = dict(run=public_label(directory, root), condition=config["condition"], verified_complete=True,
                  configuration={key: config[key] for key in BUDGET_KEYS},
                  selected_step=manifest["selected_step"], **origin,
                  source_cache_group_sha256=object_digest(origin),
                  verified_prediction_count=sum(len(rows) for rows in groups.values()),
                  phases=summaries, quadrants=quadrants(groups))
    training_path = directory / "training.json"
    if training_path.is_file():
        history = read_json(training_path).get("training", [])
        if not history or history[-1].get("step") != config["updates"]:
            raise ValueError("Training history does not reach the declared update budget")
        record["training_prompt_tokens"] = history[-1].get("prompt_tokens_seen")
        record["training_seconds"] = history[-1].get("seconds")
    return record, groups


def entity_scores(groups, quadrant, method="real"):
    query_kind, bank = quadrant.split("/")
    templates = (CANONICAL_QUERY,) if query_kind == "canonical_query" else HELDOUT_QUERIES
    reference, per_entity = None, defaultdict(list)
    for template in templates:
        rows = groups[(bank + "/" + template, method)]
        current = labels(rows)
        if reference is not None and current != reference:
            raise ValueError("Within-entity query views have different ID/label sets")
        reference = current
        for row in rows:
            per_entity[row["id"]].append(row["em"])
    return {identity: (reference[identity], sum(scores)/len(scores)) for identity, scores in per_entity.items()}


def percentile(ordered, probability):
    rank = (len(ordered) - 1) * probability
    lower, upper = math.floor(rank), math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def paired_bootstrap(first, second, samples=10000, seed=123):
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    if not first or first.keys() != second.keys():
        raise ValueError("Paired contrasts require full fact ID equality, never intersections")
    ordered = sorted(first)
    if any(first[identity][0] != second[identity][0] for identity in ordered):
        raise ValueError("Paired fact labels disagree")
    differences = [first[identity][1] - second[identity][1] for identity in ordered]
    count, rng = len(differences), random.Random(seed)
    distribution = sorted(sum(rng.choices(differences, k=count))/count for _ in range(samples))
    return dict(n_paired_facts=count, delta_em=sum(differences)/count,
                ci95=[percentile(distribution, .025), percentile(distribution, .975)],
                bootstrap_samples=samples, bootstrap_seed=seed,
                paired_id_label_sha256=object_digest([[identity, first[identity][0]] for identity in ordered]),
                unit="fact ID; all query-template views remain in the same cluster",
                scope="Conditional fact-sampling uncertainty, excluding training-seed uncertainty")


def cross_run_compatible(first, second):
    if first["source_cache_group_sha256"] != second["source_cache_group_sha256"]:
        raise ValueError("Source/cache/model/data provenance differs")
    if first["configuration"] != second["configuration"]:
        raise ValueError("Training seed, update schedule, or fact budget differs")


def collect(runs_root, samples=10000, seed=123):
    root = runs_root.resolve()
    if not root.is_dir():
        raise ValueError("runs-root must be an existing directory")
    suites = [root] if root.name.startswith("generalization_") else sorted(root.glob("generalization_*"))
    directories = sorted({path.parent for suite in suites if suite.is_dir()
                          for path in suite.glob("*/config.json") if path.parent.name != "source"})
    records, predictions, skipped = [], {}, []
    for directory in directories:
        try:
            record, groups = load_run(directory, root)
        except (ValueError, OSError, TypeError, KeyError) as error:
            reason = str(error) if isinstance(error, ValueError) else type(error).__name__
            skipped.append(dict(run=public_label(directory, root), reason=reason))
            continue
        records.append(record)
        predictions[record["run"]] = groups
    comparisons, refused, memory_effects = [], [], []
    for record in records:
        groups = predictions[record["run"]]
        for quadrant, scores in record["quadrants"].items():
            real = entity_scores(groups, quadrant, "real")
            for control_method in ("shuffled", "empty"):
                estimate = paired_bootstrap(real, entity_scores(groups, quadrant, control_method), samples, seed)
                memory_effects.append(dict(
                    run=record["run"], condition=record["condition"], quadrant=quadrant,
                    contrast="real_minus_" + control_method,
                    first_method="real", second_method=control_method,
                    first_em=scores["real"]["em"], second_em=scores[control_method]["em"],
                    interpretation="Same-checkpoint paired memory intervention; a between-training-condition gain alone is insufficient evidence of memory use",
                    **estimate,
                ))
    for first_condition, second_condition in (("augment", "canonical"), ("invariant", "canonical"), ("invariant", "augment")):
        for first in [record for record in records if record["condition"] == first_condition]:
            for second in [record for record in records if record["condition"] == second_condition]:
                contrast = dict(contrast=first_condition + "_minus_" + second_condition,
                                first_run=first["run"], second_run=second["run"])
                try:
                    cross_run_compatible(first, second)
                    estimates = []
                    for quadrant in first["quadrants"]:
                        estimate = paired_bootstrap(entity_scores(predictions[first["run"]], quadrant),
                                                    entity_scores(predictions[second["run"]], quadrant), samples, seed)
                        estimates.append(dict(contrast, quadrant=quadrant, **estimate))
                    comparisons.extend(estimates)
                except ValueError as error:
                    refused.append(dict(contrast, reason=str(error)))
    return dict(generated_at=datetime.now(timezone.utc).isoformat(), schema_version=1,
                completed_run_count=len(records), runs=records, paired_comparisons=comparisons,
                within_run_memory_effects=memory_effects,
                refused_comparisons=refused, skipped_runs=skipped, bootstrap_samples=samples,
                bootstrap_seed=seed, limitations=[
                    "Only completed suites and runs with prediction-level verification enter summaries.",
                    "Held-out query EM is a macro average over XML, CSV, and dialogue templates; CI resamples facts, not views.",
                    "Held-out support is one of three formats per fact, round-robin; it is not all nine query/support combinations per fact.",
                    "Equal optimizer updates/exposures do not imply equal prompt-token counts or FLOPs; paired-view consistency adds work.",
                    "Single-seed confidence intervals exclude training randomness and have no multiplicity correction.",
                    "Finite 16-word synthetic labels and authored format stress tests are not a natural-language benchmark.",
                    "Uniform guessing among 16 labels has expected EM 6.25%; emitting any fixed vocabulary word also scores 6.25% on the balanced confirmation facts. Improvement toward this level need not indicate fact-specific memory use.",
                    "Memory-use evidence compares real reads against both shuffled values and zero-residual reads within the same checkpoint and facts, separately from between-condition training gains; a positive real-minus-empty effect alone may reflect output-vocabulary adaptation.",
                    "The historical failed paraphrase is now a training view, not confirmation evidence.",
                    "Forced-correct-value oracle changes the per-token read distribution and is not a mathematical upper bound.",
                    "Empty means zero memory residual; shuffled values can accidentally preserve the answer category.",
                ])


def render(summary):
    lines = ["# Template generalization experiment summary", "", "Generated at (UTC): " + summary["generated_at"], "",
             "Include only completed suites/runs with predictions consistent with statistics. The four quadrants independently vary query and observation formats.",
             "Heldout-query columns equally average XML, CSV, and dialogue templates. The three views refer to the same facts and are not independent samples.", "",
             "| Condition | selected / updates | canonical query + canonical support | heldout query + canonical support | canonical query + heldout support | heldout query + heldout support |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    order = ("canonical_query/canonical_support", "heldout_query/canonical_support",
             "canonical_query/heldout_support", "heldout_query/heldout_support")
    for record in summary["runs"]:
        cells = [record["condition"], f"{record['selected_step']} / {record['configuration']['updates']}"]
        cells.extend(f"{record['quadrants'][key]['real']['em']:.1%}" for key in order)
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "Real retrieval, forced-correct-value, shuffled-value, and zero-residual diagnostics for each quadrant:", "",
                  "| Condition | Quadrant | Real | Forced value | Shuffled | Empty |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for record in summary["runs"]:
        for key in order:
            lines.append("| " + " | ".join([record["condition"], key, *[f"{record['quadrants'][key][method]['em']:.1%}" for method in METHODS]]) + " |")
    lines.extend(["", "Heldout query templates, using real retrieval:", "", "| Condition | Support bank | Query template | EM | Recall@4 |",
                  "| --- | --- | --- | ---: | ---: |"])
    for record in summary["runs"]:
        for bank in BANKS:
            for template in HELDOUT_QUERIES:
                phase = record["phases"][bank + "/" + template + "/real"]
                lines.append(f"| {record['condition']} | {bank} | {template} | {phase['em']:.1%} | {phase.get('recall_at_4', 0):.1%} |")
    lines.extend(["", "Paired differences with 95% bootstrap CIs, in percentage points, resampling facts:", "",
                  "| Contrast | Quadrant | Δ EM (pp) | 95% CI (pp) | Facts |",
                  "| --- | --- | ---: | --- | ---: |"])
    for row in summary["paired_comparisons"]:
        lo, hi = row["ci95"]
        lines.append(f"| {row['contrast']} | {row['quadrant']} | {100 * row['delta_em']:.2f} | [{100 * lo:.2f}, {100 * hi:.2f}] | {row['n_paired_facts']} |")
    lines.extend(["", "Memory intervention differences within the same checkpoint, in percentage points with fact-clustered 95% CIs:", "",
                  "Uniform guessing over 16 candidates has expected accuracy 6.25%; always outputting one candidate also scores 6.25% on these balanced facts. "
                  "Therefore, an increase from 0% to roughly 6% between training conditions cannot establish memory generalization by itself. Also examine real retrieval against shuffled values and zero residuals; "
                  "beating zero residuals without beating shuffled values may only reflect learning to output candidate words. CIs reflect fact sampling, not training-seed variation.", "",
                  "| Condition | Quadrant | Control | Real / control EM | Δ EM (pp) | 95% CI (pp) |",
                  "| --- | --- | --- | --- | ---: | --- |"])
    for row in summary.get("within_run_memory_effects", []):
        lo, hi = row["ci95"]
        lines.append(f"| {row['condition']} | {row['quadrant']} | {row['second_method']} | "
                     f"{row['first_em']:.1%} / {row['second_em']:.1%} | {100 * row['delta_em']:.2f} | "
                     f"[{100 * lo:.2f}, {100 * hi:.2f}] |")
    lines.extend(["", "Limitations:", "", *["- " + limitation for limitation in summary["limitations"]]])
    if summary["skipped_runs"] or summary["refused_comparisons"]:
        lines.extend(["", "Excluded runs or comparisons:", ""])
        for row in summary["skipped_runs"]:
            lines.append("- " + row["run"] + ": " + row["reason"])
        for row in summary["refused_comparisons"]:
            lines.append("- " + row["contrast"] + ": " + row["reason"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = collect(args.runs_root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render(summary), encoding="utf-8")
    print(json.dumps(dict(completed_runs=summary["completed_run_count"], skipped_runs=len(summary["skipped_runs"]))))


if __name__ == "__main__":
    main()
