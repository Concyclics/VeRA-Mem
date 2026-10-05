"""Publish aggregate-only pilot results from a local, possibly partial suite.

Usage: python scripts/summarize_results.py --suite RUN_DIRECTORY --output REPORT_DIRECTORY
Uses only the standard library. No network, model loading, or GPU work occurs.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re


BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 123
PHASE = re.compile(r"block_(\d+)$")
METHOD = re.compile(r"(?:vdb_(?:real|oracle|empty|shuffled)|text_(?:tfidf|oracle)|frozen|lora\d+_r\d+|latent_(?:knn|oracle|empty|shuffled))$")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_json(path, alias, evidence, warnings):
    if not path.exists():
        return None
    raw = path.read_bytes()
    evidence[alias] = {"file": alias, "sha256": hashlib.sha256(raw).hexdigest(),
                       "bytes": len(raw), "mtime_ns": path.stat().st_mtime_ns}
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        warnings.append(f"Unfinished or invalid JSON ignored: {alias}")
        return None


def load_predictions(path, alias, evidence, warnings):
    if not path.exists():
        return []
    raw = path.read_bytes()
    evidence[alias] = {"file": alias, "sha256": hashlib.sha256(raw).hexdigest(),
                       "bytes": len(raw), "mtime_ns": path.stat().st_mtime_ns}
    rows, seen = [], set()
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index == len(lines) - 1 and not raw.endswith(b"\n"):
                warnings.append(f"Ignored in-progress final JSONL record: {alias}")
                break
            raise ValueError(f"Invalid JSONL record in {alias}, line {index + 1}")
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("phase"), str):
            raise ValueError(f"Invalid prediction schema in {alias}, line {index + 1}")
        identity = (row["phase"], row["id"])
        if identity in seen:
            raise ValueError(f"Duplicate fact/phase records in {alias}; refusing ambiguous pairing")
        if row.get("em") not in (0, 1, 0.0, 1.0):
            raise ValueError(f"Nonbinary exact match in {alias}")
        seen.add(identity)
        rows.append(row)
    return rows


def summarize_phase(rows, fallback=None):
    if not rows:
        fallback = fallback or {}
        n = fallback.get("n", 0)
        em = fallback.get("em")
        return {"n": n, "correct": round(n * em) if finite(em) and isinstance(n, int) else None,
                "em": em if finite(em) else None,
                "choice_accuracy": fallback.get("choice_accuracy"),
                "answer_token_nll": fallback.get("answer_token_nll"),
                "answer_token_ppl": fallback.get("answer_token_ppl"),
                "retrieval_hit_at_1": fallback.get("retrieval_hit_at_1"),
                "retrieval_hit_at_k": fallback.get("retrieval_hit_at_k"),
                "source": "reported_aggregate" if fallback else "unavailable"}
    correct = sum(int(row["em"]) for row in rows)
    scored = [row for row in rows if finite(row.get("nll_sum"))
              and isinstance(row.get("answer_tokens"), int) and row["answer_tokens"] > 0]
    tokens = sum(row["answer_tokens"] for row in scored)
    nll = sum(row["nll_sum"] for row in scored) / tokens if tokens else None
    choices = [row["choice_correct"] for row in rows if row.get("choice_correct") in (0, 1)]
    retrieval = [row for row in rows if row.get("expected_id") is not None]
    hits_k = [row["retrieval_hit_at_k"] for row in rows if finite(row.get("retrieval_hit_at_k"))]
    ppl = math.exp(nll) if nll is not None and nll < 709 else None
    return {"n": len(rows), "correct": correct, "em": correct / len(rows),
            "choice_accuracy": sum(choices) / len(choices) if choices else None,
            "choice_n": len(choices), "answer_tokens": tokens, "nll_scored_n": len(scored),
            "answer_token_nll": nll, "answer_token_ppl": ppl,
            "ppl_overflow": nll is not None and nll >= 709,
            "retrieval_hit_at_1": sum(row.get("retrieved_id") == row["expected_id"] for row in retrieval) / len(retrieval) if retrieval else None,
            "retrieval_hit_at_k": sum(hits_k) / len(hits_k) if hits_k else None,
            "source": "recomputed_from_predictions"}


def retention(matrix):
    valid = sorted((row for row in matrix if isinstance(row, dict)
                    and isinstance(row.get("after_writes"), int)), key=lambda row: row["after_writes"])
    if not valid:
        return {"by_block": [], "mean_previous_best_minus_final": None}
    final = valid[-1]
    results = []
    for block, score in enumerate(final.get("by_block", [])):
        final_em = score.get("em")
        prior = [row["by_block"][block]["em"] for row in valid[:-1]
                 if len(row.get("by_block", [])) > block and finite(row["by_block"][block].get("em"))]
        if not finite(final_em):
            continue
        results.append({"block_index": block, "final_n": score.get("n"), "final_em": final_em,
                        "previous_best_em": max(prior) if prior else None,
                        "previous_best_minus_final": max(prior) - final_em if prior else None,
                        "first_observed_minus_final": prior[0] - final_em if prior else None,
                        "best_including_final_minus_final": max([final_em, *prior]) - final_em})
    deltas = [row["previous_best_minus_final"] for row in results if row["previous_best_minus_final"] is not None]
    return {"by_block": results, "final_after_writes": final["after_writes"],
            "mean_previous_best_minus_final": sum(deltas) / len(deltas) if deltas else None,
            "definition": "max earlier block-evaluation EM minus final EM; negative means improvement; no earlier measurement is null"}


def baseline_paraphrase_bug(suite, evidence):
    path = suite / "source/src/vera_mem/run.py"
    if not path.exists():
        return None
    raw = path.read_bytes()
    alias = "source/src/vera_mem/run.py"
    evidence[alias] = {"file": alias, "sha256": hashlib.sha256(raw).hexdigest(),
                       "bytes": len(raw), "mtime_ns": path.stat().st_mtime_ns}
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return None
    found = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != "evaluate":
            continue
        if len(call.args) >= 3 and isinstance(call.args[2], ast.Constant) and call.args[2].value == "paraphrase":
            explicit = any(keyword.arg == "paraphrase" and isinstance(keyword.value, ast.Constant)
                           and keyword.value.value is True for keyword in call.keywords)
            found.append(not explicit)
    return any(found) if found else None


def percentile(values, quantile):
    rank = (len(values) - 1) * quantile
    lower, upper = math.floor(rank), math.ceil(rank)
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def paired_bootstrap(first, second):
    rng = random.Random(BOOTSTRAP_SEED)
    differences = [first[index] - second[index] for index in range(len(first))]
    n = len(differences)
    sampled = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(BOOTSTRAP_SAMPLES))
    return {"n_paired_facts": n, "delta_em": sum(differences) / n,
            "ci_95_percentile": [percentile(sampled, 0.025), percentile(sampled, 0.975)],
            "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
            "unit": "paired fact ID", "interpretation": "descriptive fact-sampling interval conditional on these trained checkpoints; not training-seed uncertainty"}


def collect(suite, output):
    evidence, warnings, result_rows, pairing_data = {}, [], [], {}
    suite_manifest = load_json(suite / "suite.json", "suite.json", evidence, warnings) or {}
    jobs = suite_manifest.get("jobs", [])
    jobs_by_name = {job.get("name"): job for job in jobs if isinstance(job, dict)}
    candidates = [path for path in sorted(suite.iterdir()) if path.is_dir()
                  and path.name not in ("source", "paraphrase_correction") and path.resolve() != output.resolve()
                  and any((path / name).exists() for name in ("summary.json", "config.json", "manifest.json"))]
    broken_paraphrase = baseline_paraphrase_bug(suite, evidence)
    all_runs_complete = bool(candidates)
    public_runs = []
    for index, directory in enumerate(candidates, start=1):
        label = directory.name if re.fullmatch(r"(?:vector|baseline|smoke|pilot)_[A-Za-z0-9_-]+", directory.name) else f"run_{index:02d}"
        manifest = load_json(directory / "manifest.json", f"{label}/manifest.json", evidence, warnings) or {}
        config = load_json(directory / "config.json", f"{label}/config.json", evidence, warnings) or {}
        summary = load_json(directory / "summary.json", f"{label}/summary.json", evidence, warnings) or []
        if not isinstance(summary, list):
            raise ValueError(f"Unexpected run summary format: {label}")
        reported = {item["method"]: item for item in summary if isinstance(item, dict) and isinstance(item.get("method"), str)}
        methods = set(reported) | set(config.get("methods", []))
        methods |= {path.name for path in directory.iterdir() if path.is_dir() and (path / "predictions.jsonl").exists()}
        dataset = manifest.get("dataset")
        if dataset not in ("synthetic", "medmcqa"):
            dataset = next((value.get("dataset") for value in reported.values()
                            if value.get("dataset") in ("synthetic", "medmcqa")), "unknown")
        run_complete = manifest.get("complete") is True and bool(methods)
        completed_methods = 0
        for method in sorted(methods):
            if not METHOD.fullmatch(method):
                warnings.append(f"Skipped an unknown method label in {label}")
                run_complete = False
                continue
            metrics = load_json(directory / method / "metrics.json", f"{label}/{method}/metrics.json", evidence, warnings)
            metrics = metrics if isinstance(metrics, dict) else reported.get(method, {})
            predictions = load_predictions(directory / method / "predictions.jsonl", f"{label}/{method}/predictions.jsonl", evidence, warnings)
            phases = {}
            for row in predictions:
                phases.setdefault(row["phase"], []).append(row)
            blocks = [(int(match.group(1)), phase) for phase in phases if (match := PHASE.fullmatch(phase))]
            final_writes, final_phase = max(blocks) if blocks else (None, None)
            final_rows = phases.get(final_phase, [])
            final = summarize_phase(final_rows, metrics.get("final"))
            declared_size = config.get("stream_size")
            complete = bool(metrics) and final_phase is not None and len(final_rows) == final_writes
            if isinstance(declared_size, int):
                complete = complete and final_writes == declared_size
            completed_methods += int(complete)
            run_complete &= complete
            para = summarize_phase(phases.get("paraphrase", []), metrics.get("paraphrase"))
            originally_reported_para_em = para.get("em")
            para_valid = not (broken_paraphrase and not method.startswith("vdb_"))
            correction_used = False
            if dataset == "synthetic" and not method.startswith("vdb_"):
                correction = suite / "paraphrase_correction" / method
                correction_alias = f"paraphrase_correction/{method}"
                corrected_metrics = load_json(correction / "metrics.json", f"{correction_alias}/metrics.json", evidence, warnings)
                corrected_predictions = load_predictions(correction / "predictions.jsonl", f"{correction_alias}/predictions.jsonl", evidence, warnings)
                corrected_rows = [row for row in corrected_predictions if row["phase"] in ("paraphrase", "paraphrase_corrected")]
                original_by_id = {row["id"]: row for row in final_rows}
                correction_complete = bool(original_by_id) and len(corrected_rows) == len(original_by_id)
                correction_complete &= {row["id"] for row in corrected_rows} == set(original_by_id)
                correction_complete &= all(row.get("answer") == original_by_id.get(row["id"], {}).get("answer") for row in corrected_rows)
                if correction_complete:
                    fallback = corrected_metrics.get("paraphrase", corrected_metrics) if isinstance(corrected_metrics, dict) else None
                    para = summarize_phase(corrected_rows, fallback)
                    para_valid, correction_used = True, True
                elif corrected_metrics is not None or corrected_predictions:
                    warnings.append(f"{label}/{method}: paraphrase correction is incomplete or fact/label-mismatched; excluded until complete")
            if not para_valid and para.get("n"):
                warnings.append(f"{label}/{method}: phase named paraphrase reused the original question in the saved runner; excluded from paraphrase EM")
            training_seed = manifest.get("training_seed", metrics.get("training_seed", metrics.get("seed", config.get("seed"))))
            evaluation_seed = manifest.get("evaluation_seed", config.get("eval_seed", config.get("seed", metrics.get("seed"))))
            if "evaluation_seed" in manifest and "eval_seed" in config and manifest["evaluation_seed"] != config["eval_seed"]:
                raise ValueError(f"Conflicting evaluation seed in manifest/config: {label}")
            record = {"run": label, "dataset": dataset, "method": method,
                      "seed": metrics.get("seed", config.get("seed")),
                      "training_seed": training_seed, "evaluation_seed": evaluation_seed,
                      "status": "complete" if complete else "partial", "final_phase": final_phase,
                      "final_after_writes": final_writes, "expected_stream_size": declared_size,
                      "final": final, "paraphrase_em": para.get("em") if para_valid else None,
                      "paraphrase_valid": para_valid, "reported_paraphrase_em": originally_reported_para_em,
                      "paraphrase_correction_used": correction_used,
                      "paraphrase_note": "posthoc implementation-bug correction using frozen checkpoints; no retraining" if correction_used else None,
                      "shared_trainable_parameters": metrics.get("offline_trainable_parameters", metrics.get("trainable_parameters")),
                      "active_adapter_parameters": metrics.get("active_adapter_parameters"),
                      "key_bytes": metrics.get("vdb_numeric_resident_bytes", {}).get("key_bytes", metrics.get("external_key_bytes")),
                      "value_bytes": metrics.get("vdb_numeric_resident_bytes", {}).get("value_bytes", metrics.get("external_value_bytes")),
                      "random_projection_buffer_bytes": metrics.get("random_projection_buffer_bytes"),
                      "online_gradient_steps": metrics.get("online_gradient_steps"),
                      "forgetting": retention(metrics.get("retention_matrix", [])),
                      "prediction_records": len(predictions),
                      "retrieval_scope": "final prompt token" if method in ("vdb_real", "vdb_shuffled") else "oracle record selection" if "oracle" in method else "request-level or unavailable"}
            for phase in ("priming_before", "priming_after", "control_before", "control_after"):
                record[phase] = summarize_phase(phases.get(phase, []), metrics.get(phase))
            result_rows.append(record)
            pairing_data[(label, method)] = {"rows": final_rows, "record": record,
                                            "stream_hash": manifest.get("data_sha256", {}).get("stream"),
                                            "priming_hash": manifest.get("data_sha256", {}).get("priming")}
        job = jobs_by_name.get(directory.name, {})
        if job and job.get("exit_code") != 0:
            run_complete = False
        all_runs_complete &= run_complete
        public_runs.append({"run": label, "dataset": dataset, "status": "complete" if run_complete else "partial",
                            "completed_methods": completed_methods, "expected_methods": len(methods),
                            "exit_code": job.get("exit_code") if isinstance(job.get("exit_code"), int) else None})
    pairs = []
    for key, first in pairing_data.items():
        a = first["record"]
        if a["dataset"] != "synthetic" or a["method"] != "vdb_real" or not first["rows"]:
            continue
        for other_key, second in pairing_data.items():
            b = second["record"]
            if key == other_key or b["dataset"] != "synthetic" or not second["rows"]:
                continue
            eligible = a["evaluation_seed"] is not None and a["evaluation_seed"] == b["evaluation_seed"]
            eligible &= a["final_after_writes"] == b["final_after_writes"]
            eligible &= a["status"] == b["status"] == "complete"
            first_by_id, second_by_id = ({row["id"]: row for row in value["rows"]} for value in (first, second))
            eligible &= set(first_by_id) == set(second_by_id)
            eligible &= all(first_by_id[id].get("answer") == second_by_id.get(id, {}).get("answer") for id in first_by_id)
            for field in ("stream_hash", "priming_hash"):
                if first[field] and second[field] and first[field] != second[field]:
                    eligible = False
            if not eligible:
                warnings.append(f"Not paired: {a['run']}/{a['method']} vs {b['run']}/{b['method']} (different or incomplete facts, evaluation seed, snapshot or observations)")
                continue
            ids = sorted(first_by_id)
            result = paired_bootstrap([first_by_id[id]["em"] for id in ids], [second_by_id[id]["em"] for id in ids])
            pairs.append({"first": f"{a['run']}/{a['method']}", "second": f"{b['run']}/{b['method']}",
                          "evaluation_seed": a["evaluation_seed"],
                          "first_training_seed": a["training_seed"], "second_training_seed": b["training_seed"], **result})
    listed_jobs_complete = bool(jobs) and all(job.get("exit_code") == 0 for job in jobs if isinstance(job, dict))
    candidate_names = {path.name for path in candidates}
    all_listed_runs_present = all(name in candidate_names for name in jobs_by_name)
    complete = suite_manifest.get("complete") is True and all_runs_complete and listed_jobs_complete and all_listed_runs_present
    return {"format_version": 1, "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "status": "complete" if complete else "partial", "complete": complete,
            "runs": public_runs, "methods": result_rows, "paired_bootstrap": pairs,
            "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
            "warnings": sorted(set(warnings)), "evidence": sorted(evidence.values(), key=lambda row: row["file"]),
            "limitations": [
                "Each method is represented by one training realization: fact-bootstrap intervals condition on the observed checkpoints and online evaluation seed; they do not capture training randomness or support population-level method superiority. Different evaluation seeds are not pooled.",
                "Synthetic entity-to-value associations share 16 output words and few templates; these results do not establish open-domain memory or held-out-value generalization.",
                "MedMCQA vector runs transfer an interface trained on synthetic facts; domain-transfer failure alone does not reject the architecture. Public QA may overlap pretraining.",
                "Answer-token NLL is sum of answer-token negative log likelihood divided by the number of scored answer tokens; PPL is exp of that NLL. Neither is full-prompt or full-corpus perplexity.",
                "Vector real/shuffled retrieval metrics refer to the final prompt token, not all forward/generation tokens. Oracle selection is a separate diagnostic.",
                "A rolled value bank changes record identity but can preserve the same answer word; shuffled is not guaranteed to be wrong-valued for every query.",
                "The unwritten synthetic control labels are random hidden values; their EM measures unexposed guessing rather than abstention accuracy.",
                "Parameter counts, bank vector bytes and frozen projection bytes are distinct costs; this pilot does not establish matched total storage or deployment latency.",
            ]}


def percent(value):
    return f"{100 * value:.2f}%" if finite(value) else "—"


def number(value):
    return f"{value:.4f}" if finite(value) else "—"


def before_after(record, prefix):
    return f"{percent(record[prefix + '_before'].get('em'))} → {percent(record[prefix + '_after'].get('em'))}"


def report(summary):
    lines = ["# Preliminary memory experiment results", "", f"Suite status: **{summary['status']}**.", "",
             "Only aggregate measurements are published here. A partial suite is not evidence that unfinished methods failed.", "",
             "## Final observed-stream performance", "",
             "For a partial method, the final phase is the greatest available block number; it may be earlier than the requested end of the stream.", "",
             "| Run | Dataset | Method | Training / evaluation seed | State | Phase | Correct / n | EM | Choice accuracy | Paraphrase EM | Answer-token NLL | Answer-token PPL | Retrieval @1 / @k |", "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in summary["methods"]:
        f = item["final"]
        lines.append(f"| {item['run']} | {item['dataset']} | {item['method']} | {item['training_seed']} / {item['evaluation_seed']} | {item['status']} | {item['final_phase'] or 'unavailable'} | {f.get('correct')} / {f.get('n')} | {percent(f.get('em'))} | {percent(f.get('choice_accuracy'))} | {percent(item['paraphrase_em'])} | {number(f.get('answer_token_nll'))} | {number(f.get('answer_token_ppl'))} | {percent(f.get('retrieval_hit_at_1'))} / {percent(f.get('retrieval_hit_at_k'))} |")
    lines += ["", "Choice accuracy scores the four option candidates and is distinct from generated-output EM. A dash denotes unavailable or invalid evidence, not zero.", "",
              "## Retention and numerical storage", "",
              "| Run / method | Priming EM before → after | Control EM before → after | Previous-best minus final EM | Shared trained parameters | Active adapter parameters | Key bytes | Value bytes | Frozen projection bytes |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in summary["methods"]:
        fields = [item["shared_trainable_parameters"], item["active_adapter_parameters"], item["key_bytes"], item["value_bytes"], item["random_projection_buffer_bytes"]]
        lines.append(f"| {item['run']} / {item['method']} | {before_after(item, 'priming')} | {before_after(item, 'control')} | {percent(item['forgetting']['mean_previous_best_minus_final'])} | " + " | ".join(str(x) if x is not None else "—" for x in fields) + " |")
    corrected = [f"{item['run']}/{item['method']}" for item in summary["methods"] if item["paraphrase_correction_used"]]
    if corrected:
        lines += ["", "Paraphrase scores were corrected after detecting an evaluation-call bug for: " + ", ".join(corrected) + ". These are posthoc implementation-bug corrections evaluated with frozen checkpoints; no retraining was performed. Original mislabeled scores remain identified in the JSON and are not used as paraphrase evidence."]
    lines += ["", "Forgetting is calculated per block as its maximum earlier evaluation EM minus its final EM, then averaged over blocks with an earlier measurement. Negative values indicate improvement. A block first evaluated at the final snapshot is excluded from this average. The JSON retains every block and the alternative nonnegative best-including-final statistic.", "",
              "## Paired exploratory comparisons", "",
              "Differences are first minus second. These 10,000 paired fact-bootstrap replicates (bootstrap seed 123) are descriptive intervals conditional on these checkpoints, not uncertainty over model training or independent replications. Pairing requires the same fact IDs, labels, online evaluation seed, final write count and compatible observation hashes. Training seeds may differ and are recorded separately; different online evaluation seeds are never paired.", "",
              "| First | Second | Paired facts | EM difference | 95% percentile interval |", "|---|---|---:|---:|---:|"]
    for pair in summary["paired_bootstrap"]:
        low, high = pair["ci_95_percentile"]
        lines.append(f"| {pair['first']} | {pair['second']} | {pair['n_paired_facts']} | {percent(pair['delta_em'])} | [{percent(low)}, {percent(high)}] |")
    if not summary["paired_bootstrap"]:
        lines.append("| No eligible completed pair | — | — | — | — |")
    lines += ["", "## Limitations", ""]
    lines += [f"- {limitation}" for limitation in summary["limitations"]]
    if summary["warnings"]:
        lines += ["", "## Evidence checks", ""] + [f"- {warning}" for warning in summary["warnings"]]
    return "\n".join(lines) + "\n"


def protect_newer_evidence(output, current):
    path = output / "summary.json"
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous.get("format_version") != 1 or "evidence" not in previous:
        raise ValueError("Refusing to overwrite an existing summary with an unknown provenance format")
    if previous.get("complete") and not current["complete"]:
        raise ValueError("Refusing to replace a complete report with a partial evidence snapshot")
    new_evidence = {row["file"]: row for row in current["evidence"]}
    for old in previous["evidence"]:
        new = new_evidence.get(old["file"])
        if new is None or (new["sha256"] != old["sha256"] and new["mtime_ns"] < old["mtime_ns"]):
            raise ValueError("Refusing to overwrite a report with missing or older source evidence")
    new_methods = {(row["run"], row["method"]): row for row in current["methods"]}
    for old in previous["methods"]:
        new = new_methods.get((old["run"], old["method"]))
        if new is None or new["prediction_records"] < old["prediction_records"]:
            raise ValueError("Refusing to overwrite a report with fewer prediction records")
        if old["status"] == "complete" and new["status"] != "complete":
            raise ValueError("Refusing to downgrade a completed method to partial")
        if old.get("paraphrase_correction_used") and not new.get("paraphrase_correction_used"):
            raise ValueError("Refusing to discard previously complete paraphrase-correction evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.suite.is_dir():
        parser.error("--suite must be a local suite directory")
    summary = collect(args.suite, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    protect_newer_evidence(args.output, summary)
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # Only whitelisted fields/aggregate values are serialized; no source
    # commands, absolute paths, host/user metadata, questions or answers.
    for filename, content in (("report.md", report(summary)), ("summary.json", serialized)):
        destination = args.output / filename
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
    print(json.dumps({"status": summary["status"], "runs": len(summary["runs"]),
                      "methods": len(summary["methods"]), "paired_comparisons": len(summary["paired_bootstrap"])}))


if __name__ == "__main__":
    main()
