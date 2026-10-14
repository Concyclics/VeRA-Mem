"""Paired, article-clustered analysis of a completed corpus round.

This module never turns an incomplete run into a quality result. Bootstrap
intervals describe the held-out articles, not robustness across training seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .corpus_data import atomic_json, sha256


MODES = ("diagonal", "factor_mean_outer", "block_outer")


def keyed(records):
    result = {}
    for row in records:
        key = (row["id"], row["style"])
        if key in result:
            raise ValueError(f"Duplicate observation/style key: {key}")
        result[key] = row
    if not result:
        raise ValueError("An empty evaluation is not a result")
    return result


def paired(left, right, metric="em"):
    left, right = keyed(left), keyed(right)
    if left.keys() != right.keys():
        raise ValueError("Paired conditions must contain exactly the same observation/style keys")
    keys = sorted(left)
    if any(left[k]["article"] != right[k]["article"] for k in keys):
        raise ValueError("An observation changed source article")
    return (np.asarray([left[k][metric] - right[k][metric] for k in keys], dtype=float),
            np.asarray([left[k]["article"] for k in keys]))


def cluster_interval(values, articles, *, seed=20261013, repetitions=2000):
    """Resample whole articles with replacement; preserve within-article pairing.

    The estimand is the query-weighted mean. Variable article sizes therefore
    contribute both their sum and their count in every bootstrap replicate.
    """
    values, articles = np.asarray(values, dtype=float), np.asarray(articles)
    if values.ndim != 1 or values.shape != articles.shape or not len(values):
        raise ValueError("Require nonempty, aligned values and article labels")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite analysis input")
    unique, inverse = np.unique(articles, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(seed).integers(0, len(unique), (repetitions, len(unique)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    low, high = np.quantile(means, [.025, .975])
    return {"mean": float(values.mean()), "ci95": [float(low), float(high)],
            "queries": len(values), "articles": len(unique), "bootstrap_repetitions": repetitions}


def contrast(left, right):
    return {metric: cluster_interval(*paired(left, right, metric)) for metric in ("em", "f1")}


def joint_success(*conditions):
    mappings = [keyed(rows) for rows in conditions]
    if any(rows.keys() != mappings[0].keys() for rows in mappings[1:]):
        raise ValueError("Joint success requires matching A/B/restoration keys")
    keys = sorted(mappings[0])
    if any(row[k]["article"] != mappings[0][k]["article"] for row in mappings[1:] for k in keys):
        raise ValueError("Joint success article mismatch")
    values = np.prod([[row[k]["em"] for k in keys] for row in mappings], axis=0)
    return cluster_interval(values, [mappings[0][k]["article"] for k in keys])


def ppl_ratio_interval(left, right, excluded=(), *, repetitions=2000):
    """Paired token-weighted PPL ratio, clustering complete source articles."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2:
        raise ValueError("Require aligned per-article [NLL sum, target count] arrays")
    if not np.array_equal(left[:, 1], right[:, 1]):
        raise ValueError("PPL methods scored different targets")
    keep = np.ones(len(left), dtype=bool)
    keep[list(excluded)] = False
    left, right = left[keep], right[keep]
    if not len(left) or (left[:, 1] <= 0).any():
        raise ValueError("PPL analysis needs scored articles")
    difference = left[:, 0] - right[:, 0]
    draws = np.random.default_rng(20261013).integers(0, len(left), (repetitions, len(left)))
    ratios = np.exp(difference[draws].sum(1) / left[draws, 1].sum(1))
    return {"ratio": float(np.exp(difference.sum() / left[:, 1].sum())),
            "ci95": np.quantile(ratios, [.025, .975]).tolist(), "articles": len(left),
            "target_tokens": int(left[:, 1].sum())}


def analyze_evaluation(directory):
    directory = Path(directory)
    if not (directory / "complete.json").exists():
        raise ValueError(f"Evaluation is incomplete: {directory}")
    manifest = json.loads((directory / "evaluation.json").read_text())
    continual = directory / ("continual_" + manifest["split"])
    summary = json.loads((continual / "complete.json").read_text())
    if summary["parameter_digests_before"] != summary["parameter_digests_after"]:
        raise ValueError("Online evaluation modified shared weights or statistics")

    def read(name, phase):
        paths = sorted(continual.glob(f"{name}_{phase}.jsonl"))
        if not paths:
            raise ValueError(f"Missing condition: {name}/{phase}")
        rows = [json.loads(line) for path in paths for line in path.open()]
        keyed(rows)
        return rows

    result = {"manifest": manifest, "frozen_parameter_hashes_match": True,
              "groups": summary["groups"], "methods": {}, "static": {}}
    empty, lora = read("base", "empty"), read("lora", "prewrite")
    for name in MODES:
        first = read(name, "write_a_*")
        final = read(name, "final_a")
        pre = read(name, "prewrite")
        shuffled = read(name, "shuffled_a")
        oracle = read(name, "oracle_a")
        edit = read(name, "write_b_*")
        stale = read(name, "stale_b_*")
        restore = read(name, "restore_a_*")
        gain = contrast(final, empty)
        shuffled_gain = contrast(final, shuffled)
        # Keep derived thresholds JSON-native (NumPy comparisons return np.bool_).
        oracle_gain = float(paired(oracle, empty)[0].mean())
        recovery = gain["em"]["mean"] / oracle_gain if oracle_gain > 0 else None
        row = {"stream_a_vs_prewrite": contrast(first, pre), "stream_a_vs_empty": contrast(first, empty),
               "stream_a_vs_lora": contrast(first, lora), "final_a_vs_empty": gain,
               "final_a_vs_shuffled": shuffled_gain, "final_a_vs_oracle": contrast(final, oracle),
               "edit_b_vs_stale_a": contrast(edit, stale), "restore_a_vs_first_a": contrast(restore, first),
               "joint_a_b_restore_em": joint_success(first, edit, restore),
               "positive_oracle_gain": float(oracle_gain) if oracle_gain > 0 else None,
               "fraction_of_positive_oracle_gain_recovered": recovery,
               "retention": {}, "diagnostic_gates": {
                   "final_a_at_least_10pp_over_empty_and_shuffled": all(
                       item["em"]["mean"] >= .1 for item in (gain, shuffled_gain)),
                   "positive_paired_lower_bounds": all(item["em"]["ci95"][0] > 0 for item in (gain, shuffled_gain)),
                   "at_least_half_positive_oracle_gain": recovery >= .5 if recovery is not None else None}}
        if not manifest["development"]:
            row["stream_a_vs_retrieved_text"] = contrast(first, read("base", "retrieved_context_*"))
        for path in sorted(continual.glob(f"{name}_retention_after_*.jsonl")):
            after = [json.loads(line) for line in path.open()]
            row["retention"][path.stem.removeprefix(name + "_")] = contrast(after, read(name, "write_a_0000"))
        result["methods"][name] = row
    for split in ("validation", "test"):
        path = directory / f"static_{split}.json"
        if not path.exists():
            continue
        static = json.loads(path.read_text())
        methods = static["methods"]
        result["static"][split] = {"excluded_primary_articles": static["excluded_primary_articles"],
                                    "methods": {}}
        for name, item in methods.items():
            ratios = {base: ppl_ratio_interval(item["article_nll_tokens"], methods[base]["article_nll_tokens"],
                                              static["excluded_primary_articles"])
                      for base in ("base", "lora")}
            result["static"][split]["methods"][name] = {
                **{key: value for key, value in item.items() if key != "article_nll_tokens"},
                "ratios": ratios}
        result["static"][split]["block_vs_factor_mean"] = ppl_ratio_interval(
            methods["block_outer"]["article_nll_tokens"], methods["factor_mean_outer"]["article_nll_tokens"],
            static["excluded_primary_articles"])
    shuffled = json.loads((directory / "shuffled_validation.json").read_text())
    base_ppl = result["static"]["validation"]["methods"]["base"]["ppl"]
    for name in MODES:
        ratio = shuffled["methods"][name]["ppl"] / base_ppl
        result["methods"][name]["shuffled_validation_ppl_ratio_to_base"] = ratio
        result["methods"][name]["diagnostic_gates"]["shuffled_validation_within_2pct_of_base"] = ratio <= 1.02
    result["block_vs_factor_mean_stream_a"] = contrast(read("block_outer", "write_a_*"),
                                                       read("factor_mean_outer", "write_a_*"))
    result["block_vs_diagonal_stream_a"] = contrast(read("block_outer", "write_a_*"), read("diagonal", "write_a_*"))
    return result


def render(result):
    lines = ["# WikiText-103 first-round results", "",
             "One complete training pass, seed 20261013. Article-bootstrap intervals describe this held-out sample; "
             "they do not measure training-seed robustness. Comparisons are exploratory and are not adjusted for multiplicity.", "",
             f"Training LM targets: **{result['training']['lm_tokens']:,}**. "
             f"Additional episodic targets: **{result['training']['episode_tokens']:,}**.", ""]
    confirmation = result["confirmation"]
    lines += ["## Static language modeling", "", "| Method | Clean test PPL | PPL / frozen base |",
              "| --- | ---: | ---: |"]
    for name, row in confirmation["static"]["test"]["methods"].items():
        lines.append(f"| {name} | {row['ppl']:.4f} | {row['ratios']['base']['ratio']:.4f} |")
    lines += ["", "The train-overlapping article is excluded from primary test PPL. The machine-readable result "
              "also retains the full official split statistic.", "", "## Frozen-weight continual memory", "",
              "| Reader | Stream A EM gain vs pre-write (pp, 95% CI) | B gain vs stale A (pp, 95% CI) | Joint A/B/restore EM |",
              "| --- | ---: | ---: | ---: |"]
    def interval(item):
        low, high = item["ci95"]
        return f"{100 * item['mean']:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]"
    for name, row in confirmation["methods"].items():
        lines.append(f"| {name} | {interval(row['stream_a_vs_prewrite']['em'])} | "
                     f"{interval(row['edit_b_vs_stale_a']['em'])} | {100 * row['joint_a_b_restore_em']['mean']:.2f}% |")
    lines += ["", "These are three-word anchored passage completions with two unseen query styles, not open-domain QA. "
              "All reader/writer parameters and statistics were unchanged during writes, edits, and restoration.", "",
              "## Interpretation limits", "",
              "A static PPL gain alone demonstrates corpus adaptation. Evidence for online memory requires paired gains "
              "after new writes and counterfactual edits, together with restoration and retention. The full JSON contains "
              "pre-write, shuffled, oracle, retrieved-text, and LoRA controls, paired operator comparisons, and all "
              "diagnostic gates. No gate caused early stopping.", "",
              "This matches the WikiText corpus scale, not the parameter counts, multi-epoch budgets, backbones, or "
              "reported perplexities of Memory Decoder or MLP Memory.", ""]
    return "\n".join(lines)


def main(args):
    root, output = Path(args.round), Path(args.output)
    training = json.loads((root / "training/complete.json").read_text())
    registration = json.loads((root / "training/registration.json").read_text())
    if not training["complete"] or training["lm_tokens"] != registration["full_pass_target_tokens"]:
        raise ValueError("Only an exact full-pass training completion can be reported")
    inputs = [p for folder in ("training", "development_10m", "confirmation", "stages")
              for p in (root / folder).rglob("*") if p.suffix in (".json", ".jsonl")]
    result = {"schema": "vera-corpus-analysis-v1", "training": training, "registration": registration,
              "analysis_source_sha256": sha256(__file__),
              "bootstrap": {"cluster": "source article", "repetitions": 2000, "seed": 20261013,
                            "estimand": "query-weighted paired difference", "multiplicity_adjusted": False},
              "development": analyze_evaluation(root / "development_10m"),
              "confirmation": analyze_evaluation(root / "confirmation"),
              "input_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted(inputs)}}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "analysis.json", result)
    (output / "report.md").write_text(render(result))
    atomic_json(output / "complete.json", {name: sha256(output / name) for name in ("analysis.json", "report.md")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
