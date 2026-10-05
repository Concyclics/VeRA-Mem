"""Plot fully audited template-generalization results as PNG and editable SVG.

Input is summary.json from summarize_generalization.py. All three matched
conditions must be complete; partial, ambiguous, or incompatible runs fail
explicitly. This script never fills absent measurements with zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="vera_mem_mpl_")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


CONDITIONS = ("canonical", "augment", "invariant")
CONDITION_LABELS = ("Single template\n(control)", "Augmentation", "Augmentation\n+ consistency")
SHORT_LABELS = ("Single template", "Augmentation", "Augmentation + consistency")
QUADRANTS = (
    "canonical_query/canonical_support",
    "heldout_query/canonical_support",
    "canonical_query/heldout_support",
    "heldout_query/heldout_support",
)
QUADRANT_LABELS = (
    "Canonical query\nCanonical observation",
    "Held-out queries\nCanonical observation",
    "Canonical query\nHeld-out observations",
    "Held-out queries\nHeld-out observations",
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _integer(value, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid {label}: expected integer >= {minimum}")
    return value


def _probability(value, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Invalid probability: " + label)
    return float(value)


def select_runs(summary: dict) -> list[dict]:
    """Require one completed run per arm and matching causal/budget evidence."""
    if not isinstance(summary, dict) or summary.get("schema_version") != 1:
        raise ValueError("Expected summarize_generalization.py schema version 1")
    rows = summary.get("runs")
    if not isinstance(rows, list) or summary.get("completed_run_count") != len(rows):
        raise ValueError("Completed-run count does not match the summary")
    if len(rows) != len(CONDITIONS):
        raise ValueError("Exactly three completed main conditions are required")
    selected = []
    for condition in CONDITIONS:
        matches = [row for row in rows if isinstance(row, dict) and row.get("condition") == condition]
        if len(matches) != 1:
            raise ValueError("Expected exactly one complete run for " + condition)
        row = matches[0]
        if row.get("verified_complete") is not True:
            raise ValueError("Missing explicit prediction-audited completion flag: " + condition)
        provenance = row.get("source_cache_group_sha256")
        if not isinstance(provenance, str) or not SHA256.fullmatch(provenance):
            raise ValueError("Missing source/cache/model provenance: " + condition)
        config = row.get("configuration")
        if not isinstance(config, dict):
            raise ValueError("Missing training configuration: " + condition)
        for field in ("updates", "train_size", "eval_size", "batch_size", "alignment_steps", "validate_every"):
            _integer(config.get(field), condition + "/" + field, 1)
        _integer(config.get("seed"), condition + "/seed")
        oracle = _integer(config.get("oracle_updates"), condition + "/oracle_updates")
        selected_step = _integer(row.get("selected_step"), condition + "/selected_step", 1)
        if oracle > config["updates"] or selected_step > config["updates"]:
            raise ValueError("Checkpoint or curriculum exceeds declared update budget")
        if config["train_size"] != 4096 or config["eval_size"] != 128:
            raise ValueError("Figure requires the main 4096-train/128-test protocol")
        # Two support banks x four query templates x four methods; pre-write,
        # immediate, and the two 32-fact control phases are additionally audited.
        expected_predictions = config["eval_size"] * (2 * 4 * 4 + 2) + 64
        if row.get("verified_prediction_count") != expected_predictions:
            raise ValueError("Incomplete verified prediction coverage: " + condition)
        for quadrant in QUADRANTS:
            methods = row.get("quadrants", {}).get(quadrant)
            if not isinstance(methods, dict):
                raise ValueError("Missing quadrant: " + quadrant)
            expected_templates = 3 if quadrant.startswith("heldout_query/") else 1
            for method in ("real", "oracle", "shuffled", "empty"):
                cell = methods.get(method)
                if not isinstance(cell, dict):
                    raise ValueError("Missing quadrant/method: " + quadrant + "/" + method)
                if cell.get("n_facts") != config["eval_size"] or cell.get("n_templates") != expected_templates:
                    raise ValueError("Incorrect fact/template counts: " + quadrant)
                predictions = _integer(cell.get("n_predictions"), "n_predictions", 1)
                correct = _integer(cell.get("correct"), "correct")
                em = _probability(cell.get("em"), "em")
                if (predictions != config["eval_size"] * expected_templates or correct > predictions
                        or not math.isclose(em, correct / predictions, abs_tol=1e-9)):
                    raise ValueError("Inconsistent exact-match counts: " + quadrant + "/" + method)
                if method == "real":
                    _probability(cell.get("recall_at_4"), "recall_at_4")
        selected.append(row)
    if len({row["source_cache_group_sha256"] for row in selected}) != 1:
        raise ValueError("Source/cache/model/data provenance differs across conditions")
    if any(row["configuration"] != selected[0]["configuration"] for row in selected[1:]):
        raise ValueError("Training budgets or seeds differ across conditions")
    return selected


def _matrix_panel(ax, values, annotations, title, subtitle, *, row_labels):
    mesh = ax.imshow(values, cmap="Blues", norm=Normalize(0, 100), aspect="auto")
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", pad=33)
    ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=9, color="#53616C")
    ax.set_xticks(range(3), CONDITION_LABELS, fontsize=9)
    ax.set_yticks(range(4), QUADRANT_LABELS if row_labels else [""] * 4, fontsize=10)
    ax.tick_params(axis="both", which="major", length=0, pad=9)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for r in range(4):
        for c in range(3):
            color = "white" if values[r, c] >= 60 else "#17384B"
            ax.text(c, r, annotations[r][c], ha="center", va="center", fontsize=11,
                    color=color, linespacing=1.5)
    return mesh


def draw(summary: dict, output_prefix: Path, input_sha256: str) -> tuple[Path, Path]:
    rows = select_runs(summary)
    if not isinstance(input_sha256, str) or not SHA256.fullmatch(input_sha256):
        raise ValueError("Input summary SHA256 is required")
    config = rows[0]["configuration"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 7.4))
    figure.subplots_adjust(left=.20, right=.96, bottom=.31, top=.77, wspace=.10)
    figure.suptitle("VeRA-Mem: generalization beyond the training template",
                   x=.04, y=.97, ha="left", fontsize=16, fontweight="semibold")
    figure.text(.04, .916,
                f"Qwen3-4B-Instruct-2507  |  {config['train_size']:,} offline facts  |  "
                f"{config['updates']:,} LM updates  |  {config['eval_size']} new facts",
                fontsize=10, color="#53616C")
    real = np.array([[100 * row["quadrants"][quadrant]["real"]["em"]
                      for row in rows] for quadrant in QUADRANTS])
    recall = np.array([[100 * row["quadrants"][quadrant]["real"]["recall_at_4"]
                        for row in rows] for quadrant in QUADRANTS])
    real_text = [[f"{real[r, c]:.1f}%\n{rows[c]['quadrants'][quadrant]['real']['correct']}/"
                  f"{rows[c]['quadrants'][quadrant]['real']['n_predictions']}"
                  for c in range(3)] for r, quadrant in enumerate(QUADRANTS)]
    recall_text = [[f"{recall[r, c]:.1f}%" for c in range(3)] for r in range(4)]
    _matrix_panel(axes[0], real, real_text, "A  Exact match with sparse retrieval",
                  "Correct generations / query views", row_labels=True)
    mesh = _matrix_panel(axes[1], recall, recall_text, "B  Addressing recall@4",
                         "Correct fact in the final-prompt top-4", row_labels=False)
    color_axis = figure.add_axes((.765, .245, .19, .013))
    colorbar = figure.colorbar(mesh, cax=color_axis, orientation="horizontal", ticks=[0, 50, 100])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8, length=0, pad=3)
    colorbar.set_label("Percent (shared scale)", fontsize=8, labelpad=2)
    figure.text(.04, .195, "Memory controls: maximum EM across the four quadrants",
                fontsize=9, fontweight="semibold", color="#354956")
    for position, row, label in zip((.04, .36, .68), rows, SHORT_LABELS):
        shuffle = max(row["quadrants"][quadrant]["shuffled"]["em"] for quadrant in QUADRANTS)
        empty = max(row["quadrants"][quadrant]["empty"]["em"] for quadrant in QUADRANTS)
        figure.text(position, .15, f"{label}: shuffled {100 * shuffle:.1f}%; zero residual {100 * empty:.1f}%",
                    fontsize=8, color="#53616C")
    figure.text(.04, .06,
                "Held-out queries: XML, CSV, and dialogue; macro average across three correlated views per fact.\n"
                "Held-out observations: one of those three formats per fact, in a mixed bank. "
                f"Development-selected checkpoints; one training seed ({config['seed']}); uncertainty not shown.\n"
                "Matched LM examples and update budgets; prompt lengths and consistency computation differ. "
                "Synthetic 16-word vocabulary; no claim of general natural-language transfer.",
                fontsize=8, color="#53616C", va="bottom", linespacing=1.65)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png, svg = Path(str(output_prefix) + ".png"), Path(str(output_prefix) + ".svg")
    metadata = {"Title": "VeRA-Mem template generalization",
                "Description": "Completed, prediction-audited matched conditions; input summary SHA256=" + input_sha256}
    figure.savefig(png, dpi=240, metadata=metadata)
    figure.savefig(svg, metadata=metadata)
    plt.close(figure)
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    png, svg = draw(json.loads(raw), args.output_prefix, hashlib.sha256(raw).hexdigest())
    print(json.dumps({"png": str(png), "svg": str(svg), "completed_conditions": 3}))


if __name__ == "__main__":
    main()
