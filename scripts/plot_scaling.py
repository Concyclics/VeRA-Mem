"""Plot completed summarize_scaling.py results as PNG and editable SVG.

Missing planned conditions are marked pending; smoke results are excluded.
Ambiguous duplicate conditions and incompatible evidence raise an error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="vera_mem_mpl_")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np


COLORS = {"raw": "#A35A32", "stable": "#0072B2"}
DISPLAY = {"raw": "Raw recipe", "stable": "Stable recipe"}


def completed_runs(summary: dict) -> list[dict]:
    rows = summary.get("runs")
    if not isinstance(rows, list) or summary.get("completed_run_count") != len(rows):
        raise ValueError("Input must be a complete summarize_scaling.py summary schema")
    results = []
    for row in rows:
        if row.get("train_size") == 32 or "smoke" in row.get("run", "").lower():
            continue
        # The summarizer admits only manifest-complete and prediction-verified
        # runs. Honour an explicit incompleteness marker if another producer
        # adds one, but never infer completion from a nonzero metric alone.
        if row.get("complete") is False or row.get("status", "complete") != "complete":
            continue
        evidence = row.get("evidence_sha256")
        if not isinstance(evidence, dict) or not {
            "config.json", "manifest.json", "metrics.json", "predictions.jsonl",
        } <= set(evidence):
            raise ValueError("Run lacks the summarizer's complete-evidence hashes: " + row.get("run", "?"))
        results.append(row)
    return results


def unique_run(rows: list[dict], label: str, **condition) -> dict | None:
    matches = [row for row in rows if all(row.get(key) == value for key, value in condition.items())]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous duplicate candidates for {label}: " + ", ".join(row["run"] for row in matches))
    return matches[0] if matches else None


def metric(row: dict | None, method: str) -> dict | None:
    if row is None:
        return None
    result = row.get("phases", {}).get("final/" + method)
    if result is None:
        raise ValueError(f"Complete run lacks final/{method}: {row['run']}")
    n, correct, em = result.get("n"), result.get("correct"), result.get("em")
    if (not isinstance(n, int) or isinstance(n, bool) or n <= 0
            or not isinstance(correct, int) or isinstance(correct, bool) or not 0 <= correct <= n
            or not isinstance(em, (int, float)) or not math.isfinite(em)
            or not math.isclose(em, correct / n, abs_tol=1e-9)):
        raise ValueError("Inconsistent exact-match count: " + row["run"])
    return result


def select_panels(rows: list[dict]) -> tuple[dict, dict, dict]:
    base = dict(batch_size=8, alignment_steps=400, eval_size=128)
    size_panel = {}
    for variant, sizes in (("raw", (128, 4096)), ("stable", (128, 1024, 4096))):
        for size in sizes:
            size_panel[variant, size] = unique_run(
                rows, f"size/{variant}/{size}", **base, variant=variant, train_size=size,
                training_lm_updates=512, oracle_updates=256, training_cold_records=0,
                deployment_cold_records=0, evaluation_only=False,
            )
    budget_panel = {}
    for updates in (512, 1536):
        budget_panel[updates] = unique_run(
            rows, f"budget/{updates}", **base, variant="stable", train_size=4096,
            training_lm_updates=updates, oracle_updates=updates // 2,
            training_cold_records=0, deployment_cold_records=0, evaluation_only=False,
        )
    cold_panel = {}
    for training in (0, 128):
        for deployment in (0, 128):
            cold_panel[training, deployment] = unique_run(
                rows, f"cold/train{training}/deploy{deployment}", **base,
                variant="stable", train_size=4096, training_lm_updates=512,
                oracle_updates=256, training_cold_records=training,
                deployment_cold_records=deployment,
                evaluation_only=(training != deployment),
            )
    selected = {row["run"]: row for panel in (size_panel, budget_panel, cold_panel)
                for row in panel.values() if row is not None}
    for field in ("source_cache_group_sha256", "model_revision", "training_seed"):
        values = {row.get(field) for row in selected.values()}
        if None in values or len(values) > 1:
            raise ValueError(f"Figure conditions must have matched {field}; split incompatible summaries")
    for training in (0, 128):
        row_pair = [cold_panel[training, deployment] for deployment in (0, 128)]
        if all(row is not None for row in row_pair):
            hashes = {row.get("checkpoint_sha256") for row in row_pair}
            if None in hashes or len(hashes) != 1:
                raise ValueError(f"Cold-start deployment comparison must share one checkpoint: train={training}")
    return size_panel, budget_panel, cold_panel


def style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8D949A")
    ax.tick_params(colors="#343A40", labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E1E5E8", linewidth=.7)


def pending_box(ax, labels: list[str]) -> None:
    if labels:
        ax.text(.03, .97, "Pending: " + "; ".join(labels), transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color="#666A70", wrap=True,
                bbox=dict(boxstyle="round,pad=.35", facecolor="#F0F1F2", edgecolor="none"))


def draw(summary: dict, output_prefix: Path, input_sha256: str) -> tuple[Path, Path]:
    rows = completed_runs(summary)
    size_panel, budget_panel, cold_panel = select_panels(rows)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 6.),
                                gridspec_kw={"width_ratios": [1.35, .95, 1.]})
    figure.subplots_adjust(left=.06, right=.965, top=.79, bottom=.32, wspace=.38)
    figure.suptitle("VeRA-Mem: data scale, training budget, and cold start",
                   x=.06, y=.975, ha="left", fontsize=15, fontweight="semibold")
    figure.text(.06, .92, "Qwen3-4B-Instruct-2507  |  Held-out synthetic facts  |  Completed runs only",
                fontsize=10, color="#56616A")

    ax = axes[0]
    style_axis(ax)
    ax.set_title("A  Training-data scale", loc="left", pad=29, fontweight="semibold")
    ax.text(0, 1.04, "512 LM updates; 4,096 target exposures", transform=ax.transAxes,
            fontsize=9, color="#56616A")
    locations = {128: 128, 1024: 1024, 4096: 4096}
    missing = []
    for variant, sizes in (("raw", (128, 4096)), ("stable", (128, 1024, 4096))):
        x = [locations[size] for size in sizes]
        for method in ("real", "oracle"):
            values = [metric(size_panel[variant, size], method) for size in sizes]
            y = [100 * value["em"] if value is not None else np.nan for value in values]
            ax.plot(x, y, color=COLORS[variant],
                    linestyle="-" if method == "real" else "--",
                    marker="o" if method == "real" else "s", markersize=5,
                    linewidth=2 if method == "real" else 1.4,
                    alpha=1 if method == "real" else .43, zorder=3 if method == "real" else 2)
            if method == "real":
                for position, value in zip(x, values):
                    if value is not None:
                        # Offset the two recipes in opposite directions to
                        # keep coincident or nearly coincident labels readable.
                        horizontal = -10 if variant == "raw" else 10
                        vertical = 10 if variant == "stable" or value["em"] < .08 else -15
                        ax.annotate(f"{value['correct']}/{value['n']}",
                                    (position, 100 * value["em"]),
                                    xytext=(horizontal, vertical),
                                    textcoords="offset points", fontsize=8, color=COLORS[variant],
                                    ha="left" if variant == "stable" else "right", clip_on=False)
        pending = [str(size) for size in sizes if size_panel[variant, size] is None]
        if pending:
            missing.append(variant.capitalize() + " " + ", ".join(pending))
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 1024, 4096], ["128", "1,024", "4,096"])
    ax.set_xlim(86, 6000)
    ax.set_ylim(-5, 106)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Final exact match (%)")
    ax.set_xlabel("Offline training facts (log scale)", labelpad=9)
    pending_box(ax, missing)
    legend = [Line2D([0], [0], color=COLORS[variant], marker="o", lw=2,
                     label=DISPLAY[variant] + ": real retrieval") for variant in ("raw", "stable")]
    legend.append(Line2D([0], [0], color="#6A737B", marker="s", linestyle="--", alpha=.55,
                        label="Dashed: forced value diagnostic"))
    figure.legend(handles=legend, loc="lower left", bbox_to_anchor=(.055, .13), ncol=3,
                  frameon=False, fontsize=8, handlelength=2.5, borderaxespad=0.)

    ax = axes[1]
    style_axis(ax)
    ax.set_title("B  Training budget", loc="left", pad=29, fontweight="semibold")
    ax.text(0, 1.04, "Stable recipe; 4,096 training facts", transform=ax.transAxes,
            fontsize=9, color="#56616A")
    for position, updates in enumerate((512, 1536)):
        value = metric(budget_panel[updates], "real")
        if value is None:
            ax.bar(position, 100, color="#F1F2F3", edgecolor="#D6DADD", width=.58, hatch="//")
            ax.text(position, 50, "pending", rotation=90, ha="center", va="center", color="#777C82")
        else:
            y = 100 * value["em"]
            ax.bar(position, y, color=COLORS["stable"], width=.58, alpha=.9)
            ax.text(position, y + 3, f"{y:.1f}%\n{value['correct']}/{value['n']}",
                    ha="center", va="bottom", fontsize=9, color="#243746")
    ax.set_xticks([0, 1], ["512\n(1×)", "1,536\n(3×)"])
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-.58, 1.58)
    ax.set_ylabel("Final real-retrieval EM (%)")
    ax.set_xlabel("LM updates (training budget)", labelpad=8)

    ax = axes[2]
    ax.set_title("C  Cold-start cross-evaluation", loc="left", pad=29, fontweight="semibold")
    ax.text(0, 1.04, "Stable; 4,096 facts; 512 LM updates", transform=ax.transAxes,
            fontsize=9, color="#56616A")
    cells = np.full((2, 2), np.nan)
    for r, training in enumerate((0, 128)):
        for c, deployment in enumerate((0, 128)):
            value = metric(cold_panel[training, deployment], "real")
            if value is not None:
                cells[r, c] = 100 * value["em"]
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#ECEEEF")
    ax.imshow(np.ma.masked_invalid(cells), cmap=cmap, norm=Normalize(0, 100), aspect="auto")
    for r, training in enumerate((0, 128)):
        for c, deployment in enumerate((0, 128)):
            value = metric(cold_panel[training, deployment], "real")
            label = "pending" if value is None else f"{100 * value['em']:.1f}%\n{value['correct']}/{value['n']}"
            color = "#747B82" if value is None else ("white" if cells[r, c] >= 60 else "#15354A")
            ax.text(c, r, label, ha="center", va="center", fontsize=11, color=color)
    ax.set_xticks([0, 1], ["0", "128"])
    ax.set_yticks([0, 1], ["0", "128"])
    ax.set_xlabel("Deployment initial bank (records)", labelpad=9)
    ax.set_ylabel("Training background bank (records)")
    ax.set_xticks([-.5, .5, 1.5], minor=True)
    ax.set_yticks([-.5, .5, 1.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(.5, -.24, "Same checkpoint within each row", transform=ax.transAxes,
            ha="center", va="top", fontsize=8, color="#56616A")

    figure.text(.06, .055,
                "Dev-selected checkpoints; budgets denote full training runs. Forced value is not an upper bound.\n"
                "One training seed; no training-seed uncertainty shown. Raw / stable are different training recipes.",
                fontsize=8, color="#56616A", va="bottom", linespacing=1.6)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png, svg = Path(str(output_prefix) + ".png"), Path(str(output_prefix) + ".svg")
    provenance = "Aggregated completed results; input summary SHA256=" + input_sha256
    figure.savefig(png, dpi=240, metadata={"Title": "VeRA-Mem scaling ablations", "Description": provenance})
    figure.savefig(svg, metadata={"Title": "VeRA-Mem scaling ablations", "Description": provenance})
    plt.close(figure)
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="summary.json from summarize_scaling.py")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    summary = json.loads(raw)
    png, svg = draw(summary, args.output_prefix, hashlib.sha256(raw).hexdigest())
    print(json.dumps({"png": str(png), "svg": str(svg), "completed_input_runs": summary["completed_run_count"]}))


if __name__ == "__main__":
    main()
