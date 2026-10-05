"""Create shareable figures from completed raw runs; no raw question text."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--initial-suite", type=Path, required=True)
p.add_argument("--staged-suite", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
for suite in (a.initial_suite, a.staged_suite):
    if not json.loads((suite / "suite.json").read_text()).get("complete"):
        raise RuntimeError("Figures require completed suites")
initial = {r["method"]:r for r in json.loads((a.initial_suite / "baseline_medmcqa" / "summary.json").read_text())}
staged = {r["method"]:r for r in json.loads((a.staged_suite / "vector_synthetic" / "summary.json").read_text())}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
labels = ["Evidence\ntop-1", "Evidence\ntop-4", "Real VDB\nanswer EM", "Oracle value\nanswer EM"]
scores = [staged["vdb_real"]["final"]["retrieval_hit_at_1"], staged["vdb_real"]["final"]["retrieval_hit_at_k"], staged["vdb_real"]["final"]["em"], staged["vdb_oracle"]["final"]["em"]]
bars = axes[0].bar(labels, [100*x for x in scores], color=["#34699a", "#34699a", "#be6b38", "#be6b38"], width=.65)
for bar, score in zip(bars, scores):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f"{round(score*32)}/32", ha="center")
axes[0].set_title("Staged synthetic run: retrieval does not imply readout\nTraining seed 42, evaluation seed 44", fontsize=11)
methods = ["frozen", "lora1_r4", "lora4_r1", "lora4_r4", "text_tfidf"]
labels = ["Frozen", "LoRA\n1 x r4", "LoRA\n4 x r1", "LoRA\n4 x r4", "Text\nTF-IDF"]
scores = [initial[m]["final"]["choice_accuracy"] for m in methods]
bars = axes[1].bar(labels, [100*x for x in scores], color=["#999999", "#537f68", "#537f68", "#537f68", "#34699a"], width=.65)
for bar, score in zip(bars, scores):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f"{round(score*32)}/32", ha="center")
axes[1].set_title("MedMCQA: previously observed questions\nEvaluation seed 42; retention, not unseen-question generalization", fontsize=11)
for ax in axes:
    ax.set_ylim(0, 112)
    ax.set_ylabel("Percent")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
a.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output.with_suffix(".png"), dpi=180)
fig.savefig(a.output.with_suffix(".svg"))
svg = a.output.with_suffix(".svg")
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
plt.close(fig)
