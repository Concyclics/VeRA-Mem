"""CPU-only, training-fitted style-subspace ablation for VeRA addressing.

Compare ordinary centered features with removal of the subspace spanned by
training-template mean differences. Both conditions train identical Q/K
encoders for a fixed contrastive budget. Only offline train/development rows
are inspected; no confirmation prediction or language-model score is produced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vera_mem.stable_vector_vera import StableVectorVeRA

METHODS = ("centered", "template_mean_projection")
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def normalize_input(x):
    return F.normalize(x.float(), dim=-1, eps=1e-12) * math.sqrt(x.shape[-1])


@torch.no_grad()
def fit_training_style(raw):
    """Estimate domain mean and nuisance directions from training views only."""
    if raw.ndim != 3 or raw.shape[0] != 4096 or raw.shape[-1] != 9728:
        raise ValueError("Expected train [4096, templates, 9728]")
    means = []
    for view in range(raw.shape[1]):
        features = normalize_input(raw[:, view])
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite training features")
        means.append(features.mean(0))
    means = torch.stack(means)
    center = means.mean(0)
    differences = means.double() - center.double()
    # Eight query / four observation means imply at most seven / three
    # independent style directions. Never infer extra rank from rounding.
    _, singular, vh = torch.linalg.svd(differences, full_matrices=False)
    threshold = max(float(singular.max()) * 1e-5, 1e-10)
    rank = min(int((singular > threshold).sum()), raw.shape[1] - 1)
    basis = vh[:rank].T.float().contiguous()
    total_energy = 0.
    projected_energy = 0.
    for view in range(raw.shape[1]):
        centered = normalize_input(raw[:, view]) - center
        total_energy += float(centered.square().sum())
        projected_energy += float((centered @ basis).square().sum())
    return dict(center=center, basis=basis), dict(
        training_facts=raw.shape[0], training_views=raw.shape[1], projection_rank=rank,
        singular_values=singular.tolist(), singular_relative_cutoff=1e-5,
        projected_fraction_of_training_centered_energy=projected_energy / total_energy,
        style_mean_energy_fraction=float((means-center).square().sum()) * raw.shape[0] / total_energy,
    )


@torch.no_grad()
def transform(raw, fitted, projection):
    flattened = raw.reshape(-1, raw.shape[-1])
    output = torch.empty_like(flattened, dtype=torch.float32)
    for start in range(0, len(flattened), 512):
        value = normalize_input(flattened[start:start+512]) - fitted["center"]
        if projection:
            basis = fitted["basis"]
            value = value - (value @ basis) @ basis.T
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite transformed features")
        output[start:start+len(value)] = value
    return output.reshape(raw.shape)


def draw_views(features, indices, rng):
    views = [rng.randrange(features.shape[1]) for _ in indices]
    return features[indices, views]


@torch.no_grad()
def evaluate(module, q, s, names):
    result = {}
    truth = torch.arange(len(q))
    for qi, query_name in enumerate(names["q"]):
        query = F.normalize(module.Wq(q[:, qi]), dim=-1, eps=1e-12)
        for si, support_name in enumerate(names["s"]):
            key = F.normalize(module.Wk(s[:, si]), dim=-1, eps=1e-12)
            similarities = query @ key.T
            hit1 = int((similarities.argmax(-1) == truth).sum())
            hit4 = int((similarities.topk(4, -1).indices == truth[:, None]).any(-1).sum())
            result[query_name + "/" + support_name] = dict(
                count=len(truth), correct_at_1=hit1, correct_at_4=hit4,
                recall_at_1=hit1 / len(truth), recall_at_4=hit4 / len(truth),
                contrastive_nll=float(F.cross_entropy(similarities / .1, truth)),
            )
    return result


def run(method, training, development, fitted, names, config, output):
    started = time.perf_counter()
    projection = method == "template_mean_projection"
    q, s = [transform(training[domain], fitted[domain], projection) for domain in ("q", "s")]
    qd, sd = [transform(development[domain], fitted[domain], projection) for domain in ("q", "s")]
    # Reuse the actual VeRA initializer, including its fixed-projection RNG
    # consumption. Only Wq/Wk participate in this diagnostic.
    module = StableVectorVeRA(9728, 2560, rank=64, key_dim=64, top_k=4,
                              temperature=.05, seed=config["seed"])
    parameters = [*module.Wq.parameters(), *module.Wk.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
    fact_rng = random.Random(config["seed"] + 10)
    view_rng = random.Random(config["seed"] + 20)
    history = []
    for step in range(config["steps"]):
        indices = fact_rng.sample(range(len(q)), config["batch_size"])
        qbatch = draw_views(q, indices, view_rng)
        sbatch = draw_views(s, indices, view_rng)
        query = F.normalize(module.Wq(qbatch), dim=-1, eps=1e-12)
        key = F.normalize(module.Wk(sbatch), dim=-1, eps=1e-12)
        logits = query @ key.T / .1
        labels = torch.arange(len(indices))
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite addressing loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == config["steps"]:
            row = dict(step=step+1, loss=float(loss.detach()), elapsed_seconds=time.perf_counter()-started)
            history.append(row)
            print(json.dumps(dict(method=method, training=row)), flush=True)
    metrics = evaluate(module, qd, sd, names)
    weights = output.parent / (output.stem + "_" + method + "_weights.pt")
    torch.save(dict(Wq=module.Wq.state_dict(), Wk=module.Wk.state_dict(),
                    fitted=fitted, config=config, method=method), weights)
    result = dict(method=method, fixed_final_step=config["steps"], development=metrics,
                  loss_history=history, elapsed_seconds=time.perf_counter()-started,
                  weights_sha256=digest(weights), weights_filename=weights.name)
    print(json.dumps(dict(method=method, development=metrics)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if (args.steps < 1 or not 2 <= args.batch_size <= 4096
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("Invalid probe budget")
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing probe result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    packet = torch.load(args.cache, map_location="cpu", weights_only=True)
    if packet["protocol"] != "generalization-v1-layer20" or packet["model_revision"] != REVISION:
        raise ValueError("Unexpected frozen feature provenance")
    # Only these two splits are selected. Confirmation examples and features
    # are neither encoded, scored, nor used to fit any statistic.
    training, development = [packet["features"][split] for split in ("train", "dev")]
    names = packet["template_ids"]["dev"]
    for domain, views in (("q", 8), ("s", 4)):
        if tuple(training[domain].shape) != (4096, views, 9728):
            raise ValueError("Unexpected offline training feature shape")
        if tuple(development[domain].shape) != (64, 3, 9728):
            raise ValueError("Unexpected development feature shape")
    fitted, style_stats = {}, {}
    for domain in ("q", "s"):
        fitted[domain], style_stats[domain] = fit_training_style(training[domain])
        print(json.dumps(dict(domain=domain, fitted_style=style_stats[domain])), flush=True)
    config = dict(steps=args.steps, batch_size=args.batch_size, seed=args.seed,
                  learning_rate=args.learning_rate, contrastive_temperature=.1,
                  clipping_norm=1., optimizer="Adam", training_facts=4096,
                  query_views=8, support_views=4, key_dimension=64)
    source_root = Path(__file__).resolve().parents[1]
    result = dict(diagnostic="offline address-only training-style-subspace ablation; NOT full VeRA memory EM",
                  complete=False, started_at=datetime.now(timezone.utc).isoformat(),
                  cache_sha256=digest(args.cache), data_fingerprint=packet["data_fingerprint"],
                  model_revision=packet["model_revision"], source_script_sha256=digest(Path(__file__)),
                  module_source_sha256={name: digest(source_root / "src" / "vera_mem" / name)
                                        for name in ("stable_vector_vera.py", "vector_vera.py")},
                  device="cpu", torch_threads=4, torch_interop_threads=1,
                  torch_version=str(torch.__version__), python_version=platform.python_version(),
                  config=config, fitted_training_statistics=style_stats,
                  evaluated_splits=["dev"], confirmation_used=False, dev_selection=False,
                  chance_recall_at_1=1/64, chance_recall_at_4=4/64,
                  limitations=[
                      "Fixed final 400-step budget by default; development does not choose projection rank or checkpoint.",
                      "Two views of each domain are drawn independently; entity and view streams are matched across conditions.",
                      "Projection removes training-template mean directions, potentially also discarding useful identity information.",
                      "This probe omits the VeRA residual, value writer, language modeling and online token decoding.",
                      "Positive development addressing does not establish end-to-end generalization; negative results do not rule out other projections.",
                      "64 development facts and one initialization seed; no confirmation cases are evaluated.",
                  ], runs=[])
    del packet
    write_json(args.output, result)
    for method in METHODS:
        result["runs"].append(run(method, training, development, fitted, names, config, args.output))
        write_json(args.output, result)
    result.update(complete=True, completed_at=datetime.now(timezone.utc).isoformat())
    write_json(args.output, result)
    print("PROBE_COMPLETE " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
