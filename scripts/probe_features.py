"""CPU-only auxiliary linear probes of frozen support/query layer inputs.

This diagnoses whether an offline answer is decodable from a feature. It is
neither the VeRA-Mem model nor an online-memory accuracy result. No online split
is evaluated, and no raw text, record ID, or per-example prediction is exported.
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
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import torch
from torch import nn
from torch.nn import functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def initial_normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1, eps=1e-12) * math.sqrt(features.shape[-1])


def normalize_with_center(features: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    centered = initial_normalize(features) - center
    return centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-6)


def wilson_interval(correct: int, count: int) -> list[float]:
    z = 1.959963984540054
    p = correct / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0., center - half), min(1., center + half)]


@torch.no_grad()
def evaluate(probe: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> dict:
    logits = probe(features)
    predictions = logits.argmax(-1)
    correct = int((predictions == labels).sum())
    classes = logits.shape[-1]
    counts = torch.bincount(labels * classes + predictions, minlength=classes * classes)
    return dict(
        count=len(labels), correct=correct, accuracy=correct / len(labels),
        mean_cross_entropy=float(F.cross_entropy(logits, labels)),
        accuracy_wilson_95=wilson_interval(correct, len(labels)),
        label_counts=torch.bincount(labels, minlength=classes).tolist(),
        prediction_counts=torch.bincount(predictions, minlength=classes).tolist(),
        confusion_matrix=counts.reshape(classes, classes).tolist(),
    )


def run_probe(domain: str, size: int, data: dict, labels: dict, config: dict) -> dict:
    started = time.perf_counter()
    raw_train = data["train"][domain][:size]
    raw_dev = data["dev"][domain]
    # Each condition fits its own center solely on that condition's training
    # domain. A query probe must not borrow a support-domain center or labels.
    center = initial_normalize(raw_train).mean(0)
    train = normalize_with_center(raw_train, center)
    dev = normalize_with_center(raw_dev, center)
    train_labels = labels["train"][:size]
    torch.manual_seed(config["seed"])
    probe = nn.Linear(train.shape[-1], config["class_count"], device="cpu")
    optimizer = torch.optim.Adam(probe.parameters(), lr=config["learning_rate"])
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    order, cursor = torch.empty(0, dtype=torch.long), 0
    history = []
    for step in range(config["steps"]):
        if cursor + config["batch_size"] > len(order):
            order, cursor = torch.randperm(size, generator=generator), 0
        selected = order[cursor:cursor + config["batch_size"]]
        cursor += len(selected)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train[selected]), train_labels[selected])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite probe loss")
        loss.backward()
        optimizer.step()
        if (step + 1) % 64 == 0 or step + 1 == config["steps"]:
            history.append(dict(step=step + 1, minibatch_loss=float(loss.detach())))
    probe.eval()
    result = dict(
        domain="support" if domain == "s" else "query", train_size=size,
        parameter_count=sum(parameter.numel() for parameter in probe.parameters()),
        feature_dimension=train.shape[-1], center_fit_size=size,
        steps=config["steps"], batch_size=config["batch_size"],
        target_exposures=config["steps"] * config["batch_size"],
        train_passes=config["steps"] * config["batch_size"] / size,
        seed=config["seed"], optimizer="Adam", learning_rate=config["learning_rate"],
        weight_decay=0., selection="fixed final step; no development selection",
        normalization="training-domain mean after RMS normalization; centered RMS epsilon=1e-6",
        training=evaluate(probe, train, train_labels),
        development=evaluate(probe, dev, labels["dev"]),
        loss_history=history, elapsed_seconds=time.perf_counter() - started,
    )
    print(json.dumps({key: result[key] for key in (
        "domain", "train_size", "steps", "elapsed_seconds",
    )} | {"train_accuracy": result["training"]["accuracy"],
          "dev_accuracy": result["development"]["accuracy"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.batch_size > 128 or 128 % args.batch_size:
        parser.error("Use positive steps and a batch size dividing 128")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    packet = torch.load(args.cache, map_location="cpu", weights_only=True)
    if packet["protocol"] != "scaling-v1-final-prompt-layer20":
        raise ValueError("Unexpected feature-cache protocol")
    classes = sorted({example["answer"] for example in packet["examples"]["train"]})
    if len(classes) != 16:
        raise ValueError("This diagnostic requires the declared 16-class training vocabulary")
    class_ids = {word: i for i, word in enumerate(classes)}
    # Deliberately select only the offline training/development splits.
    data = {split: packet["features"][split] for split in ("train", "dev")}
    labels = {
        split: torch.tensor([class_ids[example["answer"]] for example in packet["examples"][split]])
        for split in ("train", "dev")
    }
    for split, required_count in (("train", 4096), ("dev", 64)):
        if len(labels[split]) != required_count:
            raise ValueError(f"Expected {required_count} offline {split} examples")
        for domain in ("s", "q"):
            feature = data[split][domain]
            if tuple(feature.shape) != (required_count, 9728) or not torch.isfinite(feature).all():
                raise ValueError(f"Invalid {split}/{domain} frozen feature matrix")
    output = dict(
        diagnostic="offline linear answer-decodability probe; NOT VeRA-Mem or online-memory accuracy",
        complete=False, started_at=datetime.now(timezone.utc).isoformat(),
        source_script_sha256=sha256(Path(__file__)), cache_sha256=sha256(args.cache),
        protocol=packet["protocol"], model_revision=packet["model_revision"],
        device="cpu", torch_threads=4, torch_interop_threads=1,
        torch_version=str(torch.__version__), python_version=platform.python_version(),
        class_count=len(classes), class_labels=classes, chance_accuracy=1 / len(classes),
        evaluated_splits=["train", "dev"], online_splits_used=False,
        dev_selection=False,
        caveats=[
            "Support observations legitimately contain the revealed answer; this is a feature-decoding diagnostic.",
            "A supervised 16-class linear head is not the VeRA value writer/reader and does not establish memory efficacy.",
            "Query accuracy near chance detects no strong held-out answer predictability; it is not proof of absence of leakage.",
            "Development has only 64 facts and one seed; Wilson intervals condition on this trained probe, not training variability.",
            "Each size/domain fits its own training-only center; the probe measures that complete fixed normalization/training recipe.",
        ], runs=[],
    )
    del packet
    config = dict(steps=args.steps, batch_size=args.batch_size, seed=args.seed,
                  learning_rate=args.learning_rate, class_count=len(classes))
    atomic_json(args.output, output)
    for domain in ("s", "q"):
        for size in (128, 4096):
            output["runs"].append(run_probe(domain, size, data, labels, config))
            atomic_json(args.output, output)
    output.update(complete=True, completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(args.output, output)


if __name__ == "__main__":
    main()
