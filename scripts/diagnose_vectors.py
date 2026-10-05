#!/usr/bin/env python3
"""Inspect a saved vector bank for collapsed or saturated value encodings.

CPU only; does not modify the bank/checkpoint, load a language model, or fit any
parameters. Example:

    python scripts/diagnose_vectors.py --bank runs/example/vdb.pt \
        --checkpoint runs/example/vector_vera.pt --output diagnostics.json

Pairwise/SVD statistics use at most --max-pair-rows evenly spaced observations.
Other statistics, including exact unique rows, use the full saved bank.
Activation-specific saturation/derivative statistics are emitted only for
verified or explicitly selected tanh values. RMS-normalized values may exceed
one coordinate-wise; unknown activations receive only shared matrix statistics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: Tensor) -> dict:
    values = values.detach().cpu().double().flatten()
    finite = values[torch.isfinite(values)]
    result = {"count": values.numel(), "nonfinite_count": values.numel() - finite.numel()}
    if finite.numel() == 0:
        return result
    quantiles = torch.quantile(finite, torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64))
    result.update(zip(("min", "p25", "median", "p75", "max"), quantiles.tolist()))
    result["mean"] = finite.mean().item()
    return result


def matrix_diagnostics(matrix: Tensor, max_pair_rows: int) -> dict:
    matrix = matrix.detach().cpu().float()
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError("Stored keys/values must be finite 2-D tensors")
    count, dimension = matrix.shape
    result = {"shape": [count, dimension], "numeric_bytes": matrix.numel() * matrix.element_size()}
    if not count:
        result["unique_rows_exact"] = 0
        return result
    standard_deviation = matrix.std(dim=0, correction=0)
    norms = matrix.norm(dim=-1)
    result.update(
        unique_rows_exact=len(torch.unique(matrix, dim=0)),
        dimension_std=distribution(standard_deviation),
        constant_dimensions_exact=int((standard_deviation == 0).sum()),
        dimensions_std_below_1e_6=int((standard_deviation < 1e-6).sum()),
        row_l2_norm=distribution(norms),
        row_rms=distribution(matrix.square().mean(dim=-1).sqrt()),
        elements=distribution(matrix),
        zero_norm_rows=int((norms == 0).sum()),
    )
    indices = torch.linspace(0, count - 1, min(count, max_pair_rows)).round().long()
    sample = matrix[indices]
    normalized = F.normalize(sample, dim=-1)
    pairs = torch.triu_indices(len(sample), len(sample), offset=1)
    valid = (sample.norm(dim=-1)[pairs[0]] > 0) & (sample.norm(dim=-1)[pairs[1]] > 0)
    cosine = (normalized @ normalized.T)[pairs[0], pairs[1]][valid]
    result["pairwise_scope"] = {
        "rows": len(sample), "all_rows": len(sample) == count,
        "selection": "evenly_spaced_in_storage_order", "valid_pairs": len(cosine),
    }
    result["pair_cosine"] = distribution(cosine)
    if len(cosine):
        result["pair_cosine_fraction_ge_0.999"] = float((cosine >= .999).float().mean())
    singular = torch.linalg.svdvals(sample.double() - sample.double().mean(dim=0, keepdim=True))
    energy = singular.square()
    total = energy.sum().item()
    if total > 0:
        proportion = energy / total
        positive = proportion[proportion > 0]
        result["centered_spectrum"] = {
            "total_energy": total,
            "effective_rank": float(torch.exp(-(positive * positive.log()).sum())),
            "leading_5_variance_share": proportion[:5].tolist(),
        }
    else:
        result["centered_spectrum"] = {
            "total_energy": 0., "effective_rank": 0., "leading_5_variance_share": [],
        }
    return result


def resolve_activation(checkpoint: dict | None, requested: str) -> tuple[str, dict]:
    """Use checkpoint architecture evidence, never the observed value range."""
    if requested not in {"auto", "tanh", "rmsnorm"}:
        raise ValueError("value activation must be auto, tanh, or rmsnorm")
    if requested != "auto":
        return requested, {"source": "explicit_cli_override", "requested": requested}
    if checkpoint is None:
        return "unknown", {"source": "unidentified", "reason": "No model checkpoint supplied"}
    config = checkpoint.get("config", {})
    config = config if isinstance(config, dict) else {}
    parameters = checkpoint.get("module", checkpoint.get("state_dict", {}))
    parameters = parameters if isinstance(parameters, dict) else {}
    evidence = []
    variant = config.get("variant")
    if variant == "stable":
        evidence.append(("rmsnorm", "checkpoint.config.variant=stable"))
    elif variant == "raw":
        evidence.append(("tanh", "checkpoint.config.variant=raw"))
    if {"query_center", "support_center", "statistics_fitted"} <= set(parameters):
        evidence.append(("rmsnorm", "StableVectorVeRA training-statistics buffers"))
    # These are the two legacy VectorVeRA config formats used by this repo:
    # complete offline-run settings, or its constructor configuration. Require
    # the matching module state as well; an arbitrary centering flag is not
    # evidence of an activation. An unknown named variant is not legacy.
    signature = {"A", "B", "Wq.weight", "Wk.weight", "Wv.weight", "b"}
    legacy_run = {"rank", "key_dim", "offline_train_size", "offline_dev_size", "offline_epochs"}
    legacy_constructor = {"in_features", "out_features", "rank", "key_dim", "value_centering"}
    if variant is None and signature <= set(parameters) and (
        legacy_run <= set(config) or legacy_constructor <= set(config)
    ) and not {"query_center", "support_center", "statistics_fitted"} <= set(parameters):
        evidence.append(("tanh", "recognized legacy VectorVeRA config and state signature"))
    kinds = {kind for kind, _ in evidence}
    provenance = {"source": "checkpoint", "evidence": [item for _, item in evidence]}
    if len(kinds) == 1:
        return next(iter(kinds)), provenance
    provenance.update(
        source="unidentified",
        reason="Conflicting activation evidence" if kinds else "No recognized activation evidence",
    )
    return "unknown", provenance


def diagnose(bank_path: Path, checkpoint_path: Path | None, threshold: float,
             max_pair_rows: int, value_activation: str = "auto") -> dict:
    state = torch.load(bank_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not {"keys", "values"} <= set(state):
        raise ValueError("Bank checkpoint needs keys and values tensor fields")
    keys, values = state["keys"], state["values"]
    if keys.shape[0] != values.shape[0]:
        raise ValueError("Bank key/value counts differ")
    checkpoint = None
    parameters = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Model checkpoint must be a mapping")
        parameters = checkpoint.get("module", checkpoint.get("state_dict"))
        if not isinstance(parameters, dict):
            raise ValueError("Model checkpoint needs a module or state_dict mapping")
    activation, provenance = resolve_activation(checkpoint, value_activation)
    result = {
        "diagnosed_at": datetime.now(timezone.utc).isoformat(),
        "bank": {"name": bank_path.name, "sha256": sha256(bank_path)},
        "keys": matrix_diagnostics(keys, max_pair_rows),
        "values": matrix_diagnostics(values, max_pair_rows),
        "value_activation": activation,
        "value_activation_provenance": provenance,
    }
    if values.numel():
        result["values"]["all_rows_identical"] = result["values"]["unique_rows_exact"] == 1
    if activation == "tanh" and values.numel():
        if (values.abs() > 1).any():
            raise ValueError("Tanh diagnostics require values in [-1, 1]; check activation provenance")
        saturated = values.abs() >= threshold
        result["values"].update(
            saturation_threshold=threshold,
            saturation_fraction=float(saturated.float().mean()),
            fully_saturated_rows=int(saturated.all(dim=-1).sum()),
            tanh_derivative_from_output=distribution(1. - values.double().square()),
        )
    elif activation != "tanh":
        result["activation_specific_metrics_skipped"] = (
            "RMS-normalized coordinates are not tanh outputs"
            if activation == "rmsnorm" else "Value activation could not be identified"
        )
    if checkpoint_path is not None:
        result["checkpoint"] = {
            "name": checkpoint_path.name, "sha256": sha256(checkpoint_path),
            "selected_epoch": checkpoint.get("selected_epoch"),
            "selected_step": checkpoint.get("step"), "parameters": {},
        }
        for name in ("Wq.weight", "Wk.weight", "Wv.weight", "b", "value_center", "query_center", "support_center"):
            if name in parameters:
                value = parameters[name].detach().cpu().float()
                stats = {"shape": list(value.shape), "l2_norm": float(value.norm()),
                         "absolute_elements": distribution(value.abs())}
                if value.ndim == 2:
                    stats["row_l2_norm"] = distribution(value.norm(dim=-1))
                result["checkpoint"]["parameters"][name] = stats
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--value-activation", choices=["auto", "tanh", "rmsnorm"], default="auto",
                        help="Auto inspects checkpoint architecture; unidentified values skip tanh-only metrics")
    parser.add_argument("--saturation-threshold", type=float, default=.999)
    parser.add_argument("--max-pair-rows", type=int, default=2048)
    args = parser.parse_args()
    if not 0 < args.saturation_threshold <= 1 or args.max_pair_rows < 1:
        parser.error("threshold must be in (0, 1] and max-pair-rows must be positive")
    torch.set_num_threads(1)
    result = diagnose(args.bank, args.checkpoint, args.saturation_threshold, args.max_pair_rows,
                      value_activation=args.value_activation)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
