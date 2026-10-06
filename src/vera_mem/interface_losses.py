"""Fixed-scale, unclipped counterfactual behavior supervision.

Calibrate once on a preselected *training* teacher sample, freeze the returned
scalar, then use it unchanged across arms, batches, and evaluation. The helper
cannot establish sample provenance; the caller must record IDs and cache hashes.
Neither candidate-margin matching nor its teacher gate certifies full-vocabulary
or complete-answer accuracy. Keep the common gold/FKL anchors and evaluate both
worlds jointly. These objectives do not establish an on-policy sampling method.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F

from .counterfactual_losses import pair_behavior_loss


def _fixed_positive(name: str, value: float) -> float:
    # A batch tensor must not silently become an adaptive or learnable scale.
    if (not isinstance(value, Real) or isinstance(value, bool)
            or not math.isfinite(float(value)) or value <= 0):
        raise ValueError(f"{name} must be a fixed, finite positive scalar")
    return float(value)


def _distribution(values: Tensor) -> dict[str, float]:
    """Small JSON-safe population summary; caller supplies nonempty CPU doubles."""
    quantiles = torch.quantile(values, torch.tensor(
        [0., .05, .25, .5, .75, .95, 1.], dtype=values.dtype))
    return {
        **dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                   map(float, quantiles))),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
    }


def fit_margin_scale(teacher_delta: Tensor, valid_mask: Tensor, *,
                     split: str = "train", min_scale: float = 1e-3,
                     legacy_target_clip: float = 10.) -> dict:
    """Fit median absolute teacher effect on valid training pairs only.

    ``teacher_delta`` is the unclipped effect at the SAME temperature later used
    by ``normalized_behavior_loss``. ``valid_mask`` is the existing candidate
    teacher gate, not a mask chosen from student outcomes. An invalid teacher is
    excluded from calibration but retained in the all-pair clipping diagnostic.
    An even-size median averages the middle two values. A positive floor protects
    degenerate scales; it does not force target variance or discard large effects.

    The result is JSON serializable, including raw/normalized target distributions
    and legacy clipping rates with explicit all-pair and valid-pair denominators.
    This function rejects non-training split labels but cannot verify those labels.
    """
    if split != "train":
        raise ValueError("Margin scale must be calibrated on training examples only")
    minimum = _fixed_positive("min_scale", min_scale)
    clip = _fixed_positive("legacy_target_clip", legacy_target_clip)
    if (not isinstance(teacher_delta, Tensor) or teacher_delta.ndim != 1
            or teacher_delta.numel() == 0 or not teacher_delta.is_floating_point()):
        raise ValueError("teacher_delta must be a nonempty floating-point vector")
    if (not isinstance(valid_mask, Tensor) or valid_mask.shape != teacher_delta.shape
            or valid_mask.dtype != torch.bool):
        raise ValueError("valid_mask must be a matching boolean vector")
    values = teacher_delta.detach().to(device="cpu", dtype=torch.float64)
    valid = valid_mask.detach().to(device="cpu")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Teacher effects must be finite, including invalid pairs")
    if not bool(valid.any()):
        raise ValueError("Cannot calibrate without a valid teacher pair")
    selected = values[valid]
    if bool((selected <= 0).any()):
        raise ValueError("Valid teacher effects must be positive under the candidate gate")
    median = float(torch.quantile(selected.abs(), .5))
    scale = max(median, minimum)
    clipped = values.abs() > clip
    return {
        "scale": scale,
        "calibration_scope": "train",
        "estimator": "median_absolute_valid_teacher_delta",
        "count": values.numel(),
        "valid_count": int(valid.sum()),
        "valid_fraction": float(valid.double().mean()),
        "minimum_scale": minimum,
        "unfloored_scale": median,
        "floor_applied": median < minimum,
        "legacy_target_clip": clip,
        "legacy_clipped_count_all": int(clipped.sum()),
        "legacy_clipped_fraction_all": float(clipped.double().mean()),
        "legacy_clipped_count_valid": int(clipped[valid].sum()),
        "legacy_clipped_fraction_valid": float(clipped[valid].double().mean()),
        "teacher_delta_distribution_all": _distribution(values),
        "teacher_delta_distribution_valid": _distribution(selected),
        "teacher_target_normalized_distribution_valid": _distribution(selected / scale),
        "legacy_target_distribution_valid": _distribution(selected.clamp(-clip, clip)),
    }


def normalized_behavior_loss(
        student_logits_a: Tensor, student_logits_b: Tensor,
        teacher_logits_a: Tensor, teacher_logits_b: Tensor,
        answer_a_ids: Tensor, answer_b_ids: Tensor, scale: float, *,
        temperature: float = 1., huber_beta: float = 1., gold_weight: float = 1.,
        min_teacher_effect: float = 1e-3, legacy_target_clip: float = 10.,
        reduction: str = "mean") -> dict[str, Tensor]:
    """Match unclipped teacher effect with a fixed normalized Huber transition.

    For each valid pair, loss = scale * SmoothL1(delta_S / scale,
    stopgrad(delta_T) / scale, beta=huber_beta). This is exactly raw-effect
    SmoothL1 with beta=scale*huber_beta. The factor ``scale`` retains the raw
    effect derivative bound of one before loss weighting/reduction (each logit
    derivative additionally scales by 1/temperature). It does not guarantee
    equal parameter gradients or training cost across objectives. No example-wise
    division, clipping, or batch-wise scale fitting is performed.

    The original candidate teacher gate is evaluated in raw effect units, and
    the original gold CE remains unscaled and applies even to invalid teachers.
    Return fields shared with ``pair_behavior_loss`` retain their raw units.
    ``clipped_mask`` is always false; ``legacy_clipped_mask`` reports what the old
    clipping rule would have discarded. All diagnostic tensors are detached.
    ``delta_loss`` excludes gold, allowing a runner to retain its existing common
    anchors without counting them twice. Teachers receive no gradients.
    """
    fixed_scale = _fixed_positive("scale", scale)
    clip = _fixed_positive("legacy_target_clip", legacy_target_clip)
    result = pair_behavior_loss(
        student_logits_a, student_logits_b, teacher_logits_a, teacher_logits_b,
        answer_a_ids, answer_b_ids, temperature=temperature, target_clip=None,
        huber_beta=huber_beta, gold_weight=gold_weight,
        min_teacher_effect=min_teacher_effect, reduction=reduction)
    # Reuse the original validation, *raw-unit* teacher gate, gold anchors and
    # diagnostics. Reconstruct only student effects because diagnostics detach.
    rows = torch.arange(student_logits_a.shape[0], device=student_logits_a.device)
    ids_a = torch.as_tensor(answer_a_ids, device=rows.device).long()
    ids_b = torch.as_tensor(answer_b_ids, device=rows.device).long()
    margin_a = (student_logits_a[rows, ids_a].float()
                - student_logits_a[rows, ids_b].float()) / temperature
    margin_b = (student_logits_b[rows, ids_a].float()
                - student_logits_b[rows, ids_b].float()) / temperature
    normalized_student = (margin_a - margin_b) / fixed_scale
    normalized_teacher = result["teacher_delta_raw"] / fixed_scale
    per_example = fixed_scale * F.smooth_l1_loss(
        normalized_student, normalized_teacher, beta=huber_beta, reduction="none")
    per_example = per_example.masked_fill(~result["valid_mask"], 0.)
    if reduction == "none":
        delta_loss = per_example
    elif reduction == "sum":
        delta_loss = per_example.sum()
    else:  # pair_behavior_loss already validates reduction.
        delta_loss = per_example.sum() / result["valid_count"].clamp_min(1)
    result.update({
        "loss": delta_loss + gold_weight * result["gold_loss"],
        "delta_loss": delta_loss,
        "per_example": per_example + gold_weight * result["gold_per_example"],
        "delta_per_example": per_example,
        "student_delta_normalized": normalized_student.detach(),
        "teacher_delta_normalized": normalized_teacher.detach(),
        "teacher_target_normalized": normalized_teacher.detach(),
        "scale": normalized_teacher.new_tensor(fixed_scale),
        "legacy_clipped_mask": result["teacher_delta_raw"].abs() > clip,
    })
    return result
