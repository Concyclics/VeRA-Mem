"""Content-change and equivalent-observation objectives for VeRA memory.

Callers must compare the *same question before its first answer token* under
two different observed values. Subtracting states after two different gold
answer prefixes would confound the memory intervention with answer leakage.
These helpers cannot establish that causal pairing from tensor shapes alone.

Teacher tensors are always detached. Student branches both retain gradients.
Matching a difference does not identify absolute predictions: gold/FKL and
explicit fact-addressing objectives remain necessary. Paraphrase consistency
alone also admits collapse and must never serve as the only content objective.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    if (isinstance(value, bool) or not math.isfinite(float(value))
            or value < 0 or (value == 0 and not allow_zero)):
        suffix = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {suffix}")


def _matrices(name: str, *values: Tensor) -> None:
    first = values[0]
    if first.ndim != 2 or first.numel() == 0:
        raise ValueError(f"{name} must be nonempty [batch, features] matrices")
    if any(value.shape != first.shape or value.device != first.device for value in values):
        raise ValueError(f"{name} must have matching shapes and devices")
    if any(not value.is_floating_point() for value in values):
        raise ValueError(f"{name} must use floating-point tensors")


def _reduce(values: Tensor, reduction: str, valid: Tensor | None = None) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        if valid is None:
            return values.mean()
        # All-invalid batches return a differentiable zero; they supply no
        # counterfactual supervision and must be reported as zero coverage.
        return values.sum() / valid.sum().clamp_min(1)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def hidden_delta_loss(student_a: Tensor, student_b: Tensor,
                      teacher_a: Tensor, teacher_b: Tensor,
                      min_teacher_norm: float = 1e-3,
                      reduction: str = "mean", normalization_epsilon: float = 1e-8
                      ) -> dict[str, Tensor]:
    """Relative squared error of differences between L2-normalized states.

    Let dS = unit(S_A) - unit(S_B), dT = unit(T_A) - unit(T_B). Per valid
    pair, loss = ||dS - stopgrad(dT)||^2 / ||dT||^2. Teacher differences with
    norm below ``min_teacher_norm`` are excluded, not interpreted as evidence
    that unchanged student states are correct. The floor bounds amplification.
    On a valid teacher pair, zero student difference costs exactly one.

    ``mean`` averages only valid pairs; ``none`` returns masked per-example
    losses (invalid entries zero). Coverage and all diagnostics are detached.
    No separately normalized delta or cosine loss is used: those could discard
    response magnitude or produce poorly scaled gradients at zero student delta.
    """
    _matrices("Hidden states", student_a, student_b, teacher_a, teacher_b)
    _positive("min_teacher_norm", min_teacher_norm)
    _positive("normalization_epsilon", normalization_epsilon)
    delta_s = (F.normalize(student_a.float(), dim=-1, eps=normalization_epsilon)
               - F.normalize(student_b.float(), dim=-1, eps=normalization_epsilon))
    delta_t = (F.normalize(teacher_a.detach().float(), dim=-1, eps=normalization_epsilon)
               - F.normalize(teacher_b.detach().float(), dim=-1, eps=normalization_epsilon))
    teacher_energy = delta_t.square().sum(-1)
    valid = teacher_energy >= min_teacher_norm ** 2
    per_example = ((delta_s - delta_t).square().sum(-1)
                   / teacher_energy.clamp_min(min_teacher_norm ** 2))
    per_example = per_example.masked_fill(~valid, 0.)
    return {
        "loss": _reduce(per_example, reduction, valid),
        "per_example": per_example,
        "valid_mask": valid.detach(),
        "valid_count": valid.sum().detach(),
        "teacher_delta_norm": teacher_energy.sqrt().detach(),
        "student_delta_norm": delta_s.detach().norm(dim=-1),
        "delta_cosine": F.cosine_similarity(delta_s.detach(), delta_t, dim=-1,
                                             eps=normalization_epsilon).detach(),
    }


def _candidate_ids(ids, batch: int, vocabulary: int, device: torch.device, name: str) -> Tensor:
    result = torch.as_tensor(ids, device=device)
    if result.shape != (batch,) or result.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must contain one integer token ID per example")
    if bool(((result < 0) | (result >= vocabulary)).any()):
        raise ValueError(f"{name} contains a token outside the vocabulary")
    return result.long()


def pair_behavior_loss(student_logits_a: Tensor, student_logits_b: Tensor,
                       teacher_logits_a: Tensor, teacher_logits_b: Tensor,
                       answer_a_ids: Tensor, answer_b_ids: Tensor,
                       temperature: float = 1., target_clip: float | None = 10.,
                       huber_beta: float = 1., gold_weight: float = 1.,
                       min_teacher_effect: float = 1e-3,
                       reduction: str = "mean") -> dict[str, Tensor]:
    """First-token candidate-margin change plus full-vocabulary gold anchors.

    In *both* branches margin = (logit(answer_A)-logit(answer_B))/temperature,
    exactly the difference of the corresponding log probabilities. The memory
    effect is margin_A - margin_B. SmoothL1 matches student effect to detached
    teacher effect, clipped to +/- ``target_clip`` when that parameter is set.
    Student effects are not clipped, so gradients persist outside that interval.
    Teacher-only clipping can oppose excess student confidence; use the delta
    term as a small auxiliary objective and report ``clipped_mask`` frequency.

    Delta supervision is valid only if teacher ranks A above B in branch A,
    ranks B above A in branch B, and has at least ``min_teacher_effect``. This
    pairwise gate does NOT certify teacher accuracy against all other tokens.
    Gold CE uses the complete vocabulary at temperature 1 in both branches,
    averages those branches, and applies even when the teacher gate rejects a
    pair. It prevents solving only the A/B contrast while preferring a third
    token. Gold+FKL supervision and complete-answer scoring belong in the runner.

    First-token IDs must differ. Shared prefixes require a separately aligned
    sequence-level comparison, not this function. For ``mean``, delta averages
    valid pairs while gold averages all pairs; ``none`` returns their per-pair
    weighted sum. Components are returned separately for explicit weighting.
    """
    _matrices("Logits", student_logits_a, student_logits_b, teacher_logits_a, teacher_logits_b)
    _positive("temperature", temperature)
    _positive("huber_beta", huber_beta)
    _positive("gold_weight", gold_weight, allow_zero=True)
    _positive("min_teacher_effect", min_teacher_effect)
    if target_clip is not None:
        _positive("target_clip", target_clip)
    batch, vocabulary = student_logits_a.shape
    ids_a = _candidate_ids(answer_a_ids, batch, vocabulary, student_logits_a.device, "answer_a_ids")
    ids_b = _candidate_ids(answer_b_ids, batch, vocabulary, student_logits_a.device, "answer_b_ids")
    if bool((ids_a == ids_b).any()):
        raise ValueError("Counterfactual answers must have different first-token IDs")
    rows = torch.arange(batch, device=student_logits_a.device)

    def margin(logits):
        # The shared logsumexp normalizer cancels exactly; do not materialize a
        # full-vocabulary logsoftmax just to subtract these two log probabilities.
        return (logits[rows, ids_a].float() - logits[rows, ids_b].float()) / temperature

    margin_sa, margin_sb = margin(student_logits_a), margin(student_logits_b)
    margin_ta = margin(teacher_logits_a.detach())
    margin_tb = margin(teacher_logits_b.detach())
    delta_s, delta_t = margin_sa - margin_sb, margin_ta - margin_tb
    valid = (margin_ta > 0) & (margin_tb < 0) & (delta_t >= min_teacher_effect)
    target = delta_t if target_clip is None else delta_t.clamp(-target_clip, target_clip)
    clipped = torch.zeros_like(valid) if target_clip is None else delta_t.abs() > target_clip
    delta_per_example = F.smooth_l1_loss(delta_s, target, beta=huber_beta, reduction="none")
    delta_per_example = delta_per_example.masked_fill(~valid, 0.)
    gold_per_example = (
        F.cross_entropy(student_logits_a.float(), ids_a, reduction="none")
        + F.cross_entropy(student_logits_b.float(), ids_b, reduction="none")) / 2
    delta_loss = _reduce(delta_per_example, reduction, valid)
    gold_loss = _reduce(gold_per_example, reduction)
    return {
        "loss": delta_loss + gold_weight * gold_loss,
        "delta_loss": delta_loss,
        "gold_loss": gold_loss,
        "per_example": delta_per_example + gold_weight * gold_per_example,
        "delta_per_example": delta_per_example,
        "gold_per_example": gold_per_example,
        "valid_mask": valid.detach(),
        "valid_count": valid.sum().detach(),
        "clipped_mask": clipped.detach(),
        "teacher_delta_raw": delta_t.detach(),
        "teacher_delta_target": target.detach(),
        "student_delta": delta_s.detach(),
        "teacher_margin_a": margin_ta.detach(),
        "teacher_margin_b": margin_tb.detach(),
        "student_margin_a": margin_sa.detach(),
        "student_margin_b": margin_sb.detach(),
    }


def paraphrase_invariance_loss(logits_a: Tensor, logits_b: Tensor,
                               hidden_a: Tensor | None = None, hidden_b: Tensor | None = None,
                               temperature: float = 1., hidden_weight: float = .1,
                               reduction: str = "mean") -> dict[str, Tensor]:
    """Symmetric JS + optional hidden cosine for equivalent support rewrites.

    Both inputs are student branches and both retain gradients. Jensen-Shannon
    divergence uses full-vocabulary float32 log probabilities and logaddexp for
    the mixture. The behavioral term is multiplied by temperature squared.
    Caller must ensure fact/value/question/prefix are unchanged across branches.
    """
    _matrices("Logits", logits_a, logits_b)
    _positive("temperature", temperature)
    _positive("hidden_weight", hidden_weight, allow_zero=True)
    if (hidden_a is None) != (hidden_b is None):
        raise ValueError("Supply both hidden branches or neither")
    log_a = F.log_softmax(logits_a.float() / temperature, dim=-1)
    log_b = F.log_softmax(logits_b.float() / temperature, dim=-1)
    log_mix = torch.logaddexp(log_a, log_b) - math.log(2.)
    js = ((log_a.exp() * (log_a - log_mix)).sum(-1)
          + (log_b.exp() * (log_b - log_mix)).sum(-1)) * (.5 * temperature ** 2)
    if hidden_a is None:
        hidden = js.new_zeros(js.shape)
    else:
        _matrices("Hidden states", hidden_a, hidden_b)
        if hidden_a.shape[0] != logits_a.shape[0] or hidden_a.device != logits_a.device:
            raise ValueError("Hidden states and logits must have matching batches and devices")
        hidden = 1 - F.cosine_similarity(hidden_a.float(), hidden_b.float(), dim=-1)
    per_example = js + hidden_weight * hidden
    return {"loss": _reduce(per_example, reduction),
            "js_loss": _reduce(js, reduction), "hidden_loss": _reduce(hidden, reduction),
            "per_example": per_example, "js_per_example": js, "hidden_per_example": hidden}
