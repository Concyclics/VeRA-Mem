"""The corrected objective retains teacher amplitude without a scale shortcut."""
import json

import pytest
import torch
from torch.nn import functional as F

from vera_mem.counterfactual_losses import pair_behavior_loss
from vera_mem.interface_losses import fit_margin_scale, normalized_behavior_loss


def inputs(effects=(40., 80.), student_effects=None):
    effects = torch.tensor(effects)
    zeros = torch.zeros_like(effects)
    ta = torch.stack((effects / 2, zeros, zeros), -1).requires_grad_()
    tb = torch.stack((zeros, effects / 2, zeros), -1).requires_grad_()
    student = zeros if student_effects is None else torch.tensor(student_effects)
    sa = torch.stack((student / 2, zeros, zeros), -1).requires_grad_()
    sb = torch.stack((zeros, student / 2, zeros), -1).requires_grad_()
    return sa, sb, ta, tb, torch.zeros(len(effects), dtype=torch.long), torch.ones(
        len(effects), dtype=torch.long)


def test_calibration_uses_only_valid_train_pairs_and_logs_old_saturation():
    teacher = torch.tensor([20., 40., 60., 1000., -200.], requires_grad=True)
    report = fit_margin_scale(teacher, torch.tensor([True, True, True, False, False]))
    assert report["scale"] == 40.
    assert report["count"] == 5 and report["valid_count"] == 3
    assert report["legacy_clipped_fraction_all"] == 1.
    assert report["legacy_clipped_fraction_valid"] == 1.
    assert report["teacher_target_normalized_distribution_valid"]["std"] > 0.
    assert report["legacy_target_distribution_valid"]["std"] == 0.
    assert report["teacher_delta_distribution_all"]["max"] == 1000.
    json.dumps(report, allow_nan=False)
    assert teacher.grad is None


def test_calibration_even_median_and_floor_are_explicit():
    report = fit_margin_scale(torch.tensor([20., 40., 60., 1000.]), torch.ones(4, dtype=torch.bool))
    assert report["scale"] == 50.
    floored = fit_margin_scale(torch.tensor([1e-5]), torch.tensor([True]), min_scale=.1)
    assert floored["floor_applied"] and floored["scale"] == .1


@pytest.mark.parametrize("delta,valid,kwargs", [
    (torch.tensor([1.]), torch.tensor([True]), {"split": "confirmation"}),
    (torch.tensor([1.]), torch.tensor([False]), {}),
    (torch.tensor([float("nan")]), torch.tensor([False]), {}),
    (torch.tensor([-1.]), torch.tensor([True]), {}),
    (torch.tensor([0.]), torch.tensor([True]), {}),
    (torch.tensor([]), torch.tensor([], dtype=torch.bool), {}),
    (torch.tensor([[1.]]), torch.tensor([[True]]), {}),
    (torch.tensor([1]), torch.tensor([True]), {}),
    (torch.tensor([1.]), torch.tensor([1]), {}),
    (torch.tensor([1.]), torch.tensor([True, True]), {}),
])
def test_calibration_rejects_unusable_or_nontraining_inputs(delta, valid, kwargs):
    with pytest.raises(ValueError):
        fit_margin_scale(delta, valid, **kwargs)


def test_unclipped_targets_preserve_amplitude_and_match_explicit_formula():
    args = inputs(student_effects=(8., 12.))
    result = normalized_behavior_loss(*args, scale=60., gold_weight=.3)
    old = pair_behavior_loss(*args, gold_weight=.3)
    expected = 60. * F.smooth_l1_loss(torch.tensor([8., 12.]) / 60.,
                                    torch.tensor([40., 80.]) / 60.)
    torch.testing.assert_close(result["delta_loss"], expected)
    torch.testing.assert_close(result["gold_loss"], old["gold_loss"])
    torch.testing.assert_close(result["loss"], expected + .3 * old["gold_loss"])
    assert result["teacher_delta_target"].tolist() == [40., 80.]
    assert old["teacher_delta_target"].tolist() == [10., 10.]
    assert result["legacy_clipped_mask"].all() and not result["clipped_mask"].any()
    assert not result["teacher_target_normalized"].requires_grad
    assert not result["student_delta_normalized"].requires_grad


def test_correct_optimum_is_teacher_effect_and_legacy_constant_is_not_optimum():
    matched = normalized_behavior_loss(*inputs(student_effects=(40., 80.)), scale=60., gold_weight=0.)
    clipped = normalized_behavior_loss(*inputs(student_effects=(10., 10.)), scale=60., gold_weight=0.)
    zero = normalized_behavior_loss(*inputs(), scale=60., gold_weight=0.)
    assert matched["loss"].item() == 0.
    assert clipped["loss"].item() > 0. and zero["loss"].item() > clipped["loss"].item()


def test_student_branches_receive_gradients_but_teacher_does_not():
    args = inputs()
    result = normalized_behavior_loss(*args, scale=60., gold_weight=0.)
    result["loss"].backward()
    sa, sb, ta, tb, *_ = args
    assert sa.grad.abs().sum() > 0. and sb.grad.abs().sum() > 0.
    torch.testing.assert_close(sa.grad, -sb.grad)
    assert ta.grad is None and tb.grad is None


def test_scale_factor_preserves_saturated_raw_logit_gradient_bound():
    gradients = []
    for scale in (2., 60.):
        args = inputs(effects=(1000.,))
        result = normalized_behavior_loss(*args, scale=scale, gold_weight=0.)
        result["loss"].backward()
        gradients.append(args[0].grad)
    torch.testing.assert_close(gradients[0], gradients[1])
    assert gradients[0].abs().max().item() == 1.


@pytest.mark.parametrize("temperature", [1., 2., 10.])
def test_teacher_gate_and_raw_diagnostics_are_unchanged(temperature):
    sa, sb, ta, tb, ia, ib = inputs(effects=(40., -2., .001))
    args = sa, sb, ta, tb, ia, ib
    old = pair_behavior_loss(*args, temperature=temperature, target_clip=None)
    new = normalized_behavior_loss(*args, scale=60., temperature=temperature)
    for name in ("valid_mask", "valid_count", "teacher_delta_raw", "teacher_delta_target",
                 "student_delta", "teacher_margin_a", "teacher_margin_b", "gold_loss"):
        torch.testing.assert_close(new[name], old[name])


def test_invalid_batch_has_differentiable_zero_but_gold_anchor_still_available():
    args = inputs(effects=(-40., -80.))
    result = normalized_behavior_loss(*args, scale=60., gold_weight=0.)
    assert result["valid_count"].item() == 0 and result["loss"].item() == 0.
    assert result["gold_loss"].item() > 0.
    result["loss"].backward()
    assert args[0].grad.eq(0.).all() and args[1].grad.eq(0.).all()


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_scale_one_matches_original_unclipped_loss(reduction):
    args = inputs(effects=(40., -2., 80.), student_effects=(2., 1., 3.))
    new = normalized_behavior_loss(*args, scale=1., reduction=reduction, gold_weight=.5)
    old = pair_behavior_loss(*args, target_clip=None, reduction=reduction, gold_weight=.5)
    for name in ("loss", "delta_loss", "gold_loss", "per_example", "delta_per_example"):
        torch.testing.assert_close(new[name], old[name])


def test_fixed_scale_does_not_depend_on_batch_composition():
    args = inputs(student_effects=(8., 12.))
    full = normalized_behavior_loss(*args, scale=60., reduction="none")
    pieces = [normalized_behavior_loss(*(arg[i:i + 1] for arg in args),
                                      scale=60., reduction="none")["loss"] for i in range(2)]
    torch.testing.assert_close(full["loss"], torch.cat(pieces))


@pytest.mark.parametrize("scale", [0., -1., float("nan"), float("inf"), True,
                                  torch.tensor(60.), [60.]])
def test_scale_must_be_a_fixed_positive_scalar(scale):
    with pytest.raises(ValueError):
        normalized_behavior_loss(*inputs(), scale=scale)
