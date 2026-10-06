"""Counterfactual supervision must distinguish content response from collapse."""
import math

import pytest
import torch
from torch.nn import functional as F

from vera_mem.counterfactual_losses import (
    hidden_delta_loss, pair_behavior_loss, paraphrase_invariance_loss,
)


def test_hidden_zero_student_effect_costs_one_and_has_both_branch_gradients():
    sa = torch.tensor([[0., 0., 1.]], requires_grad=True)
    sb = sa.detach().clone().requires_grad_(True)
    ta = torch.tensor([[1., 0., 0.]], requires_grad=True)
    tb = torch.tensor([[0., 1., 0.]], requires_grad=True)
    result = hidden_delta_loss(sa, sb, ta, tb)
    torch.testing.assert_close(result["loss"], torch.tensor(1.))
    assert result["valid_mask"].tolist() == [True]
    assert result["student_delta_norm"].tolist() == [0.]
    assert result["delta_cosine"].tolist() == [0.]
    result["loss"].backward()
    assert sa.grad.abs().sum() > 0 and sb.grad.abs().sum() > 0
    torch.testing.assert_close(sa.grad, -sb.grad)
    assert ta.grad is None and tb.grad is None


def test_hidden_exact_scaled_match_and_wrong_direction_are_distinguished():
    ta = torch.tensor([[1., 0., 0.]])
    tb = torch.tensor([[0., 1., 0.]])
    exact = hidden_delta_loss(ta * 9., tb * .2, ta, tb)
    reverse = hidden_delta_loss(tb, ta, ta, tb)
    torch.testing.assert_close(exact["loss"], torch.tensor(0.))
    torch.testing.assert_close(exact["delta_cosine"], torch.tensor([1.]))
    torch.testing.assert_close(reverse["loss"], torch.tensor(4.))
    torch.testing.assert_close(reverse["delta_cosine"], torch.tensor([-1.]))


def test_hidden_micro_teacher_delta_is_excluded_with_differentiable_zero():
    sa = torch.tensor([[1., 0., 0.], [0., 0., 1.]], requires_grad=True)
    sb = torch.tensor([[0., 1., 0.], [1., 0., 0.]], requires_grad=True)
    ta = torch.tensor([[1., 0., 0.], [1., 0., 0.]], requires_grad=True)
    tb = torch.tensor([[1., 0., 0.], [1., 1e-5, 0.]], requires_grad=True)
    result = hidden_delta_loss(sa, sb, ta, tb, min_teacher_norm=1e-3)
    assert result["valid_mask"].tolist() == [False, False]
    assert result["valid_count"].item() == 0
    assert result["loss"].item() == 0 and result["loss"].requires_grad
    result["loss"].backward()
    assert sa.grad.eq(0).all() and sb.grad.eq(0).all()
    assert ta.grad is None and tb.grad is None


def test_hidden_mean_counts_only_valid_pairs_and_none_retains_batch_order():
    sa = torch.tensor([[0., 0., 1.], [0., 0., 1.]])
    sb = sa.clone()
    ta = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    tb = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
    result = hidden_delta_loss(sa, sb, ta, tb)
    torch.testing.assert_close(result["loss"], torch.tensor(1.))
    torch.testing.assert_close(result["per_example"], torch.tensor([1., 0.]))
    none = hidden_delta_loss(sa, sb, ta, tb, reduction="none")
    torch.testing.assert_close(none["loss"], result["per_example"])
    assert result["valid_count"].item() == 1


def behavior_inputs():
    sa = torch.tensor([[2., 0., .4, -.2]], requires_grad=True)
    sb = torch.tensor([[.3, 1.5, 0., -.1]], requires_grad=True)
    ta = torch.tensor([[12., -2., 0., 0.]], requires_grad=True)
    tb = torch.tensor([[-3., 10., 0., 0.]], requires_grad=True)
    return sa, sb, ta, tb


def test_behavior_matches_clipped_teacher_delta_and_full_vocab_gold_ce():
    sa, sb, ta, tb = behavior_inputs()
    result = pair_behavior_loss(sa, sb, ta, tb, [0], [1])
    # logp(A)-logp(B) cancels the shared full-vocabulary log normalizer.
    def logprob_margin(logits):
        probs = logits.log_softmax(-1)
        return probs[0, 0] - probs[0, 1]
    delta_s = logprob_margin(sa) - logprob_margin(sb)
    delta_t = (logprob_margin(ta) - logprob_margin(tb)).detach().clamp(-10., 10.)
    expected_delta = F.smooth_l1_loss(delta_s, delta_t)
    expected_gold = (F.cross_entropy(sa, torch.tensor([0]))
                     + F.cross_entropy(sb, torch.tensor([1]))) / 2
    torch.testing.assert_close(result["delta_loss"], expected_delta)
    torch.testing.assert_close(result["gold_loss"], expected_gold)
    torch.testing.assert_close(result["loss"], expected_delta + expected_gold)
    assert result["clipped_mask"].tolist() == [True]
    assert result["teacher_delta_raw"].tolist() == [27.]
    assert result["teacher_delta_target"].tolist() == [10.]
    result["loss"].backward()
    assert sa.grad.abs().sum() > 0 and sb.grad.abs().sum() > 0
    assert ta.grad is None and tb.grad is None


def test_behavior_student_is_not_clipped_so_overshoot_still_has_gradient():
    sa = torch.tensor([[100., 0., 0.]], requires_grad=True)
    sb = torch.tensor([[0., 100., 0.]], requires_grad=True)
    ta, tb = torch.tensor([[30., 0., 0.]]), torch.tensor([[0., 30., 0.]])
    result = pair_behavior_loss(sa, sb, ta, tb, [0], [1], gold_weight=0.)
    assert result["student_delta"].item() == 200.
    torch.testing.assert_close(result["delta_loss"], torch.tensor(189.5))
    result["loss"].backward()
    assert sa.grad.abs().sum() > 0 and sb.grad.abs().sum() > 0


def test_gold_anchor_exposes_third_token_shortcut_despite_perfect_pair_contrast():
    # A/B contrasts are correct and match teacher, but both students overwhelmingly
    # predict unrelated token C. Candidate-only contrast would call this solved.
    sa = torch.tensor([[4., 1., 50.]], requires_grad=True)
    sb = torch.tensor([[1., 4., 50.]], requires_grad=True)
    ta, tb = torch.tensor([[4., 1., -10.]]), torch.tensor([[1., 4., -10.]])
    result = pair_behavior_loss(sa, sb, ta, tb, [0], [1], target_clip=None)
    assert result["delta_loss"].item() == 0.
    assert result["gold_loss"].item() > 40.
    result["loss"].backward()
    assert sa.grad[0, 2] > 0 and sb.grad[0, 2] > 0
    assert sa.grad[0, 0] < 0 and sb.grad[0, 1] < 0


def test_wrong_teacher_direction_masks_delta_but_gold_is_still_trained():
    sa, sb, ta, tb = behavior_inputs()
    result = pair_behavior_loss(sa, sb, tb, ta, [0], [1])
    assert result["valid_count"].item() == 0
    assert result["delta_loss"].item() == 0.
    torch.testing.assert_close(result["loss"], result["gold_loss"])
    result["loss"].backward()
    assert sa.grad.abs().sum() > 0 and sb.grad.abs().sum() > 0
    assert ta.grad is None and tb.grad is None


def test_behavior_temperature_scales_effect_but_not_gold_anchor():
    sa, sb, ta, tb = behavior_inputs()
    cold = pair_behavior_loss(sa, sb, ta, tb, [0], [1], target_clip=None)
    warm = pair_behavior_loss(sa, sb, ta, tb, [0], [1], temperature=2., target_clip=None)
    torch.testing.assert_close(warm["student_delta"], cold["student_delta"] / 2)
    torch.testing.assert_close(warm["teacher_delta_raw"], cold["teacher_delta_raw"] / 2)
    torch.testing.assert_close(warm["gold_loss"], cold["gold_loss"])
    assert not warm["clipped_mask"].any()


def test_behavior_swapping_both_branches_and_candidates_preserves_objective():
    sa, sb, ta, tb = behavior_inputs()
    direct = pair_behavior_loss(sa, sb, ta, tb, [0], [1])
    swapped = pair_behavior_loss(sb, sa, tb, ta, [1], [0])
    torch.testing.assert_close(direct["loss"], swapped["loss"])
    torch.testing.assert_close(direct["teacher_delta_target"], swapped["teacher_delta_target"])


def test_behavior_full_precision_handles_large_bfloat16_logits():
    sa = torch.tensor([[1000., -1000., 0.]], dtype=torch.bfloat16, requires_grad=True)
    sb = torch.tensor([[-1000., 1000., 0.]], dtype=torch.bfloat16, requires_grad=True)
    result = pair_behavior_loss(sa, sb, sa.detach(), sb.detach(), [0], [1])
    assert result["loss"].dtype == torch.float32 and torch.isfinite(result["loss"])
    result["loss"].backward()
    assert torch.isfinite(sa.grad).all() and torch.isfinite(sb.grad).all()


def test_paraphrase_js_is_symmetric_and_trains_both_hidden_and_behavior_branches():
    la = torch.tensor([[1., -1., 0.], [.3, 1., -1.]], requires_grad=True)
    lb = torch.tensor([[-.5, .5, 0.], [.1, -.2, .4]], requires_grad=True)
    ha = torch.tensor([[1., 0., 0.], [0., 1., 0.]], requires_grad=True)
    hb = torch.tensor([[0., 1., 0.], [0., 0., 1.]], requires_grad=True)
    result = paraphrase_invariance_loss(la, lb, ha, hb, reduction="none")
    swapped = paraphrase_invariance_loss(lb, la, hb, ha, reduction="none")
    pa, pb = la.softmax(-1), lb.softmax(-1)
    mixture = (pa + pb) / 2
    expected = ((pa * (pa.log() - mixture.log())).sum(-1)
                + (pb * (pb.log() - mixture.log())).sum(-1)) / 2
    torch.testing.assert_close(result["js_per_example"], expected)
    torch.testing.assert_close(result["loss"], swapped["loss"])
    torch.testing.assert_close(result["loss"], expected + .1)
    result["loss"].mean().backward()
    assert all(value.grad.abs().sum() > 0 for value in (la, lb, ha, hb))


def test_paraphrase_identity_and_extreme_disagreement_stay_finite():
    la = torch.tensor([[1000., -1000., 0.]], dtype=torch.bfloat16, requires_grad=True)
    lb = torch.tensor([[-1000., 1000., 0.]], dtype=torch.bfloat16, requires_grad=True)
    identical = paraphrase_invariance_loss(la, la)
    torch.testing.assert_close(identical["loss"], torch.tensor(0.))
    result = paraphrase_invariance_loss(la, lb)
    torch.testing.assert_close(result["js_loss"], torch.tensor(math.log(2.)))
    assert result["loss"].dtype == torch.float32
    result["loss"].backward()
    assert torch.isfinite(la.grad).all() and torch.isfinite(lb.grad).all()


@pytest.mark.parametrize("ids_a,ids_b", [([0], [0]), ([-1], [1]), ([0.2], [1]), ([True], [1]), ([0], [4])])
def test_behavior_rejects_ambiguous_or_invalid_first_token_pairs(ids_a, ids_b):
    with pytest.raises(ValueError):
        pair_behavior_loss(*behavior_inputs(), ids_a, ids_b)


@pytest.mark.parametrize("kwargs", [
    {"temperature": 0}, {"temperature": float("nan")}, {"target_clip": 0},
    {"huber_beta": 0}, {"gold_weight": -1}, {"min_teacher_effect": 0},
    {"reduction": "invalid"},
])
def test_behavior_rejects_invalid_loss_parameters(kwargs):
    with pytest.raises(ValueError):
        pair_behavior_loss(*behavior_inputs(), [0], [1], **kwargs)


def test_hidden_and_paraphrase_shape_and_threshold_validation():
    sample = torch.ones(2, 3)
    with pytest.raises(ValueError, match="min_teacher_norm"):
        hidden_delta_loss(sample, sample, sample, sample, min_teacher_norm=0)
    with pytest.raises(ValueError, match="matching shapes"):
        hidden_delta_loss(sample, sample[:1], sample, sample)
    with pytest.raises(ValueError, match="both hidden"):
        paraphrase_invariance_loss(sample, sample, hidden_a=sample)
    with pytest.raises(ValueError, match="matching batches"):
        paraphrase_invariance_loss(sample, sample, torch.ones(1, 4), torch.ones(1, 4))
