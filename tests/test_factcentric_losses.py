"""Fact identity, positive-style coverage, and frozen-anchor loss contracts."""

import math

import pytest
import torch
from torch.nn import functional as F

from vera_mem.factcentric_losses import (
    all_view_retrieval_loss, canonical_anchor_loss, factcentric_losses, style_block_retrieval_loss,
)


def test_all_styles_are_positive_and_different_facts_are_negative():
    query = torch.eye(3)[:, None, :].expand(-1, 2, -1)
    key = torch.eye(3)[:, None, :].expand(-1, 3, -1)
    result = all_view_retrieval_loss(query, key, temperature=0.01)
    # No spurious pressure to distinguish the same entity's duplicate styles.
    torch.testing.assert_close(result["query_to_key"], torch.tensor(math.log(3)))
    torch.testing.assert_close(result["key_to_query"], torch.tensor(math.log(2)))
    assert result["retrieval"] < 1
    swapped = all_view_retrieval_loss(query, key.roll(1, dims=0), temperature=0.01)
    assert swapped["retrieval"] > 99


def test_repeated_fact_rows_are_not_false_negatives_and_ids_are_not_answer_classes():
    query = torch.tensor([[[1., 0.]], [[1., 0.]], [[0., 1.]]])
    ids = torch.tensor([91, 91, 12])
    result = all_view_retrieval_loss(query, query, entity_ids=ids, temperature=0.3)
    logits = query[:, 0] @ query[:, 0].T / 0.3
    target = (ids[:, None] == ids[None, :]).float()
    expected = -(F.log_softmax(logits, dim=-1) * target / target.sum(1, keepdim=True)).sum(1).mean()
    torch.testing.assert_close(result["retrieval"], expected)
    # Without a repeated identity the two equal vectors are distinct facts,
    # and gradients encourage distinguishing them rather than joining them.
    distinct = torch.eye(3)[:, None, :]
    repeated = all_view_retrieval_loss(distinct, distinct, entity_ids=ids)["retrieval"]
    separate = all_view_retrieval_loss(distinct, distinct)["retrieval"]
    assert repeated > separate + 3


def test_fact_renaming_row_order_and_view_order_do_not_change_loss():
    torch.manual_seed(16)
    query, key = torch.randn(4, 3, 7), torch.randn(4, 2, 7)
    ids, renamed = torch.tensor([4, 7, 4, 8]), torch.tensor([104, -6, 104, 0])
    original = all_view_retrieval_loss(query, key, entity_ids=ids)
    permutation = torch.tensor([3, 1, 0, 2])
    altered = all_view_retrieval_loss(
        query[permutation].flip(1), key[permutation].flip(1), entity_ids=renamed[permutation],
    )
    for name in original:
        torch.testing.assert_close(original[name], altered[name])


def test_uniform_positive_target_cannot_ignore_one_hard_positive_view():
    query = torch.eye(2)[:, None, :]
    easy = torch.eye(2)[:, None, :].expand(-1, 2, -1).clone()
    hard = easy.clone()
    hard[:, 1] *= -1
    loss_easy = all_view_retrieval_loss(query, easy, temperature=.05)["query_to_key"]
    loss_hard = all_view_retrieval_loss(query, hard, temperature=.05)["query_to_key"]
    assert loss_hard > loss_easy + 15
    # A logsumexp-positive objective nearly ignores the deliberately bad view.
    logits = query[:, 0] @ hard.reshape(4, 2).T / .05
    positives = torch.eye(2).bool().repeat_interleave(2, dim=1)
    permissive = (logits.logsumexp(1) - logits.masked_fill(~positives, -torch.inf).logsumexp(1)).mean()
    assert permissive < .01


def test_symmetric_loss_sends_gradients_to_query_and_key_encoders():
    torch.manual_seed(24)
    query = torch.randn(4, 2, 5, requires_grad=True)
    key = torch.randn(4, 3, 5, requires_grad=True)
    result = all_view_retrieval_loss(query, key)
    torch.testing.assert_close(result["retrieval"], (result["query_to_key"] + result["key_to_query"]) / 2)
    result["retrieval"].backward()
    for value in (query, key):
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


def test_canonical_anchor_has_no_gradient_and_every_student_style_is_supervised():
    torch.manual_seed(19)
    student = torch.randn(4, 3, 5, requires_grad=True)
    anchor = torch.randn(4, 5, requires_grad=True)
    cosine = canonical_anchor_loss(student, anchor)
    mse = canonical_anchor_loss(student, anchor, metric="normalized_mse")
    torch.testing.assert_close(mse, 2 * cosine)
    cosine.backward()
    assert anchor.grad is None
    assert torch.all(student.grad.abs().sum(-1) > 0)
    same = anchor.detach()[:, None, :].expand(-1, 3, -1)
    assert canonical_anchor_loss(same, anchor).abs() < 1e-6


def test_combined_components_weights_and_value_anchor_are_explicit():
    torch.manual_seed(14)
    query, key, value = torch.randn(3, 2, 4), torch.randn(3, 3, 4), torch.randn(3, 3, 6)
    anchors = [torch.randn(3, dimension, requires_grad=True) for dimension in (4, 4, 6)]
    value.requires_grad_()
    result = factcentric_losses(
        query, key, value, query_anchor=anchors[0], key_anchor=anchors[1], value_anchor=anchors[2],
        query_anchor_weight=.2, key_anchor_weight=.3, value_anchor_weight=.4,
    )
    expected = result["retrieval"] + .2 * result["query_anchor"] + .3 * result["key_anchor"] + .4 * result["value_anchor"]
    torch.testing.assert_close(result["total"], expected)
    result["total"].backward()
    assert value.grad.abs().sum() > 0
    assert all(anchor.grad is None for anchor in anchors)
    no_anchors = factcentric_losses(query, key)
    assert all(no_anchors[name].item() == 0 for name in ("query_anchor", "key_anchor", "value_anchor"))
    torch.testing.assert_close(no_anchors["total"], no_anchors["retrieval"])


def test_input_validation_and_half_precision_remain_finite():
    query, key = torch.randn(3, 2, 4), torch.randn(3, 3, 4)
    for bad in (0, -1, True, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="temperature"):
            all_view_retrieval_loss(query, key, temperature=bad)
    for bad_ids in (torch.ones(3), torch.ones(4, dtype=torch.long), torch.ones(3, dtype=torch.bool)):
        with pytest.raises(ValueError, match="entity_ids"):
            all_view_retrieval_loss(query, key, entity_ids=bad_ids)
    with pytest.raises(ValueError, match="share fact count"):
        all_view_retrieval_loss(query, key[:2])
    with pytest.raises(ValueError, match="nonempty"):
        all_view_retrieval_loss(query[:, :0], key)
    with pytest.raises(ValueError, match="anchor must have shape"):
        canonical_anchor_loss(query, torch.randn(3, 2, 4))
    with pytest.raises(ValueError, match="metric"):
        canonical_anchor_loss(query, torch.randn(3, 4), metric="bogus")
    with pytest.raises(ValueError, match="value is required"):
        factcentric_losses(query, key, value_anchor=torch.randn(3, 6))
    with pytest.raises(ValueError, match="weight"):
        factcentric_losses(query, key, value_anchor_weight=-1)
    half_loss = all_view_retrieval_loss(query.half(), key.half(), temperature=.001)
    assert half_loss["retrieval"].dtype == torch.float32
    assert torch.isfinite(half_loss["retrieval"])


def test_style_block_has_style_constant_negative_banks_and_no_style_only_solution():
    # Both views encode only their style; every fact in that view is identical.
    query = torch.eye(2)[None, :, :].expand(5, -1, -1)
    key = query.clone()
    aligned = style_block_retrieval_loss(query, key, temperature=.03)
    incompatible = style_block_retrieval_loss(query, -key.flip(1), temperature=.03)
    for name in aligned:
        torch.testing.assert_close(aligned[name], torch.tensor(math.log(5)))
        torch.testing.assert_close(incompatible[name], aligned[name])
    # Encoding fact identity can solve every style pair, regardless of style.
    facts = torch.eye(5)[:, None, :].expand(-1, 2, -1)
    assert style_block_retrieval_loss(facts, facts, temperature=.03)["retrieval"] < 1e-6


def test_style_block_matches_explicit_all_style_pair_cross_entropy_and_gradients():
    torch.manual_seed(171)
    query = torch.randn(4, 2, 6, requires_grad=True)
    key = torch.randn(4, 3, 6, requires_grad=True)
    result = style_block_retrieval_loss(query, key, temperature=.2)
    q_losses, k_losses = [], []
    for q_style in range(2):
        for k_style in range(3):
            logits = F.normalize(query[:, q_style], dim=-1) @ F.normalize(key[:, k_style], dim=-1).T / .2
            q_losses.append(F.cross_entropy(logits, torch.arange(4)))
            k_losses.append(F.cross_entropy(logits.T, torch.arange(4)))
    torch.testing.assert_close(result["query_to_key"], torch.stack(q_losses).mean())
    torch.testing.assert_close(result["key_to_query"], torch.stack(k_losses).mean())
    result["retrieval"].backward()
    for value in (query, key):
        assert torch.isfinite(value.grad).all()
        assert torch.all(value.grad.abs().sum((0, 2)) > 0)  # Every style participates.


def test_style_block_repeated_id_and_renaming_contract():
    torch.manual_seed(13)
    query, key = torch.randn(3, 2, 4), torch.randn(3, 4, 4)
    ids = torch.tensor([1, 1, 8])
    actual = style_block_retrieval_loss(query, key, entity_ids=ids)
    explicit = []
    for q_style in range(2):
        for k_style in range(4):
            explicit.append(all_view_retrieval_loss(
                query[:, q_style:q_style + 1], key[:, k_style:k_style + 1], entity_ids=ids,
            )["retrieval"])
    torch.testing.assert_close(actual["retrieval"], torch.stack(explicit).mean())
    altered = style_block_retrieval_loss(query.flip(0), key.flip(0), entity_ids=torch.tensor([31, 2, 2]))
    for name in actual:
        torch.testing.assert_close(actual[name], altered[name])
