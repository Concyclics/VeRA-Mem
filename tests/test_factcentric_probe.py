"""Validate the frozen-feature probe's fact balance and representation path."""
import importlib.util
from pathlib import Path
import random

import pytest
import torch


_SPEC = importlib.util.spec_from_file_location(
    "probe_factcentric_training", Path(__file__).parents[1] / "scripts" / "probe_factcentric_training.py")
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_balanced_draw_keeps_same_answer_negative_facts_distinct_and_reproducible():
    labels = [str(i % 16) for i in range(256)]
    a = probe.balanced_indices(labels, 128, random.Random(98))
    b = probe.balanced_indices(labels, 128, random.Random(98))
    assert a == b
    assert len(a) == len(set(a)) == 128
    for label in set(labels):
        assert sum(labels[i] == label for i in a) == 8
    with pytest.raises(ValueError, match="balanced"):
        probe.balanced_indices(labels, 127, random.Random(98))


def test_cached_transform_matches_actual_stable_module_for_queries_keys_values():
    generator = torch.Generator().manual_seed(138)
    q = torch.randn(12, 5, 8, generator=generator)
    s = torch.randn(12, 3, 8, generator=generator)
    module = probe.StableVectorVeRA(8, 7, rank=4, key_dim=3)
    module.fit_statistics(q.reshape(-1, 8), s.reshape(-1, 8))
    torch.testing.assert_close(probe.training_center(q), module.query_center)
    torch.testing.assert_close(probe.training_center(s), module.support_center)
    transformed_q = probe.transform(q, module.query_center, "cpu")
    transformed_s = probe.transform(s, module.support_center, "cpu")
    encoded_q, encoded_k, encoded_v = probe.encode(module, transformed_q, transformed_s)
    torch.testing.assert_close(encoded_q, module.encode_query(q))
    torch.testing.assert_close(encoded_k, module.encode_key(s))
    torch.testing.assert_close(encoded_v, module.encode_value(s))


def test_retrieval_metrics_count_entity_identity_not_equal_values_or_diagonal_loss():
    query = torch.eye(8)
    identity = probe.retrieval_metrics(query, query)
    assert identity["correct_at_1"] == identity["correct_at_4"] == 8
    swapped = probe.retrieval_metrics(query, query.roll(1, 0))
    assert swapped["correct_at_1"] == 0
    assert swapped["contrastive_nll"] > identity["contrastive_nll"]


def test_pair_consistency_requires_distinct_view_agreement_and_keeps_gradients():
    q = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    k = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    v = torch.tensor([[.3, .7], [.5, -.1]], requires_grad=True)
    identical = probe.paired_consistency(q, q, k, k, v, v)
    assert float(identical.detach()) == 0.
    perturbed = probe.paired_consistency(q, q.roll(1, 0), k, k.roll(1, 0), v, v.roll(1, 0))
    perturbed.backward()
    assert float(perturbed.detach()) > 0.
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.norm() > 0
