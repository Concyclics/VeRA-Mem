"""Tests of memory-conditioned computation, addressing, and learning paths."""

import io

import pytest
import torch

from vera_mem.vector_vera import VectorVeRA


def _module(**kwargs):
    return VectorVeRA(in_features=8, out_features=6, rank=4, key_dim=3, **kwargs)


def test_random_projections_frozen_and_parameter_budget():
    module = _module()
    assert set(dict(module.named_buffers())) == {"A", "B"}
    assert not module.A.requires_grad and not module.B.requires_grad
    assert set(dict(module.named_parameters())) == {"b", "Wq.weight", "Wk.weight", "Wv.weight"}
    assert module.trainable_numel() == 6 + 8 * (3 + 3 + 4)
    assert torch.equal(module.Wq.weight, module.Wk.weight)
    assert module.Wq.weight.data_ptr() != module.Wk.weight.data_ptr()


def test_empty_bank_is_exact_zero_for_each_token():
    module = _module()
    with torch.no_grad():
        module.b.fill_(1)
    x = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    residual, info = module(x, torch.empty(0, 3), torch.empty(0, 4), return_info=True)
    assert residual.shape == (2, 5, 6)
    assert residual.dtype == x.dtype
    assert torch.count_nonzero(residual) == 0
    assert info["indices"].shape == (2, 5, 0)
    assert info["weights"].numel() == 0
    assert info["indices"].dtype == torch.long


def test_initialized_nonempty_bank_is_exact_no_op():
    module = _module()
    support = torch.randn(7, 8)
    residual = module(torch.randn(2, 3, 8), module.encode_key(support), module.encode_value(support))
    assert torch.count_nonzero(residual) == 0


def test_retrieval_is_per_token_and_values_change_computation():
    module = VectorVeRA(3, 3, rank=3, key_dim=3, top_k=1)
    with torch.no_grad():
        module.Wq.weight.copy_(torch.eye(3))
        module.Wk.weight.copy_(torch.eye(3))
        module.A.copy_(torch.eye(3))
        module.B.copy_(torch.eye(3))
        module.b.fill_(1)
    x = torch.tensor([[[2.0, 0, 0], [0, 3.0, 0], [0, 0, 4.0]]])
    keys = torch.eye(3)
    values = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.7, 0.8, 0.9]])
    residual, info = module(x, keys, values, return_info=True)
    assert info["indices"].tolist() == [[[0], [1], [2]]]
    torch.testing.assert_close(info["scores"], torch.ones(1, 3, 1))
    torch.testing.assert_close(info["weights"], torch.ones(1, 3, 1))
    torch.testing.assert_close(residual, x * values.unsqueeze(0))
    torch.testing.assert_close(module(x, keys, values * 2), residual * 2)
    assert not torch.equal(module(x, keys, values.flip(0)), residual)


def test_offline_selected_scores_values_and_encoders_receive_gradients():
    torch.manual_seed(20)
    module = _module(top_k=3, temperature=0.7)
    with torch.no_grad():
        module.b.fill_(0.3)
    x = torch.randn(2, 8, requires_grad=True)
    support = torch.randn(5, 8, requires_grad=True)
    keys = module.encode_key(support)
    values = module.encode_value(support)
    keys.retain_grad()
    values.retain_grad()
    residual, info = module(x, keys, values, return_info=True)
    (residual - torch.randn_like(residual)).square().mean().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name
    assert x.grad.abs().sum() > 0
    assert support.grad.abs().sum() > 0
    assert keys.grad.abs().sum() > 0
    assert values.grad.abs().sum() > 0
    torch.testing.assert_close(info["weights"].sum(dim=-1), torch.ones(2))


def test_zero_b_gets_first_step_gradient_and_address_auxiliary_is_independent():
    torch.manual_seed(3)
    module = _module(top_k=2)
    x, support = torch.randn(3, 8), torch.randn(4, 8)
    result = module(x, module.encode_key(support), module.encode_value(support))
    (result - torch.randn_like(result)).square().mean().backward()
    assert module.b.grad.abs().sum() > 0
    for projection in (module.Wq, module.Wk, module.Wv):
        assert projection.weight.grad.abs().sum() == 0
    module.zero_grad(set_to_none=True)
    loss = module.contrastive_loss(x, support[:3])
    assert torch.isfinite(loss)
    loss.backward()
    assert module.Wq.weight.grad.abs().sum() > 0
    assert module.Wk.weight.grad.abs().sum() > 0
    assert module.b.grad is None
    assert module.Wv.weight.grad is None


def test_encoding_normalization_and_scale_invariance():
    module = _module()
    x = torch.randn(5, 8)
    torch.testing.assert_close(module.encode_query(x).norm(dim=-1), torch.ones(5))
    torch.testing.assert_close(module.encode_key(x).norm(dim=-1), torch.ones(5))
    torch.testing.assert_close(module.encode_query(3 * x), module.encode_query(x))
    torch.testing.assert_close(module.encode_value(3 * x), module.encode_value(x))
    assert module.encode_value(x).abs().max() <= 1
    assert torch.count_nonzero(module.encode_query(torch.zeros_like(x))) == 0
    assert torch.isfinite(module.encode_value(torch.zeros_like(x))).all()


def test_external_mixed_value_path_matches_forward_and_broadcasts():
    module = _module(top_k=3)
    with torch.no_grad():
        module.b.fill_(0.5)
    support, x = torch.randn(4, 8), torch.randn(2, 3, 8, dtype=torch.bfloat16)
    keys, values = module.encode_key(support), module.encode_value(support)
    expected, info = module(x, keys, values, return_info=True)
    mixed = (values[info["indices"]] * info["weights"].unsqueeze(-1)).sum(dim=-2)
    torch.testing.assert_close(module.delta_from_value(x, mixed), expected, rtol=0, atol=0)
    one_value = torch.randn(4, requires_grad=True)
    result = module.delta_from_value(x, one_value)
    assert result.dtype == x.dtype and result.shape == (2, 3, 6)
    torch.testing.assert_close(result, module.delta_from_value(x, one_value.expand(2, 3, 4)))
    result.float().square().sum().backward()
    assert one_value.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        module.delta_from_value(x, torch.randn(5, 4))


def test_state_roundtrip_reproduces_predictions_and_seeds_do_not_pollute_rng():
    torch.manual_seed(13)
    before = torch.random.get_rng_state().clone()
    module = _module(seed=100, top_k=2)
    assert torch.equal(before, torch.random.get_rng_state())
    with torch.no_grad():
        module.b.fill_(1)
    support, x = torch.randn(4, 8), torch.randn(3, 8)
    expected = module(x, module.encode_key(support), module.encode_value(support))
    storage = io.BytesIO()
    torch.save({"config": module.configuration(), "state_dict": module.state_dict()}, storage)
    storage.seek(0)
    checkpoint = torch.load(storage, weights_only=True)
    restored = VectorVeRA(**checkpoint["config"])
    restored.load_state_dict(checkpoint["state_dict"])
    actual = restored(x, restored.encode_key(support), restored.encode_value(support))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_small_bank_and_invalid_shapes():
    module = _module(top_k=10)
    support = torch.randn(1, 8)
    _, info = module(torch.randn(8), module.encode_key(support), module.encode_value(support), True)
    assert info["indices"].shape == (1,)
    with pytest.raises(ValueError):
        module(torch.randn(8), torch.randn(2, 3), torch.randn(1, 4))
    with pytest.raises(ValueError):
        module.contrastive_loss(torch.randn(2, 8), torch.randn(3, 8))
    with pytest.raises(ValueError):
        _module(temperature=0)


def test_centering_disabled_preserves_legacy_state_and_value_formula():
    legacy = _module(seed=91)
    explicit = _module(seed=91, value_centering=False)
    assert "value_center" not in explicit.state_dict()
    assert set(legacy.state_dict()) == set(explicit.state_dict())
    # Strictly loading an old checkpoint requires no missing-key workaround.
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    x = torch.randn(5, 8)
    expected = torch.tanh(legacy.Wv(legacy._normalized_input(x)))
    torch.testing.assert_close(explicit.encode_value(x), expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="Enable value_centering"):
        explicit.fit_value_center(x)


def test_value_center_uses_only_training_mean_and_leaves_other_paths_unchanged():
    plain = _module(seed=12)
    centered = _module(seed=12, value_centering=True)
    training = torch.randn(20, 8) + torch.arange(8).float()
    query_before = centered.encode_query(training).detach().clone()
    key_before = centered.encode_key(training).detach().clone()
    original_parameters = {name: value.detach().clone() for name, value in centered.named_parameters()}
    centered.fit_value_center(training)
    normalized = centered._normalized_input(training)
    torch.testing.assert_close(centered.value_center, normalized.mean(dim=0))
    torch.testing.assert_close(
        (normalized - centered.value_center).mean(dim=0), torch.zeros(8), atol=2e-7, rtol=0
    )
    torch.testing.assert_close(centered.encode_query(training), query_before, rtol=0, atol=0)
    torch.testing.assert_close(centered.encode_key(training), key_before, rtol=0, atol=0)
    assert torch.equal(centered.A, plain.A) and torch.equal(centered.B, plain.B)
    for name, value in centered.named_parameters():
        assert torch.equal(value, original_parameters[name]), name
    assert not centered.value_center.requires_grad
    # Later held-out encoding must not update the fitted training statistic.
    center = centered.value_center.clone()
    centered.encode_value(torch.randn(11, 8) - 3)
    assert torch.equal(centered.value_center, center)


def test_centered_values_preserve_offline_gradients_and_checkpoint_state():
    torch.manual_seed(34)
    module = _module(value_centering=True, top_k=3, temperature=0.7)
    module.fit_value_center(torch.randn(15, 8) + 2)
    with torch.no_grad():
        module.b.fill_(0.3)
    x = torch.randn(3, 8)
    supports = torch.randn(6, 8, requires_grad=True)
    residual = module(x, module.encode_key(supports), module.encode_value(supports))
    (residual - torch.randn_like(residual)).square().mean().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    assert supports.grad.abs().sum() > 0
    assert module.value_center.grad is None
    storage = io.BytesIO()
    torch.save({"config": module.configuration(), "state": module.state_dict()}, storage)
    storage.seek(0)
    checkpoint = torch.load(storage, weights_only=True)
    restored = VectorVeRA(**checkpoint["config"])
    restored.load_state_dict(checkpoint["state"])
    assert torch.equal(restored.value_center, module.value_center)
    torch.testing.assert_close(
        restored(x, restored.encode_key(supports), restored.encode_value(supports)), residual,
        rtol=0, atol=0,
    )
