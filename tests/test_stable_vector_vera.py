"""Regression checks for frozen training statistics and stable memory values."""

import io

import pytest
import torch
from torch.nn import functional as F

from vera_mem.stable_vector_vera import StableVectorVeRA
from vera_mem.vector_vera import VectorVeRA


def _module(**kwargs):
    return StableVectorVeRA(8, 6, rank=4, key_dim=3, **kwargs)


def _fit(module):
    generator = torch.Generator().manual_seed(73)
    query = torch.randn(19, module.in_features, generator=generator) + 2
    support = torch.randn(23, module.in_features, generator=generator) - 3
    return module.fit_statistics(query, support)


def test_statistics_are_separate_fixed_and_do_not_change_projection_initialization():
    module = _module(seed=9)
    original = VectorVeRA(8, 6, rank=4, key_dim=3, seed=9)
    query, support = torch.randn(17, 8) + 2, torch.randn(11, 8) - 3
    before = {k: v.clone() for k, v in module.named_parameters()}
    module.fit_statistics(query, support)
    torch.testing.assert_close(module.query_center, module._normalized_input(query).mean(0))
    torch.testing.assert_close(module.support_center, module._normalized_input(support).mean(0))
    assert not torch.equal(module.query_center, module.support_center)
    assert torch.equal(original.A, module.A) and torch.equal(original.B, module.B)
    assert not module.A.requires_grad and not module.B.requires_grad
    for name, value in module.named_parameters():
        assert torch.equal(value, before[name]), name
    assert set(dict(module.named_parameters())) == {"b", "Wq.weight", "Wk.weight", "Wv.weight"}
    assert module.trainable_numel() == original.trainable_numel()
    state_before = {k: v.clone() for k, v in module.state_dict().items()}
    for _ in range(3):
        module.encode_query(torch.randn(4, 8))
        module.encode_key(torch.randn(5, 8))
        module.encode_value(torch.randn(6, 8))
    for name, value in module.state_dict().items():
        assert torch.equal(value, state_before[name]), name
    with pytest.raises(RuntimeError, match="already fitted"):
        module.fit_statistics(query + 100, support - 100)


def test_fit_is_required_and_invalid_inputs_cannot_partially_mutate_statistics():
    module = _module()
    for encode in (module.encode_query, module.encode_key, module.encode_value):
        with pytest.raises(RuntimeError, match="fit_statistics"):
            encode(torch.randn(2, 8))
    for invalid in (torch.empty(0, 8), torch.zeros(3, 9), torch.full((3, 8), float("nan"))):
        with pytest.raises(ValueError):
            module.fit_statistics(torch.randn(4, 8), invalid)
        assert not module._statistics_ready and not module.statistics_fitted.item()
        assert module.query_center.count_nonzero() == 0
        assert module.support_center.count_nonzero() == 0
    for epsilon in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            _module(input_epsilon=epsilon)
        with pytest.raises(ValueError):
            _module(value_epsilon=epsilon)


def test_center_then_normalize_exposes_small_content_variation():
    module = StableVectorVeRA(4, 4, rank=4, key_dim=4)
    common = torch.tensor([100., -100., 100., -100.])
    variation = torch.tensor([[1., 1., -1., -1.], [-1., -1., 1., 1.]])
    observations = common + variation
    module.fit_statistics(observations, observations)
    with torch.no_grad():
        module.Wq.weight.copy_(torch.eye(4))
        module.Wk.weight.copy_(torch.eye(4))
        module.Wv.weight.copy_(torch.eye(4))
    raw = F.normalize(observations, dim=-1)
    assert (raw[0] @ raw[1]).item() > .999
    centered_keys = module.encode_key(observations)
    assert (centered_keys[0] @ centered_keys[1]).item() < -.999
    values = module.encode_value(observations)
    assert torch.isfinite(values).all()
    assert not torch.equal(values[0], values[1])
    torch.testing.assert_close(values.square().mean(-1), torch.ones(2), atol=2e-5, rtol=0)


def test_values_have_bounded_rms_without_elementwise_tanh_saturation():
    module = StableVectorVeRA(4, 4, rank=4, key_dim=4)
    module.fit_statistics(torch.eye(4), -torch.eye(4))
    with torch.no_grad():
        module.Wv.weight.zero_()
        module.Wv.weight[0].fill_(3)
    value = module.encode_value(torch.ones(2, 4))
    assert value[:, 0].min() > 1.9  # tanh could never return this.
    assert value[:, 1:].count_nonzero() == 0
    assert (value.square().mean(-1) <= 1.00001).all()
    assert torch.isfinite(module.encode_value(torch.zeros(1, 4))).all()
    degenerate = StableVectorVeRA(4, 4, rank=4, key_dim=4)
    degenerate.fit_statistics(torch.zeros(3, 4), torch.zeros(3, 4))
    for encode in (degenerate.encode_query, degenerate.encode_key, degenerate.encode_value):
        assert encode(torch.zeros(1, 4)).count_nonzero() == 0


def test_zero_init_then_all_learnable_parameters_receive_gradients():
    torch.manual_seed(17)
    module = _fit(_module(top_k=3, temperature=.7))
    x, supports = torch.randn(3, 8), torch.randn(6, 8)
    result = module(x, module.encode_key(supports), module.encode_value(supports))
    assert result.count_nonzero() == 0
    (result - torch.randn_like(result)).square().mean().backward()
    assert module.b.grad.abs().sum() > 0
    for projection in (module.Wq, module.Wk, module.Wv):
        assert projection.weight.grad.count_nonzero() == 0
    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.b.fill_(.3)
    supports.requires_grad_(True)
    x.requires_grad_(True)
    result = module(x, module.encode_key(supports), module.encode_value(supports))
    (result - torch.randn_like(result)).square().mean().backward()
    for name, value in module.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        assert value.grad.abs().sum() > 0, name
    assert supports.grad.abs().sum() > 0 and x.grad.abs().sum() > 0
    assert module.query_center.grad is None and module.support_center.grad is None


def test_every_actual_token_addresses_values_and_external_mix_matches():
    module = StableVectorVeRA(3, 3, rank=3, key_dim=3, top_k=1)
    # Symmetric fitting gives zero centers, making expected addresses explicit.
    supports = torch.cat([torch.eye(3), -torch.eye(3)])
    module.fit_statistics(supports, supports)
    with torch.no_grad():
        for matrix in (module.Wq.weight, module.Wk.weight, module.A, module.B):
            matrix.copy_(torch.eye(3))
        module.b.fill_(1)
    x = torch.tensor([[[2., 0, 0], [0, 3., 0], [0, 0, 4.]]])
    values = torch.tensor([[.2, .3, .4], [.5, .6, .7], [.7, .8, .9]])
    result, info = module(x, torch.eye(3), values, return_info=True)
    assert info["indices"].tolist() == [[[0], [1], [2]]]
    torch.testing.assert_close(result, x * values.unsqueeze(0))
    mixed = (values[info["indices"]] * info["weights"].unsqueeze(-1)).sum(-2)
    torch.testing.assert_close(result, module.delta_from_value(x, mixed), rtol=0, atol=0)
    assert not torch.equal(result, module(x, torch.eye(3), values.roll(1, dims=0)))
    assert module(x, torch.empty(0, 3), torch.empty(0, 3)).count_nonzero() == 0


def test_checkpoint_restores_statistics_fitted_state_and_outputs_exactly():
    torch.manual_seed(23)
    before_rng = torch.random.get_rng_state().clone()
    module = _module(seed=14)
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    _fit(module)
    with torch.no_grad():
        module.b.fill_(.5)
    support, x = torch.randn(5, 8), torch.randn(2, 3, 8)
    expected = module(x, module.encode_key(support), module.encode_value(support))
    buffer = io.BytesIO()
    torch.save({"config": module.configuration(), "state": module.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = StableVectorVeRA(**saved["config"])
    restored.load_state_dict(saved["state"])
    assert restored._statistics_ready and restored.statistics_fitted.item()
    torch.testing.assert_close(
        expected, restored(x, restored.encode_key(support), restored.encode_value(support)),
        rtol=0, atol=0,
    )
    # Unfitted states must not silently become valid after serialization either.
    unfitted = _module()
    restored.load_state_dict(unfitted.state_dict())
    with pytest.raises(RuntimeError, match="fit_statistics"):
        restored.encode_query(x)
