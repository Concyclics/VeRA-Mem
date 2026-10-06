"""Explicit support pooling and learned-B controls preserve the old interface."""
import io

import pytest
import torch
from torch import nn

from vera_mem.interface_variants import InterfaceVectorVeRA, masked_mean_support
from vera_mem.stable_vector_vera import StableVectorVeRA


def make_baseline():
    generator = torch.Generator().manual_seed(93)
    module = StableVectorVeRA(5, 6, rank=3, key_dim=4, top_k=3, temperature=.6, seed=17)
    module.fit_statistics(torch.randn(19, 5, generator=generator),
                          torch.randn(23, 5, generator=generator))
    with torch.no_grad():
        module.b.copy_(torch.linspace(.2, .7, 6))
    return module


def migrate(baseline=None, **options):
    baseline = make_baseline() if baseline is None else baseline
    module = InterfaceVectorVeRA(**baseline.configuration(), **options)
    module.load_legacy_state_dict(baseline.state_dict())
    return module


def test_default_migration_preserves_every_tensor_output_and_parameter_gradient():
    original = make_baseline()
    old_state = {name: value.clone() for name, value in original.state_dict().items()}
    module = migrate(original)
    assert set(module.state_dict()) == set(old_state) | {"interface_architecture"}
    assert set(dict(module.named_parameters())) == set(dict(original.named_parameters()))
    assert module.configuration() == dict(original.configuration(), writer_mode="last_token",
                                          train_B=False, value_mlp_hidden=0, architecture_version=1)
    for name, value in original.state_dict().items():
        assert torch.equal(value, old_state[name]) and torch.equal(module.state_dict()[name], value)
    generator = torch.Generator().manual_seed(100)
    x = torch.randn(2, 4, 5, generator=generator)
    support = torch.randn(7, 5, generator=generator)
    for operation in ("encode_query", "encode_key", "encode_value"):
        torch.testing.assert_close(getattr(module, operation)(support),
                                   getattr(original, operation)(support), rtol=0, atol=0)
    actual, actual_info = module(x, module.encode_key(support), module.encode_value(support), return_info=True)
    expected, expected_info = original(x, original.encode_key(support), original.encode_value(support), return_info=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name in expected_info:
        torch.testing.assert_close(actual_info[name], expected_info[name], rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for name, parameter in module.named_parameters():
        reference = dict(original.named_parameters())[name]
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)


def test_learned_B_starts_identical_but_updates_shared_readout_only():
    baseline, learned = make_baseline(), migrate(train_B=True)
    assert "B" in dict(learned.named_parameters()) and "B" not in dict(learned.named_buffers())
    assert "A" in dict(learned.named_buffers()) and "A" not in dict(learned.named_parameters())
    assert learned.trainable_numel() == baseline.trainable_numel() + baseline.out_features * baseline.rank
    x, values = torch.randn(3, 5), torch.randn(3, 3)
    torch.testing.assert_close(learned.delta_from_value(x, values), baseline.delta_from_value(x, values), rtol=0, atol=0)
    frozen_A, initial_B = learned.A.clone(), learned.B.detach().clone()
    optimizer = torch.optim.SGD([learned.B], lr=.03)
    learned.delta_from_value(x, values).square().sum().backward()
    assert learned.B.grad.abs().sum() > 0 and learned.A.grad is None
    optimizer.step()
    assert torch.equal(learned.A, frozen_A) and not torch.equal(learned.B, initial_B)
    assert learned.encode_value(torch.randn(4, 5)).shape == (4, baseline.rank)


def test_pooling_excludes_prefix_scaffolding_and_padding_with_exact_mask_gradients():
    features = torch.tensor([[[100., 100.], [2., 4.], [6., 8.], [-100., -100.]],
                             [[float("nan"), 0.], [9., 12.], [999., 999.], [99., 99.]]],
                            requires_grad=True)
    mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=torch.bool)
    pooled = masked_mean_support(features, mask)
    torch.testing.assert_close(pooled, torch.tensor([[4., 6.], [9., 12.]]), rtol=0, atol=0)
    pooled.sum().backward()
    expected_grad = torch.tensor([[[0., 0.], [.5, .5], [.5, .5], [0., 0.]],
                                  [[0., 0.], [1., 1.], [0., 0.], [0., 0.]]])
    torch.testing.assert_close(features.grad, expected_grad, rtol=0, atol=0)


def test_pooling_supports_per_query_banks_and_accumulates_bfloat16_in_float32():
    features = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5).bfloat16()
    mask = torch.zeros(2, 3, 4, dtype=torch.int64)
    mask[..., 1:3] = 1
    actual = masked_mean_support(features, mask)
    assert actual.dtype == torch.float32 and actual.shape == (2, 3, 5)
    torch.testing.assert_close(actual, features[..., 1:3, :].float().mean(-2), rtol=0, atol=0)


@pytest.mark.parametrize("features,mask", [
    (torch.tensor(1.), torch.tensor(1)),
    (torch.ones(0, 5), torch.ones(0, dtype=torch.bool)),
    (torch.ones(2, 0), torch.ones(2, dtype=torch.bool)),
    (torch.ones(2, 5, dtype=torch.int64), torch.ones(2, dtype=torch.bool)),
    (torch.ones(2, 5), torch.ones(1, dtype=torch.bool)),
    (torch.ones(2, 5), torch.ones(2)),
    (torch.ones(2, 5), torch.tensor([1, 2])),
    (torch.ones(2, 3, 5), torch.tensor([[1, 1, 1], [0, 0, 0]])),
    (torch.tensor([[float("nan"), 1.]]), torch.ones(1, dtype=torch.bool)),
])
def test_pooling_rejects_invalid_masks_shapes_and_nonfinite_selected_features(features, mask):
    with pytest.raises(ValueError):
        masked_mean_support(features, mask)


def test_pooled_writer_uses_separate_immutable_train_center_and_old_key_path():
    baseline = make_baseline()
    pooled = migrate(baseline, writer_mode="masked_mean")
    generator = torch.Generator().manual_seed(108)
    train_tokens = torch.randn(11, 6, 5, generator=generator)
    train_mask = torch.tensor([[0, 1, 1, 1, 0, 0]] * 11, dtype=torch.bool)
    train_pool = pooled.pool_support(train_tokens, train_mask)
    with pytest.raises(RuntimeError, match="Fit pooled value statistics"):
        pooled.encode_value(train_pool)
    old_query, old_support = pooled.query_center.clone(), pooled.support_center.clone()
    pooled.fit_value_statistics(train_pool)
    expected_center = pooled._normalized_input(train_pool).mean(0)
    torch.testing.assert_close(pooled.value_center, expected_center, rtol=0, atol=0)
    assert pooled.value_statistics_fitted.item() and pooled._value_statistics_ready
    assert torch.equal(old_query, pooled.query_center) and torch.equal(old_support, pooled.support_center)
    assert not torch.allclose(pooled.value_center, pooled.support_center)
    heldout_tokens = torch.randn(4, 6, 5, generator=generator)
    last_token = heldout_tokens[:, -1]
    torch.testing.assert_close(pooled.encode_key(last_token), baseline.encode_key(last_token), rtol=0, atol=0)
    test_mask = train_mask[:4]
    actual = pooled.encode_pooled_value(heldout_tokens, test_mask)
    normalized = pooled._domain_input(heldout_tokens[:, 1:4].mean(1), pooled.value_center)
    expected = pooled._rms_normalize(pooled.Wv(normalized), pooled.value_epsilon)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual, pooled.encode_value(last_token))
    assert torch.equal(pooled.value_center, expected_center)
    with pytest.raises(RuntimeError, match="already fitted"):
        pooled.fit_value_statistics(last_token)


def test_new_value_stats_validate_before_mutation_and_pooling_requires_explicit_mode():
    pooled = migrate(writer_mode="masked_mean")
    before = pooled.value_center.clone()
    for invalid in (torch.empty(0, 5), torch.randn(3, 4), torch.randn(2, 3, 5), torch.full((3, 5), float("nan"))):
        with pytest.raises(ValueError):
            pooled.fit_value_statistics(invalid)
        assert torch.equal(pooled.value_center, before) and not pooled._value_statistics_ready
    with pytest.raises(ValueError, match="Expected token feature width"):
        pooled.pool_support(torch.tensor(1.), torch.tensor(True))
    with pytest.raises(ValueError, match="masked_mean"):
        migrate().pool_support(torch.randn(2, 3, 5), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="Separate value statistics"):
        migrate().fit_value_statistics(torch.randn(3, 5))


def test_residual_MLP_is_exact_at_initialization_and_both_layers_can_learn():
    baseline = make_baseline()
    rng_before = torch.random.get_rng_state().clone()
    richer = migrate(baseline, value_mlp_hidden=7)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert richer.trainable_numel() == baseline.trainable_numel() + 5 * 7 + 7 * 3
    assert not any(name.startswith("value_mlp") for name, _ in migrate().named_parameters())
    inputs = torch.randn(6, 5)
    torch.testing.assert_close(richer.encode_value(inputs), baseline.encode_value(inputs), rtol=0, atol=0)
    weights = torch.randn(6, 3)
    (richer.encode_value(inputs) * weights).sum().backward()
    assert richer.value_mlp[-1].weight.grad.abs().sum() > 0
    assert richer.value_mlp[0].weight.grad.count_nonzero() == 0
    with torch.no_grad():
        richer.value_mlp[-1].weight.add_(richer.value_mlp[-1].weight.grad, alpha=-.01)
    richer.zero_grad()
    (richer.encode_value(inputs) * weights).sum().backward()
    assert richer.value_mlp[0].weight.grad.abs().sum() > 0


def test_checkpoint_roundtrip_restores_architecture_centers_flags_and_B_parameter():
    module = migrate(writer_mode="masked_mean", train_B=True, value_mlp_hidden=7)
    module.fit_value_statistics(torch.randn(13, 5))
    with torch.no_grad():
        module.B.add_(.03)
        module.value_mlp[-1].weight.add_(.05)
    buffer = io.BytesIO()
    torch.save(module.checkpoint_payload(), buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=True)
    assert all(isinstance(value, torch.Tensor) for value in payload["module"].values())
    restored = InterfaceVectorVeRA.from_checkpoint(payload)
    assert restored.configuration() == module.configuration()
    assert restored._value_statistics_ready and restored._statistics_ready
    assert isinstance(restored.B, nn.Parameter) and "A" in dict(restored.named_buffers())
    for name, value in module.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    inputs = torch.randn(4, 5)
    torch.testing.assert_close(restored.encode_value(inputs), module.encode_value(inputs), rtol=0, atol=0)
    torch.testing.assert_close(restored.delta_from_value(inputs, restored.encode_value(inputs)),
                               module.delta_from_value(inputs, module.encode_value(inputs)), rtol=0, atol=0)


def test_regular_load_rejects_old_or_conflicting_architectures_before_mutation():
    module = migrate()
    before = {name: value.clone() for name, value in module.state_dict().items()}
    for incompatible in (make_baseline().state_dict(), migrate(train_B=True).state_dict(),
                         migrate(value_mlp_hidden=7).state_dict(), migrate(writer_mode="masked_mean").state_dict()):
        with pytest.raises(RuntimeError, match="architecture"):
            module.load_state_dict(incompatible)
        assert all(torch.equal(value, before[name]) for name, value in module.state_dict().items())
    corrupted = dict(module.state_dict())
    corrupted["interface_architecture"] = torch.tensor([1., 0., 0., 0.])
    with pytest.raises(RuntimeError, match="Invalid interface architecture"):
        module.load_state_dict(corrupted)


def test_legacy_restore_requires_explicit_configuration_and_preserves_strict_validation():
    baseline = make_baseline()
    with pytest.raises(ValueError, match="explicit legacy_configuration"):
        InterfaceVectorVeRA.from_checkpoint({"module": baseline.state_dict()})
    restored = InterfaceVectorVeRA.from_checkpoint({"module": baseline.state_dict()},
                                                  legacy_configuration=baseline.configuration())
    assert restored._statistics_ready and torch.equal(restored.Wv.weight, baseline.Wv.weight)
    incomplete = dict(baseline.state_dict())
    del incomplete["Wv.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        migrate().load_legacy_state_dict(incomplete)
    with pytest.raises(ValueError, match="already has interface architecture"):
        restored.load_legacy_state_dict(restored.state_dict())
    pooled = migrate(writer_mode="masked_mean")
    pooled.fit_value_statistics(torch.randn(4, 5))
    with pytest.raises(RuntimeError, match="before fitting"):
        pooled.load_legacy_state_dict(baseline.state_dict())


def test_nested_checkpoint_load_restores_pooled_statistics_and_rejects_wrong_architecture():
    source = nn.ModuleDict({"memory": migrate(writer_mode="masked_mean", train_B=True)})
    source["memory"].fit_value_statistics(torch.randn(4, 5))
    target = nn.ModuleDict({"memory": migrate(writer_mode="masked_mean", train_B=True)})
    target.load_state_dict(source.state_dict())
    assert target["memory"]._value_statistics_ready and target["memory"]._statistics_ready
    wrong = nn.ModuleDict({"memory": migrate()})
    with pytest.raises(RuntimeError, match="architecture differs"):
        wrong.load_state_dict(source.state_dict())


def test_pooled_writer_and_learned_B_inherit_independent_banks_without_cross_row_gradients():
    module = migrate(writer_mode="masked_mean", train_B=True)
    module.fit_value_statistics(torch.randn(10, 5))
    x = torch.randn(2, 3, 5)
    support = torch.randn(2, 7, 5, requires_grad=True)
    tokens = torch.randn(2, 7, 4, 5, requires_grad=True)
    mask = torch.zeros(2, 7, 4, dtype=torch.bool)
    mask[..., 1:3] = True
    keys, values = module.encode_key(support), module.encode_pooled_value(tokens, mask)
    actual = module(x, keys, values)
    expected = torch.stack([module(x[row], keys[row], values[row]) for row in range(2)])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual[0].square().sum().backward()
    assert support.grad[0].abs().sum() > 0 and support.grad[1].count_nonzero() == 0
    assert tokens.grad[0, :, 1:3].abs().sum() > 0 and tokens.grad[1].count_nonzero() == 0
    assert tokens.grad[:, :, [0, 3]].count_nonzero() == 0
    assert module.B.grad.abs().sum() > 0 and module.Wv.weight.grad.abs().sum() > 0
    assert module.A.grad is None


@pytest.mark.parametrize("options", [
    {"writer_mode": "mean_without_mask"}, {"train_B": 1}, {"value_mlp_hidden": -1},
    {"value_mlp_hidden": True}, {"value_mlp_hidden": 2.5}, {"architecture_version": 2},
])
def test_invalid_architecture_options_fail_explicitly(options):
    with pytest.raises((ValueError, TypeError)):
        InterfaceVectorVeRA(5, 6, **options)
