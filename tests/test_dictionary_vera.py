"""Actual two-bank gates, dense backward, and train-only compression contracts."""
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vera_mem.dictionary_vera import DictionaryVeRA, merge_bank
from vera_mem.interface_variants import InterfaceVectorVeRA
from vera_mem.vector_vera import VectorVeRA
from vera_mem.vector_store import PersistentVectorDB
from vera_mem.backend import QwenBackend


def module(**options):
    config = dict(in_features=5, out_features=6, rank=3, key_dim=4, top_k=2,
                  base_top_k=2, base_size=7, temperature=.6, seed=29)
    config.update(options)
    result = DictionaryVeRA(**config)
    rng = torch.Generator().manual_seed(17)
    result.fit_statistics(torch.randn(17, 5, generator=rng), torch.randn(19, 5, generator=rng))
    if result.writer_mode == "masked_mean":
        result.fit_value_statistics(torch.randn(23, 5, generator=rng))
    with torch.no_grad():
        result.b.copy_(torch.linspace(.2, .7, 6))
    return result


def data():
    rng = torch.Generator().manual_seed(49)
    return (torch.randn(2, 3, 5, generator=rng), torch.randn(8, 4, generator=rng),
            torch.randn(8, 3, generator=rng))


def explicit_mix(query, keys, values, top_k, temperature):
    similarity = query @ F.normalize(keys, dim=-1).T
    score, index = similarity.topk(min(top_k, len(keys)), -1)
    return (F.softmax(score / temperature, -1).unsqueeze(-1) * values[index]).sum(-2)


def test_two_quotas_then_one_gate_exactly_match_manual_composition():
    m = module()
    x, keys, values = data()
    query = m.encode_query(x)
    episodic = explicit_mix(query, keys, values, m.top_k, m.temperature)
    base = explicit_mix(query, m.base_keys, m.base_values, m.base_top_k, m.temperature)
    expected = VectorVeRA.delta_from_value(m, x, episodic + .25 * base)
    actual, info = m(x, keys, values, return_info=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert info["indices"].shape == info["base_indices"].shape == (2, 3, 2)
    assert (info["indices"] < len(keys)).all() and (info["base_indices"] < m.base_size).all()
    torch.testing.assert_close(info["weights"].sum(-1), torch.ones(2, 3))
    torch.testing.assert_close(info["base_weights"].sum(-1), torch.ones(2, 3))
    torch.testing.assert_close(m.delta_from_value(x, episodic), expected, rtol=0, atol=0)
    torch.testing.assert_close(m.final_delta(x, episodic, base_mixed=base), expected, rtol=0, atol=0)
    assert not torch.allclose(expected, VectorVeRA.delta_from_value(m, x, episodic + .5 * base))


def test_real_cpu_episodic_store_keeps_base_branch_and_explicit_cpu_base_is_equivalent():
    m = module().eval()
    x, keys, values = data()
    store = PersistentVectorDB(m.key_dim, m.rank, m.temperature, m.top_k)
    for i, (k, v) in enumerate(zip(keys, values)):
        store.write(str(i), k, v, i)
    query = m.encode_query(x)
    epi = store.search(query)["mixed_value"]
    base_store = m.base_cpu_store()
    expected = m(x, keys, values)
    torch.testing.assert_close(m.delta_from_value(x, epi), expected, rtol=1e-5, atol=1e-6)
    base = base_store.search(query)["mixed_value"]
    torch.testing.assert_close(m.final_delta(x, epi, base_mixed=base), expected, rtol=1e-5, atol=1e-6)
    assert not base.requires_grad
    empty_result = m(x, keys[:0], values[:0])
    assert empty_result.abs().sum() > 0
    torch.testing.assert_close(empty_result, m.delta_from_value(x, torch.zeros_like(epi)))
    assert not torch.allclose(empty_result, expected)


def test_existing_backend_hook_includes_base_once_on_tensor_and_cpu_paths():
    m = module().eval()
    x, keys, values = data()
    backend = QwenBackend.__new__(QwenBackend)
    backend.capture_layer_input, backend.mode = False, "vector_vera"
    backend.vector_vera, backend.vector_keys, backend.vector_values = m, keys, values
    backend.vector_store, backend.vector_override, backend.prefill_retrieval = None, None, None
    output = torch.randn(*x.shape[:-1], 6)
    tensor_result = backend._inject(None, (x,), output)
    store = PersistentVectorDB(m.key_dim, m.rank, m.temperature, m.top_k)
    for i, (key, value) in enumerate(zip(keys, values)):
        store.write(str(i), key, value, i)
    backend.vector_store = store
    with m.use_cpu_base(m.base_keys, m.base_values):
        cpu_result = backend._inject(None, (x,), output)
    torch.testing.assert_close(cpu_result, tensor_result, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cpu_result - output, m(x, keys, values), rtol=1e-5, atol=1e-6)


def test_cpu_override_off_arbitrary_size_nested_failure_and_hash_restore():
    m = module().eval()
    x, keys, values = data()
    before_state = {name: value.clone() for name, value in m.state_dict().items()}
    prior_trace = m.last_base_retrieval
    episodic = explicit_mix(m.encode_query(x), keys, values, m.top_k, m.temperature)
    with m.use_cpu_base(m.base_keys.detach()[:3], m.base_values.detach()[:3], ["a", "b", "c"]) as store:
        checksum = store.hash()
        expected_base = store.search(m.encode_query(x))["mixed_value"]
        expected = VectorVeRA.delta_from_value(m, x, episodic + .25 * expected_base)
        torch.testing.assert_close(m.delta_from_value(x, episodic), expected)
        assert m.last_base_retrieval["ids"]
        outer_trace = m.last_base_retrieval
        with pytest.raises(RuntimeError, match="intentional"):
            with m.use_cpu_base(keys[:0], values[:0]):
                torch.testing.assert_close(m(x, keys, values), VectorVeRA.delta_from_value(m, x, episodic))
                raise RuntimeError("intentional")
        assert m.base_cpu_override is store and m.last_base_retrieval is outer_trace
        assert store.hash() == checksum
    assert m.base_cpu_override is None and m.last_base_retrieval is prior_trace
    assert all(torch.equal(v, before_state[k]) for k, v in m.state_dict().items())
    with pytest.raises(ValueError, match="unique"):
        with m.use_cpu_base(keys[:2], values[:2], ["same", "same"]):
            pass
    m.train()
    with pytest.raises(RuntimeError, match="eval"):
        m.base_cpu_store()
    with pytest.raises(RuntimeError, match="eval"):
        with m.use_cpu_base(keys, values):
            pass


@pytest.mark.parametrize("mode", ["sparse", "dense", "straight_through"])
def test_independent_banks_match_separate_forwards_without_cross_row_gradients(mode):
    m = module(routing_mode=mode)
    x, shared_keys, shared_values = data()
    keys = torch.stack([shared_keys, shared_keys.roll(1, 0)]).requires_grad_()
    values = torch.stack([shared_values, shared_values * 2]).requires_grad_()
    actual = m(x, keys, values)
    expected = torch.stack([m(x[i], keys[i], values[i]) for i in range(2)])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual[0].square().sum().backward()
    assert keys.grad[0].abs().sum() > 0 and values.grad[0].abs().sum() > 0
    assert keys.grad[1].count_nonzero() == values.grad[1].count_nonzero() == 0
    assert m.base_keys.grad.abs().sum() > 0 and m.base_values.grad.abs().sum() > 0
    assert m.A.grad is None and m.B.grad is None


def test_st_has_sparse_forward_but_dense_gradients_including_unselected_values():
    x, input_keys, input_values = data()
    x = x[:1, :1]
    references = {}
    for mode in ("sparse", "dense", "straight_through"):
        m = module(routing_mode=mode)
        keys, values = input_keys.clone().requires_grad_(), input_values.clone().requires_grad_()
        mixed, info = m.mix_from_banks(x, keys, values, return_info=True)
        probe = torch.tensor([[[.7, -.2, 1.3]]])
        loss = (mixed * probe).sum()
        gradients = torch.autograd.grad(loss, (keys, values, m.base_keys, m.base_values, m.Wq.weight))
        references[mode] = mixed.detach(), info, gradients
    sparse, dense, st = (references[k] for k in ("sparse", "dense", "straight_through"))
    torch.testing.assert_close(st[0], sparse[0], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(dense[0], sparse[0])
    for actual, expected in zip(st[2], dense[2]):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    mask = torch.ones(len(input_values), dtype=torch.bool)
    mask[sparse[1]["indices"].flatten()] = False
    assert sparse[2][1][mask].count_nonzero() == 0
    assert (st[2][1][mask].norm(dim=-1) > 0).all()
    base_mask = torch.ones(7, dtype=torch.bool)
    base_mask[sparse[1]["base_indices"].flatten()] = False
    assert sparse[2][3][base_mask].count_nonzero() == 0
    assert (st[2][3][base_mask].norm(dim=-1) > 0).all()
    assert sparse[1]["indices"].shape[-1] == st[1]["indices"].shape[-1] == 2
    assert dense[1]["indices"].shape[-1] == len(input_values)
    assert bool(st[1]["dense_backward"]) and not bool(sparse[1]["dense_backward"])


@pytest.mark.parametrize("training_mode", ["dense", "straight_through"])
def test_eval_forces_sparse_for_both_banks_and_retained_dense_mass_is_true_mass(training_mode):
    m = module(routing_mode=training_mode)
    x, keys, values = data()
    m.eval()
    result, info = m(x, keys, values, return_info=True)
    query = m.encode_query(x)
    dense_weights = F.softmax(query @ F.normalize(keys, dim=-1).T / m.temperature, -1)
    expected_mass = dense_weights.gather(-1, info["topk_indices"]).sum(-1)
    torch.testing.assert_close(info["retained_dense_mass"], expected_mass)
    assert info["routing_mode_code"].item() == 0 and m.routing_mode == training_mode
    m.set_routing_mode("sparse")
    torch.testing.assert_close(result, m(x, keys, values), rtol=0, atol=0)


def test_frozen_dictionary_survives_runner_requires_grad_true_and_same_initialization():
    learned, frozen = module(), module(base_trainable=False)
    torch.testing.assert_close(learned.base_keys, frozen.base_keys, rtol=0, atol=0)
    torch.testing.assert_close(learned.base_values, frozen.base_values, rtol=0, atol=0)
    frozen.requires_grad_(False).requires_grad_(True)
    assert not frozen.base_keys.requires_grad and not frozen.base_values.requires_grad
    x, keys, values = data()
    frozen(x, keys, values).square().sum().backward()
    assert frozen.base_keys.grad is None and frozen.base_values.grad is None
    assert frozen.Wq.weight.grad.abs().sum() > 0
    frozen.set_base_trainable(True)
    assert frozen.base_keys.requires_grad and frozen.base_values.requires_grad
    frozen.set_base_trainable(False)
    assert frozen.configuration()["base_trainable"] is False


def test_dictionary_checkpoint_roundtrip_restores_modes_flags_and_learned_B_control():
    m = module(writer_mode="masked_mean", train_B=True, value_mlp_hidden=7,
               routing_mode="straight_through", base_trainable=False)
    payload = io.BytesIO()
    torch.save(m.checkpoint_payload(), payload)
    payload.seek(0)
    restored = DictionaryVeRA.from_checkpoint(torch.load(payload, weights_only=True))
    assert restored.configuration() == m.configuration()
    assert restored._statistics_ready and restored._value_statistics_ready
    assert isinstance(restored.B, nn.Parameter) and restored.A.grad is None
    assert not restored.base_keys.requires_grad
    assert restored.routing_mode == "straight_through"
    x, keys, values = data()
    torch.testing.assert_close(restored(x, keys, values), m(x, keys, values), rtol=0, atol=0)
    assert all(isinstance(v, torch.Tensor) for v in restored.state_dict().values())
    incompatible = module(alpha_base=.5)
    before = incompatible.base_keys.detach().clone()
    with pytest.raises(RuntimeError, match="architecture|alpha"):
        incompatible.load_state_dict(module().state_dict())
    assert torch.equal(before, incompatible.base_keys)


def test_base_off_checkpoint_roundtrip_and_explicit_activation_are_consistent():
    m = module(alpha_base=0., base_trainable=False)
    restored = DictionaryVeRA.from_checkpoint(m.checkpoint_payload())
    assert restored.alpha_base == 0. and restored.dictionary_alpha == 0.
    x, keys, values = data()
    expected = VectorVeRA.delta_from_value(m, x, explicit_mix(m.encode_query(x), keys, values, m.top_k, m.temperature))
    torch.testing.assert_close(restored(x, keys, values), expected)
    restored.set_alpha_base(.25)
    assert restored.configuration()["alpha_base"] == float(restored.dictionary_alpha) == .25
    active = DictionaryVeRA.from_checkpoint(restored.checkpoint_payload())
    torch.testing.assert_close(active(x, keys, values), restored(x, keys, values))


def test_interface_migration_and_zero_base_weight_match_previous_path():
    old = InterfaceVectorVeRA(5, 6, rank=3, key_dim=4, top_k=2, temperature=.6, seed=29)
    old.fit_statistics(torch.randn(13, 5), torch.randn(17, 5))
    with torch.no_grad():
        old.b.fill_(.4)
    new = module(alpha_base=0.)
    keys_before = new.base_keys.clone()
    new.load_interface_state_dict(old.state_dict())
    assert torch.equal(new.base_keys, keys_before)
    x, keys, values = data()
    torch.testing.assert_close(new(x, keys, values), old(x, keys, values), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="dictionary architecture"):
        new.load_state_dict(old.state_dict())


def test_weighted_spherical_merge_preserves_value_mean_scale_and_reports_variance():
    keys = torch.tensor([[1., 0.], [2., 0.], [0., 1.], [0., 3.]])
    values = torch.tensor([[1., 0.], [3., 0.], [0., 2.], [0., 6.]])
    weights = torch.tensor([1., 3., 2., 2.])
    before = torch.random.get_rng_state().clone()
    result = merge_bank(keys, values, 2, split="train", weights=weights, seed=9)
    assert torch.equal(before, torch.random.get_rng_state())
    for axis, expected_mean, variance in [(0, torch.tensor([2.5, 0.]), .375),
                                          (1, torch.tensor([0., 4.]), 2.)]:
        cluster = int(result["keys"][:, axis].argmax())
        torch.testing.assert_close(result["values"][cluster], expected_mean)
        assert result["counts"][cluster] == 2 and result["weight_sums"][cluster] == 4
        assert result["within_value_variance"][cluster] == variance
    assert result["counts"].sum() == 4
    assert not result["exact_function_preserving"] and not result["count_bias_applied"]
    assert not result["values"].requires_grad


def test_merging_duplicate_and_antipodal_keys_is_finite_deterministic_and_nonempty():
    keys = torch.tensor([[1., 0.]] * 6 + [[-1., 0.]] * 2)
    values = torch.arange(24.).view(8, 3)
    first = merge_bank(keys, values, 5, split="train", seed=5)
    second = merge_bank(keys, values, 5, split="train", seed=5)
    for name in ["keys", "values", "counts", "within_value_variance", "assignments"]:
        assert torch.equal(first[name], second[name]) and torch.isfinite(first[name]).all()
    assert (first["counts"] > 0).all()
    torch.testing.assert_close(first["keys"].norm(dim=-1), torch.ones(5))


def test_initialization_requires_training_provenance_and_preserves_record_bank():
    m = module(base_size=3)
    _, keys, values = data()
    keys_before, values_before = keys.clone(), values.clone()
    result = m.initialise_base(keys, values, split="train", seed=6)
    torch.testing.assert_close(m.base_keys, result["keys"])
    torch.testing.assert_close(m.base_values, result["values"])
    assert torch.equal(keys, keys_before) and torch.equal(values, values_before)
    for split in ["dev", "confirm", "test", None]:
        with pytest.raises(ValueError, match="train"):
            m.initialise_base(keys, values, split=split)
    with pytest.raises(ValueError, match="positive"):
        merge_bank(keys, values, 3, split="train", weights=torch.zeros(8))
    with pytest.raises(ValueError, match="nonzero"):
        merge_bank(keys * 0, values, 3, split="train")


@pytest.mark.parametrize("option", [dict(base_size=0), dict(base_top_k=True), dict(alpha_base=-1),
                                   dict(alpha_base=float("nan")), dict(base_trainable=1),
                                   dict(routing_mode="fake_sparse"), dict(dictionary_version=2)])
def test_invalid_dictionary_configuration_fails(option):
    with pytest.raises((ValueError, TypeError)):
        module(**option)


def test_construction_does_not_change_global_rng_and_defaults_keep_64_dimension_path():
    before = torch.random.get_rng_state().clone()
    m = DictionaryVeRA(5, 6)
    assert torch.equal(before, torch.random.get_rng_state())
    assert m.rank == m.key_dim == 64 and m.top_k == m.base_top_k == 4
    assert m.alpha_base == .25
    assert "A" in dict(m.named_buffers()) and "B" in dict(m.named_buffers())
    assert isinstance(m.base_keys, nn.Parameter) and isinstance(m.base_values, nn.Parameter)
