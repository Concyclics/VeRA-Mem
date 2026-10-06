"""Whole-block operator controls and exact CPU-bank mechanisms, no model load."""
import math

import pytest
import torch
from torch.nn import functional as F

from vera_mem.block_vera import BlockVeRA, BlockVectorDB
from vera_mem.qkv_vera import QKVVeRA

MODES = ("diagonal", "pooled_outer", "block_outer")


def module(mode="block_outer", out_features=6):
    m = BlockVeRA(5, out_features, rank=3, key_dim=4, temperature=.3, seed=31, block_mode=mode)
    g = torch.Generator().manual_seed(91)
    m.fit_statistics(torch.randn(21, 5, generator=g), torch.randn(24, 5, generator=g))
    m.fit_value_statistics(torch.randn(24, 5, generator=g))
    with torch.no_grad():
        m.b.fill_(.3)
    return m


def populate(m, features):
    store = m.new_store(record_routes=True)
    keys, values = m.encode_bank(features)
    for i in range(features.shape[0]):
        store.write_group(str(i), keys[3*i:3*i+3], values[3*i:3*i+3], 0)
    return store, keys, values


def test_common_initialization_and_matched_factor_parameter_budget_preserve_rng():
    before = torch.random.get_rng_state().clone()
    models = [module(mode) for mode in MODES]
    assert torch.equal(before, torch.random.get_rng_state())
    common = dict(models[0].named_parameters())
    for m in models:
        for name, parameter in common.items():
            assert torch.equal(parameter, dict(m.named_parameters())[name]), name
        for name in ("A", "query_center", "support_center", "value_center"):
            assert torch.equal(m.state_dict()[name], models[0].state_dict()[name])
        assert not m.A.requires_grad and m.B.requires_grad
    for m in models[1:]:
        assert m.trainable_numel() - models[0].trainable_numel() == 2*3*3+3
        torch.testing.assert_close(m.P_in.weight, torch.eye(3), rtol=0, atol=0)
        torch.testing.assert_close(m.P_out.weight, torch.eye(3), rtol=0, atol=0)
        torch.testing.assert_close(m.P_in.bias.square().mean(), torch.tensor(1.))
        assert m.P_in.bias.requires_grad
    for name, p in models[1].named_parameters():
        assert torch.equal(p, dict(models[2].named_parameters())[name]), name


@pytest.mark.parametrize("mode", MODES)
def test_formula_retains_common_diagonal_and_outer_is_rank_one_or_three(mode):
    m = module(mode); x = torch.randn(2, 4, 5); values = torch.randn(2, 4, 3, 3)
    z = F.linear(x, m.A); mean = values.mean(-2); expected = z * mean
    if mode != "diagonal":
        rows = mean.unsqueeze(-2) if mode == "pooled_outer" else values
        u, w = m.block_factors(rows)
        operator = torch.einsum("...sr,...sk->...rk", w, u) / rows.shape[-2]
        expected = expected + torch.einsum("...rk,...k->...r", operator, z)
        assert torch.linalg.matrix_rank(operator).max() <= (1 if mode == "pooled_outer" else 3)
        torch.testing.assert_close(u.norm(dim=-1), torch.ones_like(u[..., 0]))
    torch.testing.assert_close(m.latent_from_block(x, values), expected)
    torch.testing.assert_close(m.delta_from_block(x, values), F.linear(expected, m.B) * m.b)


def test_same_mean_different_blocks_are_not_pooled_outer_rewrites():
    pooled, block = module("pooled_outer"), module("block_outer")
    for m in (pooled, block):
        with torch.no_grad():
            m.A.zero_(); m.A[:, :3].copy_(torch.eye(3)); m.P_in.bias.fill_(1.)
    x = torch.tensor([1., 0., 0., 0., 0.])
    v = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 0., 0.]])
    zeros = torch.zeros_like(v)
    assert torch.equal(v.mean(0), zeros.mean(0))
    torch.testing.assert_close(pooled.latent_from_block(x, v), pooled.latent_from_block(x, zeros))
    assert not torch.allclose(block.latent_from_block(x, v), block.latent_from_block(x, zeros))


@pytest.mark.parametrize("mode", MODES)
def test_uniform_operator_is_slot_permutation_invariant(mode):
    m = module(mode); x = torch.randn(2, 5); values = torch.randn(2, 3, 3)
    torch.testing.assert_close(m.delta_from_block(x, values), m.delta_from_block(x, values[:, [2, 0, 1]]))


@pytest.mark.parametrize("mode", ["pooled_outer", "block_outer"])
def test_nonzero_affine_input_bias_breaks_outer_sign_invariance(mode):
    m = module(mode); x = torch.randn(2, 5); values = torch.randn(2, 3, 3)
    z = F.linear(x, m.A)
    def outer(v): return m.latent_from_block(x, v) - z * v.mean(-2)
    assert not torch.allclose(outer(values), outer(-values))
    with torch.no_grad(): m.P_in.bias.zero_()
    # Positive control: two bias-free linear sides have the unwanted symmetry.
    torch.testing.assert_close(outer(values), outer(-values))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_same_encoded_full_bank_cpu_and_differentiable_tensor_paths_match(mode, dtype):
    m = module(mode); store, keys, values = populate(m, torch.randn(5, 3, 5)); before = store.hash()
    x = torch.randn(2, 4, 5).to(dtype)
    tensor, ti = m(x, keys, values, return_info=True)
    online, ci = m.cpu_delta(x, store, return_info=True)
    torch.testing.assert_close(online, tensor, rtol=3e-6, atol=3e-6)
    for name in ("indices", "scores", "weights", "selected_values", "queries", "selected_fact"):
        torch.testing.assert_close(ci[name], ti[name], rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(ci["weights"], torch.full_like(ci["weights"], 1/3))
    assert ci["selected_values"].shape == (2, 4, 3, 3) and online.dtype == dtype
    assert store.hash() == before and store.resident_bytes()["total_bytes"] == 5*3*(4+3)*4
    assert "mixed_value" not in ci


def test_saved_query_proves_global_logsumexp_winning_group_and_selected_values():
    m = module(); store, _, _ = populate(m, torch.randn(4, 3, 5))
    query = m.encode_query(torch.randn(2, 3, 5)); info = store.search(query)
    record = store.route_log[-1]
    assert all(not t.requires_grad and t.device.type == "cpu" for t in record.values() if isinstance(t, torch.Tensor))
    scores = F.normalize(record["queries"], dim=-1) @ F.normalize(store.keys, dim=-1).T
    group_logits = torch.logsumexp(scores.reshape(2, 3, 4, 3) / store.temperature, -1)
    assert torch.equal(group_logits.argmax(-1), record["selected_fact"])
    expected = 3 * record["selected_fact"][..., None] + torch.arange(3)
    assert torch.equal(expected, record["indices"])
    torch.testing.assert_close(record["scores"], scores.gather(-1, expected))
    torch.testing.assert_close(info["selected_values"], store.values[expected])
    # Operator weights have a different contract from selection scores.
    assert not torch.allclose(info["weights"], torch.softmax(info["scores"] / store.temperature, -1))


@pytest.mark.parametrize("mode", MODES)
def test_independent_banks_have_no_cross_row_value_or_query_gradients(mode):
    m = module(mode); x = torch.randn(2, 4, 5, requires_grad=True)
    features = torch.randn(2, 4, 3, 5, requires_grad=True)
    keys, values = m.encode_bank(features); values.retain_grad()
    actual = m(x, keys, values)
    expected = torch.stack([m(x[i], keys[i], values[i]) for i in range(2)])
    torch.testing.assert_close(actual, expected)
    changed = values.detach().clone(); changed[1] += 20
    torch.testing.assert_close(m(x, keys, changed)[0], actual[0], rtol=0, atol=0)
    actual[0].square().sum().backward()
    assert features.grad is None and values.grad[1].eq(0).all() and x.grad[1].eq(0).all()
    assert values.grad[0].abs().sum() > 0


def test_ce_does_not_train_discrete_group_keys_but_dense_address_does():
    m = module(); x = torch.randn(1, 1, 5, requires_grad=True)
    keys = torch.randn(9, 4, requires_grad=True); values = torch.randn(9, 3, requires_grad=True)
    result, info = m(x, keys, values, return_info=True)
    kg, vg = torch.autograd.grad(result.square().sum(), (keys, values), allow_unused=True, retain_graph=True)
    assert kg is None
    selected = info["indices"].flatten().tolist(); other = [i for i in range(9) if i not in selected]
    assert vg[other].eq(0).all() and bool((vg[selected].norm(dim=-1) > 0).all())
    address = m.group_address_loss(x, torch.tensor([[2]]), keys)
    dense, = torch.autograd.grad(address, (keys,))
    assert bool((dense.norm(dim=-1) > 0).all())


@pytest.mark.parametrize("mode", MODES)
def test_fresh_writer_graph_all_training_parameters_and_frozen_features(mode):
    m = module(mode); features = torch.randn(2, 12, 5, requires_grad=True); m.set_feature_bank(features)
    optimizer = torch.optim.SGD(m.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad(); x = torch.randn(2, 3, 5, requires_grad=True)
        loss = m(x).square().mean() + .2 * m.group_address_loss(x, torch.tensor([[0, 1, 2], [1, 2, 3]]))
        loss.backward()
        for name, parameter in m.named_parameters():
            assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
            assert parameter.grad.abs().sum() > 0, name
        assert features.grad is None and m.A.grad is None and x.grad.abs().sum() > 0
        optimizer.step()


@pytest.mark.parametrize("mode", MODES)
def test_empty_zero_without_statistics_and_teacher_strict_no_memory_access(mode):
    m = BlockVeRA(5, 6, rank=3, key_dim=4, block_mode=mode); x = torch.randn(2, 3, 5)
    empty = m.new_store(record_routes=True)
    with torch.no_grad(): m.b.fill_(1)
    a, info = m(x, torch.empty(0, 4), torch.empty(0, 3), True)
    b, online = m.cpu_delta(x, empty, return_info=True)
    assert a.eq(0).all() and b.eq(0).all()
    assert info["selected_values"].shape == online["selected_values"].shape == (2, 3, 0, 3)
    def forbidden(*args, **kwargs): raise AssertionError("teacher accessed memory")
    empty.search = forbidden; m.encode_query = forbidden
    actual, teacher = m.cpu_delta(x, empty, teacher=True, return_info=True)
    assert actual.eq(0).all() and teacher == {"teacher_bypass": True}


def test_atomic_write_snapshot_roundtrip_and_read_mode_rejection(tmp_path):
    m = module(); store, keys, values = populate(m, torch.randn(3, 3, 5)); original = store.snapshot()
    for timestamp, bad in [(-1, keys[:3]), (1, keys[:3].clone())]:
        before = store.hash()
        if timestamp == 1: bad[-1, -1] = float("nan")
        with pytest.raises(ValueError): store.write_group("0", bad, values[:3], timestamp)
        assert store.hash() == before
    store.write_group("1", keys[3:6], values[3:6] + 1, 1)
    assert store.timestamps[3:6] == (1,) * 3
    torch.testing.assert_close(store.values[:3], original["store"]["values"][:3], rtol=0, atol=0)
    torch.testing.assert_close(store.values[6:], original["store"]["values"][6:], rtol=0, atol=0)
    path = tmp_path / "bank.pt"; store.save(path); clone = BlockVectorDB.load(path)
    assert type(clone) is BlockVectorDB and clone.hash() == store.hash()
    for field, value in [("slot_weighting", "softmax"), ("read_mode", "flat"), ("block_version", 2)]:
        with pytest.raises(ValueError): BlockVectorDB.from_snapshot(dict(store.snapshot(), **{field: value}))
    partial = store.snapshot(); partial["store"]["timestamps"][3] = 2
    with pytest.raises(ValueError): BlockVectorDB.from_snapshot(partial)


def test_configuration_load_guard_and_old_mixed_value_hook_fail_closed():
    m = module(); clone = BlockVeRA(**m.configuration()); clone.load_state_dict(m.state_dict())
    for mode in ("diagonal", "pooled_outer"):
        wrong = BlockVeRA(**dict(m.configuration(), block_mode=mode))
        with pytest.raises(RuntimeError, match="architecture"): wrong.load_state_dict(m.state_dict())
    with pytest.raises(RuntimeError, match="BlockBackend"):
        m.delta_from_value(torch.randn(1, 5), torch.randn(1, 3))
    old = QKVVeRA(5, 6, rank=3, key_dim=4, read_mode="grouped").new_store()
    with pytest.raises(ValueError): m.cpu_delta(torch.randn(1, 5), old)
    for config in ({"read_mode": "flat"}, {"readout": "additive"}, {"slot_weighting": "softmax"},
                   {"block_version": 2}, {"residual_outer": False}, {"factor_epsilon": math.nan}):
        with pytest.raises(ValueError): BlockVeRA(5, 6, **config)


def test_cpu_query_validation_and_zero_query_batch():
    m = module(); store, _, _ = populate(m, torch.randn(2, 3, 5))
    for query in (torch.zeros(4), torch.full((4,), float("nan")), torch.full((4,), 1e38), torch.ones(5)):
        with pytest.raises(ValueError): store.search(query)
    info = store.search(torch.empty(0, 4))
    assert info["selected_values"].shape == (0, 3, 3) and info["indices"].shape == (0, 3) and info["ids"] == []
