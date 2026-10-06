"""Real hook and causal capture plumbing using a tiny frozen CPU backbone."""
import pytest
import torch

from vera_mem.block_backend import BlockBackend
from test_block_vera import module, populate
from test_context_distillation import make_backend as make_plain_backend


def backend(mode="block_outer"):
    result = make_plain_backend(); result.hook.remove(); result.__class__ = BlockBackend
    result.vector_vera = module(mode, out_features=5)
    store, keys, values = populate(result.vector_vera, torch.randn(4, 3, 5))
    result.vector_keys, result.vector_values, result.vector_store = keys, values, None
    result.hook = result.target.register_forward_hook(result._inject)
    return result, store


@pytest.mark.parametrize("mode", ["diagonal", "pooled_outer", "block_outer"])
def test_actual_cpu_hook_uses_matrix_and_preserves_capture_and_trace(mode):
    b, store = backend(mode); x = torch.randn(2, 4, 5, requires_grad=True); base = torch.randn(2, 4, 5)
    expected = b.vector_vera(x, b.vector_keys, b.vector_values)
    b.vector_store = store; b.capture_layer_input = True; b.trace_retrieval = True
    rows = b._answer_capture_batch_indices = torch.tensor([0, 1, 1])
    positions = b._answer_capture_positions = torch.tensor([1, 1, 3])
    original = b.vector_vera.delta_from_value
    def forbidden(*args, **kwargs): raise AssertionError("CPU silently used pooled legacy hook")
    b.vector_vera.delta_from_value = forbidden
    actual = b._inject(None, (x,), base)
    torch.testing.assert_close(actual, base + expected)
    torch.testing.assert_close(b.last_answer_query_inputs, x[rows, positions])
    assert b.last_answer_query_inputs.requires_grad and len(store.route_log) == 1
    assert b.retrieval_trace[-1]["phase"] == "prefill"
    torch.testing.assert_close(b.retrieval_trace[-1]["indices"], store.route_log[-1]["indices"][:, -1])
    assert b.last_answer_residual_ratio.shape == (3,) and b.last_retrieval is b.prefill_retrieval
    torch.testing.assert_close(b.captured_input, x[0, -1].detach())
    b.vector_vera.delta_from_value = original


def test_causal_answer_positions_and_training_graph_reach_reader_writer_only():
    b, _ = backend(); questions, targets = ["q", "longer"], [[3, 4], [6, 7, 8]]
    result = b.forward_sequences(questions, targets)
    assert result["counts"].tolist() == [2, 3] and result["query_inputs"].shape == (5, 5)
    assert result["positions"].tolist() == [1, 2, 6, 7, 8]
    assert result["logits"].requires_grad and result["hidden"].requires_grad
    loss = result["logits"].square().mean() + .2*b.vector_vera.group_address_loss(result["query_inputs"], torch.zeros(5, dtype=torch.long), b.vector_keys)
    loss.backward()
    for name, p in b.vector_vera.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0 and bool(torch.isfinite(p.grad).all()), name
    assert all(p.grad is None for p in b.model.parameters())


def test_teacher_bypasses_full_matrix_and_restores_student_diagnostics():
    b, store = backend(); b.vector_store = store; b.trace_retrieval = True
    sentinel = {"indices": torch.tensor([[[0, 1, 2]]])}; b.last_retrieval = sentinel
    def forbidden(*args, **kwargs): raise AssertionError("teacher accessed block memory")
    b.vector_vera.cpu_delta = forbidden; b.vector_vera.encode_query = forbidden; store.search = forbidden
    x, base = torch.randn(1, 2, 5), torch.randn(1, 2, 5)
    with b._teacher_forward():
        torch.testing.assert_close(b._inject(None, (x,), base), base, rtol=0, atol=0)
    assert b.mode == "vector_vera" and b.last_retrieval is sentinel and b.trace_retrieval
    assert b.retrieval_trace == [] and store.route_log == []


@pytest.mark.parametrize("field", ["oracle_values", "vector_override"])
def test_vector_only_overrides_fail_closed(field):
    b, store = backend(); b.vector_store = store; setattr(b, field, torch.ones(1, 3))
    with pytest.raises(ValueError, match="full value block"):
        b._inject(None, (torch.randn(1, 1, 5),), torch.zeros(1, 1, 5))
    assert store.route_log == []
