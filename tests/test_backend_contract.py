"""No-model regression checks for the actual adaptation hook boundary.

Constructing the backend is deliberately bypassed: these tests exercise its
real hook/state methods without CUDA, model weights, or downloaded tokenizers.
"""

import pytest
import torch
from torch.nn import functional as F

from vera_mem.backend import QwenBackend
from vera_mem.vector_vera import VectorVeRA


class RecordingCPUStore:
    """Small deterministic store that records what the production hook sends."""

    def __init__(self):
        self.calls = []
        self.values = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.7, 0.8, 0.9]])

    def search(self, query):
        assert query.device.type == "cpu"
        assert not query.requires_grad
        self.calls.append(query.clone())
        index = query.argmax(dim=-1)
        return {
            "mixed_value": self.values[index],
            "indices": index.unsqueeze(-1),
            "scores": query.max(dim=-1).values.unsqueeze(-1),
            "weights": torch.ones((*query.shape[:-1], 1)),
        }


def _backend():
    backend = object.__new__(QwenBackend)
    backend.mode = "vector_vera"
    backend.capture_layer_input = False
    backend.captured_input = None
    backend.vector_vera = VectorVeRA(3, 3, rank=3, key_dim=3, top_k=2)
    with torch.no_grad():
        backend.vector_vera.Wq.weight.copy_(torch.eye(3))
        backend.vector_vera.Wk.weight.copy_(torch.eye(3))
        backend.vector_vera.A.copy_(torch.eye(3))
        backend.vector_vera.B.copy_(torch.eye(3))
        backend.vector_vera.b.fill_(1)
    backend.vector_store = RecordingCPUStore()
    backend.vector_override = None
    backend.vector_keys = None
    backend.vector_values = None
    backend.last_retrieval = None
    backend.prefill_retrieval = None
    return backend


def test_hook_queries_actual_input_for_every_token_and_uses_store_values():
    backend = _backend()
    x = torch.tensor([[[2.0, 0, 0], [0, 3.0, 0], [0, 0, 4.0]]], requires_grad=True)
    unchanged_output = torch.randn_like(x)
    result = backend._inject(None, (x,), unchanged_output)
    assert len(backend.vector_store.calls) == 1
    torch.testing.assert_close(backend.vector_store.calls[0], F.normalize(x.detach(), dim=-1))
    assert backend.vector_store.calls[0].shape == x.shape
    expected_delta = x * backend.vector_store.values.unsqueeze(0)
    torch.testing.assert_close(result, unchanged_output + expected_delta)
    # A second invocation receives a newly computed query, not a cached one.
    changed = x.detach().flip(dims=[1])
    backend._inject(None, (changed,), torch.zeros_like(changed))
    torch.testing.assert_close(backend.vector_store.calls[1], F.normalize(changed, dim=-1))
    assert not torch.equal(backend.vector_store.calls[0], backend.vector_store.calls[1])


def test_prefill_retrieval_survives_decode_and_teacher_forced_scoring():
    backend = _backend()
    prompt = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]])
    backend._inject(None, (prompt,), torch.zeros_like(prompt))
    first_indices = backend.prefill_retrieval["indices"].clone()
    decode = torch.tensor([[[0.0, 0, 1.0]]])
    backend._inject(None, (decode,), torch.zeros_like(decode))
    teacher_forced = torch.tensor([[[0.0, 0, 1.0]] * 5])
    backend._inject(None, (teacher_forced,), torch.zeros_like(teacher_forced))
    assert torch.equal(backend.prefill_retrieval["indices"], first_indices)
    assert backend.last_retrieval["indices"].shape == (1, 5, 1)
    assert backend.prefill_retrieval["indices"][0, -1].tolist() == [1]
    # Starting a new request is explicit, so its own prefill becomes measurable.
    backend.prefill_retrieval = None
    backend._inject(None, (decode,), torch.zeros_like(decode))
    assert backend.prefill_retrieval["indices"][0, -1].tolist() == [2]


def test_offline_hook_preserves_selected_key_value_and_query_gradients():
    torch.manual_seed(11)
    backend = _backend()
    backend.vector_store = None
    support = torch.randn(5, 3, requires_grad=True)
    backend.vector_keys = backend.vector_vera.encode_key(support)
    backend.vector_values = backend.vector_vera.encode_value(support)
    x = torch.randn(1, 3, 3, requires_grad=True)
    adapted = backend._inject(None, (x,), torch.zeros_like(x))
    # A frozen downstream projection must still transmit activation gradients.
    frozen_downstream = torch.nn.Linear(3, 4).requires_grad_(False)
    loss = (frozen_downstream(adapted) - torch.randn(1, 3, 4)).square().mean()
    loss.backward()
    for name, parameter in backend.vector_vera.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    assert support.grad.abs().sum() > 0
    assert all(not value.requires_grad for value in backend.last_retrieval.values())
    assert all(parameter.grad is None for parameter in frozen_downstream.parameters())


def test_disabled_feature_capture_skips_memory_and_restores_mode_on_error():
    backend = _backend()
    backend.capture_layer_input = True
    x, output = torch.randn(1, 4, 3), torch.randn(1, 4, 3)
    with pytest.raises(RuntimeError, match="observation failed"):
        with backend.disabled():
            assert backend.mode == "none"
            result = backend._inject(None, (x,), output)
            assert result is output
            torch.testing.assert_close(backend.captured_input, x[0, -1])
            assert not backend.captured_input.requires_grad
            raise RuntimeError("observation failed")
    assert backend.mode == "vector_vera"
    assert backend.vector_store.calls == []
    assert backend.prefill_retrieval is None


def test_oracle_and_empty_override_bypass_search_without_mutating_bank():
    backend = _backend()
    x, output = torch.randn(1, 4, 3), torch.randn(1, 4, 3)
    before = backend.vector_store.values.clone()
    backend.vector_override = torch.tensor([0.2, -0.3, 0.4])
    oracle = backend._inject(None, (x,), output)
    torch.testing.assert_close(oracle, output + x * backend.vector_override)
    assert backend.last_retrieval["indices"].numel() == 0
    backend.vector_override = torch.zeros(3)
    torch.testing.assert_close(backend._inject(None, (x,), output), output, rtol=0, atol=0)
    assert backend.vector_store.calls == []
    assert torch.equal(backend.vector_store.values, before)
