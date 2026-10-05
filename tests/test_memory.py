"""Mechanism checks: selection, state isolation, and gradient flow."""

import pytest
import torch

from vera_mem.memory import ExactMemory, FrozenRouter, LatentResidual, LexicalMemory, LoRABank


def test_lora_zero_and_parameter_budget():
    bank = LoRABank(12, 8, rank=4, slots=4)
    x = torch.randn(2, 3, 12, dtype=torch.bfloat16)
    for slot in range(4):
        residual = bank(x, slot)
        assert residual.shape == (2, 3, 8)
        assert residual.dtype == x.dtype
        assert torch.count_nonzero(residual) == 0
    assert bank.trainable_numel(active=True) == 4 * (12 + 8)
    assert bank.trainable_numel() == 4 * bank.trainable_numel(active=True)


def test_lora_only_selected_slot_changes():
    bank = LoRABank(12, 8, rank=4, slots=4)
    before = {name: value.detach().clone() for name, value in bank.named_parameters()}
    optimizer = torch.optim.SGD(bank.parameters(), lr=0.1)
    target = torch.randn(2, 8)
    ((bank(torch.randn(2, 12), 2) - target) ** 2).mean().backward()
    optimizer.step()
    assert not torch.equal(before["B.2"], bank.B[2])
    for slot in [0, 1, 3]:
        assert torch.equal(before[f"A.{slot}"], bank.A[slot])
        assert torch.equal(before[f"B.{slot}"], bank.B[slot])
        assert bank.A[slot].grad is None
        assert bank.B[slot].grad is None


def test_lora_initialization_does_not_change_global_rng():
    torch.manual_seed(7)
    state = torch.random.get_rng_state().clone()
    first = LoRABank(12, 8, seed=9)
    second = LoRABank(12, 8, seed=9)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(first.A[0], second.A[0])


def test_latent_reader_learns_then_encoder_receives_gradients():
    module = LatentResidual(8, rank=3)
    support = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    assert torch.count_nonzero(module(module.encode(support))) == 0
    ((module(module.encode(support)) - target) ** 2).mean().backward()
    assert module.reader.weight.grad.abs().sum() > 0
    assert module.encoder.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    support.grad = None
    ((module(module.encode(support)) - target) ** 2).mean().backward()
    assert module.encoder.weight.grad.abs().sum() > 0
    assert support.grad.abs().sum() > 0


def test_exact_memory_empty_replace_and_cosine():
    bank = ExactMemory()
    assert bank.retrieve(torch.tensor([1.0, 0.0])) == []
    bank.write("one", torch.tensor([2.0, 0.0]), "first", timestamp=1)
    bank.write("two", torch.tensor([0.0, 3.0]), "second", timestamp=2)
    bank.write("one", torch.tensor([5.0, 0.0]), "updated", timestamp=3)
    found = bank.retrieve(torch.tensor([10.0, 0.0]), top_k=5)
    assert len(bank) == len(found) == 2
    assert found[0] == {"id": "one", "score": 1.0, "value": "updated", "timestamp": 3}
    assert found[1]["score"] == 0.0


def test_exact_memory_copies_inputs_outputs_and_restored_state():
    bank = ExactMemory()
    key = torch.tensor([1.0, 0.0])
    value = torch.tensor([3.0], requires_grad=True)
    bank.write("one", key, value, 1)
    key.zero_()
    with torch.no_grad():
        value.fill_(4)
    query = torch.tensor([1.0, 0.0])
    retrieved = bank.retrieve(query)[0]["value"]
    assert not retrieved.requires_grad
    retrieved.zero_()
    assert bank.retrieve(query)[0]["value"].item() == 3
    state = bank.state_dict()
    restored = ExactMemory()
    restored.load_state_dict(state)
    state["entries"][0]["value"].zero_()
    restored.write("one", query, torch.tensor([9.0]), 2)
    assert bank.retrieve(query)[0]["value"].item() == 3
    assert restored.retrieve(query)[0]["value"].item() == 9


def test_exact_memory_rejects_invalid_keys_without_changing_state():
    bank = ExactMemory()
    bank.write("one", torch.tensor([1.0, 0.0]), "one", 1)
    for invalid in [torch.zeros(2), torch.tensor([float("nan"), 1.0]), torch.ones(3)]:
        with pytest.raises(ValueError):
            bank.write("bad", invalid, "bad", 2)
    assert len(bank) == 1


def test_frozen_router_stable_seed_and_no_mutation():
    features = torch.tensor([[1.0, 0.1], [1.0, -0.1], [-1.0, 0.1], [-1.0, -0.1]])
    first = FrozenRouter(2, seed=4).fit(features)
    second = FrozenRouter(2, seed=4).fit(features)
    assert torch.equal(first.centroids, second.centroids)
    initial = first.centroids
    positive = first.route(torch.tensor([1.0, 0.0]))
    negative = first.route(torch.tensor([-1.0, 0.0]))
    assert positive != negative
    assert first.route(torch.tensor([2.0, 0.0])) == positive
    exposed = first.centroids
    exposed.zero_()
    assert torch.equal(first.centroids, initial)
    with pytest.raises(RuntimeError):
        first.fit(features)
    restored = FrozenRouter(2)
    restored.load_state_dict(first.state_dict())
    assert restored.route(torch.tensor([1.0, 0.0])) == positive


def test_router_handles_empty_clusters_and_degenerate_features():
    router = FrozenRouter(4).fit(torch.ones(8, 3))
    assert torch.isfinite(router.centroids).all()
    assert router.route(torch.ones(3)) == 0


def test_lexical_memory_retrieval_update_and_checkpoint():
    memory = LexicalMemory()
    assert memory.retrieve("Paris") == []
    memory.write("one", "Ada has a violet badge", 1)
    memory.write("two", "Lin has a green badge", 2)
    assert memory.retrieve("What badge does Ada have?")[0]["id"] == "one"
    memory.write("one", "Ada has a yellow badge", 3)
    assert len(memory) == 2
    assert memory.retrieve("Ada")[0]["text"] == "Ada has a yellow badge"
    state = memory.state_dict()
    restored = LexicalMemory()
    restored.load_state_dict(state)
    state["entries"][0]["text"] = "contaminated"
    assert restored.retrieve("Ada")[0]["text"] == "Ada has a yellow badge"
    assert restored.retrieve("Ada") == memory.retrieve("Ada")
