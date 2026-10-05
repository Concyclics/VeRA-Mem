"""CPU protocol tests for augmentation, validation, and causal online writes."""

from copy import deepcopy
from dataclasses import asdict
import json
import random
from types import SimpleNamespace

import pytest
import torch

from vera_mem import augmentation_data as data
from vera_mem import generalization_run as run
from vera_mem.stable_vector_vera import StableVectorVeRA
from vera_mem.vector_store import PersistentVectorDB


@pytest.fixture
def cpu_tensor_transfers(monkeypatch):
    """Exercise protocol logic without changing production CUDA requirements."""
    original = torch.Tensor.to

    def redirect(tensor, *args, **kwargs):
        if args and args[0] == "cuda":
            args = ("cpu", *args[1:])
        if kwargs.get("device") == "cuda":
            kwargs = {**kwargs, "device": "cpu"}
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", redirect)


def _module():
    module = StableVectorVeRA(8, 6, rank=64, key_dim=64, top_k=4, seed=67)
    generator = torch.Generator().manual_seed(10)
    module.fit_statistics(torch.randn(24, 8, generator=generator),
                          torch.randn(24, 8, generator=generator))
    return module


def test_choose_views_preserves_fact_alignment_and_does_not_use_global_rng():
    # Encode fact and view IDs independently so accidental cross-fact selection
    # or a scalar shared template choice would be observable.
    features = torch.tensor([[[entity, view] for view in range(4)] for entity in range(12)])
    indices = [8, 2, 11, 0, 7, 6, 5, 4]
    random.seed(900)
    before = random.getstate()
    canonical, views = run.choose_views(features, indices, random.Random(17), False)
    assert views == [0] * len(indices)
    assert canonical.tolist() == [[entity, 0] for entity in indices]
    actual, views = run.choose_views(features, indices, random.Random(17), True)
    assert len(set(views)) > 1
    assert actual.tolist() == [[entity, view] for entity, view in zip(indices, views)]
    duplicate, duplicate_views = run.choose_views(features, indices, random.Random(17), True)
    torch.testing.assert_close(actual, duplicate)
    assert views == duplicate_views
    assert random.getstate() == before


def test_consistency_trains_only_selected_encoders_and_preserves_random_branch():
    module = _module()
    generator = torch.Generator().manual_seed(84)
    q1, q2, s1, s2 = [torch.randn(12, 8, generator=generator) for _ in range(4)]
    buffers_before = {name: tensor.clone() for name, tensor in module.named_buffers()}
    addressing_loss = run.view_consistency(module, q1, q2, s1, s2, values=False)
    addressing_loss.backward()
    for layer in (module.Wq, module.Wk):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.norm() > 0
    assert module.Wv.weight.grad is None
    assert module.b.grad is None
    module.zero_grad(set_to_none=True)
    all_loss = run.view_consistency(module, q1, q2, s1, s2, values=True)
    assert all_loss > addressing_loss.detach()
    all_loss.backward()
    assert module.Wv.weight.grad is not None
    assert torch.isfinite(module.Wv.weight.grad).all()
    assert module.Wv.weight.grad.norm() > 0
    assert module.b.grad is None
    assert module.A.grad is None and module.B.grad is None
    for name, tensor in module.named_buffers():
        torch.testing.assert_close(tensor, buffers_before[name])
    # Paired identical views are already invariant; the objective should not
    # create a spurious value-distance penalty in this case.
    identical = run.view_consistency(module, q1, q1, s1, s1)
    assert abs(float(identical.detach())) < 1e-6


class FeatureBackend:
    def __init__(self):
        self.layer_feature_cache = {"old": "unused"}
        self.calls = []

    def layer_features(self, texts, batch_size):
        self.calls.append(list(texts))
        assert batch_size == 32
        return torch.tensor([[len(text), sum(map(ord, text))] for text in texts], dtype=torch.float32)


def test_preparation_preserves_template_fact_axes_and_never_encodes_controls(tmp_path, monkeypatch):
    actual = data.datasets(16, 16)
    tiny = {split: rows[:2] for split, rows in actual.items()}
    monkeypatch.setattr(run.data, "datasets", lambda *args, **kwargs: tiny)
    backend = FeatureBackend()
    cache = tmp_path / "features.pt"
    packet = run.prepare_features(backend, cache)
    assert cache.exists() and not cache.with_suffix(".partial").exists()
    assert "control" not in packet["features"]
    assert packet["examples"]["control"] == [asdict(example) for example in tiny["control"]]
    assert packet["features"]["train"]["q"].shape == (2, 8, 2)
    assert packet["features"]["train"]["s"].shape == (2, 4, 2)
    assert packet["features"]["dev"]["q"].shape == (2, 3, 2)
    assert packet["features"]["test"]["s"].shape == (2, 4, 2)
    encoded = [text for call in backend.calls for text in call]
    for example in tiny["control"]:
        assert all(example.metadata["entity"] not in text for text in encoded)
    for split in ("train", "dev", "test"):
        for kind, renderer in (("q", data.render_question), ("s", data.render_support)):
            for view, template_id in enumerate(packet["template_ids"][split][kind]):
                for index, example in enumerate(tiny[split]):
                    text = renderer(example, template_id)
                    if kind == "s":
                        text = "Remember this information: " + text
                    torch.testing.assert_close(packet["features"][split][kind][index, view],
                                               torch.tensor([len(text), sum(map(ord, text))], dtype=torch.float32))
    assert backend.layer_feature_cache == {}
    call_count = len(backend.calls)
    reloaded = run.prepare_features(backend, cache)
    assert len(backend.calls) == call_count
    torch.testing.assert_close(reloaded["features"]["train"]["q"], packet["features"]["train"]["q"])


@pytest.mark.parametrize("field", ["protocol", "model_revision", "data_fingerprint"])
def test_cache_rejects_each_provenance_mismatch_before_feature_extraction(tmp_path, field):
    packet = {"protocol": run.PROTOCOL, "model_revision": run.REVISION,
              "data_fingerprint": data.protocol_fingerprint()}
    packet[field] = "wrong-provenance"
    cache = tmp_path / "wrong.pt"
    torch.save(packet, cache)
    backend = FeatureBackend()
    with pytest.raises(ValueError, match="provenance"):
        run.prepare_features(backend, cache)
    assert backend.calls == []


def test_validation_uses_only_development_facts_and_macro_averages_conditions(monkeypatch, cpu_tensor_transfers):
    examples = data.datasets(16, 16)["dev"][:2]
    qforms = run.templates("dev", "query")
    sforms = run.templates("dev", "support")
    packet = {
        "examples": {"dev": [asdict(example) for example in examples]},
        "features": {"dev": {"q": torch.tensor([[[view]] for view in range(3)]).transpose(0, 1).repeat(2, 1, 1),
                               "s": torch.tensor([[[view * 10]] for view in range(3)]).transpose(0, 1).repeat(2, 1, 1)}},
        "template_ids": {"dev": {"q": [form.id for form in qforms], "s": [form.id for form in sforms]}},
    }
    calls = []

    def fake_validation(backend, module, rows, q, s):
        assert not torch.is_grad_enabled()
        qi, si = int(q[0, 0]), int(s[0, 0]) // 10
        assert [row.id for row in rows] == [example.id for example in examples]
        assert [row.question for row in rows] == [data.render_question(example, qforms[qi]) for example in examples]
        calls.append((qi, si))
        return {"real_answer_token_nll": float(qi + 10 * si)}

    monkeypatch.setattr(run, "validation", fake_validation)
    result = run.validate(object(), object(), packet)
    assert calls == [(0, 0), (1, 0), (2, 0), (1, 1), (2, 2)]
    assert result["selection_nll"] == pytest.approx((0 + 1 + 2 + 11 + 22) / 5)
    assert all(not form.id.startswith("test") for form in qforms + sforms)


def test_online_write_boundaries_independent_banks_and_frozen_parameters(tmp_path, monkeypatch, cpu_tensor_transfers):
    dataset = data.datasets(16, 16)
    examples = dataset["test"][:6]
    qforms, sforms = run.templates("test", "query"), run.templates("test", "support")
    features = torch.randn(6, 4, 8, generator=torch.Generator().manual_seed(929))
    packet = {
        "examples": {"test": [asdict(example) for example in examples],
                     "control": [asdict(example) for example in dataset["control"]]},
        "features": {"test": {"s": features}},
        "template_ids": {"test": {"q": [form.id for form in qforms], "s": [form.id for form in sforms]}},
    }
    module = _module().requires_grad_(False).eval()
    before = deepcopy(module.state_dict())
    calls = []

    def fake_evaluate(backend, current_module, db, rows, method, phase, output):
        assert current_module is module
        assert backend.vector_vera is module
        assert all(not parameter.requires_grad for parameter in module.parameters())
        calls.append((phase, method, tuple(db.ids), tuple(row.id for row in rows)))
        if phase == "pre_write":
            assert rows[0].id not in db.ids
            assert len(db.ids) == examples.index(rows[0])
        elif phase == "immediate":
            assert rows[0].id == db.ids[-1]
        elif phase == "control_before":
            assert len(db.ids) == 0
            assert all(row.metadata["never_written"] for row in rows)
        elif phase == "control_after":
            assert len(db.ids) == len(examples)
            assert all(row.metadata["never_written"] for row in rows)
            assert set(db.ids).isdisjoint(row.id for row in rows)
        else:
            assert tuple(db.ids) == tuple(example.id for example in examples)
        return [dict(id=row.id, method=method, expected_in_bank=row.id in db.ids,
                     selected_ids=[row.id] if row.id in db.ids else [], em=1,
                     nll_sum=1., tokens=1) for row in rows]

    monkeypatch.setattr(run, "evaluate", fake_evaluate)
    result = run.online(SimpleNamespace(), module, {"eval_size": 6}, packet, tmp_path)
    assert result["online_gradient_steps"] == 0 and result["shared_parameters_unchanged"]
    assert result["eval_facts"] == 6
    writes = json.loads((tmp_path / "writes.json").read_text())
    assert len(writes) == 12
    for bank in ("canonical_support", "heldout_support"):
        saved = PersistentVectorDB.load(tmp_path / (bank + "_vdb.pt"))
        bank_writes = [row for row in writes if row["bank"] == bank]
        assert len(saved.ids) == 6
        assert [row["id"] for row in bank_writes] == [example.id for example in examples]
        assert [row["timestamp"] for row in bank_writes] == list(range(6))
        for i, example in enumerate(examples):
            view = 0 if bank == "canonical_support" else 1 + i % 3
            assert bank_writes[i]["support_template"] == sforms[view].id
            torch.testing.assert_close(saved.keys[i], module.encode_key(features[i, view]))
            torch.testing.assert_close(saved.values[i], module.encode_value(features[i, view]))
        assert set(saved.ids).isdisjoint(example.id for example in dataset["control"])
    readout_calls = [call for call in calls if "/" in call[0]]
    assert len(readout_calls) == 2 * 4 * 4
    assert len([call for call in calls if call[0] == "control_before"]) == 1
    assert len([call for call in calls if call[0] == "control_after"]) == 1
    for phase in {call[0] for call in readout_calls}:
        assert {call[1] for call in readout_calls if call[0] == phase} == {"real", "oracle", "shuffled", "empty"}
    for name, tensor in module.state_dict().items():
        torch.testing.assert_close(tensor, before[name])
