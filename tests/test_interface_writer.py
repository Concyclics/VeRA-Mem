"""Value fitting has fixed canonical targets and cannot update address/readout."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from vera_mem import interface_writer as writer
from vera_mem.interface_variants import InterfaceVectorVeRA
from test_interface_variants import make_baseline, migrate


def make_packet():
    g = torch.Generator().manual_seed(67)
    n, s = 6, 3
    return dict(protocol=writer.PROTOCOL, model_revision=writer.REVISION, task="classic", split="train",
                writer_prefix=writer.PREFIX, qids=["train_query_00", "train_query_01"],
                sids=[f"train_support_{v:02d}" for v in range(s)],
                rows=[dict(id=f"id{i}", entity=f"entity{i}", questions=[f"q{i}v{v}" for v in range(2)],
                           supports=[[f"s{i}w{w}v{v}" for v in range(s)] for w in range(2)]) for i in range(n)],
                q=torch.randn(n, 2, 5, generator=g), last=torch.randn(n, 2, s, 5, generator=g),
                pool=torch.randn(n, 2, s, 5, generator=g))


def make_args(tmp_path, name="run", updates=6, batch_size=12, seed=41):
    checkpoint = tmp_path / "teacher.pt"
    if not checkpoint.exists():
        torch.save({"module": make_baseline().state_dict()}, checkpoint)
    return SimpleNamespace(checkpoint=checkpoint, run_dir=tmp_path / name, updates=updates,
                           writer_batch_size=batch_size, seed=seed)


@pytest.mark.parametrize("mode,mlp", [("last_token", 0), ("masked_mean", 0), ("last_token", 7)])
def test_writer_uses_each_world_canonical_target_and_only_updates_value_parameters(tmp_path, monkeypatch, mode, mlp):
    packet, args = make_packet(), make_args(tmp_path)
    baseline = make_baseline()
    module = migrate(baseline, writer_mode=mode, value_mlp_hidden=mlp, train_B=True)
    initial = deepcopy(module)
    if mode == "masked_mean":
        initial.fit_value_statistics(packet["pool"].reshape(-1, 5))
    before = {name: value.clone() for name, value in initial.state_dict().items()}
    constructed_teachers = []
    original_teacher_class = writer.BatchedStableVectorVeRA
    class ObservedTeacher(original_teacher_class):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.observed_inputs = []
            constructed_teachers.append(self)
        def encode_value(self, inputs):
            assert not torch.is_grad_enabled() and all(not p.requires_grad for p in self.parameters())
            self.observed_inputs.append(inputs.clone())
            result = super().encode_value(inputs)
            assert not result.requires_grad
            return result
    monkeypatch.setattr(writer, "BatchedStableVectorVeRA", ObservedTeacher)
    result = writer.train_writer(SimpleNamespace(device="cpu"), module, packet, args)
    log = [json.loads(line) for line in (args.run_dir / "training.jsonl").read_text().splitlines()]
    first = log[0]
    ids, worlds, views = [torch.tensor(first[name]) for name in ("facts", "worlds", "views")]
    with torch.no_grad():
        value_features = packet["pool" if mode == "masked_mean" else "last"]
        actual_input = value_features[ids, worlds, views]
        canonical_target = baseline.encode_value(packet["last"][ids, worlds, 0])
        expected_loss = F.mse_loss(initial.encode_value(actual_input), canonical_target)
        wrong_view_loss = F.mse_loss(initial.encode_value(actual_input), baseline.encode_value(packet["last"][ids, worlds, views]))
    assert first["loss"] == pytest.approx(expected_loss.item(), abs=1e-7)
    assert abs(first["loss"] - wrong_view_loss.item()) > 1e-4
    assert first["target_rms"] == pytest.approx(1., abs=1e-3)
    assert len(constructed_teachers) == 1
    teacher = constructed_teachers[0]
    torch.testing.assert_close(torch.cat(teacher.observed_inputs), packet["last"][:, :, 0].reshape(-1, 5))
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert not result["targets_require_grad"] and result["teacher_unchanged"]
    assert result["target_shape"] == [6, 2, 3]
    for name, value in module.state_dict().items():
        if name.startswith("Wv.") or name.startswith("value_mlp."):
            assert not torch.equal(value, before[name]), name
        else:
            assert torch.equal(value, before[name]), name
    assert module.B.grad is None and module.A.grad is None
    assert all(p.grad is None and not p.requires_grad for p in module.parameters())
    assert not module.training
    assert result["trainable_names"] == ["Wv.weight"] + (["value_mlp.0.weight", "value_mlp.2.weight"] if mlp else [])
    assert result["trainable_count"] == 15 + ((5 * mlp + mlp * 3) if mlp else 0)
    if mode == "masked_mean":
        torch.testing.assert_close(module.value_center, module._normalized_input(packet["pool"].reshape(-1, 5)).mean(0))
    assert [entry["step"] for entry in log] == list(range(1, args.updates + 1))
    assert result["training_examples"] == args.updates * args.writer_batch_size


def test_schedule_can_be_recomputed_and_all_variants_share_exact_sampling(tmp_path):
    packet = make_packet()
    schedules = []
    for i, options in enumerate(({}, {"writer_mode": "masked_mean"}, {"value_mlp_hidden": 7})):
        args = make_args(tmp_path, f"arm{i}", updates=4)
        result = writer.train_writer(SimpleNamespace(device="cpu"), migrate(**options), packet, args)
        rows = [json.loads(line) for line in (args.run_dir / "training.jsonl").read_text().splitlines()]
        g, digest = torch.Generator().manual_seed(args.seed), hashlib.sha256()
        for step, row in enumerate(rows, 1):
            expected = dict(step=step, facts=torch.randint(6, (12,), generator=g).tolist(),
                            worlds=torch.randint(2, (12,), generator=g).tolist(),
                            views=torch.randint(3, (12,), generator=g).tolist())
            assert all(row[k] == v for k, v in expected.items())
            digest.update(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode())
            assert row["sample_schedule_sha256"] == digest.hexdigest()
        assert result["sample_schedule_sha256"] == digest.hexdigest()
        assert {w for row in rows for w in row["worlds"]} == {0, 1}
        assert {v for row in rows for v in row["views"]} == {0, 1, 2}
        schedules.append(result["sample_schedule_sha256"])
    assert len(set(schedules)) == 1


def test_fixed_last_checkpoint_roundtrip_contains_budget_adam_and_rng_not_best_selection(tmp_path):
    args = make_args(tmp_path, updates=5)
    module = migrate(writer_mode="masked_mean")
    result = writer.train_writer(SimpleNamespace(device="cpu"), module, make_packet(), args)
    checkpoint = torch.load(args.run_dir / "last.pt", weights_only=True)
    assert checkpoint["protocol"] == "memory-interface-v1"
    assert checkpoint["step"] == 5 and checkpoint["configuration"]["updates"] == 5
    assert checkpoint["configuration"]["checkpoint_selection"] == "fixed final update"
    assert checkpoint["seed"] == args.seed
    assert checkpoint["sample_schedule_sha256"] == result["sample_schedule_sha256"]
    assert all(int(state["step"]) == 5 for state in checkpoint["optimizer"]["state"].values())
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == 1e-4
    assert checkpoint["sampler_rng_state"].dtype == torch.uint8
    recovered = InterfaceVectorVeRA(**checkpoint["architecture"])
    recovered.load_state_dict(checkpoint["module"])
    assert recovered._value_statistics_ready and recovered._statistics_ready
    for name, value in module.state_dict().items():
        assert torch.equal(recovered.state_dict()[name], value)
    assert json.loads((args.run_dir / "training_status.json").read_text()) == result
    with pytest.raises(FileExistsError, match="new run directory"):
        writer.train_writer(SimpleNamespace(device="cpu"), migrate(), make_packet(), args)


@pytest.mark.parametrize("alter", [
    lambda p: p.update(split="dev"), lambda p: p.update(split="confirm"), lambda p: p.update(task="extended"),
    lambda p: p.update(protocol="old-v1"), lambda p: p.update(model_revision="unknown"),
    lambda p: p.update(writer_prefix=""), lambda p: p.update(last=p["last"][:, 0]),
    lambda p: p.update(pool=p["pool"][:, :, :2]), lambda p: p.update(q=p["q"][:, :, :4]),
    lambda p: p.update(sids=list(reversed(p["sids"]))),
    lambda p: p["last"].__setitem__((0, 0, 0, 0), float("nan")),
    lambda p: p["rows"][1].update(entity=p["rows"][0]["entity"]),
])
def test_invalid_split_prefix_shape_and_finiteness_rejected_before_state_mutation(tmp_path, alter):
    packet = make_packet()
    alter(packet)
    module = migrate()
    before = {name: value.clone() for name, value in module.state_dict().items()}
    args = make_args(tmp_path, updates=1)
    with pytest.raises(ValueError):
        writer.train_writer(SimpleNamespace(device="cpu"), module, packet, args)
    assert not args.run_dir.exists()
    assert all(torch.equal(value, before[name]) for name, value in module.state_dict().items())


def test_canonical_teacher_cannot_silently_use_an_interface_checkpoint(tmp_path):
    args = make_args(tmp_path, updates=1)
    torch.save(migrate().checkpoint_payload(), args.checkpoint)
    with pytest.raises(ValueError, match="original pre-interface"):
        writer.train_writer(SimpleNamespace(device="cpu"), migrate(), make_packet(), args)
