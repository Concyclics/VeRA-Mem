"""Sampling and intervention tests independent of expensive model execution."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vera_mem import interface_data as data
from vera_mem import interface_run as run
from vera_mem import interface_training
from vera_mem.interface_variants import InterfaceVectorVeRA


def records(size=128):
    qids = [t.id for t in data.get_templates("train", "query")]
    sids = [t.id for t in data.get_templates("train", "support")]
    return [dict(id=p.id, entity=p.entity, relation=p.relation, a=p.answer_a, b=p.answer_b,
                 questions=[data.render_question(p, q) for q in qids],
                 supports=[[data.render_support(p, s, world=w) for s in sids] for w in ("A", "B")])
            for p in data.datasets(train_size=size)["train"]]


def packet(size=128):
    rows = records(size)
    last = torch.arange(size * 2 * 4 * 5, dtype=torch.float32).reshape(size, 2, 4, 5)
    q = torch.arange(size * 8 * 5, dtype=torch.float32).reshape(size, 8, 5)
    return dict(protocol=run.PROTOCOL, model_revision=run.REVISION, split="train", task="extended",
                rows=rows, q=q, last=last, pool=last + 100000.)


def test_actual_extended_sample_retains_both_values_in_real_a_bank_and_all_relations():
    rows = records()
    sampler = run.InterfaceSampler(rows, seed=42)
    for _ in range(32):
        sample = sampler.next(8, 4)
        assert len(sample["episode"]) == len(set(sample["episode"])) == 72
        assert [sample["episode"][c] for c in sample["columns"]] == sample["targets"]
        for target in sample["targets"]:
            fact = rows[target]
            for world in ("a", "b"):
                candidates = [j for j in sample["episode"] if rows[j]["a"] == fact[world]
                              and rows[j]["entity"] != fact["entity"]]
                assert candidates, f"No original A-bank negative for {target} world {world}"
            same_entity = {i for i, r in enumerate(rows) if r["entity"] == fact["entity"]}
            assert len(same_entity) == len(data.RELATIONS)
            assert same_entity <= set(sample["episode"])


def test_sampler_rejects_b_only_virtual_negative():
    # The old two-entity A/B duplicate group did not place B in the original
    # bank. Indexing both worlds as though they were present must not pass.
    rows = [dict(entity=f"E{i}", a="same A", b="only in B") for i in range(8)]
    with pytest.raises(ValueError, match="real-bank"):
        run.InterfaceSampler(rows, seed=42).next(8, 4)


def test_five_arms_have_identical_sampling_and_replaying_steps_resumes_exactly():
    rows = records()
    samplers = [run.InterfaceSampler(rows, seed=43) for _ in range(5)]
    schedule = []
    for _ in range(24):
        current = [s.next(8, 4) for s in samplers]
        assert all(s == current[0] for s in current)
        schedule.append(current[0])
    resumed = run.InterfaceSampler(rows, seed=43)
    for _ in range(17):
        resumed.next(8, 4)
    assert resumed.next(8, 4) == schedule[17]


def test_batch_replaces_only_target_with_separate_key_value_features_and_private_context():
    p = packet()
    sample = run.InterfaceSampler(p["rows"], seed=42).next(8, 4)
    before = {k: p[k].clone() for k in ("q", "last", "pool")}
    batch = run.make_batch(p, sample, p["pool"], .05)
    targets, episode, columns = (sample[k] for k in ("targets", "episode", "columns"))
    assert batch["columns"] == columns
    torch.testing.assert_close(batch["q_views"], p["q"][episode])
    for row, (target, column) in enumerate(zip(targets, columns)):
        support = sample["sviews"][column]
        paraphrase = (support + 1) % 4
        assert batch["questions"][row] == p["rows"][target]["questions"][sample["qviews"][row]]
        assert p["rows"][target]["id"] not in batch["questions"][row]
        for w in ("a", "b"):
            assert p["rows"][target][w] not in batch["questions"][row]
        for banks, feature in ((batch["key_banks"], "last"), (batch["value_banks"], "pool")):
            for world in (1, 2):
                changed = (banks[world][row] != banks[0][row]).any(-1).nonzero().flatten().tolist()
                assert changed == [column]
            for entry, identity in enumerate(episode):
                torch.testing.assert_close(banks[0][row, entry], p[feature][identity, 0, sample["sviews"][entry]])
            torch.testing.assert_close(banks[1][row, column], p[feature][target, 1, support])
            torch.testing.assert_close(banks[2][row, column], p[feature][target, 0, paraphrase])
        for world, w, view in ((0, 0, support), (1, 1, support), (2, 0, paraphrase)):
            assert batch["contexts"][world][row] == p["rows"][target]["supports"][w][view]
            assert batch["answers"][world][row] == p["rows"][target]["b" if w else "a"]
    changed_rows = (batch["key_views"][1] != batch["key_views"][0]).any(-1).any(-1).nonzero().flatten().tolist()
    assert set(changed_rows) == set(columns)
    for field, original in before.items():
        torch.testing.assert_close(p[field], original, rtol=0, atol=0)


def tiny_module():
    m = InterfaceVectorVeRA(5, 5, rank=3, key_dim=3, top_k=3, temperature=.7, seed=6)
    rng = torch.Generator().manual_seed(71)
    m.fit_statistics(torch.randn(23, 5, generator=rng), torch.randn(23, 5, generator=rng))
    with torch.no_grad():
        m.b.fill_(.5)
    return m


def test_training_forks_share_optimizer_start_and_sample_schedule(tmp_path, monkeypatch):
    # Isolate runner scheduling from neural objectives, already tested with real
    # CPU forward/backward in test_interface_training.py. Fake loss has real
    # optimizer state and all parameter groups receive a deterministic gradient.
    calls = []
    def step(backend, module, batch, method, scale, generator, on_policy=False):
        calls.append(dict(method=method, on_policy=on_policy, query=batch["q_views"].clone(),
                          key=batch["key_banks"][0].clone(), columns=list(batch["columns"])))
        loss = sum(value.square().mean() for value in module.parameters())
        return loss, dict(total_loss=float(loss.detach()), target_tokens=1), []
    monkeypatch.setattr(interface_training, "interface_step", step)
    backend = SimpleNamespace(device=torch.device("cpu"))
    warm_dir = tmp_path / "warm"
    warm_dir.mkdir()
    common = dict(seed=42, scale=6., key_consistency=.05)
    args = SimpleNamespace(**common, run_dir=warm_dir, updates=3, resume_optimizer=False, method="base")
    run.train_model(backend, tiny_module(), packet(), {}, args)
    checkpoint = torch.load(warm_dir / "last.pt", weights_only=True)
    assert checkpoint["step"] == 3 and checkpoint["optimizer"]["state"]
    schedules, statuses = [], []
    for method in interface_training.METHODS:
        output = tmp_path / method
        output.mkdir()
        module = tiny_module()
        module.load_state_dict(checkpoint["module"])
        args = SimpleNamespace(**common, run_dir=output, updates=4, resume_optimizer=True, method=method)
        start = len(calls)
        statuses.append(run.train_model(backend, module, packet(), deepcopy(checkpoint), args))
        schedules.append(calls[start:])
        saved = torch.load(output / "last.pt", weights_only=True)
        assert saved["step"] == 7
        assert {int(state["step"]) for state in saved["optimizer"]["state"].values()} == {7}
    assert len({status["schedule_sha256"] for status in statuses}) == 1
    assert all(status["start_step"] == 3 and status["updates"] == 4 for status in statuses)
    for step_index in range(4):
        for schedule in schedules[1:]:
            assert schedule[step_index]["columns"] == schedules[0][step_index]["columns"]
            torch.testing.assert_close(schedule[step_index]["query"], schedules[0][step_index]["query"])
            torch.testing.assert_close(schedule[step_index]["key"], schedules[0][step_index]["key"])
    assert [s["on_policy"] for s in schedules[-1]] == [False, False, False, True]
    assert not any(s["on_policy"] for schedule in schedules[:-1] for s in schedule)


def test_interface_checkpoint_restores_statistics_architecture_and_exact_weights(tmp_path):
    module = tiny_module()
    path = tmp_path / "module.pt"
    torch.save(dict(protocol=run.PROTOCOL, architecture=module.configuration(), module=module.state_dict()), path)
    loaded, _ = run.load_module(path, "cpu")
    assert loaded.configuration() == module.configuration()
    assert loaded._statistics_ready
    for key, value in module.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("kwargs", [{"writer": "masked_mean"}, {"learned_b": True}, {"mlp": 8}])
def test_explicit_architecture_changes_cannot_be_silently_ignored(tmp_path, kwargs):
    module = tiny_module()
    path = tmp_path / "module.pt"
    torch.save(dict(protocol=run.PROTOCOL, architecture=module.configuration(), module=module.state_dict()), path)
    with pytest.raises(ValueError, match="conflicts"):
        run.load_module(path, "cpu", **kwargs)


@pytest.mark.parametrize("stage", ["train", "calibrate"])
def test_confirmation_cache_is_rejected_before_training_or_teacher_calibration(stage):
    class ForbiddenBackend:
        def __getattr__(self, name):
            raise AssertionError("A confirmation packet reached model execution")
    cached = {"split": "confirm"}
    with pytest.raises(ValueError, match="train|Training"):
        if stage == "train":
            run.train_model(ForbiddenBackend(), None, cached, {}, None)
        else:
            run.calibrate(ForbiddenBackend(), cached, None)
