"""No KD expansion or confirmation can bypass its fixed evidence gates."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("plan_interface_kd_under_test", SCRIPTS / "plan_interface_kd.py")
kd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kd)
from plan_interface_followup import REMOTE, STAGE3_SCALE, step2_plans, training_plans


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def cli_config(arguments):
    result, i = {}, 0
    while i < len(arguments):
        key = arguments[i][2:].replace("-", "_")
        if i + 1 < len(arguments) and not arguments[i + 1].startswith("--"):
            value = arguments[i + 1]
            if key in {"seed", "updates", "eval_size"}:
                value = int(value)
            elif key in {"scale", "key_consistency"}:
                value = float(value)
            result[key] = value
            i += 2
        else:
            result[key] = True
            i += 1
    return result


def add_job(root, suite, name, configuration, result, **extra):
    path = root / suite / "suite.json"
    metadata = kd.read_json(path) if path.exists() else dict(protocol="interface-experiment-v1", jobs=[])
    metadata["jobs"].append(dict(name=name, status="complete", exit_code=0))
    write(path, metadata)
    directory = root / suite / name
    manifest = dict(protocol="memory-interface-v1", complete=True, backbone_unchanged=True,
                    configuration=configuration, result=result, **extra)
    write(directory / "manifest.json", manifest)
    return directory


@pytest.fixture
def selection():
    return dict(tag="20261006", remote_workspace=REMOTE, model=REMOTE + "/model", selected_writer="last",
                original_checkpoint=REMOTE + "/old/base/last.pt", original_checkpoint_sha256="a" * 64,
                evidence=[dict(writer="last", seed=seed, checkpoint=f"{REMOTE}/runs/writer/last_{seed}/last.pt",
                               checkpoint_sha256="e" * 64) for seed in (42, 43, 44)])


def confirmation_fixture(root, selection, queues=3, candidates=None):
    candidates = candidates or {seed: (36, 94) for seed in (42, 43, 44)}
    assignments = dict(selected_case_ids=[str(i) for i in range(128)],
                       assignments=[dict(id=str(i), entity=f"e{i}", relation="memory", q=i % 2 + 1, s=(i+1) % 2 + 1) for i in range(128)])
    directories = {}
    for suite, jobs in step2_plans(selection, queues).items():
        for job in jobs:
            baseline = job["name"] == "baseline_confirm"
            cfg = cli_config(job["arguments"])
            cfg.setdefault("task", "classic")
            cfg.setdefault("seed", 42)
            cfg["max_cases"] = None
            hc, cc = (10, 100) if baseline else candidates[cfg["seed"]]
            metrics = dict(protocol="interface-cpu-vdb-eval-v1", complete=True, split="confirm", evaluated_facts=128,
                           bank_records=128, max_new_tokens=32, frozen_online=True, online_gradient_steps=0,
                           shared_weights_before="b" * 64, shared_weights_after="b" * 64,
                           phases={p: dict(count=128, bank_records=128, methods={"real": dict(count=128,
                                      both_correct=cc if p == "CC" else hc, paired_switch_em=(cc if p == "CC" else hc) / 128)})
                                   for p in ("CC", "CH", "HC", "HH")})
            directory = add_job(root, suite, job["name"], cfg, metrics, checkpoint_sha256="a" * 64 if baseline else "e" * 64,
                                cache_sha256="c" * 64)
            write(directory / "metrics.json", metrics)
            write(directory / "assignments.json", assignments)
            directories[job["name"]] = directory
    return directories


def save_checkpoint(path, step, seed, method="base", missing_adam=False):
    g = torch.Generator().manual_seed(seed)
    names = ["Wv.weight", "b", "Wq.weight", "Wk.weight"]
    shapes = [(3, 5), (6,), (4, 5), (4, 5)]
    params = [torch.nn.Parameter(torch.randn(shape, generator=g)) for shape in shapes]
    optimizer = torch.optim.Adam([dict(params=params[:1], lr=1e-4), dict(params=params[1:2], lr=.005),
                                  dict(params=params[2:], lr=1e-5)])
    sum(p.square().sum() for p in params).backward()
    optimizer.step()
    optimizer_state = optimizer.state_dict()
    for value in optimizer_state["state"].values():
        value["step"].fill_(step)
    checkpoint = dict(protocol="memory-interface-v1", step=step, seed=seed, method=method, key_consistency=.05,
                      architecture=dict(in_features=5, out_features=6, rank=3, key_dim=4,
                                        writer_mode="last_token", train_B=False, value_mlp_hidden=0),
                      module={name: p.detach() for name, p in zip(names, params)})
    if not missing_adam:
        checkpoint["optimizer"] = optimizer_state
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def log_rows(seed, start, updates):
    return [dict(step=step, margin_scale=STAGE3_SCALE,
                 sample=dict(targets=list(range(8)), seed=seed, qviews=[step % 4] * 8))
            for step in range(start + 1, start + updates + 1)]


def schedule_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row["sample"], sort_keys=True).encode())
    return digest.hexdigest()


def save_logs(directory, rows):
    (directory / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def warm_fixture(root, selection, queues=3):
    add_job(root, "interface_extended_prepare_20261006", "prepare", dict(stage="prepare", task="extended"),
            dict(splits=dict(train=4096, dev=256, confirm=256), entity_disjoint=True))
    directories = {}
    for suite, jobs in training_plans(selection, queues).items():
        for job in jobs:
            cfg = cli_config(job["arguments"])
            cfg["resume_optimizer"] = False
            rows = log_rows(cfg["seed"], 0, 512)
            status = dict(complete=True, updates=512, start_step=0, schedule_sha256=schedule_digest(rows))
            directory = add_job(root, suite, job["name"], cfg, status, checkpoint_sha256="e" * 64, cache_sha256="f" * 64)
            write(directory / "training_status.json", status)
            save_logs(directory, rows)
            save_checkpoint(directory / "last.pt", 512, cfg["seed"])
            directories[job["name"]] = directory
    return directories


def gate(passed):
    return dict(passed=passed, seeds=[42, 43, 44] if passed else [42], updates=512 if passed else 256)


def continuations_fixture(root, selection, gate_value, queues=3):
    generated, _ = kd.plans(selection, gate_value, root, queues, STAGE3_SCALE, "train")
    directories = {}
    for suite, jobs in generated.items():
        for job in jobs:
            cfg = cli_config(job["arguments"])
            source = root / Path(cfg["checkpoint"]).relative_to(Path(REMOTE) / "runs")
            rows = log_rows(cfg["seed"], 512, gate_value["updates"])
            status = dict(complete=True, updates=gate_value["updates"], start_step=512, schedule_sha256=schedule_digest(rows))
            directory = add_job(root, suite, job["name"], cfg, status, checkpoint_sha256=kd.sha(source), cache_sha256="f" * 64)
            write(directory / "training_status.json", status)
            save_logs(directory, rows)
            save_checkpoint(directory / "last.pt", 512 + gate_value["updates"], cfg["seed"], cfg["method"])
            directories[job["name"]] = directory
    return directories


def mutate_result(directory, filename, mutate):
    path = directory / filename
    data = kd.read_json(path)
    mutate(data)
    write(path, data)
    manifest = kd.read_json(directory / "manifest.json")
    manifest["result"] = data
    write(directory / "manifest.json", manifest)


def test_gate_requires_all_three_seeds_to_exceed_both_paired_thresholds(tmp_path, selection):
    directories = confirmation_fixture(tmp_path, selection)
    result = kd.confirmation_gate(selection, tmp_path, 3)
    assert result["passed"] and result["updates"] == 512 and result["seeds"] == [42, 43, 44]
    assert all(d["hc_delta"] == pytest.approx(26 / 128) and d["cc_delta"] == pytest.approx(-6 / 128)
               and d["passed"] for d in result["decisions"])
    assert len(result["evidence"]) == 4
    def lower_hc(metrics):
        metrics["phases"]["HC"]["methods"]["real"].update(both_correct=35, paired_switch_em=35 / 128)
    mutate_result(directories["last_44_confirm"], "metrics.json", lower_hc)
    failed = kd.confirmation_gate(selection, tmp_path, 3)
    assert not failed["passed"] and failed["seeds"] == [42] and failed["updates"] == 256
    assert [d["passed"] for d in failed["decisions"]] == [True, True, False]


def test_cc_regression_alone_fails_gate_despite_large_hc_gain(tmp_path, selection):
    confirmation_fixture(tmp_path, selection, candidates={42: (100, 100), 43: (100, 93), 44: (100, 100)})
    result = kd.confirmation_gate(selection, tmp_path, 3)
    assert not result["passed"] and not result["decisions"][1]["passed"]


@pytest.mark.parametrize("failure", ["denominator", "different_assignments", "duplicate_ids", "partial", "wrong_checkpoint",
                                     "wrong_seed", "wrong_model", "wrong_budget", "mutated_weights", "dev"])
def test_confirmation_rejects_unpaired_partial_or_wrong_model_evidence(tmp_path, selection, failure):
    directory = confirmation_fixture(tmp_path, selection)["last_43_confirm"]
    if failure in {"different_assignments", "duplicate_ids"}:
        assignment = kd.read_json(directory / "assignments.json")
        if failure == "different_assignments":
            assignment["assignments"][0]["s"] += 1
        else:
            assignment["selected_case_ids"][0] = assignment["selected_case_ids"][1]
        write(directory / "assignments.json", assignment)
    elif failure in {"partial", "wrong_checkpoint", "wrong_seed", "wrong_model"}:
        manifest = kd.read_json(directory / "manifest.json")
        if failure == "partial": manifest["complete"] = False
        elif failure == "wrong_checkpoint": manifest["checkpoint_sha256"] = "d" * 64
        elif failure == "wrong_seed": manifest["configuration"]["seed"] = 42
        else: manifest["configuration"]["model"] = "/another/model"
        write(directory / "manifest.json", manifest)
    else:
        def mutate(metrics):
            if failure == "denominator": metrics["phases"]["HC"]["methods"]["real"]["count"] = 32
            elif failure == "wrong_budget": metrics["max_new_tokens"] = 16
            elif failure == "mutated_weights": metrics["shared_weights_after"] = "d" * 64
            else: metrics["split"] = "dev"
        mutate_result(directory, "metrics.json", mutate)
    with pytest.raises(ValueError):
        kd.confirmation_gate(selection, tmp_path, 3)


@pytest.mark.parametrize("passed,queues", [(False, 2), (False, 3), (True, 3)])
def test_fixed_gate_budget_forks_five_methods_from_same_actual_warm_and_adam(tmp_path, selection, passed, queues):
    directories = warm_fixture(tmp_path, selection, queues)
    generated, evidence = kd.plans(selection, gate(passed), tmp_path, queues, STAGE3_SCALE, "train")
    jobs = [j for queue in generated.values() for j in queue]
    assert len(jobs) == (15 if passed else 5)
    per_seed = {}
    for job in jobs:
        cfg = cli_config(job["arguments"])
        assert cfg["updates"] == (512 if passed else 256) and cfg["resume_optimizer"] is True
        assert cfg["stage"] == "train" and cfg["task"] == "extended" and cfg["key_consistency"] == .05
        assert cfg["scale"] == STAGE3_SCALE
        per_seed.setdefault(cfg["seed"], []).append(cfg)
    assert set(per_seed) == ({42, 43, 44} if passed else {42})
    for seed, configs in per_seed.items():
        assert {cfg["method"] for cfg in configs} == set(kd.METHODS)
        assert len({cfg["checkpoint"] for cfg in configs}) == 1
        assert f"decoupled_{seed}/last.pt" in configs[0]["checkpoint"]
        actual = evidence["warm_checkpoints"][str(seed)]
        assert actual["optimizer_step"] == 512
        assert actual["checkpoint_sha256"] == kd.sha(directories[f"decoupled_{seed}"] / "last.pt")
        assert len(actual["optimizer_sha256"]) == 64
    assert evidence["continuations"] == []


@pytest.mark.parametrize("failure", ["missing_adam", "bad_adam_step", "bad_warm_step", "different_schedule", "wrong_initialization", "incomplete"])
def test_kd_training_plan_rejects_invalid_warm_adam_or_control_evidence(tmp_path, selection, failure):
    directory = warm_fixture(tmp_path, selection)["decoupled_42"]
    if failure in {"missing_adam", "bad_adam_step", "bad_warm_step"}:
        path = directory / "last.pt"
        checkpoint = torch.load(path, weights_only=True)
        if failure == "missing_adam": del checkpoint["optimizer"]
        elif failure == "bad_warm_step": checkpoint["step"] = 256
        else: next(iter(checkpoint["optimizer"]["state"].values()))["step"].fill_(511)
        torch.save(checkpoint, path)
    elif failure == "different_schedule":
        rows = log_rows(42, 0, 512)
        rows[0]["sample"]["qviews"][0] = 999
        save_logs(directory, rows)
        mutate_result(directory, "training_status.json", lambda s: s.update(schedule_sha256=schedule_digest(rows)))
    else:
        manifest = kd.read_json(directory / "manifest.json")
        if failure == "incomplete": manifest["complete"] = False
        else: manifest["checkpoint_sha256"] = "1" * 64
        write(directory / "manifest.json", manifest)
    with pytest.raises(ValueError):
        kd.plans(selection, gate(False), tmp_path, 3, STAGE3_SCALE, "train")


@pytest.mark.parametrize("passed", [False, True])
def test_all_final_checkpoints_and_adam_states_are_required_before_confirmation(tmp_path, selection, passed):
    warm_fixture(tmp_path, selection)
    directories = continuations_fixture(tmp_path, selection, gate(passed))
    generated, evidence = kd.plans(selection, gate(passed), tmp_path, 3, STAGE3_SCALE, "confirm")
    jobs = [j for queue in generated.values() for j in queue]
    assert len(jobs) == len(evidence["continuations"]) == (15 if passed else 5)
    assert {item["optimizer_step"] for item in evidence["continuations"]} == {1024 if passed else 768}
    for seed in ([42, 43, 44] if passed else [42]):
        members = [item for item in evidence["continuations"] if item["seed"] == seed]
        assert len(members) == 5
        assert len({item["warm_optimizer_sha256"] for item in members}) == 1
        assert len({item["warm_checkpoint_sha256"] for item in members}) == 1
        assert len({item["schedule_sha256"] for item in members}) == 1
    for entry in jobs:
        cfg = cli_config(entry["arguments"])
        assert cfg["stage"] == "eval" and cfg["task"] == "extended" and cfg["eval_size"] == 256
        assert cfg["cache"].endswith("/prepare/confirm.pt")
        assert "resume_optimizer" not in cfg
    broken = directories["on_policy_42"]
    manifest = kd.read_json(broken / "manifest.json")
    manifest["complete"] = False
    write(broken / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Incomplete"):
        kd.plans(selection, gate(passed), tmp_path, 3, STAGE3_SCALE, "confirm")


@pytest.mark.parametrize("failure", ["parent_hash", "no_resume", "different_schedule", "wrong_final_adam", "wrong_start", "wrong_task"])
def test_continuation_confirmation_refuses_wrong_initialization_budget_or_schedule(tmp_path, selection, failure):
    warm_fixture(tmp_path, selection)
    directory = continuations_fixture(tmp_path, selection, gate(False))["hidden_42"]
    if failure == "wrong_final_adam":
        path = directory / "last.pt"
        checkpoint = torch.load(path, weights_only=True)
        next(iter(checkpoint["optimizer"]["state"].values()))["step"].fill_(256)
        torch.save(checkpoint, path)
    elif failure == "different_schedule":
        rows = log_rows(42, 512, 256)
        rows[0]["sample"]["qviews"][0] = 888
        save_logs(directory, rows)
        mutate_result(directory, "training_status.json", lambda s: s.update(schedule_sha256=schedule_digest(rows)))
    elif failure == "wrong_start":
        mutate_result(directory, "training_status.json", lambda s: s.update(start_step=0))
    else:
        manifest = kd.read_json(directory / "manifest.json")
        if failure == "parent_hash": manifest["checkpoint_sha256"] = "8" * 64
        elif failure == "no_resume": manifest["configuration"]["resume_optimizer"] = False
        else: manifest["configuration"]["task"] = "classic"
        write(directory / "manifest.json", manifest)
    with pytest.raises(ValueError):
        kd.plans(selection, gate(False), tmp_path, 3, STAGE3_SCALE, "confirm")


def test_failed_gate_budget_cannot_be_overridden_to_three_seeds(tmp_path, selection):
    bad = gate(False)
    bad["seeds"] = [42, 43, 44]
    with pytest.raises(ValueError, match="Gate budget"):
        kd.plans(selection, bad, tmp_path, 3, STAGE3_SCALE, "train")
