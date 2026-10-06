"""Reconstruction-specific launcher contracts; shared scheduler tests stay shared."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_reconstruction_suite as serial
import run_reconstruction_eval_suite as parallel
import summarize_reconstruction as summary

GPU = "GPU-00000000-1111-2222-3333-444444444444"


def serial_plan():
    return [dict(name="prepare", entry="reconstruction", arguments=["--stage", "prepare", "--model", "/sealed/model"])]


def make_repository(path):
    repo = path / "VeRA-Mem"
    for folder in ("src", "scripts", "tests", "configs", "docs"):
        (repo / folder).mkdir(parents=True)
        (repo / folder / "marker.txt").write_text(folder)
    for name in ("README.md", "pyproject.toml"):
        (repo / name).write_text(name)
    (path / "env_deps").mkdir()
    return repo


@pytest.fixture
def serial_workspace(tmp_path, monkeypatch):
    make_repository(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(serial_plan()))
    monkeypatch.setattr(serial, "gpu_memory_used", lambda _gpu: 0)
    monkeypatch.setattr(serial.shutil, "disk_usage", lambda _path: SimpleNamespace(free=20 * 1024**3))
    args = ["--workspace", str(tmp_path), "--shared-root", str(tmp_path), "--gpu", GPU,
            "--name", "reconstruction_test", "--plan", str(plan)]
    return tmp_path, args


@pytest.mark.parametrize("mutation", [
    lambda p: p[0].update(entry="coldstart"),
    lambda p: p[0].update(name="../escape"),
    lambda p: p[0].update(name=12),
    lambda p: p.append(copy.deepcopy(p[0])),
    lambda p: p[0].update(unexpected=True),
    lambda p: p[0]["arguments"].extend(["--run-dir", "/tmp/override"]),
    lambda p: p[0]["arguments"].append("--run-dir=/tmp/override"),
    lambda p: p[0]["arguments"].extend(["--stage", "train"]),
    lambda p: p[0]["arguments"].__setitem__(0, "--stage=prepare"),
    lambda p: p[0]["arguments"].__setitem__(1, "download"),
])
def test_serial_plan_rejects_wrong_scope_and_output_ownership(mutation):
    plan = serial_plan(); mutation(plan)
    with pytest.raises(ValueError):
        serial.validate_plan(plan)


def test_serial_launch_uses_frozen_entry_and_accepts_only_matching_child(serial_workspace, monkeypatch):
    root, args = serial_workspace
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        assert command[1:4] == ["-u", "-m", "vera_mem.reconstruction_run"]
        assert kwargs["cwd"] == root / "runs/reconstruction_test/source"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == GPU
        assert str(kwargs["cwd"] / "src") in kwargs["env"]["PYTHONPATH"]
        output = Path(command[-1]); output.mkdir()
        (output / "manifest.json").write_text(json.dumps(dict(protocol=serial.PROTOCOL, stage="prepare", complete=True)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(serial.subprocess, "run", fake_run)
    assert serial.main(args) == 0 and len(calls) == 1
    manifest = json.loads((root / "runs/reconstruction_test/suite.json").read_text())
    assert manifest["complete"] and manifest["source_unchanged"]
    assert manifest["jobs"][0]["manifest_sha256"]
    with pytest.raises(FileExistsError):
        serial.main(args)


@pytest.mark.parametrize("corruption", ["protocol", "stage", "incomplete", "source"])
def test_serial_zero_exit_cannot_validate_wrong_child_or_mutated_source(serial_workspace, monkeypatch, corruption):
    root, args = serial_workspace
    def fake_run(command, **kwargs):
        output = Path(command[-1]); output.mkdir()
        child = dict(protocol=serial.PROTOCOL, stage="prepare", complete=True)
        if corruption == "protocol": child["protocol"] = "dictionary-coldstart-v1"
        if corruption == "stage": child["stage"] = "train"
        if corruption == "incomplete": child["complete"] = False
        if corruption == "source": (kwargs["cwd"] / "src/marker.txt").write_text("mutated")
        (output / "manifest.json").write_text(json.dumps(child))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(serial.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        serial.main(args)
    manifest = json.loads((root / "runs/reconstruction_test/suite.json").read_text())
    assert not manifest["complete"] and manifest["jobs"][0]["status"] == "failed"


@pytest.fixture
def eval_inputs(tmp_path):
    model = tmp_path / "models/pinned"; model.mkdir(parents=True)
    (model.parent / "manifest.json").write_text('{"revision":"pinned"}')
    cache = tmp_path / "known.pt"; cache.write_bytes(b"sealed-cache")
    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"sealed-checkpoint")
    arguments = ["--stage", "eval", "--model", str(model), "--cache", str(cache),
                 "--checkpoint", str(checkpoint), "--eval-part", "development", "--seed", "71042"]
    plan = [dict(name="known_development", entry="reconstruction", arguments=arguments)]
    return plan


def replace_argument(plan, flag, value):
    args = plan[0]["arguments"]
    args[args.index(flag) + 1] = value


@pytest.mark.parametrize("mutation", [
    lambda p: p[0].update(entry="interface"),
    lambda p: p[0].update(name="../escape"),
    lambda p: p[0].update(name=None),
    lambda p: p.__setitem__(0, None),
    lambda p: p.append(copy.deepcopy(p[0])),
    lambda p: replace_argument(p, "--stage", "train"),
    lambda p: replace_argument(p, "--eval-part", "preflight"),
    lambda p: replace_argument(p, "--cache", "relative-cache.pt"),
    lambda p: replace_argument(p, "--seed", "-1"),
    lambda p: replace_argument(p, "--seed", str(2**63)),
    lambda p: p[0]["arguments"].extend(["--updates", "1024"]),
    lambda p: p[0]["arguments"].extend(["--run-dir", "/tmp/override"]),
    lambda p: p[0]["arguments"].append("--eval-part=confirmation"),
])
def test_eval_plan_rejects_training_flags_duplicate_options_and_wrong_scope(eval_inputs, mutation):
    mutation(eval_inputs)
    with pytest.raises(ValueError):
        parallel.validate_plan(eval_inputs)


def test_eval_records_keep_original_arguments_and_frozen_input_paths(eval_inputs, tmp_path):
    validated = parallel.validate_plan(eval_inputs)
    job = parallel.job_records(validated, tmp_path / "suite")[0]
    assert job["command"][1:4] == ["-u", "-m", "vera_mem.reconstruction_run"]
    assert job["command"][4:-2] == eval_inputs[0]["arguments"]
    assert job["configuration"]["eval_part"] == "development"
    assert job["configuration"]["seed"] == 71042
    assert job["status"] == "pending" and job["pid"] is None
    assert Path(job["inputs"]["cache"]).is_file()


@pytest.fixture
def sealed_eval(eval_inputs, tmp_path):
    validated = parallel.validate_plan(eval_inputs)
    source = tmp_path / "source"; source.mkdir()
    (source / "frozen.py").write_text("value = 1\n")
    job = parallel.job_records(validated, tmp_path / "suite")[0]
    manifest = dict(input_files=parallel.common.input_metadata(validated),
                    source_files_sha256=parallel.source_hashes(source), jobs=[job])
    child = dict(protocol=parallel.CHILD_PROTOCOL, complete=True, stage="eval", backbone_unchanged=True,
                 configuration=dict(job["configuration"]),
                 cache_sha256=manifest["input_files"][job["inputs"]["cache"]]["sha256"],
                 checkpoint_sha256=manifest["input_files"][job["inputs"]["checkpoint"]]["sha256"],
                 result=dict(protocol="reconstruction-cpu-vdb-v1", complete=True, shared_parameters_unchanged=True))
    output = Path(job["output"]); output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(child))
    return job, manifest, source, child


def test_matching_eval_child_requires_complete_frozen_cpu_bank_contract(sealed_eval):
    job, manifest, source, _child = sealed_eval
    parallel.verify_child_result(job, manifest, source)
    assert job["source_unchanged"] and job["inputs_unchanged"] and job["manifest_sha256"]
    assert "grouped CPU-bank" in job["artifact_validation"]


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(protocol="dictionary-coldstart-v1"),
    lambda c: c.update(stage="train"),
    lambda c: c.update(complete=False),
    lambda c: c.update(backbone_unchanged=False),
    lambda c: c.update(cache_sha256="0" * 64),
    lambda c: c.update(checkpoint_sha256="0" * 64),
    lambda c: c["configuration"].update(stage="train"),
    lambda c: c["configuration"].update(eval_part="confirmation"),
    lambda c: c["configuration"].update(seed=71043),
    lambda c: c["configuration"].update(model="/another/model"),
    lambda c: c["configuration"].update(cache="/another/cache.pt"),
    lambda c: c["configuration"].update(checkpoint="/another/checkpoint.pt"),
    lambda c: c["result"].update(protocol="interface-cpu-vdb-eval-v1"),
    lambda c: c["result"].update(complete=False),
    lambda c: c["result"].update(shared_parameters_unchanged=False),
])
def test_zero_exit_child_cannot_change_plan_or_claim_other_evaluation_protocol(sealed_eval, mutation):
    job, manifest, source, child = sealed_eval
    mutation(child)
    (Path(job["output"]) / "manifest.json").write_text(json.dumps(child))
    with pytest.raises(RuntimeError):
        parallel.verify_child_result(job, manifest, source)


@pytest.mark.parametrize("field", ["cache", "checkpoint", "model_manifest"])
def test_input_sha_validation_rejects_same_size_same_mtime_content_change(sealed_eval, field):
    job, manifest, source, _ = sealed_eval
    path = Path(job["inputs"][field]); stat = path.stat()
    content = path.read_bytes()
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="SHA256"):
        parallel.verify_frozen_inputs(job, manifest, source)


def test_frozen_source_change_is_rejected_even_when_inputs_stay_equal(sealed_eval):
    job, manifest, source, _ = sealed_eval
    (source / "frozen.py").write_text("value = 2\n")
    with pytest.raises(RuntimeError, match="source snapshot"):
        parallel.verify_frozen_inputs(job, manifest, source)


def test_validate_only_never_queries_gpu_spawns_or_creates_suite(eval_inputs, tmp_path, monkeypatch):
    make_repository(tmp_path)
    plan = tmp_path / "plan.json"; plan.write_text(json.dumps(eval_inputs))
    monkeypatch.setattr(parallel.shutil, "disk_usage", lambda _path: SimpleNamespace(free=20 * 1024**3))
    monkeypatch.setattr(parallel.common, "run_jobs", lambda *_a, **_kw: pytest.fail("Dispatch in validate-only"))
    monkeypatch.setattr(parallel.common, "probe_gpu", lambda *_a, **_kw: pytest.fail("GPU query in validate-only"))
    args = ["--workspace", str(tmp_path), "--shared-root", str(tmp_path), "--gpu", GPU,
            "--name", "eval_test", "--plan", str(plan), "--workers", "4", "--validate-only"]
    assert parallel.main(args) == 0
    assert not (tmp_path / "runs/eval_test").exists()


def test_artifact_rejection_turns_process_zero_into_visible_failure_without_killing():
    class Process:
        pid = 314
        def poll(self): return 0
        def terminate(self): pytest.fail("Must not terminate")
        def kill(self): pytest.fail("Must not kill")
    job = {}
    def rejected(): raise RuntimeError("wrong reconstruction scope")
    child = parallel._AuditedChild(Process(), job, rejected)
    assert child.poll() == 1 and child.poll() == 1
    assert job["process_exit_code"] == 0 and "wrong reconstruction scope" in job["artifact_error"]


def confirmation_cli(eval_inputs, tmp_path, monkeypatch):
    make_repository(tmp_path)
    replace_argument(eval_inputs, "--eval-part", "confirmation")
    plan = tmp_path / "confirmation_plan.json"; plan.write_text(json.dumps(eval_inputs))
    monkeypatch.setattr(parallel.shutil, "disk_usage", lambda _path: SimpleNamespace(free=20 * 1024**3))
    return ["--workspace", str(tmp_path), "--shared-root", str(tmp_path), "--gpu", GPU,
            "--name", "confirmation_barrier", "--plan", str(plan), "--workers", "2"]


def test_confirmation_without_selection_fails_before_snapshot_or_dispatch(eval_inputs, tmp_path, monkeypatch, capsys):
    args = confirmation_cli(eval_inputs, tmp_path, monkeypatch)
    monkeypatch.setattr(parallel.common, "run_jobs", lambda *_a, **_k: pytest.fail("Unsealed confirmation dispatched"))
    with pytest.raises(SystemExit) as error:
        parallel.main(args)
    assert error.value.code == 2
    assert "requires sealed development selection evidence" in capsys.readouterr().err
    assert not (tmp_path / "runs/confirmation_barrier").exists()


@pytest.mark.parametrize("content", ['{"sealed":true,', 'null', '{}'])
def test_confirmation_rejects_unreadable_selection_before_any_suite_creation(eval_inputs, tmp_path, monkeypatch, content):
    args = confirmation_cli(eval_inputs, tmp_path, monkeypatch)
    evidence = tmp_path / "bad_selection.json"; evidence.write_text(content)
    monkeypatch.setattr(parallel.common, "run_jobs", lambda *_a, **_k: pytest.fail("Invalid selection dispatched"))
    with pytest.raises(SystemExit) as error:
        parallel.main(args + ["--selection-evidence", str(evidence)])
    assert error.value.code == 2
    assert not (tmp_path / "runs/confirmation_barrier").exists()


def sealed_selection(protocol):
    """Build the real selector output from all 36 development cells and teacher.

    Numbers are synthetic, deliberately with no eligible architecture; no GPU or
    real held-out predictions are consulted to exercise the launch barrier.
    """
    records = []
    for arm in summary.ARMS:
        for seed in summary.SEEDS:
            directory = f"/sealed/development/{arm}_{seed}"
            records.append(dict(kind="training", arm=arm, seed=seed,
                run_dir=directory + "/train", trainable_parameters=1000,
                artifacts={name: "a" * 64 for name in ("manifest.json", "training_status.json", "training.jsonl", "initial.pt", "last.pt", "step_1024.pt")}))
            for split in ("known", "dev"):
                records.append(dict(kind="evaluation", arm=arm, seed=seed,
                    run_dir=directory + "/" + split, split=split, part="development", bytes_per_fact=512,
                    groups=dict(CC_B=dict(count=16, pair_em=1.),
                        CC_C=dict(count=16, update_restore_em=0., updated_em=0., shuffle_em=0., empty_em=0., locality_joint=1.)),
                    artifacts={name: "b" * 64 for name in ("manifest.json", "summary.json", "predictions.jsonl", "pairs.json", "encoded_payloads.pt", "bank_events.pt")}))
    records.append(dict(kind="teacher", split="train", part="preflight",
        run_dir="/sealed/development/teacher", groups={world: dict(count=16, correct=16) for world in ("CC_A", "CC_B")},
        artifacts={name: "c" * 64 for name in ("manifest.json", "summary.json", "predictions.jsonl")}))
    return dict(protocol=summary.SELECTION_PROTOCOL, sealed=True, created_at="2026-10-06T00:00:00+00:00",
        script_sha256=serial.sha256(Path(summary.__file__)), protocol_sha256=serial.sha256(protocol),
        decision=summary.development_selection(records))


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(sealed=False),
    lambda p: p.update(protocol="other-selection-v1"),
    lambda p: p.update(protocol_sha256="0" * 64),
    lambda p: p["decision"]["policy"].update(c_update_restore_min=0.),
    lambda p: p["decision"]["development_barrier"].update(observed=35),
    lambda p: p["decision"]["development_barrier"].update(complete=False),
    lambda p: p["decision"]["candidates"].pop(),
    lambda p: p["decision"]["candidates"][0]["per_seed"].pop(),
    lambda p: p["decision"]["candidates"][0].update(eligible=True),
    lambda p: p["decision"].update(candidates=None),
    lambda p: p["decision"]["candidates"].__setitem__(0, None),
    lambda p: p["decision"]["candidates"][0].update(per_seed=None),
    lambda p: p["decision"]["candidates"][0]["per_seed"].__setitem__(0, None),
    lambda p: p["decision"]["candidates"][0]["per_seed"][0].update(passed=0),
    lambda p: p["decision"]["candidates"][0]["per_seed"][0].update(ab_pair=.9),
    lambda p: p["decision"]["candidates"][0]["per_seed"][0].update(c_real=True),
    lambda p: p["decision"]["candidates"][0]["per_seed"][0].update(c_real=float("nan")),
    lambda p: p["decision"]["candidates"][0]["per_seed"][0].update(c_real=float("inf")),
    lambda p: p["decision"]["candidates"][0]["per_seed"][0].update(c_real=-1.),
    lambda p: p["decision"]["candidates"][0]["per_seed"][0].update(c_shuffle=2.),
    lambda p: p["decision"]["candidates"][0]["per_seed"][0].update(ab_pair=14/16),
    lambda p: p["decision"]["candidates"][0]["per_seed"][0].update(c_update_restore=1/16),
    lambda p: p["decision"]["candidates"][0]["per_seed"][0].update(checks=None),
    lambda p: p["decision"]["candidates"][0].update(minimum_seed_c_update_restore=1/16),
    lambda p: p["decision"]["candidates"][0].update(trainable_parameters=True),
    lambda p: p["decision"]["candidates"][0].update(bytes_per_fact=0),
    lambda p: p["decision"].update(selected_arm="S1_fixedB"),
    lambda p: p["decision"]["evidence"].pop(),
    lambda p: p["decision"]["evidence"][0]["artifacts"].update(**{"manifest.json": "not-a-sha"}),
    lambda p: p["decision"]["evidence"].__setitem__(0, copy.deepcopy(p["decision"]["evidence"][1])),
    lambda p: p["decision"]["evidence"][0].update(run_dir="/sealed/../escape"),
    lambda p: p["decision"]["evidence"][0].update(artifacts={"unexpected.pt": "a" * 64}),
    lambda p: p["decision"]["evidence"][0].update(artifacts={name: "a" * 64 for name in ("manifest.json", "summary.json", "predictions.jsonl")}),
])
def test_selection_semantic_corruption_rejected_before_dispatch(eval_inputs, tmp_path, monkeypatch, mutation):
    args = confirmation_cli(eval_inputs, tmp_path, monkeypatch)
    protocol = tmp_path / "VeRA-Mem/docs/reconstruction_protocol.md"
    protocol.write_text("frozen protocol\n")
    # The selector returns the shared GATE_POLICY object; isolate test mutation.
    payload = copy.deepcopy(sealed_selection(protocol)); mutation(payload)
    evidence = tmp_path / "corrupt_selection.json"; evidence.write_text(json.dumps(payload))
    monkeypatch.setattr(parallel.common, "run_jobs", lambda *_a, **_k: pytest.fail("Corrupt selection dispatched"))
    with pytest.raises(SystemExit) as error:
        parallel.main(args + ["--selection-evidence", str(evidence)])
    assert error.value.code == 2
    assert not (tmp_path / "runs/confirmation_barrier").exists()


@pytest.mark.parametrize("counts,parameters,bytes_,expected", [
    ([15, 16, 15, 15], [1, 100, 1, 1], [1, 100, 1, 1], "S3_fixedB"),
    ([16, 16, 16, 16], [20, 20, 10, 20], [1, 1, 100, 1], "S1_trainB"),
    ([16, 16, 16, 16], [10, 10, 10, 10], [100, 100, 100, 50], "S3_trainB"),
    ([16, 16, 16, 16], [10, 10, 10, 10], [100, 100, 100, 100], "S1_fixedB"),
])
def test_selection_recomputes_all_four_ranking_tiebreakers(tmp_path, counts, parameters, bytes_, expected):
    protocol = tmp_path / "protocol.md"; protocol.write_text("sealed\n")
    payload = copy.deepcopy(sealed_selection(protocol))
    for c, count, params, size in zip(payload["decision"]["candidates"], counts, parameters, bytes_):
        for s in c["per_seed"]:
            s.update(ab_pair=15/16, c_update_restore=count/16, c_real=1., c_shuffle=.5, c_empty=.5,
                c_locality_joint=1., passed=True,
                checks=dict(ab_pair=True, c_update_restore=True, c_real_minus_shuffle=True,
                    c_real_minus_empty=True, c_locality_joint=True))
        c.update(eligible=True, minimum_seed_c_update_restore=count/16,
            trainable_parameters=params, bytes_per_fact=size)
    payload["decision"]["selected_arm"] = expected
    assert parallel.validate_selection(payload, protocol) is payload
    payload["decision"]["selected_arm"] = next(a for a in summary.ARMS if a != expected)
    with pytest.raises(ValueError, match="fixed ranking"):
        parallel.validate_selection(payload, protocol)


def test_selection_exact_bytes_and_hash_are_persisted_before_first_dispatch(eval_inputs, tmp_path, monkeypatch):
    args = confirmation_cli(eval_inputs, tmp_path, monkeypatch)
    protocol = tmp_path / "VeRA-Mem/docs/reconstruction_protocol.md"
    protocol.write_text("frozen protocol\n")
    evidence = tmp_path / "sealed_selection.json"
    payload = sealed_selection(protocol)
    evidence.write_text(json.dumps(payload, indent=2) + "\n")
    original = evidence.read_bytes()
    dispatches = []
    def fake_scheduler(manifest, persist, _gpu, _env, source, **kwargs):
        dispatches.append(kwargs["workers"])
        snapshot = source.parent / "selection_evidence.json"
        on_disk = json.loads((source.parent / "suite.json").read_text())
        assert snapshot.read_bytes() == original
        assert manifest["selection_evidence_sha256"] == on_disk["selection_evidence_sha256"] == serial.sha256(snapshot)
        assert manifest["selection_evidence_source"] == str(evidence.resolve())
        assert on_disk["jobs"] and all(j["status"] == "pending" for j in on_disk["jobs"])
        evidence.write_text('{"caller_changed_after_snapshot":true}\n')
        assert snapshot.read_bytes() == original
        manifest.update(status="complete", complete=True); persist()
        return 0
    monkeypatch.setattr(parallel.common, "run_jobs", fake_scheduler)
    assert parallel.main(args + ["--selection-evidence", str(evidence)]) == 0
    assert dispatches == [2]
    snapshot = tmp_path / "runs/confirmation_barrier/selection_evidence.json"
    assert snapshot.read_bytes() == original and evidence.read_bytes() != original
