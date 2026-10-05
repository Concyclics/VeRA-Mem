"""Launch safety and reproducibility checks without starting a GPU process."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "run_generalization_suite", Path(__file__).resolve().parents[1] / "scripts" / "run_generalization_suite.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def workspace(tmp_path):
    repository = tmp_path / "VeRA-Mem"
    for directory in ("src", "configs", "scripts", "tests"):
        (repository / directory).mkdir(parents=True)
        (repository / directory / "fixture.txt").write_text(directory)
    (repository / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (repository / "README.md").write_text("Fixture documentation\n")
    (tmp_path / "models" / "Qwen3-4B-Instruct-2507").mkdir(parents=True)
    (tmp_path / "env_deps").mkdir()
    (tmp_path / "features.pt").write_bytes(b"prepared-feature-fixture")
    return tmp_path


def arguments(workspace, profile="augment"):
    return ["--workspace", str(workspace), "--gpu", "GPU-fixture",
            "--name", "new_run", "--profile", profile,
            "--cache", str(workspace / "features.pt")]


def test_control_augmentation_and_invariance_have_identical_training_budgets():
    jobs = [launcher.profile_jobs(profile)[0] for profile in ("control", "augment", "invariant")]
    assert [job["condition"] for job in jobs] == ["canonical", "augment", "invariant"]
    budgets = [{key: value for key, value in job.items() if key not in ("name", "condition")}
               for job in jobs]
    assert budgets[0] == budgets[1] == budgets[2]
    assert budgets[0] == dict(train_size=4096, updates=1536, oracle_updates=768,
                             batch_size=8, alignment_steps=400, validate_every=256,
                             eval_size=128, seed=42)


def test_occupied_gpu_creates_failed_manifest_without_starting_process(workspace, monkeypatch):
    monkeypatch.setattr(launcher, "gpu_memory_used", lambda _: 40000)
    def forbidden(*args, **kwargs):
        pytest.fail("An occupied GPU must never start a subprocess")
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    assert launcher.main(arguments(workspace)) == 1
    suite = workspace / "runs" / "new_run"
    manifest = json.loads((suite / "manifest.json").read_text())
    assert manifest["status"] == "failed" and not manifest["complete"]
    assert manifest["jobs"][0]["status"] == "failed"
    assert manifest["jobs"][0]["gpu_memory_used_mib_before_start"] == 40000
    assert not (suite / "augment4096.log").exists()


def test_success_runs_snapshot_with_pinned_environment_and_records_hashes(workspace, monkeypatch):
    monkeypatch.setattr(launcher, "gpu_memory_used", lambda _: 0)
    calls = []
    def record(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess, "run", record)
    assert launcher.main(arguments(workspace, profile="smoke")) == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    suite = workspace / "runs" / "new_run"
    manifest = json.loads((suite / "manifest.json").read_text())
    assert json.loads((suite / "suite.json").read_text()) == manifest
    assert manifest["complete"] and manifest["status"] == "complete"
    assert manifest["jobs"][0]["exit_code"] == 0
    assert kwargs["cwd"] == suite / "source"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-fixture"
    assert kwargs["env"]["PYTHONPATH"].endswith(str(suite / "source" / "src"))
    assert kwargs["env"]["OMP_NUM_THREADS"] == "4"
    assert command[command.index("--run-dir") + 1] == str(suite / "augment32smoke")
    assert command[command.index("--condition") + 1] == "augment"
    assert command[command.index("--updates") + 1] == "8"
    assert "vera_mem.generalization_run" in command
    for name, digest in manifest["source_files_sha256"].items():
        assert launcher.sha256(suite / "source" / name) == digest
    assert manifest["cache_sha256"] == launcher.sha256(workspace / "features.pt")


def test_existing_suite_is_never_overwritten(workspace, monkeypatch):
    suite = workspace / "runs" / "new_run"
    suite.mkdir(parents=True)
    sentinel = suite / "manifest.json"
    sentinel.write_text("prior evidence")
    with pytest.raises(FileExistsError):
        launcher.main(arguments(workspace))
    assert sentinel.read_text() == "prior evidence"


def test_training_failure_retains_exit_status_without_claiming_completion(workspace, monkeypatch):
    monkeypatch.setattr(launcher, "gpu_memory_used", lambda _: 0)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=7))
    assert launcher.main(arguments(workspace)) == 7
    manifest = json.loads((workspace / "runs" / "new_run" / "manifest.json").read_text())
    assert not manifest["complete"] and manifest["status"] == "failed"
    assert manifest["jobs"][0]["exit_code"] == 7
    assert manifest["jobs"][0]["status"] == "failed"


def test_missing_cache_rejects_launch_before_creating_run_directory(workspace):
    (workspace / "features.pt").unlink()
    with pytest.raises(SystemExit):
        launcher.main(arguments(workspace))
    assert not (workspace / "runs").exists()
