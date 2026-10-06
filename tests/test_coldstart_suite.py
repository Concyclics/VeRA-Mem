"""Launch only the frozen coldstart entry and refuse unsafe/incomplete runs."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("coldstart_suite",SCRIPTS/"run_coldstart_suite.py")
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)


def plan():return [dict(name="prepare",entry="coldstart",arguments=["--stage","prepare","--domain","wikipedia"])]


def workspace(tmp_path,monkeypatch):
    repo=tmp_path/"VeRA-Mem"
    for folder in ("src","scripts","tests","configs","docs"):
        (repo/folder).mkdir(parents=True);(repo/folder/"marker.txt").write_text(folder)
    for f in ("README.md","pyproject.toml"):(repo/f).write_text(f)
    path=tmp_path/"plan.json";path.write_text(json.dumps(plan()))
    monkeypatch.setattr(suite,"gpu_memory_used",lambda _:0)
    monkeypatch.setattr(suite.shutil,"disk_usage",lambda _:SimpleNamespace(free=20*1024**3))
    return ["--workspace",str(tmp_path),"--shared-root",str(tmp_path),"--gpu","GPU-test","--name","coldstart_test","--plan",str(path)]


@pytest.mark.parametrize("change",[
    lambda p:p[0].update(entry="interface"),
    lambda p:p.append(dict(p[0])),
    lambda p:p[0].update(name="../escape"),
    lambda p:p[0]["arguments"].extend(["--run-dir","/tmp/override"]),
    lambda p:p[0]["arguments"].extend(["--stage","train"]),
])
def test_plan_rejects_wrong_entry_paths_duplicate_or_overridden_output(change):
    p=plan();change(p)
    with pytest.raises(ValueError):suite.validate_plan(p)


def test_launch_records_frozen_command_and_complete_child(tmp_path,monkeypatch):
    args=workspace(tmp_path,monkeypatch);seen=[]
    def run(command,**kwargs):
        seen.append((command,kwargs))
        target=Path(command[-1]);target.mkdir()
        (target/"manifest.json").write_text(json.dumps(dict(protocol="dictionary-coldstart-v1",stage="prepare",complete=True)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(suite.subprocess,"run",run)
    assert suite.main(args)==0
    command,kwargs=seen[0]
    assert command[1:4]==["-u","-m","vera_mem.coldstart_run"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"]=="GPU-test"
    assert str(kwargs["cwd"]).endswith("runs/coldstart_test/source")
    manifest=json.loads((tmp_path/"runs/coldstart_test/suite.json").read_text())
    assert manifest["complete"] and manifest["source_unchanged"] and manifest["jobs"][0]["manifest_sha256"]
    with pytest.raises(FileExistsError):suite.main(args)


def test_busy_gpu_never_spawns_job_and_preserves_failed_manifest(tmp_path,monkeypatch):
    args=workspace(tmp_path,monkeypatch)
    monkeypatch.setattr(suite,"gpu_memory_used",lambda _:1001)
    monkeypatch.setattr(suite.subprocess,"run",lambda *a,**k:pytest.fail("Started on busy GPU"))
    with pytest.raises(RuntimeError,match="GPU already"):suite.main(args)
    m=json.loads((tmp_path/"runs/coldstart_test/suite.json").read_text())
    assert not m["complete"] and m["jobs"][0]["status"]=="failed"


def test_teacher_choice_is_snapshotted_before_training(tmp_path,monkeypatch):
    args=workspace(tmp_path,monkeypatch)
    evidence=tmp_path/"choice.json"
    evidence.write_text(json.dumps(dict(selected_strategy="gold_annotated",probe_hash="fixed")))
    expected=evidence.read_bytes()
    def run(command,**kwargs):
        root=kwargs["cwd"].parent
        snapshot=root/"teacher_selection.json"
        manifest=json.loads((root/"suite.json").read_text())
        assert snapshot.read_bytes()==expected
        assert manifest["selection_evidence_sha256"]==suite.sha256(snapshot)
        evidence.write_text("caller file changed later")
        out=Path(command[-1]);out.mkdir()
        (out/"manifest.json").write_text(json.dumps(dict(protocol="dictionary-coldstart-v1",stage="prepare",complete=True)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(suite.subprocess,"run",run)
    assert suite.main(args+["--selection-evidence",str(evidence)])==0
    assert (tmp_path/"runs/coldstart_test/teacher_selection.json").read_bytes()==expected


@pytest.mark.parametrize("mutation",["incomplete_child","changed_source"])
def test_zero_exit_does_not_override_incomplete_artifact_or_modified_source(tmp_path,monkeypatch,mutation):
    args=workspace(tmp_path,monkeypatch)
    def run(command,**kwargs):
        out=Path(command[-1]);out.mkdir()
        (out/"manifest.json").write_text(json.dumps(dict(protocol="dictionary-coldstart-v1",stage="prepare",complete=mutation!="incomplete_child")))
        if mutation=="changed_source":(kwargs["cwd"]/"src/marker.txt").write_text("changed")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(suite.subprocess,"run",run)
    with pytest.raises(RuntimeError):suite.main(args)
    assert not json.loads((tmp_path/"runs/coldstart_test/suite.json").read_text())["complete"]
