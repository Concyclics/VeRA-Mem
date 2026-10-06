"""Run an explicit reconstruction plan from a frozen source snapshot on one GPU.

Uses the same plan layout and resource guard as run_interface_suite.py, but
accepts only entry=reconstruction and vera_mem.reconstruction_run. No automatic retries,
job cancellation, downloads, or remote operations are performed here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from run_generalization_suite import atomic_json, gpu_memory_used, sha256, utc_now

PROTOCOL="reconstruction-experiment-v1"


def validate_plan(plan):
    if not isinstance(plan,list) or not plan:raise ValueError("Plan must be a nonempty job list")
    names=set()
    for job in plan:
        if not isinstance(job,dict) or set(job)!={"name","entry","arguments"}:raise ValueError("Job requires exactly name/entry/arguments")
        name=job["name"]
        if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",name) or name in names:
            raise ValueError("Invalid or duplicate job name")
        names.add(name)
        if job["entry"]!="reconstruction":raise ValueError("Only entry=reconstruction is permitted")
        args=job["arguments"]
        if not isinstance(args,list) or not all(isinstance(v,str) and v for v in args):raise ValueError("Arguments must be nonempty strings")
        if any(v=="--run-dir" or v.startswith("--run-dir=") for v in args):raise ValueError("The suite owns --run-dir")
        if args.count("--stage")!=1 or any(v.startswith("--stage=") for v in args):raise ValueError("Exactly one --stage VALUE is required")
        index=args.index("--stage")
        if index+1==len(args) or args[index+1] not in {"prepare","teacher","train","eval"}:raise ValueError("Invalid reconstruction stage")
    return plan


def source_hashes(source):
    return {str(f.relative_to(source)):sha256(f) for f in sorted(source.rglob("*"))
            if f.is_file() and f.suffix!=".pyc" and not {"__pycache__",".pytest_cache"}&set(f.relative_to(source).parts)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--shared-root",type=Path,required=True)
    p.add_argument("--gpu",required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--selection-evidence",type=Path,
                   help="Optional fixed teacher-selection evidence, snapshotted before any job")
    p.add_argument("--min-free-gib",type=float,default=5.)
    a=p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",a.name):p.error("Use one new suite name")
    if not a.gpu.startswith("GPU-") or "," in a.gpu or any(c.isspace() for c in a.gpu):p.error("Select one GPU UUID")
    if not 0<a.min_free_gib<100000:p.error("Invalid disk-space requirement")
    try:plan=validate_plan(json.loads(a.plan.read_text()))
    except (ValueError,TypeError,KeyError) as error:p.error(str(error))
    w,shared=a.workspace.resolve(),a.shared_root.resolve()
    if shutil.disk_usage(w).free<a.min_free_gib*1024**3:p.error("Insufficient free workspace disk space")
    suite=w/"runs"/a.name;suite.mkdir(parents=True,exist_ok=False)
    source=suite/"source"
    manifest=dict(protocol=PROTOCOL,complete=False,status="preparing",started_at=utc_now(),launcher_pid=os.getpid(),
                  gpu_uuid=a.gpu,plan_sha256=sha256(a.plan),jobs=[],min_free_gib=a.min_free_gib)
    persist=lambda:atomic_json(suite/"suite.json",manifest)
    persist();active=None
    try:
        for folder in ("src","scripts","tests","configs","docs"):
            shutil.copytree(w/"VeRA-Mem"/folder,source/folder,ignore=shutil.ignore_patterns("__pycache__","*.pyc",".pytest_cache"))
        for name in ("README.md","pyproject.toml"):shutil.copy2(w/"VeRA-Mem"/name,source/name)
        atomic_json(suite/"plan.json",plan)
        if a.selection_evidence is not None:
            evidence=json.loads(a.selection_evidence.read_text())
            if evidence.get("selected_strategy") not in {"quote_instruction","gold_annotated"}:
                raise ValueError("A supplementary training suite needs a qualified teacher strategy")
            shutil.copy2(a.selection_evidence,suite/"teacher_selection.json")
            manifest["teacher_selection"]={"source_path":str(a.selection_evidence.resolve()),
                "snapshot_path":str(suite/"teacher_selection.json"),
                "sha256":sha256(suite/"teacher_selection.json")}
            manifest["selection_evidence_sha256"]=manifest["teacher_selection"]["sha256"]
        manifest["source_files_sha256"]=source_hashes(source)
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false",HF_HOME=str(w/"cache/huggingface"),
            PYTHONPATH=str(shared/"env_deps")+os.pathsep+str(source/"src"))
        manifest.update(status="running",runtime_environment={k:env[k] for k in (
            "CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","TOKENIZERS_PARALLELISM","HF_HOME","PYTHONPATH")})
        persist()
        for job in plan:
            name=job["name"];output=suite/name
            command=[sys.executable,"-u","-m","vera_mem.reconstruction_run",*job["arguments"],"--run-dir",str(output)]
            active=dict(name=name,command=command,output=str(output),log=str(suite/(name+".log")),status="preflight",exit_code=None)
            manifest["jobs"].append(active)
            active["disk_free_bytes_before_start"]=shutil.disk_usage(w).free
            active["gpu_memory_used_mib_before_start"]=gpu_memory_used(a.gpu)
            if active["disk_free_bytes_before_start"]<a.min_free_gib*1024**3:raise RuntimeError("Insufficient free workspace disk space")
            if active["gpu_memory_used_mib_before_start"]>1000:raise RuntimeError("GPU already uses more than 1000 MiB; refusing to start")
            if source_hashes(source)!=manifest["source_files_sha256"]:raise RuntimeError("Source snapshot changed before job")
            active.update(status="running",started_at=utc_now());persist();started=time.monotonic()
            print("START "+name,flush=True)
            with Path(active["log"]).open("w") as log:
                result=subprocess.run(command,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=False)
            active.update(exit_code=result.returncode,elapsed_seconds=time.monotonic()-started,finished_at=utc_now())
            if result.returncode:raise RuntimeError(f"{name} failed with exit {result.returncode}")
            child=json.loads((output/"manifest.json").read_text())
            stage=job["arguments"][job["arguments"].index("--stage")+1]
            if child.get("complete") is not True or child.get("protocol")!="reconstruction-experiment-v1" or child.get("stage")!=stage:
                raise RuntimeError("Successful process did not produce a complete matching job manifest")
            if source_hashes(source)!=manifest["source_files_sha256"]:raise RuntimeError("Source snapshot changed during job")
            active.update(status="complete",manifest_sha256=sha256(output/"manifest.json"),source_unchanged=True);persist()
            print(f"FINISH {name} exit=0",flush=True);active=None
        manifest.update(status="complete",complete=True,source_unchanged=True,completed_at=utc_now());persist();return 0
    except BaseException as error:
        if active is not None:active.update(status="failed",error=repr(error))
        manifest.update(status="failed",complete=False,error=repr(error),failed_at=utc_now());persist();raise


if __name__=="__main__":raise SystemExit(main())
