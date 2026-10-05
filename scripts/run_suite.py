"""Sequential single-GPU suite with durable exit codes and source snapshots."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--workspace", type=Path, required=True)
p.add_argument("--gpu", required=True, help="Explicit GPU UUID")
p.add_argument("--name", required=True)
p.add_argument("--profile", choices=["initial", "centered", "staged"], default="initial")
p.add_argument("--wait-for-suite", type=Path, help="Wait at most 30 minutes for a preceding suite.json to complete")
a = p.parse_args()
if a.wait_for_suite:
    print("Waiting for prerequisite suite completion", flush=True)
    deadline = time.monotonic() + 1800
    while True:
        try:
            prerequisite = json.loads(a.wait_for_suite.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            prerequisite = {}
        if prerequisite.get("complete"):
            break
        if any(job.get("exit_code", 0) != 0 for job in prerequisite.get("jobs", [])):
            raise RuntimeError("Prerequisite suite failed; no new experiment started")
        if time.monotonic() > deadline:
            raise TimeoutError("Prerequisite did not complete within 30 minutes")
        time.sleep(10)
w = a.workspace.resolve()
repo = w / "VeRA-Mem"
suite = w / "runs" / a.name
suite.mkdir(parents=True, exist_ok=False)
for directory in ("src", "configs", "scripts", "tests"):
    shutil.copytree(repo / directory, suite / "source" / directory,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
shutil.copy(repo / "pyproject.toml", suite / "source" / "pyproject.toml")
source = suite / "source"
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
           HF_HOME=str(w / "cache" / "huggingface"),
           PYTHONPATH=str(w / "env_deps") + os.pathsep + str(source / "src"))
memory_used = int(subprocess.check_output([
    "nvidia-smi", "--id=" + a.gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
if memory_used > 1000:
    raise RuntimeError(f"Selected GPU already uses {memory_used} MiB; no job started")
manifest = {"name": a.name, "gpu_uuid": a.gpu, "started_at": datetime.now(timezone.utc).isoformat(), "jobs": []}
model = str(w / "models" / "Qwen3-4B-Instruct-2507")
jobs = [
    ("vector_synthetic", "vera_mem.vector_run", "vector_pilot.json", "synthetic", []),
    ("baseline_synthetic", "vera_mem.run", "baselines.json", "synthetic", []),
    ("baseline_medmcqa", "vera_mem.run", "baselines.json", "medmcqa", []),
    ("vector_medmcqa_transfer", "vera_mem.vector_run", "vector_pilot.json", "medmcqa",
        ["--checkpoint", str(suite / "vector_synthetic" / "vector_vera.pt"), "--methods", "vdb_real", "vdb_oracle", "vdb_empty"]),
]
if a.profile == "centered":
    jobs = [
        ("vector_synthetic", "vera_mem.vector_run", "vector_centered.json", "synthetic", []),
        ("baseline_synthetic", "vera_mem.run", "baselines_eval43.json", "synthetic", []),
    ]
if a.profile == "staged":
    jobs = [
        ("vector_synthetic", "vera_mem.vector_run", "vector_staged.json", "synthetic", []),
        ("baseline_synthetic", "vera_mem.run", "baselines_eval44.json", "synthetic", []),
    ]
for name, module, config, dataset, extra in jobs:
    command = [sys.executable, "-u", "-m", module,
        "--config", str(source / "configs" / config), "--model", model,
        "--data", str(w / "data" / dataset), "--output", str(suite / name),
        "--dataset", dataset, *extra]
    record = {"name": name, "command": command, "started_at": datetime.now(timezone.utc).isoformat()}
    manifest["jobs"].append(record)
    (suite / "suite.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("START " + name, flush=True)
    with (suite / (name + ".log")).open("w") as log:
        result = subprocess.run(command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, finished_at=datetime.now(timezone.utc).isoformat())
    (suite / "suite.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"FINISH {name} exit={result.returncode}", flush=True)
    if result.returncode:
        sys.exit(result.returncode)
manifest.update(complete=True, completed_at=datetime.now(timezone.utc).isoformat())
(suite / "suite.json").write_text(json.dumps(manifest, indent=2)+"\n")
print("SUITE_COMPLETE", flush=True)
