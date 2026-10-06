"""Execute an explicit counterfactual experiment plan from a frozen snapshot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_generalization_suite import atomic_json, gpu_memory_used, sha256, utc_now


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--shared-root", type=Path, required=True, help="Existing model/dependencies root; read-only")
    p.add_argument("--gpu", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    if Path(a.name).name != a.name or a.name in ("", ".", ".."):
        p.error("Use one new suite name")
    if not a.gpu.startswith("GPU-") or "," in a.gpu or any(c.isspace() for c in a.gpu):
        p.error("Select one GPU UUID")
    w, shared = a.workspace.resolve(), a.shared_root.resolve()
    if shutil.disk_usage(w).free < 5*1024**3:
        p.error("At least 5 GiB user-available workspace space required")
    plan = json.loads(a.plan.read_text())
    if not isinstance(plan, list) or not plan or len({j["name"] for j in plan}) != len(plan):
        p.error("Plan must contain uniquely named jobs")
    suite = w/"runs"/a.name
    suite.mkdir(parents=True, exist_ok=False)
    source = suite/"source"
    manifest = dict(protocol="counterfactual-context-v2", complete=False, status="preparing",
                    started_at=utc_now(), launcher_pid=os.getpid(), gpu_uuid=a.gpu,
                    plan_sha256=sha256(a.plan), jobs=[])
    persist = lambda: atomic_json(suite/"suite.json", manifest)
    persist()
    active = None
    try:
        for folder in ("src", "scripts", "tests", "configs", "docs"):
            shutil.copytree(w/"VeRA-Mem"/folder, source/folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        for filename in ("README.md", "pyproject.toml"):
            shutil.copy2(w/"VeRA-Mem"/filename, source/filename)
        atomic_json(suite/"plan.json", plan)
        manifest["source_files_sha256"] = {str(f.relative_to(source)): sha256(f) for f in sorted(source.rglob("*")) if f.is_file()}
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   TOKENIZERS_PARALLELISM="false", HF_HOME=str(w/"cache/huggingface"),
                   PYTHONPATH=str(shared/"env_deps") + os.pathsep + str(source/"src"))
        manifest.update(status="running", runtime_environment={k: env[k] for k in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM", "HF_HOME", "PYTHONPATH")})
        persist()
        for job in plan:
            name = job["name"]
            if Path(name).name != name or name in ("", ".", ".."):
                raise ValueError("Invalid job name")
            if job["entry"] == "prepare":
                entry = [str(source/"scripts/prepare_counterfactual.py")]
                option = "--output"
            elif job["entry"] == "run":
                entry, option = ["-m", "vera_mem.counterfactual_run"], "--run-dir"
            else:
                raise ValueError("Only prepare/run entries are supported")
            output = suite/name
            command = [sys.executable, "-u", *entry, *job["arguments"], option, str(output)]
            active = dict(name=name, command=command, output=str(output), log=str(suite/(name+".log")), status="preflight", exit_code=None)
            manifest["jobs"].append(active)
            used = gpu_memory_used(a.gpu)
            active["gpu_memory_used_mib_before_start"] = used
            if used > 1000:
                raise RuntimeError(f"GPU already uses {used} MiB; refusing to start")
            active.update(status="running", started_at=utc_now())
            persist()
            started = time.monotonic()
            print("START " + name, flush=True)
            with Path(active["log"]).open("w") as log:
                result = subprocess.run(command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=False)
            active.update(status="complete" if result.returncode == 0 else "failed", exit_code=result.returncode,
                          elapsed_seconds=time.monotonic()-started, finished_at=utc_now())
            persist()
            print(f"FINISH {name} exit={result.returncode}", flush=True)
            if result.returncode:
                raise RuntimeError(f"{name} failed with exit {result.returncode}")
            active = None
        manifest.update(status="complete", complete=True, completed_at=utc_now())
        persist()
        return 0
    except BaseException as error:
        if active is not None:
            active.update(status="failed", error=repr(error))
        manifest.update(status="failed", complete=False, error=repr(error), failed_at=utc_now())
        persist()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
