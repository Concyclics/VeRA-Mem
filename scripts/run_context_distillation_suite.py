"""Run context-distillation ablations from an immutable single-GPU snapshot."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_generalization_suite import atomic_json, gpu_memory_used, sha256, utc_now


METHODS = ("initial", "ce", "off_kd", "off_kd_hidden", "on_kd", "on_kd_hidden")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--updates", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=4)
    a = p.parse_args()
    if Path(a.name).name != a.name or a.name in ("", ".", ".."):
        p.error("Use one new directory name")
    if not a.gpu.startswith("GPU-") or "," in a.gpu or any(c.isspace() for c in a.gpu):
        p.error("An explicit single GPU UUID is required")
    if len(a.methods) != len(set(a.methods)):
        p.error("Repeated methods would overwrite a run")
    if min(a.updates, a.batch_size, a.eval_size, a.max_new_tokens) < 1:
        p.error("Budgets must be positive")
    w = a.workspace.resolve()
    repo, model = w / "VeRA-Mem", w / "models/Qwen3-4B-Instruct-2507"
    cache, checkpoint = a.cache.resolve(), a.checkpoint.resolve()
    if not cache.is_file() or not checkpoint.is_file() or not model.is_dir():
        p.error("Existing cache, checkpoint and model directory required")
    suite = w / "runs" / a.name
    suite.mkdir(parents=True, exist_ok=False)
    source = suite / "source"
    manifest = dict(protocol="context-distillation-pilot-v1", started_at=utc_now(),
                    complete=False, status="preparing", gpu_uuid=a.gpu,
                    launcher_pid=os.getpid(), jobs=[])
    persist = lambda: atomic_json(suite / "suite.json", manifest)
    persist()
    active = None
    try:
        for directory in ("src", "scripts", "tests", "configs", "docs"):
            shutil.copytree(repo / directory, source / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        for filename in ("README.md", "pyproject.toml"):
            shutil.copy2(repo / filename, source / filename)
        manifest.update(cache_sha256=sha256(cache), checkpoint_sha256=sha256(checkpoint),
                        source_files_sha256={str(f.relative_to(source)): sha256(f)
                                             for f in sorted(source.rglob("*")) if f.is_file()})
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                   HF_HOME=str(w / "cache/huggingface"),
                   PYTHONPATH=str(w / "env_deps") + os.pathsep + str(source / "src"))
        manifest.update(status="running", runtime_environment={k: env[k] for k in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "TOKENIZERS_PARALLELISM", "HF_HOME", "PYTHONPATH")})
        persist()
        for method in a.methods:
            command = [sys.executable, "-u", "-m", "vera_mem.context_distillation_run",
                       "--model", str(model), "--cache", str(cache), "--checkpoint", str(checkpoint),
                       "--run-dir", str(suite / method), "--method", "ce" if method == "initial" else method,
                       "--updates", str(a.updates), "--batch-size", str(a.batch_size),
                       "--eval-size", str(a.eval_size), "--max-new-tokens", str(a.max_new_tokens)]
            command += ["--evaluate-only"] if method == "initial" else ["--skip-teacher-eval"]
            log = suite / (method + ".log")
            active = dict(name=method, command=command, log=str(log), status="preflight", exit_code=None)
            manifest["jobs"].append(active)
            persist()
            used = gpu_memory_used(a.gpu)
            active["gpu_memory_used_mib_before_start"] = used
            if used > 1000:
                raise RuntimeError(f"GPU uses {used} MiB; no experiment started")
            active.update(status="running", started_at=utc_now())
            persist()
            print(f"START {method}", flush=True)
            started = time.monotonic()
            with log.open("w") as handle:
                result = subprocess.run(command, env=env, cwd=source, stdout=handle,
                                        stderr=subprocess.STDOUT, check=False)
            active.update(exit_code=result.returncode, elapsed_seconds=time.monotonic()-started,
                          finished_at=utc_now(), status="complete" if result.returncode == 0 else "failed")
            persist()
            print(f"FINISH {method} exit={result.returncode}", flush=True)
            if result.returncode:
                raise RuntimeError(f"Experiment {method} failed with exit {result.returncode}")
            active = None
        manifest.update(complete=True, status="complete", completed_at=utc_now())
        persist()
        print("SUITE_COMPLETE", flush=True)
        return 0
    except Exception as error:
        if active is not None:
            active.update(status="failed", error=repr(error))
        manifest.update(status="failed", error=repr(error), failed_at=utc_now())
        persist()
        print(repr(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
