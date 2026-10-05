"""Run a sequential single-GPU scaling suite from an immutable source snapshot.

The feature cache must already exist. This launcher never prepares data, waits
for another experiment, terminates unrelated processes, or resumes output dirs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    """Publish complete JSON snapshots; readers never see partial writes."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_memory_used(gpu: str) -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--id=" + gpu, "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        text=True, timeout=30,
    ).strip()
    rows = output.splitlines()
    if len(rows) != 1:
        raise RuntimeError("Select exactly one GPU; nvidia-smi returned multiple rows")
    used = int(rows[0])
    if used < 0:
        raise RuntimeError("nvidia-smi returned an invalid memory measurement")
    return used


def profile_jobs(profile: str) -> list[dict]:
    common = dict(updates=512, alignment_steps=400, oracle_updates=256,
                  cold_bank_size=0, eval_size=128, seed=42)
    if profile == "raw":
        jobs = [dict(name=f"raw{size}", variant="raw", train_size=size)
                for size in (128, 4096)]
    elif profile == "stable":
        jobs = [dict(name=f"stable{size}", variant="stable", train_size=size)
                for size in (128, 1024, 4096)]
        jobs.append(dict(name="standardtrained_coldinit", variant="stable", train_size=4096,
                         cold_bank_size=128, checkpoint_from="stable4096"))
    elif profile == "cold":
        jobs = [dict(name="stable4096cold128", variant="stable", train_size=4096,
                     cold_bank_size=128),
                dict(name="coldtrained_emptyinit", variant="stable", train_size=4096,
                     cold_bank_size=0, checkpoint_from="stable4096cold128")]
    elif profile == "extended":
        jobs = [dict(name="stable4096long", variant="stable", train_size=4096,
                     updates=1536, oracle_updates=768)]
    elif profile == "smoke":
        jobs = [dict(name="stable32", variant="stable", train_size=32,
                     updates=128, alignment_steps=100, oracle_updates=64,
                     eval_size=16)]
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return [{**common, **job} for job in jobs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="One explicit GPU UUID")
    parser.add_argument("--name", required=True, help="New directory name under workspace/runs")
    parser.add_argument("--profile", choices=["raw", "stable", "cold", "smoke", "extended"], required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Existing prepared feature-cache file")
    args = parser.parse_args(argv)
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be one directory name")
    if not args.gpu.strip() or "," in args.gpu:
        parser.error("--gpu must identify exactly one GPU")

    workspace = args.workspace.resolve()
    repository = workspace / "VeRA-Mem"
    model = workspace / "models" / "Qwen3-4B-Instruct-2507"
    cache = args.cache.resolve()
    if not cache.is_file():
        parser.error(f"Prepared feature cache does not exist: {cache}; no job started")
    if not model.is_dir():
        parser.error(f"Model directory does not exist: {model}; no job started")
    if not (workspace / "env_deps").is_dir():
        parser.error(f"Dependency directory does not exist: {workspace / 'env_deps'}")
    for item in ("src", "configs", "scripts", "tests", "pyproject.toml"):
        if not (repository / item).exists():
            parser.error(f"Repository source is incomplete: {repository / item}")

    suite = workspace / "runs" / args.name
    suite.mkdir(parents=True, exist_ok=False)
    source = suite / "source"
    manifest = dict(
        name=args.name, profile=args.profile, gpu_uuid=args.gpu,
        started_at=utc_now(), complete=False, status="preparing",
        workspace=str(workspace), source=str(source), model=str(model),
        cache=str(cache), cache_sha256=sha256(cache),
        python=sys.executable, launcher_pid=os.getpid(), jobs=[],
    )

    def persist() -> None:
        atomic_json(suite / "manifest.json", manifest)
        atomic_json(suite / "suite.json", manifest)

    persist()
    active_record = None
    try:
        for directory in ("src", "configs", "scripts", "tests"):
            shutil.copytree(
                repository / directory, source / directory,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
            )
        shutil.copy2(repository / "pyproject.toml", source / "pyproject.toml")
        manifest["source_files_sha256"] = {
            str(path.relative_to(source)): sha256(path)
            for path in sorted(source.rglob("*")) if path.is_file()
        }
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=args.gpu,
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            NUMEXPR_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
            HF_HOME=str(workspace / "cache" / "huggingface"),
            PYTHONPATH=str(workspace / "env_deps") + os.pathsep + str(source / "src"),
        )
        manifest["runtime_environment"] = {key: env[key] for key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM",
            "HF_HOME", "PYTHONPATH",
        )}
        manifest["status"] = "running"
        persist()

        for job in profile_jobs(args.profile):
            output = suite / job["name"]
            command = [sys.executable, "-u", "-m", "vera_mem.scaling_run",
                       "--model", str(model), "--cache", str(cache),
                       "--output", str(output)]
            for key in ("variant", "train_size", "updates", "alignment_steps",
                        "oracle_updates", "cold_bank_size", "eval_size", "seed"):
                command.extend(["--" + key.replace("_", "-"), str(job[key])])
            checkpoint = None
            if "checkpoint_from" in job:
                checkpoint = suite / job["checkpoint_from"] / "best.pt"
                command.extend(["--checkpoint", str(checkpoint)])
            log_path = suite / (job["name"] + ".log")
            active_record = dict(
                name=job["name"], output=str(output), log=str(log_path), command=command,
                status="preflight", preflight_at=utc_now(), exit_code=None,
                mode="evaluation_only" if checkpoint is not None else "training_and_evaluation",
            )
            manifest["jobs"].append(active_record)
            persist()
            if not cache.is_file():
                raise FileNotFoundError(f"Prepared feature cache disappeared: {cache}")
            if checkpoint is not None and not checkpoint.is_file():
                raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint}")
            memory_used = gpu_memory_used(args.gpu)
            active_record["gpu_memory_used_mib_before_start"] = memory_used
            if memory_used > 1000:
                raise RuntimeError(
                    f"Selected GPU already uses {memory_used} MiB; threshold is 1000 MiB; no job started"
                )
            active_record.update(status="running", started_at=utc_now())
            persist()
            print("START " + job["name"], flush=True)
            started = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, env=env, cwd=source,
                    stdout=log, stderr=subprocess.STDOUT, check=False,
                )
            active_record.update(
                exit_code=result.returncode, finished_at=utc_now(),
                elapsed_seconds=time.monotonic() - started,
                status="complete" if result.returncode == 0 else "failed",
            )
            persist()
            print(f"FINISH {job['name']} exit={result.returncode}", flush=True)
            if result.returncode:
                manifest.update(status="failed", failed_at=utc_now())
                persist()
                return result.returncode if result.returncode > 0 else 128 - result.returncode
            active_record = None

        manifest.update(complete=True, status="complete", completed_at=utc_now())
        persist()
        print("SUITE_COMPLETE", flush=True)
        return 0
    except Exception as error:
        if active_record is not None and active_record["status"] in ("preflight", "running"):
            active_record.update(status="failed", finished_at=utc_now(), error=repr(error))
        manifest.update(status="failed", failed_at=utc_now(), error=repr(error))
        persist()
        print(f"SUITE_FAILED: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
