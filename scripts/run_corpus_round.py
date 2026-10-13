"""Durable single-GPU first-round controller. Run on an authorized compute host.

No process is terminated, no occupied GPU is used, and no quality gate stops the
full pass. The engineering pilot must succeed before formal training starts.
All inputs must already be local. This controller never downloads models.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidates = [args.gpu, *args.candidate_gpu]
    if len(set(candidates)) != len(candidates) or any(not item.startswith("GPU-") or "," in item for item in candidates):
        raise ValueError("Provide distinct explicit GPU UUIDs")
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = Path(__file__).resolve().parents[1]
    names = ("corpus_data.py", "corpus_features.py", "corpus_index.py", "corpus_reader.py",
             "corpus_backend.py", "corpus_episodes.py", "corpus_train.py", "corpus_qualify.py")
    sources = {name: hashlib.sha256((source / "src/vera_mem" / name).read_bytes()).hexdigest() for name in names}
    code_id = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:12]
    qualification = root / ("qualification_" + code_id)
    pilot = root / ("pilot_" + code_id)
    training = root / "training"
    index = root / "index_initial"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(source / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    receipt_directory = root / "stages"
    receipt_directory.mkdir(exist_ok=True)
    previous_state = json.loads((root / "controller.json").read_text()) if (root / "controller.json").exists() else {}
    selected_gpu = previous_state.get("gpu") if previous_state.get("gpu") in candidates else None
    state = {"schema": "vera-corpus-controller-v1", "host": socket.gethostname(), "gpu": selected_gpu,
             "gpu_candidates": candidates,
             "code_id": code_id, "source_sha256": sources, "pid": os.getpid(), "started_at_utc": now(),
             "status": "starting", "stages": []}
    write(root / "controller.json", state)
    def gpu_available(gpu):
        report = subprocess.check_output(["nvidia-smi", "-i", gpu,
                                          "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        used, utilization = [int(x.strip()) for x in report.strip().split(",")]
        processes = subprocess.check_output(["nvidia-smi", "-i", gpu,
                                             "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        return used <= 1024 and utilization <= 5 and not processes.strip()
    def stage(name, module, options, *, gpu=False, marker=None):
        nonlocal selected_gpu
        command = [sys.executable, "-u", "-m", module, *map(str, options)]
        receipt = receipt_directory / (name + ".json")
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old["command"] == command and old["exit_code"] == 0 and (marker is None or Path(marker).exists()):
                state["stages"].append(old)
                return
        if gpu:
            while True:
                choices = [selected_gpu] if selected_gpu is not None else candidates
                available = next((item for item in choices if gpu_available(item)), None)
                if available is not None:
                    selected_gpu = available
                    state["gpu"] = selected_gpu
                    env["CUDA_VISIBLE_DEVICES"] = selected_gpu
                    break
                state.update(status="waiting_for_selected_gpu", stage=name, updated_at_utc=now())
                write(root / "controller.json", state)
                time.sleep(30)
        if shutil.disk_usage(root).free < 60 * 2**30:
            raise RuntimeError("Less than 60 GiB free; stop before allocating further artifacts")
        state.update(status="running", stage=name, updated_at_utc=now())
        write(root / "controller.json", state)
        t0 = time.monotonic()
        record = {"name": name, "command": command, "started_at_utc": now(), "code_id": code_id}
        print(json.dumps(record), flush=True)
        with (root / (name + ".log")).open("a") as stream:
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            state["child_pid"] = process.pid
            write(root / "controller.json", state)
            exit_code = process.wait()
        record.update(exit_code=exit_code, seconds=time.monotonic() - t0, finished_at_utc=now())
        write(receipt, record)
        state["stages"].append(record)
        write(root / "controller.json", state)
        if exit_code != 0:
            raise RuntimeError(f"Stage {name} failed with exit code {exit_code}; inspect {name}.log")
        if marker is not None and not Path(marker).exists():
            raise RuntimeError(f"Stage {name} omitted its completion artifact")
    try:
        stage("index", "vera_mem.corpus_index", ["--features", args.features, "--data", args.data,
                                                  "--output", index, "--backend", args.index_backend],
              gpu=args.index_backend == "ivf_gpu", marker=index / "complete.json")
        stage("qualify_" + code_id, "vera_mem.corpus_qualify",
              ["--features", args.features, "--data", args.data, "--model", args.model,
               "--index", index, "--output", qualification], gpu=True, marker=qualification / "qualification.json")
        common = ["--features", args.features, "--data", args.data, "--model", args.model,
                  "--episodes", args.episodes, "--index", qualification / "index"]
        stage("pilot_" + code_id, "vera_mem.corpus_train",
              [*common, "--output", pilot, "--batch-size", args.batch_size, "--max-updates", "16"],
              gpu=True, marker=pilot / "pilot_complete.json")
        if args.through == "pilot":
            state.update(status="engineering_complete", updated_at_utc=now())
            write(root / "controller.json", state)
            return
        stage("train_to_10m", "vera_mem.corpus_train",
              [*common, "--output", training, "--batch-size", args.batch_size, "--pause-after-tokens", "10000000"],
              gpu=True, marker=training / "development_pause.json")
        stage("evaluate_10m", "vera_mem.corpus_eval",
              [*common, "--checkpoint", training / "checkpoints/development_10m.pt", "--output", root / "development_10m",
               "--development"], gpu=True, marker=root / "development_10m/complete.json")
        stage("train_full", "vera_mem.corpus_train", [*common, "--output", training, "--batch-size", args.batch_size],
              gpu=True, marker=training / "complete.json")
        stage("evaluate_full", "vera_mem.corpus_eval",
              [*common, "--checkpoint", training / "checkpoints/final.pt", "--output", root / "confirmation"],
              gpu=True, marker=root / "confirmation/complete.json")
        stage("analyze", "vera_mem.corpus_report", ["--round", root, "--output", root / "analysis"],
              marker=root / "analysis/complete.json")
        state.update(status="complete", finished_at_utc=now())
        write(root / "controller.json", state)
    except Exception as error:
        state.update(status="failed", error=str(error), updated_at_utc=now())
        write(root / "controller.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "episodes", "features", "model", "output", "gpu"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-gpu", action="append", default=[],
                        help="Additional authorized UUID; choose one free GPU once, never launch duplicate runs")
    parser.add_argument("--through", choices=("pilot", "complete"), default="complete")
    parser.add_argument("--index-backend", choices=("ivf_gpu", "hnsw"), default="ivf_gpu")
    main(parser.parse_args())
