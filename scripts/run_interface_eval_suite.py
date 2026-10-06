"""Run independent, read-only interface evaluations, at most two per GPU.

The plan format and suite manifest match run_interface_suite.py. Only
``entry=interface`` with ``--stage eval`` is accepted; every model, checkpoint,
and cache must exist before launch. One invocation owns one explicit GPU UUID.
Occupancy checks are observations, not an exclusive GPU reservation. No process
is ever terminated by this runner: after any failure it stops dispatching and
waits for its already-started evaluations to finish naturally.

Offline checks: ``python scripts/run_interface_eval_suite.py --self-test``.
Add ``--validate-only`` to normal arguments to validate inputs without consulting
the GPU, snapshotting sources, or starting a process.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from run_generalization_suite import atomic_json, sha256, utc_now


PROTOCOL = "interface-experiment-v1"
INITIAL_MEMORY_LIMIT_MIB = 1000
RUNNING_MEMORY_LIMIT_MIB = 60000


def _name(value: str) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).name == value and value not in (".", "..")


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def validate_plan(plan: list) -> list[dict]:
    """Validate every job before any output or child process is created."""
    if not isinstance(plan, list) or not plan:
        raise ValueError("Plan must be a nonempty list")
    parser = _ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--stage", choices=("eval",), required=True)
    for field in ("model", "cache", "checkpoint"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--task", choices=("classic", "extended"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--eval-size", type=int)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--key-consistency", type=float)
    parser.add_argument("--method", choices=("base", "clip", "normalized", "hidden", "on_policy"))
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--writer", choices=("last_token", "masked_mean"))
    parser.add_argument("--learned-b", action="store_true", default=None)
    parser.add_argument("--mlp", type=int)
    names, validated = set(), []
    for job in plan:
        if not isinstance(job, dict) or not _name(job.get("name")) or job["name"] in names:
            raise ValueError("Each planned job must have a unique single-component name")
        names.add(job["name"])
        if job.get("entry") != "interface":
            raise ValueError("This runner accepts only entry='interface'")
        arguments = job.get("arguments")
        if not isinstance(arguments, list) or not arguments or any(not isinstance(x, str) for x in arguments):
            raise ValueError("Job arguments must be a nonempty string list")
        seen = set()
        for token in arguments:
            if token.startswith("--"):
                option = token.split("=", 1)[0]
                if option in seen:
                    raise ValueError(f"Duplicate option in {job['name']}: {option}")
                seen.add(option)
        # In particular --run-dir, --updates and --resume-optimizer are forbidden.
        cfg = parser.parse_args(arguments)
        if cfg.max_cases is not None and cfg.max_cases < 1:
            raise ValueError("--max-cases must be positive")
        for field in ("eval_size", "train_size"):
            if getattr(cfg, field) is not None and getattr(cfg, field) < 1:
                raise ValueError(f"--{field.replace('_', '-')} must be positive")
        if cfg.seed is not None and not 0 <= cfg.seed <= 2**63 - 1:
            raise ValueError("--seed must be a nonnegative signed 64-bit integer")
        if cfg.scale is not None and (not math.isfinite(cfg.scale) or cfg.scale <= 0):
            raise ValueError("--scale must be finite and positive")
        if cfg.key_consistency is not None and (not math.isfinite(cfg.key_consistency) or cfg.key_consistency < 0):
            raise ValueError("--key-consistency must be finite and nonnegative")
        if cfg.mlp is not None and cfg.mlp < 0:
            raise ValueError("--mlp must be nonnegative")
        inputs, requested_paths = {}, {}
        for field in ("model", "cache", "checkpoint"):
            path = Path(getattr(cfg, field))
            if not path.is_absolute():
                raise ValueError(f"{field} must use an absolute, already-existing path")
            requested_paths[field] = str(path)
            path = path.resolve(strict=True)
            if not (path.is_dir() if field == "model" else path.is_file()):
                raise ValueError(f"Invalid {field} path: {path}")
            inputs[field] = str(path)
        # Match interface_run's manifest lookup beside the supplied model path,
        # including a possible model-directory symlink.
        model_manifest = Path(requested_paths["model"]).parent / "manifest.json"
        if not model_manifest.is_file():
            raise ValueError(f"Model revision manifest is missing: {model_manifest}")
        inputs["model_manifest"] = str(model_manifest.resolve())
        requested_paths["model_manifest"] = str(model_manifest)
        validated.append(dict(name=job["name"], arguments=list(arguments), inputs=inputs,
                              input_requested_paths=requested_paths))
    return validated


def probe_gpu(gpu: str) -> dict:
    """Read exact GPU UUID/memory and all compute-app PIDs on that GPU."""
    output = subprocess.check_output(
        ["nvidia-smi", "--id=" + gpu, "--query-gpu=uuid,memory.used",
         "--format=csv,noheader,nounits"], text=True, timeout=30)
    rows = [row for row in csv.reader(output.splitlines()) if row]
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][0].strip() != gpu:
        raise RuntimeError("nvidia-smi did not identify exactly the requested full GPU UUID")
    used = int(rows[0][1].strip())
    if used < 0:
        raise RuntimeError("Invalid GPU memory measurement")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, timeout=30)
    pids = set()
    for row in csv.reader(apps.splitlines()):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 2:
            raise RuntimeError("Unrecognized nvidia-smi compute-process record")
        if row[0].strip() == gpu:
            pid = int(row[1].strip())
            if pid <= 0:
                raise RuntimeError("Invalid compute-process PID")
            pids.add(pid)
    return dict(gpu_uuid=gpu, memory_used_mib=used, compute_pids=sorted(pids), observed_at=utc_now())


def check_gpu(snapshot: dict, live_owned_pids: set[int], *, initial: bool) -> None:
    pids = set(snapshot["compute_pids"])
    limit = INITIAL_MEMORY_LIMIT_MIB if initial else RUNNING_MEMORY_LIMIT_MIB
    if initial and pids:
        raise RuntimeError(f"Selected GPU already has compute processes: {sorted(pids)}")
    foreign = pids - live_owned_pids
    if foreign:
        raise RuntimeError(f"GPU has processes outside this runner's live children: {sorted(foreign)}")
    used = snapshot["memory_used_mib"]
    if used < 0 or used >= limit:
        raise RuntimeError(f"GPU uses {used} MiB; launch requires less than {limit} MiB")


def input_metadata(validated: list[dict]) -> dict:
    result = {}
    for job in validated:
        for key in ("cache", "checkpoint", "model_manifest"):
            path = Path(job["inputs"][key])
            if str(path) not in result:
                before = path.stat()
                digest = sha256(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Input changed while hashing: {path}")
                result[str(path)] = dict(sha256=digest, bytes=after.st_size, mtime_ns=after.st_mtime_ns)
    return result


def _check_inputs(job: dict, manifest: dict) -> None:
    requested = job.get("input_requested_paths", job["inputs"])
    for field, expected in job["inputs"].items():
        if str(Path(requested[field]).resolve(strict=True)) != expected:
            raise RuntimeError(f"Input path/symlink target changed before launch: {requested[field]}")
    for field in ("cache", "checkpoint", "model_manifest"):
        path = Path(job["inputs"][field])
        current, expected = path.stat(), manifest["input_files"][str(path)]
        if not path.is_file() or (current.st_size, current.st_mtime_ns) != (expected["bytes"], expected["mtime_ns"]):
            raise RuntimeError(f"Read-only input disappeared or changed before launch: {path}")
    if not Path(job["inputs"]["model"]).is_dir():
        raise RuntimeError("Model directory disappeared before launch")


@dataclass
class _Child:
    process: object
    record: dict
    log: object
    started: float


def run_jobs(manifest, persist, gpu, env, source, *, workers=2, poll_seconds=1.,
             probe=probe_gpu, popen=subprocess.Popen, sleep=time.sleep,
             monotonic=time.monotonic, stop_requested=lambda: None) -> int:
    """Bounded scheduler; exceptions and failed jobs stop dispatch, never kill.

    Injectable process/GPU/clock hooks permit a fully offline lifecycle test.
    PIDs are authorized only while their own Popen object still polls as alive.
    """
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("At most four evaluation workers may share the GPU")
    active: list[_Child] = []
    next_job = 0
    failure = False
    started_any = False
    persistence_failed = False
    began = monotonic()
    manifest.setdefault("failures", [])

    def fail(error):
        nonlocal failure
        failure = True
        message = str(error)
        if message not in [item["error"] for item in manifest["failures"]]:
            manifest["failures"].append(dict(error=message, at=utc_now()))
        manifest.update(status="draining" if active else "failed", complete=False)

    def save():
        nonlocal persistence_failed
        if persistence_failed:
            return
        try:
            persist()
        except BaseException as error:
            persistence_failed = True
            fail(f"Manifest persistence failed: {error!r}")
            print(manifest["failures"][-1]["error"], file=sys.stderr, flush=True)

    def reap():
        for child in list(active):
            code = child.process.poll()
            if code is None:
                continue
            child.log.close()
            child.record.update(status="complete" if code == 0 else "failed", exit_code=code,
                                elapsed_seconds=monotonic() - child.started, finished_at=utc_now())
            active.remove(child)
            print(f"FINISH {child.record['name']} pid={child.process.pid} exit={code}", flush=True)
            if code != 0:
                fail(f"Evaluation {child.record['name']} failed with exit {code}")
            save()

    try:
        first = probe(gpu)
        check_gpu(first, set(), initial=True)
        manifest.update(initial_gpu_snapshot=first, status="running")
        save()
    except BaseException as error:
        fail(repr(error))
        save()

    while active or (not failure and next_job < len(manifest["jobs"])):
        try:
            reap()
            requested = stop_requested()
            if requested:
                fail(requested)
            while not failure and len(active) < workers and next_job < len(manifest["jobs"]):
                job = manifest["jobs"][next_job]
                _check_inputs(job, manifest)
                snapshot = probe(gpu)
                # Query after the probe as well: a historical/exited child PID
                # is not permission to share with a newly reused process ID.
                owned = {child.process.pid for child in active if child.process.poll() is None}
                check_gpu(snapshot, owned, initial=not started_any)
                requested = stop_requested()
                if requested:
                    fail(requested)
                    break
                job.update(status="starting", gpu_memory_used_mib_before_start=snapshot["memory_used_mib"],
                           gpu_snapshot_before_start=snapshot, started_at=utc_now())
                save()
                if failure:
                    break
                log = Path(job["log"]).open("x")
                start = monotonic()
                try:
                    process = popen(job["command"], env=env, cwd=source, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    job.update(status="failed", finished_at=utc_now(), elapsed_seconds=monotonic() - start)
                    raise
                # Register immediately; all failure paths subsequently drain it.
                active.append(_Child(process, job, log, start))
                job.update(pid=process.pid, status="running")
                started_any = True
                next_job += 1
                print(f"START {job['name']} pid={process.pid}", flush=True)
                save()
                # A fast child failure must stop the next fill in this pass.
                reap()
            if active:
                sleep(poll_seconds)
        except BaseException as error:
            fail(repr(error))
            save()
            # The loop continues with dispatch disabled, polling only owned
            # evaluations until they exit naturally. No terminate/kill call.
            if active:
                try:
                    sleep(poll_seconds)
                except BaseException as wait_error:
                    fail(repr(wait_error))

    for job in manifest["jobs"]:
        if job["status"] in ("pending", "starting"):
            job.update(status="not_started", reason="Suite stopped dispatching after a failure or stop request")
    manifest.update(complete=not failure, status="failed" if failure else "complete",
                    elapsed_seconds=monotonic() - began)
    manifest["failed_at" if failure else "completed_at"] = utc_now()
    save()
    return 1 if failure or persistence_failed else 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--self-test"]:
        return self_test()
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--shared-root", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    p.add_argument("--poll-seconds", type=float, default=1.)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args(argv)
    if not _name(a.name):
        p.error("Use one new suite directory name")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", a.gpu):
        p.error("Select one complete GPU UUID")
    if not 0 < a.poll_seconds <= 10:
        p.error("--poll-seconds must lie in (0, 10]")
    w, shared = a.workspace.resolve(), a.shared_root.resolve()
    repository, suite = w / "VeRA-Mem", w / "runs" / a.name
    try:
        plan = json.loads(a.plan.read_text())
        validated = validate_plan(plan)
        if suite.exists():
            raise FileExistsError(f"Suite already exists: {suite}")
        for folder in ("src", "scripts", "tests", "configs", "docs"):
            if not (repository / folder).is_dir():
                raise ValueError(f"Repository directory missing: {folder}")
        for filename in ("README.md", "pyproject.toml"):
            if not (repository / filename).is_file():
                raise ValueError(f"Repository file missing: {filename}")
        if not (shared / "env_deps").is_dir():
            raise ValueError("Shared dependency directory is missing")
        if shutil.disk_usage(w).free < 5 * 1024**3:
            raise ValueError("At least 5 GiB user-available workspace space is required")
        inputs = input_metadata(validated)
    except (OSError, ValueError) as error:
        p.error(str(error))
    if a.validate_only:
        print(json.dumps(dict(valid=True, read_only_evaluation_jobs=len(validated), workers=a.workers,
                              plan_sha256=sha256(a.plan), inputs=inputs)), flush=True)
        return 0

    suite.mkdir(parents=True, exist_ok=False)
    source = suite / "source"
    manifest = dict(protocol=PROTOCOL, complete=False, status="preparing", started_at=utc_now(),
                    launcher_pid=os.getpid(), gpu_uuid=a.gpu, workers=a.workers,
                    execution_mode="parallel_independent_read_only_evaluations",
                    plan_sha256=sha256(a.plan), input_files=inputs, jobs=[])
    persist = lambda: atomic_json(suite / "suite.json", manifest)
    persist()
    requested_stop = {"reason": None}
    previous_handlers = {}
    try:
        # Freeze all code before starting any independent evaluation.
        for folder in ("src", "scripts", "tests", "configs", "docs"):
            shutil.copytree(repository / folder, source / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        for filename in ("README.md", "pyproject.toml"):
            shutil.copy2(repository / filename, source / filename)
        atomic_json(suite / "plan.json", plan)
        manifest["source_files_sha256"] = {str(f.relative_to(source)): sha256(f) for f in sorted(source.rglob("*")) if f.is_file()}
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   TOKENIZERS_PARALLELISM="false", HF_HOME=str(w / "cache/huggingface"),
                   PYTHONPATH=str(shared / "env_deps") + os.pathsep + str(source / "src"))
        manifest["runtime_environment"] = {k: env[k] for k in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "TOKENIZERS_PARALLELISM", "HF_HOME", "PYTHONPATH")}
        for job in validated:
            output = suite / job["name"]
            command = [sys.executable, "-u", "-m", "vera_mem.interface_run", *job["arguments"], "--run-dir", str(output)]
            manifest["jobs"].append(dict(name=job["name"], command=command, output=str(output),
                                         log=str(suite / (job["name"] + ".log")), inputs=job["inputs"],
                                         input_requested_paths=job["input_requested_paths"],
                                         status="pending", exit_code=None, pid=None))
        persist()
        def request_stop(signum, _frame):
            requested_stop["reason"] = f"Received signal {signum}; draining owned evaluations without termination"
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, request_stop)
        return run_jobs(manifest, persist, a.gpu, env, source, workers=a.workers,
                        poll_seconds=a.poll_seconds, stop_requested=lambda: requested_stop["reason"])
    except BaseException as error:
        # Child lifecycles are handled inside run_jobs. This covers preparation,
        # where no child has been started, or a final persistence failure.
        manifest.update(status="failed", complete=False, error=repr(error), failed_at=utc_now())
        persist()
        raise
    finally:
        for signum, old in previous_handlers.items():
            signal.signal(signum, old)


def self_test() -> int:
    """Offline tests use synthetic PIDs, a fake GPU probe and a fake clock."""
    import tempfile
    import unittest

    class Clock:
        now = 0.
        def __call__(self):
            return self.now
        def sleep(self, seconds):
            self.now += seconds

    class Child:
        def __init__(self, pid, clock, duration, code):
            self.pid, self.clock, self.end, self.code = pid, clock, clock.now + duration, code
        def poll(self):
            return self.code if self.clock.now >= self.end else None
        def terminate(self):
            raise AssertionError("Must never terminate a process")
        kill = terminate

    class Tests(unittest.TestCase):
        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.addCleanup(self.tmp.cleanup)
            self.path = Path(self.tmp.name)
            model = self.path / "models/model"
            model.mkdir(parents=True)
            (model.parent / "manifest.json").write_text('{}')
            self.cache, self.checkpoint = self.path / "cache.pt", self.path / "checkpoint.pt"
            self.cache.write_bytes(b"cache")
            self.checkpoint.write_bytes(b"checkpoint")
            self.arguments = ["--stage", "eval", "--model", str(model), "--cache", str(self.cache),
                              "--checkpoint", str(self.checkpoint), "--teacher"]
        def make(self, n=3):
            plan = [dict(name=f"job{i}", entry="interface", arguments=self.arguments) for i in range(n)]
            validated = validate_plan(plan)
            manifest = dict(jobs=[dict(name=j["name"], command=[j["name"]], inputs=j["inputs"],
                                      input_requested_paths=j["input_requested_paths"],
                                      log=str(self.path / (j["name"] + ".log")), status="pending", exit_code=None, pid=None)
                                 for j in validated], input_files=input_metadata(validated))
            return manifest
        def execute(self, manifest, durations=(2., 5., 1.), codes=(0, 0, 0), special_probe=None, stop=None, fail_spawn=None):
            clock, children, snapshots = Clock(), [], []
            peak = [0]
            def spawn(command, **kwargs):
                if fail_spawn is not None and len(children) == fail_spawn:
                    raise OSError("synthetic spawn failure")
                index = len(children)
                child = Child(100 + index, clock, durations[index], codes[index])
                children.append(child)
                peak[0] = max(peak[0], sum(c.poll() is None for c in children))
                return child
            def probe(_):
                pids = [c.pid for c in children if c.poll() is None]
                snapshot = dict(memory_used_mib=9000 * len(pids), compute_pids=pids)
                if special_probe:
                    special_probe(snapshot, children, clock)
                snapshots.append(snapshot.copy())
                return snapshot
            result = run_jobs(manifest, lambda: None, "GPU-test", {}, self.path,
                              probe=probe, popen=spawn, sleep=clock.sleep, monotonic=clock,
                              stop_requested=(lambda: stop(clock)) if stop else (lambda: None))
            return result, children, clock, peak[0], snapshots
        def test_validates_only_existing_independent_eval_inputs(self):
            good = dict(name="eval", entry="interface", arguments=self.arguments)
            self.assertEqual(len(validate_plan([good])), 1)
            for bad in (
                dict(good, entry="run"), dict(good, name="../escape"),
                dict(good, arguments=[*self.arguments, "--stage", "train"]),
                dict(good, arguments=[*self.arguments, "--run-dir", "/tmp/override"]),
                dict(good, arguments=[*self.arguments, "--updates", "1"]),
                dict(good, arguments=[x if x != "eval" else "train" for x in self.arguments]),
            ):
                with self.assertRaises(ValueError):
                    validate_plan([bad])
            with self.assertRaises(ValueError):
                validate_plan([good, good])
            self.checkpoint.unlink()
            with self.assertRaises(FileNotFoundError):
                validate_plan([good])
        def test_real_plan_helper_accepts_stage3_and_stage4_eval_metadata(self):
            from plan_interface_followup import job, STAGE3_SCALE
            selection = {"model": self.arguments[self.arguments.index("--model") + 1]}
            extras = [
                ["--task", "extended", "--eval-size", "256", "--seed", "42", "--scale", str(STAGE3_SCALE), "--teacher"],
                ["--task", "extended", "--seed", "42", "--method", "on_policy", "--scale", str(STAGE3_SCALE),
                 "--key-consistency", "0.05", "--eval-size", "256"],
            ]
            planned = [job(f"step{index + 3}_confirm", "eval", selection, str(self.cache), str(self.checkpoint), extra)
                       for index, extra in enumerate(extras)]
            accepted = validate_plan(planned)
            self.assertEqual([j["arguments"] for j in accepted], [j["arguments"] for j in planned])
            for arguments in (
                [*self.arguments, "--scale", "nan"], [*self.arguments, "--scale", "0"],
                [*self.arguments, "--key-consistency", "-1"], [*self.arguments, "--eval-size", "0"],
                [*self.arguments, "--method", "unknown"], [*self.arguments, "--resume-optimizer"],
            ):
                with self.assertRaises(ValueError):
                    validate_plan([dict(name="bad", entry="interface", arguments=arguments)])
        def test_success_keeps_capacity_two_and_records_individual_pids_times(self):
            manifest = self.make()
            code, children, clock, peak, _ = self.execute(manifest)
            self.assertEqual(code, 0)
            self.assertEqual(peak, 2)
            self.assertEqual(clock.now, 5.)
            self.assertTrue(manifest["complete"])
            self.assertEqual([j["pid"] for j in manifest["jobs"]], [100, 101, 102])
            self.assertEqual([j["elapsed_seconds"] for j in manifest["jobs"]], [2., 5., 1.])
            self.assertTrue(all(c.poll() == 0 for c in children))
        def test_failed_job_stops_dispatch_but_peer_finishes(self):
            manifest = self.make()
            code, children, clock, peak, _ = self.execute(manifest, codes=(7, 0, 0))
            self.assertEqual(code, 1)
            self.assertEqual(len(children), 2)
            self.assertEqual(clock.now, 5.)
            self.assertEqual([j["status"] for j in manifest["jobs"]], ["failed", "complete", "not_started"])
            self.assertEqual(manifest["jobs"][0]["exit_code"], 7)
        def test_initial_foreign_process_or_1000mib_refuses_all_launches(self):
            for field, value in (("compute_pids", [999]), ("memory_used_mib", 1000)):
                manifest = self.make()
                code, children, *_ = self.execute(manifest, special_probe=lambda snap, _c, _t: snap.update({field: value}))
                self.assertEqual(code, 1)
                self.assertEqual(children, [])
        def test_foreign_occupancy_between_launches_drains_first_without_starting_second(self):
            manifest = self.make()
            def foreign(snap, children, clock):
                if children:
                    snap["compute_pids"].append(999)
            code, children, clock, *_ = self.execute(manifest, special_probe=foreign)
            self.assertEqual(code, 1)
            self.assertEqual(len(children), 1)
            self.assertEqual(clock.now, 2.)
            self.assertEqual(manifest["jobs"][0]["status"], "complete")
        def test_memory_cap_and_dead_owned_pids_are_not_permission(self):
            with self.assertRaises(RuntimeError):
                check_gpu(dict(memory_used_mib=60000, compute_pids=[100]), {100}, initial=False)
            with self.assertRaises(RuntimeError):
                check_gpu(dict(memory_used_mib=9000, compute_pids=[99]), {100}, initial=False)
        def test_spawn_failure_and_stop_request_both_wait_for_existing_children(self):
            manifest = self.make()
            code, children, clock, *_ = self.execute(manifest, fail_spawn=1)
            self.assertEqual(code, 1)
            self.assertEqual(len(children), 1)
            self.assertEqual(clock.now, 2.)
            # Fresh paths avoid overwriting the intentionally preserved logs.
            for f in self.path.glob("*.log"):
                f.unlink()
            manifest = self.make()
            code, children, clock, *_ = self.execute(manifest, stop=lambda t: "synthetic stop" if t.now >= 1 else None)
            self.assertEqual(code, 1)
            self.assertEqual(len(children), 2)
            self.assertEqual(clock.now, 5.)
        def test_changed_input_is_rejected_before_start(self):
            manifest = self.make()
            self.cache.write_bytes(b"changed cache")
            code, children, *_ = self.execute(manifest)
            self.assertEqual(code, 1)
            self.assertEqual(children, [])
        def test_retargeted_symlink_is_not_treated_as_original_hashed_input(self):
            link = self.path / "cache-link.pt"
            link.symlink_to(self.cache)
            self.arguments[self.arguments.index("--cache") + 1] = str(link)
            manifest = self.make()
            alternative = self.path / "alternative.pt"
            alternative.write_bytes(b"other")
            link.unlink()
            link.symlink_to(alternative)
            code, children, *_ = self.execute(manifest)
            self.assertEqual(code, 1)
            self.assertEqual(children, [])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
