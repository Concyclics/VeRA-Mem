"""Lock a dev-only writer choice and emit fixed, local follow-up job plans.

No SSH, torch, training, or confirmation-result reading is performed. First run:
  python scripts/plan_interface_followup.py --runs-root ../runs/xtrah100 \
      --plans-root ../plans --queues 3
After all six step-3 training jobs have completed and been backed up locally:
  python scripts/plan_interface_followup.py --runs-root ../runs/xtrah100 \
      --plans-root ../plans --queues 3 --phase confirm

The second command creates step-3 confirmation plans only after checking the
six completed training artifacts. Existing differing selections/plans are never
overwritten. It does not evaluate a go/no-go threshold or generate step 4.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re


PROTOCOL = "interface-followup-plan-v1"
SEEDS = (42, 43, 44)
WRITERS = {"last": ("last_token", 0), "mean": ("masked_mean", 0), "mlp": ("last_token", 256)}
REMOTE = "/ssd3/chenhan/VeRA-Mem-Workspace"
STAGE3_SCALE = 71.015625


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_value(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256: " + name)
    return value


def immutable_outputs(outputs):
    """Precheck the entire output set before creating any new artifact."""
    for path, value in outputs.items():
        if path.exists():
            require(read_json(path) == value, "Refusing to overwrite different locked selection/plan: " + str(path))
    for path, value in outputs.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                handle.write(serialized(value))


def suite_job(runs_root, suite_name, job_name):
    directory = runs_root / suite_name
    suite = read_json(directory / "suite.json")
    require(suite.get("protocol") == "interface-experiment-v1", "Wrong suite protocol: " + suite_name)
    jobs = [job for job in suite["jobs"] if job["name"] == job_name]
    require(len(jobs) == 1 and jobs[0].get("status") == "complete" and jobs[0].get("exit_code") == 0,
            "Required job not complete: " + suite_name + "/" + job_name)
    run = directory / job_name
    manifest = read_json(run / "manifest.json")
    require(manifest.get("complete") is True and manifest.get("protocol") == "memory-interface-v1",
            "Incomplete or wrong job manifest: " + str(run))
    require(manifest.get("backbone_unchanged") is True, "Missing frozen-backbone verification: " + str(run))
    return run, manifest


def paired_fraction(metrics, phase, method):
    phase_data = metrics["phases"][phase]
    summary = phase_data["methods"][method]
    count, correct = summary["count"], summary["both_correct"]
    require(type(count) is int and count == 32 and type(correct) is int and 0 <= correct <= count,
            "Writer selection requires 32 paired development targets")
    require(phase_data["count"] == count and phase_data["bank_records"] == 128, "Inconsistent development denominator")
    score = summary["paired_switch_em"]
    require(type(score) in (float, int) and math.isfinite(score) and math.isclose(score, correct / count, abs_tol=1e-12),
            "Paired development score differs from integer numerator/denominator")
    return Fraction(correct, count)


def select_writer(runs_root, tag, remote_workspace):
    """Read exactly the nine writer dev runs and their training provenance."""
    evidence, ranked, comparable, original_checkpoints, train_caches = [], [], [], set(), set()
    schedule_by_seed = {seed: set() for seed in SEEDS}
    for complexity, (writer, (mode, mlp)) in enumerate(WRITERS.items()):
        scores = []
        for seed in SEEDS:
            suite = f"interface_writer_{'a' if seed == 42 else 'b'}_{tag}"
            train_name, eval_name = f"{writer}_{seed}", f"{writer}_{seed}_dev"
            train, tm = suite_job(runs_root, suite, train_name)
            evaluation, em = suite_job(runs_root, suite, eval_name)
            tc, ec = tm["configuration"], em["configuration"]
            status = read_json(train / "training_status.json")
            metrics = read_json(evaluation / "metrics.json")
            assignments = read_json(evaluation / "assignments.json")
            require(tc.get("stage") == "writer" and tc.get("seed") == seed and tc.get("updates") == 2000,
                    "Wrong fixed writer budget/seed")
            require(tc.get("writer") == mode and tc.get("mlp") == mlp and not tc.get("learned_b"),
                    "Wrong writer architecture")
            require(status.get("complete") is True and tm.get("result") == status and status["updates"] == 2000,
                    "Writer status/manifest mismatch")
            require(status["configuration"]["batch_size"] == 128 and status["configuration"]["learning_rate"] == 1e-4
                    and status["configuration"]["gradient_clip"] == 1. and status["training_examples"] == 256000,
                    "Writer optimizer/exposure budget differs")
            require(status["teacher_unchanged"] and status["frozen_state_unchanged"] and not status["targets_require_grad"],
                    "Writer frozen-target contract failed")
            require(status["teacher_checkpoint_sha256"] == tm["checkpoint_sha256"], "Writer teacher provenance differs")
            schedule_by_seed[seed].add(sha_value(status["sample_schedule_sha256"], "writer schedule"))
            train_caches.add(sha_value(tm["cache_sha256"], "writer train cache"))
            expected_checkpoint = f"{remote_workspace}/runs/{suite}/{train_name}/last.pt"
            require(ec.get("stage") == "eval" and ec.get("checkpoint") == expected_checkpoint,
                    "Development result is not linked to the expected seed checkpoint")
            require(ec.get("cache") == f"{remote_workspace}/runs/interface_classic_prepare_{tag}/prepare/dev.pt",
                    "Development result uses unexpected cache path")
            require(metrics.get("protocol") == "interface-cpu-vdb-eval-v1" and metrics.get("split") == "dev"
                    and metrics.get("complete") is True and em.get("result") == metrics,
                    "Selection only accepts complete development metrics")
            require(metrics["writer_mode"] == mode and metrics["bank_records"] == 128 and metrics["evaluated_facts"] == 32
                    and metrics["max_new_tokens"] == 32, "Wrong development evaluation setting")
            require(metrics["frozen_online"] and metrics["online_gradient_steps"] == 0
                    and metrics["shared_weights_before"] == metrics["shared_weights_after"], "Evaluation mutated shared weights")
            require(set(metrics["phases"]) == {"CC", "CH", "HC", "HH"}, "Development phases are incomplete")
            selected = assignments["selected_case_ids"]
            require(len(selected) == len(set(selected)) == 32 and len(assignments["assignments"]) == 128,
                    "Development target/bank assignment counts differ")
            comparable.append((sha_value(em["cache_sha256"], "development cache"), serialized(assignments),
                               ec["model"], metrics["max_new_tokens"]))
            original_checkpoints.add((tc["checkpoint"], sha_value(tm["checkpoint_sha256"], "original checkpoint")))
            points = tuple(paired_fraction(metrics, phase, method) for phase, method in
                           (("HC", "real"), ("HC", "canonical_key"), ("CC", "real")))
            scores.append(points)
            files = {"training_manifest": train / "manifest.json", "training_status": train / "training_status.json",
                     "development_manifest": evaluation / "manifest.json", "development_metrics": evaluation / "metrics.json",
                     "development_assignments": evaluation / "assignments.json"}
            evidence.append(dict(writer=writer, seed=seed, checkpoint=expected_checkpoint,
                                 checkpoint_sha256=sha_value(em["checkpoint_sha256"], "evaluation checkpoint"),
                                 sample_schedule_sha256=status["sample_schedule_sha256"],
                                 real_hc=float(points[0]), canonical_key_hc=float(points[1]), real_cc=float(points[2]),
                                 files={name: dict(path=str(path.relative_to(runs_root)), sha256=sha(path)) for name, path in files.items()}))
        means = tuple(sum(points[i] for points in scores) / len(SEEDS) for i in range(3))
        ranked.append((means, -complexity, writer))
    require(len(set(comparable)) == 1 and len(original_checkpoints) == 1 and len(train_caches) == 1,
            "Writer arms differ in evaluation cases, model, training cache or original checkpoint")
    require(all(len(values) == 1 for values in schedule_by_seed.values()), "Same-seed writers used different sampling schedules")
    ranked.sort(reverse=True)
    selected = ranked[0][2]
    original_path, original_hash = next(iter(original_checkpoints))
    return dict(protocol=PROTOCOL, selection_split="dev", tag=tag, selected_writer=selected,
                writer_mode=WRITERS[selected][0], value_mlp_hidden=WRITERS[selected][1], seeds=list(SEEDS),
                selection_rule=["maximize mean real HC paired EM", "maximize mean canonical-key HC paired EM",
                                "maximize mean real CC paired EM", "simpler: last-linear, mean-linear, last-MLP256"],
                gate_evaluated=False, gate_note="No deployment gate is inferred from development selection; formal confirmation is separate.",
                ranking=[dict(writer=writer, real_hc_mean=float(means[0]), canonical_key_hc_mean=float(means[1]),
                              real_cc_mean=float(means[2])) for means, _, writer in ranked],
                original_checkpoint=original_path, original_checkpoint_sha256=original_hash,
                stage3_margin_scale=STAGE3_SCALE,
                stage3_calibration_source=f"{remote_workspace}/runs/interface_extended_preflight_{tag}/calibrate/calibration.json",
                stage3_scale_scope="fixed train-only preflight calibration; no confirmation data",
                model=comparable[0][2], remote_workspace=remote_workspace, evidence=evidence)


def job(name, stage, selection, cache, checkpoint, extra=()):
    return dict(name=name, entry="interface", arguments=["--stage", stage, "--model", selection["model"],
                "--cache", cache, "--checkpoint", checkpoint, *extra])


def queue_names(prefix, queues, tag):
    return [f"interface_{prefix}_{chr(97 + i)}_{tag}" for i in range(queues)]


def training_plans(selection, queues):
    root, tag = selection["remote_workspace"], selection["tag"]
    names = queue_names("step3_train", queues, tag)
    plans = {name: [] for name in names}
    checkpoints = {row["seed"]: row["checkpoint"] for row in selection["evidence"] if row["writer"] == selection["selected_writer"]}
    for index, seed in enumerate(SEEDS):
        for arm, weight in (("coupled", "0"), ("decoupled", "0.05")):
            plans[names[index % queues]].append(job(f"{arm}_{seed}", "train", selection,
                f"{root}/runs/interface_extended_prepare_{tag}/prepare/train.pt", checkpoints[seed],
                ["--task", "extended", "--method", "base", "--updates", "512", "--seed", str(seed),
                 "--key-consistency", weight, "--scale", str(STAGE3_SCALE)]))
    return plans


def step2_plans(selection, queues):
    root, tag = selection["remote_workspace"], selection["tag"]
    names = queue_names("step2_confirm", queues, tag)
    plans = {name: [] for name in names}
    cache = f"{root}/runs/interface_classic_prepare_{tag}/prepare/confirm.pt"
    plans[names[0]].append(job("baseline_confirm", "eval", selection, cache, selection["original_checkpoint"], ["--teacher"]))
    checkpoints = {row["seed"]: row["checkpoint"] for row in selection["evidence"] if row["writer"] == selection["selected_writer"]}
    for index, seed in enumerate(SEEDS):
        queue = index if queues == 3 else min(index, 1)
        plans[names[queue]].append(job(f"{selection['selected_writer']}_{seed}_confirm", "eval", selection, cache,
                                     checkpoints[seed], ["--seed", str(seed)]))
    return plans


def confirm_plans(selection, queues, runs_root):
    """Validate all six completed training jobs before emitting any eval plan."""
    root, tag = selection["remote_workspace"], selection["tag"]
    _, preparation = suite_job(runs_root, f"interface_extended_prepare_{tag}", "prepare")
    require(preparation["configuration"].get("stage") == "prepare"
            and preparation["configuration"].get("task") == "extended"
            and preparation["result"].get("splits") == {"train": 4096, "dev": 256, "confirm": 256}
            and preparation["result"].get("entity_disjoint") is True,
            "Extended preparation must declare disjoint train4096/dev256/confirm256")
    training = training_plans(selection, queues)
    names = queue_names("step3_confirm", queues, tag)
    plans = {name: [] for name in names}
    evidence, schedules = [], {seed: set() for seed in SEEDS}
    cache_hashes = set()
    for index, (suite, jobs) in enumerate(training.items()):
        for planned in jobs:
            name = planned["name"]
            directory, manifest = suite_job(runs_root, suite, name)
            cfg = manifest["configuration"]
            status = read_json(directory / "training_status.json")
            arm, seed_text = name.rsplit("_", 1)
            seed = int(seed_text)
            source = next(row for row in selection["evidence"] if row["writer"] == selection["selected_writer"] and row["seed"] == seed)
            require(cfg.get("stage") == "train" and cfg.get("task") == "extended" and cfg.get("method") == "base"
                    and cfg.get("updates") == 512 and cfg.get("seed") == seed and not cfg.get("resume_optimizer"),
                    "Step-3 training protocol/budget mismatch")
            require(cfg.get("cache") == f"{root}/runs/interface_extended_prepare_{tag}/prepare/train.pt"
                    and cfg.get("model") == selection["model"], "Step-3 training model/cache differs")
            require(cfg.get("key_consistency") == (0. if arm == "coupled" else .05), "Wrong key-consistency control")
            require(cfg.get("scale") == STAGE3_SCALE, "Wrong fixed train-calibrated margin scale")
            require(cfg.get("checkpoint") == source["checkpoint"] and manifest["checkpoint_sha256"] == source["checkpoint_sha256"],
                    "Step-3 initialization differs from locked selected writer")
            require(status.get("complete") is True and manifest.get("result") == status
                    and status["updates"] == 512 and status["start_step"] == 0, "Incomplete step-3 training")
            logs = [json.loads(line) for line in (directory / "training.jsonl").read_text().splitlines() if line.strip()]
            require(len(logs) == 512 and [row["step"] for row in logs] == list(range(1, 513))
                    and all(len(row["sample"]["targets"]) == 8 and row["margin_scale"] == STAGE3_SCALE for row in logs),
                    "Step-3 must complete 512 batch-8 updates with the fixed calibrated scale")
            digest = hashlib.sha256()
            for row in logs:
                digest.update(json.dumps(row["sample"], sort_keys=True).encode())
            require(digest.hexdigest() == status["schedule_sha256"], "Step-3 sample schedule digest mismatch")
            schedules[seed].add(status["schedule_sha256"])
            cache_hashes.add(sha_value(manifest["cache_sha256"], "extended training cache"))
            require((directory / "last.pt").is_file(), "Back up each final training checkpoint before confirmation planning")
            extras = ["--task", "extended", "--eval-size", "256", "--seed", str(seed), "--scale", str(STAGE3_SCALE)]
            if arm == "coupled" and seed == 42:
                extras.append("--teacher")
            plans[names[index]].append(job(name + "_confirm", "eval", selection,
                f"{root}/runs/interface_extended_prepare_{tag}/prepare/confirm.pt",
                f"{root}/runs/{suite}/{name}/last.pt", extras))
            evidence.append(dict(run=suite + "/" + name, checkpoint_sha256=sha(directory / "last.pt"),
                                 training_manifest_sha256=sha(directory / "manifest.json"),
                                 training_status_sha256=sha(directory / "training_status.json"),
                                 sample_schedule_sha256=status["schedule_sha256"]))
    require(len(cache_hashes) == 1 and all(len(values) == 1 for values in schedules.values()),
            "Coupled/decoupled controls differ in training cache or same-seed schedule")
    return plans, dict(protocol=PROTOCOL, all_six_training_jobs_complete=True, updates=512, batch_size=8,
                       confirmation_facts=256, confirmation_entities=64, evidence=evidence)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--plans-root", type=Path)
    parser.add_argument("--queues", type=int, choices=(2, 3), default=2)
    parser.add_argument("--tag", default="20261006")
    parser.add_argument("--remote-workspace", default=REMOTE)
    parser.add_argument("--phase", choices=("plan", "confirm"), default="plan")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.runs_root is None or args.plans_root is None:
        parser.error("--runs-root and --plans-root are required")
    require(re.fullmatch(r"[0-9]{8}", args.tag), "Use an eight-digit experiment tag")
    require(args.remote_workspace.startswith("/") and not args.remote_workspace.endswith("/"), "Remote workspace must be an absolute path without trailing slash")
    runs_root, plans_root = args.runs_root.resolve(), args.plans_root.resolve()
    selection_path = plans_root / "selection.json"
    selection = select_writer(runs_root, args.tag, args.remote_workspace)
    calibration_path = runs_root / f"interface_extended_preflight_{args.tag}/calibrate/calibration.json"
    if calibration_path.exists():
        calibration = read_json(calibration_path)
        require(calibration.get("calibration_scope") == "train" and calibration.get("scale") == STAGE3_SCALE,
                "Available train calibration conflicts with the fixed stage-3 scale")
    if selection_path.exists():
        require(read_json(selection_path) == selection, "Refusing to overwrite a different locked writer selection or source evidence")
    elif args.phase == "confirm":
        raise ValueError("Lock selection with --phase plan before requesting step-3 confirmation plans")
    if args.phase == "plan":
        plans = {**step2_plans(selection, args.queues), **training_plans(selection, args.queues)}
        outputs = {selection_path: selection, **{plans_root / (name + ".json"): jobs for name, jobs in plans.items()}}
    else:
        plans, prerequisites = confirm_plans(selection, args.queues, runs_root)
        outputs = {plans_root / (name + ".json"): jobs for name, jobs in plans.items()}
        outputs[plans_root / f"interface_step3_confirmation_prerequisites_{args.tag}.json"] = prerequisites
    immutable_outputs(outputs)
    print(serialized(dict(selected_writer=selection["selected_writer"], selection=str(selection_path), phase=args.phase,
                          queues=args.queues, plans={name: [job["name"] for job in jobs] for name, jobs in plans.items()},
                          stage4_generated=False, remote_actions=False)))
    return 0


def self_test():
    """Dependency-free plan/immutability tests; all files stay in a temp tree."""
    import tempfile
    import unittest

    class Checks(unittest.TestCase):
        def write(self, path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialized(value))

        def fixture_runs(self, root, scores=None):
            scores = scores or {"last": (12, 16, 24), "mean": (16, 16, 24), "mlp": (16, 20, 24)}
            assignments = dict(selected_case_ids=[str(i) for i in range(32)],
                               assignments=[dict(id=str(i), entity=f"e{i}", relation="memory", q=1, s=2) for i in range(128)])
            for writer, (mode, mlp) in WRITERS.items():
                for seed in SEEDS:
                    suite = f"interface_writer_{'a' if seed == 42 else 'b'}_20261006"
                    directory = root / suite
                    suite_path = directory / "suite.json"
                    meta = read_json(suite_path) if suite_path.exists() else dict(protocol="interface-experiment-v1", jobs=[])
                    train_name, eval_name = f"{writer}_{seed}", f"{writer}_{seed}_dev"
                    status = dict(complete=True, updates=2000, configuration=dict(batch_size=128, learning_rate=1e-4, gradient_clip=1.),
                                  training_examples=256000, teacher_unchanged=True, frozen_state_unchanged=True,
                                  targets_require_grad=False, teacher_checkpoint_sha256="a" * 64,
                                  sample_schedule_sha256=hashlib.sha256(str(seed).encode()).hexdigest())
                    tc = dict(stage="writer", seed=seed, updates=2000, writer=mode, mlp=mlp,
                              checkpoint=REMOTE + "/old/last.pt")
                    tm = dict(protocol="memory-interface-v1", complete=True, backbone_unchanged=True, configuration=tc,
                              result=status, checkpoint_sha256="a" * 64, cache_sha256="b" * 64)
                    hc, kc, cc = scores[writer]
                    def score(correct):
                        return dict(count=32, both_correct=correct, paired_switch_em=correct / 32)
                    metrics = dict(protocol="interface-cpu-vdb-eval-v1", split="dev", complete=True, writer_mode=mode,
                                   bank_records=128, evaluated_facts=32, max_new_tokens=32, frozen_online=True,
                                   online_gradient_steps=0, shared_weights_before="d" * 64, shared_weights_after="d" * 64,
                                   phases={phase: dict(count=32, bank_records=128, methods={"real": score(cc if phase == "CC" else hc),
                                                                                       "canonical_key": score(kc)}) for phase in ("CC", "CH", "HC", "HH")})
                    ec = dict(stage="eval", checkpoint=f"{REMOTE}/runs/{suite}/{train_name}/last.pt",
                              cache=f"{REMOTE}/runs/interface_classic_prepare_20261006/prepare/dev.pt", model=REMOTE + "/model")
                    em = dict(protocol="memory-interface-v1", complete=True, backbone_unchanged=True, configuration=ec,
                              result=metrics, checkpoint_sha256="e" * 64, cache_sha256="c" * 64)
                    for path, value in ((directory/train_name/"manifest.json", tm), (directory/train_name/"training_status.json", status),
                                        (directory/eval_name/"manifest.json", em), (directory/eval_name/"metrics.json", metrics),
                                        (directory/eval_name/"assignments.json", assignments)):
                        self.write(path, value)
                    meta["jobs"].extend(dict(name=name, status="complete", exit_code=0) for name in (train_name, eval_name))
                    self.write(suite_path, meta)

        def selection(self):
            return dict(tag="20261006", remote_workspace=REMOTE, model=REMOTE + "/models/Qwen3-4B-Instruct-2507",
                        selected_writer="mean", original_checkpoint=REMOTE + "/old/last.pt",
                        evidence=[dict(writer="mean", seed=seed, checkpoint=f"{REMOTE}/seed{seed}/last.pt") for seed in SEEDS])

        def test_nine_run_selection_uses_secondary_then_tertiary_and_simple_tie(self):
            for scores, expected in ((None, "mlp"),
                                     ({"last": (16, 20, 23), "mean": (16, 20, 24), "mlp": (16, 20, 24)}, "mean"),
                                     ({writer: (16, 20, 24) for writer in WRITERS}, "last")):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    self.fixture_runs(root, scores)
                    result = select_writer(root, "20261006", REMOTE)
                    self.assertEqual(result["selected_writer"], expected)
                    self.assertEqual(len(result["evidence"]), 9)
                    self.assertFalse(result["gate_evaluated"])
                    self.assertEqual(result, select_writer(root, "20261006", REMOTE))

        def test_selection_refuses_incomplete_dev_and_different_same_seed_schedule(self):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.fixture_runs(root)
                path = root / "interface_writer_b_20261006/mlp_44_dev/manifest.json"
                original = read_json(path)
                self.write(path, dict(original, complete=False))
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    select_writer(root, "20261006", REMOTE)
                self.write(path, original)
                path = root / "interface_writer_b_20261006/mlp_44/training_status.json"
                status = read_json(path)
                status["sample_schedule_sha256"] = "f" * 64
                self.write(path, status)
                manifest_path = path.parent / "manifest.json"
                manifest = read_json(manifest_path)
                manifest["result"] = status
                self.write(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, "different sampling"):
                    select_writer(root, "20261006", REMOTE)

        def test_confirmation_planning_cannot_bypass_six_training_completion(self):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                prep = root / "interface_extended_prepare_20261006"
                self.write(prep / "suite.json", dict(protocol="interface-experiment-v1", jobs=[dict(name="prepare", status="complete", exit_code=0)]))
                self.write(prep / "prepare/manifest.json", dict(protocol="memory-interface-v1", complete=True, backbone_unchanged=True,
                           configuration=dict(stage="prepare", task="extended"),
                           result=dict(splits=dict(train=4096, dev=256, confirm=256), entity_disjoint=True)))
                with self.assertRaises(FileNotFoundError):
                    confirm_plans(self.selection(), 3, root)

        def test_completed_six_job_fixture_creates_only_full_extended_confirm_plans(self):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                selection = self.selection()
                for evidence in selection["evidence"]:
                    evidence["checkpoint_sha256"] = "a" * 64
                prep = root / "interface_extended_prepare_20261006"
                self.write(prep / "suite.json", dict(protocol="interface-experiment-v1", jobs=[dict(name="prepare", status="complete", exit_code=0)]))
                self.write(prep / "prepare/manifest.json", dict(protocol="memory-interface-v1", complete=True, backbone_unchanged=True,
                           configuration=dict(stage="prepare", task="extended"),
                           result=dict(splits=dict(train=4096, dev=256, confirm=256), entity_disjoint=True)))
                for suite, jobs in training_plans(selection, 3).items():
                    self.write(root / suite / "suite.json", dict(protocol="interface-experiment-v1",
                               jobs=[dict(name=j["name"], status="complete", exit_code=0) for j in jobs]))
                    for planned in jobs:
                        name = planned["name"]
                        directory = root / suite / name
                        arm, seed = name.rsplit("_", 1)
                        arguments = planned["arguments"]
                        cfg = {arguments[i][2:].replace("-", "_"): arguments[i + 1] for i in range(0, len(arguments), 2)}
                        cfg.update(seed=int(seed), updates=512, key_consistency=0. if arm == "coupled" else .05,
                                   scale=STAGE3_SCALE, resume_optimizer=False)
                        rows = [dict(step=step, margin_scale=STAGE3_SCALE, sample=dict(targets=list(range(8)), seed=int(seed))) for step in range(1, 513)]
                        digest = hashlib.sha256()
                        for row in rows:
                            digest.update(json.dumps(row["sample"], sort_keys=True).encode())
                        status = dict(complete=True, updates=512, start_step=0, schedule_sha256=digest.hexdigest())
                        self.write(directory / "manifest.json", dict(protocol="memory-interface-v1", complete=True, backbone_unchanged=True,
                                   configuration=cfg, result=status, checkpoint_sha256="a" * 64, cache_sha256="b" * 64))
                        self.write(directory / "training_status.json", status)
                        (directory / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                        (directory / "last.pt").write_bytes(b"fixture-checkpoint")
                plans, evidence = confirm_plans(selection, 3, root)
                self.assertTrue(evidence["all_six_training_jobs_complete"])
                self.assertEqual(evidence["confirmation_facts"], 256)
                self.assertEqual(len(evidence["evidence"]), 6)
                jobs = [job for queue in plans.values() for job in queue]
                self.assertEqual(len(jobs), 6)
                self.assertEqual(sum("--teacher" in job["arguments"] for job in jobs), 1)
                for planned in jobs:
                    arguments = planned["arguments"]
                    for key, value in (("--stage", "eval"), ("--task", "extended"), ("--eval-size", "256"), ("--scale", "71.015625")):
                        self.assertEqual(arguments[arguments.index(key) + 1], value)
                    self.assertNotIn("--max-cases", arguments)
                broken = root / "interface_step3_train_c_20261006/decoupled_44/manifest.json"
                self.write(broken, dict(read_json(broken), complete=False))
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    confirm_plans(selection, 3, root)

        def test_seed_pairs_share_queue_and_six_fixed_jobs(self):
            for queues in (2, 3):
                plans = training_plans(self.selection(), queues)
                self.assertEqual(len(plans), queues)
                self.assertEqual(sum(map(len, plans.values())), 6)
                for jobs in plans.values():
                    for seed in SEEDS:
                        names = {j["name"] for j in jobs if j["name"].endswith(str(seed))}
                        self.assertIn(names, (set(), {f"coupled_{seed}", f"decoupled_{seed}"}))
                    for job in jobs:
                        args = job["arguments"]
                        self.assertEqual(args[args.index("--updates") + 1], "512")
                        self.assertEqual(args[args.index("--task") + 1], "extended")
                        self.assertEqual(args[args.index("--scale") + 1], "71.015625")
                        self.assertNotIn("--resume-optimizer", args)

        def test_step2_has_one_original_teacher_and_three_selected_seeds(self):
            for queues in (2, 3):
                plans = step2_plans(self.selection(), queues)
                jobs = [j for js in plans.values() for j in js]
                self.assertEqual(len(jobs), 4)
                self.assertEqual(sum("--teacher" in j["arguments"] for j in jobs), 1)
                self.assertTrue(all("confirm.pt" in " ".join(j["arguments"]) for j in jobs))
                self.assertTrue(all("--max-cases" not in j["arguments"] for j in jobs))

        def test_immutable_precheck_does_not_partially_write_changed_outputs(self):
            with tempfile.TemporaryDirectory() as temp:
                a, b = Path(temp) / "selection.json", Path(temp) / "plan.json"
                immutable_outputs({a: {"writer": "last"}})
                immutable_outputs({a: {"writer": "last"}})
                with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                    immutable_outputs({b: [], a: {"writer": "mean"}})
                self.assertFalse(b.exists())

        def test_invalid_metric_denominator_or_rate_cannot_select_writer(self):
            metrics = dict(phases={"HC": dict(count=32, bank_records=128, methods={"real": dict(count=32, both_correct=16, paired_switch_em=.5)})})
            self.assertEqual(paired_fraction(metrics, "HC", "real"), Fraction(1, 2))
            metrics["phases"]["HC"]["methods"]["real"]["paired_switch_em"] = .7
            with self.assertRaises(ValueError):
                paired_fraction(metrics, "HC", "real")

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
