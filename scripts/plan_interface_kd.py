"""Plan the fixed KD continuation after writer confirmation and six warm runs.

Reads local evidence only; does not launch a process or choose a checkpoint
using confirmation scores. A failed writer gate permits only the preregistered
seed-42 diagnostic budget. The five methods always share a warm optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from plan_interface_followup import (confirm_plans, immutable_outputs, job,
    queue_names, read_json, require, serialized, sha, step2_plans, suite_job,
    training_plans, sha_value)

METHODS = ("base", "clip", "normalized", "hidden", "on_policy")


def confirmation_gate(selection, runs, queues):
    records = {}
    for suite, jobs in step2_plans(selection, queues).items():
        for entry in jobs:
            directory, manifest = suite_job(runs, suite, entry["name"])
            metrics = read_json(directory / "metrics.json")
            config = manifest["configuration"]
            arguments = entry["arguments"]
            checkpoint = arguments[arguments.index("--checkpoint") + 1]
            source = None if entry["name"] == "baseline_confirm" else next(
                row for row in selection["evidence"] if row["checkpoint"] == checkpoint)
            expected_hash = selection["original_checkpoint_sha256"] if source is None else source["checkpoint_sha256"]
            require(config.get("stage") == "eval" and config.get("task") == "classic"
                    and config.get("model") == selection["model"]
                    and config.get("checkpoint") == checkpoint and manifest["checkpoint_sha256"] == expected_hash
                    and config.get("cache") == f"{selection['remote_workspace']}/runs/interface_classic_prepare_{selection['tag']}/prepare/confirm.pt"
                    and config.get("max_cases") is None,
                    "Confirmation is not the locked writer/checkpoint and full classic cache")
            require(source is None or config.get("seed") == source["seed"], "Confirmation seed differs from locked writer")
            require(metrics.get("protocol") == "interface-cpu-vdb-eval-v1"
                    and metrics["complete"] is True and metrics["split"] == "confirm"
                    and metrics["evaluated_facts"] == metrics["bank_records"] == 128,
                    "Writer gate requires the full 128-fact confirmation")
            require(metrics["max_new_tokens"] == 32 and metrics["frozen_online"] is True
                    and metrics["online_gradient_steps"] == 0
                    and metrics["shared_weights_before"] == metrics["shared_weights_after"],
                    "Confirmation budget/frozen-state contract differs")
            require(set(metrics["phases"]) == {"CC", "CH", "HC", "HH"}, "Incomplete confirmation phase matrix")
            require(manifest["result"] == metrics, "Confirmation manifest mismatch")
            assignments = read_json(directory / "assignments.json")
            assigned = assignments["assignments"]
            selected = assignments["selected_case_ids"]
            require(len(selected) == len(set(selected)) == 128 and len(assigned) == 128
                    and len({row["id"] for row in assigned}) == 128
                    and set(selected) == {row["id"] for row in assigned}, "Gate requires all 128 unique assigned facts")
            scores = {}
            for phase in ("CC", "HC"):
                phase_data = metrics["phases"][phase]
                score = phase_data["methods"]["real"]
                require(phase_data["count"] == phase_data["bank_records"] == 128
                        and score["count"] == 128 and type(score["both_correct"]) is int
                        and 0 <= score["both_correct"] <= 128
                        and type(score["paired_switch_em"]) in (float, int) and math.isfinite(score["paired_switch_em"])
                        and math.isclose(score["paired_switch_em"], score["both_correct"] / 128),
                        "Invalid confirmation denominator")
                scores[phase] = score["both_correct"] / 128
            records[entry["name"]] = dict(scores=scores,
                run=suite+"/"+entry["name"], manifest_sha256=sha(directory/"manifest.json"),
                metrics_sha256=sha(directory/"metrics.json"), cache_sha256=sha_value(manifest["cache_sha256"], "confirmation cache"),
                checkpoint_sha256=sha_value(manifest["checkpoint_sha256"], "confirmation checkpoint"),
                assignments_sha256=sha(directory/"assignments.json"))
    require(len({r["cache_sha256"] for r in records.values()}) == 1
            and len({r["assignments_sha256"] for r in records.values()}) == 1,
            "Gate runs must evaluate identical facts/styles")
    baseline = records["baseline_confirm"]["scores"]
    decisions = []
    for seed in (42,43,44):
        candidate = records[f"{selection['selected_writer']}_{seed}_confirm"]["scores"]
        hc, cc = candidate["HC"]-baseline["HC"], candidate["CC"]-baseline["CC"]
        decisions.append(dict(seed=seed, hc_delta=hc, cc_delta=cc,
                              passed=hc >= .2-1e-12 and cc >= -.05-1e-12))
    passed = all(d["passed"] for d in decisions)
    return dict(protocol="interface-kd-plan-v1", gate_split="confirm", passed=passed,
                thresholds=dict(hc_gain_min=.2,cc_change_min=-.05), decisions=decisions,
                evidence=records, updates=512 if passed else 256,
                seeds=[42,43,44] if passed else [42],
                interpretation="three-seed comparison" if passed else "single-seed diagnostic only")


def checkpoint_evidence(path, *, step, seed, method):
    """Inspect the real shared Adam state on CPU; no backbone is loaded."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(checkpoint.get("protocol") == "memory-interface-v1" and checkpoint.get("step") == step
            and checkpoint.get("seed") == seed and checkpoint.get("method") == method
            and checkpoint.get("key_consistency") == .05, "Wrong continuation checkpoint step/seed/method")
    require("optimizer" in checkpoint, "Shared warm checkpoint is missing Adam state")
    architecture, weights = checkpoint["architecture"], checkpoint["module"]
    require(not architecture.get("train_B"), "This fixed KD protocol does not train B")
    mlp = architecture.get("value_mlp_hidden", 0)
    names = [["Wv.weight"] + (["value_mlp.0.weight", "value_mlp.2.weight"] if mlp else []),
             ["b"], ["Wq.weight", "Wk.weight"]]
    optimizer = checkpoint["optimizer"]
    groups, states = optimizer["param_groups"], optimizer["state"]
    require(len(groups) == 3, "Unexpected shared Adam parameter groups")
    digest, ids = hashlib.sha256(), []
    for group, expected, rate in zip(groups, names, (1e-4, .005, 1e-5)):
        require(len(group["params"]) == len(expected) and group["lr"] == rate
                and tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8
                and group["weight_decay"] == 0, "Shared Adam configuration differs")
        digest.update(json.dumps(group, sort_keys=True).encode())
        for identity, name in zip(group["params"], expected):
            ids.append(identity)
            require(identity in states and name in weights, "Missing shared Adam parameter state")
            state = states[identity]
            require("step" in state and float(state["step"]) == step, "Shared Adam step differs from checkpoint budget")
            digest.update(name.encode())
            for field in ("step", "exp_avg", "exp_avg_sq"):
                value = state[field]
                require(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()), "Nonfinite/malformed shared Adam state")
                if field != "step":
                    require(value.shape == weights[name].shape, "Adam moment shape differs from parameter")
                digest.update(field.encode()); digest.update(str(value.dtype).encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    require(len(ids) == len(set(ids)) and set(ids) == set(states), "Duplicate or unexpected Adam parameter state")
    return dict(checkpoint_sha256=sha(path), optimizer_sha256=digest.hexdigest(),
                optimizer_step=step, seed=seed, architecture=architecture)


def plans(selection, gate, runs, queues, scale, phase):
    require(phase in {"train", "confirm"}, "Unknown KD planning phase")
    require(gate.get("seeds") == ([42, 43, 44] if gate.get("passed") is True else [42])
            and gate.get("updates") == (512 if gate.get("passed") is True else 256), "Gate budget was altered")
    # Checks all six common warm runs and their sample schedule before KD.
    _, warm_evidence = confirm_plans(selection, queues, runs)
    root, tag = selection["remote_workspace"], selection["tag"]
    names = queue_names("step4_" + phase, queues, tag)
    outputs = {name: [] for name in names}
    sources = {}
    warm_checkpoints = {}
    for suite, entries in training_plans(selection, queues).items():
        for entry in entries:
            if entry["name"].startswith("decoupled_"):
                seed = int(entry["name"].split("_")[-1])
                sources[seed] = (suite,entry["name"])
                warm_checkpoints[seed] = checkpoint_evidence(runs/suite/entry["name"]/"last.pt", step=512, seed=seed, method="base")
    job_evidence, schedules = [], {}
    for index,(seed,method) in enumerate((s,m) for s in gate["seeds"] for m in METHODS):
        name = f"{method}_{seed}"
        suite, source = sources[seed]
        warm = f"{root}/runs/{suite}/{source}/last.pt"
        extra = ["--task","extended","--seed",str(seed),"--method",method,
                 "--scale",str(scale),"--key-consistency","0.05"]
        training_suite = queue_names("step4_train",queues,tag)[index % queues]
        if phase == "train":
            cache = f"{root}/runs/interface_extended_prepare_{tag}/prepare/train.pt"
            extra += ["--updates",str(gate["updates"]),"--resume-optimizer"]
            outputs[names[index%queues]].append(job(name,"train",selection,cache,warm,extra))
        else:
            directory, manifest = suite_job(runs,training_suite,name)
            cfg, status = manifest["configuration"],read_json(directory/"training_status.json")
            require(cfg["checkpoint"] == warm and cfg["method"] == method and cfg["seed"] == seed
                    and cfg["resume_optimizer"] and cfg["scale"] == scale
                    and cfg["key_consistency"] == .05 and cfg["updates"] == gate["updates"]
                    and cfg.get("stage") == "train" and cfg.get("task") == "extended"
                    and cfg.get("model") == selection["model"]
                    and cfg.get("cache") == f"{root}/runs/interface_extended_prepare_{tag}/prepare/train.pt"
                    and manifest["checkpoint_sha256"] == warm_checkpoints[seed]["checkpoint_sha256"],
                    "KD continuation differs from fixed plan")
            require(status["complete"] is True and manifest.get("result") == status and status["start_step"] == 512
                    and status["updates"] == gate["updates"], "Incomplete KD continuation")
            rows = [json.loads(x) for x in (directory/"training.jsonl").read_text().splitlines() if x]
            require([r["step"] for r in rows] == list(range(513,513+gate["updates"]))
                    and all(len(r["sample"]["targets"]) == 8 and r["margin_scale"] == scale for r in rows), "KD exposure mismatch")
            schedule = hashlib.sha256()
            for row in rows: schedule.update(json.dumps(row["sample"],sort_keys=True).encode())
            require(schedule.hexdigest() == status["schedule_sha256"],"KD schedule log mismatch")
            schedules.setdefault(seed,set()).add(schedule.hexdigest())
            require((directory/"last.pt").is_file(),"Back up final KD checkpoint before confirmation")
            final_evidence = checkpoint_evidence(directory/"last.pt", step=512+gate["updates"], seed=seed, method=method)
            cache = f"{root}/runs/interface_extended_prepare_{tag}/prepare/confirm.pt"
            checkpoint = f"{root}/runs/{training_suite}/{name}/last.pt"
            outputs[names[index%queues]].append(job(name+"_confirm","eval",selection,cache,checkpoint,extra+["--eval-size","256"]))
            job_evidence.append(dict(run=training_suite+"/"+name,**final_evidence,
                                     warm_checkpoint_sha256=warm_checkpoints[seed]["checkpoint_sha256"],
                                     warm_optimizer_sha256=warm_checkpoints[seed]["optimizer_sha256"],
                                     manifest_sha256=sha(directory/"manifest.json"),schedule_sha256=schedule.hexdigest()))
    require(all(len(v)==1 for v in schedules.values()),"KD arms differ in exposure schedule")
    return {k:v for k,v in outputs.items() if v}, dict(warm=warm_evidence,
        warm_checkpoints={str(seed):evidence for seed,evidence in warm_checkpoints.items()}, continuations=job_evidence)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root",type=Path,required=True)
    parser.add_argument("--plans-root",type=Path,required=True)
    parser.add_argument("--queues",type=int,choices=(2,3),default=3)
    parser.add_argument("--phase",choices=("train","confirm"),default="train")
    args=parser.parse_args()
    selection=read_json(args.plans_root/"selection.json"); tag=selection["tag"]
    gate=confirmation_gate(selection,args.runs_root,args.queues)
    calibration_path=args.runs_root/f"interface_extended_preflight_{tag}/calibrate/calibration.json"
    calibration=read_json(calibration_path)
    require(calibration["calibration_scope"] == "train" and calibration["teacher_interface_passed"],
            "Train-only teacher calibration must have passed")
    scale=calibration["scale"]
    require(math.isfinite(scale) and scale>0,"Invalid calibration scale")
    gate.update(selection_sha256=sha(args.plans_root/"selection.json"),
                calibration_sha256=sha(calibration_path),scale=scale)
    generated,evidence=plans(selection,gate,args.runs_root,args.queues,scale,args.phase)
    immutable_outputs({args.plans_root/f"interface_step4_gate_{tag}.json":gate,
        args.plans_root/f"interface_step4_{args.phase}_prerequisites_{tag}.json":evidence,
        **{args.plans_root/(name+".json"):entries for name,entries in generated.items()}})
    print(serialized(dict(gate_passed=gate["passed"],updates=gate["updates"],seeds=gate["seeds"],
                          phase=args.phase,plans={n:[j["name"] for j in v] for n,v in generated.items()})))


if __name__=="__main__": main()
