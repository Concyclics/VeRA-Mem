"""Audit local coldstart_* runs without inference, SSH or tensor deserialization.

python scripts/summarize_coldstart.py --runs-root WORKSPACE/runs/xtrah100 --output summary.json
Incomplete manifests are listed explicitly; completed malformed evidence fails.
No old interface/counterfactual runs are silently included. Counts and eligibility
are recomputed from predictions; foundation usage is recomputed from read traces.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

# Reuse the independently checked full-answer/paired-bank audit, not logged rates.
_path = Path(__file__).with_name("summarize_interface.py")
_spec = importlib.util.spec_from_file_location("_coldstart_interface_audit", _path)
ia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ia)

PROTOCOL = "coldstart-audit-v1"
RUN_PROTOCOL = "dictionary-coldstart-v1"
FOUNDATION_PROTOCOL = "coldstart-cpu-banks-v1"
FORMAL_ARMS = ("no_warm","matched_warm","wiki_warm","fixed","learned_sparse","dense_warm","straight_through","wiki_joint")


class Usage:
    """Sparse Python-only accumulation; no torch/model/cache loading."""
    def __init__(self, size):
        self.size = size
        self.top1, self.topk, self.mass = Counter(), Counter(), defaultdict(float)
        self.search_calls = self.query_positions = 0

    def add(self, row, top_k):
        ia.require(row["records"] == self.size, "Foundation read bank size differs")
        for key in ("search_calls", "query_positions"):
            ia.number(row[key], key, integer=True)
        self.search_calls += row["search_calls"]
        self.query_positions += row["query_positions"]
        counts = row["sparse_counts"]
        ia.require(len({x["index"] for x in counts}) == len(counts), "Repeated foundation slot")
        one, many, mass = 0, 0, 0.
        for x in counts:
            i = x["index"]
            ia.number(i,"slot index",integer=True)
            ia.require(i < self.size, "Foundation index out of bounds")
            ia.number(x["top1"],"top1",integer=True)
            ia.number(x["topk"],"topk",integer=True,minimum=1)
            ia.number(x["weight_mass"],"weight mass")
            ia.require(x["top1"] <= x["topk"], "Top1 exceeds top-k")
            self.top1[i] += x["top1"]; self.topk[i] += x["topk"]; self.mass[i] += x["weight_mass"]
            one += x["top1"]; many += x["topk"]; mass += x["weight_mass"]
        ia.same_number(one, row["query_positions"] if self.size else 0, "Foundation top1 denominator")
        ia.same_number(many, row["query_positions"] * min(self.size,top_k), "Foundation top-k denominator")
        ia.same_number(mass, row["query_positions"] if self.size else 0, "Foundation probability mass")
        local = Usage(self.size)
        local.search_calls, local.query_positions = row["search_calls"],row["query_positions"]
        for x in counts:
            local.top1[x["index"]] = x["top1"]; local.topk[x["index"]] = x["topk"]
            local.mass[x["index"]] = x["weight_mass"]
        check_usage(row,local.summary())

    def summary(self):
        mass = sum(self.mass.values())
        probabilities = [v/mass for v in self.mass.values() if v>0] if mass else []
        entropy = -sum(p*math.log(p) for p in probabilities)
        one, many = sum(v>0 for v in self.top1.values()),sum(v>0 for v in self.topk.values())
        return dict(search_calls=self.search_calls,query_positions=self.query_positions,records=self.size,
            top1_used=one,topk_used=many,top1_coverage=one/self.size if self.size else None,
            topk_coverage=many/self.size if self.size else None,weight_mass=mass,
            weight_entropy=entropy,effective_slots=math.exp(entropy) if mass else 0.,
            sparse_counts=[dict(index=i,top1=self.top1[i],topk=self.topk[i],weight_mass=self.mass[i]) for i in sorted(self.topk) if self.topk[i]])


def check_usage(logged,recomputed):
    for key,value in recomputed.items():
        if key == "sparse_counts":
            ia.require(len(logged[key]) == len(value), "Foundation active slot count mismatch")
            for actual,expected in zip(logged[key],value):
                ia.require(set(actual) == set(expected), "Foundation slot count fields")
                for name,v in expected.items(): ia.same_number(actual[name],v,"Foundation slot "+name)
        elif value is None:
            ia.require(logged[key] is None,"Empty-bank coverage must be null")
        else: ia.same_number(logged[key],value,"Foundation usage "+key)


def audit_foundation(directory,manifest,record):
    metrics = manifest["result"]
    c = metrics["coldstart"]
    ia.require(c["protocol"] == FOUNDATION_PROTOCOL,"Wrong foundation protocol")
    ia.require(c["base_mode"] in {"full","off","random256","cluster256"},"Invalid foundation mode")
    ia.require(c["base_mode"] == manifest["configuration"].get("base_mode","full"),"Foundation mode differs from configuration")
    ia.require(c["foundation_cpu"] and c["episodic_cpu"] and c["evaluated_routing"] == "sparse","Evaluation did not use both sparse CPU banks")
    ia.require(c["values_rms_recalibrated"] is False and c["count_bias_applied"] is False and c["dynamic_records_compressed"] is False,"Unregistered compression transformation")
    ia.require(c["bank_hash_before"] == c["bank_hash_after"],"Foundation bank mutated")
    ia.sha(c["bank_hash_before"],"Foundation bank")
    ia.require(c["original_shared_weights_before"] == c["original_shared_weights_after"],"Original foundation checkpoint mutated")
    ia.require(c["original_state_before"] == c["original_state_after"],"Original routing/grad/training state not restored")
    ia.number(c["alpha_base"],"actual checkpoint foundation weight")
    effective = c["alpha_base"] > 0 and c["active_records"] > 0
    effect_status = "weight_zero" if c["alpha_base"] == 0 else "bank_empty" if not c["active_records"] else "weighted_branch_enabled"
    weighted_top_k = min(c["base_top_k"],c["active_records"]) if effective else 0
    effects = dict(foundation_effective=effective,foundation_effect_status=effect_status,
        foundation_weighted_top_k=weighted_top_k,
        maximum_weighted_records=min(c["episodic_top_k"],metrics["bank_records"])+weighted_top_k)
    # Early smoke artifacts predate these explanatory fields; derive from the
    # actual loaded checkpoint alpha, never eval CLI defaults.
    for key,value in effects.items():
        if key in c: ia.require(c[key] == value,"Wrong foundation contribution label: "+key)
    for stem in ("foundation_snapshot","foundation_usage"):
        path = directory/c[stem]
        ia.require(path.parent == directory and path.is_file(),"Missing foundation sidecar")
        ia.require(ia.file_hash(path) == c[stem+"_sha256"],"Foundation sidecar SHA256 mismatch")
        record["artifacts"][c[stem]] = dict(path=str(path),sha256=c[stem+"_sha256"])
    predictions = ia.read_jsonl(directory/"predictions.jsonl")
    calls = ia.read_jsonl(directory/c["foundation_usage"])
    grouped, counts, total = defaultdict(list),defaultdict(lambda:Usage(c["active_records"])),Usage(c["active_records"])
    for index,row in enumerate(calls):
        ia.require(row["call_index"] == index,"Foundation call order mismatch")
        p = row["prediction_index"]
        ia.require(type(p) is int and 0 <= p < len(predictions),"Foundation prediction index out of range")
        prediction = predictions[p]
        ia.require(row["teacher"] == (prediction["method"] == "teacher"),"Foundation teacher flag differs from prediction")
        ia.require(row["bank_hash"] == c["bank_hash_before"],"Foundation call bank differs")
        ia.require(row["kind"] in {"generation","answer_scoring"},"Invalid foundation call kind")
        ia.require((row["search_calls"] == 0) if row["teacher"] else (row["search_calls"] > 0),"Teacher read foundation, or student did not use CPU foundation path")
        grouped[p].append(row["kind"])
        total.add(row,c["base_top_k"])
        label = "/".join([prediction["phase"],prediction["method"],row["kind"]])
        counts[label].add(row,c["base_top_k"])
    ia.require(set(grouped) == set(range(len(predictions))),"Foundation calls missing predictions")
    ia.require([r["prediction_index"] for r in calls if r["kind"]=="generation"] == list(range(len(predictions))),"Generation call ordering differs")
    for i,p in enumerate(predictions):
        ia.require(grouped[i] == ["generation"]+["answer_scoring"]*len(p["answer_results"]),"Missing/repeated foundation generation/scoring call")
    check_usage(c["usage"],total.summary())
    record["foundation"] = {k:v for k,v in c.items() if k != "usage"}
    record["foundation"].update(effects)
    record["foundation"]["contribution_interpretation"] = (
        "Effective means positive configured weight and a nonempty bank, not measured utility. "
        "At alpha_base=0 route coverage reflects audit-only computation and cannot be described as effective memory utilization. "
        "Top-k quotas describe potential contributors; zero-weight foundation reads still consume CPU search/transfer time included in generation timing.")
    record["foundation"]["usage"] = total.summary()
    record["foundation"]["usage_by_phase_method_call"] = {k:v.summary() for k,v in sorted(counts.items())}
    record["foundation"]["snapshot_validation"] = "Snapshot SHA256 checked; no tensor loading or replay. Usage recomputed from CPU read sidecar."
    return record


def audit_training(directory,manifest):
    for name in ("training_status.json","training.jsonl","usage.pt","last.pt"):
        ia.require((directory/name).is_file(),"Missing completed training artifact: "+name)
    for name in ("cache_sha256","checkpoint_sha256"): ia.sha(manifest.get(name),name)
    status = ia.read_json(directory/"training_status.json")
    ia.require(status["complete"] and status == manifest["result"],"Training status/manifest differs or incomplete")
    config = manifest["configuration"]
    updates = ia.number(status["updates"],"training updates",integer=True,minimum=1)
    ia.require(updates == config["updates"],"Training stopped at wrong update")
    rows = ia.read_jsonl(directory/"training.jsonl")
    ia.require([x["step"] for x in rows] == list(range(1,updates+1)),"Training step sequence differs")
    totals,target_ids,bank_ids,schedule = defaultdict(int),set(),set(),hashlib.sha256()
    targets,bank_exposures,routes = 0,0,Counter()
    for row in rows:
        sample = row["sample"]
        ia.require(len(sample["targets"]) == config["batch_size"],"Training batch size differs")
        ia.require(len(set(sample["episode"])) == len(sample["episode"]),"Duplicate bank entry")
        ia.require(len(sample["columns"]) == len(sample["targets"]) and
            [sample["episode"][i] for i in sample["columns"]] == sample["targets"],"Training target addresses differ")
        ia.require(all(x == 0 for x in sample["qviews"]+sample["sviews"]),"Held-out style entered coldstart training")
        targets += len(sample["targets"]); bank_exposures += len(sample["episode"])
        target_ids.update(sample["targets"]); bank_ids.update(sample["episode"])
        schedule.update(json.dumps(sample,sort_keys=True).encode())
        expected_route = config["routing"]
        if expected_route == "dense_warm": expected_route = "dense" if row["step"] <= updates//2 else "sparse"
        ia.require(row["routing"] == expected_route,"Training routing schedule differs")
        routes[row["routing"]] += 1
        for key,value in row.items():
            if key.endswith("tokens") or key.endswith("forward_calls"):
                ia.number(value,key,integer=True); totals[key] += value
    ia.require(dict(totals) == status["totals"],"Training token/call totals differ from raw log")
    ia.require(schedule.hexdigest() == status["schedule_sha256"],"Training sample schedule differs")
    ia.require(len(target_ids) == status["target_unique_count"] and len(bank_ids) == status["bank_unique_count"],"Training exposure coverage differs")
    ia.require(status["elapsed_seconds"] >= rows[-1]["elapsed_seconds"],"Training elapsed time precedes last update")
    gradients = {}
    for name,label in (("base_keys","key"),("base_values","value")):
        counts = [ia.number(row[name+"_gradient_slots"],"gradient slots",integer=True) for row in rows]
        coverage = ia.number(status[label+"_gradient_coverage"],"gradient coverage",integer=True)
        ia.require(max(counts) <= coverage <= sum(counts),"Gradient union count outside bounds")
        gradients[name] = dict(reported_union_slots=coverage,per_step_min=min(counts),per_step_max=max(counts),
            per_step_mean=sum(counts)/len(counts),validation="Per-update counts and union bounds checked; usage.pt hashed, not tensor-deserialized.")
    return dict(kind="training",updates=updates,split="train",schedule_sha256=schedule.hexdigest(),
        target_unique_count=len(target_ids),bank_unique_count=len(bank_ids),routing_updates=dict(routes),
        gradient_coverage=gradients,trainable_parameters=status["trainable_parameters"],
        routing_usage={k:v for k,v in status.items() if k.startswith("sampled_last_forward_") or k=="usage_scope"},
        costs=dict(target_exposures=targets,episodic_record_exposures=bank_exposures,
            processed_input_positions=sum(totals.get(k,0) for k in ("student_input_tokens","teacher_input_tokens","rollout_input_tokens")),
            backbone_forward_calls=sum(totals.get(k,0) for k in ("student_forward_calls","teacher_forward_calls","rollout_forward_calls")),
            token_call_totals=dict(totals),training_elapsed_seconds=status["elapsed_seconds"]),
        artifacts=ia.artifacts(directory,["manifest.json","training.jsonl","training_status.json","usage.pt","last.pt"]))


def evaluation_table(records):
    """Compact report rows; the complete audited vectors remain in records."""
    result = []
    for r in records:
        if r["kind"] != "evaluation": continue
        base = r["foundation"]
        row = dict(run_dir=r["run_dir"],arm=r["arm"],domain=r["domain"],split=r["split"],
            analysis_group="supplementary_teacher_followup" if is_teacher_followup(r) else "predeclared_main_or_diagnostic",
            evaluation_purpose="seen_train_memorization" if r["domain"]=="synthetic_train_probe" else r["split"],
            seed=r["seed"],base_mode=base["base_mode"],alpha_base=base["alpha_base"],
            foundation_effective=base["foundation_effective"],foundation_effect_status=base["foundation_effect_status"],
            foundation_weighted_top_k=base["foundation_weighted_top_k"],maximum_weighted_records=base["maximum_weighted_records"],
            base_records=base["active_records"],bank_records=r["bank_records"],cases=r["evaluated_facts"],
            foundation_all_calls_usage={k:base["usage"][k] for k in ("query_positions","top1_used","topk_used","top1_coverage","topk_coverage","effective_slots")},
            phases={})
        for phase in ia.PHASES:
            real = r["phases"][phase]["methods"]["real"]
            teacher = r["linked_teacher_qualification"]
            qualified = teacher["phases"][phase] if teacher else None
            row["phases"][phase] = dict(real={k:real[k] for k in ("count","a_correct","b_correct","both_correct","paired_switch_em", "a_answer_token_nll","b_answer_token_nll","a_recall_at_1","b_recall_at_1","a_recall_at_4","b_recall_at_4","paired_answer_containment_rate")},
                teacher_pair_qualified=qualified["strict_pair_qualified"] if qualified else None,
                teacher_pair_denominator=qualified["count"] if qualified else None,
                teacher_pair_rate=qualified["strict_pair_qualified"]/qualified["count"] if qualified else None,
                teacher_reaches_90_percent=(qualified["strict_pair_qualified"]/qualified["count"]>=.9) if qualified else None,
                teacher_condition="raw_observation_context" if qualified else "not_measured_or_linked",
                teacher_interpretation=("Raw-context teacher fails the 90% paired threshold; student errors in this condition do not isolate writable-memory/generalization failure."
                    if qualified and qualified["strict_pair_qualified"]/qualified["count"]<.9 else
                    "Raw-context teacher meets the paired threshold." if qualified else "No qualified teacher evidence for this exact condition."),
                teacher_qualified_real=qualified["methods"]["real"] if qualified else None,
                foundation_real_generation_usage={k:base["usage_by_phase_method_call"][phase+"/real/generation"][k]
                    for k in ("query_positions","top1_used","topk_used","top1_coverage","topk_coverage","effective_slots")})
        result.append(row)
    return result


def is_teacher_followup(record):
    return isinstance(record.get("arm"),str) and record["arm"].startswith("wiki_teacher_")


def aggregate_costs(records):
    return dict(training_updates=sum(r.get("updates",0) for r in records if r["kind"]=="training"),
        training_target_exposures=sum(r.get("costs",{}).get("target_exposures",0) for r in records),
        training_input_positions=sum(r.get("costs",{}).get("processed_input_positions",0) for r in records),
        training_backbone_calls=sum(r.get("costs",{}).get("backbone_forward_calls",0) for r in records),
        training_process_seconds=sum(r.get("costs",{}).get("training_elapsed_seconds",0) for r in records),
        evaluation_generation_calls=sum(r.get("costs",{}).get("generation_calls",0) for r in records),
        evaluation_generation_tokens=sum(r.get("costs",{}).get("generation_tokens",0) for r in records),
        evaluation_generation_seconds=sum(r.get("costs",{}).get("generation_seconds",0) for r in records),
        evaluation_answer_scoring_target_tokens=sum(r.get("costs",{}).get("answer_scoring_tokens",0) for r in records))


def formal_training_accounting(records):
    """Separate the fixed main training budget from retained smoke/preflight work.

    Match declared conditions, never outcome scores or a directory-name heuristic.
    Duplicates remain costed and prevent a complete-budget claim.
    """
    expected = {
        ("matched_warm", "matched", 1024),
        ("wiki_warm", "wikipedia", 1024),
        ("wiki_joint_warm", "wikipedia", 1024),
        *((arm, "synthetic", 512) for arm in (
            "no_warm", "matched_warm_transfer", "wiki_warm_transfer", "fixed",
            "learned_sparse", "dense_warm", "straight_through", "wiki_joint")),
    }
    found = defaultdict(list)
    formal, other = [], []
    for row in records:
        if row["kind"] != "training": continue
        key = (row.get("arm"), row.get("domain"), row.get("updates"))
        if key in expected and row.get("seed") == 63042:
            formal.append(row)
            found[key].append(row["run_dir"])
        else:
            other.append(row)
    describe = lambda key: dict(arm=key[0], domain=key[1], updates=key[2], seed=63042)
    return dict(
        scope="Retained main training only: three 1024-step warm runs plus eight 512-step downstream runs, seed 63042. Excludes smoke/preflight, supplementary teacher follow-up, feasibility probes and explicitly excluded suites. Matching is by arm/domain/updates/seed, not outcomes.",
        expected_jobs=len(expected), expected_training_updates=sum(k[2] for k in expected),
        completed_jobs=len(formal), completed_conditions=len(found),
        complete=set(found)==expected and all(len(v)==1 for v in found.values()),
        missing=[describe(key) for key in sorted(expected-set(found))],
        duplicates=[dict(**describe(key),run_dirs=value) for key,value in sorted(found.items()) if len(value)>1],
        run_dirs=[r["run_dir"] for r in formal], costs=aggregate_costs(formal),
        other_retained_training=dict(
            scope="Other retained non-supplementary training, including smoke/preflight or nonmatching budgets/seeds; counted in top-level costs but excluded from formal_training_costs.",
            jobs=[{k:r.get(k) for k in ("run_dir","arm","domain","updates","seed")} for r in other],
            costs=aggregate_costs(other)))


def followup_scope(records,requested):
    """Separate follow-up completion; never inflate the original eight-arm plan."""
    expected={
        ("wikipedia","dev",16,64),("synthetic","dev",16,64),
        ("wikipedia","confirm",64,128),("synthetic","confirm",64,128),
        ("synthetic_train_probe","dev",8,128),
    }
    found=defaultdict(list);trained=defaultdict(list)
    expected_training={"wiki_teacher_warm":1024,"wiki_teacher_repair":512}
    for row in records:
        if row["kind"]=="training" and row["arm"] in expected_training and row["updates"]==expected_training[row["arm"]]:
            trained[row["arm"]].append(row["run_dir"])
        if row["kind"]!="evaluation" or row["arm"]!="wiki_teacher_repair":continue
        if row["seed"]!=63042 or row["foundation"]["base_mode"]!="full":continue
        key=(row["domain"],row["split"],row["evaluated_facts"],row["bank_records"])
        if key in expected:found[key].append(row["run_dir"])
    describe=lambda key:dict(domain=key[0],split=key[1],cases=key[2],bank_records=key[3])
    complete=(set(found)==expected and all(len(v)==1 for v in found.values()) and
        set(trained)==set(expected_training) and all(len(v)==1 for v in trained.values()))
    return dict(requested=requested,predeclared_main_arm=False,expected_eval_conditions=5,
        completed_eval_conditions=len(found),complete=complete if requested else None,
        missing=[describe(key) for key in sorted(expected-set(found))],
        duplicates=[dict(**describe(key),run_dirs=value) for key,value in sorted(found.items()) if len(value)>1],
        expected_training_updates=expected_training,training_runs=dict(trained),
        scope="Optional wiki_teacher_repair follow-up: five evaluations plus separate 1024-step Wiki warm and 512-step synthetic transfer. This does not change the main eight arms or 22 confirmation conditions.",
        strategy_selection_rule="On the fixed train-only 64-pair validation: prefer quote_instruction if paired EM >=90%; otherwise gold_annotated only if >=90%; otherwise no follow-up training. The train-only teacher-probe artifact must substantiate this rule separately.",
        supervision_note="gold_annotated is explicit answer-label localization supervision, not raw-context distillation. Seen-train probe phase labels duplicate canonical views and are not held-out formats.")


def formal_scope(records):
    expected = {(arm,domain,"full") for arm in FORMAL_ARMS for domain in ("wikipedia","synthetic")}
    expected |= {("straight_through",domain,mode) for domain in ("wikipedia","synthetic") for mode in ("off","random256","cluster256")}
    found = defaultdict(list)
    for row in records:
        if row["kind"] != "evaluation" or row["split"] != "confirm": continue
        if row["evaluated_facts"] != 64 or row["bank_records"] != 128 or row["seed"] != 63042: continue
        key = (row["arm"],row["domain"],row["foundation"]["base_mode"])
        if key in expected: found[key].append(row["run_dir"])
    describe = lambda key: dict(arm=key[0],domain=key[1],base_mode=key[2])
    return dict(protocol_document="docs/coldstart_protocol.md",expected_confirm_conditions=len(expected),
        completed_conditions=len(found),complete=set(found)==expected and all(len(v)==1 for v in found.values()),
        missing=[describe(key) for key in sorted(expected-set(found))],
        duplicates=[dict(**describe(key),run_dirs=value) for key,value in sorted(found.items()) if len(value)>1],
        scope="Eight full-bank arms × two domains, plus three straight-through compression controls × two domains; confirm 64 targets/128 records, seed 63042. This checks declared conditions, not a source-code or dataset-equivalence proof.")


def _in_scope(path,root):
    components = (root.name,*path.relative_to(root).parts[:-1])
    return (any(name.startswith("coldstart_") for name in components) and
        not any(name in {"source","source_snapshot","source_snapshots",".git"} for name in components))


def discover(runs_root):
    root = Path(runs_root).resolve()
    ia.require(root.is_dir(),"Runs root does not exist")
    return sorted(path for path in root.rglob("manifest.json") if _in_scope(path,root))


def exclusions(runs_root):
    """Explicit sidecars exclude their descendants; never infer from scores."""
    root = Path(runs_root).resolve()
    result = {}
    for path in sorted(root.rglob("analysis_excluded.json")):
        if not _in_scope(path,root): continue
        data = ia.read_json(path)
        ia.require(isinstance(data,dict) and isinstance(data.get("reason"),str) and data["reason"].strip(),
            "Analysis exclusion needs a nonempty reason: "+str(path))
        ia.require(bool(data.get("evidence")) and bool(data.get("superseded_by")),
            "Analysis exclusion needs evidence and superseded_by: "+str(path))
        result[path.parent] = dict(suite_dir=str(path.parent),status="excluded",reason=data["reason"],
            evidence=data["evidence"],superseded_by=data["superseded_by"],metadata=data,
            sidecar=dict(path=str(path),sha256=ia.file_hash(path)),jobs=[])
    return result


def discarded_job_compute(directory):
    """Only cost accounting from excluded logs; no accuracy or gradient claims.

    A damaged/incomplete log is retained as unavailable rather than blocking valid
    replacement runs or silently inventing zero compute. This never audits an old
    run as scientifically valid and never adds its costs to the retained totals.
    """
    paths = [name for name in ("training.jsonl","predictions.jsonl") if (directory/name).is_file()]
    out = dict(costs={},artifacts=ia.artifacts(directory,paths),available=False,
        scope="Excluded work only. Counts come from successfully parsed logged steps/reads; last logged elapsed time omits startup/finalization and may undercount interrupted work. No quality results are analyzed; process times can overlap.")
    try:
        if "training.jsonl" in paths:
            rows = ia.read_jsonl(directory/"training.jsonl")
            ia.require(rows and [r["step"] for r in rows] == list(range(1,len(rows)+1)),"Excluded training step sequence is incomplete")
            tokens,calls,exposures = 0,0,0
            for row in rows:
                for key in ("student_input_tokens","teacher_input_tokens","rollout_input_tokens"):
                    tokens += ia.number(row.get(key,0),"discarded "+key,integer=True)
                for key in ("student_forward_calls","teacher_forward_calls","rollout_forward_calls"):
                    calls += ia.number(row.get(key,0),"discarded "+key,integer=True)
                exposures += len(row["sample"]["targets"])
            out["costs"].update(training_updates=len(rows),training_target_exposures=exposures,
                training_input_positions=tokens,training_backbone_calls=calls,
                training_process_seconds=ia.number(rows[-1]["elapsed_seconds"],"discarded elapsed seconds"))
        if "predictions.jsonl" in paths:
            rows = ia.read_jsonl(directory/"predictions.jsonl")
            out["costs"].update(evaluation_generation_calls=len(rows),
                evaluation_generation_tokens=sum(ia.number(r["generation_tokens"],"discarded generation tokens",integer=True) for r in rows),
                evaluation_generation_seconds=sum(ia.number(r["generation_seconds"],"discarded generation seconds") for r in rows),
                evaluation_answer_scoring_target_tokens=sum(ia.number(r["answer_scoring_tokens"],"discarded scoring tokens",integer=True) for r in rows))
        out["available"] = bool(paths)
    except (OSError,ValueError,KeyError,TypeError) as error:
        out.update(available=False,costs={},error=repr(error))
    return out


def audit_teacher_probe(directory,manifest):
    """Recompute the supplementary train-only teacher decision from all 480 rows."""
    metrics=ia.read_json(directory/"metrics.json")
    selection=ia.read_json(directory/"selection.json")
    ia.require(manifest.get("complete") is True and manifest.get("result")==metrics,"Teacher probe manifest incomplete/differs")
    ia.require(metrics.get("protocol")=="coldstart-teacher-feasibility-v1" and metrics.get("complete") is True,"Wrong teacher probe protocol")
    ia.require(metrics["teacher_only"] and metrics["shared_question"] and metrics["gradient_steps"]==0,"Teacher probe changed shared training state")
    ia.require(metrics["backbone_hash_before"]==metrics["backbone_hash_after"] and metrics["backbone_unchanged"],"Teacher probe backbone changed")
    ia.require(manifest["selection"]==metrics["selection"]==selection and selection["source_split"]=="train","Teacher selection differs")
    ia.require(selection["selection_policy"]=="train-only-label-disjoint-v2" and selection["validation_seed"]==95043,"Unregistered teacher candidate selection")
    rejected=selection["excluded_records"]
    excluded_indices={x["index"] for x in rejected}
    ia.require(len(excluded_indices)==len(rejected)==selection["excluded_count"],"Duplicate teacher exclusions")
    eligible=[i for i in range(selection["source_records"]) if i not in excluded_indices]
    ia.require(len(eligible)==selection["eligible_records"],"Eligible teacher count differs")
    expected=dict(calibration=eligible[:16],validation=sorted(ia.random.Random(95043).sample(eligible[16:],64)))
    ia.require(selection["selected_indices"]==expected,"Teacher targets differ from fixed eligible sampling")
    for row in rejected:
        ia.require(row["reasons"] and all((" "+ia.norm(x["answer"])+" ") in (" "+ia.norm(row["question"])+" ") for x in row["reasons"]),"Excluded record lacks complete question-label overlap")
    preflight=selection["teacher_context_validation"]
    ia.require(preflight["checked_contexts"]==2*selection["source_records"] and preflight["passed"] and not preflight["failures"],"Gold annotation full-training preflight not passed")
    raw=ia.read_jsonl(directory/"predictions.jsonl")
    strategies=("baseline","quote_instruction","gold_annotated")
    groups=defaultdict(dict);costs=defaultdict(float);question_map={};source_map={};suffixes=defaultdict(set)
    cost_fields=("generation_tokens","generation_seconds","generation_forward_calls","generation_input_positions",
        "prompt_tokens","answer_scoring_tokens","scoring_seconds","scoring_forward_calls","scoring_input_positions")
    for row in raw:
        subset,strategy,identity,world=(row[k] for k in ("subset","strategy","id","world"))
        ia.require(subset in expected and strategy in strategies and world in {"A","B"},"Unknown teacher probe condition")
        indices=selection["selected_indices"][subset];ids=selection["selected_ids"][subset]
        ia.require(identity in ids and row["source_index"]==indices[ids.index(identity)],"Teacher row target not selected")
        key=(subset,strategy,identity)
        ia.require(world not in groups[key],"Repeated teacher world");groups[key][world]=row
        qkey=(subset,identity)
        ia.require(qkey not in question_map or question_map[qkey]==row["question"],"Teacher question changed across strategies/worlds")
        question_map[qkey]=row["question"]
        ia.require((" "+ia.norm(row["answer"])+" ") not in (" "+ia.norm(row["question"])+" "),"Teacher selected question contains label")
        source,context=row["source_context"],row["teacher_context"]
        source_key=(subset,identity,world)
        ia.require(source_key not in source_map or source_map[source_key]==(source,row["answer"]),"Teacher source/answer changed across strategies")
        source_map[source_key]=(source,row["answer"])
        ia.require(row["extra_gold_localization"]==(strategy=="gold_annotated") and row["source_only"]==(strategy!="gold_annotated"),"Gold localization mislabeled")
        ia.require(row["question_unchanged"] and row["source_context_preserved"],"Teacher source/question changed")
        if strategy=="baseline":ia.require(context==source,"Baseline context changed")
        else:
            ia.require(context.startswith(source+"\n\n"),"Teacher original source not preserved")
            suffix=context[len(source)+2:]
            if strategy=="gold_annotated":
                tag="<requested_span>"+row["answer"]+"</requested_span>"
                ia.require(suffix.endswith(tag) and row["answer"] in source,"Gold annotation changed label or source")
                suffix=suffix[:-len(tag)]
            suffixes[strategy].add(suffix)
        ia.same_number(row["em"],int(ia.norm(row["prediction"])==ia.norm(row["answer"])),"Teacher probe EM")
        ia.same_number(row["literal_em"],int(row["prediction"].strip()==row["answer"]),"Teacher literal EM")
        ia.same_number(row["answer_containment"],int((" "+ia.norm(row["answer"])+" ") in (" "+ia.norm(row["prediction"])+" ")),"Teacher containment")
        ia.require(row["budget_hit"]==(row["generation_tokens"]>=32),"Teacher token budget flag")
        ia.same_number(row["prediction_words"],len(row["prediction"].split()),"Teacher output words")
        ia.number(row["answer_nll_sum"],"Teacher answer NLL")
        for field in cost_fields:costs[field]+=ia.number(row[field],"Teacher "+field)
    ia.require(len(raw)==480 and all(len(suffixes[s])==1 for s in ("quote_instruction","gold_annotated")),"Teacher condition changed or predictions incomplete")
    summaries={}
    for subset in expected:
        summaries[subset]={}
        for strategy in strategies:
            pairs=[groups[(subset,strategy,i)] for i in selection["selected_ids"][subset]]
            ia.require(all(set(pair)=={"A","B"} for pair in pairs),"Missing teacher pair")
            rows=[pair[w] for pair in pairs for w in ("A","B")]
            both=sum(pair["A"]["em"] and pair["B"]["em"] for pair in pairs)
            recalc=dict(count=len(pairs),predictions=len(rows),a_correct=sum(p["A"]["em"] for p in pairs),
                b_correct=sum(p["B"]["em"] for p in pairs),both_correct=both,paired_em=both/len(pairs),
                exact_literal_pair_correct=sum(p["A"]["literal_em"] and p["B"]["literal_em"] for p in pairs),
                paired_containment=sum(p["A"]["answer_containment"] and p["B"]["answer_containment"] for p in pairs)/len(pairs),
                answer_token_nll=sum(r["answer_nll_sum"] for r in rows)/sum(r["answer_scoring_tokens"] for r in rows),
                budget_hits=sum(r["budget_hit"] for r in rows),three_word_outputs=sum(r["prediction_words"]==3 for r in rows))
            logged=metrics["subsets"][subset][strategy]
            for k,v in recalc.items():ia.same_number(logged[k],v,"Teacher summary "+k)
            ia.require(logged["extra_gold_localization"]==(strategy=="gold_annotated") and logged["source_only"]==(strategy!="gold_annotated"),"Teacher summary supervision mislabeled")
            ia.require(logged["teacher_pair_reaches_90_percent"]==(both/len(pairs)>=.9),"Teacher threshold differs")
            summaries[subset][strategy]={**recalc,"extra_gold_localization":strategy=="gold_annotated","source_only":strategy!="gold_annotated"}
    costs["generation_calls"]=costs["scoring_calls"]=480
    costs["total_backbone_forward_calls"]=costs["generation_forward_calls"]+costs["scoring_forward_calls"]
    costs["total_backbone_input_positions"]=costs["generation_input_positions"]+costs["scoring_input_positions"]
    for key,value in costs.items():ia.same_number(metrics["costs"][key],value,"Teacher cost "+key)
    for name in ("predictions.jsonl","selection.json"):
        ia.require(ia.file_hash(directory/name)==metrics["artifacts"][name],"Teacher artifact SHA differs")
    validation=summaries["validation"]
    chosen=("quote_instruction" if validation["quote_instruction"]["paired_em"]>=.9 else
            "gold_annotated" if validation["gold_annotated"]["paired_em"]>=.9 else None)
    return dict(kind="teacher_feasibility",run_dir=str(directory),source_split="train",subsets=summaries,
        selected_strategy=chosen,selection_rule="quote>=0.9 else annotated>=0.9 else none",costs=dict(costs),
        selection=selection,predeclared_main_arm=False,
        interpretation="Gold-annotated success is extra answer-localization supervision, not raw-context teacher success; these are training-distribution feasibility subsets, not document-generalization tests.",
        artifacts=ia.artifacts(directory,["manifest.json","metrics.json","selection.json","predictions.jsonl"]))


def summarize(runs_root):
    records,pending,other,teacher_probes,teacher_probe_pending = [],[],[],[],[]
    excluded = exclusions(runs_root)
    for path in discover(runs_root):
        excluded_parent = next((parent for parent in path.parents if parent in excluded),None)
        if excluded_parent is not None:
            # Invalid old evidence is quarantined before strict result auditing.
            # Its sidecar remains independently hashed and fully visible.
            job = dict(run_dir=str(path.parent),manifest_sha256=ia.file_hash(path),
                discarded_compute=discarded_job_compute(path.parent))
            try:
                old = ia.read_json(path)
                ia.require(isinstance(old,dict),"Excluded manifest is not an object")
                job.update(stage=old.get("stage"),original_complete=old.get("complete"),
                    original_error=old.get("error"),configuration=old.get("configuration"))
                accounting = job["discarded_compute"]
                if old.get("stage") == "train":
                    accounting["cost_evidence_complete"] = (old.get("complete") is True and accounting["available"] and
                        accounting["costs"].get("training_updates") == old.get("configuration",{}).get("updates"))
                elif old.get("stage") == "eval":
                    accounting["cost_evidence_complete"] = (old.get("complete") is True and accounting["available"] and
                        accounting["costs"].get("evaluation_generation_calls") == old.get("result",{}).get("generation_calls"))
            except (OSError,ValueError,TypeError) as error: job["manifest_read_error"] = repr(error)
            excluded[excluded_parent]["jobs"].append(job)
            continue
        manifest = ia.read_json(path)
        if manifest.get("protocol")=="coldstart-teacher-feasibility-v1":
            if manifest.get("complete") is True:teacher_probes.append(audit_teacher_probe(path.parent,manifest))
            else:teacher_probe_pending.append(dict(run_dir=str(path.parent),error=manifest.get("error"),complete=False))
            continue
        if manifest.get("protocol") != RUN_PROTOCOL:
            other.append(dict(path=str(path),protocol=manifest.get("protocol"))); continue
        if manifest.get("stage") not in {"prepare","init","train","eval"}:
            # Feature repair uses the same cache protocol, but is an auxiliary
            # data-provenance artifact rather than an experiment-run schema.
            other.append(dict(path=str(path),protocol=manifest.get("protocol"),stage=manifest.get("stage"),
                repair_protocol=manifest.get("repair_protocol"),complete=manifest.get("complete"),
                manifest_sha256=ia.file_hash(path),classification="auxiliary_preparation_not_model_result"))
            continue
        config = manifest["configuration"]
        ia.require(manifest["stage"] == config["stage"],"Manifest stage differs")
        identity = dict(run_dir=str(path.parent),stage=manifest["stage"],arm=config.get("arm"),
            domain=config.get("domain"),seed=config.get("seed"),configuration=config,
            started_at=manifest.get("started_at"),finished_at=manifest.get("finished_at"),
            cache_sha256=manifest.get("cache_sha256"),checkpoint_sha256=manifest.get("checkpoint_sha256"))
        if manifest.get("complete") is not True:
            pending.append(dict(**identity,status="failed" if "error" in manifest else "incomplete",error=manifest.get("error")))
            continue
        stage = manifest["stage"]
        if stage != "init": ia.require(manifest.get("backbone_unchanged") is True,"Backbone preservation missing")
        if stage == "eval": record = audit_foundation(path.parent,manifest,ia.audit_evaluation(path.parent,manifest))
        elif stage == "train": record = audit_training(path.parent,manifest)
        else: record = dict(kind=stage,result=manifest["result"],artifacts=ia.artifacts(path.parent,["manifest.json"]))
        record.update(identity)
        records.append(record)
    # Reuse teacher eligibility only with identical full bank/text/style protocol.
    # This is explicitly linked evidence, not a generated teacher observation.
    teachers = defaultdict(list)
    for r in records:
        if r["kind"] == "evaluation" and any(p["teacher_acceptance"] is not None for p in r["phases"].values()):
            teachers[r["comparison_key"]].append(r)
    for r in records:
        if r["kind"] != "evaluation": continue
        candidates = teachers[r["comparison_key"]]
        r["linked_teacher_qualification"] = None
        if not candidates: continue
        first = candidates[0]
        for other_run in candidates[1:]:
            ia.require(all(first["pair_vectors"][phase]["teacher"] == other_run["pair_vectors"][phase]["teacher"] for phase in ia.PHASES),"Identical teacher conditions produced different eligibility")
        linked = {}
        for phase in ia.PHASES:
            eligible = [key for key,value in first["pair_vectors"][phase]["teacher"].items() if value]
            linked[phase] = dict(count=r["evaluated_facts"],strict_pair_qualified=len(eligible),qualified_ids=eligible,
                methods={method:dict(count=len(eligible),both_correct=sum(vector[i] for i in eligible),
                    paired_switch_em=sum(vector[i] for i in eligible)/len(eligible) if eligible else None)
                    for method,vector in r["pair_vectors"][phase].items() if method != "teacher"})
        r["linked_teacher_qualification"] = dict(source_runs=[x["run_dir"] for x in candidates],
            verification="Same comparison key: cache, full bank observations, selected targets, styles, model and token budget.",phases=linked)
    all_records=records
    followup_records=[r for r in all_records if is_teacher_followup(r)]
    records=[r for r in all_records if not is_teacher_followup(r)]
    followup_pending=[r for r in pending if is_teacher_followup(r)]
    pending=[r for r in pending if not is_teacher_followup(r)]
    costs=aggregate_costs(records)
    formal_training=formal_training_accounting(records)
    followup=followup_scope(followup_records,bool(followup_records or followup_pending))
    formal = formal_scope(records)
    discarded_totals = defaultdict(float)
    unavailable = []
    for suite in excluded.values():
        for job in suite["jobs"]:
            accounting = job["discarded_compute"]
            if accounting["available"]:
                for key,value in accounting["costs"].items(): discarded_totals[key] += value
            elif job.get("stage") in {"train","eval"}: unavailable.append(job["run_dir"])
    return dict(protocol=PROTOCOL,generated_at=datetime.now(timezone.utc).isoformat(),
        runs_root=str(Path(runs_root).resolve()),audit_passed=True,
        partial=bool(pending) or not formal["complete"] or bool(followup_pending) or (followup["requested"] and not followup["complete"]),
        main_partial=bool(pending) or not formal["complete"],complete_jobs=len(records),incomplete_jobs=pending,
        retained_total_complete_jobs=len(all_records),all_retained_costs=aggregate_costs(all_records),
        supplementary_teacher_followup=dict(scope=followup,records=followup_records,
            incomplete_jobs=followup_pending,costs=aggregate_costs(followup_records),
            evaluation_table=evaluation_table(followup_records)),
        teacher_feasibility=teacher_probes,teacher_feasibility_incomplete=teacher_probe_pending,
        teacher_feasibility_costs={key:sum(r["costs"][key] for r in teacher_probes)
            for key in teacher_probes[0]["costs"]} if teacher_probes else {},
        all_retained_costs_scope="All retained primary-group model jobs, including smoke/preflight, plus supplementary model jobs. Separately audited teacher_feasibility_costs and discarded_compute are not silently mixed into those totals; this is not the formal main training budget.",
        costs_scope="Retained primary-group jobs including smoke/preflight, excluding supplementary teacher follow-up, feasibility probes and explicitly excluded suites. Use formal_training_costs for the declared 7168-update main training budget.",
        formal_training_scope=formal_training,
        formal_training_costs=formal_training["costs"],
        other_retained_training_costs=formal_training["other_retained_training"]["costs"],
        formal_scope=formal,evaluation_table=evaluation_table(records),
        excluded_suites=list(excluded.values()),
        discarded_compute=dict(costs=dict(discarded_totals),jobs_with_unavailable_costs=unavailable,
            partial_logged_evidence=any(not job["discarded_compute"].get("cost_evidence_complete",False)
                for suite in excluded.values() for job in suite["jobs"] if job.get("stage") in {"train","eval"}),
            included_in_retained_costs=False,scope="Explicitly excluded suites only; cost accounting is not a scientific result or evidence of model performance. Logged process times may overlap and incomplete logs can undercount discarded work."),
        skipped_nonexperiment_manifests=other,records=records,costs=costs,
        scripts={Path(__file__).name:ia.file_hash(Path(__file__)),_path.name:ia.file_hash(_path)},
        limitations=["Formal completeness additionally requires the 22 declared confirmation conditions. A complete discovered job set alone does not imply the full experiment has finished.",
            "Raw EM/containment, paired updates, teacher eligibility and foundation CPU usage are recomputed. Saved tensor snapshots/checkpoints are hashed without deserialization or inference replay.",
            "All completed discovered jobs are itemized; smoke/preflight are not silently excluded. Compare intended formal arms explicitly and do not select on confirm scores.",
            "Top-level costs includes retained primary-group smoke/preflight. Formal main training is separately matched to 11 declared conditions (7168 updates); optional teacher follow-up has its own 1536-update budget. Counts are observed completed work, never imputed for missing jobs.",
            "An analysis_excluded.json sidecar excludes all descendant jobs from retained records, pending jobs, teacher linkage, formal completion and costs. Reasons, evidence, replacements and sidecar hashes remain visible; discarded compute is a separate log-based account, never model-quality evidence.",
            "Optional wiki_teacher_* runs are listed and costed separately as a supplementary teacher-feasibility follow-up, never an original ninth arm. Main formal completion remains eight arms/22 confirmation conditions; all_retained_costs explicitly adds the optional follow-up.",
            "Low raw-context teacher paired accuracy, including Wikipedia extraction errors, confounds student interpretation: student zeros cannot by themselves establish writable-memory semantic failure. High gold-annotated train-probe scores instead validate an explicitly stronger supervision condition.",
            "Empty disables episodic memory only. off+empty disables both banks. Its strict A/B paired EM is logically zero for distinct answers and one deterministic output.",
            "Foundation routing counts include prompt/repeated-prefix/scoring and any padding positions. High top-k coverage alone does not establish semantic usage or generalization.",
            "no_warm/matched_warm/wiki_warm have actual alpha_base=0: foundation routes are audit-only, foundation_effective=false/weight_zero regardless of coverage. Full mode can still incur extra CPU search cost. Top-4+4 potential output contributions require positive alpha and a nonempty foundation bank; reported effective=true is structural, not proof of usefulness.",
            "Mean clustering preserves arithmetic means, not retrieval function or output norm; random and clustered banks change capacity, top-k multiplicity and probability mass.",
            "Training input positions are unpadded logged positions, not FLOPs. Evaluation lacks full prompt/scoring time costs. Summed process seconds may overlap on shared GPUs and are not exclusive latency or total end-to-end cost.",
            "Training uses canonical view 0 only; P equals A, so AP consistency supplies no independent paraphrase augmentation. All main comparisons share one initialization/optimization seed.",
            "Teacher sees the target observation while student reads the full bank. Teacher-qualified subsets supplement, never replace, full denominators. Linked teacher evidence is labeled with its actual source runs."])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True,help="Output JSON file; its parent is created")
    args = p.parse_args(argv)
    report = summarize(args.runs_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output),complete_jobs=report["complete_jobs"],partial=report["partial"])))


if __name__ == "__main__": main()
