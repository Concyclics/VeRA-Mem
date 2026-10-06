"""Audit only interface_* suites and summarize the four experiment steps.

Run on a local backup or on a host where --runs-root is a filesystem path:
  python scripts/summarize_interface.py --runs-root WORKSPACE/runs --output REPORT
No SSH, torch, inference, checkpoint loading or historical-run discovery occurs.
Complete jobs in a still-running suite are usable but marked suite_partial.
Malformed completed evidence is rejected, rather than silently omitted.
  python scripts/summarize_interface.py --self-test
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import unicodedata

PHASES = ("CC", "CH", "HC", "HH")
METHODS = {"CC": ("real", "shuffled", "empty"), "CH": ("real",),
           "HC": ("real", "shuffled", "empty", "canonical_key"), "HH": ("real",)}
BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED = 2000, 68043
PROTOCOL = "interface-audit-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + x)))


def read_jsonl(path):
    return [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + x)))
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split())


def number(value, name, *, integer=False, minimum=0):
    require(type(value) in ((int,) if integer else (int, float)) and math.isfinite(value) and value >= minimum,
            "Invalid " + name)
    return value


def same_number(actual, expected, name):
    require(type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-7, rel_tol=1e-7),
            f"Incorrect {name}: logged={actual!r}, recomputed={expected!r}")


def sha(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Missing/invalid SHA256: " + name)
    return value


def artifacts(directory, names):
    return {name: dict(path=str(directory / name), sha256=file_hash(directory / name))
            for name in names if (directory / name).is_file()}


def paired_summary(pairs):
    if not pairs:
        return dict(count=0)
    count = len(pairs)
    a = [int(norm(left["prediction"]) == norm(left["answer_results"]["A"]["answer"])) for left, _ in pairs]
    b = [int(norm(right["prediction"]) == norm(right["answer_results"]["B"]["answer"])) for _, right in pairs]
    changed = [norm(left["prediction"]) != norm(right["prediction"]) for left, right in pairs]
    both = sum(x and y for x, y in zip(a, b))
    out = dict(count=count, a_correct=sum(a), b_correct=sum(b), both_correct=both,
               a_em=sum(a)/count, b_em=sum(b)/count, paired_switch_em=both/count,
               changed_count=sum(changed), changed_rate=sum(changed)/count,
               changed_but_not_both_correct_rate=sum(c and not (x and y) for c,x,y in zip(changed,a,b))/count,
               b_correct_given_a_correct=both/sum(a) if sum(a) else None,
               a_correct_given_b_correct=both/sum(b) if sum(b) else None)
    for side, offset, label in (("a", 0, "A"), ("b", 1, "B")):
        rows = [p[offset] for p in pairs]
        tokens = sum(r["answer_results"][label]["tokens"] for r in rows)
        out[side+"_answer_token_nll"] = sum(r["answer_results"][label]["nll_sum"] for r in rows)/tokens
        for key in ("recall_at_1", "recall_at_4"):
            values = [r[key] for r in rows if r[key] is not None]
            out[side+"_"+key] = sum(values)/len(values) if values else None
        decode_count = sum(r["decode_query_count"] for r in rows)
        out[side+"_decode_query_count"] = decode_count
        out[side+"_decode_correct_residency"] = sum(r["decode_correct_hits"] or 0 for r in rows)/decode_count if decode_count else None
        out[side+"_answer_containment_rate"] = sum(r["answer_results"][label]["answer_containment"] for r in rows)/count
        out[side+"_budget_hit_rate"] = sum(r["budget_hit"] for r in rows)/count
        out[side+"_generation_tokens"] = sum(r["generation_tokens"] for r in rows)
        out[side+"_generation_seconds"] = sum(r["generation_seconds"] for r in rows)
    out["paired_answer_containment_rate"] = sum(l["answer_results"]["A"]["answer_containment"] and r["answer_results"]["B"]["answer_containment"] for l,r in pairs)/count
    return out


def check_summary(logged, recomputed, name):
    require(set(logged) == set(recomputed), name + " summary fields differ")
    for key, value in recomputed.items():
        if value is None:
            require(logged[key] is None, name + "/" + key + " must be null")
        else:
            same_number(logged[key], value, name + "/" + key)


def audit_evaluation(directory, manifest):
    metrics = read_json(directory / "metrics.json")
    require(metrics.get("protocol") == "interface-cpu-vdb-eval-v1" and metrics.get("complete") is True,
            "Unsupported/incomplete interface evaluation")
    require(metrics.get("split") in {"dev", "confirm"}, "Evaluation split is not dev/confirm")
    require(manifest["configuration"].get("stage") == "eval", "Evaluation stage mismatch")
    require(metrics.get("frozen_online") is True and metrics.get("online_gradient_steps") == 0, "Online weights were not frozen")
    require(metrics["shared_weights_before"] == metrics["shared_weights_after"], "Evaluation weights changed")
    sha(metrics["shared_weights_before"], "shared weights")
    require(metrics["state_before"] == metrics["state_after"], "Evaluation training flags were not restored")
    for key in ("cache_sha256", "checkpoint_sha256"):
        sha(manifest.get(key), key)
    require(manifest.get("result") == metrics, "Evaluation manifest/result mismatch")
    assignments = read_json(directory / "assignments.json")
    selected = assignments["selected_case_ids"]
    require(len(selected) == len(set(selected)) == metrics["evaluated_facts"] and selected, "Duplicate/missing selected targets")
    all_assignments = {a["id"]: a for a in assignments["assignments"]}
    require(len(all_assignments) == len(assignments["assignments"]) == metrics["bank_records"], "Bank assignment size mismatch")
    require(set(selected) <= set(all_assignments), "Selected target absent from full bank")
    phases, grouped, signatures, costs = {}, defaultdict(dict), {}, defaultdict(float)
    method_costs = defaultdict(lambda: defaultdict(float))
    raw = read_jsonl(directory / "predictions.jsonl")
    require(set(metrics["phases"]) == set(PHASES), "Missing evaluation phases")
    allowed = {phase: METHODS[phase]+(("teacher",) if metrics["include_teacher"] else ()) for phase in PHASES}
    for method in ("real", "shuffled", "empty", "canonical_key", "teacher"):
        require(metrics["control_scope"][method] == [p for p in PHASES if method in allowed[p]], "Control scope mismatch")
    write_rows = read_jsonl(directory / "writes.jsonl")
    b_hashes, observations = {}, {}
    for row in write_rows:
        k = (row["phase"], row["bank_kind"], row["world"], row["id"])
        require(k not in observations, "Duplicate bank write")
        observations[k] = row
        if row["world"] == "B":
            b_hashes[k] = row["bank_hash"]
    for event in raw:
        phase, method, identity, world = (event[x] for x in ("phase", "method", "target_id", "world"))
        require(phase in allowed and method in allowed[phase] and identity in selected, "Undeclared phase/method/target")
        require(event["query_id"] == identity and not event["unrelated"], "Unexpected unrelated query")
        require(world in (("A_and_B",) if method == "empty" else ("A", "B")), "Invalid method/world")
        key = (phase, method, identity)
        require(world not in grouped[key], "Duplicate prediction world")
        grouped[key][world] = event
        a = all_assignments[identity]
        require((event["entity"], event["relation"]) == (a["entity"], a["relation"]), "Fact metadata mismatch")
        require(event["query_view"] == (a["q"] if phase[1] == "H" else 0) and
                event["support_view"] == (a["s"] if phase[0] == "H" else 0), "Style assignment mismatch")
        require(event["bank_hash_before"] == event["bank_hash_after"], "Read mutated bank")
        sha(event["bank_hash_before"], "bank")
        if method in {"real", "canonical_key", "teacher", "empty"}:
            kind = "canonical_key" if method == "canonical_key" else "real"
            expected_hash = metrics["phases"][phase]["base_hashes_before"][kind] if world in {"A", "A_and_B"} else b_hashes[(phase,kind,"B",identity)]
            require(event["bank_hash_before"] == expected_hash, "Read bank differs from declared intervention")
        context = event["teacher_context"]
        require((context is not None) == (method == "teacher"), "Teacher context leaked or is missing")
        if method == "teacher":
            require(context == observations[(phase,"real",world,identity)]["value_support"], "Teacher context differs from current observation")
        require(set(event["answer_results"]) == ({"A", "B"} if method == "empty" else {world}), "Wrong answer labels")
        for label, result in event["answer_results"].items():
            em = int(norm(event["prediction"]) == norm(result["answer"]))
            containment = int((" "+norm(result["answer"])+" ") in (" "+norm(event["prediction"])+" "))
            same_number(result["em"], em, "row EM")
            same_number(result["answer_containment"], containment, "row answer containment")
            number(result["tokens"], "answer tokens", integer=True, minimum=1)
            number(result["nll_sum"], "answer NLL")
            signature = (event["entity"], event["relation"], event["question"], result["answer"], event["query_view"], event["support_view"])
            sk = (phase, identity, label)
            require(sk not in signatures or signatures[sk] == signature, "Question/answer differs across methods")
            signatures[sk] = signature
        same_number(event["answer_scoring_tokens"], sum(x["tokens"] for x in event["answer_results"].values()), "Scoring token count")
        number(event["generation_tokens"], "generation tokens", integer=True)
        number(event["generation_seconds"], "generation seconds")
        require(event["budget_hit"] == (event["generation_tokens"] >= metrics["max_new_tokens"]), "Budget hit mismatch")
        trace = event["token_retrieval_sequence"]
        routed = method in {"real", "shuffled", "canonical_key"}
        if routed:
            require(trace and trace[0]["phase"] == "prefill" and all(r["step"] == i for i,r in enumerate(trace)), "Missing/invalid retrieval trace")
            selected_ids = trace[0]["selected_ids"]
            require(event["selected_ids"] == selected_ids, "Prefill retrieval mismatch")
            require(all(set(r["selected_ids"]) <= set(all_assignments) for r in trace), "Unknown retrieved ID")
            same_number(event["recall_at_1"], int(selected_ids[:1] == [identity]), "R1")
            same_number(event["recall_at_4"], int(identity in selected_ids[:4]), "R4")
            decode = [r for r in trace if r["phase"] == "decode"]
            same_number(event["decode_query_count"], len(decode), "Decode denominator")
            same_number(event["decode_correct_hits"], sum(identity in r["selected_ids"] for r in decode), "Decode hits")
        else:
            require(not trace and event["recall_at_1"] is None and event["recall_at_4"] is None and event["decode_query_count"] == 0,
                    "Nonretrieval method has routing evidence")
        for field in ("generation_tokens", "generation_seconds", "answer_scoring_tokens"):
            costs[field] += event[field]
            method_costs[method][field] += event[field]
        costs["generation_calls"] += 1
        method_costs[method]["generation_calls"] += 1
    vectors = {}
    for phase in PHASES:
        logged = metrics["phases"][phase]
        require(logged["base_hashes_before"] == logged["base_hashes_after"] and logged["read_only_verified"] is True, "Phase bank mutation")
        same_number(logged["count"], len(selected), "Phase targets")
        require(set(logged["methods"]) == set(allowed[phase]), "Phase methods differ")
        phases[phase], vectors[phase] = dict(methods={}), {}
        pairs_by_method = {}
        for method in allowed[phase]:
            pairs = []
            vector = {}
            for identity in selected:
                members = grouped[(phase, method, identity)]
                require(set(members) == ({"A_and_B"} if method == "empty" else {"A", "B"}), "Missing A/B prediction member")
                left, right = (members["A_and_B"],)*2 if method == "empty" else (members["A"], members["B"])
                require(norm(left["answer_results"]["A"]["answer"]) != norm(right["answer_results"]["B"]["answer"]), "Equal A/B answers")
                pairs.append((left,right))
                vector[identity] = int(norm(left["prediction"]) == norm(left["answer_results"]["A"]["answer"]) and
                                       norm(right["prediction"]) == norm(right["answer_results"]["B"]["answer"]))
            summary = paired_summary(pairs)
            check_summary(logged["methods"][method], summary, phase+"/"+method)
            phases[phase]["methods"][method], vectors[phase][method], pairs_by_method[method] = summary, vector, pairs
        if metrics["include_teacher"]:
            eligible = [identity for identity in selected if vectors[phase]["teacher"][identity]]
            acceptance = logged["teacher_acceptance"]
            require(acceptance["qualified_ids"] == eligible, "Teacher-qualified IDs mismatch")
            same_number(acceptance["strict_pair_qualified"], len(eligible), "Teacher denominator")
            same_number(acceptance["count"], len(selected), "Teacher total denominator")
            for method, pairs in pairs_by_method.items():
                if method != "teacher":
                    check_summary(logged["teacher_qualified_methods"][method], paired_summary([pair for identity,pair in zip(selected,pairs) if identity in eligible]), phase+"/teacher-qualified/"+method)
        else:
            require(logged["teacher_acceptance"] is None and logged["teacher_qualified_methods"] is None, "Teacher subset without teacher reads")
        phases[phase]["teacher_acceptance"] = logged["teacher_acceptance"]
    for field, value in costs.items():
        same_number(metrics[field], value, "Evaluation cost/"+field)
    identity_metadata = {identity: {k:all_assignments[identity][k] for k in ("entity","relation")} for identity in selected}
    key_signatures = [{"key":list(key),"value":list(value)} for key,value in sorted(signatures.items())]
    # Include *all* bank assignments and observation texts in the pairing hash.
    # Equal queried IDs alone does not establish equal background memory.
    comparability = dict(split=metrics["split"], cache_sha256=manifest["cache_sha256"],
        model_path=manifest["configuration"].get("model"),
        max_new_tokens=metrics["max_new_tokens"], selected_case_ids=sorted(selected),
        assignments_sha256=digest(assignments), content_sha256=digest(key_signatures),
        observations_sha256=digest([{k:r[k] for k in ("phase","bank_kind","world","id","key_support_view","value_support_view","key_support","value_support")} for r in write_rows]))
    record = dict(kind="evaluation", split=metrics["split"], phases=phases, costs=dict(costs),
        bank_records=metrics["bank_records"], evaluated_facts=len(selected), comparability=comparability,
        comparison_key=digest(comparability), identity_metadata=identity_metadata, pair_vectors=vectors,
        artifacts=artifacts(directory,["manifest.json","metrics.json","assignments.json","writes.jsonl","predictions.jsonl",*metrics["artifacts"]["banks"]]),
        bank_validation="Read/write hashes checked; bank files SHA256 recorded. Tensor-bank replay and model inference are not rerun.")
    record["costs"]["by_method"] = {key:dict(value) for key,value in method_costs.items()}
    record["costs"]["measurement_scope"] = (
        "Generation tokens/time and answer-scoring target tokens only. Prompt/context tokens, "
        "answer-scoring wall time, writer encoding and VDB write/read timings were not separately logged; "
        "these fields are not total inference tokens or total evaluation wall time.")
    record["group_clusters"] = infer_groups(signatures, identity_metadata)
    return record


def infer_groups(signatures, metadata):
    """Supplementary four-entity clusters only when full groups are observed."""
    entities = {x["entity"] for x in metadata.values()}
    relations = defaultdict(set)
    neighbors = {e:set() for e in entities}
    profiles = defaultdict(set)
    for identity, info in metadata.items():
        relations[info["entity"]].add(info["relation"])
        answers = tuple(sorted(norm(signatures[("CC",identity,w)][3]) for w in ("A","B")))
        profiles[(info["relation"],answers)].add(info["entity"])
    for group in profiles.values():
        for entity in group:
            neighbors[entity].update(group)
    components, remaining = [], set(entities)
    while remaining:
        todo, found = [min(remaining)], set()
        while todo:
            entity = todo.pop()
            if entity not in found:
                found.add(entity); todo.extend(neighbors[entity]-found)
        remaining -= found
        components.append(sorted(found))
    verified = all(len(c)==4 and all(len(relations[e])==4 for e in c) for c in components)
    clusters = {identity:"group-"+digest(next(c for c in components if info["entity"] in c))[:16]
                for identity,info in metadata.items()} if verified else None
    return dict(complete_four_entity_groups_verified=verified, assignments=clusters,
        observed_relations_per_entity={e:len(rs) for e,rs in sorted(relations.items())},
        note="Groups reconstructed by shared relation and unordered A/B answer pairs; all four entities and four relations required." if verified else
             "Four-entity groups cannot be certified from this target subset. Primary intervals cluster observed facts by entity only; shared-group dependence may make intervals too narrow.")


def audit_diagnostic(directory, manifest):
    result = read_json(directory/"results.json")
    require(result.get("protocol") == "interface-four-bank-dev-v1" and result.get("single_target_updates") is True, "Unknown four-bank diagnostic")
    require(manifest.get("results") == result and manifest.get("parameters_unchanged") is True, "Diagnostic manifest mismatch")
    sha(manifest.get("checkpoint_sha256"), "diagnostic checkpoint")
    raw = read_jsonl(directory/"predictions.jsonl")
    groups, styles, ids, costs = {}, set(), set(), defaultdict(float)
    for row in raw:
        key=(row["id"],row["style"],row["kind"],row["world"])
        require(key not in groups and row["world"] in {"A","B"}, "Duplicate/invalid diagnostic world")
        require(row["kind"] in {"KcVc","KhVc","KcVh","KhVh","teacher"}, "Unknown cross-bank condition")
        groups[key]=row; ids.add(row["id"]); styles.add(row["style"])
        same_number(row["em"], int(norm(row["prediction"])==norm(row["answer"])), "Diagnostic row EM")
        number(row["tokens"],"diagnostic generation tokens",integer=True)
        number(row["seconds"],"diagnostic generation seconds")
        costs["generation_calls"]+=1; costs["generation_tokens"]+=row["tokens"]; costs["generation_seconds"]+=row["seconds"]
        if row["kind"] != "teacher":
            require(row["bank_unchanged"] is True, "Diagnostic bank mutated")
            sha(row["bank_hash"],"diagnostic bank")
    require(len(ids)==manifest["size"] and len(styles)==2, "Diagnostic size/style mismatch")
    summaries, vectors, checks = {}, {}, 0
    for style in sorted(styles):
        summaries[style], vectors[style] = {}, {}
        for kind in ("KcVc","KhVc","KcVh","KhVh","teacher"):
            pairs=[]
            for identity in sorted(ids):
                require(all((identity,style,kind,w) in groups for w in ("A","B")), "Missing diagnostic A/B member")
                pairs.append(tuple(groups[(identity,style,kind,w)] for w in ("A","B")))
            pairs_correct={identity:int(a["em"] and b["em"]) for identity,(a,b) in zip(sorted(ids),pairs)}
            summary=dict(paired_correct=sum(pairs_correct.values()),n=len(ids),single_correct=sum(r["em"] for p in pairs for r in p),
                         first_correct=sum(r.get("first_correct",0) for p in pairs for r in p),recall1=sum(r.get("recall1",0) for p in pairs for r in p))
            check_summary(result["summary"][style][kind],summary,"Diagnostic "+style+"/"+kind)
            summaries[style][kind]=dict(**summary,paired_switch_em=summary["paired_correct"]/len(ids))
            vectors[style][kind]=pairs_correct
        for identity in sorted(ids):
            for world in ("A","B"):
                for key_prefix in ("Kc","Kh"):
                    a,b=(groups[(identity,style,key_prefix+value,world)] for value in ("Vc","Vh"))
                    require(a["indices"]==b["indices"] and a["weights"]==b["weights"], "Value-only first-prefix routing changed")
                    checks+=1
    same_number(result["value_only_route_equal_checks"],checks,"Routing equality checks")
    same_number(result["all_rows"],len(raw),"Diagnostic row count")
    return dict(kind="diagnostic",split="dev",phases=summaries,pair_vectors=vectors,
        identity_metadata={identity:dict(entity=identity,relation="assigned memory word") for identity in sorted(ids)},
        costs=dict(costs),value_only_route_equal_checks=checks,
        comparison_key=digest(dict(ids=sorted(ids),checkpoint=manifest["checkpoint_sha256"],seed=manifest["seed"])),
        artifacts=artifacts(directory,["manifest.json","results.json","predictions.jsonl"]),
        note="Development diagnostic despite confirmation-style template names. Routing equality applies at the same fixed first prefix only; free-generation prefixes may diverge. First-position forward/scoring cost was not logged.")


def audit_training(directory, manifest):
    status = read_json(directory/"training_status.json")
    require(status.get("complete") is True and manifest.get("result")==status,"Training manifest/status incomplete or differs")
    config=manifest["configuration"]
    rows=read_jsonl(directory/"training.jsonl")
    updates=number(status["updates"],"updates",integer=True,minimum=1)
    require(len(rows)==updates,"Missing/duplicate training updates")
    start=status.get("start_step",0)
    require([r["step"] for r in rows]==list(range(start+1,start+updates+1)),"Training step sequence mismatch")
    same_number(config["updates"],updates,"Configured updates")
    for key in ("cache_sha256","checkpoint_sha256"):
        sha(manifest.get(key),key)
    costs=dict(training_elapsed_seconds=status["elapsed_seconds"])
    number(costs["training_elapsed_seconds"],"training elapsed seconds")
    gates={}
    if config["stage"]=="writer":
        require(status["teacher_unchanged"] and status["frozen_state_unchanged"] and not status["targets_require_grad"],"Writer freeze violated")
        exposures=sum(len(row["facts"]) for row in rows)
        same_number(status["training_examples"],exposures,"Writer examples")
        costs.update(training_examples=exposures,optimizer_seconds=status["training_seconds"],backbone_training_input_tokens=0,
                     token_note="Cached writer features: examples are exposures, not tokens. Cache extraction cost is separate.")
        schedule=status["sample_schedule_sha256"]
        require(rows[-1]["sample_schedule_sha256"]==schedule,"Writer schedule digest differs")
    else:
        recomputed={key:sum(row[key] for row in rows) for key in status["totals"]}
        for key,value in recomputed.items():
            require(key.endswith("tokens") or key.endswith("forward_calls"),"Unexpected additive training total")
            same_number(status["totals"][key],value,"Training total/"+key)
        costs.update(token_totals=recomputed,processed_input_tokens=sum(recomputed.get(k,0) for k in ("student_input_tokens","teacher_input_tokens","rollout_input_tokens")),
                     token_note="Unpadded student+teacher+rollout inputs only. Padded rollout and overlapping gold/distillation/target counts are not added again.")
        exposures=sum(len(row["sample"]["targets"]) for row in rows)
        costs["pair_exposures"]=exposures
        for key in ("behavior_valid_pairs","legacy_behavior_clipped_valid_pairs","hidden_valid_pairs","teacher_first_token_joint_correct","student_first_token_joint_correct"):
            if all(key in row for row in rows):
                count=sum(row[key] for row in rows)
                require(0<=count<=exposures,"Gate count exceeds pair exposures")
                gates[key]=dict(count=count,denominator=exposures,rate=count/exposures)
        schedule=status["schedule_sha256"]
    sha(schedule,"training sample schedule")
    return dict(kind="training",split="train",updates=updates,start_step=start,costs=costs,gates=gates,
        final_diagnostics={k:rows[-1][k] for k in rows[-1] if isinstance(rows[-1][k],(int,float)) and not isinstance(rows[-1][k],bool)},
        sample_schedule_sha256=schedule,trainable_parameters=status.get("trainable_count",status.get("trainable_parameters")),
        artifacts=artifacts(directory,["manifest.json","training_status.json","training.jsonl"]))


def bootstrap(left,right,metadata,*,group_assignments=None):
    require(set(left)==set(right)==set(metadata) and left,"Bootstrap target sets differ")
    clusters=defaultdict(list)
    for identity in sorted(left):
        cluster=group_assignments[identity] if group_assignments else metadata[identity]["entity"]
        clusters[cluster].append(float(left[identity])-float(right[identity]))
    values=[(sum(clusters[k]),len(clusters[k])) for k in sorted(clusters)]
    rng=random.Random(BOOTSTRAP_SEED)
    draws=[]
    for _ in range(BOOTSTRAP_SAMPLES):
        sample=[values[rng.randrange(len(values))] for _ in values]
        draws.append(sum(x[0] for x in sample)/sum(x[1] for x in sample))
    draws.sort()
    return dict(delta=sum(left[k]-right[k] for k in left)/len(left),ci95=[draws[49],draws[1949]],
        facts=len(left),clusters=len(values),cluster_unit="four-entity group" if group_assignments else "observed entity",
        samples=BOOTSTRAP_SAMPLES,seed=BOOTSTRAP_SEED,
        note="Paired percentile cluster bootstrap; world A/B collapsed before resampling. Conditions/methods are not independent replicates; no seed or template-family uncertainty and no multiplicity adjustment.")


def parse_arguments(arguments):
    result={}
    i=0
    while i<len(arguments):
        token=arguments[i]
        if token.startswith("--"):
            key=token[2:].replace("-","_")
            if i+1<len(arguments) and not arguments[i+1].startswith("--"):
                result[key]=arguments[i+1]; i+=1
            else: result[key]=True
        i+=1
    return result


def classify(suite,name,config,kind):
    low=(suite+"/"+name).lower()
    scope="smoke" if "smoke" in low else "preflight" if "preflight" in low else "formal"
    if kind=="diagnostic": step="step1_four_bank"
    elif "writer" in suite or "step2" in suite or config.get("stage")=="writer": step="step2_writer"
    elif any(word in low for word in ("distill","_kd","_kd_","step4")) or config.get("method") in {"clip","normalized","hidden","on_policy"}: step="step4_kd"
    elif config.get("stage")=="prepare": step="preparation"
    elif config.get("stage")=="calibrate": step="calibration"
    else: step="step3_address"
    return scope,step


def discover(root):
    root=Path(root).resolve()
    require(root.is_dir(),"--runs-root must be an existing local filesystem directory (run this script on remote host for remote paths)")
    paths=set(root.rglob("interface_*/suite.json"))
    if root.name.startswith("interface_") and (root/"suite.json").exists(): paths.add(root/"suite.json")
    runs,pending,suites=[],[],[]
    for path in sorted(paths):
        if "source" in path.relative_to(root).parts: continue
        directory=path.parent; suite=read_json(path)
        require(suite.get("protocol")=="interface-experiment-v1","Unexpected interface suite protocol: "+str(path))
        plan=read_json(directory/"plan.json")
        sha(suite.get("plan_sha256"),"suite plan")
        names=[job["name"] for job in plan]
        require(len(set(names))==len(names),"Duplicate planned jobs")
        scheduled={job["name"]:job for job in suite["jobs"]}
        require(len(scheduled)==len(suite["jobs"]) and set(scheduled)<=set(names),"Duplicate/unplanned suite jobs")
        if suite.get("complete"):
            require(set(scheduled)==set(names) and all(j.get("exit_code")==0 and j.get("status")=="complete" for j in scheduled.values()),"Complete suite contains unfinished jobs")
        suites.append(dict(name=directory.name,path=str(directory),complete=suite.get("complete") is True,status=suite.get("status"),
                           scheduled=len(scheduled),planned=len(plan),artifacts=artifacts(directory,["suite.json","plan.json"])))
        for entry in plan:
            name=entry["name"]; job=scheduled.get(name,{}); run_dir=directory/name
            require(Path(name).name==name and name not in {".",".."},"Unsafe plan job name")
            if job.get("exit_code")!=0 or job.get("status")!="complete":
                pending.append(dict(suite=directory.name,job=name,status=job.get("status","not_started"),exit_code=job.get("exit_code")))
                continue
            manifest=read_json(run_dir/"manifest.json")
            require(manifest.get("complete") is True,"Completed suite job has incomplete manifest: "+str(run_dir))
            config=manifest.get("configuration",parse_arguments(entry.get("arguments",[])))
            kind="diagnostic" if entry["entry"]=="diagnostic" else config.get("stage","legacy")
            if kind=="diagnostic": record=audit_diagnostic(run_dir,manifest)
            elif kind=="eval": record=audit_evaluation(run_dir,manifest)
            elif kind in {"writer","train"}: record=audit_training(run_dir,manifest)
            else:
                # Preparations/calibrations are evidence, not performance arms.
                # Legacy counterfactual entries remain visibly outside this new protocol.
                record=dict(kind=kind,split="train" if kind in {"prepare", "calibrate"} else "not_applicable",
                            result=manifest.get("result"),costs={},artifacts=artifacts(run_dir,["manifest.json","calibration.json"]))
            scope,step=classify(directory.name,name,config,record["kind"])
            record.update(run_id=directory.name+"/"+name,suite=directory.name,job=name,directory=str(run_dir),
                suite_partial=not suite.get("complete",False),scope=scope,step=step,configuration=config,
                seed=int(config.get("seed",manifest.get("seed",42))),seed_source="manifest configuration",
                cache_sha256=manifest.get("cache_sha256"),checkpoint_sha256=manifest.get("checkpoint_sha256"),
                job_elapsed_seconds=job.get("elapsed_seconds"),parent_checkpoint=config.get("checkpoint"))
            runs.append(record)
    # Evaluation often uses default --seed42; infer training seed/arm from its
    # checkpoint's run, never confuse generation seed with optimization seed.
    by_id={r["run_id"]:r for r in runs}
    for r in runs:
        checkpoint=Path(r.get("parent_checkpoint") or "")
        parent_id="/".join(checkpoint.parts[-3:-1])
        parent=by_id.get(parent_id)
        r["training_parent"]=parent_id if parent else None
        if r["kind"]=="evaluation":
            if parent:
                r.update(seed=parent["seed"],seed_source="training checkpoint lineage",step=parent["step"])
                arm=re.sub(r"_(?:seed)?\d+$","",parent["job"])
            else:
                checkpoint_family=re.fullmatch(r"(last|mean|mlp)_(\d+)",checkpoint.parent.name)
                match=re.fullmatch(r"(.+?)_(\d+)_(?:dev|confirm)",r["job"])
                if checkpoint_family:
                    arm,seed=checkpoint_family.groups()
                    r.update(seed=int(seed),seed_source="checkpoint pathname; parent training artifact unavailable",step="step2_writer")
                elif match:
                    arm,seed=match.groups(); r.update(seed=int(seed),seed_source="explicit job-name fallback; parent training artifact unavailable")
                else: arm=re.sub(r"_(?:dev|confirm)$","",r["job"])
            r["arm"]=arm
        else: r["arm"]=re.sub(r"_(?:seed)?\d+$","",r["job"])
    return runs,pending,suites


def teacher_compatible(left,right):
    """A frozen-base teacher can be shared across training stages, not inputs."""
    return (left["scope"]==right["scope"] and left["comparison_key"]==right["comparison_key"]
            and left["configuration"].get("model")==right["configuration"].get("model")
            and left["comparability"]["max_new_tokens"]==right["comparability"]["max_new_tokens"])


def aggregate(runs,pending,suites):
    evaluations=[r for r in runs if r["kind"]=="evaluation"]
    comparisons,seed_groups=[],defaultdict(list)
    for run in evaluations:
        compatible=[r for r in evaluations if teacher_compatible(run,r)]
        for phase,scores in run["phases"].items():
            teachers=[r for r in compatible if "teacher" in r["pair_vectors"][phase]]
            if teachers:
                reference=teachers[0]
                require(all(r["pair_vectors"][phase]["teacher"]==reference["pair_vectors"][phase]["teacher"] for r in teachers),"Compatible teacher runs disagree")
                eligible={k for k,v in reference["pair_vectors"][phase]["teacher"].items() if v}
                scores["teacher_qualification"]=dict(source_run=reference["run_id"],shared_from_identical_inputs=reference is not run,
                    all_facts=run["evaluated_facts"],qualified=len(eligible),
                    methods={m:dict(correct=sum(vector[k] for k in eligible),denominator=len(eligible),
                                    paired_switch_em=sum(vector[k] for k in eligible)/len(eligible) if eligible else None)
                             for m,vector in run["pair_vectors"][phase].items() if m!="teacher"})
            else: scores["teacher_qualification"]=dict(source_run=None,qualified=None,note="No teacher evidence with identical split/bank/cases/questions/styles/answers/budget")
            for method in scores["methods"]:
                if method=="teacher": continue
                key=(run["scope"],run["step"],run["split"],run["arm"],phase,method,run["comparison_key"])
                seed_groups[key].append(run)
            for control in ("shuffled","empty","canonical_key"):
                if control in scores["methods"]:
                    comparisons.append(compare(run,run,phase,"real",control))
    for left,right in itertools.combinations(evaluations,2):
        if (left["scope"],left["step"],left["comparison_key"],left["seed"]) != (right["scope"],right["step"],right["comparison_key"],right["seed"]): continue
        if left["arm"]==right["arm"]: continue
        for phase in PHASES: comparisons.append(compare(left,right,phase,"real","real"))
    seed_summary=[]
    for key,items in sorted(seed_groups.items()):
        scope,step,split,arm,phase,method,comparison_key=key
        require(len({r["seed"] for r in items})==len(items),"Duplicate arm/seed evaluation; cannot treat reruns as independent seeds")
        values=[r["phases"][phase]["methods"][method]["paired_switch_em"] for r in items]
        seed_summary.append(dict(scope=scope,step=step,split=split,arm=arm,phase=phase,method=method,
            seeds=[r["seed"] for r in items],runs=[r["run_id"] for r in items],n_seeds=len(items),
            mean=statistics.mean(values),min=min(values),max=max(values),comparison_key=comparison_key,
            note="Mean/min/max over optimization seeds; a single seed provides no seed-robustness estimate."))
    diagnostics=[r for r in runs if r["kind"]=="diagnostic"]
    for r in diagnostics:
        for phase in r["pair_vectors"]:
            for left,right in (("KcVh","KcVc"),("KhVc","KcVc"),("KhVh","KcVc")):
                comparisons.append(compare(r,r,phase,left,right))
    cost_by_scope={}
    for scope in ("formal","smoke","preflight"):
        selected=[r for r in runs if r["scope"]==scope]
        cost_by_scope[scope]=dict(completed_jobs=len(selected),
            job_wall_seconds=sum(r["job_elapsed_seconds"] or 0 for r in selected),
            train_elapsed_seconds=sum(r["costs"].get("training_elapsed_seconds",0) for r in selected),
            training_processed_input_tokens=sum(r["costs"].get("processed_input_tokens",0) for r in selected),
            evaluation_generation_calls=sum(r["costs"].get("generation_calls",0) for r in selected),
            evaluation_generation_tokens=sum(r["costs"].get("generation_tokens",0) for r in selected),
            evaluation_generation_seconds=sum(r["costs"].get("generation_seconds",0) for r in selected),
            evaluation_answer_scoring_tokens=sum(r["costs"].get("answer_scoring_tokens",0) for r in selected),
            note="Physical completed jobs counted once, including warm jobs once; wall time and training/generation components overlap and must not be added. Historical initialization excluded. Incomplete jobs excluded.")
    return dict(protocol=PROTOCOL,audited_at=datetime.now(timezone.utc).isoformat(),
        audit_complete=True,all_suites_complete=bool(suites) and all(s["complete"] for s in suites),suites=suites,pending=pending,runs=runs,
        seed_summary=seed_summary,paired_comparisons=comparisons,cost_by_scope=cost_by_scope,
        writer_gate=writer_gate(evaluations),
        limitations=["Only interface_* suite manifests are discovered; historical counterfactual/scaling runs are never silently added.",
            "Smoke/preflight evidence is displayed separately and excluded from formal seed aggregation and formal costs.",
            "Development scores support model selection; confirmation scores are distinct. This audit does not authorize selection on confirmation.",
            "HC canonical_key and step1 four banks are diagnostic interventions, not deployable methods or upper bounds.",
            "Paired bootstrap clusters observed entity facts. Partial target sampling can omit relations or members of four-entity groups; supplementary group intervals require complete reconstructible groups.",
            "Development generation chooses the writer family only. The formal writer gate uses the subsequent fixed classic confirmation, HC +20 points and CC degradation at most 5 points in seeds 42/43/44; absent confirmation remains pending. Teacher eligibility is separate.",
            "Budget-hit means generated length reached the limit, not proof of truncation: EOS could be the last token.",
            "Evaluation prompt/context tokens, answer-scoring time, writer encoding and VDB write/read timings were not separately logged; generated and target-token counters are not total inference cost.",
            "SHA/provenance and row arithmetic are audited; this script does not replay tensors, model generation or training, and does not verify external historical checkpoint contents."])


def writer_gate(evaluations):
    """Dev selects a family; a later fixed confirm evaluates the formal gate.

    Never infer a formal failure from development. Confirmation is not used to
    reselect among families; this script only reports the recorded chosen arms.
    """
    reports={}
    for split in ("dev","confirm"):
        candidates=[r for r in evaluations if r["scope"]=="formal" and r["step"]=="step2_writer" and r["split"]==split]
        baselines=[r for r in candidates if r["arm"]=="baseline"]
        grouped=defaultdict(list)
        for r in candidates:
            if r["arm"]!="baseline": grouped[(r["arm"],r["comparison_key"])].append(r)
        split_reports=[]
        for (arm,key),members in sorted(grouped.items()):
            reference=[r for r in baselines if r["comparison_key"]==key]
            if not reference:
                split_reports.append(dict(arm=arm,comparison_key=key,status="pending_matching_baseline",seeds=[r["seed"] for r in members]))
                continue
            baseline=reference[0]
            require(all(r["pair_vectors"]==baseline["pair_vectors"] for r in reference),"Matching baseline repeats disagree")
            seed_rows=[]
            for r in members:
                hc=r["phases"]["HC"]["methods"]["real"]["paired_switch_em"]-baseline["phases"]["HC"]["methods"]["real"]["paired_switch_em"]
                cc=r["phases"]["CC"]["methods"]["real"]["paired_switch_em"]-baseline["phases"]["CC"]["methods"]["real"]["paired_switch_em"]
                item=dict(seed=r["seed"],run=r["run_id"],hc_delta=hc,cc_delta=cc)
                if split=="confirm": item["pass_gate"]=hc>=.2-1e-12 and cc>=-.05-1e-12
                seed_rows.append(item)
            ready={42,43,44} <= {r["seed"] for r in members}
            if split=="dev": status="diagnostic_only"
            elif not ready: status="pending_three_seeds"
            else: status="pass" if all(r["pass_gate"] for r in seed_rows if r["seed"] in {42,43,44}) else "fail"
            split_reports.append(dict(arm=arm,comparison_key=key,baseline_run=baseline["run_id"],baseline_shared_across_seeds=True,
                status=status,seed_results=seed_rows))
        reports[split]=dict(split=split,arms=split_reports,
            status=("available" if split_reports else "pending_development") if split=="dev" else
                   ("pending_confirmation" if not split_reports else "assessed" if all(r["status"] in {"pass","fail"} for r in split_reports) else "pending_evidence"))
    document=Path(__file__).resolve().parents[1]/"docs/interface_protocol.md"
    return dict(source="docs/interface_protocol.md#writer-validation",source_sha256=file_hash(document) if document.exists() else None,
        thresholds=dict(hc_absolute_gain_min=.2,cc_absolute_change_min=-.05,required_seeds=[42,43,44]),
        dev_diagnostic=reports["dev"],confirm_gate=reports["confirm"],
        note="Dev scores select the family only, with no formal pass/fail. Formal deployment gate is assessed on subsequent classic confirm versus its matching baseline in three seeds. No confirmation-based reselection or automatic execution.")


def compare(left,right,phase,method,control):
    l,r=left["pair_vectors"][phase][method],right["pair_vectors"][phase][control]
    require(left["identity_metadata"]==right["identity_metadata"],"Paired comparison metadata differs")
    value=bootstrap(l,r,left["identity_metadata"])
    groups=left.get("group_clusters",{})
    group_value=bootstrap(l,r,left["identity_metadata"],group_assignments=groups["assignments"]) if groups.get("complete_four_entity_groups_verified") else None
    return dict(scope=left["scope"],step=left["step"],split=left["split"],phase=phase,
        left_run=left["run_id"],right_run=right["run_id"],left_method=method,right_method=control,
        entity_bootstrap=value,four_entity_group_bootstrap=group_value)


def markdown(summary):
    pct=lambda x:"—" if x is None else f"{100*x:.1f}%"
    lines=["# VeRA-Mem interface experiment audit","",f"Audited {len(summary['runs'])} completed jobs; {len(summary['pending'])} incomplete or unstarted jobs are excluded from scores.", "",
           "The primary metric requires correctness in both A/B worlds of the same fact. Tables are recomputed from individual predictions; an output change alone is not success.", ""]
    for scope,title in (("formal","Formal experiments"),("smoke","Smoke: pipeline checks only"),("preflight","Preflight checks only")):
        items=[r for r in summary["runs"] if r["scope"]==scope]
        if not items: continue
        lines += ["## "+title,""]
        for step,label in (("step1_four_bank","Step 1: four banks"),("step2_writer","Step 2: writer"),("step3_address","Step 3: addressing/readout"),("step4_kd","Step 4: KD"),("preparation","Cache preparation"),("calibration","Teacher/loss calibration")):
            selected=[r for r in items if r["step"]==step]
            if not selected: continue
            lines += ["### "+label,""]
            evaluations=[r for r in selected if r["kind"]=="evaluation"]
            if evaluations:
                lines += ["| Run | Split | Seed | Phase | Real paired | A/B R@1 | Decode A/B | Shuffled | Empty | Teacher qualified/total |", "|---|---|---:|---|---:|---|---|---:|---:|---|"]
                for r in evaluations:
                    for phase,entry in r["phases"].items():
                        m=entry["methods"]; real=m["real"]; q=entry["teacher_qualification"]
                        lines.append(f"| {r['job']} | {r['split']} | {r['seed']} | {phase} | {real['both_correct']}/{real['count']} ({pct(real['paired_switch_em'])}) | {pct(real['a_recall_at_1'])}/{pct(real['b_recall_at_1'])} | {pct(real['a_decode_correct_residency'])}/{pct(real['b_decode_correct_residency'])} | {pct(m.get('shuffled',{}).get('paired_switch_em'))} | {pct(m.get('empty',{}).get('paired_switch_em'))} | {q['qualified'] if q['qualified'] is not None else 'not measured'}/{real['count']} |")
                lines += ["", "See summary.json for numerators/denominators in each teacher-qualified subset, canonical-key diagnostics, costs, and all paired outcome vectors.", ""]
            for r in selected:
                if r["kind"]=="diagnostic":
                    lines += ["Development diagnostic: confirmation in a template name does not make this a confirmation-set result.", "", "| Support style | Bank | Paired |", "|---|---|---|"]
                    for phase,methods in r["phases"].items():
                        for method,value in methods.items(): lines.append(f"| {phase} | {method} | {value['paired_correct']}/{value['n']} |")
                    lines += ["",f"Routing equality at the fixed first position was checked {r['value_only_route_equal_checks']} times. Later free-generation prefixes may diverge.", ""]
                elif r["kind"]=="training":
                    lines += [f"- {r['job']}: seed {r['seed']}, {r['updates']} new updates, starting step {r['start_step']}, {r['trainable_parameters']} trainable parameters; training {r['costs']['training_elapsed_seconds']:.1f} seconds."]
                elif r["kind"] not in {"evaluation","training"}: lines += [f"- {r['job']}: {r['kind']} completed; source hashes and results are in JSON."]
            lines += [""]
    lines += ["## Stability across seeds", "", "Aggregate only matching scope, step, split, and complete paired content. A single seed does not establish stability.", "", "| Scope | Step | Split | Arm | Phase/method | Seeds | Mean / min / max |", "|---|---|---|---|---|---|---|"]
    for row in summary["seed_summary"]:
        if row["method"]!="real": continue
        lines.append(f"| {row['scope']} | {row['step']} | {row['split']} | {row['arm']} | {row['phase']}/{row['method']} | {row['seeds']} | {pct(row['mean'])} / {pct(row['min'])} / {pct(row['max'])} |")
    lines += ["", "## Paired differences", "", "Fixed seed 68043; 2000 entity-cluster resamples. Values are paired EM differences in percentage points with 95% percentile intervals. Comparisons require the same split, complete case IDs/content/bank backgrounds/budgets; cross-run comparisons also require matching optimization seeds.", "", "| Scope/split | Left − right | Phase | Δ pp [95% CI] | Clusters |", "|---|---|---|---|---|"]
    for row in summary["paired_comparisons"]:
        v=row["entity_bootstrap"]
        lines.append(f"| {row['scope']}/{row['split']} | {row['left_run'].split('/')[-1]}:{row['left_method']} − {row['right_run'].split('/')[-1]}:{row['right_method']} | {row['phase']} | {100*v['delta']:+.1f} [{100*v['ci95'][0]:+.1f}, {100*v['ci95'][1]:+.1f}] | {v['clusters']} |")
    lines += ["", "## Costs and gates", "", "| Scope | Completed jobs | Job wall seconds | Train seconds | Train input tokens | Eval generated tokens | Answer scoring tokens |", "|---|---:|---:|---:|---:|---:|---:|"]
    for scope,cost in summary["cost_by_scope"].items():
        lines.append(f"| {scope} | {cost['completed_jobs']} | {cost['job_wall_seconds']:.1f} | {cost['train_elapsed_seconds']:.1f} | {cost['training_processed_input_tokens']} | {cost['evaluation_generation_tokens']} | {cost['evaluation_answer_scoring_tokens']} |")
    lines += ["", "Job wall time includes initialization and overlaps training/generation time; these cannot be added. Empty-bank A_and_B is generated once, without double counting; warm-up is counted once. Writer-cache sample exposures are not token counts.", "", "Training valid-pair gates and teacher free-generation qualification are recorded separately. Development selects only the writer. The subsequent classic confirmation gate compares three seeds against baseline: HC +20pp and CC decline <=5pp. Without confirmation, status stays pending; development failure cannot decide the formal gate, and confirmation cannot be used to reselect models.", ""]
    for key,label in (("dev_diagnostic","Development diagnostic"),("confirm_gate","Formal confirmation gate")):
        section=summary["writer_gate"][key]
        lines += [f"- {label}: {section['status']}."]
        for row in section["arms"]:
            lines += [f"  - Writer {row['arm']}: {row['status']}."]
    for r in summary["runs"]:
        if r.get("gates"):
            lines += [f"- {r['run_id']}: "+"; ".join(f"{k} {v['count']}/{v['denominator']}" for k,v in r["gates"].items())]
    lines += ["", "## Limitations", ""]+["- "+item for item in summary["limitations"]]
    return "\n".join(lines)+"\n"


def self_test():
    """Dependency-free regression checks; source artifacts are never touched."""
    import copy
    import tempfile
    import unittest

    class Checks(unittest.TestCase):
        def test_wrong_summary_rejected(self):
            with self.assertRaisesRegex(ValueError,"Incorrect"):
                check_summary(dict(count=2,both_correct=2),dict(count=2,both_correct=1),"fixture")

        def test_whole_answer_not_changed_output(self):
            def event(world,prediction,answer):
                return dict(prediction=prediction,answer_results={world:dict(answer=answer,tokens=3,nll_sum=6.,answer_containment=int(norm(answer) in norm(prediction)))},
                    recall_at_1=None,recall_at_4=None,decode_query_count=0,decode_correct_hits=None,budget_hit=False,generation_tokens=4,generation_seconds=.1)
            pairs=[(event("A","WRONG A","apple amber birch"),event("B","WRONG B","river cedar flint"))]
            score=paired_summary(pairs)
            self.assertEqual(score["paired_switch_em"],0)
            self.assertEqual(score["changed_rate"],1)
            self.assertEqual(score["changed_but_not_both_correct_rate"],1)

        def test_bootstrap_pairs_entities_and_rejects_intersection(self):
            meta={str(i):dict(entity=f"e{i//4}") for i in range(8)}
            a={k:int(int(k)<4) for k in meta}; b={k:0 for k in meta}
            value=bootstrap(a,b,meta)
            self.assertEqual(value["clusters"],2)
            self.assertEqual(value["delta"],.5)
            self.assertEqual(value["ci95"],[0.,1.])
            self.assertEqual(value,bootstrap(a,b,meta))
            with self.assertRaisesRegex(ValueError,"sets differ"):
                bootstrap(a,{k:v for k,v in b.items() if k!="0"},meta)

        def test_groups_require_four_entities_and_all_relations(self):
            meta={f"{e}-{r}":dict(entity=f"e{e}",relation=f"r{r}") for e in range(4) for r in range(4)}
            signs={("CC",identity,w):(item["entity"],item["relation"],"q",item["relation"]+w,0,0) for identity,item in meta.items() for w in ("A","B")}
            self.assertTrue(infer_groups(signs,meta)["complete_four_entity_groups_verified"])
            del meta["3-3"]
            self.assertFalse(infer_groups(signs,meta)["complete_four_entity_groups_verified"])

        def test_scope_and_stage_are_separate(self):
            self.assertEqual(classify("interface_writer_smoke_20261006","last_42_dev",{},"evaluation"),("smoke","step2_writer"))
            self.assertEqual(classify("interface_diagnostic_20261006","four_bank",{},"diagnostic"),("formal","step1_four_bank"))
            self.assertEqual(classify("interface_experiment_20261006","clip_42",dict(method="clip"),"training"),("formal","step4_kd"))
            self.assertEqual(classify("interface_experiment_20261006","normalized_42",dict(method="normalized"),"training"),("formal","step4_kd"))
            self.assertEqual(classify("interface_step2_confirm_20261006","chosen_42_confirm",{},"evaluation"),("formal","step2_writer"))

        def test_teacher_can_cross_steps_only_with_identical_model_inputs_and_budget(self):
            teacher=dict(scope="formal",step="step3_address",comparison_key="identical",
                         configuration=dict(model="fixed-model"),comparability=dict(max_new_tokens=32))
            student=dict(teacher,step="step4_kd")
            self.assertTrue(teacher_compatible(teacher,student))
            self.assertFalse(teacher_compatible(teacher,dict(student,configuration=dict(model="other-model"))))
            self.assertFalse(teacher_compatible(teacher,dict(student,comparability=dict(max_new_tokens=16))))
            self.assertFalse(teacher_compatible(teacher,dict(student,comparison_key="different-cases")))
            self.assertFalse(teacher_compatible(teacher,dict(student,scope="smoke")))

        def test_writer_gate_separates_dev_selection_from_three_seed_confirmation(self):
            def row(arm,seed,hc,cc,split="dev"):
                return dict(scope="formal",step="step2_writer",split=split,arm=arm,seed=seed,
                    run_id=f"suite/{arm}_{seed}_{split}",comparison_key="same",pair_vectors={},
                    phases={p:dict(methods=dict(real=dict(paired_switch_em=v))) for p,v in (("HC",hc),("CC",cc))})
            baseline=row("baseline",42,0.,1.)
            members=[row("mean",seed,.25,.96) for seed in (42,43,44)]
            initial=writer_gate([baseline,*members])
            self.assertEqual(initial["dev_diagnostic"]["arms"][0]["status"],"diagnostic_only")
            self.assertEqual(initial["confirm_gate"]["status"],"pending_confirmation")
            self.assertNotIn("pass_gate",initial["dev_diagnostic"]["arms"][0]["seed_results"][0])
            confirmed=[row("baseline",42,0.,1.,"confirm"),*[row("mean",seed,.25,.96,"confirm") for seed in (42,43,44)]]
            self.assertEqual(writer_gate([baseline,*members,*confirmed])["confirm_gate"]["arms"][0]["status"],"pass")
            self.assertEqual(writer_gate([baseline,*members,*confirmed[:-1]])["confirm_gate"]["arms"][0]["status"],"pending_three_seeds")
            members[-1]=row("mean",44,.19,.96)
            self.assertEqual(writer_gate([baseline,*members,*confirmed])["confirm_gate"]["arms"][0]["status"],"pass")
            confirmed[-1]=row("mean",44,.19,.96,"confirm")
            self.assertEqual(writer_gate([baseline,*members,*confirmed])["confirm_gate"]["arms"][0]["status"],"fail")

        def test_empty_root_excludes_historical(self):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/"counterfactual_historical"; p.mkdir(); (p/"suite.json").write_text("{}")
                self.assertEqual(discover(tmp),([],[],[]))

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args(argv)
    if args.self_test: return self_test()
    if args.runs_root is None or args.output is None: parser.error("--runs-root and --output are required")
    # Any integrity failure aborts before writing a new success report.
    runs,pending,suites=discover(args.runs_root)
    summary=aggregate(runs,pending,suites)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,text in (("summary.json",json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n"),("report.md",markdown(summary))):
        path=args.output/name; temporary=path.with_suffix(path.suffix+".partial")
        temporary.write_text(text,encoding="utf-8"); temporary.replace(path)
    print(json.dumps(dict(audit_complete=True,suites=len(suites),completed_jobs=len(runs),pending_jobs=len(pending),output=str(args.output))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
