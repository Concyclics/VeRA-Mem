"""Read-only answer diagnostics for completed local cold-start evaluations.

python scripts/diagnose_coldstart_answers.py --runs-root ../runs/xtrah100 --output ../plans/coldstart_answer_diagnostics.json
python scripts/diagnose_coldstart_answers.py --self-test

Standard library only. No model/tensors/network; no raw answers, predictions,
questions or fact identifiers are copied into the aggregate output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

PROTOCOL = "coldstart-answer-diagnostics-v1"
RUN_PROTOCOL = "dictionary-coldstart-v1"
PHASES = ("CC", "CH", "HC", "HH")
MAIN_ARMS = {"no_warm", "matched_warm", "wiki_warm", "fixed", "learned_sparse",
             "dense_warm", "straight_through", "wiki_joint"}
COUNTS = ("normalized_em", "full_answer_containment", "first_word_correct", "empty_output",
          "three_raw_words", "three_normalized_words", "budget_hits")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def normalize(value):
    require(isinstance(value, str), "Expected an answer/prediction string")
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in value).split())


def exclusion_markers(directory, root):
    """Skip ancestors before reading manifests; do not publish marker prose."""
    directory, root = Path(directory).resolve(), Path(root).resolve()
    require(directory.is_relative_to(root), "Exclusion lookup outside runs root")
    found = []
    while True:
        path = directory / "analysis_excluded.json"
        if path.exists():
            metadata = read_json(path)
            require(isinstance(metadata, dict) and isinstance(metadata.get("reason"), str)
                    and metadata["reason"].strip(), "Malformed analysis exclusion marker")
            found.append(dict(path=str(path), sha256=sha(path)))
        if directory == root:
            return found
        directory = directory.parent


def classify(directory, root, arm):
    if any("smoke" in part.lower() or "preflight" in part.lower() for part in (root.name,*directory.relative_to(root).parts)):
        return "smoke_or_preflight"
    if arm == "wiki_teacher_repair":
        return "supplementary_teacher_feasibility"
    return "original_experiment" if arm in MAIN_ARMS else None


def score(prediction, answer, tokens, budget):
    expected, predicted = normalize(answer), normalize(prediction)
    require(expected, "Empty normalized gold answer")
    require(type(tokens) is int and tokens >= 0 and type(budget) is int and budget > 0, "Invalid generation count/budget")
    em = predicted == expected
    contained = " " + expected + " " in " " + predicted + " "
    first = bool(predicted) and predicted.split()[0] == expected.split()[0]
    category = ("normalized_exact" if em else "answer_contained_with_extra_text" if contained
                else "first_word_matches_but_full_answer_absent" if first else "full_answer_absent_and_first_word_wrong_or_empty")
    return dict(normalized_em=int(em), full_answer_containment=int(contained), first_word_correct=int(first),
                empty_output=int(not predicted), three_raw_words=int(len(prediction.split()) == 3),
                three_normalized_words=int(len(predicted.split()) == 3), budget_hits=int(tokens >= budget),
                raw_words=len(prediction.split()), normalized_words=len(predicted.split()),
                characters=len(prediction), generation_tokens=tokens, error_category=category)


def aggregate(scores):
    require(scores, "Cannot aggregate an empty prediction list")
    counts = {key: sum(row[key] for row in scores) for key in COUNTS}
    lengths = {}
    for field in ("raw_words", "normalized_words", "characters", "generation_tokens"):
        values = sorted(row[field] for row in scores)
        lengths[field] = dict(total=sum(values), mean=sum(values)/len(values), minimum=values[0], maximum=values[-1],
                             histogram=dict(sorted(Counter(values).items())))
    return dict(prediction_exposures=len(scores), counts=counts,
                rates={key: value/len(scores) for key, value in counts.items()},
                error_categories=dict(sorted(Counter(row["error_category"] for row in scores).items())), lengths=lengths)


def diagnose_run(directory, manifest, scope, suite_path=None):
    metrics_path, raw_path, assignment_path = [directory / name for name in ("metrics.json", "predictions.jsonl", "assignments.json")]
    inputs = [directory/"manifest.json", metrics_path, raw_path, assignment_path]
    if suite_path is not None:inputs.append(suite_path)
    before = {str(path): sha(path) for path in inputs}
    metrics, assignment = read_json(metrics_path), read_json(assignment_path)
    require(metrics.get("complete") is True and metrics.get("protocol") == "interface-cpu-vdb-eval-v1", "Wrong/incomplete evaluation metrics")
    require(manifest.get("result") == metrics, "Manifest and metrics differ")
    require(metrics.get("coldstart", {}).get("protocol") == "coldstart-cpu-banks-v1", "Not a coldstart CPU-bank evaluation")
    selected = assignment["selected_case_ids"]
    require(selected and len(selected) == len(set(selected)) == metrics["evaluated_facts"], "Selected case count/identity mismatch")
    expected = {(phase, identity, world) for phase in PHASES for identity in selected for world in ("A", "B")}
    observed, predictions = {}, {}
    raw_count = 0
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():continue
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            raw_count += 1
            if row["method"] != "real":continue
            phase, identity, world = (row[key] for key in ("phase", "target_id", "world"))
            key = phase, identity, world
            require(key in expected and key not in observed, "Unexpected/duplicate real prediction")
            require(row["query_id"] == identity and row.get("unrelated") is False, "Real query identity mismatch")
            require(row.get("teacher_context") is None, "Student received teacher context")
            require(set(row["answer_results"]) == {world}, "Wrong real answer label")
            answer = row["answer_results"][world]
            actual = score(row["prediction"], answer["answer"], row["generation_tokens"], metrics["max_new_tokens"])
            require(answer["em"] == actual["normalized_em"], "Logged EM differs from prediction")
            require(answer["answer_containment"] == actual["full_answer_containment"], "Logged containment differs from prediction")
            require(row["budget_hit"] == bool(actual["budget_hits"]), "Logged budget hit differs from token count")
            observed[key] = actual
            predictions[key] = normalize(row["prediction"])
    require(set(observed) == expected, "Incomplete real A/B prediction grid")
    require(raw_count == metrics["generation_calls"], "Raw generation count differs from complete metrics")
    phases = {}
    for phase in PHASES:
        pair_counts = Counter()
        for identity in selected:
            left, right = observed[(phase,identity,"A")], observed[(phase,identity,"B")]
            pair_counts["both_normalized_em"] += left["normalized_em"] and right["normalized_em"]
            pair_counts["both_full_answer_containment"] += left["full_answer_containment"] and right["full_answer_containment"]
            pair_counts["both_first_word_correct"] += left["first_word_correct"] and right["first_word_correct"]
            changed = predictions[(phase,identity,"A")] != predictions[(phase,identity,"B")]
            pair_counts["same_normalized_output_across_worlds"] += not changed
            pair_counts["output_changed_but_not_both_em"] += changed and not (left["normalized_em"] and right["normalized_em"])
        phases[phase] = dict(A=aggregate([observed[(phase,i,"A")] for i in selected]),
                            B=aggregate([observed[(phase,i,"B")] for i in selected]),
                            combined=aggregate([observed[(phase,i,w)] for i in selected for w in ("A","B")]),
                            paired=dict(case_exposures=len(selected), counts=dict(pair_counts),
                                        rates={k:v/len(selected) for k,v in pair_counts.items()}))
    after = {str(path): sha(path) for path in inputs}
    require(before == after, "Input artifact changed while diagnosing answers")
    cfg = manifest["configuration"]
    split = "train_probe" if cfg.get("domain") == "synthetic_train_probe" else metrics["split"]
    require(split in {"train_probe", "dev", "confirm"}, "Undeclared evaluation split")
    mode = cfg.get("base_mode", "full")
    require(mode == metrics["coldstart"]["base_mode"], "Foundation control mode mismatch")
    return dict(run_dir=str(directory), experiment_scope=scope, arm=cfg["arm"], domain=cfg["domain"], split=split,
                base_mode=mode, seed=cfg.get("seed"), method="real", max_new_tokens=metrics["max_new_tokens"],
                checkpoint_sha256=manifest["checkpoint_sha256"], cache_sha256=manifest["cache_sha256"],
                case_set_sha256=digest(sorted(selected)), unique_facts_within_run=len(selected),
                total_prediction_exposures=len(observed), paired_case_exposures=4*len(selected),
                all_phases_descriptive=aggregate(list(observed.values())), phases=phases,
                input_sha256=before, input_hashes_unchanged=True)


def diagnose(runs_root):
    root = Path(runs_root).resolve()
    require(root.is_dir(), "Missing runs root")
    results, pending, excluded, ignored = [], [], [], []
    for path in sorted(root.rglob("manifest.json")):
        parts = path.relative_to(root).parts
        if any(part in {"source", "source_snapshot", "source_snapshots", ".git"} for part in parts):continue
        if not any(part.startswith("coldstart_") for part in (root.name,*parts[:-1])):continue
        markers = exclusion_markers(path.parent, root)
        if markers:
            excluded.append(dict(run_dir=str(path.parent), markers=markers, manifest_not_read=True))
            continue
        manifest = read_json(path)
        if manifest.get("protocol") != RUN_PROTOCOL or manifest.get("stage") != "eval":continue
        if manifest.get("complete") is not True:
            pending.append(dict(run_dir=str(path.parent), reason="Incomplete manifest; no predictions/metrics read"))
            continue
        cfg = manifest["configuration"]
        require(cfg.get("stage") == "eval", "Stage mismatch")
        scope = classify(path.parent, root, cfg.get("arm"))
        if scope is None:
            ignored.append(dict(run_dir=str(path.parent), reason="Unrecognized nonformal arm; no predictions read"))
            continue
        suite_path = path.parent.parent / "suite.json"
        if suite_path.is_file():
            suite = read_json(suite_path)
            require(suite.get("protocol") == "coldstart-experiment-v1", "Wrong suite protocol")
            jobs = [job for job in suite["jobs"] if job["name"] == path.parent.name]
            require(len(jobs) == 1, "Suite does not uniquely identify evaluation job")
            job = jobs[0]
            if job.get("status") != "complete" or job.get("exit_code",job.get("process_exit_code")) != 0:
                pending.append(dict(run_dir=str(path.parent), reason="Launcher job not complete; no predictions/metrics read"))
                continue
            require(job.get("manifest_sha256") == sha(path), "Suite/child manifest SHA mismatch")
        else:suite_path = None
        results.append(diagnose_run(path.parent, manifest, scope, suite_path))
    identities = [(r["experiment_scope"],r["arm"],r["domain"],r["split"],r["base_mode"],r["checkpoint_sha256"],r["cache_sha256"]) for r in results]
    require(len(identities) == len(set(identities)), "Duplicate completed evaluation cell requires explicit run selection")
    scopes = defaultdict(list)
    for index,result in enumerate(results):scopes[result["experiment_scope"]].append(index)
    return dict(protocol=PROTOCOL,created_at=datetime.now(timezone.utc).isoformat(),checks_passed=True,
        complete=bool(results) and not pending, completeness_scope="Discovered recognized local evaluations only; not planned experiment coverage",
        method="real",per_arm_domain_split=results,scopes={key:dict(record_indices=value,runs=len(value)) for key,value in scopes.items()},
        pending=pending,excluded=excluded,ignored=ignored,
        definitions=dict(normalization="NFKC + casefold + Unicode punctuation replaced with spaces + whitespace collapse; articles preserved",
            containment="Whole normalized answer phrase occurs with word boundaries in normalized output",
            first_word="First normalized output word equals first normalized gold word; this is not first generated token accuracy",
            budget_hit="generation_tokens >= max_new_tokens; possible truncation, not proof that EOS was absent",
            error_categories="Disjoint: normalized exact; contained with extra text; correct first word but answer absent; wrong/empty first word and answer absent"),
        limitations=["Answers, predictions, questions and case identifiers are not copied into this aggregate artifact; input paths and SHA256 identify inspectable private evidence.",
            "Containment without full EM can indicate extra text, but does not establish that formatting alone explains an error or that the model selected the correct fact.",
            "All-phase totals count repeated exposures to the same facts. A/B worlds, styles, model arms and compression controls are correlated; no independent-sample inference, pooled model accuracy or confidence interval is produced.",
            "Smoke, original eight-arm evidence and the optional teacher follow-up remain separate. Train-probe results never count as held-out dev or confirmation.",
            "Only the real retrieval method is diagnosed. This script does not reproduce model generation, validate bank tensors, or replace the full experiment audit."])


def write_output(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def self_test():
    import tempfile
    import unittest
    class Checks(unittest.TestCase):
        def setUp(self):
            self.temp = tempfile.TemporaryDirectory()
            self.root = Path(self.temp.name)/"coldstart_fixture"
            self.root.mkdir()
        def tearDown(self):self.temp.cleanup()
        def fixture(self, name="formal", arm="straight_through"):
            output = self.root/name
            output.mkdir()
            a,b = "Amber blue cedar","Silver gold delta"
            outputs = {"CC":(a.upper()+".","Answer: "+b),"CH":("Amber wrong words","wrong tokens here"),"HC":("",b),"HH":(a,a)}
            rows=[]
            for phase in PHASES:
                for world,answer,prediction in zip(("A","B"),(a,b),outputs[phase]):
                    tokens=32 if (phase,world)==("HH","B") else len(prediction.split())+1
                    scores=score(prediction,answer,tokens,32)
                    rows.append(dict(method="real",phase=phase,target_id="private-fact-id",query_id="private-fact-id",world=world,
                        unrelated=False,teacher_context=None,prediction=prediction,generation_tokens=tokens,
                        budget_hit=bool(scores["budget_hits"]),answer_results={world:dict(answer=answer,
                            em=scores["normalized_em"],answer_containment=scores["full_answer_containment"])}))
            metrics=dict(protocol="interface-cpu-vdb-eval-v1",complete=True,split="dev",evaluated_facts=1,max_new_tokens=32,
                         generation_calls=8,coldstart=dict(protocol="coldstart-cpu-banks-v1",base_mode="full"))
            manifest=dict(protocol=RUN_PROTOCOL,stage="eval",complete=True,checkpoint_sha256="a"*64,cache_sha256="b"*64,
                configuration=dict(stage="eval",arm=arm,domain="wikipedia",base_mode="full"),result=metrics)
            (output/"manifest.json").write_text(json.dumps(manifest))
            (output/"metrics.json").write_text(json.dumps(metrics))
            (output/"assignments.json").write_text(json.dumps(dict(selected_case_ids=["private-fact-id"])))
            (output/"predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            return output,manifest,rows
        def test_normalization_boundaries_and_wrong_remaining_words(self):
            self.assertEqual(score("ＡＭＢＥＲ, blue cedar!","Amber blue cedar",5,32)["normalized_em"],1)
            self.assertEqual(score("xAmber blue cedar","Amber blue cedar",5,32)["full_answer_containment"],0)
            self.assertEqual(score("Amber wrong words","Amber blue cedar",5,32)["first_word_correct"],1)
            self.assertEqual(score("\n","Amber blue cedar",1,32)["empty_output"],1)
        def test_full_grid_recomputation_and_no_raw_answer_disclosure(self):
            self.fixture()
            result=diagnose(self.root)
            row=result["per_arm_domain_split"][0]
            self.assertEqual(row["total_prediction_exposures"],8)
            self.assertEqual(row["unique_facts_within_run"],1)
            counts=row["all_phases_descriptive"]["counts"]
            self.assertEqual([counts[k] for k in ("normalized_em","full_answer_containment","first_word_correct","budget_hits")],[3,4,4,1])
            self.assertEqual(row["phases"]["CC"]["paired"]["counts"]["both_full_answer_containment"],1)
            self.assertEqual(row["phases"]["HH"]["paired"]["counts"]["same_normalized_output_across_worlds"],1)
            serialized=json.dumps(result)
            for private in ("Amber","Silver","private-fact-id","wrong tokens here"):
                self.assertNotIn(private,serialized)
            self.assertEqual(len(row["input_sha256"]),4)
        def test_missing_duplicate_and_logged_error_rejected(self):
            output,manifest,rows=self.fixture()
            path=output/"predictions.jsonl"
            for changed in (rows[:-1],rows+[rows[0]]):
                path.write_text("\n".join(json.dumps(row) for row in changed)+"\n")
                with self.assertRaises(ValueError):diagnose(self.root)
            rows[0]["answer_results"]["A"]["em"]=0
            path.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            with self.assertRaisesRegex(ValueError,"Logged EM"):diagnose(self.root)
        def test_excluded_and_partial_confirmation_raw_is_not_opened(self):
            for name,excluded in (("old",True),("partial",False)):
                output=self.root/name
                output.mkdir()
                if excluded:
                    (output/"analysis_excluded.json").write_text(json.dumps(dict(reason="Excluded experiment")))
                    output=output/"nested_eval"
                    output.mkdir()
                    (output/"manifest.json").write_text("unreadable excluded manifest")
                else:
                    (output/"manifest.json").write_text(json.dumps(dict(protocol=RUN_PROTOCOL,stage="eval",complete=False)))
                (output/"predictions.jsonl").write_text("must not be read")
                (output/"metrics.json").write_text("must not be read")
            result=diagnose(self.root)
            self.assertFalse(result["complete"])
            self.assertEqual(len(result["excluded"]),1)
            self.assertEqual(len(result["pending"]),1)
            self.assertFalse(result["per_arm_domain_split"])
        def test_scopes_separate_original_supplementary_and_smoke(self):
            self.fixture()
            self.fixture("followup","wiki_teacher_repair")
            self.fixture("coldstart_smoke_check","smoke")
            result=diagnose(self.root)
            self.assertEqual({k:v["runs"] for k,v in result["scopes"].items()},
                dict(original_experiment=1,supplementary_teacher_feasibility=1,smoke_or_preflight=1))
        def test_incomplete_launcher_does_not_open_completed_child_predictions(self):
            output,manifest,rows=self.fixture()
            (self.root/"suite.json").write_text(json.dumps(dict(protocol="coldstart-experiment-v1",jobs=[dict(name=output.name,status="running")])) )
            (output/"predictions.jsonl").write_text("must not read until launcher finishes")
            result=diagnose(self.root)
            self.assertEqual(len(result["pending"]),1)
            self.assertFalse(result["per_arm_domain_split"])
        def test_output_never_overwrites(self):
            path=self.root/"result.json"
            write_output(dict(marker="original"),path)
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):write_output(dict(marker="replacement"),path)
            self.assertEqual(path.read_bytes(),before)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args(argv)
    if args.self_test:return self_test()
    if args.runs_root is None or args.output is None:parser.error("--runs-root and --output are required")
    require(not args.output.exists(),"Refusing to overwrite output")
    result=diagnose(args.runs_root)
    result["audit_script_sha256"]=sha(Path(__file__))
    write_output(result,args.output)
    print(json.dumps(dict(complete=result["complete"],runs=len(result["per_arm_domain_split"]),
                         scopes={k:v["runs"] for k,v in result["scopes"].items()},output=str(args.output))))
    return 0


if __name__=="__main__":raise SystemExit(main())
