"""CPU foundation controls, causal episodic reuse, and caller-state restoration."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from vera_mem.coldstart_eval import evaluate_coldstart, prepare_base_bank, _Usage
from vera_mem.dictionary_vera import DictionaryVeRA
from vera_mem.run import tensor_digest
from test_interface_eval import FakeBackend, packet


class TinyDictionary(DictionaryVeRA):
    def __init__(self):
        super().__init__(3, 3, rank=1, key_dim=2, top_k=1, base_size=4,
                         base_top_k=2, seed=17, base_trainable=False,
                         routing_mode="straight_through")

    def encode_query(self, x):
        return torch.nn.functional.normalize(x[..., :2], dim=-1)

    def encode_key(self, x):
        return x[..., :2]

    def encode_value(self, x):
        return x[..., 2:]


class FoundationBackend(FakeBackend):
    """Exercise the real CPU foundation read during the existing causal fixture."""
    def __init__(self, module, *, fail=None):
        super().__init__(module)
        self.fail = fail

    def read_foundation(self):
        m = self.vector_vera
        assert not m.training and m.routing_mode == "sparse"
        assert m.base_cpu_override is not None
        m.delta_from_value(torch.tensor([[[1., .5, 0.]]]), torch.zeros(1, 1, 1))

    def generate(self, *args, **kwargs):
        if self.mode != "none" or self.fail == "teacher_read":
            self.read_foundation()
        return super().generate(*args, **kwargs)

    def score(self, *args, **kwargs):
        if self.mode != "none":
            self.read_foundation()
        if self.fail == "exception":
            raise RuntimeError("intentional scoring failure")
        if self.fail == "bank":
            self.vector_vera.base_cpu_override._values[0] += 1
        return super().score(*args, **kwargs)


def run(tmp_path, mode="full", teacher=True, fail=None, **options):
    m = TinyDictionary()
    backend = FoundationBackend(m, fail=fail)
    result = evaluate_coldstart(backend, m, packet(), tmp_path,
        include_teacher=teacher, max_cases=2, base_mode=mode, **options)
    return result, m, backend


def test_frozen_two_cpu_banks_trace_teacher_and_original_state_restore(tmp_path):
    result, m, backend = run(tmp_path)
    c = result["coldstart"]
    assert result["complete"] and c["foundation_cpu"] and c["episodic_cpu"]
    assert c["original_shared_weights_before"] == c["original_shared_weights_after"] == tensor_digest(m)
    assert c["original_state_before"] == c["original_state_after"]
    assert m.training and m.routing_mode == "straight_through"
    assert not m.base_keys.requires_grad and m.Wv.weight.requires_grad
    assert m.base_cpu_override is None and m.last_base_retrieval is None
    assert backend.mode == "original" and "generate" not in backend.__dict__ and "score" not in backend.__dict__
    assert c["bank_hash_before"] == c["bank_hash_after"]
    assert c["alpha_base"] == .25 and c["foundation_effective"]
    assert c["foundation_effect_status"] == "weighted_branch_enabled"
    assert c["foundation_weighted_top_k"] == 2 and c["maximum_weighted_records"] == 3
    assert result["phases"]["CC"]["methods"]["real"]["paired_switch_em"] == 1
    assert result["phases"]["HC"]["methods"]["canonical_key"]["paired_switch_em"] == 1
    events = [json.loads(x) for x in (tmp_path/"foundation_usage.jsonl").read_text().splitlines()]
    predictions = [json.loads(x) for x in (tmp_path/"predictions.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["kind"] == "generation"]) == len(predictions) == 48
    assert sum(e["query_positions"] for e in events) == c["usage"]["query_positions"] == 68
    assert all(e["search_calls"] == (0 if e["teacher"] else 1) for e in events)
    assert all(e["teacher"] == (predictions[e["prediction_index"]]["method"] == "teacher") for e in events)
    for phase in result["phases"].values():
        assert phase["teacher_acceptance"]["strict_pair_qualified"] == 2
    snapshot = torch.load(tmp_path/"foundation.pt", weights_only=True)
    assert torch.equal(snapshot["original_values"], m.base_values)
    assert not snapshot["values_rms_recalibrated"] and not c["dynamic_records_compressed"]


def test_foundation_off_keeps_episodic_predictions_and_has_zero_hit_mass(tmp_path):
    result, _, _ = run(tmp_path, "off", teacher=False)
    c = result["coldstart"]
    assert c["active_records"] == c["usage"]["records"] == 0
    assert not c["foundation_effective"] and c["foundation_effect_status"] == "bank_empty"
    assert c["foundation_weighted_top_k"] == 0 and c["maximum_weighted_records"] == 1
    assert c["usage"]["search_calls"] == c["usage"]["query_positions"] == 68
    assert c["usage"]["weight_mass"] == c["usage"]["effective_slots"] == 0
    assert c["usage"]["top1_coverage"] is None
    assert result["phases"]["CC"]["methods"]["real"]["paired_switch_em"] == 1
    assert "off+empty" in c["empty_control"]


@pytest.mark.parametrize("mode", ["random256", "cluster256"])
def test_compression_uses_only_checkpoint_and_does_not_rescale_values(tmp_path, mode):
    m = TinyDictionary()
    with torch.no_grad():
        m.base_keys.copy_(torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]]))
        m.base_values.copy_(torch.tensor([[1.],[3.],[10.],[14.]]))
    before = tensor_digest(m)
    state = torch.random.get_rng_state()
    selected = prepare_base_bank(m, mode, 2)
    again = prepare_base_bank(m, mode, 2)
    assert torch.equal(state, torch.random.get_rng_state())
    assert tensor_digest(m) == before
    assert torch.equal(selected["keys"], again["keys"])
    assert selected["active_records"] == 2 and selected["source_records"] == 4
    for i, members in enumerate(selected["members"]):
        torch.testing.assert_close(selected["values"][i], m.base_values.detach()[members].mean(0), rtol=0, atol=0)
    if mode == "cluster256":
        assert sorted(selected["values"].flatten().tolist()) == [2., 12.]
    backend = FoundationBackend(m)
    result = evaluate_coldstart(backend, m, packet(), tmp_path, max_cases=2, base_mode=mode, merge_count=2)
    assert result["bank_records"] == 2 and result["coldstart"]["usage"]["records"] == 2


@pytest.mark.parametrize("fail, match", [("exception", "intentional scoring"), ("bank", "mutated"), ("teacher_read", "Teacher unexpectedly")])
def test_failure_restores_outer_cpu_bank_flags_and_backend_methods(tmp_path, fail, match):
    m = TinyDictionary().eval()
    backend = FoundationBackend(m, fail=fail)
    original_state = copy.deepcopy(m.state_dict())
    before = tensor_digest(m)
    with m.use_cpu_base(m.base_keys[:1],m.base_values[:1],ids=["existing"]) as previous:
        previous_hash = previous.hash()
        sentinel = {"sentinel": True}
        m.last_base_retrieval = sentinel
        with pytest.raises(RuntimeError, match=match):
            evaluate_coldstart(backend,m,packet(),tmp_path,include_teacher=True,max_cases=2)
        assert m.base_cpu_override is previous and previous.hash() == previous_hash
        assert m.last_base_retrieval is sentinel
        assert not m.training and m.routing_mode == "straight_through"
        assert tensor_digest(m) == before
        assert "generate" not in backend.__dict__ and "score" not in backend.__dict__
    assert set(original_state) == set(m.state_dict())


def test_usage_accumulation_handles_all_leading_axes_and_empty_bank():
    usage = _Usage(3)
    usage.add(dict(indices=torch.tensor([[[0,1],[2,1]]]),weights=torch.tensor([[[.75,.25],[.5,.5]]])))
    result = usage.snapshot()
    assert result["query_positions"] == 2 and result["search_calls"] == 1
    assert result["top1_used"] == 2 and result["topk_used"] == 3
    assert result["weight_mass"] == 2
    assert result["sparse_counts"] == [dict(index=0,top1=1,topk=1,weight_mass=.75),
        dict(index=1,top1=0,topk=2,weight_mass=.75),dict(index=2,top1=1,topk=1,weight_mass=.5)]
    empty = _Usage(0)
    empty.add(dict(indices=torch.empty(2,4,0,dtype=torch.long),weights=torch.empty(2,4,0)))
    assert empty.snapshot()["query_positions"] == 8


def test_invalid_controls_and_overwrite_are_rejected_before_evaluation(tmp_path):
    m = TinyDictionary()
    for mode, count in [("wrong",None),("full",2),("off",1),("random256",0),("cluster256",5),("random256",True)]:
        with pytest.raises(ValueError):
            prepare_base_bank(m,mode,count)
    (tmp_path/"predictions.jsonl").touch()
    with pytest.raises(FileExistsError):
        evaluate_coldstart(FoundationBackend(m),m,packet(),tmp_path,max_cases=2)


def summary_script():
    path = Path(__file__).resolve().parents[1]/"scripts/summarize_coldstart.py"
    spec = importlib.util.spec_from_file_location("_test_coldstart_summary",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_manifest(directory,result,*,stage="eval",**config):
    manifest = dict(protocol="dictionary-coldstart-v1",stage=stage,complete=True,
        backbone_unchanged=True,configuration={"stage":stage,"base_mode":"full","model":"fixed","arm":"test",**config},
        cache_sha256="1"*64,checkpoint_sha256="2"*64,result=result)
    (directory/"manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_summary_recomputes_predictions_usage_and_explicit_linked_teacher(tmp_path):
    script = summary_script()
    parent = tmp_path/"coldstart_fixture"
    teacher,_,_ = run(parent/"teacher",teacher=True)
    save_manifest(parent/"teacher",teacher)
    student,_,_ = run(parent/"student",teacher=False)
    save_manifest(parent/"student",student)
    pending = parent/"pending"
    pending.mkdir()
    (pending/"manifest.json").write_text(json.dumps(dict(protocol="dictionary-coldstart-v1",
        stage="train",configuration={"stage":"train"},complete=False)))
    report = script.summarize(tmp_path)
    assert report["audit_passed"] and report["partial"] and report["complete_jobs"] == 2
    assert report["costs"]["evaluation_generation_calls"] == 80
    assert report["costs"]["evaluation_generation_tokens"] == 320
    for record in report["records"]:
        assert record["phases"]["CC"]["methods"]["real"]["paired_switch_em"] == 1
        assert record["foundation"]["usage"]["query_positions"] == 68
        linked = record["linked_teacher_qualification"]
        assert linked["source_runs"] == [str(parent/"teacher")]
        assert linked["phases"]["CC"]["strict_pair_qualified"] == 2
        assert linked["phases"]["CC"]["methods"]["real"]["paired_switch_em"] == 1
    # Logged EM must not override a changed raw answer.
    rawpath = parent/"student/predictions.jsonl"
    rows = [json.loads(x) for x in rawpath.read_text().splitlines()]
    rows[0]["prediction"] = "definitely wrong"
    rawpath.write_text("\n".join(map(json.dumps,rows))+"\n")
    with pytest.raises(ValueError,match="row EM"):
        script.summarize(tmp_path)


def test_summary_checks_foundation_usage_not_just_sidecar_digest(tmp_path):
    script = summary_script()
    directory = tmp_path/"coldstart_badusage"
    result,_,_ = run(directory,teacher=False)
    manifest = save_manifest(directory,result)
    rawpath = directory/"foundation_usage.jsonl"
    rows = [json.loads(x) for x in rawpath.read_text().splitlines()]
    rows[0]["sparse_counts"][0]["top1"] += 1
    rawpath.write_text("\n".join(map(json.dumps,rows))+"\n")
    manifest["result"]["coldstart"]["foundation_usage_sha256"] = hashlib.sha256(rawpath.read_bytes()).hexdigest()
    (directory/"manifest.json").write_text(json.dumps(manifest))
    (directory/"metrics.json").write_text(json.dumps(manifest["result"]))
    with pytest.raises(ValueError,match="Top1|denominator"):
        script.summarize(tmp_path)


def test_zero_weight_foundation_routes_are_audited_but_never_effective_memory(tmp_path):
    m = TinyDictionary()
    m.set_alpha_base(0.)
    directory = tmp_path/"coldstart_weight_zero"
    result = evaluate_coldstart(FoundationBackend(m),m,packet(),directory,max_cases=2)
    c = result["coldstart"]
    assert c["alpha_base"] == 0 and c["base_mode"] == "full"
    assert c["usage"]["search_calls"] == 68 and c["usage"]["topk_coverage"] > 0
    assert not c["foundation_effective"] and c["foundation_effect_status"] == "weight_zero"
    assert c["foundation_weighted_top_k"] == 0 and c["maximum_weighted_records"] == 1
    assert c["zero_weight_search_cost_included"]
    assert "zero foundation output weight" in c["usage_scope"]
    save_manifest(directory,result,alpha_base=.9)  # Eval CLI defaults do not change loaded checkpoint.
    script = summary_script()
    report = script.summarize(tmp_path)
    table = report["evaluation_table"][0]
    assert table["alpha_base"] == 0 and table["foundation_effect_status"] == "weight_zero"
    assert not table["foundation_effective"] and table["foundation_weighted_top_k"] == 0
    # A plausible coverage record cannot hide a false output-contribution label.
    result["coldstart"]["foundation_effective"] = True
    (directory/"metrics.json").write_text(json.dumps(result))
    save_manifest(directory,result)
    with pytest.raises(ValueError,match="contribution label"):
        script.summarize(tmp_path)


def test_summary_derives_effect_labels_for_early_smoke_artifacts(tmp_path):
    directory = tmp_path/"coldstart_early_smoke"
    result,_,_ = run(directory,teacher=False)
    for key in ("foundation_effective","foundation_effect_status","foundation_weighted_top_k","maximum_weighted_records"):
        del result["coldstart"][key]
    (directory/"metrics.json").write_text(json.dumps(result))
    save_manifest(directory,result,alpha_base=0.)
    record = summary_script().summarize(tmp_path)["evaluation_table"][0]
    assert record["alpha_base"] == .25 and record["foundation_effective"]
    assert record["foundation_effect_status"] == "weighted_branch_enabled"


def test_training_summary_recomputes_exposure_schedule_and_routing(tmp_path):
    script = summary_script()
    directory = tmp_path/"coldstart_training"
    directory.mkdir()
    samples = [dict(targets=[0,1],episode=[1,0],columns=[1,0],qviews=[0,0],sviews=[0,0]),
        dict(targets=[1,2],episode=[2,1,0],columns=[1,0],qviews=[0,0],sviews=[0,0,0])]
    rows = [dict(step=i+1,sample=s,routing="dense" if i==0 else "sparse",elapsed_seconds=i+1,
        base_keys_gradient_slots=2,base_values_gradient_slots=3,
        student_input_tokens=100,teacher_input_tokens=150,rollout_input_tokens=0,
        student_forward_calls=3,teacher_forward_calls=3,rollout_forward_calls=0)
        for i,s in enumerate(samples)]
    schedule = hashlib.sha256()
    for s in samples: schedule.update(json.dumps(s,sort_keys=True).encode())
    status = dict(complete=True,updates=2,schedule_sha256=schedule.hexdigest(),target_unique_count=3,bank_unique_count=3,
        key_gradient_coverage=3,value_gradient_coverage=4,elapsed_seconds=2.1,trainable_parameters=99,
        totals={k:2*v for k,v in rows[0].items() if k.endswith("tokens") or k.endswith("forward_calls")})
    (directory/"training_status.json").write_text(json.dumps(status))
    (directory/"training.jsonl").write_text("\n".join(map(json.dumps,rows))+"\n")
    (directory/"last.pt").write_bytes(b"hashed fixture, deliberately not deserialized")
    (directory/"usage.pt").write_bytes(b"hashed fixture, deliberately not deserialized")
    save_manifest(directory,status,stage="train",updates=2,batch_size=2,routing="dense_warm")
    result = script.summarize(tmp_path)
    assert result["partial"] and not result["formal_scope"]["complete"]
    assert result["costs"]["training_target_exposures"] == 4
    assert result["costs"]["training_input_positions"] == 500
    assert result["costs"]["training_backbone_calls"] == 12
    record = result["records"][0]
    assert record["target_unique_count"] == 3 and record["routing_updates"] == {"dense":1,"sparse":1}
    rows[1]["routing"] = "dense"
    (directory/"training.jsonl").write_text("\n".join(map(json.dumps,rows))+"\n")
    with pytest.raises(ValueError,match="routing schedule"):
        script.summarize(tmp_path)


def test_summary_does_not_discover_historical_interface_jobs_and_empty_is_partial(tmp_path):
    directory = tmp_path/"interface_old"
    directory.mkdir()
    (directory/"manifest.json").write_text("not even valid JSON")
    result = summary_script().summarize(tmp_path)
    assert result["partial"] and result["complete_jobs"] == 0


def test_feature_repair_manifest_is_provenance_not_a_model_run(tmp_path):
    directory=tmp_path/"coldstart_repair/features";directory.mkdir(parents=True)
    (directory/"manifest.json").write_text(json.dumps(dict(protocol="dictionary-coldstart-v1",stage="repair",
        repair_protocol="coldstart-anchor-overlap-repair-v2",complete=True,configuration={"cache":"source.pt"})))
    result=summary_script().summarize(tmp_path)
    assert result["records"]==[] and result["costs"]["training_updates"]==0
    assert result["skipped_nonexperiment_manifests"][0]["classification"]=="auxiliary_preparation_not_model_result"


def test_formal_completion_requires_all_22_conditions_and_no_duplicate_or_smoke():
    script = summary_script()
    records = []
    for arm in script.FORMAL_ARMS:
        for domain in ("wikipedia","synthetic"):
            for mode in (("full","off","random256","cluster256") if arm == "straight_through" else ("full",)):
                records.append(dict(kind="evaluation",split="confirm",evaluated_facts=64,bank_records=128,
                    seed=63042,arm=arm,domain=domain,foundation={"base_mode":mode},run_dir=f"{arm}/{domain}/{mode}"))
    assert len(records) == 22
    assert script.formal_scope(records)["complete"]
    assert not script.formal_scope(records[:-1])["complete"]
    assert not script.formal_scope(records+[records[-1]])["complete"]
    smoke = copy.deepcopy(records)
    smoke[0]["evaluated_facts"] = 16
    assert script.formal_scope(smoke)["completed_conditions"] == 21


def mark_excluded(directory):
    marker = dict(reason="Counterfactual natural passage replaced an overlapping earlier span",
        evidence={"index":13896,"warm_step":940},superseded_by=[directory.name+"_v2"])
    (directory/"analysis_excluded.json").write_text(json.dumps(marker))
    return marker


def test_excluded_old_training_never_enters_records_pending_or_retained_costs(tmp_path):
    script = summary_script()
    old = tmp_path/"coldstart_warm_a_20261006"
    job = old/"wiki_warm";job.mkdir(parents=True)
    # Deliberately lacks valid scientific result/checkpoint artifacts; exclusion
    # must precede strict auditing rather than certify or repair old evidence.
    (job/"manifest.json").write_text(json.dumps(dict(protocol="dictionary-coldstart-v1",stage="train",
        complete=True,configuration={"stage":"train","arm":"wiki_warm"})))
    rows = [dict(step=i+1,sample={"targets":[1,2]},student_input_tokens=100,teacher_input_tokens=200,
        rollout_input_tokens=0,student_forward_calls=3,teacher_forward_calls=3,rollout_forward_calls=0,
        elapsed_seconds=(i+1)*4) for i in range(2)]
    (job/"training.jsonl").write_text("\n".join(map(json.dumps,rows))+"\n")
    failed = old/"not_finished";failed.mkdir()
    (failed/"manifest.json").write_text(json.dumps(dict(protocol="dictionary-coldstart-v1",stage="train",
        complete=False,error="invalid old run",configuration={"stage":"train"})))
    marker = mark_excluded(old)
    fresh = tmp_path/"coldstart_warm_a_v2_20261006/initial";fresh.mkdir(parents=True)
    save_manifest(fresh,{"valid_new_initialization":True},stage="init")
    result = script.summarize(tmp_path)
    assert result["complete_jobs"] == 1 and result["records"][0]["run_dir"] == str(fresh)
    assert result["incomplete_jobs"] == []
    assert result["costs"]["training_updates"] == result["costs"]["training_input_positions"] == 0
    assert len(result["excluded_suites"]) == 1
    excluded = result["excluded_suites"][0]
    assert excluded["reason"] == marker["reason"] and excluded["superseded_by"] == marker["superseded_by"]
    assert len(excluded["jobs"]) == 2 and excluded["status"] == "excluded"
    discarded = result["discarded_compute"]
    assert not discarded["included_in_retained_costs"]
    assert discarded["costs"]["training_updates"] == 2
    assert discarded["costs"]["training_input_positions"] == 600
    assert discarded["costs"]["training_target_exposures"] == 4
    assert discarded["jobs_with_unavailable_costs"] == [str(failed)]
    # Passing a suite itself as runs-root still observes its root sidecar.
    nested = script.summarize(old)
    assert nested["complete_jobs"] == 0 and len(nested["excluded_suites"]) == 1


def test_excluded_teacher_is_never_linked_to_retained_student(tmp_path):
    script = summary_script()
    old = tmp_path/"coldstart_old_teacher"
    teacher,_,_ = run(old/"eval",teacher=True)
    save_manifest(old/"eval",teacher)
    mark_excluded(old)
    fresh = tmp_path/"coldstart_replacement/student"
    student,_,_ = run(fresh,teacher=False)
    save_manifest(fresh,student)
    report = script.summarize(tmp_path)
    assert report["complete_jobs"] == 1 and report["records"][0]["linked_teacher_qualification"] is None
    assert report["costs"]["evaluation_generation_calls"] == 32
    assert report["discarded_compute"]["costs"]["evaluation_generation_calls"] == 48
    assert report["formal_scope"]["completed_conditions"] == 0


def test_exclusion_marker_requires_reviewable_reason_evidence_and_replacement(tmp_path):
    old = tmp_path/"coldstart_invalid_marker";old.mkdir()
    path = old/"analysis_excluded.json"
    for marker in ({},{"reason":"oops"},{"reason":"oops","evidence":"found","superseded_by":[]}):
        path.write_text(json.dumps(marker))
        with pytest.raises(ValueError,match="exclusion"):
            summary_script().summarize(tmp_path)


def test_followup_records_and_costs_are_separate_from_main_and_raw_teacher_failure_is_clear(tmp_path):
    script=summary_script()
    directory=tmp_path/"coldstart_main/eval"
    m=TinyDictionary();backend=FoundationBackend(m);backend.wrapped_teacher=True
    main=evaluate_coldstart(backend,m,packet(),directory,include_teacher=True,max_cases=2)
    save_manifest(directory,main,arm="no_warm",domain="wikipedia",seed=63042)
    follow_dir=tmp_path/"coldstart_teacher_repair/eval"
    follow,_,_=run(follow_dir,teacher=False)
    save_manifest(follow_dir,follow,arm="wiki_teacher_repair",domain="wikipedia",seed=63042)
    result=script.summarize(tmp_path)
    assert len(result["records"])==1 and result["records"][0]["arm"]=="no_warm"
    assert result["complete_jobs"]==1 and result["retained_total_complete_jobs"]==2
    assert result["costs"]["evaluation_generation_calls"]==48
    supplementary=result["supplementary_teacher_followup"]
    assert len(supplementary["records"])==1 and supplementary["costs"]["evaluation_generation_calls"]==32
    assert result["all_retained_costs"]["evaluation_generation_calls"]==80
    assert supplementary["scope"]["requested"] and not supplementary["scope"]["complete"]
    assert not supplementary["scope"]["predeclared_main_arm"]
    for table in (result["evaluation_table"],supplementary["evaluation_table"]):
        cc=table[0]["phases"]["CC"]
        assert cc["teacher_pair_qualified"]==1 and not cc["teacher_reaches_90_percent"]
        assert cc["teacher_condition"]=="raw_observation_context"
        assert "do not isolate" in cc["teacher_interpretation"]
    assert result["formal_scope"]["completed_conditions"]==0


def test_followup_completion_requires_five_eval_and_two_training_conditions():
    script=summary_script()
    records=[dict(kind="training",arm="wiki_teacher_warm",updates=1024,run_dir="warm"),
        dict(kind="training",arm="wiki_teacher_repair",updates=512,run_dir="transfer")]
    for domain,split,cases,bank in (("wikipedia","dev",16,64),("synthetic","dev",16,64),
            ("wikipedia","confirm",64,128),("synthetic","confirm",64,128),("synthetic_train_probe","dev",8,128)):
        records.append(dict(kind="evaluation",arm="wiki_teacher_repair",domain=domain,split=split,
            evaluated_facts=cases,bank_records=bank,seed=63042,foundation={"base_mode":"full"},run_dir=domain+split))
    assert script.followup_scope(records,True)["complete"]
    assert not script.followup_scope(records[:-1],True)["complete"]
    assert not script.followup_scope(records[1:],True)["complete"]
    assert script.followup_scope([],False)["complete"] is None
    assert script.formal_scope(records)["completed_conditions"]==0


def test_formal_training_costs_separate_smoke_and_reject_incomplete_or_duplicate_budget():
    script=summary_script()
    specs=[("matched_warm","matched",1024),("wiki_warm","wikipedia",1024),
        ("wiki_joint_warm","wikipedia",1024)]
    specs += [(arm,"synthetic",512) for arm in ("no_warm","matched_warm_transfer",
        "wiki_warm_transfer","fixed","learned_sparse","dense_warm","straight_through","wiki_joint")]
    records=[dict(kind="training",arm=arm,domain=domain,updates=updates,seed=63042,
        run_dir=arm,costs={"target_exposures":updates*8}) for arm,domain,updates in specs]
    smoke=dict(kind="training",arm="unspecified",domain="wikipedia",updates=4,
        seed=63042,run_dir="smoke",costs={"target_exposures":32})
    records.append(smoke)
    result=script.formal_training_accounting(records)
    assert result["complete"] and result["completed_jobs"]==result["expected_jobs"]==11
    assert result["costs"]["training_updates"]==result["expected_training_updates"]==7168
    assert result["costs"]["training_target_exposures"]==7168*8
    assert result["other_retained_training"]["costs"]["training_updates"]==4
    assert script.aggregate_costs(records)["training_updates"]==7172
    assert not script.formal_training_accounting(records[1:])["complete"]
    duplicate=script.formal_training_accounting(records+[dict(records[0],run_dir="duplicate")])
    assert not duplicate["complete"] and duplicate["duplicates"]
    assert duplicate["costs"]["training_updates"]==8192
    wrong_seed=copy.deepcopy(records);wrong_seed[0]["seed"]=7
    assert not script.formal_training_accounting(wrong_seed)["complete"]
    assert script.formal_training_accounting(wrong_seed)["other_retained_training"]["costs"]["training_updates"]==1028


def test_summary_recomputes_teacher_probe_and_flags_gold_as_extra_supervision(tmp_path):
    from test_coldstart_teacher import FakeBackend as TeacherFake,packet as teacher_packet,probe
    directory=tmp_path/"coldstart_teacher_probe/probe"
    metrics=probe.run_probe(TeacherFake(),teacher_packet(),directory)
    manifest=dict(protocol="coldstart-teacher-feasibility-v1",complete=True,
        selection=metrics["selection"],result=metrics)
    (directory/"manifest.json").write_text(json.dumps(manifest))
    script=summary_script();result=script.summarize(tmp_path)
    assert result["records"]==[] and result["teacher_feasibility_costs"]["generation_calls"]==480
    teacher=result["teacher_feasibility"][0]
    assert teacher["selected_strategy"]=="gold_annotated"
    assert teacher["subsets"]["validation"]["gold_annotated"]["both_correct"]==64
    assert teacher["subsets"]["validation"]["gold_annotated"]["extra_gold_localization"]
    assert teacher["subsets"]["validation"]["quote_instruction"]["paired_em"]==0
    assert teacher["subsets"]["validation"]["quote_instruction"]["paired_containment"]==1
    assert result["costs"]["evaluation_generation_calls"]==0
    # Extra-gold conditions cannot be relabeled as source-only success.
    path=directory/"predictions.jsonl";raw=[json.loads(x) for x in path.read_text().splitlines()]
    next(row for row in raw if row["strategy"]=="gold_annotated")["extra_gold_localization"]=False
    path.write_text("\n".join(map(json.dumps,raw))+"\n")
    with pytest.raises(ValueError,match="Gold localization"):
        script.summarize(tmp_path)
