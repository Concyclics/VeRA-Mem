"""Independent v2 repair and exclusion tests; synthetic CPU data only."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
torch=pytest.importorskip("torch")

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("coldstart_audit_v2_under_test",SCRIPTS/"audit_coldstart.py")
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

TARGET=4
STATISTICS=3


@pytest.fixture
def repair_pair():
    """The actual failure structure, with a small untouched statistics prefix."""
    rows=[]
    for i in range(6):
        anchor=f"Unique anchor for note {i}"
        a,b="alpha beta gamma","delta epsilon zeta"
        support=f"Note E{i}.\n{anchor} {a} tail"
        rows.append(dict(id=f"record-{i}",entity=f"E{i}",relation="three-word continuation",split="train",
            a=a,b=b,a_tokens=[1,2,3],b_tokens=[4,5,6],questions=[f"What follows {anchor}?"],
            supports=[[support],[support.replace(a,b,1)]],provenance=dict(anchor=anchor,page_id=str(i))))
    anchor="References Sources 1916 Ship launches"
    a,b="Ship launches Ship","Marble silver canyon"
    source="Note E4.\nPreface "+anchor+" Ship launches Ship launches"
    rows[TARGET].update(a=a,b=b,questions=[f"What follows {anchor}?"],
        supports=[[source],[source.replace(a,b,1)]],provenance=dict(anchor=anchor,page_id=str(TARGET)))
    parent=dict(protocol="dictionary-coldstart-v1",task="wikipedia",split="train",model_revision="model-revision",
        qids=["canonical"],sids=["canonical"],writer_prefix="Remember this information: ",pooling="content-only",
        q=torch.arange(18,dtype=torch.float16).reshape(6,1,3),
        last=torch.arange(36,dtype=torch.float16).reshape(6,2,1,3),
        pool=torch.arange(36,dtype=torch.float16).reshape(6,2,1,3)+.25,
        rows=rows,token_repairs=[dict(id="record-1",old_b="earlier words here",new_b=rows[1]["b"],donor_id="donor-id")],
        data_manifest_sha256="1"*64,data_jsonl_sha256="2"*64)
    repaired=deepcopy(parent)
    repaired["rows"][TARGET]["supports"][1]=["Note E4.\nPreface "+anchor+" "+b+" launches"]
    repaired["last"][TARGET,1,0,1]+=.5
    repaired["pool"][TARGET,1,0,2]-=.75
    repaired["feature_repair"]=dict(record_id=rows[TARGET]["id"],target_index=TARGET,kind="addressed-anchor-span")
    repaired["data_manifest_sha256"]="3"*64
    repaired["data_jsonl_sha256"]="4"*64
    return parent,repaired


def verify(pair,**kwargs):
    parent,repaired=pair
    return audit.verify_repair_packets(parent,repaired,kwargs.pop("target_id",f"record-{TARGET}"),
        kwargs.pop("target_index",TARGET),statistics_count=kwargs.pop("statistics_count",STATISTICS),**kwargs)


def test_repair_proves_only_addressed_b_change_and_preserves_statistics(repair_pair):
    parent,repaired=repair_pair
    before=deepcopy(parent),deepcopy(repaired)
    proof=verify(repair_pair)
    assert proof["status"]=="passed"
    for field in ("query_bitwise_unchanged","all_a_features_bitwise_unchanged",
                  "all_other_features_bitwise_unchanged","common_statistics_inputs_bitwise_unchanged"):
        assert proof[field] is True
    assert proof["changed_record_ids"]==[f"record-{TARGET}"] and proof["target_index"]==TARGET
    # The verifier must not repair inputs silently or mutate source tensors.
    for actual,expected in zip(repair_pair,before):
        for field in ("q","last","pool"):
            assert torch.equal(actual[field].view(torch.uint8),expected[field].view(torch.uint8))
        assert actual["rows"]==expected["rows"] and actual["token_repairs"]==expected["token_repairs"]


@pytest.mark.parametrize("feature,index",[
    ("q",(TARGET,0,1)),("q",(0,0,1)),
    ("last",(TARGET,0,0,1)),("pool",(TARGET,0,0,1)),
    ("last",(TARGET-1,1,0,1)),("pool",(TARGET+1,1,0,1)),
    ("last",(0,0,0,1)),("pool",(0,1,0,1)),
])
def test_repair_rejects_any_unrelated_feature_element_change(repair_pair,feature,index):
    repair_pair[1][feature][index]+=.5
    with pytest.raises(ValueError):verify(repair_pair)


def test_bitwise_query_contract_catches_signed_zero(repair_pair):
    assert repair_pair[0]["q"][0,0,0]==0
    repair_pair[1]["q"][0,0,0]=-0.
    assert torch.equal(repair_pair[0]["q"],repair_pair[1]["q"])
    assert not torch.equal(repair_pair[0]["q"].view(torch.uint8),repair_pair[1]["q"].view(torch.uint8))
    with pytest.raises(ValueError):verify(repair_pair)


@pytest.mark.parametrize("feature",["q","last","pool"])
def test_repair_rejects_changed_feature_shape_or_dtype(repair_pair,feature):
    pair=deepcopy(repair_pair)
    pair[1][feature]=pair[1][feature].float()
    with pytest.raises(ValueError):verify(pair)
    pair=deepcopy(repair_pair)
    pair[1][feature]=pair[1][feature][:-1]
    with pytest.raises(ValueError):verify(pair)


@pytest.mark.parametrize("feature",["last","pool"])
@pytest.mark.parametrize("value",[float("nan"),float("inf")])
def test_target_b_allowed_slot_must_still_be_finite(repair_pair,feature,value):
    repair_pair[1][feature][TARGET,1,0,0]=value
    with pytest.raises(ValueError,match="nonfinite"):
        verify(repair_pair)


@pytest.mark.parametrize("mutate",[
    lambda p:p["rows"][TARGET].update(a="different answer words"),
    lambda p:p["rows"][TARGET].update(b="different donor words"),
    lambda p:p["rows"][TARGET]["a_tokens"].append(8),
    lambda p:p["rows"][TARGET]["b_tokens"].append(8),
    lambda p:p["rows"][TARGET]["questions"].append("extra query"),
    lambda p:p["rows"][TARGET]["supports"][0].append("extra A support"),
    lambda p:p["rows"][TARGET]["provenance"].update(anchor="another anchor"),
    lambda p:p["rows"][TARGET-1]["provenance"].update(extra="changed other row"),
    lambda p:p["rows"].reverse(),
    lambda p:p["token_repairs"][0].update(donor_id="different donor"),
    lambda p:p.update(writer_prefix="Remember differently: "),
    lambda p:p.update(model_revision="different revision"),
    lambda p:p.update(split="confirm"),
    lambda p:p.update(extra_tensor=torch.ones(1)),
    lambda p:p.update(undeclared_metadata="added"),
])
def test_repair_rejects_changes_outside_target_b_support_and_declared_lineage(repair_pair,mutate):
    mutate(repair_pair[1])
    with pytest.raises(ValueError):verify(repair_pair)


def test_repair_checks_additional_existing_tensor_metadata(repair_pair):
    parent,repaired=repair_pair
    parent["additional_feature"]=torch.tensor([0.,1.],dtype=torch.float16)
    repaired["additional_feature"]=parent["additional_feature"].clone()
    repaired["additional_feature"][1]=2.
    with pytest.raises(ValueError):verify(repair_pair)


@pytest.mark.parametrize("kind",["old_bad_b","append_text","replace_all","remove_tail"])
def test_repaired_b_must_be_exact_replacement_after_anchor(repair_pair,kind):
    parent,repaired=repair_pair
    row=repaired["rows"][TARGET]
    if kind=="old_bad_b":row["supports"][1]=deepcopy(parent["rows"][TARGET]["supports"][1])
    elif kind=="append_text":row["supports"][1][0]+=" additional text"
    elif kind=="replace_all":row["supports"][1]=[row["supports"][0][0].replace(row["a"],row["b"])]
    elif kind=="remove_tail":row["supports"][1][0]=row["supports"][1][0].removesuffix(" launches")
    with pytest.raises(ValueError):verify(repair_pair)


def test_target_address_and_statistics_slice_are_enforced(repair_pair):
    for kwargs in (dict(target_id="wrong-id"),dict(target_index=TARGET-1),dict(target_index=-1),
                   dict(target_index=6),dict(statistics_count=TARGET+1)):
        with pytest.raises(ValueError):verify(repair_pair,**kwargs)


def marker(directory):
    metadata=dict(reason="Overlapping substring broke the anchor",evidence=["record-4","warm-step-940"],
                  superseded_by="coldstart_warm_a_v2_20261006")
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"analysis_excluded.json").write_text(json.dumps(metadata))
    return metadata


def test_exclusion_lookup_preserves_ancestor_provenance(tmp_path):
    ancestor=tmp_path/"archived"
    expected=marker(ancestor)
    child=ancestor/"coldstart_old"/"job"
    child.mkdir(parents=True)
    found=audit.analysis_exclusions(child,tmp_path)
    assert len(found)==1 and found[0]["metadata"]==expected
    assert found[0]["sha256"]==audit.file_hash(ancestor/"analysis_excluded.json")
    assert audit.analysis_exclusions(tmp_path/"unexcluded",tmp_path)==[]


def test_excluded_suite_bad_json_never_read_and_never_counts_pending(tmp_path):
    suite=tmp_path/"coldstart_warm_a_20261006"
    expected=marker(suite)
    (suite/"suite.json").write_text("deliberately invalid excluded suite JSON")
    (suite/"plan.json").write_text("must not read excluded plan")
    child=suite/"wiki_warm"
    child.mkdir()
    (child/"manifest.json").write_text("must not read excluded child")
    result=audit.run_audit(tmp_path)
    assert result["checks_passed"] and not result["complete"]
    assert result["trainings"]==result["evaluations"]==result["pending_jobs"]==[]
    assert len(result["excluded_suites"])==1
    assert expected["reason"] in json.dumps(result["excluded_suites"])
    assert len(result["missing_training_arms"])==11 and len(result["missing_evaluation_cells"])==52
    with pytest.raises(ValueError,match="coverage incomplete"):
        audit.run_audit(tmp_path,require_complete=True)


def test_excluded_completed_child_bad_manifest_never_read(tmp_path):
    suite=tmp_path/"coldstart_mixed_suite"
    child=suite/"superseded_job"
    marker(child)
    (child/"manifest.json").write_text("excluded child invalid JSON")
    plan=[dict(name="superseded_job"),dict(name="still_running")]
    (suite/"plan.json").write_text(json.dumps(plan))
    (suite/"suite.json").write_text(json.dumps(dict(protocol="coldstart-experiment-v1",complete=False,
        source_files_sha256={},jobs=[dict(name="superseded_job",status="complete",exit_code=0),
                                    dict(name="still_running",status="running",exit_code=None)])))
    result=audit.run_audit(tmp_path)
    assert result["checks_passed"] and not result["complete"]
    assert result["trainings"]==result["evaluations"]==[]
    assert result["pending_jobs"]==[dict(suite=suite.name,job="still_running")]
    assert "superseded_job" in json.dumps(result["excluded_suites"])


def test_excluded_unstarted_planned_job_does_not_become_pending(tmp_path):
    suite=tmp_path/"coldstart_planned_suite"
    marker(suite/"superseded_job")
    (suite/"plan.json").write_text(json.dumps([dict(name="superseded_job")]))
    (suite/"suite.json").write_text(json.dumps(dict(protocol="coldstart-experiment-v1",complete=True,
        source_files_sha256={},jobs=[])))
    result=audit.run_audit(tmp_path)
    assert result["pending_jobs"]==[] and len(result["excluded_suites"])==1
    assert not result["complete"] and not result["trainings"]


def test_repair_absent_or_incomplete_manifest_never_loads_partial_artifacts(tmp_path):
    class ForbiddenInputs:
        root=tmp_path
        def load(self,*args,**kwargs):raise AssertionError("Partial repair cache was read")
        def resolve(self,*args,**kwargs):raise AssertionError("Partial repair dependency was resolved")
        def sha(self,*args,**kwargs):raise AssertionError("Partial repair artifact was opened")
    inputs=ForbiddenInputs()
    assert audit.audit_repair(inputs)["status"]=="pending_repair_manifest"
    directory=tmp_path/audit.REPAIR_DIR
    directory.mkdir(parents=True)
    (directory/"manifest.json").write_text(json.dumps(dict(complete=False,configuration={"cache":"forbidden"})))
    (directory/"train.pt").write_text("partial invalid tensor data")
    (directory/"preservation.json").write_text("partial invalid JSON")
    assert audit.audit_repair(inputs)["status"]=="pending_repair_complete"


@pytest.mark.parametrize("repair_status",["pending_repair_manifest","passed"])
@pytest.mark.parametrize("normalized_overlap",[False,True])
def test_preparation_token_repairs_do_not_shadow_global_feature_repair_status(tmp_path,monkeypatch,repair_status,normalized_overlap):
    """Exercise all nine preparation cells, including actual token-repair loops.

    Token repair records deliberately have no ``status`` key. Rebinding the
    outer feature-repair object to one of them must fail this integration test.
    """
    root=tmp_path/"runs"
    raw_v1,raw_v2=tmp_path/"raw_v1",tmp_path/"raw_v2"
    json_files={raw_v1/"manifest.json":dict(complete=True),raw_v2/"manifest.json":dict(complete=True)}
    jsonl_files={}
    packets={}
    loaded=[]
    read_sources=[]
    answers={
        "train":["Amber blue cedar","Silver gold delta"],
        "dev":["Birch copper elm","Marble linen grove"],
        "confirm":["Juniper quartz ridge","Saffron moss vale"],
    }
    sha=lambda path:hashlib.sha256(str(Path(path)).encode()).hexdigest()
    repair=dict(status=repair_status,sentinel="feature-repair-status-object")
    if repair_status=="passed":
        repair.update(data_manifest=str(raw_v2/"manifest.json"),
                      known_excluded_issue=dict(id="previous-bad-row",active_dataset_error=False))

    for domain in ("wikipedia","matched","synthetic"):
        directory=root/f"coldstart_prepare_{domain}_20261006"/"features"
        manifest=dict(complete=True,stage="prepare",configuration=dict(domain=domain,data_dir=str(raw_v1)),result={})
        for split in ("train","dev","confirm"):
            views=1 if split=="train" else 2
            family="wiki" if domain in ("wikipedia","matched") else "synthetic"
            raw=[]
            for i in range(2):
                identity=f"{family}-{split}-record-{i}"
                entity=f"{family}-{split}-entity-{i}"
                anchor=f"Address for {family} {split} note {i}"
                a,b=answers[split][i],answers[split][1-i]
                # The source B label of row 0 requires a tokenizer repair in
                # every training domain. Its replacement donor really exists.
                if split=="train" and i==0:b="Earlier collision words"
                passage=anchor+" "+a
                if domain!="matched":passage="Additional observation text. "+passage+" preserved tail"
                support=[f"Note {entity}.\n{passage}",f"BEGIN NOTE {entity}\n{passage}\nEND NOTE"]
                raw.append(dict(id=identity,entity=entity,relation="three-word continuation",split=split,a=a,b=b,
                    questions=[f"Which words follow {anchor}?",f"Look up {anchor} in {entity}."],
                    supports=[support,[s.replace(a,b,1) for s in support]],
                    provenance=dict(anchor=anchor,page_id=f"{split}-page-{i}",cluster_id=f"{split}-cluster-{i}",
                                    b_donor_page_id=f"{split}-page-{1-i}",b_donor_cluster_id=f"{split}-cluster-{1-i}")))
                if normalized_overlap and domain in ('wikipedia','matched') and split=='train' and i==0:
                    raw[-1]['questions'][0]+=' Reference words: Amber, blue cedar.'
            rows=deepcopy(raw)
            repairs=[]
            for i,row in enumerate(rows):
                if split=="train" and i==0:
                    token_repair=dict(id=row["id"],old_b=row["b"],new_b=rows[1]["a"],donor_id=rows[1]["id"])
                    repairs.append(token_repair)
                    row["b"]=token_repair["new_b"]
                    row["provenance"]["token_repair"]=deepcopy(token_repair)
                    row["supports"][1]=[s.replace(row["a"],row["b"],1) for s in row["supports"][0]]
                row["a_tokens"]=[11+i,20,30]
                row["b_tokens"]=[11+(1-i),20,30]
                row["questions"]=row["questions"][:views]
                row["supports"]=[world[:views] for world in row["supports"]]
            use_repaired=domain=="wikipedia" and split=="train" and repair_status=="passed"
            active_directory=root/audit.REPAIR_DIR if use_repaired else directory
            raw_directory=raw_v2 if use_repaired else raw_v1
            source=raw_directory/domain/(split+".jsonl")
            jsonl_files[source]=raw
            packets[active_directory/(split+".pt")]=dict(task=domain,split=split,rows=rows,token_repairs=repairs,
                qids=["canonical","heldout"][:views],sids=["canonical","heldout"][:views],
                q=torch.arange(2*views*3,dtype=torch.float16).reshape(2,views,3),
                last=torch.ones(2,2,views,3,dtype=torch.float16),pool=torch.ones(2,2,views,3,dtype=torch.float16)*2,
                data_manifest_sha256=sha(raw_directory/"manifest.json"),data_jsonl_sha256=sha(source))
            json_files[active_directory/(split+"_repairs.json")]=repairs
            manifest["result"][split]=dict(records=2,token_repairs=len(repairs))
        json_files[directory/"manifest.json"]=manifest

    class FakeInputs:
        def __init__(self):self.root=root
        def resolve(self,path):
            path=Path(path)
            if path not in json_files and path not in jsonl_files and path not in packets:
                raise FileNotFoundError(path)
            return path
        def sha(self,path):return sha(path)
        def load(self,path,expected=None):
            path=self.resolve(path)
            if expected is not None:assert expected==sha(path)
            loaded.append(path)
            return packets[path]
    def fake_read_jsonl(path):
        read_sources.append(Path(path))
        return jsonl_files[Path(path)]
    calls=[]
    def fake_audit_repair(inputs):
        calls.append(inputs)
        return repair
    monkeypatch.setattr(audit,"read_json",lambda path:json_files[Path(path)])
    monkeypatch.setattr(audit,"read_jsonl",fake_read_jsonl)
    monkeypatch.setattr(audit,"audit_repair",fake_audit_repair)
    inputs=FakeInputs()
    result=audit.audit_preparation(root,inputs=inputs)
    assert calls==[inputs]
    assert result["checks_passed"] and result["semantic_errors"]==[]
    assert result["feature_repair"] is repair and result["feature_repair"]["status"]==repair_status
    assert result["complete"]==(repair_status=="passed")
    assert len(result["prepared"])==len(result["separation"])==len(loaded)==len(read_sources)==9
    assert sum(row["token_repairs"] for row in result["prepared"])==3
    assert len(result["wiki_matched_pairing"])==3
    assert all(row["query_features_bitwise_equal"] and row["gold_query_identity_exact"] for row in result["wiki_matched_pairing"])
    assert result["train_memorization_probe"] is None
    assert result["known_excluded_issues"]==([repair["known_excluded_issue"]] if repair_status=="passed" else [])
    overlaps=result['normalized_question_answer_overlap']['records']
    assert len(overlaps)==(2 if normalized_overlap else 0)
    assert all(r['split']=='train' and r['world']=='a' and r['training_target_steps']==list(range(1,1025)) for r in overlaps)
