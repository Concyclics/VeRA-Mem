"""Train-only teacher feasibility without GPU/model downloads."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vera_mem.coldstart_teacher import (STRATEGIES,QUOTE_INSTRUCTION,make_teacher_context,
    make_teacher_question,question_exposes_answer,strategy_metadata)

spec=importlib.util.spec_from_file_location("teacher_probe",Path(__file__).resolve().parents[1]/"scripts/probe_coldstart_teacher.py")
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def packet():
    rows=[]
    for i in range(96):
        anchor=f"the literal five word anchor {i}"
        a,b=f"apple amber number{i}",f"river cedar number{i}"
        rows.append(dict(id=f"r{i}",entity=f"note{i}",split="train",a=a,b=b,
            questions=[f"In note{i}, what three words follow '{anchor}'? Return only the three words."],
            supports=[[f"Note note{i}.\nSome introduction {anchor} {answer} followed by the final words"] for answer in (a,b)],
            provenance={"anchor":anchor}))
    return dict(protocol=probe.CACHE_PROTOCOL,model_revision=probe.REVISION,split="train",task="wikipedia",rows=rows)


def test_context_only_strategies_do_not_look_at_gold_and_questions_never_change():
    r=packet()["rows"][0];context=r["supports"][0][0];question=r["questions"][0]
    for strategy in STRATEGIES:
        assert make_teacher_question(r,question,strategy)==question
        assert make_teacher_context(r,context,r["a"],strategy).startswith(context)
        assert strategy_metadata(strategy)["extra_gold_localization"]==(strategy=="gold_annotated")
    for strategy in ("baseline","quote_instruction"):
        assert make_teacher_context({},context,None,strategy)==make_teacher_context({"a":"wrong"},context,"wrong label",strategy)
    assert make_teacher_context(r,context,r["a"],"baseline")==context
    assert make_teacher_context(r,context,r["a"],"quote_instruction")==context+"\n\n"+QUOTE_INSTRUCTION
    gold=make_teacher_context(r,context,r["a"],"gold_annotated")
    assert gold.endswith("<requested_span>"+r["a"]+"</requested_span>")
    with pytest.raises(ValueError):make_teacher_context(r,context,r["b"],"gold_annotated")
    with pytest.raises(ValueError):make_teacher_context(r,context,r["a"],"unknown")
    amp="Note. A & B"
    assert make_teacher_context({},amp,"A & B","gold_annotated").endswith("<requested_span>A & B</requested_span>")


def test_fixed_training_subsets_are_disjoint_and_reject_other_splits_or_broken_anchor():
    p=packet();original=copy.deepcopy(p);selected=probe.select_rows(p)
    assert p==original and selected==probe.select_rows(p)
    assert selected["selected_indices"]["calibration"]==list(range(16))
    assert len(selected["selected_indices"]["validation"])==64
    assert not set(selected["selected_indices"]["validation"])&set(range(16))
    for change in (lambda p:p.update(split="dev"),lambda p:p.update(task="synthetic"),
                   lambda p:p["rows"][-1]["supports"][1].__setitem__(0,"broken anchor")):
        invalid=copy.deepcopy(p);change(invalid)
        with pytest.raises(ValueError):probe.select_rows(invalid)


def test_token_bounded_leakage_filter_has_provenance_and_deterministic_eligible_sampling():
    p=packet();r=p["rows"][0]
    r["a"]="David Croft and";r["provenance"]["anchor"]="David Croft, and written by"
    r["questions"]=["What follows 'David Croft, and written by'? Return three words."]
    r["supports"]=[["Note. David Croft, and written by "+answer+" next"] for answer in (r["a"],r["b"])]
    selected=probe.select_rows(p)
    assert selected["excluded_count"]==1 and selected["eligible_records"]==95
    assert selected["excluded_records"][0]["index"]==0
    assert selected["excluded_records"][0]["reasons"][0]["world"]=="A"
    assert selected["selected_indices"]["calibration"]==list(range(1,17))
    assert len(selected["selected_indices"]["validation"])==64
    assert 0 not in selected["selected_indices"]["validation"]
    assert not set(selected["selected_indices"]["validation"])&set(range(1,17))
    assert selected==probe.select_rows(p)
    assert selected["teacher_context_validation"]["checked_contexts"]==192
    assert selected["teacher_context_validation"]["passed"]
    assert question_exposes_answer("David Croft, and written by","David Croft and")
    assert not question_exposes_answer("red blue hello","red blue he")
    assert not question_exposes_answer("a red blue anchor","the red blue")


def test_unselected_full_training_annotation_failure_is_exported_before_inference():
    p=packet();r=p["rows"][-1]
    r["a"]="literal <tag> value"
    r["supports"][0]=["Note. "+r["provenance"]["anchor"]+" "+r["a"]+" end"]
    selected=probe.select_rows(p)
    validation=selected["teacher_context_validation"]
    assert validation["checked_contexts"]==192 and validation["failed_contexts"]==1
    assert validation["failures"][0]["index"]==95 and not validation["passed"]


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(1.),requires_grad=False);self.eval()
    def forward(self,input_ids):return input_ids*self.weight


class FakeBackend:
    def __init__(self):
        self.model=FakeModel();self.mode="none";self.calls=[]
    def prompt_ids(self,question,context=None):return [1]*(len(question.split())+len(context.split())+3)
    def generate(self,question,context,max_new_tokens):
        assert max_new_tokens==32 and not torch.is_grad_enabled()
        self.calls.append((question,context))
        self.model(input_ids=torch.tensor([self.prompt_ids(question,context)]))
        for _ in range(3):self.model(input_ids=torch.ones(1,1,dtype=torch.long))
        if "<requested_span>" in context:
            prediction=context.split("<requested_span>")[1].split("</requested_span>")[0]
        elif QUOTE_INSTRUCTION in context:
            anchor=question.split("'")[1]
            answer=" ".join(context.split(anchor,1)[1].split()[:3])
            prediction="The span is "+answer
        else:prediction="unknown invented phrase"
        return prediction,4,.01
    def score(self,question,answer,context):
        self.model(input_ids=torch.tensor([self.prompt_ids(question,context)+[1]*len(answer.split())]))
        return SimpleNamespace(nll_sum=3.,tokens=3)


def test_480_teacher_only_predictions_full_costs_and_explicit_gold_qualification(tmp_path):
    backend=FakeBackend();p=packet()
    result=probe.run_probe(backend,p,tmp_path)
    assert result["complete"] and result["backbone_unchanged"] and result["gradient_steps"]==0
    assert result["backbone_hash_before"]==result["backbone_hash_after"]
    assert result["costs"]["generation_calls"]==result["costs"]["scoring_calls"]==480
    assert result["costs"]["generation_forward_calls"]==1920
    assert result["costs"]["scoring_forward_calls"]==480
    rows=[json.loads(x) for x in (tmp_path/"predictions.jsonl").read_text().splitlines()]
    assert len(rows)==480
    assert result["costs"]["generation_input_positions"]==sum(r["prompt_tokens"]+3 for r in rows)
    for subset,count in (("calibration",16),("validation",64)):
        summaries=result["subsets"][subset]
        assert summaries["baseline"]["paired_em"]==0
        assert summaries["quote_instruction"]["paired_em"]==0
        assert summaries["quote_instruction"]["paired_containment"]==1
        assert summaries["gold_annotated"]["both_correct"]==count
        assert summaries["gold_annotated"]["extra_gold_localization"]
    assert all(r["question"]==p["rows"][r["source_index"]]["questions"][0] for r in rows)
    assert not backend.model._forward_pre_hooks
    with pytest.raises(FileExistsError):probe.run_probe(backend,p,tmp_path)


def test_probe_refuses_attached_memory_and_removes_hook_after_failure(tmp_path):
    backend=FakeBackend();backend.vector_store=object()
    with pytest.raises(RuntimeError,match="must not attach memory"):
        probe.run_probe(backend,packet(),tmp_path/"badmemory")
    backend=FakeBackend()
    def broken(*args,**kwargs):raise RuntimeError("generation broke")
    backend.generate=broken
    with pytest.raises(RuntimeError,match="generation broke"):
        probe.run_probe(backend,packet(),tmp_path/"failed")
    assert not backend.model._forward_pre_hooks
