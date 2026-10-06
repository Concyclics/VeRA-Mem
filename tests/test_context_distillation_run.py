"""Runner protocol tests using the actual causal tiny backend, without a GPU."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from vera_mem import augmentation_data as data
from vera_mem import context_distillation_run as run
from vera_mem.vector_store import PersistentVectorDB
from test_context_distillation import make_backend


def _inputs():
    generator = torch.Generator().manual_seed(61)
    return torch.randn(3,8,5,generator=generator), torch.randn(3,4,5,generator=generator)


def _config(method):
    return dict(method=method,max_new_tokens=4,sampling_temperature=1.,seed=42,
                updates=4,batch_size=2,eval_size=4,skip_teacher_eval=True)


def test_offline_selection_never_accesses_confirmation_or_control():
    class Guard(dict):
        def __getitem__(self,key):
            assert key in ("train","dev"), "Attempted confirmation/control access"
            return super().__getitem__(key)
    packet = dict(protocol="generalization-v1-layer20",model_revision=run.REVISION,
                  data_fingerprint=data.protocol_fingerprint())
    packet.update({name:Guard(train={"allowed":1},dev={"allowed":2})
                   for name in ("features","examples","template_ids")})
    selected = run.select_offline_packet(packet)
    for name in ("features","examples","template_ids"):
        assert tuple(selected[name]) == ("train","dev")
    packet["data_fingerprint"]="bad"
    with pytest.raises(ValueError,match="fingerprint"):
        run.select_offline_packet(packet)


def test_per_sequence_reduction_equalizes_short_and_long_trajectories():
    losses=torch.tensor([1.,3.,10.],requires_grad=True)
    result=run.sequence_mean(losses,torch.tensor([0,0,1]),torch.tensor([2,1]))
    assert float(result.detach())==6.
    result.backward()
    torch.testing.assert_close(losses.grad,torch.tensor([.25,.25,.5]))
    with pytest.raises(ValueError,match="account"):
        run.sequence_mean(losses,torch.tensor([0,0,1]),torch.tensor([1,1]))


def test_episode_schedule_matches_across_methods_and_has_distinct_negative_facts():
    samplers=[run.EpisodeSampler(128,8,42) for _ in run.METHODS]
    seen=[]
    for _ in range(16):
        batches=[sampler.next() for sampler in samplers]
        assert all(batch==batches[0] for batch in batches)
        batch=batches[0]
        assert len(batch["episode"])==len(set(batch["episode"]))==72
        assert [batch["episode"][i] for i in batch["mapping"]]==batch["targets"]
        seen.extend(batch["targets"])
    assert len(seen)==len(set(seen))==128


@pytest.mark.parametrize("method",run.METHODS)
def test_step_exact_prefix_teacher_first_and_differentiable_bank_after_rollout(method,monkeypatch):
    backend=make_backend()
    module=backend.vector_vera
    queries,support=_inputs()
    events=[]
    original=backend.forward_sequences
    def tracked(questions,continuations,contexts=None,teacher=False):
        events.append(dict(teacher=teacher,contexts=contexts,tokens=[list(t) for t in continuations],
                           bank_grad=backend.vector_values.requires_grad))
        return original(questions,continuations,contexts=contexts,teacher=teacher)
    monkeypatch.setattr(backend,"forward_sequences",tracked)
    sampled=[[7],[9,11,1]]
    def sample(questions,max_new_tokens,temperature,generator):
        assert not backend.vector_keys.requires_grad
        assert not backend.vector_values.requires_grad
        assert backend.vector_override is None and backend.oracle_values is None
        events.append(dict(sample=True))
        return sampled
    monkeypatch.setattr(backend,"sample_student",sample)
    loss,metrics,tokens,golden=run.training_step(
        backend,module,queries,dict(chosen=support[:,2],all=support),["question1","question2"],
        ["observed A","observed B"],["a","bb"],[0,2],method,_config(method),None,
        replay_questions=["canonical1","canonical2"])
    if method.startswith("on_"):
        assert events[0]==dict(sample=True)
        assert tokens==sampled
        # The first sampled trajectory has no EOS; the runner must not add it.
        assert tokens[0]==[7]
        assert metrics["sampled_tokens"]==4
    else:
        assert tokens==golden
    forwards=[event for event in events if "teacher" in event]
    if method=="ce":
        assert len(forwards)==2 and not any(event["teacher"] for event in forwards)
        assert metrics["reverse_kl"] is None
    else:
        assert len(forwards)==3 and forwards[0]["teacher"]
        assert forwards[0]["contexts"]==["observed A","observed B"]
        assert forwards[0]["tokens"]==forwards[1]["tokens"]==tokens
        assert metrics["reverse_kl"]>=0.
    assert all(event["contexts"] is None for event in forwards if not event["teacher"])
    assert all(event["bank_grad"] for event in forwards)
    assert metrics["replay_target_tokens"]==sum(map(len,golden))
    assert forwards[-1]["tokens"]==golden
    loss.backward()
    for parameter in (module.b,module.Wv.weight,module.Wq.weight,module.Wk.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm()>0
    assert all(parameter.grad is None for parameter in backend.model.parameters())


def test_training_loop_final_checkpoint_and_replay_cadence(tmp_path):
    backend=make_backend()
    module=backend.vector_vera
    examples=data.datasets(train_size=16)["train"]
    generator=torch.Generator().manual_seed(103)
    packet=dict(examples=dict(train=[asdict(ex) for ex in examples]),
                features=dict(train=dict(q=torch.randn(16,8,5,generator=generator),
                                         s=torch.randn(16,4,5,generator=generator))),
                template_ids=dict(train=dict(q=[t.id for t in data.get_templates("train","query")],
                                              s=[t.id for t in data.get_templates("train","support")])))
    result=run.train(backend,module,_config("ce"),packet,tmp_path)
    assert result["finished"] and result["selected_step"]==4
    assert result["target_exposures"]==8 and result["canonical_replay_exposures"]==2
    rows=[json.loads(line) for line in (tmp_path/"training.jsonl").read_text().splitlines()]
    assert [row["canonical_replay_ce"] is not None for row in rows]==[False,False,False,True]
    rollouts=[json.loads(line) for line in (tmp_path/"rollouts.jsonl").read_text().splitlines()]
    assert len(rollouts)==8
    assert all(row["continuation_token_ids"][-1]==backend.tokenizer.eos_token_id for row in rollouts)
    checkpoint=torch.load(tmp_path/"last.pt",weights_only=True)
    assert checkpoint["step"]==4
    assert backend.vector_keys is None and backend.vector_values is None and backend.mode=="none"
    assert all(not parameter.requires_grad for parameter in module.parameters())


def test_diagnostic_reports_base_and_first_token_metrics_without_updating_memory(tmp_path):
    backend=make_backend()
    module=backend.vector_vera
    examples=data.datasets(train_size=16)["dev"][:2]
    contexts=[data.render_support(ex,"train_support_00") for ex in examples]
    db=PersistentVectorDB(module.key_dim,module.rank)
    generator=torch.Generator().manual_seed(4)
    with torch.no_grad():
        for index,example in enumerate(examples):
            x=torch.randn(5,generator=generator)
            db.write(example.id,module.encode_key(x),module.encode_value(x),index)
    before=run.tensor_digest(module),db.hash()
    metrics=run.diagnostic_alignment(backend,module,db,examples,contexts,2,"dev",tmp_path)
    assert metrics["count"]==2
    assert all(name in metrics for name in ("reverse_kl","hidden_cosine","no_memory_reverse_kl",
                                            "first_token_reverse_kl","first_token_no_memory_hidden_cosine"))
    assert before==(run.tensor_digest(module),db.hash())
    rows=[json.loads(line) for line in (tmp_path/"alignment_predictions.jsonl").read_text().splitlines()]
    assert metrics["tokens"]==sum(row["tokens"] for row in rows)


def test_generation_budget_wrapper_restores_backend_on_error(monkeypatch,tmp_path):
    backend=make_backend()
    calls=[]
    def original(question,context=None,max_new_tokens=None):
        calls.append(max_new_tokens)
        return "",0,0.
    backend.generate=original
    def fake_evaluate(*args):
        args[0].generate("question",max_new_tokens=4)
        raise RuntimeError("planned")
    monkeypatch.setattr(run,"evaluate",fake_evaluate)
    with pytest.raises(RuntimeError,match="planned"):
        run.evaluate_student(backend,backend.vector_vera,None,[],"real","dev",tmp_path,7)
    assert calls==[7]
    assert backend.generate is original


def test_four_quadrants_rotate_only_development_styles_and_save_frozen_stores(monkeypatch,tmp_path):
    backend=make_backend()
    module=backend.vector_vera
    examples=data.datasets(train_size=16)["dev"]
    generator=torch.Generator().manual_seed(407)
    features=torch.randn(64,3,5,generator=generator)
    packet=dict(examples=dict(dev=[asdict(ex) for ex in examples]),
                features=dict(dev=dict(s=features)),
                template_ids=dict(dev=dict(q=["train_query_00","dev_query_00","dev_query_01"],
                                           s=["train_support_00","dev_support_00","dev_support_01"])))
    calls=[]
    observed_query_styles={}
    def student(backend,module,db,rendered,method,phase,output,max_new_tokens):
        calls.append((phase,method))
        support_ood=phase.startswith("heldout_support")
        query_ood=phase.endswith("heldout_query")
        if query_ood:
            observed_query_styles[phase.split("/")[0]]=[
                next(view for view in (1,2)
                     if ex.question==data.render_question(ex,packet["template_ids"]["dev"]["q"][view]))
                for ex in rendered]
        assert list(db.ids)==[ex.id for ex in examples]
        for index,example in enumerate(rendered):
            qview=1+(index//2)%2 if query_ood else 0
            assert example.question==data.render_question(examples[index],packet["template_ids"]["dev"]["q"][qview])
            sview=1+index%2 if support_ood else 0
            torch.testing.assert_close(db.keys[index],module.encode_key(features[index,sview]))
        assert not any(parameter.requires_grad for parameter in module.parameters())
        return [dict(em=0,nll_sum=1.,tokens=1,method=method,expected_in_bank=False)]*len(rendered)
    def alignment(backend,module,db,rendered,contexts,batch_size,phase,output):
        for index,(ex,context) in enumerate(zip(rendered,contexts)):
            view=1+index%2 if phase.startswith("heldout_support") else 0
            assert context==data.render_support(ex,packet["template_ids"]["dev"]["s"][view])
        return dict(count=len(rendered))
    monkeypatch.setattr(run,"evaluate_student",student)
    monkeypatch.setattr(run,"diagnostic_alignment",alignment)
    before=run.tensor_digest(module)
    config={**_config("ce"),"eval_size":64}
    metrics=run.evaluate_development(backend,module,config,packet,tmp_path)
    assert len(calls)==16 and len(metrics["conditions"])==4
    assert all(value["teacher"] is None for value in metrics["conditions"].values())
    assert before==run.tensor_digest(module)
    writes=json.loads((tmp_path/"writes.json").read_text())
    assert len(writes)==128
    support_styles=[packet["template_ids"]["dev"]["s"].index(row["support_template"])
                    for row in writes if row["bank"]=="heldout_support"]
    assert observed_query_styles["canonical_support"]==observed_query_styles["heldout_support"]
    pairs=list(zip(observed_query_styles["heldout_support"],support_styles))
    assert set(pairs)=={(1,1),(1,2),(2,1),(2,2)}
    assert all(pairs.count(pair)==16 for pair in set(pairs))
    for name in ("canonical_support","heldout_support"):
        assert (tmp_path/(name+"_vdb.pt")).exists()
        assert (tmp_path/(name+"_initial_vdb.pt")).exists()


def test_parameter_digest_supports_bfloat16_and_detects_changes():
    model=torch.nn.Linear(4,3).to(torch.bfloat16).requires_grad_(False)
    before=run.parameter_digest(model)
    with torch.no_grad():
        model.weight[0,0]+=1
    assert run.parameter_digest(model)!=before


def test_cli_rejects_invalid_budgets_and_preserves_defaults():
    required=["--model","/tmp/model","--cache","/tmp/cache","--checkpoint","/tmp/anchor","--run-dir","/tmp/run"]
    cfg=run.parse_args(required)
    assert cfg.updates==256 and cfg.batch_size==8 and cfg.eval_size==64 and cfg.max_new_tokens==4
    for args in (["--updates","0"],["--eval-size","65"],["--sampling-temperature","nan"]):
        with pytest.raises(SystemExit):run.parse_args(required+args)
