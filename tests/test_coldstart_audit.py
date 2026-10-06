"""Independent corruption checks for paired budgets and foundation usage."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
torch=pytest.importorskip("torch")
SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("audit_coldstart",SCRIPTS/"audit_coldstart.py")
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def stats(n):
    entropy=-sum(p*math.log(p) for p in (.75,.25)) if n else 0.
    return dict(records=2,search_calls=int(n>0),query_positions=n,weight_mass=float(n),weight_entropy=entropy,
        effective_slots=math.exp(entropy) if n else 0.,top1_used=int(n>0),topk_used=2 if n else 0,
        sparse_counts=[dict(index=0,top1=n,topk=n,weight_mass=.75*n),dict(index=1,top1=0,topk=n,weight_mass=.25*n)] if n else [])


def usage_fixture():
    raw=[dict(method='real',answer_results={'A':{}}),dict(method='teacher',answer_results={'A':{}})]
    rows=[]
    for index,(prediction,kind,teacher,n) in enumerate(((0,'generation',False,2),(0,'answer_scoring',False,1),(1,'generation',True,0),(1,'answer_scoring',True,0))):
        rows.append(dict(call_index=index,prediction_index=prediction,kind=kind,teacher=teacher,bank_hash='abc',**stats(n)))
    total=stats(3);total['search_calls']=2
    return rows,raw,total


def test_usage_counts_generation_and_scoring_once_and_rejects_corruption():
    rows,raw,total=usage_fixture()
    actual=audit.audit_usage(rows,raw,total,'abc',2,2)
    assert actual['calls']==4 and actual['generation_calls']==2 and actual['query_positions']==3
    wrong=copy.deepcopy(total);wrong['weight_mass']+=1
    with pytest.raises(ValueError,match='mass'):audit.audit_usage(rows,raw,wrong,'abc',2,2)
    wrong=copy.deepcopy(rows);wrong[2]['search_calls']=1
    with pytest.raises(ValueError,match='Teacher used'):audit.audit_usage(wrong,raw,total,'abc',2,2)
    with pytest.raises(ValueError,match='Missing foundation'):audit.audit_usage(rows[:-1],raw,total,'abc',2,2)


class Inputs:
    def __init__(self,packets):self.packets=packets
    def load(self,path):return self.packets[path]


def paired_fixture():
    token_totals=dict(gold_target_tokens=16,target_tokens=16,distillation_target_tokens=16,student_input_tokens=64,teacher_input_tokens=120)
    common=dict(updates=1,target_exposures=8,schedule_sha256='same',gold_token_schedule_sha256='gold',initial_checkpoint_sha256='init',token_totals=token_totals)
    trainings=[dict(common,arm='wiki_warm',cache='wiki'),dict(common,arm='matched_warm',cache='matched')]
    row=dict(id='id',entity='entity',relation='r',a='a b c',b='d e f',a_tokens=[1,2,3],b_tokens=[4,5,6],questions=['q'])
    packets={domain:dict(rows=[copy.deepcopy(row)],token_repairs=[]) for domain in ('wiki','matched')}
    return trainings,Inputs(packets)


def test_pairing_compares_exact_gold_tokens_not_only_counts():
    trainings,inputs=paired_fixture()
    assert audit.warm_pairing(trainings,inputs)['status']=='passed'
    inputs.packets['matched']['rows'][0]['b_tokens']=[4,5,9]
    with pytest.raises(ValueError,match='gold/query'):audit.warm_pairing(trainings,inputs)
    assert audit.warm_pairing(trainings[:1],inputs)['status']=='pending_both_completed_warmups'


def test_pairing_rejects_equal_token_budget_with_different_schedule():
    trainings,inputs=paired_fixture();trainings[1]['schedule_sha256']='changed'
    with pytest.raises(ValueError,match='schedule'):audit.warm_pairing(trainings,inputs)


def test_empty_run_is_partial_and_cannot_pass_final_acceptance(tmp_path):
    result=audit.run_audit(tmp_path)
    assert result['checks_passed'] and not result['complete'] and len(result['missing_evaluation_cells'])==52
    with pytest.raises(ValueError,match='coverage incomplete'):audit.run_audit(tmp_path,True)


@pytest.mark.parametrize('mode',['full','off','random256','cluster256'])
def test_foundation_snapshot_replay_and_arithmetic_merge(tmp_path,mode):
    from vera_mem.dictionary_vera import DictionaryVeRA
    from vera_mem.coldstart_eval import prepare_base_bank,_Usage
    from vera_mem.run import tensor_digest
    module=DictionaryVeRA(4,3,rank=2,key_dim=2,base_size=8,seed=4).eval()
    checkpoint=tmp_path/'checkpoint.pt'
    torch.save(dict(module=module.state_dict(),architecture=module.configuration()),checkpoint)
    selected=prepare_base_bank(module,mode,2 if mode in ('random256','cluster256') else None)
    with module.use_cpu_base(selected['keys'],selected['values'],ids=selected['ids']) as store:
        bank_hash=store.hash();snapshot=dict(protocol='coldstart-cpu-banks-v1',**selected,store=store.snapshot(),
            bank_hash=bank_hash,alpha_base=module.alpha_base,original_shared_weights=tensor_digest(module))
        torch.save(snapshot,tmp_path/'foundation.pt')
    (tmp_path/'foundation_usage.jsonl').write_text('');(tmp_path/'predictions.jsonl').write_text('')
    cold=dict(protocol='coldstart-cpu-banks-v1',bank_hash_before=bank_hash,bank_hash_after=bank_hash,
        original_shared_weights_before=tensor_digest(module),original_shared_weights_after=tensor_digest(module),
        original_state_before={},original_state_after={},evaluated_routing='sparse',foundation_cpu=True,episodic_cpu=True,
        values_rms_recalibrated=False,count_bias_applied=False,dynamic_records_compressed=False,
        foundation_snapshot='foundation.pt',foundation_snapshot_sha256=audit.file_hash(tmp_path/'foundation.pt'),
        foundation_usage='foundation_usage.jsonl',foundation_usage_sha256=audit.file_hash(tmp_path/'foundation_usage.jsonl'),
        base_mode=mode,active_records=selected['active_records'],source_records=8,alpha_base=module.alpha_base,
        usage=_Usage(selected['active_records']).snapshot())
    (tmp_path/'metrics.json').write_text(json.dumps(dict(coldstart=cold)))
    manifest=dict(configuration=dict(checkpoint=str(checkpoint),base_mode=mode),checkpoint_sha256=audit.file_hash(checkpoint))
    result=audit.audit_foundation(tmp_path,manifest,audit.Inputs(tmp_path))
    assert result['mode']==mode and result['bank_hash']==bank_hash
    snapshot['values']=snapshot['values']+.1
    torch.save(snapshot,tmp_path/'foundation.pt');cold['foundation_snapshot_sha256']=audit.file_hash(tmp_path/'foundation.pt')
    (tmp_path/'metrics.json').write_text(json.dumps(dict(coldstart=cold)))
    if mode!='off':
        with pytest.raises(ValueError):audit.audit_foundation(tmp_path,manifest,audit.Inputs(tmp_path))


def test_training_recomputes_gold_eos_budget_and_rejects_frozen_gradients(tmp_path):
    from vera_mem.dictionary_vera import DictionaryVeRA
    module=DictionaryVeRA(4,3,rank=2,key_dim=2,base_size=8,base_trainable=False)
    initial=tmp_path/'initial.pt';cache=tmp_path/'train.pt';output=tmp_path/'job';output.mkdir()
    torch.save(dict(protocol='dictionary-coldstart-v1',module=module.state_dict(),architecture=module.configuration(),step=0),initial)
    records=[]
    for i in range(16):
        a,b=('alpha one two','beta one two') if i%2 else ('beta one two','alpha one two')
        records.append(dict(id=f'i{i}',entity=f'e{i}',a=a,b=b,a_tokens=[1,2,3],b_tokens=[4,2,3],questions=['q'],supports=[['a'],['b']]))
    torch.save(dict(protocol='dictionary-coldstart-v1',split='train',task='synthetic',rows=records,
        q=torch.zeros(16,1,4),last=torch.zeros(16,2,1,4),pool=torch.zeros(16,2,1,4)),cache)
    sample=dict(targets=list(range(8)),episode=list(range(16)),columns=list(range(8)),qviews=[0]*8,sviews=[0]*16)
    totals=dict(gold_target_tokens=96,distillation_target_tokens=96,target_tokens=96,student_input_tokens=100,teacher_input_tokens=200,rollout_input_tokens=0)
    row=dict(step=1,routing='sparse',sample=sample,gradient_norms=[1.,1.,1.],base_keys_gradient_slots=0,base_values_gradient_slots=0,**totals)
    status=dict(complete=True,updates=1,schedule_sha256=hashlib.sha256(json.dumps(sample,sort_keys=True).encode()).hexdigest(),
        totals=totals,target_unique_count=8,bank_unique_count=16,key_gradient_coverage=0,value_gradient_coverage=0,
        sampled_last_forward_top1_coverage=0,sampled_last_forward_topk_coverage=0,elapsed_seconds=1.)
    (output/'training_status.json').write_text(json.dumps(status));(output/'training.jsonl').write_text(json.dumps(row)+'\n')
    torch.save(dict(protocol='dictionary-coldstart-v1',module=module.state_dict(),architecture=module.configuration(),step=1),output/'last.pt')
    torch.save(dict(key_gradient=torch.zeros(8,dtype=torch.bool),value_gradient=torch.zeros(8,dtype=torch.bool),top1=torch.zeros(8,dtype=torch.long),topk=torch.zeros(8,dtype=torch.long)),output/'usage.pt')
    config=dict(cache=str(cache),checkpoint=str(initial),updates=1,batch_size=8,base_trainable=False,routing='sparse',arm='fixed')
    manifest=dict(configuration=config,result=status,cache_sha256=audit.file_hash(cache),checkpoint_sha256=audit.file_hash(initial),backbone_unchanged=True)
    result=audit.audit_training(output,manifest,audit.Inputs(tmp_path))
    assert result['token_totals']['gold_target_tokens']==96 and all(result['base_tensors_equal_to_initial'].values())
    row['base_keys_gradient_slots']=1;(output/'training.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='Frozen foundation received gradients'):audit.audit_training(output,manifest,audit.Inputs(tmp_path))
    row['base_keys_gradient_slots']=0;row['gold_target_tokens']=97;(output/'training.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='whole-answer'):audit.audit_training(output,manifest,audit.Inputs(tmp_path))
