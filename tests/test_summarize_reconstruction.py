"""Raw-text gate checks, selection barriers, and immutable evidence binding."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC=importlib.util.spec_from_file_location('reconstruction_summary',Path(__file__).resolve().parents[1]/'scripts/summarize_reconstruction.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


def arch(slots=1,learned=False,seed=71042):
    return dict(slots=slots,train_B=learned,seed=seed,rank=64,key_dim=64,top_k=4,
        in_features=9728,out_features=2560,writer_mode='masked_mean',value_mlp_hidden=0)


def raw_fixture(tmp_path,*,wrong_local=False):
    directory=tmp_path/'case';directory.mkdir()
    rows=[dict(id='id'+str(i),questions=['question '+str(i),'held '+str(i)],
        a='alpha beta '+str(i),b='bravo gamma '+str(i),c='charlie delta '+str(i),d='delta echo '+str(i),swap_a='echo foxtrot '+str(i)) for i in range(16)]
    raw=[];pairs=[]
    def add(i,w,ph,cond,role,case,trigger):
        answer=rows[i][audit.FIELDS[w]]
        wrong=cond!='real' or (wrong_local and w=='A' and i==1)
        prediction='incorrect' if wrong else answer
        slots=0 if cond=='empty' else 16
        chosen=[] if not slots else [i]+[j for j in range(16) if j!=i][:3]
        h=audit.digest([ph,'A'] if role=='initial' else [ph,case,trigger,role if role!='locality' else 'updated',cond])
        r=dict(id=rows[i]['id'],case_id=case,phase=ph,world=w,condition=cond,role=role,trigger_world=trigger,
            question=rows[i]['questions'][audit.PHASES[ph][1]],answer=answer,prediction=prediction,em=int(not wrong),
            generation_tokens=3,generation_seconds=.01,budget_hit=False,nll_sum=2.,answer_tokens=4,bank_hash=h,bank_slots=slots,
            trace=[dict(phase='prefill',indices=[chosen])],first_fact_recall_at_4=int(bool(slots)),decode_hits=0,decode_queries=0)
        raw.append(r);return r
    for ph,ws in audit.expected_axes('known','development').items():
        initial=[add(i,'A',ph,'real','initial',rows[i]['id'],'A') for i in range(16)]
        for w in ws:
            for i in range(16):
                u=add(i,w,ph,'real','updated',rows[i]['id'],w)
                pair=dict(id=rows[i]['id'],phase=ph,world=w,a_em=initial[i]['em'],updated_em=u['em'],
                    pair_em=initial[i]['em']*u['em'],restored_em=None,locality_equal=None,locality_correct=None)
                if ph=='CC':
                    ni=(i+1)%16;l=add(ni,'A',ph,'real','locality',rows[i]['id'],w);z=add(i,'A',ph,'real','restored',rows[i]['id'],w)
                    pair.update(locality_equal=int(l['prediction']==initial[ni]['prediction']),locality_correct=l['em'],
                        locality_joint=l['em']*initial[ni]['em'],restored_em=z['em'],update_restore_em=u['em']*z['em'],all_three_em=initial[i]['em']*u['em']*z['em'])
                    if w!='SWAP':
                        for cond in ('shuffle','empty'):pair[cond+'_em']=add(i,w,ph,cond,'control',rows[i]['id'],w)['em']
                pairs.append(pair)
    groups={}
    for key in dict.fromkeys(p['phase']+'_'+p['world'] for p in pairs):
        pp=[p for p in pairs if p['phase']+'_'+p['world']==key];g={'count':len(pp)}
        for name in ('a_em','updated_em','pair_em','restored_em','update_restore_em','all_three_em','locality_equal','locality_correct','locality_joint','shuffle_em','empty_em'):
            vs=[p[name] for p in pp if p.get(name) is not None]
            if vs:g[name]=sum(vs)/len(vs)
        groups[key]=g
    summary=dict(protocol='reconstruction-cpu-vdb-v1',complete=True,groups=groups,shared_parameters_unchanged=True,shared_parameter_sha256='a'*64,
        generation_calls=len(raw),generation_tokens=len(raw)*3,generation_seconds=len(raw)*.01,single_group_updates=len(pairs),
        independent_interventions=96,restorations=48,real_group_write_events=144,
        bank_facts=16,slots_per_fact=1,resident_bytes=dict(key_bytes=4096,value_bytes=4096,total_bytes=8192),architecture=arch())
    manifest=dict(protocol=audit.RUN_PROTOCOL,stage='eval',complete=True,backbone_unchanged=True,
        configuration={'eval_part':'development'},cache_sha256='b'*64,checkpoint_sha256='c'*64,result=summary)
    def save():
        for name,obj in (('manifest.json',manifest),('summary.json',summary),('pairs.json',pairs)):(directory/name).write_text(json.dumps(obj))
        (directory/'predictions.jsonl').write_text('\n'.join(map(json.dumps,raw))+'\n')
        for name in ('encoded_payloads.pt','bank_events.pt'):(directory/name).write_bytes(b'bound fixture tensor')
    save();return directory,manifest,rows,raw,pairs,summary,save


def test_recompute_complete_known_grid_restore_and_control_counts(tmp_path):
    d,m,rows,*_=raw_fixture(tmp_path)
    r=audit.audit_evaluation(d,m,rows,'known')
    assert r['costs']['generation_calls']==320 and r['single_group_updates']==96
    assert r['groups']['CC_C']['update_restore_em']==1 and r['groups']['CC_C']['empty_em']==0
    assert r['bytes_per_fact']==512


@pytest.mark.parametrize('attack',['fake_em','wrong_trigger','extra_row','recall','restore_answer','locality_bank'])
def test_tampered_raw_evidence_is_rejected(tmp_path,attack):
    d,m,rows,raw,pairs,s,save=raw_fixture(tmp_path)
    if attack=='fake_em':raw[0]['em']=0
    if attack=='wrong_trigger':next(r for r in raw if r['role']=='restored')['trigger_world']='C'
    if attack=='extra_row':raw.append(raw[-1].copy())
    if attack=='recall':raw[0]['first_fact_recall_at_4']=0
    if attack=='restore_answer':next(r for r in raw if r['role']=='restored')['answer']='other payload'
    if attack=='locality_bank':next(r for r in raw if r['role']=='locality')['bank_hash']='d'*64
    save()
    with pytest.raises(ValueError):audit.audit_evaluation(d,m,rows,'known')


def test_constant_wrong_neighbor_is_equal_but_not_jointly_correct(tmp_path):
    d,m,rows,*_=raw_fixture(tmp_path,wrong_local=True)
    r=audit.audit_evaluation(d,m,rows,'known');g=r['groups']['CC_C']
    assert g['locality_equal']==1 and g['locality_joint']==15/16
    assert g['locality_joint']<.95


def development_records():
    records=[]
    for a in audit.ARMS:
        slots=int(a[1]);learned=a.endswith('trainB')
        for seed in audit.SEEDS:
            base=dict(arm=a,seed=seed,artifacts={'manifest.json':audit.digest([a,seed])})
            records.append(dict(base,kind='training',run_dir=a+str(seed)+'train',trainable_parameters=100+(163840 if learned else 0)))
            g=dict(count=16,pair_em=1,updated_em=1,update_restore_em=1,shuffle_em=0,empty_em=0,locality_joint=1)
            records.append(dict(base,kind='evaluation',run_dir=a+str(seed)+'known',split='known',part='development',
                groups={'CC_B':g.copy(),'CC_C':g.copy()},bytes_per_fact=slots*512))
            records.append(dict(base,kind='evaluation',run_dir=a+str(seed)+'dev',split='dev',part='development',groups={}))
    records.append(dict(kind='teacher',run_dir='preflight',split='train',part='preflight',
        groups={'CC_A':{'correct':16,'count':16,'em':1},'CC_B':{'correct':16,'count':16,'em':1}},artifacts={'manifest.json':'d'*64}))
    return records


def test_selection_requires_all_twelve_conditions_and_all_seeds_pass():
    records=development_records();d=audit.development_selection(records)
    assert d['selected_arm']=='S1_fixedB' and d['development_barrier']['expected']==36
    with pytest.raises(ValueError,match='barrier'):audit.development_selection(records[1:])
    with pytest.raises(ValueError,match='barrier'):audit.development_selection(records+[records[0]])
    for r in records:
        if r.get('arm')=='S1_fixedB' and r.get('part')=='development' and r.get('split')=='known' and r['seed']==71043:
            r['groups']['CC_C']['locality_joint']=15/16
    d=audit.development_selection(records)
    assert not d['candidates'][0]['eligible'] and d['selected_arm']=='S3_fixedB'


def test_selector_uses_worst_seed_then_parameters_bytes_never_newentity_score():
    r=development_records()
    for x in r:
        if x.get('kind')=='evaluation' and x.get('split')=='known' and x['arm']=='S1_fixedB':x['groups']['CC_C']['update_restore_em']=15/16
    assert audit.development_selection(r)['selected_arm']=='S3_fixedB'
    # A high capacity arm cannot substitute a single failing seed.
    for x in r:
        if x.get('kind')=='evaluation' and x.get('split')=='known' and x['seed']==71044:x['groups']['CC_C']['updated_em']=.4
    assert audit.development_selection(r)['selected_arm'] is None


def test_empty_pair_zero_does_not_replace_single_world_memory_gate():
    r=development_records()
    for x in r:
        if x.get('kind')=='evaluation' and x.get('split')=='known':
            x['groups']['CC_C']['empty_em']=.75
    assert audit.development_selection(r)['selected_arm'] is None


def test_final_confirmation_cannot_replace_selected_architecture_or_dev_evidence():
    records=development_records();decision=audit.development_selection(records)
    selection=dict(protocol=audit.SELECTION_PROTOCOL,sealed=True,decision=copy.deepcopy(decision))
    for a in audit.ARMS:
        for seed in audit.SEEDS:
            records += [dict(kind='evaluation',arm=a,seed=seed,split='known',part='confirmation',
                run_dir='known'+a+str(seed),groups={'CC_D':{'count':16,'update_restore_em':1}}),
                dict(kind='evaluation',arm=a,seed=seed,split='confirm',part='confirmation',
                run_dir='confirm'+a+str(seed),groups={'CC_B':{'count':64,'pair_em':51/64 if a=='S1_fixedB' else 1}})]
    result=audit.final_gate({'records':records},selection)
    assert result['selected_arm']=='S1_fixedB' and not result['eligible_to_expand']
    # 80 percent requires ceil(.8*64) = 52; do not round 51/64 upwards.
    for r in records:
        if r.get('arm')=='S1_fixedB' and r.get('split')=='confirm':r['groups']['CC_B']['pair_em']=52/64
    assert audit.final_gate({'records':records},selection)['eligible_to_expand']
    records[0]['artifacts']['manifest.json']='f'*64
    with pytest.raises(ValueError,match='Sealed development'):audit.final_gate({'records':records},selection)


def test_teacher_preflight_and_formal_architecture_are_strict():
    r=development_records();r[-1]['groups']['CC_A']['correct']=15
    with pytest.raises(ValueError,match='teacher'):audit.development_selection(r)
    with pytest.raises(ValueError,match='seed'):audit.architecture_identity(arch(seed=42))
    with pytest.raises(ValueError,match='rank'):audit.architecture_identity(dict(arch(),rank=128))


def test_discovery_excludes_source_and_only_explicit_smoke(tmp_path):
    for rel in ('reconstruction_prepare/features','reconstruction_prepare/teacher_preflight','reconstruction_prepare/smoke',
                'reconstruction_prepare/source/copied','other_task/train'):
        p=tmp_path/rel;p.mkdir(parents=True);(p/'manifest.json').write_text('{}')
    found=audit.discover(tmp_path)
    assert len(found)==3
    assert sum(audit.ignored_reason(p,tmp_path) is not None for p in found)==1
    assert audit.ignored_reason(tmp_path/'reconstruction_prepare/teacher_preflight/manifest.json',tmp_path) is None


def test_cli_seals_once_and_refuses_any_confirmation_child(tmp_path,monkeypatch):
    records=development_records();report=dict(records=records,pending=[],partial=True,audit_passed=True)
    monkeypatch.setattr(audit,'summarize',lambda root:copy.deepcopy(report))
    out=tmp_path/'dev.json';selection=tmp_path/'dev.selection.json'
    audit.main(['--runs-root',str(tmp_path),'--output',str(out),'--select'])
    sealed=json.loads(selection.read_text());assert sealed['sealed'] and sealed['decision']['selected_arm']=='S1_fixedB'
    with pytest.raises(FileExistsError):audit.main(['--runs-root',str(tmp_path),'--output',str(out),'--select'])
    report['pending']=[dict(stage='eval',part='confirmation')]
    with pytest.raises(ValueError,match='confirmation child'):
        audit.main(['--runs-root',str(tmp_path),'--output',str(tmp_path/'unsafe.json'),'--select'])
    assert not(tmp_path/'unsafe.selection.json').exists()


def test_training_audit_recomputes_exhaustive_schedule_and_actual_one_call_per_step(tmp_path):
    import hashlib,random
    cfg=dict(updates=1024,batch_size=8,seed=71042,slots=1,learned_b=False)
    rng=random.Random(71042);queue=[];raw=[];h=hashlib.sha256()
    for step in range(1,1025):
        if not queue:queue=list(range(16));rng.shuffle(queue)
        targets=queue[:8];del queue[:8];h.update(json.dumps(targets).encode())
        raw.append(dict(step=step,targets=targets,metrics=[dict(ce=1.,address=.2,loss=1.04)]*2,
            gradient_norms=[1.,2.,3.],elapsed_seconds=float(step)))
    status=dict(complete=True,updates=1024,schedule_sha256=h.hexdigest(),architecture=arch(),
        learning_rates=[1e-4,3e-4,.005],trainable_parameters=100,elapsed_seconds=1024.,
        totals=dict(target_exposures=8192,gold_tokens=65536,input_positions=131072,padded_input_positions=131072,backbone_calls=1024))
    m=dict(configuration=cfg,result=status,cache_sha256='a'*64)
    for name,obj in (('manifest.json',m),('training_status.json',status)):(tmp_path/name).write_text(json.dumps(obj))
    (tmp_path/'training.jsonl').write_text('\n'.join(map(json.dumps,raw))+'\n')
    for name in ('initial.pt','last.pt','step_1024.pt'):(tmp_path/name).write_bytes(name.encode())
    result=audit.audit_training(tmp_path,m)
    assert result['updates']==1024 and result['costs']['backbone_calls']==1024
    status['totals']['backbone_calls']=2048
    (tmp_path/'training_status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError,match='one combined A/B'):audit.audit_training(tmp_path,m)
    status['totals']['backbone_calls']=1024;(tmp_path/'training_status.json').write_text(json.dumps(status))
    raw[1]['targets']=raw[0]['targets']
    (tmp_path/'training.jsonl').write_text('\n'.join(map(json.dumps,raw))+'\n')
    with pytest.raises(ValueError,match='target schedule'):audit.audit_training(tmp_path,m)
