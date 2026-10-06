"""QKV raw reconstruction, token/schedule accounting, and sealed-gate contracts."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

SPEC=importlib.util.spec_from_file_location('qkv_summary',Path(__file__).resolve().parents[1]/'scripts/summarize_qkv.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


def architecture(seed=81042,mode='grouped',readout='vera'):
    return dict(slots=3,train_B=True,seed=seed,rank=64,key_dim=64,top_k=4,temperature=.05,
        in_features=9728,out_features=2560,writer_mode='masked_mean',value_mlp_hidden=0,qkv_version=1,
        read_mode=mode,readout=readout)


def packet_rows(n=16):
    rows=[]
    for i in range(n):
        row=dict(id=f'fact{i}',entity=f'entity{i}',questions=[f'canonical {i}',f'heldout {i}'],
            a=f'alpha beta item{i}',b=f'bravo gamma item{i}',c=f'charlie delta item{i}',d=f'delta echo item{i}',swap_a=f'echo foxtrot item{i}',
            worlds=['A','B','C','D','SWAP'],provenance={'edited_word_index':i%3})
        row['supports']=[[f'{w} canonical {i}',f'{w} heldout {i}'] for w in row['worlds']];rows.append(row)
    return rows


def raw_fixture(tmp_path,split='known',part='development',mode='grouped',wrong_local=False):
    d=tmp_path/'evaluation';d.mkdir();rows=packet_rows();raw=[];pairs=[]
    def add(i,w,ph,cond,role,case,trigger):
        answer=rows[i][audit.FIELDS[w]];wrong=cond!='real' or (wrong_local and w=='A' and i==1)
        prediction='incorrect' if wrong else answer
        if cond=='empty':ii=[];scores=[];weights=[]
        elif mode=='grouped':ii=[3*i+j for j in range(3)];scores=[0.,.05*math.log(3),0.];weights=[.2,.6,.2]
        else:ii=[3*i+2,3*i+1,3*((i+1)%16),3*i];scores=[.05*math.log(j) for j in (4,3,2,1)];weights=[.4,.3,.2,.1]
        masses=[sum(weight for j,weight in zip(ii,weights) if j==3*i+s) for s in range(3)]
        trace=[dict(phase='prefill' if t==0 else 'decode',token_id=token,indices=[ii],weights=[weights],scores=[scores],target_slot_mass=masses,target_fact_mass=sum(masses)) for t,token in enumerate([7,8,9])]
        bank_hash=audit.digest([ph,'A'] if role=='initial' else [ph,case,trigger,'updated' if role=='locality' else role,cond])
        r=dict(id=rows[i]['id'],case_id=case,phase=ph,world=w,condition=cond,role=role,trigger_world=trigger,
            question=rows[i]['questions'][audit.PHASES[ph][1]],answer=answer,prediction=prediction,em=int(not wrong),
            generated_token_ids=[7,8,9],generation_tokens=3,generation_seconds=.01,budget_hit=False,nll_sum=2.,answer_tokens=4,
            bank_hash=bank_hash,bank_slots=0 if cond=='empty' else 48,trace=trace,first_fact_recall_at_read=int(bool(ii)),
            first_fact_recall_at_1=int(bool(ii)),decode_hits=2*int(bool(ii)),decode_queries=2,
            first_target_slot_mass=masses,first_target_fact_mass=sum(masses),actual_read_slots=len(ii))
        raw.append(r);return r
    for ph,worlds in audit.expected_axes(split,part).items():
        initial=[add(i,'A',ph,'real','initial',rows[i]['id'],'A') for i in range(16)]
        for world in worlds:
            for i in range(16):
                u=add(i,world,ph,'real','updated',rows[i]['id'],world);ni=(i+1)%16
                l=add(ni,'A',ph,'real','locality',rows[i]['id'],world);z=add(i,'A',ph,'real','restored',rows[i]['id'],world)
                p=dict(id=rows[i]['id'],phase=ph,world=world,a_em=initial[i]['em'],updated_em=u['em'],pair_em=initial[i]['em']*u['em'],
                    restored_em=z['em'],update_restore_em=u['em']*z['em'],all_three_em=initial[i]['em']*u['em']*z['em'],
                    locality_equal=int(l['prediction']==initial[ni]['prediction']),locality_correct=l['em'],locality_joint=l['em']*initial[ni]['em'],shuffle_em=None,empty_em=None)
                if world!='SWAP':
                    for cond in ('shuffle','empty'):p[cond+'_em']=add(i,world,ph,cond,'control',rows[i]['id'],world)['em']
                pairs.append(p)
    groups={}
    for key in dict.fromkeys(p['phase']+'_'+p['world'] for p in pairs):
        pp=[p for p in pairs if p['phase']+'_'+p['world']==key]
        groups[key]=dict(count=len(pp),**{k:sum(p[k] for p in pp)/len(pp) if pp[0][k] is not None else None for k in audit.PAIR_METRICS})
    summary=dict(protocol='qkv-online-cpu-v1',complete=True,architecture=architecture(mode=mode),groups=groups,
        shared_parameters_unchanged=True,shared_parameter_sha256='a'*64,bank_facts=16,slots_per_fact=3,
        resident_bytes=dict(key_bytes=12288,value_bytes=12288,total_bytes=24576),generation_calls=len(raw),generation_tokens=3*len(raw),generation_seconds=.01*len(raw),
        answer_scoring_tokens=4*len(raw),independent_interventions=len(pairs),restorations=len(pairs),real_group_write_events=2*len(pairs))
    manifest=dict(protocol=audit.RUN_PROTOCOL,stage='eval',complete=True,backbone_unchanged=True,configuration={'eval_part':part},cache_sha256='b'*64,checkpoint_sha256='c'*64,result=summary)
    def save():
        for name,obj in (('manifest.json',manifest),('summary.json',summary),('pairs.json',pairs)):(d/name).write_text(json.dumps(obj))
        (d/'predictions.jsonl').write_text('\n'.join(map(json.dumps,raw))+'\n')
        for name in ('encoded_payloads.pt','bank_events.pt'):(d/name).write_bytes(b'tensor hash only')
    save();return d,manifest,rows,raw,pairs,summary,save


@pytest.mark.parametrize('split,part,calls,interventions',[('known','development',224,48),('dev','development',176,32),('known','confirmation',96,16),('confirm','confirmation',704,128)])
@pytest.mark.parametrize('mode',['flat','grouped'])
def test_all_grids_have_independent_update_locality_restore_and_controls(tmp_path,split,part,calls,interventions,mode):
    d,m,rows,*_=raw_fixture(tmp_path,split,part,mode)
    r=audit.audit_evaluation(d,m,rows,split,'rebind')
    assert r['costs']['generation_calls']==calls and r['independent_interventions']==interventions
    assert r['restoration_writes']==interventions and r['real_group_write_events']==2*interventions and r['bytes_per_fact']==1536
    assert all(g['locality_joint']==1 for g in r['groups'].values())
    if 'CC_C' in r['updated_diagnostics']:
        diag=r['updated_diagnostics']['CC_C']['novel_content']
        assert diag['count']==16 and diag['edited_slot_ever_read']==16
        assert 'no gold-forced' in diag['alignment_scope']


@pytest.mark.parametrize('attack',['em','token_count','token_id','trace_count','phase','weights','mass','r1','read_width','trigger','bank','answer','extra_row','pairs','group_order'])
def test_raw_tampering_rejected(tmp_path,attack):
    d,m,rows,raw,pairs,summary,save=raw_fixture(tmp_path)
    if attack=='em':raw[0]['em']=0
    if attack=='token_count':raw[0]['generation_tokens']=4
    if attack=='token_id':raw[0]['trace'][0]['token_id']=99
    if attack=='trace_count':raw[0]['trace'].pop()
    if attack=='phase':raw[0]['trace'][0]['phase']='decode'
    if attack=='weights':raw[0]['trace'][0]['weights'][0][0]=.1
    if attack=='mass':raw[0]['trace'][0]['target_slot_mass'][1]=0
    if attack=='r1':raw[0]['first_fact_recall_at_1']=0
    if attack=='read_width':raw[0]['actual_read_slots']=4
    if attack=='trigger':next(r for r in raw if r['role']=='restored')['trigger_world']='C'
    if attack=='bank':next(r for r in raw if r['role']=='locality')['bank_hash']='f'*64
    if attack=='answer':next(r for r in raw if r['role']=='restored')['answer']='different gold'
    if attack=='extra_row':raw.append(raw[-1])
    if attack=='pairs':pairs[0]['pair_em']=0
    if attack=='group_order':raw[0]['trace'][0]['indices'][0]=[1,0,2]
    save()
    with pytest.raises(ValueError):audit.audit_evaluation(d,m,rows,'known','static')


def test_wrong_neighbor_unchanged_does_not_count_as_correct_locality(tmp_path):
    d,m,rows,*_=raw_fixture(tmp_path,wrong_local=True)
    result=audit.audit_evaluation(d,m,rows,'known','static')
    assert result['groups']['CC_C']['locality_equal']==1
    assert result['groups']['CC_C']['locality_joint']==15/16


def teacher_fixture(tmp_path,failed=False):
    d=tmp_path/'teacher';d.mkdir();rows=packet_rows(64)
    for r in rows:r['worlds']=['A','B'];r['questions']=r['questions'][:1];r['supports']=[s[:1] for s in r['supports'][:2]]
    raw=[]
    for world in ['A','B']:
        for i,row in enumerate(rows):
            ok=not(failed and i==0 and world=='B')
            raw.append(dict(id=row['id'],phase='CC',world=world,question=row['questions'][0],context=row['supports'][row['worlds'].index(world)][0],answer=row[audit.FIELDS[world]],
                prediction=row[audit.FIELDS[world]] if ok else 'wrong',em=int(ok),generation_tokens=2,generated_token_ids=[10,11],generation_seconds=.1,
                teacher_bypasses_memory=True,memory_access=dict(mode='none',store_present=False,trace=[],last_retrieval_present=False)))
    groups={f'CC_{world}':dict(count=64,correct=64-int(failed and world=='B'),em=(64-int(failed and world=='B'))/64) for world in ['A','B']}
    summary=dict(complete=True,groups=groups,qualified=not failed,generation_calls=128,generation_tokens=256,generation_seconds=12.8)
    manifest=dict(result=summary,configuration={'eval_part':'preflight'},cache_sha256='b'*64)
    def save():
        (d/'manifest.json').write_text(json.dumps(manifest));(d/'summary.json').write_text(json.dumps(summary));(d/'predictions.jsonl').write_text('\n'.join(map(json.dumps,raw)))
    save();return d,manifest,rows,raw,summary,save


def test_teacher_keeps_failed_case_full_denominator_and_audits_memory_bypass(tmp_path):
    d,m,rows,raw,s,save=teacher_fixture(tmp_path,True)
    result=audit.audit_teacher(d,m,rows,'train')
    assert not result['qualified'] and result['paired']['CC_B']['count']==64 and result['paired']['CC_B']['both_correct']==63
    raw[0]['memory_access']['trace']=[{'indices':[0]}];save()
    with pytest.raises(ValueError,match='Teacher accessed memory'):audit.audit_teacher(d,m,rows,'train')


def test_independent_episode_matches_runtime_and_regime_blind_target_order():
    from vera_mem.qkv_data import episode
    for seed in audit.SEEDS:
        for step in (0,1,7,8,67,2047):
            for regime in ('static','rebind'):assert audit.episode(step,regime,seed)==episode(step,regime,seed=seed)
            a,b=audit.episode(step,'static',seed),audit.episode(step,'rebind',seed)
            assert all(a[k]==b[k] for k in ('targets','background','bank_entities','local_targets'))


def train_fixture(tmp_path,regime='rebind'):
    d=tmp_path/'training';d.mkdir();raw=[];totals=dict(target_exposures=0,gold_tokens=0,input_positions=0,padded_input_positions=0,backbone_calls=0)
    import hashlib
    h=hashlib.sha256()
    for step in range(2048):
        ep=audit.episode(step,regime,81042)
        aid=[ep[k][j] for k in ('a_payload_indices','b_payload_indices') for j in ep['local_targets']]
        gl=[3+i%4 for i in aid];il=[g+10+e%3 for e,g in zip(ep['targets']*2,gl)]
        raw.append(dict(step=step+1,episode=ep,metrics=dict(ce=1.,address=2.,loss=1.4),gradient_norms=[1.]*4,
            parameter_gradient_norms={'b':1.},input_lengths=il,gold_token_lengths=gl,elapsed_seconds=float(step+1)))
        h.update(json.dumps(ep,sort_keys=True).encode());totals['target_exposures']+=8;totals['gold_tokens']+=sum(gl)
        totals['input_positions']+=sum(il);totals['padded_input_positions']+=16*max(il);totals['backbone_calls']+=1
    status=dict(complete=True,updates=2048,regime=regime,architecture=architecture(),trainable_parameters=2034368,
        schedule_sha256=h.hexdigest(),totals=totals,learning_rates=[1e-4,3e-4,.005,3e-4],elapsed_seconds=2049.)
    manifest=dict(result=status,configuration=dict(updates=2048,regime=regime,read_mode='grouped',readout='vera',seed=81042),cache_sha256='b'*64)
    def save():
        for name,obj in (('manifest.json',manifest),('training_status.json',status)):(d/name).write_text(json.dumps(obj))
        (d/'training.jsonl').write_text('\n'.join(map(json.dumps,raw)))
        for name in ('initial.pt','last.pt','step_1024.pt','step_2048.pt'):(d/name).write_bytes(name.encode())
    save();return d,manifest,raw,status,save


def test_train_recomputes_2048_schedule_token_costs_and_payload_residency(tmp_path):
    d,m,raw,s,save=train_fixture(tmp_path)
    r=audit.audit_training(d,m,{})
    assert r['costs']['target_exposures']==16384 and r['costs']['backbone_calls']==2048
    assert set(r['exposure']['target_counts'])==set(r['exposure']['supervised_payload_counts'])=={256}
    assert sum(r['exposure']['physical_payload_bank_residency'])==2048*16*16
    assert r['costs']['gold_tokens']==256*sum(3+i%4 for i in range(128))
    s['totals']['input_positions']+=1;save()
    with pytest.raises(ValueError,match='token/exposure ledger'):audit.audit_training(d,m,{})


@pytest.mark.parametrize('attack',['schedule','lengths','missing_ledger','loss','padding'])
def test_train_detects_schedule_and_cost_tampering(tmp_path,attack):
    d,m,raw,s,save=train_fixture(tmp_path)
    if attack=='schedule':raw[17]['episode']['mapping'][0][0]=127
    if attack=='lengths':raw[17]['gold_token_lengths'][0]+=1
    if attack=='missing_ledger':raw[0]['input_lengths']=[]
    if attack=='loss':raw[0]['metrics']['loss']=1.5
    if attack=='padding':s['totals']['padded_input_positions']+=16
    save()
    with pytest.raises(ValueError):audit.audit_training(d,m,{})


def development_records():
    records=[]
    for arm in audit.ARMS:
        for seed in audit.SEEDS:
            common=dict(arm=arm,seed=seed)
            records.append(dict(common,kind='training',run_dir=f'{arm}/{seed}/train',trainable_parameters=2034368,
                artifacts={n:audit.digest([arm,seed,n]) for n in audit.ARTIFACT_NAMES['training']}))
            g=dict(count=16,pair_em=1.,updated_em=1.,update_restore_em=1.,shuffle_em=0.,empty_em=0.,locality_joint=1.)
            for split in ('known','dev'):
                records.append(dict(common,kind='evaluation',run_dir=f'{arm}/{seed}/{split}',split=split,part='development',
                    groups={'CC_B':g.copy(),'CC_C':g.copy()},bytes_per_fact=1536,
                    artifacts={n:audit.digest([arm,seed,split,n]) for n in audit.ARTIFACT_NAMES['evaluation']}))
    records.append(dict(kind='teacher',run_dir='preflight',split='train',part='preflight',
        groups={'CC_A':dict(count=64,correct=64,em=1.),'CC_B':dict(count=64,correct=64,em=1.)},
        artifacts={n:audit.digest(['teacher',n]) for n in audit.ARTIFACT_NAMES['teacher']}))
    for split in ('known','dev'):
        records.append(dict(kind='teacher',run_dir=split+'teacher',split=split,part='development'))
    return records


def selection_fixture(tmp_path):
    protocol=tmp_path/'protocol.md';protocol.write_text('frozen protocol')
    report=dict(records=development_records(),pending=[],teacher_scope={'complete':True})
    return report,audit.make_selection(report,protocol),protocol


def test_selection_requires_all_fifteen_models_and_excludes_additive(tmp_path):
    report,selection,protocol=selection_fixture(tmp_path)
    assert selection['decision']['selected_arm']=='static_flat'
    assert len(selection['development_artifacts'])==46
    assert len(selection['decision']['candidates'])==4
    assert audit.validate_selection(selection,protocol) is selection
    # The diagnostic arm still participates in completion, never candidate ranking.
    report['records']=[r for r in report['records'] if r.get('arm')!='rebind_grouped_additive']
    with pytest.raises(ValueError,match='barrier'):audit.development_selection(report['records'])


def test_worst_seed_new_entity_ranking_and_single_world_controls(tmp_path):
    report,_,protocol=selection_fixture(tmp_path)
    for r in report['records']:
        if r.get('arm')=='static_flat' and r.get('split')=='dev':r['groups']['CC_C']['updated_em']=8/16
    assert audit.development_selection(report['records'])['selected_arm']=='static_grouped'
    for r in report['records']:
        if r.get('kind')=='evaluation' and r['split']=='known' and r['seed']==81044:r['groups']['CC_C']['empty_em']=.75
    decision=audit.development_selection(report['records'])
    assert decision['selected_arm'] is None
    assert all(not c['eligible'] for c in decision['candidates'])


@pytest.mark.parametrize('attack',['additive','ranking','gates','fraction','protocol','evidence_duplicate','evidence_missing','evidence_hash','barrier','preflight','unsealed','mode'])
def test_launcher_validator_rejects_self_inconsistent_selection(tmp_path,attack):
    _,s,p=selection_fixture(tmp_path)
    if attack=='additive':s['decision']['candidates'][0]['arm']='rebind_grouped_additive'
    if attack=='ranking':s['decision']['selected_arm']='rebind_grouped'
    if attack=='gates':s['decision']['candidates'][0]['per_seed'][0]['checks']['ab_pair']=False
    if attack=='fraction':s['decision']['candidates'][0]['per_seed'][0]['c_real']=.99
    if attack=='protocol':p.write_text('changed')
    if attack=='evidence_duplicate':s['development_artifacts'][1]['run_dir']=s['development_artifacts'][0]['run_dir']
    if attack=='evidence_missing':s['development_artifacts'].pop()
    if attack=='evidence_hash':s['development_artifacts'][0]['artifacts']['manifest.json']='bad'
    if attack=='barrier':s['development_barrier']['observed']=44
    if attack=='preflight':s['teacher_preflight']['correct']=127
    if attack=='unsealed':s['sealed']=False
    if attack=='mode':next(e for e in s['development_artifacts'] if e['kind']=='evaluation')['part']='confirmation'
    with pytest.raises(ValueError):audit.validate_selection(s,p)


def test_confirmation_child_prevents_sealing_including_pending_teacher(tmp_path):
    report,_,p=selection_fixture(tmp_path)
    report['pending']=[dict(stage='teacher',part='confirmation')]
    with pytest.raises(ValueError,match='confirmation child'):audit.make_selection(report,p)


def test_final_no_reselection_and_teacher_qualification_required(tmp_path):
    report,selection,p=selection_fixture(tmp_path)
    for arm in audit.ARMS:
        for seed in audit.SEEDS:
            for split in ('known','confirm'):
                g=dict(count=16,update_restore_em=1.,updated_em=1.,shuffle_em=0.,empty_em=0.,locality_joint=1.)
                report['records'].append(dict(kind='evaluation',arm=arm,seed=seed,part='confirmation',split=split,
                    run_dir=f'confirm/{arm}/{seed}/{split}',groups={'CC_D':g},teacher_qualification=dict(paired={'CC_D':dict(count=16,both_correct=16)})))
    result=audit.final_gate(report,selection,p)
    assert result['eligible_to_expand']
    selected=next(r for r in report['records'] if r.get('arm')=='static_flat' and r.get('part')=='confirmation')
    selected['teacher_qualification']['paired']['CC_D']['both_correct']=15
    assert not audit.final_gate(report,selection,p)['eligible_to_expand']
    selected['teacher_qualification']['paired']['CC_D']['both_correct']=16
    selected['groups']['CC_D']['update_restore_em']=11/16
    assert not audit.final_gate(report,selection,p)['eligible_to_expand']
    report['records'][0]['artifacts']['manifest.json']='f'*64
    with pytest.raises(ValueError,match='artifacts changed'):audit.final_gate(report,selection,p)


def test_discovery_ignores_nonqkv_and_source_copies_and_smoke(tmp_path):
    for name in ('qkv_formal/child','other/child','qkv_formal/source/child','qkv_formal/smoke'):
        p=tmp_path/name;p.mkdir(parents=True);(p/'manifest.json').write_text('{}')
    paths=audit.discover(tmp_path)
    assert len(paths)==2
    assert sum(audit.ignored_reason(p,tmp_path) is not None for p in paths)==1


def test_source_contract_reads_immutable_snapshot_not_live_code(tmp_path):
    d=tmp_path/'suite'/'child';d.mkdir(parents=True);source=d.parent/'source'/'src'/'vera_mem';source.mkdir(parents=True)
    hashes={}
    for name in ('qkv_run.py','qkv_eval.py','qkv_vera.py','qkv_data.py'):
        (source/name).write_text(name);hashes[name]=audit.sha(source/name)
    assert audit.audit_sources(d,{'source_sha256':hashes})['source_sha256']==hashes
    (source/'qkv_data.py').write_text('changed')
    with pytest.raises(ValueError,match='Frozen source differs'):audit.audit_sources(d,{'source_sha256':hashes})


def registered_fixture(tmp_path):
    d=tmp_path/'qkv_train'/'model';d.mkdir(parents=True)
    docs=d.parent/'source'/'docs';docs.mkdir(parents=True);(docs/'qkv_protocol.md').write_text('protocol')
    config=dict(stage='train',cache='/remote/train.pt',seed=81042,read_mode='flat')
    plan=[dict(name='model',entry='qkv',arguments=['--stage','train','--cache','/remote/train.pt','--seed','81042','--read-mode','flat'])]
    (d.parent/'plan.json').write_text(json.dumps(plan));ph=audit.sha(d.parent/'plan.json')
    (d.parent/'suite.json').write_text(json.dumps({'plan_sha256':ph}))
    sources={'qkv_run.py':'a'*64,'qkv_eval.py':'b'*64}
    manifest=dict(stage='train',configuration=config,source_sha256=sources,cache_sha256='c'*64)
    registration=dict(source_python_sha256={'src/vera_mem/'+k:v for k,v in sources.items()},protocol_document_sha256=audit.sha(docs/'qkv_protocol.md'),
        feature_result={'caches':{'train':{'cache_sha256':'c'*64}}},plans_sha256={'train.json':ph})
    return d,manifest,registration


@pytest.mark.parametrize('attack',[None,'source','cache','plan','option','protocol','suite'])
def test_formal_registration_binds_runtime_plan_explicit_arguments_and_data(tmp_path,attack):
    d,m,r=registered_fixture(tmp_path)
    if attack=='source':m['source_sha256']['qkv_eval.py']='e'*64
    if attack=='cache':m['cache_sha256']='e'*64
    if attack=='plan':(d.parent/'plan.json').write_text('[]')
    if attack=='option':m['configuration']['seed']=81043
    if attack=='protocol':(d.parent/'source/docs/qkv_protocol.md').write_text('changed')
    if attack=='suite':(d.parent/'suite.json').write_text(json.dumps({'plan_sha256':'f'*64}))
    if attack:
        with pytest.raises(ValueError):audit.registered_run(d,m,r)
    else:assert audit.registered_run(d,m,r)['plan_sha256']==r['plans_sha256']['train.json']


def test_registration_loader_checks_model_matrix_protocol_and_44_source_inventory(tmp_path):
    p=tmp_path/'protocol.md';p.write_text('sealed')
    registration=dict(protocol='qkv-registration-v1',teacher_preflight_passed=True,smoke_passed=True,
        protocol_document_sha256=audit.sha(p),arms=list(audit.ARMS),seeds=list(audit.SEEDS),updates=2048,
        model_revision='cdbee75f17c01a7cc42f958dc650907174af0554',
        source_python_sha256={f'src/vera_mem/file{i}.py':audit.digest(i) for i in range(44)},plans_sha256={'plan.json':'a'*64})
    path=tmp_path/'registration.json';path.write_text(json.dumps(registration))
    assert audit.load_registration(path,p)==registration
    registration['source_python_sha256'].pop('src/vera_mem/file0.py');path.write_text(json.dumps(registration))
    with pytest.raises(ValueError,match='44 runtime'):audit.load_registration(path,p)


def test_seal_cli_is_exclusive_and_cannot_overwrite_existing_evidence(tmp_path,monkeypatch):
    report=dict(records=development_records(),pending=[],teacher_scope={'complete':False},partial=True)
    monkeypatch.setattr(audit,'summarize',lambda *args:copy.deepcopy(report))
    output=tmp_path/'summary.json';selection=tmp_path/'selection.json'
    args=['--runs-root',str(tmp_path),'--output',str(output),'--select','--selection-output',str(selection)]
    audit.main(args);before=selection.read_bytes()
    with pytest.raises(FileExistsError):audit.main(args)
    assert selection.read_bytes()==before
