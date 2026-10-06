"""Independent CPU replay of completed, locally backed QKV experiment artifacts.

No SSH/GPU/model generation. --training-only never opens evaluation artifacts.
The full audit verifies declared local suites; the paired training matrix is
separately required to contain the fixed 15 formal training conditions.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
import torch
from torch.nn import functional as F
from vera_mem.qkv_vera import QKVVeRA,QKVVectorDB
from vera_mem.qkv_data import episode,validate_dataset,validate_training_data,digest,PROTOCOL as DATA_PROTOCOL
from vera_mem.context_distillation_run import REVISION
from vera_mem.reconstruction_data import WRITER_PREFIX
from vera_mem.metrics import normalize_answer
from audit_reconstruction import Inputs,require,same,state_digest,file_hash,read_json,read_jsonl
from replay_coldstart_banks import analysis_exclusions
from summarize_qkv import load_registration,registered_run,validate_selection,development_selection

PROTOCOL='qkv-attention-memory-v1'
AUDIT_PROTOCOL='qkv-independent-artifact-audit-v1'
SEEDS=(81042,81043,81044)
ARMS={('static','flat','vera'),('static','grouped','vera'),('rebind','flat','vera'),
      ('rebind','grouped','vera'),('rebind','grouped','additive')}
FIELDS={'A':'a','B':'b','C':'c','D':'d','SWAP':'swap_a'}
PHASES={'CC':(0,0),'HC':(1,0),'CH':(0,1),'HH':(1,1)}
METRICS=('a_em','updated_em','pair_em','restored_em','update_restore_em','all_three_em',
         'locality_equal','locality_correct','locality_joint','shuffle_em','empty_em')


def close(left,right,message):
    require(math.isfinite(float(left)) and math.isfinite(float(right))
            and math.isclose(left,right,rel_tol=1e-6,abs_tol=1e-7),message)


def axes(split,part):
    if part=='preflight' and split=='train':return {'CC':['B']}
    if part=='smoke' and split=='known':return {'CC':['B']}
    if part=='development' and split=='known':return {'CC':['B','C','SWAP']}
    if part=='development' and split=='dev':return {'CC':['B','C']}
    if part=='confirmation' and split=='known':return {'CC':['D']}
    if part=='confirmation' and split=='confirm':return {p:['B','D'] for p in PHASES}
    raise ValueError('Unrecognized evaluation grid')


def check_packet(packet):
    require(packet['protocol']==PROTOCOL and packet['model_revision']==REVISION
            and packet['writer_prefix']==WRITER_PREFIX and packet['rows_sha256']==digest(packet['rows']),
            'Cache protocol/revision/prefix/row digest differs')
    rows=packet['rows'];require(len({r['id'] for r in rows})==len(rows),'Duplicate packet IDs')
    if packet['split']=='train':
        validate_training_data(dict(protocol=DATA_PROTOCOL,seed=packet['data_seed'],entities=packet['entities'],
                                    payloads=packet['payloads'],rows=rows))
        require(packet['worlds']==['A','B'] and packet['views']==1 and packet['q'].shape==(64,9728)
                and packet['matrix'].shape==(64,128,3,9728),'Training cache shape/scope differs')
        tensors=(packet['q'],packet['matrix']);observations=64*128
    else:
        require(packet['split'] in ('known','dev','confirm') and len(rows)==16 and packet['worlds']==list(FIELDS)
                and packet['views']==2 and packet['q'].shape==(16,2,9728)
                and packet['slots3'].shape==(16,5,2,3,9728),'Evaluation feature axes differ')
        tensors=(packet['q'],packet['slots3']);observations=16*5*2
    require(all(t.is_floating_point() and not t.requires_grad and bool(torch.isfinite(t).all()) for t in tensors),
            'Features must be finite detached floating tensors')
    counts=packet['payload_token_counts']
    require(isinstance(counts,list) and len(counts)==observations and all(isinstance(c,list) and len(c)==3
            and all(type(n)is int and n>0 for n in c) for c in counts),'Payload word-token counts differ')
    return packet


def load_packet(inputs,path,expected):
    p=inputs.resolve(path);packet=inputs.load(str(p),expected)
    checked=getattr(inputs,'checked_qkv_packets',set())
    if p not in checked:check_packet(packet);checked.add(p);inputs.checked_qkv_packets=checked
    return packet


def source_check(directory,manifest,inputs):
    sources=manifest['source_sha256'];frozen=directory.parent/'source/src/vera_mem'
    required={'qkv_vera.py','qkv_run.py','qkv_eval.py','qkv_data.py'}
    require(required<=set(sources),'Missing QKV source binding')
    for name,expected in sources.items():
        require(Path(name).name==name and name.endswith('.py'),'Unsafe source filename')
        path=frozen/name if frozen.is_dir() else REPO/'src/vera_mem'/name
        require(inputs.sha(path)==expected,'Frozen source SHA differs: '+name)
    replay=('qkv_vera.py','qkv_data.py','reconstruction_vera.py','interface_variants.py',
            'counterfactual_backend.py','stable_vector_vera.py','vector_vera.py','vector_store.py')
    for name in replay:
        require(file_hash(REPO/'src/vera_mem'/name)==sources[name],'Review changed CPU replay dependency: '+name)
    return dict(bound_source_files=len(sources),current_replay_dependencies=list(replay),frozen_snapshot=frozen.is_dir())


def validate_configuration(argv,cfg,run_dir):
    expected=dict(seed=81042,data_seed=121042,regime='static',read_mode='flat',readout='vera',updates=2048,
                  eval_part='development',cache=None,checkpoint=None)
    seen=set();i=0
    while i<len(argv):
        require(argv[i].startswith('--') and i+1<len(argv),'Malformed plan arguments')
        name=argv[i][2:].replace('-','_');require(name in set(expected)|{'stage','model'} and name not in seen,'Unknown/duplicate flag')
        value=argv[i+1];seen.add(name);i+=2
        expected[name]=int(value) if name in ('seed','data_seed','updates') else value
    require({'stage','model'}<=seen and all(cfg.get(k)==v for k,v in expected.items()) and cfg['run_dir']==run_dir,
            'Planned/default and child configurations differ')


def token_costs(raw,result):
    for r in raw:
        require(type(r['generation_tokens'])is int and 1<=r['generation_tokens']<=32,'Invalid generation count')
        ids=r['generated_token_ids']
        require(isinstance(ids,list) and len(ids)==r['generation_tokens'] and all(type(i)is int and i>=0 for i in ids),
                'Invalid generated token ID ledger')
        require(math.isfinite(r['generation_seconds']) and r['generation_seconds']>=0,'Invalid generation time')
    require(result['generation_calls']==len(raw) and result['generation_tokens']==sum(r['generation_tokens'] for r in raw),'Generation totals differ')
    close(result['generation_seconds'],sum(r['generation_seconds'] for r in raw),'Generation seconds differ')
    if 'answer_scoring_tokens' in result:
        require(result['answer_scoring_tokens']==sum(r['answer_tokens'] for r in raw),'Scoring token sum differs')


def audit_teacher(directory,manifest,packet):
    raw=read_jsonl(directory/'predictions.jsonl');result=read_json(directory/'summary.json')
    require(result==manifest['result'] and result['complete'],'Teacher summary differs')
    part=manifest['configuration']['eval_part'];indexes=range(min(2,len(packet['rows'])) if part=='smoke' else len(packet['rows']))
    expected=[(p,w,i) for p,ws in axes(packet['split'],part).items() for w in ['A',*[w for w in ws if w!='SWAP']] for i in indexes]
    require(len(raw)==len(expected),'Teacher grid count differs');groups=defaultdict(list)
    for r,(phase,world,i) in zip(raw,expected):
        fact=packet['rows'][i];sv,qv=PHASES[phase]
        require((r['id'],r['phase'],r['world'])==(fact['id'],phase,world) and r['question']==fact['questions'][qv]
                and r['answer']==fact[FIELDS[world]] and r['context']==fact['supports'][packet['worlds'].index(world)][sv],
                'Teacher text/source identity differs')
        require(r['teacher_bypasses_memory'] is True and r['memory_access']==dict(mode='none',store_present=False,
                trace=[],last_retrieval_present=False),'Teacher recorded memory access')
        em=int(normalize_answer(r['prediction'])==normalize_answer(r['answer']));require(r['em']==em,'Teacher EM differs')
        groups[phase+'_'+world].append(em)
    computed={k:dict(count=len(v),correct=sum(v),em=sum(v)/len(v)) for k,v in groups.items()}
    require(result['groups']==computed and result['qualified']==all(all(v) for v in groups.values()),'Teacher qualification differs')
    token_costs(raw,result)
    return dict(groups=computed,generation_calls=len(raw),zero_recorded_memory_access=True)


def validate_read(row,db,fact,world,phase,condition,role,case_id,trigger):
    require((row['id'],row['world'],row['phase'],row['condition'],row['role'],row['case_id'],row['trigger_world'])
            ==(fact['id'],world,phase,condition,role,case_id,trigger),'Read identity/order differs')
    require(row['question']==fact['questions'][PHASES[phase][1]] and row['answer']==fact[FIELDS[world]] and not row.get('context'),
            'Student prompt/answer/source differs')
    require(row['bank_hash']==db.hash() and row['bank_slots']==len(db),'Read references another bank')
    require(row['em']==int(normalize_answer(row['prediction'])==normalize_answer(row['answer']))
            and row['budget_hit']==(row['generation_tokens']>=32),'Read EM/budget differs')
    require(type(row['answer_tokens'])is int and row['answer_tokens']>0 and math.isfinite(row['nll_sum']) and row['nll_sum']>=-1e-6,'Invalid teacher-forced score')
    target=db.fact_ids.index(fact['id']) if fact['id'] in db.fact_ids else -1
    trace=row['trace'];require(len(trace)==row['generation_tokens']==len(row['generated_token_ids']),'Trace/token count differs')
    hits=[];slot_masses=[]
    for position,(event,token) in enumerate(zip(trace,row['generated_token_ids'])):
        require(event['phase']==('prefill' if position==0 else 'decode') and event['token_id']==token,'Token/route position differs')
        width=0 if not len(db) else 3 if db.read_mode=='grouped' else min(4,len(db))
        ii=event['indices'];require(isinstance(ii,list) and len(ii)==1 and len(ii[0])==width,'Wrong read width')
        indices=ii[0];require(len(set(indices))==width and all(type(i)is int and 0<=i<len(db) for i in indices),'Invalid route slot IDs')
        scores=torch.tensor(event['scores']);weights=torch.tensor(event['weights'])
        require(scores.shape==weights.shape==(1,width) and bool(torch.isfinite(scores).all()) and bool(torch.isfinite(weights).all()),'Invalid route score/weight tensors')
        if width:
            require(bool(((scores>=-1.00001)&(scores<=1.00001)).all()),'Invalid cosine scores')
            require(torch.allclose(weights,torch.softmax(scores/db.temperature,-1),rtol=1e-5,atol=1e-6),'Scores/softmax weights differ')
            if db.read_mode=='grouped':
                require(indices==[3*(indices[0]//3)+s for s in range(3)],'Grouped read is not one canonical complete fact')
            else:require(bool((scores[:,:-1]>=scores[:,1:]).all()),'Flat selected scores not descending')
        ww=event['weights'][0];masses=[sum(w for i,w in zip(indices,ww) if target>=0 and i==3*target+s) for s in range(3)]
        require(len(event['target_slot_mass'])==3,'Missing target-slot masses')
        for got,want in zip(event['target_slot_mass'],masses):close(got,want,'Target slot mass differs')
        close(event['target_fact_mass'],sum(masses),'Target fact mass differs');slot_masses.append(masses)
        hits.append(int(target>=0 and any(i//3==target for i in indices)))
    first=trace[0];indices=first['indices'][0];weights=first['weights'][0]
    r1=int(bool(indices) and target>=0 and indices[max(range(len(indices)),key=weights.__getitem__)]//3==target)
    require(row['first_fact_recall_at_read']==hits[0] and row['first_fact_recall_at_1']==r1
            and row['decode_hits']==sum(hits[1:]) and row['decode_queries']==len(hits)-1
            and row['actual_read_slots']==len(indices),'Recorded routing counters differ')
    require(len(row['first_target_slot_mass'])==3,'Missing first target-slot masses')
    for got,want in zip(row['first_target_slot_mass'],slot_masses[0]):close(got,want,'First target slot mass differs')
    close(row['first_target_fact_mass'],sum(slot_masses[0]),'First target fact mass differs')
    if world in ('C','D') and condition=='real' and role=='updated':
        edited=fact['provenance']['edited_word_index'];require(type(edited)is int and 0<=edited<3,'Invalid edited slot')
        return dict(edited_slot=edited,first_edited_slot_mass=slot_masses[0][edited],
                    all_generation_edited_slot_mass=sum(m[edited] for m in slot_masses),generation_queries=len(trace))
    return None


def checkpoint_parent(directory,manifest,inputs):
    path=inputs.resolve(manifest['configuration']['checkpoint']);checkpoint=inputs.load(str(path),manifest['checkpoint_sha256'])
    parent_path=path.parent/'manifest.json';require(not analysis_exclusions(path.parent,inputs.root),'Excluded training parent')
    parent=read_json(parent_path);inputs.sha(parent_path);cfg=parent['configuration']
    require(parent['complete'] is True and parent['stage']=='train' and parent['protocol']==PROTOCOL
            and checkpoint['protocol']==PROTOCOL and path.name=='last.pt' and checkpoint['step']==cfg['updates'],
            'Evaluation must load a completed final QKV checkpoint')
    require(checkpoint['cache_sha256']==parent['cache_sha256'] and checkpoint['seed']==cfg['seed']
            and checkpoint['regime']==cfg['regime'],'Parent checkpoint/cache identity differs')
    packet=load_packet(inputs,cfg['cache'],parent['cache_sha256']);require(packet['split']=='train','Parent used nontraining cache')
    arch=checkpoint['architecture']
    require(arch['read_mode']==cfg['read_mode'] and arch['readout']==cfg['readout'] and arch['seed']==cfg['seed'],'Parent architecture differs')
    return checkpoint,dict(checkpoint_sha256=inputs.sha(path),parent_manifest_sha256=inputs.sha(parent_path),
                           parent_train=str(path.parent),regime=cfg['regime'],seed=cfg['seed'])


def audit_eval(directory,manifest,packet,inputs):
    checkpoint,lineage=checkpoint_parent(directory,manifest,inputs)
    model=QKVVeRA(**checkpoint['architecture']);model.load_state_dict(checkpoint['module']);model.eval().requires_grad_(False)
    result=read_json(directory/'summary.json');raw=read_jsonl(directory/'predictions.jsonl');saved_pairs=read_json(directory/'pairs.json')
    require(result==manifest['result'] and result['protocol']=='qkv-online-cpu-v1' and result['complete']
            and result['architecture']==checkpoint['architecture'] and result['shared_parameters_unchanged'] is True
            and result['shared_parameter_sha256']==state_digest(checkpoint['module']),'Online model/summary identity differs')
    encoded=inputs.load(str(directory/'encoded_payloads.pt'));banks=inputs.load(str(directory/'bank_events.pt'))
    rows=packet['rows'];n=len(rows)
    require(encoded['fact_ids']==[r['id'] for r in rows] and encoded['worlds']==packet['worlds']
            and banks['protocol']=='qkv-bank-events-v1','Saved bank feature identity differs')
    errors={}
    for name,reference in [('keys',model.encode_key(packet['slots3'].float())),('values',model.encode_value(packet['slots3'].float()))]:
        actual=encoded[name];require(actual.shape==reference.shape and actual.dtype==torch.float32
                and bool(torch.isfinite(actual).all()) and torch.allclose(actual,reference,rtol=2e-5,atol=2e-5),'Writer re-encoding differs')
        errors[name]=float((actual-reference).abs().max())
    part=manifest['configuration']['eval_part'];grid=axes(packet['split'],part);indexes=list(range(min(2,n) if part=='smoke' else n))
    require(set(banks['bases'])==set(grid),'Missing/extra phase base banks')
    expected_count=sum(len(v)*len(indexes) for v in grid.values())
    require(len(banks['events'])==len(saved_pairs)==expected_count,'Bank event/pair count differs')
    cursor=event_cursor=0;pairs=[];edit_routes=[];base_hashes={};shuffles=0
    def consume(db,fact,world,phase,condition,role,case_id,trigger):
        nonlocal cursor
        require(cursor<len(raw),'Missing raw generation');r=raw[cursor];cursor+=1
        diagnostic=validate_read(r,db,fact,world,phase,condition,role,case_id,trigger)
        if diagnostic is not None:edit_routes.append(diagnostic)
        return r
    for phase,worlds in grid.items():
        sv,_=PHASES[phase];base=QKVVectorDB.from_snapshot(banks['bases'][phase]);expected=model.new_store()
        for i,f in enumerate(rows):expected.write_group(f['id'],encoded['keys'][i,0,sv],encoded['values'][i,0,sv],0)
        require(base.hash()==expected.hash(),'A snapshot differs from source payloads');base_hashes[phase]=base.hash()
        needed=set(indexes)|{(i+1)%n for i in indexes}
        initial={i:consume(base,rows[i],'A',phase,'real','initial',rows[i]['id'],'A') for i in sorted(needed)}
        for world in worlds:
            wi=packet['worlds'].index(world)
            for i in indexes:
                f=rows[i];event=banks['events'][event_cursor];event_cursor+=1
                require((event['phase'],event['world'],event['index'],event['id'],event['timestamp'])==(phase,world,i,f['id'],1)
                        and event['before_hash']==base.hash(),'Intervention identity differs')
                require(same(event['keys'],encoded['keys'][i,wi,sv]) and same(event['values'],encoded['values'][i,wi,sv]),'Patch differs from actual content writer')
                changed=QKVVectorDB.from_snapshot(base.snapshot());changed.write_group(f['id'],event['keys'],event['values'],1)
                require(changed.hash()==event['after_hash'],'Patch replay hash differs')
                mask=torch.ones(len(base),dtype=torch.bool);mask[3*i:3*i+3]=False
                require(same(changed.keys[mask],base.keys[mask]) and same(changed.values[mask],base.values[mask])
                        and changed.fact_ids==base.fact_ids and changed.timestamps[3*i:3*i+3]==(1,)*3
                        and all(a==b for j,(a,b) in enumerate(zip(base.timestamps,changed.timestamps)) if bool(mask[j])),
                        'Patch changed another fact or only part of its own group')
                updated=consume(changed,f,world,phase,'real','updated',f['id'],world);neighbor=(i+1)%n
                local=consume(changed,rows[neighbor],'A',phase,'real','locality',f['id'],world)
                restored=QKVVectorDB.from_snapshot(changed.snapshot());restored.write_group(f['id'],encoded['keys'][i,0,sv],encoded['values'][i,0,sv],2)
                require(restored.hash()==event['restored_hash'] and same(restored.keys,base.keys) and same(restored.values,base.values)
                        and restored.timestamps[3*i:3*i+3]==(2,)*3,'Restoration differs')
                rr=consume(restored,f,'A',phase,'real','restored',f['id'],world)
                pair=dict(id=f['id'],phase=phase,world=world,a_em=initial[i]['em'],updated_em=updated['em'],
                    pair_em=initial[i]['em']*updated['em'],restored_em=rr['em'],update_restore_em=updated['em']*rr['em'],
                    all_three_em=initial[i]['em']*updated['em']*rr['em'],locality_equal=int(local['prediction']==initial[neighbor]['prediction']),
                    locality_correct=local['em'],locality_joint=local['em']*initial[neighbor]['em'],shuffle_em=None,empty_em=None)
                if world!='SWAP':
                    order=list(range(n));random.Random(131042+i).shuffle(order);perm=list(range(n))
                    for a,b in zip(order,order[1:]+order[:1]):perm[a]=b
                    require(event['value_permutation']==perm and all(i!=j for i,j in enumerate(perm)),'Shuffle is not declared group derangement')
                    state=changed.snapshot();state['store']['values']=state['store']['values'].reshape(n,3,model.rank)[perm].reshape(-1,model.rank).clone()
                    wrong=QKVVectorDB.from_snapshot(state);require(wrong.hash()==event['shuffle_hash'],'Shuffle replay hash differs')
                    for condition,db in [('shuffle',wrong),('empty',model.new_store())]:
                        pair[condition+'_em']=consume(db,f,world,phase,condition,'control',f['id'],world)['em']
                    shuffles+=1
                require(base.hash()==base_hashes[phase],'Independent intervention contaminated A bank');pairs.append(pair)
    require(cursor==len(raw) and pairs==saved_pairs,'Unexpected raw records or pair arithmetic differs')
    grouped=defaultdict(list)
    for p in pairs:grouped[p['phase']+'_'+p['world']].append(p)
    groups={}
    for k,ps in grouped.items():
        group=dict(count=len(ps))
        for name in METRICS:
            values=[p[name] for p in ps if p[name] is not None];group[name]=sum(values)/len(values) if values else None
        groups[k]=group
    require(groups==result['groups'] and result['bank_facts']==n and result['slots_per_fact']==3
            and result['resident_bytes']==base.resident_bytes() and result['independent_interventions']==expected_count
            and result['restorations']==expected_count and result['real_group_write_events']==2*expected_count,'Eval summary differs')
    token_costs(raw,result)
    return dict(groups=groups,generation_calls=len(raw),independent_interventions=expected_count,restorations=expected_count,
        real_group_write_events=2*expected_count,shuffled_group_controls=shuffles,writer_max_abs_errors=errors,
        cached_writer_reencoded=True,token_route_masses_verified=True,recorded_routes_verified=True,
        edited_slot_diagnostics=dict(updated_predictions=len(edit_routes),position_counts=dict(Counter(r['edited_slot'] for r in edit_routes)),
            first_mass_sum=sum(r['first_edited_slot_mass'] for r in edit_routes),
            all_generation_mass_sum=sum(r['all_generation_edited_slot_mass'] for r in edit_routes),
            generation_queries=sum(r['generation_queries'] for r in edit_routes)),
        lineage=lineage,read_mode=model.read_mode,readout=model.readout,online_state_sha256=result['shared_parameter_sha256'])


def training_schema(directory,manifest,smoke):
    frozen=directory.parent/'source/src/vera_mem/qkv_run.py'
    path=frozen if frozen.is_file() else REPO/'src/vera_mem/qkv_run.py'
    require(file_hash(path)==manifest['source_sha256']['qkv_run.py'],'Training diagnostic source changed')
    train=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train')
    fields={n.arg for n in ast.walk(train) if isinstance(n,ast.keyword)}
    require(('input_lengths' in fields)==('gold_token_lengths' in fields),'Partial source token ledger')
    result=dict(token_ledger='input_lengths' in fields,read_diagnostics='read_diagnostics' in fields,
                source_sha256=file_hash(path))
    require(smoke or (result['token_ledger'] and result['read_diagnostics']),'Formal source lacks declared training diagnostics')
    return result


def audit_training_log(raw,status,cfg,smoke=False,schema=None):
    schema=schema or dict(token_ledger=True,read_diagnostics=True)
    require(len(raw)==cfg['updates'] and status['updates']==len(raw) and status['complete'],'Training update budget differs')
    schedule=hashlib.sha256();targets_hash=hashlib.sha256();epochs=defaultdict(lambda:dict(entities=Counter(),payloads=Counter()))
    parameter_names={'Wq.weight','Wk.weight','Wv.weight','b','B','slot_position'}
    nonzero=Counter();max_norm=Counter();ledger=dict(target_exposures=0,gold_tokens=0,input_positions=0,padded_input_positions=0,backbone_calls=0)
    ledger_steps=0;diagnostics=[]
    for step,row in enumerate(raw):
        require(row['step']==step+1,'Training step order differs');ep=episode(step,cfg['regime'],cfg['seed'])
        require(row['episode']==ep,'Recorded global/local/bank/payload episode differs')
        schedule.update(json.dumps(ep,sort_keys=True).encode())
        address={k:ep[k] for k in ('step','seed','epoch','batch_index','targets','background','bank_entities','local_targets')}
        targets_hash.update(json.dumps(address,sort_keys=True).encode())
        epoch=epochs[ep['epoch']];epoch['entities'].update(ep['targets'])
        epoch['payloads'].update(ep[field][i] for field in ('a_payload_indices','b_payload_indices') for i in ep['local_targets'])
        m=row['metrics'];close(m['loss'],m['ce']+.2*m['address'],'Training objective arithmetic differs')
        require(all(math.isfinite(m[k]) and m[k]>=-1e-6 for k in ('ce','address','loss')),'Nonfinite/negative loss')
        grads=row['parameter_gradient_norms'];require(set(grads)==parameter_names,'Missing/extra gradient telemetry')
        for name,value in grads.items():
            require(value is not None and math.isfinite(value) and value>=0,'Invalid parameter gradient norm')
            nonzero[name]+=int(value>0);max_norm[name]=max(max_norm[name],value)
        require(len(row['gradient_norms'])==4 and all(math.isfinite(v) and v>=0 for v in row['gradient_norms']),'Invalid clipped group norm ledger')
        require(('input_lengths' in row)==('gold_token_lengths' in row)==schema['token_ledger'],'Token ledger differs from bound source')
        if schema['token_ledger']:
            lengths=row['input_lengths'];gold=row['gold_token_lengths']
            require(len(lengths)==len(gold)==16 and all(type(a)is int and type(b)is int and 0<b<a for a,b in zip(lengths,gold)),
                    'Invalid 16-sequence token ledger')
            ledger_steps+=1;ledger['gold_tokens']+=sum(gold);ledger['input_positions']+=sum(lengths)
            ledger['padded_input_positions']+=16*max(lengths)
        require(('read_diagnostics' in row)==schema['read_diagnostics'],'Read diagnostic schema differs from bound source')
        if schema['read_diagnostics']:
            d=row['read_diagnostics'];expected=(step==0 or (step+1)%128==0)
            require((d is not None)==expected,'Read diagnostic schedule differs')
            if d is not None:
                require(d['scope']=='captured gold-prefix prediction positions including EOS; parameter state before update'
                        and type(d['prediction_positions'])is int and d['prediction_positions']>0
                        and (not schema['token_ledger'] or d['prediction_positions']==sum(gold)),'Read diagnostic position/scope differs')
                require(all(math.isfinite(d[k]) and d[k]>=0 for k in ('input_rms','mixed_value_rms','residual_rms','target_fact_mass'))
                        and d['target_fact_mass']<=1.00001 and d['actual_read_slots']==(4 if cfg['read_mode']=='flat' else 3),
                        'Invalid actual-query read diagnostic')
                if step==0:require(d['residual_rms']==0,'Zero-b initialization must be an exact initial no-op')
                diagnostics.append(dict(step=step+1,**d))
        ledger['target_exposures']+=8;ledger['backbone_calls']+=1
    require(status['schedule_sha256']==schedule.hexdigest(),'Training schedule digest differs')
    require(status['totals']['target_exposures']==8*len(raw) and status['totals']['backbone_calls']==len(raw),'Training count totals differ')
    require(ledger_steps in (0,len(raw)),'Partial token ledger')
    if ledger_steps:require(ledger==status['totals'],'Token/input ledger totals differ')
    completed=0
    for epoch,counts in epochs.items():
        if sum(counts['entities'].values())==64:
            require(counts['entities']==Counter(range(64)) and counts['payloads']==Counter(range(128)),'Epoch exposures differ')
            completed+=1
        else:require(smoke and epoch==max(epochs),'Incomplete formal training epoch')
    if not smoke:
        require(len(raw)==2048 and completed==256 and all(nonzero[n]>0 for n in parameter_names),'Formal training budget or gradient coverage differs')
    return dict(schedule_sha256=schedule.hexdigest(),target_bank_schedule_sha256=targets_hash.hexdigest(),completed_epochs=completed,
        each_complete_epoch_entities_once_and_payloads_once=True,parameter_nonzero_gradient_steps=dict(nonzero),
        parameter_max_gradient_norm=dict(max_norm),token_totals=status['totals'],
        token_totals_independently_resummed=bool(ledger_steps),training_diagnostic_schema=schema,
        actual_query_read_diagnostics=diagnostics,
        gradient_scope='Logged parameter-level norm; nonzero is not a useful-gradient or per-coordinate coverage claim.')


def audit_train(directory,manifest,packet,inputs,smoke):
    cfg=manifest['configuration'];raw=read_jsonl(directory/'training.jsonl');status=read_json(directory/'training_status.json')
    require(packet['split']=='train' and status==manifest['result'],'Training packet/status differs')
    logproof=audit_training_log(raw,status,cfg,smoke,training_schema(directory,manifest,smoke))
    initial=inputs.load(str(directory/'initial.pt'));final=inputs.load(str(directory/'last.pt'))
    terminal_path=directory/f"step_{cfg['updates']}.pt";terminal=inputs.load(str(terminal_path))
    for c in (initial,final,terminal):
        require(c['protocol']==PROTOCOL and c['seed']==cfg['seed'] and c['regime']==cfg['regime']
                and c['cache_sha256']==manifest['cache_sha256'],'Checkpoint identity/cache differs')
    require(initial['step']==0 and final['step']==terminal['step']==cfg['updates'] and final['schedule_sha256']==terminal['schedule_sha256']==logproof['schedule_sha256'],'Checkpoint step/schedule differs')
    require(initial['architecture']==final['architecture']==terminal['architecture']==status['architecture'],'Architecture changed during training')
    arch=final['architecture'];require(arch['read_mode']==cfg['read_mode'] and arch['readout']==cfg['readout'] and arch['seed']==cfg['seed'],'Training factors differ')
    model=QKVVeRA(**arch);model.load_state_dict(initial['module']);params=dict(model.named_parameters())
    names=[['Wq.weight','Wk.weight','slot_position'],['Wv.weight'],['b'],['B']];rates=[1e-4,3e-4,.005,3e-4]
    require(set(params)=={n for g in names for n in g} and 'A' in dict(model.named_buffers())
            and status['trainable_parameters']==sum(p.numel() for p in params.values()) and status['learning_rates']==rates,'Trainable parameter contract differs')
    fixed=['A','query_center','support_center','value_center','statistics_fitted','value_statistics_fitted','interface_architecture','reconstruction_architecture','qkv_architecture']
    require(all(same(initial['module'][k],final['module'][k]) for k in fixed),'Frozen buffers changed')
    require(set(final['module'])==set(terminal['module']) and all(same(v,terminal['module'][k]) for k,v in final['module'].items()),'Terminal step and last parameters differ')
    # Refit only declared static A/B train observations, not the full crossproduct.
    probe=QKVVeRA(**arch);i=torch.arange(64);features=packet['matrix'][i[:,None],2*i[:,None]+torch.arange(2)].reshape(-1,9728).float()
    probe.fit_statistics(packet['q'].float(),features);probe.fit_value_statistics(features)
    stat_errors={}
    for name in ('query_center','support_center','value_center'):
        ref=getattr(probe,name);actual=initial['module'][name]
        require(torch.allclose(actual,ref,rtol=1e-5,atol=1e-6),'Initial train-only statistics differ')
        stat_errors[name]=float((actual-ref).abs().max())
    opt=terminal['optimizer'];require(len(opt['param_groups'])==4,'Adam group count differs');seen=[]
    for group,group_names,lr in zip(opt['param_groups'],names,rates):
        require(len(group['params'])==len(group_names) and group['lr']==lr and tuple(group['betas'])==(.9,.999)
                and group['eps']==1e-8 and group['weight_decay']==0 and not group['amsgrad'],'Adam group/hyperparameter mismatch')
        for pid,name in zip(group['params'],group_names):
            seen.append(pid);s=opt['state'][pid];require(float(s['step'])==cfg['updates'],'Adam step differs')
            require(all(s[k].shape==params[name].shape and s[k].dtype==params[name].dtype and bool(torch.isfinite(s[k]).all())
                        for k in ('exp_avg','exp_avg_sq')) and bool((s['exp_avg_sq']>=0).all()),'Adam moment shape/value differs')
    require(len(seen)==len(set(seen)) and set(seen)==set(opt['state']),'Unexpected/duplicate Adam parameter')
    return dict(logproof,training_configuration=cfg,updates=cfg['updates'],trainable_parameters=status['trainable_parameters'],
        same_terminal_optimizer_and_last_state=True,A_is_fixed_buffer_outside_optimizer=True,A_unused_for_additive=arch['readout']=='additive',
        statistics_recomputed_from_static_train_AB=True,statistics_max_abs_errors=stat_errors,
        initial_checkpoint_sha256=inputs.sha(directory/'initial.pt'),final_checkpoint_sha256=inputs.sha(directory/'last.pt'),
        terminal_optimizer_checkpoint_sha256=inputs.sha(terminal_path))


def training_matrix(jobs,inputs):
    groups=defaultdict(dict);cache=set()
    for job in jobs:
        if job['stage']!='train' or job['scope']!='formal':continue
        c=job['training_configuration'];arm=(c['regime'],c['read_mode'],c['readout']);seed=c['seed']
        require(seed in SEEDS and arm in ARMS and arm not in groups[seed],'Unexpected/duplicate formal training matrix arm')
        groups[seed][arm]=job
    reports=[];missing=[]
    for seed in SEEDS:
        arms=groups[seed];missing.extend(dict(seed=seed,arm=list(arm)) for arm in sorted(ARMS-set(arms)))
        if not arms:continue
        states={arm:inputs.load(str(Path(j['run_dir'])/'initial.pt')) for arm,j in arms.items()};reference=next(iter(states.values()))
        reference_job=next(iter(arms.values()));names=set(reference['module'])-{'qkv_architecture'}
        within={}
        for arm,c in states.items():
            cache.add(c['cache_sha256']);j=arms[arm]
            require({k:v for k,v in c['architecture'].items() if k not in ('read_mode','readout')}
                    =={k:v for k,v in reference['architecture'].items() if k not in ('read_mode','readout')},
                    'Same-seed common architecture differs')
            require(set(c['module'])-{'qkv_architecture'}==names and all(same(c['module'][n],reference['module'][n]) for n in names),'Same-seed initial parameters/statistics differ')
            require(j['target_bank_schedule_sha256']==reference_job['target_bank_schedule_sha256'],'Entity/background/order schedules differ across regimes')
            for key in ('target_exposures','gold_tokens','input_positions','backbone_calls'):
                require(j['token_totals'][key]==reference_job['token_totals'][key],'Cross-regime marginal token/input budget differs')
            if arm[0] in within:
                previous=within[arm[0]]
                require(previous['schedule_sha256']==j['schedule_sha256'] and previous['token_totals']==j['token_totals'],
                        'Within-regime reader/readout schedule or padding budget differs')
            within[arm[0]]=dict(schedule_sha256=j['schedule_sha256'],token_totals=j['token_totals'])
            require(j['trainable_parameters']==reference_job['trainable_parameters'],'Training parameter count differs')
        reports.append(dict(seed=seed,arms=len(arms),initial_tensors_except_qkv_metadata_bitwise_equal=True,
            initial_common_state_sha256=state_digest({n:reference['module'][n] for n in names}),
            target_bank_schedule_sha256=reference_job['target_bank_schedule_sha256'],within_regime=within,
            trainable_parameters=reference_job['trainable_parameters'],cross_regime_padding_equality_required=False))
    require(len(cache)<=1,'Formal training arms use different cache')
    return dict(complete=not missing,expected_conditions=15,observed_conditions=sum(map(len,groups.values())),missing=missing,
        seeds=reports,training_cache_sha256=next(iter(cache),None),
        scope='Four VeRA arms and one additive diagnostic per seed; same initial tensors except explicit read metadata.')


def audit_job(directory,manifest,inputs):
    require(manifest['protocol']==PROTOCOL and manifest['complete'] is True and manifest['backbone_unchanged'] is True,'Not a complete frozen-backbone job')
    source=source_check(directory,manifest,inputs);stage=manifest['stage'];smoke='smoke' in directory.name.lower()
    if stage=='prepare':
        data_path=directory/'dataset.json';require(inputs.sha(data_path)==manifest['result']['dataset_sha256'],'Dataset file SHA differs')
        data=read_json(data_path);proof=validate_dataset(data);caches={}
        for split,record in manifest['result']['caches'].items():
            packet=load_packet(inputs,str(directory/(split+'.pt')),record['cache_sha256'])
            rows=read_json(directory/(split+'.json'));inputs.sha(directory/(split+'.json'))
            require(packet['split']==split and rows==data[split] and packet['rows']==(rows['rows'] if split=='train' else rows),'Dataset/json/cache split binding differs')
            caches[split]=record['cache_sha256']
        require(set(caches)=={'train','known','dev','confirm'},'Missing prepared split')
        result=dict(data_proof=proof,cache_sha256=caches)
    else:
        packet=load_packet(inputs,manifest['configuration']['cache'],manifest['cache_sha256'])
        names=['manifest.json']+(['training.jsonl','training_status.json','initial.pt','last.pt',f"step_{manifest['configuration']['updates']}.pt"]
               if stage=='train' else ['predictions.jsonl','summary.json']+(['pairs.json','encoded_payloads.pt','bank_events.pt'] if stage=='eval' else []))
        before={name:file_hash(directory/name) for name in names}
        if stage=='train':result=audit_train(directory,manifest,packet,inputs,smoke)
        elif stage=='eval':result=audit_eval(directory,manifest,packet,inputs)
        elif stage=='teacher':result=audit_teacher(directory,manifest,packet)
        else:raise ValueError('Unknown QKV stage')
        require(before=={name:file_hash(directory/name) for name in names},'Raw input changed during audit')
        result.update(artifact_sha256=before,input_hashes_unchanged=True,cache_sha256=manifest['cache_sha256'])
    return dict(run_dir=str(directory),stage=stage,scope='smoke' if smoke else 'formal',source=source,complete=True,
                configuration=manifest['configuration'],cache_split=None if stage=='prepare' else packet['split'],**result)


def arm_name(regime,read_mode,readout):
    return regime+'_'+read_mode+('_additive' if readout=='additive' else '')


def model_identity(job):
    if job['stage']=='train':
        cfg=job['training_configuration']
        return arm_name(cfg['regime'],cfg['read_mode'],cfg['readout']),cfg['seed']
    lineage=job['lineage']
    return arm_name(lineage['regime'],job['read_mode'],job['readout']),lineage['seed']


def formal_inventory(jobs):
    model=Counter();teachers=Counter()
    for j in jobs:
        if j['scope']!='formal':continue
        stage=j['stage']
        if stage in ('train','eval'):
            arm,seed=model_identity(j)
            split='train' if stage=='train' else j['cache_split']
            part='train' if stage=='train' else j['configuration']['eval_part']
            model[arm,seed,split,part]+=1
        elif stage=='teacher':teachers[j['cache_split'],j['configuration']['eval_part']]+=1
    expected={(arm_name(*a),seed,split,part) for a in ARMS for seed in SEEDS
              for split,part in [('train','train'),('known','development'),('dev','development'),
                                 ('known','confirmation'),('confirm','confirmation')]}
    expected_teachers={('train','preflight'),('known','development'),('dev','development'),
                       ('known','confirmation'),('confirm','confirmation')}
    require(set(model)<=expected and all(v==1 for v in model.values()),'Unknown/duplicate formal model condition')
    require(set(teachers)<=expected_teachers and all(v==1 for v in teachers.values()),'Unknown/duplicate teacher condition')
    return dict(complete=set(model)==expected and set(teachers)==expected_teachers,
        expected_model_conditions=75,observed_model_conditions=len(model),
        missing_model_conditions=[list(k) for k in sorted(expected-set(model))],
        expected_teacher_conditions=5,observed_teacher_conditions=len(teachers),
        missing_teacher_conditions=[list(k) for k in sorted(expected_teachers-set(teachers))])


def audit_confirmation_barrier(inputs,path,suites,jobs):
    """Bind saved launch receipts to the original 45-condition development seal."""
    require(path is not None and Path(path).is_file(),'Confirmation requires original local selection evidence')
    selection=validate_selection(read_json(path),REPO/'docs/qkv_protocol.md');sealed_sha=inputs.sha(path)
    copies=[]
    for directory,suite in suites:
        copy_path=directory/'selection_evidence.json'
        require(suite.get('selection_evidence_sha256')==sealed_sha and inputs.sha(copy_path)==sealed_sha,
                'Confirmation selection snapshot differs from original sealed evidence')
        copies.append(dict(suite=str(directory),snapshot_sha256=sealed_sha))
    by_path={Path(j['run_dir']).resolve():j for j in jobs};records=[];checked=0
    for e in selection['development_artifacts']:
        resolved=inputs.resolve(str(Path(e['run_dir'])/'manifest.json')).parent
        require(resolved in by_path and by_path[resolved]['scope']=='formal','Sealed development evidence was not audited')
        j=by_path[resolved];expected_kind={'train':'training','eval':'evaluation','teacher':'teacher'}[j['stage']]
        require(e['kind']==expected_kind,'Sealed evidence stage differs')
        for name,expected in e['artifacts'].items():
            require(Path(name).name==name and inputs.sha(resolved/name)==expected,'Sealed development artifact changed: '+str(resolved/name))
            checked+=1
        r=dict(kind=expected_kind,groups=j.get('groups'))
        if j['stage'] in ('train','eval'):
            arm,seed=model_identity(j);require((e['arm'],e['seed'])==(arm,seed),'Sealed development arm/seed differs')
            r.update(arm=arm,seed=seed)
        if j['stage']=='train':r.update(trainable_parameters=j['trainable_parameters'])
        else:
            split,part=j['cache_split'],j['configuration']['eval_part']
            require((e['split'],e['part'])==(split,part),'Sealed development split/part differs')
            r.update(split=split,part=part)
            if j['stage']=='eval':r['bytes_per_fact']=3*(64+64)*4
        r['run_dir']=str(resolved);records.append(r)
    require(development_selection(records)==selection['decision'],'Sealed candidate decision differs from audited development evidence')
    return dict(selection_sha256=sealed_sha,confirmation_suites=copies,development_model_conditions=45,
        teacher_preflight_conditions=1,evidence_artifacts_rehashed=checked,
        sealed_evidence_unchanged=True,decision_recomputed_from_audited_development=True,
        selected_arm=selection['decision']['selected_arm'],
        limitation='Snapshot/hash and frozen launcher checks establish the recorded barrier; wall-clock order is not treated as trusted execution attestation.')


def run_audit(root,training_only=False,registration_path=None,selection_path=None,require_final=False):
    inputs=Inputs(root);results=[];pending=[];excluded=[];confirmation_suites=[]
    if registration_path is None:
        candidates=(REPO.parent/'plans/qkv_registration_20261006.json',REPO/'docs/results/qkv/registration.json')
        registration_path=next((p for p in candidates if p.is_file()),None)
    registration=load_registration(registration_path,REPO/'docs/qkv_protocol.md') if registration_path else None
    registration_record=None
    if registration is not None:
        registration_record=dict(path=str(Path(registration_path).resolve()),sha256=inputs.sha(registration_path),
            protocol_document_sha256=inputs.sha(REPO/'docs/qkv_protocol.md'),
            validation_helper_sha256=inputs.sha(REPO/'scripts/summarize_qkv.py'))
    for path in sorted(inputs.root.glob('qkv_*/suite.json')):
        suite_dir=path.parent;markers=analysis_exclusions(suite_dir,inputs.root)
        if markers:excluded.append(dict(path=str(suite_dir),markers=markers));continue
        suite=read_json(path);require(suite['protocol']==PROTOCOL,'Wrong suite protocol');inputs.sha(path)
        plan=read_json(suite_dir/'plan.json');inputs.sha(suite_dir/'plan.json')
        require(isinstance(plan,list) and plan and len({p['name'] for p in plan})==len(plan),'Invalid plan')
        if not training_only and any('--eval-part' in p['arguments'] and p['arguments'][p['arguments'].index('--eval-part')+1]=='confirmation' for p in plan):
            confirmation_suites.append((suite_dir,suite))
        jobs={j['name']:j for j in suite['jobs']};require(len(jobs)==len(suite['jobs']) and set(jobs)<={p['name'] for p in plan},'Unplanned/duplicate launched jobs')
        for spec in plan:
            require(spec['entry']=='qkv','Wrong plan entry');args=spec['arguments'];stage=args[args.index('--stage')+1]
            if training_only and stage!='train':continue
            directory=suite_dir/spec['name'];markers=analysis_exclusions(directory,inputs.root)
            if markers:excluded.append(dict(path=str(directory),markers=markers));continue
            j=jobs.get(spec['name'])
            if not j or j.get('status')!='complete' or j.get('exit_code')!=0:
                pending.append(dict(path=str(directory),reason='Launcher not complete; raw artifacts not read'));continue
            command=j['command'];start=command.index('vera_mem.qkv_run')+1
            require(command[start:-2]==args and command[-2]=='--run-dir','Launcher argv differs from plan')
            mp=directory/'manifest.json';require(inputs.sha(mp)==j['manifest_sha256'],'Child manifest SHA differs');m=read_json(mp)
            if not m.get('complete'):pending.append(dict(path=str(directory),reason='Child incomplete; raw artifacts not read'));continue
            validate_configuration(args,m['configuration'],command[-1]);require(m['stage']==stage,'Child/plan stage differs')
            smoke='smoke' in directory.name.lower()
            require(smoke or stage not in ('train','eval') or registration is not None,'Formal model run requires sealed registration')
            binding=registered_run(directory,m,registration) if registration is not None and not smoke else None
            result=audit_job(directory,m,inputs);result['registration_check']=binding;results.append(result)
            print(json.dumps(dict(audited=str(directory),stage=stage,scope=result['scope'])),flush=True)
        for relative,expected in suite['source_files_sha256'].items():
            require(inputs.sha(suite_dir/'source'/relative)==expected,'Suite source snapshot changed')
        if not training_only and not suite.get('complete'):pending.append(dict(path=str(suite_dir),reason='Suite incomplete'))
    matrix=training_matrix(results,inputs);scopes={};inventory=formal_inventory(results)
    selection_check=None
    if confirmation_suites:
        if selection_path is None:selection_path=REPO.parent/'plans/qkv_selection_20261006.json'
        selection_check=audit_confirmation_barrier(inputs,selection_path,confirmation_suites,results)
    if require_final:
        require(not training_only and inventory['complete'] and matrix['complete'] and not pending,
                'Final audit requires all 15 train, 60 eval and five teacher conditions complete')
        require(selection_check is not None and len(confirmation_suites)==4,'Final audit requires four sealed confirmation suites')
    for scope in ('formal','smoke'):
        selected=[r for r in results if r['scope']==scope]
        scopes[scope]=dict(jobs=len(selected),stages=dict(Counter(r['stage'] for r in selected)),
            **{k:sum(r.get(k,0) for r in selected) for k in ('generation_calls','independent_interventions','restorations','real_group_write_events')})
    return dict(protocol=AUDIT_PROTOCOL,audited_at=datetime.now(timezone.utc).isoformat(),checks_passed=True,
        complete=bool(results) and not pending and (not training_only or matrix['complete']),training_only=training_only,
        jobs=results,pending=pending,excluded=excluded,scopes=scopes,training_matrix=matrix,
        registration=registration_record,formal_inventory=inventory,confirmation_barrier=selection_check,
        final_scope_required=require_final,
        input_sha256={str(k):v for k,v in inputs.hashes.items()},cuda_initialized=torch.cuda.is_initialized(),
        limitations=['No LM generation, hidden-feature extraction or full-backbone gradient replay is performed.',
            'Selected scores/weights and fact/slot masses are verified; absent actual query vectors prevent proving global top-k or winning-group optimality.',
            'Generated IDs align one-to-one with recorded routes. Edited-slot mass is not aligned here to decoded target-word boundaries.',
            'Teacher and unchanged-state telemetry are source-bound records, not trusted-execution attestation.',
            'Gradient norms are finite logged parameter norms, not per-coordinate or useful-gradient guarantees.',
            'Episode binding and marginal exposures are independently reconstructed; cross-regime padded token counts can differ.',
            'Facts/payloads recur across seeds/arms; repeated observations are not independent samples.',
            'Additive is a separate readout diagnostic and cannot replace the main VeRA result; A remains an unused buffer there.',
            'Overall completeness covers declared local suites; the 15-training matrix has its own completeness flag.'])


def self_test():
    """Real CPU store/evaluator fixtures; generation alone is a fake decoder."""
    import contextlib
    import copy
    import io
    import tempfile
    from types import SimpleNamespace
    import unittest
    from unittest.mock import patch
    from vera_mem import qkv_eval
    from vera_mem.qkv_data import dataset
    from vera_mem.context_distillation import ContextDistillationBackend

    class FakeBackend:
        _teacher_forward=ContextDistillationBackend._teacher_forward
        _TEACHER_TRANSIENT_FIELDS=ContextDistillationBackend._TEACHER_TRANSIENT_FIELDS
        def __init__(self,rows):
            self.device='cpu';self.mode='none';self.last_retrieval=None;self.retrieval_trace=[]
            self.vector_store=None;self.trace_retrieval=False
            self.by_question={q:r for r in rows for q in r['questions']}
            self.by_context={s:r[FIELDS[w]] for r in rows for w,views in zip(r['worlds'],r['supports']) for s in views}
        @contextlib.contextmanager
        def disabled(self):
            old=self.mode;self.mode='none'
            try:yield
            finally:self.mode=old
        def generate(self,question,context=None):
            if self.mode=='vector_vera':
                for i in range(3):
                    info=self.vector_store.search(torch.ones(1,1,self.vector_store.key_dim))
                    self.last_retrieval=info
                    if self.trace_retrieval:self.retrieval_trace.append(dict(phase='prefill' if i==0 else 'decode',indices=info['indices'][:,-1]))
            return (self.by_context[context] if context else self.by_question[question]['a']),[71,72,73],.01
        def score(self,question,answer):return SimpleNamespace(nll_sum=1.2,tokens=3)

    class Tests(unittest.TestCase):
        def setUp(self):
            self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
            self.suite=self.root/'qkv_fixture';self.suite.mkdir();self.serial=0
        def tearDown(self):self.temp.cleanup()
        def fixture(self,mode='grouped',readout='vera',part='smoke',teacher=False):
            self.serial+=1;rows=dataset()['known'];rng=torch.Generator().manual_seed(991)
            packet=dict(split='known',rows=rows,worlds=list(FIELDS),slots3=torch.randn(16,5,2,3,5,generator=rng))
            m=QKVVeRA(5,6,rank=3,key_dim=4,seed=81042,read_mode=mode,readout=readout)
            m.fit_statistics(torch.randn(16,5,generator=rng),torch.randn(16,5,generator=rng))
            m.fit_value_statistics(torch.randn(16,5,generator=rng));m.b.data.fill_(.3)
            traincache=self.suite/f'train{self.serial}.pt';torch.save(dict(split='train'),traincache)
            parent=self.suite/f'train_parent{self.serial}';parent.mkdir();checkpoint=parent/'last.pt'
            torch.save(dict(protocol=PROTOCOL,architecture=m.configuration(),module=m.state_dict(),step=1,
                seed=81042,regime='static',cache_sha256=file_hash(traincache)),checkpoint)
            (parent/'manifest.json').write_text(json.dumps(dict(protocol=PROTOCOL,complete=True,stage='train',
                cache_sha256=file_hash(traincache),configuration=dict(cache=str(traincache),updates=1,seed=81042,
                    regime='static',read_mode=mode,readout=readout))))
            directory=self.suite/f'smoke_eval{self.serial}';directory.mkdir()
            args=SimpleNamespace(run_dir=directory,eval_part=part)
            with patch.object(qkv_eval,'generate_tokens',lambda backend,*a,**kw:backend.generate(*a,**kw)):
                result=qkv_eval.teacher(FakeBackend(rows),packet,args) if teacher else qkv_eval.evaluate(FakeBackend(rows),m,packet,args)
            manifest=dict(configuration=dict(checkpoint=str(checkpoint),eval_part=part),
                checkpoint_sha256=file_hash(checkpoint),result=result)
            return directory,manifest,packet
        def replay(self,d,m,p):
            # Tiny features deliberately bypass only the production 9728 shape gate.
            # All parent bindings, actual stores, writer encoding and raw reads run.
            with patch.dict(globals(),load_packet=lambda inputs,path,expected:inputs.load(path,expected)):
                return audit_eval(d,m,p,Inputs(self.root))
        def write_raw(self,d,rows):(d/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        def test_cpu_flat_grouped_additive_replay_actual_thirteen_generations(self):
            for mode,readout in [('flat','vera'),('grouped','vera'),('grouped','additive')]:
                with self.subTest(mode=mode,readout=readout):
                    result=self.replay(*self.fixture(mode,readout))
                    self.assertEqual(result['generation_calls'],13)
                    self.assertEqual(result['real_group_write_events'],4)
                    self.assertEqual(result['groups']['CC_B']['locality_joint'],1)
                    self.assertEqual(result['groups']['CC_B']['pair_em'],0)
                    self.assertEqual(result['groups']['CC_B']['restored_em'],1)
        def test_complete_development_axes_and_edited_slots(self):
            result=self.replay(*self.fixture(part='development'))
            self.assertEqual(result['generation_calls'],224)
            self.assertEqual(result['independent_interventions'],48)
            self.assertEqual(result['restorations'],48)
            self.assertEqual(result['shuffled_group_controls'],32)
            self.assertEqual(result['edited_slot_diagnostics']['updated_predictions'],16)
            self.assertEqual(result['edited_slot_diagnostics']['position_counts'],{0:6,1:5,2:5})
        def test_patched_value_must_equal_reencoded_payload(self):
            d,m,p=self.fixture();events=torch.load(d/'bank_events.pt',weights_only=True)
            events['events'][0]['values'][0,0]+=1;torch.save(events,d/'bank_events.pt')
            with self.assertRaisesRegex(ValueError,'Patch differs'):self.replay(d,m,p)
        def test_encoded_writer_must_match_cache(self):
            d,m,p=self.fixture();encoded=torch.load(d/'encoded_payloads.pt',weights_only=True)
            encoded['values'][0,0,0,0,0]+=1;torch.save(encoded,d/'encoded_payloads.pt')
            with self.assertRaisesRegex(ValueError,'Writer re-encoding'):self.replay(d,m,p)
        def test_trace_token_scores_mass_and_first_mass_shape(self):
            d,m,p=self.fixture();original=read_jsonl(d/'predictions.jsonl')
            mutations=[lambda r:r['trace'][0].update(token_id=99),
                       lambda r:r['trace'][0]['weights'][0].__setitem__(0,.33),
                       lambda r:r['trace'][0].update(target_slot_mass=[2,0,0]),
                       lambda r:r.update(first_target_slot_mass=[])]
            for change in mutations:
                with self.subTest(change=change):
                    rows=copy.deepcopy(original);change(rows[0]);self.write_raw(d,rows)
                    with self.assertRaises(ValueError):self.replay(d,m,p)
            self.write_raw(d,original);self.replay(d,m,p)
        def test_grouped_indices_are_complete_canonical_fact(self):
            d,m,p=self.fixture();rows=read_jsonl(d/'predictions.jsonl')
            rows[0]['trace'][0]['indices'][0][-1]=(rows[0]['trace'][0]['indices'][0][-1]+3)%48
            self.write_raw(d,rows)
            with self.assertRaisesRegex(ValueError,'canonical complete fact'):self.replay(d,m,p)
        def test_teacher_access_and_full_token_costs(self):
            d,m,p=self.fixture(teacher=True,part='development')
            self.assertEqual(audit_teacher(d,m,p)['generation_calls'],48)
            rows=read_jsonl(d/'predictions.jsonl');rows[0]['memory_access']['store_present']=True;self.write_raw(d,rows)
            with self.assertRaisesRegex(ValueError,'memory access'):audit_teacher(d,m,p)
        def log(self,regime='rebind',steps=8):
            cfg=dict(updates=steps,regime=regime,seed=81042,read_mode='grouped');raw=[];h=hashlib.sha256()
            for step in range(steps):
                ep=episode(step,regime,81042);h.update(json.dumps(ep,sort_keys=True).encode())
                raw.append(dict(step=step+1,episode=ep,metrics=dict(ce=1.,address=2.,loss=1.4),
                    parameter_gradient_norms={k:1. for k in ('Wq.weight','Wk.weight','Wv.weight','slot_position','B','b')},
                    gradient_norms=[1.]*4,input_lengths=[10]*16,gold_token_lengths=[4]*16,
                    read_diagnostics=dict(scope='captured gold-prefix prediction positions including EOS; parameter state before update',
                        prediction_positions=64,input_rms=1.,mixed_value_rms=1.,residual_rms=0.,
                        target_fact_mass=.5,actual_read_slots=3) if step==0 or (step+1)%128==0 else None))
            status=dict(updates=steps,complete=True,schedule_sha256=h.hexdigest(),totals=dict(target_exposures=8*steps,
                gold_tokens=64*steps,input_positions=160*steps,padded_input_positions=160*steps,backbone_calls=steps))
            return raw,status,cfg
        def test_episode_marginals_and_full_2048_token_ledger(self):
            static=audit_training_log(*self.log('static',2048));rebind=audit_training_log(*self.log('rebind',2048))
            self.assertEqual(static['completed_epochs'],256)
            self.assertEqual(static['target_bank_schedule_sha256'],rebind['target_bank_schedule_sha256'])
            self.assertNotEqual(static['schedule_sha256'],rebind['schedule_sha256'])
            self.assertEqual(len(rebind['actual_query_read_diagnostics']),17)
            self.assertTrue(rebind['token_totals_independently_resummed'])
        def test_episode_single_target_and_ledger_tampering(self):
            for mutation in ('episode','gold','gradient','diagnostic'):
                raw,status,cfg=self.log()
                if mutation=='episode':raw[0]['episode']['local_targets'][0]=17
                elif mutation=='gold':raw[0]['gold_token_lengths'][0]+=1
                elif mutation=='gradient':raw[0]['parameter_gradient_norms']['B']=float('nan')
                else:raw[0]['read_diagnostics']['residual_rms']=1.
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):audit_training_log(raw,status,cfg,smoke=True)
        def test_old_smoke_schema_is_explicit_and_cannot_be_formal(self):
            raw,status,cfg=self.log(steps=2)
            for row in raw:
                for key in ('input_lengths','gold_token_lengths','read_diagnostics'):del row[key]
            result=audit_training_log(raw,status,cfg,smoke=True,schema=dict(token_ledger=False,read_diagnostics=False))
            self.assertFalse(result['token_totals_independently_resummed'])
            with self.assertRaises(ValueError):audit_training_log(raw,status,cfg,smoke=False,schema=dict(token_ledger=False,read_diagnostics=False))
        def test_same_seed_matrix_preserves_common_architecture_and_stats(self):
            jobs=[]
            for seed in SEEDS:
                for regime,mode,readout in sorted(ARMS):
                    d=self.suite/f'{seed}_{regime}_{mode}_{readout}';d.mkdir()
                    model=QKVVeRA(5,6,rank=3,key_dim=4,seed=seed,read_mode=mode,readout=readout)
                    torch.save(dict(module=model.state_dict(),architecture=model.configuration(),cache_sha256='same'),d/'initial.pt')
                    jobs.append(dict(stage='train',scope='formal',run_dir=str(d),
                        training_configuration=dict(seed=seed,regime=regime,read_mode=mode,readout=readout),
                        target_bank_schedule_sha256=str(seed),schedule_sha256=str(seed)+regime,trainable_parameters=999,
                        token_totals=dict(target_exposures=16384,gold_tokens=50000,input_positions=999999,
                            padded_input_positions=1000000+(regime=='rebind'),backbone_calls=2048)))
            result=training_matrix(jobs,Inputs(self.root));self.assertTrue(result['complete'])
            self.assertEqual(result['observed_conditions'],15)
            path=Path(jobs[-1]['run_dir'])/'initial.pt';state=torch.load(path,weights_only=True)
            state['architecture']['temperature']=.9;torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'common architecture'):training_matrix(jobs,Inputs(self.root))
        def test_exclusion_skips_poisoned_suite_and_output_no_overwrite(self):
            (self.suite/'analysis_excluded.json').write_text(json.dumps(dict(reason='synthetic exclusion',
                evidence='synthetic fixture',superseded_by='synthetic replacement')))
            (self.suite/'suite.json').write_text('not json')
            result=run_audit(self.root);self.assertEqual(len(result['excluded']),1)
            self.assertFalse(result['complete']);self.assertFalse(result['pending'])
            output=self.root/'exists.json';output.write_text('keep')
            with self.assertRaisesRegex(ValueError,'overwrite'):main(['--runs-root',str(self.root),'--output',str(output)])
            self.assertEqual(output.read_text(),'keep')
        def test_final_inventory_requires_sixty_evals_and_five_teachers(self):
            jobs=[]
            for seed in SEEDS:
                for regime,mode,readout in ARMS:
                    jobs.append(dict(scope='formal',stage='train',training_configuration=dict(seed=seed,regime=regime,read_mode=mode,readout=readout)))
                    for split,part in [('known','development'),('dev','development'),('known','confirmation'),('confirm','confirmation')]:
                        jobs.append(dict(scope='formal',stage='eval',lineage=dict(seed=seed,regime=regime),
                            read_mode=mode,readout=readout,cache_split=split,configuration=dict(eval_part=part)))
            for split,part in [('train','preflight'),('known','development'),('dev','development'),('known','confirmation'),('confirm','confirmation')]:
                jobs.append(dict(scope='formal',stage='teacher',cache_split=split,configuration=dict(eval_part=part)))
            self.assertTrue(formal_inventory(jobs)['complete'])
            self.assertFalse(formal_inventory(jobs[:-1])['complete'])
            with self.assertRaisesRegex(ValueError,'duplicate'):formal_inventory(jobs+[jobs[0]])
        def test_confirmation_snapshot_and_original_artifact_hashes(self):
            directory=self.suite/'train';directory.mkdir();artifact=directory/'manifest.json';artifact.write_text('{}')
            decision=dict(selected_arm=None)
            value=dict(decision=decision,development_artifacts=[dict(run_dir=str(directory),kind='training',
                arm='static_flat',seed=81042,artifacts={'manifest.json':file_hash(artifact)})])
            path=self.root/'selection.json';path.write_text(json.dumps(value));saved=self.suite/'selection_evidence.json';saved.write_bytes(path.read_bytes())
            suite=dict(selection_evidence_sha256=file_hash(path))
            job=dict(run_dir=str(directory),scope='formal',stage='train',trainable_parameters=2034368,
                training_configuration=dict(seed=81042,regime='static',read_mode='flat',readout='vera'))
            # This unit isolates immutable receipt binding; JSON seal/ranking
            # contracts have independent summarizer tests and run unmocked in CLI.
            with patch.dict(globals(),validate_selection=lambda v,p:v,development_selection=lambda rows:decision):
                result=audit_confirmation_barrier(Inputs(self.root),path,[(self.suite,suite)],[job])
                self.assertEqual(result['evidence_artifacts_rehashed'],1)
                saved.write_text('{}')
                with self.assertRaisesRegex(ValueError,'selection snapshot'):audit_confirmation_barrier(Inputs(self.root),path,[(self.suite,suite)],[job])
                saved.write_bytes(path.read_bytes());artifact.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'artifact changed'):audit_confirmation_barrier(Inputs(self.root),path,[(self.suite,suite)],[job])

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs-root',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--training-only',action='store_true');p.add_argument('--self-test',action='store_true')
    p.add_argument('--registration',type=Path,help='Sealed registration; default workspace plans, then public copy')
    p.add_argument('--selection',type=Path,help='Original development selection, required when confirmation suites exist')
    p.add_argument('--require-final',action='store_true',help='Require complete 15 train / 60 eval / five teacher matrix');args=p.parse_args(argv)
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    if args.self_test:return self_test()
    if args.output is not None:require(not args.output.exists(),'Refusing to overwrite audit output')
    if args.runs_root is None or args.output is None:p.error('--runs-root and --output are required')
    start=time.perf_counter();result=run_audit(args.runs_root,args.training_only,args.registration,args.selection,args.require_final)
    result.update(elapsed_seconds=time.perf_counter()-start,audit_source_sha256=file_hash(Path(__file__)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(complete=result['complete'],scopes=result['scopes'],output=str(args.output))))
    return 0


if __name__=='__main__':raise SystemExit(main())
