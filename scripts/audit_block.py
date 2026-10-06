"""CPU-only replay of preregistered full-block experiments.

Reuses source-independent file, teacher and token-arithmetic helpers. Routing is
independently recomputed from the saved actual query and the entire CPU key bank;
no CUDA, backbone execution, feature extraction, or model selection is performed.
Only launcher-complete jobs are opened. --training-only does not open eval output.
"""
from __future__ import annotations
import argparse
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

os.environ['CUDA_VISIBLE_DEVICES'] = ''
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src')); sys.path.insert(0, str(REPO/'scripts'))
import torch
from torch.nn import functional as F
from vera_mem.block_vera import BlockVeRA, BlockVectorDB
from vera_mem.block_data import (episode, validate_dataset, validate_training_data,
                                 digest, PROTOCOL as DATA_PROTOCOL)
from vera_mem.context_distillation_run import REVISION
from vera_mem.reconstruction_data import WRITER_PREFIX
from vera_mem.metrics import normalize_answer
from audit_reconstruction import Inputs, require, same, state_digest, file_hash, read_json, read_jsonl
from audit_qkv import axes, close, token_costs, audit_teacher, FIELDS, PHASES, METRICS
from replay_coldstart_banks import analysis_exclusions

PROTOCOL = 'block-attention-memory-v1'
AUDIT_PROTOCOL = 'block-independent-artifact-audit-v1'
SEEDS = (91042, 91043, 91044)
MODES = ('diagonal', 'pooled_outer', 'block_outer')
ARMS = {(regime, mode) for regime in ('static', 'rebind') for mode in MODES}
COMMON_NAMES = {'Wq.weight','Wk.weight','Wv.weight','b','B','slot_position'}
OUTER_NAMES = {'P_in.weight','P_in.bias','P_out.weight'}


def check_packet(packet):
    require(packet['protocol']==PROTOCOL and packet['model_revision']==REVISION
            and packet['data_seed']==221042 and packet['writer_prefix']==WRITER_PREFIX
            and packet['rows_sha256']==digest(packet['rows']), 'Cache protocol/revision/data/rows differ')
    require(len({r['id'] for r in packet['rows']})==len(packet['rows']), 'Duplicate cache IDs')
    if packet['split']=='train':
        validate_training_data(dict(protocol=DATA_PROTOCOL,seed=packet['data_seed'],entities=packet['entities'],
                                    payloads=packet['payloads'],rows=packet['rows']))
        require(packet['worlds']==['A','B'] and packet['views']==1
                and packet['q'].shape==(64,9728) and packet['matrix'].shape==(64,128,3,9728), 'Training axes differ')
        tensors=(packet['q'],packet['matrix']); observations=8192
    else:
        require(packet['split'] in ('known','dev','confirm') and len(packet['rows'])==16
                and packet['worlds']==list(FIELDS) and packet['views']==2
                and packet['q'].shape==(16,2,9728) and packet['slots3'].shape==(16,5,2,3,9728), 'Evaluation axes differ')
        tensors=(packet['q'],packet['slots3']); observations=160
    require(all(t.is_floating_point() and not t.requires_grad and bool(torch.isfinite(t).all()) for t in tensors),
            'Features must be finite detached floating tensors')
    counts=packet['payload_token_counts']
    require(isinstance(counts,list) and len(counts)==observations and all(isinstance(c,list) and len(c)==3
            and all(type(n)is int and n>0 for n in c) for c in counts), 'Word-token ledger differs')
    return packet


def load_packet(inputs,path,expected):
    p=inputs.resolve(path); packet=inputs.load(str(p),expected)
    checked=getattr(inputs,'checked_block_packets',set())
    if p not in checked: check_packet(packet); checked.add(p); inputs.checked_block_packets=checked
    return packet


def source_check(directory,manifest,inputs):
    sources=manifest['source_sha256']; frozen=directory.parent/'source/src/vera_mem'
    require({'block_vera.py','block_backend.py','block_run.py','block_eval.py','block_data.py'}<=set(sources), 'Missing block runtime binding')
    for name,expected in sources.items():
        require(Path(name).name==name and name.endswith('.py'), 'Unsafe runtime source name')
        require(inputs.sha(frozen/name)==expected, 'Frozen runtime source differs: '+name)
    replay=('block_vera.py','block_data.py','qkv_vera.py','qkv_data.py','reconstruction_vera.py',
            'reconstruction_data.py','interface_variants.py','stable_vector_vera.py','vector_vera.py','vector_store.py')
    for name in replay: require(file_hash(REPO/'src/vera_mem'/name)==sources[name], 'Changed CPU replay dependency: '+name)
    return dict(bound_source_files=len(sources),replay_dependencies=list(replay),frozen_snapshot=True)


def configuration(arguments):
    result=dict(seed=91042,data_seed=221042,regime='static',block_mode='diagonal',updates=2048,
                eval_part='development',cache=None,checkpoint=None)
    allowed=set(result)|{'stage','model'}; seen=set()
    require(isinstance(arguments,list) and len(arguments)%2==0, 'Malformed plan arguments')
    for flag,value in zip(arguments[::2],arguments[1::2]):
        require(isinstance(flag,str) and flag.startswith('--') and isinstance(value,str), 'Malformed plan flag')
        name=flag[2:].replace('-','_'); require(name in allowed and name not in seen, 'Unknown/duplicate plan flag')
        seen.add(name); result[name]=int(value) if name in ('seed','data_seed','updates') else value
    require({'stage','model'}<=seen, 'Missing plan stage/model')
    return result


def replay_route(event,db):
    """Independent full-bank cosine/LSE implementation; no store.search call."""
    query=torch.tensor(event['query'],dtype=torch.float32)
    require(query.shape==(1,db.key_dim) and bool(torch.isfinite(query).all()), 'Missing/nonfinite actual query')
    width=3 if len(db) else 0; indices=event['indices']
    require(isinstance(indices,list) and len(indices)==1 and len(indices[0])==width
            and all(type(i)is int for i in indices[0]), 'Invalid selected slot indices')
    scores=torch.tensor(event['scores'],dtype=torch.float32); weights=torch.tensor(event['weights'],dtype=torch.float32)
    require(scores.shape==weights.shape==(1,width) and bool(torch.isfinite(scores).all())
            and bool(torch.isfinite(weights).all()), 'Invalid route scores/weights')
    require(event['operator_weighting']=='uniform'
            and event['weight_interpretation']=='uniform membership; not signed outer coefficients', 'Wrong weighting semantics')
    if not width:
        require(not bool(query.any()), 'Empty-bank query must match explicit zero-query bypass')
        return dict(score_max_abs_error=0.,selected_fact=None)
    require(float(query.norm())>0, 'Nonempty-bank query has zero norm')
    q=query/query.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    keys=db.keys.float(); k=keys/keys.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    all_scores=q @ k.T
    logits=torch.logsumexp(all_scores.reshape(1,-1,3)/db.temperature,dim=-1)
    winner=int(logits.argmax(-1)); expected=[3*winner+s for s in range(3)]
    require(indices[0]==expected, 'Saved actual query does not select the recorded full-bank winning group')
    expected_scores=all_scores[:,expected]
    require(torch.allclose(scores,expected_scores,rtol=1e-6,atol=2e-7), 'Selected cosine scores differ from actual query')
    require(same(weights,torch.full_like(weights,1/3)), 'Selected block must have uniform FP32 row weights')
    return dict(score_max_abs_error=float((scores-expected_scores).abs().max()),selected_fact=winner)


def validate_read(row,db,fact,world,phase,condition,role,case_id,trigger):
    require(tuple(row[k] for k in ('id','world','phase','condition','role','case_id','trigger_world'))
            ==(fact['id'],world,phase,condition,role,case_id,trigger), 'Generation identity/order differs')
    require(row['question']==fact['questions'][PHASES[phase][1]] and row['answer']==fact[FIELDS[world]]
            and not row.get('context'), 'Student prompt or answer binding differs')
    require(row['bank_hash']==db.hash() and row['bank_slots']==len(db), 'Generation bank hash differs')
    require(row['em']==int(normalize_answer(row['prediction'])==normalize_answer(row['answer']))
            and row['budget_hit']==(row['generation_tokens']>=32), 'EM/budget arithmetic differs')
    require(type(row['answer_tokens'])is int and row['answer_tokens']>0
            and math.isfinite(row['nll_sum']) and row['nll_sum']>=-1e-6, 'Invalid answer score')
    trace=row['trace']; require(len(trace)==row['generation_tokens']==len(row['generated_token_ids'])>0, 'Token/route alignment differs')
    target=db.fact_ids.index(fact['id']) if fact['id'] in db.fact_ids else -1
    hits=[]; masses=[]; max_error=0.
    for position,(event,token) in enumerate(zip(trace,row['generated_token_ids'])):
        require(event['phase']==('prefill' if position==0 else 'decode') and event['token_id']==token, 'Predictive position differs')
        proof=replay_route(event,db); max_error=max(max_error,proof['score_max_abs_error'])
        expected=[sum(w for i,w in zip(event['indices'][0],event['weights'][0]) if target>=0 and i==3*target+s) for s in range(3)]
        require(len(event['target_slot_mass'])==3, 'Missing slot membership')
        for got,want in zip(event['target_slot_mass'],expected): close(got,want,'Slot membership differs')
        close(event['target_fact_mass'],sum(expected),'Fact membership differs')
        masses.append(expected); hits.append(int(target>=0 and proof['selected_fact']==target))
    width=len(trace[0]['indices'][0])
    require(row['first_fact_recall_at_read']==row['first_fact_recall_at_1']==hits[0]
            and row['decode_hits']==sum(hits[1:]) and row['decode_queries']==len(hits)-1
            and row['actual_read_slots']==width and row['sparse_values_transferred_per_token']==width*db.value_dim
            and row['injection_input']=='full_selected_value_block', 'Read counters/block interface differ')
    require(len(row['first_target_slot_mass'])==3, 'Missing first slot membership')
    for got,want in zip(row['first_target_slot_mass'],masses[0]): close(got,want,'First slot membership differs')
    close(row['first_target_fact_mass'],sum(masses[0]),'First fact membership differs')
    return dict(queries=len(trace),score_max_abs_error=max_error)


def checkpoint_parent(manifest,inputs):
    cfg=manifest['configuration']; path=inputs.resolve(cfg['checkpoint'])
    ck=inputs.load(str(path),manifest['checkpoint_sha256']); parent_path=path.parent/'manifest.json'; parent=read_json(parent_path)
    require(not analysis_exclusions(path.parent,inputs.root), 'Excluded training parent')
    pc=parent['configuration']; arch=ck['architecture']
    require(parent['complete'] is True and parent['stage']=='train' and parent['protocol']==PROTOCOL
            and path.name=='last.pt' and ck['protocol']==PROTOCOL and ck['step']==pc['updates'], 'Non-final training parent')
    require(ck['cache_sha256']==parent['cache_sha256'] and ck['seed']==pc['seed']==arch['seed']
            and ck['regime']==pc['regime'] and ck['block_mode']==pc['block_mode']==arch['block_mode'], 'Parent factors differ')
    require(load_packet(inputs,pc['cache'],parent['cache_sha256'])['split']=='train', 'Nontraining parent cache')
    return ck,dict(seed=ck['seed'],regime=ck['regime'],block_mode=ck['block_mode'],parent_train=str(path.parent),
                   checkpoint_sha256=inputs.sha(path),parent_manifest_sha256=inputs.sha(parent_path))


@torch.no_grad()
def audit_eval(directory,manifest,packet,inputs):
    ck,lineage=checkpoint_parent(manifest,inputs); model=BlockVeRA(**ck['architecture'])
    model.load_state_dict(ck['module']); model.eval().requires_grad_(False); before=state_digest(model.state_dict())
    result=read_json(directory/'summary.json'); raw=read_jsonl(directory/'predictions.jsonl'); saved_pairs=read_json(directory/'pairs.json')
    require(result==manifest['result'] and result['protocol']=='block-online-cpu-v1' and result['complete'] is True
            and result['architecture']==ck['architecture'] and result['shared_parameters_unchanged'] is True
            and result['shared_parameter_sha256']==before, 'Frozen evaluation identity differs')
    encoded=inputs.load(str(directory/'encoded_payloads.pt')); banks=inputs.load(str(directory/'bank_events.pt')); rows=packet['rows']; n=len(rows)
    require(encoded['fact_ids']==[r['id'] for r in rows] and encoded['worlds']==packet['worlds']
            and banks['protocol']=='block-bank-events-v1', 'Bank source identity differs')
    errors={}
    for name,reference in [('keys',model.encode_key(packet['slots3'].float())),('values',model.encode_value(packet['slots3'].float()))]:
        actual=encoded[name]
        require(actual.shape==reference.shape and actual.dtype==torch.float32 and bool(torch.isfinite(actual).all())
                and torch.allclose(actual,reference,rtol=2e-5,atol=2e-5), 'Writer re-encoding differs')
        errors[name]=float((actual-reference).abs().max())
    part=manifest['configuration']['eval_part']; grid=axes(packet['split'],part); indices=list(range(min(2,n) if part=='smoke' else n))
    count=sum(len(w)*len(indices) for w in grid.values())
    require(set(banks['bases'])==set(grid) and len(banks['events'])==len(saved_pairs)==count, 'Evaluation grid differs')
    cursor=event_cursor=route_count=0; max_error=0.; pairs=[]; shuffles=0
    def consume(db,fact,world,phase,condition,role,case_id,trigger):
        nonlocal cursor,route_count,max_error
        require(cursor<len(raw),'Missing generation'); row=raw[cursor]; cursor+=1
        proof=validate_read(row,db,fact,world,phase,condition,role,case_id,trigger)
        route_count+=proof['queries']; max_error=max(max_error,proof['score_max_abs_error']); return row
    for phase,worlds in grid.items():
        sv,_=PHASES[phase]; base=BlockVectorDB.from_snapshot(banks['bases'][phase]); expected=model.new_store()
        for i,f in enumerate(rows): expected.write_group(f['id'],encoded['keys'][i,0,sv],encoded['values'][i,0,sv],0)
        require(base.hash()==expected.hash(),'Base differs from encoded A'); base_hash=base.hash()
        needed=set(indices)|{(i+1)%n for i in indices}
        initial={i:consume(base,rows[i],'A',phase,'real','initial',rows[i]['id'],'A') for i in sorted(needed)}
        for world in worlds:
            wi=packet['worlds'].index(world)
            for i in indices:
                fact=rows[i]; event=banks['events'][event_cursor]; event_cursor+=1
                require(tuple(event[k] for k in ('phase','world','index','id','timestamp'))==(phase,world,i,fact['id'],1)
                        and event['before_hash']==base_hash, 'Intervention identity differs')
                require(same(event['keys'],encoded['keys'][i,wi,sv]) and same(event['values'],encoded['values'][i,wi,sv]), 'Write differs from writer')
                changed=BlockVectorDB.from_snapshot(base.snapshot()); changed.write_group(fact['id'],event['keys'],event['values'],1)
                require(changed.hash()==event['after_hash'],'Updated bank hash differs')
                mask=torch.ones(len(base),dtype=torch.bool); mask[3*i:3*i+3]=False
                require(same(changed.keys[mask],base.keys[mask]) and same(changed.values[mask],base.values[mask])
                        and changed.fact_ids==base.fact_ids and changed.timestamps[3*i:3*i+3]==(1,)*3
                        and all(x==y for j,(x,y) in enumerate(zip(changed.timestamps,base.timestamps)) if mask[j]), 'Non-target or partial-group write')
                updated=consume(changed,fact,world,phase,'real','updated',fact['id'],world); neighbor=(i+1)%n
                local=consume(changed,rows[neighbor],'A',phase,'real','locality',fact['id'],world)
                restored=BlockVectorDB.from_snapshot(changed.snapshot()); restored.write_group(fact['id'],encoded['keys'][i,0,sv],encoded['values'][i,0,sv],2)
                require(restored.hash()==event['restored_hash'] and same(restored.keys,base.keys) and same(restored.values,base.values)
                        and restored.timestamps[3*i:3*i+3]==(2,)*3,'Restoration differs')
                rr=consume(restored,fact,'A',phase,'real','restored',fact['id'],world)
                pair=dict(id=fact['id'],phase=phase,world=world,a_em=initial[i]['em'],updated_em=updated['em'],
                    pair_em=initial[i]['em']*updated['em'],restored_em=rr['em'],update_restore_em=updated['em']*rr['em'],
                    all_three_em=initial[i]['em']*updated['em']*rr['em'],locality_equal=int(local['prediction']==initial[neighbor]['prediction']),
                    locality_correct=local['em'],locality_joint=local['em']*initial[neighbor]['em'],shuffle_em=None,empty_em=None)
                if world!='SWAP':
                    order=list(range(n)); random.Random(131042+i).shuffle(order); perm=list(range(n))
                    for a,b in zip(order,order[1:]+order[:1]): perm[a]=b
                    require(event['value_permutation']==perm and all(j!=p for j,p in enumerate(perm)), 'Shuffle differs from declared derangement')
                    state=changed.snapshot(); state['store']['values']=state['store']['values'].reshape(n,3,model.rank)[perm].reshape(-1,model.rank).clone()
                    wrong=BlockVectorDB.from_snapshot(state)
                    require(wrong.hash()==event['shuffle_hash'] and same(wrong.keys,changed.keys)
                            and wrong.timestamps==changed.timestamps and wrong.fact_ids==changed.fact_ids, 'Shuffle changed keys/IDs/timestamps or hash')
                    for control,db in [('shuffle',wrong),('empty',model.new_store())]:
                        pair[control+'_em']=consume(db,fact,world,phase,control,'control',fact['id'],world)['em']
                    shuffles+=1
                require(base.hash()==base_hash,'Intervention contaminated base'); pairs.append(pair)
    require(cursor==len(raw) and pairs==saved_pairs and state_digest(model.state_dict())==before, 'Raw/pair arithmetic or frozen model differs')
    grouped=defaultdict(list)
    for pair in pairs: grouped[pair['phase']+'_'+pair['world']].append(pair)
    groups={}
    for key,items in grouped.items():
        groups[key]=dict(count=len(items))
        for metric in METRICS:
            values=[p[metric] for p in items if p[metric] is not None]; groups[key][metric]=sum(values)/len(values) if values else None
    require(groups==result['groups'] and result['bank_facts']==n and result['slots_per_fact']==3
            and result['resident_bytes']==base.resident_bytes() and result['independent_interventions']==count
            and result['restorations']==count and result['real_group_write_events']==2*count, 'Evaluation totals differ')
    token_costs(raw,result)
    return dict(groups=groups,lineage=lineage,generation_calls=len(raw),independent_interventions=count,restorations=count,
        real_group_write_events=2*count,shuffled_group_controls=shuffles,writer_max_abs_errors=errors,
        actual_queries_replayed=route_count,full_bank_group_winner_verified=True,all_three_slots_and_uniform_weights_verified=True,
        selected_score_max_abs_error=max_error,cached_writer_reencoded=True,online_state_sha256=before,
        telemetry_limitation='Q/K replay verifies the recorded query, not its derivation from an independently rerun backbone; uniform membership is not signed outer contribution.')


def parameter_groups(mode):
    groups=[['Wq.weight','Wk.weight','slot_position'],['Wv.weight'],['b'],['B']]; rates=[1e-4,3e-4,.005,3e-4]
    if mode!='diagonal': groups.append(['P_in.weight','P_in.bias','P_out.weight']); rates.append(3e-4)
    return groups,rates


def audit_training_log(raw,status,cfg,smoke=False):
    require(status['complete'] is True and len(raw)==cfg['updates']==status['updates'], 'Training budget differs')
    mode=cfg['block_mode']; names=COMMON_NAMES|(OUTER_NAMES if mode!='diagonal' else set()); groups,rates=parameter_groups(mode)
    schedule=hashlib.sha256(); addresses=hashlib.sha256(); epochs=defaultdict(lambda:dict(entities=Counter(),payloads=Counter()))
    nonzero=Counter(); maxima=Counter(); diagnostics=[]
    totals=dict(target_exposures=0,gold_tokens=0,input_positions=0,padded_input_positions=0,backbone_calls=0)
    last_time=-1.
    for step,row in enumerate(raw):
        ep=episode(step,cfg['regime'],cfg['seed']); require(row['step']==step+1 and row['episode']==ep, 'Training episode differs')
        schedule.update(json.dumps(ep,sort_keys=True).encode())
        address={k:ep[k] for k in ('step','seed','epoch','batch_index','targets','background','bank_entities','local_targets')}
        addresses.update(json.dumps(address,sort_keys=True).encode())
        epochs[ep['epoch']]['entities'].update(ep['targets'])
        epochs[ep['epoch']]['payloads'].update(ep[field][i] for field in ('a_payload_indices','b_payload_indices') for i in ep['local_targets'])
        metrics=row['metrics']; close(metrics['loss'],metrics['ce']+.2*metrics['address'],'Training loss arithmetic differs')
        require(all(math.isfinite(metrics[k]) and metrics[k]>=-1e-6 for k in ('loss','ce','address')), 'Invalid training losses')
        grads=row['parameter_gradient_norms']; require(set(grads)==names,'Gradient parameter coverage differs')
        for name,value in grads.items():
            require(value is not None and math.isfinite(value) and value>=0,'Invalid parameter gradient')
            nonzero[name]+=int(value>0); maxima[name]=max(maxima[name],value)
        require(len(row['gradient_norms'])==len(groups) and all(math.isfinite(x) and x>=0 for x in row['gradient_norms']), 'Clipped group telemetry differs')
        lengths=row['input_lengths']; gold=row['gold_token_lengths']
        require(len(lengths)==len(gold)==16 and all(type(a)is int and type(b)is int and 0<b<a for a,b in zip(lengths,gold)), 'Token ledger differs')
        totals['target_exposures']+=8; totals['gold_tokens']+=sum(gold); totals['input_positions']+=sum(lengths)
        totals['padded_input_positions']+=16*max(lengths); totals['backbone_calls']+=1
        d=row['read_diagnostics']; require((d is not None)==(step==0 or (step+1)%128==0), 'Read diagnostic timing differs')
        if d is not None:
            require(d['scope']=='captured gold-prefix prediction positions including EOS; parameter state before update'
                    and d['prediction_positions']==sum(gold) and d['actual_read_slots']==3, 'Read diagnostic positions differ')
            require(all(math.isfinite(d[k]) and d[k]>=0 for k in ('input_rms','mixed_value_rms','residual_rms','target_fact_mass'))
                    and d['target_fact_mass']<=1.00001, 'Nonfinite read diagnostics')
            if step==0: require(d['residual_rms']==0, 'Initial zero-b is not a no-op')
            singular=d['outer_operator_mean_singular_values']
            if mode=='diagonal': require(singular is None,'Diagonal unexpectedly has outer diagnostic')
            else:
                require(isinstance(singular,list) and len(singular)==64 and all(math.isfinite(x) and x>=0 for x in singular), 'Invalid outer singular values')
                require(all(a+1e-6>=b for a,b in zip(singular,singular[1:])), 'Mean singular spectrum not descending')
                bound=1 if mode=='pooled_outer' else 3
                require(max(singular[bound:])<=max(1e-5,singular[0]*2e-5), 'Extra operator exceeds declared numerical rank')
            diagnostics.append(dict(step=step+1,**d))
        require(math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>=last_time,'Invalid cumulative elapsed time'); last_time=row['elapsed_seconds']
    require(totals==status['totals'] and schedule.hexdigest()==status['schedule_sha256'], 'Token/schedule totals differ')
    require(math.isfinite(status['elapsed_seconds']) and status['elapsed_seconds']>=last_time,
            'Final cumulative elapsed time precedes the last training record')
    complete_epochs=0
    for epoch,counts in epochs.items():
        if sum(counts['entities'].values())==64:
            require(counts['entities']==Counter(range(64)) and counts['payloads']==Counter(range(128)), 'Entity/payload marginal exposure differs')
            complete_epochs+=1
        else: require(smoke and epoch==max(epochs),'Incomplete formal epoch')
    if not smoke: require(len(raw)==2048 and complete_epochs==256 and all(nonzero[n]>0 for n in names), 'Formal budget or gradient coverage differs')
    return dict(schedule_sha256=schedule.hexdigest(),target_bank_schedule_sha256=addresses.hexdigest(),completed_epochs=complete_epochs,
        token_totals=totals,parameter_nonzero_gradient_steps=dict(nonzero),parameter_max_gradient_norm=dict(maxima),
        actual_query_read_diagnostics=diagnostics,training_elapsed_seconds=status['elapsed_seconds'],
        gradient_scope='Finite logged parameter norms only. Uniform hard selection supplies no CE router gradient; address CE trains Q/K/slot_position.')


@torch.no_grad()
def audit_train(directory,manifest,packet,inputs,smoke=False):
    cfg=manifest['configuration']; status=read_json(directory/'training_status.json')
    require(packet['split']=='train' and status==manifest['result'],'Training status/cache differs')
    proof=audit_training_log(read_jsonl(directory/'training.jsonl'),status,cfg,smoke)
    initial=inputs.load(str(directory/'initial.pt')); final=inputs.load(str(directory/'last.pt'))
    terminal_path=directory/f"step_{cfg['updates']}.pt"; terminal=inputs.load(str(terminal_path))
    for ck in (initial,final,terminal):
        require(ck['protocol']==PROTOCOL and ck['seed']==cfg['seed'] and ck['regime']==cfg['regime']
                and ck['block_mode']==cfg['block_mode'] and ck['cache_sha256']==manifest['cache_sha256'], 'Checkpoint identity differs')
    require(initial['step']==0 and final['step']==terminal['step']==cfg['updates']
            and final['schedule_sha256']==terminal['schedule_sha256']==proof['schedule_sha256'], 'Checkpoint step/schedule differs')
    require(initial['architecture']==final['architecture']==terminal['architecture']==status['architecture'], 'Architecture changed')
    arch=final['architecture']; require(arch['block_mode']==cfg['block_mode'] and arch['seed']==cfg['seed']
            and arch['read_mode']=='grouped' and arch['readout']=='vera' and arch['rank']==arch['key_dim']==64
            and arch['slots']==3 and arch['temperature']==.05 and arch['factor_epsilon']==1e-6
            and arch['in_features']==9728 and arch['out_features']==2560, 'Registered architecture differs')
    model=BlockVeRA(**arch)
    fresh={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(initial['module']); params=dict(model.named_parameters())
    groups,rates=parameter_groups(cfg['block_mode']); names={n for g in groups for n in g}
    require(set(params)==names and 'A' in dict(model.named_buffers()) and status['learning_rates']==rates
            and status['trainable_parameters']==sum(p.numel() for p in params.values()), 'Adam/parameter contract differs')
    count=2034368+(8256 if cfg['block_mode']!='diagonal' else 0)
    require(status['trainable_parameters']==count, 'Registered trainable count differs')
    # Verify deterministic construction before train-fitted center overwrite.
    centers={'query_center','support_center','value_center','statistics_fitted','value_statistics_fitted'}
    # Cross-host CPU RMS reduction may differ by a few float32 ULPs for c.
    # Same-seed saved outer states remain subject to byte equality below.
    require(set(fresh)==set(initial['module']) and all(same(v,initial['module'][k]) for k,v in fresh.items()
            if k not in centers|{'P_in.bias'}), 'Initial random/common/outer parameters differ from declared seed')
    bias_error=None
    if 'P_in.bias' in fresh:
        require(torch.allclose(fresh['P_in.bias'],initial['module']['P_in.bias'],rtol=1e-6,atol=1e-6),
                'Initial affine bias differs from seeded unit-RMS construction')
        bias_error=float((fresh['P_in.bias']-initial['module']['P_in.bias']).abs().max())
    require(set(final['module'])==set(terminal['module']) and all(same(v,terminal['module'][k]) for k,v in final['module'].items()), 'Final/optimizer terminal state differs')
    parameter_updates={}
    for name in params:
        start=initial['module'][name]; finish=final['module'][name]
        difference=finish.double()-start.double()
        parameter_updates[name]=dict(bitwise_changed=not same(start,finish),
            changed_coordinates=int((finish!=start).sum()),coordinates=start.numel(),
            delta_l2=float(difference.norm()),delta_max_abs=float(difference.abs().max()))
    if not smoke and cfg['block_mode']!='diagonal':
        require(all(parameter_updates[name]['bitwise_changed'] for name in OUTER_NAMES),
                'Declared extra outer parameters did not change during formal training')
    fixed=set(initial['module'])-set(params)
    require(all(same(initial['module'][k],final['module'][k]) for k in fixed),'Frozen buffers changed')
    for state in (initial['module'],final['module']): require(all(bool(torch.isfinite(v).all()) for v in state.values()), 'Nonfinite model state')
    probe=BlockVeRA(**arch); i=torch.arange(64); f=packet['matrix'][i[:,None],2*i[:,None]+torch.arange(2)].reshape(-1,9728).float()
    probe.fit_statistics(packet['q'].float(),f); probe.fit_value_statistics(f); errors={}
    for name in centers:
        actual=initial['module'][name]; expected=probe.state_dict()[name]
        require(torch.allclose(actual,expected,rtol=1e-5,atol=1e-6),'Static train-only statistics differ')
        errors[name]=float((actual.float()-expected.float()).abs().max())
    optimizer=terminal['optimizer']; require(len(optimizer['param_groups'])==len(groups),'Adam groups differ'); seen=[]
    for group,names,lr in zip(optimizer['param_groups'],groups,rates):
        require(len(group['params'])==len(names) and group['lr']==lr and tuple(group['betas'])==(.9,.999)
                and group['eps']==1e-8 and group['weight_decay']==0 and not group['amsgrad'], 'Adam hyperparameters differ')
        for pid,name in zip(group['params'],names):
            seen.append(pid); state=optimizer['state'][pid]; require(float(state['step'])==cfg['updates'],'Adam steps differ')
            require(all(state[k].shape==params[name].shape and state[k].dtype==params[name].dtype
                        and bool(torch.isfinite(state[k]).all()) for k in ('exp_avg','exp_avg_sq'))
                    and bool((state['exp_avg_sq']>=0).all()), 'Adam moments differ')
    require(len(set(seen))==len(seen) and set(seen)==set(optimizer['state']),'Extra/duplicate Adam state')
    return dict(proof,training_configuration=cfg,updates=cfg['updates'],trainable_parameters=count,
        statistics_max_abs_errors=errors,initial_checkpoint_sha256=inputs.sha(directory/'initial.pt'),
        final_checkpoint_sha256=inputs.sha(directory/'last.pt'),terminal_optimizer_sha256=inputs.sha(terminal_path),
        deterministic_initialization_verified=True,seeded_bias_reconstruction_max_abs_error=bias_error,
        initial_to_final_parameter_updates=parameter_updates,
        A_is_fixed_buffer_outside_optimizer=True)


def training_matrix(jobs,inputs):
    groups=defaultdict(dict); cache=set(); reports=[]; missing=[]
    for job in jobs:
        if job['stage']!='train' or job['scope']!='formal': continue
        cfg=job['training_configuration']; arm=(cfg['regime'],cfg['block_mode']); seed=cfg['seed']
        require(seed in SEEDS and arm in ARMS and arm not in groups[seed], 'Duplicate/unknown formal training condition')
        groups[seed][arm]=job
    for seed in SEEDS:
        arms=groups[seed]; missing.extend(dict(seed=seed,arm=list(a)) for a in sorted(ARMS-set(arms)))
        if not arms: continue
        states={arm:inputs.load(str(Path(j['run_dir'])/'initial.pt')) for arm,j in arms.items()}
        reference=next(iter(states.values())); reference_job=next(iter(arms.values()))
        common=set(reference['module'])-OUTER_NAMES-{'block_architecture'}; within={}; outer_reference=None
        for arm,ck in states.items():
            job=arms[arm]; cache.add(ck['cache_sha256'])
            require({k:v for k,v in ck['architecture'].items() if k!='block_mode'}==
                    {k:v for k,v in reference['architecture'].items() if k!='block_mode'}, 'Common architecture differs')
            require(set(ck['module'])-OUTER_NAMES-{'block_architecture'}==common
                    and all(same(ck['module'][n],reference['module'][n]) for n in common), 'Same-seed common initial tensors differ')
            if arm[1]!='diagonal':
                if outer_reference is None: outer_reference=ck
                require(all(same(ck['module'][n],outer_reference['module'][n]) for n in OUTER_NAMES), 'Matched outer initial tensors differ')
            require(job['target_bank_schedule_sha256']==reference_job['target_bank_schedule_sha256'], 'Target/background/order schedule differs')
            for key in ('target_exposures','gold_tokens','input_positions','backbone_calls'):
                require(job['token_totals'][key]==reference_job['token_totals'][key], 'Marginal token/exposure budget differs')
            current=dict(schedule_sha256=job['schedule_sha256'],token_totals=job['token_totals'])
            if arm[0] in within: require(current==within[arm[0]], 'Within-regime full schedule/padding differs')
            within[arm[0]]=current
        reports.append(dict(seed=seed,arms=len(arms),common_initial_tensors_bitwise_equal=True,
            matched_outer_initial_tensors_bitwise_equal=True,common_initial_state_sha256=state_digest({n:reference['module'][n] for n in common}),
            within_regime=within,cross_regime_padding_equality_required=False,
            trainable_parameters_by_arm={f'{a[0]}_{a[1]}':j['trainable_parameters'] for a,j in sorted(arms.items())}))
    require(len(cache)<=1,'Formal arms use different training caches')
    return dict(complete=not missing,expected_conditions=18,observed_conditions=sum(map(len,groups.values())),
                missing=missing,seeds=reports,training_cache_sha256=next(iter(cache),None))


def audit_job(directory,manifest,inputs):
    require(manifest['protocol']==PROTOCOL and manifest['complete'] is True and manifest['backbone_unchanged'] is True,
            'Incomplete job or unfrozen backbone')
    source=source_check(directory,manifest,inputs); stage=manifest['stage']; smoke='smoke' in directory.name.lower()
    if stage=='prepare':
        require(inputs.sha(directory/'dataset.json')==manifest['result']['dataset_sha256'],'Dataset file differs')
        data=read_json(directory/'dataset.json'); data_proof=validate_dataset(data); caches={}
        for split,record in manifest['result']['caches'].items():
            packet=load_packet(inputs,str(directory/(split+'.pt')),record['cache_sha256']); rows=read_json(directory/(split+'.json'))
            inputs.sha(directory/(split+'.json'))
            require(packet['split']==split and rows==data[split] and packet['rows']==(rows['rows'] if split=='train' else rows), 'Dataset/cache split differs')
            caches[split+'.pt']=record['cache_sha256']
        require(set(caches)=={'train.pt','known.pt','dev.pt','confirm.pt'},'Missing prepared split')
        result=dict(data_proof=data_proof,feature_cache_sha256=caches)
    else:
        packet=load_packet(inputs,manifest['configuration']['cache'],manifest['cache_sha256'])
        names=['manifest.json']+(['training.jsonl','training_status.json','initial.pt','last.pt',f"step_{manifest['configuration']['updates']}.pt"]
              if stage=='train' else ['predictions.jsonl','summary.json']+(['pairs.json','encoded_payloads.pt','bank_events.pt'] if stage=='eval' else []))
        before={n:file_hash(directory/n) for n in names}
        if stage=='train': result=audit_train(directory,manifest,packet,inputs,smoke)
        elif stage=='eval': result=audit_eval(directory,manifest,packet,inputs)
        elif stage=='teacher': result=audit_teacher(directory,manifest,packet)
        else: raise ValueError('Unknown block stage')
        require(before=={n:file_hash(directory/n) for n in names},'Artifact changed during audit')
        result.update(artifact_sha256=before,input_hashes_unchanged=True,cache_sha256=manifest['cache_sha256'])
    return dict(run_dir=str(directory),stage=stage,scope='smoke' if smoke else 'formal',complete=True,
                configuration=manifest['configuration'],cache_split=None if stage=='prepare' else packet['split'],source=source,**result)


def inventory(jobs):
    models=Counter(); teachers=Counter(); prepare=0
    audited_training={Path(j['run_dir']).resolve():j for j in jobs if j['stage']=='train'}
    for job in jobs:
        if job['scope']!='formal': continue
        stage=job['stage']; cfg=job['configuration']
        if stage in ('train','eval'):
            factor=cfg if stage=='train' else job['lineage']
            if stage=='eval':
                parent=audited_training.get(Path(factor['parent_train']).resolve())
                require(parent is not None and parent['scope']=='formal'
                        and parent['final_checkpoint_sha256']==factor['checkpoint_sha256'],
                        'Formal evaluation parent is not a completed independently audited formal training checkpoint')
            cell=(factor['seed'],factor['regime'],factor['block_mode'],
                  'train' if stage=='train' else job['cache_split'], 'train' if stage=='train' else cfg['eval_part'])
            models[cell]+=1
        elif stage=='teacher': teachers[job['cache_split'],cfg['eval_part']]+=1
        elif stage=='prepare': prepare+=1
    expected={(s,*a,split,part) for s in SEEDS for a in ARMS for split,part in
              [('train','train'),('known','development'),('dev','development'),('known','confirmation'),('confirm','confirmation')]}
    et={('train','preflight'),('known','development'),('dev','development'),('known','confirmation'),('confirm','confirmation')}
    require(set(models)<=expected and all(n==1 for n in models.values()), 'Extra/duplicate formal model condition')
    require(set(teachers)<=et and all(n==1 for n in teachers.values()) and prepare<=1, 'Extra/duplicate teacher/prepare condition')
    return dict(complete=set(models)==expected and set(teachers)==et and prepare==1,
                expected_model_conditions=90,observed_model_conditions=len(models),expected_teacher_conditions=5,
                observed_teacher_conditions=len(teachers),prepare_jobs=prepare,
                missing_model_conditions=[list(x) for x in sorted(expected-set(models))],
                missing_teacher_conditions=[list(x) for x in sorted(et-set(teachers))])


def read_registration(path,inputs):
    if path is None: return None
    value=read_json(path)
    require(value['protocol']=='block-preregistration-v1'
            and value['selection_policy']=='all_18_final_checkpoints_no_selection'
            and value['teacher_preflight_passed'] is True and value['smoke_passed'] is True, 'Invalid preregistration')
    require(value['seeds']==list(SEEDS) and set(value['arms'])=={f'{r}_{m}' for r,m in ARMS}
            and len(value['arms'])==6 and type(value['updates'])is int and value['updates']==2048
            and type(value['data_seed'])is int and value['data_seed']==221042 and value['model_revision']==REVISION,
            'Registered design factors differ')
    require(inputs.sha(REPO/'docs/block_protocol.md')==value['protocol_document_sha256'], 'Registered protocol changed')
    sources=value['source_python_sha256']; require(isinstance(sources,dict) and sources, 'Missing registered sources')
    actual={str(p.relative_to(REPO)):inputs.sha(p) for p in sorted((REPO/'src/vera_mem').glob('*.py'))}
    require(sources==actual,'Registration does not bind the complete current runtime set')
    for relative,expected in sources.items():
        require(Path(relative).parts[:2]==('src','vera_mem') and len(Path(relative).parts)==3, 'Unsafe registered runtime path')
        require(inputs.sha(REPO/relative)==expected,'Registered runtime changed: '+relative)
    require(isinstance(value['plans_sha256'],dict) and value['plans_sha256'], 'Missing plan seal')
    require(set(value['feature_cache_sha256'])=={'train.pt','known.pt','dev.pt','confirm.pt'}, 'Missing feature cache seals')
    evidence=value['preflight_artifacts_sha256']; receipts=value['preflight_receipts']
    require(set(receipts)=={'features','teacher_train',*[f'smoke_{m}{suffix}' for m in MODES for suffix in ('','_eval')]},
            'Missing preflight/smoke receipts')
    manifest_names=set()
    for relative,expected in evidence.items():
        rel=Path(relative)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe preregistration evidence path')
        path=inputs.root/rel; require(inputs.sha(path)==expected, 'Sealed preflight evidence changed: '+relative)
        if rel.name=='manifest.json':
            name=rel.parent.name; manifest_names.add(name)
            require(name in receipts and read_json(path)==receipts[name], 'Preflight receipt differs from bound manifest')
    require(manifest_names==set(receipts), 'Preflight receipt lacks hashed original manifest')
    require(receipts['teacher_train']['result']['qualified'] is True
            and receipts['teacher_train']['result']['generation_calls']==128, 'Registered train teacher not qualified')
    return value


def bind_registration(directory,suite,plan,manifest,registration,registration_sha,inputs):
    require(suite.get('registration_sha256')==registration_sha
            and inputs.sha(directory/'registration.json')==registration_sha, 'Suite lacks the original preregistration snapshot')
    require(suite['plan_sha256'] in registration['plans_sha256'].values(), 'Suite plan is not registered')
    require({f'src/vera_mem/{k}':v for k,v in manifest['source_sha256'].items()}==registration['source_python_sha256'],
            'Formal runtime differs from sealed source set')
    path=manifest['configuration'].get('cache')
    if path: require(manifest['cache_sha256']==registration['feature_cache_sha256'].get(Path(path).name),'Formal cache not registered')
    return dict(registration_sha256=registration_sha,plan_sha256=suite['plan_sha256'],runtime_and_cache_bound=True)


def requires_registration(cfg,smoke):
    return not smoke and (cfg['stage'] in ('train','eval') or
                          (cfg['stage']=='teacher' and cfg['eval_part']!='preflight'))


def include_job(cfg,training_only=False,development_only=False):
    if training_only: return cfg['stage']=='train'
    return not development_only or cfg['eval_part']!='confirmation'


def run_audit(root,registration_path=None,training_only=False,require_final=False,development_only=False):
    require(not (training_only and development_only),'Choose only one partial-audit scope')
    inputs=Inputs(root); jobs=[]; pending=[]; excluded=[]; suites=0
    registration=read_registration(registration_path,inputs)
    registration_sha=inputs.sha(registration_path) if registration is not None else None
    for suite_path in sorted(inputs.root.glob('block_*/suite.json')):
        directory=suite_path.parent; markers=analysis_exclusions(directory,inputs.root)
        if markers: excluded.append(dict(path=str(directory),markers=markers)); continue
        suite=read_json(suite_path); inputs.sha(suite_path); require(suite['protocol']==PROTOCOL,'Wrong suite protocol')
        plan_path=directory/'plan.json'; plan=read_json(plan_path); inputs.sha(plan_path)
        # Suite copies are pretty-reformatted JSON; bind semantic plan content to
        # the named registered original and raw original SHA, not copy byte SHA.
        if registration is not None and suite.get('registration_sha256'):
            candidates=[name for name,h in registration['plans_sha256'].items() if h==suite['plan_sha256']]
            require(len(candidates)==1,'Ambiguous/unregistered suite plan SHA')
            original=REPO.parent/'plans'/candidates[0]
            require(inputs.sha(original)==suite['plan_sha256'] and read_json(original)==plan,'Registered original and suite plan differ')
        require(isinstance(plan,list) and plan and len({j['name'] for j in plan})==len(plan),'Invalid suite plan')
        launched={j['name']:j for j in suite['jobs']}
        require(len(launched)==len(suite['jobs']) and set(launched)<={p['name'] for p in plan},'Unplanned launched job')
        for spec in plan:
            require(spec['entry']=='block','Wrong entry'); cfg=configuration(spec['arguments']); stage=cfg['stage']
            if not include_job(cfg,training_only,development_only): continue
            child_dir=directory/spec['name']; markers=analysis_exclusions(child_dir,inputs.root)
            if markers: excluded.append(dict(path=str(child_dir),markers=markers)); continue
            job=launched.get(spec['name'])
            if not job or job.get('status')!='complete' or job.get('exit_code')!=0:
                pending.append(dict(path=str(child_dir),reason='Launcher incomplete; artifacts not read')); continue
            command=job['command']; start=command.index('vera_mem.block_run')+1
            require(command[start:-2]==spec['arguments'] and command[-2]=='--run-dir', 'Launcher arguments differ')
            mp=child_dir/'manifest.json'; require(inputs.sha(mp)==job['manifest_sha256'], 'Child manifest SHA differs'); manifest=read_json(mp)
            if manifest.get('complete') is not True:
                pending.append(dict(path=str(child_dir),reason='Child incomplete; artifacts not read')); continue
            require(manifest['configuration']==dict(cfg,run_dir=command[-1]) and manifest['stage']==stage,'Child planned/default configuration differs')
            smoke='smoke' in child_dir.name.lower(); binding=None
            if requires_registration(cfg,smoke):
                require(registration is not None,'Formal model run lacks preregistration')
                binding=bind_registration(directory,suite,plan,manifest,registration,registration_sha,inputs)
            result=audit_job(child_dir,manifest,inputs); result['registration_check']=binding; jobs.append(result)
            print(json.dumps(dict(audited=str(child_dir),stage=stage,scope=result['scope'])),flush=True)
        for relative,expected in suite['source_files_sha256'].items():
            require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'Unsafe source snapshot path')
            require(inputs.sha(directory/'source'/relative)==expected,'Suite source snapshot changed')
        suites+=1
        if not training_only and suite.get('complete') is not True: pending.append(dict(path=str(directory),reason='Suite incomplete'))
    matrix=training_matrix(jobs,inputs); coverage=inventory(jobs); scopes={}
    for scope in ('formal','smoke'):
        selected=[j for j in jobs if j['scope']==scope]
        scopes[scope]=dict(jobs=len(selected),stages=dict(Counter(j['stage'] for j in selected)),
            **{k:sum(j.get(k,0) for j in selected) for k in ('generation_calls','independent_interventions','restorations','real_group_write_events','actual_queries_replayed')})
    if require_final:
        require(not training_only and not development_only and not pending and matrix['complete'] and coverage['complete'],'Final audit requires all 96 formal children complete')
        formal=scopes['formal']; require(formal['generation_calls']==22048 and formal['real_group_write_events']==8064,'Formal generation/write budget differs')
        prepared=[j for j in jobs if j['stage']=='prepare' and j['scope']=='formal']
        require(prepared[0]['feature_cache_sha256']==registration['feature_cache_sha256'],'Registered data cache seal differs from prepare')
        require(scopes['smoke']['stages']=={'train':3,'eval':3} and scopes['smoke']['generation_calls']==39,
                'Final audit requires all three preflight smoke train/eval pairs')
    return dict(protocol=AUDIT_PROTOCOL,audited_at=datetime.now(timezone.utc).isoformat(),checks_passed=True,
        complete=bool(jobs) and not pending and (matrix['complete'] if training_only else coverage['complete']),
        final_scope_required=require_final,training_only=training_only,development_only=development_only,
        suites=suites,jobs=jobs,pending=pending,excluded=excluded,
        scopes=scopes,training_matrix=matrix,formal_inventory=coverage,registration_sha256=registration_sha,
        input_sha256={str(k):v for k,v in inputs.hashes.items()},cuda_initialized=torch.cuda.is_initialized(),
        audit_implementation_sha256={n:file_hash(REPO/'scripts'/n) for n in
            ('audit_block.py','audit_qkv.py','audit_reconstruction.py','replay_coldstart_banks.py')},
        limitations=['No backbone execution or frozen-feature extraction was repeated; saved query derivation is source-bound telemetry.',
            'All stored actual queries are replayed against complete keys, and selected values/writes are reconstructed from bank artifacts.',
            'Shared-state and frozen-backbone telemetry are source-bound records, not trusted execution attestation.',
            'Uniform slot membership is not the signed outer coefficient or a causal contribution estimate.',
            'Logged gradients do not establish per-coordinate coverage or useful generalization.',
            'Matched updates and marginal tokens do not imply matched FLOPs, padding, or convergence.',
            'Seeds and factorial evaluation views share underlying facts; they are not independent semantic samples.'])


def self_test():
    """Exercise real block CPU banks/evaluator with a fake language decoder."""
    import contextlib
    import copy
    import tempfile
    import unittest
    from types import SimpleNamespace
    from unittest.mock import patch
    from vera_mem import block_eval
    from vera_mem.block_data import dataset
    from vera_mem.context_distillation import ContextDistillationBackend

    class FakeBackend:
        _teacher_forward=ContextDistillationBackend._teacher_forward
        _TEACHER_TRANSIENT_FIELDS=ContextDistillationBackend._TEACHER_TRANSIENT_FIELDS
        def __init__(self,rows):
            self.device='cpu'; self.mode='none'; self.last_retrieval=None; self.retrieval_trace=[]
            self.vector_store=None; self.trace_retrieval=False
            self.questions={q:r for r in rows for q in r['questions']}
            self.contexts={s:r[FIELDS[w]] for r in rows for w,views in zip(r['worlds'],r['supports']) for s in views}
        @contextlib.contextmanager
        def disabled(self):
            previous=self.mode; self.mode='none'
            try: yield
            finally: self.mode=previous
        def generate(self,question,context=None):
            if self.mode=='vector_vera':
                for index in range(3):
                    q=torch.ones(1,1,self.vector_store.key_dim) if len(self.vector_store) else torch.zeros(1,1,self.vector_store.key_dim)
                    info=self.vector_store.search(q); self.last_retrieval=info
                    if self.trace_retrieval: self.retrieval_trace.append(dict(phase='prefill' if index==0 else 'decode',indices=info['indices'][:,-1]))
            return self.contexts[context] if context else self.questions[question]['a'],[71,72,73],.01
        def score(self,question,answer): return SimpleNamespace(nll_sum=1.2,tokens=3)

    class Tests(unittest.TestCase):
        def setUp(self):
            self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.serial=0
        def tearDown(self): self.temp.cleanup()
        def fixture(self,mode='block_outer',part='smoke',teacher=False):
            self.serial+=1; rows=dataset()['known']; rng=torch.Generator().manual_seed(921)
            packet=dict(split='known',rows=rows,worlds=list(FIELDS),slots3=torch.randn(16,5,2,3,5,generator=rng))
            model=BlockVeRA(5,6,rank=3,key_dim=4,seed=91042,block_mode=mode)
            model.fit_statistics(torch.randn(16,5,generator=rng),torch.randn(16,5,generator=rng))
            model.fit_value_statistics(torch.randn(16,5,generator=rng)); model.b.data.fill_(.3)
            cache=self.root/f'train{self.serial}.pt'; torch.save(dict(split='train'),cache)
            parent=self.root/f'parent{self.serial}'; parent.mkdir(); checkpoint=parent/'last.pt'
            torch.save(dict(protocol=PROTOCOL,architecture=model.configuration(),module=model.state_dict(),step=2,
                seed=91042,regime='static',block_mode=mode,cache_sha256=file_hash(cache)),checkpoint)
            (parent/'manifest.json').write_text(json.dumps(dict(protocol=PROTOCOL,complete=True,stage='train',cache_sha256=file_hash(cache),
                configuration=dict(cache=str(cache),updates=2,seed=91042,regime='static',block_mode=mode))))
            directory=self.root/f'smoke_eval{self.serial}'; directory.mkdir(); args=SimpleNamespace(run_dir=directory,eval_part=part)
            with patch.object(block_eval,'generate_tokens',lambda backend,*a,**k:backend.generate(*a,**k)):
                result=block_eval.teacher(FakeBackend(rows),packet,args) if teacher else block_eval.evaluate(FakeBackend(rows),model,packet,args)
            manifest=dict(configuration=dict(checkpoint=str(checkpoint),eval_part=part),checkpoint_sha256=file_hash(checkpoint),result=result)
            return directory,manifest,packet
        def replay(self,d,m,p):
            # Production feature shape is deliberately bypassed only for this
            # tiny fixture; all real writers, hashes and CPU routes are replayed.
            with patch.dict(globals(),load_packet=lambda inputs,path,expected:inputs.load(path,expected)):
                return audit_eval(d,m,p,Inputs(self.root))
        def raw_write(self,d,rows): (d/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        def test_three_modes_real_bank_replay(self):
            for mode in MODES:
                with self.subTest(mode=mode):
                    result=self.replay(*self.fixture(mode))
                    self.assertEqual(result['generation_calls'],13); self.assertEqual(result['actual_queries_replayed'],39)
                    self.assertEqual(result['real_group_write_events'],4); self.assertTrue(result['full_bank_group_winner_verified'])
        def test_changed_query_rejected_even_when_saved_indices_scores_consistent(self):
            d,m,p=self.fixture(); raw=read_jsonl(d/'predictions.jsonl'); raw[0]['trace'][0]['query']=[[-1.]*4]
            self.raw_write(d,raw)
            with self.assertRaisesRegex(ValueError,'winning group|cosine scores'): self.replay(d,m,p)
        def test_softmax_instead_of_uniform_rejected(self):
            d,m,p=self.fixture(); raw=read_jsonl(d/'predictions.jsonl'); raw[0]['trace'][0]['weights']=[[.2,.3,.5]]; self.raw_write(d,raw)
            with self.assertRaisesRegex(ValueError,'uniform'): self.replay(d,m,p)
        def test_omitted_or_shuffled_slot_order_rejected(self):
            for reverse in (False,True):
                d,m,p=self.fixture(); raw=read_jsonl(d/'predictions.jsonl'); event=raw[0]['trace'][0]
                event['indices'][0]=event['indices'][0][::-1] if reverse else event['indices'][0][:2]; self.raw_write(d,raw)
                with self.assertRaises(ValueError): self.replay(d,m,p)
        def test_writer_corruption_rejected(self):
            d,m,p=self.fixture(); path=d/'encoded_payloads.pt'; value=torch.load(path,weights_only=True); value['values'][0,1,0,0,0]+=.2; torch.save(value,path)
            with self.assertRaisesRegex(ValueError,'Writer re-encoding'): self.replay(d,m,p)
        def test_event_corruption_and_restore_hash_rejected(self):
            for name in ('values','restored_hash'):
                d,m,p=self.fixture(); path=d/'bank_events.pt'; value=torch.load(path,weights_only=True)
                if name=='values': value['events'][0]['values'][0,0]+=.2
                else: value['events'][0][name]='0'*64
                torch.save(value,path)
                with self.assertRaises(ValueError): self.replay(d,m,p)
        def test_missing_generation_cannot_make_pair_vacuously_true(self):
            d,m,p=self.fixture(); raw=read_jsonl(d/'predictions.jsonl'); self.raw_write(d,raw[:-1])
            with self.assertRaisesRegex(ValueError,'Missing generation'): self.replay(d,m,p)
        def test_empty_query_bypass_and_token_alignment(self):
            for field in ('query','token_id'):
                d,m,p=self.fixture(); raw=read_jsonl(d/'predictions.jsonl'); row=next(r for r in raw if r['condition']=='empty')
                row['trace'][0][field]=[[1.]*4] if field=='query' else 999; self.raw_write(d,raw)
                with self.assertRaises(ValueError): self.replay(d,m,p)
        def test_actual_teacher_no_memory_and_arithmetic(self):
            d,m,p=self.fixture(teacher=True); result=audit_teacher(d,m,p)
            self.assertEqual(result['generation_calls'],4); self.assertTrue(result['zero_recorded_memory_access'])
        def test_global_group_winner_not_merely_best_slot(self):
            db=BlockVectorDB(2,3,temperature=1.)
            db.write_group('spike',torch.tensor([[1.,0.],[-1.,0.],[-1.,0.]]),torch.ones(3,3),0)
            db.write_group('consistent',torch.tensor([[.7,.7141428]]*3),torch.ones(3,3),0)
            info=db.search(torch.tensor([[1.,0.]])); self.assertEqual(info['indices'].tolist(),[[3,4,5]])
            event=dict(query=[[1.,0.]],indices=info['indices'].tolist(),scores=info['scores'].tolist(),weights=info['weights'].tolist(),
                operator_weighting='uniform',weight_interpretation='uniform membership; not signed outer coefficients')
            self.assertEqual(replay_route(event,db)['selected_fact'],1)
        def test_new_defaults_and_duplicate_flags(self):
            cfg=configuration(['--stage','eval','--model','frozen']); self.assertEqual(cfg['seed'],91042); self.assertEqual(cfg['data_seed'],221042)
            with self.assertRaises(ValueError): configuration(['--stage','eval','--model','x','--seed','1','--seed','2'])
        def test_postregistration_teacher_cannot_bypass_barrier(self):
            for stage in ('train','eval','teacher'):
                for part in ('development','confirmation'):
                    cfg=dict(stage=stage,eval_part=part)
                    self.assertTrue(requires_registration(cfg,False)); self.assertFalse(requires_registration(cfg,True))
            self.assertFalse(requires_registration(dict(stage='teacher',eval_part='preflight'),False))
            self.assertFalse(requires_registration(dict(stage='prepare',eval_part='development'),False))
        def test_partial_scope_never_opens_confirmation_outputs(self):
            for stage in ('eval','teacher'):
                self.assertFalse(include_job(dict(stage=stage,eval_part='confirmation'),development_only=True))
                self.assertTrue(include_job(dict(stage=stage,eval_part='development'),development_only=True))
                self.assertFalse(include_job(dict(stage=stage,eval_part='development'),training_only=True))
            self.assertTrue(include_job(dict(stage='train',eval_part='development'),development_only=True))
        def test_training_log_outer_gradients_and_token_ledger(self):
            for mode in MODES:
                cfg=dict(updates=2,regime='rebind',seed=91042,block_mode=mode); rows=[]; h=hashlib.sha256()
                for step in range(2):
                    ep=episode(step,'rebind',91042); h.update(json.dumps(ep,sort_keys=True).encode())
                    d=None if step else dict(scope='captured gold-prefix prediction positions including EOS; parameter state before update',
                        prediction_positions=64,actual_read_slots=3,input_rms=1.,mixed_value_rms=1.,residual_rms=0.,target_fact_mass=.5,
                        outer_operator_mean_singular_values=None if mode=='diagonal' else [1.,0.,0.]+[0.]*61)
                    rows.append(dict(step=step+1,episode=ep,metrics=dict(loss=1.2,ce=1.,address=1.),
                        parameter_gradient_norms={n:1. for n in COMMON_NAMES|(OUTER_NAMES if mode!='diagonal' else set())},
                        gradient_norms=[1.]*len(parameter_groups(mode)[0]),input_lengths=[12]*16,gold_token_lengths=[4]*16,
                        read_diagnostics=d,elapsed_seconds=float(step)))
                status=dict(complete=True,updates=2,schedule_sha256=h.hexdigest(),elapsed_seconds=2.,
                    totals=dict(target_exposures=16,gold_tokens=128,input_positions=384,padded_input_positions=384,backbone_calls=2))
                self.assertEqual(audit_training_log(rows,status,cfg,True)['token_totals']['gold_tokens'],128)
                bad=copy.deepcopy(rows); bad[1]['gold_token_lengths'][0]+=1
                with self.assertRaisesRegex(ValueError,'totals'): audit_training_log(bad,status,cfg,True)
                if mode!='diagonal':
                    bad=copy.deepcopy(rows); del bad[0]['parameter_gradient_norms']['P_in.bias']
                    with self.assertRaisesRegex(ValueError,'coverage'): audit_training_log(bad,status,cfg,True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root',type=Path); parser.add_argument('--registration',type=Path)
    parser.add_argument('--output',type=Path)
    scope=parser.add_mutually_exclusive_group()
    scope.add_argument('--training-only',action='store_true')
    scope.add_argument('--development-only',action='store_true',help='Include preparation/train/smoke/development; never open confirmation outputs')
    parser.add_argument('--require-final',action='store_true'); parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv); torch.set_num_threads(4)
    if args.self_test: return self_test()
    if args.runs_root is None or args.output is None: parser.error('--runs-root and --output required')
    registration=args.registration
    if registration is None:
        candidate=REPO.parent/'plans/block_registration_20261006.json'
        if candidate.is_file(): registration=candidate
    output=args.output if args.output.suffix=='.json' else args.output/'audit.json'
    if output.exists(): parser.error('Refusing to overwrite audit: '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True); started=time.monotonic()
    try:
        result=run_audit(args.runs_root,registration,args.training_only,args.require_final,args.development_only); code=0
    except Exception as error:
        result=dict(protocol=AUDIT_PROTOCOL,checks_passed=False,complete=False,error=repr(error),
                    runs_root=str(args.runs_root),registration=str(registration) if registration else None,
                    training_only=args.training_only,development_only=args.development_only,final_scope_required=args.require_final); code=1
    result['elapsed_seconds']=time.monotonic()-started
    with output.open('x') as stream: json.dump(result,stream,indent=2,ensure_ascii=False); stream.write('\n')
    print(json.dumps(dict(output=str(output),checks_passed=result['checks_passed'],complete=result['complete'],
                         elapsed_seconds=result['elapsed_seconds'],error=result.get('error')),ensure_ascii=False),flush=True)
    return code


if __name__=='__main__': raise SystemExit(main())
