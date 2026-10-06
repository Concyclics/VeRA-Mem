"""Frozen whole-fact writes, free generation, restoration and locality controls."""
from __future__ import annotations
from collections import defaultdict
import copy
import json
import random
import torch
from .context_distillation_run import clear_bank, append_jsonl
from .metrics import normalize_answer
from .run import json_write, tensor_digest
from .reconstruction_vera import GroupedVectorDB

FIELDS={'A':'a','B':'b','C':'c','D':'d','SWAP':'swap_a'}
PHASES={'CC':(0,0),'HC':(1,0),'CH':(0,1),'HH':(1,1)}


def phases(packet,part):
    if part in ('smoke','preflight'):return ['CC']
    if packet['split']=='known':return ['CC','HC','CH','HH'] if part=='development' else ['CC']
    if packet['split']=='dev' and part=='development':return ['CC']
    if packet['split']=='confirm' and part=='confirmation':return list(PHASES)
    raise ValueError('Evaluation packet/part mismatch')


def worlds(packet,part,phase):
    if part in ('smoke','preflight'):return ['B']
    if packet['split']=='known' and phase=='CC':return ['B','C','SWAP'] if part=='development' else ['D']
    return ['B']


def match(pred,answer):return int(normalize_answer(pred)==normalize_answer(answer))


@torch.no_grad()
def teacher(backend,packet,args):
    rows=packet['rows'];out=[]
    indexes=range(min(2,len(rows))) if args.eval_part=='smoke' else range(len(rows))
    for phase in phases(packet,args.eval_part):
        sv,qv=PHASES[phase]
        ws=['A']+worlds(packet,args.eval_part,phase)
        for w in ws:
            if w=='SWAP':continue
            wi=packet['worlds'].index(w)
            for i in indexes:
                r=rows[i];q=r['questions'][qv];context=r['supports'][wi][sv]
                clear_bank(backend);backend.last_retrieval=None;backend.retrieval_trace=[]
                with backend._teacher_forward():
                    pred,tokens,secs=backend.generate(q,context=context,max_new_tokens=32)
                    access=dict(mode=backend.mode,store_present=backend.vector_store is not None,trace=list(backend.retrieval_trace),last_retrieval_present=backend.last_retrieval is not None)
                if access['mode']!='none' or access['store_present'] or access['trace'] or access['last_retrieval_present']:raise AssertionError('Teacher accessed memory')
                row=dict(id=r['id'],phase=phase,world=w,answer=r[FIELDS[w]],prediction=pred,
                         em=match(pred,r[FIELDS[w]]),generation_tokens=tokens,generation_seconds=secs,
                         context=context,question=q,teacher_bypasses_memory=True,memory_access=access)
                out.append(row);append_jsonl(args.run_dir/'predictions.jsonl',[row])
    groups=defaultdict(list)
    for r in out:groups[r['phase']+'_'+r['world']].append(r['em'])
    summary={k:dict(correct=sum(v),count=len(v),em=sum(v)/len(v)) for k,v in groups.items()}
    result=dict(complete=True,groups=summary,qualified=all(v['em']>=.95 for v in summary.values()),
                generation_calls=len(out),generation_tokens=sum(r['generation_tokens'] for r in out),
                generation_seconds=sum(r['generation_seconds'] for r in out))
    json_write(args.run_dir/'summary.json',result);return result


def populate(module,rows,keys,values,view):
    db=module.new_store()
    for i,r in enumerate(rows):db.write_group(r['id'],keys[i,0,view],values[i,0,view],0)
    return db


def changed_group(before,after,index,slots):
    """No other payload or timestamp may change in a group replacement."""
    a,b=before['store'],after['store']
    if before['slots']!=slots or after['slots']!=slots or a['config']!=b['config']:raise AssertionError('Group configuration changed')
    if not 0<=index<len(before['fact_ids']):raise AssertionError('Invalid target group')
    for name in ('keys','values'):
        if a[name].shape!=b[name].shape or a[name].dtype!=b[name].dtype:raise AssertionError('Payload shape/dtype changed')
    times=b['timestamps'][index*slots:(index+1)*slots]
    if len(times)!=slots or len(set(times))!=1 or times[0]<=max(a['timestamps'][index*slots:(index+1)*slots]):raise AssertionError('Partial or stale group timestamps')
    mask=torch.ones(a['keys'].shape[0],dtype=torch.bool)
    mask[index*slots:(index+1)*slots]=False
    for name in ('keys','values'):
        if not torch.equal(a[name][mask].contiguous().view(torch.uint8),b[name][mask].contiguous().view(torch.uint8)):raise AssertionError('Non-target payload changed')
    if before['fact_ids']!=after['fact_ids'] or a['ids']!=b['ids']:raise AssertionError('Record identity changed')
    if [x for j,x in enumerate(a['timestamps']) if bool(mask[j])] != [x for j,x in enumerate(b['timestamps']) if bool(mask[j])]:raise AssertionError('Non-target timestamp changed')


def shuffled(db,seed):
    """Derange whole value groups, preserving keys and within-group slot order."""
    state=db.snapshot();n=len(state['fact_ids']);order=list(range(n))
    if n<2:raise ValueError('Shuffle needs at least two facts')
    random.Random(seed).shuffle(order)
    perm=list(range(n))
    for left,right in zip(order,order[1:]+order[:1]):perm[left]=right
    values=state['store']['values'].reshape(n,db.slots,db.value_dim)
    state['store']['values']=values[perm].reshape(-1,db.value_dim).clone()
    out=GroupedVectorDB.from_snapshot(state)
    if not torch.equal(out.keys,db.keys):raise AssertionError('Shuffle changed keys')
    return out,perm


@torch.no_grad()
def read(backend,module,db,row,world,phase,condition,role,case_id,output,trigger_world=None):
    _,qv=PHASES[phase];question=row['questions'][qv];answer=row[FIELDS[world]]
    before=db.hash();clear_bank(backend);backend.mode='vector_vera';backend.vector_vera=module;backend.vector_store=db
    backend.trace_retrieval=True;backend.retrieval_trace=[];db.record_routes=True;db.clear_routes()
    prediction,tokens,seconds=backend.generate(question,max_new_tokens=32)
    trace=[dict(phase=x['phase'],indices=x['indices'].tolist()) for x in backend.retrieval_trace]
    if len(trace)!=len(db.route_log):raise AssertionError('Generation/CPU route trace mismatch')
    for trace_row,raw in zip(trace,db.route_log):
        if raw['indices'][:,-1].tolist()!=trace_row['indices']:raise AssertionError('CPU trace IDs differ')
        trace_row['weights']=raw['weights'][:,-1].tolist();trace_row['scores']=raw['scores'][:,-1].tolist()
    backend.trace_retrieval=False;db.record_routes=False;db.clear_routes()
    score=backend.score(question,answer)
    if db.hash()!=before:raise AssertionError('Read mutated bank')
    target=db.fact_ids.index(row['id']) if row['id'] in db.fact_ids else -1
    hit=[]
    for x in trace:
        hit.append(target>=0 and any(int(j)//module.slots==target for j in x['indices'][0]))
    result=dict(id=row['id'],case_id=case_id,phase=phase,world=world,condition=condition,role=role,
                trigger_world=trigger_world or world,question=question,answer=answer,prediction=prediction,em=match(prediction,answer),
                generation_tokens=tokens,generation_seconds=seconds,budget_hit=tokens>=32,
                nll_sum=score.nll_sum,answer_tokens=score.tokens,bank_hash=before,bank_slots=len(db),
                trace=trace,first_fact_recall_at_4=int(hit[0]) if hit else 0,
                first_fact_recall_at_1=int(bool(trace and trace[0]['indices'][0] and target>=0 and trace[0]['indices'][0][0]//module.slots==target)),
                decode_hits=sum(hit[1:]),decode_queries=max(0,len(hit)-1))
    append_jsonl(output/'predictions.jsonl',[result]);return result


@torch.no_grad()
def evaluate(backend,module,packet,args):
    if packet['split']=='train' or args.eval_part=='preflight':raise ValueError('Use separate known packet for student probes; preflight is teacher only')
    module.eval().requires_grad_(False);module.clear_feature_bank();frozen=tensor_digest(module)
    rows=packet['rows'];features=packet['slots'+str(module.slots)].to(backend.device)
    keys=module.encode_key(features).cpu();values=module.encode_value(features).cpu()
    torch.save(dict(keys=keys,values=values,fact_ids=[r['id'] for r in rows],worlds=packet['worlds']),args.run_dir/'encoded_payloads.pt')
    indexes=list(range(min(2,len(rows)))) if args.eval_part=='smoke' else list(range(len(rows)))
    predictions=[];pairs=[];events=[];bank_snapshots={}
    for phase in phases(packet,args.eval_part):
        sv,qv=PHASES[phase];base=populate(module,rows,keys,values,sv)
        bank_snapshots[phase]=base.snapshot();a_reads={}
        # Each baseline A read is generated once and reused only by exact ID/phase.
        needed=set(indexes)
        if packet['split']=='known' and phase=='CC':needed|={(i+1)%len(rows) for i in indexes}
        for i in sorted(needed):
            rr=read(backend,module,base,rows[i],'A',phase,'real','initial',rows[i]['id'],args.run_dir)
            a_reads[i]=rr;predictions.append(rr)
        for world in worlds(packet,args.eval_part,phase):
            wi=packet['worlds'].index(world)
            for i in indexes:
                r=rows[i];db=GroupedVectorDB.from_snapshot(base.snapshot())
                before=db.snapshot();db.write_group(r['id'],keys[i,wi,sv],values[i,wi,sv],1)
                changed_group(before,db.snapshot(),i,module.slots)
                event=dict(phase=phase,world=world,id=r['id'],index=i,before_hash=base.hash(),after_hash=db.hash(),
                           keys=keys[i,wi,sv],values=values[i,wi,sv],timestamp=1)
                rr=read(backend,module,db,r,world,phase,'real','updated',r['id'],args.run_dir);predictions.append(rr)
                pair=dict(id=r['id'],phase=phase,world=world,a_em=a_reads[i]['em'],updated_em=rr['em'],
                          pair_em=a_reads[i]['em']*rr['em'],restored_em=None,locality_equal=None,locality_correct=None)
                if packet['split']=='known' and phase=='CC':
                    neighbor=(i+1)%len(rows)
                    local=read(backend,module,db,rows[neighbor],'A',phase,'real','locality',r['id'],args.run_dir,trigger_world=world);predictions.append(local)
                    pair['locality_equal']=int(local['prediction']==a_reads[neighbor]['prediction'])
                    pair['locality_correct']=local['em']
                    pair['locality_joint']=local['em']*a_reads[neighbor]['em']
                    db.write_group(r['id'],keys[i,0,sv],values[i,0,sv],2)
                    changed_group(base.snapshot(),db.snapshot(),i,module.slots)
                    for name in ('keys','values'):
                        if not torch.equal(db.snapshot()['store'][name],base.snapshot()['store'][name]):raise AssertionError('Restoration payload mismatch')
                    event['restored_hash']=db.hash()
                    restored=read(backend,module,db,r,'A',phase,'real','restored',r['id'],args.run_dir,trigger_world=world);predictions.append(restored)
                    pair['restored_em']=restored['em'];pair['update_restore_em']=rr['em']*restored['em']
                    pair['all_three_em']=a_reads[i]['em']*rr['em']*restored['em']
                    if world!='SWAP':
                        updated=GroupedVectorDB.from_snapshot(base.snapshot());updated.write_group(r['id'],keys[i,wi,sv],values[i,wi,sv],1)
                        wrong,perm=shuffled(updated,111042+i)
                        event['value_permutation']=perm;event['shuffle_hash']=wrong.hash()
                        for condition,control in (('shuffle',wrong),('empty',module.new_store())):
                            pr=read(backend,module,control,r,world,phase,condition,'control',r['id'],args.run_dir)
                            predictions.append(pr);pair[condition+'_em']=pr['em']
                events.append(event);pairs.append(pair)
        torch.save(dict(base_banks=bank_snapshots,updates=events),args.run_dir/'bank_events.pt')
        print(json.dumps(dict(phase_complete=phase,predictions=len(predictions),pairs=len(pairs))),flush=True)
    if tensor_digest(module)!=frozen:raise AssertionError('Online shared weights changed')
    groups=defaultdict(list)
    for pair in pairs:groups[pair['phase']+'_'+pair['world']].append(pair)
    summaries={}
    for key,items in groups.items():
        entry=dict(count=len(items))
        for metric in ('a_em','updated_em','pair_em','restored_em','update_restore_em','all_three_em','locality_equal','locality_correct','locality_joint','shuffle_em','empty_em'):
            vv=[p[metric] for p in items if p.get(metric) is not None]
            if vv:entry[metric]=sum(vv)/len(vv)
        summaries[key]=entry
    result=dict(protocol='reconstruction-cpu-vdb-v1',complete=True,groups=summaries,shared_parameters_unchanged=True,shared_parameter_sha256=frozen,
                generation_calls=len(predictions),generation_tokens=sum(r['generation_tokens'] for r in predictions),
                generation_seconds=sum(r['generation_seconds'] for r in predictions),single_group_updates=len(events),
                independent_interventions=len(events),restorations=sum('restored_hash' in e for e in events),
                real_group_write_events=len(events)+sum('restored_hash' in e for e in events),
                update_count_scope='single_group_updates counts independent A-to-X interventions; real_group_write_events also counts A restoration, excluding initial/control-bank materialization',
                bank_facts=len(rows),slots_per_fact=module.slots,resident_bytes=base.resident_bytes(),architecture=module.configuration(),
                audit_scope='All bank reads immutable; only target group updated; restoration payload exact. No LM replay claim.')
    json_write(args.run_dir/'pairs.json',pairs);json_write(args.run_dir/'summary.json',result)
    return result
