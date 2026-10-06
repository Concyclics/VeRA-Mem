"""Frozen online full-block writes with token-aligned sparse-route telemetry."""
from __future__ import annotations
from collections import defaultdict
import copy
import random
import time
import torch
from .context_distillation_run import clear_bank, append_jsonl
from .metrics import normalize_answer
from .run import json_write, tensor_digest
from .reconstruction_eval import changed_group
from .block_vera import BlockVectorDB

FIELDS={'A':'a','B':'b','C':'c','D':'d','SWAP':'swap_a'}
PHASES={'CC':(0,0),'HC':(1,0),'CH':(0,1),'HH':(1,1)}


def axes(split,part):
    if part=='preflight' and split=='train':return {'CC':['B']}
    if part=='smoke' and split=='known':return {'CC':['B']}
    if part=='development' and split=='known':return {'CC':['B','C','SWAP']}
    if part=='development' and split=='dev':return {'CC':['B','C']}
    if part=='confirmation' and split=='known':return {'CC':['D']}
    if part=='confirmation' and split=='confirm':return {p:['B','D'] for p in PHASES}
    raise ValueError('Invalid packet/part combination')


def match(pred,answer):return int(normalize_answer(pred)==normalize_answer(answer))


@torch.no_grad()
def generate_tokens(backend,question,context=None):
    prompt=backend.prompt_ids(question,context);ids=torch.tensor([prompt],device=backend.device)
    backend.injection_start=len(prompt)-1
    torch.cuda.synchronize();start=time.perf_counter()
    output=backend.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=32,
        do_sample=False,use_cache=True,pad_token_id=backend.tokenizer.eos_token_id)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    tokens=output[0,len(prompt):].cpu().tolist()
    return backend.tokenizer.decode(tokens,skip_special_tokens=True),tokens,seconds


@torch.no_grad()
def teacher(backend,packet,args):
    rows=packet['rows'];out=[];indexes=list(range(min(2,len(rows)) if args.eval_part=='smoke' else len(rows)))
    for phase,worlds in axes(packet['split'],args.eval_part).items():
        sv,qv=PHASES[phase]
        for world in ['A']+[w for w in worlds if w!='SWAP']:
            wi=packet['worlds'].index(world)
            for i in indexes:
                r=rows[i];question=r['questions'][qv];context=r['supports'][wi][sv]
                clear_bank(backend);backend.last_retrieval=None;backend.retrieval_trace=[]
                with backend._teacher_forward():
                    pred,tokens,seconds=generate_tokens(backend,question,context)
                    access=dict(mode=backend.mode,store_present=backend.vector_store is not None,
                        trace=list(backend.retrieval_trace),last_retrieval_present=backend.last_retrieval is not None)
                if access['mode']!='none' or access['store_present'] or access['trace'] or access['last_retrieval_present']:
                    raise AssertionError('Teacher accessed VDB')
                row=dict(id=r['id'],phase=phase,world=world,question=question,context=context,
                    answer=r[FIELDS[world]],prediction=pred,em=match(pred,r[FIELDS[world]]),
                    generation_tokens=len(tokens),generated_token_ids=tokens,generation_seconds=seconds,
                    teacher_bypasses_memory=True,memory_access=access)
                out.append(row);append_jsonl(args.run_dir/'predictions.jsonl',[row])
    groups=defaultdict(list)
    for r in out:groups[r['phase']+'_'+r['world']].append(r['em'])
    result=dict(complete=True,groups={k:dict(count=len(v),correct=sum(v),em=sum(v)/len(v)) for k,v in groups.items()},
        qualified=all(all(v) for v in groups.values()),generation_calls=len(out),
        generation_tokens=sum(x['generation_tokens'] for x in out),generation_seconds=sum(x['generation_seconds'] for x in out))
    json_write(args.run_dir/'summary.json',result);return result


def shuffle(db,seed):
    state=db.snapshot();n=len(db.fact_ids);order=list(range(n));random.Random(seed).shuffle(order)
    perm=list(range(n))
    for a,b in zip(order,order[1:]+order[:1]):perm[a]=b
    values=state['store']['values'].reshape(n,3,db.value_dim)
    state['store']['values']=values[perm].reshape(-1,db.value_dim).clone()
    return BlockVectorDB.from_snapshot(state),perm


@torch.no_grad()
def read(backend,module,db,row,world,phase,condition,role,case_id,trigger,output):
    question=row['questions'][PHASES[phase][1]];answer=row[FIELDS[world]];before=db.hash()
    clear_bank(backend);backend.mode='vector_vera';backend.vector_vera=module;backend.vector_store=db
    backend.trace_retrieval=True;backend.retrieval_trace=[];db.record_routes=True;db.clear_routes()
    pred,tokens,seconds=generate_tokens(backend,question)
    trace=[];target=db.fact_ids.index(row['id']) if row['id'] in db.fact_ids else -1
    if len(backend.retrieval_trace)!=len(db.route_log) or len(db.route_log)!=len(tokens):
        raise AssertionError('Each generated token requires one actual CPU route')
    for token,observed,raw in zip(tokens,backend.retrieval_trace,db.route_log):
        indices=raw['indices'][:,-1];weights=raw['weights'][:,-1];scores=raw['scores'][:,-1]
        if indices.tolist()!=observed['indices'].tolist():raise AssertionError('Backend/store routes differ')
        ii=indices[0].tolist();ww=weights[0].tolist()
        masses=[sum(w for j,w in zip(ii,ww) if target>=0 and j==3*target+s) for s in range(3)]
        trace.append(dict(phase=observed['phase'],token_id=token,indices=indices.tolist(),
            weights=weights.tolist(),scores=scores.tolist(),query=raw['queries'][:,-1].tolist(),
            operator_weighting='uniform',weight_interpretation='uniform membership; not signed outer coefficients',target_slot_mass=masses,target_fact_mass=sum(masses)))
    backend.trace_retrieval=False;db.record_routes=False;db.clear_routes();score=backend.score(question,answer)
    if db.hash()!=before:raise AssertionError('Read mutated memory')
    hits=[int(target>=0 and any(j//3==target for j in t['indices'][0])) for t in trace]
    first=trace[0];idx=first['indices'][0];weights=first['weights'][0]
    r1=int(bool(idx) and target>=0 and idx[max(range(len(idx)),key=weights.__getitem__)]//3==target)
    result=dict(id=row['id'],case_id=case_id,trigger_world=trigger,phase=phase,world=world,
        condition=condition,role=role,question=question,answer=answer,prediction=pred,em=match(pred,answer),
        generated_token_ids=tokens,generation_tokens=len(tokens),generation_seconds=seconds,budget_hit=len(tokens)>=32,
        nll_sum=score.nll_sum,answer_tokens=score.tokens,bank_hash=before,bank_slots=len(db),trace=trace,
        first_fact_recall_at_read=hits[0],first_fact_recall_at_1=r1,decode_hits=sum(hits[1:]),decode_queries=len(hits)-1,
        first_target_slot_mass=first['target_slot_mass'],first_target_fact_mass=first['target_fact_mass'],actual_read_slots=len(idx),injection_input='full_selected_value_block',
        sparse_values_transferred_per_token=len(idx)*module.rank)
    append_jsonl(output/'predictions.jsonl',[result]);return result


@torch.no_grad()
def evaluate(backend,module,packet,args):
    if packet['split']=='train' or args.eval_part=='preflight':raise ValueError('Student eval requires separate evaluation packet')
    module.eval().requires_grad_(False);module.clear_feature_bank();frozen=tensor_digest(module)
    rows=packet['rows'];features=packet['slots3'].to(backend.device)
    keys=module.encode_key(features).cpu();values=module.encode_value(features).cpu()
    torch.save(dict(keys=keys,values=values,fact_ids=[r['id'] for r in rows],worlds=packet['worlds']),args.run_dir/'encoded_payloads.pt')
    indexes=list(range(min(2,len(rows)) if args.eval_part=='smoke' else len(rows)))
    predictions=[];pairs=[];events=[];bases={}
    for phase,worlds in axes(packet['split'],args.eval_part).items():
        sv,qv=PHASES[phase];base=module.new_store()
        for i,r in enumerate(rows):base.write_group(r['id'],keys[i,0,sv],values[i,0,sv],0)
        bases[phase]=base.snapshot();needed=set(indexes)|{(i+1)%len(rows) for i in indexes};initial={}
        for i in sorted(needed):
            r=read(backend,module,base,rows[i],'A',phase,'real','initial',rows[i]['id'],'A',args.run_dir)
            initial[i]=r;predictions.append(r)
        for world in worlds:
            wi=packet['worlds'].index(world)
            for i in indexes:
                row=rows[i];db=BlockVectorDB.from_snapshot(base.snapshot());before=db.snapshot()
                db.write_group(row['id'],keys[i,wi,sv],values[i,wi,sv],1);changed_group(before,db.snapshot(),i,3)
                event=dict(phase=phase,world=world,index=i,id=row['id'],before_hash=base.hash(),after_hash=db.hash(),
                    keys=keys[i,wi,sv],values=values[i,wi,sv],timestamp=1)
                def get(bank,item,w,condition,role):
                    result=read(backend,module,bank,item,w,phase,condition,role,row['id'],world,args.run_dir)
                    predictions.append(result);return result
                updated=get(db,row,world,'real','updated');neighbor=(i+1)%len(rows)
                local=get(db,rows[neighbor],'A','real','locality')
                db.write_group(row['id'],keys[i,0,sv],values[i,0,sv],2);changed_group(before,db.snapshot(),i,3)
                if not torch.equal(db.keys,base.keys) or not torch.equal(db.values,base.values):raise AssertionError('Restore differs')
                event['restored_hash']=db.hash();restored=get(db,row,'A','real','restored')
                pair=dict(id=row['id'],phase=phase,world=world,a_em=initial[i]['em'],updated_em=updated['em'],
                    pair_em=initial[i]['em']*updated['em'],restored_em=restored['em'],update_restore_em=updated['em']*restored['em'],
                    all_three_em=initial[i]['em']*updated['em']*restored['em'],locality_equal=int(local['prediction']==initial[neighbor]['prediction']),
                    locality_correct=local['em'],locality_joint=local['em']*initial[neighbor]['em'],shuffle_em=None,empty_em=None)
                if world!='SWAP':
                    db=BlockVectorDB.from_snapshot(base.snapshot());db.write_group(row['id'],keys[i,wi,sv],values[i,wi,sv],1)
                    wrong,perm=shuffle(db,131042+i);event.update(value_permutation=perm,shuffle_hash=wrong.hash())
                    pair['shuffle_em']=get(wrong,row,world,'shuffle','control')['em']
                    pair['empty_em']=get(module.new_store(),row,world,'empty','control')['em']
                pairs.append(pair);events.append(event)
    if tensor_digest(module)!=frozen:raise AssertionError('Online shared parameters changed')
    groups={}
    for p in pairs:
        groups.setdefault(p['phase']+'_'+p['world'],[]).append(p)
    summaries={}
    for name,items in groups.items():
        g={'count':len(items)}
        for key in ('a_em','updated_em','pair_em','restored_em','update_restore_em','all_three_em','locality_equal','locality_correct','locality_joint','shuffle_em','empty_em'):
            vs=[x[key] for x in items if x[key] is not None];g[key]=sum(vs)/len(vs) if vs else None
        summaries[name]=g
    json_write(args.run_dir/'pairs.json',pairs)
    torch.save(dict(protocol='block-bank-events-v1',bases=bases,events=events),args.run_dir/'bank_events.pt')
    result=dict(protocol='block-online-cpu-v1',complete=True,architecture=module.configuration(),groups=summaries,
        shared_parameters_unchanged=True,shared_parameter_sha256=frozen,bank_facts=len(rows),slots_per_fact=3,
        resident_bytes=base.resident_bytes(),generation_calls=len(predictions),
        generation_tokens=sum(x['generation_tokens'] for x in predictions),generation_seconds=sum(x['generation_seconds'] for x in predictions),
        answer_scoring_tokens=sum(x['answer_tokens'] for x in predictions),independent_interventions=len(events),
        restorations=len(events),real_group_write_events=2*len(events))
    json_write(args.run_dir/'summary.json',result);clear_bank(backend);return result
