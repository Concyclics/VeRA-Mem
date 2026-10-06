"""Strict CPU/text audit of QKV runs; seal development selection before confirmation.

Does not deserialize tensors, execute a tokenizer/model, or use GPUs. Tensor banks,
checkpoints and frozen source are hash-bound; audit_qkv.py independently replays them.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata

RUN_PROTOCOL='qkv-attention-memory-v1'
PROTOCOL='qkv-audit-v1'
SELECTION_PROTOCOL='qkv-selection-v1'
DATA_PROTOCOL='qkv-binding-episodes-v1'
SEEDS=(81042,81043,81044)
MAIN_ARMS=('static_flat','static_grouped','rebind_flat','rebind_grouped')
ARMS=MAIN_ARMS+('rebind_grouped_additive',)
PHASES={'CC':(0,0),'HC':(1,0),'CH':(0,1),'HH':(1,1)}
FIELDS={'A':'a','B':'b','C':'c','D':'d','SWAP':'swap_a'}
PAIR_METRICS=('a_em','updated_em','pair_em','restored_em','update_restore_em','all_three_em',
              'locality_equal','locality_correct','locality_joint','shuffle_em','empty_em')
GATE_POLICY=dict(ab_pair_min=12/16,c_update_restore_min=12/16,c_real_minus_shuffle_min=.5,
    c_real_minus_empty_min=.5,c_locality_joint_min=15/16,new_c_updated_min=8/16,
    d_update_restore_min=12/16,d_real_minus_shuffle_min=.5,d_real_minus_empty_min=.5,
    d_locality_joint_min=15/16,confirm_d_updated_min=8/16,
    selection_order=['maximum minimum-seed known C update_restore_em','maximum minimum-seed new-entity C updated_em',
                     'minimum trainable_parameters','minimum bytes_per_fact','fixed main arm order'],
    seeds=list(SEEDS),arm_order=list(MAIN_ARMS),diagnostic_excluded=['rebind_grouped_additive'])
ARTIFACT_NAMES={
 'training':('manifest.json','training_status.json','training.jsonl','initial.pt','last.pt','step_1024.pt','step_2048.pt'),
 'evaluation':('manifest.json','summary.json','predictions.jsonl','pairs.json','encoded_payloads.pt','bank_events.pt'),
 'teacher':('manifest.json','summary.json','predictions.jsonl')}


def require(ok,message):
    if not ok:raise ValueError(message)


def loads(text):
    return json.loads(text,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))


def read_json(path):return loads(Path(path).read_text())
def lines(path):return [loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def valid_hash(x):return isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) is not None

def norm(value):
    require(isinstance(value,str),'Prediction/answer is not text')
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c
        for c in unicodedata.normalize('NFKC',value).casefold()).split())

def number(x,name,integer=False):
    require(type(x) in ((int,) if integer else (int,float)) and math.isfinite(x) and x>=0,'Invalid '+name)
    return x

def same(a,b,message):
    if isinstance(a,dict):
        require(isinstance(b,dict) and set(a)==set(b),message+' keys')
        for k in a:same(a[k],b[k],message+'/'+k)
    elif isinstance(a,list):
        require(isinstance(b,list) and len(a)==len(b),message+' list')
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,message+'/'+str(i))
    elif type(a) in (int,float) and type(b) in (int,float):
        require(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8),message)
    else:require(a==b,message)

def close(a,b,message):
    require(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b)
        and math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-6),message)

def artifacts(directory,names):
    out={}
    for name in names:
        p=Path(directory)/name;require(p.is_file(),'Missing artifact '+str(p));out[name]=sha(p)
    return out

def architecture_identity(a,regime=None):
    for k,v in dict(slots=3,train_B=True,rank=64,key_dim=64,top_k=4,in_features=9728,out_features=2560,
                    writer_mode='masked_mean',value_mlp_hidden=0,qkv_version=1).items():
        require(a.get(k)==v,'Architecture differs: '+k)
    require(a.get('seed') in SEEDS and a.get('read_mode') in ('flat','grouped') and a.get('readout') in ('vera','additive'),'Unknown QKV architecture')
    same(a.get('temperature'),.05,'Wrong retrieval temperature')
    if regime is None:return None,a['seed']
    require(regime in ('static','rebind'),'Unknown binding regime')
    arm=regime+'_'+a['read_mode']+('_additive' if a['readout']=='additive' else '')
    require(arm in ARMS,'Unregistered QKV arm')
    return arm,a['seed']


def episode(step,regime,seed):
    """Independent pure-Python reconstruction of the preregistered random streams."""
    def rng(*parts):return random.Random(int(digest([DATA_PROTOCOL,seed,*parts])[:16],16))
    epoch,batch=divmod(step,8);order=list(range(64));rng('targets',epoch).shuffle(order)
    targets=order[batch*8:batch*8+8]
    background=rng('background',epoch,batch).sample([i for i in range(64) if i not in targets],8)
    bank=targets+background; rng('bank-order',epoch,batch).shuffle(bank)
    payloads=list(range(128))
    if regime=='rebind':rng('payload-map',epoch).shuffle(payloads)
    mapping=[payloads[2*i:2*i+2] for i in range(64)]
    return dict(protocol=DATA_PROTOCOL,step=step,seed=seed,regime=regime,epoch=epoch,batch_index=batch,
        targets=targets,background=background,bank_entities=bank,local_targets=[bank.index(i) for i in targets],
        mapping=mapping,a_payload_indices=[mapping[i][0] for i in bank],b_payload_indices=[mapping[i][1] for i in bank])


def audit_training(directory,manifest,train):
    cfg=manifest['configuration'];status=read_json(directory/'training_status.json')
    same(status,manifest['result'],'Training status/manifest mismatch')
    require(status.get('complete') is True and status.get('updates')==cfg.get('updates')==2048,'Formal training budget must be 2048')
    arm,seed=architecture_identity(status['architecture'],cfg['regime'])
    require(seed==cfg['seed'] and status['regime']==cfg['regime'],'Training seed/regime differs')
    for k in ('read_mode','readout'):require(cfg[k]==status['architecture'][k],'Training mode mismatch')
    require(status['trainable_parameters']==2034368,'Wrong trainable parameter count')
    same(status['learning_rates'],[1e-4,3e-4,.005,3e-4],'Learning rates differ')
    raw=lines(directory/'training.jsonl');require(len(raw)==2048,'Wrong training log length')
    h=hashlib.sha256();common=hashlib.sha256();targets=Counter();payloads=Counter();background=Counter();resident=Counter()
    totals=dict(target_exposures=0,gold_tokens=0,input_positions=0,padded_input_positions=0,backbone_calls=0)
    prompt_lengths={};answer_lengths={};last_elapsed=0
    for step,row in enumerate(raw):
        ep=episode(step,cfg['regime'],seed);same(ep,row['episode'],'Episode schedule mismatch')
        require(row['step']==step+1,'Wrong training step')
        h.update(json.dumps(ep,sort_keys=True).encode())
        common.update(json.dumps({k:ep[k] for k in ('targets','background','bank_entities','local_targets')},sort_keys=True).encode())
        targets.update(ep['targets']);background.update(ep['background'])
        answer_ids=[ep[key][j] for key in ('a_payload_indices','b_payload_indices') for j in ep['local_targets']]
        payloads.update(answer_ids)
        # Physical per-sample bank residency: eight A and eight B banks; B replaces only its target.
        resident.update({p:16 for p in ep['a_payload_indices']})
        for j in ep['local_targets']:resident[ep['a_payload_indices'][j]]-=1;resident[ep['b_payload_indices'][j]]+=1
        metrics=row['metrics']
        for key in ('ce','address','loss'):number(metrics[key],key)
        close(metrics['loss'],metrics['ce']+.2*metrics['address'],'Loss is not CE + .2 address')
        require(len(row['gradient_norms'])==4,'Wrong optimizer group count')
        for value in row['gradient_norms']:number(value,'gradient norm')
        require(isinstance(row['parameter_gradient_norms'],dict) and bool(row['parameter_gradient_norms']),'Missing parameter gradients')
        for value in row['parameter_gradient_norms'].values():
            if value is not None:number(value,'parameter gradient norm')
        elapsed=number(row['elapsed_seconds'],'elapsed_seconds');require(elapsed>=last_elapsed,'Elapsed time decreases');last_elapsed=elapsed
        il=row['input_lengths'];gl=row['gold_token_lengths']
        require(len(il)==len(gl)==16,'Token ledger must contain sixteen independent sequences')
        for entity,aid,n,g in zip(ep['targets']*2,answer_ids,il,gl):
            number(n,'input length',True);number(g,'gold length',True)
            require(n>g>0,'Invalid sequence token counts')
            if entity in prompt_lengths:require(prompt_lengths[entity]==n-g,'Same question has inconsistent prompt token length')
            if aid in answer_lengths:require(answer_lengths[aid]==g,'Same payload has inconsistent answer token length')
            prompt_lengths[entity]=n-g;answer_lengths[aid]=g
        totals['target_exposures']+=8;totals['gold_tokens']+=sum(gl);totals['input_positions']+=sum(il)
        totals['padded_input_positions']+=16*max(il);totals['backbone_calls']+=1
    require(set(targets)==set(range(64)) and set(targets.values())=={256},'Target entity exposures differ')
    require(set(payloads)==set(range(128)) and set(payloads.values())=={256},'Supervised payload exposures differ')
    require(sum(resident.values())==2048*16*16,'Physical bank residency total differs')
    same(totals,status['totals'],'Training token/exposure ledger differs from totals')
    require(status['schedule_sha256']==h.hexdigest(),'Schedule digest mismatch')
    require(number(status['elapsed_seconds'],'Training process seconds')>=last_elapsed,'Training elapsed shorter than final step')
    hashes=artifacts(directory,ARTIFACT_NAMES['training'])
    return dict(kind='training',arm=arm,seed=seed,regime=cfg['regime'],run_dir=str(directory),architecture=status['architecture'],
        cache_sha256=manifest['cache_sha256'],checkpoint_sha256=hashes['last.pt'],updates=2048,
        trainable_parameters=2034368,schedule_sha256=h.hexdigest(),common_schedule_sha256=common.hexdigest(),
        costs=dict(totals,training_process_seconds=status['elapsed_seconds']),
        exposure=dict(target_counts=[targets[i] for i in range(64)],supervised_payload_counts=[payloads[i] for i in range(128)],
            background_entity_counts=[background[i] for i in range(64)],physical_payload_bank_residency=[resident[i] for i in range(128)],
            residency_scope='All sixteen independent sample banks per step, not supervised examples or actual retrieved slots'),
        token_ledger=dict(prompt_lengths=[prompt_lengths[i] for i in range(64)],answer_with_eos_lengths=[answer_lengths[i] for i in range(128)],
            scope='Recomputed from logged sequence lengths; tokenizer execution is not independently repeated'),artifacts=hashes)


def expected_axes(split,part):
    if split=='train' and part=='preflight':return {'CC':['B']}
    if split=='known' and part=='development':return {'CC':['B','C','SWAP']}
    if split=='dev' and part=='development':return {'CC':['B','C']}
    if split=='known' and part=='confirmation':return {'CC':['D']}
    if split=='confirm' and part=='confirmation':return {p:['B','D'] for p in PHASES}
    raise ValueError('Unexpected packet/part combination')


def validate_tokens(row):
    ids=row['generated_token_ids'];require(isinstance(ids,list) and 0<len(ids)<=32,'Invalid generated token list')
    require(all(type(i) is int and i>=0 for i in ids),'Invalid generated token ID')
    require(row['generation_tokens']==len(ids),'Generated token count mismatch')
    number(row['generation_seconds'],'generation seconds')
    return ids


def validate_trace(row,index,mode):
    ids=validate_tokens(row);trace=row['trace'];require(len(trace)==len(ids),'Trace must align one-to-one with generated tokens')
    width=0 if row['condition']=='empty' else 3 if mode=='grouped' else 4
    hits=[];ranks=[]
    for position,(token,t) in enumerate(zip(ids,trace)):
        require(t['token_id']==token and t['phase']==('prefill' if position==0 else 'decode'),'Token/trace position mismatch')
        for k in ('indices','weights','scores'):require(isinstance(t[k],list) and len(t[k])==1,'Unexpected route batch dimension')
        ii,ww,ss=(t[k][0] for k in ('indices','weights','scores'))
        require(len(ii)==len(ww)==len(ss)==width and len(set(ii))==width,'Wrong sparse read width')
        require(all(type(j) is int and 0<=j<48 for j in ii),'Invalid slot index')
        if width:
            if mode=='grouped':require(ii==[3*(ii[0]//3)+j for j in range(3)],'Grouped route is not one ordered three-slot fact')
            else:require(all(a>=b-2e-6 for a,b in zip(ss,ss[1:])),'Flat scores are not ranked')
            require(all(type(x) in (int,float) and math.isfinite(x) and -1.0001<=x<=1.0001 for x in ss),'Invalid cosine score')
            z=[math.exp((s-max(ss))/.05) for s in ss];den=sum(z)
            for w,e in zip(ww,z):number(w,'route weight');close(w,e/den,'Route weight differs from score softmax')
            close(sum(ww),1.,'Route weights not normalized')
        masses=[sum(w for j,w in zip(ii,ww) if j==3*index+s) for s in range(3)]
        require(len(t['target_slot_mass'])==3,'Missing three-slot mass')
        for actual,expected in zip(t['target_slot_mass'],masses):close(actual,expected,'Target slot mass mismatch')
        close(t['target_fact_mass'],sum(masses),'Target fact mass mismatch')
        hits.append(int(any(j//3==index for j in ii)))
        ranks.append(int(bool(ii) and ii[max(range(len(ii)),key=ww.__getitem__)]//3==index))
    same(row['first_fact_recall_at_read'],hits[0],'First actual-read recall mismatch')
    same(row['first_fact_recall_at_1'],ranks[0],'First maximum-weight recall mismatch')
    same(row['decode_hits'],sum(hits[1:]),'Decode hit mismatch');same(row['decode_queries'],len(ids)-1,'Decode count mismatch')
    same(row['actual_read_slots'],width,'Actual read width mismatch')
    for a,b in zip(row['first_target_slot_mass'],trace[0]['target_slot_mass']):close(a,b,'First slot mass mismatch')
    require(len(row['first_target_slot_mass'])==3,'First slot mass length mismatch')
    close(row['first_target_fact_mass'],trace[0]['target_fact_mass'],'First fact mass mismatch')


def prediction_statistics(rows,byid=None,train_payloads=()):
    total=len(rows);gold=sum(r['answer_tokens'] for r in rows)
    words=[(norm(r['prediction']).split(),norm(r['answer']).split()) for r in rows]
    out=dict(count=total,correct=sum(r['em'] for r in rows),em=sum(r['em'] for r in rows)/total if total else None,
        token_nll=sum(r['nll_sum'] for r in rows)/gold if gold else None,nll_includes_eos=False,
        full_answer_containment=sum((' '+norm(r['answer'])+' ') in (' '+norm(r['prediction'])+' ') for r in rows),
        first_word_correct=sum(bool(p) and bool(a) and p[0]==a[0] for p,a in words),
        exactly_three_normalized_words=sum(len(p)==3 for p,a in words),
        normalized_word_count_histogram=dict(sorted(Counter(str(len(p)) for p,a in words).items(),key=lambda kv:int(kv[0]))),
        first_fact_recall_at_read=sum(r['first_fact_recall_at_read'] for r in rows)/total if total else None,
        first_fact_recall_at_1=sum(r['first_fact_recall_at_1'] for r in rows)/total if total else None,
        decode_hits=sum(r['decode_hits'] for r in rows),decode_queries=sum(r['decode_queries'] for r in rows),
        budget_hits=sum(r['budget_hit'] for r in rows),generation_calls=total,
        generation_tokens=sum(r['generation_tokens'] for r in rows),generation_seconds=sum(r['generation_seconds'] for r in rows),answer_scoring_tokens=gold)
    if byid is not None:
        novel=[r for r in rows if r['world'] in ('C','D')]
        if novel:
            train_set={norm(p) for p in train_payloads};first=[];decode=[];all_mass=[];touched=0;correct=0
            for r in novel:
                source=byid[r['id']];j=source['provenance']['edited_word_index'];require(type(j) is int and 0<=j<3,'Invalid edited word provenance')
                pw=norm(r['prediction']).split();aw=norm(r['answer']).split()
                correct+=int(len(pw)>j and len(aw)>j and pw[j]==aw[j])
                ms=[t['target_slot_mass'][j] for t in r['trace']];first.append(ms[0]);decode.extend(ms[1:]);all_mass.extend(ms)
                touched+=int(any(3*list(byid).index(r['id'])+j in t['indices'][0] for t in r['trace']))
            out['novel_content']=dict(count=len(novel),edited_word_position_correct=correct,
                old_own_a=sum(norm(r['prediction'])==norm(byid[r['id']]['a']) for r in novel),
                old_own_b=sum(norm(r['prediction'])==norm(byid[r['id']]['b']) for r in novel),
                any_train_payload=sum(norm(r['prediction']) in train_set for r in novel) if train_payloads else None,
                edited_slot_ever_read=touched,edited_slot_first_mass_mean=sum(first)/len(first),
                edited_slot_decode_mass_mean=sum(decode)/len(decode) if decode else None,
                edited_slot_all_token_mass_mean=sum(all_mass)/len(all_mass),
                alignment_scope='Mass of source edited_word_index at actual generation token positions; no gold-forced token/word alignment or tokenizer decoding is inferred')
    return out


def audit_evaluation(directory,manifest,rows,split,regime=None,train_payloads=()):
    summary=read_json(directory/'summary.json');same(summary,manifest['result'],'Eval summary/manifest mismatch')
    require(summary.get('protocol')=='qkv-online-cpu-v1' and summary.get('complete') is True,'Unknown/incomplete QKV evaluation')
    require(summary.get('shared_parameters_unchanged') is True and valid_hash(summary.get('shared_parameter_sha256')),'Shared parameter immutability missing')
    arm,seed=architecture_identity(summary['architecture'],regime)
    require(len(rows)==summary['bank_facts']==16 and summary['slots_per_fact']==3,'Bank must contain sixteen three-slot facts')
    resident=dict(key_bytes=12288,value_bytes=12288,total_bytes=24576);same(summary['resident_bytes'],resident,'Resident bytes differ')
    axes=expected_axes(split,manifest['configuration']['eval_part']);raw=lines(directory/'predictions.jsonl');offset=0
    byid={r['id']:r for r in rows};require(len(byid)==16,'Duplicate fact IDs');ids=list(byid)
    pairs=[];updated_rows=defaultdict(list);initial_byphase={};updated_bygroup={}
    def take(index,world,phase,condition,role,case,trigger):
        nonlocal offset
        require(offset<len(raw),'Missing prediction');r=raw[offset];offset+=1
        for k,v in dict(id=ids[index],world=world,phase=phase,condition=condition,role=role,case_id=case,trigger_world=trigger).items():
            require(r.get(k)==v,'Prediction identity/order differs: '+k)
        source=rows[index]
        require(r['question']==source['questions'][PHASES[phase][1]] and r['answer']==source[FIELDS[world]],'Question/answer provenance differs')
        same(r['em'],int(norm(r['prediction'])==norm(r['answer'])),'Raw EM mismatch')
        for k in ('answer_tokens','bank_slots','decode_hits','decode_queries'):number(r[k],k,True)
        number(r['nll_sum'],'NLL sum');require(r['answer_tokens']>0,'Empty scored answer')
        require(r['budget_hit']==(r['generation_tokens']>=32),'Wrong budget-hit flag')
        require(valid_hash(r['bank_hash']) and r['bank_slots']==(0 if condition=='empty' else 48),'Invalid bank metadata')
        validate_trace(r,index,summary['architecture']['read_mode']);return r
    for phase,worlds in axes.items():
        initial=[take(i,'A',phase,'real','initial',ids[i],'A') for i in range(16)];initial_byphase[phase]=initial
        require(len({r['bank_hash'] for r in initial})==1,'Initial reads use different banks')
        for world in worlds:
            updated_bygroup[phase+'_'+world]=[]
            for i in range(16):
                u=take(i,world,phase,'real','updated',ids[i],world);ni=(i+1)%16
                l=take(ni,'A',phase,'real','locality',ids[i],world);z=take(i,'A',phase,'real','restored',ids[i],world)
                require(u['bank_hash']!=initial[i]['bank_hash'] and l['bank_hash']==u['bank_hash'] and z['bank_hash']!=u['bank_hash'],'Invalid update/locality/restore bank hashes')
                pair=dict(id=ids[i],phase=phase,world=world,a_em=initial[i]['em'],updated_em=u['em'],pair_em=initial[i]['em']*u['em'],
                    restored_em=z['em'],update_restore_em=u['em']*z['em'],all_three_em=initial[i]['em']*u['em']*z['em'],
                    locality_equal=int(l['prediction']==initial[ni]['prediction']),locality_correct=l['em'],locality_joint=l['em']*initial[ni]['em'],shuffle_em=None,empty_em=None)
                if world!='SWAP':
                    for condition in ('shuffle','empty'):
                        c=take(i,world,phase,condition,'control',ids[i],world);pair[condition+'_em']=c['em']
                        if condition=='shuffle':require(c['bank_hash']!=u['bank_hash'],'Shuffle bank unchanged')
                pairs.append(pair);updated_rows[phase+'_'+world].append(u)
                updated_bygroup[phase+'_'+world].append(dict(id=ids[i],a_prediction=initial[i]['prediction'],prediction=u['prediction'],a_em=initial[i]['em'],updated_em=u['em']))
    require(offset==len(raw),'Unexpected/duplicate predictions')
    same(pairs,read_json(directory/'pairs.json'),'Saved pairs differ from raw')
    grouped=defaultdict(list)
    for p in pairs:grouped[p['phase']+'_'+p['world']].append(p)
    groups={k:dict(count=len(v),**{m:sum(p[m] for p in v)/len(v) if v[0][m] is not None else None for m in PAIR_METRICS}) for k,v in grouped.items()}
    same(groups,summary['groups'],'Summary groups differ from recomputed pairs')
    costs=prediction_statistics(raw)
    for k in ('generation_calls','generation_tokens','generation_seconds','answer_scoring_tokens'):same(costs[k],summary[k],'Evaluation cost differs: '+k)
    for k,v in dict(independent_interventions=len(pairs),restorations=len(pairs),real_group_write_events=2*len(pairs)).items():same(summary[k],v,'Write count differs: '+k)
    changed={k:dict(count=len(v),changed=sum(x['a_prediction']!=x['prediction'] for x in v),
        changed_both_wrong=sum(x['a_prediction']!=x['prediction'] and not x['a_em'] and not x['updated_em'] for x in v)) for k,v in updated_bygroup.items()}
    return dict(kind='evaluation',arm=arm,seed=seed,run_dir=str(directory),split=split,part=manifest['configuration']['eval_part'],
        architecture=summary['architecture'],cache_sha256=manifest['cache_sha256'],checkpoint_sha256=manifest['checkpoint_sha256'],groups=groups,
        updated_diagnostics={k:prediction_statistics(v,byid,train_payloads) for k,v in updated_rows.items()},changed_output=changed,
        costs=costs,independent_interventions=len(pairs),restoration_writes=len(pairs),real_group_write_events=2*len(pairs),
        bank_facts=16,bytes_per_fact=1536,artifacts=artifacts(directory,ARTIFACT_NAMES['evaluation']))


def audit_teacher(directory,manifest,rows,split):
    summary=read_json(directory/'summary.json');same(summary,manifest['result'],'Teacher summary mismatch')
    require(summary.get('complete') is True,'Incomplete teacher')
    raw=lines(directory/'predictions.jsonl');axes=expected_axes(split,manifest['configuration']['eval_part'])
    require(len(rows)==(64 if split=='train' else 16),'Wrong teacher fact count')
    expected=[(ph,w,r,sv,qv) for ph,worlds in axes.items() for sv,qv in [PHASES[ph]]
        for w in ['A']+[x for x in worlds if x!='SWAP'] for r in rows]
    require(len(raw)==len(expected),'Teacher generation grid incomplete');groups=defaultdict(list);indexed={}
    for r,(phase,world,source,sv,qv) in zip(raw,expected):
        require((r['id'],r['phase'],r['world'])==(source['id'],phase,world),'Teacher identity/order mismatch')
        require(r['question']==source['questions'][qv] and r['context']==source['supports'][source['worlds'].index(world)][sv]
            and r['answer']==source[FIELDS[world]],'Teacher provenance mismatch')
        require(r['teacher_bypasses_memory'] is True,'Teacher memory bypass missing')
        same(r['memory_access'],dict(mode='none',store_present=False,trace=[],last_retrieval_present=False),'Teacher accessed memory')
        validate_tokens(r);em=int(norm(r['prediction'])==norm(r['answer']));same(r['em'],em,'Teacher EM mismatch')
        groups[phase+'_'+world].append(em);indexed[phase,world,r['id']]=em
    result={k:dict(count=len(v),correct=sum(v),em=sum(v)/len(v)) for k,v in groups.items()}
    same(summary['groups'],result,'Teacher group mismatch');require(summary['qualified']==all(all(v) for v in groups.values()),'Teacher qualification mismatch')
    costs=dict(generation_calls=len(raw),generation_tokens=sum(r['generation_tokens'] for r in raw),generation_seconds=sum(r['generation_seconds'] for r in raw))
    for k,v in costs.items():same(summary[k],v,'Teacher cost mismatch '+k)
    paired={}
    for phase,worlds in axes.items():
        for world in worlds:
            if world=='SWAP':continue
            eligible=[r['id'] for r in rows if indexed[phase,'A',r['id']] and indexed[phase,world,r['id']]]
            paired[phase+'_'+world]=dict(count=len(rows),both_correct=len(eligible),paired_em=len(eligible)/len(rows),eligible_ids=eligible)
    return dict(kind='teacher',run_dir=str(directory),split=split,part=manifest['configuration']['eval_part'],groups=result,
        qualified=summary['qualified'],paired=paired,cache_sha256=manifest['cache_sha256'],costs=costs,
        artifacts=artifacts(directory,ARTIFACT_NAMES['teacher']))


def completeness(records,final=False):
    kinds=['train','known_development','dev_development']+(['known_confirmation','confirm_confirmation'] if final else [])
    expected={(a,s,k) for a in ARMS for s in SEEDS for k in kinds};found=defaultdict(list)
    for r in records:
        if r['kind'] not in ('training','evaluation'):continue
        key=(r['arm'],r['seed'],'train' if r['kind']=='training' else r['split']+'_'+r['part'])
        if key in expected:found[key].append(r['run_dir'])
    return dict(expected=len(expected),observed=len(found),complete=set(found)==expected and all(len(v)==1 for v in found.values()),
        missing=[list(k) for k in sorted(expected-set(found))],duplicates=[dict(condition=list(k),run_dirs=v) for k,v in found.items() if len(v)>1])


def seed_checks(s):
    return dict(ab_pair=s['ab_pair']>=12/16,c_update_restore=s['c_update_restore']>=12/16,
        c_real_minus_shuffle=s['c_real']-s['c_shuffle']>=.5,c_real_minus_empty=s['c_real']-s['c_empty']>=.5,
        c_locality_joint=s['c_locality_joint']>=15/16,new_c_updated=s['new_c_updated']>=8/16)


def rank_candidates(candidates):
    eligible=[c for c in candidates if c['eligible']]
    eligible.sort(key=lambda c:(-c['minimum_seed_c_update_restore'],-c['minimum_seed_new_c_updated'],c['trainable_parameters'],c['bytes_per_fact'],MAIN_ARMS.index(c['arm'])))
    return eligible[0]['arm'] if eligible else None


def development_selection(records):
    barrier=completeness(records);require(barrier['complete'],'Development barrier incomplete or duplicate: '+json.dumps(barrier))
    teachers=[r for r in records if r['kind']=='teacher' and r['split']=='train' and r['part']=='preflight']
    require(len(teachers)==1 and set(teachers[0]['groups'])=={'CC_A','CC_B'} and all(g['count']==g['correct']==64 for g in teachers[0]['groups'].values()),'Require unique 128/128 training teacher preflight')
    index={(r['arm'],r['seed'],'train' if r['kind']=='training' else r['split']+'_'+r['part']):r for r in records if r['kind'] in ('training','evaluation')}
    candidates=[]
    for arm in MAIN_ARMS:
        per_seed=[]
        for seed in SEEDS:
            train=index[arm,seed,'train'];known=index[arm,seed,'known_development'];new=index[arm,seed,'dev_development']
            b,c,n=known['groups']['CC_B'],known['groups']['CC_C'],new['groups']['CC_C']
            require(b['count']==c['count']==n['count']==16,'Development gate denominator must be sixteen')
            require(train['trainable_parameters']==2034368 and known['bytes_per_fact']==new['bytes_per_fact']==1536,'Parameter/byte budget differs')
            s=dict(seed=seed,ab_pair=b['pair_em'],c_update_restore=c['update_restore_em'],c_real=c['updated_em'],c_shuffle=c['shuffle_em'],c_empty=c['empty_em'],c_locality_joint=c['locality_joint'],new_c_updated=n['updated_em'])
            s['checks']=seed_checks(s);s['passed']=all(s['checks'].values());per_seed.append(s)
        candidates.append(dict(arm=arm,eligible=all(s['passed'] for s in per_seed),per_seed=per_seed,
            minimum_seed_c_update_restore=min(s['c_update_restore'] for s in per_seed),minimum_seed_new_c_updated=min(s['new_c_updated'] for s in per_seed),
            trainable_parameters=2034368,bytes_per_fact=1536))
    return dict(policy=loads(json.dumps(GATE_POLICY)),selected_arm=rank_candidates(candidates),candidates=candidates)


def selection_artifacts(records):
    selected=[r for r in records if r['kind']=='training' or (r['kind']=='evaluation' and r['part']=='development')
              or (r['kind']=='teacher' and r['split']=='train' and r['part']=='preflight')]
    return sorted([dict(run_dir=r['run_dir'],kind=r['kind'],arm=r.get('arm'),seed=r.get('seed'),split=r.get('split'),part=r.get('part'),artifacts=dict(r['artifacts'])) for r in selected],key=lambda r:r['run_dir'])


def validate_selection(value,protocol_path):
    """Pure JSON contract for the confirmation launcher; does not open run artifacts."""
    require(isinstance(value,dict) and value.get('protocol')==SELECTION_PROTOCOL and value.get('sealed') is True
            and value.get('development_complete') is True,'Not a sealed QKV development selection')
    require(value.get('protocol_sha256')==sha(protocol_path),'Sealed protocol SHA differs')
    same(value.get('development_barrier'),dict(expected=45,observed=45,complete=True,missing=[],duplicates=[]),'Wrong development barrier')
    decision=value['decision'];require(set(decision)=={'policy','selected_arm','candidates'},'Invalid decision schema');same(decision['policy'],GATE_POLICY,'Changed selection policy')
    candidates=decision['candidates'];require(isinstance(candidates,list) and [c['arm'] for c in candidates]==list(MAIN_ARMS),'Candidates must be the four registered VeRA arms')
    numeric=('ab_pair','c_update_restore','c_real','c_shuffle','c_empty','c_locality_joint','new_c_updated')
    for c in candidates:
        require([s['seed'] for s in c['per_seed']]==list(SEEDS),'Candidate seeds differ')
        require(c['trainable_parameters']==2034368 and c['bytes_per_fact']==1536,'Candidate budget differs')
        for s in c['per_seed']:
            for k in numeric:
                n=number(s[k],k);require(n<=1 and math.isclose(n*16,round(n*16),abs_tol=1e-8),'Gate score must use sixteen-case integer denominator')
            require(s['c_update_restore']<=s['c_real'],'Joint update/restore cannot exceed updated accuracy')
            same(s['checks'],seed_checks(s),'Candidate checks differ');require(s['passed'] is all(s['checks'].values()),'Candidate seed pass flag differs')
        same(c['minimum_seed_c_update_restore'],min(s['c_update_restore'] for s in c['per_seed']),'Wrong worst-seed known C score')
        same(c['minimum_seed_new_c_updated'],min(s['new_c_updated'] for s in c['per_seed']),'Wrong worst-seed new C score')
        require(c['eligible'] is all(s['passed'] for s in c['per_seed']),'Candidate eligibility differs')
    require(decision['selected_arm']==rank_candidates(candidates),'Selected arm violates frozen ranking')
    evidence=value['development_artifacts'];require(isinstance(evidence,list) and len(evidence)==46,'Selection requires 45 model conditions plus one teacher preflight')
    require(len({e['run_dir'] for e in evidence})==46 and all(isinstance(e['run_dir'],str) and e['run_dir'] for e in evidence),'Duplicate/empty evidence run directory')
    conditions=[];teacher=0
    for e in evidence:
        require(e['kind'] in ARTIFACT_NAMES and set(e['artifacts'])==set(ARTIFACT_NAMES[e['kind']]),'Wrong evidence artifact inventory')
        require(all(valid_hash(v) for v in e['artifacts'].values()),'Invalid evidence SHA')
        if e['kind']=='teacher':
            teacher+=1;require(e['split']=='train' and e['part']=='preflight' and e['arm'] is None and e['seed'] is None,'Unexpected teacher evidence')
        else:
            require(e['arm'] in ARMS and e['seed'] in SEEDS,'Unexpected model evidence')
            if e['kind']=='training':require(e['split'] is None and e['part'] is None,'Training evidence cannot carry evaluation metadata')
            key='train' if e['kind']=='training' else e['split']+'_'+e['part'];conditions.append((e['arm'],e['seed'],key))
    expected={(a,s,k) for a in ARMS for s in SEEDS for k in ('train','known_development','dev_development')}
    require(teacher==1 and len(conditions)==45 and set(conditions)==expected,'Evidence condition inventory differs')
    same(value.get('teacher_preflight'),dict(count=128,correct=128,qualified=True),'Teacher preflight receipt missing')
    return value


def make_selection(report,protocol_path):
    require(not any(r.get('part')=='confirmation' for r in report['records']+report['pending']),'Cannot select after any confirmation child exists')
    require(not report['pending'],'Cannot seal while a non-smoke QKV child is unfinished')
    decision=development_selection(report['records'])
    require(all(sum(r['kind']=='teacher' and r['split']==split and r['part']=='development' for r in report['records'])==1 for split in ('known','dev')),'Development teacher results must be saved before selection')
    value=dict(protocol=SELECTION_PROTOCOL,sealed=True,development_complete=True,created_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(protocol_path),script_sha256=sha(__file__),decision=decision,development_barrier=completeness(report['records']),
        development_artifacts=selection_artifacts(report['records']),teacher_preflight=dict(count=128,correct=128,qualified=True))
    validate_selection(value,protocol_path);return value


def final_gate(report,selection,protocol_path):
    validate_selection(selection,protocol_path)
    same(selection['decision'],development_selection(report['records']),'Sealed development decision changed')
    same(selection['development_artifacts'],selection_artifacts(report['records']),'Sealed development artifacts changed')
    arm=selection['decision']['selected_arm'];entries=[]
    for seed in SEEDS:
        got={r['split']:r for r in report['records'] if r['kind']=='evaluation' and r['part']=='confirmation' and r['arm']==arm and r['seed']==seed}
        if set(got)!={'known','confirm'}:entries.append(dict(seed=seed,complete=False));continue
        d,n=got['known']['groups']['CC_D'],got['confirm']['groups']['CC_D'];require(d['count']==n['count']==16,'Final gate denominator must be sixteen')
        checks=dict(d_update_restore=d['update_restore_em']>=12/16,d_real_minus_shuffle=d['updated_em']-d['shuffle_em']>=.5,
            d_real_minus_empty=d['updated_em']-d['empty_em']>=.5,d_locality_joint=d['locality_joint']>=15/16,new_d_updated=n['updated_em']>=8/16)
        eligible=[]
        for split in ('known','confirm'):
            q=got[split].get('teacher_qualification');g=q['paired'].get('CC_D') if q else None
            eligible.append(bool(g and g['count']==g['both_correct']==16))
        entries.append(dict(seed=seed,complete=True,checks=checks,teacher_qualified=all(eligible),passed=all(checks.values()) and all(eligible),
            d_update_restore=d['update_restore_em'],d_real=d['updated_em'],d_shuffle=d['shuffle_em'],d_empty=d['empty_em'],d_locality_joint=d['locality_joint'],new_d_updated=n['updated_em']))
    complete=completeness(report['records'],True)['complete'] and not report['pending'] and report.get('teacher_scope',{}).get('complete',False)
    passed=arm is not None and all(e.get('passed',False) for e in entries)
    return dict(selected_arm=arm,per_seed=entries,all_formal_results_complete=complete,selected_architecture_gate_passed=passed,eligible_to_expand=complete and passed,
        note='No confirmation reselection, seed selection, or additive substitution. Teacher-ineligible conditions retain full denominators but cannot authorize expansion.')


def discover(root):
    root=Path(root).resolve();require(root.is_dir(),'Runs root does not exist')
    return [p for p in sorted(root.rglob('manifest.json')) if any(x.startswith('qkv_') for x in (root.name,*p.relative_to(root).parts[:-1]))
        and not any(x in ('source','source_snapshot','source_snapshots','.git') for x in p.relative_to(root).parts)]

def ignored_reason(path,root):
    if any(re.search(r'(^|[_-])smoke($|[_-])',x) for x in (root.name,*path.relative_to(root).parts[:-1])):return 'Explicitly named smoke job/suite'
    for p in path.parents:
        if p==root.parent:break
        if (p/'analysis_excluded.json').is_file():
            evidence=read_json(p/'analysis_excluded.json');require(evidence.get('reason'),'Missing exclusion reason');return 'Explicit exclusion: '+evidence['reason']
    return None


def audit_sources(directory,manifest):
    source=manifest.get('source_sha256');require(isinstance(source,dict) and bool(source),'Missing source hashes')
    for key in ('qkv_run.py','qkv_eval.py','qkv_data.py','qkv_vera.py'):require(valid_hash(source.get(key)),'Missing QKV source hash '+key)
    # Standard suite snapshot is its child run directory's sibling; do not bind to mutable live code.
    snapshot=directory.parent/'source'/'src'/'vera_mem'
    require(snapshot.is_dir(),'Missing immutable source snapshot for '+str(directory))
    for name,value in source.items():
        require(name==Path(name).name and valid_hash(value),'Invalid declared source entry')
        require((snapshot/name).is_file() and sha(snapshot/name)==value,'Frozen source differs: '+name)
    return dict(source_sha256=source,source_snapshot=str(snapshot))


def load_registration(path,protocol_path):
    value=read_json(path)
    require(value.get('protocol')=='qkv-registration-v1' and value.get('teacher_preflight_passed') is True
            and value.get('smoke_passed') is True,'Invalid formal registration')
    require(value['protocol_document_sha256']==sha(protocol_path),'Registered protocol document changed')
    same(value['arms'],list(ARMS),'Registered arms differ');same(value['seeds'],list(SEEDS),'Registered seeds differ')
    require(value['updates']==2048 and value['model_revision']=='cdbee75f17c01a7cc42f958dc650907174af0554','Registered budget/model differs')
    source=value['source_python_sha256']
    require(len(source)==44 and all(k.startswith('src/vera_mem/') and Path(k).name.endswith('.py') and valid_hash(v) for k,v in source.items()),'Registration must bind 44 runtime source files')
    require(bool(value['plans_sha256']) and all(valid_hash(v) for v in value['plans_sha256'].values()),'Invalid registered plan hashes')
    return value


def registered_run(directory,manifest,registration):
    """Bind completed formal child to registered code, plan and frozen data."""
    stage=manifest['stage'];part=manifest['configuration'].get('eval_part')
    if stage=='prepare':
        same(manifest['result'],registration['feature_result'],'Prepared feature result differs from registration')
        return dict(scope='registered feature-result hash; own pre-telemetry source snapshot')
    require(manifest['cache_sha256'] in {v['cache_sha256'] for v in registration['feature_result']['caches'].values()},'Run cache not registered')
    if stage=='teacher' and part=='preflight':
        suffix='/'+directory.parent.name+'/'+directory.name+'/manifest.json'
        receipts=[v for k,v in registration['preflight_artifacts_sha256'].items() if k.endswith(suffix)]
        require(receipts==[sha(directory/'manifest.json')],'Preflight manifest differs from registered receipt')
        return dict(scope='registered preflight receipt; own pre-diagnostics source snapshot')
    expected={Path(k).name:v for k,v in registration['source_python_sha256'].items()}
    same(manifest['source_sha256'],expected,'Formal run runtime differs from registered source')
    protocol_copy=directory.parent/'source'/'docs'/'qkv_protocol.md'
    require(sha(protocol_copy)==registration['protocol_document_sha256'],'Formal source snapshot protocol differs')
    plan_path=directory.parent/'plan.json';plan_hash=sha(plan_path)
    require(plan_hash in registration['plans_sha256'].values(),'Suite plan is not a registered plan')
    suite=read_json(directory.parent/'suite.json')
    require(suite.get('plan_sha256')==plan_hash,'Suite plan hash differs from snapshot')
    jobs=[j for j in read_json(plan_path) if j['name']==directory.name]
    require(len(jobs)==1 and jobs[0]['entry']=='qkv','Child not uniquely present in registered plan')
    args=jobs[0]['arguments'];require(len(args)%2==0,'Registered job requires option/value arguments')
    seen=set()
    for flag,value in zip(args[::2],args[1::2]):
        require(flag.startswith('--') and flag not in seen,'Invalid or duplicate planned option');seen.add(flag)
        key=flag[2:].replace('-','_')
        require(str(manifest['configuration'].get(key))==value,'Manifest differs from registered job option '+flag)
    require(manifest['configuration']['stage']==stage,'Manifest stage/configuration mismatch')
    return dict(scope='registered runtime, protocol, plan, explicit options, and cache hashes',plan_sha256=plan_hash)


def aggregate_costs(records):
    out=defaultdict(float)
    for r in records:
        prefix='training_' if r['kind']=='training' else 'teacher_' if r['kind']=='teacher' else 'evaluation_'
        for k in ('target_exposures','gold_tokens','input_positions','padded_input_positions','backbone_calls','training_process_seconds','generation_calls','generation_tokens','generation_seconds','answer_scoring_tokens'):
            if k in r.get('costs',{}):out[k if k=='training_process_seconds' else prefix+k]+=r['costs'][k]
        if r['kind']=='training':out['training_updates']+=r['updates']
        if r['kind']=='evaluation':
            for k in ('independent_interventions','restoration_writes','real_group_write_events'):out['evaluation_'+k]+=r[k]
    return dict(out)


def summarize(root,registration_path=None):
    root=Path(root).resolve();jobs=[];ignored=[];pending=[];preparations=[];caches={};records=[];sources=[]
    repo=Path(__file__).resolve().parents[1]
    if registration_path is None:
        candidates=(repo.parent/'plans/qkv_registration_20261006.json',repo/'docs/results/qkv/registration.json')
        registration_path=next((p for p in candidates if p.is_file()),None)
    registration=load_registration(registration_path,repo/'docs/qkv_protocol.md') if registration_path else None
    for path in discover(root):
        reason=ignored_reason(path,root)
        if reason:ignored.append(dict(run_dir=str(path.parent),reason=reason,manifest_sha256=sha(path)));continue
        m=read_json(path)
        if m.get('protocol')!=RUN_PROTOCOL:continue
        require(m.get('stage') in ('prepare','train','eval','teacher'),'Unknown QKV stage')
        if m.get('complete') is not True:
            pending.append(dict(run_dir=str(path.parent),stage=m['stage'],part=m['configuration'].get('eval_part'),manifest_sha256=sha(path)));continue
        require(m.get('backbone_unchanged') is True,'Backbone preservation missing')
        source=audit_sources(path.parent,m)
        if m['stage'] in ('train','eval'):require(registration is not None,'Formal model runs require the sealed registration')
        if registration:source['registration_check']=registered_run(path.parent,m,registration)
        sources.append(dict(stage=m['stage'],part=m['configuration'].get('eval_part'),run_dir=str(path.parent),**source))
        if m['stage']=='prepare':
            result=m['result'];dataset=read_json(path.parent/'dataset.json')
            require(sha(path.parent/'dataset.json')==result['dataset_sha256'] and dataset['protocol']==DATA_PROTOCOL and dataset['seed']==121042,'Prepared data hash/protocol differs')
            for split,meta in result['caches'].items():
                require(split in ('train','known','dev','confirm'),'Unexpected prepared split')
                cp=path.parent/(split+'.pt');rp=path.parent/(split+'.json');data=read_json(rp)
                require(sha(cp)==meta['cache_sha256'],'Prepared cache hash changed');same(data,dataset[split],'Prepared JSON/dataset differs')
                rows=data['rows'] if split=='train' else data
                require(len(rows)==(64 if split=='train' else 16),'Wrong prepared row count')
                if split=='train':
                    require(set(data)=={'protocol','seed','entities','payloads','rows'} and len(set(data['entities']))==64 and len(set(data['payloads']))==128,'Invalid train-only projection')
                    require(meta['entities']==64 and meta['payloads']==128 and meta['contextual_observations']==8192,'Wrong contextual feature inventory')
                    require(all(r['worlds']==['A','B'] and len(r['questions'])==1 and r['a']==data['payloads'][2*i] and r['b']==data['payloads'][2*i+1] for i,r in enumerate(rows)),'Train rows are not canonical static A/B')
                else:require(meta['records']==16 and all(r['worlds']==['A','B','C','D','SWAP'] and len(r['questions'])==2 for r in rows),'Invalid evaluation projection')
                entry=dict(split=split,rows=rows,data=data,rows_sha256=sha(rp))
                if meta['cache_sha256'] in caches:same(caches[meta['cache_sha256']],entry,'Duplicate cache ambiguity')
                caches[meta['cache_sha256']]=entry
            require(set(result['caches'])=={'train','known','dev','confirm'},'Incomplete prepare inventory')
            preparations.append(dict(kind='prepare',run_dir=str(path.parent),artifacts=artifacts(path.parent,['manifest.json','dataset.json']+[s+ext for s in ('train','known','dev','confirm') for ext in ('.pt','.json')])));continue
        jobs.append((path.parent,m))
    for directory,m in jobs:
        if m['stage']!='train':continue
        require(m.get('cache_sha256') in caches and caches[m['cache_sha256']]['split']=='train','Training must bind prepared train-only cache')
        records.append(audit_training(directory,m,caches[m['cache_sha256']]['data']))
    parents=defaultdict(list)
    for r in records:parents[r['checkpoint_sha256']].append(r)
    for directory,m in jobs:
        if m['stage']=='train':continue
        require(m.get('cache_sha256') in caches,'Eval/teacher must bind prepared cache');cache=caches[m['cache_sha256']]
        if m['stage']=='teacher':r=audit_teacher(directory,m,cache['rows'],cache['split'])
        else:
            parent=parents[m['checkpoint_sha256']];require(len(parent)==1,'Eval must bind exactly one final training checkpoint')
            r=audit_evaluation(directory,m,cache['rows'],cache['split'],parent[0]['regime'],caches[parent[0]['cache_sha256']]['data']['payloads'])
            same(r['architecture'],parent[0]['architecture'],'Eval architecture differs from checkpoint parent');r['parent_train']=parent[0]['run_dir']
        records.append(r)
    for r in records:
        if r['kind']!='evaluation':continue
        tt=[t for t in records if t['kind']=='teacher' and t['cache_sha256']==r['cache_sha256'] and t['part']==r['part']]
        require(len(tt)<=1,'Duplicate teacher results for identical conditions')
        r['teacher_qualification']=dict(source_runs=[t['run_dir'] for t in tt],paired=tt[0]['paired']) if tt else None
        if tt:
            pair_rows=read_json(Path(r['run_dir'])/'pairs.json');eligible={}
            for key,g in tt[0]['paired'].items():
                keep=[p for p in pair_rows if p['phase']+'_'+p['world']==key and p['id'] in set(g['eligible_ids'])]
                eligible[key]=dict(count=len(keep),**{metric:sum(p[metric] for p in keep)/len(keep) if keep else None for metric in ('pair_em','updated_em','update_restore_em')})
            r['teacher_eligible_diagnostics']=eligible
    for seed in SEEDS:
        ts=[r for r in records if r['kind']=='training' and r['seed']==seed]
        if not ts:continue
        require(len({r['common_schedule_sha256'] for r in ts})==1,'Target/background/order schedule differs across arms')
        for k in ('target_exposures','gold_tokens','input_positions','backbone_calls'):
            require(len({r['costs'][k] for r in ts})==1,'Unpadded training exposure/token budget differs: '+k)
        for r in ts[1:]:same(r['token_ledger'],ts[0]['token_ledger'],'Tokenization differs across same-seed arms')
        for regime in ('static','rebind'):
            group=[r for r in ts if r['regime']==regime]
            require(len({r['schedule_sha256'] for r in group})<=1,'Same-regime schedules differ')
            require(len({r['costs']['padded_input_positions'] for r in group})<=1,'Same-regime padded budgets differ')
    train_records=[r for r in records if r['kind']=='training']
    require(len({r['cache_sha256'] for r in train_records})<=1,'Formal models use different training caches')
    for split in ('known','dev','confirm'):
        require(len({r['cache_sha256'] for r in records if r['kind']=='evaluation' and r['split']==split})<=1,'Models use different '+split+' evaluation caches')
    # Prepare and teacher preflight intentionally predate later logging-only additions.
    # Every stage must match its own immutable snapshot; the 15 formal trained models
    # and their 60 evaluations additionally require one common complete src inventory.
    model_sources=[s['source_sha256'] for s in sources if s['stage'] in ('train','eval')]
    if model_sources:
        for s in model_sources[1:]:same(s,model_sources[0],'Runtime source hashes differ across formal model runs')
    teacher_expected={('train','preflight'):128,('known','development'):48,('dev','development'):48,('known','confirmation'):32,('confirm','confirmation'):192}
    teachers=[r for r in records if r['kind']=='teacher'];inventory=Counter((r['split'],r['part']) for r in teachers)
    require(all(k in teacher_expected and n==1 for k,n in inventory.items()),'Unknown/duplicate teacher condition')
    teacher_complete=set(inventory)==set(teacher_expected)
    for r in teachers:require(r['costs']['generation_calls']==teacher_expected[r['split'],r['part']],'Teacher call budget differs')
    return dict(protocol=PROTOCOL,generated_at=datetime.now(timezone.utc).isoformat(),runs_root=str(root),audit_passed=True,
        development_scope=completeness(records),final_scope=completeness(records,True),teacher_scope=dict(expected_conditions=5,observed_conditions=len(teachers),complete=teacher_complete,expected_generations=448),
        partial=bool(pending) or not completeness(records,True)['complete'] or not teacher_complete,
        records=records,pending=pending,ignored=ignored,preparation=preparations,costs=aggregate_costs(records),script_sha256=sha(__file__),
        source_sha256=model_sources[0] if model_sources else None,source_lineage=sources,
        registration=dict(path=str(Path(registration_path).resolve()),sha256=sha(registration_path),protocol_sha256=registration['protocol_document_sha256']) if registration else None,
        source_lineage_scope='Each run matches its own immutable snapshot; all formal train/eval share identical src hashes. Earlier prepare/teacher snapshots may differ in later-added telemetry and are preserved separately.',
        limitations=['Raw text, token IDs, routing weights and scalar schedules are audited; this does not rerun LM generation, tokenizer decoding, or tensor-bank replay.',
            'Token costs are recomputed from logged lengths and cross-checked for repeated question/payload consistency, not independently re-tokenized.',
            'Fact R@read uses four flat slots or three grouped slots. First R@1 uses the maximum weight, never the first group slot.',
            'Edited-slot masses align to actual generation token positions only; no gold-forced word/token correspondence is inferred.',
            'Teacher-forced answer NLL excludes EOS, while training gold CE includes EOS. High retrieval recall does not establish content reconstruction.',
            'C/D and known/new entities share controlled payloads; seeds share one dataset. Neither seed pooling nor template history gives independent semantic generalization.',
            'Empty paired EM for two distinct answers is logically zero. Single-world memory-control differences and correct locality are reported.',
            'Summed process seconds may overlap and are not exclusive GPU latency; smoke and excluded artifacts are not formal costs.'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--select',action='store_true');p.add_argument('--selection-output',type=Path);p.add_argument('--selection',type=Path)
    p.add_argument('--registration',type=Path,help='Sealed registration; defaults to workspace plans, then public docs/results/qkv/registration.json')
    a=p.parse_args(argv);require(not(a.select and a.selection),'Choose selection creation OR final validation')
    require(a.select or a.selection_output is None,'--selection-output requires --select')
    report=summarize(a.runs_root,a.registration);protocol_path=Path(__file__).resolve().parents[1]/'docs/qkv_protocol.md'
    if a.select:
        selection=make_selection(report,protocol_path);dest=a.selection_output or a.output.with_name(a.output.stem+'.selection.json');dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x') as f:json.dump(selection,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        report['sealed_selection']=dict(path=str(dest.resolve()),sha256=sha(dest),selected_arm=selection['decision']['selected_arm'])
    if a.selection:
        selection=read_json(a.selection);report['final_gate']=final_gate(report,selection,protocol_path)
        report['sealed_selection']=dict(path=str(a.selection.resolve()),sha256=sha(a.selection),selected_arm=selection['decision']['selected_arm'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output),audit_passed=True,partial=report['partial'],records=len(report['records']),
        selected_arm=report.get('sealed_selection',{}).get('selected_arm'),eligible_to_expand=report.get('final_gate',{}).get('eligible_to_expand'))))


if __name__=='__main__':main()
