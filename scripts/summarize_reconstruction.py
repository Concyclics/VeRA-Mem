"""Audit reconstruction runs and seal a development-only architecture selection.

No model execution, GPU, SSH, or tensor deserialization. Raw predictions and
pairs are recomputed; checkpoints/bank tensors are hash-bound, not replayed.
Use --select before ANY confirmation child exists, then --selection PATH for
final validation. A sealed selection is created exclusively, never overwritten.
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

RUN_PROTOCOL = 'reconstruction-experiment-v1'
PROTOCOL = 'reconstruction-audit-v1'
SELECTION_PROTOCOL = 'reconstruction-selection-v1'
SEEDS = (71042, 71043, 71044)
ARMS = ('S1_fixedB', 'S3_fixedB', 'S1_trainB', 'S3_trainB')
PHASES = {'CC': (0, 0), 'HC': (1, 0), 'CH': (0, 1), 'HH': (1, 1)}
FIELDS = {'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd', 'SWAP': 'swap_a'}
GATE_POLICY = dict(ab_pair_min=15/16, c_update_restore_min=15/16,
    c_real_minus_shuffle_min=.5, c_real_minus_empty_min=.5, c_locality_joint_min=.95,
    d_update_restore_min=15/16, confirm_cc_pair_min=.8,
    selection_order=['maximum minimum-seed C update_restore_em',
                     'minimum trainable_parameters', 'minimum bytes_per_fact', 'fixed arm order'],
    seeds=list(SEEDS), arm_order=list(ARMS))


def require(ok, message):
    if not ok: raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def norm(value):
    value=unicodedata.normalize('NFKC',value).casefold()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in value).split())


def number(value, name, integer=False):
    require(type(value) in ((int,) if integer else (int,float)) and math.isfinite(value) and value>=0, 'Invalid '+name)
    return value


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


def architecture_identity(a):
    require(a.get('slots') in (1,3) and type(a.get('train_B')) is bool,'Unknown reconstruction architecture')
    require(a.get('seed') in SEEDS,'Unexpected formal model seed')
    require(a.get('rank')==a.get('key_dim')==64 and a.get('top_k')==4,'Architecture rank/key/top-k differs')
    require(a.get('in_features')==9728 and a.get('out_features')==2560,'Wrong backbone interface dimensions')
    require(a.get('writer_mode')=='masked_mean' and a.get('value_mlp_hidden')==0,'Wrong writer architecture')
    return f"S{a['slots']}_"+('trainB' if a['train_B'] else 'fixedB'),a['seed']


def artifacts(directory,names):
    result={}
    for name in names:
        path=Path(directory)/name;require(path.is_file(),'Missing artifact '+str(path))
        result[name]=sha(path)
    return result


def discover(root):
    root=Path(root).resolve();require(root.is_dir(),'Runs root does not exist')
    result=[]
    for path in sorted(root.rglob('manifest.json')):
        parts=(root.name,*path.relative_to(root).parts[:-1])
        if not any(p.startswith('reconstruction_') for p in parts):continue
        if any(p in ('source','source_snapshot','source_snapshots','.git') for p in parts):continue
        result.append(path)
    return result


def ignored_reason(path,root):
    parts=(root.name,*path.relative_to(root).parts[:-1])
    if any(re.search(r'(^|[_-])smoke($|[_-])',p) for p in parts):return 'Explicitly named smoke job/suite'
    for p in path.parents:
        if p==root.parent:break
        marker=p/'analysis_excluded.json'
        if marker.is_file():
            evidence=read_json(marker);require(evidence.get('reason'),'Missing exclusion reason')
            return 'Explicit analysis exclusion: '+evidence['reason']
    return None


def audit_training(directory,manifest):
    cfg=manifest['configuration'];status=read_json(directory/'training_status.json')
    same(status,manifest['result'],'Training status/manifest mismatch')
    require(status.get('complete') is True and status.get('updates')==cfg.get('updates')==1024,'Formal train budget must be 1024')
    require(cfg.get('batch_size')==8,'Formal batch size must be 8')
    arm,seed=architecture_identity(status['architecture'])
    require(seed==cfg['seed'] and cfg['slots']==status['architecture']['slots'] and cfg['learned_b']==status['architecture']['train_B'],'Training configuration differs from architecture')
    raw=lines(directory/'training.jsonl');require(len(raw)==1024,'Wrong training log length')
    rng=random.Random(seed);queue=[];exposures=Counter();h=hashlib.sha256()
    for step,r in enumerate(raw,1):
        if not queue:queue=list(range(16));rng.shuffle(queue)
        targets=queue[:8];del queue[:8]
        require(r['step']==step and r['targets']==targets,'Wrong step/target schedule')
        h.update(json.dumps(targets).encode());exposures.update(targets)
        require(len(r['metrics'])==2,'Exactly A/B branches required')
        for world in r['metrics']:
            for k in ('ce','address','loss'):number(world[k],k)
            require(math.isclose(world['loss'],world['ce']+.2*world['address'],rel_tol=1e-6,abs_tol=1e-6),'Loss is not CE + .2 address')
        for x in r['gradient_norms']:number(x,'gradient norm')
        number(r['elapsed_seconds'],'elapsed_seconds')
    require(set(exposures.values())=={512} and set(exposures)==set(range(16)),'Unequal target exposure')
    require(status['schedule_sha256']==h.hexdigest(),'Schedule digest mismatch')
    expected_lrs=[1e-4,3e-4,.005]+([3e-4] if status['architecture']['train_B'] else [])
    same(status['learning_rates'],expected_lrs,'Learning rates differ')
    totals=status['totals']
    for k in ('target_exposures','gold_tokens','input_positions','padded_input_positions','backbone_calls'):number(totals[k],k,True)
    require(totals['target_exposures']==8192 and totals['backbone_calls']==1024,'Training exposure/call total differs from one combined A/B forward per update')
    require(totals['padded_input_positions']>=totals['input_positions']>=totals['gold_tokens']>0,'Impossible training token totals')
    hashes=artifacts(directory,('manifest.json','training_status.json','training.jsonl','initial.pt','last.pt','step_1024.pt'))
    return dict(kind='training',arm=arm,seed=seed,run_dir=str(directory),architecture=status['architecture'],
        cache_sha256=manifest['cache_sha256'],checkpoint_sha256=hashes['last.pt'],updates=1024,
        schedule_sha256=h.hexdigest(),trainable_parameters=number(status['trainable_parameters'],'trainable_parameters',True),
        costs=dict(totals,training_process_seconds=number(status['elapsed_seconds'],'training process seconds')),
        artifacts=hashes)


def expected_axes(split,part):
    if part=='preflight':require(split=='train','Preflight must be train only');return {'CC':['B']}
    if split=='known' and part=='development':return {'CC':['B','C','SWAP'],'HC':['B'],'CH':['B'],'HH':['B']}
    if split=='known' and part=='confirmation':return {'CC':['D']}
    if split=='dev' and part=='development':return {'CC':['B']}
    if split=='confirm' and part=='confirmation':return {p:['B'] for p in PHASES}
    raise ValueError('Unexpected packet/part combination')


def prediction_statistics(rows):
    total=len(rows);gold=sum(r['answer_tokens'] for r in rows)
    words=[(norm(r['prediction']).split(),norm(r['answer']).split()) for r in rows]
    return dict(count=total,correct=sum(r['em'] for r in rows),em=sum(r['em'] for r in rows)/total if total else None,
        token_nll=sum(r['nll_sum'] for r in rows)/gold if gold else None,
        full_answer_containment=sum((' '+norm(r['answer'])+' ') in (' '+norm(r['prediction'])+' ') for r in rows),
        first_word_correct=sum(bool(p) and bool(a) and p[0]==a[0] for p,a in words),
        first_two_words_correct=sum(len(p)>=2 and len(a)>=2 and p[:2]==a[:2] for p,a in words),
        third_word_correct=sum(len(p)>=3 and len(a)>=3 and p[2]==a[2] for p,a in words),
        exactly_three_normalized_words=sum(len(p)==3 for p,a in words),
        normalized_word_count_histogram=dict(sorted(Counter(str(len(p)) for p,a in words).items(),key=lambda x:int(x[0]))),
        first_fact_recall_at_4=sum(r['first_fact_recall_at_4'] for r in rows)/total if total else None,
        decode_hits=sum(r['decode_hits'] for r in rows),decode_queries=sum(r['decode_queries'] for r in rows),
        budget_hits=sum(r['budget_hit'] for r in rows),
        generation_calls=total,generation_tokens=sum(r['generation_tokens'] for r in rows),
        generation_seconds=sum(r['generation_seconds'] for r in rows),answer_scoring_tokens=gold)


def audit_evaluation(directory,manifest,rows,split):
    summary=read_json(directory/'summary.json');same(summary,manifest['result'],'Eval summary/manifest mismatch')
    require(summary.get('protocol')=='reconstruction-cpu-vdb-v1','Unknown reconstruction evaluation protocol')
    require(summary.get('complete') is True and summary.get('shared_parameters_unchanged') is True,'Online parameter immutability missing')
    require(re.fullmatch('[0-9a-f]{64}',summary.get('shared_parameter_sha256','')) is not None,'Missing shared parameter digest')
    arm,seed=architecture_identity(summary['architecture']);slots=summary['architecture']['slots']
    count={'known':16,'dev':32,'confirm':64}[split]
    require(len(rows)==summary['bank_facts']==count and summary['slots_per_fact']==slots,'Bank fact/slot count differs')
    resident=dict(key_bytes=count*slots*64*4,value_bytes=count*slots*64*4,total_bytes=count*slots*128*4)
    same(summary['resident_bytes'],resident,'Resident vector bytes differ')
    axes=expected_axes(split,manifest['configuration']['eval_part']);raw=lines(directory/'predictions.jsonl');offset=0
    byid={r['id']:r for r in rows};require(len(byid)==count,'Duplicate fact IDs')
    ids=list(byid);pairs=[];group_rows=defaultdict(list)
    def take(index,world,phase,condition,role,case,trigger):
        nonlocal offset
        require(offset<len(raw),'Missing prediction');r=raw[offset];offset+=1
        for k,v in dict(id=ids[index],world=world,phase=phase,condition=condition,role=role,case_id=case,trigger_world=trigger).items():
            require(r.get(k)==v,'Prediction identity/order differs: '+k)
        source=rows[index];_,qview=PHASES[phase]
        require(r['question']==source['questions'][qview] and r['answer']==source[FIELDS[world]],'Prediction question/answer provenance differs')
        em=int(norm(r['prediction'])==norm(r['answer']));same(r['em'],em,'Raw EM mismatch')
        for k in ('generation_tokens','answer_tokens','bank_slots','decode_hits','decode_queries'):number(r[k],k,True)
        for k in ('generation_seconds','nll_sum'):number(r[k],k)
        require(r['answer_tokens']>0 and r['generation_tokens']<=32,'Answer/generation budget differs')
        require(r['budget_hit']==(r['generation_tokens']>=32),'Wrong budget-hit flag')
        require(re.fullmatch('[0-9a-f]{64}',r['bank_hash']) is not None,'Invalid bank hash')
        expected_slots=0 if condition=='empty' else count*slots
        require(r['bank_slots']==expected_slots,'Wrong active bank size')
        trace=r['trace'];require(isinstance(trace,list) and bool(trace),'Missing actual retrieval trace')
        hits=[]
        for j,t in enumerate(trace):
            require(t['phase']==('prefill' if j==0 else 'decode'),'Trace phase differs')
            indices=t['indices'];require(isinstance(indices,list) and len(indices)==1,'Unexpected trace batch size')
            indices=indices[0]
            require(len(indices)==min(4,expected_slots) and len(set(indices))==len(indices),'Wrong sparse read width')
            require(all(type(x) is int and 0<=x<expected_slots for x in indices),'Trace slot index out of range')
            hits.append(any(x//slots==index for x in indices))
        same(r['first_fact_recall_at_4'],int(hits[0]),'Fact R@4 mismatch')
        same(r['decode_queries'],len(hits)-1,'Decode count mismatch');same(r['decode_hits'],sum(hits[1:]),'Decode hits mismatch')
        return r
    for phase,worlds in axes.items():
        initial=[take(i,'A',phase,'real','initial',ids[i],'A') for i in range(count)]
        require(len({r['bank_hash'] for r in initial})==1,'Initial A reads do not share a bank')
        for world in worlds:
            for i in range(count):
                updated=take(i,world,phase,'real','updated',ids[i],world)
                require(updated['bank_hash']!=initial[i]['bank_hash'],'Updated bank hash equals A bank')
                pair=dict(id=ids[i],phase=phase,world=world,a_em=initial[i]['em'],updated_em=updated['em'],
                    pair_em=initial[i]['em']*updated['em'],restored_em=None,locality_equal=None,locality_correct=None)
                group_rows[phase+'_'+world].append(updated)
                if split=='known' and phase=='CC':
                    neighbor=(i+1)%count;local=take(neighbor,'A',phase,'real','locality',ids[i],world)
                    restored=take(i,'A',phase,'real','restored',ids[i],world)
                    require(local['bank_hash']==updated['bank_hash'],'Locality not read from updated bank')
                    require(restored['bank_hash']!=updated['bank_hash'],'Restore bank equals updated bank')
                    pair.update(locality_equal=int(local['prediction']==initial[neighbor]['prediction']),locality_correct=local['em'],
                        locality_joint=local['em']*initial[neighbor]['em'],restored_em=restored['em'],
                        update_restore_em=updated['em']*restored['em'],all_three_em=initial[i]['em']*updated['em']*restored['em'])
                    if world!='SWAP':
                        for condition in ('shuffle','empty'):
                            control=take(i,world,phase,condition,'control',ids[i],world)
                            pair[condition+'_em']=control['em']
                            if condition=='shuffle':require(control['bank_hash']!=updated['bank_hash'],'Shuffle bank unchanged')
                pairs.append(pair)
    require(offset==len(raw),'Unexpected/duplicate predictions after expected grid')
    same(pairs,read_json(directory/'pairs.json'),'Saved pairs do not match raw generation')
    grouped=defaultdict(list)
    for p in pairs:grouped[p['phase']+'_'+p['world']].append(p)
    metrics=('a_em','updated_em','pair_em','restored_em','update_restore_em','all_three_em','locality_equal','locality_correct','locality_joint','shuffle_em','empty_em')
    groups={}
    for key,items in grouped.items():
        g={'count':len(items)}
        for metric in metrics:
            values=[p[metric] for p in items if p.get(metric) is not None]
            if values:g[metric]=sum(values)/len(values)
        groups[key]=g
    same(groups,summary['groups'],'Summary groups differ from recomputed pairs')
    costs=prediction_statistics(raw)
    for k in ('generation_calls','generation_tokens','generation_seconds'):same(costs[k],summary[k],'Eval cost mismatch '+k)
    same(len(pairs),summary['single_group_updates'],'Single target update count mismatch')
    restores=sum(p.get('restored_em') is not None for p in pairs)
    same(summary['independent_interventions'],len(pairs),'Intervention count mismatch')
    same(summary['restorations'],restores,'Restoration write count mismatch')
    same(summary['real_group_write_events'],len(pairs)+restores,'Real write event count mismatch')
    return dict(kind='evaluation',arm=arm,seed=seed,run_dir=str(directory),split=split,
        part=manifest['configuration']['eval_part'],architecture=summary['architecture'],
        cache_sha256=manifest['cache_sha256'],checkpoint_sha256=manifest['checkpoint_sha256'],
        groups=groups,updated_diagnostics={k:prediction_statistics(v) for k,v in group_rows.items()},
        costs=costs,single_group_updates=len(pairs),restoration_writes=restores,real_group_write_events=len(pairs)+restores,
        write_count_scope='single_group_updates counts intervention events, excludes restoration writes and initial bank population',
        bank_facts=count,bytes_per_fact=resident['total_bytes']//count,
        artifacts=artifacts(directory,('manifest.json','summary.json','predictions.jsonl','pairs.json','encoded_payloads.pt','bank_events.pt')))


def audit_teacher(directory,manifest,rows,split):
    summary=read_json(directory/'summary.json');same(summary,manifest['result'],'Teacher summary mismatch')
    raw=lines(directory/'predictions.jsonl');axes=expected_axes(split,manifest['configuration']['eval_part'])
    expected=[]
    for phase,worlds in axes.items():
        sv,qv=PHASES[phase]
        for world in ['A']+[w for w in worlds if w!='SWAP']:
            for row in rows:expected.append((phase,world,row,sv,qv))
    require(len(raw)==len(expected),'Teacher generation grid incomplete')
    grouped=defaultdict(list)
    for r,(phase,world,row,sv,qv) in zip(raw,expected):
        wi=row['worlds'].index(world)
        require((r['id'],r['phase'],r['world'])==(row['id'],phase,world),'Teacher identity/order mismatch')
        require(r['question']==row['questions'][qv] and r['context']==row['supports'][wi][sv] and r['answer']==row[FIELDS[world]],'Teacher context/question/answer provenance mismatch')
        require(r.get('teacher_bypasses_memory') is True,'Teacher memory bypass missing')
        em=int(norm(r['prediction'])==norm(r['answer']));same(em,r['em'],'Teacher EM mismatch')
        grouped[phase+'_'+world].append(em)
        number(r['generation_tokens'],'teacher tokens',True);number(r['generation_seconds'],'teacher seconds')
    groups={k:dict(correct=sum(v),count=len(v),em=sum(v)/len(v)) for k,v in grouped.items()}
    same(groups,summary['groups'],'Teacher group mismatch')
    require(summary['qualified']==all(v['em']>=.95 for v in groups.values()),'Teacher qualification mismatch')
    costs=dict(generation_calls=len(raw),generation_tokens=sum(r['generation_tokens'] for r in raw),generation_seconds=sum(r['generation_seconds'] for r in raw))
    for k,v in costs.items():same(v,summary[k],'Teacher cost mismatch '+k)
    indexed={(r['phase'],r['world'],r['id']):r['em'] for r in raw}
    paired={}
    for phase,worlds in axes.items():
        for world in worlds:
            if world=='SWAP':continue
            count=len(rows);correct=sum(indexed[phase,'A',r['id']]*indexed[phase,world,r['id']] for r in rows)
            paired[phase+'_'+world]=dict(count=count,both_correct=correct,paired_em=correct/count)
    return dict(kind='teacher',run_dir=str(directory),split=split,part=manifest['configuration']['eval_part'],groups=groups,
        qualified=summary['qualified'],paired=paired,cache_sha256=manifest['cache_sha256'],costs=costs,
        artifacts=artifacts(directory,('manifest.json','summary.json','predictions.jsonl')))


def completeness(records,final=False):
    expected={(arm,seed,kind) for arm in ARMS for seed in SEEDS for kind in ('train','known_development','dev_development')}
    if final:expected|={(arm,seed,kind) for arm in ARMS for seed in SEEDS for kind in ('known_confirmation','confirm_confirmation')}
    found=defaultdict(list)
    for r in records:
        if r['kind'] not in ('training','evaluation'):continue
        key=(r['arm'],r['seed'],'train' if r['kind']=='training' else r['split']+'_'+r['part'])
        if key in expected:found[key].append(r['run_dir'])
    return dict(expected=len(expected),observed=len(found),complete=set(found)==expected and all(len(v)==1 for v in found.values()),
        missing=[list(k) for k in sorted(expected-set(found))],duplicates=[dict(condition=list(k),run_dirs=v) for k,v in found.items() if len(v)>1])


def development_selection(records):
    barrier=completeness(records);require(barrier['complete'],'Development barrier incomplete or duplicate: '+json.dumps(barrier))
    teacher=[r for r in records if r['kind']=='teacher' and r['split']=='train' and r['part']=='preflight']
    require(bool(teacher) and all(set(r['groups'])=={'CC_A','CC_B'} and all(g['count']==g['correct']==16 for g in r['groups'].values()) for r in teacher),'Training A/B teacher preflight must be 16/16 both worlds')
    indexed={(r['arm'],r['seed'],'train' if r['kind']=='training' else r['split']+'_'+r['part']):r
             for r in records if r['kind'] in ('training','evaluation')}
    candidates=[]
    for arm in ARMS:
        per_seed=[];params=set();bytes_=set()
        for seed in SEEDS:
            train=indexed[arm,seed,'train'];e=indexed[arm,seed,'known_development'];b=e['groups']['CC_B'];c=e['groups']['CC_C']
            require(b['count']==c['count']==16,'Known gate denominator must be 16')
            tests=dict(ab_pair=b['pair_em']>=15/16,c_update_restore=c['update_restore_em']>=15/16,
                c_real_minus_shuffle=c['updated_em']-c['shuffle_em']>=.5,c_real_minus_empty=c['updated_em']-c['empty_em']>=.5,
                c_locality_joint=c['locality_joint']>=.95)
            per_seed.append(dict(seed=seed,passed=all(tests.values()),checks=tests,ab_pair=b['pair_em'],
                c_update_restore=c['update_restore_em'],c_real=c['updated_em'],c_shuffle=c['shuffle_em'],c_empty=c['empty_em'],c_locality_joint=c['locality_joint']))
            params.add(train['trainable_parameters']);bytes_.add(e['bytes_per_fact'])
        require(len(params)==len(bytes_)==1,'Arm parameter/byte counts vary across seeds')
        candidates.append(dict(arm=arm,eligible=all(s['passed'] for s in per_seed),per_seed=per_seed,
            minimum_seed_c_update_restore=min(s['c_update_restore'] for s in per_seed),
            trainable_parameters=params.pop(),bytes_per_fact=bytes_.pop()))
    eligible=sorted([x for x in candidates if x['eligible']],key=lambda x:(-x['minimum_seed_c_update_restore'],x['trainable_parameters'],x['bytes_per_fact'],ARMS.index(x['arm'])))
    evidence=[]
    for r in records:
        if r['kind']=='training' or (r['kind']=='evaluation' and r['part']=='development') or r in teacher:
            evidence.append(dict(run_dir=r['run_dir'],artifacts=r['artifacts']))
    return dict(policy=GATE_POLICY,selected_arm=eligible[0]['arm'] if eligible else None,candidates=candidates,
        evidence=sorted(evidence,key=lambda x:x['run_dir']),development_barrier=barrier,
        note='Selected architecture only; all three seeds retained. No D/confirmation scores or outcome-based record filtering enter selection.')


def aggregate_costs(records):
    out=defaultdict(float)
    for r in records:
        prefix='training_' if r['kind']=='training' else 'teacher_' if r['kind']=='teacher' else 'evaluation_'
        for k,v in r.get('costs',{}).items():
            if k in ('target_exposures','gold_tokens','input_positions','padded_input_positions','backbone_calls','training_process_seconds','generation_calls','generation_tokens','generation_seconds','answer_scoring_tokens'):
                out[k if k.startswith('training_') else prefix+k]+=v
        if r['kind']=='training':out['training_updates']+=r['updates']
        if r['kind']=='evaluation':
            out['evaluation_single_group_updates']+=r['single_group_updates']
            out['evaluation_restoration_writes']+=r.get('restoration_writes',0)
            out['evaluation_real_group_write_events']+=r.get('real_group_write_events',r['single_group_updates'])
    return dict(out)


def summarize(root):
    root=Path(root).resolve();jobs=[];ignored=[];pending=[];other=[];caches={};records=[]
    for p in discover(root):
        reason=ignored_reason(p,root)
        if reason:ignored.append(dict(run_dir=str(p.parent),reason=reason,manifest_sha256=sha(p)));continue
        m=read_json(p)
        if m.get('protocol')!=RUN_PROTOCOL:continue
        require(m.get('stage') in ('prepare','train','eval','teacher'),'Unknown reconstruction stage')
        if m.get('complete') is not True:
            pending.append(dict(run_dir=str(p.parent),stage=m['stage'],part=m['configuration'].get('eval_part'),manifest_sha256=sha(p)));continue
        require(m.get('backbone_unchanged') is True,'Backbone preservation missing')
        if m['stage']=='prepare':
            for split,data in m['result'].items():
                cp=p.parent/(split+'.pt');rp=p.parent/(split+'.json');require(cp.is_file() and rp.is_file(),'Missing prepared cache/rows')
                require(sha(cp)==data['cache_sha256'],'Prepared cache hash changed')
                rows=read_json(rp);require(digest(rows)==data['rows_sha256'] and len(rows)==data['records'],'Prepared row metadata mismatch')
                entry=dict(split=split,rows=rows,source=str(cp),rows_sha256=data['rows_sha256'])
                if data['cache_sha256'] in caches:same(entry,caches[data['cache_sha256']],'Duplicate cache ambiguity')
                caches[data['cache_sha256']]=entry
            other.append(dict(kind='prepare',run_dir=str(p.parent),manifest_sha256=sha(p)));continue
        jobs.append((p.parent,m))
    for directory,m in jobs:
        if m['stage']!='train':continue
        require(m.get('cache_sha256') in caches and caches[m['cache_sha256']]['split']=='train','Train must link a prepared train-only cache')
        cache=caches[m['cache_sha256']];require(len(cache['rows'])==16 and all(r.get('worlds')==['A','B'] and len(r['questions'])==1 for r in cache['rows']),'Training projection is not canonical A/B only')
        records.append(audit_training(directory,m))
    parents=defaultdict(list)
    for r in records:parents[r['checkpoint_sha256']].append(r)
    for directory,m in jobs:
        if m['stage']=='train':continue
        require(m.get('cache_sha256') in caches,'Eval/teacher must link prepared cache')
        cache=caches[m['cache_sha256']]
        if m['stage']=='teacher':r=audit_teacher(directory,m,cache['rows'],cache['split'])
        else:
            r=audit_evaluation(directory,m,cache['rows'],cache['split'])
            parent=parents[r['checkpoint_sha256']];require(len(parent)==1,'Eval must bind exactly one complete final training checkpoint')
            require((r['arm'],r['seed'])==(parent[0]['arm'],parent[0]['seed']),'Eval architecture differs from loaded checkpoint training lineage')
            same(r['architecture'],parent[0]['architecture'],'Eval/training architecture differs')
            r['parent_train']=parent[0]['run_dir']
        records.append(r)
    for r in records:
        if r['kind']!='evaluation':continue
        teachers=[t for t in records if t['kind']=='teacher' and t['cache_sha256']==r['cache_sha256'] and t['part']==r['part']]
        if teachers:
            for t in teachers[1:]:same(t['paired'],teachers[0]['paired'],'Conflicting teacher outcomes on identical conditions')
        r['teacher_qualification']=dict(source_runs=[t['run_dir'] for t in teachers],paired=teachers[0]['paired']) if teachers else None
    # Same seed must mean the exact same exposure schedule and token budget.
    for seed in SEEDS:
        ts=[r for r in records if r['kind']=='training' and r['seed']==seed]
        if ts:
            require(len({r['schedule_sha256'] for r in ts})==1,'Arms do not share a target schedule')
            for k in ('target_exposures','gold_tokens','input_positions','padded_input_positions','backbone_calls'):
                require(len({r['costs'][k] for r in ts})==1,'Arms do not share training budget '+k)
    return dict(protocol=PROTOCOL,generated_at=datetime.now(timezone.utc).isoformat(),runs_root=str(root),
        audit_passed=True,development_scope=completeness(records),final_scope=completeness(records,final=True),
        partial=bool(pending) or not completeness(records,final=True)['complete'],records=records,pending=pending,
        ignored=ignored,preparation=other,costs=aggregate_costs(records),script_sha256=sha(__file__),
        limitations=['This audit recomputes scalar metrics from raw text/trace and binds artifact hashes; no tensor bank replay or LM generation is performed.',
            'Teacher-forced NLL is not free generation. C/D share the first two words with A and probe novel third-word combinations, not unseen vocabulary.',
            'Locality checks one fixed neighbor; identical wrong outputs are not jointly correct. Empty paired EM is logically zero for distinct answers.',
            'Three seeds share the same small facts, so do not pool them as independent facts. Writer slots change addressing and granularity together.',
            'Same update/token budget is not equal parameter count, vector bytes, FLOPs, or wall-clock latency. Summed process seconds can overlap.',
            'Smoke and explicit exclusions are separately itemized, never mixed into formal metrics. Tensor initial-weight equality needs the separate architecture audit.'])


def final_gate(report,selection):
    expected=development_selection(report['records'])
    require(selection.get('protocol')==SELECTION_PROTOCOL and selection.get('sealed') is True,'Not a sealed development selection')
    same(selection['decision'],expected,'Sealed development evidence or decision changed')
    arm=expected['selected_arm'];entries=[]
    for seed in SEEDS:
        got={r['split']:r for r in report['records'] if r['kind']=='evaluation' and r['part']=='confirmation' and r['arm']==arm and r['seed']==seed}
        if set(got)!={'known','confirm'}:entries.append(dict(seed=seed,complete=False));continue
        d=got['known']['groups']['CC_D'];c=got['confirm']['groups']['CC_B']
        require(d['count']==16 and c['count']==64,'Wrong final gate denominator')
        entries.append(dict(seed=seed,complete=True,d_update_restore=d['update_restore_em'],confirm_cc_pair=c['pair_em'],
            passed=d['update_restore_em']>=15/16 and c['pair_em']>=.8))
    complete=completeness(report['records'],final=True)['complete'] and not report.get('pending')
    selected_passed=arm is not None and all(x.get('passed',False) for x in entries)
    return dict(selected_arm=arm,per_seed=entries,all_formal_results_complete=complete,
        selected_architecture_gate_passed=selected_passed,eligible_to_expand=complete and selected_passed,
        note='Only the preselected architecture can pass; other confirmation outcomes never replace it. No candidate or a failed seed stops scaling.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--select',action='store_true');p.add_argument('--selection-output',type=Path);p.add_argument('--selection',type=Path)
    args=p.parse_args(argv);require(not(args.select and args.selection),'Choose selection creation OR final validation')
    require(args.select or args.selection_output is None,'--selection-output requires --select')
    report=summarize(args.runs_root)
    if args.select:
        require(not any(r.get('part')=='confirmation' for r in report['records']+report['pending']),
                'Cannot select after a confirmation child exists, including incomplete jobs')
        require(not report['pending'],'Cannot seal while a non-smoke reconstruction job is unfinished')
        decision=development_selection(report['records'])
        selection=dict(protocol=SELECTION_PROTOCOL,sealed=True,created_at=datetime.now(timezone.utc).isoformat(),
            script_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).resolve().parents[1]/'docs/reconstruction_protocol.md'),decision=decision)
        dest=args.selection_output or args.output.with_name(args.output.stem+'.selection.json');dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x') as f:json.dump(selection,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        report['sealed_selection']=dict(path=str(dest.resolve()),sha256=sha(dest),selected_arm=decision['selected_arm'])
    if args.selection:
        selection=read_json(args.selection);report['final_gate']=final_gate(report,selection)
        report['sealed_selection']=dict(path=str(args.selection.resolve()),sha256=sha(args.selection))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),audit_passed=True,partial=report['partial'],records=len(report['records']),
        selected_arm=report.get('sealed_selection',{}).get('selected_arm'),eligible_to_expand=report.get('final_gate',{}).get('eligible_to_expand'))))


if __name__=='__main__':main()
