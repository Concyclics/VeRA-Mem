"""CPU-only replay and arithmetic audit of completed reconstruction jobs.

python scripts/audit_reconstruction.py --runs-root ../runs/xtrah100 --output NEW.json
python scripts/audit_reconstruction.py --self-test

No SSH, GPU, language-model execution, checkpoint selection or output overwrite.
Completeness refers to locally declared suite jobs, not an assumed arm count.
Smoke CHILD jobs are separate even when the same suite also prepares real data.
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

os.environ['CUDA_VISIBLE_DEVICES'] = ''
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))
import torch
from vera_mem.reconstruction_vera import ReconstructionVeRA, GroupedVectorDB
from vera_mem.reconstruction_data import digest, validate_training_records, training_records, validate_records, WRITER_PREFIX
from vera_mem.context_distillation_run import REVISION
from vera_mem.metrics import normalize_answer
from replay_coldstart_banks import analysis_exclusions

PROTOCOL = 'reconstruction-experiment-v1'
AUDIT_PROTOCOL = 'reconstruction-artifact-audit-v1'
PHASES = {'CC': (0,0), 'HC': (1,0), 'CH': (0,1), 'HH': (1,1)}
FIELDS = {'A':'a', 'B':'b', 'C':'c', 'D':'d', 'SWAP':'swap_a'}
PAIR_METRICS = ('a_em','updated_em','pair_em','restored_em','update_restore_em','all_three_em',
                'locality_equal','locality_correct','locality_joint','shuffle_em','empty_em')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def same(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       right.detach().cpu().contiguous().reshape(-1).view(torch.uint8))


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes, self.packets = {}, {}

    def resolve(self, name):
        p = Path(name)
        candidates = [p]
        if 'runs' in p.parts:
            candidates.append(self.root.joinpath(*p.parts[p.parts.index('runs')+1:]))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_relative_to(self.root) and resolved.is_file():
                return resolved
        raise ValueError('Missing locally backed input: ' + str(name))

    def sha(self, path):
        path = Path(path).resolve()
        if path not in self.hashes:
            self.hashes[path] = file_hash(path)
        return self.hashes[path]

    def load(self, name, expected=None):
        path = self.resolve(name)
        if expected is not None:
            require(self.sha(path) == expected, 'Input SHA mismatch: ' + str(path))
        if path not in self.packets:
            self.packets[path] = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        return self.packets[path]


def design(packet, part, teacher=False):
    """Independent implementation of the predeclared evaluation grid."""
    split = packet['split']; n = len(packet['rows'])
    indexes = list(range(min(2,n) if part == 'smoke' else n))
    if part == 'preflight':
        require(teacher and split == 'train', 'Preflight must be a canonical training teacher')
        return [('CC',['B'])], indexes
    if part == 'smoke':
        return [('CC',['B'])], indexes
    if split == 'known' and part == 'development':
        return [('CC',['B','C','SWAP']),('HC',['B']),('CH',['B']),('HH',['B'])], indexes
    if split == 'known' and part == 'confirmation':
        return [('CC',['D'])], indexes
    if split == 'dev' and part == 'development':
        return [('CC',['B'])], indexes
    if split == 'confirm' and part == 'confirmation':
        return [(phase,['B']) for phase in PHASES], indexes
    raise ValueError('Unrecognized packet/evaluation-part combination')


def check_packet(packet):
    require(packet['protocol'] == PROTOCOL and packet['rows_sha256'] == digest(packet['rows']), 'Cache record provenance mismatch')
    require(packet.get('model_revision') == REVISION and packet.get('writer_prefix') == WRITER_PREFIX, 'Cache model revision/writer prefix mismatch')
    rows, n, v, w = packet['rows'], len(packet['rows']), packet['views'], len(packet['worlds'])
    require(n > 0 and len({r['id'] for r in rows}) == n, 'Duplicate/empty cached records')
    if packet['split'] == 'train':
        validate_training_records(rows)
        require(v == 1 and packet['worlds'] == ['A','B'], 'Training cache includes probe fields')
    else:
        require(packet['split'] in {'known','dev','confirm'} and v == 2 and packet['worlds'] == list(FIELDS), 'Wrong evaluation cache axes')
    require(packet['q'].ndim == 3 and packet['q'].shape[:2] == (n,v) and packet['q'].is_floating_point()
            and bool(torch.isfinite(packet['q']).all()), 'Query feature shape/dtype/contents mismatch')
    d = packet['q'].shape[-1]
    for slots in (1,3):
        tensor = packet['slots'+str(slots)]
        require(tensor.shape == (n,w,v,slots,d) and tensor.is_floating_point(), 'Content feature shape/dtype mismatch')
        require(bool(torch.isfinite(tensor).all()), 'Nonfinite cached content features')
    return packet


def source_check(directory, manifest, inputs):
    sources = manifest.get('source_sha256', {})
    require({'reconstruction_vera.py','reconstruction_run.py','reconstruction_eval.py','reconstruction_data.py'} <= set(sources), 'Missing runtime source binding')
    frozen = directory.parent/'source/src/vera_mem'
    for name, sha in sources.items():
        require(Path(name).name == name and name.endswith('.py'), 'Invalid source filename')
        path = frozen/name if frozen.is_dir() else REPO/'src/vera_mem'/name
        require(inputs.sha(path) == sha, 'Runtime source SHA mismatch: '+name)
    # CPU re-encoding uses the current module, so explicitly bind that code.
    require(file_hash(REPO/'src/vera_mem/reconstruction_vera.py') == sources['reconstruction_vera.py'],
            'Reconstruction module changed; review replay compatibility before proceeding')
    return dict(files=len(sources), frozen_snapshot=frozen.is_dir())


def costs(raw, result):
    for row in raw:
        require(type(row['generation_tokens']) is int and 1 <= row['generation_tokens'] <= 32,
                'Invalid generation token count')
        require(math.isfinite(row['generation_seconds']) and row['generation_seconds'] >= 0, 'Invalid generation time')
    require(result['generation_calls'] == len(raw), 'Generation count mismatch')
    require(result['generation_tokens'] == sum(r['generation_tokens'] for r in raw), 'Generation token sum mismatch')
    require(math.isclose(result['generation_seconds'],sum(r['generation_seconds'] for r in raw),rel_tol=1e-10,abs_tol=1e-8), 'Generation time sum mismatch')


def audit_teacher(directory, manifest, packet):
    raw = read_jsonl(directory/'predictions.jsonl'); result = read_json(directory/'summary.json')
    require(result == manifest['result'] and result['complete'], 'Teacher result/manifest mismatch')
    grid, indexes = design(packet, manifest['configuration']['eval_part'], teacher=True)
    expected = [(phase,world,i) for phase,worlds in grid for world in ['A',*worlds] if world != 'SWAP' for i in indexes]
    require(len(raw) == len(expected), 'Incomplete/duplicate teacher grid')
    grouped = defaultdict(list)
    for row, (phase, world, index) in zip(raw,expected):
        fact = packet['rows'][index]; sv,qv = PHASES[phase]; wi=packet['worlds'].index(world)
        require((row['id'],row['phase'],row['world']) == (fact['id'],phase,world), 'Teacher grid identity mismatch')
        require(row['answer'] == fact[FIELDS[world]] and row['question'] == fact['questions'][qv]
                and row['context'] == fact['supports'][wi][sv], 'Teacher question/context/answer differs')
        require(row['teacher_bypasses_memory'] is True and row['memory_access'] == dict(
            mode='none',store_present=False,trace=[],last_retrieval_present=False), 'Teacher recorded memory access')
        em = int(normalize_answer(row['prediction']) == normalize_answer(row['answer']))
        require(row['em'] == em, 'Teacher EM mismatch'); grouped[phase+'_'+world].append(em)
    expected_groups={k:dict(correct=sum(v),count=len(v),em=sum(v)/len(v)) for k,v in grouped.items()}
    require(result['groups'] == expected_groups and result['qualified'] == all(v['em'] >= .95 for v in expected_groups.values()), 'Teacher qualification arithmetic differs')
    costs(raw,result)
    return dict(groups=expected_groups,qualified=result['qualified'],zero_recorded_memory_access=True,
                generation_calls=len(raw),generation_tokens=result['generation_tokens'])


def validate_read(row, db, fact, world, phase, condition, role, case_id, trigger_world=None, schema=None):
    schema=schema or dict(route_weights=True,first_fact_R1=True,explicit_write_counts=True)
    require((row['id'],row['world'],row['phase'],row['condition'],row['role'],row['case_id'])
            == (fact['id'],world,phase,condition,role,case_id), 'Prediction identity/order mismatch')
    require(row['trigger_world'] == (trigger_world or world),'Intervention trigger world mismatch')
    require(row['question'] == fact['questions'][PHASES[phase][1]] and row['answer'] == fact[FIELDS[world]]
            and not row.get('context'), 'Student prompt/answer differs or has context')
    require(row['bank_hash'] == db.hash() and row['bank_slots'] == len(db), 'Prediction references wrong bank')
    em = int(normalize_answer(row['prediction']) == normalize_answer(row['answer']))
    require(row['em'] == em and row['budget_hit'] == (row['generation_tokens'] >= 32), 'Prediction arithmetic mismatch')
    trace = row['trace']; require(len(trace) == row['generation_tokens'], 'Incomplete token routing trace')
    target = db.fact_ids.index(fact['id']) if fact['id'] in db.fact_ids else -1
    hits=[]
    for i, event in enumerate(trace):
        require(event['phase'] == ('prefill' if i == 0 else 'decode'), 'Wrong generation route phase')
        indices=event['indices']; k=min(db.top_k,len(db))
        require(isinstance(indices,list) and len(indices)==1 and len(indices[0])==k, 'Routing shape mismatch')
        require(len(set(indices[0]))==k and all(type(j) is int and 0 <= j < len(db) for j in indices[0]), 'Invalid/duplicate route indices')
        if schema['route_weights']:
            scores=torch.tensor(event['scores'],dtype=torch.float32);weights=torch.tensor(event['weights'],dtype=torch.float32)
            require(scores.shape==weights.shape==(1,k) and bool(torch.isfinite(scores).all()) and bool(torch.isfinite(weights).all()),'Route score/weight shape or finiteness differs')
            if k:
                require(bool((scores[:,:-1]>=scores[:,1:]).all()) and bool(((scores>=-1.00001)&(scores<=1.00001)).all()),'Invalid cosine top-k scores')
                require(torch.allclose(weights,torch.softmax(scores/db.temperature,-1),rtol=1e-5,atol=1e-6),'Route weights differ from cosine softmax')
        else:require('scores' not in event and 'weights' not in event,'Legacy route schema differs from bound source')
        hits.append(target >= 0 and any(j//db.slots == target for j in indices[0]))
    require(row['first_fact_recall_at_4'] == int(hits[0]) and row['decode_hits'] == sum(hits[1:])
            and row['decode_queries'] == len(hits)-1, 'Fact routing counters differ')
    first=trace[0]['indices'][0]
    if schema['first_fact_R1']:
        require(row['first_fact_recall_at_1']==int(bool(first and target>=0 and first[0]//db.slots==target)),'First fact R1 counter differs')
    else:require('first_fact_recall_at_1' not in row,'Legacy R1 schema differs from bound source')
    require(type(row['answer_tokens']) is int and row['answer_tokens'] > 0 and math.isfinite(row['nll_sum'])
            and row['nll_sum'] >= -1e-6, 'Invalid answer score')
    return row


def replay_update(base, event, fact, index, keys, values, world, phase):
    require((event['id'],event['index'],event['world'],event['phase'],event['timestamp'])
            == (fact['id'],index,world,phase,1), 'Group update identity mismatch')
    require(event['before_hash'] == base.hash() and same(event['keys'],keys) and same(event['values'],values), 'Group patch source differs')
    changed=GroupedVectorDB.from_snapshot(base.snapshot()); changed.write_group(fact['id'],keys,values,1)
    require(event['after_hash'] == changed.hash(), 'Group update replay hash mismatch')
    mask=torch.ones(len(base),dtype=torch.bool);mask[index*base.slots:(index+1)*base.slots]=False
    require(same(base.keys[mask],changed.keys[mask]) and same(base.values[mask],changed.values[mask]), 'Cross-fact update leakage')
    require(changed.fact_ids == base.fact_ids and all(a == b for j,(a,b) in enumerate(zip(base.timestamps,changed.timestamps)) if bool(mask[j])), 'Update changed other IDs/timestamps')
    return changed


def evaluation_schema(directory,manifest):
    """Allow missing diagnostic fields only in SHA-bound historical smoke code."""
    frozen=directory.parent/'source/src/vera_mem/reconstruction_eval.py'
    path=frozen if frozen.is_file() else REPO/'src/vera_mem/reconstruction_eval.py'
    require(file_hash(path)==manifest['source_sha256']['reconstruction_eval.py'],'Unbound evaluation schema source')
    tree=ast.parse(path.read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    def fields(function):
        return {n.value for n in ast.walk(function) if isinstance(n,ast.Constant) and isinstance(n.value,str)}|{
                n.arg for n in ast.walk(function) if isinstance(n,ast.keyword)}
    read_fields=fields(functions['read']);eval_fields=fields(functions['evaluate'])
    route={'scores','weights'};counts={'independent_interventions','restorations','real_group_write_events'}
    require(not(route&read_fields) or route<=read_fields,'Inconsistent legacy route schema')
    require(not(counts&eval_fields) or counts<=eval_fields,'Inconsistent legacy write schema')
    schema=dict(route_weights=route<=read_fields,first_fact_R1='first_fact_recall_at_1' in read_fields,
                explicit_write_counts=counts<=eval_fields)
    require(all(schema.values()) or 'smoke' in directory.name.lower(),'Formal evaluation lacks required diagnostics')
    return dict(schema,source_sha256=file_hash(path),legacy_smoke=not all(schema.values()))


def audit_eval(directory, manifest, packet, inputs):
    schema=evaluation_schema(directory,manifest)
    checkpoint=inputs.load(manifest['configuration']['checkpoint'],manifest['checkpoint_sha256'])
    require(checkpoint['protocol']==PROTOCOL,'Checkpoint protocol mismatch')
    parent=inputs.resolve(manifest['configuration']['checkpoint']);parent_path=parent.parent/'manifest.json'
    require(not analysis_exclusions(parent.parent,inputs.root),'Evaluation references excluded training')
    parent_manifest=read_json(parent_path);inputs.sha(parent_path)
    require(parent_manifest.get('protocol')==PROTOCOL and parent_manifest.get('stage')=='train'
            and parent_manifest.get('complete') is True,'Evaluation parent is not a completed reconstruction training run')
    parent_cfg=parent_manifest['configuration']
    require(checkpoint['cache_sha256']==parent_manifest['cache_sha256'],'Checkpoint/parent training cache differs')
    parent_packet=check_packet(inputs.load(parent_cfg['cache'],parent_manifest['cache_sha256']))
    require(parent_packet['split']=='train','Evaluation checkpoint parent used nontraining cache')
    require(checkpoint['architecture']['slots']==parent_cfg['slots'] and checkpoint['architecture']['train_B']==parent_cfg['learned_b']
            and checkpoint['seed']==parent_cfg['seed'],'Checkpoint parent architecture/seed differs')
    expected_step=parent_cfg['updates'] if parent.name=='last.pt' else (
        int(parent.stem.removeprefix('step_')) if parent.stem.startswith('step_') and parent.stem[5:].isdigit() else None)
    require(expected_step is not None and checkpoint['step']==expected_step and 0<expected_step<=parent_cfg['updates'],
            'Unrecognized/nonfinal checkpoint step')
    module=ReconstructionVeRA(**checkpoint['architecture']);module.load_state_dict(checkpoint['module']);module.eval().requires_grad_(False)
    result=read_json(directory/'summary.json'); raw=read_jsonl(directory/'predictions.jsonl'); saved_pairs=read_json(directory/'pairs.json')
    require(result==manifest['result'] and result['complete'],'Evaluation result/manifest mismatch')
    require(result['shared_parameters_unchanged'] is True and result['shared_parameter_sha256']==state_digest(checkpoint['module']), 'Online shared state digest mismatch')
    require(result['architecture']==checkpoint['architecture'] and result['slots_per_fact']==module.slots, 'Online architecture changed')
    encoded=inputs.load(str(directory/'encoded_payloads.pt')); banks=inputs.load(str(directory/'bank_events.pt'))
    rows=packet['rows'];n=len(rows);s=module.slots
    require(encoded['fact_ids']==[r['id'] for r in rows] and encoded['worlds']==packet['worlds'],'Encoded payload provenance differs')
    features=packet['slots'+str(s)].float()
    with torch.no_grad():
        reference_keys=module.encode_key(features);reference_values=module.encode_value(features)
    errors={}
    for name,expected in [('keys',reference_keys),('values',reference_values)]:
        actual=encoded[name]
        require(actual.dtype==torch.float32 and actual.shape==expected.shape and bool(torch.isfinite(actual).all()),'Invalid encoded payload')
        require(torch.allclose(actual,expected,rtol=2e-5,atol=2e-5),'Writer encoding differs from checkpoint/cache')
        errors[name]=float((actual-expected).abs().max())
    grid,indexes=design(packet,manifest['configuration']['eval_part'])
    require(set(banks['base_banks'])=={p for p,_ in grid},'Unexpected/missing phase bank')
    expected_events=sum(len(worlds)*len(indexes) for _,worlds in grid)
    require(len(banks['updates'])==len(saved_pairs)==expected_events==result['single_group_updates'],'Group update/pair count differs')
    cursor=0;event_cursor=0;pairs=[];base_hashes={};shuffles=restores=0
    def consume(db,fact,world,phase,condition,role,case_id,trigger_world=None):
        nonlocal cursor
        require(cursor<len(raw),'Missing prediction')
        value=validate_read(raw[cursor],db,fact,world,phase,condition,role,case_id,trigger_world,schema);cursor+=1;return value
    for phase,worlds in grid:
        sv,_=PHASES[phase]
        base=GroupedVectorDB.from_snapshot(banks['base_banks'][phase]); expected=module.new_store()
        for i,fact in enumerate(rows):expected.write_group(fact['id'],encoded['keys'][i,0,sv],encoded['values'][i,0,sv],0)
        require(base.hash()==expected.hash(),'Base bank differs from encoded original A payloads')
        base_hashes[phase]=base.hash(); initial={}
        needed=set(indexes)
        if packet['split']=='known' and phase=='CC':needed.update((i+1)%n for i in indexes)
        for i in sorted(needed):initial[i]=consume(base,rows[i],'A',phase,'real','initial',rows[i]['id'])
        for world in worlds:
            wi=packet['worlds'].index(world)
            for i in indexes:
                fact=rows[i];event=banks['updates'][event_cursor];event_cursor+=1
                db=replay_update(base,event,fact,i,encoded['keys'][i,wi,sv],encoded['values'][i,wi,sv],world,phase)
                updated=consume(db,fact,world,phase,'real','updated',fact['id'])
                pair=dict(id=fact['id'],phase=phase,world=world,a_em=initial[i]['em'],updated_em=updated['em'],
                    pair_em=initial[i]['em']*updated['em'],restored_em=None,locality_equal=None,locality_correct=None)
                if packet['split']=='known' and phase=='CC':
                    neighbor=(i+1)%n;local=consume(db,rows[neighbor],'A',phase,'real','locality',fact['id'],world)
                    pair.update(locality_equal=int(local['prediction']==initial[neighbor]['prediction']),
                                locality_correct=local['em'],locality_joint=local['em']*initial[neighbor]['em'])
                    restored=GroupedVectorDB.from_snapshot(db.snapshot())
                    restored.write_group(fact['id'],encoded['keys'][i,0,sv],encoded['values'][i,0,sv],2)
                    require(restored.hash()==event['restored_hash'] and same(restored.keys,base.keys)
                            and same(restored.values,base.values),'Restoration mismatch')
                    require(restored.timestamps[i*s:(i+1)*s]==(2,)*s,'Restoration timestamp must be 2 for every slot')
                    rr=consume(restored,fact,'A',phase,'real','restored',fact['id'],world);restores+=1
                    pair.update(restored_em=rr['em'],update_restore_em=updated['em']*rr['em'],all_three_em=initial[i]['em']*updated['em']*rr['em'])
                    if world!='SWAP':
                        order=list(range(n));random.Random(111042+i).shuffle(order);perm=list(range(n))
                        for left,right in zip(order,order[1:]+order[:1]):perm[left]=right
                        require(event['value_permutation']==perm and all(i!=j for i,j in enumerate(perm)),'Shuffle mapping mismatch')
                        state=db.snapshot();state['store']['values']=state['store']['values'].reshape(n,s,module.rank)[perm].reshape(-1,module.rank).clone()
                        wrong=GroupedVectorDB.from_snapshot(state);require(wrong.hash()==event['shuffle_hash'],'Shuffled bank hash mismatch')
                        for condition,control in [('shuffle',wrong),('empty',module.new_store())]:
                            row=consume(control,fact,world,phase,condition,'control',fact['id']);pair[condition+'_em']=row['em']
                        shuffles+=1
                require(base.hash()==base_hashes[phase],'A bank contaminated by prior target intervention')
                pairs.append(pair)
    require(cursor==len(raw),'Unexpected/duplicate prediction records')
    require(pairs==saved_pairs,'Saved pair metrics differ from independently reconstructed reads')
    groups=defaultdict(list)
    for pair in pairs:groups[pair['phase']+'_'+pair['world']].append(pair)
    computed={}
    for name,items in groups.items():
        group=dict(count=len(items))
        for field in PAIR_METRICS:
            values=[p[field] for p in items if p.get(field) is not None]
            if values:group[field]=sum(values)/len(values)
        computed[name]=group
    require(result['groups']==computed and result['bank_facts']==n and result['resident_bytes']==base.resident_bytes(),'Evaluation summary mismatch')
    if schema['explicit_write_counts']:
        require(result['independent_interventions']==expected_events and result['restorations']==restores
                and result['real_group_write_events']==expected_events+restores,'Logical write accounting differs')
    else:require(not {'independent_interventions','restorations','real_group_write_events'}&set(result),
                 'Legacy write schema differs from bound source')
    costs(raw,result)
    return dict(groups=computed,generation_calls=len(raw),single_group_updates=expected_events,restorations=restores,
                independent_interventions=expected_events,real_group_write_events=expected_events+restores,
                shuffled_group_controls=shuffles,base_hashes=base_hashes,online_state_sha256=result['shared_parameter_sha256'],
                checkpoint_provenance=dict(path=str(parent),sha256=manifest['checkpoint_sha256'],step=expected_step,
                    parent_manifest_sha256=inputs.sha(parent_path),training_cache_sha256=parent_manifest['cache_sha256']),
                cached_writer_reencoded=True,writer_max_abs_errors=errors,raw_trace_indices_and_fact_counters_verified=True,
                recorded_diagnostic_schema=schema)


def backbone_layout(directory,manifest):
    """Recognize separate/combined A/B from bound runtime AST, not current HEAD."""
    frozen=directory.parent/'source/src/vera_mem/reconstruction_run.py'
    path=frozen if frozen.is_file() else REPO/'src/vera_mem/reconstruction_run.py'
    require(file_hash(path)==manifest['source_sha256']['reconstruction_run.py'],'Unbound training source')
    train=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train')
    calls=[]
    def visit(node,loops):
        if isinstance(node,ast.For):loops=[*loops,node]
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='forward_sequences':calls.append(loops)
        for child in ast.iter_child_nodes(node):visit(child,loops)
    visit(train,[]);require(len(calls)==1,'Unsupported backbone forward call layout')
    loops=calls[0];require(loops and isinstance(loops[0].target,ast.Name) and loops[0].target.id=='step','Unrecognized update loop')
    if len(loops)==1:return dict(mode='combined_AB',calls_per_update=1,source_sha256=file_hash(path))
    expected=ast.parse("enumerate(('a','b'))",mode='eval').body
    require(len(loops)==2 and ast.dump(loops[1].iter)==ast.dump(expected),'Unsupported nested backbone loop')
    return dict(mode='separate_AB',calls_per_update=2,source_sha256=file_hash(path))


def training_parameter_contract(initial, final, step_checkpoint, status):
    """Verify the recorded terminal Adam state and architecture's trainable set.

    Parameter identity follows the SHA-bound runner's explicit group order;
    shape alone would be insufficient to distinguish Wq from Wk. Checkpoints
    cannot prove absence of every transient gradient, so A is checked both as a
    buffer outside those groups and bitwise unchanged at the saved endpoints.
    """
    module=ReconstructionVeRA(**initial['architecture']);module.load_state_dict(initial['module'])
    parameters=dict(module.named_parameters());buffers=dict(module.named_buffers())
    groups=[['Wq.weight','Wk.weight'],['Wv.weight'],['b']]
    rates=[1e-4,3e-4,.005]
    if initial['architecture']['train_B']:groups.append(['B']);rates.append(3e-4)
    names=[name for group in groups for name in group]
    require(set(parameters)==set(names),'Unexpected trainable parameter set')
    require('A' in buffers and 'A' not in parameters and same(initial['module']['A'],final['module']['A']),
            'A is not a fixed unchanged buffer')
    require(status['trainable_parameters']==sum(p.numel() for p in parameters.values())
            and status['learning_rates']==rates,'Trainable count or learning rates differ')
    require(initial['protocol']==final['protocol']==step_checkpoint['protocol']==PROTOCOL,
            'Training checkpoint protocol differs')
    for field in ('architecture','step','seed','cache_sha256','schedule_sha256'):
        require(step_checkpoint[field]==final[field],'Terminal step checkpoint differs: '+field)
    require(set(step_checkpoint['module'])==set(final['module'])
            and all(same(t,step_checkpoint['module'][k]) for k,t in final['module'].items()),
            'Final and terminal step parameter states differ')
    optimizer=step_checkpoint['optimizer'];saved_groups=optimizer['param_groups']
    require(len(saved_groups)==len(groups),'Adam group count differs')
    ids=[];named_shapes={}
    for saved,named,lr in zip(saved_groups,groups,rates):
        require(len(saved['params'])==len(named) and saved['lr']==lr
                and tuple(saved['betas'])==(.9,.999) and saved['eps']==1e-8
                and saved['weight_decay']==0 and saved['amsgrad'] is False,
                'Adam parameter group or hyperparameters differ')
        for identifier,name in zip(saved['params'],named):
            ids.append(identifier);state=optimizer['state'][identifier];parameter=parameters[name]
            require(float(state['step'])==final['step'],'Adam update count differs')
            for field in ('exp_avg','exp_avg_sq'):
                tensor=state[field]
                require(tensor.shape==parameter.shape and tensor.dtype==parameter.dtype
                        and bool(torch.isfinite(tensor).all()),'Adam moment shape/dtype/content differs')
            require(bool((state['exp_avg_sq']>=0).all()),'Adam second moment is negative')
            named_shapes[name]=list(parameter.shape)
    require(len(ids)==len(set(ids)) and set(ids)==set(optimizer['state']), 'Adam contains duplicated or undeclared parameters')
    return dict(trainable_parameter_names=names,trainable_parameters=status['trainable_parameters'],
                optimizer_parameter_shapes=named_shapes,adam_updates=final['step'],
                terminal_optimizer_and_last_state_identical=True,A_is_unchanged_buffer_outside_optimizer=True)


def audit_training_matrix(jobs, inputs, expected_seeds=(71042,71043,71044)):
    """Compare completed formal arms without opening any evaluation artifacts.

    Missing arms/seeds produce an explicit incomplete matrix. Completed
    counterparts must match; malformed or duplicate arms fail closed. The
    formal matrix is the fixed 2 slots x 2 B settings x 3 seeds design.
    """
    groups=defaultdict(dict);expected_seeds=set(expected_seeds);expected_seed_count=len(expected_seeds)
    for job in jobs:
        if job['stage']!='train' or job['scope']!='formal':continue
        cfg=job['training_configuration'];seed=cfg['seed'];arm=(cfg['slots'],cfg['learned_b'])
        require(seed in expected_seeds,'Unplanned formal training seed')
        require(arm in {(1,False),(1,True),(3,False),(3,True)},'Unrecognized training matrix arm')
        require(arm not in groups[seed],'Duplicate formal training arm/seed')
        groups[seed][arm]=job
    require(len(groups)<=expected_seed_count,'More than the predeclared training seeds')
    expected={(1,False),(1,True),(3,False),(3,True)};reports=[];missing=[]
    initial_names=['A','B','Wq.weight','Wk.weight','Wv.weight','b']
    stats_names=['query_center','support_center','value_center','statistics_fitted','value_statistics_fitted']
    all_caches=set()
    for seed,arms in sorted(groups.items()):
        for slots,learned in sorted(expected-set(arms)):
            missing.append(dict(seed=seed,slots=slots,learned_b=learned))
        packets={arm:inputs.load(str(Path(job['run_dir'])/'initial.pt')) for arm,job in arms.items()}
        reference=next(iter(packets.values()));ref_job=next(iter(arms.values()))
        for arm,packet in packets.items():
            job=arms[arm];cfg=job['training_configuration'];all_caches.add(packet['cache_sha256'])
            require(packet['seed']==seed and packet['architecture']['seed']==seed,
                    'Initial checkpoint seed differs from matrix')
            require(cfg['updates']==1024 and cfg['batch_size']==8,'Formal matrix update/batch budget differs')
            require(all(same(packet['module'][name],reference['module'][name]) for name in initial_names),
                    'Same-seed arms have different initial A/B/Wq/Wk/Wv/b')
            require(job['schedule_sha256']==ref_job['schedule_sha256']
                    and job['token_totals']==ref_job['token_totals'],
                    'Same-seed training schedules or token/input budgets differ')
            architecture={k:v for k,v in packet['architecture'].items() if k not in {'slots','train_B'}}
            ref_arch={k:v for k,v in reference['architecture'].items() if k not in {'slots','train_B'}}
            require(architecture==ref_arch,'Uncontrolled same-seed architecture change')
        within_stats={};parameter_counts={}
        for slots in (1,3):
            fixed,learned=(slots,False),(slots,True)
            if fixed not in packets or learned not in packets:continue
            left,right=packets[fixed]['module'],packets[learned]['module']
            require(all(same(left[name],right[name]) for name in stats_names),
                    'Same-slot fixed/learned B initial statistics differ')
            fc=arms[fixed]['trainable_parameters'];lc=arms[learned]['trainable_parameters']
            require(set(arms[learned]['trainable_parameter_names'])
                    ==set(arms[fixed]['trainable_parameter_names'])|{'B'}
                    and lc-fc==left['B'].numel(),'Learned B does not only add B parameters')
            within_stats[str(slots)]=dict(bitwise_equal=True,sha256=state_digest({k:left[k] for k in stats_names}))
            parameter_counts[str(slots)]=dict(fixed=fc,learned=lc,extra_B=left['B'].numel())
        reports.append(dict(seed=seed,arms=len(arms),initial_tensors=initial_names,
            initial_tensors_bitwise_equal=True,initial_tensors_sha256=state_digest({k:reference['module'][k] for k in initial_names}),
            within_slots_statistics=within_stats,cross_slots_statistics_equality_required=False,
            parameter_counts=parameter_counts,schedule_sha256=ref_job['schedule_sha256'],
            equal_reported_token_totals=ref_job['token_totals'],
            initial_checkpoints={f'S{s}_'+('learnedB' if b else 'fixedB'):dict(
                path=str(Path(arms[(s,b)]['run_dir'])/'initial.pt'),
                sha256=inputs.sha(Path(arms[(s,b)]['run_dir'])/'initial.pt')) for s,b in sorted(arms)}))
    require(len(all_caches)<=1,'Formal training matrix uses different train caches')
    complete=len(groups)==expected_seed_count and not missing
    return dict(complete=complete,status='passed' if complete else 'pending',expected_seed_count=expected_seed_count,
        expected_seeds=sorted(expected_seeds),
        expected_arms_per_seed=4,completed_training_jobs=sum(len(v) for v in groups.values()),
        observed_seed_count=len(groups),missing_observed_seed_arms=missing,
        missing_seed_count=expected_seed_count-len(groups),training_cache_sha256=next(iter(all_caches),None),seeds=reports,
        token_budget_scope='Cross-arm equality of saved totals; no tokenizer/backbone reexecution.',
        A_freeze_scope='A is a buffer, excluded from terminal Adam, and unchanged between saved initial/final states.')


def audit_train(directory,manifest,packet,inputs):
    require(packet['split']=='train','Training read a nontraining packet');validate_training_records(packet['rows'])
    cfg=manifest['configuration'];raw=read_jsonl(directory/'training.jsonl');status=read_json(directory/'training_status.json')
    require(status==manifest['result'] and status['complete'],'Training status mismatch')
    require(len(raw)==cfg['updates'] and [x['step'] for x in raw]==list(range(1,cfg['updates']+1)),'Training update budget differs')
    n=len(packet['rows']);batch=cfg['batch_size'];rng=random.Random(cfg['seed']);queue=[];schedule=hashlib.sha256()
    require(type(batch)is int and 1<=batch<=n and n%batch==0,'Invalid exhaustive batch size')
    layout=backbone_layout(directory,manifest)
    for row in raw:
        if not queue:queue=list(range(n));rng.shuffle(queue)
        targets=queue[:batch];del queue[:batch];require(row['targets']==targets,'Training sample schedule differs')
        require(len(row['metrics'])==2,'Training did not score both A/B worlds')
        for m in row['metrics']:
            require(all(math.isfinite(m[k]) for k in ('ce','address','loss')) and math.isclose(m['loss'],m['ce']+.2*m['address'],rel_tol=1e-5,abs_tol=1e-5),'Training objective differs')
        schedule.update(json.dumps(targets).encode())
    require(status['schedule_sha256']==schedule.hexdigest() and status['totals']['target_exposures']==batch*len(raw)
            and status['totals']['backbone_calls']==layout['calls_per_update']*len(raw),'Training accounting differs')
    initial=inputs.load(str(directory/'initial.pt'));final=inputs.load(str(directory/'last.pt'))
    require(initial['step']==0 and final['step']==cfg['updates'] and initial['seed']==final['seed']==cfg['seed'],'Wrong initial/final training checkpoint')
    require(initial['cache_sha256']==final['cache_sha256']==manifest['cache_sha256'] and final['schedule_sha256']==schedule.hexdigest(),'Checkpoint cache/schedule differs')
    require(initial['architecture']==final['architecture']==status['architecture'],'Training architecture changed')
    require(final['architecture']['slots']==cfg['slots'] and final['architecture']['train_B']==cfg['learned_b'],'Declared writer/readout factor differs')
    fixed=['A','query_center','support_center','value_center','statistics_fitted','value_statistics_fitted']
    if not cfg['learned_b']:fixed.append('B')
    require(all(same(initial['module'][k],final['module'][k]) for k in fixed),'Frozen projection/statistics changed')
    step_path=directory/f"step_{cfg['updates']}.pt";terminal=inputs.load(str(step_path))
    contract=training_parameter_contract(initial,final,terminal,status)
    return dict(updates=len(raw),schedule_sha256=schedule.hexdigest(),both_worlds_each_update=True,
                fixed_tensors_verified=fixed,backbone_layout=layout,final_checkpoint_sha256=inputs.sha(directory/'last.pt'),
                terminal_optimizer_checkpoint_sha256=inputs.sha(step_path),training_configuration=cfg,**contract,
                token_totals=status['totals'],token_cost_scope='Reported totals; per-token schedule absent from training logs')


def audit_job(directory,manifest,inputs):
    require(manifest['complete'] is True and manifest['protocol']==PROTOCOL and manifest['backbone_unchanged'] is True,'Not a completed frozen-backbone job')
    source=source_check(directory,manifest,inputs);stage=manifest['stage'];files=[directory/'manifest.json']
    if stage=='prepare':
        artifacts={}
        for split,result in manifest['result'].items():
            path=directory/(split+'.pt');packet=check_packet(inputs.load(str(path),result['cache_sha256']))
            require(packet['rows']==read_json(directory/(split+'.json')) and packet['split']==split,'Preparation row/cache mismatch')
            artifacts[split]=dict(records=len(packet['rows']),cache_sha256=inputs.sha(path))
        require(set(artifacts)=={'train','known','dev','confirm'},'Preparation packets missing')
        train=inputs.load(str(directory/'train.pt'));known=inputs.load(str(directory/'known.pt'))
        require(train['rows']==training_records(known['rows']),'Canonical training projection differs from known facts')
        isolation=validate_records(dict(train=known['rows'],dev=inputs.load(str(directory/'dev.pt'))['rows'],
                                        confirm=inputs.load(str(directory/'confirm.pt'))['rows']))
        result=dict(packets=artifacts,training_projection_verified=True,cross_split_data_checks=isolation)
    else:
        packet=check_packet(inputs.load(manifest['configuration']['cache'],manifest['cache_sha256']))
        files += [directory/name for name in (['training.jsonl','training_status.json','initial.pt','last.pt',f"step_{manifest['configuration']['updates']}.pt"] if stage=='train'
                  else ['predictions.jsonl','summary.json']+(['pairs.json','bank_events.pt','encoded_payloads.pt'] if stage=='eval' else []))]
        before={str(p):file_hash(p) for p in files}
        if stage=='teacher':result=audit_teacher(directory,manifest,packet)
        elif stage=='train':result=audit_train(directory,manifest,packet,inputs)
        elif stage=='eval':result=audit_eval(directory,manifest,packet,inputs)
        else:raise ValueError('Unknown reconstruction stage')
        require(before=={str(p):file_hash(p) for p in files},'Input artifact changed during audit')
        result.update(input_hashes_before=before,input_hashes_unchanged=True,cache_sha256=manifest['cache_sha256'])
    return dict(run_dir=str(directory),stage=stage,scope='smoke' if 'smoke' in directory.name.lower() else 'formal',
                complete=True,source=source,**result)


def validate_configuration(arguments,configuration,run_dir):
    """Bind planned flags (and version-1 defaults) to the child configuration."""
    expected=dict(seed=71042,data_seed=101042,slots=1,learned_b=False,updates=1024,batch_size=8,
                  train_size=16,dev_size=32,confirm_size=64,eval_part='development',cache=None,checkpoint=None)
    integers={k for k,v in expected.items() if type(v)is int};seen=set();i=0
    allowed=set(expected)|{'stage','model'}
    while i<len(arguments):
        flag=arguments[i];require(flag.startswith('--'),'Unexpected positional plan argument');key=flag[2:].replace('-','_')
        require(key in allowed and key not in seen,'Unknown/duplicate planned flag');seen.add(key)
        if key=='learned_b':value=True;i+=1
        else:
            require(i+1<len(arguments),'Planned flag lacks value');value=arguments[i+1];i+=2
            if key in integers:value=int(value)
        expected[key]=value
    require({'stage','model'}<=seen,'Missing required planned stage/model')
    require(all(configuration.get(k)==v for k,v in expected.items()),'Child configuration differs from plan/defaults')
    require(configuration.get('run_dir')==run_dir,'Child run directory differs from launcher')


def run_audit(root, *, training_only=False):
    inputs=Inputs(root);results=[];pending=[];excluded=[]
    for suite_path in sorted(inputs.root.glob('reconstruction_*/suite.json')):
        suite_dir=suite_path.parent;markers=analysis_exclusions(suite_dir,inputs.root)
        if markers:excluded.append(dict(path=str(suite_dir),markers=markers));continue
        suite=read_json(suite_path);require(suite['protocol']==PROTOCOL,'Wrong reconstruction suite protocol')
        plan_path=suite_dir/'plan.json';plan=read_json(plan_path)
        require(isinstance(plan,list) and plan and len({p['name'] for p in plan})==len(plan),'Invalid/duplicate plan jobs')
        require(all(p['entry']=='reconstruction' and isinstance(p['arguments'],list) for p in plan),'Wrong plan entry')
        # plan.json is reformatted by the launcher; original source plan SHA is
        # not its serialized-file SHA. Job argv below binds planned execution.
        jobs={j['name']:j for j in suite['jobs']};require(len(jobs)==len(suite['jobs']),'Duplicate launcher jobs')
        require(set(jobs)<={p['name'] for p in plan},'Unplanned launcher job')
        for spec in plan:
            if training_only and spec['arguments'][spec['arguments'].index('--stage')+1]!='train':continue
            name=spec['name'];directory=suite_dir/name
            markers=analysis_exclusions(directory,inputs.root)
            if markers:excluded.append(dict(path=str(directory),markers=markers));continue
            job=jobs.get(name)
            if not job or job.get('status')!='complete' or job.get('exit_code')!=0:
                pending.append(dict(path=str(directory),reason='Launcher job not complete; raw artifacts not opened'));continue
            args=spec['arguments'];command=job['command'];start=command.index('vera_mem.reconstruction_run')+1
            require(command[start:-2]==args and command[-2]=='--run-dir','Launcher command differs from plan')
            manifest_path=directory/'manifest.json';require(inputs.sha(manifest_path)==job['manifest_sha256'],'Launcher child manifest SHA mismatch')
            manifest=read_json(manifest_path)
            if not manifest.get('complete'):
                pending.append(dict(path=str(directory),reason='Child manifest incomplete; raw artifacts not opened'));continue
            require(manifest['stage']==args[args.index('--stage')+1],'Plan/manifest stage differs')
            validate_configuration(args,manifest['configuration'],command[-1])
            result=audit_job(directory,manifest,inputs);results.append(result)
            print(json.dumps(dict(audited=str(directory),stage=result['stage'],scope=result['scope'])),flush=True)
        for relative,expected in suite['source_files_sha256'].items():
            require(inputs.sha(suite_dir/'source'/relative)==expected,'Suite source snapshot changed')
        if not training_only and not suite.get('complete'):
            pending.append(dict(path=str(suite_dir),reason='Suite not complete'))
    scopes={}
    for scope in ('formal','smoke'):
        rs=[r for r in results if r['scope']==scope]
        scopes[scope]=dict(jobs=len(rs),stages=dict(Counter(r['stage'] for r in rs)),
                          generation_calls=sum(r.get('generation_calls',0) for r in rs),
                          independent_interventions=sum(r.get('independent_interventions',0) for r in rs),
                          restorations=sum(r.get('restorations',0) for r in rs),
                          real_group_write_events=sum(r.get('real_group_write_events',0) for r in rs))
    matrix=audit_training_matrix(results,inputs)
    return dict(protocol=AUDIT_PROTOCOL,audited_at=datetime.now(timezone.utc).isoformat(),checks_passed=True,
        complete=bool(results) and not pending and (not training_only or matrix['complete']),
        audit_scope='training_only' if training_only else 'all_declared_jobs',
        jobs=results,pending=pending,excluded=excluded,scopes=scopes,
        formal_training_comparability=matrix,
        input_sha256={str(k):v for k,v in inputs.hashes.items()},cuda_initialized=torch.cuda.is_initialized(),
        limitations=['Checks recorded routes and replays saved CPU writes; does not regenerate language-model outputs or query logits.',
            'Writer K/V are independently re-encoded from cached frozen features, not from raw text through the backbone.',
            'Online module state digest is bound to the checkpoint; unchanged-state flags and teacher telemetry are recorded evidence, not trusted execution proof.',
            'Facts repeat across models, styles, worlds and controls. No independent-sample significance or pooled accuracy is claimed.',
            'Overall completeness covers declared local suite plans; formal_training_comparability separately requires the predeclared 12-training matrix.',
            'Initial tensors and terminal Adam state verify saved parameter identity/counts and endpoint freeze; they do not prove absence of every transient gradient.'])


def self_test():
    """Synthetic artifacts from a CPU fake decoder; no real result files."""
    import contextlib
    import copy
    import tempfile
    from types import SimpleNamespace
    import unittest
    from vera_mem.reconstruction_data import datasets, training_records
    from vera_mem.reconstruction_eval import evaluate, teacher
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

        def generate(self,question,context=None,max_new_tokens=32):
            if self.mode=='vector_vera':
                for i in range(3):
                    info=self.vector_store.search(torch.ones(1,1,self.vector_store.key_dim))
                    self.last_retrieval=info
                    if self.trace_retrieval:self.retrieval_trace.append(dict(phase='prefill' if i==0 else 'decode',indices=info['indices'][:,-1]))
            return (self.by_context[context] if context else self.by_question[question]['a']),3,.01

        def score(self,question,answer):
            return SimpleNamespace(nll_sum=1.2,tokens=3)

    class Tests(unittest.TestCase):
        def setUp(self):
            self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
            self.suite=self.root/'reconstruction_prepare_fixture';self.suite.mkdir()

        def tearDown(self):self.temp.cleanup()

        def fixture(self,slots=3,stage='eval',part='smoke'):
            raw=datasets()['train'];rows=training_records(raw) if stage=='teacher' else raw
            split='train' if stage=='teacher' else 'known';v=1 if stage=='teacher' else 2;w=2 if stage=='teacher' else 5
            rng=torch.Generator().manual_seed(571)
            packet=dict(protocol=PROTOCOL,model_revision=REVISION,writer_prefix=WRITER_PREFIX,
                        split=split,rows=rows,worlds=['A','B'] if stage=='teacher' else list(FIELDS),
                        views=v,rows_sha256=digest(rows),q=torch.randn(16,v,5,generator=rng),
                        slots1=torch.randn(16,w,v,1,5,generator=rng),slots3=torch.randn(16,w,v,3,5,generator=rng))
            cache=self.suite/f'{stage}_cache.pt';torch.save(packet,cache)
            model=ReconstructionVeRA(5,5,rank=3,key_dim=4,slots=slots,seed=81)
            model.fit_statistics(torch.randn(16,5,generator=rng),torch.randn(16,5,generator=rng))
            model.fit_value_statistics(torch.randn(16,5,generator=rng));model.b.data.fill_(.3)
            train_rows=training_records(raw)
            train_packet=dict(packet,split='train',rows=train_rows,rows_sha256=digest(train_rows),worlds=['A','B'],views=1,
                              q=packet['q'][:,:1],slots1=packet['slots1'][:,:2,:1],slots3=packet['slots3'][:,:2,:1])
            training_cache=self.suite/f'training_cache_{slots}.pt';torch.save(train_packet,training_cache)
            parent=self.suite/f'training_parent_{slots}';parent.mkdir();checkpoint=parent/'last.pt'
            torch.save(dict(protocol=PROTOCOL,architecture=model.configuration(),module=model.state_dict(),step=1,seed=81,
                            cache_sha256=file_hash(training_cache)),checkpoint)
            (parent/'manifest.json').write_text(json.dumps(dict(protocol=PROTOCOL,complete=True,stage='train',
                cache_sha256=file_hash(training_cache),configuration=dict(cache=str(training_cache),slots=slots,learned_b=False,seed=81,updates=1))))
            directory=self.suite/(('smoke_' if part=='smoke' else 'formal_')+stage+str(slots));directory.mkdir()
            args=SimpleNamespace(run_dir=directory,eval_part=part)
            with contextlib.redirect_stdout(io.StringIO()):
                result=teacher(FakeBackend(rows),packet,args) if stage=='teacher' else evaluate(FakeBackend(rows),model,packet,args)
            source={name:file_hash(REPO/'src/vera_mem'/name) for name in ('reconstruction_vera.py','reconstruction_run.py','reconstruction_eval.py','reconstruction_data.py')}
            manifest=dict(protocol=PROTOCOL,complete=True,backbone_unchanged=True,stage=stage,source_sha256=source,
                          configuration=dict(stage=stage,cache=str(cache),checkpoint=str(checkpoint),eval_part=part),
                          cache_sha256=file_hash(cache),checkpoint_sha256=file_hash(checkpoint),result=result)
            (directory/'manifest.json').write_text(json.dumps(manifest))
            return directory,manifest,packet

        def test_single_and_three_slot_replays_and_locality_pair_arithmetic(self):
            for slots in (1,3):
                with self.subTest(slots=slots):
                    directory,manifest,packet=self.fixture(slots)
                    result=audit_job(directory,manifest,Inputs(self.root))
                    self.assertEqual(result['generation_calls'],13)
                    self.assertEqual(result['single_group_updates'],2)
                    self.assertEqual(result['restorations'],2)
                    self.assertEqual(result['groups']['CC_B']['pair_em'],0)
                    self.assertEqual(result['groups']['CC_B']['restored_em'],1)
                    self.assertEqual(result['groups']['CC_B']['locality_joint'],1)
                    self.assertEqual(result['scope'],'smoke')

        def test_actual_full_development_grid_baselines_are_counted_once(self):
            directory,manifest,packet=self.fixture(1,part='development')
            result=audit_job(directory,manifest,Inputs(self.root))
            self.assertEqual(result['generation_calls'],320)
            self.assertEqual(result['single_group_updates'],96)
            self.assertEqual(result['restorations'],48)
            self.assertEqual(result['shuffled_group_controls'],32)
            self.assertEqual(result['scope'],'formal')

        def test_changed_patch_rejected_independently_of_summary(self):
            directory,manifest,packet=self.fixture()
            payload=torch.load(directory/'bank_events.pt',weights_only=True)
            payload['updates'][0]['values'][0,0]+=1
            torch.save(payload,directory/'bank_events.pt')
            with self.assertRaisesRegex(ValueError,'patch source'):
                audit_job(directory,manifest,Inputs(self.root))

        def test_missing_prediction_and_false_pair_metric_rejected(self):
            directory,manifest,packet=self.fixture()
            pairs=read_json(directory/'pairs.json');pairs[0]['pair_em']=1
            (directory/'pairs.json').write_text(json.dumps(pairs))
            with self.assertRaisesRegex(ValueError,'pair metrics'):
                audit_job(directory,manifest,Inputs(self.root))
            rows=read_jsonl(directory/'predictions.jsonl');rows.pop()
            (directory/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError,'Missing prediction'):
                audit_job(directory,manifest,Inputs(self.root))

        def test_shared_parameter_digest_must_match_checkpoint(self):
            directory,manifest,packet=self.fixture()
            manifest['result']['shared_parameter_sha256']='0'*64
            (directory/'summary.json').write_text(json.dumps(manifest['result']))
            with self.assertRaisesRegex(ValueError,'state digest'):
                audit_job(directory,manifest,Inputs(self.root))

        def test_teacher_preflight_all_AB_records_and_access_telemetry(self):
            directory,manifest,packet=self.fixture(stage='teacher',part='preflight')
            result=audit_job(directory,manifest,Inputs(self.root))
            self.assertEqual(result['generation_calls'],32);self.assertTrue(result['qualified'])
            rows=read_jsonl(directory/'predictions.jsonl');rows[0]['memory_access']['store_present']=True
            (directory/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError,'memory access'):
                audit_job(directory,manifest,Inputs(self.root))

        def test_cache_SHA_mismatch_is_rejected_before_raw_diagnostics(self):
            directory,manifest,packet=self.fixture();manifest['cache_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'Input SHA'):
                audit_job(directory,manifest,Inputs(self.root))

        def test_incomplete_child_and_excluded_suite_never_open_poisoned_raw(self):
            plan=[dict(name='unfinished',entry='reconstruction',arguments=['--stage','eval'])]
            (self.suite/'plan.json').write_text(json.dumps(plan))
            (self.suite/'suite.json').write_text(json.dumps(dict(protocol=PROTOCOL,complete=False,
                jobs=[dict(name='unfinished',status='running')],source_files_sha256={})))
            child=self.suite/'unfinished';child.mkdir();(child/'manifest.json').write_text('NOT JSON')
            result=run_audit(self.root);self.assertFalse(result['complete']);self.assertFalse(result['jobs'])
            (self.suite/'suite.json').write_text('NOT JSON')
            (self.suite/'analysis_excluded.json').write_text(json.dumps(dict(reason='synthetic excluded',evidence={},superseded_by='replacement')))
            result=run_audit(self.root);self.assertEqual(len(result['excluded']),1);self.assertFalse(result['pending'])

        def test_existing_output_is_never_overwritten(self):
            path=self.root/'keep.json';path.write_text('keep')
            with self.assertRaisesRegex(ValueError,'overwrite'):
                main(['--runs-root',str(self.root),'--output',str(path)])
            self.assertEqual(path.read_text(),'keep')

        def test_plan_configuration_cannot_change_a_factor_or_checkpoint(self):
            args=['--stage','train','--model','model','--slots','3','--learned-b','--updates','512']
            config=dict(stage='train',model='model',seed=71042,data_seed=101042,slots=3,learned_b=True,
                        updates=512,batch_size=8,train_size=16,dev_size=32,confirm_size=64,eval_part='development',
                        cache=None,checkpoint=None,run_dir='output')
            validate_configuration(args,config,'output')
            for field,value in [('slots',1),('seed',71043),('checkpoint','arbitrary.pt'),('eval_part','confirmation')]:
                wrong=dict(config,**{field:value})
                with self.assertRaisesRegex(ValueError,'configuration differs'):validate_configuration(args,wrong,'output')

        def test_runtime_AST_distinguishes_legacy_and_combined_AB(self):
            source=self.suite/'source/src/vera_mem';source.mkdir(parents=True);path=source/'reconstruction_run.py'
            directory=self.suite/'train'
            variants=[('def train():\n for step in range(3):\n  for world,label in enumerate((\'a\',\'b\')):\n   backend.forward_sequences(q,t)\n',2),
                      ('def train():\n for step in range(3):\n  backend.forward_sequences(q,t)\n',1)]
            for text,count in variants:
                path.write_text(text);manifest=dict(source_sha256={'reconstruction_run.py':file_hash(path)})
                self.assertEqual(backbone_layout(directory,manifest)['calls_per_update'],count)

        def test_legacy_diagnostic_compatibility_is_source_bound_and_smoke_only(self):
            source=self.suite/'source/src/vera_mem';source.mkdir(parents=True);path=source/'reconstruction_eval.py'
            path.write_text('def read():\n return dict(first_fact_recall_at_4=1)\ndef evaluate():\n return dict(single_group_updates=2)\n')
            manifest=dict(source_sha256={'reconstruction_eval.py':file_hash(path)})
            flags=evaluation_schema(self.suite/'smoke_eval_S1',manifest)
            self.assertTrue(flags['legacy_smoke']);self.assertFalse(flags['route_weights']);self.assertFalse(flags['explicit_write_counts'])
            with self.assertRaisesRegex(ValueError,'Formal evaluation'):evaluation_schema(self.suite/'formal_eval',manifest)
            path.write_text(path.read_text()+'# changed\n')
            with self.assertRaisesRegex(ValueError,'Unbound'):evaluation_schema(self.suite/'smoke_eval_S1',manifest)

        def training_matrix_fixture(self):
            jobs=[]
            for seed in (71042,71043,71044):
                for slots in (1,3):
                    for learned in (False,True):
                        module=ReconstructionVeRA(5,6,rank=3,key_dim=4,seed=seed,slots=slots,train_B=learned)
                        # Different pooling can legitimately induce different
                        # S1/S3 centers, but promoting B cannot change centers.
                        module.query_center.fill_(.1)
                        module.support_center.fill_(slots/10)
                        module.value_center.fill_(slots/20)
                        directory=self.suite/f'S{slots}_{learned}_{seed}';directory.mkdir()
                        packet=dict(protocol=PROTOCOL,architecture=module.configuration(),module=module.state_dict(),
                                    step=0,seed=seed,cache_sha256='same_training_cache')
                        torch.save(packet,directory/'initial.pt')
                        jobs.append(dict(stage='train',scope='formal',run_dir=str(directory),
                            training_configuration=dict(seed=seed,slots=slots,learned_b=learned,updates=1024,batch_size=8),
                            schedule_sha256=f'same_schedule_for_seed_{seed}',
                            token_totals=dict(gold_tokens=128,input_positions=1024,padded_input_positions=1408,
                                              backbone_calls=1024,target_exposures=8192),
                            trainable_parameter_names=[k for k,p in module.named_parameters()],
                            trainable_parameters=sum(p.numel() for p in module.parameters())))
            return jobs

        def test_complete_training_matrix_same_init_stats_and_budget(self):
            jobs=self.training_matrix_fixture();result=audit_training_matrix(jobs,Inputs(self.root))
            self.assertTrue(result['complete']);self.assertEqual(result['completed_training_jobs'],12)
            self.assertEqual(len(result['seeds']),3)
            for row in result['seeds']:
                self.assertEqual(row['parameter_counts']['1']['extra_B'],18)
                self.assertFalse(row['cross_slots_statistics_equality_required'])
                self.assertNotEqual(row['within_slots_statistics']['1']['sha256'],row['within_slots_statistics']['3']['sha256'])

        def test_training_matrix_rejects_changed_random_initialization(self):
            jobs=self.training_matrix_fixture();path=Path(jobs[1]['run_dir'])/'initial.pt'
            packet=torch.load(path,weights_only=True);packet['module']['Wv.weight'][0,0]+=.001;torch.save(packet,path)
            with self.assertRaisesRegex(ValueError,'different initial'):
                audit_training_matrix(jobs,Inputs(self.root))

        def test_training_matrix_rejects_stats_schedule_and_token_changes(self):
            jobs=self.training_matrix_fixture()
            for field in ('schedule_sha256','token_totals','trainable_parameters'):
                wrong=copy.deepcopy(jobs)
                if field=='token_totals':wrong[1][field]['input_positions']+=1
                elif field=='trainable_parameters':wrong[1][field]+=1
                else:wrong[1][field]='different'
                with self.assertRaises(ValueError):audit_training_matrix(wrong,Inputs(self.root))
            path=Path(jobs[1]['run_dir'])/'initial.pt';packet=torch.load(path,weights_only=True)
            packet['module']['support_center'][0]+=.01;torch.save(packet,path)
            with self.assertRaisesRegex(ValueError,'statistics differ'):
                audit_training_matrix(jobs,Inputs(self.root))

        def test_incomplete_training_matrix_is_pending_and_duplicate_arm_rejected(self):
            jobs=self.training_matrix_fixture()
            result=audit_training_matrix(jobs[:3],Inputs(self.root))
            self.assertFalse(result['complete']);self.assertEqual(result['status'],'pending')
            self.assertEqual(result['missing_seed_count'],2);self.assertEqual(len(result['missing_observed_seed_arms']),1)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                audit_training_matrix(jobs+[jobs[0]],Inputs(self.root))
            wrong=copy.deepcopy(jobs);wrong[0]['training_configuration']['seed']=999
            with self.assertRaisesRegex(ValueError,'Unplanned'):
                audit_training_matrix(wrong,Inputs(self.root))

        def test_training_only_does_not_require_or_open_pending_evaluation(self):
            from unittest.mock import patch
            matrix_jobs=self.training_matrix_fixture();plan=[];launcher=[];by_dir={}
            for result in matrix_jobs:
                directory=Path(result['run_dir']);cfg=result['training_configuration']
                args=['--stage','train','--model','model','--seed',str(cfg['seed']),'--slots',str(cfg['slots'])]
                if cfg['learned_b']:args.append('--learned-b')
                config=dict(stage='train',model='model',data_seed=101042,train_size=16,dev_size=32,
                            confirm_size=64,eval_part='development',cache=None,checkpoint=None,
                            run_dir=str(directory),**cfg)
                manifest=directory/'manifest.json';manifest.write_text(json.dumps(dict(complete=True,stage='train',configuration=config)))
                plan.append(dict(name=directory.name,entry='reconstruction',arguments=args))
                launcher.append(dict(name=directory.name,status='complete',exit_code=0,manifest_sha256=file_hash(manifest),
                                     command=['python','-m','vera_mem.reconstruction_run',*args,'--run-dir',str(directory)]))
                by_dir[str(directory)]=result
            bad=self.suite/'pending_eval';bad.mkdir();(bad/'manifest.json').write_text('not valid JSON')
            plan.append(dict(name=bad.name,entry='reconstruction',arguments=['--stage','eval','--model','model']))
            (self.suite/'plan.json').write_text(json.dumps(plan))
            (self.suite/'suite.json').write_text(json.dumps(dict(protocol=PROTOCOL,complete=False,jobs=launcher,source_files_sha256={})))
            def fake_job(directory,manifest,inputs):return by_dir[str(directory)]
            with patch(__name__+'.audit_job',side_effect=fake_job),contextlib.redirect_stdout(io.StringIO()):
                result=run_audit(self.root,training_only=True)
            self.assertTrue(result['complete']);self.assertFalse(result['pending'])
            self.assertEqual(result['audit_scope'],'training_only');self.assertEqual(len(result['jobs']),12)

        def optimizer_fixture(self,learned=True):
            module=ReconstructionVeRA(5,6,rank=3,key_dim=4,seed=71042,slots=3,train_B=learned)
            initial=dict(protocol=PROTOCOL,architecture=module.configuration(),module=copy.deepcopy(module.state_dict()),
                         step=0,seed=71042,cache_sha256='cache')
            groups=[dict(params=[module.Wq.weight,module.Wk.weight],lr=1e-4),
                    dict(params=[module.Wv.weight],lr=3e-4),dict(params=[module.b],lr=.005)]
            if learned:groups.append(dict(params=[module.B],lr=3e-4))
            optimizer=torch.optim.Adam(groups)
            with torch.enable_grad():
                sum((p.square().sum()+p.sum()) for p in module.parameters()).backward();optimizer.step()
            final=dict(initial,module=copy.deepcopy(module.state_dict()),step=1,schedule_sha256='schedule')
            step=dict(final,optimizer=optimizer.state_dict())
            status=dict(trainable_parameters=sum(p.numel() for p in module.parameters()),learning_rates=[g['lr'] for g in groups])
            return initial,final,step,status

        def test_terminal_Adam_parameter_contract_fixed_and_learned(self):
            for learned in (False,True):
                result=training_parameter_contract(*self.optimizer_fixture(learned))
                self.assertTrue(result['A_is_unchanged_buffer_outside_optimizer'])
                self.assertEqual('B' in result['trainable_parameter_names'],learned)
                self.assertEqual(result['adam_updates'],1)

        def test_terminal_Adam_wrong_moments_step_or_extra_parameter_rejected(self):
            initial,final,step,status=self.optimizer_fixture()
            for change in ('shape','step','extra','final'):
                wrong=copy.deepcopy(step);identifier=wrong['optimizer']['param_groups'][0]['params'][0]
                if change=='shape':wrong['optimizer']['state'][identifier]['exp_avg_sq']=torch.zeros(1)
                elif change=='step':wrong['optimizer']['state'][identifier]['step']+=1
                elif change=='extra':wrong['optimizer']['state'][999]=copy.deepcopy(wrong['optimizer']['state'][identifier])
                else:wrong['module']['b'][0]+=.01
                with self.assertRaises(ValueError):training_parameter_contract(initial,final,wrong,status)

    import io
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--training-only',action='store_true',help='Audit training jobs only; do not open evaluation artifacts or require eval completion')
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args(argv)
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    if args.self_test:return self_test()
    if args.runs_root is None or args.output is None:parser.error('--runs-root and --output are required')
    require(not args.output.exists(),'Refusing to overwrite audit output');start=time.perf_counter()
    result=run_audit(args.runs_root,training_only=args.training_only);result.update(elapsed_seconds=time.perf_counter()-start,audit_source_sha256=file_hash(Path(__file__)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(complete=result['complete'],scopes=result['scopes'],output=str(args.output))))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
