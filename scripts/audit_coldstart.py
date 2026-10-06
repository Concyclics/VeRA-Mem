"""Read-only CPU audit of coldstart suites, paired budgets and frozen banks.

Run with a torch-enabled Python: --runs-root RUNS --output AUDIT.
Completed jobs in partial suites are audited; unfinished job contents are not
opened. --require-complete additionally requires the predeclared 44 evaluations,
8 memorization probes, 11 training jobs and matched/Wiki warmup pairing.
No encoders, generation, training, network calls or GPU allocation occur.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
from itertools import zip_longest
import re
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from vera_mem.vector_store import PersistentVectorDB
from audit_interface_banks import audit_run as audit_episodic,restore
from summarize_interface import audit_evaluation,read_json,read_jsonl,require,same_number,file_hash,digest,PHASES
from replay_coldstart_banks import analysis_exclusions

ARMS=("no_warm","matched_warm","wiki_warm","fixed","learned_sparse","dense_warm","straight_through","wiki_joint")
TRAIN_ARMS={"no_warm","matched_warm","wiki_warm","matched_warm_transfer","wiki_warm_transfer","fixed","learned_sparse","dense_warm","straight_through","wiki_joint_warm","wiki_joint"}
PROBE="train_memorization_duplicate_canonical_views"
REPAIR_ID="coldwiki-6184b88599664708a482acba"
REPAIR_INDEX=13896
REPAIR_DIR="coldstart_repair_20261006/features"
FOLLOWUP_ARM='wiki_teacher_repair'
FOLLOWUP_TRAIN_ARMS={'wiki_teacher_warm',FOLLOWUP_ARM}
SELECTION_RULE='quote>=0.9 else annotated>=0.9 else none'


class Inputs:
    def __init__(self,root):
        self.root=Path(root).resolve();self.hashes={};self.packets={}
        runs=next((p for p in [self.root,*self.root.parents] if p.name=="runs"),None)
        self.workspace=runs.parent if runs else self.root.parent
    def resolve(self,name):
        p=Path(name)
        if p.is_file():return p
        parts=p.parts
        for segment,root in (("runs",self.root),("data",self.workspace/"data")):
            if segment in parts:
                q=root.joinpath(*parts[parts.index(segment)+1:])
                if q.is_file():return q
                if segment=="data":
                    q=root/self.root.name/Path(*parts[parts.index(segment)+1:])
                    if q.is_file():return q
        raise FileNotFoundError("Missing locally backed artifact: "+str(p))
    def sha(self,path):
        path=Path(path).resolve()
        if path not in self.hashes:self.hashes[path]=file_hash(path)
        return self.hashes[path]
    def load(self,name,expected=None):
        path=self.resolve(name)
        if expected is not None:require(self.sha(path)==expected,"Input artifact SHA mismatch: "+str(path))
        if path not in self.packets:self.packets[path]=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
        return self.packets[path]


def bitwise_equal(left,right):
    if left.shape!=right.shape or left.dtype!=right.dtype:return False
    if left.ndim==0:left,right=left.reshape(1),right.reshape(1)
    return all(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
               for a,b in zip(left.detach().cpu().split(128),right.detach().cpu().split(128)))


def verify_repair_packets(parent,repaired,target_id,target_index,statistics_count=8192):
    """Independent byte comparisons; no encoder and no CUDA recomputation."""
    from vera_mem.coldstart_data import replace_target_after_anchor
    require(target_index>=statistics_count,"Repair intersects common statistics slice")
    old,new=parent['rows'],repaired['rows']
    require(len(old)==len(new) and 0<=target_index<len(old),"Repair record count/index mismatch")
    require(old[target_index]['id']==new[target_index]['id']==target_id,"Repair target identity mismatch")
    changed=[i for i,(a,b) in enumerate(zip(old,new)) if a!=b]
    require(changed==[target_index],"Repair must change exactly the target record")
    a,b=old[target_index],new[target_index]
    require(set(a)==set(b) and all(a[k]==b[k] for k in a if k!='supports'),"Repair changed target labels/query/provenance")
    require(len(a['supports'])==len(b['supports'])==2 and a['supports'][0]==b['supports'][0],"Repair changed A support")
    expected=[replace_target_after_anchor(s,a['provenance']['anchor'],a['a'],a['b']) for s in a['supports'][0]]
    require(b['supports'][1]==expected and len(expected)==1,"Repair does not address canonical B after anchor")
    allowed={'rows','data_manifest_sha256','data_jsonl_sha256','feature_repair'}
    require(set(repaired)-set(parent)<={'feature_repair'} and not set(parent)-set(repaired),"Repair changed packet schema")
    changed_elements={}
    for name,value in parent.items():
        if name in allowed:continue
        other=repaired[name]
        if isinstance(value,torch.Tensor):
            require(isinstance(other,torch.Tensor) and value.shape==other.shape and value.dtype==other.dtype,"Repair tensor schema mismatch: "+name)
            if name in {'last','pool'}:
                require(value.ndim==4 and value.shape[:3]==(len(old),2,1),"Repair support view shape")
                require(bool(torch.isfinite(other[target_index,1,0]).all()),"Repair target feature is nonfinite: "+name)
                require(bitwise_equal(value[:target_index],other[:target_index]) and bitwise_equal(value[target_index+1:],other[target_index+1:]) and bitwise_equal(value[target_index,0],other[target_index,0]),"Repair changed protected A/other features: "+name)
                require(bitwise_equal(value[:statistics_count],other[:statistics_count]),"Repair changed common statistics: "+name)
                changed_elements[name]=int((value[target_index,1,0].view(torch.uint8)!=other[target_index,1,0].view(torch.uint8)).sum())
            else:require(bitwise_equal(value,other),"Repair changed protected tensor: "+name)
        else:require(value==other,"Repair changed metadata: "+name)
    require(all(name in parent for name in ('q','last','pool')),"Missing feature tensors")
    return dict(status='passed',target_index=target_index,changed_record_ids=[target_id],
        query_bitwise_unchanged=True,all_a_features_bitwise_unchanged=True,all_other_features_bitwise_unchanged=True,
        common_statistics_inputs_bitwise_unchanged=True,common_statistics_records=statistics_count,
        token_repairs_unchanged=True,target_feature_changed_bytes=changed_elements,
        scope='Byte equality, including signed zeros; only the target B canonical last/pool slots are exempt. Centers are not refitted.')


def audit_repair(inputs):
    """Open repaired cache only after a completed independent repair manifest."""
    directory=inputs.root/REPAIR_DIR;path=directory/'manifest.json'
    if not path.is_file():return dict(status='pending_repair_manifest')
    manifest=read_json(path)
    if manifest.get('complete') is not True:return dict(status='pending_repair_complete',manifest=str(path))
    require(manifest.get('repair_protocol')=='coldstart-anchor-overlap-repair-v2' and manifest.get('stage')=='repair','Unexpected repair protocol/stage')
    require(manifest.get('backbone_unchanged') is True,'Repair encoder backbone freeze was not verified')
    cfg=manifest['configuration'];parent_path=inputs.resolve(cfg['cache']);new_path=directory/'train.pt'
    parent=inputs.load(str(parent_path),manifest['cache_sha256']);repaired=inputs.load(str(new_path),manifest['output_train_sha256'])
    proof=verify_repair_packets(parent,repaired,REPAIR_ID,REPAIR_INDEX)
    from repair_coldstart_overlap import protected_digests,tensor_digest
    recorded=read_json(directory/'preservation.json')
    require(recorded==manifest['result'] and recorded['id']==REPAIR_ID and recorded['index']==REPAIR_INDEX,'Repair preservation sidecar/manifest mismatch')
    require(recorded['reencoded_supports']==1 and recorded['protected_tensors_bitwise_unchanged'] and recorded['token_repairs_unchanged'] and recorded['only_target_row_changed'],'Repair preservation claims incomplete')
    require(protected_digests(parent,REPAIR_INDEX)==recorded['protected_tensors_before'] and protected_digests(repaired,REPAIR_INDEX)==recorded['protected_tensors_after'],'Repair preserved hashes do not match actual tensors')
    for name in ('last','pool'):
        require(tensor_digest(parent[name][REPAIR_INDEX,1,0])==recorded['target_before'][name] and tensor_digest(repaired[name][REPAIR_INDEX,1,0])==recorded['target_after'][name],'Repair target tensor digest mismatch')
    require(repaired['token_repairs']==read_json(directory/'train_repairs.json'),'Repair tokenizer sidecar mismatch')
    data_path=inputs.resolve(str(Path(cfg['data_dir'])/'manifest.json'));data=read_json(data_path)
    require(data.get('complete') and data.get('data_revision')=='v2' and data.get('repair_protocol')==manifest['repair_protocol'],'Unsealed v2 raw data')
    require(inputs.sha(data_path)==manifest['data_manifest_sha256']==repaired['data_manifest_sha256'],'v2 manifest linkage mismatch')
    parent_data_path=inputs.resolve(str(Path(data['parent_data_directory'])/'manifest.json'));parent_data=read_json(parent_data_path)
    require(inputs.sha(parent_data_path)==data['parent_manifest_sha256']==parent['data_manifest_sha256'],'v1 parent raw linkage mismatch')
    lineage=repaired['feature_repair']
    require(lineage['parent_cache_sha256']==inputs.sha(parent_path) and lineage['parent_data_manifest_sha256']==inputs.sha(parent_data_path),'Feature repair parent SHA mismatch')
    require(lineage['protocol']==manifest['repair_protocol'] and lineage['id']==REPAIR_ID and lineage['index']==REPAIR_INDEX and lineage['reencoded_supports']==1 and lineage['first_1024_update_target_steps']==[940],'Feature repair scope provenance mismatch')
    require(len(data['repairs'])==1 and data['repairs'][0]['id']==REPAIR_ID and data['repairs'][0]['index']==REPAIR_INDEX,'Raw repair scope mismatch')
    raw_changed=[];raw_files=[]
    from vera_mem.coldstart_data import digest as data_digest,replace_target_after_anchor
    for domain in ('wikipedia','matched','synthetic'):
        for split in ('train','dev','confirm'):
            old_path=inputs.resolve(str(parent_data_path.parent/domain/(split+'.jsonl')))
            current_path=inputs.resolve(str(data_path.parent/domain/(split+'.jsonl')))
            require(inputs.sha(old_path)==parent_data['files'][domain][split]['sha256'] and inputs.sha(current_path)==data['files'][domain][split]['sha256'],'Raw repair file hash mismatch')
            if (domain,split)!=('wikipedia','train'):
                require(inputs.sha(old_path)==inputs.sha(current_path),'Raw repair changed an unrelated file')
            else:
                require(parent['data_jsonl_sha256']==inputs.sha(old_path) and repaired['data_jsonl_sha256']==inputs.sha(current_path),'Feature/raw JSONL linkage mismatch')
                with old_path.open() as left,current_path.open() as right:
                    for index,(old_line,new_line) in enumerate(zip_longest(left,right)):
                        require(old_line is not None and new_line is not None,'Raw repair changed line count')
                        if old_line==new_line:continue
                        before,after=json.loads(old_line),json.loads(new_line)
                        require(index==REPAIR_INDEX and before['id']==after['id']==REPAIR_ID,'Raw repair changed an unapproved record')
                        expected=dict(before,supports=[before['supports'][0],[replace_target_after_anchor(s,before['provenance']['anchor'],before['a'],before['b']) for s in before['supports'][0]]])
                        require(expected==after and before!=after,'Raw repair changed fields beyond the target B supports')
                        require(data_digest(before)==data['repairs'][0]['row_before_sha256'] and data_digest(after)==data['repairs'][0]['row_after_sha256'],'Raw repaired row digest mismatch')
                        raw_changed.append(dict(domain=domain,split=split,index=index,id=REPAIR_ID,views=len(before['supports'][1])))
            raw_files.append(dict(domain=domain,split=split,parent_sha256=inputs.sha(old_path),sha256=inputs.sha(current_path),unchanged=inputs.sha(old_path)==inputs.sha(current_path)))
    require(len(raw_changed)==1,'Raw repair must change exactly one record')
    return dict(proof,manifest_sha256=inputs.sha(path),parent_cache_sha256=inputs.sha(parent_path),cache_sha256=inputs.sha(new_path),
        cache=str(new_path),data_manifest=str(data_path),raw_scope=raw_changed,raw_files=raw_files,
        known_excluded_issue=dict(id=REPAIR_ID,index=REPAIR_INDEX,split='train',domain='wikipedia',world='B',
            parent_cache=str(parent_path),training_target_step=940,reason=data['repair_reason'],
            original_artifacts_preserved=True,active_dataset_error=False),recorded_preservation_recomputed=True)


def audit_preparation(root,inputs=None,repair=None):
    """Only completed feature jobs and optional probe; never training artifacts."""
    inputs=inputs or Inputs(root);repair=audit_repair(inputs) if repair is None else repair
    packets={};summaries=[];raw_records={};semantic_errors=[];overlapping_answers=[];query_overlaps=[]
    from vera_mem.coldstart_teacher import question_exposes_answer
    for domain in ("wikipedia","matched","synthetic"):
        directory=inputs.root/f"coldstart_prepare_{domain}_20261006"/"features"
        manifest=read_json(directory/"manifest.json")
        require(manifest.get("complete") is True and manifest["stage"]=="prepare","Incomplete feature preparation")
        cfg=manifest["configuration"]
        require(cfg["domain"]==domain,"Feature domain mismatch")
        for split in ("train","dev","confirm"):
            repaired=domain=='wikipedia' and split=='train' and repair['status']=='passed'
            active_directory=inputs.root/REPAIR_DIR if repaired else directory
            cache=active_directory/(split+".pt");packet=inputs.load(str(cache));rows=packet["rows"]
            data_manifest=Path(repair['data_manifest']) if repaired else inputs.resolve(str(Path(cfg["data_dir"])/"manifest.json"))
            source=inputs.resolve(str(data_manifest.parent/domain/(split+".jsonl")))
            raw=read_jsonl(source);raw_records[(domain,split)]=raw;packets[(domain,split)]=packet
            require(packet["task"]==domain and packet["split"]==split,"Feature packet domain/split mismatch")
            require(packet["data_manifest_sha256"]==inputs.sha(data_manifest) and packet["data_jsonl_sha256"]==inputs.sha(source),"Feature source data hash mismatch")
            require(len(raw)==len(rows)==manifest["result"][split]["records"],"Feature record count mismatch")
            views=1 if split=="train" else 2
            require(packet["q"].shape[:2]==(len(rows),views) and packet["qids"]==['canonical','heldout'][:views] and packet["sids"]==['canonical','heldout'][:views],"Query view leakage/dimensions")
            for name in ("last","pool"):
                require(packet[name].shape[:3]==(len(rows),2,views),"Support view leakage/dimensions")
            for name in ("q","last","pool"):
                require(all(bool(torch.isfinite(part).all()) for part in packet[name].split(128)),"Nonfinite cached feature")
            repairs=packet["token_repairs"]
            require(repairs==read_json(active_directory/(split+"_repairs.json")) and len(repairs)==manifest["result"][split]["token_repairs"],"Repair sidecar/manifest mismatch")
            repairs_by_id={r['id']:r for r in repairs};require(len(repairs_by_id)==len(repairs),"Duplicate token repair")
            by_id={r['id']:r for r in rows};by_answer=defaultdict(list);lengths=Counter();changed=0
            for r in rows:by_answer[r['a']].append(r)
            for row_index,(r,original) in enumerate(zip(rows,raw)):
                for view,question in enumerate(r['questions']):
                    for world in ('a','b'):
                        if question_exposes_answer(question,r[world]):
                            query_overlaps.append(dict(domain=domain,split=split,index=row_index,id=r['id'],world=world,view=view,
                                answer=r[world],question=question,anchor=r['provenance'].get('anchor')))
                require(all(r[k]==original[k] for k in ('id','entity','relation','a','split')),"Prepared identity/A label changed")
                require(r['questions']==original['questions'][:views] and r['supports'][0]==original['supports'][0][:views],"Prepared query/A support changed")
                if domain=='synthetic':
                    require(r['supports'][1]==[s.replace(r['a'],r['b'],1) for s in r['supports'][0]],"B is not a single source span replacement")
                    require(all(s.count(r['a'])==1 for s in r['supports'][0]),"A span ambiguous")
                else:
                    anchor=r['provenance']['anchor']
                    for view in range(views):
                        source_a=r['supports'][0][view]
                        positions=[m.start() for m in re.finditer('(?='+re.escape(anchor)+')',source_a)]
                        occurrences=len(list(re.finditer('(?='+re.escape(r['a'])+')',source_a)))
                        if occurrences!=1:overlapping_answers.append(dict(domain=domain,split=split,index=row_index,id=r['id'],view=view,answer_occurrences=occurrences))
                        expected_b=None
                        if len(positions)==1:
                            end=positions[0]+len(anchor);tail=source_a[end:];offset=end+len(tail)-len(tail.lstrip())
                            if ' '.join(tail.split()[:3])==r['a']:
                                expected_b=source_a[:offset]+r['b']+source_a[offset+len(r['a']):]
                        for world,label in ((0,'a'),(1,'b')):
                            text=r['supports'][world][view]
                            points=[m.start() for m in re.finditer('(?='+re.escape(anchor)+')',text)]
                            after=[' '.join(text[i+len(anchor):].split()[:3]) for i in points]
                            if after!=[r[label]]:
                                semantic_errors.append(dict(domain=domain,split=split,index=row_index,id=r['id'],world=label,view=view,
                                    reason='anchor must occur exactly once and its next three words must match label',anchor_occurrences=len(points),actual_continuations=after,expected=r[label]))
                        if expected_b is None or r['supports'][1][view]!=expected_b:
                            semantic_errors.append(dict(domain=domain,split=split,index=row_index,id=r['id'],world='b',view=view,
                                reason='B differs from replacement at the uniquely addressed anchor boundary'))
                require(all(r[k].casefold() not in q.casefold() for q in r['questions'] for k in ('a','b')),"Question contains answer")
                require(r['a_tokens'] and r['b_tokens'] and r['a_tokens'][0]!=r['b_tokens'][0],"Token repair did not separate first tokens")
                require(all(1<len(r[k+'_tokens'])<32 for k in ('a','b')),"Answer token budget")
                lengths.update(len(r[k+'_tokens']) for k in ('a','b'))
                donors=[d for d in by_answer[r['b']] if d['entity']!=r['entity']]
                require(donors,"B donor absent from actual same-split A bank")
                if r['id'] in repairs_by_id:
                    token_repair=repairs_by_id[r['id']];changed+=1
                    require(token_repair['old_b']==original['b'] and token_repair['new_b']==r['b'] and token_repair==r['provenance']['token_repair'],"Repair provenance differs")
                    require(token_repair['donor_id'] in {d['id'] for d in donors},"Wrong repair donor identity")
                else:require(r['b']==original['b'],"Undeclared B answer change")
                if domain!='synthetic':
                    require(any(d['provenance']['page_id']==r['provenance']['b_donor_page_id'] and d['provenance']['cluster_id']==r['provenance']['b_donor_cluster_id'] for d in donors),"Wiki donor page/cluster provenance differs")
                    require(all(r['b'].casefold() not in s.casefold() for s in r['supports'][0]),"B already present in original observation")
            require(changed==len(repairs),"Repair count mismatch")
            summaries.append(dict(domain=domain,split=split,records=len(rows),views=views,token_repairs=len(repairs),
                answer_token_length_counts=dict(sorted(lengths.items())),cache_sha256=inputs.sha(cache),data_jsonl_sha256=inputs.sha(source),data_manifest_sha256=inputs.sha(data_manifest),
                feature_shapes={name:list(packet[name].shape) for name in ('q','last','pool')},finite=True,valid_actual_a_bank_b_donors=True))
    from vera_mem.coldstart_run import BankSampler
    for domain in ('wikipedia','matched','synthetic'):
        affected=[r for r in query_overlaps if r['domain']==domain and r['split']=='train']
        if not affected:continue
        sampler=BankSampler(packets[(domain,'train')]['rows'],63042);steps={r['index']:[] for r in affected}
        for step in range(1,(512 if domain=='synthetic' else 1024)+1):
            sample=sampler.next()
            for index in set(steps)&set(sample['targets']):steps[index].append(step)
        for row in affected:
            row['training_target_steps']=steps[row['index']]
            row['in_common_statistics_first8192']=domain in {'wikipedia','matched'} and row['index']<8192
    overlap_counts=[dict(domain=domain,split=split,record_count=len({r['id'] for r in query_overlaps if r['domain']==domain and r['split']==split}),
        events=sum(r['domain']==domain and r['split']==split for r in query_overlaps))
        for domain in ('wikipedia','matched','synthetic') for split in ('train','dev','confirm')]
    pairing=[]
    for split in ('train','dev','confirm'):
        left,right=packets[('wikipedia',split)],packets[('matched',split)]
        keys=('id','entity','relation','a','b','a_tokens','b_tokens','questions')
        require(len(left['rows'])==len(right['rows']) and left['token_repairs']==right['token_repairs'],"Paired record or repair mismatch")
        require(all(all(a[k]==b[k] for k in keys) for a,b in zip(left['rows'],right['rows'])),"Wiki/matched IDs/labels/tokens/questions differ")
        q_equal=all(torch.equal(a,b) for a,b in zip(left['q'].split(128),right['q'].split(128)))
        maxdiff=max(float((a.float()-b.float()).abs().max()) for a,b in zip(left['q'].split(128),right['q'].split(128)))
        pairing.append(dict(split=split,records=len(left['rows']),gold_query_identity_exact=True,token_repairs_equal=True,
            query_features_bitwise_equal=q_equal,query_features_max_absolute_difference=maxdiff,
            token_repairs=len(left['token_repairs']),paired_labels_sha256=digest([{k:r[k] for k in keys} for r in left['rows']])))
    separation=[]
    for domain in ('wikipedia','matched','synthetic'):
        seen_ids=set();seen_entities=set();seen_answers=set();seen_pages=set();seen_clusters=set()
        for split in ('train','dev','confirm'):
            rows=packets[(domain,split)]['rows'];ids={r['id'] for r in rows};entities={r['entity'] for r in rows};answers={r[k].casefold() for r in rows for k in ('a','b')}
            require(not (ids&seen_ids or entities&seen_entities or answers&seen_answers),'Prepared cross-split identity/answer overlap')
            pages={r['provenance']['page_id'] for r in rows} if domain!='synthetic' else set()
            clusters={r['provenance']['cluster_id'] for r in rows} if domain!='synthetic' else set()
            require(not (pages&seen_pages or clusters&seen_clusters),'Prepared cross-split article cluster overlap')
            separation.append(dict(domain=domain,split=split,entities=len(entities),full_answers=len(answers),article_clusters=len(clusters),intersections_with_earlier_splits_zero=True))
            seen_ids|=ids;seen_entities|=entities;seen_answers|=answers;seen_pages|=pages;seen_clusters|=clusters
    probe=None
    try:probe_path=inputs.resolve('/ssd3/chenhan/VeRA-Mem-Workspace/data/coldstart_wiki_20261006/synthetic_train_probe.pt')
    except FileNotFoundError:pass
    else:
        p=inputs.load(str(probe_path));original=packets[('synthetic','train')]
        require(p.get('diagnostic')==PROBE and p['original_split']=='train' and p['task']=='synthetic_train_probe','Probe provenance missing')
        require(p['source_cache_sha256']==inputs.sha(inputs.root/'coldstart_prepare_synthetic_20261006/features/train.pt'),'Probe source cache mismatch')
        from vera_mem.coldstart_run import BankSampler
        sampler=BankSampler(original['rows'],63042);expected=[]
        for _ in range(16):expected.extend(sampler.next()['targets'])
        require(p['source_indices']==expected and len(p['rows'])==128,'Probe target schedule mismatch')
        for r,index in zip(p['rows'],expected):
            require(r['id']==original['rows'][index]['id'] and r['split']=='train','Probe identity/original split mismatch')
            require(r['questions']==original['rows'][index]['questions']*2 and r['supports']==[v*2 for v in original['rows'][index]['supports']],'Probe uses novel views')
        require(torch.equal(p['q'][:,0],p['q'][:,1]) and torch.equal(p['q'][:,0],original['q'][expected,0]),'Probe query feature mismatch')
        for name in ('last','pool'):
            require(torch.equal(p[name][:,:,0],p[name][:,:,1]) and torch.equal(p[name][:,:,0],original[name][expected,:,0]),'Probe support feature mismatch')
        probe=dict(records=128,cache_sha256=inputs.sha(probe_path),diagnostic=PROBE,first128_training_targets_verified=True,duplicate_canonical_views_bitwise_equal=True)
    return dict(protocol='coldstart-preparation-audit-v2',complete=repair['status']=='passed',checks_passed=not semantic_errors,cuda_initialized=torch.cuda.is_initialized(),
        prepared=summaries,wiki_matched_pairing=pairing,separation=separation,train_memorization_probe=probe,
        semantic_errors=semantic_errors,globally_repeated_answer_spans=overlapping_answers,feature_repair=repair,
        normalized_question_answer_overlap=dict(counts=overlap_counts,records=query_overlaps,
            normalization='NFKC + casefold + Unicode punctuation to spaces + whitespace collapse; full phrase bounded by spaces; articles preserved',
            interpretation='Reported shared training-data limitation; records are not silently repaired or omitted from main training. No-target-update does not mean absent from normalization or foundation source corpus.'),
        known_excluded_issues=[repair['known_excluded_issue']] if repair['status']=='passed' else [],
        scope='Only completed preparation caches/source JSONL and optional canonical train probe; no formal warm checkpoint or generation output read.',
        limitations=['Stored token arrays and their lengths are compared exactly; the tokenizer is not reloaded to independently re-encode source strings.',
            'Article isolation covers observed exact-dedup clusters; general semantic near-duplicate exclusion is not claimed.',
            'Question/answer separation must be qualified by the normalized overlap report; literal-string checks alone do not exclude punctuation-normalized phrase matches.'])


def audit_training(directory,manifest,inputs):
    cfg=manifest["configuration"];status=read_json(directory/"training_status.json");rows=read_jsonl(directory/"training.jsonl")
    require(status.get("complete") is True and status==manifest["result"],"Training status incomplete/different")
    require(len(rows)==cfg["updates"]==status["updates"] and [r["step"] for r in rows]==list(range(1,len(rows)+1)),"Training update sequence")
    packet=inputs.load(cfg["cache"],manifest["cache_sha256"])
    require(packet["split"]=="train" and packet["protocol"]=="dictionary-coldstart-v1","Wrong training cache")
    records=packet["rows"]
    answer_indices=defaultdict(list)
    for i,record in enumerate(records):answer_indices[record["a"]].append(i)
    require(packet["q"].shape[1]==packet["last"].shape[2]==packet["pool"].shape[2]==1,"Heldout training feature views")
    require(all(len(r["questions"])==1 and all(len(w)==1 for w in r["supports"]) for r in records),"Heldout text present in train packet")
    schedule=hashlib.sha256();gold_schedule=hashlib.sha256();target_ids=set();bank_ids=set();totals=Counter()
    target_exposures=0
    for row in rows:
        sample=row["sample"];targets=sample["targets"];episode=sample["episode"]
        require(len(targets)==cfg["batch_size"] and len(set(targets))==len(targets),"Invalid batch targets")
        require(len(episode)==len(set(episode)) and all(type(i) is int and 0<=i<len(records) for i in episode),"Invalid episode")
        require(sample["columns"]==[episode.index(i) for i in targets],"Wrong target-to-bank mapping")
        require(sample["qviews"]==[0]*len(targets) and sample["sviews"]==[0]*len(episode),"Noncanonical training sample")
        expected_mode="dense" if cfg["routing"]=="dense_warm" and row["step"]<=len(rows)//2 else "sparse" if cfg["routing"]=="dense_warm" else cfg["routing"]
        require(row["routing"]==expected_mode,"Wrong scheduled training routing")
        labels=[dict(id=records[i]["id"],a=records[i]["a_tokens"],b=records[i]["b_tokens"]) for i in targets]
        expected_gold=sum(2*(len(r["a"])+1)+len(r["b"])+1 for r in labels)
        for key in ("gold_target_tokens","distillation_target_tokens","target_tokens"):
            same_number(row[key],expected_gold,"A/B/A whole-answer+EOS "+key)
        require(row.get("rollout_input_tokens",0)==0 and row.get("sampled_tokens",0)==0,"Unexpected on-policy tokens")
        # Required B donors are actual original-A records in this episode.
        for i in targets:
            options=[j for j in answer_indices.get(records[i]["b"],[]) if records[j]["entity"]!=records[i]["entity"]]
            require(not options or bool(set(options)&set(episode)),"Available counterfactual donor omitted from episode")
        require(len(row["gradient_norms"])==(4 if cfg["base_trainable"] else 3) and all(math.isfinite(v) and v>=0 for v in row["gradient_norms"]),"Invalid gradient norms")
        if not cfg["base_trainable"]:
            require(row["base_keys_gradient_slots"]==row["base_values_gradient_slots"]==0,"Frozen foundation received gradients")
        schedule.update(json.dumps(sample,sort_keys=True).encode());gold_schedule.update(json.dumps(labels,sort_keys=True).encode())
        target_ids.update(targets);bank_ids.update(episode);target_exposures+=len(targets)
        for key in status["totals"]:
            require(key.endswith("tokens") or key.endswith("forward_calls"),"Nonadditive training total")
            totals[key]+=row[key]
    require(schedule.hexdigest()==status["schedule_sha256"],"Training sample schedule hash mismatch")
    for key,value in totals.items():same_number(status["totals"][key],value,"Training total "+key)
    require(status["target_unique_count"]==len(target_ids) and status["bank_unique_count"]==len(bank_ids),"Unique exposure count mismatch")
    initial=inputs.load(cfg["checkpoint"],manifest["checkpoint_sha256"])
    initial_path=inputs.resolve(cfg['checkpoint'])
    initial_excluded=analysis_exclusions(initial_path.parent,inputs.root) if initial_path.is_relative_to(inputs.root) else []
    final=torch.load(directory/"last.pt",map_location="cpu",weights_only=True,mmap=True)
    require(final["protocol"]==initial["protocol"]=="dictionary-coldstart-v1" and final["step"]==len(rows),"Final checkpoint step/protocol")
    require(final["architecture"]["base_trainable"]==cfg["base_trainable"],"Foundation freeze flag mismatch")
    frozen=[name for name in initial["module"] if name in {"A","B"} or "center" in name]
    require(all(torch.equal(initial["module"][k],final["module"][k]) for k in frozen),"Fixed projections or train-fitted centers changed")
    base_equal={k:torch.equal(initial["module"][k],final["module"][k]) for k in ("base_keys","base_values")}
    usage=torch.load(directory/"usage.pt",map_location="cpu",weights_only=True)
    for axis in ("key","value"):
        observed=int(usage[axis+"_gradient"].sum());same_number(status[axis+"_gradient_coverage"],observed,"Gradient coverage")
        counts=[r["base_"+("keys" if axis=="key" else "values")+"_gradient_slots"] for r in rows]
        require(max(counts)<=observed<=sum(counts),"Gradient coverage inconsistent with per-step slots")
    if not cfg["base_trainable"]:
        require(all(base_equal.values()) and status["key_gradient_coverage"]==status["value_gradient_coverage"]==0,"Frozen foundation changed")
    for key,field in (("top1","sampled_last_forward_top1_coverage"),("topk","sampled_last_forward_topk_coverage")):
        same_number(status[field],int((usage[key]>0).sum()),"Logged routing coverage")
    require(manifest.get("backbone_unchanged") is True,"Backbone freeze flag missing")
    return dict(arm=cfg["arm"],task=packet["task"],updates=len(rows),target_exposures=target_exposures,target_unique_count=len(target_ids),bank_unique_count=len(bank_ids),
        schedule_sha256=schedule.hexdigest(),gold_token_schedule_sha256=gold_schedule.hexdigest(),token_totals=dict(totals),
        processed_input_tokens=sum(totals[k] for k in ("student_input_tokens","teacher_input_tokens","rollout_input_tokens")),
        initial_checkpoint_sha256=manifest["checkpoint_sha256"],final_checkpoint_sha256=inputs.sha(directory/"last.pt"),
        initial_checkpoint_analysis_exclusions=initial_excluded,
        base_trainable=cfg["base_trainable"],base_tensors_equal_to_initial=base_equal,frozen_projection_centers=frozen,
        key_gradient_coverage=status["key_gradient_coverage"],value_gradient_coverage=status["value_gradient_coverage"],
        training_elapsed_seconds=status["elapsed_seconds"],sample_scope="Training target and episode exposures; not whole-corpus target coverage",cache=cfg["cache"],
        cache_sha256=manifest['cache_sha256'],configuration=cfg,teacher_strategy=cfg.get('teacher_strategy','baseline'))


def select_teacher_strategy(validation):
    """Fixed priority; neither calibration nor later evaluation selects a strategy."""
    for name in ('quote_instruction','gold_annotated'):
        count,correct=validation[name]['count'],validation[name]['both_correct']
        require(type(count) is int and count==64 and type(correct) is int and 0<=correct<=count,'Teacher selection requires 64 complete validation pairs')
    return next((name for name in ('quote_instruction','gold_annotated')
                 if validation[name]['both_correct']/validation[name]['count']>=.9),None)


def verify_teacher_probe(packet,manifest,metrics,events):
    from probe_coldstart_teacher import select_rows
    from vera_mem.coldstart_teacher import STRATEGIES,make_teacher_context
    from vera_mem.metrics import normalize_answer
    selection=select_rows(packet)
    require(manifest.get('complete') is True and metrics.get('complete') is True and manifest.get('protocol')==metrics.get('protocol')=='coldstart-teacher-feasibility-v1','Incomplete/incorrect teacher probe')
    require(manifest.get('no_dev_or_confirm') is True and manifest['selection']==metrics['selection']==selection,'Teacher probe selection is not fixed train-only')
    require(manifest['result']==metrics and manifest['model_revision']==packet['model_revision'],'Teacher probe result/revision mismatch')
    require(metrics['max_new_tokens']==32 and metrics['teacher_only'] and metrics['shared_question'] and metrics['gradient_steps']==0 and metrics['backbone_unchanged'] and metrics['backbone_hash_before']==metrics['backbone_hash_after'],'Teacher probe is not frozen common-question evaluation')
    expected={(subset,strategy,index,world) for subset,indices in selection['selected_indices'].items()
              for strategy in STRATEGIES for index in indices for world in ('A','B')}
    observed={};pairs=defaultdict(dict)
    for event in events:
        key=(event['subset'],event['strategy'],event['source_index'],event['world'])
        require(key in expected and key not in observed,'Unexpected/duplicate teacher probe observation')
        observed[key]=event;subset,strategy,index,world=key;row=packet['rows'][index];wi=0 if world=='A' else 1
        answer=row[world.lower()];context=row['supports'][wi][0]
        require(event['id']==row['id'] and event['answer']==answer and event['question']==row['questions'][0] and event['source_context']==context,'Teacher probe changed target/question/source')
        require(event['teacher_context']==make_teacher_context(row,context,answer,strategy),'Teacher intervention differs from fixed strategy')
        correct=int(normalize_answer(event['prediction'])==normalize_answer(answer))
        require(event['em']==correct,'Teacher probe EM arithmetic differs')
        pairs[(subset,strategy,index)][world]=correct
    require(set(observed)==expected,'Teacher probe observations incomplete')
    counts={}
    for subset,indices in selection['selected_indices'].items():
        counts[subset]={}
        for strategy in STRATEGIES:
            values=[pairs[(subset,strategy,i)] for i in indices]
            count=len(values);both=sum(p['A'] and p['B'] for p in values)
            stored=metrics['subsets'][subset][strategy]
            require(stored['count']==count and stored['both_correct']==both and stored['a_correct']==sum(p['A'] for p in values) and stored['b_correct']==sum(p['B'] for p in values),'Teacher paired/single EM counts differ')
            same_number(stored['paired_em'],both/count,'Teacher paired rate')
            counts[subset][strategy]=dict(count=count,both_correct=both)
    selected=select_teacher_strategy(counts['validation'])
    return dict(selected_strategy=selected,rule=SELECTION_RULE,counts=counts,selection=selection,
        raw_observations=len(events),questions_unchanged=True,source_contexts_unchanged=True,
        no_dev_or_confirm=True,extra_gold_localization=selected=='gold_annotated')


def audit_teacher_selection(directory,suite,inputs):
    path=directory/'teacher_selection.json';evidence=read_json(path)
    sha=inputs.sha(path)
    require(suite.get('selection_evidence_sha256')==sha and suite.get('teacher_selection',{}).get('sha256')==sha,'Teacher selection evidence is not sealed by launcher')
    require(evidence.get('rule')==SELECTION_RULE,'Teacher selection rule changed')
    require(evidence['protocol_document_sha256']==suite['source_files_sha256']['docs/coldstart_protocol.md'],'Teacher selection protocol snapshot differs')
    artifacts={}
    for key in ('probe_manifest','probe_metrics','probe_predictions'):
        item=evidence[key];artifacts[key]=inputs.resolve(item['path'])
        require(inputs.sha(artifacts[key])==item['sha256'],'Teacher selection evidence SHA mismatch: '+key)
    manifest,metrics=read_json(artifacts['probe_manifest']),read_json(artifacts['probe_metrics'])
    require(manifest.get('complete') is True,'Teacher selection came from unfinished probe')
    packet=inputs.load(manifest['configuration']['cache'],manifest['cache_sha256'])
    proof=verify_teacher_probe(packet,manifest,metrics,read_jsonl(artifacts['probe_predictions']))
    require(proof['selected_strategy'] is not None and evidence['selected_strategy']==proof['selected_strategy'],'Teacher selection violated validation threshold/priority')
    require(metrics['artifacts']['predictions.jsonl']==inputs.sha(artifacts['probe_predictions']),'Teacher raw predictions hash mismatch')
    require(metrics['artifacts']['selection.json']==inputs.sha(artifacts['probe_manifest'].parent/'selection.json'),'Teacher selection artifact hash mismatch')
    require(read_json(artifacts['probe_manifest'].parent/'selection.json')==metrics['selection'],'Teacher selection artifact contents differ')
    for name,relative in (('probe_coldstart_teacher.py','scripts/probe_coldstart_teacher.py'),('coldstart_teacher.py','src/vera_mem/coldstart_teacher.py'),('coldstart_data.py','src/vera_mem/coldstart_data.py')):
        require(manifest['source_sha256'][name]==suite['source_files_sha256'][relative],'Teacher probe/source snapshot differs: '+name)
    return dict(proof,evidence_sha256=sha,cache_sha256=manifest['cache_sha256'],model=manifest['configuration']['model'],
                artifacts=evidence,training_suite=directory.name)


def audit_followup(trainings,evaluations,initializations,main_trainings,main_evaluations,selections,inputs,strict=False):
    expected={(split,domain,FOLLOWUP_ARM,'full') for split in ('dev','confirm') for domain in ('wikipedia','synthetic')}
    expected.add(('train_probe','synthetic_train_probe',FOLLOWUP_ARM,'full'))
    observed=[(r['scope'],r['domain'],r['arm'],r['base_mode']) for r in evaluations]
    require(len(set(observed))==len(observed) and set(observed)<=expected,'Duplicate or undeclared follow-up eval cell')
    indexed={r['arm']:r for r in trainings}
    require(len(indexed)==len(trainings) and set(indexed)<=FOLLOWUP_TRAIN_ARMS,'Duplicate/undeclared follow-up training arm')
    present=bool(trainings or evaluations or initializations or selections)
    if not present:return dict(status='absent',complete=True,optional=True,trainings=[],evaluations=[],expected_training_jobs=2,expected_evaluation_cells=5)
    require(len(selections)<=1 and len(initializations)<=1,'Repeated follow-up selection/prototype initialization')
    missing=sorted(expected-set(observed));missing_training=sorted(FOLLOWUP_TRAIN_ARMS-set(indexed))
    complete=not missing and not missing_training and len(initializations)==len(selections)==1
    if strict:require(complete,'Supplementary teacher follow-up coverage incomplete')
    main={r['arm']:r for r in main_trainings}
    if 'wiki_teacher_warm' in indexed:
        require(len(selections)==1,'Supplementary warmup lacks sealed teacher selection')
        warm=indexed['wiki_teacher_warm'];selection=selections[0]
        require(warm['teacher_strategy']==selection['selected_strategy'] and warm['cache_sha256']==selection['cache_sha256'] and warm['configuration']['model']==selection['model'],'Supplementary warmup strategy/cache/model differs from selection')
        if 'wiki_warm' in main:
            reference=main['wiki_warm']
            for key in ('cache_sha256','updates','target_exposures','schedule_sha256','gold_token_schedule_sha256','initial_checkpoint_sha256'):
                require(warm[key]==reference[key],'Supplementary warmup changed '+key)
            for key in ('gold_target_tokens','distillation_target_tokens','target_tokens','student_input_tokens'):
                require(warm['token_totals'][key]==reference['token_totals'][key],'Supplementary warmup changed student/gold token budget')
            for key in ('routing','base_trainable','alpha_base','seed','batch_size'):
                require(warm['configuration'][key]==reference['configuration'][key],'Supplementary warmup changed optimizer/routing configuration')
    if initializations:
        init=initializations[0];require('wiki_teacher_warm' in indexed,'Prototype initialization lacks completed parent warmup')
        require(init['initial_checkpoint_sha256']==indexed['wiki_teacher_warm']['final_checkpoint_sha256'] and init['cache_sha256']==indexed['wiki_teacher_warm']['cache_sha256'],'Supplementary prototype initialization parent/cache mismatch')
    if FOLLOWUP_ARM in indexed:
        final=indexed[FOLLOWUP_ARM];cfg=final['configuration']
        require(final['teacher_strategy']=='baseline' and cfg['routing']=='straight_through' and cfg['base_trainable'] and cfg['alpha_base']==.25,'Supplementary transfer must restore baseline teacher with sparse-forward/dense-backward prototype training')
        require(len(initializations)==1 and final['initial_checkpoint_sha256']==initializations[0]['final_checkpoint_sha256'],'Supplementary transfer checkpoint skips selected-teacher prototypes')
        if 'straight_through' in main:
            reference=main['straight_through']
            for key in ('cache_sha256','updates','target_exposures','schedule_sha256','gold_token_schedule_sha256'):
                require(final[key]==reference[key],'Supplementary synthetic transfer changed '+key)
            for key in ('gold_target_tokens','distillation_target_tokens','target_tokens','student_input_tokens','teacher_input_tokens'):
                require(final['token_totals'][key]==reference['token_totals'][key],'Supplementary transfer changed target/input budget')
    for run in evaluations:
        require(FOLLOWUP_ARM in indexed and run['checkpoint_sha256']==indexed[FOLLOWUP_ARM]['final_checkpoint_sha256'],'Supplementary evaluation uses wrong final checkpoint')
        reference=next((r for r in main_evaluations if r['scope']==run['scope'] and r['domain']==run['domain'] and r['arm']=='straight_through' and r['base_mode']=='full'),None)
        if reference:require(run['arithmetic']['comparison_key']==reference['arithmetic']['comparison_key'],'Supplementary evaluation changed cases/questions/features')
    if complete:
        require({'wiki_warm','straight_through'}<=set(main),'Complete follow-up requires the corresponding main-arm comparisons')
        require(all(any(r['scope']==f['scope'] and r['domain']==f['domain'] and r['arm']=='straight_through' and r['base_mode']=='full' for r in main_evaluations) for f in evaluations),'Complete follow-up lacks matching main evaluation cells')
    return dict(status='passed' if complete else 'pending',complete=complete,optional=True,
        missing_training_arms=missing_training,missing_evaluation_cells=missing,trainings=trainings,evaluations=evaluations,
        initializations=initializations,selection=selections[0] if selections else None,
        expected_training_jobs=2,expected_evaluation_cells=5,main_counts_unchanged=True,
        verified_generation_calls=sum(r['arithmetic']['costs']['generation_calls'] for r in evaluations),
        interpretation='Supplementary selected-teacher follow-up. Gold annotation is extra label-localization supervision, not raw-context corpus learning. Main 11 training/52 evaluation scope remains separate.')


def warm_pairing(trainings,inputs):
    indexed={t["arm"]:t for t in trainings}
    if not {"wiki_warm","matched_warm"}<=set(indexed):return dict(status="pending_both_completed_warmups")
    wiki,matched=indexed["wiki_warm"],indexed["matched_warm"]
    for key in ("updates","target_exposures","schedule_sha256","gold_token_schedule_sha256","initial_checkpoint_sha256"):
        require(wiki[key]==matched[key],"Wiki/matched warmup differs: "+key)
    for key in ("gold_target_tokens","target_tokens","distillation_target_tokens","student_input_tokens"):
        require(wiki["token_totals"][key]==matched["token_totals"][key],"Wiki/matched token budget mismatch: "+key)
    left,right=inputs.load(wiki["cache"]),inputs.load(matched["cache"])
    require(len(left["rows"])==len(right["rows"]) and left["token_repairs"]==right["token_repairs"],"Wiki/matched record count or repairs differ")
    keys=("id","entity","relation","a","b","a_tokens","b_tokens","questions")
    require(all(all(a[k]==b[k] for k in keys) for a,b in zip(left["rows"],right["rows"])),"Wiki/matched gold/query mismatch")
    return dict(status="passed",records=len(left["rows"]),updates=wiki["updates"],target_exposures=wiki["target_exposures"],
        schedule_sha256=wiki["schedule_sha256"],gold_token_schedule_sha256=wiki["gold_token_schedule_sha256"],
        gold_target_tokens=wiki["token_totals"]["gold_target_tokens"],student_input_tokens=wiki["token_totals"]["student_input_tokens"],
        wiki_teacher_input_tokens=wiki["token_totals"]["teacher_input_tokens"],matched_teacher_input_tokens=matched["token_totals"]["teacher_input_tokens"],
        note="Gold sequences and student inputs match exactly; teacher/support input costs are intentionally different. EOS identity follows the same pinned model revision.")


def audit_usage(rows,raw,summary,bank_hash,size,topk):
    totals=Counter();slots=defaultdict(lambda:Counter());expected=[]
    for i,event in enumerate(raw):
        expected.append((i,"generation",event["method"]=="teacher"))
        expected.extend((i,"answer_scoring",event["method"]=="teacher") for _ in event["answer_results"])
    require(len(rows)==len(expected),"Missing foundation generation/scoring usage calls")
    for index,(row,target) in enumerate(zip(rows,expected)):
        require(row["call_index"]==index and (row["prediction_index"],row["kind"],row["teacher"])==target,"Foundation usage order/teacher mismatch")
        require(row["bank_hash"]==bank_hash and row["records"]==size,"Foundation usage bank mismatch")
        if row["teacher"]:require(row["search_calls"]==row["query_positions"]==0,"Teacher used foundation memory")
        seen=set();top1=topn=0;mass=0.
        for entry in row["sparse_counts"]:
            slot=entry["index"];require(0<=slot<size and slot not in seen,"Invalid/duplicate usage slot");seen.add(slot)
            require(type(entry["top1"]) is int and type(entry["topk"]) is int and entry["topk"]>0,"Invalid sparse usage counts")
            for key in ("top1","topk","weight_mass"):
                value=entry[key];require(math.isfinite(value) and value>=0,"Invalid usage value");slots[slot][key]+=value
            top1+=entry["top1"];topn+=entry["topk"];mass+=entry["weight_mass"]
        if size:
            require(top1==row["query_positions"] and topn==row["query_positions"]*min(topk,size),"Foundation position count mismatch")
            require(math.isclose(mass,row["query_positions"],rel_tol=1e-5,abs_tol=1e-5),"Foundation attention weights not normalized")
        else:require(not seen and mass==0,"Empty foundation has occupied slots")
        same_number(row["weight_mass"],mass,"Per-call foundation mass")
        used1=sum(e["top1"]>0 for e in row["sparse_counts"])
        same_number(row["top1_used"],used1,"Per-call top1 slots")
        same_number(row["topk_used"],len(seen),"Per-call topk slots")
        probabilities=[e["weight_mass"]/mass for e in row["sparse_counts"] if e["weight_mass"]>0] if mass else []
        entropy=-sum(p*math.log(p) for p in probabilities)
        same_number(row["weight_entropy"],entropy,"Per-call entropy")
        same_number(row["effective_slots"],math.exp(entropy) if mass else 0.,"Per-call effective slots")
        for key in ("search_calls","query_positions"):totals[key]+=row[key]
    observed={r["index"]:r for r in summary["sparse_counts"]}
    require(set(observed)==set(slots),"Aggregate foundation slots differ")
    for slot,values in slots.items():
        for key,value in values.items():same_number(observed[slot][key],value,"Foundation aggregate "+key)
    for key,value in totals.items():same_number(summary[key],value,"Foundation aggregate "+key)
    total=sum(s["weight_mass"] for s in slots.values());same_number(summary["weight_mass"],total,"Foundation total mass")
    positive=[s["weight_mass"]/total for s in slots.values() if s["weight_mass"]>0] if total else []
    entropy=-sum(p*math.log(p) for p in positive)
    same_number(summary["weight_entropy"],entropy,"Foundation entropy")
    same_number(summary["effective_slots"],math.exp(entropy) if total else 0,"Foundation effective slots")
    same_number(summary["top1_used"],sum(s["top1"]>0 for s in slots.values()),"Foundation top1 coverage")
    same_number(summary["topk_used"],len(slots),"Foundation topk coverage")
    return dict(calls=len(rows),generation_calls=len(raw),**totals,active_slots=len(slots),weight_mass=total)


def audit_foundation(directory,manifest,inputs):
    metrics=read_json(directory/"metrics.json");cold=metrics["coldstart"];cfg=manifest["configuration"]
    require(cold["protocol"]=="coldstart-cpu-banks-v1","Unsupported foundation protocol")
    require(cold["bank_hash_before"]==cold["bank_hash_after"] and cold["original_shared_weights_before"]==cold["original_shared_weights_after"],"Foundation/shared weights changed")
    require(cold["original_state_before"]==cold["original_state_after"],"Original module state was not restored")
    require(cold["evaluated_routing"]=="sparse" and cold["foundation_cpu"] and cold["episodic_cpu"],"Evaluation not sparse CPU banks")
    require(not cold["values_rms_recalibrated"] and not cold["count_bias_applied"] and not cold["dynamic_records_compressed"],"Undeclared compression transform")
    for field in ("foundation_snapshot","foundation_usage"):
        require(inputs.sha(directory/cold[field])==cold[field+"_sha256"],"Foundation artifact SHA mismatch")
    packet=torch.load(directory/cold["foundation_snapshot"],map_location="cpu",weights_only=True)
    checkpoint=inputs.load(cfg["checkpoint"],manifest["checkpoint_sha256"])
    for k,field in (("base_keys","original_keys"),("base_values","original_values")):
        require(torch.equal(packet[field],checkpoint["module"][k]),"Foundation source differs from trained checkpoint")
    require(packet["original_shared_weights"]==cold["original_shared_weights_before"],"Snapshot module fingerprint mismatch")
    mode=cold["base_mode"];require(mode==cfg["base_mode"]==packet["mode"],"Foundation mode mismatch")
    original=len(packet["original_keys"]);n=packet["active_records"]
    require(n==cold["active_records"] and original==cold["source_records"],"Foundation size mismatch")
    if mode in ("full","off","random256"):
        selected=list(range(original)) if mode=="full" else []
        if mode=="random256":
            generator=torch.Generator(device="cpu").manual_seed(packet["compression_seed"])
            selected=sorted(torch.randperm(original,generator=generator)[:packet["merge_count"]].tolist())
        require(selected==packet["source_indices"] and len(selected)==n,"Wrong foundation subset")
        require(torch.equal(packet["keys"],packet["original_keys"][selected]) and torch.equal(packet["values"],packet["original_values"][selected]),"Subset changed original foundation payload")
    elif mode=="cluster256":
        labels=packet["merge"]["assignments"];counts=torch.bincount(labels,minlength=n)
        require(len(labels)==original and len(counts)==n and bool((counts>0).all()),"Invalid/empty merge assignments")
        means=torch.zeros_like(packet["values"]).index_add_(0,labels,packet["original_values"])/counts[:,None]
        require(torch.equal(means,packet["values"]),"Merged values are not unnormalized arithmetic means")
        normalized=torch.nn.functional.normalize(packet["original_keys"],dim=-1)
        key_means=torch.zeros_like(packet["keys"]).index_add_(0,labels,normalized)/counts[:,None]
        for index in (key_means.norm(dim=-1)<=1e-12).nonzero().flatten().tolist():
            key_means[index]=normalized[(labels==index).nonzero().flatten()[0]]
        require(torch.equal(torch.nn.functional.normalize(key_means,dim=-1),packet["keys"]),"Merged keys differ from spherical means")
        require(packet["members"]==[(labels==i).nonzero().flatten().tolist() for i in range(n)],"Merge member provenance mismatch")
        require(torch.equal(counts,packet["merge"]["counts"]),"Merge counts mismatch")
    else:raise ValueError("Unknown foundation mode")
    store=restore(packet["store"])
    replay=PersistentVectorDB(store.key_dim,store.value_dim,store.temperature,store.top_k)
    for identity,key,value in zip(packet["ids"],packet["keys"],packet["values"]):replay.write(identity,key,value,0)
    require(store.hash()==replay.hash()==packet["bank_hash"]==cold["bank_hash_before"],"Foundation CPU replay hash mismatch")
    require(packet["alpha_base"]==cold["alpha_base"]==checkpoint["architecture"]["alpha_base"],"Foundation alpha differs from checkpoint")
    raw=read_jsonl(directory/"predictions.jsonl")
    usage=audit_usage(read_jsonl(directory/cold["foundation_usage"]),raw,cold["usage"],store.hash(),n,store.top_k)
    return dict(mode=mode,source_records=original,active_records=n,bank_hash=store.hash(),alpha_base=cold["alpha_base"],usage=usage,
        values_rms_recalibrated=False,episodic_records_compressed=False)


def run_audit(root,require_complete=False):
    root=Path(root).resolve();inputs=Inputs(root);trainings=[];evaluations=[];pending=[];suites=[];excluded=[]
    followup_initializations=[];selections=[];followup_suites=set()
    require(root.is_dir(),"Missing runs-root directory")
    for path in sorted(root.rglob("coldstart_*/suite.json")):
        if "source" in path.relative_to(root).parts:continue
        markers=analysis_exclusions(path.parent,root)
        if markers:
            excluded.append(dict(suite=path.parent.name,reason='ancestor analysis_excluded marker',markers=markers,
                scope='All descendant jobs excluded without reading suite/job manifests, scores, or checkpoints.'))
            continue
        suite=read_json(path);require(suite.get("protocol")=="coldstart-experiment-v1","Unknown coldstart suite")
        if any(t in path.parent.name for t in ("smoke","preflight")):
            excluded.append(dict(suite=path.parent.name,reason="smoke/preflight excluded from formal coverage"));continue
        if "source_files_sha256" not in suite:
            require(not suite["complete"],"Completed suite lacks source snapshot")
            pending.append(dict(suite=path.parent.name,job="source_snapshot_pending"));continue
        planned={j["name"]:j for j in read_json(path.parent/"plan.json")}
        planned_arms={j['arguments'][j['arguments'].index('--arm')+1] for j in planned.values() if '--arm' in j.get('arguments',[])}
        if planned_arms & (FOLLOWUP_TRAIN_ARMS|{'teacher_prototypes'}):followup_suites.add(path.parent.name)
        if 'teacher_selection' in suite:selections.append(audit_teacher_selection(path.parent,suite,inputs))
        for name,sha in suite["source_files_sha256"].items():require(inputs.sha(path.parent/"source"/name)==sha,"Frozen source snapshot changed")
        suites.append(dict(name=path.parent.name,complete=suite["complete"],jobs=len(suite["jobs"])))
        for job in suite["jobs"]:
            require(job["name"] in planned,"Unplanned coldstart job")
            markers=analysis_exclusions(path.parent/job['name'],root)
            if markers:
                excluded.append(dict(suite=path.parent.name,job=job['name'],reason='ancestor analysis_excluded marker',markers=markers))
                continue
            if job.get("status")!="complete" or job.get("exit_code")!=0:
                pending.append(dict(suite=path.parent.name,job=job["name"]));continue
            directory=path.parent/job["name"];manifest=read_json(directory/"manifest.json")
            require(manifest.get("complete") and manifest["protocol"]=="dictionary-coldstart-v1","Incomplete child manifest")
            if job.get("manifest_sha256"):require(inputs.sha(directory/"manifest.json")==job["manifest_sha256"],"Child manifest differs from launcher")
            common=dict(suite=path.parent.name,job=job["name"],suite_partial=not suite["complete"],job_elapsed_seconds=job.get("elapsed_seconds"))
            stage=manifest["stage"]
            for name,sha in manifest.get('sources',{}).items():
                require(suite['source_files_sha256'].get('src/vera_mem/'+name)==sha,'Child source differs from frozen suite snapshot')
            if stage=="train":
                cfg=manifest["configuration"]
                require(cfg['arm'] in TRAIN_ARMS|FOLLOWUP_TRAIN_ARMS,'Undeclared training arm')
                expected_updates=1024 if cfg["arm"] in {"wiki_warm","matched_warm","wiki_joint_warm","wiki_teacher_warm"} else 512
                require(cfg["updates"]==expected_updates and cfg["batch_size"]==8 and cfg["seed"]==63042,"Undeclared training budget/seed")
                if cfg['arm'] not in FOLLOWUP_TRAIN_ARMS:require(cfg.get('teacher_strategy','baseline')=='baseline','Main experiment teacher strategy changed')
                trainings.append(dict(common,**audit_training(directory,manifest,inputs)))
            elif stage=='init' and manifest['configuration']['arm']=='teacher_prototypes':
                cfg=manifest['configuration']
                require(cfg['cluster_init'] and cfg['seed']==63042,'Supplementary prototype initialization configuration changed')
                packet=inputs.load(cfg['cache'],manifest['cache_sha256'])
                initial=inputs.load(cfg['checkpoint'],manifest['checkpoint_sha256'])
                final=inputs.load(str(directory/'last.pt'));cluster=inputs.load(str(directory/'cluster_initialization.pt'))
                require(packet['task']=='wikipedia' and packet['split']=='train' and final['step']==0,'Supplementary initialization is not train-only')
                require(set(initial['module'])==set(final['module']) and all(bitwise_equal(v,final['module'][k]) for k,v in initial['module'].items() if k not in {'base_keys','base_values'}),'Supplementary initialization changed nonprototype parameters')
                require(bitwise_equal(final['module']['base_keys'],cluster['keys']) and bitwise_equal(final['module']['base_values'],cluster['values']),'Supplementary prototype payload differs from initialization artifact')
                require(len(cluster['assignments'])==len(packet['rows']) and final['module']['base_keys'].shape[0]==1024,'Supplementary prototype source/slot counts differ')
                followup_initializations.append(dict(common,configuration=cfg,cache_sha256=manifest['cache_sha256'],
                    initial_checkpoint_sha256=manifest['checkpoint_sha256'],final_checkpoint_sha256=inputs.sha(directory/'last.pt'),
                    source_records=len(packet['rows']),prototype_count=1024,nonprototype_parameters_bitwise_unchanged=True,
                    encoder_recomputed=False))
            elif stage=="eval":
                cfg=manifest["configuration"];packet=inputs.load(cfg["cache"],manifest["cache_sha256"])
                require(packet["task"]==cfg["domain"],"Evaluation cache domain mismatch")
                probe=packet.get("diagnostic")==PROBE
                require(probe==(cfg["domain"]=="synthetic_train_probe"),"Probe/domain marking mismatch")
                if probe:
                    require(packet["original_split"]=="train" and all(r["split"]=="train" and r["questions"][0]==r["questions"][1] and all(w[0]==w[1] for w in r["supports"]) for r in packet["rows"]),"Probe is not duplicated canonical training data")
                result=audit_evaluation(directory,manifest)
                require(cfg.get('teacher_strategy','baseline')=='baseline','Evaluation changed the ordinary contextual teacher')
                foundation=audit_foundation(directory,manifest,inputs);banks=audit_episodic(directory)
                evaluations.append(dict(common,arm=cfg["arm"],domain=cfg["domain"],scope="train_probe" if probe else packet["split"],
                    diagnostic=packet.get("diagnostic"),base_mode=cfg["base_mode"],checkpoint_sha256=manifest["checkpoint_sha256"],
                    arithmetic=result,foundation=foundation,episodic_bank_replay=banks))
        for name in set(planned)-{j["name"] for j in suite["jobs"]}:
            markers=analysis_exclusions(path.parent/name,root)
            if markers:excluded.append(dict(suite=path.parent.name,job=name,reason='ancestor analysis_excluded marker',markers=markers))
            else:pending.append(dict(suite=path.parent.name,job=name))
    followup_trainings=[r for r in trainings if r['arm'] in FOLLOWUP_TRAIN_ARMS]
    followup_evaluations=[r for r in evaluations if r['arm']==FOLLOWUP_ARM]
    trainings=[r for r in trainings if r['arm'] not in FOLLOWUP_TRAIN_ARMS]
    evaluations=[r for r in evaluations if r['arm']!=FOLLOWUP_ARM]
    pairing=warm_pairing(trainings,inputs)
    trained={t["arm"]:t for t in trainings}
    require(len(trained)==len(trainings),"Duplicate training arm")
    for run in evaluations:
        source={"wiki_warm":"wiki_warm_transfer","matched_warm":"matched_warm_transfer"}.get(run["arm"],run["arm"])
        if source in trained:
            require(run["checkpoint_sha256"]==trained[source]["final_checkpoint_sha256"],"Evaluation is not its declared final trained checkpoint")
            run["final_checkpoint_lineage_verified"]=True
        else:run["final_checkpoint_lineage_verified"]=False
    expected={(split,domain,arm,"full") for split in ("dev","confirm") for domain in ("wikipedia","synthetic") for arm in ARMS}
    expected|={(split,domain,"straight_through",mode) for split in ("dev","confirm") for domain in ("wikipedia","synthetic") for mode in ("off","random256","cluster256")}
    expected|={("train_probe","synthetic_train_probe",arm,"full") for arm in ARMS}
    observed=[(r["scope"],r["domain"],r["arm"],r["base_mode"]) for r in evaluations]
    require(len(set(observed))==len(observed) and set(observed)<=expected,"Duplicate or undeclared evaluation arm")
    groups=defaultdict(list)
    for run in evaluations+followup_evaluations:groups[(run["scope"],run["domain"])].append(run)
    for (scope,domain),members in groups.items():
        require(len({r["arithmetic"]["comparison_key"] for r in members})==1,"Cross-arm cases/prompts/background banks differ")
        for run in members:
            require(run["arithmetic"]["evaluated_facts"]==dict(dev=16,confirm=64,train_probe=8)[scope],"Wrong evaluation case budget")
            has_teacher=run["arithmetic"]["phases"]["CC"]["teacher_acceptance"] is not None
            require(has_teacher==(run["arm"]=="no_warm" and run["base_mode"]=="full"),"Undeclared teacher scope")
        baseline=next((r for r in members if r["arm"]=="no_warm"),None)
        for run in members:
            if baseline:
                run["shared_teacher_qualification"]={p:dict(count=baseline["arithmetic"]["phases"][p]["teacher_acceptance"]["strict_pair_qualified"],
                    denominator=baseline["arithmetic"]["evaluated_facts"],source_run=baseline["suite"]+"/"+baseline["job"],
                    source_comparison_key=baseline["arithmetic"]["comparison_key"]) for p in PHASES}
    for domain in ("wikipedia","synthetic"):
        for split in ("dev","confirm"):
            compression=[r for r in evaluations if r["scope"]==split and r["domain"]==domain and r["arm"]=="straight_through"]
            require(len({r["checkpoint_sha256"] for r in compression})<=1,"Compression arms use different checkpoints")
    repair=audit_repair(inputs)
    for run in trainings+followup_trainings:
        if run['arm'] in {'wiki_warm','wiki_joint_warm','wiki_teacher_warm'}:
            require(repair['status']=='passed' and inputs.sha(inputs.resolve(run['cache']))==repair['cache_sha256'],'Retained Wiki warmup did not use the repaired cache')
            run['repaired_wikipedia_cache_verified']=True
    prepare_paths=[root/f'coldstart_prepare_{domain}_20261006/features/manifest.json' for domain in ('wikipedia','matched','synthetic')]
    preparation=(audit_preparation(root,inputs,repair) if all(p.is_file() and read_json(p).get('complete') for p in prepare_paths)
                 else dict(complete=False,checks_passed=True,status='pending_preparation_manifests'))
    missing=sorted(expected-set(observed));missing_training=sorted(TRAIN_ARMS-{t["arm"] for t in trainings})
    followup=audit_followup(followup_trainings,followup_evaluations,followup_initializations,trainings,evaluations,selections,inputs,require_complete)
    main_complete=not missing and not missing_training and not [p for p in pending if p['suite'] not in followup_suites] and pairing["status"]=="passed" and all(s["complete"] for s in suites if s['name'] not in followup_suites) and preparation['complete'] and preparation['checks_passed'] and repair['status']=='passed'
    complete=main_complete and followup['complete'] and not pending and all(s['complete'] for s in suites)
    if require_complete:require(complete,"Predeclared experiment coverage incomplete")
    return dict(protocol="coldstart-independent-audit-v2",audited_at=datetime.now(timezone.utc).isoformat(),checks_passed=preparation['checks_passed'],complete=complete,
        cuda_initialized=torch.cuda.is_initialized(),suites=suites,excluded_suites=excluded,pending_jobs=pending,missing_evaluation_cells=missing,missing_training_arms=missing_training,
        matched_wiki_pairing=pairing,trainings=trainings,evaluations=evaluations,feature_repair=repair,preparation=preparation,
        main_scope=dict(complete=main_complete,expected_training_jobs=11,expected_evaluation_cells=52),followup_scope=followup,
        verified_generation_calls=sum(r["arithmetic"]["costs"]["generation_calls"] for r in evaluations),
        verified_episodic_updates=sum(r["episodic_bank_replay"]["verified_single_target_updates"] for r in evaluations),
        limitations=["Replays persisted CPU banks and recomputes logged arithmetic, not CUDA encoders or generation.",
            "Foundation usage includes prompt and answer-scoring positions; it is not first-token factual recall.",
            "Train probe uses split=dev only for compatibility; scope=train_probe is excluded from heldout results.",
            "Processed student/teacher/rollout input tokens are additive; gold, distillation and target token counts overlap.",
            "Shared teacher qualification is permitted only for identical case/prompts/cache/model fingerprints; it is not extra teacher generation.",
            "One optimization seed does not measure initialization/data-split uncertainty; root protocol governs scientific thresholds."])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--runs-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--require-complete",action="store_true");p.add_argument("--preparation-only",action="store_true")
    a=p.parse_args(argv);torch.set_num_threads(4);started=time.perf_counter()
    result=audit_preparation(a.runs_root) if a.preparation_only else run_audit(a.runs_root,a.require_complete)
    result["elapsed_seconds"]=time.perf_counter()-started
    result["audit_source_sha256"]={p.name:file_hash(p) for p in (Path(__file__),Path(__file__).with_name("audit_interface_banks.py"),Path(__file__).with_name("summarize_interface.py"),Path(__file__).with_name("replay_coldstart_banks.py"),Path(__file__).with_name("repair_coldstart_overlap.py"))}
    a.output.mkdir(parents=True,exist_ok=True);destination=a.output/"coldstart_audit.json"
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(checks_passed=result["checks_passed"],complete=result["complete"],training_jobs=len(result.get("trainings",[])),eval_jobs=len(result.get("evaluations",[])),
        generation_calls=result.get("verified_generation_calls",0),elapsed_seconds=result["elapsed_seconds"],output=str(destination))))
    if not result["checks_passed"]:raise SystemExit(2)


if __name__=="__main__":main()
