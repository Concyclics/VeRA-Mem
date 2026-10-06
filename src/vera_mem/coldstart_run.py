"""Reproducible Wikipedia cold start with separate foundation and episodic banks."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
import torch

from .context_distillation_run import REVISION, clear_bank, parameter_digest, sha256, append_jsonl
from .counterfactual_backend import CounterfactualBackend
from .dictionary_vera import DictionaryVeRA
from .interface_features import support_features
from .interface_run import make_batch
from .interface_training import interface_step
from .run import json_write

PROTOCOL = 'dictionary-coldstart-v1'


class BankSampler:
    """Exhaustive shuffled target epochs; real same-answer donors where available."""
    def __init__(self, rows, seed, batch_size=8, bank_size=72):
        self.rows, self.rng = rows, random.Random(seed)
        self.batch_size, self.bank_size = batch_size, min(bank_size, len(rows))
        self.by_answer, self.by_entity = {}, {}
        for i, row in enumerate(rows):
            self.by_answer.setdefault(row['a'], []).append(i)
            self.by_entity.setdefault(row['entity'], []).append(i)
        self.queue = []

    def next(self, nq=1, ns=1):
        if len(self.queue) < self.batch_size:
            self.queue = list(range(len(self.rows))); self.rng.shuffle(self.queue)
        targets = self.queue[:self.batch_size]; del self.queue[:self.batch_size]
        chosen = set(targets)
        for i in targets:
            row = self.rows[i]
            chosen.update(self.by_entity[row['entity']])
            for world in ('a', 'b'):
                candidates = [j for j in self.by_answer.get(row[world], []) if self.rows[j]['entity'] != row['entity']]
                if candidates: chosen.add(self.rng.choice(candidates))
        if len(chosen) > self.bank_size: raise ValueError('Hard negatives exceed bank size')
        remaining = [j for j in range(len(self.rows)) if j not in chosen]
        chosen.update(self.rng.sample(remaining, self.bank_size-len(chosen)))
        episode = sorted(chosen); self.rng.shuffle(episode)
        return dict(targets=targets, episode=episode, columns=[episode.index(i) for i in targets],
                    qviews=[0]*len(targets), sviews=[0]*len(episode))


def prepare(backend, args):
    from .coldstart_data import load_records, replace_target_after_anchor
    result = {}
    for split in args.splits.split(','):
        rows = load_records(args.data_dir, args.domain, split)
        natural = load_records(args.data_dir, 'wikipedia', split) if args.domain == 'matched' else rows
        replacements = []
        for r in rows:
            r['a_tokens'] = backend.tokenizer.encode(r['a'], add_special_tokens=False)
        for i, r in enumerate(rows):
            b = backend.tokenizer.encode(r['b'], add_special_tokens=False)
            if r['a_tokens'][0] == b[0]:
                # Same deterministic repair for natural/matched supports, no eval labels in train.
                donor = next(rows[(i+j)%len(rows)] for j in range(1,len(rows))
                             if rows[(i+j)%len(rows)]['a_tokens'][0] != r['a_tokens'][0]
                             and rows[(i+j)%len(rows)]['entity'] != r['entity']
                             and all(rows[(i+j)%len(rows)]['a'].casefold() not in s.casefold()
                                     for s in natural[i]['supports'][0]+natural[i]['questions']))
                replacements.append(dict(id=r['id'],old_b=r['b'],new_b=donor['a'],donor_id=donor['id'],
                                         reason='distinct full-answer first token required by paired distillation'))
                r['b'] = donor['a']; b = donor['a_tokens']
                if any(s.count(r['a']) != 1 for s in r['supports'][0]): raise ValueError('Ambiguous replacement')
                anchor=r['provenance'].get('anchor')
                r['supports'][1] = [replace_target_after_anchor(s,anchor,r['a'],r['b']) if anchor
                                     else s.replace(r['a'],r['b'],1) for s in r['supports'][0]]
                r['provenance']['token_repair'] = replacements[-1]
                for key, source_key in (('b_donor_page_id','page_id'),('b_donor_cluster_id','cluster_id')):
                    if key in r['provenance']: r['provenance'][key]=donor['provenance'][source_key]
            r['b_tokens'] = b
            if not b or max(len(b),len(r['a_tokens'])) >= 32: raise ValueError('Answer token budget')
        views = 1 if split == 'train' else 2
        # Preserve heldout strings in raw source only, never in training tensors/rows.
        for r in rows:
            r['questions'] = r['questions'][:views]
            r['supports'] = [world[:views] for world in r['supports']]
        q = backend.layer_features([q for r in rows for q in r['questions']]).half()
        last, pool = support_features(backend, [s for r in rows for world in r['supports'] for s in world], batch_size=16)
        n = len(rows)
        packet = dict(protocol=PROTOCOL, task=args.domain, split=split, model_revision=REVISION,
                      rows=rows,qids=['canonical','heldout'][:views],sids=['canonical','heldout'][:views],
                      q=q.view(n,views,-1),last=last.half().view(n,2,views,-1),pool=pool.half().view(n,2,views,-1),
                      writer_prefix='Remember this information: ', pooling='content-only',token_repairs=replacements,
                      data_manifest_sha256=sha256(args.data_dir/'manifest.json'),
                      data_jsonl_sha256=sha256(args.data_dir/args.domain/(split+'.jsonl')))
        torch.save(packet,args.run_dir/(split+'.pt'))
        json_write(args.run_dir/(split+'_repairs.json'),replacements)
        result[split] = dict(records=n,views=views,token_repairs=len(replacements),
                            data_manifest_sha256=packet['data_manifest_sha256'],data_jsonl_sha256=packet['data_jsonl_sha256'])
        print(json.dumps(result[split]),flush=True)
    return result


def load_module(path, device):
    c = torch.load(path,map_location='cpu',weights_only=True)
    if c['protocol'] != PROTOCOL: raise ValueError('Wrong coldstart checkpoint protocol')
    m = DictionaryVeRA(**c['architecture']); m.load_state_dict(c['module'])
    return m.to(device), c


def save_module(module,path,**extra):
    torch.save(dict(protocol=PROTOCOL,architecture=module.configuration(),module=module.state_dict(),**extra),path)


def initialize(packet,args):
    if packet['split'] != 'train': raise ValueError('Train-only initialization')
    if args.checkpoint:
        module, checkpoint = load_module(args.checkpoint,'cpu')
    else:
        module = DictionaryVeRA(9728,2560,rank=64,key_dim=64,top_k=4,temperature=.05,
                                seed=args.seed,writer_mode='masked_mean',base_size=1024,
                                base_trainable=False,alpha_base=0.)
        module.fit_statistics(packet['q'][:8192].float().flatten(0,1),packet['last'][:8192].float().flatten(0,2))
        module.fit_value_statistics(packet['pool'][:8192].float().flatten(0,2))
    if args.cluster_init:
        with torch.no_grad():
            keys=torch.cat([module.encode_key(x[:,0,0].float()) for x in packet['last'].split(256)])
            vals=torch.cat([module.encode_value(x[:,0,0].float()) for x in packet['pool'].split(256)])
            audit=module.initialise_base(keys,vals,split='train',seed=args.seed,max_iterations=20)
        torch.save(audit,args.run_dir/'cluster_initialization.pt')
    save_module(module,args.run_dir/'last.pt',step=0)
    return dict(train_records=len(packet['rows']),cluster_init=args.cluster_init,architecture=module.configuration())


def train(backend,module,packet,args):
    if packet['split'] != 'train': raise ValueError('Train-only optimization')
    if module.train_B or module.value_mlp_hidden:
        raise ValueError('This controlled pilot requires frozen B and a linear value writer')
    for key in ('q','last','pool'): packet[key]=packet[key].float().to(backend.device)
    module.set_alpha_base(args.alpha_base)
    module.set_base_trainable(args.base_trainable)
    module.requires_grad_(True).train()
    groups=[dict(params=list(module.Wv.parameters()),lr=1e-4),dict(params=[module.b],lr=.005),
            dict(params=list(module.Wq.parameters())+list(module.Wk.parameters()),lr=1e-5)]
    if args.base_trainable: groups.append(dict(params=[module.base_keys,module.base_values],lr=1e-3))
    optimizer=torch.optim.Adam(groups)
    sampler=BankSampler(packet['rows'],args.seed,args.batch_size)
    generator=torch.Generator(device=backend.device).manual_seed(args.seed+5000)
    totals={}; target_ids=set(); bank_ids=set(); schedule=hashlib.sha256(); started=time.perf_counter()
    touched_keys=torch.zeros(module.base_size,dtype=torch.bool); touched_values=touched_keys.clone()
    route1=torch.zeros(module.base_size,dtype=torch.long); routek=route1.clone()
    for step in range(1,args.updates+1):
        routing='dense' if args.routing=='dense_warm' and step<=args.updates//2 else ('sparse' if args.routing=='dense_warm' else args.routing)
        module.set_routing_mode(routing)
        sample=sampler.next(); target_ids.update(sample['targets']); bank_ids.update(sample['episode'])
        batch=make_batch(packet,sample,packet['pool'],.05)
        if getattr(args,'teacher_strategy','baseline') != 'baseline':
            from .coldstart_teacher import make_teacher_context
            records=[packet['rows'][i] for i in sample['targets']]
            batch['contexts']=[[make_teacher_context(r,c,a,args.teacher_strategy)
                                for r,c,a in zip(records,contexts,answers)]
                               for contexts,answers in zip(batch['contexts'],batch['answers'])]
        optimizer.zero_grad(set_to_none=True)
        loss,metrics,_=interface_step(backend,module,batch,'base',20.,generator)
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite loss')
        loss.backward()
        gradients={}
        for name,touched in (('base_keys',touched_keys),('base_values',touched_values)):
            grad=getattr(module,name).grad
            mask=torch.zeros(module.base_size,dtype=torch.bool) if grad is None else (grad.detach().norm(dim=1)>0).cpu()
            touched |= mask; gradients[name+'_gradient_slots']=int(mask.sum())
        info=module.last_base_retrieval
        if info is not None and 'retained_dense_mass' in info:
            mass=info['retained_dense_mass'].detach().float()
            gradients['base_retained_dense_mass_mean']=float(mass.mean())
            gradients['base_retained_dense_mass_min']=float(mass.min())
        if info is not None and 'topk_indices' in info:
            ix=info['topk_indices'].detach().cpu().reshape(-1,min(module.base_top_k,module.base_size))
            route1+=torch.bincount(ix[:,0],minlength=module.base_size)
            routek+=torch.bincount(ix.flatten(),minlength=module.base_size)
        norms=[float(torch.nn.utils.clip_grad_norm_(g['params'],1.)) for g in groups]
        if not all(torch.isfinite(torch.tensor(norms))): raise FloatingPointError('Nonfinite gradient')
        optimizer.step(); clear_bank(backend)
        schedule.update(json.dumps(sample,sort_keys=True).encode())
        for k,v in metrics.items():
            if k.endswith('tokens') or k.endswith('forward_calls'): totals[k]=totals.get(k,0)+v
        row=dict(step=step,routing=routing,sample=sample,elapsed_seconds=time.perf_counter()-started,
                 gradient_norms=norms,**gradients,**metrics)
        append_jsonl(args.run_dir/'training.jsonl',[row])
        if step==1 or step%64==0: print(json.dumps({k:v for k,v in row.items() if k!='sample'}),flush=True)
        if step%128==0 or step==args.updates:
            save_module(module,args.run_dir/'last.pt.partial',optimizer=optimizer.state_dict(),step=step,seed=args.seed)
            (args.run_dir/'last.pt.partial').replace(args.run_dir/'last.pt')
            result=dict(complete=step==args.updates,updates=step,totals=totals,schedule_sha256=schedule.hexdigest(),
                        target_unique_count=len(target_ids),bank_unique_count=len(bank_ids),
                        key_gradient_coverage=int(touched_keys.sum()),value_gradient_coverage=int(touched_values.sum()),
                        sampled_last_forward_top1_coverage=int((route1>0).sum()),sampled_last_forward_topk_coverage=int((routek>0).sum()),
                        usage_scope='last student forward per update, includes padded positions; diagnostic only',
                        elapsed_seconds=time.perf_counter()-started,trainable_parameters=sum(p.numel() for p in module.parameters() if p.requires_grad))
            json_write(args.run_dir/'training_status.json',result)
    torch.save(dict(top1=route1,topk=routek,key_gradient=touched_keys,value_gradient=touched_values),args.run_dir/'usage.pt')
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['prepare','init','train','eval'],required=True)
    p.add_argument('--model'); p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--data-dir',type=Path); p.add_argument('--domain',default='wikipedia')
    p.add_argument('--splits',default='train,dev,confirm'); p.add_argument('--cache',type=Path)
    p.add_argument('--checkpoint',type=Path); p.add_argument('--arm',default='unspecified')
    p.add_argument('--cluster-init',action='store_true'); p.add_argument('--seed',type=int,default=63042)
    p.add_argument('--updates',type=int,default=512); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--routing',choices=['sparse','dense_warm','straight_through'],default='sparse')
    p.add_argument('--teacher-strategy',choices=['baseline','quote_instruction','gold_annotated'],default='baseline')
    p.add_argument('--alpha-base',type=float,default=0.); p.add_argument('--base-trainable',action='store_true')
    p.add_argument('--base-mode',default='full'); p.add_argument('--merge-count',type=int)
    p.add_argument('--teacher',action='store_true'); p.add_argument('--max-cases',type=int,default=64)
    args=p.parse_args(); args.run_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(args.seed); random.seed(args.seed)
    manifest=dict(protocol=PROTOCOL,stage=args.stage,complete=False,started_at=datetime.now(timezone.utc).isoformat(),
                  configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  sources={f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')})
    json_write(args.run_dir/'manifest.json',manifest)
    try:
        packet=None
        if args.cache:
            packet=torch.load(args.cache,map_location='cpu',weights_only=True)
            if packet['protocol']!=PROTOCOL or packet['model_revision']!=REVISION: raise ValueError('Feature protocol mismatch')
            manifest['cache_sha256']=sha256(args.cache)
        if args.checkpoint: manifest['checkpoint_sha256']=sha256(args.checkpoint)
        if args.stage=='init': result=initialize(packet,args)
        else:
            if json.loads((Path(args.model).parent/'manifest.json').read_text())['revision']!=REVISION: raise ValueError('Model revision')
            backend=CounterfactualBackend(args.model,layer=20); before=parameter_digest(backend.model)
            if args.stage=='prepare': result=prepare(backend,args)
            else:
                module,_=load_module(args.checkpoint,backend.device)
                manifest['loaded_architecture']=module.configuration()
                if args.stage=='train': result=train(backend,module,packet,args)
                else:
                    from .coldstart_eval import evaluate_coldstart
                    result=evaluate_coldstart(backend,module,packet,args.run_dir,include_teacher=args.teacher,
                                             max_cases=args.max_cases,base_mode=args.base_mode,merge_count=args.merge_count)
                manifest['effective_architecture']=module.configuration()
            if before!=parameter_digest(backend.model) or any(p.grad is not None for p in backend.model.parameters()): raise AssertionError('Backbone mutated')
            manifest['backbone_unchanged']=True
        manifest.update(complete=True,result=result,finished_at=datetime.now(timezone.utc).isoformat())
        json_write(args.run_dir/'manifest.json',manifest)
    except BaseException as error:
        manifest['error']=repr(error); json_write(args.run_dir/'manifest.json',manifest); raise


if __name__=='__main__': main()
