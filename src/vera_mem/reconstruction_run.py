"""Small-data content reconstruction before scaling writable VeRA memory.

Training loads only the canonical A/B projection. Frozen-writer feature caches
and all online C/D/format interventions live in separate evaluation packets.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
import torch

from .context_distillation_run import REVISION, clear_bank, parameter_digest, sha256, append_jsonl, gold_tokens, supervised_loss, sequence_mean
from .counterfactual_backend import CounterfactualBackend
from .reconstruction_vera import ReconstructionVeRA, pool_content_slots
from .reconstruction_data import datasets, training_records, evaluation_records, payload_span, digest, WORLDS, WRITER_PREFIX
from .run import json_write, tensor_digest

PROTOCOL='reconstruction-experiment-v1'


def payload_encoding(tokenizer,support,answer):
    """Locate the observed payload, never synthesize unseen answer tokens."""
    p0,p1=payload_span(support,answer)
    messages=[{'role':'system','content':'You are a helpful assistant. Follow the requested answer format.'},
              {'role':'user','content':WRITER_PREFIX+support}]
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    begin=rendered.index(WRITER_PREFIX)+len(WRITER_PREFIX)
    if rendered[begin:begin+len(support)]!=support:raise AssertionError('Support offset mismatch')
    spans=[(begin+p0+m.start(),begin+p0+m.end()) for m in re.finditer(r'\S+',answer)]
    enc=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    masks=[]
    for lo,hi in spans:
        mask=[]
        for a,b in enc['offset_mapping']:
            inside=b>a and b>lo and a<hi
            if inside and (rendered[a:min(b,lo)].strip() or rendered[max(a,hi):b].strip()):
                raise ValueError('Tokenizer token crosses payload word boundary')
            mask.append(int(inside))
        if not any(mask):raise ValueError('Empty word span')
        masks.append(mask)
    if any(sum(v)>1 for v in zip(*masks)):raise ValueError('Overlapping word spans')
    return enc['input_ids'],masks


@torch.no_grad()
def payload_features(backend,supports,answers,batch_size=16):
    ones,threes,token_counts=[],[],[]
    for start in range(0,len(supports),batch_size):
        ss,aa=supports[start:start+batch_size],answers[start:start+batch_size]
        enc=[payload_encoding(backend.tokenizer,s,a) for s,a in zip(ss,aa)]
        sequences=[e[0] for e in enc]
        if any(seq!=backend.prompt_ids(WRITER_PREFIX+s) for seq,s in zip(sequences,ss)):
            raise AssertionError('Writer prompt mismatch')
        ids,attention=backend._right_pad(sequences)
        masks=torch.zeros((len(ss),3,ids.shape[1]),dtype=torch.bool,device=backend.device)
        for i,(_,m) in enumerate(enc):masks[i,:,:len(m[0])]=torch.tensor(m,dtype=torch.bool,device=backend.device)
        captured=[]
        def hook(_module,args):
            x=args[0].detach().float(); content=masks.any(1)
            captured.append((pool_content_slots(x,content,slots=1).cpu(),
                             pool_content_slots(x,content,slots=3,slot_masks=masks).cpu()))
        handle=backend.target.register_forward_pre_hook(hook)
        try:
            with backend.disabled():backend.model.model(input_ids=ids,attention_mask=attention,use_cache=False)
        finally:handle.remove()
        if len(captured)!=1:raise AssertionError('Wrong writer hook count')
        ones.append(captured[0][0]);threes.append(captured[0][1]);token_counts.extend(masks.sum(-1).cpu().tolist())
    return torch.cat(ones),torch.cat(threes),token_counts


def prepare(backend,args):
    raw=datasets(args.train_size,args.dev_size,args.confirm_size,seed=args.data_seed)
    packets={'train':training_records(raw['train']),'known':evaluation_records(raw['train']),
             'dev':evaluation_records(raw['dev']),'confirm':evaluation_records(raw['confirm'])}
    result={}
    for split,rows in packets.items():
        views=len(rows[0]['questions']);worlds=['A','B'] if split=='train' else list(WORLDS)
        fields={'A':'a','B':'b','C':'c','D':'d','SWAP':'swap_a'}
        for r in rows:
            for world in worlds:
                tokens=backend.tokenizer.encode(r[fields[world]],add_special_tokens=False)
                if not tokens or len(tokens)>=32:raise ValueError('Target does not fit generation budget')
        supports=[s for r in rows for ss in r['supports'] for s in ss]
        answers=[r[fields[w]] for r in rows for w in worlds for _ in range(views)]
        one,three,counts=payload_features(backend,supports,answers)
        q=backend.layer_features([q for r in rows for q in r['questions']])
        packet=dict(protocol=PROTOCOL,split=split,model_revision=REVISION,data_seed=args.data_seed,
                    rows=rows,worlds=worlds,views=views,rows_sha256=digest(rows),
                    q=q.view(len(rows),views,-1),
                    slots1=one.view(len(rows),len(worlds),views,1,-1),
                    slots3=three.view(len(rows),len(worlds),views,3,-1),
                    payload_token_counts=counts,writer_prefix=WRITER_PREFIX,
                    feature_scope='frozen down_proj input, observed payload word spans only; entity precedes payload')
        torch.save(packet,args.run_dir/(split+'.pt'))
        json_write(args.run_dir/(split+'.json'),rows)
        result[split]=dict(records=len(rows),worlds=worlds,views=views,rows_sha256=digest(rows),cache_sha256=sha256(args.run_dir/(split+'.pt')))
        print(json.dumps(dict(prepared=split,**result[split])),flush=True)
    return result


def save_module(module,path,**extra):
    torch.save(dict(protocol=PROTOCOL,architecture=module.configuration(),module=module.state_dict(),**extra),path)


def load_module(path,device):
    c=torch.load(path,map_location='cpu',weights_only=True)
    if c['protocol']!=PROTOCOL:raise ValueError('Wrong checkpoint protocol')
    m=ReconstructionVeRA(**c['architecture']);m.load_state_dict(c['module'])
    return m.to(device),c


def make_features(packet,slots,device):
    return packet['slots'+str(slots)].to(device=device,dtype=torch.float32)


def initialize(packet,args,device):
    if packet['split']!='train' or packet['worlds']!=['A','B'] or packet['views']!=1:
        raise ValueError('Only canonical A/B training projection may fit model/statistics')
    from .reconstruction_data import validate_training_records
    validate_training_records(packet['rows'])
    if (packet.get('protocol')!=PROTOCOL or packet.get('model_revision')!=REVISION
            or packet.get('writer_prefix')!=WRITER_PREFIX
            or packet.get('rows_sha256')!=digest(packet['rows'])):
        raise ValueError('Training cache protocol/model/prefix/row hash mismatch')
    count=len(packet['rows']);width=9728
    expected={'q':(count,1,width),'slots1':(count,2,1,1,width),'slots3':(count,2,1,3,width)}
    for name,shape in expected.items():
        feature=packet.get(name)
        if (not isinstance(feature,torch.Tensor) or tuple(feature.shape)!=shape
                or not feature.is_floating_point() or feature.requires_grad
                or not bool(torch.isfinite(feature).all())):
            raise ValueError('Invalid detached canonical A/B cache feature: '+name)
    counts=packet.get('payload_token_counts')
    if (not isinstance(counts,list) or len(counts)!=count*2
            or any(not isinstance(group,list) or len(group)!=3
                   or any(type(n) is not int or n<=0 for n in group) for group in counts)):
        raise ValueError('Training cache payload word-token counts mismatch')
    m=ReconstructionVeRA(9728,2560,rank=64,key_dim=64,top_k=4,temperature=.05,seed=args.seed,
                          slots=args.slots,train_B=args.learned_b).to(device)
    features=make_features(packet,args.slots,device).flatten(0,3)
    m.fit_statistics(packet['q'][:,0].to(device),features)
    m.fit_value_statistics(features)
    return m


class ExhaustiveSampler:
    def __init__(self,size,batch_size,seed):
        if size%batch_size or not 1<=batch_size<=size:raise ValueError('Require exhaustive whole batches')
        self.size,self.batch_size,self.rng,self.queue=size,batch_size,random.Random(seed),[]
    def next(self):
        if not self.queue:self.queue=list(range(self.size));self.rng.shuffle(self.queue)
        result=self.queue[:self.batch_size];del self.queue[:self.batch_size];return result


def independent_features(features,targets,world):
    """Only each batch row's target fact changes from the clean A bank."""
    base=features[:,0,0]
    batch=base.unsqueeze(0).expand(len(targets),*base.shape).clone()
    if world:
        rr=torch.arange(len(targets),device=features.device)
        tt=torch.tensor(targets,device=features.device)
        batch[rr,tt]=features[tt,world,0]
    return batch


def train(backend,packet,args):
    m=initialize(packet,args,backend.device)
    m.train().requires_grad_(True)
    save_module(m,args.run_dir/'initial.pt',step=0,seed=args.seed,cache_sha256=sha256(args.cache))
    groups=[dict(params=list(m.Wq.parameters())+list(m.Wk.parameters()),lr=1e-4),
            dict(params=list(m.Wv.parameters()),lr=3e-4),dict(params=[m.b],lr=.005)]
    if m.train_B:groups.append(dict(params=[m.B],lr=3e-4))
    optimizer=torch.optim.Adam(groups)
    features=make_features(packet,m.slots,backend.device)
    sampler=ExhaustiveSampler(len(packet['rows']),args.batch_size,args.seed)
    schedule=hashlib.sha256();totals=dict(target_exposures=0,gold_tokens=0,input_positions=0,padded_input_positions=0,backbone_calls=0)
    start=time.perf_counter()
    for step in range(1,args.updates+1):
        targets=sampler.next();questions=[packet['rows'][i]['questions'][0] for i in targets]
        optimizer.zero_grad(set_to_none=True);metrics=[]
        # A/B share one batched backbone call, with independent per-example
        # banks; losses remain equal means over worlds and target sequences.
        bank=torch.cat([independent_features(features,targets,w) for w in (0,1)],dim=0)
        keys,values=m.encode_bank(bank)
        clear_bank(backend);backend.mode='vector_vera';backend.vector_vera=m
        backend.vector_keys,backend.vector_values=keys,values
        answers=[packet['rows'][i][label] for label in ('a','b') for i in targets]
        tokens=gold_tokens(backend,answers);batched_questions=questions+questions
        branch=backend.forward_sequences(batched_questions,tokens)
        ce,ce_tokens=supervised_loss(branch,tokens)
        tt=torch.tensor(targets+targets,device=backend.device)[branch['batch_indices']]
        address_tokens=m.group_address_loss(branch['query_inputs'],tt,keys,batch_indices=branch['batch_indices'],reduction='none')
        address=sequence_mean(address_tokens,branch['batch_indices'],branch['counts'])
        loss=ce+.2*address
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        for w in (0,1):
            selected=(branch['batch_indices']>=w*len(targets))&(branch['batch_indices']<(w+1)*len(targets))
            rows=branch['batch_indices'][selected]-w*len(targets);counts=branch['counts'][w*len(targets):(w+1)*len(targets)]
            wc=sequence_mean(ce_tokens.detach()[selected],rows,counts)
            wa=sequence_mean(address_tokens.detach()[selected],rows,counts)
            metrics.append(dict(ce=float(wc),address=float(wa),loss=float(wc+.2*wa)))
        lengths=[len(backend.prompt_ids(q))+len(t) for q,t in zip(batched_questions,tokens)]
        totals['gold_tokens']+=sum(map(len,tokens));totals['input_positions']+=sum(lengths)
        totals['padded_input_positions']+=len(lengths)*max(lengths);totals['backbone_calls']+=1
        del branch,keys,values,bank,loss,ce,address,ce_tokens,address_tokens
        norms=[float(torch.nn.utils.clip_grad_norm_(g['params'],1.)) for g in groups]
        if not all(torch.isfinite(torch.tensor(norms))):raise FloatingPointError('Nonfinite gradients')
        optimizer.step();clear_bank(backend)
        totals['target_exposures']+=len(targets)
        schedule.update(json.dumps(targets).encode())
        row=dict(step=step,targets=targets,metrics=metrics,gradient_norms=norms,elapsed_seconds=time.perf_counter()-start)
        append_jsonl(args.run_dir/'training.jsonl',[row])
        if step==1 or step%64==0:print(json.dumps(row),flush=True)
        if step in (512,args.updates):
            save_module(m,args.run_dir/f'step_{step}.pt',step=step,seed=args.seed,optimizer=optimizer.state_dict(),cache_sha256=sha256(args.cache),schedule_sha256=schedule.hexdigest())
        if step%128==0 or step==args.updates:
            result=dict(complete=step==args.updates,updates=step,totals=totals,schedule_sha256=schedule.hexdigest(),
                        elapsed_seconds=time.perf_counter()-start,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
                        architecture=m.configuration(),learning_rates=[g['lr'] for g in groups])
            json_write(args.run_dir/'training_status.json',result)
    save_module(m,args.run_dir/'last.pt',step=args.updates,seed=args.seed,cache_sha256=sha256(args.cache),schedule_sha256=schedule.hexdigest())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','teacher','train','eval'),required=True)
    p.add_argument('--model',required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--seed',type=int,default=71042);p.add_argument('--data-seed',type=int,default=101042)
    p.add_argument('--slots',type=int,choices=(1,3),default=1);p.add_argument('--learned-b',action='store_true')
    p.add_argument('--updates',type=int,default=1024);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--train-size',type=int,default=16);p.add_argument('--dev-size',type=int,default=32);p.add_argument('--confirm-size',type=int,default=64)
    p.add_argument('--eval-part',choices=('development','confirmation','smoke','preflight'),default='development')
    args=p.parse_args();args.run_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed)
    manifest=dict(protocol=PROTOCOL,complete=False,stage=args.stage,started_at=datetime.now(timezone.utc).isoformat(),
                  configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  source_sha256={f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')})
    if args.cache:manifest['cache_sha256']=sha256(args.cache)
    if args.checkpoint:manifest['checkpoint_sha256']=sha256(args.checkpoint)
    json_write(args.run_dir/'manifest.json',manifest)
    try:
        model_manifest=json.loads((Path(args.model).parent/'manifest.json').read_text())
        if model_manifest['revision']!=REVISION:raise ValueError('Model revision mismatch')
        backend=CounterfactualBackend(args.model,layer=20);before=parameter_digest(backend.model)
        packet=torch.load(args.cache,map_location='cpu',weights_only=True) if args.cache else None
        if packet and (packet['protocol']!=PROTOCOL or packet['model_revision']!=REVISION):raise ValueError('Cache protocol mismatch')
        if args.stage=='prepare':result=prepare(backend,args)
        elif args.stage=='train':result=train(backend,packet,args)
        else:
            from .reconstruction_eval import evaluate,teacher
            if args.stage=='teacher':result=teacher(backend,packet,args)
            else:
                module,c=load_module(args.checkpoint,backend.device)
                result=evaluate(backend,module,packet,args)
        if before!=parameter_digest(backend.model) or any(p.grad is not None for p in backend.model.parameters()):raise AssertionError('Frozen backbone changed')
        manifest.update(complete=True,backbone_unchanged=True,result=result,finished_at=datetime.now(timezone.utc).isoformat())
        json_write(args.run_dir/'manifest.json',manifest)
    except BaseException as error:
        manifest['error']=repr(error);json_write(args.run_dir/'manifest.json',manifest);raise

if __name__=='__main__':main()
