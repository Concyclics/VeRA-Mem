"""Controlled episodic block training with actual contextual entity/payload caches."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
import torch

from .context_distillation_run import (REVISION, clear_bank, parameter_digest, sha256,
    append_jsonl, gold_tokens, supervised_loss, sequence_mean)
from .block_backend import BlockBackend
from .reconstruction_run import payload_features
from .reconstruction_data import WRITER_PREFIX, render_support
from .block_data import dataset, episode, digest, validate_dataset, validate_training_data
from .block_data import PROTOCOL as DATA_PROTOCOL
from .block_vera import BlockVeRA
from .run import json_write

PROTOCOL = 'block-attention-memory-v1'


@torch.no_grad()
def prepare(backend, args):
    data = dataset(seed=args.data_seed)
    validate_dataset(data)
    train = data['train']; entities, payloads = train['entities'], train['payloads']
    supports = [render_support(e, p) for e in entities for p in payloads]
    answers = payloads * len(entities)
    _, slots, counts = payload_features(backend, supports, answers, batch_size=32)
    q = backend.layer_features([r['questions'][0] for r in train['rows']])
    packet = dict(protocol=PROTOCOL, split='train', model_revision=REVISION,
        data_seed=args.data_seed, rows=train['rows'], entities=entities, payloads=payloads,
        worlds=['A','B'], views=1, q=q, matrix=slots.reshape(len(entities), len(payloads), 3, -1),
        rows_sha256=digest(train['rows']), payload_token_counts=counts, writer_prefix=WRITER_PREFIX,
        feature_scope='All actual canonical entity+payload strings, not reassigned old hidden vectors',
        statistics_scope='Only 64 static A/B pairs (128 observations), q64; no full-crossproduct statistics')
    torch.save(packet, args.run_dir/'train.pt')
    json_write(args.run_dir/'train.json', train)
    result = {'train': dict(entities=len(entities), payloads=len(payloads),
        contextual_observations=len(supports), cache_sha256=sha256(args.run_dir/'train.pt'))}
    for split in ('known', 'dev', 'confirm'):
        rows = data[split]
        ss = [s for r in rows for ws in r['supports'] for s in ws]
        aa = [r[w] for r in rows for w in ('a','b','c','d','swap_a') for _ in range(2)]
        _, f, tc = payload_features(backend, ss, aa, batch_size=32)
        qq = backend.layer_features([x for r in rows for x in r['questions']])
        packet = dict(protocol=PROTOCOL, split=split, model_revision=REVISION, data_seed=args.data_seed,
            rows=rows, worlds=['A','B','C','D','SWAP'], views=2,
            q=qq.reshape(len(rows),2,-1), slots3=f.reshape(len(rows),5,2,3,-1),
            rows_sha256=digest(rows), payload_token_counts=tc, writer_prefix=WRITER_PREFIX)
        torch.save(packet,args.run_dir/(split+'.pt')); json_write(args.run_dir/(split+'.json'),rows)
        result[split]=dict(records=len(rows),cache_sha256=sha256(args.run_dir/(split+'.pt')))
    json_write(args.run_dir/'dataset.json', data)
    return dict(caches=result, dataset_sha256=sha256(args.run_dir/'dataset.json'))


def save_module(module, path, **extra):
    torch.save(dict(protocol=PROTOCOL, architecture=module.configuration(), module=module.state_dict(), **extra),path)


def load_module(path, device):
    c=torch.load(path,map_location='cpu',weights_only=True)
    if c['protocol']!=PROTOCOL: raise ValueError('Checkpoint protocol mismatch')
    m=BlockVeRA(**c['architecture']);m.load_state_dict(c['module'])
    return m.to(device),c


def initialize(packet,args,device):
    if (packet['protocol']!=PROTOCOL or packet['split']!='train' or packet['worlds']!=['A','B']
            or packet['views']!=1 or packet['model_revision']!=REVISION
            or packet['rows_sha256']!=digest(packet['rows'])):
        raise ValueError('Require canonical training-only packet')
    validate_training_data(dict(protocol=DATA_PROTOCOL,seed=packet['data_seed'],
        entities=packet['entities'],payloads=packet['payloads'],rows=packet['rows']))
    if packet.get('writer_prefix')!=WRITER_PREFIX:raise ValueError('Writer prefix mismatch')
    if packet['matrix'].shape!=(64,128,3,9728) or packet['q'].shape!=(64,9728):
        raise ValueError('Unexpected training feature dimensions')
    if any(not t.is_floating_point() or t.requires_grad or not bool(torch.isfinite(t).all())
            for t in (packet['matrix'],packet['q'])):raise ValueError('Require finite detached floating cache')
    counts=packet.get('payload_token_counts')
    if (not isinstance(counts,list) or len(counts)!=64*128
            or any(not isinstance(c,list) or len(c)!=3 or any(type(x)is not int or x<=0 for x in c) for c in counts)):
        raise ValueError('Invalid cached word token counts')
    m=BlockVeRA(9728,2560,rank=64,key_dim=64,top_k=4,temperature=.05,seed=args.seed,
        slots=3,train_B=True,read_mode="grouped",block_mode=args.block_mode).to(device)
    i=torch.arange(64);f=packet['matrix'][i[:,None],2*i[:,None]+torch.arange(2)].reshape(-1,9728).to(device)
    m.fit_statistics(packet['q'].to(device),f);m.fit_value_statistics(f)
    return m


def independent_banks(matrix, ep):
    """Each B branch replaces its own target, never a shared batch bank."""
    device=matrix.device
    ids=torch.tensor(ep['bank_entities'],device=device)
    aa=torch.tensor(ep['a_payload_indices'],device=device)
    bb=torch.tensor(ep['b_payload_indices'],device=device)
    targets=torch.tensor(ep['local_targets'],device=device)
    base=matrix[ids,aa]
    a=base.unsqueeze(0).expand(len(targets),*base.shape).clone(); b=a.clone()
    rr=torch.arange(len(targets),device=device)
    b[rr,targets]=matrix[ids[targets],bb[targets]]
    return torch.cat((a,b),0)


def target_answers(packet,ep):
    return [packet['payloads'][ep[key][j]] for key in ('a_payload_indices','b_payload_indices') for j in ep['local_targets']]


def train(backend,packet,args):
    m=initialize(packet,args,backend.device);m.train().requires_grad_(True)
    common=dict(seed=args.seed,regime=args.regime,block_mode=args.block_mode,cache_sha256=sha256(args.cache))
    save_module(m,args.run_dir/'initial.pt',step=0,**common)
    groups=[dict(params=[*m.Wq.parameters(),*m.Wk.parameters(),m.slot_position],lr=1e-4),
            dict(params=list(m.Wv.parameters()),lr=3e-4),dict(params=[m.b],lr=.005),dict(params=[m.B],lr=3e-4)]
    if args.block_mode != "diagonal":
        groups.append(dict(params=[*m.P_in.parameters(),*m.P_out.parameters()],lr=3e-4))
    optimizer=torch.optim.Adam(groups)
    matrix=packet['matrix'].to(backend.device,dtype=torch.float32)
    schedule=hashlib.sha256(); totals=dict(target_exposures=0,gold_tokens=0,input_positions=0,padded_input_positions=0,backbone_calls=0)
    start=time.perf_counter()
    for step in range(args.updates):
        ep=episode(step,args.regime,seed=args.seed)
        bank=independent_banks(matrix,ep); keys,values=m.encode_bank(bank)
        questions=[packet['rows'][i]['questions'][0] for i in ep['targets']]*2
        answers=target_answers(packet,ep);tokens=gold_tokens(backend,answers)
        optimizer.zero_grad(set_to_none=True);clear_bank(backend)
        backend.mode='vector_vera';backend.vector_vera=m;backend.vector_keys,backend.vector_values=keys,values
        branch=backend.forward_sequences(questions,tokens)
        ce,ce_tokens=supervised_loss(branch,tokens)
        tt=torch.tensor(ep['local_targets']*2,device=backend.device)[branch['batch_indices']]
        address_tokens=m.group_address_loss(branch['query_inputs'],tt,keys,batch_indices=branch['batch_indices'],reduction='none')
        address=sequence_mean(address_tokens,branch['batch_indices'],branch['counts']);loss=ce+.2*address
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in m.named_parameters()}
        norms=[float(torch.nn.utils.clip_grad_norm_(g['params'],1.)) for g in groups]
        if not all(torch.isfinite(torch.tensor(norms))):raise FloatingPointError('Nonfinite gradients')
        metrics=dict(ce=float(ce.detach()),address=float(address.detach()),loss=float(loss.detach()))
        read_diagnostics=None
        if step==0 or (step+1)%128==0:
            # Replay only the actual captured gold prediction positions, before
            # updating parameters. No extra backbone call or loss is introduced.
            with torch.no_grad():
                bi=branch['batch_indices']; xx=branch['query_inputs'].detach().unsqueeze(1)
                kk,vv=keys.detach()[bi],values.detach()[bi]
                delta,info=m(xx,kk,vv,return_info=True)
                ii,ww=info['indices'],info['weights']
                selected=vv[torch.arange(len(bi),device=vv.device)[:,None,None],ii]
                mixed=(selected*ww.unsqueeze(-1)).sum(-2)
                mass=(ww*(ii//3==tt[:,None,None])).sum(-1)
                block=selected
                with torch.no_grad():
                    singular=None
                    if args.block_mode!='diagonal':
                        factored=block.mean(-2,keepdim=True) if args.block_mode=='pooled_outer' else block
                        u,w=m.block_factors(factored)
                        operator=torch.einsum('...sr,...sc->...rc',w,u)/factored.shape[-2]
                        singular=torch.linalg.svdvals(operator.float()).mean(dim=tuple(range(operator.ndim-2))).cpu().tolist()
                read_diagnostics=dict(outer_operator_mean_singular_values=singular,scope='captured gold-prefix prediction positions including EOS; parameter state before update',
                    prediction_positions=len(bi),input_rms=float(xx.square().mean().sqrt()),
                    mixed_value_rms=float(mixed.square().mean().sqrt()),residual_rms=float(delta.float().square().mean().sqrt()),
                    target_fact_mass=float(mass.mean()),actual_read_slots=ii.shape[-1])
        optimizer.step();clear_bank(backend)
        lengths=[len(backend.prompt_ids(q))+len(t) for q,t in zip(questions,tokens)]
        totals['target_exposures']+=8;totals['gold_tokens']+=sum(map(len,tokens))
        totals['input_positions']+=sum(lengths);totals['padded_input_positions']+=len(lengths)*max(lengths);totals['backbone_calls']+=1
        schedule.update(json.dumps(ep,sort_keys=True).encode())
        row=dict(step=step+1,episode=ep,metrics=metrics,gradient_norms=norms,
                 input_lengths=lengths,gold_token_lengths=list(map(len,tokens)),
                 read_diagnostics=read_diagnostics,
                 parameter_gradient_norms=grads,elapsed_seconds=time.perf_counter()-start)
        append_jsonl(args.run_dir/'training.jsonl',[row])
        if step==0 or (step+1)%128==0:print(json.dumps({k:v for k,v in row.items() if k!='episode'}),flush=True)
        del branch,keys,values,bank,loss,ce,address,ce_tokens,address_tokens
        if step+1 in (1024,args.updates):
            save_module(m,args.run_dir/f'step_{step+1}.pt',step=step+1,optimizer=optimizer.state_dict(),schedule_sha256=schedule.hexdigest(),**common)
        if (step+1)%128==0 or step+1==args.updates:
            result=dict(complete=step+1==args.updates,updates=step+1,totals=totals,schedule_sha256=schedule.hexdigest(),
                architecture=m.configuration(),trainable_parameters=sum(p.numel() for p in m.parameters()),
                learning_rates=[g['lr'] for g in groups],elapsed_seconds=time.perf_counter()-start,regime=args.regime)
            json_write(args.run_dir/'training_status.json',result)
    save_module(m,args.run_dir/'last.pt',step=args.updates,schedule_sha256=schedule.hexdigest(),**common)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','teacher','train','eval'),required=True)
    p.add_argument('--model',required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--seed',type=int,default=91042);p.add_argument('--data-seed',type=int,default=221042)
    p.add_argument('--regime',choices=('static','rebind'),default='static')
    p.add_argument('--block-mode',choices=('diagonal','pooled_outer','block_outer'),default='diagonal')
    p.add_argument('--updates',type=int,default=2048)
    p.add_argument('--eval-part',choices=('development','confirmation','preflight','smoke'),default='development')
    args=p.parse_args();args.run_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed)
    manifest=dict(protocol=PROTOCOL,complete=False,stage=args.stage,started_at=datetime.now(timezone.utc).isoformat(),
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_sha256={f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')})
    if args.cache:manifest['cache_sha256']=sha256(args.cache)
    if args.checkpoint:manifest['checkpoint_sha256']=sha256(args.checkpoint)
    json_write(args.run_dir/'manifest.json',manifest)
    try:
        if json.loads((Path(args.model).parent/'manifest.json').read_text())['revision']!=REVISION:
            raise ValueError('Model revision mismatch')
        backend=BlockBackend(args.model,layer=20);before=parameter_digest(backend.model)
        packet=torch.load(args.cache,map_location='cpu',weights_only=True) if args.cache else None
        if packet and (packet['protocol']!=PROTOCOL or packet['model_revision']!=REVISION):raise ValueError('Cache protocol mismatch')
        if args.stage=='prepare':result=prepare(backend,args)
        elif args.stage=='train':result=train(backend,packet,args)
        else:
            from .block_eval import evaluate,teacher
            if args.stage=='teacher':result=teacher(backend,packet,args)
            else:
                module,_=load_module(args.checkpoint,backend.device);result=evaluate(backend,module,packet,args)
        if before!=parameter_digest(backend.model) or any(p.grad is not None for p in backend.model.parameters()):
            raise AssertionError('Frozen backbone changed')
        manifest.update(complete=True,backbone_unchanged=True,result=result,finished_at=datetime.now(timezone.utc).isoformat())
        json_write(args.run_dir/'manifest.json',manifest)
    except BaseException as error:
        manifest['error']=repr(error);json_write(args.run_dir/'manifest.json',manifest);raise


if __name__=='__main__':main()
