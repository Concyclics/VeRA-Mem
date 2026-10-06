"""Staged writer/interface experiments on frozen Qwen and sparse vector memory."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F

from .counterfactual_backend import CounterfactualBackend, BatchedStableVectorVeRA
from .context_distillation_run import REVISION, clear_bank, parameter_digest, sha256, append_jsonl, gold_tokens, supervised_loss, sequence_mean
from .context_distillation import distillation_loss
from .counterfactual_run import first_positions, independent_banks
from .interface_features import support_features
from .run import json_write, tensor_digest

PROTOCOL="memory-interface-v1"


def classic_rows(split,size):
    from . import counterfactual_data as d
    from .data import synthetic_dataset
    if split=="train":
        pairs=d.datasets()["train"][:size]
        qids=d.template_ids()["train"]["q"]; sids=d.template_ids()["train"]["s"]
    else:
        seed={"dev":48042,"confirm":58042}[split]
        pairs=d._make_pairs(synthetic_dataset(seed,size,0)["stream"],seed+1000)
        qids=["train_query_00"]+[t.id for t in d.get_templates("dev" if split=="dev" else "confirmation","query")]
        sids=["train_support_00"]+[t.id for t in d.get_templates("dev" if split=="dev" else "confirmation","support")]
    rows=[]
    for p in pairs:
        qs=[d.render_question(p,q) for q in qids]
        if split=="confirm":
            qs=[qs[0]]+[q+"\nReturn only the assigned memory word, without JSON, table, or explanation." for q in qs[1:]]
        rows.append(dict(id=p.id,entity=p.entity,relation="assigned memory word",a=p.a.answer,b=p.b.answer,
                         questions=qs,supports=[[d.render_support(p,s,world=w) for s in sids] for w in ("A","B")]))
    return rows,qids,sids


def prepare(backend,args):
    if args.task=="extended":
        from . import interface_data as d
        datasets=d.datasets(train_size=args.train_size,dev_size=args.eval_size,confirm_size=args.eval_size)
    packets={}
    for split,size in (("train",args.train_size),("dev",args.eval_size),("confirm",args.eval_size)):
        if args.task=="classic": rows,qids,sids=classic_rows(split,size)
        else:
            qids=[t.id for t in d.get_templates(split,"query")]
            sids=[t.id for t in d.get_templates(split,"support")]
            if split!="train":
                qids=[d.get_templates("train","query")[0].id]+qids
                sids=[d.get_templates("train","support")[0].id]+sids
            rows=[dict(id=p.id,entity=p.entity,relation=p.relation,a=p.answer_a,b=p.answer_b,
                       questions=[d.render_question(p,q) for q in qids],
                       supports=[[d.render_support(p,s,world=w) for s in sids] for w in ("A","B")]) for p in datasets[split]]
        q=backend.layer_features([t for r in rows for t in r["questions"]]).view(size,len(qids),-1)
        texts=[t for r in rows for world in r["supports"] for t in world]
        last,pool=support_features(backend,texts)
        for r in rows:
            aa=backend.tokenizer.encode(r["a"],add_special_tokens=False); bb=backend.tokenizer.encode(r["b"],add_special_tokens=False)
            if not aa or not bb or aa[0]==bb[0]: raise ValueError("Distinct first answer tokens required")
            if max(len(aa),len(bb))>=32: raise ValueError("Answer exceeds budget")
            r["a_tokens"],r["b_tokens"]=aa,bb
        token_audit = d.validate_tokenization(datasets[split],lambda text:backend.tokenizer.encode(text,add_special_tokens=False)) if args.task=="extended" else None
        packet=dict(protocol=PROTOCOL,task=args.task,split=split,model_revision=REVISION,rows=rows,qids=qids,sids=sids,
                    q=q,last=last.view(size,2,len(sids),-1),pool=pool.view(size,2,len(sids),-1),
                    writer_prefix="Remember this information: ",pooling="raw down_proj inputs; only full support-content tokens")
        packet.update(tokenization_audit=token_audit,data_protocol=d.PROTOCOL_VERSION if args.task=="extended" else "classic-new-splits-v1")
        torch.save(packet,args.run_dir/(split+".pt")); packets[split]=rows
        print(json.dumps(dict(prepared=split,facts=size,qviews=len(qids),sviews=len(sids))),flush=True)
    sets={s:{r["entity"] for r in rows} for s,rows in packets.items()}
    if any(sets[a]&sets[b] for a,b in (("train","dev"),("train","confirm"),("dev","confirm"))): raise AssertionError("Entity overlap")
    if args.task=="extended":
        answers={s:{r[k] for r in rows for k in ("a","b")} for s,rows in packets.items()}
        if any(answers[a]&answers[b] for a,b in (("train","dev"),("train","confirm"),("dev","confirm"))): raise AssertionError("Answer overlap")
    return dict(splits={s:len(rows) for s,rows in packets.items()},entity_disjoint=True)


def load_module(path,device,writer=None,learned_b=None,mlp=None):
    from .interface_variants import InterfaceVectorVeRA
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    if checkpoint.get("protocol")==PROTOCOL:
        for key,value in (("writer_mode",writer),("train_B",learned_b),("value_mlp_hidden",mlp)):
            if value is not None and checkpoint["architecture"][key]!=value:
                raise ValueError(f"Explicit {key} conflicts with saved architecture")
        module=InterfaceVectorVeRA(**checkpoint["architecture"])
        module.load_state_dict(checkpoint["module"])
    else:
        module=InterfaceVectorVeRA(9728,2560,rank=64,key_dim=64,top_k=4,temperature=.05,seed=42,
                                  writer_mode=writer or "last_token",train_B=bool(learned_b),value_mlp_hidden=mlp or 0)
        module.load_legacy_state_dict(checkpoint["module"])
    return module.to(device),checkpoint


def train_writer(backend,module,packet,args):
    from .interface_writer import train_writer as fit
    return fit(backend,module,packet,args)


def evaluate(backend,module,packet,args):
    from .interface_eval import evaluate_interface
    return evaluate_interface(backend,module,packet,args.run_dir,include_teacher=args.teacher,
                              max_new_tokens=32,max_cases=args.max_cases)


class InterfaceSampler:
    def __init__(self,rows,seed,batch_size=8,bank_size=72):
        self.rows,self.rng,self.batch_size,self.bank_size=rows,random.Random(seed),batch_size,bank_size
        self.by_answer={}; self.by_entity={}
        for i,r in enumerate(rows):
            self.by_answer.setdefault(r["a"],[]).append(i)
            self.by_entity.setdefault(r["entity"],[]).append(i)

    def next(self,nq,ns):
        targets=self.rng.sample(range(len(self.rows)),self.batch_size)
        chosen=set(targets)
        for i in targets:
            r=self.rows[i]
            chosen.update(self.by_entity[r["entity"]])
            for world in ("a","b"):
                options=[j for j in self.by_answer.get(r[world],[]) if self.rows[j]["entity"]!=r["entity"]]
                if not options: raise ValueError("Missing real-bank same-answer other-entity hard negative")
                chosen.add(self.rng.choice(options))
        if len(chosen)>self.bank_size: raise ValueError("Required hard negatives exceed bank budget")
        remaining=[j for j in range(len(self.rows)) if j not in chosen]
        chosen.update(self.rng.sample(remaining,min(self.bank_size-len(chosen),len(remaining))))
        episode=sorted(chosen); self.rng.shuffle(episode)
        return dict(targets=targets,episode=episode,columns=[episode.index(i) for i in targets],
                    qviews=[self.rng.randrange(nq) for _ in targets],sviews=[self.rng.randrange(ns) for _ in episode])


def make_batch(packet,sample,value_features,key_weight):
    targets,episode,columns=[sample[k] for k in ("targets","episode","columns")]
    last=packet["last"][episode]; vals=value_features[episode]
    dev=last.device; pos=torch.arange(len(episode),device=dev)
    sviews=sample["sviews"]; sv=[sviews[c] for c in columns]; pv=[(v+1)%last.shape[2] for v in sv]
    def banks(features):
        return independent_banks(features[pos,0,sviews],features[columns,1,sv],features[columns,0,pv],columns)
    records=[packet["rows"][i] for i in targets]
    questions=[r["questions"][v] for r,v in zip(records,sample["qviews"])]
    contexts=[[r["supports"][world][view] for r,view in zip(records,views)] for world,views in ((0,sv),(1,sv),(0,pv))]
    alternative=last[:,0].clone(); alternative[columns]=last[columns,1]
    return dict(questions=questions,contexts=contexts,answers=[[r[w] for r in records] for w in ("a","b","a")],
                key_banks=banks(last),value_banks=banks(vals),columns=columns,q_views=packet["q"][episode],
                key_views=[last[:,0],alternative],key_consistency_weight=key_weight)


def train_model(backend,module,packet,checkpoint,args):
    from .interface_training import interface_step
    if packet["split"]!="train": raise ValueError("Only train split may fit parameters")
    for key in ("q","last","pool"): packet[key]=packet[key].to(backend.device)
    features=packet["pool" if module.writer_mode=="masked_mean" else "last"]
    module.requires_grad_(True).train()
    if module.writer_mode=="masked_mean" and not module._value_statistics_ready:
        module.fit_value_statistics(features.flatten(0,2))
    groups=[dict(params=list(module.Wv.parameters())+(list(module.value_mlp.parameters()) if module.value_mlp is not None else []),lr=1e-4),
            dict(params=[module.b],lr=.005),dict(params=list(module.Wq.parameters())+list(module.Wk.parameters()),lr=1e-5)]
    if module.train_B: groups.append(dict(params=[module.B],lr=1e-4))
    optimizer=torch.optim.Adam(groups)
    if args.resume_optimizer:
        if "optimizer" not in checkpoint: raise ValueError("Missing common optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])
    sampler=InterfaceSampler(packet["rows"],args.seed)
    start_step=checkpoint.get("step",0) if args.resume_optimizer else 0
    for _ in range(start_step): sampler.next(packet["q"].shape[1],packet["last"].shape[2])
    generator=torch.Generator(device=backend.device).manual_seed(args.seed+5000)
    schedule=hashlib.sha256(); began=time.perf_counter(); totals={}
    for step in range(1,args.updates+1):
        sample=sampler.next(packet["q"].shape[1],packet["last"].shape[2])
        batch=make_batch(packet,sample,features,args.key_consistency)
        optimizer.zero_grad(set_to_none=True)
        loss,metrics,trajectories=interface_step(backend,module,batch,args.method,args.scale,generator,
                                               on_policy=args.method=="on_policy" and step%4==0)
        if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite loss")
        loss.backward()
        norms=[float(torch.nn.utils.clip_grad_norm_(g["params"],1.)) for g in groups]
        if not all(torch.isfinite(torch.tensor(norms))): raise FloatingPointError("Nonfinite gradient")
        optimizer.step(); clear_bank(backend)
        schedule.update(json.dumps(sample,sort_keys=True).encode())
        for key,value in metrics.items():
            if key.endswith("tokens") or key.endswith("forward_calls"): totals[key]=totals.get(key,0)+value
        row=dict(step=step+start_step,local_step=step,sample=sample,gradient_norms=norms,elapsed_seconds=time.perf_counter()-began,**metrics)
        append_jsonl(args.run_dir/"training.jsonl",[row])
        if trajectories: append_jsonl(args.run_dir/"trajectories.jsonl",[dict(step=step,**x) for x in trajectories])
        if step==1 or step%32==0: print(json.dumps({k:v for k,v in row.items() if k not in ("sample",)}),flush=True)
        if step%128==0 or step==args.updates:
            torch.save(dict(protocol=PROTOCOL,architecture=module.configuration(),module=module.state_dict(),optimizer=optimizer.state_dict(),
                            step=step+start_step,seed=args.seed,method=args.method,key_consistency=args.key_consistency),args.run_dir/"last.pt.partial")
            (args.run_dir/"last.pt.partial").replace(args.run_dir/"last.pt")
            status=dict(complete=step==args.updates,updates=step,start_step=start_step,
                        schedule_sha256=schedule.hexdigest(),totals=totals,elapsed_seconds=time.perf_counter()-began,
                        trainable_parameters=sum(p.numel() for p in module.parameters() if p.requires_grad))
            json_write(args.run_dir/"training_status.json",status)
    module.requires_grad_(False).eval()
    return status


@torch.no_grad()
def calibrate(backend,packet,args):
    from .interface_losses import fit_margin_scale
    from .counterfactual_losses import pair_behavior_loss
    if packet["split"]!="train": raise ValueError("Calibration only on training facts")
    deltas=[]; valid=[]; rows=packet["rows"][:256]; teacher_rows=[]
    for start in range(0,len(rows),8):
        chunk=rows[start:start+8]; qs=[r["questions"][0] for r in chunk]
        labels=[torch.tensor([r[k+"_tokens"][0] for r in chunk],device=backend.device) for k in ("a","b")]
        branches=[backend.forward_sequences(qs,[[int(x)] for x in labels[wi]],contexts=[r["supports"][wi][0] for r in chunk],teacher=True) for wi in (0,1)]
        loss=pair_behavior_loss(branches[0]["logits"],branches[1]["logits"],branches[0]["logits"],branches[1]["logits"],*labels,gold_weight=0.)
        deltas.append(loss["teacher_delta_raw"].cpu()); valid.append(loss["valid_mask"].cpu())
    result=fit_margin_scale(torch.cat(deltas),torch.cat(valid),split="train")
    result.update(calibration_ids=[r["id"] for r in rows],temperature=1.)
    # Teacher interface preflight uses train facts, with fixed prompts/budget.
    for i,r in enumerate(rows[:16]):
        for qi,q in enumerate(r["questions"]):
            for world in (0,1):
                clear_bank(backend)
                pred,tokens,seconds=backend.generate(q,context=r["supports"][world][i%len(r["supports"][world])],max_new_tokens=32)
                from .metrics import exact_match
                teacher_rows.append(dict(id=r["id"],query_view="train_"+str(qi),world=world,answer=r["a" if world==0 else "b"],prediction=pred,
                                         em=exact_match(pred,r["a" if world==0 else "b"]),tokens=tokens))
    if packet["task"]=="extended":
        from . import interface_data as data
        q0=data.get_templates("train","query")[0].id; s0=data.get_templates("train","support")[0].id
        qs=[t.id for t in data.get_templates("confirm","query")]
        ss=[t.id for t in data.get_templates("confirm","support")]
        for r in rows[:16]:
            fact=data.InterfaceFact(r["id"],r["entity"],r["relation"],r["a"],r["b"],"train")
            for family,(qh,sh) in enumerate(zip(qs,ss)):
                for condition,qid,sid in (("CH",qh,s0),("HC",q0,sh),("HH",qh,sh)):
                    for world in (0,1):
                        q=data.render_question(fact,qid)
                        context=data.render_support(fact,sid,world="A" if world==0 else "B")
                        clear_bank(backend); pred,tokens,_=backend.generate(q,context=context,max_new_tokens=32)
                        answer=r["a" if world==0 else "b"]
                        teacher_rows.append(dict(id=r["id"],query_view=f"interface_{family}_{condition}",world=world,
                                                 answer=answer,prediction=pred,em=exact_match(pred,answer),tokens=tokens))
    append_jsonl(args.run_dir/"teacher_preflight.jsonl",teacher_rows)
    result["teacher_preflight"]={str(q):dict(correct=sum(r["em"] for r in teacher_rows if r["query_view"]==q),n=sum(r["query_view"]==q for r in teacher_rows)) for q in sorted({r["query_view"] for r in teacher_rows})}
    result["teacher_interface_passed"]=all(x["correct"]/x["n"]>=.95 for x in result["teacher_preflight"].values())
    json_write(args.run_dir/"calibration.json",result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=("prepare","writer","train","eval","calibrate"),required=True)
    p.add_argument("--model",required=True); p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--task",choices=("classic","extended"),default="classic")
    p.add_argument("--cache",type=Path); p.add_argument("--checkpoint",type=Path)
    p.add_argument("--train-size",type=int,default=4096); p.add_argument("--eval-size",type=int,default=128)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--updates",type=int,default=512)
    p.add_argument("--writer",choices=("last_token","masked_mean"))
    p.add_argument("--learned-b",action="store_true",default=None); p.add_argument("--mlp",type=int)
    p.add_argument("--method",default="base"); p.add_argument("--scale",type=float,default=20.)
    p.add_argument("--teacher",action="store_true")
    p.add_argument("--max-cases",type=int)
    p.add_argument("--key-consistency",type=float,default=.05)
    p.add_argument("--resume-optimizer",action="store_true")
    a=p.parse_args(); a.run_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(a.seed); random.seed(a.seed)
    manifest=dict(protocol=PROTOCOL,complete=False,started_at=datetime.now(timezone.utc).isoformat(),
                  configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  sources={x.name:sha256(x) for x in Path(__file__).parent.glob("*.py")})
    json_write(a.run_dir/"manifest.json",manifest)
    try:
        if json.loads((Path(a.model).parent/"manifest.json").read_text())["revision"]!=REVISION: raise ValueError("Wrong model revision")
        backend=CounterfactualBackend(a.model,layer=20)
        before=parameter_digest(backend.model)
        if a.stage=="prepare": result=prepare(backend,a)
        else:
            packet=torch.load(a.cache,map_location="cpu",weights_only=True)
            if packet["protocol"]!=PROTOCOL or packet["model_revision"]!=REVISION: raise ValueError("Wrong feature protocol")
            module,checkpoint=load_module(a.checkpoint,backend.device,a.writer,a.learned_b,a.mlp)
            manifest.update(cache_sha256=sha256(a.cache),checkpoint_sha256=sha256(a.checkpoint))
            if a.stage=="writer": result=train_writer(backend,module,packet,a)
            elif a.stage=="eval": result=evaluate(backend,module,packet,a)
            elif a.stage=="calibrate": result=calibrate(backend,packet,a)
            else: result=train_model(backend,module,packet,checkpoint,a)
        if before!=parameter_digest(backend.model) or any(p.grad is not None for p in backend.model.parameters()): raise AssertionError("Backbone mutated")
        manifest.update(complete=True,result=result,backbone_unchanged=True,finished_at=datetime.now(timezone.utc).isoformat())
        json_write(a.run_dir/"manifest.json",manifest)
    except BaseException as error:
        manifest.update(error=repr(error)); json_write(a.run_dir/"manifest.json",manifest); raise

if __name__=="__main__": main()
