"""Template-held-out augmentation experiments with causal writable VeRA memory.

The backbone, random projections, per-token query path, and VDB are unchanged.
Only offline wording diversity and optional paired-view consistency differ.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from . import augmentation_data as data
from .data import Example
from .metrics import save_examples
from .run import json_write, tensor_digest, digest_file
from .scaling_backend import ScalingQwenBackend
from .scaling_run import validation, evaluate, stats
from .stable_vector_vera import StableVectorVeRA
from .vector_store import PersistentVectorDB

REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
PROTOCOL = "generalization-v1-layer20"


def templates(split, kind):
    train = data.get_templates("train", kind)
    return train if split == "train" else (train[0], *data.get_templates(split, kind))


def prepare_features(backend, cache):
    fingerprint = data.protocol_fingerprint()
    if cache.exists():
        packet = torch.load(cache, map_location="cpu", weights_only=True)
        if (packet["protocol"], packet["model_revision"], packet["data_fingerprint"]) != (PROTOCOL, REVISION, fingerprint):
            raise ValueError("Feature cache provenance mismatch")
        return packet
    examples = data.datasets()
    packet = dict(protocol=PROTOCOL, model_revision=REVISION, data_fingerprint=fingerprint,
                  examples={k: [asdict(e) for e in v] for k,v in examples.items()},
                  features={}, template_ids={})
    for split, rows in examples.items():
        if split == "control":
            continue
        packet["features"][split], packet["template_ids"][split] = {}, {}
        for kind in ("q", "s"):
            forms = templates(split, "query" if kind == "q" else "support")
            packet["template_ids"][split][kind] = [t.id for t in forms]
            render = data.render_question if kind == "q" else data.render_support
            arrays = []
            for form in forms:
                print(f"FEATURES {split} {form.id}", flush=True)
                # Observation rendering contains the revealed fact. Queries do not.
                texts = [render(e, form.id) for e in rows]
                if kind == "s":
                    texts = ["Remember this information: " + text for text in texts]
                arrays.append(backend.layer_features(texts, batch_size=32))
                backend.layer_feature_cache.clear()
            packet["features"][split][kind] = torch.stack(arrays, dim=1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(".partial")
    torch.save(packet, partial)
    partial.replace(cache)
    return packet


def make_module(backend, seed):
    return StableVectorVeRA(backend.target.in_features, backend.target.out_features,
                            rank=64, key_dim=64, top_k=4, temperature=0.05, seed=seed).to("cuda")


def choose_views(features, indices, rng, diversity):
    views = [rng.randrange(features.shape[1]) if diversity else 0 for _ in indices]
    index = torch.tensor(indices, device=features.device)
    return features[index, views], views


def view_consistency(module, q1, q2, s1, s2, values=True):
    qloss = (1 - (module.encode_query(q1) * module.encode_query(q2)).sum(-1)).mean()
    kloss = (1 - (module.encode_key(s1) * module.encode_key(s2)).sum(-1)).mean()
    vloss = F.mse_loss(module.encode_value(s1), module.encode_value(s2)) if values else qloss.new_zeros(())
    return qloss + kloss + vloss


@torch.no_grad()
def validate(backend, module, packet):
    rows = [Example(**e) for e in packet["examples"]["dev"]]
    q, s = [packet["features"]["dev"][k].to("cuda") for k in ("q", "s")]
    result = {}
    # Development-only wording families; confirmation families are never scored here.
    for qi, si in [(0,0), (1,0), (2,0), (1,1), (2,2)]:
        examples = [replace(e, question=data.render_question(e, packet["template_ids"]["dev"]["q"][qi])) for e in rows]
        result[f"q{qi}_s{si}"] = validation(backend, module, examples, q[:,qi], s[:,si])
    result["selection_nll"] = float(np.mean([r["real_answer_token_nll"] for r in result.values()]))
    return result


def train(backend, cfg, packet, output):
    module = make_module(backend, cfg["seed"])
    backend.vector_vera = module
    n = cfg["train_size"]
    examples = [Example(**e) for e in packet["examples"]["train"][:n]]
    qt, st = [packet["features"]["train"][k][:n].to("cuda") for k in ("q", "s")]
    diversity = cfg["condition"] != "canonical"
    invariant = cfg["condition"] == "invariant"
    module.fit_statistics((qt if diversity else qt[:,:1]).reshape(-1,qt.shape[-1]),
                          (st if diversity else st[:,:1]).reshape(-1,st.shape[-1]))
    alignment_fact_rng = random.Random(cfg["seed"]+10)
    rng = random.Random(cfg["seed"]+20)
    auxiliary_rng = random.Random(cfg["seed"]+30)
    history = dict(alignment=[], training=[])
    address_parameters = [*module.Wq.parameters(), *module.Wk.parameters()]
    optimizer = torch.optim.Adam(address_parameters, lr=1e-4)
    for step in range(cfg["alignment_steps"]):
        indices = alignment_fact_rng.sample(range(n), min(128,n))
        q, _ = choose_views(qt,indices,rng,diversity)
        s, _ = choose_views(st,indices,rng,diversity)
        optimizer.zero_grad(set_to_none=True)
        loss = module.contrastive_loss(q,s,temperature=0.1)
        # Each column is a distinct entity. Paired views are never false negatives.
        if invariant:
            q2,_ = choose_views(qt,indices,auxiliary_rng,True)
            s2,_ = choose_views(st,indices,auxiliary_rng,True)
            loss = loss + cfg["consistency_weight"] * view_consistency(module,q,q2,s,s2,values=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(address_parameters,1.)
        optimizer.step()
        if (step+1)%100 == 0 or step+1 == cfg["alignment_steps"]:
            row=dict(step=step+1,loss=float(loss.detach()))
            history["alignment"].append(row)
            print(json.dumps(dict(alignment=row)),flush=True)
    optimizer = torch.optim.Adam([
        dict(params=module.Wv.parameters(),lr=1e-4),
        dict(params=[module.b],lr=0.005),
        dict(params=address_parameters,lr=1e-5),
    ])
    order,cursor,best,losses=[],0,float("inf"),[]
    start=time.perf_counter()
    # Separate RNG means template choices do not change which facts are targets.
    fact_rng=random.Random(cfg["seed"]+100)
    rng=random.Random(cfg["seed"]+120)
    auxiliary_rng=random.Random(cfg["seed"]+130)
    seen_tokens=0
    for step in range(cfg["updates"]):
        if cursor+cfg["batch_size"]>len(order):
            order=list(range(n)); fact_rng.shuffle(order); cursor=0
        targets=order[cursor:cursor+cfg["batch_size"]]; cursor+=len(targets)
        episode=list(dict.fromkeys(targets+fact_rng.sample(range(n),min(64,n))))
        fact_rng.shuffle(episode)
        q, qviews=choose_views(qt,episode,rng,diversity)
        s, sviews=choose_views(st,episode,rng,diversity)
        mapping=torch.tensor([episode.index(i) for i in targets],device="cuda")
        phase="oracle" if step<cfg["oracle_updates"] else "real"
        backend.mode,backend.vector_store,backend.vector_override="vector_vera",None,None
        backend.vector_keys,module_values=module.encode_key(s),module.encode_value(s)
        backend.vector_values=module_values
        backend.oracle_values=module_values[mapping] if phase=="oracle" else None
        questions=[data.render_question(examples[i],packet["template_ids"]["train"]["q"][qviews[episode.index(i)]]) for i in targets]
        seen_tokens+=sum(len(backend.prompt_ids(question)) for question in questions)
        optimizer.zero_grad(set_to_none=True)
        lm=backend.batched_loss(questions,[examples[i].answer for i in targets])
        # Keep negative entities present even during the oracle curriculum:
        # positive-view consistency alone must not erase learned addressing.
        address=module.contrastive_loss(q,s,temperature=0.1)
        if phase=="real":
            labels=mapping[backend.last_answer_batch_indices]
            token_q=module.encode_query(backend.last_answer_query_inputs)
            address=address+F.cross_entropy(token_q@backend.vector_keys.T/0.1,labels)
        consistency=lm.new_zeros(())
        if invariant:
            q2,_=choose_views(qt,episode,auxiliary_rng,True)
            s2,_=choose_views(st,episode,auxiliary_rng,True)
            consistency=view_consistency(module,q,q2,s,s2)
        total=lm+0.2*address+cfg["consistency_weight"]*consistency
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite training loss")
        total.backward()
        for parameters in ([module.b],list(module.Wv.parameters()),address_parameters):
            torch.nn.utils.clip_grad_norm_(parameters,1.)
        optimizer.step()
        losses.append(float(lm.detach()))
        if (step+1)%cfg["validate_every"]==0 or step+1 in (cfg["updates"],cfg["oracle_updates"]):
            val=validate(backend,module,packet)
            row=dict(step=step+1,phase=phase,exposures=(step+1)*cfg["batch_size"],
                     train_lm_loss=float(np.mean(losses)),address_loss=float(address.detach()),
                     consistency_loss=float(consistency.detach()),seconds=time.perf_counter()-start,
                     prompt_tokens_seen=seen_tokens,validation=val)
            losses.clear(); history["training"].append(row)
            checkpoint=dict(module=module.state_dict(),config=cfg,step=step+1)
            torch.save(checkpoint,output/"last.pt")
            if val["selection_nll"]<best:
                best=val["selection_nll"]; torch.save(checkpoint,output/"best.pt")
            json_write(output/"training.json",history)
            print(json.dumps(dict(training=row)),flush=True)
    checkpoint=torch.load(output/"best.pt",map_location="cuda",weights_only=True)
    module.load_state_dict(checkpoint["module"])
    module.requires_grad_(False).eval()
    backend.oracle_values=None; backend.mode="none"
    return module,checkpoint["step"]


def online(backend,module,cfg,packet,output):
    initial_hash=tensor_digest(module)
    rows=[Example(**e) for e in packet["examples"]["test"][:cfg["eval_size"]]]
    controls=[Example(**e) for e in packet["examples"]["control"]]
    support=packet["features"]["test"]["s"][:len(rows)]
    backend.vector_vera=module
    results={}; writes=[]
    # Independent stores separate query OOD from observation OOD. No replay or
    # parameter updates online; each support is revealed only at its write.
    for bank in ("canonical_support","heldout_support"):
        db=PersistentVectorDB(64,64,top_k=module.top_k,temperature=module.temperature)
        pre,immediate=[],[]
        if bank=="canonical_support":
            before=evaluate(backend,module,db,controls,"real","control_before",output)
        for i,ex in enumerate(rows):
            if bank=="canonical_support":
                pre.extend(evaluate(backend,module,db,[ex],"real","pre_write",output))
            view=0 if bank=="canonical_support" else 1+i%(support.shape[1]-1)
            with torch.no_grad():
                hidden=support[i,view].to("cuda")
                db.write(ex.id,module.encode_key(hidden).cpu(),module.encode_value(hidden).cpu(),i)
            writes.append(dict(id=ex.id,bank=bank,support_template=packet["template_ids"]["test"]["s"][view],timestamp=i))
            if bank=="canonical_support":
                immediate.extend(evaluate(backend,module,db,[ex],"real","immediate",output))
        db.save(output/(bank+"_vdb.pt"))
        db_hash=db.hash()
        scores={}
        for qi,template_id in enumerate(packet["template_ids"]["test"]["q"]):
            examples=[replace(e,question=data.render_question(e,template_id)) for e in rows]
            scores[template_id]={}
            for method in ("real","oracle","shuffled","empty"):
                phase=bank+"/"+template_id
                scores[template_id][method]=stats(evaluate(backend,module,db,examples,method,phase,output))
            print(json.dumps(dict(evaluation=bank,template=template_id,scores=scores[template_id])),flush=True)
        if db.hash()!=db_hash:
            raise AssertionError("Read-only evaluation changed VDB")
        results[bank]=dict(scores=scores,records=len(db.ids),numeric_vdb_bytes=db.resident_bytes())
        if pre:
            after=evaluate(backend,module,db,controls,"real","control_after",output)
            results[bank].update(pre_write=stats(pre),immediate=stats(immediate),
                                 control_before=stats(before),control_after=stats(after))
    if tensor_digest(module)!=initial_hash:
        raise AssertionError("Online evaluation changed module weights")
    result=dict(conditions=results,online_gradient_steps=0,shared_parameters_unchanged=True,
                eval_facts=len(rows),model_id="Qwen/Qwen3-4B-Instruct-2507")
    json_write(output/"writes.json",writes)
    json_write(output/"metrics.json",result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--run-dir",type=Path)
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--condition",choices=["canonical","augment","invariant"],default="augment")
    p.add_argument("--train-size",type=int,choices=[32,128,1024,4096],default=4096)
    p.add_argument("--eval-size",type=int,choices=[16,32,128],default=128)
    p.add_argument("--updates",type=int,default=1536)
    p.add_argument("--oracle-updates",type=int,default=768)
    p.add_argument("--alignment-steps",type=int,default=400)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--validate-every",type=int,default=256)
    p.add_argument("--consistency-weight",type=float,default=0.1)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if min(a.updates,a.alignment_steps,a.batch_size,a.validate_every)<1 or not 0<=a.oracle_updates<=a.updates:
        raise ValueError("Invalid training budget")
    if a.batch_size>a.train_size or a.consistency_weight<0:
        raise ValueError("Invalid batch size or regularization")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(4)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    backend=ScalingQwenBackend(a.model,layer=20)
    manifest=json.loads((Path(a.model).parent/"manifest.json").read_text())
    if manifest["revision"]!=REVISION:
        raise ValueError("Pinned model revision required")
    packet=prepare_features(backend,a.cache)
    if a.prepare_only:
        print("FEATURE_CACHE_COMPLETE",flush=True); return
    if a.run_dir is None:
        raise ValueError("--run-dir required")
    if a.train_size==32:
        # Smoke must not consume confirmation cases: evaluate development facts.
        packet["examples"]["test"]=packet["examples"]["dev"]
        packet["features"]["test"]=packet["features"]["dev"]
        packet["template_ids"]["test"]=packet["template_ids"]["dev"]
        packet["examples"]["control"]=[asdict(replace(Example(**e),metadata={
            **e["metadata"],"split":"smoke_control","never_written":True}))
            for e in packet["examples"]["dev"][-16:]]
        cfg["smoke_uses_dev_only"]=True
    a.run_dir.mkdir(parents=True,exist_ok=False)
    json_write(a.run_dir/"config.json",cfg)
    protocol=data.protocol_manifest()
    protocol["manifest_describes"]="Full immutable feature-cache pool; actual run selection follows"
    protocol["actual_run_selection"]=dict(train_size=a.train_size,eval_size=a.eval_size,
        evaluation_fact_split="dev" if a.train_size==32 else "test",
        smoke=a.train_size==32,control_size=len(packet["examples"]["control"]))
    json_write(a.run_dir/"data_protocol.json",protocol)
    provenance=dict(started_at=datetime.now(timezone.utc).isoformat(),complete=False,
                    cache_sha256=digest_file(a.cache),data_fingerprint=packet["data_fingerprint"],
                    model_revision=REVISION,gpu=torch.cuda.get_device_name(0),
                    source_files_sha256={f.name:digest_file(f) for f in Path(__file__).parent.glob("*.py")})
    json_write(a.run_dir/"manifest.json",provenance)
    for split,rows in packet["examples"].items():
        if split=="train": rows=rows[:a.train_size]
        save_examples(a.run_dir/(split+".jsonl"),[Example(**e) for e in rows])
    module,step=train(backend,cfg,packet,a.run_dir)
    result=online(backend,module,cfg,packet,a.run_dir)
    provenance.update(complete=True,selected_step=step,completed_at=datetime.now(timezone.utc).isoformat())
    json_write(a.run_dir/"manifest.json",provenance)
    print(json.dumps(dict(RUN_COMPLETE=result)),flush=True)

if __name__=="__main__":
    main()
