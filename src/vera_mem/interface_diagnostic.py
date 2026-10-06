"""Four-bank interventions preserving real sparse retrieval mixtures.

This is a development diagnostic, never used to select on confirmation scores.
Only the target record differs from the canonical bank. First-position routing
equality is verified for value-only interventions; free generation is separate.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import torch

from . import counterfactual_data as data
from .data import synthetic_dataset
from .counterfactual_backend import BatchedStableVectorVeRA, CounterfactualBackend
from .counterfactual_eval import _frozen_eval, _independent_update
from .context_distillation_run import REVISION, clear_bank, parameter_digest, sha256
from .metrics import exact_match
from .run import json_write, tensor_digest
from .vector_store import PersistentVectorDB

PREFIX = "Remember this information: "


def cross_bank(base, index, canonical_key, canonical_value, held_key, held_value, kind):
    if kind not in ("KcVc", "KhVc", "KcVh", "KhVh"):
        raise ValueError("Unknown intervention")
    key = held_key if kind.startswith("Kh") else canonical_key
    value = held_value if kind.endswith("Vh") else canonical_value
    return _independent_update(base, base.ids[index], key, value, 1)


@torch.no_grad()
def evaluate(backend, module, pairs, output, max_new_tokens=16):
    styles = ["train_support_00", "cf_confirmation_support_json", "cf_confirmation_support_markdown"]
    texts = [data.render_support(p, style, world=w) for p in pairs for w in ("A", "B") for style in styles]
    feats = backend.layer_features([PREFIX + t for t in texts], batch_size=32).view(len(pairs), 2, 3, -1)
    keys = module.encode_key(feats.to(backend.device)).cpu()
    values = module.encode_value(feats.to(backend.device)).cpu()
    base = PersistentVectorDB(module.key_dim, module.rank, module.temperature, module.top_k)
    for i, p in enumerate(pairs):
        base.write(p.id, keys[i, 0, 0], values[i, 0, 0], 0)
    base.save(output/"base_bank")
    torch.save(dict(features=feats, pairs=[asdict(p) for p in pairs], styles=styles), output/"features.pt")
    rows = []
    equality_checks = 0
    with _frozen_eval(backend, module), (output/"predictions.jsonl").open("w") as log:
        for i, pair in enumerate(pairs):
            question = data.render_question(pair, "train_query_00")
            a_id = backend.tokenizer.encode(pair.a.answer, add_special_tokens=False)[0]
            b_id = backend.tokenizer.encode(pair.b.answer, add_special_tokens=False)[0]
            for style_index in (1, 2):
                for world_index, world in enumerate(("A", "B")):
                    answer = pair.a.answer if world == "A" else pair.b.answer
                    references = {}
                    for kind in ("KcVc", "KhVc", "KcVh", "KhVh", "teacher"):
                        if kind == "teacher":
                            clear_bank(backend)
                            backend.mode = "none"
                            context = data.render_support(pair, styles[style_index], world=world)
                            prediction, length, seconds = backend.generate(question, context=context, max_new_tokens=max_new_tokens)
                            row = dict(id=pair.id, style=styles[style_index], world=world, kind=kind,
                                       prediction=prediction, answer=answer, em=exact_match(prediction, answer),
                                       tokens=length, seconds=seconds)
                        else:
                            db = cross_bank(base, i, keys[i,world_index,0], values[i,world_index,0],
                                            keys[i,world_index,style_index], values[i,world_index,style_index], kind)
                            before = db.hash()
                            clear_bank(backend)
                            backend.mode, backend.vector_vera, backend.vector_store = "vector_vera", module, db
                            first = backend.forward_sequences([question], [[a_id]])
                            q = module.encode_query(first["query_inputs"]).cpu()
                            search = db.search(q)
                            route = (search["indices"].clone(), search["weights"].clone())
                            key_family = kind[:2]
                            if key_family in references:
                                if not all(torch.equal(x,y) for x,y in zip(route,references[key_family])):
                                    raise AssertionError("Value-only intervention changed first-position routing")
                                equality_checks += 1
                            else:
                                references[key_family] = route
                            logits = first["logits"][0]
                            margin = float(logits[a_id]-logits[b_id])
                            first_correct = int(logits.argmax() == (a_id if world == "A" else b_id))
                            del first, logits
                            backend.retrieval_trace, backend.trace_retrieval = [], True
                            prediction, length, seconds = backend.generate(question, max_new_tokens=max_new_tokens)
                            backend.trace_retrieval = False
                            trace = [r["indices"][0].tolist() for r in backend.retrieval_trace]
                            if db.hash() != before:
                                raise AssertionError("Read mutated bank")
                            row = dict(id=pair.id, style=styles[style_index], world=world, kind=kind,
                                       prediction=prediction, answer=answer, em=exact_match(prediction,answer),
                                       tokens=length, seconds=seconds, first_correct=first_correct,
                                       margin_a_minus_b=margin, indices=route[0].tolist(), weights=route[1].tolist(),
                                       recall1=int(route[0][0,0] == i), recall4=int(i in route[0][0].tolist()),
                                       decode_hits=sum(i in x for x in trace[1:]), decode_steps=len(trace)-1,
                                       bank_hash=before, bank_unchanged=True)
                        rows.append(row)
                        log.write(json.dumps(row)+"\n"); log.flush()
            if (i+1)%8 == 0:
                print(json.dumps(dict(completed_facts=i+1,total=len(pairs))), flush=True)
    summary={}
    for style in styles[1:]:
        summary[style]={}
        for kind in ("KcVc", "KhVc", "KcVh", "KhVh", "teacher"):
            subset=[r for r in rows if r["style"]==style and r["kind"]==kind]
            paired=sum(all(r["em"] for r in subset if r["id"]==p.id) for p in pairs)
            summary[style][kind]=dict(paired_correct=paired,n=len(pairs),single_correct=sum(r["em"] for r in subset),
                first_correct=sum(r.get("first_correct",0) for r in subset),
                recall1=sum(r.get("recall1",0) for r in subset))
    result=dict(protocol="interface-four-bank-dev-v1",summary=summary,value_only_route_equal_checks=equality_checks,
                max_new_tokens=max_new_tokens,all_rows=len(rows),single_target_updates=True)
    json_write(output/"results.json",result)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--run-dir",type=Path,required=True); p.add_argument("--size",type=int,default=64); p.add_argument("--seed",type=int,default=47042)
    a=p.parse_args(); a.run_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(42)
    manifest=dict(complete=False,started_at=datetime.now(timezone.utc).isoformat(),seed=a.seed,size=a.size,
                  checkpoint_sha256=sha256(a.checkpoint),source_sha256={x.name:sha256(x) for x in Path(__file__).parent.glob("*.py")})
    json_write(a.run_dir/"manifest.json",manifest)
    try:
        if json.loads((Path(a.model).parent/"manifest.json").read_text())["revision"] != REVISION:
            raise ValueError("Wrong model revision")
        fresh=synthetic_dataset(a.seed,a.size,0)["stream"]
        pairs=data._make_pairs(fresh,a.seed+1000)
        historic=data.datasets()
        excluded={p.entity for group in (historic["train"],historic["dev"],historic["confirmation"]) for p in group}
        if any(p.entity in excluded for p in pairs): raise ValueError("Entity split collision")
        backend=CounterfactualBackend(a.model,layer=20)
        module=BatchedStableVectorVeRA(9728,2560,rank=64,key_dim=64,top_k=4,temperature=.05,seed=42).to(backend.device)
        module.load_state_dict(torch.load(a.checkpoint,map_location="cpu",weights_only=True)["module"])
        before=parameter_digest(backend.model); module_before=tensor_digest(module)
        result=evaluate(backend,module,pairs,a.run_dir)
        if before != parameter_digest(backend.model) or module_before != tensor_digest(module):
            raise AssertionError("Diagnostic changed parameters")
        manifest.update(complete=True,results=result,parameters_unchanged=True,finished_at=datetime.now(timezone.utc).isoformat())
        json_write(a.run_dir/"manifest.json",manifest)
    except BaseException as error:
        manifest.update(error=repr(error)); json_write(a.run_dir/"manifest.json",manifest); raise

if __name__=="__main__": main()
