"""Make a clearly labeled, CPU-only memorization probe from seen train targets.

Both view slots are exact copies of canonical view 0. The compatibility field
split=dev is NOT a heldout claim: original_split=train and the diagnostic marker
must take precedence in reporting. No encoders or model inference are run.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
DIAGNOSTIC="train_memorization_duplicate_canonical_views"


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()


def build_probe(packet,source_sha256,*,count=128,seed=63042):
    import torch
    from vera_mem.coldstart_run import BankSampler,PROTOCOL
    if packet.get("protocol")!=PROTOCOL or packet.get("split")!="train" or packet.get("task")!="synthetic":
        raise ValueError("Probe requires the synthetic training cache")
    if count<8 or count%8 or len(packet["rows"])<count:raise ValueError("Probe count must be a batch-aligned training subset")
    if packet["q"].shape[1]!=1 or packet["last"].shape[2]!=1 or packet["pool"].shape[2]!=1:
        raise ValueError("Only canonical training features may enter this probe")
    sampler=BankSampler(packet["rows"],seed,batch_size=8)
    selected=[]
    while len(selected)<count:selected.extend(sampler.next()["targets"])
    if len(set(selected))!=count:raise ValueError("Selected probe targets are not unique")
    rows=copy.deepcopy([packet["rows"][i] for i in selected])
    for index,row in zip(selected,rows):
        if row.get("split")!="train" or len(row["questions"])!=1 or any(len(v)!=1 for v in row["supports"]):
            raise ValueError("Train rows unexpectedly contain heldout text")
        row["questions"]=[row["questions"][0]]*2
        row["supports"]=[[world[0]]*2 for world in row["supports"]]
        row["provenance"]=dict(row.get("provenance",{}),original_split="train",source_index=index,diagnostic=DIAGNOSTIC)
    output={key:copy.deepcopy(value) for key,value in packet.items() if key not in {"rows","q","last","pool","split","task","qids","sids"}}
    output.update(rows=rows,q=packet["q"][selected].repeat(1,2,1).clone(),
        last=packet["last"][selected].repeat(1,1,2,1).clone(),pool=packet["pool"][selected].repeat(1,1,2,1).clone(),
        split="dev",original_split="train",task="synthetic_train_probe",diagnostic=DIAGNOSTIC,
        qids=["canonical","canonical_duplicate"],sids=["canonical","canonical_duplicate"],
        source_cache_sha256=source_sha256,source_indices=selected,
        target_selection=dict(sampler="coldstart_run.BankSampler",seed=seed,batch_size=8,first_target_exposures=count),
        reporting_note="Seen training facts with duplicated canonical views; CC/CH/HC/HH labels here do not represent heldout formats or generalization.")
    assert torch.equal(output["q"][:,0],output["q"][:,1])
    assert all(torch.equal(output[k][:,:,0],output[k][:,:,1]) for k in ("last","pool"))
    return output


def main(argv=None):
    import torch
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv);torch.set_num_threads(4)
    if a.output.exists() or a.output.with_suffix(".manifest.json").exists():raise FileExistsError("Refusing to replace probe")
    packet=torch.load(a.cache,map_location="cpu",weights_only=True,mmap=True)
    probe=build_probe(packet,file_sha(a.cache));a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(probe,a.output)
    manifest=dict(complete=True,diagnostic=DIAGNOSTIC,original_split="train",compatibility_split="dev",task=probe["task"],
        records=len(probe["rows"]),target_selection=probe["target_selection"],source_cache=str(a.cache),source_cache_sha256=probe["source_cache_sha256"],
        output_sha256=file_sha(a.output),source_indices=probe["source_indices"],selected_ids=[r["id"] for r in probe["rows"]],
        source_script_sha256=file_sha(Path(__file__)),all_view_slots_bitwise_equal=True,cuda_initialized=torch.cuda.is_initialized(),reporting_note=probe["reporting_note"])
    a.output.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("complete","diagnostic","records","output_sha256","cuda_initialized")}))


if __name__=="__main__":main()
