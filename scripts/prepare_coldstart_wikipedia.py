"""Pin, download and prepare local Wikipedia/matched/fresh-synthetic JSONL.

Raw source and generated records belong outside Git. Only read-only public
Hub requests are made. No model weights, GPUs or remote hosts are used.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from vera_mem.coldstart_data import PROTOCOL, SPLITS, SYNTHETIC_SEEDS, WIKI_SPLIT_SEED, digest, prepare_wikipedia, matched_records, synthetic_records

REPO="wikimedia/wikipedia"
CONFIG="20231101.en"
SIZES={"0000.parquet":420296449,"0001.parquet":351448367}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()


def json_write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def resolve_metadata(output,offline=False):
    import requests
    paths={"convert":output/"hub_convert_revision.json","source":output/"hub_source_revision.json"}
    urls={"convert":f"https://huggingface.co/api/datasets/{REPO}/revision/refs%2Fconvert%2Fparquet",
          "source":f"https://huggingface.co/api/datasets/{REPO}"}
    values={}
    for name,path in paths.items():
        if not path.exists():
            if offline:raise FileNotFoundError(path)
            response=requests.get(urls[name],timeout=60);response.raise_for_status();json_write(path,response.json())
        values[name]=json.loads(path.read_text())
        if not isinstance(values[name].get("sha"),str) or len(values[name]["sha"])!=40:raise ValueError("Missing immutable Hub commit")
    readme=output/"source_README.md"
    if not readme.exists():
        if offline:raise FileNotFoundError(readme)
        response=requests.get(f'https://huggingface.co/datasets/{REPO}/raw/{values["source"]["sha"]}/README.md',timeout=60)
        response.raise_for_status();readme.write_text(response.text)
    return dict(dataset=REPO,config=CONFIG,source_split="train",convert_revision=values["convert"]["sha"],
                source_card_revision=values["source"]["sha"],license=values["source"].get("cardData",{}).get("license"),
                metadata_urls=urls,source_readme_sha256=file_sha(readme))


def shard(output,source,index,offline=False):
    import requests
    name=f"{index:04d}.parquet";path=output/"raw"/name;path.parent.mkdir(exist_ok=True)
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{source["convert_revision"]}/{CONFIG}/train/{name}'
    if not path.exists():
        if offline:raise FileNotFoundError(path)
        response=requests.get(url,stream=True,timeout=(30,120));response.raise_for_status()
        partial=path.with_suffix(".parquet.partial")
        with partial.open("wb") as f:
            for chunk in response.iter_content(2**20):f.write(chunk)
        if partial.stat().st_size!=SIZES[name]:raise ValueError("Unexpected shard size")
        partial.replace(path)
    if path.stat().st_size!=SIZES[name]:raise ValueError("Truncated/wrong shard")
    return dict(path=str(path.resolve()),filename=name,url=url,bytes=path.stat().st_size,sha256=file_sha(path))


def article_iterator(shards):
    import pyarrow.parquet as pq
    for shard in shards:
        offset=0
        for batch in pq.ParquetFile(shard["path"]).iter_batches(batch_size=256,columns=["id","url","title","text"]):
            for row in batch.to_pylist():
                yield dict(row,source_shard=shard["filename"],source_row=offset)
                offset+=1
        print(json.dumps({"scanned_shard":shard["filename"],"rows":offset}),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--train-size",type=int,default=32768)
    p.add_argument("--dev-size",type=int,default=64)
    p.add_argument("--confirm-size",type=int,default=128)
    p.add_argument("--max-shards",type=int,choices=[1,2],default=2)
    p.add_argument("--offline",action="store_true")
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=True)
    manifest_path=a.output/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("complete"):
        previous=json.loads(manifest_path.read_text())
        requested=dict(train=a.train_size,dev=a.dev_size,confirm=a.confirm_size)
        if previous["requested_counts"]!=requested:raise ValueError("Complete dataset exists with different sizes")
        for domain,parts in previous["files"].items():
            for split,info in parts.items():
                if file_sha(a.output/domain/(split+".jsonl"))!=info["sha256"]:raise ValueError("Existing data changed")
        print(json.dumps({"complete":True,"reused":True,"counts":previous["counts"]}));return
    started=time.perf_counter();source=resolve_metadata(a.output,a.offline)
    counts=dict(train=a.train_size,dev=a.dev_size,confirm=a.confirm_size)
    manifest=dict(protocol=PROTOCOL,complete=False,started_at=datetime.now(timezone.utc).isoformat(),source=source,
                  requested_counts=counts,seeds=dict(wikipedia_split=WIKI_SPLIT_SEED,synthetic=SYNTHETIC_SEEDS))
    json_write(manifest_path,manifest)
    shards=[]
    for index in range(a.max_shards):
        shards.append(shard(a.output,source,index,a.offline))
        print(json.dumps({"source_pinned":source["convert_revision"],"shard":shards[-1]}),flush=True)
        try:
            wiki,selection=prepare_wikipedia(article_iterator(shards),counts)
            break
        except ValueError as error:
            if not str(error).startswith("Insufficient") or index+1==a.max_shards:raise
            print(str(error),flush=True)
    matched={split:matched_records(rows) for split,rows in wiki.items()}
    synth,synthetic_stats=synthetic_records(extra_excluded_answers={r[k] for rows in wiki.values() for r in rows for k in ("a","b")})
    files={};all_data=dict(wikipedia=wiki,matched=matched,synthetic=synth)
    for domain,data in all_data.items():
        folder=a.output/domain;folder.mkdir(exist_ok=True);files[domain]={}
        for split,rows in data.items():
            path=folder/(split+".jsonl")
            with path.open("w") as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
            files[domain][split]=dict(records=len(rows),bytes=path.stat().st_size,sha256=file_sha(path),
                entity_count=len({r["entity"] for r in rows}),answer_count=len({r[k].casefold() for r in rows for k in ("a","b")}))
    manifest.update(complete=True,finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
        shards=shards,selection_counts=selection,synthetic_selection=synthetic_stats,files=files,
        counts={domain:{split:len(rows) for split,rows in data.items()} for domain,data in all_data.items()},
        source_code_sha256={str(path.relative_to(Path(__file__).resolve().parents[1])):file_sha(path) for path in
            (Path(__file__).resolve(),Path(__file__).resolve().parents[1]/"src/vera_mem/coldstart_data.py")},
        matched_contract="Wiki/matched have identical IDs, queries and A/B gold strings; only observed supports differ. Target token schedules can be exactly paired; support/input token costs differ.",
        split_contract="Cluster before selection using page ID and exact normalized title/body/leading-passage matches; one record per cluster; normalized full answers disjoint across train/dev/confirm.",
        heldout_contract="View 0 is canonical; view 1 BEGIN/END note or association blocks is eval-only and must never be used for fitting or model selection on confirm.",
        task_contract="Observed Wikipedia cloze with random note ID, 5-8 word anchor and exact 3-word answer; B is a one-span counterfactual using another article's answer in the same split.",
        limitations=["Exact deduplication does not establish general near-duplicate or semantic separation; no redirect graph or approximate similarity search was available.",
                     "One or two early parquet shards are not an unbiased sample of all English Wikipedia.",
                     "Base-model pretraining exposure to Wikipedia is unknown; counterfactual B tests local observation overrides, not a pure factual-knowledge benchmark.",
                     "Component words/token IDs may overlap across splits. Only complete normalized answers, article clusters and entities are separated.",
                     "Tokenizer checks (answer length, first-token collisions) must be run with the pinned experiment tokenizer before model use."])
    json_write(manifest_path,manifest)
    print(json.dumps({"complete":True,"counts":manifest["counts"],"elapsed_seconds":manifest["elapsed_seconds"],"manifest":str(manifest_path)}),flush=True)


if __name__=="__main__":main()
