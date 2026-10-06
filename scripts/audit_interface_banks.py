"""Replay saved CPU vector banks for completed interface evaluation jobs.

Use a torch-enabled CPU Python environment:
  python scripts/audit_interface_banks.py --runs-root WORKSPACE/runs --output AUDIT
Only complete suite jobs with complete eval manifests/metrics are opened.
Partial confirmation generations are never read. No encoders, model inference,
CUDA allocation, training checkpoints, or feature caches are loaded. This
complements summarize_interface.py's independent prediction-arithmetic audit.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import torch

from vera_mem.vector_store import PersistentVectorDB
from summarize_interface import PHASES, METHODS, artifacts, file_hash, read_json, read_jsonl, require, sha


PROTOCOL = "interface-tensor-bank-audit-v1"


def restore(snapshot):
    """Use the real validated loader, retaining its exact non-renormalizing path."""
    buffer=io.BytesIO()
    torch.save(snapshot,buffer)
    buffer.seek(0)
    return PersistentVectorDB.load(buffer)


def shuffle_bank(bank, permutation):
    result=copy.deepcopy(bank)
    result._values=bank.values[permutation].contiguous().clone()
    require(torch.equal(result.keys,bank.keys) and result.ids==bank.ids and result.timestamps==bank.timestamps,
            "Shuffle changed keys/IDs/timestamps")
    return result


def replay_patch(base, target, patch):
    require(set(patch)=={"write_key","key","value","timestamp","bank_hash","parent_hash"},"Unexpected B patch fields")
    require(patch["parent_hash"]==base.hash(),"Patch parent hash differs from base")
    require(target in base.ids,"B target missing from bank")
    index=base.ids.index(target)
    require(type(patch["timestamp"]) is int and patch["timestamp"]==len(base)+index,"B timestamp differs from independent-update protocol")
    for name,dimension in (("write_key",base.key_dim),("key",base.key_dim),("value",base.value_dim)):
        tensor=patch[name]
        require(isinstance(tensor,torch.Tensor) and tensor.device.type=="cpu" and tensor.dtype==torch.float32 and tensor.shape==(dimension,) and torch.isfinite(tensor).all(),
                "Invalid CPU patch tensor: "+name)
    changed=copy.deepcopy(base)
    # write_key is the ORIGINAL encoder output. Rewriting the already stored
    # normalized key can change its last bits and cannot reproduce this hash.
    changed.write(target,patch["write_key"],patch["value"],patch["timestamp"])
    require(changed.hash()==patch["bank_hash"],"Replayed B bank hash mismatch")
    require(torch.equal(changed.keys[index],patch["key"]),"Saved normalized key differs from actual upsert")
    require(torch.equal(changed.values[index],patch["value"]),"Saved value differs from upsert")
    require(changed.ids==base.ids,"B patch changed ID order")
    others=[i for i in range(len(base)) if i!=index]
    require(torch.equal(changed.keys[others],base.keys[others]) and torch.equal(changed.values[others],base.values[others]) and
            all(changed.timestamps[i]==base.timestamps[i] for i in others),"B patch changed another record")
    require(changed.timestamps[index]!=base.timestamps[index],"B patch was not a target update")
    return changed


def audit_run(directory):
    directory=Path(directory)
    manifest=read_json(directory/"manifest.json")
    require(manifest.get("complete") is True and manifest.get("configuration",{}).get("stage")=="eval","Not a completed eval manifest")
    metrics=read_json(directory/"metrics.json")
    require(metrics.get("complete") is True and metrics.get("protocol")=="interface-cpu-vdb-eval-v1","Not a completed interface eval")
    require(metrics.get("split") in {"dev","confirm"},"Unexpected split")
    require(manifest.get("result")==metrics,"Manifest/metrics mismatch")
    assignments=read_json(directory/"assignments.json")
    selected=assignments["selected_case_ids"]
    all_ids=[row["id"] for row in assignments["assignments"]]
    require(len(all_ids)==len(set(all_ids))==metrics["bank_records"] and len(selected)==len(set(selected))==metrics["evaluated_facts"],"Bank/target size mismatch")
    require(set(selected)<=set(all_ids),"Unknown selected target")
    assignment_by_id={row["id"]:row for row in assignments["assignments"]}
    raw=read_jsonl(directory/"predictions.jsonl")
    raw_by_phase={phase:[] for phase in PHASES}
    for row in raw:
        require(row["phase"] in raw_by_phase,"Unexpected raw phase")
        raw_by_phase[row["phase"]].append(row)
    writes={}
    for row in read_jsonl(directory/"writes.jsonl"):
        key=(row["phase"],row["bank_kind"],row["world"],row["id"])
        require(key not in writes,"Duplicate write-log entry")
        writes[key]=row
    expected_writes=set()
    phase_results={}; retained={}; normalized_keys={}; patched_hashes={}
    verified_reads=0; verified_updates=0; shuffled_banks=0
    for phase in PHASES:
        path=directory/"banks"/(phase+".pt")
        require(str(path.relative_to(directory)) in metrics["artifacts"]["banks"],"Missing declared bank snapshot")
        packet=torch.load(path,map_location="cpu",weights_only=True)
        require(packet["protocol"]=="interface-cpu-vdb-eval-v1" and packet["phase"]==phase,"Bank snapshot protocol/phase mismatch")
        require(packet["assignments"]==assignments["assignments"],"Snapshot assignment differs from evaluation")
        kinds={"real","canonical_key"} if phase=="HC" else {"real"}
        require(set(packet["bases"])==set(packet["base_hashes"])==kinds,"Wrong base bank kinds")
        permutation=torch.tensor(packet["shuffle_permutation"],dtype=torch.long)
        require(permutation.tolist()==torch.arange(len(all_ids)).roll(1).tolist(),"Shuffle does not match fixed nonidentity roll")
        bases={kind:restore(snapshot) for kind,snapshot in packet["bases"].items()}
        expected={}; real_by_target={}; checked_hashes={}
        for kind,base in bases.items():
            before=base.hash()
            require(base.ids==tuple(all_ids) and base.timestamps==tuple(range(len(all_ids))),"Base record order/timestamps mismatch")
            require(before==packet["base_hashes"][kind]==metrics["phases"][phase]["base_hashes_before"][kind]==metrics["phases"][phase]["base_hashes_after"][kind],"Base hash mismatch")
            retained[(phase,kind)]=base
            for identity in selected:
                expected[(identity,kind,"A")]=before
            if kind=="real" and phase in {"CC","HC"}:
                shuffled=shuffle_bank(base,permutation); shash=shuffled.hash(); shuffled_banks+=1
                for identity in selected: expected[(identity,"shuffled","A")]=shash
            for index,identity in enumerate(all_ids):
                key=(phase,kind,"A",identity); expected_writes.add(key)
                require(key in writes,"Base write-log record missing")
                row=writes[key]
                si=assignment_by_id[identity]["s"] if phase[0]=="H" else 0
                ki=0 if kind=="canonical_key" else si
                require(row["timestamp"]==index and row["key_support_view"]==ki and row["value_support_view"]==si,"Base write-log timestamp/view mismatch")
            checked_hashes[kind]=before
        interventions=packet["interventions"]
        require([p["id"] for p in interventions]==selected,"Missing/reordered/duplicate target interventions")
        target_results=[]
        for intervention in interventions:
            identity,index=intervention["id"],intervention["index"]
            require(type(index) is int and 0<=index<len(all_ids) and all_ids[index]==identity,"Patch target index mismatch")
            require(set(intervention["banks"])==kinds,"Patch bank kinds mismatch")
            hashes={}
            for kind,patch in intervention["banks"].items():
                base=bases[kind]
                changed=replay_patch(base,identity,patch)
                verified_updates+=1
                expected[(identity,kind,"B")]=changed.hash()
                hashes[kind]=changed.hash()
                patched_hashes[(phase,kind,identity)]=changed.hash()
                normalized_keys[(phase,kind,identity)]=patch["key"].clone()
                real_by_target[(kind,identity)]=changed
                key=(phase,kind,"B",identity); expected_writes.add(key)
                require(key in writes,"B write-log record missing")
                row=writes[key]
                require(row["parent_hash"]==base.hash() and row["bank_hash"]==changed.hash() and row["timestamp"]==patch["timestamp"],"B write log differs from snapshot")
                si=assignment_by_id[identity]["s"] if phase[0]=="H" else 0
                ki=0 if kind=="canonical_key" else si
                require(row["key_support_view"]==ki and row["value_support_view"]==si,"B write-log support views mismatch")
                if kind=="real" and phase in {"CC","HC"}:
                    shuffled=shuffle_bank(changed,permutation)
                    expected[(identity,"shuffled","B")]=shuffled.hash(); shuffled_banks+=1
                require(base.hash()==checked_hashes[kind],"Replaying a patch mutated its base")
            if phase=="HC":
                require(torch.equal(real_by_target[("real",identity)].values,real_by_target[("canonical_key",identity)].values),"HC canonical-key diagnostic changed values")
                require(torch.equal(normalized_keys[(phase,"canonical_key",identity)],normalized_keys[("CC","real",identity)]),"HC canonical-key B update differs from canonical CC key")
            target_results.append(dict(id=identity,index=index,bank_hashes=hashes))
        methods=METHODS[phase]+(("teacher",) if metrics["include_teacher"] else ())
        expected_events={(identity,method,world) for identity in selected for method in methods for world in (("A_and_B",) if method=="empty" else ("A","B"))}
        seen=set()
        for row in raw_by_phase[phase]:
            identity,method,world=row["target_id"],row["method"],row["world"]
            event=(identity,method,world)
            require(event in expected_events and event not in seen,"Unexpected/duplicate raw bank read")
            seen.add(event)
            kind=method if method in {"shuffled","canonical_key"} else "real"
            expected_hash=expected[(identity,kind,"A" if world=="A_and_B" else world)]
            require(row["bank_hash_before"]==row["bank_hash_after"]==expected_hash,"Raw read hash does not match reconstructed bank/world/method")
            sha(expected_hash,"reconstructed read")
            verified_reads+=1
        require(seen==expected_events,"Missing raw bank reads")
        phase_results[phase]=dict(bank_records=len(all_ids),target_interventions=len(selected),base_hashes=checked_hashes,
            raw_reads=len(seen),single_target_updates_verified=True,shuffle_address_invariant_verified=phase in {"CC","HC"},targets=target_results,
            snapshot_sha256=file_hash(path))
    require(set(writes)==expected_writes,"Extra/missing bank write log records")
    require(retained[("CC","real")].hash()==retained[("CH","real")].hash(),"Canonical support bank differs across query conditions")
    require(retained[("HC","real")].hash()==retained[("HH","real")].hash(),"Heldout support bank differs across query conditions")
    require(torch.equal(retained[("HC","canonical_key")].keys,retained[("CC","real")].keys),"HC canonical keys differ from CC keys")
    require(torch.equal(retained[("HC","canonical_key")].values,retained[("HC","real")].values),"HC canonical-key bank differs in values")
    for identity in selected:
        require(patched_hashes[("CC","real",identity)]==patched_hashes[("CH","real",identity)],"Canonical B intervention differs across query conditions")
        require(patched_hashes[("HC","real",identity)]==patched_hashes[("HH","real",identity)],"Heldout B intervention differs across query conditions")
    return dict(directory=str(directory),split=metrics["split"],complete=True,bank_records=len(all_ids),evaluated_facts=len(selected),
        verified_raw_reads=verified_reads,verified_single_target_updates=verified_updates,verified_shuffled_banks=shuffled_banks,
        phases=phase_results,artifacts=artifacts(directory,["manifest.json","metrics.json","assignments.json","writes.jsonl","predictions.jsonl"]),
        note="Teacher read hashes identify the bound observation bank even though teacher mode disables its residual. Empty uses a zero-value override with the unchanged A bank. These mode semantics are not re-executed by this audit.")


def audit_completed(root):
    root=Path(root).resolve()
    require(root.is_dir(),"runs-root must be a filesystem directory")
    paths=set(root.rglob("interface_*/suite.json"))
    if root.name.startswith("interface_") and (root/"suite.json").exists(): paths.add(root/"suite.json")
    results,skipped=[],[]
    for path in sorted(paths):
        if "source" in path.relative_to(root).parts: continue
        suite=read_json(path)
        require(suite.get("protocol")=="interface-experiment-v1","Unsupported suite")
        planned={entry["name"]:entry for entry in read_json(path.parent/"plan.json")}
        for job in suite["jobs"]:
            name=job["name"]
            require(name in planned and Path(name).name==name and name not in {".",".."},"Unplanned/unsafe job")
            if job.get("status")!="complete" or job.get("exit_code")!=0:
                skipped.append(dict(suite=path.parent.name,job=name,reason="unfinished job; artifacts not opened")); continue
            directory=path.parent/name
            manifest=read_json(directory/"manifest.json")
            if manifest.get("configuration",{}).get("stage")!="eval": continue
            require(manifest.get("complete") is True,"Completed job manifest is incomplete")
            result=audit_run(directory)
            result.update(suite=path.parent.name,job=name,suite_partial=not suite.get("complete",False),
                scope="smoke" if "smoke" in path.parent.name.lower() else "preflight" if "preflight" in path.parent.name.lower() else "formal")
            results.append(result)
    return dict(protocol=PROTOCOL,audited_at=datetime.now(timezone.utc).isoformat(),complete=True,device="cpu",cuda_initialized=torch.cuda.is_initialized(),
        evaluations=results,skipped=skipped,verified_raw_reads=sum(r["verified_raw_reads"] for r in results),
        verified_single_target_updates=sum(r["verified_single_target_updates"] for r in results),
        limitations=["Replays persisted float32 CPU writes and permutations; does not recompute CUDA encoder outputs or model predictions.",
                     "No query vectors/weights are reconstructed. This validates the bank bound to a recorded read, not the model's internal execution.",
                     "Content hashes are integrity checks for these artifacts, not cryptographic proof of trusted experimental execution.",
                     "Only completed evaluation jobs are opened; unfinished confirmation raw files are never inspected."])


def self_test():
    import unittest
    class Checks(unittest.TestCase):
        def setUp(self):
            self.base=PersistentVectorDB(3,2,.05,2)
            self.base.write("one",torch.tensor([.37,.82,1.3]),torch.tensor([1.,2.]),0)
            self.base.write("two",torch.tensor([-.2,.41,.95]),torch.tensor([3.,4.]),1)
            raw=torch.tensor([.12,-.34,.89])
            changed=copy.deepcopy(self.base); changed.write("one",raw,torch.tensor([7.,8.]),2)
            self.patch=dict(write_key=raw,key=changed.keys[0],value=changed.values[0],timestamp=2,bank_hash=changed.hash(),parent_hash=self.base.hash())

        def test_exact_restore_and_single_target_replay(self):
            restored=restore(self.base.snapshot())
            self.assertEqual(restored.hash(),self.base.hash())
            changed=replay_patch(restored,"one",self.patch)
            self.assertEqual(changed.timestamps,(2,1))
            self.assertTrue(torch.equal(changed.keys[1],self.base.keys[1]))
            self.assertEqual(restored.hash(),self.base.hash())

        def test_key_hash_value_and_timestamp_corruption_rejected(self):
            for field,value in (("key",torch.ones(3)),("value",torch.zeros(2)),("write_key",torch.ones(3)),("bank_hash","0"*64),("parent_hash","0"*64),("timestamp",9)):
                with self.subTest(field=field),self.assertRaises(ValueError):
                    replay_patch(self.base,"one",dict(self.patch,**{field:value}))

        def test_shuffle_preserves_exact_addresses_and_changes_payload_hash(self):
            changed=shuffle_bank(self.base,torch.tensor([1,0]))
            self.assertTrue(torch.equal(changed.keys,self.base.keys))
            self.assertTrue(torch.equal(changed.values,self.base.values.flip(0)))
            self.assertEqual(changed.ids,self.base.ids)
            self.assertEqual(changed.timestamps,self.base.timestamps)
            self.assertNotEqual(changed.hash(),self.base.hash())

        def test_invalid_snapshot_is_rejected_without_normalizing(self):
            state=self.base.snapshot(); state["keys"][0]*=2
            with self.assertRaises(ValueError): restore(state)

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    require(not torch.cuda.is_initialized(),"CUDA was unexpectedly initialized")
    if args.self_test: return self_test()
    if args.runs_root is None or args.output is None: parser.error("--runs-root and --output required")
    started=time.perf_counter(); result=audit_completed(args.runs_root)
    require(not torch.cuda.is_initialized(),"Audit unexpectedly initialized CUDA")
    result["elapsed_seconds"]=time.perf_counter()-started
    result["audit_script_sha256"]=file_hash(Path(__file__))
    result["vector_store_source_sha256"]=file_hash(Path(__file__).resolve().parents[1]/"src/vera_mem/vector_store.py")
    args.output.mkdir(parents=True,exist_ok=True)
    destination=args.output/"bank_audit.json"; temporary=destination.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8"); temporary.replace(destination)
    print(json.dumps(dict(complete=True,evaluations=len(result["evaluations"]),reads=result["verified_raw_reads"],updates=result["verified_single_target_updates"],elapsed_seconds=result["elapsed_seconds"],output=str(destination))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
