"""Run independent reconstruction eval jobs with the audited bounded scheduler.

Only entry=reconstruction / --stage eval is accepted. All read-only inputs must exist
before launch. run_interface_eval_suite.run_jobs owns scheduling; only its shared
worker-capacity validation was expanded from two to four (the interface CLI still
accepts one/two). Coldstart accepts one/two/four. The scheduler retains its
live-PID GPU guards (<1000 MiB initially, <60000 MiB on refill), failure draining,
and per-job PIDs/times. This adapter substitutes validation, the Python entry
module and reconstruction suite protocol; it never cancels another or its own process.

--validate-only does not query the GPU or create a suite. Lifecycle tests remain in the shared scheduler; reconstruction validation is
tested separately.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

import run_interface_eval_suite as common
from run_reconstruction_suite import source_hashes
from run_generalization_suite import atomic_json, sha256, utc_now

PROTOCOL = "reconstruction-experiment-v1"
CHILD_PROTOCOL = "reconstruction-experiment-v1"
ENTRY_MODULE = "vera_mem.reconstruction_run"


def validate_plan(plan):
    if not isinstance(plan,list) or not plan:raise ValueError("Nonempty plan required")
    parser=common._ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument("--stage",choices=("eval",),required=True)
    for field in ("model","cache","checkpoint"):parser.add_argument("--"+field,required=True)
    parser.add_argument("--eval-part",choices=("development","confirmation","smoke"),required=True)
    parser.add_argument("--seed",type=int,default=71042)
    translated=[];configs=[];names=set()
    for job in plan:
        if not isinstance(job,dict) or set(job)!={"name","entry","arguments"} or job["entry"]!="reconstruction":raise ValueError("Only reconstruction eval jobs")
        if not isinstance(job["name"],str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",job["name"]) or job["name"] in names:raise ValueError("Duplicate/invalid job name")
        names.add(job["name"])
        args=job["arguments"]
        if not isinstance(args,list) or any(not isinstance(x,str) or not x for x in args):raise ValueError("Invalid arguments")
        flags=[x.split("=",1)[0] for x in args if x.startswith("--")]
        if len(flags)!=len(set(flags)):raise ValueError("Duplicate options")
        cfg=parser.parse_args(args)
        if not 0<=cfg.seed<=2**63-1:raise ValueError("seed must be a nonnegative signed 64-bit integer")
        reduced=["--stage","eval"]
        for field in ("model","cache","checkpoint"):reduced += ["--"+field,getattr(cfg,field)]
        translated.append(dict(name=job["name"],entry="interface",arguments=reduced));configs.append(vars(cfg))
    validated=common.validate_plan(translated)
    for job,original,cfg in zip(validated,plan,configs):job.update(arguments=list(original["arguments"]),entry="reconstruction",configuration=cfg)
    return validated


def job_records(validated,suite):
    return [dict(name=j["name"],
        command=[sys.executable,"-u","-m",ENTRY_MODULE,*j["arguments"],"--run-dir",str(suite/j["name"])],
        output=str(suite/j["name"]),log=str(suite/(j["name"]+".log")),inputs=j["inputs"],
        input_requested_paths=j["input_requested_paths"],configuration=j["configuration"],
        status="pending",exit_code=None,pid=None) for j in validated]


def verify_frozen_inputs(job,manifest,source):
    common._check_inputs(job,manifest)
    for field in ("cache","checkpoint","model_manifest"):
        path = job["inputs"][field]
        if sha256(Path(path)) != manifest["input_files"][path]["sha256"]:
            raise RuntimeError("Read-only input SHA256 changed: "+path)
    if source_hashes(source) != manifest["source_files_sha256"]:
        raise RuntimeError("Frozen source snapshot changed")


def verify_child_result(job,manifest,source):
    verify_frozen_inputs(job,manifest,source)
    path = Path(job["output"])/"manifest.json"
    child = json.loads(path.read_text())
    if (child.get("protocol") != CHILD_PROTOCOL or child.get("stage") != "eval"
            or child.get("complete") is not True or child.get("backbone_unchanged") is not True
            or child.get("configuration",{}).get("stage") != "eval"):
        raise RuntimeError("Zero-exit child lacks a complete matching frozen reconstruction eval manifest")
    for field in ("cache","checkpoint"):
        if child.get(field+"_sha256") != manifest["input_files"][job["inputs"][field]]["sha256"]:
            raise RuntimeError("Child evaluated a different "+field)
    cfg=child["configuration"]
    for field in ("model","cache","checkpoint","seed"):
        if cfg.get(field)!=job["configuration"][field]:raise RuntimeError("Child configuration differs: "+field)
    metrics=child.get("result",{})
    if metrics.get("protocol")!="reconstruction-cpu-vdb-v1" or metrics.get("complete") is not True or metrics.get("shared_parameters_unchanged") is not True:
        raise RuntimeError("Incomplete frozen grouped-CPU-bank evaluation")
    if child["configuration"].get("eval_part")!=job["configuration"]["eval_part"]:raise RuntimeError("Eval part differs")
    job.update(manifest_sha256=sha256(path),source_unchanged=True,inputs_unchanged=True,
               artifact_validation="complete reconstruction eval, input/source hashes, grouped CPU-bank protocol")


class _AuditedChild:
    """Turn an artifact-validation failure into scheduler-visible failure.

    Polling never kills or waits on an unrelated PID. The original return code
    remains recorded when an otherwise successful process fails artifact checks.
    """
    def __init__(self,process,job,audit):
        self.process,self.job,self.audit = process,job,audit
        self.pid,self.finished,self.code = process.pid,False,None

    def poll(self):
        if self.finished: return self.code
        code = self.process.poll()
        if code is None: return None
        self.job["process_exit_code"] = code
        if code == 0:
            try: self.audit()
            except BaseException as error:
                self.job["artifact_error"] = repr(error)
                code = 1
        self.finished,self.code = True,code
        return code


def guarded_spawn(manifest,source,popen=subprocess.Popen):
    children = []
    def spawn(command,**kwargs):
        matches = [job for job in manifest["jobs"] if job["command"] == command]
        if len(matches) != 1: raise RuntimeError("Undeclared or ambiguous child command")
        job = matches[0]
        verify_frozen_inputs(job,manifest,source)
        # A peer may finish between the scheduler's reap and its next launch.
        # Recheck after hash work, so an already observable failure never admits
        # another child merely because the scheduler has not reaped it yet.
        if any((code:=child.poll()) is not None and code != 0 for child in children):
            raise RuntimeError("An owned evaluation failed before the next dispatch")
        process = popen(command,**kwargs)
        child = _AuditedChild(process,job,lambda:verify_child_result(job,manifest,source))
        children.append(child)
        return child
    return spawn


def validate_selection(value, protocol_path):
    """Validate the sealed selector schema, 16-case arithmetic, and fixed ranking.

    The selector records exact rates rather than numerator fields. Recover only
    exact integer counts on its predeclared 16-case grid; do not trust cached
    ``checks``, ``passed``, eligibility, or minimum-seed ranking fields. Artifact
    bindings are checked structurally here; the selector/auditor validates their
    source bytes before sealing. Remote run paths need not exist on this host.
    """
    from summarize_reconstruction import SELECTION_PROTOCOL,GATE_POLICY,ARMS,SEEDS

    def require(ok, message):
        if not ok:raise ValueError(message)

    def sha(value):
        return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None

    def count16(rate, label):
        require(type(rate) in (int,float) and math.isfinite(rate) and 0<=rate<=1,
                "Invalid selection rate: "+label)
        scaled=rate*16
        require(scaled==int(scaled),"Selection rate is not an exact 16-case count: "+label)
        return int(scaled)

    require(isinstance(value,dict) and value.get("protocol")==SELECTION_PROTOCOL
            and value.get("sealed") is True,"Expected a sealed reconstruction selection")
    require(value.get("protocol_sha256")==sha256(protocol_path),"Selection protocol document differs")
    require(sha(value.get("script_sha256")),"Missing selection script hash")
    require(isinstance(value.get("created_at"),str) and bool(value["created_at"].strip()),
            "Missing selection creation time")
    decision=value.get("decision")
    require(isinstance(decision,dict),"Missing selection decision")
    # JSON encoding distinguishes bools from numbers, unlike Python dict ==.
    policy=decision.get("policy")
    require(isinstance(policy,dict),"Selection policy differs")
    try:policy_equal=json.dumps(policy,sort_keys=True,allow_nan=False)==json.dumps(GATE_POLICY,sort_keys=True,allow_nan=False)
    except (TypeError,ValueError):policy_equal=False
    require(policy_equal,"Selection policy differs")
    barrier=decision.get("development_barrier")
    require(isinstance(barrier,dict) and type(barrier.get("expected")) is int
            and type(barrier.get("observed")) is int and barrier["expected"]==barrier["observed"]==36
            and barrier.get("complete") is True and barrier.get("missing")==[]
            and barrier.get("duplicates")==[],"Development barrier is incomplete")

    candidates=decision.get("candidates")
    require(isinstance(candidates,list) and len(candidates)==4
            and all(isinstance(c,dict) and isinstance(c.get("arm"),str) for c in candidates),
            "Missing architecture candidates")
    require({c["arm"] for c in candidates}==set(ARMS),"Missing architecture candidates")
    for c in candidates:
        seeds=c.get("per_seed")
        require(isinstance(seeds,list) and len(seeds)==3
                and all(isinstance(s,dict) and type(s.get("seed")) is int for s in seeds),
                "Missing seed decisions")
        require({s["seed"] for s in seeds}==set(SEEDS),"Missing seed decisions")
        recomputed=[];c_counts=[]
        for s in seeds:
            counts={k:count16(s.get(k),k) for k in
                    ("ab_pair","c_update_restore","c_real","c_shuffle","c_empty","c_locality_joint")}
            require(counts["c_update_restore"]<=counts["c_real"],"Impossible update/restore joint count")
            checks=dict(ab_pair=counts["ab_pair"]>=15,
                c_update_restore=counts["c_update_restore"]>=15,
                c_real_minus_shuffle=counts["c_real"]-counts["c_shuffle"]>=8,
                c_real_minus_empty=counts["c_real"]-counts["c_empty"]>=8,
                c_locality_joint=counts["c_locality_joint"]/16>=GATE_POLICY["c_locality_joint_min"])
            saved=s.get("checks")
            require(isinstance(saved,dict) and set(saved)==set(checks)
                    and all(type(saved[k]) is bool and saved[k]==v for k,v in checks.items()),
                    "Seed gate checks differ from 16-case counts")
            passed=all(checks.values())
            require(type(s.get("passed")) is bool and s["passed"]==passed,"Seed eligibility differs from counts")
            recomputed.append(passed);c_counts.append(counts["c_update_restore"])
        require(type(c.get("eligible")) is bool and c["eligible"]==all(recomputed),
                "Candidate eligibility inconsistent")
        require(count16(c.get("minimum_seed_c_update_restore"),"minimum_seed_c_update_restore")==min(c_counts),
                "Minimum-seed ranking statistic differs")
        for field in ("trainable_parameters","bytes_per_fact"):
            require(type(c.get(field)) is int and c[field]>0,"Invalid candidate "+field)
    eligible=sorted([c for c in candidates if c["eligible"]],
        key=lambda c:(-c["minimum_seed_c_update_restore"],c["trainable_parameters"],c["bytes_per_fact"],ARMS.index(c["arm"])))
    require(decision.get("selected_arm")== (eligible[0]["arm"] if eligible else None),
            "Selection differs from fixed ranking")

    artifact_types={
        frozenset(("manifest.json","training_status.json","training.jsonl","initial.pt","last.pt","step_1024.pt")):"train",
        frozenset(("manifest.json","summary.json","predictions.jsonl","pairs.json","encoded_payloads.pt","bank_events.pt")):"eval",
        frozenset(("manifest.json","summary.json","predictions.jsonl")):"teacher",
    }
    evidence=decision.get("evidence")
    require(isinstance(evidence,list) and len(evidence)==37,
            "Selection must bind 36 development conditions and the teacher preflight")
    seen=set();kinds=dict(train=0,eval=0,teacher=0)
    for e in evidence:
        require(isinstance(e,dict) and isinstance(e.get("run_dir"),str)
                and bool(e["run_dir"]) and Path(e["run_dir"]).is_absolute()
                and ".." not in Path(e["run_dir"]).parts and isinstance(e.get("artifacts"),dict),
                "Missing selection artifact bindings")
        normalized=str(Path(e["run_dir"]))
        require(normalized not in seen,"Duplicate selection evidence run_dir");seen.add(normalized)
        names=frozenset(e["artifacts"])
        require(names in artifact_types,"Selection artifact filenames differ")
        require(all(sha(h) for h in e["artifacts"].values()),"Invalid selection artifact hash")
        kinds[artifact_types[names]]+=1
    require(kinds==dict(train=12,eval=24,teacher=1),"Selection evidence condition counts differ")
    return value


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--shared-root",type=Path,required=True)
    p.add_argument("--gpu",required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--workers",type=int,choices=(1,2,4),default=2)
    p.add_argument("--poll-seconds",type=float,default=1.)
    p.add_argument("--validate-only",action="store_true")
    p.add_argument("--selection-evidence",type=Path)
    a = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",a.name): p.error("Use one new suite directory name")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",a.gpu):
        p.error("Select one complete GPU UUID")
    if not 0 < a.poll_seconds <= 10: p.error("--poll-seconds must lie in (0,10]")
    w,shared = a.workspace.resolve(),a.shared_root.resolve()
    repository,suite = w/"VeRA-Mem",w/"runs"/a.name
    try:
        plan = json.loads(a.plan.read_text()); validated = validate_plan(plan)
        if any(j["configuration"]["eval_part"]=="confirmation" for j in validated) and a.selection_evidence is None:
            raise ValueError("Confirmation requires sealed development selection evidence")
        if a.selection_evidence is not None:
            selection_bytes=a.selection_evidence.read_bytes();validate_selection(json.loads(selection_bytes),repository/"docs/reconstruction_protocol.md")
        if suite.exists(): raise FileExistsError("Suite already exists: "+str(suite))
        for folder in ("src","scripts","tests","configs","docs"):
            if not (repository/folder).is_dir(): raise ValueError("Repository directory missing: "+folder)
        for name in ("README.md","pyproject.toml"):
            if not (repository/name).is_file(): raise ValueError("Repository file missing: "+name)
        if not (shared/"env_deps").is_dir(): raise ValueError("Shared dependency directory missing")
        if shutil.disk_usage(w).free < 5*1024**3: raise ValueError("At least 5 GiB free workspace space is required")
        inputs = common.input_metadata(validated)
    except (OSError,ValueError) as error: p.error(str(error))
    if a.validate_only:
        print(json.dumps(dict(valid=True,protocol=PROTOCOL,entry_module=ENTRY_MODULE,
            read_only_evaluation_jobs=len(validated),workers=a.workers,plan_sha256=sha256(a.plan),inputs=inputs)),flush=True)
        return 0
    suite.mkdir(parents=True,exist_ok=False)
    source = suite/"source"
    manifest = dict(protocol=PROTOCOL,complete=False,status="preparing",started_at=utc_now(),
        launcher_pid=os.getpid(),gpu_uuid=a.gpu,workers=a.workers,
        execution_mode="parallel_independent_read_only_evaluations",entry_module=ENTRY_MODULE,
        scheduler="run_interface_eval_suite.run_jobs (only shared worker-capacity validation expanded to <=4; scheduling/occupancy/draining unchanged; interface CLI still 1/2)",
        plan_sha256=sha256(a.plan),input_files=inputs,jobs=[])
    persist = lambda:atomic_json(suite/"suite.json",manifest)
    persist()
    requested_stop,previous_handlers = {"reason":None},{}
    try:
        for folder in ("src","scripts","tests","configs","docs"):
            shutil.copytree(repository/folder,source/folder,ignore=shutil.ignore_patterns("__pycache__","*.pyc",".pytest_cache"))
        for name in ("README.md","pyproject.toml"): shutil.copy2(repository/name,source/name)
        atomic_json(suite/"plan.json",plan)
        if a.selection_evidence is not None:
            (suite/"selection_evidence.json").write_bytes(selection_bytes)
            manifest["selection_evidence_sha256"]=sha256(suite/"selection_evidence.json")
            manifest["selection_evidence_source"]=str(a.selection_evidence.resolve())
        manifest["source_files_sha256"] = source_hashes(source)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false",HF_HOME=str(w/"cache/huggingface"),
            PYTHONPATH=str(shared/"env_deps")+os.pathsep+str(source/"src"))
        manifest["runtime_environment"] = {key:env[key] for key in (
            "CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
            "TOKENIZERS_PARALLELISM","HF_HOME","PYTHONPATH")}
        manifest["jobs"] = job_records(validated,suite)
        persist()
        def request_stop(signum,_frame):
            requested_stop["reason"] = f"Received signal {signum}; draining owned evaluations without termination"
        for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum,request_stop)
        code = common.run_jobs(manifest,persist,a.gpu,env,source,workers=a.workers,
            poll_seconds=a.poll_seconds,popen=guarded_spawn(manifest,source),
            stop_requested=lambda:requested_stop["reason"])
        if code == 0:
            manifest.update(source_unchanged=True,inputs_unchanged=True)
            persist()
        return code
    except BaseException as error:
        # Child exceptions are drained inside the unchanged scheduler.
        manifest.update(status="failed",complete=False,error=repr(error),failed_at=utc_now())
        persist(); raise
    finally:
        for signum,previous in previous_handlers.items(): signal.signal(signum,previous)


if __name__ == "__main__": raise SystemExit(main())
