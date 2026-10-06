"""Run independent coldstart eval jobs with the audited bounded scheduler.

Only entry=coldstart / --stage eval is accepted. All read-only inputs must exist
before launch. run_interface_eval_suite.run_jobs owns scheduling; only its shared
worker-capacity validation was expanded from two to four (the interface CLI still
accepts one/two). Coldstart accepts one/two/four. The scheduler retains its
live-PID GPU guards (<1000 MiB initially, <60000 MiB on refill), failure draining,
and per-job PIDs/times. This adapter substitutes validation, the Python entry
module and coldstart suite protocol; it never cancels another or its own process.

--validate-only does not query the GPU or create a suite. --self-test exercises
the original offline lifecycle suite plus only the coldstart-specific changes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

import run_interface_eval_suite as common
from run_coldstart_suite import source_hashes
from run_generalization_suite import atomic_json, sha256, utc_now

PROTOCOL = "coldstart-experiment-v1"
CHILD_PROTOCOL = "dictionary-coldstart-v1"
ENTRY_MODULE = "vera_mem.coldstart_run"


def validate_plan(plan):
    if not isinstance(plan,list) or not plan:
        raise ValueError("Plan must be a nonempty list")
    p = common._ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument("--stage",choices=("eval",),required=True)
    for field in ("model","cache","checkpoint"): p.add_argument("--"+field,required=True)
    p.add_argument("--arm",default="unspecified")
    p.add_argument("--domain",choices=("wikipedia","matched","synthetic","synthetic_train_probe"),default="wikipedia")
    p.add_argument("--base-mode",choices=("full","off","random256","cluster256"),default="full")
    p.add_argument("--merge-count",type=int)
    p.add_argument("--teacher",action="store_true")
    p.add_argument("--max-cases",type=int,default=64)
    p.add_argument("--seed",type=int,default=63042)
    names,translated,configs = set(),[],[]
    for job in plan:
        if not isinstance(job,dict) or set(job)!={"name","entry","arguments"}:
            raise ValueError("Job requires exactly name/entry/arguments")
        name = job["name"]
        if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",name) or name in names:
            raise ValueError("Each job needs a unique safe single-component name")
        names.add(name)
        if job["entry"] != "coldstart": raise ValueError("Only entry=coldstart is accepted")
        args = job["arguments"]
        if not isinstance(args,list) or not args or any(not isinstance(x,str) or not x for x in args):
            raise ValueError("Arguments must be a nonempty string list")
        seen = set()
        for token in args:
            if token.startswith("--"):
                option = token.split("=",1)[0]
                if option in seen: raise ValueError("Duplicate option: "+option)
                seen.add(option)
        cfg = p.parse_args(args)  # --run-dir and all training switches are unknown.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",cfg.arm): raise ValueError("Invalid arm label")
        if cfg.max_cases < 1: raise ValueError("--max-cases must be positive")
        if not 0 <= cfg.seed <= 2**63-1: raise ValueError("--seed must be a nonnegative signed 64-bit integer")
        if cfg.merge_count is not None:
            if cfg.merge_count < 1: raise ValueError("--merge-count must be positive")
            if cfg.base_mode not in ("random256","cluster256"):
                raise ValueError("--merge-count applies only to compressed foundation banks")
        # Reuse the original absolute/existing-path and model revision-manifest
        # validation without mutating its globals or permitting its extra args.
        reduced = ["--stage","eval"]
        for field in ("model","cache","checkpoint"): reduced += ["--"+field,getattr(cfg,field)]
        translated.append(dict(name=name,entry="interface",arguments=reduced))
        configs.append(vars(cfg))
    validated = common.validate_plan(translated)
    for job,original,cfg in zip(validated,plan,configs):
        job.update(arguments=list(original["arguments"]),entry="coldstart",configuration=cfg)
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
        raise RuntimeError("Zero-exit child lacks a complete matching frozen coldstart eval manifest")
    for field in ("cache","checkpoint"):
        if child.get(field+"_sha256") != manifest["input_files"][job["inputs"][field]]["sha256"]:
            raise RuntimeError("Child evaluated a different "+field)
    metrics = child.get("result",{})
    if metrics.get("protocol") != "interface-cpu-vdb-eval-v1" or metrics.get("complete") is not True:
        raise RuntimeError("Child evaluation metrics incomplete")
    c = metrics.get("coldstart",{})
    if c.get("protocol") != "coldstart-cpu-banks-v1" or not c.get("foundation_cpu") or not c.get("episodic_cpu"):
        raise RuntimeError("Child lacks the CPU two-bank evaluation contract")
    if c.get("base_mode") != job["configuration"]["base_mode"]:
        raise RuntimeError("Child foundation condition differs from plan")
    job.update(manifest_sha256=sha256(path),source_unchanged=True,inputs_unchanged=True,
               artifact_validation="complete coldstart eval manifest, input/source hashes, CPU-bank protocol")


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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--self-test"]: return self_test()
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--shared-root",type=Path,required=True)
    p.add_argument("--gpu",required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--workers",type=int,choices=(1,2,4),default=2)
    p.add_argument("--poll-seconds",type=float,default=1.)
    p.add_argument("--validate-only",action="store_true")
    a = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",a.name): p.error("Use one new suite directory name")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",a.gpu):
        p.error("Select one complete GPU UUID")
    if not 0 < a.poll_seconds <= 10: p.error("--poll-seconds must lie in (0,10]")
    w,shared = a.workspace.resolve(),a.shared_root.resolve()
    repository,suite = w/"VeRA-Mem",w/"runs"/a.name
    try:
        plan = json.loads(a.plan.read_text()); validated = validate_plan(plan)
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


def self_test():
    import tempfile
    import unittest
    from types import SimpleNamespace

    class Tests(unittest.TestCase):
        def setUp(self):
            temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
            self.path = Path(temporary.name)
            model = self.path/"models/model"; model.mkdir(parents=True)
            (model.parent/"manifest.json").write_text("{}")
            self.cache,self.checkpoint = self.path/"cache.pt",self.path/"checkpoint.pt"
            self.cache.write_bytes(b"cache"); self.checkpoint.write_bytes(b"checkpoint")
            self.arguments = ["--stage","eval","--model",str(model),"--cache",str(self.cache),
                "--checkpoint",str(self.checkpoint),"--arm","straight_through","--domain","synthetic",
                "--base-mode","full","--max-cases","64","--seed","63042"]
        def plan(self,args=None): return [dict(name="eval",entry="coldstart",arguments=args or self.arguments)]
        def fixture(self,count=1):
            plan = [dict(name="job"+str(i),entry="coldstart",arguments=self.arguments) for i in range(count)]
            validated = validate_plan(plan)
            source = self.path/"source"; source.mkdir(); (source/"file.py").write_text("fixed")
            manifest = dict(jobs=job_records(validated,self.path),input_files=common.input_metadata(validated),source_files_sha256=source_hashes(source))
            return manifest,source
        def write_result(self,job,manifest,complete=True):
            path = Path(job["output"]); path.mkdir()
            child = dict(protocol=CHILD_PROTOCOL,stage="eval",complete=complete,backbone_unchanged=True,
                configuration={"stage":"eval"},
                result=dict(protocol="interface-cpu-vdb-eval-v1",complete=True,
                    coldstart=dict(protocol="coldstart-cpu-banks-v1",foundation_cpu=True,episodic_cpu=True,base_mode="full")))
            for field in ("cache","checkpoint"): child[field+"_sha256"] = manifest["input_files"][job["inputs"][field]]["sha256"]
            (path/"manifest.json").write_text(json.dumps(child))
        def test_accepts_exact_formal_compression_and_probe_metadata(self):
            for mode in ("full","off","random256","cluster256"):
                for domain in ("wikipedia","synthetic","synthetic_train_probe"):
                    args = list(self.arguments)
                    args[args.index("--base-mode")+1] = mode
                    args[args.index("--domain")+1] = domain
                    if mode.endswith("256"): args += ["--merge-count","256"]
                    args += ["--teacher"]
                    validated = validate_plan(self.plan(args))
                    self.assertEqual(validated[0]["arguments"],args)
                    self.assertEqual(validated[0]["entry"],"coldstart")
                    self.assertEqual(job_records(validated,self.path)[0]["command"][1:4],["-u","-m",ENTRY_MODULE])
        def test_rejects_training_and_unknown_or_invalid_eval_options(self):
            bad = [["--updates","512"],["--run-dir","/tmp/x"],["--alpha-base","0"],["--routing","sparse"],
                   ["--resume-optimizer"],["--stage","train"],["--teacher","--teacher"]]
            for extra in bad:
                with self.assertRaises(ValueError): validate_plan(self.plan(self.arguments+extra))
            for key,value in (("--stage","train"),("--base-mode","dense"),("--seed","-1"),("--max-cases","0"),("--domain","unknown")):
                args = list(self.arguments); args[args.index(key)+1] = value
                with self.assertRaises(ValueError): validate_plan(self.plan(args))
            with self.assertRaises(ValueError): validate_plan(self.plan(self.arguments+["--merge-count","256"]))
            with self.assertRaises(ValueError): validate_plan([dict(self.plan()[0],entry="interface")])
        def test_missing_input_and_source_or_content_mutation_prevent_spawn(self):
            manifest,source = self.fixture()
            spawn = guarded_spawn(manifest,source,lambda *a,**k:self.fail("unexpected process"))
            self.checkpoint.write_bytes(b"mutated")
            with self.assertRaises(RuntimeError): spawn(manifest["jobs"][0]["command"])
            self.checkpoint.write_bytes(b"checkpoint")
            manifest["input_files"] = common.input_metadata(validate_plan(self.plan()))
            (source/"file.py").write_text("changed")
            with self.assertRaisesRegex(RuntimeError,"source"): spawn(manifest["jobs"][0]["command"])
            self.cache.unlink()
            with self.assertRaises(FileNotFoundError): validate_plan(self.plan())
        def test_success_validates_child_manifest_and_keeps_real_exit_code(self):
            manifest,source = self.fixture(); job = manifest["jobs"][0]
            self.write_result(job,manifest)
            child = guarded_spawn(manifest,source,lambda *a,**k:SimpleNamespace(pid=123,poll=lambda:0))(job["command"])
            self.assertEqual(child.poll(),0); self.assertEqual(child.poll(),0)
            self.assertEqual(job["process_exit_code"],0)
            self.assertTrue(job["source_unchanged"] and job["inputs_unchanged"])
            self.assertEqual(job["manifest_sha256"],sha256(Path(job["output"])/"manifest.json"))
        def test_bad_zero_exit_artifact_stops_dispatch_and_drains_live_peer(self):
            manifest,source = self.fixture(3)
            clock,children = [0.],[]
            def popen(command,**kwargs):
                i = len(children); job = manifest["jobs"][i]
                self.write_result(job,manifest,complete=i!=0)
                end = clock[0]+(1 if i==0 else 3)
                child = SimpleNamespace(pid=100+i,poll=lambda:0 if clock[0]>=end else None)
                children.append(child); return child
            def probe(gpu):
                pids = [c.pid for c in children if c.poll() is None]
                return dict(memory_used_mib=9000*len(pids),compute_pids=pids)
            code = common.run_jobs(manifest,lambda:None,"GPU-test",{},source,workers=2,
                probe=probe,popen=guarded_spawn(manifest,source,popen),
                sleep=lambda sec:clock.__setitem__(0,clock[0]+sec),monotonic=lambda:clock[0])
            self.assertEqual(code,1); self.assertEqual(len(children),2); self.assertEqual(clock[0],3)
            self.assertEqual([j["status"] for j in manifest["jobs"]],["failed","complete","not_started"])
            self.assertEqual(manifest["jobs"][0]["process_exit_code"],0)
            self.assertIn("artifact_error",manifest["jobs"][0])
        def test_validate_only_uses_coldstart_schema_and_never_creates_suite(self):
            from contextlib import redirect_stdout
            import io
            repo = self.path/"VeRA-Mem"
            for folder in ("src","scripts","tests","configs","docs"): (repo/folder).mkdir(parents=True)
            for name in ("README.md","pyproject.toml"): (repo/name).write_text(name)
            (self.path/"env_deps").mkdir()
            planpath = self.path/"plan.json"; planpath.write_text(json.dumps(self.plan()))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["--workspace",str(self.path),"--shared-root",str(self.path),
                    "--gpu","GPU-00000000-0000-0000-0000-000000000000","--name","coldstart_offline",
                    "--plan",str(planpath),"--workers","2","--validate-only"])
            result = json.loads(stdout.getvalue())
            self.assertEqual(code,0); self.assertEqual(result["protocol"],PROTOCOL)
            self.assertEqual(result["entry_module"],ENTRY_MODULE); self.assertEqual(result["workers"],2)
            self.assertFalse((self.path/"runs/coldstart_offline").exists())
        def test_four_workers_refill_fifth_and_keep_foreign_pid_guard(self):
            for foreign in (False,True):
                if foreign:
                    for path in self.path.glob("job*"):
                        if path.is_dir():shutil.rmtree(path)
                        else:path.unlink()
                    shutil.rmtree(self.path/"source")
                manifest,source=self.fixture(5)
                clock,children,peak=[0.],[],[0]
                durations=(1,2,3,4,1)
                def popen(command,**kwargs):
                    i=len(children);self.write_result(manifest["jobs"][i],manifest)
                    end=clock[0]+durations[i]
                    child=SimpleNamespace(pid=100+i,poll=lambda:0 if clock[0]>=end else None)
                    children.append(child);peak[0]=max(peak[0],sum(c.poll() is None for c in children))
                    return child
                def probe(gpu):
                    pids=[c.pid for c in children if c.poll() is None]
                    if foreign and len(children)==4 and clock[0]>=1:pids.append(999)
                    return dict(memory_used_mib=9000*len(pids),compute_pids=pids)
                code=common.run_jobs(manifest,lambda:None,"GPU-test",{},source,workers=4,
                    probe=probe,popen=guarded_spawn(manifest,source,popen),
                    sleep=lambda sec:clock.__setitem__(0,clock[0]+sec),monotonic=lambda:clock[0])
                self.assertEqual(peak[0],4);self.assertEqual(clock[0],4)
                self.assertEqual(code,1 if foreign else 0)
                self.assertEqual(len(children),4 if foreign else 5)
                self.assertEqual(manifest["jobs"][-1]["status"],"not_started" if foreign else "complete")
                self.assertTrue(all(child.poll()==0 for child in children))

    if common.self_test(): return 1
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__": raise SystemExit(main())
