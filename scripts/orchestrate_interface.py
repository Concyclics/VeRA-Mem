"""Complete the preregistered four-step study, recording every dependent launch.

Run locally after source review and the initial writer suites are started.
This coordinator never cancels a job and never restarts a failed suite. An
external GPU occupant triggers the suite's own refusal rather than eviction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from interface_transport import ssh_command, rsync_transport, control_path

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
REMOTE="/ssd3/chenhan/VeRA-Mem-Workspace"
TAG="20261006"
GPUS=("0","1","3")


def now(): return datetime.now(timezone.utc).isoformat()


def run(arguments, **kwargs):
    return subprocess.run(arguments,cwd=ROOT,check=True,text=True,**kwargs)


def states(names):
    source="""import json,pathlib
r=pathlib.Path(ROOT)/'runs'
out={}
def is_alive(pid):
 proc=pathlib.Path('/proc')/str(pid)/'stat'
 try: return proc.read_text().rsplit(')',1)[1].split()[0]!='Z'
 except FileNotFoundError: return False
for name in NAMES:
 p=r/name/'suite.json'
 if not p.exists():
  launch=r/(name+'_launch.json')
  if not launch.exists(): out[name]={'status':'absent'}; continue
  pid=json.loads(launch.read_text())['pid']
  alive=is_alive(pid)
  log=r/(name+'.log')
  out[name]={'status':'starting' if alive else 'failed','launcher_pid':pid,
             'error':None if alive else (log.read_text()[-2000:] if log.exists() else 'Launcher exited before creating suite manifest')}
  continue
 d=json.loads(p.read_text())
 out[name]={k:d.get(k) for k in ('complete','status','error','launcher_pid','gpu_uuid','plan_sha256')}
 out[name]['jobs']=[{k:j.get(k) for k in ('name','status','exit_code','elapsed_seconds')} for j in d.get('jobs',[])]
 if not d.get('complete') and d.get('status') in ('preparing','running','draining') and not is_alive(d.get('launcher_pid')):
  out[name].update(status='failed',error='Suite launcher exited with an incomplete manifest; child jobs were not cancelled')
print(json.dumps(out))
""".replace("ROOT",repr(REMOTE)).replace("NAMES",repr(list(names)))
    for attempt in range(3):
        try:
            return json.loads(run([*ssh_command(),"xtrah100","python3","-"],input=source,capture_output=True).stdout)
        except (subprocess.CalledProcessError,json.JSONDecodeError):
            if attempt==2: raise
            time.sleep(2*(attempt+1))


def sync_source():
    run(["rsync","-az","-e",rsync_transport(),"--exclude=.git","--exclude=__pycache__","--exclude=.pytest_cache",
         str(ROOT)+"/","xtrah100:"+REMOTE+"/VeRA-Mem/"])


def sync_results(verify=False):
    run([sys.executable,"scripts/sync_backup.py","--host","xtrah100","--remote-root",REMOTE,
         "--local-root",str(WORKSPACE),"--ssh-control-path",str(control_path())]+(["--verify"] if verify else []))
    manifest=json.loads((WORKSPACE/"backup_manifest.json").read_text())
    if not manifest.get("complete") or (verify and (not manifest.get("checksum_verified")
            or any(r.get("checksum_differences") for r in manifest.get("records",[])))):
        raise RuntimeError("Backup did not satisfy the requested completeness/checksum checks")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds",type=int,default=30)
    parser.add_argument("--resume",action="store_true",help="Adopt only matching existing suites after an inspected coordinator failure")
    args=parser.parse_args()
    if not 10<=args.poll_seconds<=60: parser.error("Poll interval must be 10–60 seconds")
    path=WORKSPACE/"plans"/f"interface_orchestration_{TAG}.json"
    if path.exists():
        if not args.resume: raise FileExistsError("Coordinator already recorded; inspect it instead of duplicating jobs")
        state=json.loads(path.read_text())
        if state["status"]!="failed": raise ValueError("Only an inspected failed coordinator can be resumed")
        state.update(status="running",resumed_at=now());state.pop("error",None)
    else:
        if args.resume: raise FileNotFoundError("No coordinator state to resume")
        state=dict(protocol="interface-orchestration-v1",status="running",started_at=now(),events=[],observations={})
    def save():
        state["updated_at"]=now()
        temporary=path.with_suffix(".tmp");temporary.write_text(json.dumps(state,indent=2)+"\n");temporary.replace(path)
    def event(action,**extra):
        record=dict(at=now(),action=action,**extra);state["events"].append(record);save();print(json.dumps(record),flush=True)
    def check(names):
        observed=states(names);state["observations"].update(observed);save()
        failed={n:d for n,d in observed.items() if d.get("status")=="failed"}
        if failed: raise RuntimeError("A suite failed; no automatic retry: "+json.dumps(failed))
        return observed
    def wait(names):
        previous=None
        while True:
            observed=check(names)
            compact={n:[(j["name"],j["status"]) for j in d.get("jobs",[])] for n,d in observed.items()}
            if compact!=previous: event("progress",suites=compact);previous=compact
            if all(d.get("complete") for d in observed.values()): return
            time.sleep(args.poll_seconds)
    def launch(name,gpu,evaluation=False):
        # No duplicate, including a failed/partial suite: inspection is required.
        existing=states([name])[name]
        if existing["status"]!="absent":
            expected_gpu={"0":"GPU-2aea6294-c3e6-5b4a-65e8-c5ce6985fe39",
                          "1":"GPU-d128ba8f-52fc-77f3-76c4-c238e94e40df",
                          "3":"GPU-b39c5755-644d-a3b3-8e76-75beb4142100"}[gpu]
            expected_plan=hashlib.sha256((WORKSPACE/"plans"/(name+".json")).read_bytes()).hexdigest()
            if (args.resume and existing["status"] in ("running","complete")
                    and existing.get("plan_sha256")==expected_plan and existing.get("gpu_uuid")==expected_gpu):
                event("adopt_existing_suite",suite=name,gpu=gpu,record=existing);return
            raise FileExistsError("Existing suite cannot be safely adopted: "+name)
        command=[sys.executable,"scripts/launch_interface.py",name,gpu]
        if evaluation: command += ["--eval-workers","2"]
        result=run(command,capture_output=True)
        event("launch",suite=name,gpu=gpu,evaluation=evaluation,record=json.loads(result.stdout))
    base=["--runs-root",str(WORKSPACE/"runs/xtrah100"),"--plans-root",str(WORKSPACE/"plans"),"--queues","3"]
    try:
        save();event("await_writer_development")
        wait([f"interface_writer_{s}_{TAG}" for s in ("a","b")])
        sync_results()
        run([sys.executable,"scripts/plan_interface_followup.py",*base])
        sync_source()
        step2=[f"interface_step2_confirm_{s}_{TAG}" for s in ("a","b","c")]
        step3=[f"interface_step3_train_{s}_{TAG}" for s in ("a","b","c")]
        for name,gpu in zip(step2,GPUS): launch(name,gpu,True)
        started=set()
        while len(started)<3:
            observed=check(step2)
            for i,(name,gpu) in enumerate(zip(step3,GPUS)):
                if i not in started and observed[step2[i]].get("complete"):
                    launch(name,gpu);started.add(i)
            if len(started)<3: time.sleep(args.poll_seconds)
        wait(step3)
        sync_results()
        run([sys.executable,"scripts/plan_interface_kd.py",*base,"--phase","train"])
        step4=[f"interface_step4_train_{s}_{TAG}" for s in ("a","b","c")]
        for name,gpu in zip(step4,GPUS): launch(name,gpu)
        wait(step4)
        sync_results()
        run([sys.executable,"scripts/plan_interface_followup.py",*base,"--phase","confirm"])
        run([sys.executable,"scripts/plan_interface_kd.py",*base,"--phase","confirm"])
        confirm3=[f"interface_step3_confirm_{s}_{TAG}" for s in ("a","b","c")]
        confirm4=[f"interface_step4_confirm_{s}_{TAG}" for s in ("a","b","c")]
        for name,gpu in zip(confirm3,GPUS): launch(name,gpu,True)
        started=set()
        while len(started)<3:
            observed=check(confirm3)
            for i,(name,gpu) in enumerate(zip(confirm4,GPUS)):
                if i not in started and observed[confirm3[i]].get("complete"):
                    launch(name,gpu,True);started.add(i)
            if len(started)<3: time.sleep(args.poll_seconds)
        wait(confirm4)
        sync_results(verify=True)
        event("all_experiments_complete")
        state.update(status="complete",completed_at=now());save()
    except BaseException as error:
        state.update(status="failed",error=repr(error),error_stderr=getattr(error,"stderr",None));save()
        print(json.dumps(dict(coordinator_failed=repr(error),running_jobs_not_cancelled=True)),flush=True)
        raise


if __name__=="__main__":main()
