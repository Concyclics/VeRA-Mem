"""Launch an already synchronized plan on the authorized H100 workspace."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from interface_transport import ssh_command, rsync_transport

p=argparse.ArgumentParser(); p.add_argument("name"); p.add_argument("gpu",choices=("0","1","3"))
p.add_argument("--eval-workers",type=int,choices=(1,2,4),default=1)
p.add_argument("--selection-evidence",type=Path)

a=p.parse_args()
if not re.fullmatch(r"qkv_[a-z0-9_]+",a.name): p.error("Use a qkv experiment name")
w="/ssd3/chenhan/VeRA-Mem-Workspace"
uuid={"0":"GPU-2aea6294-c3e6-5b4a-65e8-c5ce6985fe39","1":"GPU-d128ba8f-52fc-77f3-76c4-c238e94e40df","3":"GPU-b39c5755-644d-a3b3-8e76-75beb4142100"}[a.gpu]
plan=Path(__file__).resolve().parents[2]/"plans"/(a.name+".json")
subprocess.run(["rsync","-az","-e",rsync_transport(),str(plan),"xtrah100:"+w+"/plans/"],check=True)
if a.selection_evidence is not None:
    subprocess.run(["rsync","-az","-e",rsync_transport(),str(a.selection_evidence),"xtrah100:"+w+"/plans/"],check=True)
remote="""import json,subprocess
from pathlib import Path
w=Path(ROOT); name=NAME
command=['python3',str(w/'VeRA-Mem/scripts'/RUNNER),'--workspace',str(w),'--shared-root',str(w),'--gpu',GPU,'--name',name,'--plan',str(w/'plans'/(name+'.json'))]+EXTRA
with (w/'runs'/(name+'.log')).open('x') as log:
 p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,command=command)
(w/'runs'/(name+'_launch.json')).write_text(json.dumps(r,indent=2)+'\\n')
print(json.dumps(r))
""".replace("ROOT",repr(w)).replace("NAME",repr(a.name)).replace("GPU",repr(uuid))
remote=remote.replace("RUNNER",repr("run_qkv_eval_suite.py" if a.eval_workers>1 else "run_qkv_suite.py"))
extra=["--workers",str(a.eval_workers)] if a.eval_workers>1 else []
if a.selection_evidence is not None:extra += ["--selection-evidence",w+"/plans/"+a.selection_evidence.name]
remote=remote.replace("EXTRA",repr(extra))
result=subprocess.run([*ssh_command(),"xtrah100","python3","-"],input=remote,text=True,capture_output=True)
if result.returncode:
    raise RuntimeError("SSH launch failed; inspect remote launch record before retrying: "+result.stderr[-2000:])
record=json.loads(result.stdout); plan.with_name(a.name+"_launch.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record))
