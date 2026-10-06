"""Dispatch the fixed remaining lane and supplementary evaluations after dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import time
from interface_transport import ssh_command
from orchestrate_coldstart_evaluation import REMOTE,ROOT

def main():
    log=ROOT.parent/'plans/coldstart_tail_controller_20261006.json'
    if log.exists():raise FileExistsError(log)
    state=dict(complete=False,launches=[])
    children=[]
    dispatched=set()
    def save():log.write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        while True:
            response=subprocess.run(ssh_command('xtrah100')+['python3','-'],input=REMOTE,text=True,capture_output=True,check=True)
            suites=json.loads(response.stdout)
            relevant=['coldstart_teacher_repair_20261006','coldstart_eval_confirm_a_20261006','coldstart_eval_confirm_b_20261006']
            relevant += list(dispatched-{'lane_c'})
            for name in relevant:
                if suites.get(name,{}).get('status')=='failed':raise RuntimeError('Failed dependency '+name)
            for process in children:
                if process.poll() is not None and process.returncode:raise RuntimeError('Lane C controller failed')
            done=lambda name:suites.get(name,{}).get('complete') is True
            teacher_done=done('coldstart_teacher_repair_20261006')
            if teacher_done and 'lane_c' not in dispatched:
                command=[sys.executable,str(ROOT/'scripts/orchestrate_coldstart_evaluation.py'),'c']
                child=subprocess.Popen(command);children.append(child)
                state['launches'].append(dict(name='lane_c',pid=child.pid,command=command))
                dispatched.add('lane_c');save();print('LAUNCH lane_c',flush=True)
            jobs=[('dev','0','coldstart_eval_confirm_a_20261006'),
                  ('probe','1','coldstart_eval_confirm_b_20261006'),
                  ('confirm','1','coldstart_teacher_repair_eval_probe_20261006')]
            for stage,gpu,dependency in jobs:
                name=f'coldstart_teacher_repair_eval_{stage}_20261006'
                if teacher_done and done(dependency) and name not in dispatched:
                    if (ROOT.parent/'plans'/(name+'_launch.json')).exists():raise RuntimeError('Existing launch needs inspection '+name)
                    command=[sys.executable,str(ROOT/'scripts/launch_coldstart.py'),name,gpu,'--eval-workers','4']
                    response=subprocess.run(command,text=True,capture_output=True,check=True)
                    state['launches'].append(dict(name=name,**json.loads(response.stdout)))
                    dispatched.add(name);save();print('LAUNCH '+name,flush=True)
            expected={f'coldstart_teacher_repair_eval_{s}_20261006' for s in ('dev','probe','confirm')}
            if 'lane_c' in dispatched and all(c.poll()==0 for c in children) and all(done(n) for n in expected):
                state['complete']=True;save();print('COMPLETE remaining evaluations',flush=True);return
            time.sleep(15)
    except BaseException as error:
        state['error']=repr(error);save();raise

if __name__=='__main__':main()
