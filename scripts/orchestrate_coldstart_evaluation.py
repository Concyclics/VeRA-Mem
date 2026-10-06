"""Continue one fixed evaluation lane; completed stages are never resubmitted."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import re
from interface_transport import ssh_command

ROOT=Path(__file__).resolve().parents[1]
REMOTE="""import json
from pathlib import Path
w=Path('/ssd3/chenhan/VeRA-Mem-Workspace/runs')
print(json.dumps({p.parent.name:json.loads(p.read_text()) for p in w.glob('coldstart_*/suite.json')}))
"""

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('lane',choices=('a','b','c'))
    p.add_argument('--workers',type=int,choices=(2,4),default=4)
    p.add_argument('--probe-name',help='Explicit new plan name after an inspected preflight-only failure')
    p.add_argument('--record-suffix',default='')
    a=p.parse_args()
    if a.probe_name and not re.fullmatch(r'coldstart_[a-z0-9_]+',a.probe_name):p.error('Unsafe plan name')
    if a.record_suffix and not re.fullmatch(r'_[a-z0-9]+',a.record_suffix):p.error('Unsafe record suffix')
    gpu=dict(a='0',b='1',c='3')[a.lane]
    names=[f'coldstart_eval_dev_{a.lane}_20261006',a.probe_name or f'coldstart_probe_{a.lane}_20261006',
           f'coldstart_eval_confirm_{a.lane}_20261006']
    record=dict(complete=False,lane=a.lane,gpu=gpu,names=names,launches=[])
    output=ROOT.parent/'plans'/f'coldstart_eval_controller_{a.lane}{a.record_suffix}_20261006.json'
    if output.exists():raise FileExistsError(output)
    def save():output.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        for name in names:
            if '_confirm_' in name:
                choice=ROOT.parent/'plans/coldstart_teacher_selection_20261006.json'
                if not choice.is_file():raise RuntimeError('Fix the train-only teacher decision before confirmation')
            while True:
                r=subprocess.run(ssh_command('xtrah100')+['python3','-'],input=REMOTE,text=True,capture_output=True,check=True)
                state=json.loads(r.stdout).get(name)
                if state is None:
                    launch_record=ROOT.parent/'plans'/(name+'_launch.json')
                    if launch_record.exists():raise RuntimeError('Launch exists but no remote state; inspect before retrying')
                    command=[sys.executable,str(ROOT/'scripts/launch_coldstart.py'),name,gpu,'--eval-workers',str(a.workers)]
                    r=subprocess.run(command,text=True,capture_output=True,check=True)
                    record['launches'].append(dict(name=name,**json.loads(r.stdout)));save()
                    print('LAUNCH '+name,flush=True)
                elif state.get('status')=='failed':raise RuntimeError('Failed suite '+name+': '+str(state.get('error')))
                elif state.get('complete') is True:
                    print('COMPLETE '+name,flush=True);break
                time.sleep(15)
        record['complete']=True;save()
    except BaseException as error:
        record['error']=repr(error);save();raise

if __name__=='__main__':main()
