"""Advance the fixed local plans only after completed remote dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import time
from interface_transport import ssh_command

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT.parent/'plans/coldstart_training_controller_20261006.json'
REMOTE="""import json
from pathlib import Path
w=Path('/ssd3/chenhan/VeRA-Mem-Workspace/runs')
print(json.dumps({p.parent.name:json.loads(p.read_text()) for p in w.glob('coldstart_*/suite.json')}))
"""
GROUPS=[
 [('coldstart_prepare_'+domain+'_20261006',None) for domain in ('wikipedia','matched','synthetic')]+[('coldstart_smoke_20261006',None)],
 [('coldstart_warm_'+lane+'_20261006',gpu) for lane,gpu in zip('abc','013')],
 [('coldstart_dictionary_'+lane+'_20261006',gpu) for lane,gpu in zip('ab','01')],
]

def main():
    log=dict(complete=False,groups=GROUPS,launches=[])
    def save():STATE.write_text(json.dumps(log,indent=2)+'\n')
    save()
    try:
        for group in GROUPS:
            while True:
                response=subprocess.run(ssh_command('xtrah100')+['python3','-'],input=REMOTE,text=True,capture_output=True,check=True)
                states=json.loads(response.stdout)
                for name,gpu in group:
                    state=states.get(name)
                    if state is None:
                        if gpu is None: raise RuntimeError('Missing prerequisite '+name)
                        launched=subprocess.run([sys.executable,str(ROOT/'scripts/launch_coldstart.py'),name,gpu],text=True,capture_output=True,check=True)
                        record=json.loads(launched.stdout);log['launches'].append(dict(name=name,**record));save()
                        print('LAUNCH '+name,flush=True)
                    elif state['status']=='failed':raise RuntimeError('Failed prerequisite '+name+': '+str(state.get('error')))
                if all(states.get(n,{}).get('complete') is True for n,_ in group):break
                time.sleep(10)
            print('COMPLETE '+','.join(n for n,_ in group),flush=True)
        log['complete']=True;save()
    except BaseException as error:
        log['error']=repr(error);save();raise

if __name__=='__main__':main()
