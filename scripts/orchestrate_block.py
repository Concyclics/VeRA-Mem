"""Execute every preregistered arm; final checkpoints are never selected by eval."""
import json
from pathlib import Path
import subprocess
import sys
import time
from orchestrate_qkv import status
from prepare_block_plans import DATE,SEEDS
from block_registry import validate_registration,sha
ROOT=Path(__file__).resolve().parents[2]
REGISTRATION=ROOT/'plans'/f'block_registration_{DATE}.json'
STATE=ROOT/'plans'/f'block_orchestration_{DATE}.json'
GPUS=dict(zip(SEEDS,('0','1','3')))

def main():
    if STATE.exists():raise FileExistsError('Do not overwrite a controller state')
    registration=json.loads(REGISTRATION.read_text())
    validate_registration(registration,ROOT/'VeRA-Mem/docs/block_protocol.md')
    for name,digest in registration['plans_sha256'].items():
        if sha(ROOT/'plans'/name)!=digest:raise ValueError('Plan changed: '+name)
    state=dict(complete=False,phase='train',launched=[],registration_sha256=sha(REGISTRATION))
    def save():STATE.write_text(json.dumps(state,indent=2)+'\n')
    def launch(name,gpu,workers=1):
        command=[sys.executable,str(Path(__file__).with_name('launch_block.py')),name,gpu,'--registration',str(REGISTRATION)]
        if workers>1:command+=['--eval-workers',str(workers)]
        subprocess.run(command,check=True)
        state['launched'].append(dict(name=name,gpu=gpu,workers=workers));save()
    def wait(names):
        while True:
            current=status(names);state['status']=current;save()
            if any(x['status']=='failed' for x in current.values()):raise RuntimeError('An owned suite failed; no retries')
            if all(x.get('complete') for x in current.values()):return
            print(json.dumps(dict(phase=state['phase'],status={k:v['status'] for k,v in current.items()})),flush=True)
            time.sleep(20)
    save()
    try:
        names=[f'block_train_{seed}_{DATE}' for seed in SEEDS]
        for seed,name in zip(SEEDS,names):launch(name,GPUS[seed])
        wait(names)
        for part in ('development','confirmation'):
            state['phase']='teacher_'+part;save();name=f'block_teacher_{part}_{DATE}'
            launch(name,'0');wait([name])
            state['phase']=part;save();names=[f'block_{part}_{seed}_{DATE}' for seed in SEEDS]
            for seed,name in zip(SEEDS,names):launch(name,GPUS[seed],4)
            wait(names)
        state.update(complete=True,phase='complete');save()
    except BaseException as error:
        state['error']=repr(error);save();raise

if __name__=='__main__':main()
