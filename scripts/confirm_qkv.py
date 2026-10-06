"""Complete all predeclared confirmation conditions after a sealed selection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from orchestrate_qkv import status, WORKSPACE, GPUS, DATE, SEEDS
from summarize_qkv import validate_selection


def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);a=p.parse_args()
    selection=json.loads(a.selection.read_text())
    validate_selection(selection,Path(__file__).resolve().parents[1]/'docs/qkv_protocol.md')
    state_path=WORKSPACE/'plans'/f'qkv_confirmation_orchestration_{DATE}.json'
    state=dict(complete=False,phase='confirmation',launched=[],selection_sha256=hashlib.sha256(a.selection.read_bytes()).hexdigest(),selected_arm=selection['decision']['selected_arm'])
    with state_path.open('x') as f:json.dump(state,f,indent=2)
    def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
    def launch(name,gpu,workers=1):
        args=[sys.executable,str(Path(__file__).with_name('launch_qkv.py')),name,gpu,'--selection-evidence',str(a.selection)]
        if workers>1:args+=['--eval-workers',str(workers)]
        subprocess.run(args,check=True);state['launched'].append(dict(name=name,gpu=gpu,workers=workers));save()
    def wait(names):
        while True:
            current=status(names);state['status']=current;save()
            if any(x['status']=='failed' for x in current.values()):raise RuntimeError('Confirmation failed; inspect owned logs, no automatic resubmission')
            if all(x.get('complete') for x in current.values()):return
            print(json.dumps(dict(phase=state['phase'],completed_jobs={n:sum(j['status']=='complete' for j in v.get('jobs',[])) for n,v in current.items()})),flush=True)
            time.sleep(20)
    try:
        names=[f'qkv_confirmation_{seed}_{DATE}' for seed in SEEDS]
        for seed,name in zip(SEEDS,names):launch(name,GPUS[seed],4)
        wait(names);state['phase']='teacher_confirmation';save()
        name=f'qkv_teacher_confirmation_{DATE}';launch(name,'0');wait([name])
        state.update(phase='complete',complete=True);save()
    except BaseException as error:state['error']=repr(error);save();raise


if __name__=='__main__':main()
