"""Run the predeclared five-arm QKV matrix; stop before confirmation.

Only the named own suites are launched. No cancellation or automatic resubmission.
Confirmation requires a separately sealed development selection.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from interface_transport import ssh_command
from prepare_qkv_plans import ROOT,DATE,SEEDS

WORKSPACE=Path(__file__).resolve().parents[2]
STATE=WORKSPACE/'plans'/('qkv_orchestration_'+DATE+'.json')
GPUS=dict(zip(SEEDS,('0','1','3')))


def status(names):
    code="""import json
from pathlib import Path
root=Path(ROOT)/'runs'
out={}
for name in NAMES:
 p=root/name/'suite.json'
 if not p.exists():out[name]={'status':'missing'};continue
 m=json.loads(p.read_text());out[name]={'status':m['status'],'complete':m.get('complete',False),'jobs':[{k:j.get(k) for k in ('name','status','exit_code')} for j in m['jobs']]}
print(json.dumps(out))
""".replace('ROOT',repr(ROOT)).replace('NAMES',repr(names))
    r=subprocess.run(ssh_command('xtrah100')+['python3','-'],input=code,text=True,capture_output=True,timeout=45)
    r.check_returncode();return json.loads(r.stdout)


def run():
    if STATE.exists():raise FileExistsError('Do not overwrite an existing controller state')
    state=dict(complete=False,launched=[],phase='preflight')
    def save():STATE.write_text(json.dumps(state,indent=2)+'\n')
    def launch(name,gpu,workers=1):
        args=[sys.executable,str(Path(__file__).with_name('launch_qkv.py')),name,gpu]
        if workers>1:args+=['--eval-workers',str(workers)]
        subprocess.run(args,check=True);state['launched'].append(dict(name=name,gpu=gpu,workers=workers));save()
    def wait(names):
        while True:
            current=status(names);state['status']=current;save()
            if any(x['status']=='failed' for x in current.values()):raise RuntimeError('Suite failed; inspect own logs before any new launch')
            if all(x.get('complete') for x in current.values()):return
            print(json.dumps(dict(phase=state['phase'],status={k:v['status'] for k,v in current.items()})),flush=True)
            time.sleep(20)
    save()
    wait(['qkv_prepare_'+DATE,'qkv_preflight_'+DATE,'qkv_telemetry_smoke_'+DATE])
    # Local receipt binds teacher qualification and the frozen protocol/sources.
    registration=json.loads((WORKSPACE/'plans'/('qkv_registration_'+DATE+'.json')).read_text())
    if not registration.get('teacher_preflight_passed') or not registration.get('smoke_passed'):raise RuntimeError('Preflight not sealed')
    import hashlib
    repo=Path(__file__).resolve().parents[1]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(repo/'docs/qkv_protocol.md')!=registration['protocol_document_sha256']:raise RuntimeError('Registered protocol changed')
    for relative,digest in registration['source_python_sha256'].items():
        if sha(repo/relative)!=digest:raise RuntimeError('Registered runtime source changed: '+relative)
    for filename,digest in registration['plans_sha256'].items():
        if sha(WORKSPACE/'plans'/filename)!=digest:raise RuntimeError('Registered plan changed: '+filename)
    try:
        state['phase']='train';save()
        names=[f'qkv_train_{seed}_{DATE}' for seed in SEEDS]
        for seed,name in zip(SEEDS,names):launch(name,GPUS[seed])
        wait(names)
        state['phase']='teacher_development';save()
        name=f'qkv_teacher_development_{DATE}';launch(name,'0');wait([name])
        state['phase']='development';save()
        names=[f'qkv_development_{seed}_{DATE}' for seed in SEEDS]
        for seed,name in zip(SEEDS,names):launch(name,GPUS[seed],4)
        wait(names)
        state.update(phase='ready_for_sealed_selection',complete=True);save()
    except BaseException as error:
        state['error']=repr(error);save();raise

if __name__=='__main__':run()
