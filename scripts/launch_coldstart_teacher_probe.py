"""Launch the fixed, train-only teacher probe on the authorized H100 GPU 3."""
import subprocess
from interface_transport import ssh_command,rsync_transport

subprocess.run(['rsync','-az','-e',rsync_transport(),'src','scripts','tests','docs',
    'xtrah100:/ssd3/chenhan/VeRA-Mem-Workspace/VeRA-Mem/'],check=True)
REMOTE=r'''
from pathlib import Path
import hashlib,json,os,shutil,subprocess
w=Path('/ssd3/chenhan/VeRA-Mem-Workspace')
root=w/'runs/coldstart_teacher_probe_v2_20261006'
gpu='GPU-b39c5755-644d-a3b3-8e76-75beb4142100'
used=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
if used>1000:raise RuntimeError('GPU 3 is not free')
root.mkdir(exist_ok=False)
source=root/'source'
for directory in ('src','scripts','tests','configs','docs'):
 shutil.copytree(w/'VeRA-Mem'/directory,source/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
command=['python3','-u',str(source/'scripts/probe_coldstart_teacher.py'),'--model',str(w/'models/Qwen3-4B-Instruct-2507'),
 '--cache',str(w/'runs/coldstart_repair_20261006/features/train.pt'),'--run-dir',str(root/'probe')]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
 TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(w/'env_deps')+':'+str(source/'src'))
hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
with (root/'probe.log').open('x') as log:
 child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=source,start_new_session=True)
record=dict(pid=child.pid,command=command,gpu_uuid=gpu,gpu_memory_used_mib_before_start=used,source_files_sha256=hashes)
(root/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(pid=child.pid,root=str(root))))
'''
response=subprocess.run(ssh_command('xtrah100')+['python3','-'],input=REMOTE,text=True,capture_output=True,check=True)
print(response.stdout)
