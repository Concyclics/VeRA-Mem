"""Read current manifests and short log tails on the authorized H100 host."""
import subprocess
from interface_transport import ssh_command

REMOTE = r'''
from pathlib import Path
import json
root=Path('/ssd3/chenhan/VeRA-Mem-Workspace/runs')
for path in sorted(root.glob('coldstart_*/suite.json')):
 d=json.loads(path.read_text())
 print(path.parent.name,d['status'],[(j['name'],j['status']) for j in d['jobs']])
 for j in d['jobs']:
  if j['status'] in ('running','failed'):
   status=Path(j['output'])/'training_status.json'
   if status.exists():
    s=json.loads(status.read_text());print({k:s.get(k) for k in ('updates','target_unique_count','key_gradient_coverage','value_gradient_coverage','elapsed_seconds')})
   training=Path(j['output'])/'training.jsonl'
   if training.exists():
    with training.open('rb') as h:
     h.seek(max(0,training.stat().st_size-24000));lines=h.read().splitlines()
    try:
     s=json.loads(lines[-1]);print({k:s.get(k) for k in ('step','total_loss','full_sequence_ce','base_values_gradient_slots','base_retained_dense_mass_mean','elapsed_seconds')})
    except (ValueError,IndexError):pass
   p=Path(j['log'])
   if p.exists() and (not training.exists() or j['status']=='failed'):print(p.read_text()[-500:])
'''
if __name__ == '__main__':
    result=subprocess.run(ssh_command('xtrah100')+['python3','-'],input=REMOTE,text=True,capture_output=True,check=True)
    print(result.stdout,end='')
