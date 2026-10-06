"""Bind completed preflight receipts, cache hashes, and the fixed design."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from interface_transport import ssh_command
from block_registry import sha,validate_registration
from prepare_block_plans import ROOT,DATE,SEEDS,ARMS,plans
LOCAL=Path(__file__).resolve().parents[2]
REPO=LOCAL/'VeRA-Mem'

def main():
    destination=LOCAL/'plans'/f'block_registration_{DATE}.json'
    if destination.exists():raise FileExistsError('Do not overwrite a sealed registration')
    code='''import json,hashlib
from pathlib import Path
root=Path(ROOT)/'runs'
records={}; artifacts={}
for name in NAMES:
 suite=root/name; packet=json.loads((suite/'suite.json').read_text())
 if not packet.get('complete'):raise RuntimeError('Preflight incomplete: '+name)
 paths=[suite/'suite.json']
 for j in packet['jobs']:
  p=suite/j['name']/'manifest.json';d=json.loads(p.read_text())
  if not d.get('complete') or not d.get('backbone_unchanged'):raise RuntimeError('Child incomplete: '+str(p))
  records[j['name']]=d;paths.append(p)
  for f in ('summary.json','training_status.json'):
   if (p.parent/f).exists():paths.append(p.parent/f)
 for p in paths:artifacts[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(dict(records=records,artifacts=artifacts)))
'''.replace('ROOT',repr(ROOT)).replace('NAMES',repr([f'block_prepare_{DATE}',f'block_preflight_{DATE}',f'block_smoke_{DATE}']))
    result=subprocess.run(ssh_command('xtrah100')+['python3','-'],input=code,text=True,capture_output=True,timeout=90)
    result.check_returncode();receipts=json.loads(result.stdout);records=receipts['records']
    teacher=records['teacher_train']['result']
    if not teacher['qualified'] or teacher['generation_calls']!=128:raise ValueError('Teacher qualification failed')
    smoke=[r for k,r in records.items() if k.startswith('smoke_')]
    if len(smoke)!=6 or not all(r['complete'] for r in smoke):raise ValueError('Three train/eval smokes required')
    feature=records['features']['result']
    registration=dict(protocol='block-preregistration-v1',sealed_at=datetime.now(timezone.utc).isoformat(),
        selection_policy='all_18_final_checkpoints_no_selection',teacher_preflight_passed=True,smoke_passed=True,
        arms=list(ARMS),seeds=list(SEEDS),updates=2048,data_seed=221042,
        model_revision='cdbee75f17c01a7cc42f958dc650907174af0554',
        protocol_document_sha256=sha(REPO/'docs/block_protocol.md'),
        source_python_sha256={str(f.relative_to(REPO)):sha(f) for f in sorted((REPO/'src/vera_mem').glob('*.py'))},
        plans_sha256={name+'.json':sha(LOCAL/'plans'/(name+'.json')) for name in plans()},
        feature_cache_sha256={s+'.pt':r['cache_sha256'] for s,r in feature['caches'].items()},
        feature_result=feature,preflight_artifacts_sha256=receipts['artifacts'],preflight_receipts=records,
        known_limitations=['uniform group membership weights are not signed outer contributions',
          'preparation snapshot precedes final trace-only changes; formal runtime is frozen separately',
          'same budget is not equal convergence; no adaptive continuation',
          'three annotated word spans; no arbitrary document fact discovery'])
    validate_registration(registration,REPO/'docs/block_protocol.md')
    with destination.open('x') as f:json.dump(registration,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(destination),sha256=sha(destination),teacher_correct=128,smoke_jobs=6,sources=len(registration['source_python_sha256']))))

if __name__=='__main__':main()
