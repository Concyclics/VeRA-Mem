"""Validate the fixed design receipt before training or confirmation."""
import hashlib
import json
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def validate_registration(value,protocol_path):
    if value.get('protocol')!='block-preregistration-v1':raise ValueError('Wrong registration protocol')
    if value.get('selection_policy')!='all_18_final_checkpoints_no_selection':raise ValueError('No adaptive selection is permitted')
    if not value.get('teacher_preflight_passed') or not value.get('smoke_passed'):raise ValueError('Preflight not passed')
    path=Path(protocol_path)
    if sha(path)!=value['protocol_document_sha256']:raise ValueError('Protocol changed')
    repo=path.parent.parent
    actual={str(f.relative_to(repo)):sha(f) for f in sorted((repo/'src/vera_mem').glob('*.py'))}
    if actual!=value['source_python_sha256']:raise ValueError('Runtime sources differ from registration')
    return True


def validate_inputs(value,plan_path,plan):
    path=Path(plan_path)
    if sha(path)!=value['plans_sha256'].get(path.name):raise ValueError('Unregistered or changed plan')
    for job in plan:
        args=job['arguments']
        if '--cache' in args:
            cache=Path(args[args.index('--cache')+1])
            if sha(cache)!=value['feature_cache_sha256'].get(cache.name):raise ValueError('Unregistered or changed feature cache')
    return True
