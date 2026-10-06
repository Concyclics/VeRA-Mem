"""The actual launch gate must reject changed plans, sources, and cache bytes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
spec=importlib.util.spec_from_file_location('block_registry',SCRIPTS/'block_registry.py')
registry=importlib.util.module_from_spec(spec);spec.loader.exec_module(registry)

def setup_files(tmp_path):
    (tmp_path/'docs').mkdir();(tmp_path/'src/vera_mem').mkdir(parents=True)
    protocol=tmp_path/'docs/block_protocol.md';protocol.write_text('sealed protocol')
    source=tmp_path/'src/vera_mem/block_run.py';source.write_text('frozen runtime')
    cache=tmp_path/'train.pt';cache.write_bytes(b'frozen features')
    plan=[dict(name='one',entry='block',arguments=['--stage','train','--cache',str(cache)])]
    plan_path=tmp_path/'block_train.json';plan_path.write_text(json.dumps(plan))
    value=dict(protocol='block-preregistration-v1',selection_policy='all_18_final_checkpoints_no_selection',
        teacher_preflight_passed=True,smoke_passed=True,protocol_document_sha256=registry.sha(protocol),
        source_python_sha256={'src/vera_mem/block_run.py':registry.sha(source)},
        plans_sha256={plan_path.name:registry.sha(plan_path)},feature_cache_sha256={'train.pt':registry.sha(cache)})
    return value,protocol,source,cache,plan_path,plan

def test_matching_receipt_accepts(tmp_path):
    value,protocol,source,cache,path,plan=setup_files(tmp_path)
    assert registry.validate_registration(value,protocol)
    assert registry.validate_inputs(value,path,plan)

@pytest.mark.parametrize('field',['protocol','source','cache','plan'])
def test_changed_evidence_rejected(tmp_path,field):
    value,protocol,source,cache,path,plan=setup_files(tmp_path)
    targets=dict(protocol=protocol,source=source,cache=cache,plan=path)
    targets[field].write_bytes(b'changed')
    with pytest.raises(ValueError):
        registry.validate_registration(value,protocol);registry.validate_inputs(value,path,plan)

@pytest.mark.parametrize('field,value',[('teacher_preflight_passed',False),('smoke_passed',False),('selection_policy','pick_best_after_eval')])
def test_false_preflight_or_adaptive_selection_rejected(tmp_path,field,value):
    receipt,protocol,*_=setup_files(tmp_path);receipt[field]=value
    with pytest.raises(ValueError):registry.validate_registration(receipt,protocol)
