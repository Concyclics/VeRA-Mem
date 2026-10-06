"""Seen-target probe preserves canonical tensors and explicit train provenance."""
import copy
import importlib.util
from pathlib import Path

import pytest
torch=pytest.importorskip('torch')
PATH=Path(__file__).resolve().parents[1]/'scripts/prepare_coldstart_train_probe.py'
spec=importlib.util.spec_from_file_location('coldstart_probe',PATH)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def packet():
    count=256
    rows=[dict(id=f'id{i}',entity=f'entity{i}',relation='r',split='train',a='alpha one two',b='beta one two',
        questions=[f'question{i}'],supports=[[f'A support{i}'],[f'B support{i}']],a_tokens=[1,2,3],b_tokens=[4,2,3],provenance={}) for i in range(count)]
    return dict(protocol='dictionary-coldstart-v1',task='synthetic',split='train',rows=rows,
        q=torch.arange(count*4).reshape(count,1,4).half(),last=torch.arange(count*8).reshape(count,2,1,4).half(),
        pool=torch.arange(count*8).reshape(count,2,1,4).float()+1,qids=['canonical'],sids=['canonical'],model_revision='fixed')


def test_probe_first_sampler_targets_and_equal_view_slots_are_not_heldout():
    from vera_mem.coldstart_run import BankSampler
    source=packet();original=copy.deepcopy(source);out=probe.build_probe(source,'a'*64)
    sampler=BankSampler(source['rows'],63042);expected=[]
    for _ in range(16):expected.extend(sampler.next()['targets'])
    assert out['source_indices']==expected and len(out['rows'])==128
    assert out['split']=='dev' and out['original_split']=='train' and out['diagnostic']==probe.DIAGNOSTIC
    assert out['task']=='synthetic_train_probe'
    assert all(r['split']=='train' and r['questions'][0]==r['questions'][1] for r in out['rows'])
    for name in ('last','pool'):
        assert torch.equal(out[name][:,:,0],source[name][expected,:,0])
        assert torch.equal(out[name][:,:,0],out[name][:,:,1])
    assert torch.equal(out['q'][:,0],out['q'][:,1])
    assert source['rows']==original['rows'] and torch.equal(source['q'],original['q'])


def test_probe_rejects_test_cache_and_preexisting_heldout_training_features():
    source=packet();source['split']='confirm'
    with pytest.raises(ValueError,match='training cache'):probe.build_probe(source,'a'*64)
    source=packet();source['q']=source['q'].repeat(1,2,1)
    with pytest.raises(ValueError,match='canonical'):probe.build_probe(source,'a'*64)
