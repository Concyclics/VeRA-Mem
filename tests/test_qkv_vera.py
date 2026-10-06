"""Matched QKV controls, differentiable routing and immutable grouped CPU reads."""
import contextlib
import math

import pytest
import torch
from torch.nn import functional as F

from vera_mem.qkv_vera import QKVVeRA, QKVVectorDB
from vera_mem.reconstruction_vera import ReconstructionVeRA
from vera_mem.backend import QwenBackend
from vera_mem.context_distillation import ContextDistillationBackend


def module(mode="flat", readout="vera"):
    m=QKVVeRA(5,6,rank=3,key_dim=4,temperature=.5,seed=31,read_mode=mode,readout=readout)
    g=torch.Generator().manual_seed(91)
    m.fit_statistics(torch.randn(21,5,generator=g),torch.randn(24,5,generator=g))
    m.fit_value_statistics(torch.randn(24,5,generator=g))
    with torch.no_grad():m.b.fill_(.3)
    return m


def populate(m,features):
    store=m.new_store(record_routes=True);keys,values=m.encode_bank(features)
    for i in range(features.shape[0]):store.write_group(str(i),keys[3*i:3*i+3],values[3*i:3*i+3],0)
    return store,keys,values


def test_all_modes_match_parameters_statistics_and_rng():
    before=torch.random.get_rng_state().clone()
    models=[module(mode,readout) for mode in ('flat','grouped') for readout in ('vera','additive')]
    assert torch.equal(before,torch.random.get_rng_state())
    reference=models[0]
    for m in models:
        assert set(dict(m.named_parameters()))==set(dict(reference.named_parameters()))
        for name,p in m.named_parameters():assert torch.equal(p,dict(reference.named_parameters())[name]),name
        for name in ('A','query_center','support_center','value_center'):
            assert torch.equal(m.state_dict()[name],reference.state_dict()[name])
        assert not m.A.requires_grad and m.B.requires_grad and m.slot_position.eq(0).all()
    assert reference.trainable_numel()==2*5*4+5*3+6+6*3+3*4


def test_zero_position_flat_exactly_matches_old_three_slot_module():
    m=module();cfg=m.configuration();cfg.pop('read_mode');cfg.pop('readout');cfg.pop('qkv_version')
    old=ReconstructionVeRA(**cfg)
    old.load_state_dict({k:v for k,v in m.state_dict().items() if k not in ('slot_position','qkv_architecture')})
    features=torch.randn(4,3,5);keys,values=m.encode_bank(features);ok,ov=old.encode_bank(features)
    torch.testing.assert_close(keys,ok,rtol=0,atol=0);torch.testing.assert_close(values,ov,rtol=0,atol=0)
    x=torch.randn(2,7,5)
    torch.testing.assert_close(m(x,keys,values),old(x,ok,ov),rtol=0,atol=0)


def test_position_is_before_key_norm_and_repeatable_across_grouped_or_flat_shapes():
    m=module();features=torch.randn(2,4,3,5,requires_grad=True);x=torch.randn(2,3,5)
    query_before=m.encode_query(x).detach().clone()
    with torch.no_grad():m.slot_position.copy_(torch.arange(12).reshape(3,4)/10)
    grouped=m.encode_key(features);flat=m.encode_key(features.reshape(2,12,5))
    projected=m.Wk(m._domain_input(features.detach(),m.support_center))
    torch.testing.assert_close(grouped,F.normalize(projected+m.slot_position,dim=-1),rtol=0,atol=0)
    torch.testing.assert_close(grouped.reshape(2,12,4),flat,rtol=0,atol=0)
    torch.testing.assert_close(m.encode_query(x),query_before,rtol=0,atol=0)
    grouped[...,0].sum().backward()
    assert features.grad is None and m.slot_position.grad.abs().sum()>0 and m.Wk.weight.grad.abs().sum()>0
    for bad in (torch.randn(5),torch.randn(4,5),torch.randn(0,5),torch.randn(3,4)):
        with pytest.raises(ValueError):m.encode_key(bad)


def test_group_selects_logsumexp_not_best_individual_slot_and_reads_all_slots():
    store=QKVVectorDB(2,2,temperature=1.,read_mode='grouped',record_routes=True)
    store.write_group('max_slot',torch.tensor([[1.,0.],[0.,1.],[0.,1.]]),torch.zeros(3,2),0)
    k=torch.tensor([[.95,math.sqrt(1-.95**2)]]).expand(3,2)
    v=torch.tensor([[1.,0.],[0.,2.],[3.,4.]])
    store.write_group('group',k,v,0);info=store.search(torch.tensor([1.,0.]))
    assert info['indices'].tolist()==[3,4,5] and int(info['selected_fact'])==1
    torch.testing.assert_close(info['weights'],torch.ones(3)/3)
    torch.testing.assert_close(info['mixed_value'],v.mean(0))
    assert info['fact_ids']==[['group']*3] and info['slot_ids'].tolist()==[0,1,2]
    assert len(store.route_log)==1 and store.route_log[0]['indices'].tolist()==[3,4,5]


@pytest.mark.parametrize('mode',['flat','grouped'])
@pytest.mark.parametrize('readout',['vera','additive'])
def test_same_encoded_bank_cpu_and_tensor_reads_match_without_mutation(mode,readout):
    m=module(mode,readout);features=torch.randn(5,3,5);store,k,v=populate(m,features);before=store.hash()
    x=torch.randn(2,4,5);expected,train=m(x,k,v,return_info=True)
    actual,online=m.cpu_delta(x,store,return_info=True)
    torch.testing.assert_close(actual,expected,rtol=3e-6,atol=3e-6)
    for name in ('indices','scores','weights'):torch.testing.assert_close(online[name],train[name],rtol=3e-6,atol=3e-6)
    assert store.hash()==before and all(t.device.type=='cpu' and not t.requires_grad for t in (store.keys,store.values))
    assert online['indices'].shape[-1]==(4 if mode=='flat' else 3)
    assert store.resident_bytes()['total_bytes']==5*3*(4+3)*4


@pytest.mark.parametrize('mode',['flat','grouped'])
def test_independent_banks_match_per_row_and_no_cross_row_gradients(mode):
    m=module(mode);x=torch.randn(2,4,5);features=torch.randn(2,4,3,5,requires_grad=True)
    keys,values=m.encode_bank(features);keys.retain_grad();values.retain_grad()
    result=m(x,keys,values)
    reference=torch.stack([m(x[i],keys[i],values[i]) for i in range(2)])
    torch.testing.assert_close(result,reference,rtol=2e-6,atol=2e-6)
    changed=values.detach().clone();changed[1]+=20
    torch.testing.assert_close(m(x,keys,changed)[0],result[0],rtol=0,atol=0)
    result[0].square().sum().backward()
    assert features.grad is None and keys.grad[1].eq(0).all() and values.grad[1].eq(0).all()
    assert values.grad[0].abs().sum()>0


def test_grouped_CE_path_is_selected_group_only_but_dense_address_reaches_other_groups():
    m=module('grouped');x=torch.randn(1,1,5,requires_grad=True)
    keys=torch.randn(9,4,requires_grad=True);values=torch.randn(9,3,requires_grad=True)
    residual,info=m(x,keys,values,return_info=True);loss=residual.square().sum()
    kg,vg=torch.autograd.grad(loss,(keys,values),retain_graph=True)
    selected=set(info['indices'].flatten().tolist());other=[i for i in range(9) if i not in selected]
    assert kg[other].eq(0).all() and vg[other].eq(0).all()
    address=m.group_address_loss(x,torch.tensor([[2]]),keys)
    dense,=torch.autograd.grad(address,(keys,))
    assert bool((dense[other].norm(dim=-1)>0).all())


@pytest.mark.parametrize('mode',['flat','grouped'])
def test_fresh_feature_bank_graph_trains_position_writer_and_readout(mode):
    m=module(mode);features=torch.randn(2,12,5,requires_grad=True);m.set_feature_bank(features)
    optimizer=torch.optim.SGD(m.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad();x=torch.randn(2,3,5,requires_grad=True)
        loss=m(x).square().mean()+.2*m.group_address_loss(x,torch.tensor([[0,1,2],[1,2,3]]))
        loss.backward()
        for name in ('Wq.weight','Wk.weight','Wv.weight','b','B','slot_position'):
            assert dict(m.named_parameters())[name].grad.abs().sum()>0,name
        assert features.grad is None and x.grad.abs().sum()>0 and m.A.grad is None
        optimizer.step()


@pytest.mark.parametrize('mode',['flat','grouped'])
@pytest.mark.parametrize('readout',['vera','additive'])
def test_empty_bank_zero_and_teacher_never_reads_memory(mode,readout):
    m=module(mode,readout);empty=m.new_store(record_routes=True);x=torch.randn(2,3,5)
    tensor,info=m(x,torch.empty(0,4),torch.empty(0,3),return_info=True)
    online,oi=m.cpu_delta(x,empty,return_info=True)
    assert tensor.eq(0).all() and online.eq(0).all() and info['indices'].shape==oi['indices'].shape==(2,3,0)
    def forbidden(*args,**kwargs):raise AssertionError('teacher accessed memory')
    empty.search=forbidden;m.encode_query=forbidden
    actual,teacher=m.cpu_delta(x,empty,teacher=True,return_info=True)
    assert actual.eq(0).all() and teacher=={'teacher_bypass':True}


def test_additive_formula_same_retrieval_and_unused_A():
    vera=module('grouped','vera');add=module('grouped','additive')
    features=torch.randn(4,3,5);k,v=vera.encode_bank(features);x=torch.randn(2,4,5)
    _,vi=vera(x,k,v,True);actual,ai=add(x,k,v,True)
    for name in ('indices','scores','weights'):torch.testing.assert_close(vi[name],ai[name],rtol=0,atol=0)
    mixed=(v[ai['indices']]*ai['weights'][...,None]).sum(-2)
    torch.testing.assert_close(actual,F.linear(mixed,add.B)*add.b)
    before=actual.detach().clone()
    with torch.no_grad():add.A.add_(100)
    torch.testing.assert_close(add(x,k,v),before,rtol=0,atol=0)


@pytest.mark.parametrize('mode',['flat','grouped'])
def test_atomic_replacement_snapshot_roundtrip_and_mode_hash(mode,tmp_path):
    m=module(mode);store,k,v=populate(m,torch.randn(3,3,5));before=store.snapshot();digest=store.hash()
    for invalid,stamp in [(k[:3],-1),(k[:3].clone(),1)]:
        if stamp==1:invalid[2,0]=float('nan')
        with pytest.raises(ValueError):store.write_group('0',invalid,v[:3],stamp)
        assert store.hash()==digest
    store.write_group('1',k[3:6],v[3:6]+.5,1)
    updated_hash=store.hash()
    with pytest.raises(ValueError,match='Stale'):store.write_group('1',k[3:6],v[3:6],0)
    assert store.hash()==updated_hash
    assert torch.equal(store.values[:3],before['store']['values'][:3])
    assert torch.equal(store.values[6:],before['store']['values'][6:])
    path=tmp_path/'bank.pt';store.save(path);clone=QKVVectorDB.load(path,record_routes=True)
    assert type(clone)is QKVVectorDB and clone.read_mode==mode and clone.hash()==store.hash()
    with pytest.raises(ValueError):QKVVectorDB.from_snapshot(dict(store.snapshot(),read_mode='unknown'))
    changed=store.snapshot();changed['read_mode']='grouped' if mode=='flat' else 'flat'
    assert QKVVectorDB.from_snapshot(changed).hash()!=store.hash()
    partial=store.snapshot();partial['store']['timestamps'][3]=2
    with pytest.raises(ValueError):QKVVectorDB.from_snapshot(partial)


def test_grouped_CPU_query_validation_and_zero_query_batch():
    m=module('grouped');store,_,_=populate(m,torch.randn(2,3,5))
    for query in (torch.zeros(4),torch.full((4,),float('nan')),torch.full((4,),1e38),torch.ones(5)):
        with pytest.raises(ValueError):store.search(query)
    info=store.search(torch.empty(0,4))
    assert info['mixed_value'].shape==(0,3) and info['indices'].shape==(0,3) and info['ids']==[]


def test_configuration_roundtrip_rejects_wrong_read_mode_or_readout():
    m=module('grouped','additive');clone=QKVVeRA(**m.configuration());clone.load_state_dict(m.state_dict())
    for field,value in [('read_mode','flat'),('readout','vera')]:
        wrong=QKVVeRA(**dict(m.configuration(),**{field:value}))
        with pytest.raises(RuntimeError,match='architecture'):wrong.load_state_dict(m.state_dict())
    with pytest.raises(ValueError):m.cpu_delta(torch.randn(1,5),module('flat').new_store())
    for cfg in ({'slots':1},{'top_k':3},{'read_mode':'unknown'},{'readout':'unknown'}):
        with pytest.raises(ValueError):QKVVeRA(5,6,**cfg)


def test_real_backend_teacher_scope_bypasses_both_read_modes():
    class Backend:
        _teacher_forward=ContextDistillationBackend._teacher_forward
        _TEACHER_TRANSIENT_FIELDS=ContextDistillationBackend._TEACHER_TRANSIENT_FIELDS
        @contextlib.contextmanager
        def disabled(self):
            old=self.mode;self.mode='none'
            try:yield
            finally:self.mode=old
    for mode in ('flat','grouped'):
        backend=Backend();backend.mode='vector_vera';backend.capture_layer_input=False
        backend.vector_vera=module(mode);backend.vector_store=backend.vector_vera.new_store();backend.vector_override=None
        backend.last_retrieval={'sentinel':torch.tensor(7)};backend.prefill_retrieval=None;backend.retrieval_trace=[]
        def forbidden(*args,**kwargs):raise AssertionError('teacher searched')
        backend.vector_store.search=forbidden;backend.vector_vera.encode_query=forbidden
        x=torch.randn(1,2,5);base=torch.randn(1,2,6)
        with backend._teacher_forward():
            assert torch.equal(QwenBackend._inject(backend,None,(x,),base),base)
        assert backend.mode=='vector_vera' and backend.last_retrieval['sentinel']==7
