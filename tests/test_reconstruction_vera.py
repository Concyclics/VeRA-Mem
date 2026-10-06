"""Content slots, grouped updates and the actual-input sparse VeRA contract."""
import pytest
import torch

from vera_mem.interface_variants import InterfaceVectorVeRA
from vera_mem.reconstruction_vera import ReconstructionVeRA, GroupedVectorDB, pool_content_slots


def module(slots=1, train_B=False, top_k=4):
    result = ReconstructionVeRA(5, 5, rank=3, key_dim=4, top_k=top_k,
                                temperature=.5, seed=31, slots=slots, train_B=train_B)
    generator = torch.Generator().manual_seed(53)
    result.fit_statistics(torch.randn(25, 5, generator=generator), torch.randn(23, 5, generator=generator))
    result.fit_value_statistics(torch.randn(21, 5, generator=generator))
    with torch.no_grad():
        result.b.fill_(.3)
    return result


def test_one_slot_is_exact_existing_pooled_interface_and_fixed_B_path():
    model = module()
    config = model.configuration()
    config.pop("slots")
    config.pop("reconstruction_version")
    baseline = InterfaceVectorVeRA(**config)
    baseline.load_state_dict({k:v for k,v in model.state_dict().items() if k != "reconstruction_architecture"})
    features = torch.randn(7, 1, 5)
    keys, values = model.encode_bank(features)
    torch.testing.assert_close(keys, baseline.encode_key(features[:, 0]), rtol=0, atol=0)
    torch.testing.assert_close(values, baseline.encode_value(features[:, 0]), rtol=0, atol=0)
    x = torch.randn(2, 4, 5)
    expected, old_info = baseline(x, keys, values, return_info=True)
    actual, info = model(x, keys, values, return_info=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name in old_info:
        torch.testing.assert_close(info[name], old_info[name], rtol=0, atol=0)
    assert torch.equal(info['fact_indices'], info['indices']) and info['slot_indices'].eq(0).all()
    assert 'A' not in dict(model.named_parameters()) and 'B' not in dict(model.named_parameters())


def test_two_factors_start_with_identical_shared_weights_without_consuming_rng():
    state = torch.random.get_rng_state().clone()
    variants = [module(slots, learned) for slots in (1,3) for learned in (False,True)]
    assert torch.equal(torch.random.get_rng_state(), state)
    reference = variants[0]
    for model in variants:
        for name in ['A','B','b','Wq.weight','Wk.weight','Wv.weight','query_center','support_center','value_center']:
            assert torch.equal(model.state_dict()[name], reference.state_dict()[name]), name
        assert not model.A.requires_grad
        assert model.B.requires_grad == model.train_B


def test_word_masks_exclude_scaffolding_and_show_token_vs_word_weighting():
    features = torch.tensor([[[float('nan'),99.], [1.,2.], [3.,4.], [5.,6.], [7.,8.], [100.,100.]]])
    content = torch.tensor([[0,1,1,1,1,0]], dtype=torch.bool)
    masks = torch.tensor([[[0,1,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0]]], dtype=torch.bool)
    slots = pool_content_slots(features, content, slots=3, slot_masks=masks)
    whole = pool_content_slots(features, content)
    torch.testing.assert_close(slots, torch.tensor([[[2.,3.],[5.,6.],[7.,8.]]]), rtol=0, atol=0)
    torch.testing.assert_close(whole, torch.tensor([[[4.,5.]]]), rtol=0, atol=0)
    assert not torch.equal(whole[:,0], slots.mean(1))
    torch.testing.assert_close(pool_content_slots(features,content,slots=3), slots, rtol=0, atol=0)


@pytest.mark.parametrize('error', ['outside','overlap','missing','empty','float'])
def test_slot_masks_reject_ambiguous_content_partitions(error):
    features = torch.randn(1,5,5)
    content = torch.tensor([[0,1,1,1,0]],dtype=torch.bool)
    masks = torch.zeros(1,3,5,dtype=torch.bool)
    masks[0,torch.arange(3),torch.arange(1,4)] = True
    if error=='outside':masks[0,0,0]=True
    elif error=='overlap':masks[0,1,1]=True
    elif error=='missing':masks[0,0,1]=False
    elif error=='empty':masks[0,0,1]=False;masks[0,1,1]=True
    else:masks=masks.float()
    with pytest.raises(ValueError):pool_content_slots(features,content,slots=3,slot_masks=masks)


def test_grouped_features_preserve_slot_order_and_detach_writer_only():
    model = module(3, True)
    features = torch.randn(2,4,3,5,requires_grad=True)
    keys,values = model.encode_bank(features)
    assert keys.shape==(2,12,4) and values.shape==(2,12,3)
    torch.testing.assert_close(keys[:,3:6],model.encode_key(features[:,1]))
    x = torch.randn(2,4,5,requires_grad=True)
    model(x,keys,values).square().sum().backward()
    assert features.grad is None
    assert x.grad.abs().sum()>0
    for name in ['Wq.weight','Wk.weight','Wv.weight','B','b']:
        assert dict(model.named_parameters())[name].grad.abs().sum()>0,name
    assert model.A.grad is None


def test_feature_bank_recomputes_graph_each_step_and_is_not_checkpoint_state():
    model = module(3)
    features = torch.randn(2,9,5,requires_grad=True)
    model.set_feature_bank(features)
    source_copy = features.detach().clone()
    with torch.no_grad():features.add_(100)
    optimizer = torch.optim.SGD(model.parameters(),lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        x = torch.randn(2,3,5)
        model(x,None,None).square().mean().backward()
        assert model.Wv.weight.grad.abs().sum()>0
        optimizer.step()
    assert features.grad is None
    assert torch.equal(model._feature_bank[0],source_copy)
    assert not any('feature_bank' in key for key in model.state_dict())
    clone = ReconstructionVeRA(**model.configuration())
    clone.load_state_dict(model.state_dict())
    assert clone._feature_bank is None
    model.clear_feature_bank()
    with pytest.raises(RuntimeError,match='feature bank'):model(torch.randn(1,1,5))


def test_nonfinite_or_incomplete_features_cannot_replace_a_valid_bank():
    model=module(3);model.set_feature_bank(torch.randn(6,5))
    previous=model._feature_bank
    for value in [torch.randn(5,5),torch.full((6,5),float('nan')),torch.ones(6,5,dtype=torch.long)]:
        with pytest.raises(ValueError):model.set_feature_bank(value)
        assert model._feature_bank is previous
    with pytest.raises(ValueError):model.encode_bank(torch.full((2,3,5),float('inf')))


def test_independent_bank_cannot_leak_other_example_values_or_gradients():
    model = module(3)
    x = torch.randn(2,3,5)
    keys = torch.randn(2,9,4,requires_grad=True)
    values = torch.randn(2,9,3,requires_grad=True)
    before = model(x,keys,values)
    changed = values.detach().clone();changed[1]+=20
    after = model(x,keys,changed)
    torch.testing.assert_close(before[0],after[0],rtol=0,atol=0)
    assert not torch.allclose(before[1],after[1])
    before[0].sum().backward()
    assert values.grad[1].eq(0).all() and keys.grad[1].eq(0).all()


def test_group_address_loss_matches_probability_sum_and_all_slot_gradients():
    model = module(3,top_k=1)
    query_inputs = torch.randn(2,5,requires_grad=True)
    keys = torch.randn(9,4,requires_grad=True)
    targets = torch.tensor([0,2])
    actual = model.group_address_loss(query_inputs,targets,keys,reduction='none')
    logits = model.encode_query(query_inputs) @ torch.nn.functional.normalize(keys,dim=-1).T / model.temperature
    probabilities = logits.softmax(-1).reshape(2,3,3).sum(-1)
    torch.testing.assert_close(actual,-probabilities[torch.arange(2),targets].log())
    actual.sum().backward()
    assert bool((keys.grad.norm(dim=-1)>0).all()) # address supervision is deliberately dense
    assert query_inputs.grad.abs().sum()>0 and model.Wq.weight.grad.abs().sum()>0


def test_group_address_flattened_answer_positions_select_correct_independent_banks():
    model = module(3)
    model.set_feature_bank(torch.randn(2,9,5))
    query_inputs = torch.randn(5,5)
    rows = torch.tensor([0,0,1,1,1]);targets=torch.tensor([2,2,1,1,1])
    actual = model.group_address_loss(query_inputs,targets,batch_indices=rows,reduction='none')
    keys,_ = model.encoded_feature_bank()
    expected = torch.stack([model.group_address_loss(query_inputs[i:i+1],targets[i:i+1],keys[rows[i]]) for i in range(5)])
    torch.testing.assert_close(actual,expected)
    actual.mean().backward()
    assert model.Wk.weight.grad.abs().sum()>0
    with pytest.raises(ValueError):model.group_address_loss(query_inputs,targets)
    with pytest.raises(ValueError):model.group_address_loss(query_inputs,torch.full((5,),3),batch_indices=rows)


def populate(slots=3):
    store = GroupedVectorDB(4,3,.5,4,slots=slots,record_routes=True)
    generator=torch.Generator().manual_seed(173)
    for fact in ['a|b','c']:
        store.write_group(fact,torch.randn(slots,4,generator=generator),torch.randn(slots,3,generator=generator),1)
    return store


@pytest.mark.parametrize('failure',['shape','stale','lastslot_nan','lastslot_zerokey'])
def test_invalid_or_stale_group_write_is_atomic(failure):
    store=populate();before=store.hash();snapshot=store.snapshot()
    keys=torch.randn(3,4);values=torch.randn(3,3);timestamp=2
    if failure=='shape':values=values[:2]
    elif failure=='stale':timestamp=0
    elif failure=='lastslot_nan':values[-1,0]=float('nan')
    else:keys[-1]=0
    with pytest.raises(ValueError):store.write_group('a|b',keys,values,timestamp)
    assert store.hash()==before
    assert torch.equal(store.keys,snapshot['store']['keys']) and torch.equal(store.values,snapshot['store']['values'])


def test_successful_group_replacement_changes_only_target_and_logs_fact_slot_routes(tmp_path):
    store=populate();old=store.snapshot();oldhash=store.hash()
    store.write_group('a|b',torch.randn(3,4),torch.randn(3,3),2)
    assert store.hash()!=oldhash and store.fact_ids==('a|b','c') and len(store)==6
    assert store.timestamps==(2,2,2,1,1,1)
    assert torch.equal(store.keys[3:],old['store']['keys'][3:])
    assert torch.equal(store.values[3:],old['store']['values'][3:])
    before=store.hash();info=store.search(torch.randn(2,3,4))
    assert info['indices'].shape==(2,3,4) and len(info['fact_ids'])==6
    for i,indices in enumerate(info['indices'].reshape(-1,4).tolist()):
        assert info['fact_ids'][i]==[store.record_fact_ids[j] for j in indices]
    assert torch.equal(info['slot_ids'],info['indices']%3)
    assert len(store.route_log)==1 and store.hash()==before
    info['weights'].zero_()
    assert store.route_log[0]['weights'].sum()>0
    path=tmp_path/'bank.pt';store.save(path);restored=GroupedVectorDB.load(path)
    assert restored.hash()==store.hash() and restored.fact_ids==store.fact_ids
    assert restored.route_log==[] and not restored.record_routes
    store.clear_routes();assert not store.route_log


@pytest.mark.parametrize('failure',['missing_slot','reordered','partial_timestamp','slot_count'])
def test_group_snapshot_rejects_broken_atomic_group_contract(failure):
    state=populate().snapshot()
    if failure=='missing_slot':state['fact_ids'].pop()
    elif failure=='reordered':state['fact_ids'].reverse()
    elif failure=='partial_timestamp':state['store']['timestamps'][1]=2
    else:state['slots']=1
    with pytest.raises(ValueError):GroupedVectorDB.from_snapshot(state)


@pytest.mark.parametrize('slots',[1,3])
def test_cpu_vdb_matches_differentiable_retrieval_and_store_is_detached(slots):
    model=module(slots);features=torch.randn(4,slots,5,requires_grad=True)
    keys,values=model.encode_bank(features)
    store=model.new_store(record_routes=True)
    for index in range(4):model.write_group(store,str(index),features[index],timestamp=0)
    x=torch.randn(2,3,5)
    expected,expected_info=model(x,keys,values,return_info=True)
    actual,info=model.cpu_delta(x,store,return_info=True)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    assert torch.equal(info['indices'],expected_info['indices'])
    assert not store.keys.requires_grad and not store.values.requires_grad and not actual.requires_grad
    assert len(store.route_log)==1


def test_teacher_bypass_does_not_query_encode_or_touch_store(monkeypatch):
    model=module(3);store=model.new_store(record_routes=True)
    def forbidden(*args,**kwargs):raise AssertionError('teacher touched memory')
    monkeypatch.setattr(model,'encode_query',forbidden)
    monkeypatch.setattr(store,'search',forbidden)
    delta,info=model.cpu_delta(torch.randn(2,4,5),store,teacher=True,return_info=True)
    assert delta.eq(0).all() and info=={'teacher_bypass':True} and not store.route_log


def test_existing_backend_teacher_scope_bypasses_group_store_and_restores_student(monkeypatch):
    from test_context_distillation import make_backend
    backend=make_backend();model=module(3);store=model.new_store(record_routes=True)
    backend.vector_vera=model;backend.vector_store=store;backend.mode='vector_vera'
    marker={'keep':'student'};backend.last_retrieval=marker
    def forbidden(*args,**kwargs):raise AssertionError('teacher touched memory')
    monkeypatch.setattr(model,'encode_query',forbidden)
    monkeypatch.setattr(store,'search',forbidden)
    teacher=backend.forward_sequences(['a'],[[1,2]],contexts=['teacher-only context'],teacher=True)
    assert not teacher['logits'].requires_grad
    assert backend.mode=='vector_vera' and backend.vector_store is store and backend.last_retrieval is marker
    assert not store.route_log


def test_empty_cpu_and_tensor_bank_return_exact_noop_and_empty_routes():
    model=module(3);x=torch.randn(2,4,5);store=model.new_store(record_routes=True)
    actual,info=model.cpu_delta(x,store,return_info=True)
    expected,tensor_info=model(x,torch.empty(0,4),torch.empty(0,3),return_info=True)
    assert actual.eq(0).all() and expected.eq(0).all()
    assert info['indices'].shape==tensor_info['indices'].shape==(2,4,0)
    assert info['fact_ids']==[[] for _ in range(8)]


def test_architecture_and_statistics_roundtrip_reject_cross_factor_silent_load():
    model=module(3,True);clone=ReconstructionVeRA(**model.configuration())
    clone.load_state_dict(model.state_dict())
    for key,value in model.state_dict().items():assert torch.equal(value,clone.state_dict()[key])
    x=torch.randn(3,5)
    torch.testing.assert_close(model.encode_value(x),clone.encode_value(x),rtol=0,atol=0)
    for wrong in [module(1,True),module(3,False)]:
        with pytest.raises(RuntimeError):wrong.load_state_dict(model.state_dict())


def test_learned_B_can_update_while_A_and_fixed_control_stay_fixed():
    learned=module(1,True);fixed=module(1,False)
    x=torch.randn(3,5);values=torch.randn(3,3)
    before_A=learned.A.clone();before_B=learned.B.detach().clone()
    torch.testing.assert_close(learned.delta_from_value(x,values),fixed.delta_from_value(x,values),rtol=0,atol=0)
    optimizer=torch.optim.SGD([learned.B],lr=.1)
    learned.delta_from_value(x,values).square().sum().backward();optimizer.step()
    assert not torch.equal(learned.B,before_B) and torch.equal(learned.A,before_A)
    assert torch.equal(fixed.B,before_B) and fixed.B.grad is None
