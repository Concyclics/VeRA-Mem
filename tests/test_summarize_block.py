"""Independent scalar/raw audit contracts for the block-reader experiment."""
import copy
import importlib.util
import json
from pathlib import Path
import hashlib

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


audit = load('block_summary_test', ROOT/'scripts/summarize_block.py')
# Reuse only synthetic fixture construction; no old tests or production evaluator run.
fixture_helpers = load('qkv_fixture_helpers', ROOT/'tests/test_summarize_qkv.py')


def architecture(mode='diagonal', seed=91042):
    return dict(fixture_helpers.architecture(seed=seed), block_mode=mode, slot_weighting='uniform',
                block_version=1, factor_epsilon=1e-6, residual_outer=True)


def raw_fixture(tmp_path, split='known', part='development', mode='diagonal', wrong_local=False):
    directory, manifest, rows, raw, pairs, summary, save = fixture_helpers.raw_fixture(
        tmp_path, split=split, part=part, wrong_local=wrong_local)
    manifest['protocol'] = audit.RUN_PROTOCOL; summary['protocol'] = 'block-online-cpu-v1'
    summary['architecture'] = architecture(mode)
    for row in raw:
        row['injection_input'] = 'full_selected_value_block'
        row['sparse_values_transferred_per_token'] = row['actual_read_slots']*64
        index = next(i for i, r in enumerate(rows) if r['id'] == row['id'])
        for trace in row['trace']:
            width = len(trace['indices'][0]); trace['weights'] = [[1/3]*width]
            trace['query'] = [[1. if width else 0.]*64]
            trace['operator_weighting'] = 'uniform'
            trace['weight_interpretation'] = 'uniform membership; not signed outer coefficients'
            trace['target_slot_mass'] = [sum(w for j, w in zip(trace['indices'][0], trace['weights'][0]) if j == 3*index+s) for s in range(3)]
            trace['target_fact_mass'] = sum(trace['target_slot_mass'])
        row['first_target_slot_mass'] = row['trace'][0]['target_slot_mass'][:]
        row['first_target_fact_mass'] = row['trace'][0]['target_fact_mass']
    save(); return directory, manifest, rows, raw, pairs, summary, save


@pytest.mark.parametrize('split,part,calls,writes', [
    ('known', 'development', 224, 48), ('dev', 'development', 176, 32),
    ('known', 'confirmation', 96, 16), ('confirm', 'confirmation', 704, 128)])
@pytest.mark.parametrize('mode', audit.MODES)
def test_every_grid_mode_recomputed_from_raw(tmp_path, split, part, calls, writes, mode):
    d, m, rows, *_ = raw_fixture(tmp_path, split, part, mode)
    result = audit.audit_evaluation(d, m, rows, split, 'rebind')
    assert result['arm'] == 'rebind_'+mode and result['seed'] == 91042
    assert result['costs']['generation_calls'] == calls
    assert result['independent_interventions'] == result['restoration_writes'] == writes
    assert result['real_group_write_events'] == 2*writes and result['bytes_per_fact'] == 1536
    assert all(g['locality_joint'] == 1 for g in result['groups'].values())


@pytest.mark.parametrize('attack', ['em', 'pairs', 'summary', 'query_missing', 'query_short', 'query_nonfinite', 'query_zero',
    'nonuniform', 'interpretation', 'slot_mass', 'fact_width', 'token_id', 'token_count', 'phase', 'r1', 'trigger',
    'answer', 'bank_hash', 'input_type', 'transfer_count', 'duplicate', 'missing', 'parameter_count_mode'])
def test_corrupt_evidence_rejected(tmp_path, attack):
    d, m, rows, raw, pairs, summary, save = raw_fixture(tmp_path)
    row, trace = raw[0], raw[0]['trace'][0]
    if attack == 'em': row['em'] = 0
    if attack == 'pairs': pairs[0]['update_restore_em'] = 0
    if attack == 'summary': summary['groups']['CC_C']['updated_em'] = 0
    if attack == 'query_missing': trace.pop('query')
    if attack == 'query_short': trace['query'][0].pop()
    if attack == 'query_nonfinite': trace['query'][0][0] = float('nan')
    if attack == 'query_zero': trace['query'][0] = [0.]*64
    if attack == 'nonuniform': trace['weights'] = [[.2, .6, .2]]
    if attack == 'interpretation': trace['weight_interpretation'] = 'attention output contribution'
    if attack == 'slot_mass': trace['target_slot_mass'][0] = 0
    if attack == 'fact_width': trace['indices'][0][2] = 9
    if attack == 'token_id': trace['token_id'] = 999
    if attack == 'token_count': row['generation_tokens'] = 10
    if attack == 'phase': trace['phase'] = 'decode'
    if attack == 'r1': row['first_fact_recall_at_1'] = 0
    if attack == 'trigger': raw[16]['trigger_world'] = 'C'
    if attack == 'answer': row['answer'] = 'the wrong answer'
    if attack == 'bank_hash': raw[16]['bank_hash'] = raw[0]['bank_hash']
    if attack == 'input_type': row['injection_input'] = 'pooled_vector'
    if attack == 'transfer_count': row['sparse_values_transferred_per_token'] = 64
    if attack == 'duplicate': raw.append(copy.deepcopy(raw[-1]))
    if attack == 'missing': raw.pop()
    if attack == 'parameter_count_mode': summary['architecture']['residual_outer'] = False
    save()
    with pytest.raises((ValueError, KeyError)):
        audit.audit_evaluation(d, m, rows, 'known', 'static')


def test_locality_wrong_to_wrong_is_not_success(tmp_path):
    d, m, rows, *_ = raw_fixture(tmp_path, wrong_local=True)
    result = audit.audit_evaluation(d, m, rows, 'known', 'static')
    assert result['groups']['CC_B']['locality_equal'] == 1
    assert result['groups']['CC_B']['locality_joint'] == 15/16


def test_strict_word_diagnostics_require_exactly_three_and_joint_preserved_words():
    source = dict(id='f', a='oak glass horse', b='pine stone bear', provenance={'edited_word_index': 1})
    examples = [('oak gold horse', 1), ('oak gold horse extra', 0), ('oak gold bear', 0), ('oak stone horse', 0)]
    rows = [dict(id='f', answer='oak gold horse', prediction=p, em=em, role='updated', world='C', phase='CC', condition='real', budget_hit=False) for p, em in examples]
    g = audit.answer_diagnostics(rows, {'f': source}, ['oak glass horse'])[0]
    assert g['counts']['em'] == 1 and g['counts']['full_answer_containment'] == 2
    assert g['counts']['edited_word_correct'] == 2 and g['counts']['both_unchanged_words_correct'] == 2
    assert g['counts']['exactly_three_words'] == 3 and g['count'] == 4
    assert g['edited_positions']['1']['count'] == 4


def training_fixture(tmp_path, mode='diagonal', regime='rebind'):
    d = tmp_path/'training'; d.mkdir(); raw = []
    totals = dict(target_exposures=0, gold_tokens=0, input_positions=0, padded_input_positions=0, backbone_calls=0)
    h = hashlib.sha256(); outer = mode != 'diagonal'
    for step in range(2048):
        ep = audit.episode(step, regime, 91042)
        aids = [ep[k][j] for k in ('a_payload_indices', 'b_payload_indices') for j in ep['local_targets']]
        gl = [3+a%4 for a in aids]; il = [g+10+e%3 for e, g in zip(ep['targets']*2, gl)]
        raw.append(dict(step=step+1, episode=ep, metrics=dict(ce=2. if step < 1920 else 1., address=2., loss=2.4 if step < 1920 else 1.4),
            gradient_norms=[1.]*(4+outer), parameter_gradient_norms={'b': 1.}, input_lengths=il, gold_token_lengths=gl, elapsed_seconds=step+1.))
        h.update(json.dumps(ep, sort_keys=True).encode()); totals['target_exposures'] += 8
        totals['gold_tokens'] += sum(gl); totals['input_positions'] += sum(il)
        totals['padded_input_positions'] += 16*max(il); totals['backbone_calls'] += 1
    status = dict(complete=True, updates=2048, regime=regime, architecture=architecture(mode),
        trainable_parameters=2034368+8256*outer, schedule_sha256=h.hexdigest(), totals=totals,
        learning_rates=[1e-4, 3e-4, .005, 3e-4]+([3e-4] if outer else []), elapsed_seconds=2049.)
    manifest = dict(result=status, configuration=dict(stage='train', updates=2048, regime=regime, block_mode=mode, seed=91042), cache_sha256='b'*64)
    def save():
        for name, obj in (('manifest.json', manifest), ('training_status.json', status)): (d/name).write_text(json.dumps(obj))
        (d/'training.jsonl').write_text('\n'.join(map(json.dumps, raw)))
        for name in ('initial.pt', 'last.pt', 'step_1024.pt', 'step_2048.pt'): (d/name).write_bytes(name.encode())
    save(); return d, manifest, raw, status, save


@pytest.mark.parametrize('mode', audit.MODES)
def test_training_exact_exposure_costs_and_convergence(tmp_path, mode):
    d, m, raw, status, _ = training_fixture(tmp_path, mode)
    r = audit.audit_training(d, m, {})
    assert r['costs']['target_exposures'] == 16384 and r['costs']['backbone_calls'] == 2048
    assert set(r['exposure']['target_counts']) == set(r['exposure']['supervised_payload_counts']) == {256}
    assert sum(r['exposure']['physical_payload_bank_residency']) == 2048*16*16
    assert r['convergence']['last_128']['ce']['mean'] == 1 and r['convergence']['first_128']['ce']['mean'] == 2
    assert r['trainable_parameters'] == 2034368+(8256 if mode != 'diagonal' else 0)


@pytest.mark.parametrize('attack', ['episode', 'token_ledger', 'loss', 'gradients', 'lr', 'parameter_count', 'cost', 'final_step', 'seed'])
def test_training_tampering_rejected(tmp_path, attack):
    d, m, raw, status, save = training_fixture(tmp_path, 'block_outer')
    if attack == 'episode': raw[7]['episode']['mapping'][0][0] = 999
    if attack == 'token_ledger': raw[7]['gold_token_lengths'][0] += 1
    if attack == 'loss': raw[7]['metrics']['loss'] += 1
    if attack == 'gradients': raw[7]['gradient_norms'].pop()
    if attack == 'lr': status['learning_rates'][-1] = 1e-3
    if attack == 'parameter_count': status['trainable_parameters'] -= 64
    if attack == 'cost': status['totals']['padded_input_positions'] -= 16
    if attack == 'final_step': raw[-1]['step'] = 2047
    if attack == 'seed': m['configuration']['seed'] = 91043
    save()
    with pytest.raises(ValueError): audit.audit_training(d, m, {})


def test_inherited_schedule_is_exact_current_runtime():
    from vera_mem.block_data import episode
    for seed in audit.SEEDS:
        for step in (0, 7, 8, 2047):
            for regime in ('static', 'rebind'): assert audit.episode(step, regime, seed) == episode(step, regime, seed=seed)


def gate_records():
    out = []
    for regime in ('static', 'rebind'):
        for seed in audit.SEEDS:
            for mode in audit.MODES:
                for part in ('development', 'confirmation'):
                    world = 'C' if part == 'development' else 'D'; score = .5 if mode == 'block_outer' else .25
                    g = dict(count=16, pair_em=.75, updated_em=score, update_restore_em=score, shuffle_em=0., empty_em=0.)
                    groups = {'CC_'+world: g}
                    if part == 'development': groups['CC_B'] = dict(g, pair_em=.75)
                    out.append(dict(kind='evaluation', arm=regime+'_'+mode, seed=seed, split='known', part=part,
                        groups=groups, teacher_qualification={'paired': {k: dict(count=16, both_correct=16) for k in groups}}))
    return out


def test_all_three_seed_matched_gate_no_selection_or_automatic_expansion():
    records = gate_records(); gate = audit.continuation_gate(records)
    assert all(r['passed'] for r in gate['regimes']) and gate['automatic_corpus_expansion'] is False
    fail = next(r for r in records if r['arm'] == 'static_block_outer' and r['seed'] == 91044 and r['part'] == 'confirmation')
    fail['groups']['CC_D']['update_restore_em'] -= 1/16
    gate = audit.continuation_gate(records)
    assert not gate['regimes'][0]['passed'] and gate['regimes'][1]['passed']
    assert gate['supports_further_block_reader_investment']


@pytest.mark.parametrize('attack', ['teacher', 'ab', 'shuffle', 'empty', 'pooled', 'missing'])
def test_gate_rejects_single_seed_or_control_failures(attack):
    records = gate_records()
    r = next(r for r in records if r['arm'] == 'static_block_outer' and r['seed'] == 91042 and r['part'] == 'development')
    if attack == 'teacher': r['teacher_qualification']['paired']['CC_C']['both_correct'] = 15
    if attack == 'ab': r['groups']['CC_B']['pair_em'] = 11/16
    if attack == 'shuffle': r['groups']['CC_C']['shuffle_em'] = 5/16
    if attack == 'empty': r['groups']['CC_C']['empty_em'] = 5/16
    if attack == 'pooled':
        p = next(p for p in records if p['arm'] == 'static_pooled_outer' and p['seed'] == 91042 and p['part'] == 'development')
        p['groups']['CC_C']['update_restore_em'] = 5/16
    if attack == 'missing': records.remove(r)
    assert not audit.continuation_gate(records)['regimes'][0]['passed']


def test_inventory_requires_all_eighteen_models_and_seventy_two_evals():
    records = []
    for arm in audit.ARMS:
        for seed in audit.SEEDS:
            records.append(dict(kind='training', arm=arm, seed=seed, run_dir=f'{arm}/{seed}/train'))
            for split, part in (('known', 'development'), ('dev', 'development'), ('known', 'confirmation'), ('confirm', 'confirmation')):
                records.append(dict(kind='evaluation', arm=arm, seed=seed, split=split, part=part, run_dir=f'{arm}/{seed}/{split}/{part}'))
    assert audit.completeness(records)['complete'] and audit.completeness(records)['observed'] == 90
    assert not audit.completeness(records[:-1])['complete']
    with pytest.raises(ValueError, match='Duplicate'): audit.completeness(records+[records[-1]])


def test_supplementary_files_bind_actual_summary_and_script_hash(tmp_path):
    report = dict(records=[], partial=True, formal_jobs={'complete': False}, teacher_scope={'complete': False}, costs={},
        continuation=audit.continuation_gate([]), endpoints=[], limitations=['fixture'])
    output = tmp_path/'summary.json'; audit.write_outputs(report, output)
    for name in ('training_convergence.json', 'answer_diagnostics.json', 'key_facts.json'):
        value = audit.read_json(tmp_path/name)
        assert value['partial'] and value['sources']['summary_sha256'] == audit.sha(output)
        assert value['sources']['script_sha256'] == audit.sha(audit.__file__)


def test_saved_query_contract_does_not_claim_global_top1_verification(tmp_path):
    d, m, rows, raw, _, _, save = raw_fixture(tmp_path)
    # JSON cannot recompute cosine against tensor-bank keys. The independent
    # tensor audit, not this script, proves global winning-group correctness.
    raw[0]['trace'][0]['query'][0][0] = -3.; save()
    assert audit.audit_evaluation(d, m, rows, 'known', 'static')['costs']['generation_calls'] == 224


def test_resolve_remote_run_inputs_and_reject_missing(tmp_path):
    p = tmp_path/'block_prepare_20261006/features/train.pt'; p.parent.mkdir(parents=True); p.write_bytes(b'data')
    assert audit.resolve_run_input('/ssd3/chenhan/VeRA-Mem-Workspace/runs/block_prepare_20261006/features/train.pt', tmp_path) == p.resolve()
    with pytest.raises(ValueError): audit.resolve_run_input('/unknown/train.pt', tmp_path)


def test_teacher_failed_raw_case_retains_full_denominator(tmp_path):
    d, m, rows, raw, summary, save = fixture_helpers.teacher_fixture(tmp_path, True)
    m['protocol'] = audit.RUN_PROTOCOL; save()
    result = audit.common.audit_teacher(d, m, rows, 'train')
    assert not result['qualified'] and result['paired']['CC_B']['count'] == 64
    assert result['paired']['CC_B']['both_correct'] == 63


def registration_fixture(tmp_path):
    protocol = tmp_path/'block_protocol.md'; protocol.write_text('frozen design')
    caches = {s: dict(cache_sha256=audit.common.digest(s)) for s in ('train', 'known', 'dev', 'confirm')}
    registration = dict(protocol='block-preregistration-v1', teacher_preflight_passed=True, smoke_passed=True,
        selection_policy='all_18_final_checkpoints_no_selection', protocol_document_sha256=audit.sha(protocol),
        arms=list(audit.ARMS), seeds=list(audit.SEEDS), updates=2048, model_revision=audit.MODEL_REVISION,
        source_python_sha256={'src/vera_mem/block_run.py': 'a'*64}, plans_sha256={'plan.json': 'b'*64},
        feature_result={'caches': caches}, feature_cache_sha256={k+'.pt': v['cache_sha256'] for k, v in caches.items()})
    path = tmp_path/'registration.json'; path.write_text(json.dumps(registration))
    return path, protocol, registration


@pytest.mark.parametrize('attack', [None, 'budget', 'selection', 'model', 'cache', 'protocol', 'seed'])
def test_registration_accepts_only_fixed_plan_and_consistent_cache_hashes(tmp_path, attack):
    path, protocol, value = registration_fixture(tmp_path)
    if attack == 'budget': value['updates'] = 4096
    if attack == 'selection': value['selection_policy'] = 'best_dev_only'
    if attack == 'model': value['model_revision'] = 'f'*40
    if attack == 'cache': value['feature_cache_sha256']['train.pt'] = 'c'*64
    if attack == 'protocol': protocol.write_text('edited after sealing')
    if attack == 'seed': value['seeds'][-1] += 1
    path.write_text(json.dumps(value))
    if attack:
        with pytest.raises(ValueError): audit.load_registration(path, protocol)
    else: assert audit.load_registration(path, protocol) == value


def test_prepare_binds_tensor_hashes_to_all_raw_json_and_train_only_projection(tmp_path):
    from vera_mem.block_data import dataset
    data = dataset(); d = tmp_path/'features'; d.mkdir(); caches = {}
    (d/'dataset.json').write_text(json.dumps(data))
    for split in ('train', 'known', 'dev', 'confirm'):
        (d/(split+'.json')).write_text(json.dumps(data[split])); (d/(split+'.pt')).write_bytes(split.encode())
        caches[split] = dict(cache_sha256=audit.sha(d/(split+'.pt')), **(
            dict(entities=64, payloads=128, contextual_observations=8192) if split == 'train' else dict(records=16)))
    manifest = dict(result=dict(dataset_sha256=audit.sha(d/'dataset.json'), caches=caches))
    result = audit.prepare_cache(d, manifest)
    assert len(result) == 4 and sorted(r['split'] for r in result.values()) == ['confirm', 'dev', 'known', 'train']
    (d/'train.pt').write_bytes(b'changed tensor')
    with pytest.raises(ValueError, match='tensor cache digest'): audit.prepare_cache(d, manifest)


@pytest.mark.parametrize('attack', [None, 'plan_option', 'plan_hash', 'source', 'protocol_copy'])
def test_registered_child_plan_source_and_protocol_are_hash_bound(tmp_path, attack):
    directory = tmp_path/'suite'/'child'; directory.mkdir(parents=True)
    docs = directory.parent/'source/docs'; docs.mkdir(parents=True)
    (docs/'block_protocol.md').write_text('frozen')
    manifest = dict(stage='train', cache_sha256='c'*64, source_sha256={'block_run.py': 'a'*64},
        configuration={'stage': 'train', 'updates': 2048})
    plan = [dict(name='child', entry='block', arguments=['--stage', 'train', '--updates', '2048'])]
    (directory.parent/'plan.json').write_text(json.dumps(plan)); ph = audit.sha(directory.parent/'plan.json')
    (directory.parent/'suite.json').write_text(json.dumps({'plan_sha256': ph}))
    reg = dict(feature_result={'caches': {'train': {'cache_sha256': 'c'*64}}},
        source_python_sha256={'src/vera_mem/block_run.py': 'a'*64},
        protocol_document_sha256=audit.sha(docs/'block_protocol.md'), plans_sha256={'suite.json': ph})
    if attack == 'plan_option': manifest['configuration']['updates'] = 1024
    if attack == 'plan_hash': reg['plans_sha256']['suite.json'] = 'd'*64
    if attack == 'source': manifest['source_sha256']['block_run.py'] = 'e'*64
    if attack == 'protocol_copy': (docs/'block_protocol.md').write_text('changed')
    if attack:
        with pytest.raises(ValueError): audit.registered_run(directory, manifest, reg)
    else: audit.registered_run(directory, manifest, reg)


@pytest.mark.parametrize('prefix', ['', '/remote/runs/'])
def test_preflight_receipt_accepts_exact_relative_or_absolute_path_suffix(tmp_path, prefix):
    directory = tmp_path/'block_preflight_20261006/teacher_train'; directory.mkdir(parents=True)
    manifest = dict(stage='teacher', cache_sha256='a'*64, configuration={'eval_part': 'preflight'})
    (directory/'manifest.json').write_text(json.dumps(manifest))
    key = prefix+'block_preflight_20261006/teacher_train/manifest.json'
    registration = dict(feature_result={'caches': {'train': {'cache_sha256': 'a'*64}}},
        preflight_artifacts_sha256={key: audit.sha(directory/'manifest.json')})
    audit.registered_run(directory, manifest, registration)
    registration['preflight_artifacts_sha256'][key] = 'f'*64
    with pytest.raises(ValueError, match='receipt'): audit.registered_run(directory, manifest, registration)
