"""Post-hoc diagnostics do not turn route position into gold-word alignment."""
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    'qkv_diagnostics', Path(__file__).resolve().parents[1] / 'scripts/diagnose_qkv.py')
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def fixture(tmp_path, *, world='C', position=0, prediction=None, masses=None):
    dataset_path = tmp_path / 'qkv_prepare_fixture/features/dataset.json'
    dataset_path.parent.mkdir(parents=True)
    a = ['alpha', 'beta', 'gamma']
    c = a.copy(); c[position] = 'novel'
    d = a.copy(); d[position] = 'fresh'
    row = dict(id='target', a=' '.join(a), b='delta epsilon zeta',
               c=' '.join(c), d=' '.join(d), swap_a='eta theta iota',
               provenance={'edited_word_index': position})
    data = dict(train={'payloads': [row['a'], row['b']]}, known=[row], dev=[], confirm=[])
    directory = tmp_path / 'qkv_eval_fixture/job'
    directory.mkdir(parents=True)
    manifest = dict(protocol='qkv-attention-memory-v1', complete=True,
                    backbone_unchanged=True, configuration=dict(stage='eval', eval_part='development',
                        checkpoint='/runs/rebind_grouped_seed81042/last.pt', cache='/features/known.pt', seed=81042),
                    result=dict(complete=True, architecture={'read_mode': 'grouped'}))
    answer = row[{'B': 'b', 'C': 'c', 'D': 'd', 'SWAP': 'swap_a'}[world]]
    prediction = answer if prediction is None else prediction
    masses = masses or [[.1, .2, .3], [.3, .2, .1], [0., 0., 0.], [.2, .3, .1]]
    tokens = list(range(10, 10 + len(masses)))
    raw = [dict(id='target', phase='CC', world=world, condition='real', role='updated',
                answer=answer, prediction=prediction,
                em=int(diagnostics.normalize_answer(answer) == diagnostics.normalize_answer(prediction)),
                first_fact_recall_at_1=1, first_fact_recall_at_read=1,
                generation_tokens=len(tokens), generated_token_ids=tokens,
                trace=[dict(token_id=t, target_slot_mass=m, target_fact_mass=sum(m))
                       for t, m in zip(tokens, masses)])]

    def save():
        dataset_path.write_text(json.dumps(data))
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        (directory / 'predictions.jsonl').write_text('\n'.join(map(json.dumps, raw)) + '\n')
    save()
    return data, manifest, raw, save


@pytest.mark.parametrize('world', ['C', 'D'])
@pytest.mark.parametrize('position', [0, 1, 2])
def test_changed_word_uses_declared_position_and_complete_generation_trace(tmp_path, world, position):
    _, _, raw, _ = fixture(tmp_path, world=world, position=position)
    report = diagnostics.analyze(tmp_path)
    group = report['records'][0]['groups'][0]
    metrics = group['metrics']
    trace = raw[0]['trace']
    assert metrics['em'] == metrics['edited_word_correct'] == metrics['both_unchanged_words_correct'] == 1
    assert metrics['first_edited_slot_mass'] == trace[0]['target_slot_mass'][position]
    assert metrics['mean_edited_slot_mass'] == pytest.approx(
        sum(t['target_slot_mass'][position] for t in trace) / len(trace))
    assert metrics['max_edited_slot_mass'] == max(t['target_slot_mass'][position] for t in trace)
    assert group['edited_positions'][str(position)] == dict(count=1, em=1, edited_word_correct=1)
    for other in set(range(3)) - {position}:
        assert group['edited_positions'][str(other)] == dict(count=0, em=None, edited_word_correct=None)
    assert len(report['input_sha256']) == 3  # dataset, manifest and predictions
    assert any('no gold-word alignment' in line for line in report['limitations'])


@pytest.mark.parametrize('prediction,valid,correct', [
    ('', 0, 0), ('novel beta', 0, 0), ('novel beta gamma extra', 0, 0),
    ('novel wrong gamma', 1, 0), ('NOVEL, beta gamma!', 1, 1),
])
def test_word_scores_require_exactly_three_normalized_words(tmp_path, prediction, valid, correct):
    fixture(tmp_path, prediction=prediction)
    metrics = diagnostics.analyze(tmp_path)['records'][0]['groups'][0]['metrics']
    assert metrics['exactly_three_words'] == valid and metrics['em'] == correct
    if not valid:
        assert all(metrics[f'word_{i}_correct'] == 0 for i in range(3))
        assert metrics['edited_word_correct'] == metrics['both_unchanged_words_correct'] == 0
    elif prediction == 'novel wrong gamma':
        assert metrics['edited_word_correct'] == 1 and metrics['both_unchanged_words_correct'] == 0


def test_old_training_payload_is_separate_from_new_answer_success(tmp_path):
    fixture(tmp_path, prediction='alpha beta gamma')
    metrics = diagnostics.analyze(tmp_path)['records'][0]['groups'][0]['metrics']
    assert metrics['output_is_training_payload'] == metrics['output_is_own_old_a'] == 1
    assert metrics['em'] == metrics['edited_word_correct'] == metrics['output_is_own_old_b'] == 0
    assert metrics['both_unchanged_words_correct'] == 1


def test_macro_trajectory_mass_is_not_pooled_token_mass(tmp_path):
    import copy
    data, _, raw, save = fixture(tmp_path, world='B', masses=[[0., 0., 0.]])
    other = copy.deepcopy(data['known'][0]); other['id'] = 'other'; data['known'].append(other)
    second = copy.deepcopy(raw[0]); second.update(id='other', generation_tokens=3, generated_token_ids=[4, 5, 6])
    second['trace'] = [dict(token_id=t, target_slot_mass=[1., 0., 0.], target_fact_mass=1.) for t in [4, 5, 6]]
    raw.append(second); save()
    group = diagnostics.analyze(tmp_path)['records'][0]['groups'][0]
    assert group['count'] == 2
    assert group['metrics']['mean_target_fact_mass'] == .5  # pooled-token average would be .75
    assert group['edited_positions'] == {} and 'mean_edited_slot_mass' not in group['metrics']


@pytest.mark.parametrize('attack', ['wrong_gold', 'wrong_em', 'bad_position', 'token_mismatch', 'empty_trace'])
def test_inconsistent_inputs_fail_closed(tmp_path, attack):
    data, _, raw, save = fixture(tmp_path)
    if attack == 'wrong_gold': raw[0]['answer'] = 'different gold answer'
    elif attack == 'wrong_em': raw[0]['em'] = 0
    elif attack == 'bad_position': data['known'][0]['provenance']['edited_word_index'] = True
    elif attack == 'token_mismatch': raw[0]['trace'][0]['token_id'] = 99
    else: raw[0]['trace'] = []
    save()
    with pytest.raises(ValueError): diagnostics.analyze(tmp_path)


def test_smoke_and_incomplete_jobs_are_not_descriptive_formal_results(tmp_path):
    _, manifest, _, save = fixture(tmp_path)
    manifest['configuration']['eval_part'] = 'smoke'; save()
    assert diagnostics.analyze(tmp_path)['records'] == []
    manifest['configuration']['eval_part'] = 'development'; manifest['complete'] = False; save()
    assert diagnostics.analyze(tmp_path)['records'] == []
