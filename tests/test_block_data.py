"""Fresh block-memory data must not reuse either preceding confirmation set."""
from collections import Counter
import copy
import json
import random
import re

import pytest

from vera_mem import block_data as data
from vera_mem import qkv_data as qkv
from vera_mem import reconstruction_data as recon


@pytest.fixture(scope='module')
def raw():
    return data.dataset()


def history_sets():
    old = recon.datasets(seed=101042)
    prior = qkv.dataset(seed=121042)
    entities = {r['entity'] for rows in old.values() for r in rows}
    answers = {r[f] for rows in old.values() for r in rows for f in ('a', 'b', 'c', 'd')}
    entities.update(prior['train']['entities'])
    answers.update(prior['train']['payloads'])
    for split in ('known', 'dev', 'confirm'):
        entities.update(r['entity'] for r in prior[split])
        answers.update(r[f] for r in prior[split] for f in data.FIELDS)
    return entities, answers


def test_entire_new_dataset_excludes_both_historical_identity_and_answer_unions(raw):
    old_entities, old_answers = history_sets()
    assert len(old_entities) == 208 and len(old_answers) == 608
    entities = set(raw['train']['entities']) | {r['entity'] for s in ('dev', 'confirm') for r in raw[s]}
    answers = set(raw['train']['payloads']) | {r[f] for r in raw['known'] for f in ('c', 'd')}
    assert len(entities) == 96 and entities.isdisjoint(old_entities)
    assert len(answers) == 160 and answers.isdisjoint(old_answers)
    proof = data.validate_dataset(raw)
    assert proof['historical_exclusion_count'] == 608 and proof['historical_entity_exclusion_count'] == 208
    assert proof['novel_complete_payloads'] == 32
    assert raw['provenance']['history'] == data.history_manifest()
    assert [x['seed'] for x in data.history_manifest()['sources']] == [101042, 121042]


def test_historical_generators_remain_byte_equivalent_before_and_after_new_generation():
    # Locked semantic JSON digests come from the preceding immutable datasets.
    expected = ('033c102e192fe700ff6ad3703b73e8114a5480716cf7f0c2cd24d70edae8d301',
                'fc5fbc089ed63f9f5f74613c0678cd3bd7ec28f247465eab9f8e4717e711fdb6')
    before = (data.digest(recon.datasets(seed=101042)), data.digest(qkv.dataset(seed=121042)))
    assert before == expected
    new = data.dataset()
    new['provenance']['history']['sources'][0]['entity_count'] = -1
    new['train']['payloads'][0] = 'mutated data'
    assert (data.digest(recon.datasets(seed=101042)), data.digest(qkv.dataset(seed=121042))) == expected
    assert data.history_manifest()['sources'][0]['entity_count'] == 112
    assert data.dataset()['train']['payloads'][0] != 'mutated data'


def test_fixed_dimensions_balance_and_protocol_identity(raw):
    train = raw['train']
    assert raw['seed'] == train['seed'] == 221042
    assert raw['protocol'] == train['protocol'] == data.PROTOCOL != qkv.PROTOCOL
    assert len(train['entities']) == len(train['rows']) == 64 and len(train['payloads']) == 128
    assert all(r['id'].startswith('block-') for r in train['rows'])
    for position, vocab in enumerate(data.VOCABULARIES):
        assert Counter(p.split()[position] for p in train['payloads']) == Counter({w: 8 for w in vocab})
    assert data.validate_dataset(raw)['edit_position_counts'] == {0: 6, 1: 5, 2: 5}


def test_factorial_pairing_and_novel_edit_contract(raw):
    train = raw['train']; new_answers = set()
    groups = {s: {r['entity'] for r in raw[s]} for s in ('known', 'dev', 'confirm')}
    assert groups['known'] <= set(train['entities'])
    assert groups['dev'].isdisjoint(train['entities']) and groups['confirm'].isdisjoint(train['entities'])
    assert groups['dev'].isdisjoint(groups['confirm'])
    for i, known in enumerate(raw['known']):
        assert known['a'] == train['payloads'][2*i] and known['b'] == train['payloads'][2*i+1]
        for field in ('c', 'd'):
            assert known[field] not in set(train['payloads']) | new_answers
            new_answers.add(known[field])
            assert [j for j, (a, b) in enumerate(zip(known['a'].split(), known[field].split())) if a != b] == [i % 3]
        for split in ('known', 'dev', 'confirm'):
            row = raw[split][i]
            assert [row[f] for f in data.FIELDS] == [known[f] for f in data.FIELDS]
            donor = raw[split][i ^ 1]
            assert row['swap_a'] == donor['a'] and row['provenance']['swap_donor_id'] == donor['id']
            assert row['entity'] in row['supports'][4][0] and donor['entity'] not in row['supports'][4][0]
    assert len(new_answers) == 32


def test_model_text_contains_only_legitimate_observation_or_entity_query(raw):
    vocab = set(sum(data.VOCABULARIES, ()))
    for split in ('known', 'dev', 'confirm'):
        for row in raw[split]:
            for question in row['questions']:
                assert row['entity'] in question and row['id'] not in question
                assert vocab.isdisjoint(re.findall(r'[a-z]+', question.casefold()))
            for field, supports in zip(data.FIELDS, row['supports']):
                for support in supports:
                    start, stop = data.payload_span(support, row[field])
                    assert support[start:stop] == row[field]
                    assert row['entity'] in support and row['id'] not in support


def test_training_whitelist_does_not_export_probe_labels_entities_or_styles(raw):
    train = data.training_data(raw)
    assert set(train) == {'protocol', 'seed', 'entities', 'payloads', 'rows'}
    assert data.validate_training_data(train)['probe_fields_absent']
    wire = json.dumps(train)
    for split in ('known', 'dev', 'confirm'):
        for row in raw[split]:
            assert row['c'] not in wire and row['d'] not in wire and row['questions'][1] not in wire
            if split != 'known': assert row['entity'] not in wire
    assert all(r['worlds'] == ['A', 'B'] and len(r['questions']) == 1 and list(map(len, r['supports'])) == [1, 1] for r in train['rows'])
    train['rows'][0]['a'] = 'changed'
    assert raw['train']['rows'][0]['a'] != 'changed'


@pytest.mark.parametrize('attack', ['extra_c', 'heldout', 'eval_id', 'wrong_worlds', 'new_entity', 'bad_balance'])
def test_training_data_rejects_probe_smuggling_and_unbalanced_inputs(raw, attack):
    train = copy.deepcopy(raw['train'])
    if attack == 'extra_c': train['c'] = raw['known'][0]['c']
    elif attack == 'heldout': train['rows'][0]['questions'].append(raw['known'][0]['questions'][1])
    elif attack == 'eval_id': train['rows'][0]['provenance']['eval_id'] = raw['dev'][0]['id']
    elif attack == 'wrong_worlds': train['rows'][0]['worlds'].append('D')
    elif attack == 'new_entity': train['entities'][0] = raw['dev'][0]['entity']
    else: train['payloads'][0] = train['payloads'][1]
    with pytest.raises(ValueError): data.validate_training_data(train)


@pytest.mark.parametrize('source', ['reconstruction', 'qkv'])
def test_balanced_valid_length_payload_contamination_is_still_rejected(raw, source):
    desired = (recon.datasets(seed=101042)['train'][0]['a'] if source == 'reconstruction'
               else qkv.dataset(seed=121042)['train']['payloads'][0])
    train = copy.deepcopy(raw['train']); words = [p.split() for p in train['payloads']]
    # Swap only within each column to keep every component count exactly eight.
    for col, word in enumerate(desired.split()):
        donor = next(i for i in range(1, len(words)) if words[i][col] == word)
        words[0][col], words[donor][col] = words[donor][col], words[0][col]
    train['payloads'] = [' '.join(v) for v in words]
    assert len(set(train['payloads'])) == 128
    train['rows'] = [data._train_row(e, *train['payloads'][2*i:2*i+2]) for i, e in enumerate(train['entities'])]
    with pytest.raises(ValueError, match='Historical entity or complete-answer overlap'):
        data.validate_training_data(train)


@pytest.mark.parametrize('attack', ['c_is_training', 'c_equals_d', 'confirm_identity', 'wrong_pairing', 'support_mismatch', 'history_hash', 'episode_protocol'])
def test_eval_or_history_contract_tampering_rejected(raw, attack):
    broken = copy.deepcopy(raw)
    if attack == 'c_is_training': broken['known'][0]['c'] = broken['known'][0]['a']
    elif attack == 'c_equals_d': broken['known'][0]['d'] = broken['known'][0]['c']
    elif attack == 'confirm_identity': broken['confirm'][0]['entity'] = broken['dev'][0]['entity']
    elif attack == 'wrong_pairing': broken['dev'][0]['b'] = broken['dev'][1]['b']
    elif attack == 'support_mismatch': broken['known'][0]['supports'][2][0] = broken['known'][0]['supports'][0][0]
    elif attack == 'history_hash': broken['provenance']['history']['excluded_answers_sha256'] = '0' * 64
    else: broken['provenance']['episode_protocol'] = data.PROTOCOL
    with pytest.raises(ValueError): data.validate_dataset(broken)


def test_exact_schedule_reuse_and_random_state_isolation(raw):
    assert data.episode is qkv.episode and data.episodes is qkv.episodes and data.validate_episode is qkv.validate_episode
    assert data.EPISODE_PROTOCOL == qkv.PROTOCOL != data.PROTOCOL
    random.seed(8294); state = random.getstate()
    assert data.dataset() == raw and data.dataset(221043) != raw
    for step in (0, 7, 8, 2047):
        static, rebind = [data.episode(step, r, seed=91042) for r in data.REGIMES]
        assert data.validate_episode(rebind)['valid']
        for key in ('targets', 'background', 'bank_entities', 'local_targets'):
            assert static[key] == rebind[key]
        assert rebind == qkv.episode(step, 'rebind', seed=91042)
    assert random.getstate() == state


def test_full_matched_budget_is_independent_of_payload_token_lengths(raw):
    lengths = [sum(1 + len(w) % 3 for w in p.split()) + 1 for p in raw['train']['payloads']]
    budgets = []
    for regime in data.REGIMES:
        entities, payloads, token_count = Counter(), Counter(), 0
        for e in data.episodes(2048, regime, seed=91042):
            entities.update(e['targets'])
            used = [p for target in e['targets'] for p in e['mapping'][target]]
            payloads.update(used); token_count += sum(lengths[p] for p in used)
            for target, local in zip(e['targets'], e['local_targets']):
                after = list(e['a_payload_indices']); after[local] = e['b_payload_indices'][local]
                assert [i for i, (a, b) in enumerate(zip(e['a_payload_indices'], after)) if a != b] == [local]
                assert e['bank_entities'][local] == target
        assert entities == Counter({i: 256 for i in range(64)})
        assert payloads == Counter({i: 256 for i in range(128)})
        budgets.append((entities, payloads, token_count))
    assert budgets[0] == budgets[1]


@pytest.mark.parametrize('seed', [101042, 121042, True, -1, '221042', 1.5])
def test_historical_or_invalid_data_seed_rejected(seed):
    with pytest.raises(ValueError): data.dataset(seed)
