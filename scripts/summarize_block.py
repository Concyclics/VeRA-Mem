"""Strict JSON-only reconstruction of the fixed block-reader experiment.

No model, tokenizer, torch checkpoint, or GPU is executed. All completed formal
conditions are reported; there is no selection or confirmation-dependent tuning.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_qkv as common

require, read_json, lines, sha = common.require, common.read_json, common.lines, common.sha
same, close, norm, number = common.same, common.close, common.norm, common.number
valid_hash, artifacts, episode = common.valid_hash, common.artifacts, common.episode
PHASES, FIELDS, PAIR_METRICS = common.PHASES, common.FIELDS, common.PAIR_METRICS
RUN_PROTOCOL = 'block-attention-memory-v1'
PROTOCOL = 'block-results-audit-v1'
DATA_PROTOCOL = 'block-outer-binding-data-v1'
SEEDS = (91042, 91043, 91044)
MODES = ('diagonal', 'pooled_outer', 'block_outer')
ARMS = tuple(r+'_'+m for r in ('static', 'rebind') for m in MODES)
MODEL_REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
TEACHERS = {('train', 'preflight'): 128, ('known', 'development'): 48,
            ('dev', 'development'): 48, ('known', 'confirmation'): 32,
            ('confirm', 'confirmation'): 192}
ARTIFACTS = common.ARTIFACT_NAMES


def architecture_identity(a, regime):
    expected = dict(slots=3, train_B=True, rank=64, key_dim=64, top_k=4,
        in_features=9728, out_features=2560, writer_mode='masked_mean', value_mlp_hidden=0,
        qkv_version=1, read_mode='grouped', readout='vera', block_version=1,
        slot_weighting='uniform', residual_outer=True)
    for key, value in expected.items(): same(a.get(key), value, 'Architecture differs: '+key)
    require(a.get('seed') in SEEDS and a.get('block_mode') in MODES, 'Unknown block architecture')
    same(a.get('temperature'), .05, 'Wrong temperature')
    same(a.get('factor_epsilon'), 1e-6, 'Wrong factor epsilon')
    require(regime in ('static', 'rebind'), 'Unknown regime')
    return regime+'_'+a['block_mode'], a['seed']


def training_convergence(raw):
    windows = {}
    for label, selected in (('first_128', raw[:128]), ('last_128', raw[-128:])):
        windows[label] = dict(steps=len(selected), **{
            key: dict(mean=sum(r['metrics'][key] for r in selected)/len(selected),
                minimum=min(r['metrics'][key] for r in selected),
                maximum=max(r['metrics'][key] for r in selected),
                last=selected[-1]['metrics'][key]) for key in ('ce', 'address', 'loss')})
    return windows


def audit_training(directory, manifest, train):
    cfg = manifest['configuration']; status = read_json(directory/'training_status.json')
    same(status, manifest['result'], 'Training status/manifest mismatch')
    require(status.get('complete') is True and status.get('updates') == cfg.get('updates') == 2048,
            'Formal training requires exactly 2048 updates')
    arm, seed = architecture_identity(status['architecture'], cfg['regime'])
    require(status['regime'] == cfg['regime'] and cfg['seed'] == seed and
            cfg['block_mode'] == status['architecture']['block_mode'], 'Training configuration mismatch')
    outer = cfg['block_mode'] != 'diagonal'; params = 2034368 + 8256*outer
    require(status['trainable_parameters'] == params, 'Wrong trainable parameter count')
    lrs = [1e-4, 3e-4, .005, 3e-4] + ([3e-4] if outer else [])
    same(status['learning_rates'], lrs, 'Optimizer learning rates differ')
    raw = lines(directory/'training.jsonl'); require(len(raw) == 2048, 'Wrong training log length')
    h = hashlib.sha256(); schedule = hashlib.sha256()
    targets = Counter(); payloads = Counter(); backgrounds = Counter(); resident = Counter()
    totals = dict(target_exposures=0, gold_tokens=0, input_positions=0, padded_input_positions=0, backbone_calls=0)
    prompt_lengths = {}; answer_lengths = {}; elapsed = 0.; bindings = Counter()
    for step, row in enumerate(raw):
        ep = episode(step, cfg['regime'], seed)
        same(ep, row['episode'], 'Episode schedule mismatch'); require(row['step'] == step+1, 'Wrong training step')
        h.update(common.json.dumps(ep, sort_keys=True).encode())
        schedule.update(common.json.dumps({k: ep[k] for k in ('targets', 'background', 'bank_entities', 'local_targets')}, sort_keys=True).encode())
        targets.update(ep['targets']); backgrounds.update(ep['background'])
        aids = [ep[k][j] for k in ('a_payload_indices', 'b_payload_indices') for j in ep['local_targets']]
        payloads.update(aids); bindings.update(zip(ep['targets']*2, aids))
        resident.update({p: 16 for p in ep['a_payload_indices']})
        for j in ep['local_targets']:
            resident[ep['a_payload_indices'][j]] -= 1; resident[ep['b_payload_indices'][j]] += 1
        metrics = row['metrics']
        for key in ('ce', 'address', 'loss'): number(metrics[key], key)
        close(metrics['loss'], metrics['ce']+.2*metrics['address'], 'Wrong loss formula')
        require(len(row['gradient_norms']) == len(lrs), 'Wrong optimizer group count')
        for v in row['gradient_norms']: number(v, 'gradient norm')
        require(isinstance(row['parameter_gradient_norms'], dict) and row['parameter_gradient_norms'], 'Missing parameter gradients')
        for v in row['parameter_gradient_norms'].values():
            if v is not None: number(v, 'parameter gradient')
        now = number(row['elapsed_seconds'], 'elapsed'); require(now >= elapsed, 'Elapsed time decreases'); elapsed = now
        il, gl = row['input_lengths'], row['gold_token_lengths']
        require(len(il) == len(gl) == 16, 'Token ledger must contain sixteen independent sequences')
        for e, p, n, g in zip(ep['targets']*2, aids, il, gl):
            number(n, 'input length', True); number(g, 'gold length', True)
            require(n > g > 0, 'Invalid sequence length')
            require(e not in prompt_lengths or prompt_lengths[e] == n-g, 'Question tokenization differs')
            require(p not in answer_lengths or answer_lengths[p] == g, 'Answer tokenization differs')
            prompt_lengths[e] = n-g; answer_lengths[p] = g
        totals['target_exposures'] += 8; totals['gold_tokens'] += sum(gl)
        totals['input_positions'] += sum(il); totals['padded_input_positions'] += 16*max(il); totals['backbone_calls'] += 1
    require(set(targets) == set(range(64)) and set(targets.values()) == {256}, 'Entity exposure differs')
    require(set(payloads) == set(range(128)) and set(payloads.values()) == {256}, 'Payload exposure differs')
    require(sum(resident.values()) == 2048*16*16, 'Bank residency total differs')
    same(totals, status['totals'], 'Training token/exposure totals differ')
    require(status['schedule_sha256'] == h.hexdigest(), 'Schedule digest differs')
    require(number(status['elapsed_seconds'], 'Training seconds') >= elapsed, 'Invalid final elapsed')
    hashes = artifacts(directory, ARTIFACTS['training'])
    return dict(kind='training', arm=arm, seed=seed, regime=cfg['regime'], run_dir=str(directory),
        architecture=status['architecture'], cache_sha256=manifest['cache_sha256'], checkpoint_sha256=hashes['last.pt'],
        updates=2048, trainable_parameters=params, artifacts=hashes, schedule_sha256=h.hexdigest(),
        common_schedule_sha256=schedule.hexdigest(), convergence=training_convergence(raw),
        costs=dict(totals, training_process_seconds=status['elapsed_seconds']),
        exposure=dict(target_counts=[targets[i] for i in range(64)], supervised_payload_counts=[payloads[i] for i in range(128)],
            background_entity_counts=[backgrounds[i] for i in range(64)], physical_payload_bank_residency=[resident[i] for i in range(128)],
            known_canonical_binding_target_counts=[dict(entity=i, a=bindings[i, 2*i], b=bindings[i, 2*i+1]) for i in range(16)]),
        token_ledger=dict(prompt_lengths=[prompt_lengths[i] for i in range(64)], answer_with_eos_lengths=[answer_lengths[i] for i in range(128)]))


def validate_trace(row, index):
    tokens = common.validate_tokens(row); trace = row['trace']
    require(len(trace) == len(tokens), 'Trace/token count differs')
    width = 0 if row['condition'] == 'empty' else 3; hits = []
    for pos, (token, t) in enumerate(zip(tokens, trace)):
        require(t['token_id'] == token and t['phase'] == ('prefill' if pos == 0 else 'decode'), 'Trace token/phase differs')
        for k in ('indices', 'scores', 'weights', 'query'):
            require(isinstance(t[k], list) and len(t[k]) == 1, 'Wrong route batch dimension')
        ii, ss, ww, query = (t[k][0] for k in ('indices', 'scores', 'weights', 'query'))
        require(len(ii) == len(ss) == len(ww) == width and len(set(ii)) == width, 'Wrong selected width')
        require(len(query) == 64 and all(type(x) in (int, float) and math.isfinite(x) for x in query), 'Invalid saved query')
        if width:
            require(sum(x*x for x in query) > 0, 'Nonempty-bank query is zero')
            require(all(type(i) is int and 0 <= i < 48 for i in ii) and ii == [3*(ii[0]//3)+s for s in range(3)], 'Incomplete/unordered fact block')
            require(all(type(x) in (int, float) and math.isfinite(x) and -1.0001 <= x <= 1.0001 for x in ss), 'Invalid cosine score')
            for w in ww: close(w, 1/3, 'Block weights are not uniform')
        require(t.get('operator_weighting') == 'uniform' and
            t.get('weight_interpretation') == 'uniform membership; not signed outer coefficients', 'Wrong weight interpretation')
        masses = [sum(w for j, w in zip(ii, ww) if j == 3*index+s) for s in range(3)]
        require(len(t['target_slot_mass']) == 3, 'Wrong target slot width')
        for a, b in zip(t['target_slot_mass'], masses): close(a, b, 'Slot mass differs')
        close(t['target_fact_mass'], sum(masses), 'Fact mass differs')
        hits.append(int(any(j//3 == index for j in ii)))
    for k, v in dict(first_fact_recall_at_read=hits[0], first_fact_recall_at_1=hits[0], decode_hits=sum(hits[1:]),
                     decode_queries=len(tokens)-1, actual_read_slots=width, sparse_values_transferred_per_token=width*64).items():
        same(row[k], v, 'Route summary differs: '+k)
    require(row['injection_input'] == 'full_selected_value_block', 'Wrong injection input')
    require(len(row['first_target_slot_mass']) == 3, 'Wrong first slot width')
    for a, b in zip(row['first_target_slot_mass'], trace[0]['target_slot_mass']): close(a, b, 'First slot mass differs')
    close(row['first_target_fact_mass'], trace[0]['target_fact_mass'], 'First fact mass differs')


def answer_diagnostics(predictions, byid, train_payloads):
    train = {norm(x) for x in train_payloads}; grouped = defaultdict(list)
    for r in predictions:
        if r['role'] not in ('updated', 'control'): continue
        source = byid[r['id']]; words = norm(r['prediction']).split(); target = norm(r['answer']).split()
        require(len(target) == 3, 'Diagnostics require three-word target')
        exact = len(words) == 3
        v = dict(em=r['em'], exactly_three_words=int(exact), full_answer_containment=int(
            (' '+norm(r['answer'])+' ') in (' '+norm(r['prediction'])+' ')),
            budget_hit=int(r['budget_hit']), output_is_training_payload=int(norm(r['prediction']) in train),
            output_is_own_old_a=int(norm(r['prediction']) == norm(source['a'])),
            output_is_own_old_b=int(norm(r['prediction']) == norm(source['b'])),
            **{'word_'+str(j)+'_correct': int(exact and words[j] == target[j]) for j in range(3)})
        if r['world'] in ('C', 'D'):
            j = source['provenance']['edited_word_index']; require(type(j) is int and 0 <= j < 3, 'Invalid edited word index')
            v.update(edited_position=j, edited_word_correct=v['word_'+str(j)+'_correct'],
                both_unchanged_words_correct=int(all(v['word_'+str(k)+'_correct'] for k in range(3) if k != j)))
        grouped[r['phase'], r['world'], r['condition']].append((r, v))
    out = []
    for (ph, world, condition), items in grouped.items():
        counts = {k: sum(v[k] for _, v in items) for k in items[0][1] if k != 'edited_position'}
        positions = {}
        if world in ('C', 'D'):
            for j in range(3):
                take = [v for _, v in items if v['edited_position'] == j]
                positions[str(j)] = dict(count=len(take), edited_word_correct=sum(v['edited_word_correct'] for v in take),
                    both_unchanged_words_correct=sum(v['both_unchanged_words_correct'] for v in take), correct=sum(v['em'] for v in take))
        out.append(dict(phase=ph, world=world, condition=condition, count=len(items), counts=counts,
            rates={k: v/len(items) for k, v in counts.items()}, edited_positions=positions,
            word_count_histogram=dict(sorted(Counter(len(norm(r['prediction']).split()) for r, _ in items).items())),
            common_outputs=[dict(text=p, count=n) for p, n in Counter(r['prediction'] for r, _ in items).most_common(8)]))
    return out


def audit_evaluation(directory, manifest, rows, split, regime, train_payloads=()):
    summary = read_json(directory/'summary.json'); same(summary, manifest['result'], 'Eval summary/manifest mismatch')
    require(summary.get('protocol') == 'block-online-cpu-v1' and summary.get('complete') is True, 'Unknown/incomplete block evaluation')
    require(summary.get('shared_parameters_unchanged') is True and valid_hash(summary.get('shared_parameter_sha256')), 'Missing frozen parameter proof')
    arm, seed = architecture_identity(summary['architecture'], regime)
    require(len(rows) == summary['bank_facts'] == 16 and summary['slots_per_fact'] == 3, 'Wrong evaluation bank size')
    same(summary['resident_bytes'], dict(key_bytes=12288, value_bytes=12288, total_bytes=24576), 'Wrong resident bytes')
    part = manifest['configuration']['eval_part']; axes = common.expected_axes(split, part)
    raw = lines(directory/'predictions.jsonl'); byid = {r['id']: r for r in rows}; ids = list(byid)
    require(len(ids) == 16, 'Duplicate fact IDs'); offset = 0; pairs = []; updated = defaultdict(list); changed = defaultdict(list)
    def take(index, world, phase, condition, role, case, trigger):
        nonlocal offset
        require(offset < len(raw), 'Missing prediction'); r = raw[offset]; offset += 1
        for k, v in dict(id=ids[index], world=world, phase=phase, condition=condition, role=role, case_id=case, trigger_world=trigger).items():
            require(r.get(k) == v, 'Prediction identity/order differs: '+k)
        require(r['question'] == rows[index]['questions'][PHASES[phase][1]] and r['answer'] == rows[index][FIELDS[world]], 'Prediction provenance differs')
        same(r['em'], int(norm(r['prediction']) == norm(r['answer'])), 'Raw EM mismatch')
        for k in ('answer_tokens', 'bank_slots', 'decode_hits', 'decode_queries'): number(r[k], k, True)
        number(r['nll_sum'], 'NLL'); require(r['answer_tokens'] > 0, 'Empty gold score')
        require(r['budget_hit'] == (r['generation_tokens'] >= 32), 'Wrong budget-hit flag')
        require(valid_hash(r['bank_hash']) and r['bank_slots'] == (0 if condition == 'empty' else 48), 'Invalid bank metadata')
        validate_trace(r, index); return r
    for phase, worlds in axes.items():
        initial = [take(i, 'A', phase, 'real', 'initial', ids[i], 'A') for i in range(16)]
        require(len({r['bank_hash'] for r in initial}) == 1, 'Initial bank differs')
        for world in worlds:
            for i in range(16):
                u = take(i, world, phase, 'real', 'updated', ids[i], world); ni = (i+1)%16
                neighbor = take(ni, 'A', phase, 'real', 'locality', ids[i], world)
                restored = take(i, 'A', phase, 'real', 'restored', ids[i], world)
                require(u['bank_hash'] != initial[i]['bank_hash'] and neighbor['bank_hash'] == u['bank_hash'] and
                        restored['bank_hash'] != u['bank_hash'], 'Invalid intervention/locality/restore hashes')
                p = dict(id=ids[i], phase=phase, world=world, a_em=initial[i]['em'], updated_em=u['em'],
                    pair_em=initial[i]['em']*u['em'], restored_em=restored['em'], update_restore_em=u['em']*restored['em'],
                    all_three_em=initial[i]['em']*u['em']*restored['em'], locality_equal=int(neighbor['prediction'] == initial[ni]['prediction']),
                    locality_correct=neighbor['em'], locality_joint=neighbor['em']*initial[ni]['em'], shuffle_em=None, empty_em=None)
                if world != 'SWAP':
                    for condition in ('shuffle', 'empty'):
                        control = take(i, world, phase, condition, 'control', ids[i], world); p[condition+'_em'] = control['em']
                        if condition == 'shuffle': require(control['bank_hash'] != u['bank_hash'], 'Shuffle bank unchanged')
                pairs.append(p); updated[phase+'_'+world].append(u)
                changed[phase+'_'+world].append(dict(changed=initial[i]['prediction'] != u['prediction'],
                    both_wrong=initial[i]['prediction'] != u['prediction'] and not initial[i]['em'] and not u['em']))
    require(offset == len(raw), 'Unexpected/duplicate predictions')
    same(pairs, read_json(directory/'pairs.json'), 'Saved pairs differ from raw')
    grouped = defaultdict(list)
    for p in pairs: grouped[p['phase']+'_'+p['world']].append(p)
    groups = {k: dict(count=len(v), **{m: sum(p[m] for p in v)/len(v) if v[0][m] is not None else None for m in PAIR_METRICS}) for k, v in grouped.items()}
    same(groups, summary['groups'], 'Summary groups differ from raw')
    costs = common.prediction_statistics(raw)
    for k in ('generation_calls', 'generation_tokens', 'generation_seconds', 'answer_scoring_tokens'): same(costs[k], summary[k], 'Eval cost mismatch '+k)
    for k, v in dict(independent_interventions=len(pairs), restorations=len(pairs), real_group_write_events=2*len(pairs)).items(): same(summary[k], v, 'Write count differs')
    return dict(kind='evaluation', arm=arm, seed=seed, regime=regime, run_dir=str(directory), split=split, part=part,
        architecture=summary['architecture'], cache_sha256=manifest['cache_sha256'], checkpoint_sha256=manifest['checkpoint_sha256'],
        groups=groups, updated_diagnostics={k: common.prediction_statistics(v, byid, train_payloads) for k, v in updated.items()},
        answer_diagnostics=answer_diagnostics(raw, byid, train_payloads),
        changed_output={k: dict(count=len(v), changed=sum(x['changed'] for x in v), changed_both_wrong=sum(x['both_wrong'] for x in v)) for k, v in changed.items()},
        costs=costs, independent_interventions=len(pairs), restoration_writes=len(pairs), real_group_write_events=2*len(pairs),
        bank_facts=16, bytes_per_fact=1536, artifacts=artifacts(directory, ARTIFACTS['evaluation']))


def completeness(records):
    kinds = ('train', 'known_development', 'dev_development', 'known_confirmation', 'confirm_confirmation')
    expected = {(a, s, k) for a in ARMS for s in SEEDS for k in kinds}; found = defaultdict(list)
    for r in records:
        if r['kind'] not in ('training', 'evaluation'): continue
        key = (r['arm'], r['seed'], 'train' if r['kind'] == 'training' else r['split']+'_'+r['part'])
        require(key in expected, 'Unexpected formal condition'); found[key].append(r['run_dir'])
    duplicates = [dict(condition=list(k), runs=v) for k, v in found.items() if len(v) > 1]
    require(not duplicates, 'Duplicate formal condition: '+str(duplicates))
    return dict(expected=90, observed=len(found), complete=set(found) == expected,
                missing=[list(k) for k in sorted(expected-set(found))], duplicates=duplicates)


def endpoints(records):
    result = []
    for arm in ARMS:
        for seed in SEEDS:
            found = [r for r in records if r.get('arm') == arm and r.get('seed') == seed]
            if not found: continue
            entry = dict(arm=arm, seed=seed, groups={})
            for r in found:
                if r['kind'] == 'training':
                    entry.update(trainable_parameters=r['trainable_parameters'], last_128_ce=r['convergence']['last_128']['ce']['mean'],
                        last_128_address=r['convergence']['last_128']['address']['mean'], train_run=r['run_dir'])
                else:
                    for key, group in r['groups'].items():
                        label=r['split']+'_'+r['part']+'_'+key
                        entry['groups'][label] = dict(group, updated= r['updated_diagnostics'][key],
                            teacher_paired_eligible=r.get('teacher_qualification', {}).get('paired', {}).get(key))
            result.append(entry)
    return result


def continuation_gate(records):
    index = {(r.get('arm'), r.get('seed'), r.get('split'), r.get('part')): r for r in records if r['kind'] == 'evaluation'}
    outcomes = []
    for regime in ('static', 'rebind'):
        per_seed = []
        for seed in SEEDS:
            need = [(regime+'_'+mode, seed, 'known', part) for mode in ('pooled_outer', 'block_outer') for part in ('development', 'confirmation')]
            if not all(k in index for k in need): per_seed.append(dict(seed=seed, complete=False, passed=False)); continue
            blocks = [index[regime+'_block_outer', seed, 'known', p] for p in ('development', 'confirmation')]
            pools = [index[regime+'_pooled_outer', seed, 'known', p] for p in ('development', 'confirmation')]
            b = blocks[0]['groups']['CC_B']; checks = dict(known_ab_pair=b['pair_em'] >= 12/16); values = dict(known_ab_pair=b['pair_em'])
            for j, world in enumerate(('C', 'D')):
                key = 'CC_'+world; g, control = blocks[j]['groups'][key], pools[j]['groups'][key]
                delta = g['update_restore_em']-control['update_restore_em']; values[world] = dict(
                    block_update_restore=g['update_restore_em'], pooled_update_restore=control['update_restore_em'], improvement=delta,
                    real_minus_shuffle=g['updated_em']-g['shuffle_em'], real_minus_empty=g['updated_em']-g['empty_em'])
                checks[world+'_matched_improvement'] = delta >= 4/16
                checks[world+'_shuffle_effect'] = values[world]['real_minus_shuffle'] >= 4/16
                checks[world+'_empty_effect'] = values[world]['real_minus_empty'] >= 4/16
            qualified = True
            for record, keys in ((blocks[0], ('CC_B', 'CC_C')), (blocks[1], ('CC_D',))):
                for key in keys:
                    g = record.get('teacher_qualification', {}).get('paired', {}).get(key, {})
                    qualified &= g.get('count') == g.get('both_correct') == 16
            per_seed.append(dict(seed=seed, complete=True, checks=checks, values=values, teacher_qualified=qualified,
                                 passed=all(checks.values()) and qualified))
        outcomes.append(dict(regime=regime, per_seed=per_seed, complete=all(s['complete'] for s in per_seed),
                             passed=all(s['passed'] for s in per_seed)))
    return dict(policy='Fixed block-versus-pooled, within-regime/seed, all three seeds; no model selection',
        regimes=outcomes, supports_further_block_reader_investment=any(r['passed'] for r in outcomes),
        automatic_corpus_expansion=False, note='A research evidence threshold, not proof of semantic generalization or automatic authority to expand.')


def audit_sources(directory, manifest):
    result = common.audit_sources(directory, manifest)
    for name in ('block_run.py', 'block_eval.py', 'block_data.py', 'block_vera.py', 'block_backend.py'):
        require(valid_hash(result['source_sha256'].get(name)), 'Missing block source '+name)
    return result


def load_registration(path, protocol_path):
    r = read_json(path)
    require(r.get('protocol') == 'block-preregistration-v1' and r.get('teacher_preflight_passed') is True and r.get('smoke_passed') is True,
            'Invalid block preregistration')
    require(r.get('selection_policy') == 'all_18_final_checkpoints_no_selection', 'Unexpected selection policy')
    require(r['protocol_document_sha256'] == sha(protocol_path), 'Registered protocol changed')
    same(r['arms'], list(ARMS), 'Registered arms differ'); same(r['seeds'], list(SEEDS), 'Registered seeds differ')
    require(r['updates'] == 2048 and r['model_revision'] == MODEL_REVISION, 'Registered model/budget differs')
    require(r['source_python_sha256'] and all(k.startswith('src/vera_mem/') and Path(k).name.endswith('.py') and valid_hash(v)
        for k, v in r['source_python_sha256'].items()), 'Invalid registered runtime hashes')
    require(r['plans_sha256'] and all(valid_hash(v) for v in r['plans_sha256'].values()), 'Invalid registered plan hashes')
    expected_caches = {split+'.pt': v['cache_sha256'] for split, v in r['feature_result']['caches'].items()}
    require(set(expected_caches) == {'train.pt', 'known.pt', 'dev.pt', 'confirm.pt'} and
            all(valid_hash(v) for v in expected_caches.values()), 'Invalid registered feature inventory')
    same(r['feature_cache_sha256'], expected_caches, 'Registered feature hash tables differ')
    return r


def registered_run(directory, manifest, registration):
    if manifest['stage'] == 'prepare':
        same(manifest['result'], registration['feature_result'], 'Registered feature result differs'); return
    require(manifest['cache_sha256'] in {v['cache_sha256'] for v in registration['feature_result']['caches'].values()}, 'Unregistered cache')
    if manifest['stage'] == 'teacher' and manifest['configuration'].get('eval_part') == 'preflight':
        suffix = (directory.parent.name, directory.name, 'manifest.json')
        receipts = [v for k, v in registration['preflight_artifacts_sha256'].items() if Path(k).parts[-3:] == suffix]
        require(receipts == [sha(directory/'manifest.json')], 'Registered preflight receipt differs'); return
    same(manifest['source_sha256'], {Path(k).name: v for k, v in registration['source_python_sha256'].items()}, 'Formal runtime differs from registration')
    require(sha(directory.parent/'source/docs/block_protocol.md') == registration['protocol_document_sha256'], 'Frozen protocol copy differs')
    plan_path = directory.parent/'plan.json'; ph = sha(plan_path)
    require(ph in registration['plans_sha256'].values(), 'Unregistered suite plan')
    require(read_json(directory.parent/'suite.json')['plan_sha256'] == ph, 'Suite plan SHA differs')
    jobs = [j for j in read_json(plan_path) if j['name'] == directory.name]
    require(len(jobs) == 1 and jobs[0]['entry'] == 'block', 'Job not uniquely in plan')
    args = jobs[0]['arguments']; require(len(args)%2 == 0, 'Invalid planned option/value list'); seen = set()
    for flag, value in zip(args[::2], args[1::2]):
        require(flag.startswith('--') and flag not in seen, 'Duplicate/invalid planned option'); seen.add(flag)
        require(str(manifest['configuration'].get(flag[2:].replace('-', '_'))) == value, 'Manifest planned option differs '+flag)


def prepare_cache(directory, manifest):
    data = read_json(directory/'dataset.json'); result = manifest['result']; caches = {}
    require(result['dataset_sha256'] == sha(directory/'dataset.json') and data['protocol'] == DATA_PROTOCOL and data['seed'] == 221042,
            'Prepared dataset identity/hash differs')
    require(set(result['caches']) == {'train', 'known', 'dev', 'confirm'}, 'Incomplete prepared split inventory')
    for split, meta in result['caches'].items():
        cp, jp = directory/(split+'.pt'), directory/(split+'.json'); table = read_json(jp)
        require(sha(cp) == meta['cache_sha256'], 'Prepared tensor cache digest differs')
        same(table, data[split], 'Prepared JSON/dataset differs'); rows = table['rows'] if split == 'train' else table
        require(len(rows) == (64 if split == 'train' else 16) and len({r['id'] for r in rows}) == len(rows), 'Wrong prepared rows')
        if split == 'train':
            require(set(table) == {'protocol', 'seed', 'entities', 'payloads', 'rows'} and len(set(table['entities'])) == 64 and len(set(table['payloads'])) == 128, 'Invalid train-only table')
            require(meta['entities'] == 64 and meta['payloads'] == 128 and meta['contextual_observations'] == 8192, 'Wrong feature inventory')
            require(all(r['worlds'] == ['A', 'B'] and len(r['questions']) == 1 and r['entity'] == table['entities'][i]
                        and r['a'] == table['payloads'][2*i] and r['b'] == table['payloads'][2*i+1]
                        for i, r in enumerate(rows)), 'Training rows not canonical static A/B')
        else:
            require(meta['records'] == 16 and all(r['worlds'] == ['A', 'B', 'C', 'D', 'SWAP'] and len(r['questions']) == 2 for r in rows), 'Invalid evaluation table')
        caches[meta['cache_sha256']] = dict(split=split, rows=rows, data=table, path=str(cp.resolve()), json_sha256=sha(jp))
    return caches


def resolve_run_input(value, root):
    path = Path(value)
    if path.is_file(): return path.resolve()
    parts = path.parts
    require('runs' in parts, 'Input cannot be mapped to local backup: '+str(path))
    local = root.joinpath(*parts[parts.index('runs')+1:])
    require(local.is_file(), 'Missing locally backed-up input '+str(local)); return local.resolve()


def summarize(root, registration_path=None):
    root = Path(root).resolve(); require(root.is_dir(), 'Runs root missing')
    repo = Path(__file__).resolve().parents[1]
    if registration_path is None:
        candidates = (repo.parent/'plans/block_registration_20261006.json', repo/'docs/results/block/registration.json')
        registration_path = next((p for p in candidates if p.is_file()), None)
    registration = load_registration(registration_path, repo/'docs/block_protocol.md') if registration_path else None
    records = []; jobs = []; caches = {}; preparations = []; pending = []; ignored = []; sources = []
    paths = [p for p in sorted(root.rglob('manifest.json')) if any(x.startswith('block_') for x in (root.name, *p.relative_to(root).parts[:-1]))
        and not any(x in ('source', 'source_snapshot', 'source_snapshots', '.git') for x in p.relative_to(root).parts)]
    for path in paths:
        reason = common.ignored_reason(path, root)
        if reason: ignored.append(dict(run_dir=str(path.parent), reason=reason, manifest_sha256=sha(path))); continue
        m = read_json(path)
        if m.get('protocol') != RUN_PROTOCOL: continue
        stage = m.get('stage'); require(stage in ('prepare', 'train', 'eval', 'teacher'), 'Unknown block stage')
        require(m['configuration']['stage'] == stage, 'Stage/configuration mismatch')
        if m.get('complete') is not True:
            pending.append(dict(run_dir=str(path.parent), stage=stage, manifest_sha256=sha(path))); continue
        require(m.get('backbone_unchanged') is True, 'Frozen backbone proof missing')
        source = audit_sources(path.parent, m)
        if stage in ('train', 'eval'): require(registration is not None, 'Formal models require preregistration')
        if registration: registered_run(path.parent, m, registration)
        sources.append(dict(stage=stage, run_dir=str(path.parent), **source))
        if stage == 'prepare':
            new = prepare_cache(path.parent, m)
            require(not caches, 'Multiple retained preparations are ambiguous'); caches.update(new)
            preparations.append(dict(run_dir=str(path.parent), artifacts=artifacts(path.parent, ['manifest.json', 'dataset.json']+
                [s+ext for s in ('train', 'known', 'dev', 'confirm') for ext in ('.pt', '.json')]))); continue
        jobs.append((path.parent, m))
    for directory, m in jobs:
        require(m.get('cache_sha256') in caches, 'Run does not bind a prepared cache')
        cp = resolve_run_input(m['configuration']['cache'], root)
        require(str(cp) == caches[m['cache_sha256']]['path'], 'Declared cache path differs from digest provenance')
        if m['stage'] == 'train':
            require(caches[m['cache_sha256']]['split'] == 'train', 'Training used evaluation cache')
            records.append(audit_training(directory, m, caches[m['cache_sha256']]['data']))
    parents = defaultdict(list)
    for r in records: parents[r['checkpoint_sha256']].append(r)
    for directory, m in jobs:
        if m['stage'] == 'train': continue
        cache = caches[m['cache_sha256']]
        if m['stage'] == 'teacher': r = common.audit_teacher(directory, m, cache['rows'], cache['split'])
        else:
            parent = parents[m['checkpoint_sha256']]; require(len(parent) == 1, 'Evaluation must bind one final training checkpoint')
            cp = resolve_run_input(m['configuration']['checkpoint'], root)
            require(cp == (Path(parent[0]['run_dir'])/'last.pt').resolve(), 'Evaluation did not use matching terminal checkpoint')
            r = audit_evaluation(directory, m, cache['rows'], cache['split'], parent[0]['regime'], caches[parent[0]['cache_sha256']]['data']['payloads'])
            same(r['architecture'], parent[0]['architecture'], 'Evaluation/parent architecture mismatch'); r['parent_train'] = parent[0]['run_dir']
        records.append(r)
    for r in records:
        if r['kind'] != 'evaluation': continue
        teachers = [t for t in records if t['kind'] == 'teacher' and t['cache_sha256'] == r['cache_sha256'] and t['part'] == r['part']]
        require(len(teachers) <= 1, 'Duplicate matching teachers')
        r['teacher_qualification'] = dict(source_runs=[t['run_dir'] for t in teachers], paired=teachers[0]['paired'] if teachers else {})
        eligible = {}; pairs = read_json(Path(r['run_dir'])/'pairs.json')
        for key, g in r['teacher_qualification']['paired'].items():
            take = [p for p in pairs if p['phase']+'_'+p['world'] == key and p['id'] in set(g['eligible_ids'])]
            eligible[key] = dict(count=len(take), **{k: sum(p[k] for p in take)/len(take) if take else None for k in ('pair_em', 'updated_em', 'update_restore_em')})
        r['teacher_eligible_diagnostics'] = eligible
    model_sources = [s['source_sha256'] for s in sources if s['stage'] in ('train', 'eval')]
    for source in model_sources[1:]: same(source, model_sources[0], 'Formal runtime source differs')
    for seed in SEEDS:
        rs = [r for r in records if r['kind'] == 'training' and r['seed'] == seed]
        require(len({r['common_schedule_sha256'] for r in rs}) <= 1, 'Across-arm target/background schedule differs')
        for k in ('target_exposures', 'gold_tokens', 'input_positions', 'backbone_calls'):
            require(len({r['costs'][k] for r in rs}) <= 1, 'Across-arm unpadded budget differs '+k)
        for r in rs[1:]: same(r['token_ledger'], rs[0]['token_ledger'], 'Across-arm tokenization differs')
        for regime in ('static', 'rebind'):
            rr = [r for r in rs if r['regime'] == regime]
            require(len({r['schedule_sha256'] for r in rr}) <= 1 and len({r['costs']['padded_input_positions'] for r in rr}) <= 1, 'Same-regime schedule/padding differs')
    teachers = [r for r in records if r['kind'] == 'teacher']; inv = Counter((r['split'], r['part']) for r in teachers)
    require(all(k in TEACHERS and n == 1 for k, n in inv.items()), 'Unknown/duplicate teacher condition')
    for r in teachers: require(r['costs']['generation_calls'] == TEACHERS[r['split'], r['part']], 'Teacher budget differs')
    if any(r['kind'] == 'training' for r in records):
        preflight = [r for r in teachers if r['split'] == 'train' and r['part'] == 'preflight']
        require(len(preflight) == 1 and preflight[0]['qualified'] and all(g['count'] == g['correct'] == 64 for g in preflight[0]['groups'].values()), 'Formal training lacks 128/128 teacher preflight')
    scope = completeness(records); complete = scope['complete'] and set(inv) == set(TEACHERS) and len(preparations) == 1 and not pending
    return dict(protocol=PROTOCOL, generated_at=datetime.now(timezone.utc).isoformat(), runs_root=str(root), audit_passed=True,
        partial=not complete, formal_scope=scope, formal_jobs=dict(expected=96, observed=len(records)+len(preparations), complete=complete),
        teacher_scope=dict(expected=5, observed=len(teachers), complete=set(inv) == set(TEACHERS), expected_generations=448,
            correct=sum(g['correct'] for t in teachers for g in t['groups'].values()), count=sum(g['count'] for t in teachers for g in t['groups'].values())),
        records=records, preparation=preparations, pending=pending, ignored=ignored, costs=common.aggregate_costs(records),
        endpoints=endpoints(records), continuation=continuation_gate(records), script_sha256=sha(__file__),
        helper_sha256=sha(common.__file__), source_lineage=sources,
        registration=dict(path=str(Path(registration_path).resolve()), sha256=sha(registration_path)) if registration else None,
        limitations=['JSON/raw text and scalar schedules are independently recomputed; no LM generation, tokenizer decoding, checkpoint tensors, or VDB tensor replay is executed here.',
            'Full queries are checked for shape/finiteness, but global winning-group selection requires the independent tensor-bank audit.',
            'Uniform weights measure block membership, not signed outer coefficients or causal content contribution.',
            'Three-word position scores require exactly three normalized output words; changed and preserved-word counts are not necessarily true for the same examples.',
            'NLL excludes EOS; training CE includes EOS. Repeated cases, templates and seeds are not independent datasets.',
            'Empty paired EM is logically zero for two distinct answers; the gate uses same-world single-answer control differences.',
            'Equal updates/parameters/storage do not imply equal convergence or FLOPs; process seconds can overlap.',
            'No checkpoint, seed or architecture is selected; the continuation gate is not automatic corpus expansion.'])


def write_outputs(report, output):
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    def save(path, value): path.write_text(common.json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    save(output, report); source = dict(summary_sha256=sha(output), script_sha256=sha(__file__), helper_sha256=sha(common.__file__))
    training = [dict(arm=r['arm'], seed=r['seed'], run_dir=r['run_dir'], convergence=r['convergence'], costs=r['costs'],
        training_jsonl_sha256=r['artifacts']['training.jsonl']) for r in report['records'] if r['kind'] == 'training']
    diagnostics = [dict(arm=r['arm'], seed=r['seed'], split=r['split'], part=r['part'], run_dir=r['run_dir'], groups=r['answer_diagnostics'],
        raw_sha256=r['artifacts']['predictions.jsonl']) for r in report['records'] if r['kind'] == 'evaluation']
    save(output.with_name('training_convergence.json'), dict(protocol='block-training-convergence-v1', partial=report['partial'], sources=source,
        records=training, scope='Training-only first/last 128 steps. Equal terminal updates do not certify equal fit.'))
    save(output.with_name('answer_diagnostics.json'), dict(protocol='block-answer-diagnostics-v1', partial=report['partial'], sources=source,
        records=diagnostics, limitations=report['limitations']))
    save(output.with_name('key_facts.json'), dict(protocol='block-key-facts-v1', partial=report['partial'], sources=source,
        formal_jobs=report['formal_jobs'], teacher_scope=report['teacher_scope'], costs=report['costs'],
        continuation=report['continuation'], endpoints=report['endpoints'], limitations=report['limitations']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Summary JSON; three companion JSON files are written alongside')
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--require-final', action='store_true')
    args=parser.parse_args(argv); report=summarize(args.runs_root, args.registration)
    require(not args.require_final or not report['partial'], 'All 96 formal jobs must be complete for --require-final')
    write_outputs(report, args.output)
    print(common.json.dumps(dict(output=str(args.output), partial=report['partial'], formal_jobs=report['formal_jobs'],
        teacher=report['teacher_scope'], continuation=report['continuation']['supports_further_block_reader_investment'])))


if __name__ == '__main__': main()
