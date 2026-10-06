#!/usr/bin/env python3
"""Offline final-only answer audit. --self-test never opens experiment artifacts.

Run only after the owner announces final backup completion:
  python scripts/audit_interface_answers.py --execute-final --output docs/results/interface/answer_errors.json
All six suites and all eleven evaluations must be complete before any raw
predictions are opened. This script never performs inference, SSH, or updates.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

PHASES = ('CC', 'CH', 'HC', 'HH')
LEGACY_WORDS = frozenset(('apple', 'river', 'candle', 'garden', 'tiger', 'pencil', 'window', 'basket',
                          'cloud', 'lemon', 'mirror', 'rabbit', 'island', 'silver', 'orange', 'forest'))
EXPECTED = {f'{arm}_{seed}_confirm' for seed in (42, 43, 44) for arm in ('coupled', 'decoupled')} | {
    f'{arm}_42_confirm' for arm in ('base', 'clip', 'normalized', 'hidden', 'on_policy')}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def norm(text):
    # Exact normalization contract of src/vera_mem/metrics.py; no article removal.
    text = unicodedata.normalize('NFKC', text).casefold()
    return re.sub(r'\s+', ' ', ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)).strip()


def read_json(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(row):
    world = row['world']
    require(world in ('A', 'B'), 'Diagnostic only accepts separate A/B rows')
    answer = row['answer_results'][world]['answer']
    pred, gold = norm(row['prediction']), norm(answer)
    words, expected = pred.split(), gold.split()
    require(len(expected) == 3, 'Extended gold answer must have three words')
    em = int(pred == gold)
    contained = int((' ' + gold + ' ') in (' ' + pred + ' '))
    require(em == row['answer_results'][world]['em'], 'Raw EM mismatch')
    require(contained == row['answer_results'][world]['answer_containment'], 'Raw containment mismatch')
    if not words:
        shape = 'empty'
    elif len(words) == 1 and pred in LEGACY_WORDS:
        shape = 'legacy_16_single_word'
    elif len(words) == 1:
        shape = 'other_single_word'
    elif len(words) == 3:
        shape = 'three_words'
    else:
        shape = 'other_multiword'
    text = row['prediction']
    return dict(prediction=pred, answer=gold, em=em, containment=contained,
                first_word_correct=int(bool(words) and words[0] == expected[0]),
                gold_first_word_anywhere=int(expected[0] in words), word_count=len(words), shape=shape,
                answer_contained_with_extra_words=int(contained and not em),
                generation_tokens=row['generation_tokens'], budget_hit=int(row['budget_hit']),
                format_markers=dict(json_braces=bool(re.search(r'[{}\[\]]', text)), code_fence='```' in text,
                    pipe_table='|' in text, newline='\n' in text,
                    label_prefix=bool(re.match(r'^\s*(?:answer|response|result|output|phrase|stored phrase)\s*:', text, re.I))))


def top(counter, limit=12):
    return [{'text': text, 'count': count} for text, count in sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:limit]]


def event_summary(rows):
    d = [describe(r) for r in rows]
    n = len(d)
    require(n > 0, 'Empty diagnostic denominator')
    sums = {k: sum(x[k] for x in d) for k in ('em', 'containment', 'first_word_correct',
            'gold_first_word_anywhere', 'answer_contained_with_extra_words', 'budget_hit')}
    evidence = []
    for row, desc in zip(rows, d):
        if len(evidence) < 8 and not desc['em']:
            evidence.append(dict(target_id=row['target_id'], world=row['world'],
                prediction=row['prediction'], expected=desc['answer'], shape=desc['shape'],
                generation_tokens=desc['generation_tokens'], budget_hit=bool(desc['budget_hit'])))
    if sums['em'] == 0 and sums['containment'] == 0:
        interpretation = '零完整 EM 且无输出含完整正确答案；不能仅用删除额外标签/格式解释。它不单独区分提示导致内容遗漏、知识绑定、寻址或读出错误。'
    elif sums['em'] == 0:
        interpretation = '零完整 EM 中有一部分输出包含完整答案，存在可见额外文字/格式问题；仅对这些 containment 命中能由删除多余文字恢复，不推广到全部错误。'
    else:
        interpretation = '分别报告严格正确、完整答案包含和缺词/错词；containment 是宽松文本指标，不能替代语义正确或完整答案 EM。'
    return dict(count=n, numerators=sums, rates={k:v/n for k,v in sums.items()},
        normalized_word_count_distribution=dict(sorted(Counter(x['word_count'] for x in d).items())),
        generation_token_count_distribution=dict(sorted(Counter(x['generation_tokens'] for x in d).items())),
        output_shape_counts=dict(Counter(x['shape'] for x in d)),
        three_word_but_wrong=sum(x['word_count'] == 3 and not x['em'] for x in d),
        legacy_single_word_correct_first_word=sum(x['shape']=='legacy_16_single_word' and x['first_word_correct'] for x in d),
        em_wrong_nonbudget_count=sum(not x['em'] and not x['budget_hit'] for x in d),
        format_marker_counts={k:sum(x['format_markers'][k] for x in d) for k in d[0]['format_markers']},
        most_common_normalized_outputs=top(Counter(x['prediction'] for x in d)),
        most_common_raw_outputs=top(Counter(r['prediction'] for r in rows)),
        first_error_examples=evidence, interpretation=interpretation)


def pair_summary(pairs):
    counts = Counter()
    examples = []
    for a, b in pairs:
        x, y = describe(a), describe(b)
        require(x['answer'] != y['answer'], 'A/B answers must differ')
        changed = x['prediction'] != y['prediction']
        both = bool(x['em'] and y['em'])
        wrong = not x['em'] and not y['em']
        counts['both_full_em'] += both
        counts['both_answer_containment'] += bool(x['containment'] and y['containment'])
        counts['both_first_word_correct'] += bool(x['first_word_correct'] and y['first_word_correct'])
        counts['changed'] += changed
        counts['changed_both_correct'] += changed and both
        counts['changed_exactly_one_correct'] += changed and bool(x['em'] != y['em'])
        counts['changed_both_wrong'] += changed and wrong
        counts['changed_but_not_both_correct'] += changed and not both
        counts['unchanged'] += not changed
        counts['unchanged_both_wrong'] += not changed and wrong
        counts['reverse_swapped_answers'] += x['prediction'] == y['answer'] and y['prediction'] == x['answer']
        if changed and wrong and len(examples) < 6:
            examples.append(dict(target_id=a['target_id'], prediction_a=a['prediction'],
                                 prediction_b=b['prediction'], answer_a=x['answer'], answer_b=y['answer']))
    n = len(pairs)
    require(n > 0 and counts['changed'] == counts['changed_both_correct'] + counts['changed_exactly_one_correct'] + counts['changed_both_wrong'], 'Pair partition mismatch')
    return dict(count=n, numerators=dict(counts), rates={k:v/n for k,v in counts.items()},
                changed_both_wrong_examples=examples)


def metadata_barrier(runs, tag):
    """Do not open predictions until ALL expected jobs pass this barrier."""
    suites = [runs/f'interface_step{stage}_confirm_{queue}_{tag}' for stage in (3, 4) for queue in 'abc']
    entries = []
    for directory in suites:
        suite = read_json(directory/'suite.json')
        require(suite.get('protocol') == 'interface-experiment-v1' and suite.get('complete') is True
                and suite.get('status') == 'complete', 'All six suites must be complete: '+str(directory))
        for job in suite['jobs']:
            require(job.get('status') == 'complete' and job.get('exit_code') == 0, 'Incomplete eval job')
            entries.append((directory, job))
    names = [job['name'] for _, job in entries]
    require(len(names) == 11 and len(set(names)) == 11 and set(names) == EXPECTED, 'Require exactly the eleven preregistered formal evaluations')
    output = []
    for suite_dir, job in entries:
        directory = suite_dir/job['name']
        m, v = read_json(directory/'manifest.json'), read_json(directory/'metrics.json')
        require(m.get('protocol') == 'memory-interface-v1' and m.get('complete') is True and m.get('backbone_unchanged') is True, 'Incomplete/frozen job manifest')
        c = m['configuration']
        require(c.get('stage') == 'eval' and c.get('task') == 'extended' and c.get('max_cases') is None, 'Wrong or partial evaluation')
        require(v.get('protocol') == 'interface-cpu-vdb-eval-v1' and v.get('complete') is True
                and v.get('split') == 'confirm' and v.get('evaluated_facts') == v.get('bank_records') == 256
                and v.get('max_new_tokens') == 32 and set(v['phases']) == set(PHASES), 'Incomplete/wrong confirm metrics')
        require(m.get('result') == v, 'Manifest/result mismatch')
        require(v['shared_weights_before'] == v['shared_weights_after'] and v['frozen_online']
                and v['online_gradient_steps'] == 0, 'Evaluation mutation')
        output.append(dict(name=job['name'], directory=directory, manifest=m, metrics=v,
            metadata_hashes={f:sha(directory/f) for f in ('manifest.json','metrics.json','assignments.json')},
            suite_sha256=sha(suite_dir/'suite.json')))
    require(len({x['manifest']['cache_sha256'] for x in output}) == 1, 'Confirmation caches differ')
    require(len({x['metadata_hashes']['assignments.json'] for x in output}) == 1, 'Confirmation assignments differ')
    return output


def audit_run(entry):
    directory, metrics = entry['directory'], entry['metrics']
    expected_ids = set(read_json(directory/'assignments.json')['selected_case_ids'])
    require(len(expected_ids) == 256, 'Wrong selected IDs')
    expected_keys = set()
    for phase in PHASES:
        methods = ['real'] + (['shuffled','empty'] if phase in ('CC','HC') else []) + (['canonical_key'] if phase=='HC' else [])
        if metrics['include_teacher']:
            methods.append('teacher')
        for method in methods:
            for identity in expected_ids:
                for world in (('A_and_B',) if method=='empty' else ('A','B')):
                    expected_keys.add((phase,method,identity,world))
    seen, selected = set(), defaultdict(dict)
    digest = hashlib.sha256()
    with (directory/'predictions.jsonl').open('rb') as f:
        for line in f:
            digest.update(line)
            row = json.loads(line)
            key = (row['phase'],row['method'],row['target_id'],row['world'])
            require(key in expected_keys and key not in seen, 'Unexpected/duplicate prediction row')
            seen.add(key)
            require(row['budget_hit'] == (row['generation_tokens']>=32), 'Wrong budget-hit flag')
            require(row['bank_hash_before'] == row['bank_hash_after'], 'Bank mutated')
            if row['method'] not in ('real','teacher'):
                continue
            require(row['query_id']==row['target_id'] and not row['unrelated'], 'Unexpected target/query identity')
            require((row['teacher_context'] is not None) == (row['method']=='teacher'), 'Teacher context leakage')
            describe(row)
            selected[(row['phase'],row['method'])][(row['target_id'],row['world'])] = row
    require(seen == expected_keys and len(seen) == metrics['generation_calls'], 'Incomplete raw rows')
    phases = {}
    for phase in PHASES:
        phases[phase] = {}
        for method in ('real','teacher'):
            if method=='teacher' and not metrics['include_teacher']:
                continue
            members = selected[(phase,method)]
            pairs = [(members[(i,'A')],members[(i,'B')]) for i in sorted(expected_ids)]
            report = dict(all_worlds=event_summary([x for pair in pairs for x in pair]),
                A=event_summary([a for a,b in pairs]), B=event_summary([b for a,b in pairs]), pairs=pair_summary(pairs))
            logged = metrics['phases'][phase]['methods'][method]
            require(report['A']['numerators']['em']==logged['a_correct'] and report['B']['numerators']['em']==logged['b_correct']
                    and report['pairs']['numerators']['both_full_em']==logged['both_correct'], 'Diagnostic/main summary EM mismatch')
            phases[phase][method] = report
    for name, hashed in entry['metadata_hashes'].items():
        require(sha(directory/name)==hashed, 'Metadata changed while auditing')
    return dict(name=entry['name'],directory=str(directory),cache_sha256=entry['manifest']['cache_sha256'],
        checkpoint_sha256=entry['manifest']['checkpoint_sha256'],complete=True,raw_rows=len(seen),
        real_rows=2048,raw_sha256=digest.hexdigest(),metadata_sha256=entry['metadata_hashes'],
        suite_sha256=entry['suite_sha256'],phases=phases)


def execute(runs, tag, output):
    entries = metadata_barrier(runs, tag)
    reports = [audit_run(x) for x in entries]
    # Final barrier again; no report is emitted if any artifact is incomplete.
    after = metadata_barrier(runs, tag)
    require([(x['name'],x['metadata_hashes'],x['suite_sha256']) for x in entries]
            == [(x['name'],x['metadata_hashes'],x['suite_sha256']) for x in after], 'Metadata changed during full audit')
    result = dict(schema_version=1,audited_at_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,required_complete_evaluations=11,real_rows=22528,script_sha256=sha(Path(__file__)),
        scope='本地已全部完成的 stage3 六组与 stage4 五组；real 四 phase 全样本，teacher 仅已存控制；未推理或改模型。',
        definitions=dict(full_em='与主报告完全相同的 NFKC+casefold+Unicode punctuation/whitespace normalization 后整串相等',
            first_word_accuracy='normalized 输出首词严格等于 gold 首词；标签前缀也会使其错误，另列 gold_first_word_anywhere',
            containment='完整 normalized 三词 gold 以词边界连续出现在输出中；非语义裁判',
            budget_hit='generation_tokens >= 32；只表示达到预算，未保存 EOS token 时不进一步断言截断',
            format_markers='非互斥可见标记，不等价错误；引用/标点已被原 EM normalization 容忍',
            output_shape='empty / legacy_16_single_word / other_single_word / three_words / other_multiword 互斥',
            changed_both_wrong='A/B normalized 输出不同且 A、B 各自完整 EM 均为 0；区别于仅未两边同时答对',
            statistics='重复模型/phase/世界共用同一256facts；不把22528行当独立样本，不按本诊断选模型'),
        runs=reports)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(complete=True,evaluations=len(reports),real_rows=22528,output=str(output))))


def self_test():
    import tempfile
    import unittest
    def row(pred,answer='apple amber reed',world='A',tokens=5):
        return dict(world=world,prediction=pred,target_id='test',generation_tokens=tokens,budget_hit=tokens>=32,
            answer_results={world:dict(answer=answer,em=int(norm(pred)==norm(answer)),
                answer_containment=int((' '+norm(answer)+' ') in (' '+norm(pred)+' ')))})
    class Tests(unittest.TestCase):
        def test_punctuation_is_already_em_tolerant(self):
            self.assertEqual(describe(row('"Apple, amber reed."'))['em'],1)
        def test_format_only_containment(self):
            d=describe(row('Answer: apple amber reed'));self.assertEqual((d['em'],d['containment'],d['first_word_correct']),(0,1,0))
        def test_partial_legacy_is_not_format_recoverable(self):
            d=describe(row('apple'));self.assertEqual((d['shape'],d['containment'],d['first_word_correct']),('legacy_16_single_word',0,1))
        def test_substring_is_not_whole_word(self):
            self.assertEqual(describe(row('pineapple amber reed'))['containment'],0)
        def test_empty_and_budget(self):
            d=describe(row('',tokens=32));self.assertEqual((d['shape'],d['budget_hit']),('empty',1))
        def test_wrong_to_wrong_vs_one_correct(self):
            pairs=[(row('x'),row('y','river frost shell','B')),
                   (row('apple amber reed'),row('z','river frost shell','B'))]
            s=pair_summary(pairs)['numerators'];self.assertEqual((s['changed_both_wrong'],s['changed_exactly_one_correct'],s['changed_but_not_both_correct']),(1,1,2))
        def test_reverse_swap_not_success(self):
            s=pair_summary([(row('river frost shell'),row('apple amber reed','river frost shell','B'))])['numerators']
            self.assertEqual((s['reverse_swapped_answers'],s['changed_both_wrong'],s['both_full_em']),(1,1,0))
        def test_old_constant_single_word_pair_fails(self):
            s=pair_summary([(row('apple'),row('apple','river frost shell','B'))])['numerators']
            self.assertEqual((s['unchanged_both_wrong'],s['changed_both_wrong']),(1,0))
        def test_barrier_stops_before_predictions(self):
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'interface_step3_confirm_a_TEST';p.mkdir()
                (p/'suite.json').write_text(json.dumps(dict(protocol='interface-experiment-v1',complete=False,status='running')))
                with self.assertRaisesRegex(ValueError,'All six suites must be complete'):
                    metadata_barrier(Path(td),'TEST')
        def test_partition_and_example_summary(self):
            s=event_summary([row(''),row('apple'),row('wrong three words'),row('Answer: apple amber reed'),row('apple amber reed')])
            self.assertEqual(sum(s['output_shape_counts'].values()),5)
            self.assertEqual((s['numerators']['em'],s['numerators']['containment']), (1,2))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-final',action='store_true',help='Explicitly authorize reading all final local predictions')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--runs-root',type=Path,default=Path(__file__).resolve().parents[2]/'runs/xtrah100')
    p.add_argument('--tag',default='20261006')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'runs/local/interface_answer_error_audit_20261006.json')
    a=p.parse_args()
    if a.self_test:
        require(not a.execute_final,'Self-test and final execution are mutually exclusive')
        self_test()
    elif a.execute_final:
        execute(a.runs_root,a.tag,a.output)
    else:
        p.error('Waiting for complete final backup: choose --self-test or explicitly --execute-final')

if __name__=='__main__':
    main()
