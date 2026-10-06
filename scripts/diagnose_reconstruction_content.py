"""Posthoc C/D content diagnostics from an already audited reconstruction summary.

python scripts/diagnose_reconstruction_content.py --summary ../plans/reconstruction_dev_summary_20261006.json --runs-root ../runs/xtrah100 --worlds C --output NEW.json

Standard library only; no tensors, model execution, SSH or checkpoint selection.
Only completed known-split cells listed in the supplied summary are opened.
Outputs contain aggregate counts, never raw answers or predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import summarize_reconstruction as summary_audit
from summarize_reconstruction import require, read_json, lines, sha, digest, norm
from diagnose_coldstart_answers import exclusion_markers

PROTOCOL='reconstruction-content-posthoc-v1'
COUNTS=('normalized_em','full_target_containment','exact_own_A','exact_own_B','exact_any_train_AB',
        'exact_other_train_AB','first_two_words_preserved','third_word_correct_at_position',
        'first_two_and_third_correct','three_normalized_words','empty_output','budget_hits',
        'first_fact_R1','first_fact_R4')


class Inputs:
    def __init__(self,root):
        self.root=Path(root).resolve();require(self.root.is_dir(),'Missing runs root');self.hashes={}

    def path(self,name):
        p=Path(name);candidates=[p]
        if 'runs' in p.parts:candidates.append(self.root.joinpath(*p.parts[p.parts.index('runs')+1:]))
        for c in candidates:
            c=c.resolve()
            if c.is_relative_to(self.root) and c.exists():return c
        raise ValueError('Missing local input within runs root: '+str(name))

    def bind(self,path,expected=None):
        path=Path(path);got=sha(path)
        if expected is not None:require(got==expected,'Artifact SHA differs: '+str(path))
        if str(path) in self.hashes:require(self.hashes[str(path)]==got,'Input changed while reading')
        self.hashes[str(path)]=got
        return got

    def unchanged(self):
        require(all(sha(p)==h for p,h in self.hashes.items()),'Input changed during content diagnosis')


def content_score(row,fact,target_index,slots,training_answers):
    """Position tests refer to normalized whitespace words, not tokenizer IDs."""
    world=row['world'];require(world in ('C','D'),'Only C/D target updates are supported')
    expected=norm(fact[world.lower()]);a=norm(fact['a']);b=norm(fact['b'])
    predicted=norm(row['prediction']);gold_words=expected.split();words=predicted.split()
    require(len(gold_words)==len(a.split())==len(b.split())==3,'Expected three-word source payloads')
    require(gold_words[:2]==a.split()[:2] and expected not in training_answers,'Target is not a novel A-prefix combination')
    require(row['answer']==fact[world.lower()] and row['question']==fact['questions'][0]
            and row['id']==row['case_id']==fact['id'] and row['trigger_world']==world
            and row['phase']=='CC' and row['condition']=='real' and row['role']=='updated'
            and not row.get('context'),'Wrong selected prediction identity or leaked context')
    tokens=row['generation_tokens'];trace=row['trace']
    require(type(tokens)is int and 1<=tokens<=32 and len(trace)==tokens,'Missing generation-position routes')
    require(type(row['bank_slots'])is int and row['bank_slots']>0 and row['bank_slots']%slots==0,'Wrong grouped bank size')
    fact_hits=[];slot2_hits=[]
    for position,event in enumerate(trace):
        require(event['phase']==('prefill' if position==0 else 'decode'),'Wrong route position phase')
        indices=event['indices'];require(isinstance(indices,list) and len(indices)==1,'Wrong route batch shape')
        indices=indices[0]
        require(len(indices)==min(4,row['bank_slots']) and len(set(indices))==len(indices)
                and all(type(i)is int and 0<=i<row['bank_slots'] for i in indices),'Invalid sparse route indices')
        fact_hits.append(any(i//slots==target_index for i in indices))
        slot2_hits.append(target_index*3+2 in indices if slots==3 else False)
    first_r1=int(trace[0]['indices'][0][0]//slots==target_index)
    require(row['first_fact_recall_at_1']==first_r1 and row['first_fact_recall_at_4']==int(fact_hits[0])
            and row['decode_hits']==sum(fact_hits[1:]) and row['decode_queries']==tokens-1,'Recorded fact routing counters differ')
    em=int(predicted==expected);require(row['em']==em,'Recorded EM differs from normalized prediction')
    require(row['budget_hit']==(tokens==32),'Recorded budget flag differs')
    prefix=len(words)>=2 and words[:2]==gold_words[:2]
    third=len(words)>=3 and words[2]==gold_words[2]
    return dict(normalized_em=em,full_target_containment=int(' '+expected+' ' in ' '+predicted+' '),
        exact_own_A=int(predicted==a),exact_own_B=int(predicted==b),
        exact_any_train_AB=int(predicted in training_answers),
        exact_other_train_AB=int(predicted in training_answers-{a,b}),
        first_two_words_preserved=int(prefix),third_word_correct_at_position=int(third),
        first_two_and_third_correct=int(prefix and third),three_normalized_words=int(len(words)==3),
        empty_output=int(not words),budget_hits=int(tokens==32),generation_tokens=tokens,
        normalized_words=len(words),characters=len(row['prediction']),first_fact_R1=first_r1,
        first_fact_R4=int(fact_hits[0]),decode_fact_hits=sum(fact_hits[1:]),decode_queries=tokens-1,
        first_target_slot2_inclusion=int(slot2_hits[0]) if slots==3 else None,
        decode_target_slot2_hits=sum(slot2_hits[1:]) if slots==3 else None,
        any_generation_target_slot2_inclusion=int(any(slot2_hits)) if slots==3 else None)


def aggregate(scores):
    require(scores,'Cannot aggregate no completed C/D predictions');n=len(scores)
    counts={k:sum(r[k] for r in scores) for k in COUNTS}
    lengths={}
    for key in ('generation_tokens','normalized_words','characters'):
        values=[r[key] for r in scores]
        lengths[key]=dict(total=sum(values),mean=sum(values)/n,minimum=min(values),maximum=max(values),
                         histogram=dict(sorted(Counter(values).items())))
    queries=sum(r['decode_queries'] for r in scores);hits=sum(r['decode_fact_hits'] for r in scores)
    slot_rows=[r for r in scores if r['first_target_slot2_inclusion'] is not None]
    slot2=dict(applicable_prediction_exposures=len(slot_rows),first_position_inclusions=None,
               decode_hits=None,decode_queries=None,any_generation_inclusions=None)
    if slot_rows:
        sq=sum(r['decode_queries'] for r in slot_rows);sh=sum(r['decode_target_slot2_hits'] for r in slot_rows)
        slot2.update(first_position_inclusions=sum(r['first_target_slot2_inclusion'] for r in slot_rows),
            decode_hits=sh,decode_queries=sq,decode_hit_rate=sh/sq if sq else None,
            any_generation_inclusions=sum(r['any_generation_target_slot2_inclusion'] for r in slot_rows),
            no_generation_inclusions=sum(not r['any_generation_target_slot2_inclusion'] for r in slot_rows))
    return dict(prediction_exposures=n,counts=counts,rates={k:v/n for k,v in counts.items()},lengths=lengths,
                decode_fact_routing=dict(hits=hits,queries=queries,hit_rate=hits/queries if queries else None),
                third_payload_slot_routing=slot2)


def prepared_rows(report,inputs):
    caches={}
    for entry in report['preparation']:
        directory=inputs.path(entry['run_dir'])
        if exclusion_markers(directory,inputs.root):continue
        path=directory/'manifest.json';inputs.bind(path,entry['manifest_sha256']);manifest=read_json(path)
        require(manifest['protocol']==summary_audit.RUN_PROTOCOL and manifest['stage']=='prepare'
                and manifest['complete'] is True,'Incomplete/wrong preparation source')
        loaded={}
        for split in ('train','known'):
            metadata=manifest['result'][split];rp=directory/(split+'.json');cp=directory/(split+'.pt')
            inputs.bind(rp);inputs.bind(cp,metadata['cache_sha256']);rows=read_json(rp)
            require(digest(rows)==metadata['rows_sha256'] and len(rows)==metadata['records'],'Prepared rows provenance differs')
            require(len({r['id'] for r in rows})==len(rows),'Duplicate prepared IDs');loaded[split]=rows
        train={r['id']:r for r in loaded['train']};known={r['id']:r for r in loaded['known']}
        require(set(train)==set(known),'Known IDs differ from training projection')
        require(all(train[i][k]==known[i][k] for i in train for k in ('a','b')),'Known A/B differ from actual training answers')
        require(all(r['worlds']==['A','B'] and len(r['questions'])==1 for r in train.values()),'Wrong training projection')
        answers={norm(r[k]) for r in train.values() for k in ('a','b')}
        require(len(answers)==2*len(train),'Training complete-answer strings are not unique')
        key=manifest['result']['known']['cache_sha256'];require(key not in caches,'Ambiguous known cache preparation')
        caches[key]=dict(rows=loaded['known'],training_answers=answers)
    return caches


def diagnose(summary_path,runs_root,worlds=('C',)):
    require(set(worlds)<= {'C','D'} and worlds and len(worlds)==len(set(worlds)),'Choose unique C and/or D worlds')
    summary_path=Path(summary_path).resolve();report=read_json(summary_path);summary_sha=sha(summary_path)
    require(report['protocol']==summary_audit.PROTOCOL and report['audit_passed'] is True,'Input summary has not passed strict reconstruction audit')
    inputs=Inputs(runs_root);caches=prepared_rows(report,inputs);results=[];excluded=[];grouped=defaultdict(list);seen=set()
    for record in report['records']:
        if record['kind']!='evaluation' or record['split']!='known':continue
        world='C' if record['part']=='development' else 'D' if record['part']=='confirmation' else None
        if world not in worlds:continue
        directory=inputs.path(record['run_dir']);markers=exclusion_markers(directory,inputs.root)
        if markers:excluded.append(dict(run_dir=str(directory),markers=markers));continue
        require('smoke' not in directory.name.lower(),'Smoke record cannot enter posthoc formal diagnostics')
        for name,expected in record['artifacts'].items():
            require(Path(name).name==name,'Unsafe artifact name');inputs.bind(directory/name,expected)
        manifest=read_json(directory/'manifest.json')
        require(manifest['protocol']==summary_audit.RUN_PROTOCOL and manifest['stage']=='eval'
                and manifest['complete'] is True and manifest['backbone_unchanged'] is True,'Not a complete frozen formal evaluation')
        require(manifest['cache_sha256']==record['cache_sha256'] and manifest['checkpoint_sha256']==record['checkpoint_sha256']
                and manifest['configuration']['eval_part']==record['part'],'Summary/manifest provenance differs')
        source=caches[record['cache_sha256']];rows=source['rows']
        # Reuse the strict scalar audit rather than invent a weaker C-only
        # schema: it verifies all ordered reads/pairs of this complete cell.
        verified=summary_audit.audit_evaluation(directory,manifest,rows,'known')
        for key in ('arm','seed','split','part','architecture','groups','artifacts','checkpoint_sha256','cache_sha256'):
            require(verified[key]==record[key],'Bound summary differs on '+key)
        identity=record['arm'],record['seed'],world;require(identity not in seen,'Duplicate completed arm/seed/world');seen.add(identity)
        byid={r['id']:(i,r) for i,r in enumerate(rows)};scores=[];selected=set()
        for raw in lines(directory/'predictions.jsonl'):
            if (raw['world'],raw['phase'],raw['condition'],raw['role'])!=(world,'CC','real','updated'):continue
            require(raw['id'] in byid and raw['id'] not in selected,'Unexpected or duplicate C/D fact')
            selected.add(raw['id']);index,fact=byid[raw['id']]
            require(raw['bank_slots']==len(rows)*record['architecture']['slots'],'Wrong C/D bank fact coverage')
            scores.append(content_score(raw,fact,index,record['architecture']['slots'],source['training_answers']))
        require(selected==set(byid),'Incomplete C/D updated prediction grid')
        out=aggregate(scores);group=record['groups']['CC_'+world]
        require(group['count']==len(scores) and group['updated_em']==out['rates']['normalized_em'],'C/D summary count/EM mismatch')
        results.append(dict(arm=record['arm'],seed=record['seed'],world=world,run_dir=str(directory),
            phase='CC',split='known',part=record['part'],raw_sha256=record['artifacts']['predictions.jsonl'],
            checkpoint_sha256=record['checkpoint_sha256'],diagnostics=out))
        grouped[(record['arm'],world)].extend(scores)
    require(results,'No complete requested C/D evaluations in the supplied summary')
    inputs.unchanged();require(sha(summary_path)==summary_sha,'Input summary changed during diagnosis')
    return dict(protocol=PROTOCOL,created_at=datetime.now(timezone.utc).isoformat(),checks_passed=True,
        posthoc=True,may_select_or_tune_models=False,worlds=list(worlds),cells=len(results),
        complete_for_listed_eligible_cells=True,excluded=excluded,per_run=results,
        descriptive_per_arm_world=[dict(arm=arm,world=world,diagnostics=aggregate(scores)) for (arm,world),scores in sorted(grouped.items())],
        summary_binding=dict(path=str(summary_path),sha256=summary_sha),input_sha256=inputs.hashes,
        input_hashes_unchanged=True,definitions=dict(
            exact_match='Full normalized string equality, using the declared NFKC/case/punctuation/whitespace normalization.',
            any_train_AB='Any complete A or B answer from the actual canonical training projection, including own A/B.',
            first_two_words='First two normalized whitespace words equal target C/D prefix (the same prefix as own A).',
            third_word='Normalized whitespace word at zero-based position 2 equals the new target third word; additional words allowed.',
            slot2='S3 only: exact flattened index target_fact_index*3+2; S1 is not applicable.',
            routing='First position predicts the first generated token; decode counts exclude first position.'),
        limitations=[
            'Posthoc descriptive diagnostics only, never checkpoint/architecture selection, gates or training changes.',
            'The same 16 facts recur across seeds and arms; aggregate counts are repeated prediction exposures, not independent facts.',
            'Fact R@4 can hit any slot. It does not establish retrieval of the third-word slot.',
            'Route traces lack generated token IDs; slot2 access is not aligned to the third-word prediction time.',
            'Retrieval inclusion does not prove useful value contribution or causal readout failure. No gradient or intervention claim is made.',
            'Exact old-label repetition is a description of outputs, not proof of where the label was stored or why it was emitted.',
            'Only complete requested cells in the supplied summary are read; no claim that all planned confirmations are complete.'])


def self_test():
    import copy
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def fixture(self,prediction='amber birch crane',slots=3):
            fact=dict(id='id0',a='amber birch crane',b='delta elm finch',c='amber birch goose',d='amber birch hawk',questions=['Recall id0'])
            route=[0,1,3,4] if slots==3 else [0,1,2,3]
            trace=[dict(phase='prefill',indices=[route]),dict(phase='decode',indices=[[2,1,3,4] if slots==3 else route])]
            row=dict(id='id0',case_id='id0',world='C',trigger_world='C',phase='CC',condition='real',role='updated',
                question=fact['questions'][0],answer=fact['c'],prediction=prediction,em=int(norm(prediction)==fact['c']),
                generation_tokens=2,bank_slots=4*slots,trace=trace,budget_hit=False,first_fact_recall_at_1=1,
                first_fact_recall_at_4=1,decode_hits=1,decode_queries=1)
            return row,fact,0,slots,{fact['a'],fact['b'],'iris juniper kite'}

        def test_old_A_has_prefix_without_new_third_and_fact_hit_without_slot2(self):
            result=content_score(*self.fixture())
            self.assertEqual((result['exact_own_A'],result['exact_any_train_AB'],result['first_two_words_preserved']),(1,1,1))
            self.assertEqual((result['third_word_correct_at_position'],result['first_fact_R4'],result['first_target_slot2_inclusion']),(0,1,0))
            self.assertEqual(result['decode_target_slot2_hits'],1)

        def test_B_other_training_answer_and_correct_plus_extra_are_distinct(self):
            b=content_score(*self.fixture('delta elm finch'));other=content_score(*self.fixture('iris juniper kite'))
            correct=content_score(*self.fixture('AMBER, birch goose extra'))
            self.assertEqual(b['exact_own_B'],1);self.assertEqual(other['exact_other_train_AB'],1)
            self.assertEqual((correct['normalized_em'],correct['full_target_containment'],correct['first_two_and_third_correct']),(0,1,1))

        def test_S1_slot2_is_not_applicable(self):
            score=content_score(*self.fixture(slots=1));out=aggregate([score])
            self.assertIsNone(score['first_target_slot2_inclusion'])
            self.assertEqual(out['third_payload_slot_routing']['applicable_prediction_exposures'],0)
            self.assertIsNone(out['third_payload_slot_routing']['decode_queries'])

        def test_wrong_EM_ID_and_route_are_rejected(self):
            for field,value in [('em',1),('id','other'),('decode_hits',0),('first_fact_recall_at_1',0),('answer','wrong')]:
                args=list(self.fixture());args[0][field]=value
                with self.assertRaises(ValueError):content_score(*args)
            args=list(self.fixture());args[0]['trace'][1]['indices']=[[2,2,3,4]]
            with self.assertRaises(ValueError):content_score(*args)

        def test_D_is_scored_against_D_and_mixed_slot_aggregate_keeps_denominator(self):
            args=list(self.fixture('amber birch hawk'));args[0].update(world='D',trigger_world='D',answer=args[1]['d'],em=1)
            score=content_score(*args);self.assertEqual(score['normalized_em'],1)
            out=aggregate([score,content_score(*self.fixture(slots=1))])
            self.assertEqual(out['prediction_exposures'],2)
            self.assertEqual(out['third_payload_slot_routing']['applicable_prediction_exposures'],1)
            self.assertEqual(out['third_payload_slot_routing']['decode_queries'],1)

        def test_SHA_mutation_no_overwrite_and_exclusion_before_raw(self):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);path=root/'raw';path.write_text('first');inputs=Inputs(root)
                expected=inputs.bind(path);path.write_text('changed')
                with self.assertRaisesRegex(ValueError,'SHA'):Inputs(root).bind(path,expected)
                with self.assertRaisesRegex(ValueError,'changed'):inputs.unchanged()
                with self.assertRaisesRegex(ValueError,'overwrite'):main(['--output',str(path)])
                self.assertEqual(path.read_text(),'changed')
                (root/'analysis_excluded.json').write_text(json.dumps(dict(reason='excluded fixture')))
                self.assertTrue(exclusion_markers(root,root))

    return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path);p.add_argument('--runs-root',type=Path)
    p.add_argument('--worlds',nargs='+',choices=('C','D'),default=['C']);p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true');args=p.parse_args(argv)
    if args.self_test:return self_test()
    if args.output is not None:require(not args.output.exists(),'Refusing to overwrite output')
    if args.summary is None or args.runs_root is None or args.output is None:p.error('--summary, --runs-root and --output are required')
    start=time.perf_counter();result=diagnose(args.summary,args.runs_root,args.worlds)
    result.update(elapsed_seconds=time.perf_counter()-start,script_sha256=sha(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(cells=result['cells'],worlds=result['worlds'],checks_passed=True,output=str(args.output))))
    return 0


if __name__=='__main__':raise SystemExit(main())
