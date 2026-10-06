"""Supplementary teacher selection must not enlarge or replace the main matrix."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest
pytest.importorskip('torch')
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('coldstart_followup_audit',SCRIPTS/'audit_coldstart.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def probe_fixture(quote=58,annotated=64):
    from probe_coldstart_teacher import select_rows,REVISION
    from vera_mem.coldstart_teacher import STRATEGIES,make_teacher_context
    rows=[]
    for i in range(80):
        anchor='a unique preceding anchor'
        a,b=f'Alpha{i} three words',f'Beta{i} three words'
        rows.append(dict(id=f'id{i}',split='train',a=a,b=b,questions=[f'Which words follow {anchor}?'],
            supports=[[f'Note {i}: {anchor} {a}'],[f'Note {i}: {anchor} {b}']],provenance=dict(anchor=anchor)))
    packet=dict(protocol='dictionary-coldstart-v1',model_revision=REVISION,split='train',task='wikipedia',rows=rows)
    selection=select_rows(packet);events=[];counts={}
    for subset,indices in selection['selected_indices'].items():
        counts[subset]={}
        for strategy in STRATEGIES:
            success={'baseline':0,'quote_instruction':quote,'gold_annotated':annotated}[strategy] if subset=='validation' else 0
            for rank,index in enumerate(indices):
                row=rows[index]
                for world,wi in (('A',0),('B',1)):
                    answer=row[world.lower()];correct=world=='A' or rank<success
                    events.append(dict(subset=subset,strategy=strategy,source_index=index,world=world,id=row['id'],
                        answer=answer,question=row['questions'][0],source_context=row['supports'][wi][0],
                        teacher_context=make_teacher_context(row,row['supports'][wi][0],answer,strategy),
                        prediction=answer if correct else 'wrong answer here',em=int(correct)))
            counts[subset][strategy]=dict(count=len(indices),both_correct=min(success,len(indices)),
                a_correct=len(indices),b_correct=min(success,len(indices)),paired_em=min(success,len(indices))/len(indices))
    metrics=dict(protocol='coldstart-teacher-feasibility-v1',complete=True,selection=selection,subsets=counts,
        max_new_tokens=32,teacher_only=True,shared_question=True,gradient_steps=0,backbone_unchanged=True,
        backbone_hash_before='frozen',backbone_hash_after='frozen')
    manifest=dict(protocol=metrics['protocol'],complete=True,no_dev_or_confirm=True,selection=selection,
        result=metrics,model_revision=REVISION)
    return packet,manifest,metrics,events


@pytest.mark.parametrize('quote,annotated,expected',[(58,64,'quote_instruction'),(57,58,'gold_annotated'),(57,57,None)])
def test_selection_uses_64_validation_pairs_fixed_priority_not_calibration(quote,annotated,expected):
    data=probe_fixture(quote,annotated)
    result=audit.verify_teacher_probe(*data)
    assert result['selected_strategy']==expected and result['raw_observations']==480
    assert result['no_dev_or_confirm'] and result['questions_unchanged']


@pytest.mark.parametrize('corrupt',['question','source','teacher_context','answer','em','missing','duplicate','selection','counts','gradient'])
def test_probe_rejects_corrupt_or_unpaired_evidence(corrupt):
    packet,manifest,metrics,events=probe_fixture()
    if corrupt in {'question','source','teacher_context','answer','em'}:
        key={'source':'source_context'}.get(corrupt,corrupt)
        events[0][key]=0 if key=='em' else 'changed'
    if corrupt=='missing':events.pop()
    if corrupt=='duplicate':events[-1]=copy.deepcopy(events[0])
    if corrupt=='selection':metrics['selection']['selected_indices']['validation'][0]=0
    if corrupt=='counts':metrics['subsets']['validation']['quote_instruction']['both_correct']+=1
    if corrupt=='gradient':metrics['gradient_steps']=1
    with pytest.raises(ValueError):audit.verify_teacher_probe(packet,manifest,metrics,events)


def followup_fixture():
    totals=dict(gold_target_tokens=1,distillation_target_tokens=1,target_tokens=1,student_input_tokens=2,teacher_input_tokens=3)
    cfg=dict(model='model',routing='sparse',base_trainable=False,alpha_base=0.,seed=63042,batch_size=8)
    warm=dict(arm='wiki_teacher_warm',teacher_strategy='quote_instruction',cache_sha256='wiki-cache',updates=1024,
        target_exposures=8192,schedule_sha256='wiki-schedule',gold_token_schedule_sha256='wiki-gold',
        initial_checkpoint_sha256='common-init',final_checkpoint_sha256='teacher-warm',token_totals=copy.deepcopy(totals),configuration=cfg)
    final=dict(arm='wiki_teacher_repair',teacher_strategy='baseline',cache_sha256='synthetic-cache',updates=512,
        target_exposures=4096,schedule_sha256='synthetic-schedule',gold_token_schedule_sha256='synthetic-gold',
        initial_checkpoint_sha256='teacher-prototypes',final_checkpoint_sha256='teacher-final',token_totals=copy.deepcopy(totals),
        configuration=dict(cfg,routing='straight_through',base_trainable=True,alpha_base=.25))
    mains=[dict(copy.deepcopy(warm),arm='wiki_warm',teacher_strategy='baseline',final_checkpoint_sha256='main-warm'),
        dict(copy.deepcopy(final),arm='straight_through',initial_checkpoint_sha256='main-prototypes',final_checkpoint_sha256='main-final')]
    inits=[dict(initial_checkpoint_sha256='teacher-warm',cache_sha256='wiki-cache',final_checkpoint_sha256='teacher-prototypes')]
    selections=[dict(selected_strategy='quote_instruction',cache_sha256='wiki-cache',model='model')]
    cells=[(s,d) for s in ('dev','confirm') for d in ('wikipedia','synthetic')]+[('train_probe','synthetic_train_probe')]
    evaluations=[dict(scope=s,domain=d,arm='wiki_teacher_repair',base_mode='full',checkpoint_sha256='teacher-final',
        arithmetic=dict(comparison_key=s+d,costs=dict(generation_calls=1))) for s,d in cells]
    main_evals=[dict(copy.deepcopy(e),arm='straight_through',checkpoint_sha256='main-final') for e in evaluations]
    return [warm,final],evaluations,inits,mains,main_evals,selections,None


def test_followup_is_optional_zero_or_exact_two_training_five_eval_separate_scope():
    absent=audit.audit_followup([],[],[],[],[],[],None,strict=True)
    assert absent['status']=='absent' and absent['complete']
    result=audit.audit_followup(*followup_fixture(),strict=True)
    assert result['status']=='passed' and result['main_counts_unchanged']
    assert len(result['trainings'])==2 and len(result['evaluations'])==5
    assert result['verified_generation_calls']==5


@pytest.mark.parametrize('corrupt',['missing_eval','missing_train','duplicate_eval','compression','strategy','schedule','cache','common_init','prototype_parent','final_parent','synthetic_teacher','token_budget','cases','eval_checkpoint'])
def test_followup_rejects_incomplete_or_changed_comparison(corrupt):
    args=list(followup_fixture());tr,ev,init,main,me,sel,_=args
    if corrupt=='missing_eval':ev.pop()
    if corrupt=='missing_train':tr.pop()
    if corrupt=='duplicate_eval':ev.append(copy.deepcopy(ev[0]))
    if corrupt=='compression':ev[0]['base_mode']='off'
    if corrupt=='strategy':tr[0]['teacher_strategy']='gold_annotated'
    if corrupt=='schedule':tr[0]['schedule_sha256']='changed'
    if corrupt=='cache':tr[0]['cache_sha256']='changed'
    if corrupt=='common_init':tr[0]['initial_checkpoint_sha256']='changed'
    if corrupt=='prototype_parent':init[0]['initial_checkpoint_sha256']='changed'
    if corrupt=='final_parent':tr[1]['initial_checkpoint_sha256']='changed'
    if corrupt=='synthetic_teacher':tr[1]['teacher_strategy']='gold_annotated'
    if corrupt=='token_budget':tr[1]['token_totals']['student_input_tokens']+=1
    if corrupt=='cases':ev[0]['arithmetic']['comparison_key']='changed'
    if corrupt=='eval_checkpoint':ev[0]['checkpoint_sha256']='changed'
    with pytest.raises(ValueError):audit.audit_followup(*args,strict=True)


def test_partial_followup_never_reports_complete_in_monitoring_mode():
    args=list(followup_fixture());args[1].pop()
    result=audit.audit_followup(*args)
    assert result['status']=='pending' and not result['complete'] and len(result['missing_evaluation_cells'])==1


@pytest.mark.parametrize('corrupt',[None,'snapshot','protocol','predictions','source','strategy','selection_file'])
def test_selection_evidence_is_bound_to_probe_and_frozen_training_source(tmp_path,monkeypatch,corrupt):
    packet,manifest,metrics,events=probe_fixture(57,60)
    directory=tmp_path/'suite';probe=tmp_path/'probe'
    manifest['configuration']=dict(cache='cache.pt',model='pinned-model')
    manifest['cache_sha256']='cache-sha'
    manifest['source_sha256']={k:k+'-sha' for k in ('probe_coldstart_teacher.py','coldstart_teacher.py','coldstart_data.py')}
    metrics['artifacts']={'predictions.jsonl':'predictions-sha','selection.json':'selection-sha'}
    files={directory/'teacher_selection.json':dict(selected_strategy='gold_annotated',rule=audit.SELECTION_RULE,
        protocol_document_sha256='protocol-sha',probe_manifest=dict(path=str(probe/'manifest.json'),sha256='manifest-sha'),
        probe_metrics=dict(path=str(probe/'metrics.json'),sha256='metrics-sha'),
        probe_predictions=dict(path=str(probe/'predictions.jsonl'),sha256='predictions-sha')),
        probe/'manifest.json':manifest,probe/'metrics.json':metrics,probe/'selection.json':copy.deepcopy(metrics['selection'])}
    shas={directory/'teacher_selection.json':'evidence-sha',probe/'manifest.json':'manifest-sha',probe/'metrics.json':'metrics-sha',
        probe/'predictions.jsonl':'predictions-sha',probe/'selection.json':'selection-sha'}
    suite=dict(selection_evidence_sha256='evidence-sha',teacher_selection=dict(sha256='evidence-sha'),source_files_sha256={
        'docs/coldstart_protocol.md':'protocol-sha','scripts/probe_coldstart_teacher.py':'probe_coldstart_teacher.py-sha',
        'src/vera_mem/coldstart_teacher.py':'coldstart_teacher.py-sha','src/vera_mem/coldstart_data.py':'coldstart_data.py-sha'})
    if corrupt=='snapshot':suite['selection_evidence_sha256']='wrong'
    if corrupt=='protocol':files[directory/'teacher_selection.json']['protocol_document_sha256']='wrong'
    if corrupt=='predictions':shas[probe/'predictions.jsonl']='wrong'
    if corrupt=='source':manifest['source_sha256']['coldstart_teacher.py']='wrong'
    if corrupt=='strategy':files[directory/'teacher_selection.json']['selected_strategy']='quote_instruction'
    if corrupt=='selection_file':files[probe/'selection.json']['selected_indices']['calibration'][0]=79
    class Inputs:
        def resolve(self,path):return Path(path)
        def sha(self,path):return shas[Path(path)]
        def load(self,path,expected=None):
            assert path=='cache.pt' and expected=='cache-sha'
            return packet
    monkeypatch.setattr(audit,'read_json',lambda path:files[Path(path)])
    monkeypatch.setattr(audit,'read_jsonl',lambda path:events)
    if corrupt:
        with pytest.raises(ValueError):audit.audit_teacher_selection(directory,suite,Inputs())
    else:
        proof=audit.audit_teacher_selection(directory,suite,Inputs())
        assert proof['selected_strategy']=='gold_annotated' and proof['evidence_sha256']=='evidence-sha'
