"""Descriptive answer/route diagnostics for completed QKV runs; no generation.

Word-position scores require exactly three normalized output words. Route masses
include all generated tokens, including EOS, and are NOT aligned to gold words.
Repeated cases/seeds are not treated as independent statistical observations.
"""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vera_mem.metrics import normalize_answer


def analyze(root):
    preparations=list(root.glob('qkv_prepare_*/features/dataset.json'))
    if len(preparations)!=1:raise ValueError('Require exactly one QKV dataset')
    dataset=json.loads(preparations[0].read_text())
    payloads={normalize_answer(p) for p in dataset['train']['payloads']}
    rows={r['id']:r for split in ('known','dev','confirm') for r in dataset[split]}
    out=[];inputs={str(preparations[0]):hashlib.sha256(preparations[0].read_bytes()).hexdigest()}
    for path in sorted(root.glob('qkv_*/*/manifest.json')):
        manifest=json.loads(path.read_text());cfg=manifest.get('configuration',{})
        if manifest.get('complete') is not True or cfg.get('stage')!='eval' or cfg.get('eval_part')=='smoke':continue
        if (manifest.get('protocol')!='qkv-attention-memory-v1'
                or manifest.get('backbone_unchanged') is not True
                or cfg.get('eval_part') not in ('development','confirmation')
                or manifest.get('result',{}).get('complete') is not True):
            raise ValueError('Require complete frozen QKV evaluation evidence')
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        prediction=path.with_name('predictions.jsonl')
        inputs[str(prediction)]=hashlib.sha256(prediction.read_bytes()).hexdigest()
        groups=defaultdict(list)
        for line in prediction.read_text().splitlines():
            p=json.loads(line)
            if p['role'] not in ('updated','control'):continue
            r=rows[p['id']];pred=normalize_answer(p['prediction']);words=pred.split();answer=normalize_answer(p['answer']).split()
            field={'B':'b','C':'c','D':'d','SWAP':'swap_a'}.get(p['world'])
            if field is None or normalize_answer(r[field]).split()!=answer or len(answer)!=3:
                raise ValueError('Expected the declared three-word target from the dataset')
            em=int(pred==normalize_answer(p['answer']))
            if em!=p['em']:raise ValueError('Stored EM disagrees with raw answer')
            trace=p['trace'];position=r['provenance']['edited_word_index']
            if not trace or len(trace)!=p['generation_tokens'] or len(trace)!=len(p['generated_token_ids']):
                raise ValueError('Require one route per generated token, including EOS')
            if any(t['token_id']!=token for t,token in zip(trace,p['generated_token_ids'])):
                raise ValueError('Route and generated token IDs differ')
            if p['world'] in ('C','D') and (type(position)is not int or position not in (0,1,2)):
                raise ValueError('Invalid edited word position')
            item=dict(em=em,exactly_three_words=int(len(words)==3),
                output_is_training_payload=int(pred in payloads),output_is_own_old_a=int(pred==normalize_answer(r['a'])),
                output_is_own_old_b=int(pred==normalize_answer(r['b'])),
                first_fact_recall_at_1=p['first_fact_recall_at_1'],first_fact_recall_at_read=p['first_fact_recall_at_read'],
                first_target_fact_mass=trace[0]['target_fact_mass'],
                mean_target_fact_mass=sum(t['target_fact_mass'] for t in trace)/len(trace))
            for i in range(3):item['word_'+str(i)+'_correct']=int(len(words)==3 and words[i]==answer[i])
            if p['world'] in ('C','D'):
                item.update(edited_position=position,edited_word_correct=item['word_'+str(position)+'_correct'],
                    both_unchanged_words_correct=int(all(item['word_'+str(i)+'_correct'] for i in range(3) if i!=position)),
                    first_edited_slot_mass=trace[0]['target_slot_mass'][position],
                    mean_edited_slot_mass=sum(t['target_slot_mass'][position] for t in trace)/len(trace),
                    max_edited_slot_mass=max(t['target_slot_mass'][position] for t in trace))
            groups[(p['phase'],p['world'],p['condition'])].append(item)
        stats=[]
        for (phase,world,condition),values in groups.items():
            metrics={k:sum(v[k] for v in values)/len(values) for k in values[0] if k!='edited_position'}
            by_position={}
            if world in ('C','D'):
                for i in range(3):
                    selected=[v for v in values if v['edited_position']==i]
                    by_position[str(i)]=dict(count=len(selected),em=sum(v['em'] for v in selected)/len(selected) if selected else None,
                        edited_word_correct=sum(v['edited_word_correct'] for v in selected)/len(selected) if selected else None)
            stats.append(dict(phase=phase,world=world,condition=condition,count=len(values),metrics=metrics,edited_positions=by_position))
        architecture=manifest['result']['architecture']
        arm=Path(cfg['checkpoint']).parent.name.rsplit('_seed',1)[0]
        out.append(dict(run_dir=str(path.parent),arm=arm,seed=cfg['seed'],part=cfg['eval_part'],split=Path(cfg['cache']).stem,
            architecture=architecture,groups=stats))
    return dict(protocol='qkv-answer-route-description-v1',records=out,input_sha256=inputs,
        limitations=['Descriptive only; not used for architecture selection or gate changes.',
            'Per-position words require exactly three whitespace-separated normalized output words.',
            'Mean/max slot masses cover the complete free-generation trajectory, including EOS; no gold-word alignment is implied.',
            'Group means give each prediction equal weight; mean trajectory masses are not pooled-token averages. Missing edited-position strata have null rates.',
            'Fact/slot inclusion is not itself evidence of correct content decoding.',
            'Contextual slot vectors are not isolated words: editing an earlier word can change later slots. Edited-slot mass is not exclusive information attribution.',
            'Target cases recur across seeds/arms; no independent-sample confidence interval is claimed.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.runs_root)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(runs=len(result['records']),output=str(a.output))))
