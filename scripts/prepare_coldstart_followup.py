"""Apply the fixed train-only teacher gate and write optional follow-up plans."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
REMOTE='/ssd3/chenhan/VeRA-Mem-Workspace'
PROBE='coldstart_teacher_probe_v2_20261006/probe'
SUITE='coldstart_teacher_repair_20261006'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n')

def main():
    local=WORKSPACE/'runs/xtrah100'/PROBE
    manifest=json.loads((local/'manifest.json').read_text())
    metrics=json.loads((local/'metrics.json').read_text())
    assert manifest['complete'] and metrics['complete'] and metrics['backbone_unchanged']
    assert manifest['no_dev_or_confirm'] and metrics['source_split']=='train'
    validation=metrics['subsets']['validation']
    selected=None
    for strategy in ('quote_instruction','gold_annotated'):
        entry=validation[strategy]
        assert entry['count']==64
        if entry['paired_em']>=.9:
            selected=strategy;break
    evidence=dict(selected_strategy=selected,rule='quote>=0.9 else annotated>=0.9 else none',
        protocol_document_sha256=sha(ROOT/'docs/coldstart_protocol.md'),
        source_split='train',validation=validation)
    for key,filename in [('probe_manifest','manifest.json'),('probe_metrics','metrics.json'),('probe_predictions','predictions.jsonl')]:
        evidence[key]=dict(path=f'{REMOTE}/runs/{PROBE}/{filename}',sha256=sha(local/filename))
    destination=WORKSPACE/'plans/coldstart_teacher_selection_20261006.json'
    write(destination,evidence)
    if selected is None:
        print(json.dumps(dict(selected_strategy=None,plans_written=False)));return
    model=f'{REMOTE}/models/Qwen3-4B-Instruct-2507'
    wiki=f'{REMOTE}/runs/coldstart_repair_20261006/features/train.pt'
    synthetic=f'{REMOTE}/runs/coldstart_prepare_synthetic_20261006/features/train.pt'
    prefix=f'{REMOTE}/runs/{SUITE}'
    initial=f'{REMOTE}/runs/coldstart_prepare_wikipedia_20261006/initial/last.pt'
    def job(name,stage,cache,checkpoint,domain,extra=()):
        return dict(name=name,entry='coldstart',arguments=['--stage',stage,'--cache',cache,'--model',model,
            '--arm',name,'--checkpoint',checkpoint,'--domain',domain,*extra])
    jobs=[job('wiki_teacher_warm','train',wiki,initial,'wikipedia',['--updates','1024','--teacher-strategy',selected]),
        job('teacher_prototypes','init',wiki,prefix+'/wiki_teacher_warm/last.pt','wikipedia',['--cluster-init']),
        job('wiki_teacher_repair','train',synthetic,prefix+'/teacher_prototypes/last.pt','synthetic',
            ['--updates','512','--alpha-base','0.25','--base-trainable','--routing','straight_through'])]
    write(WORKSPACE/'plans'/(SUITE+'.json'),jobs)
    for split in ('dev','confirm','probe'):
        evaluations=[]
        for domain in (('synthetic_train_probe',) if split=='probe' else ('wikipedia','synthetic')):
            cache=(f'{REMOTE}/data/coldstart_wiki_20261006/synthetic_train_probe.pt' if split=='probe'
                else f'{REMOTE}/runs/coldstart_prepare_{domain}_20261006/features/{split}.pt')
            name='wiki_teacher_repair_'+domain+'_full'
            extra=['--base-mode','full','--max-cases',str({'dev':16,'confirm':64,'probe':8}[split]),'--seed','63042']
            entry=job('wiki_teacher_repair','eval',cache,prefix+'/wiki_teacher_repair/last.pt',domain,extra)
            entry['name']=name;evaluations.append(entry)
        write(WORKSPACE/'plans'/f'coldstart_teacher_repair_eval_{split}_20261006.json',evaluations)
    print(json.dumps(dict(selected_strategy=selected,plans_written=True,selection_evidence=str(destination))))

if __name__=='__main__':main()
