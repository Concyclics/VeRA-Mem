"""Fixed six-arm full-block experiment; no outcome-dependent model selection."""
from pathlib import Path
import json
ROOT='/ssd3/chenhan/VeRA-Mem-Workspace'
DATE='20261006'
SEEDS=(91042,91043,91044)
MODES=('diagonal','pooled_outer','block_outer')
ARMS=tuple(f'{r}_{m}' for r in ('static','rebind') for m in MODES)
MODEL=ROOT+'/models/Qwen3-4B-Instruct-2507'
FEATURES=ROOT+'/runs/block_prepare_'+DATE+'/features'

def job(name,stage,cache=None,**kwargs):
    args=['--stage',stage,'--model',MODEL]
    if cache is not None:args+=['--cache',cache]
    for k,v in kwargs.items():args+=['--'+k.replace('_','-'),str(v)]
    return dict(name=name,entry='block',arguments=args)

def plans():
    out={f'block_prepare_{DATE}':[job('features','prepare',data_seed=221042)],
         f'block_preflight_{DATE}':[job('teacher_train','teacher',FEATURES+'/train.pt',eval_part='preflight')]}
    smoke=[]
    name=f'block_smoke_{DATE}'
    for mode in MODES:
        child='smoke_'+mode
        smoke.append(job(child,'train',FEATURES+'/train.pt',block_mode=mode,regime='rebind',updates=2))
        smoke.append(job(child+'_eval','eval',FEATURES+'/known.pt',checkpoint=f'{ROOT}/runs/{name}/{child}/last.pt',eval_part='smoke'))
    out[name]=smoke
    for seed in SEEDS:
        name=f'block_train_{seed}_{DATE}';train=[];dev=[];confirm=[]
        for regime in ('static','rebind'):
            for mode in MODES:
                child=f'{regime}_{mode}_seed{seed}'
                train.append(job(child,'train',FEATURES+'/train.pt',seed=seed,regime=regime,block_mode=mode,updates=2048))
                checkpoint=f'{ROOT}/runs/{name}/{child}/last.pt'
                for split,part,target in [('known','development',dev),('dev','development',dev),('known','confirmation',confirm),('confirm','confirmation',confirm)]:
                    target.append(job(f'{child}_{split}_{part}','eval',f'{FEATURES}/{split}.pt',checkpoint=checkpoint,seed=seed,eval_part=part))
        out[name]=train;out[f'block_development_{seed}_{DATE}']=dev;out[f'block_confirmation_{seed}_{DATE}']=confirm
    for part,splits in [('development',('known','dev')),('confirmation',('known','confirm'))]:
        out[f'block_teacher_{part}_{DATE}']=[job(f'teacher_{s}_{part}','teacher',f'{FEATURES}/{s}.pt',eval_part=part) for s in splits]
    return out

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]/'plans'
    for name,plan in plans().items():
        with (root/(name+'.json')).open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print(json.dumps(dict(plans=len(plans()),directory=str(root))))
