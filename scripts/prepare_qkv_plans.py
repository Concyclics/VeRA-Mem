"""Emit the predeclared QKV matrix at historical, explicit workspace paths."""
from pathlib import Path
import json

ROOT='/ssd3/chenhan/VeRA-Mem-Workspace'
DATE='20261006'
SEEDS=(81042,81043,81044)
ARMS=('static_flat','static_grouped','rebind_flat','rebind_grouped','rebind_grouped_additive')
MODEL=ROOT+'/models/Qwen3-4B-Instruct-2507'
FEATURES=ROOT+'/runs/qkv_prepare_'+DATE+'/features'


def job(name,stage,cache,**kwargs):
    arguments=['--stage',stage,'--model',MODEL,'--cache',cache]
    for k,v in kwargs.items():arguments += ['--'+k.replace('_','-'),str(v)]
    return dict(name=name,entry='qkv',arguments=arguments)


def plans():
    out={}
    for seed in SEEDS:
        name=f'qkv_train_{seed}_{DATE}';train=[];dev=[];confirm=[]
        for arm in ARMS:
            regime,mode=arm.split('_')[:2];readout='additive' if arm.endswith('_additive') else 'vera'
            child=f'{arm}_seed{seed}'
            train.append(job(child,'train',FEATURES+'/train.pt',seed=seed,regime=regime,
                read_mode=mode,readout=readout,updates=2048))
            checkpoint=f'{ROOT}/runs/{name}/{child}/last.pt'
            for split,part,target in [('known','development',dev),('dev','development',dev),
                                       ('known','confirmation',confirm),('confirm','confirmation',confirm)]:
                target.append(job(f'{child}_{split}_{part}','eval',f'{FEATURES}/{split}.pt',
                    checkpoint=checkpoint,seed=seed,eval_part=part))
        out[name]=train;out[f'qkv_development_{seed}_{DATE}']=dev;out[f'qkv_confirmation_{seed}_{DATE}']=confirm
    for part,splits in [('development',('known','dev')),('confirmation',('known','confirm'))]:
        out[f'qkv_teacher_{part}_{DATE}']=[job(f'teacher_{s}_{part}','teacher',f'{FEATURES}/{s}.pt',eval_part=part) for s in splits]
    name=f'qkv_telemetry_smoke_{DATE}'
    out[name]=[job('smoke_grouped_telemetry','train',FEATURES+'/train.pt',updates=2,regime='rebind',read_mode='grouped'),
        job('smoke_grouped_telemetry_eval','eval',FEATURES+'/known.pt',
            checkpoint=f'{ROOT}/runs/{name}/smoke_grouped_telemetry/last.pt',eval_part='smoke')]
    return out


if __name__=='__main__':
    destination=Path(__file__).resolve().parents[2]/'plans'
    for name,plan in plans().items():
        path=destination/(name+'.json')
        with path.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print(json.dumps(dict(plans=len(plans()),directory=str(destination))))
