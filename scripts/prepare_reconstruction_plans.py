"""Write explicit historical H100 plans; never launch or overwrite a run."""
import json
from pathlib import Path

ROOT='/ssd3/chenhan/VeRA-Mem-Workspace'
DATE='20261006'
SEEDS=(71042,71043,71044)
ARMS=(('S1_fixedB',1,False),('S3_fixedB',3,False),('S1_trainB',1,True),('S3_trainB',3,True))


def build():
    model=ROOT+'/models/Qwen3-4B-Instruct-2507'
    cache=ROOT+'/runs/reconstruction_prepare_'+DATE+'/features/'
    plans={};checkpoints={}
    def job(name,stage,extra):return dict(name=name,entry='reconstruction',arguments=['--stage',stage,'--model',model]+extra)
    for seed in SEEDS:
        base=f'reconstruction_baseline_{seed}_{DATE}';struct=f'reconstruction_structures_{seed}_{DATE}'
        plans[base]=[];plans[struct]=[]
        plans[f'reconstruction_baseline_eval_{seed}_{DATE}']=[];plans[f'reconstruction_structures_eval_{seed}_{DATE}']=[]
        plans[f'reconstruction_confirm_{seed}_{DATE}']=[]
        for arm,slots,learned in ARMS:
            name=f'{arm}_seed{seed}';suite=base if arm=='S1_fixedB' else struct
            plans[suite].append(job(name,'train',['--cache',cache+'train.pt','--seed',str(seed),'--slots',str(slots),'--updates','1024']+(['--learned-b'] if learned else [])))
            checkpoint=f'{ROOT}/runs/{suite}/{name}/last.pt';checkpoints[name]=checkpoint
            esuite=f'reconstruction_baseline_eval_{seed}_{DATE}' if arm=='S1_fixedB' else f'reconstruction_structures_eval_{seed}_{DATE}'
            for packet,part in (('known','development'),('dev','development'),('known','confirmation'),('confirm','confirmation')):
                destination=esuite if part=='development' else f'reconstruction_confirm_{seed}_{DATE}'
                plans[destination].append(job(name+'_'+packet+'_'+part,'eval',['--cache',cache+packet+'.pt','--checkpoint',checkpoint,'--eval-part',part]))
    for part,packets in (('development',('known','dev')),('confirmation',('known','confirm'))):
        plans[f'reconstruction_teacher_{part}_{DATE}']=[job('teacher_'+packet+'_'+part,'teacher',['--cache',cache+packet+'.pt','--eval-part',part]) for packet in packets]
    return plans


if __name__=='__main__':
    folder=Path(__file__).resolve().parents[2]/'plans';folder.mkdir(exist_ok=True)
    plans=build()
    for name,plan in plans.items():
        with (folder/(name+'.json')).open('x') as out:json.dump(plan,out,indent=2);out.write('\n')
    print(json.dumps(dict(plans=len(plans),jobs=sum(map(len,plans.values())),names=list(plans))))
