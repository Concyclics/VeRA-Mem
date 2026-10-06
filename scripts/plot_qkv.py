"""Export the final predeclared QKV endpoints; seeds shown individually, no CI."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=('static_flat','static_grouped','rebind_flat','rebind_grouped','rebind_grouped_additive')
LABELS=('Static\nflat','Static\ngrouped','Rebind\nflat','Rebind\ngrouped','Rebind\nadditive*')
SEEDS=(81042,81043,81044)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--summary',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    report=json.loads(args.summary.read_text())
    if not report['final_scope']['complete'] or report['partial']:raise ValueError('Require complete final summary')
    rows={(r['arm'],r['seed'],r['split'],r['part']):r for r in report['records'] if r['kind']=='evaluation'}
    fig,axes=plt.subplots(1,3,figsize=(12.2,4.6),sharey=True)
    endpoints=[('Known entities: A/B pair','known','development','CC_B','pair_em',12),
               ('Known entities: novel D + restore','known','confirmation','CC_D','update_restore_em',12),
               ('New entities: novel D','confirm','confirmation','CC_D','updated_em',8)]
    colors=('#476d95','#476d95','#d77931','#d77931','#797979')
    for ax,(title,split,part,group,metric,threshold) in zip(axes,endpoints):
        for i,arm in enumerate(ARMS):
            counts=[]
            for j,seed in enumerate(SEEDS):
                value=rows[arm,seed,split,part]['groups'][group];assert value['count']==16
                count=value[metric]*16
                if abs(count-round(count))>1e-8:raise ValueError('Noninteger sixteen-case result')
                counts.append(int(round(count)))
                ax.scatter(i+(j-1)*.16,count,s=42,color=colors[i],marker=('o','s','^')[j],edgecolor='white',linewidth=.5,zorder=3)
            ax.text(i,17.0,'/'.join(map(str,counts)),ha='center',va='center',fontsize=8,color=colors[i])
        ax.axhline(threshold,color='#999999',linestyle=':',linewidth=1)
        ax.set_title(title,fontsize=10,pad=17);ax.set_xticks(range(5),LABELS,fontsize=8)
        ax.set_ylim(-.7,18);ax.set_yticks([0,4,8,12,16]);ax.grid(axis='y',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Correct cases / 16')
    fig.suptitle('QKV binding and sparse readout: fixed-budget experiments',fontsize=13,y=.99)
    fig.text(.5,.025,'Markers/counts: seeds 81042 / 81043 / 81044 on the same cases; no independent-sample CI. Dotted lines = individual gates.\nLeft panel: development A/B; D panels: sealed confirmation. All panels use canonical support/query.\n*Additive readout is a diagnostic, excluded from VeRA selection.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.14,1,.96));args.output_dir.mkdir(parents=True,exist_ok=True)
    for extension in ('png','pdf','svg'):fig.savefig(args.output_dir/f'qkv_endpoints.{extension}',dpi=180,bbox_inches='tight')
    svg=args.output_dir/'qkv_endpoints.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
