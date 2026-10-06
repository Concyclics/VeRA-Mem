"""Read-only CPU training/operator diagnostics for the fixed block experiment.

Never opens evaluation predictions or confirmation results. The stored spectra
are means of per-gold-position SVDs of the ADDITIONAL outer operator. They are
not SVDs of an averaged matrix or of the complete diagonal-plus-outer reader.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
import torch
from vera_mem.block_vera import BlockVeRA
from vera_mem.block_data import episode,validate_training_data,PROTOCOL as DATA_PROTOCOL
from audit_reconstruction import Inputs,require,same,state_digest,file_hash,read_json,read_jsonl
from replay_coldstart_banks import analysis_exclusions

PROTOCOL='block-attention-memory-v1'
SEEDS=(91042,91043,91044)
MODES=('diagonal','pooled_outer','block_outer')
REGIMES=('static','rebind')
OUTER_NAMES={'P_in.weight','P_in.bias','P_out.weight'}
SCOPE='captured gold-prefix prediction positions including EOS; parameter state before update'


def mean(values):return statistics.fmean(values) if values else None


def spectrum_summary(singular,mode,rank):
    if mode=='diagonal':
        require(singular is None,'Diagonal control must not claim a measured outer spectrum')
        return dict(outer_present=False,outer_rank_upper_bound=0,diagonal_rank_upper_bound=rank,
            complete_operator_rank_upper_bound=rank,diagonal_and_complete_spectra_observed=False)
    require(isinstance(singular,list) and len(singular)==rank and all(math.isfinite(s) and s>=0 for s in singular),
            'Require a finite nonnegative full-width mean singular spectrum')
    require(all(a+1e-7>=b for a,b in zip(singular,singular[1:])),'Mean singular spectrum not ordered')
    bound=1 if mode=='pooled_outer' else 3
    first=singular[0];den=max(first,1e-30);total=sum(singular)
    ratios=[s/den if first>0 else None for s in singular]
    tail=sum(singular[bound:])/den if first>0 else 0.
    # Floating matrix multiplication/SVD can leave a numerical tail. This is
    # structural verification, not a useful-rank threshold fitted to outcomes.
    require(max(singular[bound:],default=0.)<=1e-4*first+1e-7,'Spectrum violates outer rank bound beyond numerical tolerance')
    entropy=math.exp(-sum((s/total)*math.log(s/total) for s in singular if s>0)) if total>0 else 0.
    return dict(outer_present=True,outer_rank_upper_bound=bound,diagonal_rank_upper_bound=rank,
        complete_operator_rank_upper_bound=rank,mean_ordered_singular_values=singular,
        mean_sigma2_over_mean_sigma1=ratios[1] if rank>1 else None,
        mean_sigma3_over_mean_sigma1=ratios[2] if rank>2 else None,
        mean_spectrum_entropy_effective_rank=entropy,
        mean_spectrum_tail_mass_over_sigma1=tail,
        mean_sigma2_below_one_thousandth_sigma1=(first>0 and rank>1 and ratios[1]<1e-3),
        mean_sigma3_below_one_thousandth_sigma1=(first>0 and rank>2 and ratios[2]<1e-3),
        diagonal_and_complete_spectra_observed=False)


def binding_exposures(episodes):
    """Count the actually supervised entity/payload pair in BOTH A/B branches."""
    a=Counter();b=Counter();entities=Counter();payloads=Counter();background=Counter()
    for ep in episodes:
        entities.update(ep['targets']);background.update(ep['background'])
        for entity,local in zip(ep['targets'],ep['local_targets']):
            require(ep['bank_entities'][local]==entity,'Target/local bank mapping differs')
            ai=ep['a_payload_indices'][local];bi=ep['b_payload_indices'][local]
            a[entity,ai]+=1;b[entity,bi]+=1;payloads.update((ai,bi))
    known=[]
    for entity in range(16):
        for world,payload in [('A',2*entity),('B',2*entity+1)]:
            ac,bc=a[entity,payload],b[entity,payload]
            known.append(dict(entity_index=entity,reference_world=world,payload_index=payload,
                as_A_target_count=ac,as_B_target_count=bc,as_any_target_count=ac+bc))
    combined=a+b
    return dict(known_reference_bindings=known,known_reference_binding_count=32,
        known_zero_supervised_bindings=sum(r['as_any_target_count']==0 for r in known),
        known_zero_by_reference_world={w:sum(r['as_any_target_count']==0 for r in known if r['reference_world']==w) for w in ('A','B')},
        all_entity_payload_pairs=64*128,all_zero_supervised_bindings=64*128-len(combined),
        entity_target_counts=[entities[i] for i in range(64)],payload_supervised_counts=[payloads[i] for i in range(128)],
        entity_background_episode_counts=[background[i] for i in range(64)],
        counts_scope='Targets only count each supervised A/B sequence once; background counts are episode membership, not per-student-bank residence or supervision.',
        known_scope='First 16 train entities and canonical payload indices 2e/2e+1. Either training world counts as exposure because the question has no A/B world label.')


def log_diagnostics(raw,status,cfg,parameter_names,rank=64,require_full=True):
    steps=cfg['updates'];require(len(raw)==steps==status['updates'] and status['complete'],'Incomplete update budget')
    require(not require_full or steps==2048,'Formal diagnostics require fixed final 2048 updates')
    schedule=hashlib.sha256();common_schedule=hashlib.sha256();episodes=[];spectra=[]
    gradients={n:[] for n in parameter_names};group_norms=[]
    totals=dict(target_exposures=0,gold_tokens=0,input_positions=0,padded_input_positions=0,backbone_calls=0)
    for step,row in enumerate(raw):
        ep=episode(step,cfg['regime'],cfg['seed'])
        require(row['step']==step+1 and row['episode']==ep,'Actual episode differs from deterministic assignment')
        episodes.append(ep);schedule.update(json.dumps(ep,sort_keys=True).encode())
        common={k:ep[k] for k in ('step','seed','epoch','batch_index','targets','background','bank_entities','local_targets')}
        common_schedule.update(json.dumps(common,sort_keys=True).encode())
        m=row['metrics'];require(all(math.isfinite(m[k]) and m[k]>=0 for k in ('ce','address','loss')),'Invalid training loss')
        require(math.isclose(m['loss'],m['ce']+.2*m['address'],rel_tol=1e-6,abs_tol=1e-7),'Training loss arithmetic differs')
        norms=row['parameter_gradient_norms'];require(set(norms)==set(parameter_names),'Parameter gradient inventory differs')
        for name in parameter_names:
            v=norms[name];require(v is not None and math.isfinite(v) and v>=0,'Invalid/absent parameter gradient norm')
            gradients[name].append(v)
        groups=row['gradient_norms'];require(len(groups)==(4 if cfg['block_mode']=='diagonal' else 5)
            and all(math.isfinite(v) and v>=0 for v in groups),'Invalid group-gradient telemetry')
        group_norms.append(groups)
        lengths,gold=row['input_lengths'],row['gold_token_lengths']
        require(len(lengths)==len(gold)==16 and all(type(i)is int and type(g)is int and 0<g<i for i,g in zip(lengths,gold)),
                'Invalid full sequence/token ledger')
        totals['target_exposures']+=8;totals['gold_tokens']+=sum(gold);totals['input_positions']+=sum(lengths)
        totals['padded_input_positions']+=16*max(lengths);totals['backbone_calls']+=1
        d=row['read_diagnostics'];require((d is not None)==(step==0 or (step+1)%128==0),'Unexpected diagnostic sampling schedule')
        if d is not None:
            require(d['scope']==SCOPE and d['prediction_positions']==sum(gold) and d['actual_read_slots']==3,'Operator diagnostic scope differs')
            require(all(math.isfinite(d[k]) and d[k]>=0 for k in ('input_rms','mixed_value_rms','residual_rms','target_fact_mass'))
                    and d['target_fact_mass']<=1.00001,'Invalid diagnostic magnitudes')
            if step==0:require(d['residual_rms']==0,'Initial zero-b residual is not zero')
            spectra.append(dict(step=step+1,prediction_positions=d['prediction_positions'],input_rms=d['input_rms'],
                mixed_value_rms=d['mixed_value_rms'],complete_residual_rms=d['residual_rms'],target_fact_mass=d['target_fact_mass'],
                **spectrum_summary(d['outer_operator_mean_singular_values'],cfg['block_mode'],rank)))
    require(schedule.hexdigest()==status['schedule_sha256'] and totals==status['totals'],'Schedule or token totals differ')
    exposure=binding_exposures(episodes)
    if require_full:
        require(set(exposure['entity_target_counts'])==set(exposure['payload_supervised_counts'])=={256},'Marginal entity/payload budget differs')
    g_summary={n:dict(nonzero_steps=sum(v>0 for v in values),zero_steps=sum(v==0 for v in values),
        first_nonzero_step=next((i+1 for i,v in enumerate(values) if v>0),None),max_norm=max(values),
        last_128_mean_norm=mean(values[-128:])) for n,values in gradients.items()}
    require(not require_full or all(v['nonzero_steps'] for v in g_summary.values()),'A parameter never had a nonzero logged gradient')
    last=raw[-min(128,len(raw)):]
    return dict(schedule_sha256=schedule.hexdigest(),target_bank_schedule_sha256=common_schedule.hexdigest(),
        token_totals=totals,parameter_gradients=g_summary,
        optimizer_group_gradients=[dict(group=i,nonzero_steps=sum(r[i]>0 for r in group_norms),
            max_norm=max(r[i] for r in group_norms),last_128_mean_norm=mean([r[i] for r in group_norms[-128:]])) for i in range(len(group_norms[0]))],
        last_128_steps=dict(first_step=last[0]['step'],last_step=last[-1]['step'],steps=len(last),
            **{k+'_mean':mean([r['metrics'][k] for r in last]) for k in ('ce','address','loss')}),
        sampled_operator_diagnostics=spectra,final_preupdate_operator_diagnostics=spectra[-1],binding_exposures=exposure)


def source_binding(directory,manifest,inputs):
    source=manifest['source_sha256'];snapshot=directory.parent/'source/src/vera_mem'
    require(snapshot.is_dir(),'Frozen training source snapshot absent')
    for name,h in source.items():
        require(Path(name).name==name and inputs.sha(snapshot/name)==h,'Training source SHA differs: '+name)
    for name in ('block_vera.py','block_data.py','qkv_data.py','reconstruction_data.py'):
        require(source[name]==file_hash(REPO/'src/vera_mem'/name),'Current CPU diagnostic dependency differs: '+name)
    return dict(source_files=len(source),snapshot=str(snapshot))


def constructor_parameters_match(model,state):
    """Only the seeded bias RMS reduction may differ by cross-host FP32 ULPs."""
    parameters=dict(model.named_parameters())
    require(all(same(state[n],p) for n,p in parameters.items() if n!='P_in.bias')
            and same(state['A'],model.A),'Initial trainable tensors/A differ from declared seeded constructor')
    error=None
    if 'P_in.bias' in parameters:
        require(torch.allclose(state['P_in.bias'],parameters['P_in.bias'],rtol=1e-6,atol=1e-6),
                'Initial affine bias differs from seeded unit-RMS constructor')
        error=float((state['P_in.bias']-parameters['P_in.bias']).abs().max())
    return error


def diagnose_run(directory,manifest,inputs):
    require(manifest['protocol']==PROTOCOL and manifest['stage']=='train' and manifest['complete'] is True
            and manifest['backbone_unchanged'] is True,'Require completed frozen-backbone block training')
    cfg=manifest['configuration'];require(cfg['seed'] in SEEDS and cfg['regime'] in REGIMES and cfg['block_mode'] in MODES,'Unregistered training condition')
    source=source_binding(directory,manifest,inputs)
    names=('manifest.json','training.jsonl','training_status.json','initial.pt','last.pt')
    before={n:file_hash(directory/n) for n in names}
    initial=inputs.load(str(directory/'initial.pt'));last=inputs.load(str(directory/'last.pt'))
    require(initial['architecture']==last['architecture'],'Training architecture changed')
    arch=initial['architecture'];require(arch['rank']==arch['key_dim']==64 and arch['slots']==3
        and arch['block_mode']==cfg['block_mode'] and arch['seed']==cfg['seed'] and arch['residual_outer'] is True,
        'Unexpected operator configuration')
    model=BlockVeRA(**arch);parameters=dict(model.named_parameters());reference=model.state_dict()
    bias_error=constructor_parameters_match(model,initial['module'])
    model.load_state_dict(initial['module']);model.load_state_dict(last['module'])
    for ckpt,step in ((initial,0),(last,cfg['updates'])):
        require(ckpt['protocol']==PROTOCOL and ckpt['step']==step and ckpt['seed']==cfg['seed']
            and ckpt['regime']==cfg['regime'] and ckpt['block_mode']==cfg['block_mode']
            and ckpt['cache_sha256']==manifest['cache_sha256'],'Checkpoint lineage differs')
        require(all(bool(torch.isfinite(t).all()) for t in ckpt['module'].values() if t.is_floating_point()),'Nonfinite checkpoint tensor')
    fixed=set(reference)-set(parameters)
    require(all(same(initial['module'][n],last['module'][n]) for n in fixed),'Frozen buffers/centers changed')
    packet=inputs.load(cfg['cache'],manifest['cache_sha256'])
    require(packet['protocol']==PROTOCOL and packet['split']=='train','Training cache scope differs')
    validate_training_data(dict(protocol=DATA_PROTOCOL,seed=packet['data_seed'],entities=packet['entities'],payloads=packet['payloads'],rows=packet['rows']))
    status=read_json(directory/'training_status.json');require(status==manifest['result'],'Training status differs from manifest')
    count=sum(p.numel() for p in parameters.values());expected=2034368+(8256 if cfg['block_mode']!='diagonal' else 0)
    require(count==status['trainable_parameters']==expected,'Trainable parameter budget differs')
    raw=read_jsonl(directory/'training.jsonl');diagnostics=log_diagnostics(raw,status,cfg,parameters)
    require(last['schedule_sha256']==diagnostics['schedule_sha256'],'Final checkpoint schedule differs')
    require(before=={n:file_hash(directory/n) for n in names},'Inputs changed while diagnosing')
    return dict(run_dir=str(directory),seed=cfg['seed'],regime=cfg['regime'],block_mode=cfg['block_mode'],updates=cfg['updates'],
        trainable_parameters=count,outer_extra_parameters=expected-2034368,source=source,
        parameter_numel={n:p.numel() for n,p in parameters.items()},
        initial_constructor_matches=True,seeded_bias_reconstruction_max_abs_error=bias_error,
        frozen_A_and_statistics_endpoint_equal=True,
        initial_module_sha256=state_digest(initial['module']),last_module_sha256=state_digest(last['module']),
        artifact_sha256=before,cache_sha256=manifest['cache_sha256'],**diagnostics)


def matched_initializations(jobs,inputs):
    grouped=defaultdict(dict);seen=set()
    for j in jobs:
        key=(j['seed'],j['regime'],j['block_mode']);require(key not in seen,'Duplicate training condition');seen.add(key)
        grouped[j['seed']][key[1:]]=j
    expected={(seed,regime,mode) for seed in SEEDS for regime in REGIMES for mode in MODES};reports=[]
    for seed,conditions in sorted(grouped.items()):
        states={key:inputs.load(str(Path(j['run_dir'])/'initial.pt')) for key,j in conditions.items()}
        reference=next(iter(states.values()));common=set(reference['module'])-OUTER_NAMES-{'block_architecture'}
        outer=[];regime_schedules={};base_job=next(iter(conditions.values()))
        for key,state in states.items():
            j=conditions[key]
            require(set(state['module'])-OUTER_NAMES-{'block_architecture'}==common
                and all(same(state['module'][n],reference['module'][n]) for n in common),'Same-seed common initial tensors differ')
            require({k:v for k,v in state['architecture'].items() if k!='block_mode'}
                =={k:v for k,v in reference['architecture'].items() if k!='block_mode'},'Common architecture differs')
            if key[1]!='diagonal':outer.append({n:state['module'][n] for n in OUTER_NAMES})
            if key[0] in regime_schedules:require(regime_schedules[key[0]]==j['schedule_sha256'],'Within-regime schedules differ')
            regime_schedules[key[0]]=j['schedule_sha256']
            require(j['target_bank_schedule_sha256']==base_job['target_bank_schedule_sha256'],'Target/background schedules differ across readers/regimes')
            for name in ('target_exposures','gold_tokens','input_positions','backbone_calls'):
                require(j['token_totals'][name]==base_job['token_totals'][name],'Marginal training budget differs')
        require(all(all(same(state[n],outer[0][n]) for n in OUTER_NAMES) for state in outer),'Same-seed outer factors/bias initial tensors differ')
        reports.append(dict(seed=seed,observed_conditions=len(conditions),common_initial_tensors_bitwise_equal=True,
            common_initial_sha256=state_digest({n:reference['module'][n] for n in common}),
            outer_conditions=len(outer),all_outer_initials_bitwise_equal=True,
            outer_initial_sha256=state_digest(outer[0]) if outer else None,
            common_target_bank_schedule_sha256=base_job['target_bank_schedule_sha256']))
    return dict(complete=seen==expected,observed=len(seen),expected=18,missing=[list(k) for k in sorted(expected-seen)],seeds=reports)


def diagnose(root):
    inputs=Inputs(root);jobs=[];pending=[];excluded=[]
    # Deliberately do not discover eval/teacher/confirmation directories.
    for suite_path in sorted(inputs.root.glob('block_train_*/suite.json')):
        directory=suite_path.parent;markers=analysis_exclusions(directory,inputs.root)
        if markers:excluded.append(dict(path=str(directory),markers=markers));continue
        suite=read_json(suite_path);inputs.sha(suite_path);require(suite['protocol']==PROTOCOL,'Wrong training suite protocol')
        plan=read_json(directory/'plan.json');inputs.sha(directory/'plan.json');launched={j['name']:j for j in suite['jobs']}
        for spec in plan:
            argv=spec['arguments'];require(argv[argv.index('--stage')+1]=='train','Train suite contains nontraining job')
            child=directory/spec['name'];markers=analysis_exclusions(child,inputs.root)
            if markers:excluded.append(dict(path=str(child),markers=markers));continue
            job=launched.get(spec['name'])
            if not job or job.get('status')!='complete' or job.get('exit_code')!=0:
                pending.append(dict(path=str(child),reason='Not complete; training artifacts not opened'));continue
            require(inputs.sha(child/'manifest.json')==job['manifest_sha256'],'Launcher/child manifest SHA differs')
            manifest=read_json(child/'manifest.json')
            if not manifest.get('complete'):
                pending.append(dict(path=str(child),reason='Child manifest incomplete'));continue
            for flag,value in zip(argv[::2],argv[1::2]):
                require(str(manifest['configuration'][flag[2:].replace('-','_')])==value,'Plan/child configuration differs')
            result=diagnose_run(child,manifest,inputs);jobs.append(result)
            print(json.dumps(dict(diagnosed=str(child))),flush=True)
    matrix=matched_initializations(jobs,inputs)
    bindings={}
    for j in jobs:
        key=str(j['seed'])+'_'+j['regime']
        if key in bindings:require(bindings[key]==j['binding_exposures'],'Reader arms saw different binding exposures')
        bindings[key]=j['binding_exposures']
    return dict(protocol='block-operator-training-diagnostics-v1',complete=matrix['complete'] and not pending,
        checks_passed=True,audited_at=datetime.now(timezone.utc).isoformat(),jobs=jobs,matched_initialization=matrix,
        canonical_known_AB_binding_exposures=bindings,pending=pending,excluded=excluded,
        input_sha256={str(p):h for p,h in inputs.hashes.items()},cuda_initialized=torch.cuda.is_initialized(),
        interpretation=[
            'Spectra are arithmetic means of ordered SVD values computed separately at sampled actual gold-prefix positions, including EOS, before that update.',
            'Mean sigma2/mean sigma1 is a ratio of means, not the mean of per-position ratios. Spectrum entropy is computed on the mean ordered spectrum, not a mean per-position effective rank.',
            'The 1e-3 small-component flags are descriptive numerical thresholds, not a fitted gate, evidence of causality or proof that every query has collapsed.',
            'Only the extra outer operator was logged. The diagonal and complete diagonal-plus-outer spectra cannot be reconstructed from these logs; their rank upper bound remains 64.',
            'Nonzero parameter/group gradient norms do not establish coordinate coverage or useful optimization. With uniform slots and discrete group selection, Q/K receive the dense address loss, not a differentiable CE selection path.',
            'The final diagnostic is sampled before update 2048; last.pt is after update 2048. These are deliberately not called the same parameter state.',
            'Last-128 CE includes EOS, averages gold prediction tokens within each sequence, and then averages sequences/steps; it is training convergence evidence, not free-generation EM.',
            'Canonical known A/B target bindings can have zero or few supervised exposures under rebind despite matched global entity and payload marginals.',
            'Three seeds/readers share one dataset. Repeated facts, spectra and targets are not independent semantic-generalization samples.',
            'This tool does not run the language model, inspect evaluation results, replay Adam updates or recompute missing complete operators.'])


def self_test():
    import copy
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def test_spectrum_distinguishes_rank_bound_and_mean_ratio(self):
            r=spectrum_summary([9.,3.,1.]+[1e-8]*61,'block_outer',64)
            self.assertEqual(r['outer_rank_upper_bound'],3);self.assertEqual(r['complete_operator_rank_upper_bound'],64)
            self.assertAlmostEqual(r['mean_sigma2_over_mean_sigma1'],1/3)
            self.assertGreater(r['mean_spectrum_entropy_effective_rank'],1)
            self.assertFalse(r['diagonal_and_complete_spectra_observed'])
            self.assertFalse(spectrum_summary(None,'diagonal',64)['outer_present'])
        def test_invalid_spectra_fail_instead_of_claiming_rank(self):
            for values,mode in [([1.,.1]+[0.]*62,'pooled_outer'),([1.,.5,.4,.3]+[0.]*60,'block_outer'),
                                 ([1.,float('nan')]+[0.]*62,'block_outer'),([1.]*64,'diagonal')]:
                with self.assertRaises(ValueError):spectrum_summary(values,mode,64)
        def test_static_target_binding_exposure_is_not_payload_marginal(self):
            r=binding_exposures([episode(i,'static',91042) for i in range(2048)])
            self.assertEqual(r['known_zero_supervised_bindings'],0)
            self.assertEqual({e['as_any_target_count'] for e in r['known_reference_bindings']},{256})
            self.assertEqual(r['all_zero_supervised_bindings'],64*128-128)
            self.assertEqual(set(r['entity_target_counts']),{256});self.assertEqual(set(r['payload_supervised_counts']),{256})
        def test_rebind_counts_sum_A_and_B_worlds_for_identical_question(self):
            episodes=[episode(i,'rebind',91042) for i in range(2048)];r=binding_exposures(episodes)
            independent=Counter()
            for ep in episodes:
                for entity in ep['targets']:
                    local=ep['bank_entities'].index(entity)
                    independent[entity,ep['a_payload_indices'][local]]+=1
                    independent[entity,ep['b_payload_indices'][local]]+=1
            for row in r['known_reference_bindings']:
                self.assertEqual(row['as_any_target_count'],independent[row['entity_index'],row['payload_index']])
            self.assertEqual(set(r['payload_supervised_counts']),{256})
            self.assertGreater(len({e['as_any_target_count'] for e in r['known_reference_bindings']}),1)
        def log(self):
            cfg=dict(updates=8,seed=91042,regime='rebind',block_mode='block_outer');raw=[];h=hashlib.sha256()
            for step in range(8):
                ep=episode(step,'rebind',91042);h.update(json.dumps(ep,sort_keys=True).encode())
                raw.append(dict(step=step+1,episode=ep,metrics=dict(ce=1.,address=2.,loss=1.4),
                    parameter_gradient_norms={'b':float(step>0),'Wq.weight':1.},gradient_norms=[1.]*5,
                    input_lengths=[10]*16,gold_token_lengths=[4]*16,read_diagnostics=None if step else dict(
                        scope=SCOPE,prediction_positions=64,actual_read_slots=3,input_rms=1.,mixed_value_rms=1.,
                        residual_rms=0.,target_fact_mass=.5,outer_operator_mean_singular_values=[1.,.2,.1]+[0.]*61)))
            status=dict(complete=True,updates=8,schedule_sha256=h.hexdigest(),totals=dict(target_exposures=64,
                gold_tokens=512,input_positions=1280,padded_input_positions=1280,backbone_calls=8))
            return raw,status,cfg
        def test_log_reconstructs_budget_spectrum_and_last_window(self):
            r=log_diagnostics(*self.log(),['b','Wq.weight'],require_full=False)
            self.assertEqual(r['parameter_gradients']['b']['nonzero_steps'],7)
            self.assertEqual(r['last_128_steps']['ce_mean'],1.)
            self.assertEqual(r['last_128_steps']['steps'],8)
        def test_episode_tokens_and_diagnostic_scope_tampering_rejected(self):
            for key in ('episode','gold','spectrum','scope'):
                raw,status,cfg=self.log()
                if key=='episode':raw[0]['episode']['local_targets'][0]=17
                elif key=='gold':raw[0]['gold_token_lengths'][0]+=1
                elif key=='spectrum':raw[0]['read_diagnostics']['outer_operator_mean_singular_values'][4]=.7
                else:raw[0]['read_diagnostics']['scope']='postupdate'
                with self.assertRaises(ValueError):log_diagnostics(raw,status,cfg,['b','Wq.weight'],require_full=False)
        def test_no_completed_input_is_not_successful_complete(self):
            with tempfile.TemporaryDirectory() as tmp:
                r=diagnose(Path(tmp));self.assertFalse(r['complete']);self.assertEqual(r['matched_initialization']['observed'],0)
        def test_matched_initials_include_outer_bias_and_statistic_buffers(self):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);jobs=[]
                for seed in SEEDS:
                    for regime in REGIMES:
                        for mode in MODES:
                            directory=root/f'{seed}_{regime}_{mode}';directory.mkdir()
                            m=BlockVeRA(5,6,rank=3,key_dim=4,seed=seed,block_mode=mode)
                            torch.save(dict(module=m.state_dict(),architecture=m.configuration()),directory/'initial.pt')
                            jobs.append(dict(seed=seed,regime=regime,block_mode=mode,run_dir=str(directory),
                                schedule_sha256=str(seed)+regime,target_bank_schedule_sha256=str(seed),
                                token_totals=dict(target_exposures=16384,gold_tokens=170000,input_positions=2500000,backbone_calls=2048)))
                result=matched_initializations(jobs,Inputs(root));self.assertTrue(result['complete'])
                path=Path(jobs[-1]['run_dir'])/'initial.pt';original=torch.load(path,weights_only=True)
                bad=copy.deepcopy(original);bad['module']['P_in.bias'][0]+=.1;torch.save(bad,path)
                with self.assertRaisesRegex(ValueError,'outer factors'):matched_initializations(jobs,Inputs(root))
                bad=copy.deepcopy(original);bad['module']['query_center'][0]+=.1;torch.save(bad,path)
                with self.assertRaisesRegex(ValueError,'common initial'):matched_initializations(jobs,Inputs(root))
        def test_constructor_allows_only_bias_reduction_ulp_difference(self):
            m=BlockVeRA(5,6,rank=3,key_dim=4,seed=91042,block_mode='block_outer')
            state={n:p.clone() for n,p in m.state_dict().items()};state['P_in.bias'][0]+=2.4e-7
            self.assertLess(constructor_parameters_match(m,state),3e-7)
            state['P_in.weight'][0,0]+=2.4e-7
            with self.assertRaisesRegex(ValueError,'trainable tensors'):constructor_parameters_match(m,state)
            state={n:p.clone() for n,p in m.state_dict().items()};state['P_in.bias'][0]+=.01
            with self.assertRaisesRegex(ValueError,'affine bias'):constructor_parameters_match(m,state)
        def test_pending_launcher_prevents_opening_invalid_child_artifacts(self):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);suite=root/'block_train_fixture';suite.mkdir();child=suite/'not_done';child.mkdir()
                (suite/'suite.json').write_text(json.dumps(dict(protocol=PROTOCOL,jobs=[])))
                (suite/'plan.json').write_text(json.dumps([dict(name='not_done',arguments=['--stage','train'])]))
                (child/'manifest.json').write_text('invalid while being written')
                (child/'training.jsonl').write_text('must not be read')
                result=diagnose(root);self.assertFalse(result['complete']);self.assertEqual(len(result['pending']),1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs-root',type=Path)
    p.add_argument('--output',type=Path,default=REPO/'docs/results/block/operator_diagnostics.json')
    p.add_argument('--self-test',action='store_true');args=p.parse_args(argv)
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    if args.self_test:return self_test()
    if args.runs_root is None:p.error('--runs-root is required')
    require(not args.output.exists(),'Refusing to overwrite operator diagnostic output')
    start=time.perf_counter();report=diagnose(args.runs_root)
    report.update(elapsed_seconds=time.perf_counter()-start,script_sha256=file_hash(Path(__file__)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(complete=report['complete'],models=len(report['jobs']),output=str(args.output))))
    return 0


if __name__=='__main__':raise SystemExit(main())
