"""All-row development report. Does not select a configuration or change any method."""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rap import planner as P

ROOT=Path(os.environ['B_REAL'] if 'B_REAL' in os.environ else os.environ['B'])
HERE=ROOT/'experiments/optimization_r5_20261024'
sys.path.insert(0,str(ROOT/'scripts/A_method'))
from k31_episodes import runs

def ratio(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b!=0)
def ci(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(ci_lo=float(np.percentile(a,2.5)) if len(a) else np.nan,
                ci_hi=float(np.percentile(a,97.5)) if len(a) else np.nan,valid_draws=len(a))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['residual','readout','smoke','dev','learnsmoke','learned','calsmoke','calibrated','lift','liftsmoke','iddev','idsmoke','persist','persistsmoke','low','lowsmoke','compact','compactsmoke'],required=True)
    ap.add_argument('--partial',action='store_true');ap.add_argument('--dataset-only',choices=['nusc','av2'])
    ap.add_argument('--out-name');args=ap.parse_args()
    out=HERE/'reports'/(args.out_name or args.stage);out.mkdir(parents=True,exist_ok=True)
    summaries=[];intervals=[];contrasts=[];episodes=[];diags=[];valid=[];frames=[]
    pp,cp=P.PlannerParams(),P.CostParams()
    cmd=np.array([P.commanded_decel(i,pp) for i in range(3)])
    for ds in ([args.dataset_only] if args.dataset_only else ('nusc','av2')):
        ref=ROOT/'results/k30/dev_devref'/ds
        expected=sorted(p.stem for p in ref.glob('*.pkl'))
        if args.stage in ('smoke','learnsmoke','calsmoke','liftsmoke','idsmoke','persistsmoke','lowsmoke','compactsmoke'):expected=sorted({c['unit'] for c in json.loads((HERE/'case_manifest_with_severity.json').read_text())['cases'] if c['dataset']==ds})
        folder=HERE/'results'/args.stage/ds
        records=[];seen=[]
        for file in sorted(folder.glob('*.pkl')):
            assert file.stem in expected
            audit=json.loads(file.with_suffix('.parity.json').read_text())
            assert not any(audit['differences'].values()),audit
            rows=pickle.load(file.open('rb'))
            for r in rows:
                records.append({k:v for k,v in r.items() if np.isscalar(v) or v is None})
            seen.append(file.stem)
            valid.append(dict(dataset=ds,unit=file.stem,frames=len(rows),baseline_differing=0))
        if not args.partial:assert seen==expected,(ds,len(seen),len(expected))
        if not records:continue
        d=pd.DataFrame(records);d=d[d['gt'].astype(bool)].sort_values(['unit','frame'])
        assert not d.duplicated(['unit','frame']).any()
        units=sorted(d.unit.unique());n=len(units)
        draws=np.random.default_rng(0).integers(0,n,(2000,n))
        C=np.array([np.bincount(x,minlength=n) for x in draws],float)
        labels={'v6S':'v6-S','v6S_b2A_median':'objectwise median','v6S_b2A_hardvote':'hard-brake corroboration','v6S_b2A_framevote':'framewise median','v6S_b2A_retro':'round1 combined','v6S_b2A_contain_retro':'containment merge','v6S_b2A_union_retro':'union coverage','v6S_b2A_semantic_retro':'semantic consistency','v6S_b2A_one_extent_retro':'one-round extent',
                'v6S_b2A_bridge':'mixed-history C1','v6S_b2A_bridge_retro':'mixed-history + recovery',
                'v6S_b2A_transfer_retro':'cheap TTC transfer','v6S_b2A_extent_retro':'critical extent retention',
                'v6S_b2A_motion_retro':'motion-conditioned field','v6S_b2A_missing_retro':'learned missing history',
                'v6S_b2A_allgeo_retro':'learned geometric','v6S_b2A_alldec_retro':'learned decision',
                'v6S_b2A_upper_retro':'upper-quantile missing history','v6S_b2A_query_retro':'uncertainty query',
                'v6S_b2A_liftgeo_retro':'joint geometric lift','v6S_b2A_liftdec_retro':'joint decision lift',
                'v6S_b2A_upper_extent_retro':'upper quantile + extent retention',
                'v6S_b2A_upper_identity_retro':'identity gate + uncertain fallback',
                'v6S_b2A_missing_extent_retro':'point fallback + extent retention',
                'v6S_b2A_missing_observed_retro':'observed-history point fallback',
                'v6S_b2A_missing_corroborated_retro':'corroborated upper fallback',
                'v6S_b2A_missing_compact_retro':'compact point fallback',
                'v6S_b2A_missing_compact_observed_retro':'compact observed-history fallback'}
        for mode in ['rollback','missing','both','median_gate']:
            for q in [50,80,95]:
                labels['v6S_r5_'+mode+'_'+str(q)]='R5 '+mode+' q'+str(q)
        labels['v6S_r5_matched_both_80']='R5 matched-only both q80'
        for mode in ['domain_both','residual','residual_both','extent_both','point_residual','extent_residual']:
            for q in [50,80,95]:
                labels['v6S_r5_'+mode+'_'+str(q)]='R5 '+mode+' q'+str(q)
        labels['v6S_r5_residual_ungated']='R5 residual ungated'
        for parent in ['round1','point']:
            for model in ['pooled','conditional']:
                for q in [50,80,95]:
                    labels['v6S_r5_cond_'+parent+'_'+model+'_'+str(q)]='R5 cond '+parent+' '+model+' q'+str(q)
        pipes={'cheap':'a_cheap'}
        for col in d:
            if col.startswith('a_v6S'):
                prefix,refiner=col[2:].split('|')
                pipes[labels[prefix]+('' if refiner=='v3a' or '_lift' in prefix else ' / '+refiner)]=col
        dist={};points={}
        for name,col in pipes.items():
            assert col in d and d[col].notna().all(),col
            prefix=col[2:].split('|')[0]
            unitrows=[]
            for unit,g in d.groupby('unit',sort=True):
                act=g[col].to_numpy(int);gt=g.act_gt.to_numpy(int);demand=g.a_gt.to_numpy(float)
                loss=cp.lam_risk*np.maximum(0,demand-cmd[act])**2
                req=runs(gt>=1);false=sum((gt[s:s+l]<1).all() for s,l in runs(act>=1))
                data=dict(unit=unit,frames=len(g),false=false,cheap_loss=g.J_cheap.sum(),
                          false_frames=int(((act>=1)&(gt<1)).sum()), hard_false_frames=int(((act>=2)&(gt<1)).sum()),
                          excess_cost=(cp.lam_brake*np.maximum(0,cmd[act]-demand)**2).sum(),
                          loss=loss.sum(),cheap_risk=(cp.lam_risk*np.maximum(0,demand-cmd[g.a_cheap.to_numpy(int)])**2).sum(),J=g['J'+col[1:]].sum(),
                          px=g['px_'+prefix].sum(),calls=g.get('ncalls_'+prefix,pd.Series(np.full(len(g),np.nan))).sum(min_count=1))
                for severity in ('all','below_hard','at_or_above_hard'):
                    data[severity+'_n']=data[severity+'_miss']=data[severity+'_delay']=data[severity+'_loss']=0
                for ei,(s,l) in enumerate(req):
                    hit=np.flatnonzero(act[s:s+l]>=1);miss=not len(hit);delay=int(hit[0]) if len(hit) else l
                    peak=float(demand[s:s+l].max());severity='at_or_above_hard' if peak>=3.5 else 'below_hard'
                    eloss=loss[s:s+l].sum()
                    for level in ('all',severity):
                        data[level+'_n']+=1;data[level+'_miss']+=int(miss)
                        data[level+'_delay']+=delay*float(g.dt.iloc[0]);data[level+'_loss']+=eloss
                    episodes.append(dict(dataset=ds,unit=unit,pipeline=name,episode=ei,start_frame=int(g.frame.iloc[s]),
                        end_frame=int(g.frame.iloc[s+l-1]),frames=l,severity=severity,peak_demand=peak,missed=miss,
                        delay_s=delay*float(g.dt.iloc[0]),insufficient_loss=float(eloss)))
                unitrows.append(data)
            u=pd.DataFrame(unitrows).set_index('unit').reindex(units)
            numer={'G':u.cheap_loss-u.J,'false_per_1000':1000*u.false,
                   'false_braking_frames_per_1000':1000*u.false_frames,'false_hard_frames_per_1000':1000*u.hard_false_frames,'overbraking_cost_per_frame':u.excess_cost,'insufficient_loss_per_frame':u.loss,
                   'insufficient_loss_share_cheap':u.loss,'mean_network_px':u.px,'mean_crop_detector_calls':u.calls}
            denom={'G':u.cheap_loss,'false_per_1000':u.frames,
                   'false_braking_frames_per_1000':u.frames,'false_hard_frames_per_1000':u.frames,'overbraking_cost_per_frame':u.frames,'insufficient_loss_per_frame':u.frames,
                   'insufficient_loss_share_cheap':u.cheap_risk,'mean_network_px':u.frames,'mean_crop_detector_calls':u.frames}
            for severity in ('all','below_hard','at_or_above_hard'):
                for metric,num,den in [('missed_rate','miss','n'),('onset_delay_s','delay','n')]:
                    key=severity+'_'+metric;numer[key]=u[severity+'_'+num];denom[key]=u[severity+'_'+den]
                key=severity+'_episode_loss_share_cheap';numer[key]=u[severity+'_loss'];denom[key]=u.cheap_risk
            values={};samples={}
            for metric,num in numer.items():
                num=np.asarray(num,float);den=np.asarray(denom[metric],float)
                value=float(ratio(num.sum(),den.sum()));sample=ratio(C@num,C@den)
                values[metric]=value;samples[metric]=sample
                intervals.append(dict(dataset=ds,pipeline=name,metric=metric,estimate=value,**ci(sample)))
            row=dict(dataset=ds,pipeline=name,units=n,frames=len(d),expected_units=len(expected),
                     episodes=int(u.all_n.sum()),missed=int(u.all_miss.sum()),hard_episodes=int(u.at_or_above_hard_n.sum()),**values)
            summaries.append(row);points[name]=values;dist[name]=samples
        for name in pipes:
            if name=='v6-S':continue
            for metric in points[name]:
                contrasts.append(dict(dataset=ds,pipeline=name,comparator='v6-S',metric=metric,
                    candidate_minus_baseline=points[name][metric]-points['v6-S'][metric],
                    **ci(dist[name][metric]-dist['v6-S'][metric])))
        for name in pipes:
            if name=='round1 combined':continue
            for metric in points[name]:
                contrasts.append(dict(dataset=ds,pipeline=name,comparator='round1 combined',metric=metric,
                    candidate_minus_baseline=points[name][metric]-points['round1 combined'][metric],
                    **ci(dist[name][metric]-dist['round1 combined'][metric])))
        for comparator in ['point fallback + extent retention','critical extent retention']:
            if comparator not in points:continue
            for name in pipes:
                if name==comparator:continue
                for metric in points[name]:
                    contrasts.append(dict(dataset=ds,pipeline=name,comparator=comparator,metric=metric,
                        candidate_minus_baseline=points[name][metric]-points[comparator][metric],
                        **ci(dist[name][metric]-dist[comparator][metric])))
        for name in labels:
            for col in d:
                if (col.startswith(('diag_','retro_','closure_')) and col.endswith('_'+name)):
                    diags.append(dict(dataset=ds,pipeline=name,counter=col[:-len(name)-1],
                                      sum=float(d[col].sum()),frames=len(d),units=n))
        fcols=['unit','frame','act_gt','a_gt']+list(pipes.values())+[c for c in d if c.startswith(('diag_','retro_','closure_','px_','ncalls_'))]
        ff=d[fcols].copy();ff.insert(0,'dataset',ds);frames.append(ff)
    for filename,rows in [('summary',summaries),('intervals',intervals),('contrasts',contrasts),('episodes',episodes),
                          ('diagnostics',diags),('baseline_parity',valid)]:
        pd.DataFrame(rows).to_csv(out/(filename+'.csv'),index=False)
    if frames:pd.concat(frames,ignore_index=True).to_csv(out/'frames.csv',index=False)
    lines=['# Round 5 sandbox development experiments — '+args.stage,'',
        'Development screen only; no method selected or promoted. Frozen sources and K34 are unchanged.',
        'All intervals: paired unit bootstrap, 2,000, seed 0; pointwise, exploratory. No GPU wall time is a Jetson latency measurement.',
        'G retains the original K30 cost convention (jerk relative to previous cheap action).',
        'Every executed policy/refiner row is retained. Logical requested pixels, not Jetson padded pixels.',
        'Generic-component/equal-total-budget controls remain necessary before a selector-superiority claim.','',
        '| Dataset | Pipeline | Units | Miss rate | False/1000 | Insufficient loss/frame | G | Mean px/native |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append('| '+str(r['dataset'])+' | '+r['pipeline']+' | '+str(r['units'])+' | '+
            ' | '.join(f"{r[k]:.4f}" for k in ['all_missed_rate','false_per_1000','insufficient_loss_per_frame','G','mean_network_px'])+' |')
    lines+=['','Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.',
            'Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.',
            'No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(stage=args.stage,summary_rows=len(summaries),units=len(valid),out=str(out))),flush=True)
if __name__=='__main__':main()
