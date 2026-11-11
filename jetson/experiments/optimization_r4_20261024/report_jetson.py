"""Report R4 simple-readout feasibility with all measured tails and parity."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rap import planner as P

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'experiments/optimization_r4_jetson_20261024'
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R4_20261024/jetson'
LABELS={'ours':'v6-S','ours_b2A_retro':'round1 combined','ours_b2A_extent_retro':'critical extent retention','ours_b2A_median_retro':'objectwise median full','ours_b2A_medianskip_retro':'objectwise median skip'}
METRICS=['mean_ms','p95_ms','p99_ms','above50_pct','above66_7_pct','above100_pct',
         'mean_detector_executions','mean_network_px','mean_extra_encoded_crops','mean_head_items',
         'G_prefix','insufficient_loss_per_frame']

def weighted_quantiles(x,w):
    order=np.argsort(x);x=x[order];w=w[order];c=np.cumsum(w);n=c[-1]
    targets=np.array([.95,.99])*(n-1)
    lo=np.floor(targets);hi=np.ceil(targets)
    return x[np.searchsorted(c,lo,side='right')]*(hi-targets)+x[np.searchsorted(c,hi,side='right')]*(targets-lo) if (lo!=hi).all() else np.array([x[np.searchsorted(c,l,side='right')] if l==h else x[np.searchsorted(c,l,side='right')]*(h-t)+x[np.searchsorted(c,h,side='right')]*(t-l) for t,l,h in zip(targets,lo,hi)])

def metrics(g,w):
    x=g.ms.to_numpy();n=w.sum()
    q=weighted_quantiles(x,w)
    av=lambda name:np.dot(g[name],w)/n
    return np.array([np.dot(x,w)/n,*q,100*np.dot(x>50,w)/n,100*np.dot(x>66.7,w)/n,100*np.dot(x>100,w)/n,
                     av('executions'),av('pixels'),av('encoded'),av('head_items'),
                     1-np.dot(g.cost,w)/np.dot(g.cheap_cost,w),av('insufficient_loss')])

def main():
    global RUN,OUT,LABELS
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['direct','fast'],default='direct');a=ap.parse_args()
    if a.variant=='fast':
        RUN=ROOT/'experiments/optimization_r4_fast_jetson_20261024'
        OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R4_20261024/jetson_fast'
        LABELS={k:v for k,v in LABELS.items() if k not in ('ours','ours_b2A_extent_retro')}
    s=json.loads((RUN/'logs/state.json').read_text())
    assert s['status']=='complete' and s['window_released'] and not s['restore_errors'],s
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'window_audit.json').write_text(json.dumps(s,indent=2)+'\n')
    pp,cp=P.PlannerParams(),P.CostParams();commands=np.array([P.commanded_decel(i,pp) for i in range(3)])
    frames=[];quality=[];manifest=[];memory=[];cold=[]
    runs=[r for r in s['runs'] if not r['smoke']]
    assert len(runs)==6
    for r in runs:
        folder=Path(r['run']);d=pd.read_pickle(folder/'frames.pkl');meta=json.loads((folder/'meta.json').read_text())
        assert len(d)==712 and d.warm.sum()==200
        manifest.append(d[[c for c in ['unit','frame','image_filename','annotation_timestamp_ns','warm'] if c in d]].assign(dataset=r['dataset']))
        fullcol='a_ours_b2A_median_retro@trt32';skipcol='a_ours_b2A_medianskip_retro@trt32'
        different=int((d[fullcol]!=d[skipcol]).sum());assert not different,(r['dataset'],r['rep'],different)
        cold.append(dict(dataset=r['dataset'],rep=r['rep'],frames_including_warmup=len(d),median_abstention_action_differences=different))
        memory.append(dict(dataset=r['dataset'],rep=r['rep'],retained_history_bytes=int(d.gpu_history_bytes.max()),
            peak_gpu_allocated_bytes=meta['peak_gpu_allocated_bytes'],peak_gpu_reserved_bytes=meta['peak_gpu_reserved_bytes']))
        d=d[~d.warm].copy()
        for prefix,label in LABELS.items():
            k=prefix+'@trt32';f=d[['unit','frame','act_gt','a_gt','dt']].copy()
            for name,col in [('ms','ms_'),('action','a_'),('cost','J_'),('executions','calls_'),('pixels','px_')]:f[name]=d[col+k].to_numpy()
            for name,col in [('encoded','adapter_encoded_'),('head_items','adapter_items_'),('refined','n_refined_')]:f[name]=d[col+k].to_numpy() if col+k in d else 0
            f['cheap_cost']=d.J_cheap.to_numpy();f['insufficient_loss']=cp.lam_risk*np.maximum(0,f.a_gt.to_numpy()-commands[f.action.to_numpy(int)])**2
            f['dataset'],f['rep'],f['pipeline']=r['dataset'],r['rep'],label;frames.append(f)
            for suffix in ('','_noref'):
                action=d['a_'+k+suffix].to_numpy(int)
                quality.append(dict(dataset=r['dataset'],rep=r['rep'],pipeline=label+(' / none' if suffix else ''),frames=len(d),
                    G_prefix=1-d['J_'+k+suffix].sum()/d.J_cheap.sum(),
                    insufficient_loss_per_frame=np.mean(cp.lam_risk*np.maximum(0,d.a_gt.to_numpy()-commands[action])**2)))
    d=pd.concat(frames,ignore_index=True);assert not d.duplicated(['dataset','unit','frame','rep','pipeline']).any()
    d.groupby(['dataset','pipeline'],as_index=False).refined.mean().to_csv(OUT/'refinement_counts.csv',index=False)
    d.to_csv(OUT/'frames.csv',index=False);pd.concat(manifest).drop_duplicates().to_csv(OUT/'frame_manifest.csv',index=False)
    pd.DataFrame(quality).to_csv(OUT/'quality_by_rep.csv',index=False);pd.DataFrame(memory).to_csv(OUT/'memory.csv',index=False)
    pd.DataFrame(cold).to_csv(OUT/'abstention_parity.csv',index=False)
    groups=d.groupby(['dataset','unit','frame','pipeline']).action.nunique()
    parity=pd.DataFrame([dict(dataset=ds,groups=len(g),differing_action_groups=int((g>1).sum())) for ds,g in groups.groupby(level=0)])
    parity.to_csv(OUT/'repetition_parity.csv',index=False)
    summaries=[];intervals=[];contrasts=[];reps=[];conditional=[]
    for ds,dd in d.groupby('dataset'):
        units=sorted(dd.unit.unique());n=len(units);draws=np.random.default_rng(0).integers(0,n,(2000,n))
        counts=np.array([np.bincount(a,minlength=n) for a in draws]);boot={};points={}
        for label,g in dd.groupby('pipeline'):
            g=g.sort_values(['unit','rep','frame']);idx=pd.Categorical(g.unit,categories=units).codes
            points[label]=metrics(g,np.ones(len(g)));boot[label]=np.array([metrics(g,c[idx]) for c in counts])
            summaries.append(dict(dataset=ds,pipeline=label,units=n,timed_frames_per_rep=512,repetitions=3,**dict(zip(METRICS,points[label]))))
            for j,m in enumerate(METRICS):
                lo,hi=np.percentile(boot[label][:,j],[2.5,97.5]);intervals.append(dict(dataset=ds,pipeline=label,metric=m,estimate=points[label][j],ci_lo=lo,ci_hi=hi))
            for rep,h in g.groupby('rep'):reps.append(dict(dataset=ds,pipeline=label,rep=rep,**dict(zip(METRICS,metrics(h,np.ones(len(h)))))))
        comparisons=[(c,b) for c in LABELS.values() for b in ('v6-S','round1 combined') if c!=b and c!='v6-S' and b in LABELS.values()]
        comparisons.append(('objectwise median skip','objectwise median full'))
        for candidate,baseline in comparisons:
            for j,m in enumerate(METRICS):
                lo,hi=np.percentile(boot[candidate][:,j]-boot[baseline][:,j],[2.5,97.5])
                contrasts.append(dict(dataset=ds,pipeline=candidate,comparator=baseline,metric=m,candidate_minus_baseline=points[candidate][j]-points[baseline][j],ci_lo=lo,ci_hi=hi))
        cand=dd[dd.pipeline=='objectwise median skip'].set_index(['unit','frame','rep']).sort_index()
        base=dd[dd.pipeline=='round1 combined'].set_index(['unit','frame','rep']).sort_index();assert cand.index.equals(base.index)
        active=cand.refined<base.refined
        for label,g in [('candidate',cand),('baseline same frames',base)]:
            x=g.loc[active,'ms'];conditional.append(dict(dataset=ds,row=label,samples=len(x),unique_frames=len(x.reset_index()[['unit','frame']].drop_duplicates()),mean_ms=x.mean(),p95_ms=x.quantile(.95),p99_ms=x.quantile(.99)))
    for name,rows in [('summary',summaries),('intervals',intervals),('contrasts',contrasts),('by_repetition',reps),('triggered_latency',conditional)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    for ds in ('nusc_val','av2_dev'):(OUT/('smoke_parity_'+ds+'.json')).write_text((RUN/'logs'/('parity_'+ds+'.json')).read_text())
    lines=['# R4 simple readout — Jetson ('+a.variant+')','',
        'Three repetitions of 512 timed development frames per dataset after 200 warm-up; MAXN, locked clocks, fan255, exclusive window. Existing trt32 engines; no new model or engine. Rotated pipeline order; independent histories.',
        'Paired unit bootstrap 2,000 seed0, repetitions nested within scenes/logs; pooled percentiles. Fixed development prefixes are runtime feasibility evidence, not a replacement for full-population quality.',
        'Both median rows use the same round1 acquisition. Skipping preserves median ACTION, not the baseline v6-S action or continuous geometry. Exact full/skip action parity is checked on all warm-up and timed frames.',
        'Source/engine integrity, fresh baseline-port smoke parity and hardware restoration are audited.','',
        '| Dataset | Pipeline | Mean ms (95% CI) | p95 ms | p99 ms | >50 ms % | >66.7 ms % | >100 ms % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    ci=pd.DataFrame(intervals)
    for r in summaries:
        q=ci[(ci.dataset==r['dataset'])&(ci.pipeline==r['pipeline'])&(ci.metric=='mean_ms')].iloc[0]
        lines.append('| '+r['dataset']+' | '+r['pipeline']+' | '+f'{r["mean_ms"]:.2f} [{q.ci_lo:.2f}, {q.ci_hi:.2f}]'+' | '+' | '.join(f'{r[m]:.2f}' for m in METRICS[1:6])+' |')
    lines+=['','Every paired effect and interval is in contrasts.csv / intervals.csv. Pixels count padded detector work including historical recovery; mean refinement counts are in refinement_counts.csv.',
        'Timing excludes image decode and model startup. Quality on these fixed prefixes is diagnostic only. Cross-VUW decision differences are reported separately. No real-time guarantee or core promotion.',
        'Refinement item reduction is not proportional latency reduction: matching, CNN batches and detector work remain. Process peak memory includes all benchmark rows and is not a per-method deployment estimate.',
        'Reproduce: scripts/py experiments/optimization_r4_20261024/report_jetson.py --variant direct (or --variant fast). Exact commands, warm-up and hardware logs: window_audit.json.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,ds in zip(axs,['nusc_val','av2_dev']):
        q=pd.DataFrame(summaries);q=q[q.dataset==ds].set_index('pipeline').loc[list(LABELS.values())]
        for j,m in enumerate(['mean_ms','p95_ms','p99_ms']):ax.bar(np.arange(len(q))+(j-1)*.25,q[m],.25,label=m)
        ax.set_xticks(np.arange(len(q)));ax.set_xticklabels(q.index,rotation=25,ha='right',fontsize=7);ax.set_title(ds);ax.set_ylabel('Milliseconds');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'latency.pdf');fig.savefig(OUT/'latency.png',dpi=160);plt.close(fig)
    (OUT/'DONE.json').write_text(json.dumps(dict(status='complete',rows=len(summaries),bootstrap=2000,seed=0),indent=2)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__':main()
