"""Report all paired runtime measurements after the exclusive window is released."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from compare import compare

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'docs/paper/handoff/FAST_RECOVERY_20261023'


def metrics(x):
    return np.array([np.mean(x),np.percentile(x,95),np.percentile(x,99),
        100*np.mean(x>50),100*np.mean(x>66.7),100*np.mean(x>100)])


def main():
    OUT.mkdir(exist_ok=True)
    while True:
        path=HERE/'logs/state.json'
        if path.exists():
            state=json.loads(path.read_text())
            (OUT/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
            if state.get('status')=='failed' and state.get('window_released'):
                (OUT/'REPORT.md').write_text('# Runtime experiment — stopped\n\n'+state.get('error','')+
                    '\n\nInspect experiments/temporal_fast_20261023/logs/. The parity gate or runtime failed; full results are not established.\n')
                return
            if state.get('status')=='timing_complete' and state.get('window_released'):
                break
        time.sleep(20)
    frames=[];parity=[]
    for ds in ('nusc_val','av2_dev'):
        for rep in (1,2,3):
            runs={r['runtime']:Path(r['run']) for r in state['runs']
                  if r['dataset']==ds and r['rep']==rep and not r['smoke']}
            assert len(runs)==2
            result=compare(runs['original'],runs['fast'])
            parity.append(dict(dataset=ds,rep=rep,frames=result['frames'],differing=0))
            for runtime,folder in runs.items():
                d=pd.read_pickle(folder/'frames.pkl')
                assert len(d)==456 and int(d.warm.sum())==200
                for source,pipeline in [('ours@trt32','v6-S'),('ours_retro@trt32','v6-S + recovery'),('ours_b2A@trt32','B2K-A'),('ours_b2A_retro@trt32','B2K-A + recovery')]:
                    f=d.loc[~d.warm,['unit','frame','ms_'+source,'a_'+source,'px_'+source,'calls_'+source]].copy()
                    f.columns=['unit','frame','ms','action','network_pixels','detector_executions']
                    f['dataset'],f['rep'],f['runtime'],f['pipeline']=ds,rep,runtime,pipeline
                    frames.append(f)
    d=pd.concat(frames,ignore_index=True)
    d.to_csv(OUT/'frames.csv',index=False)
    pd.DataFrame(parity).to_csv(OUT/'parity.csv',index=False)
    d[['dataset','unit','frame']].drop_duplicates().to_csv(OUT/'frame_manifest.csv',index=False)
    names=['mean_ms','p95_ms','p99_ms','above50_pct','above66_7_pct','above100_pct']
    summaries=[];intervals=[];contrasts=[];all_boot={};all_points={}
    for (ds,pipeline),g in d.groupby(['dataset','pipeline'],sort=True):
        units=sorted(g.unit.unique());n=len(units)
        pair={r:[g[(g.runtime==r)&(g.unit==u)].sort_values(['rep','frame']).ms.to_numpy() for u in units]
              for r in ('original','fast')}
        for i in range(n):assert len(pair['original'][i])==len(pair['fast'][i])
        points={r:metrics(np.concatenate(v)) for r,v in pair.items()}
        draws=np.random.default_rng(0).integers(0,n,(2000,n))
        boot={r:np.array([metrics(np.concatenate([v[j] for j in sample])) for sample in draws])
              for r,v in pair.items()}
        all_boot[(ds,pipeline)]=boot
        all_points[(ds,pipeline)]=points
        for runtime in pair:
            summaries.append(dict(dataset=ds,pipeline=pipeline,runtime=runtime,units=n,
                timed_frames_per_rep=256,repetitions=3,**dict(zip(names,points[runtime]))))
            for i,name in enumerate(names):
                low,high=np.percentile(boot[runtime][:,i],[2.5,97.5])
                intervals.append(dict(dataset=ds,pipeline=pipeline,runtime=runtime,metric=name,
                    estimate=points[runtime][i],ci_lo=low,ci_hi=high))
        for i,name in enumerate(names):
            low,high=np.percentile(boot['fast'][:,i]-boot['original'][:,i],[2.5,97.5])
            contrasts.append(dict(dataset=ds,pipeline=pipeline,metric=name,
                fast_minus_original=points['fast'][i]-points['original'][i],ci_lo=low,ci_hi=high))
    overhead=[]
    for ds in ('nusc_val','av2_dev'):
        for candidate,base in [('v6-S + recovery','v6-S'),('B2K-A + recovery','B2K-A')]:
            samples={mode:all_boot[(ds,candidate)][mode][:,0]-all_boot[(ds,base)][mode][:,0] for mode in ('original','fast')}
            values={mode:all_points[(ds,candidate)][mode][0]-all_points[(ds,base)][mode][0] for mode in ('original','fast')}
            lo,hi=np.percentile(samples['fast']-samples['original'],[2.5,97.5])
            overhead.append(dict(dataset=ds,pipeline=candidate,original_overhead_ms=values['original'],
                fast_overhead_ms=values['fast'],overhead_reduction_ms=values['fast']-values['original'],ci_lo=lo,ci_hi=hi))
    pd.DataFrame(overhead).to_csv(OUT/'overhead_contrasts.csv',index=False)
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'intervals.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv',index=False)
    lines=['# Jetson exact recovery bookkeeping pilot','',
        'All rows retained. Existing trt32 engines; frozen method and main source unchanged.',
        'Dev prefix only: 256 timed frames/dataset/repetition after 200 warm-up frames, three repetitions.',
        'Full action/cost/crop/execution parity holds on all 456 chronological frames of every paired run.',
        'Intervals: paired unit bootstrap, 2,000, seed 0, repetitions nested within units; pooled quantiles.',
        'Shared detection caches are disabled during timing. Disk decode excluded; upload included.','',
        '| Dataset | Pipeline | Runtime | Mean ms | p95 ms | p99 ms | >50 ms % | >66.7 ms % | >100 ms % |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append('| '+r['dataset']+' | '+r['pipeline']+' | '+r['runtime']+' | '+
                     ' | '.join(f'{r[k]:.3f}' for k in names)+' |')
    lines+=['','Paired differences and intervals: contrasts.csv. Recovery overhead adjusted for the base pipeline: overhead_contrasts.csv. These are development runtime results on a small',
        'chronological prefix, not a full-dataset speed or method-quality claim.',
        'Exact manifests: frame_manifest.csv. Hardware, engine hashes, source hashes, clocks and telemetry:',
        'experiments/temporal_fast_20261023/logs/. All owned clocks/fan/pause controls were released.',
        'This pilot preserves the original detector inputs, historical work, decisions and memory retention;',
        'the fast helper reuses existing chains and avoids matching/logging when no cheap object reaches the threshold.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'DONE.json').write_text(json.dumps(dict(status='complete',samples=len(d),source_engine_hashes_unchanged=True),indent=2)+'\n')


if __name__=='__main__':
    main()
