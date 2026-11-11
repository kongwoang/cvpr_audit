"""Report all paired runtime measurements after the exclusive window is released."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from compare import compare

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'docs/paper/handoff/RUNTIME_20261023'


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
                    '\n\nInspect experiments/runtime_20261023/logs/. The parity gate or runtime failed; full results are not established.\n')
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
            result=compare(runs['reference'],runs['staged'])
            parity.append(dict(dataset=ds,rep=rep,frames=result['frames'],differing=0))
            for runtime,folder in runs.items():
                d=pd.read_pickle(folder/'frames.pkl')
                assert len(d)==712 and int(d.warm.sum())==200
                for source,pipeline in [('ours@trt32','v6-S'),('ours_b2@trt32','v6-S <=2'),('ours_b2A@trt32','B2K-A')]:
                    f=d.loc[~d.warm,['unit','frame','ms_'+source,'a_'+source,'px_'+source,'calls_'+source]].copy()
                    f.columns=['unit','frame','ms','action','network_pixels','detector_executions']
                    f['dataset'],f['rep'],f['runtime'],f['pipeline']=ds,rep,runtime,pipeline
                    frames.append(f)
    d=pd.concat(frames,ignore_index=True)
    d.to_csv(OUT/'frames.csv',index=False)
    pd.DataFrame(parity).to_csv(OUT/'parity.csv',index=False)
    d[['dataset','unit','frame']].drop_duplicates().to_csv(OUT/'frame_manifest.csv',index=False)
    names=['mean_ms','p95_ms','p99_ms','above50_pct','above66_7_pct','above100_pct']
    summaries=[];intervals=[];contrasts=[]
    for (ds,pipeline),g in d.groupby(['dataset','pipeline'],sort=True):
        units=sorted(g.unit.unique());n=len(units)
        pair={r:[g[(g.runtime==r)&(g.unit==u)].sort_values(['rep','frame']).ms.to_numpy() for u in units]
              for r in ('reference','staged')}
        for i in range(n):assert len(pair['reference'][i])==len(pair['staged'][i])
        points={r:metrics(np.concatenate(v)) for r,v in pair.items()}
        draws=np.random.default_rng(0).integers(0,n,(2000,n))
        boot={r:np.array([metrics(np.concatenate([v[j] for j in sample])) for sample in draws])
              for r,v in pair.items()}
        for runtime in pair:
            summaries.append(dict(dataset=ds,pipeline=pipeline,runtime=runtime,units=n,
                timed_frames_per_rep=512,repetitions=3,**dict(zip(names,points[runtime]))))
            for i,name in enumerate(names):
                low,high=np.percentile(boot[runtime][:,i],[2.5,97.5])
                intervals.append(dict(dataset=ds,pipeline=pipeline,runtime=runtime,metric=name,
                    estimate=points[runtime][i],ci_lo=low,ci_hi=high))
        for i,name in enumerate(names):
            low,high=np.percentile(boot['staged'][:,i]-boot['reference'][:,i],[2.5,97.5])
            contrasts.append(dict(dataset=ds,pipeline=pipeline,metric=name,
                staged_minus_reference=points['staged'][i]-points['reference'][i],ci_lo=low,ci_hi=high))
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'intervals.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv',index=False)
    lines=['# Jetson crop-buffer staging experiment','',
        'All rows retained. Existing trt32 engines; frozen method and main source unchanged.',
        'Dev prefix only: 512 timed frames/dataset/repetition after 200 warm-up frames, three repetitions.',
        'Full action/cost/crop/execution parity holds on all 712 chronological frames of every paired run.',
        'Intervals: paired unit bootstrap, 2,000, seed 0, repetitions nested within units; pooled quantiles.',
        'Shared detection caches are disabled during timing. Disk decode excluded; upload included.','',
        '| Dataset | Pipeline | Runtime | Mean ms | p95 ms | p99 ms | >50 ms % | >66.7 ms % | >100 ms % |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append('| '+r['dataset']+' | '+r['pipeline']+' | '+r['runtime']+' | '+
                     ' | '.join(f'{r[k]:.3f}' for k in names)+' |')
    lines+=['','Paired differences and intervals: contrasts.csv. These are development runtime results on a small',
        'chronological prefix, not a full-dataset speed or method-quality claim.',
        'Exact manifests: frame_manifest.csv. Hardware, engine hashes, source hashes, clocks and telemetry:',
        'experiments/runtime_20261023/logs/. All owned clocks/fan/pause controls were released.',
        'The implementation writes current float letterboxes directly into persistent engine input buffers;',
        'the candidate also uses explicit-stream enqueue. This experiment does not separate their individual effects.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'DONE.json').write_text(json.dumps(dict(status='complete',samples=len(d),source_engine_hashes_unchanged=True),indent=2)+'\n')


if __name__=='__main__':
    main()
