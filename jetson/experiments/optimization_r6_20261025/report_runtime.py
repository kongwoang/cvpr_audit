"""Scene-bootstrap component timings; no end-to-end interpretation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
def metrics(g,w):
    x=g.ms.to_numpy();order=np.argsort(x);xx=x[order];ww=w[order];cum=np.cumsum(ww);nt=cum[-1]
    qq=[]
    for q in (.95,.99):
        t=q*(nt-1);lo=int(np.floor(t));hi=int(np.ceil(t));frac=t-lo
        qq.append((1-frac)*xx[np.searchsorted(cum,lo,side='right')]+frac*xx[np.searchsorted(cum,hi,side='right')])
    return np.array([np.dot(x,w)/w.sum(),*qq])
def main():
    out=HERE/'reports/jetson';d=pd.read_csv(out/'frames.csv');names=['mean_ms','p95_ms','p99_ms'];summary=[];cis=[];diff=[];reps=[];active=[]
    for ds,dd in d.groupby('dataset'):
        units=sorted(dd.unit.unique());n=len(units);C=np.array([np.bincount(i,minlength=n) for i in np.random.default_rng(0).integers(0,n,(2000,n))]);bs={};values={}
        assert set(dd.rep)=={1,2,3};assert n==dict(nusc=152,av2=300)[ds]
        for lag,g in dd.groupby('lag'):
            codes=pd.Categorical(g.unit,categories=units).codes;values[lag]=metrics(g,np.ones(len(g)));
            x=g.ms.to_numpy();order=np.argsort(x);xx=x[order];ii=codes[order];samples=[]
            for count in C:
                w=count[codes];cum=np.cumsum(count[ii]);nt=cum[-1];qq=[]
                for q in (.95,.99):
                    t=q*(nt-1);lo=int(np.floor(t));hi=int(np.ceil(t));frac=t-lo
                    qq.append((1-frac)*xx[np.searchsorted(cum,lo,side='right')]+frac*xx[np.searchsorted(cum,hi,side='right')])
                samples.append([np.dot(x,w)/w.sum(),*qq])
            bs[lag]=np.array(samples)
            summary.append(dict(dataset=ds,lag=lag,units=n,frames_per_rep=len(g)//3,repetitions=3,**dict(zip(names,values[lag]))))
            for j,name in enumerate(names):
                lo,hi=np.percentile(bs[lag][:,j],[2.5,97.5]);cis.append(dict(dataset=ds,lag=lag,metric=name,estimate=values[lag][j],ci_lo=lo,ci_hi=hi))
            for rep,h in g.groupby('rep'):reps.append(dict(dataset=ds,lag=lag,rep=rep,**dict(zip(names,metrics(h,np.ones(len(h)))))))
            for scope,sel in [('nonzero_extra',g.extra_action>0),('raises_r5',g.raises_r5)]:
                h=g[sel]
                if len(h):active.append(dict(dataset=ds,lag=lag,scope=scope,frame_evaluations=len(h),**dict(zip(names,metrics(h,np.ones(len(h)))))))
        for j,name in enumerate(names):
            lo,hi=np.percentile(bs[2][:,j]-bs[1][:,j],[2.5,97.5]);diff.append(dict(dataset=ds,comparison='two-lag minus one-lag component',metric=name,estimate=values[2][j]-values[1][j],ci_lo=lo,ci_hi=hi))
    for name,rows in [('summary',summary),('intervals',cis),('contrasts',diff),('by_repetition',reps),('active_frames',active)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    (out/'REPORT.md').write_text('# Jetson bounded-readout component timing\n\nExclusive MAXN, locked clocks, fan255, one thread; three repetitions of every original development frame, with 200-frame warm-up per dataset and alternating row order. Paired unit bootstrap 2,000 seed0. See all intervals and per-repetition rows.\n\nTimes cover only the extra current-observation matching, C1 and capped-action function. Detector inference, cross-crop proposal suppression, tensor copies, buffering, R5 classifier and image decode are excluded. These are not total method latency, incremental full-pipeline latency, FPS or TRT parity. `active_frames.csv` is conditional timing, not an unbiased population average. `parity.json` checks CPU readout actions on stored observations against VUW.\n')
    tab=['','| Dataset | Prior observations | Mean ms [95% CI] | p95 ms [95% CI] | p99 ms [95% CI] |','|---|---:|---|---|---|']
    for r in summary:
        values=[]
        for metric in names:
            ci=next(x for x in cis if x['dataset']==r['dataset'] and x['lag']==r['lag'] and x['metric']==metric)
            values.append(f'{ci["estimate"]:.3f} [{ci["ci_lo"]:.3f}, {ci["ci_hi"]:.3f}]')
        tab.append('| '+r['dataset']+' | '+str(r['lag'])+' | '+' | '.join(values)+' |')
    with (out/'REPORT.md').open('a') as f:f.write('\n'.join(tab)+'\n')
    (out/'DONE.json').write_text(json.dumps(dict(status='complete',bootstrap=2000,seed=0,scope='component microbenchmark'),indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
