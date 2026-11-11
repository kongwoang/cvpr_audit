"""Mean/p95 logical detector pixels, all full-development acquisition rows."""
import hashlib
import numpy as np
import pandas as pd
from common import HERE,save
STAGES=['final_nusc','final_av2']

def main():
    cache={};rows=[];contrasts=[]
    for stage in STAGES:
        file=HERE/'reports'/stage/'frames.csv';d=pd.read_csv(file)
        for ds,g in d.groupby('dataset'):
            g=g.sort_values(['unit','frame']);units=sorted(g.unit.unique());n=len(units);assert n=={'nusc':152,'av2':300}[ds]
            draws=np.random.default_rng(0).integers(0,n,(2000,n));C=np.array([np.bincount(x,minlength=n) for x in draws]);idx=pd.Categorical(g.unit,categories=units).codes
            samples={};points={}
            for col in g:
                if not col.startswith('px_'):continue
                x=g[col].to_numpy(float);key=(ds,hashlib.sha256(x.tobytes()).hexdigest());order=np.argsort(x);xx=x[order];ii=idx[order]
                if key not in cache:
                    sums=np.bincount(idx,weights=x,minlength=n);sizes=np.bincount(idx,minlength=n)
                    means=(C@sums)/(C@sizes);q=[]
                    for count in C:
                        cum=np.cumsum(count[ii]);target=.95*(cum[-1]-1);lo=int(np.floor(target));hi=int(np.ceil(target));f=target-lo
                        q.append((1-f)*xx[np.searchsorted(cum,lo,side='right')]+f*xx[np.searchsorted(cum,hi,side='right')])
                    cache[key]=np.array([means,q])
                samples[col]=cache[key];points[col]=[x.mean(),np.percentile(x,95)]
                for j,metric in enumerate(['mean_network_px','p95_network_px']):
                    lo,hi=np.percentile(samples[col][j],[2.5,97.5]);rows.append(dict(suite=stage,dataset=ds,acquisition=col[3:],metric=metric,estimate=points[col][j],ci_lo=lo,ci_hi=hi,units=n,frames=len(g)))
            for col,bs in samples.items():
                for base in ['px_v6S','px_v6S_b2A_retro']:
                    if col==base:continue
                    for j,metric in enumerate(['mean_network_px','p95_network_px']):
                        lo,hi=np.percentile(bs[j]-samples[base][j],[2.5,97.5]);contrasts.append(dict(suite=stage,dataset=ds,acquisition=col[3:],comparator=base[3:],metric=metric,candidate_minus_baseline=points[col][j]-points[base][j],ci_lo=lo,ci_hi=hi))
            print(stage,ds,'complete',flush=True)
    out=HERE/'reports/detector_cost';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    (out/'REPORT.md').write_text('# Detector work distribution\n\nAll complete acquisition rows; downstream refiners share the same detection path and therefore pixels. Mean and p95 sum all requested draft/current/historical detector input pixels divided by native frame pixels. Paired unit bootstrap 2,000 seed0; percentile is recomputed in each bootstrap sample with NumPy linear interpolation. These are VUW logical canvases; use Jetson reports for actual engine padding, executions and latency.\n')
    save(out/'DONE.json',dict(status='complete',rows=len(rows),bootstrap=2000,seed=0))
if __name__=='__main__':main()
