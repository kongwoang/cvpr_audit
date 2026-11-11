"""Training-holdout interval calibration with unit-fold diagnostic coverage."""
import hashlib
import numpy as np
import pandas as pd
from common import HERE,save,sha

def correction(scores):
    x=np.sort(np.asarray(scores,float));assert len(x)
    return float(max(0,x[min(len(x)-1,int(np.ceil(.8*(len(x)+1)))-1)]))
def fit(d):
    residual=np.maximum(d.q10-d.target,d.target-d.q90);global_delta=correction(residual)
    groups={}
    for (kind,obs),g in d.groupby(['kind','observations']):
        groups[str(kind)+'/'+str(int(obs))]=dict(n=len(g),delta=correction(residual.loc[g.index]) if len(g)>=100 else global_delta,
                                               pooled=len(g)<100)
    return dict(global_delta=global_delta,groups=groups)
def apply(d,cal):
    delta=np.array([cal['groups'].get(str(k)+'/'+str(int(o)),{}).get('delta',cal['global_delta']) for k,o in zip(d.kind,d.observations)])
    return np.clip(d.q10.to_numpy()-delta,0,2),np.clip(d.q90.to_numpy()+delta,0,2)
def main():
    out=HERE/'reports/calibration';out.mkdir(parents=True,exist_ok=True);rows=[]
    for ds in ('nusc','av2'):
        for objective in ('geometry','decision'):
            folder=HERE/'models'/ds/(objective+'_s0');file=folder/'validation.csv';d=pd.read_csv(file)
            d['fold']=[int(hashlib.md5(str(u).encode()).hexdigest(),16)%5 for u in d.unit]
            d['cal_lo']=np.nan;d['cal_hi']=np.nan
            for fold in sorted(d.fold.unique()):
                tr=d[d.fold!=fold];va=d[d.fold==fold];cal=fit(tr);lo,hi=apply(va,cal)
                d.loc[va.index,'cal_lo']=lo;d.loc[va.index,'cal_hi']=hi
            cal=fit(d);cal.update(dataset=ds,objective=objective,coverage_target=.8,validation_sha256=sha(file),
                               scope='original training holdouts only; no finite-sample coverage claim after model selection')
            save(folder/'calibration.json',cal)
            for (kind,obs),g in d.groupby(['kind','observations']):
                for name,lo,hi in [('raw',g.q10,g.q90),('unit_fold_calibrated',g.cal_lo,g.cal_hi)]:
                    x=pd.DataFrame(dict(unit=g.unit,covered=((g.target>=lo)&(g.target<=hi)).astype(int),width=hi-lo,n=1))
                    u=x.groupby('unit')[['covered','width','n']].sum();n=len(u)
                    draws=np.random.default_rng(0).integers(0,n,(2000,n));C=np.array([np.bincount(a,minlength=n) for a in draws]);den=C@u.n.to_numpy()
                    for metric in ('covered','width'):
                        v=u[metric].to_numpy();bs=(C@v)/den;ci=np.percentile(bs,[2.5,97.5])
                        rows.append(dict(dataset=ds,objective=objective,kind=kind,observations=int(obs),variant=name,units=n,samples=len(g),
                            metric='coverage' if metric=='covered' else 'mean_width_per_s',estimate=v.sum()/u.n.sum(),ci_lo=ci[0],ci_hi=ci[1]))
            d.to_csv(out/(ds+'_'+objective+'_fold_predictions.csv'),index=False)
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);save(out/'STATUS.json',dict(status='complete'))
    print('Calibration complete',len(rows),'rows',flush=True)
if __name__=='__main__':main()
