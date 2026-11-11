"""Paired unit intervals for matched-object holdout diagnostics, not scene metrics."""
import numpy as np
import pandas as pd
from common import HERE,save

def main():
    out=HERE/'reports/model_audit';summary=[];contrasts=[];paired=[]
    for ds in ('nusc','av2'):
        base=pd.read_csv(out/(ds+'_baseline.csv'));models={}
        for objective in ('geometry','decision'):
            df=pd.read_csv(HERE/'models'/ds/(objective+'_s0')/'validation.csv')
            assert len(base)==len(df)
            for c in ('unit','frame','kind'):assert (base[c].to_numpy()==df[c].to_numpy()).all(),(ds,c)
            for c in ('target','raw_range','gt_range','gt_demand'):assert np.allclose(base[c],df[c],atol=1e-5,equal_nan=True),(ds,c)
            models[objective]=dict(rate=df.rate.to_numpy(),factor=df.factor.to_numpy(),dem=df.pred_demand.to_numpy())
        models['frozen C1 + v3a']=dict(rate=base.C1.to_numpy(),factor=base.v3a_factor.to_numpy(),dem=base.v3a_demand.to_numpy())
        models['frozen C1 + raw']=dict(rate=base.C1.to_numpy(),factor=np.ones(len(base)),dem=base.raw_demand.to_numpy())
        strata={'all':np.ones(len(base),bool),'complete_crop':(base.kind=='crop')&(base.observations==3),
                'missing_crop':(base.kind=='crop')&(base.observations<3),'bridge':base.kind=='bridge',
                'cheap':base.kind=='cheap','known_switch':base.known_switch.astype(bool)}
        for stratum,ix in strata.items():
            b=base[ix].copy();units=sorted(b.unit.unique());n=len(units);assert n
            draws=np.random.default_rng(0).integers(0,n,(2000,n));C=np.array([np.bincount(a,minlength=n) for a in draws]);stats={}
            for name,values in models.items():
                rate,factor,dem=(values[k][ix] for k in ('rate','factor','dem'))
                action=(dem>=1).astype(int)+(dem>=3.5).astype(int);cmd=np.array([0,2.5,6])[action]
                req=b.gt_demand.to_numpy()>=1;false=(dem>=1)&~req;miss=(dem<1)&req
                frame=pd.DataFrame(dict(unit=b.unit.to_numpy(),n=1,rate_mae=np.abs(rate-b.target),
                    range_log_mae=np.abs(np.log(factor)-np.log(b.true_factor)),
                    log_demand_mae=np.abs(np.log1p(dem)-np.log1p(b.gt_demand)),
                    insufficient_loss=np.maximum(0,b.gt_demand-cmd)**2,false=false.astype(int),miss=miss.astype(int),
                    nonrequired=(~req).astype(int),required=req.astype(int)))
                u=frame.groupby('unit').sum(numeric_only=True).reindex(units);points={};dist={}
                for metric,den in [('rate_mae','n'),('range_log_mae','n'),('log_demand_mae','n'),('insufficient_loss','n'),('false','nonrequired'),('miss','required')]:
                    a=u[metric].to_numpy();d=u[den].to_numpy();bsden=C@d
                    bs=np.divide(C@a,bsden,out=np.full(2000,np.nan),where=bsden>0);lo,hi=np.nanpercentile(bs,[2.5,97.5]) if np.isfinite(bs).any() else [np.nan,np.nan]
                    point=float(a.sum()/d.sum()) if d.sum() else np.nan;points[metric]=point;dist[metric]=bs
                    summary.append(dict(dataset=ds,stratum=stratum,model=name,metric=metric,units=n,samples=len(b),estimate=point,ci_lo=lo,ci_hi=hi))
                stats[name]=(points,dist)
            for name,(points,dist) in stats.items():
                for comparator in ['frozen C1 + v3a','geometry']:
                    if name==comparator:continue
                    p0,b0=stats[comparator]
                    for metric,bs in dist.items():
                        lo,hi=np.nanpercentile(bs-b0[metric],[2.5,97.5]) if np.isfinite(bs-b0[metric]).any() else [np.nan,np.nan]
                        contrasts.append(dict(dataset=ds,stratum=stratum,model=name,comparator=comparator,metric=metric,
                            difference=points[metric]-p0[metric],ci_lo=lo,ci_hi=hi))
        base['geometry_rate']=models['geometry']['rate'];base['decision_rate']=models['decision']['rate']
        base['geometry_demand']=models['geometry']['dem'];base['decision_demand']=models['decision']['dem']
        base.to_csv(out/(ds+'_paired.csv'),index=False)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    save(out/'STATUS.json',dict(status='complete',units='training holdout only',rows=len(summary),bootstrap=2000,seed=0,
        caution='Object-matched diagnostic, all-object refinement. Not end-to-end scene, episode, coverage or test evidence.'))
    print('Model diagnostic complete',len(summary),'rows',flush=True)
if __name__=='__main__':main()
