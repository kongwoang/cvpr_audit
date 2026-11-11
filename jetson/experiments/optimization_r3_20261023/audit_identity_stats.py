"""Unit-bootstrap uncertainty for original-sample alignment and pair classifier."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import HERE,save

def counts(n):
    return np.array([np.bincount(x,minlength=n) for x in np.random.default_rng(0).integers(0,n,(2000,n))])
def stats(point,bs):
    lo,hi=np.nanpercentile(bs,[2.5,97.5]);return dict(estimate=float(point),ci_lo=float(lo),ci_hi=float(hi))
def main():
    out=HERE/'reports/training_audit';rows=[]
    for ds in ('nusc','av2'):
        d=pd.read_csv(out/(ds+'_original_alignment.csv'));d=d[np.isfinite(d.target)&(d.matches==1)].copy()
        d['known']=d.known_all.eq(True)|d.known_all.eq('True');d['wrong']=d.switched.eq(True)|d.switched.eq('True')
        for (sp,kind),g in d.groupby(['split','kind']):
            u=g.groupby('unit')[['known','wrong']].sum();C=counts(len(u));den=C@u.known.to_numpy()
            bs=np.divide(C@u.wrong.to_numpy(),den,out=np.full(2000,np.nan),where=den>0)
            rows.append(dict(dataset=ds,split=sp,kind=kind,units=len(u),original_samples=len(g),known_chains=int(u.known.sum()),
                             switched_chains=int(u.wrong.sum()),**stats(u.wrong.sum()/u.known.sum(),bs)))
    pd.DataFrame(rows).to_csv(out/'original_identity_intervals.csv',index=False)
    out=HERE/'reports/identity_classifier';out.mkdir(exist_ok=True);rows=[]
    for ds in ('nusc','av2'):
        path=HERE/'models'/ds/'identity_pair_s0/validation.csv';d=pd.read_csv(path)
        units=sorted(d.unit.unique());index={u:i for i,u in enumerate(units)};u=np.array([index[x] for x in d.unit]);y=d.bad.to_numpy(bool);p=d.score.to_numpy();n=len(units)
        pos=np.bincount(u[y],minlength=n);neg=np.bincount(u[~y],minlength=n);A=np.zeros((n,n))
        for j in range(n):
            s=np.sort(p[(u==j)&~y]);wins=.5*(np.searchsorted(s,p[y],side='left')+np.searchsorted(s,p[y],side='right'))
            A[:,j]=np.bincount(u[y],weights=wins,minlength=n)
        point=A.sum()/pos.sum()/neg.sum();assert abs(point-roc_auc_score(y,p))<1e-10
        C=counts(n);den=(C@pos)*(C@neg);bs=np.divide(np.sum((C@A)*C,axis=1),den,out=np.full(2000,np.nan),where=den>0)
        rows.append(dict(dataset=ds,metric='ROC_AUC',units=n,pairs=len(d),**stats(point,bs)))
        reject=d.reject.eq(True).to_numpy()|d.reject.eq('True').to_numpy()
        for metric,select in [('switch_recall',y),('valid_pair_rejection',~y)]:
            num=np.bincount(u[select],weights=reject[select],minlength=n);dd=np.bincount(u[select],minlength=n);den=C@dd
            bs=np.divide(C@num,den,out=np.full(2000,np.nan),where=den>0)
            rows.append(dict(dataset=ds,metric=metric,units=n,pairs=len(d),**stats(num.sum()/dd.sum(),bs)))
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    save(out/'STATUS.json',dict(status='complete',bootstrap=2000,seed=0,scope='model/threshold selection holdout, exploratory'))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
