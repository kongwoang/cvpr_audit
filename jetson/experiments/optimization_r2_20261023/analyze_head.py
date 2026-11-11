"""Every predeclared rate stratum, original frozen validation predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R2_20261023/head_audit'
LABELS=['zero','(0,0.1)','[0.1,0.25)','[0.25,0.5)','[0.5,2]']

def main():
    rows=[];contrasts=[];all_d=[];reliability=[]
    for ds in ['nusc','av2']:
        status=json.loads((OUT/(ds+'_STATUS.json')).read_text());assert status['status']=='complete'
        d=pd.read_csv(OUT/(ds+'_predictions.csv'));all_d.append(d)
        assert len(d)==status['samples']
        d['bin']=np.select([d.target==0,d.target<.1,d.target<.25,d.target<.5],LABELS[:4],default=LABELS[4])
        units=sorted(d.unit.unique());n=len(units);draws=np.random.default_rng(0).integers(0,n,(2000,n))
        weights=np.array([np.bincount(x,minlength=n) for x in draws],float)
        d['prediction_bin']=np.select([d.C1==0,d.C1<.1,d.C1<.25,d.C1<.5],LABELS[:4],default=LABELS[4])
        for name,g in d.groupby('prediction_bin'):
            g=g.copy();g['residual']=g.target-g.C1
            u=g.groupby('unit').agg(residual=('residual','sum'),n=('residual','size')).reindex(units,fill_value=0)
            den=weights@u.n.to_numpy();num=weights@u.residual.to_numpy()
            bs=np.divide(num,den,out=np.full(len(den),np.nan),where=den>0)
            lo,hi=np.nanpercentile(bs,[2.5,97.5])
            reliability.append(dict(dataset=ds,predicted_rate_bin=name,n=len(g),units=int(g.unit.nunique()),
                predicted_mean=float(g.C1.mean()),target_mean=float(g.target.mean()),
                residual_mean=float(g.residual.mean()),residual_median=float(g.residual.median()),
                residual_q75=float(g.residual.quantile(.75)),residual_mean_ci_lo=lo,residual_mean_ci_hi=hi))
        for kind in ['all','crop','u960']:
            for rate_bin in ['all']+LABELS:
                mask=np.ones(len(d),bool)
                if kind!='all':mask&=d.kind==kind
                if rate_bin!='all':mask&=d['bin']==rate_bin
                g=d[mask].copy()
                if g.empty:continue
                for model in ['C1','T4']:
                    error=g[model]-g.target;pos=g.target>0
                    rows.append(dict(dataset=ds,kind=kind,rate_bin=rate_bin,model=model,n=len(g),units=int(g.unit.nunique()),
                        target_mean=float(g.target.mean()),prediction_mean=float(g[model].mean()),mae=float(error.abs().mean()),
                        bias=float(error.mean()),under_fraction=float((error<0).mean()),over_fraction=float((error>0).mean()),
                        below_half_of_true=float((g.loc[pos,model]<.5*g.loc[pos,'target']).mean()) if pos.any() else np.nan))
                for metric in ['mae','bias']:
                    g['delta']=(g.C1-g.target).abs()-(g.T4-g.target).abs() if metric=='mae' else g.C1-g.T4
                    u=g.groupby('unit').agg(delta=('delta','sum'),n=('delta','size')).reindex(units,fill_value=0)
                    den=weights@u.n.to_numpy();num=weights@u.delta.to_numpy()
                    bs=np.divide(num,den,out=np.full(len(den),np.nan),where=den>0)
                    lo,hi=np.nanpercentile(bs,[2.5,97.5])
                    contrasts.append(dict(dataset=ds,kind=kind,rate_bin=rate_bin,metric=metric,
                        C1_minus_T4=float(g.delta.mean()),ci_lo=lo,ci_hi=hi,valid_draws=int(np.isfinite(bs).sum())))
    s=pd.DataFrame(rows);s.to_csv(OUT/'strata.csv',index=False);pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv',index=False)
    pd.DataFrame(reliability).to_csv(OUT/'reliability.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),sharey=True)
    for ax,ds in zip(axes,['nusc','av2']):
        for name,col in [('C1','#4477aa'),('T4','#cc6677')]:
            g=s[(s.dataset==ds)&(s.kind=='all')&(s.model==name)].set_index('rate_bin').reindex(LABELS)
            ax.plot(range(5),g.bias,'o-',label=name,color=col)
        ax.axhline(0,color='gray',ls='--');ax.set_xticks(range(5));ax.set_xticklabels(LABELS,fontsize=8)
        ax.set(xlabel='True relative closing rate (1/s)',title=ds);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Mean predicted − true rate (1/s)')
    fig.suptitle('Frozen C1: original checkpoint-selection validation set\nConditional bias is descriptive; no new training or calibration',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(OUT/'rate_bias.png',dpi=180);fig.savefig(OUT/'rate_bias.pdf');plt.close(fig)
    print(s[(s.kind=='all')].to_string(index=False))

if __name__=='__main__':main()
