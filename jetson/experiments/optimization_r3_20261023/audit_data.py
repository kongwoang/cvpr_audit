"""CPU identity audit on regenerated training records, with paired-unit bootstrap."""
import json
import numpy as np
import pandas as pd
from common import HERE,save

def main():
    out=HERE/'reports/training_audit';out.mkdir(parents=True,exist_ok=True);rows=[];manifest=[]
    for ds in ('nusc','av2'):
        folder=HERE/'training_data'/ds
        done=[json.loads(p.read_text()) for p in folder.glob('DONE_*.json')]
        expected=sorted(u for d in done for u in d['units']);assert len(expected)==len(set(expected))
        actual=sorted(p.stem for p in folder.glob('*.npz'));assert expected==actual
        for f in sorted(folder.glob('*.npz')):
            meta=json.loads(f.with_suffix('.json').read_text());assert meta['dataset']==ds;manifest.append(meta)
            with np.load(f,allow_pickle=False) as z:
                if not len(z['frame']):continue
                kind=z['kind'];sp=z['split'];mask=z['mask'];identity=z['identity'];rate=z['target'];ids=z['ids']
                for k in np.unique(kind):
                    ix=kind==k;available=mask[ix].sum(1);ident=identity[ix];gtid=ids[ix];y=rate[ix]
                    full=available==3;known=full&(gtid>=0).all(1);wrong=np.any(ident[:,1:]==0,axis=1)
                    rows.append(dict(dataset=ds,split=str(sp[ix][0]),unit=f.stem,kind=str(k),samples=int(ix.sum()),
                        full=int(full.sum()),known_full=int(known.sum()),switched_full=int((known&wrong).sum()),
                        incomplete=int((available<3).sum()),unmatched_current=int((gtid[:,0]<0).sum()),
                        finite_target=int(np.isfinite(y).sum()),high_rate=int((y>=.5).sum())))
    u=pd.DataFrame(rows);u.to_csv(out/'units.csv',index=False);summary=[]
    metrics={'known_full_switch_rate':('switched_full','known_full'),'incomplete_rate':('incomplete','samples'),
             'unmatched_current_rate':('unmatched_current','samples')}
    for (ds,sp,kind),g in u.groupby(['dataset','split','kind']):
        n=len(g);draws=np.random.default_rng(0).integers(0,n,(2000,n));C=np.array([np.bincount(a,minlength=n) for a in draws])
        totals=g.select_dtypes(include='number').sum().to_dict()
        for metric,(num,den) in metrics.items():
            nn=g[num].to_numpy();dd=g[den].to_numpy();bn=C@nn;bd=C@dd
            bs=np.divide(bn,bd,out=np.full(2000,np.nan),where=bd>0)
            lo,hi=np.nanpercentile(bs,[2.5,97.5]) if np.isfinite(bs).any() else [np.nan,np.nan]
            summary.append(dict(dataset=ds,split=sp,kind=kind,units=n,metric=metric,estimate=nn.sum()/max(dd.sum(),1),ci_lo=lo,ci_hi=hi,**totals))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    save(out/'manifest.json',dict(status='complete',units=manifest,bootstrap=2000,seed=0))
    print(u.groupby(['dataset','split','kind'])[['samples','known_full','switched_full','incomplete']].sum().to_string())
if __name__=='__main__':main()
