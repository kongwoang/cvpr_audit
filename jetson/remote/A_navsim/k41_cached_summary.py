#!/usr/bin/env python3
"""K41 CPU-only audit of already saved NAVSIM interactions and planning scores."""
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

B=Path(os.environ['B']);OUT=B/'results/k41/nav_cached';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(B/'results/k40r/vendor'))
from analysis.common import Bootstrap
sys.path.insert(0,str(B/'scripts/A_method'))
from k40r_cpu import interval
MANIFEST=[]

def read(pattern):
    rows=[]
    for f in sorted(B.glob(pattern)):
        assert 'holdout' not in str(f).lower()
        raw=f.read_bytes();MANIFEST.append(dict(path=str(f),md5=hashlib.md5(raw).hexdigest(),bytes=len(raw)))
        rows.extend(pickle.loads(raw))
    d=pd.DataFrame(rows)
    assert len(d)>0,pattern
    assert not d.duplicated(['log','token']).any(),pattern
    return d

def estimate(d,num,den):
    bt=Bootstrap(d.log)
    val,lo,hi,n=interval(bt,np.asarray(num,float),np.asarray(den,float))
    return dict(value=val,lo=lo,hi=hi,valid_draws=n,tokens=len(d),logs=d.log.nunique(),numerator=float(np.sum(num)),denominator=float(np.sum(den)))

def main():
    out=[]
    d=read('results/k20/pairs__*.pkl')
    for key,den in [('pairs_moving','n_pairs_tested'),('probe_drop_moving','n_probe_drop_tested')]:
        hit=d[key].map(len).to_numpy();eligible=d[den].to_numpy()
        out.append(dict(task='K41-2',consumer='PDM-Closed',metric=key,**estimate(d,hit,eligible)))
        out.append(dict(task='K41-2',consumer='PDM-Closed',metric=key+'_tokens',**estimate(d,hit>0,np.ones(len(d)))))
    d=read('results/k31/p2_pairs/*.pkl')
    for den in ['pairs','pairs_neither_single']:
        out.append(dict(task='K41-2',consumer='bottleneck selector',metric='joint_probe_changes / '+den,
          **estimate(d,d.pairs_joint_change,d[den])))
    pd.DataFrame(out).to_csv(OUT/'k41_2_interaction_audit.csv',index=False)
    d=read('results/k31/p2/*.pkl'); out=[]
    assert len(d)==12146
    for name in ['cheap','native','oracle','real']:
        valid=d['pi_'+name]>=0
        for sub,m in [('all',valid),('cheap_disagrees',valid&(d.pi_cheap!=d.pi_full960))]:
            out.append(dict(task='K41-1',row=name,subset=sub,metric='agreement',**estimate(d,(d['pi_'+name]==d.pi_full960)&m,m)))
        for pre in ['px','rounds','capped']:
            col=pre+'_'+name
            if col in d:
                q=d[valid&d[col].notna()];out.append(dict(task='K41-1',row=name,subset='all',metric='summed_crop_area_fraction' if pre=='px' else pre,**estimate(q,q[col],np.ones(len(q)))))
    pd.DataFrame(out).to_csv(OUT/'k41_1_selector_cached.csv',index=False)
    mapping=d[['log','token']]
    # PDMS below is historical navsim_pdms score; it is distinct from the official devkit EPDMS export.
    pdm=read('results/navsim_exp/pdms_final/*.pkl')
    k5=read('results/k39/k5/*.pkl')
    pdm=pdm.merge(k5,on=['log','token'],validate='1:1');out=[]
    for sub,m in [('all',np.ones(len(pdm),bool)),('trajectory_differs',pdm.traj_differs.astype(bool))]:
        q=pdm[m]
        for br in ['cheap','full960','native','spec','spec_pdm','reference']:
            col='score_'+br
            if col not in q:continue
            ok=q[col].notna()&q.score_spec_pdm.notna();z=q[ok]
            out.append(dict(task='K41-1',protocol='historical PDMS',row=br,subset=sub,metric='score',**estimate(z,z[col],np.ones(len(z)))))
            out.append(dict(task='K41-1',protocol='historical PDMS',row=br,subset=sub,metric='spec_pdm_minus_row',**estimate(z,z.score_spec_pdm-z[col],np.ones(len(z)))))
    for col in ['iou','pdm_only_share','brk_only_share','traj_differs','area_pdm','area_brk','n_pdm_crops','n_brk_crops']:
        out.append(dict(task='K41-1',protocol='K39-5 cached real crops',row='PDM vs braking',subset='all',metric=col,**estimate(pdm,pdm[col],np.ones(len(pdm)))))
    ep=pd.read_parquet(B/'tmp/desp_k40/artifacts/records/navsim_navtest.parquet')
    ep=ep.pivot(index='token',columns='branch',values='score').reset_index().merge(mapping,on='token',validate='1:1')
    assert len(ep)==12146 and not ep.isna().any().any()
    for br in ['cheap','full960','native','spec_pdm','hybrid_pdm','reference']:
        out.append(dict(task='K41-1',protocol='official devkit one-stage EPDMS',row=br,subset='all',metric='score',**estimate(ep,ep[br],np.ones(len(ep)))))
        out.append(dict(task='K41-1',protocol='official devkit one-stage EPDMS',row=br,subset='all',metric='spec_pdm_minus_row',**estimate(ep,ep.spec_pdm-ep[br],np.ones(len(ep)))))
    pd.DataFrame(out).to_csv(OUT/'k41_1_downstream_cached.csv',index=False)
    # The zero-hit bootstrap interval is degenerate and is not an upper confidence bound on unobserved interactions.
    pd.DataFrame(MANIFEST).to_csv(OUT/'sources.csv',index=False)
    (OUT/'run.json').write_text(json.dumps(dict(md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),draws=2000,seed=0,
      caveat='No new planner calls. Zero-event cluster bootstrap does not prove a zero population rate. Bottleneck downstream PDMS not saved.'),indent=2)+'\n')
    print(pd.read_csv(OUT/'k41_2_interaction_audit.csv').to_string(index=False),flush=True)
    print('cached NAVSIM audit complete',flush=True)

if __name__=='__main__':main()
