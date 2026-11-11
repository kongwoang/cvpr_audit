#!/usr/bin/env python3
"""K41-6 / LAT-1 CPU profiling of immutable K39 traces, never a new timing run."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

B=Path(os.environ['B']);ROOT=B/'results/latency_opt/inputs/results/raw';OUT=B/'results/latency_opt/cached'
sys.path.insert(0,str(B/'results/k40r/vendor'))
from analysis.common import Bootstrap
from k40r_cpu import interval

def quantile_ci(d,col,q):
    # Exactly the original unit draws; all three repetitions of sampled units remain together.
    bt=Bootstrap(d.unit); vals=d[col].to_numpy(float);order=np.argsort(vals,kind='stable');v=vals[order];ui=bt.inv[order]
    dist=[]
    for ids in bt.idx:
        counts=np.bincount(ids,minlength=len(bt.units));ww=counts[ui];cum=np.cumsum(ww);n=cum[-1]
        if n==0:continue
        at=(n-1)*q;lo=int(np.floor(at));hi=int(np.ceil(at));f=at-lo
        a=v[np.searchsorted(cum,lo+1)];b=v[np.searchsorted(cum,hi+1)]
        dist.append(a+(b-a)*f)
    return np.percentile(dist,[2.5,97.5])

def stats(d,col,ci=False):
    x=d[col].to_numpy(float);out=dict(frames_x_reps=len(x),units=d.unit.nunique(),mean_ms=float(x.mean()),max_ms=float(x.max()))
    for q in (50,90,95,99):out[f'p{q}_ms']=float(np.percentile(x,q))
    for deadline in (33.3,50,66.7,100):out[f'late_{deadline:g}']=float(np.mean(x>deadline))
    if ci:
        bt=Bootstrap(d.unit);_,lo,hi,_=interval(bt,x,np.ones(len(x)));out['mean_lo']=lo;out['mean_hi']=hi
        for q in (95,99):out[f'p{q}_lo'],out[f'p{q}_hi']=quantile_ci(d,col,q/100)
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True);sources=[];tables={};profiles=[];conds=[];miss=[]
    for path in sorted(ROOT.glob('*/frames.pkl')):
        cfg=json.loads((path.parent/'config.json').read_text());d=pd.read_pickle(path);d=d[~d.warm].copy();d.unit=d.unit.astype(str)
        sources.append(dict(path=str(path),md5=hashlib.md5(path.read_bytes()).hexdigest(),frames=len(d),dataset=cfg['dataset'],breakdown=cfg['breakdown']))
        if cfg['breakdown']:
            for c in d:
                if not c.startswith('bd_'):continue
                z=d[['unit',c]].copy();z[c]=z[c].fillna(0)
                profiles.append(dict(task='LAT-1',dataset=cfg['dataset'],protocol='K38 staged fast instrumented dev prefix; sync overhead; no static graphs',
                    stage=c,**stats(z,c),gpu_kernel_launches=np.nan,transferred_bytes=np.nan))
            continue
        d['rep']=cfg['rep'];tables.setdefault(cfg['dataset'],[]).append(d)
    rows=[]
    for ds,parts in tables.items():
        assert len(parts)==3
        d=pd.concat(parts,ignore_index=True);expected=805 if ds=='nusc_test20' else 2968
        assert all(len(p)==expected for p in parts)
        for col in d:
            if not col.startswith('ms_') or '_noref' in col or '_dl' in col:continue
            key=col[3:];r=dict(task='LAT-8',dataset=ds,row=key,protocol='historical K39 static graphs; measured 2026-10-27',**stats(d,col,True))
            for metric in ('execs','px'):
                c=metric+'_'+key;r[metric+'_mean']=float(d[c].mean()) if c in d else np.nan
            r['throughput_fps_reciprocal_mean']=1000/r['mean_ms'];r['new_optimization']=False
            rows.append(r)
        name='ours@trt32';col='ms_'+name
        for feature in ['rounds_'+name,'n_crops_'+name,'crop_engine_exec_'+name,'n_refined_'+name,'a_'+name]:
            for val,g in d.groupby(feature):
                conds.append(dict(task='K41-6',dataset=ds,condition=feature,value=val,**stats(g,col)))
                for D in (50,66.7):
                    miss.append(dict(task='K41-6',dataset=ds,condition=feature,value=val,deadline_ms=D,
                        frames_x_reps=len(g),late_frames=int((g[col]>D).sum()),all_late_frames=int((d[col]>D).sum()),
                        share_of_late_frames=float((g[col]>D).sum()/max((d[col]>D).sum(),1)),
                        interpretation='association; conditions overlap; not causal attribution'))
    pd.DataFrame(rows).to_csv(OUT/'BENCHMARK.csv',index=False)
    pd.DataFrame(profiles).to_csv(OUT/'PROFILE.csv',index=False)
    pd.DataFrame(conds).to_csv(OUT/'k41_6_conditionals.csv',index=False)
    pd.DataFrame(miss).to_csv(OUT/'k41_6_late_associations.csv',index=False)
    pd.DataFrame(sources).to_csv(OUT/'sources.csv',index=False)
    (OUT/'run.json').write_text(json.dumps(dict(md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),new_device_execution=False,
        missing=['per-call GPU timing','actual guard count','transferred bytes','kernel launches','full K39 13-stage breakdown','new optimization impact']),indent=2)+'\n')
    print(pd.DataFrame(rows)[['dataset','row','mean_ms','p95_ms','p99_ms','p95_lo','p95_hi']].to_string(index=False),flush=True)

if __name__=='__main__':main()
