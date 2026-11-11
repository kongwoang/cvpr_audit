"""Same-host exact-output K31 selector cache benchmark, frozen K34 token manifest."""
import fcntl
import hashlib
import json
import os
import pickle
import socket
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from common import HERE,REMOTE,save,sha
sys.path.insert(0,str(REMOTE/'scripts/A_navsim'))
import k31_nav_p2 as K
import k34_selector_cost as T
import selector_cache as FAST

OUT=HERE/'reports/nav_selector'

def measure(fr,rec,cl,gz,priors,grid,method):
    start=time.perf_counter();cam=K.NP.Cam(*rec['cam']);ann=fr['anns']
    base,per=K.branch_objects(ann,rec['cheap'],cam,gz,priors)
    scene=K.Scene(ann,base,zip(per,np.asarray(rec['cheap']['xyxy'],float).reshape(-1,4)))
    sel=K.Selector(cl if isinstance(cl,np.ndarray) and len(cl)>=2 else None,float(np.hypot(*np.asarray(fr['ego_dynamic_state'],float)[:2])))
    probes=T.probes(cam,gz,priors,grid);prepared=time.perf_counter()
    output=(K.field_support if method=='original' else FAST.field_support)(sel,scene,probes)
    end=time.perf_counter();n,m=len(scene.o),len(probes)
    near=int((np.hypot(scene.o[:,0],scene.o[:,1])<=sel.reach+np.hypot(scene.o[:,5],scene.o[:,6])*K.CAP).sum())
    npb=sum(np.hypot(row[0],row[1])<=sel.reach+np.hypot(row[5],row[6])*K.CAP for row,_ in probes)
    count=15*(n+m) if method=='cached' else 15*(n+m*(n+1))
    ttcs=15*(near+npb) if method=='cached' else 15*(near*(1+m)+npb)
    return output,dict(method=method,grid=grid,setup_and_projection_ms=1000*(prepared-start),selector_ms=1000*(end-prepared),total_ms=1000*(end-start),
        objects=n,probes=m,candidate_score_reductions=15*(1+m),candidate_object_entries=count,candidate_object_TTC_evaluations=ttcs,
        whole_scene_h_evaluations=1 if method=='cached' else 1+m,probe_only_h_batches=int(m>0) if method=='cached' else 0,support_additional_scores=0)

def equal(a,b):
    return a[0]==b[0] and np.array_equal(np.asarray(a[1]),np.asarray(b[1])) and np.array_equal(np.asarray(a[2]),np.asarray(b[2]))

def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources={str(p):sha(p) for p in [Path(K.__file__),Path(K.NP.__file__),Path(K.NH.__file__),Path(T.__file__),Path(FAST.__file__),Path(__file__),REMOTE/'results/k34/selector/token_manifest.csv']}
    save(OUT/'sources_before.json',sources)
    manifest=pd.read_csv(REMOTE/'results/k34/selector/token_manifest.csv');assert len(manifest)==1000 and not manifest.duplicated(['unit','token']).any()
    manifest.to_csv(OUT/'token_manifest.csv',index=False)
    gz=float((K.EXP/'dets/ground_z.txt').read_text());priors=pickle.load((K.EXP/'dets/size_priors.pkl').open('rb'))
    rows=[];checks=[];warmed=False;start=time.time()
    save(OUT/'STATUS.json',dict(status='running',host=socket.gethostname(),pid=os.getpid(),started=start,scope='post-hoc runtime only, exact fixed K34/K20 navtest subset; no quality selection'))
    for unit,g in manifest.groupby('unit',sort=True):
        dets=pickle.load((K.EXP/'dets_k30b'/(unit+'.pkl')).open('rb'));cent=pickle.load((K.EXP/'k30_center'/(unit+'.pkl')).open('rb'))
        frames={r['token']:r for r in pickle.load((K.DATA/'navsim_logs/test'/(unit+'.pkl')).open('rb'))}
        old={r['token']:r for r in pickle.load((REMOTE/'results/k31/p2'/(unit+'.pkl')).open('rb'))}
        for token in g.token:
            args=(frames[token],dets[token],cent.get(token),gz,priors)
            if not warmed:
                for _ in range(5):
                    for grid in ['7x7','3x3']:
                        for method in ['original','cached']:measure(*args,grid,method)
                warmed=True
            for rep in [1,2,3]:
                for grid in (['7x7','3x3'] if (rep+len(checks))%2 else ['3x3','7x7']):
                    outputs={}
                    for method in (['original','cached'] if (rep+len(checks))%2 else ['cached','original']):
                        output,row=measure(*args,grid,method);outputs[method]=output
                        rows.append(dict(unit=unit,token=token,rep=rep,**row))
                    assert equal(outputs['original'],outputs['cached']),(unit,token,rep,grid)
                    if grid=='7x7':
                        a,field,support=outputs['original']
                        assert (a,len(field),len(support))==(old[token]['pi_cheap'],old[token]['field_cells_r1'],old[token]['support_r1'])
                    checks.append(dict(unit=unit,token=token,rep=rep,grid=grid,exact_action_field_support=True))
        pd.DataFrame(rows).to_csv(OUT/'frames.partial.csv',index=False)
        save(OUT/'STATUS.json',dict(status='running',host=socket.gethostname(),pid=os.getpid(),started=start,tokens=len(checks)//6,updated=time.time()))
        print(unit,'tokens',len(checks)//6,flush=True)
    d=pd.DataFrame(rows);assert len(d)==12000
    d.to_csv(OUT/'frames.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'parity.csv',index=False)
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'sources_after.json',{p:sha(p) for p in sources})
    summarize(d)
    save(OUT/'DONE.json',dict(status='complete',host=socket.gethostname(),tokens=1000,units=manifest.unit.nunique(),repetitions=3,
        comparisons=len(checks),exact_action_field_support_differences=0,source_hashes_unchanged=True,started=start,finished=time.time(),
        scope='post-hoc runtime on frozen existing navtest subset; no detection rerun, quality selection, core edits or general PDM-Closed speedup claim'))
    save(OUT/'STATUS.json',dict(status='complete',finished=time.time()))

def summarize(d):
    summary=[];contrasts=[]
    for grid,g in d.groupby('grid'):
        units=sorted(g.unit.unique());n=len(units);draw=np.random.default_rng(0).integers(0,n,(2000,n));C=np.array([np.bincount(x,minlength=n) for x in draw])
        samples={};points={}
        for method,q in g.groupby('method'):
            u=q.groupby('unit').sum(numeric_only=True).reindex(units);counts=q.groupby('unit').size().reindex(units).to_numpy()
            for metric in ['total_ms','selector_ms','setup_and_projection_ms','candidate_object_entries','candidate_object_TTC_evaluations','whole_scene_h_evaluations']:
                arr=q[metric].to_numpy();idx=pd.Categorical(q.unit,categories=units).codes
                bs=(C@u[metric].to_numpy())/(C@counts);point=float(arr.mean());samples[(method,metric)]=bs;points[(method,metric)]=point
                summary.append(dict(grid=grid,method=method,metric='mean_'+metric,estimate=point,ci_lo=np.percentile(bs,2.5),ci_hi=np.percentile(bs,97.5)))
                if metric=='total_ms':
                    order=np.argsort(arr);xx=arr[order];ii=idx[order];values=[]
                    for c in C:
                        cumulative=np.cumsum(c[ii]);target=.95*(cumulative[-1]-1);low=int(np.floor(target));high=int(np.ceil(target));frac=target-low
                        values.append((1-frac)*xx[np.searchsorted(cumulative,low,side='right')]+frac*xx[np.searchsorted(cumulative,high,side='right')])
                    samples[(method,'p95_total_ms')]=np.asarray(values);points[(method,'p95_total_ms')]=float(np.percentile(arr,95))
                    summary.append(dict(grid=grid,method=method,metric='p95_total_ms',estimate=np.percentile(arr,95),ci_lo=np.percentile(values,2.5),ci_hi=np.percentile(values,97.5)))
        for method,metric in samples:
            if method!='cached':continue
            bs=samples[('cached',metric)]-samples[('original',metric)]
            contrasts.append(dict(grid=grid,metric=metric,cached_minus_original=points[('cached',metric)]-points[('original',metric)],ci_lo=np.percentile(bs,2.5),ci_hi=np.percentile(bs,97.5)))
        ratio=samples[('original','total_ms')]/samples[('cached','total_ms')]
        contrasts.append(dict(grid=grid,metric='mean_total_speedup_ratio',cached_minus_original=points[('original','total_ms')]/points[('cached','total_ms')],ci_lo=np.percentile(ratio,2.5),ci_hi=np.percentile(ratio,97.5)))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv',index=False)

if __name__=='__main__':
    try:main()
    except BaseException as e:save(OUT/'STATUS.json',dict(status='failed',error=repr(e)));raise
