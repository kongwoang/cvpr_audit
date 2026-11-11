"""K34-4 CPU timing of frozen K31 bottleneck selector, K20's 1,000 tokens."""
import json
import hashlib
import os
import pickle
import socket
import time
from pathlib import Path
import numpy as np
import pandas as pd
import k31_nav_p2 as K

B=Path(os.environ['B']);O=B/'results/k34/selector';E=K.EXP

def probes(cam,gz,priors,grid):
    xs,ys=(K.X_GRID,K.Y_GRID) if grid=='7x7' else (np.array([8.,23.,55.]),np.array([-3.5,0.,3.5]))
    L,W,H=priors['vehicle'];out=[]
    for x in xs:
        for y in ys:
            box=cam.project(np.array([x,y,gz+H/2,L,W,H,0.]))
            if box is not None:out.append((np.array([x,y,0.,L,W,0.,0.]),np.asarray(box,float)))
    return out

def measure(fr,rec,cl,gz,priors,grid):
    t0=time.perf_counter();cam=K.NP.Cam(*rec['cam']);ann=fr['anns']
    base,per=K.branch_objects(ann,rec['cheap'],cam,gz,priors)
    sc=K.Scene(ann,base,zip(per,np.asarray(rec['cheap']['xyxy'],float).reshape(-1,4)))
    sel=K.Selector(cl if isinstance(cl,np.ndarray) and len(cl)>=2 else None,float(np.hypot(*np.asarray(fr['ego_dynamic_state'],float)[:2])))
    t1=time.perf_counter();s,H=sel.score(sc.o);a=int(np.argmax(s));t2=time.perf_counter()
    prb=probes(cam,gz,priors,grid);field=[ib for row,ib in prb if sel.pi(np.vstack([sc.o,row]))!=a];t3=time.perf_counter()
    others=np.arange(len(s))!=a
    sup=[sc.dbox[k] for k in range(len(sc.o)) if sc.dbox[k] is not None and (H[others,k]<=s[a]).any()];t4=time.perf_counter()
    n=len(sc.o);near=int((np.hypot(sc.o[:,0],sc.o[:,1])<=sel.reach+np.hypot(sc.o[:,5],sc.o[:,6])*K.CAP).sum())
    near_probe=sum(np.hypot(row[0],row[1])<=sel.reach+np.hypot(row[5],row[6])*K.CAP for row,_ in prb)
    out={'variant':grid,'scene_selector_setup_ms':1000*(t1-t0),'base_score_ms':1000*(t2-t1),'field_ms':1000*(t3-t2),
         'binding_support_ms':1000*(t4-t3),'total_ms':1000*(t4-t0),'base_selector_calls':1,'probe_selector_calls':len(prb),
         'support_selector_calls':0,'candidate_score_evaluations':15*(1+len(prb)),
         'candidate_object_matrix_entries':15*(n+len(prb)*(n+1)),
         'candidate_object_TTC_evaluations':15*(near*(1+len(prb))+near_probe),
         'field_sensitive_cells':len(field),'binding_support_objects':len(sup),'selected_candidate':a}
    # Original implementation compared outside the timed window, on every token/grid.
    aa,ff,ss=K.field_support(sel,sc,prb)
    assert aa==a and np.array_equal(np.asarray(ff),np.asarray(field)) and np.array_equal(np.asarray(ss),np.asarray(sup))
    return out

def main():
    assert (B/'results/k34/GATE_K34_1_COMPLETE.json').exists()
    assert all((B/'results/k34/coverage'/ds/'DONE.json').exists() for ds in ('nusc','av2','ahd'))
    O.mkdir(parents=True,exist_ok=True)
    candidates=[];sources=[]
    def record_source(f):
        sources.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size})
    for f in (Path(__file__),B/'scripts/A_navsim/k31_nav_p2.py',B/'scripts/A_navsim/k20_pdm.py',E/'dets/ground_z.txt',E/'dets/size_priors.pkl'):
        record_source(f)
    for f in sorted((E/'field_test').glob('*.pkl')):
        record_source(f)
        candidates.extend((f.stem,t) for t,r in sorted(pickle.load(f.open('rb')).items()) if 'err' not in r)
    rng=np.random.default_rng(0);todo=[candidates[i] for i in sorted(rng.choice(len(candidates),size=min(1000,len(candidates)),replace=False))]
    assert len(todo)==1000
    pd.DataFrame(todo,columns=['unit','token']).to_csv(O/'token_manifest.csv',index=False)
    gz=float((E/'dets/ground_z.txt').read_text());priors=pickle.load((E/'dets/size_priors.pkl').open('rb'));rows=[];checks=[]
    warmed=False
    for log in sorted({u for u,t in todo}):
        for f in (E/'dets_k30b'/(log+'.pkl'),E/'k30_center'/(log+'.pkl'),K.DATA/'navsim_logs/test'/(log+'.pkl'),B/'results/k31/p2'/(log+'.pkl')):
            record_source(f)
        dets=pickle.load((E/'dets_k30b'/(log+'.pkl')).open('rb'));cent=pickle.load((E/'k30_center'/(log+'.pkl')).open('rb'))
        frames={r['token']:r for r in pickle.load((K.DATA/'navsim_logs/test'/(log+'.pkl')).open('rb'))}
        old={r['token']:r for r in pickle.load((B/'results/k31/p2'/(log+'.pkl')).open('rb'))}
        for _,tok in [q for q in todo if q[0]==log]:
            assert tok in dets and tok in frames and tok in old,(log,tok)
            args=(frames[tok],dets[tok],cent.get(tok),gz,priors)
            if not warmed:
                for _ in range(5):measure(*args,'7x7');measure(*args,'3x3')
                warmed=True
            for grid in (('7x7','3x3') if len(rows)%4==0 else ('3x3','7x7')):
                row=measure(*args,grid);rows.append({'unit':log,'token':tok,**row})
                if grid=='7x7':
                    assert row['selected_candidate']==old[tok]['pi_cheap']
                    assert row['field_sensitive_cells']==old[tok]['field_cells_r1']
                    assert row['binding_support_objects']==old[tok]['support_r1']
                    checks.append({'unit':log,'token':tok,'original_K31_equal':True})
        pd.DataFrame(rows).to_csv(O/'selector_cost_frames.partial.csv',index=False)
        print(log,'tokens',len(rows)//2,flush=True)
    assert len(rows)==2000
    pd.DataFrame(rows).to_csv(O/'selector_cost_frames.csv',index=False)
    pd.DataFrame(checks).to_csv(O/'validation.csv',index=False)
    # Historical measured PDM costs, same tokens, explicitly different host/protocol.
    prior=[]
    for f in sorted((E/'k30_probe_time').glob('*.pkl')):
        record_source(f);prior.extend(pickle.load(f.open('rb')))
    p=pd.DataFrame(prior);p=p[p.variant.isin(['7x7','3x3'])].rename(columns={'log':'unit'})
    assert not p.duplicated(['unit','token','variant']).any()
    want=pd.DataFrame(rows)[['unit','token','variant']]
    p=want.merge(p,on=['unit','token','variant'],how='left',validate='one_to_one',indicator=True)
    assert (p._merge=='both').all();p['total_ms']=p.wall_s*1000
    p.to_csv(O/'pdm_historical_frames.csv',index=False)
    pd.DataFrame(sources).to_csv(O/'source_manifest.csv',index=False)
    (O/'DONE.json').write_text(json.dumps({'host':socket.gethostname(),'tokens':1000,'frames':2000,'warmup_repeats':5,
                                        'pdm_comparison':'historical K30 measured PDM-Closed identical-output fast path; host not controlled',
                                        'load_average':os.getloadavg()},indent=2)+'\n')
    print('K34 SELECTOR DONE',flush=True)

if __name__=='__main__':main()
