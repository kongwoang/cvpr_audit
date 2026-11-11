#!/usr/bin/env python3
"""CPU cached scene diagnostic: exact bottleneck min reductions, no detector/PDM rollout."""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd

B=Path(os.environ['B']);X=B/'results/navsim_exp';OUT=B/'results/k41/scene_diagnostic';DATA=Path(os.environ['OPENSCENE_DATA_ROOT'])
s=importlib.util.spec_from_file_location('k31',B/'scripts/A_navsim/k31_nav_p2.py');K=importlib.util.module_from_spec(s);s.loader.exec_module(K)
from vtf import speculative as SP

def union_area(rects):
    # Exact continuous rectangle union; avoids millions of image-mask cells per token pair.
    r=np.asarray(rects,float).reshape(-1,4)
    if not len(r):return 0.
    xs=np.unique(r[:,[0,2]]);area=0.
    for a,b in zip(xs[:-1],xs[1:]):
        q=r[(r[:,0]<b)&(r[:,2]>a)];intervals=sorted((c[1],c[3]) for c in q if c[3]>c[1])
        length=0.;last=None
        for lo,hi in intervals:
            if last is None:last=[lo,hi]
            elif lo>last[1]:length+=last[1]-last[0];last=[lo,hi]
            else:last[1]=max(last[1],hi)
        if last is not None:length+=last[1]-last[0]
        area+=(b-a)*length
    return area

def iou(a,b):
    aa,bb=union_area(a),union_area(b);u=union_area(list(a)+list(b));return (aa+bb-u)/u if u else 1.

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-logs',type=int,default=0);args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    gz=float((X/'dets/ground_z.txt').read_text());priors=pickle.loads((X/'dets/size_priors.pkl').read_bytes())
    manifests=[]
    for li,f in enumerate(sorted((B/'results/k31/p2').glob('*.pkl'))):
        if args.max_logs and li>=args.max_logs:break
        dest=OUT/f.name
        if dest.exists():continue
        prev=pd.DataFrame(pickle.loads(f.read_bytes())).set_index('token')
        dets=pickle.loads((X/'dets_k30b'/f.name).read_bytes());cen=pickle.loads((X/'k30_center'/f.name).read_bytes())
        frames={r['token']:r for r in pickle.loads((DATA/'navsim_logs/test'/f.name).read_bytes())};rows=[]
        for tok in sorted(prev.index):
            t0=time.perf_counter();rec,fr=dets[tok],frames[tok];v=float(np.hypot(*np.asarray(fr['ego_dynamic_state'],float)[:2]));cam=K.NP.Cam(*rec['cam'])
            cl=cen.get(tok);sel=K.Selector(cl if isinstance(cl,np.ndarray) and len(cl)>=2 else None,v)
            base,per=K.branch_objects(fr['anns'],rec['cheap'],cam,gz,priors)
            sc=K.Scene(fr['anns'],base,zip(per,np.asarray(rec['cheap']['xyxy'],float).reshape(-1,4)))
            prb=K.probes(cam,gz,priors);score,Hm=sel.score(sc.o);a=int(score.argmax())
            Hp=sel.h(np.asarray([r for r,_ in prb]).reshape(-1,7));plans=np.minimum(score[:,None],Hp).argmax(axis=0);mask=plans!=a
            fb=[ib for yes,(_,ib) in zip(mask,prb) if yes]
            others=np.arange(15)!=a;sup=[sc.dbox[k] for k in range(len(sc.o)) if sc.dbox[k] is not None and (Hm[others,k]<=score[a]).any()]
            crops=K.crops_of(fb,sup,[],SP,1920,1080)
            assert a==int(prev.loc[tok,'pi_cheap']) and len(fb)==int(prev.loc[tok,'field_cells_r1']) and len(sup)==int(prev.loc[tok,'support_r1']),(f.stem,tok)
            # Probe cell IDs are world coordinates, so different cameras/projected grids remain explicit.
            cells=[tuple(row[:2]) for row,_ in prb];selected=[cells[i] for i in np.flatnonzero(mask)]
            rows.append(dict(log=f.stem,token=tok,v=v,candidate=a,cells=cells,sensitive=selected,crops=crops,
                crop_area=union_area(crops)/(1920*1080),real_area=float(prev.loc[tok,'px_real']),
                real_decision=int(prev.loc[tok,'pi_real']),target_decision=int(prev.loc[tok,'pi_full960']),
                cpu_s=time.perf_counter()-t0,planner_rollouts=0,singleton_queries_avoided=len(prb),
                camera_hash=hashlib.md5(b''.join(np.asarray(x).tobytes() for x in rec['cam'])).hexdigest(),
                candidate_poses_hash=hashlib.md5(sel.P.tobytes()).hexdigest()))
        with dest.open('wb') as fh:pickle.dump(rows,fh,pickle.HIGHEST_PROTOCOL)
        print(f.stem,len(rows),'parity passed',flush=True)
    d=pd.DataFrame([r for f in sorted(OUT.glob('*.pkl')) for r in pickle.loads(f.read_bytes())]);d['speed_bin']=np.floor(d.v).astype(int)
    pair=[]
    for key,g in d.groupby(['speed_bin','candidate']):
        g=g.sort_values('token');rr=g.to_dict('records')
        for a,b in zip(rr[::2],rr[1::2]):
            sa,sb=set(a['sensitive']),set(b['sensitive']);u=sa|sb
            pair.append(dict(log_a=a['log'],log_b=b['log'],token_a=a['token'],token_b=b['token'],speed_bin=key[0],candidate=key[1],
                speed_difference=abs(a['v']-b['v']),same_ego_speed=a['v']==b['v'],same_camera=a['camera_hash']==b['camera_hash'],
                same_candidate_poses=a['candidate_poses_hash']==b['candidate_poses_hash'],same_probe_domain=a['cells']==b['cells'],
                field_iou=len(sa&sb)/len(u) if u else 1.,field_differs=sa!=sb,crop_iou=iou(a['crops'],b['crops']),
                requested_crops_differ=canon(a['crops'])!=canon(b['crops']),crop_area_abs_difference=abs(a['crop_area']-b['crop_area']),
                real_area_abs_difference=abs(a['real_area']-b['real_area']),real_decision_differs=a['real_decision']!=b['real_decision'],
                target_decision_differs=a['target_decision']!=b['target_decision']))
    pd.DataFrame(pair).to_csv(OUT/'pairs.csv',index=False)
    d.drop(columns=['cells','sensitive','crops']).to_csv(OUT/'tokens.csv',index=False)
    (OUT/'run.json').write_text(json.dumps(dict(md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),tokens=len(d),pairs=len(pair),
        limitation='Matched speed bin and candidate index, not necessarily identical ego pose/candidate trajectories/camera. Descriptive association, not causal scene isolation.'),indent=2)+'\n')

def canon(c):return [np.asarray(v).tolist() for v in c]
if __name__=='__main__':main()
