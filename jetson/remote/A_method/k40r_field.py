#!/usr/bin/env python3
"""K40-4/7 static-tube equality and cached-state oracle replay; K41-3 CPU grid only."""
import argparse
import gzip
import hashlib
import itertools
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rap import planner as P
from vtf import dsf
from vtf import speculative as SP
import k41_cached_objects as O

B=Path(os.environ['B']);OUT=B/'results/k40r/field';PP=P.PlannerParams()
RG,YG=SP.R_GRID,SP.Y_GRID

def mask(v,a,rg=RG,yg=YG,w=1.8,center=False):
    if a==2 or v<=.1:return np.zeros((len(rg),len(yg)),bool)
    tau=(PP.thr_decel,PP.thr_hard)[a]
    length=PP.standoff+v*PP.reaction_t+v*v/(2*tau)
    lat=(np.abs(yg)<PP.corridor_half_w) if center else ((yg-w/2<PP.corridor_half_w)&(yg+w/2>-PP.corridor_half_w))
    return ((rg<=length)&(rg<=PP.max_range)&(tau<=PP.a_cap))[:,None]&lat[None,:]

def reference(v,a,rg=RG,yg=YG,w=1.8,vc=0.):
    if a==2:return np.zeros((len(rg),len(yg)),bool)
    return (dsf._insert_demand(np.asarray(rg,float),v,PP,vc)>=(PP.thr_decel,PP.thr_hard)[a])[:,None]&((yg-w/2<PP.corridor_half_w)&(yg+w/2>-PP.corridor_half_w))[None,:]

def project(m,cam,W,H,rg=RG,yg=YG,w=1.8,h=1.6):
    if not m.any():return None
    mat=cam.calib.P2;fx,fy,cx,cy=mat[0,0],mat[1,1],mat[0,2],mat[1,2]
    rr,yy=np.meshgrid(rg,yg,indexing='ij');rr,yy=rr[m],yy[m]
    vb=cy+fy*cam.cam_h/rr;uc=cx-fx*yy/rr;hw=fx*w/(2*rr)
    return np.array([max((uc-hw).min(),0),max((vb-fy*h/rr).min(),0),min((uc+hw).max(),W),min(vb.max(),H)])

def bits(m):return np.packbits(m.ravel()).tobytes().hex()
def canon(x):
    if x is None:return None
    if isinstance(x,(list,tuple)):return [canon(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=O.NAMES,required=True);ap.add_argument('--max-units',type=int,default=0)
    args=ap.parse_args();ds=args.dataset;O.AV2_MASK=ds=='av2';name=O.NAMES[ds];od=OUT/ds;od.mkdir(parents=True,exist_ok=True)
    base=pd.read_parquet(O.R/(name+'.parquet'));K24=O.module('k40r_k24',B/'scripts/A_method/k24_test.py');KL=K24.KL
    checks=[];manifest=[];n=0
    for unit,cam,gen in K24.units(ds):
        d=base[base.unit==unit]
        if not len(d):continue
        if args.max_units and n>=args.max_units:break
        n+=1
        target=od/(unit+'.json.gz')
        if target.exists():continue
        x=O.load(B/'results/k30/apdump'/ds/(unit+'.pkl'),manifest)
        cr={int(r['frame']):r for r in O.load(B/'results/k30/match'/ds/(unit+'.pkl'),manifest)}
        W,H=O.SIZES[ds];trace=[]
        for r in d.to_dict('records'):
            fr=int(r['frame']);v=float(r['v']);det=x['dets'];ch=det[(fr,'cheap')]['sc'];f=det[(fr,'wf960+C1')]['sc'];o=det[(fr,'v6S')]['sc']
            crops=cr[fr]['crops_v6S'];nc=cr[fr]['ncrops_v6S']
            f0=O.no_ref(f,cam,ch,crops,True);o0=O.no_ref(o,cam,ch,crops[:nc],grown=len(crops)>nc,v=v)
            source_parity=O.act(f0,v)==r['a_wf960+C1|none'] and o0 is not None and O.act(o0,v)==r['a_v6S|none']
            records=[]
            for stage,a,geo in [('cheap',int(r['a_cheap']),ch['geo']),('desp_no_ref',int(r['a_v6S|none']),o0)]+[(f'synthetic_action_{a}',a,None) for a in range(3)]:
                ref=reference(v,a);tube=mask(v,a);literal=mask(v,a,center=True)
                actual_equal=None if geo is None else bool(np.array_equal(ref,dsf.insertion_grid(geo,v,RG,YG,1.8,PP,0.)))
                rc,tc=project(ref,cam,W,H),project(tube,cam,W,H)
                records.append(dict(stage=stage,action=a,mask_reference=bits(ref),mask_tube=bits(tube),
                    equal=bool(np.array_equal(ref,tube)),literal_center_equal=bool(np.array_equal(ref,literal)),
                    source_parity=source_parity,actual_scene_mask_equal=actual_equal,
                    sensitive_cells=int(ref.sum()),reference_crop=canon(rc),tube_crop=canon(tc),
                    projection_equal=(rc is None and tc is None) or (rc is not None and tc is not None and np.array_equal(rc,tc))))
            cheap={k:np.asarray(ch[k]) for k in ('xyxy','conf','coarse')};cheap['mono']=ch['geo']
            original=KL.field_crop_g
            histories=[]
            for alternative in (False,True):
                field_log=[];requests=[]
                def field(geo,vv,cc,ww,hh,pp,vcs,rg,yg,probes):
                    a=O.act(geo,vv)
                    m=mask(vv,a,rg,yg) if alternative else dsf.insertion_grid(geo,vv,rg,yg,1.8,pp,0.)
                    crop=project(m,cc,ww,hh,rg,yg)
                    field_log.append(dict(action=a,mask=bits(m),crop=canon(crop)))
                    return crop
                def detect(c):
                    requests.append(canon(np.asarray(c)))
                    keep=SP.centres_in(np.asarray(f['xyxy']),[c])
                    return {k:np.asarray(f[k])[keep].copy() for k in ('xyxy','conf','coarse')}
                def lift(full):
                    idx=[]
                    for i,b in enumerate(full['xyxy']):
                        js=np.flatnonzero(np.all(np.asarray(f['xyxy'])==b,axis=1)&(np.asarray(f['coarse'])==full['coarse'][i])&(np.asarray(f['conf'])==full['conf'][i]))
                        assert len(js)>0,(unit,fr,'missing oracle detection identity')
                        assert all(np.allclose(z[js],z[js[0]],rtol=0,atol=0,equal_nan=True) for z in f0),(unit,fr,'ambiguous oracle state')
                        idx.append(js[0])
                    return tuple(z[np.asarray(idx,int)] for z in f0)
                KL.field_crop_g=field
                try:s,c,g,k,conv=KL.run_g(cheap,v,cam,W,H,detect,lift,PP)
                finally:KL.field_crop_g=original
                histories.append(dict(field_calls=field_log,requests=requests,crops=canon(c),grown=canon(g),rounds=k,converged=conv,
                    action=O.act(s['mono'],v),scene_boxes=canon(s['xyxy']),scene_mono=canon(s['mono'])))
            equal=json.dumps(histories[0],sort_keys=True)==json.dumps(histories[1],sort_keys=True)
            trace.append(dict(unit=unit,frame=fr,v=v,states=records,oracle_equal=equal,oracle_reference=histories[0],oracle_tube=histories[1]))
        with gzip.open(target,'wt') as fh:json.dump(trace,fh)
        print(ds,unit,len(trace),'oracle mismatches',sum(not t['oracle_equal'] for t in trace),flush=True)
    # Summaries and frame records include existing completed units if resumed.
    for p in sorted(od.glob('*.json.gz')):
        for t in json.load(gzip.open(p,'rt')):
            for z in t['states']:checks.append(dict(dataset=name,unit=t['unit'],frame=t['frame'],v=t['v'],oracle_equal=t['oracle_equal'],**z))
    C=pd.DataFrame(checks);C.to_csv(OUT/f'k40_4_7_frames_{ds}.csv',index=False)
    C.groupby('stage').agg(frames=('equal','size'),exact_matches=('equal','sum'),literal_matches=('literal_center_equal','sum'),projection_matches=('projection_equal','sum'),oracle_matches=('oracle_equal','sum')).reset_index().to_csv(OUT/f'k40_4_7_summary_{ds}.csv',index=False)
    # Full CPU geometric grid sweep, vectorised over all saved speeds and draft actions.
    grid=[];vs=base.v.to_numpy(float);aa=base.a_cheap.to_numpy(int)
    for rm,lm,(pn,w,h),vc in itertools.product((.5,1,2),(.5,1,2),(('ped',.6,1.8),('car',1.8,1.6),('truck',2.6,3.6)),(0.,5.,10.)):
        rg=np.sort(np.r_[RG,(RG[:-1]+RG[1:])/2]) if rm==.5 else (RG if rm==1 else RG[[0,2,4,6,7]])
        yg=np.arange(-6,6+1e-9,1.5*lm);t0=time.perf_counter();cells=[]
        for v,a in zip(vs,aa):cells.append(int(reference(v,a,rg,yg,w,vc).sum()))
        elapsed=time.perf_counter()-t0
        grid.append(dict(dataset=name,range_factor=rm,lateral_factor=lm,probe=pn,v_close=vc,frames=len(base),cells=len(rg)*len(yg),
            mean_sensitive_cells=float(np.mean(cells)),nonempty_share=float(np.mean(np.array(cells)>0)),cpu_total_s=elapsed,cpu_mean_us=elapsed/len(base)*1e6,
            measurement='CPU field only; VUW; no detector; no G/pixels claim'))
    pd.DataFrame(grid).to_csv(OUT/f'k41_3_grid_cpu_{ds}.csv',index=False)
    # Numeric boundaries: immediate floating neighbours around the analytical speed crossing of every grid range.
    boundary=[]
    for a,rr in itertools.product((0,1),RG):
        tau=(1.,3.5)[a];v=-.4*tau+np.sqrt((.4*tau)**2+2*tau*(rr-2))
        for label,s in [('below',np.nextafter(v,-np.inf)),('at',v),('above',np.nextafter(v,np.inf))]:
            boundary.append(dict(action=a,range=rr,edge=label,v=s,equal=bool(np.array_equal(reference(s,a),mask(s,a))),
                reference=bits(reference(s,a)),tube=bits(mask(s,a))))
    pd.DataFrame(boundary).to_csv(OUT/f'boundary_{ds}.csv',index=False)
    pd.DataFrame(manifest).to_csv(OUT/f'sources_{ds}.csv',index=False)
    (OUT/f'run_{ds}.json').write_text(json.dumps(dict(md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),max_units=args.max_units,
        oracle='Frozen cached target states; no image inference, no new C1 histories. Exact representation check only.'),indent=2)+'\n')

if __name__=='__main__':main()
