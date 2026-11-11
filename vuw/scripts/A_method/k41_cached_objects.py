#!/usr/bin/env python3
"""CPU-only K40-1(c), K41-3/4 extraction from original K30 records.

No model inference. Restore pre-refiner spatial geometry with the original lift;
TTC is unchanged by v3a. Retained cheap detections use their stored cheap state.
Every reconstructed action must match the corresponding release row before use.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rap import planner as P
from rap.mono import box_iou, predicted_geometry
from vtf import speculative as SP

B=Path(os.environ['B']); R=B/'tmp/desp_k40/artifacts/records'
OUT=B/'results/k41/objects_v4'; PP=P.PlannerParams(); AV2_MASK=False
NAMES={'nusc':'nuscenes','av2':'av2','ahd':'argoverse_hd'}
SIZES={'nusc':(1600,900),'av2':(1550,2048),'ahd':(1920,1200)}

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def load(path, manifest):
    assert 'holdout' not in str(path).lower()
    raw=path.read_bytes(); manifest.append(dict(path=str(path),md5=hashlib.md5(raw).hexdigest(),bytes=len(raw)))
    return pickle.loads(raw)

def match(a,b,threshold=.5,classes=True):
    n,m=len(a['xyxy']),len(b['xyxy']); out=np.full(n,-1,int)
    if not n or not m:return out
    iou=box_iou(a['xyxy'],b['xyxy'])
    eligible=iou>=threshold
    if classes:eligible&=np.asarray(a['coarse'])[:,None]==np.asarray(b['coarse'])[None,:]
    edges=sorted(((-float(iou[i,j]),int(i),int(j)) for i,j in zip(*np.where(eligible))))
    used=set()
    for _,i,j in edges:
        if out[i]<0 and j not in used:out[i]=j;used.add(j)
    return out

def act(geo,v):return P.discrete_action(P.required_decel(*geo,v,PP)[0],PP)
def demands(geo,v):return P.required_decel(*geo,v,PP)[1]

def spatial(sc,cam):
    g=predicted_geometry({'xyxy':sc['xyxy'],'coarse':sc['coarse']},None,
       cam.calib,cam.cam_h,float(cam.dt),cam_to_ego=cam.transform('ego'))
    values=[np.asarray(g[k],float).copy() for k in ('z','lat_min','lat_max')]
    if AV2_MASK:
        b=np.asarray(sc['xyxy'],float);height=np.maximum(b[:,3]-b[:,1],1e-6)
        masked=np.maximum(b[:,3]-np.maximum(b[:,1],1830.),0)/height>=.5
        values[0][masked]=np.inf  # exact k10_masked_run contract; lateral coordinates unchanged
    return tuple(values)

def no_ref(sc,cam,cheap,crops,whole=False,grown=False,v=None):
    geo=list(spatial(sc,cam))+[np.asarray(sc['geo'][3],float).copy()]
    if not whole:
        # Exact inverse of k_loop.detect_merge: cheap[~out] is the prefix, followed by full.
        # Guard regions remove cheap objects through IoU, not centre membership in the grown union.
        candidates=[]
        for n in range(min(len(sc['xyxy']),len(cheap['xyxy']))+1):
            out=SP.centres_in(np.asarray(cheap['xyxy']),crops)
            full=np.asarray(sc['xyxy'])[n:]
            if grown and len(full) and len(cheap['xyxy']):out|=box_iou(cheap['xyxy'],full).max(1)>=.5
            ids=np.flatnonzero(~out)
            if len(ids)!=n:continue
            if all(np.array_equal(np.asarray(sc[k])[:n],np.asarray(cheap[k])[ids]) for k in ('xyxy','coarse','conf')):
                candidates.append(ids)
        restored=[];prefixes=[]
        for ids in candidates:
            cg=[z.copy() for z in geo];n=len(ids)
            for i,j in enumerate(ids):
                for k in range(4):cg[k][i]=cheap['geo'][k][j]
            # Spatial refiner only scales full-support z/lateral, preserving TTC.
            support=set(SP.support(tuple(cg),v,PP)) if v is not None else set(range(n,len(sc['xyxy'])))
            okay=True
            for i in range(len(sc['xyxy'])):
                before=np.array([cg[k][i] for k in range(3)])
                after=np.array([sc['geo'][k][i] for k in range(3)])
                if i<n or i not in support:
                    okay &= bool(np.allclose(before,after,rtol=0,atol=1e-9,equal_nan=True))
                elif np.isfinite(before).all() and np.isfinite(after).all():
                    c=after[0]/before[0] if before[0]!=0 else np.nan
                    okay &= bool(c>0 and np.allclose(before*c,after,rtol=1e-10,atol=1e-9))
            if okay:restored.append(cg);prefixes.append(n)
        if not restored:return None
        if any(not all(np.allclose(a,b,rtol=0,atol=1e-9,equal_nan=True) for a,b in zip(restored[0],candidate)) for candidate in restored[1:]):return None
        geo=restored[0]
        sc['_prefix_min'],sc['_prefix_max']=min(prefixes),max(prefixes)
    return tuple(geo)

def state_diff(a,b,i,j):
    ra,rb=float(a[0][i]),float(b[0][j]); ta,tb=float(a[3][i]),float(b[3][j])
    return abs(ra-rb)>.01*max(abs(ra),1e-9) or np.isfinite(ta)!=np.isfinite(tb) or (np.isfinite(ta) and np.isfinite(tb) and abs(ta-tb)>.01*max(abs(ta),1e-9))

def main():
    global AV2_MASK
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=NAMES,required=True);ap.add_argument('--max-units',type=int,default=0)
    args=ap.parse_args();ds=args.dataset;AV2_MASK=ds=='av2';name=NAMES[ds];od=OUT/ds;od.mkdir(parents=True,exist_ok=True)
    base=pd.read_parquet(R/(name+'.parquet')).set_index(['unit','frame']); allowed=set(base.index.get_level_values(0))
    K24=module('k41_k24',B/'scripts/A_method/k24_test.py')
    manifest=[]; checks=[]; counts=0
    for unit,cam,gen in K24.units(ds):
        if unit not in allowed:continue
        if args.max_units and counts>=args.max_units:break
        counts+=1
        target=od/(unit+'.pkl')
        if target.exists():
            old=pickle.loads(target.read_bytes());checks.extend(old['checks']);manifest.extend(old['sources']);continue
        src=B/'results/k30/apdump'/ds/(unit+'.pkl'); cropfile=B/'results/k30/match'/ds/(unit+'.pkl')
        apd=load(src,manifest); crd={int(r['frame']):r for r in load(cropfile,manifest)}
        rows={int(r['frame']):r for r in apd['rows']}; dets=apd['dets']; frout=[]; objects=[]; taxonomy=[]
        for fr,path,v,g,cached in gen():
            if (unit,fr) not in base.index:continue
            rr=base.loc[(unit,fr)]; r=rows[fr]; cr=crd[fr]; crops=list(cr['crops_v6S']);nc=int(cr['ncrops_v6S'])
            ch=dets[(fr,'cheap')]['sc']; f=dets[(fr,'wf960+C1')]['sc']; o=dets[(fr,'v6S')]['sc']
            f0=no_ref(f,cam,ch,crops,True);o0=no_ref(o,cam,ch,crops[:nc],grown=len(crops)>nc,v=v)
            if o0 is None:
                checks.append(dict(dataset=name,unit=unit,frame=fr,reconstruction_ok=False,reason='inverse merge has distinct possible pre-refiner states'))
                continue
            af,ao=act(f0,v),act(o0,v)
            ck=dict(dataset=name,unit=unit,frame=fr,reconstruction_ok=True,cheap_ok=act(ch['geo'],v)==int(rr.a_cheap),
                target_noref_ok=af==int(rr['a_wf960+C1|none']),desp_noref_ok=ao==int(rr['a_v6S|none']),
                target_ref_ok=act(f['geo'],v)==int(rr['a_wf960+C1|v3a']),desp_ref_ok=act(o['geo'],v)==int(rr['a_v6S|v3a']),
                match_desp_ok=int(cr['a_v6S|none'])==int(rr['a_v6S|none']),
                crops_px_ok=abs(float(cr['px_v6S'])-float(rr.px_v6S))<1e-9)
            checks.append(ck)
            if not all(ck[k] for k in ck if k.endswith('_ok')):
                continue  # retain failed cases in parity CSV, never force them into an attribution
            fd,odm=demands(f0,v),demands(o0,v)
            fi=int(np.argmax(fd)) if len(fd) and fd.max()>0 else -1
            oi=int(np.argmax(odm)) if len(odm) and odm.max()>0 else -1
            fm=match(f,o);om=match(o,f); ca=match(f,o,classes=False)
            fcov=SP.centres_in(np.asarray(f['xyxy']),crops)
            # Locality requires an actual crop detection. The merged scene can retain cheap boxes.
            prefix_max=o['_prefix_max'];prefix_min=o['_prefix_min']
            oc={k:np.asarray(o[k])[prefix_max:] for k in ('xyxy','coarse','conf')}
            ocgeo=tuple(z[prefix_max:] for z in o0)
            gt={'xyxy':np.stack([g[k] for k in ('x1','y1','x2','y2')],axis=1) if len(g) else np.zeros((0,4)),
                'coarse':np.asarray(g['coarse']) if len(g) else np.zeros(0,str)}
            # Annotation matching follows K30's person/cyclist compatibility and Hungarian assignment.
            fg=K24.IE.matched(gt['xyxy'],gt['coarse'],f['xyxy'],f['coarse'])
            og=K24.IE.matched(gt['xyxy'],gt['coarse'],o['xyxy'],o['coarse'])
            frmeta=dict(dataset=name,unit=unit,frame=fr,v=v,target_count=len(fd),
              corridor_count=int(P.in_corridor(f0[1],f0[2],PP.corridor_half_w).sum()),
              support_size=len(SP.support(ch['geo'],v,PP)),
              decisive_height=float(f['xyxy'][fi][3]-f['xyxy'][fi][1]) if fi>=0 else np.nan,
              decisive_range=float(f0[0][fi]) if fi>=0 else np.nan,
              decisive_closing=float(f0[0][fi]/f0[3][fi]) if fi>=0 and np.isfinite(f0[3][fi]) and f0[3][fi]>0 else (0. if fi>=0 else np.nan),
              decisive_covered=bool(fcov[fi]) if fi>=0 else np.nan,
              target_action=af,desp_action=ao,draft_action=int(rr.a_cheap),
              rounds=int(cr['rounds_v6S']),ncrops=nc,nguard=int(cr['nguard_v6S']),
              target_relevant=int((fd>=PP.thr_decel).sum()),covered_relevant=int(fcov[fd>=PP.thr_decel].sum()))
            viol=[]
            for t in (.5,.7):
                mt=match(f,oc,t); at=match(f,oc,t,False)
                tv=np.zeros(len(fd),bool)
                for i in range(len(fd)):
                    j=int(mt[i]);k=int(at[i]); box=np.asarray(f['xyxy'][i]); h=max(box[3]-box[1],1e-9);cx,cy=(box[:2]+box[2:])/2
                    containing=[(q,c) for q,c in enumerate(crops) if c[0]<=cx<=c[2] and c[1]<=cy<=c[3]]
                    edge=max((min(cx-c[0],c[2]-cx,cy-c[1],c[3]-cy)/h for _,c in containing),default=np.nan)
                    guard=any(q>=nc for q,c in containing)
                    good=j>=0
                    tv[i]=good and not state_diff(f0,ocgeo,i,j)
                    fin=good and np.isfinite(f0[3][i]) and np.isfinite(ocgeo[3][j])
                    objects.append(dict(dataset=name,unit=unit,frame=fr,obj=i,iou_threshold=t,
                      class_name=str(f['coarse'][i]),height=h,range=float(f0[0][i]),ttc=float(f0[3][i]),
                      closing=float(f0[0][i]/f0[3][i]) if np.isfinite(f0[3][i]) and f0[3][i]>0 else 0.,
                      demand=float(fd[i]),relevant=bool(fd[i]>=PP.thr_decel),decisive=i==fi,covered=bool(fcov[i]),
                      matched=good,range_abs_error=abs(f0[0][i]-ocgeo[0][j]) if good else np.nan,
                      range_rel_error=abs(f0[0][i]-ocgeo[0][j])/max(abs(f0[0][i]),1e-9) if good else np.nan,
                      ttc_abs_error=abs(f0[3][i]-ocgeo[3][j]) if fin else np.nan,
                      ttc_rel_error=abs(f0[3][i]-ocgeo[3][j])/max(abs(f0[3][i]),1e-9) if fin else np.nan,
                      ttc_finite_mismatch=(np.isfinite(f0[3][i])!=np.isfinite(ocgeo[3][j])) if good else np.nan,
                      class_mismatch=(f['coarse'][i]!=oc['coarse'][k]) if k>=0 else np.nan,
                      crop_identity_ambiguous=prefix_max!=prefix_min,
                      edge_ratio=edge,containing_guard=guard,locality_state_ok=tv[i],draft_action=int(rr.a_cheap),
                      final_rounds=int(cr['rounds_v6S']),origin_round='unavailable',origin_canvas='unavailable'))
                decisive_bad=fi>=0 and fcov[fi] and not tv[fi]
                sup=SP.support(o0,v,PP)
                oms=match(o,f,t)
                support_bad=any(oms[j]<0 or state_diff(o0,f0,j,oms[j]) for j in sup if j>=prefix_max)
                frmeta[f'locality_violation_{t}']=bool(decisive_bad or support_bad)
            frmeta['coverage_miss']=bool(fi>=0 and not fcov[fi])
            frout.append(frmeta)
            for setting,ogeo,oa in [('no_refiner',o0,ao),('full_DESP_vs_noref_target',o['geo'],int(rr['a_v6S|v3a']))]:
                if oa==af:continue
                dd=demands(ogeo,v);di=int(np.argmax(dd)) if len(dd) and dd.max()>0 else -1
                outside=fi>=0 and fm[fi]<0 and not fcov[fi]
                state=fi>=0 and fm[fi]>=0 and state_diff(f0,ogeo,fi,fm[fi])
                extra=di>=0 and om[di]<0
                cat='i' if outside else 'ii' if state else 'iii' if extra else 'iv'
                taxonomy.append(dict(dataset=name,unit=unit,frame=fr,setting=setting,category=cat,
                    target_missing_outside=outside,matched_state_difference=state,desp_only_decisive=extra,
                    target_action=af,desp_action=oa,required_action=int(rr.act_gt),
                    target_correct=af==int(rr.act_gt),desp_correct=oa==int(rr.act_gt),
                    target_decisive_unmatched_to_annotation=bool(fi>=0 and fi not in set(fg)),
                    desp_decisive_unmatched_to_annotation=bool(di>=0 and di not in set(og)),
                    history_provenance='TTC preserved; per-object history identities not saved',
                    loss_advantage=float(rr['J_wf960+C1|none']-rr['J_v6S|'+('none' if setting=='no_refiner' else 'v3a')]),
                    cheap_loss=float(rr.J_cheap)))
        with target.open('wb') as fh:pickle.dump(dict(frames=frout,objects=objects,taxonomy=taxonomy,
            checks=[c for c in checks if c['unit']==unit],sources=manifest[-2:]),fh,pickle.HIGHEST_PROTOCOL)
        print(ds,unit,'frames',len(frout),'objects',len(objects),'parity failures',sum(not all(c[k] for k in c if k.endswith('_ok')) for c in checks if c['unit']==unit),flush=True)
    pd.DataFrame(checks).to_csv(OUT/f'parity_{ds}.csv',index=False)
    pd.DataFrame(manifest).to_csv(OUT/f'sources_{ds}.csv',index=False)
    (OUT/f'run_{ds}.json').write_text(json.dumps(dict(script=str(Path(__file__)),md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),max_units=args.max_units),indent=2)+'\n')

if __name__=='__main__':main()
