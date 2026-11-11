"""Fresh training-only evidence with explicit identity and availability labels.

Uses the original training acquisition policy, not any development annotations.
Outputs atomic per-unit files so interrupted runs can be resumed without lost units.
"""
import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path
import time
import numpy as np
import cv2

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE,REMOTE,save,sha
B=Path(os.environ['B'])
spec=importlib.util.spec_from_file_location('original_tdata',B/'scripts/A_method/k28_tdata.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
IE,KL,K27=M.IE,M.KL,M.K27
from rap import planner as P
from rap.mono import predicted_geometry,box_iou
from rap.detect import Mode,TwoFidelityDetector
sys.path.insert(0,str(B/'scripts/common'))
from nusc_refiner_crop import crop64

def label(d,g):
    n=len(d['xyxy']);ids=np.full(n,-1,int);gi=np.full(n,-1,int)
    if len(g) and n:
        b=np.stack([g[k] for k in ('x1','y1','x2','y2')],1)
        mm=IE.matched(b,np.asarray(g['coarse']),d['xyxy'],d['coarse'])
        for j,i in enumerate(mm):
            if i>=0:ids[i]=int(g['track_id'][j]);gi[i]=j
    return ids,gi

def assoc(x,c,prev):
    out=np.full(len(x),-1,int)
    if prev is None or not len(x) or not len(prev['xyxy']):return out
    scores=box_iou(x,prev['xyxy']);scores[np.asarray(c)[:,None]!=prev['coarse'][None,:]]=-1
    while scores.size and scores.max()>=.3:
        i,j=np.unravel_index(scores.argmax(),scores.shape);out[i]=j;scores[i,:]=-1;scores[:,j]=-1
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['nusc','av2'],required=True)
    ap.add_argument('--shard',default='0/1');ap.add_argument('--max-units',type=int,default=0)
    a=ap.parse_args();si,sn=map(int,a.shard.split('/'))
    import torch
    torch.set_num_threads(4)
    out=HERE/'training_data'/a.dataset;out.mkdir(parents=True,exist_ok=True)
    det=TwoFidelityDetector(str(B/'models/yolov8s.pt'),device='cuda:0',backend='torch')
    pp=P.PlannerParams();pol=K27.parse('T4+T2_1')
    if a.dataset=='nusc':CH,SCALE,F,MS=None,.6,Mode('f576x960',576,960),.6
    else:CH,SCALE,F,MS=Mode('av2_c320',320,256),960/2048,Mode('av2_f960',960,768),960/2048
    expected=[]
    for ui,(unit,cam,gen,split,every) in enumerate(M.units(a.dataset,a.max_units)):
        if ui%sn!=si:continue
        expected.append(unit);fo=out/(unit+'.npz')
        if fo.exists() and fo.with_suffix('.json').exists():continue
        start=time.time();R,t=cam.transform('ego');dt=float(cam.dt)
        st={};history={};imgs=[];prev_c=None;acc={}
        def put(**row):
            for k,vv in row.items():acc.setdefault(k,[]).append(vv)
        for fr,path,v,g,cached in gen():
            img=cv2.imread(path)
            if img is None:raise RuntimeError(('image missing',unit,fr,path))
            H,W=img.shape[:2]
            def geom(d,pv=None):
                z=predicted_geometry({'xyxy':d['xyxy'],'coarse':d['coarse']},pv,cam.calib,cam.cam_h,dt,cam_to_ego=(R,t))
                return tuple(np.asarray(z[k],float) for k in ('z','lat_min','lat_max','ttc'))
            def keep(d):
                k=d['conf']>=IE.THR
                return dict(xyxy=d['xyxy'][k].astype(float),conf=d['conf'][k].astype(float),coarse=d['coarse'][k])
            def detect_crop(cr):
                x1,y1=int(cr[0]),int(cr[1]);x2,y2=max(int(math.ceil(cr[2])),x1+8),max(int(math.ceil(cr[3])),y1+8)
                nw=max(32,int(math.ceil((min(x2,W)-x1)*SCALE/32))*32);nh=max(32,int(math.ceil((min(y2,H)-y1)*SCALE/32))*32)
                d=keep(det.detect(img[y1:y2,x1:x2],Mode('c%dx%d'%(nh,nw),nh,nw))[0])
                d['xyxy']+=np.array([x1,y1,x1,y1]);return d
            def lift(full):
                z,lo,hi,_=geom(full);h=st.get('crop',{})
                tt,_=K27.ttc_ss(full['xyxy'],h.get('s1'),h.get('s2'),SCALE,dt,pol['gate'],pol['t4'])
                return z,lo,hi,tt
            if cached is not None:ch=cached
            else:
                ch=keep(det.detect(img,CH)[0]);ch['mono']=geom(ch,prev_c)
                prev_c={k:ch[k] for k in ('xyxy','coarse')}
            base={k:ch[k] for k in ('xyxy','conf','coarse','mono')}
            sc,*_=KL.run_g(base,float(v),cam,W,H,detect_crop,lift,pp,v_closes=(0.,))
            ff=np.asarray(sc['from_full'],bool)
            crop={k:np.asarray(sc[k])[ff] for k in ('xyxy','conf','coarse')};crop['mono']=tuple(np.asarray(x)[ff] for x in sc['mono'])
            d9=keep(det.detect(img,F)[0]);d9['mono']=geom(d9)
            now={'crop':crop,'u960':d9,'cheap':base}
            for d in now.values():d['ids'],d['gi']=label(d,g)
            if fr%every==0:
                for kind,d in now.items():
                    z,lo,hi,_=d['mono'];X=np.asarray(d['xyxy'])
                    idx=np.flatnonzero(np.isfinite(z)&(z>0)&(z<=80)&(hi>-3.2)&(lo<3.2))
                    if not len(idx):continue
                    zz=predicted_geometry({'xyxy':X[idx],'coarse':np.asarray(d['coarse'])[idx]},None,cam.calib,cam.cam_h,dt,cam_to_ego=(R,t))
                    modes=[(kind,kind)]+([('bridge','cheap')] if kind=='crop' else [])
                    for outputkind,hkind in modes:
                        hh=history.get(hkind,[]);p1=hh[-1] if len(hh)>0 else None;p2=hh[-2] if len(hh)>1 else None
                        # Same-fidelity uses the original class-agnostic greedy association;
                        # bridge matching is class-aware, with a declared IoU .3 threshold.
                        if outputkind=='bridge':
                            j1=assoc(X,d['coarse'],p1);j12=assoc(p1['xyxy'],p1['coarse'],p2) if p1 is not None else []
                        else:
                            j1=K27.assoc_idx(X,p1['xyxy']) if p1 is not None else np.full(len(X),-1,int)
                            j12=K27.assoc_idx(p1['xyxy'],p2['xyxy']) if p1 is not None and p2 is not None else np.full(len(p1['xyxy']) if p1 else 0,-1,int)
                        for n,i in enumerate(idx):
                            gi=int(d['gi'][i]);target=np.full(4,np.nan)
                            if gi>=0:
                                target=np.array([g[k][gi] for k in ('long_near','lat_min','lat_max','ttc')],float)
                            rate=0. if np.isinf(target[3]) else (1/max(target[3],.5) if np.isfinite(target[3]) and target[3]>0 else np.nan)
                            # Match original temporal target, including actual range_rate and has_rate.
                            if gi>=0:
                                has=bool(g['has_rate'][gi]) if 'has_rate' in g.dtype.names else np.isfinite(g['range_rate'][gi])
                                rate=max(-float(g['range_rate'][gi]),0)/max(float(g['long_near'][gi]),.5) if has else np.nan
                            boxes=np.zeros((3,4),float);boxes[0]=X[i];mask=np.array([1,0,0],np.float32)
                            ids=np.array([d['ids'][i],-1,-1],int);cr=np.zeros((3,64,64,3),np.uint8);cr[0]=crop64(img,X[i])
                            j=int(j1[i]);k=int(j12[j]) if j>=0 else -1
                            if j>=0:
                                boxes[1]=p1['xyxy'][j];mask[1]=1;ids[1]=p1['ids'][j];cr[1]=crop64(imgs[-1],boxes[1])
                            if k>=0:
                                boxes[2]=p2['xyxy'][k];mask[2]=1;ids[2]=p2['ids'][k];cr[2]=crop64(imgs[-2],boxes[2])
                            heights=np.maximum(boxes[:,3]-boxes[:,1],1e-3)
                            hr=np.ones(2);hr[0]=heights[0]/heights[1] if mask[1] else 1;hr[1]=heights[1]/heights[2] if mask[2] else 1
                            x1,y1,x2,y2=X[i];zg,zh=zz['z_ground'][n],zz['z_height'][n]
                            feats=np.array([x1/W,y1/H,x2/W,y2/H,(x2-x1)/W,(y2-y1)/H,d['conf'][i],M.CLS[str(d['coarse'][i])],z[i],zg,zh,np.log(max(zg,.5)/max(zh,.5)),*np.log(np.clip(hr,1e-3,None)),dt,1],np.float32)
                            known=(mask.astype(bool))&(ids>=0);validity=np.where((ids>=0)&(ids[0]>=0),(ids==ids[0]).astype(float),np.nan)
                            put(crops=cr,feats=feats,mask=mask,ids=ids,identity=validity,boxes=boxes,frame=fr,kind=outputkind,
                                split=split,unit=unit,target=rate,gt_mono=target,raw_mono=np.array([x[i] for x in d['mono']]),v=v)
            for name,d in now.items():
                h=st.setdefault(name,{});h['s2']=h.get('s1');h['s1']=np.asarray(d['xyxy']).copy()
                history[name]=(history.get(name,[])+[{k:np.asarray(d[k]).copy() for k in ('xyxy','coarse','ids')}])[-2:]
            imgs=(imgs+[img])[-2:]
        arrays={k:np.asarray(vv) for k,vv in acc.items()}
        if not arrays:arrays={'frame':np.zeros(0,int)}
        tmp=fo.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(fo)
        meta=dict(dataset=a.dataset,unit=unit,split=split,samples=len(arrays['frame']),seconds=time.time()-start,sha256=sha(fo))
        save(fo.with_suffix('.json'),meta);print(meta,flush=True)
    save(out/('DONE_'+a.shard.replace('/','of')+'.json'),dict(status='complete',units=expected))

if __name__=='__main__':main()
