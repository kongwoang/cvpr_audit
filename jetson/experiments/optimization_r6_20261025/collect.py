"""Recover current proposals from exactly the parent's stored crop canvases."""
import argparse,importlib.util,math,os,pickle,sys,time
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
from common import HERE,REMOTE,save,sha
from rap.detect import Mode,TwoFidelityDetector
from rap.mono import predicted_geometry,box_iou

PREFIX='v6S_b2A_missing_extent_retro'
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--shard',default='0/1');a=p.parse_args()
    ds=a.dataset;si,sn=map(int,a.shard.split('/'));B=Path(os.environ['B'])
    spec=importlib.util.spec_from_file_location('k27_r6collect',B/'scripts/A_method/k27_dev.py');K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
    torch.set_num_threads(4);scale=.6 if ds=='nusc' else 960/2048
    assert ds!='av2' or os.environ.get('K10_MASK_Y0')=='1830'
    src=REMOTE/'experiments/optimization_r3_20261023/results/low'/ds
    expected=sorted(p.stem for p in src.glob('*.pkl'));assert len(expected)==dict(nusc=152,av2=300)[ds]
    allowed=set(expected[si::sn]);out=HERE/'proposals'/ds;out.mkdir(parents=True,exist_ok=True)
    det=TwoFidelityDetector(str(B/'models/yolov8s.pt'),device='cuda:0',backend='torch')
    seen=[];total_calls=0;total_px=0;start=time.time()
    for unit,cam,gen in K.units(ds):
        if unit not in allowed:continue
        fo=out/(unit+'.pkl');seen.append(unit)
        if fo.exists() and fo.with_suffix('.json').exists():continue
        rows=pickle.load((src/(unit+'.pkl')).open('rb'));by={int(r['frame']):r for r in rows}
        R,t=cam.transform('ego');result=[];audits=[];calls=px=0;unitstart=time.time()
        for fr,path,v,g,cached in gen():
            if fr not in by:continue
            r=by[fr];trace=r['trace_'+PREFIX];crops=trace['crops']
            img=cv2.imread(path) if len(crops) else None
            if len(crops) and img is None:raise RuntimeError((unit,fr,path))
            h,w=(img.shape[:2] if img is not None else ((900,1600) if ds=='nusc' else (2048,1550)))
            keys=[];boxes=[];conf=[];classes=[];source=[]
            for cr in crops:
                x1,y1=int(cr[0]),int(cr[1]);x2,y2=max(int(math.ceil(cr[2])),x1+8),max(int(math.ceil(cr[3])),y1+8)
                nw=max(32,int(math.ceil((min(x2,w)-x1)*scale/32))*32);nh=max(32,int(math.ceil((min(y2,h)-y1)*scale/32))*32)
                key=(x1,y1,x2,y2,nh,nw)
                if key in keys:continue
                keys.append(key);d=det.detect(img[y1:y2,x1:x2],Mode('c%dx%d'%(nh,nw),nh,nw))[0]
                boxes.append(d['xyxy'].astype(float)+np.array([x1,y1,x1,y1]));conf.append(d['conf']);classes.append(d['coarse']);source.extend([len(keys)-1]*len(d['conf']))
                calls+=1;px+=nh*nw
            b=np.concatenate(boxes) if boxes else np.zeros((0,4));c=np.concatenate(conf) if conf else np.zeros(0);cl=np.concatenate(classes) if classes else np.zeros(0,str)
            # Audit against pre-merge proposals, preserving all detector class IDs.
            full=np.flatnonzero(trace['from_full']);high=c>=K.IE.THR
            for j in full:
                ix=np.flatnonzero(high&(cl==trace['coarse'][j]))
                if len(ix):
                    errors=np.abs(b[ix]-trace['xyxy'][j]).max(1);best=ix[np.argmin(errors)];error=float(errors.min());ce=abs(float(c[best])-float(trace['conf'][j]))
                else:error=ce=float('inf')
                audits.append(dict(frame=fr,index=int(j),max_abs_box_error=error,confidence_error=ce))
            if len(b):
                ids=np.array([dict(vehicle=0,person=1,cyclist=2)[str(x)] for x in cl])
                ix=torchvision.ops.batched_nms(torch.as_tensor(b,dtype=torch.float32),torch.as_tensor(c,dtype=torch.float32),torch.as_tensor(ids),.65).numpy()
                b,c,cl=b[ix],c[ix],cl[ix];source_arr=np.asarray(source)[ix]
            else:source_arr=np.zeros(0,int)
            mono=predicted_geometry(dict(xyxy=b,coarse=cl),None,cam.calib,cam.cam_h,cam.dt,cam_to_ego=(R,t))
            result.append(dict(unit=unit,frame=fr,xyxy=b,conf=c,coarse=cl,source_crop=source_arr,crop_keys=keys,
                mono=tuple(np.asarray(mono[k],float) for k in ('z','lat_min','lat_max','ttc')),threshold=float(K.IE.THR)))
        assert [r['frame'] for r in result]==[r['frame'] for r in rows]
        bad=[x for x in audits if x['max_abs_box_error']>.05 or x['confidence_error']>.002]
        # Persist diagnostics even if equivalence fails; never hide imperfect replay.
        save(fo.with_suffix('.audit.json'),dict(full_boxes=len(audits),bad=bad,max_abs_box_error=max([x['max_abs_box_error'] for x in audits]+[0]),max_conf_error=max([x['confidence_error'] for x in audits]+[0])))
        if bad:raise RuntimeError(('high proposal replay mismatch',unit,len(bad),bad[:3]))
        tmp=fo.with_suffix('.tmp');pickle.dump(result,tmp.open('wb'));tmp.replace(fo)
        meta=dict(dataset=ds,unit=unit,frames=len(rows),proposals=sum(len(r['conf']) for r in result),low_proposals=sum(int((r['conf']<r['threshold']).sum()) for r in result),offline_calls=calls,offline_pixels=px,seconds=time.time()-unitstart,source_sha256=sha(src/(unit+'.pkl')),high_replay_bad=0)
        save(fo.with_suffix('.json'),meta);print(meta,flush=True);total_calls+=calls;total_px+=px
    assert set(seen)==allowed
    save(out/('DONE_'+a.shard.replace('/','of')+'.json'),dict(status='complete',units=sorted(seen),seconds=time.time()-start,offline_calls_this_attempt=total_calls,offline_pixels_this_attempt=total_px))
if __name__=='__main__':main()
