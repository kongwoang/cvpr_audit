"""Check deployed features/predictions against cached training examples, not GT."""
import argparse
import importlib.util
import sys
import numpy as np
import cv2
import torch
from common import HERE,REMOTE,save
from inference import Runtime
from learned import decode

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--shard',default='0/1');a=ap.parse_args()
    torch.set_num_threads(4);sys.path.insert(0,str(REMOTE/'scripts/common'))
    from nusc_refiner_crop import crop64
    M=module('base_c1',REMOTE/'scripts/A_method/k28_ttrain.py');K=module('k27',REMOTE/'scripts/A_method/k27_dev.py')
    ck=torch.load(REMOTE/'out/k28_temporal/temporal_head.pt',map_location='cpu',weights_only=False)
    encoder=M.TempHead(16);encoder.load_state_dict(ck['state']);encoder=encoder.cuda().eval()
    runtime=Runtime(a.dataset,encoder,crop64,K.assoc_idx);checks=[]
    chosen=None
    for f in sorted((HERE/'embeddings'/a.dataset).glob('*.npz')):
        z=np.load(f,allow_pickle=False)
        if len(z['frame']) and (z['split']=='val').all() and len(z['frame'])>=50:chosen=f;break
        z.close()
    assert chosen is not None
    unit=chosen.stem
    if a.dataset=='nusc':
        from rap.nusc import NuScenesDB
        from rap.paths import NUSCENES_TRAINVAL
        db=NuScenesDB(NUSCENES_TRAINVAL,'v1.0-trainval');token=next(t for t in db.scenes if db.scene_name(t)==unit);tokens=db.samples(token)
        path=lambda f:db.image_path(tokens[f])
    else:
        from vtf.av2 import AV2Log
        cam=AV2Log(REMOTE/'datasets/av2/sensor/train',unit);path=cam.image_path
    data={k:z[k] for k in z.files};z.close();images={}
    for i in range(len(data['frame'])):
        kind=str(data['kind'][i]);mask=data['mask'][i];fr=int(data['frame'][i]);b=data['boxes'][i]
        if kind=='u960' or (kind=='crop' and mask.sum()<3):continue
        if fr<2:continue
        runtime.frame(fr)
        for t in (fr-2,fr-1,fr):
            if t not in images:images[t]=cv2.imread(str(path(t)));assert images[t] is not None
        img=images[fr];imgs=[images[fr-2],images[fr-1]];H,W=img.shape[:2]
        fe=data['feats'][i];coarse=np.array([('vehicle','person','cyclist')[int(fe[7])]])
        full=dict(xyxy=b[:1],conf=np.array([fe[6]]),coarse=coarse)
        h=dict(s1=b[1:2] if mask[1] else np.zeros((0,4)),s2=b[2:3] if mask[2] else np.zeros((0,4)),
               c1=coarse if mask[1] else np.zeros(0,str),c2=coarse if mask[2] else np.zeros(0,str))
        chn=[(0,0,0)] if mask.sum()==3 and kind!='bridge' else []
        zz=dict(z_ground=np.array([fe[9]]),z_height=np.array([fe[10]]))
        raw=tuple(np.array([v]) for v in data['raw_mono'][i]);v=float(data['v'][i]);dt=float(fe[14])
        for objective in ('geometry','decision'):
            pred,actualmask=runtime.infer(objective,full,raw,zz,h,chn,h,imgs,img,W,H,v,dt,kind='cheap' if kind=='cheap' else 'crop')
            e=torch.from_numpy(data['embedding'][i:i+1].astype(np.float32)).cuda()
            f=torch.from_numpy(data['fe'][i:i+1]).cuda();mm=torch.from_numpy(mask[None,:]).cuda();ex=torch.from_numpy(data['extra'][i:i+1]).cuda()
            with torch.no_grad():
                r,c,q,valid=decode(runtime.nets[objective](e,f,mm,ex));reference=torch.cat([r[:,None],c[:,None],q,valid],1).cpu().numpy()
            assert np.array_equal(actualmask[0],mask),(unit,i,kind,actualmask,mask)
            delta=float(np.max(np.abs(pred-reference)));assert np.isfinite(delta) and delta<.005,(unit,i,kind,pred,reference,delta)
            checks.append(dict(objective=objective,kind=kind,frame=fr,index=i,max_abs_delta=delta))
        if len(checks)>=120:break
    out=HERE/'reports/runtime_check';out.mkdir(parents=True,exist_ok=True)
    save(out/(a.dataset+'.json'),dict(status='complete',unit=unit,checks=checks,max_abs_delta=max(x['max_abs_delta'] for x in checks),
        note='Training embeddings fp16 vs runtime fp32; feature/mask consistency checked on actual images.'))
    print(a.dataset,len(checks),max(x['max_abs_delta'] for x in checks),flush=True)
if __name__=='__main__':main()
