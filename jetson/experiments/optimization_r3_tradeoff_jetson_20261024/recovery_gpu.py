"""Real historical-image crop inference, with source preprocessing preserved."""
import math
import numpy as np
import torch
from vtf.gpuprep import PAD,letterbox
from vtf import gpupost as GP


class TracedCrops:
    def __init__(self,inner,gpu,scale,log,common=False):
        self.inner,self.gpu,self.scale,self.log,self.common=inner,gpu,scale,log,common

    def __call__(self,crops):
        out=self.inner(crops)
        keys=[]
        for crop in crops:
            x1,y1=int(crop[0]),int(crop[1])
            x2,y2=max(math.ceil(crop[2]),x1+8),max(math.ceil(crop[3]),y1+8)
            sub=self.gpu.x[:,y1:y2,x1:x2]
            h=min(1024,max(32,math.ceil(sub.shape[1]*self.scale/32)*32))
            w=min(1024,max(32,math.ceil(sub.shape[2]*self.scale/32)*32))
            keys.append((x1,y1,x2,y2,h,w))
        if self.common and keys:
            hc,wc=max(k[4] for k in keys),max(k[5] for k in keys)
        for det,key in zip(out,keys):
            canvas=(hc,wc) if self.common else key[-2:]
            for box in det['xyxy']:
                self.log[tuple(box)]=key+canvas
        return out


class Historical:
    def __init__(self,cpu_images,gpu_images,engine,mode,threshold,stats):
        self.cpu,self.gpu=cpu_images,gpu_images
        self.engine,self.Mode,self.threshold,self.stats=engine,mode,threshold,stats
        assert len(cpu_images)==len(gpu_images)
        self.provenance=[]

    def __call__(self,image,key):
        ids=[i for i,im in enumerate(self.cpu) if image is im]
        assert len(ids)==1 and len(self.gpu)==2, 'historical image provenance mismatch'
        self.provenance.append(ids[0])
        x1,y1,x2,y2,h,w,hc,wc=key
        sub=self.gpu[ids[0]][:,y1:y2,x1:x2]
        own,ratio,pad=letterbox(sub,h,w)
        if (h,w)==(hc,wc):
            inp=own[None]
        else:
            inp=torch.full((1,3,hc,wc),PAD,device=own.device,dtype=own.dtype)
            inp[0,:,:h,:w]=own
        raw=self.engine(inp)
        self.stats['calls']+=1
        mode=self.Mode(f'c{hc}x{wc}',hc,wc)
        columns=GP.host_columns(raw,mode,self.threshold)[0]
        det=GP.post_lean(columns,mode,ratio,pad,(y2-y1,x2-x1,3),min_conf=self.threshold)
        keep=det['conf']>=self.threshold
        return dict(xyxy=det['xyxy'][keep].astype(float)+np.array([x1,y1,x1,y1],float),
                    conf=det['conf'][keep].astype(float),coarse=det['coarse'][keep])
