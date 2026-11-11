"""New MLP on exact features from the existing C1 encoder bank."""
import numpy as np
import torch
from inference import Runtime
from vtf import c1fast as CF

def assoc(curr,prev,thr=.3):
    from rap.mono import box_iou
    out=np.full(len(curr),-1,int)
    if not len(curr) or not len(prev):return out
    iou=box_iou(curr,prev);taken=set()
    for i in np.argsort(-iou.max(1)):
        for j in np.argsort(-iou[i]):
            if iou[i,j]<thr:break
            if int(j) not in taken:out[i]=j;taken.add(int(j));break
    return out

class MissingRuntime(Runtime):
    def __init__(self,dataset,head,enc):
        super().__init__(dataset,head,CF.crop64_fast,assoc)
        self.fast_encoder=enc;self.shared_bank=None
    def encode(self,items):
        bank=self.shared_bank;bank.wait();need=[];seen=set();keys=[]
        for fid,img,box in items:
            key=CF._key(fid,box);keys.append(key)
            if key not in bank.row and key not in seen:need.append((key,img,box));seen.add(key)
        if need:
            cr=np.stack(bank.crop_many([(key[0],img,box) for key,img,box in need]))
            x=torch.from_numpy(cr).cuda().permute(0,3,1,2).float()/255
            e=self.fast_encoder(x);rows=bank.alloc(len(need))
            bank.feat[torch.tensor(rows,device='cuda')]=e
            for (key,_,_),row in zip(need,rows):bank.row[key]=row
            self.encoded+=len(need)
        return bank.feat.index_select(0,torch.tensor([bank.row[key] for key in keys],device='cuda',dtype=torch.long))
