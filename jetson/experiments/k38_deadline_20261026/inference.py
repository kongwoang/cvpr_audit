"""Deployment-only masked-head inference on real current/past images."""
import numpy as np
import json
import torch
from torch.nn import functional as F
from common import HERE
from learned import Adapter,decode,extras
from bridge import match

class Runtime:
    def __init__(self,dataset,encoder,crop64,assoc_idx,calibrated=False):
        self.encoder=encoder;self.crop64=crop64;self.nets={};self.ck={};self.bank={};self.pred={}
        self.assoc_idx=assoc_idx
        self.calibrations={}
        self.dataset=dataset;self.pair=None
        self.encoded=0;self.head_items=0
        for objective in ('geometry','decision'):
            ck=torch.load(HERE/'models'/dataset/(objective+'_s0')/'adapter.pt',map_location='cpu',weights_only=False)
            net=Adapter(ck['n_extra']);net.load_state_dict(ck['state']);self.nets[objective]=net.cuda().eval();self.ck[objective]=ck
            if calibrated:self.calibrations[objective]=json.loads((HERE/'models'/dataset/(objective+'_s0')/'calibration.json').read_text())
    def frame(self,fr):
        self.fr=fr;self.bank={k:v for k,v in self.bank.items() if fr-2<=k[0]<=fr};self.pred={};self.encoded=0;self.head_items=0
    def encode(self,items):
        missing=[];seen=set()
        for fid,img,box in items:
            key=(fid,tuple(box))
            if key not in self.bank and key not in seen:missing.append((key,img,box));seen.add(key)
        if missing:
            for start in range(0,len(missing),128):
                sub=missing[start:start+128];cr=np.stack([self.crop64(im,b) for _,im,b in sub])
                x=torch.from_numpy(cr).permute(0,3,1,2).cuda().float()/255
                x=F.interpolate(x.flip(1),size=128,mode='bilinear',align_corners=False)
                with torch.no_grad():e=self.encoder.cnn((x-self.encoder.mu)/self.encoder.sd).cpu().numpy()
                for (key,_,_),value in zip(sub,e):self.bank[key]=value
            self.encoded+=len(missing)
        return np.stack([self.bank[(fid,tuple(box))] for fid,_,box in items]) if items else np.zeros((0,512),np.float32)
    def infer(self,objective,full,raw,zz,h,chn,cheap_history,imgs,img,W,H,v,dt,kind='crop',only_indices=None):
        n=len(full['xyxy']);result=np.full((n,7),np.nan,np.float32);masks=np.zeros((n,3),np.float32)
        if not n:return result,masks
        raw=np.stack(raw,1);valid=np.flatnonzero(np.isfinite(raw[:,:3]).all(1)&(raw[:,0]>0)&(raw[:,0]<=80)&(raw[:,1]<3.2)&(raw[:,2]>-3.2))
        if only_indices is not None:valid=np.array([i for i in valid if i in only_indices],int)
        if not len(valid):return result,masks
        chains={i:(j,k) for i,j,k in chn};history=cheap_history
        j1=match(full['xyxy'],full['coarse'],history['s1'],history['c1']) if history.get('s1') is not None else np.full(n,-1,int)
        j12=match(history['s1'],history['c1'],history['s2'],history['c2']) if history.get('s2') is not None else np.full(len(history.get('s1',[])),-1,int)
        if kind=='cheap':
            j1=self.assoc_idx(full['xyxy'],history['s1']) if history.get('s1') is not None else np.full(n,-1,int)
            j12=self.assoc_idx(history['s1'],history['s2']) if history.get('s2') is not None else np.full(len(history.get('s1',[])),-1,int)
        features=[];boxes=[];kinds=[];allitems=[];positions=[]
        for i in valid:
            b=np.zeros((3,4));b[0]=full['xyxy'][i];mask=np.array([1.,0.,0.]);source=kind
            if i in h.get('_reject_current',set()):
                source='bridge' # current-only fallback after a learned identity rejection
            elif i in chains and len(imgs)==2:
                j,k=chains[i];b[1]=h['s1'][j];b[2]=h['s2'][k];mask[1:]=1
            elif kind=='crop':
                source='bridge';j=j1[i];k=j12[j] if j>=0 else -1
                if j>=0 and len(imgs)>0:b[1]=history['s1'][j];mask[1]=1
                if k>=0 and len(imgs)>1:b[2]=history['s2'][k];mask[2]=1
            elif kind=='cheap':
                j=j1[i];k=j12[j] if j>=0 else -1
                if j>=0 and len(imgs)>0:b[1]=history['s1'][j];mask[1]=1
                if k>=0 and len(imgs)>1:b[2]=history['s2'][k];mask[2]=1
            hh=np.maximum(b[:,3]-b[:,1],1e-3);hr=[hh[0]/hh[1] if mask[1] else 1,hh[1]/hh[2] if mask[2] else 1]
            x1,y1,x2,y2=b[0];zg,zh=zz['z_ground'][i],zz['z_height'][i]
            features.append([x1/W,y1/H,x2/W,y2/H,(x2-x1)/W,(y2-y1)/H,full['conf'][i],{'vehicle':0,'person':1,'cyclist':2}[str(full['coarse'][i])],raw[i,0],zg,zh,np.log(max(zg,.5)/max(zh,.5)),*np.log(np.clip(hr,1e-3,None)),dt,1.])
            masks[i]=mask;boxes.append(b);kinds.append(source)
            for lag in range(3):
                if mask[lag]:
                    positions.append((len(boxes)-1,lag));allitems.append((self.fr-lag,img if lag==0 else imgs[-lag],b[lag]))
        device=next(self.nets[objective].parameters()).device
        e=torch.zeros((len(valid),3,512),dtype=torch.float32,device=device)
        for (i,lag),value in zip(positions,self.encode(allitems)):e[i,lag]=value
        feats=np.array(features,np.float32);mask=masks[valid];ex=extras(feats,mask,np.array(kinds),np.full(len(valid),v),raw[valid],np.array(boxes))
        ck=self.ck[objective];fe=((feats-ck['fmu'])/ck['fsd']).astype(np.float32)
        with torch.no_grad():
            o=self.nets[objective](e,torch.from_numpy(fe).to(device),torch.from_numpy(mask).to(device),torch.from_numpy(ex).to(device))
            rate,factor,q,identity=decode(o)
            decoded=torch.cat([rate[:,None],factor[:,None],q,identity],1).cpu().numpy()
        if objective in self.calibrations:
            cal=self.calibrations[objective]
            for j,(k,obs) in enumerate(zip(kinds,mask.sum(1))):
                delta=cal['groups'].get(k+'/'+str(int(obs)),{}).get('delta',cal['global_delta'])
                decoded[j,2]=max(0,decoded[j,2]-delta);decoded[j,4]=min(2,decoded[j,4]+delta)
        result[valid]=decoded;self.head_items+=len(valid)
        return result,masks

    def identity_filter(self,full,h,chn,imgs,img,dt):
        if not chn:return chn,[]
        if self.pair is None:
            ck=torch.load(HERE/'models'/self.dataset/'identity_pair_s0/pair.pt',map_location='cpu',weights_only=False)
            net=torch.nn.Sequential(torch.nn.Linear(ck['n_in'],128),torch.nn.ReLU(),torch.nn.Linear(128,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
            net.load_state_dict(ck['state']);self.pair=(net.cuda().eval(),ck)
        items=[];boxes=[]
        for i,j,k in chn:
            b=np.stack([full['xyxy'][i],h['s1'][j],h['s2'][k]]);boxes.append(b)
            items.extend([(self.fr,img,b[0]),(self.fr-1,imgs[-1],b[1]),(self.fr-2,imgs[-2],b[2])])
        e=self.encode(items).reshape(-1,3,512);e/=np.maximum(np.linalg.norm(e,axis=2,keepdims=True),1e-6)
        boxes=np.array(boxes);sizes=np.maximum(boxes[:,:,2:]-boxes[:,:,:2],1);centers=(boxes[:,:,:2]+boxes[:,:,2:])/2
        features=[]
        for lag in (1,2):
            cur=e[:,0];past=e[:,lag];onehot=np.tile([1,0,0,0],(len(chn),1)) # current verified, original/retro verified history
            geom=np.c_[(cur*past).sum(1),(centers[:,lag]-centers[:,0])/sizes[:,0],np.log(sizes[:,lag]/sizes[:,0]),np.full(len(chn),dt*lag),onehot]
            features.append(np.c_[cur,past,abs(cur-past),geom].astype(np.float32))
        net,ck=self.pair;x=(np.concatenate(features)-ck['mu'])/ck['sd']
        with torch.no_grad():scores=net(torch.from_numpy(x).cuda())[:,0].sigmoid().cpu().numpy().reshape(2,-1).T
        rejected=[int(c[0]) for c,p in zip(chn,scores) if np.any(p>ck['threshold'])]
        return [c for c in chn if c[0] not in rejected],[(int(c[0]),p.tolist(),int(c[0]) in rejected) for c,p in zip(chn,scores)]
