"""Masked C1 adapters; labels never enter deployment feature construction."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KINDS=('crop','u960','cheap','bridge')

def extras(feats,mask,kind,v,raw,boxes):
    n=len(feats);onehot=np.stack([np.asarray(kind)==s for s in KINDS],1).astype(np.float32)
    sizes=np.maximum(boxes[:,0,2:]-boxes[:,0,:2],1.)
    center=(boxes[:,:,:2]+boxes[:,:,2:])/2
    delta=((center[:,1:]-center[:,:1])/sizes[:,None,:]).reshape(n,4)
    delta*=np.repeat(mask[:,1:],2,axis=1)
    aspect=np.log(np.maximum(boxes[:,:,2]-boxes[:,:,0],1)/np.maximum(boxes[:,:,3]-boxes[:,:,1],1))
    aspect=(aspect[:,1:]-aspect[:,:1])*mask[:,1:]
    return np.nan_to_num(np.c_[mask,onehot,np.asarray(v)/20,np.asarray(raw)[:,1:3]/20,
                                np.clip(delta,-5,5),np.clip(aspect,-3,3)],nan=0,posinf=0,neginf=0).astype(np.float32)

def features(e,fe,mask,extra):
    # Unavailable embeddings are masked exactly; no replicated crop is presented as history.
    e=e*mask[:,:,None]
    cs=F.cosine_similarity(e[:,:1],e[:,1:],dim=2)*mask[:,1:]
    return torch.cat([e.flatten(1),fe,extra,cs],1)

class Adapter(nn.Module):
    def __init__(self,n_extra,old_state=None):
        super().__init__();self.net=nn.Sequential(nn.Linear(1536+16+n_extra+2,256),nn.ReLU(),nn.Linear(256,7))
        if old_state is not None:
            with torch.no_grad():
                self.net[0].weight.zero_();self.net[0].weight[:,:1552].copy_(old_state['head.0.weight'])
                self.net[0].bias.copy_(old_state['head.0.bias'])
                self.net[2].weight.zero_();self.net[2].bias.zero_()
                for i in (0,2,3,4):
                    self.net[2].weight[i].copy_(old_state['head.2.weight'][0]);self.net[2].bias[i].copy_(old_state['head.2.bias'][0])
                self.net[2].bias[2]-=.1;self.net[2].bias[4]+=.1
    def forward(self,e,fe,mask,extra):return self.net(features(e,fe,mask,extra))

def decode(out):
    rate=out[:,0].clamp(0,2);factor=out[:,1].clamp(-1.1,1.1).exp()
    quantiles=out[:,2:5].clamp(0,2).sort(dim=1).values
    identity=out[:,5:7].sigmoid()
    return rate,factor,quantiles,identity

def demand(raw,v,rate,factor,soft=False):
    z=raw[:,0]*factor;lo=raw[:,1]*factor;hi=raw[:,2]*factor
    a_stop=v.square()/(2*(z-2-.4*v).clamp(min=.05))
    # Same 1,000 s cap as the frozen planner when rate is zero.
    rel=(z*rate.clamp(min=.001,max=10)).clamp(0,60)
    a_close=rel.square()/(2*(z-2).clamp(min=.05))
    inc=((lo<1.2)&(hi>-1.2)&(z<=80)&(v>.1)).float()
    if soft:
        inc=torch.sigmoid((1.2-lo)/.15)*torch.sigmoid((hi+1.2)/.15)*(z<=80).float()*(v>.1).float()
    return torch.maximum(a_stop,a_close).clamp(0,9)*inc
