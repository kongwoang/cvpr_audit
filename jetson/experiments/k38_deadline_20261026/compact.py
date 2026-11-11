"""Image-free masked-history readout and deployment wrapper."""
import numpy as np
import torch
from torch import nn
from inference import Runtime
from common import HERE

class Compact(nn.Module):
    def __init__(self,mu,sd):
        super().__init__();self.register_buffer('mu',torch.as_tensor(mu,dtype=torch.float32));self.register_buffer('sd',torch.as_tensor(sd,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(len(mu),128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,e,fe,mask,extra):
        x=(torch.cat([fe,extra],1)-self.mu)/self.sd;rate=self.net(x)
        return torch.cat([rate,torch.zeros_like(rate),rate.repeat(1,3),torch.zeros_like(rate).repeat(1,2)],1)

class CompactRuntime(Runtime):
    def __init__(self,dataset,encoder,crop64,assoc_idx):
        ck=torch.load(HERE/'models'/dataset/'compact_s0/compact.pt',map_location='cpu',weights_only=False)
        net=Compact(ck['mu'],ck['sd']);net.load_state_dict(ck['state'])
        self.encoder=encoder;self.crop64=crop64;self.assoc_idx=assoc_idx;self.dataset=dataset
        self.nets={'geometry':net.eval()};self.ck={'geometry':ck};self.bank={};self.pred={};self.pair=None;self.calibrations={}
        self.encoded=0;self.head_items=0
    def encode(self,items):
        # Real observation boxes still drive mask/height/center features; no ROI is read.
        return torch.zeros((len(items),512),dtype=torch.float32)
