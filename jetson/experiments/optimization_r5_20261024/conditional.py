"""Decision-disagreement features, no labels in inference."""
import numpy as np
from features import demands,features,trace_features

def augment_features(x,raw,refined,v):
    raw=np.asarray(raw,float).reshape(-1,4);ref=np.asarray(refined,float).reshape(-1,4)
    scale=ref[:,0]/np.maximum(raw[:,0],1e-6)
    return np.nan_to_num(np.c_[x,ref[:,:3],np.log(np.clip(scale,1e-6,1e6)),demands(ref,v),
        (ref[:,1]<1.2)&(ref[:,2]>-1.2),demands(raw,v)-demands(ref,v)],nan=0,posinf=1e3,neginf=-1e3).astype(np.float32)

def trace_input(t,v,ds):
    return augment_features(trace_features(t,v,ds),np.stack(t['raw'],1),np.stack(t['refined'],1),v)
