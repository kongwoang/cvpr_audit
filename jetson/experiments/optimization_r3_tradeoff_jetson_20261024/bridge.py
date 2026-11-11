"""GT-free causal bridge and acquisition hypotheses."""
import numpy as np
from rap.mono import box_iou
from rap import planner as P

def match(x,c,p,pc):
    out=np.full(len(x),-1,int)
    if not len(x) or not len(p):return out
    scores=box_iou(x,p);scores[np.asarray(c)[:,None]!=np.asarray(pc)[None,:]]=-1
    while scores.max()>=.3:
        i,j=np.unravel_index(scores.argmax(),scores.shape);out[i]=j;scores[i,:]=-1;scores[:,j]=-1
    return out

def augment(full,h,chn,cheap_history):
    """Preserve original valid chains; append actual cheap observations only for gaps."""
    if cheap_history.get('s2') is None:return h,chn,0
    x=full['xyxy'];missing=set(range(len(x)))-{i for i,_,_ in chn}
    j1=match(x,full['coarse'],cheap_history['s1'],cheap_history['c1'])
    j12=match(cheap_history['s1'],cheap_history['c1'],cheap_history['s2'],cheap_history['c2'])
    p1=list(h['s1']) if h.get('s1') is not None else []
    p2=list(h['s2']) if h.get('s2') is not None else []
    chn=list(chn);n=0
    for i in sorted(missing):
        j=j1[i];k=j12[j] if j>=0 else -1
        if k<0:continue
        chn.append((i,len(p1),len(p2)));p1.append(cheap_history['s1'][j]);p2.append(cheap_history['s2'][k]);n+=1
    h=dict(h,s1=np.asarray(p1).reshape(-1,4),s2=np.asarray(p2).reshape(-1,4))
    return h,chn,n

def transfer(full,mono,chn,base):
    out=[np.asarray(x).copy() for x in mono];known={i for i,_,_ in chn}
    links=match(full['xyxy'],full['coarse'],base['xyxy'],base['coarse']);count=0
    for i,j in enumerate(links):
        if i in known or j<0:continue
        if np.isfinite(base['mono'][3][j]) and base['mono'][3][j]>0:
            out[3][i]=base['mono'][3][j];count+=1
    return tuple(out),count

def extent_seed(h,base,W,H,cropfn):
    """At most one previous verified braking object, re-localized by current cheap."""
    item=h.get('extent')
    if item is None:return []
    box,cls=item;j=match(np.asarray([box]),np.asarray([cls]),base['xyxy'],base['coarse'])[0]
    if j>=0:box=base['xyxy'][j]
    return [cropfn(box,W,H,2.)]

def extent_update(h,scene,refined,v,pp):
    _,d=P.required_decel(*refined,v,pp)
    ids=np.flatnonzero(np.asarray(scene['from_full'])&(d>=pp.thr_decel))
    if len(ids):
        i=int(ids[np.argmax(d[ids])]);h['extent']=(scene['xyxy'][i].copy(),str(scene['coarse'][i]))
    else:h['extent']=None

def motion_probe(base,pp):
    z,lo,hi,tt=map(np.asarray,base['mono'])
    use=np.isfinite(z)&(z>0)&(z<=80)&np.isfinite(tt)&(tt>0)&(lo<3.2)&(hi>-3.2)
    if not use.any():return 0.
    return float(np.clip(np.max(z[use]/tt[use]),0,10))
