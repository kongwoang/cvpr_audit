"""Causal, GT-free estimator proposals. No imports from the evaluator."""
import numpy as np
from rap.mono import box_iou


def unique_links(boxes, classes, previous_boxes, previous_classes, threshold=.3):
    """Mutually unambiguous class-compatible observations; reject crowded matches."""
    out=np.full(len(boxes),-1,int)
    if not len(boxes) or not len(previous_boxes):return out
    compatible=(box_iou(np.asarray(boxes),np.asarray(previous_boxes))>=threshold)
    compatible &= np.asarray(classes)[:,None]==np.asarray(previous_classes)[None,:]
    for i in np.flatnonzero(compatible.sum(1)==1):
        j=int(np.flatnonzero(compatible[i])[0])
        if compatible[:,j].sum()==1:out[i]=j
    return out


def motion_floors(boxes,classes,h1,h2,scale,dt,delta_px=1.):
    """Looming lower estimates under an ASSUMED +/- delta detector-pixel error.

    These are sensitivity estimates, not calibrated confidence/safety bounds.
    h1/h2 contain actual same-fidelity observations (boxes, classes). Rates use
    past/current only. hw requires both width and height growth; h uses height.
    """
    x=np.asarray(boxes,float).reshape(-1,4);n=len(x)
    two=np.zeros(n);height=np.zeros(n);both=np.zeros(n)
    if h1 is None or not len(x):return height,both,two
    p,c1=h1;j1=unique_links(x,classes,p,c1)
    j2=unique_links(p,c1,*h2) if h2 is not None else np.full(len(p),-1,int)
    delta=delta_px/scale
    def lower(a,b,elapsed):
        sizes_a=a[2:]-a[:2];sizes_b=b[2:]-b[:2]
        if np.any(sizes_a<=delta) or np.any(sizes_b<=delta):return np.zeros(2)
        return ((sizes_a-delta)/(sizes_b+delta)-1.)/elapsed
    for i,j in enumerate(j1):
        if j<0:continue
        r01=lower(x[i],p[j],dt)
        two[i]=np.clip(r01.min(),0,2)
        k=j2[j]
        if k<0:continue
        q=h2[0][k]
        rates=np.stack([r01,lower(p[j],q,dt),lower(x[i],q,2*dt)])
        height[i]=np.clip(rates[:,1].min(),0,2)
        both[i]=np.clip(rates.min(),0,2)
    return height,both,two


def add_floor(mono,rates,eligible):
    out=[np.asarray(x,float).copy() for x in mono]
    old=np.zeros(len(out[0]));valid=np.isfinite(out[3])&(out[3]>0)
    old[valid]=1/out[3][valid]
    change=np.asarray(eligible,bool)&(rates>old)&(rates>0)
    out[3][change]=1/rates[change]
    return tuple(out),change


def scale_refinement(raw,factor,translation=None):
    out=[np.asarray(x,float).copy() for x in raw]
    good=np.isfinite(out[0])&(out[0]>0)&np.isfinite(factor)
    out[0][good]*=factor[good]
    if translation is None:
        for a in out[1:3]:a[good]*=factor[good]
    else:
        tx,ty=translation[:2]
        eligible=good&(np.asarray(raw[0])>tx+1e-6)&(out[0]>tx+1e-6)
        ray=np.ones(len(out[0]));ray[eligible]=(out[0][eligible]-tx)/(np.asarray(raw[0])[eligible]-tx)
        for a in out[1:3]:
            a[good]*=factor[good]
        for i in (1,2):out[i][eligible]=ty+ray[eligible]*(np.asarray(raw[i])[eligible]-ty)
    return tuple(out)
