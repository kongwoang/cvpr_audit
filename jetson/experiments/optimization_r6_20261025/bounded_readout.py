"""Small deployable-form observation readout; frozen scene and history unchanged."""
import numpy as np
from rap import planner as P
from rap.mono import box_iou
PP=P.PlannerParams()
def action(x):return (np.asarray(x)>=1).astype(int)+(np.asarray(x)>=3.5).astype(int)
def links(x,c,p,pc):
    out=np.full(len(x),-1,int)
    if not len(x) or not len(p):return out
    scores=box_iou(x,p);scores[((np.asarray(c)=='vehicle')[:,None])!=((np.asarray(pc)=='vehicle')[None,:])]=-1
    used=set()
    for i in np.argsort(-scores.max(1)):
        for j in np.argsort(-scores[i]):
            if scores[i,j]<.3:break
            if int(j) not in used:out[i]=j;used.add(int(j));break
    return out
def readout(scene,proposal,history,v,dt,scale,lag):
    """Returns one extra action using only current observations and past full boxes.

    Time this after the unchanged detector plus cross-crop proposal deduplication.
    Costs for those operations and R5's classifier are outside this function.
    """
    n=len(proposal['xyxy'])
    if not n or not history:return 0
    x,c=proposal['xyxy'],proposal['coarse'];s=scene
    # Full candidate array participates in matching, exactly as in the reference.
    free=(links(x,c,s['xyxy'],s['coarse'])<0)&(proposal['conf']<proposal['threshold'])
    if not free.any():return 0
    h1=history[-1];s1=h1['scene'];ix1=np.flatnonzero(s1['from_full']);x1=s1['xyxy'][ix1];c1=s1['coarse'][ix1]
    j1=links(x,c,x1,c1);ok=j1>=0;keep=free&ok
    if not keep.any():return 0
    pd1=P.required_decel(*s1['refined'],h1['v'],PP)[1];a1=np.zeros(n,int);a1[ok]=action(pd1[ix1[j1[ok]]])
    ttc=np.full(n,np.inf);j2=np.full(n,-1,int);a2=np.zeros(n,int)
    if len(history)>=2:
        h2=history[-2];s2=h2['scene'];ix2=np.flatnonzero(s2['from_full']);x2=s2['xyxy'][ix2];c2=s2['coarse'][ix2]
        j12=links(x1,c1,x2,c2);j2[ok]=j12[j1[ok]];ok2=j2>=0
        hh=x[:,3]-x[:,1];hh1=np.full(n,np.nan);hh2=np.full(n,np.nan)
        hh1[ok]=x1[j1[ok],3]-x1[j1[ok],1];hh2[ok2]=x2[j2[ok2],3]-x2[j2[ok2],1]
        with np.errstate(invalid='ignore',divide='ignore'):
            rate=np.median(np.stack([(hh1/hh2-1)/dt,(hh/hh1-1)/dt,(hh/hh2-1)/(2*dt)]),axis=0)
            valid=ok2&(hh>1)&(hh1>1)&(hh2>1)&(rate>1e-3/dt)&((hh-hh2)*scale>1.)
            ttc[valid]=np.minimum(1/rate[valid],1e3)
        if lag==2:
            pd2=P.required_decel(*s2['refined'],h2['v'],PP)[1];a2[ok2]=action(pd2[ix2[j2[ok2]]])
    current=P.required_decel(*proposal['mono'][:3],ttc,v,PP)[1];a=np.minimum(action(current),a1)
    if lag==2:keep&=j2>=0;a=np.minimum(a,a2)
    return int(a[keep].max()) if keep.any() else 0
