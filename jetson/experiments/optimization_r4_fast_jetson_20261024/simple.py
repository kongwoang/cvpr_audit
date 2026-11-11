"""Small geometric/semantic/readout alternatives, with no training or GT input."""
import numpy as np
from rap.mono import box_iou

def compatible(a,b):
    # Person/cyclist can flicker within one physical vulnerable road user.
    return (np.asarray(a)=='vehicle')[:,None]==(np.asarray(b)=='vehicle')[None,:]

def union_fraction(box,crops):
    x1,y1,x2,y2=np.asarray(box,float);area=max((x2-x1)*(y2-y1),1e-6)
    clips=[]
    for c in crops:
        r=(max(x1,c[0]),max(y1,c[1]),min(x2,c[2]),min(y2,c[3]))
        if r[2]>r[0] and r[3]>r[1]:clips.append(r)
    if not clips:return 0.
    xs=sorted({x for r in clips for x in (r[0],r[2])});total=0.
    for left,right in zip(xs,xs[1:]):
        ys=sorted((r[1],r[3]) for r in clips if r[0]<right and r[2]>left)
        height=0.;end=-np.inf
        for bottom,top in ys:
            height+=max(0,top-max(bottom,end));end=max(end,top)
        total+=(right-left)*height
    return float(np.clip(total/area,0,1))

def covered_union(box,crops,tol=.95):return union_fraction(box,crops)>=tol

def links(x,c,p,pc,thr=.3):
    out=np.full(len(x),-1,int)
    if not len(x) or not len(p):return out
    iou=box_iou(x,p);iou[~compatible(c,pc)]=-1;used=set()
    for i in np.argsort(-iou.max(1)):
        for j in np.argsort(-iou[i]):
            if iou[i,j]<thr:break
            if int(j) not in used:out[i]=j;used.add(int(j));break
    return out

def chains(full,h):
    if h.get('s1') is None or h.get('s2') is None:return []
    a=links(full['xyxy'],full['coarse'],h['s1'],h['c1']);b=links(h['s1'],h['c1'],h['s2'],h['c2'])
    return [(i,int(j),int(b[j])) for i,j in enumerate(a) if j>=0 and b[j]>=0]

def replace_cheap(cheap,full,crops,covered):
    out=np.array([covered(b,crops) for b in cheap['xyxy']],bool)
    if len(full['xyxy']) and len(cheap['xyxy']):
        overlaps=box_iou(cheap['xyxy'],full['xyxy'])
        out|=((overlaps>=.5)&compatible(cheap['coarse'],full['coarse'])).any(1)
    return out

def readout_demands(cheap,scene,raw_dem,ref_dem,cheap_dem):
    m=links(scene['xyxy'],scene['coarse'],cheap['xyxy'],cheap['coarse'])
    present=(m>=0)&np.asarray(scene['from_full'],bool)
    cd=np.array([cheap_dem[j] if j>=0 else np.nan for j in m])
    median=np.asarray(ref_dem,float).copy();median[present]=np.median(np.c_[cd[present],raw_dem[present],ref_dem[present]],axis=1)
    hard=np.asarray(ref_dem,float).copy();veto=present&(cd>=3.5)&(raw_dem>=3.5)&(ref_dem<3.5);hard[veto]=raw_dem[veto]
    # Exact action-level abstention: both thresholds agree, so any third vote
    # leaves the object action unchanged under majority.
    action=lambda d:(np.asarray(d)>=1).astype(int)+(np.asarray(d)>=3.5).astype(int)
    skip=present&(action(cd)==action(raw_dem))
    return median,hard,skip,m
