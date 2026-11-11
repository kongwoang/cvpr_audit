"""A bounded request for real future observations, never a fabricated C1 chain."""
import numpy as np
from rap import planner as P
from estimators import unique_links

class Pending:
    def __init__(self):
        self.item=None
        self.requested=0
        self.armed=0
        self.completed=0
        self.expired=0

    def seed(self,base,W,H,box_crop):
        if self.item is None:return []
        box,cls,_=self.item
        j=unique_links(np.asarray([box]),np.asarray([cls]),base['xyxy'],base['coarse'])[0]
        target=np.asarray(base['xyxy'][j]).copy() if j>=0 else box.copy()
        self.target=target
        self.requested+=1
        return [box_crop(target,W,H,2.)]

    def update(self,scene,base,chain_ids,v,pp):
        ids=np.flatnonzero(scene['from_full']);full_x=np.asarray(scene['xyxy'])[ids]
        full_c=np.asarray(scene['coarse'])[ids];chain=set(chain_ids)
        if self.item is not None:
            box,cls,life=self.item
            j=unique_links(np.asarray([self.target]),np.asarray([cls]),full_x,full_c)[0]
            life-=1
            if j>=0 and int(ids[j]) in chain:
                self.item=None;self.completed+=1
            elif life<=0:
                self.item=None;self.expired+=1
            else:
                self.item=(full_x[j].copy() if j>=0 else box,cls,life)
        if self.item is not None:return
        links=unique_links(full_x,full_c,base['xyxy'],base['coarse'])
        _,dc=P.required_decel(*base['mono'],v,pp)
        _,df=P.required_decel(*scene['mono'],v,pp)
        z,lo,hi,_=scene['mono'];candidates=[]
        for local,i in enumerate(ids):
            j=links[local]
            if (j>=0 and i not in chain and dc[j]>=pp.thr_decel and df[i]<pp.thr_decel
                and np.isfinite(z[i]) and 0<z[i]<=pp.max_range
                and lo[i]<pp.corridor_half_w and hi[i]>-pp.corridor_half_w):
                candidates.append((-float(dc[j]),int(i)))
        if candidates:
            _,i=min(candidates)
            self.item=(np.asarray(scene['xyxy'][i]).copy(),str(scene['coarse'][i]),2)
            self.armed+=1
