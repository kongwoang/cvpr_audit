"""Same recovery policy with redundant CPU work removed."""
import numpy as np
from rap import planner as P
from helpers import Retro,unique_match


class FastRetro(Retro):
    def augment(self,full,raw,history,known_chains=None):
        x=np.asarray(full['xyxy']).reshape(-1,4)
        chn=(list(known_chains) if known_chains is not None else
             (self.chains(x,history.get('s1'),history.get('s2')) if len(self.imgs)==2 else []))
        missing=set(range(len(x)))-{i for i,_,_ in chn}
        if not self.used and len(self.imgs)==2 and missing:
            if not hasattr(self,'_cheap_demand'):
                self._cheap_demand=np.asarray(P.required_decel(*self.base['mono'],self.v,self.pp)[1])
            dc=self._cheap_demand
            if (dc>=self.pp.thr_decel).any():
                eligible=[i for i in sorted(missing) if np.isfinite(raw[0][i]) and raw[0][i]<=self.pp.max_range
                    and raw[1][i]<self.pp.corridor_half_w and raw[2][i]>-self.pp.corridor_half_w
                    and tuple(x[i]) in self.log]
                if eligible:
                    df=np.asarray(P.required_decel(*raw,self.v,self.pp)[1])
                    candidates=[]
                    for i in eligible:
                        if not df[i]<self.pp.thr_decel:continue
                        j=unique_match(x[i],full['coarse'][i],self.base)
                        if j>=0 and dc[j]>=self.pp.thr_decel:candidates.append((-float(dc[j]),i))
                    if candidates:
                        _,i=min(candidates);self.used=True;self.attempted+=1
                        key=self.log[tuple(x[i])];past=[];box=x[i]
                        for lag in (1,2):
                            det=self.run_det(self.imgs[-lag],key)
                            self.calls+=1;self.pixels+=key[-2]*key[-1]
                            j=unique_match(box,full['coarse'][i],det)
                            if j<0:break
                            box=np.asarray(det['xyxy'][j]).copy();past.append(box)
                        if len(past)==2:
                            self.recovered[tuple(x[i])]=past;self.success+=1
        applicable=[i for i in sorted(missing) if tuple(x[i]) in self.recovered]
        h=dict(history)
        if not applicable:
            for name in ('s1','s2'):
                value=h.get(name)
                h[name]=np.asarray(value).reshape(-1,4) if value is not None else np.zeros((0,4))
            return h,chn
        p1=list(np.asarray(h['s1'])) if h.get('s1') is not None else []
        p2=list(np.asarray(h['s2'])) if h.get('s2') is not None else []
        for i in applicable:
            past=self.recovered[tuple(x[i])]
            chn.append((i,len(p1),len(p2)));p1.append(past[0]);p2.append(past[1])
        h['s1']=np.asarray(p1).reshape(-1,4);h['s2']=np.asarray(p2).reshape(-1,4)
        return h,chn
