"""Randomized semantic checks for acquisition order, state, history and counters."""
import json
import time
from pathlib import Path
import numpy as np
from rap import planner as P
from helpers import Retro
from fast import FastRetro
from rap.mono import box_iou


def assoc(x,y):
    out=-np.ones(len(x),int)
    if y is None or not len(y) or not len(x):return out
    iou=box_iou(x,y);taken=set()
    for i in np.argsort(-iou.max(1)):
        for j in np.argsort(-iou[i]):
            if iou[i,j]<.3:break
            if j not in taken:taken.add(int(j));out[i]=j;break
    return out


def chains(x,a,b):
    if a is None or b is None or not len(a) or not len(b) or not len(x):return []
    first=assoc(x,a);second=assoc(a,b)
    return [(i,int(j),int(second[j])) for i,j in enumerate(first) if j>=0 and second[j]>=0]


def main():
    pp=P.PlannerParams();rng=np.random.default_rng(0);successes=0;tested=0
    for case in range(1000):
        n=int(rng.integers(1,18));xy=rng.uniform(0,400,(n,4));xy[:,2:]=xy[:,:2]+rng.uniform(10,60,(n,2))
        coarse=rng.choice(['vehicle','person','cyclist'],n)
        full=dict(xyxy=xy,coarse=coarse,conf=np.ones(n))
        z=rng.uniform(25,100,n);lo=rng.uniform(-4,3,n);hi=lo+1
        if case%4==0:z[0]=40;lo[0]=-.5;hi[0]=.5
        raw=(z,lo,hi,np.full(n,np.inf))
        bz=rng.uniform(10,100,n)
        if case%4==0:bz[0]=8
        bt=np.full(n,np.inf)
        if case%17==0:bt[-1]=np.nan
        base=dict(full,mono=(bz,np.full(n,-.5),np.full(n,.5),bt))
        imgs=[object() for _ in range(2 if case%5 else case%2)]
        keep1=rng.random(n)<.5;keep2=rng.random(n)<.5
        history=dict(s1=xy[keep1].copy(),s2=xy[keep2].copy())
        if case%7==0:history['s1']=None
        if case%11==0:history['s2']=None
        log={tuple(b):(0,0,640,480,64,96) for b in xy}
        requests=[[],[]]
        def detector(which):
            def run(im,key):
                lag=next(i for i,p in enumerate(imgs) if p is im)
                requests[which].append((lag,key))
                out={k:v.copy() for k,v in full.items()}
                if case%13==0:out={k:v[:0] for k,v in out.items()}
                return out
            return run
        old=Retro(base,imgs,log,detector(0),chains,6.,pp)
        fast=FastRetro(base,imgs,log,detector(1),chains,6.,pp)
        for _ in range(3):
            ch=chains(xy,history.get('s1'),history.get('s2')) if len(imgs)==2 else []
            a,ca=old.augment(full,raw,history)
            b,cb=fast.augment(full,raw,history,known_chains=ch)
            assert ca==cb
            for name in ('s1','s2'):assert np.array_equal(a[name],b[name],equal_nan=True)
            for name in ('used','calls','pixels','attempted','success'):assert getattr(old,name)==getattr(fast,name),(case,name)
            assert requests[0]==requests[1]
            assert old.recovered.keys()==fast.recovered.keys()
            for key in old.recovered:
                assert all(np.array_equal(x,y) for x,y in zip(old.recovered[key],fast.recovered[key]))
            tested+=1
        successes+=old.success
    assert successes>0
    result=dict(status='PASS',randomized_frames=1000,augment_comparisons=tested,successful_recoveries=successes,
                notes='Includes cold/partial/full history, cached second lifts, empty detections and NaN cheap TTC.')
    out=Path(__file__).resolve().parent/'logs';out.mkdir(exist_ok=True)
    (out/'cpu_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
