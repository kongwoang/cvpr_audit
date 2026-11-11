"""CPU structural checks: unchanged path parity, finite closure, actual temporal observations."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rap import planner as P
from vtf import speculative as SP
from helpers import RefineClosure, Retro, unique_match
import sandbox_loop as candidate

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('original_loop',ROOT/'remote/A_method/k_loop.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
pp=P.PlannerParams()
cam=SimpleNamespace(calib=SimpleNamespace(P2=np.array([[400,0,320,0],[0,400,240,0],[0,0,1,0]],float)),cam_h=1.5)
rng=np.random.default_rng(0)
for case in range(200):
    n=int(rng.integers(0,12));xy=rng.uniform([10,10,20,20],[400,300,90,100],(n,4))
    xy[:,2:]+=xy[:,:2]
    all_det=dict(xyxy=xy,conf=rng.uniform(.3,.95,n),coarse=np.full(n,'vehicle'))
    def geom(d):
        z=700/np.maximum(d['xyxy'][:,3]-d['xyxy'][:,1],1)
        return z,np.full(len(z),-.5),np.full(len(z),.5),np.full(len(z),np.inf)
    cheap={k:x[::2].copy() for k,x in all_det.items()};cheap['mono']=geom(cheap)
    def detect(c):
        m=SP.centres_in(all_det['xyxy'],[c])
        return {k:x[m].copy() for k,x in all_det.items()}
    v=float(rng.uniform(0,20))
    a=original.run_g(cheap,v,cam,640,480,detect,geom,pp)
    b=candidate.run_g(cheap,v,cam,640,480,detect,geom,pp)
    for k in a[0]:
        if k=='mono': assert all(np.array_equal(x,y) for x,y in zip(a[0][k],b[0][k]))
        else: assert np.array_equal(a[0][k],b[0][k]),(case,k)
    for i in (1,2):
        assert len(a[i])==len(b[i]) and all(np.array_equal(x,y) for x,y in zip(a[i],b[i]))
    assert a[3:]==b[3:]

scene=dict(xyxy=np.array([[0,0,10,10],[20,0,30,10]],float),coarse=np.array(['vehicle']*2),
           conf=np.ones(2),from_full=np.ones(2,bool),mono=(np.array([10.,25.]),np.array([-.5,-.5]),np.array([.5,.5]),np.full(2,np.inf)))
called=[]
def refine(s,name,support):
    t=[x.copy() for x in s['mono']]
    for i in support(t):
        called.append(i)
        for x in t[:3]:x[i]*=10
    return tuple(t)
c=RefineClosure(refine,10,pp)
out=c(scene)
assert called==[0,1],called
assert np.array_equal(out['mono'][0],[100,250])
assert np.array_equal(c(out)['mono'][0],[100,250]) and called==[0,1]
assert np.array_equal(scene['mono'][0],[10,25])

X=np.array([[10,10,30,30]],float)
det=dict(xyxy=X,conf=np.ones(1),coarse=np.array(['vehicle']))
base=dict(det,mono=(np.array([5.]),np.array([-.5]),np.array([.5]),np.array([.1])))
past1=dict(det,xyxy=X+np.array([1,1,-1,-1]))
past2=dict(det,xyxy=X+np.array([2,2,-2,-2]))
images=[object(),object()]; seen=[]
def run(im,key):
    seen.append(im)
    return past1 if im is images[-1] else past2
retro=Retro(base,images,{tuple(X[0]):(0,0,100,100,64,64)},run,lambda *a:[],1,pp)
raw=(np.array([20.]),np.array([-.5]),np.array([.5]),np.array([np.inf]))
h,ch=retro.augment(det,raw,{})
assert ch==[(0,0,0)] and retro.calls==2 and retro.success==1
assert seen==[images[-1],images[-2]] and np.array_equal(h['s1'],past1['xyxy'])
retro.augment(det,raw,{})
assert retro.calls==2 and retro.pixels==8192
assert unique_match(X[0],'vehicle',dict(xyxy=np.repeat(X,2,0),coarse=np.array(['vehicle']*2))) == -1
print(json.dumps({'frozen_loop_cases':200,'closure_idempotence':True,'new_support_refinement':True,
                  'retro_real_images_and_budget':True,'ambiguous_association_rejected':True,'status':'PASS'}))
