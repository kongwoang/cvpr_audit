"""Ensure a crop request cannot itself fabricate a temporal observation."""
import numpy as np
from proactive import Pending
from rap import planner as P

box=np.array([[100.,100.,140.,140.]])
cls=np.array(['vehicle']);pp=P.PlannerParams()
base=dict(xyxy=box,coarse=cls,mono=(np.array([20.]),np.array([-.5]),np.array([.5]),np.array([2.])))
scene=dict(base,mono=(np.array([20.]),np.array([-.5]),np.array([.5]),np.array([np.inf])),from_full=np.array([True]))
p=Pending();assert p.seed(base,640,480,lambda *args:args[0])==[]
p.update(scene,base,[],1.,pp)
assert p.item is not None and p.armed==1
old=box.copy();seeds=p.seed(base,640,480,lambda *args:args[0])
assert len(seeds)==1 and np.array_equal(box,old)
quiet=dict(base,mono=scene['mono'])
missing=dict(scene,from_full=np.array([False]))
p.update(missing,quiet,[],1.,pp)
assert p.item[2]==1 and p.completed==0
assert len(p.seed(quiet,640,480,lambda *args:args[0]))==1
p.update(missing,quiet,[],1.,pp)
assert p.item is None and p.expired==1 and p.completed==0
p.update(scene,base,[],1.,pp);p.seed(base,640,480,lambda *args:args[0])
p.update(scene,quiet,[0],1.,pp)
assert p.item is None and p.completed==1
print('PASS: bounded outstanding request, quiet-frame continuation, expiry, actual-chain completion')
