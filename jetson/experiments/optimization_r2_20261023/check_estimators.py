"""Semantic checks: observation identity, causal motion bounds, geometry coupling."""
import numpy as np
from estimators import unique_links,motion_floors,add_floor,scale_refinement

def box(size):return np.array([[0.,0.,size,size]])
c=np.array(['vehicle']);x=box(100)
assert unique_links(x,c,x,c).tolist()==[0]
assert unique_links(x,c,np.r_[x,x],np.r_[c,c]).tolist()==[-1]
assert unique_links(x,c,x,np.array(['cyclist'])).tolist()==[-1]
for sizes in [(100,100,100),(80,90,100)]:
    rates=motion_floors(box(sizes[0]),c,(box(sizes[1]),c),(box(sizes[2]),c),1.,.5)
    assert all(np.all(r==0) for r in rates)
rates=motion_floors(box(121),c,(box(110),c),(box(100),c),1.,.5)
assert all(0<r[0]<.2 for r in rates),rates
assert np.all(motion_floors(x,c,None,None,1.,.5)[0]==0)
mono=(np.array([20.]),np.array([-1.]),np.array([1.]),np.array([np.inf]))
out,changed=add_floor(mono,np.array([.2]),np.array([True]))
assert changed[0] and out[3][0]==5 and np.isinf(mono[3][0])
out2,changed=add_floor(out,np.array([.1]),np.array([True]))
assert not changed[0] and out2[3][0]==5
factor=np.array([2.]);origin=scale_refinement(mono,factor)
anchor=scale_refinement(mono,factor,np.array([2.,.5,1.5]))
assert origin[0][0]==anchor[0][0]==40
for i in [1,2]:
    assert np.isclose((anchor[i][0]-.5)/(anchor[0][0]-2),(mono[i][0]-.5)/(mono[0][0]-2))
print('PASS: zero/shrinking evidence, conservative growth, identity ambiguity, no stale motion, ray coupling')
