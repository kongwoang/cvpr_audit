"""CPU oracle parity of isolated loop and cap invariants, no detector network."""
import json
import math
import os
from pathlib import Path
import numpy as np
from rap import planner as P
from vtf import speculative as SP
import k_loop as KL
from k41_variant_loop import run,guarded


def eq(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return np.array_equal(np.asarray(a),np.asarray(b))


def main():
    from types import SimpleNamespace
    cam=SimpleNamespace(calib=SimpleNamespace(P2=np.array([[120.,0,80.,0],[0,120.,45.,0],[0,0,1.,0]])),cam_h=1.5)
    W,H=160,90;pp=P.PlannerParams();checks=[]
    for seed in range(60):
        rng=np.random.RandomState(seed);n=1+seed%5;z=rng.choice(SP.R_GRID,n);y=rng.choice(SP.Y_GRID,n)
        bx=np.stack([80-120*y/z-108/z,45+180/z-192/z,80-120*y/z+108/z,45+180/z],1)
        conf=np.linspace(.9,.5,n);g=(z,y-.9,y+.9,np.full(n,np.inf));seen=rng.rand(n)<.5
        cheap=dict(xyxy=bx[seen],conf=conf[seen],coarse=np.array(['vehicle']*n)[seen],mono=tuple(a[seen] for a in g))
        def detect(c):
            m=SP.centres_in(bx,[c]);return dict(xyxy=bx[m].copy(),conf=conf[m].copy(),coarse=np.array(['vehicle']*int(m.sum())))
        def lift(d):
            ids=[int(np.flatnonzero((conf==v)&np.all(bx==b,axis=1))[0]) for b,v in zip(d['xyxy'],d['conf'])]
            return tuple(a[ids] for a in g)
        v=float(rng.uniform(0,25))
        def requests(s,cs,gs,step):
            fc=KL.field_crop_g(s['mono'],v,cam,W,H,pp,(0.,),SP.R_GRID,SP.Y_GRID,(KL.CAR,));new=[]
            if fc is not None and not SP.covered(fc,cs):new.append(fc)
            for k in SP.support(s['mono'],v,pp):
                c=KL.box_crop_ctx(s['xyxy'][k],W,H,2.)
                if not SP.covered(c,cs+gs+new):new.append(c)
            return new
        canvas=lambda c:max(1,math.ceil(c[2]-c[0]))*max(1,math.ceil(c[3]-c[1]))
        reference=KL.run_g(cheap,v,cam,W,H,detect,lift,pp)
        actual=run(cheap,requests,detect,lift,W,H,canvas)
        assert eq(reference,actual[:5]),('oracle_loop',seed)
        for budget in (0.,500.,5000.):
            bounded=run(cheap,requests,detect,lift,W,H,canvas,budget=budget,cap='strict')
            used=sum(t['pixels'] for t in bounded[-1] if t['executed']);assert used<=budget+1e-6
            if budget==0:assert eq(bounded[0]['xyxy'],cheap['xyxy']) and not bounded[1] and not bounded[2]
        checks.append(dict(seed=seed,uncapped_reference_parity=True,strict_caps_pass=True))
    # Force guard truncation then a budget-denied growth: pending detection retained.
    c=np.array([20.,20.,60.,60.]);b=np.array([[20.,22.,30.,42.]])
    det=lambda _:dict(xyxy=b.copy(),conf=np.array([.8]),coarse=np.array(['vehicle']))
    used=[0.];trace=[];got,grown,executed=guarded(det,c,W,H,lambda _:100.,100.,used,trace,1)
    assert executed and eq(got,det(c)) and not grown and len(trace)==2 and not trace[1]['executed']
    out=Path(os.environ['B'])/'results/k41/variant_cpu_tests';out.mkdir(parents=True,exist_ok=True)
    (out/'checks.json').write_text(json.dumps(dict(oracle_fixtures=checks,guard_pending_keep=True,network_inference=False),indent=2)+'\n')
    print('60 oracle loop fixtures, 180 strict-cap checks and pending-guard test passed')


if __name__=='__main__':main()
