"""CPU fixtures for literal K39 cap semantics and action-independent tubes."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rap import planner as P
from vtf import speculative as SP
import instr_eval as IE
import k41_k39_cap_reference as REF
from k41_variant_loop import run
from k40r_tube_geometry import tube


def eq(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True) if np.asarray(a).dtype.kind not in 'UOS' else np.array_equal(a,b)


def main():
    W,H=160,90
    c1=np.array([20.,20.,60.,60.]);c2=np.array([100.,20.,140.,60.]);box=np.array([[20.,22.,30.,42.]])
    cheap=dict(xyxy=np.array([[115.,30.,125.,45.]]),conf=np.array([.8]),coarse=np.array(['vehicle']),mono=(np.array([10.]),np.array([-1.]),np.array([1.]),np.array([np.inf])))
    def detect(c):
        # The first request triggers guard. Grown detections are absent.
        return dict(xyxy=box.copy(),conf=np.array([.9]),coarse=np.array(['vehicle'])) if np.array_equal(c,c1) else dict(xyxy=np.zeros((0,4)),conf=np.zeros(0),coarse=np.zeros(0,'<U8'))
    lift=lambda d:tuple(np.full(len(d['conf']),v) for v in (10.,-1.,1.,np.inf))
    count=0
    for budget in [0.,99.,100.,199.,200.,299.,300.,1000.]:
        selected=[];used=0
        for c in [c1,c2]:
            if used+100<=budget:selected.append(c);used+=100
        tpx=[0.]
        def make_detect(tpx,scale):
            def inner(c):tpx[0]+=100;return detect(c)
            return inner
        REF.np=np;REF.SP=SP;REF.make_detect=make_detect;REF.W=W;REF.H=H;REF.scale9=1.;REF.ckey=lambda c,s:(0,0,0,0,10,10)
        reference,allc=IE.merge_once(cheap,selected,REF.strict_guarded(tpx,budget,[0,0]),lift)
        actual=run(cheap,lambda scene,cs,gs,step:[c1,c2] if step==0 else [],detect,lift,W,H,lambda c:100.,budget=budget,cap='k39_strict',max_rounds=1)
        assert eq(reference,actual[0]),('literal cap',budget)
        assert eq(allc,actual[1]+actual[2]),('selected regions',budget)
        assert sum(t['pixels'] for t in actual[-1] if t['executed'])==tpx[0]
        count+=1
    literal=run(cheap,lambda *a:[c1,c2],detect,lift,W,H,lambda c:100.,budget=200.,cap='k39_strict',max_rounds=1)
    observed=run(cheap,lambda *a:[c1,c2],detect,lift,W,H,lambda c:100.,budget=200.,cap='strict_observed',max_rounds=1)
    assert len(literal[0]['conf'])==0 and len(observed[0]['conf'])==1
    assert sum(t['pixels'] for t in literal[-1] if t['executed'])==200
    cam=SimpleNamespace(calib=SimpleNamespace(P2=np.array([[120.,0,80.,0],[0,120.,45.,0],[0,0,1.,0]])),cam_h=1.5)
    pp=P.PlannerParams();checks=0
    for speed in np.r_[np.arange(0,40.1,.1),np.sqrt(np.maximum((SP.R_GRID-pp.standoff)*2,0))]:
        previous=None
        for tau in [1.,3.5]:
            crop,info=tube(float(speed),tau,cam,W,H,pp)
            length=pp.standoff+speed*pp.reaction_t+speed*speed/(2*tau)
            assert info['analytic_length_m']==length
            assert np.array_equal(info['mask'].any(1),SP.R_GRID<=min(length,pp.max_range))
            if previous is not None:assert np.all(~info['mask']|previous)
            previous=info['mask'];checks+=1
    out=Path(os.environ['B'])/'results/k40r/d02_cpu_tests';out.mkdir(parents=True,exist_ok=True)
    (out/'checks.json').write_text(json.dumps(dict(literal_cap_cases=count,denied_region_difference_explicit=True,tube_cases=checks,network_inference=False,script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(f'{count} literal K39 cap cases, denied-region counterexample, {checks} tube geometry cases passed')


if __name__=='__main__':main()
