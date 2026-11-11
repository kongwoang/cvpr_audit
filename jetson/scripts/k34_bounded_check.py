"""CPU regression: common-canvas B2K-A vs frozen K31 semantics, and keep=False vs K29."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from vtf import speculative as SP
from rap import planner as P
from k34_bounded import bounded_keep

ROOT = Path(__file__).resolve().parents[1]

def function(path, name, env):
    tree = ast.parse(path.read_text()); node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env[name]

spec = importlib.util.spec_from_file_location('kl', ROOT/'remote/A_method/k_loop.py')
KL = importlib.util.module_from_spec(spec); spec.loader.exec_module(KL)
ref = function(ROOT/'remote/A_method/k31_eval.py', 'run_bounded_keep', {'np':np,'SP':SP,'KL':KL})
old = function(ROOT/'scripts/59_bounded_latency.py', 'bounded_run', {'np':np,'SP':SP})
pp = P.PlannerParams()
cam = SimpleNamespace(calib=SimpleNamespace(P2=np.array([[100.,0,100.,0],[0,100.,70.,0],[0,0,1.,0]])),cam_h=1.5)
orig_field, orig_support = SP.field_crop, SP.support
# A fixed projected field and no support isolate merge/guard/cap semantics.
SP.field_crop = lambda *a,**k: np.array([20.,20.,120.,120.])
KL.field_crop_g = lambda *a,**k: SP.field_crop()
SP.support = lambda *a,**k: np.zeros(0,int)
changed = 0

def lift(d):
    n=len(d['conf']);return (np.full(n,10.),np.full(n,-.5),np.full(n,.5),np.full(n,np.inf))

def detector(seed):
    # Identical deterministic outputs for each crop, including empty, truncated and duplicate boxes.
    def one(c):
        rng=np.random.default_rng(seed + int(np.sum(c)*100))
        n=int(rng.integers(0,5));boxes=[]
        for i in range(n):
            x1,y1,x2,y2=c
            if i%2: boxes.append([x1+10,y1+10,min(x1+40,x2-5),min(y1+40,y2-5)])
            else: boxes.append([x1,y1+3,min(x1+40,x2),min(y1+50,y2)])
        return {'xyxy':np.array(boxes,float).reshape(-1,4),'conf':rng.uniform(.3,.99,n),'coarse':np.full(n,'car',dtype='<U8')}
    return one

def equal(a,b):
    assert set(a)==set(b)
    for k in a:
        if k=='mono':assert all(np.array_equal(x,y) for x,y in zip(a[k],b[k])),k
        else:assert np.array_equal(a[k],b[k]),k

try:
    for seed in range(200):
        cheap={'xyxy':np.array([[30.,30.,60.,60.]]),'conf':np.array([.5]),'coarse':np.array(['car'])}
        cheap['mono']=lift(cheap)
        detect=detector(seed)
        for cap in (1,2):
            a,ac,hit=bounded_keep(cheap,10,cam,200,200,lambda cs:[detect(c) for c in cs],lift,pp,0,cap)
            b,cr,gr,bc,conv=ref(cheap,10,cam,200,200,detect,lift,pp,cap,'A')
            equal(a,b);assert ac==bc and hit==(not conv) and ac<=cap
            c,cc,ch=bounded_keep(cheap,10,cam,200,200,lambda cs:[detect(c) for c in cs],lift,pp,0,cap,keep_pending=False)
            d,dc,dh=old(cheap,10,cam,200,200,lambda cs:[detect(c) for c in cs],lift,pp,0,cap)
            equal(c,d);assert (cc,ch)==(dc,dh)
            changed += not np.array_equal(a['xyxy'],c['xyxy'])
    assert changed>0,'fixtures never exercised pending detections'
finally:
    SP.field_crop,SP.support=orig_field,orig_support
print('PASS: 400 B2K-A cases match K31; 400 keep=False cases match K29; pending changes',changed)
