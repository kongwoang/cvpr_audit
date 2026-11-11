"""Real-image input and detector-output parity before runtime timings."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(HERE))
from staged import StagedCrops, StagedOneCall, letterbox_into
from vtf.gpuprep import letterbox

spec = importlib.util.spec_from_file_location('runtime_harness', HERE/'scripts/79_runtime_latency.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def main():
    out = HERE/'logs'
    out.mkdir(exist_ok=True)
    checks = []
    for ds in ('nusc','av2'):
        args = SimpleNamespace(max_units=1)
        gen = H.frames_pkl(args,'val') if ds == 'nusc' else H.frames_k29_av2(args)
        paths = []
        for row in gen:
            paths.append(row[2])
            if len(paths) == 2:
                break
        assert len(paths) == 2
        gpu = H.K.GPUDetect()
        scale = 960/H.NWS[ds]
        ref = H.KF.Crops(gpu,scale)
        staged = StagedCrops(gpu,scale,H.ENG/'yolov8s_dyn.engine',H.Mode,H.THR,H.STATS)
        old_one, new_one = H.KB.OneCall(gpu,scale,ref.eng), StagedOneCall(staged)
        for path in paths:
            image = cv2.imread(path)
            assert image is not None
            gpu.frame(image)
            height, width = image.shape[:2]
            c1 = [0.,height*.30,width*.65,height*.85]
            c2 = [width*.30,height*.42,width*.92,height*.96]
            c3 = [width*.445,height*.40,width*.705,height*.69]
            crops = [c1,c2,c3]+[c3]*8  # mixed sizes, repeats, batch splitting
            for item in staged.items(crops):
                inp,r,pad = letterbox(item[1],item[2],item[3])
                view = staged.eng.input_view((1,3,item[2],item[3]))
                rr,pp = letterbox_into(item[1],view[0])
                assert (r,pad)==(rr,pp)
                assert torch.equal(inp.to(dtype=staged.eng.in_dtype),view[0])
            for mode,a,b in [('grouped',ref,staged),('common',old_one,new_one)]:
                d1,d2 = a(crops),b(crops)
                assert len(d1)==len(d2)==len(crops)
                for i,(x,y) in enumerate(zip(d1,d2)):
                    for key in ('xyxy','conf','coarse'):
                        assert np.array_equal(x[key],y[key]),(ds,mode,path,i,key)
                checks.append(dict(dataset=ds,path=path,mode=mode,crops=len(crops),
                    detections=sum(len(x['conf']) for x in d1),identical=True))
        del ref,staged,old_one,new_one,gpu
        torch.cuda.empty_cache()
    assert sum(r['detections'] for r in checks)>0
    (out/'precheck.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2)+'\n')
    print(json.dumps(dict(status='PASS',groups=len(checks),crop_outputs=sum(c['crops'] for c in checks),
                          detections=sum(c['detections'] for c in checks))),flush=True)


if __name__=='__main__':
    main()
