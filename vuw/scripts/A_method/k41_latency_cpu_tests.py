"""CPU behavioral tests only: no images from evaluation data or model inference."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest
import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
try:
    from columns_only import host_columns
except ModuleNotFoundError:
    from k41_columns_only import host_columns
from vtf import gpupost as GP
from vtf.c1fast import crop64_fast, Bank
from vtf.fastpost import nms, nms_matrix
from rap.detect import Mode

torch.set_num_threads(1)
RESULTS = []


def reference_crop(img, box):
    # Exact release crop function captured as text, no imports from mutable release.
    x1,y1,x2,y2=box
    cx,cy,w,h=(x1+x2)/2,(y1+y2)/2,(x2-x1)*1.5,(y2-y1)*1.5
    side=max(w,h,8.)
    X1,Y1,X2,Y2=int(cx-side/2),int(cy-side/2),int(cx+side/2),int(cy+side/2)
    H,W=img.shape[:2]
    pad=cv2.copyMakeBorder(img,max(-Y1,0),max(Y2-H,0),max(-X1,0),max(X2-W,0),cv2.BORDER_CONSTANT,value=(114,114,114))
    oy,ox=max(-Y1,0),max(-X1,0)
    return cv2.resize(pad[Y1+oy:Y2+oy,X1+ox:X2+ox],(64,64),interpolation=cv2.INTER_AREA)


class Parity(unittest.TestCase):
    def test_columns_and_nms(self):
        rng=np.random.RandomState(0);cases=0
        for anchors in (0,17,4999,5000,6000):
            for batch in (1,3):
                for dtype in (torch.float32,torch.float16):
                    a=np.zeros((batch,84,anchors),np.float32)
                    a[:,:4]=rng.uniform(1,100,(batch,4,anchors))
                    # Sparse candidates, ties, strict detector threshold, inclusive min_conf.
                    for i,conf in enumerate((0.,.1,.25,.5,.5,1.)):
                        if i<anchors:a[:,4+GP.KEEP[i%len(GP.KEEP)],i]=conf
                    for minimum in (None,.25):
                        mode=Mode('fixture',128,128)
                        raw=torch.as_tensor(a,dtype=dtype)
                        for tensor in (raw,raw[0]):
                            before=GP.host_columns(tensor,mode,minimum)
                            after=host_columns(tensor,mode,minimum)
                            self.assertEqual(len(before),len(after))
                            for b,c in zip(before,after):
                                self.assertTrue(np.array_equal(b,c))
                                if anchors:
                                    db=GP.post_lean(b,mode,1.,(0,0),(128,128,3),minimum)
                                    dc=GP.post_lean(c,mode,1.,(0,0),(128,128,3),minimum)
                                    for key in db:self.assertTrue(np.array_equal(db[key],dc[key]))
                            cases+=1
        RESULTS.append(dict(check='columns_and_postprocess',cases=cases,mismatches=0,scope='CPU synthetic tensors; device parity pending'))

    def test_pixel_exact_crop(self):
        rng=np.random.RandomState(1);cases=0
        for H,W in ((9,11),(64,64),(73,121),(900,1600),(1550,2048)):
            img=rng.randint(0,256,(H,W,3),dtype=np.uint8)
            boxes=[[0,0,1,1],[-10,-20,30,40],[W-2,H-2,W+20,H+20],[-50,-50,-20,-10],[W+1,0,W+20,10],[.1,.9,8.1,8.9],[0,0,W,H]]
            for _ in range(50):
                x,y=rng.uniform(-20,W),rng.uniform(-20,H);w,h=rng.uniform(.01,100,2)
                boxes.append([x,y,x+w,y+h])
            for b in boxes:
                self.assertTrue(np.array_equal(reference_crop(img,b),crop64_fast(img,b)));cases+=1
        RESULTS.append(dict(check='existing_ROI_first_crop',cases=cases,mismatches=0,scope='CPU pixels; existing K29 optimization, not new'))

    def test_crop_bank_history(self):
        bank=Bank();a=np.full((31,29,3),7,np.uint8);b=a.copy();b[:]=99
        box=np.array([0.,0.,16.,16.]);bank.new_frame(0)
        p=bank.crop(0,a,box)
        self.assertTrue(np.array_equal(p,bank.crop(0,a,box)))
        bank.new_frame(1);q=bank.crop(1,b,box)
        self.assertFalse(np.array_equal(p,q))
        out=bank.crop_many([(1,b,box),(1,b,box),(0,a,box),(1,b,box+.25)])
        for got,item in zip(out,[(1,b,box),(1,b,box),(0,a,box),(1,b,box+.25)]):
            self.assertTrue(np.array_equal(got,crop64_fast(item[1],item[2])))
        bank.new_frame(3);self.assertTrue(all(k[0]>=1 for k in bank.crops))
        bank.reset();self.assertFalse(bank.crops)
        RESULTS.append(dict(check='existing_crop_bank_history',cases=8,mismatches=0,scope='CPU crop cache; no encoder or TTC reuse tested'))

    def test_nms_ties(self):
        xy=np.array([[0,0,10,10],[0,0,10,10],[20,20,30,30]],np.float32)
        scores=np.array([.5,.5,.5],np.float32)
        self.assertTrue(np.array_equal(nms(xy,scores,.5),[0,2]))
        self.assertTrue(np.array_equal(nms_matrix(xy,scores,.5),[0,2]))
        RESULTS.append(dict(check='existing_NMS_stable_ties',cases=2,mismatches=0,scope='CPU synthetic; frozen implementation unchanged'))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Parity))
    if not result.wasSuccessful():sys.exit(1)
    out=Path(os.environ.get('LATENCY_TEST_OUTPUT', str(ROOT/'docs/paper/handoff/LATENCY_OPT_20261030/PARITY_CPU.csv')))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        w=csv.DictWriter(f,fieldnames=['check','cases','mismatches','scope']);w.writeheader();w.writerows(RESULTS)
    candidate=Path(sys.modules[host_columns.__module__].__file__)
    out.with_suffix('.json').write_text(json.dumps(dict(script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),candidate_md5=hashlib.md5(candidate.read_bytes()).hexdigest(),cuda_used=False),indent=2)+'\n')
