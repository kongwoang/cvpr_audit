"""Fixed-ego/camera/candidates scene transplant diagnostic; CPU only, no planner rollout."""
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from k41_scene_diagnostic import K, X, DATA, union_area, iou
from vtf import speculative as SP

B=Path(os.environ['B']);OUT=B/'results/k41/scene_controlled'
sys.path.insert(0,str(B/'results/k40r/vendor'))
from analysis.common import Bootstrap


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gz=float((X/'dets/ground_z.txt').read_text());priors=pickle.loads((X/'dets/size_priors.pkl').read_bytes())
    manifest=[];rows=[]
    for f in sorted((B/'results/k31/p2').glob('*.pkl')):
        inputs=[f,X/'dets_k30b'/f.name,X/'k30_center'/f.name,DATA/'navsim_logs/test'/f.name]
        for p in inputs:manifest.append(dict(path=str(p),md5=hashlib.md5(p.read_bytes()).hexdigest()))
        prev=pd.DataFrame(pickle.loads(f.read_bytes())).set_index('token')
        dets=pickle.loads(inputs[1].read_bytes());cen=pickle.loads(inputs[2].read_bytes())
        frames={r['token']:r for r in pickle.loads(inputs[3].read_bytes())};tokens=sorted(prev.index)
        scenes={}
        for tok in tokens:
            rec,fr=dets[tok],frames[tok];cam=K.NP.Cam(*rec['cam'])
            base,per=K.branch_objects(fr['anns'],rec['cheap'],cam,gz,priors)
            scenes[tok]=K.Scene(fr['anns'],base,zip(per,np.asarray(rec['cheap']['xyxy'],float).reshape(-1,4))).o
        for a,b in zip(tokens[::2],tokens[1::2]):
            fr,rec=frames[a],dets[a];v=float(np.hypot(*np.asarray(fr['ego_dynamic_state'],float)[:2]))
            cl=cen.get(a);sel=K.Selector(cl if isinstance(cl,np.ndarray) and len(cl)>=2 else None,v)
            cam=K.NP.Cam(*rec['cam']);prb=K.probes(cam,gz,priors)
            Hp=sel.h(np.asarray([r for r,_ in prb]).reshape(-1,7))
            fields=[];crops=[];actions=[];scores=[]
            for tok in (a,b):
                s,_=sel.score(scenes[tok]);act=int(s.argmax());m=np.minimum(s[:,None],Hp).argmax(0)!=act
                fields.append(m);crops.append(K.crops_of([ib for yes,(_,ib) in zip(m,prb) if yes],[],[],SP,1920,1080));actions.append(act);scores.append(s)
            assert actions[0]==int(prev.loc[a,'pi_cheap']) and int(fields[0].sum())==int(prev.loc[a,'field_cells_r1'])
            # Non-equivalent fixed geometric trajectory footprint (all 15 trajectories, 4s).
            geo=[]
            for pose in sel.P[:,:41:5].reshape(-1,3):
                ib=cam.project(np.array([pose[0],pose[1],gz+.8,K.EGO_L,K.EGO_W+1.,1.6,pose[2]]))
                if ib is not None:geo.append(ib)
            geometric=K.crops_of(geo,[],[],SP,1920,1080)
            union=fields[0]|fields[1]
            rows.append(dict(log=f.stem,receiver=a,donor=b,speed=v,receiver_action=actions[0],donor_action=actions[1],
                same_action=actions[0]==actions[1],field_differs=bool(np.any(fields[0]!=fields[1])),field_iou=float((fields[0]&fields[1]).sum()/union.sum()) if union.any() else 1.,
                crop_iou=iou(crops[0],crops[1]),crop_differs=not(len(crops[0])==len(crops[1]) and all(np.array_equal(x,y) for x,y in zip(crops[0],crops[1]))),
                field_area_receiver=union_area(crops[0])/(1920*1080),field_area_donor=union_area(crops[1])/(1920*1080),
                geometric_area=union_area(geometric)/(1920*1080),geometric_crop_changes=0,probe_count=len(prb),
                field_queries=2*(len(prb)+1),actual_selector_reductions=2,detector_calls=0,
                mask_receiver=np.packbits(fields[0]).tobytes().hex(),mask_donor=np.packbits(fields[1]).tobytes().hex()))
        print(f.stem,'pairs',len(tokens)//2,flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'pairs.csv',index=False);pd.DataFrame(manifest).to_csv(OUT/'sources.csv',index=False)
    bt=Bootstrap(d.log);out=[]
    metrics=['field_differs','field_iou','crop_iou','crop_differs','field_area_receiver','field_area_donor','geometric_area']
    for subset,m in [('all',np.ones(len(d),bool)),('same_action',d.same_action.to_numpy(bool))]:
        den=bt.sums(m);bd=den[bt.idx].sum(1)
        for metric in metrics:
            num=bt.sums(m*d[metric].to_numpy(float));z=num[bt.idx].sum(1)[bd>0]/bd[bd>0];lo,hi=np.percentile(z,[2.5,97.5])
            out.append(dict(subset=subset,metric=metric,value=num.sum()/den.sum(),lo=lo,hi=hi,pairs=int(m.sum()),logs=d.loc[m,'log'].nunique(),resampling_logs=d.log.nunique()))
    pd.DataFrame(out).to_csv(OUT/'summary.csv',index=False)
    (OUT/'run.json').write_text(json.dumps(dict(script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),pairs=len(d),tokens_used=len(d)*2,scope='Synthetic cached scene transplant; identical receiver ego/candidate/camera/probes; no real crops or downstream scores'),indent=2)+'\n')


if __name__=='__main__':main()
