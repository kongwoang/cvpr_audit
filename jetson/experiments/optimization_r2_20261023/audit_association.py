"""CPU-only identity audit of every original recovery attempt in diagnostic units."""
import json
import pickle
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from rap.mono import box_iou
from association import backward_box,margin_match

B=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR');HERE=B/'experiments/optimization_r2_20261023'
COMPAT={'vehicle':{'vehicle'},'person':{'person','cyclist'},'cyclist':{'cyclist','person'}}

def identity(box,cls,g):
    if not len(g):return -1,0.,False
    boxes=np.stack([g[k] for k in ('x1','y1','x2','y2')],axis=1)
    scores=box_iou(np.asarray(box)[None],boxes)[0]
    scores=np.where(np.array([str(c) in COMPAT[str(cls)] for c in g['coarse']]),scores,0)
    ids=np.argsort(-scores);best=ids[0];second=scores[ids[1]] if len(ids)>1 else 0
    return (int(g['track_id'][best]) if scores[best]>=.3 else -1,float(scores[best]),bool(scores[best]>=.5 and scores[best]-second>=.2))

def main():
    cv2.setNumThreads(1)
    out=HERE/'association';out.mkdir(exist_ok=True)
    cases=json.loads((HERE/'case_manifest_with_severity.json').read_text())['cases']
    chosen=sorted({(c['dataset'],c['unit']) for c in cases});nusc=None;rows=[]
    for ds,unit in chosen:
        if ds=='nusc':
            from rap.nusc import NuScenesDB,make_adapter
            from rap.paths import NUSCENES_TRAINVAL
            if nusc is None:
                db=NuScenesDB(NUSCENES_TRAINVAL,'v1.0-trainval');ad=make_adapter(db)
                nusc=db,ad,{db.scene_name(t):t for t in db.scenes}
            db,ad,by=nusc;toks=db.samples(by[unit]);geom=ad.geometry(unit)
            path=lambda fr:db.image_path(toks[fr])
        else:
            from vtf.av2 import AV2Log
            cam=AV2Log(B/'datasets/av2/sensor/train',unit);geom=cam.geometry();path=cam.image_path
        @lru_cache(maxsize=3)
        def image(fr):
            im=cv2.imread(str(path(fr)));assert im is not None
            return im
        def gt(fr):
            g=geom[geom['frame']==fr];return g[(g['y2']-g['y1'])>=10]
        records=pickle.load((HERE/'results/smoke'/ds/(unit+'.pkl')).open('rb'))
        for r in records:
            events=r['trace_v6S_b2A_retro']['retro_events']
            if not events:continue
            fr=int(r['frame']);current=events[0]['target'];cls=events[0]['coarse']
            origin,origin_iou,origin_strong=identity(current,cls,gt(fr))
            for name in ('original unique','rank margin','flow + rank margin'):
                target=current.copy();alive=True
                for ev in events:
                    lag=ev['lag'];det=ev['detections'];flow={};query=target
                    if not alive:break
                    if name.startswith('flow'):
                        prediction,flow=backward_box(image(fr-lag+1),image(fr-lag),target)
                        if prediction is not None:query=prediction
                    j=ev['match'] if name=='original unique' else margin_match(query,cls,det)
                    track,iou,strong=identity(det['xyxy'][j],cls,gt(fr-lag)) if j>=0 else (-1,0.,False)
                    prediction_id,prediction_iou,prediction_strong=identity(query,cls,gt(fr-lag))
                    rows.append(dict(dataset=ds,unit=unit,frame=fr,lag=lag,pipeline=name,
                        current_gt_track=origin,current_gt_iou=origin_iou,current_gt_strong=origin_strong,
                        selected=j,matched_gt_track=track,matched_gt_iou=iou,matched_gt_strong=strong,
                        identity_known=origin>=0 and track>=0,same_identity=origin>=0 and origin==track,
                        confident_switch=origin_strong and strong and origin!=track,
                        predicted_gt_track=prediction_id,predicted_gt_iou=prediction_iou,
                        predicted_same_identity=origin>=0 and origin==prediction_id,
                        **{'flow_'+k:v for k,v in flow.items() if k!='displacement'}))
                    if j>=0:target=det['xyxy'][j].copy()
                    else:alive=False
        print(ds,unit,'done',flush=True)
    d=pd.DataFrame(rows);d.to_csv(out/'events.csv',index=False)
    summary=[]
    for (ds,name),g in d.groupby(['dataset','pipeline']):
        first=g[g.lag==1]
        summary.append(dict(dataset=ds,pipeline=name,attempts=len(first),stored_steps=len(g),
            first_matches=int((first.selected>=0).sum()),first_known=int(first.identity_known.sum()),
            first_same=int(first.same_identity.sum()),first_confident_switch=int(first.confident_switch.sum()),
            all_confident_switch=int(g.confident_switch.sum()),flow_accepted=int(g.get('flow_success',pd.Series(dtype=bool)).fillna(False).sum()),
            flow_cpu_ms_mean=float(g.flow_cpu_ms.mean()) if 'flow_cpu_ms' in g else None))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    (out/'DONE.json').write_text(json.dumps(dict(status='complete',units=len(chosen),rows=len(d),
        limitation='Original historical crops/detections only; missing second steps are unavailable, not failures of a fresh method.'),indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':main()
