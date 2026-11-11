"""K34 CPU replay from cached detections and C1 TTC, with threshold-specific support.
v3a changes z/lateral coordinates, never TTC: reconstruct pre-refiner spatial
geometry from boxes/calibration, retain the cached C1 TTC, verify base actions.
No detector or C1 inference, no changes to frozen modules. Also extracts annotated
decisive-object metadata for the later K34-2 analysis (not coverage aggregation).
"""
import argparse
import pickle
import time
from dataclasses import replace
import cv2
import numpy as np
import pandas as pd
import torch
from rap import planner as P
from rap.mono import predicted_geometry
from vtf import speculative as SP
import k24_test as K24
from nusc_refiner_crop import crop64
from k34_common import *

GRID=(.6,.8,1.,1.2,1.5)
CLS={'vehicle':0,'person':1,'cyclist':2}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--shard',default='0/1')
    ap.add_argument('--max_units',type=int,default=0);ap.add_argument('--tag',default='');args=ap.parse_args()
    ds=args.dataset;si,sn=map(int,args.shard.split('/'));torch.set_num_threads(1);cv2.setNumThreads(1)
    od=OUT/('replay'+('_'+args.tag if args.tag else ''))/ds;od.mkdir(parents=True,exist_ok=True)
    checkpoint=B/'out/resnet18_v3a/image_refiner.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    net=K24.KT.Res18x3(len(ck['fmu']));net.load_state_dict(ck['state']);net=net.eval()
    pp=P.PlannerParams();pps={t:replace(pp,thr_decel=t) for t in GRID}
    capture={}
    if ds=='nusc':
        import rap.nusc as NS
        original=NS.make_adapter
        def wrap(db):
            capture['db']=db;capture['by']={db.scene_name(t):t for t in db.scenes}
            return original(db)
        NS.make_adapter=wrap
    H,W={'nusc':(900,1600),'av2':(2048,1550),'ahd':(1200,1920)}[ds]
    nfinished=0
    for ui,(unit,cam,gen) in enumerate(K24.units(ds)):
        if ui%sn!=si:continue
        target=od/(unit+'.pkl')
        if target.exists():continue
        if args.max_units and nfinished>=args.max_units:break
        tic=time.perf_counter();src=B/'results/k30/apdump'/ds/(unit+'.pkl')
        if not src.exists():continue  # final completeness gate checks the frozen unit manifest
        dump=pickle.load(src.open('rb'));ar={int(r['frame']):r for r in dump['rows']};dets=dump['dets']
        R,t=cam.transform('ego');dt=float(cam.dt);rows=[];annrows=[];checks=[]
        expected={int(r['frame']) for r in dump['rows'] if r['gt']}
        for fr,path,v,g,cached in gen():
            if fr not in expected:continue
            assert fr in ar
            row={'unit':unit,'frame':int(fr),'v':float(v)};img=None
            cheap=dets[(fr,'cheap')]['sc']
            row['demand_cheap']=P.required_decel(*cheap['geo'],v,pp)[0]
            row['a_cheap']=P.discrete_action(row['demand_cheap'],pp)
            assert row['a_cheap']==ar[fr]['a_cheap'],(unit,fr,'cheap')
            for key in ('wf960+C1',f'wf{NWS[ds]}+C1'):
                sc=dets[(fr,key)]['sc'];xy=np.asarray(sc['xyxy'],float).reshape(-1,4)
                geom=predicted_geometry({'xyxy':xy,'coarse':sc['coarse']},None,cam.calib,cam.cam_h,dt,cam_to_ego=(R,t))
                if ds=='av2' and len(xy):
                    mask=np.maximum(xy[:,3]-np.maximum(xy[:,1],1830),0)/np.maximum(xy[:,3]-xy[:,1],1e-6)>=.5
                    for k in ('z','z_ground','z_height','ttc'):
                        geom[k]=np.asarray(geom[k],float).copy();geom[k][mask]=np.inf
                mono=tuple(np.asarray(geom[k],float) for k in ('z','lat_min','lat_max'))+(np.asarray(sc['geo'][3],float),)
                demand=P.required_decel(*mono,v,pp)[0];row['demand_'+key+'|none']=demand
                a=P.discrete_action(demand,pp)
                assert a==ar[fr]['a_'+key+'|none'],(unit,fr,key,'pre-refiner',a,ar[fr]['a_'+key+'|none'])
                supports={th:SP.support(mono,v,p) for th,p in pps.items()}
                union=sorted(set(int(i) for sup in supports.values() for i in sup));ratios={}
                if union:
                    if img is None:img=cv2.imread(path)
                    assert img is not None,(unit,fr,path)
                    assert img.shape[:2]==(H,W),(ds,img.shape)
                    F=[]
                    for i in union:
                        x1,y1,x2,y2=xy[i];zg=float(geom['z_ground'][i]);zh=float(geom['z_height'][i])
                        F.append([x1/W,y1/H,x2/W,y2/H,(x2-x1)/W,(y2-y1)/H,float(sc['conf'][i]),
                                  float(CLS[str(sc['coarse'][i])]),float(mono[0][i]),zg,zh,float(np.log(max(zg,.5)/max(zh,.5))),1.])
                    feats=((np.array(F,np.float32)-ck['fmu'])/ck['fsd']).astype(np.float32)
                    crops=torch.from_numpy(np.stack([crop64(img,xy[i]) for i in union])).permute(0,3,1,2).float()/255.
                    with torch.no_grad():pred=net(crops,torch.from_numpy(feats)).float().numpy().reshape(len(union),-1)
                    ratios={i:float(np.exp(pred[j,0])) for j,i in enumerate(union)}
                for th,p in pps.items():
                    tt=[x.copy() for x in mono]
                    for i in supports[th]:
                        for x in tt[:3]:x[i]*=ratios[i]
                    req=P.required_decel(*tt,v,p)[0]
                    row[f'demand_{key}|v3a@{th:g}']=req;row[f'a_{key}|v3a@{th:g}']=P.discrete_action(req,p)
                # Record cross-device refiner parity; never silently change a stored action.
                checks.append({'unit':unit,'frame':fr,'pipeline':key,'noref_matches_apdump':True,
                               'ref_matches_apdump':row[f'a_{key}|v3a@1']==ar[fr]['a_'+key+'|v3a']})
            # Exact annotated relative range rate, including receding objects and unavailable rates.
            if len(g):
                demand,each=P.required_decel(*(g[k] for k in ('long_near','lat_min','lat_max','ttc')),v,pp)
                if P.discrete_action(demand,pp)>=1:
                    idx=int(np.argmax(each));obj=g[idx];rel=np.flatnonzero(each>=pp.thr_decel)
                    boxes=np.stack([g[k][rel] for k in ('x1','y1','x2','y2')],1)
                    match=K24.IE.matched(boxes,g['coarse'][rel],cheap['xyxy'],cheap['coarse'])
                    typ=str(obj['type'])
                    if ds=='nusc':
                        raw=capture['db'].annotations_for(capture['by'][unit],fr)
                        typ=raw[int(obj['track_id'])]['category']
                    annrows.append({'unit':unit,'frame':fr,'demand_gt':float(demand),'dec_index':idx,
                                    'raw_class':typ,'closing_speed_mps':float(-obj['range_rate']) if obj['has_rate'] else np.nan,
                                    'has_rate':bool(obj['has_rate']),'cheap_detected':bool(match[np.flatnonzero(rel==idx)[0]]>=0),
                                    'box':np.array([obj[k] for k in ('x1','y1','x2','y2')],float)})
            rows.append(row)
        assert {r['frame'] for r in rows}==expected,(unit,'missing frames')
        payload={'rows':rows,'annotations':annrows,'checks':checks,'source':str(src),'source_sha256':sha(src),
                 'checkpoint_sha256':sha(checkpoint),'code_sha256':sha(__file__),'elapsed_s':time.perf_counter()-tic,
                 'host':socket.gethostname()}
        temp=Path(str(target)+'.tmp');temp.write_bytes(pickle.dumps(payload));temp.replace(target)
        nfinished+=1
        print(ds,unit,'frames',len(rows),'seconds',round(payload['elapsed_s'],1),
              'ref_disagreements',sum(not x['ref_matches_apdump'] for x in checks),flush=True)
    done(od/f'SHARD_{si}_of_{sn}.json',dataset=ds,shard=args.shard,tag=args.tag,code_sha256=sha(__file__))
    print('K34 REPLAY SHARD DONE',ds,args.shard,flush=True)

if __name__=='__main__':main()
