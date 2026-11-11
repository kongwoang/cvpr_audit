"""Replay frozen v3a on changed episodes where the stored trace was joint-refined."""
import argparse
import importlib.util
import os
import pickle
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE,REMOTE,save,source_hashes
from rap import planner as P
from rap.mono import predicted_geometry
from vtf import speculative as SP

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['nusc','av2'],required=True);ap.add_argument('--stage',choices=['learned','calibrated'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);before=source_hashes();B=Path(os.environ['B']);sys.path.insert(0,str(B/'scripts/common'))
    from nusc_refiner_crop import crop64
    spec=importlib.util.spec_from_file_location('k27_audit',B/'scripts/A_method/k27_dev.py');K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
    ck=torch.load(REMOTE/'out/resnet18_v3a/image_refiner.pt',map_location='cpu',weights_only=False)
    net=K.KT.Res18x3(len(ck['fmu']));net.load_state_dict(ck['state']);net.eval();pp=P.PlannerParams()
    e=pd.read_csv(HERE/'reports'/a.stage/'episodes.csv');e=e[e.dataset==a.dataset]
    names={'learned geometric':'v6S_b2A_allgeo_retro','learned decision':'v6S_b2A_alldec_retro','uncertainty query':'v6S_b2A_query_retro'}
    base=e[e.pipeline=='round1 combined'].set_index(['unit','start_frame']);changes=[]
    for label,prefix in names.items():
        q=e[e.pipeline==label].set_index(['unit','start_frame']).join(base[['missed']],rsuffix='_baseline')
        for (unit,start),ep in q[q.missed!=q.missed_baseline].iterrows():changes.append(dict(unit=unit,start=int(start),end=int(ep.end_frame),peak=ep.peak_demand,pipeline=label,prefix=prefix,transition='newly_missed' if ep.missed else 'rescued'))
    wanted={c['unit'] for c in changes};evidence={};checks=[]
    for unit,cam,gen in K.units(a.dataset):
        if unit not in wanted:continue
        records={int(r['frame']):r for r in pickle.load((HERE/'results'/a.stage/a.dataset/(unit+'.pkl')).open('rb'))}
        needs={}
        for c in changes:
            if c['unit']==unit:
                for fr in range(c['start'],c['end']+1):needs.setdefault(fr,set()).add(c['prefix'])
        R,t=cam.transform('ego')
        for fr,path,v,g,cached in gen():
            if fr not in needs:continue
            img=cv2.imread(path);assert img is not None;H,W=img.shape[:2];r=records[int(fr)]
            for prefix in sorted(needs[fr]):
                tr=r['trace_'+prefix];raw=tuple(np.asarray(x).copy() for x in tr['raw']);tt=[x.copy() for x in raw]
                sup=[i for i in SP.support(raw,v,pp) if tr['from_full'][i]]
                if sup:
                    zz=predicted_geometry({'xyxy':tr['xyxy'][sup],'coarse':tr['coarse'][sup]},None,cam.calib,cam.cam_h,float(cam.dt),cam_to_ego=(R,t));fe=[]
                    for j,i in enumerate(sup):
                        x1,y1,x2,y2=tr['xyxy'][i];zg,zh=zz['z_ground'][j],zz['z_height'][j]
                        fe.append([x1/W,y1/H,x2/W,y2/H,(x2-x1)/W,(y2-y1)/H,tr['conf'][i],{'vehicle':0,'person':1,'cyclist':2}[str(tr['coarse'][i])],raw[0][i],zg,zh,np.log(max(zg,.5)/max(zh,.5)),1.])
                    fe=((np.array(fe,np.float32)-ck['fmu'])/ck['fsd']).astype(np.float32)
                    images=torch.from_numpy(np.stack([crop64(img,tr['xyxy'][i]) for i in sup])).permute(0,3,1,2).float()/255
                    with torch.no_grad():factors=np.exp(net(images,torch.from_numpy(fe)).numpy()[:,0])
                    for i,factor in zip(sup,factors):
                        for x in tt[:3]:x[i]*=float(factor)
                maximum,dem=P.required_decel(*tt,v,pp);action=P.discrete_action(maximum,pp);expected=int(r['a_'+prefix+'|v3a'])
                dec=int(r['dec']);match=int(K.IE.matched(r['gt_boxes'][dec:dec+1],r['gt_classes'][dec:dec+1],tr['xyxy'],tr['coarse'])[0]) if dec>=0 else -1
                row=dict(dataset=a.dataset,stage=a.stage,unit=unit,frame=int(fr),prefix=prefix,action=action,stored_action=expected,action_equal=action==expected,gt_action=r['act_gt'],dec_found=match>=0,dec_demand=float(dem[match]) if match>=0 else np.nan)
                checks.append(row);evidence[unit,int(fr),prefix]=row
        print(a.stage,a.dataset,unit,'done',flush=True)
    out=HERE/'reports/v3a_attribution';out.mkdir(parents=True,exist_ok=True);tag=a.stage+'_'+a.dataset
    pd.DataFrame(checks).to_csv(out/(tag+'_frames.csv'),index=False)
    rows=[]
    for c in changes:
        relevant=[evidence[c['unit'],fr,c['prefix']] for fr in range(c['start'],c['end']+1) if (c['unit'],fr,c['prefix']) in evidence]
        assert relevant and all(r['action_equal'] for r in relevant),'CPU/GPU refiner action discrepancy; inspect frame CSV'
        supported=[r for r in relevant if r['action']>=1 and r['gt_action']>=1 and r['dec_found'] and r['dec_demand']>=1]
        hard=[r for r in relevant if r['action']>=2 and r['gt_action']>=2 and r['dec_found'] and r['dec_demand']>=3.5]
        rows.append(dict(dataset=a.dataset,stage=a.stage,**c,any_decisive_supported_brake=bool(supported),any_decisive_supported_hard_brake=bool(hard)))
    pd.DataFrame(checks).to_csv(out/(tag+'_frames.csv'),index=False);pd.DataFrame(rows).to_csv(out/(tag+'_episodes.csv'),index=False)
    summary=[]
    for label in names:
        own=[r for r in rows if r['pipeline']==label]
        if not own:continue
        rescued=[r for r in own if r['transition']=='rescued'];lost=[r for r in own if r['transition']=='newly_missed']
        summary.append(dict(dataset=a.dataset,stage=a.stage,pipeline=label,rescued=len(rescued),newly_missed=len(lost),rescued_with_decisive_evidence=sum(r['any_decisive_supported_brake'] for r in rescued),rescued_without_decisive_evidence=sum(not r['any_decisive_supported_brake'] for r in rescued),hard_peak_rescued=sum(r['peak']>=3.5 for r in rescued),hard_peak_rescued_with_hard_evidence=sum(r['peak']>=3.5 and r['any_decisive_supported_hard_brake'] for r in rescued)))
    pd.DataFrame(summary).to_csv(out/(tag+'_summary.csv'),index=False)
    assert source_hashes()==before
    save(out/(tag+'_DONE.json'),dict(status='complete',frames=len(checks),changed_episodes=len(rows),cpu_refiner_action_differences=sum(not x['action_equal'] for x in checks),frozen_hashes_unchanged=True))
    (out/'REPORT.md').write_text('# v3a-specific changed-episode attribution\n\nEarlier traces stored the last (joint-adapt) refinement for all-object/query policies. This audit replays the frozen v3a refiner on the actual stored final raw scenes and original current images, on CPU only, using the same support and features. Every audited action must match the stored GPU v3a action. No acquisition, histories, checkpoint, thresholds or aggregate score changes. All changed episodes vs round1, including lost episodes, are included. A matched decisive object that brakes is evidence, not an identity or safety guarantee.\n')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__':main()
