"""CPU-only real-image case panels from immutable development traces."""
import csv
import hashlib
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rap import planner as P
from rap.mono import box_iou

B=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
HERE=B/'experiments/optimization_r2_20261023'
OUT=HERE/'cases'
COMPAT={'vehicle':{'vehicle'},'person':{'person','cyclist'},'cyclist':{'cyclist','person'}}

def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float) and not np.isfinite(value):return str(value)
    return value

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((HERE/'case_manifest_with_severity.json').read_text())
    pp=P.PlannerParams();nusc=None;summary=[]
    for case in manifest['cases']:
        ds,unit,fr=case['dataset'],case['unit'],case['frame']
        source=B/'experiments/w1_dev_20261023/results/dev'/ds/(unit+'.pkl')
        records={int(r['frame']):r for r in pickle.load(source.open('rb'))}
        chosen=[f for f in range(fr-2,fr+1) if f in records]
        if ds=='nusc':
            from rap.nusc import NuScenesDB,make_adapter
            from rap.paths import NUSCENES_TRAINVAL
            from rap.cache import Camera
            if nusc is None:
                db=NuScenesDB(NUSCENES_TRAINVAL,'v1.0-trainval');ad=make_adapter(db)
                nusc=(db,ad,{db.scene_name(t):t for t in db.scenes})
            db,ad,by=nusc;cam=Camera('nuScenes',unit);toks=db.samples(by[unit])
            geom=ad.geometry(unit)
            image_path=lambda f: Path(db.image_path(toks[f]))
        else:
            from vtf.av2 import AV2Log
            cam=AV2Log(B/'datasets/av2/sensor/train',unit)
            geom=cam.geometry();image_path=cam.image_path
        target=OUT/case['case_id'];target.mkdir(exist_ok=True)
        fig,axes=plt.subplots(1,len(chosen),figsize=(6*len(chosen),7),squeeze=False)
        packed=[]
        for ax,frame in zip(axes[0],chosen):
            r=records[frame];tr=r['trace_v6S'];raw=tr['raw'];ref=tr['refined']
            path=image_path(frame);img=cv2.imread(str(path));assert img is not None
            g=geom[geom['frame']==frame];g=g[(g['y2']-g['y1'])>=10]
            gtmono=tuple(np.asarray(g[k],float) for k in ['long_near','lat_min','lat_max','ttc'])
            demand,gd=P.required_decel(*gtmono,r['v'],pp)
            assert abs(demand-r['a_gt'])<1e-6,(ds,unit,frame,demand,r['a_gt'])
            a0,d0=P.required_decel(*raw,r['v'],pp);a1,d1=P.required_decel(*ref,r['v'],pp)
            assert P.discrete_action(a0,pp)==r['a_v6S|none']
            assert P.discrete_action(a1,pp)==r['a_v6S|v3a']
            ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));ax.axis('off')
            for crop in tr['crops']:
                x1,y1,x2,y2=crop
                ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec='#ffe45c',lw=1,ls='--'))
            ids=set(np.flatnonzero(np.asarray(d0)>=.5))|set(np.flatnonzero(np.asarray(d1)>=.5))
            decisive=int(np.argmax(gd)) if len(gd) else -1
            matched=-1
            if decisive>=0:
                gb=np.array([[g[decisive][k] for k in ['x1','y1','x2','y2']]])
                io=box_iou(gb,np.asarray(tr['xyxy']))[0]
                compatible=np.array([str(c) in COMPAT[str(g[decisive]['coarse'])] for c in tr['coarse']],dtype=bool)
                candidates=np.flatnonzero((io>=.3)&compatible)
                if len(candidates):matched=int(candidates[np.argmax(io[candidates])]);ids.add(matched)
                if demand>0:assert (matched>=0)==r['dec_found_v6S'],(ds,unit,frame,'match parity')
                x1,y1,x2,y2=gb[0]
                ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec='#03a9ff',lw=2.5))
            details=[]
            for i in sorted(ids):
                b=tr['xyxy'][i];x1,y1,x2,y2=b;full=bool(tr['from_full'][i]);chain=i in tr['chain_ids']
                color='#3aff72' if full else '#ff6fcb'
                ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec=color,lw=1.5))
                ax.text(x1,y1,str(i),color='black',fontsize=9,bbox=dict(fc=color,alpha=.8,pad=1))
                details.append(dict(index=i,coarse=str(tr['coarse'][i]),from_full=full,chain=chain,
                    raw_z=float(raw[0][i]),ref_z=float(ref[0][i]),raw_y=[float(raw[1][i]),float(raw[2][i])],
                    ref_y=[float(ref[1][i]),float(ref[2][i])],ttc=float(raw[3][i]),
                    raw_demand=float(d0[i]),ref_demand=float(d1[i]),matches_decisive=i==matched))
            label=f"frame {frame}: GT {r['act_gt']} | cheap {r['a_cheap']} | raw {r['a_v6S|none']} | ref {r['a_v6S|v3a']}"
            ax.set_title(label+'\n'+f"demand GT {demand:.3f}; raw {a0:.3f} → ref {a1:.3f}; ego {r['v']:.2f} m/s",fontsize=10)
            # Keep real frames for detailed inspection, not only an overlay rendering.
            cv2.imwrite(str(target/f'frame_{frame:03d}.jpg'),img)
            data=dict(frame=frame,record=r,ground_truth=g,image=str(path),objects=details,
                      decisive_gt_index=decisive,decisive_match=matched,camera_R=cam.transform('ego')[0],
                      camera_t=cam.transform('ego')[1],camera_P2=cam.calib.P2)
            packed.append(data)
            summary.append(dict(case_id=case['case_id'],dataset=ds,unit=unit,frame=frame,
                gt_action=r['act_gt'],cheap_action=r['a_cheap'],raw_action=r['a_v6S|none'],ref_action=r['a_v6S|v3a'],
                gt_demand=demand,raw_demand=a0,ref_demand=a1,matched_decisive=matched,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        fig.suptitle(case['case_id']+' — '+unit+'\nBlue: decisive GT; green: verified; pink: retained cheap; dashed yellow: executed crops',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.92))
        fig.savefig(target/'panel.png',dpi=140);fig.savefig(target/'panel.pdf');plt.close(fig)
        panel=cv2.imread(str(target/'panel.png'));ratio=min(1.,1800/panel.shape[1])
        cv2.imwrite(str(target/'preview.jpg'),cv2.resize(panel,None,fx=ratio,fy=ratio),[cv2.IMWRITE_JPEG_QUALITY,85])
        with (target/'records.pkl').open('wb') as f:pickle.dump(packed,f)
        report=dict(case=case,frames=[dict(frame=p['frame'],objects=p['objects'],
            decisive_gt=plain(p['ground_truth'][p['decisive_gt_index']]) if p['decisive_gt_index']>=0 else None,
            camera_t=plain(p['camera_t'])) for p in packed])
        (target/'details.json').write_text(json.dumps(plain(report),indent=2)+'\n')
        print(case['case_id'],'done',flush=True)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    (OUT/'DONE.json').write_text(json.dumps(dict(status='complete',cases=len(manifest['cases']),frames=len(summary)),indent=2)+'\n')

if __name__=='__main__':main()
