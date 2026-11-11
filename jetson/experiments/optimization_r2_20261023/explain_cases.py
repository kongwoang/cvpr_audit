"""Local numerical/visual evidence for identity swaps and coincidental decisions."""
import json
import pickle
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rap import planner as P
from rap.mono import box_iou
from vtf import dsf

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R2_20261023'
class Compat(pickle.Unpickler):
    def find_class(self,module,name):return super().find_class(module.replace('numpy._core','numpy.core'),name)
def read(p):
    with p.open('rb') as f:return Compat(f).load()
def gb(g):return np.stack([g[k] for k in ('x1','y1','x2','y2')],1)

def main():
    pp=P.PlannerParams();facts=[]
    cases=json.loads((HERE/'case_manifest_with_severity.json').read_text())['cases']
    for c in cases:
        item=read(OUT/'cases'/c['case_id']/'records.pkl')[-1]
        r=item['record'];g=item['ground_truth'][item['decisive_gt_index']]
        static=float(dsf._insert_demand(np.array([g['long_near']]),r['v'],pp,0.)[0])
        facts.append(dict(case_id=c['case_id'],frame=r['frame'],ego_mps=r['v'],gt_demand=r['a_gt'],
            gt_z=float(g['long_near']),gt_ttc=float(g['ttc']),
            zero_closing_probe_demand_at_same_range=static,
            zero_closing_probe_reaches_decel=static>=pp.thr_decel,
            caveat='range-only probe; excludes footprint width/grid/projection/occlusion'))
    pd.DataFrame(facts).to_csv(OUT/'case_static_probe.csv',index=False)

    case=OUT/'cases/01_nusc_decisive_verified_missing_chain';packed=read(case/'records.pkl')
    rows=read(HERE/'smoke_records/nusc/scene-0150.pkl');r=next(r for r in rows if r['frame']==29)
    events=r['trace_v6S_b2A_retro']['retro_events']
    selected={29:events[0]['target'],**{29-e['lag']:e['detections']['xyxy'][e['match']] for e in events}}
    records=[];fig,axes=plt.subplots(1,3,figsize=(12.5,4.4))
    for ax,item in zip(axes,packed):
        fr=item['frame'];g=item['ground_truth'];im=cv2.imread(str(case/f'frame_{fr:03d}.jpg'));assert im is not None
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set(xlim=(650,970),ylim=(575,395));ax.axis('off')
        for tid,col in [(21,'#00b8ff'),(25,'#ffb400')]:
            gg=g[g['track_id']==tid]
            if len(gg):
                x1,y1,x2,y2=gb(gg)[0]
                ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color=col,lw=2))
                ax.text(x1,y1-3,f'GT ID {tid}',color=col,fontsize=9)
        b=selected[fr];iou=box_iou(b[None],gb(g))[0];j=int(np.argmax(iou))
        assert int(g['track_id'][j])==(21 if fr==29 else 25)
        records.append(dict(case_id='01',frame=fr,selected_gt_track=int(g['track_id'][j]),iou=float(iou[j]),
            current_target_gt_track=21,is_current=fr==29,selected_box=b.tolist()))
        ax.add_patch(Rectangle(tuple(b[:2]),b[2]-b[0],b[3]-b[1],fill=False,color='#ff3258',lw=2,ls='--'))
        ax.set_title(f'Frame {fr}: selected detection matches ID {int(g["track_id"][j])}',fontsize=10)
    fig.suptitle('Recovery reports a chain, but the past detections belong to another vehicle\n'
        'Red dashed = selected detection; blue = current target GT ID 21; orange = GT ID 25',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.86));fig.savefig(OUT/'recovery_identity_switch.png',dpi=180);fig.savefig(OUT/'recovery_identity_switch.pdf');plt.close(fig)
    pd.DataFrame(records).to_csv(OUT/'recovery_identity_switch.csv',index=False)

    case=OUT/'cases/02_nusc_decisive_not_found';item=read(case/'records.pkl')[-1]
    g=item['ground_truth'];r=next(r for r in read(HERE/'smoke_records/nusc/scene-0366.pkl') if r['frame']==10)
    t=r['trace_v6S_b2A_pending_retro'];_,dem=P.required_decel(*t['refined'],r['v'],pp);i=int(np.argmax(dem))
    scores=box_iou(t['xyxy'][i:i+1],gb(g))[0];j=int(np.argmax(scores));target=g[j]
    td=P.required_decel(*tuple(np.array([target[k]]) for k in ('long_near','lat_min','lat_max','ttc')),r['v'],pp)[0]
    fact=dict(dataset='nusc',unit='scene-0366',frame=10,episode_rescued=True,trigger_predicted_demand=float(dem[i]),
        trigger_gt_track=int(target['track_id']),trigger_gt_iou=float(scores[j]),trigger_gt_demand=float(td),
        trigger_predicted_ttc=float(t['refined'][3][i]),trigger_gt_ttc=float(target['ttc']),
        decisive_gt_track=int(g[item['decisive_gt_index']]['track_id']),decisive_gt_demand=float(r['a_gt']),
        decisive_found=bool(r['dec_found_v6S_b2A_pending_retro']),
        interpretation='Action correct by compensating error on a different non-required object; not recovery of decisive object.')
    assert fact['trigger_gt_iou']>.7 and fact['trigger_gt_demand']<1 and not fact['decisive_found']
    (OUT/'coincidental_rescue.json').write_text(json.dumps(fact,indent=2)+'\n')
    print('Confirmed ID switch, coincidental rescue, and all nine range-only static-probe counterfactuals')

if __name__=='__main__':main()
