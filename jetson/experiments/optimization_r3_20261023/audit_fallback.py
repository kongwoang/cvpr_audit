"""Relate changed frame decisions to the winning object's actual fallback evidence."""
import argparse
import pickle
import numpy as np
import pandas as pd
from rap import planner as P
from common import HERE,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['low','compact'],required=True);ap.add_argument('--dataset-only',choices=['nusc','av2']);a=ap.parse_args()
    out=HERE/'reports'/('fallback_'+a.stage);out.mkdir(parents=True,exist_ok=True)
    pp=P.PlannerParams();rows=[]
    for ds in ([a.dataset_only] if a.dataset_only else ['nusc','av2']):
        files=sorted((HERE/'results'/a.stage/ds).glob('*.pkl'));assert len(files)=={'nusc':152,'av2':300}[ds]
        for f in files:
            for r in pickle.load(f.open('rb')):
                if not r['gt']:continue
                baseline=r['a_v6S_b2A_retro|v3a']
                for key,evidence in r.items():
                    if not key.startswith('fallback_evidence_'):continue
                    prefix=key[len('fallback_evidence_'):];action=r['a_'+prefix+'|v3a']
                    if action==baseline:continue
                    tr=r['trace_'+prefix];_,dem=P.required_decel(*tr['refined'],r['v'],pp)
                    idx=int(np.argmax(dem)) if len(dem) else -1;ev=None
                    if idx>=0:
                        matching=[e for e in evidence if np.array_equal(e['box'],tr['xyxy'][idx])]
                        ev=matching[-1] if matching else None
                    event='new benchmark false-braking frame' if r['act_gt']==0 and action>=1 and baseline==0 else ('new required braking frame' if r['act_gt']>=1 and action>=1 and baseline==0 else 'other action change')
                    rows.append(dict(dataset=ds,unit=r['unit'],frame=r['frame'],pipeline=prefix,event=event,gt_action=r['act_gt'],baseline_action=baseline,action=action,
                        winning_object_has_fallback=ev is not None,observations=ev['observations'] if ev else -1,
                        used_upper=bool(ev['use_upper']) if ev else False,rate=ev['rate'] if ev else np.nan,
                        max_predicted_demand=dem[idx] if idx>=0 else 0,winning_class=tr['coarse'][idx] if idx>=0 else 'none'))
    d=pd.DataFrame(rows);d.to_csv(out/'changed_frames.csv',index=False)
    summary=d.groupby(['dataset','pipeline','event','winning_object_has_fallback','observations','used_upper'],dropna=False).agg(frames=('frame','size'),units=('unit','nunique'),mean_rate=('rate','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    (out/'REPORT.md').write_text('# Fallback evidence on changed frames\n\nWinning object = maximal final predicted demand; match to actual fallback box is exact. Observations are actual current/past detections, including causal cheap history.\n\nThis diagnostic describes evidence used; it is not a causal intervention or proof of correct identity. Benchmark false frames can contain unannotated objects. CSVs retain all action changes, including weaker decisions.\n')
    save(out/'DONE.json',dict(status='complete',stage=a.stage,datasets=sorted(d.dataset.unique()),changed_frames=len(d)))
    print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
