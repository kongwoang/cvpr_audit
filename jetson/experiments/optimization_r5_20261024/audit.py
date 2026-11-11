"""Population failure signatures; labels are diagnostics, not causal certification."""
import argparse
import importlib.util
import pickle
import sys
import numpy as np
import pandas as pd
from common import HERE, REMOTE, save
from features import PP, action, complete_history
from rap import planner as P
sys.path.insert(0,str(REMOTE/'scripts/A_method'))
from k31_episodes import runs
s=importlib.util.spec_from_file_location('k27_r5audit',REMOTE/'scripts/A_method/k27_dev.py')
K=importlib.util.module_from_spec(s);s.loader.exec_module(K)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args()
    out=HERE/'reports/audit';out.mkdir(parents=True,exist_ok=True);frames=[];eps=[]
    cmd=np.array([0.,2.5,6.]);cp=P.CostParams()
    for f in sorted((REMOTE/'experiments/optimization_r4_20261024/results/dev'/a.dataset).glob('*.pkl')):
        rows=pickle.load(f.open('rb'));unit=[]
        for r in rows:
            if not r['gt']:continue
            t=r['trace_v6S_b2A_retro'];c=r['trace_cheap'];v=r['v'];dec=int(r['dec'])
            _,raw=P.required_decel(*t['raw'],v,PP);_,ref=P.required_decel(*t['refined'],v,PP)
            _,cheap=P.required_decel(*c['mono'],v,PP)
            act=r['a_v6S_b2A_retro|v3a'];ra=r['a_v6S_b2A_retro|none'];gt=r['act_gt'];history=complete_history(t)
            j=int(K.IE.matched(r['gt_boxes'][dec:dec+1],r['gt_classes'][dec:dec+1],t['xyxy'],t['coarse'])[0]) if dec>=0 else -1
            cj=int(K.IE.matched(r['gt_boxes'][dec:dec+1],r['gt_classes'][dec:dec+1],c['xyxy'],c['coarse'])[0]) if dec>=0 else -1
            row=dict(dataset=a.dataset,unit=r['unit'],frame=r['frame'],v=v,gt_demand=r['a_gt'],gt_action=gt,
                action=act,raw_action=ra,cheap_action=r['a_cheap'],found=j>=0,cheap_found=cj>=0,
                raw_dec_demand=raw[j] if j>=0 else np.nan,ref_dec_demand=ref[j] if j>=0 else np.nan,
                cheap_dec_demand=cheap[cj] if cj>=0 else np.nan,
                insufficient=max(0,r['a_gt']-cmd[act])**2,raw_insufficient=max(0,r['a_gt']-cmd[ra])**2,
                correct_action_residual=max(0,r['a_gt']-cmd[gt])**2,
                overbraking=.12*max(0,cmd[act]-r['a_gt'])**2,
                raw_overbraking=.12*max(0,cmd[ra]-r['a_gt'])**2,
                history=bool(history[j]) if j>=0 else False,from_full=bool(t['from_full'][j]) if j>=0 else False,
                raw_in_corridor=bool(t['raw'][1][j]<1.2 and t['raw'][2][j]>-1.2) if j>=0 else False,
                ref_in_corridor=bool(t['refined'][1][j]<1.2 and t['refined'][2][j]>-1.2) if j>=0 else False,
                raw_z=float(t['raw'][0][j]) if j>=0 else np.nan,raw_ttc=float(t['raw'][3][j]) if j>=0 else np.nan,
                raw_lmin=float(t['raw'][1][j]) if j>=0 else np.nan,raw_lmax=float(t['raw'][2][j]) if j>=0 else np.nan,
                ref_z=float(t['refined'][0][j]) if j>=0 else np.nan,
                ref_lmin=float(t['refined'][1][j]) if j>=0 else np.nan,ref_lmax=float(t['refined'][2][j]) if j>=0 else np.nan)
            thr=1. if gt==1 else 3.5
            row['refiner_dec_downgrade']=bool(j>=0 and raw[j]>=thr and ref[j]<thr)
            row['frame_refiner_downgrade']=ra>=gt and act<gt
            if act>=gt:signature='adequate_discrete_action'
            elif j<0:signature='decisive_detection_absent'
            elif row['refiner_dec_downgrade']:signature='refiner_threshold_downgrade'
            elif not row['raw_in_corridor']:signature='raw_lateral_exclusion'
            elif row['raw_z']>80:signature='raw_out_of_range'
            elif row['from_full'] and not row['history']:signature='verified_history_missing'
            else:signature='remaining_geometry_or_rate_underestimate'
            row['signature']=signature;unit.append(row);frames.append(row)
        d=pd.DataFrame(unit)
        for ei,(start,length) in enumerate(runs(d.gt_action.to_numpy()>=1)):
            q=d.iloc[start:start+length];hit=q.action.to_numpy()>=1
            eps.append(dict(dataset=a.dataset,unit=f.stem,episode=ei,start_frame=int(q.frame.iloc[0]),end_frame=int(q.frame.iloc[-1]),
                frames=len(q),missed=not hit.any(),peak_demand=q.gt_demand.max(),insufficient=q.insufficient.sum(),
                detection_absent_frames=int((~q.found).sum()),history_missing_frames=int((q.found&q.from_full&~q.history).sum()),
                lateral_exclusion_frames=int((q.found&~q.raw_in_corridor).sum()),
                refiner_dec_downgrade_frames=int(q.refiner_dec_downgrade.sum()),
                refiner_frame_downgrade_frames=int(q.frame_refiner_downgrade.sum())))
    d=pd.DataFrame(frames);d.to_csv(out/(a.dataset+'_frames.csv'),index=False)
    pd.DataFrame(eps).to_csv(out/(a.dataset+'_episodes.csv'),index=False)
    summary=[]
    for scope,sel in [('all',np.ones(len(d),bool)),('required',d.gt_action>=1),('hard_required',d.gt_action>=2),('missed_braking', (d.gt_action>=1)&(d.action==0))]:
        q=d[sel]
        for sig,g in q.groupby('signature'):
            summary.append(dict(dataset=a.dataset,scope=scope,signature=sig,frames=len(g),units=g.unit.nunique(),
                insufficient_loss=g.insufficient.sum(),loss_share=g.insufficient.sum()/q.insufficient.sum() if q.insufficient.sum() else np.nan,
                correct_action_residual=g.correct_action_residual.sum()))
    pd.DataFrame(summary).to_csv(out/(a.dataset+'_signatures.csv'),index=False)
    save(out/(a.dataset+'_DONE.json'),dict(status='complete',units=d.unit.nunique(),frames=len(d),
         required=int((d.gt_action>=1).sum()),missed_braking_frames=int(((d.gt_action>=1)&(d.action==0)).sum()),
         refiner_harm_frames=int((d.insufficient>d.raw_insufficient).sum()),
         refiner_help_frames=int((d.insufficient<d.raw_insufficient).sum()),
         refiner_risk_delta=float((d.insufficient-d.raw_insufficient).sum()),
         refiner_overbraking_delta=float((d.overbraking-d.raw_overbraking).sum())))
    print(a.dataset,'audit done',len(d),flush=True)

if __name__=='__main__':main()
