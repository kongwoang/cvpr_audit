"""All-row metrics and controls after the complete replay; no model selection."""
import argparse
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd
from common import HERE,REMOTE,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args()
    ds=a.dataset;folder=HERE/'results/readout'/ds
    prefix='v6S_b2A_missing_extent_retro'
    # An acquired-scene prior control, not a new R5 inference result.
    for f in sorted(folder.glob('*.pkl')):
        rows=pickle.load(f.open('rb'))
        old={r['frame']:r for r in pickle.load((REMOTE/'experiments/optimization_r3_20261023/results/low'/ds/f.name).open('rb'))}
        for r in rows:
            o=old[r['frame']]
            for k in ['a_gt','act_gt','a_cheap','J_cheap']:assert r[k]==o[k],(f,k)
            for key in ['a_','J_']:r[key+prefix+'|v3a']=o[key+prefix+'|v3a']
            for key in ['px_','ncalls_','dec_found_','dec_dem_']:
                if key+prefix in o:r[key+prefix]=o[key+prefix]
        tmp=f.with_suffix('.tmp');pickle.dump(rows,tmp.open('wb'));tmp.replace(f)
    out='evaluation_'+ds
    subprocess.run([sys.executable,str(HERE/'analyze.py'),'--stage','readout','--dataset-only',ds,'--out-name',out],check=True)
    subprocess.run([sys.executable,str(HERE/'attribution.py'),'--stage','readout','--report',out],check=True)
    obj=pd.read_csv(HERE/'reports/objects'/(ds+'.csv'));cal=[]
    for scope,sel in [('verified',np.ones(len(obj),bool)),('verified_missing',~obj.history),
        ('unmatched',~obj.annotation_matched),('raw_ref_disagreement',(obj.raw_demand>=1)&(obj.ref_demand<1)),
        ('median_disagreement',(obj.median_demand>=1)&(obj.ref_demand<1))]:
        for q in [.5,.8,.95]:
            for level,col in [(1,'p_decel'),(2,'p_hard')]:
                pred=sel&(obj[col]>=q);true=sel&(obj.gt_object_action>=level);tp=int((pred&true).sum())
                cal.append(dict(dataset=ds,model='all',scope=scope,q=q,level=level,samples=int(sel.sum()),
                    predicted=int(pred.sum()),required=int(true.sum()),correct=tp,
                    precision=tp/pred.sum() if pred.any() else np.nan,recall=tp/true.sum() if true.any() else np.nan))
    pd.DataFrame(cal).to_csv(HERE/'reports'/out/'dev_object_thresholds.csv',index=False)
    # Severity at individual frames, which an episode-level miss count can hide.
    details=[];command=np.array([0.,2.5,6.])
    for f in sorted(folder.glob('*.pkl')):
        d=pd.DataFrame(pickle.load(f.open('rb')));d=d[d['gt'].astype(bool)]
        gt=d.a_gt.to_numpy();hard=gt>=3.5
        for col in d:
            if not col.startswith('a_v6S'):continue
            act=d[col].to_numpy(int);risk=np.maximum(0,gt-command[act])**2
            details.append(dict(dataset=ds,unit=f.stem,pipeline_column=col,frames=len(d),hard_frames=int(hard.sum()),
                hard_not_hard=int((hard&(act<2)).sum()),hard_no_brake=int((hard&(act==0)).sum()),
                hard_loss=float(risk[hard].sum()),continuous_residual_at_correct_action=float(np.maximum(0,gt-command[d.act_gt.to_numpy(int)])[hard].dot(np.maximum(0,gt-command[d.act_gt.to_numpy(int)])[hard]))))
    pd.DataFrame(details).to_csv(HERE/'reports'/out/'severe_unit_metrics.csv',index=False)
    save(HERE/'reports'/out/'DONE.json',dict(status='complete',dataset=ds,units=len(list(folder.glob('*.pkl'))),
        confidence_or_model_selected=False,all_candidate_rows_reported=True))

if __name__=='__main__':main()
