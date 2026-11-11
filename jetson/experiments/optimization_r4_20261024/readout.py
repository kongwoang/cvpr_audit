"""Exact fixed final-readout policies on stored R3 round1 scenes (CPU only)."""
import argparse
import importlib.util
import json
import pickle
import sys
import numpy as np
from common import HERE,REMOTE,save,source_hashes
from rap import planner as P
from vtf import speculative as SP
import simple

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['nusc','av2'],required=True);a=ap.parse_args()
    before=source_hashes();pp,cp=P.PlannerParams(),P.CostParams();src=REMOTE/'experiments/optimization_r3_20261023/results/dev'/a.dataset
    # Reuse the frozen annotated matcher only for diagnostic attribution.
    sys.path.insert(0,str(REMOTE/'scripts/A_method'))
    s=importlib.util.spec_from_file_location('k27_readout',REMOTE/'scripts/A_method/k27_dev.py');K=importlib.util.module_from_spec(s);s.loader.exec_module(K)
    out=HERE/'results/readout'/a.dataset;out.mkdir(parents=True,exist_ok=True);audit=[]
    for f in sorted(src.glob('*.pkl')):
        rows=pickle.load(f.open('rb'));prev=None;result=[]
        for r in rows:
            row={k:v for k,v in r.items() if np.isscalar(v) or v is None}
            row={k:v for k,v in row.items() if not any(tag in k for tag in ('_bridge','_transfer','_motion','_extent'))}
            v=float(r['v']);trace=r['trace_v6S_b2A_retro'];cheap=r['trace_cheap']
            rawmax,raw=P.required_decel(*trace['raw'],v,pp);refmax,ref=P.required_decel(*trace['refined'],v,pp);chmax,chdem=P.required_decel(*cheap['mono'],v,pp)
            assert P.discrete_action(rawmax,pp)==r['a_v6S_b2A_retro|none']
            assert P.discrete_action(refmax,pp)==r['a_v6S_b2A_retro|v3a']
            median,hard,skip,links=simple.readout_demands(cheap,trace,raw,ref,chdem)
            choices={'median':P.discrete_action(float(median.max()) if len(median) else 0,pp),
                     'hardvote':P.discrete_action(float(hard.max()) if len(hard) else 0,pp),
                     'framevote':int(np.median([r['a_cheap'],r['a_v6S_b2A_retro|none'],r['a_v6S_b2A_retro|v3a']]))}
            demands={'median':median,'hardvote':hard}
            sup=np.array([i for i in SP.support(trace['raw'],v,pp) if trace['from_full'][i]],int)
            fixedhard=bool(np.any(skip&(raw>=pp.thr_hard)))
            saved=len(sup) if fixedhard else int(skip[sup].sum())
            row['diag_refiner_objects_v6S_b2A_median']=len(sup);row['diag_refiner_skippable_v6S_b2A_median']=saved
            for name,action in choices.items():
                prefix='v6S_b2A_'+name
                row['a_'+prefix+'|v3a']=action;row['J_'+prefix+'|v3a']=P.decision_cost(action,r['a_gt'],prev,pp,cp)['J']
                row['px_'+prefix]=r['px_v6S_b2A_retro'];row['ncalls_'+prefix]=r['ncalls_v6S_b2A_retro']
                if name in demands and int(r['dec'])>=0:
                    idx=int(r['dec']);j=int(K.IE.matched(r['gt_boxes'][idx:idx+1],r['gt_classes'][idx:idx+1],trace['xyxy'],trace['coarse'])[0])
                    row['dec_found_'+prefix]=j>=0;row['dec_dem_'+prefix]=float(demands[name][j]) if j>=0 else np.nan
            # Baseline parity independently recomputed, including its jerk convention.
            assert abs(P.decision_cost(r['a_v6S_b2A_retro|v3a'],r['a_gt'],prev,pp,cp)['J']-r['J_v6S_b2A_retro|v3a'])<1e-8
            prev=r['a_cheap'];result.append(row)
            replaced=SP.centres_in(cheap['xyxy'],trace['crops'])
            adequate=np.array([SP.covered(b,trace['crops']) for b in cheap['xyxy']],bool)
            match=simple.links(cheap['xyxy'],cheap['coarse'],trace['xyxy'],trace['coarse'])
            flags=replaced&~adequate&(match<0)
            audit.append(dict(dataset=a.dataset,unit=r['unit'],frame=r['frame'],gt_action=r['act_gt'],cheap_braking_objects=int((chdem>=1).sum()),
                partial_unmatched_cheap=int(flags.sum()),partial_unmatched_cheap_braking=int((flags&(chdem>=1)).sum()),
                refiner_objects=len(sup),median_skippable_refiner_objects=saved,median_all_refiners_skippable=fixedhard or saved==len(sup),
                median_action=choices['median'],deployed_action=r['a_v6S_b2A_retro|v3a']))
        dst=out/f.name;tmp=dst.with_suffix('.tmp');pickle.dump(result,tmp.open('wb'));tmp.replace(dst)
        save(dst.with_suffix('.parity.json'),dict(dataset=a.dataset,unit=f.stem,frames=len(result),differences={'raw':0,'v3a':0,'cost':0},recomputed=['raw','v3a','cost']))
    assert source_hashes()==before
    import pandas as pd
    report=HERE/'reports/mechanisms';report.mkdir(parents=True,exist_ok=True);pd.DataFrame(audit).to_csv(report/(a.dataset+'_frames.csv'),index=False)
    save(report/(a.dataset+'_DONE.json'),dict(status='complete',units=len(list(src.glob('*.pkl'))),frames=len(audit),main_sources_unchanged=True,
        scope='Final-readout-only policy; no query/history change. Skippable refiner counts are exact action opportunities, not latency measurements.'))
    print(a.dataset,'readout complete',len(audit),flush=True)
if __name__=='__main__':main()
