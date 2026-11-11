"""CPU feasibility of requesting temporal evidence for a refiner downgrade.

Diagnostic only: no detector/head inference, veto, configuration selection or
claimed recovery. Trigger uses runtime states; GT appears only in evaluation.
"""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

B=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
E=B/'experiments/w1_dev_20261023'
sys.path.insert(0,str(B/'scripts/A_method'))
from rap import planner as P
from k31_episodes import runs


def main():
    pp=P.PlannerParams();frames=[];episodes=[]
    for ds,n in [('nusc',152),('av2',300)]:
        files=sorted((E/'results/dev'/ds).glob('*.pkl'));assert len(files)==n
        for file in files:
            rr=[r for r in pickle.load(file.open('rb')) if r['gt']];eligible=[]
            for r in rr:
                t=r['trace_v6S'];_,raw=P.required_decel(*t['raw'],r['v'],pp)
                _,ref=P.required_decel(*t['refined'],r['v'],pp)
                known=set(t['chain_ids'])
                candidates=[i for i in range(len(raw)) if t['from_full'][i] and i not in known
                    and raw[i]>=pp.thr_decel and ref[i]<pp.thr_decel]
                gate=bool(r['a_cheap']>=1 and r['a_v6S|none']>=1 and r['a_v6S|v3a']==0 and candidates)
                eligible.append(gate)
                frames.append(dict(dataset=ds,unit=file.stem,frame=r['frame'],eligible=gate,
                    candidates=len(candidates),required_keep=r['act_gt']>=1 and r['a_v6S|v3a']==0,
                    existing_recovery_attempted=r['retro_attempted_v6S_retro']>0))
            for start,length in runs(np.array([r['act_gt'] for r in rr])>=1):
                rows=rr[start:start+length]
                episodes.append(dict(dataset=ds,unit=file.stem,start_frame=rows[0]['frame'],
                    baseline_missed=not any(r['a_v6S|v3a']>=1 for r in rows),
                    recovery_missed=not any(r['a_v6S_retro|v3a']>=1 for r in rows),
                    cheap_missed=not any(r['a_cheap']>=1 for r in rows),
                    has_post_query_opportunity=any(eligible[start:start+length])))
    f=pd.DataFrame(frames);e=pd.DataFrame(episodes);summary=[]
    for ds,g in f.groupby('dataset'):
        z=e[e.dataset==ds]
        summary.append(dict(dataset=ds,frames=len(g),eligible_frames=int(g.eligible.sum()),
            eligible_fraction=g.eligible.mean(),max_extra_detector_executions_per_frame=2*g.eligible.mean(),
            required_keep_eligible_frames=int((g.required_keep&g.eligible).sum()),
            baseline_missed_episodes_with_opportunity=int((z.baseline_missed&z.has_post_query_opportunity).sum()),
            remaining_recovery_missed_episodes_with_opportunity=int((z.recovery_missed&z.has_post_query_opportunity).sum()),
            baseline_only_vs_cheap_with_opportunity=int((z.baseline_missed&~z.cheap_missed&z.has_post_query_opportunity).sum())))
    out=E/'reports/post_query_opportunity';out.mkdir(exist_ok=True)
    f.to_csv(out/'frames.csv',index=False);e.to_csv(out/'episodes.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    (out/'DONE.json').write_text(json.dumps(dict(status='complete',scope='runtime-predicate opportunity only; not query success, recovery, or quality gain'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
