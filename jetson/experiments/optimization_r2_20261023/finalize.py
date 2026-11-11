"""Finish all-row comparisons and audit source integrity/artifact completeness."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rap import planner as P

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R2_20261023'

def main():
    e=pd.read_csv(OUT/'dev/episodes.csv');transitions=[];changed=[]
    for ds,d in e.groupby('dataset'):
        for comparator in ['v6-S','round1 combined','cheap']:
            b=d[d.pipeline==comparator].set_index(['unit','start_frame'])
            for name,g in d.groupby('pipeline'):
                x=g.set_index(['unit','start_frame']).join(b[['missed']],rsuffix='_baseline')
                rescued=x.missed_baseline&~x.missed;lost=~x.missed_baseline&x.missed
                transitions.append(dict(dataset=ds,pipeline=name,comparator=comparator,
                    rescued=int(rescued.sum()),newly_missed=int(lost.sum()),
                    hard_rescued=int((rescued&(x.peak_demand>=3.5)).sum()),hard_newly_missed=int((lost&(x.peak_demand>=3.5)).sum())))
                for (unit,start),r in x[rescued|lost].iterrows():
                    changed.append(dict(dataset=ds,pipeline=name,comparator=comparator,unit=unit,start_frame=start,
                        end_frame=r.end_frame,peak_demand=r.peak_demand,missed=bool(r.missed),baseline_missed=bool(r.missed_baseline)))
    pd.DataFrame(transitions).to_csv(OUT/'dev/episode_transitions.csv',index=False)
    pd.DataFrame(changed).to_csv(OUT/'dev/changed_episodes.csv',index=False)
    f=pd.read_csv(OUT/'dev/frames.csv');lossrows=[];pp=P.PlannerParams()
    cmd=np.array([P.commanded_decel(i,pp) for i in range(3)])
    for ds,d in f.groupby('dataset'):
        units=sorted(d.unit.unique());n=len(units);draws=np.random.default_rng(0).integers(0,n,(2000,n))
        C=np.array([np.bincount(x,minlength=n) for x in draws],float)
        denom=d.groupby('unit').size().reindex(units).to_numpy();bden=C@denom
        cheap=np.maximum(0,d.a_gt-cmd[d.a_cheap.to_numpy(int)])**2
        for col in [x for x in d if x.startswith('a_v6S')]:
            delta=np.maximum(0,d.a_gt-cmd[d[col].to_numpy(int)])**2-cheap
            num=pd.Series(delta.to_numpy(),index=d.index).groupby(d.unit).sum().reindex(units).to_numpy()
            bs=(C@num)/bden;lo,hi=np.percentile(bs,[2.5,97.5])
            lossrows.append(dict(dataset=ds,action_column=col,comparator='cheap',loss_difference_per_frame=float(num.sum()/denom.sum()),
                                 ci_lo=lo,ci_hi=hi))
    pd.DataFrame(lossrows).to_csv(OUT/'dev/loss_vs_cheap.csv',index=False)
    hashes=json.loads((HERE/'fork_manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    states=[]
    for folder in [HERE/'job_status',ROOT/'experiments/optimization_r2_assoc_20261023/job_status']:
        for p in folder.glob('*.json'):
            if '.sources_' in p.name:continue
            s=json.loads(p.read_text());assert s['status']=='complete' and not s['changed_main_sources'],s
            states.append(dict(experiment=folder.parent.name,job=s['name'],host=s['host'],exit_code=s['exit_code']))
    parity=pd.read_csv(OUT/'dev/baseline_parity.csv')
    assert len(parity)==452 and parity.baseline_differing.sum()==0 and parity.frames.sum()==29782
    cases=json.loads((HERE/'case_manifest_with_severity.json').read_text())['cases']
    for c in cases:
        for name in ['panel.pdf','panel.png','preview.jpg','details.json','records.pkl']:
            assert (OUT/'cases'/c['case_id']/name).exists(),(c['case_id'],name)
    head=[json.loads((OUT/'head_audit'/(ds+'_STATUS.json')).read_text()) for ds in ['nusc','av2']]
    assert all(s['status']=='complete' for s in head) and sum(s['samples'] for s in head)==14696
    assert len(pd.read_csv(OUT/'screen/summary.csv'))==20
    assert len(pd.read_csv(OUT/'dev/summary.csv'))==32
    assert len(pd.read_csv(OUT/'association_inloop/summary.csv'))==20
    result=dict(status='complete',main_source_hashes_unchanged=True,case_units=9,case_frames=27,
        cpu_screen_rows=20,full_inloop_rows=32,full_inloop_units=452,full_inloop_frames=29782,
        association_pilot_rows=20,association_pilot_units=9,head_validation_examples=14696,
        recomputed_controls_differing=0,jobs=states,jetson_new_window=False,
        method_promoted=False,intentional_repartition=json.loads((HERE/'repartition.json').read_text()))
    (OUT/'completion_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','intentional_repartition')}))

if __name__=='__main__':main()
