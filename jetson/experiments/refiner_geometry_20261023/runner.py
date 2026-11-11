"""Stored-scene geometry attribution; no inference or mutation of frozen records."""
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

B=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
HERE=B/'experiments/refiner_geometry_20261023'
sys.path.insert(0,str(B/'scripts/A_method'))
from k31_episodes import runs
from rap import planner as P

LABELS=('raw','deployed v3a','range only','lateral only','deployed geometry, fixed closing velocity')


def main():
    out=HERE/'reports';out.mkdir(exist_ok=True)
    state=dict(status='running',pid=os.getpid())
    def save(**kw):
        state.update(kw);p=out/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(out/'state.json')
    save()
    pp,cp=P.PlannerParams(),P.CostParams()
    command=np.array([P.commanded_decel(i,pp) for i in range(3)])
    frame_rows=[];unit_rows=[];audits=[];source_hashes=[]
    try:
        for ds,expected in [('nusc',152),('av2',300)]:
            folder=B/'experiments/w1_dev_20261023/results/dev'/ds
            files=sorted(folder.glob('*.pkl'));assert len(files)==expected
            for file in files:
                data=file.read_bytes();source_hashes.append(dict(dataset=ds,unit=file.stem,sha256=hashlib.sha256(data).hexdigest()))
                rows=pickle.loads(data);prev=None;records=[];counts=dict(required_downgrade_frames=0,range_alone_suppresses=0,lateral_alone_suppresses=0,both_alone_suppress=0)
                for r in rows:
                    trace=r['trace_v6S'];raw=tuple(np.asarray(x,float) for x in trace['raw'])
                    ref=tuple(np.asarray(x,float) for x in trace['refined'])
                    ratio=np.ones(len(raw[0]));good=np.isfinite(raw[0])&(raw[0]>0)&np.isfinite(ref[0])
                    ratio[good]=ref[0][good]/raw[0][good]
                    fixed=ref[3].copy();finite=np.isfinite(fixed);fixed[finite]*=ratio[finite]
                    states=[raw,ref,(ref[0],raw[1],raw[2],raw[3]),(raw[0],ref[1],ref[2],raw[3]),(ref[0],ref[1],ref[2],fixed)]
                    actions=[P.discrete_action(P.required_decel(*s,r['v'],pp)[0],pp) for s in states]
                    assert actions[0]==r['a_v6S|none'] and actions[1]==r['a_v6S|v3a'],(ds,file.stem,r['frame'])
                    costs=[P.decision_cost(a,r['a_gt'],prev,pp,cp)['J'] for a in actions]
                    assert abs(costs[0]-r['J_v6S|none'])<1e-8 and abs(costs[1]-r['J_v6S|v3a'])<1e-8
                    if r['gt']:
                        record=dict(dataset=ds,unit=file.stem,frame=r['frame'],act_gt=r['act_gt'],demand=r['a_gt'],cheap_cost=r['J_cheap'])
                        for label,a,j in zip(LABELS,actions,costs):
                            record['action_'+label]=a;record['cost_'+label]=j
                            record['loss_'+label]=cp.lam_risk*max(0,r['a_gt']-command[a])**2
                        records.append(record);frame_rows.append(record)
                        if r['act_gt']>=1 and actions[0]>=1 and actions[1]==0:
                            counts['required_downgrade_frames']+=1
                            counts['range_alone_suppresses']+=int(actions[2]==0)
                            counts['lateral_alone_suppresses']+=int(actions[3]==0)
                            counts['both_alone_suppress']+=int(actions[2]==0 and actions[3]==0)
                    prev=r['a_cheap']
                d=pd.DataFrame(records);required=runs(d.act_gt.to_numpy()>=1)
                for label in LABELS:
                    action=d['action_'+label].to_numpy();gt=d.act_gt.to_numpy()
                    misses=sum(not (action[start:start+length]>=1).any() for start,length in required)
                    false=sum((gt[start:start+length]<1).all() for start,length in runs(action>=1))
                    unit_rows.append(dict(dataset=ds,unit=file.stem,pipeline=label,frames=len(d),episodes=len(required),
                        missed=misses,false=false,cheap_cost=d.cheap_cost.sum(),cost=d['cost_'+label].sum(),loss=d['loss_'+label].sum()))
                audits.append(dict(dataset=ds,unit=file.stem,baseline_actions_costs_equal=True,**counts))
            save(dataset=ds,units_done=expected)
        u=pd.DataFrame(unit_rows);summary=[];contrasts=[];intervals=[]
        for ds,d in u.groupby('dataset'):
            units=sorted(d.unit.unique());n=len(units)
            draws=np.random.default_rng(0).integers(0,n,(2000,n));c=np.array([np.bincount(x,minlength=n) for x in draws],float)
            points={};boot={}
            for label,g in d.groupby('pipeline'):
                g=g.set_index('unit').reindex(units)
                num=dict(G=(g.cheap_cost-g.cost).to_numpy(),missed_rate=g.missed.to_numpy(),false_per_1000=1000*g.false.to_numpy(),insufficient_loss_per_frame=g.loss.to_numpy())
                den=dict(G=g.cheap_cost.to_numpy(),missed_rate=g.episodes.to_numpy(),false_per_1000=g.frames.to_numpy(),insufficient_loss_per_frame=g.frames.to_numpy())
                points[label]={};boot[label]={}
                for metric in num:
                    points[label][metric]=num[metric].sum()/den[metric].sum()
                    nn,dd=c@num[metric],c@den[metric]
                    sample=np.divide(nn,dd,out=np.full(len(nn),np.nan),where=dd!=0);boot[label][metric]=sample
                    lo,hi=np.nanpercentile(sample,[2.5,97.5])
                    intervals.append(dict(dataset=ds,pipeline=label,metric=metric,estimate=points[label][metric],ci_lo=lo,ci_hi=hi))
                summary.append(dict(dataset=ds,pipeline=label,units=n,frames=int(g.frames.sum()),episodes=int(g.episodes.sum()),missed=int(g.missed.sum()),**points[label]))
            for label in LABELS:
                if label=='deployed v3a':continue
                for metric in points[label]:
                    lo,hi=np.nanpercentile(boot[label][metric]-boot['deployed v3a'][metric],[2.5,97.5])
                    contrasts.append(dict(dataset=ds,pipeline=label,comparator='deployed v3a',metric=metric,
                        candidate_minus_baseline=points[label][metric]-points['deployed v3a'][metric],ci_lo=lo,ci_hi=hi))
        for name,data in [('summary',summary),('intervals',intervals),('contrasts',contrasts),('unit_counts',audits),('unit_metrics',unit_rows),('source_hashes',source_hashes),('frames',frame_rows)]:
            pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)
        lines=['# Refiner geometry — stored-scene feasibility diagnostic','',
            'All 152 nuScenes/300 AV2 development units; baseline raw/deployed actions and costs reproduced exactly.',
            'No inference or main-source change. Five prespecified rows, all retained.',
            'Paired unit bootstrap 2000 seed0, pointwise 95% intervals in intervals.csv/contrasts.csv.','',
            '| Dataset | Counterfactual | Missed / episodes | False/1000 | Insufficient loss/frame | G |',
            '|---|---|---:|---:|---:|---:|']
        for r in summary:lines.append(f"| {r['dataset']} | {r['pipeline']} | {r['missed']}/{r['episodes']} | {r['false_per_1000']:.4f} | {r['insufficient_loss_per_frame']:.4f} | {r['G']:.4f} |")
        lines+=['','Range-only and lateral-only can violate camera ray/depth coupling. These isolate mechanisms;',
            'favorable decision metrics alone do not establish accurate geometry or a deployable remedy.',
            'No generic whole-frame controls, selected method, or untouched confirmatory evidence is established.',
            'Unit attribution counts overlap and do not prove that a raw support detection is the true decisive object.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n');save(status='complete',units=452)
    except BaseException as exc:save(status='failed',error=repr(exc));raise


if __name__=='__main__':main()
