"""All-row latency/cost report after the temporal Jetson feasibility window."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rap import planner as P

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'docs/paper/handoff/VERIFIED_JETSON_20261023'
NAMES=['mean_ms','p95_ms','p99_ms','above50_pct','above66_7_pct','above100_pct',
       'mean_detector_executions','mean_network_pixels','recovery_attempts_per_1000',
       'recovery_success_per_1000','G_prefix','insufficient_loss_per_frame']


def metrics(d):
    x=d.ms.to_numpy()
    return np.array([x.mean(),np.percentile(x,95),np.percentile(x,99),100*np.mean(x>50),
        100*np.mean(x>66.7),100*np.mean(x>100),d.executions.mean(),d.pixels.mean(),
        1000*d.attempt.mean(),1000*d.success.mean(),1-d.cost.sum()/d.cheap_cost.sum(),d.insufficient_loss.mean()])


def main():
    OUT.mkdir(exist_ok=True)
    while True:
        path=HERE/'logs/state.json'
        if path.exists():
            s=json.loads(path.read_text())
            (OUT/'STATUS.json').write_text(json.dumps(s,indent=2)+'\n')
            if s.get('status')=='failed' and s.get('window_released'):
                (OUT/'REPORT.md').write_text('# Temporal Jetson experiment — stopped\n\n'+s.get('error','')+
                    '\n\nAll attempted logs are retained under experiments/verified_jetson_20261023/logs/.\n')
                return
            if s.get('status')=='timing_complete' and s.get('window_released'):break
        time.sleep(20)
    pp,cp=P.PlannerParams(),P.CostParams()
    commands=np.array([P.commanded_decel(i,pp) for i in range(3)])
    frames=[];memory=[]
    mapping={'ours':'v6-S','ours_retro':'v6-S + recovery','ours_b2A':'B2K-A','ours_b2A_retro':'B2K-A + recovery'}
    for r in s['runs']:
        if r['smoke']:continue
        folder=Path(r['run']);d=pd.read_pickle(folder/'frames.pkl')
        assert len(d)==712 and d.warm.sum()==200
        meta=json.loads((folder/'meta.json').read_text())
        memory.append(dict(dataset=r['dataset'],rep=r['rep'],
            retained_history_bytes=int(d.gpu_history_bytes.max()),
            peak_gpu_allocated_bytes=meta['peak_gpu_allocated_bytes'],
            peak_gpu_reserved_bytes=meta['peak_gpu_reserved_bytes']))
        d=d[~d.warm]
        for source,label in mapping.items():
            k=source+'@trt32'
            f=d[['unit','frame','act_gt','a_gt','dt']].copy()
            for name,col in [('ms','ms_'),('action','a_'),('cost','J_'),('executions','calls_'),('pixels','px_'),
                             ('attempt','retro_attempted_'),('success','retro_success_'),('retro_calls','retro_calls_'),('retro_pixels','retro_pixels_')]:
                f[name]=d[col+k].to_numpy()
            f['cheap_cost']=d.J_cheap.to_numpy()
            f['insufficient_loss']=cp.lam_risk*np.maximum(0,f.a_gt.to_numpy()-commands[f.action.to_numpy(int)])**2
            f['dataset'],f['rep'],f['pipeline']=r['dataset'],r['rep'],label
            frames.append(f)
    d=pd.concat(frames,ignore_index=True)
    assert set(d.dataset)=={'av2_dev'}
    actual=pd.concat([pd.read_pickle(Path(r['run'])/'frames.pkl')[['unit','frame','image_filename','annotation_timestamp_ns','warm']] for r in s['runs'] if not r['smoke']],ignore_index=True).drop_duplicates()
    actual.to_csv(OUT/'frame_manifest.csv',index=False)
    (OUT/'verified_log_manifest.json').write_text((HERE/'verified_log_manifest.json').read_text())
    assert not d.duplicated(['dataset','unit','frame','rep','pipeline']).any()
    d.to_csv(OUT/'frames.csv',index=False)
    pd.DataFrame(memory).to_csv(OUT/'memory.csv',index=False)
    consistency=d.groupby(['dataset','unit','frame','pipeline']).action.nunique()
    pd.DataFrame([dict(differing_action_groups=int((consistency>1).sum()),groups=len(consistency))]).to_csv(OUT/'repetition_parity.csv',index=False)
    summaries=[];intervals=[];contrasts=[];conditional=[]
    for ds,dd in d.groupby('dataset'):
        units=sorted(dd.unit.unique());n=len(units)
        draws=np.random.default_rng(0).integers(0,n,(2000,n))
        boot={};points={}
        for label,g in dd.groupby('pipeline'):
            groups=[g[g.unit==u].sort_values(['rep','frame']) for u in units]
            points[label]=metrics(g)
            boot[label]=np.array([metrics(pd.concat([groups[j] for j in sample],ignore_index=True)) for sample in draws])
            summaries.append(dict(dataset=ds,pipeline=label,units=n,timed_frames_per_rep=512,repetitions=3,
                                  **dict(zip(NAMES,points[label]))))
            for i,name in enumerate(NAMES):
                lo,hi=np.percentile(boot[label][:,i],[2.5,97.5])
                intervals.append(dict(dataset=ds,pipeline=label,metric=name,estimate=points[label][i],ci_lo=lo,ci_hi=hi))
        for candidate,comparator in [('v6-S + recovery','v6-S'),('B2K-A','v6-S'),('B2K-A + recovery','B2K-A'),('B2K-A + recovery','v6-S')]:
            for i,name in enumerate(NAMES):
                lo,hi=np.percentile(boot[candidate][:,i]-boot[comparator][:,i],[2.5,97.5])
                contrasts.append(dict(dataset=ds,pipeline=candidate,comparator=comparator,metric=name,
                    candidate_minus_baseline=points[candidate][i]-points[comparator][i],ci_lo=lo,ci_hi=hi))
        for candidate,comparator in [('v6-S + recovery','v6-S'),('B2K-A + recovery','B2K-A')]:
            g=dd[dd.pipeline==candidate].set_index(['unit','frame','rep'])
            base=dd[dd.pipeline==comparator].set_index(['unit','frame','rep'])
            active=g.attempt>0
            assert g.index.equals(base.index)
            for label,values in [('candidate',g.loc[active,'ms']),('base_same_frames',base.loc[active,'ms'])]:
                conditional.append(dict(dataset=ds,pipeline=candidate,population='recovery attempted',row=label,
                    samples=len(values),unique_frames=values.reset_index()[['unit','frame']].drop_duplicates().shape[0],
                    mean_ms=values.mean(),p95_ms=values.quantile(.95),p99_ms=values.quantile(.99)))
    for name,rows in [('summary',summaries),('intervals',intervals),('contrasts',contrasts),('triggered_latency',conditional)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    lines=['# AV2 verified CFG-DEV — Jetson timing','',
        'All four prespecified rows retained; existing reference crop runtime/trt32 engines.',
        'Verified AV2 CFG-DEV only, 512 timed frames/rep after 200 warm-up frames; three repetitions.',
        'Original/fast helper parity passed on the first 80 verified frames. Actual training/selection unit IDs were excluded.',
        'Paired unit bootstrap 2000 seed0; repetitions nested in units; pooled quantiles.','',
        '| Dataset | Pipeline | Mean ms | p95 ms | p99 ms | >50 ms % | >66.7 ms % | >100 ms % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append('| '+r['dataset']+' | '+r['pipeline']+' | '+' | '.join(f'{r[k]:.3f}' for k in NAMES[:6])+' |')
    lines+=['','Full quality evidence comes from the VUW development factorial. Prefix G/loss here are diagnostic',
        'and use the original cheap-previous-action jerk convention; no independent closed-loop rollout.',
        'Historical execution and pixels are included. All four rows use the validated exact fast bookkeeping helper. Retaining two prior uploaded frames per recovery pipeline',
        'uses extra GPU memory; memory.csv records retained tensors and process peaks. Comparators do not share histories.',
        'Triggered latency is reported separately in triggered_latency.csv; an inactive prefix cannot characterize recovery tails.',
        'Backend/canvas differences from VUW remain possible. This is not a cross-backend parity claim.',
        'All main source/engine hashes were checked and the exclusive window released. No method is promoted.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'DONE.json').write_text(json.dumps(dict(status='complete',samples=len(d)),indent=2)+'\n')


if __name__=='__main__':main()
