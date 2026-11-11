"""K34-3: all timing rows and VUW parity; paired scene/log bootstrap, 2000, seed 0."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1];OUT=R/'docs/paper/handoff/K34';L=R/'logs/k34j'
NAMES={'ours@trt32':'v6-S','ours_b2@trt32':'v6-S <= 2','ours_b2A@trt32':'B2K-A'}
VUW={'v6-S':'v6S','v6-S <= 2':'v6Sb2','B2K-A':'v6Sb2A'}
MET=['mean_ms','p95_ms','p99_ms','share_gt_50ms','share_gt_66_7ms','share_gt_100ms']

def stats(v):return np.array([np.mean(v),np.percentile(v,95),np.percentile(v,99),np.mean(v>50),np.mean(v>66.7),np.mean(v>100)])

def main():
    state=json.loads((L/'state.json').read_text());runs=[r for r in state['runs'] if not r['smoke']]
    assert len(runs)==6 and {(r['dataset'],r['rep']) for r in runs}=={(d,i) for d in ['nusc_test20','av2'] for i in (1,2,3)}
    records=[];checks=[];frames=[]
    for r in runs:
        p=Path(r['run']);assert (p/'DONE.json').exists()
        d=pd.read_pickle(p/'frames.pkl');expected=json.loads((OUT/'reference'/(r['dataset']+'_frames.json')).read_text())
        actual=[{'unit':str(t.unit),'frame':int(t.frame),'warm':bool(t.warm)} for t in d.itertuples()]
        assert actual==expected,(r['dataset'],r['rep'],'manifest mismatch')
        assert not json.loads((p/'meta.json').read_text())['missing_images']
        d=d[~d.warm].copy()
        for col,name in NAMES.items():
            assert np.isfinite(d['ms_'+col]).all() and (d['ms_'+col]>0).all()
            if name!='v6-S':assert (d['calls_loop_'+col]<=2).all()
            f=pd.DataFrame({'dataset':r['dataset'],'rep':r['rep'],'unit':d.unit,'frame':d.frame,'pipeline':name,'latency_ms':d['ms_'+col],
                            'action':d['a_'+col],'decision_loss':d['J_'+col],'tensor_px':d['px_'+col],
                            # On trt32, STATS.calls counts the draft and ordinary crop
                            # executions; STATS.exec counts bounded crop executions.
                            # Raw crop_engine_exec only records the latter counter.
                            'detector_engine_executions':d['calls_'+col],'crop_engine_executions':d['calls_'+col]-d['draft_calls_'+col],
                            'draft_calls':d['draft_calls_'+col],'crop_calls':d.get('calls_loop_'+col,np.nan),'bound_hit':d.get('bound_hit_'+col,False)})
            frames.append(f)
            records.append({'dataset':r['dataset'],'rep':r['rep'],'pipeline':name,'frames':len(d),'units':d.unit.nunique(),**dict(zip(MET,stats(f.latency_ms.to_numpy()))),'run':str(p)})
        checks.append({'dataset':r['dataset'],'rep':r['rep'],'frames':len(d),'frame_manifest_equal':True,'frames_sha256':hashlib.sha256((p/'frames.pkl').read_bytes()).hexdigest()})
    F=pd.concat(frames,ignore_index=True);assert not F.duplicated(['dataset','rep','unit','frame','pipeline']).any()
    F.to_csv(OUT/'jetson_frames.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'jetson_validation.csv',index=False)
    intervals=[];contrasts=[]
    for ds,dsf in F.groupby('dataset',sort=True):
        units=sorted(dsf.unit.unique());rng=np.random.default_rng(0);draws=rng.integers(0,len(units),(2000,len(units)));points={};bs={}
        for name in NAMES.values():
            q=dsf[dsf.pipeline==name];samples=[q.loc[q.unit==u,'latency_ms'].to_numpy() for u in units]
            points[name]=stats(q.latency_ms.to_numpy());bs[name]=np.stack([stats(np.concatenate([samples[i] for i in draw])) for draw in draws])
            row={'dataset':ds,'rep':'pooled_3','pipeline':name,'frames':len(q),'unique_frames':len(q[['unit','frame']].drop_duplicates()),'units':len(units),**dict(zip(MET,points[name]))};records.append(row)
            for k,metric in enumerate(MET):
                lo,hi=np.percentile(bs[name][:,k],[2.5,97.5]);intervals.append({'dataset':ds,'pipeline':name,'metric':metric,'estimate':points[name][k],'ci_lo':lo,'ci_hi':hi,'unit_count':len(units),'bootstrap_n':2000,'seed':0})
        for first,second in [('B2K-A','v6-S'),('B2K-A','v6-S <= 2'),('v6-S','v6-S <= 2')]:
            for k,metric in enumerate(MET):
                diff=bs[first][:,k]-bs[second][:,k];lo,hi=np.percentile(diff,[2.5,97.5]);contrasts.append({'dataset':ds,'first':first,'second':second,'metric':metric,'difference':points[first][k]-points[second][k],'ci_lo':lo,'ci_hi':hi,'bootstrap_n':2000,'seed':0})
    pd.DataFrame(records).to_csv(OUT/'jetson_latency.csv',index=False);pd.DataFrame(intervals).to_csv(OUT/'jetson_latency_intervals.csv',index=False);pd.DataFrame(contrasts).to_csv(OUT/'jetson_latency_contrasts.csv',index=False)
    ref=pd.read_csv(OUT/'reference/vuw_b2ka_actions.csv');assert not ref.duplicated(['dataset','unit','frame']).any()
    parity=[];dis=[];missing=[]
    for (ds,rep,name),q in F.groupby(['dataset','rep','pipeline'],sort=True):
        v=ref[ref.dataset==('nusc' if ds=='nusc_test20' else ds)][['unit','frame',VUW[name]]].rename(columns={VUW[name]:'vuw_action'})
        m=q.merge(v,on=['unit','frame'],how='left',validate='one_to_one');good=m.vuw_action.notna();common=m[good].copy();same=(common.action==common.vuw_action)
        units=sorted(common.unit.unique());count=np.array([np.sum(common.unit==u) for u in units]);hit=np.array([np.sum(same[common.unit==u]) for u in units]);draws=np.random.default_rng(0).integers(0,len(units),(2000,len(units))) if len(units) else np.zeros((2000,0),int)
        vals=hit[draws].sum(1)/count[draws].sum(1) if len(units) else np.full(2000,np.nan)
        lo,hi=np.nanpercentile(vals,[2.5,97.5])
        parity.append({'dataset':ds,'rep':rep,'pipeline':name,'timed_frames':len(q),'common_frames':len(common),'missing_vuw_frames':int((~good).sum()),'differing_frames':int((~same).sum()),'agreement':float(same.mean()),'ci_lo':lo,'ci_hi':hi})
        dis.append(common.loc[~same]);missing.append(m.loc[~good])
    pd.DataFrame(parity).to_csv(OUT/'jetson_parity.csv',index=False);pd.concat(dis,ignore_index=True).to_csv(OUT/'jetson_disagreements.csv',index=False);pd.concat(missing,ignore_index=True).to_csv(OUT/'jetson_missing_reference.csv',index=False)
    # Repetition consistency is reported, not used to pick a favourable repetition.
    repeat=[]
    for (ds,name),q in F.groupby(['dataset','pipeline']):
        a=q.pivot(index=['unit','frame'],columns='rep',values='action');repeat.append({'dataset':ds,'pipeline':name,'frames':len(a),'frames_with_repetition_disagreement':int((a.nunique(axis=1)>1).sum())})
    pd.DataFrame(repeat).to_csv(OUT/'jetson_repetition_parity.csv',index=False)
    text=['# K34 — post-hoc frozen-v6-S analyses','', '**K34-3 complete; K34-1, K34-2 and K34-4 not run.** User requested Jetson first on 2026-10-22. No test-driven selection; every repetition and pipeline is reported.', '',
          'K34-3 uses the existing trt32 fast path on Jetson AGX Xavier, MAXN, locked clocks and fan PWM 255. Three fresh repetitions, independent pipeline histories, balanced pipeline order. Exact K29 manifests: nuScenes 805 timed frames / 20 scenes (plus 200 warm-up frames), AV2 586 timed frames / 8 logs (200 earlier frames excluded, ego mask y=1830). Timings include frame upload/preprocessing, draft, crop detection, geometry, C1, refiner and action; disk I/O is excluded. No shared detection cache during timing.', '',
          'Bounded variants limit crop batches after the draft to two; actual detector engine executions can differ because of batch splitting. B2K-A retains pending truncated detections at the cap, as K31 mode A. Original K29/K31 sources are unchanged. CPU port checks: 400 K31 comparisons and 400 unchanged-K29 comparisons passed; shared-detection GPU smoke parity passed before timing.', '',
          'Intervals: paired scene/log bootstrap, 2,000, seed 0, 95% percentile. Each drawn unit carries all three repetitions. Quantiles are of pooled latency samples, not averages of per-run quantiles. These are pointwise intervals; AV2 has only eight logs. Fractions above thresholds use strict > comparisons.', '',
          '| Dataset | Pipeline | Mean ms [95% CI] | p95 ms | p99 ms | >50 ms | >66.7 ms | >100 ms |','|---|---|---|---|---|---|---|---|']
    for r in [r for r in records if r['rep']=='pooled_3']:
        ci=next(x for x in intervals if x['dataset']==r['dataset'] and x['pipeline']==r['pipeline'] and x['metric']=='mean_ms')
        text.append('| {} | {} | {:.2f} [{:.2f}, {:.2f}] | {:.2f} | {:.2f} | {:.2%} | {:.2%} | {:.2%} |'.format(r['dataset'],r['pipeline'],r['mean_ms'],ci['ci_lo'],ci['ci_hi'],r['p95_ms'],r['p99_ms'],r['share_gt_50ms'],r['share_gt_66_7ms'],r['share_gt_100ms']))
    text+=['','All metric CIs: jetson_latency_intervals.csv. All paired contrasts: jetson_latency_contrasts.csv. Per-run rows: jetson_latency.csv; raw evaluated frames: jetson_frames.csv.', '',
           'B2K-A reduces mean and p95 versus v6-S on both datasets: paired mean differences are -1.53 ms [-2.83, -0.53] on nuScenes and -1.94 ms [-2.94, -1.03] on AV2; paired p95 differences are -10.59 ms [-14.65, -2.50] and -13.82 ms [-22.79, -6.35]. Retaining pending detections has a cost versus the ordinary bounded variant: mean +0.41 ms [0.20, 0.64] / +1.99 ms [1.04, 3.07]. On AV2, B2K-A does not improve every deadline: its >50 ms share is nominally higher than v6-S by 1.93 percentage points [-1.75, 5.88].', '',
           'The two-crop-batch limit is satisfied on every bounded frame. It is not a two-engine-execution limit: the observed maximum including the draft is four on nuScenes and three on AV2. In the exported CSV, crop_engine_executions is reconstructed as total detector executions minus the one draft execution; the raw harness counter of that name only counts the bounded path and is zero for unbounded v6-S. Timing and action records are unchanged.', '',
           '## Decision parity with VUW','', '| Dataset | Rep | Pipeline | Common / timed | Differing | Agreement |','|---|---|---|---|---|---|']
    for r in parity:text.append('| {} | {} | {} | {} / {} | {} | {:.3%} |'.format(r['dataset'],r['rep'],r['pipeline'],r['common_frames'],r['timed_frames'],r['differing_frames'],r['agreement']))
    text+=['','All differing frames remain in the timings and are listed in jetson_disagreements.csv. Every timed frame has a VUW reference. Each pipeline has identical actions across all three repetitions; the same discrepancies recur. B2K-A differs on three distinct nuScenes frames and two distinct AV2 frames. All three nuScenes discrepancies also occur for the ordinary bounded variant; on AV2 one is shared and one is additional. The exact source of each discrepancy has not been isolated; differing detector/canvas backends and resulting temporal histories are possible causes, not an established diagnosis. Numerical equivalence to VUW is not achieved. See jetson_parity.csv for paired-unit agreement intervals and jetson_repetition_parity.csv for consistency across repetitions.', '',
           '## Provenance and remaining work','', 'Exact run commands/configurations and environment are in each raw run folder; run paths are recorded in jetson_latency.csv and logs/k34j/state.json. Source hashes: logs/k34j/source_hashes.json; run-time truncated C1/refiner/head checksums: each run/meta.json. Full post-run engine hashes and audit checks: jetson_audit.json (these do not substitute for pre-run detector hashes). Hardware and telemetry: logs/k34j/hardware_before.json, clock_before.cfg and tegrastats.log. K29/VUW reference manifests and hashes: reference/. Figure: jetson_latency.pdf and jetson_latency.png.', '',
           'Driver: `python3 scripts/k34_jetson_window.py`. Harness: `scripts/py scripts/77_k34_latency.py --dataset {nusc_test20,av2} --config v6S --wf none --bounds 2 --paths trt32 --rep {1,2,3} --tag r{1,2,3}`; AV2 adds `--mask_y0 1830`. CPU report: `scripts/py scripts/78_k34_report.py`.', '',
           'K34-1 severity/loss/operating curves, K34-2 coverage and K34-4 NAVSIM cost remain pending. This timing result alone does not resolve W1 or establish a safety improvement.']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
    print('K34-3 REPORT DONE',len(F),'timed frame x pipeline x repetition rows',flush=True)

if __name__=='__main__':main()
