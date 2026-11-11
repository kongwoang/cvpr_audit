"""Package every R6 result, integrity record and caveat for the paper session."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R6_20261025';MIRROR=ROOT/'results/vuw_summaries/OPTIMIZATION_R6_20261025'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy(src,dst):
    for p in src.rglob('*'):
        if p.is_file() and p.suffix in ('.md','.csv','.json','.pdf','.png','.log','.cfg'):
            q=dst/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def main():
    OUT.mkdir(parents=True,exist_ok=True);copy(HERE/'reports',OUT);copy(HERE/'job_status',OUT/'job_status');copy(HERE/'logs',OUT/'jetson/window');copy(HERE/'protocol_versions',OUT/'protocol_versions')
    for name in ['protocol.md','followup_protocol.md','runtime_protocol.md','RESEARCH_NOTES.md','CASE_REVIEW.md']:
        if (HERE/name).exists():shutil.copy2(HERE/name,OUT/name)
    frames={};summary=[];contrasts=[];intervals=[]
    for ds in ['nusc','av2']:
        assert json.loads((OUT/('followup_'+ds)/'DONE.json').read_text())['status']=='complete'
        for label,acc in [('summary',summary),('contrasts',contrasts),('intervals',intervals)]:acc.append(pd.read_csv(OUT/('followup_'+ds)/(label+'.csv')))
    S=pd.concat(summary,ignore_index=True);C=pd.concat(contrasts,ignore_index=True);I=pd.concat(intervals,ignore_index=True)
    assert len(S)==58 and S.groupby('dataset').size().to_dict()=={'av2':29,'nusc':29};assert not S.duplicated(['dataset','pipeline']).any()
    S.to_csv(OUT/'all_rows.csv',index=False);C.to_csv(OUT/'all_contrasts.csv',index=False);I.to_csv(OUT/'all_intervals.csv',index=False)
    primary=['R6 track_agree1_r5','R6 track_agree2_r5'];base='R5 point_residual q50'
    chosen=['cheap','v6-S','point fallback + extent retention',base,*primary,'R6 res_low','R6 combo1_r5']
    S[S.pipeline.isin(chosen)].to_csv(OUT/'key_results.csv',index=False)
    C[(C.pipeline.isin(primary))&(C.comparator==base)].to_csv(OUT/'incremental_effects.csv',index=False)
    def point(ds,pipe,metric):return float(S[(S.dataset==ds)&(S.pipeline==pipe)].iloc[0][metric])
    def interval(ds,pipe,metric):
        r=I[(I.dataset==ds)&(I.pipeline==pipe)&(I.metric==metric)].iloc[0];return f'{r.estimate:.6f} [{r.ci_lo:.6f}, {r.ci_hi:.6f}]'
    def delta(ds,pipe,metric):
        r=C[(C.dataset==ds)&(C.pipeline==pipe)&(C.comparator==base)&(C.metric==metric)].iloc[0];return f'{r.candidate_minus_baseline:+.6f} [{r.ci_lo:+.6f}, {r.ci_hi:+.6f}]'
    lines=['# R6 — current-observation corroboration after negative verification','',
        'Completed 2026-10-25. A small, interpretable decision improvement is feasible, but the remaining episode-miss weakness is not solved. No candidate is promoted. Frozen method/core, checkpoints, K34, action thresholds and test splits are unchanged.',
        '',
        'All 152 nuScenes new-val scenes (6,117 frames) and 300 AV2 CFG-DEV logs (23,665 frames): 20 declared rules plus two explicitly adaptive compositions, seven controls per dataset, **58 complete rows**. No training or test selection. All intervals use paired scene/log bootstrap, 2,000, seed0, pointwise exploratory. G keeps the previous-cheap jerk convention.',
        '',
        'The useful addition follows [ByteTrack](https://arxiv.org/abs/2110.06864): a current low-score crop box may continue a previously verified object. Current geometry and genuine observed C1 determine demand; the action is capped by one or two previous verified actions, then max-composed with the R5 candidate. No unobserved track produces a brake. This borrows an association principle, not a tracking-performance guarantee. [Research rationale and rejected alternatives](RESEARCH_NOTES.md).',
        '',
        '| Dataset | Row | Missed / required episodes | Loss/frame [95% CI] | G [95% CI] |',
        '|---|---|---:|---|---|']
    for ds in ['nusc','av2']:
        for pipe in [base,*primary]:
            r=S[(S.dataset==ds)&(S.pipeline==pipe)].iloc[0]
            lines.append(f'| {ds} | {pipe} | {int(r.missed)}/{int(r.episodes)} | {interval(ds,pipe,"insufficient_loss_per_frame")} | {interval(ds,pipe,"G")} |')
    lines+=['','| Dataset | Addition to R5 | Δ loss/frame [95% CI] | Δ false-braking frames/1,000 [95% CI] | Δ false-HARD frames/1,000 [95% CI] |','|---|---|---|---|---|']
    for ds in ['nusc','av2']:
        for pipe in primary:lines.append(f'| {ds} | {pipe} | {delta(ds,pipe,"insufficient_loss_per_frame")} | {delta(ds,pipe,"false_braking_frames_per_1000")} | {delta(ds,pipe,"false_hard_frames_per_1000")} |')
    lines+=['',
        '**Interpretation.** One-lag composition reduces loss relative to R5 by 4.09% / 0.28% (nuScenes / AV2); two-lag by 1.88% / 0.20%. Neither rescues an additional required episode. One-lag adds two correctly supported HARD-action frames in nuScenes and one in AV2; two-lag retains one in each dataset. These are frame recoveries within episodes already detected elsewhere. All nuScenes incremental gain is scene-1049; AV2 one-/two-lag gains occur in seven/two logs, with the top log contributing 70.2%/96.2%. Most incremental intervals touch or cross zero. See severity.csv and concentration.csv in each followup directory.',
        '',
        '**Negative result.** Broad retention/composition lowers misses to 31/99 and 38/366, but raises false braking. The aggressive R5 composition has false episodes/1,000 of 9.318 / 18.804 versus R5 5.558 / 10.015; AV2 G falls 0.350→0.304. Some episode rescues lack decisive-object evidence. Lower false-episode counts in bounded variants can arise by joining braking runs; false frames/excess braking must be read alongside them. Every point and interval is in all_rows.csv, all_intervals.csv and all_contrasts.csv.',
        '',
        '**Evidence and cost.** Replayed exactly the stored current crop canvases: all 112,611 stored final full boxes have zero coordinate/confidence difference against replayed high-score proposals. Recovered 100,022 low-score proposals; no new deployed detector query/pixels is requested. Cross-crop coarse-class NMS and bounded readout are new postprocessing. All detector mean/p95 values are reported. Core integrity covers 203 local source files and remote source/checkpoint snapshots per job.',
        '',
        '**Jetson.** See jetson/REPORT.md, intervals.csv and window/state.json for the exclusive component benchmark. It measures only matching/C1/capped-action readout, on stored VUW observations. It excludes proposal deduplication, copies, buffering, R5 classifier and detector/refiner runtime. It is not full-pipeline overhead, trt32 parity, FPS or an end-to-end speed claim.',
        '',
        '**Remaining failure.** Observation persistence repairs some night-car/edge-object dropouts, but a real passing bus can still have wrong corridor/range demand. Retaining its existence preserves that decision error. Fragmented trucks and severe near/edge cases remain; a missing IoU match does not certify invisibility or bad annotation. GT and every original frame remain in all scores. The finite discrete controller also leaves residual continuous loss; it is not silently changed.',
        'The conservative two-lag composition still misses 43 AV2 episodes versus cheap 30, and its loss/frame .332616 exceeds cheap .313958; it has much lower false-braking frame exposure (59.582 versus 175.153 per1,000). The original W1 tradeoff is reduced, not eliminated. These controls and their intervals remain in all_rows.csv/all_intervals.csv.',
        '',
        '**Reproduce.** Source the absolute VUW env.sh first. Launch collect nusc shards 0/2 and 1/2 on VUW and collect av2 shards on VUW1 through launch_gpu.py (unchanged wrapper, SHARE=1, MIN_FREE_MB=10000, two workers, OMP4). Then job.py finish/compose/availability/export_runtime/cases per dataset; validate all. Commands, hosts, source/protocol SHA256 and exits are in job_status/. Local Jetson: scripts/py experiments/optimization_r6_20261025/window.py; report_runtime.py; consolidate.py. Original stage reports, the pre-outcome protocol, and the post-nuScenes composition protocol are preserved.',
        '',
        'Artifacts: [all rows](all_rows.csv), [all paired contrasts](all_contrasts.csv), [case review](CASE_REVIEW.md), [completion audit](completion_audit.json). Mirror: results/vuw_summaries/OPTIMIZATION_R6_20261025/.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,ds in zip(axs,['nusc','av2']):
        d=S[S.dataset==ds];ax.scatter(d.false_braking_frames_per_1000,d.insufficient_loss_per_frame,c='.7',s=15,label='Every R6 row/control')
        for pipe,marker,color in [(base,'s','black'),('R6 track_agree1_r5','o','tab:blue'),('R6 track_agree2_r5','^','tab:green'),('R6 combo1_r5','x','tab:red'),('v6-S','D','tab:orange')]:
            q=d[d.pipeline==pipe].iloc[0];ax.scatter(q.false_braking_frames_per_1000,q.insufficient_loss_per_frame,marker=marker,c=color,s=55,label=pipe)
        ax.set_title(ds+' — adaptive development');ax.set_xlabel('False-braking frames / 1,000');ax.set_ylabel('Insufficient-braking loss / frame');ax.grid(alpha=.2)
    axs[1].legend(fontsize=7);fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(OUT/('tradeoff.'+ext),dpi=160)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,5.6))
    for i,ds in enumerate(['nusc','av2']):
        for j,metric in enumerate(['insufficient_loss_per_frame','false_braking_frames_per_1000']):
            ax=axs[i,j]
            for k,pipe in enumerate(primary):
                r=C[(C.dataset==ds)&(C.pipeline==pipe)&(C.comparator==base)&(C.metric==metric)].iloc[0];ax.plot([r.ci_lo,r.ci_hi],[k,k],c='tab:blue');ax.plot(r.candidate_minus_baseline,k,'o',c='tab:blue')
            ax.axvline(0,color='.4',ls='--');ax.set_yticks([0,1]);ax.set_yticklabels(['one-lag cap + R5','two-lag cap + R5']);ax.set_title(ds);ax.set_xlabel(metric.replace('_',' '));ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.grid(axis='x',alpha=.2)
    fig.suptitle('Incremental effects versus R5; paired unit 95% CI, 2,000 resamples, seed0');fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(OUT/('incremental_effects.'+ext),dpi=160)
    plt.close(fig)
    jobs=[]
    for f in (HERE/'job_status').glob('*.json'):
        if '.sources_' in f.name:continue
        d=json.loads(f.read_text())
        if 'status' in d and 'exit_code' in d:jobs.append(d)
    assert len(jobs)==17 and all(j['status']=='complete' and j['exit_code']==0 and not j.get('changed_main_sources') for j in jobs)
    before=json.loads((HERE/'local_sources_before.json').read_text());changed=[p for p,h in before.items() if sha(p)!=h];assert not changed
    window=json.loads((HERE/'logs/state.json').read_text());assert window['status']=='complete' and window['window_released'] and not window['restore_errors']
    parity=json.loads((OUT/'jetson/parity.json').read_text());assert not parity['mismatches']
    sources={str(p.relative_to(ROOT)):sha(p) for p in HERE.rglob('*.py')};(OUT/'source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    audit=dict(status='complete',remote_jobs=len(jobs),remote_failures=0,quality_rows=58,unique_new_configurations_per_dataset=22,local_core_files=len(before),local_core_changes=changed,source_hashes=sources,jetson_window_released=True,jetson_readout_parity_mismatches=0,core_promoted=False,test_quality_used=False,bootstrap=2000,seed=0)
    (OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    MIRROR.mkdir(parents=True,exist_ok=True);copy(OUT,MIRROR)
    manifest={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()};assert all(sha(MIRROR/k)==h for k,h in manifest.items())
    (HERE/'FINISH_STATUS.json').write_text(json.dumps(dict(status='complete',report=str(OUT/'REPORT.md'),mirror_files=len(manifest)),indent=2)+'\n')
    print(OUT/'REPORT.md');print(json.dumps(audit,default=str)[:300])
if __name__=='__main__':main()
