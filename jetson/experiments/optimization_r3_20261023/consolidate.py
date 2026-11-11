"""Final local handoff assembled from completed all-row CSVs, without promotion."""
import hashlib
import json
import shutil
from pathlib import Path
import time
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R3_20261023'
STAGES=['dev','learned','calibrated','lift','persist','iddev','low','compact']
CHOICES=[('dev','cheap'),('dev','v6-S'),('dev','round1 combined'),('dev','critical extent retention'),
         ('learned','learned missing history'),('low','point fallback + extent retention'),
         ('compact','compact point fallback'),('low','corroborated upper fallback'),
         ('learned','uncertainty query'),('lift','joint geometric lift'),('lift','joint decision lift'),
         ('lift','upper quantile + extent retention'),('persist','upper quantile + extent retention')]

def md(d):
    return ['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |',
            *['| '+' | '.join(map(str,row))+' |' for row in d.itertuples(index=False,name=None)]]

def main():
    full=pd.read_csv(OUT/'overview/all_complete_rows.csv');contrasts=pd.read_csv(OUT/'overview/all_complete_contrasts.csv')
    assert len(full)==228 and json.loads((OUT/'overview/MANIFEST.json').read_text())['status']=='complete'
    selected=pd.concat([full[(full.suite==s)&(full.pipeline==n)] for s,n in CHOICES],ignore_index=True)
    selected.to_csv(OUT/'key_results.csv',index=False)
    interval=pd.concat([pd.read_csv(OUT/s/'intervals.csv').assign(suite=s) for s in STAGES],ignore_index=True)
    interval.to_csv(OUT/'all_intervals.csv',index=False)
    metric_order=['all_missed_rate','false_per_1000','insufficient_loss_per_frame','G','mean_network_px']
    keyci=selected[['suite','dataset','pipeline']].merge(contrasts,on=['suite','dataset','pipeline'])
    keyci=keyci[(keyci.comparator=='round1 combined')&keyci.metric.isin(metric_order)]
    keyci.to_csv(OUT/'key_contrasts.csv',index=False)
    table=selected[['suite','dataset','pipeline','missed','episodes','false_per_1000','insufficient_loss_per_frame','G','mean_network_px']].copy()
    table['Missed / required']=table.missed.astype(str)+'/'+table.episodes.astype(str)
    for c in ['false_per_1000','G']:table[c]=table[c].map(lambda v:f'{v:.3f}')
    for c in ['insufficient_loss_per_frame','mean_network_px']:table[c]=table[c].map(lambda v:f'{v:.5f}')
    table=table.drop(columns=['missed','episodes'])
    citable=keyci.copy()
    citable['delta [95% CI]']=citable.apply(lambda r:f'{r.candidate_minus_baseline:+.5f} [{r.ci_lo:+.5f}, {r.ci_hi:+.5f}]',axis=1)
    citable=citable[['suite','dataset','pipeline','metric','delta [95% CI]']]
    jt=pd.read_csv(OUT/'jetson_tradeoff/summary.csv');jc=pd.read_csv(OUT/'jetson_tradeoff/contrasts.csv')
    jt.to_csv(OUT/'jetson_results.csv',index=False)
    jt=jt[['dataset','pipeline','mean_ms','p95_ms','p99_ms','above50_pct','above66_7_pct','above100_pct']].copy()
    for c in jt.columns[2:]:jt[c]=jt[c].map(lambda v:f'{v:.2f}')
    latency=jc[(jc.comparator=='round1 combined')&(jc.metric=='mean_ms')].copy()
    latency['delta ms [95% CI]']=latency.apply(lambda r:f'{r.candidate_minus_baseline:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]',axis=1)
    latency=latency[['dataset','pipeline','delta ms [95% CI]']]
    lines=['# Round 3 — completed mechanism-repair experiments','',
        'Completed 2026-10-24. All 87 supervised VUW/VUW1 jobs exited successfully; 228 full-development policy/refiner rows retained across eight suites. Seven suites use both datasets; the identity gate uses nuScenes only because AV2 failed its training-holdout feasibility gate. All smoke/failed attempts and negative rows remain available.',
        'No main source, deployed checkpoint, action threshold, K34 artifact, or test split changed. This is an exploratory sandbox study, not a change to frozen v6-S.','',
        '## Scope and interpretation','',
        'Training evidence: 385 original nuScenes training/holdout units and 400 AV2 REF-TRAIN/REF-VAL logs. Development: 152 nuScenes scenes / 6,117 frames / 99 required episodes and 300 AV2 CFG-DEV logs / 23,665 frames / 366 episodes. Original train/holdout units are disjoint from development. All heads use seed 0; epoch selection uses original training holdouts. Those holdouts were already used for earlier checkpoint selection.',
        'Intervals: paired unit bootstrap, 2,000 resamples, seed 0, pointwise 95%; adaptive multi-round exploration without multiplicity correction or untouched confirmation. Episode definitions remain K31 (DECELERATE threshold 1 m/s²; hard 3.5 m/s²). Onset delay includes an episode-duration penalty for wholly missed episodes. G keeps the previous-cheap-action jerk convention.',
        'All comparisons below are against round1 B2K-A+recovery, unless explicitly labelled. This baseline already improves on frozen v6-S; do not attribute the entire v6-S difference to R3.',
        'VUW network pixels include draft plus all requested current/historical detector canvases, normalized by native image area; mean and p95 with intervals are in detector_cost/. Jetson includes actual engine padding and separately counts encoder work. Pixels cannot substitute for latency.','',
        '## Main findings','',
        '- Critical-extent persistence is the smallest measured additional cost: loss/frame decreases 0.00969 nuScenes and 0.00744 AV2, with unchanged missed episodes and essentially unchanged false episodes. Mean Jetson overhead +0.90/+1.18 ms on fixed development prefixes; p99 increases, so this is not a universal latency win. G intervals still include zero.',
        '- Image-point fallback plus extent persistence is the most balanced quality candidate among these tests: missed 38→35/99 and 49→43/366, loss −8.5%/−9.1%, false episodes +1.308/+0.803 per 1,000. Loss deltas −0.03145 [−0.05458,−0.01249] and −0.03362 [−0.05357,−0.01825]; G deltas +0.032 [0.006,0.062] and +0.029 [0.009,0.052]. Its combined latency has NOT been measured; individual component costs are not additive guarantees.',
        '- The image-free compact head removes additional CNN crops but not all processing overhead: missed 33/99 and 48/366; mean Jetson overhead +1.86/+3.72 ms vs round1. G gains are inconclusive on both datasets. It is a useful lightweight control, not a strong common solution.',
        '- Uncertainty query with original v3a rescues 12 AV2 episodes with no new misses (49→37); 11 rescues have decisive-object braking evidence. It adds 1.817 false episodes/1,000 and its complete Jetson cost is unmeasured. Five rescued episodes have hard-level peak demand, but none has decisive-object hard-brake evidence. Episode success alone overstates severe-case repair.',
        '- Joint decision-aware lifting reduces nuScenes misses 38→28, but AV2 misses 49→52 and loss rises 0.04668 [0.02101,0.07606] while G improves 0.076. Do not select it using G alone. The related support-only joint refiner also worsens AV2 loss.',
        '- Raw upper quantiles reduce misses but materially increase false braking. Empirical calibration increases false alarms further; it is not a safety guarantee. Identity classifier diagnostics improve, but the deployed nuScenes identity gate rescues no extra episode and does not reject the previously inspected scene-0150 switch.',
        '- W1 is not eliminated: point+extent still misses 35/99 versus cheap 34/99 on nuScenes, and 43/366 versus cheap 30/366 on AV2. AV2 loss 0.33489 remains above cheap 0.31396, although false episodes 9.804/1,000 are much lower than cheap 35.960/1,000. These are different operating points, not a proof that one method dominates the other.',
        '- No candidate resolves all weaknesses with negligible cost. No core promotion or confirmatory test has been performed.','',
        '## Key rows (all 228 rows in overview/REPORT.md)','',*md(table),'',
        'The `persist` suite uses empirically calibrated quantiles; `lift` uses raw quantiles. Identical policy display names across those suites are different policies. `/none`, `/half`, `/speed`, `/speedhalf` and `/adapt` rows remain in their complete suite tables. Joint-lift rows are complete joint estimators, not no-refiner ablations.','',
        '## Key paired effects and intervals','',*md(citable),'',
        '## Jetson: three repetitions, all six tradeoff rows','',
        'Fixed development prefixes: 200 warm-up + 512 timed frames per dataset and repetition (13 timed nuScenes scenes, 8 AV2 logs). Existing trt32 engines; MAXN, locked clocks,fan 255; independent pipeline histories and rotated order. The compact head runs on CPU FP32. All actual latency tails are retained.','',*md(jt),'',*md(latency),'',
        'All repeated decisions are identical. Within-Jetson baseline/image-point port parity and Torch/TRT encoder action parity pass on 80 frames/dataset. Common VUW/Jetson GT matches on all 512 timed frames/dataset; action differences remain1–6/512 depending on pipeline (jetson_tradeoff_backend/). Detector/backend/canvas differences prohibit an exact cross-backend quality claim.',
        'Both exclusive windows released cleanly and restored hardware. No engines built. Cold model loading occurs in warm-up; these are steady-state measurements. Process peak memory includes all benchmark rows and is not a per-method deployment-memory estimate. First three-row window is retained separately in jetson_point/.','',
        '## What the cases establish','',
        'Point+extent: three nuScenes and six AV2 rescues, no newly missed episodes; decisive-object evidence in 3/3 and 5/6. Compact: 5 nuScenes/1 AV2 rescues, evidence in 4/5 and 1/1, no extra hard-level rescue. Corroborated upper: 5 nuScenes rescues all with evidence; AV2 mirrors point fallback. All new/lost episodes, not just favourable cases, are listed in suite changed_episode_attribution.csv.',
        'For geometric/decision/query pipelines the main trace stores the last `/adapt` refinement. v3a_attribution/ explicitly replays the frozen v3a on changed episodes and verifies exact actions; it prevents attributing `/adapt` evidence to v3a rows. All 187 audited frame-policy pairs match. cases/cases.pdf and timelines.csv expose late/insufficient braking and wrong-object compensation.',
        'Missing-history rate estimates often use a single-image prior: restricting fallback to at least two actual observations removes most episode gains. The compact head has the same issue despite no image encoder. Causal history association still can switch identity. A correct frame action is not evidence of correct hazard reconstruction.',
        'Annotated matching can be incomplete; missing decisive-object evidence does not prove every other object irrelevant. These are offline consumer evaluations, not closed-loop collision/stopping claims.','',
        '## Training and reproducibility','',
        'Every finite-target original C1 example matches uniquely by exact crop bytes: 15,360 nuScenes and 81,605 AV2. Known full crop-chain identity switches: nuScenes 306/3077 train and 115/609 holdout; AV2 1210/21861 train and 263/3973 holdout. This is annotation-association evidence, not manual certification. Regeneration includes partial chains and changes some current-label matches; original_alignment CSVs retain those differences.',
        'Runtime feature checks pass on actual frames; compact CPU/GPU portability checks pass. All training epochs/selection criteria, calibration diagnostics and identity-gate feasibility results are retained under training/,model_audit/,training_audit/,calibration/,identity_classifier/.',
        'Authoritative protocols/scripts: experiments/optimization_r3_20261023/. GPU: unchanged run_when_gpu_free.sh SHARE=1 MIN_FREE_MB=10000, two shard workers/group; both servers source env.sh before actions and share one workspace. Exact commands, exit codes and before/after hashes: experiments/optimization_r3_20261023/job_status/. Failed early import/report-rendering attempts are described in execution_notes.md and preserved.',
        'Reproduce analysis on VUW after sourcing env.sh: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$B/experiments/optimization_r3_20261023/analyze.py" --stage <suite>` (`--dataset-only nusc` for iddev), then audit_rescues.py, audit_v3a_rescues.py, cost_audit.py and summarize.py. Exact argument lists are in their scripts and job metadata.',
        'Reproduce local timing reports: `scripts/py experiments/optimization_r3_20261023/report_jetson.py --variant point` and `--variant tradeoff`; exclusive window drivers live under experiments/optimization_r3*_jetson_20261024/.',
        'Reproduce this handoff: `scripts/py experiments/optimization_r3_20261023/consolidate.py`. Server records/checkpoints remain at /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/experiments/optimization_r3_20261023/. Every summary CSV/figure is mirrored locally to results/vuw_summaries/OPTIMIZATION_R3_20261023/.','',
        '## Disposition','',
        'Retain extent persistence as the lowest-cost measured candidate, and point+extent as the stronger quality candidate with unresolved combined latency. Retain compact as a small-head control. Do not promote joint decision-aware correction on AV2 or calibrated upper fallback as a low-tradeoff fix. Severe peak misses, current-only motion priors and association identity errors remain open. Any next candidate must have a newly fixed development protocol and fresh independent evaluation before a general claim.']
    (OUT/'TONG_HOP.md').write_text('\n'.join(lines)+'\n')
    report=['# Round 3 — final handoff (2026-10-24)','',
        '**Done:** 87 supervised jobs, 228 full-development rows across 8 suites, both Jetson windows (3 repetitions each). Main source/weights/thresholds/K34 unchanged; no test selection or core promotion. Full analysis, caveats and commands: [TONG_HOP.md](TONG_HOP.md).','',
        'Development: 152 nuScenes scenes (6,117 frames, 99 episodes) and 300 AV2 CFG-DEV logs (23,665 frames, 366 episodes). Training/holdout regeneration 785 units. Paired unit bootstrap 2,000, seed 0, pointwise 95%; repeated adaptive exploration, not confirmation. Comparison baseline: round1 B2K-A+recovery.','',
        '| Finding | nuScenes | AV2 | Assessment |','|---|---|---|---|',
        '| Extent persistence: loss delta [95% CI] | −0.00969 [−0.02748,−0.00036] | −0.00744 [−0.01776,−0.00114] | Misses unchanged; false episodes essentially unchanged |',
        '| Extent persistence: extra mean ms [95% CI] | +0.90 [0.01,2.27] | +1.18 [0.14,3.00] | Smallest measured overhead; tails still increase |',
        '| Point+extent: missed episodes | 38→35/99 | 49→43/366 | 3/3 and 5/6 rescues have decisive-object evidence |',
        '| Point+extent: loss delta [95% CI] | −0.03145 [−0.05458,−0.01249] | −0.03362 [−0.05357,−0.01825] | Better loss; combined Jetson cost unmeasured |',
        '| Point+extent: extra false episodes/1,000 [95% CI] | +1.308 [0.327,2.452] | +0.803 [0.381,1.267] | Measurable tradeoff |',
        '| Compact: missed episodes / extra mean ms | 33/99; +1.86 [1.06,2.78] | 48/366; +3.72 [1.63,6.16] | Weak AV2 benefit; G intervals include zero |','',
        '**Negative evidence:** joint decision-aware lifting raises AV2 loss +0.04668 [0.02101,0.07606] despite higher G; calibrated upper fallback raises false alarms substantially; learned identity gate rescues no extra episode. AV2 uncertainty query rescues 12 episodes, but no decisive hard-brake evidence in its 5 hard-peak rescues. Severe-case weakness remains. Point+extent still misses 43 AV2 episodes versus cheap 30, and its loss 0.33489 remains above cheap 0.31396 (at much lower false-alarm rate).','',
        '**All rows and evidence:** [overview](overview/REPORT.md), [key CSV](key_results.csv), [all intervals](all_intervals.csv), [paired effects](key_contrasts.csv), [detector mean/p95](detector_cost/REPORT.md), [Jetson](jetson_tradeoff/REPORT.md), [cross-backend differences](jetson_tradeoff_backend/REPORT.md), [case timelines](cases/REPORT.md), [completion audit](completion_audit.json). The calibrated/raw distinction is explicit in suite names.','',
        'Exact worker commands/code hashes: experiments/optimization_r3_20261023/job_status/; local 243-source snapshot unchanged. Failed attempts retained. Summary CSVs/PDF/PNG mirrors: results/vuw_summaries/OPTIMIZATION_R3_20261023/. No claim of closed-loop safety, independent calibration, equal total budget superiority, or negligible latency tradeoff.']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    audit=json.loads((OUT/'server_completion_audit.json').read_text());frozen=json.loads((ROOT/'experiments/optimization_r3_20261023/fork_manifest.json').read_text())
    changed=[p for p,h in frozen.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h];assert not changed
    windows={}
    for name,folder in [('point','optimization_r3_jetson_20261024'),('tradeoff','optimization_r3_tradeoff_jetson_20261024')]:
        state=json.loads((ROOT/'experiments'/folder/'logs/state.json').read_text());assert state['status']=='complete' and state['window_released'] and not state['restore_errors']
        assert len([r for r in state['runs'] if not r['smoke']])==6
        windows[name]=dict(status='complete',timing_runs=6,source_engine_hashes_unchanged=state['source_engine_hashes_unchanged'],window_released=True,restore_errors=[])
    for tag in ['learned_nusc','learned_av2','calibrated_nusc','calibrated_av2']:assert json.loads((OUT/'v3a_attribution'/(tag+'_DONE.json')).read_text())['status']=='complete'
    audit.update(local_source_files=len(frozen),local_sources_unchanged=True,full_development_rows=len(full),jetson_windows=windows,
        status='complete',report_generated=time.time(),unmeasured=['point+extent combined Jetson latency','uncertainty-query Jetson latency','joint-lift Jetson latency'],
        no_core_promotion=True,summary_mirror='results/vuw_summaries/OPTIMIZATION_R3_20261023/')
    (OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (OUT/'STATUS.json').write_text(json.dumps(dict(phase='complete',updated=time.time(),full_development_rows=len(full),supervised_jobs=audit['jobs'],report='REPORT.md',core_changed=False),indent=2)+'\n')
    # The writer only needs local summaries/figures, not remote raw caches.
    mirror=ROOT/'results/vuw_summaries/OPTIMIZATION_R3_20261023'
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix in ('.csv','.pdf','.png','.md','.json'):
            dest=mirror/p.relative_to(OUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    print('Final handoff:',OUT/'REPORT.md','rows',len(full),'supervised jobs',audit['jobs'])

if __name__=='__main__':main()
