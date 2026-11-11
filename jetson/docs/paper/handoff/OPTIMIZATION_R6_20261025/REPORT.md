# R6 — current-observation corroboration after negative verification

Completed 2026-10-25. A small, interpretable decision improvement is feasible, but the remaining episode-miss weakness is not solved. No candidate is promoted. Frozen method/core, checkpoints, K34, action thresholds and test splits are unchanged.

All 152 nuScenes new-val scenes (6,117 frames) and 300 AV2 CFG-DEV logs (23,665 frames): 20 declared rules plus two explicitly adaptive compositions, seven controls per dataset, **58 complete rows**. No training or test selection. All intervals use paired scene/log bootstrap, 2,000, seed0, pointwise exploratory. G keeps the previous-cheap jerk convention.

The useful addition follows [ByteTrack](https://arxiv.org/abs/2110.06864): a current low-score crop box may continue a previously verified object. Current geometry and genuine observed C1 determine demand; the action is capped by one or two previous verified actions, then max-composed with the R5 candidate. No unobserved track produces a brake. This borrows an association principle, not a tracking-performance guarantee. [Research rationale and rejected alternatives](RESEARCH_NOTES.md).

| Dataset | Row | Missed / required episodes | Loss/frame [95% CI] | G [95% CI] |
|---|---|---:|---|---|
| nusc | R5 point_residual q50 | 34/99 | 0.317066 [0.199852, 0.451294] | 0.414969 [0.210275, 0.558276] |
| nusc | R6 track_agree1_r5 | 34/99 | 0.304088 [0.195737, 0.432269] | 0.434102 [0.214388, 0.574712] |
| nusc | R6 track_agree2_r5 | 34/99 | 0.311097 [0.197934, 0.441324] | 0.424330 [0.213675, 0.565802] |
| av2 | R5 point_residual q50 | 43/366 | 0.333299 [0.264270, 0.411843] | 0.349601 [0.301651, 0.398602] |
| av2 | R6 track_agree1_r5 | 43/366 | 0.332363 [0.263914, 0.411158] | 0.350448 [0.302303, 0.399177] |
| av2 | R6 track_agree2_r5 | 43/366 | 0.332616 [0.263978, 0.411467] | 0.350380 [0.302110, 0.399310] |

| Dataset | Addition to R5 | Δ loss/frame [95% CI] | Δ false-braking frames/1,000 [95% CI] | Δ false-HARD frames/1,000 [95% CI] |
|---|---|---|---|---|
| nusc | R6 track_agree1_r5 | -0.012978 [-0.038986, +0.000000] | +0.326958 [+0.000000, +0.817929] | +0.163479 [+0.000000, +0.491159] |
| nusc | R6 track_agree2_r5 | -0.005969 [-0.017931, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| av2 | R6 track_agree1_r5 | -0.000936 [-0.002454, -0.000085] | +0.126769 [+0.000000, +0.295708] | +0.042256 [+0.000000, +0.127000] |
| av2 | R6 track_agree2_r5 | -0.000683 [-0.002041, +0.000000] | +0.042256 [+0.000000, +0.127054] | +0.000000 [+0.000000, +0.000000] |

**Interpretation.** One-lag composition reduces loss relative to R5 by 4.09% / 0.28% (nuScenes / AV2); two-lag by 1.88% / 0.20%. Neither rescues an additional required episode. One-lag adds two correctly supported HARD-action frames in nuScenes and one in AV2; two-lag retains one in each dataset. These are frame recoveries within episodes already detected elsewhere. All nuScenes incremental gain is scene-1049; AV2 one-/two-lag gains occur in seven/two logs, with the top log contributing 70.2%/96.2%. Most incremental intervals touch or cross zero. See severity.csv and concentration.csv in each followup directory.

**Negative result.** Broad retention/composition lowers misses to 31/99 and 38/366, but raises false braking. The aggressive R5 composition has false episodes/1,000 of 9.318 / 18.804 versus R5 5.558 / 10.015; AV2 G falls 0.350→0.304. Some episode rescues lack decisive-object evidence. Lower false-episode counts in bounded variants can arise by joining braking runs; false frames/excess braking must be read alongside them. Every point and interval is in all_rows.csv, all_intervals.csv and all_contrasts.csv.

**Evidence and cost.** Replayed exactly the stored current crop canvases: all 112,611 stored final full boxes have zero coordinate/confidence difference against replayed high-score proposals. Recovered 100,022 low-score proposals; no new deployed detector query/pixels is requested. Cross-crop coarse-class NMS and bounded readout are new postprocessing. All detector mean/p95 values are reported. Core integrity covers 203 local source files and remote source/checkpoint snapshots per job.

**Jetson.** See jetson/REPORT.md, intervals.csv and window/state.json for the exclusive component benchmark. It measures only matching/C1/capped-action readout, on stored VUW observations. It excludes proposal deduplication, copies, buffering, R5 classifier and detector/refiner runtime. It is not full-pipeline overhead, trt32 parity, FPS or an end-to-end speed claim.

**Remaining failure.** Observation persistence repairs some night-car/edge-object dropouts, but a real passing bus can still have wrong corridor/range demand. Retaining its existence preserves that decision error. Fragmented trucks and severe near/edge cases remain; a missing IoU match does not certify invisibility or bad annotation. GT and every original frame remain in all scores. The finite discrete controller also leaves residual continuous loss; it is not silently changed.
The conservative two-lag composition still misses 43 AV2 episodes versus cheap 30, and its loss/frame .332616 exceeds cheap .313958; it has much lower false-braking frame exposure (59.582 versus 175.153 per1,000). The original W1 tradeoff is reduced, not eliminated. These controls and their intervals remain in all_rows.csv/all_intervals.csv.

**Reproduce.** Source the absolute VUW env.sh first. Launch collect nusc shards 0/2 and 1/2 on VUW and collect av2 shards on VUW1 through launch_gpu.py (unchanged wrapper, SHARE=1, MIN_FREE_MB=10000, two workers, OMP4). Then job.py finish/compose/availability/export_runtime/cases per dataset; validate all. Commands, hosts, source/protocol SHA256 and exits are in job_status/. Local Jetson: scripts/py experiments/optimization_r6_20261025/window.py; report_runtime.py; consolidate.py. Original stage reports, the pre-outcome protocol, and the post-nuScenes composition protocol are preserved.

Artifacts: [all rows](all_rows.csv), [all paired contrasts](all_contrasts.csv), [case review](CASE_REVIEW.md), [completion audit](completion_audit.json). Mirror: results/vuw_summaries/OPTIMIZATION_R6_20261025/.
