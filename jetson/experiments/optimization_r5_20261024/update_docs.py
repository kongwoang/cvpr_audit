"""Idempotent R5 entries in the paper-writing handoff index."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TITLE='## R5 — conditional decision repair (2026-10-25, complete)'
result='''
## R5 — conditional decision repair (2026-10-25, complete)

Completed 23 supervised jobs, eight small classifiers and 104 full-development rows (44 new readout configurations per dataset plus eight controls). Original train/REF-TRAIN and unit-disjoint holdout only; all 152 nuScenes scenes / 300 masked AV2 CFG-DEV logs evaluated. Frozen core, weights, thresholds and K34 unchanged; no test-quality selection, promotion or new Jetson timing. [Report](handoff/OPTIMIZATION_R5_20261024/REPORT.md), [all rows](handoff/OPTIMIZATION_R5_20261024/all_rows.csv), [all paired effects](handoff/OPTIMIZATION_R5_20261024/all_contrasts.csv), [integrity audit](handoff/OPTIMIZATION_R5_20261024/completion_audit.json).

The useful incremental change is confidence-gated retention of current cheap evidence after a crop was executed but no compatible final detection remains. Added to R3 point+extent at fixed q=.5, nuScenes misses 35→34/99, loss .336997→.317066 (delta −.019931 [−.048364,−.000560]); AV2 misses remain 43/366, loss .334890→.333299 (−.001591 [−.004150,−.000227]). False episodes/1,000 rise +.326958 [0,.817929] / +.211282 [.042126,.463824]; false-braking frames rise +.490436 [0,1.143796] / +.760617 [.210790,1.565301]. All CIs paired unit bootstrap 2,000 seed0, pointwise exploratory. The additional nuScenes episode has decisive-object evidence; no extra hard-peak episode is rescued by this gated residual family.

Risk gains occur in only four nuScenes scenes and nine AV2 logs; one unit contributes 52.1%/67.9%. At q=.8, all nuScenes benefit comes from scene-1050 and its interval touches zero. Ungated cheap retention strongly increases false braking. All-object missing-history/rollback gates improve over round1 but do not replace R3's stronger severe-loss result. Conditional training on refiner downgrades gives only small gains; an extra AV2 episode at q=.5 lacks matched decisive-object evidence. Confidence is poorly calibrated specifically in the selected disagreement population.

The deeper image/population audit distinguishes a failed IoU match from absent visual evidence: truck fragments and image-edge/mask-overlap annotations complicate some severe cases. No GT, episode, matcher or frame is excluded. Scalar decision repair leaves the original geometry/refiner state and future acquisition intact; it does not establish repaired 3D estimates or extend the frozen selector theorem. Detector mean/p95 requested pixels are reported; classifier overhead and Jetson parity remain unmeasured. [Cases](handoff/OPTIMIZATION_R5_20261024/CASE_REVIEW.md), [calibration](handoff/OPTIMIZATION_R5_20261024/CALIBRATION.md), [synthesis](handoff/OPTIMIZATION_R5_20261024/TONG_HOP.md). Mirror: results/vuw_summaries/OPTIMIZATION_R5_20261024/.
'''
def append(path,marker,text):
    p=ROOT/path;s=p.read_text()
    if marker not in s:p.write_text(s.rstrip()+'\n\n'+text.strip()+'\n')
append('docs/paper/03_results.md',TITLE,result)
append('docs/paper/handoff/README.md','### R5 — conditional decision repair','''
### R5 — conditional decision repair (complete, 2026-10-25)

[Report](OPTIMIZATION_R5_20261024/REPORT.md), [104 development rows](OPTIMIZATION_R5_20261024/all_rows.csv), [incremental effects](OPTIMIZATION_R5_20261024/incremental_effects.pdf), [nine inspected cases](OPTIMIZATION_R5_20261024/CASE_REVIEW.md), [conditional calibration](OPTIMIZATION_R5_20261024/CALIBRATION.md), [completion audit](OPTIMIZATION_R5_20261024/completion_audit.json). All 23 jobs complete. Conditional cheap-evidence retention adds modest loss gains to R3 point+extent, concentrated in few units, with additional false braking. Learned disagreement gates do not solve hard misses. Frozen core, test data and K34 untouched; no new Jetson latency claim or promotion.
''')
p=ROOT/'docs/paper/06_pending.md';s=p.read_text();row='| Decision-repair optimization round 5 (2026-10-24–25) | DONE: 23 jobs, 104 full-development rows, 8 small classifiers, 9 inspected case panels. Conditional cheap retention adds modest concentrated gains to R3; hard misses and cross-dataset tradeoffs remain. All rows/failed export attempts retained; no core promotion or new Jetson timing. | handoff/OPTIMIZATION_R5_20261024/REPORT.md; CASE_REVIEW.md; CALIBRATION.md; completion_audit.json |\n'
if 'Decision-repair optimization round 5' not in s:
    anchor='| K20 PDM-Closed probing cost, variants, pairs |';assert anchor in s;s=s.replace(anchor,row+anchor);p.write_text(s)
append('docs/directions_log.md','## 2026-10-25 — R5 decision repair', '''
## 2026-10-25 — R5 decision repair

User redirected the optimization focus from speed to decision weaknesses. Completed the full population failure audit, training-only simple ordinal gates with unmatched-negative controls, conditional refiner-disagreement training, and all 104 full-development rows. 23 supervised jobs complete; two initial statistics-export failures are retained and corrected. No frozen-method/test changes or Jetson reservation.

Most useful incremental mechanism: preserve current cheap evidence when an executed verification removes it, only with learned action corroboration. On R3 point+extent, q=.5 adds one supported nuScenes episode rescue and reduces loss by 5.9%; AV2 loss changes only 0.48% and misses stay 43. Small additional false-brake exposure remains. Gains concentrate in four/nine units; q=.8 nuScenes gain is entirely one night scene. Ungated retention is negative, and new confidence gates cannot replace R3's stronger severe-frame behavior.

Direct images show fragmented trucks, a cheap-detected night car lost by verification, and near/edge AV2 boxes. A missing frozen IoU match is not necessarily absent visual evidence; image-edge/mask flags do not certify invalid annotations. All GT remains in the primary scores. Conditional calibration fails to generalize strongly from original acquisition to deployed disagreements; conditional training has only two/30 HARD positives. An extra AV2 conditional-model rescue lacks decisive-object evidence, so it is not a solved mechanism.

Reports, every row/CI, severity and gain-concentration CSVs, nine image panels and four result figures: docs/paper/handoff/OPTIMIZATION_R5_20261024/. No candidate promoted. Future work should improve actual evidence/identity under the deployed acquisition rather than add global confidence corrections. R5 is complete; these are research hypotheses, not active jobs.
''')
print('Updated paper results, pending table, handoff index and directions log.')
