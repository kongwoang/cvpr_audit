"""Index completed R6 artifacts without changing the frozen method."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TITLE='### R6 — current-observation corroboration (2026-10-25)'
def append(path,marker,text):
    p=ROOT/path;s=p.read_text()
    if marker not in s:p.write_text(s.rstrip()+'\n\n'+text.strip()+'\n')
append('docs/paper/03_results.md',TITLE,TITLE+'''

Completed the final paper-informed sandbox round: 58 full-development rows (22 new configurations plus seven controls per dataset), all 152 nuScenes scenes and 300 AV2 CFG-DEV logs, 17 remote jobs and one exclusive Jetson component window. No training, test selection, core/checkpoint/threshold/K34 change or promotion. [Report](handoff/OPTIMIZATION_R6_20261025/REPORT.md), [every row](handoff/OPTIMIZATION_R6_20261025/all_rows.csv), [paired effects](handoff/OPTIMIZATION_R6_20261025/all_contrasts.csv), [direct cases](handoff/OPTIMIZATION_R6_20261025/CASE_REVIEW.md).

Borrowing ByteTrack's low-score continuation principle, R6 requires a current crop proposal and previous full observations, then caps the new action by observed historical actions. Two compositions with the R5 q=.5 candidate were explicitly declared after the nuScenes first-stage screen and before seeing AV2 results. They are adaptive, not confirmation. All 112,611 stored full boxes exactly match replayed high-score proposals; 100,022 low-score boxes were recovered from the same current queries, without extra logical detector pixels.

Against R5, the one-lag composition changes loss by −.012978 [−.038986,0] / −.000936 [−.002454,−.000085] (nuScenes / AV2), while adding +.326958 [0,.817929] / +.126769 [0,.295708] false-braking frames per1,000. The two-lag composition changes loss by −.005969 [−.017931,0] / −.000683 [−.002041,0]; no new false-HARD frame, no new nuScenes false frame, one new AV2 false frame. Misses remain34/99 and43/366. All intervals paired unit bootstrap2,000 seed0, pointwise exploratory.

The one-/two-lag rules repair two/one correctly supported HARD frames in nuScenes and one in AV2, within episodes already detected elsewhere. All nuScenes incremental gain is one night-car scene; AV2 gains remain concentrated in the same near-bus log seen in R5. An aggressive combination reaches31/38 misses but markedly increases false braking and lowers AV2 G. None of its five additional AV2 episodes has matched decisive-object evidence. Existence persistence does not fix wrong range/corridor estimates, truck fragments or unobserved near-edge objects.

The Jetson experiment measures only the bounded-readout CPU component on stored observations, three repetitions of all29,782 frames, with178,692 action evaluations and zero cross-platform mismatches. It excludes detector, refiner, R5 classifier, proposal deduplication/copies/buffering; it is not full-pipeline latency or trt32 parity. [Timing scope and intervals](handoff/OPTIMIZATION_R6_20261025/jetson/REPORT.md). Eight inspected three-frame case panels, every negative row, corrected-caption archive and source/exit/hardware audits are included. Mirror: results/vuw_summaries/OPTIMIZATION_R6_20261025/.
''')
append('docs/paper/handoff/README.md',TITLE,TITLE+'''

[Report](OPTIMIZATION_R6_20261025/REPORT.md), [58 rows](OPTIMIZATION_R6_20261025/all_rows.csv), [research rationale](OPTIMIZATION_R6_20261025/RESEARCH_NOTES.md), [eight direct cases](OPTIMIZATION_R6_20261025/CASE_REVIEW.md), [Jetson component timing](OPTIMIZATION_R6_20261025/jetson/REPORT.md), [completion audit](OPTIMIZATION_R6_20261025/completion_audit.json). Bounded continuation of current low-score evidence adds small, concentrated HARD-frame/loss gains to R5; episode misses remain unchanged and broad retention is unfavorable. No core promotion, test tuning or end-to-end latency claim.
''')
p=ROOT/'docs/paper/06_pending.md';s=p.read_text();label='Final observation-corroboration round 6 (2026-10-25)'
if label not in s:
    anchor='| K20 PDM-Closed probing cost, variants, pairs |';assert anchor in s
    s=s.replace(anchor,'| '+label+' | DONE: 17 remote jobs, 58 complete development rows, eight inspected case panels, three-repetition Jetson readout benchmark. Small concentrated HARD-frame gains; no new episode rescue for bounded compositions. No core promotion or remaining R6 job. | handoff/OPTIMIZATION_R6_20261025/REPORT.md; CASE_REVIEW.md; completion_audit.json |\n'+anchor);p.write_text(s)
append('docs/directions_log.md','## 2026-10-25 — R6 final observation study','''
## 2026-10-25 — R6 final observation study

The user requested a final creative, literature-informed improvement round. Read ByteTrack, OC-SORT, threshold-dependent Bernoulli filtering and selective one-sided prediction against the source and observed failures. The concrete experiment reuses weak current crop detections discarded by the normal confidence threshold, requiring explicit current/previous observations and bounded historical action. It creates neither low-score births nor unobserved ghost braking.

Completed all 58 development rows (20 initial rules plus two adaptive compositions and seven controls on each dataset), 17 remote jobs, eight three-frame panels, source/proposal parity and one exclusive Jetson component window. Box replay is exact for112,611 final full boxes;178,692 Jetson readout action evaluations have zero mismatch. No main source/checkpoint/threshold/test change. Two figure captions were corrected to show their actual residual-low policy; initial renders remain archived, scores unchanged.

R5 plus two-lag bounded continuation reduces loss1.88%/0.20% on nuScenes/AV2, with no added false HARD and only one extra AV2 false frame. Misses remain34/99 and43/366. One-lag continuation has larger small gains but two/three extra false frames, including one HARD on each dataset. All nuScenes gain is one night scene; AV2 gain is concentrated in the near-bus log. Pointwise intervals commonly touch/cross zero. Broad continuation reduces episode misses while increasing false braking; its extra AV2 episode rescues have no matched decisive evidence.

The inspected night car and bus show real recoverable weak evidence. A real side bus with wrong demand and a recurring ego-hood artifact show why observation persistence alone cannot validate geometry. Truck fragmentation and near-edge visibility cases remain unresolved. The result is a small complementary decision repair, not a general solution to the method's HARD/miss weakness. No candidate is promoted; all required R6 work is complete. Standalone CPU timing cannot substitute for end-to-end combined-method latency.

Self-contained report and mirror: docs/paper/handoff/OPTIMIZATION_R6_20261025/ and results/vuw_summaries/OPTIMIZATION_R6_20261025/.
''')
print('Updated paper results, pending table, handoff index and directions log.')
