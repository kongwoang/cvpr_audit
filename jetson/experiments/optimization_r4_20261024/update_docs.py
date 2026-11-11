"""Add completed R4 evidence to the local writer index, without changing frozen method."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R4_20261024'

def append_once(path,marker,body):
    p=ROOT/path;s=p.read_text()
    if marker in s:return
    p.write_text(s.rstrip()+'\n\n'+body.strip()+'\n')

def main():
    audit=json.loads((OUT/'completion_audit.json').read_text());assert audit['status']=='complete'
    marker='R4 — simple mechanisms and exact score reuse'
    result='''## R4 — simple mechanisms and exact score reuse (2026-10-24, complete)

Isolated follow-up to R3: all 8 supervised GPU jobs, 58 development rows across acquisition/readout suites (42 distinct dataset/policy/refiner rows), two Jetson windows with three repetitions per dataset, and an exact CPU bottleneck-selector optimization. Frozen core, checkpoints, thresholds and K34 unchanged; no promotion or test-based quality selection.

The clearest new result is implementation-only: K31's fixed bottleneck selector reuses base object scores and batches probe-only columns. On K34's frozen 1,000-token / 125-unit NAVSIM navtest timing subset (post-hoc runtime only), mean 7×7 cost 100.969→10.957 ms, speedup 9.215× [8.245,10.182]; 3×3 24.112→6.392 ms, 3.773× [3.530,3.995]. Same host/process, one CPU thread, three repetitions, all 6,000 action/field/support comparisons identical. This is neither a PDM-Closed speedup nor full-perception timing.

Braking quality still trades off: objectwise cheap/raw/refined median reduces AV2 misses 49→37/366 and insufficient loss −0.05336 [−0.07145,−0.03851], but false-braking frames/1,000 53.29→108.94; nuScenes misses unchanged, no loss benefit. Nine of twelve AV2 rescues have decisive-object evidence, none is a new hard-peak rescue. One-round extent harms nuScenes despite lower pixels; containment/union/semantic changes add no episode rescue. Hard corroboration worsens AV2 G. All negative rows retained.

Refiner abstention preserves the new median policy, not frozen v6-S. Direct implementation overhead and its exact optimized implementation are timed separately; full/skip and old/new decisions are audited. Mean/p95/p99, threshold exceedance shares, pointwise paired bootstrap 2,000 seed0, all backend differences and hardware restoration are in the handoff. See [R4 report](handoff/OPTIMIZATION_R4_20261024/REPORT.md), [component review](handoff/OPTIMIZATION_R4_20261024/COMPONENT_REVIEW.md), [synthesis](handoff/OPTIMIZATION_R4_20261024/TONG_HOP.md), [completion audit](handoff/OPTIMIZATION_R4_20261024/completion_audit.json). Mirror: results/vuw_summaries/OPTIMIZATION_R4_20261024/.
'''
    append_once('docs/paper/03_results.md',marker,result)
    pending=ROOT/'docs/paper/06_pending.md';s=pending.read_text()
    if 'Simple-mechanism optimization round 4' not in s:
        row='| Simple-mechanism optimization round 4 (2026-10-24) | DONE: 8 GPU jobs, 58 development rows, two 3-repeat Jetson windows. Exact K31 selector cache gives 9.21× / 3.77× stage speedup on fixed post-hoc runtime tokens with zero action/field/support differences. Simple braking voting remains a dataset-dependent risk/false-braking tradeoff; no core promotion. | handoff/OPTIMIZATION_R4_20261024/REPORT.md; COMPONENT_REVIEW.md; TONG_HOP.md; completion_audit.json |\n'
        s=s.replace('| K20 PDM-Closed probing cost, variants, pairs |',row+'| K20 PDM-Closed probing cost, variants, pairs |');pending.write_text(s)
    append_once('docs/paper/handoff/README.md',marker,'''### R4 — simple mechanisms and exact score reuse (complete, 2026-10-24)

[Report](OPTIMIZATION_R4_20261024/REPORT.md), [Vietnamese synthesis](OPTIMIZATION_R4_20261024/TONG_HOP.md), [13-component review](OPTIMIZATION_R4_20261024/COMPONENT_REVIEW.md), [exactness notes](OPTIMIZATION_R4_20261024/SIMPLIFICATION_NOTES.md), [completion audit](OPTIMIZATION_R4_20261024/completion_audit.json). All 58 development rows and both Jetson windows retained. Simple voting helps AV2 missed episodes at more false braking; no common quality dominance. Exact K31 bottleneck-score reuse speeds its selector stage 9.21× / 3.77× (7×7 / 3×3), with all 6,000 action/field/support comparisons identical on fixed post-hoc NAVSIM timing tokens. Main frozen; no test-quality selection or promotion.
''')
    append_once('docs/directions_log.md',marker,'''## 2026-10-24 — R4 — simple mechanisms and exact score reuse

Reviewed 13 components and tested fixed simple alternatives without new learned heads or core changes. All 8 GPU jobs and 58 development suite rows complete. Containment merge, exact crop union and semantic group checks add no episode rescues; one-round extent trades pixels for worse nuScenes loss. Objectwise median improves AV2 moderate episode misses 49→37 and loss, but doubles false-frame exposure and does not help nuScenes. Hard-vote episode counts hide excess severe braking. Preserve all negative and unsupported-attribution cases.

Two exclusive Jetson windows distinguish direct voting overhead from an exact action-equivalent implementation that skips matching when no refiner support exists, reuses matches and votes on ordinal actions. All source/engine hashes and output parity audited, hardware restored. No new median policy is promoted.

Source algebra uncovered a stronger exact optimization in K31 NAVSIM: reuse the base bottleneck score instead of rescoring every existing object for every probe. Fixed same-host three-repeat benchmark on K34's existing 1,000-token navtest timing subset, explicitly post-hoc runtime/parity only, reports mean 100.969→10.957 ms (7×7) and 24.112→6.392 ms (3×3); speedups 9.215× [8.245,10.182] / 3.773× [3.530,3.995]. All 6,000 action/field/support comparisons and 800 synthetic empty/tie edge cases match. This preserves the selector, not a surrogate planner replacement; no claim about total perception or generic PDM-Closed speedup.

Evidence: docs/paper/handoff/OPTIMIZATION_R4_20261024/REPORT.md, COMPONENT_REVIEW.md, SIMPLIFICATION_NOTES.md and completion_audit.json. Main source and frozen reports unchanged; summaries mirrored locally.
''')

if __name__=='__main__':main()
