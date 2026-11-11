# W1 development sandbox — registered 2026-10-23 before new inference

Authorization: user requested trying the proposals separately without changing main source.
All code, generated forks, logs, records and reports belong to this sandbox or its handoff.
No writes to src/, remote/A_method/, deployed scripts/, frozen weights or K34 outputs.
Remote root: $B/experiments/w1_dev_20261023. Main source hashes are checked before and after jobs.

Development only: nuScenes val (152 scenes), AV2 CFG-DEV (300 logs, fixed y=1830 ego mask).
Reference manifests: existing K30 dev_devref per-unit records; verify equality of units, frames,
ground truth, cheap actions and frozen v6-S actions/costs. Training split disjointness is audited.
No A-HD, nuScenes test, AV2 test or new training/model selection in this study.

Fixed independent variants, no combination selection:
- v6S: frozen baseline, with/without v3a.
- v6S_retro: existing C1 weights; at most one current crop canvas re-detected at t-1 and t-2
  per frame (two extra detector executions). Trigger: no C1 chain, finite corridor geometry,
  class-compatible unambiguous cheap match whose demand >= thr_decel while current demand is
  below thr_decel. Order by cheap demand. Same detector scale; genuine consecutive images;
  unique class-compatible IoU >= .3 associations; do not reuse a stale box as an observation.
- v6S_close: start with the frozen loop; persist once-only v3a corrections and refine newly
  supporting verified objects; recompute field/support and permit one extra verification round.
  Corrections apply to raw geometry, never compounded. Report unresolved closure at the cap.
- v6S_recheck: start with the frozen loop; one extra context-3 crop of the highest-demand cheap
  support object removed inside executed crops without a compatible verified match. Existing
  truncation guard applies. No forced action floor; unresolved absence uses the normal merge.

Controls: cheap, u960/native + C1 + v3a, hybrid tsp8 and unc2 at the baseline v6-S crop-pixel budget.
Whole-frame closure also reported with the same persistent-refinement rule (no added detections).
Other generic-component controls and equal-total-budget variants are required before claiming a
selector advantage; this initial screen does not establish that claim. No Jetson timing claim
from cached/shared GPU development runs. All extra historical/guard input pixels and calls counted.

First run: first four lexicographic reference units per dataset, for parity and mechanism smoke.
Then all dev units if structural tests and baseline parity pass. Smoke units are part of dev,
not an independent validation set. No automatic promotion or main-code update.

Outputs: every row, required-episode miss rate (K31 gap<=1), onset delay, false episodes/1000,
insufficient-braking loss, high-demand strata, G, compute, and mechanism counters. Paired unit
bootstrap 2000, seed 0. Equal frame sets required. Positive direction for miss/loss is worse.
Screening rule: report Pareto tradeoffs, no selected winner. Future selection requires predeclared
false-alarm/latency margins and generic-component/equal-budget controls. Existing test results
inspired these hypotheses; any later reuse of those test sets must be labeled post-hoc.
