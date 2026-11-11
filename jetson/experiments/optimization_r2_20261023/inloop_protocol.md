# Fresh development inference, fixed before execution

The completed frozen-query CPU screen is exploratory. It found limited extra
coverage from motion floors; half-log refinement rescued episodes but increased
false alarms. Camera anchoring lost one AV2 episode; median smoothing lost two
and rescued two. All rows remain in the screen report.

Fresh in-loop rows use independent actual observation histories:
1. Original v6-S replay control.
2. Round-1 B2K-A + retrospective recovery replay control.
3. B2K-A + proactive completion (no retrospective recovery).
4. B2K-A + proactive completion + retrospective recovery.
5. B2K-A + retrospective recovery + three-observation width/height looming floor.

Each acquisition row reports original no-refiner, v3a, and fixed half-log-v3a
actions. The latter is an explicitly evaluated tradeoff control, not a silently
selected replacement. Main checkpoints, planner thresholds and source stay frozen.

Proactive completion keeps at most one outstanding object for the next two
frames. Arm it only for an actual verified detection without an original C1
three-frame chain, with raw demand below DECELERATE, valid in-corridor range
<=80 m, and a mutually unique same-class IoU>=0.3 cheap match whose demand is
at least DECELERATE. Highest cheap demand wins, then current detection index.
For the next query, use the current unique cheap match to locate that object;
otherwise use its last actual verified box. Request context 2 at scale 960,
coalescing if already covered by the normal first-batch field. This location
prediction is never inserted into C1 history as an observation. Only real
current full detections enter history. Retire after two frames or completion
of an original C1 chain. A new current measured trigger may arm a new item.
No future images or GT enter acquisition. Additional crops share the original
B2K-A first batch; the two-batch bound remains, but total crop executions and
pixels may increase and must be reported.

The motion floor uses the screen's fixed +/-1 detector-pixel sensitivity and
mutually unique same-class links in the original two previous verified scenes.
It augments C1 during every lift, so field, support and refinement are recomputed
from the changed scene. No synthetic observation or extra detector call is
introduced directly by the floor. This is an engineering estimate, not a
calibrated uncertainty bound. It can add queries indirectly.

Smoke gate: all nine diagnostic-case units, full sequences from frame zero.
Preserve rich traces for those units, including cheap boxes, original chain
indices and retrospective crop detections/association outcomes. Baseline and
round-1 control actions/costs/pixels must match the prior full-development run.
Smoke outcomes do not determine which rows to report or whether to omit units.
Then evaluate all 152 nuScenes and 300 AV2 CFG-DEV units; AV2 fixed y0=1830 mask.
Every row, paired unit bootstrap 2,000 seed0, original cheap-previous jerk term.
Compare new policies with both v6-S and round-1 combined. Identify rescued/lost
episodes, hard-demand misses, loss, false alarms, onset, pixels and calls.

GPU execution follows the unchanged wrapper, SHARE=1 MIN_FREE_MB=10000, two
workers per wrapper. Inference caches are shared only within a current frame
for exactly identical detector inputs; bookkeeping charges every requested
execution. These GPU runs are not latency measurements. New Jetson timing is
deferred until a credible quality/cost candidate exists; no test-based selection.
