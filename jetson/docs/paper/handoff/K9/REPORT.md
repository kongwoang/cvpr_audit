# K9 — a second max-composable consumer: time-to-collision warning (2026-10-04)

**Consumer (defined in `instr_eval.py`, md5 2177d413…).** Per object o (monocular lift: range z, lateral extent, ttc
from the looming range rate): demand d(o) = 1 / ttc(o) if o is inside the ego corridor (|lateral| overlap with ±1.2 m,
the braking controller's corridor_half_w) and z ≤ 80 m, else 0. Action = quantised max of demands: 0 none,
1 WARN if max d ≥ 1/4 s⁻¹ (ttc ≤ 4 s), 2 BRAKE if max d ≥ 1/2 s⁻¹ (ttc ≤ 2 s) — a member of the proposition's class.
Required action = the same rule on the ground-truth geometry (GT range and range rate). Loss: cost matrix
C[required, taken] = [[0, 1, 5], [3, 0, 2], [10, 4, 0]] (missed brake 10, brake taken as warn 4, missed warn 3, false brake
5, over-reaction warn→brake 2, false warn 1); G = Σ(C_cheap − C_m) / Σ C_cheap, no history term.
**Field / support for THIS consumer (closed form, no planner calls):** a static probe (1.8 m wide) at grid cell (r, y)
inside the corridor, closing at ego speed + v_close (v_close of the dataset's configuration: nuScenes 10, AV2 0), is
sensitive if (v + v_close)/r reaches the next action's threshold; support = objects whose own demand reaches the
current action's threshold. Same loop, guard, merge, crop scale and history as ours (`vtf.spec_general.run`).
Uncertainty crops at ours' per-frame crop area as the same-budget control.

## Results (consumer2.csv; scene / log bootstrap of ours − method, 2,000)
| split | cheap | u960 | **ours (TTC field + support)** | uncertainty (same area) |
|---|---|---|---|---|
| nuScenes new val (development) | 0 | 0.165 (px 0.384) | **0.187** (px 0.124; − u960 [−0.045, +0.100]) | 0.192 (px 0.178) |
| nuScenes new test (post-hoc) | 0 | 0.091 | **0.114** (px 0.121; − u960 [−0.025, +0.081]) | 0.149 (px 0.167; ours − it [−0.088, +0.017]) |
| AV2 val (post-hoc) | 0 | 0.071 | **0.139** (px 0.059; − u960 [+0.021, +0.122]) | **0.211** (px 0.092; ours − it [−0.124, −0.025]) |
Agreement with π(F) (u960, TTC consumer): nuScenes test ours 0.957 vs uncertainty 0.955, cheap 0.923; AV2 ours 0.825 vs
0.816, cheap 0.763.

**Reading / against the method:** the method transfers to a second consumer of the class without change: ours beats
the whole frame at u960 on AV2 (significant) and matches it on nuScenes, at 1/3–1/4 of its pixels. But **uncertainty
crops at the same crop area beat ours on AV2 (significant) and on nuScenes test (point estimate)** for this consumer —
the TTC demand is driven by range-rate errors of already-detected objects, which re-detecting uncertain draft boxes
fixes, while the TTC field (with v_close 0 on AV2) is small. Ours' px here is the TTC loop's own crops (not the braking
loop's).

## Follow-up 2026-10-04: dev tuning and an equal-pixel control
**Script and runs.** `remote/A_method/k9_tune.py` (md5 6cec0743…) on VUW, via run_when_gpu_free.sh:
- Dev grid: `k9_tune_chain.sh`. Test, run once with the dev-selected configurations: `k9_test_chain.sh`.
- Uncertainty G-vs-px curve: `k9_curve_chain.sh`.
- Sources: `results/vuw_summaries/k9_tune/summary_*.csv`.
- **Test-split numbers below are post-hoc analysis.**

**Why a new control.** The "same area" uncertainty control above uses 1.4–1.6× ours' detector pixels: many small crops are each rounded up to a 32-px canvas, and its guard grows crops. The new control fills uncertainty crops strictly up to a multiple of ours' detector pixels on each frame, counting guard re-detections.

**Dev grid** (`consumer2_dev_grid.csv`): closing speed of the field probe vc ∈ {0, 5, 10, 15} × support margin m ∈ {0, 0.25, 0.5}.
- nuScenes new val: best is vc 0 / m 0.25, G 0.201 at px 0.090 (default 0.187 at px 0.124).
- AV2 dev (60 train logs): the default vc 0 / m 0 stays best, G 0.108 at px 0.064.
- On both dev splits the strictly px-matched uncertainty control is lower: 0.135 and 0.100.

**Test** (`consumer2_followup_test.csv`, `consumer2_G_vs_px.{pdf,png,csv}`):
- **The nuScenes dev choice did not transfer:** 0.110 at px 0.088 vs the default 0.114 at px 0.121. Report the default.
- **Uncertainty curve at 0.75 / 1 / 1.25 / 1.5 / 2× ours' pixels:**
  - nuScenes: G 0.080 / 0.111 / 0.132 / 0.136 / 0.158 at px 0.069–0.124.
  - AV2: G 0.045 / 0.080 / 0.100 / 0.127 / 0.165 at px 0.038–0.070.
- **At equal pixels, ours and uncertainty crops are not significantly different:**
  - nuScenes: ours (px 0.121) − uncertainty at 2× (px 0.124) = **−0.044 [−0.100, +0.008]**.
  - AV2: ours (px 0.059) − uncertainty at 1.5× (px 0.058) = **+0.012 [−0.042, +0.065]**.

**Reading (correction of the K9 statement above).**
- K9's "uncertainty crops beat ours on AV2 (significant)" came from the uncertainty control using more pixels. At equal pixels it does not hold.
- For this consumer, the field + support loop is **on a par** with the best uncertainty heuristic at the same pixels: below its curve on nuScenes (point estimate −0.044, n.s.), on it on AV2.
- Both beat u960 on AV2 at about a quarter of its pixels.
- For the TTC consumer, ours' advantage is therefore the guarantee (exactness w.r.t. π(F) under locality, 0.957 / 0.825 agreement), not a higher G than a well-chosen heuristic.

**Note (2026-10-05, K10):** every AV2 number in this report is unmasked. The AV2 ego hood drives 51% of the cheap draft's actions (81% of its loss); see handoff/K10 for the masked re-evaluation of the AV2 table.
