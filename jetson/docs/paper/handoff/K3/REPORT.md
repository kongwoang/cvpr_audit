# K3 — necessity ablations (post-hoc analysis on the test splits, 2026-10-04)

**Splits:** nuScenes new test (228 scenes, 9,181 frames) and AV2 val (150 logs, 11,818 frames), labelled post-hoc; one
table per split, no mixing. Same instrumented run as K2 (`instr_eval.py` md5 2177d413…, `instr_analyze.py` md5
73cacf72…). Each variant changes one thing in the final configuration (nuScenes v3 + refiner v1; AV2 v4-lite +
refiner v2); the 'full' row is `vtf.speculative.run` itself (parity with the confirmatory runs: K2). Variants 2–8 use a
switchable copy of the loop (`run_variant`), identical to the frozen loop when all switches are on.
**What already existed:** (5) one round — the nuScenes operating points (handoff/K8/plot_operating_points_nusc.csv:
v_close 10, 1 round, G 0.016 / + ref 0.059) match this run exactly; temporal_crit (K1) is not variant 3: it crops the
PREVIOUS frame's merged objects ranked by braking demand (+25% context) within a fixed budget, while support-only crops
the CURRENT draft's support set (objects whose demand alone reaches the current action), iterated, with no budget.

Columns (ablation_{nusc,av2}.csv): G without / with refiner, network px (detector tensors incl. guard re-detections),
crops / rounds / guard re-detections per frame, agreement with π(F) (F = u960), harmed share (with refiner;
harmed / affected frames), and the scene / log bootstrap interval of G(full) − G(variant), without and with refiner.

## nuScenes new test (G with refiner; full − variant)
full 0.170 (px 0.100) · **field only 0.118** [−0.077, +0.181] · **support only −0.107** [+0.018, +0.498] · pivots 0.171
[−0.026, +0.031] · **one round 0.059** [−0.008, +0.287] · no guard 0.131 [−0.032, +0.134] · draft history 0.137
[+0.001, +0.081] · no guard + draft history 0.165 [−0.059, +0.087].
## AV2 val
full 0.561 (px 0.074) · **field only −0.002** [+0.513, +0.614] · support only 0.557 [−0.012, +0.026] · **pivots 0.516**
[+0.036, +0.055] · one round 0.559 [−0.014, +0.023] · **no guard 0.552** [+0.002, +0.017] · **draft history 0.537**
[+0.009, +0.042] · no guard + draft history 0.539 [+0.005, +0.040].

## Reading
- Each part is necessary on at least one dataset, but not on both: the **field** is necessary on nuScenes (support-only
  collapses to −0.107; v_close 10 makes the field cover the corridor ahead) while on AV2 (v_close 0: empty field below
  ~2.5 m/s, K8 iii) support-only already gives 0.557 ≈ full; the **support** is necessary on AV2 (field-only −0.002)
  while on nuScenes field-only keeps 0.118 (interval of the difference spans 0).
- **Pivots vs support:** pivots lose on AV2 [+0.036, +0.055] and tie on nuScenes — the counterexample of the theory
  (two supporting objects, neither a pivot) occurs on AV2.
- **Iteration:** one round loses 0.11 on nuScenes (interval touches 0 at −0.008) and nothing on AV2.
- **Guard and merged-scene history** matter on AV2 (small, significant) and are within noise on nuScenes; dropping both
  together on nuScenes is no worse than full (0.165).
- Agreement with π(F) is highest for full on both datasets (nuScenes 0.951, AV2 0.810 without refiner).

**Against the method:** no single variant is significantly worse than full on BOTH datasets; on nuScenes only
support-only (and draft history, marginally) are significantly worse; on AV2 support-only, one round and no guard are
within noise of full. The necessity counterexamples hold dataset-wise, not universally.

**Note (2026-10-05, K10):** every AV2 number in this report is unmasked. The AV2 ego hood drives 51% of the cheap draft's actions (81% of its loss); see handoff/K10 for the masked re-evaluation of the AV2 table.
