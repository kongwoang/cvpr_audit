# K12: locality on AV2, instrumented pixels, routers re-matched, fine-tuned verifier + refiner (2026-10-05)

All numbers below are **post-hoc analysis** on test splits (nuScenes new test, AV2 val, Argoverse-HD val). Nothing was tuned. Intervals are scene / log bootstraps with 2,000 resamples. Scripts are in `remote/A_method/` (VUW `scripts/A_method/`); summaries are in `results/vuw_summaries/k12/`.

## (a) Locality decomposition on AV2 (`locality_av2.csv`)
- **Script:** `k12_locality.py` (md5 841da74f…), on the first 50 AV2 val logs (3,958 frames).
- **What was compared:** ours without refiner (the instr_eval loop) against F = u960. Three variants:
  - real crops;
  - exact locality: each crop returns F's own detections whose centres lie inside it;
  - exact locality + F's previous detections as the ttc history.
- **Each variant is run with v_close 0 (the AV2 configuration) and with v_close 10 (nuScenes').**

Agreement with π(F) [95% CI]:

| | v_close 0 | v_close 10 |
|---|---|---|
| cheap draft | 0.541 [0.488, 0.596] | — |
| real crops | 0.802 [0.762, 0.840] | 0.812 [0.772, 0.847] |
| exact locality | 0.923 [0.904, 0.941] | 0.963 [0.952, 0.973] |
| + F's ttc history | 0.945 [0.929, 0.959] | 0.975 [0.964, 0.983] |

**Reading.**
- As on nuScenes (0.950 → 0.990 → 0.994), most of the gap to π(F) comes from the crop detector not reproducing F (locality): it accounts for 12 points on AV2.
- With the static probe (v_close 0), exact locality still misses 5.5 points. With v_close 10 it misses 2.5. So about 4 points of the residual come from the static probe not covering closing objects (only 1 point with real crops), at 1.5× the crops per frame.
- AV2's real-crop agreement over these 50 logs (0.802) matches K2's 0.810 over all 150.

## (b) Instrumented network pixels (`px_instrumented.csv`)
Detector input tensors: draft canvas plus every crop canvas, including guard re-detections and 32-px rounding.

| Method | nuScenes test px (mean / p95) | G (no refiner) | AV2 val px (mean / p95) | G (no refiner) |
|---|---|---|---|---|
| ours | **0.100** / 0.248 | 0.135 | **0.074** / 0.160 | 0.537 |
| uncertainty crops | 0.133 / 0.493 | −0.092 | 0.109 / 0.318 | 0.428 |
| temporal_crit crops | 0.129 / 0.459 | 0.115 | 0.111 / 0.321 | 0.143 |
| random crops | 0.185 / 0.693 | −0.083 | 0.131 / 0.413 | 0.106 |
| traj_support crops | 0.140 (mean only) | 0.055 | not run on AV2 (K1) | |

- The heuristic rows use the strict budget of ours' per-frame crop area (instr_eval, same machinery). **At equal crop area they spend 30–85% more detector pixels than ours**: many small crops, each rounded up to 32 px, plus more guard re-detections.
- traj_support: `baselines_eq.py --strict --arms traj_support` under the counting launcher `k12_count_run.py`. That script always re-runs v3 as well, so v3's instrumented crop pixels (0.0577 per frame) were subtracted from the total (2.04e9 px over 9,181 frames).
- Argoverse-HD: ours final has 0.070 (p95 0.172), counted in the K10 run; K7's area-based value was 0.065.

## (c) Routers re-matched to ours' tensor pixels (`routers_rematched.csv`)
- **Script:** `k12_routers.py` (md5 b1c32cc2…).
- **What was redone:** the original scores and calibration were kept; ρ is now matched to 0.100 (nuScenes) and 0.074 (AV2) instead of the area-based pixels.
- **ours** = final configuration with refiner (nuScenes 0.171, AV2 0.561).

| Router at matched pixels | nuScenes G | ours − it [95% CI] | AV2 G | ours − it [95% CI] |
|---|---|---|---|---|
| oracle frame router (cond / skip ρ) | 0.345 / 0.345 | [−0.252, −0.090] | 0.413 / 0.450 | [+0.108, +0.189] / [+0.075, +0.148] |
| Budget-Adaptive Routing official (best) | 0.044 (skip_offbin) | [+0.048, +0.215] | 0.087 (cond_moric) | [+0.430, +0.521] |
| decision-value targets (best) | 0.020 (cond_V) | [+0.050, +0.247] | 0.252 (cond_V) | [+0.268, +0.349] |

**Reading.** The conclusions of the router table do not change:
- ours beats every learned router on both datasets;
- the oracle frame router is above ours on nuScenes and below ours on AV2.

**Caveat:** on AV2 the oracle row and every AV2 G share the hood artefact quantified in K10.

## (d) nuScenes fine-tuned verifier + refiner (`ft_refiner.csv`)
- **Script:** `k12_ftref.py` (md5 8d97170e…).
- **Setup:** exactly fovea_ft.py's "v3 with the fine-tuned verifier":
  - YOLOv8s fine-tuned at 960 on the new train scenes, seeds 0 and 1;
  - the frozen refiner v1 then applied to the support objects from crops.
- **Parity:** without the refiner, the reruns are identical to fovea_ft on 100% of frames.

| Seed | v3-ft | **v3-ft + refiner v1** | u960-ft | v3-ft+ref − v3-ft | v3-ft+ref − u960-ft |
|---|---|---|---|---|---|
| 0 | 0.297 | **0.330** | 0.241 | [+0.009, +0.068] | [+0.035, +0.156] |
| 1 | 0.295 | **0.332** | 0.272 | [+0.008, +0.077] | [−0.004, +0.138] |

**Reading.**
- The refiner helps the fine-tuned verifier as well: +0.033 / +0.038, significant for both seeds.
- With the refiner, ours beats u960-ft significantly with seed 0; with seed 1 the point estimate is +0.060 but the CI includes 0.
- The mean over the two seeds is 0.331 vs 0.257.

**Note (2026-10-05, K10):** every AV2 number in this report is unmasked. The AV2 ego hood drives 51% of the cheap draft's actions (81% of its loss); see handoff/K10 for the masked re-evaluation of the AV2 table.
