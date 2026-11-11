# K28 phase 2: the single test pass of v6 (CONFIRMATORY, method_freeze_draft §2n / §2n.1)

v6 was frozen in §2n before the pass. This is R2's one test pass:
- nuScenes new test: 228 scenes, 9,181 frames;
- AV2 val with the ego mask: 150 logs, 11,818 frames;
- Argoverse-HD val: 24 sequences, 15,006 frames with 3D labels.

G is computed with each configuration's refiner unless a column says otherwise. Intervals are paired unit bootstraps (2,000 resamples, seed 0) of (v6 − row). The verdicts follow the K24 rule:
- **outperforms:** lower bound > 0;
- **matches:** lower bound > −0.02;
- **below:** otherwise.

"Below" therefore includes rows where v6 has the higher point estimate but the interval is wide. The column "point estimate" separates the two cases.

## 1. v6 against v5.1-S, per split and budget

| split | budget | GFLOPs / v5.1-S | G v5.1-S | G v6 | G v6 no refiner | v6 − v5.1-S [95% CI] | verdict |
|---|---|---|---|---|---|---|---|
| nuScenes new test | v6-S | 1.00 | 0.199 | 0.211 | 0.181 | +0.012 [+0.004, +0.026] | outperforms |
| nuScenes new test | v6-M | 1.35 | 0.199 | 0.298 | 0.268 | +0.099 [-0.005, +0.218] | matches |
| nuScenes new test | v6-L | 1.62 | 0.199 | 0.315 | 0.287 | +0.116 [-0.031, +0.290] | below |
| AV2 val (ego mask) | v6-S | 1.00 | 0.262 | 0.294 | 0.241 | +0.032 [+0.013, +0.050] | outperforms |
| AV2 val (ego mask) | v6-M | 1.44 | 0.262 | 0.281 | 0.230 | +0.019 [-0.012, +0.049] | matches |
| AV2 val (ego mask) | v6-L | 1.76 | 0.262 | 0.292 | 0.242 | +0.030 [+0.007, +0.053] | outperforms |
| Argoverse-HD val | v6-S | 1.01 | 0.338 | 0.418 | 0.422 | +0.079 [+0.054, +0.107] | outperforms |
| Argoverse-HD val | v6-M | 1.48 | 0.338 | 0.428 | 0.412 | +0.090 [+0.059, +0.119] | outperforms |
| Argoverse-HD val | v6-L | 1.77 | 0.338 | 0.431 | 0.419 | +0.092 [+0.065, +0.120] | outperforms |

- **v6-S** (YOLOv8s crops 960 + C1 + v3a) is above v5.1-S on all three splits at the same GFLOPs.
- **v6-M** (YOLOv8m crops 768 + C1 + own refiner) matches v5.1-S on nuScenes and AV2, and is above it on Argoverse-HD.
- **v6-L** (YOLO11m crops 960 + C1 + own refiner):
  - nuScenes: below v5.1-S by the verdict rule. Its point estimate is the largest of the three (+0.116), but the interval is wide (lower bound −0.031).
  - AV2 and Argoverse-HD: above v5.1-S.
- **AV2:** v6-M (0.281) and v6-L (0.292) are not above v6-S (0.294). The larger crop detectors add nothing on AV2 test.
- **nuScenes:** the M / L gain over v5.1-S is large in the point estimate (+0.10 / +0.12) and not significant.

## 2. Every row v6 does not outperform

Each baseline row is compared with the v6 configuration of its GFLOPs class (S ≤ 1.02, M ≤ 1.53, L ≤ 2.55 × v5.1-S). Rows above the L cap are compared with v6-L and flagged "over".
- In-class rows not outperformed: **102** (full list in §8.1).
- Over-budget rows not outperformed: **133** (§8.2).
- not_outperformed_v6.csv holds every (v6 configuration, row) pair, 728 pairs in all.

**In-class rows where the baseline has the higher point estimate** (the oracle router is listed separately):

| split | budget | row | GFLOPs / v5.1-S | v6 − row [95% CI] |
|---|---|---|---|---|
| AV2 | M (v6-M) | K28 heuristic tsp8@S + C1 | 1.33 | −0.015 [−0.037, +0.003] |
| AV2 | M (v6-M) | K28 heuristic tsp2@S + C1 | 1.23 | −0.005 [−0.034, +0.022] |
| AV2 | M (v6-M) | K28 whole frame YOLOv8s 512 + C1 | 1.26 | −0.004 [−0.069, +0.059] |
| AV2 | L (v6-L) | K28 heuristic tsp8@L + C1 | 2.18 | −0.050 [−0.112, +0.004] |
| AV2 | L (v6-L) | K28 heuristic tsp8@M + C1 | 2.03 | **−0.036 [−0.080, −0.001]** |
| AV2 | L (v6-L) | K28 heuristic cor8@L + C1 | 1.92 | −0.032 [−0.097, +0.027] |
| AV2 | L (v6-L) | K28 heuristic tsp0@L + C1 | 2.39 | −0.026 [−0.076, +0.014] |
| AV2 | L (v6-L) | K28 whole frame YOLOv8s 640 + C1 | 1.93 | −0.019 [−0.081, +0.034] |
| AV2 | L (v6-L) | K28 heuristic cor8@M + C1 | 1.78 | −0.017 [−0.065, +0.026] |
| AV2 | L (v6-L) | K28 heuristic tsp4@M + C1 | 1.68 | −0.012 [−0.040, +0.011] |
| AV2 | L (v6-L) | K28 heuristic cor0@L + C1 | 2.13 | −0.006 [−0.056, +0.040] |
| Argoverse-HD | L (v6-L) | K28 heuristic cor0@M + C1 | 2.11 | **−0.060 [−0.126, −0.001]** |
| nuScenes | L (v6-L) | K28 whole frame YOLO11m 448 + C1 | 2.45 | −0.003 [−0.097, +0.090] |

- **Oracle frame router** (top-k frames by the true V; not deployable), re-matched to the px of each v6 configuration (§5):
  - nuScenes: above v6-S at S (−0.154 [−0.231, −0.073]); M and L are below-verdict with point estimates −0.067 and −0.051.
  - AV2: above v6 at every budget (−0.125 … −0.138).
  - Argoverse-HD: "matches" at S (v6-S − oracle +0.049 [−0.002, +0.101], so v6-S does not outperform it; with components +0.062 [+0.014, +0.112]); v6 outperforms it at M and L. (Corrected 2026-10-14; this line first read "below v6 at every budget".)
- **Over budget** (§8.2): whole frame with YOLO11m / YOLOv8m + C1 at 2.4–52 × v5.1-S GFLOPs is above v6-L on nuScenes and AV2 at most resolutions. Examples:
  - nuScenes YOLOv8m native + C1: 0.384, against v6-L at −0.069 [−0.118, −0.025];
  - AV2 YOLO11m 1280 + C1: 0.342, against v6-L at −0.050 [−0.101, −0.008].
- **The other in-class "below" rows** have positive point estimates for v6 and wide intervals, mostly on nuScenes at budget L. Examples:
  - K28 heuristic unc6@L + C1: +0.312 [−0.021, +0.664];
  - whole frame u640 + components: +0.384 [−0.043, +0.977].

## 3. Scorecard items decided by phase 2

**S1** asks: at each budget, does v6 beat every frame / region baseline + components on ≥ 2 of 3 datasets, and is it below none? Rows below are in-class and deployable. The heuristics, whole frame and routers are counted in every form (plain, + T4, + C1, + components).

| budget | split | deployable rows | outperforms | matches | below, v6 higher | below, row higher | beats every deployable row | oracle router (true V) |
|---|---|---|---|---|---|---|---|---|
| S | nuScenes new test | 6 | 6 | 0 | 0 | 0 | yes | below, below |
| S | AV2 val (ego mask) | 4 | 4 | 0 | 0 | 0 | yes | below, below |
| S | Argoverse-HD val | 6 | 6 | 0 | 0 | 0 | yes | matches, outperforms |
| M | nuScenes new test | 64 | 55 | 4 | 5 | 0 | no | below, below |
| M | AV2 val (ego mask) | 62 | 51 | 7 | 1 | 3 | no | below, below |
| M | Argoverse-HD val | 18 | 17 | 1 | 0 | 0 | no | outperforms, outperforms |
| L | nuScenes new test | 97 | 48 | 23 | 25 | 1 | no | – |
| L | AV2 val (ego mask) | 79 | 59 | 9 | 3 | 8 | no | – |
| L | Argoverse-HD val | 84 | 81 | 2 | 0 | 1 | no | – |

- **S: met for every deployable row.** v6-S is above every in-class deployable row on all three splits.
  - The true-V oracle router is above v6-S on nuScenes and AV2.
  - Whether the oracle counts as a "frame baseline" for S1 is a presentation decision left to the user. If it counts, S is not met.
- **M: not met.** v6-M is above every deployable row on 0 of 3 splits. Remaining rows:
  - Argoverse-HD: one matches row (K28 heuristic tsp2@S + C1, +0.023 [−0.004, +0.049]).
  - nuScenes: 5 below-verdict rows, all with v6 point estimates higher.
  - AV2: 3 rows with the baseline higher, none significant.
- **L: not met.** v6-L is above every deployable row on 0 of 3 splits. Two rows are significantly above v6-L:
  - AV2 K28 heuristic tsp8@M + C1;
  - Argoverse-HD cor0@M + C1.

**S2: not met.** It asks that v6 match whole frame + components at every resolution, on all three datasets, with ≤ 25% of the px, and outperform it on ≥ 1 dataset. Comparison is v6-S against YOLOv8s whole frame, with the px share in the table:
- **vs K24 whole frame + components (T4 + v3a):**
  - Argoverse-HD: outperforms at all five resolutions.
  - AV2: outperforms at 640 and 1536; matches at 960 and native; below at 1280 (+0.022 [−0.031, +0.070]).
  - nuScenes: below at all five, with point estimates +0.280 (640), +0.114 (960), −0.058 (1280), −0.069 (1536) and +0.014 (native), and intervals 0.2–0.7 wide.
- **vs K28 whole frame + C1** (our control with the C1 lift, as the user directed on 2026-10-12):
  - nuScenes and AV2: below at every resolution. AV2's point estimates are negative from 640 up (−0.017 … −0.055; 1280: −0.055 [−0.113, −0.005]).
  - Argoverse-HD: outperforms up to 1280, matches at 1536, below at native (−0.012 [−0.048, +0.028]).
- v6-M and v6-L do not change the picture (§8.2).

| split | row | long side | G row | px v6-S / row | v6-S − row [95% CI] | verdict |
|---|---|---|---|---|---|---|
| nuScenes new test | K28 + C1 (v3a) | 448 | 0.086 | 0.97 | +0.125 [-0.066, +0.327] | below |
| nuScenes new test | K28 + C1 (v3a) | 512 | 0.124 | 0.76 | +0.087 [-0.093, +0.279] | below |
| nuScenes new test | K28 + C1 (v3a) | 640 | -0.036 | 0.45 | +0.246 [-0.070, +0.666] | below |
| nuScenes new test | K28 + C1 (v3a) | 960 | 0.112 | 0.20 | +0.099 [-0.078, +0.308] | below |
| nuScenes new test | K28 + C1 (v3a) | 1280 | 0.292 | 0.12 | -0.082 [-0.165, +0.017] | below |
| nuScenes new test | K28 + C1 (v3a) | 1536 | 0.302 | 0.08 | -0.091 [-0.218, +0.065] | below |
| nuScenes new test | K28 + C1 (v3a) | 1600 | 0.222 | 0.08 | -0.011 [-0.087, +0.122] | below |
| nuScenes new test | K24 + components (T4 + v3a) | 640 | -0.069 | 0.45 | +0.280 [-0.024, +0.691] | below |
| nuScenes new test | K24 + components (T4 + v3a) | 960 | 0.096 | 0.20 | +0.114 [-0.060, +0.320] | below |
| nuScenes new test | K24 + components (T4 + v3a) | 1280 | 0.269 | 0.12 | -0.058 [-0.151, +0.041] | below |
| nuScenes new test | K24 + components (T4 + v3a) | 1536 | 0.280 | 0.08 | -0.069 [-0.205, +0.087] | below |
| nuScenes new test | K24 + components (T4 + v3a) | native | 0.197 | 0.08 | +0.014 [-0.065, +0.143] | below |
| AV2 val (ego mask) | K28 + C1 (v3a) | 512 | 0.285 | 0.80 | +0.009 [-0.055, +0.071] | below |
| AV2 val (ego mask) | K28 + C1 (v3a) | 640 | 0.311 | 0.52 | -0.017 [-0.076, +0.035] | below |
| AV2 val (ego mask) | K28 + C1 (v3a) | 960 | 0.338 | 0.23 | -0.045 [-0.102, +0.004] | below |
| AV2 val (ego mask) | K28 + C1 (v3a) | 1280 | 0.348 | 0.13 | -0.055 [-0.113, -0.005] | below |
| AV2 val (ego mask) | K28 + C1 (v3a) | 1536 | 0.328 | 0.09 | -0.034 [-0.102, +0.025] | below |
| AV2 val (ego mask) | K28 + C1 (v3a) | 2048 | 0.294 | 0.05 | -0.000 [-0.077, +0.080] | below |
| AV2 val (ego mask) | K24 + components (T4 + v3a) | 640 | 0.219 | 0.52 | +0.075 [+0.026, +0.118] | outperforms |
| AV2 val (ego mask) | K24 + components (T4 + v3a) | 960 | 0.255 | 0.23 | +0.039 [-0.012, +0.084] | matches |
| AV2 val (ego mask) | K24 + components (T4 + v3a) | 1280 | 0.272 | 0.13 | +0.022 [-0.031, +0.070] | below |
| AV2 val (ego mask) | K24 + components (T4 + v3a) | 1536 | 0.233 | 0.09 | +0.060 [+0.003, +0.114] | outperforms |
| AV2 val (ego mask) | K24 + components (T4 + v3a) | native | 0.224 | 0.05 | +0.069 [-0.008, +0.151] | matches |
| Argoverse-HD val | K28 + C1 (v3a) | 448 | 0.247 | 1.26 | +0.170 [+0.028, +0.371] | outperforms |
| Argoverse-HD val | K28 + C1 (v3a) | 512 | 0.263 | 0.99 | +0.155 [+0.026, +0.348] | outperforms |
| Argoverse-HD val | K28 + C1 (v3a) | 640 | 0.279 | 0.61 | +0.139 [+0.026, +0.329] | outperforms |
| Argoverse-HD val | K28 + C1 (v3a) | 960 | 0.297 | 0.28 | +0.121 [+0.017, +0.320] | outperforms |
| Argoverse-HD val | K28 + C1 (v3a) | 1280 | 0.315 | 0.16 | +0.103 [+0.013, +0.278] | outperforms |
| Argoverse-HD val | K28 + C1 (v3a) | 1536 | 0.357 | 0.11 | +0.061 [-0.005, +0.172] | matches |
| Argoverse-HD val | K28 + C1 (v3a) | 1920 | 0.430 | 0.07 | -0.012 [-0.048, +0.028] | below |
| Argoverse-HD val | K24 + components (T4 + v3a) | 640 | 0.278 | 0.61 | +0.140 [+0.102, +0.180] | outperforms |
| Argoverse-HD val | K24 + components (T4 + v3a) | 960 | 0.287 | 0.28 | +0.131 [+0.080, +0.191] | outperforms |
| Argoverse-HD val | K24 + components (T4 + v3a) | 1280 | 0.262 | 0.16 | +0.156 [+0.077, +0.256] | outperforms |
| Argoverse-HD val | K24 + components (T4 + v3a) | 1536 | 0.268 | 0.11 | +0.150 [+0.073, +0.243] | outperforms |
| Argoverse-HD val | K24 + components (T4 + v3a) | native | 0.283 | 0.07 | +0.135 [+0.050, +0.233] | outperforms |

**S4, phase-2 part** (ablation of v6-S, §2n.1; the guard ablation comes from phase 3, cell ng6S). A component counts as significant on a split when the interval of v6-S minus the cell without that component is above 0.

| component | nuScenes | AV2 | Argoverse-HD | significant on ≥ 2 |
|---|---|---|---|---|
| field (v6-S − support only) | +0.306 [+0.060, +0.532] | +0.075 [+0.031, +0.130] | +0.105 [+0.020, +0.228] | yes (3 / 3) |
| support (v6-S − field only) | +0.037 [−0.080, +0.159] | +0.273 [+0.217, +0.331] | +0.338 [+0.255, +0.434] | yes (2 / 3) |
| looming, C1 vs T4 (v6-S − v5.1-S) | +0.012 [+0.004, +0.026] | +0.032 [+0.013, +0.050] | +0.079 [+0.054, +0.107] | yes (3 / 3) |
| looming, C1 vs none | +0.035 [+0.014, +0.066] | +0.086 [+0.039, +0.130] | +0.233 [+0.173, +0.310] | yes (3 / 3) |
| refiner (v6-S − C1 only) | +0.030 [−0.000, +0.068] | +0.053 [+0.011, +0.095] | −0.004 [−0.083, +0.101] | **no (1 / 3)** |
| guard | phase 3 | phase 3 | phase 3 | – |

S3 (Jetson), S5 (second detector family, fine-tuned detectors) and S6 (refiner seeds against the smallest S1 margin) come from phase 3 and the final Jetson window.

## 4. Parity

15 required checks (K28 v5.1-S and whole frame 960 vs K24): min identical share 1.0000; 24 ablation checks: min 1.0000; informational K24-form heuristics: min 0.9741 (temporal_crit + T4 differs on 0.01–2.6% of frames: the K28 heuristic re-runs the T4 lift on its own crops)

## 5. Routers and the oracle re-matched to v6 (§2n.1)

The K24 routers, picked on val, are re-matched within 2% of the mean tensor px of v6-S / M / L:
- the deployable routers are below v6 everywhere: v6 outperforms them at every budget on nuScenes and AV2;
- Argoverse-HD has no trained router, so only the oracle is computed there.

| split | router | px | within 2% | G | v6 − router [95% CI] | verdict |
|---|---|---|---|---|---|---|
| nuScenes new test | oracle frame router (true V, top-k) @ v6-S | 0.078 | True | 0.365 | -0.154 [-0.231, -0.073] | below |
| nuScenes new test | oracle frame router (true V, top-k) @ v6-S + components | 0.078 | True | 0.349 | -0.138 [-0.211, -0.057] | below |
| nuScenes new test | oracle frame router (true V, top-k) @ v6-M | 0.105 | True | 0.365 | -0.067 [-0.163, +0.010] | below |
| nuScenes new test | oracle frame router (true V, top-k) @ v6-M + components | 0.105 | True | 0.348 | -0.050 [-0.140, +0.021] | below |
| nuScenes new test | oracle frame router (true V, top-k) @ v6-L | 0.125 | True | 0.365 | -0.051 [-0.169, +0.070] | below |
| nuScenes new test | oracle frame router (true V, top-k) @ v6-L + components | 0.125 | True | 0.348 | -0.033 [-0.150, +0.083] | below |
| nuScenes new test | best Budget-Adaptive Routing (picked on val): cond_moric @ v6-S | 0.079 | True | -0.012 | +0.223 [+0.125, +0.348] | outperforms |
| nuScenes new test | best Budget-Adaptive Routing (picked on val): cond_moric @ v6-S + components | 0.079 | True | -0.017 | +0.227 [+0.131, +0.357] | outperforms |
| nuScenes new test | best decision-value router (picked on val): cond_V @ v6-S | 0.077 | True | 0.030 | +0.181 [+0.094, +0.263] | outperforms |
| nuScenes new test | best decision-value router (picked on val): cond_V @ v6-S + components | 0.077 | True | 0.033 | +0.178 [+0.092, +0.261] | outperforms |
| nuScenes new test | best Budget-Adaptive Routing (picked on val): skip_offbin @ v6-M | 0.104 | True | 0.042 | +0.256 [+0.152, +0.338] | outperforms |
| nuScenes new test | best Budget-Adaptive Routing (picked on val): skip_offbin @ v6-M + components | 0.104 | True | 0.053 | +0.245 [+0.138, +0.328] | outperforms |
| nuScenes new test | best decision-value router (picked on val): cond_V @ v6-M | 0.105 | True | 0.026 | +0.272 [+0.100, +0.484] | outperforms |
| nuScenes new test | best decision-value router (picked on val): cond_V @ v6-M + components | 0.105 | True | 0.032 | +0.266 [+0.092, +0.479] | outperforms |
| nuScenes new test | best Budget-Adaptive Routing (picked on val): skip_offbin @ v6-L | 0.128 | True | 0.068 | +0.247 [+0.106, +0.379] | outperforms |
| nuScenes new test | best Budget-Adaptive Routing (picked on val): skip_offbin @ v6-L + components | 0.128 | True | 0.078 | +0.237 [+0.096, +0.372] | outperforms |
| nuScenes new test | best decision-value router (picked on val): cond_V @ v6-L | 0.123 | True | 0.035 | +0.280 [+0.072, +0.550] | outperforms |
| nuScenes new test | best decision-value router (picked on val): cond_V @ v6-L + components | 0.123 | True | 0.040 | +0.275 [+0.065, +0.543] | outperforms |
| AV2 val (ego mask) | oracle frame router (true V, top-k) @ v6-S | 0.054 | True | 0.418 | -0.125 [-0.173, -0.080] | below |
| AV2 val (ego mask) | oracle frame router (true V, top-k) @ v6-S + components | 0.054 | True | 0.396 | -0.102 [-0.147, -0.062] | below |
| AV2 val (ego mask) | oracle frame router (true V, top-k) @ v6-M | 0.077 | True | 0.419 | -0.138 [-0.184, -0.096] | below |
| AV2 val (ego mask) | oracle frame router (true V, top-k) @ v6-M + components | 0.077 | True | 0.392 | -0.111 [-0.155, -0.070] | below |
| AV2 val (ego mask) | oracle frame router (true V, top-k) @ v6-L | 0.094 | True | 0.419 | -0.127 [-0.180, -0.080] | below |
| AV2 val (ego mask) | oracle frame router (true V, top-k) @ v6-L + components | 0.094 | True | 0.398 | -0.106 [-0.157, -0.060] | below |
| AV2 val (ego mask) | best Budget-Adaptive Routing (picked on val): skip_offbin @ v6-S | 0.053 | True | 0.007 | +0.286 [+0.228, +0.346] | outperforms |
| AV2 val (ego mask) | best Budget-Adaptive Routing (picked on val): skip_offbin @ v6-S + components | 0.053 | True | 0.021 | +0.272 [+0.208, +0.334] | outperforms |
| AV2 val (ego mask) | best decision-value router (picked on val): cond_V @ v6-S | 0.053 | True | 0.013 | +0.281 [+0.220, +0.343] | outperforms |
| AV2 val (ego mask) | best decision-value router (picked on val): cond_V @ v6-S + components | 0.053 | True | 0.027 | +0.266 [+0.206, +0.326] | outperforms |
| AV2 val (ego mask) | best Budget-Adaptive Routing (picked on val): cond_moric @ v6-M | 0.077 | True | 0.047 | +0.234 [+0.175, +0.294] | outperforms |
| AV2 val (ego mask) | best Budget-Adaptive Routing (picked on val): cond_moric @ v6-M + components | 0.077 | True | 0.065 | +0.216 [+0.161, +0.269] | outperforms |
| AV2 val (ego mask) | best decision-value router (picked on val): cond_V @ v6-M | 0.078 | True | 0.029 | +0.252 [+0.196, +0.307] | outperforms |
| AV2 val (ego mask) | best decision-value router (picked on val): cond_V @ v6-M + components | 0.078 | True | 0.054 | +0.226 [+0.173, +0.280] | outperforms |
| AV2 val (ego mask) | best Budget-Adaptive Routing (picked on val): cond_moric @ v6-L | 0.093 | True | 0.062 | +0.230 [+0.165, +0.294] | outperforms |
| AV2 val (ego mask) | best Budget-Adaptive Routing (picked on val): cond_moric @ v6-L + components | 0.093 | True | 0.082 | +0.209 [+0.142, +0.271] | outperforms |
| AV2 val (ego mask) | best decision-value router (picked on val): cond_V @ v6-L | 0.093 | True | 0.048 | +0.243 [+0.180, +0.306] | outperforms |
| AV2 val (ego mask) | best decision-value router (picked on val): cond_V @ v6-L + components | 0.093 | True | 0.079 | +0.212 [+0.151, +0.272] | outperforms |
| Argoverse-HD val | oracle frame router (true V, top-k) @ v6-S | 0.070 | True | 0.369 | +0.049 [-0.002, +0.101] | matches |
| Argoverse-HD val | oracle frame router (true V, top-k) @ v6-S + components | 0.070 | True | 0.356 | +0.062 [+0.014, +0.112] | outperforms |
| Argoverse-HD val | oracle frame router (true V, top-k) @ v6-M | 0.103 | True | 0.369 | +0.059 [+0.010, +0.104] | outperforms |
| Argoverse-HD val | oracle frame router (true V, top-k) @ v6-M + components | 0.103 | True | 0.365 | +0.063 [+0.017, +0.108] | outperforms |
| Argoverse-HD val | oracle frame router (true V, top-k) @ v6-L | 0.124 | True | 0.369 | +0.061 [+0.018, +0.105] | outperforms |
| Argoverse-HD val | oracle frame router (true V, top-k) @ v6-L + components | 0.124 | True | 0.367 | +0.064 [+0.022, +0.109] | outperforms |

## 6. Ablation of v6-S (§2n.1)

| split | cell | G | px | v6-S − cell [95% CI] | verdict | unnecessary / 1k | missed / 1k |
|---|---|---|---|---|---|---|---|
| nuScenes new test | v6-S (refiner v3a, C1) | 0.211 | 0.078 | – | – | 15.2 | 38.0 |
| nuScenes new test | v6-S without refiner (C1 only) | 0.181 | 0.078 | +0.030 [-0.000, +0.068] | matches | 21.6 | 42.5 |
| nuScenes new test | v6-S without C1 (refiner only; = v5) | 0.176 | 0.077 | +0.035 [+0.014, +0.066] | outperforms | 32.6 | 34.4 |
| nuScenes new test | v6-S without refiner and without C1 (= v5) | 0.133 | 0.077 | +0.078 [+0.037, +0.133] | outperforms | 42.7 | 38.6 |
| nuScenes new test | T4 in place of C1 (= v5.1-S) | 0.199 | 0.077 | +0.012 [+0.004, +0.026] | outperforms | 21.7 | 37.3 |
| nuScenes new test | T4 in place of C1, no refiner | 0.162 | 0.077 | +0.049 [+0.016, +0.092] | outperforms | 30.9 | 41.6 |
| nuScenes new test | support only + C1 + refiner | -0.095 | 0.050 | +0.306 [+0.060, +0.532] | outperforms | 11.0 | 46.3 |
| nuScenes new test | field only + C1 + refiner | 0.174 | 0.068 | +0.037 [-0.079, +0.159] | below | 50.5 | 33.4 |
| nuScenes new test | support only + C1, no refiner | -0.121 | 0.050 | +0.332 [+0.087, +0.548] | outperforms | 13.8 | 51.5 |
| nuScenes new test | field only + C1, no refiner | 0.164 | 0.068 | +0.046 [-0.074, +0.176] | below | 54.1 | 37.0 |
| AV2 val (ego mask) | v6-S (refiner v3a, C1) | 0.294 | 0.054 | – | – | 52.5 | 38.8 |
| AV2 val (ego mask) | v6-S without refiner (C1 only) | 0.241 | 0.054 | +0.053 [+0.011, +0.095] | outperforms | 138.0 | 29.1 |
| AV2 val (ego mask) | v6-S without C1 (refiner only; = v5) | 0.208 | 0.053 | +0.086 [+0.039, +0.130] | outperforms | 112.4 | 30.2 |
| AV2 val (ego mask) | v6-S without refiner and without C1 (= v5) | 0.031 | 0.053 | +0.262 [+0.187, +0.335] | outperforms | 206.5 | 23.5 |
| AV2 val (ego mask) | T4 in place of C1 (= v5.1-S) | 0.262 | 0.053 | +0.032 [+0.013, +0.050] | outperforms | 76.5 | 34.5 |
| AV2 val (ego mask) | T4 in place of C1, no refiner | 0.147 | 0.053 | +0.146 [+0.089, +0.204] | outperforms | 167.0 | 27.2 |
| AV2 val (ego mask) | support only + C1 + refiner | 0.218 | 0.041 | +0.075 [+0.031, +0.130] | outperforms | 46.5 | 50.4 |
| AV2 val (ego mask) | field only + C1 + refiner | 0.021 | 0.037 | +0.273 [+0.217, +0.331] | outperforms | 180.9 | 30.0 |
| AV2 val (ego mask) | support only + C1, no refiner | 0.190 | 0.041 | +0.104 [+0.050, +0.158] | outperforms | 104.3 | 42.9 |
| AV2 val (ego mask) | field only + C1, no refiner | 0.020 | 0.037 | +0.273 [+0.210, +0.344] | outperforms | 222.7 | 24.2 |
| Argoverse-HD val | v6-S (refiner v3a, C1) | 0.418 | 0.070 | – | – | 130.8 | 51.8 |
| Argoverse-HD val | v6-S without refiner (C1 only) | 0.422 | 0.070 | -0.004 [-0.083, +0.101] | below | 202.9 | 23.7 |
| Argoverse-HD val | v6-S without C1 (refiner only; = v5) | 0.185 | 0.070 | +0.233 [+0.173, +0.310] | outperforms | 240.8 | 34.7 |
| Argoverse-HD val | v6-S without refiner and without C1 (= v5) | 0.039 | 0.070 | +0.379 [+0.260, +0.504] | outperforms | 325.8 | 15.3 |
| Argoverse-HD val | T4 in place of C1 (= v5.1-S) | 0.338 | 0.070 | +0.079 [+0.054, +0.107] | outperforms | 169.1 | 45.6 |
| Argoverse-HD val | T4 in place of C1, no refiner | 0.281 | 0.070 | +0.136 [+0.042, +0.243] | outperforms | 243.3 | 21.8 |
| Argoverse-HD val | support only + C1 + refiner | 0.313 | 0.056 | +0.105 [+0.020, +0.228] | outperforms | 125.3 | 92.0 |
| Argoverse-HD val | field only + C1 + refiner | 0.080 | 0.043 | +0.338 [+0.255, +0.434] | outperforms | 298.5 | 29.9 |
| Argoverse-HD val | support only + C1, no refiner | 0.314 | 0.056 | +0.104 [-0.023, +0.233] | below | 183.2 | 69.7 |
| Argoverse-HD val | field only + C1, no refiner | 0.090 | 0.043 | +0.328 [+0.235, +0.433] | outperforms | 318.3 | 17.5 |

## 7. Test-pass record

**Code (md5, as frozen in §2n / §2n.1):**
- k28_test.py acb53791;
- k28_test_chain.sh 32daaa9b;
- k28_test_abl.py 692a3716;
- k28_test_abl_chain.sh 4a70322b.

**Launches:**
- Main pass: 2026-10-13 00:36 +07 (VUW 06:36).
- Ablation and router pass: 02:27 +07.
- Every unit finished at VUW 16:32:52.

**Operational events.** These were not code changes, and the result files are deterministic per unit:
- VUW 07:42: the host ran out of memory. Shard processes died:
  - all three ours-nuScenes shards;
  - baseheur-yolov8m nuScenes 0/3;
  - baseheur nuScenes 2/6. This one died silently, with no "Killed" line.
- They were re-run through the wrapper with the same commands. Finished units are skipped. Fills: test/ours_nusc_g0_fill.log, fill2_nusc.log, fill3_heur_nusc.log.

**Analysis fixes (analysis code only, made after the pass and before any table existed):**
- k28_analyze_test.py and k28_analyze_test2.py crashed with `KeyError: 'J_cheap_ours'`. The column list held J_cheap twice, and the @k28 rename renamed both copies.
- Fixed by excluding J_cheap from the prefix list:
  - k28_analyze_test.py: d0dd013a → ca1c34b4;
  - k28_analyze_test2.py: 26468b2b → 725e481e.
- Both were re-run once on the unchanged result files. The required parity checks are 100% identical (§4).
- The same latent bug in k28_p3_analyze.py ctable() was fixed before phase 3 reached it (a208656c → bd7035fd).

**Figure:** pareto_v6.pdf / .png shows G with refiner against GFLOPs per frame relative to v5.1-S, with the S / M / L caps drawn as dashed lines.

## 8. Full lists

### 8.1 In-class rows v6 does not outperform

| split | class | v6 | row | group | GFLOPs | G row | v6 − row [95% CI] | verdict | point estimate |
|---|---|---|---|---|---|---|---|---|---|
| Argoverse-HD val | S | v6-S | oracle frame router (true V, top-k) @ v5.1-S px | router | 1.00 | 0.369 | +0.049 [-0.002, +0.101] | matches | v6 higher |
| AV2 val (ego mask) | S | v6-S | oracle frame router (true V, top-k) @ v5.1-S px | router | 1.00 | 0.418 | -0.124 [-0.173, -0.079] | below | baseline higher |
| AV2 val (ego mask) | S | v6-S | oracle frame router (true V, top-k) @ v5.1-S px + components | router + components | 1.00 | 0.395 | -0.102 [-0.146, -0.062] | below | baseline higher |
| nuScenes new test | S | v6-S | oracle frame router (true V, top-k) @ v5.1-S px | router | 1.00 | 0.365 | -0.154 [-0.231, -0.073] | below | baseline higher |
| nuScenes new test | S | v6-S | oracle frame router (true V, top-k) @ v5.1-S px + components | router + components | 1.00 | 0.349 | -0.138 [-0.211, -0.057] | below | baseline higher |
| Argoverse-HD val | M | v6-M | K28 heuristic tsp2@S+C1 | K28 heuristic+C1 | 1.41 | 0.405 | +0.023 [-0.004, +0.049] | matches | v6 higher |
| AV2 val (ego mask) | M | v6-M | oracle frame router (true V, top-k) @ v5.1-L px | router | 1.32 | 0.419 | -0.138 [-0.184, -0.096] | below | baseline higher |
| AV2 val (ego mask) | M | v6-M | oracle frame router (true V, top-k) @ v5.1-L px + components | router + components | 1.32 | 0.393 | -0.112 [-0.157, -0.071] | below | baseline higher |
| AV2 val (ego mask) | M | v6-M | K28 heuristic tsp8@S+C1 | K28 heuristic+C1 | 1.33 | 0.296 | -0.015 [-0.037, +0.003] | below | baseline higher |
| AV2 val (ego mask) | M | v6-M | K28 heuristic tsp2@S+C1 | K28 heuristic+C1 | 1.23 | 0.286 | -0.005 [-0.034, +0.022] | below | baseline higher |
| AV2 val (ego mask) | M | v6-M | K28 whole frame yolov8s 512 +C1 | K28 whole frame + C1 | 1.26 | 0.285 | -0.004 [-0.069, +0.059] | below | baseline higher |
| AV2 val (ego mask) | M | v6-M | K28 heuristic cor0@M+C1 | K28 heuristic+C1 | 1.42 | 0.271 | +0.009 [-0.025, +0.047] | below | v6 higher |
| AV2 val (ego mask) | M | v6-M | traj_support @ 1x v5.1-L px + components | heuristic + components | 1.51 | 0.270 | +0.011 [-0.017, +0.039] | matches | v6 higher |
| AV2 val (ego mask) | M | v6-M | traj_support @ 1x v5.1-S px + components | heuristic + components | 1.33 | 0.266 | +0.015 [-0.013, +0.044] | matches | v6 higher |
| AV2 val (ego mask) | M | v6-M | K28 heuristic tsp8@S+T4 | K28 heuristic+T4 | 1.33 | 0.266 | +0.015 [-0.013, +0.044] | matches | v6 higher |
| AV2 val (ego mask) | M | v6-M | K28 heuristic cor8@S+C1 | K28 heuristic+C1 | 1.24 | 0.260 | +0.021 [-0.010, +0.057] | matches | v6 higher |
| AV2 val (ego mask) | M | v6-M | K28 heuristic tsp2@S+T4 | K28 heuristic+T4 | 1.23 | 0.259 | +0.022 [-0.011, +0.053] | matches | v6 higher |
| AV2 val (ego mask) | M | v6-M | traj_support @ 0.5x v5.1-L px + components | heuristic + components | 1.34 | 0.249 | +0.032 [-0.002, +0.070] | matches | v6 higher |
| AV2 val (ego mask) | M | v6-M | corridor @ 2x v5.1-L px + components | heuristic + components | 1.35 | 0.247 | +0.034 [-0.000, +0.073] | matches | v6 higher |
| nuScenes new test | M | v6-M | oracle frame router (true V, top-k) @ v5.1-L px | router | 1.25 | 0.365 | -0.067 [-0.163, +0.010] | below | baseline higher |
| nuScenes new test | M | v6-M | oracle frame router (true V, top-k) @ v5.1-L px + components | router + components | 1.25 | 0.348 | -0.050 [-0.140, +0.021] | below | baseline higher |
| nuScenes new test | M | v6-M | K28 heuristic tsp8@S+C1 | K28 heuristic+C1 | 1.40 | 0.174 | +0.124 [-0.023, +0.302] | below | v6 higher |
| nuScenes new test | M | v6-M | K28 whole frame yolov8s 512 +C1 | K28 whole frame + C1 | 1.32 | 0.124 | +0.174 [-0.084, +0.485] | below | v6 higher |
| nuScenes new test | M | v6-M | K28 whole frame yolov8s 512 +T4 | K28 whole frame + T4 | 1.32 | 0.093 | +0.205 [-0.031, +0.506] | below | v6 higher |
| nuScenes new test | M | v6-M | K28 whole frame yolov8s 448 +C1 | K28 whole frame + C1 | 1.03 | 0.086 | +0.212 [-0.059, +0.531] | below | v6 higher |
| nuScenes new test | M | v6-M | K28 whole frame yolov8s 448 +T4 | K28 whole frame + T4 | 1.03 | 0.073 | +0.225 [-0.038, +0.539] | below | v6 higher |
| nuScenes new test | M | v6-M | traj_support @ 1x v5.1-S px + components | heuristic + components | 1.40 | 0.165 | +0.133 [-0.013, +0.309] | matches | v6 higher |
| nuScenes new test | M | v6-M | K28 heuristic tsp8@S+T4 | K28 heuristic+T4 | 1.40 | 0.165 | +0.133 [-0.013, +0.309] | matches | v6 higher |
| nuScenes new test | M | v6-M | K28 heuristic tsp2@S+C1 | K28 heuristic+C1 | 1.28 | 0.114 | +0.184 [-0.015, +0.437] | matches | v6 higher |
| nuScenes new test | M | v6-M | K28 heuristic tsp2@S+T4 | K28 heuristic+T4 | 1.28 | 0.102 | +0.196 [-0.002, +0.446] | matches | v6 higher |
| Argoverse-HD val | L | v6-L | K28 heuristic cor0@M+C1 | K28 heuristic+C1 | 2.11 | 0.491 | -0.060 [-0.126, -0.001] | below | baseline higher |
| Argoverse-HD val | L | v6-L | K28 heuristic tsp4@M+C1 | K28 heuristic+C1 | 2.47 | 0.420 | +0.010 [-0.020, +0.042] | matches | v6 higher |
| Argoverse-HD val | L | v6-L | K28 heuristic cor8@M+C1 | K28 heuristic+C1 | 2.27 | 0.382 | +0.048 [-0.013, +0.130] | matches | v6 higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic tsp8@L+C1 | K28 heuristic+C1 | 2.18 | 0.341 | -0.050 [-0.112, +0.004] | below | baseline higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic tsp8@M+C1 | K28 heuristic+C1 | 2.03 | 0.328 | -0.036 [-0.080, -0.001] | below | baseline higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic cor8@L+C1 | K28 heuristic+C1 | 1.92 | 0.324 | -0.032 [-0.097, +0.027] | below | baseline higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic tsp0@L+C1 | K28 heuristic+C1 | 2.39 | 0.318 | -0.026 [-0.076, +0.014] | below | baseline higher |
| AV2 val (ego mask) | L | v6-L | K28 whole frame yolov8s 640 +C1 | K28 whole frame + C1 | 1.93 | 0.311 | -0.019 [-0.081, +0.034] | below | baseline higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic cor8@M+C1 | K28 heuristic+C1 | 1.78 | 0.309 | -0.017 [-0.065, +0.026] | below | baseline higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic tsp4@M+C1 | K28 heuristic+C1 | 1.68 | 0.304 | -0.012 [-0.040, +0.011] | below | baseline higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic cor0@L+C1 | K28 heuristic+C1 | 2.13 | 0.297 | -0.006 [-0.056, +0.040] | below | baseline higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic tcr0@M+C1 | K28 heuristic+C1 | 2.39 | 0.285 | +0.007 [-0.023, +0.033] | below | v6 higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic tcr3@M+C1 | K28 heuristic+C1 | 2.38 | 0.278 | +0.014 [-0.027, +0.051] | below | v6 higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic tsp8@M+T4 | K28 heuristic+T4 | 2.03 | 0.266 | +0.026 [-0.023, +0.070] | below | v6 higher |
| AV2 val (ego mask) | L | v6-L | traj_support @ 2x v5.1-L px + components | heuristic + components | 1.60 | 0.267 | +0.024 [-0.009, +0.055] | matches | v6 higher |
| AV2 val (ego mask) | L | v6-L | traj_support @ 2x v5.1-S px + components | heuristic + components | 1.58 | 0.264 | +0.028 [-0.002, +0.056] | matches | v6 higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic tcr0@M+T4 | K28 heuristic+T4 | 2.39 | 0.259 | +0.033 [-0.008, +0.071] | matches | v6 higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic tsp4@M+T4 | K28 heuristic+T4 | 1.68 | 0.256 | +0.035 [-0.004, +0.076] | matches | v6 higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic tsp8@L+T4 | K28 heuristic+T4 | 2.18 | 0.255 | +0.037 [-0.013, +0.081] | matches | v6 higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic tsp0@L+T4 | K28 heuristic+T4 | 2.39 | 0.254 | +0.038 [-0.015, +0.086] | matches | v6 higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic cor8@M+T4 | K28 heuristic+T4 | 1.78 | 0.253 | +0.038 [-0.013, +0.087] | matches | v6 higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic tcr3@S+C1 | K28 heuristic+C1 | 1.55 | 0.242 | +0.050 [-0.003, +0.095] | matches | v6 higher |
| AV2 val (ego mask) | L | v6-L | K28 heuristic cor8@L+T4 | K28 heuristic+T4 | 1.92 | 0.241 | +0.051 [-0.001, +0.100] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 whole frame yolo11m 448 +C1 | K28 whole frame + C1 | 2.45 | 0.318 | -0.003 [-0.097, +0.090] | below | baseline higher |
| nuScenes new test | L | v6-L | K28 whole frame yolo11m 448 +T4 | K28 whole frame + T4 | 2.45 | 0.306 | +0.008 [-0.094, +0.100] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tsp2@S-yolo11m+T4 | K28 heuristic+T4 | 2.29 | 0.303 | +0.011 [-0.047, +0.068] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tsp2@S-yolov8m+C1 | K28 heuristic+C1 | 2.54 | 0.303 | +0.012 [-0.048, +0.072] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tsp2@S-yolo11m+C1 | K28 heuristic+C1 | 2.29 | 0.302 | +0.012 [-0.045, +0.070] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tsp2@S-yolov8m+T4 | K28 heuristic+T4 | 2.54 | 0.300 | +0.015 [-0.044, +0.072] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tsp2@S-yolo11m | K28 heuristic plain | 2.29 | 0.292 | +0.022 [-0.038, +0.080] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tsp2@S-yolov8m | K28 heuristic plain | 2.54 | 0.282 | +0.033 [-0.028, +0.091] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 whole frame yolo11m 448 | K28 whole frame | 2.45 | 0.262 | +0.052 [-0.035, +0.135] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr3@M+T4 | K28 heuristic+T4 | 1.80 | 0.211 | +0.103 [-0.021, +0.240] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tsp4@M+C1 | K28 heuristic+C1 | 1.77 | 0.162 | +0.152 [-0.038, +0.398] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tsp4@M+T4 | K28 heuristic+T4 | 1.77 | 0.152 | +0.163 [-0.028, +0.403] | below | v6 higher |
| nuScenes new test | L | v6-L | traj_support @ 1x v5.1-L px + components | heuristic + components | 1.64 | 0.146 | +0.168 [-0.030, +0.420] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tsp8@M+C1 | K28 heuristic+C1 | 2.19 | 0.131 | +0.184 [-0.068, +0.507] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr0@M+C1 | K28 heuristic+C1 | 2.01 | 0.111 | +0.203 [-0.022, +0.500] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tsp8@M+T4 | K28 heuristic+T4 | 2.19 | 0.107 | +0.208 [-0.046, +0.530] | below | v6 higher |
| nuScenes new test | L | v6-L | traj_support @ 2x v5.1-L px + components | heuristic + components | 1.87 | 0.093 | +0.221 [-0.022, +0.540] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tsp8@M | K28 heuristic plain | 2.19 | 0.091 | +0.224 [-0.035, +0.548] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic cor8@M+C1 | K28 heuristic+C1 | 2.12 | 0.022 | +0.292 [-0.071, +0.774] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic cor0@M+C1 | K28 heuristic+C1 | 1.70 | 0.013 | +0.301 [-0.022, +0.757] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic unc6@L+T4 | K28 heuristic+T4 | 2.50 | 0.006 | +0.308 [-0.030, +0.664] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic unc6@L+C1 | K28 heuristic+C1 | 2.50 | 0.003 | +0.312 [-0.021, +0.664] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic cor8@M+T4 | K28 heuristic+T4 | 2.12 | -0.017 | +0.332 [-0.025, +0.804] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 whole frame yolov8s 640 +C1 | K28 whole frame + C1 | 2.21 | -0.036 | +0.350 [-0.090, +0.942] | below | v6 higher |
| nuScenes new test | L | v6-L | whole frame u640 + components | whole frame + components | 2.21 | -0.069 | +0.384 [-0.043, +0.977] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 whole frame yolov8s 640 +T4 | K28 whole frame + T4 | 2.21 | -0.069 | +0.384 [-0.043, +0.977] | below | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr3@S-yolo11m+T4 | K28 heuristic+T4 | 2.15 | 0.278 | +0.036 [-0.014, +0.086] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr3@S-yolo11m+C1 | K28 heuristic+C1 | 2.15 | 0.274 | +0.040 [-0.010, +0.090] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr3@S-yolo11m | K28 heuristic plain | 2.15 | 0.272 | +0.042 [-0.012, +0.100] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr0@S-yolo11m+T4 | K28 heuristic+T4 | 2.46 | 0.269 | +0.046 [-0.003, +0.093] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr0@S-yolo11m | K28 heuristic plain | 2.46 | 0.268 | +0.047 [-0.007, +0.100] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr3@S-yolov8m+C1 | K28 heuristic+C1 | 2.48 | 0.264 | +0.051 [-0.020, +0.119] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr3@S-yolov8m+T4 | K28 heuristic+T4 | 2.48 | 0.259 | +0.056 [-0.014, +0.124] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr3@S-yolov8m | K28 heuristic plain | 2.48 | 0.249 | +0.065 [-0.007, +0.134] | matches | v6 higher |
| nuScenes new test | L | v6-L | temporal_crit @ 2x v5.1-L px + components | heuristic + components | 1.99 | 0.215 | +0.100 [-0.015, +0.234] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr3@M+C1 | K28 heuristic+C1 | 1.80 | 0.215 | +0.100 [-0.019, +0.237] | matches | v6 higher |
| nuScenes new test | L | v6-L | temporal_crit @ 2x v5.1-S px + components | heuristic + components | 1.64 | 0.211 | +0.103 [-0.011, +0.236] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr3@M | K28 heuristic plain | 1.80 | 0.194 | +0.121 [-0.012, +0.257] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tsp4@M | K28 heuristic plain | 1.77 | 0.137 | +0.178 [-0.014, +0.415] | matches | v6 higher |
| nuScenes new test | L | v6-L | traj_support @ 1x v5.1-L px | heuristic | 1.64 | 0.127 | +0.188 [-0.011, +0.434] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr0@M+T4 | K28 heuristic+T4 | 2.01 | 0.103 | +0.211 [-0.013, +0.506] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic tcr0@M | K28 heuristic plain | 2.01 | 0.093 | +0.222 [-0.012, +0.513] | matches | v6 higher |
| nuScenes new test | L | v6-L | traj_support @ 2x v5.1-S px + components | heuristic + components | 1.81 | 0.091 | +0.224 [-0.019, +0.543] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic cor0@M+T4 | K28 heuristic+T4 | 1.70 | 0.000 | +0.314 [-0.010, +0.762] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic unc6@L | K28 heuristic plain | 2.50 | -0.014 | +0.329 [-0.004, +0.681] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic cor8@M | K28 heuristic plain | 2.12 | -0.035 | +0.349 [-0.007, +0.819] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic unc2@L+T4 | K28 heuristic+T4 | 2.39 | -0.037 | +0.351 [-0.003, +0.737] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic unc2@L | K28 heuristic plain | 2.39 | -0.041 | +0.355 [-0.003, +0.744] | matches | v6 higher |
| nuScenes new test | L | v6-L | K28 heuristic unc2@M | K28 heuristic plain | 1.84 | -0.041 | +0.356 [-0.005, +0.743] | matches | v6 higher |

### 8.2 Over-budget rows v6-L does not outperform

| split | class | v6 | row | group | GFLOPs | G row | v6 − row [95% CI] | verdict | point estimate |
|---|---|---|---|---|---|---|---|---|---|
| Argoverse-HD val | over | v6-L | K28 whole frame yolov8s 1920 +C1 | K28 whole frame + C1 | 14.49 | 0.430 | +0.001 [-0.036, +0.042] | below | v6 higher |
| Argoverse-HD val | over | v6-L | K28 heuristic cor0@L+C1 | K28 heuristic+C1 | 3.05 | 0.407 | +0.024 [-0.111, +0.233] | below | v6 higher |
| Argoverse-HD val | over | v6-L | K28 heuristic tsp0@L+C1 | K28 heuristic+C1 | 3.39 | 0.364 | +0.067 [-0.047, +0.264] | below | v6 higher |
| Argoverse-HD val | over | v6-L | K28 heuristic tsp8@S-yolov8m+C1 | K28 heuristic+C1 | 3.57 | 0.423 | +0.008 [-0.009, +0.031] | matches | v6 higher |
| Argoverse-HD val | over | v6-L | K28 heuristic tsp8@S-yolo11m+C1 | K28 heuristic+C1 | 3.09 | 0.422 | +0.008 [-0.004, +0.025] | matches | v6 higher |
| Argoverse-HD val | over | v6-L | K28 heuristic tsp2@S-yolov8m+C1 | K28 heuristic+C1 | 3.06 | 0.415 | +0.015 [-0.008, +0.045] | matches | v6 higher |
| Argoverse-HD val | over | v6-L | K28 heuristic tsp2@S-yolo11m+C1 | K28 heuristic+C1 | 2.70 | 0.415 | +0.015 [-0.001, +0.035] | matches | v6 higher |
| Argoverse-HD val | over | v6-L | K28 whole frame yolo11m 1920 +C1 | K28 whole frame + C1 | 34.52 | 0.404 | +0.026 [-0.014, +0.075] | matches | v6 higher |
| Argoverse-HD val | over | v6-L | K28 heuristic cor8@S-yolov8m+C1 | K28 heuristic+C1 | 3.62 | 0.404 | +0.026 [-0.006, +0.070] | matches | v6 higher |
| Argoverse-HD val | over | v6-L | K28 heuristic cor8@S-yolo11m+C1 | K28 heuristic+C1 | 3.15 | 0.399 | +0.032 [-0.000, +0.074] | matches | v6 higher |
| Argoverse-HD val | over | v6-L | K28 whole frame yolov8m 1920 +C1 | K28 whole frame + C1 | 39.89 | 0.389 | +0.042 [-0.014, +0.124] | matches | v6 higher |
| Argoverse-HD val | over | v6-L | K28 heuristic tsp8@M+C1 | K28 heuristic+C1 | 2.58 | 0.381 | +0.050 [-0.001, +0.125] | matches | v6 higher |
| Argoverse-HD val | over | v6-L | K28 whole frame yolov8m 1536 +C1 | K28 whole frame + C1 | 25.19 | 0.348 | +0.082 [-0.004, +0.251] | matches | v6 higher |
| Argoverse-HD val | over | v6-L | K28 heuristic cor8@L+C1 | K28 heuristic+C1 | 2.84 | 0.337 | +0.094 [-0.012, +0.276] | matches | v6 higher |
| Argoverse-HD val | over | v6-L | K28 heuristic tsp8@L+C1 | K28 heuristic+C1 | 3.18 | 0.335 | +0.095 [-0.000, +0.271] | matches | v6 higher |
| Argoverse-HD val | over | v6-L | K28 whole frame yolov8m 1280 +C1 | K28 whole frame + C1 | 17.49 | 0.333 | +0.097 [-0.004, +0.298] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8s 1280 +C1 | K28 whole frame + C1 | 7.49 | 0.348 | -0.057 [-0.119, -0.003] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolo11m 1280 +C1 | K28 whole frame + C1 | 17.86 | 0.342 | -0.050 [-0.101, -0.008] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8s 960 +C1 | K28 whole frame + C1 | 4.35 | 0.338 | -0.047 [-0.108, +0.005] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolo11m 896 +C1 | K28 whole frame + C1 | 8.87 | 0.337 | -0.045 [-0.103, +0.003] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolo11m 512 +C1 | K28 whole frame + C1 | 3.00 | 0.333 | -0.042 [-0.100, +0.007] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolo11m 640 +C1 | K28 whole frame + C1 | 4.61 | 0.331 | -0.039 [-0.099, +0.011] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8m 1280 +C1 | K28 whole frame + C1 | 20.63 | 0.330 | -0.038 [-0.101, +0.016] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8s 1536 +C1 | K28 whole frame + C1 | 10.73 | 0.328 | -0.036 [-0.104, +0.020] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 heuristic tcr3@L+C1 | K28 heuristic+C1 | 3.58 | 0.324 | -0.033 [-0.079, +0.009] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolo11m 1536 +C1 | K28 whole frame + C1 | 25.57 | 0.322 | -0.031 [-0.074, +0.006] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8m 640 +C1 | K28 whole frame + C1 | 5.32 | 0.322 | -0.031 [-0.090, +0.023] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8m 896 +C1 | K28 whole frame + C1 | 10.25 | 0.320 | -0.028 [-0.090, +0.026] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8m 512 +C1 | K28 whole frame + C1 | 3.46 | 0.319 | -0.027 [-0.092, +0.031] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8s 2048 +C1 | K28 whole frame + C1 | 18.95 | 0.294 | -0.002 [-0.081, +0.077] | below | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 heuristic tsp8@S-yolov8m+C1 | K28 heuristic+C1 | 2.92 | 0.286 | +0.005 [-0.022, +0.038] | below | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 heuristic unc6@L+C1 | K28 heuristic+C1 | 3.10 | 0.283 | +0.008 [-0.048, +0.062] | below | v6 higher |
| AV2 val (ego mask) | over | v6-L | whole frame u1280 + components | whole frame + components | 7.49 | 0.272 | +0.020 [-0.035, +0.071] | below | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8s 1280 +T4 | K28 whole frame + T4 | 7.49 | 0.272 | +0.020 [-0.035, +0.071] | below | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 heuristic unc2@L+C1 | K28 heuristic+C1 | 2.73 | 0.260 | +0.031 [-0.020, +0.088] | below | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8m 1536 +C1 | K28 whole frame + C1 | 29.55 | 0.253 | +0.038 [-0.050, +0.149] | below | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolo11m 2048 +C1 | K28 whole frame + C1 | 45.16 | 0.242 | +0.049 [-0.052, +0.165] | below | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8m 2048 +C1 | K28 whole frame + C1 | 52.18 | 0.209 | +0.083 [-0.024, +0.216] | below | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 heuristic tsp8@S-yolo11m+C1 | K28 heuristic+C1 | 2.60 | 0.292 | -0.000 [-0.007, +0.006] | matches | baseline higher |
| AV2 val (ego mask) | over | v6-L | K28 heuristic tsp8@S-yolo11m+T4 | K28 heuristic+T4 | 2.60 | 0.277 | +0.014 [-0.007, +0.035] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 heuristic tsp2@S-yolov8m+C1 | K28 heuristic+C1 | 2.55 | 0.268 | +0.023 [-0.002, +0.052] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 heuristic cor8@S-yolov8m+C1 | K28 heuristic+C1 | 2.66 | 0.264 | +0.028 [-0.002, +0.058] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | temporal_crit @ 2x v5.1-L px + components | heuristic + components | 2.71 | 0.263 | +0.028 [-0.010, +0.065] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 heuristic tsp8@S-yolov8m+T4 | K28 heuristic+T4 | 2.92 | 0.261 | +0.030 [-0.007, +0.069] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | whole frame u960 + components | whole frame + components | 4.35 | 0.255 | +0.037 [-0.017, +0.084] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8s 960 +T4 | K28 whole frame + T4 | 4.35 | 0.255 | +0.037 [-0.017, +0.084] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 heuristic tcr3@L+T4 | K28 heuristic+T4 | 3.57 | 0.250 | +0.042 [-0.002, +0.083] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolo11m 1536 +T4 | K28 whole frame + T4 | 25.57 | 0.249 | +0.042 [-0.010, +0.093] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolo11m 896 +T4 | K28 whole frame + T4 | 8.87 | 0.245 | +0.046 [-0.004, +0.092] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8m 1280 +T4 | K28 whole frame + T4 | 20.63 | 0.239 | +0.053 [-0.007, +0.108] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | whole frame native + components | whole frame + components | 18.95 | 0.224 | +0.067 [-0.009, +0.146] | matches | v6 higher |
| AV2 val (ego mask) | over | v6-L | K28 whole frame yolov8s 2048 +T4 | K28 whole frame + T4 | 18.95 | 0.224 | +0.067 [-0.009, +0.146] | matches | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 1600 +C1 | K28 whole frame + C1 | 36.69 | 0.384 | -0.069 [-0.118, -0.025] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 1280 +C1 | K28 whole frame + C1 | 23.28 | 0.378 | -0.063 [-0.110, -0.017] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 960 +C1 | K28 whole frame + C1 | 11.83 | 0.369 | -0.055 [-0.113, -0.000] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 960 +C1 | K28 whole frame + C1 | 13.66 | 0.369 | -0.055 [-0.106, -0.004] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 768 +C1 | K28 whole frame + C1 | 7.36 | 0.369 | -0.054 [-0.102, -0.006] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 1280 +T4 | K28 whole frame + T4 | 23.28 | 0.363 | -0.048 [-0.095, -0.004] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 640 +C1 | K28 whole frame + C1 | 5.26 | 0.361 | -0.047 [-0.090, -0.001] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 1600 +T4 | K28 whole frame + T4 | 36.69 | 0.360 | -0.046 [-0.090, -0.004] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 768 +T4 | K28 whole frame + T4 | 7.36 | 0.353 | -0.039 [-0.096, +0.010] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 1280 +C1 | K28 whole frame + C1 | 20.15 | 0.351 | -0.037 [-0.087, +0.015] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 960 +T4 | K28 whole frame + T4 | 11.83 | 0.345 | -0.031 [-0.079, +0.019] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 1280 | K28 whole frame | 23.28 | 0.345 | -0.031 [-0.091, +0.029] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 960 +T4 | K28 whole frame + T4 | 13.66 | 0.343 | -0.028 [-0.077, +0.018] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 1600 | K28 whole frame | 36.69 | 0.342 | -0.027 [-0.088, +0.032] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 640 +T4 | K28 whole frame + T4 | 5.26 | 0.338 | -0.023 [-0.070, +0.025] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 768 +C1 | K28 whole frame + C1 | 8.50 | 0.337 | -0.023 [-0.080, +0.034] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 960 | K28 whole frame | 11.83 | 0.333 | -0.019 [-0.080, +0.047] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 1280 +T4 | K28 whole frame + T4 | 20.15 | 0.331 | -0.017 [-0.069, +0.037] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 1600 +C1 | K28 whole frame + C1 | 31.75 | 0.328 | -0.013 [-0.077, +0.044] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 960 | K28 whole frame | 13.66 | 0.327 | -0.013 [-0.077, +0.050] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 640 +C1 | K28 whole frame + C1 | 6.07 | 0.325 | -0.010 [-0.104, +0.088] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 heuristic tsp8@S-yolov8m+C1 | K28 heuristic+C1 | 2.95 | 0.322 | -0.007 [-0.052, +0.031] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 768 +T4 | K28 whole frame + T4 | 8.50 | 0.315 | -0.000 [-0.062, +0.052] | below | baseline higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 1280 | K28 whole frame | 20.15 | 0.313 | +0.002 [-0.059, +0.068] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tsp8@S-yolov8m+T4 | K28 heuristic+T4 | 2.95 | 0.311 | +0.004 [-0.041, +0.042] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 640 +T4 | K28 whole frame + T4 | 6.07 | 0.304 | +0.010 [-0.090, +0.105] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8s 1536 +C1 | K28 whole frame + C1 | 11.91 | 0.302 | +0.013 [-0.086, +0.111] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 768 | K28 whole frame | 7.36 | 0.302 | +0.013 [-0.036, +0.073] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tsp8@S-yolov8m | K28 heuristic plain | 2.95 | 0.301 | +0.013 [-0.033, +0.057] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 640 | K28 whole frame | 5.26 | 0.294 | +0.020 [-0.039, +0.087] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8s 1280 +C1 | K28 whole frame + C1 | 8.46 | 0.292 | +0.022 [-0.083, +0.125] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tsp8@S-yolo11m+C1 | K28 heuristic+C1 | 2.61 | 0.290 | +0.025 [-0.031, +0.078] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tsp8@S-yolo11m+T4 | K28 heuristic+T4 | 2.61 | 0.287 | +0.027 [-0.028, +0.081] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 1600 +T4 | K28 whole frame + T4 | 31.75 | 0.286 | +0.029 [-0.058, +0.111] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tsp8@S-yolo11m | K28 heuristic plain | 2.61 | 0.285 | +0.030 [-0.026, +0.083] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 768 | K28 whole frame | 8.50 | 0.282 | +0.032 [-0.037, +0.099] | below | v6 higher |
| nuScenes new test | over | v6-L | whole frame u1536 + components | whole frame + components | 11.91 | 0.280 | +0.035 [-0.058, +0.128] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8s 1536 +T4 | K28 whole frame + T4 | 11.91 | 0.280 | +0.035 [-0.058, +0.128] | below | v6 higher |
| nuScenes new test | over | v6-L | whole frame u1280 + components | whole frame + components | 8.46 | 0.269 | +0.046 [-0.052, +0.142] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8s 1280 +T4 | K28 whole frame + T4 | 8.46 | 0.269 | +0.046 [-0.052, +0.142] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 640 | K28 whole frame | 6.07 | 0.261 | +0.054 [-0.045, +0.143] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolo11m 1600 | K28 whole frame | 31.75 | 0.261 | +0.054 [-0.039, +0.145] | below | v6 higher |
| nuScenes new test | over | v6-L | whole frame u1536 | whole frame | 11.91 | 0.257 | +0.058 [-0.044, +0.162] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8s 1536 | K28 whole frame | 11.91 | 0.257 | +0.058 [-0.044, +0.162] | below | v6 higher |
| nuScenes new test | over | v6-L | whole frame u1280 | whole frame | 8.46 | 0.251 | +0.063 [-0.041, +0.162] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8s 1280 | K28 whole frame | 8.46 | 0.251 | +0.063 [-0.041, +0.162] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8s 1600 +C1 | K28 whole frame + C1 | 13.33 | 0.222 | +0.093 [-0.061, +0.244] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 448 +C1 | K28 whole frame + C1 | 2.83 | 0.204 | +0.111 [-0.072, +0.345] | below | v6 higher |
| nuScenes new test | over | v6-L | whole frame native + components | whole frame + components | 13.33 | 0.197 | +0.118 [-0.031, +0.263] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8s 1600 +T4 | K28 whole frame + T4 | 13.33 | 0.197 | +0.118 [-0.031, +0.263] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tcr3@L+C1 | K28 heuristic+C1 | 2.76 | 0.196 | +0.119 [-0.059, +0.335] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tcr3@L+T4 | K28 heuristic+T4 | 2.76 | 0.191 | +0.124 [-0.052, +0.337] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 448 +T4 | K28 whole frame + T4 | 2.83 | 0.179 | +0.136 [-0.042, +0.364] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tsp0@L+C1 | K28 heuristic+C1 | 3.09 | 0.177 | +0.138 [-0.087, +0.408] | below | v6 higher |
| nuScenes new test | over | v6-L | whole frame native | whole frame | 13.33 | 0.176 | +0.139 [-0.021, +0.287] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8s 1600 | K28 whole frame | 13.33 | 0.176 | +0.139 [-0.021, +0.287] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tcr3@L | K28 heuristic plain | 2.76 | 0.155 | +0.160 [-0.021, +0.366] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tsp0@L+T4 | K28 heuristic+T4 | 3.09 | 0.150 | +0.164 [-0.052, +0.434] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8m 448 | K28 whole frame | 2.83 | 0.148 | +0.167 [-0.021, +0.392] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tsp8@L+C1 | K28 heuristic+C1 | 2.98 | 0.141 | +0.174 [-0.062, +0.474] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tsp8@L+T4 | K28 heuristic+T4 | 2.98 | 0.126 | +0.189 [-0.053, +0.488] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tsp0@L | K28 heuristic plain | 3.09 | 0.124 | +0.190 [-0.024, +0.456] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic cor0@L+C1 | K28 heuristic+C1 | 3.01 | 0.118 | +0.196 [-0.099, +0.569] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8s 960 +C1 | K28 whole frame + C1 | 4.96 | 0.112 | +0.203 [-0.092, +0.585] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tsp8@L | K28 heuristic plain | 2.98 | 0.097 | +0.218 [-0.023, +0.511] | below | v6 higher |
| nuScenes new test | over | v6-L | whole frame u960 + components | whole frame + components | 4.96 | 0.096 | +0.218 [-0.069, +0.596] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8s 960 +T4 | K28 whole frame + T4 | 4.96 | 0.096 | +0.218 [-0.069, +0.596] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic cor0@L+T4 | K28 heuristic+T4 | 3.01 | 0.082 | +0.233 [-0.049, +0.596] | below | v6 higher |
| nuScenes new test | over | v6-L | whole frame u960 | whole frame | 4.96 | 0.055 | +0.260 [-0.027, +0.625] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 whole frame yolov8s 960 | K28 whole frame | 4.96 | 0.055 | +0.260 [-0.027, +0.625] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic cor8@L+C1 | K28 heuristic+C1 | 2.88 | 0.001 | +0.314 [-0.064, +0.824] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic cor8@L+T4 | K28 heuristic+T4 | 2.88 | -0.027 | +0.342 [-0.032, +0.839] | below | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic cor8@S-yolov8m+C1 | K28 heuristic+C1 | 2.98 | 0.282 | +0.033 [-0.014, +0.075] | matches | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic cor0@S-yolov8m+C1 | K28 heuristic+C1 | 2.58 | 0.276 | +0.038 [-0.020, +0.092] | matches | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic cor8@S-yolo11m+C1 | K28 heuristic+C1 | 2.65 | 0.274 | +0.040 [-0.005, +0.084] | matches | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic cor8@S-yolo11m+T4 | K28 heuristic+T4 | 2.65 | 0.271 | +0.043 [-0.002, +0.089] | matches | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tcr0@S-yolov8m+T4 | K28 heuristic+T4 | 2.86 | 0.271 | +0.043 [-0.008, +0.089] | matches | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic tcr0@S-yolov8m+C1 | K28 heuristic+C1 | 2.86 | 0.271 | +0.044 [-0.007, +0.089] | matches | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic cor8@S-yolov8m+T4 | K28 heuristic+T4 | 2.98 | 0.270 | +0.045 [-0.004, +0.089] | matches | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic cor0@S-yolov8m+T4 | K28 heuristic+T4 | 2.58 | 0.264 | +0.051 [-0.008, +0.103] | matches | v6 higher |
| nuScenes new test | over | v6-L | K28 heuristic cor0@L | K28 heuristic plain | 3.01 | 0.046 | +0.269 [-0.007, +0.623] | matches | v6 higher |

### 8.3 S2 table (v6-S against YOLOv8s whole frame)

See §3.

## Files
- table_main_{nusc,av2,ahd}_v6.csv: every row, with G, px, GFLOPs, the class and (P − row) for P in v6-S / M / L and v5.1-S.
- headline_v6.csv, class_matched_v6.csv, not_outperformed_v6.csv, parity_v6.csv.
- ablation_v6S_{ds}.csv, routers_v6_{ds}.csv, parity_abl_v6.csv.
- pareto_v6.pdf / .png.
