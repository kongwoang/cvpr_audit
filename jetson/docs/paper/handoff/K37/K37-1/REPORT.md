# K37-1 — matched-component safety–utility operating curves (thr_decel grid) and the pre-registered constrained comparison

Frozen v6-S. The only varied quantity is the consumer's thr_decel ∈ {0.6, 0.8, 1.0, 1.2, 1.5} m/s², with thr_hard 3.5
fixed. GT, required episodes and G's cost stay at the deployed definition (τ1 1.0, τ2 3.5).
- **Pre-registration.** PREREG.md (sha256 18ca8386…), frozen before any test or holdout run.
- **Labels.** Test results are post-hoc. The holdout result is the pre-registered single evaluation.
- **Harness.** VUW experiments/k37_thr_20261026/k30_thr.py (sha256 db43cabf…); holdout k30_thr_h.py (75528901…).
- **Analysis.** scripts/A_method/k37_k1_analyze.py (sha256 37295f4f…, unchanged between the dev choice and the final
  run).
- **Statistics.** Paired unit bootstrap: 2,000 draws, seed 0.

## Holdout uses (as required)
The K35-4 holdout (Argoverse 1.1 TRAIN / Argoverse-HD train.json, 65 units):
1. **K35-4**: PREREG 2026-10-25; run 2026-10-25 20:09 – 2026-10-26 03:40 NZ.
2. **K37-1(c)**: PREREG 2026-10-26; run 2026-10-26 12:19 – 15:13 NZ. This is the second use.

**After K37 the holdout counts as used and must not be used for any further selection.**

## Parity (parity.json)
- **thr 1.0:** every row equals the stored deployed rows on every split, 0 differences.
  - dev: K30 dev_devref;
  - test: K28 instr, and K30 match for the hybrid at ×1.00;
  - holdout: the K35-4 records.
- **Other thresholds, whole-frame rows on test:** they equal K34-1's CPU replay (continuous_demands) except one frame
  each in AV2 u960 @0.8, Argoverse-HD native @1.2 and native @1.5.
  - K34-1 reports cross-source / backend discrepancies of this kind (validation.csv).
  - This run is the GPU original; K34-1 is a CPU replay.

## (c) Constrained comparison (pre-registered)
- **Dev choice** (k1_choice_dev.csv; recorded in STATUS before any test / holdout number was read). Rule: the largest
  threshold whose dev missed rate ≤ cheap's dev missed rate at 1.0 (nuScenes 34/99, AV2 30/366).

  | row | nuScenes | AV2 (also Argoverse-HD test and holdout) |
  |---|---|---|
  | cheap | 1.0 | 1.0 |
  | u960 + C1 + v3a | 0.8 (21/99) | 0.6 (22/366) |
  | native + C1 + v3a | 0.8 (25/99) | 0.6 (20/366) |
  | hybrid tsp8 (DESP budget) | 0.8 (27/99) | 0.6 (30/366) |
  | **DESP** | 0.8 (26/99) | **infeasible**: lowest miss 31/366 at 0.6, so 0.6 is used, labelled |

- **Evaluated once** (k1_constrained.csv). Missed = missed / required episodes. Δ = row − DESP at DESP's constrained
  point [95% CI]; bold = excludes 0.

| split | row (thr) | missed | false / 1,000 | G | px | ΔG | Δfalse | Δmissed rate | Δpx |
|---|---|---|---|---|---|---|---|---|---|
| nuScenes test | cheap (1.0) | 51 / 147 | 22.87 | 0 | 0.043 | **−0.261** | **+15.90** | +0.027 [−0.053, +0.098] | **−0.037** |
| nuScenes test | u960 (0.8) | 36 / 147 | 10.02 | 0.172 | 0.384 | −0.089 [−0.301, +0.087] | **+3.05** | **−0.075** | **+0.304** |
| nuScenes test | native (0.8) | 37 / 147 | 8.28 | 0.276 | 1.031 | +0.015 [−0.115, +0.091] | +1.31 [−0.33, +3.15] | **−0.068** | **+0.951** |
| nuScenes test | hybrid (0.8) | 45 / 147 | 7.30 | 0.223 | 0.113 | −0.038 [−0.097, +0.009] | +0.33 [−0.54, +1.20] | −0.014 [−0.035, 0.000] | **+0.033** |
| nuScenes test | **DESP (0.8)** | 47 / 147 | 6.97 | 0.261 | 0.080 | – | – | – | – |
| AV2 test | cheap (1.0) | 21 / 190 | 34.78 | 0 | 0.026 | **−0.376** | **+24.20** | +0.011 [−0.034, +0.054] | **−0.031** |
| AV2 test | u960 (0.6) | 13 / 190 | 12.69 | 0.415 | 0.232 | +0.039 [−0.002, +0.088] | **+2.12** | **−0.032** | **+0.175** |
| AV2 test | native (0.6) | 14 / 190 | 14.55 | 0.365 | 1.012 | −0.011 [−0.095, +0.068] | **+3.98** | −0.026 [−0.063, +0.005] | **+0.955** |
| AV2 test | hybrid (0.6) | 18 / 190 | 10.49 | 0.378 | 0.078 | +0.002 [−0.009, +0.013] | −0.08 [−0.93, +0.76] | −0.005 [−0.017, 0.000] | **+0.021** |
| AV2 test | **DESP (0.6, infeasible on dev)** | 19 / 190 | 10.58 | 0.376 | 0.057 | – | – | – | – |
| Argoverse-HD test | cheap (1.0) | 12 / 71 | 25.12 | 0 | 0.031 | **−0.485** | **+18.06** | +0.042 [−0.071, +0.134] | **−0.042** |
| Argoverse-HD test | u960 (0.6) | 9 / 71 | 6.73 | 0.366 | 0.253 | **−0.119** | −0.33 [−1.29, +0.66] | 0.000 | **+0.180** |
| Argoverse-HD test | native (0.6) | 10 / 71 | 9.20 | 0.499 | 1.013 | +0.014 [−0.022, +0.047] | **+2.13** | +0.014 [−0.029, +0.065] | **+0.940** |
| Argoverse-HD test | hybrid (0.6) | 9 / 71 | 7.26 | 0.469 | 0.112 | **−0.016** | +0.20 [−1.30, +2.08] | 0.000 | **+0.039** |
| Argoverse-HD test | **DESP (0.6)** | 9 / 71 | 7.06 | 0.485 | 0.073 | – | – | – | – |
| **Holdout** | cheap (1.0) | 18 / 154 | 34.48 | 0 | 0.031 | **−0.474** | **+27.35** | −0.013 [−0.076, +0.060] | **−0.048** |
| **Holdout** | u960 (0.6) | 20 / 154 | 8.02 | 0.457 | 0.253 | **−0.016** | +0.89 [−0.03, +1.88] | 0.000 [−0.026, +0.027] | **+0.174** |
| **Holdout** | native (0.6) | 19 / 154 | 9.53 | 0.483 | 1.013 | +0.009 [−0.006, +0.023] | **+2.39** | −0.007 [−0.022, 0.000] | **+0.934** |
| **Holdout** | hybrid (0.6) | 24 / 154 | 8.58 | 0.474 | 0.134 | +0.000 [−0.008, +0.011] | **+1.45** | +0.026 [0.000, +0.056] | **+0.054** |
| **Holdout** | **DESP (0.6)** | 20 / 154 | 7.13 | 0.474 | 0.080 | – | – | – | – |

Below / at-or-above-hard missed rates and insufficient-braking loss: k1_constrained.csv.

### Reading of (c)
- **At the constrained point, DESP reaches cheap's miss level and brakes falsely far less.**
  - Missed rate vs cheap: n.s. on every split; on the holdout 20 vs 18 / 154.
  - False episodes 16–27 per 1,000 fewer than cheap, at 1.9–2.6× cheap's px.
- **Among the C1 + v3a rows**, DESP has:
  - the **fewest false episodes**, or within n.s. of the fewest: the hybrid on AV2 test is 0.08 lower and u960 on
    Argoverse-HD test 0.33 lower, both n.s.;
  - the **lowest px** after cheap: 3–5× less than u960, 13–18× less than native, 27–40% less than the hybrid;
  - **G within the interval of the best row**, except where it is significantly higher: vs u960 on Argoverse-HD test and
    on the holdout, vs the hybrid on Argoverse-HD test.
- **Whole frames still miss fewer episodes than DESP on nuScenes and AV2 test** at their (lower-or-equal) constrained
  thresholds: u960 −0.075 and native −0.068 on nuScenes; u960 −0.032 on AV2. They pay 2–4 more false episodes per 1,000
  and 3–18× px.
  - On Argoverse-HD test and the holdout the misses are equal within ±1 episode.
- **The hybrid at DESP's budget** has G equal to DESP within ±0.04, the intervals including 0, except Argoverse-HD test
  (−0.016, significant). It misses ≤ 2 episodes fewer on the test splits and 4 more on the holdout, with 37–68% more px.
- **DESP is "infeasible" under the dev rule on AV2**, missing 31 vs 30 dev episodes. It is evaluated at its lowest-miss
  point (0.6), as pre-registered.

## (b) Operating curves (all points, k1_points.csv; paired row − DESP at the same threshold, k1_diffs.csv)

Test splits (post-hoc), missed episodes and G per threshold.

| test split | thr | cheap | u960 | native | hybrid | DESP | G: cheap / u960 / native / hybrid / DESP |
|---|---|---|---|---|---|---|---|
| nuScenes (147) | 0.6 | 35 | 29 | 29 | 36 | 36 | 0.069 / 0.199 / 0.315 / 0.263 / 0.301 |
| | 0.8 | 41 | 36 | 37 | 45 | 47 | 0.038 / 0.172 / 0.276 / 0.223 / 0.261 |
| | 1.0 | 51 | 51 | 53 | 58 | 60 | 0 / 0.112 / 0.222 / 0.174 / 0.211 |
| | 1.2 | 63 | 69 | 74 | 78 | 82 | −0.039 / 0.068 / 0.173 / 0.118 / 0.161 |
| | 1.5 | 70 | 83 | 85 | 89 | 93 | −0.080 / 0.015 / 0.108 / 0.078 / 0.112 |
| AV2 (190) | 0.6 | 16 | 13 | 14 | 18 | 19 | 0.029 / 0.415 / 0.365 / 0.378 / 0.376 |
| | 0.8 | 19 | 20 | 18 | 26 | 27 | 0.026 / 0.391 / 0.347 / 0.344 / 0.343 |
| | 1.0 | 21 | 31 | 29 | 35 | 36 | 0 / 0.338 / 0.294 / 0.296 / 0.294 |
| | 1.2 | 27 | 46 | 45 | 53 | 50 | −0.033 / 0.276 / 0.221 / 0.223 / 0.216 |
| | 1.5 | 34 | 64 | 61 | 70 | 74 | −0.099 / 0.146 / 0.094 / 0.072 / 0.068 |
| Argoverse-HD (71) | 0.6 | 11 | 9 | 10 | 9 | 9 | 0.010 / 0.366 / 0.499 / 0.469 / 0.485 |
| | 0.8 | 11 | 11 | 15 | 14 | 14 | 0.010 / 0.352 / 0.481 / 0.459 / 0.470 |
| | 1.0 | 12 | 26 | 27 | 28 | 27 | 0 / 0.297 / 0.430 / 0.409 / 0.418 |
| | 1.2 | 13 | 34 | 36 | 33 | 32 | −0.013 / 0.236 / 0.364 / 0.346 / 0.359 |
| | 1.5 | 15 | 39 | 41 | 37 | 38 | −0.039 / 0.169 / 0.295 / 0.285 / 0.292 |

- **Lowering thr_decel moves every C1 + v3a row along the same safety–utility curve.**
  - Misses fall steeply: DESP 60 → 36 on nuScenes, 36 → 19 on AV2 and 27 → 9 on Argoverse-HD from 1.0 to 0.6.
  - G rises on every split.
  - False episodes rise: DESP 4.0 → 12.0, 6.5 → 10.6 and 6.3 → 7.1 per 1,000.
- **The deployed threshold 1.0 is not the G-maximising grid point for any C1 + v3a row on any split.** No threshold
  was re-chosen on these numbers.
- **DESP vs the hybrid at the same threshold.**
  - nuScenes: G is higher by 0.035–0.043 at every threshold, the intervals including 0 or touching it.
  - AV2: equal within ±0.006.
  - Argoverse-HD: higher by 0.008–0.016, significant at 0.6 and 1.2.
  - DESP has lower px at every point.
- **W1 at matched components.** At the same threshold, u960 / native + C1 + v3a miss fewer episodes than DESP on
  nuScenes and AV2, at 3–18× px. On Argoverse-HD the misses are equal at 0.6.
  - Cheap's misses are matched by lowering DESP's threshold, not by DESP at the deployed threshold.

## Files
k1_points.csv, k1_diffs.csv, k1_choice_dev.csv (the frozen dev choice), k1_choice.csv (identical: same sha256),
k1_constrained.csv, parity.json, PREREG.md.
