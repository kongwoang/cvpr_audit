# K37-2 — decision quality under deadlines (post-hoc, CPU, existing Jetson logs; APPROXIMATION)

**Label.** This is an approximation. Each row's per-frame latency and action come from separate Jetson runs, each with
its own history: K29, K30-J1 and K34-3, all on the trt32 fast path. Substituting a held or fallback action does not
re-run the pipeline, so its later history (C1 chains, refiner) is that of the original run.

## Setup
- **Frames.** The K29 frames: nuScenes test, 20 scenes / 805 frames; AV2, 8 logs / 586 frames with the ego mask.
  Three repetitions each. Only 15 (nuScenes) and 8 (AV2) required episodes, so missed rates are coarse.
- **Rows and runs.**
  - cheap, u960 + C1 + v3a, native + C1 + v3a: K29 runs r{1,2,3}b.
  - hybrid tsp8 at ×1.0 + C1 + v3a: K30-J1 runs j1x_r{1,2,3}.
  - v6-S, v6-S ≤ 2 batches, B2K-A: K34-3 jetson_frames.csv.
  - Repetition r of every row is paired.
- **Rule (a), hold (every row).** If a frame's latency > D, its action is the row's last on-time action in that unit
  (KEEP before the first one).
- **Rule (b), anytime (DESP only).** If v6-S is late, use the v6-S ≤ 2 action of the same repetition if it is on time.
  Otherwise use the cheap action of the frame (latency = the cheap latency of the K29 run). Hold if even that is late,
  which never happened.
- **Cost.** K28 decision_cost(action, a_gt, previous cheap action), with the stored GT demand a_gt (results/k30/gt).
  - Check (checks.json): recomputing every stored action's cost reproduces the stored J / decision_loss to 1e-15 on all
    frames.
  - The previous cheap action is taken over the whole run, warm-up frames included, as the stored costs do.
  - At D = ∞ the G values reproduce K30-J1: v6-S 0.380 / 0.419, hybrid 0.373 / 0.417.
- **Statistics.**
  - G = 1 − ΣJ / ΣJ_cheap; the cheap denominator has no deadline (cheap is never late).
  - Missed rate: K31 required episodes (L = 1). False episodes per 1,000 frames.
  - Paired unit bootstrap: 2,000 draws, seed 0; each drawn unit carries all three repetitions.
- **Files.** k37_2_rows.csv (all rows × D, with intervals and the anytime source shares), k37_2_diffs.csv (row − v6-S
  hold, paired), checks.json. Script: scripts/k37_2_deadline.py.

### nuScenes test 20 scenes (805 frames × 3 reps; 15 required episodes): G [95% CI] / deadline-miss share

| row | D = 33.3 | D = 50.0 | D = 66.7 | D = 83.3 | D = 100.0 | D = ∞ |
|---|---|---|---|---|---|---|
| cheap | 0.000 [0.00, 0.00] / 0.0% | 0.000 [0.00, 0.00] / 0.0% | 0.000 [0.00, 0.00] / 0.0% | 0.000 [0.00, 0.00] / 0.0% | 0.000 [0.00, 0.00] / 0.0% | 0.000 [0.00, 0.00] / 0.0% |
| u960 + C1 + v3a | 0.438 [0.29, 0.57] / 3.2% | 0.391 [0.24, 0.49] / 0.5% | 0.386 [0.23, 0.48] / 0.0% | 0.386 [0.23, 0.48] / 0.0% | 0.386 [0.23, 0.48] / 0.0% | 0.386 [0.23, 0.48] / 0.0% |
| native + C1 + v3a | -8.024 [-14.47, -0.08] / 90.2% | 0.364 [0.14, 0.59] / 1.2% | 0.364 [0.14, 0.59] / 0.0% | 0.364 [0.14, 0.59] / 0.0% | 0.364 [0.14, 0.59] / 0.0% | 0.364 [0.14, 0.59] / 0.0% |
| hybrid tsp8@x1.0 + C1 + v3a | -0.290 [-1.13, 0.32] / 22.9% | 0.121 [-0.60, 0.43] / 14.0% | 0.193 [-0.38, 0.45] / 7.3% | 0.267 [-0.08, 0.46] / 3.1% | 0.345 [0.13, 0.49] / 1.1% | 0.373 [0.21, 0.51] / 0.0% |
| v6-S | -0.342 [-1.18, 0.26] / 22.2% | 0.297 [-0.02, 0.48] / 9.3% | 0.370 [0.19, 0.50] / 2.6% | 0.374 [0.19, 0.51] / 1.0% | 0.377 [0.20, 0.51] / 0.2% | 0.380 [0.21, 0.51] / 0.0% |
| v6-S <= 2 | -0.128 [-0.71, 0.28] / 19.1% | 0.385 [0.23, 0.51] / 3.2% | 0.391 [0.24, 0.52] / 0.2% | 0.382 [0.21, 0.51] / 0.0% | 0.382 [0.21, 0.51] / 0.0% | 0.382 [0.21, 0.51] / 0.0% |
| B2K-A | -0.353 [-1.19, 0.26] / 20.2% | 0.369 [0.18, 0.51] / 4.6% | 0.391 [0.24, 0.52] / 0.4% | 0.382 [0.21, 0.51] / 0.0% | 0.382 [0.21, 0.51] / 0.0% | 0.382 [0.21, 0.51] / 0.0% |
| DESP anytime (v6-S -> <= 2 -> cheap) | 0.099 [0.02, 0.20] / 22.2% | 0.378 [0.23, 0.51] / 9.3% | 0.383 [0.22, 0.51] / 2.6% | 0.382 [0.21, 0.51] / 1.0% | 0.380 [0.21, 0.51] / 0.2% | 0.380 [0.21, 0.51] / 0.0% |

### nuScenes test 20 scenes (805 frames × 3 reps; 15 required episodes): missed rate / false per 1,000

| row | D = 33.3 | D = 50.0 | D = 66.7 | D = 83.3 | D = 100.0 | D = ∞ |
|---|---|---|---|---|---|---|
| cheap | 0.27 / 8.7 | 0.27 / 8.7 | 0.27 / 8.7 | 0.27 / 8.7 | 0.27 / 8.7 | 0.27 / 8.7 |
| u960 + C1 + v3a | 0.40 / 2.5 | 0.40 / 5.0 | 0.40 / 5.0 | 0.40 / 5.0 | 0.40 / 5.0 | 0.40 / 5.0 |
| native + C1 + v3a | 1.00 / 0.0 | 0.40 / 1.2 | 0.40 / 1.2 | 0.40 / 1.2 | 0.40 / 1.2 | 0.40 / 1.2 |
| hybrid tsp8@x1.0 + C1 + v3a | 0.73 / 0.0 | 0.64 / 0.8 | 0.53 / 1.2 | 0.47 / 2.5 | 0.40 / 2.5 | 0.40 / 2.5 |
| v6-S | 0.76 / 0.0 | 0.60 / 2.5 | 0.40 / 2.5 | 0.40 / 2.5 | 0.40 / 2.9 | 0.40 / 3.7 |
| v6-S <= 2 | 0.82 / 0.0 | 0.40 / 2.9 | 0.40 / 3.7 | 0.40 / 3.7 | 0.40 / 3.7 | 0.40 / 3.7 |
| B2K-A | 0.82 / 0.0 | 0.47 / 2.5 | 0.40 / 3.7 | 0.40 / 3.7 | 0.40 / 3.7 | 0.40 / 3.7 |
| DESP anytime (v6-S -> <= 2 -> cheap) | 0.40 / 6.2 | 0.40 / 3.7 | 0.40 / 3.7 | 0.40 / 3.7 | 0.40 / 3.7 | 0.40 / 3.7 |

### AV2 8 logs (586 frames × 3 reps; 8 required episodes): G [95% CI] / deadline-miss share

| row | D = 33.3 | D = 50.0 | D = 66.7 | D = 83.3 | D = 100.0 | D = ∞ |
|---|---|---|---|---|---|---|
| cheap | 0.000 [0.00, 0.00] / 0.0% | 0.000 [0.00, 0.00] / 0.0% | 0.000 [0.00, 0.00] / 0.0% | 0.000 [0.00, 0.00] / 0.0% | 0.000 [0.00, 0.00] / 0.0% | 0.000 [0.00, 0.00] / 0.0% |
| u960 + C1 + v3a | -0.757 [-2.38, 0.77] / 54.4% | 0.393 [0.22, 0.65] / 4.6% | 0.323 [0.12, 0.63] / 0.1% | 0.323 [0.12, 0.63] / 0.0% | 0.323 [0.12, 0.63] / 0.0% | 0.323 [0.12, 0.63] / 0.0% |
| native + C1 + v3a | -1.450 [-3.58, 0.77] / 100.0% | -1.450 [-3.58, 0.77] / 100.0% | -1.450 [-3.58, 0.77] / 83.8% | 0.474 [0.16, 0.77] / 11.1% | 0.410 [0.24, 0.68] / 0.2% | 0.410 [0.24, 0.68] / 0.0% |
| hybrid tsp8@x1.0 + C1 + v3a | 0.246 [-0.22, 0.77] / 58.1% | 0.085 [-0.53, 0.67] / 38.3% | 0.384 [0.16, 0.63] / 22.0% | 0.467 [0.35, 0.65] / 12.6% | 0.402 [0.23, 0.64] / 7.1% | 0.417 [0.25, 0.66] / 0.0% |
| v6-S | 0.243 [-0.22, 0.77] / 56.3% | 0.323 [0.11, 0.59] / 35.7% | 0.310 [0.14, 0.56] / 16.2% | 0.431 [0.26, 0.66] / 8.0% | 0.427 [0.26, 0.66] / 3.2% | 0.419 [0.25, 0.66] / 0.0% |
| v6-S <= 2 | 0.243 [-0.22, 0.77] / 56.5% | 0.476 [0.25, 0.73] / 28.6% | 0.433 [0.26, 0.67] / 7.3% | 0.423 [0.25, 0.66] / 2.3% | 0.423 [0.25, 0.66] / 0.5% | 0.423 [0.25, 0.66] / 0.0% |
| B2K-A | 0.242 [-0.23, 0.77] / 57.2% | 0.380 [0.13, 0.70] / 37.6% | 0.447 [0.28, 0.68] / 9.6% | 0.435 [0.27, 0.66] / 4.0% | 0.428 [0.27, 0.66] / 1.1% | 0.428 [0.27, 0.66] / 0.0% |
| DESP anytime (v6-S -> <= 2 -> cheap) | 0.064 [0.02, 0.14] / 56.3% | 0.200 [0.10, 0.36] / 35.7% | 0.359 [0.17, 0.66] / 16.2% | 0.419 [0.25, 0.66] / 8.0% | 0.423 [0.25, 0.66] / 3.2% | 0.419 [0.25, 0.66] / 0.0% |

### AV2 8 logs (586 frames × 3 reps; 8 required episodes): missed rate / false per 1,000

| row | D = 33.3 | D = 50.0 | D = 66.7 | D = 83.3 | D = 100.0 | D = ∞ |
|---|---|---|---|---|---|---|
| cheap | 0.00 / 56.3 | 0.00 / 56.3 | 0.00 / 56.3 | 0.00 / 56.3 | 0.00 / 56.3 | 0.00 / 56.3 |
| u960 + C1 + v3a | 0.58 / 0.0 | 0.12 / 6.8 | 0.12 / 10.2 | 0.12 / 10.2 | 0.12 / 10.2 | 0.12 / 10.2 |
| native + C1 + v3a | 1.00 / 0.0 | 1.00 / 0.0 | 1.00 / 0.0 | 0.12 / 2.8 | 0.12 / 5.1 | 0.12 / 5.1 |
| hybrid tsp8@x1.0 + C1 + v3a | 0.62 / 0.0 | 0.38 / 1.7 | 0.25 / 6.3 | 0.12 / 8.5 | 0.12 / 8.5 | 0.12 / 8.5 |
| v6-S | 0.62 / 0.0 | 0.12 / 4.0 | 0.12 / 6.3 | 0.12 / 8.5 | 0.12 / 8.5 | 0.12 / 8.5 |
| v6-S <= 2 | 0.62 / 0.0 | 0.12 / 5.1 | 0.12 / 6.8 | 0.12 / 8.5 | 0.12 / 8.5 | 0.12 / 8.5 |
| B2K-A | 0.62 / 0.0 | 0.12 / 1.7 | 0.12 / 5.1 | 0.12 / 8.5 | 0.12 / 8.5 | 0.12 / 8.5 |
| DESP anytime (v6-S -> <= 2 -> cheap) | 0.00 / 39.8 | 0.12 / 28.4 | 0.12 / 8.5 | 0.12 / 8.5 | 0.12 / 8.5 | 0.12 / 8.5 |

### DESP anytime: which action is used (share of frames)

| dataset | D (ms) | v6-S | v6-S ≤ 2 | cheap |
|---|---|---|---|---|
| nuScenes | 33.3 | 77.8% | 3.8% | 18.4% |
| nuScenes | 50 | 90.7% | 6.4% | 2.9% |
| nuScenes | 66.7 | 97.4% | 2.4% | 0.2% |
| AV2 | 33.3 | 43.7% | 1.0% | 55.3% |
| AV2 | 50 | 64.3% | 10.3% | 25.4% |
| AV2 | 66.7 | 83.8% | 9.7% | 6.5% |
| AV2 | 83.3 | 92.0% | 6.1% | 1.9% |

## Reading (approximation; few episodes)
- **nuScenes, D ≥ 66.7 ms (≥ 15 Hz).** Every C1 + v3a row except the hybrid stays within 0.02 of its D = ∞ G.
  - v6-S 0.370 vs 0.380.
  - The hybrid loses 0.18 at 66.7 ms (7.3% late) and stays below v6-S until 100 ms.
- **AV2 needs D ≥ 83.3 ms** for v6-S, native and the hybrid to recover their D = ∞ G (within 0.07).
  - At 66.7 ms v6-S hold is 16% late and loses 0.11 (0.310 vs 0.419); native + C1 + v3a is still 84% late.
  - B2K-A and v6-S ≤ 2 are significantly better than v6-S hold at 66.7 ms: +0.14 [0.03, 0.29] and +0.12 [0.00, 0.28].
- **At 50 ms, the bounded variants are the best DESP form.**
  - nuScenes: v6-S ≤ 2 0.385 and B2K-A 0.369 vs v6-S hold 0.297 (9.3% late).
  - AV2: v6-S ≤ 2 0.476, B2K-A 0.380, v6-S 0.323.
  - Neither difference vs v6-S hold excludes 0, given 8–20 units.
- **The anytime rule helps on nuScenes and hurts on AV2.**
  - nuScenes: it lifts DESP at 33.3 ms from −0.342 to +0.099 and at 50 ms to 0.378.
  - AV2: the cheap fallback adds false braking. At 33.3 ms it is 39.8 vs 0 false / 1,000 for hold, and G is 0.064 vs
    0.243. At 50 ms G is 0.200 vs 0.323.
  - Holding a recent C1 + v3a action is better than falling back to cheap on AV2 at these deadlines.
- **Whole-frame rows.** native + C1 + v3a collapses below 50 ms (nuScenes, 90% late at 33.3 ms) and below 83.3 ms on
  AV2 (100% late at 33.3 and 50 ms). u960 + C1 + v3a is nearly always on time on nuScenes; on AV2 it is 54% late at
  33.3 ms.
- **Caveats.**
  - Holding a braking action can lower the cost on some frames: hold G at 50 ms sometimes exceeds the D = ∞ value
    (v6-S ≤ 2 on AV2: 0.476 vs 0.423).
  - This is an artefact of not re-running the pipeline. It is one reason this analysis is labelled an approximation.
