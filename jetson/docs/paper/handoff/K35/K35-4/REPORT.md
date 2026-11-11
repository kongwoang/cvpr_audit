# K35-4 — confirmatory holdout evaluation (Argoverse 1.1 TRAIN, 65 units), run ONCE under PREREG.md

**Pre-registered verdict: v7a PASSES the kill rule.** The pass is narrow.
- G(v7a) − G(v6-S) = **+0.0029 [−0.0020, +0.0108]**. The lower bound is above −0.01, so the first kill condition does not
  fire. The difference is not significant.
- Missed rate: v7a **42 / 154 = 0.273** vs v6-S **43 / 154 = 0.279**. The second kill condition (missed rate not lower) does
  not fire, by a margin of one episode. The interval of the difference is [−0.023, 0.000].
- So the decision rule makes v7a the proposed extension. The holdout shows **no quality gain and no quality loss**
  relative to v6-S.
- What v7a changes measurably:
  - **network px −0.0079 [−0.0103, −0.0059]** (−10%);
  - **false episodes +1.02 [+0.49, +1.62] per 1,000 frames** (6.55 → 7.56).
- v7b was dropped at the K35-4b gate before PREREG and is not analysed.

## Run
- **Code and data.** PREREG.md (sha256 931c7058…) was frozen before the run. Harness: VUW
  experiments/k35_holdout_20261025 (hashes in PREREG). Launched 2026-10-25 20:09 NZ, 4 shards; all finished 2026-10-26
  03:40 NZ.
- **Integrity checks (pre-registered; all passed before the analysis).**
  - check.py train (check_train.json): runner_h's recomputed cheap / v6-S equal k28_p3_h's on all 65 units, 0 differences.
  - The analysis asserts that k33_eval_h's cheap / v6-S equal k28_p3_h's.
  - It also asserts that the evaluated frames and act_gt equal K35-0's GT-only computation: 39,265 frames, 65 units.
- **Analysis.** k35_k4_analyze.py, sha256 00ffce85…, unchanged since PREREG.
- **Statistics.** Paired unit bootstrap over 65 units: 2,000 draws, seed 0, percentile 95%.

## All rows (k4_rows.csv, k4_diffs.csv, k4_episodes.csv)

| row | G [95% CI] | px | false / 1,000 | missed (154 req.) | below hard (137) | at / above hard (17) | HARD-level missed (32) |
|---|---|---|---|---|---|---|---|
| cheap | 0 | 0.031 | 34.48 | 18 (0.117) | 18 | 0 | 4 |
| **v6-S** | **0.407 [0.313, 0.480]** | 0.077 | 6.55 | 43 (0.279) | 43 | 0 | 10 |
| u960 + C1 + v3a | 0.400 [0.321, 0.466] | 0.253 | 6.42 | 42 (0.273) | 42 | 0 | 11 |
| native + C1 + v3a | 0.417 [0.313, 0.494] | 1.013 | 8.20 | 44 (0.286) | 43 | 1 | 10 |
| hybrid tsp8@S + C1 + v3a | 0.410 [0.321, 0.480] | 0.126 | 7.67 | 44 (0.286) | 44 | 0 | 10 |
| uncertainty unc2@S + C1 + v3a | 0.338 [0.259, 0.404] | 0.152 | 20.68 | 32 (0.208) | 32 | 0 | 11 |
| AnA iterated (ana05hinfi) + C1 + v3a | 0.408 [0.333, 0.475] (G_cap 0.399) | 0.124 (cap 0.076) | 7.54 | 44 (0.286) | 44 | 0 | 11 |
| **v7a** = B2K-A + temporal evidence | **0.409 [0.324, 0.480]** | **0.069** | 7.56 | 42 (0.273) | 42 | 0 | 10 |

### Differences row − v6-S [95% CI] (bold = excludes 0)

| row | ΔG | Δpx | Δfalse / 1,000 | Δmissed rate |
|---|---|---|---|---|
| cheap | **−0.407** | **−0.045** | **+27.94** | **−0.162** |
| u960 + C1 + v3a | −0.007 [−0.021, +0.012] | **+0.177** | −0.13 [−0.72, +0.55] | −0.006 [−0.042, +0.026] |
| native + C1 + v3a | +0.010 [−0.008, +0.026] | **+0.937** | **+1.66** | +0.006 [−0.030, +0.040] |
| hybrid tsp8@S | +0.003 [−0.005, +0.011] | **+0.050** | **+1.12** | +0.006 [0.000, +0.023] |
| unc2@S | **−0.069** | **+0.075** | **+14.13** | **−0.071** |
| AnA iterated | +0.002 [−0.012, +0.022] | **+0.048** | **+0.99** | +0.006 [0.000, +0.023] |
| **v7a** | +0.003 [−0.002, +0.011] | **−0.008** | **+1.02** | −0.006 [−0.023, 0.000] |

## Reading
- **v6-S holds on untouched logs.**
  - G = 0.407 at px 0.077.
  - Every whole-frame and selection baseline that uses the same C1 + v3a is within ±0.01 G with no significant difference:
    u960, native, hybrid, AnA iterated.
  - Those baselines use 1.6× (hybrid, AnA) to 13× (native) the network px.
  - unc2@S is significantly worse: −0.069 G, +14 false episodes / 1,000.
- **W1 persists on the holdout.** v6-S misses 43 / 154 required episodes vs cheap's 18. Cheap gets there at the cost of
  34.5 false episodes / 1,000 frames vs 6.5. No C1 + v3a row closes this gap (42–44 missed), so the misses are not
  specific to v6-S's selection.
  - The 17 hard-level episodes are all detected by every row except native + C1 + v3a, which misses 1.
- **v7a.**
  - It meets the pre-registered criteria, which also hold "at least as good as v6-S at lower cost".
  - Its quality is statistically indistinguishable from v6-S: ΔG n.s., one episode fewer missed.
  - Its px is lower, as on dev.
  - It has a small but significant increase in false episodes, about one more per 1,000 frames, which the dev screen
    also showed.
  - The paper must not describe v7a as improving decision quality on the holdout. Holdout-supported claims: the
    same decision value at 10% fewer network px; and, from the K35-4b dev timing, a lower p95 latency (−11.9 / −30.3 ms).
- **Caveat (from PREREG).** Same sensor, cities and annotation process as the Argoverse-HD test split. Only 17
  hard-level episodes exist.

## Next (pre-registered consequence of a pass)
1. Run v7a once, post-hoc, on the three test splits (labelled post-hoc).
2. Time v7a on the Jetson on the K29 frames (nuScenes test 20 scenes, AV2 8 logs, trt32, 3 repetitions) next to v6-S
   and B2K-A. Use the K35 timing-note runtime, with parity reporting.
