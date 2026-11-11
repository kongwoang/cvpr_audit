# K35-4 — v7a on the three test splits (POST-HOC; run once after the holdout pass)

**Label: post-hoc.** These splits were used for every earlier frozen-v6-S result. Nothing here selects, tunes or
re-decides anything. v7a was proposed by the pre-registered holdout rule (K35-4/REPORT.md) before this run.
- **Run.** experiments/k35_holdout_20261025/r3/runner_t.py (sha256 53557b0d…), an isolated copy of R3 runner_low.
  - The frozen test unit source, and only the v6S / v6S_b2A_retro rows.
  - AV2 with the K10 ego mask.
  - VUW 2026-10-26.
- **Check.** posthoc_check.py (check.json): runner_t's recomputed cheap / v6-S (a, J, px) equal the stored K28 test
  records on all 228 / 150 / 24 units, with 0 differences.
- **Analysis.** remote/A_method/k35_k4_posthoc.py, with the metric definitions of the pre-registered holdout analysis.
  - Base: k34_common.load_base (evaluated frames 9,181 / 11,818 / 15,006).
  - Paired unit bootstrap: 2,000 draws, seed 0.
  - The other rows are the stored K28 test rows, as in K31 / K34.

## v7a − v6-S [95% CI] (posthoc_diffs.csv; bold = excludes 0)

| split | ΔG | Δpx | Δfalse / 1,000 | Δmissed rate | missed v6-S → v7a |
|---|---|---|---|---|---|
| nuScenes test (228 scenes, 147 episodes) | +0.014 [−0.001, +0.035] | **−0.0040** | +0.33 [−0.11, +0.87] | **−0.034 [−0.067, −0.007]** | 60 → 55 |
| AV2 val (150 logs, 190 episodes) | **+0.025 [+0.001, +0.063]** | **−0.0041** | **+1.10 [+0.34, +1.95]** | −0.016 [−0.042, 0.000] | 36 → 33 |
| Argoverse-HD val (24 seq., 71 episodes) | −0.005 [−0.015, +0.001] | **−0.0052** | +1.53 [−0.22, +4.07] | 0 | 27 → 27 |

- Hard stratum (peak ≥ 3.5 m/s²): nuScenes 8 → 8 / 38; AV2 6 → 5 / 55; Argoverse-HD 2 → 2 / 10.
- HARD level: unchanged on all three splits (21 / 11 / 8).
- HARD-level false episodes per 1,000 rise by +0.44 [0.11, 0.87] on nuScenes and +0.85 [0.25, 1.53] on AV2.

## All rows (posthoc_rows.csv)

| split | row | G [95% CI] | px | false / 1,000 | missed rate |
|---|---|---|---|---|---|
| nuScenes | cheap | 0 | 0.043 | 22.87 | 0.347 |
| nuScenes | v6-S | 0.211 [0.104, 0.348] | 0.078 | 4.03 | 0.408 |
| nuScenes | u960 + C1 + v3a | 0.112 [−0.183, 0.373] | 0.384 | 6.43 | 0.347 |
| nuScenes | native + C1 + v3a | 0.222 [0.077, 0.362] | 1.031 | 4.90 | 0.361 |
| nuScenes | hybrid tsp8@S | 0.174 [0.020, 0.347] | 0.108 | 4.25 | 0.395 |
| nuScenes | unc2@S | −0.059 [−0.330, 0.170] | 0.102 | 7.19 | 0.429 |
| nuScenes | **v7a** | **0.224 [0.114, 0.361]** | 0.074 | 4.36 | 0.374 |
| AV2 | cheap | 0 | 0.026 | 34.78 | 0.111 |
| AV2 | v6-S | 0.294 [0.232, 0.356] | 0.054 | 6.52 | 0.189 |
| AV2 | u960 + C1 + v3a | 0.338 [0.278, 0.399] | 0.232 | 7.87 | 0.163 |
| AV2 | native + C1 + v3a | 0.294 [0.223, 0.366] | 1.012 | 8.63 | 0.153 |
| AV2 | hybrid tsp8@S | 0.296 [0.238, 0.357] | 0.071 | 6.09 | 0.184 |
| AV2 | unc2@S | 0.219 [0.167, 0.278] | 0.080 | 14.89 | 0.163 |
| AV2 | **v7a** | **0.319 [0.268, 0.376]** | 0.050 | 7.62 | 0.174 |
| Argoverse-HD | cheap | 0 | 0.031 | 25.12 | 0.169 |
| Argoverse-HD | v6-S | 0.418 [0.314, 0.517] | 0.071 | 6.26 | 0.380 |
| Argoverse-HD | u960 + C1 + v3a | 0.297 [0.106, 0.452] | 0.253 | 5.46 | 0.366 |
| Argoverse-HD | native + C1 + v3a | 0.430 [0.322, 0.534] | 1.013 | 7.00 | 0.380 |
| Argoverse-HD | hybrid tsp8@S | 0.408 [0.307, 0.504] | 0.108 | 6.66 | 0.380 |
| Argoverse-HD | unc2@S | 0.282 [0.202, 0.370] | 0.123 | 15.86 | 0.394 |
| Argoverse-HD | **v7a** | **0.413 [0.311, 0.509]** | 0.065 | 7.80 | 0.380 |

## Reading (post-hoc; supplementary evidence only)
- The pattern matches dev and the holdout:
  - v7a uses fewer px than v6-S on every split (−5 to −8%);
  - it misses no more required episodes: −5 nuScenes, −3 AV2, 0 Argoverse-HD;
  - it adds about +0.3 to +1.5 false episodes per 1,000 frames.
- ΔG is positive on nuScenes (n.s.) and on AV2 (interval just above 0), and slightly negative on Argoverse-HD (n.s.).
- v7a's G is the highest of all rows on nuScenes and second on AV2, behind u960 + C1 + v3a at 4.6× px. On Argoverse-HD
  it is within 0.02 of the best rows (native 0.430 at 16× px).
- These are post-hoc test numbers on splits used throughout development. The confirmatory claim remains the holdout
  result: no quality difference, fewer px, slightly more false episodes. W1 persists here too: v7a still misses more
  episodes than cheap on every split.
