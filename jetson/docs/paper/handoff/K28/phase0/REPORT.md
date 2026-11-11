# K28 phase 0: v5.1-S items that need no new tuning (POST-HOC, test splits of K24)

All numbers are post-hoc analysis on the K24 test frames:
- nuScenes new test: 228 scenes, 9,181 frames;
- AV2 val with the ego mask: 150 logs, 11,818 frames;
- Argoverse-HD val: 24 sequences, 15,006 labelled frames.

Nothing here selects or changes anything (R1). Intervals are paired unit bootstraps (2,000 resamples, seed 0) of (v5.1-S − row).

**Rows where v5.1-S is NOT above** (lower bound ≤ 0; v5.1-S with refiner v3a unless noted):
- **(1) Ablation:**
  - nuScenes:
    - pivots: +0.009 [−0.022, +0.048];
    - single round: +0.115 [−0.006, +0.295];
    - without guard: +0.002 [−0.045, +0.052];
    - field only + components: +0.033 [−0.081, +0.153].
  - AV2:
    - single round: +0.004 [−0.005, +0.014];
    - without guard: +0.015 [−0.002, +0.033].
  - A-HD:
    - single round: −0.006 [−0.018, +0.004];
    - support only + components: +0.052 [−0.034, +0.175];
    - without refiner: +0.057 [−0.029, +0.169].
- **(4) SOTA:**
  - nuScenes SAHI, 640-px slices: +0.105 [−0.054, +0.299] at scale 0.6, and +0.069 [−0.092, +0.255] at scale 1.0.
  - The K12 fine-tuned-detector pipelines are all nominally above v5.1-S. They use a different detector (fine-tuned on nuScenes):
    - FOVEA-ft 0.234 / 0.244;
    - u960-ft 0.241 / 0.272;
    - v3 + fine-tuned verifier 0.297 / 0.295.
- **(5) Sign tests per scene / log:** n.s. against:
  - u1280 + components: nuScenes 45 vs 53, AV2 62 vs 70, A-HD 13 vs 8;
  - nuScenes native: 66 vs 53.

  The oracle router (true V) has the lower loss in most units.

## (1) Ablation: the three missing rows (`ablation_v51.csv`, `ablation_v51.{pdf,png}`)
G with v3a, followed by (v5.1-S − row) [interval]:

| Row | nuScenes | AV2 | A-HD |
|---|---|---|---|
| v5.1-S | 0.199 | 0.262 | 0.338 |
| pivots instead of support | 0.190, +0.009 [−0.022, +0.048] | 0.202, **+0.060 [+0.030, +0.094]** | 0.320, **+0.019 [+0.005, +0.033]** |
| single round | 0.083, +0.115 [−0.006, +0.295] | 0.257, +0.004 [−0.005, +0.014] | 0.345, −0.006 [−0.018, +0.004] |
| without guard | 0.196, +0.002 [−0.045, +0.052] | 0.247, +0.015 [−0.002, +0.033] | 0.295, **+0.043 [+0.014, +0.075]** |

- The phase-0 re-run of v5.1-S reproduces K24 exactly (J, actions and px on 100% of frames).
- Support only / field only and the 2×2 of refiner × T4 are K24's (repeated in the CSV).
- **Without refiner** the same three rows are each significantly below v5.1-S on AV2. On nuScenes, single round is significant (+0.152 [+0.016, +0.323]).
- **Reading:**
  - pivots are significantly worse on AV2 and A-HD;
  - the guard is significant on A-HD only (nominal on AV2);
  - iteration (more than 1 round) is nominal on nuScenes (+0.115, the interval touches 0) and nothing on AV2 / A-HD.

## (2) Cost statistics (`cost_stats_v51.csv`, `gflops_640.csv`)
v5.1-S tensor px, mean / p50 / p95 / p99 / max:
- nuScenes: 0.077 / 0.071 / 0.176 / 0.294 / 0.890;
- AV2: 0.053 / 0.041 / 0.128 / 0.213 / 0.679;
- A-HD: 0.070 / 0.048 / 0.173 / 0.299 / 0.692.

Other cost figures (nuScenes / AV2 / A-HD):

| | nuScenes | AV2 | A-HD |
|---|---|---|---|
| GFLOPs per frame (YOLOv8s 28.6 GFLOPs at 640²), mean | 7.8 | 11.9 | 11.3 |
| GFLOPs per frame, p95 | 17.8 | 28.7 | 28.1 |
| Frames with no crop | 42% | 34% | 27% |
| Crops per frame | 0.66 | 0.87 | 0.97 |
| Rounds 0 / 1 / 2 / 3 | 42 / 54 / 4.2 / 0.1% | 34 / 56 / 10 / 0.4% | 27 / 59 / 13 / 0.8% |

v5.1-L adds 0.019 / 0.017 / 0.019 px of retrospective re-detections.

## (4) v5.1-S minus AdaScale / FOVEA / SAHI (`sota_intervals_v51.csv`)
The stored per-frame results agree with K24's draft cost on 100% of frames. Argoverse-HD has no stored AdaScale / FOVEA / SAHI results.

| Baseline | G | v5.1-S − it, with v3a | without refiner |
|---|---|---|---|
| nuScenes AdaScale | 0.083 | **+0.115 [+0.058, +0.181]** | **+0.079 [+0.029, …]** |
| nuScenes FOVEA KDE_I 384×640 | −0.091 | **+0.290 [+0.040, +0.592]** | outperforms |
| nuScenes FOVEA KDE_S 384×640 | −0.268 | **+0.466 [+0.030, +1.030]** | outperforms |
| nuScenes FOVEA KDE_I 576×960 | 0.074 | **+0.125 [+0.028, +0.236]** | matches (lo −0.006) |
| nuScenes FOVEA KDE_S 576×960 | 0.022 | **+0.177 [+0.051, +0.298]** | outperforms |
| nuScenes SAHI, scale 0.6 / 1.0 | 0.093 / 0.130 | +0.105 [−0.054, +0.299] / +0.069 [−0.092, +0.255] | below |
| AV2 AdaScale | −0.057 | **+0.319 [+0.238, +0.394]** | outperforms |
| AV2 SAHI, scale 0.47 / 1.0 | −0.295 / −0.248 | **+0.557 / +0.509** (lower bounds +0.449 / +0.397) | outperforms |
| AV2 FOVEA KDE_I 960 | −0.023 | **+0.285 [+0.205, +0.369]** | outperforms |

## (5) Loss terms, event rates, sign tests
Files: `loss_terms_{nusc,av2,ahd}_v51.csv`, `sign_tests_v51.csv`, `collisions_v51.csv`.
- **Check:** the sum of the four terms equals J exactly (max |Δ| = 0). The previous action (the draft's) agrees with K24's on 100% of frames.
- **Where v5.1-S's value comes from** (share of the draft's total loss):
  - excessive braking falls from 0.19 to 0.09 on nuScenes, 0.53 to 0.24 on AV2, and 0.79 to 0.48 on A-HD;
  - insufficient braking falls on nuScenes (0.78 to 0.69), but rises slightly on AV2 (0.44 to 0.47) and A-HD falls (0.19 to 0.16).
- **Missed decelerations: v5.1-S misses more than whole frame.** Per 1,000 frames, v5.1-S minus row:

  | | nuScenes | AV2 | A-HD |
  |---|---|---|---|
  | u960 | [+3.9, +9.2] | [+1.9, +7.0] | [+5.6, +21.8] |
  | u1280 + components | [+3.8, +9.1] | [+4.1, +8.7] | n.s. |

  Against the component-equipped heuristics, the difference is n.s. or in v5.1-S's favour; uncertainty + components misses more than v5.1-S.
- **Collisions:** v5.1-S has 9 collision frames on nuScenes (draft 11, u960 17, u1280 6, native 7), 5 on AV2 (draft 4), and 0 on A-HD.
- **Sign tests:** v5.1-S has the lower loss per scene / log against:

  | | nuScenes | AV2 | A-HD |
  |---|---|---|---|
  | draft | 119 vs 29 | 116 vs 27 | 22 vs 1 |
  | u960 | 79 vs 39 (p 0.0003) | 104 vs 36 | 22 vs 1 |
  | previous frozen | 78 vs 31 | 116 vs 26 | 22 vs 1 |

  These are all p < 0.001 unless a p-value is shown.
  - n.s. against u1280 + components and nuScenes native.
  - Below the oracle router on nuScenes (18 vs 88) and AV2 (35 vs 103).

Code:
- `remote/A_method/k28_phase0_abl.py` (md5 c39d3920…);
- `k28_phase0.py` (94711d27…; the cost part's column filter was fixed and re-run before reading the table);
- chain `k28_p0_chain.sh`.
