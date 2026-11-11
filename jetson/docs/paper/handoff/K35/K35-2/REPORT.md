# K35-2 — frame-level recoverability classes and the best-of ceiling (POST-HOC, three test splits)

**Post-hoc analysis** of stored K28 p3 test rows (actions and decision costs); no pipeline was run.
- Code: remote/A_method/k35_k2.py (md5 16b25ed3), run on VUW 2026-10-25.
- "Correct" means the action equals the annotated required action act_gt. Required frames have act_gt ≥ DECELERATE.
- Classes are exclusive, per reference F:
  - both_correct;
  - cheap_wrong_F_correct (recoverable by escalation);
  - both_wrong;
  - cheap_correct_F_wrong (negative escalation).
- v6-S's correct rate carries a paired unit bootstrap interval (2,000, seed 0). Mean decision loss is K28's stored
  per-frame cost (the jerk term uses cheap's previous action, as in K28).

## Classes and v6-S's outcome

**F = u960 + C1 + v3a, required frames**

| dataset | class | frames | share | v6-S correct [95% CI] | v6-S under / over | mean J cheap / F / v6-S |
|---|---|---|---|---|---|---|
| nuScenes | both_correct | 428 | 0.461 | 0.956 [0.923, 0.977] | 0.040 / 0.005 | 0.85 / 0.85 / 1.27 |
| nuScenes | cheap_wrong_F_correct | 157 | 0.169 | 0.739 [0.602, 0.843] | 0.261 / 0.000 | 8.11 / 0.85 / 2.54 |
| nuScenes | both_wrong | 281 | 0.303 | 0.036 [0.015, 0.062] | 0.886 / 0.078 | 6.55 / 6.46 / 6.34 |
| nuScenes | cheap_correct_F_wrong | 62 | 0.067 | 0.242 [0.030, 0.463] | 0.677 / 0.081 | 1.83 / 17.62 / 5.25 |
| AV2 | both_correct | 1645 | 0.576 | 0.979 [0.968, 0.988] | 0.019 / 0.002 | 0.57 / 0.57 / 0.76 |
| AV2 | cheap_wrong_F_correct | 597 | 0.209 | 0.918 [0.884, 0.945] | 0.072 / 0.010 | 3.67 / 0.44 / 1.13 |
| AV2 | both_wrong | 452 | 0.158 | 0.075 [0.049, 0.104] | 0.617 / 0.308 | 4.19 / 4.15 / 3.94 |
| AV2 | cheap_correct_F_wrong | 160 | 0.056 | 0.281 [0.205, 0.359] | 0.656 / 0.062 | 0.41 / 4.14 / 3.57 |
| Argoverse-HD | both_correct | 1014 | 0.282 | 0.979 [0.965, 0.992] | 0.021 / 0.000 | 0.28 / 0.28 / 0.31 |
| Argoverse-HD | cheap_wrong_F_correct | 1232 | 0.343 | 0.950 [0.925, 0.971] | 0.045 / 0.005 | 2.50 / 0.17 / 0.25 |
| Argoverse-HD | both_wrong | 1065 | 0.296 | 0.074 [0.037, 0.153] | 0.441 / 0.485 | 2.42 / 2.33 / 2.21 |
| Argoverse-HD | cheap_correct_F_wrong | 282 | 0.078 | 0.174 [0.052, 0.340] | 0.823 / 0.004 | 0.26 / 1.95 / 1.61 |

**F = u960 + C1 + v3a, all frames**

| dataset | class | frames | share | v6-S correct [95% CI] | v6-S under / over | mean J cheap / F / v6-S |
|---|---|---|---|---|---|---|
| nuScenes | both_correct | 8176 | 0.891 | 0.997 [0.995, 0.998] | 0.002 / 0.001 | 0.10 / 0.10 / 0.13 |
| nuScenes | cheap_wrong_F_correct | 480 | 0.052 | 0.904 [0.863, 0.939] | 0.085 / 0.010 | 3.89 / 0.46 / 1.03 |
| nuScenes | both_wrong | 369 | 0.040 | 0.065 [0.042, 0.097] | 0.675 / 0.260 | 5.60 / 5.58 / 5.40 |
| nuScenes | cheap_correct_F_wrong | 156 | 0.017 | 0.532 [0.395, 0.669] | 0.269 / 0.199 | 0.86 / 7.89 / 2.30 |
| AV2 | both_correct | 8829 | 0.747 | 0.994 [0.992, 0.996] | 0.004 / 0.002 | 0.18 / 0.18 / 0.22 |
| AV2 | cheap_wrong_F_correct | 1845 | 0.156 | 0.953 [0.942, 0.964] | 0.023 / 0.023 | 2.44 / 0.39 / 0.63 |
| AV2 | both_wrong | 880 | 0.074 | 0.101 [0.076, 0.135] | 0.317 / 0.582 | 3.52 / 3.40 / 3.24 |
| AV2 | cheap_correct_F_wrong | 264 | 0.022 | 0.424 [0.332, 0.512] | 0.398 / 0.178 | 0.33 / 3.14 / 2.40 |
| Argoverse-HD | both_correct | 8922 | 0.595 | 0.996 [0.992, 0.998] | 0.002 / 0.002 | 0.11 / 0.11 / 0.12 |
| Argoverse-HD | cheap_wrong_F_correct | 2773 | 0.185 | 0.966 [0.951, 0.979] | 0.020 / 0.014 | 2.52 / 0.30 / 0.36 |
| Argoverse-HD | both_wrong | 2647 | 0.176 | 0.119 [0.054, 0.209] | 0.178 / 0.704 | 3.00 / 2.87 / 2.56 |
| Argoverse-HD | cheap_correct_F_wrong | 664 | 0.044 | 0.586 [0.225, 0.828] | 0.349 / 0.065 | 0.18 / 2.79 / 0.79 |

**F = native + C1 + v3a, required frames**

| dataset | class | frames | share | v6-S correct [95% CI] | v6-S under / over | mean J cheap / F / v6-S |
|---|---|---|---|---|---|---|
| nuScenes | both_correct | 434 | 0.468 | 0.945 [0.906, 0.970] | 0.048 / 0.007 | 0.99 / 0.99 / 1.57 |
| nuScenes | cheap_wrong_F_correct | 150 | 0.162 | 0.760 [0.640, 0.860] | 0.207 / 0.033 | 8.55 / 0.87 / 2.51 |
| nuScenes | both_wrong | 288 | 0.310 | 0.042 [0.015, 0.080] | 0.899 / 0.059 | 6.36 / 6.63 / 6.27 |
| nuScenes | cheap_correct_F_wrong | 56 | 0.060 | 0.250 [0.101, 0.427] | 0.679 / 0.071 | 0.84 / 9.70 / 3.32 |
| AV2 | both_correct | 1643 | 0.576 | 0.977 [0.966, 0.985] | 0.021 / 0.002 | 0.56 / 0.56 / 0.78 |
| AV2 | cheap_wrong_F_correct | 594 | 0.208 | 0.857 [0.811, 0.896] | 0.089 / 0.054 | 2.82 / 0.30 / 0.75 |
| AV2 | both_wrong | 455 | 0.159 | 0.160 [0.106, 0.221] | 0.591 / 0.248 | 5.29 / 5.64 / 4.42 |
| AV2 | cheap_correct_F_wrong | 162 | 0.057 | 0.315 [0.212, 0.413] | 0.630 / 0.056 | 0.50 / 4.33 / 3.30 |
| Argoverse-HD | both_correct | 1053 | 0.293 | 0.967 [0.952, 0.979] | 0.033 / 0.000 | 0.28 / 0.28 / 0.32 |
| Argoverse-HD | cheap_wrong_F_correct | 1335 | 0.372 | 0.894 [0.808, 0.933] | 0.057 / 0.049 | 2.46 / 0.18 / 0.35 |
| Argoverse-HD | both_wrong | 962 | 0.268 | 0.059 [0.032, 0.119] | 0.467 / 0.474 | 2.48 / 2.39 / 2.28 |
| Argoverse-HD | cheap_correct_F_wrong | 243 | 0.068 | 0.099 [0.043, 0.235] | 0.897 / 0.004 | 0.28 / 2.06 / 1.78 |

**F = native + C1 + v3a, all frames**

| dataset | class | frames | share | v6-S correct [95% CI] | v6-S under / over | mean J cheap / F / v6-S |
|---|---|---|---|---|---|---|
| nuScenes | both_correct | 8207 | 0.894 | 0.996 [0.994, 0.998] | 0.003 / 0.001 | 0.11 / 0.11 / 0.14 |
| nuScenes | cheap_wrong_F_correct | 496 | 0.054 | 0.875 [0.835, 0.913] | 0.062 / 0.062 | 3.94 / 0.45 / 1.12 |
| nuScenes | both_wrong | 353 | 0.038 | 0.068 [0.038, 0.104] | 0.734 / 0.198 | 5.60 / 5.83 / 5.47 |
| nuScenes | cheap_correct_F_wrong | 125 | 0.014 | 0.464 [0.339, 0.589] | 0.304 / 0.232 | 0.52 / 4.97 / 1.74 |
| AV2 | both_correct | 8821 | 0.746 | 0.994 [0.991, 0.996] | 0.004 / 0.003 | 0.18 / 0.18 / 0.22 |
| AV2 | cheap_wrong_F_correct | 1883 | 0.159 | 0.909 [0.887, 0.930] | 0.028 / 0.063 | 2.20 / 0.34 / 0.60 |
| AV2 | both_wrong | 842 | 0.071 | 0.162 [0.123, 0.203] | 0.319 / 0.519 | 4.09 / 4.12 / 3.41 |
| AV2 | cheap_correct_F_wrong | 272 | 0.023 | 0.467 [0.372, 0.554] | 0.375 / 0.158 | 0.37 / 3.19 / 2.20 |
| Argoverse-HD | both_correct | 8989 | 0.599 | 0.995 [0.990, 0.998] | 0.004 / 0.001 | 0.11 / 0.11 / 0.12 |
| Argoverse-HD | cheap_wrong_F_correct | 3148 | 0.210 | 0.870 [0.806, 0.915] | 0.024 / 0.105 | 2.56 / 0.29 / 0.64 |
| Argoverse-HD | both_wrong | 2272 | 0.151 | 0.111 [0.057, 0.193] | 0.198 / 0.691 | 3.02 / 2.68 / 2.54 |
| Argoverse-HD | cheap_correct_F_wrong | 597 | 0.040 | 0.549 [0.260, 0.777] | 0.365 / 0.085 | 0.24 / 1.92 / 0.88 |

## Ceiling (UNREALIZABLE)

The offline per-frame best-of choice uses the annotated cost of each frame: no deployable system can make it. It is
shown next to the existing K28 oracle frame router (true V, top-k, at v6-S's px, + components; routers_v6_<ds>.csv).

| dataset | v6-S G | best-of{cheap, v6-S, u960} | best-of{cheap, u960} | best-of{cheap, v6-S, native} | best-of{cheap, native} | K28 oracle frame router @ v6-S px + comp. (G; v6-S − oracle) |
|---|---|---|---|---|---|---|
| nuScenes | 0.211 [0.104, 0.348] | 0.408 [0.315, 0.483] | 0.369 [0.292, 0.455] | 0.408 [0.310, 0.491] | 0.376 [0.286, 0.459] | 0.349 at px 0.078; -0.138 [-0.211, -0.057] |
| AV2 | 0.294 [0.232, 0.356] | 0.478 [0.422, 0.532] | 0.461 [0.408, 0.516] | 0.485 [0.437, 0.537] | 0.436 [0.385, 0.491] | 0.396 at px 0.054; -0.102 [-0.147, -0.062] |
| Argoverse-HD | 0.418 [0.314, 0.517] | 0.491 [0.391, 0.586] | 0.453 [0.358, 0.548] | 0.549 [0.450, 0.641] | 0.526 [0.429, 0.614] | 0.356 at px 0.070; +0.062 [+0.014, +0.112] |

**Reading.**
- Recoverable frames are frequent. Among required frames, cheap is wrong and u960 + C1 + v3a right on 17% / 21% / 34%
  (nuScenes / AV2 / Argoverse-HD). v6-S gets 74% / 92% / 95% of them right; its misses there are almost all
  under-braking.
- Negative escalation exists: on 6–8% of required frames cheap is right and u960 wrong. v6-S is right on only 24% / 28% /
  17% of those, so on those frames it mostly follows the verified (fine) estimate.
- Both wrong: 30% / 16% / 30% of required frames. v6-S is right on 4–8% of them.
- The best-of{cheap, v6-S, F} ceiling is 0.41 / 0.48 / 0.49 (u960) and 0.41 / 0.49 / 0.55 (native). This bounds any
  per-frame selection among these three outputs. It is above the K28 oracle router at v6-S's px (0.349 / 0.396 / 0.356)
  and above v6-S (0.211 / 0.294 / 0.418).

## Files
k2_classes_<ds>.csv (every class × population × F, with all columns above) and k2_ceiling_<ds>.csv (G with intervals
for v6-S, F and both ceilings, and the paired ceiling − v6-S difference).
