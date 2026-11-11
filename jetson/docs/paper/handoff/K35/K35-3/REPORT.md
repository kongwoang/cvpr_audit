# K35-3 — frame-level temporal escalation baselines (POST-HOC, three test splits, run once)

**Post-hoc analysis**, run once (2026-10-25); no selection of any kind.
- Code:
  - remote/A_method/k35_k3_replay.py (md5 706eb68b): the no-history replay on VUW1;
  - remote/A_method/k35_k3.py (md5 fd8d084a): the schedules and statistics.
- Rows, at v6-S's mean network px per dataset:
  - (i) **periodic**: every k-th frame u960 + C1 + v3a, cheap otherwise; phase 0 registered, every phase reported;
  - (ii) **random**: each frame escalated with p = 1/k; seeds 0–4, each seed and the mean reported.
- **k from the pixel match:** k = round((px_u960 − px_cheap) / (px_v6S − px_cheap)), which gives k = 10 / 7 / 6
  (nuScenes / AV2 / Argoverse-HD). An escalated frame runs only the whole frame; an unescalated one, only the cheap
  draft. Realized px is in the tables.
- **C1 history as in deployment (registered):** an unescalated frame gives no fine observation. So an escalated frame
  has a C1 chain only if its two previous frames were escalated; then its inputs equal the continuous-history row, and
  the stored action is used. Otherwise C1 has no TTC.
  - This is computed by a CPU replay: the stored u960 boxes, TTC infinite, the frozen refiner v3a on the deployed
    support, the deployed thresholds; K34's replay is otherwise unchanged.
  - Parity: on every frame where the stored C1 TTC was already infinite for all boxes, the replay equals the stored
    action: 3936 / 3936 frames.
  - For periodic k ≥ 3, no escalated frame ever has a chain.
- **Sensitivity row (labelled, not registered):** continuous fine history on every escalated frame, the K28 / K33
  router convention. This is optimistic for an escalation policy.
- **G_cap: not meaningful.** Every escalated frame exceeds v6-S's per-frame pixel budget, so a per-frame cap removes every
  escalation and gives G_cap = 0. It is reported as NA.
- **Statistics:** costs are K28's (the jerk term uses cheap's previous action). Missed rate and false episodes per 1,000
  frames follow K31. Paired unit bootstrap of v6-S − row, 2,000 draws, seed 0. **Bold** = the interval excludes 0.
  Sign: v6-S − row > 0 means v6-S is higher, which is better for G and worse for missed / false.

## Registered rows (deployment history rule)

| dataset | row | px | G | v6-S − row G | missed rate | v6-S − row missed | false / 1,000 | v6-S − row false / 1,000 |
|---|---|---|---|---|---|---|---|---|
| nuScenes | v6-S (reference) | 0.078 | 0.211 | – | 0.408 | – | 4.0 | – |
| nuScenes | periodic k=10 phase 0 (registered) | 0.079 | -0.022 | **+0.233 [+0.119, +0.349]** | 0.354 | +0.054 [-0.012, +0.122] | 21.7 | **-17.6 [-20.8, -14.4]** |
| nuScenes | random rate 1/10 mean of 5 seeds | 0.076 | -0.002 | +0.213 (mean of seed intervals [+0.120, +0.328]) | 0.344 | +0.064 | 22.0 | -18.0 |
| AV2 | v6-S (reference) | 0.054 | 0.294 | – | 0.189 | – | 6.5 | – |
| AV2 | periodic k=7 phase 0 (registered) | 0.057 | 0.018 | **+0.276 [+0.218, +0.337]** | 0.137 | **+0.053 [+0.006, +0.101]** | 34.6 | **-28.1 [-32.9, -23.5]** |
| AV2 | random rate 1/7 mean of 5 seeds | 0.055 | 0.011 | +0.282 (mean of seed intervals [+0.226, +0.340]) | 0.122 | +0.067 | 34.2 | -27.7 |
| Argoverse-HD | v6-S (reference) | 0.070 | 0.418 | – | 0.380 | – | 6.3 | – |
| Argoverse-HD | periodic k=6 phase 0 (registered) | 0.068 | 0.065 | **+0.352 [+0.265, +0.435]** | 0.197 | **+0.183 [+0.052, +0.319]** | 27.8 | **-21.5 [-30.0, -13.9]** |
| Argoverse-HD | random rate 1/6 mean of 5 seeds | 0.068 | 0.064 | +0.353 (mean of seed intervals [+0.267, +0.435]) | 0.166 | +0.214 | 27.7 | -21.5 |

The random-mean row averages the five seeds' point estimates and interval bounds; each seed's own interval is in
k3_rows_<ds>.csv.

## Phase / seed sensitivity and the continuous-history sensitivity row

| dataset | family | history rule | G min / max over phases or seeds | v6-S − row G: smallest lower bound | missed min / max | false / 1,000 min / max |
|---|---|---|---|---|---|---|
| nuScenes | periodic (10 phases) | deployment | -0.035 / 0.057 | +0.033 | 0.327 / 0.361 | 21.2 / 22.5 |
| nuScenes | periodic (10 phases) | continuous_sensitivity | -0.031 / 0.060 | +0.029 | 0.327 / 0.354 | 21.7 / 23.1 |
| nuScenes | random (5 seeds) | deployment | -0.024 / 0.024 | +0.101 | 0.327 / 0.354 | 21.2 / 22.4 |
| nuScenes | random (5 seeds) | continuous_sensitivity | -0.021 / 0.024 | +0.092 | 0.320 / 0.347 | 21.7 / 22.7 |
| AV2 | periodic (7 phases) | deployment | -0.012 / 0.027 | +0.211 | 0.105 / 0.147 | 33.8 / 34.7 |
| AV2 | periodic (7 phases) | continuous_sensitivity | 0.039 / 0.056 | +0.180 | 0.100 / 0.137 | 33.6 / 34.9 |
| AV2 | random (5 seeds) | deployment | -0.004 / 0.025 | +0.212 | 0.116 / 0.132 | 33.7 / 34.5 |
| AV2 | random (5 seeds) | continuous_sensitivity | 0.043 / 0.066 | +0.167 | 0.100 / 0.121 | 33.4 / 34.5 |
| Argoverse-HD | periodic (6 phases) | deployment | 0.062 / 0.067 | +0.263 | 0.155 / 0.197 | 27.7 / 28.2 |
| Argoverse-HD | periodic (6 phases) | continuous_sensitivity | 0.047 / 0.052 | +0.271 | 0.155 / 0.183 | 27.5 / 28.7 |
| Argoverse-HD | random (5 seeds) | deployment | 0.062 / 0.067 | +0.266 | 0.155 / 0.183 | 27.1 / 28.5 |
| Argoverse-HD | random (5 seeds) | continuous_sensitivity | 0.046 / 0.052 | +0.272 | 0.141 / 0.183 | 26.9 / 28.2 |

**Reading.**
- At v6-S's pixels, fixed-rate escalation is far below v6-S on all three splits, for every phase and every seed:
  - periodic phase 0: +0.233 / +0.276 / +0.352 G in v6-S's favour, all significant;
  - the smallest lower bound across phases and seeds stays > 0 on every split.
- Its missed rate is lower than v6-S's (nuScenes n.s.; AV2 and Argoverse-HD significant), but its false episodes are
  about 5× v6-S's (21–35 vs 4.0–6.5 per 1,000, all significant). With 85–90% unescalated frames, it inherits the cheap
  draft's behaviour.
- Granting the escalated frames a continuous fine history (sensitivity row) changes G by at most a few hundredths and
  does not change any conclusion.

## Files
k3_rows_<ds>.csv (every phase, seed, history rule and metric, with intervals; escalated share; escalated frames with a C1
chain) and k3_meta_<ds>.csv (k, px, replay parity, v6-S reference values).
