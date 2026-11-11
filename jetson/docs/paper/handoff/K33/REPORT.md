# K33 REPORT — Table 1 v7: every baseline with our detector and our components

**Status: FINAL for the tables, 2026-10-20 (VUW 23:40).** Pre-registration: table1_design.md §1–§2 and the amendments in
STATUS.md. Every row was selected on dev by the registered rule and evaluated once on test.

- **Principle.** Every row runs the same COCO YOLOv8s: the cheap draft at 320, and the target pass at the scale the method
  chooses. Every row also gets v6-S's components:
  - C1 looming on the row's own same-fidelity history;
  - refiner v3a on the braking support;
  - for crop methods, the truncation guard and K28's merge.
  Differences between rows therefore come from where, and at what scale, the pixels are spent. Official-form rows (no
  components) are in the supplement CSVs.
- **Statistics.** Decision value G = (ΣJ_cheap − ΣJ) / ΣJ_cheap. Paired unit bootstrap of v6-S − row: 2,000 resamples,
  seed 0; units are scenes, logs or sequences. Sign: > 0 means v6-S is better. **Bold** = the 95% interval excludes 0.
- **px.** Target-detector tensor px / native px, plus the cheap draft for rows that use it (amendment 1).
- **@cap.** The row under v6-S's per-frame pixel cap (K30 V1a, m = 1). Crop methods take crops in their own order until
  the cap. Whole-frame and warp methods take the largest scale or canvas within the cap, else cheap only. SAHI scales its
  slices down; UFP shrinks the mosaic side.
- **Spurious / 1k (K30 V7).** Output boxes with estimated braking demand ≥ thr_decel that match no GT box (IoU ≥ 0.3),
  per 1,000 frames.
  - Computed by a re-run of the registered rows (--tag spur), because the first run did not store it.
  - k33_report.py asserts that every cost, action and px of that re-run equals the first run's. The assertion passed on
    all three splits.
- **Jetson.** AGX Xavier, MAXN, trt32 fast path; K29 frame sets (nuScenes test 20 scenes, AV2 8 logs); 3 repetitions.
  - v6-S's latency comes from the same runs.
  - The @cap rows of A / B are not timed on the device; their px is at most that of the uncapped row.
- **Datasets.** nuScenes new test, AV2 val (ego mask), Argoverse-HD val. Argoverse-HD uses AV2's selections, thresholds
  and policies.

## 1. Summary for the paper

1. **At v6-S's pixel budget (@cap), v6-S is at least as good as every baseline on all three datasets.** No capped row is
   significantly above it. Of the 13 A–E rows × 3 splits, v6-S is significantly better in 32 of 39 cases. The 7 n.s.
   ones are:
   - AdaScale on nuScenes and AV2;
   - UFP on nuScenes and AV2;
   - FOVEA KDE_S on AV2;
   - AnA (both variants) on AV2;
   - none on Argoverse-HD.
2. **Argoverse-HD:** v6-S (G 0.418 at px 0.070) beats every row significantly, even uncapped. This includes u1280
   (+0.103), SAHI at 22× the pixels (+0.189) and AnA (+0.103).
3. **nuScenes:** no row is significantly above v6-S (0.211 at 0.078).
   - SAHI (0.303 at 24× px), u1280 (0.292 at 8.4× px) and the decision-value router (0.214 at 4.7× px) are higher or
     level, all n.s.
4. **AV2 is the exception.** Two rows are significantly above v6-S (0.294 at 0.054):
   - u1280 + comp. (0.348 at 7.4× px; −0.055 [−0.113, −0.005]);
   - AnA iterated (0.349 at 1.46× px; −0.055 [−0.113, −0.002]).
   Also above v6-S but n.s.: AnA one pass (0.346), AdaScale (0.341), u960 (0.338), UFP (0.322), FOVEA (0.319 / 0.316)
   and u640 / u768 (0.311 / 0.308). Under the cap, AnA is n.s. below v6-S (+0.014). Mechanism in §3.
5. **Spurious braking** (per 1,000 frames, nuScenes / AV2 / Argoverse-HD): v6-S 4.2 / 5.6 / 15.0; cheap 18.0 / 16.3 /
   27.7; whole-frame rows 6.0–7.4 / 7.4–9.6 / 41–44.
   - On nuScenes, v6-S has the fewest of all rows with G > 0.
   - On AV2, three rows with G > 0 have fewer: AnA t 0.25 (4.5), counterfactual one pass (4.9) and iterated (4.6). Their
     G is 0.245–0.304, below v6-S's 0.294 except AnA t 0.25 (0.304, n.s.).
   - On Argoverse-HD, the counterfactual one pass has fewer (13.4), at G 0.307 vs 0.418.
6. **F, fine-tuned regime (AV2).** v6-S-ft (0.224) is the only fine-tuned row with G > 0. It beats every other F row
   significantly: u960-ft by +0.287, ESOD by +0.344, ZoomDet by +0.259, FOVEA-ft by +0.304, LZU-ft by +0.305 and UFP-ft
   by +0.233 (all with components).
7. **Jetson latency** (mean ms, nuScenes / AV2): v6-S 21.5 / 39.8.
   - Of the rows at or above v6-S's G: AnA iterated 25.4 / 68.3; AdaScale 43.4 / 70.3; SAHI 84.8 / 153.2.
   - The decision-value router 25.1 / 35.3; Chanakya 14.5 / 23.5, at G 0.127 / 0.286.

## 2. Table 1 v7 (test; every row + C1 + refiner v3a; selected on dev)

#### nuScenes new test (228 scenes, 9,181 frames)

| row | G | px | v6-S − row | G @cap | px @cap | v6-S − row @cap | spurious / 1k | Jetson mean / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| cheap draft | 0.000 | 0.043 | **+0.211 [+0.104, +0.348]** | – | – | – | 18.0 | 7.7 / 8.4 |
| **v6-S (ours)** | 0.211 | 0.078 | – | – | – | – | 4.2 | 21.5 / 57.4 |
| u640 + comp. | -0.036 | 0.171 | +0.246 [-0.070, +0.666] | – | – | – | 7.0 | – |
| u768 + comp. | 0.046 | 0.239 | +0.165 [-0.061, +0.459] | – | – | – | 7.0 | – |
| u960 + comp. | 0.112 | 0.384 | +0.099 [-0.078, +0.308] | – | – | – | 7.4 | – |
| u1280 + comp. | 0.292 | 0.654 | -0.082 [-0.165, +0.017] | – | – | – | 6.0 | – |
| A AdaScale (ada+1t1280) | 0.144 | 0.306 | +0.067 [-0.089, +0.228] | 0.195 | 0.055 | +0.016 [-0.065, +0.112] | 7.6 | 43.4 / 54.5 |
| A SAHI (640, ov 0.1, s 1.0) | 0.303 | 1.877 | -0.092 [-0.207, +0.029] | 0.025 | 0.064 | **+0.185 [+0.090, +0.289]** | 13.0 | 84.8 / 96.0 |
| A FOVEA KDE_I (1280, bw 64, amp 2) | 0.150 | 0.697 | +0.061 [-0.049, +0.179] | -0.366 | 0.075 | **+0.576 [+0.148, +1.020]** | 5.3 | 42.2 / 51.7 |
| A FOVEA KDE_S (1280, bw 64, amp 1) | 0.161 | 0.654 | +0.049 [-0.061, +0.197] | -0.548 | 0.071 | **+0.759 [+0.206, +1.374]** | 6.0 | 33.4 / 42.0 |
| A BAR router (1280, 50%) | 0.169 | 0.366 | +0.042 [-0.032, +0.125] | 0.005 | 0.046 | **+0.206 [+0.096, +0.345]** | 10.7 | 49.3 / 70.3 |
| A decision-value router (1280, 50%) | 0.214 | 0.364 | -0.003 [-0.088, +0.095] | 0.010 | 0.046 | **+0.200 [+0.094, +0.339]** | 9.0 | 25.1 / 46.9 |
| B UFP (scale 2.0, side 512) | -0.058 | 0.181 | +0.269 [-0.030, +0.670] | 0.128 | 0.072 | +0.083 [-0.065, +0.243] | 5.6 | 42.7 / 71.9 |
| C Chanakya (h128, B, csc) | 0.127 | 0.079 | +0.084 [-0.032, +0.207] | 0.039 | 0.046 | **+0.172 [+0.064, +0.316]** | 7.8 | 14.5 / 33.1 |
| C Chanakya dev-best, px unmatched (ref.) | 0.165 | 0.092 | +0.046 [-0.009, +0.095] | 0.030 | 0.045 | **+0.181 [+0.071, +0.324]** | 10.2 | – |
| D AnA one pass (t 0.05, T_h ∞) | 0.158 | 0.100 | +0.053 [-0.074, +0.181] | 0.092 | 0.057 | **+0.118 [+0.003, +0.273]** | 5.7 | 23.7 / 57.8 |
| D AnA iterated (t 0.05, T_h ∞) | 0.158 | 0.105 | +0.052 [-0.074, +0.181] | 0.095 | 0.058 | **+0.116 [+0.001, +0.270]** | 5.8 | 25.4 / 66.9 |
| D AnA t 0.25 (reported in any case) | -0.037 | 0.081 | +0.248 [-0.014, +0.470] | -0.065 | 0.052 | **+0.276 [+0.039, +0.485]** | 3.8 | 18.8 / 48.4 |
| E counterfactual one pass (40%, 2 s) | -0.091 | 0.053 | **+0.302 [+0.054, +0.527]** | -0.086 | 0.051 | **+0.297 [+0.049, +0.524]** | 3.2 | 14.6 / 36.9 |
| E counterfactual iterated (40%, 2 s) | -0.093 | 0.055 | **+0.304 [+0.056, +0.530]** | -0.089 | 0.052 | **+0.299 [+0.051, +0.526]** | 3.3 | 16.3 / 48.1 |

#### AV2 val, ego mask (150 logs, 11,818 frames)

| row | G | px | v6-S − row | G @cap | px @cap | v6-S − row @cap | spurious / 1k | Jetson mean / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| cheap draft | 0.000 | 0.026 | **+0.294 [+0.232, +0.356]** | – | – | – | 16.3 | 10.5 / 11.0 |
| **v6-S (ours)** | 0.294 | 0.054 | – | – | – | – | 5.6 | 39.8 / 89.1 |
| u640 + comp. | 0.311 | 0.103 | -0.017 [-0.076, +0.035] | – | – | – | 7.4 | – |
| u768 + comp. | 0.308 | 0.147 | -0.015 [-0.080, +0.043] | – | – | – | 7.5 | – |
| u960 + comp. | 0.338 | 0.232 | -0.045 [-0.102, +0.004] | – | – | – | 8.7 | – |
| u1280 + comp. | 0.348 | 0.400 | **-0.055 [-0.113, -0.005]** | – | – | – | 9.6 | – |
| A AdaScale (ada+1t1280) | 0.341 | 0.268 | -0.048 [-0.107, +0.002] | 0.263 | 0.037 | +0.031 [-0.040, +0.099] | 8.3 | 70.3 / 86.3 |
| A SAHI (640, ov 0.1, s 1.0) | 0.256 | 1.652 | +0.037 [-0.031, +0.105] | -0.036 | 0.042 | **+0.329 [+0.246, +0.424]** | 59.2 | 153.2 / 172.8 |
| A FOVEA KDE_I (1280, bw 64, amp 2) | 0.319 | 0.426 | -0.026 [-0.086, +0.027] | 0.186 | 0.051 | **+0.108 [+0.026, +0.189]** | 7.9 | 59.8 / 75.8 |
| A FOVEA KDE_S (1280, bw 64, amp 1) | 0.316 | 0.400 | -0.023 [-0.074, +0.023] | 0.276 | 0.049 | +0.017 [-0.047, +0.071] | 6.4 | 49.1 / 65.8 |
| A BAR router (1280, 50%) | 0.155 | 0.214 | **+0.138 [+0.078, +0.201]** | 0.016 | 0.029 | **+0.277 [+0.214, +0.344]** | 12.0 | 76.1 / 106.9 |
| A decision-value router (1280, 50%) | 0.189 | 0.215 | **+0.104 [+0.053, +0.157]** | 0.023 | 0.029 | **+0.271 [+0.216, +0.328]** | 11.8 | 35.3 / 65.0 |
| B UFP (scale 2.0, side 512) | 0.322 | 0.090 | -0.028 [-0.089, +0.026] | 0.252 | 0.048 | +0.041 [-0.038, +0.119] | 12.8 | 70.9 / 107.8 |
| C Chanakya (h128, B, csc) | 0.286 | 0.053 | +0.007 [-0.039, +0.051] | 0.044 | 0.029 | **+0.250 [+0.193, +0.309]** | 9.5 | 23.5 / 53.8 |
| C Chanakya dev-best, px unmatched (ref.) | 0.272 | 0.052 | +0.022 [-0.028, +0.065] | 0.041 | 0.029 | **+0.253 [+0.198, +0.311]** | 9.6 | – |
| D AnA one pass (t 0.05, T_h ∞) | 0.346 | 0.074 | -0.052 [-0.111, +0.002] | 0.278 | 0.047 | +0.016 [-0.011, +0.047] | 6.9 | 59.6 / 138.5 |
| D AnA iterated (t 0.05, T_h ∞) | 0.349 | 0.079 | **-0.055 [-0.113, -0.002]** | 0.280 | 0.050 | +0.014 [-0.013, +0.045] | 7.3 | 68.3 / 159.6 |
| D AnA t 0.25 (reported in any case) | 0.304 | 0.064 | -0.011 [-0.081, +0.063] | 0.238 | 0.043 | **+0.055 [+0.014, +0.107]** | 4.5 | 51.6 / 115.9 |
| E counterfactual one pass (40%, 2 s) | 0.245 | 0.043 | **+0.049 [+0.011, +0.098]** | 0.240 | 0.039 | **+0.053 [+0.016, +0.101]** | 4.9 | 36.4 / 81.1 |
| E counterfactual iterated (40%, 2 s) | 0.256 | 0.046 | +0.038 [-0.005, +0.090] | 0.242 | 0.042 | **+0.051 [+0.014, +0.102]** | 4.6 | 47.8 / 109.3 |

#### Argoverse-HD val (24 sequences, 15,006 frames)

| row | G | px | v6-S − row | G @cap | px @cap | v6-S − row @cap | spurious / 1k | Jetson mean / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| cheap draft | 0.000 | 0.031 | **+0.418 [+0.314, +0.517]** | – | – | – | 27.7 | – |
| **v6-S (ours)** | 0.418 | 0.070 | – | – | – | – | 15.0 | – |
| u640 + comp. | 0.279 | 0.116 | **+0.139 [+0.026, +0.329]** | – | – | – | 42.2 | – |
| u768 + comp. | 0.273 | 0.160 | **+0.145 [+0.032, +0.336]** | – | – | – | 43.8 | – |
| u960 + comp. | 0.297 | 0.253 | **+0.121 [+0.017, +0.320]** | – | – | – | 44.3 | – |
| u1280 + comp. | 0.315 | 0.444 | **+0.103 [+0.013, +0.278]** | – | – | – | 41.5 | – |
| A AdaScale (ada+1t1280) | 0.298 | 0.299 | **+0.120 [+0.017, +0.319]** | 0.210 | 0.051 | **+0.208 [+0.050, +0.414]** | 46.0 | – |
| A SAHI (640, ov 0.1, s 1.0) | 0.229 | 1.538 | **+0.189 [+0.062, +0.388]** | 0.135 | 0.055 | **+0.283 [+0.166, +0.430]** | 58.8 | – |
| A FOVEA KDE_I (1280, bw 64, amp 2) | 0.313 | 0.476 | **+0.105 [+0.016, +0.278]** | 0.021 | 0.068 | **+0.397 [+0.202, +0.602]** | 35.7 | – |
| A FOVEA KDE_S (1280, bw 64, amp 1) | 0.312 | 0.444 | **+0.106 [+0.019, +0.276]** | -0.025 | 0.065 | **+0.443 [+0.222, +0.677]** | 35.6 | – |
| A BAR router (1280, 50%) | 0.207 | 0.231 | **+0.211 [+0.124, +0.297]** | 0.009 | 0.035 | **+0.409 [+0.304, +0.508]** | 21.9 | – |
| A decision-value router (1280, 50%) | 0.011 | 0.049 | **+0.406 [+0.306, +0.505]** | 0.001 | 0.032 | **+0.417 [+0.313, +0.516]** | 27.0 | – |
| B UFP (scale 2.0, side 512) | 0.282 | 0.128 | **+0.136 [+0.021, +0.328]** | 0.267 | 0.063 | **+0.151 [+0.032, +0.346]** | 47.0 | – |
| C Chanakya (h128, B, csc) | 0.297 | 0.075 | **+0.121 [+0.025, +0.237]** | 0.022 | 0.038 | **+0.396 [+0.290, +0.497]** | 27.7 | – |
| C Chanakya dev-best, px unmatched (ref.) | 0.310 | 0.072 | **+0.108 [+0.020, +0.209]** | 0.027 | 0.038 | **+0.390 [+0.285, +0.490]** | 24.8 | – |
| D AnA one pass (t 0.05, T_h ∞) | 0.314 | 0.178 | **+0.104 [+0.007, +0.297]** | 0.397 | 0.069 | **+0.021 [+0.008, +0.034]** | 40.1 | – |
| D AnA iterated (t 0.05, T_h ∞) | 0.315 | 0.206 | **+0.103 [+0.006, +0.296]** | 0.397 | 0.074 | **+0.021 [+0.008, +0.034]** | 41.4 | – |
| D AnA t 0.25 (reported in any case) | 0.225 | 0.155 | **+0.193 [+0.036, +0.396]** | 0.311 | 0.062 | **+0.107 [+0.020, +0.232]** | 35.5 | – |
| E counterfactual one pass (40%, 2 s) | 0.307 | 0.058 | **+0.111 [+0.019, +0.233]** | 0.321 | 0.054 | **+0.097 [+0.008, +0.222]** | 13.4 | – |
| E counterfactual iterated (40%, 2 s) | 0.285 | 0.064 | **+0.133 [+0.033, +0.253]** | 0.313 | 0.059 | **+0.105 [+0.020, +0.228]** | 19.9 | – |


Selections (dev_selection.csv, dev_selection_rc.csv). Each family's official / K28 form was a dev candidate and lost to
the selected row:
- AdaScale ada+1t1280 (0.296 vs ada+0t960 0.262);
- SAHI sahi640o1s10 (0.318 vs 0.311);
- FOVEA KDE_I fovi1280b64a2 (0.297 vs 0.267);
- FOVEA KDE_S fovs1280b64a1 (0.305 vs 0.233);
- UFP ufp20s512 (0.030 vs −0.793);
- BAR bar1280_0.5; DVR dvr1280_0.5;
- Chanakya ch128B_csc: the best of the candidates whose dev px is within ±5% of v6-S's; the dev-best ch128B_reinforce
  (+17% px on nuScenes) is the reference row;
- D: ana05hinfo / ana05hinfi;
- E: cf40q20o / cf40q20i.

## 3. Rows where v6-S is not better, with the mechanism

**AV2: AnA (one pass and iterated) and u1280 / u960 / AdaScale.**
- Cost decomposition (share of ΣJ_cheap):
  - v6-S: under-braking 0.299, over-braking 0.159;
  - AnA iterated: 0.239 / 0.160;
  - u1280: 0.225 / 0.177.
  The rows above v6-S win only on under-braking; they brake no less often when not required (HARD on KEEP: v6-S 267,
  AnA 265, u1280 284 frames).
- The under-braking is not a missed object.
  - On required frames, v6-S finds the decisive object as often as AnA (96.5% vs 96.1%), and misses it on fewer
    under-braking frames (70 vs 93).
  - On the 54 frames where both found it, and v6-S under-brakes while AnA does not, both estimate the same range (median
    ratio 1.000).
  - But v6-S has a finite TTC on only 57% of them, against 94% for AnA (median demand 0.71 vs 1.61 m/s²).
- **Mechanism: history continuity of the C1 lift.**
  - AnA verifies every corridor detection in every frame, so the decisive object has an unbroken same-fidelity history,
    which C1's looming cue needs.
  - v6-S verifies an object only while it is in the braking support or the field. On AV2's 5 Hz logs the chain breaks
    more often, and C1 then has no TTC.
  - Under the pixel cap, AnA loses this advantage (+0.014 n.s.).
  - The whole-frame rows have a continuous history by construction, at 4.3–7.4× the pixels.
- On the 29 reverse frames (AnA under-brakes, v6-S right), it is the other way round: v6-S has a finite TTC on 90%,
  AnA on 72%.

**nuScenes: SAHI and u1280 (both n.s.).**
- Under-braking share: v6-S 0.495, SAHI 0.326, u1280 0.386.
- SAHI and u1280 find the decisive object on more required frames (95.5% / 94.4% vs 90.6%). v6-S misses it on 81 of
  its 349 under-braking frames, vs 33 for SAHI.
- SAHI pays for this with 2.3× v6-S's over-braking (0.135 vs 0.059) and 24× the pixels; at the cap it falls to 0.025.
- On nuScenes' 114 natural-miss frames, v6-S recovers 61% of the missed objects, vs 87% (SAHI) and 90% (u1280): small
  far objects that a whole frame at 1280 or SAHI's tiles see. Earlier mechanism: K31 item 1, longest delays.

**@cap rows near v6-S.** AdaScale (nuScenes, AV2), UFP (nuScenes, AV2) and FOVEA KDE_S (AV2) are n.s. below v6-S under
the cap. At that budget they mostly fall back to their smallest scale or canvas.

**Notes on the A rows.**
- The decision-value router's AV2 threshold does not transfer to Argoverse-HD: it routes almost no frames (px 0.049
  vs 0.031 for cheap; G 0.011).
- FOVEA's warped canvases at the cap (nuScenes) drop below cheap (G −0.37 / −0.55): warping into a canvas about 5× smaller
  produces many spurious boxes.

## 4. Missed-object regime (K28 K4 definitions)

| row | natural: recovery nuScenes / AV2 / AHD | natural: G on those frames | deletion: recovery | deletion: loss vs unmodified draft |
|---|---|---|---|---|
| cheap draft | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 4.29 / 2.17 / 1.85 |
| **v6-S (ours)** | 0.607 / 0.848 / 0.873 | 0.506 / 0.321 / 0.680 | 0.847 / 0.876 / 0.846 | 0.55 / 0.12 / 0.17 |
| u640 + comp. | 0.675 / 0.763 / 0.805 | 0.474 / 0.205 / 0.564 | 0.978 / 0.996 / 0.980 | 0.49 / -0.41 / -0.07 |
| u768 + comp. | 0.786 / 0.854 / 0.866 | 0.678 / 0.269 / 0.559 | 0.982 / 0.995 / 0.984 | 0.42 / -0.41 / -0.07 |
| u960 + comp. | 1.000 / 1.000 / 1.000 | 0.748 / 0.317 / 0.682 | 0.987 / 0.995 / 0.992 | 0.25 / -0.44 / -0.10 |
| u1280 + comp. | 0.897 / 0.965 / 0.951 | 0.744 / 0.341 / 0.683 | 0.990 / 0.996 / 0.996 | 0.03 / -0.44 / -0.10 |
| A AdaScale (ada+1t1280) | 0.752 / 0.944 / 0.962 | 0.664 / 0.302 / 0.666 | 0.982 / 0.995 / 0.994 | 0.27 / -0.44 / -0.10 |
| A SAHI (640, ov 0.1, s 1.0) | 0.872 / 0.914 / 0.956 | 0.763 / 0.280 / 0.691 | 0.994 / 0.992 / 0.983 | -0.10 / -0.40 / -0.08 |
| A FOVEA KDE_I (1280, bw 64, amp 2) | 0.769 / 0.773 / 0.882 | 0.552 / 0.341 / 0.673 | 0.954 / 0.949 / 0.967 | 0.32 / -0.38 / -0.04 |
| A FOVEA KDE_S (1280, bw 64, amp 1) | 0.752 / 0.763 / 0.796 | 0.561 / 0.314 / 0.636 | 0.966 / 0.974 / 0.968 | 0.28 / -0.41 / -0.07 |
| A BAR router (1280, 50%) | 0.521 / 0.495 / 0.516 | 0.463 / 0.172 / 0.375 | 0.629 / 0.513 / 0.550 | 2.61 / 1.17 / 1.00 |
| A decision-value router (1280, 50%) | 0.641 / 0.581 / 0.054 | 0.708 / 0.169 / 0.035 | 0.606 / 0.724 / 0.097 | 1.16 / 0.25 / 1.80 |
| B UFP (scale 2.0, side 512) | 0.556 / 0.783 / 0.738 | 0.290 / 0.254 / 0.472 | 0.968 / 0.996 / 0.988 | 0.52 / -0.41 / -0.11 |
| C Chanakya (h128, B, csc) | 0.342 / 0.439 / 0.321 | 0.504 / 0.124 / 0.173 | 0.428 / 0.303 / 0.396 | 1.71 / 1.70 / 1.54 |
| C Chanakya dev-best, px unmatched (ref.) | 0.419 / 0.356 / 0.307 | 0.429 / 0.139 / 0.151 | 0.416 / 0.236 / 0.379 | 1.76 / 2.01 / 1.57 |
| D AnA one pass (t 0.05, T_h ∞) | 0.718 / 0.801 / 0.866 | 0.508 / 0.270 / 0.624 | 0.447 / 0.534 / 0.558 | 3.25 / 1.37 / 1.22 |
| D AnA iterated (t 0.05, T_h ∞) | 0.718 / 0.808 / 0.865 | 0.508 / 0.295 / 0.626 | 0.451 / 0.559 / 0.574 | 3.21 / 1.30 / 1.08 |
| D AnA t 0.25 (reported in any case) | 0.256 / 0.452 / 0.227 | -0.121 / -0.000 / -0.002 | 0.217 / 0.276 / 0.274 | 4.42 / 2.13 / 1.86 |
| E counterfactual one pass (40%, 2 s) | 0.205 / 0.419 / 0.202 | -0.123 / -0.013 / -0.007 | 0.191 / 0.237 / 0.251 | 4.48 / 2.21 / 1.90 |
| E counterfactual iterated (40%, 2 s) | 0.214 / 0.427 / 0.206 | -0.124 / -0.007 / -0.005 | 0.197 / 0.251 / 0.268 | 4.46 / 2.15 / 1.79 |

natural missed frames / objects: {'frames': {'ahd': 1019, 'av2': 360, 'nusc': 114}, 'missed_objects': {'ahd': 1027, 'av2': 396, 'nusc': 117}}

- Natural misses (the draft misses a decision-relevant object that u960 + comp. finds): 114 / 360 / 1,019 frames
  (117 / 396 / 1,027 objects). u960 is 1.000 by definition.
- Deletion (del_all, every matched draft detection of a relevant object removed): v6-S recovers 85–88% and keeps the
  loss vs the unmodified draft at +0.55 / +0.12 / +0.17, against +4.29 / +2.17 / +1.85 for cheap.
  - The whole-frame, SAHI, FOVEA and UFP rows do not use the draft's boxes, so deletion barely affects them.
  - Draft-driven rows (AnA, counterfactual, routers, Chanakya) lose most of their gain.
  - The deletion strata by range are not reported: the deleted objects' ranges were not stored. Only all and the
    ego-speed terciles are in missed_deletion.csv.

## 5. F: fine-tuned regime (AV2 val, ego mask, one seed)

| row | refiner | G | px | HARD on KEEP | v6-S-ft − row |
|---|---|---|---|---|---|
| esod_off | none | -0.736 | – | 0.192 | **+0.960 [+0.780, +1.150]** |
| esod | none | -0.175 | – | 0.049 | **+0.399 [+0.205, +0.588]** |
| esod | v3a | -0.120 | – | 0.030 | **+0.344 [+0.141, +0.540]** |
| foveaft_off | none | -0.503 | 0.248 | 0.230 | **+0.727 [+0.576, +0.913]** |
| foveaft | none | -0.126 | 0.248 | 0.143 | **+0.350 [+0.247, +0.474]** |
| foveaft | v3a | -0.080 | 0.248 | 0.138 | **+0.304 [+0.195, +0.431]** |
| lzuft_off | none | -0.526 | 0.223 | 0.222 | **+0.750 [+0.614, +0.906]** |
| lzuft | none | -0.125 | 0.223 | 0.126 | **+0.349 [+0.252, +0.453]** |
| lzuft | v3a | -0.080 | 0.223 | 0.120 | **+0.305 [+0.216, +0.406]** |
| u960ft_off | none | -0.473 | 0.232 | 0.223 | **+0.697 [+0.550, +0.874]** |
| u960ft | none | -0.126 | 0.232 | 0.143 | **+0.350 [+0.250, +0.468]** |
| u960ft | v3a | -0.063 | 0.232 | 0.136 | **+0.287 [+0.181, +0.408]** |
| ufpft | none | -0.087 | 0.216 | 0.098 | **+0.311 [+0.223, +0.409]** |
| ufpft | v3a | -0.009 | 0.216 | 0.081 | **+0.233 [+0.156, +0.321]** |
| v6Sft | none | 0.144 | 0.070 | 0.067 | **+0.080 [+0.040, +0.124]** |
| v6Sft | v3a | 0.224 | 0.070 | 0.050 | +0.000 [+0.000, +0.000] |
| zoomdet_off | none | -0.384 | – | 0.202 | **+0.609 [+0.475, +0.768]** |
| zoomdet | none | -0.084 | – | 0.137 | **+0.309 [+0.198, +0.434]** |
| zoomdet | v3a | -0.035 | – | 0.127 | **+0.259 [+0.157, +0.376]** |
| cheap | - | 0.000 | 0.026 | 0.087 | – |

- Every fine-tuned row with components is below the cheap draft; v6-S-ft is the only one above (0.224).
- "_off" = the method's official form (DEEP mono lift, no refiner).
- **Mechanism** (ft_mechanism.csv): the fine-tuned detectors learned the benchmark's projected-cuboid boxes, which are
  larger than COCO's tight boxes.
  - The decisive object's estimated range relative to COCO u960 + comp. on the same frame: median 0.963–0.974 for the
    fine-tuned rows (u960-ft, FOVEA-ft, LZU-ft, UFP-ft, ESOD); 13–15% of them are > 10% shorter (ESOD 8%).
  - The short ranges cause over-braking: HARD on KEEP is 8.1–13.8% for u960-ft, FOVEA-ft, LZU-ft, UFP-ft and ZoomDet with
    components, against 5.0% for v6-S-ft and 8.7% for cheap.
    - ESOD with components over-brakes less (3.0%). It loses through the decisive object instead: found on 90.5% of
      frames, vs 96.8% for u960-ft.
  - v6-S-ft is affected least (median 0.986, 7.4% > 10% shorter). It looks at high resolution only where the decision
    needs it, and keeps the COCO draft's geometry elsewhere.
- The change from the K28-era F numbers (u960-ft +0.40) is entirely the masked denominator (STATUS, VUW 07:15).

## 6. Jetson latency (k33j_latency.csv)

- 3 repetitions each; the SD of the run means is ≤ 1.8 ms. Values are in the table of §2: mean / p95 per frame, including
  the cheap draft.
- **Device parity of the C rows.** BAR uses the estimator's PIL transform: max |score − VUW| 0.33 on the smoke frames.
  DVR scores equal VUW's on 68% of frames; the rest differ through the draft (VUW's nuScenes draft is the cached torch
  one).

## 7. Deviations and amendments (all in STATUS.md)

- **Amendment 1** (before any A / C run):
  - px convention;
  - SAHI keeps its own merge;
  - routers and Chanakya use K28's continuous-history convention;
  - cond_V retrained.
- **Chanakya's ±5% px-match rule** was fixed before any Chanakya test number. The first test-stage run used the wrong
  selection file; it was re-run, and its output was never read.
- **Spurious braking** was not stored by the first test runs. The registered rows were re-run once with the count added;
  the costs are identical.
- **Jetson BAR preprocessing:** a cv2 resize was replaced by the estimator's PIL transform before the C-row timing.
- **Pub @cap rows** are not timed on the device.
- **Deletion strata by range** are not available.

## 8. Files

- **Tables:**
  - table1_v7.csv (every row and its @cap form; px, deltas, spurious, Jetson);
  - missed_natural.csv, missed_deletion.csv;
  - ft_rows.csv, ft_mechanism.csv;
  - k33j_latency.csv;
  - dev_candidates.csv, dev_selection.csv, dev_candidates_rc.csv, dev_selection_rc.csv.
- **Figure:** fig_table1_v7.pdf / .png (G vs px per split; hollow = @cap).
- **Code:**
  - VUW: scripts/A_method/k33_eval.py, k33_rc.py, k33_report.py, k33_select.py, k33_spur_chain.sh;
  - Jetson: scripts/69_k33_jetson.py, k33_latency.py, k33_fig.py;
  - shared: src/vtf/k33sel.py, k33pub.py.
