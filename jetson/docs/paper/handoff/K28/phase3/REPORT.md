# K28 phase 3: post-hoc analyses of the frozen v6 (POST-HOC)

Everything here is **post-hoc**. v6 was frozen in method_freeze_draft §2n and tested once in phase 2 (phase2/REPORT.md). Phase 3 runs analyses and extra rows on the same test units. It changes nothing in v6 and selects nothing.

**Configuration.** v6-S unless a table says otherwise; v6-M / v6-L wherever the table has a budget axis.

**Splits.**
- nuScenes new test: 228 scenes, 9,181 frames.
- AV2 val with the ego mask: 150 logs, 11,818 frames.
- Argoverse-HD val: 24 sequences, 15,006 frames with 3D labels. The AP table uses every frame (15,062).

**Statistics.** Intervals are paired unit bootstraps (2,000 resamples, seed 0). The verdict rule is K24's:
- **outperforms:** lower bound > 0;
- **matches:** lower bound > −0.02;
- **below:** otherwise.

**Added 2026-10-14:** the Jetson items (§16: S3 latency front, bounded variant, KITTI with the shared configuration, streaming AP, cost-model fit) and the S6 test seeds (§1).

**Added 2026-10-14:** the qualitative and failure panels (§17, panels/).

**Added 2026-10-15:** K6 v6-S closed loop, all three runs (§14a). Phase 3 has no open item left.

## 0. Summary

- **S4, each component significant on ≥ 2 datasets: NOT MET.**
  - Met: field 3 / 3; support 2 / 3; looming estimate 3 / 3, both C1 vs T4 and C1 vs none.
  - **Guard: 0 / 3.** v6-S − (v6-S without guard) is −0.012 / +0.005 / +0.002, none significant. Without the guard, v6-S uses 18–23% fewer px.
  - **Refiner: 1 / 3** (phase 2).
  - Also not significant: the multi-round loop (≤ 3 rounds vs 1), 0 / 3.
- **S5, the ordering with a second detector family and with fine-tuned detectors: met where it can be tested at budget S.**
  - **v6-S vs v5.1-S:** v6-S is above v5.1-S in 4 of 5 settings and matches it in the fifth (YOLO11s on AV2, +0.015 [−0.010, +0.037]).
  - **In-class (S) rows:** v6-S outperforms every one. Only three settings have any: Argoverse-HD YOLO11s, nuScenes ft0, nuScenes ft1.
  - **YOLO11s, nuScenes / AV2:** no frame / region row falls in class S. The smallest whole-frame side is 1.04× on nuScenes and 1.25× on AV2; the S heuristics realise 1.15–1.7× because of guard re-detections.
  - **Fine-tuned verifiers:** no row has a higher point estimate than v6-S.
  - **YOLO11s, beyond class S:** the YOLOv8s pattern repeats. Whole frame + C1 is above v6-S on AV2 (significant at 640 = 1.9× and 960 = 4.3×) and on nuScenes at 1280 (8.6×).
- **S3, on the Jetson latency-G front in mean and p95 on both timed datasets: NOT MET** (§16).
  - **AV2 (8 test logs, ego mask): met.** v6-S, G 0.419 at 73.9 ms mean / 173 ms p95, is on the front in both. No faster row has a higher G. The nearest faster rows are v5.1-S ≤ 2 rounds (0.401, 46 ms) and whole frame native + T4 + refiner (0.388, 70 ms), both within the interval.
  - **nuScenes (20 local DEEP scenes, not test scenes): not met.** v6-S, G 0.096 at 29.5 / 78 ms, is dominated in mean and in p95. For example, whole frame u1280 + T4 + refiner reaches 0.118 at 25.6 / 30 ms. No dominating row is significantly above v6-S in G.
  - On these 20 scenes the refiner lowers v6-S's G (0.116 without it). v6-S ≤ 2 rounds without refiner (0.126, 25.9 / 63 ms) is on the nuScenes front, but it is not the frozen configuration.
  - **The C1 lift is the main latency cost.** On nuScenes, whole frame u448 + T4 + refiner takes 10.4 ms mean (15.1 p95); with C1 instead of T4 it takes 22.4 (51.6 p95).
- **KITTI, shared v6-S configuration: below whole frame + components** (§16). Mean of 4 cells: 0.235 vs native + T4 + refiner 0.370, difference [−0.182, −0.030].
- **Streaming AP, Argoverse-HD: v6-S 0.064** vs cheap 0.096 and K7's u1280 0.247 (§16). At 74.9 ms mean, v6-S processes 56% of frames and shows outputs 128 ms stale on average.
- **S6, refiner seed std < half the smallest S1 margin: met on dev and on test** (post-hoc test runs, §1).
  - On test, the largest seed std over 3 datasets × 3 configurations is 0.0099 (AV2 v6-M); on dev it is 0.0033.
  - The smallest margin S1 claims (S, deployable rows) is 0.105, so half is 0.053.
  - Even the smallest margin of any outperformed in-class row at any budget (0.023) gives 0.011, which the test std also clears.
  - Seed 0 on test reproduces the phase-2 v6 costs on 100% of frames.
- **AV2 without the ego mask** (§3):
  - v6-S is above v5.1-S (+0.009 [+0.002, +0.016]) and above all 3 in-class rows.
  - **v6-M collapses** (G 0.306 vs v6-S 0.617) and v6-L drops (0.542). Both are below many in-class rows.
  - Unmasked AV2 G is inflated by ego-hood detections of the whole-frame draft, which is why the ego mask is the protocol (K10). The v6-M collapse is consistent with its YOLOv8m crops confirming the hood. This cause is **not verified** per frame.
- **Prior works** (§4):
  - v6-S / M / L outperform AdaScale, every FOVEA setting and SAHI on AV2, at every budget.
  - nuScenes SAHI: v6-S is below-verdict with the higher point estimate (+0.118 / +0.081); v6-M / L outperform it.
  - **The fine-tuned-detector rows of nuScenes** (FOVEA-ft, u960-ft and the previous ours v3 + ft verifier, two seeds) are **above v6-S in the point estimate** (−0.024 … −0.086), all below-verdict. v6-L outperforms FOVEA-ft for both seeds.
- **Mechanism checks:**
  - **Agreement with π(u960 + C1):** v6-S agrees with it on 97.6 / 96.8 / 94.0% of frames, a tie with the best heuristic + C1 (+0.002 / +0.000 / +0.001).
  - **Missed objects:** v6-S recovers 61 / 85 / 87% of the natural draft misses that u960 finds, and 85 / 88 / 85% of the objects deleted from the draft. The uncertainty heuristic recovers 34–47% of the deleted objects.
  - **Second consumer:** the TTC-warning loop is above uncertainty crops at its own px on AV2 and Argoverse-HD, and below-verdict on nuScenes (+0.004 [−0.031, +0.041]).
- **Detection AP (§11):** v6 is a decision-relevant detector, not a better generic detector.
  - Argoverse-HD COCO AP: v6-S 0.165 vs whole frame YOLOv8s 512 0.182 at the same GFLOPs (1.02×).
  - nuScenes relevant-object precision: v6-S 0.890, the highest of every row. Its relevant-FP rate (10.6 / 1k frames) is the lowest.
- **K6 / NAVSIM** (§14):
  - NAVSIM's perception (spec_pdm) is unchanged by K28, so there is no re-run.
  - K6 ran nuScenes v3, so the configuration did change. The ours branch was re-run with v6-S (3 runs, 2026-10-14 / 15).
  - **Three runs (§14a, DONE 2026-10-15):** v6-S DS 92.54 ± 0.82, SR 0.839 ± 0.011.
    - It ties cheap, u960, u1600 and K6's v3 (every interval across 0; vs v3 −0.26 [−1.23, +0.61]).
    - It is below privileged: −4.09 [−6.36, −2.02] DS, −0.076 [−0.117, −0.038] SR.

## 1. Scorecard items decided in phase 3

### S4: ablation of v6-S, every component (with refiner; phase-2 rows plus the phase-3 cells pv6S / or6S / ng6S)

| component | nuScenes | AV2 (ego mask) | Argoverse-HD | significant on | ≥ 2 |
|---|---|---|---|---|---|
| field | +0.306 [+0.060, +0.532] **sig.** | +0.075 [+0.031, +0.130] **sig.** | +0.105 [+0.020, +0.228] **sig.** | 3 / 3 | yes |
| support | +0.037 [-0.079, +0.159] | +0.273 [+0.217, +0.331] **sig.** | +0.338 [+0.255, +0.434] **sig.** | 2 / 3 | yes |
| guard | -0.012 [-0.056, +0.030] | +0.005 [-0.009, +0.019] | +0.002 [-0.013, +0.020] | 0 / 3 | **no** |
| looming, C1 vs T4 | +0.012 [+0.004, +0.026] **sig.** | +0.032 [+0.013, +0.050] **sig.** | +0.079 [+0.054, +0.107] **sig.** | 3 / 3 | yes |
| looming, C1 vs none | +0.035 [+0.014, +0.066] **sig.** | +0.086 [+0.039, +0.130] **sig.** | +0.233 [+0.173, +0.310] **sig.** | 3 / 3 | yes |
| refiner | +0.030 [-0.000, +0.068] | +0.053 [+0.011, +0.095] **sig.** | -0.004 [-0.083, +0.101] | 1 / 3 | **no** |
| support vs pivots | +0.009 [-0.022, +0.048] | +0.066 [+0.038, +0.101] **sig.** | +0.034 [+0.021, +0.048] **sig.** | 2 / 3 | yes |
| rounds (≤ 3 vs 1) | +0.116 [-0.005, +0.296] | +0.005 [-0.004, +0.013] | -0.004 [-0.015, +0.006] | 0 / 3 | **no** |

**Notes on the cells:**
- "support vs pivots" replaces the support crops with pivot crops.
- "rounds" runs a single round of the loop.
- Neither is one of S4's five components; both are listed for completeness.

**Guard:**
- Without the guard, v6-S uses 0.063 / 0.041 / 0.054 px instead of 0.078 / 0.054 / 0.070.
- G changes by −0.012 / +0.005 / +0.002.

### S5: second detector family (YOLO11s as draft and verifier, v3a and C1 as frozen) and fine-tuned verifiers (nuScenes)

GFLOPs are proportional to px in these runs, since one architecture runs everywhere. The class is therefore px / px of v5.1-S in the same run.

| setting | split | G v6-S | G v5.1-S | v6-S − v5.1-S | rows in class S | class S: outperformed | class M (≤ 1.53×): outperformed | all rows: outperformed / matches / below | rows with higher G (significant) |
|---|---|---|---|---|---|---|---|---|---|
| YOLO11s draft + verifier, no re-tuning | nuScenes | 0.418 | 0.410 | +0.009 [+0.002, +0.025] (outperforms) | 0 | 0 / 0 | 26 / 30 | 30 / 1 / 17 | 7 (1) |
| YOLO11s draft + verifier, no re-tuning | AV2 (ego mask) | 0.261 | 0.246 | +0.015 [-0.010, +0.037] (matches) | 0 | 0 / 0 | 20 / 27 | 28 / 8 / 9 | 5 (2) |
| YOLO11s draft + verifier, no re-tuning | Argoverse-HD | 0.422 | 0.362 | +0.060 [+0.038, +0.084] (outperforms) | 3 (wf448, wf448+C1, wf448+T4) | 3 / 3 | 9 / 10 | 45 / 2 / 1 | 0 (0) |
| fine-tuned YOLOv8s verifier, seed 0 | nuScenes | 0.359 | 0.347 | +0.012 [+0.003, +0.026] (outperforms) | 3 (wf448, wf448+C1, wf448+T4) | 3 / 3 | 15 / 15 | 42 / 3 / 3 | 0 (0) |
| fine-tuned YOLOv8s verifier, seed 1 | nuScenes | 0.364 | 0.353 | +0.010 [+0.003, +0.023] (outperforms) | 3 (wf448, wf448+C1, wf448+T4) | 3 / 3 | 10 / 12 | 41 / 4 / 3 | 0 (0) |

Every row v6-S does not outperform in these runs (**51 rows**):

| setting | split | row | px / v5.1-S | class | G row | v6-S − row [95% CI] | verdict | point estimate |
|---|---|---|---|---|---|---|---|---|
| y11s | nuScenes | wf448+T4 | 1.04 | M | 0.237 | +0.182 [-0.015, +0.320] | matches | v6-S higher |
| y11s | nuScenes | wf448+C1 | 1.04 | M | 0.254 | +0.165 [-0.045, +0.311] | below | v6-S higher |
| y11s | nuScenes | wf512+T4 | 1.34 | M | 0.285 | +0.133 [-0.035, +0.248] | below | v6-S higher |
| y11s | nuScenes | wf512+C1 | 1.34 | M | 0.309 | +0.109 [-0.070, +0.234] | below | v6-S higher |
| y11s | nuScenes | wf640+T4 | 2.23 | L | 0.352 | +0.066 [-0.040, +0.126] | below | v6-S higher |
| y11s | nuScenes | wf640+C1 | 2.23 | L | 0.379 | +0.040 [-0.078, +0.111] | below | v6-S higher |
| y11s | nuScenes | wf960+T4 | 5.02 | over | 0.424 | -0.006 [-0.042, +0.049] | below | row higher |
| y11s | nuScenes | wf960 | 5.02 | over | 0.408 | +0.011 [-0.037, +0.085] | below | v6-S higher |
| y11s | nuScenes | wf960+C1 | 5.02 | over | 0.446 | -0.027 [-0.068, +0.018] | below | row higher |
| y11s | nuScenes | wf1280 | 8.55 | over | 0.434 | -0.015 [-0.072, +0.077] | below | row higher |
| y11s | nuScenes | wf1280+T4 | 8.55 | over | 0.457 | -0.038 [-0.091, +0.030] | below | row higher |
| y11s | nuScenes | wf1280+C1 | 8.55 | over | 0.483 | -0.064 [-0.114, -0.015] | below | row higher |
| y11s | nuScenes | wf1536+T4 | 12.04 | over | 0.397 | +0.022 [-0.047, +0.111] | below | v6-S higher |
| y11s | nuScenes | wf1536+C1 | 12.04 | over | 0.415 | +0.003 [-0.064, +0.078] | below | v6-S higher |
| y11s | nuScenes | wf1536 | 12.04 | over | 0.384 | +0.034 [-0.042, +0.138] | below | v6-S higher |
| y11s | nuScenes | wf1600 | 13.47 | over | 0.410 | +0.009 [-0.058, +0.102] | below | v6-S higher |
| y11s | nuScenes | wf1600+C1 | 13.47 | over | 0.443 | -0.025 [-0.082, +0.043] | below | row higher |
| y11s | nuScenes | wf1600+T4 | 13.47 | over | 0.432 | -0.014 [-0.078, +0.063] | below | row higher |
| y11s | AV2 (ego mask) | tsp2@S | 1.24 | M | 0.216 | +0.044 [-0.006, +0.090] | matches | v6-S higher |
| y11s | AV2 (ego mask) | tsp2@S+C1 | 1.24 | M | 0.248 | +0.013 [-0.002, +0.029] | matches | v6-S higher |
| y11s | AV2 (ego mask) | wf512+T4 | 1.25 | M | 0.213 | +0.047 [-0.004, +0.100] | matches | v6-S higher |
| y11s | AV2 (ego mask) | wf512+C1 | 1.25 | M | 0.310 | -0.050 [-0.109, +0.005] | below | row higher |
| y11s | AV2 (ego mask) | tsp8@S | 1.36 | M | 0.208 | +0.053 [-0.000, +0.106] | matches | v6-S higher |
| y11s | AV2 (ego mask) | tsp8@S+C1 | 1.36 | M | 0.253 | +0.008 [-0.014, +0.031] | matches | v6-S higher |
| y11s | AV2 (ego mask) | tsp8@S+T4 | 1.36 | M | 0.236 | +0.025 [-0.007, +0.057] | matches | v6-S higher |
| y11s | AV2 (ego mask) | wf640+T4 | 1.93 | L | 0.238 | +0.023 [-0.030, +0.070] | below | v6-S higher |
| y11s | AV2 (ego mask) | wf640+C1 | 1.93 | L | 0.316 | -0.056 [-0.113, -0.005] | below | row higher |
| y11s | AV2 (ego mask) | wf960+T4 | 4.34 | over | 0.240 | +0.021 [-0.036, +0.073] | below | v6-S higher |
| y11s | AV2 (ego mask) | wf960+C1 | 4.34 | over | 0.323 | -0.062 [-0.123, -0.005] | below | row higher |
| y11s | AV2 (ego mask) | wf1280+T4 | 7.47 | over | 0.212 | +0.049 [-0.003, +0.101] | matches | v6-S higher |
| y11s | AV2 (ego mask) | wf1280+C1 | 7.47 | over | 0.311 | -0.051 [-0.112, +0.008] | below | row higher |
| y11s | AV2 (ego mask) | wf1536+C1 | 10.69 | over | 0.313 | -0.053 [-0.117, +0.006] | below | row higher |
| y11s | AV2 (ego mask) | wf1536+T4 | 10.69 | over | 0.208 | +0.052 [-0.011, +0.108] | matches | v6-S higher |
| y11s | AV2 (ego mask) | wf2048+T4 | 18.88 | over | 0.165 | +0.095 [-0.031, +0.218] | below | v6-S higher |
| y11s | AV2 (ego mask) | wf2048+C1 | 18.88 | over | 0.260 | +0.000 [-0.094, +0.093] | below | v6-S higher |
| y11s | Argoverse-HD | tsp2@S+C1 | 1.39 | M | 0.411 | +0.011 [-0.002, +0.027] | matches | v6-S higher |
| y11s | Argoverse-HD | tsp8@S+C1 | 1.55 | L | 0.414 | +0.009 [-0.003, +0.020] | matches | v6-S higher |
| y11s | Argoverse-HD | wf1920+C1 | 14.68 | over | 0.412 | +0.011 [-0.025, +0.050] | below | v6-S higher |
| ft0 | nuScenes | wf640 | 1.90 | L | 0.305 | +0.054 [-0.016, +0.134] | matches | v6-S higher |
| ft0 | nuScenes | wf640+C1 | 1.90 | L | 0.323 | +0.036 [-0.014, +0.102] | matches | v6-S higher |
| ft0 | nuScenes | wf640+T4 | 1.90 | L | 0.328 | +0.031 [-0.029, +0.093] | below | v6-S higher |
| ft0 | nuScenes | wf960+C1 | 4.27 | over | 0.328 | +0.031 [-0.020, +0.094] | below | v6-S higher |
| ft0 | nuScenes | wf960+T4 | 4.27 | over | 0.330 | +0.029 [-0.032, +0.091] | below | v6-S higher |
| ft0 | nuScenes | wf1600+C1 | 11.45 | over | 0.286 | +0.073 [-0.003, +0.184] | matches | v6-S higher |
| ft1 | nuScenes | wf512+C1 | 1.13 | M | 0.322 | +0.042 [-0.011, +0.107] | matches | v6-S higher |
| ft1 | nuScenes | wf512+T4 | 1.13 | M | 0.322 | +0.042 [-0.007, +0.093] | matches | v6-S higher |
| ft1 | nuScenes | wf640+C1 | 1.88 | L | 0.317 | +0.047 [-0.006, +0.112] | matches | v6-S higher |
| ft1 | nuScenes | wf640+T4 | 1.88 | L | 0.322 | +0.042 [-0.022, +0.110] | below | v6-S higher |
| ft1 | nuScenes | wf960 | 4.24 | over | 0.311 | +0.053 [-0.012, +0.135] | matches | v6-S higher |
| ft1 | nuScenes | wf960+C1 | 4.24 | over | 0.356 | +0.008 [-0.053, +0.072] | below | v6-S higher |
| ft1 | nuScenes | wf960+T4 | 4.24 | over | 0.352 | +0.012 [-0.057, +0.075] | below | v6-S higher |

### S6: refiner seeds (dev; phase 1)

| v6 configuration (dev candidate) | nuScenes dev G, seeds 0 / 1 / 2 | std | AV2 dev G, seeds 0 / 1 / 2 | std |
|---|---|---|---|---|
| v6-S (C1; refiner v3a / v3a_s1 / v3a_s2) | 0.321 / 0.323 / 0.321 | 0.0013 | 0.309 / 0.308 / 0.312 | 0.0020 |
| v6-M (R3-2, YOLOv8m crops 768, own refiner) | 0.394 / 0.399 / 0.395 | 0.0029 | 0.311 / 0.311 / 0.317 | 0.0033 |
| v6-L (R2-1, YOLO11m crops 960, own refiner) | 0.408 / 0.407 / 0.413 | 0.0031 | 0.311 / 0.308 / 0.308 | 0.0016 |

- **Smallest margin claimed in S1:**
  - S1 is met at S only. The smallest v6-S − row margin among in-class deployable rows there is +0.105 (nuScenes), so half is 0.053.
  - Over every outperformed in-class row at any budget, the smallest margin is +0.023 (AV2, L), so half is 0.011.
- The largest seed std, 0.0033, is below both. **S6: met** on dev seeds.

### S6: refiner seeds on test (post-hoc; seeds_test_v6.csv, s6_test_v6.csv)

Run: `k28_p3.py --part ours --seeds --tag seeds` on all test units (nuScenes 228, AV2 150 with the ego mask, Argoverse-HD 24 with 3D labels), then `k28_p3_seeds.py`. The loop is refiner-independent, so only the refiner seed changes. Std uses ddof 1, as phase 1.

| dataset | configuration | G, seeds 0 / 1 / 2 | mean | std | seed 0 identical to phase 2 |
|---|---|---|---|---|---|
| nuScenes | v6-S | 0.211 / 0.210 / 0.208 | 0.209 | 0.0016 | 100% of 9,181 frames |
| nuScenes | v6-M | 0.298 / 0.303 / 0.302 | 0.301 | 0.0029 | 100% |
| nuScenes | v6-L | 0.315 / 0.310 / 0.320 | 0.315 | 0.0050 | 100% |
| AV2 (ego mask) | v6-S | 0.294 / 0.291 / 0.284 | 0.289 | 0.0049 | 100% of 11,818 frames |
| AV2 (ego mask) | v6-M | 0.281 / 0.268 / 0.288 | 0.279 | 0.0099 | 100% |
| AV2 (ego mask) | v6-L | 0.292 / 0.294 / 0.303 | 0.296 | 0.0058 | 100% |
| Argoverse-HD | v6-S | 0.418 / 0.429 / 0.429 | 0.425 | 0.0062 | 100% of 15,006 frames |
| Argoverse-HD | v6-M | 0.428 / 0.423 / 0.430 | 0.427 | 0.0040 | 100% |
| Argoverse-HD | v6-L | 0.431 / 0.427 / 0.425 | 0.428 | 0.0029 | 100% |

- **Largest test std: 0.0099** (AV2 v6-M). It is below half the smallest S1 margin at S (0.0527) and below half the smallest outperformed in-class margin at any budget (0.0113). **S6: met on test.**
- Test stds are larger than dev stds (up to 3×). The AV2 v6-M seed spread, 0.268–0.288, is close to its gap to v6-S (0.294) and v6-L (0.292), so the ordering of the three AV2 configurations is not seed-stable.

## 2. Full ablation of v6-S (k28_p3_p0.py --part abl)

| split | cell | refiner | component removed | G | px | v6-S − cell [95% CI] | verdict | unnecessary / 1k | missed / 1k | crops | rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nuScenes | support only (no field) | v3a | field | -0.095 | 0.050 | +0.306 [+0.060, +0.532] | outperforms | 11.0 | 46.3 | 0.134 | 0.128 |
| nuScenes | field only (no support) | v3a | support | 0.174 | 0.068 | +0.037 [-0.079, +0.159] | below | 50.5 | 33.4 | 0.497 | 0.497 |
| nuScenes | pivots instead of support | v3a | support (pivots) | 0.202 | 0.076 | +0.009 [-0.022, +0.048] | below | 17.0 | 38.4 | 0.637 | 0.615 |
| nuScenes | single round | v3a | rounds | 0.095 | 0.075 | +0.116 [-0.005, +0.296] | matches | 16.0 | 38.7 | 0.611 | 0.581 |
| nuScenes | without guard | v3a | guard | 0.222 | 0.063 | -0.012 [-0.056, +0.030] | below | 15.6 | 37.8 | 0.668 | 0.636 |
| nuScenes | without C1 (= v5 + v3a) | v3a | looming estimate | 0.176 | 0.077 | +0.035 [+0.014, +0.066] | outperforms | 32.6 | 34.4 | – | 0.630 |
| nuScenes | T4 instead of C1 (= v5.1-S) | v3a | C1 vs T4 | 0.199 | 0.077 | +0.012 [+0.004, +0.026] | outperforms | 21.7 | 37.3 | 0.657 | 0.626 |
| nuScenes | without refiner | none | refiner | 0.181 | 0.078 | +0.030 [-0.000, +0.068] | matches | 21.6 | 42.5 | 0.655 | 0.624 |
| nuScenes | without refiner and without C1 (= v5) | none | refiner + looming | 0.133 | 0.077 | +0.078 [+0.037, +0.133] | outperforms | 42.7 | 38.6 | – | 0.630 |
| nuScenes | support only, no refiner | none | - | -0.121 | 0.050 | +0.332 [+0.087, +0.548] | outperforms | 13.8 | 51.5 | 0.134 | 0.128 |
| nuScenes | field only, no refiner | none | - | 0.164 | 0.068 | +0.046 [-0.074, +0.176] | below | 54.1 | 37.0 | 0.497 | 0.497 |
| nuScenes | pivots, no refiner | none | - | 0.185 | 0.076 | +0.026 [-0.015, +0.079] | matches | 22.7 | 42.6 | 0.637 | 0.615 |
| nuScenes | single round, no refiner | none | - | 0.066 | 0.075 | +0.145 [+0.005, +0.318] | outperforms | 22.2 | 43.1 | 0.611 | 0.581 |
| nuScenes | without guard, no refiner | none | - | 0.188 | 0.063 | +0.023 [-0.037, +0.091] | below | 22.1 | 42.5 | 0.668 | 0.636 |
| AV2 (ego mask) | support only (no field) | v3a | field | 0.218 | 0.041 | +0.075 [+0.031, +0.130] | outperforms | 46.5 | 50.4 | 0.490 | 0.434 |
| AV2 (ego mask) | field only (no support) | v3a | support | 0.021 | 0.037 | +0.273 [+0.217, +0.331] | outperforms | 180.9 | 30.0 | 0.401 | 0.401 |
| AV2 (ego mask) | pivots instead of support | v3a | support (pivots) | 0.228 | 0.049 | +0.066 [+0.038, +0.101] | outperforms | 64.2 | 39.8 | 0.756 | 0.709 |
| AV2 (ego mask) | single round | v3a | rounds | 0.289 | 0.050 | +0.005 [-0.004, +0.013] | matches | 54.0 | 39.3 | 0.755 | 0.663 |
| AV2 (ego mask) | without guard | v3a | guard | 0.288 | 0.041 | +0.005 [-0.009, +0.019] | matches | 53.6 | 39.3 | 0.908 | 0.801 |
| AV2 (ego mask) | without C1 (= v5 + v3a) | v3a | looming estimate | 0.208 | 0.053 | +0.086 [+0.039, +0.130] | outperforms | 112.4 | 30.2 | – | 0.775 |
| AV2 (ego mask) | T4 instead of C1 (= v5.1-S) | v3a | C1 vs T4 | 0.262 | 0.053 | +0.032 [+0.013, +0.050] | outperforms | 76.5 | 34.5 | 0.871 | 0.771 |
| AV2 (ego mask) | without refiner | none | refiner | 0.241 | 0.054 | +0.053 [+0.011, +0.095] | outperforms | 138.0 | 29.1 | 0.872 | 0.772 |
| AV2 (ego mask) | without refiner and without C1 (= v5) | none | refiner + looming | 0.031 | 0.053 | +0.262 [+0.187, +0.335] | outperforms | 206.5 | 23.5 | – | 0.775 |
| AV2 (ego mask) | support only, no refiner | none | - | 0.190 | 0.041 | +0.104 [+0.050, +0.158] | outperforms | 104.3 | 42.9 | 0.490 | 0.434 |
| AV2 (ego mask) | field only, no refiner | none | - | 0.020 | 0.037 | +0.273 [+0.210, +0.344] | outperforms | 222.7 | 24.2 | 0.401 | 0.401 |
| AV2 (ego mask) | pivots, no refiner | none | - | 0.191 | 0.049 | +0.103 [+0.057, +0.151] | outperforms | 142.6 | 30.7 | 0.756 | 0.709 |
| AV2 (ego mask) | single round, no refiner | none | - | 0.229 | 0.050 | +0.064 [+0.017, +0.114] | outperforms | 138.3 | 30.0 | 0.755 | 0.663 |
| AV2 (ego mask) | without guard, no refiner | none | - | 0.236 | 0.041 | +0.058 [+0.015, +0.102] | outperforms | 138.3 | 29.5 | 0.908 | 0.801 |
| Argoverse-HD | support only (no field) | v3a | field | 0.313 | 0.056 | +0.105 [+0.020, +0.228] | outperforms | 125.3 | 92.0 | 0.530 | 0.469 |
| Argoverse-HD | field only (no support) | v3a | support | 0.080 | 0.043 | +0.338 [+0.255, +0.434] | outperforms | 298.5 | 29.9 | 0.390 | 0.390 |
| Argoverse-HD | pivots instead of support | v3a | support (pivots) | 0.384 | 0.061 | +0.034 [+0.021, +0.048] | outperforms | 149.0 | 49.2 | 0.828 | 0.790 |
| Argoverse-HD | single round | v3a | rounds | 0.421 | 0.065 | -0.004 [-0.015, +0.006] | matches | 132.4 | 52.4 | 0.814 | 0.727 |
| Argoverse-HD | without guard | v3a | guard | 0.416 | 0.054 | +0.002 [-0.013, +0.020] | matches | 134.1 | 49.2 | 1.059 | 0.951 |
| Argoverse-HD | without C1 (= v5 + v3a) | v3a | looming estimate | 0.185 | 0.070 | +0.233 [+0.173, +0.310] | outperforms | 240.8 | 34.7 | – | 0.877 |
| Argoverse-HD | T4 instead of C1 (= v5.1-S) | v3a | C1 vs T4 | 0.338 | 0.070 | +0.079 [+0.054, +0.107] | outperforms | 169.1 | 45.6 | 0.971 | 0.872 |
| Argoverse-HD | without refiner | none | refiner | 0.422 | 0.070 | -0.004 [-0.083, +0.101] | below | 202.9 | 23.7 | 0.976 | 0.875 |
| Argoverse-HD | without refiner and without C1 (= v5) | none | refiner + looming | 0.039 | 0.070 | +0.379 [+0.260, +0.504] | outperforms | 325.8 | 15.3 | – | 0.877 |
| Argoverse-HD | support only, no refiner | none | - | 0.314 | 0.056 | +0.104 [-0.023, +0.233] | below | 183.2 | 69.7 | 0.530 | 0.469 |
| Argoverse-HD | field only, no refiner | none | - | 0.090 | 0.043 | +0.328 [+0.235, +0.433] | outperforms | 318.3 | 17.5 | 0.390 | 0.390 |
| Argoverse-HD | pivots, no refiner | none | - | 0.389 | 0.061 | +0.029 [-0.046, +0.131] | below | 212.7 | 23.3 | 0.828 | 0.790 |
| Argoverse-HD | single round, no refiner | none | - | 0.419 | 0.065 | -0.001 [-0.080, +0.102] | below | 203.5 | 24.3 | 0.814 | 0.727 |
| Argoverse-HD | without guard, no refiner | none | - | 0.418 | 0.054 | -0.000 [-0.079, +0.100] | below | 206.4 | 22.8 | 1.059 | 0.951 |

## 3. AV2 val without the ego mask (all 150 logs; same rows as phase 2)

| configuration | GFLOPs / v5.1-S | G v5.1-S | G v6 | G v6 no refiner | v6 − v5.1-S [95% CI] | verdict |
|---|---|---|---|---|---|---|
| v6-S | 1.00 | 0.607 | 0.617 | 0.610 | +0.009 [+0.002, +0.016] | outperforms |
| v6-M | 1.42 | 0.607 | 0.306 | 0.339 | -0.302 [-0.353, -0.252] | below |
| v6-L | 1.88 | 0.607 | 0.542 | 0.546 | -0.065 [-0.090, -0.040] | below |

**Class-matched, as in phase 2** (each row against the v6 configuration of its GFLOPs class; "over" rows against v6-L):

| class | v6 | rows | outperforms | matches | below, v6 higher | below, row higher | row significantly higher |
|---|---|---|---|---|---|---|---|
| S | v6-S | 3 | 3 | 0 | 0 | 0 | 0 |
| M | v6-M | 36 | 21 | 0 | 3 | 12 | 11 |
| L | v6-L | 33 | 15 | 0 | 0 | 18 | 15 |
| over | v6-L | 24 | 10 | 1 | 1 | 12 | 8 |

v6-S against every row regardless of class: 75 outperformed, 12 matches, 9 below (6 with the row significantly higher) of 96.

**Every class-matched row v6 does not outperform (47):**

| row | group | class | GFLOPs / v5.1-S | G row | v6 | G v6 | v6 − row [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|
| tsp8@L+C1 | heuristic + C1 | L | 1.79 | 0.639 | v6-L | 0.542 | -0.097 [-0.131, -0.064] | below |
| tsp8@M+C1 | heuristic + C1 | L | 1.73 | 0.634 | v6-L | 0.542 | -0.092 [-0.122, -0.064] | below |
| tsp0@L+C1 | heuristic + C1 | L | 1.96 | 0.633 | v6-L | 0.542 | -0.091 [-0.122, -0.061] | below |
| tsp4@M+C1 | heuristic + C1 | L | 1.61 | 0.624 | v6-L | 0.542 | -0.082 [-0.107, -0.057] | below |
| tsp8@M+T4 | heuristic + T4 | L | 1.73 | 0.609 | v6-L | 0.542 | -0.067 [-0.098, -0.036] | below |
| tsp0@L+T4 | heuristic + T4 | L | 1.96 | 0.608 | v6-L | 0.542 | -0.066 [-0.097, -0.034] | below |
| tsp8@L+T4 | heuristic + T4 | L | 1.79 | 0.607 | v6-L | 0.542 | -0.065 [-0.096, -0.034] | below |
| unc2@L+C1 | heuristic + C1 | L | 2.13 | 0.607 | v6-L | 0.542 | -0.065 [-0.097, -0.032] | below |
| tsp4@M+T4 | heuristic + T4 | L | 1.61 | 0.607 | v6-L | 0.542 | -0.064 [-0.092, -0.036] | below |
| unc2@L+T4 | heuristic + T4 | L | 2.13 | 0.596 | v6-L | 0.542 | -0.054 [-0.086, -0.020] | below |
| unc2@M+C1 | heuristic + C1 | L | 1.87 | 0.595 | v6-L | 0.542 | -0.053 [-0.086, -0.020] | below |
| tsp4@M | heuristic plain | L | 1.61 | 0.590 | v6-L | 0.542 | -0.047 [-0.079, -0.016] | below |
| unc2@M+T4 | heuristic + T4 | L | 1.87 | 0.586 | v6-L | 0.542 | -0.043 [-0.077, -0.010] | below |
| unc2@L | heuristic plain | L | 2.13 | 0.580 | v6-L | 0.542 | -0.038 [-0.072, -0.003] | below |
| tsp8@M | heuristic plain | L | 1.73 | 0.576 | v6-L | 0.542 | -0.033 [-0.065, -0.001] | below |
| tsp8@L | heuristic plain | L | 1.79 | 0.574 | v6-L | 0.542 | -0.032 [-0.064, +0.000] | below |
| tsp0@L | heuristic plain | L | 1.96 | 0.573 | v6-L | 0.542 | -0.031 [-0.065, +0.003] | below |
| unc2@M | heuristic plain | L | 1.87 | 0.572 | v6-L | 0.542 | -0.029 [-0.065, +0.005] | below |
| tsp8@S+C1 | heuristic + C1 | M | 1.20 | 0.619 | v6-M | 0.306 | -0.313 [-0.366, -0.263] | below |
| tsp2@S+C1 | heuristic + C1 | M | 1.14 | 0.616 | v6-M | 0.306 | -0.310 [-0.362, -0.260] | below |
| tsp8@S+T4 | heuristic + T4 | M | 1.20 | 0.611 | v6-M | 0.306 | -0.306 [-0.359, -0.254] | below |
| tsp2@S+T4 | heuristic + T4 | M | 1.14 | 0.609 | v6-M | 0.306 | -0.304 [-0.357, -0.253] | below |
| tsp8@S | heuristic plain | M | 1.20 | 0.586 | v6-M | 0.306 | -0.280 [-0.333, -0.225] | below |
| tsp2@S | heuristic plain | M | 1.14 | 0.583 | v6-M | 0.306 | -0.278 [-0.333, -0.222] | below |
| unc2@S+C1 | heuristic + C1 | M | 1.45 | 0.478 | v6-M | 0.306 | -0.172 [-0.221, -0.123] | below |
| unc2@S+T4 | heuristic + T4 | M | 1.45 | 0.476 | v6-M | 0.306 | -0.170 [-0.219, -0.120] | below |
| unc2@S | heuristic plain | M | 1.45 | 0.460 | v6-M | 0.306 | -0.154 [-0.204, -0.102] | below |
| unc4@S+C1 | heuristic + C1 | M | 1.45 | 0.357 | v6-M | 0.306 | -0.051 [-0.098, -0.005] | below |
| unc4@S+T4 | heuristic + T4 | M | 1.45 | 0.356 | v6-M | 0.306 | -0.051 [-0.099, -0.004] | below |
| unc4@S | heuristic plain | M | 1.45 | 0.341 | v6-M | 0.306 | -0.035 [-0.084, +0.014] | below |
| tcr0@S+T4 | heuristic + T4 | M | 1.44 | 0.291 | v6-M | 0.306 | +0.014 [-0.037, +0.064] | below |
| tcr0@S+C1 | heuristic + C1 | M | 1.44 | 0.290 | v6-M | 0.306 | +0.016 [-0.034, +0.067] | below |
| tcr0@S | heuristic plain | M | 1.44 | 0.287 | v6-M | 0.306 | +0.019 [-0.033, +0.072] | below |
| wf2048+C1 | whole frame + C1 | over | 13.63 | 0.693 | v6-L | 0.542 | -0.151 [-0.201, -0.102] | below |
| wf1536+C1 | whole frame + C1 | over | 7.72 | 0.669 | v6-L | 0.542 | -0.127 [-0.164, -0.089] | below |
| wf2048+T4 | whole frame + T4 | over | 13.63 | 0.666 | v6-L | 0.542 | -0.124 [-0.174, -0.073] | below |
| wf2048 | whole frame plain | over | 13.63 | 0.639 | v6-L | 0.542 | -0.097 [-0.147, -0.048] | below |
| wf1536+T4 | whole frame + T4 | over | 7.72 | 0.632 | v6-L | 0.542 | -0.090 [-0.127, -0.055] | below |
| wf1536 | whole frame plain | over | 7.72 | 0.609 | v6-L | 0.542 | -0.066 [-0.103, -0.030] | below |
| wf1280+C1 | whole frame + C1 | over | 5.39 | 0.587 | v6-L | 0.542 | -0.045 [-0.078, -0.014] | below |
| unc6@L+C1 | heuristic + C1 | over | 2.58 | 0.584 | v6-L | 0.542 | -0.041 [-0.077, -0.007] | below |
| unc6@L+T4 | heuristic + T4 | over | 2.58 | 0.563 | v6-L | 0.542 | -0.021 [-0.055, +0.013] | below |
| wf1280+T4 | whole frame + T4 | over | 5.39 | 0.562 | v6-L | 0.542 | -0.020 [-0.051, +0.010] | below |
| unc6@L | heuristic plain | over | 2.58 | 0.551 | v6-L | 0.542 | -0.009 [-0.047, +0.028] | below |
| wf960+C1 | whole frame + C1 | over | 3.13 | 0.546 | v6-L | 0.542 | -0.004 [-0.043, +0.035] | below |
| wf1280 | whole frame plain | over | 5.39 | 0.534 | v6-L | 0.542 | +0.008 [-0.023, +0.039] | below |
| wf960+T4 | whole frame + T4 | over | 3.13 | 0.519 | v6-L | 0.542 | +0.024 [-0.013, +0.062] | matches |

**Reading:**
- Without the mask, the draft (whole frame YOLOv8s 320) brakes for the ego hood. The cheap loss is therefore large, and every pipeline that verifies the hood away gains, so G roughly doubles (v5.1-S 0.607 vs 0.262 with the mask).
- The masked AV2 numbers are the protocol (K10).
- v6-M's collapse is not explained here. A per-frame check of hood detections in the YOLOv8m crops was not run.

## 4. Published prior works (official code / settings, K24 numbers, no C1)

| split | prior work | G | v6-S − it | v6-M − it | v6-L − it |
|---|---|---|---|---|---|
| nuScenes | AdaScale (re-impl., fit on new train) | 0.083 | +0.127 [+0.068, +0.198] outp. | +0.215 [+0.103, +0.334] outp. | +0.231 [+0.070, +0.408] outp. |
| nuScenes | FOVEA KDE_I 384x640 | -0.091 | +0.302 [+0.051, +0.600] outp. | +0.389 [+0.064, +0.804] outp. | +0.406 [+0.041, +0.878] outp. |
| nuScenes | FOVEA KDE_S 384x640 | -0.268 | +0.478 [+0.046, +1.035] outp. | +0.565 [+0.057, +1.249] outp. | +0.582 [+0.035, +1.321] outp. |
| nuScenes | FOVEA KDE_I 576x960 | 0.074 | +0.137 [+0.040, +0.253] outp. | +0.224 [+0.063, +0.366] outp. | +0.240 [+0.056, +0.427] outp. |
| nuScenes | FOVEA KDE_S 576x960 | 0.022 | +0.189 [+0.064, +0.312] outp. | +0.276 [+0.076, +0.472] outp. | +0.293 [+0.056, +0.534] outp. |
| nuScenes | SAHI 640-px slices (scale 0.6) | 0.093 | +0.118 [-0.045, +0.320] belo. | +0.205 [+0.130, +0.329] outp. | +0.221 [+0.130, +0.339] outp. |
| nuScenes | SAHI 640-px slices (scale 1.0) | 0.130 | +0.081 [-0.084, +0.274] belo. | +0.168 [+0.098, +0.282] outp. | +0.185 [+0.098, +0.288] outp. |
| nuScenes | FOVEA, fine-tuned detector (seed 0) | 0.235 | -0.024 [-0.210, +0.163] belo. | +0.063 [-0.017, +0.171] matc. | +0.080 [+0.019, +0.174] outp. |
| nuScenes | whole frame u960, fine-tuned detector (seed 0) | 0.241 | -0.030 [-0.223, +0.181] belo. | +0.057 [-0.033, +0.186] belo. | +0.073 [-0.005, +0.188] matc. |
| nuScenes | previous ours (v3) + fine-tuned verifier (seed 0) | 0.297 | -0.086 [-0.258, +0.094] belo. | +0.001 [-0.067, +0.100] belo. | +0.017 [-0.046, +0.097] belo. |
| nuScenes | FOVEA, fine-tuned detector (seed 1) | 0.244 | -0.034 [-0.224, +0.167] belo. | +0.054 [-0.029, +0.162] belo. | +0.070 [+0.001, +0.168] outp. |
| nuScenes | whole frame u960, fine-tuned detector (seed 1) | 0.272 | -0.062 [-0.244, +0.148] belo. | +0.026 [-0.063, +0.150] belo. | +0.042 [-0.036, +0.156] belo. |
| nuScenes | previous ours (v3) + fine-tuned verifier (seed 1) | 0.295 | -0.084 [-0.257, +0.087] belo. | +0.003 [-0.064, +0.095] belo. | +0.020 [-0.039, +0.094] belo. |
| AV2 (ego mask) | AdaScale (fit on AV2 train) | -0.057 | +0.350 [+0.268, +0.432] outp. | +0.338 [+0.257, +0.418] outp. | +0.348 [+0.265, +0.431] outp. |
| AV2 (ego mask) | SAHI slices (scale 0.47) | -0.295 | +0.589 [+0.477, +0.716] outp. | +0.576 [+0.455, +0.711] outp. | +0.587 [+0.472, +0.717] outp. |
| AV2 (ego mask) | SAHI slices (scale 1.0) | -0.248 | +0.541 [+0.425, +0.670] outp. | +0.529 [+0.406, +0.663] outp. | +0.539 [+0.421, +0.668] outp. |
| AV2 (ego mask) | FOVEA KDE_I 960 | -0.023 | +0.317 [+0.233, +0.403] outp. | +0.304 [+0.223, +0.389] outp. | +0.315 [+0.231, +0.402] outp. |

**Checks:**
- frames missing: 0; J_cheap agreement min 1.000.
- The nuScenes fine-tuned rows come from results/fovea/eval_ft_newtest*.pkl. "previous ours (v3) + fine-tuned verifier" is our own earlier configuration, not a prior work.

## 5. Agreement with π(u960 + C1) and the attribution chain (K2 / K16)

Agreement is the share of frames where a pipeline (no refiner) takes the same action as whole frame u960 + C1. The interval is a unit bootstrap.

| pipeline | nuScenes | px nusc | AV2 (ego mask) | px av2 | Argoverse-HD | px ahd |
|---|---|---|---|---|---|---|
| draft (cheap) | 0.930 [0.919, 0.941] | 0.043 | 0.840 [0.822, 0.858] | 0.026 | 0.770 [0.712, 0.827] | 0.031 |
| whole frame u640 + C1 | 0.979 [0.973, 0.983] | 0.171 | 0.964 [0.956, 0.971] | 0.103 | 0.958 [0.943, 0.973] | 0.116 |
| whole frame native + C1 | 0.974 [0.966, 0.980] | 1.031 | 0.951 [0.942, 0.960] | 1.012 | 0.922 [0.877, 0.956] | 1.013 |
| v5.1-S | 0.962 [0.954, 0.969] | 0.077 | 0.931 [0.921, 0.941] | 0.053 | 0.885 [0.833, 0.924] | 0.070 |
| v6-S | 0.976 [0.970, 0.982] | 0.078 | 0.968 [0.961, 0.974] | 0.054 | 0.940 [0.892, 0.973] | 0.070 |
| v6-S support only | 0.964 [0.956, 0.972] | 0.050 | 0.929 [0.915, 0.943] | 0.041 | 0.888 [0.835, 0.936] | 0.056 |
| v6-S field only | 0.944 [0.934, 0.952] | 0.068 | 0.880 [0.866, 0.893] | 0.037 | 0.822 [0.766, 0.872] | 0.043 |
| heuristic unc2@S + C1 | 0.964 [0.955, 0.971] | 0.102 | 0.925 [0.912, 0.938] | 0.080 | 0.878 [0.829, 0.922] | 0.123 |
| heuristic tsp2@S + C1 | 0.975 [0.968, 0.980] | 0.099 | 0.953 [0.944, 0.962] | 0.065 | 0.936 [0.890, 0.966] | 0.099 |
| heuristic tsp8@S + C1 | 0.974 [0.967, 0.980] | 0.108 | 0.967 [0.961, 0.974] | 0.071 | 0.939 [0.893, 0.970] | 0.108 |
| v6-S minus the best heuristic + C1 | +0.002 [-0.000, +0.004] (tsp2@S) |  | +0.000 [-0.003, +0.003] (tsp8@S) |  | +0.001 [-0.003, +0.005] (tsp8@S) |  |

**Counterfactual chain from v6-S** (no refiner) towards F = u960 + C1. Each step is cumulative; "share removed" is that step's part of v6-S's disagreement with F.

| step | description | nuScenes agreement | nuScenes share removed | AV2 (ego mask) agreement | AV2 (ego mask) share removed | Argoverse-HD agreement | Argoverse-HD share removed |
|---|---|---|---|---|---|---|---|
| ours | ours (v6-S, no refiner) | 0.976 | – | 0.968 | – | 0.940 | – |
| a | exact locality (F's objects in ours' crops) | 0.984 | 0.31 | 0.980 | 0.37 | 0.953 | 0.21 |
| b | + F's ttc (C1 on F's history) | 0.988 | 0.19 | 0.986 | 0.19 | 0.956 | 0.05 |
| c1 | + closing probe 10 m/s | 0.991 | 0.12 | 0.989 | 0.11 | 0.963 | 0.12 |
| c2 | + closing probe 20 m/s | 0.991 | 0.00 | 0.989 | 0.00 | 0.963 | 0.00 |
| d | + extended grid | 0.992 | 0.06 | 0.993 | 0.11 | 0.968 | 0.08 |
| e | + probe templates | 0.995 | 0.10 | 0.996 | 0.11 | 0.971 | 0.06 |
| f | + round cap 10 | 0.995 | 0.00 | 0.996 | 0.00 | 0.971 | 0.00 |
| residual | disagreement left after f | – | 0.22 | – | 0.11 | – | 0.48 |

| quantity | nuScenes | AV2 (ego mask) | Argoverse-HD |
|---|---|---|---|
| frames | 9181.000 | 11818.000 | 15006.000 |
| disagreement_ours | 0.024 | 0.032 | 0.060 |
| residual_frames | 48.000 | 42.000 | 431.000 |
| residual_F_matches_annotation | 5.000 | 5.000 | 2.000 |
| residual_ours_matches_annotation | 44.000 | 34.000 | 398.000 |
| rounds_0 | 0.419 | 0.337 | 0.273 |
| rounds_1 | 0.539 | 0.557 | 0.586 |
| rounds_2 | 0.041 | 0.103 | 0.135 |
| rounds_3 | 0.001 | 0.003 | 0.006 |
| cap_without_convergence | 0.000 | 0.000 | 0.000 |
| F_raising_objects | 188.000 | 299.000 | 1161.000 |
| coverage_inR | 0.537 | 0.575 | 0.294 |
| coverage_inR_70_80m | 0.727 | 1.000 | 1.000 |
| coverage_inR_F_driver | 0.533 | 0.560 | 0.315 |

**On the residual frames** (disagreement left after step f), the annotation sides with ours far more often than with F:
- nuScenes: ours 44 vs F 5 of 48;
- AV2: ours 34 vs F 5 of 42;
- Argoverse-HD: ours 398 vs F 2 of 431.

## 6. Missed objects (K4)

**A. Natural misses:** frames where the draft misses a decision-relevant annotated object that u960 finds (nuScenes 114 frames, 117 objects; AV2 (ego mask) 360 frames, 396 objects; Argoverse-HD 1019 frames, 1027 objects). Each cell is recovery rate / G on these frames (with refiner).

| pipeline | nuScenes | AV2 (ego mask) | Argoverse-HD |
|---|---|---|---|
| whole frame u960 | 1.00 / 0.768 | 1.00 / 0.341 | 1.00 / 0.521 |
| whole frame u960 + C1 | 1.00 / 0.748 | 1.00 / 0.317 | 1.00 / 0.682 |
| whole frame native + C1 | 0.84 / 0.655 | 0.94 / 0.352 | 0.95 / 0.703 |
| v5.1-S | 0.61 / 0.511 | 0.84 / 0.333 | 0.87 / 0.587 |
| v6-S | 0.61 / 0.506 | 0.85 / 0.321 | 0.87 / 0.680 |
| v6-S support only | 0.20 / -0.124 | 0.38 / -0.122 | 0.20 / -0.005 |
| v6-S field only | 0.45 / 0.557 | 0.54 / 0.166 | 0.74 / 0.606 |
| heuristic unc2@S + C1 | 0.24 / -0.112 | 0.46 / 0.031 | 0.30 / 0.034 |
| heuristic tsp2@S + C1 | 0.56 / 0.493 | 0.74 / 0.315 | 0.78 / 0.629 |
| heuristic tsp8@S + C1 | 0.55 / 0.383 | 0.85 / 0.341 | 0.85 / 0.654 |

**B. Controlled deletion:** each frame's decision-relevant objects are deleted from the draft (nuScenes 791 frames, 823 deleted objects; AV2 (ego mask) 2584 frames, 2923 deleted objects; Argoverse-HD 2541 frames, 2723 deleted objects). Each cell is recovery rate / loss relative to the unmodified draft (with refiner).

| pipeline | nuScenes | AV2 (ego mask) | Argoverse-HD |
|---|---|---|---|
| whole frame u960 | 0.99 / +0.24 | 0.99 / -0.39 | 0.99 / +0.09 |
| whole frame u960 + C1 | 0.99 / +0.25 | 0.99 / -0.44 | 0.99 / -0.08 |
| whole frame native + C1 | 0.98 / +0.16 | 0.98 / -0.32 | 0.99 / -0.13 |
| v5.1-S | 0.85 / +0.55 | 0.87 / +0.14 | 0.85 / +0.22 |
| v6-S | 0.85 / +0.56 | 0.88 / +0.13 | 0.85 / +0.19 |
| v6-S support only | 0.15 / +4.62 | 0.19 / +2.37 | 0.24 / +2.00 |
| v6-S field only | 0.79 / +0.47 | 0.75 / +0.18 | 0.65 / +0.27 |
| heuristic unc2@S + C1 | 0.34 / +3.91 | 0.47 / +1.50 | 0.44 / +1.37 |
| heuristic tsp2@S + C1 | 0.83 / +0.74 | 0.85 / +0.20 | 0.86 / +0.07 |
| heuristic tsp8@S + C1 | 0.83 / +0.57 | 0.90 / +0.10 | 0.91 / +0.07 |

**Recovery by stratum:**

| stratum | nuScenes v6-S | nuScenes u960 + C1 | nuScenes unc2@S + C1 | AV2 (ego mask) v6-S | AV2 (ego mask) u960 + C1 | AV2 (ego mask) unc2@S + C1 | Argoverse-HD v6-S | Argoverse-HD u960 + C1 | Argoverse-HD unc2@S + C1 |
|---|---|---|---|---|---|---|---|---|---|
| range <30 m | 0.83 | 0.98 | 0.33 | 0.86 | 1.00 | 0.53 | 0.81 | 1.00 | 0.48 |
| range 30-50 m | 0.91 | 1.00 | 0.35 | 0.91 | 0.99 | 0.38 | 0.91 | 0.98 | 0.47 |
| range >50 m | 0.93 | 1.00 | 0.36 | 0.91 | 0.98 | 0.41 | 0.92 | 0.98 | 0.21 |
| speed tercile 1 | 0.63 | 1.00 | 0.35 | 0.73 | 1.00 | 0.53 | 0.76 | 1.00 | 0.62 |
| speed tercile 2 | 0.96 | 0.99 | 0.36 | 0.95 | 0.99 | 0.54 | 0.83 | 0.99 | 0.39 |
| speed tercile 3 | 0.97 | 0.97 | 0.30 | 0.93 | 1.00 | 0.36 | 0.95 | 0.99 | 0.29 |

## 7. Regime: braking required and no support object in the draft (K17)

Frames: nuScenes: 312 frames (3.4%) no-support; nuScenes: 384 frames (4.2%) K24; AV2 (ego mask): 352 frames (3.0%) no-support; AV2 (ego mask): 397 frames (3.4%) K24; Argoverse-HD: 855 frames (5.7%) no-support; Argoverse-HD: 861 frames (5.7%) K24. Each cell is G in the regime / G outside it.

| regime | pipeline | nuScenes | AV2 (ego mask) | Argoverse-HD |
|---|---|---|---|---|
| no support in the draft, braking required | whole frame u960 + C1 | 0.409 / -0.194 | 0.414 / 0.320 | 0.684 / 0.234 |
| no support in the draft, braking required | whole frame native + C1 | 0.408 / 0.031 | 0.324 / 0.286 | 0.698 / 0.386 |
| no support in the draft, braking required | v5.1-S | 0.341 / 0.052 | 0.305 / 0.251 | 0.601 / 0.296 |
| no support in the draft, braking required | v6-S | 0.331 / 0.087 | 0.289 / 0.295 | 0.684 / 0.375 |
| no support in the draft, braking required | v6-M | 0.369 / 0.225 | 0.270 / 0.284 | 0.714 / 0.382 |
| no support in the draft, braking required | v6-L | 0.410 / 0.217 | 0.282 / 0.294 | 0.713 / 0.385 |
| no support in the draft, braking required | heuristic unc2@S + C1 | 0.055 / -0.177 | 0.108 / 0.246 | 0.043 / 0.321 |
| no support in the draft, braking required | heuristic tsp8@S + C1 | 0.276 / 0.069 | 0.280 / 0.300 | 0.654 / 0.368 |
| draft below the required action (K24) | whole frame u960 + C1 | 0.384 / -0.315 | 0.443 / 0.294 | 0.673 / 0.235 |
| draft below the required action (K24) | whole frame native + C1 | 0.354 / 0.016 | 0.217 / 0.326 | 0.687 / 0.388 |
| draft below the required action (K24) | v5.1-S | 0.320 / 0.008 | 0.347 / 0.225 | 0.590 / 0.297 |
| draft below the required action (K24) | v6-S | 0.312 / 0.052 | 0.331 / 0.278 | 0.667 / 0.377 |
| draft below the required action (K24) | v6-M | 0.363 / 0.196 | 0.289 / 0.278 | 0.697 / 0.384 |
| draft below the required action (K24) | v6-L | 0.385 / 0.203 | 0.311 / 0.283 | 0.696 / 0.387 |
| draft below the required action (K24) | heuristic unc2@S + C1 | 0.070 / -0.263 | 0.146 / 0.250 | 0.037 / 0.322 |
| draft below the required action (K24) | heuristic tsp8@S + C1 | 0.271 / 0.022 | 0.331 / 0.282 | 0.643 / 0.369 |

## 8. Second consumer: TTC warning (warn ≤ 4 s, brake ≤ 2 s) (K9)

G2 uses the consumer's own loss. "loop − it" is the TTC-warning loop (field + support of this consumer, + C1) minus the row.

| pipeline | nuScenes | AV2 (ego mask) | Argoverse-HD |
|---|---|---|---|
| draft (cheap) | 0.000 (px 0.043); loop − it +0.191 [+0.096, +0.290] | 0.000 (px 0.026); loop − it +0.496 [+0.420, +0.569] | 0.000 (px 0.031); loop − it +0.750 [+0.665, +0.831] |
| whole frame u960 + C1 | 0.157 (px 0.384); loop − it +0.035 [-0.026, +0.112] | 0.448 (px 0.232); loop − it +0.049 [+0.004, +0.105] | 0.326 (px 0.253); loop − it +0.424 [+0.058, +1.040] |
| whole frame native + C1 | 0.222 (px 1.031); loop − it -0.031 [-0.074, +0.014] | 0.499 (px 1.012); loop − it -0.003 [-0.048, +0.043] | 0.629 (px 1.013); loop − it +0.121 [+0.015, +0.327] |
| v5.1-S | 0.142 (px 0.077); loop − it +0.049 [-0.021, +0.116] | 0.439 (px 0.053); loop − it +0.057 [+0.004, +0.106] | 0.406 (px 0.070); loop − it +0.344 [+0.232, +0.461] |
| v6-S | 0.153 (px 0.078); loop − it +0.038 [-0.023, +0.101] | 0.439 (px 0.054); loop − it +0.058 [+0.024, +0.097] | 0.612 (px 0.070); loop − it +0.138 [+0.060, +0.247] |
| heuristic unc2@S + C1 | 0.158 (px 0.102); loop − it +0.034 [-0.036, +0.107] | 0.388 (px 0.080); loop − it +0.108 [+0.068, +0.150] | 0.533 (px 0.123); loop − it +0.217 [+0.141, +0.313] |
| heuristic tsp8@S + C1 | 0.137 (px 0.108); loop − it +0.055 [-0.011, +0.120] | 0.436 (px 0.071); loop − it +0.060 [+0.024, +0.101] | 0.600 (px 0.108); loop − it +0.150 [+0.071, +0.256] |
| ours, TTC-warning loop (field + support of the consumer) + C1 | 0.191 (px 0.086) | 0.496 (px 0.060) | 0.750 (px 0.079) |
| uncertainty crops unc2 + C1 at the TTC loop's tensor px | 0.187 (px 0.116); loop − it +0.004 [-0.031, +0.041] | 0.418 (px 0.092); loop − it +0.079 [+0.048, +0.109] | 0.672 (px 0.143); loop − it +0.077 [+0.045, +0.117] |

## 9. Additive roadmap (test units)

| split | step | G | px | step − previous [95% CI] |
|---|---|---|---|---|
| nuScenes | draft (cheap) | 0.000 | 0.043 | – |
| nuScenes | v3 (previous frozen loop, no refiner) | 0.135 | 0.100 | +0.135 [+0.030, +0.276] |
| nuScenes | + refiner v1 (previous frozen, as reported) | 0.170 | 0.100 | +0.035 [+0.003, +0.078] |
| nuScenes | v5 loop (v_close 0, crops 960; no refiner) | 0.133 | 0.077 | -0.037 [-0.088, +0.004] |
| nuScenes | v5 + refiner v3a | 0.176 | 0.077 | +0.043 [+0.013, +0.083] |
| nuScenes | + T4 robust looming (v5.1-S) | 0.199 | 0.077 | +0.023 [+0.005, +0.046] |
| nuScenes | C1 learned temporal cue instead of T4 (v6-S) | 0.211 | 0.078 | +0.012 [+0.004, +0.026] |
| nuScenes | YOLOv8m crops 768 + own refiner (v6-M) | 0.298 | 0.065 | +0.087 [-0.018, +0.207] |
| nuScenes | YOLO11m crops 960 + own refiner (v6-L) | 0.315 | 0.077 | +0.017 [-0.046, +0.078] |
| AV2 (ego mask) | draft (cheap) | 0.000 | 0.026 | – |
| AV2 (ego mask) | v4-lite = v5 loop (no refiner) | 0.031 | 0.053 | +0.031 [-0.028, +0.096] |
| AV2 (ego mask) | + refiner v2 (previous frozen, as reported) | 0.122 | 0.053 | +0.091 [+0.046, +0.140] |
| AV2 (ego mask) | v5 + refiner v3a | 0.208 | 0.053 | +0.086 [+0.058, +0.117] |
| AV2 (ego mask) | + T4 robust looming (v5.1-S) | 0.262 | 0.053 | +0.054 [+0.012, +0.094] |
| AV2 (ego mask) | C1 learned temporal cue instead of T4 (v6-S) | 0.294 | 0.054 | +0.032 [+0.013, +0.050] |
| AV2 (ego mask) | YOLOv8m crops 768 + own refiner (v6-M) | 0.281 | 0.044 | -0.013 [-0.036, +0.007] |
| AV2 (ego mask) | YOLO11m crops 960 + own refiner (v6-L) | 0.292 | 0.054 | +0.011 [-0.013, +0.034] |
| Argoverse-HD | draft (cheap) | 0.000 | 0.031 | – |
| Argoverse-HD | v4-lite = v5 loop (no refiner) | 0.039 | 0.070 | +0.039 [-0.084, +0.174] |
| Argoverse-HD | + refiner v2 (previous frozen, as reported) | 0.091 | 0.070 | +0.051 [+0.021, +0.084] |
| Argoverse-HD | v5 + refiner v3a | 0.185 | 0.070 | +0.094 [+0.015, +0.199] |
| Argoverse-HD | + T4 robust looming (v5.1-S) | 0.338 | 0.070 | +0.153 [+0.111, +0.210] |
| Argoverse-HD | C1 learned temporal cue instead of T4 (v6-S) | 0.418 | 0.070 | +0.079 [+0.054, +0.107] |
| Argoverse-HD | YOLOv8m crops 768 + own refiner (v6-M) | 0.428 | 0.057 | +0.010 [-0.004, +0.024] |
| Argoverse-HD | YOLO11m crops 960 + own refiner (v6-L) | 0.431 | 0.070 | +0.003 [-0.011, +0.017] |

![roadmap](roadmap_v6.png)

## 10. Cost statistics of v6 (per frame)

| split | configuration | crop detector | px mean / p50 / p95 / max | GFLOPs mean / p95 | no crop | rounds | crops mean / p95 | guard re-det. | detector calls mean / p95 |
|---|---|---|---|---|---|---|---|---|---|
| nuScenes | v6-S | yolov8s | 0.078 / 0.071 / 0.176 / 0.890 | 7.9 / 17.9 | 0.42 | 0.62 | 0.66 / 2 | 0.89 | 1.55 / 6 |
| nuScenes | v6-M | yolov8m | 0.065 / 0.060 / 0.129 / 0.584 | 10.6 / 28.3 | 0.42 | 0.63 | 0.66 / 2 | 0.87 | 1.53 / 6 |
| nuScenes | v6-L | yolo11m | 0.077 / 0.071 / 0.174 / 0.931 | 12.7 / 36.1 | 0.42 | 0.63 | 0.66 / 2 | 0.79 | 1.45 / 5 |
| nuScenes | v5.1-S | yolov8s | 0.077 / 0.071 / 0.176 / 0.890 | 7.8 / 17.8 | 0.42 | 0.63 | 0.66 / 2 | 0.89 | 1.54 / 6 |
| AV2 (ego mask) | v6-S | yolov8s | 0.054 / 0.041 / 0.129 / 0.679 | 12.0 / 28.8 | 0.34 | 0.77 | 0.87 / 2 | 1.38 | 2.25 / 7 |
| AV2 (ego mask) | v6-M | yolov8m | 0.044 / 0.035 / 0.095 / 0.565 | 17.1 / 48.6 | 0.34 | 0.77 | 0.87 / 2 | 1.40 | 2.27 / 7 |
| AV2 (ego mask) | v6-L | yolo11m | 0.054 / 0.041 / 0.133 / 0.834 | 21.0 / 62.7 | 0.34 | 0.78 | 0.88 / 2 | 1.33 | 2.21 / 7 |
| AV2 (ego mask) | v5.1-S | yolov8s | 0.053 / 0.041 / 0.128 / 0.679 | 11.9 / 28.7 | 0.34 | 0.77 | 0.87 / 2 | 1.37 | 2.24 / 7 |
| Argoverse-HD | v6-S | yolov8s | 0.070 / 0.049 / 0.175 / 0.900 | 11.4 / 28.3 | 0.27 | 0.87 | 0.98 / 2 | 1.51 | 2.49 / 8 |
| Argoverse-HD | v6-M | yolov8m | 0.057 / 0.044 / 0.121 / 0.633 | 16.7 / 45.1 | 0.27 | 0.87 | 0.98 / 2 | 1.50 | 2.47 / 8 |
| Argoverse-HD | v6-L | yolo11m | 0.070 / 0.049 / 0.168 / 0.741 | 20.0 / 58.0 | 0.27 | 0.87 | 0.97 / 2 | 1.38 | 2.34 / 7 |
| Argoverse-HD | v5.1-S | yolov8s | 0.070 / 0.048 / 0.173 / 0.692 | 11.3 / 28.1 | 0.27 | 0.87 | 0.97 / 2 | 1.49 | 2.46 / 8 |

**Cost observations:**
- The rounds never hit the cap without converging (attribution detail, cap_without_convergence = 0).
- GFLOPs use the measured gflops_640.csv.

## 11. Detection metrics vs GFLOPs (k28_ap_score.py)

**Argoverse-HD val, COCO AP on the three road-user groups of K7** (crop pipelines: outputs with conf ≥ 0.1):

| pipeline | GFLOPs / v5.1-S | px | AP | AP50 | AP_s | AP_m | AP_l |
|---|---|---|---|---|---|---|---|
| draft | 0.44 | 0.031 | 0.115 | 0.196 | 0.002 | 0.070 | 0.414 |
| v5.1-S | 1.00 | 0.070 | 0.164 | 0.295 | 0.063 | 0.145 | 0.407 |
| v6-S | 1.01 | 0.070 | 0.165 | 0.295 | 0.064 | 0.145 | 0.407 |
| v6-M | 1.48 | 0.057 | 0.160 | 0.295 | 0.048 | 0.148 | 0.417 |
| v6-L | 1.77 | 0.070 | 0.189 | 0.325 | 0.098 | 0.169 | 0.417 |
| whole frame yolov8s 448 | 0.80 | 0.056 | 0.163 | 0.262 | 0.008 | 0.140 | 0.503 |
| whole frame yolov8s 512 | 1.02 | 0.071 | 0.182 | 0.294 | 0.013 | 0.169 | 0.540 |
| whole frame yolov8s 640 | 1.65 | 0.116 | 0.219 | 0.345 | 0.024 | 0.229 | 0.572 |
| whole frame yolo11m 512 | 2.42 | 0.071 | 0.246 | 0.380 | 0.030 | 0.255 | 0.674 |
| whole frame yolov8m 512 | 2.80 | 0.071 | 0.228 | 0.353 | 0.022 | 0.234 | 0.641 |
| whole frame yolov8s 960 | 3.62 | 0.253 | 0.286 | 0.447 | 0.080 | 0.343 | 0.584 |
| whole frame yolo11m 640 | 3.94 | 0.116 | 0.287 | 0.450 | 0.064 | 0.322 | 0.676 |
| whole frame yolov8m 640 | 4.55 | 0.116 | 0.273 | 0.433 | 0.053 | 0.300 | 0.667 |
| whole frame yolov8s 1280 | 6.35 | 0.444 | 0.328 | 0.506 | 0.134 | 0.393 | 0.578 |
| whole frame yolo11m 896 | 7.63 | 0.224 | 0.352 | 0.546 | 0.139 | 0.408 | 0.660 |
| whole frame yolov8m 896 | 8.82 | 0.224 | 0.333 | 0.517 | 0.106 | 0.380 | 0.709 |
| whole frame yolov8s 1536 | 9.15 | 0.640 | 0.349 | 0.542 | 0.170 | 0.413 | 0.567 |
| whole frame yolov8s 1600 | 10.17 | 0.711 | 0.352 | 0.548 | 0.175 | 0.420 | 0.558 |
| whole frame yolov8s 1920 | 14.49 | 1.013 | 0.368 | 0.573 | 0.210 | 0.423 | 0.562 |
| whole frame yolo11m 1280 | 15.14 | 0.444 | 0.380 | 0.576 | 0.198 | 0.438 | 0.625 |
| whole frame yolov8m 1280 | 17.49 | 0.444 | 0.383 | 0.582 | 0.175 | 0.444 | 0.696 |
| whole frame yolo11m 1536 | 21.80 | 0.640 | 0.386 | 0.578 | 0.218 | 0.447 | 0.592 |
| whole frame yolov8m 1536 | 25.19 | 0.640 | 0.401 | 0.607 | 0.211 | 0.463 | 0.670 |
| whole frame yolo11m 1920 | 34.52 | 1.013 | 0.389 | 0.577 | 0.244 | 0.443 | 0.565 |
| whole frame yolov8m 1920 | 39.89 | 1.013 | 0.416 | 0.622 | 0.251 | 0.472 | 0.648 |

**nuScenes new test** (det_metrics.py: every 4th frame, conf ≥ 0.25; relevant = decision-relevant objects):

| pipeline | GFLOPs / v5.1-S | px | AP50 | R all / small / relevant | FP / frame | P relevant | relevant FP / 1k frames |
|---|---|---|---|---|---|---|---|
| draft | 0.55 | 0.043 | 0.155 | 0.324 / 0.030 / 0.728 | 1.31 | 0.762 | 30.6 |
| v5.1-S | 1.00 | 0.077 | 0.177 | 0.363 / 0.195 / 0.786 | 1.71 | 0.872 | 13.6 |
| v6-S | 1.00 | 0.078 | 0.177 | 0.363 / 0.195 / 0.786 | 1.71 | 0.890 | 10.6 |
| v6-M | 1.35 | 0.065 | 0.178 | 0.365 / 0.188 / 0.803 | 1.65 | 0.878 | 11.9 |
| v6-L | 1.62 | 0.077 | 0.182 | 0.363 / 0.154 / 0.800 | 1.53 | 0.880 | 11.9 |
| whole frame yolov8s 448 | 1.03 | 0.080 | 0.206 | 0.407 / 0.177 / 0.752 | 1.80 | 0.773 | 31.4 |
| whole frame yolov8s 512 | 1.32 | 0.102 | 0.216 | 0.422 / 0.204 / 0.762 | 1.97 | 0.758 | 34.0 |
| whole frame yolov8s 640 | 2.21 | 0.171 | 0.240 | 0.466 / 0.319 / 0.779 | 2.66 | 0.751 | 36.1 |
| whole frame yolo11m 448 | 2.45 | 0.080 | 0.244 | 0.451 / 0.290 / 0.814 | 1.93 | 0.812 | 25.5 |
| whole frame yolov8m 448 | 2.83 | 0.080 | 0.241 | 0.448 / 0.276 / 0.772 | 2.12 | 0.765 | 32.7 |
| whole frame yolov8s 960 | 4.96 | 0.384 | 0.284 | 0.513 / 0.446 / 0.824 | 3.21 | 0.736 | 39.5 |
| whole frame yolo11m 640 | 5.26 | 0.171 | 0.274 | 0.500 / 0.403 / 0.821 | 2.73 | 0.775 | 33.1 |
| whole frame yolov8m 640 | 6.07 | 0.171 | 0.277 | 0.504 / 0.430 / 0.810 | 2.97 | 0.763 | 36.1 |
| whole frame yolo11m 768 | 7.36 | 0.239 | 0.290 | 0.518 / 0.450 / 0.834 | 2.95 | 0.751 | 36.1 |
| whole frame yolov8s 1280 | 8.46 | 0.654 | 0.296 | 0.526 / 0.518 / 0.821 | 3.64 | 0.743 | 37.8 |
| whole frame yolov8m 768 | 8.50 | 0.239 | 0.299 | 0.527 / 0.471 / 0.831 | 3.26 | 0.801 | 29.7 |
| whole frame yolo11m 960 | 11.83 | 0.384 | 0.306 | 0.540 / 0.509 / 0.834 | 3.59 | 0.770 | 35.2 |
| whole frame yolov8s 1536 | 11.91 | 0.922 | 0.294 | 0.525 / 0.518 / 0.807 | 3.93 | 0.746 | 38.2 |
| whole frame yolov8s 1600 | 13.33 | 1.031 | 0.298 | 0.524 / 0.518 / 0.807 | 3.92 | 0.719 | 40.7 |
| whole frame yolov8m 960 | 13.66 | 0.384 | 0.300 | 0.539 / 0.477 / 0.834 | 3.50 | 0.769 | 34.4 |
| whole frame yolo11m 1280 | 20.15 | 0.654 | 0.305 | 0.541 / 0.513 / 0.834 | 3.60 | 0.745 | 39.0 |
| whole frame yolov8m 1280 | 23.28 | 0.654 | 0.317 | 0.551 / 0.530 / 0.831 | 3.84 | 0.764 | 34.8 |
| whole frame yolo11m 1600 | 31.75 | 1.031 | 0.288 | 0.522 / 0.500 / 0.821 | 3.49 | 0.774 | 32.3 |
| whole frame yolov8m 1600 | 36.69 | 1.031 | 0.313 | 0.548 / 0.554 / 0.831 | 4.13 | 0.740 | 39.9 |

**Parity of the AP dumps with the phase-2 pass:** 54 columns, min identical share 1.0000.

![detection](detection_vs_gflops_v6.png)

## 12. Loss terms, event rates, collisions and sign tests

**Checks.** max |J of v6-S rebuilt from the loss terms − stored J| = 0.0 / 0.0 / 0.0.

| split | pipeline | G | insufficient / excessive / changes / collisions (share of cheap loss) | collision frames (units) | unnecessary / missed per 1k | missed hard brake / delayed hard brake / unnecessary hard brake per 1k | G multi-obstacle |
|---|---|---|---|---|---|---|---|
| nuScenes | cheap draft | 0.000 | 0.782 / 0.191 / 0.014 / 0.013 | 11 (4) | 50.6 / 41.8 | 3.70 / 7.84 / 18.52 | 0.000 |
| nuScenes | v5.1-S | with refiner | 0.199 | 0.688 / 0.091 / 0.011 / 0.011 | 9 (4) | 21.7 / 37.3 | 3.27 / 1.74 / 7.62 | 0.002 |
| nuScenes | v6-S | no refiner | 0.181 | 0.730 / 0.068 / 0.011 / 0.011 | 9 (4) | 21.6 / 42.5 | 3.27 / 7.62 / 4.79 | 0.007 |
| nuScenes | v6-S | with refiner | 0.211 | 0.696 / 0.071 / 0.011 / 0.011 | 9 (4) | 15.2 / 38.0 | 3.27 / 1.74 / 5.45 | 0.019 |
| nuScenes | v6-M | with refiner | 0.298 | 0.614 / 0.071 / 0.011 / 0.006 | 5 (3) | 15.1 / 37.3 | 2.40 / 1.96 / 5.23 | 0.216 |
| nuScenes | v6-L | with refiner | 0.315 | 0.597 / 0.073 / 0.011 / 0.005 | 4 (3) | 16.0 / 36.5 | 2.29 / 1.85 / 5.23 | 0.222 |
| nuScenes | whole frame u960 + components | with refiner | 0.096 | 0.751 / 0.119 / 0.012 / 0.021 | 17 (3) | 29.1 / 32.0 | 4.03 / 1.85 / 10.57 | -0.103 |
| nuScenes | whole frame native + components | with refiner | 0.197 | 0.684 / 0.098 / 0.013 / 0.009 | 7 (4) | 25.2 / 32.9 | 3.59 / 2.07 / 8.06 | 0.080 |
| AV2 (ego mask) | cheap draft | 0.000 | 0.443 / 0.534 / 0.020 / 0.003 | 4 (3) | 197.0 / 33.6 | 2.45 / 3.81 / 65.83 | 0.000 |
| AV2 (ego mask) | v5.1-S | with refiner | 0.262 | 0.472 / 0.243 / 0.020 / 0.003 | 5 (3) | 76.5 / 34.5 | 2.71 / 3.05 / 29.28 | 0.209 |
| AV2 (ego mask) | v6-S | no refiner | 0.241 | 0.440 / 0.298 / 0.017 / 0.003 | 5 (3) | 138.0 / 29.1 | 2.12 / 4.82 / 30.46 | 0.226 |
| AV2 (ego mask) | v6-S | with refiner | 0.294 | 0.498 / 0.187 / 0.019 / 0.003 | 5 (3) | 52.5 / 38.8 | 2.54 / 3.89 / 22.59 | 0.244 |
| AV2 (ego mask) | v6-M | with refiner | 0.281 | 0.509 / 0.188 / 0.019 / 0.003 | 5 (3) | 51.2 / 38.7 | 2.62 / 4.48 / 23.02 | 0.233 |
| AV2 (ego mask) | v6-L | with refiner | 0.292 | 0.499 / 0.187 / 0.019 / 0.003 | 5 (3) | 51.1 / 37.7 | 2.37 / 4.57 / 22.93 | 0.238 |
| AV2 (ego mask) | whole frame u960 + components | with refiner | 0.255 | 0.421 / 0.302 / 0.021 / 0.002 | 3 (2) | 96.1 / 30.6 | 1.95 / 3.64 / 38.16 | 0.187 |
| AV2 (ego mask) | whole frame native + components | with refiner | 0.224 | 0.477 / 0.273 / 0.021 / 0.005 | 7 (4) | 90.9 / 31.7 | 2.45 / 4.23 / 33.42 | 0.130 |
| Argoverse-HD | cheap draft | 0.000 | 0.192 / 0.793 / 0.015 / 0.000 | 0 (0) | 303.8 / 57.4 | 0.33 / 0.40 / 149.94 | 0.000 |
| Argoverse-HD | v5.1-S | with refiner | 0.338 | 0.158 / 0.484 / 0.019 / 0.000 | 0 (0) | 169.1 / 45.6 | 0.53 / 0.93 / 91.90 | 0.238 |
| Argoverse-HD | v6-S | no refiner | 0.422 | 0.110 / 0.452 / 0.016 / 0.000 | 0 (0) | 202.9 / 23.7 | 0.40 / 1.67 / 77.57 | 0.418 |
| Argoverse-HD | v6-S | with refiner | 0.418 | 0.184 / 0.379 / 0.019 / 0.000 | 0 (0) | 130.8 / 51.8 | 0.60 / 1.87 / 72.30 | 0.344 |
| Argoverse-HD | v6-M | with refiner | 0.428 | 0.172 / 0.381 / 0.019 / 0.000 | 0 (0) | 133.9 / 44.9 | 0.60 / 1.87 / 71.50 | 0.352 |
| Argoverse-HD | v6-L | with refiner | 0.431 | 0.168 / 0.382 / 0.019 / 0.000 | 0 (0) | 131.3 / 42.8 | 0.60 / 1.93 / 72.50 | 0.359 |
| Argoverse-HD | whole frame u960 + components | with refiner | 0.287 | 0.156 / 0.537 / 0.020 / 0.000 | 0 (0) | 187.9 / 45.0 | 0.33 / 1.47 / 104.03 | 0.180 |
| Argoverse-HD | whole frame native + components | with refiner | 0.283 | 0.145 / 0.550 / 0.022 / 0.000 | 0 (0) | 198.7 / 41.0 | 0.33 / 1.13 / 109.02 | 0.141 |

**Per-unit sign tests of v6-S + v3a.** Units where v6-S has lower / equal / higher total loss:

| split | against | units | v6-S lower loss | equal | v6-S higher loss | sign-test p |
|---|---|---|---|---|---|---|
| nuScenes | cheap draft | 228 | 120 | 80 | 28 | 0.0000 |
| nuScenes | v6-M | with refiner | 228 | 25 | 183 | 20 | 0.5515 |
| nuScenes | v6-L | with refiner | 228 | 22 | 180 | 26 | 0.6655 |
| nuScenes | v5.1-S | with refiner | 228 | 27 | 184 | 17 | 0.1742 |
| nuScenes | whole frame u960 + components | with refiner | 228 | 55 | 130 | 43 | 0.2664 |
| nuScenes | whole frame native + components | with refiner | 228 | 45 | 125 | 58 | 0.2369 |
| nuScenes | oracle frame router (true V, top-k) @ v5.1-S px + components | with refiner | 228 | 27 | 130 | 71 | 0.0000 |
| nuScenes | best decision-value router (picked on val): cond_V @ v5.1-S px | with refiner | 228 | 111 | 88 | 29 | 0.0000 |
| nuScenes | best decision-value router (picked on val): cond_V @ v5.1-S px + components | with refiner | 228 | 110 | 89 | 29 | 0.0000 |
| AV2 (ego mask) | cheap draft | 150 | 119 | 9 | 22 | 0.0000 |
| AV2 (ego mask) | v6-M | with refiner | 150 | 40 | 56 | 54 | 0.1797 |
| AV2 (ego mask) | v6-L | with refiner | 150 | 41 | 52 | 57 | 0.1293 |
| AV2 (ego mask) | v5.1-S | with refiner | 150 | 70 | 61 | 19 | 0.0000 |
| AV2 (ego mask) | whole frame u960 + components | with refiner | 150 | 97 | 15 | 38 | 0.0000 |
| AV2 (ego mask) | whole frame native + components | with refiner | 150 | 85 | 12 | 53 | 0.0081 |
| AV2 (ego mask) | oracle frame router (true V, top-k) @ v5.1-S px + components | with refiner | 150 | 60 | 14 | 76 | 0.1982 |
| AV2 (ego mask) | best decision-value router (picked on val): cond_V @ v5.1-S px | with refiner | 150 | 123 | 9 | 18 | 0.0000 |
| AV2 (ego mask) | best decision-value router (picked on val): cond_V @ v5.1-S px + components | with refiner | 150 | 120 | 9 | 21 | 0.0000 |
| Argoverse-HD | cheap draft | 24 | 22 | 1 | 1 | 0.0000 |
| Argoverse-HD | v6-M | with refiner | 24 | 8 | 5 | 11 | 0.6476 |
| Argoverse-HD | v6-L | with refiner | 24 | 9 | 4 | 11 | 0.8238 |
| Argoverse-HD | v5.1-S | with refiner | 24 | 20 | 3 | 1 | 0.0000 |
| Argoverse-HD | whole frame u960 + components | with refiner | 24 | 19 | 3 | 2 | 0.0002 |
| Argoverse-HD | whole frame native + components | with refiner | 24 | 15 | 2 | 7 | 0.1338 |
| Argoverse-HD | oracle frame router (true V, top-k) @ v5.1-S px + components | with refiner | 24 | 15 | 2 | 7 | 0.1338 |

The full tables (every row, every event with v6-S-minus intervals) are loss_terms_*_v6.csv, sign_tests_v6.csv and collisions_v6.csv.

## 13. Failure cases (frames where v6-S + v3a costs more than u960 + C1 + v3a)

| split | category | frames | share of frames | excess loss / cheap loss |
|---|---|---|---|---|
| nuScenes | missed: a relevant object u960 finds is not in ours' scene | 22 | 0.0024 | 0.0721 |
| nuScenes | other (same action class as required, switching cost) | 9 | 0.0010 | 0.0007 |
| nuScenes | under-reaction with the relevant objects in ours' scene (range / ttc) | 40 | 0.0044 | 0.0507 |
| nuScenes | unnecessary action (ours above the required action) | 9 | 0.0010 | 0.0020 |
| AV2 (ego mask) | missed: a relevant object u960 finds is not in ours' scene | 4 | 0.0003 | 0.0008 |
| AV2 (ego mask) | other (same action class as required, switching cost) | 25 | 0.0021 | 0.0009 |
| AV2 (ego mask) | under-reaction with the relevant objects in ours' scene (range / ttc) | 77 | 0.0065 | 0.0862 |
| AV2 (ego mask) | unnecessary action (ours above the required action) | 49 | 0.0041 | 0.0092 |
| Argoverse-HD | missed: a relevant object u960 finds is not in ours' scene | 22 | 0.0015 | 0.0033 |
| Argoverse-HD | other (same action class as required, switching cost) | 52 | 0.0035 | 0.0013 |
| Argoverse-HD | under-reaction with the relevant objects in ours' scene (range / ttc) | 55 | 0.0037 | 0.0046 |
| Argoverse-HD | unnecessary action (ours above the required action) | 42 | 0.0028 | 0.0067 |

**Panels:**
- The five largest frames per dataset and category are in failure_frames_v6.csv.
- The panels (K8 frames, K15 frames and two failure frames per category: one on Argoverse-HD) come from the capture run and go into the consolidated report.

## 14. K6 (Bench2Drive) and NAVSIM: does the perception configuration change?

- **NAVSIM: no.**
  - Its ours is spec_pdm: probing PDM-Closed itself for field + support, crops at 0.5, one round, v3's guard / merge, no refiner, no looming estimate (01_method.md).
  - K28 changed only the braking-field loop: C1 instead of T4, refiners, and the crop detector at M / L. None of these enters spec_pdm. No re-run.
- **K6 Bench2Drive: yes.**
  - K6's ours is nuScenes v3: v_close 10 m/s, crops at 0.6 × native, no refiner.
  - v6-S differs: v_close 0, crops 960, and C1 in the field, which changes which crops are taken.
  - Under K6's protocol, matched actors keep their true state, so refiner and range effects cannot act. Only the crop choice and thus the detection set can change.
  - All K6 perception branches (cheap, u960, native, ours) tie within run-to-run noise (±0.6–0.9 DS). The expected effect is small.
  - A re-run of the ours branch with v6-S (220 routes × 3 runs, CARLA on VUW) is needed only if the paper labels the K6 row as the final method. Otherwise the row should read "v3 (previous frozen nuScenes configuration)".
  - **Launched 2026-10-14** (user: "Còn gì trên VUW thì làm nốt cho xong đi").
    - Setup: branch ours_v6 (perception_pilot.py on branch ours_v6, md5 23539df7), K6's route protocol, 3 runs (results/b2d/{full,rep2,rep3}/ours_v6/).
    - Route 11755 hangs the simulator for every branch except cheap. K6 ran run 1 as chunk0 (through 4937) + chunk0b (26 routes), and runs 2 / 3 have no 11755. The first v6 launcher still ran 11755 in chunk0 and hung there. It was stopped (exact PIDs) and continued with b2d_v6_runs2.sh (chunks 0b 1 2 3). No finished route was lost.
    - Summary script: k28_b2d_v6_summary.py (md5 15476b88), as b2d_repeats_summary.py. ours_v6 is merged with the official tool, and the K6 branches are read from their existing merged.json without re-merging. Only runs where ours_v6 has all 219 routes are compared. Per-route values are averaged over those runs, then a paired route bootstrap (2,000, seed 0) is taken.

### 14a. Three runs (DONE 2026-10-15)

Every branch has 219 routes in each of runs 1–3. The interval is ours_v6 − branch, by paired route bootstrap on the per-route DS and success averaged over the 3 runs (2,000 resamples, seed 0). The K6 branches are K6's own runs, read from their merged.json; they reproduce the K6 report (privileged 96.63 ± 0.17, cheap 92.27 ± 0.92, u960 93.15 ± 0.74, u1600 91.95 ± 0.71, ours 92.80 ± 0.63).

| branch | DS mean ± std | DS range (runs) | SR mean ± std | RC mean | failed routes per run | v6-S − branch, DS | v6-S − branch, SR |
|---|---|---|---|---|---|---|---|
| privileged | 96.63 ± 0.17 | 96.44–96.74 | 0.915 ± 0.003 | 98.53 | 5–7 | **−4.09 [−6.36, −2.02]** | **−0.076 [−0.117, −0.038]** |
| cheap | 92.27 ± 0.92 | 91.22–92.89 | 0.834 ± 0.018 | 97.59 | 9–12 | +0.27 [−0.96, +1.48] | +0.005 [−0.020, +0.029] |
| u960 | 93.15 ± 0.74 | 92.46–93.93 | 0.857 ± 0.010 | 96.92 | 10–14 | −0.61 [−2.20, +1.03] | −0.018 [−0.050, +0.014] |
| u1600 | 91.95 ± 0.71 | 91.14–92.44 | 0.837 ± 0.013 | 96.94 | 11–12 | +0.59 [−1.37, +2.42] | +0.002 [−0.033, +0.033] |
| ours (K6, v3) | 92.80 ± 0.63 | 92.07–93.18 | 0.845 ± 0.012 | 97.65 | 8–13 | −0.26 [−1.23, +0.61] | −0.006 [−0.024, +0.011] |
| **ours_v6 (v6-S)** | **92.54 ± 0.82** | 91.60–93.08 | **0.839 ± 0.011** | 97.62 | 10–12 | — | — |

- **Verdict: v6-S ties every perception branch in closed loop and is below privileged.** This is K6's null result again.
  - Every interval against cheap, u960, u1600 and K6's v3 crosses 0.
  - v6-S ranks 3rd of the 5 deployable branches by mean DS, behind u960 and v3.
  - Its run-to-run std (0.82 DS) is the size of the largest difference with a deployable branch (−0.61 against u960).
  - Privileged is above v6-S in DS and SR. That gap (−4.09 DS) is larger than run 1 alone suggested (−3.49).
- **Perception per frame, mean of 3 runs:**

  | branch | in camera | hidden | phantoms | crops |
  |---|---|---|---|---|
  | v6-S | 3.08 | 1.24 | 0.87 | 0.81 |
  | K6 v3 | 3.05 | 1.16 | 0.84 | 1.15 |
  | cheap | 3.18 | 1.28 | 0.83 | — |
  | u960 | 3.25 | 0.85 | 1.26 | — |

  v6-S takes 29% fewer crops than v3 (crops at 960, v_close 0), with no measurable change in DS.
- **Operations.**
  - CARLA crashes recurred ("couldn't spawn the ego vehicle", the 600 s simulator time-out).
  - The dispatcher re-ran each crashed route with RESUME (up to 3 attempts per chunk in runs 2 / 3), and no route was lost.
  - Failed routes ("Failed - TickRuntime" and similar route outcomes) number 10–12 per run, within the 5–14 range of the K6 branches.
- **Files:** b2d_v6_per_route.csv, b2d_v6_per_run.csv, b2d_v6_summary.csv, in handoff/K28/phase3/ and results/vuw_summaries/k28/phase3/ (log k28_b2d_v6_summary.log; k28_b2d_v6_summary.py md5 15476b88). The run-1 files (*_run1.csv) are kept as the interim record.

## 15. Parity and record

**Parity:**
- Phase-3 instr rows vs the phase-2 pass: 171 columns (Argoverse-HD 57, AV2 (ego mask) 57, nuScenes 57); min identical share 1.0000.
- AP dumps: 54 columns, min identical share 1.0000.
- Loss-term rebuild: max |ΔJ| = 0.

**Code (md5):**
- k28_p3.py 66dad0fc; k28_p3_chain.sh f3687b9a; k28_render.py 8b496bf3;
- k28_p3_analyze.py bd7035fd (a208656c before the 2026-10-13 ctable fix); k28_p3_p0.py 5d726b33 (d3f19747 before the merge fix);
- k28_p3_nomask_cm.py cb1ac41e (new, post-hoc, for §3);
- k28_ap.py d095f777; k28_ap_score.py.

**Runs:**
- VUW 2026-10-13 16:36 → 2026-10-14 05:59 ("K28 P3 RUNS DONE"); analysis 06:10 ("K28 P3 ANALYZE DONE").
- AP dump 18:49 and AP scoring 19:28 VUW, 2026-10-13.

**Operational:**
- No shard died in phase 3.
- A dry run of both analysis scripts (tables_dry, 2026-10-14 05:30 VUW) passed before the chain's run. Only errors and row counts were looked at.

## 16. Jetson final window (Jetson AGX Xavier, MAXN, TensorRT FP16; 61_k28_v6_latency.py, 62_k28_s3_front.py)

**Window.**
- 2026-10-14 00:48–07:28 +07, exclusive (DEEP paused).
- The board hard-reset twice:
  - 01:40, during the YOLOv8s 864×1536 engine build. That build was not retried, so nuScenes whole frame u1536 is not timed. AV2 u1536 uses the 1536×1184 engine.
  - 04:35, during a nuScenes run with 6 whole-frame engines loaded (RAM 91–92%, 68 °C, fan in the quiet profile at 0 rpm). No OOM / thermal message.
- From 04:37 the fan was at 100% (user's instruction) and the whole-frame engines were split over two processes per dataset. No further reset; 0 throttle steps at the end of the window.
- One code fix after the runs: the Argoverse-HD summary step read the label column as `dd.gt`, which pandas resolves to the DataFrame method. It is now `dd["gt"]`, and the summary is recomputed from the saved per-frame pickle (`--post_only`, md5 8c2a2223 → 6341d349).
- Regression check: recomputing an AV2 run with `--post_only` reproduces its latency.csv and parity.csv exactly.

**Timed frames.** Every pipeline of a dataset is timed on the same frames; 200 warm-up frames are dropped.
- **nuScenes:** 20 scenes, 797 frames. These are the old DEEP scenes on the Jetson, not new-test scenes, so there is no parity with phase 2. Their G values are lower than on test (v6-S 0.096 here vs 0.211 on test).
- **AV2 with the ego mask:** 8 local val logs, all test units, 586 frames. Parity with the phase-2 pass (VUW, fp32 torch):
  - identical per-frame cost on 94–96% of frames;
  - G on the timed frames within 0.001 (v6-S), 0.004 (v6-M), 0.008 (v6-L) and −0.013 (v5.1-S) of the VUW value.
- **KITTI:** DEEP test split, 6 sequences, 3,336 frames.
- **Argoverse-HD:** all 24 val sequences, 15,062 timed frames (15,006 with 3D labels for G).

### S3: latency-G front (s3_front.csv, s3_dominance.csv, s3_front.pdf / .png)

Every timed pipeline enters the front:
- cheap;
- whole frame with YOLOv8s at every timed side, and with YOLOv8m / YOLO11m at 448 / 640 on nuScenes;
- whole frame + T4 + refiner v3a, and whole frame + C1 + refiner v3a;
- v5.1-S;
- v6-S / M / L, with and without refiner, and bounded to ≤ 2 (and ≤ 3) rounds.

A row is off the front if another row has latency ≤ and G ≥, one of them strictly. Intervals are for G(ours) − G(row): paired unit bootstraps, 2,000 resamples, seed 0.

| dataset | pipeline | G | mean ms | p95 ms | px | on front (mean / p95) |
|---|---|---|---|---|---|---|
| nuScenes | **v6-S** | 0.096 | 29.5 | 78.4 | 0.081 | **no / no** |
| nuScenes | v6-S, no refiner | 0.116 | 29.0 | 78.0 | 0.081 | no / no |
| nuScenes | v6-S, ≤ 2 rounds | 0.098 | 26.4 | 64.1 | 0.084 | no / no |
| nuScenes | v6-S, ≤ 2 rounds, no refiner | 0.126 | 25.9 | 63.2 | 0.084 | yes / yes |
| nuScenes | v6-M | 0.098 | 31.8 | 84.1 | 0.067 | no / no |
| nuScenes | v6-L | 0.115 | 33.3 | 86.5 | 0.081 | no / no |
| nuScenes | v6-L, ≤ 2 rounds, no refiner | 0.135 | 30.2 | 72.9 | 0.082 | yes / yes |
| nuScenes | v5.1-S | 0.080 | 30.1 | 67.1 | 0.081 | no / no |
| nuScenes | whole frame u448 + T4 + refiner | 0.084 | 10.4 | 15.1 | 0.080 | yes / yes |
| nuScenes | whole frame u960 + T4 + refiner | 0.087 | 19.5 | 21.5 | 0.384 | yes / yes |
| nuScenes | whole frame u1280 + T4 + refiner | 0.118 | 25.6 | 29.9 | 0.654 | yes / yes |
| nuScenes | whole frame u640 + C1 + refiner | 0.099 | 27.0 | 56.1 | 0.171 | no / no |
| nuScenes | whole frame native + C1 + refiner | 0.130 | 50.1 | 75.1 | 1.031 | no / no |
| AV2 | **v6-S** | 0.419 | 73.9 | 173.0 | 0.058 | **yes / yes** |
| AV2 | v6-S, no refiner | 0.374 | 70.4 | 168.0 | 0.058 | no / no |
| AV2 | v6-S, ≤ 2 rounds | 0.423 | 81.3 | 202.2 | 0.070 | yes / no |
| AV2 | v6-M | 0.424 | 82.5 | 194.0 | 0.047 | yes / no |
| AV2 | v6-L | 0.443 | 83.5 | 189.8 | 0.059 | yes / yes |
| AV2 | v5.1-S | 0.397 | 52.6 | 113.2 | 0.057 | no / no |
| AV2 | v5.1-S, ≤ 2 rounds | 0.401 | 46.4 | 92.9 | 0.070 | yes / yes |
| AV2 | whole frame u512 + T4 + refiner | 0.283 | 19.0 | 27.0 | 0.067 | yes / yes |
| AV2 | whole frame u1536 + T4 + refiner | 0.350 | 46.7 | 55.0 | 0.573 | no / yes |
| AV2 | whole frame native + T4 + refiner | 0.388 | 70.3 | 77.8 | 1.012 | no / yes |
| AV2 | whole frame native + C1 + refiner | 0.410 | 145.6 | 213.2 | 1.012 | no / no |

**v6-S against the faster rows** (G difference, v6-S − row):
- **AV2:**
  - native + T4 + refiner: +0.032 [−0.059, +0.124];
  - u1536 + T4 + refiner: +0.069 [+0.026, +0.128];
  - v5.1-S ≤ 2 rounds: +0.018 [−0.007, +0.042];
  - every T4 row at ≤ 1280: outperforms.
- **nuScenes:**
  - u1280 + T4 + refiner: −0.022 [−0.149, +0.068];
  - u640 + C1 + refiner: −0.003 [−0.054, +0.041];
  - u448 + T4 + refiner: +0.012 [−0.044, +0.093].
  - Within ours, v6-S ≤ 2 rounds without refiner is above v6-S: +0.030, interval for v6-S − it [−0.081, −0.011].

**Verdict: S3 not met.**
- v6-S is on the AV2 front in mean and p95.
- On nuScenes it is on neither: its G is not significantly different from the faster rows, and the refiner lowers G on these 20 scenes.
- The two nuScenes front points that belong to us (v6-S / v6-L, ≤ 2 rounds, no refiner) are post-hoc variants, not the frozen v6.

**Where the time goes.**
- **C1 lift.** On nuScenes, whole frame u448 + T4 + refiner takes 10.4 ms mean / 15.1 p95; + C1 + refiner takes 22.4 / 51.6. On AV2 the C1 rows take 100–146 ms mean, more than v6-S itself (u512: + T4 19.0 ms, + C1 135.8 ms).
  - The C1 head runs as torch fp32 with per-object chains (not TensorRT). Its cost on the C1 rows is measured, not profiled; that it scales with the number of whole-frame boxes is plausible but not verified.
- **Cost model** (cost_model.csv; per-frame least squares, ms = a + b·verifier calls + c·tensor px):
  - nuScenes: R² 0.93–0.95, 9.7–11.2 ms per call; mean 2.9 calls, p95 7.
  - AV2: R² 0.78–0.89, 15–19 ms per call; mean 4.5 calls, p95 10–11.
  - The px coefficient is not stable: it is collinear with calls, and negative for v6-M on AV2. **The p95 latency is set by the number of verifier calls.**
- **Bounded variant (≤ 2 rounds).** It cuts nuScenes p95 from 78 to 64 ms. On AV2 it raises both mean and p95 (81 / 202 vs 74 / 173 ms) because it uses more px (0.070 vs 0.058).

### KITTI with the shared v6-S configuration (kitti_v6S_latency.csv, kitti_v6S_cells.csv)

Configuration: crop long side scaled by 1242; refiner v3a for every pipeline; DEEP's 4 cells; sequence bootstrap.

| pipeline | brake mono G | brake oracle | Planner B mono | Planner B oracle | mean of 4 | v6-S − row, mean of 4 | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| v6-S | 0.144 | 0.271 | −0.022 | 0.549 | 0.235 | | 38.4 | 111.8 |
| v6-S, ≤ 2 rounds | 0.156 | 0.278 | −0.019 | 0.549 | 0.241 | [−0.022, +0.004] | 35.4 | 94.9 |
| whole frame native (1248) | 0.319 | 0.468 | −0.023 | 0.579 | 0.336 | [−0.216, +0.056] | 15.4 | 16.4 |
| native + T4 + refiner | 0.371 | 0.503 | +0.025 | 0.579 | 0.370 | **[−0.182, −0.030]** | 18.3 | 22.2 |
| native + C1 + refiner | 0.288 | 0.540 | +0.011 | 0.579 | 0.354 | [−0.183, −0.025] | 63.1 | 112.0 |
| u960 + T4 + refiner | 0.254 | 0.364 | −0.007 | 0.563 | 0.294 | [−0.105, −0.012] | 15.7 | 19.8 |

- **v6-S is below whole frame + components on KITTI**, and slower: 38 ms vs 18 ms mean.
- KITTI frames are small (1242×375), so native whole frame costs 15 ms.
- **Discrepancy with the kit's KITTI section (03_results §6): cause found 2026-10-15. The refiner rows of this table are affected; a re-run is pending.**
  - **The discrepancy.** There, the refiner lifted every pipeline on Planner B mono (≈ −0.02 → ≈ 0.44). Here Planner B mono stays at −0.02 … +0.03 for every pipeline.
  - **Cause: the camera flag fed to refiner v3a.**
    - v3a's 13th feature is a camera flag: 0.0 for KITTI and 1.0 for the nuScenes / AV2 cameras (k23_train_refiner.load).
    - This run's refiner (58.RefinerBatch, used by 61 for every dataset) hard-codes 1.0, so v3a treated the KITTI crops as nuScenes / AV2 crops.
    - The K1 runs of §6 fed 0.0 (40_e2e_v2.TRTRefiner).
  - **Check (CPU, VUW; remote/A_method/k28_kitti_flag_check.py, md5 7359eb1b; kitti_refiner_flag_check.csv).** v3a on KITTI boxes of its own training / selection splits (no test unit), mean |log(annotated / estimated range)|:

    | KITTI split | lift | v3a, flag 0.0 (as trained) | v3a, flag 1.0 (as run here) | boxes corrected the wrong way, 0.0 / 1.0 |
    |---|---|---|---|---|
    | val (4,825 boxes) | 0.210 | 0.055 | 0.084 | 5.7% / 17.5% |
    | train (14,605) | 0.146 | 0.010 | 0.070 | 2.6% / 21.0% |

    - **With flag 1.0 the mean correction falls short.** Mean predicted log-ratio vs target: val −0.144 vs −0.188 (flag 0.0: −0.185); train −0.036 vs −0.098 (flag 0.0: −0.095).
    - Ranges therefore stay over-estimated, which is what Planner B mono is sensitive to.
  - **What is affected:**
    - every "+ refiner" row of this KITTI table (v6-S, v6-S ≤ 2, native / u960 + T4 + refiner, native + C1 + refiner);
    - hence the KITTI shared-configuration verdict "below whole frame + components" (both sides use the refiner).
    - Plain whole frame native is not affected.
    - nuScenes, AV2 and Argoverse-HD are not affected: their cameras are trained with flag 1.0.
  - **Not yet shown:** that flag 0.0 restores the Planner B mono lift on the test cells. That needs the Jetson re-run of this table with the flag set per dataset (≈ 20 min; the Jetson is paused for a power-supply problem, see handoff/K29). Until then, the KITTI shared-configuration numbers with refiner should not be used.
  - The K29 fast refiner (c1fast.RefFast) hard-codes 1.0 too. It is only used on nuScenes / AV2, where 1.0 is correct.

### Streaming AP, Argoverse-HD (sap_ahd_v6.csv; k28_ahd_sap.py, md5 c44b8ea7, run on VUW)

Setup:
- Simulation as in K7 (ahd_sap.py): single processor, newest frame first, newest finished output shown at every annotated frame.
- Detections: the phase-2 AP dump of v6-S (conf ≥ 0.1, the K7 definition for crop pipelines).
- Latency: the per-frame Jetson ms of this window (ahd_v6S_latency.csv).

| pipeline | sAP | sAP50 | sAP small | mean / p95 ms | frames processed | mean staleness ms |
|---|---|---|---|---|---|---|
| v6-S + refiner v3a (this window) | 0.064 | 0.173 | 0.024 | 74.9 / 169.6 | 56% | 128 |
| v6-S, timing without refiner | 0.066 | 0.176 | 0.025 | 71.8 / 164.9 | 57% | 123 |
| K7: cheap | 0.096 | 0.185 | 0.002 | 9.8 | 100% | 33 |
| K7: whole frame u960 | 0.221 | 0.413 | 0.065 | 21.4 | 100% | 33 |
| K7: whole frame u1280 | 0.247 | 0.465 | 0.106 | 29.4 | 100% | 34 |
| K7: ours (v4-lite + v2) | 0.105 | 0.238 | 0.045 | 31.5 | 88% | 60 |

- **v6-S is not a streaming detector.** At 30 fps it processes 56% of frames, and its sAP is below even the cheap draft.
- Its non-streaming AP on the same frames is 0.165 (§11). The K7 rows come from K7's own detections and timings.

## 17. Qualitative and failure panels (panels/; k28_render.py, md5 8b496bf3)

**Frames.** 30 frames from the capture run (k28_p3.py --part instr --capture; AV2 with the ego mask). Selection: results/k28/p3/capture_frames.csv.
- **K8 categories, 8 frames.** One per category for nuScenes and AV2: spurious brake removed, range corrected, missed object recovered by the field, failure.
- **K15 frames, 4.** A relevant object missed by the draft and by uncertainty crops and recovered by v6-S: 2 nuScenes, 2 AV2.
- **Failure categories of §13, 18 frames** (missed, under-reaction, unnecessary action, other):
  - nuScenes, 7 frames: 1 missed, 2 of each other category;
  - AV2, 7 frames: 1 under-reaction, 2 of each other category;
  - Argoverse-HD, 4 frames: 1 per category.

**Files.** Each frame has six panels, PDF (vector overlays) + PNG at native resolution, with no text in the image. K8 / K15 colours:
- `_a_draft`: draft detections (blue; dashed = replaced by a verified detection), and missed relevant objects (dashed red);
- `_b_bev`: bird's-eye view with the sensitive cells of the draft's field (orange), the corridor, the ego and the draft objects (blue);
- `_c_crops`: v6-S field crops (orange), support crops (blue) and guard-grown crops (grey);
- `_d_final`: the v6-S final scene, with detections verified on a crop (red), detections kept from the draft (blue) and the annotation (thin green; relevant objects dashed);
- `_e_native`: whole frame at native resolution, detections in red on an orange-tinted image;
- `_f_unc`: uncertainty crops unc2@S + C1 (orange) and their verified detections (red).

**Captions.** `examples.csv` gives, per frame:
- the actions: required, draft, v6-S without and with refiner, u960 + C1, native + C1, unc2@S + C1;
- crops, rounds, and tensor px of v6-S and unc2@S;
- the reason text, for captions;
- the file stem (`panel_stem`).

File stems shorten the unit name (e.g. `nusc_scene-10_0006` is scene-1081, frame 6), so use `examples.csv` to map them.

## Files

**Analysis CSVs:**
- ablation_v6.csv, ctable_{y11s,ft0,ft1,nomask}.csv;
- headline_nomask_v6.csv, class_matched_nomask_v6.csv;
- sota_intervals_v6.csv;
- agreement_v6.csv, attribution_v6.csv, attribution_detail_v6.csv;
- missed_natural_v6.csv, missed_controlled_v6.csv;
- regime_v6.csv, consumer2_v6.csv, roadmap_v6.csv, cost_stats_v6.csv;
- loss_terms_{nusc,av2,ahd}_v6.csv, sign_tests_v6.csv, collisions_v6.csv;
- failure_v6.csv, failure_frames_v6.csv;
- parity_v6.csv.

**Detection metrics:** ap_ahd_v6.csv, detm_nusc_v6.csv, parity_ap_v6.csv.

**Figures:** roadmap_v6.pdf / .png, detection_vs_gflops_v6.pdf / .png.

**Seeds (§1):** seeds_test_v6.csv, s6_test_v6.csv.

**Panels (§17; panels/):** 30 frames × 6 panels × PDF + PNG, and examples.csv.

**Jetson (§16; jetson/):** s3_front.csv, s3_dominance.csv, cost_model.csv, s3_front.pdf / .png, kitti_v6S_latency.csv, kitti_v6S_cells.csv, ahd_v6S_latency.csv (summary), runs.csv (every final-window run with its parity), sap_ahd_v6.csv. Raw per-frame pickles: results/raw/20261014_*_k28_v6_latency_*_final*.
