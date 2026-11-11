# K29 — fast C1 / v3a on the Jetson AGX Xavier, and re-timing of the latency table — FINAL REPORT

**State on 2026-10-18 07:30 +07. FINAL.**
- Step 1 (fast path) and the R2 acceptance test on the full dev splits: **DONE** (2026-10-15).
- Step 2 (re-timing on the test frame sets): **DONE** (2026-10-18 06:32). It ran at MAXN with a new 140 W USB-C PD supply, with 3 repetitions, both heuristics and the breakdown, and 0 resets (§7).
- **The final tables are §6 (latency_fast.csv / latency_ref.csv), §6a (fronts), §6b (tail) and §6c (breakdown).** The interim repetition-1 table is kept in maxn_partial/ for the record only.
- All test-frame G values in this report are **post-hoc test evaluations**: the timing runs compute them, and they are identical for the reference and the fast path.
- Running log and every decision: [STATUS.md](STATUS.md).

## 0. K6 closed loop with v6-S, all three runs (from K28 phase 3 §14a)

Bench2Drive, 220 routes per run (219 compared), K6's evaluator; v6-S without refiner, v_close 0, crops 0.6, C1.
Differences are v6-S − branch: paired-route bootstrap, 2,000, seed 0.

| branch | DS mean ± sd (3 runs) | SR mean ± sd | RC | v6-S − branch, DS | v6-S − branch, SR |
|---|---|---|---|---|---|
| **v6-S (ours_v6)** | **92.54 ± 0.82** | **0.839 ± 0.011** | 97.62 | — | — |
| privileged | 96.63 ± 0.17 | 0.915 ± 0.003 | 98.53 | −4.09 [−6.36, −2.02] | −0.076 [−0.117, −0.038] |
| cheap | 92.27 ± 0.92 | 0.834 ± 0.018 | 97.59 | +0.27 [−0.95, +1.48] | +0.005 [−0.020, +0.029] |
| u960 | 93.15 ± 0.74 | 0.857 ± 0.010 | 96.92 | −0.61 [−2.20, +1.03] | −0.018 [−0.050, +0.014] |
| u1600 | 91.95 ± 0.71 | 0.837 ± 0.013 | 96.94 | +0.59 [−1.37, +2.42] | +0.002 [−0.033, +0.033] |
| ours (v3, K6) | 92.80 ± 0.63 | 0.845 ± 0.012 | 97.65 | −0.26 [−1.23, +0.61] | −0.006 [−0.024, +0.011] |

- v6-S ties cheap, u960, u1600 and v3: every interval contains 0. It is below privileged.
- Crops per frame: v6-S 0.81, v3 1.15.
- Sources: handoff/K28/phase3/b2d_v6_summary.csv, b2d_v6_per_run.csv, b2d_v6_per_route.csv.

## 1. Summary

- **A fast path for C1 and the range refiner that changes no decision.**
  - Batched, bucketed TensorRT encoder engines; a per-crop feature and crop cache; exact corridor / 80 m pruning of C1; a bit-identical crop cutter.
  - Selected variant **trt32**: C1 encoder in TensorRT FP32, with pruning.
- **R2 met on the full dev splits, with no differing frame** (§4).
  - Splits: nuScenes new val, 152 scenes, 6,117 frames; AV2 CFG-DEV, 128 logs, 10,123 frames, ego mask.
  - **trt32 gives 100% identical actions in all 32 C1 / refiner pipelines on both splits.** Dev G of the reference and of trt32 are equal in every row.
  - Every other candidate also passes R2: worst row 99.98%, at most 7 differing frames per split in total.
- **Same fast path for every C1 / v3a pipeline (R3).** Whole frame + C1 + refiner, both heuristics + C1, v5.1-S's refiner, v6-S / M / L.
- **Speed-up, test frames, final (3 repetitions at MAXN; reference → trt32, ms):**

  | pipeline | nuScenes test 20 | nuScenes old 20 (K28 set) | AV2 8 logs |
  |---|---|---|---|
  | v6-S, mean | 28.8 → 21.9 (1.31×) | 28.6 → 23.0 (1.25×) | 72.3 → 41.7 (1.73×) |
  | v6-S, p95 | 83.5 → 59.2 (1.41×) | 76.8 → 54.4 (1.41×) | 169.7 → 93.8 (1.81×) |
  | v6-S ≤ 2 rounds, mean / p95 | 29.3 / 90.1 → 19.8 / 46.2 | 25.9 / 63.1 → 20.6 / 40.6 | 80.3 / 198.9 → 37.1 / 73.5 |
  | whole frame u960 + C1 + ref, mean | 50.8 → 22.7 (2.24×) | 34.4 → 20.5 (1.68×) | 98.2 → 34.3 (2.86×) |
  | heuristic tsp2@S + C1 + ref, mean | 28.2 → 22.6 | 26.2 → 21.2 | 65.1 → 41.7 |

  - **With the fast path, C1 costs about 2 ms over T4 in whole-frame pipelines.** Example, u960 on nuScenes test: + T4 + ref 20.5 ms, + C1 + ref 22.7 ms.
  - **Decision parity on every timed frame: 100%.** 0 differing frames in 378 pipeline × repetition rows (275,688 timed frames). The G of every row is identical across the 3 repetitions (spread 0).
- **Where v6-S lands with the fast path (§6a; test frames, post-hoc):**
  - **v6-S itself is on no latency-G front, in mean or p95, on any of the three frame sets.** Its own ≤ 2-round bound has the same G within 0.004 (n.s. or −0.000 upper bound) and is faster.
  - **v6-S ≤ 2 rounds is on the mean front on nuScenes test 20 and AV2, and also on the p95 front on AV2.** It is on no front on the old nuScenes 20.
  - **Against whole frame + C1 + refiner, v6-S ties on nuScenes but is faster or above on AV2.**
    - nuScenes test 20: u960 + C1 + ref has G 0.386 vs v6-S 0.380 (−0.007 [−0.049, +0.044]) at 22.7 vs 21.9 ms mean and 30.9 vs 59.2 ms p95. It is on both fronts.
    - AV2: v6-S is above u960 + C1 + ref by +0.097 [+0.001, +0.216] (34.3 vs 41.7 ms) and above u1280 + C1 + ref by +0.108 [+0.002, +0.252] at a similar mean (43.6 ms). It ties native + C1 + ref (+0.010 [−0.074, +0.093]) at 55% of its mean latency.
  - **The heuristics + C1 are not beaten.**
    - nuScenes test 20: tsp2 v6-S − heuristic is +0.033 [−0.009, +0.093]; tsp8 (YOLO11m) is −0.012 [−0.070, +0.045].
    - AV2: tsp2 is above v6-S by 0.012 ([−0.029, −0.000]) at the same mean (41.7 ms) but a higher p95 (105.5 vs 93.8 ms). That puts tsp2 on the AV2 mean front instead of v6-S.
  - **The p95 tail is v6-S's weakness.** v6-S's p95 is 2.2–2.7× its mean; whole frame + C1 + ref is 1.2–1.4×. The breakdown (§6c) puts the tail in the guard's detector calls, not in C1.

## 2. Fast path (step 1)

| item | implementation | exact? |
|---|---|---|
| a. batching | one encoder call per lift for all crops not yet cached, padded to buckets 1 / 2 / 4 / 8 / 16 / 32 (chunks of 32 above); one MLP call per lift; v3a one call per frame, buckets 1 / 2 / 4 / 8 / 16 | numerics only |
| b. engines | C1 image encoder (flip, bilinear 64 → 128, normalisation, ResNet-18) → ONNX (c1enc_dyn.onnx) → one static TensorRT engine per bucket, FP16 and FP32. MLP head: torch FP32. Second option: torch FP16 + one CUDA graph per bucket (t16g). | numerics only |
| c. cache | per pipeline, encoder feature and 64×64 crop of (frame, exact box bytes), reused at f + 1 / f + 2 and in later rounds; the crop is shared by C1 and v3a | exact up to batch numerics |
| crop | `crop64_fast` pads only the patch, not the whole image | bit-identical: 0 mismatches in 2,400 random boxes and in 11,850 in-run checks of the dev parity runs |
| d. pruning | C1 skipped for chained objects outside the corridor or beyond 80 m (the lift's own z / lat); they get ttc = +∞ | exact: such an object has braking demand 0 for every ttc, is never support and so never refined, and has TTC-warning demand 0 (argument in STATUS) |
| e. overlap | the only overlap the data dependency allows: encode a round's non-truncated detections on a second stream while the guard's calls run (speculative) | exact up to numerics |

- **The loop keeps one C1 call per round.** Round k + 1's crops depend on round k's C1 ttc, so one call per frame is impossible.
- **v3a was already a TensorRT FP16 engine in the reference.** Its fast path only re-buckets the batch and takes the crops from the cache.
- **crop64's cost is cv2.resize INTER_AREA, not the border copy.** The cache cuts each crop once. A 6-thread pool cuts ≥ 8 missing crops at once, with identical output. GPU resizing was rejected: it is not bit-identical.

## 3. Every variant tried (R4)

Dev selection timing: v6-S mean (p95), ms, on nuScenes new val (first 20 scenes, 595 timed frames) and AV2 CFG-DEV (first 10 logs, 584 frames). MAXN, jetson_clocks.
Parity: the worst pipeline and the total number of differing frames over the 32 pipeline rows of each full dev split.

| path | encoder / option | nuScenes v6-S | AV2 v6-S | avg of means | parity nuScenes (worst / frames) | parity AV2 (worst / frames) | note |
|---|---|---|---|---|---|---|---|
| reference (K28) | torch FP32 TempHead, no cache, no pruning | 37.36 (96.7) | 91.85 (189.4) | 64.61 | — | — | re-timed in select2: 37.35 / 93.77 |
| trt16 | TensorRT FP16, pruning | 26.62 (62.9) | 46.89 (93.0) | **36.76** | 99.98% / 1 | 99.99% / 2 | fastest |
| **trt32 (selected)** | TensorRT FP32, pruning | 26.53 (63.2) | 47.16 (93.4) | 36.84 | **100% / 0** | **100% / 0** | within 0.5 ms of trt16 → tie rule takes FP32 |
| t16g (fixed) | torch FP16 + CUDA graphs, pruning | 27.04 (64.7) | 48.98 (98.0) | 38.01 | 99.98% / 2 | 99.98% / 7 | re-timed in select2 after the fix |
| trt16ov | trt16 + speculative overlap | 27.22 (64.7) | 48.69 (96.4) | 37.95 | 99.98% / 1 | 99.99% / 4 | overlap is slower than trt16 |
| t32 | torch FP32 eager encoder, pruning | 27.69 (66.8) | 53.79 (104.4) | 40.74 | 100% / 0 | 100% / 0 | |
| trt16np | trt16 without pruning | 28.11 (66.3) | 48.29 (93.3) | — | 99.98% / 2 | 99.99% / 5 | exactness check of the pruning, not a candidate |
| t16g (bug) | CUDA graphs reading a freed FP16 module | 26.69 (63.2) | 47.53 (92.5) | — | v6-S 98.8%, u960 + C1 + ref 96.3% (selection frames) | 99.1% / 97.6% | **bug, not an FP16 result**; fixed in c1fast b03f431a |

- **Selection rule.** It was fixed before any timing; the tie rule was added after the nuScenes selection numbers were seen (disclosed in STATUS).
  - Take the lowest average of the two v6-S dev means among the candidates that pass R2 on the full dev splits.
  - Tie rule: within 0.5 ms, take the higher precision. trt16 is lowest, but trt32 is within 0.08 ms, so **trt32**.
- **The pruning changes no decision.** trt16np (no pruning) and trt16 differ only by single frames, both ways. These are FP16 batch-composition numerics; the FP32 paths with pruning have 0 differing frames.
- **Encoder features vs torch FP32**, 160 real AV2 crops, relative error:

  | variant | relative error |
  |---|---|
  | trt32 | 6.8e-7 |
  | trt16 | 3.0e-3 |
  | t16 eager | 3.7e-3 |
  | t16g fixed | 3.7e-3 |
  | t16g bug | 68 |

- **u960 + C1 + ref, dev mean (ms), nuScenes / AV2:**

  | path | nuScenes | AV2 |
  |---|---|---|
  | reference | 49.9 | 102.0 |
  | trt16 | 21.5 | 32.7 |
  | trt32 | 21.9 | 34.0 |
  | t16g fixed | 22.0 | 34.9 |
  | trt16ov | 21.3 | 32.2 |
  | t32 | 24.2 | 40.9 |
  | trt16np | 25.9 | 39.4 |

## 4. Decision parity on the full dev splits (R2), selected path trt32

Identical actions per frame, against the reference, with detections shared between the paths, so only C1 / v3a differ. Dev G = (J_cheap − J) / J_cheap over the split.
**trt32 differs from the reference on 0 frames in every row**, so dev G is the same for both paths. Every candidate is in parity_dev.csv.

| pipeline | nuScenes val: differing frames / G (ref = trt32) | AV2 dev: differing frames / G (ref = trt32) |
|---|---|---|
| whole frame u448 + T4 + ref | 0 / 0.143 | 0 / 0.231 |
| whole frame u448 + C1 + ref | 0 / 0.173 | 0 / 0.375 |
| whole frame u512 + T4 + ref | 0 / 0.186 | 0 / 0.257 |
| whole frame u512 + C1 + ref | 0 / 0.212 | 0 / 0.388 |
| whole frame u640 + T4 + ref | 0 / 0.023 | 0 / 0.269 |
| whole frame u640 + C1 + ref | 0 / 0.062 | 0 / 0.393 |
| whole frame u960 + T4 + ref | 0 / 0.211 | 0 / 0.284 |
| whole frame u960 + C1 + ref | 0 / 0.253 | 0 / 0.402 |
| whole frame u1280 + T4 + ref | 0 / 0.290 | 0 / 0.293 |
| whole frame u1280 + C1 + ref | 0 / 0.320 | 0 / 0.394 |
| whole frame native + T4 + ref | 0 / 0.272 | 0 / 0.250 |
| whole frame native + C1 + ref | 0 / 0.303 | 0 / 0.348 |
| v6-S | 0 / 0.317 | 0 / 0.368 |
| v6-S, no refiner | 0 / 0.310 | 0 / 0.292 |
| v6-S, rounds ≤ 2 | 0 / 0.243 | 0 / 0.368 |
| v6-S, rounds ≤ 2, no refiner | 0 / 0.235 | 0 / 0.297 |
| v5.1-S | 0 / 0.297 | 0 / 0.309 |
| v5.1-S, no refiner | 0 / 0.293 | 0 / 0.172 |
| v5.1-S, rounds ≤ 2 | 0 / 0.223 | 0 / 0.312 |
| v5.1-S, rounds ≤ 2, no refiner | 0 / 0.218 | 0 / 0.180 |
| heuristic tsp2@S + C1 + ref (YOLOv8s) | 0 / 0.270 | 0 / 0.354 |
| same, no refiner | 0 / 0.261 | 0 / 0.279 |
| heuristic tsp8@S + C1 + own ref (YOLO11m) | 0 / 0.392 | 0 / 0.363 |
| same, no refiner | 0 / 0.389 | 0 / 0.279 |
| v6-M | 0 / 0.374 | 0 / 0.361 |
| v6-M, no refiner | 0 / 0.370 | 0 / 0.286 |
| v6-M, rounds ≤ 2 | 0 / 0.342 | 0 / 0.361 |
| v6-M, rounds ≤ 2, no refiner | 0 / 0.337 | 0 / 0.292 |
| v6-L | 0 / 0.391 | 0 / 0.366 |
| v6-L, no refiner | 0 / 0.387 | 0 / 0.284 |
| v6-L, rounds ≤ 2 | 0 / 0.374 | 0 / 0.372 |
| v6-L, rounds ≤ 2, no refiner | 0 / 0.369 | 0 / 0.292 |

- Frames: 6,117 (nuScenes) and 10,123 (AV2) per row.
- Missing images: 0.
- crop64_fast checks in these runs: 1,083 (nuScenes) and 10,767 (AV2), 0 mismatches.

## 5. C1 work on the dev splits (c1stats_dev.csv)

| split | chained objects | pruned (exact) | crops encoded, trt32 | crops encoded without pruning (trt16np) | crops encoded by the reference (3 per chain) |
|---|---|---|---|---|---|
| nuScenes new val | 137,264 | 109,619 (80%) | 34,690 | 186,843 | 411,792 |
| AV2 CFG-DEV | 668,152 | 491,528 (74%) | 199,353 | 764,123 | 2,004,456 |

- The cache alone cuts encoder work 2.2× (nuScenes) and 2.6× (AV2) against the reference: crops are re-used across frames and rounds.
- Pruning cuts it a further 5.4× and 3.8×.
- Summed over every C1 pipeline of the dev parity runs.

## 6. Re-timing on the test frame sets — FINAL (3 repetitions, MAXN)

**Setup:**
- MAXN with jetson_clocks: GPU 1,377 MHz, EMC 2,133 MHz, CPU 8 × 2,265.6 MHz; fan 100%. 140 W USB-C PD supply from 2026-10-18 (§7); repetition 1 and part of repetition 2 ran on 2026-10-15 with the old supply, same clocks.
- The reference and trt32 run in the same process on the same frames. Every row has 3 repetitions; the latency is pooled over them.
- Frame sets:
  - nuScenes new test, first 20 scene tokens sorted (805 frames, after a discarded warm-up pass of 200);
  - K28's 20 old DEEP scenes (797 timed frames);
  - AV2 val, K28's 8 timed logs with the ego mask (586 timed frames).
- Heuristics + C1: one heuristic per process, 3 repetitions each (h1 = tsp2@S YOLOv8s + C1 + v3a; h2 = tsp8@S YOLO11m + C1 + own refiner).
- **G is a post-hoc test evaluation**; it is identical for both paths and across repetitions.
- Cheap and plain whole frame have no C1 / refiner, so they have one path.

### 6a. Latency table and fronts (latency_fast.csv, latency_ref.csv, front.csv, front_intervals.csv, front_fast / front_ref .pdf / .png)

Selected rows below; every one of the 39 pipelines per frame set (with median and max) is in the CSVs.

**nuScenes new test 20** (805 timed frames × 3 repetitions). Fronts: M = on the mean front, P = on the p95 front (fast path). v6-S − row: paired unit bootstrap.

| pipeline | G | mean / p95 / p99, reference | mean / p95 / p99, trt32 | front | v6-S − row |
|---|---|---|---|---|---|
| cheap (u320) | 0.000 | — | 7.5 / 8.2 / 8.6 | MP | +0.380 [+0.196, +0.511] |
| whole frame u448 | 0.040 | — | 8.9 / 9.6 / 9.9 | MP | +0.340 [+0.142, +0.686] |
| whole frame u960 | 0.066 | — | 18.2 / 19.0 / 19.3 | – | +0.314 [+0.182, +0.519] |
| whole frame native | 0.098 | — | 33.4 / 34.4 / 35.0 | – | +0.282 [+0.015, +0.569] |
| whole frame u448 + T4 + ref | 0.277 | 11.1 / 15.6 / 16.5 | 11.2 / 15.0 / 15.7 | MP | +0.102 [-0.040, +0.264] |
| whole frame u960 + T4 + ref | 0.369 | 20.4 / 24.7 / 25.9 | 20.5 / 24.2 / 25.3 | P | +0.010 [-0.025, +0.065] |
| whole frame native + T4 + ref | 0.302 | 35.8 / 40.0 / 41.3 | 35.9 / 39.6 / 40.6 | – | +0.078 [-0.114, +0.247] |
| whole frame u448 + C1 + ref | 0.331 | 36.1 / 71.4 / 94.8 | 13.0 / 21.4 / 29.7 | MP | +0.049 [-0.046, +0.130] |
| whole frame u512 + C1 + ref | 0.336 | 37.8 / 78.1 / 95.1 | 13.9 / 22.1 / 31.7 | MP | +0.044 [-0.047, +0.119] |
| whole frame u640 + C1 + ref | 0.330 | 42.8 / 85.1 / 110.8 | 16.7 / 25.1 / 33.5 | – | +0.050 [+0.001, +0.101] |
| whole frame u960 + C1 + ref | 0.386 | 50.8 / 91.5 / 112.5 | 22.7 / 30.9 / 39.5 | MP | -0.007 [-0.049, +0.044] |
| whole frame u1280 + C1 + ref | 0.395 | 57.1 / 100.5 / 120.9 | 28.6 / 36.5 / 41.1 | P | -0.015 [-0.119, +0.151] |
| whole frame native + C1 + ref | 0.364 | 66.0 / 107.5 / 124.6 | 37.9 / 45.5 / 50.2 | – | +0.016 [-0.200, +0.207] |
| heuristic tsp2@S + C1 + ref (YOLOv8s) | 0.347 | 28.2 / 91.1 / 136.9 | 22.6 / 72.1 / 97.5 | – | +0.033 [-0.009, +0.093] |
| heuristic tsp8@S + C1 + own ref (YOLO11m) | 0.392 | 35.7 / 119.0 / 171.8 | 28.6 / 91.2 / 120.1 | – | -0.012 [-0.070, +0.045] |
| v5.1-S | 0.339 | 21.4 / 57.1 / 80.7 | 21.0 / 57.2 / 80.5 | – | +0.041 [+0.006, +0.116] |
| v5.1-S, rounds ≤ 2 | 0.345 | 19.2 / 42.0 / 52.1 | 19.0 / 41.4 / 50.7 | M | +0.035 [+0.000, +0.107] |
| **v6-S** | 0.380 | 28.8 / 83.5 / 115.0 | 21.9 / 59.2 / 84.6 | – | — |
| v6-S, rounds ≤ 2 | 0.382 | 29.3 / 90.1 / 110.2 | 19.8 / 46.2 / 55.1 | M | -0.003 [-0.011, +0.000] |
| v6-S, rounds ≤ 3 | 0.380 | 32.4 / 108.0 / 148.2 | 21.1 / 52.7 / 70.0 | – | -0.000 [-0.001, +0.000] |
| v6-S, no refiner | 0.289 | 27.6 / 80.5 / 114.7 | 20.9 / 57.0 / 80.6 | – | +0.090 [-0.043, +0.252] |
| v6-M | 0.394 | 31.9 / 97.9 / 128.4 | 24.7 / 68.6 / 96.9 | – | -0.014 [-0.075, +0.037] |
| v6-M, rounds ≤ 2 | 0.407 | 32.4 / 100.7 / 129.5 | 22.7 / 55.3 / 71.5 | MP | -0.027 [-0.080, +0.000] |
| v6-L | 0.405 | 32.2 / 94.1 / 118.7 | 25.3 / 69.2 / 91.6 | – | -0.026 [-0.087, +0.004] |
| v6-L, rounds ≤ 2 | 0.400 | 32.9 / 100.0 / 114.3 | 23.7 / 59.2 / 74.4 | – | -0.021 [-0.081, +0.010] |

**nuScenes old 20 (K28 set)** (797 timed frames × 3 repetitions). Fronts: M = on the mean front, P = on the p95 front (fast path). v6-S − row: paired unit bootstrap.

| pipeline | G | mean / p95 / p99, reference | mean / p95 / p99, trt32 | front | v6-S − row |
|---|---|---|---|---|---|
| cheap (u320) | 0.000 | — | 7.2 / 8.1 / 8.4 | MP | +0.096 [+0.014, +0.328] |
| whole frame u448 | 0.031 | — | 8.6 / 9.5 / 9.8 | MP | +0.065 [-0.009, +0.299] |
| whole frame u960 | 0.060 | — | 18.0 / 18.9 / 19.5 | – | +0.035 [-0.011, +0.173] |
| whole frame native | 0.042 | — | 33.3 / 34.2 / 35.1 | – | +0.054 [-0.033, +0.293] |
| whole frame u448 + T4 + ref | 0.084 | 10.2 / 14.4 / 20.7 | 10.3 / 14.1 / 20.4 | MP | +0.012 [-0.039, +0.092] |
| whole frame u960 + T4 + ref | 0.087 | 19.5 / 21.5 / 29.0 | 19.6 / 22.0 / 28.8 | – | +0.009 [-0.053, +0.102] |
| whole frame native + T4 + ref | 0.092 | 34.9 / 39.2 / 44.0 | 35.0 / 38.9 / 43.8 | – | +0.004 [-0.081, +0.100] |
| whole frame u448 + C1 + ref | 0.092 | 22.4 / 50.8 / 65.9 | 11.1 / 19.6 / 29.4 | MP | +0.004 [-0.041, +0.069] |
| whole frame u512 + C1 + ref | 0.095 | 23.7 / 51.7 / 73.7 | 11.9 / 20.0 / 30.1 | MP | +0.001 [-0.054, +0.067] |
| whole frame u640 + C1 + ref | 0.099 | 27.0 / 55.9 / 81.1 | 14.5 / 22.6 / 30.1 | MP | -0.003 [-0.051, +0.044] |
| whole frame u960 + C1 + ref | 0.098 | 34.4 / 64.9 / 87.2 | 20.5 / 28.5 / 35.8 | – | -0.002 [-0.049, +0.049] |
| whole frame u1280 + C1 + ref | 0.123 | 41.4 / 68.4 / 96.5 | 26.7 / 35.6 / 42.5 | P | -0.027 [-0.133, +0.022] |
| whole frame native + C1 + ref | 0.130 | 50.3 / 76.6 / 106.8 | 35.9 / 44.4 / 50.3 | P | -0.034 [-0.155, +0.021] |
| heuristic tsp2@S + C1 + ref (YOLOv8s) | 0.085 | 26.2 / 71.4 / 121.9 | 21.2 / 54.8 / 86.6 | – | +0.011 [+0.000, +0.063] |
| heuristic tsp8@S + C1 + own ref (YOLO11m) | 0.099 | 35.0 / 103.4 / 165.8 | 29.4 / 83.0 / 128.1 | – | -0.003 [-0.064, +0.059] |
| v5.1-S | 0.080 | 22.4 / 51.9 / 68.4 | 22.0 / 51.5 / 67.7 | – | +0.016 [-0.011, +0.093] |
| v5.1-S, rounds ≤ 2 | 0.082 | 20.1 / 37.3 / 43.0 | 20.1 / 37.2 / 42.6 | – | +0.014 [-0.021, +0.093] |
| **v6-S** | 0.096 | 28.6 / 76.8 / 103.9 | 23.0 / 54.4 / 73.8 | – | — |
| v6-S, rounds ≤ 2 | 0.098 | 25.9 / 63.1 / 84.8 | 20.6 / 40.6 / 48.4 | – | -0.002 [-0.012, +0.000] |
| v6-S, rounds ≤ 3 | 0.098 | 29.3 / 83.2 / 132.2 | 22.1 / 49.1 / 64.8 | – | -0.002 [-0.012, +0.000] |
| v6-S, no refiner | 0.116 | 28.0 / 75.2 / 101.8 | 22.5 / 53.7 / 73.0 | – | -0.020 [-0.078, +0.000] |
| v6-M | 0.098 | 31.2 / 83.8 / 119.1 | 25.7 / 61.7 / 86.1 | – | -0.002 [-0.034, +0.029] |
| v6-M, rounds ≤ 2 | 0.098 | 28.8 / 71.0 / 99.5 | 23.8 / 51.2 / 66.4 | – | -0.002 [-0.034, +0.029] |
| v6-L | 0.115 | 32.5 / 85.1 / 118.1 | 27.1 / 66.3 / 86.4 | – | -0.019 [-0.101, +0.018] |
| v6-L, rounds ≤ 2 | 0.115 | 30.3 / 75.1 / 104.9 | 25.1 / 54.1 / 67.8 | – | -0.019 [-0.101, +0.018] |

**AV2 8 logs, ego mask** (586 timed frames × 3 repetitions). Fronts: M = on the mean front, P = on the p95 front (fast path). v6-S − row: paired unit bootstrap.

| pipeline | G | mean / p95 / p99, reference | mean / p95 / p99, trt32 | front | v6-S − row |
|---|---|---|---|---|---|
| cheap (u320) | 0.000 | — | 10.4 / 10.9 / 11.3 | MP | +0.419 [+0.251, +0.658] |
| whole frame u448 | 0.056 | — | 12.5 / 13.1 / 13.4 | MP | +0.364 [+0.186, +0.632] |
| whole frame u960 | 0.083 | — | 23.3 / 23.9 / 24.4 | P | +0.337 [+0.175, +0.614] |
| whole frame native | 0.125 | — | 64.9 / 65.9 / 66.3 | – | +0.294 [+0.150, +0.466] |
| whole frame u448 + T4 + ref | 0.284 | 17.1 / 25.1 / 29.6 | 17.0 / 24.6 / 28.6 | MP | +0.136 [+0.062, +0.231] |
| whole frame u960 + T4 + ref | 0.276 | 27.9 / 35.6 / 40.4 | 27.7 / 35.1 / 39.2 | – | +0.143 [+0.021, +0.295] |
| whole frame native + T4 + ref | 0.388 | 69.4 / 76.3 / 82.2 | 69.2 / 75.5 / 80.0 | – | +0.032 [-0.052, +0.123] |
| whole frame u448 + C1 + ref | 0.366 | 130.9 / 185.1 / 196.1 | 23.2 / 38.1 / 44.1 | MP | +0.054 [+0.007, +0.110] |
| whole frame u512 + C1 + ref | 0.363 | 133.9 / 187.0 / 203.7 | 24.3 / 39.7 / 44.8 | – | +0.056 [+0.014, +0.110] |
| whole frame u640 + C1 + ref | 0.351 | 130.5 / 207.2 / 232.0 | 27.9 / 43.5 / 49.1 | – | +0.069 [+0.006, +0.148] |
| whole frame u960 + C1 + ref | 0.323 | 98.2 / 185.2 / 210.2 | 34.3 / 49.2 / 56.3 | – | +0.097 [+0.001, +0.216] |
| whole frame u1280 + C1 + ref | 0.311 | 109.5 / 179.8 / 207.1 | 43.6 / 59.3 / 66.9 | – | +0.108 [+0.002, +0.252] |
| whole frame native + C1 + ref | 0.410 | 143.8 / 210.7 / 244.3 | 75.7 / 88.7 / 94.1 | – | +0.010 [-0.074, +0.093] |
| heuristic tsp2@S + C1 + ref (YOLOv8s) | 0.431 | 65.1 / 169.5 / 226.5 | 41.7 / 105.5 / 142.2 | M | -0.012 [-0.029, -0.000] |
| heuristic tsp8@S + C1 + own ref (YOLO11m) | 0.441 | 87.7 / 216.4 / 317.7 | 57.6 / 149.9 / 217.7 | – | -0.021 [-0.035, +0.000] |
| v5.1-S | 0.397 | 37.2 / 82.0 / 110.1 | 36.3 / 81.0 / 108.5 | – | +0.022 [-0.004, +0.045] |
| v5.1-S, rounds ≤ 2 | 0.401 | 32.6 / 64.0 / 73.6 | 32.0 / 60.9 / 73.3 | MP | +0.018 [-0.007, +0.040] |
| **v6-S** | 0.419 | 72.3 / 169.7 / 208.8 | 41.7 / 93.8 / 122.0 | – | — |
| v6-S, rounds ≤ 2 | 0.423 | 80.3 / 198.9 / 239.0 | 37.1 / 73.5 / 92.3 | MP | -0.004 [-0.007, +0.000] |
| v6-S, rounds ≤ 3 | 0.423 | 100.9 / 262.1 / 337.5 | 42.1 / 88.6 / 113.5 | – | -0.004 [-0.007, +0.000] |
| v6-S, no refiner | 0.374 | 68.8 / 163.0 / 197.7 | 39.5 / 89.9 / 116.0 | – | +0.046 [-0.072, +0.211] |
| v6-M | 0.424 | 76.8 / 181.4 / 231.0 | 47.2 / 110.7 / 146.5 | – | -0.004 [-0.013, +0.004] |
| v6-M, rounds ≤ 2 | 0.424 | 84.3 / 204.2 / 261.7 | 42.3 / 89.6 / 122.3 | P | -0.004 [-0.013, +0.004] |
| v6-L | 0.443 | 77.1 / 183.8 / 223.9 | 48.7 / 111.7 / 148.9 | MP | -0.023 [-0.047, -0.002] |
| v6-L, rounds ≤ 2 | 0.440 | 83.8 / 194.7 / 275.0 | 44.5 / 99.5 / 139.8 | MP | -0.021 [-0.046, +0.002] |

**Reading:**
- **v6-S is on no front.** Every set has a faster row with at least the same G point estimate: its own ≤ 2-round bound everywhere; on the old nuScenes 20 also v5.1-S and v6-S without refiner; on AV2 also tsp2 (same mean, G +0.012).
- **v6-S ≤ 2 rounds is the useful operating point of our method.** It is on the mean front on nuScenes test 20 (19.8 ms, G 0.382) and on both fronts on AV2 (37.1 / 73.5 ms, G 0.423).
- **Among our loops, v6-M ≤ 2 rounds is on both fronts of nuScenes test 20** (G 0.407 at 22.7 / 55.3 ms; v6-S − it −0.027 [−0.080, +0.000]).
- **The fast path makes whole frame + C1 + refiner strong on nuScenes.** u960 + C1 + ref is on both fronts of nuScenes test 20 (G 0.386 at 22.7 / 30.9 ms). With the reference C1 it was at 50.8 ms.
- **On AV2 v6-S keeps a G margin over the whole-frame + C1 rows at u448–u1280** (+0.054 to +0.108, all intervals above 0), but these rows are faster except u1280.
- On the old 20 nuScenes scenes (K28's set, G ≈ 0.1 for every pipeline), almost every v6-S − row interval contains 0. Exceptions: v6-S is above cheap and whole frame u512; it is below the no-refiner bounded loops (v6-S ≤ 2 rounds without refiner −0.030 [−0.082, −0.011], v6-S ≤ 3 without refiner, v6-L ≤ 2 without refiner). On this set the refiner lowers G, as in K28.

### 6b. Tail (tail.csv)

| pipeline | nuScenes test 20, p95 / p99 / max (trt32) | nuScenes old 20 | AV2 |
|---|---|---|---|
| cheap (u320) | 8.2 / 8.6 / 9.4 | 8.1 / 8.4 / 9.2 | 10.9 / 11.3 / 12.5 |
| whole frame u960 + C1 + ref | 30.9 / 39.5 / 57.3 | 28.5 / 35.8 / 51.7 | 49.2 / 56.3 / 65.6 |
| whole frame u1280 + C1 + ref | 36.5 / 41.1 / 73.7 | 35.6 / 42.5 / 53.3 | 59.3 / 66.9 / 71.8 |
| whole frame native + C1 + ref | 45.5 / 50.2 / 60.1 | 44.4 / 50.3 / 58.6 | 88.7 / 94.1 / 102.7 |
| v6-S | 59.2 / 84.6 / 102.1 | 54.4 / 73.8 / 82.5 | 93.8 / 122.0 / 143.6 |
| v6-S, rounds ≤ 2 | 46.2 / 55.1 / 72.7 | 40.6 / 48.4 / 60.9 | 73.5 / 92.3 / 109.6 |

- v6-S's p95 / mean ratio is 2.2–2.7; the whole-frame + C1 + refiner rows have 1.2–1.4.
- The ≤ 2-round bound cuts v6-S's p95 by 22–25% (59.2 → 46.2, 54.4 → 40.6, 93.8 → 73.5 ms) at a G change of +0.002 to +0.004.

### 6c. Per-component breakdown of v6-S (breakdown.csv; instrumented pass, one repetition, ms)

| component | nuScenes test 20, ref → trt32 (mean / p95) | nuScenes old 20 | AV2 |
|---|---|---|---|
| draft | 7.3 / 7.9 → 7.3 / 7.9 | 7.0 / 7.8 → 7.0 / 7.8 | 9.9 / 10.3 → 10.0 / 10.4 |
| field | 1.0 / 1.9 → 0.9 / 1.8 | 1.1 / 1.9 → 1.1 / 1.8 | 1.2 / 2.4 → 1.1 / 2.2 |
| crops_r1 | 4.0 / 8.6 → 3.9 / 8.3 | 4.6 / 7.7 → 4.6 / 7.7 | 6.1 / 16.2 → 6.2 / 16.4 |
| crops_r2 | 0.3 / 0.0 → 0.3 / 0.0 | 0.2 / 0.0 → 0.2 / 0.0 | 1.4 / 7.9 → 1.3 / 7.4 |
| crops_r3 | 0.0 / 0.0 → 0.0 / 0.0 | 0.0 / 0.0 → 0.0 / 0.0 | 0.1 / 0.0 → 0.0 / 0.0 |
| guard | 5.2 / 30.2 → 5.2 / 29.2 | 6.4 / 26.5 → 6.4 / 26.5 | 10.3 / 33.9 → 10.2 / 33.7 |
| c1 | 8.0 / 34.5 → 1.4 / 8.2 | 6.6 / 26.9 → 1.2 / 6.5 | 36.0 / 101.5 → 5.9 / 16.8 |
| refiner | 1.0 / 5.5 → 0.8 / 3.6 | 0.4 / 0.4 → 0.3 / 0.4 | 3.2 / 11.5 → 1.7 / 4.7 |
| rest (support, merge, lift) | 2.9 / 7.8 → 2.8 / 7.5 | 2.8 / 6.6 → 2.7 / 6.5 | 5.4 / 12.3 → 5.1 / 11.7 |
| total | 29.7 / 87.0 → 22.5 / 61.2 | 29.1 / 77.6 → 23.4 / 54.2 | 73.4 / 172.2 → 41.6 / 93.7 |

- **C1 was the largest component of the reference on nuScenes test 20 and AV2** (27% and 49% of the mean; 23% on the old nuScenes 20, just below the draft's 24%). trt32 cuts it 5.9× / 5.6× / 6.1× in mean and 4.2× / 4.2× / 6.0× in p95.
- **With trt32, the guard's verifier calls set the tail.** Guard p95 is 29.2 / 26.5 / 33.7 ms, unchanged by the fast path (the guard is the YOLOv8s crop verifier, already TensorRT in the reference).
- The draft (cheap detector, 7–10 ms) is now 24–32% of v6-S's mean.
- The instrumented pass adds synchronisation, so its totals (22.5 / 23.4 / 41.6 ms) differ slightly from the timed runs.

## 7. Jetson hard resets from input-voltage droop (window paused 2026-10-15; resolved 2026-10-18)

**Resolution (2026-10-18).** The 140 W USB-C PD supply removed the resets.
- Idle VDD_IN 19.82 V (old supply 18.72 V).
- 45 s CPU + GPU stress at 55.8 W rail power: min VDD_IN 19.35 V.
- Window C, 03:12–06:32 (11,822 s of 1 s means, including the run that had reset the board twice): min VDD_IN 19.57 V at 30 W, max 31 W. **0 resets**; the board has been up since 2026-10-17 18:44.
- The reset diagnosis below is kept as the record of why repetition 1 and part of repetition 2 ran on 2026-10-15.


**Resets in window C (2026-10-15):**

| time | in which run | power just before |
|---|---|---|
| ~14:58 and ~15:06 | v5.1-S + heuristics | ~20 W |
| 16:53 | whole frame u960 / u1280 / native | 30 W |
| 17:16 | the same run, 4 s after its GPU load started | 26 W of rails |

None of these is a peak load, and every time the fan was at 100% and the temperature ≤ 64 °C.

**Reset reason: `SYS_RESET_N`** (PMC reset source, /sys/devices/platform/c360000.pmc/reset_reason).
- That is the external hardware reset line. A software reboot reads MAINSWRST; a watchdog reads BCCPLEXWDT.
- kern.log has **22 SYS_RESET_N boots since 2026-09-06**, including every reset of K28's builds and windows.
- There is no OOM, panic or thermal message before any of them.
- The 17:16 reset left the board off until 18:02.

**Input voltage before the 17:16 reset** (scripts/k29_vin_logger.py, INA3221 bus voltage, 1 s means; figure vin_reset.pdf / .png):

| state | VDD_IN | rail power |
|---|---|---|
| idle | 18.72 V | 2 W |
| light load | 18.62 V | 8 W |
| first 3 s of GPU load | 18.32–18.36 V | 24–26 W |

- **The averaged droop is already 0.4 V at about 26 W.** The INA3221 averages over many samples, so short dips are deeper than this.
- The Xavier design guide (dV/dt chapter, quoted on the NVIDIA forums) says VDDIN_PWR_BAD_N starts the shutdown at a drop of about 0.5 V.
- A supply rated 65 W should not droop like this at 26 W. That points at the adapter, cable, barrel jack or outlet, not at the code: a user process cannot assert SYS_RESET_N.

**Code review of the process that reset twice (STATUS 15:20).**
- No bug in it can explain a reset: shapes stay inside every engine profile, and buffers are sized for the profile maximum.
- One latent bug was found in K28's reference refiner (58.RefinerBatch). It ignores a refused `set_binding_shape` above 16 support objects.
  - It never triggered: no TensorRT shape error in any log, and dev parity is 100%.
  - It now fails loudly (64 bdd820e4, trtdyn 5953026c).

**Decision (user):**
- Wait for a 140 W USB-C PD supply. Then measure VDD_IN under load again and resume at MAXN with `scripts/k29_test_window_maxn.sh trt32`. Finished runs are skipped, so repetition 1 and the done part of repetition 2 stay valid.
- Fallback, prepared but not applied: lock the GPU at 905 MHz (`k29_test_window.sh trt32 905250000`; needs the user's sudo for devfreq max_freq). The table would then be a new, separately labelled one.

## 8. Engines and code (iii)

- **engines_md5.csv:**
  - C1 encoder engines: FP16 / FP32, buckets 1–32;
  - the three refiner engines;
  - the three dynamic crop verifiers;
  - the 14 whole-frame / cheap YOLOv8s engines used;
  - the C1 weights k28_temporal_head.pt (1b1e25d1).
  - Each run also writes the md5s it used into its meta.json.
- **code_md5.csv:** no git repository, so md5s of the code files.

| file | md5 |
|---|---|
| src/vtf/c1fast.py | b03f431a |
| src/vtf/trtdyn.py | 5953026c |
| scripts/64_k29_latency.py | bdd820e4 |
| scripts/61_k28_v6_latency.py (reference) | 6341d349 |
| engines/c1enc_dyn.onnx | 083cb050 |

- **Earlier versions of scripts/64_k29_latency.py:**
  - e73b1f9e: dev window 2 and the r1a / r1b runs;
  - 0850caf5: adds `--heur_only` and progress markers;
  - bdd820e4: the refiner guard, from nuScenes r1a onward.

  The pipeline code is the same in all three.

## 9. Status of the deliverables

1. Window C at MAXN: **DONE 2026-10-18 06:32** (45 runs, 0 failed): repetitions 2 and 3, the breakdown pass (iv), both heuristics + C1 (one heuristic per process, 3 repetitions).
2. `scripts/65_k29_report.py --sel trt32`: **DONE 06:34** → latency table (i: latency_fast.csv, latency_ref.csv, latency_runs.csv), tail (ii: tail.csv), timed parity (iii: parity_timed.csv), breakdown (iv: breakdown.csv), fronts with the dominating rows (v: front.csv, front_intervals.csv, front_{fast,ref}.{pdf,png}).
3. Final REPORT: this file. The test-frame G of the timing runs are post-hoc test evaluations (also listed in the kit).
4. Kit updated (03_results latency section, 06_pending, INDEX). The Jetson stays paused for K30 J (same window); it is released after K30 J.

## 10. Files in handoff/K29

| file | contents |
|---|---|
| STATUS.md | running log, rules, every decision |
| selection_dev.csv | dev selection timing, every candidate path |
| parity_dev.csv | R2 on the full dev splits: every pipeline × path, differing frames, dev G of both paths |
| c1stats_dev.csv | chains, pruned, crops encoded per path |
| latency_fast.csv, latency_ref.csv | **final** latency table, every pipeline × frame set, 3 repetitions: G, mean / median / p95 / p99 / max, px |
| latency_runs.csv | every run × pipeline × path |
| parity_timed.csv | identical actions per run × pipeline on the timed frames (378 rows, 0 differing frames) |
| front.csv, front_intervals.csv, front_{fast,ref}.{pdf,png} | latency-G fronts (mean and p95) with dominating rows; v6-S − row paired bootstrap |
| tail.csv | p95 / p99 / max of the key rows, both paths |
| breakdown.csv | per-component time of v6-S, both paths |
| engines_md5.json | engines used by the final runs |
| maxn_partial/ | interim (2026-10-15) repetition-1 tables; superseded |
| vin_reset.{pdf,png} | input voltage and rail power before the 17:16 reset |
| engines_md5.csv, code_md5.csv | engine and code md5s |

**Scripts:**
- scripts/64_k29_latency.py (timing / parity / breakdown);
- scripts/65_k29_report.py (tables);
- scripts/66_k29_interim.py (this report's rep-1 table and the voltage figure);
- scripts/k29_dev_window2.sh, k29_test_window_maxn.sh, k29_test_window.sh (GPU-cap variant), k29_vin_logger.py, k29_frames.py;
- engines/export_c1_encoder.py, engines/build_k29.sh.
