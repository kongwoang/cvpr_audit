**Updated 2026-10-04 (follow-up below): with the crop geometry of the evaluations (ceil32 canvases, as on VUW), ours is NOT slower than any whole-frame baseline with G ≥ ours** — nuScenes ours 30.2 ms / G 0.173 vs the only higher-G pass native 35.2 ms / 0.175; AV2 ours 53.0 ms / 0.545 vs native 67.8 ms / 0.553 — but passes with statistically equal G are faster (nuScenes u640 13.4 ms / 0.157, u960 19.2 ms / 0.152; AV2 u1536 44.3 ms / 0.543; all ours − it CIs include 0 except AV2 − u960 [+0.016, +0.280]). First measurement (static canvases): Yes: on both datasets ours is SLOWER than a whole-frame baseline with equal or higher G. nuScenes: u640 runs in 12.9 ms with G 0.157; ours runs in 28.1 ms with G 0.150. AV2: u1280 runs in 36.5 ms with G 0.506; ours runs in 49.2 ms with G 0.448. Neither G difference is significant.

# K5: Jetson latency on large frames, 2026-10-04

## What was run
- **Hardware and software:** Jetson AGX Xavier, `jetson_clocks` (GPU 1,377 MHz), TensorRT FP16, one process, sequential, CUDA syncs at stage boundaries.
- **Script:** `scripts/55_large_latency.py` (md5 103a611a…), launched by `scripts/k5_chain.sh`:
  ```
  scripts/py scripts/55_large_latency.py --dataset {av2,nusc}
  ```
- **Code versions:** rap commit caacd0f5 (clean); `src/vtf/speculative.py` md5 651da631…; `gpuprep.py` md5 13afabab….
- **Timing:** wall clock per frame, from the frame's upload to its decision.
- **Warm-up:** the first 200 frames are warm-up and are not reported.
- **Power:** board power from the INA3221 rails, sampled at about 100 Hz. Energy = mean power during the frame × its duration, idle power included. Idle board power: nuScenes 7.8 W, AV2 6.0 W.
- **Configurations:** frozen. nuScenes uses v3 + refiner v1; AV2 uses v4-lite + refiner v2. The refiners are TensorRT FP16 engines.
- **Detector:** the same frozen COCO YOLOv8s for every pipeline. Whole-frame engines run at the listed long side.

**Frames (not the confirmatory test splits; post-hoc engineering measurement):**
- **nuScenes:** CAM_FRONT, 1600×900. The 85 old DEEP scenes stored on the Jetson; they are **not** the new test split. 3,176 timed frames from 80 scenes.
- **AV2:** ring_front_center, 2048×1550, 10 val logs copied to the Jetson. 586 timed frames from 8 logs (the warm-up used the first 2 logs and part of a third).
- **Why G differs from the main table:** G here is computed on these frames with TensorRT FP16 detections, so it does not equal the main-table G. The main tables still define the accuracy claim.
- **CI:** paired bootstrap over scenes or logs, 2,000 resamples.

## Results (latency_nusc.csv, latency_av2.csv)

| nuScenes | G | mean ms | p95 | p99 | engine calls | network px (mean / p95) | mJ / frame | G: ours − it [95% CI] |
|---|---|---|---|---|---|---|---|---|
| cheap 320 | 0 | 8.7 | 10.4 | 11.3 | 1 | 0.043 | 213 | [+0.063, +0.269] |
| u640 | 0.157 | 12.9 | 14.5 | 15.4 | 1 | 0.171 | 315 | [−0.073, +0.059] |
| u960 | 0.152 | 19.7 | 21.4 | 22.3 | 1 | 0.384 | 482 | [−0.046, +0.045] |
| u1280 | 0.143 | 26.4 | 28.5 | 29.6 | 1 | 0.654 | 655 | [−0.038, +0.060] |
| u1600 native | 0.175 | 37.4 | 39.5 | 40.8 | 1 | 1.031 | 950 | [−0.126, +0.062] |
| ours, no refiner | 0.126 | 27.3 | 53.9 | 74.4 | 3.03 | 0.170 / 0.437 | 688 | [−0.001, +0.065] |
| **ours** | **0.150** | **28.1** | 56.7 | 78.1 | 3.03 (+refiner) | 0.170 / 0.437 | 709 | — |

| AV2 | G | mean ms | p95 | p99 | engine calls | network px (mean / p95) | mJ / frame | G: ours − it [95% CI] |
|---|---|---|---|---|---|---|---|---|
| cheap 320 | 0 | 12.6 | 13.1 | 25.9 | 1 | 0.026 | 311 | [+0.235, +0.642] |
| u640 | 0.031 | 18.2 | 19.2 | 35.1 | 1 | 0.103 | 453 | [+0.185, +0.647] |
| u960 | 0.415 | 25.3 | 26.8 | 44.7 | 1 | 0.232 | 640 | [−0.042, +0.117] |
| u1280 | 0.506 | 36.5 | 37.6 | 65.8 | 1 | 0.400 | 937 | [−0.162, +0.058] |
| u1536 | 0.543 | 47.0 | 49.3 | 88.0 | 1 | 0.573 | 1,240 | [−0.216, +0.034] |
| u2048 native | 0.553 | 75.3 | 82.7 | 96.2 | 1 | 1.012 | 2,075 | [−0.261, +0.057] |
| ours, no refiner | 0.438 | 41.3 | 85.5 | 120.2 | 4.25 | 0.146 / 0.378 | 1,131 | [−0.000, +0.021] |
| **ours** | **0.448** | **49.2** | 101.8 | 143.5 | 4.25 (+refiner) | 0.146 / 0.378 | 1,337 | — |

### Per-stage breakdown of ours (breakdown.csv; mean / p95 in ms)

| Stage | nuScenes | AV2 |
|---|---|---|
| Draft (cheap pass) | 7.7 / 8.7 | 11.5 / 12.8 |
| Crop preparation | 1.4 / 3.3 | 2.1 / 5.4 |
| Crop detection rounds | 4.1 / 8.0 (0.99 calls) | 8.6 / 20.0 (1.49 calls) |
| Guard re-detections | 3.4 / 13.6 (0.91 calls) | 3.9 / 14.7 (1.09 calls) |
| Lift | 0.8 / 1.9 | 1.3 / 3.1 |
| Refiner | 0.9 / 4.9 | 7.9 / 29.1 |
| **Residual: field + support + merge (CPU Python)** | **9.8 / 21.9** | **14.0 / 31.9** |

**Worst frames (worst_frame.csv):**
- nuScenes scene-0031, frame 10: 22 crops, 25 engine calls, 196 ms, network px 1.71. Its residual is 73 ms and its guard 71 ms.
- AV2 log 04994d08: 143–164 ms.

**Frames with no crops:** 11% on nuScenes, 22% on AV2.

**Per-frame latencies:** `per_frame_ours_{nusc,av2}.csv`. **Plot:** `scatter_latency_G.{pdf,png,csv}`, where the error bars run from the mean to the p95.

## Reading and caveats (against the method)
1. **Latency is not a win on large frames.** Ours uses 15–17% of native pixels but costs 2.2× (nuScenes) and 1.9× (AV2) the latency of u960. Its tail is long: p99 is 78 ms and 143 ms.
   - The extra time is spent outside the network: the CPU-side field, support and merge code (10–14 ms mean), the sequential guard calls, and on AV2 the refiner (8 ms).
   - Pixels are not wall-clock time on this board: small crop batches under-use the GPU.
2. **On these frames G does not separate ours from u960 / u1280.** All CIs include 0. The latency–G Pareto front is formed by the whole-frame pipelines.
   - The accuracy claims stay with the main tables, where these frames are not the evaluation split.
3. **Possible speed-ups were not done:** vectorising the residual, batching the guard calls, and an INT8 refiner. A faster implementation would need a parity check against the frozen decisions.
4. **Energy per frame** tracks latency, so ours uses more energy per frame than u960.

## Follow-up 2026-10-04: implementation fixes (decision-identical where stated)

**Script.** `scripts/57_large_latency_opt.py` (md5 see results/INDEX.md). It imports 55 unchanged and patches the implementation:
- **`--gpupost`** (`src/vtf/gpupost.py`): candidate selection on the GPU before the host copy (canvases ≥ 5,000 anchors), plus lean post-processing.
  - **Bit-identical outputs** (330 / 330 calls checked, batched too).
  - Applied to every pipeline, so whole-frame passes gain the most (native AV2 post-processing 9.0 → 2.8 ms).
- **`--dyncrops`** (`src/vtf/trtdyn.py`, `engines/build_dynamic.sh`): one dynamic-shape engine. Each crop runs on a canvas of exactly ceil32(crop × scale) with the frame letterbox, which is **the geometry of every VUW evaluation (main tables)**.
  - The first run's static canvases padded crops to the smallest fixed engine that held them. That changed detections (different padding) and doubled the crop pixels.
- **Profiling:** per engine call, about 6 ms of fixed overhead on this board (Python, resize kernels, post-processing) vs 2–3 ms of compute for a small crop.
- **Removing the breakdown syncs (`--nosync`)** saved only about 1 ms, so it is not used.

**Command:** `scripts/py scripts/57_large_latency_opt.py --dataset {nusc,av2} --gpupost --dyncrops`. Same frames as above.

**Runs:** `results/raw/20261004_143529_large_latency_opt_nusc`, `results/raw/20261004_144842_large_latency_opt_av2`. Tables: `latency_{nusc,av2}_v8s_dyn.csv`. Plot: `scatter_latency_G_v2.{pdf,png}`, with both measurements, data in `scatter_latency_G_all.csv`.

| | G | mean ms | p95 | p99 | engine calls | network px (mean / p95) | mJ | G: ours − it [95% CI] |
|---|---|---|---|---|---|---|---|---|
| nuScenes u640 | 0.157 | 13.4 | 14.8 | 15.4 | 1 | 0.171 | 318 | [−0.044, +0.076] |
| nuScenes u960 | 0.152 | 19.2 | 20.7 | 21.4 | 1 | 0.384 | 461 | [−0.021, +0.072] |
| nuScenes u1280 | 0.143 | 25.1 | 26.7 | 27.6 | 1 | 0.654 | 607 | [−0.006, +0.081] |
| nuScenes native u1600 | 0.175 | 35.2 | 37.1 | 38.0 | 1 | 1.031 | 874 | [−0.104, +0.085] |
| **nuScenes ours + ref** | **0.173** | **30.2** | 63.8 | 82.3 | 3.14 | 0.096 / 0.228 | 742 | — |
| AV2 u960 | 0.415 | 24.7 | 25.9 | 43.2 | 1 | 0.232 | 613 | **[+0.016, +0.280]** |
| AV2 u1280 | 0.506 | 35.0 | 35.7 | 62.0 | 1 | 0.400 | 886 | [−0.011, +0.115] |
| AV2 u1536 | 0.543 | 44.3 | 46.0 | 78.5 | 1 | 0.573 | 1,157 | [−0.046, +0.075] |
| AV2 native u2048 | 0.553 | 67.8 | 72.3 | 92.2 | 1 | 1.012 | 1,859 | [−0.083, +0.072] |
| **AV2 ours + ref** | **0.545** | **53.0** | 106.8 | 142.0 | 4.61 | 0.083 / 0.188 | 1,429 | — |

**Reading**
1. **With the evaluation's crop geometry, ours' G on these frames rises:** 0.150 → 0.173 (nuScenes) and 0.448 → 0.545 (AV2), at half the pixels (0.096 / 0.083). Ours now lies on or above the whole-frame latency–G front:
   - nuScenes: G of native at 86% of its latency;
   - AV2: between u1536 and native.
   The first run under-stated ours, because static canvases changed its detections.
2. **Against the method:**
   - Ours still does not beat equal-G whole-frame passes on latency (u640 / u960 on nuScenes, u1536 on AV2).
   - Its tail is long (p99 82 / 142 ms).
   - Energy follows latency.
   - The compute claim stays in pixels. Latency is "competitive with the front", not a win.
3. **Heavy-verifier operating point** (YOLOv8l for crops and all whole-frame passes; cheap draft YOLOv8s 320; dynamic v8l crop engine with max batch 2).
   - **AV2** (`latency_av2_v8l_dyn.csv`, run `results/raw/20261004_151612_large_latency_opt_av2`):

     | v8l pipeline | ms | G |
     |---|---|---|
     | ours + refiner | 80.5 (p99 237) | 0.377 |
     | ours, no refiner | 72.1 | 0.422 |
     | u960 | 63.3 | 0.089 |
     | u1280 | 93.1 | 0.461 |
     | u1536 | 128.6 | 0.555 |
     | native | 215.7 | 0.556 |

     Ours + refiner − u1280 = [−0.163, −0.036]. Ours (no refiner) is on the front between u960 and u1280 and dominates nothing.
   - **Refiner v2 lowers G on YOLOv8l boxes** (trained on YOLOv8s boxes).
   - **nuScenes with v8l could not be measured:** the v8l engines, together with the nuScenes devkit tables, exceed the Jetson's 16 GB of shared memory (CUDA out of memory while loading the u1280 engine; logs/k5_v8l_nusc.log). No number is substituted.
   - **Conclusion:** a heavier verifier does not turn ours into a latency win on this board. Unlike KITTI (small frames, earlier measurement), the large-frame crop sets still need about 4.6 engine calls per frame.

**Note (2026-10-05, K10):** every AV2 number in this report is unmasked. The AV2 ego hood drives 51% of the cheap draft's actions (81% of its loss); see handoff/K10 for the masked re-evaluation of the AV2 table.
