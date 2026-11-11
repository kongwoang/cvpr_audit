# K29 STATUS

**Task:** make C1 (learned looming estimate) and v3a (range refiner) fast on the Jetson AGX Xavier without changing any decision, then re-time every pipeline of the paper's latency table.
**Started:** 2026-10-14 23:30 +07.
**Phase (2026-10-15 19:30):** PAUSED. Step 1 and the dev parity are DONE (trt32 selected). Step 2 timing is partial (MAXN rep 1, part of rep 2), waiting for a new Jetson power supply. INTERIM REPORT written.

**Rules in force (task text):**
- R1. No configuration, threshold, weight or decision-rule change. The frozen v6-S (method_freeze_draft §2n) is the reference. No choice is made on a test number.
- R2. Decision parity is the acceptance criterion: ≥ 99.9% identical actions on every timed frame and on the full nuScenes new val and AV2 CFG-DEV splits, for v6-S, v6-S ≤ 2 rounds and every baseline that uses C1 or v3a. Report the parity rate, the number of differing frames, and dev G of both paths. Below 99.9%: fall back to FP32 for the offending part, or to the reference path, and say so.
- R3. Every pipeline that uses C1 / v3a gets exactly the same fast path and pruning as ours, timed in the same window.
- R4. Report every fast-path variant tried, with its parity and latency.

## What the reference path does (read from the code, 2026-10-14)

The reference is K28's Jetson implementation, `scripts/61_k28_v6_latency.py`:
- detectors are TensorRT FP16 engines;
- C1 runs as `TempHead` in torch FP32;
- v3a runs as `refiner_v3a_vuw_dyn.engine` (TensorRT FP16, dynamic batch 1–16).

On the Jetson, K28 S3 measured the fast path's effect on the detectors only. That is why parity in K29 is measured against this Jetson reference, with the detectors held fixed. The VUW torch-FP32 parity of the reference itself is K28's: 94–96% identical per-frame cost on AV2.

How the reference already batches, and what is actually slow:
- **C1 is batched per lift call, not per object.** `lift_c1` stacks every chained object of one call into one `head(xin, fin)`. That is one call per frame for whole-frame pipelines and one per round for the loop. The same holds on VUW (`k28_test.lift_c`).
  - **One call per frame is impossible for the loop.** Round k + 1's crops depend on round k's C1 ttc through the field and the support, so the loop keeps one C1 call per round.
  - **Every lift call re-encodes all chained objects.** In the loop that means the objects of earlier rounds too, and the t − 1 / t − 2 crops that were already encoded in earlier frames.
- **v3a is already one TensorRT FP16 call per frame**, over all support objects.
- **The crop extraction `crop64` is the hidden cost.** It runs `cv2.copyMakeBorder` on the WHOLE image for every crop: 1600×900 or 1550×2048 BGR copied 3× per chained object.

## Fast path (step 1), what is implemented

| item | implementation | exact? |
|---|---|---|
| a. batching | one encoder call per lift for all crops not yet cached, padded to batch buckets 1 / 2 / 4 / 8 / 16 / 32 (chunks of 32 above); one MLP call per lift; v3a one call per frame, padded to buckets 1 / 2 / 4 / 8 / 16 | numerics only (bucket padding rows are discarded) |
| b. engines | the C1 image encoder (flip, bilinear 64 → 128, normalisation, ResNet-18) exported to ONNX, then one static TensorRT engine per bucket (FP16; FP32 kept as the fallback). The MLP head stays torch FP32 on the GPU (≈ 0.4 MFLOP per object). v3a: static FP16 engines per bucket built from K28's ONNX. Second option: torch FP16 + one CUDA graph per bucket. | numerics only |
| c. caching | per pipeline, the encoder feature of crop64(image of frame f, box b) is stored under (f, exact box bytes) and reused at f + 1 / f + 2 and in later rounds of the same frame. The 64×64 crop itself is stored under the same key and shared by C1 and v3a. Frames older than f − 2 are dropped. | exact up to batch numerics |
| crop | `crop64_fast`: pads only the patch, not the image. The patch pixels are identical, so the resize output is bit-identical. | exact (checked bit-for-bit) |
| d. pruning | C1 is skipped for chained objects that are outside the corridor or beyond 80 m (on the lift's own z / lat); they get ttc = +inf. v3a is not pruned. | exact (argument below) |
| e. overlap | Round k's C1 needs round k's crop detections, and round k + 1's crops need round k's C1, so the loop has no detector ‖ C1 overlap. Only one overlap is allowed by the data dependency: encoding the crops of the non-truncated detections of a round on a second CUDA stream while the guard's grown-crop calls of that round run. This is speculative; unused features are discarded. Variant "overlap", reported separately. | exact up to numerics |

### Exactness of the pruning (d)

**Braking consumer.** DEEP's `planner.required_decel` (rap/planner.py l. 52–80) sets an object's demand to 0 unless `in_corridor(lat_min, lat_max) & (dist <= max_range)` holds.
- `max_range` = 80 m. ttc enters only `a_close`, which is computed for the selected objects.
- So an object outside the corridor or beyond 80 m has demand 0 for every ttc.

Every place where ttc can reach the decision:
1. **The action.** It is the max over the selected objects.
2. **`speculative.support`.** These are the objects whose demand reaches the threshold. A demand-0 object is never support.
3. **`dsf.insertion_grid` (the field).** It reads the scene only through `required_decel` → the action.
4. **The refiner (v3a / own).**
   - It runs on support objects only and rescales their z / lat_min / lat_max.
   - A pruned object has demand 0, so it is never support and never refined. Its refined geometry equals its unrefined geometry, and its demand stays 0 after refinement.
   - The refiner features do not include ttc.
5. **History.** It is box-only (`push` stores boxes), so it is unchanged.

Hence the actions of the loop rounds, the crops, the final action and the cost are identical with or without the pruned C1 values.

**TTC-warning consumer (K9, `instr_eval.c2_demand`).** Its demand is 1 / ttc only for `in_corridor & z <= max_range & z > 0`, and 0 otherwise. So the same pruning rule is exact for it too. It has no refiner.

**Rules NOT used:**
- **v ≤ 0.1.** It zeroes every braking demand but not the TTC-warning demand.
- **a_cap saturation.** A saturated object is support, so it is refined, and its demand can then depend on ttc again.

**Empirical confirmation.** The parity runs include a "fast without pruning" path; the difference with pruning must be 0 frames apart from numerics.

### Crop extraction measured (2026-10-15, CPU, AV2 / nuScenes frames)
- **crop64_fast is bit-identical to crop64.** Checked on 2,400 random boxes, including off-image boxes: 0 mismatches.
- **It saves little, though: 4.1 vs 5.1 ms per crop on random boxes up to 800 px.**
  - The cost is cv2.resize INTER_AREA on the large patch, not the border copy.
  - On realistic boxes, 90 crops take 25 ms (boxes ≤ 60 px high), 44 ms (≤ 150 px) and 146 ms (≤ 400 px).
- **Two exact remedies:**
  - **The bank**, which cuts each crop once.
  - **A thread pool of 6 for ≥ 8 missing crops** (cv2 releases the GIL). It gives 28 / 48 ms for the last two cases above, with identical output.
- **GPU resizing is rejected.** It is not bit-identical to cv2 INTER_AREA, and it would change C1's input preprocessing.

### Variant selection rule (fixed before any timing; R1, R4)
- **Candidate paths:**
  - trt16, trt32, t32, t16g: encoder as named, with pruning;
  - trt16np: trt16 without pruning, for the exactness check only;
  - trt16ov: trt16 + overlap.
- **Selection.** The fast path of the timing window is the candidate with the lowest mean latency of v6-S on dev frames, among the candidates that meet R2 (≥ 99.9% identical actions) on the full dev splits for every C1 / v3a pipeline.
  - Dev frames for the latency: nuScenes new val, first 20 scenes; AV2 CFG-DEV, first 10 logs.
- **Fallbacks.** If no candidate meets R2, use trt32 for the part that fails. If that still fails, use the reference path for that part.
- **Every candidate's parity and dev latency is reported (R4).**
- Test frames (nuScenes new test 20, AV2 val logs) are timed only with the selected path and the reference path.
- **Aggregation, written 2026-10-15 ≈ 01:16.** I had read the nuScenes selection results, but not the AV2 ones (that run ended at 01:15:42). The tie rule was added at this point, after I had seen that trt32 was faster than trt16 on nuScenes; it is disclosed as such.
  - "lowest mean latency" = the lowest average of the two per-dataset v6-S mean latencies (nuScenes val 20 scenes, AV2 dev 10 logs; equal weight). trt16np is not a candidate, as above.
  - **Ties:** if two candidates are within 0.5 ms of each other on that average, the one with the higher precision is taken (FP32 before FP16).

## Data on the Jetson (separate tree ~/datasets/k29; DEEP's datasets untouched)

- **nuScenes new val:** 152 scenes. **nuScenes new test:** the first 20 scene tokens in sorted order (list in ~/datasets/k29/k29_nusc_test20.txt; the first is scene-0187, token 019b370a…). Together 6,922 CAM_FRONT keyframes, 1.04 GB.
- **AV2 CFG-DEV:** 128 train logs (md5 of the log name starts with 0 / 1 / 2, as `instr_eval.units_av2("dev")`). Only the images at the stride-2 frames + metadata are copied: 10,635 files, 3.15 GB.
- Copy: `scripts/k29_copy_data.sh`, 8 parallel rsync streams per list. The link runs at ≈ 0.5 MB/s per stream. Log: logs/k29/copy_data.log.

## Plan

1. Export the C1 encoder → ONNX (Jetson, CPU). Write `src/vtf/c1fast.py` and `scripts/64_k29_latency.py` (= 61 + fast path + parity mode + dev frame generators + heuristics).
2. **Jetson window A (builds, exclusive, one at a time, workspace ≤ 1024):** C1 encoder FP16 / FP32 per bucket; v3a FP16 per bucket; YOLOv8s 448×352 (AV2 u448, not built before). Then a smoke parity on a few hundred frames.
3. **Dev parity (Jetson, exclusive, no timing):** nuScenes new val and AV2 CFG-DEV with the ego mask.
   - Every C1 / v3a pipeline in both paths. Detections are shared between the paths (crop / frame detection cache), so only C1 / v3a differ.
   - Report the parity rate, differing frames, and dev G of both paths.
4. **Jetson window B (timing):** locked clocks, fan at 100%, 3 repetitions, on three frame sets:
   - nuScenes new test 20 scenes (warm-up pass of 200 frames first, discarded);
   - nuScenes old DEEP 20 scenes (K28's set: 25 scenes, first 200 frames warm-up);
   - AV2 8 val logs with the ego mask (K28's 10 logs, first 200 frames warm-up).

   Pipelines:
   - cheap;
   - whole frame u448 / u512 / u640 / u960 / u1280 / native: plain, + T4 + ref, + C1 + ref (fast and reference);
   - v6-S ≤ 3 rounds and ≤ 2 rounds, with and without refiner (fast and reference);
   - v5.1-S;
   - the dev-best S heuristic + C1: YOLOv8s tsp2@S + C1 + v3a, and YOLO11m tsp8@S + C1 + own refiner (the best S heuristic over both detectors);
   - v6-M and v6-L.

   Then one extra instrumented repetition of v6-S for the per-component breakdown.
5. REPORT (i)–(v), plus the K6 three-run table at the top once K28 runs 2 / 3 are done.

## Log
- 2026-10-14 23:30 — task received. Read 61_k28_v6_latency.py, speculative.py, rap.planner, instr_eval (c2 consumer), k28_test.py (heuristics).
- 2026-10-14 23:45 — data lists written on VUW (tmp/k29_*.txt); copy started (`k29_copy_data.sh`). One of the 16 rsync streams was refused by VUW's sshd. `k29_copy_retry.sh` re-runs every list once the first pass ends.
- 2026-10-14 23:57 — `engines/export_c1_encoder.py`: c1enc_dyn.onnx, md5 083cb050.
- 2026-10-14 23:58 — **Jetson window A opened.** ~/.jetson_pause set; DEEP (risk-aware-perception-a5) informed and acknowledged. MAXN, fan manual 100%, 0 throttle steps, 36 °C. `engines/build_k29.sh` started: 12 C1 encoder engines, then YOLOv8s 448×352.
- Code: `src/vtf/c1fast.py`, `scripts/64_k29_latency.py`, `scripts/k29_smoke.sh`.
- 2026-10-15 00:23 — **builds DONE**, no reset, ≈ 1–2.5 min per engine.
  - 12 C1 encoder engines (FP16 / FP32 × buckets 1–32) and yolov8s_crop_448x352.engine (md5 b64fa5b9).
  - The md5 of every engine is written into each run's meta.json.
- 2026-10-15 00:00 — data copy DONE and verified (`COPY VERIFIED`): 6,922 nuScenes keyframes; AV2 dev complete, 0 missing images in the runs.
- 2026-10-15 00:28–00:43 — **smoke (`k29_smoke.sh`), dev only:**
  - **Parity mode, every candidate path: 100% identical actions in every pipeline.**
    - nuScenes val, 3 scenes, 119 frames: u448 / u960 + T4 + ref, + C1 + ref, ours, ours ≤ 2.
    - AV2 dev, 2 logs, 157 frames: u512 / u960, ours, ours ≤ 2.
    - v5.1-S + both heuristics on the same 157 AV2 frames.
  - **crop64_fast against crop64:** 363 + 1,305 crops checked in the runs, 0 mismatches.
  - **C1 work in the 119 nuScenes frames** (summed over the C1 pipelines):
    - 255 chains, of which 177 pruned (69%);
    - crops encoded: 99 with pruning, 370 without, against 765 in the reference (3 per chain per lift call).
  - **Timing smoke, nuScenes val, 200 frames.** GPU clocks were NOT locked yet (devfreq governor), so these numbers are indicative only:

    | pipeline | reference ms | trt16 ms | parity |
    |---|---|---|---|
    | u960 + C1 + ref, mean | 47.4 | 20.6 | 100% |
    | u960 + T4 + ref, mean | 20.1 | 20.2 | 100% |
    | v6-S, mean / p95 | 44.6 / 110 | 34.8 / 80 | 100% |
    | v6-S ≤ 2, mean / p95 | 34.7 / 79 | 26.2 / 45 | 100% |

- 2026-10-15 00:44 — **`jetson_clocks` applied:** GPU 1,377 MHz, EMC 2,133 MHz, CPU 2,265.6 MHz, all locked.
  - K5 used jetson_clocks. **Whether K28's final window (§16) did is not recorded**, and the board reset twice during it, which undoes jetson_clocks. K29 therefore re-times the reference path in the same window rather than reusing K28's numbers.
- 2026-10-15 00:44 — **window B (`scripts/k29_dev_window.sh`)** started: dev selection timing, then full dev parity for every candidate path.
- 2026-10-15 01:15 — **dev selection timing DONE** (nuScenes val 20 scenes, 595 timed frames; AV2 dev 10 logs, 584 timed frames).

  v6-S mean latency (ms) per candidate:

  | path | nuScenes | AV2 | average |
  |---|---|---|---|
  | reference | 37.36 | 91.85 | 64.61 |
  | trt16 | 26.62 | 46.89 | **36.76** |
  | trt32 | 26.53 | 47.16 | 36.84 |
  | t16g | 26.69 | 47.53 | 37.11 |
  | trt16ov | 27.22 | 48.69 | 37.95 |
  | t32 | 27.69 | 53.79 | 40.74 |
  | trt16np (no pruning; not a candidate) | 28.11 | 48.29 | — |

  - **Selected: trt32.** trt16 is lowest, but trt32 is within 0.5 ms of it, so the tie rule takes trt32. The selection is final only once trt32 passes R2 on the full dev splits.
  - **Parity on the selection frames: 100% for every path except t16g.**
    - u960 + C1 + ref: 96.3% on nuScenes (22 differing frames in 3 scenes), 97.6% on AV2 (14 frames).
    - v6-S: 98.8% (nuScenes) and 99.1% (AV2).
    - dev G: nuScenes v6-S 0.095 → −0.263; AV2 0.361 → 0.359.
    - Its cause (a capture bug or torch FP16 precision) is under investigation.
  - **The overlap variant (trt16ov) is slower than trt16** (+0.6 / +1.8 ms). The speculative encodes cost more than the overlap with the guard calls saves.
- 2026-10-15 01:15 — full dev parity started. The process RSS is ≈ 10 GB (NuScenesDB plus 7 paths), with 3 GB available.
  - From the next process on, the refiner engine and the C1 head are shared across the fast paths (memory only; md5 of 64_k29_latency.py 125e937d, c1fast.py 7c595f18).
  - A memory watchdog is running.
- 2026-10-15 01:18 — **window B stopped by hand: memory.** The first parity process (nuScenes val, u448 / u512 / u640, 7 paths) reached 13.1 GB RSS with 0.5 GB available. I killed it (exact PIDs of the window script and its python) before the board ran out of memory.
  - **Cause, measured:** the NuScenesDB (v1.0-trainval) alone takes ≈ 9 GB peak RSS on the Jetson, and every nuScenes process loaded it. K28's final window also loaded it in its nuScenes runs (61 → 55.frames_nusc). That may be related to the 04:35 hard reset there (RAM 91–92%); this is not verified.
  - **Fix: `scripts/k29_frames.py`** writes the nuScenes frame lists once to ~/datasets/k29/frames_nusc_{test20,val,old}.pkl, using the same code as the generators; the runs read those.
    - test20: 805 frames, the same count as VUW's per-frame results for these 20 scenes.
    - val: 6,117 frames in 152 scenes.
    - old: 997 frames in 25 scenes = K28's set (797 timed after the 200 warm-up frames).
  - **Also:** `torch.cuda.empty_cache()` every 100 frames in parity mode; the refiner engine and the C1 head are shared by the fast paths.
  - **Memory test** (6 val scenes, 238 frames, 7 paths): RSS stable at 5.5 GB, ≈ 8.5 GB available.
- 2026-10-15 01:27 — **t16g was a bug, not FP16.** The CUDA graphs read the weights of a half-precision module that was not kept alive after `Encoder.__init__`, so they read freed memory. Encoder features against torch FP32, on 160 real AV2 crops:

  | variant | relative error | max \|feature\| |
  |---|---|---|
  | t16g, before the fix | 68 | 1,404 |
  | t16g, fixed (c1fast md5 b03f431a) | 3.7e-3 | 11.3 |
  | t16 eager | 3.7e-3 | |
  | trt16 | 3.0e-3 | |
  | trt32 | 6.8e-7 | |

  - The fixed t16g gives 100% parity on the 238 memory-test frames.
  - The parity of the buggy t16g on the selection frames (above) is reported as a bug, not as an FP16 result.
- 2026-10-15 01:30 — **window B restarted (`scripts/k29_dev_window2.sh`; 64_k29_latency.py md5 e73b1f9e):**
  - first, the fixed t16g re-timed on the selection frames (with the reference);
  - then the full dev parity, every candidate path.

  Even with the latency of the first t16g timing (37.11 ms average), the rule still selects trt32: t16g is FP16, and trt32 is within 0.5 ms of the fastest candidate.
- 2026-10-15 01:42 — **fixed t16g re-timed on the selection frames:**
  - v6-S: 27.04 ms (nuScenes) / 48.98 ms (AV2), average 38.01 ms.
  - Parity 100% on every pipeline (u960 + T4 / + C1 + ref, v6-S, v6-S ≤ 2, with and without refiner).
  - The reference re-timed in the same runs: 37.35 / 93.77 ms, against 37.36 / 91.85 in the first selection runs.
  - **The selection stays trt32**, conditional on the full dev parity.
- 2026-10-15 01:35 — **label check, nuScenes new test 20 scenes** (labels only, no pipeline output read).
  - The Jetson frame list matches VUW's phase-2 per-frame results on all 805 frames: same frames, same ego speed (max |Δ| 2e-15), same ground-truth action.
- 2026-10-15 04:40 — **full dev parity, nuScenes new val (152 scenes, 6,117 frames): DONE.**
  - **Selected path trt32: 100% identical actions in every pipeline**, 0 differing frames:
    - whole frame u448 … native, + T4 + ref and + C1 + ref;
    - v6-S and v6-S ≤ 2, with and without refiner;
    - v5.1-S and v5.1-S ≤ 2;
    - both heuristics + C1, with and without refiner;
    - v6-M and v6-L, ≤ 3 and ≤ 2 rounds.
  - **The same holds for t32, and for trt16 / t16g / trt16np / trt16ov**, except one frame in each of these rows (99.98%, still above R2's 99.9%):
    - u1280 + C1 + ref, for t16g and trt16np;
    - native + C1 + ref, for trt16, t16g, trt16np and trt16ov.
  - **Pruning, in the u448 / u512 / u640 + ours process:**
    - 42,478 of 52,905 chains skipped (80%);
    - crops encoded: 13,212 with pruning, 71,993 without, 158,715 in the reference.
  - **trt16np (trt16 without pruning) is as exact as trt16:** 100% in every row, except the same single native + C1 + ref frame, plus one u1280 + C1 + ref frame (trt16 has 0 there). These single-frame differences come from FP16 numerics under a different batch composition, not from the pruning.
  - **crop64_fast against crop64:** 537 crops checked, 0 mismatches.
  - Then AV2 CFG-DEV started (04:40).
- 2026-10-15 07:55 — **full dev parity, AV2 CFG-DEV, process a: DONE.** Process a covers u448 / u512 / u640 (+ T4 + ref, + C1 + ref) plus v6-S and v6-S ≤ 2, with and without refiner; 128 logs, 10,123 frames.
  - **Selected path trt32: 100% identical actions in every row**, 0 differing frames. The same holds for t32.
  - **Other paths, all still above R2's 99.9%:**
    - u448 + C1 + ref: 1 frame (99.99%) for trt16, t16g, trt16np and trt16ov;
    - t16g: 2 frames in v6-S (99.98%) and 1 in v6-S ≤ 2;
    - trt16np and trt16ov: 1 frame each in v6-S and in v6-S without refiner (99.99%).
    - *(Corrected 14:38. The first version of this entry said trt16 was also 100%; it had missed the u448 rows.)*
  - **Pruning:** 177,303 of 240,272 chains skipped (74%). Crops encoded: 71,478 with pruning, 273,762 without (trt16np).
  - **Overlap:** trt16ov encoded 62,848 crops on demand plus 13,984 speculatively.
  - **crop64_fast:** 3,777 crops checked, 0 mismatches.
  - Process b (u960 / u1280 / native) started 07:55, then c, d and e.
- 2026-10-15 09:55 — **AV2 CFG-DEV, process b (u960 / u1280 / native, + T4 + ref and + C1 + ref): DONE.**
  - **trt32 and t32: 100% in every row.**
  - **One frame differs (99.99%) in these rows:**
    - u960 + C1 + ref, for trt16, t16g and trt16ov;
    - u1280 + C1 + ref, for trt16np.
  - **Pruning:** 180,734 of 229,486 chains skipped. Crops encoded: 60,620 with pruning, 280,280 without.
  - Process c (v5.1-S + both heuristics + C1) started at 09:55.
- 2026-10-15 11:56 — **AV2 CFG-DEV, process c: DONE.** Process c covers v5.1-S and v5.1-S ≤ 2, plus both heuristics + C1, with and without refiner.
  - **trt32, trt16, t32 and trt16ov: 100% in every row.**
  - **Other paths, in tsp2S + C1:** t16g differs on 2 frames (99.98%); trt16np on 1 frame (99.99%).
  - **Pruning:** 39,084 of 55,809 chains skipped. Crops encoded: 21,387 with pruning, 71,851 without.
  - Process d (v6-M) started at 11:56, then e (v6-L).
- 2026-10-15 13:16 — **AV2 CFG-DEV, process d (v6-M, ≤ 3 and ≤ 2 rounds, with and without refiner): DONE.**
  - **Every path: 100% in every row**, 0 differing frames.
  - **Pruning:** 47,550 of 71,610 chains skipped. Crops encoded: 22,788 with pruning, 69,204 without.
  - Process e (v6-L) started at 13:16; it is the last dev run.
- 2026-10-15 14:37 — **AV2 CFG-DEV, process e (v6-L): DONE.** Every path: 100% in every row. Pruning: 46,857 of 70,975 chains skipped; crops encoded 23,080 with pruning, 69,026 without.
- 2026-10-15 14:37 — **full dev parity DONE (window 2, 12 runs, all rc 0).**
  - **Summary over all `parity.csv` rows of both dev splits:**
    - trt32 and t32: minimum parity 1.0 in every row.
    - trt16, t16g, trt16np and trt16ov: minimum 0.99980 (at most 2 differing frames).
    - Every candidate path passes R2.
  - **Selection confirmed: trt32**, by the rule fixed before the parity runs (fastest within 0.5 ms → higher precision). It is also the only TensorRT path with 0 differing frames on both dev splits.
  - Next: window C (`scripts/k29_test_window.sh trt32`), the timing of every pipeline on the timed frame sets, reference and trt32 in the same processes.
- 2026-10-15 14:38 — **pre-launch smoke test of window C** (tag `smoke`, ignored by 65): v6-S + u448 timing on 40 frames of nusc_test20, nusc and av2, plus the breakdown mode on av2. All rc 0, 0 missing images, trt32 identical to ref on every smoke frame. Smoke latencies are not reported.
- 2026-10-15 14:40 — **window C started: `scripts/k29_test_window.sh trt32`** (md5 f93ad6f0; 64 e73b1f9e; c1fast b03f431a), log logs/k29/test_window.log, per-run logs in logs/k29/runs/. ~/.jetson_pause held, jetson_clocks on. 3 repetitions × {nusc_test20, nusc, av2} × processes a–e, then one breakdown pass per dataset.
- 2026-10-15 14:42 — **scripts/65_k29_report.py run selection fixed** (md5 40015843):
  - selection_dev takes both the `select` and `select2` runs; the buggy t16g rows of `select` are labelled "t16g (lifetime bug, not reported as FP16)";
  - parity_dev takes only the tags a–e (no smoke, no memtest);
  - the breakdown takes only the tag `bd`.
  - Dev outputs written: selection_dev.csv, parity_dev.csv (384 rows, 10 runs), c1stats_dev.csv.
  - Differing frames, summed over all 32 rows per split: trt32 0 / 0 and t32 0 / 0 (nuScenes / AV2); trt16 1 / 2; trt16ov 1 / 4; t16g 2 / 7; trt16np 2 / 5.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         - 2026-10-15 ~14:58 — **board hard reset #1 of window C**, about 40 s into the first v5.1-S + heuristics process (nusc_test20, tag r1c).
  - Agent history at 5 s resolution, up to 14:57:50: GPU 36–49%, power 17–21 W, max temperature 53 °C, memory 44%; fan manual 100%; no throttling.
  - The run log stops after the `run:` line; the rest is NUL bytes, i.e. lost on the reset.
  - kern.log and syslog show nothing but the usual `ata1: SATA link down` lines (1.3 M of them; the SATA port is empty). No OOM, panic or thermal message.
  - r1a and r1b (v6-S + u448–u640; u960 / u1280 / native) had finished at 14:57; their frames.pkl are intact.
  - The reset undoes the fan and jetson_clocks settings. The fan was set back to manual 100%; jetson_clocks is re-applied by the window script.
- 2026-10-15 15:05 — window C restarted, finished runs skipped (k29_test_window.sh 7e09aa72). The log of the first part is logs/k29/test_window_part1.log.
- 2026-10-15 ~15:06 — **board hard reset #2, in the same process** (r1c restarted at 15:05:11). Its run log is empty, and the agent history before the reset was lost.
  - Two resets within minutes of the start of the same process point at that process, not at chance. The dev window ran 13 h without a reset, including the same v5.1-S + heuristics process in parity mode, on both dev splits.
  - **This process had never been timed on the Jetson before.** K28's final window timed v5.1-S without the heuristics.
- 2026-10-15 15:13 — **window C restarted, reordered** (user: "chạy tuần tự cho đỡ sập"). Log of the second part: logs/k29/test_window_part2.log. Scripts: k29_test_window.sh 7d34ac25; 64 0850caf5, which adds `--heur_only` (one heuristic per process) and setup / early-frame progress markers, with no change to any pipeline.
  - **New order:**
    - per repetition and dataset: v6-S + u448–u640; u960 / u1280 / native; v5.1-S alone (tag c0); v6-M; v6-L;
    - then the breakdown pass;
    - then the heuristics last, one heuristic per process (h1 = tsp2S + C1, h2 = tsp8S-yolo11m + C1).
  - **Instrumentation of the heuristic runs:** their logs are written and synced line by line, and tegrastats runs at 200 ms with sync. A further reset would leave the last stage and the power trace on disk.
  - The empty run directories of the two crashed r1c attempts have no frames.pkl, so 65 ignores them.
- 2026-10-15 15:20 — **code review of the v5.1-S + heuristics process** (user: "xem code phần đó có lỗi gì ko").
  - **No bug found that can explain a board reset.**
    - Every dynamic-engine caller keeps shapes inside the engine profiles: 58.Crops and 59.OneCall cap canvases at 1024 and chunk by the profile batch (yolo11m_dyn 2, yolov8s_dyn 8); RefFast chunks by 16; the C1 encoder by 32. DynTRT buffers have the profile's maximum size, so even a refused shape cannot write out of bounds.
    - The v5.1-S verifier is yolov8s, so the process holds a single yolo11m context.
    - Memory at reset #1 was 44%.
  - **Difference from the dev runs that did not reset:** in parity mode, the heuristics' crop detections are cached per frame, so yolo11m ran once per frame. In timing mode, ref and trt32 each run it, back to back, with clocks locked. That is more sustained GPU load, but the 5 s average power was only ~20 W.
  - **Latent bug, K28 reference code:** 58.RefinerBatch (and 59's copy) sends all support objects of a frame in one call and ignores the return value of `set_binding_shape`. The refiner engines' profile maximum is 16. Above 16 support objects, TensorRT would keep the previous shape and return uninitialised corrections.
    - Never triggered: no TensorRT shape error in any log under logs/, and trt32 (which chunks by 16) is identical to the reference on every dev frame.
    - It cannot reset the board (all buffers are in bounds).
  - **Guards added (fail loudly; no change when shapes are valid):**
    - 64 wraps the reference refiners: an error above the profile batch (64 md5 bdd820e4);
    - DynTRT raises when `set_binding_shape` is refused (trtdyn md5 5953026c).
    - They apply from window C's nusc r1a onward. The earlier runs used: r1a and r1b, 64 e73b1f9e; r1c0, r1d and r1e, 64 0850caf5. The pipeline code is the same in all three versions.
- 2026-10-15 16:53 — **board reset #3 of window C**, during nusc r2b (whole frame u960 / u1280 / native, v6-S config): about 1.5 min into the run, at 200 frames. **Not the heuristics process.**
  - Agent history at 16:53:00: power 30.4 W (peak 31.2 W), 63 °C, memory 44%.
- 2026-10-15 17:10 — **the reset reason is SYS_RESET_N**, read from /sys/devices/platform/c360000.pmc/reset_reason and from the `tegra-pmc: PMC reset source` boot line.
  - SYS_RESET_N is the external hardware reset line. Software reboots show MAINSWRST; watchdog resets show BCCPLEXWDT.
  - The kern.log history has 22 SYS_RESET_N boots since 2026-09-06 (and 2 BCCPLEXWDT). That includes every reset during K28's builds and windows, and the three of today.
  - **Forum and design-guide evidence** (NVIDIA developer forums):
    - The Xavier VDDIN_PWR_BAD_N circuit "is designed to start the shutdown as soon as a voltage drop of ~0.5V is detected" (Design Guide, dV/dt chapter, quoted in the thread "Detecting undervoltage"). That thread reports SYS_RESET_N after every unexpected power-off.
    - NVIDIA staff attribute such cuts to dV/dt / power-supply quality ("AGX Xavier power cut suddenly"); users fixed MAXN-load reboots by replacing the adapter, cable or barrel jack.
- 2026-10-15 17:15 — window C restarted, with an input-voltage logger: scripts/k29_vin_logger.py reads the INA3221 bus voltage (VDD_IN) and rail currents at ~50 Hz and writes fsynced 1 s min / mean lines to logs/k29/vin_*.log. Log of the previous part: logs/k29/test_window_part3.log.
- 2026-10-15 17:16:01 — **board reset #4, 4 s after the GPU load of nusc r2b started.** The board stayed off until 18:02 (no agent samples 17:16–18:02).
  - **Input voltage (INA3221 averages, so short dips are deeper than shown):**

    | condition | VDD_IN | rail power |
    |---|---|---|
    | idle | 18.72 V | 2.3 W |
    | earlier runs, light load | 18.61–18.64 V | 7–9 W |
    | r2b, first seconds of GPU load | **18.32–18.36 V (min 18.320)** | 24–26 W |
    | reset | — | 17:16:01–02 |

    - The averaged drop is already ~0.4 V at about 26 W of rail power (≈ 0.25–0.3 Ω effective source resistance). A GPU load step of a few amps then crosses the ~0.5 V power-bad threshold.
    - Average power is far below the 65 W supply rating, so this is supply droop (adapter / cable / barrel jack / outlet), not overload. It fits SYS_RESET_N, no kernel message, power-off rather than reboot, and resets only under GPU-heavy load with locked clocks.
  - **Window C is paused, pending the user's decision** (hardware check of the supply, or a lower clock cap that changes the latency numbers). ~/.jetson_pause kept.
- 2026-10-15 18:40 — **user decision: "Chạy tiếp, hạ clock GPU".** Window C restarts from scratch at a capped GPU clock.
  - Runs at different clocks are never mixed. The 1377 MHz runs (rep 1 complete without heuristics, rep 2 partial) are kept as a partial MAXN record: `65 --out_sub maxn_partial`, tables in handoff/K29/maxn_partial/.
  - **Cap: 905.25 MHz**, NVIDIA's 30 W-mode GPU cap, about half the GPU power. CPU (8 × 2265.6 MHz) and EMC (2133 MHz) stay at the MAXN maxima.
  - Setting it needs root on the GPU devfreq max_freq, which the user runs (sudo NOPASSWD covers only nvpmodel -m and jetson_clocks; every lower nvpmodel mode also caps the CPU). jetson_clocks then locks GPU min = max = 905.25 MHz.
  - **Scripts:**
    - k29_test_window.sh (md5 0eeebcf4) takes `<sel> <gpu Hz>`, suffixes every tag with _g905, checks the GPU cap before each run and stops if it is lost (a reset restores 1377 MHz), and runs the input-voltage logger throughout.
    - The 1377 MHz version is kept as k29_test_window_maxn.sh (7d34ac25).
    - 65_k29_report.py (md5 2cf0b693) gains `--tagsuf` and `--out_sub`.
  - **Consequence for the report:** the K29 latency table is at GPU 905 MHz. Ref and the fast path are timed in the same processes, so their comparison holds; absolute numbers are not comparable with K5 / K28 (MAXN, 1377 MHz).
- 2026-10-15 18:55 — **user: wait for a 140 W USB-C PD supply instead of capping the GPU** ("hay tôi đợi nguồn 140W PD usbc vào có được không" … "ok, thế tạm thời pause mấy cái trên jetson").
  - The Jetson stays paused: ~/.jetson_pause kept, nothing running. The GPU cap was NOT applied; the user did not run the sudo command.
  - **Resume plan once the PD supply is in:**
    1. idle and loaded VDD_IN with scripts/k29_vin_logger.py; load = nusc r2b, the run that reset the board twice; target a droop well below 0.5 V;
    2. then `scripts/k29_test_window_maxn.sh trt32` (MAXN 1377 MHz, finished runs skipped, so the 1377 MHz runs done so far stay valid);
    3. if the droop persists, fall back to the 905 MHz cap (`k29_test_window.sh trt32 905250000`, tags _g905).
  - QUEUE and 06_pending updated (K29 PAUSED).
- 2026-10-15 19:30 — **INTERIM REPORT written** (user: "ok, chạy đi. rồi coi như xong 1 phần K29"): handoff/K29/REPORT.md.
  - Contents: K6 three-run table; summary; fast path; every variant (R4); full dev parity; C1 work; repetition-1 MAXN timing; reset diagnosis; md5s; what remains.
  - New files:
    - engines_md5.csv and code_md5.csv;
    - maxn_partial/ (65 with `--out_sub maxn_partial`);
    - maxn_partial/latency_rep1.csv and vin_reset.{pdf,png} (scripts/66_k29_interim.py, md5 4dc0858e).
  - Kit: 03_results (K29 interim bullet under S3), results/INDEX.md, QUEUE.
- 2026-10-18 03:00 +07 — **Window C resumed at MAXN (user: "cho chạy nốt các job còn lại trên jetson").** The new supply is in.
  - Idle VDD_IN 19.82 V (old supply 18.72 V).
  - 45 s CPU + GPU stress at about 56 W rail power: min VDD_IN 19.35 V (droop 0.47 V at 56 W; the old supply drooped 0.4 V at only 26 W). No reset, throttled false.
  - Setup: MAXN, jetson_clocks (GPU 1377 MHz, CPU 2265.6 MHz, EMC 2133 MHz), fan manual 255. ~/.jetson_pause kept (busy.sh "paused"); the DEEP session (risk-aware-perception-1d) was told and acknowledged.
  - Voltage logger logs/k29/vin_1018_030152_window.log runs throughout.
  - `scripts/k29_test_window_maxn.sh trt32` restarted: the done runs were skipped, and it resumed at nusc r2b (the run that reset the board twice). That run held at 32.6 W peak with min VDD_IN 19.56 V (droop 0.26 V).
  - Queued scripts/k29_after_window.sh: after "K29 TEST WINDOW DONE" it runs 65_k29_report.py --sel trt32 (final MAXN tables; logs/k29/after_window.log). The Jetson stays paused afterwards: a session releases it (rm ~/.jetson_pause) and tells DEEP. Estimated window length ≈ 10 h (rep 2 rest, rep 3, breakdown, heuristics h1 / h2 × 3 reps). The paper's '[TODO: K29: repetitions 2 and 3]' (§4.6, Table 3) is filled from these tables.
- 2026-10-18 07:30 +07 — **Window C DONE (06:32, 45 runs, 0 failed, 0 resets) and FINAL REPORT written.**
  - 65_k29_report.py --sel trt32 ran at 06:34 (logs/k29/after_window.log, exit 0): latency_fast / latency_ref / latency_runs, parity_timed (378 rows, 0 differing frames), tail, breakdown, front / front_intervals, front_{fast,ref}.{pdf,png}.
  - Input voltage over the window (03:12–06:32): min VDD_IN 19.57 V at 30 W, max rail power 31 W.
  - Result (post-hoc test G): v6-S on no latency-G front; v6-S ≤ 2 rounds on the nuScenes test 20 mean front and on both AV2 fronts; u960 + C1 + ref on both nuScenes test 20 fronts (ties v6-S); on AV2 tsp2@S + C1 + ref above v6-S by 0.012 at the same mean. S3 still not met.
  - REPORT.md: FINAL (§1 summary, §6a–c, §7 resolution, §9, §10). Kit: 03_results (K29 FINAL bullet replaces the interim one), 06_pending, results/INDEX.md, QUEUE.
  - The Jetson stays paused: the K30 J chain (scripts/k30_j_window.sh) started at 06:36 in the same window; release after it.
