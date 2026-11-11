# K7 — Argoverse-HD val (AV2 configuration unchanged), 2026-10-04

**What was run.** `remote/A_method/argoverse_hd_eval.py` (md5 583f7b55…; adapter `src/vtf/av1.py` md5 8c8debf8…), chain
`argoverse_hd_chain.sh` on VUW: `--stage run --shard i/4` (GPU, torch FP16, frozen COCO YOLOv8s) then `--stage score`.
Data: Argoverse-HD val = Argoverse 1.1 tracking val, ring_front_center 1920×1200 @ 30 Hz, 24 sequences, 15,062 frames
(images, ego poses, calibration and 3D amodal labels stream-extracted from the AV1.1 tracking-val tar, because the
Argoverse-HD S3 buckets are closed; 2D annotations = the authors' COCO-format package). Configuration: AV2's final one
(v4-lite: v_close 0, crops at 960 / long side = 0.5, guard, merged-scene history, ≤ 3 rounds, refiner v2) — **no
Argoverse-HD tuning; this is the first and only run.** Source CSVs: results/vuw_summaries/argoverse_hd/summary_{ap,G}.csv.

**Labels used for G.** 3D amodal sweep labels (per_sweep_annotations_amodal, LiDAR 10 Hz): every camera frame takes
the sweep closest in time if within 50 ms (all but 56 frames: **15,006 frames** with geometry), motion-compensated
from LiDAR to camera time; vehicle / pedestrian / two-wheeler classes; braking consumer, monocular lift with dt = 1/30 s,
cheap action history at 30 Hz. Interval: sequence bootstrap (24 sequences), 2,000.
**AP.** COCO AP on three road-user groups (vehicle = car / bus / truck; person; two-wheeler = bicycle / motorcycle; the
traffic-light and stop-sign classes of Argoverse-HD are excluded because our pipeline keeps road users only), detections
with conf ≥ 0.1; ours' detections = cheap boxes outside its crops ∪ de-duplicated crop boxes.

## Results (ahd_summary.csv)
| method | AP | AP50 | AP_small | G | network px | ours + ref − it |
|---|---|---|---|---|---|---|
| cheap 320 (224×320) | 0.115 | 0.196 | 0.002 | 0 | 0.031 | |
| u640 | 0.219 | 0.345 | 0.024 | −0.034 | 0.116 | [+0.073, +0.181] |
| u960 | 0.286 | 0.447 | 0.080 | −0.032 | 0.253 | [+0.059, +0.192] |
| u1280 | 0.328 | 0.506 | 0.134 | −0.067 | 0.444 | [+0.061, +0.265] |
| u1600 | 0.352 | 0.548 | 0.175 | −0.081 | 0.711 | [+0.075, +0.282] |
| native u1920 | 0.368 | 0.573 | 0.210 | −0.092 | 1.013 | [+0.077, +0.304] |
| ours (no refiner) | 0.163 | 0.291 | 0.063 | 0.039 | 0.065 | [+0.021, +0.084] |
| **ours + refiner v2** | (same boxes) | | | **0.091** | 0.065 | — |

**Reading.** On a fifth dataset, untouched by any tuning, every whole-frame resolution makes braking decisions WORSE
than the cheap draft (G −0.03 to −0.09: more detections at 30 Hz → more phantom / flickering brakes), while ours
improves them (+0.091 with refiner) and beats every resolution significantly at 6% of the native pixels.
**Against the method:** standard detection AP is much lower for ours (0.163 vs 0.286 for u960 and 0.368 native; ours is
between cheap and u640) — by design it verifies only decision-relevant regions, so whole-image AP is not its target.

## Streaming AP (sAP), added 2026-10-04 (ahd_sap.csv; columns sAP… in ahd_summary.csv)

**How it was measured**
- **Latency:** per-frame latency of every pipeline on all 15,062 Argoverse-HD frames, measured on the Jetson AGX Xavier.
  - Settings: TensorRT FP16, clocks locked, the same fast path and stage timing as K5.
  - Command: `scripts/py scripts/56_ahd_latency.py` (md5 81fe35d9…); engines from `engines/build_ahd.sh`. A warm-up pass over 200 frames was discarded.
  - Output: `results/raw/20261004_055244_ahd_latency/` → `ahd_latency_jetson.csv`.
- **Parity check on the Jetson:** G with TensorRT detections is 0.084 for ours + refiner (VUW 0.091) and −0.001 to −0.080 for the whole-frame passes. Crops per frame: 2.33.
- **Simulation:** `python scripts/A_method/ahd_sap.py results/argoverse_hd/ahd_latency_per_frame.csv` on VUW (md5 704708c2…), following Li et al., ECCV 2020.
  - One sequential processor with a greedy policy: when idle, it takes the newest arrived frame.
  - The offline detections from the K7 run are used, with the Jetson latencies.
  - At each annotated frame, the newest finished output is scored with COCO AP on the same 3 groups.

| Pipeline | Jetson mean ms | Frames processed | Mean staleness ms | sAP | sAP50 | offline AP |
|---|---|---|---|---|---|---|
| cheap 320 | 9.8 | 99.8% | 33 | 0.096 | 0.185 | 0.115 |
| u640 | 14.7 | 99.8% | 33 | 0.174 | 0.321 | 0.219 |
| u960 | 21.4 | 99.8% | 33 | 0.221 | 0.413 | 0.286 |
| **u1280** | 29.4 | 99.6% | 34 | **0.247** | 0.465 | 0.328 |
| u1600 | 42.5 | 78.6% | 81 | 0.172 | 0.382 | 0.352 |
| native u1920 | 55.9 | 59.7% | 100 | 0.151 | 0.351 | 0.368 |
| ours, no-refiner timing | 28.4 (p95 58.8) | 92.0% | 53 | 0.110 | 0.245 | 0.163 |
| **ours + refiner v2** | 31.5 (p95 65.2) | 88.1% | 60 | **0.105** | 0.238 | 0.163 |

**Reading (against the method)**
- **Ours has the lowest sAP of all pipelines except cheap.** It reaches 0.105 vs u960 0.221 and u1280 0.247.
- **Two reasons:**
  - Its offline whole-image AP is already low by design.
  - Its long latency tail (p95 65 ms) makes it skip 12% of the frames, so its staleness doubles.
- **The streaming optimum is u1280**, the largest whole-frame pass that still fits in the 33 ms frame period. u1600 and native lose frames.
- **G and sAP disagree:** ours has the best braking-decision value and the worst streaming detection AP. The paper must not claim streaming-detection gains.
- **No CI for sAP:** COCO AP is not additive over sequences, and pycocotools' accumulation cannot take an image set resampled with replacement, so a sequence bootstrap was not run. The gaps are large (ours − u960 = −0.116).
- **Caveat on mixing sources:** detections come from VUW (torch FP16) while latencies come from the Jetson (TensorRT). This is the standard offline-detections-plus-runtime simulation.
Caveat: G at 30 Hz uses a 1/30 s ttc baseline (noisier looming than at 5 Hz on AV2).
