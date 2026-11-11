# Speculative perception v1 — method freeze and confirmatory pre-registration (DRAFT)

*Drafted 2026-09-27 08:10. Becomes binding when the refiner (§2.4) is frozen and this header is changed to FINAL with a
date. Nothing below may be changed after that without a dated amendment.*

## 1. Why a freeze

About sixty exploratory analyses (directions_log D1–D76) used the KITTI and nuScenes test splits of DEEP. Every number
there is development evidence. The paper's headline numbers must come from one run of a frozen method on data that
the exploration did not touch.

## 2. The frozen method

1. **Draft.** YOLOv8s at 96×320 (DEEP's cheap mode), threshold 0.25, DEEP's ego-frame monocular lift.
2. **Decision-sensitivity field.** Closed form for max-composable consumers (`vtf/dsf.py`, verified equal to probing
   on all 8,008 KITTI frames): ground grid r ∈ {12.5, 17.5, 22.5, 27.5, 35, 45, 55, 70} m, y ∈ {−6, −4.5, …, 6} m;
   probe object 1.8 m × 1.6 m, static; discovery crop = bounding box of the projected sensitive cells.
   Non-max-composable consumers: vectorised probing of the same grid.
3. **Pivots.** Objects whose removal changes the draft decision (closed form for max-composable consumers); crop =
   2× the pivot box.
4. **Refiner.** One image-level range refiner (crop + box features), trained on training units only, epoch chosen on
   validation units; its architecture and recipe are fixed by B1 before this document becomes FINAL.
5. **Verification detector.** YOLOv8s on each crop at the 960-px scale (0.773 × KITTI width; nuScenes 0.6), threshold
   0.25; detections whose centre lies in a crop replace the draft's there.
6. **One round** (no second speculative round), no pre-escalation gate (every frame runs the loop; the D62b gate is a
   secondary analysis).

## 3. Confirmatory data (not used in the exploration)

- **nuScenes**: scenes outside the 85-scene blob used so far (DEEP's scene split rule applied to the new scenes), CAM_FRONT,
  same consumers; requires downloading further trainval blobs.
- **nuPlan** with PDM-Closed and IDM, front camera, real detections (DEEP's Task 5 pipeline), logs not in DEEP's split,
  plus the benchmark's own planning scores.
- KITTI has no unused labelled sequences; KITTI results stay development evidence.

## 4. Primary endpoints

- **E1** decision value (share of the all-cheap loss) of speculative v1 vs uniform escalation to 640 and to 960, paired
  unit bootstrap (2,000 draws), per cell.
- **E2** end-to-end latency and energy per frame on the Jetson (TensorRT FP16, same process, image decode excluded)
  vs uniform 640, uniform 960 and DEEP's cascade.
- **E3** on nuPlan: the benchmark's planning score of PDM-Closed fed by speculative v1 vs uniform 640/960 perception.

Secondary: whole-image recall/precision (reported as the trade-off), decision agreement with the full mode, harmed
share of affected inputs, crops per frame.

## 5. Decision rule

- **Claim "matches uniform escalation"** in a cell if the E1 interval of (speculative − uniform 960) lies above −0.02 of
  the all-cheap loss; **"beats"** if it lies above 0.
- **Claim "cheaper"** only if E2's median and p95 are below uniform 960's.
- The paper is written around E1–E3 as they come out, including cells where the method loses.
