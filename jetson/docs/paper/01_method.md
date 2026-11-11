# Method (final)

## Idea in one paragraph
A cheap detector (YOLOv8s at long side 320) runs on every frame and produces a draft scene. Instead of escalating whole
frames to a higher resolution, we ask the *planner* where its decision could change: (i) the **decision-sensitivity
field** — insert an unseen probe object at every cell of a ground grid and mark the cells where the planner's action
would change (closed form for max-composable consumers such as braking; probing the real planner for PDM-Closed);
(ii) the **support** — the draft objects the current action rests on. Only those image regions are re-detected at high
resolution ("verification crops"), merged back (crop detections replace draft boxes whose centre lies in a crop), and
the loop iterates (≤ 3 rounds) until no new crop is needed. A truncation guard re-detects objects cut by a crop edge in
a grown crop (restores locality); the lift's temporal history comes from the previous merged scene; an optional
ResNet-18 range refiner corrects the range of support objects detected in crops.

Theory (docs/paper_theory.md): for consumers whose action is a quantised max of per-object demands (braking, IDM-style
following, TTC alarms), under locality and exact sensitivity the loop returns exactly the decision the full detector
would give (Proposition, §3); field-only, support-only and pivot-only variants each fail on explicit counterexamples
(§5); termination is guaranteed. Argmin planners (Planner B, PDM-Closed) are outside the class — evaluated empirically.

Empirical source of the gain (state plainly; 03_results §7): mostly removing phantom obstacles in frames without a
relevant object, and correcting the range of already-visible relevant objects; newly entering hazards are rare.

## Components and code
| component | definition | code |
|---|---|---|
| draft | YOLOv8s (COCO), long side 320, conf ≥ 0.25, DEEP's ego-frame monocular lift | rap.detect, rap.mono |
| braking field | grid r ∈ {12.5,…,70} m × y ∈ {−6,…,6} m, probe 1.8×1.6 m, closing at v_close | src/vtf/dsf.py, speculative.field_crop |
| support | objects whose demand alone reaches the current action's threshold; crop = 2× box | speculative.support / box_crop |
| loop | crops = field footprint ∪ support boxes, merge, iterate ≤ 3 rounds | src/vtf/speculative.py run() |
| guard | detections truncated by an interior crop edge re-detected in a grown crop (≤ 2 growths) | speculative.guarded / guarded_batch |
| refiner | ResNet-18 on a 64×64 crop + box features → log range correction (v1: KITTI+nuScenes; v2: + all nuScenes) | scripts/common/train_refiner_remote.py |
| PDM-Closed field (NAVSIM) | 7×7 probe grid (x 8–55 m, y ±10.5 m), cell sensitive if the 4 s plan moves > 0.5 m; support = objects whose removal moves it | remote/A_navsim/navsim_field.py, navsim_detect.spec_pdm_dets |
| general consumers | Planner B field + pivots; union of consumers | src/vtf/spec_general.py |
| deployment fast path | GPU letterbox, all crops of a round in one batched TensorRT call | src/vtf/gpuprep.py, speculative.run(batch_round=True) |

## Configuration per benchmark (each chosen on that benchmark's dev split; docs/method_freeze_draft.md)
| benchmark | configuration | chosen on |
|---|---|---|
| nuScenes | **v3**: v_close 10 m/s, ≤ 3 rounds, crops at 0.6 × native, guard, scene history, refiner v1 | frozen before the confirmatory run (pre-registered); the new-val-tuned v4 did not transfer (reported as variant) |
| AV2 | **v4-lite**: v_close 0, ≤ 3 rounds, crops at 960 / long side (0.469), guard, refiner v2 | AV2 train (confirmatory 3) |
| NAVSIM | **spec_pdm**: probing PDM-Closed itself (field + support from the planner), crops at 0.5, one round, v3's guard/merge | OpenScene mini (dev) |
| KITTI | union of braking field (v_close 0) and Planner B field, ≤ 3 rounds, 96×288 crop canvas, refiner (round-1 dev choice) | DEEP KITTI train+val |
| Argoverse-HD, Bench2Drive | AV2 config / nuScenes v3 config unchanged (no tuning) | — |
