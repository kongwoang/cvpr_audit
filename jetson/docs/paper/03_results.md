# Final results (authoritative numbers)

Canonical per-row CSVs with px mean/p95 and harmed share: docs/paper/handoff/K1/table_main_{nusc,av2}.csv (harmed share =
harmed / affected frames everywhere). Conventions: G = decision value (share of the all-cheap loss; braking consumer, monocular lift unless stated); CI =
95% paired bootstrap (2,000) of **ours − baseline** unless stated; px = network pixels as a share of the native frame.
Same COCO YOLOv8s everywhere unless "fine-tuned". "VUW:" paths are under /vol/dl-nguyenb5-solar/users/hoangpc/CVPR;
other paths are in this repo. Section names in brackets point to docs/directions_log.md for the full tables.

## 1. Headline: decision value at a fraction of the pixels (large-frame datasets)

### AV2 val — 150 logs, 11,818 frames (test split; config chosen on AV2 train)
| method | G | px | ours − it [95% CI] |
|---|---|---|---|
| **ours (v4-lite + refiner v2)** | **0.561** | 0.068 | — |
| v3 + refiner (frozen nuScenes config) | 0.545 | 0.086 | [+0.005, +0.027] |
| uniform 2048 (native) | 0.558 | 1.012 | [−0.042, +0.050] |
| uniform 1536 | 0.540 | 0.573 | [−0.015, +0.064] |
| uniform 1280 | 0.484 | 0.400 | [+0.050, +0.109] |
| uniform 960 | 0.446 | 0.232 | [+0.087, +0.147] |
| uniform 640 | 0.144 | 0.103 | [+0.367, +0.472] |
| uncertainty crops (same per-frame pixels as v3) | 0.494 | | [+0.037, +0.098] |
| AdaScale (fit on AV2 train) | 0.317 | 0.170 | [+0.209, +0.281] |
| Budget-Adaptive Routing, Geng's adaptive, at our pixel budget | 0.077 | ≈0.068 | [+0.439, +0.531] |
| best router (cond_V, decision-value target; our variant) at our budget | 0.230 | ≈0.068 | [+0.288, +0.371] |
| **oracle frame router** at our budget | 0.371 | ≈0.068 | [+0.149, +0.232] |
| FOVEA KDE_I (frozen) | 0.328 | | [+0.193, +0.276] |
| temporal-criticality crops | 0.174 | | [+0.349, +0.426] |
| SAHI (640-px slices; 320-px slices −0.140) | 0.088 | | [+0.418, +0.531] |
Sources: VUW results/av2/eval_summary__val_v4_final.csv, results/av2/baselines_summary__val.csv,
results/native/summary_av2.csv, results/av2_sota/summary_av2val.csv. Harmed share: ours 0.087 (lowest), u960 0.155.
Message: ties the best whole-frame resolution (native) with ~1/15 of its pixels; beats every SOTA baseline; even a
perfect frame-level router at the same pixels reaches only 0.371.

### nuScenes new test — 228 scenes, 9,181 keyframes (pre-registered confirmatory run of frozen v3)
| method | G | px | v3 + ref − it [95% CI] |
|---|---|---|---|
| **v3 + refiner** | **0.171** | 0.088 | — |
| v3 | 0.135 | 0.088 | [+0.004, +0.078] |
| uniform 1280 | 0.201 | 0.654 | [−0.141, +0.094] |
| uniform 1600 (native) | 0.121 | 1.031 | [−0.048, +0.199] |
| uniform 960 | 0.010 | 0.384 | [−0.002, +0.346] |
| uniform 640 (DEEP's full mode) | −0.181 | 0.171 | [+0.062, +0.741] |
| AdaScale (re-impl., fit on new train) | 0.083 | 0.173 | [+0.028, +0.153] |
| FOVEA KDE_I 576×960 / KDE_S | 0.074 / 0.022 | 0.384 | [+0.021, +0.191] / [+0.036, +0.255] |
| Budget-Adaptive Routing, official estimators at our budget (skip_offbin seeds 42/1/2; cond_moric = Geng's arbiter choice) | −0.010 … 0.079 | ≈0.08–0.09 | [+0.019, +0.309] |
| router with decision-value target (cond_V; skip_V seeds 42/1/2) | 0.028; −0.016 / 0.026 / 0.066 | ≈0.09–0.10 | [+0.017, +0.310] |
| **oracle frame router** at our budget | 0.342 | 0.088 | [−0.250, −0.086] (oracle above ours) |
| SAHI (640 slices) | 0.130 | ≫ | (point estimate below) |
| temporal-criticality crops (strict budget) | 0.114 | ≤ v3 | |
| uncertainty crops (strict budget) | −0.092 | ≤ v3 | |
| LZU frozen / UFP frozen | −0.861 / −0.678 | | |
Pre-registered rule (method_freeze §5): v3+ref "beats" u640 and "matches" u960 (lower bound −0.002 > −0.02); harmed
share 0.39 vs 0.47 (u960). Sensitivity (not pre-registered): one scene (scene-1051) drives large full-model losses;
without it u960 0.166, v3+ref 0.209, v3+ref − u960 [−0.011, +0.102]. Sources: results/final/confirm_nuscenes_newtest_v3.csv,
results/final/baseline_*newtest.csv, VUW results/native/summary_nusc.csv, VUW results/router/summary_newtest*.csv.
[CONFIRMATORY nuScenes run; SOTA baselines — results; Native-resolution uniform baselines]

Refiner provenance (K1 C): v1 = KITTI + nuScenes DEEP-85 train/val; v2 = KITTI + all nuScenes train/val scenes (incl.
151 new-val scenes for epoch selection); neither saw nuScenes new test or AV2.

## 2. Fine-tuned detectors (each trained on the benchmark's train split)
nuScenes new test (two seeds where given):
| method | G (seed 0 / 1) | notes |
|---|---|---|
| **ours (v3, fine-tuned verifier)** | **0.297 / 0.295** | − u960-ft s0 [+0.015, +0.109], s1 [−0.032, +0.082] |
| ZoomDet (official, 736²) | 0.259 / 0.235 | ours − it s0 [−0.007, +0.092], s1 [+0.014, +0.127] |
| u960-ft | 0.241 / 0.272 | |
| FOVEA-ft | 0.235 / 0.244 | |
| ESOD (official, YOLOv5s 1536) | 0.221 / 0.229 | ours − it [−0.007, +0.178] |
| LZU-ft | 0.219 / 0.184 | identical-recipe uniform control 0.236 / 0.227 |
| UFP-ft (MP-Det-style) | 0.202 / 0.180 | ours − it s1 [+0.038, +0.187] |
AV2 val (trained on AV2 train, one seed): **ours-ft + ref 0.546** · ours-ft 0.527 · ZoomDet 0.437 · u960-ft 0.402 ·
ESOD 0.295; ours-ft + ref − ZoomDet [+0.083, +0.139], − u960-ft [+0.116, +0.176], − ESOD [+0.173, +0.333].
Note: on AV2, fine-tuning lowers G for the whole-frame detector (u960-ft 0.402 < COCO 0.446) and slightly for ours.
Sources: VUW results/fovea/summary_ft_newtest*.csv, results/zoomdet/, results/esod/, results/av2_ft/summary.csv.

## 3. Standard planning benchmarks
NAVSIM v1.1 navtest (12,146 scenes; PDM-Closed fed by CAM_F0 perception; config chosen on OpenScene mini):
reference (GT objects) 0.9000 · **ours (spec_pdm) 0.8945** · u960 0.8949 · u640 0.8944 · u800 0.8951 · u480 0.8943 ·
cheap 0.8930 · SAHI 0.8947 · FOVEA 0.8945 · AdaScale 0.8950 · routers at our budget 0.8937–0.8948 · random router 0.8938 ·
oracle router 0.9028 · braking-field v3 0.8926. ours − every non-oracle baseline: CI contains 0 (e.g. − u960
[−0.0018, +0.0013]); ours − v3 [+0.0002, +0.0040]. → a tie (NAVSIM's PDMS barely reacts to front-camera perception
quality; only an oracle separates). Sources: VUW results/navsim_exp/pdms_summary_final_vs_spec_pdm.csv,
results/navsim_exp/sota/summary_navtest.csv. Our reference PDM-Closed 0.900 vs published 0.881 (setup difference, note).
nuPlan-mini (1,440 states, PDM-Closed; same logs as NAVSIM dev → dev evidence only): v3 0.9401–0.9404 best; FOVEA
0.9369, u960 0.9368, full 0.9346, cheap 0.9333, SAHI 0.926; v3 − full [+0.0001, +0.0120].

## 4. Detection metrics (nuScenes new test, 2,356 frames, conf ≥ 0.25)
AP50 / relevant recall / relevant precision / phantom-brake detections per 1k frames:
cheap .155 / .728 / .762 / 30.6 · u960 .284 / .824 / .736 / 39.5 · **v3 + ref .191 / .793 / .879 / 12.7** ·
v4-lite + ref .176 / .786 / .889 / 11.5 · SAHI ≈50 phantom-brake detections / 1k. → lower whole-image AP by design,
3× fewer phantom-brake detections and the highest precision on decision-relevant objects.

## 5. Deployment latency (Jetson AGX Xavier, TensorRT FP16, clocks locked; KITTI test 3,536 frames, 1242×375)
G / mean / median / p95 ms: cheap 0 / 6.1 / 6.3 / 7.0 · u640 0.125 / 8.3 · u960 0.127 / 12.6 / 12.6 / 13.5 ·
u1248 (native) 0.288 / 15.5 / 15.5 / 16.5 · v4-lite + ref 0.226 / 20.0 / 18.0 / 46.2 · v3 + ref 0.234 / 26.7 / 26.7 / 50.5
(earlier CPU path 33.8 ms). Large-frame (AV2 / nuScenes) latency: see §8 (K5) — ours is NOT faster there. Source:
results/raw/*_fast_latency (latest run with u1248). [Jetson latency engineering]

## 6. KITTI (small frames) — DEEP's four cells, test (6 sequences)
mean of 4 cells (brake mono / brake oracle / Planner B mono / Planner B oracle):
u1248 + our refiner 0.449 (.315/.470/.440/.570) · round-1 dev choice (union field + ref) 0.401 (.229/.371/.432/.570) ·
round-2 dev choice (native crops) 0.395 · v3 + ref 0.389 · u1248 0.323 (.284/.460/−.022/.571) · SAHI 0.327 · u960 0.231.
Intervals vs u1248 (mean4): u1248+ref [+0.037, +0.164], round 1 [+0.027, +0.107], round 2 [+0.003, +0.108], SAHI
[−0.103, +0.083] (handoff/K1/kitti_cells.csv). Dev (train+val, 15 seq) ranks ours first (0.798 vs u1248 0.654); dev and test disagree. Refiner lifts every pipeline on
Planner B mono (≈−0.02 → ≈0.44). Sources: results/raw/*_kitti_final, *_kitti_final2, *_kitti_dev_grid, *_kitti_dev_round2.

## 7. Analyses for the discussion
- Where the gain comes from (AV2 per-frame split vs temporal-crit / uncertainty / random): frames without a relevant
  object (phantoms removed) and frames with visible relevant objects (range); newly entering objects: 33 AV2 / 45
  nuScenes frames, no measurable gain. [AV2 val baselines DONE]
- Budget curve (nuScenes test): iterating matters (1 round G 0.016 vs 3 rounds 0.135 at equal pixels); crop scale 0.4
  too coarse (−0.056); v4-lite matches v3 with 42% fewer crop pixels. [v3 operating points]
- Frame routing vs region verification: at our pixels the oracle frame router reaches 0.371 vs ours 0.561 on AV2, but 0.342 vs ours 0.171 on nuScenes (an oracle choice of frames beats ours there; learned routers reach ≤ 0.079).
- Refiner generality: improves u1248 on KITTI dev (brake mono [+0.003, +0.038]) and every pipeline on Planner B mono.
- Heuristic rankings flip across datasets (temporal-crit best on nuScenes, collapses on AV2; uncertainty the reverse);
  ours is the only policy on top of both.

## 8. Results of the writer-handoff tasks (2026-10-04; details and CSVs in docs/paper/handoff/<TASK>/REPORT.md)
- **Argoverse-HD val (K7; AV2 configuration unchanged, first and only run; 24 sequences, 15,006 frames with 3D labels)**:
  G ours + refiner **0.091** (no refiner 0.039) at network px 0.065; every whole-frame resolution is NEGATIVE: u640
  −0.034, u960 −0.032, u1280 −0.067, u1600 −0.081, native u1920 −0.092; ours + ref − each: all intervals > 0 (e.g. − u960
  [+0.059, +0.192], − native [+0.077, +0.304]). AP (3 road-user groups): ours 0.163 vs u960 0.286, native 0.368 (lower by
  design). Streaming AP (Jetson per-frame latencies, Li et al. simulation): ours 0.105 (31.5 ms, 12% frames skipped) vs u960
  0.221, u1280 0.247 (best), native 0.151 — ours is NOT a streaming-detection method. Source:
  results/vuw_summaries/argoverse_hd/summary_{ap,G,sap}.csv, docs/paper/handoff/K7/.
- **Bench2Drive closed loop (K6; three runs, 219 routes; DS mean ± std: privileged 96.63 ± 0.17, cheap 92.27 ± 0.92, u960
  93.15 ± 0.74, u1600 91.95 ± 0.71, ours 92.80 ± 0.63; ours − each perception branch: all CIs include 0; − privileged
  −3.83 [−6.24, −1.56]). First run (single-run details)**: DS
  privileged 96.00, cheap 92.74, u960 92.63, u1600 91.86, ours 92.75 (SR 0.909 / 0.850 / 0.850 / 0.841 / 0.836); ours −
  each non-privileged branch: all CIs include 0 (e.g. − u960 DS +0.12 [−1.88, +2.15]); − privileged −3.26 [−5.92, −0.84].
  No closed-loop gain; ours hides as many in-camera actors as cheap (1.19 vs 1.16 / frame; u960 0.91). Shared failures:
  AccidentTwoWays / ConstructionObstacleTwoWays fail for every perception branch. Source:
  results/vuw_summaries/b2d/b2d_summary.csv, docs/paper/handoff/K6/.
- **Large-frame Jetson latency (K5; post-hoc engineering measurement)**: with the evaluations' crop geometry (ceil32
  canvases via a dynamic-shape engine; GPU-side candidate selection, bit-identical) — nuScenes (85 old DEEP scenes, 3,176
  timed frames): ours + ref 30.2 ms (p95 63.8, p99 82.3), G 0.173, px 0.096; u640 13.4 ms / 0.157; u960 19.2 / 0.152;
  native 35.2 / 0.175. AV2 (10 val logs, 586 frames): ours + ref 53.0 ms (p99 142), G 0.545, px 0.083; u960 24.7 / 0.415
  (ours − u960 [+0.016, +0.280]); u1280 35.0 / 0.506; u1536 44.3 / 0.543; native 67.8 / 0.553. Ours is on the latency–G
  front (no faster pass has a higher G) but equal-G passes (CIs include 0) are faster; long tail. First run with static
  engine canvases (ours 28.1 ms / 0.150 and 49.2 / 0.448) under-stated ours' G. Source: docs/paper/handoff/K5/
  (latency_*_v8s_dyn.csv; results/raw/20261004_143529_…_opt_nusc, 20261004_144842_…_opt_av2).
- **K11 decision-identical fast implementation (Jetson, parity 100% of timed frames)**: without the per-stage
  instrumentation syncs ours + ref runs 25.3 ms (p95 53.2, p99 69.2) on nuScenes (u960 17.8, u1280 23.9, native 33.4 ms)
  and 42.0 ms (p99 117) on AV2 (u1536 41.1, native 64.9 ms) — use THESE latencies (the K5 numbers include breakdown
  syncs). Dev-selected (nuScenes old scenes) config ctx 2.5 + field margin 0.15: 24.7 ms; on nuScenes new test (post-hoc) G 0.195 vs frozen 0.170, + 0.025 [−0.028, +0.077] n.s. at px 0.125 vs 0.100 — keep frozen.
  Source: docs/paper/handoff/K11/.
- **AV2 ego-hood artefact (K10, post-hoc robustness check)**: COCO-YOLOv8s detects the AV2 ego hood (rows ≥ 1830);
  the cheap draft's action is hood-driven on 50.9% of AV2 val frames = 81% of its decision loss (u960 11.7% / 31%, native
  0.3%, ours 5.6% / 21%). With the hood masked for EVERY pipeline: all-cheap loss 23,174 → 9,294; ours final G 0.561 →
  **0.122** (still above every realizable row, e.g. − u960 (−0.021) [+0.088, +0.203], − native (−0.058) [+0.092, +0.276],
  − uncertainty crops (0.057) [+0.007, +0.124]); every whole-frame resolution becomes negative (as on Argoverse-HD); the
  oracle frame router (0.315) is now ABOVE ours. Argoverse-HD and nuScenes: no ego part, tables unchanged. REPORT THE
  MASKED AV2 NUMBERS. Source: docs/paper/handoff/K10/.
- **K12 (post-hoc)**: AV2 locality decomposition (50 val logs): real crops 0.802 → exact locality 0.923 → + F's ttc 0.945
  (v_close 10: 0.812 → 0.963 → 0.975). Instrumented detector px at ours' crop AREA: heuristics spend 30–85% more pixels
  (nuScenes ours 0.100 vs uncertainty 0.133 / temporal_crit 0.129 / random 0.185 / traj_support 0.140; AV2 ours 0.074 vs
  0.109 / 0.111 / 0.131); Argoverse-HD ours 0.070. Routers re-matched to tensor px: conclusions unchanged (nuScenes oracle
  0.345 > ours 0.171; learned routers ≤ 0.044 / AV2 ≤ 0.252, all below ours). Fine-tuned verifier + refiner v1:
  0.330 / 0.332 (seeds 0 / 1) vs v3-ft 0.297 / 0.295 and u960-ft 0.241 / 0.272. Source: docs/paper/handoff/K12/.
- **K13 (post-hoc, ego mask): every pre-K10 AV2 analysis redone with the mask.** Ablation pattern holds (field alone
  −0.035; support only 0.122 = full; guard, merged history needed; pivots worse), but the refiner now carries most of AV2 G
  (0.031 → 0.122). Agreement with π(u960) 0.861 / 0.842 (ours still highest, margin over best heuristic 2 pts, was 7).
  Locality 0.854 → 0.932 → 0.952. Natural-miss recovery 0.838, del_all 0.870 (field_only 0.754, was 0.240). K9 TTC
  equal-px +0.006 [−0.048, +0.061] (unchanged). Jetson masked: ours 37.8 ms / G 0.173 = best G, faster than u1536 (42.1 /
  0.147) and native (66.0 / 0.125), CIs include 0. Tensor px masked ours 0.053 (K10's 0.049 is area-based); heuristics at
  ours' crop area use 52–72% more px; WITHOUT refiner ours is not above them (− uncertainty [−0.108, +0.002]).
  FINE-TUNED DETECTORS: none detects the hood; masked G all negative (ours-ft + ref −0.100, u960-ft −0.473, ZoomDet −0.384,
  ESOD −0.736) — their unmasked G was the hood; COCO ours 0.122 is significantly above ours-ft + ref [+0.157, +0.294].
  Argoverse-HD ablation (refiner v2): full 0.091; field only −0.066; support only 0.127 (n.s.); one round 0.106
  (slightly better); no guard 0.043 (worse). (g) masked AV2-train dev check (700 logs): v4 + ref v2 WINS (0.134 vs reported v4-lite + ref v2 0.115, [+0.006, +0.032];
  unmasked v4-lite had won); as a variant on masked AV2 val (same single K10 pass): 0.150 vs 0.122, +0.028 [+0.007, +0.048]
  at 1.4x crop px (0.067 vs 0.049). Keep v4-lite frozen; report v4 as the mask-selected variant. Source: docs/paper/handoff/K13/.
- **K15 (post-hoc): Figure 1 on real frames.** Four frames from the natural-miss sets: nuScenes 0273/6 and 0961/8, and masked AV2 87ca/17 and c85a/17. On each, the draft and the uncertainty crops at ours' crop area both KEEP while DECELERATE is required. Ours recovers the missed object (vehicle at 22 / 15 / 52 m, cyclist at 38 m) and decelerates at 0.056–0.122 network px. Whole-frame native decelerates on three of the four frames, at 1.01–1.03 px. On nuScenes 0961/8 whole frame (u960 and native) also KEEPs. Panels: docs/paper/handoff/K15/.
- **K16 (post-hoc): attribution of ours' disagreement with π(F)** (ours without refiner; full test splits; AV2 masked). The counterfactual chain lifts agreement 0.951 → 0.996 on nuScenes and 0.861 → 0.993 on AV2. Share of the original disagreement by cause:
  - nuScenes: locality 81%, history 4%, closing probe 0%, grid 3%, probe size 5%, round cap 0%, residual 7%.
  - AV2: locality 62%, history 11%, closing probe 21% (all of it at 10 m/s), grid 1%, probe size 1%, round cap 0%, residual 4%.

  In the residual, F is wrong (it matches the annotated scene on 3/31 and 3/64 frames) and ours is right on 27/31 and 51/64. Typical case: F hard-brakes on an object at 5–8 m, below the grid. Rounds 0/1/2/3:
  - nuScenes: 20.6/75.4/3.9/0.1%;
  - AV2: 33.7/55.6/10.2/0.5%;
  - at most 0.04% of frames stop at the cap without converging.

  Coverage of F's raising objects by R: 0.86 nuScenes and 0.58 AV2; 70–80 m objects are inside the footprint 91% / 77% of the time. Source: docs/paper/handoff/K16/.
- **K14 (post-hoc): "+ refiner" controls.** The same refiner is applied to every pipeline's support objects from high-resolution passes. Ours final is NOT significantly above every "+ refiner" control on any dataset:
  - nuScenes: highest G (0.170); significant only vs random + ref and support only + ref; native + ref 0.167 at 10x px.
  - AV2 masked: 0.122, tied with support only + ref 0.122, uncertainty + ref 0.118 and corridor + ref 0.113; above u960 + ref, random and field only.
  - Argoverse-HD: the ego-corridor tube + ref is significantly ABOVE ours (0.122 vs 0.091, [-0.066, -0.001]).

  The area-matched heuristics use 24–80% more tensor px. Table-2 cells (AV2 masked, heuristics without refiner), ours final minus: uncertainty [-0.016, +0.092], temporal_crit [+0.019, +0.120], random [+0.066, +0.175]. Source: docs/paper/handoff/K14/.
- **K19 (post-hoc): safety quantities.** Ours' decision value comes from fewer unnecessary decelerations / hard brakes and fewer delayed hard brakes. Ours misses MORE decelerations than u960 on all three datasets:
  - nuScenes: +3.3 to +8.5 per 1,000 frames;
  - AV2: +0.5 to +4.0;
  - Argoverse-HD: +0.9 to +4.7.

  Collision term is rare: nuScenes 8 frames / 3 scenes, AV2 5 / 3, Argoverse-HD 0. Per scene / log ours beats u960 in 82 vs 52 nuScenes scenes (p 0.012) and 97 vs 45 AV2 logs (p < 0.001). On multi-obstacle frames: G 0.103 on AV2, ~0 on nuScenes and Argoverse-HD. Source: docs/paper/handoff/K19/.
- **K20 (post-hoc): cost of planner probing (NAVSIM).** The frozen spec_pdm makes 50 PDM-Closed calls per frame (45 probe, 4 support), 14.9 s per frame on one CPU thread.
  - An identical-output implementation (planner state built once per frame, ego roll-outs reused) takes 6.2 s, and 1.36 s on 8 workers (11x). Poses are bit-identical.
  - The 5x5 grid, the 3x3 grid and probing only the plan's corridor cost no measurable PDMS (all within ±0.0013 of 0.8945) at 14–26 calls.
  - Interacting pairs are 0.06% of pairs (6 / 1,000 frames); adding them to the support changes no PDMS.
  - Bench2Drive perception makes no planner query. Source: docs/paper/handoff/K20/.
- **K17 (post-hoc): mechanism of the field** (full vs support only, with refiner). The field changes the action on 3.7% / 7.7% / 12.3% of frames (nuScenes / AV2 masked / Argoverse-HD). Loss on those frames, full vs support only:
  - nuScenes: 571 vs 1,937 (helps);
  - AV2: 1,370 vs 1,374 (wash);
  - Argoverse-HD: 3,498 vs 2,907 (hurts).

  True discoveries always help (e.g. A-HD 588 vs 1,898). Extra detections in field crops add false alarms (unnecessary decel. / hard brakes rise on all datasets) that cancel the gain on AV2 / A-HD. Where braking is required but the draft holds no supporting object (3–6% of frames), the field is decisive: G 0.35 / 0.33 / 0.55 vs 0.01 / 0.12 / 0.01. At equal tensor px (larger support crops or random extra crops) full is not significantly better. A faster probe (v_close 5–10) raises recovery (0.87 → 0.94 AV2) but lowers G. Source: docs/paper/handoff/K17/.
- **K18(A) (post-hoc): decision value vs tensor px.** With heuristic crops matched on per-frame TENSOR px (realised 0.96–1.18× ours), ours final is above or tied with every heuristic + refiner:
  - nuScenes: above temporal_crit, random and corridor; uncertainty and traj_support n.s.
  - AV2 masked: above traj_support, corridor and random; uncertainty and temporal_crit n.s.

  On AV2 the heuristics reach ours' G only at ≥ 1.5× its px. Frame routers (best BAR, decision-value) stay at G ≈ 0 at 0.5–4× ours' px; the oracle router is above. AV2 ego-corridor at ours' crop area: G 0.029 / 0.113 (± refiner), px 0.066. (B) Jetson, NEW variant with at most 2 detector calls after the draft:
  - AV2 masked: p95/p99/max 58/77/94 ms vs frozen ours 82/110/134 and native 66/88/128, at G 0.167 vs 0.173 (98.6% same action);
  - nuScenes: 40/49/97 ms vs frozen ours 53/69/125 and native 34/35/36.

  Source: docs/paper/handoff/K18/.
- **K21 (post-hoc): robustness.**
  - **(1) Probe templates × v_close.** Dev would pick truck@0 (nuScenes) / ped@0 (AV2), but both are below the frozen car probe on test (0.156 vs 0.170; 0.115 vs 0.122). Faster probes trade G for recovery (AV2 0.87 → 0.94). The frozen probe stays.
  - **(2) YOLO11s as draft and target, no re-tuning:**
    - nuScenes: ours + refiner 0.388 ties u960 0.368 / native 0.362 at 0.10 px;
    - AV2: ours + refiner 0.139 is above u960 (−0.041) and native (−0.121);
    - the existing refiner transfers on AV2 (+0.079 [+0.040, +0.121]).
  - **(3) Ego mask.** Eight of nine settings give ours 0.121–0.124. Only (1880, 0.7) lets the hood through, and ours' actions still change on 0.01% of frames.

  Source: docs/paper/handoff/K21/.
- **K22 (post-hoc): follow-up of K14 / K18.**
  - **Argoverse-HD at equal tensor px** (heuristics at 1.27–1.53x ours' px): no heuristic + refiner is above ours final (0.091); corridor + ref 0.082 [−0.023, +0.041]; ours is significantly above only temporal_crit.
  - **Same footing without refiner:**
    - nuScenes: ours above 4 of 5 heuristics;
    - AV2: above corridor / traj_support, uncertainty higher in point (0.080 vs 0.032);
    - Argoverse-HD: all n.s.
  - **K17's group d is a lift problem.** The decisive field-crop object is estimated closer (median −17.5 / −10.5 m) and closing faster (15.8 vs 5.6 m/s; 43.5 vs 6.1 m/s) than annotated. The annotated ttc alone removes the extra braking in 59% / 71% of frames. u960 detects the same object in 96% / 93% and is driven by it in 49% / 40%.

  Source: docs/paper/handoff/K22/.
- **K23 (development data only): v5 frozen pending the confirmatory run.** v5 is one shared configuration: frozen loop with v_close 0, crops at 960 px per long side, refiner v3a (v2 data + AV2 REF-TRAIN).
  - Dev results: nuScenes val 0.263 → 0.276 [+0.003, +0.031] at 0.77x px; AV2 CFG-DEV 0.113 → 0.204 [+0.061, +0.120] at 1.0x px.
  - Rejected: lateral / validity heads, evidence threshold (helps AV2, hurts nuScenes), temporal support, geometry, bounded loop. v5-L = v5-S.
  - Baselines also gain from v3a on dev (u960 0.207 / 0.158).

  Source: docs/paper/handoff/K23/; method_freeze_draft §2e–§2f.
- **K27 (development data only): v5.1 = v5 + robust looming, frozen pending the confirmatory run.**
  - Cause, from the diagnosis on v5: ours brakes alone mostly because of the ttc (AV2 88%), and 95% of these brakes are unnecessary. The crop box jitters between frames: 2.2 px on a 28-px box, vs 0.5 px for u960's box of the same object. Pairing with a draft box is a minor cause (26% on AV2).
  - Fix: the ttc of a crop object uses the median growth over 3 same-scale observations (T4), with a 1-px gate.
  - Dev results with v3a: nuScenes 0.276 → 0.302 [+0.011, +0.052]; AV2 0.204 → 0.272 [+0.038, +0.096], at 1.0× px.
  - v5.1-L (+ retrospective re-detection, 1.3× px) is n.s. over S.
  - The same fix helps every baseline (AV2: u960 +0.083). v5.1 stays above u960 + T4 on AV2 (+0.030 [+0.010, +0.053]); on nuScenes the difference is n.s.
  - Cost: G on the no-support regime falls on AV2, −0.054 [−0.138, +0.010] (n.s.).

  Source: docs/paper/handoff/K27/; method_freeze_draft §2g–§2h.
- **K28 consolidated (2026-10-14): handoff/K28/REPORT.md.** It maps every number of this file to its final v6 value (§1 there), gives the scorecard
  (S1 met at S only; S2, S3, S4 not met; S5 met at S with caveats; S6 met), lists every row where v6 is not above a baseline (908 pairs, 106 with the
  row significantly above; final/not_above_all.csv) and every test evaluation (final/test_evaluations.csv). **The §1–§7 numbers above are the
  pre-K28 configurations (v3 / v4-lite, AV2 unmasked); use the final column of that map.** Correction: on Argoverse-HD the true-V oracle router
  at S is "matches" against v6-S (+0.049 [−0.002, +0.101]), not outperformed.
- **K28 phase 3 (POST-HOC, on the phase-2 test units; v6 unchanged).**
  - **S4: not met.**
    - Significant on ≥ 2 datasets: field 3/3, support 2/3, looming estimate 3/3 (both C1 vs T4 and C1 vs none).
    - Guard 0/3: v6-S − without guard is −0.012 / +0.005 / +0.002; without the guard v6-S uses 18–23% fewer px.
    - Refiner 1/3. The multi-round loop (vs one round) is 0/3.
  - **S5: met at budget S, where rows exist.**
    - With YOLO11s (draft + verifier, no re-tuning), v6-S − v5.1-S is +0.009 [+0.002, +0.025] on nuScenes, +0.015 [−0.010, +0.037] on AV2 and +0.060 [+0.038, +0.084] on Argoverse-HD.
    - With fine-tuned verifiers on nuScenes: +0.012 [+0.003, +0.026] (seed 0) and +0.010 [+0.003, +0.023] (seed 1).
    - Every in-class (S) row is outperformed, but in-class rows exist only for Argoverse-HD YOLO11s and nuScenes ft0 / ft1 (whole frame 448).
    - With YOLO11s, whole frame + C1 is above v6-S on AV2 at 640 (1.9×) and 960 (4.3×), and on nuScenes at 1280 (8.6×).
  - **S3 (Jetson AGX Xavier, final window 2026-10-14): not met.**
    - AV2, 8 test logs with the ego mask, 586 frames: v6-S (G 0.419, 73.9 ms mean / 173 ms p95) is on the latency-G front in mean and p95. Nearest faster rows: v5.1-S ≤ 2 rounds 0.401 at 46 ms (+0.018 [−0.007, +0.042]); whole frame native + T4 + refiner 0.388 at 70 ms (+0.032 [−0.059, +0.124]).
    - nuScenes, 20 local DEEP scenes (not test), 797 frames: v6-S (0.096, 29.5 / 78 ms) is off the front in both. Example: whole frame u1280 + T4 + refiner, 0.118 at 25.6 / 30 ms (−0.022 [−0.149, +0.068]). On these scenes the refiner lowers v6-S's G (0.116 without it).
    - The C1 lift dominates the cost: whole frame u448 + T4 + refiner takes 10.4 / 15.1 ms; + C1 + refiner takes 22.4 / 51.6 ms.
    - Verifier calls set p95: 10–19 ms per call, R² 0.78–0.95.
    - Source: handoff/K28/phase3/jetson/ (s3_front.csv, s3_front.pdf).
    - **K29 FINAL (2026-10-18): a decision-identical fast C1 / refiner path, and the re-timed latency table (3 repetitions, MAXN; G post-hoc on test frames).**
      - Fast path trt32: C1 encoder in TensorRT FP32 with buckets, a crop / feature cache, and exact corridor / 80 m pruning.
      - Parity: 100% identical actions in all 32 C1 / refiner pipelines on the full nuScenes new val (6,117 frames) and AV2 CFG-DEV (10,123 frames); 0 differing frames on 275,688 timed test frames. Dev and test G are unchanged.
      - v6-S mean / p95, reference → trt32: nuScenes new test 20 scenes 28.8 / 83.5 → 21.9 / 59.2 ms; K28's old 20 scenes 28.6 / 76.8 → 23.0 / 54.4; AV2 8 logs 72.3 / 169.7 → 41.7 / 93.8.
      - Whole frame u960 + C1 + refiner: 50.8 → 22.7 ms (nuScenes test) and 98.2 → 34.3 (AV2). With the fast path C1 costs ≈ 2 ms over T4 in whole-frame pipelines.
      - **S3 is still not met: v6-S is on no latency-G front, in mean or p95, on any frame set.** Its own ≤ 2-round bound has the same G (+0.002 to +0.004) and is faster.
      - **v6-S ≤ 2 rounds is on the mean front on nuScenes test 20 (G 0.382, 19.8 ms) and on both fronts on AV2 (0.423, 37.1 / 73.5 ms).**
      - nuScenes test 20: u960 + C1 + ref is on both fronts (0.386 at 22.7 / 30.9 ms); v6-S − it −0.007 [−0.049, +0.044].
      - AV2: v6-S is above every whole frame + C1 + ref from u448 to u1280 (+0.054 to +0.108, all intervals > 0) and ties native + C1 + ref (+0.010 [−0.074, +0.093]) at 55% of its mean latency. tsp2@S + C1 + ref is above v6-S by 0.012 ([−0.029, −0.000]) at the same mean (41.7 ms) and a higher p95.
      - Tail: v6-S's p95 is 2.2–2.7× its mean (whole frame + C1 + ref 1.2–1.4×). After the fast path, the guard's verifier calls set the tail (p95 26–34 ms); C1 drops 5.6–6.1× in mean.
      - Source: handoff/K29/REPORT.md (§1, §6a–c; latency_fast.csv, front.csv, front_intervals.csv, tail.csv, breakdown.csv, front_fast.pdf).
  - **KITTI with the shared v6-S configuration:** mean of 4 cells 0.235 vs native + T4 + refiner 0.370 ([−0.182, −0.030]), at 38 vs 18 ms. v6-S is below whole frame + components. Planner B mono stays ≈ 0 for every pipeline here, unlike §6. **Cause found (2026-10-15): this run fed refiner v3a the nuScenes / AV2 camera flag (1.0) instead of KITTI's 0.0, so every "+ refiner" KITTI number here is affected (KITTI val |log range error| 0.084 vs 0.055 with the right flag). Do not use these refiner rows until the Jetson re-run (handoff/K28/phase3/REPORT.md §16).**
    - **K30 re-run on VUW (2026-10-16, torch YOLOv8s, k30_kitti.py; handoff/K30/STATUS.md).** The flag-1.0 rows reproduce the Jetson table within 0.005 on the same 3,336 frames (ours 0.240 vs 0.235, native + comp 0.368 vs 0.370). With the trained flag 0.0, Planner B mono comes back (v6-S 0.355, native + T4 + ref 0.381; ≈ 0 with flag 1.0). **Corrected table (flag 0.0, all 3,536 frames), mean of 4: v6-S 0.323 vs native + T4 + ref 0.448 (v6-S − it −0.126 [−0.175, −0.021]), native + C1 + ref 0.439 (−0.116 [−0.182, −0.031]), u960 + T4 + ref 0.368 (−0.046 [−0.088, 0.007]), native 0.327 (−0.005 n.s.).** The verdict "below whole frame + components" stands with the right flag. (KITTI timing is not re-measured; the 38 vs 18 ms Jetson numbers do not depend on the flag.)
  - **Streaming AP, Argoverse-HD** (K28, reference C1; superseded by K30 J3 with the fast path: v6-S 0.099, still below whole frame + C1 + v3a 0.144–0.198):** v6-S 0.064 (processes 56% of frames at 74.9 ms mean; 128 ms mean staleness) vs K7 cheap 0.096, u1280 0.247 and K7 ours 0.105. Source: handoff/K28/phase3/jetson/sap_ahd_v6.csv.
  - **S6: met on dev and on test (post-hoc).** Refiner seed std of G ≤ 0.0033 on dev and ≤ 0.0099 on test (AV2 v6-M; nuScenes v6-S 0.211 / 0.210 / 0.208), vs half the smallest S1 margin, 0.053 (or 0.011 over every in-class outperform margin). Seed 0 reproduces phase 2 on 100% of frames. AV2 v6-M spans 0.268–0.288 across seeds, so the AV2 S / M / L ordering is not seed-stable. Source: handoff/K28/phase3/seeds_test_v6.csv.
  - **AV2 without the ego mask:**
    - v6-S 0.617 vs v5.1-S 0.607, +0.009 [+0.002, +0.016]; v6-S is above its 3 in-class rows.
    - v6-M drops to 0.306 and v6-L to 0.542, below many heuristics. The unmasked G is inflated by ego-hood detections (K10); the cause of the v6-M drop is not verified.
  - **Prior works:**
    - v6-S/M/L outperform AdaScale and FOVEA on nuScenes, and AdaScale, FOVEA and SAHI on AV2.
    - nuScenes SAHI: v6-S is below-verdict with v6 higher.
    - The nuScenes fine-tuned-detector rows (FOVEA-ft, u960-ft, v3 + ft verifier) are above v6-S in the point estimate (−0.024 … −0.086, all below-verdict).
  - **Agreement and attribution:**
    - v6-S agrees with π(u960 + C1) on 97.6 / 96.8 / 94.0% of frames, a tie with the best heuristic + C1.
    - Along the counterfactual chain, exact locality removes 31 / 37 / 21% of the disagreement.
    - On the residual frames, the annotation sides with ours 44 / 34 / 398 times vs F 5 / 5 / 2.
  - **Missed objects:** v6-S recovers 61 / 85 / 87% of natural draft misses (u960 finds 100% by construction) and 85 / 88 / 85% of deleted objects; uncertainty crops recover 34–47%.
  - **Second consumer (TTC warning):** our loop vs uncertainty crops at the same px is +0.004 [−0.031, +0.041] on nuScenes, +0.079 [+0.048, +0.109] on AV2 and +0.077 [+0.045, +0.117] on Argoverse-HD.
  - **Detection AP:**
    - Argoverse-HD: v6-S 0.165 vs whole frame YOLOv8s 512 0.182 at the same GFLOPs.
    - nuScenes: v6-S has the highest relevant-object precision of every row (0.890) and the lowest relevant-FP rate (10.6 / 1k frames).
  - **K6 / NAVSIM:**
    - NAVSIM (spec_pdm) is unchanged by K28.
    - K6 ran nuScenes v3. The ours branch was re-run with v6-S (3 runs, 2026-10-14 / 15; route 11755 excluded as in K6).
    - **Three runs (DONE 2026-10-15):** v6-S DS 92.54 ± 0.82, SR 0.839 ± 0.011, RC 97.62 (219 routes per run).
      - Paired route bootstrap of v6-S − branch on DS averaged over the runs: cheap +0.27 [−0.96, +1.48], u960 −0.61 [−2.20, +1.03], u1600 +0.59 [−1.37, +2.42], K6 ours (v3) −0.26 [−1.23, +0.61], privileged −4.09 [−6.36, −2.02] (SR −0.076 [−0.117, −0.038]).
      - So v6-S ties every perception branch, the same null result as K6. It takes 29% fewer crops per frame than v3 (0.81 vs 1.15).
      - Source: handoff/K28/phase3/REPORT.md §14a, b2d_v6_{per_route,per_run,summary}.csv.

  Source: docs/paper/handoff/K28/phase3/REPORT.md.
- **K28 phase 2 (CONFIRMATORY, test splits run once): v6.** Configurations:
  - v6-S: YOLOv8s crops 960 + learned looming C1 + v3a;
  - v6-M: YOLOv8m crops 768 + C1 + own refiner;
  - v6-L: YOLO11m crops 960 + C1 + own refiner.
  - **Against v5.1-S (G with refiner):**
    - nuScenes: v6-S 0.211 vs 0.199, +0.012 [+0.004, +0.026] (outperforms); v6-M 0.298 (matches); v6-L 0.315, +0.116 [−0.031, +0.290] (below by the verdict rule).
    - AV2 masked: v6-S 0.294 vs 0.262, +0.032 [+0.013, +0.050]; v6-M 0.281 (matches); v6-L 0.292 (outperforms).
    - Argoverse-HD: v6-S 0.418 vs 0.338, +0.079 [+0.054, +0.107]; v6-M 0.428 and v6-L 0.431 (both outperform).
  - **S1 (in-class deployable baselines, every form incl. + C1):**
    - met at S: v6-S is above every in-class deployable row on all three splits.
    - not met at M / L, where the AV2 heuristics + C1 and one A-HD heuristic + C1 are above v6.
    - The true-V oracle router is above v6 on nuScenes and AV2.
  - **S2 (vs whole frame + components): not met.**
    - nuScenes: every resolution is below-verdict, with wide intervals.
    - K28 whole frame + C1 is above v6-S on AV2 from 640 up (n.s. except 1280).
    - Argoverse-HD: v6-S outperforms whole frame + components at every resolution.
  - **S4 (phase-2 part):** field 3/3, support 2/3 and looming 3/3 are significant; the refiner only on AV2 (1/3). The guard is still to come (phase 3).

  Source: docs/paper/handoff/K28/phase2/REPORT.md; method_freeze_draft §2n / §2n.1.
- **K24 (CONFIRMATORY, test splits run once): v5.1-S.**
  - **Against the previous frozen configuration (G with refiner):**
    - nuScenes new test: 0.199 vs v3 + v1 0.170, +0.029 [−0.018, +0.070] (matches);
    - AV2 val masked: 0.262 vs v4-lite + v2 0.122, +0.139 [+0.089, +0.190] (outperforms);
    - Argoverse-HD val: 0.338 vs 0.091, +0.248 [+0.159, +0.350] (outperforms).
    - px is equal or lower (0.077 / 0.053 / 0.070).
  - **Outperformed on every split:** learned routers (plain and + components), random crops, almost every plain heuristic, and plain whole frame on AV2 / A-HD.
  - **Not outperformed:**
    - most "+ components" baselines (v3a + T4): whole frame + components, and traj_support / corridor / temporal_crit / uncertainty + components. These are mostly "matches", lower bound > −0.02.
    - On nuScenes, the old v_close 10 loop + components (0.212, −0.014 [−0.043, +0.004]) and whole frame u1280 / u1536 (G 0.25–0.28 at 8–12× px).
    - The oracle router (true V), everywhere.
  - **2×2:** T4 is significant on all three splits (+0.023 / +0.054 / +0.154) and cuts unnecessary actions per 1k by 11 / 36 / 72. v3a is significant on nuScenes and AV2.
  - **Support only / field only + components:** each fails on at least one split. Support only gives K17(3)-regime G ≈ 0 everywhere; field only gives 0.003 / 0.050 on AV2 / A-HD.

  Source: docs/paper/handoff/K24/; method_freeze_draft §2i (+ revision).
- **Agreement with π(F), F = u960 (K2, post-hoc)**: ours (no refiner) nuScenes 0.951 [0.942, 0.959], AV2 0.810 [0.790, 0.831];
  same-area heuristics nuScenes 0.927–0.943, AV2 0.601–0.736; u640 0.943 / 0.685; cheap 0.915 / 0.555. Locality
  decomposition on 50 nuScenes test scenes: 0.950 → exact locality 0.990 → + F's ttc history 0.994.
- **Necessity ablations (K3, post-hoc; G with refiner, full − variant)**: nuScenes support-only −0.107 [+0.018, +0.498],
  field-only 0.118, one round 0.059, pivots 0.171; AV2 field-only −0.002 [+0.513, +0.614], pivots 0.516 [+0.036, +0.055],
  no guard 0.552 [+0.002, +0.017], draft history 0.537 [+0.009, +0.042], support-only 0.557 (n.s.).
- **Field contribution (K4, post-hoc)**: controlled deletion of every relevant draft detection — recovery ours / field-only /
  support-only / uncertainty / temporal_crit: nuScenes 0.964 / 0.920 / 0.149 / 0.389 / 0.819; AV2 0.789 / 0.240 / 0.186 /
  0.503 / 0.843. Natural misses (rare: 114 nuScenes, 360 AV2 frames): recovery ours 0.624 / 0.765.
- **Second consumer, TTC warning (K9)**: G ours (TTC field + support) nuScenes val 0.187, test 0.114, AV2 0.139 vs u960
  0.165 / 0.091 / 0.071 (AV2 ours − u960 [+0.021, +0.122]); uncertainty crops at the same area 0.192 / 0.149 / 0.211
  (beats ours on AV2: ours − it [−0.124, −0.025]). Follow-up (post-hoc): that control used 1.4–1.6× ours' pixels; at EQUAL pixels ours ≈ uncertainty (nuScenes −0.044 [−0.100, +0.008], AV2 +0.012 [−0.042, +0.065]); dev-tuned TTC loop did not transfer (nuScenes test 0.110 vs default 0.114). Source: results/vuw_summaries/k9_tune/, handoff/K9 follow-up.
- Network px recomputed from detector tensors (incl. 32-px rounding and guard re-detections; K3): ours nuScenes 0.100
  (p95 0.248), AV2 0.074 (p95 0.160) — the area-based 0.088 / 0.068 of §1 under-count by ~10–14%.
- **K30 post-hoc analyses of the frozen v6-S (DONE 2026-10-18 incl. Jetson J; handoff/K30/REPORT.md, rows in not_above_all.csv + J/j1_intervals.csv).**
  - **V1a frame-matched budget** (heuristics capped at m × v6-S's px per frame; realised px 1.2–1.8 × v6-S's because the guard grows crops):
    - at m = 1, v6-S is ≥ every S heuristic + C1 + v3a on all three datasets; the hybrid tsp8 ties it (nuScenes +0.037 [−0.012, 0.096], AV2 −0.003, Argoverse-HD +0.009);
    - with ~2 × the pixels the hybrid (AV2, m ≥ 1.25: −0.039 [−0.078, −0.008]) and the corridor rule cor0 (Argoverse-HD, m = 1.5: −0.072) overtake v6-S.
  - **V1b** pixel axis: flat around (960, 2); larger crops gain ≤ 0.017 at 1.4–1.8 × px. **V1c** hybrid with the TTC consumer's own support ties v6-S everywhere.
  - **V2** field vs tube: the share of v6-S's field outside the hybrid's tube is 4.6% / 10.3% / 20.8% (nuScenes / AV2 / Argoverse-HD). Where the decisive object lies outside the tube, v6-S beats the hybrid only on Argoverse-HD (+0.027 [0.008, 0.052]).
  - **V2b** IDM consumer: the frozen braking-driven v6-S is ≥ every row (nuScenes above ours-IDM / hybrid / unc2; Argoverse-HD above u960 / ours-IDM / unc2 / hybrid; AV2 n.s.).
  - **V3**: v6-S disagrees with u960 + C1 on 3.7% / 12.5% / 16.0% of frames and is right in 70% / 81% / 73% of them. Of its 60 costliest safety-critical frames, 46 are lift errors on a found object, 11 crop misses, 3 selection misses.
  - **V5** (27 cost-weight settings, 5 consumer settings, probes): v6-S ranks 2 of 10 on nuScenes / Argoverse-HD and 3–4 on AV2. It is above FOVEA on AV2 in every setting; vs SAHI on nuScenes it is n.s. except λ_brake × 2 (v6-S above). The vc10 probe beats the frozen probe on nuScenes by +0.026 [0.0003, 0.066].
  - **V7**: v6-S has the highest decision-relevant precision on all three datasets (0.799 / 0.719 / 0.511); whole frames have 1.5–1.6 × its AP50.
  - **V8c KITTI**, refiner flag fixed: v6-S 0.323 vs native + T4 + ref 0.448 (below). 2D AP Car (moderate) 42.9 vs draft 51.6 / native 75.3, because the crops truncate near cars.
  - **V8d** speed × 1.5: v6-S above the hybrid (+0.043 [0.003, 0.087]); × 0.5 all n.s.
  - **V8e**: corner-case objects (cones / barriers / bollards) are invisible to every COCO pipeline (recall ≤ 0.04).
  - **V6 (ii)**: measured probing time relative to 7 × 7 is 0.51 (5 × 5), 0.32 (3 × 3), 0.33 (plan corridor).
  - **V4** official Bench2Drive metrics: ties (DS v6-S 92.5, cheap 92.3, u960 93.1); Overtaking v6-S − u960 −0.096 [−0.188, −0.015].
  - **V4 in-loop** (4th Bench2Drive run with logging): v6-S makes fewer hard brakes per km than cheap (−19 [−28, −10]) and u1600 (−54), and fewer phantom-caused ones (vs cheap −2.0 [−4.1, −0.2]); its recall of actors within 3 s TTC is n.s. vs every branch; DS 93.2 vs cheap 92.5 (closed-loop outcome still ties).
  - **V6 (i) / V8a NAVSIM**: PDMS spec_pdm 0.8945 vs fixed-tube hybrid 0.8943 / native 0.8952; navtest EPDMS 0.8647 vs cheap 0.8624 / native 0.8644 / hybrid 0.8653; navhard two-stage EPDMS 0.509 vs 0.508–0.510 — all ties.
  - **K30-DEV** (dev only, pre-registered): conservative refiner variants are worse (AV2 dev −0.03 to −0.05); closing-speed probes below the bar → v6-S unchanged.
  - **J1 Jetson heuristics + C1 + refiner (fast path, ×1.0 budget, 3 reps)**: nuScenes test 20 v6-S 0.380 at 21.5 / 57.7 ms (mean / p95) vs hybrid tsp8 0.373 at 24.0 / 75.0 (+0.007 n.s.), tsp8-YOLO11m 0.392 at 28.2 / 89.2 (n.s.); AV2 v6-S 0.419 at 39.9 / 89.5 vs hybrid 0.417 at 44.5 / 109.0, tsp2 0.431 at 40.1 / 100.2 (−0.012 [−0.029, −0.000]). v6-S has the lowest p95 among the C1 heuristics on nuScenes; G ties all but AV2 tsp2.
  - **J2 energy per frame** (module rails, above the cheap run; nuScenes test 20 / AV2): v6-S 0.29 / 0.54 J; hybrid 0.28 / 0.62; whole frame + C1 + v3a u960 0.54 / 0.76, native 1.12 / 2.39, u448 0.20 / 0.41. One run each, no intervals.
  - **J3 Argoverse-HD streaming AP with the fast path**: v6-S 0.099 (K28 0.064; 82% of frames processed vs 56%), ≤ 2 calls 0.104; cheap 0.097; u960 + C1 + v3a 0.198; native + C1 + v3a 0.144. Corridor streaming AP50: v6-S 0.379 vs cheap 0.358, whole frame + C1 + v3a 0.434–0.468. Streaming AP stays below whole frames: detection coverage, not latency.
- **K31 post-hoc analyses of the frozen v6-S (FINAL 2026-10-20; handoff/K31/REPORT.md).**
  - **Item 1 episodes** (required level, ≥ DECELERATE; nuScenes / AV2 / Argoverse-HD):
    - v6-S has 4.0 / 6.5 / 6.3 false episodes per 1,000 frames (u960 + C1 + v3a 6.4 / 7.9 / 5.5) and 1.38 / 2.07 / 1.20 A→B→A reversals per 100 frames.
    - But on nuScenes it misses more required episodes than u960 + C1 + v3a: 0.408 vs 0.347 (+0.061 [0.014, 0.114]), mean onset +0.197 s [0.100, 0.313].
    - The refiner causes most late onsets (v6-S without it misses 0.388 / 0.142 / 0.211); at the hard-brake level no pipeline differs significantly from v6-S.
  - **Item 2 consumer adaptivity**:
    - The TTC loop beats the hybrid with the TTC support at the loop's own px on all three datasets: +0.034 [0.006, 0.069] / +0.024 [0.005, 0.045] / +0.026 [0.010, 0.044].
    - Re-targeting to IDM does not help: ours-IDM − frozen v6-S −0.056 / −0.008 (n.s.) / −0.011. Both find the leader; they differ in its lift.
  - **Item 3 curved-path consumer**:
    - Curved v6-S G 0.207 / 0.224 / 0.394. It ties the curved-tube hybrid (+0.022 / +0.025 / +0.016, n.s.) and beats the straight hybrid on Argoverse-HD (+0.042 [0.022, 0.064]).
    - On AV2, u960 + C1 + v3a stays above (−0.051 [−0.106, −0.007]), as in K28. The refiner on the curved support costs AV2's high-|κ| frames 0.10 G.
  - **Item 4 bounded loop** (≤ 2 calls, full test splits):
    - G 0.091 / 0.282 / 0.419 vs v6-S 0.211 / 0.294 / 0.418; nuScenes loss −0.120 [−0.299, −0.0001], 97% of it from one scene. Jetson parity 99.6–99.8%.
    - B2K-A (keep the pending truncated detection at the bound; dev-selected; post-hoc): test G 0.209 / 0.293 / 0.415 at the bounded loop's px (0.073 / 0.049 / 0.064). It is within ±0.003 of v6-S on every split (n.s.), so ≤ 2 detector calls lose nothing once the pending detection is kept.
  - **Item 6 NAVSIM, bottleneck selector** (argmax over 15 candidates of min(progress, min-object TTC); 12,146 navtest tokens):
    - DESP with oracle locality agrees with π(F) on 99.71% [99.58, 99.83] of tokens. All 35 disagreements are probe-grid misses: an object outside the crops that is sensitive alone, never a joint effect.
    - Real crops agree on 96.76%; the cheap draft on 94.04%; native whole frame on 95.92%.
    - Lemma check: 0 of 769,561 probe pairs change π jointly when neither does alone.
  - **Item 5 shift strata**: no night / rain / city stratum reverses the ranking. The only significant loss is AV2 WDC vs native + C1 + v3a (−0.200 [−0.444, −0.006], 18 logs).
- **K33 Table 1 v7: every baseline with our detector and our components (FINAL tables 2026-10-20; handoff/K33/REPORT.md, table1_v7.csv, fig_table1_v7).**
  Rows were dev-selected (12 candidates per family) and evaluated once on test. G is relative to the cheap draft;
  "v6-S − row" uses the paired unit bootstrap.
  - **At v6-S's per-frame pixel budget (@cap):** v6-S is significantly above 32 of the 39 A–E row × split pairs, and
    no capped row is above it. The 7 n.s. pairs are AdaScale and UFP on nuScenes / AV2, FOVEA KDE_S on AV2, and AnA ×2
    on AV2.
  - **Argoverse-HD:** v6-S (0.418 at px 0.070) beats every row, uncapped included: u1280 + comp. +0.103 [0.013, 0.278];
    SAHI (22× px) +0.189; AnA +0.103.
  - **nuScenes:** no row is significantly above v6-S (0.211). SAHI 0.303 (24× px) and u1280 0.292 (8.4× px) are n.s.
  - **AV2 (exception):** u1280 + comp. 0.348 (7.4× px; −0.055 [−0.113, −0.005]) and AnA iterated 0.349 (1.46× px;
    −0.055 [−0.113, −0.002]) are above v6-S (0.294).
    - Mechanism: same decisive object found, same range. On the frames where v6-S under-brakes and AnA does not, v6-S
      has a finite C1 TTC on only 57% vs AnA's 94%: AnA verifies every corridor detection in every frame, so its
      same-fidelity history is unbroken.
    - At equal px, AnA is n.s. (+0.014).
  - **Spurious braking per 1,000 frames:** v6-S 4.2 / 5.6 / 15.0, cheap 18.0 / 16.3 / 27.7.
  - **F (AV2 fine-tuned):** v6-S-ft 0.224 is the only F row above cheap; every other F row + comp. is below cheap.
    - v6-S-ft − u960-ft +0.287, − ESOD +0.344, − ZoomDet +0.259, − FOVEA-ft +0.304, − LZU-ft +0.305, − UFP-ft +0.233
      (all significant).
    - The fine-tuned detectors learned projected-cuboid boxes: shorter ranges and more HARD on KEEP.
  - **Jetson** (mean ms, nuScenes / AV2): v6-S 21.5 / 39.8; AnA iterated 25.4 / 68.3; AdaScale 43.4 / 70.3;
    SAHI 84.8 / 153.2; decision-value router 25.1 / 35.3; Chanakya 14.5 / 23.5.

- **K34-3 post-hoc frozen-v6-S Jetson timing (DONE 2026-10-22; handoff/K34/REPORT.md).**
  Exact K29 nuScenes test20 / AV2 8-log frame sets, trt32, three repetitions, paired unit bootstrap 2,000 seed 0; every run retained.
  - Mean / p95 / p99 ms (nuScenes; AV2): v6-S 21.84 / 59.71 / 83.69; 40.39 / 91.22 / 120.02. Ordinary ≤2: 19.90 / 46.20 / 55.57; 36.47 / 72.28 / 91.96. B2K-A: 20.31 / 49.12 / 59.96; 38.46 / 77.40 / 102.38.
  - B2K-A − v6-S mean: −1.53 [−2.83, −0.53] / −1.94 [−2.94, −1.03] ms; p95: −10.59 [−14.65, −2.50] / −13.82 [−22.79, −6.35] ms. B2K-A is slower than ordinary ≤2: mean +0.41 [0.20, 0.64] / +1.99 [1.04, 3.07] ms.
  - B2K-A >50 / >66.7 / >100 ms: 4.55% / 0.37% / 0%; 37.60% / 9.61% / 1.08%. All metric intervals and per-run rows: handoff/K34/jetson_latency_intervals.csv, jetson_latency_contrasts.csv, jetson_latency.csv.
  - All 12,519 timed rows validated; decisions identical across repetitions. B2K-A differs from VUW on 3/805 nuScenes and 2/586 AV2 frames, retained in jetson_disagreements.csv; exact parity is not achieved. The cap is two crop batches after the draft, not two total engine executions.
  - K34 CPU analyses are now complete below; the timing result alone does not resolve W1.

<!-- K34_CPU_AUTO -->
- **K34 post-hoc CPU analyses (complete).** Source: handoff/K34/REPORT.md and all CSVs. Severity/operating analyses retain every pipeline and threshold; coverage uses actual verified crops, with unknown classes retained. Selector/PDM timings compare different planners and historical hosts.
  - W1 remains: below-hard-demand missed episodes are higher than cheap on AV2 and Argoverse-HD. On Argoverse-HD, high-demand episode loss is also higher: +0.0234 [0.0029, 0.0515] of cheap total dataset loss (ten episodes).
  - Deployed v6-S is not dominated by sampled whole-frame points on nuScenes/AV2, but u960 + C1 + v3a at threshold 0.8 dominates its Argoverse-HD point estimate; false-alarm difference CI includes zero. Fine car/truck/bus coverage is unavailable on Argoverse-HD generic annotations.
  - nuScenes v6-S minus native insufficient-braking loss/frame: +0.0020 [-0.0551, +0.0494].
  - AV2 v6-S minus native insufficient-braking loss/frame: -0.0066 [-0.0737, +0.0518].
  - Argoverse-HD v6-S minus native insufficient-braking loss/frame: +0.0018 [-0.0130, +0.0148].

## Separate development optimization, 2026-10-23 (not a replacement for frozen results)

All changes are isolated under `experiments/`; main sources, weights, thresholds
and K34 remain unchanged. Full quality: 152 nuScenes scenes + 300 AV2 CFG-DEV
logs, 29,782 frames. All 30 factorial/control rows are retained. Intervals are
paired unit bootstrap, 2,000 draws, seed 0, pointwise and exploratory.
Source: [consolidated handoff](handoff/OPTIMIZATION_20261023/TONG_HOP.md);
remote summary copies: `results/vuw_summaries/optimization_20261023/manifest.json`.

- **B2K-A + temporal evidence versus frozen v6-S:** missed episodes 40→38/99
  nuScenes and 59→49/366 AV2, with no newly missed episode versus v6-S in these
  development records. G 0.321→0.351 / 0.309→0.319; differences +0.0294
  [0.0008, 0.0626] / +0.0107 [0.0018, 0.0215]. False episodes/1,000 increase
  +0.981 [0.163, 1.960] / +1.141 [0.593, 1.690]. Mean VUW pixels decline
  6.7% / 7.5%; guard overruns prevent an exact equal-realized-cost selector claim.
- **W1 remains:** cheap misses 34/99 / 30/366. The combined AV2 missed-rate
  excess is +5.19 [2.27, 8.01] percentage points. AV2 insufficient-braking
  loss/frame remains 0.3685 versus cheap 0.3140, +0.0546 [0.0235, 0.0844],
  even though below v6-S's 0.3798. G includes other costs and uses the prior
  cheap action for jerk; it is not a closed-loop safety outcome.
- **Jetson corrected AV2 CFG-DEV prefix:** existing trt32, 512 timed frames
  from 8 logs after 200 warm-up frames, three repetitions, independent policy
  histories; actual historical inference included. v6-S→combined mean
  37.80→34.60 ms, delta −3.21 [−7.33, 0.38]; p95 114.21→84.26 ms,
  −29.95 [−35.44, 2.37]; p99 143.41→109.30 ms, −34.11 [−46.03, −5.78].
  >100 ms declines 7.29%→1.56%, −5.73 [−13.92, −0.16] percentage points.
  >50 ms increases 24.15%→25.59%, difference CI includes zero. Actual tensor
  pixels increase 23.4% (common padded canvases), while detector executions
  fall 3.586→2.199/frame. Source: `handoff/VERIFIED_JETSON_20261023/`.
- **Exact implementation optimizations:** NAVSIM selector scope 7×7 mean
  115.86→12.98 ms, 8.93× [7.98, 9.89]; 3×3 28.11→7.62 ms, 3.69×
  [3.46, 3.92], all scores/actions/fields/support identical on 1,000 tokens ×
  two grids × three paired repetitions. This is not full planner/PDM-Closed
  latency. Crop-buffer staging reduces v6-S mean by 0.51/0.82 ms on the older
  nuScenes/AV2 engineering prefixes, with unchanged outputs. Fast recovery
  bookkeeping also preserves outputs; its pilot and staging gains have not
  been measured together. Sources: `handoff/NAVSIM_REUSE_20261023/`,
  `handoff/RUNTIME_20261023/`, `handoff/FAST_RECOVERY_20261023/`.
- **Dataset and parity limits:** K29's cache named `av2_dev` includes REF-TRAIN/
  REF-VAL logs; older AV2 engineering timings are retained with correction.
  The corrected manifest was fixed before inference. Actual available
  training/selection unit IDs have zero overlap with full quality development;
  checkpoint normalization recomputes exactly. Cross-backend parity remains
  imperfect: verified AV2 1/512 differences for v6-S and 3/512 for B2K-A/combined;
  nuScenes 2–3/512. All differing frames and unresolved causes are disclosed.
  No Argoverse-HD tuning, new confirmatory test or method promotion was done.


## Case-led optimization round 2 (2026-10-23; separate development exploration)

Completed full 152-scene/300-log evaluation, 29,782 frames and all 32 acquisition/refiner rows, plus 20 frozen-query CPU rows and a 20-row diagnostic association pilot. No main source, weights, thresholds or K34 change. All intervals paired unit bootstrap 2,000 seed0. [Consolidated report](handoff/OPTIMIZATION_R2_20261023/TONG_HOP.md), [all-row report](handoff/OPTIMIZATION_R2_20261023/REPORT.md), [completion audit](handoff/OPTIMIZATION_R2_20261023/completion_audit.json).

Relative to round-1 B2K-A+recovery, proactive+recovery reduces missed episodes 38→37/99 nuScenes and 49→48/366 AV2, with +0.81%/+1.55% requested pixels. AV2 false episodes/1,000 increase +0.423 [0.169,0.719]. The sole additional nuScenes rescue is a correct frame action caused by an erroneous TTC on a different non-required object; the actual decisive object remains unmatched. Verified looming+recovery yields 38/99 and 47/366, no additional hard-demand rescue, and only −0.00017 AV2 loss/frame. Proactive alone loses useful recovery episodes, including hard cases.

Half-log refinement on round-1 combined yields 37/99 and 43/366, AV2 loss/frame 0.36851→0.32631 (delta −0.04220 [−0.05261,−0.03270]), but false episodes +1.521 [0.590,2.585]/1,000 and G +0.00986 [−0.00218,0.02244]. nuScenes loss/G worsen. AV2 loss difference vs cheap becomes +0.01235 [−0.01467,0.03768], while missed episodes remain above cheap's 30/366. No method promoted.

Direct images and instrumented recovery confirm a same-class identity switch (GT track21→25) and a range improvement that crosses the wrong action threshold. A stricter rank-margin association removes the known switch but loses a useful early recovery in another episode; the diagnostic pilot is negative. The initial case export's strict person/cyclist class match was corrected to the frozen compatible matcher; aggregate evaluations were unaffected.

Frozen C1 audit reproduces original validation MAE within 3e-7 on 14,696 samples: C1-minus-T4 MAE −0.04665 [−0.07908,−0.01161] nuScenes and −0.03158 [−0.03934,−0.02355] AV2. High-true-rate underestimation does not imply an actionable global correction: AV2 predicted-rate bins are near calibrated. Original training triplets lack a historical-GT-identity check. This is checkpoint-selection validation, not independent test evidence. New Jetson timing was not justified by the small/negative quality gains.

## Mechanism-repair optimization round 3 (2026-10-23–24; exploratory)

Completed all 87 supervised VUW/VUW1 jobs and 228 full-development policy/refiner rows across eight suites. Seven suites use both 152 nuScenes scenes and 300 AV2 CFG-DEV logs; the identity gate uses nuScenes only because AV2 failed its training-holdout feasibility criterion. Regenerated 785 original training/holdout units, exactly aligned all 96,965 finite original C1 examples, and retained every negative row/failed attempt. Main source, deployed checkpoints, thresholds and K34 remain frozen. All intervals below: paired unit bootstrap 2,000, seed 0, pointwise 95%, exploratory. [Final handoff](handoff/OPTIMIZATION_R3_20261023/REPORT.md), [all numbers/intervals](handoff/OPTIMIZATION_R3_20261023/TONG_HOP.md), [completion audit](handoff/OPTIMIZATION_R3_20261023/completion_audit.json).

Against round1 B2K-A+recovery, critical-extent persistence leaves misses unchanged (38/99, 49/366) and false episodes essentially unchanged; loss/frame deltas are −0.00969 [−0.02748,−0.00036] nuScenes and −0.00744 [−0.01776,−0.00114] AV2. Its Jetson mean overhead is +0.90 [0.01,2.27] / +1.18 [0.14,3.00] ms; tails increase. Point fallback plus extent persistence reduces misses to 35/99 and 43/366, with loss deltas −0.03145 [−0.05458,−0.01249] / −0.03362 [−0.05357,−0.01825], G +0.032 [0.006,0.062] / +0.029 [0.009,0.052], and false episodes +1.308 [0.327,2.452] / +0.803 [0.381,1.267] per 1,000. Combined Jetson latency is unmeasured; component times must not be added as a guarantee.

The compact image-free readout reaches 33/99 and 48/366 misses, but G effects remain inconclusive; extra mean Jetson time is +1.86 [1.06,2.78] / +3.72 [1.63,6.16] ms. Uncertainty query with original v3a rescues 12 AV2 episodes with no new misses; 11 have decisive-object evidence, but none of its five hard-peak rescues has decisive-object hard-brake evidence. Joint decision-aware lifting instead increases AV2 loss +0.04668 [0.02101,0.07606], despite G +0.076 [0.034,0.118]; reject a G-only interpretation. Calibrated upper estimates increase false braking, and the learned identity gate rescues no additional nuScenes episode.

W1 remains: point+extent still misses 43 AV2 episodes versus cheap's 30, and loss 0.33489 is above cheap 0.31396, though false episodes 9.804/1,000 are much lower than cheap 35.960/1,000. All primary improvements are development findings; no independent test or closed-loop safety claim. Actual severe/unsupported rescue timelines and all changed episodes are retained. Frozen v3a CPU replay verifies all 187 changed-episode frame-policy actions where the primary trace stored a different refinement.

Both exclusive Jetson windows completed and restored hardware; existing engines only, 200 warm-up + 512 timed development frames per dataset, three repetitions. Within-Jetson port/encoder smoke parity and repetition parity pass. VUW/Jetson GT matches on common frames, but action differences of 1–6/512 remain across pipelines. Summary CSVs and PDF/PNG figures are mirrored under `results/vuw_summaries/OPTIMIZATION_R3_20261023/`. No core promotion.

## R4 — simple mechanisms and exact score reuse (2026-10-24, complete)

Isolated follow-up to R3: all 8 supervised GPU jobs, 58 development rows across acquisition/readout suites (42 distinct dataset/policy/refiner rows), two Jetson windows with three repetitions per dataset, and an exact CPU bottleneck-selector optimization. Frozen core, checkpoints, thresholds and K34 unchanged; no promotion or test-based quality selection.

The clearest new result is implementation-only: K31's fixed bottleneck selector reuses base object scores and batches probe-only columns. On K34's frozen 1,000-token / 125-unit NAVSIM navtest timing subset (post-hoc runtime only), mean 7×7 cost 100.969→10.957 ms, speedup 9.215× [8.245,10.182]; 3×3 24.112→6.392 ms, 3.773× [3.530,3.995]. Same host/process, one CPU thread, three repetitions, all 6,000 action/field/support comparisons identical. This is neither a PDM-Closed speedup nor full-perception timing.

Braking quality still trades off: objectwise cheap/raw/refined median reduces AV2 misses 49→37/366 and insufficient loss −0.05336 [−0.07145,−0.03851], but false-braking frames/1,000 53.29→108.94; nuScenes misses unchanged, no loss benefit. Nine of twelve AV2 rescues have decisive-object evidence, none is a new hard-peak rescue. One-round extent harms nuScenes despite lower pixels; containment/union/semantic changes add no episode rescue. Hard corroboration worsens AV2 G. All negative rows retained.

Refiner abstention preserves the new median policy, not frozen v6-S. Direct implementation overhead and its exact optimized implementation are timed separately; full/skip and old/new decisions are audited. Mean/p95/p99, threshold exceedance shares, pointwise paired bootstrap 2,000 seed0, all backend differences and hardware restoration are in the handoff. See [R4 report](handoff/OPTIMIZATION_R4_20261024/REPORT.md), [component review](handoff/OPTIMIZATION_R4_20261024/COMPONENT_REVIEW.md), [synthesis](handoff/OPTIMIZATION_R4_20261024/TONG_HOP.md), [completion audit](handoff/OPTIMIZATION_R4_20261024/completion_audit.json). Mirror: results/vuw_summaries/OPTIMIZATION_R4_20261024/.

## R5 — conditional decision repair (2026-10-25, complete)

Completed 23 supervised jobs, eight small classifiers and 104 full-development rows (44 new readout configurations per dataset plus eight controls). Original train/REF-TRAIN and unit-disjoint holdout only; all 152 nuScenes scenes / 300 masked AV2 CFG-DEV logs evaluated. Frozen core, weights, thresholds and K34 unchanged; no test-quality selection, promotion or new Jetson timing. [Report](handoff/OPTIMIZATION_R5_20261024/REPORT.md), [all rows](handoff/OPTIMIZATION_R5_20261024/all_rows.csv), [all paired effects](handoff/OPTIMIZATION_R5_20261024/all_contrasts.csv), [integrity audit](handoff/OPTIMIZATION_R5_20261024/completion_audit.json).

The useful incremental change is confidence-gated retention of current cheap evidence after a crop was executed but no compatible final detection remains. Added to R3 point+extent at fixed q=.5, nuScenes misses 35→34/99, loss .336997→.317066 (delta −.019931 [−.048364,−.000560]); AV2 misses remain 43/366, loss .334890→.333299 (−.001591 [−.004150,−.000227]). False episodes/1,000 rise +.326958 [0,.817929] / +.211282 [.042126,.463824]; false-braking frames rise +.490436 [0,1.143796] / +.760617 [.210790,1.565301]. All CIs paired unit bootstrap 2,000 seed0, pointwise exploratory. The additional nuScenes episode has decisive-object evidence; no extra hard-peak episode is rescued by this gated residual family.

Risk gains occur in only four nuScenes scenes and nine AV2 logs; one unit contributes 52.1%/67.9%. At q=.8, all nuScenes benefit comes from scene-1050 and its interval touches zero. Ungated cheap retention strongly increases false braking. All-object missing-history/rollback gates improve over round1 but do not replace R3's stronger severe-loss result. Conditional training on refiner downgrades gives only small gains; an extra AV2 episode at q=.5 lacks matched decisive-object evidence. Confidence is poorly calibrated specifically in the selected disagreement population.

The deeper image/population audit distinguishes a failed IoU match from absent visual evidence: truck fragments and image-edge/mask-overlap annotations complicate some severe cases. No GT, episode, matcher or frame is excluded. Scalar decision repair leaves the original geometry/refiner state and future acquisition intact; it does not establish repaired 3D estimates or extend the frozen selector theorem. Detector mean/p95 requested pixels are reported; classifier overhead and Jetson parity remain unmeasured. [Cases](handoff/OPTIMIZATION_R5_20261024/CASE_REVIEW.md), [calibration](handoff/OPTIMIZATION_R5_20261024/CALIBRATION.md), [synthesis](handoff/OPTIMIZATION_R5_20261024/TONG_HOP.md). Mirror: results/vuw_summaries/OPTIMIZATION_R5_20261024/.

### R6 — current-observation corroboration (2026-10-25)

Completed the final paper-informed sandbox round: 58 full-development rows (22 new configurations plus seven controls per dataset), all 152 nuScenes scenes and 300 AV2 CFG-DEV logs, 17 remote jobs and one exclusive Jetson component window. No training, test selection, core/checkpoint/threshold/K34 change or promotion. [Report](handoff/OPTIMIZATION_R6_20261025/REPORT.md), [every row](handoff/OPTIMIZATION_R6_20261025/all_rows.csv), [paired effects](handoff/OPTIMIZATION_R6_20261025/all_contrasts.csv), [direct cases](handoff/OPTIMIZATION_R6_20261025/CASE_REVIEW.md).

Borrowing ByteTrack's low-score continuation principle, R6 requires a current crop proposal and previous full observations, then caps the new action by observed historical actions. Two compositions with the R5 q=.5 candidate were explicitly declared after the nuScenes first-stage screen and before seeing AV2 results. They are adaptive, not confirmation. All 112,611 stored full boxes exactly match replayed high-score proposals; 100,022 low-score boxes were recovered from the same current queries, without extra logical detector pixels.

Against R5, the one-lag composition changes loss by −.012978 [−.038986,0] / −.000936 [−.002454,−.000085] (nuScenes / AV2), while adding +.326958 [0,.817929] / +.126769 [0,.295708] false-braking frames per1,000. The two-lag composition changes loss by −.005969 [−.017931,0] / −.000683 [−.002041,0]; no new false-HARD frame, no new nuScenes false frame, one new AV2 false frame. Misses remain34/99 and43/366. All intervals paired unit bootstrap2,000 seed0, pointwise exploratory.

The one-/two-lag rules repair two/one correctly supported HARD frames in nuScenes and one in AV2, within episodes already detected elsewhere. All nuScenes incremental gain is one night-car scene; AV2 gains remain concentrated in the same near-bus log seen in R5. An aggressive combination reaches31/38 misses but markedly increases false braking and lowers AV2 G. None of its five additional AV2 episodes has matched decisive-object evidence. Existence persistence does not fix wrong range/corridor estimates, truck fragments or unobserved near-edge objects.

The Jetson experiment measures only the bounded-readout CPU component on stored observations, three repetitions of all29,782 frames, with178,692 action evaluations and zero cross-platform mismatches. It excludes detector, refiner, R5 classifier, proposal deduplication/copies/buffering; it is not full-pipeline latency or trt32 parity. [Timing scope and intervals](handoff/OPTIMIZATION_R6_20261025/jetson/REPORT.md). Eight inspected three-frame case panels, every negative row, corrected-caption archive and source/exit/hardware audits are included. Mirror: results/vuw_summaries/OPTIMIZATION_R6_20261025/.
- **K32 closed-loop AEB, CARLA / Bench2Drive (FINAL 2026-10-25; handoff/K32/REPORT.md).**
  Setup: 155 routes × 3 seeds, L0 / L1; refiner and C1 off for every branch by the dev rule, so this is "selection with
  the geometric lift".
  - **Collisions:** 30–35% of routes in every branch, privileged included, almost all non-preventable. Not
    discriminative.
  - **Primary (G3, registered after an interim look; disclosed):** onset delay vs privileged on avoidable events, v6-S
    0.040 / 0.084 s (L0 / L1):
    - earlier than cheap (−0.050 / −0.039) and unc (−0.036 / −0.046);
    - ties hybrid and v6-S ≤ 2;
    - later than u960 / native (+0.011 / +0.026 s).
  - **Unnecessary hard-brake episodes per km at L0:** v6-S 39.5 vs cheap 49.8, u960 49.5, native 51.1 (all significant).
  - **Avoidable collisions at L1:** v6-S − cheap −0.019 [−0.037, −0.002].
  - **E1:** v6-S's decision changes are right 2.1–2.4× as often as wrong.
  - **E2':** v6-S avoids 0.9–1.3% and causes 0.3–0.4% of model-forked contacts.
  - **Shadow:** per-frame agreement with u960 + comp. 0.944–0.948.
