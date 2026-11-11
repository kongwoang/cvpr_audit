# K1 — canonical main-table CSVs and side statistics (2026-10-03)

**What was run.** No new experiment. Recomputed from existing per-frame results on VUW with
`remote/A_method/k1_tables.py` (md5 a2854f86…; run `python scripts/A_method/k1_tables.py` in the VUW main venv) and,
for KITTI, a local script over `results/raw/` (inline in this report's Part E). Copies: `results/vuw_summaries/handoff_K1/`.
Splits: nuScenes **new test** (228 scenes, 9,181 keyframes; confirmatory), AV2 **val** (150 logs, 11,818 frames), KITTI
**test** (6 sequences, 3,536 frames). All rows are the same frames as ours (cheap cost asserted equal on ≥ 99.9%).

**Definitions used for every row.** G = Σ(J_cheap − J_m)/Σ J_cheap (3 decimals). Interval = 95% paired bootstrap over
scenes / logs (2,000 resamples, seed 0, identical resamples for all rows) of G(ours final) − G(method). **Harmed share
= frames with J_m > J_cheap divided by affected frames (J_m ≠ J_cheap)** — the 0.036 in av2/baselines_summary__val.csv
was harmed frames / all frames; every row is recomputed with the affected-frame definition (uncertainty on AV2 = 0.085).
**Network px** = detector input pixels per frame / native pixels: uniform = its letterboxed tensor; ours and the crop
heuristics = draft tensor (nuScenes 192×320, AV2 256×320) + Σ crop area × scale² (scale 0.6 / 0.469); FOVEA KDE_I adds
the draft (its saliency); KDE_S, LZU (dataset saliency) have no draft; SAHI = slices × input² + full-image pass; UFP =
draft + its mosaic tensor; routers = draft (+ u960 on routed frames; skipping estimators drop the draft there).
Caveats: crop px use area × scale², not the 32-px-rounded tensors (slight under-count for ours and the heuristics);
guard re-detections are inside ours' crop list but **not counted for the heuristics** (their stored px are the selected
crops only); AdaScale's 320×180 scale-classifier input is not a detector pass and is not counted.

## A. Main tables → `table_main_nusc.csv`, `table_main_av2.csv`
Key rows (G, px mean / p95, harmed share, ours − it):
- nuScenes: ours final 0.171 (0.088 / 0.220, 0.385) · ours w/o refiner 0.135 [+0.004, +0.078] · u960 0.010 [−0.002,
  +0.346] · u1280 0.201 (0.654) [−0.141, +0.094] · native 0.121 [−0.048, +0.199] · SAHI 640/384-px slices 0.130 / 0.093
  [−0.121, +0.219] / [−0.081, +0.268] · uncertainty −0.092 [+0.018, +0.470] · temporal_crit 0.114 [+0.014, +0.111] ·
  random −0.018 [+0.088, +0.309] · traj_support 0.055 [+0.011, +0.226] · oracle router 0.342 [−0.250, −0.086] (oracle
  beats ours) · v4-lite + ref v1 / v2 0.166 / 0.169.
- AV2: ours final 0.561 (0.068 / 0.144, 0.087) · ours w/o refiner 0.537 [+0.002, +0.050] · random 0.102 [+0.418,
  +0.501] · native 0.558 [−0.042, +0.050] · oracle router 0.371 [+0.149, +0.232] · v3 + ref v1 0.545 [+0.005, +0.027].
Not run (no row): AV2 — FOVEA KDE_S, LZU, UFP, traj_support; Budget-Adaptive Routing on AV2 has one seed.
**Correction to 03_results:** the nuScenes range "0.007–0.080" was skip_offbin over its 3 seeds (0.028 / 0.079 / 0.007).
The official estimators at our budget span **−0.010 (cond_moric; Geng's arbiter picks it for all 3 seeds) to 0.079**.

## B. Definitions
- **uncertainty crops**: 2×-context boxes of the cheap detections, lowest confidence first (ties: highest class
  entropy), taken while they fit the frame's budget (strict).
- **temporal_crit** (Self-Cueing-style): the previous frame's *merged* objects of this policy, ordered by braking demand
  (then by range), each box enlarged by 25% per side; taken while they fit. Differs from support-only (K3 variant 3):
  it uses the previous frame's objects and a demand ranking instead of the current draft's support set, and never
  crops the field.
- **random**: up to 40 candidate boxes with sizes drawn from the cheap boxes (2× context; 200×120 px if none), random
  positions in the lower 65% of the image (seeded); taken while they fit.
- **traj_support** (VeteranAD-style): the support crops of the current draft (not capped) + an ego-corridor tube from
  10 m to 70 m, widened in 0.25 m steps until it fills the remaining budget.
- Budget (all four) = the per-frame crop area of frozen v3 in its confirmatory run; same detector, scale 0.6 (AV2:
  0.469), truncation guard, merge rule and merged-scene history as ours.
- **oracle router**: cheap draft on every frame; the k = ρ·n frames with the largest true decision value
  V = J_cheap − J_u960 (from the per-frame costs) are escalated to u960 (576×960 / 960×768 whole frame, lift with u960
  history); ρ = (px_ours − px_cheap)/px_u960, so mean network px equals ours (0.088 nuScenes, 0.068 AV2).
- **cond_V / skip_V** (our additions on bgt-ada's estimators): same models as the official cond_moric (XGBoost on the
  weak detections' features) / skip_offbin (MobileNetV2-Lite on the image), trained on V (cond_V: regression of V with
  ego speed added; skip_V: focal classification of V > 0) instead of ORIC; routed with bgt-ada's calibrated threshold.

## C. Refiner provenance → `refiner_provenance.csv`
- v1 (`out/resnet18`): KITTI DEEP train 11 seq (training) / val 4 seq (epoch) + nuScenes DEEP-85 train 45 scenes /
  val 16 scenes. KITTI test and nuScenes DEEP-85 test units are held out. **No overlap with nuScenes new test / new val
  or any AV2 split.**
- v2 (`out/resnet18_v2`): KITTI as v1 + nuScenes train 428 scenes (old + new train) / val 167 scenes for epoch
  selection, **including 151 of the 152 nuScenes new-val scenes**; **no nuScenes new-test scene; no AV2 data.**
  So no number in 03_results that uses v2 on nuScenes test is contaminated (v4-lite + ref v2 0.169; fine-tuned/curve rows).

## D. Router label statistics → `router_label_stats.csv`
nuScenes new train (15,475 frames): ORIC > 0 62.1%, V > 0 4.9% (V < 0 3.0%), Spearman(ORIC, V) +0.029 — confirms the
log. AV2 train (router train logs, 22,715 frames): 90.2%, 36.3% (6.8%), −0.003. Sources: router/frames__train.pkl,
av2_sota/frames__train.pkl.

## E. KITTI test, four cells → `kitti_cells.csv`
Mean of 4 cells, (each − u1248): u1248 0.323 · u1248 + refiner 0.449 [+0.037, +0.164] · round-1 choice 0.401 [+0.027,
+0.107] · round-2 choice 0.395 [+0.003, +0.108] · v3 + ref 0.389 [−0.005, +0.098] · SAHI 0.327 [−0.103, +0.083] ·
u960 0.231 [−0.148, −0.019]. Brake cells: every speculative variant is at or below u1248 (brake oracle, round 2
[−0.169, −0.010]); the mean is carried by Planner B mono where the refiner fixes range (+0.13 to +0.64).
SAHI comes from a separate run with identical cheap costs on all frames (results/raw/20260929_035016_kitti_baseline_cells).

## F. Fine-tuned table
nuScenes `v3_ftverifier`: draft = COCO YOLOv8s cheap (cached), verifier on crops = YOLOv8s fine-tuned at 960 on new
train; **no refiner** (no "+ refiner" result exists for either seed). AV2: FOVEA-ft, LZU-ft and UFP-ft were **not run**
(only u960-ft, ours-ft ± refiner v2, ZoomDet, ESOD).

## G. Cost statistics of ours → `cost_stats.csv`
nuScenes (v3): crops/frame 2.23 (p95 7), zero-crop frames 20.6%, rounds 0/1/2/3 = 20.6 / 75.4 / 3.9 / 0.1%, network px
0.088 (p95 0.220). AV2 (v4-lite): crops/frame 2.65 (p95 8), zero-crop 23.0%, network px 0.068 (p95 0.144).
**Not producible from existing files:** guard re-detections per frame (the crop count includes guard-grown crops but
they are not stored separately) and the AV2 v4-lite round distribution (not logged). They will come from the
instrumented re-run planned for K2/K3.

## H. NAVSIM reference 0.900 vs published 0.881
Sub-scores of our reference (GT objects): NC 0.946, DAC 0.990, TTC 0.866, C 0.998, **EP 0.936** (results/navsim_exp/
pdms_summary_final.csv). The PDM-Closed row reported in the NAVSIM paper (to be verified by the writer) has EP ≈ 0.899,
NC/DAC/TTC/C close to ours. EP is progress normalised by the best PDM-Closed proposal stored in the metric cache; our
reference planner is built exactly like the metric cache's PDM-Closed (MetricCacheProcessor planner inputs, same
proposals), so its progress is close to the normaliser. The EP term alone accounts for most of the gap
((5·EP + 5·TTC + 2·C)/12: 0.917 vs 0.903). Other candidates, not isolated: trajectory handover at 8 poses × 0.5 s (the
scorer interpolates), devkit v1.1 (commit 3e8291b) vs the paper's version. **Not pinned down** — state as a setup
difference; all branches share it.

## I. nuScenes sensitivity → `sensitivity_nusc.csv`, `loso_nusc.csv`
All 228 scenes: u960 0.010, u1280 0.201, ours 0.171 (w/o ref 0.135). Without scene-1051: u960 0.165, u1280 0.174,
ours 0.209 (0.162); ours − u960 [−0.011, +0.102], ours − u1280 [−0.032, +0.112]. Leave-one-scene-out range of
ours − u960: +0.043 … +0.179; of ours − u1280: −0.048 … +0.035.

**Against the method:** u1280 is above ours on nuScenes (point estimate, interval spans 0); the oracle frame router
beats ours on nuScenes (0.342 vs 0.171) though not on AV2; on KITTI brake cells u1248 is at or above every speculative
variant.

**Note (2026-10-05, K10):** every AV2 number in this report is unmasked. The AV2 ego hood drives 51% of the cheap draft's actions (81% of its loss); see handoff/K10 for the masked re-evaluation of the AV2 table.
