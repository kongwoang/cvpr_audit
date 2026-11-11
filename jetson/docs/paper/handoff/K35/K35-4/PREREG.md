# K35-4 PRE-REGISTRATION — confirmatory holdout evaluation (written 2026-10-25, BEFORE any holdout run)

This file is frozen. The holdout is run ONCE with the code and configuration hashed below. Every row is reported,
whatever the outcome. No selection, tuning, threshold or episode-definition change is based on holdout numbers.

## Holdout
- **Data.** Argoverse 1.1 3D-tracking TRAIN: the 65 logs of Argoverse-HD train.json (sha256 96eae4c6…af58481).
  - Unit manifest: unit_manifest.csv (sha256 31639158…4dafb69); units hold00…hold64 = train.json sid 0…64, 39,384 images.
  - Provenance: K35-0 (PASS), with no prior use found.
  - Ring_front_center at 30 Hz. Every frame is run for the histories. A frame is evaluated iff it has 3D labels:
    39,265 frames.
  - Required episodes: 154 (137 below hard / 17 at or above), from GT only (K35-0).
- **Protocol.** The frozen Argoverse-HD test path (k24_test 'ahd'), applied to train.json with roots train1–4.
  - Objects with projected height ≥ 10 px.
  - Deployed planner (τ1 1.0, τ2 3.5 m/s²), K28 cost with cheap's previous action in the jerk term, K31 episodes.
  - The AV2 configuration transfers, as for Argoverse-HD. All weights (YOLOv8s, C1, v3a) are frozen, and no AV1 data was
    ever used for training or selection.
- **Caveat (stated in advance).** Same sensor, cities and annotation process as the Argoverse-HD test split: these are
  independent logs, not an independent domain. Only 17 hard-level episodes exist, so hard-stratum results are imprecise.

## Rows (all frozen; no new tuning)
| row | source (isolated harness) | per-frame columns (action / cost / px) |
|---|---|---|
| v6-S (frozen) | k28_p3_h --part instr | a_v6S\|v3a / J_v6S\|v3a / px_v6S |
| cheap | k28_p3_h | a_cheap / J_cheap / px_cheap |
| u960 + C1 + v3a | k28_p3_h | a_wf960+C1\|v3a / J_wf960+C1\|v3a / px_wf960 |
| native + C1 + v3a | k28_p3_h | a_wf1920+C1\|v3a / … / px_wf1920 |
| support + tube hybrid tsp8@S + C1 + v3a | k28_p3_h | a_tsp8@S+C1\|v3a / … / px_tsp8@S+C1 |
| uncertainty unc2@S + C1 + v3a | k28_p3_h | a_unc2@S+C1\|v3a / … / px_unc2@S+C1 |
| AnA-style iterated (ana05hinfi) + C1 + v3a | k33_eval_h --part de --rows ana05hinfi --cap | a_ana05hinfi\|v3a / …; G_cap from @cap |
| **v7a** = B2K-A + temporal evidence | r3/runner_h (row v6S_b2A_retro) | a_v6S_b2A_retro\|v3a / … / px_v6S_b2A_retro (incl. recovery pixels) |

- v7a is exactly the W1_BOUNDED / OPTIMIZATION_20261023 "B2K-A + temporal evidence" row, with no R3 / R5 / R6 add-ons.
- **v7b is not a row.** It was dropped by the K35-4b gate before this file: its nuScenes-val p95 was 65.26 vs 62.75 ms
  for v6-S (K35-4b/REPORT.md).
- v7b's columns are computed by the same runner process, because runner_h executes its fixed row set. They are a
  by-product only. They are NOT analysed for any decision, and if reported at all they are labelled "dropped at the gate,
  not pre-registered".

## Primary metrics (per row, on the 39,265 evaluated frames)
G, missed rate of K31 required episodes (all; also below / at-or-above hard and the HARD level), false episodes per
1,000 frames, and mean network px. Paired unit bootstrap over the 65 units: 2,000 draws, seed 0, percentile 95%
(k34_common.bootstrap / interval). Differences are row − v6-S.

## Kill rule and decision rule (v7a primary; the only candidate)
- v7a is **negative and not proposed** if:
  - the lower bound of the CI of G(v7a) − G(v6-S) is < −0.01, **or**
  - missed rate(v7a) ≥ missed rate(v6-S) (point estimate, all required episodes).
- Otherwise v7a **passes** and is the proposed extension.
- If it passes, it is then run once post-hoc on the three test splits (labelled post-hoc), and timed on the Jetson on the
  K29 frames next to v6-S and B2K-A. That timing uses the K35 timing-note runtime and parity reporting.
- v7b (dropped) is not eligible under any outcome.

## Integrity checks (run before the analysis; failure = stop and report, no rerun with changes)
- check.py train: per unit, runner_h's recomputed cheap / v6-S (a, J, px) equal k28_p3_h's; 0 differences required.
- The analysis also asserts:
  - k33_eval_h's cheap / v6-S equal k28_p3_h's;
  - the evaluated frames and act_gt equal K35-0's GT-only computation;
  - 65 units, 39,265 frames.
- Adapter validation, already done before this file: Argoverse-HD VAL seq05 through the same harness in K35_AHD_SPLIT=val
  mode reproduces the stored test records exactly. k28_p3: 134 columns; k33: 18 columns; runner_h cheap / v6-S:
  6 columns; 0 differences.

## Code and configuration hashes (sha256)
Harness: VUW $B/experiments/k35_holdout_20261025 (build.py 71133d77…253c839f1; check.py 6ba6dfae…2660506d).
Generated:

| file | sha256 |
|---|---|
| k24_holdout.py | 4f4d0f80ef8062a64e02915566d269bbdc17e1671c8003ac963f4bd50404e8e4 |
| k28_p3_h.py | 63fa04ecaa1b542b941341e66412aa57d6cc47bf316dc83b7db682671d657d50 |
| k33_eval_h.py | 97811f28ef2dd1b38c41ead6d0e6574ea275d41732ffccfc30e167e46e5e94ed |
| r3/runner_h.py | d15e48bfc7df9269419346d93b2bded363d7d2390149a3e914db7b1ba5c71b25 |

Frozen sources:

| file | sha256 |
|---|---|
| scripts/A_method/k24_test.py | 61a82aa0d6036dba52bffc3cc5a3ad4109e0b72e30dcac117774c77166def638 |
| scripts/A_method/k28_p3.py | d59435860a8da5967addb86eca89e3c1953c0714ac5c32b712a6ff6f08956a8c |
| scripts/A_method/k33_eval.py | 61f57e0a504e49e75cc3f33f73ddd525f26df8229e7507f59ede57bc02448f78 |
| experiments/optimization_r3_20261023/runner_low.py | 27f8c2809d59cc7d5a6e02c0098a1aa2cad42115b111c42a3aaabe3012e472d1 |
| scripts/A_method/argoverse_hd_eval.py | 74b249fa8abc69e651908c1d0a55c75c4fe6a7d044cc2c0dc5f4c22cf74fc8cb |

Launch and analysis:

| file | sha256 |
|---|---|
| scripts/A_method/k35_k4_chain.sh | 6b430a9a35c1a975cd2fbb38b19f18a58c314a549307cd78d0ce1a745128158a |
| scripts/A_method/k35_k4_analyze.py | 00ffce851c39e3c92936a571c5a66c7f8b0dafcb47ce445447d750f85099855b |

Weights:

| file | sha256 |
|---|---|
| out/resnet18_v3a/image_refiner.pt (v3a) | 46bb704a6a83068116bc7443724931d56d6d4d098cea247ef55bb3ebc64fbaa9 |
| out/k28_temporal/temporal_head.pt (C1) | b9ebd4cc18d78e2ee9f3ad5bbbd01e43951ffeb02d126d3460df4feaa098916d |
| models/yolov8s.pt | 1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a |

- There is no git commit for the VUW workspace; the hashes above identify the code. The frozen-file manifests of
  K28 / K34 are unchanged (Codex review, 247 files).
- Launch: `k35_k4_chain.sh train i/4`, i = 0…3. Each is one run_when_gpu_free.sh group of three processes (k28_p3_h,
  k33_eval_h, runner_h), on the same unit shard.
- Outputs: results/k35/k4/train/. Analysis: `k35_k4_analyze.py` → results/k35/k4/train/tables/, copied with the
  REPORT to handoff/K35/K35-4/.
