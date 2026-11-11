# K28 consolidated REPORT: v6, the final configuration, and every number the paper uses

K28 asked for the method to be the best row "wherever that is achievable with legitimate means" under the hard rules R1–R6 (method_freeze_draft §2j), and for every open number in the paper to be closed. This report is item (i)–(iv) of the task's phase 3. The details stay in the phase reports:
- phase0/REPORT.md: v5.1-S items that needed no new detector run (post-hoc);
- phase 1: no REPORT of its own; method_freeze_draft §2j–§2m and phase1/*.csv (development data only);
- phase2/REPORT.md: the one confirmatory test pass of v6 (§2n / §2n.1);
- phase3/REPORT.md: every remaining item, post-hoc on the phase-2 test units (incl. the Jetson window §16, the panels §17 and K6 §14).

**State on 2026-10-15 (VUW 09:40).** Everything is final. The K6 Bench2Drive re-run with v6-S finished all three runs (§14a of phase 3); its rows below use the three-run table.

**Final configuration.** v6, frozen in §2n before the test pass. G is with each configuration's refiner unless stated.

| name | draft | crop detector | crop long side | looming estimate | refiner |
|---|---|---|---|---|---|
| **v6-S** (the main configuration) | YOLOv8s 320 | YOLOv8s | 960 | C1 (learned, 3 crops) | v3a |
| v6-M | YOLOv8s 320 | YOLOv8m | 768 | C1 | own YOLOv8m refiner, seed 0 |
| v6-L | YOLOv8s 320 | YOLO11m | 960 | C1 | own YOLO11m refiner, seed 0 |

**Statistics.** Intervals are paired unit bootstraps (2,000 resamples, seed 0) of ours − row, by scene / log / sequence (Bench2Drive: by route). Verdicts are K24's: outperforms (lower bound > 0), matches (lower bound > −0.02), below (otherwise). "Below" includes rows where ours has the higher point estimate.

## 0. Summary

- **v6 vs v5.1-S (confirmatory):** v6-S is above v5.1-S on all three test splits at the same cost: nuScenes 0.211 vs 0.199, AV2 (ego mask) 0.294 vs 0.262, Argoverse-HD 0.418 vs 0.338. v6-M / v6-L add little on nuScenes and AV2 (wide intervals; AV2 v6-M 0.281 and v6-L 0.292 are not above v6-S) and a little on Argoverse-HD (0.428 / 0.431).
- **Scorecard (§2):** S1 met at budget S only; S2 not met; S3 not met; S4 not met (guard 0 / 3, refiner 1 / 3); S5 met at S where in-class rows exist; S6 met.
- **Where v6 is not the best row (§3):** 908 (ours, row) pairs across all benchmarks are not "outperforms"; in 106 of them the row is significantly above ours.
  - In the confirmatory pass, only two deployable rows are significantly above v6 within its own cost class: AV2 heuristic tsp8@M + C1 and Argoverse-HD heuristic cor0@M + C1, both against v6-L.
  - The other 42 significant pairs of the pass compare v6 with rows of a higher cost class (up to 37× the GFLOPs of v5.1-S), and 16 are the true-V oracle router.
  - The rest come from post-hoc settings: AV2 without the ego mask (34), YOLO11s (3), the Jetson front (own variants, 4), KITTI (4) and the privileged Bench2Drive branch (1).
- **Main-text numbers (§1):** every headline number of 03_results changes. The largest changes: AV2 ours 0.561 (unmasked, v4-lite) → 0.294 (masked, v6-S), and the AV2 oracle router, which the kit says "reaches only 0.371", is ABOVE v6-S on the masked protocol (0.418, −0.125 [−0.173, −0.080]). The Argoverse-HD number moves from 0.091 to 0.418. Streaming AP gets worse (0.105 → 0.064).
- **Corrections carried into this report (§5):** the phase-2 sentence "Argoverse-HD: [the oracle router is] below v6 at every budget" is wrong at S: there it is "matches" (+0.049 [−0.002, +0.101]), so v6-S does not outperform it. Phase 2's REPORT is corrected in place.

## 1. Main-text number → final number

"Kit" is docs/paper/03_results.md as of 2026-10-14 (§ numbers refer to it). "Final" is v6-S unless another configuration is named, on the K24 / K28 protocol: nuScenes new test, AV2 val **with the ego mask** (K10: report the masked numbers), Argoverse-HD val. Where the kit number is on a protocol that K28 did not re-run, the final column says so; such numbers either stay as they are (labelled with their configuration) or leave the main text.

### 1a. Headline tables (kit §1)

| # | kit claim | kit number | final number | source | note |
|---|---|---|---|---|---|
| 1 | AV2 ours | v4-lite + refiner v2 **0.561** (unmasked), px 0.068; masked 0.122 (K10) | **v6-S 0.294**, px 0.054 (v6-M 0.281, v6-L 0.292); without mask (post-hoc) 0.617 | phase2 §1, phase3 §3 | masked protocol (K10) |
| 2 | AV2 v3 + refiner | 0.545 | not re-run; previous frozen v4-lite + v2 (masked) 0.122, v6-S − it +0.171 [+0.115, +0.229] | phase2 table_main_av2 | v3 row leaves the AV2 table |
| 3 | AV2 uniform native | 0.558, tie | native + components 0.224: +0.069 [−0.008, +0.151] (matches); native + C1 (2048) 0.294: −0.000 [−0.077, +0.080] (below); plain 0.155: +0.139 [+0.060, +0.222] | phase2 §3 S2 | v6-S uses 5% of native px |
| 4 | AV2 uniform 1536 | 0.540 | + components 0.233: +0.060 [+0.003, +0.114]; + C1 0.328: −0.034 [−0.102, +0.025] | phase2 §3 | |
| 5 | AV2 uniform 1280 | 0.484 | + components 0.272: +0.022 [−0.031, +0.070] (below); + C1 0.348: **−0.055 [−0.113, −0.005]** | phase2 §3 | row significantly above v6-S (13% of its px) |
| 6 | AV2 uniform 960 | 0.446 | + components 0.255: +0.039 [−0.012, +0.084] (matches); + C1 0.338: −0.045 [−0.102, +0.004] | phase2 §3 | |
| 7 | AV2 uniform 640 | 0.144 | + components 0.219: +0.075 [+0.026, +0.118]; + C1 0.311: −0.017 [−0.076, +0.035] | phase2 §3 | |
| 8 | AV2 uncertainty crops | 0.494 | uncertainty @ 1× v5.1-S px + components 0.210: +0.083 [+0.028, +0.134] | phase2 table_main_av2 | |
| 9 | AV2 AdaScale | 0.317 | −0.057 (masked): +0.350 [+0.268, +0.432] | phase3 §4 | |
| 10 | AV2 Budget-Adaptive Routing | 0.077 | best BAR picked on val (cond_moric) 0.023: +0.270 [+0.206, +0.338]; re-matched to v6-S px (skip_offbin) 0.007: +0.286 [+0.228, +0.346] | phase2 §5 | |
| 11 | AV2 best router (cond_V) | 0.230 | 0.013: +0.281 [+0.220, +0.343] | phase2 §5 | |
| 12 | AV2 oracle frame router | 0.371 ("reaches only 0.371") | **0.418 @ v6-S px: −0.125 [−0.173, −0.080]** | phase2 §5 | the oracle is ABOVE v6-S; the kit's message reverses |
| 13 | AV2 FOVEA KDE_I | 0.328 | −0.023: +0.317 [+0.233, +0.403] | phase3 §4 | |
| 14 | AV2 temporal-criticality crops | 0.174 | temporal_crit @ 1× + components 0.224: +0.070 [+0.020, +0.117] | phase2 table_main_av2 | |
| 15 | AV2 SAHI (640 / 320 slices) | 0.088 / −0.140 | scale 0.47 −0.295, scale 1.0 −0.248: +0.589 / +0.541 (lower bounds +0.477 / +0.425) | phase3 §4 | |
| 16 | AV2 harmed share | ours 0.087, u960 0.155 | not recomputed for v6 | – | leaves the main text or stays labelled v4-lite (unmasked) |
| 17 | AV2 message | "ties native with ~1/15 of its px; beats every SOTA baseline; a perfect router reaches only 0.371" | v6-S matches native + components at 5% of its px; K28 whole frame + C1 has the higher point estimate from 640 up (significant at 1280); v6-S outperforms every prior work; the oracle router is above v6-S | phase2 §3, §5; phase3 §4 | |
| 18 | nuScenes ours | v3 + refiner **0.171**, px 0.088 | **v6-S 0.211**, px 0.078 (v6-M 0.298, v6-L 0.315); v6-S − v3 + v1 +0.041 [−0.005, +0.087] (matches) | phase2 §1, table_main_nusc | |
| 19 | nuScenes v3 (no refiner) | 0.135 | unchanged (roadmap step) | phase3 §9 | |
| 20 | nuScenes uniform 1280 | 0.201 | + components 0.269: −0.058 [−0.151, +0.041]; + C1 0.292: −0.082 [−0.165, +0.017] | phase2 §3 | below by verdict; rows higher in point estimate |
| 21 | nuScenes uniform native | 0.121 | plain 0.176, + components 0.197: +0.014 [−0.065, +0.143]; + C1 (1600) 0.222: −0.011 [−0.087, +0.122] | phase2 §3 | K24 / K28 rows carry the v3a refiner |
| 22 | nuScenes uniform 960 | 0.010 | + components 0.096: +0.114 [−0.060, +0.320]; + C1 0.112: +0.099 [−0.078, +0.308] | phase2 §3 | below by verdict, v6 higher |
| 23 | nuScenes uniform 640 | −0.181 | + components −0.069: +0.280 [−0.024, +0.691]; + C1 −0.036: +0.246 [−0.070, +0.666] | phase2 §3 | |
| 24 | nuScenes AdaScale | 0.083 | 0.083: +0.127 [+0.068, +0.198] | phase3 §4 | |
| 25 | nuScenes FOVEA KDE_I / KDE_S 576×960 | 0.074 / 0.022 | +0.137 [+0.040, +0.253] / +0.189 [+0.064, +0.312] | phase3 §4 | |
| 26 | nuScenes Budget-Adaptive Routing | −0.010 … 0.079 | best BAR picked on val (cond_moric) −0.012: +0.223 [+0.125, +0.348] | phase2 §5 | |
| 27 | nuScenes decision-value router | 0.028 | cond_V 0.030: +0.181 [+0.094, +0.263] | phase2 §5 | |
| 28 | nuScenes oracle frame router | 0.342 (above ours) | 0.365 @ v6-S px: **−0.154 [−0.231, −0.073]** | phase2 §5 | still above |
| 29 | nuScenes SAHI (640 slices) | 0.130 | scale 1.0 0.130: +0.081 [−0.084, +0.274]; scale 0.6 0.093: +0.118 [−0.045, +0.320] | phase3 §4 | below by verdict, v6 higher |
| 30 | nuScenes temporal-crit / uncertainty (strict budget) | 0.114 / −0.092 | @ 1× + components 0.162 / −0.062: +0.049 [+0.020, +0.090] / +0.273 [+0.033, +0.482] | phase2 table_main_nusc | |
| 31 | nuScenes LZU / UFP frozen | −0.861 / −0.678 | G unchanged (K24 results reused); no v6 interval computed | – | open only if the paper keeps these rows with intervals |
| 32 | nuScenes pre-registered rule of confirmatory 1 ("beats u640, matches u960") and the scene-1051 sensitivity | v3 + ref | historical (confirmatory 1 of v3); not recomputed for v6 | method_freeze §5 | keep labelled v3 |

### 1b. Fine-tuned detectors, planning benchmarks, detection metrics, latency, KITTI (kit §2–§7)

| # | kit claim | kit number | final number | source | note |
|---|---|---|---|---|---|
| 33 | nuScenes ours with fine-tuned verifier | v3 0.297 / 0.295 (K12: + refiner v1 0.330 / 0.332) | **v6-S with fine-tuned verifier 0.359 / 0.364** (seeds 0 / 1); v6-S − v5.1-S in the same setting +0.012 [+0.003, +0.026] / +0.010 [+0.003, +0.023] | phase3 §1 S5 | post-hoc |
| 34 | nuScenes fine-tuned prior rows vs ours | ours − u960-ft s0 [+0.015, +0.109] | COCO v6-S vs FOVEA-ft −0.024 [−0.210, +0.163] / −0.034; u960-ft −0.030 / −0.062; v3 + fine-tuned verifier −0.086 / −0.084 (all below, row higher) | phase3 §4 | COCO v6-S is not above the fine-tuned rows |
| 35 | nuScenes ZoomDet / ESOD / LZU-ft / UFP-ft | 0.259 … 0.180 | G unchanged; no v6 interval computed | – | |
| 36 | AV2 fine-tuned (ours-ft + ref 0.546 unmasked; masked K13 −0.100) | | not re-run with v6; K13's masked values stand (fine-tuned AV2 rows are hood-driven, K13) | handoff/K13 | |
| 37 | NAVSIM navtest, ours (spec_pdm) | 0.8945 | **unchanged** (spec_pdm does not use the braking-field loop that K28 changed) | phase3 §14 | |
| 38 | nuPlan-mini | v3 0.9401–0.9404 | unchanged (dev evidence; v3) | – | |
| 39 | Detection metrics nuScenes (AP50 / relevant recall / relevant precision / relevant FP per 1k) | v3 + ref .191 / .793 / .879 / 12.7 | **v6-S .177 / .786 / .890 / 10.6**; cheap .155 / .728 / .762 / 30.6 and u960 .284 / .824 / .736 / 39.5 unchanged | phase3 §11 | v6-S has the highest relevant precision and the lowest relevant-FP rate of every row |
| 40 | KITTI latency (Jetson) | v3 + ref G 0.234, 26.7 ms mean / 50.5 p95 | v6-S (shared configuration, v3a) brake mono G 0.144, **38.4 ms mean / 111.8 p95**; native + T4 + refiner 0.371 at 18.3 ms; native 0.319 at 15.4 ms | phase3 §16 | different refiner (v3a vs K1) |
| 41 | KITTI four cells, mean of 4 | u1248 + our refiner 0.449; v3 + ref 0.389 | **v6-S 0.235**; native + T4 + refiner 0.370 (v6-S − it [−0.182, −0.030]); native 0.336; u960 + T4 + refiner 0.294 | phase3 §16 | Planner B mono ≈ 0 for every pipeline here, unlike kit §6 (open, §6 below) |
| 42 | Budget curve: 1 round vs 3 rounds (nuScenes) | 0.016 vs 0.135 (v3) | v6-S single round 0.095 vs 0.211: +0.116 [−0.005, +0.296]; AV2 +0.005, Argoverse-HD −0.004 | phase3 §2 | multi-round not significant on any split (0 / 3) |
| 43 | Frame routing vs region verification | AV2 oracle 0.371 vs ours 0.561; nuScenes 0.342 vs 0.171 | AV2 (masked) oracle 0.418 vs v6-S 0.294; nuScenes 0.365 vs 0.211; Argoverse-HD 0.369 vs 0.418 (+0.049 [−0.002, +0.101], matches) | phase2 §5 | the oracle is above on two of three splits |
| 44 | Heuristic rankings flip across datasets | qualitative | not re-checked as a claim; per-row values in phase2 table_main_*_v6.csv | – | |

### 1c. Writer-handoff items (kit §8)

| # | kit claim | kit number | final number | source | note |
|---|---|---|---|---|---|
| 45 | K7 Argoverse-HD G | v4-lite + v2 **0.091**; whole frame negative | **v6-S 0.418** (v6-M 0.428, v6-L 0.431); whole frame + components 0.262–0.287, v6-S above each (e.g. native + components +0.135 [+0.050, +0.233]) | phase2 §1, §3 | |
| 46 | K7 AP (3 road-user groups) | ours 0.163; u960 0.286; native 0.368 | v6-S 0.165 (v6-L 0.189); whole frame YOLOv8s 512 0.182 at the same GFLOPs; 960 0.286; 1920 0.368 | phase3 §11 | |
| 47 | K7 streaming AP | ours 0.105 (31.5 ms) | **v6-S 0.064** (74.9 ms mean, 56% of frames processed, 128 ms staleness); K7 cheap 0.096, u1280 0.247 | phase3 §16 | worse than K7 ours and than cheap |
| 48 | K6 Bench2Drive | ours (v3) DS 92.80 ± 0.63 (3 runs); ties every perception branch | **v6-S DS 92.54 ± 0.82, SR 0.839 ± 0.011 (3 runs)**; ties cheap (+0.27 [−0.96, +1.48]), u960 (−0.61 [−2.20, +1.03]), u1600 (+0.59 [−1.37, +2.42]) and K6 v3 (−0.26 [−1.23, +0.61]); below privileged (−4.09 [−6.36, −2.02]) | phase3 §14a | the row may read v6-S; the closed-loop result is null as in K6 |
| 49 | K5 / K11 Jetson latency, nuScenes | ours + ref 25.3 ms (p95 53.2), G 0.173 (85 old DEEP scenes) | v6-S 29.5 ms mean / 78.4 p95, G 0.096 (20 old DEEP scenes, 797 frames) | phase3 §16 | different scene set; not the new test |
| 50 | K5 / K11 / K13 Jetson latency, AV2 | 42.0 ms (p99 117) unmasked; K13 masked 37.8 ms, G 0.173 | v6-S 73.9 ms mean / 173 p95, G 0.419 (8 val logs, mask, 586 frames) | phase3 §16 | C1 and verifier calls set the cost |
| 51 | K5 "ours is on the latency-G front" | AV2 yes, equal-G passes faster | S3: AV2 on the front (mean and p95); nuScenes off the front in both | phase3 §16 | |
| 52 | K10 masked AV2 | ours 0.122 | v6-S 0.294 (= row 1) | phase2 §1 | |
| 53 | K12 tensor px of ours | nuScenes 0.100, AV2 0.074 | v6-S 0.078 / 0.054 / 0.070 (mean), p95 0.176 / 0.129 / 0.175 | phase3 §10 | v6-S takes fewer px than v3 |
| 54 | K12 routers re-matched | nuScenes oracle 0.345 > ours 0.171; learned routers ≤ 0.044 / AV2 ≤ 0.252 | oracle 0.365 / 0.418 / 0.369 vs v6-S 0.211 / 0.294 / 0.418; learned routers ≤ 0.078 (nuScenes) and ≤ 0.082 (AV2) after re-matching to v6-S / M / L, all outperformed | phase2 §5 | |
| 55 | K13 masked ablation pattern | field alone −0.035; support only 0.122 = full; guard needed; pivots worse | v6-S: field only 0.021 (+0.273 sig.), support only 0.218 (+0.075 sig.), guard n.s. (+0.005), pivots +0.066 sig., refiner +0.053 sig. | phase3 §2 | the guard is not significant on AV2 (or anywhere) |
| 56 | K14 "+ refiner" controls | ours not significantly above every + refiner control | superseded by phase 2: every control with T4 + v3a and with C1 + v3a; S1 met at S only | phase2 §3 | |
| 57 | K15 Figure 1 frames | v3 / v4-lite on 4 frames | the same 4 frames re-rendered with v6-S (K15 frames of the panel set) | phase3 §17 | |
| 58 | K16 attribution of the disagreement with π(F) | F = u960; locality 81% / 62%, residual 7% / 4% | F = u960 + C1; disagreement 2.4 / 3.2 / 6.0%; exact locality removes 31 / 37 / 21%; residual 22 / 11 / 48%; on residual frames the annotation sides with ours 44 / 34 / 398 vs F 5 / 5 / 2 | phase3 §5 | different reference policy |
| 59 | K17 field decisive in the no-support regime | G 0.35 / 0.33 / 0.55 | v6-S 0.331 / 0.289 / 0.684; u960 + C1 0.409 / 0.414 / 0.684 | phase3 §7 | whole frame + C1 is higher on nuScenes / AV2 in this regime (no interval) |
| 60 | K18 tensor-px matched heuristics | ours above or tied with every heuristic + refiner | superseded by phase 2 class matching (S met, M / L not) | phase2 §3 | |
| 61 | K18(B) bounded variant (≤ 2 calls) | AV2 p95 58 ms vs 82 | v6-S ≤ 2 rounds: nuScenes p95 64 vs 78 ms; AV2 202 vs 173 ms (worse: more px) | phase3 §16 | |
| 62 | K19 missed decelerations vs u960 | ours misses more: +3.3–8.5 / +0.5–4.0 / +0.9–4.7 per 1k | v6-S − u960 + components: [+4.46, +9.69] / [+4.31, +10.67] / [+0.45, +12.63] per 1k; − u960 + C1: [+3.05, +7.63] / [+0.34, +4.14] / [−1.95, +3.85] | phase3 loss_terms_*_v6.csv | v6-S still misses more decelerations |
| 63 | K19 collisions | 8 frames / 3 scenes; 5 / 3; 0 | v6-S 9 (4) / 5 (3) / 0 | phase3 §12 | |
| 64 | K19 sign tests vs u960 | 82 vs 52 (p 0.012); 97 vs 45 (p < 0.001) | vs u960 + components: 55 vs 43 (p 0.27); 97 vs 38 (p < 0.001); 19 vs 2 (p 0.0002) | phase3 §12 | n.s. on nuScenes |
| 65 | K19 multi-obstacle G | AV2 0.103; ~0 elsewhere | v6-S 0.019 / 0.244 / 0.344 | phase3 §12 | |
| 66 | K21 YOLO11s, no re-tuning | 0.388 / 0.139 | **v6-S 0.418 / 0.261 / 0.422**; v6-S − v5.1-S +0.009 [+0.002, +0.025] / +0.015 [−0.010, +0.037] / +0.060 [+0.038, +0.084] | phase3 §1 S5 | |
| 67 | K22 Argoverse-HD at equal px | corridor + ref 0.082 vs ours 0.091 (n.s.) | v6-S above every in-class row; at L, heuristic cor0@M + C1 is above v6-L (−0.060 [−0.126, −0.001]) | phase2 §2 | |
| 68 | K24 v5.1-S | 0.199 / 0.262 / 0.338 | v6-S 0.211 / 0.294 / 0.418 | phase2 §1 | |
| 69 | K2 agreement with π(F) | F = u960: 0.951 / 0.810 | F = u960 + C1: v6-S 0.976 / 0.968 / 0.940; best heuristic + C1 ties (+0.002 / +0.000 / +0.001) | phase3 §5 | different reference policy |
| 70 | K3 necessity ablations | v3 / v4-lite (unmasked AV2) | v6-S, every component (phase3 §2) | phase3 §1–§2 | |
| 71 | K4 field contribution: deletion / natural-miss recovery | ours 0.964 / 0.789; 0.624 / 0.765 | v6-S deletion 0.85 / 0.88 / 0.85; natural 0.61 / 0.85 / 0.87; uncertainty crops 0.34–0.47 | phase3 §6 | lower on nuScenes than v3's 0.964 |
| 72 | K9 TTC consumer vs uncertainty at equal px | −0.044 [−0.100, +0.008] / +0.012 | +0.004 [−0.031, +0.041] / +0.079 [+0.048, +0.109] / +0.077 [+0.045, +0.117] | phase3 §8 | |
| 73 | K20 NAVSIM probing cost | 50 calls, 14.9 s / frame | unchanged | – | |
| 74 | Network px "recomputed from detector tensors" | 0.100 / 0.074 | v6-S 0.078 / 0.054 / 0.070 | phase3 §10 | |

## 2. Scorecard (reported, never enforced)

| item | verdict | supporting rows |
|---|---|---|
| **S1** at each matched budget, ours outperforms every frame- / region-level baseline + components on ≥ 2 of 3 datasets and is below none | **met at S; not met at M and L** | S: v6-S outperforms every in-class deployable row on all three splits (6 / 4 / 6 rows). M: v6-M beats every deployable row on 0 / 3 splits (AV2: tsp8@S + C1 −0.015 [−0.037, +0.003], tsp2@S + C1 −0.005, whole frame 512 + C1 −0.004; Argoverse-HD: tsp2@S + C1 matches). L: 0 / 3; two rows significantly above v6-L (AV2 tsp8@M + C1 −0.036 [−0.080, −0.001]; Argoverse-HD cor0@M + C1 −0.060 [−0.126, −0.001]). If the true-V oracle router counts as a frame baseline, S fails too (nuScenes −0.154, AV2 −0.125): a presentation decision for the user. |
| **S2** vs whole frame + components: matches at every resolution on all three datasets with ≤ 25% of the px, outperforms on ≥ 1 | **not met** | Argoverse-HD: outperforms at every resolution (K24 + components) and up to 1280 (K28 + C1). AV2: K24 + components below at 1280 (+0.022 [−0.031, +0.070]); K28 + C1 below at every side, significantly at 1280 (−0.055 [−0.113, −0.005]). nuScenes: below at every resolution with intervals 0.2–0.7 wide. |
| **S3** on the Jetson latency-G front in mean and p95 on both timed datasets | **not met** | AV2 (8 logs, mask): v6-S on the front in mean and p95 (G 0.419, 73.9 / 173 ms). nuScenes (20 old DEEP scenes): off the front in both (G 0.096, 29.5 / 78 ms; e.g. whole frame u1280 + T4 + refiner 0.118 at 25.6 / 30 ms, −0.022 [−0.149, +0.068]). |
| **S4** each component (field, support, guard, looming estimate, refiner) significant on ≥ 2 datasets | **not met** | field 3 / 3, support 2 / 3, looming estimate 3 / 3 (C1 vs T4 and C1 vs none); **guard 0 / 3** (−0.012 / +0.005 / +0.002); **refiner 1 / 3** (AV2 only). |
| **S5** ordering holds with a second detector family without re-tuning, and with fine-tuned detectors | **met at budget S where rows exist (caveats)** | v6-S − v5.1-S: YOLO11s +0.009 / +0.015 (matches) / +0.060; fine-tuned verifier +0.012 / +0.010. Every in-class S row is outperformed, but in-class rows exist only for Argoverse-HD YOLO11s and nuScenes ft0 / ft1. Beyond class S, YOLO11s whole frame + C1 is above v6-S on AV2 (640 and 960) and nuScenes (1280, 8.6×). |
| **S6** refiner-seed std of G below half the smallest S1 margin | **met** (dev and test) | largest test std 0.0099 (AV2 v6-M) vs half the smallest S1 margin at S 0.0527, and 0.0113 over every in-class outperform margin. The AV2 S / M / L ordering is not seed-stable (v6-M 0.268–0.288). |

## 3. Every row where ours is not above a baseline

The complete list is final/not_above_all.csv (908 rows; built by scripts/63_k28_consolidate.py, md5 04354246, from the phase CSVs). A pair is listed when ours does not "outperform" the row (lower bound ≤ 0), or, for the Jetson front, when the row is at least as fast and at least as good, or, for streaming AP (no interval), when the row has the higher sAP. Columns: source, benchmark, ours (configuration), row, row kind (baseline, oracle, own configuration / ablation), cost ratio, in-class flag, ours − row with interval, verdict, which side has the higher point estimate.

**Counts** (baselines only; oracle and own-configuration rows are counted separately in not_above_counts.csv):

| source | benchmark | baseline pairs not outperformed | row higher (point) | row significantly higher |
|---|---|---|---|---|
| phase 2 (confirmatory) | AV2 val (ego mask) | 195 | 76 | 13 |
| phase 2 (confirmatory) | Argoverse-HD val | 71 | 8 | 3 |
| phase 2 (confirmatory) | nuScenes new test | 416 | 143 | 28 |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | 47 | 42 | 34 |
| phase 3 K6 Bench2Drive (post-hoc; runs 1/2/3) | Bench2Drive 219 routes | 4 | 2 | 0 |
| phase 3 KITTI shared v6-S (post-hoc) | KITTI test (6 sequences) | 6 | 6 | 4 |
| phase 3 S3 Jetson (post-hoc) | nuScenes, 20 local DEEP scenes (Jetson) | 12 | 12 | 0 |
| phase 3 S5 (ft0, post-hoc) | nuScenes new test | 6 | 0 | 0 |
| phase 3 S5 (ft1, post-hoc) | nuScenes new test | 7 | 0 | 0 |
| phase 3 S5 (y11s, post-hoc) | AV2 val (ego mask) | 17 | 5 | 2 |
| phase 3 S5 (y11s, post-hoc) | Argoverse-HD val | 3 | 0 | 0 |
| phase 3 S5 (y11s, post-hoc) | nuScenes new test | 18 | 7 | 1 |
| phase 3 prior works (post-hoc) | nuScenes new test | 18 | 6 | 0 |
| phase 3 streaming AP (post-hoc) | Argoverse-HD val | 6 | 6 | 0 |

Not counted above: oracle (not deployable) 27; own configuration / ablation 53; previous ours (K7) 2.

Phase 2 in more detail (as phase2/REPORT.md §2 and §8):
- in-class rows not outperformed: 102, of which 9 oracle-router rows and 93 deployable; 13 deployable rows have the higher point estimate, 2 of them significantly (AV2 tsp8@M + C1 vs v6-L; Argoverse-HD cor0@M + C1 vs v6-L);
- over-budget rows not outperformed by v6-L: 133;
- the remaining pairs compare a row with a v6 configuration outside its class (each row is scored against v6-S, v6-M and v6-L in table_main_*_v6.csv).

**Every pair where the row is significantly above ours** (upper bound < 0; 106 pairs):

| source | benchmark | ours | row | kind | cost / v5.1-S | ours − row [95% CI] |
|---|---|---|---|---|---|---|
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-L | oracle frame router (true V, top-k) @ v5.1-L px | oracle (not deployable) | 1.32 | -0.127 [-0.180, -0.080] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-L | oracle frame router (true V, top-k) @ v5.1-S px | oracle (not deployable) | 1.00 | -0.126 [-0.179, -0.080] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-L | oracle frame router (true V, top-k) @ v5.1-S px + components | oracle (not deployable) | 1.00 | -0.104 [-0.156, -0.061] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-L | oracle frame router (true V, top-k) @ v5.1-L px + components | oracle (not deployable) | 1.32 | -0.101 [-0.153, -0.058] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-L | K28 whole frame yolov8s 1280 +C1 | baseline | 7.49 | -0.057 [-0.119, -0.003] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-L | K28 whole frame yolo11m 1280 +C1 | baseline | 17.86 | -0.050 [-0.101, -0.008] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-L | K28 heuristic tsp8@M+C1 | baseline | 2.03 | -0.036 [-0.080, -0.001] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-M | oracle frame router (true V, top-k) @ v5.1-L px | oracle (not deployable) | 1.32 | -0.138 [-0.184, -0.096] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-M | oracle frame router (true V, top-k) @ v5.1-S px | oracle (not deployable) | 1.00 | -0.137 [-0.183, -0.095] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-M | oracle frame router (true V, top-k) @ v5.1-S px + components | oracle (not deployable) | 1.00 | -0.114 [-0.157, -0.075] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-M | oracle frame router (true V, top-k) @ v5.1-L px + components | oracle (not deployable) | 1.32 | -0.112 [-0.157, -0.071] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-M | K28 whole frame yolov8s 1280 +C1 | baseline | 7.49 | -0.067 [-0.128, -0.012] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-M | K28 whole frame yolo11m 1280 +C1 | baseline | 17.86 | -0.061 [-0.112, -0.011] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-M | K28 heuristic tsp8@L+C1 | baseline | 2.18 | -0.060 [-0.124, -0.005] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-M | K28 whole frame yolov8s 960 +C1 | baseline | 4.35 | -0.057 [-0.117, -0.003] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-M | K28 whole frame yolo11m 896 +C1 | baseline | 8.87 | -0.056 [-0.113, -0.002] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-M | K28 heuristic tsp8@M+C1 | baseline | 2.03 | -0.047 [-0.090, -0.010] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-M | K28 heuristic tcr3@L+C1 | baseline | 3.58 | -0.043 [-0.086, -0.004] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-S | oracle frame router (true V, top-k) @ v5.1-L px | oracle (not deployable) | 1.32 | -0.125 [-0.174, -0.080] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-S | oracle frame router (true V, top-k) @ v5.1-S px | oracle (not deployable) | 1.00 | -0.124 [-0.173, -0.079] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-S | oracle frame router (true V, top-k) @ v5.1-S px + components | oracle (not deployable) | 1.00 | -0.102 [-0.146, -0.062] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-S | oracle frame router (true V, top-k) @ v5.1-L px + components | oracle (not deployable) | 1.32 | -0.099 [-0.145, -0.058] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-S | K28 whole frame yolov8s 1280 +C1 | baseline | 7.49 | -0.055 [-0.113, -0.005] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-S | K28 whole frame yolo11m 1280 +C1 | baseline | 17.86 | -0.048 [-0.098, -0.004] |
| phase 2 (confirmatory) | AV2 val (ego mask) | v6-S | K28 heuristic tsp8@M+C1 | baseline | 2.03 | -0.034 [-0.074, -0.004] |
| phase 2 (confirmatory) | Argoverse-HD val | v6-L | K28 heuristic cor0@M+C1 | baseline | 2.11 | -0.060 [-0.126, -0.001] |
| phase 2 (confirmatory) | Argoverse-HD val | v6-M | K28 heuristic cor0@M+C1 | baseline | 2.11 | -0.063 [-0.124, -0.010] |
| phase 2 (confirmatory) | Argoverse-HD val | v6-S | K28 heuristic cor0@M+C1 | baseline | 2.11 | -0.073 [-0.132, -0.019] |
| phase 2 (confirmatory) | nuScenes new test | v6-L | K28 whole frame yolov8m 1600 +C1 | baseline | 36.69 | -0.069 [-0.118, -0.025] |
| phase 2 (confirmatory) | nuScenes new test | v6-L | K28 whole frame yolov8m 1280 +C1 | baseline | 23.28 | -0.063 [-0.110, -0.017] |
| phase 2 (confirmatory) | nuScenes new test | v6-L | K28 whole frame yolo11m 960 +C1 | baseline | 11.83 | -0.055 [-0.113, -0.000] |
| phase 2 (confirmatory) | nuScenes new test | v6-L | K28 whole frame yolov8m 960 +C1 | baseline | 13.66 | -0.055 [-0.106, -0.004] |
| phase 2 (confirmatory) | nuScenes new test | v6-L | K28 whole frame yolo11m 768 +C1 | baseline | 7.36 | -0.054 [-0.102, -0.006] |
| phase 2 (confirmatory) | nuScenes new test | v6-L | K28 whole frame yolov8m 1280 +T4 | baseline | 23.28 | -0.048 [-0.095, -0.004] |
| phase 2 (confirmatory) | nuScenes new test | v6-L | K28 whole frame yolo11m 640 +C1 | baseline | 5.26 | -0.047 [-0.090, -0.001] |
| phase 2 (confirmatory) | nuScenes new test | v6-L | K28 whole frame yolov8m 1600 +T4 | baseline | 36.69 | -0.046 [-0.090, -0.004] |
| phase 2 (confirmatory) | nuScenes new test | v6-M | K28 whole frame yolov8m 1600 +C1 | baseline | 36.69 | -0.086 [-0.135, -0.029] |
| phase 2 (confirmatory) | nuScenes new test | v6-M | K28 whole frame yolov8m 1280 +C1 | baseline | 23.28 | -0.080 [-0.128, -0.023] |
| phase 2 (confirmatory) | nuScenes new test | v6-M | K28 whole frame yolo11m 960 +C1 | baseline | 11.83 | -0.071 [-0.134, -0.001] |
| phase 2 (confirmatory) | nuScenes new test | v6-M | K28 whole frame yolov8m 960 +C1 | baseline | 13.66 | -0.071 [-0.123, -0.011] |
| phase 2 (confirmatory) | nuScenes new test | v6-M | K28 whole frame yolo11m 768 +C1 | baseline | 7.36 | -0.071 [-0.120, -0.010] |
| phase 2 (confirmatory) | nuScenes new test | v6-M | K28 whole frame yolov8m 1280 +T4 | baseline | 23.28 | -0.065 [-0.112, -0.011] |
| phase 2 (confirmatory) | nuScenes new test | v6-M | K28 whole frame yolo11m 640 +C1 | baseline | 5.26 | -0.064 [-0.111, -0.005] |
| phase 2 (confirmatory) | nuScenes new test | v6-M | K28 whole frame yolov8m 1600 +T4 | baseline | 36.69 | -0.062 [-0.109, -0.009] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | K28 whole frame yolov8m 1600 +C1 | baseline | 36.69 | -0.173 [-0.325, -0.035] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | K28 whole frame yolov8m 1280 +C1 | baseline | 23.28 | -0.167 [-0.318, -0.026] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | K28 whole frame yolo11m 960 +C1 | baseline | 11.83 | -0.159 [-0.313, -0.013] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | K28 whole frame yolov8m 960 +C1 | baseline | 13.66 | -0.159 [-0.312, -0.013] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | K28 whole frame yolo11m 768 +C1 | baseline | 7.36 | -0.158 [-0.312, -0.014] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | oracle frame router (true V, top-k) @ v5.1-S px | oracle (not deployable) | 1.00 | -0.154 [-0.231, -0.073] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | oracle frame router (true V, top-k) @ v5.1-L px | oracle (not deployable) | 1.25 | -0.154 [-0.231, -0.073] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | K28 whole frame yolov8m 1280 +T4 | baseline | 23.28 | -0.152 [-0.311, -0.005] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | K28 whole frame yolo11m 640 +C1 | baseline | 5.26 | -0.151 [-0.306, -0.006] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | K28 whole frame yolov8m 1600 +T4 | baseline | 36.69 | -0.150 [-0.311, -0.005] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | K28 whole frame yolo11m 1280 +C1 | baseline | 20.15 | -0.140 [-0.302, -0.009] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | oracle frame router (true V, top-k) @ v5.1-S px + components | oracle (not deployable) | 1.00 | -0.138 [-0.211, -0.057] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | oracle frame router (true V, top-k) @ v5.1-L px + components | oracle (not deployable) | 1.25 | -0.138 [-0.211, -0.057] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | K28 whole frame yolo11m 1600 +C1 | baseline | 31.75 | -0.117 [-0.257, -0.004] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | K28 whole frame yolov8m 640 +C1 | baseline | 6.07 | -0.114 [-0.206, -0.014] |
| phase 2 (confirmatory) | nuScenes new test | v6-S | K28 whole frame yolo11m 448 +C1 | baseline | 2.45 | -0.107 [-0.205, -0.002] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | wf2048+C1 | baseline | 13.63 | -0.151 [-0.201, -0.102] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | wf1536+C1 | baseline | 7.72 | -0.127 [-0.164, -0.089] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | wf2048+T4 | baseline | 13.63 | -0.124 [-0.174, -0.073] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | wf2048 | baseline | 13.63 | -0.097 [-0.147, -0.048] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | tsp8@L+C1 | baseline | 1.79 | -0.097 [-0.131, -0.064] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | tsp8@M+C1 | baseline | 1.73 | -0.092 [-0.122, -0.064] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | tsp0@L+C1 | baseline | 1.96 | -0.091 [-0.122, -0.061] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | wf1536+T4 | baseline | 7.72 | -0.090 [-0.127, -0.055] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | tsp4@M+C1 | baseline | 1.61 | -0.082 [-0.107, -0.057] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | tsp8@M+T4 | baseline | 1.73 | -0.067 [-0.098, -0.036] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | wf1536 | baseline | 7.72 | -0.066 [-0.103, -0.030] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | tsp0@L+T4 | baseline | 1.96 | -0.066 [-0.097, -0.034] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | tsp8@L+T4 | baseline | 1.79 | -0.065 [-0.096, -0.034] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | unc2@L+C1 | baseline | 2.13 | -0.065 [-0.097, -0.032] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | tsp4@M+T4 | baseline | 1.61 | -0.064 [-0.092, -0.036] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | unc2@L+T4 | baseline | 2.13 | -0.054 [-0.086, -0.020] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | unc2@M+C1 | baseline | 1.87 | -0.053 [-0.086, -0.020] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | tsp4@M | baseline | 1.61 | -0.047 [-0.079, -0.016] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | wf1280+C1 | baseline | 5.39 | -0.045 [-0.078, -0.014] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | unc2@M+T4 | baseline | 1.87 | -0.043 [-0.077, -0.010] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | unc6@L+C1 | baseline | 2.58 | -0.041 [-0.077, -0.007] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | unc2@L | baseline | 2.13 | -0.038 [-0.072, -0.003] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-L | tsp8@M | baseline | 1.73 | -0.033 [-0.065, -0.001] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-M | tsp8@S+C1 | baseline | 1.20 | -0.313 [-0.366, -0.263] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-M | tsp2@S+C1 | baseline | 1.14 | -0.310 [-0.362, -0.260] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-M | tsp8@S+T4 | baseline | 1.20 | -0.306 [-0.359, -0.254] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-M | tsp2@S+T4 | baseline | 1.14 | -0.304 [-0.357, -0.253] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-M | tsp8@S | baseline | 1.20 | -0.280 [-0.333, -0.225] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-M | tsp2@S | baseline | 1.14 | -0.278 [-0.333, -0.222] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-M | unc2@S+C1 | baseline | 1.45 | -0.172 [-0.221, -0.123] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-M | unc2@S+T4 | baseline | 1.45 | -0.170 [-0.219, -0.120] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-M | unc2@S | baseline | 1.45 | -0.154 [-0.204, -0.102] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-M | unc4@S+C1 | baseline | 1.45 | -0.051 [-0.098, -0.005] |
| phase 3 AV2 without ego mask (post-hoc) | AV2 val (no ego mask) | v6-M | unc4@S+T4 | baseline | 1.45 | -0.051 [-0.099, -0.004] |
| phase 3 K6 Bench2Drive (post-hoc; runs 1/2/3) | Bench2Drive 219 routes | v6-S | privileged | oracle (not deployable) | – | -4.088 [-6.364, -2.022] |
| phase 3 KITTI shared v6-S (post-hoc) | KITTI test (6 sequences) | v6-S | native+comp | baseline | – | -0.134 [-0.182, -0.030] |
| phase 3 KITTI shared v6-S (post-hoc) | KITTI test (6 sequences) | v6-S | native+c1comp | baseline | – | -0.119 [-0.183, -0.025] |
| phase 3 KITTI shared v6-S (post-hoc) | KITTI test (6 sequences) | v6-S | u960+comp | baseline | – | -0.058 [-0.105, -0.012] |
| phase 3 KITTI shared v6-S (post-hoc) | KITTI test (6 sequences) | v6-S | u960+c1comp | baseline | – | -0.048 [-0.094, -0.021] |
| phase 3 S3 Jetson (post-hoc) | nuScenes, 20 local DEEP scenes (Jetson) | v6-M | v6-L, rounds <= 2, no refiner | own configuration / ablation | – | -0.036 [-0.121, -0.010] |
| phase 3 S3 Jetson (post-hoc) | nuScenes, 20 local DEEP scenes (Jetson) | v6-S | v6-L, rounds <= 2, no refiner | own configuration / ablation | – | -0.039 [-0.130, -0.000] |
| phase 3 S3 Jetson (post-hoc) | nuScenes, 20 local DEEP scenes (Jetson) | v6-S | v6-S, rounds <= 2, no refiner | own configuration / ablation | – | -0.030 [-0.081, -0.011] |
| phase 3 S3 Jetson (post-hoc) | nuScenes, 20 local DEEP scenes (Jetson) | v6-S | v6-S, rounds <= 3, no refiner | own configuration / ablation | – | -0.022 [-0.081, -0.002] |
| phase 3 S5 (y11s, post-hoc) | AV2 val (ego mask) | v6-S | wf960+C1 | baseline | – | -0.062 [-0.123, -0.005] |
| phase 3 S5 (y11s, post-hoc) | AV2 val (ego mask) | v6-S | wf640+C1 | baseline | – | -0.056 [-0.113, -0.005] |
| phase 3 S5 (y11s, post-hoc) | nuScenes new test | v6-S | wf1280+C1 | baseline | – | -0.064 [-0.114, -0.015] |

**Not-above results without a G interval** (secondary metrics; values in the phase-3 sections named):
- detection AP, Argoverse-HD: whole frame YOLOv8s 512 0.182 vs v6-S 0.165 at the same GFLOPs; every larger whole frame is higher (§11);
- nuScenes AP50 and recall: every whole-frame row is above v6-S (v6-S has the highest relevant precision) (§11);
- missed decelerations per 1k: v6-S misses more than u960 + components on all three splits, significantly (§1c row 62);
- sign tests (units where v6-S / the row has the lower loss): whole frame native + components 45 / 58 scenes on nuScenes (p 0.24) and 15 / 7 sequences on Argoverse-HD (p 0.13), both n.s.; the oracle router + components 60 / 76 AV2 logs (p 0.20) (§12);
- no-support regime: u960 + C1 has the higher regime G on nuScenes (0.409 vs 0.331) and AV2 (0.414 vs 0.289) (§7);
- agreement with π(u960 + C1): a tie with the best heuristic + C1 on every split (§5);
- missed objects: whole frame u960 recovers 99–100% (by construction for natural misses); v6-S 61–88% (§6);
- the TTC-warning consumer on nuScenes: n.s. against uncertainty crops at equal px (§8);
- NAVSIM: every non-oracle baseline ties spec_pdm (kit §3; unchanged).

## 4. Every test evaluation made in the project

final/test_evaluations.csv (28 rows; K28 rows added to the draft of 2026-10-13). The K28 rows:

| date | tz | evaluation | data | configurations | type |
|---|---|---|---|---|---|
| 2026-10-11 | +07 | K28 phase 0 | K24 test units (all three) | v5.1-S: three ablation rows (pivots, single round, without guard) re-run; cost, prior-work intervals, loss terms from stored results | post-hoc |
| 2026-10-11 23:26-23:51 / 2026-10-12 07:04-07:28 | +07 | K28 phase 1, Jetson timing windows | nuScenes 20 old DEEP scenes (797 timed frames; not the new test) and AV2 val, 8 local logs with the ego mask (586 frames; test units) | functional smoke of 60_k28_latency.py (100 nuScenes frames), then B candidates: ours with YOLO11m / YOLOv8m / YOLOv8l / YOLOv8s on the crops (+ v3a, T4) and same-detector whole frame plain / + components; G was computed next to the timing | timing for phase 1 (reported, not a selection input; the round-1 decision used dev G only, §2k) |
| 2026-10-13 00:36 (VUW 06:36) - VUW 16:32 | +07 | K28 phase 2 (CONFIRMATORY) | nuScenes new test (228) / AV2 val with ego mask (150) / Argoverse-HD val (24) | v6-S / v6-M / v6-L, v5.1-S (parity), YOLOv8s / YOLO11m / YOLOv8m whole frame and heuristics plain / + T4 / + C1; prior works and routers reused from K24 (no re-run) | the ONE confirmatory pass of K28 (pre-registered §2n) |
| 2026-10-13 02:27 (VUW 08:27) | +07 | K28 phase 2 addendum §2n.1 (same pass) | same three test splits | support only / field only of v5.1-S and v6-S (T4 / C1); routers and the true-V oracle re-matched to v6-S / M / L (analysis of stored router results) | part of the single confirmatory pass (written before any test value was seen) |
| VUW 2026-10-13 16:36 - 2026-10-14 05:59 | NZT | K28 phase 3 VUW runs | nuScenes new test / AV2 val (ego mask; and without the mask) / Argoverse-HD val | instrumented v6-S (K2 agreement, K4 natural misses and deletion, K9 TTC consumer, K16 chain, K17 regime, failures), K3 rows of v6-S (pivots, one round, no guard), YOLO11s draft + verifier without re-tuning, fine-tuned verifier seeds 0 / 1 (nuScenes), AV2 without the ego mask (all rows) | post-hoc (v6 unchanged) |
| VUW 2026-10-13 18:49 / 19:28 | NZT | K28 phase 3 AP dump and scoring | Argoverse-HD val (all 15,062 frames) / nuScenes new test (every 4th frame) | v5.1-S, v6-S / M / L and whole frame YOLOv8s / YOLO11m / YOLOv8m: COCO AP (3 road-user groups) and det_metrics | post-hoc |
| VUW 2026-10-14 06:32 - 08:23 | NZT | K28 phase 3 capture run (panels) | 30 frames of nuScenes new test / AV2 val (mask) / Argoverse-HD val (11 / 12 / 3 units) | v6-S instrumented with per-frame capture; rendered by k28_render.py 8b496bf3 | post-hoc (qualitative) |
| 2026-10-14 (VUW, done 11:05) | NZT | K28 phase 3 S6 test seeds | nuScenes new test / AV2 val (mask) / Argoverse-HD val | v6-S / M / L scored with refiner seeds 0 / 1 / 2 (seed 0 = phase 2) | post-hoc |
| 2026-10-14 00:48-07:28 | +07 | K28 final Jetson window | nuScenes 20 old DEEP scenes (797 frames; not the new test) / AV2 val 8 local logs with mask (586 frames; test units) / KITTI test 6 seq (3,336 frames) / Argoverse-HD val 24 seq (15,062 frames) | v6-S / M / L (with / without refiner, rounds <= 2 / 3), v5.1-S, cheap, whole frame plain / + T4 + v3a / + C1 + v3a; KITTI shared v6-S configuration (4 cells); Argoverse-HD per-frame timing | post-hoc |
| 2026-10-14 07:35 | +07 | K28 streaming AP | Argoverse-HD val (24 seq) | v6-S detections of the phase-2 AP dump + Jetson per-frame latency of the final window (K7 simulation) | post-hoc |
| 2026-10-14 - 2026-10-15 09:38 | NZT | K28 K6 Bench2Drive v6-S | Bench2Drive 220 routes (route 11755 excluded as in K6), 3 runs | PDM-Lite with v6-S perception in the loop (branch ours_v6, perception_pilot.py 23539df7); K6 branches reused | post-hoc |

**Disclosure.** The phase-1 Jetson window of 2026-10-12 computed G on 8 AV2 val logs (test units) for the B candidates while timing them. It was reported only; the round-1 decision used development G (§2k). It is listed above for R5.

## 5. Corrections and fix notes

- **Phase-2 wording (corrected 2026-10-14).** phase2/REPORT.md §2 said the Argoverse-HD oracle router is "below v6 at every budget". At S it is "matches": v6-S − oracle +0.049 [−0.002, +0.101] (and +0.062 [+0.014, +0.112] for oracle + components). At M and L v6 outperforms it. phase2/REPORT.md is corrected in place; the scorecard S1 entry was already right (§3 table: "matches, outperforms").
- **Analysis-code fixes after the test pass (no result file changed):**
  - k28_analyze_test.py d0dd013a → ca1c34b4 and k28_analyze_test2.py 26468b2b → 725e481e: `KeyError: 'J_cheap_ours'` (J_cheap listed twice); re-run once on the unchanged result files.
  - k28_p3_analyze.py a208656c → bd7035fd (same latent bug, fixed before use); k28_p3_p0.py d3f19747 → 5d726b33 (merge fix).
  - 61_k28_v6_latency.py 8c2a2223 → 6341d349: the Argoverse-HD summary read `dd.gt` (the pandas method) instead of `dd["gt"]`; recomputed from the saved per-frame pickle (`--post_only`); an AV2 run recomputed the same way reproduces its CSVs exactly.
- **Operational events (no code change):** phase-2 shards killed by the host's out-of-memory event at VUW 07:42 (2026-10-13) were re-run with the same commands; finished units were kept. The phase-1 Jetson reboots (YOLOv8m 448×768 builds) and the final-window resets (01:40 YOLOv8s 864×1536 build, 04:35 during a run) are in phase3 §16.
- **K6 route 11755.** The first v6 Bench2Drive launcher still ran route 11755, which hangs the simulator for every branch except cheap and which K6 excludes. The run was stopped (exact PIDs) and continued with K6's protocol (chunk0b); no finished route was lost (phase3 §14).

## 6. Open items and decisions for the user

- K6 Bench2Drive with v6-S: DONE 2026-10-15 (three runs; §1c row 48, phase3 §14a). Whether the paper's K6 row reads v6-S or keeps v3 is a presentation choice; the two tie (−0.26 [−1.23, +0.61]).
- **KITTI discrepancy (phase3 §16): cause found 2026-10-15, re-run pending.**
  - The Jetson reference refiner (58.RefinerBatch) fed refiner v3a the nuScenes / AV2 camera flag (1.0) on KITTI; v3a was trained with 0.0 for KITTI.
  - On KITTI val boxes, the flag raises v3a's |log range error| from 0.055 to 0.084 and leaves the range bias mostly uncorrected.
  - Every "+ refiner" row of the KITTI shared-configuration table is affected; the other datasets are not.
  - **Decision for the user:** re-run that table on the Jetson with the flag set per dataset once the power supply is fixed (≈ 20 min), or drop the KITTI shared-configuration row.
- Presentation decisions, not made here: whether the true-V oracle router counts as a "frame baseline" in S1; whether the paper's AV2 tables use the masked protocol only (K10 says yes); whether the fine-tuned AV2 rows, LZU / UFP / ZoomDet / ESOD (no v6 interval) and the v3 / v4-lite rows of kit §1 stay, labelled with their configuration.

## Files (handoff/K28/final/)

- not_above_all.csv, not_above_counts.csv: item (iii); scripts/63_k28_consolidate.py.
- test_evaluations.csv: item (iv) (test_evaluations_draft.csv is the 2026-10-13 draft).
- The tables behind items (i) and (ii) are in the phase folders named in each row.
