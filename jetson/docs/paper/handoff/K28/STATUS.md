# K28 STATUS

**Phase:** 0 DONE; phase 1 DONE (rounds 1–3, §2k–§2m; v6-S = C1, v6-M = YOLOv8m + C1 crops 768, v6-L = YOLO11m + C1); §2m follow-ups DONE; v6 FROZEN (§2n); phase 2 DONE 2026-10-13 (phase2/REPORT.md; single test pass launched 00:36 +07 / VUW 06:36; §2n.1 ablation + re-matched routers launched 02:27 +07); phase 3 VUW part DONE 2026-10-14 (phase3/REPORT.md); panels (capture) running; final Jetson window + consolidated REPORT pending. Started 2026-10-11 19:20 (+07; VUW 2026-10-12 01:20).
**Rules in force:**
- R1–R6 of the task, pre-registered in method_freeze_draft §2j.
- Phase-1 hard stop on 2026-10-18.

**Notes on the rules**
- The "K25 (bản cập nhật)" text with items (1)–(11) was never pasted into this session; only the original K25 is here. Phases 0 and 3 follow the K28 task's own enumeration of those items:
  - phase 0: (1) the three missing ablation rows, (2) cost statistics, (4) intervals vs AdaScale / FOVEA / SAHI, (5) loss terms / event rates / sign tests;
  - phase 3: the remaining list.
- KITTI images are not on VUW. The B refiners therefore keep the KITTI part of the v3a training data with YOLOv8s boxes. This deviation is disclosed in §2j.
- Argoverse-HD has no stored AdaScale / FOVEA / SAHI results, so phase 0 item (4) covers nuScenes and AV2 only.

## Running (2026-10-12 03:20 +07 / VUW 09:20)
| What | Where | State | Expected |
|---|---|---|---|
| Round 1, family A + v5.1-S reference | VUW | DONE (nuScenes 152 / 152, AV2 300 / 300) | — |
| Round 1, family E (baseline grids) | VUW `k28_r1_chain.sh` | nuScenes 114 / 152, AV2 79 / 300 | nuScenes ~2 h; AV2 ~20 h more (needed before phase 2 only) |
| Family B dev | VUW `k28_bdev_chain.sh` | YOLOv8m and YOLO11m DONE (152 / 300). YOLOv8l: AV2 250 / 300, nuScenes 0 / 152 (its first group died at start-up with CUDA OOM; re-run `k28_bdev_fix.sh`, md5 c05f09fa, from 08:58 VUW) | ~2.5 h |
| Family C | VUW | data DONE 06:12; head DONE 09:00 (md5 7bfdecec; val MAE 0.065 vs T4 0.099, zero predictor 0.215); dev evaluation `k28_tdev.py` started 09:05 | ~4 h |
| Family D for the L candidate (YOLO11m refiner, seeds 1 / 2) | VUW `k28_dseeds_chain.sh` (md5 cc68c858) | queued for a GPU | ~1 h |
| Jetson window (DEEP informed 03:09) | Jetson | building YOLO11m whole-frame engines 256×448 / 384×640 and YOLOv8m 448×768 (workspace 1024), then the B latency runs | ~2 h |

## Preliminary round-1 B result (dev; YOLOv8l nuScenes still missing; NOT yet the round decision)
| | nuScenes G (own refiner) | (B − v5.1-S) [95% CI] | AV2 G | (B − v5.1-S) [95% CI] | GFLOPs ratio | class |
|---|---|---|---|---|---|---|
| v5.1-S | 0.302 | — | 0.272 | — | 1.00 | S |
| YOLO11m on crops | 0.409 | +0.107 [−0.003, +0.220] | 0.293 | **+0.022 [+0.008, +0.037]** | 1.78 | L |
| YOLOv8m on crops | 0.384 | +0.082 [−0.013, +0.183] | 0.285 | **+0.013 [+0.001, +0.028]** | 1.97 | L |
| YOLOv8l on crops | (running) | | 0.285 (252 / 300 logs) | +0.021 [+0.005, +0.039] | 3.65 | over |

Both YOLO11m and YOLOv8m pass the §2j rule; YOLO11m has the higher mean dev G → provisional L choice.
- Same-detector whole frame (`bwholeframe_r1.csv`, + components, own refiner):
  - YOLO11m, nuScenes: 0.337 at 448 (2.4× GFLOPs) and 0.413 at 768 (7.1×);
  - YOLO11m, AV2: ≤ 0.279 at every long side (≥ 2.9×).

## Next steps
0. **Deferred to the very end (user, 2026-10-12):** any further Jetson TensorRT engine builds, e.g. YOLO11m whole frame 448×768, or an AV2 whole-frame set for 11m. They run in one dedicated window after everything else, with nothing else running on the board.
   - The YOLOv8m 448×768 build rebooted the board twice (2026-10-11 ~20:39, 2026-10-12 ~03:44). It is not retried.
   - Until then, the latency runs use only the engines that exist.
1. When B / C dev results exist: k28_analyze.py (round-1 rule of §2j), then the round-1 amendment (adopted / rejected).
2. Round 2: combinations of the passing candidates, plus D (seeds of the selected refiner), with a written amendment first.
3. Phase 2 only if something is adopted. Phase 3 after that.

## Log
- 2026-10-11 19:20 — K28 received. §2j (round-1 pre-registration) written before any K28 run. QUEUE row added; K25 marked superseded by K28.
- 2026-10-11 — DEEP ladder reruns: route_11755 privileged crashed a 3rd time (CARLA segfault, deterministic). It is stopped per DEEP's rule; cheap / full are still running.
- 2026-10-11 19:30 — round-1 dev chain (A + E), B-data chain and phase-0 chain launched. Smoke parity: v5.1-S / T3 in k28_dev = K27 (100%). The heuristic px exceed the per-frame budget because of guard re-detections, as in K18 / K24 (reported as realised px).
- 2026-10-11 19:35 — VUW's wrapper serialises launches (~3 min each), so shard processes are now grouped 3 per wrapper call (memory note).
- 2026-10-11 20:25 — phase 0 DONE → phase0/REPORT.md.
  - The re-run of v5.1-S reproduces K24 on 100% of frames.
  - The cost-table merge filter was fixed and re-run before the table was read.
  - Not-above rows are listed in phase0/REPORT.md. Among them: nuScenes SAHI is n.s., and nuScenes pivots / no guard / one round are n.s. With v3a, AV2 single round and no guard are n.s.
- 2026-10-11 23:20 — session restart.
  - The Jetson rebooted at ~20:39 during a TensorRT build. The pause file was removed at 23:19. The DEEP session was not running, so it could not be messaged; it will be told when it is back.
  - The C smoke test passed (231 nuScenes + 45 AV2 samples). The smoke files were moved out of the training glob (data/smoke_k28/). The full C chain and the B dev chain were launched.
- 2026-10-11 23:30 — round-1 family A analysed (dev): none of the 11 passes the §2j rule.
  - The α × speed v_close variants are identical to v5.1-S by construction: with v_close ≤ ego speed, the probe's closing-speed demand never exceeds its static term (dsf._insert_demand).
  - Closest: support context 2.5, nuScenes +0.033 [−0.003, +0.075], AV2 +0.004 [−0.003, +0.012].
  - v_close 5 / 10 are significantly worse on AV2.
  - Tables: VUW results/k28/tables/candidates_r1.csv / selection_r1.csv (partial until B / C are in).
- 2026-10-11 23:26–23:51 — Jetson window, with DEEP informed at start and release.
  - Built the v3a refiner and YOLO11m dyn engines.
  - Functional test of scripts/60_k28_latency.py on 100 nuScenes frames: ours mean 17.0 ms / p95 39.7 ms; u960 18.1 / 18.8; native 33.6 / 34.8; bounded N2 15.7 / 36.0. Smoke only.
- 2026-10-12 02:56 — B dev DONE except the YOLOv8l group g0 (CUDA OOM at start-up: the GPU was full). It was re-launched with MIN_FREE_MB 9000; finished units are skipped.
- 2026-10-12 03:00 — C head trained: best epoch 12, val MAE 0.065 (AV2 0.063, nuScenes 0.076) vs T4 0.099 (0.094 / 0.122).
  - Disclosed deviation from the §2j text: the C training data also include nuScenes new TRAIN scenes (development data; a hold-out of them is in the validation split), not AV2 REF-TRAIN alone.
- 2026-10-12 03:05 — preliminary round-1 analysis.
  - B: YOLO11m and YOLOv8m pass, both in class L.
  - k28_analyze.py gained the B same-detector whole-frame table (bwholeframe_<tag>.csv; reporting only; md5 3abc9588).
  - The D seeds of the YOLO11m refiner were launched (training only).
- 2026-10-12 03:09 — Jetson window opened (pause set; DEEP informed and acknowledged).
  - scripts/60_k28_latency.py: nuScenes whole-frame sizes u448 / u768 were added (only when requested with --wf). The default set is unchanged.
- 2026-10-12 06:57 — session restart. The Jetson had rebooted at ~03:44 during the YOLOv8m 448×768 build, the second reboot on the same engine; it is not retried. The YOLO11m 256×448 / 384×640 engines were built.
- 2026-10-12 07:00 — **round 1 complete (dev)** (k28_analyze.py --tag r1):
  - adopted S = M = C1 (learned temporal cue replacing T4) and L = YOLO11m on the crops with its own refiner;
  - C2, YOLOv8m and YOLOv8l pass but are not chosen (YOLOv8l is over budget);
  - no A candidate passes.

  Full table and reasons in method_freeze_draft §2k.
  - R4 note: u960 + C1 on AV2 dev (0.328) is above ours + C1 (0.309). Every baseline receives C1 in round 2 / phase 2.
- 2026-10-12 07:04–07:28 — Jetson window (DEEP session risk-aware-perception-ae informed / released): B latency, 797 nuScenes frames (phase1/latency_B_jetson.csv). Ours with YOLO11m: mean 28.6 ms, p95 71 ms. Whole frame YOLO11m + components: 14.4 ms at 448, 20.4 ms at 640. Ours with YOLOv8s: 22.9 ms. **Ours is slower than same-detector whole frame at 640 on the mean**, because of the multi-round tail. AV2 ours: 56.5 ms with v8s, 66.6 ms with 11m.
- 2026-10-12 07:10 — the E nuScenes base had 114 / 152 units: two shard processes ended without an error. Re-run with k28_efix.sh (finished units are skipped).
- 2026-10-12 07:40 — §2k written: the round-1 decision and the round-2 pre-registration.
  - Round-2 candidates: YOLO11m + C1, YOLO11m + C2, and YOLO11m + C1 at crop sides 768 / 640.
  - Family D seeds.
  - R4: baselines receive C1 (whole frame at all long sides; the E dev-best heuristic configurations and K24 forms); YOLO11m heuristics at the S budget.
  - Code: k28_r2dev.py, k28_analyze_r2.py, k28_r2_chain.sh. A smoke run (one scene) is queued; it is checked for parity only.
- 2026-10-12 07:45 — label bug fixed before use: k28_analyze.py marked cor2 / tsp2 as the K24 corridor forms, but K24's tube is near 10 m / far 70 m / widest (TUBE index 8 → cor8 / tsp8). Only the K24_form flag in baselines_r1.csv was affected, not the dev-best choice.

- 2026-10-12 08:50–11:28 — round 2 progress. Ours parts DONE (nuScenes 152, AV2 300); whole frame + C1 DONE on nuScenes (152), AV2 190 / 300. Family E DONE: nuScenes re-run 152 / 152 at VUW 14:55; AV2 300 at 16:48 (r1_done.log). k28_analyze --tag r1 re-run on the complete E data: the round-1 choice is unchanged (S = M = C1, L = ver-yolo11m).
  - Heuristic + C1 configurations taken from baselines_r1.csv (dev_best ∪ K24_form):
    - YOLOv8s, 26: unc2 / unc4 / tcr0 / tcr3 / cor0 / cor8 / tsp2 / tsp8 / rnd @S; unc2 / unc8 / tcr0 / tcr3 / cor0 / cor8 / tsp4 / tsp8 / rnd @M; unc2 / unc6 / tcr3 / cor0 / cor8 / tsp0 / tsp8 / rnd @L.
    - YOLO11m, the 9 @S configurations.
  - The heuristic groups launched at VUW 17:04–17:13. The slowest part is AV2 YOLOv8s (~0.6 units / min). Expected finish: VUW ~01:00–02:00 (≈ 19:00–20:00 +07); `k28_analyze_r2.py` then runs automatically.
- 2026-10-12 15:40–16:05 — PRELIMINARY round-2 analysis ($B/tmp/k28_r2prelim/, k28_analyze_r2.main with OUT redirected). The ours and whole-frame parts are complete, so the round-2 selection is final; the AV2 heuristic parts are partial (YOLOv8s 105 / 300, YOLO11m 250 / 300). The chain's own run writes the official tables.
  - Parity: 100% identical on every frame (nuScenes 6,117; AV2 23,665) for v51S, C1, B11, px, wf960 + T4, and wf960 + C1 = u960 + C1.
  - Rule result: S = C1 (unchanged); **M = R2-4 YOLO11m + C1, crops 640** (0.396 / 0.306, GFLOPs 1.09×; R2-3 crops 768 mean G 0.3510 vs 0.3512); **L = R2-1 YOLO11m + C1** (0.408 / 0.311, 1.78×). Round 2 adopts at M and L.
  - D seeds: std ≤ 0.006 for every row.
  - R4 baselines with C1, not selection inputs:
    - nuScenes: ours is ahead at S / M. At L, the YOLO11m heuristic tsp2@S + C1 reaches 0.414 vs ours 0.408 (≈ 2.3× GFLOPs).
    - AV2: whole frame + C1 reaches 0.316 at 512 (1.21×) and 0.322 at 640 (1.86×), vs ours M 0.306 / L 0.311.
    - AV2 heuristics + C1 (partial): cor0@M +0.080 and cor0@L +0.106 vs v5.1-S on the same units, vs ours M +0.035 / L +0.039.
- 2026-10-12 16:25 — §2l written: the round-2 decision and the round-3 pre-registration.
  - Round-2 decision: M = R2-4, L = R2-1.
  - Round 3: YOLOv8m + C1 (crops 960 / 768 / 640) and YOLOv8l + C1 (crops 768 / 640). C2 is excluded because it was below C1 in both rounds.
  - Phase-2 direction from the user: published prior works run with their official code and settings (K24), without C1. Whole frame and the heuristics keep C1 (R4).
  - Phase 3 adds AP vs px and streaming AP for v6.
  - Code: k28_r3dev.py (f5f74b28), k28_analyze_r3.py (a56758cd), k28_r3_chain.sh (completeness re-run built in).
  - A one-scene smoke run is queued, for parity only.
- 2026-10-12 17:15 — phase-2 code ready, not run on test.
  - k28_test.py (c85eeba9) builds on k28_r2dev with the K24 test units, including Argoverse-HD, and V6 = the round-2 choices (to be re-frozen in §2m after round 3).
  - k28_analyze_test.py builds on k24_analyze, with the K24 rows and routers unchanged plus the K28 rows. It refuses to write tables before every part is complete; checked on VUW: "NOT READY".
  - Dev smoke (nuScenes scene-0108, `--split dev`): identical on 100% of frames for every parity column:
    - v6S / v6M / v6L = r2 C1 / B11C1s640 / B11C1;
    - wf960 plain / + T4 = E;
    - wf960 + C1 and the 6 YOLO11m whole-frame + C1 rows = r2;
    - YOLO11m whole frame plain / + T4 = r1;
    - the 5 K24-form heuristics plain / + T4 (J and px) = E, and + C1 = r2;
    - YOLO11m heuristics + T4 / + C1 = r2.
- 2026-10-12 20:10 — round 3 done on dev; preliminary analysis (tmp/k28_r3prelim, parity 100%); §2m written.
  - Mean dev G (own seed-0 refiner): R3-1 B8mC1 0.3572 (L), R3-2 B8mC1s768 0.3523 (M, 1.46×), R3-3 0.3237 (M), R3-4 B8lC1s768 0.3603 (over: 2.58×), R3-5 0.3580 (L).
  - Adopted: M = R3-2 (YOLOv8m + C1, crops 768) over R2-4 (0.3512); margin 0.0011, disclosed as a near tie. S = C1 and L = R2-1 unchanged. Phase 1 ends.
  - Follow-ups (§2l (a)/(b), dev only, not selection inputs): YOLOv8m own refiners seeds 1 / 2 then D; YOLOv8m whole frame (6 sides, plain / + T4 / + C1) and YOLOv8m heuristics at S (9 configs, + T4 / + C1).
    - Code: k28_r3fu.py (cfc60d54), k28_analyze_r3fu.py (6b1ec433), k28_r3fu_chain.sh (dc14dc7c). Smoke parity 100% (v5.1-S; YOLOv8m whole frame plain / + T4 = r1). Chain PID 2812983.
  - k28_test.py (acb53791) / k28_analyze_test.py (d0dd013a) updated: V6-M = YOLOv8m crops 768; each crop detector loads its own refiner (own-<det>); basewf per detector (`--det`, `--wf ""`). Dev smoke queued, parity only.
- 2026-10-12 21:10 — round-2 chain done (VUW 03:05, "K28 R2 ANALYZE DONE").
  - Official tables vs preliminary: seeds_r2 and parity_r2 identical; candidates_r2 / selection_r2 identical in G, CIs, passes, class and choice. Only GFLOPs ratios differ (≤ 0.0027), because the official run read the measured phase0/gflops_640.csv. Parity_r2 min 1.0 (14 rows).
  - YOLOv8m refiners seeds 1 / 2 trained (md5 d2a92d8f / c88c9742; best epochs 23 / 20). The follow-up whole-frame nuScenes group is done; the others are running.
  - k28_test.py dev smoke after §2m (tmp/k28_v6smoke2_parity.py): 100% identical on every parity column (v6S = r2 C1; v6M = r3 B8mC1s768; v6L = r2 B11C1; YOLOv8m whole frame plain / + T4 = r1, + C1 = fu; YOLO11m whole frame = r1 / r2; wf960 = E / r2). The k28_analyze_test.py column logic was checked on the smoke outputs (171 required columns, none missing).
- 2026-10-12 21:25 — round-3 chain analysis done (VUW 03:10). The official candidates_r3 / selection_r3 / parity_r3 equal the preliminary ones (only the GFLOPs ratios of the R2 rows differ, ≤ 0.0027, measured GF). Choices confirmed: S = C1, M = R3-2, L = R2-1. Phase 1 closed.
  - The final R4 round-2 baselines are recorded in the §2m confirmation. AV2 heuristics + C1 are above ours at M / L (e.g. tsp0@L 0.346, cor0@M 0.315 vs ours 0.311).
  - The phase-1 tables were copied to handoff/K28/phase1/ and results/vuw_summaries/k28/phase1/.
  - Next: wait for the §2m follow-ups (D, YOLOv8m baselines), then §2n freeze + the single test pass (k28_test_chain.sh, drafted).
- 2026-10-13 00:40 — §2m follow-ups done (VUW 06:32), parity_fu 100% (82 rows).
  - D for R3-2: nuScenes 0.394 / 0.399 / 0.395, AV2 0.311 / 0.311 / 0.317 (std 0.003).
  - YOLOv8m whole frame: every side is over the L cap. YOLOv8m heuristics at S: only two rows fall within a class (both L, both below v6-L). No YOLOv8m baseline is in class M.
  - §2n written: v6 frozen with md5s (weights, head, detectors, code), the test rows (prior works reused from K24 without C1; controls plain / + T4 / + C1), and the analysis outputs.
  - Phase-2 test chain launched (k28_test_chain.sh 32daaa9b, PID 3843551). Local watcher bxdfugxnh does crash checks only.
- 2026-10-13 02:30 — §2n.1 addendum written while the pass runs, before any test value was looked at.
  - It adds two items from the task's phase-2 list that §2n left out:
    - the support only / field only / 2 × 2 ablation of v6-S (k28_test_abl.py 692a3716);
    - routers and the oracle re-matched to the v6 GFLOPs (k28_analyze_test2.py 26468b2b).
  - Dev smoke: no crash; v5.1-S J and px are 100% identical to r1.
  - k28_test_abl_chain.sh (4a70322b) launched, PID 4179495. Its analysis waits for "K28 V6 TEST DONE".
  - Phase-3 prep (post-hoc, reporting only):
    - k28_ap.py (d095f777, part apdump): low-confidence detection dumps for v6 and for whole frame on Argoverse-HD val / nuScenes test.
    - k28_ap_score.py: COCO AP (AHD) / det_metrics (nuScenes) vs px and GFLOPs.
    - k28_ap_chain.sh (64ba33dc): starts only after "K28 V6 ANALYZE DONE".
    - nuScenes dev smoke: J / px parity 1.0 vs smoke2; the scorer runs. The AV2 dev smoke is queued for a GPU.
- 2026-10-13 03:00 — phase-3 VUW code (post-hoc; v6 unchanged; no test value looked at):
  - k28_p3.py: k28_test.py copied; k28_test.py itself is unchanged.
    - part `instr` (base / del_all): K2 agreement with π(u960 + C1); K4 natural misses and controlled deletion; K9 TTC-warning consumer; K16 counterfactual chain + coverage; K17 no-support flag; failure cases; `--capture` for the panels.
    - `--ver` / `--draft` / `--only_s`: YOLO11s draft + verifier without re-tuning; fine-tuned verifier seeds 0 / 1 (nuScenes).
    - `--tag nomask`: AV2 without the ego mask.
  - k28_p3_analyze.py: tables (agreement, attribution, missed objects, regime incl. v6-M / L, consumer 2, failures, ctables for y11s / ft0 / ft1 / nomask, additive roadmap).
  - k28_render.py: K8 / K15-style panels from the capture run.
  - k28_p3_chain.sh: starts only after "K28 V6 ANALYZE DONE".
  - Dev smoke 1 (nuScenes scene-0108) is running. Smoke 2 (scene-0910, frames with relevant objects: capture, del_all, analysis, render) is queued.
- 2026-10-13 03:25 — phase-3 smokes on dev (values not used):
  - Smoke 1 (scene-0108): instr, del_all, ours parity, YOLO11s and fine-tuned runs all ran. The AV2 instr step crashed on numpy-array GT geometry (`.iloc`). This was fixed by reading `g["long_near"]` as an array; the step is being re-run (smoke 1b) together with the parity script.
  - Smoke 2 (scene-0910, frames with relevant objects): capture, del_all, every analysis function and the render ran; the panels look right.
  - AV2 AP smoke: the "ap" outputs are non-None for crop pipelines and the minimum confidence is 0.100.
  - k28_p3.py later also gained the three K3 rows for v6-S (pivots pv6S, one round or6S, no guard ng6S; base setting).
  - Added k28_p3_p0.py: phase 0's sota / loss / cost / ablation items for v6. It runs at the end of the p3 chain.
  - md5s: k28_p3.py 66dad0fc, k28_p3_analyze.py a208656c, k28_p3_p0.py d3f19747, k28_render.py 8b496bf3, k28_p3_chain.sh f3687b9a.
  - Launched; both chains wait for "K28 V6 ANALYZE DONE":
    - k28_ap_chain.sh (64ba33dc), PID 154405;
    - k28_p3_chain.sh, PID 161247.
  - Jetson (final window): scripts/61_k28_v6_latency.py (8c2a2223) covers v6-S / M / L and the bounded variant with C1, whole frame plain / + T4 + v3a / + C1 + v3a, KITTI's four cells with the shared configuration, and Argoverse-HD per-frame timing for streaming AP.
    - The own refiners and the C1 head were downloaded (md5 = §2n).
    - Still to do in that window: refiner engines own_yolov8m / own_yolo11m, and whole-frame engines v8s 256x448 / 288x512 / 864x1536 (nuScenes) and 512x416 (AV2).
- 2026-10-13 03:30 — smoke 1b passed. k28_p3.py dev parity is 100% on every column checked:
  - v5.1-S / v6-S J and px;
  - wf960 plain / + C1;
  - so6S / fo6S;
  - five S heuristics + C1;
  - `--only_s` ours.
  - AV2 instr (with pv6S / or6S / ng6S) runs. The phase-3 code is validated. Both chains wait for phase 2.
- 2026-10-13 03:35 — operational, not a code change: the phase-2 shard `ours nusc 0/3` was killed at VUW 07:42, apparently by the out-of-memory killer (bash "Killed", no Traceback; host RAM is tight, about 28 GB of 187 GB available). nuScenes ours stands at 206 / 228.
  - The same command was relaunched through the wrapper (log test/ours_nusc_g0_fill.log). Finished units are skipped. The chain's own re-run remains the fallback.
- 2026-10-13 06:35 — operational, not a code change: the 03:35 entry was incomplete. All three `ours nusc` shard processes of the first launch died at VUW 07:42 (OOM; the log has no "K28 V6 TEST DONE"). The 03:35 fill re-ran shard 0 only, leaving nuScenes ours at 220 / 228 (missing units 208, 211, …, 226 of shard 1 and 227 of shard 2). The phase-2 shard `baseheur --hdet yolov8m nusc 0/3` was also OOM-killed (156 / 228).
  - Relaunched the same commands through the wrapper (test/fill2_nusc.log, VUW 12:35): ours 1/3 and 2/3, and heur8m 0/3. Finished units are skipped; the chain's own re-run remains the fallback.
  - No test value was read; only unit counts.
- 2026-10-13 06:55 — operational, not a code change: a new hourly health check (unit counts and process health only) found that the phase-2 shard `baseheur nusc 2/6` (26 configurations, group g0) had died silently, with no Traceback and no "Killed" line; it is most likely the same 07:42 OOM event. nuScenes baseheur stands at 201 / 228, missing units 68, 74, …, 224.
  - Relaunched the same command through the wrapper (test/fill3_heur_nusc.log).
  - The ours nuScenes fill (fill2) completed: 228 / 228.
  - From now on, a group log whose wrapper exited with fewer "DONE" lines than shard processes is flagged by the hourly check.
- 2026-10-13 10:45 — phase 2: every test unit finished (VUW 16:32:52 "K28 V6 TEST DONE").
  - The analysis k28_analyze_test.py crashed with `KeyError: 'J_cheap_ours'`, an analysis-code bug: `keep` listed J_cheap twice, because J_cheap also matches the "J_" prefix, and the @k28 rename then renamed both copies.
  - The chain does not check the analysis exit code, so it still wrote "K28 V6 ANALYZE DONE". That started the phase-3 and AP chains (16:36). Neither reads the phase-2 tables, so the trigger is harmless.
  - Fix: exclude J_cheap from the prefix list (md5 d0dd013a → ca1c34b4). Analysis re-run by hand (test/analyze_fix.log).
  - This is not a new test pass: it reads the same result files, and no method or row changed.
- 2026-10-13 11:00 — the same duplicate-J_cheap bug was found in two more analysis scripts:
  - k28_analyze_test2.py (§2n.1 ablation + re-matched routers): it crashed with the same KeyError at VUW 16:47, and its chain still wrote "ANALYZE2 DONE". Fixed (md5 26468b2b → 725e481e) and re-run by hand (test/analyze2_fix.log).
  - k28_p3_analyze.py ctable(): latent, since the dev smoke never reached it because the unit counts did not match. Fixed (md5 a208656c → bd7035fd) and deployed before the phase-3 chain reaches its analysis.
  - All three are analysis-only fixes on unchanged result files.
- 2026-10-13 12:00 — **phase 2 DONE → phase2/REPORT.md** (tables re-made after the analysis fixes; parity 100%).
  - v6-S is above v5.1-S on all three splits. v6-M matches on nuScenes / AV2. v6-L is below by the verdict rule on nuScenes (point +0.116).
  - S1 is met at S (deployable rows; the true-V oracle is above v6 on nuScenes / AV2) and not met at M / L. S2 is not met.
  - S4 (phase-2 part): field / support / looming yes; refiner 1 / 3.
  - Kit updated: 03_results, 06_pending, QUEUE, results/INDEX.md, results/vuw_summaries/k28/phase2.
- 2026-10-13 12:30 — dry run of k28_p3_p0.py (sota / cost / loss) on the now-complete phase-2 folders, writing to results/k28/p3/tables_dry, so that a bug surfaces before the phase-3 chain reaches it.
  - It failed with a pandas MergeError: v5.1-S columns come from both ours and abl, so the second merge clashed.
  - Fix: the @k28 rename applies only to the K24 base columns, and a column already merged from an earlier part is kept once (md5 d3f19747 → 5d726b33, deployed).
  - Re-run: all three parts write their tables. The abl part needs the phase-3 instr folders and runs at the end of the chain.
- 2026-10-14 05:35 VUW — dry run of the phase-3 analysis (k28_p3_analyze.py with OUT → tables_dry; k28_p3_p0.py --part abl) on the finished phase-3 folders, so a code bug surfaces before the chain's own run.
  - Both ran without error. Every table is written; no ablation or roadmap row is missing, and parity has 57 columns per dataset.
  - The only gap is the AV2 nomask ctable, because its heuristics were at 138 / 150 units. The chain's run makes it.
  - Only errors and row counts were looked at, no values.
- 2026-10-14 06:10 VUW — phase-3 VUW runs done (05:59 "K28 P3 RUNS DONE"; analysis 06:10 "K28 P3 ANALYZE DONE"; analyze.log notes [], p0v6.log checks 0.0, no Traceback). No shard died in phase 3.
  - Added post-hoc k28_p3_nomask_cm.py (cb1ac41e): AV2 no-mask rows class-matched as in phase 2 → tables/{headline,class_matched}_nomask_v6.csv.
  - **phase3/REPORT.md written:**
    - S4 not met (guard 0/3, refiner 1/3);
    - S5 met at S where in-class rows exist;
    - S6 met on dev seeds;
    - AV2 no-mask v6-M drop;
    - K6 / NAVSIM statement.
  - Kit updated: 03_results, 06_pending, QUEUE, results/INDEX.md, results/vuw_summaries/k28/phase3.
  - Capture run for the panels (results/k28/p3/capture_frames.csv: K8 8 frames + K15 4 + failure picks 18 = 30 frames; 11 / 12 / 3 units) launched through the wrapper at 06:32 VUW (p3/capture_g0.log).
  - Note for the consolidated report: phase2/REPORT.md says the Argoverse-HD oracle router is "below v6 at every budget". At S the verdict of v6 − oracle is "matches" (+0.049 [−0.002, +0.101]); at M / L v6 outperforms it.
- 2026-10-14 00:49 +07 — **final Jetson window opened** (~/.jetson_pause set; busy.sh printed only "paused"; DEEP session risk-aware-perception-ae informed and acknowledged).
  - scripts/k28_final_jetson.sh (md5 49f109cd) runs one process at a time:
    1. engines/build_k28_final.sh (9bf50d71): own refiner engines own_yolov8m_vuw / own_yolo11m_vuw, and YOLOv8s whole frame 256x448 / 288x512 / 512x416 / 864x1536 (fp16, workspace 1024; no YOLOv8m 448x768);
    2. 61_k28_v6_latency.py (8c2a2223): smoke, then nuScenes v6-S / M / L / v5.1-S and AV2 (mask 1830) with parity against the phase-2 per-frame results (results/k28_ref, copied from VUW), then KITTI test v6-S and Argoverse-HD v6-S timing.
  - Log: logs/k28_final_jetson.log.
- 2026-10-14 01:40 +07 — Jetson window, operational. The own-refiner export failed in part 1: the checkpoints hold numpy-2 arrays (`ModuleNotFoundError: numpy._core` in the edge env). build_refiner_v3.sh has no `set -e`, so the script printed "built" anyway.
  - Fix: on VUW (tmp/k28_refconv), the checkpoints were re-saved with fmu / fsd as float lists. State tensors and norms are identical (state md5 60e047e9 / 94bfeac0; files refiner_own_*_vuw_np1.pt 9248a31b / efaed934). The originals are kept as .orig.pt.
  - scripts/k28_final_jetson2.sh waits for part 1 to finish, then builds both refiner engines and re-runs the v6-M / v6-L latency runs (nuScenes, AV2). The part-1 v6-M / v6-L runs fail at start-up on the missing engine and are discarded.
  - No pip install; numpy in the edge env is unchanged.
- 2026-10-14 ~01:40 +07 — **the Jetson rebooted during the YOLOv8s 864x1536 TensorRT build** (build log ends at 01:38 in tactic timing; board back at ~03:56). This is the third reboot from a large static build. That engine is not retried, so nuScenes whole frame u1536 is not timed. Part 2 (waiting for part 1) died with the reboot.
  - 2026-10-14 04:00 — resumed in the same window: the pause file was still in place and busy.sh printed only "paused".
  - scripts/k28_final_jetson3.sh (6d2d9109) does: own-refiner engines (re-saved checkpoints), smoke v6-S and v6-M, then every latency run (nuScenes without u1536), KITTI test, Argoverse-HD timing. Log: logs/k28_final_jetson3.log.
  - The DEEP session was not running after the reboot, so it could not be messaged. It will be told when it is back.
- 2026-10-14 08:23 VUW — capture run done (11 / 12 / 3 units, "K28 P3 INSTR DONE", no Traceback). k28_render.py is rendering into results/k28/p3/panels (render.log).
- 2026-10-14 04:05–04:20 +07 — Jetson part 3: two more numpy-2 incompatibilities fixed the same way (no package change).
  - The C1 head out/k28_temporal/temporal_head.pt was re-saved with fmu / fsd as lists (temporal_head_np1.pt, md5 1b1e25d1; state identical). On the Jetson it is engines/k28_temporal_head.pt; the original is kept as .orig.pt.
  - The parity references are now reduced pickles (unit, frame, J_*) in results/k28_ref; the originals are in results/k28_ref_np2.
  - Smokes v6-S / v6-M passed (with --warmup 5).
  - Local nuScenes images are the 85 old DEEP scenes, not test scenes, as in K5 / K11. Jetson nuScenes G is therefore G on the timed frames; parity is checkable on AV2 only (the 10 local val logs).
  - The smoke shows the C1 head (torch fp32, ResNet-18 on 3 crops per chain) adds ~30 ms per frame: u448 + C1 + v3a 46.6 ms vs u448 + T4 + v3a 16.3 ms (20 frames, smoke only).
- 2026-10-14 04:35:45 +07 — **the Jetson hard-reset again**, this time during the first final latency run (nuScenes v6-S, 6 whole-frame engines), not a build.
  - Agent history: max temperature 68 °C, power ~25–30 W, memory 91–92% (swap 2%).
  - The kernel log stops mid-line, with no OOM / panic / thermal message. The 01:38 reset (build) had memory 35% and ~59 °C.
  - Cause not identified. The window is paused pending the user's decision.
- 2026-10-14 11:05 VUW — **S6 on test** (k28_p3.py --seeds, md5 8b791fa2; k28_p3_seeds_chain.sh / k28_p3_seeds.py): seed std of G on test ≤ 0.0099 (AV2 v6-M). Half the smallest S1 margin at S is 0.0527, and over every budget 0.0113, so **S6 met on test**. Seed 0 = phase 2 on 100% of frames. Dev smoke parity 100%.
- 2026-10-14 ~11:20 VUW — **K6 re-run with v6-S launched** (user: "Còn gì trên VUW thì làm nốt cho xong đi").
  - perception_pilot.py (23539df7) adds branch ours_v6: v6-S without refiner, v_close 0, crops 0.6, C1 via the latency script's port; the other branches are unchanged.
  - Route-1 smoke: completed (DS 100). Phantoms / crops per frame: ours_v6 3.97 / 4.60, ours (v3) 3.89 / 5.19, u1600 4.63 / 0, so route 1 is phantom-heavy for every branch.
  - b2d_v6_runs.sh (1adbc985) + b2d_dispatch_v6.sh (adbacbd1, counts only its own evaluators): 3 runs on K6's route files, MAXJ 2.
- 2026-10-14 05:30 +07 — Jetson resumed after the user's direction ("để quạt 100% đi"). The fan is manual pwm 255 (3,389 rpm); the mode stays MAXN.
  - scripts/k28_final_jetson4.sh splits the whole-frame engines over two processes per dataset (_a / _b) to lower memory, then runs v6-M / v6-L / v5.1-S, KITTI and Argoverse-HD. Log: logs/k28_final_jetson4.log.
- 2026-10-14 06:30 +07 — Phase-3 REPORT.md S6 updated with the test seed table (seeds_test_v6.csv, s6_test_v6.csv mirrored to vuw_summaries/k28/phase3 and handoff/K28/phase3); 03_results S6 line and results/INDEX.md updated. Operational: K6 ours_v6 dispatcher started 3 evaluators instead of MAXJ=2 (chunk1's evaluator had not started when chunk2 was checked); left running (GPU memory 18–21 / 24 GB, no DEEP evaluators). full: 24 / 220 routes after 1 h; per-route scores match the v3 `ours` branch closely so far.
- 2026-10-14 07:23 +07 — **Jetson final window DONE** (k28_final_jetson4.sh; no further reset with the fan at 100%). nuScenes v6-S / M / L / v5.1-S, AV2 v6-S / M / L / v5.1-S (ego mask, parity 94–96% identical per-frame cost), KITTI v6-S, Argoverse-HD v6-S. The Argoverse-HD summary step failed (`dd.gt` resolves to the pandas method); fixed to `dd["gt"]`, added `--post_only` (61_k28_v6_latency.py 8c2a2223 → 6341d349; regression on an AV2 run: latency.csv and parity.csv identical) and recomputed from the saved pickle. Pause file removed 07:28; DEEP session not running, so no message could be delivered (ListAgents empty). Fan left at manual 100%.
- 2026-10-14 07:35 +07 — Streaming AP (k28_ahd_sap.py c44b8ea7, VUW): v6-S sAP 0.064 (56% of frames processed). S3 analysis (scripts/62_k28_s3_front.py): **S3 not met** — v6-S on the AV2 front in mean and p95, off the nuScenes front (20 local scenes) in both; KITTI shared configuration below whole frame + components (mean4 0.235 vs 0.370). Written to phase3/REPORT.md §0 and §16, 03_results, 06_pending, QUEUE, INDEX. Open: KITTI Planner B mono refiner lift of §6 absent with v3a (not investigated).
- 2026-10-14 07:35 +07 — Operational: K6 ours_v6 full chunk0 logged "couldn't spawn the ego vehicle" (CARLA) at 13:22 VUW; the dispatcher relaunches a chunk with RESUME once its evaluator exits. Check at the next hourly round whether chunk0 progresses.
- 2026-10-14 13:42 VUW — K6 ours_v6 full chunk0 (24 / 55 routes): after the CARLA crash the evaluator waited for the simulator and exited on its own at 13:41:55 on the 600 s simulator time-out (a SIGTERM to its exact PID was sent at 13:41; no SIGKILL needed). The dispatcher relaunches chunk0 with RESUME when an evaluator slot frees (chunk3 is first in line). No intervention needed for this pattern in future: the 600 s time-out ends a hung evaluator.
- 2026-10-14 08:30 +07 — Panels copied (rsync, 361 files, 639 MB) to handoff/K28/phase3/panels/; phase3/REPORT.md §17 written; 06_pending updated. K6 ours_v6 full: 75 / 220 routes; chunk1 and chunk2 each had one CARLA crash (chunk2 exited 14:00:47 VUW; chunk3 launched 14:02); the dispatcher resumes crashed chunks as slots free.
- 2026-10-14 16:28 VUW — **K6 v6-S, route 11755 excluded as in K6.** ours_v6 run 1 reached route 11755 in chunk0 (28 / 55 done, through 4937) and hung there (simulator "took longer than 600 s to update", no progress for 67 min). Every K6 branch except cheap hung on this route: run 1 of K6 stopped chunk0 after 4937 and ran the other 26 routes as chunk0b, and runs 2 / 3 have no 11755 (b2d_repeats_summary.py excludes it). b2d_v6_runs.sh had used chunks 0–3 of run 1 including 11755, which was a launcher mistake. Fix: stopped b2d_v6_runs.sh, its dispatcher and the hung chunk0 evaluator (exact PIDs 1269798 / 1269802 / 2076019). The running chunk2 / chunk3 evaluators were left alone. Launched b2d_v6_runs2.sh (md5 59d4941f): run 1 continues with chunks 0b 1 2 3; runs 2 / 3 are unchanged. Progress at the switch: 124 / 219 routes of run 1.
- 2026-10-14 19:35 VUW — K6 v6-S run 1 DONE (219 routes, chunk0 28 + chunk0b 26 + 3 × 55; 10 routes Failed - TickRuntime against 7–13 for the K6 branches). Interim summary with k28_b2d_v6_summary.py (md5 15476b88; ours_v6 merged with the official tool, K6 branches read from their existing merged.json and not re-merged): DS 92.94, SR 0.845. It ties cheap / u960 / u1600 / K6 ours (all intervals across 0) and is below privileged (−3.49 [−6.07, −1.12]). Written to phase3/REPORT.md §0 and §14a, 03_results, 06_pending, INDEX; b2d_v6_*_run1.csv in handoff/K28/phase3 and vuw_summaries/k28/phase3. Run 2 started 19:25 VUW.
- 2026-10-14 ~19:00 +07 — **Consolidated REPORT written → handoff/K28/REPORT.md** (user: "trong lúc chờ, viết REPORT tổng hợp K28 trước đi"). (i) 74 kit numbers mapped to final v6 values (03_results §1–§8); (ii) scorecard S1 met at S only, S2 / S3 / S4 not met, S5 met at S with caveats, S6 met; (iii) every not-above pair in final/not_above_all.csv (908 pairs, 106 with the row significantly above; scripts/63_k28_consolidate.py md5 04354246, reads the phase CSVs only); (iv) final/test_evaluations.csv (28 rows; adds the K28 phase-1 Jetson windows, which computed G on 8 AV2 val logs while timing the B candidates: reported only, not a selection input). Correction carried: phase2/REPORT.md §2 said the Argoverse-HD oracle router is below v6 at every budget; at S it is "matches" (+0.049 [−0.002, +0.101]); fixed in place. K6 rows are run 1 of 3. Kit: 03_results (pointer + correction), 06_pending, QUEUE, INDEX.
- 2026-10-15 09:38 VUW — **K6 v6-S runs 2 / 3 DONE** (dispatch log "B2D V6 RUNS DONE"; CARLA crashes re-run by the dispatcher, no route lost).
  - Three-run summary (k28_b2d_v6_summary.py 15476b88, log results/vuw_summaries/k28/phase3/k28_b2d_v6_summary.log): v6-S DS 92.54 ± 0.82, SR 0.839 ± 0.011. It ties cheap / u960 / u1600 / K6 v3 (vs v3 −0.26 [−1.23, +0.61]) and is below privileged (−4.09 [−6.36, −2.02]).
  - The K6 branches reproduce the K6 report exactly.
  - Updated: phase3 §14a (three-run table) and summary bullets; consolidated REPORT (header, row 48, §3 via 63_k28_consolidate.py: still 908 rows / 106 significant, only the K6 rows changed; §4; §6); test_evaluations.csv; 03_results; 06_pending; QUEUE; INDEX. **K28 has no open run left.**

- 2026-10-15 19:50 — **KITTI Planner B mono discrepancy: cause found** (user: "ok, chạy đi").
  - **Cause:** the Jetson reference refiner (58.RefinerBatch, used by 61 on every dataset) hard-codes v3a's camera flag at 1.0, the nuScenes / AV2 value. v3a was trained with 0.0 for KITTI (k23_train_refiner.load); K1's runs (40_e2e_v2.TRTRefiner) fed 0.0.
  - **CPU check on VUW** (remote/A_method/k28_kitti_flag_check.py, md5 7359eb1b; KITTI train / val boxes only), |log range error|:
    - KITTI val: 0.055 with flag 0.0, 0.084 with 1.0 (lift 0.210);
    - KITTI train: 0.010 vs 0.070.
    - With 1.0, 17–21% of boxes are corrected the wrong way, and the mean bias is left mostly uncorrected.
    - Table: handoff/K28/phase3/kitti_refiner_flag_check.csv (also results/vuw_summaries/k28/phase3/).
  - **Updated:** phase3 REPORT §16, K28 REPORT §6, 03_results (KITTI shared-config line), 06_pending.
  - **Pending:** a Jetson re-run of the KITTI shared-configuration table with the per-dataset flag (≈ 20 min) once the Jetson's power supply is fixed; the user decides.
