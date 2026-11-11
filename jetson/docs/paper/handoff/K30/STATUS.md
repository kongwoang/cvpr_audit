# K30 STATUS

**Task (revised 2026-10-15):** post-hoc analyses of the FROZEN v6-S (method_freeze_draft §2n) and of the baselines of K28 Table 1.
- Parts V1–V8 run on VUW. Part J runs on the Jetson in K29's exclusive window, after the new power supply and after K29.
- **Started:** 2026-10-15 20:00 +07.
- **Everything here is post-hoc. Nothing is selected on a test number.**

**Rules (task text):**
- R1. No change to v6-S.
- R2. Baselines with the same detector and components, in K28 forms:
  - refiner v3a + C1 for every pipeline;
  - truncation guard and own history for the crop heuristics.
- R3. Each new benchmark or metric (V8) is pre-registered here before running: split, metric, pipelines, px / latency accounting.
- R4. No new licence or download. List what the user would have to accept.
- R5. Intervals: paired unit bootstrap, 2,000 resamples, seed 0.
- R6. Report everything, including the rows where v6-S is not above.

## What K28 already has (read from the code, 2026-10-15)

**Hybrid = `tsp8` (k28_test.py, baseheur).**
- Crops: the support of the cheap draft (`SP.support` of the braking controller), filled first; then the ego-corridor tube TUBE[8] = (near 10 m, far 70 m, "fill") with the remaining budget. The tube widens in 0.25 m steps until the canvas exceeds the budget.
- C1 lift on its own history, refiner v3a on its support objects, truncation guard (`SP.guarded`), merged once (`IE.merge_once`).
- K28's budget: CAPS[c] × (cheap px + v5.1-S's crop px on the same frame) − cheap px. So K28's S heuristics are frame-matched to **v5.1-S**, not to v6-S.

**The 9 S heuristics in K28 phase 3:** unc2, unc4 (uncertainty), tcr0, tcr3 (temporal criticality), cor0, cor8 (ego corridor), tsp2, tsp8 (hybrid), rnd.

**V1c, answered from the code.**
- In the instrumented run (k28_p3.py, part instr; consumer2_v6.csv), the heuristics' TTC-consumer actions (`a2_*`) come from the same crops as their braking rows.
- So the hybrid tsp8@S + C1 there used the **braking** support (`sup_d = SP.support(ch["mono"], v, pp)`), not the TTC consumer's.
- A re-run with the TTC consumer's own support (`IE.TTCWarn(pp, 0).support`) is part of the V1 GPU run (below).

**Per-frame records available:**
- phase 2: results/k28/<ds>_test_v6{ours,basewf,baseheur…}: actions and costs per pipeline;
- phase 3 instr: results/k28/p3/<ds>_test_instr: both consumers, `found_*` recall of decision-relevant annotated objects, ranges of the relevant objects, K16 coverage, captures for listed frames.

**Not stored per frame:**
- the continuous required deceleration a_gt (needed to re-score with other loss weights, V5a);
- the decisive object;
- ego yaw rate.

A CPU ground-truth pass adds these (k30_gt.py, below).

## Plan

| item | prio | how | where | status |
|---|---|---|---|---|
| GT pass | P0 | per frame: a_gt (continuous), decisive annotated object (argmax demand: range, lateral centre, box, class), relevant set, TTC-consumer GT, ego speed, yaw rate | VUW CPU, k30_gt.py | DONE |
| V1a | P0 | v6-S + the 8 S heuristics (unc2, unc4, tcr0, tcr3, cor0, cor8, tsp2, tsp8) + C1 + v3a at a per-frame cap of m × (cheap + v6-S crop px) − cheap, m ∈ {0.75, 1.0, 1.25, 1.5} | VUW GPU, k30_eval.py --part match | nuScenes, AV2 DONE; Argoverse-HD 20 / 24 (2026-10-16 10:55) |
| V1b | P0 | v6-S crop long side {960, 1152, 1280} × support context {2, 2.5, 3} (post-hoc axis, not a configuration) | VUW GPU, --part axis | DONE |
| V1c | P0 | hybrid with the TTC consumer's own support vs braking support, both scored by the TTC consumer, at K28's S budget and at the v6-S × 1.0 budget | in --part match | nuScenes, AV2 DONE |
| V2 | P0 | strata: ego-speed tercile, yaw-rate tercile, required action, decisive-object range bin; field footprint outside the tube; frames whose decisive object is outside the tube | CPU on --part match records + GT pass | nuScenes, AV2 DONE |
| V2b | P0 | leader-following (IDM) consumer, pre-registered below | VUW GPU, --part idm | DONE |
| V3 | P0 | disagreement with F = u960 + C1, safety-critical subset, top-20 costly frames with cause | CPU on p3 instr records; capture re-run of the 60 frames with k28_p3.py --capture | taxonomy DONE; causes nuScenes, AV2 DONE |
| V4 | P0 | Bench2Drive official metrics from the three finished runs | VUW CPU | DONE; in-loop logging run (4th run) RUNNING |
| V5a | P0 | loss weights, re-scoring of stored actions | CPU on phase-2 / p3 records + GT pass | DONE (+ prior works, k30_v5_prior.py) |
| V5b / V5c | P0 | consumer parameters and probe assumptions, re-run | VUW GPU, --part cons | DONE |
| V6 | P1 | NAVSIM hybrid for PDM-Closed, probing runtime | VUW CPU (K20 setup) | (ii) DONE; (i) hybrid re-run after the tube fix RUNNING |
| V7 | P1 | detection metrics (detections dumped per pipeline) | VUW GPU dump + CPU scoring | nuScenes, AV2 DONE; Argoverse-HD RUNNING |
| V8a–e | P1 | pre-register, then run | | V8c, V8d, V8e DONE; V8b decision in log; V8a navtest RUNNING, navhard RUNNING |
| V8f–h | P2 | availability check first (R4) | | V8g DONE (27 frames); V8f needs the user's registration; V8h P2 not started |
| J1–J3 | after K29 | Jetson | | waiting for the power supply |

## Pre-registration: V2b leader-following consumer (written before any run)

**Consumer `IDMFollow`**, max-composable, used through `vtf.spec_general.run` exactly like K9's TTC consumer:
- **Leaders:** objects with `in_corridor(lat_min, lat_max, 1.2 m)` (the braking controller's corridor) and 0 < z ≤ 80 m.
- **Demand of one leader:** the IDM interaction deceleration.
  - b_obj = a_max · (s* / s)², with s* = s0 + v·T + v·Δv / (2·√(a_max·b)).
  - s = max(z − s0_body, 0.5) is the gap, with s0_body = 0 (z is already the near-face range).
  - Δv = max(z / ttc, 0) is the closing speed (ttc from the lift; GT ttc for the required action).
  - IDM parameters: a_max = 1.5 m/s², b = 2.0 m/s², s0 = 2.0 m, **T = 1.5 s** (task text), δ irrelevant (v0 = v, so the free-road term is 0).
  - Demand is clipped to [0, 9] m/s² (the braking a_cap).
- **Action:** the braking controller's discrete thresholds on the max demand: KEEP < 1.0 ≤ DECELERATE < 3.5 ≤ HARD_BRAKE. v ≤ 0.1 m/s → KEEP.
- **Required action:** the same function on the annotated objects (GT range, lateral extent, ttc).
- **Cost:** `rap.planner.decision_cost` with the required IDM demand as a_req_true (default CostParams). G = (ΣJ_cheap − ΣJ) / ΣJ_cheap.
- **Field:** probe cells (r, y) of `SP.R_GRID` × `SP.Y_GRID` inside the corridor where a static probe (closing at v + v_close, v_close = 0) would reach the next action's threshold. **Support:** objects whose demand reaches the current action's threshold. Both are derived from the demand function; nothing is tuned.

**Pipelines, scored by IDMFollow:**
- cheap;
- whole frame u960 + C1 + v3a, native + C1 + v3a;
- **ours-IDM:** the loop of `spec_general.run` with IDMFollow's field and support, + C1, refiner v3a on IDMFollow's support, ≤ 3 rounds, guard. This is the analogue of K9's "ours driven by the TTC consumer". The frozen v6-S (braking-driven) is also scored by IDMFollow.
- **hybrid:** IDMFollow's support of the draft + tube TUBE[8], at a per-frame budget = ours-IDM's crop px;
- **uncertainty:** unc2 at ours-IDM's crop px.

All with + C1 + v3a; the refiner is applied to the scoring consumer's support objects. Datasets: nuScenes new test, AV2 val (ego mask), Argoverse-HD val. Intervals: R5.

## Log
- 2026-10-15 20:00 — task received.
  - Read k28_test.py (acb53791), k28_p3.py (8b791fa2), instr_eval.py (2177d413), spec_general.py (dbc735c7), rap.planner. VUW and local copies are identical.
  - Hybrid definition and V1c answer: above.
- 2026-10-15 20:40 — **GT pass DONE** (remote/A_method/k30_gt.py, md5 49396624; VUW CPU; results/k30/gt/<ds>.pkl). Frames: nuScenes 9,181 / 228 scenes; AV2 11,818 / 150 logs; Argoverse-HD 15,062 / 24 sequences (incl. frames without labels, flagged gt = False). Same counts as K28.
- 2026-10-15 21:10 — **V4 DONE** (remote/A_closedloop/k30_b2d.py, md5 db2e22d1; CARLA env, CPU; results/k30/b2d/).
  - **Official Bench2Drive code:** third_party/Bench2Drive/tools, the offline-map version of ability_benchmark.py, and efficiency_smoothness_benchmark.py.
  - 3 runs × 6 branches × 219 routes. DS / SR reproduce K28's b2d_v6 tables.
  - **Not computable from the stored runs:**
    - recall of actors within 3 s TTC, minimum TTC, and *unnecessary* hard brakes: metric_info.json logs only the ego state, and perception_*.jsonl only route means;
    - they need a re-run of the B2D runs with actor logging (user decision; K6 took ≈ 1 day per run).
  - **The ego hard-brake proxy** (entries into a_lon < −4.05 m/s², official bound) is dominated by simulator artefacts: the logged acceleration reaches −26 m/s², and the proxy reads ≈ 60 / km for every branch, privileged included. It is reported but labelled as such.
- 2026-10-15 21:40 — **V3 (stored part) and V5a DONE** (remote/A_method/k30_stored.py; results/k30/tables/v3_taxonomy.csv, v3_top_frames.csv, v5a_loss_weights.csv, v5a_check.csv).
  - V5a: re-scoring at the base weights reproduces every stored cost exactly (max |ΔJ| = 0, all 11 pipelines × 3 datasets).
  - **"Best published method" cannot be re-scored.** The prior-work results store only the per-frame cost, not the action (results/sahi/eval_newtest.pkl: J_sahi_*; AV2 baselines__val.pkl: J_fovea_kdei960). V5a / V5b for that row need a re-run of SAHI (nuScenes) / FOVEA KDE_I 960 (AV2) that logs the actions; Argoverse-HD has no prior-work row in K28.
- 2026-10-15 22:00 — **k30_eval.py** written (remote/A_method/k30_eval.py; parts match / axis / idm / cons).
  - **Smoke, nuScenes scene-0110, part match:** 100% identical to K28 phase 3 instr on every shared column (draft, v5.1-S, v6-S, tsp8@S + C1; a / J / px / TTC consumer).
- **V8 data availability (R4), checked 2026-10-15:**
  - VUW has: argoverse1, argoverse_hd, av2, navsim (navsim_logs, sensor_blobs; NAVSIM v2 code with navhard_two_stage), nuplan, nuscenes; CARLA 0.9.15 with Town01–06 and leaderboard/data/longest6.xml (carla_garage).
  - **No ZOD, Waymo Open, ONCE or CODA → V8f and V8g STOP** (what the user would have to accept: below, at report time).
  - KITTI is on the Jetson only (DEEP's split); V8c needs either the Jetson or a copy of our own KITTI images to VUW. This is not a new licence.
- 2026-10-15 22:30 — **smoke DONE and checked.**
  - Part match is 100% identical to K28 phase 3 instr on nuScenes scene-0110 and AV2 log 02678d04 (ego mask): draft, v5.1-S, v6-S, tsp8@S + C1; a / J / px / TTC consumer.
  - Parts axis (its (960, 2) point = v6-S exactly), idm and cons run.
  - k30_analyze.py runs on the smoke outputs.
- 2026-10-15 22:35 — **GPU chain launched** (scripts/A_method/k30_chain.sh, md5 757d28cf; k30_eval.py 55471d34).
  - 18 groups of 3 shard processes via run_when_gpu_free.sh (SHARE=1, ≥ 10 GB free), in order: match (6 shards per dataset), cons (6), idm (3), axis (3). AV2 runs through k10_masked_run.py.
  - Logs: logs/k30/<part>_<ds>_g<i>.log. The hourly check (k28_hourly.sh) now has a K30 section.
- 2026-10-15 23:00 — **prior-work re-runs launched** for V5. The best published row per dataset by K28 phase 3 §4 G:
  - nuScenes: SAHI scale 1.0 (G 0.130). Script k30_sahi_nusc.py (md5 d7dfdf00) = sahi_baseline.py with only that variant.
  - AV2 (ego mask): FOVEA KDE_I 960 (−0.023). Script k30_fovea_av2.py (e542a92c) = av2_baselines.py with only FOVEA, run through k10_masked_run.py.
  - Both store the action, the lifted geometry and a_gt per frame; parity against the stored J is checked at the end. Log: logs/k30/prior.log.
  - Argoverse-HD has no prior-work row in K28, so there is none in V5.
  - Prior works keep their official settings (no refiner, no C1). Their geometry does not depend on the consumer, so V5a / V5b / V5c re-score them on CPU.

## Pre-registration (R3), written 2026-10-15 23:00 before any of these runs

**V7 detection metrics** (standard metrics; no tuning; one run, report as-is):
- **Splits:**
  - nuScenes new test: every 4th keyframe per scene, as K28 det_metrics;
  - AV2 val with the ego mask: all frames;
  - Argoverse-HD val: annotated frames.
- **Pipelines** (K28 Table 1 forms; S budget):
  - cheap;
  - whole frame u640, u960, u1280, native;
  - uncertainty unc2@S, temporal-crit tcr3@S, hybrid tsp8@S;
  - v6-S.

  Geometry-based metrics use each pipeline's + C1 + v3a geometry (the draft's own for cheap).
- **Outputs:**
  - AV2 and Argoverse-HD: the K7 / K28 definition with conf ≥ 0.1 (draft boxes outside every examined region ∪ crop boxes, NMS 0.5), which needs the draft run online;
  - nuScenes (cached draft, no low-confidence boxes): the merged scene at conf ≥ 0.25, as K28 det_metrics.
- **Ground truth:**
  - the benchmark's projected 2D boxes (height ≥ 10 px), classes vehicle / person / cyclist (class-compatible matching, instr_eval.COMPAT);
  - Argoverse-HD: argoverse_hd_eval's ground truth and its three road-user groups.
- **Metrics:**
  1. whole-image AP (IoU 0.50:0.95, COCO 101-point) and AP50, mean over the three classes (Argoverse-HD: K28's COCO evaluation);
  2. corridor AP / recall:
     - GT restricted to in_corridor(lat_min, lat_max, 1.2 m) with long_near ≤ 80 m;
     - predictions matched to other GT are ignored;
     - an unmatched prediction is a false positive only if its own monocular estimate lies in that corridor;
  3. decision-relevant recall (GT with braking demand ≥ thr_decel matched at IoU ≥ 0.3, instr_eval.matched) and precision (predictions with estimated demand ≥ thr_decel that match a relevant GT, over all such predictions);
  4. recall at IoU ≥ 0.5 by GT range bin (0–30, 30–50, 50–80, > 80 m) and by native box height (< 16, 16–32, > 32 px), conf ≥ 0.25;
  5. spurious braking detections per 1,000 frames: predictions with estimated demand ≥ thr_decel matching no GT box of any class at IoU ≥ 0.3.
- **Accounting:** px = mean tensor px / native px, as K28.
- Where v6-S is below whole frame on a metric, one line explains why.

**V8a NAVSIM v2 (EPDMS):**
- Code: the installed NAVSIM v2 devkit (third_party/navsim, `navsim/evaluate/pdm_score.py`), default v2 scoring parameters.
- Split: navtest (12,146 tokens, as K20).
- Protocol: K20's perception-in-the-loop PDM-Closed (navsim_detect / navsim_pdms). The trajectories of cheap, u960 (full960), native, spec_pdm (= v6-S's role on NAVSIM: the probing field) and the hybrid (V6 i) are re-scored with EPDMS and its sub-scores.
- Interval: paired log bootstrap, 2,000, seed 0.
- navhard two-stage: only if its synthetic sensor data is on VUW (checked before running; otherwise listed as missing).

**V8c KITTI 2D detection AP:**
- Split: DEEP's KITTI test (6 tracking sequences, 3,536 frames). KITTI tracking labels give truncation / occlusion / box height, so KITTI's difficulty levels apply.
- Metric: the official KITTI 2D protocol: Car IoU 0.7, Pedestrian / Cyclist 0.5; easy / moderate / hard by min height 40 / 25 / 25 px, occlusion ≤ 0 / 1 / 2, truncation ≤ 0.15 / 0.30 / 0.50; 40 recall points; DontCare and neighbouring classes ignored as in the devkit.
- Pipelines: cheap, native (1248), v6-S with K28's shared KITTI configuration. Refiner v3a is given KITTI's camera flag 0.0 (the K28 KITTI flag bug, handoff/K28 phase3 §16); boxes are unaffected by it, G is.
- G next to AP.
- Compute: the Jetson (in window J) or a copy of our own KITTI test images to VUW (≈ 3 GB; not a new licence).

**V8d nuScenes ego speed ×{0.5, 1.5}:**
- v' = s·v replaces v in the braking controller, the field, the support, the heuristics' support / ranking and the refiner support.
- GT closing speeds are re-derived: v_close' = v_close + (s − 1)·v for each annotated object (ego contribution along the axis), ttc' = range / v_close' when v_close' > 0, else ∞. The image-based lifts (C1 / T4 ttc, monocular range) are unchanged, because they come from the logged images.
- Pipelines: cheap, u960 + C1 + v3a, native + C1 + v3a, hybrid tsp8@S (budget from v5.1-S at the same scaled speed), v6-S.
- Metrics: G, px, v6-S minus each, and the field footprint area against the tube area.

**V8e corner cases in the existing splits:**
- Frames whose raw annotations hold, inside the ego corridor (|lateral| overlapping 1.2 m, ≤ 80 m):
  - nuScenes: barrier, traffic cone or construction vehicle;
  - AV2: construction cone, construction barrel or bollard.
- On those frames: G of cheap, u960 + C1 + v3a, native + C1 + v3a, hybrid tsp8@S, v6-S (stored per-frame records), and recall of the corner-case objects: projected 2D box matched class-agnostically at IoU ≥ 0.3 by any output box (V7 dumps).
- If all fail: the COCO detector has no such classes; selection cannot recover what the target detector cannot detect.

**V8b:** not run. The installed Bench2Drive (carla_garage) ships PDM-Lite (the K6 planner) and TransFuser++, a sensor-to-trajectory network that does not take an object list, so it cannot be fed by the perception branches. No Think2Drive / other expert is installed.

**V8f / V8g:** STOP (R4). Not on VUW:
- Zenseact Open Dataset: access by application to opendataset@zenseact.com; CC BY-SA 4.0; ≈ 3.4 TB for the full dataset; Frames is a subset.
- Waymo Open (perception): Waymo Dataset License Agreement for Non-Commercial Use, accepted with a Google account; hundreds of GB.
- ONCE: registration on its site; size not determined here.
- CODA: annotations from coda-dataset.github.io (its own licence; its ONCE part needs ONCE images; its nuScenes and KITTI parts reuse frames we have).
- The user decides.

**V8h:** P2. longest6.xml and Town01–06 are installed (carla_garage); whether K6's perception_pilot runs on it without new assets is checked after P0 / P1.
- 2026-10-15 23:40 — **operational: relaunch with a lower free-memory threshold.**
  - With MIN_FREE_MB 10000 only match_nusc_g0 (and then idm_nusc_g0) had started. The GPUs hold 10–20 GB of other users' jobs, while one K30 group uses ≈ 2.5 GB (0.6–0.8 GB per process).
  - The 19 waiting wrappers were stopped by exact PID (none had started a process), and the chain was relaunched with MIN_FREE_MB 3500, skipping the two running groups (k30_chain.sh, md5 c4f526ab, SKIP list).
  - The prior-work re-runs (MIN 4000) and the k30_eval2.py smoke (MIN 3000) were also relaunched.
  - No result was affected: finished units are skipped, and nothing had been written by the stopped groups.
- 2026-10-16 00:30 — **k30_eval2.py** (parts apdump for V7 / V8e and speed for V8d; a separate file, so the running k30_eval.py jobs are untouched).
  - **Smoke:** apdump (nuScenes, AV2 ego mask) and speed (nuScenes) are 100% identical to part match on every shared column (J_cheap, J_v6S|v3a, px_v6S, J_tsp8@S + C1). V7 scoring (k30_v7.py) runs on the smoke dumps.
  - **Speed smoke:** actions do change with s (e.g. u960 + C1 on 1 / 4 frames of scene-0110 at ×0.5 / ×1.5).
    - The v6-S crop area does not change, because the field crop is ONE bounding box of the sensitive probe cells, and its top / bottom / width are set by the NEAREST sensitive cells: a 1.6 m probe's top edge rises in the image as the range shrinks.
    - So the run also records the field's sensitive extent: number of sensitive cells and the farthest sensitive range on the draft (`field_cells`, `field_rmax`).
  - **Launched** (scripts/A_method/k30_chain2.sh, md5 6b4b2b52; k30_eval2.py d8534941; k30_v7.py c1d72713): apdump × 3 datasets, speed × nuScenes, 3 shards each, MIN_FREE_MB 3500.
- 2026-10-16 00:20 — **V8e corner-case pass DONE** (k30_corner.py; raw annotations, the adapters' projection).
  - nuScenes new test: 12,464 objects of barrier / traffic cone / construction vehicle; 730 of them in the ego corridor, on 373 frames.
  - AV2 val: 29,210 cones / barrels / bollards; 1,718 in the corridor, on 533 frames.
- 2026-10-16 00:40 — **V6 (i) centerlines:**
  - k30_nav_hybrid.py (stage center, md5 f870c63e) takes PDM-Closed's route centerline per navtest token, relative to the scenario's rear-axle pose (nuPlan's ego frame = NAVSIM's annotation frame, lidar2ego = I).
  - Checked on one log: centerline straight ahead along the ego lane, |y| ≤ 0.6 m up to 60 m. (A first version used ego2global and put it 16 m to the side; fixed before the full run.)
  - 12 CPU shards launched.
- 2026-10-16 03:45 VUW — **throughput.**
  - run_when_gpu_free.sh serialises the final pick through one lock, which every waiting wrapper holds ≥ 60 s per attempt. That lock is shared with DEEP's B2D wrappers, so ~20 waiting K30 wrappers start one by one. Started so far: match nuScenes g0 / g1, match AV2 g0 / g1, idm nuScenes (223 / 228 units), prior.
  - The GPUs are 73–100% utilised by other users, so all our processes time-share them. Nothing to fix; the rest start as memory allows.
- **V6 (ii)** is mostly in K20 (handoff/K20/pdm_variants.csv, pdm_cost.csv). What K20 lacks is MEASURED (not derived) runtime of the 5×5 / 3×3 / corridor variants.
  - remote/A_navsim/k30_probe_time.py (md5 ed77ccdf) times each variant with K20's identical-output fast implementation on K20's 1,000 tokens (12 single-thread processes, as K20), plus the time inside PDM-Closed's compute_planner_trajectory.
  - Queued after the centerline stage (k30_probe_chain.sh).
- **V6 (i) chain queued** (k30_nav_chain.sh): detect hybrid_pdm + native (GPU) → PDMS of spec_pdm / hybrid_pdm / native with navsim_pdms.py → paired log bootstrap. The detect stage is being smoke-tested on one log first (nav_detect_smoke.log).
- **V8h (P2) feasibility:** carla_garage ships leaderboard/data/longest6.xml (36 routes, scenarios inline) and its own leaderboard used for PDM-Lite on Longest6. Running K6's perception_pilot there is plausible but long (36 routes × 4 branches × 3 runs, routes of ≈ 2 km); left for after P0 / P1.
- 2026-10-16 04:00 VUW — **user: "cứ tự tải và chạy đi, tôi approve hết miễn là cần thiết"** (download and run whatever is needed). This lifts R4 for public downloads and for re-runs.
  - **What the agent still cannot do:** registrations and applications in the user's name.
    - ZOD: access by e-mail application.
    - Waymo Open: Google-account licence acceptance.
    - ONCE: registration form.
    - V8f stays blocked until the user registers / applies. Everything public is being done.
  - **NAVSIM v2.2 navhard two-stage** (HuggingFace OpenScene, the devkit's download_navhard_two_stage.sh; scene pickles 0.2 GB, current sensors 12 GB, history sensors 24 GB) → datasets/navsim/navhard (scripts/A_navsim/k30_get_navhard.sh). The scene pickles are done; the sensors are downloading.
  - **CODA base-val-1500** (Google Drive link of coda-dataset.github.io, 477 MB; gdown 6.4.2 installed in the venv, pip dry-run: no numpy pin) → datasets/coda/CODA/base-val-1500. 134 nuScenes images with sample tokens, 5,937 boxes in 35 categories (barrier, bollard, traffic cone, debris, …). V8g runs on the CODA frames that fall in nuScenes new test.
  - **KITTI test sequences 0007, 0008, 0009, 0017, 0018, 0019** (+ calib, label_02, oxts) copied from the Jetson to VUW datasets/kitti_tracking (our own copy, no new licence), for V8c on VUW.
    - remote/A_method/k30_kitti.py: K28's shared KITTI configuration in torch, every refiner applied with camera flag 0.0 AND 1.0.
    - It answers V8c (KITTI 2D AP) and re-runs the K28 KITTI table with the correct flag (the K28 open item).
  - **V4 in-loop metrics: a 4th Bench2Drive run with a logging agent** (remote/A_closedloop/perception_pilot_log.py, md5 312bc36a: PerceptionPilot unchanged, plus per-frame logging).
    - Logged per frame: brake / throttle, PDM-Lite hazard flags and the hazard actor (phantom if id ≥ 10,000,000), every real actor ahead in the ego corridor within 60 m with its TTC, in-camera and hidden flags.
    - Metrics: recall of actors within 3 s TTC, hard brakes (brake ≥ 0.5) per km and the share caused by a phantom (= unnecessary), minimum TTC per route.
    - Smoke on route 1 (ours_v6, privileged) is queued; then 6 branches × 219 routes with MAXJ 2.
- **Pre-registration addendum (V8g, written before any CODA scoring):**
  - The CODA frames are the base-val-1500 nuScenes images whose sample token lies in nuScenes new test.
  - A CODA box (2D only) is "in the corridor" if its ground-contact lift overlaps |y| ≤ 1.2 m at 0 < z ≤ 80 m. The lift puts the bottom edge on the ground plane using the scene camera's height, and takes the lateral extent from the box's left / right edges.
  - Recall = any output box of the pipeline at class-agnostic IoU ≥ 0.3 (merged scene, conf ≥ 0.25, V7 apdump).
  - G of every pipeline on those frames comes from the K28 instr records.
  - Script: remote/A_method/k30_v8eg.py, which also does V8e.
- 2026-10-16 04:30 VUW — **V8a plan (NAVSIM v2 EPDMS).**
  - The installed v2.2 devkit (third_party/navsim) imports in the nuPlan env.
  - The K20 perception-in-the-loop PDM-Closed is wrapped as a v2 agent (remote/A_navsim/k30_nav_v2agent.py). Per token it uses the stored CAM_F0 detections, applies navsim_pdms.branch_annotations, plans PDM-Closed exactly as navsim_pdms.plan, and returns 8 poses at 0.5 s. The devkit's own scorers then give EPDMS and the sub-scores: run_pdm_score_one_stage.py for navtest, run_pdm_score.py two-stage for navhard.
  - **v2 metric caches** for navtest and navhard_two_stage (k30_nav_v2cache.sh; 12 processes) are queued after the V6 (ii) timing run and the navhard download.
  - **Branches on navtest:** reference, cheap, full960, native, spec_pdm (= v6-S's role: the probing field), hybrid_pdm.
  - **navhard:** its 5,462 synthetic second-stage scenes need fresh detections (and probing for spec_pdm) on the synthetic images. That stage is written after navtest works.
  - datasets/navsim/navhard_two_stage is linked to the download, as the devkit's paths expect.
- 2026-10-16 05:20 VUW — **operational: CUDA out-of-memory on GPU 0.** About 27 K30 processes (0.7–1.3 GB each) had landed there, and cons AV2 shard 0/6 died in a whole-frame detection. It was relaunched with the same command (MIN_FREE_MB 5000, log cons_av2_g0_r1.log); finished units are skipped. The hourly check now has a K30 acknowledgement list.
- **Prior-work re-runs DONE, parity exact.** nuScenes SAHI s1.0: J_c and J_sahi_s10 identical to the stored run on 9,181 / 9,181 frames. AV2 FOVEA KDE_I 960 (ego mask): J_c and J_fovea identical on 11,818 / 11,818.
- **V6 (ii) timing DONE** (04:56); the tables come at report time. The navhard download is DONE (04:35), so the v2 metric caching has started.
- **KITTI copy DONE**; k30_kitti.py launched (logs/k30/kitti.log).
- 2026-10-16 05:32 VUW — **V6 (i) detect smoke OK** (one navtest log, 79 tokens).
  - Hybrid crop area 0.833 of the image vs spec_pdm's 0.829 on the same tokens, so the frame matching holds.
  - The tube reaches a mean half-width of 9.3 m, because spec_pdm's crops cover ≈ 83% of CAM_F0 there; every token got a tube.
  - Native whole frame: 14.3 detections per token. The full detect stage (k30_nav_chain.sh) is queued.
- 2026-10-16 06:30 VUW — **V8a:**
  - The v2 navtest metric cache is DONE.
  - **The v2 agent works:** the devkit's run_pdm_score_one_stage.py scored 79 / 79 tokens of one navtest log with the reference branch, ≈ 1.1 s per token single-threaded.
  - **Full navtest EPDMS launched** (k30_nav_v2score.sh; 12 processes per branch): reference, cheap, full960, spec_pdm from dets_final, then hybrid_pdm and native once dets_k30 is complete. The navhard metric cache is being built.
- 2026-10-16 08:10 VUW — **operational: a silently killed shard.** match nuScenes shard 1/6 stopped after 19 of its 38 scenes: no traceback, so it was most likely the host's OOM killer, and the group log still said "exit 0" because its `wait` hides per-process status. It was relaunched with the same command (log match_nusc_s1_r1.log; finished scenes skipped).
  - The hourly check now lists every part / dataset / shard with missing units and no live process (tmp/k30_shards.py on VUW).
  - All other shards were alive at 08:10.
- **Also DONE:**
  - speed nuScenes (V8d), 228 / 228;
  - idm (V2b) on all three datasets;
  - axis nuScenes; cons nuScenes;
  - KITTI (k30_kitti.py, 6 sequences, exit 0);
  - navtest EPDMS reference branch;
  - dets_k30 complete (137 logs).
- 2026-10-16 09:50 VUW — **first results (post-hoc; tables in results/k30/tables/ on VUW).**
  - **nuScenes only (k30_analyze.py, 228 / 228 units; AV2 / Argoverse-HD still running):**
    - V1a (refiner v3a, m = 1.0): v6-S G 0.211 is above unc2 / unc4 / tcr0 / tcr3 / cor0 / cor8; tsp2 (+0.097 [−0.005, 0.234]) and tsp8 (+0.037 [−0.012, 0.096]) n.s. At m = 1.5 most rows are n.s.; tcr3 0.215 (v6-S − it −0.004). The heuristics' realised px are 1.2–2.2 × v6-S's at the same cap, because the guard grows their crops.
    - V1b axis: no (crop side, context) point differs from v6-S (−0.037 … +0.020, all n.s.).
    - V1c TTC consumer: v6-S G2 0.153; hybrid with braking support 0.137 (+0.016 n.s.), with the TTC consumer's support 0.146 / 0.145 (+0.007 / +0.008 n.s.).
    - V2b IDM: frozen v6-S 0.086, ours-IDM 0.030; v6-S above ours-IDM, the hybrid and unc. (AV2 v6-S 0.156 vs native + C1 0.196 n.s.; Argoverse-HD v6-S 0.313, above u960 and unc — partial units, to be re-run.)
    - V2 tube: 4.6% of v6-S's field footprint lies outside the hybrid's tube; on the 1,583 frames whose decisive object is outside the tube, G 0.033 vs 0.026 (+0.006 n.s.), decisive recall 0.757 vs 0.747.
    - V2 strata (v6-S − hybrid tsp8 @ v6-S px): significant only in the low |yaw rate| tercile (+0.256 [0.005, 0.434]) and in KEEP frames (+0.012 [0.001, 0.025]); > 60 m the hybrid is ahead in point estimate (−0.084 [−0.217, 0.041]).
    - V3 causes of the 20 costliest safety-critical v6-S frames: ttc / lateral error with the decisive object found 10, decisive object inside a crop but missed 6, range error 2, decisive object outside every crop 2.
    - V5b / V5c consumer parameters: v6-S ranks 2nd of 9 in every setting except tr0.6 (1st); native + C1 + v3a is 1st elsewhere, never significantly. v6-S stays significantly above unc2 and unc4 in every setting, and above v5.1-S in 4 of 5 (τ 1.2: +0.007 [−0.005, 0.021]).
    - V5 probe (probe − frozen v6-S, same consumer): vc5 +0.025 [−0.001, 0.065] n.s.; **vc10 +0.026 [0.0003, 0.066], barely significant: a probe variant above the frozen v6-S (post-hoc; v6-S unchanged, R1)**; ped −0.286 [−0.515, −0.035]; truck / union −0.006 [−0.031, 0.024].
  - **V8d ego speed (k30_analyze.py v8d, md5 25309105; v8d_speed.csv; nuScenes 228 units).** Safety-critical frames 301 / 928 / 2,016 at × 0.5 / 1 / 1.5.
    - × 1 reproduces the base rows (v6-S 0.211, v5.1-S 0.199, hybrid 0.174).
    - × 0.5: every difference n.s.; v6-S 0.241, hybrid 0.237, u960 + C1 + v3a 0.279, native + C1 + v3a 0.318 (v6-S − native −0.077 [−0.156, 0.032]).
    - × 1.5: v6-S 0.196 is above the hybrid (+0.043 [0.003, 0.087]) and v5.1-S (+0.006 [0.001, 0.014]); vs u960 / native + C1 + v3a n.s. (+0.011 / +0.007).
    - The field grows with speed: nonempty on 15% / 50% / 57% of frames, farthest sensitive range 14 / 28 / 43 m, field crop area 0.017 / 0.056 / 0.065 of the image. The hybrid's crop area is 0.027 / 0.073 / 0.088. (The tube area itself was not stored by the speed part; the hybrid's crop area, which lies inside its tube, stands in for it.)
  - **V6 (ii) measured probing time (k30_probe_an.py, md5 a9fca18a; results/navsim_exp/k30_probe_time/v6ii_probe_time.csv; 1,000 tokens, 125 logs).**
    - Calls per frame 50.2 / 26.2 / 14.0 / 14.4 for 7×7 / 5×5 / 3×3 / plan corridor (support calls 4.0 in each).
    - Wall clock relative to 7×7 on the same tokens (paired log bootstrap): 5×5 0.511 [0.468, 0.559], 3×3 0.322 [0.282, 0.369], corridor 0.332 [0.303, 0.367].
    - Absolute times are inflated by host load: 7×7 measured 14.8 s mean (p95 23.9 s), against 6.2 s for the same fast implementation in K20. K30's GPU jobs were running on the host at the same time. The ratios are the measurement; × K20's 6.2 s they give ≈ 3.2 / 2.0 / 2.1 s.
    - Planner share of the wall clock: 88% (7×7) to 78% (3×3).
  - **KITTI (k30_kitti_an.py, md5 b5e1e145; v8c_kitti_cells.csv, v8c_kitti_ap.csv, v8c_kitti_ap_diff.csv).**
    - **Parity with K28's Jetson table (flag 1.0, K28's 3,336 frames):** torch reproduces TensorRT within 0.005 on every row. Examples (mean of 4): ours 0.240 vs 0.235, native + comp 0.368 vs 0.370, u960 0.244 vs 0.241, ours_noref 0.230 vs 0.228.
    - **With the flag set as trained (0.0), Planner B mono comes back** for every refiner row (v6-S 0.355, native + T4 + ref 0.381, vs ≈ 0 with flag 1.0). This answers the K28 open item: the flag bug caused the near-zero Planner B mono cells.
    - **Corrected table (flag 0.0, all 3,536 frames), mean of 4:**
      - v6-S 0.323 (px 0.167);
      - native + T4 + ref 0.448 (v6-S − it −0.126 [−0.175, −0.021]); native + C1 + ref 0.439 (−0.116 [−0.182, −0.031]);
      - u960 + T4 + ref 0.368 (−0.046 [−0.088, 0.007]); u960 + C1 + ref 0.367 (−0.044 [−0.091, −0.011]);
      - native plain 0.327 (−0.005 n.s.); u960 plain 0.236 (+0.086 [0.018, 0.162]).
      - The K28 verdict "below whole frame + components" holds with the correct flag; both sides rise by about the same amount. Six sequences: wide intervals.
    - **V8c KITTI 2D AP (R40, all 3,536 frames), moderate:**

      | pipeline | Car | Pedestrian | px |
      |---|---|---|---|
      | cheap | 51.6 | 56.4 | 0.066 |
      | u960 | 75.0 | 73.6 | 0.594 |
      | native | 75.3 | 76.0 | 1.029 |
      | v6-S | 42.9 | 53.9 | 0.167 |

      - v6-S − native: Car −32.4 [−38.1, −24.8], Pedestrian −22.0 [−60.6, −15.6]; v6-S − cheap: Car −8.7 [−13.7, 5.2], Pedestrian −2.5 [−3.8, −0.2].
      - Cyclist AP ≈ 0 for every pipeline: COCO has no cyclist class (rider and bicycle are separate boxes), so the class is not measurable.
      - Interval method: sequence bootstrap with thresholds snapped to a 0.0025 grid; the full-sample grid AP is within 0.2 of the exact devkit value.
      - **Mechanism (diagnostic, tmp/k30_kitti_diag*.py on VUW):** v6-S covers more labelled cars at IoU ≥ 0.7 than the draft (0.714 vs 0.616 of moderate cars), but 34% of its vehicle boxes (conf ≥ 0.25) are crop-truncated parts lying inside a labelled vehicle box with IoU < 0.7, against 17% for the draft and 14% for native. KITTI's near cars are large in a 375-px-high frame, and the crops cut through them. At Car IoU 0.7 these parts are false positives, and the draft's full box is gone, because its centre lies inside the examined region.
  - **V4 logging agent changed before its smoke started** (perception_pilot_log.py, md5 312bc36a → 0613eb6c; driving unchanged).
    - The leaderboard calls setup() without a route index, so the old record had no route id.
    - Each record now holds the route's town and start point (the first global-plan point); the analysis matches them to the routes file.
- 2026-10-16 10:00 VUW — **V4 full logging run prepared** (not launched; waits for the smoke).
  - remote/A_closedloop/k30_b2d_log_dispatch.sh (md5 ffa2cc51): b2d_dispatch_v6.sh with the logging agent; rep2 routes (219); branches ours_v6, cheap, u960, u1600, ours, privileged; MAXJ 2.
  - In-loop analysis: remote/A_closedloop/k30_b2d_inloop.py (md5 9caa538b). Route matching by town + start point (≤ 5 m); hard brakes per km split into unnecessary (phantom hazard) / real actor / rule / other; recall of in-camera corridor actors within 3 s TTC; min TTC per route; paired route bootstrap of ours_v6 − each branch.
- **V8a navhard pipeline written** (remote/A_navsim/k30_navhard.py, md5 ae9f1d31; NAVSIM v2.2 SceneLoader with navhard_two_stage's filter: 450 stage-one real scenes + 5,462 reactive synthetic stage-two scenes, which equals the 5,912 metric-cache entries).
  - The same branch code as the navtest runs, applied to these scenes:
    - detect1 (GPU): cheap / full960 / native on each token's current CAM_F0;
    - field (CPU): navsim_field.py's probing on the cheap draft, plus PDM-Closed's route centerline;
    - detect2 (GPU): spec_pdm (spec_pdm_dets, SPEC_CROP_SCALE 0.5) and hybrid_pdm (support crops + tube at the spec_pdm area);
    - split: per-log files for the agent.
  - Scoring: the devkit's run_pdm_score.py two-stage with k30_nav_v2agent (+agent.dets_dirs=navhard_dets).
  - Smoke: 6 tokens (k30_navhard_smoke.sh, folders *_smoke), queued on the GPU.
- 2026-10-16 10:45 VUW — **Bug found in the V6 (i) hybrid's tube; fixed; hybrid_pdm re-run.**
  - **Bug.** k30_nav_hybrid.py (detect, md5 f870c63e) placed the tube at y = interp(x) over ALL route-centerline points with 0 ≤ x ≤ 100 m, sorted by x. On a turning route this mixes the route ahead with later parts of the route that come back into that x range.
  - **Extent.** On 39% of navtest tokens, a tube point lay more than 1 m off the route ahead (75th percentile of the worst point: 83 m). So the V6 (i) hybrid_pdm detections and PDMS in dets_k30 / pdms_k30 are **superseded**. spec_pdm and native are unaffected.
  - **Fix** (k30_nav_hybrid.py md5 01acda75: forward_path + hybrid_dets, stage rehybrid). The tube follows the route ahead: from the centerline point nearest the ego until the route turns back (x below its running maximum by more than 0.5 m). The rest of the hybrid is unchanged.
  - **Re-run.** rehybrid writes dets_k30b (the dets_k30 records with hybrid_pdm replaced). Then PDMS (navsim_pdms, hybrid_pdm only) and the v2 EPDMS of hybrid_pdm are re-run on dets_k30b. The running chains finish their hybrid_pdm runs on the old detections; those outputs are kept and labelled superseded.
  - navhard uses the same hybrid_dets (k30_navhard.py md5 7e579cc7).
  - Found by the navhard smoke: three of six tokens got no tube, and their centerlines showed y = 268 m at x = 0.
- **V4 logging smoke, first try: every frame logged an AttributeError** (driving unaffected). AutoPilot._get_control returns (control, driving_data), and the logger read .brake on the tuple. Fixed (perception_pilot_log.py md5 2bd87c16; errors now keep their message); the smoke was relaunched for ours_v6 and privileged.
- **navhard smoke OK** after two fixes:
  - v2.2's MetricCacheProcessor needs proposal_sampling (40 × 0.1 s, = v1.1's fixed value);
  - camera_path is relative to the sensor root (sensor_blobs/test for stage one, navhard_two_stage/sensor_blobs for synthetic scenes), and images are read with cv2.imread as on navtest.
  - Field: 46–53 planner calls per token (navtest 50).
- 2026-10-16 10:50 VUW — **launched:**
  - **Fixed-tube hybrid chain** (k30_nav_rehybrid_chain.sh, md5 7015de6a). Steps:
    - rehybrid on navtest → dets_k30b.
    - Smoke parity on one log: the 78 straight-route tokens are identical to dets_k30, the 54 turning tokens changed, other branches untouched.
    - After the old chain: hybrid_pdm PDMS → pdms_k30b; merge with pdms_k30 (k30_nav_merge.py, md5 5e85e72b; the old rows kept as hybrid_pdm_tubebug); summary pdms_summary_k30m.csv.
    - After the navtest v2 chain: EPDMS of the fixed hybrid_pdm.
  - **navhard chain** (k30_navhard_chain.sh, md5 9407cd36): detect1 → field (8 CPU shards) → detect2 → split; two-stage EPDMS of the six branches after the fixed-hybrid EPDMS.
  - **V4 full logging run** (k30_b2d_log_dispatch.sh, MAXJ 2; perception_pilot_log.py md5 2bd87c16).
    - Smoke OK: route 1711 for ours_v6 and privileged, 0 logging errors, both Completed with DS 100.
    - In-loop script check on the smoke (k30_b2d_inloop.py md5 8bcd99c4): ours_v6 22.6 unnecessary (phantom) brake episodes per km vs 0 for privileged.
    - Rising edges of brake ≥ 0.5 over-count: the controller toggles the brake within one stop, giving 75 / km even for privileged. Episodes separated by > 1 s are added as a sensitivity column (not pre-registered); the pre-registered rising-edge count stays.
  - V7 Argoverse-HD (apdump complete); V5 prior (k30_v5_prior.py).
  - k30_analyze.py on nuScenes + AV2 complete (all parts) and Argoverse-HD (all but match, 18 / 24).
- **V8e / V8g preview** (k30_v8eg.py; v8eg_corner_coda.csv):
  - nuScenes 373 corner-case frames / 730 objects; AV2 533 / 1,718.
  - **Recall of the corner-case objects ≈ 0 for every pipeline:** nuScenes 0.03–0.04 (barrier 0–0.016, cone 0–0.024; construction vehicle 0.22–0.58, i.e. COCO truck); AV2 0.008–0.013. As pre-registered: the COCO detector has no such classes.
  - G on those frames: nuScenes v6-S 0.016, hybrid 0.230 (−0.215 [−0.605, 0.000]), u960 + C1 + v3a 0.106, native 0.064, all n.s.
  - **AV2: v6-S 0.154 < hybrid 0.199 (−0.045 [−0.107, −0.009])**; u960 / native n.s.
    - The AV2 difference is spread over the action classes: of the 16.0 cost units, KEEP 7.6, DECELERATE 5.1, HARD_BRAKE 3.3.
    - v6-S has 2 more unnecessary hard brakes and 2 more under-reactions than the hybrid on these frames.
  - V8g CODA: only 27 CODA images fall in nuScenes new test (119 objects, 4 in the corridor), too few for a conclusion. Recall of all CODA objects 0.02–0.08; G on the 27 frames: v6-S 0.633, the others 0.143.
- 2026-10-16 11:10 VUW — **V8a navtest EPDMS:**
  - Done: reference, cheap, full960, **spec_pdm (12,146 / 12,146 scenarios valid, EPDMS 0.8647)**.
  - hybrid_pdm and native **failed at once in Hydra**: "Ambiguous value" for the comma-separated +agent.dets_dirs=dets_k30,dets_final. Nothing was scored; the log line said exit 0.
  - The fixed-tube hybrid step of k30_nav_rehybrid_chain.sh has the same comma and will fail the same way (harmless).
  - k30_nav_v2score2.sh (md5 c080024e) re-runs native (dets_k30) and the fixed hybrid_pdm (dets_k30b) with one folder each; both folders hold complete records. It runs after the navhard chain, so only one 12-process scorer runs at a time.
- 2026-10-16 11:15 VUW — **V6 (i) old chain DONE** (09:41; pdms_summary_k30.csv, PDMS v1 on 12,146 navtest tokens): spec_pdm 0.8945, native 0.8952 (spec_pdm − native [−0.0026, +0.0012]), hybrid_pdm with the faulty tube 0.8942 ([−0.0007, +0.0015]). The fixed-tube hybrid is being re-run (rehybrid chain).
- **V3 panels:** the capture re-run for the 60 top-cost frames (results/k30/v3_capture_frames.csv, md5 8aef766e: 11 nuScenes scenes, 13 AV2 logs, 3 Argoverse-HD sequences). It uses k28_p3.py --part instr --capture … --tag k30cap (md5 8b791fa2, unchanged; AV2 through k10_masked_run), queued on the GPU. Rendering with k28_render.py follows.
- REPORT.md draft written (handoff/K30/REPORT.md). The not-above collector is k30_not_above.py → not_above_all.csv / not_above_counts.csv.
- 2026-10-16 11:45 VUW — **V4 logging run: 5 evaluators had started at once** (ours_v6 chunks 0–3, cheap chunk 0), against MAXJ 2.
  - Cause: k30_b2d_log_dispatch.sh (like b2d_dispatch_v6.sh) counted only started evaluators, so launches queued in run_when_gpu_free.sh all started when GPUs freed.
  - The five keep running (RAM 38 GB available, no GPU OOM). The 6th CARLA server on the host is the DEEP session's (deep_ext/b2d) and was not touched.
  - Dispatcher replaced (only its PID 3765992 stopped; evaluators untouched) by k30_b2d_log_dispatch2.sh (md5 f10eee3d). It counts every (branch, chunk) job of this run, including those still waiting for a GPU, so no new job starts until fewer than 2 are active.
- Kit updated: 03_results §8 (K30 interim bullet), 06_pending (K30 remaining items), QUEUE (K30 row), results/INDEX.md (K30 row).
- 2026-10-16 12:25 VUW — **V6 (i) DONE (fixed-tube hybrid; pdms_summary_k30m.csv, PDMS v1, 12,146 navtest tokens, merge 0 missing):** spec_pdm 0.8945, hybrid_pdm 0.8943 (spec_pdm − hybrid [−0.0008, +0.0015]), native 0.8952 ([−0.0026, +0.0012]), the old faulty-tube hybrid 0.8942. All n.s.: on NAVSIM, PDM-Closed's score does not separate the field from a tube or from the native whole frame. The fixed-hybrid EPDMS step of the rehybrid chain failed in Hydra as expected; k30_nav_v2score2.sh re-runs it after navhard.
  - navhard field: 100 / 739 tokens per shard after ≈ 1 h (host load: 5 CARLA servers); ≈ 6 h more. V3 capture waits for a GPU with ≥ 9 GB free.
- 2026-10-16 14:25 VUW — **V3 panels DONE.**
  - Capture run (k30cap) on 11 + 13 + 3 units; k28_render.py drew 60 frames × panels a–f (PDF + PNG) plus examples.csv into results/k30/panels/, copied to handoff/K30/panels/.
- **V4 logging run, operational.** The five first evaluators (ours_v6 chunks 0–3, cheap chunk 0) all stopped between 13:30 and 14:11 with "Failed - Simulation crashed" (RuntimeError: time-out waiting for the simulator), after 22–25 routes each.
  - The checkpoints are resumable. dispatcher v2 now runs cheap chunks 1 and 2 (2 jobs) and will relaunch the stopped chunks with RESUME, as in K6 / K28.
  - In-loop records keep the last record per route, so a re-run route replaces a crashed one.
- 2026-10-16 19:25 VUW — **navhard field DONE**: 5,912 tokens, 0 errors, 49.0 planner calls per token. detect2 (spec_pdm, hybrid_pdm) is running (shard 0 at 1,000 / 2,956). Then split, then the two-stage EPDMS of six branches.
  - V4 logging run: 210 / 1,314 routes (2 jobs; the dispatcher's first pass reaches u960; the crashed chunks resume in the next pass).
- 2026-10-16 20:40 VUW — **navhard scoring: the devkit's aggregation crashes, so no output was written. Replaced by a wrapper.**
  - detect2 and split are DONE (5,912 tokens in 76 logs).
  - run_pdm_score.py scored every token, but about 100 of the 5,462 synthetic stage-two scenarios fail inside the devkit's IDM traffic agents ("Agent's baseline does not intersect the agent itself"). This happens for the privileged reference too, so it is not our agent.
  - A failed token gets NaN results. The pseudo-closed-loop aggregation then raises, its fallback raises again (TypeError on NaN metric arrays), and the job dies before writing its CSV. The reference and cheap runs (≈ 1 h each) were lost this way.
  - Stopped the old navhard chain (exact PIDs: chain 3774503, scorer 2189556 + its 12 workers).
  - **k30_navhard_score.py** (md5 5b7450f4) runs the devkit's run_pdm_score.run_pdm_score unchanged through worker_map. It pickles the per-token rows first (raw_rows.pkl), then applies the devkit's own aggregation functions with the failed tokens left out (a pair containing a failed token; a group whose stage-one token failed).
  - **k30_navhard_agg.py** (md5 923063f6) re-aggregates every branch two ways: on its own valid tokens, and on the common token set (the union of all branches' failures left out). The common set is the comparison to report.
  - k30_navhard_chain2.sh (md5 a12122d3) runs the six branches, then the aggregation, and finally writes "K30 NAVHARD DONE" for k30_nav_v2score2.sh.
- 2026-10-16 21:30 VUW — **navhard reference scored** (raw_rows.pkl, 5,912 rows, 100 failed in the devkit's IDM).
  - The first version of aggregate() raised ZeroDivisionError in the devkit's SceneAggregator: a group was left with no usable pair.
  - Fix (k30_navhard_score.py md5 01556765): a group is kept only if both of its stage-one tokens are valid and at least one pair has both tokens valid.
  - Offline re-aggregation (k30_navhard_agg.py): reference uses 222 of the groups. EPDMS combined 0.528, stage one 0.813, stage two 0.648.
  - The cheap run started with the old aggregate; its raw rows are saved, and the final k30_navhard_agg.py step re-aggregates every branch from raw rows anyway.
- 2026-10-17 00:20 VUW — **V4 logging run: MAXJ 2 → 4 (user approved).** GPUs 0 / 1 had ≈ 24 / 19 GB free; the jobs still go through run_when_gpu_free.sh (≥ 9 GB free). The dispatcher was restarted (only its PID); running evaluators were untouched. DEEP's E3a Bench2Drive ladder also runs on VUW (rep 1 875 / 880 finished, rep 2 447 / 880, rep 3 not started); it was not touched.
- 2026-10-17 00:40 VUW — **V8a navhard DONE** (k30_navhard_agg.py; k30_v2/navhard_summary.csv). The same 100 synthetic tokens fail in every branch (devkit IDM), so the common token set equals each branch's own; 222 groups are used.
  - Two-stage EPDMS (combined / stage one / stage two): reference 0.528 / 0.813 / 0.648; cheap 0.508 / 0.806 / 0.626; full960 0.508 / 0.808 / 0.625; native 0.510 / 0.806 / 0.629; **spec_pdm 0.509 / 0.808 / 0.626**; hybrid_pdm 0.510 / 0.807 / 0.629.
  - No intervals: the devkit's two-stage score is not a per-token mean.
- **UNATTENDED MODE (user, 2026-10-17: quota running out — let the tasks finish automatically and safely).** Everything below runs without a session:
  - VUW k30_b2d_log_dispatch2.sh (MAXJ 4): V4 logging run; relaunches crashed chunks with RESUME, ≤ 20 attempts each.
  - VUW k30_b2d_log_watchdog.sh (md5 55cbd0ab): stops an evaluator of this run only (exact PIDs of it and its CARLA descendants) when its log is idle for more than 60 min. It writes logs/k30/b2dlog/watchdog.log and ends with the run.
  - VUW k30_nav_v2score2.sh: navtest EPDMS of native and the fixed hybrid (it started after "K30 NAVHARD DONE").
  - VUW k30_finalize.sh (md5 7177af96): after the navtest re-scoring → k30_nav_v2_summary.py (navtest_summary.csv: EPDMS per branch, spec_pdm − each with a paired log bootstrap). After the Bench2Drive run → k30_b2d_inloop.py (results/k30/b2d/inloop_*.csv). Then logs/k30/final_done.log "K30 ALL DONE".
  - Jetson scripts/k30_sync_loop.sh: every 2 h copies the tables into results/vuw_summaries/k30 and handoff/K30 (k30_sync.sh). It stops after "K30 ALL DONE".
- **For the next session:** fill REPORT.md §1 / §2 rows V4 in-loop (inloop_summary.csv) and V8a (navtest_summary.csv, navhard_summary.csv), and rerun k30_not_above.py (add the V4 in-loop and navtest intervals). Update 03_results / 06_pending / QUEUE (K30 DONE except J).

## K30-DEV pre-registration (written 2026-10-17 00:20 VUW, before any run; user: "improve ours while quota remains")
- **Why.** In K30, 46 of the 60 costliest v6-S frames are lift errors on a found object (V3). Refiner v3a raises the below-required (under-reaction) share on AV2 / Argoverse-HD from 0.029 / 0.024 to 0.039 / 0.052 (V3 taxonomy). The closing-speed probe vc10 beat the frozen probe post-hoc on nuScenes test (V5c), which is not usable for selection.
- **Candidates** (remote/A_method/k30_dev.py, part devref; no new training):
  - Refiner variants on the same detections and supports:
    - v3a_n15 / v3a_n30: an object the refiner moves farther keeps its unrefined geometry when its unrefined range is < 15 / 30 m;
    - v3a_hb: an object whose unrefined demand ≥ thr_hard keeps its unrefined geometry when the refiner lowers its demand.
  - Probes: v6-S with vc5 / vc10.
  - Every refiner variant is also applied to u960 + C1 and native + C1 (R2 fairness).
- **Data: DEV ONLY.** nuScenes val (IE.units_nusc("val")) and AV2 CFG-DEV (ego mask, k10_masked_run). Argoverse-HD has no dev split. No test unit is touched.
- **Selection rule (fixed now):**
  - Score: macro mean over the two dev datasets of G (refiner applied, cost as K28).
  - A combination (refiner variant × probe) is adopted only if it beats deployed v6-S (v3a, frozen probe) by ≥ +0.005 macro, and loses ≤ 0.005 on neither dataset.
  - Among qualifying combinations, the largest macro gain wins; otherwise v6-S stays.
  - Unit-bootstrap intervals are reported, but do not select.
- **If adopted.** "v6.1-S" is run ONCE on the three test splits with the same frozen protocol as K28 (all baselines with the same refiner variant). It is reported as a post-hoc revision next to the frozen v6-S, never replacing its numbers.
- 2026-10-17 00:33 VUW — **K30-DEV launched** (unattended).
  - Smoke OK: k30_dev.py md5 8112acb3, part devref, nuScenes val scene-0108. All 25 J columns are present; on that scene no variant changes an action, as expected when no object is in play.
  - Chain k30_dev_chain.sh (md5 0af6c7ae): nuScenes val and AV2 CFG-DEV, 3 shards each, two passes, then k30_dev_an.py (md5 923f53b6).
  - Outputs: results/k30/tables/dev_devref.csv and dev_devref_selection.txt (the pre-registered rule applied); logs/k30/dev_analyze.log ends with "K30 DEV DONE". The Jetson sync loop copies both into handoff/K30.
  - **Next session:** if dev_devref_selection.txt says SELECTED: <variant>, ask the user before the single test run of v6.1-S (pre-registration above). If it says none, v6-S stays and this is reported as a negative dev result.
- 2026-10-17 01:15 VUW — **V4 logging run split over two hosts (user: "tối ưu cho 2 cái").**
  - VUW1 = cuda13.ecs.vuw.ac.nz: same NFS (/vol/dl-nguyenb5-solar shared; checked with a probe file), same venv (torch 2.6, 3× RTX A5000, ≈ 14 GB free each; another user holds ≈ 10 GB per GPU).
  - Dispatcher v3 (k30_b2d_log_dispatch3.sh, md5 0f2fb4c6; per-host dispatch log):
    - VUW (cuda12): BRANCHES ours_v6 cheap u960 u1600, MAXJ 3 (one slot left for K30-DEV);
    - VUW1 (cuda13): BRANCHES ours privileged, MAXJ 3, plus its own watchdog instance (the watchdog only sees its host's processes).
  - The old dispatcher2 was stopped by its PID; running evaluators were untouched.
  - k30_b2d_log_join.sh (md5 eb7418a1) writes "K30 B2D LOG ALL CHUNKS COMPLETE" to b2dlog/dispatch.log once both hosts are complete. That marker is what k30_finalize.sh and the watchdogs wait for.
  - Messaged the DEEP session (risk-aware-perception-fb) about VUW1, and about its three E3a evaluators on VUW stuck in simulator time-outs (not touched).
  - VUW1 smoke OK: ours chunk0 runs (CARLA up; agent ticking at game/wall 0.126). DEEP replied: it stopped its two hung route_11755 evaluators by PID; its watchdog now also stops evaluators whose CARLA has been gone for 10 min; it leaves VUW1 alone until its user decides, and asks us to keep our VUW1 MAXJ as it is (3).
- 2026-10-17 02:25 VUW — **V8a navtest DONE** (navtest_summary.csv): spec_pdm 0.8647; cheap 0.8624 (spec − it +0.0023 [−0.0001, 0.0048]); full960 / native 0.8644 (+0.0003 n.s.); hybrid fixed 0.8653 (−0.0006 n.s.); reference 0.8698 (−0.0051 [−0.0068, −0.0035]). REPORT §1 / §2 updated. K30-DEV: nuScenes val (152) and AV2 dev pass 1 done; pass 2 running.
- 2026-10-17 03:30 VUW — **K30-DEV DONE: no candidate qualifies; v6-S stays** (dev_devref.csv, dev_devref_selection.txt).
  - **Analysis bug caught before any action.** The first k30_dev_an.py (md5 923f53b6) passed the arrays to its bootstrap in the wrong order, so it reported G(deployed) − G(candidate) as the candidate's gain. Its output "SELECTED: v6S with refiner v3a_hb" was therefore inverted.
  - The bad output is kept as dev_devref_selection_signbug.txt. Fixed in md5 52269f65, re-run on the same records.
  - Correct dev results (G(candidate) − G(deployed v6-S); nuScenes val 152 scenes, AV2 CFG-DEV 300 logs):
    - Deployed v6-S: nuScenes 0.3212, AV2 0.3087.
    - **The conservative refiner variants are worse:**
      - v3a_n15: −0.0004 / −0.0310 [−0.0468, −0.0157];
      - v3a_n30: −0.0002 / −0.0329;
      - v3a_hb: −0.0006 / −0.0528 [−0.0756, −0.0285];
      - no refiner: −0.0082 / −0.0327.
      - The refiner's demand reductions are worth more than the under-reaction they add, even for near and hard-brake objects.
    - **Closing-speed probes:** vc5 −0.0009 / +0.0083 (macro +0.0037), vc10 −0.0024 / +0.0087 (macro +0.0032). Both are below the +0.005 bar and n.s. The post-hoc nuScenes-test gain of vc10 (V5c) is not confirmed on dev (nuScenes dev −0.0024).
  - Report as a negative dev result: the frozen design is a local optimum among these simple variants.
- 2026-10-17 07:30 VUW — ours_v6 chunk 0 ended at 54 / 54 with its last route 'Failed - Simulation crashed'. The dispatcher counts the chunk complete and will not re-run that route. k30_b2d_inloop.py (md5 e1528a6e) leaves routes whose final status is a crash out of the summary and prints their count per branch.
- 2026-10-17 16:14 VUW — **K30 ALL DONE.** The V4 logging run completed on both hosts (16:13), and k30_b2d_inloop.py ran (inloop_summary.csv): 212–214 matched routes per branch, 0 crash-ended routes left out.
  - v6-S hard brakes per km: vs cheap −19.0 [−28.4, −9.8]; vs u1600 −53.6 [−78.1, −32.5]; vs u960 / v3 n.s.
  - Unnecessary (phantom) hard brakes per km: vs cheap −2.0 [−4.1, −0.2]; vs u1600 −7.6 [−13.2, −3.3]; vs u960 / v3 n.s.
  - Recall within 3 s TTC: all n.s. (v6-S 0.906).
  - DS of this run: v6-S 93.2, cheap 92.5, u960 92.7.
  - REPORT §1 / §2 updated. Remaining for the next session: rerun k30_not_above.py with the V4 in-loop and navtest rows, and mark K30 DONE in the kit (J waits for K29).
- 2026-10-18 03:15 +07 — **J items (Jetson, after K29).** K29 window C resumed (new supply OK). J1 is partly covered by the window's heuristic runs (h1 tsp2S + C1, h2 tsp8S-yolo11m + C1 at the S budget, 3 reps). The x1.0-budget points of J1, J2 (per-frame energy) and J3 (Argoverse-HD sAP with the fast path) need new code: 64_k29_latency.py has no Argoverse-HD loader and no single-pipeline mode for energy attribution. Not started; the user is asked.
- 2026-10-18 03:30 +07 — **J items written and queued** (user: "làm cho đầy đủ đi"). They run after K29 window C and its report, in the same exclusive window, unattended.
  - scripts/67_k30_jetson.py (md5 49c83f21) = 64_k29_latency.py (bdd820e4, which window C uses unchanged) plus:
    - Argoverse-HD (61.frames_ahd; whole-frame engines 608×960 / 800×1280 / 1216×1920);
    - heuristics at the v6-S budget (--config v6S --heur = V1's ×1.0 budget);
    - HEUR + tsp8S+C1 (the hybrid), cor0S+C1 / cor8S+C1 (k28_test "cor" = the tube only);
    - --only (single-pipeline runs, for J2);
    - per-frame t_wall; run prefix k30j_.
  - scripts/k30_j_window.sh (md5 9a5f7dd7):
    - smoke (40 frames: heuristics at the v6-S budget, --only, Argoverse-HD); it stops on failure;
    - **J1**: tsp2 / tsp8 / tsp8-yolo11m / cor0 / cor8 + C1 + refiner at the S and ×1.0 budgets, 3 reps, nusc_test20 + AV2 (Table 3's frames), fast path trt32;
    - **J2**: energy per frame from single-pipeline runs: cheap, every whole-frame side + C1 + v3a, v6-S, v6-S ≤ 2, v5.1-S alone, hybrid at S and ×1.0 (hybrid energy = its run minus its budget configuration alone). Module rails come from scripts/k29_vin_logger.py;
    - **J3**: Argoverse-HD per-frame timing (cheap, u960 / u1280 / native + C1 + v3a, v6-S, v6-S ≤ 2);
    - then 68_k30_j_tables.py (md5 6175f757) → handoff/K30/J/{j1_latency_G, j2_energy, j3_latency}.csv;
    - and the streaming AP on VUW: remote/A_method/k30_j3_sap.py (md5 b40a6cf6) → J/j3_sap.csv (full sAP, and corridor streaming AP50 with the k30_v7 rule).
  - J3 smoke on VUW with K28's AHD latencies: v6-S sAP 0.0641, identical to K28. The K30 v6-S AHD dump equals K28's.
  - **Deviation:** the ≤ 2-call sAP row uses v6-S's detections with the bounded loop's latency (no dump of the bounded loop).
  - **Not timed in J1:** unc2 / unc4 / tcr0 / tcr3 (their selection was not ported to the Jetson harness: unc needs per-box class entropy from the draft, tcr the previous merged scene per row). tsp / cor are the corridor-type heuristics that tie v6-S in V1a.
  - Logs: logs/k30j/window.log, logs/k30j/runs/. The Jetson stays paused after "K30 J WINDOW DONE".
- 2026-10-18 09:13 +07 — **J DONE** (logs/k30j/window.log: smoke OK 06:38, J1 07:22, J2 07:56, J3 timing 09:01, tables 09:01, sAP on VUW 09:13; 40 runs, 0 failed, 0 resets).
  - 68_k30_j_tables.py → J/j1_latency_G.csv, j2_energy.csv, j3_latency.csv; k30_j3_sap.py (VUW) → J/j3_sap.csv.
  - Added afterwards (Jetson, scripts/py): J/j1_intervals.csv (paired unit bootstrap, 2,000, seed 0, v6-S / v5.1-S − each heuristic, repetition 1; G is identical across repetitions) and J/j2_energy_marginal.csv (energy above the cheap run).
  - Parity: v6-S, v5.1-S, tsp2, tsp8-YOLO11m reproduce K29's G exactly.
  - REPORT §1 (J1–J3 rows), §2 J, §3 (J1 counts), §4, §5; kit 03_results, 06_pending, QUEUE.
- 2026-10-18 09:28 +07 — **Jetson released.** k29_vin_logger (PID 36978) stopped, fan back to auto "cool", ~/.jetson_pause removed (busy.sh no longer "paused"), DEEP (risk-aware-perception-1d) told. jetson_clocks left applied (resets at reboot).
