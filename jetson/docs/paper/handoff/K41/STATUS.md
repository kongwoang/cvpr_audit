# K41 status — 2026-10-30

Inventory and PREREG written before outcomes; available CPU/cached phase completed. No detector/GPU/Jetson jobs launched. **Awaiting the single approval gate in [RUNPLAN](RUNPLAN.md)**, which lists all jobs, resource estimates, commands and unfinished adapters. No experiment is marked complete merely because its plan/code exists.

Cached NAVSIM milestone executed: K20 pair counts exactly reproduced (6/9,449; 11/139,675); K31 has 0/769,561 jointly tested individually quiet probe pairs. Historical selector agreement and PDM PDMS/official EPDMS tables computed by `remote/A_navsim/k41_cached_summary.py`, outputs `$B/results/k41/nav_cached/`. No new planner calls. Triples, new insertion tests, k=2 real crops and selector downstream scores remain pending. Official VeteranAD availability checked; K=20 confirmed. K41-2(b) local learned planner unavailable; see AVAILABILITY.md.

Object CPU milestone executed: all36,005 frames, zero action/pixel parity failures/missing frames; all descriptive strata and crop-origin locality IoU .5/.7 tables/figures generated from final objects_v4. Match rates at .5=.6468/.7572/.7507. Unavailable first-covered-round and per-object originating canvas remain explicit. NAVSIM scene diagnostic completed12,146 tokens/6,041 pairs; matched-bin pairs have differing candidate poses/speeds, so no isolated scene-effect claim. Historical latency conditionals also executed. No new detector/GPU/Jetson job launched.

| ID | Status | Evidence / outstanding |
|---|---|---|
| K41-1 | Completed | P01 selector PDMS (no gain); D04 PiP on NAVSIM (field repair > PiP (a) / (b), PDMS n.s.); D03 PiP (a / b / c) on the braking splits, base + del_all (no row beats v6-S) |
| K41-2 | Completed: P02 (1,000 tokens) + P03 (remaining 11,146 tokens); (b) unavailable | P02 triples / insertion / k2 with joint PDMS; D07 k2 ≡ singleton |
| K41-3 | Completed | strata plus the D05 81-point detector grid (243 jobs; range spacing no-op; no point wins on all 3 splits; AV2 larger-probe caveat) |
| K41-4 | Partial | final-union locality/coverage/conditionals executed; first-round and originating-scale endpoints unavailable in source cache |
| K41-5 | Completed (J04 / J03) | 18 rows, 5 deadlines, 3 repetitions; exact K38 / K34 parity; no row beats DESP after Holm |
| K41-6 | Completed (J02 + JP01–JP03) | stage breakdown: the tail is crop rounds + C1; candidates fail the screens; separate guard-depth profile not done |

Latest CPU checkpoint: controlled scene transplant6,043 pairs/136logs, including2,644 same-action pairs; field changes13.50% and requested crops11.76% in the latter. Train-only anchors use12,219 nuScenes and8,216 AV2 windows. Loop tests60 oracle fixtures/180 strict-cap bounds passed, but do not establish literal K39 strict semantics (CPU_PREPARATION.md). All88 registered protected files remain unchanged. Other new detector/planner runs remain gated; authorized Jetson pilots are recorded below.

2026-10-31 authorized latency pilot milestone: JP01 exact dev-prefix parity passed but DESP p95 feasibility screen failed on both datasets. JP02 isolated C1 bucket1 saving0.123ms/6.42%; conditional JP03 completed: p95 reductions1.19%/0.23%, both CIs include0; no meaningful tail gain. This does not complete K41-5/6 full adaptive/deadline benchmarks. See LATENCY_OPT_20261030/PILOT_REPORT.md and JP02_REPORT.md.

Final authorized Jetson feasibility subset complete. Full K41-5/6 scope remains partial. Controls restored,88 protected hashes unchanged; all other RUNPLAN jobs await authorization.

2026-10-31 execution authorization: user requested all remaining tasks after the Jetson pilot. Completing D01/D02 adapters and checks, then following the consolidated RUNPLAN order. GPU inventory: no device currently satisfies MIN_FREE_MB=12000 on VUW/VUW1; ready jobs will wait via the host-local wrapper. No processes outside B touched.

2026-10-31 shared adapter milestone: D02 strict-cap compatibility fixtures now pass (8 literal K39 cases plus explicit corrected-cap difference); 818 tube geometry and 60 oracle/180 cap checks passed. First-origin traces, actual canvas/cache-hit metadata and ground-truth objects will be collected in D02 for K41-4. GPU parity remains pending.

2026-10-31 D02 origin-trace collection queued on VUW1 after CPU checks; D01 queued on VUW. Both wait through the required wrappers. P01 official scorer import succeeds, but the historical navsim_exp/metric_cache directory is absent; auditing alternate cache availability and bounded reconstruction from original ground truth before scoring.

2026-10-31 P01 metric reconstruction smoke complete:2 lexicographic navtest tokens; exact cheap/full selector parity and exact stored K20 cheap-plan parity. Five cached rows (cheap/full/field/oracle/K30 hybrid) score successfully with official NAVSIM v1.1 PDMS. Repeating the first four rows after adding hybrid reproduces their scores/poses exactly. Smoke02 measured4.401 CPU seconds/5.993 wall seconds for2 tokens, including setup and parity; each temporary metric cache about0.49MB and~1s build wall time. These are smoke measurements, not population quality estimates or a guaranteed full-run runtime. All15 candidate/API and historical PDMS checks precede the full run.

2026-11-01 P01 milestone (complete 2026-10-31 22:37, VUW CPU, 4.05 CPU h): official NAVSIM v1.1 PDMS of the 15-candidate bottleneck selector on all 12,146 navtest tokens / 136 logs (EPDMS not supported for this selector). Cheap 0.5123, full 0.5105, field 0.5108, oracle 0.5103, hybrid 0.5109; field − cheap −0.0015 [−0.0043, +0.0014], field − full +0.0004 [−0.0017, +0.0024], all intervals contain 0. **Adverse for W7: no downstream PDMS gain of the field for this selector; perception quality does not move its PDMS.** Tables in K41/P01/. PiP / geometric rows need D04 (adapter not yet implemented).

2026-11-01 D03 launched: driver `k41_d03_detector.py` (md5 912c57c89c8af32b80e7e48b89299582; copy of the D01/D02 driver with the `pip` part and the K28 deletion option), dispatcher `k41_d03_dispatch.py`, manifest `k41_d03_manifest.json`; definitions registered in PREREG addendum before any outcome. Two 10-frame dev smokes (nuScenes base, AV2 deletion) passed the 10-endpoint frozen parity (K30 match / K28 deletion records). Queues: base on VUW (PID 853905), deletion on VUW1 (PID 2737954); smoke then full N/A/H each, ≤90 GPU h each, 12 GB minimum through the host wrapper.

2026-11-01 P02 launched (VUW CPU, 4 shards, PIDs 858417–858420): `remote/A_navsim/k41_p02_interactions.py` (md5 51e7dc7650552539139d8d5262432fdc), built on `k20_pdm.py --mode pairs`, saved K20 seed-0 1,000 tokens; removal triples of free objects, insertion pairs / triples of individually quiet cells (≤100 per family, SHA256 seed), k=2 adjacent-cell pairs, trajectories of every joint effect kept for PDMS. 2-token smoke: 278–324 planner calls and 82–91 s per token, no joint effect on these two tokens. Measured extrapolation ≈ 24 CPU h (≈ 6 h wall on 4 workers). Deviation noted: the k=2 pairs cover all individually quiet projecting adjacent cells (a superset of the plan-corridor pairs); the plan-corridor subset is selected afterwards from the saved base trajectory.

2026-11-01 19:10 D03 repair: the deletion-setting smoke (VUW1, 14:30) failed the frozen parity on scene-0121 (6 mismatches, J off by exactly 0.05 on frames 5–6 with identical actions). Cause: the jerk/switch term of the cost used the previous action of the *post-deletion* draft; K28 phase 3 uses the action of the unmodified draft (`prev_act = act(ch_raw["mono"])`). Fix in `k41_d03_detector.py` (md5 ece23becfb4b30eb129fbd8faeae5f44; a no-op for `base`), manifest updated, pre-fix smoke output moved to `results/k41/superseded/`. Both D03 dispatchers relaunched (base VUW, del_all VUW1); no D03 GPU work had run yet (the base smoke had waited 4 h 23 min for a GPU with ≥ 12 GB free).

2026-11-01 19:05 D04 launched: adapter `remote/A_navsim/k41_pip_nav.py` (md5 802156c68932379cf3bc167813df81ef; definitions in the PREREG D04 addendum, written before any outcome; one pre-run correction: `geometric` requests support first). 2-token CPU functional smoke ran (field π parity 2/2; px differs slightly on CPU, as expected). Chain `k41_d04_chain.sh` on VUW1 holds the host GPU queue: 5-token GPU smoke (stops unless field π parity holds on all) → full 136 logs, 3 shards on one GPU → P01 PDMS of the four new rows (CPU) → D03 del_all dispatcher. Measured reference: the K31 real-crop field run took ≈ 1 h on one GPU; D04 is expected at a few GPU hours (estimate, not measured).

2026-11-01 13:17 (Jetson clock) J04 window launched: harness `experiments/k41_jetson_20261030/scripts/92_k41.py` (sha256 3bc990c8…, generated by `build.py` from frozen `89_k39.py`), window `window.py`; definitions in the PREREG J04 addendum (written before any timed run). Two 20-frame dev functional smokes (nuScenes val, AV2 CFG-DEV; clocks not locked, not results) ran all 18 rows in one process. Order: dev calibration → medians (DESP/B2K-A/B1K-A + iterated AnA/CF) → 80-frame smokes → 3 rotated repetitions × (nusc_test20, av2t40) → J02 breakdown. Estimated 6–7 h from the K39-1 window durations plus the measured per-frame cost of the new rows (estimate). J03 is covered by the same repetitions; J01 not run (no candidate passed the JP01/JP03 screen).

2026-11-01 20:05 D07 prepared: driver `remote/A_navsim/k41_d07_k2.py` + PREREG D07 addendum (before any outcome). 3-token CPU smoke (field + score): rebuilt cheap plan = P02 draft plan, and the rebuilt-cache PDMS of cheap / spec_pdm equals the original stored `pdms_final` scores exactly on all three tokens. Chain `k41_d07_chain.sh` on VUW1 waits for P02 (95/125 logs at 19:50), then field → GPU detect on changed tokens only (host queue lock) → score (4 CPU shards). Jetson J04: dev calibration done (medians41.json / medians_k41.json written before test), both 80-frame smokes complete, repetition 1 running.

2026-11-01 21:00 (VUW clock) Check. D04: GPU smoke passed (field pi/px/rounds parity 5/5); full run on VUW1 GPU since ~20:08, 32/136 logs, field parity exact on every finished log so far. P02: 112/125 logs (shard 2 finished cleanly; shards 0, 1, 3 running). Interim P02 counts on the first 870 tokens (descriptive only, not final): removal triples 27 moving out of 13,276 tested (7/506 tokens); insertion pairs 3/87,000 (3 tokens); insertion triples 3/87,000 (3 tokens); k=2 adjacent pairs 0/51,249. If the final count stays at 0, D07 k2 ≡ singleton (as pre-specified). The PREREG "PDMS for each joint effect" step was not yet implemented: added `remote/A_navsim/k41_p02_score.py` (official PDMS of the base plan and of every stored joint-effect trajectory, metric cache rebuilt; plus the family summary CSVs). 1-log smoke: 5 removal-triple joint effects on token 0497454a, plan shift 13.1 m, PDMS 0.583 → 0. Chain `k41_p02_score_chain.sh` on VUW (waits for P02 → 2 CPU shards → summary → results/k41/p02_summary). D03 base: still waiting for a VUW GPU with ≥12 GB free. Jetson J04: repetition 1 av2t40 running (5/12 runs done).

2026-11-01 23:00 (VUW clock) **P02 DONE** (1,000/1,000 tokens, 0 errors). Joint-effect PDMS scored (38 trajectories on 17 tokens) and summary written to REPORT.md, with tables k41_2_p02_*.csv. k = 2 adjacent pairs: 0/58,398 moving. D07 field: 0 changed tokens, so k2 ≡ singleton (pre-specified); D07 score stage running on VUW1 (CPU), no GPU detect needed. D04 full 110/136. Jetson J04 rep 3 (9/12 runs). D03 base still waiting for a VUW GPU with ≥12 GB.

2026-11-02 00:40 (VUW clock) **Planner state leak found (NAVSIM v1.1, not our code); D07/P02 scoring rerun.**
- **What failed.** D07 score shard 1 stopped on its parity assertion: the rebuilt cheap plan of token bef3d5467aca5f04 (log 2021.09.09.18.29.25_veh-39_00427_00556) differs from the P02 draft plan by 4.395 m.
- **Ruled out:** identical detections (dets_s05 = dets_k30b), identical ground_z, identical planner code, PYTHONHASHSEED (8 seeds), and pre-seeding agent ids.
- **Cause.** `PDMObservation.update()` appends `_collided_track_ids` (including `red_light_<lane connector id>` when ego lies inside a red-light polygon), and `initialize()` never clears it. When one PDM-Closed instance is reused across scenes, red lights from earlier scenes of the same map are ignored later.
- **Reproduced exactly.** Replaying the base calls of P02 shard 1's previous 17 logs on one shared instance gives distance 0.000 m to the stored P02 plan. A fresh instance gives 4.395 m. The leaked list was `['red_light_70518', …]`.
- **Scope.** Every earlier script that reuses one MetricCacheProcessor across tokens: k20_pdm / k20_fast, navsim_field (frozen field), navsim_pdms, k30_*, k39_5, k41_p02_interactions, and the cache step of k41_selector_metrics. The official NAVSIM metric caching also reuses one processor per worker, so the effect is inherent to NAVSIM v1.1 and depends on processing order.
- **Effect on P02.** Within a token, every query sees the same leaked set, so P02 move/no-move tests are internally consistent. The reference planner differs from a fresh instance only on affected tokens. Frozen results are not modified.
- **Fix (registered deviation, before D07 outcomes).** D07 score and P02 joint-effect score now build a fresh instance per plan and per metric cache, record the distance to the P02 draft plan (`p02_base_dist`) instead of asserting it, and record the stored pdms_final scores so the leak's effect on stored scores can be counted.
- **Run state.** Old outputs moved to `results/k41/superseded/{d07_score,p02_score,p02_summary}_shared_cache_proc`. md5 k41_d07_k2.py cfffa77c…, k41_p02_score.py 3363c2ec…; rerun chain `k41_rescore_fresh.sh` on VUW1 (CPU, 4 + 2 shards).
- **P02 tables.** The `k41_2_p02_*.csv` files are kept until the fresh-cache rescore replaces them. Combination counts do not change; only the joint-effect PDMS may change.

2026-11-02 01:30 (VUW clock)
- **D04 DONE.** 136/136 logs, field parity 12,146/12,146. P01 new rows done; summary in REPORT and D04/. Main result: the field's repair beats PiP (a) by +5.1 points and (b) by +8.8 points (CIs exclude 0) at fewer px. PiP (c) and geometric are within noise. PDMS shows no difference for any row.
- **D07 DONE.** k2 ≡ singleton on all 1,000 tokens.
- **Leak audit.** 1/1,000 reference plans differ. Stored pdms_final reproduced on 1,000/1,000 tokens; shared vs fresh cache gives 0 differences.
- **P02 joint PDMS** unchanged after the fresh rescore.
- **D03.** The del_all dispatcher (relaunched by the D04 chain) has its smoke running on the VUW1 GPU. D03 base is still waiting for a VUW GPU with ≥12 GB.
- **Jetson J04 window complete** (12/12 runs, including the bd breakdown), window released; analysis next.

2026-11-02 02:30 J04/J02/J03 analysis DONE: written to REPORT, tables in K41/J04/. Remaining: D03 PiP (del_all smoke on the VUW1 GPU; base waiting for a VUW GPU), D05 81-point grid feasibility, final REPORT + REVIEW_RESPONSE_MATRIX.

2026-11-02 02:40
- **D05 feasibility (estimate).**
  - Measured D02 (v6-S base + 2 variants × 3 caps) took 2.69 GPU h on N + A + H. One grid index is about base + 1 variant, so about 0.8 GPU h.
  - 81 indices ≈ 60–120 GPU h, inside the RUNPLAN ceiling of 180–900; storage about 8 GB (tubes used 237 MB); B has 1.3 TB free. → **Run it.**
  - Chains `remote/A_method/k41_d05_chain.sh` were launched on both hosts. They wait for the host GPU queue lock, so they start after D03 on each host.
  - VUW1 runs a 2-unit nuScenes smoke first (grid 13) to measure the cost.
  - Order follows RUNPLAN (VUW 0–40, VUW1 41–80), then the other host's half, with atomic per-(index, dataset) claims in logs/k41/D05/claims. This is a registered deviation that keeps the full grid, with no reselection.
  - The wrapper's exit status is read from its log, because run_when_gpu_free.sh always exits 0.
- **P03 cost check: FAIL → not run.** P02 measured 29.1 CPU h per 1,000 tokens, so the 11,146 remaining tokens ≈ 324 CPU h > 240 CPU h cap. The k = 2 crop step also has no input, since there are 0 moving adjacent pairs.
- **D03.** del_all smokes passed and nusc full is done (1,693 s); av2 full is running on the VUW1 GPU. base is still waiting: VUW GPU 1 shows 12.1 GB free but is above 50% utilisation.
- **REVIEW_RESPONSE_MATRIX** updated (2026-11-02 section).

2026-11-02 03:35 (VUW clock)
- **D03 base moved to VUW1 (registered deviation; host only, same code path, same queue rules).** VUW GPUs are at 94–100% utilisation, and base had waited 8.5 h for an eligible GPU.
- **Dispatcher change.** `k41_d03_dispatch.py` gets an env override `K41_D03_HOST` (md5 a3016da1… → 386f8ccf…; k41_d03_manifest.json updated). The del_all dispatcher has already passed its last hash check, at the start of full_ahd, so it is unaffected.
- **Processes.** Killed our own VUW base dispatcher and its waiting wrapper; no GPU work had been done. VUW1 chain `k41_d03_base_vuw1_chain.sh`: waits for del_all to finish → base dispatcher on cuda13 → D05 chain.
- **D05 on VUW.** The VUW D05 chain now holds the cuda12 lock and is waiting for a GPU with grid 0 / nusc.
- **D03 del_all.** nusc and av2 full complete; ahd full running.

2026-11-02 05:30 (VUW clock). **User instruction: run the non-approved jobs too, do not worry about compute, optimise VUW / VUW1.**

**D05 first attempt failed safely.**
- Bash `source env.sh` (relative path) searches $PATH first and loaded the system `/usr/pkg/bin/env.sh`, so the venv was not active and `cv2` was missing.
- The legacy wrapper's exit line still read "exit 0", so 2 jobs (grid 0 nusc / av2) were falsely marked done.
- All D05 chains / workers / wrappers were killed. No detector work had run, so there were no outputs. Claims and logs were moved to logs/k41/D05/superseded_cv2_fail.
- Earlier jobs are unaffected: dispatchers use only the stdlib, the wrapper sources $B/env.sh by absolute path, and the NAVSIM chains source env_navsim.sh by absolute path.

**D05 relaunched** (`k41_d05_chain.sh` rewritten):
- env.sh is sourced by absolute path.
- Success = every reference unit has a .done.json with no failures and the current detector md5, instead of the wrapper's exit line.
- Parallel workers (`D05_WORKER`): 2 extra per host, SHARE=1, MIN_FREE_MB=14000, so they run only on our idle 24 GB GPU and not on other users' saturated GPUs. They start after the first verified grid job. Claims keep jobs disjoint.
- **Measured:** grid 0 nusc took 680 s (228 scenes, verified). Scaling by the D02 dataset ratios gives about 1.5 GPU h per index, ≈ 125 GPU h for 81 indices (estimate).

**P03 launched** (user-authorized despite the failed cost check):
- `k41_p02_interactions.py --subset remaining --fresh_per_token` (md5 06322b9a…): 11,146 tokens, 24 CPU shards (VUW 0–11, VUW1 12–23) → results/k41/p03.
- A 2-token smoke took ≈ 75 s / token, so ≈ 232 CPU h ≈ 10 h wall.
- A fresh PDM-Closed instance is built per token (state-leak fix). Within a token all queries share one instance, as in P02.

**D06 (K40-5) launched.** New driver `remote/A_method/k40r_lowconf_support.py` (md5 bf443320…), generated from k40r_detector.py, which is not modified because D05 validates its md5.
- Row v6S_lc = the frozen v6-S loop, seeded in round 1 with context-2 box crops of the draft-band boxes (0.05 ≤ conf < 0.25 from one extra cheap pass, charged in px) that do not duplicate a draft box and lie in the planner corridor.
- Frozen v6-S is the checked control.
- 2-scene smoke: parity failures 0; 5 corridor boxes; px 0.091 → 0.135 including the extra cheap pass.
- Full nuScenes (228 scenes) is queued on VUW1 (SHARE=1, MIN_FREE_MB=14000).

**D03 base** is running on VUW1, full nusc.

2026-11-02 06:50
- **D03 del_all DONE and analysed.** Parity is exact; REPORT section and K41/D03 tables written. PiP (a) loses heavily; (b) / (c) ≈ v6-S, n.s. except AV2 strict (c), where v6-S is better after Holm.
- **D03 base:** full nusc complete, av2 running (VUW1).
- **D05:** 9 grid jobs verified; 2 more workers added on VUW (GPU 1 freed) and 1 on VUW1.
- **P03:** 13 / 136 logs.
- **D06 full DONE** (exit and outputs to be verified; analysis next).

2026-11-02 07:35
- **D05:** 22 grid jobs verified, 0 failed, about 9.5 jobs/h; 2 more workers per host (VUW 7 chains, VUW1 6) because every non-saturated GPU had ≥ 20 GB free.
- **P03:** 26 / 136 logs.
- **D03 base:** av2 full running.
- **REVIEW_RESPONSE_MATRIX:** D03 del_all and D06 added.

- 2026-11-02 11:35 (VUW1 clock): D03 base full_nusc/full_av2 complete; full_ahd wrapper waited 1h20m because the 5 SHARE D05 workers occupied every GPU without other users' jobs (non-SHARE wrapper needs a GPU free of our jobs). Paused the D05 worker chains (SIGSTOP on the bash chains only; running grid jobs continue) via scripts/A_method/k41_d05_yield_vuw1.sh until the D03 ahd detector is on a GPU, then auto-resume. D05: 103 claims, 91 done, 0 failed. P03: 64/136 log outputs.

- 2026-11-02 16:30 (VUW1 clock): **D03 base DONE** (full_ahd completed 15:24; exact parity: max |ΔJ_v6S| = 0, 0 failures). The analysis is in REPORT §"D03 … natural draft". Fixed `k41_d03_analyze.py`: the no-refiner rows had used the v3a control. Both CSVs regenerated; v3a numbers unchanged. **D03 complete.** D05: 175 claims, 163 done, 0 failed. P03: 107/136 logs.

- 2026-11-02 17:40: D05 analysis script `remote/A_method/k41_d05_analyze.py` written and tested in `--partial` mode on the completed points (VUW1 → results/k41/d05_summary_partial). Per point it reports: G and Δ vs v6-S (same refiner, Holm over the 81 points), agreement, repair, decisive coverage, px, field µs and job seconds; parity is asserted per job. D05: 192 claims, 180 done, 0 failed. P03: 110/136 logs.

- 2026-11-02 18:10: **D05 partial (186/243 points analysed).**
  - Parity holds on every job.
  - **Range spacing has no effect.** Range factors 0.5 / 1 / 2 give bit-identical executed crops; verified on 1,775 frames (grid 00 vs 54, nusc + av2), with identical G and px at every matched point. Only field time changes.
  - **Why (code-verified, `vtf/dsf.insertion_grid`, `k16_attr.field_crop_g`):**
    - the insertion demand decreases monotonically with range, so the sensitive cells are always {r ≤ r*};
    - all three range grids share the nearest cell (12.5 m);
    - that cell's projected probe box spans the bounding-box bottom and lateral extent, and the top as well when the probe is at least camera height.
    - So the range spacing is a no-op for the request: a geometric property of the field, not a bug.
  - **What moves G** is probe size and lateral spacing:
    - nuScenes pedestrian probe with lateral ≥ 1: G ≈ −0.07, vs v6-S 0.211;
    - truck probe: G up to 0.268, at 1.3× px.
  - **Partial Holm results:**
    - no point beats v6-S on N or H;
    - on N, v6-S beats 13 points;
    - on AV2 (v3a), 40/62 points have G above v6-S but none is Holm-significant;
    - on AV2 without the refiner, 4 points are Holm-significantly above v6-S.
  - The full analysis follows when all 243 jobs are done.

- 2026-11-02 22:45 (VUW1 clock): **D05 grid runs DONE.** 243/243 (point, dataset) jobs done, 0 failed. Every job passed check(): all reference units have .done.json with no failures and the registered detector md5. Full analysis launched: `k41_d05_analyze.py results/k41/d05_summary` (logs/k41/D05/analyze_full.log). P03: 128/136 logs.

- 2026-11-02 23:50: **D05 DONE.** Full analysis complete: 243/243 jobs, parity exact, range invariance confirmed on every point. REPORT §D05, K41/D05/ (CSV, checks, figure k41_3_grid_G_px.pdf/png), matrix and QUEUE updated. No grid point improves on v6-S on all three splits. Adverse result: the AV2 lateral ×2 truck probe is +0.022 at 1.16× px, Holm-significant without the refiner (81-point family) and with v3a under 27-point Holm. The analysis docstring's "no grid point equals v6-S" claim was corrected: the (car, v_close 0) point equals v6-S exactly. Remaining: P03 (130/136 logs), then final consolidation of the K40R / K41 / LATENCY_OPT REPORTs and REVIEW_RESPONSE_MATRIX.

- 2026-11-03 00:30: **Final consolidation written.**
  - Six final answers at the top of the K41, K40R and LATENCY_OPT REPORTs.
  - Final W1–W7 table at the top of REVIEW_RESPONSE_MATRIX.
  - STATUS tables brought up to date: K41-1/2/3/5/6, K40-2/5, LAT-1/5/8.
  - 88 protected hashes re-checked: 0 changes.
  - Only P03 remains; its result will be added to REPORT §P02/P03, answer 5 and W7 when it finishes.

- 2026-11-03 01:20 (VUW1 clock): **P03 at 130 / 136 logs** (10,082 / 11,146 tokens). Remaining work:
  - shard 2: a 498-token log, then a 64-token log;
  - shard 7: a 366-token log;
  - shard 23: logs of 98, 26 and 12 tokens.
  - The shards are alive (about 95% CPU each). Results are written per log.
  - **Speed-up:** each queued log was started in parallel as its own process, using `k41_p03_one_log.py`. That is `k41_p02_interactions.py` plus the env `K41_P03_ONLY_LOG`, so the computation is identical and deterministic. The original shard skips a log whose output already exists. Done for the 64-token log (shard 2) and the 26- and 12-token logs (shard 23).
  - Estimate: shard 2's 498-token log at about 105 s / token (the P02 rate) needs about 5 h more. Then the score chain runs, about 1 h.

- 2026-11-03 08:10 (VUW1 clock): **P03 DONE.** 136 / 136 logs, 11,146 tokens, 0 errors. Score chain done: 603 joint effects on 182 tokens, all PDMS-scored. Results in REPORT §P03 and `K41/P03/`. Answer 5 (top of the three REPORTs) and matrix W7 updated. **K40R / K41 / LATENCY_OPT task complete.**
