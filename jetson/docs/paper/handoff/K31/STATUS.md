# K31 STATUS

Post-hoc analyses of the FROZEN v6-S (K28 §2n) and of the K28 / K30 baselines in their K28 forms (refiner v3a + C1;
guard and history for the crop heuristics). No configuration changes; nothing is selected on a test number; every row
is reported. Intervals: paired unit bootstrap, 2,000 resamples, seed 0 (units = scenes / logs / sequences).

## Pre-registration

### Item 1 — episode-level safety and temporal consistency (registered 2026-10-19, before any computation)
Source: stored per-frame records of the K28 phase-3 instrumented run, results/k28/p3/<ds>_test_instr (k28_p3.py,
setting base; AV2 with the ego mask). No re-run. Frames: the evaluated frames (gt = True): nuScenes new test 228 scenes /
9,181 frames (2 Hz, dt 0.5 s), AV2 val 150 logs / 11,818 frames (5 Hz, dt 0.2 s), Argoverse-HD val 24 sequences /
15,006 frames (30 Hz, dt 1/30 s). Within a unit, frames are ordered by frame index and consecutive evaluated frames are
treated as adjacent (Argoverse-HD has 31 one-frame holes without 3D labels; nuScenes and AV2 none).
Required action per frame: act_gt (the braking consumer on the annotated objects; KEEP 0, DECELERATE 1, HARD-BRAKE 2).
Pipeline action per frame: the stored braking action a_<pipeline>.

- Level L = 1 (required episodes, action >= DECELERATE) and L = 2 (hard-brake episodes, action = HARD-BRAKE).
- Required episode of level L: maximal run of frames with act_gt >= L, after merging gaps of <= 1 frame (a single frame
  with act_gt < L between two such frames belongs to the episode). Episodes are the same for every pipeline.
- Onset delay of pipeline X on an episode starting at frame s with length n (merged gap frames included): the smallest
  k in 0 … n-1 with a_X[s + k] >= L. If there is none, the episode is MISSED and its delay is capped at n. Seconds =
  frames x dt.
- Statistics per pipeline: number of episodes; missed-episode rate = missed / episodes; median and mean onset delay
  over all episodes (capped values; primary) and over the detected episodes only (secondary).
- False episode of level L: maximal run of frames with a_X >= L (the same <= 1-frame gap merge) on which act_gt < L on
  every frame of the run. Reported per 1,000 evaluated frames.
- Flicker: within a unit, over consecutive frames, (a) action changes a[t] != a[t-1] per 100 frames; (b) reversals: a
  change at t from A = a[t-1] to B = a[t] whose first following frame u > t with a[u] != B has u <= t + 2 and a[u] = A
  (A -> B -> A within 2 frames), per 100 frames. The same statistics of act_gt are reported as a reference.
- Pipelines (7): cheap (a_cheap); u960 + C1 + v3a (a_wf960+C1|v3a); native + C1 + v3a (a_wf{1600,2048,1920}+C1|v3a);
  hybrid tsp8@S + C1 + v3a; uncertainty unc2@S + C1 + v3a; v6-S (a_v6S|v3a, deployed); v6-S without refiner (a_v6S|none).
- Intervals: v6-S minus each other pipeline for missed rate, median and mean delay (frames and seconds), false
  episodes per 1,000 frames, changes and reversals per 100 frames, at both levels. A resample draws units with
  replacement and recomputes each statistic on the pooled episodes / frames of the drawn units (medians weighted by the
  draw counts). Point estimate on the full set, 95% percentile interval.
- The 10 longest onset delays of v6-S per dataset (L = 1 episodes, ranked by capped delay in seconds, ties by episode
  length, then unit and frame). Cause at the episode's first frame, with the K30 V3 rules (k30_analyze.v3c) on the K30
  part-match records results/k30/match (same frozen v6-S): decisive object not found and outside every crop; not found
  but inside a crop; found and its estimated range keeps the demand low (range error); found, range not the reason
  (TTC / lateral error). Episodes whose first frame has no decisive annotated object in the match record are labelled
  "no decisive object at onset". Panel id = <ds>/<unit>/<frame>; panels are rendered as in K30 V3 if a capture re-run is
  possible (stated in the report either way).
- Output: results/k31/tables/ep_summary.csv, ep_intervals.csv, ep_top_delays.csv on VUW; copies in handoff/K31/.

### Item 2 — intervals for the consumer-adaptivity claim (registered 2026-10-19, before computing)
Decision value of a consumer c: G_c(X) = (sum J_c,cheap - sum J_c,X) / sum J_c,cheap over the evaluated frames, with
the stored cost of that consumer. Paired unit bootstrap of (G_c(loop) - G_c(row)); frames joined on (unit, frame)
across the runs below (the cheap draft and the units are identical in every run).
- 2a TTC-warning consumer (cost J2). Loop: DESP's TTC loop "c2|v3a" (field + support of the TTC consumer, refiner v3a
  on its support; K28 phase-3 run results/k28/p3/<ds>_test_instr). Rows:
  (i) hybrid tsp8 with the TTC consumer's own support at K28's S budget ("tsp8c2@S+C1|v3a", K30 part match) and at
      the TTC loop's per-frame px with the x1.0 cap (MISSING: re-run, see "Re-runs");
  (ii) uncertainty unc2 at the TTC loop's per-frame px ("unc2@c2+C1|v3a", K28 phase 3);
  (iii) u960 + C1 + v3a and native + C1 + v3a (K28 phase 3).
  Reported extra rows, labelled: the hybrid with the TTC support at v6-S's per-frame px ("tsp8c2@x1.00+C1|v3a", K30)
  and the frozen braking-driven v6-S scored by the TTC consumer ("v6S|v3a", K28 phase 3).
- 2b IDM consumer (cost J3, K30 part idm records results/k30/idm). Loops: ours-IDM ("oursIDM|v3a") and the frozen
  v6-S ("v6S|v3a"). Rows for each: (i) hybrid with the IDM support at ours-IDM's px ("tsp8@IDM+C1|v3a") and at K28's S
  budget (MISSING: re-run); (ii) unc2 at ours-IDM's px ("unc2@IDM+C1|v3a"); (iii) u960 / native + C1 + v3a.
  Mechanism of ours-IDM vs frozen v6-S (descriptive, all frames): crops per frame and px of both; field size (number of
  field cells of each loop; not stored, logged by the re-run); frames where their IDM actions differ: count, direction,
  which equals the annotated IDM action act3_gt, cost difference; on those frames the decisive annotated IDM object
  (dec3): class, range, found by each loop (dec3_found_*).
- Output: results/k31/tables/consumer_ttc.csv, consumer_idm.csv, idm_mechanism.csv.

### Item 5 — distribution-shift strata (registered 2026-10-19, before computing)
Stored records of K28 phase 3 (results/k28/p3/<ds>_test_instr), braking consumer, pipelines cheap, u960 + C1 + v3a,
native + C1 + v3a, hybrid tsp8@S + C1 + v3a, uncertainty unc2@S + C1 + v3a, v6-S. Strata are fixed per unit:
- nuScenes new test: night = the scene description (v1.0-trainval scene.json) contains "night" (case-insensitive),
  else day; rain = the description contains "rain", else no rain (two separate splits of the 228 scenes).
- AV2 val: city, from the public log map file name (log_map_archive_<log>____<CITY>_city_*.json, listed from the public
  AV2 bucket; no file content is read).
- Argoverse-HD val: lighting per sequence is not annotated in Argoverse-HD / Argoverse 1.1, so per city
  (city_info.json of the Argoverse 1.1 tracking log of each sequence, via val.json seq_dirs).
Per stratum: units, frames, G (cost J) and mean px of every pipeline; v6-S minus each pipeline with the paired unit
bootstrap inside the stratum (2,000, seed 0). No stratum is dropped; strata with < 5 units are reported with a flag.
Output: results/k31/tables/strata.csv, strata_units.csv.

### Item 3 — curved-path AEB consumer (registered 2026-10-19, before any dev or test computation)
- Curvature per frame: kappa = yaw_rate / max(v, 1 m/s), clipped to |kappa| <= 0.1 1/m; yaw rate = central difference
  of the logged ego heading (k30_gt.yaw_rates; test: the stored K30 GT pass results/k30/gt; dev: the same function).
- Consumer ("curved AEB"): the paper's braking controller with every object mapped into the coordinates of the
  constant-curvature path (arc length s along the path of the object's centre at its near range, lateral offsets of its
  two lateral extremes from the path); |kappa| < 1e-6 is the straight controller exactly. Demand, thresholds, corridor
  width, range limit and loss unchanged. Required action recomputed from the annotations with the same mapping.
- Field and support of the curved consumer follow from it: the probe at ego cell (r, y) of the frozen grid is mapped to
  (s, d) and is sensitive iff its demand at s reaches the next action's threshold and it lies in the corridor at d;
  support = objects whose curved demand alone reaches the current curved action. "v6-S with the curved consumer" = the
  frozen v6-S loop code (k_loop.run_g, crops, guard, C1, refiner v3a) with only its field and support replaced by the
  curved consumer's (module-level substitution, as K28 did for the support).
- DEV first (nuScenes new val 152 scenes, AV2 CFG-DEV 300 logs (k23 av2_split role CFG-DEV) with the ego mask; draft only, no crops): share of frames
  with a non-empty field (straight vs curved, on the draft scene), distribution of |kappa|. The consumer "behaves" if the
  curved field is non-empty on a share of frames within a factor 2 of the straight field's and |kappa| > 0.01 on at least
  5% of frames; otherwise item 3 stops with the dev numbers.
- TEST (three test splits, once): rows scored by the curved consumer: cheap; u960 / native + C1 + v3a; hybrid tsp8 with
  its straight support and straight tube at K28's S budget (as K28); hybrid with the curved consumer's support and a tube
  that follows the same curved path (half-width grown until the budget, bounding box of the projected corridor from
  10 to 70 m; the straight tube when kappa = 0) at the S budget (re-designed prior, upper reference); uncertainty unc2@S;
  v6-S with the curved consumer; extra (labelled): the frozen straight v6-S scored by the curved consumer. Refiner v3a on
  the curved consumer's support (frozen straight v6-S: its deployed straight support). Report G, px and intervals of
  curved v6-S minus each row: overall, per |kappa| tercile (tercile bounds of the split's evaluated frames), and on the
  frames whose decisive object (curved requirement) lies outside the straight tube of the frame.
- Output: results/k31/tables/curv_dev.csv, curv_test.csv; per-frame records results/k31/r3/<ds>_{curvdev,curv}/.

### Item 4 — bounded loop on the full test splits (registered 2026-10-19, before running)
- v6-S with at most two detector calls = the K29 / Table 3 variant (59_bounded_latency.bounded_run: call k detects the new
  field + support crops of the current merged scene together with the guard's grown crops queued in call k-1; at the
  bound the merged scene's action; refiner as v6-S). VUW form: k_loop.run_bounded (N = 2), the same logic; the Jetson
  places the crops of one call on a common canvas, VUW letterboxes each crop alone. Parity check against the Jetson
  K29 test-window decisions (a_ours_b2@trt32) on the frames both ran (nuScenes test 20 scenes, AV2 8 logs), with the
  agreement of the unbounded v6-S on the same frames as the reference level of VUW-vs-Jetson numerics. If the
  agreement of the bounded loop is materially below that of v6-S (by more than 1 percentage point), the common canvas is
  ported to VUW and the run is repeated before reporting.
- Three test splits: G, px of v6-S <= 2, intervals vs v6-S, u960 + C1 + v3a, native + C1 + v3a, hybrid tsp8@S + C1 + v3a
  (stored K28 phase-3 rows); labelled post-hoc, does not replace v6-S.
- Output: results/k31/tables/bounded.csv, bounded_parity.csv.

### Variant B2K — bounded loop that keeps pending objects at the bound (user-approved 2026-10-19; post-hoc, dev-selected)
Motivation (item 4 mechanism, seen on test): in v6-S <= 2 calls a crop detection touching a crop edge is removed from the
merged scene until its guard-grown crop is detected in the next call; at the bound the pending detection is dropped while
the draft box of the same object is already replaced, so the object disappears (nuScenes scene-1051). Running the guard in
the last call would need a third detector execution, so both candidates keep exactly <= 2 calls:
- B2K-A: at the bound, every pending (edge-truncated, not yet grown) crop detection is added to the merged scene as
  detected (its truncated box), lifted like any crop detection (C1 lift, refiner v3a as v6-S).
- B2K-B: at the bound, for every pending detection the draft box with the largest IoU (>= 0.3) is restored to the merged
  scene with its draft lift; the truncated crop detection is not used.
Selection on DEV only (nuScenes new val 152 scenes, AV2 CFG-DEV 300 logs with the ego mask; k31_eval part b2dev): G of
v6-S, v6-S <= 2 (frozen bounded loop), B2K-A, B2K-B. Rule: choose the candidate with the higher mean dev G over the two
splits; adopt it only if it improves the mean dev G over v6-S <= 2 and is not below v6-S <= 2 on either split by more than
0.005; otherwise report "no variant qualified" and stop. The adopted candidate is then run once on the three test splits
(part b2v) and reported as a labelled post-hoc variant next to v6-S <= 2 (it replaces nothing). Calls per frame are
unchanged by construction (no extra detector execution); px identical to v6-S <= 2.
Output: results/k31/tables/b2k_dev.csv, b2k_test.csv.

### Item 6 — empirical check of a bottleneck-selector proposition on NAVSIM (P2; registered 2026-10-19 after items 1–5, before any code)
Data: NAVSIM v1.1 navtest (the K30 V6 (i) / V8a token set), stored per-token detections results/navsim_exp/dets_k30b
(cheap = YOLOv8s 192x320; F = full960 = whole frame at the crop scale 0.5; native for reference), PDM-Closed's route
centerline per token (results/navsim_exp/k30_center, object-independent), annotations of the token frame. Scene of a
detection set = the NAVSIM convention of K20 / K30 (navsim_pdms.branch_annotations: in-camera eligible annotations matched
to a detection keep their annotated state, unmatched ones are dropped, unmatched detections become lifted static objects,
everything outside the camera or not eligible is kept). For merging, every detection of a branch is mapped to its object
once per branch (the same matching), so the merged scene is the union of the objects of its detections (annotation
objects de-duplicated) — Assumption 1's "identical states".
Selector pi (fixed before any run; independent of the objects except through h):
- Candidates (15): the forward centreline (k30_nav_hybrid.forward_path, extended straight beyond its end) offset
  laterally by d in {0, -1.5, +1.5} m (ramped linearly from the ego's own lateral offset over the first 2 s) x constant
  acceleration a in {+2, +1, 0, -1.5, -3} m/s^2 from the ego speed v0, speed clipped to [0, max(v0, 15 m/s)]; 5 s at 0.1 s.
  Ego = nuPlan Pacifica rectangle (5.176 x 2.297 m, rear axle 1.461 m behind its centre), heading along the path.
- h(tau, o) = first time on the grid at which the ego rectangle overlaps object o's rectangle (o moves with constant
  velocity velocity_3d[:2]; lifted detections and probes are static), capped at 5 s.
- b(tau) = 5 s x progress(tau) / max progress over the candidates (5 s for all if the max is < 0.1 m).
- s(tau, S) = min(b(tau), min_o h(tau, o)); pi(S) = argmax s, ties to the first candidate in the order above
  (offsets 0, -1.5, +1.5; accelerations +2 ... -3).
Field: the unseen static vehicle probe (class-median size, ego heading) at each cell of the 7 x 7 grid of K20
(8 ... 55 m, +-10.5 m) that projects into CAM_F0; a cell is sensitive iff pi(S + probe) != pi(S) (full re-evaluation).
Support: in-camera eligible objects o of S with h(tau, o) <= s(tau*, S) for some tau != tau*. Crops as spec_pdm_dets
(one crop per image column of sensitive cells, union of their boxes; box_crop of each supporting object's detection).
Loop (Algorithm 1): R grows by the crops of the field and support of the current merged scene; stop when pi is
unchanged and no crop was added (a crop covered by R is not added), at most 3 rounds.
Runs: (i) oracle locality: merged scene = objects of the cheap detections whose box centre lies outside R + objects of
the full960 detections whose centre lies inside R -> expected agreement with pi(F) 100% (a disagreement can only come
from the probe grid: an object of F outside R whose single insertion changes pi; each disagreement is classified);
(ii) real crops: the crops detected at scale 0.5 with the truncation guard (spec_pdm_dets' machinery), merged as there ->
agreement with pi(F) and px (crop area incl. grown crops / image); reference rows: agreement of pi(cheap) and of
pi(native) with pi(F). Agreement with a log bootstrap (2,000, seed 0).
(iii) pairs: on the K20 1,000-token subset (k20_pdm.subset), on the draft scene, every pair of projected probe cells:
the number of pairs whose joint insertion changes pi while neither single insertion does (expected 0 by the lemma:
for a bottleneck score, s(tau, S + p + q) = min of the single-insertion scores).
Output: results/k31/p2/<log>.pkl, results/k31/tables/p2_agreement.csv, p2_pairs.csv, p2_disagreements.csv.

## Re-runs (GPU, VUW; listed before they are launched)
- R2a: hybrid tsp8 with the TTC consumer's support at the TTC loop's per-frame px (x1.0 cap), three test splits.
- R2b: hybrid tsp8 with the IDM support at K28's S budget; field size of ours-IDM and of the frozen v6-S logged.
- R3 (item 3) and R4 (item 4): registered with their items.

## Log
- 2026-10-19: K31 received. Item 1 pre-registered (above). Stored records checked: all 7 pipelines and act_gt present in
  results/k28/p3/<ds>_test_instr for the three test splits.
- 2026-10-19: item 1 DONE (k31_episodes.py, CPU, stored records): ep_summary.csv, ep_intervals.csv, ep_top_delays.csv.
  Panels for the 30 top delays not yet rendered (needs a capture re-run; queued with R2).
- 2026-10-19: item 2 stored part DONE (k31_consumers.py): consumer_ttc.csv, consumer_idm.csv, idm_mechanism.csv. Rows
  waiting for re-runs: 2a hybrid at the TTC loop's px (R2a), 2b hybrid IDM support at the S budget and field sizes (R2b).
  Amendment before R2b runs: R2b also logs, for ours-IDM and v6-S, the decisive IDM object's estimated range / TTC, whether
  it came from a crop, and whether the refiner corrected it (mechanism only).
- 2026-10-19: item 5 DONE (k31_strata.py): strata.csv, strata_units.csv. AV2 cities from the public bucket listing
  (results/k31/meta/av2_val_city.csv); ATX has 4 logs (flagged small).
- 2026-10-19 17:36 (VUW): R2 launched (scripts/A_method/k31_eval.py md5 9c805dd5 = k30_eval.py md5 55471d34 + parts ttcpx /
  idmS; k31_r2_chain.sh, 6 shards x 3 datasets x 2 parts, 12 groups). Smoke on the first unit of each split: the TTC loop
  c2 (actions, px) and the IDM rows of ours-IDM / v6-S are identical to the stored K28 / K30 records (100%).
- 2026-10-19 18:01 (VUW): items 3 / 4 pre-registered (above); k31_eval.py extended with parts curvdev / curv / b2 (md5
  e35a4973). R2 groups that start after 18:01 run this version: the ttcpx / idmS code paths are
  unchanged (one extra logged key conv_<loop>). Smoke of the new parts queued (k31_r34_smoke.sh).
- 2026-10-19 (VUW 18:25): R2 DONE (all units of the three splits, both parts). Parity: the re-run TTC loop c2 equals the
  stored K28 c2 on every frame (J2 identical 9,181 / 11,818 / 15,006); the IDM actions of ours-IDM and v6-S equal K30 V2b on
  every frame. Item 2 DONE (consumer_ttc.csv, consumer_idm.csv, idm_mechanism.csv incl. the decisive-object breakdown).
- 2026-10-19 (VUW 18:57): items 3 / 4 smoke OK (curved consumer = straight on kappa ~ 0 frames; b2 part's v6-S = K28 v6-S on
  the smoke units); k31_r34_chain.sh launched (curvdev nusc / av2 dev, curv and b2 on the three test splits).
- 2026-10-19 (VUW 19:02): the three curv (test) groups had been queued together with the dev check; they were stopped
  by PID while still waiting for a GPU (never started, no output), so the pre-registered order holds: the curv test runs
  are launched only after curv_dev.csv shows the consumer behaves.
- 2026-10-19 (VUW 19:12): item 3 dev check DONE (curv_dev.csv): non-empty field on the draft, straight / curved: nuScenes new
  val 0.510 / 0.503, AV2 CFG-DEV 0.383 / 0.383; |kappa| > 0.01 on 20.2% / 13.4% of frames (p90 0.032 / 0.017; clipped
  at 0.1 on 1.6% / 2.1%) -> the consumer behaves by the registered rule. (Correction of the registration text: AV2 CFG-DEV
  has 300 logs, k23 av2_split role CFG-DEV, not 128.) Test runs (part curv, three splits) launched.
- 2026-10-19 (VUW 19:27): item 4 runs DONE (b2 on 228 / 150 / 24 units). Parity (bounded_parity.csv): in-run v6-S = stored
  K28 v6-S on every frame; VUW vs Jetson K29 decisions on the common frames: bounded 99.63% (nuScenes 20 scenes, 805
  frames) / 99.83% (AV2 8 logs, 586), unbounded 99.63% / 99.32% -> the bounded loop is not below the unbounded level, so by
  the registered rule no common-canvas port. bounded.csv: v6-S <= 2 G 0.091 / 0.282 / 0.419 (v6-S 0.211 / 0.294 / 0.418).
  nuScenes: 574 of the 590 extra cost come from one scene (scene-1051, 8 frames where the decisive object is truncated by
  a crop and its guard-grown crop would be detected in a third call; at the bound the action stays KEEP while v6-S
  hard-brakes); not in K29's 20 timed scenes.
- 2026-10-19 (VUW 20:10): VUW is CPU-bound (load 44 on 32 cores) while VUW1 is idle: the second half of the remaining curv
  units (AV2 44, Argoverse-HD 8; results/k31/meta/vuw1_curv_<ds>.txt) run on VUW1 in parallel (same code, same output
  folder; a unit is skipped by whichever host reaches it after its file exists). Operational only.
- 2026-10-19 (VUW 20:25): B2K implemented (k31_eval.py md5 c18a782f, run_bounded_keep = k_loop.run_bounded + the two
  bound rules; parts b2dev / b2v). Smoke on 3 nuScenes val scenes + 2 AV2 CFG-DEV logs: with no pending detection at the
  bound both candidates equal the frozen bounded loop (B identical on every converged frame; A differs on 1 later frame
  through its own C1 history, as expected). Dev runs launched on VUW1 (nuScenes val 3 shards, AV2 CFG-DEV 6 shards).
  The curv processes still running loaded the previous version; the curv code path is unchanged.
- 2026-10-19 (VUW 21:44): item 3 test runs DONE (228 / 150 / 24 units, all logs exit 0); curv_test.csv written.
  B2K dev DONE (152 / 300 units; b2k_dev.csv). Dev G (nuScenes new val / AV2 CFG-DEV): v6-S 0.321 / 0.309, v6-S <= 2
  0.252 / 0.306, B2K-A 0.346 / 0.307, B2K-B 0.316 / 0.286. By the registered rule B2K-A is selected (mean 0.326 > 0.279;
  not below v6-S <= 2 on either split by more than 0.005: +0.094 / +0.001). Calls and px equal v6-S <= 2 (px 0.0744 / 0.0504).
- 2026-10-19 (VUW 21:45): B2K-A test run launched once on VUW1 (k31_b2v.sh A: part b2v --b2k_modes A, three test splits,
  3 shards each); output results/k31/r4/<ds>_b2v, then b2k_test.csv.
- 2026-10-19 (VUW 22:00): K31 REPORT.md (DRAFT) written: items 1–5 final; B2K test, item 1 panels and item 6 pending.
  curv_mechanism.csv was added as a descriptive table for the not-above rows of item 3 (refiner on the curved support).
- 2026-10-19 (VUW 22:05): item 6 pre-registered (above), then implemented: scripts/A_navsim/k31_nav_p2.py (md5 bb046934),
  k31_p2_chain.sh.
  - CPU smoke, oracle locality, 15 tokens of one log: agreement with pi(F) 15 / 15; 1–3 rounds; 1 token capped.
  - Pairs smoke, 7 tokens: 0 joint-only changes out of 5,343 pairs where neither single probe is sensitive.
  - The GPU smoke with real crops (5 tokens) is queued.
- 2026-10-19 (VUW 22:10): item 6 GPU smoke with real crops OK: 5 tokens, oracle 5 / 5, real crops 4 / 5, px including
  grown crops 0.30–1.37. The px is a sum of crop areas, so overlapping and guard-grown crops count twice.
  The full navtest run is launched (k31_p2_chain.sh run: 3 shards in one GPU group). The pairs stage follows once the
  run is done.
- 2026-10-19 (VUW 22:20): B2K-A test DONE (228 / 150 / 24 units, all logs exit 0); b2k_test.csv written.
  - G 0.209 / 0.293 / 0.415 at the bounded loop's px.
  - B2K-A − v6-S: −0.001 / −0.001 / −0.003 (all n.s.).
  - B2K-A − v6-S <= 2: +0.119 [−0.001, 0.299] on nuScenes.
  - Added to REPORT §1 / §2.4b / §3 and to 03_results.md.
- 2026-10-19 (VUW 23:36): item 6 run DONE (136 / 136 navtest logs, p2_run.log exit 0). Pairs stage launched
  (k31_p2_chain.sh pairs: CPU, 3 shards; K20 1,000-token subset).
- 2026-10-20 (VUW 01:20): item 6 DONE.
  - Pairs: 3 / 3 shards exit 0; 1,000 tokens.
  - Table stage → p2_agreement.csv, p2_pairs.csv, p2_disagreements.csv: oracle 99.71%, real crops 96.76%; 35 oracle
    disagreements, all field misses; 0 / 769,561 joint-only pair changes.
  - Item 1 panels copied to handoff/K31/panels (capture exit 0 at 22:40).
  - REPORT.md FINAL.
