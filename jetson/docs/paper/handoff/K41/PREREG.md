# K41 — pre-specified post-hoc analysis

Registered 2026-10-30 before new outcomes. Binding source: master prompt including CORRECTIONS and Appendix B. Data/statistics/isolation and shared-component rules in `../K40R/PREREG.md` apply. No external preregistration claim. All original IDs are retained. CPU cached analyses precede the single RUNPLAN gate. Missing evidence is marked partial/blocked, never silently replaced. No edits to paper or frozen K38/K39 code.

## K41-1 (P1): PiP-inspired adaptations and planning

Audit official VeteranAD paper/code/weights availability only; no installation or official reproduction claim. Fixed train-derived K from official paper when explicitly stated, otherwise K=20; horizon 4 s. Anchors fit on train, no test future trajectories. (a) all anchors; (b) top M=3, eight equally spaced 0.5 s waypoint steps, rescore after each verification; (c) same plus DESP support. Common components and nominal/strict caps, three braking splits, v6-S parity; G, pixels, paired CI, repair, natural and deletion recovery. On NAVSIM use existing 15 candidates, agreement and repair plus downstream PDMS/officially supported EPDMS, real crop detections. Existing oracle locality remains diagnostic. Use a strong geometric candidate/trajectory-region comparator with identical support/iteration where feasible and report actual pixels, planner queries, latency.

Scene-dependence diagnostic: within fixed selected candidate/action and 1 m/s speed bins, lexicographically pair distinct tokens, no quality-based selection; record actual speed difference and optionally exact-speed matches. Compare field mask/footprint IoU, requested-crop changes, query/crop cost and decisions. Distinguish matched from identical ego state; geometry/camera differences are covariates, not proof of scene dependence. Reuse K31/K39-5 saved fields/poses/detections/score records first. Small or noisy downstream gains and expensive planner queries trigger a narrower-scope recommendation, not unbounded model search. No new BEV/3D stack.

Controlled CPU follow-up (registered after matched-bin diagnostic showed confounding, before controlled outcomes): within each original log, pair consecutive lexicographically sorted tokens, taking the first token as receiver and second as donor. Evaluate both cached cheap object sets in the receiver's ego coordinates with **one fixed receiver selector (speed/15 candidate poses), camera and probe domain**. This is a synthetic scene transplant, not two real images at identical pose. Record every pair, both baseline actions, changed masks/projected field-only crops, and fixed geometric union of all candidate trajectories as comparator. Summarize all pairs and the prespecified same-action subset, paired bootstrap by original log (pairs never span logs). Do not run real crops or claim PDMS from this CPU diagnostic. No selection based on field differences. Explicitly distinguish the comparator's projected trajectory area from detector pixels.

## K41-2 (P1): interactions

Audit prior K20 6/9,449 object-removal pairs and 11/139,675 probe+removal, K31 0/769,561 probe pairs from saved logs. First 1,000 tokens use the saved K20 seed-0 subset; extension to 12,146 only after feasibility/approval. PDM triples outside support, pairs and triples of individually insensitive inserted probes; enumerate if ≤100 combinations/token/order, otherwise sample 100 without replacement using seed derived from SHA256(dataset,token,kind,0). Keep individual-effect checks, >0.5 m 4 s plan-change threshold, original scene state per query, PDMS for each joint effect. k=2 field: adjacent longitudinal/lateral cells (Manhattan distance one on the 7×7 grid) in the existing plan corridor, plus original singleton probes; report agreement increment, planner queries/time and PDMS. Large planner sweeps are in RUNPLAN, not pre-gate CPU work.

Learned object-input planner availability audit of local/VUW code, environments and weights; if none, report unavailable, do not install. Finite pairs/triples do not prove insertion stability. Provide a threshold-cardinality counterexample (k+1 individually irrelevant objects jointly flip action) and keep bounded-order diagnostics outside Theorem 1. Only a separately proved quantified interaction-order assumption could justify a k-wise theorem.

## K41-3 (P1): stress strata and grid

Fixed descriptive test terciles (np.quantile at 1/3,2/3; ≤q1, q1<x≤q2, >q2; ties retained and counts reported) for target object count; corridor count and support size separately; target decisive native box height; decisive closing speed; decisive range. Missing/no decisive objects form an explicit group. No favorable subgroup selection. Main rows: DESP, u960+comp., hybrid, uncertainty, AnA iterated; corresponding no-ref variants for target fidelity when stored. Report G/Δ CI, agreement, repair, actual px, decisive coverage, frames/units, missingness. Tercile conditioning fixes population before bootstrap, bootstraps units not individual objects. Closing-speed/edge-small cases highlighted regardless of outcome.

Grid sweep: all range spacing ×{0.5,1,2}, lateral spacing ×{0.5,1,2}, pedestrian (0.6×1.8), car (1.8×1.6), truck (2.6×3.6), v_close {0,5,10} = 81 points. Range half-spacing inserts midpoints between original grid cells; double-spacing takes original indices 0,2,4,6 and endpoint 7; lateral 0.75/1.5/3 m over −6…6. No test selection or replacement of frozen grid. Reuse precisely matching cached rows; CPU geometric sweep/time is explicitly separate from detector-dependent G/pixels/coverage. Real-crop missing points wait for RUNPLAN.

## K41-4 (P2): locality/coverage

Target object centre must lie in actual verified union. Match crop/target detections one-to-one, same-class, descending IoU with stable index ties, separately IoU≥0.5 and ≥0.7; also report class-agnostic matches for class error. Range absolute/relative error, finite TTC absolute/relative error and finite-status mismatch. Edge bins min distance from target centre to containing crop edge / native box height: ≤0.5, (0.5,1], >1; no containing crop separate. Guard yes/no; height terciles; actual crop scale groups. Report by round and draft action, target demand≥τ1 objects and decisive objects, not merely GT objects. Compute P(disagreement | support-or-decisive locality violation), P(disagreement | coverage miss), and P(disagreement | neither) on complete cases, with denominator and missingness. Missing state/crop logs block affected endpoints; GT recovery cannot be substituted for target locality.

## K41-5 (P1) and K41-6 (P2): device gates

Copy K39 harness into `experiments/k41_jetson_20261030/`; preserve baseline sources. Timed frames: nuScenes 20 scenes/805; AV2 same 40 logs/2,968; 200 warmup, three rotated repetitions in one process with independent row histories. Exclusive MAXN, locked clocks, fan 255, hardware restoration. Rows: AdaScale, Chanakya, decision router, FOVEA dataset prior, AnA one/iterated, uncertainty, counterfactual one-pass (iterated if available), u512/u640/u800/u960/u1280+C1/v3a, DESP and bounded-keep. D={33.3,50,66.7,infinity}; LAT also 100. Same hold-last-on-time-action rule. Iterated rows use K38 pre-round stop based only on dev round/finish medians. Distinguish trace-driven replay from live in-loop stopping; never label post-hoc truncation simulation as live execution. Δ G CI for DESP and bounded-keep, late share, latency tails and safety.

K41-6 profiles rounds, guards, crop batches and per-engine calls, conditional p95/p99 and 50/66.7 ms misses. Investigate guard/next-round batching, fixed-shape buckets/graphs and overlapping looming; dependencies prohibit blind merging, changed padding/pixels or temporal races. Reject exact designation on any action/crop/pixel mismatch on all dev and timed frames. K39 fixed-shape graphs already adopted; dynamic graph illegal-memory failure and A2–A4 no-room results are prior evidence, not new proposals. Compare equally optimized u960 and iterated AnA. No new timing before approval.

## P3 memo and traceability

2026-10-30 schema audit clarification, before object outcomes: K30 apdump stores post-refiner spatial geometry and unchanged C1 TTC. CPU reconstruction of no-refiner z/lateral uses the identical original `predicted_geometry` and camera calibration; cheap objects retained outside the verified union keep their saved cheap geometry. Require action parity against original rows and release before using a frame; report every failure/missing frame. K30 match crops include ordinary crops followed by guard crops. They provide final-union coverage and final round count, not a per-object originating round/canvas; these unavailable endpoints remain explicit. Annotation matching reuses K30's class-compatible matching at IoU 0.3. An unmatched annotation is an operational proxy, not proof that an object is physically spurious.

Official VeteranAD paper checked: implementation details explicitly set planning modes K=20, so the registered numerical K remains 20 (now source-backed, not fallback). Official code repository exposes VeteranAD+ and a checkpoint link; no installation or download of weights in this task. Sources and distinctions are recorded in AVAILABILITY.md.

Anchor fitting detail before execution: train-only ego trajectories sampled at .5s through4s, represented by eight ego-frame (x,y) points; KMeans K=20, seed0, n_init=10, Euclidean distance on the16 coordinates. nuScenes uses only `set=new,split=train` scenes in nusc_full/split.csv; AV2 only REF-TRAIN logs in k23/av2_split.csv, asserted disjoint from verified CFG-DEV. A-HD transfers AV2 anchors, never fits on its val or holdout. Use all valid4s windows at original keyframes (nuScenes) or .5s pose-grid starts (AV2), no outcome weighting. Record source hashes/fit counts. This is CPU training of the requested anchor prior, not detector/planner training or test tuning.

Object reconstruction correction before attribution summaries: `k_loop.detect_merge` removes cheap boxes by ordinary-crop centres and, for guard detections, IoU≥0.5; a grown-region union alone is insufficient to identify retained cheap boxes. Extraction v2 solves the original merge equation for the unique retained-cheap prefix, then restores that prefix from cached cheap state. Ambiguous inversions stop the run. Earlier `objects/` extraction is diagnostic only; use `objects_v2/` with its hashes and parity logs.

The first v2 smoke found multiple prefix assignments satisfying the merge equation. Before proceeding, retain only assignments also satisfying the saved refiner contract (unchanged cheap/non-support state; one positive common range/lateral scale for full support). Accept multiple assignments only if all reconstructed states coincide within absolute 1e-9; otherwise mark reconstruction unavailable and exclude from attribution, with full missingness counts. Never resolve ambiguity by choosing the assignment with favorable decisions or outcomes.

Geometry parity caught the missing AV2 ego mask in the reconstruction adapter (the stored refined actions already matched). v3 explicitly reproduces `k10_masked_run.py`: boxes with ≥50% of height in y≥1830 have range +infinity, lateral coordinates unchanged. This restores the frozen evaluation contract. `objects/` and `objects_v2/` remain diagnostics; only `objects_v3/` supports final summaries after parity. No scientific threshold or frozen inference code changes.

Endpoint review after v3 tables: locality must match **crop-origin** detections, not retained cheap detections in the merged scene. v4 records retained-prefix ambiguity and uses only guaranteed full/crop detections. Locality violation is evaluated on covered target decisive objects and crop-origin DESP support; objects outside the verified union belong to coverage, not locality. This corrects endpoint implementation to the original rule; v3 tables are superseded. Frozen prediction/actions and K40 taxonomy rules are unchanged. Final source is `objects_v4/`.

Memo estimates (labelled, not measured) for depth/3D/BEV lift, richer planner, TTC-state field, alternate detector and live closed loop. No mandatory new training/installation. Script hashes before execution in `code_manifest.csv`; records inventory from K40R; all new test evidence labelled post-hoc. REPORT must answer master §0.2's six questions and link W1–W7 matrix.

2026-10-31 P01 execution detail before scoring: historical NAVSIM v1.1 metric_cache is absent at its original path; no alternate cache identified in the scoped inventory. Reconstruct only the current token's metric cache using the installed official MetricCacheProcessor and original, unmodified benchmark scene/ground-truth future. Score fixed K31 selector choices with official pdm_score, save results/poses/cache hash and size, then remove only this newly generated temporary cache. No source data are removed. Selector rectangle centres are converted to rear-axle poses by subtracting EGO_RA along heading, taking indices5,10,…,40 at0.5s (8 poses/4s); no trajectory fitting. First smoke uses two lexicographic tokens and requires exact original cheap/full selector decisions plus exact stored K20 cheap PDM trajectories. Initial CPU batch scores cached cheap/full/field-real/oracle rows; hybrid and the four new comparators await D04 records. Identical trajectory bytes within a token may reuse the exact score. At most4 single-thread workers, CUDA hidden,100 aggregate CPU-hour ceiling including reconstruction/failed attempts; stop all workers at token boundaries after any failure. This is PDMS, not EPDMS.

P01 cached hybrid clarification before its scores: dets_k30b already contains the fixed K30 hybrid_pdm detections (route-centreline tube + PDM support). Feed those existing detections to the same K31 selector and score its selected trajectory; label the inherited PDM-support crop policy explicitly, rather than claiming selector-specific support parity. This supplies the fifth cached row without a detector rerun. D04's geometric and selected-support rows remain the consistent-support comparisons. First four-row smoke passed exact selector/K20 cheap-plan parity on both registered tokens; second smoke adds cached hybrid and must reproduce those four scores exactly.

## 2026-11-01 addendum — D03 executable definitions (registered before any D03 outcome)

Driver `remote/A_method/k41_d03_detector.py` (copy of the D01/D02 driver `k40r_detector.py`, which is not modified; only
the `pip` part and the deletion option are new). Anchors: the train-only K=20 anchors of `results/k41/anchors/{nusc,av2}.npz`
(Argoverse-HD uses the AV2 anchors), ordered by training-cluster size (descending); ego frame, x forward, y left;
waypoints 0.5…4.0 s.
- **Region of an anchor step t:** the ground path segment between waypoints t−1 and t (t = 0 starts at the ego origin),
  clipped to [standoff, max_range] of the frozen planner, widened by the frozen corridor half-width, projected with the
  frozen tube convention (pinhole P2, camera height, object height 1.6 m) to one image box; empty / behind-camera
  segments give no request.
- **(a) PiP-style, all anchors:** one round; all non-empty segment boxes of all 20 anchors, anchors in prior order,
  segments in time order, boxes already covered by earlier requests skipped. No DESP support.
- **(b) plan-selected, autoregressive:** up to eight rounds, one waypoint step per round. Before each step the consumer
  scores every anchor on the current merged scene: the largest required deceleration (frozen braking demand) of a scene
  object measured in that anchor's corridor, i.e. the object's lateral extent shifted by the anchor's lateral offset at
  the object's range (linear interpolation along the anchor). Top M = 3 by score, ties in prior order. The segment-t boxes
  of the three are requested; a step whose boxes are all covered advances to the next step in the same round.
- **(c)** (b) plus the DESP support crops (frozen support, box crop with context 2) of the current scene each round.
- Common: YOLOv8s crops at 960 per native long side, C1 lift, refiner none / v3a, truncation guard, merge rule and
  history as the D02 variant loop (`k41_variant_loop.py`, unchanged). Caps at v6-S's per-frame crop pixels: nominal,
  K39-literal strict, observed strict (as D02). Settings: natural draft (`--setting base`, VUW) and the K28 deletion
  setting (`--setting del_all`, VUW1: the draft's matched detections of decision-relevant annotated objects deleted).
- Parity: the v6-S control must reproduce the 10 frozen endpoints of the K30 match records (base) or the K28 phase-3
  deletion records (del_all) on every frame. Endpoints: G, px, Δ CI (v6-S − row), repair (K39-6), natural recovery
  (base) and deletion recovery (del_all; found among the deleted objects). No parameter is tuned; all nine rows are
  reported.
- Correction before any full D03 run (2026-11-01, after a 10-frame smoke showed rows (b) / (c) never verified because
  the near-ego segment boxes exceed v6-S's per-frame budget and the loop then stops): rows (b) / (c) request only boxes
  that fit the remaining budget, skipping the others and moving on (the cap rule of the crop heuristics, `fill_px`;
  remaining = budget − ordinary crops, also − guard canvases under the strict caps). Row (a) already follows this rule
  inside the loop's selection. No outcome was looked at besides pixels / rounds of the 10 smoke frames.

## 2026-11-01 addendum — D04 executable definitions (registered before any D04 outcome)

Driver `remote/A_navsim/k41_pip_nav.py` (new; imports `k31_nav_p2.py` unchanged; the K31 real-crop merge rule is copied
verbatim). Population: all 12,146 navtest tokens / 136 logs of `results/k31/p2`. Anchors: the selector's own 15 fixed
candidates (no train-derived anchors on NAVSIM). Detector, crop scale 0.5, conf 0.25, truncation guard, merge rule and
draft are those of the K31 real-crop field row.
- **Step region of candidate c, step t (t = 1…8, 0.5 s):** union of the CAM_F0 boxes of the ego rectangle
  (5.176 × 2.297 × 1.6 m, on the ground plane) at the candidate's 0.1 s poses 5(t−1)…5t with 8 ≤ x ≤ 55 m (the probing
  field's range, as the K30 tube); no projection → no request.
- **field:** the K31 DESP loop with real crops, rerun as the parity row: π, px and rounds must equal the stored
  `pi_real / px_real / rounds_real`. A mismatch is recorded per token (GPU kernels may differ from the K31 run); the full
  run stops if the 5-token smoke shows any π mismatch.
- **Budget:** the token's field ordinary-crop native area (nominal cap only; the K39 strict variants concern the
  driving loops). Cap rule of the crop heuristics: a request not fitting the remaining budget is skipped, the next is tried.
- **(a) all:** one round, step boxes of all 15 candidates, candidates by draft score s(τ, draft) descending (ties by
  index), steps in time order; covered boxes skipped. No support.
- **(b) selected:** ≤ 8 rounds, one step per round; before each step, top M = 3 candidates by s(τ, current merged scene)
  (ties by index); a step whose boxes are all covered/empty advances to the next step in the same round.
- **(c) selected_support:** (b) plus the DESP support crops (`field_support`, box crop) of the current scene each round.
- **geometric:** all candidates' step boxes plus DESP support each round, iterated exactly as the field loop (≤ 3 rounds,
  stop when nothing new) — identical support / iteration to the field.
- Records per token and row: selected index, the 8-pose trajectory exactly as P01 scores it, native px (ordinary + guard),
  detector tensor px, detector calls, detector seconds, wall seconds, rounds, skipped requests. Crops are cached per
  token, each row is charged its own crops. Endpoints: agreement with π(F = full960) and repair on π(cheap) ≠ π(F)
  tokens (log-bootstrap CIs), PDMS of every row via P01 (`--rows all,selected,selected_support,geometric`), px / calls /
  latency. No parameter is tuned; all rows are reported.
- Correction before any GPU / full D04 run (2026-11-01, after a 2-token CPU functional smoke): `geometric` requests the
  support crops before the step boxes. With step boxes first, the budget was exhausted before any support crop on both
  smoke tokens, making `geometric` identical to (a) in crops. Only crop counts / pixels of the two smoke tokens were looked
  at (both tokens have π(cheap) = π(F), so no agreement information). The CPU smoke's field px differed slightly from the
  K31 GPU record (CPU kernels alter guard growth); the GPU smoke remains the parity gate.

## 2026-11-01 addendum — J04 / J02 / J03 executable definitions (registered before any timed run)

Harness `experiments/k41_jetson_20261030/scripts/92_k41.py`, generated by `build.py` from the frozen K39-1 harness
`89_k39.py` (exact single-occurrence patches; K38/K39 sources, weights and engines unchanged), window `window.py` (K38-1
protocol: exclusive, MAXN, jetson_clocks, fan 255, tegrastats 500 ms, restoration on exit). Backend for every row: staged
fast path, trt32 C1/refiner, `--graphs static` (CUDA graphs for fixed-shape engines; dynamic crop engine without graphs
as in K39-1). All rows in one process per dataset × repetition, each with its own history.
- **Rows.** v6-S (DESP), B2K-A, B1K-A at D = ∞ and in-loop at D ∈ {33.3, 50, 66.7, 100} ms (K38-1 rule); u512 / u640 /
  u960 / u1280 / native + C1 + v3a; capped hybrid tsp8S+C1 and uncertainty crops unc2S+C1 (K28 unc2: context 2, lowest
  draft confidence first; ties by index because the device draft carries no class entropy) under the K39 heuristic
  budget (v6-S crop px incl. guard of the frame); the dev-selected K33 rows (K33 dev_selection*.csv): AnA-style one pass /
  iterated `ana05hinfo` / `ana05hinfi`, counterfactual one pass / iterated `cf40q20o` / `cf40q20i`, AdaScale
  `ada+1t1280`, FOVEA dataset saliency KDE_S `fovs1280b64a1` (and KDE_I `fovi1280b64a2`), decision-value router
  `dvr1280_0.5`, BAR router `bar1280_0.5`, Chanakya `ch128B_csc`; all + C1 + v3a. Shapes > 1024 px use the existing
  static whole-frame engines of exactly that shape; ≤ 1024 the shared dynamic engine (as K33-J).
- **Deadlines.** Iterated K41 rows get in-loop variants `<row>_dl<D>`: before each crop round, stop if elapsed +
  median(round k) + median(finish) > D, medians per dataset / row / round from the dev calibration (`medians41.json`); a
  stopped row returns the action of its last completed round's merged scene. One-pass rows (whole frame, AdaScale, FOVEA,
  routers, Chanakya, AnA / CF one pass, heuristics) do not depend on D: their result at D is their D = ∞ run with the
  K38-1 hold rule (late frame → the row's last on-time action), which is exact for them. Any row (incl. in-loop ones)
  whose actual latency still exceeds D holds its last on-time action.
- **Calibration** before any test run, same window: nuScenes val and the verified AV2 CFG-DEV manifest, first 712
  frames, 200 warm-up, no stopping; DESP / B2K-A / B1K-A medians by K39 `medians.py`, iterated K41 rows by
  `medians41.py`; sha256 of both files in `logs/state.json`.
- **Frames / repetitions.** nuScenes test 20 scenes (805 timed frames after the 200-frame warm-up pass) and the 40 AV2
  test logs of K38/K39 (`av2t40`, first 200 frames warm-up), ego mask 1830; 80-frame smoke; three repetitions, rows
  rotated by repetition (`--rep`), dataset order alternated.
- **Report.** Per row and D: G, deadline-miss share, mean / median / p95 / p99 / max latency, px; paired unit
  bootstrap (2,000, seed 0, repetitions nested in units) of DESP@D − row@D and B2K-A@D − row@D; Holm over rows per
  dataset and D in addition to raw intervals. Device action parity of the K41 rows against their VUW K33/K30 test records
  on the shared frames is reported (agreement share), not used as a gate (the VUW nuScenes draft is the cached torch
  one). All rows are reported; nothing is tuned on test latency or G.
- **J02** (same window, after step 3): one `--breakdown` repetition per test set (v6-S / bounded rows; per-stage CUDA
  syncs: draft, field, crops per round, guard, lift, C1, refiner, merge), kept out of the clean timings.
- **J03** is covered by the same clean repetitions (DESP, B2K-A, whole-frame sides, capped hybrid, iterated AnA under one
  protocol). No candidate optimization is included: JP01 and JP03 failed the registered tail screen, so the "optimized
  DESP" lane equals the original and **J01** (full-dev/test parity of an adopted candidate) is not run. Energy: window-level
  tegrastats rails only; per-row energy is not identifiable when rows interleave per frame and is not claimed.

## 2026-11-01 addendum — D07 executable definitions (registered before any D07 outcome)

Driver `remote/A_navsim/k41_d07_k2.py`, input the P02 records (1,000-token K20 seed-0 subset). **k2 field** = frozen
singleton field (`field_test`) + the cells of every moving adjacent k = 2 pair whose two cells lie in the plan corridor
(K20(3): lateral offset from the P02 draft plan ≤ 1.75 m + half the probe width); **k2all** = all moving adjacent pairs
(superset, secondary). Crops by the final run's `spec_pdm_dets` (one crop per lane column of field cells + support box
crops, guard, merge, scale 0.5) on CAM_F0 for tokens whose field changes; the singleton is recomputed on those tokens as
detector parity against the stored `spec_pdm`. Rows on all 1,000 tokens: cheap, full960 (full perception), singleton
(stored spec_pdm), k2, k2all, geometric (stored K30 hybrid_pdm: route tube + support). Endpoints: plan agreement with the
full960 plan (max pose distance ≤ 0.5 m), official NAVSIM v1.1 PDMS on metric caches rebuilt from the original scenes (as
P01; a 3-token smoke reproduced the stored original PDMS of cheap / spec_pdm exactly), crop px, planner calls (P02) and
detector wall time; paired log-bootstrap CIs of k2 − singleton and k2 − geometric. Gate: the rebuilt cheap plan must equal
the P02 draft plan on every token. If no token changes, k2 ≡ singleton on the subset and only that is reported.
