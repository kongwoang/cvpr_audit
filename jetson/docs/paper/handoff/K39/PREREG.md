# K39 — pre-registration (written 2026-10-27, before any K39 run or analysis)

Scope and constraints:
- Post-hoc. The method and the frozen v6-S configuration are unchanged.
- Test splits already used: nuScenes new test (228 scenes), AV2 val with the K10 ego mask (150 logs), Argoverse-HD val
  (24 sequences). Plus the K38-1 Jetson frame sets.
- The K35-4 holdout is not touched; it has been used twice.

Statistics, unless stated otherwise:
- Paired unit bootstrap with 2,000 draws, seed 0, 95% percentile interval. Units are scenes, logs or sequences.
- Δ = DESP − row in G = (ΣJ_cheap − ΣJ) / ΣJ_cheap. Δ > 0 means DESP is better.
- Every row is reported, and no row is dropped after seeing test numbers.
- Amendments go in a dated §A at the end, written before the run they concern.

## K39-1 (Jetson AGX Xavier) — latency–decision frontier at matched measured cost

**Frames.** Exactly the K38-1 sets:
- nuScenes: the K29 test 20 scenes (805 timed frames).
- AV2: test (val), the first 40 logs in sorted order (av2_test40_logs.txt, ego mask 1830). The 200-frame warm-up pass
  is excluded from timing.

**Run conditions:**
- MAXN, clocks locked, fan 255; TensorRT FP16 (trt32 fast path); image decode excluded.
- 3 repetitions. All rows run in the same session and the same process per repetition, in rotated order, each with its
  own history (K38-1 harness).
- Exclusive window: `~/.jetson_pause`, then busy.sh shows paused, and DEEP is informed.

**A. Output-identical engineering.**
- Candidate set, fixed now:
  - (A1) CUDA-graph capture of TensorRT executions per (engine, input shape), cached lazily;
  - (A2) page-locked host buffers and asynchronous device-to-host copies for detector and refiner outputs;
  - (A3) running the C1 encoder / refiner on a second CUDA stream overlapping independent detector work;
  - (A4) fusing crops of one round, and the guard re-detections of one growth step, into the fewest TensorRT executions
    that keep each crop's canvas. Crops are already grouped per canvas; fusion across canvases would change the network
    input and is excluded.
- Development and timing of the candidates happen only on dev prefixes: nuScenes val and the AV2 CFG-DEV manifest, as in
  K38-1 medians.
- A candidate is kept if two conditions hold:
  - its per-frame outputs are identical to the pre-change runtime on the dev prefix (actions, crops, grown crops, px of
    v6-S and B2K-A);
  - its dev mean latency is lower.
- **Gate on test.** Parity is checked on 100% of the timed test frames: the per-frame actions, crops and px of v6-S and
  B2K-A must equal the stored K38-1 records (logs/frames.pkl of the K38-1 staged runs) and K34-3 jetson_frames.csv.
  - Result goes in parity.json.
  - If parity < 100%, the change is dropped, the rows are rerun without it, and the failure is reported.
  - Before / after latency is reported for every kept change.
- Kept changes are applied to every row they apply to, whole-frame rows included, so the rows stay comparable.

**B. Rows:**
- cheap;
- whole frame + comp. (C1 + refiner v3a) at long side 512, 640, 800, 960, 1280 and native. Static engines exist for
  every side except 800. 800 runs on the existing dynamic-shape engine yolov8s_dyn (batch 1, ≤ 1024 per side), so no
  engine is built. u960 is also run on the same dynamic engine ("u960-dyn", a control for the dynamic-engine overhead);
  this row is reported but is not a frontier point;
- hybrid tsp8 + comp. under DESP's per-frame crop-px budget: the K30 J1 rule, nominal budget = v6-S's realised crop px of
  the frame, applied before the guard as in Tab. 1; its realised px is reported;
- DESP v6-S;
- DESP B2K-A: ≤ 2 crop batches after the draft, pending truncated detections kept (K31 run_bounded_keep, mode A);
- DESP B1K-A: ≤ 1 crop batch, same rule. This is a post-hoc frontier point, labelled as such.

**C. Per row:**
- mean, p50, p95 and p99 latency;
- TensorRT engine executions per frame (draft + crop/guard + whole frame + C1 encoder + refiner);
- realised px, guard included;
- G on the timed frames, pooled over repetitions. The cost is K28 decision_cost with K30 GT; prev = the previous cheap
  action of the run, warm-up included, as in K38-1.

**Deadline sweep.** D ∈ {33.3, 50, 66.7, 83.3, 100, ∞} ms.
- DESP variants (v6-S, B2K-A, B1K-A) run in-loop per D with the K38-1 rule:
  - before each round, stop if elapsed + median(next round) + median(finish) > D;
  - medians are re-measured on the dev prefixes with the final runtime before any test run, then hashed into
    medians.json;
  - B1K-A uses the same rule.
- Every row then applies hold: if latency > D, the frame keeps the row's last on-time action (KEEP before the first one).
- Reported per (D, row): G(D), late share, missed / false episodes (K31, L = 1).
- Paired bootstrap of (DESP variant@D − row@D), with repetitions nested in units, for every whole-frame row and the
  hybrid.

**D. Outputs:** k39_1_rows.csv, k39_1_deadline.csv, k39_1_diffs.csv, parity.json, and a frontier plot per dataset
(x = p95 latency, y = G, D = ∞; plus G(D) curves).

## K39-2 (VUW GPU) — strict post-guard per-frame pixel cap

**Rule.** Per frame:
- budget = DESP v6-S's realised crop px on that frame, guard included (the same accounting as px: canvas px of every
  crop-detector call, counted on cache hits too).
- Crops and guard re-detections run in the existing order: the row's selection order, each crop followed by its guard
  growth steps.
- A detector call whose canvas would push the cumulative crop px above the budget is skipped, and later smaller calls
  may still run.
- If a guard re-detection is skipped, the pending truncated detection is kept as detected. Without the cap it would be
  replaced, or dropped if the grown crop showed nothing.
- Assertion: realised px / DESP px ≤ 1.00 on every frame. A violation is reported.

**Rows.** The 8 S heuristics of K30 V1a, + C1 + refiner v3a, under the strict cap:
- tsp2 and tsp8 (support + ego-corridor tube; tsp8 = the hybrid);
- unc2 and unc4 (uncertainty);
- tcr0 and tcr3 (temporal criticality);
- cor0 and cor8 (ego corridor).

**Primary rows.**
- tsp8 for the hybrid.
- For each of uncertainty, temporal criticality and ego corridor, the variant with the higher dev G at the S budget in
  K28 phase 1 (baseheur_r2.csv: nuScenes val for nuScenes, AV2 CFG-DEV for AV2). Argoverse-HD uses AV2's choice, as in
  K33.
- The other variants are secondary.

**Parity.** The same run also computes the nominal m = 1 rows (K30 rule). Their a / J / px must equal the stored K30
match records, and v6-S must equal K28 p3.

**Reported:**
- G and realised px / DESP px (mean and max per frame);
- Δ = v6-S − row with CI, for the strict and the nominal cap.

**Outputs:** k39_2_rows.csv, k39_2_diffs.csv.

## K39-3 (CPU) — multiplicity sensitivity

**Family (a), Tab. 1 @cap, primary: 12 methods × 3 datasets = 36.**
- The methods: AdaScale, SAHI, FOVEA KDE_I, FOVEA KDE_S, BAR router, decision-value router, UFP, Chanakya, AnA one
  pass, AnA iterated, counterfactual one pass, counterfactual iterated.
- Excluded: Chanakya dev-best (px unmatched, a reference row) and AnA t 0.25 ("reported in any case"). They are reported
  as a sensitivity family of 14 × 3 = 42.

**Family (b), ablation: field only, support only, pivots instead of support (with refiner v3a) × 3 datasets = 9.**
- Source: K28 p3 P0 abl (k24v6 with p3).

**Resampling.** The deltas and the 2,000 resamples are each table's own:
- (a) k33_report.delta: sorted units, default_rng(0) integers;
- (b) k24_analyze / k28 Boot.
- The published CIs must be reproduced exactly; this is a check.

**p-values.** p = 2 · min(P*(Δ ≤ 0), P*(Δ ≥ 0)) over the bootstrap Δ, capped at 1.
- Holm–Bonferroni and Benjamini–Hochberg are applied within each family at α = 0.05.
- Sensitivity: the +1-corrected p = 2 · min(#{Δ* ≤ 0} + 1, #{Δ* ≥ 0} + 1) / 2,001.

**Output:** k39_3_pvalues.csv (family, method, dataset, delta, ci_lo, ci_hi, p_raw, p_holm, p_bh, plus the
+1-corrected columns).

## K39-4 (CPU + one VUW GPU part) — exactness cross-tabulation

**Cross-tabulation (CPU, K28 p3 test instr records, required action = act_gt).** Per dataset, eight cells
(DESP = π(F)) × (DESP = required) × (π(F) = required), with frame counts and shares.
- Without refiner: DESP = a_v6S|none and π(F) = a_wf960+C1|none.
- With refiner: a_v6S|v3a and a_wf960+C1|v3a.
- The same for native + comp. vs π(F): native = wf1600 / wf2048 / wf1920 + C1, with the matching refiner setting.

**Single interventions (VUW GPU, the same run as K39-2).** The deployed procedure is without refiner. Each intervention
of the attribution chain (k28_p3 CHAIN) is applied alone to v6-S with real crops and its own history:
- a: exact locality (the crop detector replaced by F = u960's detections whose centre is in the crop; as ch_a);
- b: F's geometry. Each real-crop detection takes the C1 lift of the F detection with the highest IoU, if IoU ≥ 0.5;
  otherwise it keeps its own C1 lift;
- c1: closing probe 10 m/s;
- c2: closing probe 20 m/s;
- d: extended grid;
- e: three probe templates;
- f: round cap 10.

**Measures.**
- D0 = frames with a_v6S|none ≠ a_wf960+C1|none.
- share_X = |{f ∈ D0 : a_X|none = a_F|none}| / |D0| per intervention. These shares are not cumulative and may overlap.
- Also reported: frames resolved by no single intervention, and frames newly made to disagree by X.
- Check: the recomputed a equals the stored ch_a on every frame.

**Outputs:** k39_4_crosstab.csv, k39_4_single.csv.

## K39-5 (CPU, VUW) — NAVSIM planner-specific field

Tokens: navtest, the 12,146 tokens of the stored runs (results/navsim_exp: field_test, dets_final, pdms_final).

**Footprints.** Both are round-1 regions on the same cheap draft:
- PDM: the probed PDM-Closed field, i.e. spec_pdm's crops from the stored field_boxes / support_boxes (one crop per
  lateral column, plus support crops; navsim_detect.spec_pdm_dets);
- braking: v3's braking field + support crops on the draft, recomputed on CPU with the NAVSIM v3 code (nuplan_spec_real
  / VS), the same v, camera and draft.
- The footprints are compared as union masks on the native CAM_F0 image.
- "Differs" (primary) = mask IoU < 0.9. Also reported: IoU < 1 (any difference), IoU < 0.5, the IoU distribution, and
  the PDM-only area share.

**Selected trajectory.** PDM-Closed (navsim_pdms.plan) is rerun on spec_pdm's and on spec's (braking-field v3)
detections. They differ if the 4 s poses differ by more than 0.5 m (MOVE_M) at any time step.
- Check: replanning on the cheap draft reproduces the stored base poses (results/k20/base__*).

**ΔPDMS.** score_spec_pdm − score_spec (stored), on the tokens with a differing trajectory and on all tokens. CI by
paired log bootstrap (2,000, seed 0).

**Wall clock per frame** (stored K20 timings, no new timing):
- 1 worker: the identical-output fast implementation (results/k20/fast__*);
- 8 workers: parallel_w8.pkl;
- also the frozen implementation (cost__*, 1 thread);
- p50 and p95 of each.

**Top 5 scenes.** Sort by |ΔPDMS| descending, ties by token ascending.

**Output:** k39_5_summary.csv (+ k39_5_top5.csv).

## K39-6 (CPU, stored test records) — target-action fidelity vs compute (registered 2026-10-27, before computing)

**Target.** π(F) = a_wf960+C1|none from K28 p3 test instr (u960 + C1, no refiner).
- Every row is taken WITHOUT refiner, with the braking controller and its own same-fidelity history, as stored. The
  history convention is Tab. 1's.
- GT frames of the K28 test splits: nuScenes 9,181, AV2 11,818 (ego mask), Argoverse-HD 15,006.
- Transitions (c) are computed on consecutive stored frames of a unit; the first frame of a unit is excluded.

**Rows and sources.** No new detector run; a row with no stored no-refiner action is reported as missing.
- cheap: K28 instr.
- Whole frame + C1, sides 448 / 512 / 640 / 960 (= F) / 1280 / 1536 / native: K28 {ds}_test_v6basewf-yolo11m (the
  YOLOv8s columns).
  - Check: its a_wf960+C1|none equals F on every frame.
  - 800 has no stored test record on the full splits (Jetson subsets only), so it is reported as missing. AV2 has no 448.
- v6-S, support only (so6S), field only (fo6S), pivots (pv6S): K28 instr (|none).
- AnA one pass / iterated, counterfactual one pass / iterated, published AdaScale / SAHI / FOVEA KDE_I / KDE_S / UFP:
  K33 de / pub, uncapped and @cap (|none).
- BAR router, decision-value router (router-G), Chanakya: K33 stored only routing sizes and refined costs.
  - Their no-refiner action is derived from the stored per-frame routing size: a_wf<s>|none of K33 pub when routed,
    else a_cheap (k33_rc's rule: a routed frame takes that whole-frame row's boxes).
  - Gate: recomputing the stored refined J from the same rule with a_wf<s>|v3a must reproduce J on every frame;
    otherwise the rows are reported as missing.
- Heuristics: tsp8 (hybrid) and the dev-best uncertainty / temporal-criticality / ego-corridor variants (K39-2 rule:
  nuScenes unc4 / tcr3 / cor0; AV2, Argoverse-HD unc2 / tcr0 / cor8), each in three forms:
  - "Tab. 1 form": @S, K28's S budget, before the guard; K28 instr;
  - under the nominal per-frame cap x1.00: K30 match;
  - under the strict cap: K39 vuw, extra.
  - No uncapped heuristic exists on test.

**Metrics** per row and dataset:
- (a) A = share of frames with action = π(F);
- (b) A | a_cheap ≠ π(F);
- (c) A | π(F)_t ≠ π(F)_{t-1};
- (d) A | act_gt ≥ DECELERATE, and A | π(F) ≥ DECELERATE;
- (e) on frames with action ≠ π(F): shares where the row equals act_gt, where π(F) does, and where neither does;
- (f) mean realised px (incl. guard, as stored) and detector passes per frame.
  - Detector passes: draft 1 if used, + stored crop-detector calls (ncrops + nguard), when stored.
  - Sequential crop batches: = rounds for loop rows, 1 for one-pass crop rows, 0 for whole frame; stated where stored.

**Statistics.** Paired unit bootstrap (2,000, seed 0; sorted units, default_rng(0) integers) of v6-S − row for (a) to
(d), as ratios of per-unit sums.

**Outputs:**
- k39_6_fidelity.csv, k39_6_diffs.csv;
- a plot per dataset (x = log px, y = (b));
- a REPORT section that states plainly where v6-S is NOT best.

## §A Amendments

### A1 (2026-10-27 10:20, after the dev engineering checks and before any K39-1 test run)

**Part A outcome** (dev only: nuScenes val prefix, 80 timed frames; logs in experiments/k39_jetson_20261027):
- **A1, CUDA graphs:**
  - For the dynamic-shape crop engine: dropped. Replay of the captured graphs crashed with a CUDA illegal memory access.
  - For the fixed-shape detector engines (cheap draft and static whole-frame engines; graphs.py mode "static"): kept.
    Per-frame actions / px / crops / executions were identical on every frame and every row. Dev mean latency was lower
    on every row: 0.2–0.4 ms for the DESP rows and 0.4–1.1 ms for the whole-frame rows.
- **A2, pinned / async host copies:** not implemented. The measured device-to-host copy is 0.15–0.4 ms per crop call
  (bench.py), so there is no measurable room.
- **A3, overlap:** not implemented. Inside a round, every engine call depends on the previous one (field → crops →
  guard → lift). The harness's speculative-overlap path ("ov") is not wired into the K38 loop.
- **A4, fusion:** already in the runtime and left as is. Crops of one round are batched per canvas (StagedCrops) and the
  guard re-detections of a growth step together (guarded_batch). Fusing across canvases changes the network input
  (excluded above).
- The kept change applies to every row, including whole frame, cheap and hybrid.

**Test gate:** actions / px / round and crop counts of v6-S and B2K-A must equal the stored K38-1 staged records.
K38-1 did not store crop rectangles, so the crop comparison is by count and px; K39 runs store the rectangles.

**Hybrid budget, precisely:** STATS tpx of the D = ∞ v6-S row of the same frame, i.e. the canvas px of every
crop-detector execution including the guard. This equals the K30 V1a / K39-2 nominal budget.

**u800:** runs on yolov8s_dyn as one whole-image call with canvas ceil32(side · scale): nuScenes 480 × 800, AV2 800 × 608.
The u960-dyn control measures the dynamic-engine overhead (dev: +1.2 ms mean vs the static u960 engine).

**Medians:** for v6-S, B2K-A and B1K-A, re-measured with --calibrate on the dev prefixes (nuScenes val, AV2 CFG-DEV;
712 frames, 200 warm-up) with the final runtime (graphs static), in the window before any test run → medians_k39.json
(sha256 recorded in the window state).
