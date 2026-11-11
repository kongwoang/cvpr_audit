# K35 STATUS

Owner: the kit agent (Jetson session), from 2026-10-25. The Mac sessions only read the Jetson; nobody else claims K35.
Task text: the user's K35 spec and K35 ADDENDUM (2026-10-25), reproduced in TASK.md.

## Hard rules (unchanged from K28–K34)
- These stay unchanged: frozen v6-S and its weights; thresholds tau1 1.0 and tau2 3.5 m/s²; K31 episode definitions;
  loss weights; splits; baseline forms (K28 R2).
- Historical handoffs are immutable.
- No test-driven selection. Every attempted row is reported.
- Statistics: paired unit bootstrap, 2,000 resamples, seed 0. Post-hoc items on test splits are labelled post-hoc.
- Outputs: handoff/K35/<item>/REPORT.md + CSVs.
- Stop point for any method change: 2026-11-03; after it, only evidence and timing.
- No pipeline run before K35-4's pre-registration (K35-0 is GT-only).

## Items and order (priority)
| item | priority | resource | gate | status |
|---|---|---|---|---|
| K35-0 untouched labelled holdout inventory (GT only) | P0 | CPU | first | DONE 2026-10-25 → K35-0/REPORT.md: PASS (AV1.1 TRAIN, 65 units, 154 episodes) |
| K35-1 missed-episode decomposition intervals + attribution (post-hoc, stored records) | P0 | CPU | — | DONE 2026-10-25 → K35-1/REPORT.md |
| K35-2 recoverability classes + best-of ceiling (post-hoc) | P1 | CPU | — | DONE 2026-10-25 → K35-2/REPORT.md |
| K35-3 periodic / random temporal escalation baselines (post-hoc, run once) | P1 | CPU (no-history replay + stored rows) | — | DONE 2026-10-25 → K35-3/REPORT.md |
| K35-4 confirmatory holdout (v6-S, baselines, v7a) | P0 | GPU, once | K35-0 pass + PREREG | **DONE: v7a passes (narrowly)**: ΔG +0.003 [−0.002, +0.011], missed 42 vs 43 / 154; px −10%, false +1.0 / 1,000 |
| K35-4b v7b secondary holdout candidate | P1 | Jetson gate first | K35-0 pass; v7b p95 ≤ v6-S on DEV prefixes | **DONE: v7b dropped** (nusc p95 65.26 > 62.75 ms) |
| K35-5 zero-value-of-computation test | P2 | GPU | — | DONE 2026-10-25 → K35-5/REPORT.md |
| K35-6 reproducible selection rule for figs/q6_* | P2 | CPU | — | DONE 2026-10-25 → K35-6/REPORT.md + panels/ |
| K35-8 copy R4 exact score-reuse selector costs | P2 | none | — | DONE 2026-10-25 → K35-8/REPORT.md |

## Log
- 2026-10-25 (kit 08:40): K35 received; folder and STATUS created before any computation. Open item outside K35: the
  K32 REPORT.md (the test finished 2026-10-22; final tables in handoff/K32/auto/).
- 2026-10-25 (VUW 14:40–15:30):
  - **K35-0.**
    - (a) AV2: results/k23/av2_split.csv assigns all 700 AV2 sensor train logs (datasets/av2/lists/train): CFG-DEV
      300, REF-TRAIN 340, REF-VAL 60. So (a) is empty.
    - (b) Argoverse 1.1: VUW only had tracking VAL (24 logs, = Argoverse-HD val, used).
      - The 65 TRAIN logs (= Argoverse-HD train.json seq_dirs) were never downloaded, and no script in scripts/ or
        code/ reads Argoverse-HD train.json (the zoomdet hits are its own AV2 train.json).
      - The TRAIN tars are being stream-extracted: calibration, poses, amodal 3D labels, ring_front_center images,
        city_info; train1–4 = 160 GB streamed (scripts/data/download_av1_train.sh, logs/data/av1_train.log).
    - (d), touched or unsuitable:
      - nuPlan cam_f0 (64 logs) is used by the A_nuplan PDMS chains;
      - NAVSIM mini (64 logs) is the same 64 nuPlan logs;
      - navtest / navhard are used (K30 / K31);
      - CODA (base-val-1500) has 2D boxes only and is used in K30 V8;
      - KITTI tracking has 21 labelled sequences (< 30 units) and is used (DEEP cells, K30).
    - (c) nuScenes: results/nusc_full/split.csv; all 850 scenes are in one of new {train 385, val 152, test 228} /
      old {train 45, val 16, test 24}. Per-scene provenance is in the systematic scan below.
    - A systematic scan is running: path + binary-safe content grep of every workspace result / data / log / cache /
      experiment tree for the 65 AV1.1 TRAIN log IDs and all 850 nuScenes scene names
      (scripts/A_method/k35_provenance.sh → results/k35/k0/).
  - **K35-1 DONE:** K35-1/REPORT.md.
    - Refiner v3a: v6-S − v6-S without refiner +0.020 (n.s.) / +0.047 / +0.169 missed rate (nuScenes / AV2 /
      Argoverse-HD).
    - Without the refiner, v6-S − cheap is n.s. on all three splits.
    - Cheap-detects / v6-S-misses episodes: mostly E2 (no finite C1 TTC) on nuScenes / AV2, mostly E3 (refiner) on
      Argoverse-HD.
- 2026-10-25: K35-2 DONE (K35-2/REPORT.md); K35-8 DONE (K35-8/REPORT.md, copied from OPTIMIZATION_R4_20261024/nav_selector).
- 2026-10-25 (VUW1 15:20–16:00): **K35-3 DONE** (K35-3/REPORT.md; run once).
  - k = 10 / 7 / 6 from the pixel match.
  - The deployment C1-history rule needed a CPU replay of u960 + C1 + v3a with no chain (k35_k3_replay.py; parity
    3,936 / 3,936 frames). The continuous-history router convention is reported as a labelled sensitivity row.
  - v6-S − periodic phase 0: G +0.233 / +0.276 / +0.352 (significant on all splits, for every phase and seed).
  - The escalation rows miss fewer episodes than v6-S but have about 5× its false episodes.
- 2026-10-25: **K35 TIMING NOTE** (user, verbatim in TASK.md). It applies to every Jetson timing of K35-4 / K35-4b: v7a,
  v7b and the dev-prefix p95 gate.
  - Use the output-identical runtime improvements already validated: the exact recovery bookkeeping helper
    (FAST_RECOVERY_20261023, as VERIFIED_JETSON did) and crop-buffer staging (RUNTIME_20261023).
  - Use the SAME runtime for v6-S and B2K-A in the same window, so every row gets the same implementation.
  - Check action parity against the reference runtime on all timed frames; report the parity rate and every differing
    frame. No other change.
- 2026-10-25 (VUW ~15:30):
  - **K35-0 GT protocol validated.** k35_k0_episodes.py applies the frozen Argoverse-HD evaluation protocol (k24_test
    'ahd' path) to train.json. On Argoverse 1.1 VAL it reproduces K31 / K34 exactly: 24 units, 15,006 labelled frames,
    71 required episodes (61 below hard / 10 at or above).
    - On the first 17 fully extracted TRAIN logs: 9,118 labelled frames, 37 required episodes (31 / 6), in 14 units.
    - Extrapolated, the 65 logs give about 140 episodes (≥ 100 needed) and 65 units (≥ 30). The decision waits for the
      complete extraction and the provenance scan.
  - **K35-5** smoke OK (one unit per split; on Argoverse-HD seq00 5 / 8,980 changed actions). Full run launched on VUW1
    (3 shards per split, results/k35/k5/).
  - **K35-6:** rule and selection done (results/k35/k6/capture_frames.csv, 22 frames, distinct per dataset). The
    capture + render run (k35_q6_chain.sh, tag k35q6) is on VUW1.
  - **K35-4 preparation:** v7a = "B2K-A + temporal evidence" = a_v6S_b2A_retro|v3a of experiments/w1_bounded_20261023
    (forked k31_eval; protocol.md: B2K-A two rounds with pending truncated detections retained at the cap, plus selective
    C1 retrospective recovery, ≤ 1 historical canvas at each of t−1 / t−2). The holdout needs an isolated dataset
    adapter (Argoverse-HD protocol on train.json); the frozen sources stay untouched.

## Next steps (the hourly K35 check follows this list; skip what is done)
1. **K35-6:** when VUW1 logs/k35/q6.log shows "K35 Q6 CAPTURE AND PANELS DONE":
   - copy results/k35/q6_panels and results/k35/k6/*.csv to K35-6/;
   - write K35-6/REPORT.md: the provenance of the existing panel set (K28 capture_frames.csv, categories, no
     within-category rule; existing_check.csv), the rule, qualifying counts, and the selected frames with their panels.
2. **K35-5:** when results/k35/k5/{nusc,av2,ahd} hold 228 / 150 / 24 units:
   - run scripts/A_method/k35_k5_report.py on VUW1 (CPU);
   - write K35-5/REPORT.md: changed-action rate vs 2.4 / 3.2 / 6.0%, dG with intervals, and the attribution
     (history / truncation / new object GT / new object spurious / other).
3. **K35-0:** when VUW logs/data/av1_train.log shows "AV1.1 TRAIN DONE" (65 logs) and results/k35/k0/DONE exists:
   - run k35_k0_episodes.py train;
   - analyse path_hits.tsv / content_hits.tsv for every candidate ID; the AV1 train IDs must have no hit outside the new
     dataset tree, the Argoverse-HD train.json and the K35 folders;
   - write K35-0/REPORT.md with the pass / fail verdict (proof of no use, ≥ 30 units, ≥ 100 required episodes).
4. **K35-4 holdout (running since 2026-10-25 20:09 NZ; K35-0 PASS, K35-4b done with v7b dropped, PREREG frozen):**
   - Hourly: VUW logs/k35/k4/train_{k28p3,k33,r3low}_{0..3}of4.log and train_group_*.log. Expect 65 units in each of
     results/k35/k4/train/{k28p3/ahd_test_instr, k33/ahd_de_test, r3low/main/ahd}.
   - A shard that died without its output: relaunch the same chain line (completed units are skipped), unchanged code.
     Record it here.
   - When all 3 × 65 units exist:
     - run `check.py train` (0 differences required, else stop and report);
     - then `k35_k4_analyze.py`;
     - copy the tables to K35-4/ and write K35-4/REPORT.md (all rows, kill-rule verdict as pre-registered).
   - If v7a passes: the post-hoc test-split run plus the Jetson K29-frame timing (K35 timing-note runtime).
- 2026-10-25 (VUW1 15:35): **K35-6 DONE.** Capture + render finished (22 frames × 6 panels × PDF / PNG, 0 errors).
  K35-6/REPORT.md is written; the panels are being copied to K35-6/panels/ (about 530 MB over the VUW link).
- 2026-10-25: K35-6 panels copied to the kit (265 files, 437 MB).
- 2026-10-25 (VUW1 16:45): **K35-5 DONE** (K35-5/REPORT.md).
  - Changed-action rate 0.00–0.09%, well below the 2.4 / 3.2 / 6.0% reference.
  - dG is about 0 on nuScenes / AV2. On Argoverse-HD it is slightly negative: −0.0012 / −0.0018.
  - The changes are mostly real objects outside the field (not decision-relevant) that, once found, raise the cost.
  - Download: train1 and train2 done (40 logs), train3 running. The provenance content grep is still scanning.
- 2026-10-25 (VUW 17:45): AV1.1 TRAIN extraction done (65 logs). Provenance scan done: 705,886 files, 0 path hits; 2 content
  hits, both a log ID hard-coded in a third-party av2 library file (not a use).
  - GT-only count: 65 units, 39,265 labelled frames, 154 required episodes (137 / 17).
  - All 850 nuScenes scenes are touched.
  - **K35-0 PASS** (K35-0/REPORT.md). Next: the K35-4b Jetson gate, then the PREREG, then K35-4.
- 2026-10-25: kit-side grep for the 65 AV1 TRAIN IDs (results, logs, experiments, docs, scripts, engines, ~/datasets).
  - Only hit: ~/datasets/argoverse_hd/Argoverse-HD/annotations/train.json, the dataset's own annotation file (not a use).
  - The K35-0 PASS stands.
- 2026-10-25 13:40 (kit): K35-4b gate window started (experiments/k35_gate_20261025; protocol.md registered before
  the window).
  - Rows: v6-S / B2K-A / v7a / v7b.
  - Every row uses the staged runtime + FAST_RECOVERY helper; the reference runtime is run once per dataset, for parity.
  - v7b = Jetson '_missing' + '_extent' flags composed, i.e. VUW R3 low `v6S_b2A_missing_extent_retro`.
  - Dry and smoke parity: 0 action differences.
  - The DEEP session was not running, so it could not be messaged; the pause-file protocol applies.
- 2026-10-25 (VUW): K35-4 holdout harness built (experiments/k35_holdout_20261025/build.py; isolated copies, unit source
  + output dirs only).
  - Adapter check on Argoverse-HD VAL seq05 is running, against the stored test records (check.py val).
  - No holdout run before PREREG.
- 2026-10-25 (VUW): K35-4 adapter check PASS on Argoverse-HD VAL seq05 (442 frames; results/k35/k4/val/check.json).
  - k28_p3_h vs the stored ahd_test_instr: 134 columns, 0 differences.
  - k33_eval_h vs ahd_de_test: 18 columns incl. ana05hinfi, 0 differences.
  - r3/runner_h cheap / v6-S vs stored: 6 columns, 0 differences.
  - v7a and v7b columns are present.
  - Next: the K35-4b gate result, then PREREG, then the single holdout run.
- 2026-10-25: **K35-4b DONE: v7b DROPPED** (K35-4b/REPORT.md).
  - nuScenes-val p95 65.26 vs v6-S 62.75 ms (point-estimate rule); AV2 115.18 vs 117.52 ms.
  - Action parity vs the reference runtime: 4 rows × 3 reps × 512 frames × 2 datasets, 0 differences (also every
    counter).
  - v7a p95 −11.9 / −30.3 ms vs v6-S.
  - Jetson released; hashes unchanged.
- 2026-10-25: **K35-4 PREREG frozen** (K35-4/PREREG.md sha256 931c7058…44182e90; copy in VUW results/k35/k4/).
  - v7a is the only candidate; kill rule as in TASK.
  - Analysis code k35_k4_analyze.py (sha256 00ffce85…) written and hashed before the run.
  - Holdout launched ONCE: k35_k4_chain.sh train 0/4 … 3/4.
- 2026-10-25 (hourly, ~21:40 VUW): K35-4 progress.
  - r3low (v7a): 65/65 units, all 4 shards DONE.
  - k33 (AnA): 55/65 (shard 0 DONE).
  - k28p3: 9/65 (slowest; all 12 processes share GPU 2; estimate ~8–10 h).
  - No errors, nothing dead.
- 2026-10-25 (hourly): K35-4 progress.
  - k33 (AnA): 65/65, all shards DONE.
  - r3low: 65/65.
  - k28p3: 19/65, 4 shards alive, no errors.
- 2026-10-26 00:21 VUW (hourly): K35-4 k28p3 42/65 (4 shards alive, no errors); k33 / r3low 65/65.
- 2026-10-26 03:40 NZ: K35-4 holdout finished (65 / 65 × 3 runners).
  - check.py train: 0 differences.
  - Pre-registered analysis (sha256 unchanged): **v7a PASSES** the kill rule. ΔG +0.0029 [−0.0020, +0.0108]; missed
    42 vs 43 / 154; Δpx −0.0079 (significant); Δfalse +1.02 / 1,000 (significant).
  - v6-S G 0.407 [0.313, 0.480]; all same-C1 + v3a baselines within ±0.01 G (n.s.); unc2 −0.069. W1 persists: v6-S
    misses 43 vs cheap 18.
  - K35-4/REPORT.md written.
  - (The background waiter never fired: its pgrep matched its own ssh command line. The check and analysis were run by
    hand, unchanged.)
- 2026-10-26 03:44 NZ: K35-4 consequence of the pass: v7a post-hoc on the three test splits launched
  (k35_k4_posthoc.sh; runner_t.py sha256 53557b0d…; posthoc_check.py 1c285495…; outputs results/k35/k4/posthoc/).
  - Next: the Jetson K29-frame timing of v6-S / B2K-A / v7a (K35 timing-note runtime).
- 2026-10-25 21:48–22:08 (kit): **K35-4 Jetson K29-frame timing DONE** (K35-4/jetson_k29/REPORT.md).
  - Parity vs the reference runtime: 0 differences.
  - v7a p95 vs v6-S: −7.7 [−12.6, −0.4] ms nuScenes, −4.8 [−17.1, +2.5] ms AV2; mean within ±1 ms.
  - B2K-A: −9.6 / −11.7 ms.
  - Jetson released.
- 2026-10-26: K36 and K36b DONE (handoff/K36, handoff/K36b).
- 2026-10-26 04:40 NZ: **K35-4 post-hoc test-split run of v7a DONE** (K35-4/posthoc/REPORT.md).
  - posthoc_check: 0 differences (228 / 150 / 24 units).
  - v7a − v6-S, ΔG: nuScenes +0.014 (n.s.); AV2 **+0.025 [+0.001, +0.063]**; Argoverse-HD −0.005 (n.s.).
  - Missed episodes 60 → 55 / 36 → 33 / 27 → 27; px −5 to −8%; false episodes +0.3 to +1.5 / 1,000.
  - **All K35 items are complete.** Remaining: paper-kit text.
