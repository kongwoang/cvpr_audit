# K34 — completed post-hoc analyses of frozen v6-S

All four items completed; no configuration changes or test-driven selection. Paired unit bootstrap: 2,000, seed 0.

K34-1, K34-2 and K34-4 completed on VUW/VUW1 after K34-3 Jetson timing. No model/configuration selection, no detector rerun. Every requested stratum, including empty and unspecified classes, is retained. Intervals are 95% paired scene/log/sequence bootstrap, 2,000 draws, seed 0; pointwise exploratory intervals.

**W1 is qualified, not refuted.** v6-S misses more below-hard-demand episodes than cheap on AV2 and Argoverse-HD (paired intervals exclude zero). High-demand Argoverse-HD episodes also accrue more insufficient-braking loss than cheap; only ten such episodes exist. Dataset-wide insufficient-braking loss versus native + C1 + v3a is inconclusive on all three datasets, which does not establish equivalence. The deployed v6-S point is not dominated by the sampled whole-frame points on nuScenes/AV2, but it is dominated in point estimates on Argoverse-HD.

## K34-1: required episodes and insufficient braking

Evaluated frames: nuScenes 9,181 / 228 scenes; AV2 11,818 / 150 logs, fixed ego mask; Argoverse-HD 15,006 / 24 sequences. Required episodes follow K31 (>= DECELERATE, merge gaps of at most one evaluated frame). Missed onset is capped at episode length. Argoverse-HD retains K31 evaluated-frame adjacency despite gaps in annotation timestamps. Severity is peak annotated demand <3.5 versus >=3.5 m/s². Cheap shortfall bins are {0}, (0,1), [1,3.5), [3.5,8), [8,infinity). Loss is lam_risk × max(required demand − commanded braking, 0)², excluding the separate collision penalty. Episode shares use cheap total decision loss across the whole dataset as denominator.

| Dataset | Peak demand | Episodes | Missed cheap | Missed v6-S | v6-S − cheap [95% CI] |
|---|---|---:|---:|---:|---|
| nuScenes | below_hard | 109 | 0.404 | 0.477 | +0.073 [-0.011, +0.167] |
| nuScenes | at_or_above_hard | 38 | 0.184 | 0.211 | +0.026 [-0.061, +0.119] |
| AV2 | below_hard | 135 | 0.126 | 0.222 | +0.096 [+0.034, +0.162] |
| AV2 | at_or_above_hard | 55 | 0.073 | 0.109 | +0.036 [+0.000, +0.094] |
| Argoverse-HD | below_hard | 61 | 0.197 | 0.410 | +0.213 [+0.083, +0.355] |
| Argoverse-HD | at_or_above_hard | 10 | 0.000 | 0.200 | +0.200 [+0.000, +0.500] |

All seven pipelines, onset delays, missed/detected episode loss decomposition and marginal/crossed strata: severity_summary.csv, severity_intervals.csv, severity_contrasts.csv; episode-level rows: episodes.csv. Zero-episode rates are NA; zero loss sums remain zero.

For episodes whose peak demand reaches thr_hard, v6-S minus cheap insufficient-braking loss as a share of cheap total dataset loss is -0.1093 [-0.2404, -0.0093] on nuScenes, -0.0075 [-0.0675, +0.0609] on AV2, and +0.0234 [+0.0029, +0.0515] on Argoverse-HD. Thus the negative missed-episode result cannot be dismissed as affecting only mild episodes.

| Dataset | v6-S − native + C1 + v3a insufficient loss per frame [95% CI] |
|---|---|
| nuScenes | +0.0020 [-0.0551, +0.0494] |
| AV2 | -0.0066 [-0.0737, +0.0518] |
| Argoverse-HD | +0.0018 [-0.0130, +0.0148] |

These are decision-loss components on annotated frames, not closed-loop collision outcomes. Reproduction checks against K31 unstratified episodes and stored K28 costs are in validation.csv.

## K34-1: operating curves

The five-point grid holds thr_hard=3.5 m/s² fixed. Cheap uses cached geometry. Whole-frame C1 spatial geometry is reconstructed from cached boxes/calibration, and its stored TTC is retained: v3a rescales spatial coordinates but never modifies TTC. Every pre-refiner base action is checked against K30 AP-dump records. For v3a, support is recomputed at every threshold and the frozen refiner runs on stored-frame crops; per-object network outputs are reused only for identical inputs. This recovers threshold-specific scenes without C1 or detector inference. continuous_demands.csv stores every demand. Cross-source/backend base discrepancies are reported in validation.csv.

The primary GT required episodes remain fixed at the deployed threshold 1.0. K30 joint-threshold runs (0.8,2.8) and (1.2,4.2), plus their base, are separate labelled points; their original threshold-specific GT is a separate sensitivity protocol. v6-S uses only stored deployed/K30 points. No interpolation between sparse v6-S points or best-threshold selection.

- nuScenes: deployed v6-S false episodes/1,000 frames = 4.030, missed rate = 0.408. Among 26 reported whole-frame rows (20 exact-grid rows plus six separately sourced K30 rows, including duplicate base operating points), 0 point-estimate dominate v6-S and 9 are dominated by it; others trade off or tie. Paired coordinate intervals are in operating_pairwise.csv.
- AV2: deployed v6-S false episodes/1,000 frames = 6.515, missed rate = 0.189. Among 26 reported whole-frame rows (20 exact-grid rows plus six separately sourced K30 rows, including duplicate base operating points), 0 point-estimate dominate v6-S and 6 are dominated by it; others trade off or tie. Paired coordinate intervals are in operating_pairwise.csv.
- Argoverse-HD: deployed v6-S false episodes/1,000 frames = 6.264, missed rate = 0.380. Among 26 reported whole-frame rows (20 exact-grid rows plus six separately sourced K30 rows, including duplicate base operating points), 4 point-estimate dominate v6-S and 6 are dominated by it; others trade off or tie. Paired coordinate intervals are in operating_pairwise.csv.

Concrete counterexample to being on/below the whole-frame curves everywhere: on Argoverse-HD, u960 + C1 + v3a at thr_decel=0.8 and fixed thr_hard=3.5 has missed rate 0.155 versus deployed v6-S 0.380 and false episodes/1,000 frames 5.531 versus 6.264. Paired v6-S minus u960 differences are +0.225 [0.114, 0.320] in misses and +0.733 [-0.432, 2.028] in false episodes; the latter interval includes zero, so simultaneous statistical dominance is not established.

Point-estimate dominance is not a simultaneous significance claim. The comparisons establish only relations among measured points, not superiority to an unmeasured continuous envelope. All points and intervals: operating_points.csv, operating_intervals.csv. Figure: operating_curves.pdf/.png.

## K34-2: decisive-object coverage

Population: one decisive annotated object per frame with act_gt >= DECELERATE. Region = union of actual field/support/guard crops accumulated by the final round, checked against stored dec_in_crop_v6S; proposed-but-unexecuted field boxes are excluded. Coverage is box-centre geometry, not detection recall. Dataset-wide decision loss is from the K28 actions. Class mappings are explicit; generic vehicle annotations are not forced into car/truck/bus. Signed closing speed is the negative longitudinal range derivative of annotated tracks, missing when no valid derivative exists; fixed per-dataset tercile cuts and ties are documented in stratum_definitions.csv.

| Dataset | Required frames | Covered [95% CI] | Uncovered conditional mean decision loss [95% CI] |
|---|---:|---|---|
| nuScenes | 928 | 0.918 [0.875, 0.948] | 10.678 [6.374, 15.252] |
| AV2 | 2854 | 0.954 [0.939, 0.969] | 5.058 [2.764, 8.521] |
| Argoverse-HD | 3593 | 0.959 [0.918, 0.984] | 2.528 [2.304, 2.920] |

All class/range/lateral/closing-speed/cheap-detection rows and uncovered loss sums/shares: coverage_summary.csv, coverage_intervals.csv, coverage_frames.csv, class_mapping.csv.

Argoverse-HD car/truck/bus coverage is not separately identifiable from its generic vehicle annotations. These rows are explicitly marked not_identifiable_from_annotations in the summary/interval tables and missing_rows.csv; zero classified rows must not be read as absence of those classes. Cyclist groups combine annotated bicycle/motorcycle classes as shown in class_mapping.csv.

## K34-4: selector cost

Frozen K31 15-candidate bottleneck selector on K20’s exact 1,000-token subset. One CPU thread, five warm-up repeats, alternating grid order; no image/detector inference. Timed scope includes scene construction, selector setup, base scoring, field probes and binding-support extraction; excludes disk loading. Every field/support output is compared to the original implementation, and 7×7 matches stored K31 first-round outputs. Binding support reuses the score matrix and needs zero additional selector evaluations.

| Method | Grid | Mean total ms [95% CI] | p95 total ms |
|---|---|---|---:|
| K31 bottleneck selector | 7x7 | 93.388 [78.862, 109.327] | 279.565 |
| K31 bottleneck selector | 3x3 | 22.359 [19.070, 25.973] | 64.935 |
| PDM-Closed historical K30 | 7x7 | 14757.150 [13188.432, 16412.380] | 23929.040 |
| PDM-Closed historical K30 | 3x3 | 4751.293 [4068.493, 5513.366] | 8366.664 |

PDM-Closed comparison uses K30’s measured identical-output fast implementation on the same tokens, not K20’s extrapolated 3×3 runtime. Host/load/protocol differ historically, and planner identities differ: this is context, not a controlled speedup of PDM-Closed. All component times, candidate-score and candidate-object counts, means/p95 and intervals: selector_cost_frames.csv, selector_cost_summary.csv, selector_cost_intervals.csv.

## Reproduction and provenance

Coordinator: python3 scripts/k34_remote_driver.py. Remote CPU scripts: scripts/A_method/k34_episodes.py, k34_replay.py, k34_operating.py, k34_coverage.py (--dataset nusc|av2|ahd); scripts/A_navsim/k34_selector_cost.py in env_navsim.sh. Every job command/host/status is mirrored under results/vuw_summaries/k34/jobs/. Source hashes: source_manifest.csv and remote per-unit replay metadata. Source/code remain frozen. Remote outputs: results/k34/; local summary mirror: results/vuw_summaries/k34/. Jetson result and parity caveats remain in JETSON_REPORT.md.


Final audit (2026-10-23): all 36,005 evaluated frames across 402 units are present. CPU replay cheap and whole-frame C1 actions, with and without v3a, match original K28 deployed actions on every frame. Source hashes are unchanged. Handoff checks cover 378 severity rows, 132 operating rows, 60 coverage rows, 1,000 NAVSIM tokens and 12,519 Jetson timing rows. Evidence: cpu_final_audit.json and handoff_audit.json. Three unavailable A-HD fine-class rows are retained in missing_rows.csv.

# K34-3 — Jetson timing

**K34-3 complete.** Every timing repetition is retained; CPU analyses are in CPU_REPORT.md.

K34-3 uses the existing trt32 fast path on Jetson AGX Xavier, MAXN, locked clocks and fan PWM 255. Three fresh repetitions, independent pipeline histories, balanced pipeline order. Exact K29 manifests: nuScenes 805 timed frames / 20 scenes (plus 200 warm-up frames), AV2 586 timed frames / 8 logs (200 earlier frames excluded, ego mask y=1830). Timings include frame upload/preprocessing, draft, crop detection, geometry, C1, refiner and action; disk I/O is excluded. No shared detection cache during timing.

Bounded variants limit crop batches after the draft to two; actual detector engine executions can differ because of batch splitting. B2K-A retains pending truncated detections at the cap, as K31 mode A. Original K29/K31 sources are unchanged. CPU port checks: 400 K31 comparisons and 400 unchanged-K29 comparisons passed; shared-detection GPU smoke parity passed before timing.

Intervals: paired scene/log bootstrap, 2,000, seed 0, 95% percentile. Each drawn unit carries all three repetitions. Quantiles are of pooled latency samples, not averages of per-run quantiles. These are pointwise intervals; AV2 has only eight logs. Fractions above thresholds use strict > comparisons.

| Dataset | Pipeline | Mean ms [95% CI] | p95 ms | p99 ms | >50 ms | >66.7 ms | >100 ms |
|---|---|---|---|---|---|---|---|
| av2 | v6-S | 40.39 [28.89, 53.65] | 91.22 | 120.02 | 35.67% | 16.21% | 3.19% |
| av2 | v6-S <= 2 | 36.47 [26.35, 48.38] | 72.28 | 91.96 | 28.61% | 7.34% | 0.51% |
| av2 | B2K-A | 38.46 [27.46, 51.48] | 77.40 | 102.38 | 37.60% | 9.61% | 1.08% |
| nusc_test20 | v6-S | 21.84 [16.86, 26.87] | 59.71 | 83.69 | 9.32% | 2.53% | 0.21% |
| nusc_test20 | v6-S <= 2 | 19.90 [15.76, 23.98] | 46.20 | 55.57 | 3.23% | 0.25% | 0.00% |
| nusc_test20 | B2K-A | 20.31 [15.99, 24.53] | 49.12 | 59.96 | 4.55% | 0.37% | 0.00% |

All metric CIs: jetson_latency_intervals.csv. All paired contrasts: jetson_latency_contrasts.csv. Per-run rows: jetson_latency.csv; raw evaluated frames: jetson_frames.csv.

B2K-A reduces mean and p95 versus v6-S on both datasets: paired mean differences are -1.53 ms [-2.83, -0.53] on nuScenes and -1.94 ms [-2.94, -1.03] on AV2; paired p95 differences are -10.59 ms [-14.65, -2.50] and -13.82 ms [-22.79, -6.35]. Retaining pending detections has a cost versus the ordinary bounded variant: mean +0.41 ms [0.20, 0.64] / +1.99 ms [1.04, 3.07]. On AV2, B2K-A does not improve every deadline: its >50 ms share is nominally higher than v6-S by 1.93 percentage points [-1.75, 5.88].

The two-crop-batch limit is satisfied on every bounded frame. It is not a two-engine-execution limit: the observed maximum including the draft is four on nuScenes and three on AV2. In the exported CSV, crop_engine_executions is reconstructed as total detector executions minus the one draft execution; the raw harness counter of that name only counts the bounded path and is zero for unbounded v6-S. Timing and action records are unchanged.

## Decision parity with VUW

| Dataset | Rep | Pipeline | Common / timed | Differing | Agreement |
|---|---|---|---|---|---|
| av2 | 1 | B2K-A | 586 / 586 | 2 | 99.659% |
| av2 | 1 | v6-S | 586 / 586 | 4 | 99.317% |
| av2 | 1 | v6-S <= 2 | 586 / 586 | 1 | 99.829% |
| av2 | 2 | B2K-A | 586 / 586 | 2 | 99.659% |
| av2 | 2 | v6-S | 586 / 586 | 4 | 99.317% |
| av2 | 2 | v6-S <= 2 | 586 / 586 | 1 | 99.829% |
| av2 | 3 | B2K-A | 586 / 586 | 2 | 99.659% |
| av2 | 3 | v6-S | 586 / 586 | 4 | 99.317% |
| av2 | 3 | v6-S <= 2 | 586 / 586 | 1 | 99.829% |
| nusc_test20 | 1 | B2K-A | 805 / 805 | 3 | 99.627% |
| nusc_test20 | 1 | v6-S | 805 / 805 | 3 | 99.627% |
| nusc_test20 | 1 | v6-S <= 2 | 805 / 805 | 3 | 99.627% |
| nusc_test20 | 2 | B2K-A | 805 / 805 | 3 | 99.627% |
| nusc_test20 | 2 | v6-S | 805 / 805 | 3 | 99.627% |
| nusc_test20 | 2 | v6-S <= 2 | 805 / 805 | 3 | 99.627% |
| nusc_test20 | 3 | B2K-A | 805 / 805 | 3 | 99.627% |
| nusc_test20 | 3 | v6-S | 805 / 805 | 3 | 99.627% |
| nusc_test20 | 3 | v6-S <= 2 | 805 / 805 | 3 | 99.627% |

All differing frames remain in the timings and are listed in jetson_disagreements.csv. Every timed frame has a VUW reference. Each pipeline has identical actions across all three repetitions; the same discrepancies recur. B2K-A differs on three distinct nuScenes frames and two distinct AV2 frames. All three nuScenes discrepancies also occur for the ordinary bounded variant; on AV2 one is shared and one is additional. The exact source of each discrepancy has not been isolated; differing detector/canvas backends and resulting temporal histories are possible causes, not an established diagnosis. Numerical equivalence to VUW is not achieved. See jetson_parity.csv for paired-unit agreement intervals and jetson_repetition_parity.csv for consistency across repetitions.

## Provenance and remaining work

Exact run commands/configurations and environment are in each raw run folder; run paths are recorded in jetson_latency.csv and logs/k34j/state.json. Source hashes: logs/k34j/source_hashes.json; run-time truncated C1/refiner/head checksums: each run/meta.json. Full post-run engine hashes and audit checks: jetson_audit.json (these do not substitute for pre-run detector hashes). Hardware and telemetry: logs/k34j/hardware_before.json, clock_before.cfg and tegrastats.log. K29/VUW reference manifests and hashes: reference/. Figure: jetson_latency.pdf and jetson_latency.png.

Driver: `python3 scripts/k34_jetson_window.py`. Harness: `scripts/py scripts/77_k34_latency.py --dataset {nusc_test20,av2} --config v6S --wf none --bounds 2 --paths trt32 --rep {1,2,3} --tag r{1,2,3}`; AV2 adds `--mask_y0 1830`. CPU report: `scripts/py scripts/78_k34_report.py`.

CPU analyses are now in CPU_REPORT.md. This timing result alone does not resolve W1 or establish a safety improvement.
