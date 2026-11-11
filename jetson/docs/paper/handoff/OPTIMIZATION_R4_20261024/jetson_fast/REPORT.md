# R4 simple readout — Jetson (fast)

Three repetitions of 512 timed development frames per dataset after 200 warm-up; MAXN, locked clocks, fan255, exclusive window. Existing trt32 engines; no new model or engine. Rotated pipeline order; independent histories.
Paired unit bootstrap 2,000 seed0, repetitions nested within scenes/logs; pooled percentiles. Fixed development prefixes are runtime feasibility evidence, not a replacement for full-population quality.
Both median rows use the same round1 acquisition. Skipping preserves median ACTION, not the baseline v6-S action or continuous geometry. Exact full/skip action parity is checked on all warm-up and timed frames.
Source/engine integrity, fresh baseline-port smoke parity and hardware restoration are audited.

| Dataset | Pipeline | Mean ms (95% CI) | p95 ms | p99 ms | >50 ms % | >66.7 ms % | >100 ms % |
|---|---|---:|---:|---:|---:|---:|---:|
| av2_dev | objectwise median full | 35.00 [20.64, 51.08] | 84.18 | 109.66 | 26.50 | 13.48 | 1.76 |
| av2_dev | objectwise median skip | 34.42 [20.37, 50.05] | 83.26 | 106.72 | 25.39 | 12.50 | 1.37 |
| av2_dev | round1 combined | 34.85 [20.67, 50.73] | 84.35 | 109.39 | 26.11 | 12.83 | 1.43 |
| nusc_val | objectwise median full | 23.72 [18.72, 28.99] | 49.87 | 60.36 | 4.88 | 0.26 | 0.00 |
| nusc_val | objectwise median skip | 23.58 [18.62, 28.84] | 49.45 | 60.32 | 4.56 | 0.33 | 0.00 |
| nusc_val | round1 combined | 23.86 [18.84, 29.13] | 50.33 | 59.80 | 5.14 | 0.33 | 0.00 |

Every paired effect and interval is in contrasts.csv / intervals.csv. Pixels count padded detector work including historical recovery; mean refinement counts are in refinement_counts.csv.
Timing excludes image decode and model startup. Quality on these fixed prefixes is diagnostic only. Cross-VUW decision differences are reported separately. No real-time guarantee or core promotion.
Refinement item reduction is not proportional latency reduction: matching, CNN batches and detector work remain. Process peak memory includes all benchmark rows and is not a per-method deployment estimate.
Reproduce: scripts/py experiments/optimization_r4_20261024/report_jetson.py --variant direct (or --variant fast). Exact commands, warm-up and hardware logs: window_audit.json.
