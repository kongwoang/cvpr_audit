# R4 simple readout — Jetson (direct)

Three repetitions of 512 timed development frames per dataset after 200 warm-up; MAXN, locked clocks, fan255, exclusive window. Existing trt32 engines; no new model or engine. Rotated pipeline order; independent histories.
Paired unit bootstrap 2,000 seed0, repetitions nested within scenes/logs; pooled percentiles. Fixed development prefixes are runtime feasibility evidence, not a replacement for full-population quality.
Both median rows use the same round1 acquisition. Skipping preserves median ACTION, not the baseline v6-S action or continuous geometry. Exact full/skip action parity is checked on all warm-up and timed frames.
Source/engine integrity, fresh baseline-port smoke parity and hardware restoration are audited.

| Dataset | Pipeline | Mean ms (95% CI) | p95 ms | p99 ms | >50 ms % | >66.7 ms % | >100 ms % |
|---|---|---:|---:|---:|---:|---:|---:|
| av2_dev | critical extent retention | 36.45 [21.10, 54.35] | 86.74 | 127.31 | 28.12 | 13.93 | 2.15 |
| av2_dev | objectwise median full | 36.17 [21.91, 52.16] | 86.98 | 110.21 | 27.67 | 13.67 | 1.95 |
| av2_dev | objectwise median skip | 36.46 [22.61, 51.96] | 86.19 | 104.85 | 27.15 | 13.35 | 1.63 |
| av2_dev | round1 combined | 35.03 [20.76, 51.02] | 84.82 | 109.80 | 26.37 | 12.83 | 1.89 |
| av2_dev | v6-S | 37.99 [20.46, 57.99] | 115.04 | 143.46 | 24.09 | 16.80 | 7.49 |
| nusc_val | critical extent retention | 24.80 [19.24, 30.63] | 51.72 | 65.58 | 6.51 | 0.78 | 0.00 |
| nusc_val | objectwise median full | 24.74 [19.68, 30.08] | 50.88 | 60.89 | 5.53 | 0.59 | 0.00 |
| nusc_val | objectwise median skip | 25.45 [20.40, 30.74] | 51.31 | 62.16 | 5.60 | 0.59 | 0.00 |
| nusc_val | round1 combined | 23.76 [18.73, 29.03] | 49.88 | 59.58 | 4.88 | 0.39 | 0.00 |
| nusc_val | v6-S | 25.57 [19.25, 33.07] | 61.70 | 80.32 | 10.29 | 2.99 | 0.20 |

Every paired effect and interval is in contrasts.csv / intervals.csv. Pixels count padded detector work including historical recovery; mean refinement counts are in refinement_counts.csv.
Timing excludes image decode and model startup. Quality on these fixed prefixes is diagnostic only. Cross-VUW decision differences are reported separately. No real-time guarantee or core promotion.
Refinement item reduction is not proportional latency reduction: matching, CNN batches and detector work remain. Process peak memory includes all benchmark rows and is not a per-method deployment estimate.
Reproduce: scripts/py experiments/optimization_r4_20261024/report_jetson.py --variant direct (or --variant fast). Exact commands, warm-up and hardware logs: window_audit.json.
