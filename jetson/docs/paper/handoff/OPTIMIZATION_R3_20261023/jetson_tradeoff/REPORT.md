# R3 tradeoff feasibility — Jetson

Complete: three repetitions of 512 timed development frames per dataset, after 200 warm-up frames; MAXN, locked clocks, fan255, exclusive window.
Existing trt32 engines; original C1 encoder reused, new MLP FP32. No TensorRT engine built. Pipelines rotate order between repetitions.
Paired unit bootstrap 2,000 seed0, repetitions nested in scenes/logs; pooled latency percentiles. Limited fixed development prefixes, not test evaluation.
All executed pipelines and no-refiner quality diagnostics are retained. Hardware state restored; main-source and engine hashes unchanged.

| Dataset | Pipeline | Mean ms (95% CI) | p95 ms | p99 ms | >50 ms % | >66.7 ms % | >100 ms % |
|---|---|---:|---:|---:|---:|---:|---:|
| av2_dev | compact point fallback | 38.46 [22.31, 56.81] | 97.17 | 124.89 | 30.86 | 17.19 | 4.10 |
| av2_dev | corroborated upper fallback | 43.64 [24.55, 65.87] | 111.83 | 148.02 | 34.18 | 24.22 | 8.98 |
| av2_dev | critical extent retention | 35.92 [20.67, 53.70] | 85.89 | 123.57 | 27.02 | 13.87 | 2.08 |
| av2_dev | learned missing history | 42.99 [24.32, 64.69] | 110.83 | 145.62 | 33.85 | 23.31 | 8.01 |
| av2_dev | round1 combined | 34.74 [20.55, 50.64] | 83.65 | 108.97 | 26.17 | 12.57 | 1.69 |
| av2_dev | v6-S | 37.71 [20.22, 57.68] | 114.05 | 144.02 | 23.89 | 16.80 | 7.29 |
| nusc_val | compact point fallback | 25.57 [19.94, 31.51] | 53.13 | 65.86 | 9.38 | 0.85 | 0.00 |
| nusc_val | corroborated upper fallback | 27.94 [21.53, 34.85] | 63.90 | 78.78 | 14.97 | 3.58 | 0.00 |
| nusc_val | critical extent retention | 24.60 [19.13, 30.33] | 51.41 | 63.12 | 6.32 | 0.85 | 0.00 |
| nusc_val | learned missing history | 27.56 [21.34, 34.23] | 62.14 | 75.81 | 14.45 | 3.06 | 0.00 |
| nusc_val | round1 combined | 23.71 [18.72, 28.91] | 49.27 | 59.02 | 4.43 | 0.26 | 0.00 |
| nusc_val | v6-S | 25.48 [19.21, 32.92] | 61.70 | 80.39 | 10.03 | 2.67 | 0.20 |

All intervals/paired differences: intervals.csv and contrasts.csv. Actual padded detector pixels include historical detections; image-encoder work is separately counted.
Quality on a small timing prefix is diagnostic only; use complete VUW development results for quality. Prefix decisions may differ from VUW because of detection/backend/canvas differences; cross-backend comparison is reported separately.
Baseline-port and Torch/TRT image-encoder smoke parity passed on 80 frames per dataset. Repetition parity is recorded, not assumed.
Lazy model loading occurs in warm-up in all six runs (cold_load_audit.csv). Steady-state latency excludes startup and does not establish a real-time guarantee.
The image fallback has a measurable latency cost despite near-constant detector pixels. The compact control uses a CPU FP32 box-only MLP. No production/core promotion is justified by detector pixels alone.
Process memory includes independent histories/models for all benchmark rows, not a per-pipeline deployment-memory estimate.

Reproduce: scripts/py experiments/optimization_r3_20261023/report_jetson.py --variant tradeoff. Raw runs, exact arguments and hardware logs are linked in window_audit.json.
