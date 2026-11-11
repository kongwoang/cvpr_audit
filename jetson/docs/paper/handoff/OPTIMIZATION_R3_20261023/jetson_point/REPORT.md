# R3 masked-history point fallback — Jetson

Complete: three repetitions of 512 timed development frames per dataset, after 200 warm-up frames; MAXN, locked clocks, fan255, exclusive window.
Existing trt32 engines; original C1 encoder reused, new MLP FP32. No TensorRT engine built. Pipelines rotate order between repetitions.
Paired unit bootstrap 2,000 seed0, repetitions nested in scenes/logs; pooled latency percentiles. Limited fixed development prefixes, not test evaluation.
All three pipelines and no-refiner quality diagnostics are retained. Hardware state restored; main-source and engine hashes unchanged.

| Dataset | Pipeline | Mean ms (95% CI) | p95 ms | p99 ms | >50 ms % | >66.7 ms % | >100 ms % |
|---|---|---:|---:|---:|---:|---:|---:|
| av2_dev | learned missing history | 43.24 [24.44, 64.96] | 110.81 | 147.59 | 34.18 | 23.57 | 8.07 |
| av2_dev | round1 combined | 35.02 [20.71, 51.06] | 85.20 | 109.37 | 26.50 | 12.89 | 1.76 |
| av2_dev | v6-S | 38.02 [20.37, 58.07] | 114.07 | 142.97 | 24.15 | 16.99 | 7.62 |
| nusc_val | learned missing history | 27.57 [21.32, 34.31] | 62.19 | 77.45 | 14.32 | 3.39 | 0.00 |
| nusc_val | round1 combined | 23.77 [18.73, 29.00] | 49.73 | 59.78 | 4.95 | 0.33 | 0.00 |
| nusc_val | v6-S | 25.53 [19.27, 32.96] | 61.64 | 79.80 | 10.16 | 2.99 | 0.20 |

All intervals/paired differences: intervals.csv and contrasts.csv. Actual padded detector pixels include historical detections; image-encoder work is separately counted.
Quality on a small timing prefix is diagnostic only; use complete VUW development results for quality. Prefix decisions may differ from VUW because of detection/backend/canvas differences; cross-backend comparison is reported separately.
Baseline-port and Torch/TRT image-encoder smoke parity passed on 80 frames per dataset. Repetition parity is recorded, not assumed.
Lazy model loading occurs in warm-up in all six runs (cold_load_audit.csv). Steady-state latency excludes startup and does not establish a real-time guarantee.
The extra fallback has a measurable latency cost despite near-constant detector pixels. No production/core promotion is justified by detector pixels alone.

Reproduce: scripts/py experiments/optimization_r3_20261023/report_jetson.py. Raw runs, exact arguments and hardware logs are linked in window_audit.json.
