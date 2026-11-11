# Jetson bounded-readout component timing

Exclusive MAXN, locked clocks, fan255, one thread; three repetitions of every original development frame, with 200-frame warm-up per dataset and alternating row order. Paired unit bootstrap 2,000 seed0. See all intervals and per-repetition rows.

Times cover only the extra current-observation matching, C1 and capped-action function. Detector inference, cross-crop proposal suppression, tensor copies, buffering, R5 classifier and image decode are excluded. These are not total method latency, incremental full-pipeline latency, FPS or TRT parity. `active_frames.csv` is conditional timing, not an unbiased population average. `parity.json` checks CPU readout actions on stored observations against VUW.

| Dataset | Prior observations | Mean ms [95% CI] | p95 ms [95% CI] | p99 ms [95% CI] |
|---|---:|---|---|---|
| nusc | 1 | 0.681 [0.577, 0.794] | 2.776 [2.641, 2.917] | 3.265 [3.095, 3.576] |
| nusc | 2 | 0.702 [0.593, 0.821] | 2.955 [2.820, 3.101] | 3.492 [3.273, 3.808] |
| av2 | 1 | 1.201 [1.127, 1.273] | 3.193 [3.117, 3.256] | 3.649 [3.562, 3.733] |
| av2 | 2 | 1.253 [1.176, 1.329] | 3.438 [3.360, 3.506] | 3.900 [3.813, 3.982] |
