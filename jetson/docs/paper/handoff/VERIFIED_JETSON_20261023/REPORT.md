# AV2 verified CFG-DEV — Jetson timing

All four prespecified rows retained; existing reference crop runtime/trt32 engines.
Verified AV2 CFG-DEV only, 512 timed frames/rep after 200 warm-up frames; three repetitions.
Original/fast helper parity passed on the first 80 verified frames. Actual training/selection unit IDs were excluded.
Paired unit bootstrap 2000 seed0; repetitions nested in units; pooled quantiles.

| Dataset | Pipeline | Mean ms | p95 ms | p99 ms | >50 ms % | >66.7 ms % | >100 ms % |
|---|---|---:|---:|---:|---:|---:|---:|
| av2_dev | B2K-A | 33.697 | 78.934 | 103.482 | 23.438 | 11.328 | 1.432 |
| av2_dev | B2K-A + recovery | 34.597 | 84.258 | 109.295 | 25.586 | 12.695 | 1.562 |
| av2_dev | v6-S | 37.802 | 114.210 | 143.406 | 24.154 | 16.667 | 7.292 |
| av2_dev | v6-S + recovery | 38.787 | 116.932 | 152.043 | 25.000 | 17.122 | 7.878 |

Full quality evidence comes from the VUW development factorial. Prefix G/loss here are diagnostic
and use the original cheap-previous-action jerk convention; no independent closed-loop rollout.
Historical execution and pixels are included. All four rows use the validated exact fast bookkeeping helper. Retaining two prior uploaded frames per recovery pipeline
uses extra GPU memory; memory.csv records retained tensors and process peaks. Comparators do not share histories.
Triggered latency is reported separately in triggered_latency.csv; an inactive prefix cannot characterize recovery tails.
Backend/canvas differences from VUW remain possible. This is not a cross-backend parity claim.
All main source/engine hashes were checked and the exclusive window released. No method is promoted.
