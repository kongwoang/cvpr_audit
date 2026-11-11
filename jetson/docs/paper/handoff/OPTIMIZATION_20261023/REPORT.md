# Optimization development phase — completed handoff

Separate experiments only; frozen method/weights/K34 unchanged. All executed rows retained.
Final synthesis: [TONG_HOP.md](TONG_HOP.md). Completion audit: [completion_audit.json](completion_audit.json).
Detailed review: ../W1_DEV_20261023/OPTIMIZATION_REVIEW.md

| Work | Status | Progress |
|---|---|---|
| w1_dev_20261023 | complete | {'nusc': 152, 'av2': 300} |
| w1_bounded_20261023 | complete | {'nusc': 152, 'av2': 300} |
| runtime_20261023 | timing_complete | 16 completed processes |
| temporal_jetson_20261023 | timing_complete | 10 completed processes |
| temporal_fast_20261023 | complete | 3 paired repetitions per dataset; exact helper parity |
| verified_jetson_20261023 | complete | AV2 verified CFG-DEV; 512 timed frames × 3 repetitions × 4 policies |
| dataset/train audit | complete | no full-dev overlap with available training/selection unit IDs |
| backend parity audit | complete, differences retained | 1–3/512 timed frames per policy; exact parity not achieved |

NAVSIM initial CPU pass: exact scores/actions/field/support on 1,000 tokens; 7x7 mean 116→13 ms,
3x3 28→7.7 ms. Independent three-repetition/two-host replication is in ../NAVSIM_REUSE_20261023/.
Runtime buffer results: ../RUNTIME_20261023/. Temporal Jetson results: ../TEMPORAL_JETSON_20261023/.
Full dev factorial: ../W1_BOUNDED_20261023/. Original screen: ../W1_DEV_20261023/.
Refiner geometry attribution: ../REFINER_GEOMETRY_20261023/.
Budget controls include historical pixels but guards may overrun; no exact equal-cost superiority claim.
No method is promoted, and existing test data are not an untouched confirmatory set.

The legacy K29 AV2 cache mixes train/selection and CFG-DEV logs. Its runtime rows
are retained for engineering only. Use [verified AV2 timing](../VERIFIED_JETSON_20261023/REPORT.md)
and [verified_av2_latency.pdf](verified_av2_latency.pdf) for the corrected manifest.
Combined B2K-A + recovery reduces p99 143.41→109.30 ms, paired delta −34.11
[−46.03, −5.78] ms. Mean/p95 differences include zero. Actual padded pixels
increase 23.4% on this Jetson prefix, despite lower full-dev VUW pixels.
AV2 insufficient-braking loss remains above cheap by +0.0546 [0.0235, 0.0844]
per frame; W1 is not resolved. Full rows and intervals remain in the linked CSVs.
