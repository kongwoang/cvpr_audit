# W1 isolated development experiments — dev

Development screen only; no method selected or promoted. Frozen sources and K34 are unchanged.
All intervals: paired unit bootstrap, 2,000, seed 0; pointwise, exploratory. No GPU wall time is a Jetson latency measurement.
G retains the original K30 cost convention (jerk relative to previous cheap action).
Controls tsp8/unc2 use baseline v6-S crop-pixel allowance; total realized pixels can differ due to guards.
Generic-component/equal-total-budget controls remain necessary before a selector-superiority claim.

| Dataset | Pipeline | Units | Miss rate | False/1000 | Insufficient loss/frame | G | Mean px/native |
|---|---|---:|---:|---:|---:|---:|---:|
| nusc | cheap | 152 | 0.3434 | 27.9549 | 0.5038 | 0.0000 | 0.0427 |
| nusc | v6-S | 152 | 0.4040 | 2.9426 | 0.3874 | 0.3212 | 0.0804 |
| nusc | v6-S no refiner | 152 | 0.3939 | 6.5392 | 0.3911 | 0.3130 | 0.0804 |
| nusc | temporal evidence | 152 | 0.3838 | 3.2696 | 0.3848 | 0.3247 | 0.0809 |
| nusc | temporal evidence no refiner | 152 | 0.3737 | 7.0296 | 0.3894 | 0.3147 | 0.0809 |
| nusc | post-refinement closure | 152 | 0.4040 | 2.9426 | 0.3874 | 0.3212 | 0.0805 |
| nusc | draft contradiction recheck | 152 | 0.4040 | 3.1061 | 0.3642 | 0.3603 | 0.0814 |
| nusc | draft recheck no refiner | 152 | 0.3939 | 6.5392 | 0.3680 | 0.3523 | 0.0814 |
| nusc | hybrid at v6 crop budget | 152 | 0.4040 | 4.5774 | 0.4211 | 0.2551 | 0.1139 |
| nusc | unc2 at v6 crop budget | 152 | 0.4141 | 6.0487 | 0.6007 | -0.0472 | 0.1073 |
| nusc | u960 + C1 + v3a | 152 | 0.3535 | 6.3757 | 0.4054 | 0.2715 | 0.3840 |
| nusc | u960 + C1 no refiner | 152 | 0.3131 | 10.2992 | 0.4078 | 0.2631 | 0.3840 |
| nusc | u960 + C1 closure | 152 | 0.3535 | 6.3757 | 0.4054 | 0.2715 | 0.3840 |
| nusc | native + C1 + v3a | 152 | 0.4040 | 4.7409 | 0.3854 | 0.3231 | 1.0311 |
| nusc | native + C1 no refiner | 152 | 0.3535 | 10.4626 | 0.3832 | 0.3184 | 1.0311 |
| nusc | native + C1 closure | 152 | 0.4040 | 4.7409 | 0.3854 | 0.3231 | 1.0311 |
| av2 | cheap | 300 | 0.0820 | 35.9603 | 0.3140 | 0.0000 | 0.0258 |
| av2 | v6-S | 300 | 0.1612 | 7.8597 | 0.3798 | 0.3087 | 0.0556 |
| av2 | v6-S no refiner | 300 | 0.1202 | 12.5079 | 0.3132 | 0.2760 | 0.0556 |
| av2 | temporal evidence | 300 | 0.1339 | 8.4513 | 0.3684 | 0.3200 | 0.0566 |
| av2 | temporal evidence no refiner | 300 | 0.0929 | 13.0995 | 0.3010 | 0.2873 | 0.0566 |
| av2 | post-refinement closure | 300 | 0.1612 | 7.7329 | 0.3791 | 0.3097 | 0.0582 |
| av2 | draft contradiction recheck | 300 | 0.1612 | 7.9865 | 0.3786 | 0.3095 | 0.0566 |
| av2 | draft recheck no refiner | 300 | 0.1202 | 12.6347 | 0.3122 | 0.2766 | 0.0566 |
| av2 | hybrid at v6 crop budget | 300 | 0.1557 | 8.4936 | 0.3766 | 0.3114 | 0.0760 |
| av2 | unc2 at v6 crop budget | 300 | 0.1503 | 17.4097 | 0.3984 | 0.2392 | 0.0890 |
| av2 | u960 + C1 + v3a | 300 | 0.1120 | 10.9867 | 0.3393 | 0.3277 | 0.2323 |
| av2 | u960 + C1 no refiner | 300 | 0.0820 | 15.2969 | 0.2754 | 0.2847 | 0.2323 |
| av2 | u960 + C1 closure | 300 | 0.1120 | 10.8177 | 0.3421 | 0.3257 | 0.2323 |
| av2 | native + C1 + v3a | 300 | 0.1284 | 10.6909 | 0.4453 | 0.2368 | 1.0116 |
| av2 | native + C1 no refiner | 300 | 0.0847 | 15.5926 | 0.3744 | 0.1931 | 1.0116 |
| av2 | native + C1 closure | 300 | 0.1284 | 10.4374 | 0.4485 | 0.2347 | 1.0116 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen.

Source integrity: local frozen hashes match the pre-experiment manifest; every remote job checked main source and weight hashes before/after.

Raw records and exact commands: experiments/w1_dev_20261023 on VUW shared workspace; mirrored job metadata in the local experiment folder.
