# W1 isolated development experiments — PARTIAL SNAPSHOT

**Incomplete development results:** nuScenes 152/152 scenes, AV2 116/300 logs. Every comparison uses the same completed units. AV2 is an intermediate prefix, not the final result.

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
| av2 | cheap | 116 | 0.0530 | 32.0751 | 0.2711 | 0.0000 | 0.0258 |
| av2 | v6-S | 116 | 0.1894 | 8.5097 | 0.3201 | 0.3141 | 0.0542 |
| av2 | v6-S no refiner | 116 | 0.1288 | 13.3101 | 0.2580 | 0.2632 | 0.0542 |
| av2 | temporal evidence | 116 | 0.1439 | 8.9461 | 0.3060 | 0.3300 | 0.0553 |
| av2 | temporal evidence no refiner | 116 | 0.0833 | 13.7465 | 0.2439 | 0.2775 | 0.0553 |
| av2 | post-refinement closure | 116 | 0.1894 | 8.7279 | 0.3182 | 0.3161 | 0.0570 |
| av2 | draft contradiction recheck | 116 | 0.1894 | 8.4006 | 0.3202 | 0.3142 | 0.0549 |
| av2 | draft recheck no refiner | 116 | 0.1288 | 13.3101 | 0.2584 | 0.2631 | 0.0549 |
| av2 | hybrid at v6 crop budget | 116 | 0.1818 | 9.6007 | 0.3171 | 0.3181 | 0.0741 |
| av2 | unc2 at v6 crop budget | 116 | 0.1667 | 16.1466 | 0.3419 | 0.2390 | 0.0868 |
| av2 | u960 + C1 + v3a | 116 | 0.1364 | 11.4554 | 0.2838 | 0.3331 | 0.2323 |
| av2 | u960 + C1 no refiner | 116 | 0.0833 | 15.8193 | 0.2268 | 0.2730 | 0.2323 |
| av2 | u960 + C1 closure | 116 | 0.1364 | 11.2372 | 0.2862 | 0.3321 | 0.2323 |
| av2 | native + C1 + v3a | 116 | 0.1591 | 11.3463 | 0.3872 | 0.2425 | 1.0116 |
| av2 | native + C1 no refiner | 116 | 0.0909 | 16.1466 | 0.3172 | 0.1857 | 1.0116 |
| av2 | native + C1 closure | 116 | 0.1591 | 11.1281 | 0.3914 | 0.2391 | 1.0116 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen.