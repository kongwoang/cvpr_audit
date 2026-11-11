# Round 6 sandbox development experiments — followup

Development screen only; no method selected or promoted. Frozen sources and K34 are unchanged.
All intervals: paired unit bootstrap, 2,000, seed 0; pointwise, exploratory. No GPU wall time is a Jetson latency measurement.
G retains the original K30 cost convention (jerk relative to previous cheap action).
Every executed policy/refiner row is retained. Logical requested pixels, not Jetson padded pixels.
Generic-component/equal-total-budget controls remain necessary before a selector-superiority claim.

| Dataset | Pipeline | Units | Miss rate | False/1000 | Insufficient loss/frame | G | Mean px/native |
|---|---|---:|---:|---:|---:|---:|---:|
| nusc | cheap | 152 | 0.3434 | 27.9549 | 0.5038 | 0.0000 | 0.0427 |
| nusc | v6-S | 152 | 0.4040 | 2.9426 | 0.3874 | 0.3212 | 0.0804 |
| nusc | round1 combined | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | point fallback + extent retention | 152 | 0.3535 | 5.2313 | 0.3370 | 0.3826 | 0.0784 |
| nusc | R5 point_residual q50 | 152 | 0.3434 | 5.5583 | 0.3171 | 0.4150 | 0.0784 |
| nusc | R5 point_residual q80 | 152 | 0.3535 | 5.2313 | 0.3266 | 0.4005 | 0.0784 |
| nusc | R5 point_residual q95 | 152 | 0.3535 | 5.2313 | 0.3370 | 0.3826 | 0.0784 |
| nusc | R6 res_ungated | 152 | 0.3131 | 12.0974 | 0.3069 | 0.3900 | 0.0784 |
| nusc | R6 res_seen1 | 152 | 0.3131 | 7.1931 | 0.3229 | 0.3931 | 0.0784 |
| nusc | R6 res_seen2 | 152 | 0.3434 | 6.3757 | 0.3359 | 0.3791 | 0.0784 |
| nusc | R6 res_agree1 | 152 | 0.3535 | 6.0487 | 0.3263 | 0.3960 | 0.0784 |
| nusc | R6 res_agree2 | 152 | 0.3535 | 5.3948 | 0.3370 | 0.3825 | 0.0784 |
| nusc | R6 res_low | 152 | 0.3434 | 6.2122 | 0.3117 | 0.4179 | 0.0784 |
| nusc | R6 res_lowgeom | 152 | 0.3535 | 5.3948 | 0.3200 | 0.4097 | 0.0784 |
| nusc | R6 res_low_or_seen1 | 152 | 0.3131 | 7.5200 | 0.3100 | 0.4103 | 0.0784 |
| nusc | R6 res_low_and_seen1 | 152 | 0.3434 | 5.7218 | 0.3246 | 0.4018 | 0.0784 |
| nusc | R6 res_low_or_seen2 | 152 | 0.3333 | 7.0296 | 0.3106 | 0.4169 | 0.0784 |
| nusc | R6 res_low_and_seen2 | 152 | 0.3535 | 5.5583 | 0.3370 | 0.3812 | 0.0784 |
| nusc | R6 res_proposal_any | 152 | 0.3333 | 6.2122 | 0.3112 | 0.4166 | 0.0784 |
| nusc | R6 track_observed1 | 152 | 0.3535 | 6.8661 | 0.3130 | 0.4156 | 0.0784 |
| nusc | R6 track_observed2 | 152 | 0.3535 | 6.7026 | 0.3201 | 0.4060 | 0.0784 |
| nusc | R6 track_agree1 | 152 | 0.3535 | 5.2313 | 0.3136 | 0.4197 | 0.0784 |
| nusc | R6 track_agree2 | 152 | 0.3535 | 5.2313 | 0.3207 | 0.4099 | 0.0784 |
| nusc | R6 combo1 | 152 | 0.3131 | 9.1548 | 0.2963 | 0.4264 | 0.0784 |
| nusc | R6 combo1_r5 | 152 | 0.3131 | 9.3183 | 0.2940 | 0.4298 | 0.0784 |
| nusc | R6 combo2 | 152 | 0.3131 | 8.9913 | 0.3035 | 0.4157 | 0.0784 |
| nusc | R6 combo2_r5 | 152 | 0.3131 | 9.1548 | 0.3012 | 0.4191 | 0.0784 |
| nusc | R6 track_agree1_r5 | 152 | 0.3434 | 5.5583 | 0.3041 | 0.4341 | 0.0784 |
| nusc | R6 track_agree2_r5 | 152 | 0.3434 | 5.5583 | 0.3111 | 0.4243 | 0.0784 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. All original development units are included; adaptive exploration after five previous rounds, not confirmation.
