# Round 6 sandbox development experiments — followup

Development screen only; no method selected or promoted. Frozen sources and K34 are unchanged.
All intervals: paired unit bootstrap, 2,000, seed 0; pointwise, exploratory. No GPU wall time is a Jetson latency measurement.
G retains the original K30 cost convention (jerk relative to previous cheap action).
Every executed policy/refiner row is retained. Logical requested pixels, not Jetson padded pixels.
Generic-component/equal-total-budget controls remain necessary before a selector-superiority claim.

| Dataset | Pipeline | Units | Miss rate | False/1000 | Insufficient loss/frame | G | Mean px/native |
|---|---|---:|---:|---:|---:|---:|---:|
| av2 | cheap | 300 | 0.0820 | 35.9603 | 0.3140 | 0.0000 | 0.0258 |
| av2 | v6-S | 300 | 0.1612 | 7.8597 | 0.3798 | 0.3087 | 0.0556 |
| av2 | round1 combined | 300 | 0.1339 | 9.0006 | 0.3685 | 0.3194 | 0.0514 |
| av2 | point fallback + extent retention | 300 | 0.1175 | 9.8035 | 0.3349 | 0.3483 | 0.0533 |
| av2 | R5 point_residual q50 | 300 | 0.1175 | 10.0148 | 0.3333 | 0.3496 | 0.0533 |
| av2 | R5 point_residual q80 | 300 | 0.1175 | 9.8458 | 0.3348 | 0.3483 | 0.0533 |
| av2 | R5 point_residual q95 | 300 | 0.1175 | 9.8035 | 0.3349 | 0.3483 | 0.0533 |
| av2 | R6 res_ungated | 300 | 0.1038 | 16.8603 | 0.3220 | 0.3183 | 0.0533 |
| av2 | R6 res_seen1 | 300 | 0.1038 | 15.2969 | 0.3253 | 0.3238 | 0.0533 |
| av2 | R6 res_seen2 | 300 | 0.1066 | 13.0150 | 0.3278 | 0.3351 | 0.0533 |
| av2 | R6 res_agree1 | 300 | 0.1148 | 11.4938 | 0.3309 | 0.3453 | 0.0533 |
| av2 | R6 res_agree2 | 300 | 0.1175 | 10.3528 | 0.3328 | 0.3475 | 0.0533 |
| av2 | R6 res_low | 300 | 0.1093 | 11.1135 | 0.3323 | 0.3465 | 0.0533 |
| av2 | R6 res_lowgeom | 300 | 0.1175 | 10.1416 | 0.3331 | 0.3494 | 0.0533 |
| av2 | R6 res_low_or_seen1 | 300 | 0.1038 | 15.3814 | 0.3245 | 0.3241 | 0.0533 |
| av2 | R6 res_low_and_seen1 | 300 | 0.1093 | 10.9867 | 0.3330 | 0.3462 | 0.0533 |
| av2 | R6 res_low_or_seen2 | 300 | 0.1038 | 13.4798 | 0.3263 | 0.3344 | 0.0533 |
| av2 | R6 res_low_and_seen2 | 300 | 0.1120 | 10.6064 | 0.3338 | 0.3472 | 0.0533 |
| av2 | R6 res_proposal_any | 300 | 0.1093 | 11.9163 | 0.3314 | 0.3428 | 0.0533 |
| av2 | R6 track_observed1 | 300 | 0.1175 | 13.8179 | 0.3311 | 0.3284 | 0.0533 |
| av2 | R6 track_observed2 | 300 | 0.1175 | 13.3531 | 0.3319 | 0.3292 | 0.0533 |
| av2 | R6 track_agree1 | 300 | 0.1175 | 9.7190 | 0.3340 | 0.3491 | 0.0533 |
| av2 | R6 track_agree2 | 300 | 0.1175 | 9.7613 | 0.3342 | 0.3490 | 0.0533 |
| av2 | R6 combo1 | 300 | 0.1038 | 18.7196 | 0.3212 | 0.3044 | 0.0533 |
| av2 | R6 combo1_r5 | 300 | 0.1038 | 18.8041 | 0.3210 | 0.3043 | 0.0533 |
| av2 | R6 combo2 | 300 | 0.1038 | 18.4238 | 0.3218 | 0.3051 | 0.0533 |
| av2 | R6 combo2_r5 | 300 | 0.1038 | 18.5083 | 0.3216 | 0.3051 | 0.0533 |
| av2 | R6 track_agree1_r5 | 300 | 0.1175 | 9.9303 | 0.3324 | 0.3504 | 0.0533 |
| av2 | R6 track_agree2_r5 | 300 | 0.1175 | 9.9725 | 0.3326 | 0.3504 | 0.0533 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. All original development units are included; adaptive exploration after five previous rounds, not confirmation.
