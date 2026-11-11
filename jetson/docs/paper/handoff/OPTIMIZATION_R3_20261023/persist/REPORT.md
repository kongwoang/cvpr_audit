# Round 3 sandbox development experiments — persist

Development screen only; no method selected or promoted. Frozen sources and K34 are unchanged.
All intervals: paired unit bootstrap, 2,000, seed 0; pointwise, exploratory. No GPU wall time is a Jetson latency measurement.
G retains the original K30 cost convention (jerk relative to previous cheap action).
Every executed policy/refiner row is retained. Logical requested pixels, not Jetson padded pixels.
Generic-component/equal-total-budget controls remain necessary before a selector-superiority claim.

| Dataset | Pipeline | Units | Miss rate | False/1000 | Insufficient loss/frame | G | Mean px/native |
|---|---|---:|---:|---:|---:|---:|---:|
| nusc | cheap | 152 | 0.3434 | 27.9549 | 0.5038 | 0.0000 | 0.0427 |
| nusc | v6-S / none | 152 | 0.3939 | 6.5392 | 0.3911 | 0.3130 | 0.0804 |
| nusc | v6-S | 152 | 0.4040 | 2.9426 | 0.3874 | 0.3212 | 0.0804 |
| nusc | round1 combined / none | 152 | 0.3737 | 7.8470 | 0.3725 | 0.3405 | 0.0750 |
| nusc | round1 combined | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | round1 combined / half | 152 | 0.3737 | 5.7218 | 0.3724 | 0.3461 | 0.0750 |
| nusc | round1 combined / speed | 152 | 0.3838 | 4.7409 | 0.3679 | 0.3495 | 0.0750 |
| nusc | round1 combined / speedhalf | 152 | 0.3838 | 4.5774 | 0.3677 | 0.3504 | 0.0750 |
| nusc | upper-quantile missing history / none | 152 | 0.2626 | 19.6175 | 0.3244 | 0.3606 | 0.0750 |
| nusc | upper-quantile missing history | 152 | 0.2929 | 14.3861 | 0.3282 | 0.3687 | 0.0750 |
| nusc | upper quantile + extent retention / none | 152 | 0.2626 | 19.4540 | 0.3005 | 0.3895 | 0.0791 |
| nusc | upper quantile + extent retention | 152 | 0.2929 | 14.5496 | 0.3050 | 0.3955 | 0.0791 |
| av2 | cheap | 300 | 0.0820 | 35.9603 | 0.3140 | 0.0000 | 0.0258 |
| av2 | v6-S / none | 300 | 0.1202 | 12.5079 | 0.3132 | 0.2760 | 0.0556 |
| av2 | v6-S | 300 | 0.1612 | 7.8597 | 0.3798 | 0.3087 | 0.0556 |
| av2 | round1 combined / none | 300 | 0.0929 | 13.4376 | 0.3013 | 0.2859 | 0.0514 |
| av2 | round1 combined | 300 | 0.1339 | 9.0006 | 0.3685 | 0.3194 | 0.0514 |
| av2 | round1 combined / half | 300 | 0.1175 | 10.5219 | 0.3263 | 0.3293 | 0.0514 |
| av2 | round1 combined / speed | 300 | 0.1393 | 8.6626 | 0.3781 | 0.3101 | 0.0514 |
| av2 | round1 combined / speedhalf | 300 | 0.1311 | 8.9584 | 0.3719 | 0.3169 | 0.0514 |
| av2 | upper-quantile missing history / none | 300 | 0.0683 | 19.4380 | 0.2707 | 0.2968 | 0.0514 |
| av2 | upper-quantile missing history | 300 | 0.0984 | 15.6349 | 0.3313 | 0.3404 | 0.0514 |
| av2 | upper quantile + extent retention / none | 300 | 0.0656 | 19.2690 | 0.2584 | 0.3025 | 0.0536 |
| av2 | upper quantile + extent retention | 300 | 0.0956 | 15.4659 | 0.3198 | 0.3440 | 0.0536 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
