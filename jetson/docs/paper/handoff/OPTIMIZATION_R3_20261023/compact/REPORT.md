# Round 3 sandbox development experiments — compact

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
| nusc | learned missing history / none | 152 | 0.3232 | 9.3183 | 0.3515 | 0.3602 | 0.0750 |
| nusc | learned missing history | 152 | 0.3535 | 5.3948 | 0.3521 | 0.3648 | 0.0750 |
| nusc | compact point fallback / none | 152 | 0.3333 | 8.9913 | 0.3636 | 0.3481 | 0.0750 |
| nusc | compact point fallback | 152 | 0.3333 | 5.2313 | 0.3592 | 0.3591 | 0.0750 |
| nusc | compact observed-history fallback / none | 152 | 0.3636 | 8.1739 | 0.3715 | 0.3388 | 0.0750 |
| nusc | compact observed-history fallback | 152 | 0.3737 | 4.2504 | 0.3676 | 0.3494 | 0.0750 |
| av2 | cheap | 300 | 0.0820 | 35.9603 | 0.3140 | 0.0000 | 0.0258 |
| av2 | v6-S / none | 300 | 0.1202 | 12.5079 | 0.3132 | 0.2760 | 0.0556 |
| av2 | v6-S | 300 | 0.1612 | 7.8597 | 0.3798 | 0.3087 | 0.0556 |
| av2 | round1 combined / none | 300 | 0.0929 | 13.4376 | 0.3013 | 0.2859 | 0.0514 |
| av2 | round1 combined | 300 | 0.1339 | 9.0006 | 0.3685 | 0.3194 | 0.0514 |
| av2 | round1 combined / half | 300 | 0.1175 | 10.5219 | 0.3263 | 0.3293 | 0.0514 |
| av2 | round1 combined / speed | 300 | 0.1393 | 8.6626 | 0.3781 | 0.3101 | 0.0514 |
| av2 | round1 combined / speedhalf | 300 | 0.1311 | 8.9584 | 0.3719 | 0.3169 | 0.0514 |
| av2 | learned missing history / none | 300 | 0.0820 | 14.1137 | 0.2817 | 0.3038 | 0.0514 |
| av2 | learned missing history | 300 | 0.1202 | 9.9303 | 0.3460 | 0.3427 | 0.0514 |
| av2 | compact point fallback / none | 300 | 0.0929 | 14.1137 | 0.2964 | 0.2877 | 0.0514 |
| av2 | compact point fallback | 300 | 0.1311 | 9.6345 | 0.3617 | 0.3248 | 0.0514 |
| av2 | compact observed-history fallback / none | 300 | 0.0929 | 14.0292 | 0.2967 | 0.2887 | 0.0514 |
| av2 | compact observed-history fallback | 300 | 0.1339 | 9.5077 | 0.3622 | 0.3253 | 0.0514 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
