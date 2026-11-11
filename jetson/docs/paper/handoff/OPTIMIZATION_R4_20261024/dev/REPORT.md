# Round 4 sandbox development experiments — dev

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
| nusc | critical extent retention / none | 152 | 0.3737 | 7.8470 | 0.3633 | 0.3535 | 0.0778 |
| nusc | critical extent retention | 152 | 0.3838 | 3.9235 | 0.3588 | 0.3640 | 0.0778 |
| nusc | containment merge / none | 152 | 0.3737 | 7.8470 | 0.3725 | 0.3405 | 0.0750 |
| nusc | containment merge | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | union coverage / none | 152 | 0.3737 | 7.8470 | 0.3725 | 0.3405 | 0.0749 |
| nusc | union coverage | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0749 |
| nusc | semantic consistency / none | 152 | 0.3737 | 7.8470 | 0.3725 | 0.3405 | 0.0750 |
| nusc | semantic consistency | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | one-round extent / none | 152 | 0.3737 | 8.8279 | 0.4049 | 0.2768 | 0.0649 |
| nusc | one-round extent | 152 | 0.3838 | 5.7218 | 0.4007 | 0.2856 | 0.0649 |
| av2 | cheap | 300 | 0.0820 | 35.9603 | 0.3140 | 0.0000 | 0.0258 |
| av2 | v6-S / none | 300 | 0.1202 | 12.5079 | 0.3132 | 0.2760 | 0.0556 |
| av2 | v6-S | 300 | 0.1612 | 7.8597 | 0.3798 | 0.3087 | 0.0556 |
| av2 | round1 combined / none | 300 | 0.0929 | 13.4376 | 0.3013 | 0.2859 | 0.0514 |
| av2 | round1 combined | 300 | 0.1339 | 9.0006 | 0.3685 | 0.3194 | 0.0514 |
| av2 | round1 combined / half | 300 | 0.1175 | 10.5219 | 0.3263 | 0.3293 | 0.0514 |
| av2 | round1 combined / speed | 300 | 0.1393 | 8.6626 | 0.3781 | 0.3101 | 0.0514 |
| av2 | round1 combined / speedhalf | 300 | 0.1311 | 8.9584 | 0.3719 | 0.3169 | 0.0514 |
| av2 | critical extent retention / none | 300 | 0.0929 | 13.3953 | 0.2926 | 0.2898 | 0.0532 |
| av2 | critical extent retention | 300 | 0.1339 | 8.9584 | 0.3611 | 0.3210 | 0.0532 |
| av2 | containment merge / none | 300 | 0.0929 | 13.4376 | 0.3013 | 0.2859 | 0.0514 |
| av2 | containment merge | 300 | 0.1339 | 8.9161 | 0.3688 | 0.3191 | 0.0514 |
| av2 | union coverage / none | 300 | 0.0929 | 13.4376 | 0.3013 | 0.2860 | 0.0513 |
| av2 | union coverage | 300 | 0.1339 | 9.0006 | 0.3686 | 0.3194 | 0.0513 |
| av2 | semantic consistency / none | 300 | 0.0929 | 13.3953 | 0.3013 | 0.2859 | 0.0514 |
| av2 | semantic consistency | 300 | 0.1339 | 8.9584 | 0.3686 | 0.3194 | 0.0514 |
| av2 | one-round extent / none | 300 | 0.0902 | 14.2404 | 0.2908 | 0.2905 | 0.0429 |
| av2 | one-round extent | 300 | 0.1257 | 11.6205 | 0.3569 | 0.3207 | 0.0429 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
