# Round 2 in-loop development experiments — dev

Development screen only; no method selected or promoted. Frozen sources and K34 are unchanged.
All intervals: paired unit bootstrap, 2,000, seed 0; pointwise, exploratory. No GPU wall time is a Jetson latency measurement.
G retains the original K30 cost convention (jerk relative to previous cheap action).
All five acquisition policies report no-refiner, v3a and fixed half-log controls. Logical requested pixels, not Jetson padded pixels.
Generic-component/equal-total-budget controls remain necessary before a selector-superiority claim.

| Dataset | Pipeline | Units | Miss rate | False/1000 | Insufficient loss/frame | G | Mean px/native |
|---|---|---:|---:|---:|---:|---:|---:|
| nusc | cheap | 152 | 0.3434 | 27.9549 | 0.5038 | 0.0000 | 0.0427 |
| nusc | v6-S | 152 | 0.4040 | 2.9426 | 0.3874 | 0.3212 | 0.0804 |
| nusc | v6-S / none | 152 | 0.3939 | 6.5392 | 0.3911 | 0.3130 | 0.0804 |
| nusc | v6-S / half | 152 | 0.3939 | 4.4139 | 0.3899 | 0.3183 | 0.0804 |
| nusc | round1 combined | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | round1 combined / none | 152 | 0.3737 | 7.8470 | 0.3725 | 0.3405 | 0.0750 |
| nusc | round1 combined / half | 152 | 0.3737 | 5.7218 | 0.3724 | 0.3461 | 0.0750 |
| nusc | proactive | 152 | 0.3939 | 3.7600 | 0.3710 | 0.3465 | 0.0752 |
| nusc | proactive / none | 152 | 0.3838 | 7.5200 | 0.3743 | 0.3382 | 0.0752 |
| nusc | proactive / half | 152 | 0.3838 | 5.3948 | 0.3740 | 0.3441 | 0.0752 |
| nusc | proactive + recovery | 152 | 0.3737 | 3.9235 | 0.3681 | 0.3507 | 0.0756 |
| nusc | proactive + recovery / none | 152 | 0.3636 | 7.8470 | 0.3721 | 0.3407 | 0.0756 |
| nusc | proactive + recovery / half | 152 | 0.3636 | 5.7218 | 0.3720 | 0.3463 | 0.0756 |
| nusc | looming + recovery | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | looming + recovery / none | 152 | 0.3737 | 7.8470 | 0.3725 | 0.3405 | 0.0750 |
| nusc | looming + recovery / half | 152 | 0.3737 | 5.7218 | 0.3724 | 0.3461 | 0.0750 |
| av2 | cheap | 300 | 0.0820 | 35.9603 | 0.3140 | 0.0000 | 0.0258 |
| av2 | v6-S | 300 | 0.1612 | 7.8597 | 0.3798 | 0.3087 | 0.0556 |
| av2 | v6-S / none | 300 | 0.1202 | 12.5079 | 0.3132 | 0.2760 | 0.0556 |
| av2 | v6-S / half | 300 | 0.1421 | 9.5922 | 0.3378 | 0.3203 | 0.0556 |
| av2 | round1 combined | 300 | 0.1339 | 9.0006 | 0.3685 | 0.3194 | 0.0514 |
| av2 | round1 combined / none | 300 | 0.0929 | 13.4376 | 0.3013 | 0.2859 | 0.0514 |
| av2 | round1 combined / half | 300 | 0.1175 | 10.5219 | 0.3263 | 0.3293 | 0.0514 |
| av2 | proactive | 300 | 0.1475 | 9.2119 | 0.3775 | 0.3090 | 0.0513 |
| av2 | proactive / none | 300 | 0.1066 | 13.6488 | 0.3117 | 0.2750 | 0.0513 |
| av2 | proactive / half | 300 | 0.1311 | 10.8177 | 0.3365 | 0.3179 | 0.0513 |
| av2 | proactive + recovery | 300 | 0.1311 | 9.4232 | 0.3664 | 0.3208 | 0.0522 |
| av2 | proactive + recovery / none | 300 | 0.0902 | 13.8179 | 0.3000 | 0.2865 | 0.0522 |
| av2 | proactive + recovery / half | 300 | 0.1148 | 10.9867 | 0.3251 | 0.3298 | 0.0522 |
| av2 | looming + recovery | 300 | 0.1284 | 9.0429 | 0.3683 | 0.3194 | 0.0514 |
| av2 | looming + recovery / none | 300 | 0.0902 | 13.5643 | 0.3011 | 0.2859 | 0.0514 |
| av2 | looming + recovery / half | 300 | 0.1148 | 10.6064 | 0.3260 | 0.3294 | 0.0514 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
