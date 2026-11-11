# Round 3 sandbox development experiments — dev

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
| nusc | mixed-history C1 / none | 152 | 0.3939 | 8.1739 | 0.3724 | 0.3407 | 0.0744 |
| nusc | mixed-history C1 | 152 | 0.4040 | 4.0870 | 0.3692 | 0.3494 | 0.0744 |
| nusc | mixed-history + recovery / none | 152 | 0.3737 | 8.3374 | 0.3706 | 0.3429 | 0.0750 |
| nusc | mixed-history + recovery | 152 | 0.3838 | 4.2504 | 0.3666 | 0.3531 | 0.0750 |
| nusc | cheap TTC transfer / none | 152 | 0.3535 | 11.6070 | 0.3663 | 0.3249 | 0.0750 |
| nusc | cheap TTC transfer | 152 | 0.3636 | 7.8470 | 0.3622 | 0.3354 | 0.0750 |
| nusc | critical extent retention / none | 152 | 0.3737 | 7.8470 | 0.3633 | 0.3535 | 0.0778 |
| nusc | critical extent retention | 152 | 0.3838 | 3.9235 | 0.3588 | 0.3640 | 0.0778 |
| nusc | motion-conditioned field / none | 152 | 0.3737 | 7.5200 | 0.3727 | 0.3378 | 0.0786 |
| nusc | motion-conditioned field | 152 | 0.3838 | 3.7600 | 0.3687 | 0.3478 | 0.0786 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
