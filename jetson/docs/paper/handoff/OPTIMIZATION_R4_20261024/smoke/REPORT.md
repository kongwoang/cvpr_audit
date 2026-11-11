# Round 4 sandbox development experiments — smoke

Development screen only; no method selected or promoted. Frozen sources and K34 are unchanged.
All intervals: paired unit bootstrap, 2,000, seed 0; pointwise, exploratory. No GPU wall time is a Jetson latency measurement.
G retains the original K30 cost convention (jerk relative to previous cheap action).
Every executed policy/refiner row is retained. Logical requested pixels, not Jetson padded pixels.
Generic-component/equal-total-budget controls remain necessary before a selector-superiority claim.

| Dataset | Pipeline | Units | Miss rate | False/1000 | Insufficient loss/frame | G | Mean px/native |
|---|---|---:|---:|---:|---:|---:|---:|
| nusc | cheap | 4 | 0.2500 | 75.0000 | 0.8889 | 0.0000 | 0.0427 |
| nusc | v6-S / none | 4 | 1.0000 | 12.5000 | 0.9829 | 0.0747 | 0.1000 |
| nusc | v6-S | 4 | 1.0000 | 6.2500 | 0.9885 | 0.0742 | 0.1000 |
| nusc | round1 combined / none | 4 | 0.7500 | 25.0000 | 0.9674 | 0.0797 | 0.0871 |
| nusc | round1 combined | 4 | 0.7500 | 12.5000 | 0.9755 | 0.0796 | 0.0871 |
| nusc | round1 combined / half | 4 | 0.7500 | 25.0000 | 0.9702 | 0.0797 | 0.0871 |
| nusc | round1 combined / speed | 4 | 0.7500 | 18.7500 | 0.9730 | 0.0792 | 0.0871 |
| nusc | round1 combined / speedhalf | 4 | 0.7500 | 18.7500 | 0.9730 | 0.0792 | 0.0871 |
| nusc | critical extent retention / none | 4 | 0.7500 | 25.0000 | 0.9674 | 0.0797 | 0.0886 |
| nusc | critical extent retention | 4 | 0.7500 | 12.5000 | 0.9755 | 0.0796 | 0.0886 |
| nusc | containment merge / none | 4 | 0.7500 | 25.0000 | 0.9674 | 0.0797 | 0.0871 |
| nusc | containment merge | 4 | 0.7500 | 12.5000 | 0.9755 | 0.0796 | 0.0871 |
| nusc | union coverage / none | 4 | 0.7500 | 25.0000 | 0.9674 | 0.0797 | 0.0871 |
| nusc | union coverage | 4 | 0.7500 | 12.5000 | 0.9755 | 0.0796 | 0.0871 |
| nusc | semantic consistency / none | 4 | 0.7500 | 25.0000 | 0.9674 | 0.0797 | 0.0871 |
| nusc | semantic consistency | 4 | 0.7500 | 12.5000 | 0.9755 | 0.0796 | 0.0871 |
| nusc | one-round extent / none | 4 | 0.7500 | 18.7500 | 0.9675 | 0.0796 | 0.0664 |
| nusc | one-round extent | 4 | 0.7500 | 12.5000 | 0.9731 | 0.0791 | 0.0664 |
| av2 | cheap | 5 | 0.0833 | 38.4615 | 0.7678 | 0.0000 | 0.0258 |
| av2 | v6-S / none | 5 | 0.1667 | 10.2564 | 1.0720 | -0.0674 | 0.0497 |
| av2 | v6-S | 5 | 0.5000 | 5.1282 | 1.3214 | -0.1850 | 0.0497 |
| av2 | round1 combined / none | 5 | 0.1667 | 10.2564 | 1.0325 | -0.0431 | 0.0470 |
| av2 | round1 combined | 5 | 0.5000 | 5.1282 | 1.2826 | -0.1515 | 0.0470 |
| av2 | round1 combined / half | 5 | 0.1667 | 15.3846 | 1.0910 | -0.0516 | 0.0470 |
| av2 | round1 combined / speed | 5 | 0.5000 | 5.1282 | 1.2838 | -0.1603 | 0.0470 |
| av2 | round1 combined / speedhalf | 5 | 0.4167 | 5.1282 | 1.2778 | -0.1555 | 0.0470 |
| av2 | critical extent retention / none | 5 | 0.1667 | 10.2564 | 1.0325 | -0.0431 | 0.0480 |
| av2 | critical extent retention | 5 | 0.5000 | 5.1282 | 1.2826 | -0.1515 | 0.0480 |
| av2 | containment merge / none | 5 | 0.1667 | 10.2564 | 1.0325 | -0.0431 | 0.0471 |
| av2 | containment merge | 5 | 0.5000 | 5.1282 | 1.2826 | -0.1515 | 0.0471 |
| av2 | union coverage / none | 5 | 0.1667 | 10.2564 | 1.0325 | -0.0431 | 0.0469 |
| av2 | union coverage | 5 | 0.5000 | 5.1282 | 1.2826 | -0.1515 | 0.0469 |
| av2 | semantic consistency / none | 5 | 0.1667 | 10.2564 | 1.0325 | -0.0431 | 0.0470 |
| av2 | semantic consistency | 5 | 0.5000 | 5.1282 | 1.2826 | -0.1515 | 0.0470 |
| av2 | one-round extent / none | 5 | 0.0833 | 7.6923 | 0.9338 | 0.0363 | 0.0389 |
| av2 | one-round extent | 5 | 0.4167 | 20.5128 | 1.1687 | -0.0656 | 0.0389 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
