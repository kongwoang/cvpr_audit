# Round 3 sandbox development experiments — smoke

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
| nusc | mixed-history C1 / none | 4 | 1.0000 | 31.2500 | 0.9739 | 0.0720 | 0.0862 |
| nusc | mixed-history C1 | 4 | 1.0000 | 18.7500 | 0.9819 | 0.0719 | 0.0862 |
| nusc | mixed-history + recovery / none | 4 | 0.7500 | 31.2500 | 0.9672 | 0.0761 | 0.0871 |
| nusc | mixed-history + recovery | 4 | 0.7500 | 18.7500 | 0.9752 | 0.0761 | 0.0871 |
| nusc | cheap TTC transfer / none | 4 | 0.7500 | 37.5000 | 0.9625 | 0.0459 | 0.0870 |
| nusc | cheap TTC transfer | 4 | 0.7500 | 25.0000 | 0.9706 | 0.0459 | 0.0870 |
| nusc | critical extent retention / none | 4 | 0.7500 | 25.0000 | 0.9674 | 0.0797 | 0.0886 |
| nusc | critical extent retention | 4 | 0.7500 | 12.5000 | 0.9755 | 0.0796 | 0.0886 |
| nusc | motion-conditioned field / none | 4 | 0.7500 | 18.7500 | 0.9737 | 0.0784 | 0.0970 |
| nusc | motion-conditioned field | 4 | 0.7500 | 6.2500 | 0.9818 | 0.0783 | 0.0970 |
| av2 | cheap | 5 | 0.0833 | 38.4615 | 0.7678 | 0.0000 | 0.0258 |
| av2 | v6-S / none | 5 | 0.1667 | 10.2564 | 1.0720 | -0.0674 | 0.0497 |
| av2 | v6-S | 5 | 0.5000 | 5.1282 | 1.3214 | -0.1850 | 0.0497 |
| av2 | round1 combined / none | 5 | 0.1667 | 10.2564 | 1.0325 | -0.0431 | 0.0470 |
| av2 | round1 combined | 5 | 0.5000 | 5.1282 | 1.2826 | -0.1515 | 0.0470 |
| av2 | round1 combined / half | 5 | 0.1667 | 15.3846 | 1.0910 | -0.0516 | 0.0470 |
| av2 | round1 combined / speed | 5 | 0.5000 | 5.1282 | 1.2838 | -0.1603 | 0.0470 |
| av2 | round1 combined / speedhalf | 5 | 0.4167 | 5.1282 | 1.2778 | -0.1555 | 0.0470 |
| av2 | mixed-history C1 / none | 5 | 0.1667 | 15.3846 | 1.0156 | -0.0334 | 0.0463 |
| av2 | mixed-history C1 | 5 | 0.4167 | 5.1282 | 1.2782 | -0.1578 | 0.0463 |
| av2 | mixed-history + recovery / none | 5 | 0.1667 | 12.8205 | 1.0222 | -0.0425 | 0.0470 |
| av2 | mixed-history + recovery | 5 | 0.4167 | 5.1282 | 1.2740 | -0.1514 | 0.0470 |
| av2 | cheap TTC transfer / none | 5 | 0.0833 | 12.8205 | 1.0117 | -0.0455 | 0.0471 |
| av2 | cheap TTC transfer | 5 | 0.3333 | 7.6923 | 1.2618 | -0.1540 | 0.0471 |
| av2 | critical extent retention / none | 5 | 0.1667 | 10.2564 | 1.0325 | -0.0431 | 0.0480 |
| av2 | critical extent retention | 5 | 0.5000 | 5.1282 | 1.2826 | -0.1515 | 0.0480 |
| av2 | motion-conditioned field / none | 5 | 0.1667 | 12.8205 | 1.0570 | -0.0668 | 0.0576 |
| av2 | motion-conditioned field | 5 | 0.5000 | 10.2564 | 1.3049 | -0.1740 | 0.0576 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
