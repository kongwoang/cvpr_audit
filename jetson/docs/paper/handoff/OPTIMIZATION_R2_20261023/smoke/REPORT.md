# Round 2 in-loop development experiments — smoke

Development screen only; no method selected or promoted. Frozen sources and K34 are unchanged.
All intervals: paired unit bootstrap, 2,000, seed 0; pointwise, exploratory. No GPU wall time is a Jetson latency measurement.
G retains the original K30 cost convention (jerk relative to previous cheap action).
All five acquisition policies report no-refiner, v3a and fixed half-log controls. Logical requested pixels, not Jetson padded pixels.
Generic-component/equal-total-budget controls remain necessary before a selector-superiority claim.

| Dataset | Pipeline | Units | Miss rate | False/1000 | Insufficient loss/frame | G | Mean px/native |
|---|---|---:|---:|---:|---:|---:|---:|
| nusc | cheap | 4 | 0.2500 | 75.0000 | 0.8889 | 0.0000 | 0.0427 |
| nusc | v6-S | 4 | 1.0000 | 6.2500 | 0.9885 | 0.0742 | 0.1000 |
| nusc | v6-S / none | 4 | 1.0000 | 12.5000 | 0.9829 | 0.0747 | 0.1000 |
| nusc | v6-S / half | 4 | 1.0000 | 12.5000 | 0.9857 | 0.0747 | 0.1000 |
| nusc | round1 combined | 4 | 0.7500 | 12.5000 | 0.9755 | 0.0796 | 0.0871 |
| nusc | round1 combined / none | 4 | 0.7500 | 25.0000 | 0.9674 | 0.0797 | 0.0871 |
| nusc | round1 combined / half | 4 | 0.7500 | 25.0000 | 0.9702 | 0.0797 | 0.0871 |
| nusc | proactive | 4 | 0.7500 | 12.5000 | 0.9690 | 0.0787 | 0.0876 |
| nusc | proactive / none | 4 | 0.7500 | 25.0000 | 0.9610 | 0.0787 | 0.0876 |
| nusc | proactive / half | 4 | 0.7500 | 25.0000 | 0.9638 | 0.0787 | 0.0876 |
| nusc | proactive + recovery | 4 | 0.5000 | 12.5000 | 0.9623 | 0.0828 | 0.0883 |
| nusc | proactive + recovery / none | 4 | 0.5000 | 25.0000 | 0.9543 | 0.0829 | 0.0883 |
| nusc | proactive + recovery / half | 4 | 0.5000 | 25.0000 | 0.9571 | 0.0829 | 0.0883 |
| nusc | looming + recovery | 4 | 0.7500 | 12.5000 | 0.9755 | 0.0796 | 0.0871 |
| nusc | looming + recovery / none | 4 | 0.7500 | 25.0000 | 0.9674 | 0.0797 | 0.0871 |
| nusc | looming + recovery / half | 4 | 0.7500 | 25.0000 | 0.9702 | 0.0797 | 0.0871 |
| av2 | cheap | 5 | 0.0833 | 38.4615 | 0.7678 | 0.0000 | 0.0258 |
| av2 | v6-S | 5 | 0.5000 | 5.1282 | 1.3214 | -0.1850 | 0.0497 |
| av2 | v6-S / none | 5 | 0.1667 | 10.2564 | 1.0720 | -0.0674 | 0.0497 |
| av2 | v6-S / half | 5 | 0.1667 | 15.3846 | 1.1305 | -0.0801 | 0.0497 |
| av2 | round1 combined | 5 | 0.5000 | 5.1282 | 1.2826 | -0.1515 | 0.0470 |
| av2 | round1 combined / none | 5 | 0.1667 | 10.2564 | 1.0325 | -0.0431 | 0.0470 |
| av2 | round1 combined / half | 5 | 0.1667 | 15.3846 | 1.0910 | -0.0516 | 0.0470 |
| av2 | proactive | 5 | 0.5000 | 5.1282 | 1.2893 | -0.1571 | 0.0466 |
| av2 | proactive / none | 5 | 0.1667 | 12.8205 | 1.0382 | -0.0447 | 0.0466 |
| av2 | proactive / half | 5 | 0.1667 | 15.3846 | 1.0976 | -0.0537 | 0.0466 |
| av2 | proactive + recovery | 5 | 0.5000 | 5.1282 | 1.2796 | -0.1494 | 0.0472 |
| av2 | proactive + recovery / none | 5 | 0.1667 | 12.8205 | 1.0239 | -0.0376 | 0.0472 |
| av2 | proactive + recovery / half | 5 | 0.1667 | 15.3846 | 1.0880 | -0.0495 | 0.0472 |
| av2 | looming + recovery | 5 | 0.4167 | 2.5641 | 1.2761 | -0.1469 | 0.0470 |
| av2 | looming + recovery / none | 5 | 0.1667 | 10.2564 | 1.0325 | -0.0431 | 0.0470 |
| av2 | looming + recovery / half | 5 | 0.1667 | 15.3846 | 1.0881 | -0.0496 | 0.0470 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
