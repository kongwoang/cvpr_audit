# Round 3 sandbox development experiments — idsmoke

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
| nusc | upper-quantile missing history / none | 4 | 0.2500 | 31.2500 | 0.8526 | 0.1682 | 0.0871 |
| nusc | upper-quantile missing history | 4 | 0.2500 | 18.7500 | 0.8606 | 0.1681 | 0.0871 |
| nusc | identity gate + uncertain fallback / none | 4 | 0.2500 | 31.2500 | 0.8526 | 0.1682 | 0.0871 |
| nusc | identity gate + uncertain fallback | 4 | 0.2500 | 18.7500 | 0.8606 | 0.1681 | 0.0871 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
