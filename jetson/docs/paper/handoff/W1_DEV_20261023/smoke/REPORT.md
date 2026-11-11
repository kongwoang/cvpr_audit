# W1 isolated development experiments — smoke

Development screen only; no method selected or promoted. Frozen sources and K34 are unchanged.
All intervals: paired unit bootstrap, 2,000, seed 0; pointwise, exploratory. No GPU wall time is a Jetson latency measurement.
G retains the original K30 cost convention (jerk relative to previous cheap action).
Controls tsp8/unc2 use baseline v6-S crop-pixel allowance; total realized pixels can differ due to guards.
Generic-component/equal-total-budget controls remain necessary before a selector-superiority claim.

| Dataset | Pipeline | Units | Miss rate | False/1000 | Insufficient loss/frame | G | Mean px/native |
|---|---|---:|---:|---:|---:|---:|---:|
| nusc | cheap | 4 | nan | 44.0252 | 0.0511 | 0.0000 | 0.0427 |
| nusc | v6-S | 4 | nan | 6.2893 | 0.0578 | 0.3683 | 0.0715 |
| nusc | v6-S no refiner | 4 | nan | 6.2893 | 0.0561 | 0.3538 | 0.0715 |
| nusc | temporal evidence | 4 | nan | 6.2893 | 0.0578 | 0.3683 | 0.0724 |
| nusc | temporal evidence no refiner | 4 | nan | 6.2893 | 0.0561 | 0.3538 | 0.0724 |
| nusc | post-refinement closure | 4 | nan | 6.2893 | 0.0578 | 0.3683 | 0.0715 |
| nusc | draft contradiction recheck | 4 | nan | 6.2893 | 0.0578 | 0.3683 | 0.0718 |
| nusc | draft recheck no refiner | 4 | nan | 6.2893 | 0.0561 | 0.3538 | 0.0718 |
| nusc | hybrid at v6 crop budget | 4 | nan | 6.2893 | 0.0578 | 0.3683 | 0.1041 |
| nusc | unc2 at v6 crop budget | 4 | nan | 6.2893 | 0.0578 | 0.3683 | 0.1122 |
| nusc | u960 + C1 + v3a | 4 | nan | 6.2893 | 0.0578 | 0.3683 | 0.3840 |
| nusc | u960 + C1 no refiner | 4 | nan | 6.2893 | 0.0578 | 0.3683 | 0.3840 |
| nusc | u960 + C1 closure | 4 | nan | 6.2893 | 0.0578 | 0.3683 | 0.3840 |
| nusc | native + C1 + v3a | 4 | nan | 6.2893 | 0.0601 | 0.3755 | 1.0311 |
| nusc | native + C1 no refiner | 4 | nan | 6.2893 | 0.0601 | 0.3755 | 1.0311 |
| nusc | native + C1 closure | 4 | nan | 6.2893 | 0.0601 | 0.3755 | 1.0311 |
| av2 | cheap | 4 | 0.2000 | 35.0318 | 0.7034 | 0.0000 | 0.0258 |
| av2 | v6-S | 4 | 0.2000 | 0.0000 | 0.7150 | 0.2124 | 0.0553 |
| av2 | v6-S no refiner | 4 | 0.2000 | 0.0000 | 0.7074 | 0.1916 | 0.0553 |
| av2 | temporal evidence | 4 | 0.2000 | 0.0000 | 0.7150 | 0.2124 | 0.0570 |
| av2 | temporal evidence no refiner | 4 | 0.2000 | 0.0000 | 0.7074 | 0.1916 | 0.0570 |
| av2 | post-refinement closure | 4 | 0.2000 | 0.0000 | 0.7150 | 0.2124 | 0.0558 |
| av2 | draft contradiction recheck | 4 | 0.2000 | 0.0000 | 0.7150 | 0.2124 | 0.0554 |
| av2 | draft recheck no refiner | 4 | 0.2000 | 0.0000 | 0.7074 | 0.1916 | 0.0554 |
| av2 | hybrid at v6 crop budget | 4 | 0.2000 | 3.1847 | 0.7130 | 0.2134 | 0.0780 |
| av2 | unc2 at v6 crop budget | 4 | 0.2000 | 12.7389 | 0.7127 | 0.1799 | 0.0952 |
| av2 | u960 + C1 + v3a | 4 | 0.2000 | 6.3694 | 0.7075 | 0.2133 | 0.2323 |
| av2 | u960 + C1 no refiner | 4 | 0.2000 | 9.5541 | 0.7000 | 0.1853 | 0.2323 |
| av2 | u960 + C1 closure | 4 | 0.2000 | 6.3694 | 0.7075 | 0.2133 | 0.2323 |
| av2 | native + C1 + v3a | 4 | 0.2000 | 3.1847 | 0.8807 | 0.0631 | 1.0116 |
| av2 | native + C1 no refiner | 4 | 0.2000 | 9.5541 | 0.8728 | 0.0309 | 1.0116 |
| av2 | native + C1 closure | 4 | 0.2000 | 3.1847 | 0.8807 | 0.0631 | 1.0116 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen.
