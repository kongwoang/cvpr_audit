# Round 5 sandbox development experiments — readout

Development screen only; no method selected or promoted. Frozen sources and K34 are unchanged.
All intervals: paired unit bootstrap, 2,000, seed 0; pointwise, exploratory. No GPU wall time is a Jetson latency measurement.
G retains the original K30 cost convention (jerk relative to previous cheap action).
Every executed policy/refiner row is retained. Logical requested pixels, not Jetson padded pixels.
Generic-component/equal-total-budget controls remain necessary before a selector-superiority claim.

| Dataset | Pipeline | Units | Miss rate | False/1000 | Insufficient loss/frame | G | Mean px/native |
|---|---|---:|---:|---:|---:|---:|---:|
| av2 | cheap | 300 | 0.0820 | 35.9603 | 0.3140 | 0.0000 | 0.0258 |
| av2 | v6-S / none | 300 | 0.1202 | 12.5079 | 0.3132 | 0.2760 | 0.0556 |
| av2 | v6-S | 300 | 0.1612 | 7.8597 | 0.3798 | 0.3087 | 0.0556 |
| av2 | round1 combined / none | 300 | 0.0929 | 13.4376 | 0.3013 | 0.2859 | 0.0514 |
| av2 | round1 combined | 300 | 0.1339 | 9.0006 | 0.3685 | 0.3194 | 0.0514 |
| av2 | objectwise median | 300 | 0.1011 | 11.5783 | 0.3151 | 0.3029 | 0.0514 |
| av2 | R5 rollback q50 | 300 | 0.1257 | 10.3951 | 0.3535 | 0.3305 | 0.0514 |
| av2 | R5 missing q50 | 300 | 0.1311 | 9.2119 | 0.3668 | 0.3205 | 0.0514 |
| av2 | R5 both q50 | 300 | 0.1257 | 10.3951 | 0.3535 | 0.3305 | 0.0514 |
| av2 | R5 median_gate q50 | 300 | 0.1311 | 9.9725 | 0.3563 | 0.3288 | 0.0514 |
| av2 | R5 rollback q80 | 300 | 0.1339 | 9.1274 | 0.3656 | 0.3221 | 0.0514 |
| av2 | R5 missing q80 | 300 | 0.1339 | 9.0006 | 0.3684 | 0.3195 | 0.0514 |
| av2 | R5 both q80 | 300 | 0.1339 | 9.1274 | 0.3656 | 0.3221 | 0.0514 |
| av2 | R5 median_gate q80 | 300 | 0.1339 | 9.1274 | 0.3663 | 0.3213 | 0.0514 |
| av2 | R5 rollback q95 | 300 | 0.1339 | 9.0429 | 0.3682 | 0.3197 | 0.0514 |
| av2 | R5 missing q95 | 300 | 0.1339 | 9.0006 | 0.3685 | 0.3194 | 0.0514 |
| av2 | R5 both q95 | 300 | 0.1339 | 9.0429 | 0.3682 | 0.3197 | 0.0514 |
| av2 | R5 median_gate q95 | 300 | 0.1339 | 9.0429 | 0.3682 | 0.3197 | 0.0514 |
| av2 | R5 matched-only both q80 | 300 | 0.1339 | 9.0851 | 0.3651 | 0.3214 | 0.0514 |
| av2 | point fallback + extent retention | 300 | 0.1175 | 9.8035 | 0.3349 | 0.3483 | 0.0533 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
