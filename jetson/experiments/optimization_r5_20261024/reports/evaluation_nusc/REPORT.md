# Round 5 sandbox development experiments — readout

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
| nusc | objectwise median | 152 | 0.3838 | 5.5583 | 0.3721 | 0.3486 | 0.0750 |
| nusc | R5 rollback q50 | 152 | 0.3737 | 4.7409 | 0.3660 | 0.3523 | 0.0750 |
| nusc | R5 missing q50 | 152 | 0.3232 | 5.5583 | 0.3573 | 0.3645 | 0.0750 |
| nusc | R5 both q50 | 152 | 0.3232 | 6.2122 | 0.3555 | 0.3656 | 0.0750 |
| nusc | R5 median_gate q50 | 152 | 0.3838 | 4.0870 | 0.3673 | 0.3513 | 0.0750 |
| nusc | R5 rollback q80 | 152 | 0.3838 | 4.0870 | 0.3681 | 0.3505 | 0.0750 |
| nusc | R5 missing q80 | 152 | 0.3737 | 4.2504 | 0.3645 | 0.3559 | 0.0750 |
| nusc | R5 both q80 | 152 | 0.3737 | 4.2504 | 0.3643 | 0.3558 | 0.0750 |
| nusc | R5 median_gate q80 | 152 | 0.3838 | 3.9235 | 0.3683 | 0.3505 | 0.0750 |
| nusc | R5 rollback q95 | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | R5 missing q95 | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | R5 both q95 | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | R5 median_gate q95 | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | R5 matched-only both q80 | 152 | 0.3434 | 4.5774 | 0.3610 | 0.3602 | 0.0750 |
| nusc | point fallback + extent retention | 152 | 0.3535 | 5.2313 | 0.3370 | 0.3826 | 0.0784 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
