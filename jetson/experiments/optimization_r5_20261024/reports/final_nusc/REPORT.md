# Round 5 sandbox development experiments — residual

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
| nusc | R5 residual ungated | 152 | 0.3434 | 11.4435 | 0.3374 | 0.3589 | 0.0750 |
| nusc | R5 domain_both q50 | 152 | 0.3232 | 6.2122 | 0.3555 | 0.3656 | 0.0750 |
| nusc | R5 residual q50 | 152 | 0.3737 | 4.2504 | 0.3485 | 0.3829 | 0.0750 |
| nusc | R5 residual_both q50 | 152 | 0.3232 | 6.5392 | 0.3362 | 0.3971 | 0.0750 |
| nusc | R5 domain_both q80 | 152 | 0.3737 | 4.2504 | 0.3643 | 0.3558 | 0.0750 |
| nusc | R5 residual q80 | 152 | 0.3838 | 3.9235 | 0.3581 | 0.3684 | 0.0750 |
| nusc | R5 residual_both q80 | 152 | 0.3737 | 4.2504 | 0.3540 | 0.3737 | 0.0750 |
| nusc | R5 domain_both q95 | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | R5 residual q95 | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | R5 residual_both q95 | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | critical extent retention | 152 | 0.3838 | 3.9235 | 0.3588 | 0.3640 | 0.0778 |
| nusc | R5 extent_both q50 | 152 | 0.3232 | 6.2122 | 0.3459 | 0.3791 | 0.0778 |
| nusc | R5 extent_both q80 | 152 | 0.3737 | 4.2504 | 0.3547 | 0.3693 | 0.0778 |
| nusc | R5 extent_both q95 | 152 | 0.3838 | 3.9235 | 0.3588 | 0.3640 | 0.0778 |
| nusc | R5 point_residual q50 | 152 | 0.3434 | 5.5583 | 0.3171 | 0.4150 | 0.0784 |
| nusc | R5 point_residual q80 | 152 | 0.3535 | 5.2313 | 0.3266 | 0.4005 | 0.0784 |
| nusc | R5 point_residual q95 | 152 | 0.3535 | 5.2313 | 0.3370 | 0.3826 | 0.0784 |
| nusc | R5 extent_residual q50 | 152 | 0.3737 | 4.2504 | 0.3388 | 0.3963 | 0.0778 |
| nusc | R5 extent_residual q80 | 152 | 0.3838 | 3.9235 | 0.3484 | 0.3819 | 0.0778 |
| nusc | R5 extent_residual q95 | 152 | 0.3838 | 3.9235 | 0.3588 | 0.3640 | 0.0778 |
| nusc | R5 cond round1 pooled q50 | 152 | 0.3737 | 4.2504 | 0.3675 | 0.3515 | 0.0750 |
| nusc | R5 cond round1 pooled q80 | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | R5 cond round1 pooled q95 | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | R5 cond round1 conditional q50 | 152 | 0.3838 | 4.4139 | 0.3675 | 0.3514 | 0.0750 |
| nusc | R5 cond round1 conditional q80 | 152 | 0.3838 | 4.2504 | 0.3681 | 0.3509 | 0.0750 |
| nusc | R5 cond round1 conditional q95 | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | R5 cond point pooled q50 | 152 | 0.3434 | 5.3948 | 0.3363 | 0.3834 | 0.0784 |
| nusc | R5 cond point pooled q80 | 152 | 0.3535 | 5.2313 | 0.3370 | 0.3826 | 0.0784 |
| nusc | R5 cond point pooled q95 | 152 | 0.3535 | 5.2313 | 0.3370 | 0.3826 | 0.0784 |
| nusc | R5 cond point conditional q50 | 152 | 0.3535 | 5.8852 | 0.3360 | 0.3836 | 0.0784 |
| nusc | R5 cond point conditional q80 | 152 | 0.3535 | 5.5583 | 0.3366 | 0.3830 | 0.0784 |
| nusc | R5 cond point conditional q95 | 152 | 0.3535 | 5.2313 | 0.3370 | 0.3826 | 0.0784 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
