# Bounded verification x temporal evidence — dev

Development screen only; no method selected or promoted. Frozen sources and K34 are unchanged.
All intervals: paired unit bootstrap, 2,000, seed 0; pointwise, exploratory. No GPU wall time is a Jetson latency measurement.
G retains the original K30 cost convention (jerk relative to previous cheap action).
Controls tsp8/unc2 use combined candidate crop-pixel allowance including recovery; total realized pixels can differ due to guards.
Generic-component/equal-total-budget controls remain necessary before a selector-superiority claim.

| Dataset | Pipeline | Units | Miss rate | False/1000 | Insufficient loss/frame | G | Mean px/native |
|---|---|---:|---:|---:|---:|---:|---:|
| nusc | cheap | 152 | 0.3434 | 27.9549 | 0.5038 | 0.0000 | 0.0427 |
| nusc | v6-S | 152 | 0.4040 | 2.9426 | 0.3874 | 0.3212 | 0.0804 |
| nusc | v6-S no refiner | 152 | 0.3939 | 6.5392 | 0.3911 | 0.3130 | 0.0804 |
| nusc | temporal evidence | 152 | 0.3838 | 3.2696 | 0.3848 | 0.3247 | 0.0809 |
| nusc | temporal evidence no refiner | 152 | 0.3737 | 7.0296 | 0.3894 | 0.3147 | 0.0809 |
| nusc | B2K-A | 152 | 0.4040 | 3.5965 | 0.3716 | 0.3462 | 0.0744 |
| nusc | B2K-A no refiner | 152 | 0.3939 | 7.3565 | 0.3748 | 0.3379 | 0.0744 |
| nusc | B2K-A + temporal evidence | 152 | 0.3838 | 3.9235 | 0.3684 | 0.3505 | 0.0750 |
| nusc | B2K-A + temporal evidence no refiner | 152 | 0.3737 | 7.8470 | 0.3725 | 0.3405 | 0.0750 |
| nusc | hybrid at combined requested budget | 152 | 0.4040 | 4.5774 | 0.4214 | 0.2564 | 0.1092 |
| nusc | unc2 at combined requested budget | 152 | 0.4444 | 6.2122 | 0.6043 | -0.0554 | 0.0998 |
| nusc | u960 + C1 + v3a | 152 | 0.3535 | 6.3757 | 0.4054 | 0.2715 | 0.3840 |
| nusc | u960 + C1 no refiner | 152 | 0.3131 | 10.2992 | 0.4078 | 0.2631 | 0.3840 |
| nusc | native + C1 + v3a | 152 | 0.4040 | 4.7409 | 0.3854 | 0.3231 | 1.0311 |
| nusc | native + C1 no refiner | 152 | 0.3535 | 10.4626 | 0.3832 | 0.3184 | 1.0311 |
| av2 | cheap | 73 | 0.0805 | 29.5797 | 0.2688 | 0.0000 | 0.0258 |
| av2 | v6-S | 73 | 0.2414 | 7.6111 | 0.3193 | 0.3236 | 0.0528 |
| av2 | v6-S no refiner | 73 | 0.1609 | 12.9735 | 0.2517 | 0.2706 | 0.0528 |
| av2 | temporal evidence | 73 | 0.1954 | 8.3031 | 0.3096 | 0.3338 | 0.0538 |
| av2 | temporal evidence no refiner | 73 | 0.1149 | 13.6655 | 0.2419 | 0.2791 | 0.0538 |
| av2 | B2K-A | 73 | 0.2414 | 8.6490 | 0.3191 | 0.3222 | 0.0481 |
| av2 | B2K-A no refiner | 73 | 0.1609 | 13.3195 | 0.2521 | 0.2657 | 0.0481 |
| av2 | B2K-A + temporal evidence | 73 | 0.1954 | 9.3409 | 0.3093 | 0.3324 | 0.0490 |
| av2 | B2K-A + temporal evidence no refiner | 73 | 0.1149 | 14.0114 | 0.2423 | 0.2741 | 0.0490 |
| av2 | hybrid at combined requested budget | 73 | 0.2414 | 8.8220 | 0.3151 | 0.3293 | 0.0695 |
| av2 | unc2 at combined requested budget | 73 | 0.2414 | 14.7033 | 0.3508 | 0.2197 | 0.0786 |
| av2 | u960 + C1 + v3a | 73 | 0.1724 | 11.4167 | 0.2777 | 0.3448 | 0.2323 |
| av2 | u960 + C1 no refiner | 73 | 0.1149 | 16.4331 | 0.2135 | 0.2823 | 0.2323 |
| av2 | native + C1 + v3a | 73 | 0.1954 | 10.2058 | 0.3624 | 0.2725 | 1.0116 |
| av2 | native + C1 no refiner | 73 | 0.1149 | 15.7412 | 0.2868 | 0.2026 | 1.0116 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen.
