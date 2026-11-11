# Round 3 sandbox development experiments — learned

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
| nusc | learned missing history / none | 152 | 0.3232 | 9.3183 | 0.3515 | 0.3602 | 0.0750 |
| nusc | learned missing history | 152 | 0.3535 | 5.3948 | 0.3521 | 0.3648 | 0.0750 |
| nusc | learned geometric / none | 152 | 0.3232 | 8.9913 | 0.3509 | 0.3619 | 0.0750 |
| nusc | learned geometric | 152 | 0.3636 | 5.3948 | 0.3528 | 0.3629 | 0.0750 |
| nusc | learned geometric / adapt | 152 | 0.3434 | 7.1931 | 0.3513 | 0.3595 | 0.0750 |
| nusc | learned decision / none | 152 | 0.3232 | 9.1548 | 0.3526 | 0.3606 | 0.0750 |
| nusc | learned decision | 152 | 0.3636 | 5.7218 | 0.3540 | 0.3618 | 0.0750 |
| nusc | learned decision / adapt | 152 | 0.3434 | 7.0296 | 0.3555 | 0.3628 | 0.0750 |
| nusc | upper-quantile missing history / none | 152 | 0.2626 | 16.3479 | 0.3266 | 0.3718 | 0.0750 |
| nusc | upper-quantile missing history | 152 | 0.2929 | 11.2800 | 0.3300 | 0.3755 | 0.0750 |
| nusc | uncertainty query / none | 152 | 0.3030 | 10.2992 | 0.3379 | 0.3718 | 0.0773 |
| nusc | uncertainty query | 152 | 0.3434 | 6.8661 | 0.3396 | 0.3742 | 0.0773 |
| nusc | uncertainty query / adapt | 152 | 0.3232 | 8.6644 | 0.3383 | 0.3714 | 0.0773 |
| av2 | cheap | 300 | 0.0820 | 35.9603 | 0.3140 | 0.0000 | 0.0258 |
| av2 | v6-S / none | 300 | 0.1202 | 12.5079 | 0.3132 | 0.2760 | 0.0556 |
| av2 | v6-S | 300 | 0.1612 | 7.8597 | 0.3798 | 0.3087 | 0.0556 |
| av2 | round1 combined / none | 300 | 0.0929 | 13.4376 | 0.3013 | 0.2859 | 0.0514 |
| av2 | round1 combined | 300 | 0.1339 | 9.0006 | 0.3685 | 0.3194 | 0.0514 |
| av2 | round1 combined / half | 300 | 0.1175 | 10.5219 | 0.3263 | 0.3293 | 0.0514 |
| av2 | round1 combined / speed | 300 | 0.1393 | 8.6626 | 0.3781 | 0.3101 | 0.0514 |
| av2 | round1 combined / speedhalf | 300 | 0.1311 | 8.9584 | 0.3719 | 0.3169 | 0.0514 |
| av2 | learned missing history / none | 300 | 0.0820 | 14.1137 | 0.2817 | 0.3038 | 0.0514 |
| av2 | learned missing history | 300 | 0.1202 | 9.9303 | 0.3460 | 0.3427 | 0.0514 |
| av2 | learned geometric / none | 300 | 0.0820 | 13.5643 | 0.2838 | 0.3043 | 0.0514 |
| av2 | learned geometric | 300 | 0.1202 | 9.2542 | 0.3502 | 0.3400 | 0.0514 |
| av2 | learned geometric / adapt | 300 | 0.1393 | 9.2119 | 0.3515 | 0.3399 | 0.0514 |
| av2 | learned decision / none | 300 | 0.0820 | 13.3531 | 0.2841 | 0.3044 | 0.0514 |
| av2 | learned decision | 300 | 0.1230 | 9.0429 | 0.3510 | 0.3397 | 0.0514 |
| av2 | learned decision / adapt | 300 | 0.1475 | 7.1413 | 0.4111 | 0.4003 | 0.0514 |
| av2 | upper-quantile missing history / none | 300 | 0.0765 | 16.1420 | 0.2747 | 0.3033 | 0.0514 |
| av2 | upper-quantile missing history | 300 | 0.1120 | 12.0008 | 0.3372 | 0.3436 | 0.0514 |
| av2 | uncertainty query / none | 300 | 0.0628 | 15.0433 | 0.2719 | 0.3138 | 0.0527 |
| av2 | uncertainty query | 300 | 0.1011 | 10.8177 | 0.3387 | 0.3500 | 0.0527 |
| av2 | uncertainty query / adapt | 300 | 0.1202 | 10.8177 | 0.3415 | 0.3483 | 0.0527 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
