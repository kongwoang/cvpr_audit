# Round 4 sandbox development experiments — readout

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
| nusc | objectwise median | 152 | 0.3838 | 5.5583 | 0.3721 | 0.3486 | 0.0750 |
| nusc | hard-brake corroboration | 152 | 0.3838 | 3.9235 | 0.3681 | 0.3506 | 0.0750 |
| nusc | framewise median | 152 | 0.3838 | 5.7218 | 0.3719 | 0.3476 | 0.0750 |
| av2 | cheap | 300 | 0.0820 | 35.9603 | 0.3140 | 0.0000 | 0.0258 |
| av2 | v6-S / none | 300 | 0.1202 | 12.5079 | 0.3132 | 0.2760 | 0.0556 |
| av2 | v6-S | 300 | 0.1612 | 7.8597 | 0.3798 | 0.3087 | 0.0556 |
| av2 | round1 combined / none | 300 | 0.0929 | 13.4376 | 0.3013 | 0.2859 | 0.0514 |
| av2 | round1 combined | 300 | 0.1339 | 9.0006 | 0.3685 | 0.3194 | 0.0514 |
| av2 | round1 combined / half | 300 | 0.1175 | 10.5219 | 0.3263 | 0.3293 | 0.0514 |
| av2 | round1 combined / speed | 300 | 0.1393 | 8.6626 | 0.3781 | 0.3101 | 0.0514 |
| av2 | round1 combined / speedhalf | 300 | 0.1311 | 8.9584 | 0.3719 | 0.3169 | 0.0514 |
| av2 | objectwise median | 300 | 0.1011 | 11.5783 | 0.3151 | 0.3029 | 0.0514 |
| av2 | hard-brake corroboration | 300 | 0.1339 | 8.5781 | 0.3560 | 0.2734 | 0.0514 |
| av2 | framewise median | 300 | 0.0984 | 11.9586 | 0.3108 | 0.3038 | 0.0514 |

Every executed row and its interval is retained in summary.csv, intervals.csv and contrasts.csv.
Severity/episode details: episodes.csv. Mechanism counts overlap and are not causal fractions: diagnostics.csv.
No latency, deployment, or confirmatory-test conclusion is supported by this development screen. Smoke units were selected diagnostic cases, not an unbiased quality sample.
