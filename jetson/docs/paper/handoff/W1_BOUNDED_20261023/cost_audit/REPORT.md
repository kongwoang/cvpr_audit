# Requested vs realized control budgets

Full development set; all pixel values are detector input pixels / native image pixels.
These are logical VUW canvas pixels, not Jetson common-padding tensor pixels or latency.
Every requested budget equals the combined candidate current+guard+retrospective work plus one cheap pass.
Controls may overrun the request due to their guards. No exact equal-total-cost comparison is asserted.
Paired unit bootstrap 2000 seed0 for realized-minus-candidate mean pixel differences.

| Dataset | Control | Requested total px | Realized total px | Frames over request |
|---|---|---:|---:|---:|
| av2 | tsp8@combinedbudget+C1 | 0.051415 | 0.074499 | 50.14% |
| av2 | unc2@combinedbudget+C1 | 0.051415 | 0.083420 | 46.10% |
| nusc | tsp8@combinedbudget+C1 | 0.074970 | 0.109184 | 35.65% |
| nusc | unc2@combinedbudget+C1 | 0.074970 | 0.099839 | 26.34% |

Exact per-frame arithmetic checks passed for both controls on every evaluated frame.
Control detector-call counts are not recorded by this harness; missing counts are not zero.
