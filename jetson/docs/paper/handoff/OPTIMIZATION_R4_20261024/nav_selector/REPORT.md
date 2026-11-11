# Exact bottleneck-selector cache: same-host CPU timing

A separate fixed extension motivated by source algebra, before this benchmark. K31 selector, unchanged 15 trajectories, scores, probe grids and tie ordering. Same frozen K34/K20 1,000 navtest tokens / 125 units, explicitly post-hoc runtime/parity only. No quality selection, detector rerun or planner replacement.
For each fixed candidate, s(S)=min(b,min_i h(i)); hence s(S+probe)=min(s(S),h(probe)). Original code recalculates every existing object for every probe. Cache s(S) and H once, batch the probe-only columns, and reuse H for binding support. All per-object TTC arithmetic still calls the original Selector.h.
One CPU thread on VUW1; five warm-up repeats, three timed repetitions of both implementations and both grids with alternating order. Scope includes scene construction, selector initialization, probe projection, field and support; excludes disk loading and image/detector inference. Every original-vs-cached action, field box and support box/order is identical in 6,000 comparisons. 7x7 also agrees with stored K31 first-round action/counts.

| grid | implementation | mean ms [95% CI] | p95 ms [95% CI] | candidate-object TTC evaluations |
| --- | --- | --- | --- | --- |
| 7x7 | original | 100.969 [85.612, 118.067] | 309.998 [257.361, 365.918] | 29681.565 [24670.196, 35223.178] |
| 7x7 | cached | 10.957 [10.310, 11.709] | 18.964 [16.785, 20.582] | 1271.025 [1162.099, 1392.187] |
| 3x3 | original | 24.112 [20.580, 28.077] | 70.933 [57.701, 84.645] | 6425.160 [5330.830, 7631.006] |
| 3x3 | cached | 6.392 [5.797, 7.054] | 13.643 [11.729, 15.278] | 750.975 [641.570, 871.653] |

| grid | speedup [95% CI] |
| --- | --- |
| 3x3 | 3.77× [3.53, 3.99] |
| 7x7 | 9.21× [8.25, 10.18] |

Paired unit bootstrap 2,000 seed0, repetitions nested within units; pooled p95 recomputed per draw. All effects: contrasts.csv; raw rows: frames.csv. Speedup ratios refer to this selector stage, not total perception latency or PDM-Closed. The 15 fixed candidate trajectories and h-independence permit this reduction; arbitrary planners need separate derivation.
Sources and hashes: sources_before.json / sources_after.json; exact token manifest and parity.csv. Isolated implementations: experiments/optimization_r4_20261024/selector_cache.py and nav_selector.py. Core and K34 artifacts remain unchanged.
