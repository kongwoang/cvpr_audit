# Exact NAVSIM bottleneck-selector score reuse

All 1,000 K34 tokens, both grids, three paired repetitions; 6,000 paired rows.
All candidate scores, actions, field and binding-support arrays match exactly on every repetition.
Each host uses one CPU thread; whole logs assigned to VUW/VUW1 by fixed hash.
Intervals: paired log bootstrap, 2,000, seed 0, repetitions nested in logs; pooled quantiles.

| Grid | Method | Mean ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|
| 3x3 | reference | 28.108 | 82.210 | 129.352 |
| 3x3 | reuse | 7.615 | 16.226 | 24.466 |
| 7x7 | reference | 115.861 | 353.612 | 554.467 |
| 7x7 | reuse | 12.979 | 22.224 | 31.059 |

The fixed selector reuses its base scene scores and scores all probe objects in one matrix.
Candidate paths, probe grid, argmax tie rule and binding support are unchanged.
Scope includes scene/selector construction and field/support extraction; excludes disk, detector,
and the rest of the perception/planning pipeline. This is not a Jetson or PDM-Closed speed claim.
Shared-host load and unit/host assignment limit hardware generalization; within-token pairs are on the same host.
All rows: frames.csv; paired contrasts: contrasts.csv; per-host results: host_summary.csv.
Original K31/K34 sources are unchanged. Raw shard metadata/source hashes retained in the experiment folder.
