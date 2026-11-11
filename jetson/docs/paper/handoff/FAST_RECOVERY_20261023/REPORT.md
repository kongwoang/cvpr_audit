> Dataset correction (2026-10-23): the legacy CLI label `av2_dev` below refers to the cached K29 runtime subset, which includes REF-TRAIN logs. It is not an independent CFG-DEV quality evaluation. All measurements are retained; use them only for runtime/output-preservation engineering. The full VUW quality screen was audited against actual available training/selection unit IDs and checkpoint feature norms, with zero overlap. Corrected AV2 timing on a verified CFG-DEV manifest is reported separately in VERIFIED_JETSON_20261023. See ../OPTIMIZATION_20261023/dataset_audit/legacy_summary.csv.

# Jetson exact recovery bookkeeping pilot

All rows retained. Existing trt32 engines; frozen method and main source unchanged.
Dev prefix only: 256 timed frames/dataset/repetition after 200 warm-up frames, three repetitions.
Full action/cost/crop/execution parity holds on all 456 chronological frames of every paired run.
Intervals: paired unit bootstrap, 2,000, seed 0, repetitions nested within units; pooled quantiles.
Shared detection caches are disabled during timing. Disk decode excluded; upload included.

| Dataset | Pipeline | Runtime | Mean ms | p95 ms | p99 ms | >50 ms % | >66.7 ms % | >100 ms % |
|---|---|---|---:|---:|---:|---:|---:|---:|
| av2_dev | B2K-A | original | 43.651 | 80.151 | 95.168 | 48.177 | 15.104 | 0.391 |
| av2_dev | B2K-A | fast | 43.735 | 79.810 | 94.857 | 48.698 | 15.104 | 0.391 |
| av2_dev | B2K-A + recovery | original | 48.829 | 87.044 | 120.598 | 58.073 | 23.958 | 3.516 |
| av2_dev | B2K-A + recovery | fast | 45.994 | 82.037 | 117.184 | 51.693 | 19.010 | 2.474 |
| av2_dev | v6-S | original | 46.400 | 95.567 | 117.926 | 43.359 | 23.698 | 3.385 |
| av2_dev | v6-S | fast | 46.517 | 94.201 | 116.367 | 43.880 | 23.047 | 3.516 |
| av2_dev | v6-S + recovery | original | 49.829 | 101.071 | 126.314 | 49.349 | 28.776 | 5.208 |
| av2_dev | v6-S + recovery | fast | 48.474 | 98.729 | 125.183 | 47.526 | 26.953 | 4.297 |
| nusc_val | B2K-A | original | 26.314 | 51.020 | 58.228 | 6.901 | 0.130 | 0.000 |
| nusc_val | B2K-A | fast | 26.233 | 51.451 | 59.255 | 6.380 | 0.130 | 0.000 |
| nusc_val | B2K-A + recovery | original | 28.360 | 56.460 | 67.613 | 11.198 | 1.172 | 0.000 |
| nusc_val | B2K-A + recovery | fast | 26.599 | 52.585 | 64.729 | 6.510 | 0.391 | 0.000 |
| nusc_val | v6-S | original | 30.162 | 67.436 | 85.612 | 16.667 | 5.208 | 0.391 |
| nusc_val | v6-S | fast | 30.049 | 67.360 | 85.004 | 16.667 | 5.208 | 0.391 |
| nusc_val | v6-S + recovery | original | 31.761 | 69.280 | 100.359 | 18.750 | 6.380 | 1.172 |
| nusc_val | v6-S + recovery | fast | 30.459 | 67.222 | 97.991 | 16.536 | 5.208 | 0.911 |

Paired differences and intervals: contrasts.csv. Recovery overhead adjusted for the base pipeline: overhead_contrasts.csv. These are development runtime results on a small
chronological prefix, not a full-dataset speed or method-quality claim.
Exact manifests: frame_manifest.csv. Hardware, engine hashes, source hashes, clocks and telemetry:
experiments/temporal_fast_20261023/logs/. All owned clocks/fan/pause controls were released.
This pilot preserves the original detector inputs, historical work, decisions and memory retention;
the fast helper reuses existing chains and avoids matching/logging when no cheap object reaches the threshold.
