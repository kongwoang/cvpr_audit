> Dataset correction (2026-10-23): the legacy CLI label `av2_dev` below refers to the cached K29 runtime subset, which includes REF-TRAIN logs. It is not an independent CFG-DEV quality evaluation. All measurements are retained; use them only for runtime/output-preservation engineering. The full VUW quality screen was audited against actual available training/selection unit IDs and checkpoint feature norms, with zero overlap. Corrected AV2 timing on a verified CFG-DEV manifest is reported separately in VERIFIED_JETSON_20261023. See ../OPTIMIZATION_20261023/dataset_audit/legacy_summary.csv.

# Jetson crop-buffer staging experiment

All rows retained. Existing trt32 engines; frozen method and main source unchanged.
Dev prefix only: 512 timed frames/dataset/repetition after 200 warm-up frames, three repetitions.
Full action/cost/crop/execution parity holds on all 712 chronological frames of every paired run.
Intervals: paired unit bootstrap, 2,000, seed 0, repetitions nested within units; pooled quantiles.
Shared detection caches are disabled during timing. Disk decode excluded; upload included.

| Dataset | Pipeline | Runtime | Mean ms | p95 ms | p99 ms | >50 ms % | >66.7 ms % | >100 ms % |
|---|---|---|---:|---:|---:|---:|---:|---:|
| av2_dev | B2K-A | reference | 42.122 | 72.370 | 89.890 | 41.471 | 8.854 | 0.391 |
| av2_dev | B2K-A | staged | 41.635 | 70.666 | 89.224 | 40.495 | 7.943 | 0.195 |
| av2_dev | v6-S | reference | 45.404 | 92.415 | 117.599 | 39.193 | 20.638 | 3.320 |
| av2_dev | v6-S | staged | 44.584 | 91.312 | 114.861 | 38.216 | 18.424 | 2.930 |
| av2_dev | v6-S <=2 | reference | 40.042 | 68.410 | 83.424 | 31.641 | 5.599 | 0.000 |
| av2_dev | v6-S <=2 | staged | 39.507 | 67.290 | 82.311 | 29.688 | 5.534 | 0.000 |
| nusc_val | B2K-A | reference | 23.327 | 48.142 | 56.690 | 4.102 | 0.065 | 0.000 |
| nusc_val | B2K-A | staged | 23.006 | 47.172 | 55.062 | 3.711 | 0.065 | 0.065 |
| nusc_val | v6-S | reference | 25.704 | 61.575 | 80.733 | 10.417 | 3.060 | 0.195 |
| nusc_val | v6-S | staged | 25.193 | 60.674 | 79.628 | 9.961 | 2.539 | 0.195 |
| nusc_val | v6-S <=2 | reference | 22.532 | 44.210 | 53.008 | 1.953 | 0.065 | 0.000 |
| nusc_val | v6-S <=2 | staged | 22.194 | 43.504 | 52.457 | 1.823 | 0.065 | 0.000 |

Paired differences and intervals: contrasts.csv. These are development runtime results on a small
chronological prefix, not a full-dataset speed or method-quality claim.
Exact manifests: frame_manifest.csv. Hardware, engine hashes, source hashes, clocks and telemetry:
experiments/runtime_20261023/logs/. All owned clocks/fan/pause controls were released.
The implementation writes current float letterboxes directly into persistent engine input buffers;
the candidate also uses explicit-stream enqueue. This experiment does not separate their individual effects.
