> Dataset correction (2026-10-23): the legacy CLI label `av2_dev` below refers to the cached K29 runtime subset, which includes REF-TRAIN logs. It is not an independent CFG-DEV quality evaluation. All measurements are retained; use them only for runtime/output-preservation engineering. The full VUW quality screen was audited against actual available training/selection unit IDs and checkpoint feature norms, with zero overlap. Corrected AV2 timing on a verified CFG-DEV manifest is reported separately in VERIFIED_JETSON_20261023. See ../OPTIMIZATION_20261023/dataset_audit/legacy_summary.csv.

# Temporal evidence recovery — Jetson feasibility

All four prespecified rows retained; existing reference crop runtime/trt32 engines.
Two dev prefixes, 512 timed frames/dataset/rep after 200 warm-up frames; three repetitions.
Disabled-recovery port parity and enabled provenance/budget checks passed before timing.
Paired unit bootstrap 2000 seed0; repetitions nested in units; pooled quantiles.

| Dataset | Pipeline | Mean ms | p95 ms | p99 ms | >50 ms % | >66.7 ms % | >100 ms % |
|---|---|---:|---:|---:|---:|---:|---:|
| av2_dev | B2K-A | 41.724 | 71.632 | 89.547 | 40.299 | 8.398 | 0.391 |
| av2_dev | B2K-A + recovery | 46.126 | 78.370 | 111.894 | 49.023 | 16.536 | 1.888 |
| av2_dev | v6-S | 44.880 | 91.727 | 116.019 | 38.346 | 19.271 | 2.930 |
| av2_dev | v6-S + recovery | 47.828 | 96.748 | 121.932 | 43.229 | 24.935 | 4.232 |
| nusc_val | B2K-A | 23.114 | 47.406 | 55.302 | 3.711 | 0.065 | 0.000 |
| nusc_val | B2K-A + recovery | 25.256 | 52.576 | 61.326 | 7.878 | 0.651 | 0.000 |
| nusc_val | v6-S | 25.444 | 61.067 | 81.025 | 10.091 | 2.865 | 0.195 |
| nusc_val | v6-S + recovery | 26.880 | 64.438 | 84.672 | 12.370 | 3.581 | 0.586 |

Full quality evidence comes from the VUW development factorial. Prefix G/loss here are diagnostic
and use the original cheap-previous-action jerk convention; no independent closed-loop rollout.
Historical execution and pixels are included. Retaining two prior uploaded frames per recovery pipeline
uses extra GPU memory; memory.csv records retained tensors and process peaks. Comparators do not share histories.
Triggered latency is reported separately in triggered_latency.csv; an inactive prefix cannot characterize recovery tails.
Backend/canvas differences from VUW remain possible. This is not a cross-backend parity claim.
All main source/engine hashes were checked and the exclusive window released. No method is promoted.
