# K35-4 consequence of the v7a pass — Jetson timing on the K29 frames (registered 2026-10-26, before the window)

Rows: v6-S (frozen), B2K-A, v7a = B2K-A + C1 retrospective recovery. trt32; same process, rotated order, independent
histories.
- Frames: the exact K29 / K34 manifests. nuScenes test 20 scenes: 805 timed frames plus a 200-frame warm-up pass. AV2: 8 timed logs,
  586 frames (the first 200 frames excluded), ego mask 1830. Asserted against docs/paper/handoff/K34/reference/.
- Three repetitions.
- Runtime (K35 timing note): every row uses FAST_RECOVERY_20261023 (--helper fast) and RUNTIME_20261023 crop staging
  (--runtime staged). The reference runtime (KF.Crops / KB.OneCall + original Retro helper) is run once per dataset on the
  same frames. Action parity is reported for every row on all timed frames (rate and every differing frame).
- 10-frame dry runs on both datasets: 0 differences (dry/).
- Report: mean / p95 / p99 / >50 / 66.7 / 100 ms; paired unit bootstrap (2,000, seed 0, repetitions nested in units) of
  the differences vs v6-S. The K34 numbers (reference runtime, other window) are cited, not pooled.
- Exclusive window; MAXN, locked clocks, fan 255, restored in finally. Source and engine hashes are checked. No engine builds.
