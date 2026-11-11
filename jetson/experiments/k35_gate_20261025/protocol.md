# K35-4b gate — Jetson dev-prefix timing (registered 2026-10-25, before the window)

Rows (all trt32, same process, rotated order, independent histories):
- ours = frozen v6-S;
- ours_b2A = B2K-A;
- ours_b2A_retro = v7a (B2K-A + C1 retrospective recovery);
- ours_b2A_missing_extent_retro = v7b (v7a + image-point fallback + critical-extent retention = VUW R3 low
  `v6S_b2A_missing_extent_retro`; the same name-flag composition of the Jetson '_missing' and '_extent' components that
  R3 timed separately).

Runtime (K35 timing note): every timed row uses the exact recovery bookkeeping helper (FAST_RECOVERY_20261023,
`--helper fast`) and crop-buffer staging (RUNTIME_20261023 staged.py, `--runtime staged`). There is no other change.
- Reference runtime = KF.Crops / KB.OneCall + the original Retro helper. It is run once per dataset on the same 712 frames.
- Action parity is reported for every row on all 512 timed frames of each staged repetition: the rate and every
  differing frame (logs/parity_<ds>_{rows,frames}.csv).
- A 10-frame dry run on nusc_val before the window matched 10/10 on all rows and counters (dry/).

Frames: nusc_val and the verified AV2 CFG-DEV manifest (verified_log_manifest.json, ego mask 1830). First 712
chronological frames, 200 warm-up, 512 timed; three staged repetitions.

Gate (K35 ADDENDUM):
- Statistic: p95 of per-frame latency (upload → decision, with refiner) over the 3 × 512 timed frames pooled.
- v7b is dropped from K35-4 if its p95 > v6-S's p95 on either dataset (point estimate).
- Also reported: mean / p99 / >50 / 66.7 / 100 ms, and paired unit bootstrap intervals (2,000, seed 0, repetitions nested in
  units) of the p95 and mean differences.
- v7a's numbers are reported, but do not gate v7a.

Exclusive Jetson window (pause file, busy.sh = paused, no ~/.r3_profiling), MAXN, locked clocks, fan 255, restore in
finally. Source and engine hashes are checked before and after. No engine builds. Dev data only.
