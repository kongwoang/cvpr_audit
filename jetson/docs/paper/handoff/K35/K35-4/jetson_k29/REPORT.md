# K35-4 — Jetson timing of v7a on the K29 frames, next to v6-S and B2K-A (consequence of the holdout pass)

Jetson AGX Xavier, exclusive window, 2026-10-25 21:48–22:08 (kit time, UTC+7). MAXN, locked clocks, fan 255; all restored.
Source and engine hashes were unchanged before vs after; no engine builds.
- **Harness.** experiments/k35_k29_20261026, the K35-4b harness with rows v6-S / B2K-A / v7a. md5:
  - 87_k35_k29.py 418ed445;
  - compare.py 7e3c8e4c;
  - window.py 921d0b14;
  - report.py a8cbfd54;
  - staged.py 244ea468.
- **Protocol.** protocol.md, registered before the window.
- **Frames.** The exact K29 / K34 manifests, asserted against handoff/K34/reference/.
  - nuScenes test: 20 scenes, 805 timed frames, after a 200-frame warm-up pass.
  - AV2: 8 timed logs, 586 frames, ego mask 1830.
  - Three repetitions.
- **Runtime (K35 timing note).** Every row uses the FAST_RECOVERY helper and crop-buffer staging.

## Action parity vs the reference runtime (KF.Crops / KB.OneCall + original Retro helper; one run per dataset)
3 rows × 3 repetitions × all timed frames:
- **0 differing actions** on both datasets (parity rate 1.000);
- 0 differences in J, px, detector calls, crop-engine executions and recovery counters;
- the 10-frame dry runs matched too (dry/).

Files: parity_<ds>_rows.csv, parity_<ds>_frames.csv (empty), parity_<ds>.json.

## Latency (ms, upload → decision with refiner; 3 repetitions pooled; k29_rows.csv)

| dataset | row | mean | median | p95 | p99 | >50 ms | >66.7 ms | >100 ms |
|---|---|---|---|---|---|---|---|---|
| nuScenes test 20 | v6-S | 21.48 | 14.15 | **58.19** | 81.76 | 8.7% | 2.1% | 0.0% |
| nuScenes test 20 | B2K-A | 20.16 | 14.10 | 48.59 | 59.29 | 4.3% | 0.4% | 0.0% |
| nuScenes test 20 | v7a | 20.55 | 14.19 | **50.50** | 60.60 | 5.1% | 0.4% | 0.0% |
| AV2 8 logs | v6-S | 39.85 | 38.00 | **89.36** | 119.24 | 34.9% | 15.7% | 3.1% |
| AV2 8 logs | B2K-A | 38.42 | 43.33 | 77.67 | 99.94 | 38.6% | 9.6% | 1.0% |
| AV2 8 logs | v7a | 40.54 | 44.95 | **84.51** | 112.52 | 41.2% | 12.6% | 1.5% |

**Differences vs v6-S** (ms). Paired unit bootstrap: 2,000 draws, seed 0, repetitions nested in units; 20 scenes / 8 logs;
k29_diffs.csv.

| dataset | row | Δp95 [95% CI] | Δmean [95% CI] |
|---|---|---|---|
| nuScenes | B2K-A | −9.60 [−13.74, −1.74] | −1.32 [−2.54, −0.39] |
| nuScenes | v7a | −7.69 [−12.58, −0.36] | −0.93 [−2.19, +0.03] |
| AV2 | B2K-A | −11.68 [−21.36, −4.80] | −1.42 [−2.33, −0.59] |
| AV2 | v7a | −4.85 [−17.06, +2.54] | +0.70 [−0.59, +2.10] |

## Reading
- **v7a vs v6-S.**
  - nuScenes: v7a's p95 is lower by 7.7 ms, and the interval excludes 0.
  - AV2: the p95 is lower by 4.8 ms, but the interval includes 0 (8 logs). Mean latency is unchanged within ±1 ms.
  - On AV2 its >50 ms share is higher (41.2% vs 34.9%), while its >66.7 ms and >100 ms shares are lower.
- **v7a vs B2K-A.** The recovery step costs +0.4 / +2.1 ms mean and +1.9 / +6.8 ms p95 over B2K-A. So most of the
  tail benefit comes from the bounded verification. This matches the dev prefixes (K35-4b).
- **Reference runtime, one run, for context only.**
  - v6-S: mean 21.68 / 40.29 ms, p95 59.19 / 92.12 ms.
  - v7a: mean 22.25 / 44.09 ms, p95 56.01 / 92.22 ms.
  - The staged runtime is about 0.2–3.5 ms faster in mean, with identical actions. The largest gain is on v7a on AV2.
  - These numbers are consistent with K34's v6-S (21.84 / 40.39 ms mean). They are not pooled with K34.
