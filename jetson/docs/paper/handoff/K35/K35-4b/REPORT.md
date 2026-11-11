# K35-4b — v7b latency gate (Jetson, DEV prefixes), before PREREG

**Verdict: v7b is DROPPED before the K35-4 pre-registration.** Its p95 exceeds v6-S's on nuScenes val: 65.26 vs
62.75 ms (+2.51 ms). The ADDENDUM gate is "p95 > v6-S's on either dataset", decided on the point estimate as registered
in protocol.md. On AV2 CFG-DEV, v7b's p95 is below v6-S's (115.18 vs 117.52 ms). K35-4 therefore pre-registers v7a as
the only candidate row.

## Setup
- **Window.** Jetson AGX Xavier, 2026-10-25 13:40–14:05, exclusive window: pause file, busy.sh = paused, MAXN, locked
  clocks, fan 255; everything restored afterwards. Source and engine hashes were unchanged before vs after
  (logs/state.json). No engine builds.
- **Harness.** experiments/k35_gate_20261025, an isolated copy of the R3 tradeoff harness. md5:
  - build.py 9dbe9e2e;
  - scripts/86_k35_gate.py c1f39172;
  - compare.py a0740d8b;
  - window.py 2b23551c;
  - report.py ded2fcae;
  - staged.py 244ea468 (= RUNTIME_20261023).
- **Rows.** Same process, rotated order, independent histories, trt32.
  - v6-S (frozen).
  - B2K-A.
  - v7a = B2K-A + C1 retrospective recovery.
  - v7b = v7a + image-point fallback + critical-extent retention. This is VUW R3 low `v6S_b2A_missing_extent_retro`,
    composed from the same Jetson '_missing' / '_extent' components that R3 timed separately. No retuning.
- **Runtime (K35 timing note).** Every timed row uses the exact recovery bookkeeping helper (FAST_RECOVERY_20261023)
  and crop-buffer staging (RUNTIME_20261023).
- **Frames.**
  - nuScenes val: 13 scenes.
  - Verified AV2 CFG-DEV manifest: 8 logs, ego mask 1830.
  - First 712 frames, 200 warm-up, 512 timed. Three repetitions, so 1,536 timed frames per row.

## Action parity vs the reference runtime (KF.Crops / KB.OneCall + original Retro helper)
All 4 rows × 3 repetitions × 512 timed frames, both datasets:
- **0 differing actions** (parity rate 1.000);
- 0 differences in J, px, detector calls, crop-engine executions, and recovery calls / attempts / successes / pixels.

The 10-frame dry run and the 80-frame smoke also matched exactly. Files: parity_<ds>_rows.csv,
parity_<ds>_frames.csv (empty = no differing frame), parity_<ds>.json, dry/.

## Latency (ms, upload → decision with refiner; 3 × 512 timed frames pooled)

| dataset | row | mean | median | p95 | p99 | >50 ms | >66.7 ms | >100 ms |
|---|---|---|---|---|---|---|---|---|
| nuScenes val | v6-S | 26.00 | 20.79 | **62.75** | 82.39 | 10.5% | 3.2% | 0.2% |
| nuScenes val | B2K-A | 23.81 | 21.01 | 48.67 | 56.40 | 4.4% | 0.2% | 0.0% |
| nuScenes val | v7a | 24.33 | 21.89 | 50.88 | 59.10 | 5.5% | 0.6% | 0.0% |
| nuScenes val | v7b | 29.63 | 28.16 | **65.26** | 89.05 | 16.9% | 4.8% | 0.6% |
| AV2 CFG-DEV | v6-S | 38.80 | 29.60 | **117.52** | 147.03 | 24.7% | 17.9% | 7.7% |
| AV2 CFG-DEV | B2K-A | 34.63 | 30.49 | 80.07 | 104.00 | 25.7% | 12.0% | 1.5% |
| AV2 CFG-DEV | v7a | 35.93 | 30.58 | 87.23 | 110.64 | 27.6% | 13.7% | 2.1% |
| AV2 CFG-DEV | v7b | 45.90 | 35.68 | **115.18** | 162.91 | 37.3% | 26.9% | 10.0% |

Per-repetition p95 for v7b vs v6-S on nuScenes: 65.70 / 64.71 / 64.18 vs 64.21 / 62.58 / 62.31 (gate_rows.csv).

**Differences vs v6-S** (ms). Paired unit bootstrap: 2,000 draws, seed 0, repetitions nested in units; 13 nuScenes / 8
AV2 units.

| dataset | row | Δp95 [95% CI] | Δmean [95% CI] |
|---|---|---|---|
| nuScenes val | B2K-A | −14.08 [−23.30, −2.13] | −2.19 [−4.86, −0.42] |
| nuScenes val | v7a | −11.87 [−21.33, −0.00] | −1.67 [−4.30, +0.14] |
| nuScenes val | v7b | +2.51 [−4.59, +16.65] | +3.64 [+0.89, +6.72] |
| AV2 CFG-DEV | B2K-A | −37.45 [−41.63, −0.95] | −4.17 [−9.28, −0.31] |
| AV2 CFG-DEV | v7a | −30.29 [−35.46, +3.32] | −2.87 [−6.97, +0.70] |
| AV2 CFG-DEV | v7b | −2.34 [−4.66, +21.92] | +7.10 [+4.15, +10.64] |

## Reading
- v7b's point fallback + extent retention costs **+3.6 / +7.1 ms mean** over v6-S (+5.3 / +10.0 ms over v7a). That
  removes most of B2K-A's tail-latency benefit.
- The nuScenes p95 excess is small, and its interval includes 0. The gate is the registered point-estimate rule,
  applied as written, without re-deciding.
- v7a keeps a p95 below v6-S's on both datasets (−11.9 / −30.3 ms); this is reported, not gated.
- Dev prefixes only. This is a runtime screen: not a quality evaluation, and not a confirmatory test.

## Files
- gate_rows.csv, gate_diffs.csv, gate.json: tables and the gate decision.
- parity_* and dry/: parity checks.
- protocol.md: registered before the window.
- Raw runs: results/raw/20261025_13*_k35_gate_* on the kit.
