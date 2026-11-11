# K35-8 — absolute selector costs of R4's exact score reuse (no new run)

**Output-identical implementation change.** The K31 bottleneck selector caches the base scene's per-candidate scores
s(S) and evaluates only the probe-only columns, since s(S + probe) = min(s(S), h(probe)). The candidate trajectories,
probe grids, tie rule and binding support are unchanged.
- All 6,000 original-vs-cached comparisons have identical action, field and support (parity.csv: 1,000 NAVSIM navtest
  tokens × 2 grids × 3 repetitions).
- This changes runtime only. No quality number changes, and it is not a PDM-Closed or end-to-end perception speedup.

**Source:** copied unchanged from handoff/OPTIMIZATION_R4_20261024/nav_selector/.
- Measurement: one CPU thread on VUW1; five warm-ups, then three timed repetitions with alternating order.
- Scope: scene construction, selector initialization, probe projection, field and support. Disk loading and detector
  inference are excluded.
- Statistics: paired unit (log) bootstrap, 2,000 draws, seed 0, repetitions nested in units, pooled p95 recomputed per
  draw. These are post-hoc runtime / parity measurements on the fixed K34 / K20 token set (125 units).

| grid | implementation | mean ms per frame [95% CI] | p95 ms [95% CI] | selector-stage mean ms [95% CI] |
|---|---|---|---|---|
| 7x7 | original | 100.969 [85.612, 118.067] | 309.998 [257.361, 365.918] | 94.132 [79.021, 110.923] |
| 7x7 | cached (exact reuse) | 10.957 [10.310, 11.709] | 18.964 [16.785, 20.582] | 4.166 [3.818, 4.555] |
| 3x3 | original | 24.112 [20.580, 28.077] | 70.933 [57.701, 84.645] | 20.449 [17.149, 24.116] |
| 3x3 | cached (exact reuse) | 6.392 [5.797, 7.054] | 13.643 [11.729, 15.278] | 2.660 [2.316, 3.042] |

Paired differences, cached − original:

| grid | mean total ms | p95 ms | speedup (mean) |
|---|---|---|---|
| 7x7 | −90.01 [−106.40, −75.25] | −291.03 [−345.22, −240.64] | 9.21× [8.25, 10.18] |
| 3x3 | −17.72 [−20.99, −14.76] | −57.29 [−69.47, −45.98] | 3.77× [3.53, 3.99] |

**Suggested supplement sentence:** "An output-identical implementation of the selector, which caches each candidate's
base score and evaluates only the probe columns, reduces its per-frame CPU cost from 101.0 to 11.0 ms (7×7 grid; 24.1 to
6.4 ms for 3×3), with identical actions, fields and supports on all 1,000 navtest tokens."

**Files** (copied, md5 of summary.csv df35e0e8, contrasts.csv af7fa137):
- summary.csv, contrasts.csv, frames.csv (12,000 timed rows), parity.csv, token_manifest.csv;
- cost.pdf / .png.
A second, two-host measurement of the same change is in handoff/NAVSIM_REUSE_20261023/ (7x7 115.9 → 13.0 ms mean).
