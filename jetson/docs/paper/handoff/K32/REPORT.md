# K32 REPORT — closed-loop AEB in CARLA / Bench2Drive with estimated states

**Status: FINAL 2026-10-25.**
- All 376 registered jobs finished on 2026-10-22: 312 test jobs + 48 uncertainty-branch jobs (amendment G1) + 16 shadow
  rollouts (G4). This gives 6,975 test route runs and 310 shadow routes.
- Final audit: AUDIT_20261022_FINAL.md. Every number below comes from the final tables in auto/tables/ (generated
  2026-10-22 15:43 VUW by k32_test.py, k32_shadow.py and k32_autoreport.py); auto/REPORT_auto_test.md is the full dump.
- Pre-registration and every amendment: STATUS.md.
- Statistics: paired bootstrap, 2,000 resamples, seed 0. **Bold** = the 95% interval excludes 0. Sign: v6-S − branch.

## 1. Gate history (dev routes, 95 Bench2Drive routes; pre-registered gate (b) privileged collisions ≤ 5%, (c) cheap measurably different from privileged)

| step | what changed (dated amendment, written before the affected run) | outcome |
|---|---|---|
| dev (registered) | — | (b) FAILED: privileged-AEB collided on 41% of routes |
| A (user-approved) | A1 standstill hold; A2 refiner and C1 off for every branch (rule (a)), so K32 tests "selection with the geometric lift", not full v6-S | dev2: 21%; (b) FAILED |
| B / B2 | range from the front bumper; privileged in-camera radius 80 m | — |
| C | box-growth TTC over 10 ticks (0.5 s; 20 Hz growth TTC produced phantom HARD braking) | dev3 (A1 + A2 + B + B2 + C): collisions 20% |
| D (user's choice) | (b) counts only collisions preventable by longitudinal control (side / rear contacts or demand lead < v / 6 + 0.4 s are non-preventable) | (b) PASSED: 1 / 95 preventable |
| F (user's choice) | (c) on any registered K32 metric | registered (c) metrics n.s. (collision −0.054, min TTC +0.05 s); missed required episodes +0.684 [0.463, 0.916] and unnecessary hard onsets +12.24 [8.56, 16.64] per route; **GATE PASSED** |
| E | E1 observational decision-change outcomes; E2 replay forks impossible (CARLA not deterministic: identical runs diverge from tick 13), so E2' model-based forks | analysis only |
| G (after the test started, after an interim look; disclosed) | G1 uncertainty-crop branch (unc); G2 latency model for unc; G3 primary metric = onset delay vs privileged on avoidable events, paired route × seed; speed stratum ≥ / < 8 m/s; G4 shadow mode | G3 was registered after an interim look, in which v6-S's onset delay vs privileged was +0.54 s n.s.; no favourable signal |

Two repairs were made after the gate.
- **Route matching (2026-10-21).** Log lines had been matched to routes by the nearest start point, but 30 test and 9
  dev route pairs start < 30 m apart. They are now matched by each chunk's run order.
  - All 3,600 lines checked match their checkpoint records; the old matcher had 187 wrong or duplicate assignments.
  - With the new matcher the dev gate is unchanged (numbers above).
  - The interim tables of 2026-10-20 are superseded.
- **Infrastructure.** Six simulator crashes / hangs were resumed. Six repaired route units are merged with checksum
  provenance (repair/). No outcome-based exclusion was applied.

## 2. Test results (155 routes × 3 traffic seeds; L0 = no latency, L1 = measured Jetson latency; privileged L0 only)

Per branch (unit = route, seeds averaged):

| branch | setting | collision | preventable | min TTC (s) | missed required episodes / route | unnecessary hard onsets / route | unnecessary HB episodes / km | RC | px | latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| privileged | L0 | 0.312 | 0.002 | 1.617 | 0.000 | 0.000 | 0.0 | 92.1 | – | – |
| cheap | L0 | 0.303 | 0.015 | 1.738 | 0.441 | 4.25 | 49.8 | 88.1 | 0.043 | 0 |
| u960 | L0 | 0.303 | 0.011 | 1.714 | 0.329 | 3.97 | 49.5 | 88.4 | 0.384 | 0 |
| native | L0 | 0.316 | 0.013 | 1.711 | 0.316 | 4.77 | 51.1 | 89.6 | 1.031 | 0 |
| hybrid | L0 | 0.320 | 0.017 | 1.722 | 0.327 | 2.99 | 40.4 | 89.3 | 0.081 | 0 |
| unc | L0 | 0.314 | 0.019 | 1.712 | 0.430 | 2.39 | 38.2 | 88.8 | 0.074 | 0 |
| **v6-S** | L0 | 0.301 | 0.006 | 1.785 | 0.295 | 2.86 | 39.5 | 89.2 | 0.069 | 0 |
| v6-S ≤ 2 | L0 | 0.310 | 0.019 | 1.774 | 0.323 | 2.91 | 40.7 | 89.5 | 0.066 | 0 |
| cheap | L1 | 0.331 | 0.032 | 1.663 | 0.641 | 4.47 | 76.6 | 89.7 | 0.043 | 7.5 |
| u960 | L1 | 0.323 | 0.013 | 1.619 | 0.432 | 4.02 | 55.3 | 89.3 | 0.384 | 22.7 |
| native | L1 | 0.325 | 0.028 | 1.659 | 0.454 | 4.29 | 78.5 | 88.3 | 1.031 | 37.9 |
| hybrid | L1 | 0.344 | 0.022 | 1.587 | 0.467 | 2.76 | 41.6 | 90.3 | 0.080 | 24.1 |
| unc | L1 | 0.331 | 0.026 | 1.560 | 0.617 | 2.33 | 42.6 | 90.3 | 0.072 | 16.1 |
| **v6-S** | L1 | 0.338 | 0.013 | 1.572 | 0.475 | 2.27 | 58.9 | 90.8 | 0.068 | 21.8 |
| v6-S ≤ 2 | L1 | 0.348 | 0.019 | 1.563 | 0.445 | 2.75 | 43.0 | 91.1 | 0.065 | 18.3 |

**Primary (G3): braking-onset delay vs privileged-AEB on avoidable events** (s; unit = route × seed; ~430 units and
3,650–5,150 events per row).

| setting | v6-S | v6-S − cheap | v6-S − u960 | v6-S − native | v6-S − hybrid | v6-S − unc | v6-S − v6-S ≤ 2 |
|---|---|---|---|---|---|---|---|
| L0 | 0.040 [0.031, 0.053] | **−0.050 [−0.069, −0.031]** | **+0.011 [+0.003, +0.024]** | **+0.011 [+0.002, +0.024]** | −0.000 [−0.011, +0.014] | **−0.036 [−0.051, −0.019]** | −0.001 [−0.011, +0.012] |
| L1 | 0.084 [0.075, 0.093] | **−0.039 [−0.057, −0.023]** | **+0.026 [+0.018, +0.033]** | **+0.026 [+0.018, +0.035]** | +0.005 [−0.004, +0.014] | **−0.046 [−0.063, −0.031]** | +0.006 [−0.004, +0.015] |

- v6-S brakes earlier than cheap and unc, ties hybrid and v6-S ≤ 2, and brakes 11–26 ms later than the whole-frame rows.
- Speed strata (≥ / < 8 m/s) and miss rates: auto/tables/g3_primary.csv and g3_primary_intervals.csv.

**Secondary (route × seed bootstrap).**
- Avoidable (D-preventable) collisions: v6-S − cheap −0.019 [−0.037, −0.002] at L1; every other contrast is n.s.
  (0.000 … −0.015 vs the perception rows).
- Unnecessary hard-brake episodes per km at L0:
  - v6-S has fewer than cheap (−10.3 [−16.0, −5.6]), native (−11.6 [−17.6, −6.8]) and u960 (−10.0 [−20.6, −1.2]);
  - it is n.s. vs hybrid, unc and v6-S ≤ 2.
- Min TTC: n.s. vs every perception row. Per-route-run quantiles: min_ttc_quantiles.

**Collisions are not discriminative.** Every branch, privileged-AEB included, collides on 30–35% of routes, almost all
non-preventable (side / rear impacts by actors outside the corridor, or demand too late to stop). Preventable rates are
0.2–3.2%. Contact types: auto/tables/test_collisions.csv.

**E1 (decision changes vs the draft, outcome over 3 s).**
- v6-S's changes are right more often than wrong: 12,620 correct vs 5,899 wrong at L0 (2.1 : 1), 13,093 vs 5,461 at L1
  (2.4 : 1).
- Front contacts within 3 s are 0.020 / 0.018 after correct changes vs 0.036 / 0.036 after wrong ones (L0 / L1).

**E2' (model-based forks, v6-S's actions vs the draft's, 3 s).**
- v6-S avoids a contact that the draft's actions would have in 0.9% / 1.3% of decision-change events, and causes one in
  0.3% / 0.4%.
- At L0 it does so with less braking than the draft (−0.016 m/s); u960 spends +0.52 m/s. All rows: e2_forks.csv,
  e2_intervals.csv.

**Shadow mode (G4)**, on the frames of the v6-S and u960 rollouts:
- per-frame agreement with u960 + comp.: v6-S 0.944 / 0.948, hybrid 0.943 / 0.947, native 0.942, unc 0.921 / 0.931,
  cheap 0.897 / 0.907;
- onset delay vs u960: v6-S 0.029 / 0.107 s, cheap 0.047 / 0.140 s;
- full table: shadow_fidelity.csv, shadow_intervals.csv.

**Latency model (G2, used for unc only):** RMSE 8.0 ms. Modelled vs measured p95: v6-S 61.7 vs 59.1 ms, hybrid 66.8 vs
74.9 ms (latency_model_validation.csv).

## 3. Every registered metric where v6-S is not better, with the mechanism

Significant (route bootstrap), from auto/REPORT_auto_test.md §4 and the G3 table:

| setting | vs | metric | v6-S − branch |
|---|---|---|---|
| L0 | cheap | §4 onset delay after the scenario trigger | +0.415 s [+0.115, +0.836] |
| L0 | native | §4 onset delay | +0.215 s [+0.068, +0.387] |
| L0 | u960 | §4 onset delay | +0.205 s [+0.060, +0.377] |
| L0 | u960 / native | G3 onset delay vs privileged | +0.011 s [+0.003, +0.024] / +0.011 s [+0.002, +0.024] |
| L0 | unc | unnecessary hard onsets / route | +0.47 [+0.19, +0.75] |
| L1 | native / u960 | §4 onset delay | +0.317 s [+0.111, +0.570] / +0.401 s [+0.164, +0.694] |
| L1 | unc / v6-S ≤ 2 | §4 onset delay | +0.241 s [+0.006, +0.547] / +0.204 s [+0.021, +0.451] |
| L1 | u960 / native | G3 onset delay vs privileged | +0.026 s [+0.018, +0.033] / +0.026 s [+0.018, +0.035] |
| L0 / L1 | privileged | missed episodes, unnecessary hard onsets, unnecessary HB / km, onset delay (L1), flicker (L0) | all positive (expected: privileged sees true states) |

Not significant but worse in point estimate: L1 unnecessary HB per km (v6-S 58.9 vs hybrid 41.6, unc 42.6, v6-S ≤ 2
43.0).

**Mechanism.**
- **Later onsets than the whole frame.** These are the same mechanism the dataset analyses found (K31 item 1, K34-1,
  K35-1):
  - v6-S verifies an object only when the draft's support or the field points at it;
  - its looming / TTC cue needs a same-fidelity chain;
  - whole-frame rows see a newly entering actor at full resolution from its first frame and brake a few ticks earlier.
  K32 runs v6-S without refiner and without C1 (rule (a)), so here the delay comes from selection and the box-growth
  TTC, not from the refiner. The gap is 11 ms (L0) to 26 ms (L1) relative to privileged on avoidable events: about
  0.5 tick.
- **The larger §4 delay vs cheap at L0 (+0.415 s).** cheap starts braking earliest after the trigger because it also
  brakes for phantoms. Its unnecessary hard onsets are 4.25 per route vs v6-S's 2.86. The §4 onset metric measures from
  the scenario trigger, not from the true demand, which is why G3 was introduced. On G3 v6-S is 0.050 s earlier than
  cheap.
- **L1 unnecessary braking per km (n.s.).** At L1, v6-S's crop rounds add latency (21.8 ms mean). Stale applied actions
  keep a hard brake one tick longer after the demand ends, which raises episodes per km on short routes.
  - 185 route runs drove < 1 m; distance-normalized metrics are extreme there.
  - The per-route counts are lower for v6-S: unnecessary hard onsets 2.27 vs hybrid 2.76 and v6-S ≤ 2 2.75.
- **vs unc at L0 (unnecessary hard onsets +0.47).** unc verifies the lowest-confidence draft boxes. It removes more
  phantom hard brakes, but misses more required episodes (0.430 vs 0.295 per route) and is 0.036 s later on G3.

## 4. Files

- auto/tables/:
  - test_per_route.csv, test_summary.csv, test_intervals.csv, test_intervals_route_seed.csv, test_min_ttc_quantiles.csv,
    test_collisions.csv;
  - g3_primary.csv, g3_primary_intervals.csv, g3_events.csv;
  - e1_events.csv, e1_episodes.csv, e1_intervals.csv;
  - e2_forks.csv, e2_intervals.csv;
  - shadow_fidelity.csv, shadow_intervals.csv;
  - dev gate tables: dev3_*.csv, dev3F_*.csv.
- auto/REPORT_auto_test.md: every table in full.
- AUDIT_20261022_FINAL.md, repair/, health/: data audit, repairs and checksums.
- latency_model_validation.csv, latency_samples.csv.
