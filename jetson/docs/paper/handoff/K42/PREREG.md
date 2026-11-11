# K42 — pre-registration (written 2026-11-11, before any K42 number is computed)

Scope and constraints:
- Post-hoc. Frozen v6-S (DGSP in the final manuscript = DESP v6-S) and every baseline configuration are unchanged; no
  tuning, no threshold change. The K35-4 holdout is not touched.
- Test splits already used: nuScenes new test (228 scenes), AV2 val with the K10 ego mask (150 logs), Argoverse-HD val
  (24 sequences).

Statistics (as K39), unless stated otherwise:
- Paired unit bootstrap, 2,000 draws, seed 0, 95% percentile interval; units = scenes / logs / sequences.
- Δ = DGSP − row. For G, Δ > 0 means DGSP is better; for missed rate and false episodes, Δ < 0 means DGSP is better.
- Where an item reproduces a published table, it uses that table's own resampling code (named per item), so the gate
  can be exact.
- Every row is reported; no row is dropped after seeing a number. Amendments go in a dated §A at the end, written
  before the computation they concern.

## Source check done before writing this (column names only, no values read)

The K28 phase-3 run `results/k28/p3/{ds}_test_basewf_y11s` (k28_p3.py `--part basewf --ver yolo11s --draft yolo11s
--tag y11s`) already stores, for every whole-frame side, `a/J_wf{s}+C1|none` and `a/J_wf{s}+C1|v3a`. In that code
(k28_p3.py, part basewf), `wf{s}+C1` is the C1 lift `lift_c("wf{s}", "C1", scale)` on the row's own same-scale history
`st["wf{s}"]`, with the YOLO11s detector, scored with refiner none and v3a. This is exactly the pipeline asked for in
K42-1. **K42-1 therefore needs no new GPU run**: it is computed from these stored records. A GPU re-run would
recompute the same records.

## K42-1 (CPU, stored records) — YOLO11s whole frame with the full lift

- DGSP_y11s = `J_v6S|v3a` of `{ds}_test_ours_y11s` (refiner v3a + C1, as in v6-S).
- Rows: whole frame YOLO11s at u512, u640, u960, u1280 and native (1600 / 2048 / 1920), each in two forms:
  - "full lift" (new): `J_wf{s}+C1|v3a` (C1 on its own same-scale history + refiner v3a);
  - "refiner only" (old, as printed in supp tab:det2): `J_wf{s}|v3a`.
- Reported per (dataset, side, form): G, tensor px (mean), Δ = DGSP_y11s − row with 95% interval, verdict.
- Resampling: k28_p3_analyze.ctable's own (k24_analyze.Boot over the merged ours / basewf / baseheur y11s records), so
  that the gate is exact. The new column is computed in the same call.
- **Gate:** the refiner-only rows (G_refiner, v6S_minus_refiner, lo / hi) and the full-lift rows must equal
  `handoff/K28/phase3/ctable_y11s.csv` to 1e-12 on every (dataset, row). If they do not, K42-1 stops and is reported.
- Output: `k42_1_y11s.csv`.

## K42-2 (CPU, stored actions) — missed / false episodes for every row of main Tab. 2

- Code: K31's `k31_episodes.py`, imported unchanged (`runs`, `unit_stats`, `analyse`; level "required (>= DECELERATE)" =
  L 1 as primary, deployed threshold 1.0 = the stored actions, merge gap 1 frame), with the pipeline list extended. The
  hard-brake level (L 2) is computed by the same call and reported as secondary.
- Population: K28 p3 `{ds}_test_instr` GT frames (nuScenes 9,181, AV2 11,818, Argoverse-HD 15,006); every added source
  is merged 1:1 on (unit, frame) and must cover every GT frame (assertion).
- Rows, all with refiner v3a, each with its own stored history:
  - the 7 K31 rows (cheap, u960 / native + C1 + v3a, tsp8@S, unc2@S, v6-S, v6-S without refiner);
  - whole frame u512, u640, u1280 + C1 + v3a: `a_wf{s}+C1|v3a` of K28 `{ds}_test_v6basewf-yolo11m` (its YOLOv8s
    columns, as K39-6);
  - AdaScale, SAHI, FOVEA KDE_I, UFP: K33 `{ds}_pub_test`, `a_{key}|v3a`;
  - AnA iterated, counterfactual iterated: K33 `{ds}_de_test`, `a_{key}|v3a`;
  - BAR router, router-G, Chanakya: action derived from the stored per-frame routing size (K33 rc, k33_rc's rule): a
    routed frame takes `a_wf{s}|v3a` of K33 pub, an unrouted frame `a_cheap`. **Gate (per row):** recomputing the stored
    refined J from the same rule must reproduce `J_{row}` on every frame; otherwise the row is reported as missing.
  - For the published / object-level / router rows, the uncapped run (Tab. 2's G and px column) is primary; the @cap
    run is reported as secondary.
  - Secondary, for completeness (Tab. 2 rows not in the task list): FOVEA KDE_S, AnA one pass, counterfactual one pass,
    and the dev-best uncertainty / temporal-criticality / ego-corridor @S + C1 + v3a of the K39-2 rule (nuScenes unc4 /
    tcr3 / cor0; AV2, Argoverse-HD unc2 / tcr0 / cor8).
- Reported: missed_rate, false_per_1000 (plus every other K31 statistic), and Δ = DGSP − row with 95% interval, by K31's
  own bootstrap (`analyse`: sorted units, default_rng(0), 2,000 draws).
- **Gate:** the 7 K31 rows of ep_summary.csv and their ep_intervals.csv rows are reproduced exactly (every column, both
  levels, 3 datasets). If not, K42-2 stops and is reported.
- Output: `k42_2_episodes.csv` (+ `k42_2_intervals.csv` with every statistic).

## K42-3 (CPU) — fidelity with the refiner, packaged

- `k42_3_fidelity_v3a.py` = `k39_6_fidelity.py` (md5 261a6308…) with every `|none` replaced by `|v3a` and nothing else,
  which is the change the writing session made. π(F) becomes `a_wf960+C1|v3a`. Output CSVs are written to
  handoff/K42/ and their md5s are listed in REPORT.md for comparison with the Mac copy (fid_v3a/); the Mac files are
  not on this machine, so the match is checked by the writer from the listed md5s.
- Router rows (BAR, router-G, Chanakya): action from `a_wf{s}|v3a` when routed, else `a_cheap`; gate = the stored refined
  J reproduced on every frame (the script's own check). Reported separately.
- Oracle-locality and finer-probe agreement with the refiner on both sides: the K39 single interventions (`a_sa … a_sf`,
  results/k39/vuw) and the K28 chain (`a_ch_a … a_ch_f`) are stored without a refiner variant. If the column check
  confirms that no refined variant exists, this is reported as "not available from stored data" with what a re-run
  would need; it is not run.

## K42-4 (CPU) — coverage on R and the 95% stopping test

- (a) Population and rule of K34-2 (one decisive annotated object per frame with act_gt ≥ DECELERATE; box-centre
  inside a crop). Region = field + support crops only = the first `ncrops_v6S` entries of the K30 match `crops_v6S`
  (k30_eval.py stores `crops + grown`, with `ncrops_v6S = len(crops)`).
  - Checks: `len(crops_v6S) = ncrops_v6S + nguard_v6S` on every frame; the all-crop coverage recomputed equals K34's
    `coverage_frames.csv` on every frame.
  - Reported: covered share (field + support) next to all crops, the difference, 95% intervals (paired unit bootstrap
    of ratios of per-unit sums, sorted units, default_rng(0).integers, 2,000).
- (b) The stopping test `covered(crop, crops, tol=0.95)` (vtf/speculative.py) skips a candidate crop when one earlier
  crop covers ≥ 95% of its area. The share of frames where it skips a crop that containment in the union would require,
  and the actions that change under strict containment, need the candidate crops of every round. These are not stored
  and crop detections are not cached (K34's replay caches only cheap and whole-frame detections), so (b) needs a GPU
  re-run. Per the task it is reported as such and not run.

## K42-5 (Jetson, exclusive window only) — energy of u512 + comp.

- Protocol J2 of K30 (handoff/K30/REPORT.md §J2): MAXN, jetson_clocks, fan 255, module rails GPU + CPU + SOC + CV +
  VDDRQ as 1 s means, integrated between first and last timed frame; energy above the cheap run; nuScenes new test 20
  scenes and AV2 val 8 logs (ego mask); one run per pipeline; no intervals (as J2).
- Row: whole frame u512 + C1 + v3a. Run in the same window as references: cheap, u448 + comp. and v6-S (to check that
  the window reproduces J2's values within run-to-run noise). Only if an exclusive window is available on this Jetson.

## §A Amendments

### A1 (2026-11-11 13:40, after the K42-2 gate check and before any K42-2 added-row value was read)

- K42-2 gate: the 7 K31 rows reproduce ep_summary.csv / ep_intervals.csv to a maximum absolute difference of 7.1e-15
  (every column, both levels, 3 datasets; NaN pattern identical; no row missing). The script required exactly 0 and
  stopped before writing any table. The tolerance is set to 1e-12, as for K42-1; floating-point summation differences
  of this size do not change any printed digit. The script is re-run with that tolerance and nothing else changed.
- K42-5: the K30 J2 window (2026-10-18 07:24 and 07:38, results/raw/20261018_072404_k30j_timing_nusc_test20_v6S_j2_u512
  and 20261018_073816_..._av2_v6S_j2_u512) already ran `--only u512+c1comp` (u512 + C1 + v3a, trt32 fast path) in the
  same window and protocol as u960 / native / v6-S; its row is in handoff/K30/J/j2_energy_marginal.csv but was not quoted
  in the K30 report text. K42-5 therefore reports that stored row; no new Jetson window is run.

### A2 (2026-11-11, after the user's go-ahead "run it, make it complete"; written before the GPU run)

The two parts that need a GPU are now run. Harness: experiments/k42_vuw_20261111/k42_eval.py, built by build.py as an
isolated copy of the K39 harness (k39_eval.py, md5 b8608ba3) with a new part `k42`. Nothing else is changed; frozen code
is not touched. Test units as K39 (nuScenes 228, AV2 150 with the ego mask, Argoverse-HD 24). Per frame:

- **K42-4(b).**
  - `v6S`: frozen v6-S. Every call of the stopping test `SP.covered(crop, crops)` is logged with its decision and the
    exact share of the candidate's area inside the union of the earlier crops (coordinate compression).
  - `v6U`: v6-S with the stopping test replaced by exact containment in the union (skip iff 100% of the candidate's area
    is inside the union of earlier crops). It has its own history and is scored with refiner none and v3a.
  - Reported:
    - the share of frames where the 95% test skipped at least one candidate that union containment would require
      (primary, as asked);
    - the reverse direction (executed now, union would skip);
    - the frames and units whose action differs between v6S and v6U (none / v3a);
    - G of v6U and Δ = v6S − v6U (paired unit bootstrap, 2,000, seed 0);
    - px of both.
- **K42-3, second part.** π(F) = u960 + C1, the K39-4 single interventions sa … sf and the K28 / K16 counterfactual chain
  ch_a … ch_f (k28_p3.py CHAIN, detect_F / lift_F) are each scored with refiner none and v3a. Agreement with π(F) is
  reported with the refiner on both sides (a_X|v3a vs a_wf960+C1|v3a), next to the stored no-refiner values. The
  table follows k39_4_single.csv: alone shares of D0 and the cumulative chain agreement.
- **Parity gates.** Without the refiner, the recomputed actions must equal the stored ones on every frame:
  - a_v6S|none, a_v6S|v3a and a_wf960+C1|none / v3a vs results/k39/vuw;
  - a_sa … a_sf|none vs results/k39/vuw;
  - a_ch_x|none vs K28 p3 instr a_ch_x.

  A frame that differs is reported. If the v6-S, π(F) or chain parity fails, the item is reported as failed rather than
  used.
- Smoke test first on one nuScenes scene and one AV2 log, with the parity checked, before the full run.
