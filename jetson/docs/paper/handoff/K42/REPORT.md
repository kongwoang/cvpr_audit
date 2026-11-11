# K42 — the remaining numbers for the final version (post hoc, frozen v6-S, no configuration change)

2026-11-11. Every item is a **post-hoc analysis** on test splits that were already used. PREREG.md was written before any
number was computed. Its amendment A1, made before any added row was read, does two things:
- it sets the K42-2 gate tolerance to 1e-12;
- it records that K42-5 already exists in K30 J2.

Nothing new was run on the GPU or the Jetson. K42-1 and K42-5 turned out to be stored already, and K42-3 (second part)
and K42-4(b) would need GPU runs, which were not done (see each item). The holdout was not touched.

How the numbers were computed:
- Statistics follow K39: paired unit bootstrap, 2,000 draws, seed 0, 95% percentile intervals, Δ = DGSP − row.
- DGSP is the manuscript's name for DESP v6-S.
- Where an item reproduces a published table, it uses that table's own resampling code, so its gate can be exact.

Code: `code/k42_*.py` (also at VUW `scripts/A_method/`). md5 values:

| script | md5 |
|---|---|
| k42_1_y11s.py | b58f71c0… |
| k42_2_episodes.py | 08ac7433… |
| k42_3_fidelity_v3a.py | 33935f5d… |
| k42_4_coverage.py | 4e84be61… |

They import k28_p3_analyze.py (bd7035fd…) and k31_episodes.py (73f07c9d…) unchanged. Outputs are in VUW
`results/k42/k{1..4}/` and are copied here.

Command, on VUW CPU after `source env.sh; export B=$PWD PYTHONPATH=$B/scripts/A_method`:
- `python scripts/A_method/k42_1_y11s.py results/k42/k1 <handoff ctable_y11s.csv>`
- `python scripts/A_method/k42_2_episodes.py results/k42/k2`
- `python scripts/A_method/k42_3_fidelity_v3a.py results/k42/k3`
- `python scripts/A_method/k42_4_coverage.py results/k42/k4`

---

## K42-1 — YOLO11s whole frame with the full lift (k42_1_y11s.csv)

**No new GPU run was needed.**
- The K28 phase-3 run `{ds}_test_basewf_y11s` (k28_p3.py `--part basewf --ver yolo11s --draft yolo11s --tag y11s`)
  already stores `wf{s}+C1|v3a` for every side.
- That row is YOLO11s whole frame, plus the C1 lift `lift_c("wf{s}")` on the row's own same-scale history, plus refiner
  v3a. This is the pipeline the task asked for.
- Supp tab:det2 printed only the `wf{s}|v3a` rows (refiner only).
- DGSP_y11s = `v6S|v3a` of `{ds}_test_ours_y11s`.

**Gate: passed.** k28_p3_analyze.ctable was re-run unchanged and reproduces all 147 rows of K28
`phase3/ctable_y11s.csv`, with a maximum difference of 1.1e-16 and identical verdicts (k42_1_gate.json).

The table shows G and Δ = DGSP_y11s − row [95% CI]. A negative Δ means the whole-frame row is better.

| dataset | side | px % | G full lift | Δ full lift | G refiner only (old) | Δ refiner only (old) |
|---|---|---:|---:|---|---:|---|
| nuScenes (DGSP 0.418, 7.7%) | u512 | 10.2 | 0.309 | +0.109 [−0.070, +0.234] | 0.255 | +0.164 [+0.010, +0.267] |
| | u640 | 17.1 | 0.379 | +0.040 [−0.078, +0.111] | 0.319 | +0.100 [+0.007, +0.153] |
| | u960 | 38.4 | 0.446 | −0.027 [−0.068, +0.018] | 0.408 | +0.011 [−0.037, +0.085] |
| | u1280 | 65.4 | **0.483** | **−0.064 [−0.114, −0.015]** | 0.434 | −0.015 [−0.072, +0.077] |
| | native | 103.1 | 0.443 | −0.025 [−0.082, +0.043] | 0.410 | +0.009 [−0.058, +0.102] |
| AV2 (DGSP 0.261, 5.4%) | u512 | 6.7 | 0.310 | −0.050 [−0.109, +0.005] | 0.133 | +0.128 [+0.070, +0.191] |
| | u640 | 10.3 | **0.316** | **−0.056 [−0.113, −0.005]** | 0.145 | +0.116 [+0.051, +0.182] |
| | u960 | 23.2 | **0.323** | **−0.062 [−0.123, −0.005]** | 0.158 | +0.103 [+0.039, +0.168] |
| | u1280 | 40.0 | 0.311 | −0.051 [−0.112, +0.008] | 0.145 | +0.116 [+0.054, +0.180] |
| | native | 101.2 | 0.260 | +0.000 [−0.094, +0.093] | 0.085 | +0.176 [+0.056, +0.298] |
| Argoverse-HD (DGSP 0.422, 6.9%) | u512 | 7.1 | 0.277 | +0.146 [+0.030, +0.337] | 0.082 | +0.341 [+0.265, +0.448] |
| | u640 | 11.6 | 0.300 | +0.122 [+0.015, +0.329] | 0.079 | +0.343 [+0.236, +0.491] |
| | u960 | 25.3 | 0.300 | +0.122 [+0.012, +0.329] | 0.111 | +0.311 [+0.199, +0.463] |
| | u1280 | 44.4 | 0.316 | +0.107 [+0.013, +0.288] | 0.101 | +0.321 [+0.196, +0.497] |
| | native | 101.3 | 0.412 | +0.011 [−0.025, +0.050] | 0.119 | +0.303 [+0.166, +0.477] |

**Conclusion (unfavourable on two of three datasets).** With the full lift, whole-frame YOLO11s does as well as or
better than DGSP on nuScenes and AV2:
- nuScenes u1280 is significantly better;
- AV2 u640 and u960 are significantly better, at 1.9× and 4.3× DGSP's pixels.

On Argoverse-HD, DGSP stays significantly better at every side except native (tie).

The supp sentence "significantly outperforms whole-frame inference with the refiner at every resolution on
Argoverse 2 and Argoverse-HD" holds only for the refiner-only rows. With the full lift it holds for Argoverse-HD only, up
to u1280. This is the same pattern as YOLOv8s (K28 phase-3 REPORT, S5).

---

## K42-2 — missed and false episodes for every row of main Tab. 2 (k42_2_episodes.csv)

**Source and method:**
- K31's `k31_episodes.py`, imported unchanged: level "required (≥ DECELERATE)", deployed actions, merge gap of 1 frame.
- The pipeline list is extended. All rows use refiner v3a and their own stored history.
- Population: the K28 p3 instr GT frames (9,181 / 11,818 / 15,006).
- The intervals are K31's own bootstrap.
- Sources of the added rows:
  - u512 / u640 / u1280 + C1: K28 v6basewf-yolo11m, YOLOv8s columns;
  - AdaScale, SAHI, FOVEA, UFP: K33 pub;
  - AnA and counterfactual: K33 de;
  - routers: K33 rc routing sizes. A routed frame takes `a_wf{s}|v3a`; otherwise it takes `a_cheap`.

**Gates:**
- **Passed.** The 7 K31 rows reproduce ep_summary.csv (48 rows) and ep_intervals.csv (288 rows) with a maximum
  difference of 7.1e-15, the same NaN pattern, and no row missing. Tolerance is 1e-12 (A1).
- **Router J: passed.** The stored refined J is reproduced from the routing sizes on every frame, for all 18 router
  rows (k42_2_checks.json).

The table covers the primary level. Rates are the missed rate (share of required episodes never braked) and false
episodes per 1,000 frames. `*` marks an interval that excludes 0. For the missed rate, Δ > 0 means DGSP misses more.

| row | nuScenes miss / Δ | nuScenes false / Δ | AV2 miss / Δ | AV2 false / Δ | A-HD miss / Δ | A-HD false / Δ |
|---|---|---|---|---|---|---|
| **DGSP (v6-S)** | 0.408 | 4.03 | 0.189 | 6.52 | 0.380 | 6.26 |
| cheap (K31) | 0.347 / +0.061 [−0.007, +0.133] | 22.87 / −18.84* | 0.111 / +0.079* | 34.78 / −28.26* | 0.169 / +0.211* | 25.12 / −18.86* |
| u512 + C1 + v3a | 0.361 / +0.048 [−0.014, +0.109] | 6.97 / −2.94* | 0.153 / +0.037 [+0.000, +0.076] | 8.97 / −2.45* | 0.408 / −0.028 | 5.60 / +0.67 |
| u640 + C1 + v3a | 0.347 / +0.061* | 7.30 / −3.27* | 0.153 / +0.037* | 9.31 / −2.79* | 0.451 / −0.070 [−0.135, +0.000] | 5.73 / +0.53 |
| u960 + C1 + v3a (K31) | 0.347 / +0.061* | 6.43 / −2.40* | 0.163 / +0.026 | 7.87 / −1.35* | 0.366 / +0.014 | 5.46 / +0.80 |
| u1280 + C1 + v3a | 0.333 / +0.075* | 6.43 / −2.40* | 0.147 / +0.042* | 7.78 / −1.27* | 0.338 / +0.042 | 6.26 / +0.00 |
| native + C1 + v3a (K31) | 0.361 / +0.048* | 4.90 / −0.87 | 0.153 / +0.037* | 8.63 / −2.12* | 0.380 / +0.000 | 7.00 / −0.73 |
| AdaScale | 0.347 / +0.061* | 6.43 / −2.40* | 0.168 / +0.021 | 8.12 / −1.61* | 0.338 / +0.042 | 6.20 / +0.07 |
| BAR router | 0.361 / +0.048 | 13.72 / −9.69* | 0.121 / +0.068* | 26.15 / −19.63* | 0.296 / +0.085 [+0.000, +0.159] | 15.26 / −9.00* |
| router-G | 0.306 / +0.102* | 14.05 / −10.02* | 0.137 / +0.053* | 26.82 / −20.31* | 0.169 / +0.211* | 25.19 / −18.93* |
| Chanakya | 0.401 / +0.007 | 13.40 / −9.37* | 0.142 / +0.047 [+0.000, +0.098] | 27.50 / −20.98* | 0.352 / +0.028 | 17.93 / −11.66* |
| SAHI | 0.333 / +0.075* | 11.87 / −7.84* | 0.132 / +0.058* | 16.16 / −9.65* | 0.296 / +0.085* | 11.20 / −4.93* |
| FOVEA KDE_I | 0.340 / +0.068* | 6.64 / −2.61* | 0.142 / +0.047* | 9.82 / −3.30* | 0.338 / +0.042 | 6.93 / −0.67 |
| UFP | 0.401 / +0.007 | 5.77 / −1.74* | 0.153 / +0.037* | 8.88 / −2.37* | 0.324 / +0.056 | 6.26 / +0.00 |
| AnA iterated | 0.388 / +0.020 | 4.90 / −0.87 [−1.85, +0.00] | 0.158 / +0.032* | 7.02 / −0.51 | 0.352 / +0.028 [+0.000, +0.067] | 5.93 / +0.33 |
| counterfactual iterated | 0.490 / −0.082* | 4.03 / +0.00 | 0.211 / −0.021 | 6.18 / +0.34 | 0.465 / −0.085 [−0.164, +0.000] | 6.13 / +0.13 |
| hybrid tsp8@S (K31) | 0.395 / +0.014 | 4.25 / −0.22 | 0.184 / +0.005 | 6.09 / +0.42 | 0.380 / +0.000 | 6.66 / −0.40 |
| uncertainty unc2@S (K31) | 0.429 / −0.020 | 7.19 / −3.16* | 0.163 / +0.026 | 14.89 / −8.38* | 0.394 / −0.014 | 15.86 / −9.60* |

Where the table gives no interval, the full interval is in k42_2_episodes.csv.

Secondary rows are in the same CSV:
- the @cap runs;
- FOVEA KDE_S, AnA one pass and counterfactual one pass;
- the dev-best unc / tcr / cor @S;
- the hard-brake level.

**Conclusion (unfavourable on missed episodes):**
- DGSP has significantly fewer false braking episodes than every whole-frame, published and router row on nuScenes and
  AV2. The exceptions, all not significant, are:
  - native on nuScenes;
  - AnA iterated, counterfactual iterated and the hybrid on both datasets.
- It misses more required-braking episodes than most whole-frame and published rows. The difference is significant
  against:
  - u640, u960, u1280, native, AdaScale, SAHI, FOVEA and router-G on nuScenes;
  - u640, u1280, native, SAHI, FOVEA, UFP, AnA, BAR and router-G on AV2.
- On Argoverse-HD, DGSP's missed rate is significantly higher only than cheap, router-G and SAHI, and its false
  episodes are not significantly different from the whole-frame rows.
- Among the primary rows, only counterfactual iterated has a significantly higher missed rate than DGSP (nuScenes).

---

## K42-3 — fidelity with the refiner, packaged (k42_3_fidelity.csv, k42_3_diffs.csv, k42_3_checks.json)

**Method.** `k42_3_fidelity_v3a.py` is `k39_6_fidelity.py` with every `|none` replaced by `|v3a` (20 lines). The only
other changes are the docstring and the output names, so π(F) = `a_wf960+C1|v3a`.

md5 of the CSVs, for comparison with the Mac copy `review/cvpr_final_2026-11-10/fid_v3a/`:
- `k42_3_fidelity.csv`: 50f5af1eefb768bd385d16bcda128010
- `k42_3_diffs.csv`: 2b197ce10577e1a209e9694cf0897c10

The Mac files are not on this machine, so the match was not checked here. If the Mac copy kept the names
`k39_6_*.csv`, compare by content: the names do not enter the CSVs.

**Checks (all 0 differing frames):**
- `wf960+C1|v3a` of basewf equals F;
- K33 pub `wf960|v3a` equals F (the label in checks.json still says "no refiner"; this is a text artefact of the
  substitution);
- **the three router rows, checked separately:** BAR, router-G and Chanakya, uncapped and @cap, × 3 datasets. Their
  action is taken from `a_wf{s}|v3a` when routed and from `a_cheap` otherwise. The stored refined J is reproduced on
  every frame in all 18 cases.

DGSP's agreement with π(F) under the refiner:

| dataset | A | A given cheap ≠ π(F) | A at F transitions | A given braking required |
|---|---|---|---|---|
| nuScenes | 0.980 | 0.788 | 0.773 | 0.903 |
| AV2 | 0.969 | 0.892 | 0.790 | 0.937 |
| A-HD | 0.942 | 0.836 | 0.637 | 0.940 |

**Where v6-S is NOT best.** Rows significantly above v6-S on overall agreement A:
- nuScenes and AV2: whole frame u960 (= F) and u1280 + C1, AnA one pass / iterated, and AdaScale;
- Argoverse-HD: u960 (= F) and AdaScale.

On the conditional metrics, more whole-frame and published rows are significantly above v6-S (k42_3_diffs.csv, column
`excludes_0` with a negative `v6S_minus_row`). As in K39-6, v6-S does this at 5–8% px, while those rows use 18–150%.

**Oracle locality and finer probes with the refiner on both sides: not available from stored data.**
- The K39 single interventions (`a_sa … a_sf`, results/k39/vuw) are stored only without a refiner.
- So is the K28 chain (`a_ch_a … a_ch_f`); the column check found no refined variant.
- The 99.5 / 99.6 / 97.1% agreements are therefore without the refiner on both sides.
- A refined version would need a re-run of the K39 VUW harness (part k39) with the s* and ch_* actions scored with v3a
  as well. That is a GPU run and was not done.

---

## K42-4 — coverage on R and the 95% stopping test

### (a) Field and support crops only (k42_4_coverage.csv, k42_4_coverage_frames.csv)

**Method.** The population and centre rule are K34-2's (one decisive object per required frame). K30 match stores
`crops_v6S = crops + grown` with `ncrops_v6S = len(crops)`, so the field and support crops are the first `ncrops_v6S`
entries.

**Checks (k42_4_checks.json):**
- `len = ncrops + nguard` on every frame;
- the all-crop coverage recomputed equals K34 on every frame;
- K34's interval is reproduced to 1e-16.

| dataset | required frames | all crops (K34) | field + support only | covered only by guard crops |
|---|---:|---|---|---|
| nuScenes | 928 | 0.918 [0.875, 0.948] | 0.897 [0.848, 0.931] | 0.022 [0.012, 0.034] |
| AV2 | 2,854 | 0.954 [0.939, 0.969] | 0.924 [0.902, 0.945] | 0.031 [0.020, 0.043] |
| Argoverse-HD | 3,593 | 0.959 [0.918, 0.984] | 0.911 [0.811, 0.972] | 0.048 [0.011, 0.108] |

**Conclusion.** On R (field + support crops only), the decisive object's centre is covered on 90–92% of required
frames. The guard crops add 2–5 points, which is how the current 92–96% is reached.

### (b) 95% stopping test versus containment in the union: needs a GPU, not run

**What the test does.** `covered(crop, crops, tol=0.95)` (vtf/speculative.py) skips a candidate crop when **one**
earlier crop covers ≥ 95% of its area. It takes the maximum over single crops, not the union.

**Why it was not run.** The frame share and the action changes need the candidate field and support crops of every
round, including the ones the test rejected:
- they are not stored;
- crop detections are not cached (K34's replay caches only cheap and whole-frame detections);
- the field crop of round r + 1 depends on the crop detections of round r.

**What it would take.** An instrumented v6-S re-run on VUW. It would log every rejected candidate with its single-crop
and union coverage (rate). A second loop with the strict rule would detect the extra crops (action change). This is
reported and not run, as the task asks.

**A note for the text.** The two rules are not nested:
- a candidate inside the union of several crops, but < 95% inside any single one, is executed now and would be skipped
  under union containment;
- the reverse case (≥ 95% in one crop but not 100% in the union) is the one the task asks about.

---

## K42-5 — energy of u512 + comp. (k42_5_energy.csv)

**Source.** Already measured in the K30 J2 window (2026-10-18), the same window and protocol as u960, native and v6-S.

The runs:
- results/raw/20261018_072404_k30j_timing_nusc_test20_v6S_j2_u512;
- results/raw/20261018_073816_k30j_timing_av2_v6S_j2_u512.

Run settings:
- `--only u512+c1comp`: u512 + C1 + v3a, trt32;
- MAXN, jetson_clocks, fan 100%;
- rails GPU + CPU + SOC + CV + VDDRQ, 1 s means;
- one run per pipeline, no intervals.

The row is in K30's j2_energy_marginal.csv but was not quoted in the K30 report text. No new window was run (A1).

Energy above the cheap run, in J per frame (wall ms above cheap in brackets):

| | u448 + comp. | **u512 + comp.** | u640 + comp. | u960 + comp. | native + comp. | DGSP (v6-S) |
|---|---|---|---|---|---|---|
| nuScenes test 20 | 0.201 (12.8) | **0.264 (14.6)** | 0.297 (16.9) | 0.539 (23.5) | 1.118 (38.4) | 0.289 (22.4) |
| AV2 8 logs | 0.409 (24.1) | **0.424 (25.4)** | 0.512 (27.8) | 0.755 (34.7) | 2.388 (77.1) | 0.536 (40.7) |

**Conclusion (unfavourable).** u512 + comp. needs less energy above the cheap pass than DGSP: 91% of DGSP on
nuScenes and 79% on AV2. It is also faster, from a single run each.
