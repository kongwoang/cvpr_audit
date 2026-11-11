# K43 — pre-registration (written 2026-11-11, before any K43 number is computed)

Scope. Post-hoc analysis on test splits that were already seen. Nothing here is an independent confirmation. Frozen
v6-S (= DGSP), thresholds, weights, data and stored results are not changed; no configuration is selected on test; nothing
new runs on the 65 held-out sequences (their stored records may be read). No method code is edited. Items that need a GPU
and have no stored data are reported as "cần GPU / needs GPU" and are not run. Suspected bugs go to FINDINGS.md.

Before writing this file only column names / file layouts of the stored records were inspected (no values), plus the
manuscript tables already published (inputs/main.tex, supplementary.tex) and the writer's prelim note
(inputs/refiner_transitions_prelim.md).

Statistics (as K39): paired unit bootstrap (unit = scene / log / sequence), 2,000 draws, `np.random.default_rng(0)`,
95% percentile interval, Δ = DGSP − row. Every comparison made is written to CSV; none is filtered on significance.
`|v3a` (refiner v3a, the paper's convention) is primary for every action; `|none` is reported as secondary.

Population: GT frames of `results/k28/p3/{ds}_test_instr` (nuScenes 228 / 9,181; AV2 150 / 11,818; A-HD 24 / 15,006).

## K43-1 — when does DGSP differ from u512 / u640?

### Rows (action column; px column)
| row | source records | action |
|---|---|---|
| u320 | K28 p3 instr | `a_cheap` |
| u448, u512, u640, u960 (= F), u1280, native | `results/k28/{ds}_test_v6basewf-yolo11m` | `a_wf{s}+C1\|v3a` (native = 1600 / 2048 / 1920) |
| support + fixed tube, Tab. 1 form | K28 p3 instr | `a_tsp8@S+C1\|v3a` |
| support + fixed tube, strict cap | `results/k39/vuw/{ds}` | `a_tsp8@strict+C1\|v3a` |
| DGSP | K28 p3 instr | `a_v6S\|v3a` |

π(F) = `a_wf960+C1|v3a` (K42-3 convention; basewf u960 must equal instr F, checked). Secondary: the same with `|none`
everywhere (π(F) = `a_wf960+C1|none`); u320 has no refiner variant and is the same in both.
u448 is absent for AV2 if no stored column exists (reported, not run).

### Strata (fixed here; never merged after seeing numbers)
Each stratum dimension is a partition of all GT frames, so the bins of a dimension sum to "all".
1. **Range** of the decisive annotated object (`results/k30/gt/{ds}.pkl` `dec_range`, metres): `<20`, `20–40` [20,40),
   `40–60` [40,60], `>60`, on frames with `act_gt ≥ 1`; `none` = frames with `act_gt = 0` (no braking required).
   A frame with `act_gt ≥ 1` and no decisive object would form bin `required, no decisive object` (expected empty).
2. **Box height** of the decisive object at native resolution, normalised by the native image height (`dec_box` y2 − y1
   divided by 900 / 1550 / 1200): terciles `low / mid / high` with cut points (1/3, 2/3 quantiles) computed on **dev**
   frames with `act_gt ≥ 1`; `none` as above. Dev decisive boxes are not stored: they are computed on CPU from the dev
   annotations with the k30_gt rule (argmax of per-object required deceleration, same planner params) by a new script
   `k43_dev_gt.py` writing to `results/k43/devgt/` (never to results/k30). Dev = nuScenes val units of `IE.units_nusc("dev")`
   and AV2 `IE.units_av2("dev")` (the CFG-DEV splits used by K28/K30 dev). Argoverse-HD has no dev split in the frozen
   protocol, so it uses the AV2 dev cut points (normalised height). The held-out 65 sequences are not used for cut points.
3. **Class** of the decisive object, from `dec_class` with K34's `class_group` mapping: `car`, `truck-bus`
   (truck + bus), `pedestrian-cyclist` (pedestrian + cyclist), `other` (other_or_unspecified); `none` as above.
4. **Decisive object in the u320 draft**: K34 `coverage_frames.csv` `cheap_detected` True / False (defined on exactly the
   `act_gt ≥ 1` frames; asserted); `none` as above.
5. **Ego speed** tercile of `v` (m/s) with cut points from dev GT frames (`results/k30/dev_devref/{nusc,av2}` `v`);
   A-HD uses the AV2 cut points. All frames are binned (no `none`).

Small bins: a (dataset, stratum) with < 30 frames or < 5 units gets counts and point values but no interval; a missed-share
interval additionally needs ≥ 5 episodes in the bin (else point only).

### Metrics per (dataset, row, stratum)
- A = agreement with π(F); repair = A on frames where u320 ≠ π(F); A_gt = agreement with `act_gt`.
- Episodes: `k31_episodes.unit_stats` imported unchanged (level 1, required ≥ DECELERATE, merge gap 1 frame), computed per
  unit on the full sequence, then **assigned** to a stratum: a required episode by the frame of peak annotated demand
  (`a_gt` argmax inside the episode, first on ties); a false episode by its first frame (for object strata false episodes
  have `act_gt = 0` and land in `none`; for speed by the first frame's speed bin). Missed share = missed / episodes of the
  bin; false episodes per 1,000 frames of the bin. Bootstrap: unit draws applied to the per-unit counts of the bin.
- Loss components: J of the row (`J_*` stored, braking consumer); insufficient = Σ J over frames with a < act_gt,
  excessive = Σ J over frames with a > act_gt, each divided by the frames of the bin. Also G inside the bin
  (1 − ΣJ_row / ΣJ_u320) as an auxiliary column.
- px mean.
Δ intervals (DGSP − row) for A, repair, A_gt, missed share, false/1000, J_ins, J_exc, G.

### Cause decomposition (frames with `a_v6S|v3a ≠ a_wf512+C1|v3a`)
Direction: "DGSP worse" if J_DGSP > J_u512, "u512 worse" if J_u512 > J_DGSP, "equal loss" otherwise (counted, not labelled
beyond (f)). The label explains the worse pipeline W against the other O, first matching rule wins:
1. **(c) refiner** — W's action changed by the refiner (`|none ≠ |v3a`) and W's `|none` loss ≤ O's `|v3a` loss.
2. **(d) outside every crop** (W = DGSP only) — decisive object exists, `dec_in_crop_v6S` False and `dec_found_v6S` False.
3. **(a) not detected** — decisive object exists and is not found (`dec_found_v6S` False).
4. **(e) short looming history** (W = DGSP) — found, in a crop at t but not at t−1 (or t−1 absent).
5. **(b) state error** — found and the discrete action of the planner on the estimated state (`dec_z`, `dec_lmin`,
   `dec_lmax`, `dec_ttc` of v6S) differs from the one on the annotated state (`dec_range`, `dec_lat_min/max`, `dec_ttc`);
   errors (range, lateral, TTC) are reported.
6. **(f) other** — includes frames without a decisive object (`a_gt = 0`, false braking), and frames whose K30 match record
   action differs from the K28 scored action (sub-count reported).
For W = u512 only (c) can be decided: u512 detections are not stored (apdump holds cheap, v6S, wf640/960/1280/native but not
wf512), so (a)/(b)/(e) for u512 are labelled "undetermined — needs GPU" (counted separately, not run).
Outputs: counts per (dataset, direction, label); the 20 frames with the largest |ΔJ| per direction (unit, frame, label).

### Noise ceiling
Agreement (all frames) and repair (frames where u320 differs from the reference of the pair) of: u1280 vs F, native vs F,
u640 vs u512, with bootstrap intervals of the agreement itself (not a Δ).

### Held-out
If `results/k35/k4/train/k28p3` stores `a_wf512+C1|v3a` (and J), add a u512 row to the held-out table (G, ΔG interval
DGSP − u512, px, false / 1,000, missed) with the K35-4 statistics; else write "không có, cần GPU" and do not run.

### u640 anomaly on nuScenes
G(u640, Learn.) = −0.036 vs u512 0.124 (supp tab:uniform). Source records: basewf `a_wf{s}+C1|v3a`; first reproduce both
values (if not reproduced, find the records that do and report). Per-scene contribution c_u = (J_u320,u − J_row,u) / ΣJ_u320
for u640 and u512; top 10 scenes by |c_u640 − c_u512|; G of both after removing the top 1, 3, 5, 10 of those scenes;
"dominated" is stated if removing ≤ 3 scenes reverses the sign of G(u512) − G(u640) (descriptive rule, no test).

### Gate (K43-1)
- The "all" row of every dataset reproduces `handoff/K42/k42_3_fidelity.csv` (A, A|cheap!=F) for all rows present there,
  and `handoff/K42/k42_2_episodes.csv` (missed_rate, false_per_1000, level 1) for the rows present there, to 1e-12.
- For every dimension, the bins' frame, episode, missed, false-episode and agreement counts sum to the "all" counts.
- basewf u960 = instr F on every frame.

## K43-2 — the missing rows
1. **Recovery (main Tab. 4)** for support + fixed tube and the action-independent τ₁ tube: computed only if stored records
   with those pipelines exist for the natural-miss population and the deletion run (`*_test_instr-del_all`, K33 / K40R);
   else "cần GPU". Gate: the published DGSP / support-only recovery values recomputed from the same records.
2. **Fidelity of single round (`or6S`) and pivots (`pv6S`)**, `|v3a` and `|none`, as K42-3 / K39-6 (A, repair, other
   conditionals, Δ intervals). Gate: pv6S reproduces k42_3 (|v3a) and k39_6 (|none) to 1e-12.
3. **Curved comparator**: from `results/k31/r3/{ds}_curv` with K31's definitions: Δ G (curved DGSP − curved support +
   curved tube), (curved DGSP − straight-tube hybrid) and (curved DGSP − DGSP straight controller scored curved), on all
   frames and on "decisive object outside the straight tube". Gate: reproduce `handoff/K31/curv_test.csv` for these rows.
4. **Theorem attribution** (`|none`): frames with `a_v6S|none ≠ a_wf960+C1|none`. Locality enforced = chain step b
   (`a_ch_b`: F's objects in DGSP's crops and F's states); step a (F's boxes, own looming) reported too. Still-disagreeing
   frames are split into: coverage violation (some F object with demand ≥ threshold has `inR` False in the stored `cov`
   list) vs none; frames without a coverage violation are listed (unit, frame, actions) as not attributable to (iii)/(iv).
   Gate: the chain agreements reproduce K28 phase3 attribution (if the CSV is present) to 1e-12.
5. **Refiner direction**: transition counts `|none → |v3a` for DGSP and u960 + C1 (KEEP changed, lowered, raised, toward /
   away from act_gt); among level-1 episodes that DGSP|v3a misses and DGSP|none does not, how many have a refiner-lowered
   action on some frame of the episode (count); among DGSP|v3a missed episodes, how many have, at the peak frame, DGSP|v3a =
   KEEP and the stored estimated demand of the decisive object (`dec_dem_v6S`, K30 match) in [0.6 τ₁, τ₁). Counting only.
   Check: the writer's prelim counts are reproduced (reported if not).

## K43-3
Only `PREREG_refiner_dev.md` is written; nothing is run.

## Comparison list (all reported)
K43-1: 3 datasets × 10 rows (non-DGSP) × (1 + bins of 5 dimensions) × 8 metrics, `|v3a`; same for `|none`; ceiling 3 pairs ×
2 metrics × 3 datasets. K43-2: as listed per item. The total count is written to `k43_1_checks.json` / REPORT.

Amendments after this point are appended below with a timestamp.

### Amendment A1 (2026-11-11, before any K43 number was read)
- AV2 dev units: `IE.units_av2("dev")` yields a different log list (only 58 of the 300 logs of `results/k30/dev_devref/av2`
  overlap). The dev decisive boxes for AV2 are therefore computed on the 300 dev_devref logs (AV2 sensor train), with the
  frame generator of `IE.units_av2` copied unchanged, so that the height and speed cut points come from the same 300 logs.
  nuScenes: `units_nusc("val")` = the 152 dev_devref scenes (all found).
- The `|none` fidelity gate uses `results/k39/k6/k39_6_fidelity.csv` (md5 7e71c247… = handoff/K39 copy); the `|v3a` gate
  uses `results/k42/k3/k42_3_fidelity.csv` (md5 50f5af1e… = handoff/K42 copy). Episodes: `results/k42/k2/k42_2_episodes.csv`
  and, for DGSP `|none`, `results/k31/tables/ep_summary.csv` ("v6-S without refiner").
