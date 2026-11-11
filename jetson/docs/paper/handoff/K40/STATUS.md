# K40 STATUS (full reproducible release; supersedes the "Đợt 21" release task)

Deadlines:
- Stage A starts now.
- Stages B / C run on free resources (K37 / K38 / K39 have priority; all three are DONE as of 2026-10-27).
- Code snapshot frozen after the method-change stop point, 2026-11-03.
- v1.0 push by 2026-11-14; v1.1 by 2026-11-22.

- 2026-10-27: task received. handoff/K40/ created.
  - **Blocker for Stage 0 / R0:** the paper sources (main.tex, supplementary.tex) are not on the kit. They are needed
    for the printed numbers and the per-item source comments. Requested from the writing session via the user.
  - **Decision needed before the push:** the public repo anonymous221206/DESP already has my K38 commit 1c0bb32
    "Initial release" and a Release "v1.0" (frozen v6-S + two weights). K40 also asks for a Release "v1.0".
    **Decided by the user (2026-10-27): keep v1.0; K40 releases are v2.0 (Stages A–B, by 11-14) and v2.1 (everything,
    by 11-22), as normal follow-up commits on main. No force-push.**
- 2026-10-27: Stage 0 started.
  - manifest_draft.csv lists 60 items: label, part, source handoff, the candidate files found, release script,
    level, acceptance.
  - Eight items have no traceable source without the .tex: tab:constructions, tab:practice, fig:locality,
    tab:benchmarks, tab:baselines, and the three fig:qual_* panel sources, plus the in-text numbers.
  - Figure items (fig:forest, fig:uniform, fig:loss, fig:coverage, fig:operating, fig:latency, fig:deadline) appear to be
    drawn by the writing session from handoff CSVs. Its figure scripts are needed too.
- 2026-10-27: Stage A started. Fresh export /tmp/desp_k40, seeded from the K38-released desp/ (gates passed), with
  skeleton dirs baselines/ analysis/ closed_loop/ planning/ jetson/.
  - Plan, in order of the R1 needs:
    1. port the record-producing pipelines: whole frame ± T4 / C1 / v3a, ablations, bounded, heuristics,
       AnA / counterfactual, routers, AdaScale, Chanakya, SAHI / FOVEA / UFP adapters;
    2. analysis/ scripts per item;
    3. data / manifests;
    4. training;
    5. closed_loop / planning / jetson.
- 2026-10-27: Stage A progress in /tmp/desp_k40:
  - desp/loop.run: ablation options field / support / guard. Defaults are unchanged, so v6-S is unaffected
    (regression to be re-run).
  - desp/consumers/braking: pivots().
  - desp/temporal.py: the T4 looming lift, ported from k27_dev.ttc_ss with policy T4+T2_1.
  - desp/detection: box entropy, needed by the uncertainty heuristic.
  - baselines/wholeframe.py: whole frame × {plain, T4, C1} × {no refiner, v3a}, each with its own history.
  - baselines/heuristics.py: unc / tcr / cor / tsp grids as in k30_eval, nominal and strict budgets.
- **Finding for R1 on nuScenes.** K28's nuScenes records read the draft from DEEP's precomputed cache
  (data/cache/nusc_det_tv/ns_cheap_320: TRT FP16 boxes AND their precomputed mono geometry, via
  _apply_range_source(..., "mono")). An online release run cannot be per-frame identical (as K38 showed).
  - Plan: ship the cached nuScenes draft scenes (xyxy, conf, coarse, entropy, mono; sanitized parquet, about 79 MB for
    all of trainval) as a Release artifact, and add `--draft cached` to the runner. R1 on nuScenes then runs against
    the same draft. The online-draft path stays the default and is documented; K38 measured live-draft G 0.178 vs 0.211.
- Next:
  - runner scripts/run.py with records named like the stored columns;
  - parity tool against the K28 / K30 / K33 records on the regression subset;
  - then AnA / counterfactual, routers / AdaScale / Chanakya, SAHI / FOVEA / UFP adapters; analysis/ scripts after
    the .tex arrives.
- 2026-10-27: **R1 smoke PASSED, AV2 test first log (79 frames).** Release scripts/run_rows.py (fresh env) vs the
  stored records, 0 differing values on every compared column:
  - K28 instr: 50 columns. cheap; v6S / so6S / fo6S / pv6S / or6S / ng6S / v51S (none + v3a + px); heuristics
    unc2 / tcr0 / cor8 / tsp8 @S; wf640 / 960 / native, plain and + C1.
  - K28 basewf: 47 columns. Whole frame 512 / 640 / 960 / 1280 / 1536 / native × plain / + T4 / + C1 × none / v3a.
  - K30 match: 24 columns. Heuristics x1.00.
  - K39 vuw: 33 columns. Heuristics strict and x1.00; u960 + C1.
  - Parity helper (private): remote/A_method/k40_parity.py. Log: VUW logs/k40/r1_smoke_av2_parity.json.
- 2026-10-27 (after the inputs arrived): inputs/k40_inputs/ extracted (paper/main.tex, supplementary.tex, figs,
  figure_scripts/, supp_fig_tools/).
  - **Stage 0 manifest** (build_manifest.py → manifest.csv): 62 labelled items. 43 carry a source comment; 19 were
    resolved:
    - generated supp plots → their table of origin (make_supp_figs.py);
    - fig:pareto → its comment 85 lines above;
    - qualitative figures → K35-6 + render_panels.py;
    - descriptive tables → the text-number pass.
  - Every referenced handoff file exists. In-text numbers: pending (separate pass).
  - **Records artifact.** remote/A_method/k40_export_records.py exports one wide parquet per dataset with every stored
    row's per-frame a / J / px (K28 k24v6 + p3, K33 de / pub / rc, K30 match, K39 vuw):
    - nuScenes 9,181 × 1,675 columns, 19 MB; AV2 11,818 × 1,647, 40 MB; AHD 15,006 × 1,660, 45 MB;
    - shared columns are identical across sources;
    - the only same-name clash is K33's wf640 / 960 / 1280 (whole frame + C1) vs K28's plain wf, kept as tab1.<col>.
  - R0 machinery:
    - extract_tables.py → release tests/paper_tables/<label>.tex (the printed rows, 45 tables);
    - release scripts/check_r0.py compares analysis/<label>.py output line by line.
  - **R0 tab:main: 25 / 25 lines identical**: every number, dagger and bold / underline, from the per-frame records,
    with one bootstrap (sorted units, default_rng(0), 2,000).
- **Issue for the writing session** (no number changed). tab:main's heuristic variants are, from the records:
  - nuScenes unc2 / tcr3 / cor0 / tsp8; AV2 unc2 / tcr3 / cor8 / tsp8; AHD unc2 / tcr0 / cor8 / tsp8.
  - K39-2 (tab:strictcap) and K39-6 (tab:fidelity) took their primary variants from the K28 baseheur dev-G rule
    (PREREG K39): nuScenes unc4, AV2 tcr0.
  - So the "Uncertainty" row on nuScenes and the "Temporal criticality" row on AV2 refer to different variants in
    tab:main vs tab:strictcap / tab:fidelity.
  - k39_2_rows.csv has all 8 variants, so the tab:main variants are available there. tab:fidelity has only the K39
    primaries; it would need a recompute with unc2 / tcr3.
- 2026-10-27: R0 progress (REPRODUCTION.md):
  - tab:main 25 / 25 and tab:ablation 10 / 10 identical.
  - tab:field: 10 / 13 lines identical. 3 cells differ by paper rounding; the release equals the handoff CSVs
    (AnA AHD natural 0.8647 printed 0.87; braking AV2 −0.00267 printed −0.002; curved path-tube AHD 0.01549 printed
    0.016).
  - New artifact records/objects_<ds>.parquet: decision-relevant objects × found_<row>, natural and deletion
    settings; K28 and K33 agree object for object.
  - records/<ds>.parquet now also carries the speed (K30 v8d), curved-consumer (K31 r3) and TTC (K31 r2) runs.
- Next (in order):
  1. **Jetson timing artifact.** Per-run per-frame latency from results/raw frames.pkl of the K29 / K33-J / K34-3 /
     K38-1 / K39-1 windows, plus the K30-J2 energy logs. Then analysis for tab:eff, tab:latency, fig:latency,
     tab:latency_baselines, tab:deadline, fig:deadline, tab:frontier.
  2. **K32 closed-loop artifact** (per route × seed × branch × setting): tab:aeb_main, tab:aeb.
  3. Supp tables from the existing records artifact: tab:uniform / _ci, tab:heuristics, tab:strictcap, tab:fidelity,
     tab:taxonomy, tab:loss, tab:locality, tab:coverage, tab:det / det2, tab:speed, tab:curved, tab:consumer2,
     tab:episodes, tab:severity, tab:missed, tab:strata, tab:holdout, tab:constrained, tab:curves, tab:versions,
     tab:hood, tab:official, tab:ft, tab:b2d, tab:main_ci; and the generated plots via make_supp_figs.py.
  4. The text-number pass.
  5. R1 runs at scale; the nuScenes draft cache.
- 2026-10-27 (later): **R0 of every table and data figure done** (REPRODUCTION.md has the full status table).
  - 39 tables with numbers: all regenerated from the released records; 33 identical, 6 differ in a few cells (paper
    rounding or markers; listed under "Differences found"; nothing corrected).
    - New since the last entry: tab:latency 8 / 8, tab:eff 6 / 6, tab:latency_baselines 15 / 15, tab:deadline 4 / 4,
      tab:frontier 12 / 12, tab:navsim 6 / 6, tab:det 20 / 20, tab:b2d 4 / 5 (u960 DS printed 93.2, source 93.148).
  - 11 data figures (pgfplots coordinates, numeric labels, interval bars; analysis/figures.py from the released
    records / the regenerated tables): 10 identical, fig:pareto 22 / 24 (oracle router px printed 0.053 / 0.077,
    source 0.0536 / 0.0776).
  - New record files: jetson_{latency,bounded,energy,baselines,deadline}.parquet, navsim_{navtest,navhard,
    navhard_groups}.parquet, b2d_{routes,inloop}.parquet, det_{predictions,frames}_<ds>.parquet, tube_frames.parquet,
    argoverse_hd_official.parquet (routers only). Records total 218 MB (one release asset).
  - The 6 descriptive tables (roadmap, constructions, practice, benchmarks, baselines, failures) and the
    illustrations (fig:teaser, method, qual*) have no regenerated numbers; their quoted numbers go to the text pass.
  - Release scripts added: scripts/reproduce.sh (all-r0 | r0 | r1 | main; replaces reproduce_main.sh),
    download_records.sh, compare_r1.py (R1 against the released records), dump_r0.py; tests/test_r0.py (every item
    must match the paper except the documented differences, tests/paper_differences.csv).
- Text-number pass started (text_numbers.py → text_numbers.csv): 1,223 numbers in the running text, 525 with
  decimals. Decimals: 119 equal a value of a cited regenerated item, 315 a value of another regenerated item,
  91 none. The 91 are settings (grid ranges, decelerations, deadlines) or results from handoffs without a released
  record (NAVSIM v1 PDMS K20, locality K31, K35-5 / K35-8, K38-2 coverage, K33 recall, K39 cap ratios, K29 B2D
  intervals, K30 Jetson hybrid latency, ...). Next: classify every number (setting / R0 via item / text value),
  and make the result numbers regenerable (analysis/text_values.py, plus small record files where missing).
