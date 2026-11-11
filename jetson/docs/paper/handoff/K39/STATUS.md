# K39 STATUS
- 2026-10-27: task received (writing session; answers the mock review review27102026.md). Due dates: K39-3..5 (CPU)
  2026-10-30; K39-1 (Jetson) and K39-2 (VUW GPU) 2026-11-02.
- 2026-10-27: PREREG.md written before any run. Plan:
  - one VUW GPU harness (isolated copy of k30_eval.py, part k39) covering K39-2 strict cap + K39-4 single interventions;
  - CPU reports for K39-3 / K39-4 crosstab / K39-5;
  - Jetson K39-1 harness from the K38-1 harness + K30 J1 hybrid + B1K-A + extra whole-frame sides; engineering on dev,
    then one exclusive window.
- 2026-10-27: K39-2 primary heuristics fixed by the PREREG rule from K28 phase1/baseheur_r2.csv (dev, @S + C1, v3a):
  nuScenes unc4 / tcr3 / cor0; AV2 (and Argoverse-HD) unc2 / tcr0 / cor8; hybrid tsp8 everywhere.
- 2026-10-27: **K39-3 DONE** (results/k39/k3 → handoff/K39/k39_3_*.csv). Published deltas / CIs reproduced to 9e-17.
  - Family a (12 × 3): 29 / 36 CIs exclude 0; Holm 25, BH 28.
  - All-rows family (14 × 3): 35, Holm 30, BH 34.
  - Ablation (9): 7, Holm 7, BH 7.
  - No significant comparison favours a baseline over v6-S.
- 2026-10-27: VUW harness experiments/k39_vuw_20261027 (k39_eval.py md5 b8608ba3). Smoke (nuScenes scene-0110, AV2 first
  log): all parity 0 (v6-S / u960 vs K28, 8 heuristics x1.00 vs K30 match, sa vs ch_a, strict-with-∞-cap = nominal);
  strict px ratio max 1.0. Full run launched 16:0x VUW (3 groups × 3 shards).
- 2026-10-27: **K39-4 crosstab DONE** (k39_4_crosstab.csv). Agreement with π(F), without / with refiner:
  - DESP v6-S: 0.976 / 0.980 (nuScenes), 0.968 / 0.969 (AV2), 0.940 / 0.942 (AHD);
  - native + comp.: 0.974 / 0.977, 0.951 / 0.958, 0.922 / 0.926.
  - On DESP ≠ π(F) frames, without refiner, DESP matches the required action vs π(F) does: 128 vs 81 (nuScenes),
    219 vs 126 (AV2), 722 vs 130 (AHD).
  - Single interventions are pending in the VUW run.
- 2026-10-27: K39-5 harness k39_5_navsim.py. Smoke on one log: base poses reproduced (max |diff| 0); braking footprint
  empty ⇔ v3 0 rounds (0 mismatches). Full run: 8 CPU shards on VUW.
- 2026-10-27: K39-1 Jetson.
  - Profiling (dev) and micro-benchmark done; PREREG §A1 records the outcome: A1 static graphs kept, A1 crop graphs crash,
    A2 / A3 / A4 not applicable.
  - Harness experiments/k39_jetson_20261027/scripts/89_k39.py (build.py); dev smoke OK (all rows, deadlines, B1K-A).
  - **Window launched 10:15** (kit time; pause file, clocks locked, fan 255): calibration on dev → medians_k39.json →
    smoke → 3 reps × (nusc_test20, av2t40), about 5 h.
  - Report script scripts/k39_1_report.py.
- 2026-10-27 17:23 VUW: **K39-5 DONE** (k39_5_summary.csv, k39_5_top5.csv, k39_5_tokens.csv.gz; 12,146 tokens /
  136 logs).
  - Checks: base poses reproduced exactly; braking footprint empty ⇔ v3 0 rounds (0 mismatches).
  - The footprints differ on 99.8% of tokens (IoU < 0.9; median IoU 0.33; 67% of the PDM footprint lies outside the
    braking field).
  - The selected trajectory differs on 4.3% [3.6, 4.9] (517 tokens). On those tokens ΔPDMS = +0.045 [+0.005, +0.091];
    on all tokens +0.0019 [+0.0002, +0.0040].
  - Wall clock: fast 1 worker p50 6.0 / p95 10.1 s; 8 workers 1.34 / 1.96 s; frozen 14.7 / 19.2 s.
- 2026-10-27: VUW nuScenes shards DONE; AV2 / AHD still running. Jetson window at rep 2.
  - On the 517 tokens with a differing trajectory, spec_pdm scores better on 186, worse on 265 and ties on 66. The
    positive mean comes from large wins: 35 tokens have |ΔPDMS| = 1, 31 of them in favour of spec_pdm. The top 5
    (rule: |Δ|, then token) are all +1.
- 2026-10-27 13:04 (kit): **K39-1 window complete.** Parity 100% vs K38-1 / K34-3; hardware restored; hashes unchanged.
- 2026-10-27: VUW all shards DONE; parity checks all 0.
- **K39 DONE (all five items).** REPORT.md written; CSVs and figures in this folder.
- 2026-10-27: **K39-6 (fidelity vs compute) DONE.** PREREG section written first; checks 0; REPORT section added
  (plain "where v6-S is NOT best"); k39_6_* CSVs and figure.
