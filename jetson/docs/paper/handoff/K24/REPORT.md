# K24: confirmatory test run of v5.1-S (v5.1-L and v5 as references)

**(a) v5.1-S against the previous frozen configuration as reported**, with refiner (G_refiner):

| Test split | Previous frozen | v5.1-S | v5.1-S − previous | Verdict | px (prev / v5.1-S) |
|---|---|---|---|---|---|
| nuScenes new test, 228 scenes | v3 + refiner v1: 0.170 | **0.199** | +0.029 [−0.018, +0.070] | matches (not outperforms) | 0.100 / 0.077 |
| AV2 val, ego mask, 150 logs | v4-lite + refiner v2: 0.122 | **0.262** | +0.139 [+0.089, +0.190] | outperforms | 0.053 / 0.053 |
| Argoverse-HD val, 24 sequences | v4-lite + refiner v2: 0.091 | **0.338** | +0.248 [+0.159, +0.350] | outperforms | 0.070 / 0.070 |

Without refiner the verdicts are the same: nuScenes +0.027 [−0.017, +0.068], AV2 +0.116 [+0.074, +0.161], A-HD +0.242 [+0.178, +0.327].

**(b) Baseline rows v5.1-S does NOT outperform** (with refiner; lower bound ≤ 0). Each entry gives (v5.1-S − row) [interval] and the row's tensor px; v5.1-S px is 0.077 / 0.053 / 0.070.

*nuScenes*
- **Previous frozen:**
  - v3 + refiner v1: +0.029 [−0.018, +0.070], matches.
  - **v3 + components: −0.014 [−0.043, +0.004], below** (0.100 px).
  - v3 + v3a (extra): +0.017 [−0.026, +0.057].
- **Whole frame, all below** (wide intervals):
  - u640 + components: +0.268 [−0.039, +0.685];
  - u960 / + components: +0.144 / +0.102;
  - u1280 / + components: −0.053 / −0.070 [−0.162, +0.026];
  - u1536 / + components: −0.058 / −0.081 [−0.215, +0.069];
  - native / + components: +0.023 / +0.002.
  - The u1280 and u1536 rows score G 0.25–0.28, at 8–12× the px.
- **Heuristics:**
  - traj_support @ 1× / + components: +0.051 [−0.005, +0.110] / +0.034 [−0.018, +0.094], matches (0.108 px).
  - temporal_crit @ 2× / + components: +0.009 [−0.046, +0.078] / −0.013 [−0.060, +0.041], below (0.127 px).
- **Ablation:** field only + components: +0.033 [−0.081, +0.153], below (0.068 px).
- **Oracle router (true V), plain / + components:** −0.167 / −0.150, below.

*AV2*
- **Whole frame + components:**
  - u640: +0.043 [−0.003, +0.084], matches;
  - u960 / u1280 / u1536 / native: +0.007 / −0.010 / +0.029 / +0.038, all below (intervals reach −0.026 to −0.061).
- **Heuristics + components:**
  - matches: uncertainty @ 1× +0.051 [−0.008, +0.105]; temporal_crit @ 1× +0.038 [−0.011, +0.084]; temporal_crit @ 2× +0.015 [−0.010, +0.041]; corridor @ 2× +0.016 [−0.011, +0.046];
  - below: traj_support @ 1× −0.004 [−0.020, +0.012] and @ 2× −0.002 [−0.023, +0.019].
- **Previous frozen v4-lite + components:** identical to v5.1-S by construction (same loop).
- **Oracle router, plain / + components:** −0.156 / −0.134, below.

*Argoverse-HD*
- **Whole frame:** native + components: +0.056 [−0.011, +0.132], matches.
- **Heuristics + components, all matches:**
  - traj_support @ 0.5× / 1× / 2×: +0.033 / +0.010 / +0.016;
  - corridor @ 1× / 2×: +0.021 / +0.020.
- **Ablation:** support only + components: +0.052 [−0.034, +0.175], below.
- **Previous frozen v4-lite + components:** identical by construction.
- **Oracle router, plain / + components:** −0.031 / −0.017, below.

*Extras* (budgets matched to v5.1-L, which are 1.25–1.4× v5.1-S px): the same families recur, as traj_support, corridor, temporal_crit and uncertainty + components. They are all in `not_outperformed_v51.csv`, which lists 28 / 26 / 18 rows for v5.1-S (nuScenes / AV2 / A-HD) and also the v5.1-L list.

**Outperformed on every split:**
- both learned routers, plain and + components (cond_moric and cond_V);
- every plain crop heuristic, except nuScenes traj_support @ 1× and temporal_crit @ 2×;
- every plain whole-frame pipeline on AV2 and A-HD;
- every random-crop row.

**(5) Ablations** (`ablation2x2_*_v51.csv`, G with v3a unless noted; columns are nuScenes / AV2 / A-HD):

| Cell | nuScenes | AV2 | A-HD | K17(3) G | Unnecessary / missed per 1k |
|---|---|---|---|---|---|
| v5.1-S (v3a + T4) | 0.199 | 0.262 | 0.338 | 0.32 / 0.35 / 0.59 | 22 / 76 / 169 · 37 / 35 / 46 |
| Without T4 (= v5) | 0.176, +0.023 [+0.005, +0.046] | 0.208, +0.054 [+0.012, +0.094] | 0.185, +0.154 [+0.111, +0.210] | 0.32 / 0.33 / 0.53 | 33 / 112 / 241 · 34 / 30 / 35 |
| Without refiner | 0.162, +0.036 [+0.006, +0.076] | 0.147, +0.114 [+0.062, +0.170] | 0.282, +0.057 [−0.029, +0.169] | 0.27 / 0.24 / 0.62 | 31 / 167 / 243 · 42 / 27 / 22 |
| Without both | 0.133 | 0.032 | 0.039 | 0.27 / 0.22 / 0.53 | 43 / 206 / 326 · 39 / 24 / 15 |
| Support only + components | −0.095, +0.294 [+0.043, +0.522] | 0.207, +0.055 [+0.006, +0.116] | 0.286, +0.052 [−0.034, +0.175] | 0.02 / 0.14 / −0.01 | 12 / 55 / 142 · 46 / 48 / 88 |
| Field only + components | 0.166, +0.033 [−0.081, +0.153] | 0.003, +0.259 [+0.202, +0.319] | 0.050, +0.288 [+0.208, +0.381] | 0.30 / 0.15 / 0.60 | 54 / 190 / 310 · 33 / 29 / 29 |

- **T4 is significant on all three splits.** It mainly removes unnecessary actions: per 1k, −11 on nuScenes, −36 on AV2 and −72 on A-HD. Missed actions rise by 3–11 per 1k.
- **The K17(3) regime gets no worse with T4.**
- **Neither half alone suffices on all splits.** Field only matches v5.1-S on nuScenes but collapses on AV2 and A-HD. Support only collapses on nuScenes and on the K17(3) regime everywhere (0.02 / 0.14 / −0.01).

**Interpretation and caveats.**
1. **Partly confirmed.** v5.1-S is far above its predecessor on AV2 and A-HD, which is mostly T4 + v3a. On nuScenes it only matches.
2. **v_close.** On nuScenes, the old v_close 10 loop with the same components (0.212) is nominally above v5.1-S (n.s.). The shared v_close 0 chosen on dev (K23/K27) is not confirmed there.
3. **"+ components" closes most of the gap.** Whole frame, traj_support and corridor with T4 + v3a are matched rather than outperformed on AV2 and A-HD. The crop-selection advantage over the best component-equipped heuristic is about 0 to +0.02 (n.s.).
4. **The oracle router is an upper bound,** not deployable. It uses true V, and it beats v5.1-S on nuScenes and AV2.
5. **v5.1-L vs v5.1-S:** L is significantly better on AV2 (+0.035 [+0.003, +0.077]) and S is better on A-HD (+0.019 [+0.009, +0.033]).

**Protocol.**
- Each split was run once. Main pass: `k24_test.py` (md5 d2ae37cc…), run 18:52–22:24 VUW.
- The revised spec arrived during the main pass, before any result was seen. It was pre-registered in method_freeze_draft §2i (revision, with md5s).
- The supplementary pass `k24_supp.py` (cc4b714f…) ran the configurations missing from the main pass on the same frames: nuScenes v3 + components, support only and field only. It ran 21:37–22:50 VUW.
- `k24_analyze.py` (3ee3290b…) was fixed before the results.
- No shard crashed and no reruns were needed. Units: 228 / 150 / 24.
- Frames: 9,181 / 11,818 / 15,006; A-HD counts only frames with 3D labels.
- Main and supplementary passes agree on J_cheap and act_gt for 100% of frames.
- One unit per split was smoke-run before the main pass, for parity with K18 only (100% match on the frozen configuration's actions, px and J_cheap); its G was not looked at.
- **Routers** (K18 protocol): re-matched within 2% of v5.1-S px (all True); J_cheap agreement with the router tables is 1.0.
- **Oracle:** the top-k frames by the true V of u960 + v3a, one mask for both settings.
- **Statistics:** paired unit bootstrap, 2,000 resamples, seed 0.
- **"+ components"** means v3a plus v5.1-S's lift (T4 with the T2_1 gate).
- **Previous frozen "as reported" values** reproduce the earlier reports (K3 nuScenes 0.170; 03_results AV2 0.122 and A-HD 0.091).

**Files.**
- table_main_{nusc,av2,ahd}_v51.csv
- ablation2x2_{nusc,av2,ahd}_v51.csv
- pareto_{nusc,av2,ahd}_v51.csv and pareto_v51.{pdf,png} (G with v3a vs tensor px; hollow markers = + components)
- headline_v51.csv
- not_outperformed_v51.csv
