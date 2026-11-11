# K31 REPORT — post-hoc analyses of the frozen v6-S and the K28 / K30 baselines

**Status: FINAL 2026-10-20 (VUW 01:20).**
- Final: items 1, 2, 3, 4 and 5, and the B2K variant (dev selection and test).
- Item 1 panels and item 6 (NAVSIM bottleneck-selector check) are done; nothing is pending.

**Everything here is post-hoc.**
- v6-S is frozen (K28 §2n). No number here selected anything, except B2K, which was selected on dev by a pre-registered rule.
- Baselines are in their K28 forms: refiner v3a + C1 for every pipeline; guard and own history for the crop heuristics.
- Intervals: paired unit bootstrap, 2,000 resamples, seed 0. Units are scenes, logs or sequences.
- Sign convention: "v6-S − X" > 0 means v6-S is better. "Significant" means the 95% interval excludes 0.
- Every row of every comparison is in the CSVs next to this file. §3 lists every row where v6-S (or the loop under test) is significantly below, with its mechanism.
- Datasets: nuScenes new test (228 scenes, 9,181 frames, 2 Hz); AV2 val with the ego mask (150 logs, 11,818 frames, 5 Hz); Argoverse-HD val (24 sequences, 15,006 frames, 30 Hz).

---

## 1. Item → numbers for the paper

| item | question | numbers for the paper | verdict |
|---|---|---|---|
| **1** episodes, required level (≥ DECELERATE) | Does v6-S start braking episodes in time, and does it flicker? | Missed required episodes, v6-S vs u960 + C1 + v3a / native + C1 + v3a / hybrid: nuScenes (147 episodes) **0.408** vs 0.347 (+0.061 [0.014, 0.114]) / 0.361 (+0.048 [0.006, 0.097]) / 0.395 (n.s.); AV2 (190) **0.189** vs 0.163 (n.s.) / 0.153 (+0.037 [0.005, 0.075]) / 0.184 (n.s.); Argoverse-HD (71) **0.380** vs 0.366 / 0.380 / 0.380 (all n.s.). Mean onset delay (capped): nuScenes 0.74 s vs u960 0.54 s (+0.197 s [0.100, 0.313]); AV2 0.22 s vs native 0.19 s (+0.027 s [0.002, 0.054]); Argoverse-HD 0.135 s (n.s. vs all three). False episodes / 1,000 frames: v6-S **4.0 / 6.5 / 6.3** vs u960 6.4 / 7.9 / 5.5 (nuScenes −2.4 [−3.6, −1.2], AV2 −1.4 [−2.4, −0.3], Argoverse-HD n.s.), unc2 7.2 / 14.9 / 15.9 (all significant), cheap 22.9 / 34.8 / 25.1. A→B→A reversals / 100 frames: v6-S **1.38** (lowest on nuScenes; −0.45 [−0.78, −0.14] vs u960), 2.07 on AV2 (n.s. vs u960), 1.20 on Argoverse-HD (**u960 lower, 0.80; +0.40 [0.15, 0.67]**) | v6-S has few false episodes and little flicker: the lowest on nuScenes, and within 1 false episode per 1,000 frames of the lowest elsewhere. But on nuScenes it starts required episodes later and misses more of them than the whole-frame rows |
| 1, hard-brake level | Same, for HARD-BRAKE episodes | Missed: nuScenes (44 episodes) v6-S 0.477 vs u960 0.500, native 0.523, hybrid 0.455 (all n.s.); AV2 (59) 0.186 vs 0.169 / 0.203 / 0.186 (n.s.); Argoverse-HD (16) 0.500 for v6-S, u960, native and hybrid | no significant difference at the hard-brake level |
| 1, refiner | Who causes the late onsets? | v6-S without the refiner, missed required episodes: 0.388 / **0.142** / **0.211** vs v6-S 0.408 / 0.189 / 0.380 (v6-S − no refiner +0.020 [0.000, 0.054] / +0.047 [0.021, 0.078] / +0.169 [0.070, 0.271]). The refiner also removes false episodes (−2.7 [−4.1, −1.3] / −5.0 [−7.3, −2.9] / n.s. per 1,000). At the hard-brake level on nuScenes the refiner **reduces** misses (0.477 vs 0.614, −0.136 [−0.258, −0.034]) and the mean delay (−0.375 s [−0.804, −0.082]) | the refiner v3a, which every "+ C1 + v3a" row shares, trades earlier DECELERATE onsets for fewer false episodes; on Argoverse-HD every refined row misses 0.37–0.39 of the required episodes vs 0.17–0.21 without the refiner |
| 1, longest delays | Why are the 10 worst v6-S onsets late? | Cause at the episode's first frame (nuScenes / AV2 / Argoverse-HD): decisive object found, TTC or lateral estimate keeps the demand low 5 / 5 / 2; found, range too long 1 / 2 / 5; inside a crop but missed 2 / 1 / 3; outside every crop 2 / 2 / 0 | 8 / 8 / 10 of the 10 are lift or detection errors on an object that v6-S did look at; selection misses 2 / 2 / 0 |
| **2a** TTC consumer | Does DESP's TTC loop beat the hybrid that uses the TTC consumer's own support, at the loop's px? | G2: DESP TTC loop (c2 + v3a) **0.191 / 0.496 / 0.750** at px 0.086 / 0.060 / 0.079. Loop − hybrid tsp8 with the TTC support at the loop's per-frame px (×1.0 cap, R2a re-run; realised px 0.124 / 0.077 / 0.128): **+0.034 [0.006, 0.069] / +0.024 [0.005, 0.045] / +0.026 [0.010, 0.044]**. Loop − the same hybrid at K28's S budget: +0.045 (n.s.) / +0.070 [0.034, 0.110] / +0.134 [0.054, 0.241]. Loop − unc2 at the loop's px: +0.004 (n.s.) / +0.079 / +0.077. Loop − native + C1 + v3a: −0.031 (n.s.) / −0.003 (n.s.) / +0.121 [0.015, 0.327]. Loop − frozen braking-driven v6-S scored by TTC: +0.038 (n.s.) / +0.058 [0.024, 0.097] / +0.138 [0.060, 0.247] | the loop driven by the consumer's own field is above the matched-px hybrid on all three datasets, and above the braking-driven v6-S on AV2 and Argoverse-HD |
| **2b** IDM consumer | Does re-targeting the loop to IDM (ours-IDM) help? | G3: ours-IDM 0.030 / 0.148 / 0.302; frozen v6-S 0.086 / 0.156 / 0.313. ours-IDM − v6-S: **−0.056 [−0.092, −0.020]** / −0.008 (n.s.) / **−0.011 [−0.024, −0.002]**. Hybrid with the IDM support at K28's S budget (R2b re-run): ours-IDM − hybrid −0.016 / −0.014 / +0.003 (all n.s.); v6-S − hybrid +0.040 / −0.007 / +0.014 (all n.s.). v6-S − hybrid at ours-IDM's px: +0.060 [0.007, 0.111] / −0.009 (n.s.) / +0.018 [0.0003, 0.044] | re-targeting to IDM does not help; the braking-driven v6-S transfers as well or better (K30 V2b confirmed with intervals and the S-budget row) |
| 2b mechanism | Why is ours-IDM not better? | ours-IDM uses 2.6–2.7× the crops of v6-S (1.71 / 2.37 / 2.52 vs 0.66 / 0.87 / 0.98 per frame) at 1.05–1.09× the px, and a larger field (6.8 vs 5.4 cells, nuScenes). Their IDM actions differ on 0.9% / 2.1% / 1.5% of frames, and ours-IDM costs 182 / 69 / 178 J3 more there (5.6% / 0.8% / 1.1% of the cheap draft's sum). On differing frames both loops found the decisive IDM object on 75 / 221 / 189 frames; there the estimated range differs (> 1%) on 62 / 122 / 85, and ours-IDM's is longer on 47 / 80 / 58. On nuScenes the decisive object is refined by v6-S only on 40 frames vs by ours-IDM only on 7 | both loops find the leader; they differ in its lift. v6-S refines the leader through the braking support, ours-IDM often does not |
| **3** curved-path AEB consumer | Does the method hold when the consumer's corridor follows the ego's curvature? | Dev: field non-empty straight / curved 0.510 / 0.503 (nuScenes val), 0.383 / 0.383 (AV2 CFG-DEV); \|κ\| > 0.01 on 20.2% / 13.4% of frames, so the consumer behaves. Test, G under the curved consumer: v6-S with the curved consumer **0.207 / 0.224 / 0.394** at px 0.077 / 0.052 / 0.068. Curved v6-S − cheap +0.207 / +0.224 / +0.394 (all significant); − hybrid with straight tube (as K28) +0.053 (n.s.) / +0.016 (n.s.) / **+0.042 [0.022, 0.064]**; − hybrid with a curved tube and curved support (re-designed prior) +0.022 / +0.025 / +0.016 (all n.s.); − u960 + C1 + v3a +0.106 (n.s.) / **−0.051 [−0.106, −0.007]** / **+0.113 [0.009, 0.310]**; − native + C1 + v3a −0.028 / −0.001 / −0.028 (n.s.); − unc2 +0.290 [0.030, 0.509] / +0.057 (n.s.) / +0.139 [0.049, 0.255]. Extra: the frozen straight v6-S scored by the curved consumer 0.227 / 0.254 / 0.334 (curved − straight −0.019 / −0.030 (n.s.) / **+0.060 [0.031, 0.093]**) | the method carries over to a curved consumer: it ties the curved-tube hybrid, which is an upper reference that knows the path. It is above the straight hybrid on Argoverse-HD. On AV2, u960 + C1 + v3a stays above, as with the straight consumer in K28 |
| 3, strata | High curvature and decisive objects outside the straight tube | Top \|κ\| tercile: curved v6-S − straight hybrid +0.025 / +0.032 / **+0.083 [0.045, 0.128]**; − frozen straight v6-S −0.035 / −0.095 (n.s.) / **+0.133 [0.053, 0.208]**; AV2 − u960 **−0.110 [−0.228, −0.012]**. Decisive object outside the straight tube (1,498 / 3,166 / 3,514 frames): curved v6-S − straight hybrid +0.028 / +0.020 / **+0.076 [0.040, 0.123]**; − curved hybrid +0.014 / +0.033 / **+0.037 [0.007, 0.076]**; AV2 − u960 **−0.117 [−0.241, −0.018]** | on Argoverse-HD the curved field is above every tube in both strata. On AV2, at high curvature, curved v6-S is below u960 (mechanism in §3) |
| **4** bounded loop, ≤ 2 detector calls | What does the Jetson's bounded variant score on the full test splits? | v6-S ≤ 2: G **0.091 / 0.282 / 0.419** at px 0.073 / 0.049 / 0.064 (v6-S 0.211 / 0.294 / 0.418). It stops at the bound on 11% / 24% / 25% of frames. v6-S ≤ 2 − v6-S: **−0.120 [−0.299, −0.0001]** / −0.012 (n.s.) / +0.001 (n.s.); − u960 −0.021 (n.s.) / **−0.057 [−0.114, −0.007]** / **+0.122 [0.013, 0.326]**; − hybrid −0.083 (n.s.) / **−0.015 [−0.032, −0.0004]** / +0.011 (n.s.). Parity with the Jetson (K29 trt32, common frames): bounded 99.63% (805 frames) / 99.83% (586), unbounded 99.63% / 99.32% | the bounded loop equals v6-S on AV2 and Argoverse-HD. On nuScenes it loses through one scene (574 of the 590 extra cost: scene-1051, a truncated decisive object dropped at the bound) |
| **4b** B2K variant (post-hoc, dev-selected) | Does keeping the pending truncated detection at the bound repair the nuScenes loss without a third call? | Dev G (nuScenes val / AV2 CFG-DEV): v6-S 0.321 / 0.309; v6-S ≤ 2 0.252 / 0.306; **B2K-A 0.346 / 0.307**; B2K-B 0.316 / 0.286. B2K-A is selected by the registered rule (mean 0.326 > 0.279). px is unchanged (0.0744 / 0.0504). Test (once): B2K-A **0.209 / 0.293 / 0.415** at the px of v6-S ≤ 2 (0.073 / 0.049 / 0.064; v6-S 0.078 / 0.054 / 0.071). B2K-A − v6-S −0.001 [−0.005, 0.002] / −0.001 [−0.007, 0.005] / −0.003 [−0.008, 0.001]; − v6-S ≤ 2 +0.119 [−0.001, 0.299] / +0.011 (n.s.) / −0.004 (n.s.); − u960 + C1 + v3a +0.098 (n.s.) / −0.046 [−0.104, 0.004] (n.s.) / **+0.119 [0.013, 0.312]**; − hybrid +0.036 / −0.003 / +0.007 (n.s.); − native −0.013 / −0.001 / −0.015 (n.s.) | with at most two detector calls and 6–10% fewer px than v6-S, B2K-A matches v6-S on all three test splits (all within ±0.003, n.s.). The nuScenes loss of the frozen bounded loop is repaired. No row is significantly above B2K-A |
| **5** distribution-shift strata | Does v6-S's position hold under night, rain and city shifts? | nuScenes night (27 scenes): v6-S 0.231, u960 −0.061 (+0.292 [−0.098, 0.605], n.s.), hybrid 0.144 (n.s.), unc2 −0.379 (+0.609 [0.075, 0.821]); day (201): v6-S 0.197, every non-cheap row n.s. except unc2 (+0.034). Rain (50 scenes): v6-S 0.440, u960 0.479 (n.s.), hybrid 0.440 (tie); no rain (178) 0.140, all n.s. except unc2 and cheap. AV2 per city (ATX 4 logs flagged): v6-S never significantly below, except **WDC (18 logs): native + C1 + v3a above, −0.200 [−0.444, −0.006]**; unc2 below v6-S in DTW, MIA, PIT. Argoverse-HD: MIA (10 sequences) all n.s. except unc2; **PIT (14): v6-S above u960 (+0.247 [0.019, 0.634]) and the hybrid (+0.020 [0.001, 0.042])** | no shift stratum reverses the ranking. The one significant loss (AV2 WDC vs native) is against a row with 19× v6-S's px |
| **6** NAVSIM, bottleneck selector (P2) | Does decision exactness hold empirically for an argmax-over-candidates selector with a min-composable (bottleneck) score? | navtest 12,146 tokens / 136 logs. Agreement with π(F) (F = whole frame at the crop scale): **oracle locality 99.71% [99.58, 99.83]** (1.51 rounds, 0.25% at the cap); real crops 96.76% [96.33, 97.15]; references: the cheap draft 94.04%, native whole frame 95.92%. On the 724 tokens where the draft disagrees with π(F): oracle 95.2%, real crops 68.9%, native 72.5%. **All 35 oracle disagreements** have an object of F outside the crops whose single insertion changes π: the 7 × 7 static-vehicle probe grid misses it (no exact sensitivity), not a joint effect. Pairs (1,000 tokens, 769,561 probe pairs with neither single insertion sensitive): **0** joint-only changes | the lemma holds on every tested pair. With oracle locality the loop reaches π(F) except where the finite probe grid misses a sensitive state. With real crops it agrees more often than native whole-frame detection does |

---

## 2. Details per item

### 2.1 Item 1 — episode-level safety and temporal consistency
Files: ep_summary.csv, ep_intervals.csv, ep_top_delays.csv; figure k31_episodes.

**Setup.**
- Source: the stored per-frame actions of the K28 phase-3 run (no re-run). Pre-registered definitions are in STATUS.md.
- Required episodes:
  - Maximal runs of act_gt ≥ L, with gaps of one frame or less merged.
  - Onset delay = the first frame at which the pipeline reaches L.
  - Missed = never reaches L within the episode. The delay is then capped at the episode length.
- False episodes: runs of the pipeline at ≥ L on which act_gt < L throughout.
- Flicker: action changes, and A→B→A reversals within 2 frames.

**Required level.** Rates are missed / mean delay (s) / false per 1,000 / reversals per 100.

| pipeline | nuScenes | AV2 | Argoverse-HD |
|---|---|---|---|
| cheap | 0.347 / 0.61 / 22.9 / 4.71 | 0.111 / 0.17 / 34.8 / 9.57 | 0.169 / 0.11 / 25.1 / 10.92 |
| u960 + C1 + v3a | 0.347 / 0.54 / 6.4 / 1.83 | 0.163 / 0.21 / 7.9 / 2.10 | 0.366 / 0.15 / 5.5 / 0.80 |
| native + C1 + v3a | 0.361 / 0.61 / 4.9 / 1.63 | 0.153 / 0.19 / 8.6 / 2.41 | 0.380 / 0.16 / 7.0 / 1.31 |
| hybrid tsp8@S + C1 + v3a | 0.395 / 0.74 / 4.2 / 1.55 | 0.184 / 0.24 / 6.1 / 1.90 | 0.380 / 0.13 / 6.7 / 1.44 |
| unc2@S + C1 + v3a | 0.429 / 0.80 / 7.2 / 1.92 | 0.163 / 0.26 / 14.9 / 3.93 | 0.394 / 0.19 / 15.9 / 3.93 |
| **v6-S** | 0.408 / 0.74 / **4.0** / **1.38** | 0.189 / 0.22 / 6.5 / 2.07 | 0.380 / 0.14 / 6.3 / 1.20 |
| v6-S without refiner | 0.388 / 0.68 / 6.8 / 1.81 | 0.142 / 0.17 / 11.5 / 2.62 | 0.211 / 0.06 / 6.7 / 0.89 |
| required action (reference) | — / — / — / 0.98 | — / — / — / 0.44 | — / — / — / 0.57 |

**Medians.**
- The median onset delay is 0.5 s (one frame) on nuScenes for every pipeline.
- On AV2 it is 0 s for every pipeline.
- On Argoverse-HD it is 0.033 s for the refined rows and 0 s for cheap and for v6-S without the refiner.
- The differences are therefore in the tail; the mean carries them.

**Ten longest v6-S delays per dataset** (ep_top_delays.csv): units, frames, delays of every other pipeline on the same episode, and the cause.
- The cause comes from the K30 V3 rules on the K30 match records.
- nuScenes:
  - TTC or lateral error: 5
  - range error: 1
  - inside a crop but missed: 2
  - outside every crop: 2
- AV2:
  - TTC or lateral error: 5
  - range error: 2
  - inside a crop but missed: 1
  - outside every crop: 2
- Argoverse-HD:
  - TTC or lateral error: 2
  - range error: 5
  - inside a crop but missed: 3
- The panels (panel_id = <ds>/<unit>/<frame>) come from the capture re-run (k31_capture.sh, done 2026-10-19). They are in
  handoff/K31/panels/: for each of the 10 episodes per dataset there are a draft / BEV / crops panel (_a_draft, _b_bev,
  _c_crops, PDF + PNG), and examples.csv maps each panel to its episode.

### 2.2 Item 2 — consumer adaptivity
Files: consumer_ttc.csv, consumer_idm.csv, idm_mechanism.csv; figure k31_consumers.

**Re-runs and parity.**
- R2a: the hybrid with the TTC support at the TTC loop's per-frame px.
- R2b: the hybrid with the IDM support at K28's S budget, plus field-size logging.
- Parity:
  - The re-run TTC loop c2 equals the stored K28 c2 on every frame (J2 identical on 9,181 / 11,818 / 15,006 frames).
  - The IDM actions of ours-IDM and of v6-S equal K30 V2b on every frame.

**TTC consumer, G2.** Each row's G2 is followed by "loop − row" in brackets.

| row | nuScenes | AV2 | Argoverse-HD |
|---|---|---|---|
| DESP TTC loop (c2 + v3a) | **0.191** | **0.496** | **0.750** |
| hybrid, TTC support, loop's px (×1.0) | 0.157 (+0.034 [0.006, 0.069]) | 0.473 (+0.024 [0.005, 0.045]) | 0.724 (+0.026 [0.010, 0.044]) |
| hybrid, TTC support, K28 S budget | 0.146 (+0.045 n.s.) | 0.427 (+0.070 [0.034, 0.110]) | 0.616 (+0.134 [0.054, 0.241]) |
| unc2 at the loop's px | 0.187 (+0.004 n.s.) | 0.418 (+0.079 [0.048, 0.109]) | 0.673 (+0.077 [0.045, 0.117]) |
| u960 + C1 + v3a | 0.157 (+0.035 n.s.) | 0.448 (+0.049 [0.004, 0.105]) | 0.326 (+0.424 [0.058, 1.040]) |
| native + C1 + v3a | 0.222 (−0.031 n.s.) | 0.500 (−0.003 n.s.) | 0.629 (+0.121 [0.015, 0.327]) |
| frozen v6-S (braking-driven) | 0.153 (+0.038 n.s.) | 0.439 (+0.058 [0.024, 0.097]) | 0.612 (+0.138 [0.060, 0.247]) |

**IDM consumer, G3.**

| row | nuScenes | AV2 | Argoverse-HD |
|---|---|---|---|
| ours-IDM | 0.030 | 0.148 | 0.302 |
| frozen v6-S | **0.086** | **0.156** | **0.313** |
| hybrid, IDM support, ours-IDM's px | 0.026 | 0.165 | 0.295 |
| hybrid, IDM support, K28 S budget | 0.046 | 0.162 | 0.299 |
| unc2 at ours-IDM's px | −0.002 | 0.078 | 0.255 |
| u960 + C1 + v3a | 0.066 | 0.161 | 0.260 |
| native + C1 + v3a | 0.053 | 0.196 | 0.309 |

Significant intervals:
- ours-IDM − v6-S: −0.056 [−0.092, −0.020] (nuScenes); −0.011 [−0.024, −0.002] (Argoverse-HD).
- ours-IDM − unc2: +0.071 (AV2).
- ours-IDM − u960: +0.042 (Argoverse-HD).
- v6-S − hybrid at ours-IDM's px: +0.060 (nuScenes); +0.018 (Argoverse-HD).
- v6-S − unc2: +0.088 / +0.078 / +0.058.
- v6-S − u960: +0.053 (Argoverse-HD).
- All other rows are n.s. (consumer_idm.csv).

**Mechanism:** see the 2b row of §1 (idm_mechanism.csv).
- Of the differing frames that have a decisive IDM object, both loops found it on 75 / 82, 221 / 227 and 189 / 197. They differ in its estimated range or TTC.
- Field size of the first round: ours-IDM 6.8 / 6.7 / 6.4 cells (px 0.033 / 0.026 / 0.020); v6-S 5.4 / 4.9 / 4.9 cells (0.027 / 0.020 / 0.016).

### 2.3 Item 3 — curved-path AEB consumer
Files: curv_dev.csv, curv_test.csv, curv_mechanism.csv; figure k31_curv.

**Setup.**
- κ = yaw rate / max(v, 1 m/s), clipped to ±0.1 1/m.
- Objects are mapped to (s, d) along the constant-curvature path. κ ≈ 0 is the straight controller exactly; the smoke check confirms equality on such frames.
- Field and support come from the curved consumer, substituted into the frozen loop code.
- Rows are as registered. The re-designed curved-tube hybrid follows the same path, 10–70 m, and grows its half-width until the S budget.

**Dev check (passed):** see §1. |κ| p50 / p90 is 0.0013 / 0.032 (nuScenes val) and 0.0007 / 0.017 (AV2 CFG-DEV); κ is at the clip on 1.6% / 2.1% of frames.

**Test, all frames.** G, with "curved v6-S − row" in brackets.

| row | nuScenes | AV2 | Argoverse-HD |
|---|---|---|---|
| curved v6-S | **0.207** | **0.224** | **0.394** |
| u960 + C1 + v3a | 0.101 (+0.106 n.s.) | 0.276 (−0.051 [−0.106, −0.007]) | 0.282 (+0.113 [0.009, 0.310]) |
| native + C1 + v3a | 0.235 (−0.028 n.s.) | 0.225 (−0.001 n.s.) | 0.422 (−0.028 n.s.) |
| hybrid, straight support + tube (K28) | 0.154 (+0.053 n.s.) | 0.209 (+0.016 n.s.) | 0.352 (+0.042 [0.022, 0.064]) |
| hybrid, curved support + tube | 0.186 (+0.022 n.s.) | 0.200 (+0.025 n.s.) | 0.379 (+0.016 n.s.) |
| unc2@S + C1 + v3a | −0.083 (+0.290 [0.030, 0.509]) | 0.168 (+0.057 n.s.) | 0.255 (+0.139 [0.049, 0.255]) |
| extra: frozen straight v6-S, scored curved | 0.227 (−0.019 n.s.) | 0.254 (−0.030 n.s.) | 0.334 (+0.060 [0.031, 0.093]) |

Per |κ| tercile and on frames with the decisive object outside the straight tube: curv_test.csv. Highlights are in §1, and the AV2 losses are explained in §3.

### 2.4 Item 4 — bounded loop (≤ 2 detector calls)
Files: bounded.csv, bounded_parity.csv; figure k31_bounded.

**Code path.**
- k_loop.run_bounded (N = 2) is the K29 / Table 3 logic.
- Within the run, v6-S equals the stored K28 v6-S on every frame.
- VUW vs Jetson agreement: bounded 99.63% / 99.83%, unbounded 99.63% / 99.32%. The bounded loop is not below the unbounded level, so by the registered rule no common-canvas port is needed.

**Numbers:** see §1. The nuScenes loss is one scene (scene-1051), which is not among K29's 20 timed scenes.
- On 8 frames the decisive object is truncated by a crop.
- Its guard-grown crop would only be detected in a third call.
- At the bound the pending detection is dropped while its draft box has already been replaced, so the action stays KEEP where v6-S hard-brakes.

### 2.4b Variant B2K (registered in STATUS before implementation; post-hoc, dev-selected)
Files: b2k_dev.csv; b2k_test.csv after the test run.

**Candidates.** Both keep ≤ 2 calls by construction.
- A: add each pending truncated detection at the bound, lifted like any crop detection.
- B: restore the draft box (IoU ≥ 0.3) with its draft lift.

**Dev selection** (nuScenes new val 152 scenes, AV2 CFG-DEV 300 logs). Values are nuScenes / AV2.

| pipeline | G | px | share at bound | frames ≠ v6-S ≤ 2 |
|---|---|---|---|---|
| v6-S | 0.321 / 0.309 | 0.080 / 0.056 | 0 / 0.0002 | 14 / 52 |
| v6-S ≤ 2 | 0.252 / 0.306 | 0.074 / 0.050 | 0.136 / 0.252 | — |
| **B2K-A** | **0.346 / 0.307** | 0.074 / 0.050 | 0.137 / 0.252 | 17 / 59 |
| B2K-B | 0.316 / 0.286 | 0.074 / 0.050 | 0.136 / 0.252 | 64 / 383 |

Rule:
- Take the higher mean dev G of the two candidates: A, 0.326.
- It must beat the frozen bounded loop's mean (0.279) — it does.
- It must not be below the bounded loop on either split by more than 0.005: +0.094 / +0.001 — it is not.
- So **B2K-A is adopted** and runs once on the three test splits (VUW1, 21:45–22:11 VUW). It is reported as a labelled post-hoc variant next to v6-S ≤ 2; it replaces nothing.

**Test** (b2k_test.csv). Rows are G (px), then B2K-A − row [95%].

| row | nuScenes | AV2 | Argoverse-HD |
|---|---|---|---|
| **B2K-A** | **0.209** (0.073) | **0.293** (0.049) | **0.415** (0.064) |
| v6-S ≤ 2 | 0.091; +0.119 [−0.001, 0.299] | 0.282; +0.011 [−0.003, 0.032] | 0.419; −0.004 [−0.013, 0.003] |
| v6-S | 0.211 (0.078); −0.001 [−0.005, 0.002] | 0.294 (0.054); −0.001 [−0.007, 0.005] | 0.418 (0.071); −0.003 [−0.008, 0.001] |
| u960 + C1 + v3a | 0.112; +0.098 [−0.079, 0.307] | 0.338; −0.046 [−0.104, 0.004] | 0.297; +0.119 [0.013, 0.312] |
| native + C1 + v3a | 0.222; −0.013 [−0.089, 0.120] | 0.294; −0.001 [−0.077, 0.079] | 0.430; −0.015 [−0.051, 0.025] |
| hybrid tsp8@S + C1 + v3a | 0.174; +0.036 [−0.013, 0.096] | 0.296; −0.003 [−0.015, 0.007] | 0.408; +0.007 [−0.003, 0.019] |

Keeping the pending truncated detection at the bound removes the scene-1051 failure without a third call. The bounded loop's significant losses on AV2 (vs u960 and vs the hybrid) are not significant for B2K-A.

### 2.5 Item 5 — distribution-shift strata
Files: strata.csv, strata_units.csv; figure k31_strata.

**Strata are fixed per unit:**
- nuScenes: "night" / "rain" in the scene description (27 night scenes, 50 rain scenes).
- AV2: city, from the public log map file name.
- Argoverse-HD: city, from Argoverse 1.1 city_info. Lighting is not annotated there, so city is the fallback, as registered.

AV2 ATX has only 4 logs and is flagged. All rows are in strata.csv; the significant ones are in §1.

### 2.6 Item 6 — bottleneck selector on NAVSIM (P2)
Files: p2_agreement.csv, p2_pairs.csv, p2_disagreements.csv. Code: scripts/A_navsim/k31_nav_p2.py (md5 bb046934);
pre-registration in STATUS.md.

**Selector π.**
- Candidates (15): the PDM route centreline offset by {0, −1.5, +1.5} m × constant accelerations {+2, +1, 0, −1.5, −3} m/s^2.
- Score s(τ, S) = min(b(τ), min_o h(τ, o)):
  - b = 5 s × normalised progress;
  - h = time of first overlap of the ego rectangle with object o (constant velocity, cap 5 s).
- Field: single probes of a static vehicle on K20's 7 × 7 grid.
- Support: objects with h(τ, o) ≤ s(τ*) for some τ ≠ τ*.
- Crops, guard and merge as spec_pdm; ≤ 3 rounds.
- Scenes follow the K20 / K30 NAVSIM convention (annotated states for matched objects, lifted detections otherwise).

**Agreement with π(F)**, log bootstrap (2,000, seed 0). F = YOLOv8s on the whole frame at the crop scale 0.5 (full960).

| row | all 12,146 tokens | 724 tokens where π(cheap) ≠ π(F) | px |
|---|---|---|---|
| π(cheap draft) | 94.04% [93.08, 94.89] | 0 (by definition) | — |
| π(native whole frame) | 95.92% [95.28, 96.55] | 72.5% [68.5, 76.0] | — |
| DESP, oracle locality | **99.71% [99.58, 99.83]** | **95.2% [93.0, 97.1]** | 0.41 |
| DESP, real crops | 96.76% [96.33, 97.15] | 68.9% [65.0, 72.5] | 0.83 |

- px is the summed crop area including the guard's grown crops, so overlapping crops count twice. It is a cost
  indicator, not an image share.
- Oracle disagreements: 35 of 12,146 tokens. On every one, at least one object of F outside the crops changes π when
  inserted alone into the final merged scene (1 object on 34, 2 on 1), so the field missed a sensitive state:
  - the probe is static and ego-aligned on a finite grid;
  - a moving object, or one between cells, can be sensitive without its cell being sensitive.
  No disagreement is left for a joint effect.
- The real-crop gap to the oracle (96.8% vs 99.7%) is locality: detection on crops differs from detection on the whole
  frame, as on the braking datasets.

**Pairs (lemma check).** The K20 1,000-token subset, the draft scene, every pair of the 45.1 projected probe cells per
token (995,490 pairs).
- 769,561 pairs have neither single insertion sensitive (5.5 sensitive cells per token on average).
- The joint insertion changes π on **0** of them, with π fully re-evaluated (no use of the min structure in the code).

---

## 3. Rows where v6-S (or the tested loop) is significantly below, with mechanism

| item | dataset | row | difference | mechanism |
|---|---|---|---|---|
| 1 missed rate (required) | nuScenes | u960 + C1 + v3a / native + C1 + v3a | +0.061 [0.014, 0.114] / +0.048 [0.006, 0.097] | Later DECELERATE onsets. The refiner raises misses for every refined row (v6-S without it: 0.388). The crop-based hybrid is at the same level (0.395). Of v6-S's 10 worst delays, 8 are lift or detection errors on an object inside its crops and 2 are selection misses (§2.1) |
| 1 mean delay (required) | nuScenes | u960 / native / v6-S without refiner | +0.197 s / +0.136 s / +0.061 s | as above; the delay is in the tail (median 1 frame for all) |
| 1 missed rate (required) | AV2 | cheap / native / v6-S without refiner | +0.079 / +0.037 / +0.047 | cheap starts early because it brakes often: 34.8 false episodes / 1,000 vs v6-S 6.5. The refiner delays onsets (no refiner 0.142) |
| 1 mean delay (required) | AV2 | native / v6-S without refiner | +0.027 s / +0.055 s | refiner |
| 1 missed rate (required) | Argoverse-HD | cheap / v6-S without refiner | +0.211 / +0.169 | refiner: every refined row misses 0.37–0.39 vs 0.17–0.21 unrefined |
| 1 mean delay (required) | Argoverse-HD | v6-S without refiner | +0.079 s | refiner |
| 1 reversals | Argoverse-HD | u960 + C1 + v3a | +0.40 / 100 frames [0.15, 0.67] | not decomposed. A likely source is crop re-selection between consecutive 30 Hz frames, which a whole-frame detector does not have. v6-S's reversals (1.20) are still below native, hybrid, unc2 and cheap |
| 2b | nuScenes / Argoverse-HD | ours-IDM − frozen v6-S | −0.056 / −0.011 | ours-IDM is the tested loop here, not v6-S: same leader found, different lift (refinement through the braking support) |
| 3 | AV2 | curved v6-S − u960 + C1 + v3a (all frames; top \|κ\| tercile; decisive object outside the straight tube) | −0.051 [−0.106, −0.007]; −0.110 [−0.228, −0.012]; −0.117 [−0.241, −0.018] | (a) u960 is also above the straight v6-S on AV2 in K28 (0.338 vs 0.294), because whole-frame detection finds the decisive object more often (decisive object found on 0.954 vs 0.922 of its frames, curv_mechanism.csv). (b) The curvature-specific part is the refiner on the curved support. Without the refiner, curved v6-S beats straight v6-S on all three datasets (cost curved − straight −67 / −57 / −707 J). With the refiner on AV2's top \|κ\| tercile, curved v6-S falls from 0.125 to 0.025, while the straight v6-S stays at 0.123 → 0.120. On the frames where the two v6-S variants differ, the curved action equals the required one far more often (524 vs 98 of 679 on AV2), but its few misses are costlier and outweigh that (+280 J) |
| 3 | Argoverse-HD | curved v6-S − native (tercile 2) | −0.042 [−0.086, −0.004] | native has 15× the px; n.s. overall (−0.028) |
| 4 | nuScenes | v6-S ≤ 2 − v6-S | −0.120 [−0.299, −0.0001] | one scene (scene-1051, §2.4). The dev-selected B2K-A repairs it: −0.001 [−0.005, 0.002] vs v6-S on test (§2.4b) |
| 4 | AV2 | v6-S ≤ 2 − u960 / − hybrid | −0.057 [−0.114, −0.007] / −0.015 [−0.032, −0.0004] | u960 is above v6-S itself on AV2 (K28). Bounding to 2 calls leaves 24% of frames at the bound and costs another 0.012 (n.s.), which turns the tie with the hybrid into a small loss |
| 5 | AV2 WDC (18 logs) | v6-S − native + C1 + v3a | −0.200 [−0.444, −0.006] | not decomposed. native uses 19× v6-S's px, and u960 is also ahead in point estimate (−0.187, n.s.). The stratum has 18 logs, and the interval nearly touches 0 |

**No significant loss of v6-S** in:
- the hard-brake level of item 1;
- item 2a (the TTC loop);
- nuScenes and Argoverse-HD of item 3;
- every other stratum of item 5.

---

## 4. Files
- **Tables** (copies of results/k31/tables on VUW):
  - item 1: ep_summary.csv, ep_intervals.csv, ep_top_delays.csv;
  - item 2: consumer_ttc.csv, consumer_idm.csv, idm_mechanism.csv;
  - item 3: curv_dev.csv, curv_test.csv, curv_mechanism.csv (descriptive, added for §3);
  - item 4: bounded.csv, bounded_parity.csv, jetson_b2_actions.csv;
  - B2K: b2k_dev.csv, b2k_test.csv;
  - item 5: strata.csv, strata_units.csv.
- **Figures** (PDF + PNG): k31_episodes, k31_consumers, k31_curv, k31_bounded, k31_strata.
- **Per-frame records:** results/k31/r2 (R2a / R2b), r3 (curved), r4 (bounded, B2K), p2 / p2_pairs (item 6).
- **Item 6 tables:** p2_agreement.csv, p2_pairs.csv, p2_disagreements.csv. **Item 1 panels:** panels/.
- **Code:** scripts/A_method/k31_*.py (k31_eval.py md5 c18a782f); local copies in remote/A_method/; figures from scripts/69_k31_figs.py.
- **Pre-registration and operational log:** STATUS.md.
