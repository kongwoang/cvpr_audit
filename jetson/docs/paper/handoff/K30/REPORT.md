# K30 REPORT — post-hoc analyses of the frozen v6-S and the K28 Table 1 baselines

**Status: DRAFT 2026-10-16 (VUW 11:40).** Final: V1, V2, V2b, V3 (taxonomy, causes, panels), V4 official metrics, V5, V6 (ii), V7, V8c, V8d, V8e, V8g. **All runs finished 2026-10-17 16:14 VUW ("K30 ALL DONE").** Not started: J (Jetson, after K29 and the new power supply), V8h (P2). Blocked: V8f (needs the user's dataset registration).

**Everything in this report is post-hoc.** v6-S is frozen (method_freeze_draft §2n, R1). No number here selected anything. Baselines are in their K28 forms (R2): refiner v3a + C1 for every pipeline; truncation guard and own history for the crop heuristics. Prior works keep their official settings. Intervals: paired unit bootstrap, 2,000 resamples, seed 0 (units = scenes / logs / sequences / routes). "v6-S − X" > 0 means v6-S is better. "Significant" means the 95% interval excludes 0.

Every comparison row is in results/k30/tables/ on VUW and in results/vuw_summaries/k30/. **Every row where v6-S is not significantly above is in not_above_all.csv** (§3 groups them, each group with its mechanism).

---

## 1. Item → numbers the paper needs

| item | question | numbers for the paper | verdict |
|---|---|---|---|
| **V1a** frame-matched budget | Is v6-S's gain only from the budget? Heuristics capped at m × v6-S's px per frame | nuScenes, m = 1: v6-S **0.211**; hybrid tsp8 0.174 (+0.037 [−0.012, 0.096]); tsp2 0.114 (+0.097 [−0.005, 0.234]); the six others: v6-S above (+0.043 … +0.280). AV2, m = 1: v6-S **0.294**; tsp8 0.296 (−0.003 [−0.014, 0.008]); tsp2 n.s.; six others: v6-S above (+0.032 … +0.105). AV2, m = 1.25 / 1.5: **tsp8 above v6-S** (−0.039 [−0.078, −0.008], −0.035 [−0.076, −0.005]) at 1.8 / 2.0 × v6-S's realised px. Argoverse-HD, m = 1: v6-S **0.418**; tsp8 0.409 (+0.009 [−0.002, 0.022]), tsp2 +0.014, cor8 +0.025 n.s.; five others: v6-S above (+0.041 … +0.138). Argoverse-HD, m = 1.5: **cor0 above v6-S** (−0.072 [−0.131, −0.019]) at 2.1 × px | v6-S ≥ every heuristic at m ≤ 1; the hybrid ties it at the same cap; two heuristics overtake it when they spend ~2 × the pixels |
| V1a px accounting | Do the heuristics use the cap? | Realised px of the heuristics at m = 1: 1.2–1.4 × v6-S's (nuScenes), 1.2–1.6 × (AV2), 1.4–1.8 × (Argoverse-HD), because the guard grows their crops past the cap | the matched comparison favours the heuristics, not v6-S |
| **V1b** pixel axis | Is (960, context 2) a lucky point? | No point of the 3 × 3 grid differs from v6-S on nuScenes (−0.037 … +0.020). AV2: (1152, 3.0) +0.017 [0.004, 0.031] at 1.44 × px. Argoverse-HD: the three 1280-px points +0.009 … +0.011 at 1.4–1.8 × px | larger crops buy small gains with more pixels; no cheaper point beats v6-S |
| **V1c** TTC consumer support | Did the hybrid lose in K28's TTC row only because it used the braking support? | nuScenes G2: v6-S 0.153; hybrid with the TTC consumer's own support 0.146 / 0.145 (S / v6-S budget; +0.007 / +0.008, n.s.); with braking support 0.137. AV2: v6-S 0.439; TTC support 0.427 / 0.426 (+0.012 [−0.003, 0.028], +0.013 [−0.002, 0.029]). Argoverse-HD: v6-S 0.612; braking support 0.600 / 0.599 (**v6-S above, +0.012 [0.001, 0.023]**), TTC support 0.616 / 0.616 (−0.005 [−0.022, 0.011]) | with the consumer's own support the hybrid ties v6-S everywhere; with the braking support it is below on Argoverse-HD |
| **V2** strata, tube | Where does the field beat a fixed tube? | nuScenes, v6-S − hybrid tsp8 @ v6-S px: low |yaw| tercile **+0.256 [0.005, 0.434]**, KEEP frames +0.012 [0.001, 0.025]; all other strata n.s. AV2: no stratum significant (−0.017 … +0.015). Argoverse-HD: frames without a decisive object +0.022 [0.008, 0.035], the rest n.s. Share of v6-S's field footprint outside the hybrid's tube: 4.6% / 10.3% / **20.8%** (nuScenes / AV2 / Argoverse-HD); on frames whose decisive object is outside the tube: 0.033 vs 0.026, 0.185 vs 0.176 (n.s.), **0.455 vs 0.428, +0.027 [0.008, 0.052]** | the field's advantage over a tube is small; it is significant where the field reaches outside the tube most (Argoverse-HD) |
| **V2b** IDM consumer | Does the method transfer to a leader-following consumer? | Frozen v6-S under IDMFollow: nuScenes 0.086 (above ours-IDM +0.056, hybrid +0.060, unc2 +0.088); AV2 0.156 (all n.s.; native + C1 + v3a 0.196, −0.040 [−0.105, 0.010]); Argoverse-HD 0.313 (above u960 + C1 + v3a +0.053, ours-IDM +0.011, unc2 +0.058, hybrid +0.018 [0.0003, 0.044]) | the braking-driven v6-S transfers; re-targeting the field to IDM (ours-IDM) does not help |
| **V3** disagreement taxonomy | When v6-S and F = u960 + C1 disagree, who is right? | Disagreement on 3.7% / 12.5% / 16.0% of frames (nuScenes / AV2 / Argoverse-HD); v6-S equals the required action on 70% / 81% / 73% of them, F on 27% / 14% / 21%. Without the refiner: 2.4% / 3.2% / 6.0%. Causes of the 20 costliest safety-critical v6-S frames (nuScenes / AV2 / Argoverse-HD): object found, TTC / lateral estimate too low 10 / 16 / 12; object inside a crop but missed 6 / 1 / 4; found, range too long 2 / 3 / 3; object outside every crop 2 / 0 / 1 | most disagreements are the refiner's; v6-S is usually right when they disagree; its worst frames are lift errors, not selection |
| **V4** Bench2Drive, official metrics | Does v6-S change closed-loop driving? | 219 routes × 3 runs. DS: v6-S 92.5, cheap 92.3, u960 93.1, u1600 92.0, v3 92.8, privileged 96.6; v6-S − cheap +0.27 [−0.96, 1.48]. Multi-Ability mean: v6-S 0.833, cheap 0.830, u960 0.855; Overtaking v6-S − u960 −0.096 [−0.188, −0.015]. Route flips: 5 routes v6-S completes and cheap fails (≥ 2 of 3 runs), 6 the other way. **In-loop (4th run with logging, 212–214 matched routes per branch; paired route bootstrap): hard brakes (brake ≥ 0.5 rising edges) per km v6-S 136 vs cheap 155 (−19.0 [−28.4, −9.8]), vs u1600 189 (−53.6 [−78.1, −32.5]), vs u960 138 (n.s.); phantom-caused (unnecessary) hard brakes per km 9.3 vs cheap 11.3 (−2.0 [−4.1, −0.2]), vs u1600 16.8 (−7.6 [−13.2, −3.3]), vs u960 8.6 (n.s.); recall of in-camera corridor actors within 3 s TTC 0.906 vs cheap 0.889 / u960 0.845 / v3 0.953 (all n.s.); DS of this run 93.2 (cheap 92.5, u960 92.7)** | closed-loop outcome ties; in the loop v6-S brakes less and less often for phantoms than cheap and u1600, without losing recall |
| **V5a** loss weights | Is the ranking an artefact of the cost weights? | 27 weight settings (λ_brake, λ_collision, λ_jerk ∈ {0.5, 1, 2} ×): v6-S rank 2 of 10 in all 27 on nuScenes and Argoverse-HD, 3–4 on AV2. u960 + C1 + v3a is significantly above v6-S on AV2 in 9 of 27 settings (never elsewhere). vs the best prior work: AV2 FOVEA, v6-S above in 27 / 27 (+0.13 … +0.50); nuScenes SAHI s1.0, v6-S above in the 9 λ_brake × 2 settings, n.s. in the other 18 (SAHI ahead in point estimate when λ_brake × 0.5) | ranking stable; the AV2 whole-frame lead and the nuScenes SAHI tie are the exceptions |
| **V5b** consumer parameters | τ ∈ {0.8, 1.2}, reaction time ∈ {0.3, 0.6} s | v6-S rank (of 9): nuScenes 2 (1 at tr 0.6), AV2 3–4, Argoverse-HD 2. Significant losses: AV2 τ 0.8, u960 + C1 + v3a −0.049 [−0.104, −0.003]. v6-S stays above unc2 / unc4 in every setting and dataset | stable |
| **V5c** probe assumptions | Probe size / class | Probe − frozen v6-S: nuScenes vc10 **+0.026 [0.0003, 0.066]** (barely significant), vc5 +0.025 n.s., pedestrian probe −0.286, truck / union −0.006 n.s.; AV2 all within ±0.01 except pedestrian −0.039; Argoverse-HD vc5 / vc10 −0.034, pedestrian −0.081, truck −0.009 | the frozen vehicle probe is near-best; a larger closing-speed probe is marginally better on nuScenes |
| **V6 (ii)** probing cost | Measured (not derived) time of cheaper probe grids | Wall clock relative to 7 × 7 on the same 1,000 tokens: 5 × 5 **0.51 [0.47, 0.56]**, 3 × 3 **0.32 [0.28, 0.37]**, plan corridor **0.33 [0.30, 0.37]**; calls 50.2 / 26.2 / 14.0 / 14.4 per frame. Absolute time inflated by host load (7 × 7 14.8 s here vs 6.2 s in K20) | K20's derived ratios (calls × mean call) are within 0.05 of the measured ones |
| **V6 (i)** NAVSIM hybrid | Does a tube hybrid match the probing field for PDM-Closed? | PDMS (v1, 12,146 navtest tokens): spec_pdm **0.8945**, hybrid_pdm (support + route tube at spec_pdm's crop area) 0.8943 (spec_pdm − hybrid [−0.0008, +0.0015]), native whole frame 0.8952 ([−0.0026, +0.0012]) | tie: PDMS does not separate field, tube or native |
| **V7** detection metrics | What do the crops do to standard detection numbers? | nuScenes (every 4th frame): AP50 cheap 0.148, v6-S 0.162, u960 + C1 0.246; relevant-object precision v6-S **0.799** (highest), recall 0.824 (u960 0.859); spurious brakes / 1k frames v6-S 3.8 vs u960 7.6. AV2: AP50 v6-S 0.115 vs u960 0.178; relevant precision 0.719 (u960 0.677), recall 0.901 (u960 0.909); spurious 5.6 vs 8.7. Argoverse-HD: AP50 0.130 vs 0.199; relevant precision **0.511** (highest; u960 0.410, native 0.456), recall 0.890 (0.909 / 0.923); spurious 15.0 vs 44.3 (native 16.2) | v6-S trades COCO-style AP for decision-relevant precision |
| **V8a** NAVSIM v2 EPDMS | navtest / navhard | navhard two-stage EPDMS (devkit aggregation, 222 groups; the same 100 synthetic tokens fail in the devkit's IDM for every branch and are left out): reference 0.528, cheap 0.508, full960 0.508, native 0.510, **spec_pdm 0.509**, hybrid 0.510 (no intervals). navtest one-stage EPDMS (12,146 tokens; spec_pdm − row, paired log bootstrap): spec_pdm **0.8647**, cheap 0.8624 (+0.0023 [−0.0001, 0.0048]), full960 0.8644 (+0.0003 n.s.), native 0.8644 (+0.0003 n.s.), hybrid (fixed tube) 0.8653 (−0.0006 [−0.0015, 0.0004]), reference 0.8698 (−0.0051 [−0.0068, −0.0035]) | tie on both: EPDMS does not separate the perception branches (only the privileged reference is above) |
| **V8c** KITTI (6 test seq.) | 2D AP and DEEP's cells with the right refiner flag | Decision cells (flag 0.0, 3,536 frames), mean of 4: v6-S **0.323** (px 0.167); native + T4 + ref 0.448 (**−0.126 [−0.175, −0.021]**); native + C1 + ref 0.439 (−0.116); u960 + T4 + ref 0.368 (−0.046 n.s.); native plain 0.327 (n.s.). 2D AP R40, moderate: Car cheap 51.6 / u960 75.0 / native 75.3 / **v6-S 42.9**; Pedestrian 56.4 / 73.6 / 76.0 / 53.9. Cyclist ≈ 0 for all (COCO has no cyclist box) | v6-S is below whole frame + components on KITTI; its AP is below even the draft (crop-truncated boxes, §3) |
| **V8d** ego speed × {0.5, 1.5} | Does the field adapt to speed? | nuScenes: × 0.5 all n.s. (v6-S 0.241; native + C1 + v3a 0.318, −0.077 [−0.156, 0.032]); × 1.5 v6-S 0.196 **above the hybrid +0.043 [0.003, 0.087]**. Field: farthest sensitive range 14 / 28 / 43 m, crop area 0.017 / 0.056 / 0.065 at × 0.5 / 1 / 1.5 | the field scales with speed; its edge over the tube appears at high speed |
| **V8e** corner cases | Non-COCO objects in the corridor | Recall of in-corridor cones / barriers / barrels / bollards ≈ 0.01–0.04 for every pipeline (construction vehicles 0.22–0.58). G on those frames: nuScenes (373) all n.s.; **AV2 (533): hybrid above v6-S, −0.045 [−0.107, −0.009]** | the COCO detector cannot see these classes; selection cannot recover them |
| **V8g** CODA | CODA images inside nuScenes new test | 27 frames, 119 objects, 4 in the corridor: too few for a conclusion (recall 0.02–0.08) | not usable |
| **K30-DEV** (dev only, pre-registered) | Can simple changes improve ours? | nuScenes val / AV2 CFG-DEV, G(candidate) − G(v6-S): conservative refiner v3a_n15 −0.0004 / −0.031, v3a_n30 −0.0002 / −0.033, v3a_hb −0.0006 / −0.053, no refiner −0.008 / −0.033; closing-speed probes vc5 −0.001 / +0.008, vc10 −0.002 / +0.009 (n.s.) | none qualifies (+0.005 macro bar): v6-S stays |
| V8b / V8f / V8h | — | V8b decided against (STATUS); V8f needs the user's registration; V8h P2, not started | — |
| **J1** heuristics on the Jetson | Latency-G of the heuristics + C1 + refiner with the fast path (K29 trt32), at the S and ×1.0 budgets | ×1.0 budget, mean / p95 ms (3 reps), v6-S − row. nuScenes test 20: v6-S **0.380 at 21.5 / 57.7**; hybrid tsp8 0.373 at 24.0 / 75.0 (+0.007 [−0.011, +0.029]); tsp2 0.347 at 22.3 / 69.6 (+0.033 n.s.); tsp8-YOLO11m 0.392 at 28.2 / 89.2 (−0.012 n.s.); cor8 0.267 at 23.8 / 72.9; cor0 −0.316 at 21.0 / 63.0. AV2: v6-S **0.419 at 39.9 / 89.5**; hybrid 0.417 at 44.5 / 109.0 (+0.003 [0.000, +0.007]); tsp2 0.431 at 40.1 / 100.2 (**−0.012 [−0.029, −0.000]**); tsp8-YOLO11m 0.441 at 56.2 / 143.4 (−0.021 [−0.035, 0.000]); cor8 0.408 at 36.8 / 81.3 (+0.011 n.s.). Heuristics use 1.2–1.5 × v6-S's px (guard growth, as V1a) | nuScenes: v6-S has the lowest p95 of every heuristic + C1 row, and only cor0 is faster in mean (by 0.5 ms). AV2: only cor0 / cor8 are faster (lower G, n.s.). G ties every heuristic except tsp2 on AV2, which is above v6-S by 0.012 at the same mean and an 11 ms higher p95 |
| **J2** energy | Energy per frame on the Jetson (module rails GPU + CPU + SOC + CV + VDDRQ, 1 s means; one run per pipeline) | Energy above the cheap run (J / frame), nuScenes test 20 / AV2: v6-S **0.29 / 0.54**; v6-S ≤ 2 0.26 / 0.51; v5.1-S 0.25 / 0.43; hybrid tsp8 at ×1.0 (its own) 0.28 / 0.62; whole frame + C1 + v3a u448 0.20 / 0.41, u960 0.54 / 0.76, u1280 0.78 / 1.12, native 1.12 / 2.39 | v6-S needs 54% / 71% of the energy of u960 + C1 + v3a and 26% / 22% of native + C1 + v3a (nuScenes / AV2); on par with the hybrid, more than u448 + C1 + v3a; one run each, no intervals |
| **J3** Argoverse-HD streaming | sAP with the fast path (K28: 0.064) | sAP / corridor streaming AP50: v6-S **0.099 / 0.379** (processes 82% of frames, 36.5 ms mean, 69 ms mean staleness); v6-S ≤ 2 0.104 / 0.393; cheap 0.097 / 0.358; u960 + C1 + v3a 0.198 / 0.441; u1280 + C1 + v3a 0.184 / 0.468; native + C1 + v3a 0.144 / 0.434 (K7 whole frame u1280 without C1: sAP 0.247) | the fast path raises v6-S's sAP by half, but it stays at the cheap detector's level and below every whole frame + C1 + v3a row, in full sAP and in the corridor |

---

## 2. Details per item

### V1a — frame-matched budget (k30_eval.py part match; v1a_matched.csv; figure k30_v1a_matched)
- **Setup.** Per frame, each of the 8 S heuristics + C1 + refiner v3a gets a cap of m × (cheap px + v6-S's crop px on that frame) − cheap px, m ∈ {0.75, 1, 1.25, 1.5}. K28's heuristics were matched to v5.1-S's px instead.
- **Realised px exceed the cap**: the truncation guard grows crops after selection. At m = 1 the heuristics use 1.2–1.4 × v6-S's px on nuScenes and 1.2–1.6 × on AV2 (px_ratio_to_v6S). So "matched" is generous to them.
- **nuScenes** (refiner v3a):

  | m | tsp8 (hybrid) | tsp2 | tcr3 | best of unc / cor | 
  |---|---|---|---|---|
  | 0.75 | +0.148 [0.014, 0.307] | +0.221 | +0.099 | v6-S above all |
  | 1.0 | +0.037 [−0.012, 0.096] | +0.097 [−0.005, 0.234] | +0.043 [0.014, 0.082] | v6-S above all |
  | 1.25 | +0.098 [−0.007, 0.246] | +0.093 n.s. | +0.040 [0.000, 0.074] | v6-S above all |
  | 1.5 | +0.080 n.s. | +0.091 n.s. | −0.004 [−0.051, 0.049] | n.s. (cor0, cor8, tcr0) / v6-S above (unc) |

- **AV2**: v6-S above all eight at m = 0.75; at m = 1 above six (tsp2 +0.007, tsp8 −0.003 n.s.); at m = 1.25 / 1.5 the hybrid is above v6-S (−0.039 [−0.078, −0.008] at 1.84 × px; −0.035 [−0.076, −0.005] at 2.02 × px); tcr0 / tcr3 / cor8 / tsp2 n.s. (and cor0 at 1.5); unc2 / unc4 below v6-S.
- **Without the refiner** (m = 1): nuScenes v6-S 0.181; tcr3 / tsp2 / tsp8 n.s.; others v6-S above. AV2 v6-S 0.241: only cor0 / cor8 significantly below v6-S; the rest n.s.
- **Argoverse-HD**: v6-S above all eight at m = 0.75; at m = 1 above five (cor8 +0.025, tsp2 +0.014, tsp8 +0.009, n.s.); at m = 1.25 cor0 / cor8 / tsp2 / tsp8 n.s.; at m = 1.5 cor0 is above v6-S (−0.072 [−0.131, −0.019], 2.10 × px), cor8 / tsp2 / tsp8 n.s. Without the refiner at m = 1, v6-S (0.422) is above all eight.

### V1b — pixel axis (part axis; v1b_axis.csv; figure k30_v1b_axis)
- 3 × 3 grid: crop long side {960, 1152, 1280} × support context {2, 2.5, 3}; v6-S = (960, 2).
- nuScenes: −0.037 … +0.020, none significant.
- AV2: (1152, 3.0) +0.017 [0.004, 0.031] at px 0.078 vs 0.054; the rest n.s.
- Argoverse-HD: (1280, 2 / 2.5 / 3) +0.009 [0.002, 0.016] / +0.011 [0.004, 0.019] / +0.011 [0.002, 0.019] at px 0.100–0.126 vs 0.070.
- The hybrid's points from part match are drawn on the same plane: at the same px the hybrid sits at or below v6-S.

### V1c — TTC-warning consumer with its own support (part match; v1c_ttc.csv)
- K28's consumer-2 row gave the hybrid the braking consumer's support. Here the hybrid also gets the TTC consumer's support (IE.TTCWarn.support), at K28's S budget and at v6-S's budget.
- nuScenes: hybrid 0.137 → 0.146 / 0.145 with the TTC support; v6-S 0.153 (+0.007 / +0.008 n.s.).
- AV2: 0.436 → 0.427 / 0.426 (the braking support is better here); v6-S 0.439 (+0.012 / +0.013 n.s.).
- Argoverse-HD: 0.600 → 0.616 / 0.616 with the TTC support; v6-S 0.612 (+0.012 [0.001, 0.023] over the braking-support hybrid; −0.005 n.s. against the TTC-support one).

### V2 — strata and the tube (v2_strata.csv, v2_tube.csv)
- Strata: ego-speed tercile, |yaw rate| tercile, required action, decisive-object range bin. Rows: v6-S, hybrid tsp8 @ v6-S px, ego corridor cor8 / cor0 @ v6-S px.
- **nuScenes**, v6-S − hybrid: low |yaw| +0.256 [0.005, 0.434]; KEEP +0.012 [0.001, 0.025]; high speed +0.072 [−0.009, 0.150]; HARD_BRAKE +0.061 [−0.042, 0.147]; < 20 m +0.063 n.s.; > 60 m −0.084 [−0.217, 0.041].
- **AV2**: every stratum within −0.017 … +0.015, n.s.
- **Argoverse-HD**: no decisive object +0.022 [0.008, 0.035]; low |yaw| +0.027 [−0.000, 0.057]; HARD_BRAKE (96 frames) −0.029 [−0.148, 0.040]; the rest within ±0.02, n.s.
- **Field vs tube**: 4.6% (nuScenes) / 10.3% (AV2) / 20.8% (Argoverse-HD) of v6-S's field footprint (area) lies outside the hybrid's tube. Frames whose decisive object lies outside the tube: nuScenes 1,583 (G 0.033 vs 0.026; decisive recall 0.757 vs 0.747; n.s.), AV2 3,397 (0.185 vs 0.176; 0.875 vs 0.872; n.s.), **Argoverse-HD 3,470 (0.455 vs 0.428, +0.027 [0.008, 0.052]; 0.859 vs 0.842)**.

### V2b — leader-following consumer (pre-registered; part idm; v2b_idm.csv)
- IDMFollow consumer, scored with the same cost. Rows: u960 / native + C1 + v3a, ours driven by IDMFollow (field and support re-targeted to the IDM demand), the frozen braking-driven v6-S, the hybrid with IDM support at ours-IDM's px, unc2 at ours-IDM's px.
- Frozen v6-S: nuScenes 0.086, AV2 0.156, Argoverse-HD 0.313. ours-IDM: 0.030 / 0.148 / 0.302.
- Frozen v6-S − row: nuScenes ours-IDM +0.056 [0.020, 0.092], hybrid +0.060 [0.006, 0.111], unc2 +0.088 [0.021, 0.162], whole frames n.s.; AV2 all n.s. except unc2 +0.078; Argoverse-HD ours-IDM +0.011 [0.001, 0.024], u960 +0.053 [0.016, 0.110], unc2 +0.058 [0.006, 0.122], hybrid +0.018 [0.0003, 0.044], native n.s.

### V3 — disagreement taxonomy and costly frames (k30_stored.py; v3_taxonomy.csv, v3_top_frames.csv, v3_causes.csv)
- Against F = u960 + C1 (+ v3a where X has it):

  | dataset | v6-S disagrees | of these: v6-S = required | F = required | without refiner: disagrees |
  |---|---|---|---|---|
  | nuScenes | 3.7% | 70% | 27% | 2.4% |
  | AV2 | 12.5% | 81% | 14% | 3.2% |
  | Argoverse-HD | 16.0% | 73% | 21% | 6.0% |

- v6-S's disagreements are mostly "lower action" (1.0% higher / 2.7% lower on nuScenes; 0.7% / 11.8% AV2): the refiner lowers range errors that make F over-brake.
- On safety-critical disagreements, v6-S loses more than F: + 75 / 71 / 19 cost per 1,000 frames (nuScenes / AV2 / Argoverse-HD).
- Causes of the 20 costliest safety-critical v6-S frames (nuScenes / AV2 / Argoverse-HD): decisive object found but its TTC / lateral estimate keeps the demand low 10 / 16 / 12; found, range too long 2 / 3 / 3; inside a crop but missed 6 / 1 / 4; outside every crop 2 / 0 / 1. So 46 of 60 are lift errors on a found object, 11 detector misses inside a crop, 3 selection misses.

### V4 — Bench2Drive official metrics (k30_b2d.py; b2d_*.csv)
- From the three finished runs (K6 / K28: full, rep2, rep3), 219 routes (11755 excluded as in K6), official tools (ability_benchmark, efficiency_smoothness_benchmark).

  | branch | DS | success | Multi-Ability mean | efficiency | smoothness |
  |---|---|---|---|---|---|
  | privileged | 96.6 | 0.915 | 0.911 | 247.5 | 0.223 |
  | cheap | 92.3 | 0.834 | 0.830 | 243.3 | 0.231 |
  | u960 | 93.1 | 0.857 | 0.855 | 241.5 | 0.243 |
  | u1600 | 92.0 | 0.837 | 0.839 | 242.4 | 0.248 |
  | ours (v3) | 92.8 | 0.845 | 0.837 | 242.2 | 0.228 |
  | **ours_v6 (v6-S)** | **92.5** | **0.839** | **0.833** | 242.4 | 0.234 |

- v6-S − cheap: DS +0.27 [−0.96, 1.48], success +0.005 [−0.020, 0.029], ability n.s. everywhere; ego hard-brake proxy −2.2 / km [−4.1, −0.3] (proxy: entries into a < −4.05 m/s², dominated by artefacts, ≈ 60 / km for every branch).
- v6-S − u960: Overtaking −0.096 [−0.188, −0.015]; the rest n.s.
- **In-loop metrics (pre-registered; k30_b2d_inloop.py, inloop_summary.csv, inloop_per_route.csv).** Fourth run with the logging agent (perception_pilot_log.py; driving unchanged), 6 branches × 219 routes on VUW + VUW1; 212–214 routes per branch matched to their in-loop records, none ended in a crash. Numbers in §1. Hard-brake rising edges over-count (the controller toggles the brake), so the absolute rates (≈ 100–190 / km) are not physical; the paired differences are. The sensitivity count (episodes > 1 s apart) is n.s. for v6-S vs every perception branch (e.g. unnecessary episodes vs cheap +0.30 [−0.21, 0.82]).

### V5 — evaluation robustness (k30_stored.py, k30_eval.py part cons, k30_v5_prior.py; v5a_loss_weights.csv, v5bc_consumer.csv, v5_prior.csv)
- **V5a** (27 weight settings, stored actions re-scored; base check max |ΔJ| = 0): v6-S rank 2 of 10 on nuScenes and Argoverse-HD in every setting, 3–4 on AV2. Significant losses: AV2, u960 + C1 + v3a in 9 / 27 settings.
- **Prior works** (re-runs with parity: nuScenes SAHI s1.0 J identical on 9,181 / 9,181 frames; AV2 FOVEA KDE_I 960 on 11,818 / 11,818): AV2 v6-S above FOVEA in every V5a / V5b setting (+0.13 … +0.50, lowest bound +0.03). nuScenes: v6-S above SAHI when λ_brake × 2 (+0.25 … +0.27); n.s. otherwise; SAHI ahead in point estimate when λ_brake × 0.5 (−0.03 … −0.05, n.s.). Base: +0.081 [−0.084, 0.274].
- **V5b** (τ 0.8 / 1.2, reaction time 0.3 / 0.6 s): ranks as in §1; v6-S above v5.1-S everywhere except nuScenes τ 1.2 (+0.007 n.s.) and AV2 τ 1.2 (+0.011 n.s.).
- **V5c** (probe assumptions, v6-S only): §1.

### V6 — NAVSIM (k30_probe_time.py, k30_probe_an.py; v6ii_probe_time.csv; figure k30_v6ii_probe)
- **(ii) measured probing cost.** Table in §1. Planner share of the wall clock 88% (7 × 7) to 78% (3 × 3). The probes that move the plan: 15% of calls (7 × 7), 46% in the plan corridor. K20's derived ratios (calls × mean frozen call / 7 × 7) are 0.52 / 0.28 / 0.29 against the measured 0.51 / 0.32 / 0.33: the fixed per-token cost (planner state, support calls) weighs more in the small grids.
- **(i) hybrid for PDM-Closed.** PDMS: spec_pdm 0.8945, hybrid_pdm 0.8943, native 0.8952, all n.s. (pdms_summary_k30m.csv; the faulty-tube hybrid scored 0.8942). **Bug found and fixed (2026-10-16):** the first run placed the tube at y = interp(x) over all route-centerline points with 0 ≤ x ≤ 100 m sorted by x, which mixes the route ahead with later parts of a turning route (39% of navtest tokens had a tube point > 1 m off the route ahead). The fixed tube follows the route ahead only (k30_nav_hybrid.forward_path). Parity on one log: 78 / 78 straight-route tokens identical, 54 / 54 turning tokens changed. The old hybrid_pdm rows are kept as hybrid_pdm_tubebug.

### V7 — detection metrics (k30_eval2.py part apdump, k30_v7.py; v7_detection.csv)
- Outputs: "sc" (final scene, conf ≥ 0.25) on nuScenes, whose cached draft has no low-confidence boxes; "ap" (conf ≥ 0.1, K7 definition) on AV2 / Argoverse-HD; AV2 predictions in the ego-mask band removed.

  | dataset | pipeline | AP | AP50 | corridor recall50 | relevant recall | relevant precision | spurious brakes / 1k |
  |---|---|---|---|---|---|---|---|
  | nuScenes | cheap | 0.057 | 0.148 | 0.544 | 0.759 | 0.505 | 16.6 |
  | | v6-S | 0.064 | 0.162 | 0.615 | 0.824 | **0.799** | **3.8** |
  | | hybrid tsp8@S | 0.067 | 0.170 | 0.608 | 0.821 | 0.780 | 3.8 |
  | | u960 + C1 | 0.094 | 0.246 | 0.716 | 0.859 | 0.665 | 7.6 |
  | | native + C1 | 0.092 | 0.245 | 0.714 | 0.848 | 0.731 | 4.2 |
  | AV2 | cheap | 0.041 | 0.107 | 0.529 | 0.804 | 0.466 | 16.3 |
  | | v6-S | 0.039 | 0.115 | 0.585 | 0.901 | **0.719** | 5.6 |
  | | hybrid tsp8@S | 0.042 | 0.119 | 0.589 | 0.900 | 0.714 | 5.9 |
  | | u960 + C1 | 0.064 | 0.178 | 0.632 | 0.909 | 0.677 | 8.7 |
  | | native + C1 | 0.065 | 0.184 | 0.650 | 0.919 | 0.665 | 14.8 |
  | Argoverse-HD | cheap | 0.041 | 0.117 | 0.482 | 0.664 | 0.312 | 27.7 |
  | | v6-S | 0.042 | 0.130 | 0.559 | 0.890 | **0.511** | 15.0 |
  | | hybrid tsp8@S | 0.038 | 0.118 | 0.560 | 0.883 | 0.491 | 14.7 |
  | | u960 + C1 | 0.065 | 0.199 | 0.608 | 0.909 | 0.410 | 44.3 |
  | | native + C1 | 0.065 | 0.202 | 0.617 | 0.923 | 0.456 | 16.2 |

- Whole frames (u960 + C1) lead AP / AP50 / corridor recall everywhere (1.5–1.6 × v6-S's AP50) and far recall (50–80 m: nuScenes 0.41 vs 0.24, AV2 0.32 vs 0.22, Argoverse-HD 0.34 vs 0.24). v6-S has the highest precision on decision-relevant objects on all three datasets, and among the fewest spurious brakes (nuScenes 3.8, tied with the hybrid; AV2 5.6 vs the hybrid's 5.9; Argoverse-HD 15.0 vs 14.7).

### V8 — extra benchmarks (pre-registered in STATUS)
- **V8a NAVSIM v2 EPDMS** (k30_nav_v2agent, devkit scorers; navtest_summary.csv, navhard_summary.csv): §1. navhard: the devkit's own aggregation crashes on failed scenarios (100 synthetic tokens fail in its IDM traffic agents for every branch, the reference included); k30_navhard_score.py keeps the devkit's scoring and aggregation and leaves those tokens out (222 groups).
- **V8c KITTI** (k30_kitti.py on VUW, torch; k30_kitti_an.py; v8c_*.csv; figure k30_v8c_kitti).
  - Parity with K28's Jetson table (flag 1.0, the same 3,336 frames): within 0.005 on every row.
  - With refiner flag 0.0 (as v3a was trained), Planner B mono comes back (v6-S 0.355, native + T4 + ref 0.381; ≈ 0 with flag 1.0). This closes the K28 open item.
  - Corrected decision table: §1. AP: §1 and v8c_kitti_ap.csv (easy / moderate / hard; sequence-bootstrap intervals with thresholds snapped to a 0.0025 grid, within 0.2 AP of the exact devkit value).
- **V8d ego speed** (k30_eval2.py part speed; v8d_speed.csv; figure k30_v8d_speed): §1. Safety-critical frames 301 / 928 / 2,016 at × 0.5 / 1 / 1.5. × 1 reproduces the base rows exactly.
- **V8e corner cases** (k30_corner.py, k30_v8eg.py): nuScenes 730 in-corridor objects on 373 frames; AV2 1,718 on 533 frames. Recall (class-agnostic IoU ≥ 0.3): nuScenes 0.032–0.040, AV2 0.008–0.013.
- **V8g CODA** (base-val-1500 ∩ nuScenes new test): 27 images.

### J — Jetson AGX Xavier (K29 fast path trt32; scripts/67_k30_jetson.py, 68_k30_j_tables.py; J/)
- **Window.** 2026-10-18 06:36–09:13 +07, in the exclusive window right after K29 window C. Setup: MAXN, jetson_clocks (GPU 1,377 MHz), fan 100%, 140 W PD supply. 40 runs (3 smoke + 12 J1 + 24 J2 + 1 J3), 0 failed, 0 resets; min VDD_IN 19.62 V, max module power 26 W.
- **J1** (j1_latency_G.csv; j1_intervals.csv).
  - Rows: tsp2 / tsp8 (the hybrid) / tsp8-YOLO11m / cor0 / cor8 + C1 + refiner, the same code as K28 phase 3 and V1. 3 repetitions each.
  - Budgets: K28's S (config v51S, the heuristics capped at v5.1-S's px per frame) and V1's ×1.0 (capped at v6-S's px per frame).
  - Frame sets: Table 3's nuScenes new test 20 scenes (805 frames) and AV2 8 logs with the ego mask (586 frames).
  - Parity: v6-S, v5.1-S, tsp2 and tsp8-YOLO11m reproduce K29's G exactly (0.380 / 0.339 / 0.347 / 0.392 on nuScenes; 0.419 / 0.397 / 0.431 / 0.441 on AV2).
  - Between repetitions, the sd of the mean is ≤ 0.22 ms and of the p95 ≤ 2.1 ms.
  - v5.1-S and v6-S use almost the same px (0.074 / 0.074 nuScenes, 0.057 / 0.058 AV2), so the two budgets give nearly equal heuristic rows.
  - Intervals: paired unit bootstrap (2,000, seed 0). With 8 AV2 logs the bootstrap has a mass at 0, so an interval that ends at 0.000 counts as not significant.
  - cor0 on nuScenes test 20 (G −0.29 / −0.32): a few scenes dominate (interval −0.03 … +1.48).
  - Not timed: unc2 / unc4 / tcr0 / tcr3. Their selection is not ported to the Jetson harness (STATUS).
- **J2** (j2_energy.csv, j2_energy_marginal.csv).
  - One single-pipeline run per row (67 `--only`), on the same frames as J1.
  - Every run also computes the cheap detector (G's baseline) and loads the images. The energy above the cheap run therefore removes image loading and the cheap detector, which are common to all rows.
  - Wall time per frame above the cheap run matches the J1 / K29 latencies within 1.5 ms (v6-S 22.4 vs 21.5–21.9 ms; native + C1 + v3a AV2 77.1 vs 75.7).
  - Power: INA3221 module rails, GPU + CPU + SOC + CV + VDDRQ, at VDD_IN (SYS5V not included), as 1 s means (scripts/k29_vin_logger.py), integrated between the first and last timed frame.
  - Hybrid: its run also runs its budget configuration (v5.1-S at S, v6-S at ×1.0). Its own energy is its run minus that configuration's run alone.
  - Mean module power: v6-S 11.5 / 10.0 W; u960 + C1 + v3a 17.0 / 12.9 W; native + C1 + v3a 22.6 / 20.6 W (nuScenes / AV2). The whole-frame + C1 rows load the GPU harder, not just for longer.
  - Caveat: one run per row and 1 s power resolution, so no intervals. Run lengths are 16–88 s.
- **J3** (j3_latency.csv, j3_sap.csv; remote/A_method/k30_j3_sap.py on VUW).
  - Per-frame latency of all 15,062 Argoverse-HD val frames (24 sequences), one run with every pipeline in the same process.
  - Streaming simulation as K28 (Li et al. 2020; one sequential processor, newest-frame policy, 30 Hz).
  - Detections: the K30 apdump (cheap, wf960 / wf1280 / wf1920 + C1, v6-S). The ≤ 2-call row has no dump of its own, so it uses v6-S's detections with the bounded loop's latency (an upper bound on its sAP).
  - Corridor streaming AP50 follows the k30_v7 rule (GT from the benchmark's 3D labels in the corridor, ≤ 80 m; 21,011 corridor GT).

  | pipeline | sAP | sAP50 | sAP small | corridor sAP50 | mean / p95 ms | frames processed | mean staleness ms |
  |---|---|---|---|---|---|---|---|
  | cheap (u320) | 0.097 | 0.186 | 0.002 | 0.358 | 9.1 / 9.7 | 100% | 33 |
  | u960 + C1 + v3a | **0.198** | 0.390 | 0.061 | 0.441 | 31.9 / 55.1 | 94% | 50 |
  | u1280 + C1 + v3a | 0.184 | 0.390 | 0.082 | **0.468** | 38.7 / 61.1 | 86% | 70 |
  | native + C1 + v3a | 0.144 | 0.342 | 0.083 | 0.434 | 60.3 / 72.6 | 56% | 107 |
  | **v6-S (fast path)** | 0.099 | 0.230 | 0.041 | 0.379 | 36.5 / 82.3 | 82% | 69 |
  | v6-S ≤ 2 calls | 0.104 | 0.236 | 0.043 | 0.393 | 32.4 / 60.9 | 86% | 62 |
  | v6-S, K28 (reference C1) | 0.064 | 0.173 | 0.024 | — | 74.9 / 169.6 | 56% | 128 |
  | K7 whole frame u960 / u1280 (no C1) | 0.221 / 0.247 | 0.413 / 0.465 | 0.065 / 0.106 | — | 21.4 / 29.4 | 100% | 33 / 34 |

  - **Mechanism.** v6-S's crops find the decisive objects but few of all objects. Its offline AP on Argoverse-HD is already 0.165 vs whole frame YOLOv8s 512 0.182 (K28), and its streaming sAP cannot exceed what it detects.
  - The fast path removed the staleness penalty: 82% of frames processed instead of 56%. The remaining gap is detection coverage, not latency.
  - Streaming AP was not a design target of the method.
  - No intervals (a single deterministic simulation, as K28).

---

## 3. Rows where v6-S is not significantly above, with mechanism

not_above_all.csv lists all of them; not_above_counts.csv counts them per item. Counts at this draft:

| item | comparisons | v6-S above | n.s. | row above |
|---|---|---|---|---|
| V1a | 192 | 118 | 71 | 3 |
| V1b | 24 | 0 | 20 | 4 |
| V1c | 12 | 2 | 10 | 0 |
| V2 strata | 126 | 41 | 85 | 0 |
| V2 tube | 3 | 1 | 2 | 0 |
| V2b | 27 | 10 | 17 | 0 |
| V4 Bench2Drive | 92 | 4 | 77 | 11 |
| V5a | 729 | 449 | 271 | 9 |
| V5a / V5b vs prior works | 64 | 41 | 23 | 0 |
| V5b | 120 | 76 | 43 | 1 |
| V5c | 15 | 5 | 9 | 1 |
| V8c KITTI cells | 28 | 6 | 16 | 6 |
| V8c KITTI AP | 27 | 0 | 11 | 16 |
| V8d | 12 | 3 | 9 | 0 |
| J1 Jetson heuristics, ×1.0 budget (J/j1_intervals.csv; not yet in not_above_all.csv) | 10 | 0 | 9 | 1 (AV2 tsp2, −0.012 [−0.029, −0.000]) |
| V8e | 6 | 0 | 5 | 1 |

**Rows where another row is significantly above v6-S, one mechanism each:**
- **V1a AV2, hybrid tsp8 at m = 1.25 / 1.5** (−0.039, −0.035): the hybrid rises from 0.296 at m = 1 (1.33 × px) to 0.332 at m = 1.25 (1.84 × px) while v6-S stays at 0.294; its tube widens with the budget and covers more of the corridor.
- **V1a Argoverse-HD, ego corridor cor0 at m = 1.5** (−0.072 [−0.131, −0.019]): cor0 rises from 0.355 at m = 1 (1.46 × px) to 0.490 at m = 1.5 (2.10 × px) while v6-S stays at 0.418: with twice the pixels the corridor rule examines the whole ego corridor; the extra pixels, not the selection rule, make the difference.
- **V1b AV2 (1152, 3.0) and Argoverse-HD (1280, ·)** (+0.009 … +0.017 over v6-S): larger crops at 1.4–1.8 × px resolve more far objects; the gain per pixel is below v6-S's.
- **V5a AV2, u960 + C1 + v3a in the 9 λ_brake × 0.5 settings** (−0.070 … −0.080): v6-S's AV2 edge over F is avoiding F's unnecessary brakes (V3: v6-S's action is lower than F's on 11.8% of frames, higher on 0.7%); halving the brake weight halves that edge, while F's better far recall (V7: 50–80 m recall 0.32 vs 0.22) still counts.
- **V5b AV2 τ 0.8, u960 + C1 + v3a** (−0.049): τ 0.8 lowers both braking thresholds, so more far, low-demand objects become decision-relevant; the whole frame sees far objects better (V7, AV2 recall at 50–80 m: 0.32 vs 0.22).
- **V5c nuScenes, probe vc10 above the frozen probe** (+0.026 [0.0003, 0.066]): a probe closing at 10 m/s reaches a braking demand at a larger range than a static one, so more far cells are sensitive and v6-S examines farther ahead; the frozen probe is static. (Post-hoc; v6-S stays as frozen.)
- **V4, privileged above v6-S** (DS −4.1, collisions, Overtaking, Emergency Brake): the privileged branch has no perception errors; this is the ceiling, not a competitor.
- **V4, u960 above v6-S in Overtaking** (−0.096 on 45 routes; ability 0.659 vs 0.756): the mechanism cannot be read from the stored records (no per-frame actor state). The fourth run logs hidden actors and hazard sources per frame and will show it.
- **V8c KITTI cells, native / u960 + T4 / C1 + refiner above v6-S** (−0.044 … −0.126): KITTI frames are small (1242 × 375), so native whole frame costs only 15 ms and sees everything; v6-S's crops add little resolution and cut near cars (next item).
- **V8c KITTI AP, whole frames above v6-S (Car −28 … −35 AP; Pedestrian −18 … −22)**, and v6-S below even the draft on Car: 34% of v6-S's vehicle boxes (conf ≥ 0.25) are crop-truncated parts lying inside a labelled vehicle box with IoU < 0.7 (draft 17%, native 14%); KITTI's near cars fill much of a 375-px frame, the crops cut them, and at Car IoU 0.7 the parts count as false positives while the draft's full box was removed (its centre lay in the examined region).
- **V8e AV2 corner-case frames, hybrid above v6-S** (−0.045 [−0.107, −0.009]): spread over action classes (KEEP 7.6, DECELERATE 5.1, HARD_BRAKE 3.3 of the 16.0 cost units); v6-S has two more unnecessary hard brakes and two more under-reactions on these 533 frames; the cones / barrels themselves are invisible to both.

**n.s. rows, grouped mechanisms:**
- **V1a nuScenes hybrid / tsp2 / tcr3 at m ≥ 1, AV2 tsp2 / tsp8 and Argoverse-HD cor8 / tsp2 / tsp8 at m = 1:** the hybrid already fills the support first (the same objects v6-S's support selects) and its tube covers the ego lane where most decisive objects lie; v6-S's field adds the off-corridor cells, which matter on few frames (V2: 4.6% / 10.3% / 20.8% of the footprint).
- **V1a at m = 1.5:** at 1.8–2.4 × v6-S's px every heuristic covers most decision-relevant regions; the selection rule stops mattering.
- **V1b (20 n.s. points):** G is flat over the 3 × 3 axis around (960, 2); the axis is not a tuned optimum.
- **V1c:** both consumers' supports overlap; the TTC consumer's own support shifts the hybrid by 0.01–0.016 and makes it tie v6-S on all three datasets.
- **V2 strata (48 n.s.):** strata hold 283–8,964 frames; differences of ±0.01–0.02 are within the unit variance.
- **V2b AV2:** every row except unc2 lies within −0.040 … +0.008 of v6-S with wide intervals; no mechanism identified.
- **V4 (77 n.s.):** the five non-privileged branches complete mostly the same routes (v6-S vs cheap: 5 routes flip one way, 6 the other in ≥ 2 of 3 runs); run-to-run variation is of the same size as the branch differences.
- **V5a / V5b / V5b-prior n.s.:** u960 / native + C1 + v3a and the hybrid are within the bootstrap noise of v6-S at most weights (ranks 1–4 within ±0.05); nuScenes SAHI vs v6-S has wide intervals, and SAHI's G falls from 0.21 to 0.02 as λ_brake goes from × 0.5 to × 2, so its cost is dominated by braking terms.
- **V8c cells (16 n.s.):** six sequences only; intervals are ±0.05–0.10.
- **V8d × 0.5:** at half speed only 301 frames are safety-critical and the field is empty on 85% of frames, so v6-S ≈ the draft; whole frames gain from seeing everything.
- **V8e nuScenes:** 373 frames, 39 scenes; wide intervals.

---

## 4. New test evaluations (all post-hoc)

| date (VUW) | dataset / split | pipelines | what |
|---|---|---|---|
| 2026-10-15 22:35 – 10-16 | nuScenes new test (228 scenes), AV2 val ego mask (150 logs), Argoverse-HD val (24 seq) | v6-S, v5.1-S, cheap, 8 S heuristics + C1 + v3a at m ∈ {0.75 … 1.5}, whole frames + C1 + v3a | V1a / V1c / V2 (part match), V1b (part axis), V2b (part idm), V5b / V5c (part cons) |
| 2026-10-16 00:30 – | same | as above + u640 / u1280 | V7 detection dumps (part apdump), V8d (part speed, nuScenes) |
| 2026-10-15 23:00 | nuScenes new test; AV2 val (ego mask) | SAHI s1.0; FOVEA KDE_I 960 | prior-work re-runs for V5 |
| 2026-10-16 | KITTI test (DEEP, 6 sequences, 3,536 frames) | cheap, u960, native (+ T4 / C1 / refiner, flags 0 and 1), v6-S | V8c and the K28 KITTI flag re-run |
| 2026-10-16 | nuScenes new test / AV2 val (stored records + annotations) | stored K28 pipelines | V8e corner cases, V8g CODA |
| 2026-10-15 / 16 | Bench2Drive 220 (219 routes) | six K6 branches | V4 official metrics from the stored three runs; fourth run with logging (running) |
| 2026-10-16 | NAVSIM navtest; navhard_two_stage | reference, cheap, full960, native, spec_pdm, hybrid_pdm | V6 (i), V6 (ii) timing (1,000 tokens), V8a (running) |

| 2026-10-18 06:36–09:13 (Jetson) | nuScenes new test 20 scenes; AV2 val 8 logs (ego mask); Argoverse-HD val (24 seq) | v6-S, v5.1-S, cheap, tsp2 / tsp8 / tsp8-YOLO11m / cor0 / cor8 + C1 + v3a (S and ×1.0), whole frames + C1 + v3a, v6-S ≤ 2 | J1 latency-G, J2 energy, J3 streaming AP (fast path trt32) |
---

## 5. Files

- **Code** (VUW scripts/, local remote/): A_method/k30_gt.py (49396624), k30_eval.py (55471d34), k30_eval2.py (d8534941), k30_stored.py (7b68a903), k30_analyze.py (25309105), k30_v7.py (c1d72713), k30_corner.py, k30_v8eg.py, k30_v5_prior.py, k30_sahi_nusc.py (d7dfdf00), k30_fovea_av2.py (e542a92c), k30_kitti.py (ebebf4b5), k30_kitti_an.py (b5e1e145), k30_not_above.py, k30_figs.py; A_closedloop/k30_b2d.py (db2e22d1), perception_pilot_log.py (2bd87c16), k30_b2d_log_dispatch.sh (ffa2cc51), k30_b2d_inloop.py (8bcd99c4); A_navsim/k30_nav_hybrid.py (01acda75), k30_probe_time.py (ed77ccdf), k30_probe_an.py (a9fca18a), k30_nav_v2agent.py, k30_navhard.py (7e579cc7), k30_nav_merge.py (5e85e72b).
- **Tables**: this folder (CSV copies) and results/vuw_summaries/k30/. Jetson J: J/j1_latency_G.csv, j1_intervals.csv, j2_energy.csv, j2_energy_marginal.csv, j3_latency.csv, j3_sap.csv; raw runs results/raw/20261018_*_k30j_timing_*; logs logs/k30j/.
- **Jetson code**: scripts/67_k30_jetson.py (49c83f21), 68_k30_j_tables.py (6175f757), k30_j_window.sh (9a5f7dd7); VUW remote/A_method/k30_j3_sap.py (b40a6cf6).
- **Figures** (PDF + PNG): k30_v1a_matched, k30_v1b_axis, k30_v8d_speed, k30_v8c_kitti, k30_v6ii_probe; panels/ (V3: the 60 costliest safety-critical v6-S frames, panels a–f per frame as K28 phase 3, with examples.csv).
