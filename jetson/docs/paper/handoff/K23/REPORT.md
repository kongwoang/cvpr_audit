**Result** (development data only; no test split touched; selection rule of method_freeze_draft §2e, fixed before running)
- **nuScenes new val, v5-S = v5-L:** frozen dev G 0.263 → v5 0.276, [+0.003, +0.031], at 0.080 tensor px vs 0.104 (0.77×).
- **AV2 CFG-DEV (300 logs, ego mask), v5-S = v5-L:** frozen dev G 0.113 → v5 0.204, [+0.061, +0.120], at 0.055 tensor px (1.00×).
- **Adopted:**
  - **C1 refiner v3a** (range head; v2 data + AV2 REF-TRAIN). Alone: nuScenes +0.005 [+0.0004, +0.014], AV2 +0.091 [+0.061, +0.120].
  - **C4 v_close 0 as the shared value.** Alone on nuScenes: +0.008 [+0.0004, +0.022] at 0.77× px; it is already AV2's value.
  - The result is **one shared configuration**: crops at 960 px per long side, context 2, no margin, v_close 0, c_new 0.25, refiner v3a.
- **Rejected:**
  - **v3b / v3c** (lateral offset head; validity veto). Worse on both datasets, also after correcting how the lateral offset is applied (see below).
  - **C2 c_new 0.35 / 0.45 / 0.55.** Significant on AV2 (+0.012 / +0.028 / +0.045 with refiner v2), but negative on nuScenes (−0.020 / −0.053 / −0.090), so it fails "not negative on the other".
  - **C2 persistence.** Negative on both.
  - **C3 temporal support.** n.s. on both.
  - **C4 crop scale, context, margin.** n.s. alone. Context 2.5 is +0.031 / +0.002 n.s. with the frozen refiner (it becomes significant on nuScenes only together with v3a).
  - **v_close 5 / 10 on AV2.** Negative.
  - **C5 bounded loop.** n.s.
- **No component passed at the L operating point (≤ 2× px), so v5-L = v5-S.** The frozen configuration is replaced on both datasets.
- **Stability.**
  - v5 is above the frozen configuration on both halves of each dev set (nuScenes 0.311 / 0.305 and 0.209 / 0.179; AV2 0.237 / 0.112 and 0.176 / 0.114).
  - v5 is NOT the highest-G candidate in `top5.csv`. The best dev rows (nuScenes context 2.5 + v3a 0.300; AV2 c_new 0.55 + v3a 0.227) were excluded by the pre-registered rule.
- **Caveat: the baselines also gain from refiner v3a on dev** (u960 0.207 / 0.158; uncertainty crops −0.060 / 0.178), so K24's "baselines for v5" must receive v3a too.

# K23: v5 selected on development data

**Data** (method_freeze_draft §2e).
- **nuScenes:** new val, 152 scenes (6,117 frames). Refiner v2 had used 151 of them for epoch selection; v3 does not.
- **AV2:** the 700 train logs split by `default_rng(20261010)` into 300 CFG-DEV, 340 REF-TRAIN and 60 REF-VAL (`av2_split.csv`, md5 39860f4d…). No refiner has seen a CFG-DEV log.
- **Argoverse-HD:** untouched; it will receive the shared configuration.

**Refiner v3** (`refiner_v3/refiner_v3_training.csv`, `train_report_v3{a,b,c}.json`; `k23_refdata.py`, `k23_train_refiner.py`).
- **Architecture:** v2's (ResNet-18 + 13 box features), with a 3-output head.
- **Training data:**
  - KITTI train: 14,605;
  - nuScenesAll TRAIN scenes: 62,527 (its validation scenes overlap nuScenes new val and are not used);
  - AV2 REF-TRAIN: 96,918 crop and u960 detections within 80 m and ±2 m of the corridor, ego mask, 90% matched.
- **Model selection:** KITTI val (4,825) + AV2 REF-VAL (17,318). 25 epochs; AdamW 5e-4, cosine schedule.
- **Validation errors** (uncorrected lift: range |log| 0.263, lateral MAE 1.03 m):

| variant | best epoch | range \|log err\| | lateral MAE (m) | validity BCE (base rate 0.23) | md5 |
|---|---|---|---|---|---|
| v3a | 23 | 0.058 | — | — | 6b29d05e8f06d121ec9e11aeb9dc2730 |
| v3b | 19 | 0.064 | 0.29 | — | a0a063a604b60e84abe2bb4473665c09 |
| v3c | 7 | 0.073 | 0.32 | 0.30 | 2643befbd4c1d0a26a9d269bafe70ab5 |

**Evaluation** (`k23_dev.py`, `k23_analyze.py`).
- **Arms:** each arm is the frozen loop plus one switch. All arms run per frame with shared crop detections; tensor px are counted per arm.
- **Scoring:** every arm is scored without refiner and with the frozen refiner, v3a, v3b, v3c, and v3c with the veto at 0.1 / 0.2 / 0.3 / 0.5.
- **Parity:** the base arm reproduces instr_eval on nuScenes new val bit for bit (actions, cost with refiner v1, px).
- **Tables:**
  - `components_{nusc,av2}.csv`: every arm × refiner, alone;
  - `sweep_{nusc,av2}.csv`: the C4 geometry rows;
  - `combos_{nusc,av2}.csv`: the passing components combined, plus v3b / v3c re-scored;
  - `top5.csv`: the 5 best candidates per operating point, regardless of the rule;
  - `halves.csv`.
- **Correction made during K23 (before selection).** The first scoring of v3b / v3c added the lateral offset to the range-scaled lateral extent, although the head was trained on the offset from the UNSCALED estimate. In the combos run, the corrected centre is the estimated centre plus the offset, with the width scaled by the range factor. v3b / v3c remain worse than v3a on both datasets after the correction (nuScenes −0.106 / −0.071; AV2 −0.049 / −0.022 vs frozen).

**Frozen** (method_freeze_draft §2f, "frozen pending confirmatory run"): v5-S = v5-L, with code and weight md5 recorded there. K24 is not started.
