# K37 STATUS

| item | state |
|---|---|
| K37-4 coverage table | DONE 2026-10-26 (K37-4/REPORT.md, table.md / .tex) |
| K37-2 deadlines | DONE 2026-10-26 (K37-2/REPORT.md; approximation) |
| K37-3 AnA vs v6-S natural misses | DONE 2026-10-26 (K37-3/REPORT.md) |
| K37-1 operating curves | DONE 2026-10-26 (K37-1/REPORT.md); holdout now USED (2nd use) |

## Log
- 2026-10-26: K37-4 and K37-2 done.
- 2026-10-26: K37-3.
  - The population reproduces tab:missed exactly: v6-S / AnA 0.607 / 0.718, 0.848 / 0.808, 0.873 / 0.865.
  - Low-confidence draft boxes were never stored. By the user's choice, the frozen pass-1 draft is re-run at conf 0.05
    on those frames (detector only).
- 2026-10-26: K37-1 harness built (k30_thr.py sha256 db43cabf…); PREREG frozen before any test / holdout run; dev smoke
  on nuScenes val scene-0108.
- 2026-10-26 12:18 NZ: dev smoke parity at thr 1.0 vs K30 dev_devref: 0 differences in all 11 columns.
  - K37-1 launched: dev (VUW: nuScenes 2 shards + AV2 4 shards); test (VUW1: nuScenes 2 + Argoverse-HD 1 + AV2 3); holdout
    (VUW: 3 shards, k30_thr_h.py sha256 75528901…, the 2nd use).
  - Chain: scripts/A_method/k37_k1_chain.sh dev|test|hold. Logs: logs/k37/k1_*.
- 2026-10-26: K37-3 DONE. AnA's nuScenes gain sits on objects with a conf 0.05–0.25 draft box (0.846 vs 0.603) and outside v6-S's crops (8/13 vs 0); v6-S leads inside the field on AV2 / AHD.
- 2026-10-26 13:45 NZ (hourly): dev nusc 152/152, av2 293/300; test nusc 228/228, av2 150/150, ahd 15/24; hold 26/65; no errors.
- 2026-10-26 14:45 NZ: K37-1 dev complete (152 / 300); dev parity at thr 1.0: 0 differences.
  - **Rule (c) choice, recorded before any test / holdout number is read** (K37-1/k1_choice_dev.csv). Bound = cheap's
    dev missed rate at 1.0: nuScenes 34/99, AV2 30/366.
  - Choices:
    - nuScenes: cheap 1.0; u960 0.8; native 0.8; hybrid 0.8; DESP 0.8.
    - AV2: cheap 1.0; u960 0.6; native 0.6; hybrid 0.6 (30/366, exactly at the bound); **DESP infeasible** (lowest miss
      31/366 at 0.6, so the 0.6 point is used, labelled).
  - Argoverse-HD test and the holdout use the AV2 choices.
- 2026-10-26 15:45 NZ: **K37-1 DONE.**
  - Every split complete. Parity at thr 1.0: 0 differences everywhere; 3 single-frame differences vs K34-1's CPU
    replay at other thresholds.
  - (c) holdout at the constrained points: DESP (0.6) misses 20 / 154 (cheap 18), 7.1 vs 34.5 false / 1,000, G 0.474;
    u960 −0.016 G (significant); native +0.009 (n.s.); hybrid +0.000 with +1.45 false (significant).
  - W1 persists vs whole frames on nuScenes / AV2 test.
  - The holdout is now USED (2 uses: K35-4, K37-1(c)).
  - **All K37 items complete.**
