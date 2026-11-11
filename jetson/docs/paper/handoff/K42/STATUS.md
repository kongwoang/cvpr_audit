# K42 STATUS

- 2026-11-11 13:15: task received (the final-version numbers). Sources were checked by column names only.
- PREREG.md was written before any computation.
- A1 (13:40), written before any added K42-2 row was read:
  - the K42-2 gate tolerance is 1e-12, after observing 7.1e-15;
  - K42-5 already exists in K30 J2.
- 2026-11-11: **K42-1 DONE.** It uses the stored y11s basewf records, so no GPU was needed.
  - Gate: 147 / 147 ctable_y11s rows reproduced (1e-16).
  - With the full lift, whole-frame YOLO11s is significantly above DGSP at nuScenes u1280 and AV2 u640 / u960.
  - DGSP stays significantly above every side up to u1280 on Argoverse-HD.
- 2026-11-11: **K42-2 DONE.**
  - Gates: the K31 rows are reproduced (7e-15), and the 18 router J checks give 0.
  - DGSP has fewer false episodes but a higher missed rate than most whole-frame and published rows.
- 2026-11-11: **K42-3 DONE**, packaged as k42_3_fidelity_v3a.py.
  - md5 of the CSVs is listed in REPORT.md; the comparison with the Mac copy is left to the writer.
  - The router gates give 0.
  - Oracle locality and finer probes with the refiner: not stored (the GPU re-run was not done).
- 2026-11-11: **K42-4 (a) DONE.** Field + support coverage is 0.897 / 0.924 / 0.911; the checks give 0.
  - **(b) needs a GPU run, not run**: candidate crops are not stored and crop detections are not cached.
- 2026-11-11: **K42-5 DONE from stored K30 J2** (u512 + comp. 0.264 / 0.424 J above cheap vs DGSP 0.289 / 0.536). No
  Jetson window was needed.
- Open, only if asked:
  - an instrumented VUW GPU re-run for K42-4(b);
  - a refined K39 single-intervention / chain run for the K42-3 second part.
