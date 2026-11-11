**Result** (post-hoc analysis; frozen configurations, nothing tuned; AV2 with the ego mask)
- **(1) Argoverse-HD at equal tensor px: no crop heuristic + refiner is above ours final, but ours is significantly above only temporal_crit.**
  - **Setup.** K18 protocol with budget b = 0.5, which is the smallest budget. Its realised heuristic px is still 1.27–1.53× ours', because guard re-detections of the heuristic crops add canvases.
  - **G with refiner, ours final (0.091, at 0.070 px) minus each:**
    - ego-corridor 0.082: [−0.023, +0.041];
    - traj_support 0.093: [−0.031, +0.027];
    - uncertainty 0.054: [−0.076, +0.168];
    - random 0.009: [−0.032, +0.207];
    - temporal_crit 0.023: **[+0.008, +0.134]**.
  - **K14's equal-AREA row.** The corridor + refiner above ours (0.122) used 1.6× ours' px. At b = 1 (1.56× px) the corridor reaches 0.097 and is no longer above ours: [−0.040, +0.025].
- **(2) Same footing without refiner: ours without refiner minus each heuristic without refiner, at b = 0.5** (realised 0.96–1.53× ours' px).
  - **nuScenes:** above temporal_crit [+0.009, +0.122], corridor, traj_support and random; uncertainty [−0.040, +0.499] n.s.
  - **AV2:** above corridor [+0.000, +0.067] and traj_support [+0.007, +0.073]; uncertainty is higher than ours (0.080 vs 0.032) [−0.103, +0.005], temporal_crit and random n.s.
  - **Argoverse-HD:** all n.s.; corridor (0.054) and traj_support (0.057) are above ours' 0.039 in the point estimate.
- **(3) K17's group d** (AV2 554 frames, Argoverse-HD 771 frames) is driven by the monocular lift's range and closing-speed estimates, not by lateral placement or class. On the frames where ours raises the action, the decisive object comes from a field crop and is absent from support-only's scene in 87% / 86% of cases.
  - **Range.** Ours places the object closer than the annotation in 99.7% / 98.3% of cases. Median error −17.5 m / −10.5 m; estimated/annotated ratio 0.75 / 0.83.
  - **Closing speed.** Median estimated closing speed 15.8 / 43.5 m/s vs annotated 5.6 / 6.1 m/s. The ttc from the history is the largest single cause: annotated ttc alone removes the extra braking in 59% / 71%, annotated range in 48% / 29%, annotated lateral extent in 21% / 15%; none of the three alone does in 10% / 17%.
  - **Class.** The class is wrong in only 0.6% / 3.6%.
  - **u960 reproduces most of it.** u960 detects the same object in 96% / 93% of these frames, and the object is also decisive for u960's action in 49% / 40%. The field largely reproduces the target detector's false alarm, which the cheap draft did not have.

# K22: follow-up of K14 / K18(A)

**(1)** `heur_ahd.csv` (format of `heur_av2.csv`; budgets b ∈ {0.5, 1, 2} × ours' per-frame crop tensor px; realised px ratio in `px_ratio_to_ours`; intervals of ours final minus each heuristic with and without refiner). Argoverse-HD val: 24 sequences, 15,006 frames with 3D labels; frames without labels are run for the histories only. `pareto.pdf/png` now has three panels (nuScenes, AV2 masked, Argoverse-HD); `pareto_ahd.csv`. Script: `remote/A_method/k18_heur.py --dataset ahd`.

**(2)** `same_footing_noref.csv`: dataset, heuristic, G_ours_noref, G_heur_noref, realised px ratio, interval (ours without refiner minus heuristic without refiner, unit bootstrap, 2,000).

**(3)** `groupd_{av2,ahd}.csv` (one row per group-d frame) and `groupd_summary.csv` (`remote/A_method/k22_diag.py`). Each unit with group-d frames is re-run from its start with the frozen loop, support only, u960 and refiner v2. On each such frame where ours (with refiner) acts above support only:
- the decisive object is the largest required deceleration in ours' scene;
- it is matched to the annotated object with the largest IoU (≥ 0.3, any class);
- each of annotated range, lateral extent and ttc is substituted alone for ours' estimate, to see whether ours' action drops;
- u960's detections are checked for the same object (IoU ≥ 0.5) and whether it decides u960's action.

On AV2, 356 of the 362 decisive objects have an annotated match; on Argoverse-HD, 586 of 595. The few without a match are detections with no annotated counterpart in the image, i.e. spurious.
