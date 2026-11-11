# K4 — what the decision-sensitivity field contributes (post-hoc analysis on the test splits, 2026-10-04)

Instrumented run of K2/K3 (`instr_eval.py` md5 2177d413…, settings base / del_all / del_half; `instr_analyze.py` md5
73cacf72…). Debugged on nuScenes new val (development; rows in the CSVs), then ONE run on nuScenes new test and AV2 val
(post-hoc). Policies: ours (full), field only, support only, uncertainty / temporal_crit / random crops at ours'
per-frame crop AREA (their tensor pixels end up larger: their crops truncate more objects, so the guard re-detects more
— see network_px), u960. "Decision-relevant" = the GT object's own required deceleration ≥ the DECELERATE threshold.
Match = class-compatible, IoU ≥ 0.3 (Hungarian). Recovery = the object is matched in the policy's final scene.

## A. Natural misses (missed_natural.csv): relevant GT object, no matched draft detection, detected by u960
| | frames / objects | ours | field only | support only | uncertainty | temporal_crit | random |
|---|---|---|---|---|---|---|---|
| nuScenes test: recovery | 114 / 117 | **0.624** | 0.479 | 0.197 | 0.248 | 0.581 | 0.197 |
| nuScenes test: G on these frames (u960 0.784) | | 0.505 | 0.554 | −0.129 | −0.111 | 0.490 | −0.033 |
| AV2 val: recovery | 360 / 396 | 0.765 | 0.240 | 0.364 | 0.490 | **0.785** | 0.396 |
| AV2 val: G on these frames (u960 0.349) | | 0.347 | 0.190 | 0.077 | 0.180 | 0.244 | 0.026 |
Agreement with π(F) on these frames: nuScenes ours 0.667 (temporal_crit 0.649), AV2 ours 0.750 (temporal_crit 0.639).

## B. Controlled deletion (missed_controlled.csv; strata by GT range of the deleted object and ego-speed tercile)
Every matched draft detection of a relevant GT object is deleted (del_all) — or each with probability 0.5 (del_half,
seeded) — from the draft and from its ttc history before the policy runs. Loss normalisation: (Σ J_policy − Σ J_cheap
unmodified) / Σ J_cheap unmodified over the frames with a deletion (0 = as good as the unmodified draft; negative =
better).
| del_all | nuScenes test (791 frames, 823 objects) recovery / loss | AV2 val (2,584 frames, 2,923 objects) recovery / loss |
|---|---|---|
| cheap (deleted) | 0 / +4.37 | 0 / +0.36 |
| **ours** | **0.964** / +0.34 | 0.789 / −0.001 |
| field only | 0.920 / +0.31 | 0.240 / +0.025 |
| support only | 0.149 / +4.62 | 0.186 / +1.59 |
| uncertainty | 0.389 / +3.74 | 0.503 / +0.82 |
| temporal_crit | 0.819 / +1.66 | **0.843** / +0.10 |
| random | 0.542 / +2.76 | 0.661 / +0.19 |
| u960 | 0.987 / +0.28 | 0.995 / −0.33 |
del_half: same ordering (nuScenes ours 0.958, field only 0.913; AV2 ours 0.789, temporal_crit 0.873). Strata
(del_all, ours): nuScenes < 30 m 0.958, 30–50 m 0.987, > 50 m 1.000; speed terciles 0.957 / 0.967 / 0.966. AV2 < 30 m
0.764, 30–50 m 0.853, > 50 m 0.820; speed terciles 0.675 / 0.849 / 0.835 (the field depends on speed; with v_close 0
it is empty when slow).

## Reading
On nuScenes (v_close 10) the field does what it is designed for: ours and field-only recover > 90% of deleted relevant
objects at every range and speed, while support-only (15%) and uncertainty crops (39%) cannot, and the decision loss of
ours returns close to u960's. On AV2 (v_close 0) the field alone recovers only 24%; ours' 79% comes from the field and
the support crops of the iterated loop together, and **temporal_crit recovers more (84%, using its previous frame's
merged objects)** at 1.7× ours' network pixels — report as is. Natural misses are rare (114 nuScenes / 360 AV2 frames).
Example frames for the qualitative figure: handoff/K8 (chosen from these runs).

**Note (2026-10-05, K10):** every AV2 number in this report is unmasked. The AV2 ego hood drives 51% of the cheap draft's actions (81% of its loss); see handoff/K10 for the masked re-evaluation of the AV2 table.
