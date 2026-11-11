# K2 — agreement of ours with π(F) on the test splits (post-hoc analysis, 2026-10-04)

**Post-hoc analysis.** Computed after the confirmatory runs on nuScenes new test (228 scenes, 9,181 frames) and AV2 val
(150 logs, 11,818 frames). The per-frame pickles of the confirmatory runs stored costs only, so the final configurations
were re-run with action logging: `remote/A_method/instr_eval.py` (md5 2177d413…; `python scripts/A_method/instr_eval.py
--dataset {nusc|av2} --split test --setting base --shard i/6 [--diag_units 50]`) and tabulated by `instr_analyze.py`
(md5 73cacf72…). Frozen code untouched; ours = `vtf.speculative.run` itself. **Parity with the confirmatory runs:** AV2
identical on 100% of frames (cheap, u960, ours, ours + refiner); nuScenes identical on 99.4–99.7% (the confirmatory run
used the CPU detector, this one GPU FP16) — G ours + ref 0.170 vs 0.171, u960 0.011 vs 0.010.
F = whole frame at the verification scale (u960) — and, second, native resolution (u1600 / u2048). Agreement = share
of frames whose braking action equals π(F); intervals: scene / log bootstrap, 2,000.

## 1. Agreement (agreement.csv)
| | nuScenes vs u960 | AV2 vs u960 | nuScenes vs native | AV2 vs native |
|---|---|---|---|---|
| ours, no refiner | **0.951** [0.942, 0.959] | **0.810** [0.790, 0.831] | 0.943 | 0.796 |
| ours, with refiner | 0.941 [0.930, 0.951] | 0.790 [0.767, 0.813] | 0.934 | 0.778 |
| uncertainty crops (same crop area) | 0.943 [0.933, 0.953] | 0.736 [0.715, 0.758] | 0.938 | 0.699 |
| temporal_crit (same crop area) | 0.939 [0.930, 0.948] | 0.641 [0.614, 0.665] | 0.935 | 0.570 |
| random crops (same crop area) | 0.927 [0.916, 0.938] | 0.601 [0.575, 0.627] | 0.921 | 0.545 |
| u640 whole frame | 0.943 [0.933, 0.952] | 0.685 [0.656, 0.712] | 0.931 | 0.575 |
| cheap draft | 0.915 [0.902, 0.927] | 0.555 [0.522, 0.586] | 0.910 | 0.472 |
On nuScenes most frames are KEEP for every pipeline (cheap already agrees on 91.5%), so differences are small and the
intervals of ours and the heuristics overlap; on AV2 ours reproduces π(F) clearly better than every same-budget policy
(0.810 vs ≤ 0.736) — but far from exactly (81%).

## 2. Confusion matrices (confusion_{nusc,av2}_ours_{noref,ref}_vs_{u960,native}.csv; rows ours, columns π(F))
nuScenes, no refiner: diagonal 7,907 / 529 / 292; off-diagonal largest: ours KEEP where F decelerates 173, ours KEEP
where F hard-brakes 65. AV2, no refiner: diagonal 5,940 / 1,975 / 1,661; ours KEEP where F hard-brakes 677, ours
DECELERATE where F hard-brakes 529.

## 3. Does disagreeing help or hurt? (disagreement_split.csv; Σ decision loss vs ground truth on the disagreeing frames)
| | frames | Σ loss ours | Σ loss F (u960) |
|---|---|---|---|
| nuScenes, ours brakes less than F | 277 | 574.8 | 457.1 (F better) |
| nuScenes, ours brakes more than F | 176 | 390.1 | 1,118.7 (ours better) |
| AV2, ours brakes less | 1,586 | 930.6 | 4,389.4 (ours better) |
| AV2, ours brakes more | 656 | 1,697.4 | 341.2 (F better) |
Disagreement is mostly beneficial overall (AV2: F's extra braking is largely phantom braking; nuScenes: F's harsher
braking costs more), but each dataset has a class where F is right (nuScenes: ours under-brakes; AV2: ours over-brakes).

## 4. Locality decomposition on test (first 50 scenes in shard order = 1,989 frames of nuScenes new test)
real crops 0.950 → exact locality (F's detections inside the crops) **0.990** → + F's ttc history **0.994**
(development split earlier: 94.8% → 95.0% → 99.7%). On test, most of the gap to exactness is locality (detector run on
crops vs on the whole frame), and with exact locality the loop reproduces π(F) on 99.0–99.4% of frames.
Not run on AV2 (diagnostic variants only in the nuScenes base run).

**Against the method / caveats:** agreement with π(F) is 95% (nuScenes) and 81% (AV2), not 100% — the proposition's
assumptions (locality, exact sensitivity, consistent history) do not hold on real crops; with the refiner agreement
drops by 1–2 points because the refiner changes ranges on purpose. On nuScenes the same-budget heuristics and u640 agree
with π(F) about as often as ours (overlapping intervals).

**Note (2026-10-05, K10):** every AV2 number in this report is unmasked. The AV2 ego hood drives 51% of the cheap draft's actions (81% of its loss); see handoff/K10 for the masked re-evaluation of the AV2 table.
