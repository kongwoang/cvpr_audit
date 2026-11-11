**Share of ours' disagreement with π(F) explained by each cause** (post-hoc analysis; ours without refiner; F = u960; full test splits; AV2 with the ego mask). Each cause is added in turn, in the order of the chain; the share given is the share of the ORIGINAL disagreeing frames that it fixes, with the decision-loss-weighted share in brackets.
- **nuScenes new test** (453 / 9,181 frames disagree):
  - crop-level detection / locality 81.0% (87.6%);
  - temporal history 4.2% (1.8%);
  - motion assumption (closing probe 20 m/s) 0.0%;
  - finite grid 2.9% (1.4%);
  - single probe size 5.1% (4.8%);
  - round cap 0.0%;
  - **residual 6.8% (4.3%)**.
- **AV2 val, masked** (1,649 / 11,818 frames disagree):
  - locality 62.0% (59.3%);
  - temporal history 11.2% (11.8%);
  - motion assumption 20.6% (23.0%; all of it at 10 m/s, nothing more at 20 m/s);
  - finite grid 1.1% (1.0%);
  - probe size 1.2% (1.3%);
  - round cap 0.0%;
  - **residual 3.9% (3.6%)**.
- **The residual is mostly F's error.** In the residual, F's action matches the annotated scene on only 3/31 (nuScenes) and 3/64 (AV2) frames; ours matches it on 27/31 and 51/64. The typical case is F braking hard on a vehicle or cyclist at 5–8 m (median 8 m) that lies below the grid's first range row and outside ours' region.

# K16: attribution of the deviation from decision-exactness

**Setup.** Same frames as K2 / K13. Ours' real-crop run reproduces instr_eval's actions exactly (agreement 0.951 nuScenes, 0.861 AV2 masked). Script: `remote/A_method/k16_attr.py` with the generalised loop. The chain replaces the crop detector by F's objects and adds one intervention at a time:
- a: exact locality, i.e. F's objects whose centre lies in a crop (guard as in K12);
- b: + F's ttc history;
- c: + closing probe (nuScenes 10 → 20 m/s; AV2 0 → 10 → 20 m/s);
- d: + grid 8–80 m, ±9 m;
- e: + probe templates (0.6×1.8, 1.8×1.6, 2.6×3.6 m; footprint = union);
- f: + round cap 10.

Bootstrap: 2,000 resamples over scenes / logs.

| step (cumulative) | nuScenes agreement [CI] | AV2 masked agreement [CI] |
|---|---|---|
| ours (real crops) | 0.951 [0.942, 0.959] | 0.861 [0.845, 0.876] |
| a. exact locality | 0.988 [0.985, 0.991] | 0.941 [0.930, 0.951] |
| b. + F's history | 0.992 | 0.959 |
| c. + closing probe | 0.992 (20 m/s) | 0.988 (10 m/s); 0.988 (20 m/s) |
| d. + extended grid | 0.993 | 0.991 |
| e. + probe templates | 0.996 | 0.993 |
| f. + round cap 10 | 0.996 [0.994, 0.998] | 0.993 [0.991, 0.995] |

Interventions also break some frames that agreed before; the counts are in `attribution_*.csv` (`newly_broken_frames`, e.g. 24 / 74 after step a). The shares above are net of the original disagreeing frames. Figure: `attribution.pdf/png`.

**Rounds** (`rounds.csv`; rounds 0 / 1 / 2 / 3; frames stopping at the cap without having converged):
- nuScenes: 20.6 / 75.4 / 3.9 / 0.1%; 0.00% stop at the cap without converging (K1 reproduced).
- AV2 masked: 33.7 / 55.6 / 10.2 / 0.5%; 0.04% stop at the cap without converging.

**Field coverage** (`coverage_*.csv`; F's objects whose demand reaches τ_{a+1} for ours' action a; share with box centre in ours' verified region R):

| | nuScenes | AV2 masked |
|---|---|---|
| all | 0.86 (295 objects) | 0.58 (1,076) |
| < 12.5 m | 0.32 | 0.31 |
| 12.5–70 m | 0.93 | 0.59 |
| 70–80 m | 0.91 (45) | 0.77 (103) |
| \|y\| > 6 m | 0.67 (6) | 1.00 (2) |
| vehicle / person / cyclist | 0.87 / 0.89 / 0.38 | 0.62 / 0.29 / 0.31 |

- **Objects at 70–80 m.** They DO mostly fall inside the footprint, because the footprint is an image bounding box: 91% and 77%.
- **How often they decide π(F).** Such an object is F's deciding object in 42 of nuScenes' 277 "ours brakes less" frames, and 96 of AV2's 959.
- **Classes.** The detector's classes are coarse: car vs truck / bus is not separable.
- **Coverage gap on AV2.** It is below 12.5 m and for pedestrians / cyclists. At v_close 0 the field is small; step c recovers 21% of the disagreement.

**Direction** (`direction_*.csv`, same frames as K2 / K13):

| | ours brakes less: frames / loss ours / F / cheap | ours brakes more: frames / loss ours / F / cheap |
|---|---|---|
| nuScenes | 277 / 575 / 457 / 498 | 176 / 390 / 1,119 / 382 |
| AV2 masked | 959 / 578 / 2,429 / 809 | 690 / 1,731 / 366 / 868 |

On AV2, when ours brakes less, F carries the loss. When ours brakes more, ours does.

**Residual frames.** `residual_*.csv` lists all of them; `residual20_*.csv` lists 20 per dataset, each with a one-line description. Most are F hard-braking on a near object (5–8 m) outside R, or below the grid, while the annotated scene requires KEEP.
