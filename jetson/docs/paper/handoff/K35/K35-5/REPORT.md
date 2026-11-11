# K35-5 — zero-value-of-computation test (POST-HOC, three test splits, GPU)

**Post-hoc analysis.**
- Code: remote/A_method/k35_k5.py (md5 4bac2079), run on VUW1 2026-10-25; tables by k35_k5_report.py (md5 577b74e3).
- Base: frozen v6-S's final scene of every frame (K30 apdump: boxes and post-refiner geometry; the action from it equals
  the stored a_v6S|v3a on every frame, asserted).
- **Extra crops**, at +25% and +50% of v6-S's crop px of the frame, 5 seeds:
  - K28's random-crop generator: sizes from the draft's boxes at 2× context, positions in the lower 65% of the image;
  - restricted to lie **strictly outside** every crop v6-S verified (field, support, guard) and outside the field
    footprint.
- **Processing of the new crops:**
  - YOLOv8s at v6-S's verification scale with the truncation guard;
  - K28 merge into the stored scene;
  - the new detections get no fine history (TTC infinite), mono-lift geometry with the AV2 ego mask, and the frozen v3a
    on the new objects in the braking support;
  - deployed thresholds; K28 decision cost.
- **Statistics:** paired unit bootstrap, 2,000, seed 0. **Bold** = the interval excludes 0. A frame with no v6-S crop px,
  or no candidate that fits outside the verified region, gets no extra crop and is unchanged.

**Prediction (registered in the task):** the changed-action rate stays ≤ the deployed disagreement rate with π(F)
(2.4 / 3.2 / 6.0%), and dG ≈ 0.

## Results

| dataset | budget | seed | frames with crops | mean extra px | changed-action rate | reference (v6-S no ref. vs π(F)) | dG = G(row) − G(v6-S) [95% CI] |
|---|---|---|---|---|---|---|---|
| nuScenes | +25% | 0 | 3196 / 9181 | 0.0050 | 0.011% | 2.4% | -0.0006 [-0.0020, +0.0000] |
| nuScenes | +25% | 1 | 3204 / 9181 | 0.0050 | 0.011% | 2.4% | -0.0006 [-0.0022, +0.0000] |
| nuScenes | +25% | 2 | 3205 / 9181 | 0.0050 | 0.000% | 2.4% | +0.0000 [+0.0000, +0.0000] |
| nuScenes | +25% | 3 | 3205 / 9181 | 0.0050 | 0.011% | 2.4% | -0.0006 [-0.0020, +0.0000] |
| nuScenes | +25% | 4 | 3195 / 9181 | 0.0049 | 0.011% | 2.4% | +0.0008 [+0.0000, +0.0028] |
| nuScenes | +50% | 0 | 4755 / 9181 | 0.0110 | 0.011% | 2.4% | -0.0006 [-0.0020, +0.0000] |
| nuScenes | +50% | 1 | 4759 / 9181 | 0.0110 | 0.011% | 2.4% | -0.0006 [-0.0022, +0.0000] |
| nuScenes | +50% | 2 | 4760 / 9181 | 0.0110 | 0.000% | 2.4% | +0.0000 [+0.0000, +0.0000] |
| nuScenes | +50% | 3 | 4760 / 9181 | 0.0110 | 0.011% | 2.4% | -0.0006 [-0.0020, +0.0000] |
| nuScenes | +50% | 4 | 4751 / 9181 | 0.0110 | 0.011% | 2.4% | +0.0008 [+0.0000, +0.0028] |
| nuScenes | +25% | mean of 5 | 3201 / 9181 | 0.0050 | 0.009% | 2.4% | -0.0002 [-0.0012, +0.0006] |
| nuScenes | +50% | mean of 5 | 4757 / 9181 | 0.0110 | 0.009% | 2.4% | -0.0002 [-0.0012, +0.0006] |
| AV2 | +25% | 0 | 6694 / 11818 | 0.0046 | 0.008% | 3.2% | -0.0003 [-0.0010, +0.0000] |
| AV2 | +25% | 1 | 6699 / 11818 | 0.0046 | 0.034% | 3.2% | -0.0006 [-0.0019, +0.0003] |
| AV2 | +25% | 2 | 6696 / 11818 | 0.0046 | 0.025% | 3.2% | -0.0011 [-0.0024, +0.0000] |
| AV2 | +25% | 3 | 6691 / 11818 | 0.0046 | 0.008% | 3.2% | -0.0005 [-0.0017, +0.0000] |
| AV2 | +25% | 4 | 6699 / 11818 | 0.0046 | 0.000% | 3.2% | +0.0000 [+0.0000, +0.0000] |
| AV2 | +50% | 0 | 7159 / 11818 | 0.0085 | 0.017% | 3.2% | -0.0006 [-0.0013, +0.0000] |
| AV2 | +50% | 1 | 7161 / 11818 | 0.0085 | 0.034% | 3.2% | -0.0006 [-0.0019, +0.0003] |
| AV2 | +50% | 2 | 7158 / 11818 | 0.0085 | 0.034% | 3.2% | **-0.0013 [-0.0027, -0.0002]** |
| AV2 | +50% | 3 | 7157 / 11818 | 0.0085 | 0.025% | 3.2% | -0.0010 [-0.0024, +0.0000] |
| AV2 | +50% | 4 | 7164 / 11818 | 0.0085 | 0.008% | 3.2% | +0.0001 [+0.0000, +0.0003] |
| AV2 | +25% | mean of 5 | 6696 / 11818 | 0.0046 | 0.015% | 3.2% | -0.0005 [-0.0014, +0.0001] |
| AV2 | +50% | mean of 5 | 7160 / 11818 | 0.0085 | 0.024% | 3.2% | -0.0007 [-0.0017, +0.0001] |
| Argoverse-HD | +25% | 0 | 8914 / 15006 | 0.0061 | 0.047% | 6.0% | **-0.0016 [-0.0030, -0.0004]** |
| Argoverse-HD | +25% | 1 | 8917 / 15006 | 0.0061 | 0.040% | 6.0% | **-0.0014 [-0.0028, -0.0002]** |
| Argoverse-HD | +25% | 2 | 8908 / 15006 | 0.0061 | 0.053% | 6.0% | -0.0006 [-0.0023, +0.0014] |
| Argoverse-HD | +25% | 3 | 8916 / 15006 | 0.0061 | 0.040% | 6.0% | **-0.0012 [-0.0025, -0.0002]** |
| Argoverse-HD | +25% | 4 | 8914 / 15006 | 0.0062 | 0.040% | 6.0% | **-0.0014 [-0.0029, -0.0002]** |
| Argoverse-HD | +50% | 0 | 10121 / 15006 | 0.0108 | 0.073% | 6.0% | **-0.0024 [-0.0044, -0.0007]** |
| Argoverse-HD | +50% | 1 | 10123 / 15006 | 0.0108 | 0.073% | 6.0% | **-0.0020 [-0.0040, -0.0006]** |
| Argoverse-HD | +50% | 2 | 10119 / 15006 | 0.0108 | 0.087% | 6.0% | -0.0009 [-0.0027, +0.0012] |
| Argoverse-HD | +50% | 3 | 10121 / 15006 | 0.0108 | 0.067% | 6.0% | **-0.0019 [-0.0039, -0.0003]** |
| Argoverse-HD | +50% | 4 | 10119 / 15006 | 0.0109 | 0.060% | 6.0% | **-0.0015 [-0.0027, -0.0005]** |
| Argoverse-HD | +25% | mean of 5 | 8914 / 15006 | 0.0061 | 0.044% | 6.0% | -0.0012 [-0.0027, +0.0001] |
| Argoverse-HD | +50% | mean of 5 | 10121 / 15006 | 0.0108 | 0.072% | 6.0% | -0.0018 [-0.0036, -0.0002] |

## Attribution of the changed actions (summed over the 5 seeds)

| dataset | budget | history | new_object_gt | new_object_spurious | other | truncation |
|---|---|---|---|---|---|---|
| Argoverse-HD | +25% | 0 | 30 | 2 | 0 | 1 |
| Argoverse-HD | +50% | 0 | 48 | 5 | 0 | 1 |
| AV2 | +25% | 0 | 4 | 2 | 1 | 2 |
| AV2 | +50% | 1 | 5 | 5 | 1 | 2 |
| nuScenes | +25% | 0 | 3 | 1 | 0 | 0 |
| nuScenes | +50% | 0 | 3 | 1 | 0 | 0 |

Classes, judged by the decisive object of the side that brakes more:
- history: a v6-S scene object replaced by a new detection without TTC;
- truncation: the decisive new detection touches its crop's border;
- new_object_gt / new_object_spurious: a new detection outside the field that matches / matches no annotated object
  (IoU ≥ 0.3);
- other.

**Reading.**
- The prediction holds on all three splits. Extra pixels outside v6-S's verified region change 0.00–0.09% of actions,
  at least 60× below the reference disagreement rates.
- On nuScenes and AV2 the effect on G is nil: |dG| ≤ 0.0013. Only one of the 20 seed × budget rows on AV2 excludes 0,
  and it is negative.
- On Argoverse-HD the extra crops slightly **lower** G: −0.0012 / −0.0018 (mean over seeds, +25% / +50%); 8 of 10 rows
  exclude 0.
- Most changes come from real objects outside the field (new_object_gt: 30 / 48 on Argoverse-HD). These are objects the
  field judged not decision-relevant. When they are found, the action changes, and on average the decision cost goes
  up. Spurious new detections and truncation account for the rest. History effects are 0–1.
- So the changes are not locality violations that the field should have covered. They are mostly braking on annotated
  objects that the annotated demand does not require (lift errors on newly seen objects).

## Files
- k5_rows.csv: every dataset × budget × seed, plus the seed means.
- k5_attribution.csv: changed frames per class.
- Per-frame records on VUW: results/k35/k5/<ds>/<unit>.pkl.
