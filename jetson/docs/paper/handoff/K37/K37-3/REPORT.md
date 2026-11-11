# K37-3 — why AnA-style iterated recovers more natural misses than v6-S on nuScenes (post-hoc)

## Population
The K33 tab:missed natural misses (k33_report.missed_natural): on the evaluated test frames, decision-relevant annotated
objects that the cheap draft misses and u960 + comp finds. Recovery = found_<row> in the stored K33 records, frozen
matching.
- The population reproduces tab:missed exactly (v6-S / AnA): nuScenes **0.607 / 0.718** (117 objects); AV2
  0.848 / 0.808 (396); Argoverse-HD 0.873 / 0.865 (1,027).
- AnA = ana05hinfi (t 0.05, T_h ∞, iterated).

## Classes
- **(i) Low-confidence draft box.** A box of the frozen K33 pass-1 draft (YOLOv8s at the cheap mode, conf ≥ 0.05 = AnA's
  draft threshold) with conf in [0.05, 0.25) and IoU ≥ 0.3 with the object. AnA verifies such a box; v6-S (draft
  threshold 0.25) never sees it.
  - These boxes were never stored. With the user's approval (2026-10-26), the pass-1 draft was re-run once, detector
    only, on the 114 / 360 / 1,019 population frames (scripts/A_method/k37_3.py detect).
  - Check: the re-run's boxes with conf ≥ 0.25 equal the stored draft exactly on AV2 (360/360 frames) and Argoverse-HD
    (1,019/1,019), where the stored draft was produced online by the same code.
  - On nuScenes the stored draft is the DEEP cache. Exact equality holds on only 28/114 frames, but the box counts agree
    on 113/114 frames and every cached box has a re-run box with IoU ≥ 0.9; the differences are sub-pixel coordinates.
    So (i) on nuScenes is defined on the online pass-1 draft that AnA itself uses, which is what the class needs.
- **(ii) Position relative to v6-S's executed crops** (K30 match records). The object's box centre is inside a field crop
  (field_v6S); else inside a support / guard crop; else outside all of v6-S's crops.

## Recovery per class (k37_3_classes.csv; per object: k37_3_objects.csv)

| dataset | class | objects | v6-S | AnA | AnA − v6-S | only AnA / only v6-S |
|---|---|---|---|---|---|---|
| nuScenes | all | 117 | 0.607 | 0.718 | +0.111 | 25 / 12 |
| nuScenes | (i) low-conf box | 78 | 0.603 | **0.846** | +0.244 | 22 / 3 |
| nuScenes | (i) none | 39 | 0.615 | 0.462 | −0.154 | 3 / 9 |
| nuScenes | (ii) field crop | 78 | 0.692 | 0.718 | +0.026 | 13 / 11 |
| nuScenes | (ii) support / guard crop | 26 | 0.654 | 0.769 | +0.115 | 4 / 1 |
| nuScenes | (ii) outside v6-S crops | 13 | 0.000 | 0.615 | +0.615 | 8 / 0 |
| AV2 | all | 396 | 0.848 | 0.808 | −0.040 | 30 / 46 |
| AV2 | (i) low-conf box | 260 | 0.892 | 0.885 | −0.008 | 21 / 23 |
| AV2 | (i) none | 136 | 0.765 | 0.662 | −0.103 | 9 / 23 |
| AV2 | (ii) field crop | 269 | 0.926 | 0.766 | −0.160 | 1 / 44 |
| AV2 | (ii) support / guard crop | 102 | 0.853 | 0.902 | +0.049 | 7 / 2 |
| AV2 | (ii) outside | 25 | 0.000 | 0.880 | +0.880 | 22 / 0 |
| Argoverse-HD | all | 1,027 | 0.873 | 0.865 | −0.009 | 98 / 107 |
| Argoverse-HD | (i) low-conf box | 754 | 0.875 | 0.943 | +0.068 | 84 / 33 |
| Argoverse-HD | (i) none | 273 | 0.868 | 0.648 | −0.220 | 14 / 74 |
| Argoverse-HD | (ii) field crop | 883 | 0.933 | 0.856 | −0.077 | 35 / 103 |
| Argoverse-HD | (ii) support / guard crop | 100 | 0.730 | 0.910 | +0.180 | 22 / 4 |
| Argoverse-HD | (ii) outside | 44 | 0.000 | 0.932 | +0.932 | 41 / 0 |

The cross-classes (i) × (ii) are in k37_3_classes.csv. On nuScenes, the (i) low-conf × outside class has 8 objects:
v6-S 0, AnA 7.

## Reading
- **AnA's nuScenes advantage comes from verifying low-confidence draft boxes.**
  - Two-thirds of the natural misses (78/117) have a conf 0.05–0.25 draft box on the object. There AnA recovers 0.846 vs
    v6-S 0.603 (22 vs 3 exclusive recoveries).
  - Without such a box, v6-S is better: 0.615 vs 0.462.
- **Position.** Part of the gain is outside v6-S's crops. v6-S cannot recover an object outside every crop it executes
  (0 by construction); AnA recovers 8 of the 13 such objects, 7 of them through a low-confidence box.
  - Inside v6-S's field crop the two are close on nuScenes: 0.692 vs 0.718.
- **AV2 and Argoverse-HD.** The same mechanism appears. AnA is better outside v6-S's crops and in support / guard crops,
  and better with a low-confidence box on Argoverse-HD.
  - But v6-S's field crop recovers clearly more inside the field (AV2 0.926 vs 0.766; AHD 0.933 vs 0.856), so v6-S
    leads overall.
  - nuScenes differs because v6-S's own recovery is much lower there, both inside the field crop (0.692 vs 0.926 on AV2
    and 0.933 on Argoverse-HD) and on objects with a low-confidence box (0.603 vs 0.892 / 0.875).
  - The class shares are similar across datasets. A low-confidence box is present for 67% / 66% / 73% of the objects;
    the object lies inside the field crop for 67% / 68% / 86%.
  - So AnA's draft-box verification fills a gap that v6-S's verified crops leave on nuScenes.
- **Implication.** The two selectors are complementary:
  - the field finds objects the draft did not hint at;
  - low-confidence draft boxes point at objects the field does not request.
  This is descriptive and post-hoc; no method change follows from it.
