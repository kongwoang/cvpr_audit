**Result** (post-hoc analysis; frozen configuration stays the reported one; AV2 with the ego mask unless stated)
- **(1) Probe templates and speeds: no setting is better on both dev and test.**
  - **nuScenes:**
    - dev favours the truck / union probe at v_close 0 (G with refiner 0.280 vs frozen car @10: 0.263), but on test it is lower (0.156 vs 0.170);
    - the pedestrian probe fails on both splits (dev −0.072, test −0.092);
    - deletion recovery on test: car / truck ≥ 5 m/s recover 0.96–0.97 of deleted objects, the pedestrian probe 0.51–0.57.
  - **AV2:**
    - dev favours the pedestrian probe (0.145 vs frozen car @0: 0.104), but on test it is slightly lower (0.115 vs 0.122);
    - faster probes lower G on both splits (car @5/10: 0.070 dev, 0.081 test) but raise recovery (0.87 → 0.94; cyclists 0.29 → 0.81);
    - the pedestrian probe recovers few pedestrians / cyclists (0.35 / 0.19–0.33).
- **(2) Second detector family (YOLO11s as draft and target, no re-tuning).**
  - **nuScenes:** ours 0.364, 0.388 with the existing refiner, is on par with u960 (0.368) and native (0.362) at 0.10 vs 0.38 / 1.03 px; ours + refiner − native [−0.027, +0.123].
  - **AV2:** ours + refiner 0.139 is above u960 (−0.041) [+0.120, +0.245] and native (−0.121) [+0.145, +0.376].
  - **Refiner transfer.** The YOLOv8s-trained refiner transfers to YOLO11s boxes on AV2 (+0.079 [+0.040, +0.121]), not significantly on nuScenes (+0.025 [−0.004, +0.071]). This differs from the K5 follow-up, where refiner v2 did not transfer to YOLOv8l.
- **(3) Ego-mask sensitivity.**
  - **Eight of nine settings agree with the reported mask:** band start row 1780 / 1830 / 1880 × containment 0.3 / 0.5 / 0.7. Ours' G ranges over 0.121–0.124, u960 over −0.022 to −0.018, and the draft's action changes on ≤ 0.06% of frames.
  - **Only (1880, 0.7) lets hood detections through.** The draft's action changes on 3.2% of frames, and every cheap-referenced G jumps (ours 0.213, u960 0.077). Even there, ours' own action changes on 0.01% of frames.
  - **The comparisons are stable:** ours − u960 [+0.09, +0.20] in every setting.

# K21: robustness checks

**(1) Probe templates × closing speeds** (`probes_{nusc,av2}.csv`)
- **Setup.** Field probes: pedestrian 0.6×1.8 m, car 1.8×1.6 m (frozen), truck 2.6×3.6 m and their union (footprint = bounding box of the union), each at v_close ∈ {0, 5, 10, 15} m/s. Script `k17_variants.py` arms `vc<v>_<probe>`; everything else is frozen.
- **Splits.** Development: nuScenes new val (152 scenes) and AV2 dev (the train subset of K13(g); 128 logs). Test: run once.
- **Recovery** is the share of deleted decision-relevant objects found again in the K4 controlled deletion on test, by class. Deleted objects:
  - nuScenes: 810 vehicles, 5 persons, 8 cyclists;
  - AV2: 2,882 vehicles, 20 persons, 21 cyclists.

  The person / cyclist rates rest on very few objects.
- **Observations:**
  - v_close 10 and 15 give identical results, because the field saturates at 10 m/s;
  - the union probe equals the truck probe everywhere, because the truck footprint contains the others;
  - agreement with π(F) moves by at most 0.01.
- **Choice on dev.** Choosing on dev would have picked truck @0 for nuScenes and ped @0 for AV2; both are below the frozen configuration on test. **The frozen car probe stays.**

**(2) Second detector family** (`detector2_{nusc,av2}.csv`; `k21_det2.py`)
- **Pipeline.** YOLO11s (COCO; not a YOLOv8 variant) serves as the draft (long side 320) and as the crop / whole-frame detector, with the frozen loop. On nuScenes the draft is computed online with the same lift as every other pass, unlike the cached YOLOv8s draft.
- **Reference.** The YOLOv8s numbers from K14 are in the same CSVs. Intervals are ours (without / with refiner) minus each whole-frame resolution.
- **Reading.** With YOLO11s every pipeline's G is higher on nuScenes (the draft is weaker relative to the full passes). The ordering "ours ≥ whole frame at a fraction of the px" holds on both datasets; on nuScenes it is a tie.

**(3) Ego-mask sensitivity on AV2 val** (`mask_sensitivity_av2.csv`; `k21_mask.py`)
- **Setup.** All nine settings are run in one pass. Detections of identical passes and crops are shared across settings, while lift histories are kept per setting.
- **Parity.** The reported setting (1830, 0.5) reproduces K13 exactly: ours 0.1223, u960 −0.0207, native −0.0578, uncertainty 0.0843.
- **Columns.** Each pipeline's G is referenced to the draft under the same mask. The table also gives the share of frames whose action changes relative to the reported mask, and the intervals of ours − u960 and ours − uncertainty crops.
