# Negative results and limitations (report all of these)

1. **Small frames (KITTI, 0.47 MP): a native-resolution whole-frame pass dominates.** u1248 G 0.288 at 15.5 ms vs ours
   0.226–0.234 at 20–27 ms (Jetson). Best KITTI test pipeline = u1248 + our refiner (0.449 mean of 4 cells). Region
   verification pays off when native resolution is expensive (nuScenes 1.4 MP, AV2 3.2 MP).
2. **NAVSIM tie.** All perception pipelines within ±0.002 PDMS; the dev gain of probing PDM-Closed (+0.003 on OpenScene
   mini) did not transfer to navtest. NAVSIM's PDMS (front camera only, privileged planner) is insensitive to perception
   fidelity except for an oracle router (+0.008).
3. **Probing an argmin planner is expensive.** The NAVSIM configuration calls PDM-Closed ≈53 times per token (~20 s on
   CPU) — not real-time; the braking configurations use a closed-form field (fast).
4. **Latency on small frames.** With the YOLOv8s verifier, ours is slower than one u960 pass on KITTI (20 vs 12.6 ms) —
   the guard's sequential re-detection steps; with a heavier verifier (YOLOv8l) the order flips (earlier measurement:
   v2 23.9 ms vs u960 31.9 ms).
5. **Dev → test non-transfer of tuning:** nuScenes v4 (val-tuned) 0.147 < frozen v3 0.171 on test; KITTI round-2 choice
   0.395 < round-1 0.401 on test. Report the pre-registered / declared choices.
6. **Whole-image AP is lower by design** (AP50 .191 vs u960 .284 on nuScenes); the method optimises decision-relevant
   detection (relevant precision .879 vs .736, 3× fewer phantom-brake detections).
7. **Theory scope:** exactness proven for max-composable consumers only; Planner B / PDM-Closed evaluated empirically
   (98–99% decision agreement with the full detector).
8. **Unseen hazards are rare in the data:** the empirical gain comes from phantom removal and range correction, not from
   discovering new objects (33–45 frames per dataset) — state plainly; the field is the mechanism the theorem needs.
9. **Wide CIs** on KITTI test (6 sequences) and on nuScenes vs u960 / u1280; one nuScenes scene drives large full-model
   losses (sensitivity analysis reported).
10. Fine-tuning the whole-frame detector on AV2 lowered its G (u960-ft 0.402 < 0.446): fine-tuned comparisons are
    reported but the frozen-COCO setting is the main one.
11. **Agreement with π(F) is not exact on real crops** (K2): 95% nuScenes, 81% AV2; exact locality restores 99%.
12. **Necessity is dataset-wise** (K3): the field is necessary on nuScenes (v_close 10) but not on AV2 (v_close 0, where
    support-only ≈ full); the support is necessary on AV2, not significantly on nuScenes.
13. **The field alone recovers few deleted objects on AV2** (K4: 24%), and temporal_crit (previous-frame objects) recovers
    more than ours there (84% vs 79%, at 1.7× ours' pixels).
14. **Second consumer** (K9): same-area uncertainty crops beat ours on AV2 for the TTC warning (0.211 vs 0.139) — but they
    use 1.4–1.6× ours' pixels; at equal pixels ours ≈ uncertainty (n.s. both datasets; nuScenes point estimate −0.044).
15. **Whole-image AP on Argoverse-HD** is 0.163 for ours vs 0.286 (u960) / 0.368 (native) (K7).
16. **Large-frame latency is not a win** (K5/K11, Jetson; K11 decision-identical: nuScenes 25.3 ms vs u960 17.8 ms, AV2 42.0 ms vs u1536 41.1 ms): with the evaluations' crop geometry ours + refiner is on the
    latency–G front (nuScenes 30.2 ms, G 0.173 vs native 35.2 ms, 0.175; AV2 53.0 ms, 0.545 vs native 67.8 ms, 0.553) but
    statistically equal-G whole-frame passes are faster (nuScenes u640 13.4 ms, AV2 u1536 44.3 ms); long tail (p99 82 /
    142 ms); ~6 ms fixed overhead per engine call on this board; a YOLOv8l verifier does not change this (AV2: ours 80.5 ms /
    0.377 vs u1280 93.1 ms / 0.461). The compute claim must be stated in pixels.
17. **No closed-loop gain on Bench2Drive** (K6): ours DS 92.75 ≈ cheap 92.74 ≈ u960 92.63 (n.s.), all ~3 DS below
    privileged perception; the braking field ignores actors irrelevant to braking but PDM-Lite also uses them (lane
    changes, junction yielding); parked props count as phantoms; three runs (ours 92.80 ± 0.63 vs u960 93.15 ± 0.74).
18. **Streaming AP on Argoverse-HD** (K7): ours sAP 0.105 vs u960 0.221 / u1280 0.247 — low offline AP plus a latency tail
    (p95 65 ms on the Jetson; 12% of frames skipped). G and sAP rank the pipelines oppositely.
19. **AV2 hood artefact** (K10): the AV2 G values of Tab. 1 are dominated by detections of the ego hood (81% of the cheap
    loss); masked, ours drops to 0.122 (still the best realizable; whole-frame passes negative) and the oracle frame
    router (0.315) exceeds ours. The paper must present the masked AV2 numbers.
    K13 (masked re-analysis): without the refiner ours (0.031) is not above crop heuristics at equal crop area; the AV2
    fine-tuned detectors (ours-ft + ref, u960-ft, ESOD, ZoomDet) all have NEGATIVE masked G — their unmasked G came from
    not detecting the hood — so the AV2 fine-tuned table must be reported masked, where COCO ours is best.

- **(K14, post-hoc) The refiner, not only the selection.** With the same range refiner applied to every pipeline's support objects, ours final is not significantly above the area-matched image- or corridor-driven crops on any dataset. On Argoverse-HD the ego-corridor tube + refiner is significantly above ours (0.122 vs 0.091). Planner-driven selection is clearly better than random crops and the field alone. Source: handoff/K14.

- **(K19, post-hoc) Missed decelerations.** Ours' gain over whole-frame u960 / native comes from fewer unnecessary braking events. Ours misses more required decelerations than u960 on all three datasets (+0.5 to +8.5 per 1,000 frames). Source: handoff/K19.

- **(K17, post-hoc) The field's net effect.** The field finds objects the draft missed and is decisive where the draft holds no supporting object. On AV2 and Argoverse-HD, though, the extra detections in its crops add false alarms of the same order, so full ≈ support only on AV2 and full < support only on Argoverse-HD (loss 3,498 vs 2,907 on the differing frames). At equal px it is not distinguishable from larger support crops or random crops. Source: handoff/K17.

- **(K27, dev) Noisy looming cue on moving crop canvases; the fix is generic.**
  - Most of v5's unnecessary braking comes from the ttc of small crop boxes that jitter between frames.
  - The robust 3-observation looming of v5.1 removes much of it. The same fix also raises u960 and the heuristics on AV2 (+0.05 to +0.08), so this gain is not specific to planner-driven crops.
  - It withholds the ttc of newly found objects for 2 frames: G on the no-support regime falls on AV2, −0.054 [−0.138, +0.010] (n.s.), and missed events rise 25.9 → 29.3 per 1,000.
  - Source: handoff/K27.

- **(K24, confirmatory) Most of v5.1's test gain is generic; crop selection is matched, not beaten, by component-equipped heuristics.**
  - With the same components (refiner v3a + T4), whole frame and traj_support / corridor / temporal_crit / uncertainty crops reach G within about 0.02 of v5.1-S on AV2 and Argoverse-HD (n.s.).
  - On nuScenes, v5.1-S only matches the previous frozen v3 + v1 (+0.029 [−0.018, +0.070]), and is nominally below the v_close 10 loop with the same components (−0.014 [−0.043, +0.004]). The shared v_close 0 chosen on dev is not confirmed on nuScenes test.
  - The temporal rule trades unnecessary for missed actions: −11 / −36 / −72 vs +3 / +4 / +11 per 1,000.
  - Source: handoff/K24.

- **(K34-3, post-hoc) B2K-A improves tails but does not guarantee a deadline or exact VUW parity.**
  - On AV2, B2K-A still exceeds 100 ms on 1.08% of frames (p99 102.38 ms); its >50 ms share is nominally higher than v6-S by 1.93 percentage points [−1.75, 5.88].
  - Retaining pending detections costs mean +0.41 [0.20, 0.64] ms on nuScenes and +1.99 [1.04, 3.07] ms on AV2 relative to the ordinary bounded variant. Two crop batches can require more than two detector-engine executions, plus the draft.
  - B2K-A differs from the VUW records on 3/805 and 2/586 frames. These same discrepancies recur across all three repetitions. Their precise causes have not been isolated; no numerical-equivalence claim is supported. AV2 intervals cover only eight logs.
  - Source: handoff/K34/REPORT.md, jetson_latency_intervals.csv, jetson_latency_contrasts.csv, jetson_disagreements.csv. K34-1 findings and qualifications are recorded below.

<!-- K34_CPU_AUTO -->
- **K34 CPU caveats (complete).** Source: handoff/K34/REPORT.md and all CSVs. Severity/operating analyses retain every pipeline and threshold; coverage uses actual verified crops, with unknown classes retained. Selector/PDM timings compare different planners and historical hosts.
  - W1 remains: below-hard-demand missed episodes are higher than cheap on AV2 and Argoverse-HD. On Argoverse-HD, high-demand episode loss is also higher: +0.0234 [0.0029, 0.0515] of cheap total dataset loss (ten episodes).
  - Deployed v6-S is not dominated by sampled whole-frame points on nuScenes/AV2, but u960 + C1 + v3a at threshold 0.8 dominates its Argoverse-HD point estimate; false-alarm difference CI includes zero. Fine car/truck/bus coverage is unavailable on Argoverse-HD generic annotations.
  - nuScenes v6-S minus native insufficient-braking loss/frame: +0.0020 [-0.0551, +0.0494].
  - AV2 v6-S minus native insufficient-braking loss/frame: -0.0066 [-0.0737, +0.0518].
  - Argoverse-HD v6-S minus native insufficient-braking loss/frame: +0.0018 [-0.0130, +0.0148].
