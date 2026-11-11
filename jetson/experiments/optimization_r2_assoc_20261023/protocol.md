# Isolated rank-margin recovery pilot

Fork of the frozen round-2 inference harness, after the documented identity audit.
Only historical recovery association changes: same-class best IoU >=.5 and
best-minus-second >=.2. Current trigger, one target, two past execution cap,
real detections, C1/refiner checkpoints, thresholds and geometry are unchanged.
No optical flow in this GPU pilot. All nine diagnostic units, full sequences;
original v6-S and round-1 combined controls must replay prior outputs exactly.
All three acquisitions have none/v3a/half-log controls. No main source change.
Smoke is diagnostic and selected by earlier failures, not an unbiased quality
sample. Any broader quality claim requires the full 152/300 development set.
Runs via unchanged GPU wrapper SHARE=1 MIN_FREE_MB=10000, two workers. CPU1
analysis, paired-unit bootstrap 2,000 seed0; retain every row and failed attempt.
