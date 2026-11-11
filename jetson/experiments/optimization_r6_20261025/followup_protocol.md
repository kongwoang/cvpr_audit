# Limited composition follow-up, 2026-10-25

Declared after viewing all nuScenes stage-1 outcomes, before viewing AV2 outcomes.
Exploratory and adaptively motivated; not independent validation.

Stage 1's `track_agree1` has two new correctly matched HARD-action frames in
scene-1049 relative to R5 q=.5, plus two new false-braking frames (one HARD).
It loses R5's single mild-episode rescue if substituted. Most other combinations
increase false braking. This motivates exactly two additional compositions:
max(R5 q=.5, track_agree1) and max(R5 q=.5, track_agree2), on both full datasets.
No threshold adjustment, no new query, and no fitting. The two-lag version is a
declared more conservative control, not chosen based on unseen AV2 quality.

Report both, with all original 20 rows and seven controls; stage-1 results remain
intact. Compare to R5 and R3, including false HARD exposure, matched evidence,
bootstrap intervals, and unit concentration. Any gain in this small selected
family still needs independent validation before promotion.

Implementation clarification: cross-crop duplicate suppression in collect.py
uses the stored coarse class (vehicle/person/cyclist) at IoU .65. Original
per-crop detector uses its unchanged COCO-class NMS. The high-proposal parity
check is before the extra cross-crop suppression. Original AV2 mask is the
geometry wrapper, not pixel blacking. These are exact code descriptions, not
new quality configurations.
