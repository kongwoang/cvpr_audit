# Fixed bridge/acquisition screen

Before dev inference: controls cheap, frozen v6-S and round-1 B2K-A+recovery;
new policies B2K-A+bridge, bridge+recovery, cheap-motion transfer+recovery,
previous critical-extent retention+recovery, motion-conditioned field+recovery.
Every policy has independent history and no-refiner/v3a output. Round-1 also
reports half-log correction and two TTC/range-coupling sensitivities.

Bridge: for a verified detection lacking a 3-observation chain (after recovery,
when enabled), greedily associate same-class detections at IoU >= .3 to the
previous cheap frame and then the earlier cheap frame. Add actual boxes only;
use original C1 on the three original-image ROI crops. This is intentionally
a mixed-fidelity distribution-shift control, not claimed calibrated inference.
Transfer: on missing chains only, take the finite TTC of a same-class current
cheap match. Do not overwrite an existing chain whose measured rate was zero.

Extent: seed at most one crop of the preceding frame's verified object with
largest refined demand >=1. Re-localize by current cheap class/IoU match if
available, else the previous box; context 2. No GT or future frame. This keeps
real observation of a previously relevant object through a changing field.
Motion field: union original zero-closing field and a field whose closing speed
is the largest finite cheap z/TTC, clipped [0,10], among objects within 80m and
lateral extent intersecting [-3.2,3.2]. No available cue means the original field.
This differs from K30's unconditional vc5/vc10 setting.

All bounded policies preserve two rounds, but may request more crops/pixels.
Support and refiner are recomputed at every altered inference state. Refiner
sensitivities scale TTC by range correction c or sqrt(c), retaining range/lateral
coupling; they express fixed-speed/intermediate assumptions and can be worse.
All rows retained. No test data or promotion. Timing requires later measurement.
