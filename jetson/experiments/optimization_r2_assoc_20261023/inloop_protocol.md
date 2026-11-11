# Identity follow-up — after direct recovery trace inspection

Case 01 proves a same-class ID switch: current GT track 21 is linked to track 25
in both historical frames (GT/detection IoU .93/.91). The original unique-class
IoU gate cannot detect that switch. Case 05 rejects a dense set containing a
high-overlap box. These are distinct from head regression error.

Before fresh GPU inference, evaluate **all original recovery attempts in the
nine diagnostic units**, not only successful or illustrative examples. Match
current and historical detections to GT by the frozen class-compatibility rule;
report unknown/unmatched cases separately. Use no GT inside either proposal:

1. Rank-margin association: same-class top IoU >=.5, top-minus-second >=.2.
2. Optical-flow location followed by the same rank-margin check. Bidirectional
   Lucas–Kanade in a context-4 local image patch, longest side <=384; at most
   40 Shi–Tomasi corners, >=6 consistent within 1 pixel, robust similarity with
   >=6 and >=60% inliers, allowed scale [.75,1.33]. Parameters are fixed engineering
   tolerances, not calibrated confidence bounds. No sweep or test selection.

The CPU screen tests associations to the already stored historical detections;
an unmatched prediction may require a shifted historical crop. It does not
prove new inference quality. A fresh flow recovery may translate the existing
canvas to the predicted target centre, retaining its pixel dimensions and the
one-target/two-historical-execution cap. Any admitted chain must consist entirely
of actual detections. Failed flow falls back to rank-margin at the unshifted
target; it never fabricates a box observation. Account for optical-flow CPU work.

This follow-up is separate from the already running five-row in-loop experiment;
do not mutate its code, inputs or results. Preserve all attempts and negatives.

This GPU pilot runs only the rank-margin proposal, with real historical inference.
