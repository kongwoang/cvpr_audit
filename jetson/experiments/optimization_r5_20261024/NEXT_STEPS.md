# Feasibility after R5

No candidate is promoted. The most defensible new component to retain for
further development is conditional preservation of current cheap evidence after
a failed verification, as an addition to the strongest R3 acquisition. Its
detector budget is unchanged, but its CPU overhead and Jetson parity remain
unmeasured. The q=.5 gain is real on these records and concentrated in few units;
q=.8 is not a validated default merely because nuScenes false counts stay equal.

| Component | Evidence from this round | Appropriate next step |
|---|---|---|
| Negative verification / merge | A high-resolution crop can lose a real night car or near edge bus already detected by cheap. Retaining everything creates many false brakes. | Study a bounded, interpretable negative-evidence rule with actual observation persistence and acquisition context; train/evaluate on the deployed acquisition. Keep raw cheap and verified provenance separate. |
| Temporal fallback | New tiny tree gives some mild-episode rescues, but the R3 point+extent still has lower severe-frame loss. | Retain R3 as a control. Require genuine motion/association evidence for hard-demand claims; a single-image prior cannot establish closing speed. |
| Refiner | Selected disagreement population is poorly calibrated even when overall predictions look strong; conditional models generalize weakly. | Collect disagreement-specific evidence under the actual B2K-A/extent acquisition on train only, including useful versus harmful downgrades. Validate by unit, then one fixed dev screen. Do not add another larger head without resolving this distribution shift. |
| Object extent / association | A cab and trailer can be separate detector boxes while one annotation spans the vehicle. IoU labels do not identify all physical parts. | Audit fragmented-object positives and matched-only negatives before retraining. Any merge needs current visual/temporal identity evidence; indiscriminate unions would create phantom corridor occupancy. |
| Near/edge visibility and field | Some high-loss decisive boxes touch image boundaries or the ego mask. This does not establish annotation error or that more crop resolution will reveal the object. | Separate feasible current-view failures from uncertain visibility, without excluding original GT. Evaluate targeted acquisition changes only where current/past images supply recoverable evidence. |
| Selector/consumer | A substantial portion of required-frame loss occurs even with the reference discrete action because commanded deceleration is finite. | Keep this residual separate from perception misses. Do not tune the controller or change GT to make an optimization look better. |

The next substantive experiment should aim at evidence availability/identity on
rare hard cases, not another global confidence threshold. These are future
research hypotheses, not pending jobs or claims of solved weaknesses. All work
specified for R5 is complete. A new latency study is useful only after a quality
candidate is chosen with its tradeoff explicit; no speed benefit is inferred
from unchanged detector pixels or the absence of another CNN.
