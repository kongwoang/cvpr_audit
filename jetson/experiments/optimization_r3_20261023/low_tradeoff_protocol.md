# Addendum: reduce the cost of uncertainty fallback

After complete nuScenes dev results: learned-point missing-history fallback rescues
3/99 episodes (all with decisive-object braking evidence), adds1.471 false episodes
/1000, improves G by .0143 [.00094,.03390]. Upper-quantile fallback rescues9/99 but
adds7.357 false episodes/1000. One rescue has no decisive-object evidence; it remains
counted but is not a mechanism repair. No AV2 full-result selection occurred.

Freeze three focused alternatives, all B2K-A+recovery and original C1 on valid
chains. Same training checkpoints and unchanged action thresholds:

1. **Point plus extent persistence:** point fallback only on missing chains, plus
   the previously specified at-most-one previous critical verified object seed.
   Individual point and extent controls are retained; this tests complementarity.
2. **Evidence-only point:** use point fallback only with >=2 real observations in
   the masked head. A current-only image estimate leaves the original unknown
   TTC infinity. This tests how much apparent gain/false braking comes from a
   single-image learned motion prior rather than temporal evidence.
3. **Corroborated upper:** start with point fallback on missing chains. Use the
   uncalibrated head's q90 only if (a) at least two real observations are available,
   (b) current verified detection has a class-aware greedy IoU>=.3 cheap match,
   (c) cheap TTC is finite positive, and (d) substituting that TTC into the CURRENT
   verified raw geometry gives that object's demand>=1. Use the same frozen
   planner to test corroboration. No current GT, future frame or additional
   detector call. This is a heuristic, not independent corroboration or a guarantee:
   the modalities share the same camera and can share systematic errors.

Report original point and upper rows alongside all new rows with independent
in-loop histories. All-object masks/fallback type are recorded for mechanism
audits. Dev only, paired-unit bootstrap2000 seed0, no promotion. Additional
iterations are exploratory and require a new untouched evaluation before claims.
