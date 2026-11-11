# Addendum: put jointly learned state inside the loop

Architectural audit during the first adapter screen: its training objective uses
the learned range factor on every matched object, whereas the conservative runtime
integration applies that factor only to the final raw support, as v3a does. A
range underestimate/overestimate that leaves raw action KEEP can therefore never
be repaired by that final support-only refiner. Good all-object validation demand
does not establish a benefit under this restricted runtime. Both integrations
must be reported, rather than interpreting a negative support-only result as a
complete test of joint state estimation.

New fixed candidates: B2K-A+recovery with the geometry adapter, or the decision
adapter, applying point rate AND ray-coupled range/lateral factor to every eligible
verified object at EACH lift call. Recompute field and support from this state;
each lift starts from original geometric estimates (never compound corrections).
No additional v3a refinement is applied. No weight retraining or threshold change.
Retain the previous support-only/rate-only rows as ablations. Original C1/v3a and
round-1 controls replay independently with exact parity checks.

This changes the experimental method's internal state and queries; frozen-query
postprocessing cannot substitute for fresh dev inference. Same 152/300 dev units,
all rows and paired-unit bootstrap 2,000 seed0. Compare misses, severity, false
alarms, loss/G, pixels and calls. No automatic main-code promotion. Counts/time
must account for more encoded missing-history objects and the eliminated v3a head.
