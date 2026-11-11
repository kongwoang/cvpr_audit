# Addendum: uncertainty-triggered spatial persistence

Diagnostic learned pilot: the upper-quantile missing-history row first brakes for
the correctly matched crossing truck in scene-0461 frame7 (predicted demand1.866,
GT4.489), whereas round-1 KEEP. It still misses the truck at the peak-demand frame8.
This counts as an L1 episode rescue, not adequate hard braking or safety success.
Its other nuScenes rescue (scene-0366 frame10) still lacks the decisive object and
must not be called a mechanism repair.

Test a composition already implementable without new thresholds: upper-quantile
missing-history inference PLUS retention of the previous verified braking object's
extent (the fixed context2, at most one seed policy from bridge_protocol.md).
Extent-only was inactive in this episode because the preceding estimate said KEEP;
the uncertainty hypothesis can activate causal spatial persistence next frame.
Apply universally on dev, not just the inspected case. Run both raw-quantile and
holdout-calibrated versions. Additional crops/false-alarm persistence are explicit
risks. All original individual components and compositions are retained.

The raw composition is included with the new joint-lift suite; the calibrated
composition is in a separate persistence suite so existing calibrated pilot rows
are not changed in flight. No test selection, core edits or new image encoder.
