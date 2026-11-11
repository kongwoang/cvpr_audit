# Exact bottleneck-selector score reuse — replication protocol, 2026-10-23

The initial 1,000-token CPU pass found identical scores, decisions, field and
support, with substantial speed improvement. Replicate before relying on timing.
No method/weights change, no detector inference or new quality selection.

Use K34's exact fixed 1,000-token manifest and both 7x7/3x3 grids. Three paired
reference/reuse measurements per token/grid; alternate order with repetition and
token index, five warm-up repeats per process. One CPU thread. Group by log to
load data once, then distribute whole logs by SHA256 prefix modulo 2 to VUW and
VUW1. No GPU use. Every candidate score, action, field and support must match on
every repetition. Validate 7x7 outputs against stored K31 outputs as well.

Timing includes scene/selector construction and field/support extraction, excludes
disk loading, detector and the rest of the perception/planning pipeline. Report
every paired row and host, pooled mean/p95/p99, paired unit bootstrap 2,000 seed0
with repetitions nested inside units, and separate host summaries. No averaging
of per-repetition quantiles and no attribution to Jetson. Shared-host load remains
a limitation; the algebra and output parity are independent of the speed claim.
