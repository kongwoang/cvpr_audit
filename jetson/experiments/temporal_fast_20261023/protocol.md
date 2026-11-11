# Exact recovery bookkeeping optimization — 2026-10-23

Measured motivation: three-repetition Jetson recovery timing shows nontrivial
average overhead despite triggers on only 1–3% of timed frames. The existing
helper repeats chains already computed by the caller, performs cheap-object IoU
matching before corridor/relevance filters, and rebuilds history arrays even
without acquired observations. Do not assume all overhead is removable: genuine
historical detection, feature encoding and retained-image memory still cost.

Intended invariant: same eligibility, priority, detector inputs/order, real past
images, recovered boxes, C1 inputs, actions and executions. No new gate or model.
If all cheap demands are below thr_decel, the existing candidate conjunction is
false for every object, so skip recovery/logging for that frame. Continue to
retain the same two per-pipeline uploaded images for possible future use.
Reuse the caller's exact chain list, evaluate independent rejection conditions
before expensive matching, and avoid rebuilding unchanged history arrays.

Gate: randomized CPU semantic parity with original helper, followed by real
end-to-end dev parity against original helper for every output action, cost,
pixel and execution count. Any difference blocks timing pending diagnosis.
Retain all failed attempts. All code is in this separate experiment.

Fixed pilot timing: first 456 chronological nusc_val/av2_dev frames, discard 200
=256 timed frames; three paired original/fast repetitions, alternating order.
Four method rows remain v6-S, recovery, B2K-A, B2K-A+recovery. Same trt32/reference
crop runtime, ego mask 1830. These small prefixes assess bookkeeping only; the
existing 512-frame method timing and full VUW development quality remain primary.
Pool quantiles; paired unit bootstrap 2000 seed0, repetitions nested in units.

Jetson exclusive window, existing engines only, source/engine hashes and control
restoration. No main-source edit, policy selection, new detector work or build.
