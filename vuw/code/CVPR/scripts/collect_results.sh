#!/usr/bin/env bash
# Gather the final tables of every exploratory run into docs/overnight_results.md (safe to re-run).
cd "$(dirname "$0")/.."
OUT=docs/overnight_results.md
{
echo "# Overnight results (auto-collected $(date '+%F %T'))"
echo
echo "Raw tables copied from the logs; interpretations live in docs/directions_log.md. Runs: results/raw/."
for f in logs/speculative_*.log logs/spec_refine_*.log logs/realspec*.log logs/realcrop_*.log logs/ensemble_*.log \
         logs/learned_refiner*.log logs/region*.log logs/secondlook_*.log logs/reuse_*.log logs/range_amb_*.log \
         logs/stakes_*.log logs/bayes_*.log logs/headroom_iou03_*.log logs/ensemble_bootstrap.log; do
  [ -f "$f" ] || continue
  echo; echo "## $f"; echo; echo '```text'
  grep -v "RuntimeWarning\|errs.append\|lift_full\|\"refined\"\| done$\|start scripts\|done scripts" "$f" | tail -n 22
  echo '```'
done
echo; echo "## Crop engine timings (trtexec, FP16, Jetson AGX Xavier)"; echo; echo '```text'
for t in engines/time_*.log; do [ -f "$t" ] && echo "$t: $(grep 'GPU Compute Time' "$t" | head -n 1 | sed 's/.*GPU Compute Time: //')"; done
echo '```'
} > "$OUT"
echo "wrote $OUT"
