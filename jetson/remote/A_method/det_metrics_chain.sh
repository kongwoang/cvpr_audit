#!/usr/bin/env bash
# Fidelity (B): detection metrics of every method on new test -- smoke (2 scenes), then every 4th keyframe; waits for
# the LZU chain so LZU-ft / plainrect-ft are included.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/det_metrics.log
$B/run_when_gpu_free.sh $L python scripts/A_method/det_metrics.py --max_scenes 2 --tag _smoke
tail -3 $L | grep -q "DET METRICS DONE" || { echo "DET METRICS SMOKE FAILED" >> $L; exit 1; }
until grep -q "LZU CHAIN DONE\|LZU ABORT" logs/A_method/lzu/*.log 2>/dev/null; do sleep 300; done
$B/run_when_gpu_free.sh $L python scripts/A_method/det_metrics.py --step 4
echo "DET METRICS CHAIN DONE" >> $L
