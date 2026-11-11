#!/usr/bin/env bash
# Baseline (6): UFP (UFPMP-Det official packing) -- smoke on 3 scenes, then the full new test.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/ufp.log
$B/run_when_gpu_free.sh $L python scripts/A_method/ufp_baseline.py 3
tail -3 $L | grep -q "UFP DONE" || { echo "UFP SMOKE FAILED" >> $L; exit 1; }
$B/run_when_gpu_free.sh $L python scripts/A_method/ufp_baseline.py
echo "UFP CHAIN DONE" >> $L
