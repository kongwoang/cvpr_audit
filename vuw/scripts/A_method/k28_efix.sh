#!/usr/bin/env bash
# K28 family E, nuScenes: one shard process in each of the two round-1 groups ended without "K28 DEV DONE" and without a
# traceback (114 / 152 units written). Re-run of all six shards; finished units are skipped (per-unit resume).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
L=$B/logs/A_method/k28
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
N="$T python $B/scripts/A_method/k28_dev.py --dataset nusc --tag r1 --part base"
SHARE=1 MIN_FREE_MB=7000 $B/run_when_gpu_free.sh $L/r1_nusc_base_rerun.log bash -c "$(for i in 0 1 2 3 4 5; do echo -n "$N --shard $i/6 & "; done) wait"
echo "$(date +%T) K28 E NUSC RERUN DONE units $(ls $B/results/k28/nusc_base_r1 | wc -l)" > $L/efix_done.log
