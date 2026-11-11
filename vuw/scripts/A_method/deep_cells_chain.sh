#!/usr/bin/env bash
# DEEP's nuScenes old-85 cells (brake x mono / oracle range) for the crop baselines vs v3, all 85 scenes (train+val+test),
# strict budget; u960 / u640 references come from DEEP's cell tables and the oracle eval of v3.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/deep_cells_old85.log
$B/run_when_gpu_free.sh $L python scripts/A_method/baselines_eq.py --set old --split train+val+test \
  --ref eval__old_train+val+test_dump.pkl --threads 4 --strict --arms uncertainty temporal_crit random v3 --tag _cells
echo "DEEP CELLS CHAIN DONE" >> $L
