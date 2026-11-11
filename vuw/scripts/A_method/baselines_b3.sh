#!/usr/bin/env bash
# Audit-driven equal-budget baselines: trajectory corridor + support, temporal criticality ROI (GPU, new test + dev).
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
$B/run_when_gpu_free.sh logs/A_method/baselines_b3_newtest.log python scripts/A_method/baselines_eq.py --threads 4 \
  --arms traj_support temporal_crit v3 --tag _b3
$B/run_when_gpu_free.sh logs/A_method/baselines_b3_dev.log python scripts/A_method/baselines_eq.py --set old --split train+val \
  --ref eval__old_train+val_v3_h1.6.pkl --threads 4 --arms traj_support temporal_crit v3 --tag _b3
echo "$(date +%T) B3 DONE" >> logs/A_method/baselines_b3_dev.log
