#!/usr/bin/env bash
# NAVSIM-v1 PDM Score of every perception branch on the 1,440 nuPlan-mini states (CPU, 6 shards), then the summary.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/env_nuplan.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SPEC_OUT=nuplan_real_spec_crops RAP_NUPLAN_CAM_F0=$B/datasets/nuplan/cam_f0_flat
L=$B/logs/A_nuplan
mkdir -p $L
cd $B/code/risk-aware-perception
for i in 0 1 2 3 4 5; do
  python $B/scripts/A_nuplan/nuplan_pdms.py --shard $i/6 > $L/pdms_$i.log 2>&1 &
done
wait
python $B/scripts/A_nuplan/nuplan_pdms.py --shard 0/1 > $L/pdms_summary.log 2>&1
echo "$(date +%T) PDMS CHAIN DONE" >> $L/pdms_summary.log
