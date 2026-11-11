#!/usr/bin/env bash
# nuPlan with REAL crops: wait for every CAM_F0 shard, build the branches (114 + real-crop spec), plan with PDM-Closed
# and IDM (115 hook, 6 shards), then the cells.  Outputs under data/cache/nuplan_real_spec_crops.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/env_nuplan.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SPEC_OUT=nuplan_real_spec_crops RAP_NUPLAN_CAM_F0=$B/datasets/nuplan/cam_f0_flat
until grep -q "EXTRACT ALL DONE" $B/logs/nuplan_extract.log; do sleep 300; done
cd $B/code/risk-aware-perception
OMP_NUM_THREADS=8 python $B/scripts/A_nuplan/nuplan_spec_real.py > $B/logs/nuplan_real_match.log 2>&1 || { echo "match failed"; exit 1; }
for P in pdm_closed idm; do
  for PH in ref branches; do
    for i in 0 1 2 3 4 5; do
      python $B/scripts/A_nuplan/nuplan_spec_counterfactual.py --planner $P --phase $PH --shard $i/6 > $B/logs/nuplan_real_${P}_${PH}_$i.log 2>&1 &
    done
    wait
    python $B/scripts/A_nuplan/nuplan_spec_counterfactual.py --planner $P --phase $PH --shard 0/1 > $B/logs/nuplan_real_${P}_${PH}_summary.log 2>&1
  done
done
python $B/scripts/A_nuplan/nuplan_spec_cells.py > $B/logs/nuplan_real_cells.log 2>&1
echo "$(date +%T) NUPLAN REAL DONE"
