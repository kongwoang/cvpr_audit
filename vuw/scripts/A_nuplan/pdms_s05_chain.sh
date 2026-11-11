#!/usr/bin/env bash
# nuPlan PDMS rerun with v3 as FROZEN (crop scale 960/1920 = 0.5) + u960 (full960) + baselines, same pipeline.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
E="export SPEC_OUT=nuplan_real_spec_crops_s05 EXTRA_MODES=full960,uncertainty,sahi,sahi_s033,fovea SPEC_CROP_SCALE=0.5 RAP_NUPLAN_CAM_F0=$B/datasets/nuplan/cam_f0_flat"
L=$B/logs/A_nuplan
mkdir -p $L
MIN_FREE_MB=8000 $B/run_when_gpu_free.sh $L/s05_match.log bash -c ". $B/env_nuplan.sh; $E; export OMP_NUM_THREADS=8; \
  cd $B/code/risk-aware-perception; python $B/scripts/A_nuplan/nuplan_spec_real.py"
[ -f $B/code/risk-aware-perception/data/cache/nuplan_real_spec_crops_s05/branches.pkl ] || { echo "PDMS S05 ABORT match" >> $L/s05_match.log; exit 1; }
. $B/env_nuplan.sh; eval "$E"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $B/code/risk-aware-perception
for i in 0 1 2 3 4 5; do
  python $B/scripts/A_nuplan/nuplan_pdms.py --shard $i/6 > $L/pdms_s05_$i.log 2>&1 &
done
wait
python $B/scripts/A_nuplan/nuplan_pdms.py --shard 0/1 > $L/pdms_s05_summary.log 2>&1
echo "$(date +%T) PDMS S05 CHAIN DONE" >> $L/pdms_s05_summary.log
