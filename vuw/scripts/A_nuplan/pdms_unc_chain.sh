#!/usr/bin/env bash
# nuPlan PDMS for the uncertainty crop baseline (strictly within the spec branch's pixels per image), same pipeline.
# per slice, FOVEA KDE_I at the FULL mode's canvas), GPU; then NAVSIM-v1 PDMS for every branch (CPU, 6 shards).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
E="export SPEC_OUT=nuplan_real_spec_crops_unc EXTRA_MODES=uncertainty RAP_NUPLAN_CAM_F0=$B/datasets/nuplan/cam_f0_flat"
L=$B/logs/A_nuplan
mkdir -p $L
MIN_FREE_MB=8000 $B/run_when_gpu_free.sh $L/unc_match.log bash -c ". $B/env_nuplan.sh; $E; export OMP_NUM_THREADS=8; \
  cd $B/code/risk-aware-perception; python $B/scripts/A_nuplan/nuplan_spec_real.py"
[ -f $B/code/risk-aware-perception/data/cache/nuplan_real_spec_crops_unc/branches.pkl ] || { echo "PDMS UNC ABORT match" >> $L/unc_match.log; exit 1; }
. $B/env_nuplan.sh; eval "$E"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $B/code/risk-aware-perception
for i in 0 1 2 3 4 5; do
  python $B/scripts/A_nuplan/nuplan_pdms.py --shard $i/6 > $L/pdms_unc_$i.log 2>&1 &
done
wait
python $B/scripts/A_nuplan/nuplan_pdms.py --shard 0/1 > $L/pdms_unc_summary.log 2>&1
echo "$(date +%T) PDMS UNC CHAIN DONE" >> $L/pdms_unc_summary.log
