#!/usr/bin/env bash
# K30 V6 (i): after the centerlines, the hybrid_pdm + native detections (GPU, 2 shards, SPEC_CROP_SCALE 0.5 as the final
# NAVSIM run), then PDMS of spec_pdm / hybrid_pdm / native with navsim_pdms.py (12 CPU shards) and its summary
# (paired log bootstrap of spec_pdm - each branch)
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k30
X=$B/results/navsim_exp
until grep -qs "K30 NAV CENTER DONE" $L/nav_center_done.log; do sleep 120; done
E=". $B/scripts/common/env_navsim.sh; cd $B/code/risk-aware-perception; export SPEC_CROP_SCALE=0.5 NAVSIM_EXP_ROOT=$X;"
for i in 0 1; do
  SHARE=1 MIN_FREE_MB=4000 $B/run_when_gpu_free.sh $L/nav_detect_$i.log bash -c "$E python $B/scripts/A_navsim/k30_nav_hybrid.py --stage detect --shard $i/2" &
  sleep 200
done
wait
. $B/scripts/common/env_navsim.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NAVSIM_DETS=$X/dets_k30 NAVSIM_PDMS_OUT=$X/pdms_k30 NAVSIM_TAG=_k30 NAVSIM_MAIN=spec_pdm \
  NAVSIM_BRANCHES=spec_pdm,hybrid_pdm,native SPEC_CROP_SCALE=0.5
cd $B/code/risk-aware-perception
for i in $(seq 0 11); do CUDA_VISIBLE_DEVICES="" python $B/scripts/A_navsim/navsim_pdms.py --stage run --shard $i/12 > $L/nav_pdms_$i.log 2>&1 & done
wait
python $B/scripts/A_navsim/navsim_pdms.py --stage summary > $L/nav_pdms_summary.log 2>&1
echo "$(date +%T) K30 NAV CHAIN DONE" >> $L/nav_pdms_summary.log
