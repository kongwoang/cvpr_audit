#!/usr/bin/env bash
# DEV set (OpenScene mini, navtest-like filter, every 4th frame): all branches at the frozen crop scale 0.5, then u960
# and the merge-rule variants spec_k050 / spec_k025 (keep confident cheap boxes that a crop did not re-detect), PDMS.
# Method choices are made HERE; navtest is scored once with the chosen variant.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim/dev
mkdir -p $L
until grep -qs "NAVSIM DEV METRIC CACHE DONE" $B/logs/A_navsim/metric_cache_dev.log; do sleep 300; done
S=$B/scripts/A_navsim
X=$B/results/navsim_exp
C="export NAVSIM_SPLIT=mini NAVSIM_CACHE=$X/metric_cache_dev SPEC_CROP_SCALE=0.5"
E=". $B/scripts/common/env_navsim.sh; cd $B/code/risk-aware-perception; $C;"
cp -n $X/dets/ground_z.txt $X/ 2>/dev/null
for i in 0 1; do
  MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/detect_$i.log bash -c "$E NAVSIM_SRC_DETS=$X/dets NAVSIM_DETS=$X/dets_dev python $S/navsim_detect.py --shard $i/2" &
  sleep 200
done
wait
for i in 0 1; do
  MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/detect2_$i.log bash -c "$E NAVSIM_REDO=full960,spec_k050,spec_k025 NAVSIM_SRC_DETS=$X/dets_dev NAVSIM_DETS=$X/dets_dev2 python $S/navsim_detect.py --shard $i/2" &
  sleep 200
done
wait
. $B/scripts/common/env_navsim.sh; eval "$C"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NAVSIM_DETS=$X/dets_dev2 NAVSIM_PDMS_OUT=$X/pdms_dev \
  NAVSIM_TAG=_dev NAVSIM_BRANCHES=reference,cheap,full,full960,spec,spec_k050,spec_k025,uncertainty,sahi,fovea
cd $B/code/risk-aware-perception
for i in $(seq 0 11); do python $S/navsim_pdms.py --stage run --shard $i/12 > $L/pdms_$i.log 2>&1 & done
wait
python $S/navsim_pdms.py --stage summary > $L/pdms_summary.log 2>&1
echo "$(date +%T) NAVSIM DEV CHAIN DONE" >> $L/pdms_summary.log
