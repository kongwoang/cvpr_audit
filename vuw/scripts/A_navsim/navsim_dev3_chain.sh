#!/usr/bin/env bash
# DEV round 2: crop-scale sweep with and without keeping un-re-detected cheap boxes (branches spec_sXXX[_kYYY]),
# detections added to the round-1 dev detections; PDMS for the new branches next to reference / u960 / spec / spec_k025.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim/dev3
mkdir -p $L
S=$B/scripts/A_navsim
X=$B/results/navsim_exp
V=spec_s075,spec_s075_k025,spec_s060_k025,spec_s100_k025
C="export NAVSIM_SPLIT=mini NAVSIM_CACHE=$X/metric_cache_dev SPEC_CROP_SCALE=0.5"
E=". $B/scripts/common/env_navsim.sh; cd $B/code/risk-aware-perception; $C;"
for i in 0 1; do
  MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/detect_$i.log bash -c "$E NAVSIM_REDO=$V NAVSIM_SRC_DETS=$X/dets_dev2 NAVSIM_DETS=$X/dets_dev3 python $S/navsim_detect.py --shard $i/2" &
  sleep 200
done
wait
until grep -qs "NAVSIM DEV CHAIN DONE" $B/logs/A_navsim/dev/pdms_summary.log; do sleep 120; done   # CPU free again
. $B/scripts/common/env_navsim.sh; eval "$C"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NAVSIM_DETS=$X/dets_dev3 NAVSIM_PDMS_OUT=$X/pdms_dev3 \
  NAVSIM_TAG=_dev3 NAVSIM_BRANCHES=reference,full960,spec,spec_k025,$V
cd $B/code/risk-aware-perception
for i in $(seq 0 11); do python $S/navsim_pdms.py --stage run --shard $i/12 > $L/pdms_$i.log 2>&1 & done
wait
python $S/navsim_pdms.py --stage summary > $L/pdms_summary.log 2>&1
echo "$(date +%T) NAVSIM DEV3 CHAIN DONE" >> $L/pdms_summary.log
