#!/usr/bin/env bash
# DEV round 3 (rerun after the spec_pdm dispatch fix): probe PDM-Closed itself (navsim_field.py, every 2nd dev token, 12 CPU shards) -> branch spec_pdm (v3's
# crop/guard/merge on the PDM-Closed field + support crops, scale 0.5) -> PDMS next to reference / cheap / full / u960 /
# spec (braking field) on the same tokens.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim/dev5
mkdir -p $L
S=$B/scripts/A_navsim
X=$B/results/navsim_exp
C="export NAVSIM_SPLIT=mini NAVSIM_CACHE=$X/metric_cache_dev SPEC_CROP_SCALE=0.5 NAVSIM_FIELD=$X/field_dev"
. $B/scripts/common/env_navsim.sh; eval "$C"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $B/code/risk-aware-perception
wait
E=". $B/scripts/common/env_navsim.sh; cd $B/code/risk-aware-perception; $C;"
for i in 0 1; do
  MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/detect_$i.log bash -c "$E NAVSIM_REDO=spec_pdm NAVSIM_SRC_DETS=$X/dets_dev4 NAVSIM_DETS=$X/dets_dev5 python $S/navsim_detect.py --shard $i/2" &
  sleep 200
done
wait
. $B/scripts/common/env_navsim.sh; eval "$C"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NAVSIM_DETS=$X/dets_dev5 NAVSIM_PDMS_OUT=$X/pdms_dev5 \
  NAVSIM_TAG=_dev5 NAVSIM_BRANCHES=reference,cheap,full,full960,spec,spec_pdm
cd $B/code/risk-aware-perception
for i in $(seq 0 11); do python $S/navsim_pdms.py --stage run --shard $i/12 > $L/pdms_$i.log 2>&1 & done
wait
python $S/navsim_pdms.py --stage summary > $L/pdms_summary.log 2>&1
echo "$(date +%T) NAVSIM DEV4 CHAIN DONE" >> $L/pdms_summary.log
