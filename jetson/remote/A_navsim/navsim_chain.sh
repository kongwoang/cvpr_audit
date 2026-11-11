#!/usr/bin/env bash
# NAVSIM navtest with perception in the loop: wait for data + metric cache, smoke stage A, stage A (GPU, 2 shards),
# stage B (CPU, 12 shards), summary.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim
mkdir -p $L
until grep -qs "NAVSIM TEST DOWNLOAD DONE" $B/logs/data/navsim_download.log && grep -qs "NAVSIM METRIC CACHE DONE" $L/metric_cache.log; do sleep 300; done
S=$B/scripts/A_navsim
E=". $B/scripts/common/env_navsim.sh; cd $B/code/risk-aware-perception;"
MIN_FREE_MB=8000 $B/run_when_gpu_free.sh $L/detect_smoke.log bash -c "$E NAVSIM_DETS=$B/results/navsim_exp/dets_smoke python $S/navsim_detect.py --max_tokens 5"
grep -q "NAVSIM DETECT DONE" $L/detect_smoke.log || { echo "NAVSIM ABORT detect smoke" >> $L/detect_smoke.log; exit 1; }
for i in 0 1; do
  MIN_FREE_MB=8000 $B/run_when_gpu_free.sh $L/detect_$i.log bash -c "$E python $S/navsim_detect.py --shard $i/2" &
  sleep 200
done
wait
n=$(ls $B/results/navsim_exp/dets/*.pkl 2>/dev/null | wc -l)
[ "$n" -ge 136 ] || { echo "NAVSIM ABORT detect: $n logs" >> $L/detect_0.log; exit 1; }
. $B/scripts/common/env_navsim.sh; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $B/code/risk-aware-perception
for i in $(seq 0 11); do python $S/navsim_pdms.py --stage run --shard $i/12 > $L/pdms_$i.log 2>&1 & done
wait
python $S/navsim_pdms.py --stage summary > $L/pdms_summary.log 2>&1
echo "$(date +%T) NAVSIM CHAIN DONE" >> $L/pdms_summary.log
