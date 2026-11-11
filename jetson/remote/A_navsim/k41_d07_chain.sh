#!/usr/bin/env bash
# D07 on VUW1 (cuda13): wait for P02 (125 logs) -> field (CPU) -> detect (GPU, changed tokens only, behind the host GPU
# queue lock, MIN_FREE_MB=12000) -> score (CPU, 4 shards, CUDA hidden). After detect, relaunch the D03 del_all dispatcher
# if the D04 chain has finished and the dispatcher is not running (it takes the same lock without waiting).
set -u
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k41/D07; mkdir -p $L
[ "$(hostname -s)" = cuda13 ] || { echo "D07 belongs on cuda13 (VUW1)"; exit 1; }
E="source $B/scripts/common/env_navsim.sh >/dev/null 2>&1; cd $B/scripts/A_navsim; export B=$B OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SPEC_CROP_SCALE=0.5"
P="python $B/scripts/A_navsim/k41_d07_k2.py"
cd $B
until [ "$(ls $B/results/k41/p02/*.pkl 2>/dev/null | wc -l)" = 125 ]; do sleep 300; done
echo "$(date +%T) P02 complete; field"
bash -c "$E CUDA_VISIBLE_DEVICES=; $P --stage field" > $L/field.log 2>&1 || { echo "$(date +%T) STOP: field failed"; exit 1; }
n=$(python3 -c "import csv;r=list(csv.DictReader(open('$B/results/k41/d07/field_summary.csv')));print(sum(x.get('changed_k2')=='True' or x.get('changed_k2all')=='True' for x in r))")
echo "$(date +%T) changed tokens: $n"
if [ "$n" -gt 0 ]; then
  exec 9>>$B/logs/k41/cuda13_gpu_queue.lock
  echo "$(date +%T) waiting for the cuda13 GPU queue lock"; flock 9; echo "$(date +%T) lock acquired"
  MIN_FREE_MB=12000 ./run_when_gpu_free.sh $L/detect.log bash -c "$E; $P --stage detect"
  flock -u 9; exec 9>&-
  grep -q "K41 D07 DETECT DONE" $L/detect.log || { echo "$(date +%T) STOP: detect incomplete"; exit 1; }
  if grep -q "D03 del_all dispatcher relaunched" $B/logs/k41/D04/chain.log && ! pgrep -f "k41_d03_dispatch.py del_all" >/dev/null; then
    setsid bash -c "cd $B && source env.sh >/dev/null 2>&1; export B=$B; cd \$B/scripts/A_method && exec python k41_d03_dispatch.py del_all" \
      >> $B/logs/k41/D03/dispatch_del_all.log 2>&1 < /dev/null &
    echo "$(date +%T) D03 del_all dispatcher relaunched by D07"
  fi
fi
for i in 0 1 2 3; do bash -c "$E CUDA_VISIBLE_DEVICES=; $P --stage score --shard $i/4" > $L/score_$i.log 2>&1 & done
wait
echo "$(date +%T) score done: $(ls $B/results/k41/d07/score/*.pkl | wc -l) logs"
