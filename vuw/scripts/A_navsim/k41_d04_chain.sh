#!/usr/bin/env bash
# D04 -> P01 (new rows) -> D03 del_all, on VUW1 (cuda13), RUNPLAN order. Holds the host's GPU queue lock during D04 so
# that only one wrapper-managed GPU job runs on this host; MIN_FREE_MB=12000, no SHARE (RUNPLAN).
#   1. GPU smoke: 5 tokens of the first log; stop unless field pi parity holds on all of them
#   2. full: 3 shards in one run_when_gpu_free.sh call (one GPU)
#   3. P01 PDMS of all / selected / selected_support / geometric (CPU, 4 workers, CUDA hidden)
#   4. relaunch the D03 del_all dispatcher (its own lock / wrapper)
# usage (VUW1): setsid nohup bash scripts/A_navsim/k41_d04_chain.sh > logs/k41/D04/chain.log 2>&1 < /dev/null &
set -u
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k41/D04; mkdir -p $L
[ "$(hostname -s)" = cuda13 ] || { echo "D04 belongs on cuda13 (VUW1)"; exit 1; }
E="source $B/scripts/common/env_navsim.sh >/dev/null 2>&1; cd $B/scripts/A_navsim; export B=$B OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 SPEC_CROP_SCALE=0.5"
P="python $B/scripts/A_navsim/k41_pip_nav.py"
LOG1=$(ls $B/results/k31/p2 | head -1 | sed 's/.pkl//')
cd $B
exec 9>>$B/logs/k41/cuda13_gpu_queue.lock
echo "$(date +%T) waiting for the cuda13 GPU queue lock"; flock 9; echo "$(date +%T) lock acquired"
if [ ! -f $B/results/k41/pip_nav_gpusmoke/$LOG1.pkl ]; then
  MIN_FREE_MB=12000 ./run_when_gpu_free.sh $L/smoke.log bash -c "$E; $P --only_log $LOG1 --max_tokens 5 --out results/k41/pip_nav_gpusmoke"
fi
bash -c "$E; python - <<'EOF'
import pickle,sys,os
rows=pickle.load(open(os.environ['B']+'/results/k41/pip_nav_gpusmoke/$LOG1.pkl','rb'))
ok=[r for r in rows if 'err' not in r]
bad=[r['token'] for r in ok if not r['field_parity']['pi']]
print('GPU smoke:',len(rows),'tokens, field parity',{k:sum(r['field_parity'][k] for r in ok) for k in ('pi','px','rounds')})
sys.exit(1 if bad or len(ok)<5 else 0)
EOF" || { echo "$(date +%T) STOP: D04 GPU smoke parity failed"; exit 1; }
MIN_FREE_MB=12000 ./run_when_gpu_free.sh $L/full.log bash -c \
  "$E; $P --shard 0/3 & $P --shard 1/3 & $P --shard 2/3 & wait"
n=$(ls $B/results/k41/pip_nav/*.done.json 2>/dev/null | wc -l)
[ "$n" = 136 ] || { echo "$(date +%T) STOP: D04 full has $n/136 logs"; exit 1; }
flock -u 9; exec 9>&-
echo "$(date +%T) D04 complete; P01 new rows"
bash -c "$E CUDA_VISIBLE_DEVICES=; python k41_selector_metrics.py --rows all,selected,selected_support,geometric \
  --inputs results/k41/pip_nav --workers 4 --out results/k41/selector_metrics" > $B/logs/k41/P01/d04_rows.log 2>&1 &
p01=$!
setsid bash -c "cd $B && source env.sh >/dev/null 2>&1; export B=$B; cd \$B/scripts/A_method && exec python k41_d03_dispatch.py del_all" \
  > $B/logs/k41/D03/dispatch_del_all.log 2>&1 < /dev/null &
echo "$(date +%T) D03 del_all dispatcher relaunched"
wait $p01; echo "$(date +%T) P01 new rows exit $?"
