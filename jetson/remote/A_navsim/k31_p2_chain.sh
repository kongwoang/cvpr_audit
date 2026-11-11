#!/usr/bin/env bash
# K31 item 6 (registered in handoff/K31/STATUS.md): k31_nav_p2.py. $1 = smoke | run | pairs.
#   smoke: 5 tokens of one log with real crops (GPU), output results/k31/p2_smoke
#   run:   stage run on navtest, 3 shards in one run_when_gpu_free.sh group (SHARE=1, MIN_FREE_MB 3500)
#   pairs: stage pairs (CPU only), 3 shards
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k31; mkdir -p $L
E="source $B/scripts/common/env_navsim.sh >/dev/null 2>&1; cd $B/code/risk-aware-perception; export B=$B OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 SPEC_CROP_SCALE=0.5"
P="python $B/scripts/A_navsim/k31_nav_p2.py"
case $1 in
  smoke)
    LOG1=$(ls $B/results/navsim_exp/dets_k30b | head -1 | sed 's/.pkl//')
    SHARE=1 MIN_FREE_MB=3500 setsid nohup $B/run_when_gpu_free.sh $L/p2_smoke.log bash -c \
      "$E K31_P2_TAG=_smoke_real; rm -rf $B/results/k31/p2_smoke_real; $P --stage run --only_log $LOG1 --max_tokens 5" > /dev/null 2>&1 < /dev/null & ;;
  run)
    SHARE=1 MIN_FREE_MB=3500 setsid nohup $B/run_when_gpu_free.sh $L/p2_run.log bash -c \
      "$E; $P --stage run --shard 0/3 & $P --stage run --shard 1/3 & $P --stage run --shard 2/3 & wait" > /dev/null 2>&1 < /dev/null & ;;
  pairs)
    for i in 0 1 2; do
      setsid nohup bash -c "$E CUDA_VISIBLE_DEVICES=; $P --stage pairs --shard $i/3; echo \"\$(date +%T) exit \$?\"" \
        > $L/p2_pairs_$i.log 2>&1 < /dev/null &
    done ;;
esac
sleep 2; echo "$(date +%T) K31 P2 $1 launched"
