#!/usr/bin/env bash
# K33 F evaluation (k33_eval.py part ft; AV2 val with the ego mask, through k10_masked_run.py): $1 = smoke | test
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/k33; mkdir -p $L
C="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 B_REAL=$B B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k33_eval.py --dataset av2 --part ft --split test"
case $1 in
  smoke) SHARE=1 MIN_FREE_MB=6000 setsid nohup $B/run_when_gpu_free.sh $L/ft_smoke.log bash -c "$C --max_units 1 --tag smoke" > /dev/null 2>&1 < /dev/null & ;;
  test)  SHARE=1 MIN_FREE_MB=6000 setsid nohup $B/run_when_gpu_free.sh $L/ft_test.log bash -c "$C --shard 0/3 & $C --shard 1/3 & $C --shard 2/3 & wait" > /dev/null 2>&1 < /dev/null & ;;
esac
sleep 2; echo "$(date +%T) K33 FT $1 launched"
