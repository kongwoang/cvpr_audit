#!/usr/bin/env bash
# K35-4 holdout runs through the isolated harness experiments/k35_holdout_20261025 (build.py). One run_when_gpu_free.sh
# group of three processes: k28_p3_h --part instr (v6-S, cheap, u960 / native + C1 + v3a, tsp8@S, unc2@S), k33_eval_h
# --part de --rows ana05hinfi --cap (AnA iterated), r3/runner_h (v6S, v7a, v7b).
# $1 = val (adapter check on seq05, reproduces the stored Argoverse-HD test records) | train (the K35-0 holdout, run ONCE)
# $2 = shard i/n of the holdout (train only; default 0/1)
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
H=$B/experiments/k35_holdout_20261025
L=$B/logs/k35/k4; mkdir -p $L
SP=$1; SH=${2:-0/1}
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B=$B B_REAL=$B K35_AHD_SPLIT=$SP"
case $SP in
  val)   U28="--shard 5/24"; U="--units seq05" ;;
  train) U28="--shard $SH"; U="--shard $SH" ;;
  *) echo "usage: $0 val|train [i/n]"; exit 1 ;;
esac
tg=${SH/\//of}
c="{ $T python $H/k28_p3_h.py --dataset ahd --split test --part instr $U28 > $L/${SP}_k28p3_$tg.log 2>&1 & p0=\$!; \
$T python $H/k33_eval_h.py --dataset ahd --split test --part de --rows ana05hinfi --cap $U > $L/${SP}_k33_$tg.log 2>&1 & p1=\$!; \
$T python $H/r3/runner_h.py --dataset ahd $U > $L/${SP}_r3low_$tg.log 2>&1 & p2=\$!; \
wait \$p0 && wait \$p1 && wait \$p2; }"
SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $L/${SP}_group_$tg.log bash -c "$c" > /dev/null 2>&1 < /dev/null &
echo "$(date +%T) K35-4 $SP $SH launched"
