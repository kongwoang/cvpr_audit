#!/usr/bin/env bash
# K33 spurious braking (table1_design.md "spurious braking detections per 1,000 frames (K30 V7)"): the first test runs did
# not store it, so the registered test rows are re-run once with k33_eval.py's spur_* counts (--tag spur). The rows,
# settings and code paths are those of the first run; k33_report.py requires every cost of the re-run to equal the first
# run's. $1 = de | pub: the three datasets one after another, 3 shards each, in one run_when_gpu_free.sh group.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/k33; mkdir -p $L
T="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 B_REAL=$B"
P=$1
case $P in
  de)  X="--rows ana05hinfo,ana05hinfi,cf40q20o,cf40q20i,ana25hinfo --cap" ;;
  pub) X="--fam ada,sahi,fov,ufp,rt,wf --rows ada+1t1280,sahi640o1s10,fovi1280b64a2,fovs1280b64a1,ufp20s512 --cap" ;;
  *) echo "usage: $0 de|pub"; exit 1 ;;
esac
cmd() {
  local ds=$1; shift
  if [ "$ds" = av2 ]; then
    echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k33_eval.py --dataset av2 --part $P $*"
  else
    echo "$T B=$B python $B/scripts/A_method/k33_eval.py --dataset $ds --part $P $*"
  fi
}
c=""
for ds in nusc av2 ahd; do
  a=(); for i in 0 1 2; do a+=("$(cmd $ds --split test $X --tag spur --shard $i/3)"); done
  c="$c${c:+ && }{ ${a[0]} & p0=\$!; ${a[1]} & p1=\$!; ${a[2]} & p2=\$!; wait \$p0 && wait \$p1 && wait \$p2; }"
done
SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $L/spur_$P.log bash -c "$c" > /dev/null 2>&1 < /dev/null &
echo "$(date +%T) K33 SPUR $P launched"
