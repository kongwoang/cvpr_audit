#!/usr/bin/env bash
# K28 round 3 (method_freeze_draft §2l), DEVELOPMENT data only.
#  1. ours3 with YOLOv8m (v51S, B8m, B8mC1, B8mC1s768, B8mC1s640) and with YOLOv8l (v51S, B8lC1s768, B8lC1s640);
#  2. after the round-2 chain has written its official tables (r2_done.log "K28 R2 ANALYZE DONE"): k28_analyze_r3.py.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
N="$T B=$B python $B/scripts/A_method/k28_r3dev.py --dataset nusc --tag r3 --part ours3"
A="$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_r3dev.py --dataset av2 --tag r3 --part ours3"
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-9000} $B/run_when_gpu_free.sh $log bash -c "$*" & sleep 10; }
launch() {   # $1: log suffix; finished units are skipped (per-unit resume)
  for d in yolov8m yolov8l; do
    k=${d: -1}
    grp $L/r3_${k}_g0$1.log "$N --det $d --shard 0/3 & $N --det $d --shard 1/3 & $N --det $d --shard 2/3 & wait"
    grp $L/r3_${k}_g1$1.log "$A --det $d --shard 0/6 & $A --det $d --shard 1/6 & $A --det $d --shard 2/6 & wait"
    grp $L/r3_${k}_g2$1.log "$A --det $d --shard 3/6 & $A --det $d --shard 4/6 & $A --det $d --shard 5/6 & wait"
  done
  wait
}
complete() {
  for d in yolov8m yolov8l; do
    [ "$(ls $B/results/k28/nusc_r3ours-${d}_r3 2>/dev/null | wc -l)" -ge 152 ] || return 1
    [ "$(ls $B/results/k28/av2_r3ours-${d}_r3 2>/dev/null | wc -l)" -ge 300 ] || return 1
  done
}
launch ""
for i in 1 2; do          # a shard process that ended without an error (as in round-1 E) is re-run
  complete && break
  echo "$(date +%T) incomplete; re-run $i" >> $L/r3_chain.log
  launch "_rerun$i"
done
complete && echo "$(date +%T) K28 R3 DEV DONE" > $L/r3_done.log || { echo "$(date +%T) K28 R3 INCOMPLETE" > $L/r3_done.log; exit 1; }
until grep -q "K28 R2 ANALYZE DONE" $L/r2_done.log 2>/dev/null; do sleep 600; done
cd $B && $T python $B/scripts/A_method/k28_analyze_r3.py > $L/analyze_r3.log 2>&1
echo "$(date +%T) K28 R3 ANALYZE DONE" >> $L/r3_done.log
