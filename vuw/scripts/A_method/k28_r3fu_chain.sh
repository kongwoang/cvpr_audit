#!/usr/bin/env bash
# K28 end of phase 1 (method_freeze_draft §2m): follow-ups of the round-3 adoption (R3-2, YOLOv8m + C1 with crops 768), DEVELOPMENT
# data only, not selection inputs:
#  1. own YOLOv8m refiners of seeds 1 / 2 (k28_train.py, round-1 recipe), then family D (k28_r3fu.py --part oursd);
#  2. at once: (R4) YOLOv8m whole frame at the B long sides (+ T4 / + C1) and YOLOv8m heuristics at the S budget (+ T4 / + C1);
#  3. k28_analyze_r3fu.py.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
N="$T B=$B python $B/scripts/A_method/k28_r3fu.py --dataset nusc --tag fu"
A="$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_r3fu.py --dataset av2 --tag fu"
CFG11=unc2@S,unc4@S,tcr0@S,tcr3@S,cor0@S,cor8@S,tsp2@S,tsp8@S,rnd@S      # the YOLO11m S configurations of §2k (r2_chain_configs.log)
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-9000} $B/run_when_gpu_free.sh $log bash -c "$*" & sleep 10; }
W="--part basewf3 --det yolov8m"
H="--part baseheur --hdet yolov8m --configs $CFG11"
D="--part oursd --det yolov8m"
seeds() {
  MIN=12000 grp $L/train_yolov8m_seeds.log "$T python $B/scripts/A_method/k28_train.py --det yolov8m --seed 1 & $T python $B/scripts/A_method/k28_train.py --det yolov8m --seed 2 & wait"
  wait
  ls $B/out/resnet18_v3a_yolov8m_s1/image_refiner.pt $B/out/resnet18_v3a_yolov8m_s2/image_refiner.pt > /dev/null || { echo "$(date +%T) K28 FU SEEDS FAILED" >> $L/fu_chain.log; exit 1; }
  echo "$(date +%T) seeds trained" >> $L/fu_chain.log
  dgroups ""
}
dgroups() {
  grp $L/fu_d_g0$1.log "$N $D --shard 0/3 & $N $D --shard 1/3 & $N $D --shard 2/3 & wait"
  grp $L/fu_d_g1$1.log "$A $D --shard 0/6 & $A $D --shard 1/6 & $A $D --shard 2/6 & wait"
  grp $L/fu_d_g2$1.log "$A $D --shard 3/6 & $A $D --shard 4/6 & $A $D --shard 5/6 & wait"
  wait
}
bgroups() {
  grp $L/fu_w_g0$1.log "$N $W --shard 0/3 & $N $W --shard 1/3 & $N $W --shard 2/3 & wait"
  grp $L/fu_w_g1$1.log "$A $W --shard 0/6 & $A $W --shard 1/6 & $A $W --shard 2/6 & wait"
  grp $L/fu_w_g2$1.log "$A $W --shard 3/6 & $A $W --shard 4/6 & $A $W --shard 5/6 & wait"
  grp $L/fu_h_g0$1.log "$N $H --shard 0/2 & $N $H --shard 1/2 & $A $H --shard 0/4 & wait"
  grp $L/fu_h_g1$1.log "$A $H --shard 1/4 & $A $H --shard 2/4 & $A $H --shard 3/4 & wait"
  wait
}
n_of() { ls $B/results/k28/$1 2>/dev/null | wc -l; }
complete() {
  for p in oursd basewf3 baseheur; do
    [ "$(n_of nusc_r3$p-yolov8m_fu)" -ge 152 ] && [ "$(n_of av2_r3$p-yolov8m_fu)" -ge 300 ] || return 1
  done
}
seeds &
bgroups "" &
wait
for i in 1 2; do          # a shard process that ended without an error is re-run (finished units are skipped)
  complete && break
  echo "$(date +%T) incomplete; re-run $i" >> $L/fu_chain.log
  dgroups "_rerun$i" & bgroups "_rerun$i" & wait
done
complete && echo "$(date +%T) K28 FU DEV DONE" > $L/fu_done.log || { echo "$(date +%T) K28 FU INCOMPLETE" > $L/fu_done.log; exit 1; }
cd $B && $T python $B/scripts/A_method/k28_analyze_r3fu.py > $L/analyze_fu.log 2>&1
echo "$(date +%T) K28 FU ANALYZE DONE" >> $L/fu_done.log
