#!/usr/bin/env bash
# K28 family D for the round-1 L choice (verifier YOLO11m, own refiner): its refiner retrained with seeds 1 and 2
# (k28_train.py --det yolo11m --seed 1 / 2; seed 0 stays the confirmatory weight). Training only; nothing is selected here.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
L=$B/logs/A_method/k28
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
SHARE=1 MIN_FREE_MB=12000 $B/run_when_gpu_free.sh $L/train_yolo11m_seeds.log bash -c "$T python $B/scripts/A_method/k28_train.py --det yolo11m --seed 1 & $T python $B/scripts/A_method/k28_train.py --det yolo11m --seed 2 & wait"
echo "$(date +%T) K28 D 11M SEEDS DONE" > $L/dseeds_done.log
