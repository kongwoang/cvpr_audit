#!/usr/bin/env bash
# Direction B kill test: congruent fine-tuning (v8m old -> v8l new, both fine-tuned on nuScenes train at 640).
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
G=$B/run_when_gpu_free.sh
$G logs/b_train_old.log python d68_train.py plain d68 15 yolov8m 640 32 b_old
$G logs/b_train_new.log python d68_train.py plain d68 15 yolov8l 640 16 b_new_plain
for n in b_old b_new_plain; do for sp in train test; do
  $G logs/b_detect.log python b_congruent.py --stage detect --weights out/d68/$n/weights/best.pt --name $n --split $sp
done; done
python b_congruent.py --stage flips > logs/b_flips.log 2>&1
python b_congruent.py --stage build >> logs/b_flips.log 2>&1
$G logs/b_train_cong.log python d68_train.py congruent b_cong 15 yolov8l 640 16 b_new_cong
$G logs/b_detect.log python b_congruent.py --stage detect --weights out/b_cong/b_new_cong/weights/best.pt --name b_new_cong --split test
python b_congruent.py --stage eval > logs/b_eval.log 2>&1
echo "$(date +%T) B DONE" >> logs/b_eval.log
