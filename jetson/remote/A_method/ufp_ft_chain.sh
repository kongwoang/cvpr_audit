#!/usr/bin/env bash
# UFP with an MP-Det-style fine-tuned fine detector: build mosaics (CPU), fine-tune YOLOv8s (GPU), evaluate (GPU).
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/ufp_ft
mkdir -p $L
ok() { tail -3 $1 | grep -q "$2"; }
python scripts/A_method/ufp_ft.py --stage build > $L/build.log 2>&1; ok $L/build.log "UFP FT BUILD DONE" || { echo "UFP FT ABORT build" >> $L/build.log; exit 1; }
$B/run_when_gpu_free.sh $L/train.log python scripts/A_ext/d68_train.py mosaic ufp_ft 15 yolov8s 960 32 mosaic
[ -f out/ufp_ft/mosaic/weights/last.pt ] || { echo "UFP FT ABORT train" >> $L/train.log; exit 1; }
$B/run_when_gpu_free.sh $L/eval.log python scripts/A_method/ufp_ft.py --stage eval
ok $L/eval.log "UFP FT EVAL DONE" || { echo "UFP FT ABORT eval" >> $L/eval.log; exit 1; }
echo "UFP FT CHAIN DONE" >> $L/eval.log
