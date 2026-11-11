#!/usr/bin/env bash
# ESOD VisDrone gap (ours 33.1 vs paper 36.0 AP for v5m at 1536): NMS IoU 0.65 and the 1.25x input (1920).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/repro/esod_visdrone_checks.log
E=". $B/scripts/A_method/esod_env.sh; cd $B/third_party/esod;"
for cfg in "1536 0.65" "1920 0.6"; do
  set -- $cfg
  echo "=== v5m img $1 iou $2" >> $L
  MIN_FREE_MB=12000 $B/run_when_gpu_free.sh $L bash -c "$E python test.py --data data/visdrone.yaml \
    --weights $B/repro/esod_weights/esod/esod_yolov5m.pt --batch-size 4 --img-size $1 --iou-thres $2 --device \$CUDA_VISIBLE_DEVICES"
done
echo "ESOD CHECKS DONE" >> $L
