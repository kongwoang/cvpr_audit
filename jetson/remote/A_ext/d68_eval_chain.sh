#!/usr/bin/env bash
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
until grep -q "D68 TRAIN DONE" logs/d68_build.log; do sleep 300; done
$B/run_when_gpu_free.sh $B/logs/d68_eval.log python d68_eval.py --device cuda:0 --models coco=models/yolov8s.pt \
  plain=out/d68/plain/weights/best.pt rand=out/d68/rand/weights/best.pt dsf=out/d68/dsf/weights/best.pt
echo "$(date +%T) D68 EVAL DONE" >> logs/d68_eval.log
