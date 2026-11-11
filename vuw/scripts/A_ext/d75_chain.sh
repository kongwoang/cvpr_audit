#!/usr/bin/env bash
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
python d75_build.py > logs/d75_build.log 2>&1 || exit 1
for arm in plain decision; do
  $B/run_when_gpu_free.sh $B/logs/d75_train_$arm.log python d68_train.py $arm d75 30
done
until grep -q "D68 TRAIN DONE" logs/d68_build.log; do sleep 300; done
$B/run_when_gpu_free.sh $B/logs/d75_eval.log python d68_eval.py --device cuda:0 --night --tag _night --models coco=models/yolov8s.pt \
  ad_plain=out/d75/plain/weights/best.pt ad_decision=out/d75/decision/weights/best.pt \
  d68_plain=out/d68/plain/weights/best.pt d68_rand=out/d68/rand/weights/best.pt d68_dsf=out/d68/dsf/weights/best.pt
echo "$(date +%T) D75 DONE" >> logs/d75_eval.log
