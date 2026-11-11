#!/usr/bin/env bash
# Second seed of ESOD-YOLOv5s on nuScenes (official code; seed offset through esod_train_seed.py), then test / measure /
# decision eval tagged _s1. GPU via run_when_gpu_free.sh; YOLOv5's --device gets the picked physical GPU.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_method/seeds/esod_s1.log
mkdir -p $B/logs/A_method/seeds
E=". $B/scripts/A_method/esod_env.sh; cd $B/third_party/esod;"
R=$B/out/esod/nusc_yolov5s_s1
for try in 1 2 3 4; do
  grep -qs "^ *29/29" $R/results.txt && break
  [ $try -gt 1 ] && sleep 1200
  if [ -f $R/weights/last.pt ]; then
    MIN_FREE_MB=11000 $B/run_when_gpu_free.sh $L bash -c "$E sed -i \"s/^device: .*/device: '\$CUDA_VISIBLE_DEVICES'/\" $R/opt.yaml; python train.py --resume $R/weights/last.pt"
  else
    MIN_FREE_MB=11000 $B/run_when_gpu_free.sh $L bash -c "$E ESOD_SEED=1 python $B/scripts/A_method/esod_train_seed.py --data data/nuscenes.yaml \
      --cfg models/cfg/esod/nuscenes_yolov5s.yaml --weights weights/pretrained/yolov5s.pt --hyp data/hyps/hyp.nuscenes.yaml \
      --batch-size 8 --img-size 1536 --epochs 30 --device \$CUDA_VISIBLE_DEVICES --workers 8 --project $B/out/esod --name nusc_yolov5s_s1 --exist-ok"
  fi
done
grep -qs "^ *29/29" $R/results.txt || { echo "$(date +%T) SEED ESOD ABORT: epoch 29 not reached" >> $L; exit 1; }
rm -rf $B/out/esod/test_new_s1
MIN_FREE_MB=8000 $B/run_when_gpu_free.sh $L bash -c "$E python test.py --data data/nuscenes.yaml --weights $R/weights/best.pt \
  --batch-size 8 --img-size 1536 --device \$CUDA_VISIBLE_DEVICES --task test --save-txt --save-conf --project $B/out/esod --name test_new_s1 --exist-ok"
n=$(ls $B/out/esod/test_new_s1/labels 2>/dev/null | wc -l)
[ "$n" -gt 1000 ] || { echo "$(date +%T) SEED ESOD ABORT: only $n prediction files" >> $L; exit 1; }
. $B/scripts/common/env.sh
ESOD_TAG=_s1 python $B/scripts/A_method/esod_eval.py $B/out/esod/test_new_s1/labels >> $L 2>&1
echo "$(date +%T) SEED ESOD DONE" >> $L
