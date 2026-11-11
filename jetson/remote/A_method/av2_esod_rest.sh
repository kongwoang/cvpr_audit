#!/usr/bin/env bash
# ESOD branch (c) of av2_ft_chain.sh after the SAM masks (relaunched separately): waits for the running mask job,
# then ESOD training (nuScenes recipe) and test.py on the AV2 val frames. av2_ft_finish.sh scores the predictions.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_method/sota_ext
G=$B/run_when_gpu_free.sh
S=$B/scripts/A_method
E=". $S/esod_env.sh; cd $B/third_party/esod;"
until grep -qs "exit" $L/av2_esod_masks.log; do sleep 300; done
grep -qs "ESOD AV2 MASKS DONE" $L/av2_esod_masks.log || { echo "$(date +%T) AV2 FT ABORT esod masks" >> $L/av2_ft_chain.log; exit 1; }
R=$B/out/esod/av2_yolov5s
for try in 1 2 3 4; do
  grep -qs "^ *29/29" $R/results.txt && break
  [ $try -gt 1 ] && sleep 1200
  if [ -f $R/weights/last.pt ]; then
    MIN_FREE_MB=11000 $G $L/av2_esod_train.log bash -c "$E sed -i \"s/^device: .*/device: '\$CUDA_VISIBLE_DEVICES'/\" $R/opt.yaml; python train.py --resume $R/weights/last.pt"
  else
    MIN_FREE_MB=11000 $G $L/av2_esod_train.log bash -c "$E python train.py --data data/av2.yaml \
      --cfg models/cfg/esod/nuscenes_yolov5s.yaml --weights weights/pretrained/yolov5s.pt --hyp data/hyps/hyp.nuscenes.yaml \
      --batch-size 8 --img-size 1536 --epochs 30 --device \$CUDA_VISIBLE_DEVICES --workers 8 --project $B/out/esod --name av2_yolov5s --exist-ok"
  fi
done
grep -qs "^ *29/29" $R/results.txt || { echo "$(date +%T) AV2 FT ABORT esod train" >> $L/av2_ft_chain.log; exit 1; }
rm -rf $B/out/esod/av2_val
MIN_FREE_MB=8000 $G $L/av2_esod_test.log bash -c "$E python test.py --data data/av2.yaml --weights $R/weights/best.pt \
  --batch-size 8 --img-size 1536 --device \$CUDA_VISIBLE_DEVICES --task test --save-txt --save-conf --project $B/out/esod --name av2_val --exist-ok"
n=$(ls $B/out/esod/av2_val/labels 2>/dev/null | wc -l)
[ "$n" -gt 5000 ] || echo "$(date +%T) AV2 FT ABORT esod test ($n files)" >> $L/av2_ft_chain.log
