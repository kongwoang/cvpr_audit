#!/usr/bin/env bash
# Fine-tuned detectors on AV2 (train on AV2 TRAIN logs only, score once on AV2 VAL): three branches in parallel
#   (a) YOLOv8s fine-tuned at 960 (u960-ft and our verifier-ft) -> av2_ft_eval --stage dets
#   (b) ZoomDet (official zoomdet_yolo, nuScenes recipe) -> inference on the AV2 val frames
#   (c) ESOD (official, SAM masks, nuScenes recipe: YOLOv5s, 1536, 30 epochs) -> test.py on the AV2 val frames
# then the prediction files are scored with the cheap action history of (a), and everything is summarised.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_method/sota_ext
G=$B/run_when_gpu_free.sh
S=$B/scripts/A_method
until grep -qs "AV2 FT PREPARE DONE" $L/av2_ft_prepare.log; do sleep 300; done
(   # (a)
  . $B/scripts/common/env.sh; cd $B
  ls $B/out/av2_ft/plain960/weights/best.pt > /dev/null 2>&1 || MIN_FREE_MB=14000 $G $L/av2_ft_yolo_train.log python scripts/A_ext/d68_train.py . av2_ft 15 yolov8s 960 32 plain960
  [ -f $B/out/av2_ft/plain960/weights/best.pt ] || { echo "$(date +%T) AV2 FT ABORT yolo train" >> $L/av2_ft_chain.log; exit 1; }
  MIN_FREE_MB=8000 $G $L/av2_ft_dets.log python $S/av2_ft_eval.py --stage dets --det plain960
  grep -q "DETS DONE" $L/av2_ft_dets.log || { echo "$(date +%T) AV2 FT ABORT dets" >> $L/av2_ft_chain.log; exit 1; }
) &
sleep 120
(   # (b)
  . $B/scripts/common/env.sh; cd $B
  python $S/zoomdet/make_config_av2.py > $L/av2_zoomdet_config.log 2>&1 || { echo "$(date +%T) AV2 FT ABORT zoomdet config" >> $L/av2_ft_chain.log; exit 1; }
  W=$B/out/zoomdet/av2
  ls $W/best_coco_bbox_mAP_epoch_*.pth > /dev/null 2>&1 || MIN_FREE_MB=11000 $G $L/av2_zoomdet_train.log bash -c ". $S/zoomdet/zoomdet_env.sh; cd $B/third_party/zoomdet_yolo; \
    python tools/train.py $S/zoomdet/zoomdet_av2.py --work-dir $W"
  CK=$(ls $W/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | tail -1); [ -n "$CK" ] || CK=$W/epoch_15.pth
  [ -f "$CK" ] || { echo "$(date +%T) AV2 FT ABORT zoomdet train" >> $L/av2_ft_chain.log; exit 1; }
  echo "checkpoint $CK" >> $L/av2_zoomdet_train.log
  MIN_FREE_MB=8000 $G $L/av2_zoomdet_infer.log bash -c ". $S/zoomdet/zoomdet_env.sh; cd $B/third_party/zoomdet_yolo; \
    ZD_LIST=$B/data/av2_ft/test_list.txt python $S/zoomdet/zoomdet_infer.py $S/zoomdet/zoomdet_av2.py $CK $B/out/zoomdet/av2_val/labels"
  grep -q "ZOOMDET INFER DONE" $L/av2_zoomdet_infer.log || { echo "$(date +%T) AV2 FT ABORT zoomdet infer" >> $L/av2_ft_chain.log; exit 1; }
) &
sleep 120
(   # (c)
  E=". $S/esod_env.sh; cd $B/third_party/esod;"
  grep -qs "ESOD AV2 MASKS DONE" $L/av2_esod_masks.log || MIN_FREE_MB=10000 $G $L/av2_esod_masks.log bash -c "$E python $S/esod_av2_masks.py"
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
  [ "$n" -gt 5000 ] || { echo "$(date +%T) AV2 FT ABORT esod test ($n files)" >> $L/av2_ft_chain.log; exit 1; }
) &
wait
. $B/scripts/common/env.sh; cd $B
grep -q "DETS DONE" $L/av2_ft_dets.log || exit 1
[ -d $B/out/zoomdet/av2_val/labels ] && python $S/av2_ft_eval.py --stage preds --method zoomdet --pred $B/out/zoomdet/av2_val/labels >> $L/av2_ft_score.log 2>&1
[ -d $B/out/esod/av2_val/labels ] && python $S/av2_ft_eval.py --stage preds --method esod --pred $B/out/esod/av2_val/labels >> $L/av2_ft_score.log 2>&1
python $S/av2_ft_eval.py --stage score >> $L/av2_ft_score.log 2>&1
echo "$(date +%T) AV2 FT CHAIN DONE" >> $L/av2_ft_chain.log
