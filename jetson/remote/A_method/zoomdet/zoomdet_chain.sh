#!/usr/bin/env bash
# ZoomDet (official zoomdet_yolo) on nuScenes: prepare COCO json, write the config from theirs, train (their recipe, 15 ep),
# infer on the new-test keyframes, score decisions (esod_eval.py, same prediction format).
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/zoomdet
mkdir -p $L
S=scripts/A_method/zoomdet
python $S/zoomdet_prepare.py > $L/prepare.log 2>&1; grep -q "ZOOMDET PREPARE DONE" $L/prepare.log || { echo "ZOOMDET ABORT prepare" >> $L/prepare.log; exit 1; }
python $S/make_config.py >> $L/prepare.log 2>&1 || { echo "ZOOMDET ABORT config" >> $L/prepare.log; exit 1; }
W=$B/out/zoomdet/nusc
ls $W/best_coco_bbox_mAP_epoch_*.pth > /dev/null 2>&1 || MIN_FREE_MB=11000 $B/run_when_gpu_free.sh $L/train.log bash -c ". $B/$S/zoomdet_env.sh; cd $B/third_party/zoomdet_yolo; \
  python tools/train.py $B/$S/zoomdet_nusc.py --work-dir $W"
CK=$(ls $W/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | tail -1); [ -n "$CK" ] || CK=$W/epoch_15.pth
[ -f "$CK" ] || { echo "ZOOMDET ABORT train (no checkpoint)" >> $L/train.log; exit 1; }
echo "checkpoint $CK" >> $L/train.log
MIN_FREE_MB=8000 $B/run_when_gpu_free.sh $L/infer.log bash -c ". $B/$S/zoomdet_env.sh; cd $B/third_party/zoomdet_yolo; \
  python $B/$S/zoomdet_infer.py $B/$S/zoomdet_nusc.py $CK $B/out/zoomdet/test_new/labels"
grep -q "ZOOMDET INFER DONE" $L/infer.log || { echo "ZOOMDET ABORT infer" >> $L/infer.log; exit 1; }
EVAL_OUT=zoomdet METHOD=zoomdet python scripts/A_method/esod_eval.py $B/out/zoomdet/test_new/labels >> $L/eval.log 2>&1
echo "ZOOMDET CHAIN DONE" >> $L/eval.log
