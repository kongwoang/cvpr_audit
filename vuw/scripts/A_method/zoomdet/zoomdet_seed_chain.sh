#!/usr/bin/env bash
# Second ZoomDet seed on nuScenes (same config / recipe as zoomdet_chain.sh; mmengine randomness.seed=1), infer on the
# new-test keyframes, score decisions tagged _s1 (esod_eval.py, EVAL_OUT=zoomdet).
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/zoomdet
S=scripts/A_method/zoomdet
W=$B/out/zoomdet/nusc_s1
ls $W/best_coco_bbox_mAP_epoch_*.pth > /dev/null 2>&1 || MIN_FREE_MB=11000 $B/run_when_gpu_free.sh $L/train_s1.log bash -c ". $B/$S/zoomdet_env.sh; cd $B/third_party/zoomdet_yolo; \
  python tools/train.py $B/$S/zoomdet_nusc.py --work-dir $W --cfg-options randomness.seed=1"
CK=$(ls $W/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | tail -1); [ -n "$CK" ] || CK=$W/epoch_15.pth
[ -f "$CK" ] || { echo "ZOOMDET S1 ABORT train (no checkpoint)" >> $L/train_s1.log; exit 1; }
echo "checkpoint $CK" >> $L/train_s1.log
MIN_FREE_MB=8000 $B/run_when_gpu_free.sh $L/infer_s1.log bash -c ". $B/$S/zoomdet_env.sh; cd $B/third_party/zoomdet_yolo; \
  python $B/$S/zoomdet_infer.py $B/$S/zoomdet_nusc.py $CK $B/out/zoomdet/test_new_s1/labels"
grep -q "ZOOMDET INFER DONE" $L/infer_s1.log || { echo "ZOOMDET S1 ABORT infer" >> $L/infer_s1.log; exit 1; }
EVAL_OUT=zoomdet METHOD=zoomdet ESOD_TAG=_s1 python scripts/A_method/esod_eval.py $B/out/zoomdet/test_new_s1/labels >> $L/eval_s1.log 2>&1
echo "ZOOMDET S1 CHAIN DONE" >> $L/eval_s1.log
