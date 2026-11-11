#!/usr/bin/env bash
# Replaces branch (a) + the final scoring of av2_ft_chain.sh (relaunched with SHARE=1 for the YOLOv8s fine-tune, user-
# approved GPU sharing): YOLOv8s-960 fine-tune -> dets stage; then waits for the ESOD branch (test predictions or ABORT)
# and scores ZoomDet / ESOD prediction files and the summary.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_method/sota_ext
G=$B/run_when_gpu_free.sh
S=$B/scripts/A_method
. $B/scripts/common/env.sh; cd $B
ls $B/out/av2_ft/plain960/weights/best.pt > /dev/null 2>&1 || SHARE=1 MIN_FREE_MB=14000 $G $L/av2_ft_yolo_train.log python scripts/A_ext/d68_train.py . av2_ft 15 yolov8s 960 32 plain960
[ -f $B/out/av2_ft/plain960/weights/best.pt ] || { echo "$(date +%T) AV2 FT ABORT yolo train" >> $L/av2_ft_chain.log; exit 1; }
SHARE=1 MIN_FREE_MB=8000 $G $L/av2_ft_dets.log python $S/av2_ft_eval.py --stage dets --det plain960
grep -q "DETS DONE" $L/av2_ft_dets.log || { echo "$(date +%T) AV2 FT ABORT dets" >> $L/av2_ft_chain.log; exit 1; }
python $S/av2_ft_eval.py --stage preds --method zoomdet --pred $B/out/zoomdet/av2_val/labels >> $L/av2_ft_score.log 2>&1
python $S/av2_ft_eval.py --stage score >> $L/av2_ft_score.log 2>&1
echo "$(date +%T) AV2 FT PARTIAL SCORE DONE (without ESOD)" >> $L/av2_ft_chain.log
until grep -qs "exit" $L/av2_esod_test.log || grep -qs "ABORT esod" $L/av2_ft_chain.log; do sleep 600; done
if [ "$(ls $B/out/esod/av2_val/labels 2>/dev/null | wc -l)" -gt 5000 ]; then
  python $S/av2_ft_eval.py --stage preds --method esod --pred $B/out/esod/av2_val/labels >> $L/av2_ft_score.log 2>&1
  python $S/av2_ft_eval.py --stage score >> $L/av2_ft_score.log 2>&1
fi
echo "$(date +%T) AV2 FT CHAIN DONE" >> $L/av2_ft_chain.log
