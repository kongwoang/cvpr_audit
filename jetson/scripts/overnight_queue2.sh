#!/usr/bin/env bash
# Second unattended batch (queued 02:47, 2026-09-27): one job at a time, then re-collect all results.
cd "$(dirname "$0")/.."
run() { echo "$(date +%T) start $*"; ./scripts/py "$@"; echo "$(date +%T) done $*"; }
run scripts/30_ensemble_bootstrap.py > logs/ensemble_bootstrap.log 2>&1
run scripts/27_real_speculative.py --dataset KITTI --geometry mono --scale 0.7729 > logs/realspec_s960_KITTI_mono.log 2>&1
run scripts/27_real_speculative.py --dataset KITTI --geometry oracle > logs/realspec_KITTI_oracle.log 2>&1
run scripts/27_real_speculative.py --dataset nuScenes --geometry oracle > logs/realspec_nuScenes_oracle.log 2>&1
# heavier full mode for the milliseconds question: YOLOv8s at 288x960 and a 96x288 crop canvas
(cd engines && for S in "288 960" "96 288"; do set -- $S; H=$1; W=$2;
  $HOME/miniforge3/envs/edge/bin/python -c "
from ultralytics import YOLO; import shutil
p = YOLO('$HOME/models/yolo/yolov8s.pt').export(format='onnx', imgsz=($H, $W), opset=12, simplify=False, dynamic=False)
shutil.move(p, 'yolov8s_crop_${H}x${W}.onnx')" > export_${H}x${W}.log 2>&1
  /usr/src/tensorrt/bin/trtexec --onnx=yolov8s_crop_${H}x${W}.onnx --saveEngine=yolov8s_crop_${H}x${W}.engine --fp16 --workspace=2048 > build_${H}x${W}.log 2>&1
  /usr/src/tensorrt/bin/trtexec --loadEngine=yolov8s_crop_${H}x${W}.engine --iterations=400 --warmUp=500 --useSpinWait --noDataTransfers > time_${H}x${W}.log 2>&1
done)
./scripts/collect_results.sh
echo "$(date +%T) QUEUE2 DONE"
