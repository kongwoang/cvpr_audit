#!/usr/bin/env bash
# K28 family B Jetson timing (one window; DEEP informed). nuScenes: ours with the YOLO11m / YOLOv8m / YOLOv8l verifier
# on the crops vs whole frame with the same detector at the long sides that bracket it (static engines), and the
# YOLOv8s reference (v5.1-S). AV2: ours with YOLO11m and with YOLOv8s (no whole frame). The refiner engine is
# refiner_v3a_vuw (the B refiners have the same Res18x3 architecture, so the timing is the same; their decisions differ).
set -u
cd "$(dirname "$0")/.."
run() { echo "$(date +%T) start $*"; scripts/py scripts/60_k28_latency.py "$@" --max_units 25 --bounds "" 2>&1 | tail -n 14; }
run --dataset nusc --verifier yolo11m --wf_det yolo11m --wf u448,u640 --tag b_yolo11m
run --dataset nusc --verifier yolov8m --wf_det yolov8m --wf u448,u640 --tag b_yolov8m
run --dataset nusc --verifier yolov8l --wf_det yolov8l --wf u640 --tag b_yolov8l
run --dataset nusc --verifier yolov8s --wf_det yolov8s --wf u640,u960 --tag b_yolov8s
run --dataset av2 --mask_y0 1830 --verifier yolo11m --wf none --tag b_yolo11m
run --dataset av2 --mask_y0 1830 --verifier yolov8s --wf none --tag b_yolov8s
echo "$(date +%T) K28 B LATENCY DONE"
