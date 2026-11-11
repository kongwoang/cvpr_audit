#!/usr/bin/env bash
# K5 heavy-verifier operating point: wait for engines/build_v8l.sh, then the large-frame latency with YOLOv8l verifier
# (crops on ceil32 canvases via the dynamic engine, GPU candidate selection), nuScenes then AV2.
cd "$(dirname "$0")/.."
while ! grep -q "V8L ENGINES DONE" engines/build_v8l.out; do sleep 120; done
scripts/py scripts/57_large_latency_opt.py --dataset nusc --gpupost --dyncrops --verifier yolov8l > logs/k5_v8l_nusc.log 2>&1
scripts/py scripts/57_large_latency_opt.py --dataset av2 --gpupost --dyncrops --verifier yolov8l > logs/k5_v8l_av2.log 2>&1
scripts/py scripts/57_large_latency_opt.py --dataset nusc --gpupost --dyncrops > logs/k5_s_dyn_nusc.log 2>&1
scripts/py scripts/57_large_latency_opt.py --dataset av2 --gpupost --dyncrops > logs/k5_s_dyn_av2.log 2>&1
echo "K5 V8L CHAIN DONE" >> logs/k5_v8l_av2.log
