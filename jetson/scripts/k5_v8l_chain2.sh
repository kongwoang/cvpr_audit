#!/usr/bin/env bash
# K5 heavy-verifier rerun: v8l dynamic crop engine with max batch 2 (memory), nuScenes and AV2.
cd "$(dirname "$0")/.."
engines/build_v8l_dyn2.sh > engines/build_v8l_dyn2.out 2>&1
scripts/py scripts/57_large_latency_opt.py --dataset nusc --gpupost --dyncrops --verifier yolov8l > logs/k5_v8l_nusc.log 2>&1
scripts/py scripts/57_large_latency_opt.py --dataset av2 --gpupost --dyncrops --verifier yolov8l > logs/k5_v8l_av2.log 2>&1
echo "K5 V8L CHAIN2 DONE" >> logs/k5_v8l_av2.log
