#!/usr/bin/env bash
# K5: wait for the u640 engines, then the full large-frame latency runs (AV2 10 val logs; nuScenes local DEEP scenes).
cd "$(dirname "$0")/.."
engines/build_u640.sh > engines/build_u640.out 2>&1
scripts/py scripts/55_large_latency.py --dataset av2 > logs/k5_av2.log 2>&1
scripts/py scripts/55_large_latency.py --dataset nusc > logs/k5_nusc.log 2>&1
echo "K5 CHAIN DONE" >> logs/k5_nusc.log
