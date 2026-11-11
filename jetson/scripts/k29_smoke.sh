#!/usr/bin/env bash
# K29 smoke (Jetson window A, after build_k29.sh; ~/.jetson_pause held): parity mode on 3 nuScenes val scenes and 2 AV2 dev
# logs with every fast-path variant, the v5.1-S + heuristics process, and a short timing run (nuScenes new test, 300 timed
# frames after a 100-frame warm-up pass). Dev data only, apart from the short timing run (latency, no selection on it).
set -u
cd "$(dirname "$0")/.."
log() { echo "$(date '+%F %T') $*"; }
R() { log "start $*"; scripts/py scripts/64_k29_latency.py "$@" 2>&1 | grep -v "Warning\|warn(" | tail -n 70; local rc=${PIPESTATUS[0]}; log "end rc=$rc"; }
R --dataset nusc_val --config v6S --parity --max_units 3 --wf u448,u960 --bounds 2 --paths ref,trt16,trt32,t32,t16g,trt16np,trt16ov --tag smoke
R --dataset av2_dev --config v6S --parity --max_units 2 --mask_y0 1830 --wf u512,u960 --bounds 2 --paths ref,trt16,t32,t16g,trt16np,trt16ov --tag smoke
R --dataset av2_dev --config v51S --parity --max_units 2 --mask_y0 1830 --wf none --bounds 2 --heur --paths ref,trt16 --tag smoke
R --dataset nusc_val --config v6S --max_frames 300 --warmup 100 --wf u960 --bounds 2 --paths ref,trt16,t16g,trt16ov --tag smoke
log "K29 SMOKE DONE"
