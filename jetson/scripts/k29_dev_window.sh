#!/usr/bin/env bash
# K29 Jetson window B (exclusive; ~/.jetson_pause held; jetson_clocks; fan 100%): dev data only.
#  1. selection timing (STATUS "Variant selection rule"): v6-S and u960 + T4 / + C1 + ref on nuScenes new val (first 20
#     scenes) and AV2 CFG-DEV (first 10 logs, ego mask), every candidate path, 200 warm-up frames each
#  2. decision parity (R2) on the FULL dev splits, every candidate path, every pipeline that uses C1 or a refiner:
#     whole frame u448 / u512 / u640 (+ ours <= 3 rounds and <= 2) and u960 / u1280 / native in two processes,
#     v5.1-S + the dev-best S heuristics + C1, v6-M, v6-L
# One process at a time.
set -u
cd "$(dirname "$0")/.."
log() { echo "$(date '+%F %T') $*"; }
R() { log "start $*"; scripts/py scripts/64_k29_latency.py "$@" 2>&1 | grep -v "Warning\|warn(" | tail -n 120; local rc=${PIPESTATUS[0]}; log "end rc=$rc"; }
C=ref,trt16,trt32,t32,t16g,trt16np,trt16ov
sudo -n /usr/bin/jetson_clocks; sudo -n /usr/bin/jetson_clocks --show | grep -E "GPU|EMC|cpu0"
R --dataset nusc_val --config v6S --max_units 20 --wf u960 --bounds 2 --paths $C --tag select
R --dataset av2_dev --config v6S --max_units 10 --mask_y0 1830 --wf u960 --bounds 2 --paths $C --tag select
for D in nusc_val av2_dev; do
  M=""; [ $D = av2_dev ] && M="--mask_y0 1830"
  R --dataset $D --config v6S --parity $M --wf u448,u512,u640 --bounds 2 --paths $C --tag a
  R --dataset $D --config v6S --parity $M --wf u960,u1280,native --ours 0 --paths $C --tag b
  R --dataset $D --config v51S --parity $M --wf none --bounds 2 --heur --paths $C --tag c
  R --dataset $D --config v6M --parity $M --wf none --bounds 2 --paths $C --tag d
  R --dataset $D --config v6L --parity $M --wf none --bounds 2 --paths $C --tag e
done
log "K29 DEV WINDOW DONE"
