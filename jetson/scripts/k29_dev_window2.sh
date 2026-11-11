#!/usr/bin/env bash
# K29 Jetson window B, restart (2026-10-15 01:30) after k29_dev_window.sh was stopped by hand at 01:18 (parity process at 13.1 GB
# RSS, 0.5 GB available: the NuScenesDB alone holds ~9 GB). Changes: nuScenes frames from k29_frames.py pickles (no DB in the
# process), torch.cuda.empty_cache every 100 frames in parity mode, shared refiner engine / C1 head across fast paths, the t16g
# lifetime fix (c1fast b03f431a). Each run logs to logs/k29/runs/. Dev data only.
#  1. selection timing of the fixed t16g (with the reference again), same frames as the first selection runs
#  2. decision parity (R2) on the FULL dev splits, every candidate path
set -u
cd "$(dirname "$0")/.."
log() { echo "$(date '+%F %T') $*"; }
R() { local lf=logs/k29/runs/$(date +%m%d_%H%M%S)_$(echo "$*" | tr -c 'a-zA-Z0-9,' '_' | cut -c1-90).log
      log "start $* -> $lf"; scripts/py scripts/64_k29_latency.py "$@" > $lf 2>&1; local rc=$?
      grep -v "Warning\|warn(" $lf | tail -n 100; log "end rc=$rc"; }
C=ref,trt16,trt32,t32,t16g,trt16np,trt16ov
sudo -n /usr/bin/jetson_clocks; sudo -n /usr/bin/jetson_clocks --show | grep -E "GPU|EMC|cpu0"
R --dataset nusc_val --config v6S --max_units 20 --wf u960 --bounds 2 --paths ref,t16g --tag select2
R --dataset av2_dev --config v6S --max_units 10 --mask_y0 1830 --wf u960 --bounds 2 --paths ref,t16g --tag select2
for D in nusc_val av2_dev; do
  M=""; [ $D = av2_dev ] && M="--mask_y0 1830"
  R --dataset $D --config v6S --parity $M --wf u448,u512,u640 --bounds 2 --paths $C --tag a
  R --dataset $D --config v6S --parity $M --wf u960,u1280,native --ours 0 --paths $C --tag b
  R --dataset $D --config v51S --parity $M --wf none --bounds 2 --heur --paths $C --tag c
  R --dataset $D --config v6M --parity $M --wf none --bounds 2 --paths $C --tag d
  R --dataset $D --config v6L --parity $M --wf none --bounds 2 --paths $C --tag e
done
log "K29 DEV WINDOW DONE"
