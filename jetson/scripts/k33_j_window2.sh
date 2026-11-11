#!/usr/bin/env bash
# K33 G, the selected A / B rows (handoff/K33/STATUS.md; DEEP informed, ~/.jetson_pause held): 3 repetitions of SAHI / FOVEA
# KDE_I / KDE_S / UFP on nusc_test20 and av2 (ego mask); then the two AdaScale engines (engines/build_k33.sh, last, one at a
# time); then ada+1t1280 x 3. The pause file is removed at the end.
set -u
cd "$(dirname "$0")/.."
L=logs/k33j; mkdir -p $L/runs
log() { echo "$(date '+%F %T') $*" | tee -a $L/window.log; }
touch ~/.jetson_pause
until [ "$(~/research/risk-aware-perception/scripts/busy.sh)" = "paused" ] && [ ! -e ~/.r3_profiling ]; do sleep 60; done
sudo -n /usr/bin/jetson_clocks
curl -s -X POST -H 'content-type: application/json' -d '{"mode":"manual","pwm":255}' localhost:8765/api/jetson/fan > /dev/null
isdone() { local ds cf tg a; local -a v=("$@")
      for ((a=0; a<${#v[@]}; a++)); do case ${v[a]} in --dataset) ds=${v[a+1]};; --config) cf=${v[a+1]};; --tag) tg=${v[a+1]};; esac; done
      ls results/raw/*_k33j_timing_${ds}_${cf}_${tg}/frames.pkl >/dev/null 2>&1; }
R() { if isdone "$@"; then log "skip (done) $*"; return 0; fi
      local lf=$L/runs/$(date +%m%d_%H%M%S)_$(echo "$*" | tr -c 'a-zA-Z0-9,+' '_' | cut -c1-100).log
      log "start $* -> $lf"; scripts/py scripts/69_k33_jetson.py "$@" > $lf 2>&1; local rc=$?
      log "end rc=$rc"; return $rc; }
ROWS=sahi640o1s10,fovi1280b64a2,fovs1280b64a1,ufp20s512
log "K33 G selected pub start"
for rep in 1 2 3; do
  for D in nusc_test20 av2; do
    M=""; [ $D = av2 ] && M="--mask_y0 1830"
    R --dataset $D --config v6S $M --wf none --bounds "" --only cheap --paths trt32 --k33 pub --k33_rows $ROWS --tag k33selpub_r$rep
  done
done
log "K33 G selected pub DONE; builds"
bash engines/build_k33.sh >> $L/window.log 2>&1; log "build_k33 rc=$?"
for rep in 1 2 3; do
  for D in nusc_test20 av2; do
    M=""; [ $D = av2 ] && M="--mask_y0 1830"
    R --dataset $D --config v6S $M --wf none --bounds "" --only cheap --paths trt32 --k33 pub --k33_rows ada+1t1280 --tag k33selada_r$rep
  done
done
log "K33 G SELECTED ADA DONE"
rm -f ~/.jetson_pause
