#!/usr/bin/env bash
# K33 G, C rows (routers bar1280_0.5 / dvr1280_0.5, Chanakya ch128B_csc; k33_rc.py): right after k33_j_window2.sh ends
# ("K33 G SELECTED ADA DONE"; the pause file is re-taken at once), a 30-frame smoke, then 3 repetitions on nusc_test20
# and av2 (ego mask). The pause file is removed at the end.
set -u
cd "$(dirname "$0")/.."
L=logs/k33j; mkdir -p $L/runs
log() { echo "$(date '+%F %T') $*" | tee -a $L/window.log; }
until grep -q "K33 G SELECTED ADA DONE" $L/window.log; do sleep 3; done
touch ~/.jetson_pause
sleep 5
until [ "$(~/research/risk-aware-perception/scripts/busy.sh)" = "paused" ] && [ ! -e ~/.r3_profiling ]; do sleep 60; done
sudo -n /usr/bin/jetson_clocks
isdone() { local ds cf tg a; local -a v=("$@")
      for ((a=0; a<${#v[@]}; a++)); do case ${v[a]} in --dataset) ds=${v[a+1]};; --config) cf=${v[a+1]};; --tag) tg=${v[a+1]};; esac; done
      ls results/raw/*_k33j_timing_${ds}_${cf}_${tg}/frames.pkl >/dev/null 2>&1; }
R() { if isdone "$@"; then log "skip (done) $*"; return 0; fi
      local lf=$L/runs/$(date +%m%d_%H%M%S)_$(echo "$*" | tr -c 'a-zA-Z0-9,+' '_' | cut -c1-100).log
      log "start $* -> $lf"; scripts/py scripts/69_k33_jetson.py "$@" > $lf 2>&1; local rc=$?
      log "end rc=$rc"; return $rc; }
ROWS=bar1280_0.5,dvr1280_0.5,ch128B_csc
log "K33 G C rows start"
if R --dataset nusc_test20 --config v6S --wf none --bounds "" --only cheap --paths trt32 --k33 pub --k33_rows $ROWS --max_frames 30 --warmup 5 --tag k33rtsmoke; then
  for rep in 1 2 3; do
    for D in nusc_test20 av2; do
      M=""; [ $D = av2 ] && M="--mask_y0 1830"
      R --dataset $D --config v6S $M --wf none --bounds "" --only cheap --paths trt32 --k33 pub --k33_rows $ROWS --tag k33selrt_r$rep
    done
  done
  log "K33 G C ROWS DONE"
else
  log "K33 G C rows smoke FAILED; no timing"
fi
rm -f ~/.jetson_pause
