#!/usr/bin/env bash
# K33 G on the Jetson (exclusive window: ~/.jetson_pause held; MAXN, jetson_clocks, fan 255; scripts/69_k33_jetson.py,
# fast path trt32; K29 frame sets nusc_test20 and av2 with the ego mask). $1 = candidates | selected <rows> | ...
#   candidates: every D / E candidate (vtf.k33sel), one repetition, latency + action only (no J; the dev selection on VUW
#               never reads these files)
#   candidates_pub: A / B candidates within 1024 px (vtf.k33pub), one repetition, latency + action only
#   selected:   the dev-selected rows $2 (comma list) and their @cap rows, 3 repetitions
set -u
cd "$(dirname "$0")/.."
L=logs/k33j; mkdir -p $L/runs
log() { echo "$(date '+%F %T') $*" | tee -a $L/window.log; }
touch ~/.jetson_pause
# wait for: no DEEP pipeline step running (busy.sh prints only "paused") and no DEEP R3 latency profile (~/.r3_profiling)
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
case $1 in
  candidates)
    log "K33 G candidates start"
    for D in nusc_test20 av2; do
      M=""; [ $D = av2 ] && M="--mask_y0 1830"
      R --dataset $D --config v6S $M --wf none --bounds "" --only cheap --paths trt32 --k33 de --tag k33de_c1
    done
    log "K33 G CANDIDATES DONE" ;;
  candidates_pub)
    log "K33 G pub candidates start"
    for D in nusc_test20 av2; do
      M=""; [ $D = av2 ] && M="--mask_y0 1830"
      R --dataset $D --config v6S $M --wf none --bounds "" --only cheap --paths trt32 --k33 pub --tag k33pub_c1
    done
    log "K33 G PUB CANDIDATES DONE" ;;
  selected)
    for rep in 1 2 3; do
      for D in nusc_test20 av2; do
        M=""; [ $D = av2 ] && M="--mask_y0 1830"
        R --dataset $D --config v6S $M --wf none --bounds "" --only cheap --paths trt32 --k33 de,pub --k33_rows $2 --k33_cap --tag k33sel${3:-}_r$rep
      done
    done
    log "K33 G SELECTED DONE" ;;
esac
rm -f ~/.jetson_pause
