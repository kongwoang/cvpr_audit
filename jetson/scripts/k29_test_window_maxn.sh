#!/usr/bin/env bash
# K29 Jetson window C (exclusive; ~/.jetson_pause held; jetson_clocks; fan 100%): re-timing of every latency-table pipeline,
# reference path (K28) and the selected fast path ($1) in the same processes, on the same frames, 3 repetitions:
#   nusc_test20  first 20 nuScenes new-test scene tokens (sorted), warm-up pass of 200 frames discarded first
#   nusc         K28's 20 old DEEP scenes (25 scenes, first 200 frames not reported)
#   av2          K28's 8 timed AV2 val logs (10 logs, first 200 frames not reported), ego mask
# Order (restart 2026-10-15 15:2x after two board resets in the v5.1-S + heuristics process; one process at a time):
#   1. per repetition and dataset: v6-S + whole frame u448 / u512 / u640 (+ ours <= 3 rounds, <= 2), whole frame u960 /
#      u1280 / native, v5.1-S alone (tag c0), v6-M, v6-L
#   2. one instrumented repetition of v6-S per dataset (per-component breakdown)
#   3. the dev-best S heuristics + C1, ONE heuristic per process (tags h1 = tsp2S+C1, h2 = tsp8S-yolo11m+C1; v5.1-S runs in
#      the same process for the per-frame budget). Logs are written line by line with sync, and tegrastats (200 ms) runs
#      alongside, so a reset leaves the last stage and the power trace on disk.
# Each run logs to logs/k29/runs/. Finished runs are skipped (restart after a reset).
set -u
SEL=${1:?usage: k29_test_window.sh <selected fast path>}
cd "$(dirname "$0")/.."
log() { echo "$(date '+%F %T') $*"; }
# a run is skipped when its result directory (same mode, dataset, config, tag) has frames.pkl
isdone() { local ds cf tg md=timing a; local -a v=("$@")
      for ((a=0; a<${#v[@]}; a++)); do case ${v[a]} in --dataset) ds=${v[a+1]};; --config) cf=${v[a+1]};; --tag) tg=${v[a+1]};;
        --breakdown) md=breakdown;; esac; done
      ls results/raw/*_k29_${md}_${ds}_${cf}_${tg}/frames.pkl >/dev/null 2>&1; }
R() { if isdone "$@"; then log "skip (done) $*"; return; fi
      local lf=logs/k29/runs/$(date +%m%d_%H%M%S)_$(echo "$*" | tr -c 'a-zA-Z0-9,' '_' | cut -c1-90).log
      log "start $* -> $lf"; scripts/py scripts/64_k29_latency.py "$@" > $lf 2>&1; local rc=$?
      grep -v "Warning\|warn(" $lf | tail -n 100; log "end rc=$rc"; }
# synced variant: every output line is appended and fsynced, so it survives a hard reset
RS() { if isdone "$@"; then log "skip (done) $*"; return; fi
      local lf=logs/k29/runs/$(date +%m%d_%H%M%S)_$(echo "$*" | tr -c 'a-zA-Z0-9,' '_' | cut -c1-90).log
      log "start (synced) $* -> $lf"; sync
      PYTHONUNBUFFERED=1 scripts/py scripts/64_k29_latency.py "$@" 2>&1 | while IFS= read -r l; do
        printf '%s %s\n' "$(date +%T.%N | cut -c1-12)" "$l" >> $lf; sync $lf; done
      local rc=${PIPESTATUS[0]}
      grep -v "Warning\|warn(" $lf | tail -n 100; log "end rc=$rc"; sync; }
sudo -n /usr/bin/jetson_clocks; sudo -n /usr/bin/jetson_clocks --show | grep -E "GPU|EMC|cpu0"
for rep in 1 2 3; do
  for D in nusc_test20 nusc av2; do
    M=""; [ $D = av2 ] && M="--mask_y0 1830"
    R --dataset $D --config v6S $M --wf u448,u512,u640 --bounds 2,3 --paths ref,$SEL --tag r${rep}a
    R --dataset $D --config v6S $M --wf u960,u1280,native --ours 0 --paths ref,$SEL --tag r${rep}b
    R --dataset $D --config v51S $M --wf none --bounds 2 --paths ref,$SEL --tag r${rep}c0
    R --dataset $D --config v6M $M --wf none --bounds 2 --paths ref,$SEL --tag r${rep}d
    R --dataset $D --config v6L $M --wf none --bounds 2 --paths ref,$SEL --tag r${rep}e
  done
  log "K29 TEST REP $rep DONE (without heuristics)"
done
for D in nusc_test20 nusc av2; do
  M=""; [ $D = av2 ] && M="--mask_y0 1830"
  R --dataset $D --config v6S $M --wf none --bounds "" --breakdown --paths ref,$SEL --tag bd
done
log "K29 MAIN DONE; heuristics next"
TL=logs/k29/tegrastats_heur_$(date +%m%d_%H%M%S).log
(tegrastats --interval 200 | while IFS= read -r l; do printf '%s %s\n' "$(date +%T.%N | cut -c1-12)" "$l" >> $TL; sync $TL; done) &
TPID=$!
for rep in 1 2 3; do
  for D in nusc_test20 nusc av2; do
    M=""; [ $D = av2 ] && M="--mask_y0 1830"
    RS --dataset $D --config v51S $M --wf none --bounds 2 --heur --heur_only tsp2S+C1 --paths ref,$SEL --tag r${rep}h1
    RS --dataset $D --config v51S $M --wf none --bounds 2 --heur --heur_only tsp8S-yolo11m+C1 --paths ref,$SEL --tag r${rep}h2
  done
  log "K29 HEUR REP $rep DONE"
done
for c in $(ps -eo pid=,ppid=,comm= | awk -v p=$TPID '$2==p && $3=="tegrastats" {print $1}'); do kill $c; done   # exact PID
log "K29 TEST WINDOW DONE"
