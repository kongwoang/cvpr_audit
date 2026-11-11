#!/usr/bin/env bash
# K30 J items on the Jetson (exclusive window, ~/.jetson_pause held; MAXN, jetson_clocks, fan 255; scripts/67_k30_jetson.py,
# fast path trt32). Starts after K29 window C and its report (logs/k29/after_window.log). Unattended; stops at the first
# failed smoke. Finished runs are skipped (restart after a reset).
#   0. smoke: J1 heuristics (v6S budget), J2 --only, J3 Argoverse-HD, 40 frames each
#   1. J1  latency-G of the heuristics + C1 + refiner (tsp2, tsp8 = the hybrid, tsp8-yolo11m, cor0, cor8) at K28's S budget
#          (config v51S) and V1's x1.0 budget (config v6S), 3 repetitions, nusc_test20 and av2 (Table 3's frame sets)
#   2. J2  energy per frame: one pipeline per run (--only), nusc_test20 and av2, with the voltage / power logger running
#   3. J3  Argoverse-HD per-frame latency (cheap, u960 / u1280 / native + C1 + v3a, v6-S, v6-S <= 2), then the streaming AP
#          on VUW (remote/A_method/k30_j3_sap.py) and the J tables (68_k30_j_tables.py)
set -u
cd "$(dirname "$0")/.."
L=logs/k30j; mkdir -p $L/runs
log() { echo "$(date '+%F %T') $*" | tee -a $L/window.log; }
until grep -qs "K29 REPORT" logs/k29/after_window.log; do sleep 300; done
VL=$(ls -t logs/k29/vin_*_window.log 2>/dev/null | head -1)
if ! ps -eo args= | grep -q "[k]29_vin_logger"; then
  VL=logs/k29/vin_$(date +%m%d_%H%M%S)_window.log
  setsid nohup scripts/py scripts/k29_vin_logger.py $VL > /dev/null 2>&1 < /dev/null &
fi
log "J window start; vin log $VL"
sudo -n /usr/bin/jetson_clocks
curl -s -X POST -H 'content-type: application/json' -d '{"mode":"manual","pwm":255}' localhost:8765/api/jetson/fan > /dev/null
isdone() { local ds cf tg a; local -a v=("$@")
      for ((a=0; a<${#v[@]}; a++)); do case ${v[a]} in --dataset) ds=${v[a+1]};; --config) cf=${v[a+1]};; --tag) tg=${v[a+1]};; esac; done
      ls results/raw/*_k30j_timing_${ds}_${cf}_${tg}/frames.pkl >/dev/null 2>&1; }
R() { if isdone "$@"; then log "skip (done) $*"; return 0; fi
      local lf=$L/runs/$(date +%m%d_%H%M%S)_$(echo "$*" | tr -c 'a-zA-Z0-9,+' '_' | cut -c1-100).log
      log "start $* -> $lf"; scripts/py scripts/67_k30_jetson.py "$@" > $lf 2>&1; local rc=$?
      log "end rc=$rc"; return $rc; }
# ---- 0. smoke
R --dataset nusc_test20 --config v6S --wf none --bounds "" --heur --paths trt32 --max_frames 40 --warmup 10 --tag jsmoke1 &&
R --dataset nusc_test20 --config v6S --wf u960 --bounds 2 --paths trt32 --only ours_b2,u960+c1comp --max_frames 40 --warmup 10 --tag jsmoke2 &&
R --dataset ahd --config v6S --wf u960 --bounds 2 --paths trt32 --only cheap,u960+c1comp,ours,ours_b2 --max_units 1 --max_frames 40 --warmup 10 --tag jsmoke3 \
  || { log "J SMOKE FAILED; stopping"; exit 1; }
log "J SMOKE OK"
# ---- 1. J1
for rep in 1 2 3; do
  for D in nusc_test20 av2; do
    M=""; [ $D = av2 ] && M="--mask_y0 1830"
    R --dataset $D --config v51S $M --wf none --bounds "" --heur --paths trt32 --tag j1s_r$rep
    R --dataset $D --config v6S $M --wf none --bounds "" --heur --paths trt32 --tag j1x_r$rep
  done
done
log "J1 DONE"
# ---- 2. J2 (one pipeline per run; the hybrid = its run minus the run of its budget configuration alone)
for D in nusc_test20 av2; do
  M=""; [ $D = av2 ] && M="--mask_y0 1830"
  if [ $D = av2 ]; then SIDES="u448 u512 u640 u960 u1280 native"; else SIDES="u448 u512 u640 u960 u1280 native"; fi
  R --dataset $D --config v6S $M --wf none --ours 0 --bounds "" --only cheap --paths trt32 --tag j2_cheap
  for s in $SIDES; do
    R --dataset $D --config v6S $M --wf $s --ours 0 --bounds "" --only $s+c1comp --paths trt32 --tag j2_$s
  done
  R --dataset $D --config v6S $M --wf none --bounds "" --only ours --paths trt32 --tag j2_v6S_alone
  R --dataset $D --config v6S $M --wf none --bounds 2 --only ours_b2 --paths trt32 --tag j2_v6S_b2
  R --dataset $D --config v51S $M --wf none --bounds "" --only ours --paths trt32 --tag j2_v51S_alone
  R --dataset $D --config v51S $M --wf none --bounds "" --heur --heur_only tsp8S+C1 --paths trt32 --tag j2_hybS
  R --dataset $D --config v6S $M --wf none --bounds "" --heur --heur_only tsp8S+C1 --paths trt32 --tag j2_hybX
done
log "J2 DONE"
# ---- 3. J3
R --dataset ahd --config v6S --wf u960,u1280,native --bounds 2 --paths trt32 \
  --only cheap,u960+c1comp,u1280+c1comp,native+c1comp,ours,ours_b2 --tag j3_r1
log "J3 TIMING DONE"
scripts/py scripts/68_k30_j_tables.py $VL > $L/tables.log 2>&1; log "J TABLES exit $?"
if [ -f docs/paper/handoff/K30/J/j3_latency.csv ]; then
  scp -q docs/paper/handoff/K30/J/j3_latency.csv VUW:/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/results/k30/j3_latency.csv &&
  ssh VUW "source /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh; cd \$B; python scripts/A_method/k30_j3_sap.py results/k30/j3_latency.csv > logs/k30/j3_sap.log 2>&1; echo rc \$?" | tee -a $L/window.log
  scp -q VUW:/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/results/k30/tables/j3_sap.csv docs/paper/handoff/K30/J/ && log "J3 SAP copied"
fi
log "K30 J WINDOW DONE (Jetson still paused: a session releases it and tells DEEP)"
