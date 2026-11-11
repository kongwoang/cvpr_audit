#!/usr/bin/env bash
# D05 (K41-3 complete 81-point grid, RUNPLAN). Main mode (one per host): sequential wrapper jobs under the host GPU queue
# lock (after the host's D03 dispatcher). Order: own RUNPLAN half first (VUW 0-40, VUW1 41-80), then the other half;
# each (index, dataset) is claimed atomically (mkdir) so no two workers run the same job. VUW1's main chain first runs a
# 2-unit nuScenes smoke (grid index 13, tag smoke).
# 2026-11-02 (user: run everything, optimise VUW/VUW1): D05_WORKER=<name> starts an extra parallel worker that skips the
# host lock and the smoke, waits until one grid job has completed, and launches with SHARE=1 MIN_FREE_MB=14000 (only a
# GPU with >= 14 GB free, i.e. an idle one, not the other users' saturated ones).
# Success = every reference unit has <unit>.done.json with no failures and the registered detector md5 (the legacy
# wrapper's "exit" line is not trusted). env.sh is sourced by absolute path: bash `source env.sh` searches $PATH first and
# picked up /usr/pkg/bin/env.sh (no venv) in the first attempt.
# usage: [D05_WORKER=w1] setsid nohup bash scripts/A_method/k41_d05_chain.sh > logs/k41/D05/<log> 2>&1 < /dev/null &
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k41/D05; C=$L/claims; mkdir -p $C
h=$(hostname -s)
case $h in cuda12) first="$(seq 0 40)"; second="$(seq 41 80)";; cuda13) first="$(seq 41 80)"; second="$(seq 0 40)";; *) echo "unknown host $h"; exit 1;; esac
cd $B
E="cd $B && . $B/env.sh >/dev/null 2>&1; export B=$B; cd $B"
check() {  # check <result dir> <dataset> [max_units]
  bash -c "$E; python - '$1' '$2' '${3:-0}'" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
d, ds, mx = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
B = Path(os.environ["B"]); md5 = hashlib.md5((B / "scripts/A_method/k40r_detector.py").read_bytes()).hexdigest()
units = sorted(p.stem for p in (B / "results/k30/match" / ds).glob("*.pkl"))
units = units[:mx] if mx else units
bad = []
for u in units:
    p = d / (u + ".done.json")
    if not p.exists():
        bad.append((u, "missing")); continue
    r = json.loads(p.read_text())
    if r["failures"] or r["script_md5"] != md5:
        bad.append((u, "failures" if r["failures"] else "md5"))
print("check", d, len(units) - len(bad), "/", len(units), "ok", bad[:3])
sys.exit(1 if bad else 0)
PY
}
W=${D05_WORKER:-}; MF=12000; SH=0
if [ -n "$W" ]; then
  MF=14000; SH=1
  until grep -qx done $C/*/status 2>/dev/null; do sleep 120; done; echo "$(date +%T) worker $W: first grid job done, starting"
else
  exec 9>>$B/logs/k41/${h}_gpu_queue.lock
  echo "$(date +%T) waiting for the $h GPU queue lock"; flock 9; echo "$(date +%T) lock acquired"
fi
if [ -z "$W" ] && [ $h = cuda13 ] && [ ! -f $L/smoke.done ]; then
  t0=$(date +%s)
  MIN_FREE_MB=12000 ./run_when_gpu_free.sh $L/smoke.log bash -c "$E; python scripts/A_method/k40r_detector.py --dataset nusc --part grid --grid_index 13 --max_units 2 --tag smoke"
  check $B/results/k41/grid_13_smoke/nusc nusc 2 || { echo "$(date +%T) STOP: smoke failed"; exit 1; }
  echo "$(date +%T) smoke done in $(( $(date +%s) - t0 )) s (incl. GPU wait)"; touch $L/smoke.done
fi
for g in $first $second; do
  for ds in nusc av2 ahd; do
    mkdir $C/${g}_${ds} 2>/dev/null || continue
    echo "$h${W:+/$W} $(date +%s)" > $C/${g}_${ds}/owner
    if [ $ds = av2 ]; then cmd="env B_REAL=$B B=$B/k10_masked K10_MASK_Y0=1830 python scripts/A_method/k10_masked_run.py scripts/A_method/k40r_detector.py --dataset av2 --part grid --grid_index $g"
    else cmd="python scripts/A_method/k40r_detector.py --dataset $ds --part grid --grid_index $g"; fi
    t0=$(date +%s)
    SHARE=$SH MIN_FREE_MB=$MF ./run_when_gpu_free.sh $L/D05_${g}_${ds}.log bash -c "$E; $cmd"
    check $B/results/k41/grid_$(printf %02d $g)/$ds $ds >> $L/D05_${g}_${ds}.log 2>&1 \
      || { echo "$(date +%T) STOP: grid $g $ds failed"; echo failed > $C/${g}_${ds}/status; exit 1; }
    echo "$(( $(date +%s) - t0 ))" > $C/${g}_${ds}/seconds; echo done > $C/${g}_${ds}/status
    echo "$(date +%T) grid $g $ds done in $(( $(date +%s) - t0 )) s"
  done
done
echo "$(date +%T) D05 ${W:-main} host $h complete"
