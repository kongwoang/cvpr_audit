#!/usr/bin/env bash
# Hourly status of the K31 / K32 / K33 runs on VUW and VUW1 (read-only). Flags: PROBLEM lines.
ssh VUW bash -s <<'EOF'
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
now=$(date +%s)
echo "VUW $(date '+%F %T')"
echo "== K31 / K33"
for f in $B/logs/k33/*.log $B/logs/k31/r34_curv_*.log $B/logs/k31/capture.log $B/logs/k31/b2k_*.log $B/logs/k31/b2v_*.log \
         $B/logs/k31/p2_run.log $B/logs/k31/p2_pairs_*.log; do
  [ -f $f ] || continue
  age=$(( (now - $(stat -c %Y $f)) / 60 ))
  last=$(tail -1 $f | cut -c1-70)
  echo "$(basename $f): age ${age}m | $last | tracebacks $(grep -c Traceback $f)"
  if ! grep -qE "^[0-9:]{8} exit [0-9]+" $f && [ $age -gt 90 ]; then echo "PROBLEM: $(basename $f) silent for ${age} min"; fi
  grep -qE "^[0-9:]{8} exit [1-9]" $f && echo "PROBLEM: $(basename $f) non-zero exit"
done
for d in $B/results/k33/*_de_dev $B/results/k33/*_pub_dev $B/results/k33/*_de_test $B/results/k33/*_pub_test; do [ -d $d ] && echo "$(basename $d): $(ls $d | wc -l) units"; done
echo "panels: $(ls $B/results/k31/panels 2>/dev/null | wc -l) files; p2 logs done: $(ls $B/results/k31/p2 2>/dev/null | wc -l)/136; p2_pairs logs: $(ls $B/results/k31/p2_pairs 2>/dev/null | wc -l)"
echo "av2_ft images: train $(ls $B/data/av2_ft/images/train 2>/dev/null | wc -l), val $(ls $B/data/av2_ft/images/val 2>/dev/null | wc -l)"
echo "== K32"
for D in $B/logs/k32/dispatch_*.log; do [ -f $D ] && echo "$(basename $D): $(grep -c launch $D) launches; last: $(tail -1 $D)"; done
for f in $(ls -t $B/logs/k32/test/*.log $B/logs/k32/shadow/*.log 2>/dev/null | head -14); do
  age=$(( (now - $(stat -c %Y $f)) / 60 ))
  n=$(grep -c 'Registering the route statistics' $f); tb=$(grep -c Traceback $f)
  st="running"; grep -qE "^[0-9:]{8} exit [0-9]+" $f && st="exited($(grep -oE '^[0-9:]{8} exit [0-9]+' $f | tail -1 | cut -c10-))"
  echo "$(basename $f): $n routes, tb $tb, age ${age}m, $st"
  if [ "$st" = running ] && grep -q "running on GPU" $f && [ $age -gt 45 ]; then echo "PROBLEM: $(basename $f) no output for ${age} min"; fi
done
# jobs whose evaluator exited with an incomplete checkpoint and that nothing relaunched (log ends with "exit", > 30 min ago)
for f in $B/logs/k32/test/*.log $B/logs/k32/shadow/*.log; do
  [ -f $f ] || continue
  tail -1 $f | grep -qE "^[0-9:]{8} exit [0-9]+" || continue
  age=$(( (now - $(stat -c %Y $f)) / 60 )); [ $age -gt 30 ] || continue
  n=${f#$B/logs/k32/}; j=$B/results/k32/${n%.log}.json
  python3 -c "import json,sys; p=json.load(open(sys.argv[1]))['_checkpoint']['progress']; sys.exit(0 if p and p[0]>=p[1] else 1)" $j 2>/dev/null || echo "PROBLEM: ${n%.log} exited ${age} min ago with an incomplete checkpoint (resume it on its claim owner host)"
done
echo "test checkpoints: $(ls $B/results/k32/test/*.json 2>/dev/null | wc -l) / 360 (incl. 48 unc); complete: $(for j in $B/results/k32/test/*.json; do python3 -c "import json,sys; p=json.load(open(sys.argv[1]))[\"_checkpoint\"][\"progress\"]; print(1 if p and p[0]>=p[1] else 0)" $j 2>/dev/null; done | grep -c 1); shadow: $(ls $B/results/k32/shadow/*.json 2>/dev/null | wc -l) / 16"
echo "VUW: load $(cut -d' ' -f1 /proc/loadavg), evaluators $(ps -eo args= | grep -c '[l]eaderboard_evaluator.*aeb_pilot'), k32 dispatchers $(ps -eo args= | grep -c '[k]32_dispatch.sh'), k31 procs $(ps -eo args= | grep -c '[k]31_\(eval\|nav_p2\).py'), waiting wrappers $(ps -eo args= | grep -c '[r]un_when_gpu_free.sh .*logs/k3')"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'; echo
EOF
ssh VUW1 bash -s <<'EOF'
echo "DEEP R2 (launched for the DEEP session on VUW1): $(tail -1 /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/deep_ext/code/logs/r2/r2.log 2>/dev/null | cut -c1-90); runner $(ps -eo args= | grep -c '[2]33_r2_vuw.sh')"
echo "VUW1: load $(cut -d' ' -f1 /proc/loadavg), evaluators $(ps -eo args= | grep -c '[l]eaderboard_evaluator.*aeb_pilot'), k32 dispatchers $(ps -eo args= | grep -c '[k]32_dispatch.sh'), k31/k33 procs $(ps -eo args= | grep -c '[k]3[13]_.*\.py\|[a]v2_ft_prepare.py'), waiting wrappers $(ps -eo args= | grep -c '[r]un_when_gpu_free.sh .*logs/k3')"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'; echo
EOF
echo "Jetson window: $(tail -1 $HOME/research/CVPR/logs/k33j/window.log | cut -c1-110); pause file $([ -e ~/.jetson_pause ] && echo held || echo free); windows running $(ps -eo args= | grep -c '[k]33_j_window')"
# only the runs after the current window's start line (failures handled earlier are logged above it)
grep -q "rc=[1-9]" <(awk '/K33 G .*start$/ {buf=""} {buf=buf $0 "\n"} END {printf "%s", buf}' $HOME/research/CVPR/logs/k33j/window.log) && echo "PROBLEM: Jetson run with non-zero rc (logs/k33j/window.log)"
