#!/usr/bin/env bash
# Bench2Drive 220 routes, closed loop, PDM-Lite with perception in the loop (perception_pilot.py), branches
# privileged / cheap / u960 / u1600 / ours, each split into 4 route chunks (one CARLA server per chunk, resume on).
# Heavy GPU jobs on VUW (user-approved), sharing busy GPUs (SHARE=1) when memory allows.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/scripts/A_closedloop/carla_env.sh
O=$B/results/b2d/full; L=$B/logs/A_closedloop/full
mkdir -p $O/routes $L
python - <<PY
import xml.etree.ElementTree as ET, copy
t = ET.parse("$WORK_DIR/leaderboard/data/bench2drive220.xml"); r = t.getroot(); rs = list(r)
for k in range(4):
    n = copy.deepcopy(r)
    for x in list(n): n.remove(x)
    for x in rs[k::4]: n.append(copy.deepcopy(x))
    ET.ElementTree(n).write("$O/routes/chunk{}.xml".format(k))
print(len(rs), "routes -> 4 chunks")
PY
for br in ${BRANCHES:-privileged cheap ours u1600 u960}; do
  for k in 0 1 2 3; do
    SHARE=1 MIN_FREE_MB=9000 $B/run_when_gpu_free.sh $L/${br}_$k.log bash -c "export PERC_BRANCH=$br; \
      bash $B/scripts/A_closedloop/b2d_run.sh $O/routes/chunk$k.xml $B/scripts/A_closedloop/perception_pilot.py none \
      $O/$br/chunk$k.json $O/$br/save_chunk$k" &
    sleep 240
  done
done
wait
echo "$(date +%T) B2D CHAIN DONE" >> $L/chain.log
