#!/usr/bin/env bash
# Smoke: the privileged PDM-Lite expert on the first Bench2Drive route (checks CARLA headless + leaderboard + agent).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/scripts/A_closedloop/carla_env.sh
O=$B/results/b2d/smoke; mkdir -p $O $B/logs/A_closedloop
python - <<PY
import xml.etree.ElementTree as ET
t = ET.parse("$WORK_DIR/leaderboard/data/bench2drive220.xml"); r = t.getroot()
for x in list(r)[1:]: r.remove(x)
t.write("$O/route1.xml")
print("route", list(r)[0].attrib)
PY
MIN_FREE_MB=8000 $B/run_when_gpu_free.sh $B/logs/A_closedloop/smoke_pdm_lite.log bash $B/scripts/A_closedloop/b2d_run.sh \
  $O/route1.xml $WORK_DIR/leaderboard/team_code/autopilot.py none $O/pdm_lite.json $O/save_pdm_lite
