#!/usr/bin/env bash
# K6 repeats 2 and 3: the same 5 branches on the 219 routes that ran in every branch (route 11755 removed from chunk0),
# one dispatcher per repeat, output results/b2d/rep<r>/<branch>/chunk*.json.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/scripts/A_closedloop/carla_env.sh
for r in 2 3; do
  O=$B/results/b2d/rep$r; mkdir -p $O/routes
  python - $B/results/b2d/full/routes $O/routes <<'PY'
import sys, xml.etree.ElementTree as ET, shutil
src, dst = sys.argv[1], sys.argv[2]
for k in range(4):
    t = ET.parse(f"{src}/chunk{k}.xml"); root = t.getroot()
    for rt in list(root):
        if rt.get("id") == "11755":
            root.remove(rt)
    t.write(f"{dst}/chunk{k}.xml")
    print(k, len(root))
PY
  B2D_OUT=$O B2D_LOG=$B/logs/A_closedloop/rep$r MAXJ=4 bash $B/scripts/A_closedloop/b2d_dispatch.sh
done
echo "B2D REPEATS DONE" >> $B/logs/A_closedloop/rep3/dispatch.log
