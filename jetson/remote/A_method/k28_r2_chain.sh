#!/usr/bin/env bash
# K28 round 2 (method_freeze_draft §2k), DEVELOPMENT data only.
#  1. ours (R2-1..4 + reference rows + family D seeds) and basewf (whole frame + C1), launched at once;
#  2. after family E has finished (r1_done.log and efix_done.log): k28_analyze.py --tag r1 -> the E dev-best configurations
#     per family / budget and the K24 forms -> baseheur + C1 (YOLOv8s), and baseheur with YOLO11m at the S budget;
#  3. k28_analyze_r2.py.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
N="$T B=$B python $B/scripts/A_method/k28_r2dev.py --dataset nusc --tag r2"
A="$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_r2dev.py --dataset av2 --tag r2"
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-7000} $B/run_when_gpu_free.sh $log bash -c "$*" & sleep 10; }
grp $L/r2_ours_g0.log "$N --part ours --shard 0/3 & $N --part ours --shard 1/3 & $N --part ours --shard 2/3 & wait"
grp $L/r2_ours_g1.log "$A --part ours --shard 0/6 & $A --part ours --shard 1/6 & $A --part ours --shard 2/6 & wait"
grp $L/r2_ours_g2.log "$A --part ours --shard 3/6 & $A --part ours --shard 4/6 & $A --part ours --shard 5/6 & wait"
grp $L/r2_wf_g0.log "$N --part basewf --shard 0/2 & $N --part basewf --shard 1/2 & $A --part basewf --shard 0/4 & wait"
grp $L/r2_wf_g1.log "$A --part basewf --shard 1/4 & $A --part basewf --shard 2/4 & $A --part basewf --shard 3/4 & wait"
# ---- heuristics + C1: after E
until grep -q "K28 R1 DEV DONE" $L/r1_done.log 2>/dev/null && grep -q "K28 E NUSC RERUN DONE" $L/efix_done.log 2>/dev/null; do sleep 600; done
cd $B && $T python $B/scripts/A_method/k28_analyze.py --tag r1 > $L/analyze_r1_final.log 2>&1
CFG=$($T python - <<'PY'
import pandas as pd
E = pd.read_csv("/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/results/k28/tables/baselines_r1.csv")
s = E[E.dev_best | E.K24_form][["config", "budget"]].drop_duplicates()
print(",".join(f"{c}@{b}" for c, b in s.itertuples(index=False)))
PY
)
CFG11=$($T python - <<'PY'
import pandas as pd
E = pd.read_csv("/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/results/k28/tables/baselines_r1.csv")
s = E[(E.dev_best | E.K24_form) & (E.budget == "S")][["config", "budget"]].drop_duplicates()
print(",".join(f"{c}@{b}" for c, b in s.itertuples(index=False)))
PY
)
echo "$(date +%T) configs: $CFG | YOLO11m: $CFG11" >> $L/r2_chain_configs.log
grp $L/r2_heur_g0.log "$N --part baseheur --configs $CFG --shard 0/2 & $N --part baseheur --configs $CFG --shard 1/2 & $A --part baseheur --configs $CFG --shard 0/4 & wait"
grp $L/r2_heur_g1.log "$A --part baseheur --configs $CFG --shard 1/4 & $A --part baseheur --configs $CFG --shard 2/4 & $A --part baseheur --configs $CFG --shard 3/4 & wait"
grp $L/r2_heur11_g0.log "$N --part baseheur --hdet yolo11m --configs $CFG11 --shard 0/2 & $N --part baseheur --hdet yolo11m --configs $CFG11 --shard 1/2 & $A --part baseheur --hdet yolo11m --configs $CFG11 --shard 0/4 & wait"
grp $L/r2_heur11_g1.log "$A --part baseheur --hdet yolo11m --configs $CFG11 --shard 1/4 & $A --part baseheur --hdet yolo11m --configs $CFG11 --shard 2/4 & $A --part baseheur --hdet yolo11m --configs $CFG11 --shard 3/4 & wait"
wait
echo "$(date +%T) K28 R2 DEV DONE" > $L/r2_done.log
cd $B && $T python $B/scripts/A_method/k28_analyze_r2.py > $L/analyze_r2.log 2>&1
echo "$(date +%T) K28 R2 ANALYZE DONE" >> $L/r2_done.log
