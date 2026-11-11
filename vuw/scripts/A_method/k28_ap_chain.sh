#!/usr/bin/env bash
# K28 PHASE 3 (reporting only): detection outputs of v6 and of whole-frame passes on Argoverse-HD val and nuScenes new test
# (k28_ap.py --part apdump), then whole-image AP / detection metrics against tensor px and GFLOPs (k28_ap_score.py).
# Starts only after the phase-2 pass has finished its analysis (test/done.log "K28 V6 ANALYZE DONE").
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
L=$B/logs/A_method/k28/test
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B B=$B"
X="$T python $B/scripts/A_method/k28_ap.py --part apdump"
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-9000} $B/run_when_gpu_free.sh $log bash -c "$*" & sleep 10; }
until grep -q "K28 V6 ANALYZE DONE" $L/done.log 2>/dev/null; do sleep 900; done
launch() {
  for ds in ahd nusc; do
    grp $L/ap_${ds}_g0$1.log "$X --dataset $ds --shard 0/6 & $X --dataset $ds --shard 1/6 & $X --dataset $ds --shard 2/6 & wait"
    grp $L/ap_${ds}_g1$1.log "$X --dataset $ds --shard 3/6 & $X --dataset $ds --shard 4/6 & $X --dataset $ds --shard 5/6 & wait"
  done
  wait
}
complete() {
  [ "$(ls $B/results/k28/ahd_test_v6apdump 2>/dev/null | wc -l)" -ge 24 ] && [ "$(ls $B/results/k28/nusc_test_v6apdump 2>/dev/null | wc -l)" -ge 228 ]
}
launch ""
for i in 1 2; do
  complete && break
  echo "$(date +%T) ap incomplete; re-run $i" >> $L/ap_chain.log
  launch "_rerun$i"
done
complete && echo "$(date +%T) K28 AP DUMP DONE" > $L/ap_done.log || { echo "$(date +%T) K28 AP DUMP INCOMPLETE" > $L/ap_done.log; exit 1; }
cd $B && $T python $B/scripts/A_method/k28_ap_score.py > $L/ap_score.log 2>&1
echo "$(date +%T) K28 AP SCORE DONE" >> $L/ap_done.log
