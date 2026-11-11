#!/usr/bin/env bash
# K30 V8a navhard: detect1 (GPU, 2 shards) -> field (CPU, NF shards) -> detect2 (GPU, 2 shards) -> split -> two-stage EPDMS
# (the devkit's run_pdm_score.py, metric_cache_v2_navhard_two_stage) for reference / cheap / full960 / native / spec_pdm /
# hybrid_pdm with k30_nav_v2agent.PerceptionPDMAgent (+agent.dets_dirs=navhard_dets). Logs logs/k30/navhard_*.log.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k30
NF=${NF:-8}
E=". $B/scripts/common/env_nuplan.sh; export OPENSCENE_DATA_ROOT=$B/datasets/navsim NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
NAVSIM_EXP_ROOT=$B/results/navsim_exp NAVSIM_DEVKIT_ROOT=$B/third_party/navsim OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 B=$B SPEC_CROP_SCALE=0.5; \
export PYTHONPATH=$B/scripts/A_navsim:$B/third_party/navsim:\$PYTHONPATH; cd $B/code/risk-aware-perception;"
S=$B/scripts/A_navsim/k30_navhard.py
for i in 0 1; do
  SHARE=1 MIN_FREE_MB=4000 $B/run_when_gpu_free.sh $L/navhard_d1_$i.log bash -c "$E python $S --stage detect1 --shard $i/2" &
  sleep 200
done
wait
for i in $(seq 0 $((NF - 1))); do
  bash -c "$E CUDA_VISIBLE_DEVICES='' python $S --stage field --shard $i/$NF" > $L/navhard_field_$i.log 2>&1 &
done
wait
for i in 0 1; do
  SHARE=1 MIN_FREE_MB=4000 $B/run_when_gpu_free.sh $L/navhard_d2_$i.log bash -c "$E python $S --stage detect2 --shard $i/2" &
  sleep 200
done
wait
bash -c "$E python $S --stage split" > $L/navhard_split.log 2>&1
until grep -qs "navtest hybrid_pdm_fixed exit" $L/v2score_done.log; do sleep 300; done   # one 12-process scorer at a time
for br in reference cheap full960 native spec_pdm hybrid_pdm; do
  bash -c "$E CUDA_VISIBLE_DEVICES='' python \$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py train_test_split=navhard_two_stage \
    metric_cache_path=\$NAVSIM_EXP_ROOT/metric_cache_v2_navhard_two_stage agent=constant_velocity_agent \
    'agent._target_=k30_nav_v2agent.PerceptionPDMAgent' +agent.branch=$br '+agent.dets_dirs=navhard_dets' \
    worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=12 \
    experiment_name=k30_v2_navhard_$br output_dir=\$NAVSIM_EXP_ROOT/k30_v2/navhard_$br" > $L/v2score_navhard_$br.log 2>&1
  echo "$(date +%T) navhard $br exit $?" >> $L/v2score_done.log
done
echo "$(date +%T) K30 NAVHARD DONE" >> $L/v2score_done.log
