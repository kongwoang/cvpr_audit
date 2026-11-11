#!/usr/bin/env bash
# Closed-loop benchmark setup (Bench2Drive, CARLA 0.9.15) inside the workspace only:
#   tools/carla_0.9.15 (simulator + additional maps), third_party/Bench2Drive, third_party/carla_garage (PDM-Lite expert),
#   envs/carla (python 3.8 via uv, carla client + leaderboard deps).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/scripts/common/env.sh
L=$B/logs/data/carla_setup.log
mkdir -p $B/logs/data $B/tools/carla_0.9.15
cd $B/tools/carla_0.9.15
for f in CARLA_0.9.15.tar.gz AdditionalMaps_0.9.15.tar.gz; do
  [ -f .$f.done ] && continue
  curl -sS -L -C - -o $f https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/$f >> $L 2>&1
  if [ $f = AdditionalMaps_0.9.15.tar.gz ]; then mv $f Import/ 2>/dev/null || { mkdir -p Import; mv $f Import/; }; else tar -xzf $f && rm -f $f; fi
  touch .$f.done; echo "$(date +%T) $f done" >> $L
done
[ -f .maps_imported ] || { bash ImportAssets.sh >> $L 2>&1 && touch .maps_imported; }
cd $B/third_party
[ -d Bench2Drive ] || git clone -q https://github.com/Thinklab-SJTU/Bench2Drive.git >> $L 2>&1
[ -d carla_garage ] || git clone -q https://github.com/autonomousvision/carla_garage.git >> $L 2>&1
echo "Bench2Drive $(git -C Bench2Drive rev-parse --short HEAD), carla_garage $(git -C carla_garage rev-parse --short HEAD)" >> $L
export UV_CACHE_DIR=$B/.cache/uv UV_PYTHON_INSTALL_DIR=$B/tools/python UV_PYTHON_BIN_DIR=$B/tools/python/bin UV_NO_MODIFY_PATH=1
if [ ! -x $B/envs/carla/bin/python ]; then
  $B/tools/uv/bin/uv venv --python 3.8 $B/envs/carla >> $L 2>&1
  W=$(ls $B/tools/carla_0.9.15/PythonAPI/carla/dist/*cp38*.whl 2>/dev/null | head -1)
  . $B/envs/carla/bin/activate
  if [ -n "$W" ]; then $B/tools/uv/bin/uv pip install "$W" >> $L 2>&1; else $B/tools/uv/bin/uv pip install carla==0.9.15 >> $L 2>&1; fi
  $B/tools/uv/bin/uv pip install "numpy<2" scipy shapely==2.0.* networkx py-trees==0.8.3 dictor six psutil requests ephem tabulate \
    opencv-python-headless pygame transforms3d simple_watchdog_timer xmlschema ujson >> $L 2>&1
  deactivate
fi
echo "$(date +%T) CARLA SETUP DONE" >> $L
