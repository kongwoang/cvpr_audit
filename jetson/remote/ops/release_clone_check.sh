#!/usr/bin/env bash
# Release step E: clone the public repo into a new dir, fresh venv from its requirements.txt, weights from Release v1.0,
# then the full pytest (unit + subset regression) queued on the GPU. Logs: logs/release/clone_*.log
set -u
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
C=$B/tmp/desp_clone; V=$B/tmp/desp_clone_env; L=$B/logs/release
[ -e $C ] && { echo "$C exists"; exit 1; }
{
  git clone https://github.com/anonymous221206/DESP.git $C && cd $C && git log --format='%H %an <%ae> | %cn <%ce> %s'
  for d in argoverse argoverse_hd av2 kitti_tracking nuscenes; do cp -a $B/tmp/desp_release/data/$d data/; done
  /usr/local/bin/python3 -m venv $V && $V/bin/pip install -q -r requirements.txt && $V/bin/pip freeze > $L/clone_freeze.txt
  echo "pip rc=$?"
  bash scripts/download_weights.sh && (cd $C && md5sum weights/*.pt)
  echo "weights rc=$?"
} > $L/clone_setup.log 2>&1
c="cd $C && $V/bin/python -m pytest -q -s tests > $L/clone_pytest.log 2>&1; echo \"pytest rc=\$?\" >> $L/clone_pytest.log"
SHARE=1 MIN_FREE_MB=3500 setsid nohup $B/run_when_gpu_free.sh $L/clone_pytest_group.log bash -c "$c" > /dev/null 2>&1 < /dev/null &
echo "clone pytest queued" >> $L/clone_setup.log
