#!/usr/bin/env bash
# Push local remote/<topic>/<file> to VUW scripts/<topic>/<file> (and CVPR/src/vtf to code/CVPR/src/vtf with --src).
# usage: remote/push.sh [--src] [<topic>/<file> ...]   (no files: the whole tree)
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
cd "$(dirname "$0")"
if [ "$1" = "--src" ]; then shift; rsync -a ../src/vtf/ VUW:$B/code/CVPR/src/vtf/; fi
if [ $# -eq 0 ]; then
  rsync -a --exclude push.sh --exclude README.md ./ VUW:$B/scripts/
else
  for f in "$@"; do rsync -a "$f" "VUW:$B/scripts/$f"; done
fi
