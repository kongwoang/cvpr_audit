#!/usr/bin/env bash
# Argoverse 1.1 tracking VAL 3D labels (per_sweep_annotations_amodal; missed by download_argoverse_hd.sh's pattern),
# stream-extracted from the 283 GB tar without storing it.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/scripts/common/env.sh
L=$B/logs/data/argoverse_hd.log
cd $B/datasets/argoverse1
if [ ! -f .labels_done ]; then
  curl -sS -L https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/tracking_val_v1.1.tar.gz | \
    tar -xz --wildcards --no-anchored "*/per_sweep_annotations_amodal/*" >> $L 2>&1 && touch .labels_done
fi
echo "$(date +%T) AV1.1 val labels: $(find . -path '*per_sweep_annotations_amodal*' -name '*.json' | wc -l) sweeps; AV1 LABELS DONE" >> $L
