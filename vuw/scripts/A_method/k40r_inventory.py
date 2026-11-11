#!/usr/bin/env python3
"""Read-only schema/provenance inventory; never opens excluded holdout records."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

import pyarrow.parquet as pq

B = Path(os.environ['B'])
SOURCE = B / 'tmp/desp_k40'
OUT = B / 'results/k40r/inventory'
ALLOW = ('nuscenes', 'av2', 'argoverse_hd', 'objects_nuscenes', 'objects_av2',
         'objects_argoverse_hd', 'det_frames_nuscenes', 'det_predictions_nuscenes',
         'tube_frames', 'nuscenes_dev_thresholds', 'av2_dev_thresholds',
         'jetson_latency', 'jetson_bounded', 'jetson_energy', 'jetson_baselines',
         'jetson_deadline', 'jetson_hybrid', 'navsim_navtest')


def md5(path):
    h = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = dict(host=platform.node(), python=sys.version, records={}, helpers={})
    for name in ALLOW:
        path = SOURCE / 'artifacts/records' / (name + '.parquet')
        assert 'holdout' not in str(path).lower()
        if not path.is_file():
            info['records'][name] = {'missing': True}
            continue
        p = pq.ParquetFile(path)
        info['records'][name] = dict(path=str(path), md5=md5(path),
            bytes=path.stat().st_size, rows=p.metadata.num_rows, columns=p.schema_arrow.names)
    for name in ('common', 'tab_loss', 'tab_main', 'tab_uniform', 'tab_episodes',
                 'tab_missed', 'tab_constrained', 'tab_coverage', 'tab_fidelity',
                 'tab_frontier', 'tab_curves', 'jetson', 'deadline'):
        path = SOURCE / 'analysis' / (name + '.py')
        dest = OUT.parent / 'vendor/analysis' / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            shutil.copyfile(path, dest)
            info['helpers'][name] = dict(source=str(path), md5=md5(path))
    (OUT.parent / 'vendor/analysis/__init__.py').touch()
    path = SOURCE / 'desp/consumers/braking.py'
    dest = OUT.parent / 'vendor/desp/consumers/braking.py'
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    for d in (dest.parent, dest.parent.parent):
        (d / '__init__.py').touch()
    info['helpers']['braking'] = dict(source=str(path), md5=md5(path))
    (OUT / 'inventory.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({k: {'rows': v.get('rows'), 'columns': len(v.get('columns', []))}
                      for k, v in info['records'].items()}, indent=2))


if __name__ == '__main__':
    main()
