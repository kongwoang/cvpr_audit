#!/usr/bin/env python3
"""Collect a completed JP03 window and run CPU-only analysis on VUW."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
B = '/vol/dl-nguyenb5-solar/users/hoangpc/CVPR'


def call(args, **kw):
    subprocess.run(args, check=True, cwd=ROOT, **kw)


def remote(command):
    call(['ssh', '-o', 'BatchMode=yes', 'VUW', command])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('window', type=Path)
    args = ap.parse_args(); p = args.window.resolve()
    assert p.parent == ROOT / 'experiments/latency_opt_20261030/logs'
    assert p.name.startswith('JP03_')
    state = json.loads((p / 'state.json').read_text())
    assert state['status'] == 'complete' and state['window_released'] and not state['restore_errors']
    assert len(state['runs']) == 18
    for manifest in ('source_hashes_before.json', 'engine_hashes_before.json'):
        for name, digest in json.loads((p / manifest).read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    names = []
    hashes = {}
    for r in state['runs']:
        folder = Path(r['run'])
        assert folder.parent == ROOT / 'results/raw'
        assert json.loads((folder/'DONE.json').read_text())['frames'] == r['frames']
        if r['stage'] == 'timing' or (r['stage'] == 'parity' and r['mode'] == 'audit'):
            for name in ('frames.pkl','DONE.json','config.json','environment.json','meta.json'):
                names.append(folder.name+'/'+name)
                hashes[str((folder/name).relative_to(ROOT))] = hashlib.sha256((folder/name).read_bytes()).hexdigest()
    filelist = p / 'cpu_transfer_files.txt'; filelist.write_text('\n'.join(names)+'\n')
    dest = B + '/results/latency_opt/jp03'
    remote('mkdir -p '+shlex.quote(dest+'/raw'))
    call(['rsync','-a','--files-from='+str(filelist),'results/raw/','VUW:'+dest+'/raw/'])
    call(['rsync','-a',str(p/'state.json'),'VUW:'+dest+'/'])
    common = 'cd '+shlex.quote(B)+' && source env.sh && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python '
    remote(common+'scripts/A_method/k41_jetson_pilot_summary.py --input results/latency_opt/jp03 --check-counters')
    remote(common+'scripts/A_method/k41_jetson_pilot_quality.py --input results/latency_opt/jp03')
    local = ROOT / 'results/vuw_summaries/latency_opt/jp03'
    local.mkdir(parents=True, exist_ok=True)
    call(['rsync','-a','VUW:'+dest+'/summary/',str(local)+'/'])
    (p/'collected_artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    call([str(ROOT/'scripts/py'), str(ROOT/'scripts/95_latency_c1_report.py'), str(local), str(p)])
    call(['python3', str(ROOT/'scripts/97_latency_pilot_metadata.py'), str(p)])
    print('JP03 transferred, CPU-analyzed and rendered; original source/engine hashes unchanged.')


if __name__ == '__main__': main()
