#!/usr/bin/env python3
"""Read-only metadata/telemetry summary after a completed Jetson pilot."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('window', type=Path)
    args = ap.parse_args(); p = args.window.resolve()
    sp = p / 'state.recovered.json'
    if not sp.exists(): sp = p / 'state.json'
    state = json.loads(sp.read_text())
    assert state['status'] in ('complete', 'recovered_complete') and state['window_released']
    pilot = p.name.split('_', 1)[0]
    out = ROOT / 'docs/paper/handoff/LATENCY_OPT_20261030'
    raw = (p/'tegrastats.log').read_text()
    telemetry = []
    for line in raw.splitlines():
        m = re.match(r'(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) ', line)
        if not m: continue
        cpu = re.search(r' CPU@([\d.]+)C', line)
        gpu = re.search(r' GPU@([\d.]+)C', line)
        if cpu and gpu:
            timestamp = dt.datetime.strptime(m[1], '%m-%d-%Y %H:%M:%S').astimezone()
            telemetry.append((timestamp, float(cpu[1]), float(gpu[1])))
    rows = []
    for r in state['runs']:
        if r['stage'] != 'timing': continue
        f = Path(r['run']); meta = json.loads((f/'meta.json').read_text())
        start = dt.datetime.strptime(f.name[:15], '%Y%m%d_%H%M%S').astimezone()
        end = dt.datetime.fromtimestamp((f/'DONE.json').stat().st_mtime).astimezone()
        values = [(c,g) for t,c,g in telemetry if start <= t <= end]
        assert values
        rows.append(dict(dataset=r['dataset'],mode=r['mode'],rep=r['rep'],run=str(f),
                         peak_torch_allocated_bytes=meta['peak_gpu_allocated_bytes'],
                         peak_torch_reserved_bytes=meta['peak_gpu_reserved_bytes'],
                         window_start=start.isoformat(),window_done_mtime=end.isoformat(),
                         telemetry_samples=len(values),cpu_max_C=max(x[0] for x in values),
                         gpu_max_C=max(x[1] for x in values),
                         scope='process peak torch allocator only, not total TRT/graph memory; temperatures cover setup/warmup/all rows'))
    with (out/f'{pilot}_RUN_METADATA.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    info = dict(pilot=pilot,state_status=state['status'],window_released=state['window_released'],
                restore_errors=state['restore_errors'],valid_telemetry_samples=len(telemetry),
                nul_bytes_retained_in_original=raw.count(chr(0)),
                max_cpu_C=max(x[1] for x in telemetry),max_gpu_C=max(x[2] for x in telemetry),
                memory='torch peak counters exclude external TensorRT/graph allocations; not total process GPU memory',
                energy='not aligned to per-row intervals; no energy-per-frame or saving claim',
                thermal_scope='whole window includes debug/setup/warmup/idle; not a per-row throttling certificate')
    (out/f'{pilot}_TELEMETRY_METADATA.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info))


if __name__ == '__main__': main()
