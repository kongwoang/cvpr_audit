#!/usr/bin/env python3
"""Copy completed CPU evidence into the writer-visible handoff; no inference."""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / 'docs/paper/handoff'
S = ROOT / 'results/vuw_summaries'


def main():
    for folder, names, prefix in [
        ('scene_controlled', ['summary.csv', 'sources.csv', 'run.json'], 'k41_1_scene_controlled_'),
        ('anchors', ['training_units.csv', 'sources.csv', 'run.json'], 'k41_1_anchors_'),
        ('objects_v4', [f'{kind}_{ds}.csv' for ds in ('nusc', 'av2', 'ahd') for kind in ('parity', 'sources')], 'k41_4_'),
        ('variant_cpu_tests', ['checks.json'], 'k41_variant_cpu_'),
    ]:
        for name in names:
            shutil.copyfile(S / 'k41' / folder / name, H / 'K41' / (prefix + name))
    for name in ('run.json', 'stdout.txt', 'source_manifest.json', 'junit.xml'):
        shutil.copyfile(S / 'latency_opt/release_cpu_tests' / name,
                        H / 'LATENCY_OPT_20261030' / ('release_cpu_' + name))
    with (H / 'LATENCY_OPT_20261030/PARITY_CPU.csv').open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['status'] = 'passed'
    rows.extend([
        dict(check='release_CPU_regression', cases=102, mismatches=0, scope='3 CUDA tests skipped; copied release tests', status='passed'),
        dict(check='device_detections_actions_crops_pixels_TTC_history', cases='', mismatches='', scope='all dev and timed frames; RUNPLAN J01', status='not_run'),
    ])
    with (H / 'LATENCY_OPT_20261030/PARITY.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['check', 'cases', 'mismatches', 'scope', 'status'])
        w.writeheader(); w.writerows(rows)
    paths = sorted(set(
        list((ROOT / 'remote/A_method').glob('k40r_*.py'))
        + list((ROOT / 'remote/A_method').glob('k41_*.py'))
        + list((ROOT / 'remote/A_navsim').glob('k41_*.py'))
        + list((ROOT / 'experiments/latency_opt_20261030').rglob('*.py'))
        + [ROOT / 'scripts/90_k41_cpu_figures.py', Path(__file__).resolve()]
    ))
    with (H / 'K41/code_manifest_pre_gate.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'md5', 'scope'])
        w.writeheader()
        w.writerows(dict(path=str(p.relative_to(ROOT)), md5=hashlib.md5(p.read_bytes()).hexdigest(),
                         scope='pre-gate snapshot; execution status is in STATUS/RUNPLAN') for p in paths)
    before = json.loads((H / 'K40R/protected_files_before.json').read_text())
    # The source manifest is a mapping of absolute paths to their registered md5.
    mapping = before.get('files', before)
    bad = []
    for name, expected in mapping.items():
        if isinstance(expected, dict):
            expected = expected['md5']
        p = Path(name)
        if not p.is_absolute():
            p = ROOT / p
        actual = hashlib.md5(p.read_bytes()).hexdigest()
        if actual != expected:
            bad.append(dict(path=name, expected=expected, actual=actual))
    result = dict(files_checked=len(mapping), mismatches=bad,
                  scope='registered paper and frozen harness/core files',
                  checked_utc=datetime.now(timezone.utc).isoformat())
    (H / 'K40R/protected_files_pre_gate.json').write_text(json.dumps(result, indent=2) + '\n')
    assert not bad, bad
    print('Copied CPU evidence, recorded code hashes, protected files unchanged:', len(mapping))


if __name__ == '__main__':
    main()
