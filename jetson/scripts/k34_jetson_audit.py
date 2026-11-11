"""CPU-only post-run provenance audit and figure for K34-3."""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
O = R / 'docs/paper/handoff/K34'
L = R / 'logs/k34j'

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    state = json.loads((L / 'state.json').read_text())
    sources = json.loads((L / 'source_hashes.json').read_text())
    matched = {f: digest(R / f) == h for f, h in sources.items()}
    assert all(matched.values())
    assert state['status'] == 'complete' and not state['restore_errors'] and not state['pause_owned']
    frames = pd.read_csv(O / 'jetson_frames.csv')
    parity = pd.read_csv(O / 'jetson_parity.csv')
    repeat = pd.read_csv(O / 'jetson_repetition_parity.csv')
    assert len(frames) == 12519 and parity.missing_vuw_frames.sum() == 0
    assert repeat.frames_with_repetition_disagreement.sum() == 0
    assert (frames.crop_engine_executions == frames.detector_engine_executions - 1).all()
    assert (frames.loc[frames.pipeline != 'v6-S', 'crop_calls'] <= 2).all()
    engines = [R / 'engines' / n for n in ('yolov8s_crop_192x320.engine', 'yolov8s_crop_320x256.engine',
                                         'yolov8s_dyn.engine', 'refiner_v3a_vuw_dyn.engine')]
    engines += sorted((R / 'engines').glob('c1enc_fp32_b*.engine'))
    start = datetime.fromisoformat(state['started_local']).timestamp()
    hashes = {str(p.relative_to(R)): {'sha256': digest(p), 'size_bytes': p.stat().st_size,
              'mtime_before_window': p.stat().st_mtime < start} for p in engines}
    telemetry = (L / 'tegrastats.log').read_text()
    audit = {'audited_at': datetime.now().astimezone().isoformat(),
             'source_hash_matches': matched, 'timed_rows': len(frames),
             'missing_vuw_frames': int(parity.missing_vuw_frames.sum()),
             'repetition_action_disagreements': int(repeat.frames_with_repetition_disagreement.sum()),
             'window_released_local': state['window_released_local'], 'restore_errors': state['restore_errors'],
             'max_sampled_gpu_temperature_C': max(map(float, re.findall(r'GPU@([\d.]+)C', telemetry))),
             'engine_hashes_post_run': hashes,
             'engine_hash_limitation': 'Captured after execution; mtimes are supporting metadata, not proof of pre-run content.',
             'counter_correction': 'trt32 crop executions = total detector executions - draft; raw bounded-only counter is not used.',
             'parity_caveat': 'B2K-A differs on 3/805 nuScenes and 2/586 AV2 frames; causes not isolated.'}
    (O / 'jetson_audit.json').write_text(json.dumps(audit, indent=2) + '\n')

    ci = pd.read_csv(O / 'jetson_latency_intervals.csv')
    names = ['v6-S', 'v6-S <= 2', 'B2K-A']
    colors = ['#334e68', '#829ab1', '#d97706']
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    for ax, ds, title in zip(axes, ['nusc_test20', 'av2'], ['nuScenes: 20 scenes', 'AV2: 8 logs']):
        for i, (name, color) in enumerate(zip(names, colors)):
            q = ci[(ci.dataset == ds) & (ci.pipeline == name)].set_index('metric').loc[['mean_ms', 'p95_ms', 'p99_ms']]
            y = q.estimate.to_numpy()
            ax.errorbar(np.arange(3) + (i - 1) * .19, y, yerr=np.array([y - q.ci_lo, q.ci_hi - y]),
                        fmt='o', color=color, capsize=3, label=name)
        ax.set_xticks(range(3), ['Mean', 'p95', 'p99'])
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylim(0, 1.08 * ci.loc[ci.metric.isin(['mean_ms', 'p95_ms', 'p99_ms']), 'ci_hi'].max())
    axes[0].set_ylabel('Latency (ms)')
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('K34-3: Jetson trt32, three repetitions')
    fig.text(.5, .015, '95% paired scene/log bootstrap intervals; 2,000 draws, seed 0', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, .96])
    fig.savefig(O / 'jetson_latency.pdf')
    fig.savefig(O / 'jetson_latency.png', dpi=180)
    plt.close(fig)
    print('K34-3 audit passed; figure and provenance saved.')

if __name__ == '__main__':
    main()
