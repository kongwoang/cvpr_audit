"""Complete the corrected-manifest handoff using stored records only.

Run with scripts/py, one CPU thread. No inference, configuration selection or
modification of frozen source. The initial FINAL_REPORT is retained.
"""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rap import planner as P

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / 'docs/paper/handoff'
OUT = H / 'OPTIMIZATION_20261023'
MAPPING = {
    'ours': ('v6S', 'v6-S'),
    'ours_retro': ('v6S_retro', 'v6-S + recovery'),
    'ours_b2A': ('v6S_b2A', 'B2K-A'),
    'ours_b2A_retro': ('v6S_b2A_retro', 'B2K-A + recovery'),
}


def stage_audit(vuw):
    """Localize discrepancies by stored action stages; do not infer causality."""
    differences = pd.read_csv(OUT / 'backend_disagreements.csv')
    details = []
    for source, experiment in [('legacy_k29', 'temporal_jetson_20261023'),
                               ('verified_cfg', 'verified_jetson_20261023')]:
        state = json.loads((ROOT / 'experiments' / experiment / 'logs/state.json').read_text())
        for run in state['runs']:
            if run['smoke']:
                continue
            ds = 'nusc' if run['dataset'] == 'nusc_val' else 'av2'
            jetson = pd.read_pickle(Path(run['run']) / 'frames.pkl').set_index(['unit', 'frame'])
            reference = vuw.loc[ds].reindex(jetson.index)
            subset = differences[(differences.source == source) &
                                 (differences.dataset == ds) & (differences.rep == run['rep'])]
            for row in subset.itertuples():
                jet, vu = next((j, u) for j, (u, label) in MAPPING.items() if label == row.pipeline)
                key = (row.unit, row.frame)
                a, b = jetson.loc[key], reference.loc[key]
                history = [k for k in jetson.index if k[0] == row.unit and k[1] <= row.frame][-3:]
                cheap_differences = sum(jetson.loc[k, 'a_cheap'] != reference.loc[k, 'a_cheap']
                                        for k in history)
                ja, va = int(a['a_' + jet + '@trt32_noref']), int(b['a_' + vu + '|none'])
                details.append(dict(
                    source=source, dataset=ds, rep=run['rep'], pipeline=row.pipeline,
                    unit=row.unit, frame=row.frame, warm=bool(a.warm),
                    jetson_action=row.jetson_action, vuw_action=row.vuw_action,
                    jetson_no_refiner_action=ja, vuw_no_refiner_action=va,
                    no_refiner_actions_differ=ja != va,
                    jetson_cheap_action=int(a.a_cheap), vuw_cheap_action=int(b.a_cheap),
                    cheap_differences_in_current_and_two_preceding_frames=cheap_differences,
                    jetson_retro_calls=int(a['retro_calls_' + jet + '@trt32']),
                    vuw_retro_calls=int(b['retro_calls_' + vu]),
                    jetson_pixels=float(a['px_' + jet + '@trt32']),
                    vuw_pixels=float(b['px_' + vu]), gt_action=int(a.act_gt),
                ))
    result = pd.DataFrame(details)
    assert len(result) == len(differences)
    result.to_csv(OUT / 'backend_stage_diagnostics.csv', index=False)
    return result


def losses_vs_cheap(frames):
    pp, cp = P.PlannerParams(), P.CostParams()
    commands = np.array([P.commanded_decel(i, pp) for i in range(3)])
    rows = []
    for ds, d in frames.reset_index().groupby('dataset'):
        units = sorted(d.unit.unique())
        draws = np.random.default_rng(0).integers(0, len(units), (2000, len(units)))
        weights = np.array([np.bincount(x, minlength=len(units)) for x in draws], float)
        cheap = cp.lam_risk * np.maximum(0, d.a_gt.to_numpy() - commands[d.a_cheap.to_numpy(int)]) ** 2
        for vu, label in MAPPING.values():
            candidate = cp.lam_risk * np.maximum(
                0, d.a_gt.to_numpy() - commands[d['a_' + vu + '|v3a'].to_numpy(int)]) ** 2
            u = pd.DataFrame(dict(unit=d.unit, cheap=cheap, candidate=candidate))
            agg = u.groupby('unit').agg(cheap=('cheap', 'sum'), candidate=('candidate', 'sum'),
                                      frames=('cheap', 'size')).reindex(units)
            delta = agg.candidate.to_numpy() - agg.cheap.to_numpy()
            boot = (weights @ delta) / (weights @ agg.frames.to_numpy())
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(dataset=ds, pipeline=label, comparator='cheap',
                             metric='insufficient_loss_per_frame',
                             candidate_loss=candidate.mean(), cheap_loss=cheap.mean(),
                             candidate_minus_cheap=delta.sum() / len(d),
                             ci_lo=lo, ci_hi=hi, bootstrap_draws=2000, seed=0))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'loss_vs_cheap.csv', index=False)
    return result


def plot_verified():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    folder = H / 'VERIFIED_JETSON_20261023'
    names = ['v6-S', 'v6-S + recovery', 'B2K-A', 'B2K-A + recovery']
    data = pd.read_csv(folder / 'summary.csv').set_index('pipeline').loc[names]
    intervals = pd.read_csv(folder / 'intervals.csv')
    colors = ['#46515f', '#1977b8', '#dc8b22', '#b83c42']
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), constrained_layout=True)
    for ax, metric in zip(axes, ['mean_ms', 'p95_ms', 'p99_ms']):
        cc = intervals[intervals.metric == metric].set_index('pipeline').loc[names]
        y = data[metric].to_numpy()
        err = np.maximum(0, np.vstack([y - cc.ci_lo.to_numpy(), cc.ci_hi.to_numpy() - y]))
        ax.bar(np.arange(4), y, color=colors, yerr=err, capsize=3, width=.65)
        ax.set(xticks=np.arange(4), xticklabels=[x.replace(' + ', '\n+ ') for x in names],
               ylabel='Milliseconds', title=metric.replace('_', ' '))
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('Jetson AV2: verified CFG-DEV prefix, 8 timed logs, 512 frames × 3 repetitions\n'
                 '95% paired log bootstrap; detector inputs include historical inference', fontsize=10)
    for suffix in ['pdf', 'png']:
        fig.savefig(OUT / ('verified_av2_latency.' + suffix), dpi=180)
    plt.close(fig)


def main():
    frames = pd.read_csv(H / 'W1_BOUNDED_20261023/dev/frames.csv').set_index(['dataset', 'unit', 'frame'])
    assert len(frames) == 29782 and not frames.index.duplicated().any()
    stage = stage_audit(frames)
    losses = losses_vs_cheap(frames)
    plot_verified()
    # Keep all older figures, with their mixed-role AV2 panels labelled in-image.
    import finalize
    finalize.np, finalize.pd = np, pd
    finalize.plot_results(pd.read_csv(H / 'W1_BOUNDED_20261023/dev/summary.csv'))
    # Preserve the remote summary convention without duplicating large raw records.
    mirror = ROOT / 'results/vuw_summaries/optimization_20261023'
    copied = []
    for name, subdir in [('W1_DEV_20261023', 'dev'), ('W1_BOUNDED_20261023', 'dev'),
                         ('NAVSIM_REUSE_20261023', ''), ('REFINER_GEOMETRY_20261023', '')]:
        source = H / name / subdir
        target = mirror / name.lower()
        target.mkdir(parents=True, exist_ok=True)
        for file in source.glob('*.csv'):
            if file.name in ['frames.csv', 'episodes.csv']:
                continue
            dest = target / file.name
            shutil.copyfile(file, dest)
            copied.append(dict(source=str(file.relative_to(ROOT)), path=str(dest.relative_to(ROOT)),
                               sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    (mirror / 'manifest.json').write_text(json.dumps(copied, indent=2) + '\n')
    state = json.loads((OUT / 'STATUS.json').read_text())
    state['updated'] = datetime.now(timezone.utc).timestamp()
    state['finalized_at_utc'] = datetime.now(timezone.utc).isoformat()
    state['status'] = 'complete'
    state['pid'] = None
    state['report'] = str(OUT / 'TONG_HOP.md')
    for exp in ['temporal_fast_20261023', 'verified_jetson_20261023']:
        entry = json.loads((ROOT / 'experiments' / exp / 'logs/state.json').read_text())
        assert entry['status'] == 'timing_complete' and entry['window_released']
        assert entry['source_engine_hashes_unchanged'] and not entry['restore_errors']
        state['local'][exp] = entry
    reports = ['W1_DEV_20261023', 'W1_BOUNDED_20261023', 'NAVSIM_REUSE_20261023',
               'RUNTIME_20261023', 'TEMPORAL_JETSON_20261023', 'REFINER_GEOMETRY_20261023',
               'FAST_RECOVERY_20261023', 'VERIFIED_JETSON_20261023']
    # W1 reports use the coordinator completion marker, the other six use DONE.
    for name in reports:
        if name.startswith('W1_'):
            marker = ROOT / 'experiments' / name.lower() / 'coordinator_state.json'
            assert json.loads(marker.read_text())['phase'] == 'complete'
        else:
            marker = H / name / 'DONE.json'
            assert json.loads(marker.read_text())['status'] == 'complete'
        state['completed_reports'][name] = True
    (OUT / 'STATUS.json').write_text(json.dumps(state, indent=2) + '\n')
    for path in [OUT / 'train_audit/DONE.json', OUT / 'dataset_audit/DONE.json', OUT / 'BACKEND_DONE.json']:
        assert json.loads(path.read_text())['status'] in ['PASS', 'complete']
    manifest = json.loads((ROOT / 'experiments/w1_bounded_20261023/fork_manifest.json').read_text())
    for path, digest in manifest.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    audit = dict(status='PASS', finalized_at_utc=state['finalized_at_utc'], quality_frames=len(frames),
                 completion_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 verified_timing_complete=True, train_development_overlap_rows=0,
                 main_source_unchanged=True, all_jetson_windows_released=True,
                 cross_backend_exact_parity=False, cross_backend_diagnostic_rows=len(stage),
                 remote_summary_files_mirrored=len(copied), no_new_inference=True)
    (OUT / 'completion_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (OUT / 'DONE.json').write_text(json.dumps(dict(status='complete', report=str(OUT / 'TONG_HOP.md'),
                                                     audit=str(OUT / 'completion_audit.json')), indent=2) + '\n')
    print(json.dumps(audit, indent=2))
    print(losses.to_string(index=False))


if __name__ == '__main__':
    main()
