#!/usr/bin/env python3
"""Render JP03 measured results after the window and CPU analysis complete."""
import argparse
import csv
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/paper/handoff/LATENCY_OPT_20261030'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('summary', type=Path)
    ap.add_argument('window', type=Path)
    args = ap.parse_args()
    state = json.loads((args.window / 'state.json').read_text())
    assert state['status'] == 'complete' and state['window_released'] and not state['restore_errors']
    assert len(state['runs']) == 18
    for f in args.summary.iterdir():
        if f.suffix in ('.csv', '.json'):
            shutil.copyfile(f, OUT / ('JP03_' + f.name))
    shutil.copyfile(args.window / 'state.json', OUT / 'JP03_window_state.json')
    for ds in ('nusc_val', 'av2_dev'):
        p = args.window / f'parity_{ds}.json'
        check = json.loads(p.read_text())
        assert check['mismatches'] == 0
        shutil.copyfile(p, OUT / f'JP03_parity_{ds}.json')
    rows = list(csv.DictReader((args.summary / 'BENCHMARK_PILOT.csv').open()))
    decision = json.loads((args.summary / 'feasibility.json').read_text())
    text = ['# JP03 — C1 graph full-pipeline feasibility', '',
            'This measured development pilot follows the pre-registered JP02 gate. The isolated candidate adds a CUDA graph to each existing independent fixed-shape FP32 C1 context. Detector graphs, contexts, buffers, weights, precision, thresholds, column processing, crops, histories, refiner and frozen v6-S are unchanged. Graphs are captured before frame processing; cold initialization is excluded. Candidate applies equally to all rows. JP01 columns-only optimization is disabled in both arms.', '',
            'Jetson AGX Xavier, MAXN, locked clocks, fan255, exclusive window. Each dataset uses its first712 development frames in original order,200 warmup and512 timed,3 fresh alternating baseline/candidate repetitions. Same dev prefixes as JP01; no test tuning. All timing runs exclude debug hooks. Paired2,000 scene/log bootstrap resamples, seed0, keep all repetitions together within a unit. These are exploratory intervals; no multiplicity-adjusted superiority claim.', '',
            '| Dataset / row | Mean baseline→graph (ms) | p95 baseline→graph (ms) | Δp95 95% CI (ms) | p99 baseline→graph (ms) |',
            '|---|---|---|---|---|']
    names = {'nusc_val': 'nuScenes dev', 'av2_dev': 'AV2 CFG-DEV'}
    selected = [('ours@trt32', 'DESP'), ('u512+c1comp@trt32', 'u512'), ('u640+c1comp@trt32', 'u640'), ('u960+c1comp@trt32', 'u960'), ('ours_b2A@trt32', 'B2K-A'), ('ours_b1A@trt32', 'B1K-A')]
    for ds in names:
        for row, label in selected:
            q = {x['metric']: x for x in rows if x['dataset'] == ds and x['row'] == row}
            pair = lambda m: f"{float(q[m]['baseline_ms']):.3f}→{float(q[m]['candidate_ms']):.3f}"
            p = q['p95']
            text.append(f"| {names[ds]} / {label} | {pair('mean')} | {pair('p95')} | [{float(p['delta_lo_ms']):+.3f}, {float(p['delta_hi_ms']):+.3f}] | {pair('p99')} |")
    text += ['', 'Pre-registered screen:≥5% DESP p95 reduction, paired interval entirely below0 for candidate−baseline, exact parity, no significant mean/p95 slowdown in included whole-frame rows.']
    for d in decision['decisions']:
        text.append(f"- {names[d['dataset']]}: p95 reduction {d['p95_reduction_percent']:.3f}%; screen {'PASS' if d['promising'] else 'NOT MET'}; whole-frame slowdown flag={d['wholeframe_significant_slowdown']}.")
    text += ['', 'Descriptive timing control (cheap row has no C1 encoder; this does not change the pre-registered decision rule):']
    for ds in names:
        q={x['metric']:x for x in rows if x['dataset']==ds and x['row']=='cheap'}
        m=q['mean']
        text.append(f"- {names[ds]} cheap mean {float(m['baseline_ms']):.3f}→{float(m['candidate_ms']):.3f}ms; paired Δ95% CI [{float(m['delta_lo_ms']):+.3f},{float(m['delta_hi_ms']):+.3f}]ms. A change here cannot be a direct C1 execution saving; inspect the per-repetition table for timing variation.")
    text += ['',
             'Debug baseline/audit comparison checks47 saved endpoints/frame on712 frames per dataset: selected detector columns/NMS outputs, crop requests/canvas metadata, lifted geometry/TTC, refined merged scenes, own history, actions, costs, crops/pixels/rounds. Every actual C1 encoder call in audit mode also asserts bit-exact graph versus direct output. Clean timing runs independently check scalar outputs and saved execution/refinement counters. This certifies only the measured prefixes; full-dev/test parity remains outstanding.', '',
             'Exact encoder graph/direct checks by dataset: '+json.dumps(decision['encoder_graph_exact_checks_by_dataset'])+'. The comparator JSON has a generic warning about columns_mode audit: JP03 leaves that code unchanged, and compares selected-column/NMS digests instead. Its encoder audit is c1_graph_mode=audit. No JP03 instrumented profile was scheduled; PROFILE_PILOT.csv is header-only, not zero stage time.', '',
             'All dataset/row metrics and intervals: JP03_BENCHMARK_PILOT.csv. Every repetition, including adverse results: JP03_REPETITIONS_PILOT.csv. Measured point changes cannot establish significance when paired intervals include0. The isolated JP02 batch1 gain must not be reported as the pipeline gain. Energy, actual pipelined throughput and complete adaptive/deadline benchmarks are not measured here.', '',
             f'Window `{args.window}` completed with original clocks/fan restored and no restoration errors. Frozen/source/engine manifests are alongside state.json. Commands: `PYTHONDONTWRITEBYTECODE=1 python3 experiments/latency_opt_20261030/c1_pipeline_window.py`; after `source env.sh` on VUW, `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/A_method/k41_jetson_pilot_summary.py --input results/latency_opt/jp03 --check-counters`.', '']
    if state.get('resume'):
        text += ['Operational interruption: the original tool execution session and wrapper disappeared during repetition2. The child finished and wrote its artifacts/DONE marker, but its exit code was not observed. Five complete clean runs were retained after validation; seven remaining runs continued in the original registered order using a detached controller. Device did not reboot; fan/telemetry remained active and locked clocks were reasserted. The idle gap interrupts continuous acquisition and is a reproducibility limitation. All repetitions, including adverse outcomes, are retained. Original state and resume metadata are preserved in the window; no trial was selected or discarded based on latency. Resume command: `python3 -u scripts/98_resume_latency_jp03.py`.','']
    (OUT / 'JP03_REPORT.md').write_text('\n'.join(text))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    plot_rows = selected[1:4] + selected[:1] + selected[4:]
    for ds, ax in zip(names, axs):
        q = {x['row']: x for x in rows if x['dataset'] == ds and x['metric'] == 'p95'}
        for key, marker, label in [('baseline_ms', 'o-', 'Baseline'), ('candidate_ms', 's-', 'C1 fixed graphs')]:
            ax.plot(range(6), [float(q[r][key]) for r, _ in plot_rows], marker, label=label)
        ax.set_xticks(range(6)); ax.set_xticklabels([label for _, label in plot_rows], rotation=25)
        ax.set_title(names[ds]); ax.set_ylabel('p95 latency (ms)'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.suptitle('JP03 dev feasibility — 512 timed frames × 3 repeats'); fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'JP03_p95.{ext}', dpi=180, bbox_inches='tight')
    print(OUT / 'JP03_REPORT.md')


if __name__ == '__main__':
    main()
