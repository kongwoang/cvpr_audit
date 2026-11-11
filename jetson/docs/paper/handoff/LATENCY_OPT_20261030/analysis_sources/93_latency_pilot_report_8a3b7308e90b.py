#!/usr/bin/env python3
"""Render the completed JP01 pilot; no model import or inference."""
import argparse
import csv
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
H=ROOT/'docs/paper/handoff/LATENCY_OPT_20261030'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('summary',type=Path);ap.add_argument('window',type=Path);args=ap.parse_args()
    for p in args.summary.iterdir():
        if p.suffix in ('.csv','.json'):
            shutil.copyfile(p,H/(p.name if p.name.endswith('_PILOT.csv') else 'JP01_'+p.name))
    state_path=args.window/'state.recovered.json'
    if not state_path.exists():state_path=args.window/'state.json'
    state=json.loads(state_path.read_text())
    assert state['status'] in ('complete','recovered_complete') and not state['restore_errors']
    shutil.copyfile(state_path,H/'JP01_window_state.json')
    shutil.copyfile(args.window/'state.json',H/'JP01_original_window_state.json')
    checks={}
    for ds in ('nusc_val','av2_dev'):
        checks[ds]=json.loads((args.window/f'parity_{ds}.json').read_text())
        assert checks[ds]['mismatches']==0
        shutil.copyfile(args.window/f'parity_{ds}.json',H/f'JP01_parity_{ds}.json')
    rows=list(csv.DictReader((args.summary/'BENCHMARK_PILOT.csv').open()))
    decision=json.loads((args.summary/'feasibility.json').read_text())
    names={'nusc_val':'nuScenes dev','av2_dev':'AV2 CFG-DEV'}
    text=['# JP01 — Jetson optimization feasibility pilot','',
        'Authorized by the user after the consolidated RUNPLAN. This is a dev-prefix engineering pilot, not the full K41 deadline benchmark or confirmation on the timed test splits. Frozen v6-S and the paper/K38/K39 sources remain unchanged. Candidate: remove unused all-class raw-count reduction in `gpupost.host_columns`; unchanged selected columns and downstream postprocessing. The candidate is applied equally to every row, including whole-frame baselines.','',
        'Protocol: Jetson AGX Xavier, MAXN, locked clocks, fan255; original K39 staged/static-graph runtime, C1 trt32 and v3a. Each dataset uses the first712 development frames in original order,200 warmup and512 timed; three clean repetitions per backend, alternating baseline/candidate order. No test-based selection. Debug output/column audit and instrumented profiles are separate from clean timing. Bootstrap:2,000 paired scene/log resamples, seed0, repetitions retained within each unit; Δ=candidate−baseline. Intervals are exploratory and not multiplicity-adjusted.','',
        '| Dataset / row | Mean baseline→candidate (ms) | p95 baseline→candidate (ms) | Δp95 95% CI (ms) | p99 baseline→candidate (ms) |',
        '|---|---|---|---|---|']
    for ds in names:
        for row,label in [('ours@trt32','DESP'),('u960+c1comp@trt32','u960+C1+refiner'),('ours_b2A@trt32','B2K-A')]:
            q={x['metric']:x for x in rows if x['dataset']==ds and x['row']==row}
            pair=lambda metric:f"{float(q[metric]['baseline_ms']):.3f}→{float(q[metric]['candidate_ms']):.3f}"
            p=q['p95'];ci=f"[{float(p['delta_lo_ms']):+.3f}, {float(p['delta_hi_ms']):+.3f}]"
            text.append(f"| {names[ds]} / {label} | {pair('mean')} | {pair('p95')} | {ci} | {pair('p99')} |")
    text+=['','All rows, achieved frame/unit counts, means/tails and intervals are in BENCHMARK_PILOT.csv; three individual repetitions in REPETITIONS_PILOT.csv. No energy saving is claimed: board telemetry was collected, but it was not aligned to individual row intervals. Cold initialization is excluded from the reported steady-state timings.','',
           'Parity: debug baseline versus candidate audit passed on712 frames per dataset, including warmup; compare saved actions, decision costs, crops, pixels, round/crop counts, own history and deterministic digests of selected detector columns/NMS, crop requests/canvas metadata, lifted C1 geometry/TTC, and refined merged scenes. Selected-column equality is also asserted against reference on each audit call. This establishes only prefix parity; full-dev/test parity remains outstanding. Clean timing runs separately require equality of all stored scalar decision/pixel/round endpoints across all three repetitions.','',
           'Feasibility screen fixed before timing:≥5% DESP p95 reduction on a dataset, paired interval below0, no output mismatch and no significant mean/p95 slowdown in the included whole-frame rows.']
    for q in decision['decisions']:
        text.append(f"- {names[q['dataset']]}: p95 reduction {q['p95_reduction_percent']:.2f}%; screen {'PASS' if q['promising'] else 'NOT MET'}; whole-frame slowdown flag={q['wholeframe_significant_slowdown']}.")
    text+=['','A failed screen rejects expansion of this particular candidate based on the pilot; it does not prove every exact optimization is impossible. PROFILE_PILOT.csv uses the original instrumented breakdown: `crops_rN` means successive crop calls, not true algorithm rounds/guard depth. Use it to locate expensive stages, not to claim causal guard attribution. Bounded variants change scheduling/outputs and are separate algorithms.','',
           'Reproduction: `PYTHONDONTWRITEBYTECODE=1 python3 experiments/latency_opt_20261030/pilot_window.py` on KIT within the user-authorized scope. This command owns the exclusive window and restores clocks/fan in finally. CPU analysis on VUW: `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/A_method/k41_jetson_pilot_summary.py --input results/latency_opt/jp01/` after `source env.sh`. Actual window: `'+str(args.window)+'`; source/engine hashes and telemetry alongside state.json. Main successful window reports no restoration errors. The first80-frame smoke succeeded before an initial wrapper-only duplicate-key error; that attempt restored controls and is retained separately, never used for timing.','']
    text=[x.replace('Main successful window reports no restoration errors.', 'The original wrapper did not finalize its state; its exit and original hardware restoration were not observed. After a machine restart, recovery verified all20 DONE markers, unchanged source/engine/protected hashes and explicitly restored controls without error. Original state.json is preserved; state.recovered.json records this distinction. The last profile exit code is unknown; all DataFrame and clean timing endpoint checks were rerun on CPU.') for x in text]
    (H/'PILOT_REPORT.md').write_text('\n'.join(text))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    selected=['u512+c1comp@trt32','u640+c1comp@trt32','u960+c1comp@trt32','ours@trt32','ours_b2A@trt32','ours_b1A@trt32']
    labels=['u512','u640','u960','DESP','B2K-A','B1K-A']
    for ds,ax in zip(names,axs):
        q={x['row']:x for x in rows if x['dataset']==ds and x['metric']=='p95'}
        baseline=[float(q[x]['baseline_ms']) for x in selected];candidate=[float(q[x]['candidate_ms']) for x in selected]
        ax.plot(range(6),baseline,'o-',label='K39 baseline');ax.plot(range(6),candidate,'s-',label='columns-only')
        ax.set_xticks(range(6));ax.set_xticklabels(labels,rotation=25);ax.set_title(names[ds]);ax.set_ylabel('p95 latency (ms)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('JP01 dev feasibility pilot — 512 timed frames × 3 repeats');fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(H/f'JP01_p95.{ext}',dpi=180,bbox_inches='tight')
    print(H/'PILOT_REPORT.md')


if __name__=='__main__':main()
