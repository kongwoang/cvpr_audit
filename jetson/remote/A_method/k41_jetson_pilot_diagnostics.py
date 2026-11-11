"""CPU-only descriptive attribution of original-runtime JP01 dev pilot tails."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    args = ap.parse_args()
    sp = args.input / 'state.recovered.json'
    if not sp.exists():
        sp = args.input / 'state.json'
    state = json.loads(sp.read_text())
    assert state['status'] in ('complete', 'recovered_complete')
    timings, profiles = {}, {}
    for r in state['runs']:
        if r['mode'] != 'baseline' or r['stage'] not in ('timing', 'profile'):
            continue
        d = pd.read_pickle(args.input / 'raw' / Path(r['run']).name / 'frames.pkl')
        d = d[~d.warm].set_index(['unit', 'frame']).sort_index()
        assert len(d) == 512 and d.index.is_unique
        if r['stage'] == 'timing':
            timings.setdefault(r['dataset'], []).append(d)
        else:
            profiles[r['dataset']] = d
    rows, stages = [], []
    for ds, repeats in timings.items():
        assert len(repeats) == 3
        ref = repeats[0]
        assert all(ref.index.equals(d.index) for d in repeats)
        own = np.stack([d['ms_ours@trt32'].to_numpy() for d in repeats]).mean(axis=0)
        threshold = np.quantile(own, .95)
        p = profiles[ds]
        assert p.index.equals(ref.index)
        # Disjoint measured components; all crop-call labels include guard calls.
        crop_cols = [c for c in p if c.startswith('bd_crops_r') and c.endswith('_ours@trt32')]
        parts = {
            'draft_including_upload_geometry': p['bd_draft_ours@trt32'].to_numpy(),
            'all_crop_calls_including_guard': p[crop_cols].fillna(0).sum(axis=1).to_numpy(),
            'C1_total_including_CPU_and_synchronization': p['bd_c1_ours@trt32'].fillna(0).to_numpy(),
            'support_refiner_including_patches': p['bd_refiner_ours@trt32'].fillna(0).to_numpy(),
        }
        parts['unattributed_residual_instrumented'] = p['ms_ours@trt32'].to_numpy() - sum(parts.values())
        assert parts['unattributed_residual_instrumented'].min() > -.001
        for group, mask in [('all', np.ones(len(ref), dtype=bool)), ('slowest_5percent_clean_mean', own >= threshold)]:
            for col in ['ms_ours@trt32','ms_u960+c1comp@trt32','execs_ours@trt32','calls_ours@trt32','crop_engine_exec_ours@trt32','n_crops_ours@trt32','rounds_ours@trt32','n_refined_ours@trt32','px_ours@trt32']:
                if col not in ref:
                    continue
                vals = np.stack([d[col].to_numpy() for d in repeats]).mean(axis=0)[mask]
                rows.append(dict(dataset=ds,group=group,endpoint=col,frames=int(mask.sum()),units=ref.index.get_level_values('unit')[mask].nunique(),mean=float(vals.mean()),median=float(np.median(vals)),max=float(vals.max()),tail_threshold_ms=float(threshold),scope='descriptive conditional association; not causal attribution'))
            for name, vals in parts.items():
                z = vals[mask]
                stages.append(dict(dataset=ds,group=group,stage=name,frames=int(mask.sum()),mean_ms=float(z.mean()),median_ms=float(np.median(z)),p95_ms=float(np.quantile(z,.95)),scope='separate instrumented run; group selected by clean baseline mean; no additive p95 claim'))
    out = args.input / 'summary'
    pd.DataFrame(rows).to_csv(out / 'DIAGNOSTICS_PILOT.csv', index=False)
    pd.DataFrame(stages).to_csv(out / 'PROFILE_GROUPS_PILOT.csv', index=False)
    (out / 'diagnostics_protocol.json').write_text(json.dumps(dict(script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),selection='baseline mean latency across three repetitions, fixed top5percent cutoff',stage_limit='instrumented wall timers include synchronization; crop-call labels are not algorithm rounds/guard depth',causal=False), indent=2)+'\n')
    print(pd.DataFrame(stages).to_string(index=False))


if __name__ == '__main__':
    main()
