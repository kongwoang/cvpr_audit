"""K35-4b gate tables: per dataset x row, latency over the 3 x 512 timed frames pooled (staged runtime + FAST_RECOVERY);
paired unit bootstrap (2,000, seed 0, repetitions nested in units) of p95 / mean differences vs v6-S; gate decision.
usage: report.py  -> docs/paper/handoff/K35/K35-4b/{gate_rows.csv, gate_diffs.csv, gate.json}"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]/'docs/paper/handoff/K35/K35-4b'
ROWS = {'ours': 'v6-S', 'ours_b2A': 'B2K-A', 'ours_b2A_retro': 'v7a', 'ours_b2A_missing_extent_retro': 'v7b'}


def stats(x):
    return {'mean_ms': x.mean(), 'median_ms': np.median(x), 'p95_ms': np.percentile(x, 95), 'p99_ms': np.percentile(x, 99),
            'max_ms': x.max(), 'gt50': (x > 50).mean(), 'gt66_7': (x > 200 / 3).mean(), 'gt100': (x > 100).mean()}


def main():
    st = json.loads((HERE/'logs/state.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows, diffs, gate = [], [], {}
    for ds in ('nusc_val', 'av2_dev'):
        runs = [Path(r['run']) for r in st['runs'] if r['dataset'] == ds and not r['smoke'] and r['runtime'] == 'staged']
        assert len(runs) == 3, (ds, runs)
        d = pd.concat([pd.read_pickle(r/'frames.pkl').assign(rep=i + 1) for i, r in enumerate(runs)])
        d = d[~d.warm].reset_index(drop=True)
        assert len(d) == 3 * 512, len(d)
        for nm, lab in ROWS.items():
            x = d[f'ms_{nm}@trt32'].to_numpy()
            rows.append({'dataset': ds, 'row': lab, 'pipeline': nm, 'frames': len(x), **stats(x),
                         **{f'p95_rep{k}': np.percentile(d.loc[d.rep == k, f'ms_{nm}@trt32'], 95) for k in (1, 2, 3)}})
        units = d.unit.astype(str).to_numpy(); uu = np.unique(units)
        idx = {u: np.flatnonzero(units == u) for u in uu}
        rng = np.random.default_rng(0)
        draws = [np.concatenate([idx[u] for u in rng.choice(uu, len(uu))]) for _ in range(2000)]
        base = d['ms_ours@trt32'].to_numpy()
        for nm, lab in list(ROWS.items())[1:]:
            x = d[f'ms_{nm}@trt32'].to_numpy()
            for stat, f in (('p95', lambda a: np.percentile(a, 95)), ('mean', np.mean)):
                bs = np.array([f(x[k]) - f(base[k]) for k in draws])
                diffs.append({'dataset': ds, 'row': lab, 'vs': 'v6-S', 'stat': stat, 'diff_ms': f(x) - f(base),
                              'ci_lo': np.percentile(bs, 2.5), 'ci_hi': np.percentile(bs, 97.5), 'units': len(uu)})
        p7, p6 = np.percentile(d['ms_ours_b2A_missing_extent_retro@trt32'], 95), np.percentile(base, 95)
        gate[ds] = {'p95_v7b': p7, 'p95_v6S': p6, 'v7b_p95_gt_v6S': bool(p7 > p6)}
    gate['v7b_dropped'] = any(g['v7b_p95_gt_v6S'] for g in gate.values() if isinstance(g, dict))
    pd.DataFrame(rows).to_csv(OUT/'gate_rows.csv', index=False)
    pd.DataFrame(diffs).to_csv(OUT/'gate_diffs.csv', index=False)
    (OUT/'gate.json').write_text(json.dumps(gate, indent=2)+'\n')
    print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(diffs).round(3).to_string()); print(gate)


if __name__ == '__main__':
    main()
