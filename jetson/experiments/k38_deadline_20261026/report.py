"""K35-4 Jetson K29-frame timing tables (v6-S / B2K-A / v7a; staged runtime + FAST_RECOVERY): per dataset x row, latency
over the 3 repetitions x K29 timed frames pooled; paired unit bootstrap (2,000, seed 0, repetitions nested in units)
of p95 / mean differences vs v6-S. usage: report.py  -> docs/paper/handoff/K35/K35-4/jetson_k29/{k29_rows.csv, k29_diffs.csv}"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]/'docs/paper/handoff/K35/K35-4/jetson_k29'
ROWS = {'ours': 'v6-S', 'ours_b2A': 'B2K-A', 'ours_b2A_retro': 'v7a'}


def stats(x):
    return {'mean_ms': x.mean(), 'median_ms': np.median(x), 'p95_ms': np.percentile(x, 95), 'p99_ms': np.percentile(x, 99),
            'max_ms': x.max(), 'gt50': (x > 50).mean(), 'gt66_7': (x > 200 / 3).mean(), 'gt100': (x > 100).mean()}


def main():
    st = json.loads((HERE/'logs/state.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows, diffs, gate = [], [], {}
    for ds in ('nusc_test20', 'av2'):
        runs = [Path(r['run']) for r in st['runs'] if r['dataset'] == ds and not r['smoke'] and r['runtime'] == 'staged']
        assert len(runs) == 3, (ds, runs)
        d = pd.concat([pd.read_pickle(r/'frames.pkl').assign(rep=i + 1) for i, r in enumerate(runs)])
        d = d[~d.warm].reset_index(drop=True)
        assert len(d) == 3 * {'nusc_test20': 805, 'av2': 586}[ds], len(d)
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
    pd.DataFrame(rows).to_csv(OUT/'k29_rows.csv', index=False)
    pd.DataFrame(diffs).to_csv(OUT/'k29_diffs.csv', index=False)
    
    print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(diffs).round(3).to_string()); 


if __name__ == '__main__':
    main()
