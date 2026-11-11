"""K38-2 report (post-hoc, test splits): v6-S vs v6-S with the tab:locality probe changes together (v6Sp: closing probe
10 m/s, extended grid, PED / CAR / TRUCK probes; real crops). Records: results/k38/k2/<ds> (k30_probe.py).
Per split: G (refiner v3a), px mean / p95, agreement with pi(F) = u960 + C1 without refiner (a_<row>|none ==
a_wf960+C1|none), decisive-object coverage (K34-2: the decisive object's box centre inside the executed field / support /
guard crops; population and closing-speed terciles from handoff/K34/coverage_frames.csv), overall and in the high
closing-speed tercile; paired unit bootstrap (2,000, seed 0) of v6Sp - v6S.
Checks: v6S (recomputed) equals the stored K28 records (a, J, px) and its coverage equals K34-2's covered flag.
usage: k38_k2_report.py <coverage_frames.csv>   -> results/k38/k2_tables/{k38_2_rows.csv, k38_2_diffs.csv, checks.json}
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
from k34_common import B, bootstrap, interval

K = B / 'results/k38/k2'
EXP = {'nusc': (228, 9181), 'av2': (150, 11818), 'ahd': (24, 15006)}


def recs(folder):
    rr = []
    for f in sorted(folder.glob('*.pkl')):
        rr += pickle.load(f.open('rb'))
    d = pd.DataFrame(rr)
    d['unit'] = d.unit.astype(str)
    return d


def main():
    cov = pd.read_csv(sys.argv[1])
    cov['unit'] = cov.unit.astype(str)
    rows, diffs, checks = [], [], {}
    for ds, (nu, nf) in EXP.items():
        d = recs(K / ds)
        ref = recs(B / 'results/k28/p3' / f'{ds}_test_instr')
        m = d.merge(ref, on=['unit', 'frame'], suffixes=('', '_k28'), validate='1:1')
        checks[f'{ds}_v6S_vs_K28'] = {c: int((~np.isclose(m[c].astype(float), m[c + '_k28'].astype(float))).sum())
                                      for c in ('a_cheap', 'J_cheap', 'a_v6S|none', 'a_v6S|v3a', 'J_v6S|v3a', 'px_v6S', 'a_wf960+C1|none')}
        d = d[d['gt'].astype(bool)].sort_values(['unit', 'frame']).reset_index(drop=True)
        assert (d.unit.nunique(), len(d)) == (nu, nf), (ds, d.unit.nunique(), len(d))
        c = cov[cov.dataset == ds].merge(d[['unit', 'frame', 'dec_in_crop_v6S', 'dec_in_crop_v6Sp']], on=['unit', 'frame'], how='left', validate='1:1')
        checks[f'{ds}_coverage_v6S_vs_K34'] = int((c.dec_in_crop_v6S.astype(bool) != c.covered.astype(bool)).sum())
        units, C = bootstrap(d.unit)
        n = d.groupby('unit').size().reindex(units).to_numpy()
        Jc = d.groupby('unit').J_cheap.sum().reindex(units).to_numpy()
        stat = {}
        for row in ('v6S', 'v6Sp'):
            Ju = d.groupby('unit')[f'J_{row}|v3a'].sum().reindex(units).to_numpy()
            Pu = d.groupby('unit')[f'px_{row}'].sum().reindex(units).to_numpy()
            agr = (d[f'a_{row}|none'] == d['a_wf960+C1|none']).astype(float)
            Au = agr.groupby(d.unit).sum().reindex(units).to_numpy()
            s = {'G': ((Jc.sum() - Ju.sum()) / Jc.sum(), 1 - (C @ Ju) / (C @ Jc)), 'px_mean': (Pu.sum() / n.sum(), (C @ Pu) / (C @ n)),
                 'agreement_with_piF': (Au.sum() / n.sum(), (C @ Au) / (C @ n))}
            for tag, sub in (('coverage_all', c), ('coverage_high_closing', c[c.closing_speed_tercile == 'high'])):
                hit = sub[f'dec_in_crop_{row}'].astype(float)
                Hu = hit.groupby(sub.unit).sum().reindex(units).fillna(0).to_numpy()
                Nu = sub.groupby('unit').size().reindex(units).fillna(0).to_numpy()
                s[tag] = (Hu.sum() / Nu.sum(), (C @ Hu) / np.maximum(C @ Nu, 1e-12))
                s[tag + '_n'] = (int(Nu.sum()), None)
            stat[row] = s
            r = {'dataset': ds, 'row': 'v6-S' if row == 'v6S' else 'v6-S + probe changes (vc 10, extended grid, 3 probes)',
                 'units': nu, 'frames': nf, 'px_p95': float(np.percentile(d[f'px_{row}'], 95))}
            for k, (pt, dist) in s.items():
                r[k] = pt
                if dist is not None:
                    iv = interval(dist); r[k + '_ci_lo'], r[k + '_ci_hi'] = iv['ci_lo'], iv['ci_hi']
            rows.append(r)
        for k in ('G', 'px_mean', 'agreement_with_piF', 'coverage_all', 'coverage_high_closing'):
            iv = interval(stat['v6Sp'][k][1] - stat['v6S'][k][1])
            diffs.append({'dataset': ds, 'statistic': k, 'v6Sp_minus_v6S': stat['v6Sp'][k][0] - stat['v6S'][k][0],
                          'ci_lo': iv['ci_lo'], 'ci_hi': iv['ci_hi'], 'excludes_0': bool(iv['ci_lo'] > 0 or iv['ci_hi'] < 0)})
    od = B / 'results/k38/k2_tables'; od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(od / 'k38_2_rows.csv', index=False)
    pd.DataFrame(diffs).to_csv(od / 'k38_2_diffs.csv', index=False)
    (od / 'checks.json').write_text(json.dumps(checks, indent=1) + '\n')
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).round(4).to_string()); print(pd.DataFrame(diffs).round(4).to_string()); print(checks)


if __name__ == '__main__':
    main()
