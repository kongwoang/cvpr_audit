"""K35-4, POST-HOC (labelled): v7a vs v6-S (and the K28 / K33 rows of the stored test records) on the three test splits.
Metrics and intervals exactly as k35_k4_analyze.py: G (stored per-frame K28 cost), missed rate of K31 required episodes
(all / severity strata; HARD level), false episodes per 1,000, mean network px; paired unit bootstrap 2,000, seed 0.
Base: k34_common.load_base (K28 instr test records + K30 GT, evaluated frames; 228 / 9,181, 150 / 11,818, 24 / 15,006).
v7a: results/k35/k4/posthoc/r3low/<ds> (posthoc_check.py passed: its cheap / v6-S equal the stored records).
usage: k35_k4_posthoc.py   -> results/k35/k4/posthoc/tables/{posthoc_rows.csv, posthoc_diffs.csv, posthoc_episodes.csv}
"""
import pickle

import numpy as np
import pandas as pd
from k34_common import B, bootstrap, interval, load_base
from k35_k4_analyze import per_unit

K = B / 'results/k35/k4/posthoc'
NW = {'nusc': 1600, 'av2': 2048, 'ahd': 1920}


def rows_of(ds):
    n = NW[ds]
    return {'cheap': ('a_cheap', 'J_cheap', 'px_cheap'), 'v6-S': ('a_v6S|v3a', 'J_v6S|v3a', 'px_v6S'),
            'u960 + C1 + v3a': ('a_wf960+C1|v3a', 'J_wf960+C1|v3a', 'px_wf960'),
            'native + C1 + v3a': (f'a_wf{n}+C1|v3a', f'J_wf{n}+C1|v3a', f'px_wf{n}'),
            'hybrid tsp8@S + C1 + v3a': ('a_tsp8@S+C1|v3a', 'J_tsp8@S+C1|v3a', 'px_tsp8@S+C1'),
            'uncertainty unc2@S + C1 + v3a': ('a_unc2@S+C1|v3a', 'J_unc2@S+C1|v3a', 'px_unc2@S+C1'),
            'v7a = B2K-A + temporal evidence': ('a_v6S_b2A_retro|v3a', 'J_v6S_b2A_retro|v3a', 'px_v6S_b2A_retro')}


def main():
    out, diffs, eps = [], [], []
    for ds in ('nusc', 'av2', 'ahd'):
        d = load_base(ds, [])
        rr = []
        for f in sorted((K / 'r3low' / ds).glob('*.pkl')):
            rr += pickle.load(f.open('rb'))
        v = pd.DataFrame(rr)[['unit', 'frame'] + [c for c in pd.DataFrame(rr[:1]).columns if 'v6S_b2A_retro' in c]]
        d = d.merge(v, on=['unit', 'frame'], how='left', validate='one_to_one')
        assert d['a_v6S_b2A_retro|v3a'].notna().all(), ds
        d['a_gt_k0'] = d['a_gt']
        units, C = bootstrap(d.unit)
        n = d.groupby('unit').size().reindex(units).to_numpy()
        Jc = d.groupby('unit').J_cheap.sum().reindex(units).to_numpy()
        stat = {}
        for name, (ac, jc, pc) in rows_of(ds).items():
            Ju = d.groupby('unit')[jc].sum().reindex(units).to_numpy()
            Pu = d.groupby('unit')[pc].sum().reindex(units).to_numpy()
            s = {'G': ((Jc.sum() - Ju.sum()) / Jc.sum(), 1 - (C @ Ju) / (C @ Jc)), 'px': (Pu.sum() / n.sum(), (C @ Pu) / (C @ n))}
            for lv, tag in ((1, ''), (2, '_hard')):
                pu = per_unit(d, ac, lv).reindex(units)
                s['false_per_1000' + tag] = (1e3 * pu.false.sum() / n.sum(), 1e3 * (C @ pu.false.to_numpy()) / (C @ n))
                for sev in (('all', 'below_hard', 'at_or_above_hard') if lv == 1 else ('all',)):
                    num, den = pu[sev + '_miss'].to_numpy(float), pu[sev + '_n'].to_numpy(float)
                    s[f'missed_rate{tag}_{sev}'] = (num.sum() / max(den.sum(), 1), (C @ num) / np.maximum(C @ den, 1e-12))
                    eps.append({'dataset': ds, 'row': name, 'level': lv, 'severity': sev, 'episodes': int(den.sum()), 'missed': int(num.sum())})
            stat[name] = s
            r = {'dataset': ds, 'row': name, 'units': len(units), 'frames': len(d)}
            for k, (pt, dist) in s.items():
                iv = interval(dist)
                r[k], r[k + '_ci_lo'], r[k + '_ci_hi'] = pt, iv['ci_lo'], iv['ci_hi']
            out.append(r)
        for name in stat:
            if name == 'v6-S':
                continue
            for k in stat[name]:
                iv = interval(stat[name][k][1] - stat['v6-S'][k][1])
                diffs.append({'dataset': ds, 'row': name, 'statistic': k, 'row_minus_v6S': stat[name][k][0] - stat['v6-S'][k][0],
                              'ci_lo': iv['ci_lo'], 'ci_hi': iv['ci_hi'], 'excludes_0': bool(iv['ci_lo'] > 0 or iv['ci_hi'] < 0)})
    od = K / 'tables'; od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(od / 'posthoc_rows.csv', index=False)
    D = pd.DataFrame(diffs); D.to_csv(od / 'posthoc_diffs.csv', index=False)
    pd.DataFrame(eps).to_csv(od / 'posthoc_episodes.csv', index=False)
    print(pd.DataFrame(out)[['dataset', 'row', 'G', 'px', 'false_per_1000', 'missed_rate_all']].round(4).to_string())
    print(D[D.row.str.startswith('v7a')].round(4).to_string())
    print('K35-4 POSTHOC DONE')


if __name__ == '__main__':
    main()
