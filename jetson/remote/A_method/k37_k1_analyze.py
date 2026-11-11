"""K37-1 analysis (written before the test / holdout records were read; rule (c) as handoff/K37/K37-1/PREREG.md).
Records: results/k37/k1/{dev,test,hold}/<ds>/*.pkl (experiments/k37_thr_20261026/k30_thr.py / k30_thr_h.py).
Grid thr_decel {0.6, 0.8, 1.0, 1.2, 1.5}, thr_hard 3.5; GT / episodes / cost at the deployed definition.
(a/b) every row x threshold x split: missed rate (K31 L = 1; all, below / at-or-above hard), false episodes / 1,000, G,
      insufficient-braking loss per frame (lam_risk * max(a_gt - commanded, 0)^2, K34-1), mean px; paired unit bootstrap
      (2,000, seed 0) of row - DESP at the same threshold.
(c)   per row and dev dataset: the largest threshold whose dev missed rate <= cheap's dev missed rate at 1.0; Argoverse-HD
      and the holdout use the AV2 choice; infeasible -> lowest-miss point (largest threshold on ties), labelled; evaluated
      once on test and once on the holdout, with intervals vs DESP's constrained point.
Parity at thr 1.0 (asserted): dev vs results/k30/dev_devref; test vs results/k28/p3/<ds>_test_instr (+ K30 match hybrid
x1.00, + K34-1 continuous_demands whole-frame actions at every threshold); holdout vs results/k35/k4/train/k28p3.
usage: k37_k1_analyze.py dev | all      -> results/k37/k1/tables/
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
from rap import planner as P
from k34_common import B, bootstrap, interval
from k35_k4_analyze import per_unit

K1 = B / 'results/k37/k1'
GRID = (0.6, 0.8, 1.0, 1.2, 1.5)
NW = {'nusc': 1600, 'av2': 2048, 'ahd': 1920}
NUNITS = {('dev', 'nusc'): 152, ('dev', 'av2'): 300, ('test', 'nusc'): 228, ('test', 'av2'): 150, ('test', 'ahd'): 24, ('hold', 'ahd'): 65}
PP, CP = P.PlannerParams(), P.CostParams()
CMD = np.array([P.commanded_decel(i, PP) for i in range(3)])


def sfx(t):
    return '' if t == 1.0 else f'@t{t:g}'


def rows_of(ds):
    n = NW[ds]
    return {'cheap': lambda s: ('a_cheap' + s, 'J_cheap' + s, 'px_cheap'),
            'u960 + C1 + v3a': lambda s: ('a_wf960+C1|v3a' + s, 'J_wf960+C1|v3a' + s, 'px_wf960'),
            'native + C1 + v3a': lambda s: (f'a_wf{n}+C1|v3a' + s, f'J_wf{n}+C1|v3a' + s, f'px_wf{n}'),
            'hybrid tsp8 (DESP budget) + C1 + v3a': lambda s: ('a_tsp8@x1.00+C1|v3a' + s, 'J_tsp8@x1.00+C1|v3a' + s, 'px_tsp8@x1.00+C1' + s),
            'DESP (v6-S loop)': lambda s: ('a_v6S|v3a' + s, 'J_v6S|v3a' + s, 'px_v6S' + s)}


def recs(folder):
    rr = []
    for f in sorted(folder.glob('*.pkl')):
        rr += pickle.load(f.open('rb'))
    return pd.DataFrame(rr)


def load(split, ds):
    d = recs(K1 / split / ds)
    assert d.unit.nunique() == NUNITS[(split, ds)], (split, ds, d.unit.nunique())
    d['unit'] = d.unit.astype(str)
    return d


def parity(split, ds, d):
    out = {}
    if split == 'dev':
        ref = recs(B / 'results/k30/dev_devref' / ds)
        cols = ['a_cheap', 'J_cheap', 'a_v6S|v3a', 'J_v6S|v3a', 'px_v6S', 'a_wf960+C1|v3a', f'a_wf{NW[ds]}+C1|v3a']
    elif split == 'test':
        ref = recs(B / 'results/k28/p3' / f'{ds}_test_instr')
        cols = ['a_cheap', 'J_cheap', 'a_v6S|v3a', 'J_v6S|v3a', 'px_v6S', 'a_wf960+C1|v3a', f'a_wf{NW[ds]}+C1|v3a']
    else:
        ref = recs(B / 'results/k35/k4/train/k28p3/ahd_test_instr')
        cols = ['a_cheap', 'J_cheap', 'a_v6S|v3a', 'J_v6S|v3a', 'px_v6S', 'a_wf960+C1|v3a', 'a_wf1920+C1|v3a']
    ref['unit'] = ref.unit.astype(str)
    m = d.merge(ref, on=['unit', 'frame'], suffixes=('', '_ref'), validate='1:1')
    assert len(m) == len(d) == len(ref), (split, ds, len(m), len(d), len(ref))
    for c in cols:
        out[c] = int((~np.isclose(m[c].astype(float), m[c + '_ref'].astype(float), atol=1e-8)).sum())
    if split == 'test':
        mt = recs(B / 'results/k30/match' / ds)
        mt['unit'] = mt.unit.astype(str)
        m = d.merge(mt[['unit', 'frame', 'a_tsp8@x1.00+C1|v3a']], on=['unit', 'frame'], suffixes=('', '_ref'), validate='1:1')
        out['a_tsp8@x1.00+C1|v3a (K30 match)'] = int((m['a_tsp8@x1.00+C1|v3a'] != m['a_tsp8@x1.00+C1|v3a_ref']).sum())
        cd = pd.read_csv(B / 'results/k34/operating' / ds / 'continuous_demands.csv')
        if True:
            cd['unit'] = cd.unit.astype(str)
            m = d.merge(cd, on=['unit', 'frame'], suffixes=('', '_k34'), validate='1:1')
            for L in (960, NW[ds]):
                for t in GRID:
                    c = f'a_wf{L}+C1|v3a@{t:g}'
                    if c in cd:
                        out[f'{c} (K34-1)'] = int((m[f'a_wf{L}+C1|v3a' + sfx(t)] != m[c]).sum())
    return out


def metrics(d, units, C, ac, jc, pc):
    n = d.groupby('unit').size().reindex(units).to_numpy()
    Jc = d.groupby('unit').J_cheap.sum().reindex(units).to_numpy()
    Ju = d.groupby('unit')[jc].sum().reindex(units).to_numpy()
    Pu = d.groupby('unit')[pc].sum().reindex(units).to_numpy()
    loss = PP_lam() * np.maximum(d.a_gt.to_numpy(float) - CMD[d[ac].to_numpy(int)], 0) ** 2
    Lu = pd.Series(loss).groupby(d.unit.to_numpy()).sum().reindex(units).to_numpy()
    s = {'G': ((Jc.sum() - Ju.sum()) / Jc.sum(), 1 - (C @ Ju) / (C @ Jc)), 'px': (Pu.sum() / n.sum(), (C @ Pu) / (C @ n)),
         'insufficient_loss_per_frame': (Lu.sum() / n.sum(), (C @ Lu) / (C @ n))}
    pu = per_unit(d, ac, 1).reindex(units)
    s['false_per_1000'] = (1e3 * pu.false.sum() / n.sum(), 1e3 * (C @ pu.false.to_numpy()) / (C @ n))
    for sev in ('all', 'below_hard', 'at_or_above_hard'):
        num, den = pu[sev + '_miss'].to_numpy(float), pu[sev + '_n'].to_numpy(float)
        s['missed_rate_' + sev] = (num.sum() / max(den.sum(), 1), (C @ num) / np.maximum(C @ den, 1e-12))
    s['_episodes'] = (int(pu.all_n.sum()), None)
    s['_missed'] = (int(pu.all_miss.sum()), None)
    return s


def PP_lam():
    return CP.lam_risk


def table(split, ds):
    d = load(split, ds)
    par = parity(split, ds, d)
    d = d[d['gt'].astype(bool)].sort_values(['unit', 'frame']).reset_index(drop=True)
    d['a_gt_k0'] = d['a_gt']
    units, C = bootstrap(d.unit)
    pts, diffs, stat = [], [], {}
    for name, f in rows_of(ds).items():
        for t in GRID:
            s = metrics(d, units, C, *f(sfx(t)))
            stat[(name, t)] = s
            r = {'split': split, 'dataset': ds, 'row': name, 'thr_decel': t, 'units': len(units), 'frames': len(d),
                 'episodes': s['_episodes'][0], 'missed': s['_missed'][0]}
            for k, (pt, dist) in s.items():
                if k.startswith('_'):
                    continue
                iv = interval(dist)
                r[k], r[k + '_ci_lo'], r[k + '_ci_hi'] = pt, iv['ci_lo'], iv['ci_hi']
            pts.append(r)
    for (name, t), s in stat.items():
        if name.startswith('DESP'):
            continue
        ref = stat[('DESP (v6-S loop)', t)]
        for k in s:
            if k.startswith('_'):
                continue
            iv = interval(s[k][1] - ref[k][1])
            diffs.append({'split': split, 'dataset': ds, 'row': name, 'thr_decel': t, 'statistic': k, 'row_minus_DESP': s[k][0] - ref[k][0],
                          'ci_lo': iv['ci_lo'], 'ci_hi': iv['ci_hi'], 'excludes_0': bool(iv['ci_lo'] > 0 or iv['ci_hi'] < 0)})
    return pts, diffs, par, stat, (d, units, C)


def choose(dev_pts):
    """rule (c) on the dev points"""
    out = []
    P_ = pd.DataFrame(dev_pts)
    for ds in ('nusc', 'av2'):
        q = P_[P_.dataset == ds]
        bound = float(q[(q.row == 'cheap') & (q.thr_decel == 1.0)].missed_rate_all.iloc[0])
        for row, g in q.groupby('row', sort=False):
            ok = g[g.missed_rate_all <= bound + 1e-12]
            if len(ok):
                t, feas = float(ok.thr_decel.max()), True
            else:
                m = g.missed_rate_all.min()
                t, feas = float(g[np.isclose(g.missed_rate_all, m)].thr_decel.max()), False
            out.append({'dev_dataset': ds, 'row': row, 'cheap_dev_missed_rate_at_1.0': bound, 'chosen_thr_decel': t, 'feasible': feas,
                        'dev_missed_rate_at_choice': float(g[g.thr_decel == t].missed_rate_all.iloc[0])})
    return pd.DataFrame(out)


def constrained(split, ds, stat, ctx, ch):
    d, units, C = ctx
    src = 'nusc' if ds == 'nusc' else 'av2'
    c = ch[ch.dev_dataset == src].set_index('row')
    t6 = float(c.loc['DESP (v6-S loop)', 'chosen_thr_decel'])
    ref = stat[('DESP (v6-S loop)', t6)]
    rows = []
    for name in rows_of(ds):
        t = float(c.loc[name, 'chosen_thr_decel'])
        s = stat[(name, t)]
        r = {'split': split, 'dataset': ds, 'row': name, 'thr_decel': t, 'feasible_on_dev': bool(c.loc[name, 'feasible']),
             'choice_from_dev': src, 'DESP_thr_decel': t6, 'episodes': s['_episodes'][0], 'missed': s['_missed'][0]}
        for k in ('G', 'false_per_1000', 'missed_rate_all', 'missed_rate_below_hard', 'missed_rate_at_or_above_hard', 'px', 'insufficient_loss_per_frame'):
            r[k] = s[k][0]
            if not name.startswith('DESP'):
                iv = interval(s[k][1] - ref[k][1])
                r[k + '_minus_DESP'], r[k + '_minus_DESP_ci_lo'], r[k + '_minus_DESP_ci_hi'] = s[k][0] - ref[k][0], iv['ci_lo'], iv['ci_hi']
        rows.append(r)
    return rows


def main():
    od = K1 / 'tables'; od.mkdir(parents=True, exist_ok=True)
    pts, diffs, par, keep = [], [], {}, {}
    for ds in ('nusc', 'av2'):
        p_, d_, pa, st, ctx = table('dev', ds)
        pts += p_; diffs += d_; par[f'dev_{ds}'] = pa
    ch = choose(pts)
    ch.to_csv(od / 'k1_choice.csv', index=False)
    print(ch.to_string())
    cons = []
    if sys.argv[1] == 'all':
        for split, dss in (('test', ('nusc', 'av2', 'ahd')), ('hold', ('ahd',))):
            for ds in dss:
                p_, d_, pa, st, ctx = table(split, ds)
                par[f'{split}_{ds}'] = pa
                cons += constrained(split, ds, st, ctx, ch)
                if split == 'test':                         # the holdout reports only the chosen points
                    pts += p_; diffs += d_
        pd.DataFrame(cons).to_csv(od / 'k1_constrained.csv', index=False)
    pd.DataFrame(pts).to_csv(od / 'k1_points.csv', index=False)
    pd.DataFrame(diffs).to_csv(od / 'k1_diffs.csv', index=False)
    (od / 'parity.json').write_text(json.dumps(par, indent=1) + '\n')
    bad = {k: {c: n for c, n in v.items() if n} for k, v in par.items()}
    print('parity nonzero:', bad)
    print(pd.DataFrame(pts)[['split', 'dataset', 'row', 'thr_decel', 'missed', 'episodes', 'missed_rate_all', 'false_per_1000', 'G', 'px']].round(4).to_string())
    if cons:
        print(pd.DataFrame(cons).round(4).to_string())
    print('K37-1 ANALYSIS DONE')


if __name__ == '__main__':
    main()
