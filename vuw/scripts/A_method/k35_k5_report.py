"""K35-5 analysis: changed-action rate, dG with paired unit bootstrap intervals (2,000 draws, seed 0) and the attribution
of the changes, per dataset x budget (+25% / +50% of v6-S's crop px) x seed, and the mean over seeds.
dG = G(row) - G(v6-S) = (sum J_v6S - sum J_row) / sum J_cheap, costs by K28's decision_cost with cheap's previous action.
Reference rates for the prediction: v6-S without refiner vs pi(F) disagreement 2.4 / 3.2 / 6.0% (K30 V3).
usage: k35_k5_report.py   -> results/k35/k5/tables/{k5_rows.csv, k5_attribution.csv}
"""
import pickle

import numpy as np
import pandas as pd
from rap import planner as P
from k34_common import B, bootstrap, interval, load_base

REF = {'nusc': 0.024, 'av2': 0.032, 'ahd': 0.060}


def main():
    od = B / 'results/k35/k5/tables'; od.mkdir(parents=True, exist_ok=True)
    pp, cp = P.PlannerParams(), P.CostParams()
    rows, att = [], []
    for ds in ('nusc', 'av2', 'ahd'):
        d = load_base(ds, [])
        rr = []
        for f in sorted((B / 'results/k35/k5' / ds).glob('*.pkl')):
            rr += pickle.load(f.open('rb'))['rows']
        r = pd.DataFrame(rr)
        assert r.groupby(['budget', 'seed']).size().nunique() == 1 and r.groupby(['budget', 'seed']).size().iloc[0] == len(d), ds
        units, C = bootstrap(d.unit)
        base = d[['unit', 'frame', 'a_gt', 'prev_cheap', 'J_cheap', 'J_v6S|v3a', 'a_v6S|v3a']]
        Lc = base.groupby('unit').J_cheap.sum().reindex(units).to_numpy()
        Jv = base.groupby('unit')['J_v6S|v3a'].sum().reindex(units).to_numpy()
        per = []
        for (bud, seed), q in r.groupby(['budget', 'seed']):
            q = base.merge(q, on=['unit', 'frame'], how='inner', validate='one_to_one')
            assert (q['a_v6S|v3a'] == q.a_v6S).all()
            J = np.array([P.decision_cost(int(a), g, None if pd.isna(p_) else int(p_), pp, cp)['J'] for a, g, p_ in zip(q.a_new, q.a_gt, q.prev_cheap)])
            Ju = pd.Series(J).groupby(q.unit.to_numpy()).sum().reindex(units).to_numpy()
            dG = (C @ Jv - C @ Ju) / (C @ Lc)
            iv = interval(dG)
            rec = {'dataset': ds, 'budget': f'+{int(bud * 100)}%', 'seed': seed, 'frames': len(q),
                   'frames_with_crops': int((q.n_crops > 0).sum()), 'mean_extra_px': float(q.extra_px.mean()),
                   'changed_rate': float(q.changed.mean()), 'reference_disagreement_rate': REF[ds],
                   'changed_rate_le_reference': bool(q.changed.mean() <= REF[ds]),
                   'dG': float((Jv.sum() - Ju.sum()) / Lc.sum()), 'dG_ci_lo': iv['ci_lo'], 'dG_ci_hi': iv['ci_hi'],
                   'dG_excludes_0': bool(iv['ci_lo'] > 0 or iv['ci_hi'] < 0)}
            rows.append(rec); per.append(rec)
            for a_, n in q[q.changed].attribution.value_counts().items():
                att.append({'dataset': ds, 'budget': f'+{int(bud * 100)}%', 'seed': seed, 'attribution': a_, 'changed_frames': int(n),
                            'share_of_changed': n / max(int(q.changed.sum()), 1)})
        for bud in sorted({x['budget'] for x in per}):
            z = pd.DataFrame([x for x in per if x['budget'] == bud])
            rows.append({'dataset': ds, 'budget': bud, 'seed': 'mean of 5', 'frames': int(z.frames.iloc[0]),
                         'frames_with_crops': float(z.frames_with_crops.mean()), 'mean_extra_px': float(z.mean_extra_px.mean()),
                         'changed_rate': float(z.changed_rate.mean()), 'reference_disagreement_rate': REF[ds],
                         'changed_rate_le_reference': bool(z.changed_rate.mean() <= REF[ds]), 'dG': float(z.dG.mean()),
                         'dG_ci_lo': float(z.dG_ci_lo.mean()), 'dG_ci_hi': float(z.dG_ci_hi.mean()), 'dG_excludes_0': None})
    pd.DataFrame(rows).to_csv(od / 'k5_rows.csv', index=False)
    pd.DataFrame(att).to_csv(od / 'k5_attribution.csv', index=False)
    print('K35-5 REPORT TABLES DONE')


if __name__ == '__main__':
    main()
