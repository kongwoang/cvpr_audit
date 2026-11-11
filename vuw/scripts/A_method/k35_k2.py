"""K35-2 (post-hoc, three test splits, stored K28 p3 rows; no pipeline run): frame-level recoverability classes and
the offline best-of ceiling.
Classes (exclusive), per reference F in {u960 + C1 + v3a, native + C1 + v3a}; "correct" = the action equals act_gt:
  both_correct; cheap_wrong_F_correct (recoverable by escalation); both_wrong; cheap_correct_F_wrong (negative
  escalation). Populations: required frames (act_gt >= DECELERATE) and all evaluated frames.
Per class: frames, share, v6-S correct rate (paired unit bootstrap interval), v6-S's action relative to act_gt, and the
mean decision loss of cheap / F / v6-S (K28 stored costs; the jerk term uses cheap's previous action, as K28).
Ceiling (explicitly UNREALIZABLE, offline per-frame oracle choice): best-of{cheap, v6-S, F} loss per frame -> G with
interval, reported next to K28's oracle frame router (true V, top-k) at v6-S's px (routers_v6_<ds>.csv).
usage: k35_k2.py --dataset nusc|av2|ahd   -> results/k35/k2/<ds>/
"""
import argparse

import numpy as np
import pandas as pd
from k34_common import B, NWS, bootstrap, interval, load_base, ratio

OUT = B / 'results/k35/k2'
CLASSES = ['both_correct', 'cheap_wrong_F_correct', 'both_wrong', 'cheap_correct_F_wrong']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ds = ap.parse_args().dataset
    od = OUT / ds; od.mkdir(parents=True, exist_ok=True)
    d = load_base(ds, [])
    units, C = bootstrap(d.unit)
    ui = pd.Index(units).get_indexer(d.unit)
    g = d.act_gt.to_numpy(int)
    ac, av = d.a_cheap.to_numpy(int), d['a_v6S|v3a'].to_numpy(int)
    Jc, Jv = d.J_cheap.to_numpy(float), d['J_v6S|v3a'].to_numpy(float)
    rows, ceil = [], []

    def usum(x):
        return np.bincount(ui, weights=np.asarray(x, float), minlength=len(units))
    for fname, col in (('u960 + C1 + v3a', 'wf960+C1|v3a'), ('native + C1 + v3a', f'wf{NWS[ds]}+C1|v3a')):
        af, Jf = d['a_' + col].to_numpy(int), d['J_' + col].to_numpy(float)
        cc, fc = ac == g, af == g
        cls = np.select([cc & fc, ~cc & fc, ~cc & ~fc, cc & ~fc], CLASSES, default='none')
        for pop, pm in (('required_frames', g >= 1), ('all_frames', np.ones(len(d), bool))):
            npop = int(pm.sum())
            for k in CLASSES:
                m = pm & (cls == k)
                vc = av[m] == g[m]
                iv = interval(ratio(C @ usum(m & (av == g)), C @ usum(m)))
                rows.append({'dataset': ds, 'reference_F': fname, 'population': pop, 'class': k, 'frames': int(m.sum()),
                             'share': m.sum() / max(npop, 1), 'v6S_correct_rate': float(vc.mean()) if m.any() else np.nan,
                             'v6S_correct_ci_lo': iv['ci_lo'], 'v6S_correct_ci_hi': iv['ci_hi'],
                             'v6S_under_rate': float((av[m] < g[m]).mean()) if m.any() else np.nan,
                             'v6S_over_rate': float((av[m] > g[m]).mean()) if m.any() else np.nan,
                             'v6S_equals_F_rate': float((av[m] == af[m]).mean()) if m.any() else np.nan,
                             'v6S_equals_cheap_rate': float((av[m] == ac[m]).mean()) if m.any() else np.nan,
                             'mean_J_cheap': float(Jc[m].mean()) if m.any() else np.nan, 'mean_J_F': float(Jf[m].mean()) if m.any() else np.nan,
                             'mean_J_v6S': float(Jv[m].mean()) if m.any() else np.nan,
                             'share_of_cheap_total_loss_cheap': Jc[m].sum() / Jc.sum(), 'share_of_cheap_total_loss_v6S': Jv[m].sum() / Jc.sum()})
        best = np.minimum(np.minimum(Jc, Jv), Jf)
        Lc = usum(Jc)
        for name, J in (('best-of{cheap, v6-S, F} (UNREALIZABLE offline per-frame oracle)', best),
                        ('best-of{cheap, F} (UNREALIZABLE offline per-frame oracle)', np.minimum(Jc, Jf)),
                        ('v6-S', Jv), ('F', Jf)):
            G = 1 - J.sum() / Jc.sum()
            dist = 1 - (C @ usum(J)) / (C @ Lc)
            dv = (C @ usum(J) - C @ usum(Jv)) / (C @ Lc)
            ceil.append({'dataset': ds, 'reference_F': fname, 'row': name, 'G': G, **interval(dist),
                         'G_row_minus_v6S': float((Jv.sum() - J.sum()) / Jc.sum()),
                         'G_row_minus_v6S_lo': float(np.percentile(-dv, 2.5)), 'G_row_minus_v6S_hi': float(np.percentile(-dv, 97.5)),
                         'best_choice_share_v6S': float(((Jv <= Jc) & (Jv <= Jf)).mean()) if J is best else np.nan})
    pd.DataFrame(rows).to_csv(od / 'k2_classes.csv', index=False)
    C2 = pd.DataFrame(ceil)
    C2.to_csv(od / 'k2_ceiling.csv', index=False)
    print('K35-2 DONE', ds, len(d), 'frames')


if __name__ == '__main__':
    main()
