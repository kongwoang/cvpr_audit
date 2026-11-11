"""K35-4 confirmatory holdout analysis (pre-registered in handoff/K35/K35-4/PREREG.md; written before the holdout run).
Input: the K35-4 holdout records (Argoverse 1.1 tracking TRAIN = Argoverse-HD train.json, 65 units hold00..hold64):
  results/k35/k4/train/k28p3/ahd_test_instr/*.pkl   cheap, v6-S, u960 / native + C1 + v3a, tsp8@S, unc2@S
  results/k35/k4/train/k33/ahd_de_test/*.pkl        AnA iterated (ana05hinfi; @cap reported as G_cap)
  results/k35/k4/train/r3low/main/ahd/*.pkl         v7a = v6S_b2A_retro (+ its own cheap / v6-S, must equal k28p3's)
Evaluated frames: gt = True (3D labels). act_gt must equal K35-0's GT-only computation (results/k35/k0/av1_train_frames.csv).
Metrics per row: G = 1 - sum J / sum J_cheap (stored per-frame K28 cost, cheap's previous action in the jerk term);
missed rate of K31 required episodes (runs of act_gt >= DECELERATE, gaps <= 1 frame merged; missed = no frame of the
episode with action >= DECELERATE), also per severity (peak a_gt < / >= tau2) and at level HARD; false episodes per
1,000 evaluated frames (K31); mean network px (stored px_<row>, incl. recovery pixels for v7a).
Intervals: paired unit bootstrap, 2,000 draws, seed 0 (k34_common.bootstrap), percentile 95%.
Kill rule (v7a): CI lower bound of G(v7a) - G(v6-S) < -0.01, or missed rate(v7a) >= missed rate(v6-S) (point) ->
negative, not proposed.
usage: k35_k4_analyze.py   -> results/k35/k4/train/tables/{k4_rows.csv, k4_diffs.csv, k4_episodes.csv, k4_decision.json}
"""
import json
import pickle

import numpy as np
import pandas as pd
from k31_episodes import runs
from k34_common import B, bootstrap, interval

K4 = B / 'results/k35/k4/train'
ROWS = {  # name: (action column, cost column, px column)
    'cheap': ('a_cheap', 'J_cheap', 'px_cheap'),
    'v6-S': ('a_v6S|v3a', 'J_v6S|v3a', 'px_v6S'),
    'u960 + C1 + v3a': ('a_wf960+C1|v3a', 'J_wf960+C1|v3a', 'px_wf960'),
    'native + C1 + v3a': ('a_wf1920+C1|v3a', 'J_wf1920+C1|v3a', 'px_wf1920'),
    'hybrid tsp8@S + C1 + v3a': ('a_tsp8@S+C1|v3a', 'J_tsp8@S+C1|v3a', 'px_tsp8@S+C1'),
    'uncertainty unc2@S + C1 + v3a': ('a_unc2@S+C1|v3a', 'J_unc2@S+C1|v3a', 'px_unc2@S+C1'),
    'AnA iterated (ana05hinfi) + C1 + v3a': ('a_ana05hinfi|v3a', 'J_ana05hinfi|v3a', 'px_ana05hinfi'),
    'v7a = B2K-A + temporal evidence': ('a_v6S_b2A_retro|v3a', 'J_v6S_b2A_retro|v3a', 'px_v6S_b2A_retro'),
}
CAP = {'AnA iterated (ana05hinfi) + C1 + v3a': ('J_ana05hinfi@cap|v3a', 'px_ana05hinfi@cap')}
REF, CAND = 'v6-S', 'v7a = B2K-A + temporal evidence'


def recs(folder):
    rr = []
    for f in sorted(folder.glob('*.pkl')):
        rr += pickle.load(f.open('rb'))
    return pd.DataFrame(rr)


def load():
    key = ['unit', 'frame']
    a = recs(K4 / 'k28p3/ahd_test_instr')
    b = recs(K4 / 'k33/ahd_de_test')
    c = recs(K4 / 'r3low/main/ahd')
    for x in (a, b, c):
        assert x.unit.nunique() == 65 and not x.duplicated(key).any(), (x.unit.nunique(), len(x))
    for col in ('a_cheap', 'J_cheap', 'a_v6S|v3a', 'J_v6S|v3a'):          # the three runners agree on the shared rows
        for o in (b, c):
            m = a[key + [col]].merge(o[key + [col]], on=key, validate='1:1')
            assert len(m) == len(a) and np.allclose(m[col + '_x'].astype(float), m[col + '_y'].astype(float), atol=1e-8), col
    keep_b = [x for x in b.columns if 'ana05hinfi' in x]
    keep_c = [x for x in c.columns if 'v6S_b2A_retro' in x]
    d = a.merge(b[key + keep_b], on=key, validate='1:1').merge(c[key + keep_c], on=key, validate='1:1')
    d = d[d['gt'].astype(bool)].sort_values(key).reset_index(drop=True)
    g = pd.read_csv(B / 'results/k35/k0/av1_train_frames.csv')
    g['unit'] = g.sid.map(lambda s: f'hold{int(s):02d}')
    m = d[key + ['act_gt']].merge(g[['unit', 'frame', 'act_gt', 'a_gt']], on=key, how='outer', validate='1:1', indicator=True)
    assert (m._merge == 'both').all() and (m.act_gt_x == m.act_gt_y).all(), 'evaluated frames / act_gt differ from K35-0'
    d = d.merge(g[['unit', 'frame', 'a_gt']].rename(columns={'a_gt': 'a_gt_k0'}), on=key, validate='1:1')
    assert len(d) == 39265 and d.unit.nunique() == 65, (len(d), d.unit.nunique())
    return d


def per_unit(d, acol, level=1, tau2=3.5):
    out = []
    for u, g in d.groupby('unit', sort=True):
        gt, a, dem = g.act_gt.to_numpy(int), g[acol].to_numpy(int), g.a_gt_k0.to_numpy(float)
        rec = {'unit': u, 'frames': len(g), 'false': sum((gt[s:s + n] < level).all() for s, n in runs(a >= level))}
        for sev in ('all', 'below_hard', 'at_or_above_hard'):
            rec[sev + '_n'] = rec[sev + '_miss'] = 0
        for s, n in runs(gt >= level):
            miss = not (a[s:s + n] >= level).any()
            sev = 'at_or_above_hard' if dem[s:s + n].max() >= tau2 else 'below_hard'
            for k in ('all', sev):
                rec[k + '_n'] += 1; rec[k + '_miss'] += int(miss)
        out.append(rec)
    return pd.DataFrame(out).set_index('unit')


def main():
    d = load()
    units, C = bootstrap(d.unit)
    Jc = d.groupby('unit').J_cheap.sum().reindex(units).to_numpy()
    stat, rows, diffs, eps = {}, [], [], []
    for name, (ac, jc, pc) in ROWS.items():
        Ju = d.groupby('unit')[jc].sum().reindex(units).to_numpy()
        Pu = d.groupby('unit')[pc].sum().reindex(units).to_numpy()
        n = d.groupby('unit').size().reindex(units).to_numpy()
        s = {'G': ((Jc.sum() - Ju.sum()) / Jc.sum(), 1 - (C @ Ju) / (C @ Jc)),
             'px': (Pu.sum() / n.sum(), (C @ Pu) / (C @ n))}
        for lv, tag in ((1, ''), (2, '_hard')):
            pu = per_unit(d, ac, lv).reindex(units)
            s['false_per_1000' + tag] = (1e3 * pu.false.sum() / n.sum(), 1e3 * (C @ pu.false.to_numpy()) / (C @ n))
            for sev in (('all', 'below_hard', 'at_or_above_hard') if lv == 1 else ('all',)):
                num, den = pu[sev + '_miss'].to_numpy(float), pu[sev + '_n'].to_numpy(float)
                s[f'missed_rate{tag}_{sev}'] = (num.sum() / max(den.sum(), 1), (C @ num) / np.maximum(C @ den, 1e-12))
                eps.append({'row': name, 'level': lv, 'severity': sev, 'episodes': int(den.sum()), 'missed': int(num.sum())})
        if name in CAP:
            jk, pk = CAP[name]
            Jk = d.groupby('unit')[jk].sum().reindex(units).to_numpy()
            s['G_cap'] = ((Jc.sum() - Jk.sum()) / Jc.sum(), 1 - (C @ Jk) / (C @ Jc))
            s['px_cap'] = (d[pk].mean(), None)
        stat[name] = s
        r = {'row': name, 'units': len(units), 'frames': len(d)}
        for k, (pt, dist) in s.items():
            r[k] = pt
            if dist is not None:
                r[k + '_ci_lo'], r[k + '_ci_hi'] = interval(dist)['ci_lo'], interval(dist)['ci_hi']
        rows.append(r)
    for name in ROWS:
        if name == REF:
            continue
        for k in stat[name]:
            if k in stat[REF] and stat[name][k][1] is not None:
                iv = interval(stat[name][k][1] - stat[REF][k][1])
                diffs.append({'row': name, 'statistic': k, 'row_minus_v6S': stat[name][k][0] - stat[REF][k][0],
                              'ci_lo': iv['ci_lo'], 'ci_hi': iv['ci_hi'], 'excludes_0': bool(iv['ci_lo'] > 0 or iv['ci_hi'] < 0)})
    od = K4 / 'tables'; od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(od / 'k4_rows.csv', index=False)
    D = pd.DataFrame(diffs); D.to_csv(od / 'k4_diffs.csv', index=False)
    pd.DataFrame(eps).to_csv(od / 'k4_episodes.csv', index=False)
    g = D[(D.row == CAND) & (D.statistic == 'G')].iloc[0]
    mv, m6 = stat[CAND]['missed_rate_all'][0], stat[REF]['missed_rate_all'][0]
    dec = {'dG_v7a_minus_v6S': g.row_minus_v6S, 'dG_ci_lo': g.ci_lo, 'dG_ci_hi': g.ci_hi,
           'kill_dG_lower_bound_below_minus_0.01': bool(g.ci_lo < -0.01),
           'missed_rate_v7a': mv, 'missed_rate_v6S': m6, 'kill_missed_not_lower': bool(mv >= m6)}
    dec['v7a_verdict'] = 'negative (not proposed)' if dec['kill_dG_lower_bound_below_minus_0.01'] or dec['kill_missed_not_lower'] else 'passes (proposed extension)'
    (od / 'k4_decision.json').write_text(json.dumps(dec, indent=2) + '\n')
    print(pd.DataFrame(rows).round(4).to_string()); print(D.round(4).to_string()); print(dec)
    print('K35-4 ANALYSIS DONE')


if __name__ == '__main__':
    main()
