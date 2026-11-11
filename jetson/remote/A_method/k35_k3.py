"""K35-3 (post-hoc, test splits, run once): frame-level temporal escalation baselines at v6-S's mean network px.
Rows: (i) periodic: every k-th frame (frame index % k == phase) u960 + C1 + v3a, cheap otherwise; phase 0 is the row,
every phase 0..k-1 is reported (phase sensitivity); (ii) random: each frame escalated independently with p = 1/k,
5 seeds (0-4), each seed and the mean reported.
k = round((px_u960 - px_cheap) / (px_v6S - px_cheap)) on the evaluated frames (pixel match; an escalated frame runs the
whole frame only, an unescalated one the cheap draft only).
C1 history as in deployment: an unescalated frame produces no fine observation, so an escalated frame has a C1 chain
only if its two previous frames were also escalated (then its inputs equal the continuous-history row: stored
a_wf960+C1|v3a); otherwise C1 has no TTC (k35_k3_replay.py: TTC infinite, the rest of the K34 replay unchanged).
Sensitivity row (labelled, NOT the registered rule): the K28 / K33 router convention, continuous fine history on every
escalated frame (stored a_wf960+C1|v3a).
Costs: P.decision_cost with cheap's previous action (K28 convention). G = 1 - sum J / sum J_cheap; px = mean network
px over evaluated frames; missed rate and false episodes per 1,000 frames per K31 (k31_episodes.unit_stats).
G_cap: not meaningful here (an escalated frame's px exceeds v6-S's per-frame budget on every frame, so the cap would
remove every escalation and give G_cap = 0); reported as NA with this reason.
Intervals: paired unit bootstrap of v6-S - row, 2,000 draws, seed 0 (shared draws).
usage: k35_k3.py --dataset nusc|av2|ahd   -> results/k35/k3/<ds>/
"""
import argparse
import hashlib
import pickle

import numpy as np
import pandas as pd
from rap import planner as P
from k31_episodes import unit_stats
from k34_common import B, bootstrap, interval, load_base, ratio

OUT = B / 'results/k35/k3'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ds = ap.parse_args().dataset
    od = OUT / ds; od.mkdir(parents=True, exist_ok=True)
    d = load_base(ds, [])
    rr = []
    for f in sorted((B / 'results/k35/k3_replay' / ds).glob('*.pkl')):
        rr += pickle.load(f.open('rb'))['rows']
    r = pd.DataFrame(rr)
    d = d.merge(r, on=['unit', 'frame'], how='left', validate='one_to_one', indicator=True)
    assert (d._merge == 'both').all(), 'replay incomplete'
    assert (d.stored_a == d['a_wf960+C1|v3a']).all()
    par = d.parity.dropna()
    parity = {'frames_checked': int(len(par)), 'agree': int(par.astype(bool).sum())}
    pp, cp = P.PlannerParams(), P.CostParams()
    px_c, px_v, px_u = d.px_cheap.mean(), d.px_v6S.mean(), d['px_wf960+C1'].mean() if 'px_wf960+C1' in d else d.px_wf960.mean()
    k = int(round((px_u - px_c) / (px_v - px_c)))
    units, C = bootstrap(d.unit)
    ui = pd.Index(units).get_indexer(d.unit)
    gt, prev = d.a_gt.to_numpy(float), d.prev_cheap.to_numpy()
    fr = d.frame.to_numpy(int)
    Jc = d.J_cheap.to_numpy(float)
    Lc = np.bincount(ui, weights=Jc, minlength=len(units))

    def cost(a):
        return np.array([P.decision_cost(int(x), g, None if pd.isna(p_) else int(p_), pp, cp)['J'] for x, g, p_ in zip(a, gt, prev)])
    cont_a = d['a_wf960+C1|v3a'].to_numpy(int)
    noh_a = d['a_u960_nohist|v3a'].to_numpy(int)
    J_cont, J_noh = cost(cont_a), cost(noh_a)
    a_cheap = d.a_cheap.to_numpy(int)

    def evaluate(esc_frames, rule):
        """esc_frames: per row, the set test for frames t, t-1, t-2 -> per evaluated frame action / cost / px"""
        e0 = esc_frames(fr)
        if rule == 'deployment':
            chain = e0 & esc_frames(fr - 1) & esc_frames(fr - 2) & (fr >= 2)
            a = np.where(e0, np.where(chain, cont_a, noh_a), a_cheap)
            J = np.where(e0, np.where(chain, J_cont, J_noh), Jc)
        else:
            chain = e0.copy()
            a = np.where(e0, cont_a, a_cheap); J = np.where(e0, J_cont, Jc)
        px = np.where(e0, px_u, px_c)
        return a, J, px, e0, chain

    def stats(a):
        g = d.assign(_a=a)
        ne, nm, nf, nfr = (np.zeros(len(units)) for _ in range(4))
        for u, q in g.groupby('unit', sort=True):
            eps, false, n, _, _ = unit_stats(q, '_a', 1)
            i = units.searchsorted(u)
            ne[i], nm[i], nf[i], nfr[i] = len(eps), sum(x[2] for x in eps), false, n
        return ne, nm, nf, nfr
    v_a, v_J = d['a_v6S|v3a'].to_numpy(int), d['J_v6S|v3a'].to_numpy(float)
    ve, vm, vf, vfr = stats(v_a)
    vJ = np.bincount(ui, weights=v_J, minlength=len(units))
    rows = []

    def record(name, family, phase_or_seed, rule, a, J, px, e0, chain):
        ne, nm, nf, nfr = stats(a)
        Ju = np.bincount(ui, weights=J, minlength=len(units))
        G = 1 - J.sum() / Jc.sum()
        dG = ((C @ Ju) - (C @ vJ)) / (C @ Lc)                                  # G(v6-S) - G(row)
        dM = ratio(C @ vm, C @ ve) - ratio(C @ nm, C @ ne)
        dF = 1000 * (ratio(C @ vf, C @ vfr) - ratio(C @ nf, C @ nfr))
        rows.append({'dataset': ds, 'row': name, 'family': family, 'phase_or_seed': phase_or_seed, 'history_rule': rule, 'k': k,
                     'escalated_share': float(e0.mean()), 'px': float(px.mean()), 'px_v6S': float(px_v),
                     'G': G, 'G_cap': np.nan, 'G_cap_note': 'not meaningful: every escalated frame exceeds the per-frame cap',
                     'missed_rate': nm.sum() / ne.sum(), 'false_per_1000': 1000 * nf.sum() / nfr.sum(),
                     'episodes': int(ne.sum()), 'escalated_frames_with_C1_chain': int(chain.sum()),
                     'v6S_minus_row_G': float((Ju.sum() - vJ.sum()) / Jc.sum()), **{f'G_{k_}': v_ for k_, v_ in interval(dG).items() if k_ in ('ci_lo', 'ci_hi')},
                     'v6S_minus_row_missed': float(vm.sum() / ve.sum() - nm.sum() / ne.sum()),
                     **{f'missed_{k_}': v_ for k_, v_ in interval(dM).items() if k_ in ('ci_lo', 'ci_hi')},
                     'v6S_minus_row_false_per_1000': float(1000 * (vf.sum() / vfr.sum() - nf.sum() / nfr.sum())),
                     **{f'false_{k_}': v_ for k_, v_ in interval(dF).items() if k_ in ('ci_lo', 'ci_hi')}})
    for rule in ('deployment', 'continuous_sensitivity'):
        for ph in range(k):
            record(f'periodic k={k} phase {ph}', 'periodic', ph, rule, *evaluate(lambda f, ph=ph: (f % k) == ph, rule))
        for seed in range(5):
            # one Bernoulli(1/k) draw per (unit, frame index), fixed by the seed: the same frame always gets the same draw
            def esc(f, seed=seed):
                h = np.array([int(hashlib.sha256(f'{seed}|{u}|{x}'.encode()).hexdigest()[:12], 16) for u, x in zip(d.unit, f)], float)
                return (h / float(16 ** 12)) < 1.0 / k
            record(f'random rate 1/{k} seed {seed}', 'random', seed, rule, *evaluate(esc, rule))
    R = pd.DataFrame(rows)
    for rule in R.history_rule.unique():
        q = R[(R.family == 'random') & (R.history_rule == rule)]
        m = q.mean(numeric_only=True).to_dict()
        rows.append({**{c: m.get(c, np.nan) for c in R.columns if c in m}, 'dataset': ds, 'row': f'random rate 1/{k} mean of 5 seeds',
                     'family': 'random_mean', 'phase_or_seed': 'mean', 'history_rule': rule,
                     'G_cap_note': 'not meaningful: every escalated frame exceeds the per-frame cap'})
    R = pd.DataFrame(rows)
    R.to_csv(od / 'k3_rows.csv', index=False)
    pd.DataFrame([{'dataset': ds, 'k': k, 'px_cheap': px_c, 'px_v6S': px_v, 'px_u960': px_u, **parity,
                   'v6S_G': 1 - v_J.sum() / Jc.sum(), 'v6S_missed': vm.sum() / ve.sum(), 'v6S_false_per_1000': 1000 * vf.sum() / vfr.sum()}]).to_csv(od / 'k3_meta.csv', index=False)
    print('K35-3 DONE', ds, 'k', k, parity)


if __name__ == '__main__':
    main()
