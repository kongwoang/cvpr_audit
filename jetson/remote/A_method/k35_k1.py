"""K35-1 (post-hoc, three test splits, stored K28 p3 / K30 match / K34 replay records; no pipeline run).
(a) Missed-episode decomposition: paired intervals of five registered contrasts, all episodes and both severity strata.
(b) Exclusive attribution of every required episode that cheap detects and v6-S misses (and the reverse).
Episodes, severity, onset capping and insufficient loss are K34-1's (k34_episodes.py) unchanged: K31 required episodes
(act_gt >= DECELERATE, runs over evaluated frames), severity = peak annotated demand < / >= thr_hard, loss = lam_risk x
max(required - commanded, 0)^2 summed over the episode. Paired unit bootstrap: 2,000 draws, seed 0, shared draws.
Decisive object and verified region: K34-2 / R3 rules unchanged (K30 match records: dec_in_crop_v6S = the decisive
object's box centre inside the union of the crops actually verified by the final round; dec_found_v6S / dec_ttc_v6S =
the v6-S scene object matched to it and its TTC; K34 replay: cheap_detected = a cheap box matches it).
usage: k35_k1.py --dataset nusc|av2|ahd   -> results/k35/k1/<ds>/
"""
import argparse
import json

import numpy as np
import pandas as pd
from rap import planner as P
from k31_episodes import runs
from k34_common import B, DT, NWS, bootstrap, interval, load_base, ratio, read_rows, sha

OUT = B / 'results/k35/k1'
SEV = ['below_hard', 'at_or_above_hard']


def pipes(ds):
    nw = NWS[ds]
    return {'cheap': 'a_cheap', 'v6-S': 'a_v6S|v3a', 'v6-S without refiner': 'a_v6S|none',
            'native + C1 + v3a': f'a_wf{nw}+C1|v3a', 'native + C1 without refiner': f'a_wf{nw}+C1|none',
            'u960 + C1 + v3a': 'a_wf960+C1|v3a', 'u960 + C1 without refiner': 'a_wf960+C1|none'}


CONTRASTS = [('v6-S', 'v6-S without refiner'), ('v6-S without refiner', 'native + C1 without refiner'),
             ('v6-S without refiner', 'u960 + C1 without refiner'), ('native + C1 + v3a', 'native + C1 without refiner'),
             ('v6-S without refiner', 'cheap')]


def episodes(d, ds):
    pp, cp = P.PlannerParams(), P.CostParams()
    cmds = np.array([P.commanded_decel(a, pp) for a in range(3)])
    rows = []
    for u, g in d.groupby('unit', sort=True):
        gt = g.a_gt.to_numpy(float)
        for ei, (s, l) in enumerate(runs(g.act_gt.to_numpy(int) >= 1)):
            peak = float(gt[s:s + l].max())
            base = {'dataset': ds, 'unit': u, 'episode': ei, 'start_frame': int(g.frame.iloc[s]), 'end_frame': int(g.frame.iloc[s + l - 1]),
                    'episode_frames': l, 'peak_required_mps2': peak, 'severity': SEV[int(peak >= pp.thr_hard)]}
            for name, col in pipes(ds).items():
                a = g[col].to_numpy(int)[s:s + l]
                hit = np.flatnonzero(a >= 1)
                rows.append({**base, 'pipeline': name, 'missed': not len(hit), 'delay_frames': int(hit[0]) if len(hit) else l,
                             'insufficient_loss': float((cp.lam_risk * np.maximum(0, gt[s:s + l] - cmds[a]) ** 2).sum())})
    return pd.DataFrame(rows)


def part_a(d, e, ds, od):
    units, C = bootstrap(d.unit)
    L = d.groupby('unit').J_cheap.sum().reindex(units).to_numpy()
    summ, ints = [], []
    for label, mask in [('all', np.ones(len(e), bool))] + [(s, (e.severity == s).to_numpy()) for s in SEV]:
        num, den = {}, {}
        for name in pipes(ds):
            q = e[mask & (e.pipeline == name).to_numpy()]
            g = q.groupby('unit')
            ne = g.size().reindex(units, fill_value=0).to_numpy(float)
            agg = g[['missed', 'delay_frames', 'insufficient_loss']].sum().reindex(units, fill_value=0)
            num[name] = {'missed_rate': agg.missed.to_numpy(float), 'mean_onset_delay_s': DT[ds] * agg.delay_frames.to_numpy(float),
                         'loss_share_cheap_total': agg.insufficient_loss.to_numpy(float)}
            den[name] = {'missed_rate': ne, 'mean_onset_delay_s': ne, 'loss_share_cheap_total': L}
            for m in num[name]:
                summ.append({'dataset': ds, 'stratum': label, 'pipeline': name, 'metric': m, 'episodes': int(ne.sum()),
                             'value': float(ratio(num[name][m].sum(), den[name][m].sum())),
                             **interval(ratio(C @ num[name][m], C @ den[name][m]))})
        for a, b in CONTRASTS:
            for m in num[a]:
                pa, pb = ratio(num[a][m].sum(), den[a][m].sum()), ratio(num[b][m].sum(), den[b][m].sum())
                dist = ratio(C @ num[a][m], C @ den[a][m]) - ratio(C @ num[b][m], C @ den[b][m])
                iv = interval(dist)
                ints.append({'dataset': ds, 'stratum': label, 'contrast': f'{a} - {b}', 'metric': m, 'a_value': float(pa),
                             'b_value': float(pb), 'diff': float(pa - pb), **iv,
                             'excludes_0': bool(iv['ci_lo'] > 0 or iv['ci_hi'] < 0)})
    pd.DataFrame(summ).to_csv(od / 'k1a_summary.csv', index=False)
    pd.DataFrame(ints).to_csv(od / 'k1a_contrasts.csv', index=False)


def part_b(d, e, ds, od, manifest):
    m = read_rows(B / 'results/k30/match' / ds, manifest).set_index(['unit', 'frame'])
    ann = []
    for f in sorted((B / 'results/k34/replay' / ds).glob('*.pkl')):
        import pickle
        ann += pickle.load(f.open('rb'))['annotations']
        manifest.append({'path': str(f), 'sha256': sha(f)})
    ann = pd.DataFrame(ann).set_index(['unit', 'frame'])
    fr = d.set_index(['unit', 'frame'])
    w = e.pivot_table(index=['unit', 'episode', 'start_frame', 'end_frame', 'severity'], columns='pipeline',
                      values=['missed', 'insufficient_loss'], aggfunc='first')
    rows = []
    for (u, ei, s, t, sev), r in w.iterrows():
        cm, vm = bool(r[('missed', 'cheap')]), bool(r[('missed', 'v6-S')])
        if cm == vm:
            continue
        g = fr.loc[u]
        g = g[(g.index >= s) & (g.index <= t)]
        direction = 'cheap_detects_v6S_misses' if not cm else 'v6S_detects_cheap_misses'
        fire = (g.a_cheap >= 1) if direction.startswith('cheap') else (g['a_v6S|v3a'] >= 1)
        F = list(g.index[fire.to_numpy()])
        mm = m.loc[[(u, f) for f in F]]
        aa = ann.reindex([(u, f) for f in F])
        inside = mm.dec_in_crop_v6S.fillna(False).astype(bool).to_numpy()
        found = mm.dec_found_v6S.fillna(False).astype(bool).to_numpy()
        fin = found & np.isfinite(pd.to_numeric(mm.dec_ttc_v6S, errors='coerce').to_numpy(float))
        noref = (g.loc[F, 'a_v6S|none'] >= 1).to_numpy()
        cdet = aa.cheap_detected.fillna(False).astype(bool).to_numpy()
        fromc = mm.dec_from_crop_v6S.fillna(False).astype(bool).to_numpy()
        act_diff = int((mm['a_v6S|v3a'].to_numpy() != g.loc[F, 'a_v6S|v3a'].to_numpy()).sum())
        if direction.startswith('cheap'):
            if not inside.any():
                cls, sub = 'E1', 'outside_verified_region'
            elif not (inside & fin).any():
                cls, sub = 'E2', ('inside_matched_no_finite_ttc' if (inside & found).any() else 'inside_not_matched')
            elif (inside & fin & noref).any():
                cls, sub = 'E3', 'refiner_lowers_demand_below_tau1'
            elif not cdet.any():
                cls, sub = 'E6', 'cheap_fires_without_decisive_object_box'
            else:
                cls, sub = 'E-other', 'inside_finite_ttc_refiner_not_decisive'
        else:
            if not found.any():
                cls, sub = 'R6', 'v6S_fires_without_decisive_object_match'
            elif (found & fromc & ~cdet).any():
                cls, sub = 'R1', 'decisive_object_only_in_verified_crop'
            elif (found & cdet).any():
                cls, sub = 'R2', 'both_see_it_v6S_estimate_brakes'
            else:
                cls, sub = 'R-other', 'other'
        rows.append({'dataset': ds, 'unit': u, 'episode': ei, 'start_frame': s, 'end_frame': t, 'severity': sev,
                     'direction': direction, 'class': cls, 'subclass': sub, 'firing_frames': len(F),
                     'firing_frames_inside': int(inside.sum()), 'firing_frames_finite_ttc': int((inside & fin).sum()),
                     'firing_frames_noref_brakes': int(noref.sum()), 'firing_frames_cheap_box': int(cdet.sum()),
                     'match_action_differs_K28': act_diff,
                     'insufficient_loss_v6S': float(r[('insufficient_loss', 'v6-S')]),
                     'insufficient_loss_cheap': float(r[('insufficient_loss', 'cheap')])})
    a = pd.DataFrame(rows)
    a.to_csv(od / 'k1b_episodes.csv', index=False)
    out = []
    for direction, q in a.groupby('direction'):
        for sev in ['all'] + SEV:
            qq = q if sev == 'all' else q[q.severity == sev]
            tot_v, tot_c = qq.insufficient_loss_v6S.sum(), qq.insufficient_loss_cheap.sum()
            for (cls, sub), z in qq.groupby(['class', 'subclass']):
                out.append({'dataset': ds, 'direction': direction, 'stratum': sev, 'class': cls, 'subclass': sub,
                            'episodes': len(z), 'share_of_direction_episodes': len(z) / max(len(qq), 1),
                            'insufficient_loss_v6S': z.insufficient_loss_v6S.sum(),
                            'loss_share_v6S': z.insufficient_loss_v6S.sum() / tot_v if tot_v > 0 else np.nan,
                            'insufficient_loss_cheap': z.insufficient_loss_cheap.sum(),
                            'loss_share_cheap': z.insufficient_loss_cheap.sum() / tot_c if tot_c > 0 else np.nan})
    pd.DataFrame(out).to_csv(od / 'k1b_classes.csv', index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ds = ap.parse_args().dataset
    od = OUT / ds; od.mkdir(parents=True, exist_ok=True)
    manifest = []
    d = load_base(ds, manifest)
    e = episodes(d, ds)
    e.to_csv(od / 'k1_episodes.csv', index=False)
    k34 = pd.read_csv(B / 'results/k34/episodes' / ds / 'episodes.csv')            # reproduction check vs K34-1
    shared = sorted(set(k34.pipeline) & set(e.pipeline))
    j = e[e.pipeline.isin(shared)].merge(k34[k34.pipeline.isin(shared)], on=['unit', 'episode', 'pipeline'], suffixes=('', '_k34'))
    check = {'pipelines': shared, 'rows': len(j), 'rows_k35': int(e.pipeline.isin(shared).sum()),
             'missed_mismatch': int((j.missed != j.missed_k34).sum()),
             'loss_max_abs_diff': float((j.insufficient_loss - j.insufficient_loss_k34).abs().max())}
    json.dump(check, open(od / 'k34_reproduction.json', 'w'), indent=1)
    assert check['rows'] == check['rows_k35'] and check['missed_mismatch'] == 0 and check['loss_max_abs_diff'] < 1e-9, check
    part_a(d, e, ds, od)
    part_b(d, e, ds, od, manifest)
    json.dump(manifest, open(od / 'manifest.json', 'w'), indent=1)
    print('K35-1 DONE', ds, len(e) // len(pipes(ds)), 'episodes')


if __name__ == '__main__':
    main()
