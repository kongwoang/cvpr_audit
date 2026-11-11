#!/usr/bin/env python3
"""K40-1(a,b), K40-2: CPU-only registered analysis; explicit non-holdout sources."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

B = Path(os.environ['B'])
OUT = B / 'results/k40r/cpu'
R = B / 'tmp/desp_k40/artifacts/records'
sys.path.insert(0, str(B / 'results/k40r/vendor'))
from analysis.common import Bootstrap, DATASETS, NATIVE, episodes
from analysis.tab_loss import terms
from analysis.tab_main import rows as main_rows
from analysis.tab_constrained import choice, rows as constrained_rows
from analysis.tab_curves import sfx, GRID


def read(ds):
    assert ds in DATASETS or ds in ('nuscenes_dev_thresholds', 'av2_dev_thresholds')
    d = pd.read_parquet(R / (ds + '.parquet')).sort_values(['unit', 'frame']).reset_index(drop=True)
    assert not d.duplicated(['unit', 'frame']).any()
    return d


def interval(bt, num, den):
    """Ratio using the original Bootstrap's identical cluster draw matrix."""
    n, d = bt.sums(num), bt.sums(den)
    denom = d[bt.idx].sum(1)
    z = n[bt.idx].sum(1)[denom > 0] / denom[denom > 0]
    if not d.sum() or not len(z):
        return np.nan, np.nan, np.nan, len(z)
    return float(n.sum()/d.sum()), *np.percentile(z, [2.5, 97.5]).tolist(), len(z)


def delta(bt, L, ours, row):
    pt, lo, hi = bt.delta(L, ours, row)
    n, d = bt.sums(row-ours), bt.sums(L)
    dist = n[bt.idx].sum(1)/d[bt.idx].sum(1)
    p = min(1., 2*min(np.mean(dist <= 0), np.mean(dist >= 0)))
    pc = min(1., 2*(1+min(np.sum(dist <= 0), np.sum(dist >= 0)))/(1+len(dist)))
    return dict(delta_G=pt, delta_lo=lo, delta_hi=hi, p_raw=p, p_plus1=pc)


def action_column(d, j):
    a = j.replace('J_', 'a_', 1)
    if a in d:
        return d[a].to_numpy(int), a
    key = j[2:]
    szc = 'size_' + key
    if szc in d:
        sz = d[szc].to_numpy(int)
        out = d.a_cheap.to_numpy(int).copy()
        for s in np.unique(sz):
            if not s:
                continue
            # K33 rc uses the K33 pub whole-frame action/loss, not the plain K28 row.
            ac = next((c for c in (f'tab1.a_wf{s}|v3a', f'a_wf{s}+C1|v3a', f'a_wf{s}|v3a') if c in d), None)
            assert ac, (j, s)
            out[sz == s] = d[ac].to_numpy(int)[sz == s]
        return out, 'derived from ' + szc + ' + K33 pub actions; gated by exact J'
    raise KeyError(j)


def specs(ds):
    out = []
    for lab, j, px, jc in main_rows(ds):
        out.append((lab, 'uncapped', j, px))
        if jc and jc != j:
            pc = jc.split('J_', 1)[1].split('|')[0]
            out.append((lab, 'nominal_cap', jc, 'px_' + pc))
    for s in (512, 640):
        out.append((f'u{s} + comp.', 'uncapped', f'J_wf{s}+C1|v3a', f'px_wf{s}'))
    out += [('DESP no refiner', 'uncapped', 'J_v6S|none', 'px_v6S'),
            ('target pi(F), no refiner', 'uncapped', 'J_wf960+C1|none', 'px_wf960'),
            ('cheap', 'uncapped', 'J_cheap', 'px_cheap')]
    return list(dict.fromkeys(out))


def holm(vals):
    p = np.asarray(vals, float); order = np.argsort(p, kind='stable')
    out = np.empty(len(p)); out[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out


def safety(d, a, level, bt):
    x = d[['unit', 'frame', 'act_gt']].copy(); x['_a'] = a
    unit = []
    for u, g in x.groupby('unit', sort=True):
        e = episodes(g, '_a', level)
        req, pred = g.act_gt.to_numpy(int) >= level, g._a.to_numpy(int) >= level
        e.update(unit=u, required_frames=int(req.sum()), missed_frames=int((req & ~pred).sum()),
                 false_frames=int((~req & pred).sum()), nonrequired_frames=int((~req).sum()))
        unit.append(e)
    tab = pd.DataFrame(unit)
    ub = Bootstrap(tab.unit)
    assert np.array_equal(ub.idx, bt.idx)
    metrics = [('missed_frame_rate', 'missed_frames', 'required_frames', 1),
               ('false_frame_rate', 'false_frames', 'nonrequired_frames', 1),
               ('missed_episode_rate', 'missed', 'episodes', 1),
               ('false_episodes_per_1000_frames', 'false', 'frames', 1000)]
    res = {}
    for name, n, den, scale in metrics:
        pt, lo, hi, valid = interval(ub, tab[n].to_numpy()*scale, tab[den].to_numpy())
        res[name] = pt; res[name+'_lo'] = lo; res[name+'_hi'] = hi; res[name+'_valid_draws'] = valid
    for c in ['episodes', 'missed', 'false', 'frames', 'required_frames', 'missed_frames', 'false_frames', 'nonrequired_frames']:
        res[c] = int(tab[c].sum())
    return res, tab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets, decomposition, weights, checks, safe, safe_delta, sweeps, selected = [], [], [], [], [], [], [], []
    choices = choice(R)  # choice() only reads the two allowed dev files; NEVER call cells(), which includes holdout.
    (OUT/'dev_threshold_choices.json').write_text(json.dumps({k:v for k,v in choices.items() if k in DATASETS}, indent=2)+'\n')
    for ds in DATASETS:
        d = read(ds); bt = Bootstrap(d.unit); L = d.J_cheap.to_numpy(float)
        meta = dict(dataset=ds, frames=len(d), units=d.unit.nunique(), analysis='pre-specified post-hoc')
        T = {}
        for lab, family, j, px in specs(ds):
            if j in T:
                continue
            a, provenance = action_column(d, j)
            t = terms(d, a)
            err = np.abs(t[:, 4]-d[j].to_numpy(float))
            checks.append(dict(dataset=ds, column=j, frames=len(d), max_abs_J_error=float(err.max()), action_source=provenance))
            assert err.max() < 1e-9, checks[-1]
            T[j] = (t, a)
        f = 'J_wf960+C1|none'; tf, af = T[f]
        for j in [f, 'J_v6S|none', 'J_v6S|v3a']:
            t, a = T[j]
            g, lo, hi, valid = interval(bt, L-t[:,4], L)
            targets.append(dict(meta, row=j, G=g, G_lo=lo, G_hi=hi, agreement=float(np.mean(a == af)),
                                **delta(bt, L, t[:,4], tf[:,4])))
            if j == f:
                continue
            for group, m in [('agree', a == af), ('disagree', a != af)]:
                for k, term in enumerate(['insufficient', 'excessive', 'command', 'collision', 'total']):
                    pt, lo, hi, valid = interval(bt, np.where(m, tf[:,k]-t[:,k], 0.), L)
                    decomposition.append(dict(meta, row=j, group=group, term=term, group_frames=int(m.sum()),
                        contribution_to_delta_G=pt, lo=lo, hi=hi, valid_draws=valid,
                        previous_action='shared prev_cheap', temporal_carryover_effect=0.,
                        max_abs_difference_on_group=float(np.abs(tf[m,k]-t[m,k]).max()) if m.any() else np.nan))
        safety_units = {}
        for lab, family, j, px in specs(ds):
            t, a = T[j]
            assert px in d, (ds, j, px)
            for w in (1,2,5,10):
                J = t[:,4] + (w-1)*t[:,0]
                Lw = T['J_cheap'][0][:,4] + (w-1)*T['J_cheap'][0][:,0]
                ow = T['J_v6S|v3a'][0]; Jo = ow[:,4] + (w-1)*ow[:,0]
                g, lo, hi, _ = interval(bt, Lw-J, Lw)
                p, plo, phi, _ = interval(bt, d[px].to_numpy(float), np.ones(len(d)))
                weights.append(dict(meta, label=lab, family=family, row=j, weight_insufficient=w,
                    G=g, G_lo=lo, G_hi=hi, px_mean=p, px_mean_lo=plo, px_mean_hi=phi,
                    px_p95=float(d[px].quantile(.95)), **delta(bt, Lw, Jo, J)))
            for level in (1,2):
                sm, units = safety(d, a, level, bt)
                safe.append(dict(meta, label=lab, family=family, row=j, level=level, **sm))
                safety_units[(j,level)] = units
        for (j,lev), u in safety_units.items():
            ours = safety_units[('J_v6S|v3a',lev)]
            ub = Bootstrap(u.unit)
            for name,n,den,sc in [('missed_frame_rate','missed_frames','required_frames',1),
                    ('false_frame_rate','false_frames','nonrequired_frames',1),
                    ('missed_episode_rate','missed','episodes',1),('false_episodes_per_1000_frames','false','frames',1000)]:
                assert np.array_equal(u[den], ours[den])
                pt,lo,hi,valid=interval(ub, (ours[n]-u[n]).to_numpy()*sc,u[den].to_numpy())
                safe_delta.append(dict(meta,row=j,level=lev,metric=name,delta=pt,lo=lo,hi=hi,valid_draws=valid))
        # Existing threshold choices and full descriptive sweep. No outcome-based selection.
        for (row,j,px),(chosen,infeasible) in zip(constrained_rows(ds), choices[ds]):
            for threshold in GRID:
                ac = 'a_'+row+sfx(threshold)
                if ac not in d:
                    continue
                sm,_=safety(d,d[ac].to_numpy(int),1,bt)
                jc = j+sfx(threshold); pc=px+sfx(threshold)
                g,glo,ghi,_=interval(bt,L-d[jc].to_numpy(float),L)
                entry=dict(meta,row=row,threshold=threshold,dev_selected=threshold==chosen,dev_constraint_infeasible=infeasible,
                    G=g,G_lo=glo,G_hi=ghi,px_mean=float(d[pc if pc in d else px].mean()),**sm)
                sweeps.append(entry)
                if threshold==chosen:
                    selected.append(entry)
        print(ds,'loss parity rows',len(T),'complete',flush=True)
    W=pd.DataFrame(weights)
    for _,idx in W.groupby(['family','weight_insufficient']).groups.items():
        W.loc[idx,'p_holm_family']=holm(W.loc[idx,'p_raw'])
    W['rank_G']=W.groupby(['dataset','family','weight_insufficient']).G.rank(method='min',ascending=False)
    base=W[W.weight_insufficient==1][['dataset','family','row','delta_G','delta_lo','delta_hi']].rename(
        columns={'delta_G':'base_delta','delta_lo':'base_lo','delta_hi':'base_hi'})
    W=W.merge(base,on=['dataset','family','row'],validate='m:1')
    W['point_sign_reversal']=W.delta_G*W.base_delta<0
    W['significant_sign_reversal']=((W.delta_lo>0)&(W.base_hi<0))|((W.delta_hi<0)&(W.base_lo>0))
    for file,data in [('k40_1_target',targets),('k40_1_decomposition',decomposition),('k40_2_weights',W),
        ('k40_2_safety',safe),('k40_2_safety_deltas',safe_delta),('k40_2_threshold_sweep',sweeps),
        ('k40_2_dev_selected',selected),('loss_parity',checks)]:
        pd.DataFrame(data).to_csv(OUT/(file+'.csv'),index=False)
    # Complete observed deployed-point Pareto sets (no thresholds selected on test).
    S=pd.DataFrame(safe); S=S[S.level==1].merge(W[W.weight_insufficient==1][['dataset','family','row','G','px_mean']],on=['dataset','family','row'])
    for _,ix in S.groupby(['dataset','family']).groups.items():
        for i in ix:
            g=S.loc[ix]; r=S.loc[i]
            S.loc[i,'nondominated_safety']=not (((g.missed_episode_rate<=r.missed_episode_rate)&(g.false_episodes_per_1000_frames<=r.false_episodes_per_1000_frames))&((g.missed_episode_rate<r.missed_episode_rate)|(g.false_episodes_per_1000_frames<r.false_episodes_per_1000_frames))).any()
            S.loc[i,'nondominated_G_px']=not (((g.G>=r.G)&(g.px_mean<=r.px_mean))&((g.G>r.G)|(g.px_mean<r.px_mean))).any()
    S.to_csv(OUT/'k40_2_pareto.csv',index=False)
    source=Path(__file__)
    (OUT/'run.json').write_text(json.dumps(dict(script=str(source),md5=hashlib.md5(source.read_bytes()).hexdigest(),
        holdout_read=False,bootstrap_draws=2000,seed=0,rows=len(W)),indent=2)+'\n')


if __name__=='__main__':
    main()
