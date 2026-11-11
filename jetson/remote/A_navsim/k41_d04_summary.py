"""D04 summary (K41-1 PiP on NAVSIM, real crops): PDMS (P01 protocol, official NAVSIM v1.1), target agreement / repair vs
the full-perception selector choice, field - row, crop cost from results/k41/pip_nav; paired log bootstrap (2,000, seed 0)."""
import json, os, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
B = Path(os.environ['B']); sys.path.insert(0, str(B / 'results/k40r/vendor'))
from analysis.common import Bootstrap
src = B / 'results/k41/selector_metrics'; out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
NAMES = ['cheap', 'full', 'field', 'oracle', 'hybrid', 'all', 'selected', 'selected_support', 'geometric']
rows = []
for p in sorted(src.glob('*.pkl')):
    for token, r in pickle.load(p.open('rb')).items():
        for n in NAMES:
            rows.append(dict(log=r['log'], token=token, row=n, choice=r['selected'][n], **r['scores'][n]))
data = pd.DataFrame(rows); assert data.log.nunique() == 136 and len(data) == 12146 * len(NAMES)
cost = []
pf = 0
for p in sorted((B / 'results/k41/pip_nav').glob('*.pkl')):
    for r in pickle.load(p.open('rb')):
        assert 'err' not in r; pf += r['field_parity']['pi'] and r['field_parity']['px'] and r['field_parity']['rounds']
        for n, s in r['stats'].items():
            cost.append(dict(log=r['log'], token=r['token'], row=n, choice_nav=r['selected'][n], **{k: s.get(k, False if k == 'capped' else np.nan) for k in ('px', 'n_crops', 'rounds', 'calls', 'det_seconds', 'wall_seconds', 'capped')}))
cost = pd.DataFrame(cost); assert len(cost) == 12146 * 5
m = data.merge(cost[['log', 'token', 'row', 'choice_nav']], on=['log', 'token', 'row'], how='left')
mm = m[m.choice_nav.notna()]; consist = int((mm.choice == mm.choice_nav).sum()), len(mm)
pv = data.pivot(index=['log', 'token'], columns='row'); bt = Bootstrap(pv.index.get_level_values('log'))
cp = cost.pivot(index=['log', 'token'], columns='row').reindex(pv.index)
def iv(num, den):
    n, d = bt.sums(num), bt.sums(den); bn, bd = n[bt.idx].sum(1), d[bt.idx].sum(1); z = bn[bd > 0] / bd[bd > 0]
    lo, hi = np.percentile(z, [2.5, 97.5]); return float(n.sum() / d.sum()), float(lo), float(hi)
ch = pv['choice']; one = np.ones(len(ch)); bad = (ch['cheap'] != ch['full']).to_numpy().astype(float)
res = []
for n in NAMES:
    s = pv['score'][n].to_numpy(); agree = (ch[n] == ch['full']).to_numpy().astype(float)
    d = dict(row=n, pdms=iv(s, one), field_minus_row=iv(pv['score']['field'].to_numpy() - s, one),
             row_minus_cheap=iv(s - pv['score']['cheap'].to_numpy(), one), target_agreement=iv(agree, one),
             repair=iv(agree * bad, bad), pdms_on_disagree=iv(s * bad, bad))
    if n in cp['px'].columns:
        for k in ('px', 'n_crops', 'rounds', 'calls', 'det_seconds', 'wall_seconds'):
            d[k] = iv(cp[k][n].to_numpy().astype(float), one)
        d['capped_share'] = float(cp['capped'][n].astype(float).mean())
    res.append(d)
flat = [{'row': d['row'], **{f'{k}{suf}': (v[i] if isinstance(v, tuple) else v) for k, v in d.items() if k != 'row'
         for i, suf in ((0, ''), (1, '_lo'), (2, '_hi')) if isinstance(v, tuple) or suf == ''}} for d in res]
pd.DataFrame(flat).to_csv(out / 'k41_1_d04_summary.csv', index=False)
data.to_csv(out / 'k41_1_d04_frames.csv.gz', index=False); cost.to_csv(out / 'k41_1_d04_cost.csv.gz', index=False)
json.dump(dict(tokens=12146, logs=136, field_parity_tokens=int(pf), selector_nav_choice_consistency=consist, disagree_tokens=int(bad.sum()),
               bootstrap='paired by log, 2000 draws, seed 0 (analysis.common.Bootstrap)'), open(out / 'run.json', 'w'), indent=2)
print('field parity', pf, '/ 12146; nav-vs-P01 choice consistency', consist, '; cheap!=full tokens', int(bad.sum()))
for d in res:
    f = lambda t: '%.4f [%.4f,%.4f]' % t
    print('%-16s PDMS %s  field-row %s  agree %s  repair %s' % (d['row'], f(d['pdms']), f(d['field_minus_row']), f(d['target_agreement']), f(d['repair'])),
          ('px %.3f crops %.2f rounds %.2f det_s %.2f capped %.3f' % (d['px'][0], d['n_crops'][0], d['rounds'][0], d['det_seconds'][0], d['capped_share'])) if 'px' in d else '')
