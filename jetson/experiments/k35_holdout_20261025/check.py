"""K35-4 harness checks. Usage: check.py val|train
val:   the adapter reproduces the stored Argoverse-HD test records on seq05: every common a_* / J_* / px_* column of
       k28_p3_h (vs results/k28/p3/ahd_test_instr) and k33_eval_h (vs results/k33/ahd_de_test), and r3/runner_h's cheap /
       v6-S (vs the stored k28_p3 instr record).
train: per holdout unit, r3/runner_h's recomputed cheap / v6-S equal k28_p3_h's (same unit, same frames).
Output: results/k35/k4/<split>/check.json"""
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

B = Path(os.environ['B'])
K4 = B/'results/k35/k4'


def rows(p):
    return pd.DataFrame(pickle.load(open(p, 'rb'))).sort_values('frame').reset_index(drop=True)


def diff(a, b, cols=None):
    assert list(a.frame) == list(b.frame), 'frames differ'
    cols = cols or sorted(c for c in set(a.columns) & set(b.columns) if c.split('_')[0] in ('a', 'J', 'px') and not c.startswith('a_gt'))
    out = {}
    for c in cols:
        x, y = pd.to_numeric(a[c], errors='coerce').to_numpy(float), pd.to_numeric(b[c], errors='coerce').to_numpy(float)
        bad = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-8))
        out[c] = int(bad.sum())
    return out


def main():
    sp = sys.argv[1]
    rep = {'split': sp}
    base = ['a_cheap', 'J_cheap', 'a_v6S|none', 'a_v6S|v3a', 'J_v6S|v3a', 'px_v6S']
    if sp == 'val':
        u = 'seq05'
        rep['k28p3'] = diff(rows(K4/'val/k28p3/ahd_test_instr'/f'{u}.pkl'), rows(B/'results/k28/p3/ahd_test_instr'/f'{u}.pkl'))
        rep['k33'] = diff(rows(K4/'val/k33/ahd_de_test'/f'{u}.pkl'), rows(B/'results/k33/ahd_de_test'/f'{u}.pkl'))
        rep['r3low_vs_stored_k28p3'] = diff(rows(K4/'val/r3low/main/ahd'/f'{u}.pkl'), rows(B/'results/k28/p3/ahd_test_instr'/f'{u}.pkl'), base)
    else:
        rep['units'] = {}
        for f in sorted((K4/'train/r3low/main/ahd').glob('*.pkl')):
            rep['units'][f.stem] = diff(rows(f), rows(K4/'train/k28p3/ahd_test_instr'/f.name), base)
    rep['total_differences'] = int(sum(v for k, d in rep.items() if isinstance(d, dict) for v in (d.values() if k != 'units' else [x for dd in d.values() for x in dd.values()])))
    rep['n_columns_compared'] = {k: len(d) for k, d in rep.items() if isinstance(d, dict) and k != 'units'}
    (K4/sp).mkdir(parents=True, exist_ok=True)
    (K4/sp/'check.json').write_text(json.dumps(rep, indent=1)+'\n')
    print(json.dumps({k: v for k, v in rep.items() if k not in ('k28p3', 'k33')}, indent=1)[:3000])
    print('nonzero:', {k: {c: n for c, n in d.items() if n} for k, d in rep.items() if isinstance(d, dict) and k != 'units'})


if __name__ == '__main__':
    main()
