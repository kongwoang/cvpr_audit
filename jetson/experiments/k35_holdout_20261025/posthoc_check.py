"""K35-4 post-hoc test run checks: per unit, runner_t's recomputed cheap / v6-S (a, J, px) equal the stored K28 instr test
records (results/k28/p3/<ds>_test_instr); units 228 / 150 / 24. Output results/k35/k4/posthoc/check.json"""
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

B = Path(os.environ['B'])
K = B/'results/k35/k4/posthoc'
COLS = ['a_cheap', 'J_cheap', 'a_v6S|none', 'a_v6S|v3a', 'J_v6S|v3a', 'px_v6S']
N = {'nusc': 228, 'av2': 150, 'ahd': 24}


def main():
    rep = {}
    for ds, n in N.items():
        fs = sorted((K/'r3low'/ds).glob('*.pkl'))
        bad = {}
        for f in fs:
            a = pd.DataFrame(pickle.load(f.open('rb'))).sort_values('frame').reset_index(drop=True)
            b = pd.DataFrame(pickle.load((B/'results/k28/p3'/f'{ds}_test_instr'/f.name).open('rb'))).sort_values('frame').reset_index(drop=True)
            assert list(a.frame) == list(b.frame), f.stem
            for c in COLS:
                x, y = a[c].astype(float).to_numpy(), b[c].astype(float).to_numpy()
                k = int((~(np.abs(x - y) <= 1e-8)).sum())
                if k:
                    bad.setdefault(f.stem, {})[c] = k
        rep[ds] = {'units': len(fs), 'expected': n, 'units_with_differences': bad}
    rep['pass'] = all(r['units'] == r['expected'] and not r['units_with_differences'] for r in rep.values() if isinstance(r, dict))
    (K/'check.json').write_text(json.dumps(rep, indent=1)+'\n'); print(json.dumps(rep, indent=1)[:2000])


if __name__ == '__main__':
    main()
