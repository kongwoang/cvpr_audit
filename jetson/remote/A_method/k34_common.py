"""K34 CPU utilities; frozen sources are read-only."""
import hashlib
import json
import os
import pickle
import socket
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

B = Path(os.environ['B'])
OUT = B / 'results/k34'
EXPECTED = {'nusc': (228, 9181), 'av2': (150, 11818), 'ahd': (24, 15006)}
NWS = {'nusc': 1600, 'av2': 2048, 'ahd': 1920}
DT = {'nusc': .5, 'av2': .2, 'ahd': 1/30}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_rows(folder, manifest):
    rows = []
    for f in sorted(Path(folder).glob('*.pkl')):
        manifest.append({'path': str(f), 'sha256': sha(f), 'bytes': f.stat().st_size})
        rows.extend(pickle.load(f.open('rb')))
    d = pd.DataFrame(rows).sort_values(['unit', 'frame']).reset_index(drop=True)
    assert not d.duplicated(['unit', 'frame']).any()
    return d

def load_base(ds, manifest):
    d = read_rows(B / 'results/k28/p3' / (ds + '_test_instr'), manifest)
    d['prev_cheap'] = d.groupby('unit').a_cheap.shift(1)
    f = B / 'results/k30/gt' / (ds + '.pkl')
    manifest.append({'path': str(f), 'sha256': sha(f), 'bytes': f.stat().st_size})
    gt = pd.read_pickle(f)
    assert not gt.duplicated(['unit', 'frame']).any()
    d = d.merge(gt[['unit','frame','a_gt','act_gt','v']], on=['unit','frame'], how='left',
                validate='one_to_one', suffixes=('', '_source'), indicator=True)
    d = d[d['gt'].astype(bool)].reset_index(drop=True)
    assert (d._merge == 'both').all()
    assert (d.act_gt == d.act_gt_source).all() and np.allclose(d.v, d.v_source)
    assert (d.unit.nunique(),len(d)) == EXPECTED[ds]
    return d.drop(columns=['_merge'])

def bootstrap(units):
    u = np.unique(units)
    draws = np.random.default_rng(0).integers(0, len(u), (2000,len(u)))
    return u, np.stack([np.bincount(x,minlength=len(u)) for x in draws]).astype(float)

def ratio(num, den):
    num, den = np.asarray(num,dtype=float), np.asarray(den,dtype=float)
    return np.divide(num,den,out=np.full(np.broadcast_shapes(num.shape,den.shape),np.nan),where=den!=0)

def interval(samples):
    good = np.asarray(samples)[np.isfinite(samples)]
    lo,hi = np.percentile(good,[2.5,97.5]) if len(good) else (np.nan,np.nan)
    return {'ci_lo':lo,'ci_hi':hi,'valid_draws':len(good),'bootstrap_n':2000,'seed':0}

def done(path, **extra):
    r = {'host':socket.gethostname(),'finished_at':datetime.now().astimezone().isoformat(),**extra}
    tmp = Path(str(path)+'.tmp'); tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(path)

def pipes(ds):
    return {'cheap':'a_cheap','u960 + C1 + v3a':'a_wf960+C1|v3a',
            'native + C1 + v3a':f'a_wf{NWS[ds]}+C1|v3a','hybrid tsp8@S + C1 + v3a':'a_tsp8@S+C1|v3a',
            'uncertainty unc2@S + C1 + v3a':'a_unc2@S+C1|v3a','v6-S':'a_v6S|v3a','v6-S without refiner':'a_v6S|none'}
