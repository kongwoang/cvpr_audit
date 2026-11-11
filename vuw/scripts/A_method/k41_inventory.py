#!/usr/bin/env python3
"""Inventory original cached schemas where sanitized artifacts lack object/state fields."""
import hashlib
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

B=Path(os.environ['B'])
OUT=B/'results/k41/inventory'


def shape(x, depth=0):
    if depth>3:
        return type(x).__name__
    if isinstance(x, pd.DataFrame):
        return dict(type='DataFrame', rows=len(x), columns=list(x.columns),
                    object_columns={k:shape(x[k].iloc[0],depth+1) for k in x if x[k].dtype==object} if len(x) else {})
    if isinstance(x, dict):
        return {str(k):shape(v,depth+1) for k,v in list(x.items())[:250]}
    if isinstance(x, (list,tuple)):
        return dict(type=type(x).__name__,length=len(x),first=shape(x[0],depth+1) if len(x) else None)
    if isinstance(x, np.ndarray):
        return dict(type='ndarray',shape=list(x.shape),dtype=str(x.dtype))
    return type(x).__name__


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[]
    for ds in ('nusc','av2','ahd'):
        for pat in [f'results/k28/p3/{ds}_test_instr/*.pkl',f'results/k28/p3/{ds}_test_instr_capture/*.pkl',
                    f'results/k30/match/{ds}/*.pkl',f'results/k30/apdump/{ds}/*.pkl',
                    f'results/k33/{ds}_pub_dev/*.pkl',f'results/k33/{ds}_pub_test/*.pkl',
                    f'results/k39/k4/{ds}/*.pkl',f'results/k37/k1/test/{ds}/*.pkl']:
            fs=sorted(B.glob(pat))
            if fs:
                paths.append((pat,len(fs),fs[0]))
    for pat in ['results/k31/p2/*.pkl','results/k31/p2_pairs/*.pkl','results/k20/pairs__*.pkl',
                'results/k20/frozen*.pkl','results/k39/k5/*.pkl','results/navsim_exp/field_test/*.pkl',
                'results/navsim_exp/pdms_final/*.pkl','results/navsim_exp/dets_k30b/*.pkl']:
        fs=sorted(B.glob(pat))
        if fs:
            paths.append((pat,len(fs),fs[0]))
    report=[]
    for pat,n,f in paths:
        assert 'holdout' not in str(f).lower()
        with f.open('rb') as fh:
            x=pickle.load(fh)
        report.append(dict(pattern=pat,files=n,sample=str(f),bytes=f.stat().st_size,schema=shape(x)))
    for nm in ['coverage_frames.csv','strata_units.csv']:
        f=B/'tmp/desp_k40/artifacts/records'/nm
        x=pd.read_csv(f,nrows=0)
        report.append(dict(sample=str(f),columns=list(x.columns)))
    (OUT/'original_schemas.json').write_text(json.dumps(report,indent=2)+'\n')
    print('\n'.join(str({k:v for k,v in r.items() if k!='schema'}) for r in report))


if __name__=='__main__':
    main()
