"""CPU compare of two completed runs. Does not infer parity from aggregate G."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def equal(a,b):
    if isinstance(a,dict) or isinstance(b,dict):
        return isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
        return isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if a is None or b is None:return a is None and b is None
    aa,bb=np.asarray(a),np.asarray(b)
    if aa.shape!=bb.shape:return False
    if aa.dtype.kind in 'fc' and bb.dtype.kind in 'fc':return bool(np.array_equal(aa,bb,equal_nan=True))
    return bool(np.array_equal(aa,bb))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('reference',type=Path);ap.add_argument('candidate',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    a,b=[pd.read_pickle(p/'frames.pkl') for p in (args.reference,args.candidate)]
    keys=['unit','frame'];a=a.set_index(keys);b=b.set_index(keys)
    assert not a.index.duplicated().any() and not b.index.duplicated().any()
    assert set(a.index)==set(b.index),'Frame sets differ'
    b=b.reindex(a.index)
    prefixes=('a_','J_','px_','crops_','rounds_','n_crops_','parity_history_','parity_outputs_')
    cols=[c for c in a if c.startswith(prefixes)]
    assert set(cols)=={c for c in b if c.startswith(prefixes)},'Parity endpoints differ'
    rows=[]
    for c in cols:
        for key,x,y in zip(a.index,a[c],b[c]):
            rows.append(dict(unit=key[0],frame=key[1],endpoint=c,equal=equal(x,y)))
    out=pd.DataFrame(rows);args.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(args.output,index=False)
    info=dict(reference=str(args.reference),candidate=str(args.candidate),frames=len(a),endpoints=len(cols),mismatches=int((~out['equal']).sum()),history_recorded=any(c.startswith('parity_history_') for c in cols),raw_detection_parity='requires separate --columns_mode audit run; not implied by this file')
    args.output.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n');print(info)
    if info['mismatches']:raise SystemExit(1)


if __name__=='__main__':main()
