"""Port parity with recovery disabled; enabled causal/budget checks on every frame."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--reference',type=Path);ap.add_argument('--disabled',action='store_true')
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    d=pd.read_pickle(a.run/'frames.pkl').set_index(['unit','frame']).sort_index()
    assert not d.index.duplicated().any()
    variants=('ours','ours_retro','ours_b2A','ours_b2A_retro')
    for name in variants:
        if name.endswith('_retro'):
            c=d[f'retro_calls_{name}@trt32'];success=d[f'retro_success_{name}@trt32']
            assert c.between(0,2).all() and success.between(0,1).all()
            assert ((success==0)|(c==2)).all()
            assert (d[f'retro_pixels_{name}@trt32']>=0).all()
            if a.disabled:assert (c==0).all()
    if a.disabled:
        for base,new in [('ours','ours_retro'),('ours_b2A','ours_b2A_retro')]:
            for pre in ('a_','J_','px_','calls_'):
                assert np.array_equal(d[pre+base+'@trt32'],d[pre+new+'@trt32']),(pre,base)
        ref=pd.read_pickle(a.reference/'frames.pkl').set_index(['unit','frame']).sort_index()
        assert d.index.equals(ref.index)
        for base in ('ours','ours_b2A'):
            for pre in ('a_','J_','px_','calls_'):
                assert np.array_equal(d[pre+base+'@trt32'],ref[pre+base+'@trt32']),(pre,base,'prior reference')
    result=dict(status='PASS',frames=len(d),disabled=a.disabled,
        recovery_attempts={name:int(d[f'retro_attempted_{name}@trt32'].sum()) for name in variants if name.endswith('_retro')},
        recovery_successes={name:int(d[f'retro_success_{name}@trt32'].sum()) for name in variants if name.endswith('_retro')})
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
