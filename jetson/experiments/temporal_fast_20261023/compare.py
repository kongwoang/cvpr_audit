"""Full-frame decision and execution parity for independently run runtime paths."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def compare(reference,staged):
    a=pd.read_pickle(reference/'frames.pkl').set_index(['unit','frame']).sort_index()
    b=pd.read_pickle(staged/'frames.pkl').set_index(['unit','frame']).sort_index()
    assert a.index.equals(b.index) and not a.index.duplicated().any()
    prefixes=('a_','J_','px_','calls_','crop_engine_exec_','draft_calls_','n_refined_',
              'n_crops_','rounds_','calls_loop_','bound_hit_','retro_calls_','retro_pixels_','retro_attempted_','retro_success_','gpu_history_bytes')
    cols=[c for c in a if c.startswith(prefixes)]
    assert cols
    differences={c:int((a[c].to_numpy()!=b[c].to_numpy()).sum()) for c in cols}
    assert not any(differences.values()),differences
    return dict(reference=str(reference),staged=str(staged),frames=len(a),
                compared_columns=cols,differences=differences,status='PASS')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--staged',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    result=compare(args.reference,args.staged)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',frames=result['frames'],columns=len(result['compared_columns']))))
