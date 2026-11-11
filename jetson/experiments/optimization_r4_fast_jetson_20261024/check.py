"""Fresh baseline-port parity and exact action parity of refiner abstention."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset');ap.add_argument('--old');ap.add_argument('--new');a=ap.parse_args()
    old,new=[pd.read_pickle(Path(p)/'frames.pkl') for p in (a.old,a.new)]
    assert old[['unit','frame','a_gt']].equals(new[['unit','frame','a_gt']])
    cols=[f'{metric}_{name}@trt32{suffix}' for metric in ('a','J','px') for name in ('ours_b2A_retro','ours_b2A_median_retro','ours_b2A_medianskip_retro') for suffix in ('','_noref')]
    parity={c:int((np.abs(old[c]-new[c])>1e-8).sum()) for c in cols};assert not any(parity.values()),parity
    diffs=int((new['a_ours_b2A_median_retro@trt32']!=new['a_ours_b2A_medianskip_retro@trt32']).sum());assert not diffs,diffs
    report=dict(status='complete',dataset=a.dataset,frames=len(new),baseline_differences=parity,median_abstention_action_differences=diffs)
    (Path(__file__).resolve().parent/'logs'/('parity_'+a.dataset+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(report)

if __name__=='__main__':main()
