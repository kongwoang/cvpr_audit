"""Recomputed baseline port parity and learned Torch/TRT action parity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset');ap.add_argument('--old');ap.add_argument('--new');ap.add_argument('--torch');a=ap.parse_args()
    here=Path(__file__).resolve().parent;root=here.parents[1]
    d=[pd.read_pickle(Path(p)/'frames.pkl') for p in (a.old,a.new,a.torch)]
    for x in d[1:]:assert x[['unit','frame','a_gt']].equals(d[0][['unit','frame','a_gt']])
    cols=[f'{metric}_{name}@trt32{suffix}' for metric in ('a','J','px') for name in ('ours','ours_b2A_retro','ours_b2A_missing_retro') for suffix in ('','_noref')]
    parity={c:int((np.abs(d[0][c]-d[1][c])>1e-8).sum()) for c in cols};assert not any(parity.values()),parity
    cols=[f'a_{name}@trt32{suffix}' for name in ('ours_b2A_missing_retro','ours_b2A_missing_compact_retro','ours_b2A_extent_retro','ours_b2A_missing_corroborated_retro') for suffix in ('','_noref')]
    encoder={c:int((d[1][c]!=d[2][c]).sum()) for c in cols}
    # Report backend differences if any; primary measurements remain trt32.
    ck=torch.load(root/'engines/k28_temporal_head.pt',map_location='cpu',weights_only=False)
    h=hashlib.sha256()
    for key,value in sorted(ck['state'].items()):h.update(key.encode());h.update(value.cpu().contiguous().numpy().tobytes())
    portable=json.loads((here/'MANIFEST.json').read_text());assert h.hexdigest()==portable['c1_state_sha256']
    from compact import Compact
    ds='nusc' if a.dataset=='nusc_val' else 'av2'
    compact=torch.load(here/'models'/ds/'compact_s0/compact.pt',map_location='cpu',weights_only=False)
    net=Compact(compact['mu'],compact['sd']);net.load_state_dict(compact['state']);net.eval()
    torch.manual_seed(0);x=torch.tensor(compact['mu'])+torch.randn(128,32)*torch.tensor(compact['sd'])
    with torch.no_grad():
        cpu=net(None,x[:,:16],None,x[:,16:]).numpy()
        gpu=net.cuda()(None,x[:,:16].cuda(),None,x[:,16:].cuda()).cpu().numpy()
    compact_delta=float(np.max(np.abs(cpu-gpu)));assert compact_delta<1e-5,compact_delta
    report=dict(compact_cpu_gpu_max_delta=compact_delta,status='complete',dataset=a.dataset,frames=len(d[0]),baseline_differences=parity,adapter_encoder_differences=encoder,c1_encoder_weights_equal=True)
    (here/'logs'/('parity_'+a.dataset+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
