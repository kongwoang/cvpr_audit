"""Portable sandbox weights; array metadata becomes lists for NumPy1 Jetson."""
import hashlib
import numpy as np
import torch
from common import HERE,REMOTE,save,sha

def statehash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):h.update(key.encode());h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    out=HERE/'jetson_exports';rows=[]
    c1=torch.load(REMOTE/'out/k28_temporal/temporal_head.pt',map_location='cpu',weights_only=False)
    for ds in ('nusc','av2'):
        for objective in ('geometry','decision'):
            src=HERE/'models'/ds/(objective+'_s0')/'adapter.pt';target=out/'models'/ds/(objective+'_s0')/'adapter.pt'
            ck=torch.load(src,map_location='cpu',weights_only=False);original=statehash(ck['state'])
            for k in ('fmu','fsd'):ck[k]=np.asarray(ck[k]).tolist()
            target.parent.mkdir(parents=True,exist_ok=True);torch.save(ck,target)
            assert statehash(torch.load(target,map_location='cpu',weights_only=False)['state'])==original
            rows.append(dict(dataset=ds,objective=objective,source_sha256=sha(src),portable_sha256=sha(target),state_sha256=original))
    save(out/'MANIFEST.json',dict(status='complete',models=rows,c1_state_sha256=statehash(c1['state']),
        c1_checkpoint_sha256=sha(REMOTE/'out/k28_temporal/temporal_head.pt'),normalization='same float32 values serialized as Python lists'))
    print('Portable weights exported',flush=True)
if __name__=='__main__':main()
