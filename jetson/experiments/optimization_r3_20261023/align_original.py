"""Join original C1 samples to regenerated identity metadata by exact crop bytes.

No frame IDs existed in the original NPZ. Ambiguous hashes are reported, never
silently picked. This separates an audit of original samples from a new corpus.
"""
import argparse
import hashlib
from collections import defaultdict
import numpy as np
import pandas as pd
from common import HERE,REMOTE,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args()
    lookup=defaultdict(list);total=0
    for f in sorted((HERE/'training_data'/a.dataset).glob('*.npz')):
        with np.load(f,allow_pickle=False) as z:
            if not len(z['frame']):continue
            ix=np.flatnonzero(np.isin(z['kind'],['crop','u960'])&(z['mask'].sum(1)==3))
            if not len(ix):continue
            crops=z['crops'];frames=z['frame'];ids=z['ids'];target=z['target'];kind=z['kind'];split=z['split']
            for i in ix:
                key=(f.stem[:16],str(kind[i]),hashlib.sha256(crops[i].tobytes()).hexdigest())
                lookup[key].append(dict(unit=f.stem,frame=int(frames[i]),ids=ids[i].tolist(),target=float(target[i]),split=str(split[i])))
                total+=1
    rows=[]
    for f in sorted((REMOTE/'data').glob('tdata__'+a.dataset+'_*of*.npz')):
        with np.load(f,allow_pickle=False) as z:
            crops=z['crops'];targets=z['target'];seq=z['seq'];kind=z['kind'];split=z['split']
            for i in range(len(crops)):
                key=(str(seq[i]),str(kind[i]),hashlib.sha256(crops[i].tobytes()).hexdigest());hits=lookup.get(key,[])
                row=dict(dataset=a.dataset,file=f.name,index=i,unit_prefix=str(seq[i]),kind=str(kind[i]),split=str(split[i]),target=float(targets[i]),matches=len(hits))
                if len(hits)==1:
                    r=hits[0];ids=np.array(r['ids']);assert r['split']==str(split[i])
                    row.update(unit=r['unit'],frame=r['frame'],known_all=bool((ids>=0).all()),switched=bool((ids>=0).all() and np.any(ids[1:]!=ids[0])),
                        regenerated_target=r['target'],identity_current=ids[0],identity_previous=ids[1],identity_earlier=ids[2])
                rows.append(row)
        print(a.dataset,f.name,len(rows),flush=True)
    out=HERE/'reports/training_audit';out.mkdir(parents=True,exist_ok=True);d=pd.DataFrame(rows)
    d.to_csv(out/(a.dataset+'_original_alignment.csv'),index=False)
    summaries=[]
    for (sp,kind),g in d.groupby(['split','kind']):
        valid=g[np.isfinite(g.target)];unique=valid[valid.matches==1];known=unique[unique.known_all==True]
        summaries.append(dict(dataset=a.dataset,split=sp,kind=kind,original_samples=len(g),finite_target_samples=len(valid),
            exact_unique_matches=len(unique),exact_ambiguous_matches=int((valid.matches>1).sum()),unmatched=int((valid.matches==0).sum()),
            known_all=len(known),known_switched=int((known.switched==True).sum()),
            target_differing=int((abs(unique.target-unique.regenerated_target)>1e-5).sum())))
    pd.DataFrame(summaries).to_csv(out/(a.dataset+'_original_alignment_summary.csv'),index=False)
    save(out/(a.dataset+'_alignment_STATUS.json'),dict(status='complete',regenerated_full_chains=total,original_samples=len(d)))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
if __name__=='__main__':main()
