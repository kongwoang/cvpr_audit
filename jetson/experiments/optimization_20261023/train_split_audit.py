"""Audit actual available training/selection unit IDs and checkpoint norms."""
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch

B=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
OUT=B/'experiments/optimization_20261023/train_audit'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    split=pd.read_csv(B/'results/k23/av2_split.csv')
    av2=sorted(split.log);nusc=sorted(pd.read_csv(B/'results/nusc_full/split.csv').seq.unique())
    allowed={ds:{p.stem for p in (B/'results/k30/dev_devref'/ds).glob('*.pkl')} for ds in ('av2','nusc')}
    rows=[];unmapped=[];files=[];cache={}
    def register(component,ds,seq,roles,file):
        for value,role in sorted(set(zip(seq.tolist(),roles.tolist()))):
            key=value.decode() if isinstance(value,bytes) else str(value)
            if (ds,key) not in cache:
                if ds=='nusc' and re.fullmatch(r'scene-\d{4}',key):matches=[key]
                else:matches=[s for s in (av2 if ds=='av2' else nusc) if s.startswith(key)]
                cache[(ds,key)]=matches
            matches=cache[(ds,key)]
            if len(matches)!=1:
                unmapped.append(dict(component=component,dataset=ds,prefix=key,matches=len(matches),file=str(file)))
                continue
            rows.append(dict(component=component,dataset=ds,unit=matches[0],sample_split=str(role),
                             in_development=matches[0] in allowed[ds],file=str(file)))
    def record_file(f):
        files.append(dict(file=str(f),bytes=f.stat().st_size,mtime=f.stat().st_mtime))
    # Match k28_ttrain.load/extra exactly, without loading image arrays.
    feats=[];roles=[];sources=[]
    for file in sorted((B/'data').glob('tdata__*of*.npz')):
        record_file(file);z=np.load(file,allow_pickle=True);ok=np.isfinite(z['target'])
        ds=file.name.split('__')[1].split('_')[0]
        register('C1',ds,z['seq'][ok],z['split'][ok],file)
        feat=np.c_[z['feats'][ok],np.log(np.clip(z['hr'][ok],1e-3,None)),z['dt'][ok],np.ones(int(ok.sum()))].astype(np.float32)
        feats.append(feat);roles.append(z['split'][ok]);sources.append(np.full(int(ok.sum()),ds))
    f=np.concatenate(feats);r=np.concatenate(roles);src=np.concatenate(sources)
    c1file=B/'out/k28_temporal/temporal_head.pt';c1=torch.load(c1file,map_location='cpu',weights_only=False)
    fm=f[r=='train'].mean(0);fs=f[r=='train'].std(0)+1e-6
    norms=[dict(component='C1',fmu_max_abs=float(np.max(np.abs(fm-c1['fmu']))),
        fsd_max_abs=float(np.max(np.abs(fs-c1['fsd']))),
        norms_match=bool(np.allclose(fm,c1['fmu'],rtol=1e-6,atol=1e-7) and np.allclose(fs,c1['fsd'],rtol=1e-6,atol=1e-7)))]
    c1counts={ds+'_'+sp:int(((src==ds)&(r==sp)).sum()) for ds in ('av2','nusc') for sp in ('train','val')}
    report=json.loads((B/'out/k28_temporal/train_report.json').read_text())
    assert c1counts==report['samples'],(c1counts,report['samples'])
    # Match k23_train_refiner.load; all training rows influence normalization.
    ff=[];rr=[];ss=[]
    for ds,file in [('KITTI',B/'data/refiner_data__KITTI.npz'),('nuScenes',B/'data/refiner_data__nuScenesAll.npz')]:
        record_file(file);z=np.load(file,allow_pickle=True)
        keep=np.isin(z['split'],['train','val']) if ds=='KITTI' else z['split']=='train'
        ff.append(np.c_[z['feats'][keep],np.full(int(keep.sum()),0. if ds=='KITTI' else 1.)])
        rr.append(z['split'][keep]);ss.append(np.full(int(keep.sum()),ds))
        if ds=='nuScenes':register('refiner','nusc',z['seq'][keep],z['split'][keep],file)
    for file in sorted((B/'data').glob('refiner_data__AV2dev_*of*.npz')):
        record_file(file);z=np.load(file,allow_pickle=True)
        register('refiner','av2',z['seq'],z['split'],file)
        ff.append(np.c_[z['feats'],np.ones(len(z['target']))]);rr.append(z['split']);ss.append(np.full(len(z['target']),'AV2'))
    f=np.concatenate(ff);r=np.concatenate(rr);src=np.concatenate(ss)
    reffile=B/'out/resnet18_v3a/image_refiner.pt';ref=torch.load(reffile,map_location='cpu',weights_only=False)
    fm=f[r=='train'].mean(0);fs=f[r=='train'].std(0)+1e-6
    norms.append(dict(component='refiner',fmu_max_abs=float(np.max(np.abs(fm-ref['fmu']))),
        fsd_max_abs=float(np.max(np.abs(fs-ref['fsd']))),
        norms_match=bool(np.allclose(fm,ref['fmu'],rtol=1e-6,atol=1e-7) and np.allclose(fs,ref['fsd'],rtol=1e-6,atol=1e-7))))
    counts={ds+'_'+sp:int(((src==ds)&(r==sp)).sum()) for ds in ('KITTI','nuScenes','AV2') for sp in ('train','val')}
    report=json.loads((B/'out/resnet18_v3a/train_report.json').read_text());assert counts==report['samples']
    units=pd.DataFrame(rows).drop_duplicates(['component','dataset','unit','sample_split'])
    units.to_csv(OUT/'training_selection_units.csv',index=False)
    pd.DataFrame(unmapped,columns=['component','dataset','prefix','matches','file']).to_csv(OUT/'unmapped.csv',index=False)
    pd.DataFrame(norms).to_csv(OUT/'checkpoint_norms.csv',index=False)
    pd.DataFrame(files).to_csv(OUT/'input_files.csv',index=False)
    overlap=units[units.in_development];overlap.to_csv(OUT/'development_overlap.csv',index=False)
    status=dict(status='PASS' if len(overlap)==0 and len(unmapped)==0 and all(n['norms_match'] for n in norms) else 'REVIEW_REQUIRED',
        development_overlap_rows=len(overlap),unmapped_rows=len(unmapped),norms=norms,
        c1_sample_counts=c1counts,refiner_sample_counts=counts,
        checkpoints={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (c1file,reffile)},
        limitation='Available training-array unit IDs, source loading recipe, report counts and checkpoint feature norms agree; original per-NPZ training-time hashes are unavailable.')
    (OUT/'DONE.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))


if __name__=='__main__':main()
