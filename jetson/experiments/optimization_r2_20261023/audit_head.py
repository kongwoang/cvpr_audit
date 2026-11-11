"""Frozen-checkpoint predictions on original validation samples, no fitting."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
import pandas as pd
import torch

B=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR');HERE=B/'experiments/optimization_r2_20261023'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['nusc','av2'],required=True);a=ap.parse_args()
    out=HERE/'head_audit';out.mkdir(exist_ok=True);state=out/(a.dataset+'_STATUS.json')
    data=dict(status='running',dataset=a.dataset,pid=os.getpid(),host=socket.gethostname(),started=time.time())
    def save(**kw):
        data.update(kw);tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(state)
    save()
    try:
        torch.set_num_threads(4)
        modfile=B/'scripts/A_method/k28_ttrain.py';spec=importlib.util.spec_from_file_location('frozen_train',modfile)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        ckfile=B/'out/k28_temporal/temporal_head.pt';ckhash=hashlib.sha256(ckfile.read_bytes()).hexdigest()
        ck=torch.load(ckfile,map_location='cpu',weights_only=False);net=m.TempHead(ck['n_feat'])
        net.load_state_dict(ck['state']);net=net.to('cuda:0').eval();rows=[];files=[]
        for f in sorted((B/'data').glob('tdata__'+a.dataset+'_*of*.npz')):
            with np.load(f,allow_pickle=True) as z:
                good=np.isfinite(z['target'])&(z['split']=='val');ix=np.flatnonzero(good)
                files.append(dict(name=f.name,samples=len(ix),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
                if not len(ix):continue
                crops=z['crops'];feats=z['feats'];hr=z['hr'];dt=z['dt'];y=z['target'];r4=z['rT4'];seq=z['seq'];kind=z['kind']
                extra=m.extra(feats,hr,dt);extra=((extra-ck['fmu'])/ck['fsd']).astype(np.float32)
                for start in range(0,len(ix),128):
                    ids=ix[start:start+128]
                    x=torch.from_numpy(crops[ids]).permute(0,1,4,2,3).contiguous().to('cuda:0').float()/255.
                    ff=torch.from_numpy(extra[ids]).to('cuda:0')
                    with torch.no_grad():p=np.clip(net(x,ff).float().cpu().numpy(),0,2)
                    for i,pred in zip(ids,p):
                        rows.append(dict(dataset=a.dataset,file=f.name,index=int(i),unit=str(seq[i]),kind=str(kind[i]),
                            target=float(np.clip(y[i],0,2)),unclipped_target=float(y[i]),C1=float(pred),T4=float(np.clip(r4[i],0,2))))
                del crops
            save(completed_files=len(files),samples=len(rows));print(a.dataset,f.name,len(rows),flush=True)
        d=pd.DataFrame(rows);d.to_csv(out/(a.dataset+'_predictions.csv'),index=False)
        expected=json.loads((B/'out/k28_temporal/train_report.json').read_text())['best']['val_mae_'+a.dataset]
        actual=float(np.mean(np.abs(d.C1-d.target)))
        assert abs(actual-expected)<1e-5,(actual,expected)
        assert hashlib.sha256(ckfile.read_bytes()).hexdigest()==ckhash
        (out/(a.dataset+'_sources.json')).write_text(json.dumps(dict(checkpoint_sha256=ckhash,
            code_sha256=hashlib.sha256(modfile.read_bytes()).hexdigest(),data=files),indent=2)+'\n')
        save(status='complete',samples=len(d),units=int(d.unit.nunique()),mae=actual,expected_mae=expected,finished=time.time())
    except BaseException as e:save(status='failed',error=repr(e));raise

if __name__=='__main__':main()
