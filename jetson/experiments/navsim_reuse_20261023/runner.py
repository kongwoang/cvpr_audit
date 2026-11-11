"""Fixed-manifest, three-repetition CPU benchmark; no frozen source mutation."""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import socket
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reuse_kernels',HERE/'kernels.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
K,C,evaluate=M.K,M.C,M.evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',required=True)
    a=ap.parse_args();si,sn=map(int,a.shard.split('/'));assert sn==2 and 0<=si<sn
    out=HERE/'reports'/f'shard{si}of{sn}';out.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',pid=os.getpid(),host=socket.gethostname(),shard=a.shard,start=time.time())
    def save(**kw):
        state.update(kw,updated=time.time());p=out/'state.tmp'
        p.write_text(json.dumps(state,indent=2)+'\n');p.replace(out/'state.json')
    save()
    sources=[K.B/'scripts/A_navsim/k31_nav_p2.py',K.B/'scripts/A_navsim/k34_selector_cost.py',HERE/'kernels.py',Path(__file__)]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest=K.B/'results/k34/selector/token_manifest.csv'
    todo=pd.read_csv(manifest).sort_values(['unit','token'])
    assert len(todo)==1000 and not todo.duplicated(['unit','token']).any()
    todo=todo[todo.unit.map(lambda u:int(hashlib.sha256(u.encode()).hexdigest()[:8],16)%sn==si)]
    todo.to_csv(out/'token_manifest.csv',index=False)
    gz=float((K.EXP/'dets/ground_z.txt').read_text())
    priors=pickle.load((K.EXP/'dets/size_priors.pkl').open('rb'))
    rows=[];warmed=False;number=0
    try:
        for unit,tasks in todo.groupby('unit',sort=True):
            dets=pickle.load((K.EXP/'dets_k30b'/(unit+'.pkl')).open('rb'))
            centers=pickle.load((K.EXP/'k30_center'/(unit+'.pkl')).open('rb'))
            frames={r['token']:r for r in pickle.load((K.DATA/'navsim_logs/test'/(unit+'.pkl')).open('rb'))}
            old={r['token']:r for r in pickle.load((K.B/'results/k31/p2'/(unit+'.pkl')).open('rb'))}
            for token in tasks.token:
                args=(frames[token],dets[token],centers.get(token),gz,priors)
                if not warmed:
                    for _ in range(5):
                        for grid in ('7x7','3x3'):
                            evaluate(*args,grid,False);evaluate(*args,grid,True)
                    warmed=True
                for rep in (1,2,3):
                    for gi,grid in enumerate(('7x7','3x3') if (number+rep)%2 else ('3x3','7x7')):
                        pair={}
                        for fast in ((False,True) if (number+rep+gi)%2 else (True,False)):
                            pair[fast]=evaluate(*args,grid,fast)
                        ref,opt=pair[False],pair[True]
                        assert ref['action']==opt['action']
                        for key in ('field','support','scores'):
                            assert np.array_equal(ref[key],opt[key],equal_nan=True),(unit,token,grid,rep,key)
                        if grid=='7x7':
                            assert ref['action']==old[token]['pi_cheap']
                            assert len(ref['field'])==old[token]['field_cells_r1']
                            assert len(ref['support'])==old[token]['support_r1']
                        rows.append(dict(unit=unit,token=token,grid=grid,rep=rep,host=socket.gethostname(),
                            reference_ms=ref['ms'],reuse_ms=opt['ms'],n_objects=ref['n_objects'],n_probes=ref['n_probes'],
                            reference_matrix_entries=ref['matrix_entries'],reuse_matrix_entries=opt['matrix_entries'],
                            all_scores_equal=True,action_equal=True,field_equal=True,support_equal=True))
                number+=1
            pd.DataFrame(rows).to_csv(out/'frames.partial.csv',index=False)
            save(tokens=number,expected_tokens=len(todo));print(f'{number}/{len(todo)} tokens',flush=True)
        assert len(rows)==6*len(todo)
        assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        pd.DataFrame(rows).to_csv(out/'frames.csv',index=False)
        save(status='complete',tokens=number,rows=len(rows),source_hashes=before,
             manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
             load_average=os.getloadavg(),omp_threads=os.environ.get('OMP_NUM_THREADS'),finish=time.time())
    except BaseException as exc:
        save(status='failed',error=repr(exc));raise


if __name__=='__main__':
    main()
