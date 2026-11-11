"""Wait for both CPU shards, pull every row, and report paired unit intervals."""
import json
import shlex
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REMOTE='/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/experiments/navsim_reuse_20261023'
OUT=ROOT/'docs/paper/handoff/NAVSIM_REUSE_20261023'


def stats(x):return np.array([x.mean(),np.percentile(x,95),np.percentile(x,99)])


def main():
    OUT.mkdir(exist_ok=True)
    while True:
        code="import pathlib,json; p=pathlib.Path("+repr(REMOTE)+"); print(json.dumps([json.loads(f.read_text()) for f in p.glob('reports/shard*/state.json')]))"
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','VUW1',shlex.join(['python3','-c',code])],
                         capture_output=True,text=True,timeout=60)
        if r.returncode:
            time.sleep(30);continue
        states=json.loads(r.stdout)
        (OUT/'STATUS.json').write_text(json.dumps(states,indent=2)+'\n')
        assert not any(x['status']=='failed' for x in states),states
        if len(states)==2 and all(x['status']=='complete' for x in states):break
        time.sleep(30)
    subprocess.run(['rsync','-a','VUW1:'+REMOTE+'/reports/',str(HERE/'reports')+'/'],check=True)
    while (Path.home()/'.jetson_pause').exists():time.sleep(20)
    d=pd.concat([pd.read_csv(p) for p in sorted((HERE/'reports').glob('shard*/frames.csv'))],ignore_index=True)
    assert len(d)==6000 and not d.duplicated(['unit','token','grid','rep']).any()
    assert d[['unit','token']].drop_duplicates().shape[0]==1000
    assert d[['all_scores_equal','action_equal','field_equal','support_equal']].all().all()
    assert (d.groupby(['unit','token','grid']).rep.nunique()==3).all()
    d.to_csv(OUT/'frames.csv',index=False)
    summaries=[];intervals=[];contrasts=[]
    names=['mean_ms','p95_ms','p99_ms']
    for grid,g in d.groupby('grid'):
        units=sorted(g.unit.unique());n=len(units)
        arrays={m:[g[g.unit==u].sort_values(['token','rep'])[m+'_ms'].to_numpy() for u in units]
                for m in ('reference','reuse')}
        points={m:stats(np.concatenate(x)) for m,x in arrays.items()}
        draws=np.random.default_rng(0).integers(0,n,(2000,n))
        boot={m:np.array([stats(np.concatenate([x[j] for j in draw])) for draw in draws]) for m,x in arrays.items()}
        for method in arrays:
            summaries.append(dict(grid=grid,method=method,units=n,tokens=1000,repetitions=3,**dict(zip(names,points[method]))))
            for i,name in enumerate(names):
                low,high=np.percentile(boot[method][:,i],[2.5,97.5])
                intervals.append(dict(grid=grid,method=method,metric=name,estimate=points[method][i],ci_lo=low,ci_hi=high))
        for i,name in enumerate(names):
            low,high=np.percentile(boot['reuse'][:,i]-boot['reference'][:,i],[2.5,97.5])
            contrasts.append(dict(grid=grid,metric=name,reuse_minus_reference=points['reuse'][i]-points['reference'][i],ci_lo=low,ci_hi=high))
        ratio=boot['reference'][:,0]/boot['reuse'][:,0]
        low,high=np.percentile(ratio,[2.5,97.5])
        contrasts.append(dict(grid=grid,metric='mean_speedup',reuse_minus_reference=points['reference'][0]/points['reuse'][0],ci_lo=low,ci_hi=high))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'intervals.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv',index=False)
    hosts=[]
    for (host,grid),g in d.groupby(['host','grid']):
        for method in ('reference','reuse'):
            hosts.append(dict(host=host,grid=grid,method=method,tokens=len(g)//3,
                              **dict(zip(names,stats(g[method+'_ms'].to_numpy())))))
    pd.DataFrame(hosts).to_csv(OUT/'host_summary.csv',index=False)
    lines=['# Exact NAVSIM bottleneck-selector score reuse','',
        'All 1,000 K34 tokens, both grids, three paired repetitions; 6,000 paired rows.',
        'All candidate scores, actions, field and binding-support arrays match exactly on every repetition.',
        'Each host uses one CPU thread; whole logs assigned to VUW/VUW1 by fixed hash.',
        'Intervals: paired log bootstrap, 2,000, seed 0, repetitions nested in logs; pooled quantiles.','',
        '| Grid | Method | Mean ms | p95 ms | p99 ms |','|---|---|---:|---:|---:|']
    for r in summaries:lines.append('| '+r['grid']+' | '+r['method']+' | '+' | '.join(f'{r[k]:.3f}' for k in names)+' |')
    lines+=['','The fixed selector reuses its base scene scores and scores all probe objects in one matrix.',
        'Candidate paths, probe grid, argmax tie rule and binding support are unchanged.',
        'Scope includes scene/selector construction and field/support extraction; excludes disk, detector,',
        'and the rest of the perception/planning pipeline. This is not a Jetson or PDM-Closed speed claim.',
        'Shared-host load and unit/host assignment limit hardware generalization; within-token pairs are on the same host.',
        'All rows: frames.csv; paired contrasts: contrasts.csv; per-host results: host_summary.csv.',
        'Original K31/K34 sources are unchanged. Raw shard metadata/source hashes retained in the experiment folder.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'DONE.json').write_text(json.dumps(dict(status='complete',tokens=1000,paired_rows=len(d)),indent=2)+'\n')


if __name__=='__main__':main()
