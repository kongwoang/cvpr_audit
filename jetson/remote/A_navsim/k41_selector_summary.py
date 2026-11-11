"""P01 completed cached-row cohort: paired log bootstrap, explicit score protocol."""
import json
import os
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
B=Path(os.environ['B']);sys.path.insert(0,str(B/'results/k40r/vendor'))
from analysis.common import Bootstrap


def interval(bt,num,den):
    n,d=bt.sums(num),bt.sums(den);bn,bd=n[bt.idx].sum(1),d[bt.idx].sum(1)
    z=bn[bd>0]/bd[bd>0]
    if not len(z) or not d.sum():return dict(value=None,lo=None,hi=None,valid_draws=0)
    lo,hi=np.percentile(z,[2.5,97.5]);return dict(value=float(n.sum()/d.sum()),lo=float(lo),hi=float(hi),valid_draws=len(z))


def main():
    src=B/'results/k41/selector_metrics';out=src/'summary';out.mkdir(exist_ok=True)
    state=json.loads((src/'state.json').read_text());assert state['status']=='complete'
    names=state['rows'];rows=[];timing=[];sources=[]
    for p in sorted(src.glob('*.pkl')):
        assert p.with_suffix('.done.json').exists()
        meta=json.loads(p.with_suffix('.done.json').read_text());assert meta['script_sha256']==state['script_sha256']
        sources.append(meta)
        for token,r in pickle.load(p.open('rb')).items():
            for name in names:
                rows.append(dict(log=r['log'],token=token,row=name,choice=r['selected'][name],**r['scores'][name]))
            timing.append(dict(log=r['log'],token=token,cache_bytes=r['cache']['bytes'],cache_build_seconds=r['cache']['build_wall_seconds'],
                               unique_scores=sum(not r['scoring'][n]['reused_identical_trajectory'] for n in names)))
    data=pd.DataFrame(rows);assert data.log.nunique()==136 and len(data)==12146*len(names)
    assert not data.duplicated(['log','token','row']).any()
    data.to_csv(out/'selector_pdms_frames.csv',index=False)
    pivot=data.pivot(index=['log','token'],columns='row');bt=Bootstrap(pivot.index.get_level_values('log'))
    choices=pivot['choice'];n=len(choices);ones=np.ones(n);bad=choices['cheap'].to_numpy()!=choices['full'].to_numpy()
    metrics=[c for c in data if c not in ('log','token','row','choice')];summary=[]
    for subset,mask in (('all',ones),('cheap_target_disagrees',bad.astype(float))):
        for name in names:
            for metric in metrics:
                x=pivot[metric][name].to_numpy();ref=pivot[metric]['field'].to_numpy()
                summary.append(dict(subset=subset,row=name,metric=metric,quantity='mean',tokens=int(mask.sum()),**interval(bt,x*mask,mask)))
                summary.append(dict(subset=subset,row=name,metric=metric,quantity='field_minus_row',tokens=int(mask.sum()),**interval(bt,(ref-x)*mask,mask)))
            match=(choices[name].to_numpy()==choices['full'].to_numpy()).astype(float)
            summary.append(dict(subset=subset,row=name,metric='target_agreement',quantity='mean',tokens=int(mask.sum()),**interval(bt,match*mask,mask)))
    table=pd.DataFrame(summary);table.to_csv(out/'selector_pdms_summary.csv',index=False)
    pd.DataFrame(timing).to_csv(out/'selector_pdms_cache_cost.csv',index=False)
    usage=list((B/'results/k41').glob('selector_metrics*/usage/*.json'))
    cpu=sum(json.loads(p.read_text())['cpu_seconds'] for p in usage)
    result=dict(task='K41-1 / P01 cached rows',protocol='official NAVSIM v1.1 PDMS; not EPDMS',tokens=n,logs=136,rows=names,
        cpu_hours_measured=cpu/3600,bootstrap_draws=2000,seed=0,remaining_rows=['all','selected','selected_support','geometric'],
        hybrid='stored K30 hybrid_pdm detections: fixed route tube with PDM support, passed to K31 selector',
        source_driver_sha256=state['script_sha256'])
    (out/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    display=table[(table.subset=='all')&(table.metric=='score')]
    print(display.to_string(index=False),flush=True)
    score=display[display.quantity=='mean'].set_index('row')
    delta=display[display.quantity=='field_minus_row'].set_index('row')
    text=['# K41-1: selector downstream PDMS (post-hoc)','',
          f'Completed five cached rows on {n:,} navtest tokens /136 logs. Official NAVSIM v1.1 PDMS; this is not EPDMS. All cheap/full selector checks passed. Original metric caches were absent, so the installed official processor reconstructed each token from unchanged ground-truth scenes.','',
          '| Row | PDMS | 95% CI | Field − row, 95% CI |','|---|---:|---|---|']
    for name in names:
        r=score.loc[name];d=delta.loc[name]
        text.append(f'| {name} | {r.value:.6f} | [{r.lo:.6f}, {r.hi:.6f}] | {d.value:+.6f} [{d.lo:+.6f}, {d.hi:+.6f}] |')
    text+=['',f'Measured CPU use including smoke/checks: {cpu/3600:.3f} CPU hours. Time waiting for GPU is unrelated to this CPU job. Paired bootstrap by log:2,000 draws, seed0. Oracle is a locality diagnostic; hybrid retains the original PDM-support crop policy. PiP/selected-support/geometric rows remain pending D04; these results do not close the full task.',
           '', 'Reproduce after `source env.sh; source scripts/common/env_navsim.sh`: `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/A_navsim/k41_selector_metrics.py --rows cheap,full,field,oracle,hybrid --workers 4`; then `python scripts/A_navsim/k41_selector_summary.py`. Source SHA256 and measured costs are in run.json and per-log metadata.']
    (out/'SELECTOR_PDMS.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
