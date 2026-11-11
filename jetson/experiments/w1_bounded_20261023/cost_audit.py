"""Audit requested vs realized heuristic budgets, including retrospective pixels."""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

B=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
E=B/'experiments/w1_bounded_20261023'


def main():
    out=E/'reports/cost_audit';out.mkdir(exist_ok=True)
    frames=[]
    for ds,expected in [('nusc',152),('av2',300)]:
        files=sorted((E/'results/dev'/ds).glob('*.pkl'));assert len(files)==expected
        cheap=(192*320/(1600*900) if ds=='nusc' else 320*256/(1550*2048))
        for file in files:
            for r in pickle.load(file.open('rb')):
                if not r['gt']:continue
                candidate=r['px_v6S_b2A_retro']
                for pipeline in ('tsp8@combinedbudget+C1','unc2@combinedbudget+C1'):
                    request=r['requested_crop_px_'+pipeline]
                    actual=r['px_'+pipeline]
                    excess=max(0.,actual-(cheap+request))
                    assert abs(request+cheap-candidate)<1e-12
                    assert abs(excess-r['guard_budget_overrun_px_'+pipeline])<1e-12
                    frames.append(dict(dataset=ds,unit=file.stem,frame=r['frame'],pipeline=pipeline,
                        requested_total_px=cheap+request,realized_total_px=actual,overrun_px=excess,
                        over_budget=excess>1e-12,candidate_total_px=candidate))
    d=pd.DataFrame(frames);d.to_csv(out/'frames.csv',index=False)
    summary=[]
    for (ds,pipeline),g in d.groupby(['dataset','pipeline']):
        units=sorted(g.unit.unique());n=len(units)
        agg=g.groupby('unit').agg(frames=('frame','count'),request=('requested_total_px','sum'),
            actual=('realized_total_px','sum'),extra=('overrun_px','sum'),above=('over_budget','sum')).reindex(units)
        draws=np.random.default_rng(0).integers(0,n,(2000,n));c=np.array([np.bincount(x,minlength=n) for x in draws],float)
        diff=c@(agg.actual-agg.request).to_numpy()/(c@agg.frames.to_numpy())
        lo,hi=np.percentile(diff,[2.5,97.5])
        summary.append(dict(dataset=ds,pipeline=pipeline,units=n,frames=len(g),
            mean_requested_total_px=g.requested_total_px.mean(),mean_realized_total_px=g.realized_total_px.mean(),
            mean_overrun_px=g.overrun_px.mean(),p95_overrun_px=g.overrun_px.quantile(.95),
            above_requested_budget_fraction=g.over_budget.mean(),
            realized_minus_candidate_mean_px=(g.realized_total_px-g.candidate_total_px).mean(),ci_lo=lo,ci_hi=hi))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    lines=['# Requested vs realized control budgets','',
        'Full development set; all pixel values are detector input pixels / native image pixels.',
        'These are logical VUW canvas pixels, not Jetson common-padding tensor pixels or latency.',
        'Every requested budget equals the combined candidate current+guard+retrospective work plus one cheap pass.',
        'Controls may overrun the request due to their guards. No exact equal-total-cost comparison is asserted.',
        'Paired unit bootstrap 2000 seed0 for realized-minus-candidate mean pixel differences.','',
        '| Dataset | Control | Requested total px | Realized total px | Frames over request |',
        '|---|---|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['dataset']} | {r['pipeline']} | {r['mean_requested_total_px']:.6f} | {r['mean_realized_total_px']:.6f} | {100*r['above_requested_budget_fraction']:.2f}% |")
    lines+=['','Exact per-frame arithmetic checks passed for both controls on every evaluated frame.',
        'Control detector-call counts are not recorded by this harness; missing counts are not zero.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'DONE.json').write_text(json.dumps(dict(status='complete',frame_rows=len(d),budget_arithmetic_equal=True),indent=2)+'\n')


if __name__=='__main__':main()
