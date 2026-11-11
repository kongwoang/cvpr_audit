"""Cross-artifact checks, episode transitions, figures and final research handoff."""
import hashlib
import json
import time
from pathlib import Path

ROOT=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/experiments/optimization_20261023/report_bundle')
H=ROOT/'docs/paper/handoff'
OUT=H/'OPTIMIZATION_20261023'


def complete():
    return (all((H/name/'DONE.json').exists() for name in ('NAVSIM_REUSE_20261023','RUNTIME_20261023',
        'TEMPORAL_JETSON_20261023','REFINER_GEOMETRY_20261023')) and
        all(json.loads((ROOT/'experiments'/name/'coordinator_state.json').read_text()).get('phase')=='complete'
            for name in ('w1_dev_20261023','w1_bounded_20261023')))


def plot_results(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#46515f','#1977b8','#dc8b22','#b83c42']
    # Required-episode tradeoff, complete dev only.
    labels=['cheap','v6-S','temporal evidence','B2K-A','B2K-A + temporal evidence',
        'u960 + C1 + v3a','native + C1 + v3a','hybrid at combined requested budget','unc2 at combined requested budget']
    short=['cheap','v6-S','recovery','B2K-A','B2K-A + recovery','u960+C1+v3a','native+C1+v3a','hybrid','unc2']
    intervals=pd.read_csv(H/'W1_BOUNDED_20261023/dev/intervals.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),constrained_layout=True)
    for ax,ds in zip(axes,['nusc','av2']):
        data=summary[summary.dataset==ds].set_index('pipeline')
        for i,(name,label) in enumerate(zip(labels,short)):
            row=data.loc[name];x=row.false_per_1000;y=100*row.all_missed_rate
            z=intervals[(intervals.dataset==ds)&(intervals.pipeline==name)].set_index('metric')
            color=('#8a8f95' if i==0 or i>=5 else colors[i-1]);marker='s' if i==0 else ('o' if i<5 else '^')
            xx=z.loc['false_per_1000'];yy=z.loc['all_missed_rate']
            ax.errorbar(x,y,xerr=np.maximum(0,[[x-xx.ci_lo],[xx.ci_hi-x]]),
                yerr=np.maximum(0,[[y-100*yy.ci_lo],[100*yy.ci_hi-y]]),color=color,alpha=.6,lw=.7,capsize=2)
            ax.scatter(x,y,color=color,marker=marker,s=43,label=label,zorder=4)
        ax.set(xlabel='False episodes / 1,000 frames',ylabel='Missed required episodes (%)',title=ds)
        ax.grid(alpha=.18)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='lower center',ncol=5,bbox_to_anchor=(.5,-.16),frameon=False)
    fig.savefig(OUT/'development_tradeoff.pdf',bbox_inches='tight');fig.savefig(OUT/'development_tradeoff.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # Exact implementation CPU scope, both grids.
    folder=H/'NAVSIM_REUSE_20261023';data=pd.read_csv(folder/'summary.csv')
    ci=pd.read_csv(folder/'intervals.csv');fig,axes=plt.subplots(1,2,figsize=(7.4,3.5),constrained_layout=True)
    for ax,metric in zip(axes,['mean_ms','p95_ms']):
        for offset,method,color in [(-.17,'reference',colors[0]),(.17,'reuse',colors[1])]:
            rows=data[data.method==method].set_index('grid').loc[['3x3','7x7']]
            cc=ci[(ci.method==method)&(ci.metric==metric)].set_index('grid').loc[['3x3','7x7']]
            y=rows[metric].to_numpy()
            ax.bar(np.arange(2)+offset,y,.32,color=color,label=method,
                yerr=np.maximum(0,np.vstack([y-cc.ci_lo.to_numpy(),cc.ci_hi.to_numpy()-y])),capsize=3)
        ax.set(xticks=[0,1],xticklabels=['3x3','7x7'],ylabel='CPU milliseconds',title=metric.replace('_',' '),yscale='log')
        ax.grid(axis='y',alpha=.18);ax.legend(frameon=False)
    fig.savefig(OUT/'navsim_cpu_scope.pdf');fig.savefig(OUT/'navsim_cpu_scope.png',dpi=180);plt.close(fig)
    # All runtime and temporal rows retained, with pooled unit intervals.
    for report,filename,runtime in [('RUNTIME_20261023','runtime_staging',True),('TEMPORAL_JETSON_20261023','temporal_latency',False)]:
        folder=H/report;data=pd.read_csv(folder/'summary.csv');ci=pd.read_csv(folder/'intervals.csv')
        fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
        order=(['v6-S','v6-S <=2','B2K-A'] if runtime else ['v6-S','v6-S + recovery','B2K-A','B2K-A + recovery'])
        for row,ds in enumerate(['nusc_val','av2_dev']):
            for col,metric in enumerate(['mean_ms','p95_ms']):
                ax=axes[row,col]
                if runtime:
                    for off,mode,color in [(-.17,'reference',colors[0]),(.17,'staged',colors[1])]:
                        z=data[(data.dataset==ds)&(data.runtime==mode)].set_index('pipeline').loc[order]
                        cc=ci[(ci.dataset==ds)&(ci.runtime==mode)&(ci.metric==metric)].set_index('pipeline').loc[order]
                        y=z[metric].to_numpy();err=np.maximum(0,np.vstack([y-cc.ci_lo.to_numpy(),cc.ci_hi.to_numpy()-y]))
                        ax.bar(np.arange(len(order))+off,y,.32,color=color,label=mode,yerr=err,capsize=2)
                    ax.legend(frameon=False)
                else:
                    z=data[data.dataset==ds].set_index('pipeline').loc[order]
                    cc=ci[(ci.dataset==ds)&(ci.metric==metric)].set_index('pipeline').loc[order]
                    y=z[metric].to_numpy();err=np.maximum(0,np.vstack([y-cc.ci_lo.to_numpy(),cc.ci_hi.to_numpy()-y]))
                    ax.bar(np.arange(len(order)),y,.65,color=colors,yerr=err,capsize=2)
                ax.set(xticks=np.arange(len(order)),xticklabels=[s.replace(' + ','\n+ ') for s in order],
                       ylabel='Milliseconds',title=ds+' — '+metric.replace('_',' '))
                ax.grid(axis='y',alpha=.18)
        fig.savefig(OUT/(filename+'.pdf'));fig.savefig(OUT/(filename+'.png'),dpi=180);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    while not complete():time.sleep(30)
    # All GPU windows have released before heavier CPU plotting/validation.
    # CPU-only reporting is offloaded; the Jetson window is on another host.
    global np,pd
    import numpy as np
    import pandas as pd
    old=H/'W1_DEV_20261023/dev';new=H/'W1_BOUNDED_20261023/dev'
    a=pd.read_csv(old/'frames.csv').set_index(['dataset','unit','frame']).sort_index()
    b=pd.read_csv(new/'frames.csv').set_index(['dataset','unit','frame']).sort_index()
    assert a.index.equals(b.index) and not a.index.duplicated().any()
    for col in ('act_gt','a_cheap','a_v6S|none','a_v6S|v3a','a_v6S_retro|v3a'):
        assert np.array_equal(a[col],b[col]),col
    assert np.allclose(a.a_gt,b.a_gt,rtol=0,atol=1e-12)
    assert len(a)==29782
    summary=pd.read_csv(new/'summary.csv');assert len(summary)==30
    old_summary=pd.read_csv(old/'summary.csv')
    for ds,n in [('nusc',152),('av2',300)]:
        assert (summary[summary.dataset==ds].units==n).all()
    episodes=pd.read_csv(new/'episodes.csv');keys=['dataset','unit','start_frame','end_frame']
    transitions=[]
    for candidate,base in [('v6-S','cheap'),('temporal evidence','v6-S'),('B2K-A','v6-S'),
                            ('B2K-A + temporal evidence','v6-S'),('B2K-A + temporal evidence','cheap')]:
        aa=episodes[episodes.pipeline==base].set_index(keys).sort_index()
        bb=episodes[episodes.pipeline==candidate].set_index(keys).sort_index()
        assert aa.index.equals(bb.index)
        for ds in ('nusc','av2'):
            x,y=aa.loc[ds,'missed'],bb.loc[ds,'missed']
            transitions.append(dict(dataset=ds,candidate=candidate,comparator=base,episodes=len(x),
                comparator_missed=int(x.sum()),candidate_missed=int(y.sum()),
                recovered=int((x&~y).sum()),introduced=int((~x&y).sum()),both_missed=int((x&y).sum())))
    pd.DataFrame(transitions).to_csv(OUT/'episode_transitions.csv',index=False)
    cheap_contrasts=[]
    for ds in ('nusc','av2'):
        units=sorted(b.loc[ds].index.get_level_values('unit').unique());n=len(units)
        draws=np.random.default_rng(0).integers(0,n,(2000,n))
        count=np.array([np.bincount(x,minlength=n) for x in draws],float)
        for severity in ('all','below_hard','at_or_above_hard'):
            e=episodes[(episodes.dataset==ds)&((episodes.severity==severity) if severity!='all' else True)]
            cheap=e[e.pipeline=='cheap'].groupby('unit').missed.agg(['sum','count']).reindex(units,fill_value=0)
            for candidate in ('v6-S','temporal evidence','B2K-A','B2K-A + temporal evidence'):
                cand=e[e.pipeline==candidate].groupby('unit').missed.agg(['sum','count']).reindex(units,fill_value=0)
                assert np.array_equal(cand['count'],cheap['count'])
                delta=(cand['sum']-cheap['sum']).to_numpy();denominator=cheap['count'].to_numpy()
                nn,dd=count@delta,count@denominator
                sample=np.divide(nn,dd,out=np.full(len(nn),np.nan),where=dd!=0)
                lo,hi=np.nanpercentile(sample,[2.5,97.5])
                cheap_contrasts.append(dict(dataset=ds,pipeline=candidate,severity=severity,
                    candidate_minus_cheap_missed_rate=delta.sum()/denominator.sum() if denominator.sum() else np.nan,
                    ci_lo=lo,ci_hi=hi,valid_draws=int(np.isfinite(sample).sum())))
    pd.DataFrame(cheap_contrasts).to_csv(OUT/'missed_vs_cheap.csv',index=False)
    manifest=json.loads((ROOT/'experiments/w1_dev_20261023/fork_manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    plot_results(summary)
    # Export the complete quality/runtime tables together; no winner selection.
    summary.to_csv(OUT/'quality_all_rows.csv',index=False)
    rows=[]
    for ds in ('nusc','av2'):
        dd=summary[summary.dataset==ds].set_index('pipeline')
        for name in ('cheap','v6-S','temporal evidence','B2K-A','B2K-A + temporal evidence'):
            r=dd.loc[name]
            rows.append(f"| {ds} | {name} | {int(r.missed)}/{int(r.episodes)} | {r.false_per_1000:.3f} | {r.insufficient_loss_per_frame:.4f} | {r.G:.4f} | {r.mean_network_px:.5f} |")
    lines=['# Optimization phase — final development handoff','',
        'Main source, frozen weights and K34 are unchanged. All experiments are separate; no candidate is promoted.',
        'The original screen and fixed factorial both cover 152 nuScenes scenes + 300 AV2 logs (29,782 frames).',
        'All baseline/cheap/GT/recovery actions match between independently replayed and reused records.',
        'Intervals throughout are paired units, 2,000 bootstrap draws, seed 0, pointwise; timing repetitions stay within units.','',
        '## Full-development quality','',
        '| Dataset | Pipeline | Missed / required episodes | False/1000 | Insufficient loss/frame | G | Pixels/native |',
        '|---|---|---:|---:|---:|---:|---:|',*rows,'',
        'Every no-refiner/whole-frame/hybrid/uncertainty control is retained in quality_all_rows.csv and the factorial handoff.',
        'Requested-budget controls include historical pixels; guards can overrun. No exact equal-cost selector claim follows.',
        'Paired episode recoveries AND newly introduced misses are listed in episode_transitions.csv.',
        'Miss-rate differences against cheap, including demand severity and paired intervals: missed_vs_cheap.csv.',
        'G uses jerk relative to the preceding cheap action; it is not a candidate-specific closed-loop outcome.','',
        '## Feasibility evidence','',
        '- [NAVSIM score reuse](../NAVSIM_REUSE_20261023/REPORT.md): same candidate scores/actions/field/support, 1,000 tokens × two grids × three repetitions; CPU scope only.',
        '- [Jetson buffer staging](../RUNTIME_20261023/REPORT.md): same actions/costs/crop and execution counts on every paired frame, three repetitions; modest measured runtime reduction.',
        '- [Temporal recovery Jetson port](../TEMPORAL_JETSON_20261023/REPORT.md): four rows, historical inference included, trigger-conditional latency and memory reported.',
        '- [Refiner geometry attribution](../REFINER_GEOMETRY_20261023/REPORT.md): every CPU counterfactual retained; coordinate decoupling is not validated physical geometry.',
        '- [Source/architecture review](../W1_DEV_20261023/OPTIMIZATION_REVIEW.md): duplicate crops and coarse-class switches have little measured headroom; closure/recheck do not establish a robust W1 remedy.','',
        '## Evidence limits and deployment boundary','',
        'Jetson timings use dev prefixes (512 timed frames/dataset/repetition), while quality uses full development units.',
        'VUW/Jetson detector and canvas differences remain; disabled-port parity is not numerical cross-backend parity.',
        'Argoverse-HD has not been used for new tuning/evaluation and its known weakness remains unresolved.',
        'Previously inspected test data informed hypotheses. A new confirmatory claim requires a verified untouched holdout.',
        'No main-code integration, threshold selection, new engine build or publication was performed.','',
        'Figures: development_tradeoff.pdf/.png, navsim_cpu_scope.pdf/.png, runtime_staging.pdf/.png, temporal_latency.pdf/.png.',
        'All deliberate scheduling stops and retained records are documented in the factorial repartition manifest and archived jobs.']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'final_audit.json').write_text(json.dumps(dict(status='PASS',frames=len(a),units=452,
        full_baseline_actions_equal=True,source_hashes_unchanged=True,quality_rows=30,episode_transition_rows=len(transitions)),indent=2)+'\n')
    (OUT/'DONE.json').write_text(json.dumps(dict(status='complete',report=str(OUT/'FINAL_REPORT.md')),indent=2)+'\n')


if __name__=='__main__':main()
