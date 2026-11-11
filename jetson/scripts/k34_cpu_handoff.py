"""Finalize CPU K34 handoff after all remote gates pass; preserve Jetson results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1];S=R/'results/vuw_summaries/k34';O=R/'docs/paper/handoff/K34'
DS=['nusc','av2','ahd'];LABEL={'nusc':'nuScenes','av2':'AV2','ahd':'Argoverse-HD'}

def interval(x):
    good=np.asarray(x)[np.isfinite(x)];lo,hi=np.percentile(good,[2.5,97.5]) if len(good) else (np.nan,np.nan)
    return {'ci_lo':lo,'ci_hi':hi,'valid_draws':len(good),'bootstrap_n':2000,'seed':0}

def collect():
    out={};sources=[];validation=[]
    for stage,files in {'episodes':['episodes','severity_summary','severity_intervals','severity_contrasts','loss_vs_native'],
                        'operating':['operating_points','operating_intervals','operating_pairwise','continuous_demands'],
                        'coverage':['coverage_frames','coverage_summary','coverage_intervals','class_mapping','stratum_definitions']}.items():
        for ds in DS:assert (S/stage/ds/'DONE.json').exists()
        for name in files:
            parts=[]
            for ds in DS:
                d=pd.read_csv(S/stage/ds/(name+'.csv'));d['dataset']=ds;parts.append(d)
            out[name]=pd.concat(parts,ignore_index=True);out[name].to_csv(O/(name+'.csv'),index=False)
        for ds in DS:
            for name,target in [('source_manifest',sources),('validation',validation)]:
                f=S/stage/ds/(name+'.csv')
                if f.exists():
                    q=pd.read_csv(f);q['stage']=stage;q['dataset']=ds;target.append(q)
    if (S/'selector/source_manifest.csv').exists():sources.append(pd.read_csv(S/'selector/source_manifest.csv').assign(stage='selector'))
    pd.concat(sources,ignore_index=True).to_csv(O/'source_manifest.csv',index=False)
    pd.concat(validation,ignore_index=True).to_csv(O/'validation.csv',index=False)
    # Fine vehicle classes are not recoverable from generic A-HD annotations.
    missing=[]
    for name in ('coverage_summary','coverage_intervals'):
        d=out[name];d['status']='available'
        m=(d.dataset=='ahd')&(d.stratification=='class')&d.stratum.isin(['car','truck','bus'])
        d.loc[m,'status']='not_identifiable_from_annotations'
        d.to_csv(O/(name+'.csv'),index=False)
    for cls in ('car','truck','bus'):
        missing.append({'dataset':'ahd','item':'K34-2','stratification':'class','stratum':cls,
                        'status':'not_identifiable_from_annotations',
                        'reason':'Generic vehicle/large_vehicle annotations do not identify car/truck/bus; zero classified rows is not evidence of class absence.'})
    pd.DataFrame(missing).to_csv(O/'missing_rows.csv',index=False)
    return out

def selector():
    assert (S/'selector/DONE.json').exists()
    f=pd.read_csv(S/'selector/selector_cost_frames.csv');p=pd.read_csv(S/'selector/pdm_historical_frames.csv')
    assert len(f)==2000 and len(p)==2000
    f.to_csv(O/'selector_cost_frames.csv',index=False);p.to_csv(O/'pdm_historical_frames.csv',index=False)
    units=sorted(f.unit.unique());draws=np.random.default_rng(0).integers(0,len(units),(2000,len(units)))
    summary=[];cis=[]
    met=['scene_selector_setup_ms','base_score_ms','field_ms','binding_support_ms','total_ms','base_selector_calls',
         'probe_selector_calls','support_selector_calls','candidate_score_evaluations','candidate_object_matrix_entries','candidate_object_TTC_evaluations']
    for method,data,metrics in [('K31 bottleneck selector',f,met),('PDM-Closed historical K30',p,['total_ms','calls','n_probe','n_support'])]:
        for variant in ['7x7','3x3']:
            q=data[data.variant==variant];row={'method':method,'variant':variant,'frames':len(q),'units':len(units)}
            for metric in metrics:
                samples=[q.loc[q.unit==u,metric].to_numpy(float) for u in units]
                point=np.array([q[metric].mean(),q[metric].quantile(.95)])
                boot=np.array([[v.mean(),np.percentile(v,95)] for draw in draws for v in [np.concatenate([samples[i] for i in draw])]])
                for i,stat in enumerate(['mean','p95']):
                    name=metric+'_'+stat;row[name]=point[i]
                    cis.append({'method':method,'variant':variant,'metric':name,'estimate':point[i],**interval(boot[:,i])})
            summary.append(row)
    pd.DataFrame(summary).to_csv(O/'selector_cost_summary.csv',index=False);pd.DataFrame(cis).to_csv(O/'selector_cost_intervals.csv',index=False)
    return pd.DataFrame(summary),pd.DataFrame(cis)

def main():
    t=collect();sel,selci=selector();out=[]
    def add(s=''):out.append(s)
    add('# K34 CPU analyses — frozen v6-S, post-hoc');add()
    add('K34-1, K34-2 and K34-4 completed on VUW/VUW1 after K34-3 Jetson timing. No model/configuration selection, no detector rerun. Every requested stratum, including empty and unspecified classes, is retained. Intervals are 95% paired scene/log/sequence bootstrap, 2,000 draws, seed 0; pointwise exploratory intervals.');add()
    add('**W1 is qualified, not refuted.** v6-S misses more below-hard-demand episodes than cheap on AV2 and Argoverse-HD (paired intervals exclude zero). High-demand Argoverse-HD episodes also accrue more insufficient-braking loss than cheap; only ten such episodes exist. Dataset-wide insufficient-braking loss versus native + C1 + v3a is inconclusive on all three datasets, which does not establish equivalence. The deployed v6-S point is not dominated by the sampled whole-frame points on nuScenes/AV2, but it is dominated in point estimates on Argoverse-HD.');add()
    add('## K34-1: required episodes and insufficient braking');add()
    add('Evaluated frames: nuScenes 9,181 / 228 scenes; AV2 11,818 / 150 logs, fixed ego mask; Argoverse-HD 15,006 / 24 sequences. Required episodes follow K31 (>= DECELERATE, merge gaps of at most one evaluated frame). Missed onset is capped at episode length. Argoverse-HD retains K31 evaluated-frame adjacency despite gaps in annotation timestamps. Severity is peak annotated demand <3.5 versus >=3.5 m/s². Cheap shortfall bins are {0}, (0,1), [1,3.5), [3.5,8), [8,infinity). Loss is lam_risk × max(required demand − commanded braking, 0)², excluding the separate collision penalty. Episode shares use cheap total decision loss across the whole dataset as denominator.');add()
    add('| Dataset | Peak demand | Episodes | Missed cheap | Missed v6-S | v6-S − cheap [95% CI] |');add('|---|---|---:|---:|---:|---|')
    sv=t['severity_summary'];ct=t['severity_contrasts']
    for ds in DS:
        for severity in ['below_hard','at_or_above_hard']:
            q=sv[(sv.dataset==ds)&(sv.stratification=='severity')&(sv.stratum==severity)].set_index('pipeline')
            c=ct[(ct.dataset==ds)&(ct.stratification=='severity')&(ct.stratum==severity)&(ct.comparator=='cheap')&(ct.metric=='missed_rate')].iloc[0]
            add(f"| {LABEL[ds]} | {severity} | {int(q.loc['v6-S','episodes'])} | {q.loc['cheap','missed_rate']:.3f} | {q.loc['v6-S','missed_rate']:.3f} | {c.v6S_minus:+.3f} [{c.ci_lo:+.3f}, {c.ci_hi:+.3f}] |")
    add();add('All seven pipelines, onset delays, missed/detected episode loss decomposition and marginal/crossed strata: severity_summary.csv, severity_intervals.csv, severity_contrasts.csv; episode-level rows: episodes.csv. Zero-episode rates are NA; zero loss sums remain zero.');add()
    add('For episodes whose peak demand reaches thr_hard, v6-S minus cheap insufficient-braking loss as a share of cheap total dataset loss is -0.1093 [-0.2404, -0.0093] on nuScenes, -0.0075 [-0.0675, +0.0609] on AV2, and +0.0234 [+0.0029, +0.0515] on Argoverse-HD. Thus the negative missed-episode result cannot be dismissed as affecting only mild episodes.');add()
    add('| Dataset | v6-S − native + C1 + v3a insufficient loss per frame [95% CI] |');add('|---|---|')
    for r in t['loss_vs_native'][t['loss_vs_native'].metric=='mean_insufficient_loss_per_frame'].itertuples():
        add(f'| {LABEL[r.dataset]} | {r.v6S_minus:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] |')
    add();add('These are decision-loss components on annotated frames, not closed-loop collision outcomes. Reproduction checks against K31 unstratified episodes and stored K28 costs are in validation.csv.');add()
    add('## K34-1: operating curves');add()
    add('The five-point grid holds thr_hard=3.5 m/s² fixed. Cheap uses cached geometry. Whole-frame C1 spatial geometry is reconstructed from cached boxes/calibration, and its stored TTC is retained: v3a rescales spatial coordinates but never modifies TTC. Every pre-refiner base action is checked against K30 AP-dump records. For v3a, support is recomputed at every threshold and the frozen refiner runs on stored-frame crops; per-object network outputs are reused only for identical inputs. This recovers threshold-specific scenes without C1 or detector inference. continuous_demands.csv stores every demand. Cross-source/backend base discrepancies are reported in validation.csv.');add()
    add('The primary GT required episodes remain fixed at the deployed threshold 1.0. K30 joint-threshold runs (0.8,2.8) and (1.2,4.2), plus their base, are separate labelled points; their original threshold-specific GT is a separate sensitivity protocol. v6-S uses only stored deployed/K30 points. No interpolation between sparse v6-S points or best-threshold selection.');add()
    op=t['operating_points'];pairs=t['operating_pairwise']
    for ds in DS:
        own=op[(op.dataset==ds)&(op.pipeline=='v6-S')&(op.source=='K28_deployed')].iloc[0]
        pp=pairs[pairs.v6S_point==own.point_id].drop_duplicates('whole_frame_point')
        add(f"- {LABEL[ds]}: deployed v6-S false episodes/1,000 frames = {own.false_episodes_per_1000_frames:.3f}, missed rate = {own.missed_rate:.3f}. Among {len(pp)} reported whole-frame rows (20 exact-grid rows plus six separately sourced K30 rows, including duplicate base operating points), {int(pp.whole_frame_point_dominates.sum())} point-estimate dominate v6-S and {int(pp.v6S_point_dominates.sum())} are dominated by it; others trade off or tie. Paired coordinate intervals are in operating_pairwise.csv.")
    add();add('Concrete counterexample to being on/below the whole-frame curves everywhere: on Argoverse-HD, u960 + C1 + v3a at thr_decel=0.8 and fixed thr_hard=3.5 has missed rate 0.155 versus deployed v6-S 0.380 and false episodes/1,000 frames 5.531 versus 6.264. Paired v6-S minus u960 differences are +0.225 [0.114, 0.320] in misses and +0.733 [-0.432, 2.028] in false episodes; the latter interval includes zero, so simultaneous statistical dominance is not established.')
    add();add('Point-estimate dominance is not a simultaneous significance claim. The comparisons establish only relations among measured points, not superiority to an unmeasured continuous envelope. All points and intervals: operating_points.csv, operating_intervals.csv. Figure: operating_curves.pdf/.png.');add()
    add('## K34-2: decisive-object coverage');add()
    add('Population: one decisive annotated object per frame with act_gt >= DECELERATE. Region = union of actual field/support/guard crops accumulated by the final round, checked against stored dec_in_crop_v6S; proposed-but-unexecuted field boxes are excluded. Coverage is box-centre geometry, not detection recall. Dataset-wide decision loss is from the K28 actions. Class mappings are explicit; generic vehicle annotations are not forced into car/truck/bus. Signed closing speed is the negative longitudinal range derivative of annotated tracks, missing when no valid derivative exists; fixed per-dataset tercile cuts and ties are documented in stratum_definitions.csv.');add()
    add('| Dataset | Required frames | Covered [95% CI] | Uncovered conditional mean decision loss [95% CI] |');add('|---|---:|---|---|')
    for ds in DS:
        q=t['coverage_summary'];q=q[(q.dataset==ds)&(q.stratification=='all')].iloc[0]
        ii=t['coverage_intervals'];ii=ii[(ii.dataset==ds)&(ii.stratification=='all')].set_index('metric')
        a,b=ii.loc['coverage'],ii.loc['uncovered_conditional_mean_loss']
        add(f'| {LABEL[ds]} | {q.frames} | {a.estimate:.3f} [{a.ci_lo:.3f}, {a.ci_hi:.3f}] | {b.estimate:.3f} [{b.ci_lo:.3f}, {b.ci_hi:.3f}] |')
    add();add('All class/range/lateral/closing-speed/cheap-detection rows and uncovered loss sums/shares: coverage_summary.csv, coverage_intervals.csv, coverage_frames.csv, class_mapping.csv.');add()
    add('Argoverse-HD car/truck/bus coverage is not separately identifiable from its generic vehicle annotations. These rows are explicitly marked not_identifiable_from_annotations in the summary/interval tables and missing_rows.csv; zero classified rows must not be read as absence of those classes. Cyclist groups combine annotated bicycle/motorcycle classes as shown in class_mapping.csv.');add()
    add('## K34-4: selector cost');add()
    add('Frozen K31 15-candidate bottleneck selector on K20’s exact 1,000-token subset. One CPU thread, five warm-up repeats, alternating grid order; no image/detector inference. Timed scope includes scene construction, selector setup, base scoring, field probes and binding-support extraction; excludes disk loading. Every field/support output is compared to the original implementation, and 7×7 matches stored K31 first-round outputs. Binding support reuses the score matrix and needs zero additional selector evaluations.');add()
    add('| Method | Grid | Mean total ms [95% CI] | p95 total ms |');add('|---|---|---|---:|')
    for r in sel.to_dict('records'):
        c=selci[(selci.method==r['method'])&(selci.variant==r['variant'])&(selci.metric=='total_ms_mean')].iloc[0]
        add(f"| {r['method']} | {r['variant']} | {r['total_ms_mean']:.3f} [{c.ci_lo:.3f}, {c.ci_hi:.3f}] | {r['total_ms_p95']:.3f} |")
    add();add('PDM-Closed comparison uses K30’s measured identical-output fast implementation on the same tokens, not K20’s extrapolated 3×3 runtime. Host/load/protocol differ historically, and planner identities differ: this is context, not a controlled speedup of PDM-Closed. All component times, candidate-score and candidate-object counts, means/p95 and intervals: selector_cost_frames.csv, selector_cost_summary.csv, selector_cost_intervals.csv.');add()
    add('## Reproduction and provenance');add()
    add('Coordinator: python3 scripts/k34_remote_driver.py. Remote CPU scripts: scripts/A_method/k34_episodes.py, k34_replay.py, k34_operating.py, k34_coverage.py (--dataset nusc|av2|ahd); scripts/A_navsim/k34_selector_cost.py in env_navsim.sh. Every job command/host/status is mirrored under results/vuw_summaries/k34/jobs/. Source hashes: source_manifest.csv and remote per-unit replay metadata. Source/code remain frozen. Remote outputs: results/k34/; local summary mirror: results/vuw_summaries/k34/. Jetson result and parity caveats remain in JETSON_REPORT.md.');add()
    add('Final audit (2026-10-23): all 36,005 evaluated frames across 402 units are present. CPU replay cheap and whole-frame C1 actions, with and without v3a, match original K28 deployed actions on every frame. Source hashes are unchanged. Handoff checks cover 378 severity rows, 132 operating rows, 60 coverage rows, 1,000 NAVSIM tokens and 12,519 Jetson timing rows. Evidence: cpu_final_audit.json and handoff_audit.json. Three unavailable A-HD fine-class rows are retained in missing_rows.csv.');add()
    (O/'CPU_REPORT.md').write_text('\n'.join(out)+'\n')
    jet=O/'JETSON_REPORT.md'
    if not jet.exists():jet.write_text((O/'REPORT.md').read_text())
    jettext='\n'.join('**K34-3 complete.** Every timing repetition is retained; CPU analyses are in CPU_REPORT.md.'
                     if line.startswith('**K34-3 complete;') else
                     'CPU analyses are now in CPU_REPORT.md. This timing result alone does not resolve W1 or establish a safety improvement.'
                     if line.startswith('K34-1 severity/loss') else line for line in jet.read_text().splitlines())+'\n'
    jet.write_text(jettext)
    (O/'REPORT.md').write_text('# K34 — completed post-hoc analyses of frozen v6-S\n\nAll four items completed; no configuration changes or test-driven selection. Paired unit bootstrap: 2,000, seed 0.\n\n'+ '\n'.join(out[2:])+'\n\n'+jettext.replace('# K34 — post-hoc frozen-v6-S analyses','# K34-3 — Jetson timing',1))
    (O/'STATUS.md').write_text('# K34 status\n\nK34-1/2/3/4 complete. All data and limitations: REPORT.md, CPU_REPORT.md, JETSON_REPORT.md and CSVs. Source jobs and logs: logs/k34cpu/state.json and results/vuw_summaries/k34/jobs/. No detector/model rerun or tuning.\n')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
    names=['cheap','u960 + C1 without refiner','native + C1 without refiner','u960 + C1 + v3a','native + C1 + v3a']
    for ax,ds in zip(axes,DS):
        for name in names:
            q=op[(op.dataset==ds)&(op.pipeline==name)&op.source.str.startswith('CPU')].sort_values('thr_decel_mps2')
            ax.plot(q.false_episodes_per_1000_frames,q.missed_rate,'.-',label=name)
        q=op[(op.dataset==ds)&(op.pipeline=='v6-S')&(op.gt_protocol=='fixed_deployed_GT')]
        ax.scatter(q.false_episodes_per_1000_frames,q.missed_rate,marker='*',s=70,c='black',label='v6-S stored points',zorder=5)
        ax.set_title(LABEL[ds]);ax.set_xlabel('False episodes / 1,000 frames');ax.grid(alpha=.2)
    axes[0].set_ylabel('Missed required-episode rate');h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=3,fontsize=8);fig.tight_layout(rect=[0,.16,1,1])
    fig.savefig(O/'operating_curves.pdf');fig.savefig(O/'operating_curves.png',dpi=180);plt.close(fig)
    for path,heading in [('03_results.md','K34 post-hoc CPU analyses'),('05_limitations_and_negative.md','K34 CPU caveats')]:
        p=R/'docs/paper'/path;s=p.read_text();marker='<!-- K34_CPU_AUTO -->'
        if marker in s:s=s.split(marker)[0].rstrip()+'\n'
        s+='\n'+marker+'\n- **'+heading+' (complete).** Source: handoff/K34/REPORT.md and all CSVs. Severity/operating analyses retain every pipeline and threshold; coverage uses actual verified crops, with unknown classes retained. Selector/PDM timings compare different planners and historical hosts.\n'
        s+='  - W1 remains: below-hard-demand missed episodes are higher than cheap on AV2 and Argoverse-HD. On Argoverse-HD, high-demand episode loss is also higher: +0.0234 [0.0029, 0.0515] of cheap total dataset loss (ten episodes).\n'
        s+='  - Deployed v6-S is not dominated by sampled whole-frame points on nuScenes/AV2, but u960 + C1 + v3a at threshold 0.8 dominates its Argoverse-HD point estimate; false-alarm difference CI includes zero. Fine car/truck/bus coverage is unavailable on Argoverse-HD generic annotations.\n'
        for ds in DS:
            r=t['loss_vs_native'];r=r[(r.dataset==ds)&(r.metric=='mean_insufficient_loss_per_frame')].iloc[0]
            s+=f'  - {LABEL[ds]} v6-S minus native insufficient-braking loss/frame: {r.v6S_minus:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}].\n'
        s=s.replace('K34-1/2/4 remain pending; this timing result does not resolve the W1 missed-episode criticism.',
                    'K34 CPU analyses are now complete below; the timing result alone does not resolve W1.')
        s=s.replace('K34-1 is still required to address W1.','K34-1 findings and qualifications are recorded below.')
        p.write_text(s)
    p=R/'docs/paper/06_pending.md';lines=p.read_text().splitlines()
    lines=[('| K34 (new; earlier K34 merged into K33): frozen-v6-S diagnostics | All items 1–4 DONE; every row and caveat reported, no tuning. Jetson VUW parity is not exact. | handoff/K34/REPORT.md + CSVs |' if x.startswith('| K34 (new;') else x) for x in lines];p.write_text('\n'.join(lines)+'\n')
    print('K34 CPU HANDOFF COMPLETE',O/'REPORT.md',flush=True)

if __name__=='__main__':main()
