"""CPU-only paired summary of completed JP01 Jetson runs; no detector imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

B = Path(os.environ['B'])
sys.path.insert(0, str(B / 'results/k40r/vendor'))
from analysis.common import Bootstrap


def weighted_quantiles(values, inv, idx, q):
    order = np.argsort(values, kind='stable'); values = values[order]; inv = inv[order]
    result = []
    for ids in idx:
        counts = np.bincount(ids, minlength=int(inv.max()) + 1)
        cumulative = np.cumsum(counts[inv]); n = cumulative[-1]
        at = (n - 1) * q; low = int(np.floor(at)); high = int(np.ceil(at)); frac = at - low
        a = values[np.searchsorted(cumulative, low + 1)]
        z = values[np.searchsorted(cumulative, high + 1)]
        result.append(a + (z - a) * frac)
    return np.asarray(result)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--check-counters', action='store_true', help='Also check actual execution/refinement counters in clean runs')
    args = ap.parse_args(); out = args.input / 'summary'; out.mkdir(exist_ok=True)
    state_path = args.input / 'state.recovered.json'
    if not state_path.exists():
        state_path = args.input / 'state.json'
    state = json.loads(state_path.read_text())
    assert state['status'] in ('complete', 'recovered_complete') and not state.get('restore_errors'), state
    assert state.get('window_released'), state
    groups = {}; profile = []; sources = []; encoder_checks = {}
    for r in state['runs']:
        if args.check_counters and r['stage'] == 'parity' and r['mode'] == 'audit':
            debug_path = args.input / 'raw' / Path(r['run']).name / 'frames.pkl'
            debug_frame = pd.read_pickle(debug_path)
            assert len(debug_frame) == 712 and not debug_frame.production_timing_valid.any()
            checks = debug_frame.debug_c1_graph_checks.to_numpy()
            assert checks[-1] > 0 and np.all(np.diff(checks) >= 0)
            encoder_checks[r['dataset']] = int(checks[-1])
        if r['stage'] not in ('timing', 'profile'):
            continue
        p = args.input / 'raw' / Path(r['run']).name / 'frames.pkl'
        d = pd.read_pickle(p)
        sources.append(dict(path=str(p), md5=hashlib.md5(p.read_bytes()).hexdigest(), **{k:r[k] for k in ('dataset','mode','rep','stage')}))
        assert len(d) == 712 and int(d.warm.sum()) == 200
        d = d[~d.warm].copy(); d['rep'] = r['rep']; d.unit = d.unit.astype(str)
        assert len(d) == 512
        if r['stage'] == 'profile':
            for c in d:
                if c.startswith('bd_') and c.endswith('_ours@trt32'):
                    x = d[c].fillna(0).to_numpy(float)
                    profile.append(dict(dataset=r['dataset'],stage=c,frames=len(d),units=d.unit.nunique(),mean_ms=x.mean(),p95_ms=np.quantile(x,.95),p99_ms=np.quantile(x,.99),scope='instrumented legacy crop-call breakdown; not true guard-depth attribution'))
            continue
        assert d.production_timing_valid.all()
        groups.setdefault((r['dataset'], r['mode']), []).append(d)
    rows = []; per_rep = []; parity = []
    for ds in ('nusc_val', 'av2_dev'):
        b, c = [pd.concat(groups[(ds, mode)]).set_index(['rep','unit','frame']).sort_index() for mode in ('baseline','candidate')]
        assert len(groups[(ds, 'baseline')]) == len(groups[(ds, 'candidate')]) == 3
        assert b.index.equals(c.index) and not b.index.duplicated().any()
        bt = Bootstrap(b.index.get_level_values('unit'))
        prefixes = ('a_','J_','px_','rounds_','n_crops_')
        if args.check_counters:
            prefixes += ('execs_', 'calls_', 'crop_engine_exec_', 'draft_calls_', 'n_refined_', 'bound_hit_', 'ncrops_')
        for col in b:
            if col.startswith(prefixes):
                assert col in c
                x,y = b[col].to_numpy(), c[col].to_numpy()
                mismatches = int(np.sum(~((x == y) | (pd.isna(x) & pd.isna(y)))))
                parity.append(dict(dataset=ds,endpoint=col,frames_x_reps=len(b),mismatches=mismatches))
                assert mismatches == 0, (ds,col,mismatches)
        for col in b:
            if not col.startswith('ms_') or col.endswith('_noref'):
                continue
            name = col[3:]; x=b[col].to_numpy(float); y=c[col].to_numpy(float)
            for rep in (1,2,3):
                for mode,frame in [('baseline',b),('candidate',c)]:
                    z=frame.xs(rep,level='rep')[col].to_numpy(float)
                    per_rep.append(dict(dataset=ds,row=name,mode=mode,rep=rep,mean_ms=z.mean(),p95_ms=np.quantile(z,.95),p99_ms=np.quantile(z,.99)))
            bx=bt.sums(x); cy=bt.sums(y); n=bt.sums(np.ones(len(x))); den=n[bt.idx].sum(1)
            for metric,qa in [('mean',None),('p50',.5),('p90',.9),('p95',.95),('p99',.99),('max',1.)]:
                xv=x.mean() if qa is None else np.quantile(x,qa)
                yv=y.mean() if qa is None else np.quantile(y,qa)
                if qa is None:
                    xb=bx[bt.idx].sum(1)/den; yc=cy[bt.idx].sum(1)/den
                else:
                    xb=weighted_quantiles(x,bt.inv,bt.idx,qa);yc=weighted_quantiles(y,bt.inv,bt.idx,qa)
                delta=yc-xb;lo,hi=np.percentile(delta,[2.5,97.5]); gain=100*(xb-yc)/xb;gl,gh=np.percentile(gain,[2.5,97.5])
                rows.append(dict(dataset=ds,row=name,metric=metric,baseline_ms=xv,candidate_ms=yv,delta_candidate_minus_baseline_ms=yv-xv,delta_lo_ms=lo,delta_hi_ms=hi,reduction_percent=100*(xv-yv)/xv,reduction_lo_percent=gl,reduction_hi_percent=gh,frames_x_reps=len(x),unique_frames=len(x)//3,units=len(bt.units),repetitions=3,scope='dev feasibility pilot; not full test benchmark'))
    d=pd.DataFrame(rows);d.to_csv(out/'BENCHMARK_PILOT.csv',index=False)
    pd.DataFrame(per_rep).to_csv(out/'REPETITIONS_PILOT.csv',index=False)
    pd.DataFrame(parity).to_csv(out/'TIMING_PARITY.csv',index=False)
    pd.DataFrame(profile, columns=['dataset','stage','frames','units','mean_ms','p95_ms','p99_ms','scope']).to_csv(out/'PROFILE_PILOT.csv',index=False)
    pd.DataFrame(sources).to_csv(out/'sources.csv',index=False)
    decision=[]
    for ds in ('nusc_val','av2_dev'):
        own=d[(d.dataset==ds)&(d.row=='ours@trt32')&(d.metric=='p95')].iloc[0]
        whole=d[(d.dataset==ds)&d.row.isin(['u512+c1comp@trt32','u640+c1comp@trt32','u960+c1comp@trt32'])&d.metric.isin(['mean','p95'])]
        slow=bool((whole.delta_lo_ms>0).any())
        decision.append(dict(dataset=ds,p95_reduction_percent=float(own.reduction_percent),p95_delta_hi_ms=float(own.delta_hi_ms),wholeframe_significant_slowdown=slow,promising=bool(own.reduction_percent>=5 and own.delta_hi_ms<0 and not slow)))
    (out/'feasibility.json').write_text(json.dumps(dict(decisions=decision,script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),window_status=state['status'],recovery=state.get('recovery'),instrumented_profile_available=bool(profile),encoder_graph_exact_checks_by_dataset=encoder_checks,statistic='paired 2000 unit bootstrap seed0; candidate minus baseline; all repetitions nested in unit',multiplicity='exploratory pilot intervals; no confirmatory superiority claim',energy='telemetry collected but no aligned per-row energy claim'),indent=2)+'\n')
    print(d[(d.row.isin(['ours@trt32','u960+c1comp@trt32'])) & d.metric.isin(['mean','p95','p99'])].to_string(index=False))
    print(decision)


if __name__=='__main__':main()
