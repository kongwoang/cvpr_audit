"""Describe decision/pixel/deadline trade-offs of all measured Jetson pilot rows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(os.environ['B']) / 'results/k40r/vendor'))
from analysis.common import Bootstrap


def ci(bt, numerator, denominator):
    n, d = bt.sums(numerator), bt.sums(denominator)
    den = d[bt.idx].sum(axis=1)
    vals = np.divide(n[bt.idx].sum(axis=1), den, out=np.full(len(den), np.nan), where=den != 0)
    low, high = np.nanpercentile(vals, [2.5, 97.5])
    return float(low), float(high)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--input', type=Path, required=True)
    args = ap.parse_args()
    sp = args.input / 'state.recovered.json'
    if not sp.exists(): sp = args.input / 'state.json'
    state = json.loads(sp.read_text())
    assert state['status'] in ('complete', 'recovered_complete')
    groups = {}
    for r in state['runs']:
        if r['stage'] != 'timing': continue
        d = pd.read_pickle(args.input / 'raw' / Path(r['run']).name / 'frames.pkl')
        d = d[~d.warm].copy(); d['rep'] = r['rep']
        assert len(d) == 512 and d.production_timing_valid.all()
        groups.setdefault((r['dataset'], r['mode']), []).append(d)
    rows, deadlines = [], []
    target = 'u960+c1comp@trt32'
    for (ds, mode), repeats in groups.items():
        assert len(repeats) == 3
        d = pd.concat(repeats).set_index(['rep', 'unit', 'frame']).sort_index()
        bt = Bootstrap(d.index.get_level_values('unit'))
        cheap = d.J_cheap.to_numpy(float)
        assert cheap.sum() > 0
        for col in d:
            if not col.startswith('ms_'): continue
            row = col[3:]; timing = d[col].to_numpy(float)
            action = d['a_' + row].to_numpy(float); truth = d.act_gt.to_numpy(float)
            loss = d['J_' + row].to_numpy(float)
            lo, hi = ci(bt, cheap - loss, cheap)
            delta = d['J_' + target].to_numpy(float) - loss
            dl, dh = ci(bt, delta, cheap)
            delta_desp = d['J_ours@trt32'].to_numpy(float) - loss
            desl, desh = ci(bt, delta_desp, cheap)
            values = dict(dataset=ds, mode=mode, row=row, frames_x_reps=len(d), unique_frames=512,
                          units=len(bt.units), G=float((cheap-loss).sum()/cheap.sum()), G_lo=lo, G_hi=hi,
                          delta_G_vs_u960_refined=float(delta.sum()/cheap.sum()), delta_G_lo=dl, delta_G_hi=dh,
                          delta_G_vs_full_desp=float(delta_desp.sum()/cheap.sum()), delta_G_vs_desp_lo=desl, delta_G_vs_desp_hi=desh,
                          action_agreement_u960_refined=float(np.mean(action == d['a_'+target].to_numpy())),
                          action_agreement_full_desp=float(np.mean(action == d['a_ours@trt32'].to_numpy())),
                          missed_braking_frames_per_rep=float(np.sum((action == 0)&(truth > 0))/3),
                          false_braking_frames_per_rep=float(np.sum((action > 0)&(truth == 0))/3),
                          insufficient_action_frames_per_rep=float(np.sum(action < truth)/3),
                          excess_action_frames_per_rep=float(np.sum(action > truth)/3),
                          tensor_pixels_native_fraction_mean=float(d['px_'+row].mean()),
                          actual_engine_executions_mean=float(d['execs_'+row].mean()) if 'execs_'+row in d else np.nan,
                          reciprocal_mean_latency_fps=float(1000/timing.mean()),
                          energy_j_per_frame=np.nan, energy_reason='board telemetry not aligned to individual pipeline intervals',
                          scope='dev prefix; agreement against refined u960 policy, not canonical detector-target fidelity; no live deadline action policy')
            rows.append(values)
            for D in (33.3, 50., 66.7, 100.):
                met = (timing <= D).astype(float)
                low, high = ci(bt, met, np.ones(len(met)))
                deadlines.append(dict(dataset=ds,mode=mode,row=row,deadline_ms=D,success_fraction=float(met.mean()),success_lo=low,success_hi=high,late_frames_x_reps=int((timing>D).sum()),frames_x_reps=len(d),scope='observed completion only; no hold-last replay or live stopping'))
    out = args.input / 'summary'
    pd.DataFrame(rows).to_csv(out / 'QUALITY_PILOT.csv', index=False)
    pd.DataFrame(deadlines).to_csv(out / 'DEADLINES_PILOT.csv', index=False)
    (out / 'quality_protocol.json').write_text(json.dumps(dict(script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),bootstrap='paired2000 scene/log resamples seed0; all repetitions kept together',G='sum(J_cheap-J_row)/sum(J_cheap), frozen shared-previous-cheap loss',agreement_reference=target,canonical_target_fidelity_available=False,counts='binary missed/false relative to discrete act_gt; ordinal insufficient/excess reported separately',throughput='reciprocal mean latency only; not actual pipelined throughput',deadline='completion probability; no delayed-action quality claim'),indent=2)+'\n')
    frame = pd.DataFrame(rows)
    print(frame[(frame['mode']=='baseline') & frame.row.isin(['ours@trt32','ours_b2A@trt32','ours_b1A@trt32',target])][['dataset','row','G','delta_G_vs_u960_refined','delta_G_lo','delta_G_hi','tensor_pixels_native_fraction_mean','missed_braking_frames_per_rep','false_braking_frames_per_rep']].to_string(index=False))


if __name__ == '__main__': main()
