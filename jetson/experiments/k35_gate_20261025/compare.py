"""Action parity of the staged runtime + FAST_RECOVERY helper against the reference runtime (KF.Crops / KB.OneCall +
original Retro helper) on the same frames: every row, every timed frame; every differing frame is listed."""
import argparse
import json
from pathlib import Path
import pandas as pd

ROWS = ('ours', 'ours_b2A', 'ours_b2A_retro', 'ours_b2A_missing_extent_retro')
COUNTERS = ('J', 'px', 'calls', 'crop_engine_exec', 'retro_calls', 'retro_attempted', 'retro_success', 'retro_pixels')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--staged', type=Path, nargs='+', required=True); ap.add_argument('--out', default='')
    a = ap.parse_args()
    r = pd.read_pickle(a.reference/'frames.pkl')
    rep, diffs = [], []
    for st in a.staged:
        s = pd.read_pickle(st/'frames.pkl')
        n = min(len(r), len(s))
        rr, ss = r.iloc[:n].reset_index(drop=True), s.iloc[:n].reset_index(drop=True)
        assert rr[['unit', 'frame', 'a_gt', 'warm']].equals(ss[['unit', 'frame', 'a_gt', 'warm']]), 'frame manifest differs'
        timed = ~rr.warm
        for nm in ROWS:
            c = f'a_{nm}@trt32'
            bad = rr[c] != ss[c]
            rec = {'staged_run': st.name, 'row': nm, 'frames': n, 'timed_frames': int(timed.sum()),
                   'action_diff_timed': int((bad & timed).sum()), 'action_diff_all': int(bad.sum()),
                   'action_parity_rate_timed': float(1 - (bad & timed).sum() / max(int(timed.sum()), 1))}
            for k in COUNTERS:
                ck = f'{k}_{nm}@trt32'
                if ck in rr and ck in ss:
                    rec[f'{k}_diff_timed'] = int(((rr[ck] - ss[ck]).abs() > 1e-8)[timed].sum())
            rep.append(rec)
            for i in bad[bad].index:
                diffs.append({'staged_run': st.name, 'row': nm, 'unit': rr.unit[i], 'frame': int(rr.frame[i]), 'warm': bool(rr.warm[i]),
                              'a_reference': int(rr[c][i]), 'a_staged': int(ss[c][i]), 'a_gt': rr.a_gt[i]})
    out = Path(a.out) if a.out else Path(__file__).resolve().parent/'logs'/f'parity_{a.dataset}'
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rep).to_csv(str(out)+'_rows.csv', index=False)
    pd.DataFrame(diffs, columns=['staged_run', 'row', 'unit', 'frame', 'warm', 'a_reference', 'a_staged', 'a_gt']).to_csv(str(out)+'_frames.csv', index=False)
    summary = {'dataset': a.dataset, 'reference': str(a.reference), 'staged': [str(x) for x in a.staged],
               'action_diff_timed_total': int(sum(x['action_diff_timed'] for x in rep))}
    (Path(str(out)+'.json')).write_text(json.dumps(summary, indent=2)+'\n')
    print(pd.DataFrame(rep).to_string()); print(summary)


if __name__ == '__main__':
    main()
