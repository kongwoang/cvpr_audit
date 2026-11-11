"""K35-6 (post-hoc, CPU): a reproducible selection rule for the qualitative panel figure.
Rule: for each (dataset, category), in the fixed category order below, the FIRST frame in time order (units sorted by
name, then frame index) that satisfies the category's formal definition and was not already selected for an earlier
category, computed from stored K28 p3 test rows only. Categories are those of
the existing panel set (K28 phase 3 §17: K8 categories and K15 on nuScenes / AV2; K28 §13 failure categories on all three
splits), formalised from their stated reasons. Object-level conditions use the per-pipeline found arrays over the frame's
relevant (demand >= tau1) annotated objects, which share one order across pipelines.
  K8 spurious_removed       act_gt = KEEP, cheap >= DECELERATE, v6-S = KEEP, u960 + C1 + v3a = KEEP
  K8 range_corrected        cheap finds a relevant object; cheap and v6-S without refiner differ from act_gt; v6-S
                            (with v3a) equals act_gt
  K8 missed_recovered_field a relevant object missed by cheap, found by u960 + C1, found by v6-S, NOT found by v6-S's
                            support-only ablation (so6S), and v6-S equals act_gt
  K8 failure                v6-S differs from act_gt and u960 + C1 + v3a equals it
  K15 missed recovered      a relevant object missed by cheap and by unc2@S + C1 and found by v6-S; v6-S equals act_gt
  failure: <cat>            K28 §13 (k28_p3_analyze.failures, unchanged): J(v6-S) > J(u960 + C1 + v3a), then
                            unnecessary action / missed / under-reaction / other in that order
Also checks each frame of the existing capture list (results/k28/p3/capture_frames.csv) against these definitions.
usage: k35_k6.py   -> results/k35/k6/{capture_frames.csv, qualifying_counts.csv, existing_check.csv}
"""
import numpy as np
import pandas as pd
from k34_common import B, load_base

OUT = B / 'results/k35/k6'
FAIL = {"unnecessary action (ours above the required action)": "failure: unnecessary action",
        "missed: a relevant object u960 finds is not in ours' scene": "failure: missed",
        "under-reaction with the relevant objects in ours' scene (range / ttc)": "failure: under-reaction",
        "other (same action class as required, switching cost)": "failure: other"}


def arr(x, n):
    return np.asarray(x, bool) if isinstance(x, (list, np.ndarray)) and len(x) == n else np.zeros(n, bool)


def categories(r):
    out = []
    g, ac, av, avn, af = int(r.act_gt), int(r.a_cheap), int(r['a_v6S|v3a']), int(r['a_v6S|none']), int(r['a_wf960+C1|v3a'])
    n = int(r.n_relevant)
    fc, fv, fu = arr(r.get('found_cheap'), n), arr(r.get('found_v6S'), n), arr(r.get('found_wf960+C1'), n)
    fso, func = arr(r.get('found_so6S'), n), arr(r.get('found_unc2@S+C1'), n)
    if g == 0 and ac >= 1 and av == 0 and af == 0:
        out.append('K8 spurious_removed')
    if n and fc.any() and ac != g and avn != g and av == g:
        out.append('K8 range_corrected')
    if n and (~fc & fu & fv & ~fso).any() and av == g:
        out.append('K8 missed_recovered_field')
    if av != g and af == g:
        out.append('K8 failure')
    if n and (~fc & ~func & fv).any() and av == g:
        out.append('K15 missed recovered')
    if float(r['J_v6S|v3a']) > float(r['J_wf960+C1|v3a']):
        if av > g:
            out.append('failure: unnecessary action')
        elif n and (fu & ~fv).any():
            out.append('failure: missed')
        elif av < g:
            out.append('failure: under-reaction')
        else:
            out.append('failure: other')
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel, counts, chk = [], [], []
    ex = pd.read_csv(B / 'results/k28/p3/capture_frames.csv')
    for ds in ('nusc', 'av2', 'ahd'):
        d = load_base(ds, []).sort_values(['unit', 'frame'], kind='stable').reset_index(drop=True)
        d['cats'] = [categories(r) for _, r in d.iterrows()]
        allowed = [c for c in ['K8 spurious_removed', 'K8 range_corrected', 'K8 missed_recovered_field', 'K8 failure', 'K15 missed recovered']
                   if ds != 'ahd'] + list(FAIL.values())
        for c in allowed:
            m = d.cats.apply(lambda x: c in x)
            counts.append({'dataset': ds, 'category': c, 'qualifying_frames': int(m.sum()), 'qualifying_units': int(d.unit[m].nunique())})
            taken = {(x['unit'], x['frame']) for x in sel if x['dataset'] == ds}
            m = m & ~pd.Series([(u, int(f)) in taken for u, f in zip(d.unit, d.frame)], index=d.index)
            if m.any():
                r = d[m].iloc[0]
                sel.append({'dataset': ds, 'unit': r.unit, 'frame': int(r.frame), 'category': c,
                            'reason': f'first in time order (unit name, frame) of {int(m.sum())} qualifying frames; act_gt {int(r.act_gt)}, '
                                      f'cheap {int(r.a_cheap)}, v6-S {int(r["a_v6S|v3a"])}, v6-S no ref {int(r["a_v6S|none"])}, u960+C1 {int(r["a_wf960+C1|v3a"])}'})
        ix = d.set_index(['unit', 'frame'])
        for _, e in ex[ex.dataset == ds].iterrows():
            have = ix.loc[(e.unit, int(e.frame)), 'cats'] if (e.unit, int(e.frame)) in ix.index else None
            cat = e.category
            chk.append({'dataset': ds, 'unit': e.unit, 'frame': int(e.frame), 'existing_category': e.category,
                        'satisfies_formal_definition': bool(have is not None and cat in have),
                        'formal_categories_of_frame': ';'.join(have) if have is not None else 'frame not evaluated'})
    pd.DataFrame(sel).to_csv(OUT / 'capture_frames.csv', index=False)
    pd.DataFrame(counts).to_csv(OUT / 'qualifying_counts.csv', index=False)
    pd.DataFrame(chk).to_csv(OUT / 'existing_check.csv', index=False)
    print('K35-6 SELECTION DONE', len(sel), 'frames')


if __name__ == '__main__':
    main()
