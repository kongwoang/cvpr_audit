"""K35-0 (b), GT only: units, labelled frames and K31 required episodes of the Argoverse 1.1 3D-tracking TRAIN logs, with
the Argoverse-HD evaluation protocol of the frozen test split (k24_test.units 'ahd' path, applied to train.json's
sequences): every ring_front_center frame of a sequence; a frame is evaluated iff it has 3D labels (AV1Log.gt_frames);
annotated objects with projected height >= 10 px; required action from the annotated geometry and ego speed with the
deployed planner parameters (tau1 1.0, tau2 3.5 m/s^2); required episodes = K31 runs of act_gt >= DECELERATE over the
evaluated frames (k31_episodes.runs); severity = peak annotated demand < / >= tau2. No pipeline output is used.
usage: k35_k0_episodes.py [train|val]   -> results/k35/k0/av1_<split>_{frames,units,episodes}.csv + summary.json
"""
import json

import numpy as np
import pandas as pd
from rap import planner as P
from vtf import av1 as A1
from k31_episodes import runs
from k34_common import B

HD = B / 'datasets/argoverse_hd/Argoverse-HD/annotations'
AV1 = B / 'datasets/argoverse1/argoverse-tracking'
OUT = B / 'results/k35/k0'


def main():
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else 'train'          # 'val' = protocol check against K31 / K34 (24 / 15,006 / 71)
    d = json.loads((HD / f'{split}.json').read_text())
    by = {}
    for im in d['images']:
        by.setdefault(im['sid'], []).append(im)
    roots = {p.name: p.parent for s in (('train1', 'train2', 'train3', 'train4') if split == 'train' else ('val',)) for p in (AV1 / s).glob('*') if p.is_dir()}
    pp = P.PlannerParams()
    frows, urows, erows = [], [], []
    for sid, ims in sorted(by.items()):
        ims = sorted(ims, key=lambda x: x['fid'])
        log = d['seq_dirs'][sid].split('/')[1]
        if log not in roots:
            urows.append({'sid': sid, 'log': log, 'status': 'not downloaded'}); continue
        ld = roots[log] / log
        complete = ((ld / 'vehicle_calibration_info.json').exists() and any((ld / 'poses').glob('*.json'))
                    and any((ld / 'per_sweep_annotations_amodal').glob('*.json'))
                    and all((ld / 'ring_front_center' / i['name']).exists() for i in ims))
        if not complete:
            urows.append({'sid': sid, 'log': log, 'status': 'incomplete (extraction running)'}); continue
        L = A1.AV1Log(roots[log], log, [i['name'] for i in ims])
        geom, speeds, gtf = L.geometry(), L.speeds(), L.gt_frames()
        seq = []
        for fr in range(len(ims)):
            if fr not in gtf:
                continue
            g = geom[geom['frame'] == fr] if len(geom) else geom
            g = g[(g['y2'] - g['y1']) >= 10] if len(g) else g
            v = float(speeds[fr])
            a_gt = float(P.required_decel(g['long_near'], g['lat_min'], g['lat_max'], g['ttc'], v, pp)[0]) if len(g) else 0.0
            seq.append({'sid': sid, 'log': log, 'frame': fr, 'v': v, 'a_gt': a_gt, 'act_gt': P.discrete_action(a_gt, pp), 'n_objects': len(g)})
        q = pd.DataFrame(seq)
        frows += seq
        eps = runs(q.act_gt.to_numpy(int) >= 1) if len(q) else []
        for ei, (s, n) in enumerate(eps):
            peak = float(q.a_gt.iloc[s:s + n].max())
            erows.append({'sid': sid, 'log': log, 'episode': ei, 'start_frame': int(q.frame.iloc[s]), 'frames': n, 'peak_mps2': peak,
                          'severity': 'at_or_above_hard' if peak >= pp.thr_hard else 'below_hard'})
        urows.append({'sid': sid, 'log': log, 'status': 'ok', 'images': len(ims), 'labelled_frames': len(q),
                      'required_frames': int((q.act_gt >= 1).sum()) if len(q) else 0, 'episodes': len(eps)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frows).to_csv(OUT / f'av1_{split}_frames.csv', index=False)
    U = pd.DataFrame(urows); U.to_csv(OUT / f'av1_{split}_units.csv', index=False)
    E = pd.DataFrame(erows); E.to_csv(OUT / f'av1_{split}_episodes.csv', index=False)
    ok = U[U.status == 'ok']
    s = {'sequences_in_train_json': len(by), 'units_ok': len(ok), 'units_missing': int((U.status != 'ok').sum()),
         'labelled_frames': int(ok.labelled_frames.sum()) if len(ok) else 0, 'required_episodes': len(E),
         'episodes_by_severity': E.severity.value_counts().to_dict() if len(E) else {},
         'units_with_episodes': int((ok.episodes > 0).sum()) if len(ok) else 0}
    json.dump(s, open(OUT / f'av1_{split}_summary.json', 'w'), indent=1)
    print('K35-0 AV1 GT', split, s)


if __name__ == '__main__':
    main()
