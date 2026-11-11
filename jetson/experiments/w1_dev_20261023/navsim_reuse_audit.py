"""CPU-only, isolated exact-score reuse experiment on K34's fixed 1,000 tokens.

K31/K34 and all frozen sources remain unchanged. One paired timing repetition;
this is a post-hoc implementation check, not Jetson or end-to-end latency.
"""
import hashlib
import json
import os
import pickle
import socket
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

B = Path(os.environ['B'])
sys.path.insert(0, str(B / 'scripts/A_navsim'))
import k31_nav_p2 as K
import k34_selector_cost as C

OUT = B / 'experiments/w1_dev_20261023/reports/navsim_reuse_audit'


def evaluate(fr, rec, cl, gz, priors, grid, reuse):
    start = time.perf_counter()
    cam = K.NP.Cam(*rec['cam'])
    ann = fr['anns']
    base, per = K.branch_objects(ann, rec['cheap'], cam, gz, priors)
    sc = K.Scene(ann, base, zip(per, np.asarray(rec['cheap']['xyxy'], float).reshape(-1, 4)))
    sel = K.Selector(cl if isinstance(cl, np.ndarray) and len(cl) >= 2 else None,
                     float(np.hypot(*np.asarray(fr['ego_dynamic_state'], float)[:2])))
    s, hm = sel.score(sc.o)
    action = int(np.argmax(s))
    prb = C.probes(cam, gz, priors, grid)
    if reuse:
        hp = sel.h(np.asarray([p[0] for p in prb], float).reshape(-1, 7))
        scores = np.minimum(s[:, None], hp).T
    else:
        scores = np.asarray([sel.score(np.vstack([sc.o, row]))[0] for row, _ in prb]).reshape(-1, len(s))
    ids = np.flatnonzero(np.argmax(scores, axis=1) != action) if len(scores) else np.zeros(0, int)
    field = np.asarray([prb[i][1] for i in ids])
    others = np.arange(len(s)) != action
    support = np.asarray([sc.dbox[i] for i in range(len(sc.o))
                          if sc.dbox[i] is not None and (hm[others, i] <= s[action]).any()])
    elapsed = 1000 * (time.perf_counter() - start)
    return dict(ms=elapsed, action=action, field=field, support=support, scores=scores,
                n_objects=len(sc.o), n_probes=len(prb), matrix_entries=15 * (
                    len(sc.o) + len(prb) if reuse else len(sc.o) + len(prb) * (len(sc.o) + 1)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [B/'scripts/A_navsim/k31_nav_p2.py', B/'scripts/A_navsim/k34_selector_cost.py']
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest = B / 'results/k34/selector/token_manifest.csv'
    todo = pd.read_csv(manifest).sort_values(['unit', 'token'])
    assert len(todo) == 1000 and not todo.duplicated(['unit', 'token']).any()
    gz = float((K.EXP / 'dets/ground_z.txt').read_text())
    priors = pickle.load((K.EXP / 'dets/size_priors.pkl').open('rb'))
    rows, warmed, number = [], False, 0
    for unit, tasks in todo.groupby('unit', sort=True):
        dets = pickle.load((K.EXP / 'dets_k30b' / (unit+'.pkl')).open('rb'))
        centers = pickle.load((K.EXP / 'k30_center' / (unit+'.pkl')).open('rb'))
        frames = {r['token']: r for r in pickle.load((K.DATA / 'navsim_logs/test' / (unit+'.pkl')).open('rb'))}
        old = {r['token']: r for r in pickle.load((B / 'results/k31/p2' / (unit+'.pkl')).open('rb'))}
        for token in tasks.token:
            args = (frames[token], dets[token], centers.get(token), gz, priors)
            if not warmed:
                for _ in range(5):
                    for grid in ('7x7', '3x3'):
                        evaluate(*args, grid, False); evaluate(*args, grid, True)
                warmed = True
            for grid in (('7x7', '3x3') if number % 2 else ('3x3', '7x7')):
                pair = {}
                for fast in ((False, True) if number % 2 else (True, False)):
                    pair[fast] = evaluate(*args, grid, fast)
                ref, opt = pair[False], pair[True]
                assert ref['action'] == opt['action'], (unit, token, grid, 'action')
                for key in ('field', 'support', 'scores'):
                    assert np.array_equal(ref[key], opt[key], equal_nan=True), (unit, token, grid, key)
                if grid == '7x7':
                    assert ref['action'] == old[token]['pi_cheap']
                    assert len(ref['field']) == old[token]['field_cells_r1']
                    assert len(ref['support']) == old[token]['support_r1']
                rows.append(dict(unit=unit, token=token, grid=grid, reference_ms=ref['ms'], reuse_ms=opt['ms'],
                    n_objects=ref['n_objects'], n_probes=ref['n_probes'],
                    reference_matrix_entries=ref['matrix_entries'], reuse_matrix_entries=opt['matrix_entries'],
                    all_candidate_scores_equal=True, action_equal=True, field_equal=True, support_equal=True))
            number += 1
        pd.DataFrame(rows).to_csv(OUT / 'frames.partial.csv', index=False)
        print(f'{number}/1000 tokens', flush=True)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    assert before == after
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'frames.csv', index=False)
    summary = []
    for grid, dd in d.groupby('grid'):
        for method in ('reference', 'reuse'):
            x = dd[method+'_ms']
            summary.append(dict(grid=grid, method=method, tokens=len(dd), mean_ms=x.mean(),
                p95_ms=x.quantile(.95), p99_ms=x.quantile(.99),
                mean_matrix_entries=dd[method+'_matrix_entries'].mean()))
    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    metadata = dict(host=socket.gethostname(), tokens=number, grids=2, repetitions=1,
        warmup_repeats=5, load_average=os.getloadavg(), frozen_sources=before,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        omp_threads=os.environ.get('OMP_NUM_THREADS'),
        scope='CPU scene/selector construction, field probes, binding support; excludes disk IO, detector and planner rollout beyond selector',
        caveat='Single paired CPU pass on shared VUW host; not Jetson, not end-to-end. K34 frozen report unchanged.')
    (OUT / 'DONE.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
