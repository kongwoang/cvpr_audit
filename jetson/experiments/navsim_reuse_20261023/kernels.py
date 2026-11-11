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

