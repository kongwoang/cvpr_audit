#!/usr/bin/env python
"""For tokens where spec's no-collision score is below cheap's: which in-camera GT objects does spec drop that cheap keeps
(and vice versa), their distance, and whether spec's crops covered them."""
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("np_", B / "scripts/A_navsim/navsim_pdms.py")
NP = importlib.util.module_from_spec(_s); _s.loader.exec_module(NP)
from navsim.common.dataclasses import SceneFilter, SensorConfig                  # noqa: E402
from navsim.common.dataloader import SceneLoader                                 # noqa: E402

REF = os.environ.get("DIAG_REF", "cheap")                     # branch to compare spec with
d = pd.read_csv(NP.EXP / os.environ.get("DIAG_STATES", "pdms_states.csv"))
bad = d[d.no_at_fault_collisions_spec < d[f"no_at_fault_collisions_{REF}"]]
SPLIT = os.environ.get("NAVSIM_SPLIT", "test")
gz = float((NP.DETS / "ground_z.txt").read_text())
priors = pickle.load(open(NP.EXP / "dets" / "size_priors.pkl", "rb"))
rows, lost_rows = [], []
for log, g in bad.groupby("log"):
    dets = pickle.load(open(NP.DETS / f"{log}.pkl", "rb"))
    sl = SceneLoader(NP.DATA / "navsim_logs" / SPLIT, NP.DATA / "sensor_blobs" / SPLIT,
                     SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                                 log_names=[log], tokens=list(g.token)), SensorConfig.build_no_sensors())
    for tok in g.token:
        sc = sl.get_scene_from_token(tok)
        ann = sc.frames[sc.scene_metadata.num_history_frames - 1].annotations
        cam = NP.Cam(*dets[tok]["cam"])
        a_c = NP.branch_annotations(ann, dets[tok][REF], cam, gz, priors, REF)
        a_s = NP.branch_annotations(ann, dets[tok]["spec"], cam, gz, priors, "spec")
        kc, ks = set(a_c.track_tokens), set(a_s.track_tokens)
        gt = {t: i for i, t in enumerate(ann.track_tokens)}
        lost = [t for t in kc - ks if t in gt]
        gained = [t for t in ks - kc if t in gt]
        fp_c = sum(1 for t in a_c.track_tokens if t.startswith("fp:")); fp_s = sum(1 for t in a_s.track_tokens if t.startswith("fp:"))
        for t in lost:
            x, y = ann.boxes[gt[t]][:2]
            lost_rows.append({"x": float(x), "y": float(y), "name": ann.names[gt[t]]})
        rows.append({"token": tok, "lost_by_spec": len(lost), "gained_by_spec": len(gained),
                     "lost_min_dist": min((float(np.hypot(*ann.boxes[gt[t]][:2])) for t in lost), default=np.nan),
                     "fp_cheap": fp_c, "fp_spec": fp_s, "n_cheap": len(dets[tok]["cheap"]["conf"]),
                     "n_spec": len(dets[tok]["spec"]["conf"]), "px": dets[tok]["info"]["pixels"]})
r = pd.DataFrame(rows)
print(len(r), "tokens where spec collides and cheap does not")
print(r.describe().round(2).to_string())
print("tokens with a GT object lost by spec:", int((r.lost_by_spec > 0).sum()), "| with more FPs in spec:", int((r.fp_spec > r.fp_cheap).sum()))

lr = pd.DataFrame(lost_rows)
if len(lr):
    print("objects the reference branch keeps and spec loses:", len(lr), "| by class", lr.name.value_counts().to_dict())
    print("forward x quantiles", lr.x.quantile([.1, .25, .5, .75, .9]).round(1).to_dict())
    print("|lateral y| quantiles", lr.y.abs().quantile([.1, .25, .5, .75, .9]).round(1).to_dict(), "| share |y| > 6 m", round(float((lr.y.abs() > 6).mean()), 2))
