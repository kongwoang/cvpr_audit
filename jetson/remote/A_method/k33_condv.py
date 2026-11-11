#!/usr/bin/env python
"""K33 A, decision-value router (amendment 1, handoff/K33/STATUS.md): cond_V of router_baseline.py stage_train, retrained
deterministically (bgt-ada XGBoostEstimator, 100 trees, depth 5, lr 0.1, regression; their 21 detection statistics of
the weak boxes with conf >= 0.25 + ego speed; target V = J_cheap - J_u960 on the TRAIN frames), from the stored K12 / K24
training frames: nuScenes results/router (new TRAIN scenes), AV2 results/av2_sota (AV2 train logs). Parity: the
predictions on the stored test frames equal the stored scores__cond_V.pkl. Output results/k33/router/condV_<ds>.json
usage: k33_condv.py
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
GENG = B / "third_party" / "bgt-ada"
OUT = B / "results" / "k33" / "router"


def feats(boxes, d, key):
    from src.features import BASE_FEATURES, extract_detection_features
    X = []
    for u, f in zip(d[key], d.frame.astype(int)):
        r = boxes[(u, f)]
        m = r["weak_conf"] >= 0.25
        dets = [(float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1]), float(c), str(k))
                for b, c, k in zip(r["weak_xyxy"][m], r["weak_conf"][m], r["weak_cls"][m])]
        e = extract_detection_features(dets)
        X.append([e[k] for k in BASE_FEATURES])
    return np.asarray(X, float)


def main():
    sys.path.insert(0, str(GENG))
    from src.models.classical import XGBoostEstimator
    OUT.mkdir(parents=True, exist_ok=True)
    for ds, src, key, vstrong in (("nusc", B / "results" / "router", "seq", "J_u960"), ("av2", B / "results" / "av2_sota", "log", "J_u960")):
        D = {sp: pd.read_pickle(src / f"frames__{sp}.pkl") for sp in ("train", "test")}
        BX = {sp: pickle.load(open(src / f"boxes__{sp}.pkl", "rb")) for sp in ("train", "test")}
        if vstrong not in D["train"]:
            print(ds, "columns", list(D["train"].columns)); continue
        X = {sp: np.column_stack([feats(BX[sp], D[sp], key), D[sp].v_ego.to_numpy(float)]) for sp in D}
        y = (D["train"].J_c - D["train"][vstrong]).to_numpy(float)
        est = XGBoostEstimator(n_estimators=100, max_depth=5, learning_rate=0.1, task_type="regression")
        est.fit(X["train"], y)
        p = np.asarray(est.predict(X["test"]), float)
        st = pickle.load(open(src / "scores__cond_V.pkl", "rb"))["test"]
        print(f"{ds}: parity with the stored cond_V test scores: max |diff| {np.max(np.abs(p - np.asarray(st, float))):.2e}", flush=True)
        est.model.save_model(str(OUT / f"condV_{ds}.json"))
    print("K33 CONDV DONE")


if __name__ == "__main__":
    main()
