#!/usr/bin/env python
"""K12(c) — frame-level routers re-matched to ours' INSTRUMENTED detector pixels (post-hoc analysis, test splits).

The router tables (router_baseline.py score stage for nuScenes new test, av2_sota.py score stage for AV2 val) matched
the routing share rho to ours' area-based pixels. Here rho is matched to ours' tensor pixels (draft canvas + every crop
canvas incl. guard re-detections, 32-px rounding; instr_eval): nuScenes 0.1004 (v3), AV2 0.0740 (v4-lite).
Same scores, same calibration (Offloader.decide_calibrated on the train predictions), same per-frame costs; ours =
the final configuration with refiner (nuScenes v3 + refiner v1, AV2 v4-lite + refiner v2); oracle = top-k by the true
V = J_cheap - J_u960; scene / log bootstrap of ours - router (2,000).
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
sys.path.insert(0, str(GENG))
from src.offloader import Offloader                                             # noqa: E402

EST = ("skip_offbin", "cond_moric", "skip_V", "cond_V")
TPX = {"nusc": 0.1004, "av2": 0.0740}


def load(ds):
    if ds == "nusc":
        O = B / "results" / "router"
        D = {sp: pd.read_pickle(O / f"frames__{sp}.pkl") for sp in ("val", "test")}
        ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
        te = D["test"].merge(ref[["seq", "frame", "J_u960", "J_spec_ref_resnet18"]], on=["seq", "frame"], suffixes=("_gpu", ""))
        ours = (te.J_c - te.J_spec_ref_resnet18).to_numpy()
        pc, ps = 320 * 192 / (1600 * 900), 576 * 960 / (1600 * 900)
    else:
        O = B / "results" / "av2_sota"
        D = {sp: pd.read_pickle(O / f"frames__{sp}.pkl").rename(columns={"log": "seq"}) for sp in ("val", "test")}
        ref = pd.read_pickle(B / "results" / "av2" / "eval__val_v4_final.pkl").rename(columns={"log": "seq"})
        te = D["test"].merge(ref[["seq", "frame", "J_spec_ref_v4lite"]], on=["seq", "frame"])
        ours = (te.J_c - te.J_spec_ref_v4lite).to_numpy()
        pc, ps = 320 * 256 / (1550 * 2048), 960 * 768 / (1550 * 2048)
    S = {n: pickle.load(open(O / f"scores__{n}.pkl", "rb")) for n in EST if (O / f"scores__{n}.pkl").exists()}
    assert len(te) == len(D["test"])
    return D, te, ours, S, pc, ps


def main():
    rows = []
    for ds in ("nusc", "av2"):
        D, te, ours, S, pc, ps = load(ds)
        L = te.J_c.to_numpy(); Vt = (te.J_c - te.J_u960).to_numpy()
        units = te.seq.to_numpy(); uq = np.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
        rng = np.random.default_rng(0)
        boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        tp = TPX[ds]
        rho = {"cond": (tp - pc) / ps, "skip": (tp - pc) / (ps - pc)}
        rows.append({"dataset": ds, "method": "ours final (with refiner)", "network_px": tp, "G": ours.sum() / L.sum()})

        def route(name, split, r):
            return Offloader("offload_binary").decide_calibrated(S[name][split], r, calibration_predictions=S[name]["train"]).mask

        def add(name, mask, px, r):
            v = Vt * mask
            dr = [(ours[b].sum() - v[b].sum()) / L[b].sum() for b in boots]
            rows.append({"dataset": ds, "method": name, "rho_target": r, "rho_actual": float(mask.mean()), "network_px": px,
                         "G": v.sum() / L.sum(), "ci_lo_ours_minus": np.percentile(dr, 2.5), "ci_hi_ours_minus": np.percentile(dr, 97.5)})
        for kind in ("cond", "skip"):
            r = rho[kind]
            n = int(round(r * len(te)))
            orc = np.zeros(len(te), bool); orc[np.argsort(-Vt, kind="stable")[:n]] = True
            add(f"oracle frame router (rho matched as {kind})", orc, pc + r * ps if kind == "cond" else (1 - r) * pc + r * ps, r)
            for name in [e for e in S if e.startswith(kind)]:
                m = route(name, "test", r)
                px = pc + m.mean() * ps if kind == "cond" else (1 - m.mean()) * pc + m.mean() * ps
                add(("Budget-Adaptive Routing official: " if not name.endswith("_V") else "decision-value target: ") + name, m, px, r)
        # Geng's adaptive arbiter between the two official estimators, each at its own matched rho, picked on val
        def G_val(name, r):
            v = D["val"]; m = route(name, "val", r)
            return float(((v.J_c - v.J_u960).to_numpy() * m).sum() / v.J_c.sum())
        pick = max((("skip_offbin", rho["skip"]), ("cond_moric", rho["cond"])), key=lambda q: G_val(*q))
        m = route(pick[0], "test", pick[1])
        px = pc + m.mean() * ps if pick[0].startswith("cond") else (1 - m.mean()) * pc + m.mean() * ps
        add(f"Budget-Adaptive Routing official: adaptive({pick[0]})", m, px, pick[1])
    res = pd.DataFrame(rows)
    res.to_csv(B / "results" / "k12_routers_rematched.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
