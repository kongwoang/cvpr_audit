#!/usr/bin/env python
"""K18(A) — frame-level routers at budgets {0.5, 1, 2, 4} x ours' mean TENSOR px (post-hoc analysis; AV2 with the ego
mask: the K10 masked router tables / scores in the shadow tree). Same scores, same calibration as k12_routers
(Offloader.decide_calibrated on the train predictions); the routing share rho is searched (bisection on the target)
until the router's actual mean px is within 2% of the budget; oracle = top-k by the true V = J_cheap - J_u960.
"best Budget-Adaptive Routing" = the official estimator (skip_offbin / cond_moric) with the higher G on VAL at that
budget; "decision-value router" = likewise among skip_V / cond_V.  px: cond = cheap + rho * u960, skip = (1 - rho) *
cheap + rho * u960 (canvases / frame px).  CI: paired unit bootstrap (2,000) of ours final minus the router.
usage: k18_routers.py  ->  results/k18/routers_{nusc,av2}.csv
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
sys.path.insert(0, str(B / "third_party" / "bgt-ada"))
from src.offloader import Offloader                                             # noqa: E402

EST = ("skip_offbin", "cond_moric", "skip_V", "cond_V")
TPX = {"nusc": 0.1004, "av2": 0.0532}                    # ours' mean tensor px (K12 nuScenes; K13(f) AV2 masked)


def load(ds):
    if ds == "nusc":
        O = B / "results" / "router"
        D = {sp: pd.read_pickle(O / f"frames__{sp}.pkl") for sp in ("val", "test")}
        ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
        te = D["test"].merge(ref[["seq", "frame", "J_spec_ref_resnet18"]], on=["seq", "frame"])
        ours = (te.J_c - te.J_spec_ref_resnet18).to_numpy()
        pc, ps = 320 * 192 / (1600 * 900), 576 * 960 / (1600 * 900)
    else:
        S = B / "k10_masked" / "results"
        O = S / "av2_sota"
        D = {sp: pd.read_pickle(O / f"frames__{sp}.pkl").rename(columns={"log": "seq"}) for sp in ("val", "test")}
        ref = pd.read_pickle(S / "av2" / "eval__val_v4_final.pkl").rename(columns={"log": "seq"})
        te = D["test"].merge(ref[["seq", "frame", "J_spec_ref_v4lite"]], on=["seq", "frame"])
        ours = (te.J_c - te.J_spec_ref_v4lite).to_numpy()
        pc, ps = 320 * 256 / (1550 * 2048), 960 * 768 / (1550 * 2048)
    S_ = {n: pickle.load(open(O / f"scores__{n}.pkl", "rb")) for n in EST if (O / f"scores__{n}.pkl").exists()}
    assert len(te) == len(D["test"])
    return D, te, ours, S_, pc, ps


def main():
    out = B / "results" / "k18"
    out.mkdir(parents=True, exist_ok=True)
    for ds in ("nusc", "av2"):
        D, te, ours, S, pc, ps = load(ds)
        L = te.J_c.to_numpy(); Vt = (te.J_c - te.J_u960).to_numpy()
        Lv = D["val"].J_c.to_numpy(); Vv = (D["val"].J_c - D["val"].J_u960).to_numpy()
        units = te.seq.to_numpy(); uq = np.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
        rng = np.random.default_rng(0)
        boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        rows = [{"dataset": ds, "budget_x": 1.0, "method": "ours final (with refiner)", "tensor_px": TPX[ds], "G": ours.sum() / L.sum()}]

        def px_of(kind, m):
            return pc + m.mean() * ps if kind == "cond" else (1 - m.mean()) * pc + m.mean() * ps

        def route(name, split, r):
            return Offloader("offload_binary").decide_calibrated(S[name][split], r, calibration_predictions=S[name]["train"]).mask

        def solve(name, split, target):
            kind = name.split("_")[0]
            lo, hi = 0.0, 1.0
            best = None
            for _ in range(40):
                r = (lo + hi) / 2
                m = route(name, split, r)
                px = px_of(kind, m)
                if best is None or abs(px - target) < abs(best[1] - target):
                    best = (m, px, r)
                if abs(px - target) <= 0.02 * target:
                    break
                lo, hi = (r, hi) if px < target else (lo, r)
            return best
        for bx in (0.5, 1.0, 2.0, 4.0):
            target = TPX[ds] * bx

            def add(name, m, px, r, note=""):
                v = Vt * m
                rec = {"dataset": ds, "budget_x": bx, "budget_px": target, "method": name, "rho": float(m.mean()),
                       "rho_target": r, "tensor_px": px, "px_within_2pct": bool(abs(px - target) <= 0.02 * target),
                       "G": v.sum() / L.sum(), "note": note}
                if bx == 1.0:
                    dr = [(ours[b].sum() - v[b].sum()) / L[b].sum() for b in boots]
                    rec.update({"ci_lo_ours_minus": np.percentile(dr, 2.5), "ci_hi_ours_minus": np.percentile(dr, 97.5)})
                rows.append(rec)
            # oracle: cheap on every frame + u960 on the top-k frames by true V (cond accounting)
            k = int(round((target - pc) / ps * len(te)))
            if 0 <= k <= len(te):
                orc = np.zeros(len(te), bool); orc[np.argsort(-Vt, kind="stable")[:k]] = True
                add("oracle frame router (true V, top-k)", orc, px_of("cond", orc), k / len(te))
            else:
                rows.append({"dataset": ds, "budget_x": bx, "budget_px": target, "method": "oracle frame router (true V, top-k)",
                             "note": "budget outside [cheap, cheap + u960]"})
            picks = {}
            for name in [e for e in EST if e in S]:
                m, px, r = solve(name, "test", target)
                gv = None
                mv, _, _ = solve(name, "val", target)
                gv = float((Vv * mv).sum() / Lv.sum())
                picks[name] = gv
                add(("Budget-Adaptive Routing official: " if not name.endswith("_V") else "decision-value router: ") + name,
                    m, px, r, f"val G {gv:.3f}")
            for grp, cand in (("best Budget-Adaptive Routing (picked on val)", ("skip_offbin", "cond_moric")),
                              ("best decision-value router (picked on val)", ("skip_V", "cond_V"))):
                c = [x for x in cand if x in picks]
                if c:
                    nm = max(c, key=lambda x: picks[x])
                    m, px, r = solve(nm, "test", target)
                    add(f"{grp}: {nm}", m, px, r)
        res = pd.DataFrame(rows)
        res.to_csv(out / f"routers_{ds}.csv", index=False)
        pd.set_option("display.width", 250)
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
