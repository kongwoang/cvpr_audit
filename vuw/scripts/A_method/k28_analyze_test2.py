#!/usr/bin/env python
"""K28 PHASE 2, addendum §2n.1 tables (CONFIRMATORY; written before any test value was looked at). Inputs: the K24 test pass
(k24_analyze.merged) and the K28 test parts ours (k28_test.py) and abl (k28_test_abl.py), merged on unit / frame (J_cheap
checked). Argoverse-HD: frames with 3D labels only. Unit bootstrap 2,000, seed 0 (k24_analyze.Boot); verdicts as K24.
  ablation_v6S_<ds>.csv   v6-S cells vs v6-S (refiner v3a, C1): without refiner; without C1 (= v5 of K24: no temporal cue),
                          with and without refiner; T4 in place of C1 (= v5.1-S); support only / field only + C1,
                          with and without refiner
  routers_v6_<ds>.csv     the K24 routers and the oracle (K18 protocol, k24_analyze.routers) re-matched to the GFLOPs of
                          v6-S / v6-M / v6-L (YOLOv8s-equivalent tensor px within 2%), vs that configuration
  parity_abl_v6.csv       abl v5.1-S = K24 v5.1-S; abl so51S / fo51S = K24 supp so51S / fo51S (J none / v3a, px)
usage: k28_analyze_test2.py
"""
from __future__ import annotations

import glob
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_a = importlib.util.spec_from_file_location("k24a", B / "scripts" / "A_method" / "k24_analyze.py")
K24A = importlib.util.module_from_spec(_a); _a.loader.exec_module(K24A)
_b = importlib.util.spec_from_file_location("k28a", B / "scripts" / "A_method" / "k28_analyze.py")
KA = importlib.util.module_from_spec(_b); _b.loader.exec_module(KA)
NATIVE = {"nusc": 1600 * 900, "av2": 2048 * 1550, "ahd": 1920 * 1200}
V6 = {"v6-S": ("v6S", "v3a", "yolov8s"), "v6-M": ("v6M", "own-yolov8m", "yolov8m"), "v6-L": ("v6L", "own-yolo11m", "yolo11m")}


def load(ds, part):
    rows, fs = [], sorted(glob.glob(str(REAL / "results" / "k28" / f"{ds}_test_v6{part}" / "*.pkl")))
    for f in fs:
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows), len(fs)


def routers_v6(d, ds, targets):
    """k24_analyze.routers with the target px replaced by `targets` (label -> YOLOv8s-equivalent tensor px)"""
    out = {}
    pc, ps = d.px_cheap.mean(), d.px_u960.mean()
    Jc, Ac = d.J_cheap.to_numpy(float), d.a_cheap.to_numpy(int)
    esc = {"": {r: (d[f"J_u960|{r}"].to_numpy(float), d[f"a_u960|{r}"].to_numpy(int)) for r in ("none", "v3a")},
           " + components": {r: (d[f"J_u960+T4|{r}"].to_numpy(float), d[f"a_u960+T4|{r}"].to_numpy(int)) for r in ("none", "v3a")}}

    def put(lab, ref, msk, px, extra=None):
        for suf, E in esc.items():
            o = {"Jn": np.where(msk, E["none"][0], Jc), "Jr": np.where(msk, E["v3a"][0], Jc),
                 "An": np.where(msk, E["none"][1], Ac), "Ar": np.where(msk, E["v3a"][1], Ac),
                 "px": float(px), "group": "router" + suf, "matched_to": ref}
            o.update(extra or {})
            out[lab + suf] = o
    V = Jc - esc[""]["v3a"][0]
    for ref, target in targets.items():
        k = int(round((target - pc) / ps * len(d)))
        orc = np.zeros(len(d), bool)
        if 0 <= k <= len(d):
            orc[np.argsort(-V, kind="stable")[:k]] = True
        px = pc + orc.mean() * ps
        put(f"oracle frame router (true V, top-k) @ {ref}", ref, orc, px, {"within_2pct": bool(abs(px - target) <= 0.02 * target)})
    if ds == "ahd":
        return out
    sys.path.insert(0, str(B / "third_party" / "bgt-ada"))
    from src.offloader import Offloader
    if ds == "nusc":
        O = B / "results" / "router"
        D = {sp: pd.read_pickle(O / f"frames__{sp}.pkl") for sp in ("val", "test")}
    else:
        O = B / "k10_masked" / "results" / "av2_sota"
        D = {sp: pd.read_pickle(O / f"frames__{sp}.pkl").rename(columns={"log": "seq"}) for sp in ("val", "test")}
    S_ = {n: pickle.load(open(O / f"scores__{n}.pkl", "rb")) for n in K24A.EST if (O / f"scores__{n}.pkl").exists()}
    te = D["test"].reset_index(drop=True)
    key = pd.DataFrame({"seq": d.unit.values, "frame": d.frame.values, "i": np.arange(len(d))})
    mm = te[["seq", "frame", "J_c"]].reset_index().merge(key, on=["seq", "frame"], how="inner")
    assert len(mm) == len(d) == len(te), (len(mm), len(d), len(te))
    jc_agree = float(np.mean(np.isclose(mm.J_c.to_numpy(float), Jc[mm.i.to_numpy()])))
    perm = mm.sort_values("i")["index"].to_numpy()
    Vv = (D["val"].J_c - D["val"].J_u960).to_numpy(); Lv = D["val"].J_c.to_numpy()

    def px_of(kind, msk):
        return pc + msk.mean() * ps if kind == "cond" else (1 - msk.mean()) * pc + msk.mean() * ps

    def route(name, split, r):
        return Offloader("offload_binary").decide_calibrated(S_[name][split], r, calibration_predictions=S_[name]["train"]).mask

    def solve(name, split, target):
        kind = name.split("_")[0]
        lo, hi, best = 0.0, 1.0, None
        for _ in range(40):
            r = (lo + hi) / 2
            msk = np.asarray(route(name, split, r), bool)
            px = px_of(kind, msk)
            if best is None or abs(px - target) < abs(best[1] - target):
                best = (msk, px, r)
            if abs(px - target) <= 0.02 * target:
                break
            lo, hi = (r, hi) if px < target else (lo, r)
        return best
    for ref, target in targets.items():
        picks = {}
        for name in S_:
            mv, _, _ = solve(name, "val", target)
            picks[name] = float((Vv * mv).sum() / Lv.sum())
        for grp, cand in (("best Budget-Adaptive Routing (picked on val)", ("skip_offbin", "cond_moric")),
                          ("best decision-value router (picked on val)", ("skip_V", "cond_V"))):
            c = [x for x in cand if x in picks]
            if not c:
                continue
            nm = max(c, key=lambda x: picks[x])
            msk, px, _ = solve(nm, "test", target)
            put(f"{grp}: {nm} @ {ref}", ref, msk[perm], px, {"within_2pct": bool(abs(px - target) <= 0.02 * target), "jc_agree": jc_agree})
    return out


def main():
    gfp = REAL / "results" / "k28" / "phase0" / "gflops_640.csv"
    GF = dict(KA.GF)
    if gfp.exists():
        GF.update(pd.read_csv(gfp).iloc[0].to_dict())
    out = REAL / "results" / "k28" / "tables_v6"
    out.mkdir(parents=True, exist_ok=True)
    par = []
    for ds in ("nusc", "av2", "ahd"):
        d, why = K24A.merged(ds, "main", "supp")
        if d is None:
            print(f"NOT READY: {why}; no table written"); return
        for part in ("ours", "abl"):
            p, n = load(ds, part)
            if n != K24A.NUNITS[ds]:
                print(f"NOT READY: {ds} {part} {n} of {K24A.NUNITS[ds]} units; no table written"); return
            keep = ["unit", "frame", "J_cheap"] + [c for c in p.columns if c != "J_cheap" and c.startswith(("J_", "a_", "px_"))]
            p = p[keep].rename(columns={c: f"{c}@{part}" for c in keep[3:] if c in d.columns})
            n0 = len(d)
            d = d.merge(p, on=["unit", "frame"], how="left", suffixes=("", f"_{part}"))
            assert len(d) == n0 and d[f"J_cheap_{part}"].notna().all(), (ds, part)
            assert np.allclose(d.J_cheap.to_numpy(float), d[f"J_cheap_{part}"].to_numpy(float)), (ds, part)
        bt = K24A.Boot(d)
        L = d.J_cheap.to_numpy(float)
        nat = NATIVE[ds]
        # ---- parity
        for a, b in (("J_v51S|v3a@abl", "J_v51S|v3a"), ("px_v51S@abl", "px_v51S"), ("J_so51S|none@abl", "J_so51S|none"),
                     ("J_so51S|v3a@abl", "J_so51S|v3a"), ("px_so51S@abl", "px_so51S"), ("J_fo51S|none@abl", "J_fo51S|none"),
                     ("J_fo51S|v3a@abl", "J_fo51S|v3a"), ("px_fo51S@abl", "px_fo51S")):
            if a in d and b in d:
                par.append({"dataset": ds, "k28": a, "k24": b, "frames": len(d), "identical": float(np.isclose(d[a], d[b]).mean())})
        # ---- ablation of v6-S
        Jp = d["J_v6S|v3a"].to_numpy(float)
        ab = []
        for lab, jk, pk in (("v6-S (refiner v3a, C1)", "v6S|v3a", "px_v6S"), ("v6-S without refiner (C1 only)", "v6S|none", "px_v6S"),
                            ("v6-S without C1 (refiner only; = v5)", "v5|v3a", "px_v5"), ("v6-S without refiner and without C1 (= v5)", "v5|none", "px_v5"),
                            ("T4 in place of C1 (= v5.1-S)", "v51S|v3a", "px_v51S"), ("T4 in place of C1, no refiner", "v51S|none", "px_v51S"),
                            ("support only + C1 + refiner", "so6S|v3a", "px_so6S"), ("field only + C1 + refiner", "fo6S|v3a", "px_fo6S"),
                            ("support only + C1, no refiner", "so6S|none", "px_so6S"), ("field only + C1, no refiner", "fo6S|none", "px_fo6S")):
            J = d[f"J_{jk}"].to_numpy(float)
            mdiff, lo, hi = bt.diff(J, Jp)                     # (v6-S - cell)
            r = {"dataset": ds, "cell": lab, "G": (L - J).sum() / L.sum(), "tensor_px_mean": d[pk].mean(),
                 "v6S_minus": mdiff, "lo": lo, "hi": hi, "verdict": K24A.verdict(lo)}
            r.update(K24A.extras(d, J, d[f"a_{jk}"]))
            ab.append(r)
        pd.DataFrame(ab).to_csv(out / f"ablation_v6S_{ds}.csv", index=False)
        # ---- routers / oracle re-matched to v6 GFLOPs
        targets, cfg = {}, {}
        for lab, (k, rf, det) in V6.items():
            px = d[f"px_{k}"]
            g = px * GF[det] if det == "yolov8s" else d.px_cheap * GF["yolov8s"] + (px - d.px_cheap) * GF[det]
            targets[lab] = float(g.mean()) / GF["yolov8s"]                 # YOLOv8s-equivalent tensor px
            cfg[lab] = d[f"J_{k}|{rf}"].to_numpy(float)
        rows = []
        for lab, e in routers_v6(d, ds, targets).items():
            Jr = np.asarray(e["Jr"], float)
            mdiff, lo, hi = bt.diff(Jr, cfg[e["matched_to"]])              # (v6-X - router)
            r = {"dataset": ds, "router": lab, "group": e["group"], "matched_to": e["matched_to"], "target_px": targets[e["matched_to"]],
                 "px": e["px"], "within_2pct": e.get("within_2pct"), "jc_agree": e.get("jc_agree", np.nan),
                 "G_no_refiner": (L - np.asarray(e["Jn"], float)).sum() / L.sum(), "G_refiner": (L - Jr).sum() / L.sum(),
                 "v6_minus": mdiff, "lo": lo, "hi": hi, "verdict": K24A.verdict(lo)}
            r.update(K24A.extras(d, Jr, e["Ar"]))
            rows.append(r)
        pd.DataFrame(rows).to_csv(out / f"routers_v6_{ds}.csv", index=False)
        print(f"{ds}: {bt.n} units, {len(d)} frames; {len(ab)} ablation cells, {len(rows)} router rows")
    pd.DataFrame(par).to_csv(out / "parity_abl_v6.csv", index=False)
    print(pd.DataFrame(par).to_string(index=False))
    print("K28 V6 TABLES2 DONE")


if __name__ == "__main__":
    main()
