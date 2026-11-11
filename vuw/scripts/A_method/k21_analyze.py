#!/usr/bin/env python
"""K21 tables (post-hoc analysis).
  detector2_{nusc,av2}.csv     YOLO11s as draft and target (k21_det2): G of u640 / u960 / native / ours without and with the
                               existing refiner, tensor px, unit bootstrap (2,000) of ours minus each whole-frame resolution
                               (ours without refiner and with refiner); the YOLOv8s reference numbers from K14 / instr
  mask_sensitivity_av2.csv     nine ego-mask settings (k21_mask): G of u960, native, uncertainty crops, ours -+ refiner (each
                               pipeline referenced to the cheap draft under the same mask), tensor px of ours, share of frames
                               whose action changes relative to the reported mask (rows >= 1830, containment 0.5)
  probes_{nusc,av2}.csv        probe templates x closing speeds (k17_variants arms vc<v>_<probe>), dev and test splits:
                               G -+ refiner, tensor px, agreement with pi(F) (u960), recovery in the controlled deletion by
                               class of the deleted decision-relevant object (test)
usage: k21_analyze.py
"""
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
S = B / "k10_masked"
OUT = B / "results" / "k21_tables"


def load(pat):
    rows = []
    for f in sorted(glob.glob(str(pat))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


def boots(units):
    uq = pd.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
    rng = np.random.default_rng(0)
    return [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]


def det2():
    for ds, root in (("nusc", B), ("av2", S)):
        d = load(root / "results" / "k21_det2" / "yolo11s" / f"{ds}__*.pkl")
        if not len(d):
            continue
        L = d.J_cheap.to_numpy(float); bs = boots(d.unit.to_numpy())
        rows = []
        V = {k: L - d[f"J_{k}"].to_numpy(float) for k in ("u640", "u960", "native", "full", "full_ref")}
        nm = {"u640": "u640", "u960": "u960", "native": "native", "full": "ours without refiner", "full_ref": "ours + existing refiner"}
        for k, v in V.items():
            r = {"dataset": ds, "detector": "YOLO11s (draft + target)", "units": d.unit.nunique(), "frames": len(d), "method": nm[k],
                 "G": v.sum() / L.sum(), "tensor_px": d[f"px_{'full' if k.startswith('full') else k}"].mean()}
            if k in ("u640", "u960", "native"):
                for o in ("full", "full_ref"):
                    dd = [(V[o][b].sum() - v[b].sum()) / L[b].sum() for b in bs]
                    r[f"{o}_minus_ci"] = f"[{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]"
            rows.append(r)
        dr = [(V["full_ref"][b].sum() - V["full"][b].sum()) / L[b].sum() for b in bs]
        rows.append({"dataset": ds, "detector": "YOLO11s", "method": "refiner effect (ours + refiner minus ours)",
                     "G": (V["full_ref"].sum() - V["full"].sum()) / L.sum(),
                     "full_ref_minus_ci": f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"})
        # YOLOv8s reference (K14 per-frame tables)
        k = load(root / "results" / "k14" / f"{ds}__*.pkl")
        Lk = k.J_cheap.to_numpy(float)
        for c, n in (("u960", "u960"), ("native", "native"), ("full", "ours without refiner"), ("full_ref", "ours + refiner")):
            rows.append({"dataset": ds, "detector": "YOLOv8s (frozen, K14)", "method": n, "G": (Lk - k[f"J_{c}"]).sum() / Lk.sum(),
                         "tensor_px": k[f"px_{'full' if c.startswith('full') else c}"].mean()})
        pd.DataFrame(rows).to_csv(OUT / f"detector2_{ds}.csv", index=False)
        print(pd.DataFrame(rows).round(4).to_string(index=False))


def mask():
    d = load(B / "results" / "k21_mask" / "av2__*.pkl")
    if not len(d):
        return
    bs = boots(d.unit.to_numpy())
    ref = "1830_0.5"
    rows = []
    for y0 in (1780, 1830, 1880):
        for c in (0.3, 0.5, 0.7):
            t = f"{y0}_{c:g}"
            L = d[f"J_cheap_{t}"].to_numpy(float)
            r = {"band_start_row": y0, "containment": c, "reported": t == ref, "frames": len(d)}
            for k, n in (("u960", "u960"), ("native", "native"), ("uncertainty", "uncertainty crops"), ("full", "ours without refiner"),
                         ("full_ref", "ours")):
                r[f"G_{n}"] = (L - d[f"J_{k}_{t}"]).sum() / L.sum()
                r[f"action_changed_vs_reported_{n}"] = float((d[f"a_{k}_{t}"] != d[f"a_{k}_{ref}"]).mean())
            r["action_changed_vs_reported_cheap"] = float((d[f"a_cheap_{t}"] != d[f"a_cheap_{ref}"]).mean())
            r["tensor_px_ours"] = d[f"px_full_{t}"].mean()
            Vo, Vu = L - d[f"J_full_ref_{t}"].to_numpy(float), L - d[f"J_u960_{t}"].to_numpy(float)
            dd = [(Vo[b].sum() - Vu[b].sum()) / L[b].sum() for b in bs]
            r["ours_minus_u960_ci"] = f"[{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]"
            Vn = L - d[f"J_uncertainty_{t}"].to_numpy(float)
            dd = [(Vo[b].sum() - Vn[b].sum()) / L[b].sum() for b in bs]
            r["ours_minus_uncertainty_ci"] = f"[{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]"
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "mask_sensitivity_av2.csv", index=False)
    print(res.round(4).to_string(index=False))


def probes():
    for ds, root, dev in (("nusc", B, "val"), ("av2", S, "dev")):
        rows = []
        for split, tag in (("dev", f"{ds}_{dev}_base"), ("test", f"{ds}_test_base")):
            d = load(root / "results" / "k17" / tag / "*.pkl")
            if not len(d):
                continue
            if "gt" in d:
                d = d[d["gt"].astype(bool)]
            L = d.J_cheap.to_numpy(float)
            arms = [c[2:] for c in d.columns if c.startswith("a_vc") and not c.endswith("_ref")] + ["full"]
            for a in arms:
                if f"J_{a}" not in d:
                    continue
                if a == "full":
                    vc, pr = (10, "car") if ds == "nusc" else (0, "car")
                else:
                    vc, pr = a[2:].split("_"); vc = int(vc)
                rows.append({"dataset": ds, "split": split, "probe": pr, "v_close": vc, "frozen": a == "full",
                             "G_no_refiner": (L - d[f"J_{a}"]).sum() / L.sum(), "G_refiner": (L - d[f"J_{a}_ref"]).sum() / L.sum(),
                             "tensor_px": d[f"px_{a}"].mean(), "agreement_u960": float((d[f"a_{a}"] == d.a_u960).mean())})
        dl = load(root / "results" / "k17" / f"{ds}_test_del_all" / "*.pkl")
        if len(dl):
            for a in [c[6:] for c in dl.columns if c.startswith("found_") and c != "found_cheap"]:
                rec = {}
                for cls in ("vehicle", "person", "cyclist", "all"):
                    n = f = 0
                    for _, r in dl.iterrows():
                        if not isinstance(r.get("deleted_mask"), np.ndarray) or not isinstance(r.get(f"found_{a}"), np.ndarray):
                            continue
                        m = r["deleted_mask"] & ((np.asarray(r["rel_cls"]) == cls) if cls != "all" else True)
                        n += int(m.sum()); f += int((r[f"found_{a}"] & m).sum())
                    rec[cls] = (f / n if n else np.nan, n)
                for row in rows:
                    if row["split"] == "test" and ((a == "full" and row["frozen"]) or a == f"vc{row['v_close']}_{row['probe']}"):
                        for cls, (v, n) in rec.items():
                            row[f"recovery_{cls}"] = v; row[f"deleted_{cls}"] = n
        if not rows:
            print(ds, "no probe runs yet"); continue
        res = pd.DataFrame(rows).sort_values(["split", "probe", "v_close"])
        res.to_csv(OUT / f"probes_{ds}.csv", index=False)
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    det2(); mask(); probes()
