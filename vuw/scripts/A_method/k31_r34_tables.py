#!/usr/bin/env python
"""K31 items 3 and 4 tables (post-hoc; pre-registered in handoff/K31/STATUS.md). Writes results/k31/tables/:
  curv_dev.csv       item 3 dev check (nuScenes new val, AV2 CFG-DEV): frames, share of frames with a non-empty field on the
                     draft (straight, curved), |kappa| quantiles and share > 0.01, the pre-registered "behaves" flag
  curv_test.csv      item 3 test: every row scored by the curved AEB consumer: G, px, curved v6-S minus row (paired unit
                     bootstrap, 2,000, seed 0) overall, per |kappa| tercile and on frames whose decisive object is outside
                     the straight tube
  bounded.csv        item 4: v6-S <= 2 calls (k_loop.run_bounded) on the full test splits: G, px, minus v6-S / u960 / native
                     + C1 + v3a / hybrid tsp8@S (stored K28 phase-3 rows); share of frames at the bound
  bounded_parity.csv item 4: agreement with the Jetson K29 decisions (bounded and unbounded) on the frames both ran, and the
                     in-run v6-S vs the stored K28 v6-S
usage: k31_r34_tables.py [--items 3,4]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B_REAL", os.environ["B"]))
OUT = B / "results" / "k31" / "tables"
NWS = {"nusc": 1600, "av2": 2048, "ahd": 1920}
DSL = {"nusc": "nuScenes new test", "av2": "AV2 val (ego mask)", "ahd": "Argoverse-HD val"}


def recs(path, cols=None):
    rows = []
    for f in sorted(glob.glob(str(path / "*.pkl"))):
        for r in pickle.load(open(f, "rb")):
            rows.append(r if cols is None else {c: r.get(c) for c in cols})
    return pd.DataFrame(rows) if rows else None


class Boot:
    def __init__(self, units, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        self.uq, self.inv = np.unique(np.asarray(units), return_inverse=True)
        self.bs = [rng.integers(0, len(self.uq), len(self.uq)) for _ in range(n)]

    def sums(self, x):
        return np.bincount(self.inv, np.asarray(x, float), len(self.uq))

    def diff(self, L, Ja, Jb):
        l, a, b = self.sums(L), self.sums(Ja), self.sums(Jb)
        v = [(b[s].sum() - a[s].sum()) / l[s].sum() for s in self.bs if l[s].sum() > 0]
        return float((b.sum() - a.sum()) / l.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def G(L, J):
    L, J = np.asarray(L, float), np.asarray(J, float)
    return float((L.sum() - J.sum()) / L.sum()) if L.sum() > 0 else np.nan


def item3():
    dev = []
    for ds in ("nusc", "av2"):
        d = recs(B / "results" / "k31" / "r3" / f"{ds}_curvdev")
        if d is None:
            continue
        ak = d.kappa.abs()
        fs, fc = (d.field_cells_straight > 0).mean(), (d.field_cells_curved > 0).mean()
        dev.append({"dataset": {"nusc": "nuScenes new val", "av2": "AV2 CFG-DEV (ego mask)"}[ds], "units": d.unit.nunique(),
                    "frames": len(d), "share_field_nonempty_straight": fs, "share_field_nonempty_curved": fc,
                    "abs_kappa_p50": ak.median(), "abs_kappa_p90": ak.quantile(0.9), "abs_kappa_p99": ak.quantile(0.99),
                    "share_abs_kappa_gt_0.01": float((ak > 0.01).mean()), "share_kappa_at_clip": float((ak >= 0.0999).mean()),
                    "act_gt_curved_ne_straight": np.nan,
                    "behaves": bool(fs > 0 and 0.5 <= fc / fs <= 2.0 and (ak > 0.01).mean() >= 0.05)})
    pd.DataFrame(dev).to_csv(OUT / "curv_dev.csv", index=False)
    rows = []
    for ds in ("nusc", "av2", "ahd"):
        d = recs(B / "results" / "k31" / "r3" / f"{ds}_curv")
        if d is None:
            continue
        d = d[d["gt"].astype(bool)].reset_index(drop=True)
        n = NWS[ds]
        R = [("cheap", "cheap", "cheap"), ("u960 + C1 + v3a", "wf960+C1|v3a", "wf960"), ("native + C1 + v3a", f"wf{n}+C1|v3a", f"wf{n}"),
             ("hybrid tsp8@S, straight support + straight tube (as K28)", "tsp8@S+C1|v3a", "tsp8@S+C1"),
             ("hybrid tsp8@S, curved support + curved tube (re-designed prior)", "tsp8c@S+C1|v3a", "tsp8c@S+C1"),
             ("uncertainty unc2@S + C1 + v3a", "unc2@S+C1|v3a", "unc2@S+C1"),
             ("extra: frozen straight v6-S (deployed), scored curved", "v6S|v3a", "v6S")]
        ak = d.kappa.abs()
        q1, q2 = ak.quantile(1 / 3), ak.quantile(2 / 3)
        strata = {"all frames": np.ones(len(d), bool), f"|kappa| tercile 1 (<= {q1:.4f})": (ak <= q1).to_numpy(),
                  f"|kappa| tercile 2": ((ak > q1) & (ak <= q2)).to_numpy(), f"|kappa| tercile 3 (> {q2:.4f})": (ak > q2).to_numpy(),
                  "decisive object outside the straight tube": ((d.decc >= 0) & (d.get("decc_in_straight_tube") == False)).to_numpy()}  # noqa: E712
        for sn, m in strata.items():
            x = d[m].reset_index(drop=True)
            if not len(x):
                continue
            bt = Boot(x.unit.to_numpy())
            L = x["Jc_cheap"].to_numpy(float)
            base = {"dataset": DSL[ds], "stratum": sn, "frames": len(x), "units": x.unit.nunique(), "sum_Jc_cheap": float(L.sum())}
            rows.append({**base, "row": "v6-S with the curved consumer", "G": G(L, x["Jc_v6Sc|v3a"]), "px_mean": x["px_v6Sc"].mean()})
            for lab, jk, pk in R:
                J = x["Jc_cheap"] if jk == "cheap" else x[f"Jc_{jk}"]
                r = {**base, "row": lab, "G": G(L, J), "px_mean": x[f"px_{pk}"].mean()}
                if L.sum() > 0:
                    pt, lo, hi = bt.diff(L, x["Jc_v6Sc|v3a"], J)
                    r.update({"v6Sc_minus_row": pt, "ci_lo": lo, "ci_hi": hi, "excludes_0": bool(lo > 0 or hi < 0)})
                rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "curv_test.csv", index=False)


def item4():
    rows, par = [], []
    jet = pd.read_csv(B / "results" / "k31" / "meta" / "jetson_b2_actions.csv")
    for ds in ("nusc", "av2", "ahd"):
        b = recs(B / "results" / "k31" / "r4" / f"{ds}_b2")
        if b is None:
            continue
        b = b[b["gt"].astype(bool)]
        n = NWS[ds]
        keys = ["unit", "frame", "gt", "J_cheap", "J_v6S|v3a", "a_v6S|v3a", "px_v6S", f"J_wf960+C1|v3a", "px_wf960",
                f"J_wf{n}+C1|v3a", f"px_wf{n}", "J_tsp8@S+C1|v3a", "px_tsp8@S+C1"]
        s = recs(B / "results" / "k28" / "p3" / f"{ds}_test_instr", keys)
        s = s[s["gt"].astype(bool)].drop(columns="gt")
        d = b[["unit", "frame", "J_v6Sb2|v3a", "J_v6Sb2|none", "a_v6Sb2|v3a", "px_v6Sb2", "rounds_v6Sb2", "conv_v6Sb2",
               "a_v6S|v3a", "J_v6S|v3a"]].rename(columns={"a_v6S|v3a": "a_v6S_r4", "J_v6S|v3a": "J_v6S_r4"}).merge(s, on=["unit", "frame"])
        par.append({"dataset": DSL[ds], "check": "in-run v6-S vs stored K28 v6-S (VUW)", "frames": len(d),
                    "agreement": float((d["a_v6S_r4"] == d["a_v6S|v3a"]).mean())})
        bt = Boot(d.unit.to_numpy())
        L = d["J_cheap"].to_numpy(float)
        rows.append({"dataset": DSL[ds], "row": "v6-S <= 2 calls", "G": G(L, d["J_v6Sb2|v3a"]), "px_mean": d["px_v6Sb2"].mean(),
                     "share_at_bound": float((~d["conv_v6Sb2"].astype(bool)).mean()), "frames": len(d), "units": d.unit.nunique()})
        for lab, jk, pk in (("v6-S", "J_v6S|v3a", "px_v6S"), ("u960 + C1 + v3a", "J_wf960+C1|v3a", "px_wf960"),
                            ("native + C1 + v3a", f"J_wf{n}+C1|v3a", f"px_wf{n}"), ("hybrid tsp8@S + C1 + v3a", "J_tsp8@S+C1|v3a", "px_tsp8@S+C1")):
            pt, lo, hi = bt.diff(L, d["J_v6Sb2|v3a"], d[jk])
            rows.append({"dataset": DSL[ds], "row": lab, "G": G(L, d[jk]), "px_mean": d[pk].mean(), "b2_minus_row": pt, "ci_lo": lo,
                         "ci_hi": hi, "excludes_0": bool(lo > 0 or hi < 0), "frames": len(d), "units": d.unit.nunique()})
        j = jet[jet.dataset == ds]
        if len(j):
            m = j.merge(b[["unit", "frame", "a_v6Sb2|v3a", "a_v6S|v3a", "a_v6Sb2|none", "a_v6S|none"]], on=["unit", "frame"])
            for lab, va, ja in (("bounded <= 2, refiner", "a_v6Sb2|v3a", "a_ours_b2_jetson"), ("unbounded v6-S, refiner", "a_v6S|v3a", "a_ours_jetson"),
                                ("bounded <= 2, no refiner", "a_v6Sb2|none", "a_ours_b2_noref_jetson"),
                                ("unbounded v6-S, no refiner", "a_v6S|none", "a_ours_noref_jetson")):
                par.append({"dataset": DSL[ds], "check": f"VUW vs Jetson K29 (trt32): {lab}", "frames": len(m),
                            "agreement": float((m[va] == m[ja]).mean()) if len(m) else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "bounded.csv", index=False)
    pd.DataFrame(par).to_csv(OUT / "bounded_parity.csv", index=False)


def b2k():
    """variant B2K: dev selection (b2k_dev.csv) and, when the test run exists, b2k_test.csv"""
    dev, G_ = [], {}
    for ds in ("nusc", "av2"):
        d = recs(B / "results" / "k31" / "r4" / f"{ds}_b2dev")
        if d is None:
            continue
        d = d[d["gt"].astype(bool)]
        L = d["J_cheap"].to_numpy(float)
        for nm in ("v6S", "v6Sb2", "v6Sb2A", "v6Sb2B"):
            if f"J_{nm}|v3a" not in d:
                continue
            g = G(L, d[f"J_{nm}|v3a"])
            G_[(ds, nm)] = g
            dev.append({"split": {"nusc": "nuScenes new val", "av2": "AV2 CFG-DEV (ego mask)"}[ds], "pipeline": nm, "G": g,
                        "px_mean": d[f"px_{nm}"].mean(), "share_at_bound": float((~d[f"conv_{nm}"].astype(bool)).mean()) if f"conv_{nm}" in d else np.nan,
                        "frames_differing_from_v6Sb2": int((d[f"a_{nm}|v3a"] != d["a_v6Sb2|v3a"]).sum()), "frames": len(d), "units": d.unit.nunique()})
    sel = "none"
    if all((ds, nm) in G_ for ds in ("nusc", "av2") for nm in ("v6Sb2", "v6Sb2A", "v6Sb2B")):
        mean = {m: np.mean([G_[(ds, m)] for ds in ("nusc", "av2")]) for m in ("v6Sb2", "v6Sb2A", "v6Sb2B")}
        best = max(("v6Sb2A", "v6Sb2B"), key=lambda m: mean[m])
        ok = mean[best] > mean["v6Sb2"] and all(G_[(ds, best)] >= G_[(ds, "v6Sb2")] - 0.005 for ds in ("nusc", "av2"))
        sel = best if ok else "none (no variant qualified)"
        dev.append({"split": "selection", "pipeline": sel, "G": mean.get(best), "px_mean": np.nan})
    pd.DataFrame(dev).to_csv(OUT / "b2k_dev.csv", index=False)
    rows = []
    for ds in ("nusc", "av2", "ahd"):
        b = recs(B / "results" / "k31" / "r4" / f"{ds}_b2v")
        if b is None:
            continue
        b = b[b["gt"].astype(bool)]
        n = NWS[ds]
        vk = [c[2:-4] for c in b.columns if c.startswith("J_v6Sb2") and c.endswith("|v3a") and c != "J_v6Sb2|v3a"][0]
        keys = ["unit", "frame", "gt", f"J_wf960+C1|v3a", "px_wf960", f"J_wf{n}+C1|v3a", f"px_wf{n}", "J_tsp8@S+C1|v3a", "px_tsp8@S+C1"]
        s_ = recs(B / "results" / "k28" / "p3" / f"{ds}_test_instr", keys)
        s_ = s_[s_["gt"].astype(bool)].drop(columns="gt")
        d = b.merge(s_, on=["unit", "frame"])
        bt = Boot(d.unit.to_numpy())
        L = d["J_cheap"].to_numpy(float)
        rows.append({"dataset": DSL[ds], "row": f"{vk} (B2K variant)", "G": G(L, d[f"J_{vk}|v3a"]), "px_mean": d[f"px_{vk}"].mean(),
                     "frames": len(d), "units": d.unit.nunique()})
        for lab, jk, pk in (("v6-S <= 2 calls", "J_v6Sb2|v3a", "px_v6Sb2"), ("v6-S", "J_v6S|v3a", "px_v6S"),
                            ("u960 + C1 + v3a", "J_wf960+C1|v3a", "px_wf960"), ("native + C1 + v3a", f"J_wf{n}+C1|v3a", f"px_wf{n}"),
                            ("hybrid tsp8@S + C1 + v3a", "J_tsp8@S+C1|v3a", "px_tsp8@S+C1")):
            pt, lo, hi = bt.diff(L, d[f"J_{vk}|v3a"], d[jk])
            rows.append({"dataset": DSL[ds], "row": lab, "G": G(L, d[jk]), "px_mean": d[pk].mean(), "variant_minus_row": pt,
                         "ci_lo": lo, "ci_hi": hi, "excludes_0": bool(lo > 0 or hi < 0), "frames": len(d), "units": d.unit.nunique()})
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "b2k_test.csv", index=False)
    print("B2K selection:", sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="3,4")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if "3" in a.items.split(","):
        item3()
    if "4" in a.items.split(","):
        item4()
    if "b2k" in a.items.split(","):
        b2k()
    print("K31 R34 TABLES DONE")


if __name__ == "__main__":
    main()
