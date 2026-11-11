#!/usr/bin/env python
"""K31 item 2 (post-hoc; stored records, pre-registered in handoff/K31/STATUS.md 2026-10-19): intervals for the consumer-
adaptivity claim. Frames joined on (unit, frame) across the runs (identical draft and units). Writes results/k31/tables/:
  consumer_ttc.csv     2a TTC-warning consumer (J2): DESP's TTC loop c2|v3a (K28 phase 3) minus the hybrid with the TTC
                       support at the S budget (K30 match), at the TTC loop's px (K31 re-run R2a, when present), at v6-S's px
                       (K30, labelled extra), unc2 at the TTC loop's px, u960 / native + C1 + v3a, the frozen v6-S
  consumer_idm.csv     2b IDM consumer (J3, K30 part idm): ours-IDM and the frozen v6-S minus the hybrid with the IDM
                       support at ours-IDM's px and at the S budget (K31 re-run R2b, when present), unc2, u960 / native
  idm_mechanism.csv    ours-IDM vs frozen v6-S: crops, px, field size (R2b, when present), differing frames, decisive objects
usage: k31_consumers.py
"""
from __future__ import annotations

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
            rows.append(r if cols is None else {c: r.get(c) for c in cols if c in r})
    if not rows:
        return None
    d = pd.DataFrame(rows)
    return d[d["gt"].astype(bool)].reset_index(drop=True)


class Boot:
    def __init__(self, units, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        self.uq, self.inv = np.unique(np.asarray(units), return_inverse=True)
        self.bs = [rng.integers(0, len(self.uq), len(self.uq)) for _ in range(n)]

    def sums(self, x):
        return np.bincount(self.inv, np.asarray(x, float), len(self.uq))

    def diff(self, L, Ja, Jb):
        """G(a) - G(b) = (sum Jb - sum Ja) / sum L"""
        l, a, b = self.sums(L), self.sums(Ja), self.sums(Jb)
        v = [(b[s].sum() - a[s].sum()) / l[s].sum() for s in self.bs]
        return float((b.sum() - a.sum()) / l.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def G(L, J):
    L, J = np.asarray(L, float), np.asarray(J, float)
    return float((L.sum() - J.sum()) / L.sum())


def ttc(ds):
    n = NWS[ds]
    keys = ["unit", "frame", "gt", "J2_cheap", "px_cheap", "J2_c2|v3a", "px_c2", "J2_unc2@c2+C1|v3a", "px_unc2@c2+C1",
            "J2_wf960+C1|v3a", "px_wf960", f"J2_wf{n}+C1|v3a", f"px_wf{n}", "J2_v6S|v3a", "px_v6S"]
    d = recs(B / "results" / "k28" / "p3" / f"{ds}_test_instr", keys)
    m = recs(B / "results" / "k30" / "match" / ds, ["unit", "frame", "gt", "J2_tsp8c2@S+C1|v3a", "px_tsp8c2@S+C1",
                                                     "J2_tsp8c2@x1.00+C1|v3a", "px_tsp8c2@x1.00+C1"])
    d = d.merge(m.drop(columns="gt"), on=["unit", "frame"], how="inner")
    r = recs(B / "results" / "k31" / "r2" / f"{ds}_ttcpx", ["unit", "frame", "gt", "J2_tsp8c2@c2+C1|v3a", "px_tsp8c2@c2+C1",
                                                             "J2_c2|v3a"])
    rows_ = [("(i) hybrid tsp8, TTC support, K28 S budget", "tsp8c2@S+C1|v3a", "tsp8c2@S+C1"),
             ("(i) hybrid tsp8, TTC support, TTC loop's px (x1.0)", "tsp8c2@c2+C1|v3a", "tsp8c2@c2+C1"),
             ("(ii) uncertainty unc2 at the TTC loop's px", "unc2@c2+C1|v3a", "unc2@c2+C1"),
             ("(iii) u960 + C1 + v3a", "wf960+C1|v3a", "wf960"),
             (f"(iii) native + C1 + v3a", f"wf{n}+C1|v3a", f"wf{n}"),
             ("extra: hybrid tsp8, TTC support, v6-S's px (x1.0)", "tsp8c2@x1.00+C1|v3a", "tsp8c2@x1.00+C1"),
             ("extra: frozen v6-S (braking-driven)", "v6S|v3a", "v6S")]
    if r is not None:
        d = d.merge(r.drop(columns="gt").rename(columns={"J2_c2|v3a": "J2_c2|v3a_r2"}), on=["unit", "frame"], how="left")
    bt = Boot(d.unit.to_numpy())
    L = d["J2_cheap"].to_numpy(float)
    out = [{"dataset": DSL[ds], "row": "DESP TTC loop (c2|v3a)", "G2": G(L, d["J2_c2|v3a"]), "px_mean": d["px_c2"].mean(),
            "frames": len(d), "units": d.unit.nunique()}]
    if r is not None and "J2_c2|v3a_r2" in d:
        ok = d["J2_c2|v3a_r2"].notna()
        out[0]["rerun_parity_frames"] = int(ok.sum())
        out[0]["rerun_parity_equal_J2"] = int((d.loc[ok, "J2_c2|v3a_r2"] == d.loc[ok, "J2_c2|v3a"]).sum())
    for lab, jk, pk in rows_:
        if f"J2_{jk}" not in d or d[f"J2_{jk}"].isna().any():
            out.append({"dataset": DSL[ds], "row": lab, "status": "missing (re-run R2a not yet done)" if "c2+C1|v3a" in jk and "tsp8" in jk else "missing"})
            continue
        pt, lo, hi = bt.diff(L, d["J2_c2|v3a"], d[f"J2_{jk}"])
        out.append({"dataset": DSL[ds], "row": lab, "G2": G(L, d[f"J2_{jk}"]), "px_mean": d[f"px_{pk}"].mean(),
                    "loop_minus_row": pt, "ci_lo": lo, "ci_hi": hi, "excludes_0": bool(lo > 0 or hi < 0),
                    "frames": len(d), "units": d.unit.nunique()})
    return out


def idm(ds):
    n = NWS[ds]
    d = recs(B / "results" / "k30" / "idm" / ds)
    r = recs(B / "results" / "k31" / "r2" / f"{ds}_idmS")
    if r is not None:
        keep = ["unit", "frame"] + [c for c in r.columns if c.startswith(("J3_tsp8@S", "px_tsp8@S", "field_", "fieldpx_", "a3_v6S|v3a",
                                                                          "a3_oursIDM|v3a", "dec3_z_", "dec3_ttc_", "dec3_from_crop_", "dec3_refined_"))]
        d = d.merge(r[keep].rename(columns={"a3_v6S|v3a": "a3_v6S|v3a_r2", "a3_oursIDM|v3a": "a3_oursIDM|v3a_r2"}),
                    on=["unit", "frame"], how="left")
    bt = Boot(d.unit.to_numpy())
    L = d["J3_cheap"].to_numpy(float)
    rows_ = [("(i) hybrid tsp8, IDM support, ours-IDM's px", "tsp8@IDM+C1|v3a", "tsp8@IDM+C1"),
             ("(i) hybrid tsp8, IDM support, K28 S budget", "tsp8@S+C1|v3a", "tsp8@S+C1"),
             ("(ii) uncertainty unc2 at ours-IDM's px", "unc2@IDM+C1|v3a", "unc2@IDM+C1"),
             ("(iii) u960 + C1 + v3a", "wf960+C1|v3a", "wf960"),
             ("(iii) native + C1 + v3a", f"wf{n}+C1|v3a", f"wf{n}")]
    out = []
    for loop, lk, lpx in (("ours-IDM", "oursIDM|v3a", "oursIDM"), ("frozen v6-S", "v6S|v3a", "v6S")):
        out.append({"dataset": DSL[ds], "loop": loop, "row": "(loop itself)", "G3": G(L, d[f"J3_{lk}"]), "px_mean": d[f"px_{lpx}"].mean(),
                    "frames": len(d), "units": d.unit.nunique()})
        for lab, jk, pk in rows_:
            if f"J3_{jk}" not in d or d[f"J3_{jk}"].isna().any():
                out.append({"dataset": DSL[ds], "loop": loop, "row": lab, "status": "missing (re-run R2b not yet done)"})
                continue
            pt, lo, hi = bt.diff(L, d[f"J3_{lk}"], d[f"J3_{jk}"])
            out.append({"dataset": DSL[ds], "loop": loop, "row": lab, "G3": G(L, d[f"J3_{jk}"]), "px_mean": d[f"px_{pk}"].mean(),
                        "loop_minus_row": pt, "ci_lo": lo, "ci_hi": hi, "excludes_0": bool(lo > 0 or hi < 0),
                        "frames": len(d), "units": d.unit.nunique()})
        if loop == "ours-IDM":
            pt, lo, hi = bt.diff(L, d["J3_oursIDM|v3a"], d["J3_v6S|v3a"])
            out.append({"dataset": DSL[ds], "loop": loop, "row": "frozen v6-S", "G3": G(L, d["J3_v6S|v3a"]), "px_mean": d["px_v6S"].mean(),
                        "loop_minus_row": pt, "ci_lo": lo, "ci_hi": hi, "excludes_0": bool(lo > 0 or hi < 0),
                        "frames": len(d), "units": d.unit.nunique()})
    # mechanism
    ai, a6, ag = d["a3_oursIDM|v3a"].astype(int), d["a3_v6S|v3a"].astype(int), d["act3_gt"].astype(int)
    df = ai != a6
    mech = {"dataset": DSL[ds], "frames": len(d), "crops_per_frame oursIDM": d["ncrops_oursIDM"].mean(),
            "crops_per_frame v6S": d["ncrops_v6S"].mean(), "px oursIDM": d["px_oursIDM"].mean(), "px v6S": d["px_v6S"].mean(),
            "share frames oursIDM 0 crops": float((d["ncrops_oursIDM"] == 0).mean()),
            "share frames v6S 0 crops": float((d["ncrops_v6S"] == 0).mean()),
            "differing frames": int(df.sum()), "differing share": float(df.mean()),
            "oursIDM higher": int((ai > a6).sum()), "oursIDM lower": int((ai < a6).sum()),
            "on differing: oursIDM = act3_gt": int((df & (ai == ag)).sum()), "on differing: v6S = act3_gt": int((df & (a6 == ag)).sum()),
            "cost oursIDM - v6S on differing (sum J3)": float((d["J3_oursIDM|v3a"] - d["J3_v6S|v3a"])[df].sum()),
            "cost oursIDM - v6S total (sum J3)": float((d["J3_oursIDM|v3a"] - d["J3_v6S|v3a"]).sum()),
            "sum J3_cheap": float(L.sum())}
    if "dec3_found_oursIDM" in d:
        has = df & d["dec3"].notna() & (d["dec3"].astype(float) >= 0)
        mech["differing with decisive IDM object"] = int(has.sum())
        mech["  found by oursIDM only"] = int((has & d["dec3_found_oursIDM"].astype(bool) & ~d["dec3_found_v6S"].astype(bool)).sum())
        mech["  found by v6S only"] = int((has & ~d["dec3_found_oursIDM"].astype(bool) & d["dec3_found_v6S"].astype(bool)).sum())
        mech["  found by both"] = int((has & d["dec3_found_oursIDM"].astype(bool) & d["dec3_found_v6S"].astype(bool)).sum())
        mech["  found by neither"] = int((has & ~d["dec3_found_oursIDM"].astype(bool) & ~d["dec3_found_v6S"].astype(bool)).sum())
    if "dec3_z_oursIDM" in d and "dec3_found_oursIDM" in d:
        both = df & d["dec3"].notna() & (d["dec3"].astype(float) >= 0) & d["dec3_found_oursIDM"].astype(bool) & d["dec3_found_v6S"].astype(bool)
        x = d[both]
        zi, z6 = pd.to_numeric(x["dec3_z_oursIDM"], errors="coerce"), pd.to_numeric(x["dec3_z_v6S"], errors="coerce")
        ti, t6 = pd.to_numeric(x["dec3_ttc_oursIDM"], errors="coerce"), pd.to_numeric(x["dec3_ttc_v6S"], errors="coerce")
        ri, r6 = x["dec3_refined_oursIDM"].astype(bool), x["dec3_refined_v6S"].astype(bool)
        mech.update({"both found: frames": int(both.sum()),
                     "both found: est. range differs > 1%": int((np.abs(np.log(zi / z6)) > 0.01).sum()),
                     "both found: oursIDM range shorter": int((zi < z6 * 0.99).sum()), "both found: oursIDM range longer": int((zi > z6 * 1.01).sum()),
                     "both found: est. TTC differs": int((~np.isclose(ti.fillna(1e9), t6.fillna(1e9), rtol=0.01)).sum()),
                     "both found: refined by oursIDM only": int((ri & ~r6).sum()), "both found: refined by v6S only": int((~ri & r6).sum()),
                     "both found: refined by both": int((ri & r6).sum()), "both found: refined by neither": int((~ri & ~r6).sum()),
                     "both found: from a crop in oursIDM only": int((x["dec3_from_crop_oursIDM"].astype(bool) & ~x["dec3_from_crop_v6S"].astype(bool)).sum()),
                     "both found: from a crop in v6S only": int((~x["dec3_from_crop_oursIDM"].astype(bool) & x["dec3_from_crop_v6S"].astype(bool)).sum())})
    for c in [c for c in d.columns if c.startswith(("field_", "fieldpx_"))]:
        mech[f"mean {c}"] = float(pd.to_numeric(d[c], errors="coerce").mean())
    if "a3_v6S|v3a_r2" in d:
        ok = d["a3_v6S|v3a_r2"].notna()
        mech["R2b parity: frames"] = int(ok.sum())
        mech["R2b parity: v6S action equal"] = int((d.loc[ok, "a3_v6S|v3a_r2"] == d.loc[ok, "a3_v6S|v3a"]).sum())
        mech["R2b parity: oursIDM action equal"] = int((d.loc[ok, "a3_oursIDM|v3a_r2"] == d.loc[ok, "a3_oursIDM|v3a"]).sum())
    return out, mech


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T, I, M = [], [], []
    for ds in ("nusc", "av2", "ahd"):
        T += ttc(ds)
        o, m = idm(ds)
        I += o; M.append(m)
        print(ds, "done", flush=True)
    pd.DataFrame(T).to_csv(OUT / "consumer_ttc.csv", index=False)
    pd.DataFrame(I).to_csv(OUT / "consumer_idm.csv", index=False)
    pd.DataFrame(M).to_csv(OUT / "idm_mechanism.csv", index=False)
    print("K31 CONSUMERS DONE")


if __name__ == "__main__":
    main()
