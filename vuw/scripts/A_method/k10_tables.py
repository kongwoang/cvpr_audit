#!/usr/bin/env python
"""K10 part 2 — post-hoc robustness check tables: the rows of handoff/K1/table_main_av2.csv and of the Argoverse-HD
table (K7) recomputed from the MASKED runs (shadow tree $B/k10_masked, k10_masked_run.py), next to the unmasked values.
G = sum(J_cheap - J_method) / sum(J_cheap) with the masked cheap pipeline as the reference (its all-cheap loss is
reported next to the unmasked one); intervals = log / sequence bootstrap (2,000) of ours final - method.
Router rows (official estimators, decision-value targets, oracle) and AdaScale come from av2_sota.py's masked score
stage (same scores / calibration; rho matched to ours' masked pixels), read from its summary.
usage: python k10_tables.py  ->  results/k10_tables/{table_main_av2_masked.csv, ahd_masked.csv}
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
S = B / "k10_masked" / "results"
OUT = B / "results" / "k10_tables"


def boot_idx(units, n=2000):
    uq = np.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
    rng = np.random.default_rng(0)
    return [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(n)]


def av2():
    ev = pd.read_pickle(S / "av2" / "eval__val_v4_final.pkl")
    ev0 = pd.read_pickle(B / "results" / "av2" / "eval__val_v4_final.pkl")
    bl = pd.read_pickle(S / "av2" / "baselines__val.pkl")
    nat = pd.read_pickle(next((S / "native").glob("*av2*.pkl")))
    ada = pd.read_pickle(S / "av2_sota" / "ada_eval_test.pkl")
    d = ev.merge(bl.drop(columns=["J_c"]), on=["log", "frame"]).merge(ada[["log", "frame", "J_ada", "net_px"]], on=["log", "frame"])
    nk = [c for c in ("J_u1280", "J_u1536", "J_u2048") if c in nat.columns]
    unit = next(c for c in ("log", "seq", "unit") if c in nat.columns)
    d = d.merge(nat.rename(columns={unit: "log"})[["log", "frame"] + nk].rename(columns={c: c + "_nat" for c in nk}), on=["log", "frame"])
    assert len(d) == len(ev) == 11818, (len(d), len(ev))
    L = d.J_c.to_numpy(); L0 = ev0.J_c.to_numpy()
    boots = boot_idx(d.log.to_numpy())
    ours = L - d.J_spec_ref_v4lite.to_numpy()
    PX = {"u640": 640 * 512, "u960": 960 * 768, "u1280": 1280 * 992, "u1536": 1536 * 1184, "native u2048": 2048 * 1568}
    rows = [("u640", "J_u640", PX["u640"] / (1550 * 2048)), ("u960", "J_u960", PX["u960"] / (1550 * 2048))]
    for c in nk:
        nm = {"J_u1280_nat": "u1280", "J_u1536_nat": "u1536", "J_u2048_nat": "native u2048"}.get(c + "_nat", c)
        if nm in PX:
            rows.append((nm, c + "_nat", PX[nm] / (1550 * 2048)))
    rows += [("AdaScale (re-impl., fit on AV2 train)", "J_ada", float(d.net_px.mean())),
             ("FOVEA KDE_I (960x768 canvas)", "J_fovea_kdei960", np.nan),
             ("SAHI (12 slices of 640x640 at 320 px + full pass)", "J_sahi_s047", np.nan),
             ("SAHI (12 slices of 640x640 at 640 px + full pass)", "J_sahi_s10", np.nan),
             ("uncertainty crops (strict budget = frozen v3's per-frame crop area)", "J_uncertainty", np.nan),
             ("temporal_crit crops (strict budget = frozen v3's per-frame crop area)", "J_temporal_crit", np.nan),
             ("random crops (strict budget = frozen v3's per-frame crop area)", "J_random", np.nan),
             ("ours without refiner (v4-lite)", "J_spec_v4lite", np.nan),
             ("nuScenes configuration v3 + refiner v1", "J_spec_ref_v3", np.nan)]
    px_ours = 320 * 256 / (1550 * 2048) + float(d.netpx_v4lite.mean())
    out = [{"method": "ours final (v4-lite + refiner v2)", "G_masked": ours.sum() / L.sum(), "network_px_masked": px_ours}]
    for nm, col, px in rows:
        if col not in d:
            continue
        V = L - d[col].to_numpy()
        dr = [(ours[b].sum() - V[b].sum()) / L[b].sum() for b in boots]
        out.append({"method": nm, "G_masked": V.sum() / L.sum(),
                    "network_px_masked": px if not (isinstance(px, float) and np.isnan(px)) else np.nan,
                    "ci_lo_ours_minus": np.percentile(dr, 2.5), "ci_hi_ours_minus": np.percentile(dr, 97.5)})
    for r in out:
        if r["method"] == "ours without refiner (v4-lite)":
            r["network_px_masked"] = px_ours
    # routers / oracle from av2_sota's masked score at rho matched to ours' masked pixels
    sm = pd.read_csv(S / "av2_sota" / "summary_av2val.csv")
    rc, rs = sm.rho_target.dropna().unique(), None
    m_cond = (px_ours - 320 * 256 / (1550 * 2048)) / (960 * 768 / (1550 * 2048))
    m_skip = (px_ours - 320 * 256 / (1550 * 2048)) / ((960 * 768 - 320 * 256) / (1550 * 2048))
    for name, rho, lab in (("oracle_V", m_cond, "oracle frame router (true V = J_cheap − J_u960, top-k)"),
                           ("skip_offbin", m_skip, "Budget-Adaptive Routing official: skip_offbin, seed 42"),
                           ("cond_moric", m_cond, "Budget-Adaptive Routing official: cond_moric, seed 42"),
                           ("skip_V", m_skip, "Budget-Adaptive Routing decision-value target: skip_V, seed 42"),
                           ("cond_V", m_cond, "Budget-Adaptive Routing decision-value target: cond_V, seed 42")):
        q = sm[(sm.method == name) & (np.isclose(sm.rho_target, round(rho, 4)))]
        if len(q):
            r = q.iloc[0]
            lo, hi = (float(x) for x in r.ours_minus.strip("[]").split(","))
            out.append({"method": lab, "G_masked": r.G, "network_px_masked": r.px, "ci_lo_ours_minus": lo, "ci_hi_ours_minus": hi,
                        "note": f"rho {rho:.3f} matched to ours' masked pixels"})
    res = pd.DataFrame(out)
    k1 = pd.read_csv(B / "results" / "handoff_K1" / "table_main_av2.csv") if (B / "results" / "handoff_K1" / "table_main_av2.csv").exists() else None
    if k1 is not None:
        res = res.merge(k1[["method", "G", "network_px_mean"]].rename(columns={"G": "G_unmasked", "network_px_mean": "network_px_unmasked"}),
                        on="method", how="left")
    res["all_cheap_loss_masked"] = L.sum(); res["all_cheap_loss_unmasked"] = L0.sum()
    res["label"] = "post-hoc robustness check (ego-hood mask rows >= 1830 applied to every pipeline)"
    return res


def ahd():
    g = pd.read_csv(S / "argoverse_hd" / "summary_G.csv")
    g0 = pd.read_csv(B / "results" / "argoverse_hd" / "summary_G.csv")
    m = g.merge(g0[["method", "G"]].rename(columns={"G": "G_unmasked"}), on="method", how="left")
    m["label"] = "post-hoc robustness check (bottom band rows >= 1140 masked; no ego part visible)"
    return m


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    a = av2(); a.to_csv(OUT / "table_main_av2_masked.csv", index=False)
    h = ahd(); h.to_csv(OUT / "ahd_masked.csv", index=False)
    pd.set_option("display.width", 250)
    print(a.drop(columns=["label"]).round(4).to_string(index=False))
    print(h.drop(columns=["label"]).round(4).to_string(index=False))
