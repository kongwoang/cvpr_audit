#!/usr/bin/env python
"""K28 phase 0 — v5.1-S items of K25 that need no new detector run (POST-HOC, test splits of K24; nothing selects
anything). Inputs: results/k24/<ds>_{main,supp}/*.pkl (K24 per-frame actions / J / px), stored baseline per-frame
results, results/k19/gt_<ds>.pkl (corridor objects), results/k28/p0abl_<ds>_p0 (the three ablation rows, k28_phase0_abl).
  --part sota  (4) v5.1-S minus AdaScale / FOVEA (KDE_I, KDE_S at 640 / 960; fine-tuned seeds) / SAHI per frame on
               nuScenes new test and AV2 val (ego mask), with and without refiner; J_cheap agreement checked
               -> sota_intervals_v51.csv
  --part loss  (5) loss terms (rap.planner.decision_cost: insufficient braking, excessive braking, command changes,
               collisions; their sum = J, checked), event rates per 1,000 frames with paired intervals of v5.1-S minus
               each row, multi-obstacle frames, per-scene / log sign tests -> loss_terms_{ds}_v51.csv,
               sign_tests_v51.csv, collisions_v51.csv
  --part cost  (2) cost statistics of v5.1-S (and v5.1-L / v5 where stored): tensor px mean / p50 / p95 / p99 / max,
               GFLOPs per frame (YOLOv8s), crops / guard re-detections / rounds (from the phase-0 run), retrospective
               crops -> cost_stats_v51.csv
  --part abl   (1) the ablation table with the three new rows and K24's support only / field only / 2 x 2
               -> ablation_v51.csv
Intervals: paired unit bootstrap, 2,000 resamples, seed 0.
usage: k28_phase0.py --part sota|loss|cost|abl
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from rap import planner as P

B = Path(os.environ["B"])
S = B / "k10_masked"
OUT = B / "results" / "k28" / "phase0"
pp, cp = P.PlannerParams(), P.CostParams()
EVENTS = {"missed hard brake": (2, 0), "delayed hard brake": (2, 1), "missed deceleration": (1, 0),
          "unnecessary deceleration": (0, 1), "unnecessary hard brake": (0, 2)}
NATIVE = {"nusc": 1600 * 900, "av2": 2048 * 1550, "ahd": 1920 * 1200}


def load(pat):
    rows = []
    for f in sorted(glob.glob(str(pat))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


def k24(ds):
    """K24 main + supp merged, previous action reconstructed (the draft's of the previous processed frame), gt frames"""
    d = load(B / "results" / "k24" / f"{ds}_main" / "*.pkl")
    s = load(B / "results" / "k24" / f"{ds}_supp" / "*.pkl")
    keep = ["unit", "frame"] + [c for c in s.columns if c.startswith(("J_", "a_", "px_")) and c not in d.columns]
    d = d.merge(s[keep], on=["unit", "frame"], how="inner")
    d = d.sort_values(["unit", "frame"]).reset_index(drop=True)
    d["prev_act"] = d.groupby("unit").a_cheap.shift(1)
    if ds == "ahd":
        d = d[d["gt"].astype(bool)].reset_index(drop=True)
    return d


class Boot:
    def __init__(self, units):
        uq = pd.unique(units); idx = [np.flatnonzero(units == x) for x in uq]
        rng = np.random.default_rng(0)
        self.bs = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        self.n = len(uq)

    def g(self, Jp, Jm, L):
        """(P - m) in G and its interval"""
        dd = np.array([(Jm[b] - Jp[b]).sum() / L[b].sum() for b in self.bs])
        return (Jm - Jp).sum() / L.sum(), np.percentile(dd, 2.5), np.percentile(dd, 97.5)


def verdict(lo):
    return "outperforms" if lo > 0 else ("matches" if lo > -0.02 else "below")


def sota():
    rows = []
    for ds in ("nusc", "av2"):
        d = k24(ds)
        if ds == "nusc":
            srcs = {"AdaScale (re-impl., fit on new train)": (B / "results/adascale/eval_newtest.pkl", "J_ada", "J_c"),
                    "FOVEA KDE_I 384x640": (B / "results/fovea/eval_newtest.pkl", "J_fovea_kdei640", "J_c"),
                    "FOVEA KDE_S 384x640": (B / "results/fovea/eval_newtest.pkl", "J_fovea_kdes640", "J_c"),
                    "FOVEA KDE_I 576x960": (B / "results/fovea/eval_newtest.pkl", "J_fovea_kdei960", "J_c"),
                    "FOVEA KDE_S 576x960": (B / "results/fovea/eval_newtest.pkl", "J_fovea_kdes960", "J_c"),
                    "SAHI 640-px slices (scale 0.6)": (B / "results/sahi/eval_newtest.pkl", "J_sahi_s06", "J_c"),
                    "SAHI 640-px slices (scale 1.0)": (B / "results/sahi/eval_newtest.pkl", "J_sahi_s10", "J_c")}
            for f in ("eval_ft_newtest.pkl", "eval_ft_newtest_s1.pkl"):
                p = B / "results" / "fovea" / f
                if p.exists():
                    for c in pd.read_pickle(p).columns:
                        if c.startswith("J_") and c != "J_c":
                            srcs[f"FOVEA fine-tuned {c[2:]} ({'seed 1' if 's1' in f else 'seed 0'})"] = (p, c, "J_c")
            key = "seq"
        else:
            srcs = {"AdaScale (fit on AV2 train)": (S / "results/av2_sota/ada_eval_test.pkl", "J_ada", "J_c"),
                    "SAHI slices (scale 0.47)": (S / "results/av2/baselines__val.pkl", "J_sahi_s047", "J_c"),
                    "SAHI slices (scale 1.0)": (S / "results/av2/baselines__val.pkl", "J_sahi_s10", "J_c"),
                    "FOVEA KDE_I 960": (S / "results/av2/baselines__val.pkl", "J_fovea_kdei960", "J_c")}
            key = "log"
        L = d.J_cheap.to_numpy(float)
        bt = Boot(d.unit.to_numpy())
        for lab, (p, col, jc) in srcs.items():
            o = pd.read_pickle(p).rename(columns={key: "unit"})
            m = d[["unit", "frame", "J_cheap", "J_v51S|none", "J_v51S|v3a"]].merge(o[["unit", "frame", jc, col]], on=["unit", "frame"], how="left")
            assert len(m) == len(d)
            miss = int(m[col].isna().sum())
            agree = float(np.mean(np.isclose(m[jc].to_numpy(float), L, equal_nan=False)))
            Jb = m[col].to_numpy(float)
            r = {"dataset": ds, "baseline": lab, "frames": len(d), "frames_missing": miss, "J_cheap_agreement": agree,
                 "G_baseline": (L - Jb).sum() / L.sum()}
            for set_, jp in (("no_refiner", "J_v51S|none"), ("refiner", "J_v51S|v3a")):
                Jp = m[jp].to_numpy(float)
                mdiff, lo, hi = bt.g(Jp, Jb, L)
                r.update({f"G_v51S_{set_}": (L - Jp).sum() / L.sum(), f"v51S_minus_{set_}": mdiff,
                          f"lo_{set_}": lo, f"hi_{set_}": hi, f"verdict_{set_}": verdict(lo)})
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "sota_intervals_v51.csv", index=False)
    print(t.round(4).to_string(index=False))


def terms(a, a_gt, prev):
    r = P.decision_cost(int(a), float(a_gt), None if prev is None or (isinstance(prev, float) and np.isnan(prev)) else int(prev), pp, cp)
    return (cp.lam_risk * r["shortfall"] ** 2, cp.lam_brake * r["excess"] ** 2, cp.lam_jerk * r["jerk"],
            cp.lam_collision * r["collision"], r["J"])


def loss():
    _s = importlib.util.spec_from_file_location("ka", B / "scripts" / "A_method" / "k24_analyze.py")
    KA = importlib.util.module_from_spec(_s); _s.loader.exec_module(KA)
    sc_rows, col_rows = [], []
    for ds in ("nusc", "av2", "ahd"):
        d = k24(ds)
        ab = load(B / "results" / "k28" / f"p0abl_{ds}_p0" / "*.pkl")[["unit", "frame", "a_gt", "prev_act"]]
        d = d.rename(columns={"prev_act": "prev_act_recon"}).merge(ab, on=["unit", "frame"], how="left")
        assert d.a_gt.notna().all(), ds
        prev_agree = float(np.mean([(a == b) or (pd.isna(a) and pd.isna(b)) for a, b in zip(d.prev_act, d.prev_act_recon)]))
        gt = pd.read_pickle(B / "results" / "k19" / f"gt_{ds}.pkl")
        d = d.merge(gt[["unit", "frame", "n_corridor", "min_ttc_corridor", "min_range_corridor"]], on=["unit", "frame"], how="left")
        M = KA.methods(d, ds)
        M.update(KA.routers(d, ds))
        pick = {"cheap draft": d.a_cheap.to_numpy(int)}
        for lab, e in M.items():
            if "extra" in e["section"]:
                continue
            pick[lab + " | with refiner (row setting)"] = np.asarray(e["Ar"], int)
            if e["group"] == "ours":
                pick[lab + " | no refiner"] = np.asarray(e["An"], int)
        ag = d.act_gt.to_numpy(int)
        a_gt_val = d.a_gt.to_numpy(float)
        T = {k: np.array([terms(a, g_, pv) for a, g_, pv in zip(v, a_gt_val, d.prev_act)]) for k, v in pick.items()}
        ref = "v5.1-S | with refiner (row setting)"
        chk = float(np.abs(T[ref][:, 4] - d["J_v51S|v3a"].to_numpy(float)).max())
        Lc = T["cheap draft"][:, 4].sum()
        units = d.unit.to_numpy()
        bt = Boot(units)
        multi = (d.n_corridor >= 2).to_numpy()
        a_ref = pick[ref]
        rows = []
        for k, a in pick.items():
            r = {"dataset": ds, "units": bt.n, "frames": len(d), "pipeline": k, "G": (Lc - T[k][:, 4].sum()) / Lc,
                 "loss_total": T[k][:, 4].sum()}
            for j, tn in enumerate(("insufficient_braking", "excessive_braking", "command_changes", "collisions")):
                r[f"{tn}_sum"] = T[k][:, j].sum(); r[f"{tn}_share_of_cheap"] = T[k][:, j].sum() / Lc
            r["collision_frames"] = int((T[k][:, 3] > 0).sum())
            r["collision_units"] = int(d.loc[T[k][:, 3] > 0, "unit"].nunique())
            r["unnecessary_per_1k"] = 1000 * (a > ag).mean(); r["missed_per_1k"] = 1000 * (a < ag).mean()
            for en, (req, tk) in EVENTS.items():
                ev = ((ag == req) & (a == tk)).astype(float)
                r[f"{en} /1000"] = 1000 * ev.mean()
                if k != ref:
                    eo = ((ag == req) & (a_ref == tk)).astype(float)
                    dd = [1000 * (eo[b].mean() - ev[b].mean()) for b in bt.bs]
                    r[f"{en} v51S-minus ci"] = f"[{np.percentile(dd, 2.5):+.2f}, {np.percentile(dd, 97.5):+.2f}]"
            Lm = T["cheap draft"][multi, 4].sum()
            r["multi_frames"] = int(multi.sum())
            r["G_multi"] = (Lm - T[k][multi, 4].sum()) / Lm if Lm > 0 else np.nan
            rows.append(r)
            m = T[k][:, 3] > 0
            col_rows.append({"dataset": ds, "pipeline": k, "collision_frames": int(m.sum()), "collision_units": int(d.loc[m, "unit"].nunique()),
                             "median_gt_min_ttc_corridor_s": float(np.median(d.loc[m, "min_ttc_corridor"])) if m.any() else np.nan,
                             "median_gt_min_range_corridor_m": float(np.median(d.loc[m, "min_range_corridor"])) if m.any() else np.nan})
        res = pd.DataFrame(rows)
        res["check_maxabs_J_v51S"] = chk
        res["prev_act_agreement_recon"] = prev_agree
        res.to_csv(OUT / f"loss_terms_{ds}_v51.csv", index=False)
        Ju = {k: pd.Series(T[k][:, 4]).groupby(units).sum() for k in pick}
        for k in pick:
            if k == ref:
                continue
            dlt = Ju[ref] - Ju[k]
            lo_, eq, hi_ = int((dlt < -1e-9).sum()), int((dlt.abs() <= 1e-9).sum()), int((dlt > 1e-9).sum())
            sc_rows.append({"dataset": ds, "against": k, "units": len(dlt), "v51S_lower": lo_, "equal": eq, "v51S_higher": hi_,
                            "sign_test_p": binomtest(lo_, lo_ + hi_).pvalue if lo_ + hi_ else np.nan})
        print(ds, "check", chk, res[["pipeline", "G"]].shape)
    pd.DataFrame(sc_rows).to_csv(OUT / "sign_tests_v51.csv", index=False)
    pd.DataFrame(col_rows).to_csv(OUT / "collisions_v51.csv", index=False)


def gflops_640():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    out = {}
    for m in ("yolov8s", "yolov8m", "yolov8l", "yolo11s", "yolo11m"):
        out[m] = float(get_flops(YOLO(str(B / "models" / f"{m}.pt")).model, imgsz=640))
    return out


def cost():
    gf = gflops_640()
    per_px = gf["yolov8s"] / (640 * 640)
    rows = []
    for ds in ("nusc", "av2", "ahd"):
        d = k24(ds)
        ab = load(B / "results" / "k28" / f"p0abl_{ds}_p0" / "*.pkl")
        if ds == "ahd" and len(ab):
            ab = ab[ab["gt"].astype(bool)]
        d = d.merge(ab[[c for c in ab.columns if c in ("unit", "frame") or c.split("|")[0].endswith("_full")]], on=["unit", "frame"], how="left") if len(ab) else d
        for lab, k in (("v5.1-S", "v51S"), ("v5.1-L", "v51L"), ("v5", "v5")):
            px = d[f"px_{k}"].to_numpy(float)
            r = {"dataset": ds, "configuration": lab, "frames": len(d), "tensor_px_mean": px.mean(),
                 "tensor_px_p50": np.median(px), "tensor_px_p95": np.quantile(px, .95), "tensor_px_p99": np.quantile(px, .99),
                 "tensor_px_max": px.max(), "GFLOPs_mean": (px * NATIVE[ds] * per_px).mean(),
                 "GFLOPs_p95": np.quantile(px * NATIVE[ds] * per_px, .95), "draft_px": d.px_cheap.mean(),
                 "share_no_crop": float(np.mean(np.isclose(px, d.px_cheap.to_numpy(float)))),
                 "rounds_mean": d[f"rounds_{k}"].mean(), **{f"share_rounds_{i}": float((d[f"rounds_{k}"] == i).mean()) for i in range(4)}}
            if f"pxretro_{k}" in d:
                r["retro_px_mean"], r["retro_crops_mean"] = d[f"pxretro_{k}"].mean(), d[f"nretro_{k}"].mean()
            if k == "v51S" and "ncrops_full" in d:
                r["parity_px_with_K24"] = float(np.mean(np.isclose(d.px_full, d.px_v51S)))
                r["parity_J_with_K24"] = float(np.mean(np.isclose(d["J_full|v3a"], d["J_v51S|v3a"])))
                for c, nm in (("ncrops_full", "crops"), ("nguard_full", "guard_redet")):
                    r[f"{nm}_mean"], r[f"{nm}_p95"], r[f"{nm}_max"] = d[c].mean(), d[c].quantile(.95), d[c].max()
            rows.append(r)
    t = pd.DataFrame(rows)
    t["yolov8s_GFLOPs_640"] = gf["yolov8s"]
    t.to_csv(OUT / "cost_stats_v51.csv", index=False)
    pd.DataFrame([gf]).to_csv(OUT / "gflops_640.csv", index=False)
    print(t.round(4).to_string(index=False))


def abl():
    rows = []
    for ds in ("nusc", "av2", "ahd"):
        d = k24(ds)
        ab = load(B / "results" / "k28" / f"p0abl_{ds}_p0" / "*.pkl")
        if ds == "ahd":
            ab = ab[ab["gt"].astype(bool)]
        keep = ["unit", "frame"] + [c for c in ab.columns if c.split("_", 1)[0] in ("J", "a", "px", "rounds", "ncrops", "nguard")
                                    and c.split("|")[0].rsplit("_", 1)[-1] in ("full", "pivots", "one", "noguard")]
        d = d.merge(ab[keep], on=["unit", "frame"], how="inner")
        assert len(d) == len(k24(ds)), ds
        L = d.J_cheap.to_numpy(float)
        bt = Boot(d.unit.to_numpy())
        Jp = d["J_v51S|v3a"].to_numpy(float)
        cells = [("v5.1-S (full)", "v51S", "v3a"), ("v5.1-S, phase-0 re-run (parity)", "full", "v3a"),
                 ("pivots instead of support", "pivots", "v3a"), ("single round", "one", "v3a"), ("without guard", "noguard", "v3a"),
                 ("support only + components", "so51S", "v3a"), ("field only + components", "fo51S", "v3a"),
                 ("without refiner (T4 only)", "v51S", "none"), ("without T4 (= v5)", "v5", "v3a"),
                 ("without refiner and without T4", "v5", "none"),
                 ("pivots, no refiner", "pivots", "none"), ("single round, no refiner", "one", "none"), ("without guard, no refiner", "noguard", "none")]
        g = d.act_gt.to_numpy(int)
        reg = ((d.act_gt >= 1) & (d.a_cheap < d.act_gt)).to_numpy()
        for lab, k, r_ in cells:
            J = d[f"J_{k}|{r_}"].to_numpy(float); A_ = d[f"a_{k}|{r_}"].to_numpy(int)
            mdiff, lo, hi = bt.g(Jp, J, L)
            rows.append({"dataset": ds, "row": lab, "refiner": r_, "G": (L - J).sum() / L.sum(),
                         "tensor_px_mean": d[f"px_{k}"].mean(), "v51S_minus": mdiff, "lo": lo, "hi": hi, "verdict": verdict(lo),
                         "G_K17_3_regime": (L[reg] - J[reg]).sum() / L[reg].sum(),
                         "unnecessary_per_1k": 1000 * (A_ > g).mean(), "missed_per_1k": 1000 * (A_ < g).mean(),
                         "crops_mean": d[f"ncrops_{k}"].mean() if f"ncrops_{k}" in d else np.nan,
                         "rounds_mean": d[f"rounds_{k}"].mean() if f"rounds_{k}" in d else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "ablation_v51.csv", index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["sota", "loss", "cost", "abl"])
    OUT.mkdir(parents=True, exist_ok=True)
    {"sota": sota, "loss": loss, "cost": cost, "abl": abl}[ap.parse_args().part]()
