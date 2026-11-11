#!/usr/bin/env python
"""K28 PHASE 3 (post-hoc, reporting only) — phase 0's v5.1-S items redone for the final configuration v6 (S, and M / L
where the table has a budget axis), from stored per-frame results only (no detector run here):
  --part sota  (4) v6 minus AdaScale / FOVEA (KDE_I, KDE_S; fine-tuned seeds) / SAHI per frame on nuScenes new test and AV2
               val (ego mask), with and without refiner -> sota_intervals_v6.csv
  --part loss  (5) loss terms, event rates per 1,000 frames with paired intervals of v6-S minus each row, multi-obstacle
               frames, per-scene / log sign tests -> loss_terms_{ds}_v6.csv, sign_tests_v6.csv, collisions_v6.csv
  --part cost  (2) cost statistics of v6-S / M / L (tensor px, GFLOPs with each crop detector, crops, guard re-detections,
               rounds, detector calls) -> cost_stats_v6.csv
  --part abl   (1) ablation of v6-S: phase-2 rows (support only / field only, v5 = without C1, v5.1-S = T4 instead of C1,
               without refiner) and phase-3 rows (pivots, single round, without guard; k28_p3.py instr) -> ablation_v6.csv
Inputs: K24 (k28_phase0.k24), results/k28/<ds>_test_v6{ours,abl,basewf-yolo11m,baseheur}, results/k28/p3/<ds>_test_instr.
Intervals: paired unit bootstrap, 2,000 resamples, seed 0 (k28_phase0.Boot); verdict as K24.
usage: k28_p3_p0.py --part sota|loss|cost|abl|all
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("p0", B / "scripts" / "A_method" / "k28_phase0.py")
P0 = importlib.util.module_from_spec(_s); _s.loader.exec_module(P0)
load, Boot, verdict, terms, EVENTS, NATIVE, S, pp, cp = P0.load, P0.Boot, P0.verdict, P0.terms, P0.EVENTS, P0.NATIVE, P0.S, P0.pp, P0.cp
OUT = B / "results" / "k28" / "p3" / "tables"
R = B / "results" / "k28"
V6 = {"v6-S": ("v6S", "v3a", "yolov8s"), "v6-M": ("v6M", "own-yolov8m", "yolov8m"), "v6-L": ("v6L", "own-yolo11m", "yolo11m")}
CFG_S = ("unc2@S", "unc4@S", "tcr0@S", "tcr3@S", "cor0@S", "cor8@S", "tsp2@S", "tsp8@S", "rnd@S")


def k24v6(ds, p3=False):
    """K24 main + supp (k28_phase0.k24) + the phase-2 parts (ours, abl, YOLOv8s whole frame, YOLOv8s heuristics) and,
    with p3, the phase-3 instr rows pv6S / or6S / ng6S; clashing columns of the K28 pass get the suffix @k28"""
    d = P0.k24(ds)
    n0, base = len(d), set(d.columns)
    parts = [R / f"{ds}_test_v6ours", R / f"{ds}_test_v6abl", R / f"{ds}_test_v6basewf-yolo11m", R / f"{ds}_test_v6baseheur"]
    if p3:
        parts.append(R / "p3" / f"{ds}_test_instr")
    for p in parts:
        x = load(p / "*.pkl")
        if p.name.endswith("instr"):
            x = x[["unit", "frame"] + [c for c in x.columns if re.match(r"^(J|a|px|rounds|ncrops|nguard|ncalls)_(pv6S|or6S|ng6S)(\||$)", c)]]
        if p.name.startswith(f"{ds}_test_v6basewf"):
            x = x[[c for c in x.columns if not re.match(r"^[a-zA-Z]+_vwf", c)]]
        keep = ["unit", "frame"] + [c for c in x.columns if c.startswith(("J_", "a_", "px_", "cpx_", "rounds_", "ncrops_", "nguard_", "ncalls_"))
                                    and c != "J_cheap" and c != "a_cheap" and c != "px_cheap"]
        x = x[keep].rename(columns={c: f"{c}@k28" for c in keep[2:] if c in base})
        x = x[["unit", "frame"] + [c for c in x.columns[2:] if c not in d.columns]]   # e.g. v5.1-S of both ours and abl: kept once
        d = d.merge(x, on=["unit", "frame"], how="left")
        assert len(d) == n0, (ds, p)
    return d


def sota():
    rows = []
    for ds in ("nusc", "av2"):
        d = k24v6(ds)
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
            m = d[["unit", "frame", "J_cheap"] + [f"J_{k}|{s_}" for k, rf, _ in V6.values() for s_ in ("none", rf)]].merge(
                o[["unit", "frame", jc, col]], on=["unit", "frame"], how="left")
            assert len(m) == len(d)
            Jb = m[col].to_numpy(float)
            r = {"dataset": ds, "baseline": lab, "frames": len(d), "frames_missing": int(m[col].isna().sum()),
                 "J_cheap_agreement": float(np.mean(np.isclose(m[jc].to_numpy(float), L))), "G_baseline": (L - Jb).sum() / L.sum()}
            for P_, (k, rf, _) in V6.items():
                for set_, jp in (("no_refiner", f"J_{k}|none"), ("refiner", f"J_{k}|{rf}")):
                    Jp = m[jp].to_numpy(float)
                    mdiff, lo, hi = bt.g(Jp, Jb, L)
                    tg = k
                    r.update({f"G_{tg}_{set_}": (L - Jp).sum() / L.sum(), f"{tg}_minus_{set_}": mdiff, f"{tg}_lo_{set_}": lo,
                              f"{tg}_hi_{set_}": hi, f"{tg}_verdict_{set_}": verdict(lo)})
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "sota_intervals_v6.csv", index=False)
    print(t.round(4).to_string(index=False))


def loss():
    _s = importlib.util.spec_from_file_location("ka", B / "scripts" / "A_method" / "k24_analyze.py")
    KA = importlib.util.module_from_spec(_s); _s.loader.exec_module(KA)
    sc_rows, col_rows = [], []
    for ds in ("nusc", "av2", "ahd"):
        d = k24v6(ds)
        ab = load(B / "results" / "k28" / f"p0abl_{ds}_p0" / "*.pkl")[["unit", "frame", "a_gt", "prev_act"]]
        d = d.rename(columns={"prev_act": "prev_act_recon"}).merge(ab, on=["unit", "frame"], how="left")
        assert d.a_gt.notna().all(), ds
        gt = pd.read_pickle(B / "results" / "k19" / f"gt_{ds}.pkl")
        d = d.merge(gt[["unit", "frame", "n_corridor", "min_ttc_corridor", "min_range_corridor"]], on=["unit", "frame"], how="left")
        M = KA.methods(d, ds)
        M.update(KA.routers(d, ds))
        pick = {"cheap draft": d.a_cheap.to_numpy(int)}
        for P_, (k, rf, _) in V6.items():
            pick[f"{P_} | with refiner"] = d[f"a_{k}|{rf}"].to_numpy(int)
            pick[f"{P_} | no refiner"] = d[f"a_{k}|none"].to_numpy(int)
        for lab, e in M.items():
            if "extra" in e["section"]:
                continue
            pick[lab + " | with refiner (row setting)"] = np.asarray(e["Ar"], int)
            if e["group"] == "ours":
                pick[lab + " | no refiner"] = np.asarray(e["An"], int)
        for c in sorted(d.columns):                       # K28 controls + C1 with refiner (YOLOv8s)
            m_ = re.match(r"^a_((wf\d+)|((unc|tcr|cor|tsp|rnd)\d*@[SML]))\+C1\|v3a$", c)
            if m_:
                pick[f"K28 {m_.group(1)} + C1 | with refiner"] = d[c].to_numpy(int)
        ag = d.act_gt.to_numpy(int)
        a_gt_val = d.a_gt.to_numpy(float)
        T = {k: np.array([terms(a, g_, pv) for a, g_, pv in zip(v, a_gt_val, d.prev_act)]) for k, v in pick.items()}
        ref = "v6-S | with refiner"
        chk = float(np.abs(T[ref][:, 4] - d["J_v6S|v3a"].to_numpy(float)).max())
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
                    r[f"{en} v6S-minus ci"] = f"[{np.percentile(dd, 2.5):+.2f}, {np.percentile(dd, 97.5):+.2f}]"
            Lm = T["cheap draft"][multi, 4].sum()
            r["multi_frames"] = int(multi.sum())
            r["G_multi"] = (Lm - T[k][multi, 4].sum()) / Lm if Lm > 0 else np.nan
            rows.append(r)
            m = T[k][:, 3] > 0
            col_rows.append({"dataset": ds, "pipeline": k, "collision_frames": int(m.sum()), "collision_units": int(d.loc[m, "unit"].nunique()),
                             "median_gt_min_ttc_corridor_s": float(np.median(d.loc[m, "min_ttc_corridor"])) if m.any() else np.nan,
                             "median_gt_min_range_corridor_m": float(np.median(d.loc[m, "min_range_corridor"])) if m.any() else np.nan})
        res = pd.DataFrame(rows)
        res["check_maxabs_J_v6S"] = chk
        res.to_csv(OUT / f"loss_terms_{ds}_v6.csv", index=False)
        Ju = {k: pd.Series(T[k][:, 4]).groupby(units).sum() for k in pick}
        for k in pick:
            if k == ref:
                continue
            dlt = Ju[ref] - Ju[k]
            lo_, eq, hi_ = int((dlt < -1e-9).sum()), int((dlt.abs() <= 1e-9).sum()), int((dlt > 1e-9).sum())
            sc_rows.append({"dataset": ds, "against": k, "units": len(dlt), "v6S_lower": lo_, "equal": eq, "v6S_higher": hi_,
                            "sign_test_p": binomtest(lo_, lo_ + hi_).pvalue if lo_ + hi_ else np.nan})
        print(ds, "check", chk, len(rows))
    pd.DataFrame(sc_rows).to_csv(OUT / "sign_tests_v6.csv", index=False)
    pd.DataFrame(col_rows).to_csv(OUT / "collisions_v6.csv", index=False)


def cost():
    gf = pd.read_csv(B / "results" / "k28" / "phase0" / "gflops_640.csv").iloc[0].to_dict()
    rows = []
    for ds in ("nusc", "av2", "ahd"):
        d = k24v6(ds)
        for lab, (k, rf, det) in list(V6.items()) + [("v5.1-S", ("v51S", "v3a", "yolov8s"))]:
            px = d[f"px_{k}"].to_numpy(float)
            pc = d.px_cheap.to_numpy(float)
            gfl = (pc * gf["yolov8s"] + (px - pc) * gf[det]) * NATIVE[ds] / 640 ** 2
            r = {"dataset": ds, "configuration": lab, "crop_detector": det, "frames": len(d), "tensor_px_mean": px.mean(),
                 "tensor_px_p50": np.median(px), "tensor_px_p95": np.quantile(px, .95), "tensor_px_p99": np.quantile(px, .99),
                 "tensor_px_max": px.max(), "GFLOPs_mean": gfl.mean(), "GFLOPs_p95": np.quantile(gfl, .95), "GFLOPs_max": gfl.max(),
                 "draft_px": pc.mean(), "share_no_crop": float(np.mean(np.isclose(px, pc))),
                 "rounds_mean": d[f"rounds_{k}"].mean() if f"rounds_{k}" in d else np.nan}
            for c, nm in ((f"ncrops_{k}", "crops"), (f"nguard_{k}", "guard_redet"), (f"ncalls_{k}", "detector_calls")):
                if c in d:
                    r[f"{nm}_mean"], r[f"{nm}_p95"], r[f"{nm}_max"] = d[c].mean(), d[c].quantile(.95), d[c].max()
            if f"rounds_{k}" in d:
                r.update({f"share_rounds_{i}": float((d[f"rounds_{k}"] == i).mean()) for i in range(4)})
            rows.append(r)
    t = pd.DataFrame(rows)
    for m_, v_ in gf.items():
        t[f"GFLOPs_640_{m_}"] = v_
    t.to_csv(OUT / "cost_stats_v6.csv", index=False)
    print(t.round(4).to_string(index=False))


def abl():
    rows = []
    for ds in ("nusc", "av2", "ahd"):
        d = k24v6(ds, p3=True)
        L = d.J_cheap.to_numpy(float)
        bt = Boot(d.unit.to_numpy())
        Jp = d["J_v6S|v3a"].to_numpy(float)
        cells = [("v6-S (full)", "v6S", "v3a", "-"), ("support only (no field)", "so6S", "v3a", "field"),
                 ("field only (no support)", "fo6S", "v3a", "support"), ("pivots instead of support", "pv6S", "v3a", "support (pivots)"),
                 ("single round", "or6S", "v3a", "rounds"), ("without guard", "ng6S", "v3a", "guard"),
                 ("without C1 (= v5 + v3a)", "v5", "v3a", "looming estimate"), ("T4 instead of C1 (= v5.1-S)", "v51S", "v3a", "C1 vs T4"),
                 ("without refiner", "v6S", "none", "refiner"), ("without refiner and without C1 (= v5)", "v5", "none", "refiner + looming"),
                 ("support only, no refiner", "so6S", "none", "-"), ("field only, no refiner", "fo6S", "none", "-"),
                 ("pivots, no refiner", "pv6S", "none", "-"), ("single round, no refiner", "or6S", "none", "-"),
                 ("without guard, no refiner", "ng6S", "none", "-")]
        g = d.act_gt.to_numpy(int)
        reg = ((d.act_gt >= 1) & (d.a_cheap < d.act_gt)).to_numpy()
        for lab, k, r_, comp in cells:
            if f"J_{k}|{r_}" not in d or d[f"J_{k}|{r_}"].isna().any():
                rows.append({"dataset": ds, "row": lab, "refiner": r_, "component": comp, "missing": True}); continue
            J = d[f"J_{k}|{r_}"].to_numpy(float); A_ = d[f"a_{k}|{r_}"].to_numpy(int)
            mdiff, lo, hi = bt.g(Jp, J, L)
            rows.append({"dataset": ds, "row": lab, "refiner": r_, "component": comp, "G": (L - J).sum() / L.sum(),
                         "tensor_px_mean": d[f"px_{k}"].mean() if f"px_{k}" in d else np.nan, "v6S_minus": mdiff, "lo": lo, "hi": hi,
                         "verdict": verdict(lo), "G_K17_3_regime": (L[reg] - J[reg]).sum() / L[reg].sum(),
                         "unnecessary_per_1k": 1000 * (A_ > g).mean(), "missed_per_1k": 1000 * (A_ < g).mean(),
                         "crops_mean": d[f"ncrops_{k}"].mean() if f"ncrops_{k}" in d else np.nan,
                         "rounds_mean": d[f"rounds_{k}"].mean() if f"rounds_{k}" in d else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "ablation_v6.csv", index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["sota", "loss", "cost", "abl", "all"])
    OUT.mkdir(parents=True, exist_ok=True)
    pt = ap.parse_args().part
    for nm, f in (("sota", sota), ("loss", loss), ("cost", cost), ("abl", abl)):
        if pt in (nm, "all"):
            try:
                f()
            except Exception as ex:                       # noqa: BLE001 — one failing part must not hide the others
                import traceback
                traceback.print_exc()
                print(f"PART {nm} FAILED: {ex}")
    print("K28 P3 P0 DONE")
