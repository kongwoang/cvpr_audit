#!/usr/bin/env python
"""K1 (writer handoff): canonical main-table CSVs and side statistics from EXISTING per-frame results (no new runs).

Outputs (results/handoff_K1/): table_main_av2.csv, table_main_nusc.csv, router_rows_*.csv, router_label_stats.csv,
cost_stats.csv, sensitivity_nusc.csv, loso_nusc.csv, refiner_provenance.csv.
Conventions: G = Σ(J_cheap − J_m) / Σ J_cheap; CI = 95% paired bootstrap over scenes / logs (2,000 resamples, seed 0,
the same resamples for every row) of G(ours_final) − G(method); harmed share = frames with J_m > J_cheap among frames
with J_m ≠ J_cheap ("affected frames"); network px = detector input tensor pixels per frame / native frame pixels.
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
R = B / "results"
OUT = R / "handoff_K1"
OUT.mkdir(parents=True, exist_ok=True)
GENG = B / "third_party" / "bgt-ada"
NB = 2000


def boots_of(units, seed=0):
    uq = pd.unique(units)
    idx = [np.flatnonzero(units == u) for u in uq]
    rng = np.random.default_rng(seed)
    return [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(NB)]


class Table:
    def __init__(self, base, unit_col, ours_col):
        self.d = base.reset_index(drop=True)
        self.L = self.d.J_c.to_numpy(float)
        self.boots = boots_of(self.d[unit_col].to_numpy())
        self.ours = (self.L - self.d[ours_col].to_numpy(float))
        self.rows = []

    def add(self, method, J, px, source, note=""):
        J = np.asarray(J, float)
        V = self.L - J
        aff = np.abs(V) > 1e-9
        r = {"method": method, "G": round(V.sum() / self.L.sum(), 3),
             "network_px_mean": round(float(np.mean(px)), 4) if px is not None else np.nan,
             "network_px_p95": round(float(np.percentile(px, 95)), 4) if px is not None else np.nan,
             "harmed_share": round(float((V < -1e-9).sum() / max(aff.sum(), 1)), 3),
             "affected_frames": int(aff.sum())}
        dr = np.array([(self.ours[b].sum() - V[b].sum()) / self.L[b].sum() for b in self.boots])
        r["ci_lo"], r["ci_hi"] = (round(float(np.percentile(dr, 2.5)), 3), round(float(np.percentile(dr, 97.5)), 3)) \
            if not np.allclose(V, self.ours) else (np.nan, np.nan)
        r["source_file"], r["note"] = source, note
        self.rows.append(r)

    def frame(self):
        return pd.DataFrame(self.rows)


def merge_check(base, other, cols, on, name):
    m = base[on + ["J_c"]].merge(other[on + ["J_c"] + cols], on=on, how="left", suffixes=("", "_o"))
    assert len(m) == len(base), name
    ok = np.isclose(m.J_c, m.J_c_o)
    assert ok.mean() > 0.999, f"{name}: cheap cost differs on {1 - ok.mean():.4f} of frames"
    return m


def routers(frames_te, frames_val, frames_tr, prefix, ours_px, px_cheap, px_strong, tags):
    """Budget-Adaptive Routing rows at the pixel budget of ours. frames_*: DataFrames with J_c, J_u960 (strong)."""
    sys.path.insert(0, str(GENG))
    from src.offloader import Offloader
    rho_c = (ours_px - px_cheap) / px_strong
    rho_s = (ours_px - px_cheap) / (px_strong - px_cheap)
    out = {}
    for tag in tags:
        S = {}
        for n in ("skip_offbin", "cond_moric", "skip_V", "cond_V"):
            p = R / prefix / f"scores__{n}{tag}.pkl"
            if not p.exists() and n.startswith("cond"):          # XGBoost estimators are seed-independent (one fit)
                p = R / prefix / f"scores__{n}.pkl"
            if p.exists():
                S[n] = pickle.load(open(p, "rb"))
        for n, sc in S.items():
            if tag and n.startswith("cond"):
                continue                                          # reported once (seed 42 row), used by the arbiter below
            rho = rho_c if n.startswith("cond") else rho_s
            m = Offloader("offload_binary").decide_calibrated(sc["test"], rho, calibration_predictions=sc["train"]).mask
            px = (px_cheap + m * px_strong) if n.startswith("cond") else ((1 - m) * px_cheap + m * px_strong)
            out[(n, tag)] = (m.astype(bool), px)
        if {"skip_offbin", "cond_moric"} <= set(S):           # Geng's arbiter: the better of the two on val at this budget
            def gval(n):
                rho = rho_c if n.startswith("cond") else rho_s
                mv = Offloader("offload_binary").decide_calibrated(S[n]["val"], rho, calibration_predictions=S[n]["train"]).mask
                return float(((frames_val.J_c - frames_val.J_u960).to_numpy() * mv).sum() / frames_val.J_c.sum())
            pick = max(("skip_offbin", "cond_moric"), key=gval)
            if (pick, tag) not in out:                       # cond_moric chosen for a seed tag: same mask as the seed-42 row
                rho = rho_c
                m = Offloader("offload_binary").decide_calibrated(S[pick]["test"], rho, calibration_predictions=S[pick]["train"]).mask
                out[(pick, tag)] = (m.astype(bool), px_cheap + m * px_strong)
                out[(f"adaptive({pick})", tag)] = out.pop((pick, tag))
            else:
                out[(f"adaptive({pick})", tag)] = out[(pick, tag)]
    return out, rho_c, rho_s


def main():
    PXC_N, PXS_N = 320 * 192 / (1600 * 900), 576 * 960 / (1600 * 900)
    PXC_A, PXS_A = 320 * 256 / (1550 * 2048), 960 * 768 / (1550 * 2048)
    # ------------------------------------------------------------------ nuScenes new test
    nat = pd.read_pickle(R / "native" / "nusc.pkl").rename(columns={"unit": "seq"})
    conf = pd.read_pickle(R / "nusc_full" / "eval__new_test_v3_confirm.pkl")
    base = nat.merge(conf[["seq", "frame", "brake_J_f", "n_crops", "rounds"]], on=["seq", "frame"])
    assert len(base) == len(nat) == 9181
    base["px_ours"] = PXC_N + base.px * 0.36
    T = Table(base, "seq", "J_spec_ref_resnet18")
    src_conf = "nusc_full/eval__new_test_v3_confirm.pkl"
    pxs = lambda v: np.full(len(base), v)                                                # noqa: E731
    T.add("u640 (DEEP full mode)", base.brake_J_f, pxs(384 * 640 / 1.44e6), src_conf, "brake_J_f = DEEP's full 640 mode")
    T.add("u960", base.J_u960, pxs(PXS_N), "native/nusc.pkl")
    T.add("u1280", base.J_u1280, base.px_u1280, "native/nusc.pkl")
    T.add("native u1600", base.J_u1600, base.px_u1600, "native/nusc.pkl")
    ada = merge_check(base, pd.read_pickle(R / "adascale" / "eval_newtest.pkl"), ["J_ada", "net_px"], ["seq", "frame"], "adascale")
    T.add("AdaScale (re-impl.)", ada.J_ada, ada.net_px, "adascale/eval_newtest.pkl", "classifier input not counted (not a detector pass)")
    fr = {sp: pd.read_pickle(R / "router" / f"frames__{sp}.pkl") for sp in ("test", "val", "train")}
    assert (fr["test"].seq.to_numpy() == base.seq.to_numpy()).all() and (fr["test"].frame.to_numpy() == base.frame.to_numpy()).all()
    ours_px_n = float(base.px_ours.mean())
    R_n, rcn, rsn = routers(fr["test"], fr["val"], fr["train"], "router", ours_px_n, PXC_N, PXS_N, ["", "_s1", "_s2"])
    Vt = (fr["test"].J_c - fr["test"].J_u960).to_numpy()
    for (n, tag), (m, px) in sorted(R_n.items()):
        seed = {"": "42", "_s1": "1", "_s2": "2"}[tag]
        off = n.startswith(("skip_offbin", "cond_moric", "adaptive"))
        T.add(f"Budget-Adaptive Routing {'official' if off else 'decision-value target'}: {n}, seed {seed}",
              base.J_c - Vt * m, px, f"router/scores__{n.split('(')[-1].rstrip(')')}{tag}.pkl",
              f"rho_cond {rcn:.4f} / rho_skip {rsn:.4f}; routed frames use u960 (960 history)")
    n_or = int(round(rcn * len(base)))
    orc = np.zeros(len(base), bool); orc[np.argsort(-Vt, kind="stable")[:n_or]] = True
    T.add("oracle frame router (true V = J_cheap − J_u960, top-k)", base.J_c - Vt * orc, PXC_N + orc * PXS_N, "router/frames__test.pkl",
          f"cheap on every frame + u960 on the top {n_or} frames by V (k = rho_cond × n)")
    fov = merge_check(base, pd.read_pickle(R / "fovea" / "eval_newtest.pkl"), ["J_fovea_kdei960", "J_fovea_kdes960"], ["seq", "frame"], "fovea")
    T.add("FOVEA KDE_I (576x960 canvas)", fov.J_fovea_kdei960, pxs(PXC_N + PXS_N), "fovea/eval_newtest.pkl", "draft pass gives the saliency")
    T.add("FOVEA KDE_S (576x960 canvas)", fov.J_fovea_kdes960, pxs(PXS_N), "fovea/eval_newtest.pkl", "dataset saliency, no draft")
    from sahi.slicing import get_slice_bboxes
    ns = len(get_slice_bboxes(900, 1600, 640, 640, overlap_height_ratio=0.2, overlap_width_ratio=0.2))
    sah = merge_check(base, pd.read_pickle(R / "sahi" / "eval_newtest.pkl"), ["J_sahi_s06", "J_sahi_s10"], ["seq", "frame"], "sahi")
    for k, sz in (("s06", 384), ("s10", 640)):
        T.add(f"SAHI ({ns} slices of 640x640 at {sz} px + full pass)", sah[f"J_sahi_{k}"],
              pxs((ns * sz * sz + sz * math.ceil(sz * 900 / 1600 / 32) * 32) / 1.44e6), "sahi/eval_newtest.pkl")
    lzu = merge_check(base, pd.read_pickle(R / "lzu" / "eval_newtest_frozen.pkl"), ["J_lzu_frozen_960"], ["seq", "frame"], "lzu")
    T.add("LZU frozen (576x960)", lzu.J_lzu_frozen_960, pxs(PXS_N), "lzu/eval_newtest_frozen.pkl", "dataset saliency, no draft")
    ufp = merge_check(base, pd.read_pickle(R / "ufp" / "eval_newtest.pkl"), [f"J_ufp_{s}" for s in (640, 960, 1333)] +
                      [f"tpx_ufp_{s}" for s in (640, 960, 1333)], ["seq", "frame"], "ufp")
    for s in (640, 960, 1333):
        T.add(f"UFP frozen (mosaic {s})", ufp[f"J_ufp_{s}"], PXC_N + ufp[f"tpx_ufp_{s}"], "ufp/eval_newtest.pkl",
              "network px = coarse (draft) pass + mosaic tensor")
    st = merge_check(base, pd.read_pickle(R / "baselines_eq__new_test_strict.pkl"),
                     [f"{p}_{a}" for a in ("random", "uncertainty", "temporal_crit", "traj_support") for p in ("J", "px")],
                     ["seq", "frame"], "strict")
    for a in ("uncertainty", "temporal_crit", "random", "traj_support"):
        T.add(f"{a} crops (strict budget = v3's per-frame crop area)", st[f"J_{a}"], PXC_N + st[f"px_{a}"] * 0.36,
              "baselines_eq__new_test_strict.pkl", "px = draft + crop area × 0.6²; guard re-detections not counted for heuristics"
              + ("; traj_support support crops are not capped by the budget" if a == "traj_support" else ""))
    T.add("ours without refiner (v3)", base.J_spec, base.px_ours, src_conf)
    T.add("ours final (v3 + refiner v1)", base.J_spec_ref_resnet18, base.px_ours, src_conf, "px = draft + crop area × 0.6² (guard crops included)")
    v4 = merge_check(base, pd.read_pickle(R / "nusc_full" / "eval__new_test_curve_vc0_r3_s0.6.pkl"),
                     ["J_spec_ref_resnet18", "J_spec_ref_resnet18_v2", "px"], ["seq", "frame"], "v4lite")
    T.add("AV2 configuration v4-lite + refiner v1", v4.J_spec_ref_resnet18, PXC_N + v4.px * 0.36, "nusc_full/eval__new_test_curve_vc0_r3_s0.6.pkl")
    T.add("AV2 configuration v4-lite + refiner v2", v4.J_spec_ref_resnet18_v2, PXC_N + v4.px * 0.36, "nusc_full/eval__new_test_curve_vc0_r3_s0.6.pkl",
          "refiner v2 trained on nuScenes train+val scenes (not test)")
    tn = T.frame(); tn.insert(0, "dataset", "nuScenes new test (228 scenes, 9,181 frames)")
    tn.to_csv(OUT / "table_main_nusc.csv", index=False)

    # ------------------------------------------------------------------ AV2 val
    fin = pd.read_pickle(R / "av2" / "eval__val_v4_final.pkl")
    natA = pd.read_pickle(R / "native" / "av2.pkl").rename(columns={"unit": "log"})
    baseA = fin.merge(natA[["log", "frame", "J_u1280", "px_u1280", "J_u1536", "px_u1536", "J_u2048", "px_u2048"]], on=["log", "frame"])
    assert len(baseA) == 11818
    baseA["px_ours"] = PXC_A + baseA.netpx_v4lite
    TA = Table(baseA, "log", "J_spec_ref_v4lite")
    srcA = "av2/eval__val_v4_final.pkl"
    pxa = lambda v: np.full(len(baseA), v)                                               # noqa: E731
    TA.add("u640", baseA.J_u640, pxa(640 * 512 / (1550 * 2048)), srcA)
    TA.add("u960", baseA.J_u960, pxa(PXS_A), srcA)
    TA.add("u1280", baseA.J_u1280, baseA.px_u1280, "native/av2.pkl")
    TA.add("u1536", baseA.J_u1536, baseA.px_u1536, "native/av2.pkl")
    TA.add("native u2048", baseA.J_u2048, baseA.px_u2048, "native/av2.pkl")
    adaA = merge_check(baseA, pd.read_pickle(R / "av2_sota" / "ada_eval_test.pkl"), ["J_ada", "net_px"], ["log", "frame"], "adaA")
    TA.add("AdaScale (re-impl., fit on AV2 train)", adaA.J_ada, adaA.net_px, "av2_sota/ada_eval_test.pkl")
    frA = {sp: pd.read_pickle(R / "av2_sota" / f"frames__{sp}.pkl") for sp in ("test", "val", "train")}
    mA = baseA[["log", "frame"]].merge(frA["test"][["log", "frame", "J_c", "J_u960"]], on=["log", "frame"], how="left", suffixes=("", "_r"))
    assert np.allclose(mA.J_c, baseA.J_c)
    order_ok = (frA["test"].log.to_numpy() == baseA.log.to_numpy()).all() and (frA["test"].frame.to_numpy() == baseA.frame.to_numpy()).all()
    assert order_ok
    ours_px_a = float(baseA.px_ours.mean())
    R_a, rca, rsa = routers(frA["test"], frA["val"], frA["train"], "av2_sota", ours_px_a, PXC_A, PXS_A, [""])
    VtA = (frA["test"].J_c - frA["test"].J_u960).to_numpy()
    for (n, tag), (m, px) in sorted(R_a.items()):
        off = n.startswith(("skip_offbin", "cond_moric", "adaptive"))
        TA.add(f"Budget-Adaptive Routing {'official' if off else 'decision-value target'}: {n}, seed 42", baseA.J_c - VtA * m, px,
               f"av2_sota/scores__{n.split('(')[-1].rstrip(')')}.pkl", f"rho_cond {rca:.4f} / rho_skip {rsa:.4f}; one seed on AV2")
    n_orA = int(round(rca * len(baseA)))
    orcA = np.zeros(len(baseA), bool); orcA[np.argsort(-VtA, kind="stable")[:n_orA]] = True
    TA.add("oracle frame router (true V = J_cheap − J_u960, top-k)", baseA.J_c - VtA * orcA, PXC_A + orcA * PXS_A, "av2_sota/frames__test.pkl",
           f"cheap on every frame + u960 on the top {n_orA} frames by V")
    bl = merge_check(baseA, pd.read_pickle(R / "av2" / "baselines__val.pkl"),
                     ["J_fovea_kdei960", "J_sahi_s047", "J_sahi_s10", "J_uncertainty", "px_uncertainty", "J_temporal_crit",
                      "px_temporal_crit", "J_random", "px_random"], ["log", "frame"], "av2 baselines")
    TA.add("FOVEA KDE_I (960x768 canvas)", bl.J_fovea_kdei960, pxa(PXC_A + PXS_A), "av2/baselines__val.pkl", "KDE_S not run on AV2")
    nsA = len(get_slice_bboxes(2048, 1550, 640, 640, overlap_height_ratio=0.2, overlap_width_ratio=0.2))
    for k, sz in (("s047", 320), ("s10", 640)):
        TA.add(f"SAHI ({nsA} slices of 640x640 at {sz} px + full pass)", bl[f"J_sahi_{k}"],
               pxa((nsA * sz * sz + sz * math.ceil(sz * 1550 / 2048 / 32) * 32) / (1550 * 2048)), "av2/baselines__val.pkl")
    for a in ("uncertainty", "temporal_crit", "random"):
        TA.add(f"{a} crops (strict budget = frozen v3's per-frame crop area)", bl[f"J_{a}"], PXC_A + bl[f"px_{a}"] * (960 / 2048) ** 2,
               "av2/baselines__val.pkl", "budget from frozen v3 (confirmatory 2), not v4-lite; guard re-detections not counted; traj_support, LZU, UFP not run on AV2")
    TA.add("ours without refiner (v4-lite)", baseA.J_spec_v4lite, baseA.px_ours, srcA)
    TA.add("ours final (v4-lite + refiner v2)", baseA.J_spec_ref_v4lite, baseA.px_ours, srcA, "px = draft + crop area × 0.469² (guard crops included)")
    TA.add("nuScenes configuration v3 + refiner v1", baseA.J_spec_ref_v3, PXC_A + baseA.netpx_v3, srcA)
    ta = TA.frame(); ta.insert(0, "dataset", "AV2 val (150 logs, 11,818 frames)")
    ta.to_csv(OUT / "table_main_av2.csv", index=False)

    # ------------------------------------------------------------------ D: router label statistics
    rows = []
    for ds, f in (("nuScenes new train", fr["train"]), ("AV2 train (router train logs)", frA["train"])):
        V = f.J_c - f.J_u960
        rows.append({"dataset": ds, "frames": len(f), "share_ORIC_gt0": round(float((f.oric > 0).mean()), 4),
                     "share_V_gt0": round(float((V > 0).mean()), 4), "share_V_lt0": round(float((V < 0).mean()), 4),
                     "spearman_ORIC_V": round(float(f.oric.corr(V, method="spearman")), 4),
                     "source_file": "router/frames__train.pkl" if ds.startswith("nu") else "av2_sota/frames__train.pkl"})
    pd.DataFrame(rows).to_csv(OUT / "router_label_stats.csv", index=False)

    # ------------------------------------------------------------------ G: cost statistics of ours
    rows = []
    for ds, n_cr, rnd, px in (("nuScenes new test (v3)", base.n_crops, base.rounds, base.px_ours),
                               ("AV2 val (v4-lite)", baseA.n_crops_v4lite, None, baseA.px_ours)):
        r = {"dataset": ds, "crops_mean": round(float(n_cr.mean()), 3), "crops_p95": float(np.percentile(n_cr, 95)),
             "share_zero_crops": round(float((n_cr == 0).mean()), 4), "network_px_mean": round(float(px.mean()), 4),
             "network_px_p95": round(float(np.percentile(px, 95)), 4)}
        if rnd is not None:
            for k in range(4):
                r[f"share_rounds_{k}"] = round(float((rnd == k).mean()), 4)
            r["rounds_mean"] = round(float(rnd.mean()), 3)
        r["guard_redetections_per_frame"] = "not stored in the run (crops count includes guard-grown crops)"
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "cost_stats.csv", index=False)

    # ------------------------------------------------------------------ I: nuScenes sensitivity (scene-1051, LOSO)
    rows = []
    for lab, mask in (("all 228 scenes", np.ones(len(base), bool)), ("without scene-1051", (base.seq != "scene-1051").to_numpy())):
        b = base[mask]
        L = b.J_c.sum()
        r = {"subset": lab, "frames": int(mask.sum())}
        for nm, col in (("u960", "J_u960"), ("u1280", "J_u1280"), ("u1600", "J_u1600"), ("ours_no_ref", "J_spec"), ("ours_final", "J_spec_ref_resnet18")):
            r[f"G_{nm}"] = round(float((b.J_c - b[col]).sum() / L), 3)
        bb = boots_of(b.seq.to_numpy())
        Lb = b.J_c.to_numpy(); Vo = (b.J_c - b.J_spec_ref_resnet18).to_numpy()
        for nm, col in (("u960", "J_u960"), ("u1280", "J_u1280")):
            Vm = (b.J_c - b[col]).to_numpy()
            dr = [(Vo[t].sum() - Vm[t].sum()) / Lb[t].sum() for t in bb]
            r[f"ci_ours_minus_{nm}"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "sensitivity_nusc.csv", index=False)
    lo = []
    for s in base.seq.unique():
        b = base[base.seq != s]
        L = b.J_c.sum()
        lo.append({"left_out": s, "ours_minus_u960": (b.J_u960 - b.J_spec_ref_resnet18).sum() / L,
                   "ours_minus_u1280": (b.J_u1280 - b.J_spec_ref_resnet18).sum() / L})
    lo = pd.DataFrame(lo)
    lo.to_csv(OUT / "loso_nusc.csv", index=False)
    print("LOSO ours−u960 range", lo.ours_minus_u960.min().round(3), lo.ours_minus_u960.max().round(3),
          "ours−u1280 range", lo.ours_minus_u1280.min().round(3), lo.ours_minus_u1280.max().round(3))

    # ------------------------------------------------------------------ C: refiner provenance
    sp = pd.read_csv(R / "nusc_full" / "split.csv")
    test_new = set(sp[(sp["set"] == "new") & (sp.split == "test")].seq)
    val_new = set(sp[(sp["set"] == "new") & (sp.split == "val")].seq)
    rows = []
    for ds in ("KITTI", "nuScenes", "nuScenesAll"):
        z = np.load(B / "data" / f"refiner_data__{ds}.npz", allow_pickle=True)
        keys = list(z.keys())
        seqs = z["seq"] if "seq" in keys else None
        spl = z["split"]
        for s_ in np.unique(spl):
            m = spl == s_
            us = sorted(set(map(str, seqs[m]))) if seqs is not None else []
            rows.append({"file": f"data/refiner_data__{ds}.npz", "used_by": {"KITTI": "v1 and v2", "nuScenes": "v1", "nuScenesAll": "v2"}[ds],
                         "split_in_file": str(s_), "role_in_training": {"train": "training", "val": "epoch selection", "test": "held-out report only"}.get(str(s_), str(s_)),
                         "boxes": int(m.sum()), "units": len(us) if us else "not stored", "units_list": " ".join(us)[:20000],
                         "overlap_nuscenes_new_test": len(set(us) & test_new) if us else "n/a",
                         "overlap_nuscenes_new_val": len(set(us) & val_new) if us else "n/a"})
    pd.DataFrame(rows).to_csv(OUT / "refiner_provenance.csv", index=False)
    print("K1 TABLES DONE")


if __name__ == "__main__":
    main()
