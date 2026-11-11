#!/usr/bin/env python
"""K39-5 (post-hoc, CPU; handoff/K39/PREREG.md): NAVSIM navtest, the planner-specific field vs the braking field.
--stage run (per log shard; env_navsim.sh), per token of the stored final run:
  footprints  PDM  = spec_pdm's crops from the probed PDM-Closed field (results/navsim_exp/field_test: one crop per
                     lateral column of sensitive cells + support crops, navsim_detect.spec_pdm_dets' construction)
              brk  = v3's round-1 braking field + support crops on the same cheap draft (nuplan_spec_real
                     speculative_dets' first round: insertion grid, one field crop, support crops), same v / camera /
                     ground plane; compared as union masks on the 1920 x 1080 CAM_F0 image (IoU, PDM-only area)
  trajectory  PDM-Closed (navsim_pdms.plan) on spec_pdm's and on spec's detections (dets_final, branch_annotations);
              differ = max position difference over the 8 poses > 0.5 m (navsim_field.MOVE_M); check: the plan on the
              cheap draft reproduces results/k20/base__ poses
--stage summary: shares, dPDMS = score_spec_pdm - score_spec (pdms_final) on differing / all tokens with paired log
  bootstrap (2,000, seed 0), K20 stored wall clocks (fast 1 worker, 8 workers, frozen) p50 / p95, top 5 by |dPDMS|
  (ties by token) -> k39_5_summary.csv, k39_5_top5.csv
usage: k39_5_navsim.py --stage run --shard i/n | --stage summary --out <dir>"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
X = B / "results" / "navsim_exp"
OUT = B / "results" / "k39" / "k5"


def mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def mask(crops, W, H):
    m = np.zeros((H, W), bool)
    for c in crops:
        x1, y1 = max(int(np.floor(c[0])), 0), max(int(np.floor(c[1])), 0)
        x2, y2 = min(int(np.ceil(c[2])), W), min(int(np.ceil(c[3])), H)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
    return m


def run(args):
    os.environ["NAVSIM_DETS"] = str(X / "dets_final")
    NP = mod("np_", B / "scripts/A_navsim/navsim_pdms.py")
    ND = mod("nd_", B / "scripts/A_navsim/navsim_detect.py")
    NF = mod("nf_", B / "scripts/A_navsim/navsim_field.py")
    NS = ND.NS
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    si, sn = (int(x) for x in args.shard.split("/"))
    gz = float((X / "dets_final" / "ground_z.txt").read_text())
    priors = pickle.load(open(X / "dets" / "size_priors.pkl", "rb"))
    W, H = NS.W, NS.H
    OUT.mkdir(parents=True, exist_ok=True)
    logs = sorted(Path(p).stem for p in glob.glob(str(X / "pdms_final" / "*.pkl")))
    split = os.environ.get("NAVSIM_SPLIT", "test")
    for li, log in enumerate(logs):
        fo = OUT / f"{log}.pkl"
        if li % sn != si or fo.exists():
            continue
        dets = pickle.load(open(X / "dets_final" / f"{log}.pkl", "rb"))
        fld = pickle.load(open(X / "field_test" / f"{log}.pkl", "rb"))
        base = {r["token"]: r["poses"] for r in pickle.load(open(B / "results/k20" / f"base__{log}.pkl", "rb"))}
        toks = [r["token"] for r in pickle.load(open(X / "pdms_final" / f"{log}.pkl", "rb"))]
        flt = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True, log_names=[log], tokens=toks)
        sl = SceneLoader(NP.DATA / "navsim_logs" / split, NP.DATA / "sensor_blobs" / split, flt, SensorConfig.build_no_sensors())
        proc = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
        rows = []
        for tok in toks:
            dd, fl = dets[tok], fld.get(tok, {})
            row = {"log": log, "token": tok}
            # ---- footprints
            cols = {}
            for b in fl.get("field_boxes", []):
                cols.setdefault(round(0.5 * (b[0] + b[2]) / 160), []).append(b)
            pdm = []
            for bs in cols.values():
                bs = np.asarray(bs)
                pdm.append(np.array([bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()]))
            for b in fl.get("support_boxes", []):
                bc = NS.VS.box_crop(np.asarray(b, float), W, H)
                if not NS.VS.covered(bc, pdm):
                    pdm.append(bc)
            cam_ns = ND.NavCam.__new__(ND.NavCam)
            K, D, R, t = dd["cam"]
            cam_ns.K, cam_ns.D, cam_ns.R, cam_ns.t = (np.asarray(a, float) for a in (K, D, R, t))
            import cv2
            border = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1], [0, H / 2], [W - 1, H / 2], [W / 2, 0], [W / 2, H - 1]], np.float64)
            und = cv2.undistortPoints(border.reshape(-1, 1, 2), cam_ns.K, cam_ns.D).reshape(-1, 2)
            cam_ns.bx, cam_ns.by = NS.CLAMP * float(np.abs(und[:, 0]).max()), NS.CLAMP * float(np.abs(und[:, 1]).max())
            v = float(dd["info"]["v"])
            keep = np.asarray(dd["cheap"]["conf"]) >= 0.25
            xy = np.asarray(dd["cheap"]["xyxy"], float)[keep]
            mono = NS.lift_boxes(cam_ns, xy, gz)
            brk = []
            sens = NS.VD.insertion_grid(mono, v, NS.R_GRID, NS.Y_GRID, NS.PROBE_W, NS.PP, NS.V_CLOSE)
            if sens.any():
                bs = [NS.cell_box(cam_ns, r, y, gz) for (a, r) in enumerate(NS.R_GRID) for (b_, y) in enumerate(NS.Y_GRID) if sens[a, b_]]
                bs = [b for b in bs if b is not None]
                if bs:
                    bs = np.asarray(bs)
                    brk.append(np.array([bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()]))
            for k_ in NS.VS.support(mono, v, NS.PP):
                bc = NS.VS.box_crop(xy[k_], W, H)
                if not NS.VS.covered(bc, brk):
                    brk.append(bc)
            mp, mb = mask(pdm, W, H), mask(brk, W, H)
            inter, uni = int((mp & mb).sum()), int((mp | mb).sum())
            row.update({"n_pdm_crops": len(pdm), "n_brk_crops": len(brk), "area_pdm": int(mp.sum()), "area_brk": int(mb.sum()),
                        "iou": inter / uni if uni else 1.0, "pdm_only_share": float((mp & ~mb).sum() / max(mp.sum(), 1)),
                        "brk_only_share": float((mb & ~mp).sum() / max(mb.sum(), 1)), "spec_rounds": dd["info"].get("rounds")})
            # ---- trajectories
            scene = sl.get_scene_from_token(tok)
            idx = scene.scene_metadata.num_history_frames - 1
            cam = NP.Cam(*dd["cam"])
            tr = {}
            for br in ("cheap", "spec", "spec_pdm"):
                ann = NP.branch_annotations(scene.frames[idx].annotations, dd[br], cam, gz, priors, br)
                tr[br] = NP.plan(proc, NF.with_ann(scene, idx, ann))
            dmax = float(np.max(np.hypot(tr["spec_pdm"].poses[:, 0] - tr["spec"].poses[:, 0], tr["spec_pdm"].poses[:, 1] - tr["spec"].poses[:, 1])))
            row.update({"traj_maxdiff_m": dmax, "traj_differs": dmax > NF.MOVE_M,
                        "base_check_maxabs": float(np.abs(tr["cheap"].poses - base[tok]).max()) if tok in base else np.nan})
            rows.append(row)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(rows)} tokens", flush=True)
    print("K39-5 RUN SHARD DONE", flush=True)


def boot(d, col, units):
    g = d.groupby("log")[col].agg(["sum", "count"]).reindex(units).fillna(0)
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(0).integers(0, len(units), (2000, len(units)))
    dist = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(s.sum() / max(n.sum(), 1)), *np.percentile(dist, [2.5, 97.5])


def summary(args):
    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    rr = []
    for f in sorted(glob.glob(str(OUT / "*.pkl"))):
        rr += pickle.load(open(f, "rb"))
    d = pd.DataFrame(rr)
    sc = pd.DataFrame([r for f in sorted(glob.glob(str(X / "pdms_final" / "*.pkl"))) for r in pickle.load(open(f, "rb"))])
    d = d.merge(sc[["log", "token", "score_spec", "score_spec_pdm"]], on=["log", "token"], validate="1:1")
    d["dpdms"] = d.score_spec_pdm - d.score_spec
    units = np.array(sorted(d.log.unique()))
    out = [{"quantity": "tokens", "value": len(d)}, {"quantity": "logs", "value": len(units)},
           {"quantity": "base poses reproduced (max |diff| over tokens)", "value": float(d.base_check_maxabs.max())},
           {"quantity": "braking footprint empty <=> v3 spec ran 0 rounds (mismatches)",
            "value": int(((d.area_brk == 0) != (d.spec_rounds == 0)).sum())}]
    for lab, m in (("footprint differs (IoU < 0.9, primary)", d.iou < 0.9), ("footprint differs at all (IoU < 1)", d.iou < 1),
                   ("footprint IoU < 0.5", d.iou < 0.5), ("PDM footprint empty", d.area_pdm == 0), ("braking footprint empty", d.area_brk == 0),
                   ("selected trajectory differs (> 0.5 m)", d.traj_differs.astype(bool))):
        d["_m"] = m.astype(float)
        pt, lo, hi = boot(d, "_m", units)
        out.append({"quantity": f"share: {lab}", "value": pt, "ci_lo": lo, "ci_hi": hi, "count": int(m.sum())})
    out.append({"quantity": "footprint IoU median", "value": float(d.iou.median())})
    out.append({"quantity": "PDM-only area share of the PDM footprint, mean (non-empty PDM)", "value": float(d.pdm_only_share[d.area_pdm > 0].mean())})
    for lab, sub in (("trajectory differs", d[d.traj_differs.astype(bool)]), ("all tokens", d), ("footprint differs (IoU < 0.9)", d[d.iou < 0.9])):
        pt, lo, hi = boot(sub, "dpdms", units)
        out.append({"quantity": f"dPDMS spec_pdm - braking field ({lab})", "value": pt, "ci_lo": lo, "ci_hi": hi, "count": len(sub)})
    k20 = B / "results" / "k20"
    for lab, rows_ in (("fast, 1 worker (K20, 1,000 tokens)", [r for f in glob.glob(str(k20 / "fast__*")) for r in pickle.load(open(f, "rb"))]),
                       ("fast, 8 workers (K20, 200 tokens)", pickle.load(open(k20 / "parallel_w8.pkl", "rb"))),
                       ("frozen implementation, 1 thread (K20, 1,000 tokens)", [r for f in glob.glob(str(k20 / "cost__*")) for r in pickle.load(open(f, "rb"))])):
        w = np.array([r["wall_token"] for r in rows_], float)
        out.append({"quantity": f"wall clock per frame s, {lab}", "value": float(np.median(w)), "p95": float(np.percentile(w, 95)),
                    "mean": float(w.mean()), "count": len(w)})
    pd.DataFrame(out).to_csv(od / "k39_5_summary.csv", index=False)
    top = d.assign(_a=d.dpdms.abs()).sort_values(["_a", "token"], ascending=[False, True]).head(5)
    top[["log", "token", "score_spec", "score_spec_pdm", "dpdms", "iou", "traj_maxdiff_m", "n_pdm_crops", "n_brk_crops"]].to_csv(od / "k39_5_top5.csv", index=False)
    d.drop(columns=["_m"]).to_csv(od / "k39_5_tokens.csv.gz", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(out).to_string(index=False))
    print(top.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "summary"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=str(OUT / "tables"))
    a = ap.parse_args()
    run(a) if a.stage == "run" else summary(a)
