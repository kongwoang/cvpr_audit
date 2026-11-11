#!/usr/bin/env python
"""K41-3 D05 analysis: complete 81-point field grid (PREREG K41-3 grid sweep; RUNPLAN D05) on the three braking splits.
Grid index order = itertools.product(range factor (.5,1,2), lateral factor (.5,1,2), probe (ped,car,truck),
v_close (0,5,10)), exactly as k40r_detector.py --part grid. Every point is reported, including unfavourable ones; no
point is selected or replaces the frozen v6-S field. (Measured afterwards: the car / v_close 0 points at lateral factor
0.5 and 1 reproduce v6-S exactly, Delta = 0 with CI [0, 0], an internal parity check.)
- Parity: every unit's .done.json has no failures and 10 comparisons per row, and the v6-S control J_v6S|v3a of every
  grid job equals the released records (max |dJ| = 0).
- G = 1 - sum J_row / sum J_cheap on gt frames; Delta = G(v6-S) - G(point) under the same refiner; K40 release paired
  unit bootstrap (2,000 draws, seed 0); p = 2 min(P(D<=0), P(D>=0)); Holm over the 81 points per dataset x refiner.
- Agreement: discrete action (refiner v3a) equal to act_gt. Repair (K39-6): no-refiner action equal to pi(F) = u960+C1
  on frames where the cheap draft differs from pi(F).
- Decisive coverage: frames with a decisive object (dec >= 0); share whose decisive box centre lies inside the union of
  the executed crops of the point (observed_crops) / of v6-S's crops (crops_v6S).
- px = mean frame pixels / native (cheap included); field time = mean CPU field-construction time per frame (us);
  GPU s = wall seconds of the (point, dataset) job from the D05 claims (includes the v6-S control pass and GPU waits).
usage: k41_d05_analyze.py <out_dir> [--partial]   (on VUW, after . $B/env.sh; --partial skips incomplete points)"""
import glob
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path("/vol/dl-nguyenb5-solar/users/hoangpc/CVPR")
REL = B / "tmp/desp_k40"
sys.path.insert(0, str(REL))
from analysis.common import Bootstrap, records  # noqa: E402  (read-only use of the K40 release helpers)

NAME = {"nusc": "nuscenes", "av2": "av2", "ahd": "argoverse_hd"}
RECS = REL / "artifacts/records"
GRID = list(itertools.product((.5, 1., 2.), (.5, 1., 2.), ("ped", "car", "truck"), (0., 5., 10.)))
CLAIMS = B / "logs/k41/D05/claims"


def G(J, L):
    return float(1 - J.sum() / L.sum())


def draws(bt, L, Ja, Jb):
    l, a, b = bt.sums(L), bt.sums(Ja), bt.sums(Jb)
    return (b[bt.idx].sum(1) - a[bt.idx].sum(1)) / l[bt.idx].sum(1)


def holm_adj(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj


def inside(box, crops):
    if box is None or not len(crops):
        return False
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(c[0] <= cx <= c[2] and c[1] <= cy <= c[3] for c in crops)


def load(g, ds):
    """gt-frame rows of grid point g reduced to the analysed columns, plus the per-unit done records"""
    n = f"grid{g:02d}@none"; out, meta = [], []
    for f in sorted(glob.glob(str(B / f"results/k41/grid_{g:02d}" / ds / "*.pkl"))):
        meta.append(json.loads(Path(f).with_suffix(".done.json").read_text()))
        for r in pickle.load(open(f, "rb")):
            if not r["gt"]:
                continue
            dec = int(r["dec"]); box = np.asarray(r["gt_boxes"])[dec] if dec >= 0 else None
            out.append({"unit": str(r["unit"]), "frame": r["frame"], "J_cheap": r["J_cheap"], "a_cheap": r["a_cheap"],
                        "act_gt": r["act_gt"], "J_v6S|v3a": r["J_v6S|v3a"], "J_v6S|none": r["J_v6S|none"],
                        "a_v6S|v3a": r["a_v6S|v3a"], "a_v6S|none": r["a_v6S|none"], "px_v6S": r["px_v6S"],
                        "J|v3a": r[f"J_{n}|v3a"], "J|none": r[f"J_{n}|none"], "a|v3a": r[f"a_{n}|v3a"], "a|none": r[f"a_{n}|none"],
                        "px": r[f"px_{n}"], "rounds": r[f"rounds_{n}"], "field_us": r[f"field_us_{n}"], "has_dec": dec >= 0,
                        "cov": inside(box, r[f"observed_crops_{n}"]), "cov_v6S": inside(box, r["crops_v6S"])})
    d = pd.DataFrame(out)
    return (d.sort_values(["unit", "frame"]).reset_index(drop=True) if len(d) else d), meta


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    partial = "--partial" in sys.argv
    checks, out = {}, []
    for ds, nm in NAME.items():
        R = records(nm, RECS)
        nunits = len(glob.glob(str(B / "results/k30/match" / ds / "*.pkl")))
        bt = F = None; block = {"v3a": [], "none": []}
        for g, (rm, lm, probe, vc) in enumerate(GRID):
            st = CLAIMS / f"{g}_{ds}" / "status"
            if not (st.exists() and st.read_text().strip() == "done"):
                assert partial, f"grid {g} {ds} not done"
                continue
            d, meta = load(g, ds)
            assert len(meta) == nunits and len(d) == len(R), (g, ds, len(meta), nunits, len(d), len(R))
            m = d[["unit", "frame", "J_v6S|v3a"]].merge(R[["unit", "frame", "J_v6S|v3a", "a_wf960+C1|none"]], on=["unit", "frame"],
                                                         how="left", suffixes=("", "@rel"), validate="1:1")
            c = {"frames": len(d), "units": len(meta), "unit_failures": int(sum(len(x["failures"]) for x in meta)),
                 "comparisons": int(sum(x["comparisons"] for x in meta)), "rows": int(sum(x["rows"] for x in meta)),
                 "max_abs_dJ_v6S_vs_release": float(np.nanmax(np.abs(m["J_v6S|v3a"] - m["J_v6S|v3a@rel"])))}
            assert c["unit_failures"] == 0 and c["comparisons"] == 10 * c["rows"] and c["max_abs_dJ_v6S_vs_release"] < 1e-9, (g, ds, c)
            checks[f"{g:02d}/{ds}"] = c
            if bt is None:   # same gt frames and order for every point (parity above), so one bootstrap per dataset
                bt, units0 = Bootstrap(d.unit), d.unit.to_numpy()
                F = m["a_wf960+C1|none"].to_numpy(int); rep = d.a_cheap.to_numpy(int) != F
                L = d.J_cheap.to_numpy(float)
            assert (d.unit.to_numpy() == units0).all()
            e6 = d["a_v6S|none"].to_numpy(int) == F; eq = d["a|none"].to_numpy(int) == F
            sd = bt.share_delta(e6, eq, rep)
            hd = d.has_dec.to_numpy(bool)
            sec = CLAIMS / f"{g}_{ds}" / "seconds"
            for ref in ("v3a", "none"):
                Jd, Jr = d[f"J_v6S|{ref}"].to_numpy(float), d[f"J|{ref}"].to_numpy(float)
                x = draws(bt, L, Jd, Jr)
                block[ref].append({"dataset": ds, "grid_index": g, "range_factor": rm, "lateral_factor": lm, "probe": probe,
                                   "v_close": vc, "refiner": ref, "units": len(bt.units), "frames": len(d),
                                   "G": G(Jr, L), "G_v6S": G(Jd, L), "delta": G(Jd, L) - G(Jr, L),
                                   "ci_lo": float(np.percentile(x, 2.5)), "ci_hi": float(np.percentile(x, 97.5)),
                                   "p_boot": float(2 * min((x <= 0).mean(), (x >= 0).mean())),
                                   "agree": float((d[f"a|{ref}"].to_numpy(int) == d.act_gt.to_numpy(int)).mean()),
                                   "agree_v6S": float((d[f"a_v6S|{ref}"].to_numpy(int) == d.act_gt.to_numpy(int)).mean()),
                                   "repair": float(eq[rep].mean()), "repair_v6S": float(e6[rep].mean()), "repair_delta": sd[0],
                                   "repair_ci_lo": sd[1], "repair_ci_hi": sd[2], "repair_frames": int(rep.sum()),
                                   "dec_frames": int(hd.sum()), "dec_coverage": float(d["cov"][hd].mean()),
                                   "dec_coverage_v6S": float(d.cov_v6S[hd].mean()),
                                   "px_mean": float(d.px.mean()), "px_v6S": float(d.px_v6S.mean()),
                                   "px_ratio": float(d.px.mean() / d.px_v6S.mean()), "rounds_mean": float(d.rounds.mean()),
                                   "field_us_mean": float(d.field_us.mean()),
                                   "gpu_job_s": float(sec.read_text()) if sec.exists() else np.nan})
            print(f"  {ds} grid {g:02d} {rm} {lm} {probe} {vc}: G {block['v3a'][-1]['G']:.3f} vs {block['v3a'][-1]['G_v6S']:.3f}", flush=True)
        for ref, b in block.items():
            for r_, q in zip(b, holm_adj([r_["p_boot"] for r_ in b])):
                r_["p_holm"] = float(q)
            out += b
    t = pd.DataFrame(out)
    t.to_csv(od / "k41_3_grid.csv", index=False)
    (od / "k41_3_grid_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    s = []
    for (ds, ref), g in t.groupby(["dataset", "refiner"]):
        s.append({"dataset": ds, "refiner": ref, "points": len(g), "G_v6S": g.G_v6S.iloc[0], "G_min": g.G.min(), "G_median": g.G.median(),
                  "G_max": g.G.max(), "best_point": int(g.grid_index[g.G.idxmax()]), "points_above_v6S": int((g.delta < 0).sum()),
                  "holm_point_better": int(((g.p_holm < .05) & (g.delta < 0)).sum()),
                  "holm_v6S_better": int(((g.p_holm < .05) & (g.delta > 0)).sum()),
                  "px_ratio_min": g.px_ratio.min(), "px_ratio_max": g.px_ratio.max()})
    pd.DataFrame(s).to_csv(od / "k41_3_grid_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(s).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
