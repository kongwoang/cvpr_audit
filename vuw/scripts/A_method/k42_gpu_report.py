#!/usr/bin/env python
"""K42 GPU part report (post-hoc; handoff/K42/PREREG.md §A2), from results/k42/vuw[_<tag>]/<ds> (GT frames).
Parity: a_v6S|none / v3a, a_wf960+C1|none / v3a and a_s*|none vs results/k39/vuw; a_ch_*|none vs K28 p3 instr a_ch_*.
K42-3b: k39_4_report.single's table (alone shares of D0; cumulative chain), with refiner none and v3a on BOTH sides.
K42-4(b): stopping-test log of the frozen v6-S (95% single-crop test vs exact union containment) and the v6U run.
Bootstrap: sorted units, default_rng(0).integers((2000, n)), ratio of per-unit sums.
usage: k42_gpu_report.py <out_dir> [<tag>] [--partial]"""
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B_REAL", os.environ["B"]))
KEY = ["unit", "frame"]
SINGLE = {"sa": "a exact locality", "sb": "b F's geometry (history)", "sc1": "c1 closing probe 10 m/s",
          "sc2": "c2 closing probe 20 m/s", "sd": "d extended grid", "se": "e three probe templates", "sf": "f round cap 10"}
CHAIN = {"ch_a": "a", "ch_b": "a+b", "ch_c1": "a..c1", "ch_c2": "a..c2", "ch_d": "a..d", "ch_e": "a..e", "ch_f": "a..f"}
NUN = {"nusc": 228, "av2": 150, "ahd": 24}


def recs(d, cols=None):
    rr = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        for r in pickle.load(open(f, "rb")):
            rr.append({c: r.get(c) for c in cols} if cols else r)
    x = pd.DataFrame(rr)
    if len(x):
        x["unit"] = x.unit.astype(str)
    return x


def boot(units):
    u = np.array(sorted(pd.unique(units)))
    ui = np.searchsorted(u, np.asarray(units))
    idx = np.random.default_rng(0).integers(0, len(u), (2000, len(u)))
    C = np.stack([np.bincount(x, minlength=len(u)) for x in idx]).astype(float)
    return ui, len(u), C


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    tag = next((a for a in sys.argv[2:] if not a.startswith("--")), "")
    partial = "--partial" in sys.argv
    par, single, stop, stop_units = {}, [], [], []
    for ds in ("nusc", "av2", "ahd"):
        d = recs(B / "results/k42" / ("vuw" + (f"_{tag}" if tag else "")) / ds)
        if not len(d):
            continue
        nunits = d.unit.nunique()
        if nunits != NUN[ds] and not partial:
            print(ds, "incomplete", nunits); continue
        d = d[d["gt"].astype(bool)].sort_values(KEY).reset_index(drop=True)
        # ---- parity
        k39 = recs(B / "results/k39/vuw" / ds)
        k39 = k39[k39["gt"].astype(bool)]
        cols39 = ["a_v6S|none", "a_v6S|v3a", "a_wf960+C1|none", "a_wf960+C1|v3a"] + [f"a_{k}|none" for k in SINGLE]
        m = d[KEY + cols39].merge(k39[KEY + cols39], on=KEY, how="left", suffixes=("", "_k39"), validate="1:1")
        p = {"frames": len(d), "units": nunits}
        for c in cols39:
            p[f"{c} vs K39 (differing frames)"] = int((m[c].astype(float) != m[c + "_k39"].astype(float)).sum())
        k28 = recs(B / "results/k28/p3" / f"{ds}_test_instr", KEY + ["gt"] + [f"a_{k}" for k in CHAIN])
        k28 = k28[k28["gt"].astype(bool)]
        m = d[KEY + [f"a_{k}|none" for k in CHAIN]].merge(k28, on=KEY, how="left", validate="1:1")
        for k in CHAIN:
            p[f"a_{k}|none vs K28 instr a_{k} (differing frames)"] = int((m[f"a_{k}|none"].astype(float) != m[f"a_{k}"].astype(float)).sum())
        par[ds] = p
        # ---- K42-3b: single interventions and chain, none and v3a on both sides
        for ref in ("none", "v3a"):
            F, O = d[f"a_wf960+C1|{ref}"].to_numpy(int), d[f"a_v6S|{ref}"].to_numpy(int)
            D0 = O != F
            anyres = np.zeros(len(d), bool)
            for k, lab in SINGLE.items():
                X = d[f"a_{k}|{ref}"].to_numpy(int)
                res = D0 & (X == F); anyres |= res
                single.append({"dataset": ds, "refiner_both_sides": ref, "intervention": lab, "mode": "alone", "frames": len(d),
                               "disagree_D0": int(D0.sum()), "agreement_before": float((~D0).mean()), "agreement_after": float((X == F).mean()),
                               "resolved": int(res.sum()), "share_of_D0": float(res.sum() / max(D0.sum(), 1)),
                               "new_disagreements": int((~D0 & (X != F)).sum())})
            single.append({"dataset": ds, "refiner_both_sides": ref, "intervention": "resolved by no single intervention", "mode": "alone",
                           "frames": len(d), "disagree_D0": int(D0.sum()), "resolved": int((D0 & ~anyres).sum()),
                           "share_of_D0": float((D0 & ~anyres).sum() / max(D0.sum(), 1))})
            prev = D0.copy()
            for k, lab in CHAIN.items():
                X = d[f"a_{k}|{ref}"].to_numpy(int)
                still = X != F
                single.append({"dataset": ds, "refiner_both_sides": ref, "intervention": lab, "mode": "cumulative chain (recomputed)",
                               "frames": len(d), "disagree_D0": int(D0.sum()), "agreement_before": float((~D0).mean()),
                               "agreement_after": float((X == F).mean()), "resolved": int((D0 & ~still).sum()),
                               "share_of_D0": float((D0 & ~still).sum() / max(D0.sum(), 1)), "step_share": float((prev & ~still).sum() / max(D0.sum(), 1))})
                prev = D0 & still
        # ---- K42-4(b)
        ui, nu, C = boot(d.unit)
        L = d.J_cheap.to_numpy(float)
        Lu = np.bincount(ui, L, minlength=nu)
        cnt = lambda v: np.bincount(ui, np.asarray(v, float), minlength=nu)  # noqa: E731
        n_u = cnt(np.ones(len(d)))
        fr_skip = d.cov_skip_union_req_v6S.to_numpy(int) > 0
        fr_rev = d.cov_req_union_skip_v6S.to_numpy(int) > 0
        r = {"dataset": ds, "units": nu, "frames": len(d)}

        def share(name, v):
            num = cnt(v); dist = (C @ num) / (C @ n_u)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            r.update({name: float(num.sum() / n_u.sum()), f"{name}_lo": lo, f"{name}_hi": hi})
        share("frames_95test_skips_a_crop_union_would_require", fr_skip)
        share("frames_union_would_skip_a_crop_95test_runs", fr_rev)
        r["stopping_test_calls"] = int(d.cov_calls_v6S.sum()); r["candidates_skipped_by_95test"] = int(d.cov_skip_v6S.sum())
        r["skipped_but_not_contained_in_union"] = int(d.cov_skip_union_req_v6S.sum())
        r["run_but_contained_in_union"] = int(d.cov_req_union_skip_v6S.sum())
        fr_ = [x for l in d.cov_skip_union_fracs_v6S for x in (l or [])]
        r["union_share_of_those_skipped_min"] = float(min(fr_)) if fr_ else np.nan
        r["union_share_of_those_skipped_median"] = float(np.median(fr_)) if fr_ else np.nan
        for ref in ("none", "v3a"):
            a6, aU = d[f"a_v6S|{ref}"].to_numpy(int), d[f"a_v6U|{ref}"].to_numpy(int)
            ch = a6 != aU
            share(f"frames_action_changes_{ref}", ch)
            r[f"frames_action_changes_{ref}_n"] = int(ch.sum()); r[f"units_with_change_{ref}"] = int(len(set(d.unit[ch])))
            r[f"changes_on_frames_with_skip_{ref}"] = int((ch & fr_skip).sum())
            J6, JU = d[f"J_v6S|{ref}"].to_numpy(float), d[f"J_v6U|{ref}"].to_numpy(float)
            r[f"G_v6S_{ref}"] = float((L.sum() - J6.sum()) / L.sum()); r[f"G_v6U_{ref}"] = float((L.sum() - JU.sum()) / L.sum())
            dist = (C @ cnt(JU) - C @ cnt(J6)) / (C @ Lu)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            r.update({f"v6S_minus_v6U_{ref}": (JU.sum() - J6.sum()) / L.sum(), f"lo_{ref}": lo, f"hi_{ref}": hi})
            r[f"agree_piF_v6S_{ref}"] = float((a6 == d[f"a_wf960+C1|{ref}"].to_numpy(int)).mean())
            r[f"agree_piF_v6U_{ref}"] = float((aU == d[f"a_wf960+C1|{ref}"].to_numpy(int)).mean())
        r["px_mean_v6S"] = float(d.px_v6S.mean()); r["px_mean_v6U"] = float(d.px_v6U.mean())
        r["ncrops_mean_v6S"] = float(d.ncrops_v6S.mean()); r["ncrops_mean_v6U"] = float(d.ncrops_v6U.mean())
        r["nguard_mean_v6S"] = float(d.nguard_v6S.mean()); r["nguard_mean_v6U"] = float(d.nguard_v6U.mean())
        stop.append(r)
    (od / "k42_gpu_parity.json").write_text(json.dumps(par, indent=1) + "\n")
    pd.DataFrame(single).to_csv(od / "k42_3b_single_refined.csv", index=False)
    pd.DataFrame(stop).to_csv(od / "k42_4b_stopping.csv", index=False)
    pd.set_option("display.width", 250)
    print(json.dumps(par, indent=1))
    print(pd.DataFrame(single).round(4).to_string(index=False))
    print(pd.DataFrame(stop).T.to_string())


if __name__ == "__main__":
    main()
