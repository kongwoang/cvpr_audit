#!/usr/bin/env python
"""K40-5 / D06 analysis: v6S_lc (low-confidence corridor seeds) vs the frozen v6-S control on nuScenes test (9,181 gt
frames). Same conventions as k41_d03_analyze.py (K40 release Bootstrap, Delta = v6-S - row, repair vs pi(F) = u960+C1,
natural-miss recovery). px of v6S_lc includes the extra cheap pass; px without it is reported too.
usage: k40r_d06_analyze.py <out_dir>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k41_d03_analyze as A  # noqa: E402


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    import glob, pickle
    rows, meta = [], []
    for f in sorted(glob.glob(str(A.B / "results/k40r/lowconf/nusc/*.pkl"))):
        rows += pickle.load(open(f, "rb")); meta.append(json.loads(Path(f).with_suffix(".done.json").read_text()))
    d = pd.DataFrame(rows); d["unit"] = d.unit.astype(str)
    d = d[d["gt"].astype(bool)].sort_values(["unit", "frame"]).reset_index(drop=True)
    R = A.records("nuscenes", A.RECS)
    m = d[["unit", "frame", "J_v6S|v3a"]].merge(R[["unit", "frame", "J_v6S|v3a", "a_wf960+C1|none"]], on=["unit", "frame"],
                                                 how="left", suffixes=("", "@rel"), validate="1:1")
    checks = {"frames": len(d), "units": len(meta), "unit_failures": int(sum(len(x["failures"]) for x in meta)),
              "max_abs_dJ_v6S_vs_release": float(np.nanmax(np.abs(m["J_v6S|v3a"] - m["J_v6S|v3a@rel"]))),
              "frames_with_corridor_seeds": int((d.lowconf_n_corridor > 0).sum()), "seeds_total": int(d.lowconf_n_corridor.sum()),
              "band_boxes_total": int(d.lowconf_n_band.sum())}
    assert len(d) == len(R) == 9181 and checks["unit_failures"] == 0 and checks["max_abs_dJ_v6S_vs_release"] < 1e-9, checks
    bt = A.Bootstrap(d.unit); L, Jd = d.J_cheap.to_numpy(float), d["J_v6S|v3a"].to_numpy(float)
    F = m["a_wf960+C1|none"].to_numpy(int); rep = d.a_cheap.to_numpy(int) != F; e6 = d["a_v6S|none"].to_numpy(int) == F
    out = []
    for ref in ("v3a", "none"):
        x = A.draws(bt, L, d[f"J_v6S|{ref}"].to_numpy(float), d[f"J_v6S_lc|{ref}"].to_numpy(float))
        eq = d["a_v6S_lc|none"].to_numpy(int) == F; sd = bt.share_delta(e6, eq, rep)
        rr, n = A.recovery(d, "v6S_lc", "base"); r6, _ = A.recovery(d, "v6S", "base")
        out.append({"row": "v6S_lc", "refiner": ref, "G": A.G(d, f"J_v6S_lc|{ref}"), "G_v6S": A.G(d, f"J_v6S|{ref}"),
                    "delta": A.G(d, f"J_v6S|{ref}") - A.G(d, f"J_v6S_lc|{ref}"), "ci_lo": float(np.percentile(x, 2.5)),
                    "ci_hi": float(np.percentile(x, 97.5)), "px_mean": float(d.px_v6S_lc.mean()),
                    "px_crops_only_mean": float(d.px_v6S_lc_crops_only.mean()), "px_v6S": float(d.px_v6S.mean()),
                    "repair_row": float(eq[rep].mean()), "repair_v6S": float(e6[rep].mean()), "repair_ci_lo": sd[1], "repair_ci_hi": sd[2],
                    "natural_recovery_row": rr, "natural_recovery_v6S": r6, "natural_misses": n,
                    "action_changed_frames": int((d["a_v6S_lc|v3a"] != d["a_v6S|v3a"]).sum())})
    pd.DataFrame(out).to_csv(od / "k40_5_lowconf.csv", index=False)
    (od / "k40_5_lowconf_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    print(pd.DataFrame(out).round(4).to_string(index=False)); print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
