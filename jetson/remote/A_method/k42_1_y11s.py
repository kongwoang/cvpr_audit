#!/usr/bin/env python
"""K42-1 (post-hoc, CPU; handoff/K42/PREREG.md): YOLO11s whole frame with the full lift (C1 on its own same-scale history
+ refiner v3a) vs DGSP_y11s (v6-S|v3a), from the stored K28 p3 y11s records (k28_p3.py --tag y11s). The table is
computed by k28_p3_analyze.ctable (k24_analyze.Boot, 2,000, seed 0) unchanged; gate = every row equals
handoff/K28/phase3/ctable_y11s.csv (copied to results/k28/p3/tables/ctable_y11s.csv on VUW).
usage: k42_1_y11s.py <out_dir> <reference ctable_y11s.csv>"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("k28p3a", B / "scripts" / "A_method" / "k28_p3_analyze.py")
A = importlib.util.module_from_spec(_s); _s.loader.exec_module(A)
NAT = {"nusc": 1600, "av2": 2048, "ahd": 1920}
SIDES = (512, 640, 960, 1280, "native")


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in ("nusc", "av2", "ahd"):
        r, why = A.ctable("y11s", ds, ("ours", "basewf", "baseheur"))
        assert not why, why
        rows += r
    new = pd.DataFrame(rows)
    old = pd.read_csv(sys.argv[2])
    key = ["dataset", "row"]
    m = old.merge(new, on=key, suffixes=("_old", ""), how="outer", validate="1:1", indicator=True)
    num = [c for c in old.columns if c not in key and old[c].dtype.kind == "f"]
    gate = {"rows_old": len(old), "rows_new": len(new), "unmatched": int((m._merge != "both").sum())}
    gate["max_abs_diff"] = {c: float(np.nanmax(np.abs(m[c] - m[c + "_old"]))) for c in num}
    gate["verdict_diffs"] = int(sum((m[c] != m[c + "_old"]).sum() for c in ("verdict_no_refiner", "verdict_refiner")))
    gate["pass"] = bool(gate["unmatched"] == 0 and max(gate["max_abs_diff"].values()) <= 1e-12 and gate["verdict_diffs"] == 0)
    (od / "k42_1_gate.json").write_text(json.dumps(gate, indent=1) + "\n")
    print(json.dumps(gate, indent=1))
    assert gate["pass"], "K42-1 gate failed"
    out = []
    for ds in ("nusc", "av2", "ahd"):
        d = new[new.dataset == ds].set_index("row")
        dg = d.loc["v6S"]
        for s in SIDES:
            L = NAT[ds] if s == "native" else s
            full, ref = d.loc[f"wf{L}+C1"], d.loc[f"wf{L}"]
            out.append({"dataset": ds, "side": f"u{s}" if s != "native" else f"native ({L})", "units": int(full.units), "frames": int(full.frames),
                        "px_mean_pct": 100 * full.tensor_px_mean, "DGSP_G": dg.G_refiner, "DGSP_px_mean_pct": 100 * dg.tensor_px_mean,
                        "G_full_lift": full.G_refiner, "DGSP_minus_full_lift": full.v6S_minus_refiner,
                        "lo_full_lift": full.lo_refiner, "hi_full_lift": full.hi_refiner, "verdict_full_lift": full.verdict_refiner,
                        "G_refiner_only": ref.G_refiner, "DGSP_minus_refiner_only": ref.v6S_minus_refiner,
                        "lo_refiner_only": ref.lo_refiner, "hi_refiner_only": ref.hi_refiner, "verdict_refiner_only": ref.verdict_refiner,
                        "full_lift_row": f"wf{L}+C1|v3a", "refiner_only_row": f"wf{L}|v3a"})
    o = pd.DataFrame(out)
    o.to_csv(od / "k42_1_y11s.csv", index=False)
    pd.set_option("display.width", 250)
    print(o.drop(columns=["full_lift_row", "refiner_only_row", "units", "frames"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
