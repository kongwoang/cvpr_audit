#!/usr/bin/env python
"""K43 (post-hoc, CPU, annotations only; handoff/K43/PREREG.md): decisive annotated object on the DEV units, with the k30_gt
rule (argmax of rap.planner.required_decel per object, default PlannerParams), to fix the box-height tercile cut points.
Dev units = the units of results/k30/dev_devref/<ds> (asserted equal to instr_eval units_<ds>(<split>)).
Writes results/k43/devgt/<ds>.pkl (unit, frame, v, a_gt, act_gt, dec_h_px, dec_class). Never writes to results/k30.
usage: k43_dev_gt.py --dataset nusc|av2 --split <instr_eval split name>"""
import argparse
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
from rap import planner as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--split", required=True)
    a = ap.parse_args(); ds = a.dataset
    want = sorted(p.stem for p in (B / "results/k30/dev_devref" / ds).glob("*.pkl"))
    gen = IE.units_nusc(a.split) if ds == "nusc" else IE.units_av2(a.split)
    pp = P.PlannerParams(); rows = []; got = []
    for unit, cam, frames in gen:
        got.append(unit)
        if unit not in want:
            continue
        for fr, path, v, g, cached in frames():
            if isinstance(cached, str):
                continue
            a_gt, a_obj = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, np.zeros(0))
            r = {"unit": unit, "frame": int(fr), "v": float(v), "a_gt": float(a_gt), "act_gt": P.discrete_action(a_gt, pp),
                 "dec_h_px": np.nan, "dec_class": ""}
            if a_gt > 0 and len(g):
                k = int(np.argmax(np.asarray(a_obj, float)))
                rr = {c: g[c][k] for c in g.dtype.names} if hasattr(g, "dtype") and g.dtype.names else g.iloc[k]
                r["dec_h_px"] = float(rr["y2"] - rr["y1"])
                r["dec_class"] = str(rr["class"]) if "class" in rr else str(rr["coarse"])
            rows.append(r)
        print(unit, len(rows), flush=True)
    missing = sorted(set(want) - set(got))
    od = B / "results/k43/devgt"; od.mkdir(parents=True, exist_ok=True)
    pickle.dump(pd.DataFrame(rows), open(od / f"{ds}.pkl", "wb"))
    print("DEV GT DONE", ds, "units wanted", len(want), "found", len(set(want) & set(got)), "missing", missing[:5], flush=True)


if __name__ == "__main__":
    main()
