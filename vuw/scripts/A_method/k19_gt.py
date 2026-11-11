#!/usr/bin/env python
"""K19 helper (CPU, no images, no inference): per frame, the annotated-scene quantities the consumer exposes —
required deceleration a* and action, number of annotated objects in the ego corridor within the planner's range,
the minimum ttc and minimum range among them, the deciding object's class. Same frames / filters as instr_eval
(objects with box height >= 10 px).
usage: k19_gt.py --dataset nusc|av2|ahd  ->  results/k19/gt_<ds>.pkl
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
_h = importlib.util.spec_from_file_location("hood", B / "scripts" / "A_method" / "k10_hood.py")
HOOD = importlib.util.module_from_spec(_h); _h.loader.exec_module(HOOD)
from rap import planner as P                                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2", "ahd"])
    args = ap.parse_args()
    pp = P.PlannerParams()
    rows = []
    for unit, cam, gen in HOOD.units(args.dataset):
        for fr, path, v, g, extra in gen():
            r = {"unit": unit, "frame": fr, "v": v, "gt": not isinstance(extra, str)}
            if len(g):
                z, lo, hi, ttc = (np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
                a, a_obj = P.required_decel(z, lo, hi, ttc, v, pp)
                inc = P.in_corridor(lo, hi, pp.corridor_half_w) & (z > 0) & (z <= pp.max_range)
                r.update({"a_gt": float(a), "act_gt": P.discrete_action(a, pp), "n_corridor": int(inc.sum()),
                          "min_ttc_corridor": float(ttc[inc].min()) if inc.any() else np.inf,
                          "min_range_corridor": float(z[inc].min()) if inc.any() else np.inf,
                          "n_relevant": int((np.asarray(a_obj) >= pp.thr_decel).sum()),
                          "decider_cls": str(np.asarray(g["coarse"])[int(np.argmax(a_obj))]) if len(a_obj) and a > 0 else ""})
            else:
                r.update({"a_gt": 0.0, "act_gt": 0, "n_corridor": 0, "min_ttc_corridor": np.inf,
                          "min_range_corridor": np.inf, "n_relevant": 0, "decider_cls": ""})
            rows.append(r)
        print(f"  {unit}: {sum(1 for x in rows if x['unit'] == unit)} frames", flush=True)
    (B / "results" / "k19").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_pickle(B / "results" / "k19" / f"gt_{args.dataset}.pkl")
    print("K19 GT DONE", flush=True)


if __name__ == "__main__":
    main()
