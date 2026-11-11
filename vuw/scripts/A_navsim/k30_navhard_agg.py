#!/usr/bin/env python
"""K30 V8a navhard: EPDMS of every branch from its raw per-token rows (k30_navhard_score.py, raw_rows.pkl), with the devkit's
aggregation (k30_navhard_score.aggregate) on (a) the branch's own valid tokens and (b) the COMMON token set: the union of the
failed tokens of all branches left out for every branch. Writes <EXP>/k30_v2/navhard_summary.csv (+ per-token csv per branch).
usage: k30_navhard_agg.py [branches, default reference,cheap,full960,native,spec_pdm,hybrid_pdm]
"""
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

sys.path.insert(0, str(Path(__file__).parent))
from k30_navhard_score import aggregate                                          # noqa: E402

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
DEV = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
BR = (sys.argv[1] if len(sys.argv) > 1 else "reference,cheap,full960,native,spec_pdm,hybrid_pdm").split(",")
COLS = ["score", "no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
        "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]


def main():
    y = yaml.safe_load((DEV / "navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml").read_text())
    mapping = y["reactive_all_mapping"]
    ps = TrajectorySampling(num_poses=40, interval_length=0.1)
    raw = {b: pd.read_pickle(EXP / "k30_v2" / f"navhard_{b}" / "raw_rows.pkl") for b in BR if (EXP / "k30_v2" / f"navhard_{b}" / "raw_rows.pkl").exists()}
    union = set().union(*[set(d.loc[~d["valid"].astype(bool), "token"]) for d in raw.values()]) if raw else set()
    out = []
    for b, d in raw.items():
        toks = set(d["token"])
        for setting, exc in (("own valid tokens", None), ("common tokens (union of failures left out)", union)):
            res, info = aggregate(d, mapping, toks, ps, exclude=exc)
            if exc is not None:
                res.to_csv(EXP / "k30_v2" / f"navhard_{b}" / "common_tokens.csv", index=False)
            row = {"branch": b, "setting": setting, "rows": len(d), "failed": int((~d["valid"].astype(bool)).sum()), **info}
            for part, name in (("combined", "extended_pdm_score_combined"), ("stage_one", "extended_pdm_score_stage_one"),
                               ("stage_two", "extended_pdm_score_stage_two")):
                r = res[res.token == name].iloc[0]
                row[f"EPDMS_{part}"] = r["score"]
                if part != "combined":
                    for c in COLS[1:]:
                        k = f"{c}_{part}"
                        if k in r.index:
                            row[k] = r[k]
            out.append(row)
    o = pd.DataFrame(out)
    o.to_csv(EXP / "k30_v2" / "navhard_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(o[["branch", "setting", "rows", "failed", "failed_or_excluded_tokens", "groups_used", "EPDMS_combined", "EPDMS_stage_one",
             "EPDMS_stage_two"]].round(4).to_string(index=False))
    print(f"union of failed tokens: {len(union)}; K30 NAVHARD AGG DONE")


if __name__ == "__main__":
    main()
