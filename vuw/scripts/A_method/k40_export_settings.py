#!/usr/bin/env python
"""K40 (private helper): per-frame records of the alternative settings (own cheap draft and rows), one parquet each:
records/<dataset>_<setting>.parquet, setting in yolo11s (K28 p3 y11s), finetuned0 / finetuned1 (nuScenes, ft0 / ft1),
nomask (AV2 without the ego mask). Parts ours / basewf / baseheur merged on (unit, frame); J_cheap must agree."""
import glob
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
NAME = {"nusc": "nuscenes", "av2": "av2", "ahd": "argoverse_hd"}
SET = {"y11s": ("yolo11s", ("nusc", "av2", "ahd")), "ft0": ("finetuned0", ("nusc",)), "ft1": ("finetuned1", ("nusc",)), "nomask": ("nomask", ("av2",))}


def recs(d):
    rr = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        rr += pickle.load(open(f, "rb"))
    x = pd.DataFrame(rr)
    x["unit"] = x.unit.astype(str)
    return x[x["gt"].astype(bool)].reset_index(drop=True) if "gt" in x else x


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    key = ["unit", "frame"]
    for tag, (name, dss) in SET.items():
        for ds in dss:
            d = None
            for p in ("ours", "basewf", "baseheur"):
                x = recs(B / "results/k28/p3" / f"{ds}_test_{p}_{tag}")
                cols = [c for c in x.columns if re.match(r"^(a|J|px|ncrops|nguard|rounds)_|^(J_cheap|a_cheap|px_cheap|a_gt|act_gt)$", c) and x[c].dtype != object]
                x = x[key + cols]
                if d is None:
                    d = x
                else:
                    shared = [c for c in cols if c in d]
                    m = d.merge(x, on=key, suffixes=("", "@r"), validate="1:1")
                    for c in shared:
                        assert np.allclose(m[c].astype(float), m[c + "@r"].astype(float), equal_nan=True), (tag, ds, p, c)
                    d = m.drop(columns=[c + "@r" for c in shared])
            f = od / f"{NAME[ds]}_{name}.parquet"
            d.sort_values(key).to_parquet(f, index=False, compression="zstd")
            print(tag, ds, d.shape, round(f.stat().st_size / 2**20, 2), "MB", flush=True)
    # AV2 fine-tuned regime (K33: every detector fine-tuned on AV2 train, ego mask)
    x = recs(B / "results/k33/av2_ft_test")
    cols = [c for c in x.columns if re.match(r"^(a|J|px)_|^(J_cheap|a_cheap|px_cheap|a_gt|act_gt)$", c) and x[c].dtype != object]
    x[key + cols].sort_values(key).to_parquet(od / "av2_finetuned.parquet", index=False, compression="zstd")
    print("av2_finetuned", x[key + cols].shape)
    # braking-threshold grid on the development splits (K37-1: the dev choice of each row's threshold)
    for ds in ("nusc", "av2"):
        x = recs(B / "results/k37/k1/dev" / ds)
        cols = [c for c in x.columns if re.match(r"^(a|J|px)_|^(J_cheap|a_cheap|px_cheap|a_gt|act_gt)(@|$)", c) and x[c].dtype != object]
        x[key + cols].sort_values(key).to_parquet(od / f"{NAME[ds]}_dev_thresholds.parquet", index=False, compression="zstd")
        print("dev thresholds", ds, x[key + cols].shape)
    x = recs(B / "results/k37/k1/hold/ahd")
    cols = [c for c in x.columns if re.match(r"^(a|J|px)_|^(J_cheap|a_cheap|px_cheap|a_gt|act_gt)(@|$)", c) and x[c].dtype != object]
    x[key + cols].sort_values(key).to_parquet(od / "holdout_thresholds.parquet", index=False, compression="zstd")
    print("holdout thresholds", x[key + cols].shape)



if __name__ == "__main__":
    main()
