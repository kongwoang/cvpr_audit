#!/usr/bin/env python
"""K40 (private helper): NAVSIM per-token records (tab:navsim) -> <out>/navsim_navtest.parquet (branch, token, score: the
devkit's one-stage EPDMS of every valid navtest token, latest scoring file per branch) and <out>/navsim_navhard.parquet +
navsim_navhard_groups.parquet (two-stage navhard, common token set = union of failed tokens left out for every branch):
per used token its final score and scene-aggregator weight (devkit create_scene_aggregators + compute_final_scores), and
per two-stage group the stage-one tokens (orig, prev) and their stage-two pairs. The combined EPDMS is the devkit's group
multiplication over these rows.
usage (env_nuplan.sh, NAVSIM_EXP_ROOT, NAVSIM_DEVKIT_ROOT, PYTHONPATH as in k30_navhard_chain2.sh): k40_export_navsim.py <out_dir>"""
import glob
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

sys.path.insert(0, str(Path(os.environ["B"]) / "scripts/A_navsim"))
import k30_navhard_score as S                                                     # noqa: E402

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
DEV = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
TEST = {"reference": "reference", "cheap": "cheap", "full960": "full960", "native": "native", "spec_pdm": "spec_pdm",
        "hybrid_pdm": "hybrid_pdm_fixed2"}
HARD = ("reference", "cheap", "full960", "native", "spec_pdm", "hybrid_pdm")


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    rows = []
    for b, dn in TEST.items():
        x = pd.read_csv(sorted(glob.glob(str(EXP / "k30_v2" / f"navtest_{dn}" / "*.csv")))[-1])
        x = x[(x.token != "average_all_frames") & x.valid.astype(bool)]
        rows.append(pd.DataFrame({"branch": b, "token": x.token.astype(str), "score": x.score.astype(float)}))
    t = pd.concat(rows, ignore_index=True)
    t.to_parquet(od / "navsim_navtest.parquet", index=False, compression="zstd")
    y = yaml.safe_load((DEV / "navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml").read_text())
    mapping = y["reactive_all_mapping"]
    ps = TrajectorySampling(num_poses=40, interval_length=0.1)
    raw = {b: pd.read_pickle(EXP / "k30_v2" / f"navhard_{b}" / "raw_rows.pkl") for b in HARD}
    union = set().union(*[set(d.loc[~d["valid"].astype(bool), "token"]) for d in raw.values()])
    tok, grp = [], []
    for b, df in raw.items():
        scene = set(df["token"])
        ok = set(df.loc[df["valid"].astype(bool), "token"]) - union
        allm = {}
        for o, p, pairs in mapping:
            if not (p in scene or o in scene):
                continue
            keep = [tuple(q) for q in pairs if all(t_ in ok for t_ in q)]
            if o in ok and p in ok and keep:
                allm[(o, p)] = keep
        used = {t_ for (o, p), pairs in allm.items() for t_ in (o, p, *[x for pr in pairs for x in pr])}
        d = df[df["token"].isin(used)].copy()
        d = S.R.create_scene_aggregators(allm, d, ps)
        d = S.R.compute_final_scores(d)
        tok.append(pd.DataFrame({"branch": b, "token": d.token.astype(str), "score": d.score.astype(float), "weight": d.weight.astype(float)}))
        for g, ((o, p), pairs) in enumerate(sorted(allm.items())):
            for q in pairs:
                grp.append({"branch": b, "group": g, "orig": o, "prev": p, "first": q[0], "second": q[1]})
        print(b, len(allm), "groups", len(used), "tokens", flush=True)
    pd.concat(tok, ignore_index=True).to_parquet(od / "navsim_navhard.parquet", index=False, compression="zstd")
    pd.DataFrame(grp).to_parquet(od / "navsim_navhard_groups.parquet", index=False, compression="zstd")
    print("navtest", t.groupby("branch").score.agg(["size", "mean"]).round(4).to_dict())


if __name__ == "__main__":
    main()
