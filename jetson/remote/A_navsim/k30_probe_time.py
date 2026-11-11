#!/usr/bin/env python
"""K30 V6 (ii) (post-hoc; CPU; nuPlan env): MEASURED planner-probing runtime per frame for the 7x7, 5x5, 3x3 grids and the
plan-corridor cells (K20 derived the last three as calls x mean frozen call), with K20's identical-output fast
implementation (k20_fast.FastToken: planner state built once per token, roll-outs reused for identical proposals), on K20's
1,000-token subset (k20_pdm.subset), one thread per process. Per token and variant: planner calls, wall clock of the whole
probing (state build + calls), and the time spent inside PDM-Closed's compute_planner_trajectory.
Grids (K20 section 3): 7x7 = navsim_field.X_GRID x Y_GRID; 5x5 x {8, 12, 23, 40, 55} y {+-10.5, +-3.5, 0}; 3x3 x {8, 23, 55}
y {0, +-3.5}; corridor = the 7x7 cells within 1.75 m + half a probe width of the draft plan. Support calls (one removal per
eligible in-camera draft object) are the same in every variant.
results/navsim_exp/k30_probe_time/<shard>.pkl ; usage: k30_probe_time.py [--shard i/n]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import time
from pathlib import Path

import numpy as np

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("kf", B / "scripts/A_navsim/k20_fast.py")
KF = importlib.util.module_from_spec(_s); _s.loader.exec_module(KF)
KP, NF, NP = KF.KP, KF.NF, KF.NP
from navsim.common.dataclasses import SceneFilter, SensorConfig                  # noqa: E402
from navsim.common.dataloader import SceneLoader                                 # noqa: E402
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor  # noqa: E402

GRIDS = {"7x7": (NF.X_GRID, NF.Y_GRID), "5x5": (np.array([8.0, 12, 23, 40, 55]), np.array([-10.5, -3.5, 0, 3.5, 10.5])),
         "3x3": (np.array([8.0, 23, 55]), np.array([-3.5, 0, 3.5])), "corridor": (NF.X_GRID, NF.Y_GRID)}


def specs_of(cam, draft, gz, L, Wd, Ht, xs, ys, keep=None):
    out = []
    for x in xs:
        for y in ys:
            if keep is not None and not keep(x, y):
                continue
            box = np.array([x, y, gz + Ht / 2, L, Wd, Ht, 0.0])
            if cam.project(box) is not None:
                out.append(("probe", x, y, box))
    for i, name in enumerate(draft.names):
        if name in NP.ELIGIBLE and cam.project(draft.boxes[i]) is not None:
            out.append(("support", i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    split = os.environ.get("NAVSIM_SPLIT", "test")
    gz = float((KP.DETS / "ground_z.txt").read_text())
    priors = pickle.load(open(KP.X / "dets" / "size_priors.pkl", "rb"))
    L, Wd, Ht = priors["vehicle"]
    todo = KP.subset()
    proc = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
    od = KP.X / "k30_probe_time"
    od.mkdir(parents=True, exist_ok=True)
    fo = od / f"shard{si}.pkl"
    if fo.exists():
        return
    logs = sorted({lg for lg, _ in todo})
    rows = []
    for li, log in enumerate(logs):
        if li % sn != si:
            continue
        toks = [t for lg, t in todo if lg == log]
        dets = pickle.load(open(KP.DETS / f"{log}.pkl", "rb"))
        sl = SceneLoader(NP.DATA / "navsim_logs" / split, NP.DATA / "sensor_blobs" / split,
                         SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                                     log_names=[log], tokens=toks), SensorConfig.build_no_sensors())
        for tok in sl.tokens:
            scene = sl.get_scene_from_token(tok)
            idx = scene.scene_metadata.num_history_frames - 1
            cam = NP.Cam(*dets[tok]["cam"])
            draft = NP.branch_annotations(scene.frames[idx].annotations, dets[tok]["cheap"], cam, gz, priors, "cheap")
            for var, (xs, ys) in GRIDS.items():
                t0 = time.perf_counter()
                ft = KF.FastToken(proc, scene, idx, draft)
                pl = ft.pl
                inner = [0.0]
                orig = pl.compute_planner_trajectory

                def timed(pi, _o=orig):
                    t_ = time.perf_counter()
                    r = _o(pi)
                    inner[0] += time.perf_counter() - t_
                    return r
                pl.compute_planner_trajectory = timed
                keep = None
                if var == "corridor":                    # cells near the draft plan (ft.base: 8 poses, ego-local)
                    pts = np.vstack([[0.0, 0.0], np.asarray(ft.base.poses, float)[:, :2]])
                    lim = 1.75 + Wd / 2

                    def keep(x, y, pts=pts, lim=lim):
                        return float(np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - y))) <= lim or \
                            float(np.min(np.abs(np.interp(x, pts[:, 0], pts[:, 1]) - y))) <= lim
                sp = specs_of(cam, draft, gz, L, Wd, Ht, xs, ys, keep)
                moved = [NF.moved(ft.base, ft.call(KF.ann_of(draft, s))) for s in sp]
                pl.compute_planner_trajectory = orig
                ft.restore()
                rows.append({"log": log, "token": tok, "variant": var, "calls": len(sp) + 1, "wall_s": time.perf_counter() - t0,
                             "planner_s": inner[0], "n_probe": sum(s[0] == "probe" for s in sp),
                             "n_support": sum(s[0] == "support" for s in sp), "n_moved": int(np.sum(moved))})
        print(f"  [{li + 1}/{len(logs)}] {log}", flush=True)
    with open(fo, "wb") as fh:
        pickle.dump(rows, fh)
    print("K30 PROBE TIME SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
