#!/usr/bin/env python
"""K29: the nuScenes frame lists of 64_k29_latency.py, written once so that the timed / parity processes do not hold the
NuScenesDB (v1.0-trainval, ~9 GB RSS on the Jetson). Per set, a pickle of [(unit, frame, image path, speed, geometry rows)]:
  test20  first 20 new-test scene tokens in sorted order (~/datasets/k29/k29_nusc_test20.txt), images under ~/datasets/k29
  val     nuScenes new val, 152 scenes (k29_nusc_val.txt), images under ~/datasets/k29
  old     K28's set = 55_large_latency.frames_nusc with max_units 25 (DB scene order, scenes whose images are all on the
          Jetson and that have a camera entry), images under ~/datasets/nuscenes/trainval
The camera (rap.cache.Camera, a JSON table) is built by the run itself. Same code as the generators it replaces.
usage: k29_frames.py"""
from __future__ import annotations

import pickle
from pathlib import Path

K29D = Path.home() / "datasets" / "k29"


def main():
    from rap.cache import Camera
    from rap.nusc import NuScenesDB, make_adapter
    from rap.paths import NUSCENES_TRAINVAL
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    for which in ("test20", "val"):
        out, miss = [], 0
        for s in [ln.split()[0] for ln in open(K29D / f"k29_nusc_{which}.txt") if ln.strip()]:
            toks = db.samples(by[s])
            geom, speeds = ad.geometry(s), ad.speeds(s)
            for fr, tk in enumerate(toks):
                g = geom[geom["frame"] == fr]
                p = K29D / "nuscenes" / Path(db.image_path(tk)).relative_to(NUSCENES_TRAINVAL)
                if not p.exists():
                    miss += 1; continue
                out.append((s, fr, str(p), float(speeds[fr]) if fr < len(speeds) else float(speeds[-1]), g[(g["y2"] - g["y1"]) >= 10]))
        pickle.dump(out, open(K29D / f"frames_nusc_{which}.pkl", "wb"))
        print(which, len(out), "frames", len({o[0] for o in out}), "scenes, missing images", miss)
    out, n_sc = [], 0                                  # = 55_large_latency.frames_nusc, max_units 25
    for sc in db.scenes:
        s = db.scene_name(sc)
        toks = db.samples(sc)
        if not all(Path(db.image_path(tk)).exists() for tk in toks):
            continue
        try:
            Camera("nuScenes", s)
        except Exception:                                                        # noqa: BLE001
            continue
        geom, speeds = ad.geometry(s), ad.speeds(s)
        for fr, tk in enumerate(toks):
            g = geom[geom["frame"] == fr]
            out.append((s, fr, str(db.image_path(tk)), float(speeds[fr]) if fr < len(speeds) else float(speeds[-1]), g[(g["y2"] - g["y1"]) >= 10]))
        n_sc += 1
        if n_sc >= 25:
            break
    pickle.dump(out, open(K29D / "frames_nusc_old.pkl", "wb"))
    print("old", len(out), "frames", n_sc, "scenes")


if __name__ == "__main__":
    main()
