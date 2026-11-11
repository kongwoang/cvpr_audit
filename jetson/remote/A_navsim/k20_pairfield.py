#!/usr/bin/env python
"""K20(4) — interacting objects added to the support of the frozen PDM-Closed field (post-hoc; 1,000-token subset).
From k20_pdm --mode pairs: (i) objects of a pair whose joint removal moves the plan although neither single removal does;
(ii) objects whose removal moves the plan only together with a probe that does not move it alone. Their cheap
detections (the box navsim_field would crop for them) are added to the token's support boxes; everything else is the
frozen field. Writes results/navsim_exp/field_k20_{pairs,pairs_probe}/<log>.pkl (subset tokens only).
usage: k20_pairfield.py
"""
from __future__ import annotations

import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("kp", B / "scripts/A_navsim/k20_pdm.py")
KP = importlib.util.module_from_spec(_s); _s.loader.exec_module(KP)
NF, NP = KP.NF, KP.NP
from navsim.common.dataclasses import SceneFilter, SensorConfig                  # noqa: E402
from navsim.common.dataloader import SceneLoader                                 # noqa: E402


def main():
    split = os.environ.get("NAVSIM_SPLIT", "test")
    gz = float((KP.DETS / "ground_z.txt").read_text())
    priors = pickle.load(open(KP.X / "dets" / "size_priors.pkl", "rb"))
    pr = {}
    for f in sorted(KP.OUT.glob("pairs__*.pkl")):
        for r in pickle.load(open(f, "rb")):
            if "err" not in r:
                pr[(r["log"], r["token"])] = r
    outs = {v: KP.X / f"field_k20_{v}" for v in ("pairs", "pairs_probe")}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)
    n_aug = {v: 0 for v in outs}
    for log in sorted({lg for lg, _ in pr}):
        toks = [t for lg, t in pr if lg == log]
        frozen = pickle.load(open(KP.FIELD / f"{log}.pkl", "rb"))
        dets = pickle.load(open(KP.DETS / f"{log}.pkl", "rb"))
        sl = SceneLoader(NP.DATA / "navsim_logs" / split, NP.DATA / "sensor_blobs" / split,
                         SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                                     log_names=[log], tokens=toks), SensorConfig.build_no_sensors())
        res = {v: {} for v in outs}
        for tok in sl.tokens:
            r = pr[(log, tok)]
            scene = sl.get_scene_from_token(tok)
            idx = scene.scene_metadata.num_history_frames - 1
            cam = NP.Cam(*dets[tok]["cam"])
            draft = NP.branch_annotations(scene.frames[idx].annotations, dets[tok]["cheap"], cam, gz, priors, "cheap")
            dx = np.asarray(dets[tok]["cheap"]["xyxy"], float).reshape(-1, 4)

            def det_box(i):
                tk = draft.track_tokens[i]
                if tk.startswith("fp:"):
                    return dx[int(tk.split(":")[-1])]
                return dx[int(np.argmax(NP.iou([cam.project(draft.boxes[i])], dx)[0]))] if len(dx) else None
            objs = {"pairs": sorted({k for p in r["pairs_moving"] for k in p})}
            objs["pairs_probe"] = sorted(set(objs["pairs"]) | {i for _, _, i in r["probe_drop_moving"]})
            for v, ids in objs.items():
                rec = dict(frozen[tok])
                extra = [b for b in (det_box(i) for i in ids) if b is not None]
                rec["support_boxes"] = list(rec.get("support_boxes", [])) + extra
                res[v][tok] = rec
                n_aug[v] += bool(extra)
        for v, d in outs.items():
            with open(d / f"{log}.pkl", "wb") as fh:
                pickle.dump(res[v], fh)
    print("tokens with added support:", n_aug, "of", len(pr))
    print("K20 PAIRFIELD DONE")


if __name__ == "__main__":
    main()
