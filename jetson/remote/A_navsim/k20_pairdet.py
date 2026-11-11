#!/usr/bin/env python
"""K20(4) — spec_pdm detections with an augmented field (k20_pairfield) for the subset tokens only: navsim_detect's
spec_pdm_dets on the CAM_F0 image with the cheap detections of navsim_detect (YOLOv8s ns_cheap_320, same weights);
every other branch copied from dets_final. Writes NAVSIM_DETS/<log>.pkl for the logs of the subset.
usage: NAVSIM_FIELD=<dir> NAVSIM_DETS=<out> k20_pairdet.py [--shard i/n]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
from pathlib import Path

import cv2

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("nd", B / "scripts/A_navsim/navsim_detect.py")
ND = importlib.util.module_from_spec(_s); _s.loader.exec_module(ND)
from rap.detect import MODES                                                     # noqa: E402

X = B / "results" / "navsim_exp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    FIELD, OUT = Path(os.environ["NAVSIM_FIELD"]), Path(os.environ["NAVSIM_DETS"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ground_z.txt").write_text((X / "dets_final" / "ground_z.txt").read_text())
    det = ND.NS._detector()
    for li, f in enumerate(sorted(FIELD.glob("*.pkl"))):
        if li % sn != si or (OUT / f.name).exists():
            continue
        fields = pickle.load(open(f, "rb"))
        src = pickle.load(open(X / "dets_final" / f.name, "rb"))
        out = {}
        for fr in pickle.load(open(ND.DATA / "navsim_logs" / ND.SPLIT / f.name, "rb")):
            if fr["token"] not in fields:
                continue
            img = cv2.imread(str(ND.DATA / "sensor_blobs" / ND.SPLIT / fr["cams"]["CAM_F0"]["data_path"]))
            dd = {"cheap": det.detect(img, MODES["ns_cheap_320"])[0]}
            rec = dict(src[fr["token"]])
            rec["spec_pdm"], rec["px_spec_pdm"] = ND.spec_pdm_dets(dd, img, fields[fr["token"]])
            k = rec["spec_pdm"]["conf"] >= 0.25
            rec["spec_pdm"] = {kk: rec["spec_pdm"][kk][k] for kk in ("xyxy", "conf", "coarse")}
            out[fr["token"]] = rec
        with open(OUT / f.name, "wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  {f.stem}: {len(out)} tokens", flush=True)
    print("K20 PAIRDET DONE", flush=True)


if __name__ == "__main__":
    main()
