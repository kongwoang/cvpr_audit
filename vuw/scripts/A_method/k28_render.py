#!/usr/bin/env python
"""K28 PHASE 3 (post-hoc): qualitative / failure panels of v6-S from the capture run (k28_p3.py --part instr --capture
frames.csv --tag capture; AV2 with the ego mask). No detector runs here. K8 / K15 colours and conventions; separate files,
PDF (vector overlays) + PNG at native resolution, no text:
  _a_draft   draft detections (blue; dashed = replaced by a verified detection), missed relevant objects (dashed red)
  _b_bev     bird's-eye view: sensitive cells of the draft's field (orange), corridor, ego, draft objects (blue)
  _c_crops   v6-S field crops (orange), support crops (blue), guard-grown crops (grey)
  _d_final   v6-S final scene: verified on a crop (red), kept from the draft (blue), annotation (green thin; relevant dashed)
  _e_native  whole frame at native resolution: detections (red), image tinted orange
  _f_unc     uncertainty crops unc2@S + C1: crops (orange), their verified detections (red)
examples.csv: per frame the actions (required / draft / v6-S without and with refiner / u960 + C1 / native + C1 / unc2@S
+ C1), crops, rounds, tensor px.
usage: k28_render.py <frames.csv> <out_dir> [<run dir, default {ds}_test_instr_capture>]
       (frames.csv: dataset, unit, frame, category, reason)
"""
from __future__ import annotations

import glob
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_k = importlib.util.spec_from_file_location("k8", B / "scripts" / "A_method" / "k8_render.py")
K8 = importlib.util.module_from_spec(_k); _k.loader.exec_module(K8)
from rap import planner as P                                                     # noqa: E402
from rap.mono import box_iou                                                     # noqa: E402

BLUE, ORANGE, GREY, RED, GREEN = K8.BLUE, K8.ORANGE, K8.GREY, K8.RED, K8.GREEN
NWS = {"nusc": 1600, "av2": 2048, "ahd": 1920}


def tinted(img, boxes, out, name):
    import matplotlib.pyplot as plt
    H, W = img.shape[:2]
    fig = plt.figure(figsize=(W / 72, H / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img[:, :, ::-1]); ax.set_axis_off()
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.add_patch(plt.Rectangle((0, 0), W, H, facecolor=ORANGE, alpha=0.15, edgecolor="none"))
    for (x1, y1, x2, y2), col, ls, lw in boxes:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, linestyle=ls, linewidth=lw))
    K8.save(fig, out, name)


def main():
    frames = pd.read_csv(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    pp = P.PlannerParams()
    ex = []
    for r in frames.itertuples():
        f = REAL / "results" / "k28" / "p3" / (sys.argv[3] if len(sys.argv) > 3 else "{ds}_test_instr_capture").format(ds=r.dataset) / f"{r.unit}.pkl"
        rows = pickle.load(open(f, "rb"))
        rec = next(x for x in rows if int(x["frame"]) == int(r.frame))
        c = rec["cap"]
        img = cv2.imread(c["path"])
        stem = f"{r.dataset}_{str(r.unit)[:8]}_{int(r.frame):04d}"
        sc = c["scenes"]["v6S"]
        dr = c["draft"]
        gt = c["gt"]
        rel = np.asarray(gt["relevant"], int)
        # draft boxes replaced in the final scene
        kept = np.zeros(len(dr["xyxy"]), bool)
        if len(dr["xyxy"]) and (~sc["from_full"]).any():
            kept = box_iou(np.asarray(dr["xyxy"], float), sc["xyxy"][~sc["from_full"]]).max(1) > 0.999
        found_d = rec.get("found_cheap", np.zeros(len(rel), bool))
        missed = [gt["xyxy"][rel[i]] for i in range(len(rel)) if not np.asarray(found_d, bool)[i]] if len(rel) else []
        K8.img_panel(img, [(b, BLUE, "-" if k else "--", 2) for b, k in zip(dr["xyxy"], kept)] + [(b, RED, "--", 2) for b in missed],
                     out, f"{stem}_a_draft")
        K8.bev_panel(c["sens"], c["v"], pp, c["draft_mono"], out, f"{stem}_b_bev")
        field = c["v6S_field"]
        sup = [x for x in c["v6S_crops"] if not any(np.allclose(x, f_) for f_ in field)]
        K8.img_panel(img, [(b, ORANGE, "-", 3) for b in field] + [(b, BLUE, "-", 2) for b in sup] + [(b, GREY, "-", 2) for b in c["v6S_grown"]],
                     out, f"{stem}_c_crops")
        relset = set(rel.tolist())
        K8.img_panel(img, [(b, GREEN, "--" if i in relset else "-", 1) for i, b in enumerate(gt["xyxy"])] +
                     [(b, RED if ff else BLUE, "-", 2) for b, ff in zip(sc["xyxy"], sc["from_full"])], out, f"{stem}_d_final")
        nat = f"wf{NWS[r.dataset]}"
        tinted(img, [(b, RED, "-", 2) for b in c["F"][nat]["xyxy"]], out, f"{stem}_e_native")
        h = c["heur"].get("unc2@S+C1")
        if h is not None:
            K8.img_panel(img, [(b, ORANGE, "-", 3) for b in h["crops"]] + [(b, RED, "-", 2) for b, ff in zip(h["xyxy"], h["from_full"]) if ff],
                         out, f"{stem}_f_unc")
        e = {"dataset": r.dataset, "unit": r.unit, "frame": int(r.frame), "category": r.category, "reason": r.reason,
             "image": c["path"], "ego_speed_mps": c["v"], "action_required": rec["act_gt"], "draft": rec["a_cheap"],
             "v6S_no_refiner": rec["a_v6S|none"], "v6S": rec["a_v6S|v3a"], "u960_C1": rec["a_wf960+C1|v3a"],
             f"native_C1": rec[f"a_{nat}+C1|v3a"], "unc2S_C1": rec.get("a_unc2@S+C1|v3a"), "crops": rec["ncrops_v6S"],
             "rounds": rec["rounds_v6S"], "px_v6S": rec["px_v6S"], "px_unc2S": rec.get("px_unc2@S+C1"), "panel_stem": stem}
        ex.append(e)
    pd.DataFrame(ex).to_csv(out / "examples.csv", index=False)
    print("K28 RENDER DONE", len(ex))


if __name__ == "__main__":
    main()
