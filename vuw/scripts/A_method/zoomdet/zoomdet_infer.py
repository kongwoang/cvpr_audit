#!/usr/bin/env python
"""ZoomDet inference on the nuScenes new-TEST keyframes (native 1600x900; their test pipeline resizes to 960x576 and their
SaliencyYOLODetector.predict unwarps the boxes). Writes one YOLO-style txt per keyframe (class cx cy w h conf, normalised
to 1600x900, classes 0 vehicle / 1 person / 2 cyclist, conf >= 0.1), the format esod_eval.py scores.
Run in envs/zoomdet with PYTHONPATH = third_party/zoomdet_yolo.  usage: zoomdet_infer.py <config> <checkpoint> <out_dir>"""
import sys
from pathlib import Path

import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config
from mmyolo.utils import register_all_modules

COARSE = {0: 1, 1: 2, 3: 2, 2: 0, 5: 0, 7: 0}          # COCO index -> 0 vehicle (car/bus/truck), 1 person, 2 cyclist


def main():
    cfg_p, ck, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    register_all_modules()
    cfg = Config.fromfile(cfg_p)
    pipe = [t for t in cfg.test_dataloader.dataset.pipeline if "LoadAnnotations" not in t["type"]]
    cfg.test_dataloader.dataset.pipeline = pipe
    cfg.test_pipeline = pipe
    model = init_detector(cfg, ck, device="cuda:0")
    # their predict() reads gt_instances.bboxes only to accumulate IoU statistics in test_cfg (no effect on the
    # predictions); at test time we give it an EMPTY ground truth (no GT used)
    import torch
    _pred = model.predict

    def predict(batch_inputs, batch_data_samples, rescale=True):
        for ds in batch_data_samples:
            if "bboxes" not in ds.gt_instances:
                ds.gt_instances.bboxes = torch.zeros((0, 4), device=batch_inputs.device)
        return _pred(batch_inputs, batch_data_samples, rescale)
    model.predict = predict
    out.mkdir(parents=True, exist_ok=True)
    import os
    lst = Path(os.environ.get("ZD_LIST", Path(cfg.data_root) / "data" / "zoomdet_nusc" / "test_list.txt")).read_text().split("\n")
    n = 0
    for line in lst:
        if not line.strip():
            continue
        name, path = line.split(" ", 1)
        res = inference_detector(model, path)
        r = res.pred_instances
        H0, W0 = (float(x) for x in res.metainfo["ori_shape"][:2])     # nuScenes 900 x 1600; AV2 2048 x 1550
        b, s, lab = r.bboxes.cpu().numpy(), r.scores.cpu().numpy(), r.labels.cpu().numpy()
        rows = []
        for (x1, y1, x2, y2), sc, lb in zip(b, s, lab):
            if sc < 0.1 or int(lb) not in COARSE:
                continue
            rows.append("%d %.6f %.6f %.6f %.6f %.5f" % (COARSE[int(lb)], (x1 + x2) / 2 / W0, (y1 + y2) / 2 / H0,
                                                         (x2 - x1) / W0, (y2 - y1) / H0, sc))
        (out / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        n += 1
        if n % 1000 == 0:
            print(f"  {n} keyframes", flush=True)
    print(f"{n} keyframes written")
    print("ZOOMDET INFER DONE")


if __name__ == "__main__":
    main()
