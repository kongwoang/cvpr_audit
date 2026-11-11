#!/usr/bin/env python
"""zoomdet_av2.py from the nuScenes config (zoomdet_nusc.py, written by make_config.py from ZoomDet's own VisDrone
YOLOv8-s config): only the data paths change (AV2 fine-tuning set of av2_ft_prepare.py, COCO json, 80 classes);
same 736x736 square input, 15 epochs, batch 8."""
import os
import re
from pathlib import Path

B = Path(os.environ["B"])
here = Path(__file__).parent
src = (here / "zoomdet_nusc.py").read_text()
D = B / "data" / "av2_ft"
rep = {r"^train_ann_file = .*$": f"train_ann_file = '{(D / 'zoomdet' / 'train.json').relative_to(B)}'",
       r"^train_data_prefix = .*$": f"train_data_prefix = '{(D / 'images' / 'train').relative_to(B)}/'",
       r"^val_ann_file = .*$": f"val_ann_file = '{(D / 'zoomdet' / 'val.json').relative_to(B)}'",
       r"^val_data_prefix = .*$": f"val_data_prefix = '{(D / 'images' / 'val').relative_to(B)}/'"}
out = src
for k, v in rep.items():
    out, n = re.subn(k, v, out, flags=re.M)
    assert n == 1, (k, n)
(here / "zoomdet_av2.py").write_text(out)
print("written zoomdet_av2.py")
