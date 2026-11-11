"""LZU (Thavamani et al., CVPR 2023; official tchittesh/lzu @ 361afb2) around an Ultralytics YOLO DetectionModel.

As their `LZUFCOSMono3D.extract_feat`: the input is zoomed with the fixed separable saliency grid of
`lzu/fixed_grid.py` (RecasensSaliencyToGridMixin), the backbone + neck run on the zoomed image, and every feature map
entering the head is unzoomed with the inverse grid of `lzu/invert_grid.py` (bilinear, zeros padding), so the head
predicts in uniform image coordinates.  Both official files are imported unchanged (registry import stubbed, the
removed `np.float` alias restored).  Grids are fixed (no gradient), computed per input shape and cached.

patch(model, saliency) swaps the classes of the first Conv (zoom) and the Detect head (unzoom) in place; the swapped
classes live in this module, so checkpoints saved by the Ultralytics trainer load wherever it is importable.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, Detect

GRID_SHAPE, FWHM = (27, 48), 10                          # configs/lzu_fcos3d_*.py
_LZU = Path(os.environ.get("B", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR")) / "third_party" / "lzu" / "lzu"


def _official():
    if not hasattr(np, "float"):
        np.float = float                                 # make1DGaussian uses the removed alias
    if "mmdet3d.models.builder" not in sys.modules:
        class _Reg:
            def register_module(self, *a, **k):
                return lambda c: c

            def build(self, *a, **k):
                return None
        b = types.ModuleType("mmdet3d.models.builder")
        b.MODELS = _Reg()
        for n in ("mmdet3d", "mmdet3d.models"):
            sys.modules.setdefault(n, types.ModuleType(n))
        sys.modules["mmdet3d.models.builder"] = b
    mods = []
    for name in ("fixed_grid", "invert_grid"):
        spec = importlib.util.spec_from_file_location(f"lzu_official_{name}", _LZU / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        mods.append(m)
    return mods


FG, IG = _official()


class LZUState:
    """The saliency (1, 1, 27, 48) and the per-shape grids (float32, cached on first use)."""

    def __init__(self, saliency: torch.Tensor):
        self.saliency = saliency.detach().float().cpu().reshape(1, 1, *GRID_SHAPE)
        self._fwd, self._inv, self.last_hw = {}, {}, None

    def __getstate__(self):
        return {"saliency": self.saliency}

    def __setstate__(self, st):
        self.__init__(st["saliency"])

    def forward_grids(self, H, W, device):
        k = (H, W, str(device))
        if k not in self._fwd:
            gen = FG.RecasensSaliencyToGridMixin(output_shape=(H, W), grid_shape=GRID_SHAPE, separable=True,
                                                 attraction_fwhm=FWHM, anti_crop=True)
            gen.filter = gen.filter.to(device)
            s = self.saliency.to(device)
            up, g = gen.separable_saliency_to_grid(torch.zeros(1), s.sum(dim=2), s.sum(dim=3), device)
            self._fwd[k] = (up, g)
        return self._fwd[k]

    def inverse_grid(self, H, W, fshape, device):
        k = (H, W, tuple(fshape[-2:]), str(device))
        if k not in self._inv:
            _, g = self.forward_grids(H, W, device)
            self._inv[k] = IG.invert_grid(g[0:1], (1, fshape[1], fshape[2], fshape[3]), separable=True)
        return self._inv[k]

    def zoom(self, x):
        H, W = x.shape[-2:]
        self.last_hw = (H, W)
        up, _ = self.forward_grids(H, W, x.device)
        return F.grid_sample(x, up.to(x.dtype).expand(len(x), -1, -1, -1), align_corners=True)

    def unzoom(self, feats):
        H, W = self.last_hw
        return [F.grid_sample(f, self.inverse_grid(H, W, f.shape, f.device).to(f.dtype).expand(len(f), -1, -1, -1),
                              mode="bilinear", align_corners=True, padding_mode="zeros") for f in feats]


class ZoomConv(Conv):
    def forward(self, x):
        return super().forward(self.lzu.zoom(x))

    def forward_fuse(self, x):
        return super().forward_fuse(self.lzu.zoom(x))


class UnzoomDetect(Detect):
    def forward(self, x):
        return super().forward(self.lzu.unzoom(list(x)))


def patch(model, saliency):
    """model: an Ultralytics DetectionModel (YOLOv8). Returns it, zoom/unzoom inserted."""
    st = LZUState(saliency)
    first, head = model.model[0], model.model[-1]
    assert type(first) is Conv and type(head) is Detect, (type(first), type(head))
    first.__class__, head.__class__ = ZoomConv, UnzoomDetect
    first.lzu = head.lzu = st
    return model
