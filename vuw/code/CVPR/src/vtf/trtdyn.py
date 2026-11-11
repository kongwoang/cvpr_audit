"""TensorRT runner for an engine with a dynamic input shape (engines/build_dynamic.sh: batch 1-8, H, W 32-1024).

The crop of the verification loop then runs on a canvas of exactly ceil32(crop x scale), the canvas the VUW (torch)
evaluations use, instead of the smallest static engine canvas that holds it (which pads ~2x the pixels on AV2).
One input / output buffer of the maximum profile size is allocated once; each call uses a view of it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class DynTRT:
    def __init__(self, engine_path, device="cuda:0"):
        import tensorrt as trt
        self.rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.rt.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.device = torch.device(device)
        self.inp = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)][0]
        self.out = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)][0]
        dt = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        self.in_dtype = dt[self.engine.get_binding_dtype(self.inp)]
        self.out_dtype = dt[self.engine.get_binding_dtype(self.out)]
        self.cache = {}
        self.max_batch = int(self.engine.get_profile_shape(0, self.inp)[2][0])

    def __call__(self, x):
        """x: (B, 3, H, W) on the device, H and W multiples of 32 -> raw head output (B, 4 + nc, N).
        One input and one output buffer of the profile's maximum size are allocated once; every shape uses a
        contiguous view of them (a per-shape cache grew without bound when crop sizes vary widely)."""
        shp = tuple(x.shape)
        if "flat" not in self.cache:
            mx = tuple(self.engine.get_profile_shape(0, self.inp)[2])
            self.ctx.set_binding_shape(self.inp, mx)
            omx = tuple(self.ctx.get_binding_shape(self.out))
            self.cache["flat"] = (torch.empty(int(np.prod(mx)), dtype=self.in_dtype, device=self.device),
                                  torch.empty(int(np.prod(omx)), dtype=self.out_dtype, device=self.device))
        if not self.ctx.set_binding_shape(self.inp, shp):       # outside the profile: TensorRT keeps the previous shape
            raise ValueError(f"input shape {shp} outside the engine profile {self.engine.get_profile_shape(0, self.inp)}")
        oshape = tuple(self.ctx.get_binding_shape(self.out))
        fi, fo = self.cache["flat"]
        ib = fi[: int(np.prod(shp))].view(shp)
        ob = fo[: int(np.prod(oshape))].view(oshape)
        ib.copy_(x.to(dtype=self.in_dtype))
        bind = [0] * self.engine.num_bindings
        bind[self.inp], bind[self.out] = int(ib.data_ptr()), int(ob.data_ptr())
        self.ctx.execute_v2(bind)
        return ob
