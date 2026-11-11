"""K39-1 A (handoff/K39/PREREG.md): TensorRT execution counters for every engine the pipelines run, and the optional
A1 change: CUDA-graph replay of TensorRT executions (one graph per engine and input shape, captured on first use
right after a plain execution of that shape, so that call's output is the plain one). Installed by monkeypatching
the runner classes in this process only; frozen sources are unchanged.
  mode none    counters only
  mode static  graphs for the fixed-shape detector engines (cheap draft, whole frame: rap.trt.TRTModule)
  mode all     + the dynamic-shape crop engine (staged.StagedTRT, one graph per (batch, h, w))"""
import math

import torch

EXEC = [0]
GRAPH = {"captures": 0, "replays": 0}


def install(mode="none"):
    from rap.trt import TRTModule
    from staged import StagedTRT
    from vtf import c1fast as CF

    _tm = TRTModule.__call__

    def tm_call(self, x):
        EXEC[0] += 1
        if mode == "none":
            return _tm(self, x)
        inp = self.buffers[self.input_idx[0]]
        inp.copy_(x.to(dtype=self.input_dtype, non_blocking=True))
        g = getattr(self, "_k39_graph", None)
        if g is None:
            self.ctx.execute_async_v2(self.bindings, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.ctx.execute_async_v2(self.bindings, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            self._k39_graph = g
            GRAPH["captures"] += 1
        else:
            g.replay()
            GRAPH["replays"] += 1
        return self.buffers[self.output_idx[0]]
    TRTModule.__call__ = tm_call

    _ex = StagedTRT.execute_staged

    def ex_staged(self):
        EXEC[0] += 1
        if mode != "all":
            return _ex(self)
        cache = self.__dict__.setdefault("_k39_graphs", {})
        key = self._shape
        if key not in cache:
            out = _ex(self)
            torch.cuda.synchronize()
            shape = tuple(self.ctx.get_binding_shape(self.out))
            output = self.cache['flat'][1][:int(math.prod(shape))].view(shape)
            bindings = [0] * self.engine.num_bindings
            bindings[self.inp] = int(self.cache['flat'][0].data_ptr())
            bindings[self.out] = int(output.data_ptr())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.ctx.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            cache[key] = (g, output)
            GRAPH["captures"] += 1
            return out
        g, output = cache[key]
        g.replay()
        GRAPH["replays"] += 1
        return output
    StagedTRT.execute_staged = ex_staged

    _run = CF._TRTStatic.run

    def st_run(self):
        EXEC[0] += 1
        return _run(self)
    CF._TRTStatic.run = st_run

    _rf = CF.RefFast.__call__

    def rf_call(self, img, items, *a):
        if items:
            EXEC[0] += int(math.ceil(len(items) / CF.REF_BUCKETS[-1]))
        return _rf(self, img, items, *a)
    CF.RefFast.__call__ = rf_call
