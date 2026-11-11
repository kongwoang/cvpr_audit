"""Isolated direct-input crop staging, preserving frozen geometry and arithmetic."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from vtf.gpuprep import PAD
from vtf.trtdyn import DynTRT
from vtf import gpupost as GP


class StagedTRT(DynTRT):
    def input_view(self, shape):
        if 'flat' not in self.cache:
            maximum = tuple(self.engine.get_profile_shape(0, self.inp)[2])
            assert self.ctx.set_binding_shape(self.inp, maximum)
            output = tuple(self.ctx.get_binding_shape(self.out))
            self.cache['flat'] = (
                torch.empty(int(np.prod(maximum)), dtype=self.in_dtype, device=self.device),
                torch.empty(int(np.prod(output)), dtype=self.out_dtype, device=self.device))
        if not self.ctx.set_binding_shape(self.inp, tuple(shape)):
            raise ValueError(('outside profile', shape))
        self._shape = tuple(shape)
        return self.cache['flat'][0][:int(np.prod(shape))].view(shape)

    def execute_staged(self):
        shape = tuple(self.ctx.get_binding_shape(self.out))
        output = self.cache['flat'][1][:int(np.prod(shape))].view(shape)
        bindings = [0] * self.engine.num_bindings
        bindings[self.inp] = int(self.cache['flat'][0].data_ptr())
        bindings[self.out] = int(output.data_ptr())
        if not self.ctx.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        return output


def letterbox_into(x, out):
    """Same float input and F.interpolate as gpuprep.letterbox; no intermediate canvas."""
    h, w = x.shape[1:]
    net_h, net_w = out.shape[1:]
    ratio = min(net_h / h, net_w / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    top = int(round((net_h-nh) / 2 - .1))
    left = int(round((net_w-nw) / 2 - .1))
    out.fill_(PAD)
    out[:, top:top+nh, left:left+nw] = F.interpolate(
        x[None], size=(nh, nw), mode='bilinear', align_corners=False)[0]
    return ratio, (left, top)


class StagedCrops:
    def __init__(self, gpu_frame, scale, engine_path, mode, threshold, stats):
        self.G, self.s = gpu_frame, scale
        self.eng = StagedTRT(engine_path)
        self.Mode, self.threshold, self.stats = mode, threshold, stats

    def items(self, crops):
        items = []
        for i, crop in enumerate(crops):
            x1, y1 = int(crop[0]), int(crop[1])
            x2 = max(math.ceil(crop[2]), x1+8)
            y2 = max(math.ceil(crop[3]), y1+8)
            sub = self.G.x[:, y1:y2, x1:x2]
            nh = min(1024, max(32, math.ceil(sub.shape[1] * self.s / 32) * 32))
            nw = min(1024, max(32, math.ceil(sub.shape[2] * self.s / 32) * 32))
            items.append((i, sub, nh, nw, (y2-y1, x2-x1, 3), x1, y1))
        return items

    def post(self, raw, metadata, mode, out):
        columns = GP.host_columns(raw, mode, self.threshold)
        for columns_i, (item, ratio, pad) in zip(columns, metadata):
            i, _, _, _, shape, x1, y1 = item
            det = GP.post_lean(columns_i, mode, ratio, pad, shape, min_conf=self.threshold)
            keep = det['conf'] >= self.threshold
            out[i] = dict(xyxy=det['xyxy'][keep].astype(float)+np.array([x1,y1,x1,y1], float),
                          conf=det['conf'][keep].astype(float), coarse=det['coarse'][keep])

    def __call__(self, crops):
        groups = {}
        for item in self.items(crops):
            groups.setdefault((item[2], item[3]), []).append(item)
        out = [None] * len(crops)
        for (nh, nw), items in groups.items():
            mode = self.Mode(f'c{nh}x{nw}', nh, nw)
            for start in range(0, len(items), self.eng.max_batch):
                chunk = items[start:start+self.eng.max_batch]
                view = self.eng.input_view((len(chunk), 3, nh, nw))
                metadata = [(item, *letterbox_into(item[1], view[k])) for k, item in enumerate(chunk)]
                raw = self.eng.execute_staged()
                self.stats['calls'] += 1
                self.stats['tpx'] = self.stats.get('tpx', 0) + len(chunk)*nh*nw
                self.post(raw, metadata, mode, out)
        return out


class StagedOneCall:
    def __init__(self, crop_runner):
        self.runner = crop_runner
        self.eng = crop_runner.eng

    def __call__(self, crops):
        if not crops:
            return []
        r = self.runner
        items = r.items(crops)
        h, w = max(it[2] for it in items), max(it[3] for it in items)
        mode = r.Mode(f'c{h}x{w}', h, w)
        out = [None] * len(crops)
        for start in range(0, len(items), self.eng.max_batch):
            chunk = items[start:start+self.eng.max_batch]
            view = self.eng.input_view((len(chunk), 3, h, w))
            view.fill_(PAD)
            metadata = [(item, *letterbox_into(item[1], view[k, :, :item[2], :item[3]]))
                        for k, item in enumerate(chunk)]
            raw = self.eng.execute_staged()
            r.stats['exec'] = r.stats.get('exec', 0)+1
            r.stats['tpx_b'] = r.stats.get('tpx_b', 0)+len(chunk)*h*w
            r.post(raw, metadata, mode, out)
        return out
