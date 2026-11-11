"""K39-1 A (development only, no test data): where does a crop-engine call spend its time? StagedCrops on a fixed
nuScenes val image; per canvas: full call (stage + execute + host copy + post), execute only, and a CUDA-graph replay
of the execute (output compared bit-for-bit with the plain execute)."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "k38_deadline_20261026"))
from staged import StagedCrops, letterbox_into                                 # noqa: E402
from rap.detect import Mode                                                     # noqa: E402
from vtf.gpuprep import PAD                                                     # noqa: E402

ENG = Path("/home/kongwoang/research/CVPR/engines")


class G:
    pass


def t(f, n=50):
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); a = time.perf_counter()
        f()
        torch.cuda.synchronize(); ts.append(1e3 * (time.perf_counter() - a))
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    img = cv2.imread(sys.argv[1])
    g = G()
    g.x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().cuda()
    stats = {"calls": 0}
    sc = StagedCrops(g, 960 / 1600, ENG / "yolov8s_dyn.engine", Mode, 0.25, stats)
    eng = sc.eng
    s = torch.cuda.Stream()
    for crop in ([700, 400, 900, 520], [500, 350, 1100, 650], [0, 300, 1600, 700], [600, 380, 760, 470]):
        items = sc.items([crop])
        nh, nw = items[0][2], items[0][3]
        full = t(lambda: sc([crop]))
        view = eng.input_view((1, 3, nh, nw))
        letterbox_into(items[0][1], view[0])
        ex = t(lambda: eng.execute_staged())
        ref = eng.execute_staged().clone()
        ref2 = eng.execute_staged().clone()
        det_plain = bool(torch.equal(ref, ref2))
        # CUDA graph of the execute for this shape (same buffers, same bindings)
        shape = tuple(eng.ctx.get_binding_shape(eng.out))
        out = eng.cache["flat"][1][:int(np.prod(shape))].view(shape)
        bindings = [0] * eng.engine.num_bindings
        bindings[eng.inp], bindings[eng.out] = int(eng.cache["flat"][0].data_ptr()), int(out.data_ptr())
        with torch.cuda.stream(s):
            eng.ctx.execute_async_v2(bindings, s.cuda_stream)
            s.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=s):
                eng.ctx.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        gt = t(lambda: gr.replay())
        same = bool(torch.equal(out, ref))
        stg = t(lambda: letterbox_into(items[0][1], view[0]))
        host = t(lambda: out.float().cpu().numpy())
        print(f"canvas {nh}x{nw}: full call {full[0]:.2f}/{full[1]:.2f} ms | execute {ex[0]:.2f}/{ex[1]:.2f} | graph replay "
              f"{gt[0]:.2f}/{gt[1]:.2f} (identical {same}, plain twice identical {det_plain}, max|diff| {float((out.float()-ref.float()).abs().max()):.3g}) | staging {stg[0]:.2f} | host copy {host[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
