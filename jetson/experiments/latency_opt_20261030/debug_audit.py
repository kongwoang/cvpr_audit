"""Debug-only deterministic audit; never enabled during production timing."""
import hashlib
import json
import numpy as np


def feed(h, value):
    if isinstance(value, dict):
        h.update(b'dict')
        for key in sorted(value):
            feed(h, key); feed(h, value[key])
    elif isinstance(value, (list, tuple)):
        h.update(('sequence:' + str(len(value))).encode())
        for item in value:
            feed(h, item)
    elif isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        h.update(json.dumps([a.dtype.str, a.shape]).encode())
        if a.dtype.hasobject:
            feed(h, a.tolist())
        else:
            h.update(a.tobytes())
    elif isinstance(value, np.generic):
        feed(h, np.asarray(value))
    else:
        h.update((type(value).__name__ + ':' + repr(value)).encode())
    h.update(b'\0')


class Audit:
    def __init__(self):
        self.active = False
        self.digests = {}
        self.counts = {}

    def start(self):
        self.active = True
        self.digests = {}
        self.counts = {}

    def add(self, kind, value):
        if self.active:
            feed(self.digests.setdefault(kind, hashlib.sha256()), value)
            self.counts[kind] = self.counts.get(kind, 0) + 1

    def finish(self):
        self.active = False
        return {k: [self.counts[k], h.hexdigest()] for k, h in sorted(self.digests.items())}

    def install(self, gp, staged_class):
        original = gp.post_lean
        def post(p, mode, r, pad, shape, min_conf=None):
            result = original(p, mode, r, pad, shape, min_conf)
            self.add('postprocess', [p, mode.conf, mode.iou, mode.max_det, r, pad,
                                     shape, min_conf, result])
            return result
        gp.post_lean = post
        items = staged_class.items
        def selected_crops(runner, crops):
            result = items(runner, crops)
            self.add('crop_requests', [crops, [[i[0], *i[2:]] for i in result]])
            return result
        staged_class.items = selected_crops
