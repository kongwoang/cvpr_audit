"""Isolated P0 candidate: omit an unused diagnostic in GP.host_columns.

No change to GP.select (its callers need n_cand_raw), preprocessing, order,
thresholds, NMS or output precision. Not enabled in frozen code. CPU parity
is necessary but device/output parity and timing remain gated experiments.
"""
import numpy as np
from vtf import gpupost as GP


def host_columns(raw, mode, min_conf=None):
    if raw.shape[-1] < GP.GPU_SELECT_MIN_ANCHORS:
        h = raw.float().cpu().numpy()
        return [h[i] for i in range(h.shape[0])] if h.ndim == 3 else [h]
    t = raw.float()
    if t.dim() == 2:
        t = t[None]
    best = t[:, 4:].index_select(1, GP._keep_idx(t.device)).max(1).values
    keep = best > mode.conf
    if min_conf is not None:
        keep &= best >= min_conf
    counts = keep.sum(1)
    selected = t.permute(0, 2, 1)[keep]
    host = selected.cpu().numpy()
    cnt = counts.cpu().numpy()
    result, offset = [], 0
    for n in cnt:
        result.append(np.ascontiguousarray(host[offset:offset + n].T))
        offset += n
    return result


def install():
    """Opt in for every row in one process; return reference for restoration."""
    reference = GP.host_columns
    GP.host_columns = host_columns
    return reference
