"""Read-only K39 strict_guarded extracted for CPU equivalence fixtures.
Source: experiments/k39_vuw_20261027/build.py md5=e89929446807f56efb46284b5e3f9bec
"""
def strict_guarded(tpx, bud_px, skips):
    """SP.guarded (max_grow 2, relevant None) with the strict cap: a call whose canvas would push tpx above
    bud_px is skipped; a skipped growth keeps its pending truncated detection as detected"""
    dcrop = make_detect(tpx, scale9)

    def over(cr):
        k_ = ckey(cr, scale9)
        return tpx[0] + k_[4] * k_[5] > bud_px + 1e-6

    def det_(crop):
        if over(crop):
            skips[0] += 1
            return {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}, []
        d = {k: np.asarray(x) for k, x in dcrop(crop).items()}
        if not len(d["conf"]):
            return d, []
        top, other = SP.truncated(d["xyxy"], crop, W, H, top_only=True)
        tr = other | top
        out = {k: x[~tr] for k, x in d.items()}
        extra = []
        live = [({k: d[k][i:i + 1] for k in d}, crop) for i in np.flatnonzero(tr)]
        for step in range(2):
            if not live:
                break
            nxt = []
            for row, c in live:
                b = row["xyxy"][0]
                g_ = SP.grow(b, c, W, H)
                if over(g_):
                    skips[1] += 1
                    out = {k: np.concatenate([out[k], row[k]]) for k in out}
                    continue
                extra.append(g_)
                dg = {k: np.asarray(x) for k, x in dcrop(g_).items()}
                if not len(dg["conf"]):
                    continue
                ix = np.clip(np.minimum(dg["xyxy"][:, 2], b[2]) - np.maximum(dg["xyxy"][:, 0], b[0]), 0, None)
                iy = np.clip(np.minimum(dg["xyxy"][:, 3], b[3]) - np.maximum(dg["xyxy"][:, 1], b[1]), 0, None)
                io = ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
                j = int(np.argmax(io))
                if io[j] < 0.5:
                    continue
                rj = {k: dg[k][j:j + 1] for k in dg}
                if not SP.truncated(dg["xyxy"][j:j + 1], g_, W, H)[0] or step == 1:
                    out = {k: np.concatenate([out[k], rj[k]]) for k in out}
                else:
                    nxt.append((rj, g_))
            live = nxt
        return out, extra
    return det_

