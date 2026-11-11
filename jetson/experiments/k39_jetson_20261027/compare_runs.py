"""K39-1 A: per-frame output identity and latency of two runs of 89_k39.py on the same frames (a change vs none)."""
import sys
import numpy as np
import pandas as pd


def main():
    a, b = (pd.read_pickle(p + "/frames.pkl") for p in sys.argv[1:3])
    assert len(a) == len(b) and (a.unit.astype(str).values == b.unit.astype(str).values).all() and (a.frame.values == b.frame.values).all()
    rows = sorted({c[3:] for c in a.columns if c.startswith("ms_")})
    bad, out = {}, []
    for c in a.columns:
        if c.startswith(("a_", "px_", "J_", "n_crops_", "rounds_", "calls_", "crop_engine_exec_", "execs_", "calls_loop_", "bound_hit_")):
            if c not in b:
                continue
            x, y = a[c].to_numpy(), b[c].to_numpy()
            n = int((~np.isclose(x.astype(float), y.astype(float), rtol=0, atol=0, equal_nan=True)).sum())
            if n:
                bad[c] = n
        elif c.startswith("crops_"):
            n = sum(1 for u, v in zip(a[c], b[c]) if len(u) != len(v) or any(not np.array_equal(p, q) for p, q in zip(u, v)))
            if n:
                bad[c] = n
    t = ~a.warm.to_numpy()
    for r in rows:
        x, y = a[f"ms_{r}"].to_numpy()[t], b[f"ms_{r}"].to_numpy()[t]
        out.append({"row": r, "mean_a": x.mean(), "mean_b": y.mean(), "p95_a": np.percentile(x, 95), "p95_b": np.percentile(y, 95)})
    print("differing columns (frames):", bad if bad else "none")
    print(pd.DataFrame(out).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
