"""Arbiters: accept or reject an escalation's change of decision, from post-escalation features only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-12
CONFIGS = (("logit", "cell"), ("logit", "pool"), ("gbm", "cell"), ("gbm", "pool"))   # tie order of the prereg


def assert_deployable(cols) -> None:
    bad = [c for c in cols if c.startswith("gt_") or c in ("V", "J_cheap", "J_full")
           or c.endswith("_J_c") or c.endswith("_J_f") or c.endswith("_V")]
    if bad:
        raise ValueError(f"non-deployable feature columns: {bad}")


def make(model: str, seed: int = 0):
    if model == "logit":
        return Pipeline([("i", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("s", StandardScaler()),
                         ("m", LogisticRegression(C=1.0, max_iter=5000))])
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.08, min_samples_leaf=20,
                                          l2_regularization=1.0, max_bins=64, early_stopping=False,
                                          random_state=seed)


class Const:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([1 - np.full(len(X), self.p), np.full(len(X), self.p)])


def fit(model: str, X, V, seed: int = 0):
    """Weighted classifier of 1[V > 0] with weight |V|: p >= 0.5 estimates E[V | x] >= 0."""
    y = (V > 0).astype(int)
    w = np.abs(V)
    keep = w > EPS
    X, y, w = X[keep], y[keep], w[keep]
    if len(np.unique(y)) < 2 or len(y) < 2:
        return Const(float(np.sum(w * y) / max(np.sum(w), EPS)) if len(y) else 1.0)
    w = w / w.mean()
    m = make(model, seed)
    if model == "logit":
        m.fit(X, y, m__sample_weight=w)
    else:
        m.fit(X, y, sample_weight=w)
    return m


def proba(m, X) -> np.ndarray:
    return m.predict_proba(X)[:, 1]


def oof_gain(model: str, pool: pd.DataFrame, target_mask: np.ndarray, cols, n_splits: int = 5, seed: int = 0):
    """Out-of-fold realised arbitration gain on the target rows, folds grouped by unit over the whole pool."""
    X = pool[cols].to_numpy(float)
    V = pool.V.to_numpy(float)
    units = pool.unit_key.to_numpy()
    uniq = np.unique(units)
    k = min(n_splits, len(uniq))
    p = np.full(len(pool), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(X, V, units):
        m = fit(model, X[tr], V[tr], seed)
        sel = te[target_mask[te]]
        if len(sel):
            p[sel] = proba(m, X[sel])
    t = target_mask
    acc = p[t] >= 0.5
    return float(np.sum(V[t][acc])), p


def best_rule(df: pd.DataFrame, feats, n_q: int = 200):
    """The single-feature threshold rule with the largest realised gain on `df` (non-test rows).

    Accept iff sign * x >= t; a missing value is accepted (falls back to the full decision).
    """
    V = df.V.to_numpy(float)
    best = ("accept_all", 1, -np.inf, float(V.sum()))
    for f in feats:
        x = df[f].to_numpy(float)
        fin = np.isfinite(x)
        if not fin.any():
            continue
        cand = np.unique(np.quantile(x[fin], np.linspace(0, 1, n_q + 1)))
        cand = np.r_[cand, np.inf]
        for sign in (1, -1):
            for t in cand:
                with np.errstate(invalid="ignore"):
                    acc = ~fin | ((x >= t) if sign == 1 else (x <= t))
                g = float(V[acc].sum())
                if g > best[3] + EPS:
                    best = (f, sign, float(t), g)
    return best


def apply_rule(df: pd.DataFrame, rule) -> np.ndarray:
    f, sign, t, _ = rule
    if f == "accept_all":
        return np.ones(len(df), bool)
    x = df[f].to_numpy(float)
    fin = np.isfinite(x)
    with np.errstate(invalid="ignore"):
        return ~fin | ((x >= t) if sign == 1 else (x <= t))
