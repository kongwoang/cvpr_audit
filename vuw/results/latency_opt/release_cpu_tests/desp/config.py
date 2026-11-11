"""Configuration loading and the start-up integrity checks (frozen configuration and weight md5s)."""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import yaml

# md5 of the released configuration files (the frozen values of the paper)
FROZEN_CONFIGS = {
    "v6s.yaml": "73fddac4a5c1a9021da7d41c0cc0e71a",
    "v6s_bounded.yaml": "e50fa8e51e5a8f2fea1c546be96d63f8",
    "wholeframe.yaml": "0b34075f027afcef6e835336645a39c4",
}
# md5 of the weights the configurations point to
WEIGHTS = {
    "yolov8s.pt": "0a1d5d0619d5a24848a267ec1e3a3118",
    "looming_c1.pt": "7bfdecec7d291a5dfe05de6426c3084e",
    "refiner_v3a.pt": "6b29d05e8f06d121ec9e11aeb9dc2730",
}


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path, root=None, strict: bool = True) -> dict:
    """Load a YAML configuration; check its md5 against the frozen one and the md5 of every weight file it uses.
    Relative weight paths resolve against `root` (default: the repository root). strict=False only warns."""
    path = Path(path)
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(path.read_text())
    problems = []
    frozen = FROZEN_CONFIGS.get(path.name)
    if frozen is not None and md5(path) != frozen:
        problems.append(f"{path.name} differs from the frozen configuration (md5 {md5(path)} != {frozen})")
    for key in ("detector", "looming", "refiner"):
        if key not in cfg:
            continue
        rel = cfg[key]["weights" if key == "detector" else "checkpoint"]
        w = (root / rel) if not Path(rel).is_absolute() else Path(rel)
        if key == "detector":
            cfg[key]["weights"] = str(w)
        else:
            cfg[key]["checkpoint"] = str(w)
        if not w.exists():
            problems.append(f"missing weights {w} (run scripts/download_weights.sh)")
        elif w.name in WEIGHTS and md5(w) != WEIGHTS[w.name]:
            problems.append(f"{w.name}: md5 {md5(w)} != {WEIGHTS[w.name]}")
    for p in problems:
        if strict:
            raise SystemExit("configuration check failed: " + p)
        warnings.warn(p)
    return cfg
