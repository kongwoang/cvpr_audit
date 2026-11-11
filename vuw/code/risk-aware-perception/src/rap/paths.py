"""Canonical paths for code, datasets, models and generated artefacts.

The original experiment ran from one Jetson account and several scripts copied that
account's absolute paths.  Defaults now follow the current user's home directory and
every external location has an environment override.  ``RAP_DATA`` and ``RAP_MODELS``
remain supported for backwards compatibility with the Phase-0 commands.
"""
from __future__ import annotations

from pathlib import Path
import os


def _path(name: str, default: Path | str) -> Path:
    # Do not resolve symlinks: nuPlan's released layout intentionally points
    # ``nuplan-v1.1/splits/mini`` at its cache directory, and preserving the
    # configured spelling keeps run metadata comparable with the frozen runs.
    return Path(os.environ.get(name, default)).expanduser()


REPO = Path(__file__).resolve().parents[2]
USER_DATA_ROOT = _path("RAP_DATA_ROOT", Path.home() / "datasets")

# External data/model roots.  These are read-only inputs to the experiment.
DATA = _path("RAP_KITTI_ROOT", os.environ.get("RAP_DATA", USER_DATA_ROOT / "kitti_tracking"))
MODELS = _path("RAP_MODEL_ROOT", os.environ.get("RAP_MODELS", Path.home() / "models" / "yolo"))
NUSCENES_ROOT = _path("RAP_NUSCENES_ROOT", USER_DATA_ROOT / "nuscenes")
NUSCENES_TRAINVAL = _path("RAP_NUSCENES_TRAINVAL", NUSCENES_ROOT / "trainval")
NUPLAN_ROOT = _path("RAP_NUPLAN_ROOT", USER_DATA_ROOT / "nuplan")
NUPLAN_DB = _path("RAP_NUPLAN_DB", NUPLAN_ROOT / "nuplan-v1.1" / "splits" / "mini")
NUPLAN_MAPS = _path("RAP_NUPLAN_MAPS", NUPLAN_ROOT / "nuplan-maps-v1.0")
NUPLAN_CAM_F0 = _path("RAP_NUPLAN_CAM_F0", NUPLAN_ROOT / "sensor_blobs_cam_f0")

KITTI_IMAGES = DATA / "training" / "image_02"
KITTI_LABELS = DATA / "training" / "label_02"
KITTI_CALIB = DATA / "training" / "calib"
KITTI_OXTS = DATA / "training" / "oxts"

CACHE = REPO / "data" / "cache"
RESULTS = REPO / "results"
RAW = RESULTS / "raw"
PROCESSED = RESULTS / "processed"
FIGURES = RESULTS / "figures"
DOCS = REPO / "docs"

for _p in (CACHE, RAW, PROCESSED, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
