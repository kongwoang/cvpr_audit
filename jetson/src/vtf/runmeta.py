"""Run directories with provenance: resolved config, environment, and the commit of the DEEP code read from."""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAP_REPO = Path(os.environ.get("RAP_REPO", ROOT.parent / "risk-aware-perception")).resolve()
RAW = ROOT / "results" / "raw"
FINAL = ROOT / "results" / "final"


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def new_run(tag: str, config: dict) -> Path:
    run = RAW / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{tag}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(config, indent=2, default=str))
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "rap_repo": str(RAP_REPO),
        "rap_commit": _git(RAP_REPO, "rev-parse", "HEAD"),
        "rap_dirty_src": bool(_git(RAP_REPO, "status", "--porcelain", "src")),
        "rap_frame": os.environ.get("RAP_FRAME", ""),
    }
    (run / "environment.json").write_text(json.dumps(env, indent=2))
    FINAL.mkdir(parents=True, exist_ok=True)
    print(f"  run: {run}", flush=True)
    return run
