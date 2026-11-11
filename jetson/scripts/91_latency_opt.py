#!/usr/bin/env python3
"""Isolated latency harness entry point; device execution requires RUNPLAN approval."""
from pathlib import Path
import runpy

if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'experiments/latency_opt_20261030/scripts/91_latency_opt.py'),run_name='__main__')
