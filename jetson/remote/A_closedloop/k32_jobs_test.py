#!/usr/bin/env python
"""K32 test job list (only after the dev gate passed): results/k32/test_routes.xml split into 8 chunks (round robin), and
results/k32/jobs_test.txt with one line per (seed, chunk, branch, setting): privileged L0; cheap / u960 / native / hybrid /
v6S / v6Sb2 in L0 and L1; seeds 0, 1, 2 (seed-major order so that partial results stay balanced). AEB_C1 / AEB_REF from
results/k32/tables/dev_gate.csv (the dev rule). jobs_test_vuw.txt / jobs_test_vuw1.txt: disjoint halves, one dispatcher per
host (each dispatcher only sees its own host's processes).
usage: k32_jobs_test.py
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

B = Path(os.environ.get("B", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
K = B / "results" / "k32"
NCH = 8
FRONT = 2.45                                              # amendment B (handoff/K32/STATUS.md); C: AEB_TTC_TICKS=10


def main():
    g = pd.read_csv(K / "tables" / os.environ.get("K32_GATE", "dev3_gate.csv")).iloc[0]   # amendment B: the dev3 gate
    assert bool(g["GATE_PASS"]), "dev gate did not pass: no test runs"
    c1, ref = int(bool(g["c1_on"])), int(bool(g["refiner_on"]))
    r = ET.parse(K / "test_routes.xml").getroot().findall("route")
    for k in range(NCH):
        out = ET.Element("routes")
        for x in r[k::NCH]:
            out.append(x)
        ET.ElementTree(out).write(K / f"test_routes_c{k}.xml", encoding="utf-8", xml_declaration=True)
    lines = []
    for seed in (0, 1, 2):
        for k in range(NCH):
            rt = K / f"test_routes_c{k}.xml"
            lines.append(f"test/privileged_L0_s{seed}_c{k}|{rt}|AEB_BRANCH=privileged AEB_LAT=L0 AEB_SEED={seed} AEB_C1={c1} AEB_REF={ref} AEB_FRONT={FRONT} AEB_TTC_TICKS=10")
            for br in ("v6S", "v6Sb2", "cheap", "u960", "native", "hybrid"):
                for st in ("L0", "L1"):
                    lines.append(f"test/{br}_{st}_s{seed}_c{k}|{rt}|AEB_BRANCH={br} AEB_LAT={st} AEB_SEED={seed} AEB_C1={c1} AEB_REF={ref} AEB_FRONT={FRONT} AEB_TTC_TICKS=10")
    (K / "jobs_test.txt").write_text("\n".join(lines) + "\n")
    # one dispatcher per host on disjoint halves (alternate lines keep both hosts on every branch / seed)
    (K / "jobs_test_vuw.txt").write_text("\n".join(lines[0::2]) + "\n")
    (K / "jobs_test_vuw1.txt").write_text("\n".join(lines[1::2]) + "\n")
    print(len(r), "routes,", len(lines), "jobs; C1", c1, "refiner", ref)


if __name__ == "__main__":
    main()
