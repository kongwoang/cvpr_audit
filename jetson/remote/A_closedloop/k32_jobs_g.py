#!/usr/bin/env python
"""K32 amendment G (handoff/K32/STATUS.md): add the G jobs to the running test queue (jobs_test_vuw.txt / _vuw1.txt):
  G1  test/unc_<L0|L1>_s<seed>_c<k>   uncertainty crops + comp. (48 jobs), appended, alternating between the two files
  G4  shadow/<v6S|u960>_L0_s0_c<k>    rollouts with AEB_LOGALL = 1 (16 jobs), placed first among each file's unstarted lines
      (they take several times longer than a test job, so they start early)
The job variables are those of the running test lines (C1 / refiner from dev3F_gate.csv via those lines). Each file is
rewritten atomically; lines already in a file are kept in order. Idempotent: jobs already present are not added twice.
usage: k32_jobs_g.py [unc|shadow]   (default: both)
"""
import os
import sys
from pathlib import Path

B = Path(os.environ.get("B", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
K = B / "results" / "k32"
NCH = 8


def main():
    files = [K / "jobs_test_vuw.txt", K / "jobs_test_vuw1.txt"]
    cur = [f.read_text().splitlines() for f in files]
    have = {ln.split("|")[0] for c in cur for ln in c if ln.strip()}
    ref = next(ln for c in cur for ln in c if ln.startswith("test/v6S_L0_s0_c0|"))
    tail = ref.split("|")[2].split("AEB_SEED=0 ")[1]                 # AEB_C1=.. AEB_REF=.. AEB_FRONT=.. AEB_TTC_TICKS=..
    unc, sh = [], []
    for seed in (0, 1, 2):
        for k in range(NCH):
            rt = K / f"test_routes_c{k}.xml"
            for st in ("L0", "L1"):
                unc.append(f"test/unc_{st}_s{seed}_c{k}|{rt}|AEB_BRANCH=unc AEB_LAT={st} AEB_SEED={seed} {tail}")
    for k in range(NCH):
        rt = K / f"test_routes_c{k}.xml"
        for br in ("v6S", "u960"):
            sh.append(f"shadow/{br}_L0_s0_c{k}|{rt}|AEB_BRANCH={br} AEB_LAT=L0 AEB_SEED=0 AEB_LOGALL=1 {tail}")
    want = sys.argv[1] if len(sys.argv) > 1 else "both"
    unc = [x for x in unc if x.split("|")[0] not in have] if want in ("unc", "both") else []
    sh = [x for x in sh if x.split("|")[0] not in have] if want in ("shadow", "both") else []

    def started(name):
        return (K / f"{name}.json").exists() or (K / "claims" / name.replace("/", "_")).exists()
    for i, (f, c) in enumerate(zip(files, cur)):
        c = [ln for ln in c if ln.strip()]
        n0 = next((j for j, ln in enumerate(c) if not started(ln.split("|")[0])), len(c))
        new = c[:n0] + sh[i::2] + c[n0:] + unc[i::2]
        tmp = f.with_suffix(".txt.tmp")
        tmp.write_text("\n".join(new) + "\n")
        os.replace(tmp, f)
        print(f.name, "lines", len(c), "->", len(new), "| shadow inserted at", n0, "| unc appended", len(unc[i::2]))


if __name__ == "__main__":
    main()
