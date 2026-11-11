#!/usr/bin/env python
"""K13 handoff: masked-vs-unmasked CSVs per item from the VUW outputs pulled into the scratch folder (argv[1]).
Unmasked = results/... ; ego mask = k10_masked/results/... (k10_masked_run.py, rows >= 1830, containment >= 0.5)."""
import sys
from pathlib import Path

import pandas as pd

S = Path(sys.argv[1]); O = Path("docs/paper/handoff/K13")
LAB = "post-hoc, ego mask"


def both(name, out, key=None, filt=None):
    rows = []
    for tag, sub in (("ego mask", "m"), ("unmasked", "u")):
        d = pd.read_csv(S / sub / f"{name}.csv")
        if filt is not None:
            d = filt(d)
        d.insert(0, "mask", tag)
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["label"] = [LAB if s == "ego mask" else "post-hoc, unmasked (pre-K10)" for s in d["mask"]]
    d.to_csv(O / out, index=False)
    return d


both("ablation", "a_ablation_av2.csv")
a = pd.read_csv(S / "ablation_ahd.csv"); a["label"] = "post-hoc; Argoverse-HD has no visible ego part (no mask needed)"
a.to_csv(O / "a_ablation_ahd.csv", index=False)
both("agreement", "b_agreement_av2.csv")
both("disagreement_split", "b_disagreement_av2.csv")
for c in ("ours_noref_vs_u960", "ours_ref_vs_u960", "ours_noref_vs_native", "ours_ref_vs_native"):
    for tag, sub in (("masked", "m"), ("unmasked", "u")):
        (O / f"b_confusion_av2_{c}_{tag}.csv").write_text((S / sub / f"confusion_{c}.csv").read_text())
lm = pd.read_csv(S / "loc_masked.csv"); lm.insert(0, "setting", "ego mask")
lu = pd.read_csv("docs/paper/handoff/K12/locality_av2.csv"); lu.insert(0, "setting", "unmasked (K12)")
pd.concat([lm, lu]).to_csv(O / "b_locality_av2.csv", index=False)
fm = pd.read_csv(S / "ft_masked.csv"); fu = pd.read_csv(S / "ft_unmasked.csv")
f = fm.merge(fu[["method", "G", "ours-ft minus", "ours-ft + ref minus"]], on="method", suffixes=("_masked", "_unmasked"))
f["label"] = LAB + "; AV2 val 150 logs, 11,818 frames"
f.to_csv(O / "c_ft_detectors_av2.csv", index=False)
both("missed_natural", "d_missed_natural_av2.csv")
both("missed_controlled", "d_missed_controlled_av2.csv")
both("consumer2", "d_k9_consumer2_av2.csv")
k = pd.read_csv(S / "k9_pairs.csv"); k["label"] = [LAB if s == "ego mask" else "post-hoc, unmasked" for s in k.setting]
k.to_csv(O / "d_k9_equal_px_av2.csv", index=False)
p = pd.read_csv(S / "px_masked.csv"); p["label"] = [LAB if s == "ego mask" else "post-hoc, unmasked" for s in p.setting]
p.to_csv(O / "f_px_av2.csv", index=False)
e = pd.read_csv(O / "e_jetson_av2_masked.csv")
e["label"] = [l if s.startswith("masked") else l.replace("post-hoc, ego mask", "post-hoc, unmasked") for s, l in zip(e.setting, e.label)]
e.to_csv(O / "e_jetson_av2_masked.csv", index=False)
print(sorted(x.name for x in O.iterdir()))
