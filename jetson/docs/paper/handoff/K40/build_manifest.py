"""K40 Stage 0: reproduction manifest from main.tex / supplementary.tex (labels, the source comment above each float,
the generated supp plots' table of origin)."""
import csv
import re
from pathlib import Path

P = Path(__file__).resolve().parent / "inputs/k40_inputs/paper"
R1 = {"tab:main", "tab:ablation", "tab:field", "tab:uniform", "tab:official", "tab:heuristics", "tab:episodes", "tab:severity",
      "tab:missed", "tab:taxonomy", "tab:locality", "tab:consumer2", "tab:det", "tab:det2", "tab:speed", "tab:curved",
      "tab:strata", "tab:holdout", "tab:constrained", "tab:curves"}
R2 = {"tab:latency", "fig:latency", "fig:deadline", "tab:frontier", "tab:deadline", "tab:latency_baselines", "tab:eff",
      "tab:aeb", "tab:aeb_main", "tab:b2d", "tab:ft", "tab:ftav2"}
DERIVED = {  # items without a source comment: generated plots (make_supp_figs.py reads the table of origin) and others
    "fig:forest": "generated from tab:main_ci (make_supp_figs.forest)",
    "fig:uniform": "generated from tab:uniform (make_supp_figs.uniform)",
    "fig:loss": "generated from tab:loss (make_supp_figs.loss)",
    "fig:locality": "generated from tab:locality (make_supp_figs.locality)",
    "fig:coverage": "generated from tab:coverage (make_supp_figs.coverage)",
    "fig:operating": "generated from K37/K37-1 k1_points.csv and tab:constrained (make_supp_figs.operating)",
    "fig:latency": "generated from tab:latency_baselines, tab:latency, tab:uniform (make_supp_figs.latency)",
    "fig:deadline": "generated from tab:deadline (make_supp_figs.deadline)",
    "fig:pareto": "K28 phase2/table_main_*_v6.csv, best tuned variant per budget, + C1 (comment 85 lines above the label)",
    "tab:heuristics": "K28 phase2/table_main_*_v6.csv (@S + C1, single) and K30/v1a_matched.csv (per-frame cap); as tab:main",
    "tab:uniform_ci": "K28 phase2/table_main_{av2,nusc,ahd}_v6.csv intervals (as tab:uniform)",
    "fig:qual_n": "K35/K35-6 frames by fixed rule; figure_scripts/qual/render_panels.py",
    "fig:qual_a1": "K35/K35-6 frames by fixed rule; figure_scripts/qual/render_panels.py",
    "fig:qual_a2": "K35/K35-6 frames by fixed rule; figure_scripts/qual/render_panels.py",
    "tab:constructions": "descriptive (counterexample constructions; no computed numbers)",
    "tab:practice": "descriptive; quoted numbers traced in the text-number pass",
    "tab:benchmarks": "descriptive; quoted numbers traced in the text-number pass",
    "tab:baselines": "descriptive; quoted fidelity numbers traced in the text-number pass",
    "tab:failures": "descriptive; quoted numbers traced in the text-number pass (refs to measured tables)",
    "tab:roadmap": "descriptive (roadmap of the supplement)",
}
rows = []
for part, f in (("main", "main.tex"), ("supp", "supplementary.tex")):
    L = (P / f).read_text().splitlines()
    for i, line in enumerate(L):
        for lab in re.findall(r"\\label\{((?:tab|fig):[^}]+)\}", line):
            j, src, cap = i, "", ""
            while j > max(0, i - 140):
                j -= 1
                if L[j].lstrip().startswith("% ---"):
                    src = L[j].strip().strip("%- ").strip()
                    k = j + 1
                    while k < i and L[k].lstrip().startswith("%") and not L[k].lstrip().startswith("% ---"):
                        src += " " + L[k].strip().lstrip("% ")
                        k += 1
                    break
                if re.search(r"\\label\{(tab|fig):", L[j]) or re.search(r"\\end\{(table|figure)", L[j]):
                    break
            for k in range(max(0, i - 40), min(len(L), i + 5)):
                m = re.search(r"\\caption\{(.{0,110})", L[k])
                if m:
                    cap = m.group(1)
            files = sorted(set(re.findall(r"[A-Za-z0-9_\-/{},.*]+\.(?:csv|json|md|py)", src)))
            lvl = "R0" + (" + R1" if (part == "main" or lab in R1) else "") + (" + R2" if lab in R2 else "")
            rows.append({"label": lab, "part": part, "line": i + 1, "caption_start": cap, "source_comment": src,
                         "source_files": ";".join(files), "release_script": f"analysis/{lab.replace(':', '_')}.py",
                         "level": lvl, "status": "pending" if src else ("source resolved (no comment): " + DERIVED[lab] if lab in DERIVED else "NO SOURCE COMMENT")})
            if not src and lab in DERIVED:
                rows[-1]["source_comment"] = DERIVED[lab]
with open(Path(__file__).resolve().parent / "manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), "items;", sum(r["status"] == "NO SOURCE COMMENT" for r in rows), "without a source comment")
for r in rows:
    if r["status"] != "pending":
        print("  ", r["part"], r["label"], r["line"], "|", r["caption_start"][:90])
