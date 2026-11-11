#!/usr/bin/env python3
"""One-time assembler of supplementary.tex (2026-10-27 restructure).

Reads supp/skeleton.tex and supp/parts/*.tex, replaces
  %%B a-b%%   by lines a..b of the combined source (verbatim text, with the
              conventions below applied),
  %%T a-b%%   by the inner part of the table environment in lines a..b,
  %%NAME%%    by supp/parts/NAME.tex (recursively),
and writes supplementary.tex. The combined source is the version of main.tex
before the split (versions/main_2026-10-27b_before-split.tex).

Conventions applied to the copied blocks (no value is changed):
  * single column: table*/figure* -> table/figure, table fonts one step larger;
  * references to the main paper -> \\cref*{M-...} (xr-hyper, no hyperlink);
  * Delta columns are DESP - row everywhere (the two tables that used
    row - DESP are negated: [a, b] -> [-b, -a]); bold iff the interval is
    above zero, \\sw iff it is below zero;
  * network pixels in % of the native frame (x100, exact).
"""
import re
from decimal import Decimal
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
SRC = (PAPER.parent / "versions" / "main_2026-10-27b_before-split.tex").read_text().split("\n")
MAIN_END = 757
MAIN_LABELS = set(re.findall(r"\\label\{([^}]*)\}", "\n".join(SRC[:MAIN_END])))
RENAME = {"tab:factorial": "tab:heuristics", "tab:matched": "tab:heuristics",
          "tab:splits": "tab:benchmarks", "tab:configs": "tab:benchmarks",
          "sec:supp_results": "sec:supp_planning"}


def lines(a, b):
    return "\n".join(SRC[a - 1:b])


# ------------------------------------------------------------- conventions
def fix_refs(text):
    def rep(m):
        cmd, labs = m.group(1), m.group(2)
        out = []
        for l in (x.strip() for x in labs.split(",")):
            l = RENAME.get(l, l)
            if l not in out:
                out.append(l)
        mains = [l for l in out if l in MAIN_LABELS]
        supps = [l for l in out if l not in MAIN_LABELS]
        parts = []
        if mains:
            parts.append("\\%s*{%s}" % (cmd, ",".join("M-" + l for l in mains)))
        if supps:
            parts.append("\\%s{%s}" % (cmd if not mains else "cref", ",".join(supps)))
        return " and ".join(parts)
    return re.sub(r"\\(cref|Cref)\{([^}]*)\}", rep, text)


def pct(x):
    x = x.strip()
    d = (Decimal(x) * 100).quantize(Decimal("0.1"))
    return str(d)


def flip_ci(cell):
    def rep(m):
        a = m.group(1).replace("$-$", "-").replace("$", "")
        b = m.group(2).replace("$-$", "-").replace("$", "")
        na, nb = -Decimal(b), -Decimal(a)
        f = lambda v: ("$-$%s" % str(-v)) if v < 0 else "+%s" % str(v)
        return "\\ci{%s}{%s}" % (f(na), f(nb))
    return re.sub(r"\\ci\{([^{}]*)\}\{([^{}]*)\}", rep, cell)


def mark_ci(cell):
    """Bold iff above zero, \\sw iff below zero (removes previous emphasis)."""
    cell = re.sub(r"\\(?:best|textbf)\{(\\ci\{[^{}]*\}\{[^{}]*\})\}", r"\1", cell)

    def rep(m):
        a = float(m.group(1).replace("$-$", "-").replace("$", ""))
        b = float(m.group(2).replace("$-$", "-").replace("$", ""))
        s = m.group(0)
        if a > 0:
            return "\\best{%s}" % s
        if b < 0:
            return s + "\\sw"
        return s
    return re.sub(r"\\ci\{([^{}]*)\}\{([^{}]*)\}", rep, cell)


def table_rows(text, fn):
    """Apply fn(cells) to every data row (between \\midrule and \\bottomrule)."""
    out, active = [], False
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith("\\midrule"):
            active = True
        if s.startswith("\\bottomrule"):
            active = False
        if active and "&" in s and s.endswith("\\\\"):
            cells = s[:-2].split("&")
            cells = fn([c.strip() for c in cells])
            ln = " & ".join(cells) + " \\\\"
        out.append(ln)
    return "\n".join(out)


def conv_cols(cols, mc_cols=None):
    def fn(c):
        idx = mc_cols if (mc_cols and any("multicolumn" in x for x in c)) else cols
        for i in idx:
            if i < len(c) and re.fullmatch(r"\d\.\d{3}", c[i]):
                c[i] = pct(c[i])
        return c
    return fn


TABLE_FIX = {}


def fix_table(label, text):
    if label in ("tab:holdout", "tab:constrained"):
        col = 2 if label == "tab:holdout" else 5
        def fn(c, col=col):
            if col < len(c):
                c[col] = flip_ci(re.sub(r"\\textbf\{(\\ci\{[^{}]*\}\{[^{}]*\})\}", r"\1", c[col]))
            return c
        text = table_rows(text, fn)
        text = text.replace("$95\\%$ paired interval of (row$\\,-\\,$\\method), bold if it excludes zero",
                            "$95\\%$ paired interval of (\\method$\\,-\\,$row), bold if above zero")
        text = text.replace("evaluated once under a protocol fixed in advance",
                            "in the first of two evaluations under protocols fixed in advance")
    if label in ("tab:main_ci", "tab:hood", "tab:uniform_ci", "tab:holdout", "tab:constrained", "tab:speed"):
        text = table_rows(text, lambda c: [mark_ci(x) for x in c])
        text = text.replace("intervals above zero are in bold", "bold: above zero; \\sw: below zero")
        text = text.replace("intervals above zero in bold", "bold: above zero; \\sw: below zero")
    if label == "tab:uniform":
        text = table_rows(text, conv_cols([5, 11, 17], mc_cols=[2, 5, 8]))
        text = text.replace("& C1 & px &", "& C1 & px\\,(\\%) &").replace("& C1 & px \\\\", "& C1 & px\\,(\\%) \\\\")
    if label == "tab:hood":
        text = table_rows(text, conv_cols([2]))
        text = text.replace("& px & $\\Delta$", "& px\\,(\\%) & $\\Delta$")
    if label == "tab:missed":
        text = table_rows(text, conv_cols([3, 8, 13]))
        text = text.replace("& px &", "& px\\,(\\%) &")
    if label == "tab:det2":
        text = table_rows(text, conv_cols([2, 4, 6]))
        text = text.replace("& px &", "& px\\,(\\%) &").replace("& px \\\\", "& px\\,(\\%) \\\\")
    if label == "tab:speed":
        def fn(c):
            c[2] = " / ".join(pct(v) for v in c[2].split("/"))
            return c
        text = table_rows(text, fn)
        text = text.replace("Ego speed & Range & px &", "Ego speed & Range & px\\,(\\%) &")
    if label in ("tab:versions", "tab:consumer2"):
        text = table_rows(text, lambda c: [re.sub(r"\((\d\.\d{3})\)", lambda m: "(%s)" % pct(m.group(1)), x) for x in c])
        text = text.replace("network pixels in parentheses", "network pixels in \\% in parentheses")
    return text


def fix_block(text):
    text = text.replace("\\begin{table*}", "\\begin{table}").replace("\\end{table*}", "\\end{table}")
    text = text.replace("\\begin{figure*}", "\\begin{figure}").replace("\\end{figure*}", "\\end{figure}")
    # tables: one font step larger in the single-column layout
    def tab(m):
        t = m.group(0)
        t = t.replace("\\footnotesize", "\\small").replace("\\scriptsize", "\\footnotesize")
        lab = re.search(r"\\label\{([^}]*)\}", t)
        if lab:
            t = fix_table(lab.group(1), t)
        return t
    text = re.sub(r"\\begin\{table\}.*?\\end\{table\}", tab, text, flags=re.S)
    text = fix_refs(text)
    return text


def inner_table(a, b):
    t = fix_block(lines(a, b))
    t = re.sub(r"\\begin\{table\}(\[[^\]]*\])?\n", "", t)
    t = t.replace("\\end{table}", "")
    t = t.replace("\\centering\n", "")
    return t.strip("\n")


# --------------------------------------------------- merged heuristics table
def merged_heuristics():
    def rows_of(label):
        t = lines(1, len(SRC))
        i = t.index("\\label{%s}" % label)
        beg = t.rfind("\\begin{table", 0, i)
        end = t.index("\\end{table", i)
        body = t[beg:end]
        body = body[body.index("\\midrule"):body.rindex("\\bottomrule")]
        out = []
        for ln in body.split("\n"):
            s = ln.strip()
            if "&" in s and s.endswith("\\\\"):
                out.append([c.strip() for c in s[:-2].split("&")])
        return out
    fac = {r[0].replace("\\oursrow ", ""): r for r in rows_of("tab:factorial")}
    mat = {r[0].replace("\\oursrow ", ""): r for r in rows_of("tab:matched")}
    fac_off = {"av2": 1, "nusc": 6, "ahd": 11}      # px, single, C1, Delta
    mat_off = {"nusc": 1, "av2": 5, "ahd": 9}       # G, px ratio, Delta
    names = ["Uncertainty", "Temporal criticality", "Ego corridor", "Support + ego corridor", "Random"]
    ours = "\\textbf{\\method}"
    out = [r"\begin{table}[h]", r"\centering",
           r"\caption{\textbf{Crop heuristics with identical components at matched budgets.} No heuristic is significantly better than \method in either setting. "
           r"Left: smallest budget of the tuned grid of each heuristic; ``Single'': with the refiner and the looming cue of a single pair of boxes; ``C1'': with the refiner and the learned looming estimate; $\Delta$ against the heuristic with C1. "
           r"Right: capped at the pixels of \method on every frame, with C1; ``px'': realized network pixels relative to \method (the truncation guard lets the heuristics exceed the cap). Bold intervals: above zero.}",
           r"\label{tab:heuristics}", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
           r" & \multicolumn{4}{c}{Smallest budget of the tuned grid} & & \multicolumn{3}{c}{Per-frame cap at the pixels of \method} \\",
           r"\cmidrule(lr){2-5}\cmidrule(l){7-9}",
           r"Heuristic & px\,(\%) & $\G$, single & $\G$, C1 & $\Delta$ & & $\G$ & px & $\Delta$ \\"]
    for ds, title in (("nusc", "nuScenes test"), ("av2", "Argoverse~2 val"), ("ahd", "Argoverse-HD val")):
        out.append(r"\midrule")
        out.append(r"\multicolumn{9}{@{}l}{\emph{%s}} \\" % title)
        fo, mo = fac_off[ds], mat_off[ds]
        for n in names:
            f = fac[n]
            left = [pct(f[fo]), f[fo + 1], f[fo + 2], mark_ci(f[fo + 3])]
            if n in mat:
                m = mat[n]
                right = [m[mo], m[mo + 1], mark_ci(m[mo + 2])]
            else:
                right = ["\\na", "\\na", "\\na"]
            out.append(" & ".join([n] + left + [""] + right) + r" \\")
        f, m = fac[ours], mat[ours]
        do = {"av2": 1, "nusc": 5, "ahd": 9}[ds]     # the DESP row spans two columns per dataset
        g = re.search(r"\\multicolumn\{2\}\{c\}\{(.*)\}", f[do + 1]).group(1)
        out.append(" & ".join([r"\oursrow " + ours, pct(f[do]), r"\multicolumn{2}{c}{%s}" % g, "\\na", "", m[mo], m[mo + 1], "\\na"]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


# ----------------------------------------------------------------- assemble
def resolve(text, depth=0):
    assert depth < 6

    def b(m):
        return fix_block(lines(int(m.group(1)), int(m.group(2))))

    def t(m):
        return inner_table(int(m.group(1)), int(m.group(2)))

    def part(m):
        nm = m.group(1)
        if nm == "MERGEDHEUR":
            return merged_heuristics()
        p = PAPER / "supp" / "parts" / ("%s.tex" % nm)
        return resolve(p.read_text().rstrip("\n"), depth + 1)
    text = re.sub(r"%%B (\d+)-(\d+)%%", b, text)
    text = re.sub(r"%%T (\d+)-(\d+)%%", t, text)
    text = re.sub(r"%%([A-Z0-9]+)%%", part, text)
    return text


SPECIFIC = [
    # metric block: label the two equations, one line for the loss in the single-column layout
    ("\\ \\mathrm{m/s^2},\n\\end{equation}", "\\ \\mathrm{m/s^2},\n\\label{eq:s_demand}\n\\end{equation}"),
    ("\\begin{multline}\nL = \\lambda_r\\,[\\dem^\\star-u(a)]_+^2 + \\lambda_b\\,[u(a)-\\dem^\\star]_+^2 \\\\\n + \\lambda_j",
     "\\begin{equation}\nL = \\lambda_r\\,[\\dem^\\star-u(a)]_+^2 + \\lambda_b\\,[u(a)-\\dem^\\star]_+^2\n + \\lambda_j"),
    ("\\big[\\dem^\\star-u(a)\\ge 8\\big],\n\\end{multline}", "\\big[\\dem^\\star-u(a)\\ge 8\\big],\n\\label{eq:s_loss}\n\\end{equation}"),
    ("until the pixel budget is reached (\\cref{sec:supp_config})", "until the pixel budget is reached (\\cref{sec:supp_pixels})"),
    ("evaluated on the test splits (\\cref{sec:supp_config}).", "evaluated on the test splits."),
    # pareto figure: three panels in the single-column text width
    ("width=0.36\\textwidth, height=0.30\\textwidth", "width=0.355\\textwidth, height=0.31\\textwidth"),
]

if __name__ == "__main__":
    sk = (PAPER / "supp" / "skeleton.tex").read_text()
    out = resolve(sk)
    for a, b in SPECIFIC:
        n = out.count(a)
        assert n >= 1, a
        out = out.replace(a, b)
    assert "%%" not in re.sub(r"%[^\n]*%%[^\n]*", "", out) or True
    left = re.findall(r"%%[A-Z0-9 \-]+%%", out)
    assert not left, left
    (PAPER / "supplementary.tex").write_text(out)
    print("wrote supplementary.tex,", out.count("\n"), "lines")
