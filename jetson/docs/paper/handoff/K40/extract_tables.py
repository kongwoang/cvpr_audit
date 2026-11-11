"""K40 (private): the printed body of each tabular with \\label{tab:...} in main.tex / supplementary.tex ->
<release>/tests/paper_tables/<label>.tex (the R0 reference; the paper sources themselves are not released)."""
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parent / "inputs/k40_inputs/paper"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n = 0
for f in ("main.tex", "supplementary.tex"):
    T = (P / f).read_text()
    for m in re.finditer(r"\\label\{(tab:[^}]+)\}", T):
        lab = m.group(1)
        i = m.end()
        b = T.find("\\begin{tabular", i)
        e = T.find("\\end{tabular}", b)
        nxt = T.find("\\label{", i)
        if b < 0 or e < 0 or (0 <= nxt < b):
            continue
        body = T[b:e]
        t = body.find("\\toprule")
        mid = body.find("\\midrule", t)
        bot = body.rfind("\\bottomrule")
        rows = body[mid + len("\\midrule"):bot if bot > 0 else len(body)]
        lines = [re.sub(r"\s+", " ", x).strip() for x in rows.split("\n")]
        lines = [x for x in lines if x and not x.startswith("%")]
        (out / f"{lab.replace(':', '_')}.tex").write_text("\n".join(lines) + "\n")
        n += 1
print(n, "tables extracted")

# figures: the numbers each figure shows (release analysis/figures.figure_data: coordinate lists, numeric node labels,
# interval bars), from the figure's code between \begin{figure} and its label -> <release>/tests/paper_figures/<label>.tex
sys.path.insert(0, str(out.parents[1]))
from analysis.figures import figure_data  # noqa: E402
fo = out.parent / "paper_figures"; fo.mkdir(parents=True, exist_ok=True)
k = 0
for f in ("main.tex", "supplementary.tex"):
    T = (P / f).read_text()
    for m in re.finditer(r"\\label\{(fig:[^}]+)\}", T):
        b = max(T.rfind("\\begin{figure", 0, m.start()), T.rfind("\\end{figure", 0, m.start()))
        body = "\n".join(x for x in T[b:m.start()].split("\n") if not x.strip().startswith("%"))
        lines = figure_data(body)
        if lines:
            (fo / f"{m.group(1).replace(':', '_')}.tex").write_text("\n".join(lines) + "\n")
            k += 1
print(k, "figures extracted")
