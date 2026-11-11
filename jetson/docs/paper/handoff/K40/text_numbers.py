#!/usr/bin/env python
"""K40 (private): text-number pass. Every number quoted in the running text of main.tex / supplementary.tex (outside
tabulars and figure code), with its paragraph, is matched against the rows that the release regenerates (scripts/
dump_r0.py output): first in the tables / figures that the paragraph cites, then in all of them. A text number matches a
regenerated value v if it equals v rounded to the text's number of decimals (sign included).
usage: scripts/py text_numbers.py <dump_dir> <out.csv>"""
import re
import sys
from pathlib import Path

import pandas as pd

P = Path(__file__).resolve().parent / "inputs/k40_inputs/paper"
NUM = re.compile(r"(?<![\w.{])(\$?(?<![-−])[-−+]\$?\s?)?(\d+(?:\{,\}\d{3})*(?:\.\d+)?)(\$?\s*(?:\\,)?\\%)?")
SKIP_ENV = re.compile(r"\\begin\{(tabular\*?|tikzpicture|equation\*?|align\*?)\}.*?\\end\{\1\}", re.S)
CMD = re.compile(r"\\(cite|citet|citep|ref|cref|Cref|crefrange|label|mref|mrefnum|eqref|includegraphics|usepackage|documentclass|"
                 r"begin|end|vspace|hspace|setlength|fontsize|definecolor|newcommand|renewcommand|input|bibliography|"
                 r"bibliographystyle|setcounter|addtolength|resizebox|scalebox|rule|multicolumn|multirow|color|textcolor)\*?"
                 r"(\[[^\]]*\])?(\{[^{}]*\})*")
REFS = re.compile(r"\\(?:cref|Cref|ref|mref)\*?\{([^}]*)\}")


def values(dump):
    pool = {}
    for f in Path(dump).glob("*.tex"):
        vs = []
        for m in re.finditer(r"(\$-\$|-|\+)?(\d+\.\d+|\d+)", f.read_text().replace("$-$", "-")):
            s = (m.group(1) or "").replace("$-$", "-")
            vs.append((("-" if s == "-" else "") + m.group(2)))
        pool[f.stem] = vs
    return pool


def matches(txt, vs, pct=False):
    if pct and "." in txt:
        return matches(txt, vs) or any(matches(x, vs) for x in pct_forms(txt))
    neg = txt.startswith("-")
    t = txt.lstrip("+-")
    d = len(t.split(".")[1]) if "." in t else 0
    tv = float(t) * (-1 if neg else 1)
    for v in vs:
        vd = len(v.split(".")[1]) if "." in v else 0
        if vd >= d and round(float(v), d) == round(tv, d):
            return True
    return False


def pct_forms(txt):
    """a percentage p with d decimals as a share with d + 2 decimals"""
    neg, t = txt.startswith("-"), txt.lstrip("+-")
    d = len(t.split(".")[1]) if "." in t else 0
    return [("-" if neg else "") + f"{float(t) / 100:.{d + 2}f}"]


def main():
    pool = values(sys.argv[1])
    rows = []
    for f in ("main.tex", "supplementary.tex"):
        T = (P / f).read_text()
        T = re.sub(r"(?m)(?<!\\)%.*$", "", T)
        T = SKIP_ENV.sub(" ", T)
        for k, para in enumerate(re.split(r"\n\s*\n", T)):
            cited = ["tab_" + x.split(":", 1)[1] if x.startswith("tab:") else "fig_" + x.split(":", 1)[1]
                     for r in REFS.findall(para) for x in r.replace("M-", "").split(",") if x.startswith(("tab:", "fig:"))]
            body = CMD.sub(" ", para)
            for m in NUM.finditer(body):
                sign = (m.group(1) or "").replace("$", "").replace("−", "-").strip()
                n = m.group(2).replace("{,}", "")
                txt = ("-" if sign == "-" else "") + n
                ctx = re.sub(r"\s+", " ", body[max(0, m.start() - 70):m.end() + 50])
                pct = bool(m.group(3)) or bool(re.match(r"\$?\s*(--|to|and|,)\s*\$?[-+]?\$?[\d.]+\$?\s*(\\,)?\\%", body[m.end():m.end() + 30]))
                hit = [c for c in cited if c in pool and matches(txt, pool[c], pct)]
                anyhit = [] if hit else [c for c, vs in pool.items() if matches(txt, vs, pct)]
                rows.append({"file": f, "para": k, "number": txt + ("%" if pct else ""), "decimals": "." in n, "cited": ";".join(dict.fromkeys(cited)),
                             "match_cited": ";".join(hit), "match_other": ";".join(anyhit[:5]), "context": ctx})
    d = pd.DataFrame(rows)
    d.to_csv(sys.argv[2], index=False)
    dec = d[d.decimals]
    print(len(d), "numbers;", len(dec), "with decimals;", (dec.match_cited != "").sum(), "matched in a cited item;",
          ((dec.match_cited == "") & (dec.match_other != "")).sum(), "matched elsewhere;",
          ((dec.match_cited == "") & (dec.match_other == "")).sum(), "unmatched")


if __name__ == "__main__":
    main()
