#!/usr/bin/env python3
"""Builds fig_method.drawio and fig_problem.drawio (plain mxGraph XML, all coordinates on a grid).
Edit the .drawio files in draw.io for manual changes; re-running this script overwrites them.
Export for the paper:  drawio -x -f pdf --crop -o ../paper/figs/<name>.pdf <name>.drawio
"""
import base64, html, os
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "Helvetica"
INK = "#262626"

class Diagram:
    def __init__(self, name):
        self.name = name; self.cells = []; self.n = 1
    def _id(self):
        self.n += 1; return f"c{self.n}"
    def vertex(self, x, y, w, h, style, value=""):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{html.escape(value, quote=True)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i
    def edge(self, src, dst, style, points=None, value=""):
        i = self._id()
        pts = ""
        if points:
            pts = '<Array as="points">' + "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points) + "</Array>"
        self.cells.append(f'<mxCell id="{i}" value="{html.escape(value, quote=True)}" style="{style}" edge="1" parent="1" source="{src}" target="{dst}">'
                          f'<mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')
        return i
    def save(self, w, h):
        xml = (f'<mxfile host="app.diagrams.net"><diagram name="{self.name}" id="{self.name}">'
               f'<mxGraphModel dx="{w}" dy="{h}" grid="1" gridSize="2" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
               f'page="0" pageScale="1" pageWidth="{w}" pageHeight="{h}" math="0" shadow="0">'
               '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(self.cells) + "</root></mxGraphModel></diagram></mxfile>")
        open(os.path.join(HERE, self.name + ".drawio"), "w").write(xml)

def text_style(size, bold=False, color=INK, align="center", valign="middle"):
    return (f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign={valign};whiteSpace=wrap;"
            f"fontFamily={FONT};fontSize={size};fontColor={color};fontStyle={1 if bold else 0};spacing=0;")
def image_style(path):
    ext = "jpeg" if path.endswith(".jpg") else "png"
    data = base64.b64encode(open(os.path.join(HERE, path), "rb").read()).decode()
    # draw.io style strings use ';' as separator, so the data URI must not contain ';base64'
    return f"shape=image;imageAspect=0;aspect=fixed;imageBorder={INK};strokeWidth=1;image=data:image/{ext},{data};"

# ------------------------------------------------------------------ method figure
def method():
    d = Diagram("fig_method")
    YC = 120                       # centre line of the pipeline
    TW, TH = 176, 99               # image thumbnails (16:9), larger than before for legibility
    BW, BH = 82, 98                # bird's-eye-view grid (aspect of bev.png)
    NW, NH = 50, 70                # detector blocks
    PW, PH = 58, 34                # planner
    AW, AH = 96, 30                # action
    G = 14                         # gap between elements
    DET = "#DAE3F3;strokeColor=#2F5597"; PLAN = "#FFF2CC;strokeColor=#BF9000"
    def det_style():   return (f"shape=mxgraph.basic.polygon;polyCoords=[[0,0],[1,0.22],[1,0.78],[0,1]];polyline=0;html=1;whiteSpace=wrap;"
                               f"fillColor={DET};strokeWidth=1.2;fontFamily={FONT};fontSize=13;fontColor={INK};spacing=0;")
    def plan_style():  return (f"rounded=1;arcSize=18;html=1;whiteSpace=wrap;fillColor={PLAN};strokeWidth=1.2;"
                               f"fontFamily={FONT};fontSize=13;fontColor={INK};")
    arrow = f"endArrow=block;endFill=1;endSize=5;html=1;strokeColor={INK};strokeWidth=1.3;rounded=0;exitX=1;exitY=0.5;entryX=0;entryY=0.5;"
    x = 10; ids = {}; xs = {}
    def place(key, w, h, style, value=""):
        nonlocal x
        ids[key] = d.vertex(x, YC - h / 2, w, h, style, value); xs[key] = (x, x + w); x += w + G
    place("frame", TW, TH, image_style("assets/frame.jpg"))
    place("cheap", NW, NH, det_style(), "cheap<br>detector")
    place("plan1", PW, PH, plan_style(), "planner")
    place("bev", BW, BH, image_style("assets/bev.png"))
    place("crops", TW, TH, image_style("assets/crops.jpg"))
    place("target", NW, NH, det_style(), "target<br>detector")
    place("merged", TW, TH, image_style("assets/merged.jpg"))
    place("plan2", PW, PH, plan_style(), "planner")
    place("action", AW, AH, f"rounded=1;arcSize=50;html=1;whiteSpace=wrap;fillColor=#E2F0D9;strokeColor=#2E7D32;strokeWidth=1.4;fontFamily={FONT};fontSize=12;fontStyle=1;fontColor=#1B5E20;", "HARD-BRAKE")
    # re-draw the boxes that are baked into crops.jpg / merged.jpg with print line widths (same pixel coordinates)
    sc = TW / 640.0
    def box(key, bx0, by0, bx1, by1, color, sw=1.6):
        x0 = xs[key][0] + bx0 * sc; y0 = YC - TH / 2 + by0 * sc
        d.vertex(x0, y0, (bx1 - bx0) * sc, (by1 - by0) * sc, f"rounded=0;html=1;fillColor=none;strokeColor={color};strokeWidth={sw};")
    box("crops", 178, 40, 430, 359, "#9E9E9E")       # guard crop (enlarged around a truncated detection)
    box("crops", 155, 183, 253, 284, "#9E9E9E")      # guard crop
    box("crops", 137, 233, 527, 359, "#ED7D31", 1.9)  # footprint of the field
    box("merged", 241, 227, 370, 332, "#E00000", 1.9) # verified detection of the lead vehicle
    W = x - G + 10
    order = ["frame", "cheap", "plan1", "bev", "crops", "target", "merged", "plan2", "action"]
    for a, b in zip(order, order[1:]):
        d.edge(ids[a], ids[b], arrow)
    # what each step produces, directly under it (one baseline)
    LY = YC + BH / 2 + 4
    labels = [("frame", "night frame", "#404040"),
              ("cheap", "no<br>detection", "#C00000"),

              ("bev", "field (orange)", "#C55A11"),
              ("crops", "field crop (orange), guard crops (grey)", "#404040"),
              ("merged", "vehicle found (red; green: annotation)", "#404040"),
]
    for key, lab, col in labels:
        x0, x1 = xs[key]; c = (x0 + x1) / 2; w = max(x1 - x0 + G, 74)
        d.vertex(c - w / 2, LY, w, 34, text_style(14, color=col, valign="top"), lab)
    # planner outputs, above the planners (the feedback loop runs below them)
    for key, lab in [("plan1", "KEEP,<br>support &#8709;"), ("plan2", "action changes")]:
        x0, x1 = xs[key]; c = (x0 + x1) / 2
        d.vertex(c - 46, YC - PH / 2 - 36, 92, 32, text_style(14, color="#404040", valign="bottom"), lab)
    # stage boxes (dashed) with titles
    top, bot = YC - BH / 2 - 10, LY + 34 + 4
    stage = "rounded=1;arcSize=4;html=1;fillColor=none;strokeColor=#7F7F7F;strokeWidth=1;dashed=1;dashPattern=4 3;"
    for a, b, name in [("frame", "cheap", "draft"), ("plan1", "bev", "planner query"), ("crops", "target", "verification"), ("merged", "action", "decision")]:
        x0, x1 = xs[a][0] - 6, xs[b][1] + 6
        d.vertex(x0, top, x1 - x0, bot - top, stage)
        d.vertex(x0, top - 22, x1 - x0, 18, text_style(15, bold=True), name)
    # feedback: the loop is repeated until the action and the crops are unchanged
    fy = bot + 16
    c1 = (xs["plan1"][0] + xs["plan1"][1]) / 2; c2 = (xs["plan2"][0] + xs["plan2"][1]) / 2
    d.edge(ids["plan2"], ids["plan1"],
           f"endArrow=block;endFill=1;endSize=6;html=1;strokeColor={INK};strokeWidth=1.6;rounded=0;dashed=1;dashPattern=5 3;"
           "edgeStyle=orthogonalEdgeStyle;exitX=0.5;exitY=1;entryX=0.5;entryY=1;", points=[(c2, fy), (c1, fy)])
    d.vertex((c1 + c2) / 2 - 300, fy + 3, 600, 16, text_style(15, color="#262626"), "repeat with the merged scene until the action and the crops are unchanged")
    d.save(int(W), int(fy + 22))

# ------------------------------------------------------------------ problem figure
def problem():
    d = Diagram("fig_problem")
    PW, PH, GAP = 196, 248, 20
    HI = "fillColor=#FBE0C8;strokeColor=#ED7D31"        # processed at high resolution
    SUP = "fillColor=#DAE3F3;strokeColor=#2F5597"       # support of the action
    OBJ = "fillColor=#BFBFBF;strokeColor=#595959"       # detected by the low-resolution pass
    def rect(x, y, w, h, style, value="", r=1, sw=1):
        return d.vertex(x, y, w, h, f"rounded={r};arcSize=8;html=1;whiteSpace=wrap;{style};strokeWidth={sw};fontFamily={FONT};fontSize=15;fontStyle=1;fontColor=#C00000;", value)
    titles = ["(a) whole frame", "(b) image-driven", "(c) planner-driven<br>(ours)"]
    for k in range(3):
        x0 = k * (PW + GAP)
        whole = (k == 0)
        rect(x0, 0, PW, PH, (HI if whole else "fillColor=#F7F7F7;strokeColor=#7F7F7F"), sw=1.2 if whole else 1)
        rect(x0 + 50, 1, 96, PH - 2, ("fillColor=#F6CFA9;strokeColor=none" if whole else "fillColor=#E3E3E3;strokeColor=none"), r=0)
        d.edge_free = None
        rect(x0 + 97, 6, 2, PH - 12, "fillColor=#FFFFFF;strokeColor=none", r=0)
        if k == 1:
            for bx, by, bw, bh in [(100, 58, 46, 62), (54, 108, 42, 58), (152, 84, 40, 58), (8, 56, 34, 34), (16, 150, 34, 34)]:
                rect(x0 + bx, by, bw, bh, HI, sw=1.2)
        if k == 2:
            rect(x0 + 100, 14, 46, 172, HI, sw=1.2)
            rect(x0 + 104, 60, 38, 58, SUP, sw=1.2)
        rect(x0 + 111, 70, 24, 38, OBJ)        # vehicle ahead
        rect(x0 + 63, 118, 24, 38, OBJ)        # oncoming vehicle
        rect(x0 + 160, 94, 24, 38, OBJ)        # parked vehicle
        rect(x0 + 18, 66, 14, 14, OBJ)         # pedestrians
        rect(x0 + 26, 160, 14, 14, OBJ)
        rect(x0 + 111, 134, 24, 24, "fillColor=#FFFFFF;strokeColor=#C00000", "?", sw=1.4)   # missed by the low-resolution pass
        rect(x0 + 111, 200, 24, 40, "fillColor=#262626;strokeColor=#000000")              # ego vehicle
        d.vertex(x0 - 10, PH + 6, PW + 20, 46, text_style(19, bold=True, valign="top"), titles[k])
    # legend
    y = PH + 62; x = 66
    for style, lab, w, val in [("fillColor=#262626;strokeColor=#000000", "ego", 34, ""), (OBJ, "detected", 76, ""),
                               ("fillColor=#FFFFFF;strokeColor=#C00000", "missed", 62, "?"), (HI, "high res.", 76, ""), (SUP, "support", 66, "")]:
        d.vertex(x, y, 16, 16, f"rounded=1;arcSize=8;html=1;{style};strokeWidth=1.2;fontFamily={FONT};fontSize=11;fontStyle=1;fontColor=#C00000;", val)
        d.vertex(x + 21, y - 2, w, 20, text_style(17, color="#404040", align="left"), lab)
        x += 21 + w + 14
    d.save(3 * PW + 2 * GAP, y + 22)

if __name__ == "__main__":
    method(); print("written")   # Fig. 1 is now drawn in TikZ: fig_problem_tikz.tex
