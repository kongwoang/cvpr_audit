#!/usr/bin/env python3
"""Re-draws the vector overlays of the kit's panel PDFs (k28_render.py output) with print-size line widths.

Input: panel PDFs from handoff/K35/K35-6/panels or handoff/K28/phase3/panels (1600x900 or 1550x2048 pages,
one raster image /I1 = the camera frame, rectangles as stroked paths). Nothing is added or moved: every rectangle,
its colour and its dash pattern come from the PDF; only the line width is scaled for print, and the image is
cropped to a window given in native pixels.
Usage: render_panels.py <panel.pdf> <out.jpg> x0 y0 x1 y1 [out_width]
"""
import io, re, sys
import pypdf
from PIL import Image, ImageDraw

def parse(pdf_path):
    page = pypdf.PdfReader(pdf_path).pages[0]
    W, H = float(page.mediabox.width), float(page.mediabox.height)
    img = Image.open(io.BytesIO(page.images[0].data)).convert('RGB')
    if img.size != (int(W), int(H)):
        img = img.resize((int(W), int(H)), Image.LANCZOS)
    content = page.get_contents().get_data().decode('latin1')
    rects = []
    for seg in content.split('re W n')[1:]:
        if '/I1 Do' in seg or not re.search(r'\bS\b', seg):
            continue
        block = seg[:re.search(r'\bS\b', seg).start()]
        lw = re.search(r'([\d.]+)\s+w\b', block)
        rgb = re.search(r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+RG', block)
        gray = re.search(r'([\d.]+)\s+G\b', block)
        dash = re.search(r'\[([\d.\s]*)\]\s+[\d.]+\s+d', block)
        if rgb:
            col = tuple(int(round(float(v) * 255)) for v in rgb.groups())
        elif gray:
            g = int(round(float(gray.group(1)) * 255)); col = (g, g, g)
        else:
            continue
        pts = [(float(x), float(y)) for x, y in re.findall(r'([\d.]+)\s+([\d.]+)\s+[ml]\b', block)]
        if not pts:
            continue
        xs = [p[0] for p in pts]; ys = [H - p[1] for p in pts]
        rects.append(dict(box=(min(xs), min(ys), max(xs), max(ys)), col=col,
                          lw=float(lw.group(1)) if lw else 1.0,
                          dash=bool(dash and dash.group(1).strip())))
    return img, rects

def dashed_rect(d, box, col, width, seg):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]:
        L = max(abs(bx - ax), abs(by - ay)); n = max(1, int(L // seg))
        for i in range(0, n, 2):
            t0, t1 = i / n, min(1, (i + 1) / n)
            d.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0), (ax + (bx - ax) * t1, ay + (by - ay) * t1)], fill=col, width=width)

def render(pdf_path, out, win, out_w=1200, scale_lw=2.2):
    img, rects = parse(pdf_path)
    x0, y0, x1, y1 = win
    s = out_w / (x1 - x0)
    crop = img.crop(win).resize((out_w, int(round((y1 - y0) * s))), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    for r in rects:
        bx0, by0, bx1, by1 = r['box']
        box = ((bx0 - x0) * s, (by0 - y0) * s, (bx1 - x0) * s, (by1 - y0) * s)
        w = max(2, int(round(r['lw'] * scale_lw)))
        if r['dash']:
            dashed_rect(d, box, r['col'], w, seg=4 * w)
        else:
            d.rectangle(box, outline=r['col'], width=w)
    crop.save(out, quality=93)
    return rects

if __name__ == '__main__':
    pdf, out = sys.argv[1], sys.argv[2]
    win = tuple(int(v) for v in sys.argv[3:7])
    ow = int(sys.argv[7]) if len(sys.argv) > 7 else 1200
    for r in render(pdf, out, win, ow):
        print(r)
