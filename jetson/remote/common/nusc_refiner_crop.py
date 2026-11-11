"""The 64x64 crop used by the image refiner (copied from scripts/34_refiner_dataset.py)."""
import cv2

S, CTXF = 64, 1.5


def crop64(img, box):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * CTXF, (y2 - y1) * CTXF
    side = max(w, h, 8.0)
    X1, Y1, X2, Y2 = int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)
    H, W = img.shape[:2]
    pad = cv2.copyMakeBorder(img, max(-Y1, 0), max(Y2 - H, 0), max(-X1, 0), max(X2 - W, 0), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    oy, ox = max(-Y1, 0), max(-X1, 0)
    patch = pad[Y1 + oy:Y2 + oy, X1 + ox:X2 + ox]
    return cv2.resize(patch, (S, S), interpolation=cv2.INTER_AREA)
