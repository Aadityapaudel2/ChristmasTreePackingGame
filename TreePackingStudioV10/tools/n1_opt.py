import math
from pathlib import Path

OUT_PATH = r"n:\kAGGLE\n1_best.txt"

VERTS = [
    (0.0, 0.8),
    (0.125, 0.5),
    (0.0625, 0.5),
    (0.2, 0.25),
    (0.1, 0.25),
    (0.35, 0.0),
    (0.075, 0.0),
    (0.075, -0.2),
    (-0.075, -0.2),
    (-0.075, 0.0),
    (-0.35, 0.0),
    (-0.1, 0.25),
    (-0.2, 0.25),
    (-0.0625, 0.5),
    (-0.125, 0.5),
]


def bounds_for_deg(deg):
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for x, y in VERTS:
        rx = x * c - y * s
        ry = x * s + y * c
        minx = min(minx, rx)
        miny = min(miny, ry)
        maxx = max(maxx, rx)
        maxy = max(maxy, ry)
    width = maxx - minx
    height = maxy - miny
    return width, height, max(width, height)


def search_best(step):
    best = None
    deg = 0.0
    while deg <= 180.0 + 1e-9:
        w, h, s = bounds_for_deg(deg)
        if best is None or s < best[0]:
            best = (s, deg, w, h)
        deg += step
    return best


def main():
    coarse = search_best(0.5)
    _, deg0, _, _ = coarse
    start = max(0.0, deg0 - 1.0)
    end = min(180.0, deg0 + 1.0)

    best = None
    deg = start
    while deg <= end + 1e-9:
        w, h, s = bounds_for_deg(deg)
        if best is None or s < best[0]:
            best = (s, deg, w, h)
        deg += 0.01

    s, deg, w, h = best
    text = (
        f"best_deg={deg:.6f}\n"
        f"s={s:.9f}\n"
        f"s^2={s*s:.9f}\n"
        f"width={w:.9f}\n"
        f"height={h:.9f}\n"
        "x=0.0\n"
        "y=0.0\n"
    )
    Path(OUT_PATH).write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
