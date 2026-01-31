import csv
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_PATH = REPO_ROOT / "comparison" / "submitted1.csv"
OUT_CSV_PATH = REPO_ROOT / "comparison" / "score_breakdown.csv"
OUT_TXT_PATH = REPO_ROOT / "comparison" / "score_summary.txt"

TREE_VERTS = [
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


def parse_sval(text):
    text = text.strip()
    if not text:
        return 0.0
    if text[0] in ("s", "S"):
        text = text[1:]
    try:
        return float(text)
    except ValueError:
        return 0.0


def rotate_point(x, y, deg):
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return (x * c - y * s), (x * s + y * c)


def update_bounds(bounds, x, y):
    if bounds is None:
        return [x, y, x, y]
    bounds[0] = min(bounds[0], x)
    bounds[1] = min(bounds[1], y)
    bounds[2] = max(bounds[2], x)
    bounds[3] = max(bounds[3], y)
    return bounds


def compute_s_for_layout(rows):
    bounds = None
    for x, y, deg in rows:
        for vx, vy in TREE_VERTS:
            rx, ry = rotate_point(vx, vy, deg)
            bounds = update_bounds(bounds, x + rx, y + ry)
    if bounds is None:
        return 0.0
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return max(width, height)


def load_layouts(path):
    layouts = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 4:
            raise SystemExit("CSV must have id,x,y,deg columns.")
        for row in reader:
            if len(row) < 4:
                continue
            idv = row[0].strip()
            if "_" not in idv:
                continue
            n_str, _ = idv.split("_", 1)
            try:
                n = int(n_str)
            except ValueError:
                continue
            if not row[1].strip().startswith(("s", "S")):
                raise SystemExit("x values must be prefixed with 's'.")
            if not row[2].strip().startswith(("s", "S")):
                raise SystemExit("y values must be prefixed with 's'.")
            if not row[3].strip().startswith(("s", "S")):
                raise SystemExit("deg values must be prefixed with 's'.")
            x = parse_sval(row[1])
            y = parse_sval(row[2])
            deg = parse_sval(row[3])
            if x < -100 or x > 100 or y < -100 or y > 100:
                raise SystemExit("x/y values must be within [-100, 100].")
            layouts[n].append((x, y, deg))
    return layouts


def score_from_s(s_by_n):
    items = [(n, s_by_n[n]) for n in sorted(s_by_n.keys())]
    if not items:
        return 0.0
    return sum((s * s) / n for n, s in items)


def main():
    layouts = load_layouts(SUBMISSION_PATH)
    s_by_n = {}
    for n, rows in layouts.items():
        s_by_n[n] = compute_s_for_layout(rows)

    Path(OUT_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    with Path(OUT_CSV_PATH).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "count", "s", "s2", "s2_over_n"])
        for n in sorted(s_by_n.keys()):
            s = s_by_n[n]
            count = len(layouts.get(n, []))
            w.writerow([n, count, f"{s:.9f}", f"{s*s:.9f}", f"{(s*s)/n:.9f}"])

    score = score_from_s(s_by_n)
    with Path(OUT_TXT_PATH).open("w", encoding="utf-8") as f:
        f.write("score_mode: sum_s2_over_n\n")
        f.write(f"submission: {SUBMISSION_PATH}\n")
        f.write(f"n_count: {len(s_by_n)}\n")
        f.write(f"score: {score:.9f}\n")

    print(f"Score (sum_s2_over_n): {score:.9f}")
    print(f"Wrote: {OUT_CSV_PATH}")
    print(f"Wrote: {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
