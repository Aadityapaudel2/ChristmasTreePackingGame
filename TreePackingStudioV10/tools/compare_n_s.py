import csv
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHA_PATH = REPO_ROOT / "comparison" / "submitted1.csv"
BETA_PATH = REPO_ROOT / "comparison" / "raw.csv"
OUT_PATH = REPO_ROOT / "comparison" / "n_s_compare.csv"

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
            x = parse_sval(row[1])
            y = parse_sval(row[2])
            deg = parse_sval(row[3])
            layouts[n].append((x, y, deg))
    return layouts


def main():
    alpha = load_layouts(ALPHA_PATH)
    beta = load_layouts(BETA_PATH)
    all_n = sorted(set(alpha.keys()) | set(beta.keys()))

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "s_alpha", "s_beta", "difference"])
        for n in all_n:
            s_a = compute_s_for_layout(alpha.get(n, []))
            s_b = compute_s_for_layout(beta.get(n, []))
            writer.writerow([n, f"{s_a:.9f}", f"{s_b:.9f}", f"{(s_b - s_a):.9f}"])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
