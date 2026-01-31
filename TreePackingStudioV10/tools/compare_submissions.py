import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHA_PATH = REPO_ROOT / "comparison" / "submitted1.csv"
BETA_PATH = REPO_ROOT / "comparison" / "raw.csv"
OUT_PATH = REPO_ROOT / "comparison" / "compare_out1.csv"

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


def compare(alpha_path, beta_path, out_path):
    alpha = load_layouts(alpha_path)
    beta = load_layouts(beta_path)
    all_n = sorted(set(alpha.keys()) | set(beta.keys()))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "alpha_s",
                "alpha_s2",
                "beta_s",
                "beta_s2",
                "diff_s",
                "better",
            ]
        )
        for n in all_n:
            s_a = compute_s_for_layout(alpha.get(n, []))
            s_b = compute_s_for_layout(beta.get(n, []))
            s2_a = s_a * s_a
            s2_b = s_b * s_b
            diff = s_b - s_a
            if abs(diff) < 1e-9:
                better = "tie"
            elif diff < 0:
                better = "beta"
            else:
                better = "alpha"
            writer.writerow(
                [
                    n,
                    f"{s_a:.9f}",
                    f"{s2_a:.9f}",
                    f"{s_b:.9f}",
                    f"{s2_b:.9f}",
                    f"{diff:.9f}",
                    better,
                ]
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compare two submission CSVs (id,x,y,deg) and output per-N S and S^2."
    )
    parser.add_argument("alpha_csv", nargs="?", help="First CSV (alpha).")
    parser.add_argument("beta_csv", nargs="?", help="Second CSV (beta).")
    parser.add_argument("--out", help="Output CSV path.")
    args = parser.parse_args()

    alpha = args.alpha_csv or ALPHA_PATH
    beta = args.beta_csv or BETA_PATH
    out_path = args.out or OUT_PATH
    compare(alpha, beta, out_path)


if __name__ == "__main__":
    main()
