
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pygame
from shapely.affinity import rotate as shp_rotate, translate as shp_translate
from shapely.geometry import Polygon
from shapely.ops import unary_union

try:
    import pandas as pd
except ImportError:
    pd = None

# ----------------------------
# Geometry: exact Kaggle tree polygon (15 vertices)
# ----------------------------
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

BASE_POLY = Polygon(TREE_VERTS)

SUBMISSION_MAX_N = 200
SUBMISSION_DECIMALS = 6
VERSION_TAG = "v10"
OUTPUT_DIR_NAME = "game_v10"
LOAD_MASTER_PATH = r"n:\kAGGLE\comparison\submitted1.csv"


def format_submission_value(value):
    text = f"{value:.{SUBMISSION_DECIMALS}f}"
    text = text.rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return f"s{text}"


def parse_submission_value(text):
    text = text.strip()
    if not text:
        return 0.0
    if text[0] in ("s", "S"):
        text = text[1:]
    try:
        return float(text)
    except ValueError:
        return 0.0


def format_id(n, idx):
    return f"{n:03d}_{idx}"


def parse_id(text):
    parts = text.strip().split("_", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def resolve_repo_root():
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidate = exe_dir.parent
        if (candidate / "v8").exists():
            return candidate
        return exe_dir
    return Path(__file__).resolve().parents[1]


def resolve_output_root():
    override = os.getenv("TREE_SANDBOX_OUT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return resolve_repo_root() / OUTPUT_DIR_NAME


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c0, c1, t):
    return (
        int(lerp(c0[0], c1[0], t)),
        int(lerp(c0[1], c1[1], t)),
        int(lerp(c0[2], c1[2], t)),
    )


def scale_color(color, factor):
    return (
        clamp(int(color[0] * factor), 0, 255),
        clamp(int(color[1] * factor), 0, 255),
        clamp(int(color[2] * factor), 0, 255),
    )


@dataclass
class TreeInstance:
    x: float
    y: float
    deg: float
    poly: Polygon
    base_poly: Polygon

    @classmethod
    def from_pose(cls, x, y, deg, base_poly=None):
        bp = base_poly if base_poly is not None else BASE_POLY
        p = shp_rotate(bp, deg, origin=(0.0, 0.0), use_radians=False)
        p = shp_translate(p, xoff=x, yoff=y)
        return cls(x=x, y=y, deg=deg, poly=p, base_poly=bp)

    def set_pose(self, x, y, deg):
        self.x = x
        self.y = y
        self.deg = deg
        p = shp_rotate(self.base_poly, deg, origin=(0.0, 0.0), use_radians=False)
        self.poly = shp_translate(p, xoff=x, yoff=y)


def strict_overlap(p, q):
    a0, b0, a1, b1 = p.bounds
    c0, d0, c1, d1 = q.bounds
    if a1 < c0 or c1 < a0 or b1 < d0 or d1 < b0:
        return False
    return p.intersects(q) and (not p.touches(q))


def polygon_to_screen(poly, world_to_screen):
    pts = list(poly.exterior.coords)
    return [world_to_screen(x, y) for x, y in pts]


def bbox_polys(polys):
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    return minx, miny, maxx, maxy


class TreeSandboxApp:
    def __init__(self):
        pygame.init()
        self.size = (1680, 980)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(f"Tree Packing Studio {VERSION_TAG}")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont(["bahnschrift", "candara", "consolas"], 22, bold=True)
        self.font_ui = pygame.font.SysFont(["bahnschrift", "candara", "consolas"], 16)
        self.font_small = pygame.font.SysFont(["bahnschrift", "candara", "consolas"], 14)
        self.font_mono = pygame.font.SysFont(["consolas", "couriernew"], 13)

        self.theme = {
            "bg_top": (10, 12, 18),
            "bg_bottom": (6, 8, 12),
            "panel": (18, 22, 30),
            "panel_alt": (16, 19, 26),
            "panel_inner": (22, 26, 34),
            "panel_border": (42, 50, 64),
            "accent": (86, 214, 170),
            "accent_alt": (234, 188, 108),
            "accent_dim": (58, 136, 114),
            "text": (230, 235, 240),
            "text_muted": (160, 170, 180),
            "danger": (232, 86, 86),
            "ok": (100, 220, 140),
            "grid": (30, 40, 52),
            "shadow": (6, 8, 10),
            "view_bg": (12, 14, 20),
        }

        self.trees = []
        self.selected = None
        self.dragging = False
        self.last_mouse_world = None
        self.last_drag_delta = (0.0, 0.0)
        self.mouse_world = (0.0, 0.0)
        self.pan_mode = False
        self.panning = False
        self.pan_last_screen = None

        self.camera = [0.0, 0.0]
        self.scale = 140.0
        self.default_scale = 140.0
        self.zoom_limits = (10.0, 8000.0)
        self.zoom_factor = 1.12
        self.nudge_step = 0.01
        self.nudge_step_fine = 0.001
        self.nudge_step_ultra = 0.0001
        self.rotate_step = 1.0
        self.rotate_step_fine = 0.1
        self.rotate_step_ultra = 0.01

        self.show_help = False

        self.stats_dirty = True
        self.stats_last_time = 0.0
        self.stats_update_interval = 0.25
        self.stats = {}
        self.union_poly = None
        self.contact_count = 0

        self.selecting = False
        self.sel_start = None
        self.sel_end = None
        self.sel_stats = {}

        self.repo_root = resolve_repo_root()
        self.out_dir = resolve_output_root()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_json = self.out_dir / "layout.json"
        self.out_csv = self.out_dir / "layout_kaggle.csv"
        self.submission_max_n = SUBMISSION_MAX_N
        self.snapshots_dir = self.out_dir / "snapshots"
        self.outputs_base = self.out_dir / "gameoutputs"
        self.outputs_best = self.outputs_base / "best"
        self.outputs_runs = self.outputs_base / "runs"
        self.outputs_ledger = self.outputs_base / "ledger.csv"

        self.canonical_dir = self.out_dir / "canonical"
        self.canonical_csv = self.canonical_dir / "canonical.csv"
        self.canonical_layouts = {}
        self.canonical_s = {}
        self.canonical_loaded = False
        self.load_canonical_csv()
        if not self.canonical_csv.exists():
            self.initialize_canonical_csv()

        self.grid_table = {}
        self.grid_path = self.find_grid_path()
        self.grid_loaded = False
        self.load_grid_table()

        self.s_target = None
        self.square_origin = [0.0, 0.0]

        self.optimizing = False
        self.opt_start_time = 0.0
        self.opt_end_time = 0.0
        self.opt_best_s = None
        self.opt_best_layout = []
        self.opt_best_stats = {}
        self.opt_iters_per_frame = 240
        self.opt_duration_sec = 8.0
        self.opt_last_improve = 0.0
        self.opt_start_layout = []
        self.opt_start_s = None

        self.shape_name = "tree"
        self.base_poly = BASE_POLY
        self.tri_poly = self.make_triangle_cover(BASE_POLY)
        self.para_poly = self.make_parallelogram_from_triangle(self.tri_poly)
        self.area_bound = float(self.base_poly.area)

        self.target_n = 12
        self.input_active = False
        self.input_text = str(self.target_n)
        self.input_replace = False
        self.auto_pass_text = ""
        self.auto_pass_active = False
        self.auto_pass_replace = False
        self.auto_pass_override = None
        self.auto_pass_menu_open = False
        self.auto_pass_options = [None, 5, 10, 15, 20, 30, 40]
        self.csv_name_text = Path(LOAD_MASTER_PATH).name
        self.csv_name_active = False
        self.csv_name_replace = False
        self.multi_opt_active = False
        self.multi_opt_left = 0
        self.multi_opt_total = 0

        self.status_msg = ""
        self.status_time = 0.0
        self.last_export = ""
        self.last_record = ""
        self.last_snapshot = ""
        self.workspace_lines = ["Ready."]

        self.auto_active = False
        self.auto_start_n = 167
        self.auto_end_n = 200
        self.auto_current_n = None
        self.auto_phase = "idle"
        self.auto_passes_total = 0
        self.auto_passes_left = 0
        self.auto_opt_running = False
        self.auto_seed_repeats = 1
        self.auto_seed_done = 0
        self.auto_seed_best_layout = None
        self.auto_seed_best_s = None
        self.auto_log_path = self.out_dir / "automation" / "trialrecord.csv"

        self.layout_ui()
        self.build_surfaces()

        self.random_place_n(self.target_n)
        self.frame_view()

    # ---------- setup ----------
    def layout_ui(self):
        w, h = self.size
        self.margin = 16
        self.top_h = 64
        self.bottom_h = 200
        self.right_w = 320

        self.top_rect = pygame.Rect(0, 0, w, self.top_h)
        self.bottom_rect = pygame.Rect(0, h - self.bottom_h, w, self.bottom_h)
        self.right_rect = pygame.Rect(w - self.right_w, self.top_h, self.right_w, h - self.top_h - self.bottom_h)
        self.view_rect = pygame.Rect(
            self.margin,
            self.top_h + self.margin,
            w - self.right_w - 2 * self.margin,
            h - self.top_h - self.bottom_h - 2 * self.margin,
        )

        self.buttons = {}
        self.button_labels = {
            "generate": "Generate (G)",
            "auto": "Auto Place (P)",
            "opt": "Optimize (O)",
            "auto_opt": "Auto O (T)",
            "record": "Record (L)",
            "load_n": "Load N (K)",
            "csv_set": "Set",
            "frame": "Frame (F)",
            "reset_view": "Reset (0)",
            "n_minus": "-1",
            "n_plus": "+1",
        }

        bx = self.bottom_rect.x + self.margin
        by = self.bottom_rect.y + self.margin
        button_w = 160
        button_h = 36
        gap = 12

        row1 = ["generate", "auto", "opt", "auto_opt"]
        row2 = ["record"]

        for col, name in enumerate(row1):
            self.buttons[name] = pygame.Rect(bx + col * (button_w + gap), by, button_w, button_h)

        grid_width = len(row1) * button_w + (len(row1) - 1) * gap
        record_rect = pygame.Rect(bx, by + button_h + gap, grid_width, button_h)
        self.buttons["record"] = record_rect
        n_panel_x = bx + grid_width + gap
        n_panel_w = 240
        n_panel_h = 268
        self.n_panel_rect = pygame.Rect(n_panel_x, by, n_panel_w, n_panel_h)
        self.n_value_rect = pygame.Rect(n_panel_x + 12, by + 16, n_panel_w - 24, 18)
        self.n_input_rect = pygame.Rect(n_panel_x + 12, by + 56, n_panel_w - 24, 26)

        mini_gap = 6
        mini_w = (n_panel_w - 24 - mini_gap) // 2
        mini_h = 20
        mini_y = self.n_input_rect.bottom + 6
        self.buttons["n_minus"] = pygame.Rect(n_panel_x + 12, mini_y, mini_w, mini_h)
        self.buttons["n_plus"] = pygame.Rect(n_panel_x + 12 + mini_w + mini_gap, mini_y, mini_w, mini_h)
        load_y = mini_y + mini_h + 6
        self.buttons["load_n"] = pygame.Rect(n_panel_x + 12, load_y, n_panel_w - 24, 22)
        auto_y = load_y + 26
        self.auto_pass_label_pos = (n_panel_x + 12, auto_y)
        self.auto_pass_rect = pygame.Rect(n_panel_x + 12, auto_y + 16, n_panel_w - 24, 22)
        self.auto_pass_menu_rect = pygame.Rect(
            self.auto_pass_rect.x,
            self.auto_pass_rect.bottom + 4,
            self.auto_pass_rect.w,
            len(self.auto_pass_options) * 20,
        )
        csv_y = self.auto_pass_rect.bottom + 30
        self.csv_label_pos = (n_panel_x + 12, csv_y)
        self.csv_name_rect = pygame.Rect(n_panel_x + 12, csv_y + 16, n_panel_w - 90, 22)
        self.buttons["csv_set"] = pygame.Rect(self.csv_name_rect.right + 6, csv_y + 16, 60, 22)

        view_panel_x = n_panel_x + n_panel_w + gap
        view_panel_w = max(200, self.bottom_rect.right - view_panel_x - self.margin)
        self.view_panel_rect = pygame.Rect(view_panel_x, by, view_panel_w, n_panel_h)

        top_btn_w = 84
        top_btn_h = 24
        top_btn_gap = 8
        top_btn_y = (self.top_h - top_btn_h) // 2
        reset_x = self.size[0] - self.margin - top_btn_w
        frame_x = reset_x - top_btn_gap - top_btn_w
        self.buttons["frame"] = pygame.Rect(frame_x, top_btn_y, top_btn_w, top_btn_h)
        self.buttons["reset_view"] = pygame.Rect(reset_x, top_btn_y, top_btn_w, top_btn_h)
        self.top_action_left = frame_x

        status_y = by + button_h * 2 + gap * 2
        self.status_rect = pygame.Rect(bx, status_y, self.bottom_rect.w - 2 * self.margin, self.bottom_rect.bottom - status_y - 12)

    def build_surfaces(self):
        self.bg_surface = pygame.Surface(self.size)
        top = self.theme["bg_top"]
        bottom = self.theme["bg_bottom"]
        h = self.size[1]
        for y in range(h):
            t = y / max(1, h - 1)
            color = lerp_color(top, bottom, t)
            pygame.draw.line(self.bg_surface, color, (0, y), (self.size[0], y))

        self.view_grid = pygame.Surface(self.view_rect.size, pygame.SRCALPHA)
        step = 40
        grid = self.theme["grid"]
        grid_color = (grid[0], grid[1], grid[2], 60)
        for x in range(0, self.view_rect.w + 1, step):
            pygame.draw.line(self.view_grid, grid_color, (x, 0), (x, self.view_rect.h), 1)
        for y in range(0, self.view_rect.h + 1, step):
            pygame.draw.line(self.view_grid, grid_color, (0, y), (self.view_rect.w, y), 1)

    def set_status(self, msg):
        self.status_msg = msg
        self.status_time = time.time()

    def set_workspace_lines(self, lines):
        cleaned = [line for line in lines if line]
        if not cleaned:
            cleaned = ["Ready."]
        self.workspace_lines = cleaned[-4:]

    def append_automation_log(self, n, passes, best_s, note=""):
        path = self.auto_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        header = "timestamp,N,passes,best_s,note\n"
        line = f"{ts},{n},{passes},{best_s:.9f},{note}\n"
        try:
            if not path.exists() or path.stat().st_size == 0:
                path.write_text(header + line, encoding="utf-8")
                return path, False
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
            return path, False
        except OSError:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            try:
                fallback.write_text(header + line, encoding="utf-8")
                return fallback, True
            except OSError:
                return None, True

    def start_automation(self, start_n=None, end_n=None):
        if self.auto_active:
            return
        if self.optimizing:
            self.stop_optimize()
        if start_n is None:
            start_n = self.auto_start_n
        if end_n is None:
            end_n = self.auto_end_n
        start_n = max(2, int(start_n))
        end_n = max(start_n, int(end_n))
        if self.shape_name != "tree":
            self.set_shape("tree")
        self.auto_active = True
        self.auto_start_n = start_n
        self.auto_end_n = end_n
        self.auto_current_n = start_n
        self.auto_phase = "autoplace"
        self.auto_passes_total = 0
        self.auto_passes_left = 0
        self.auto_opt_running = False
        self.auto_seed_done = 0
        self.auto_seed_best_layout = None
        self.auto_seed_best_s = None
        self.set_workspace_lines([f"Auto run {start_n}-{end_n}."])
        self.set_status("Automation started.")

    def stop_automation(self, note="Automation stopped."):
        self.auto_active = False
        self.auto_phase = "idle"
        self.auto_opt_running = False
        self.set_status(note)
        self.set_workspace_lines([note])

    def automation_step(self):
        if not self.auto_active:
            return
        if self.auto_current_n is None:
            self.auto_current_n = self.auto_start_n

        if self.auto_phase == "autoplace":
            n = self.auto_current_n
            self.target_n = n
            self.input_text = str(n)
            self.auto_place(n)
            if self.stats_dirty:
                self.update_stats()
                self.stats_dirty = False
            s_now = float(self.stats.get("s", 0.0))
            if self.auto_seed_best_s is None or s_now < self.auto_seed_best_s:
                self.auto_seed_best_s = s_now
                self.auto_seed_best_layout = [{"x": t.x, "y": t.y, "deg": t.deg} for t in self.trees]
            self.auto_seed_done += 1
            if self.auto_seed_done < max(1, self.auto_seed_repeats):
                return
            if self.auto_seed_best_layout:
                self.apply_layout(self.auto_seed_best_layout)
                self.update_stats()
                self.stats_dirty = False
            passes = max(1, (n + 1) // 2)
            self.auto_passes_total = passes
            self.auto_passes_left = passes
            self.auto_opt_running = False
            self.auto_phase = "optimize"
            self.set_workspace_lines([f"Auto N={n} seed done.", f"Optimize passes: {passes}"])
            return

        if self.auto_phase == "optimize":
            if self.auto_passes_left <= 0:
                self.auto_phase = "record"
                return
            if not self.auto_opt_running:
                self.start_optimize(preserve_best=True)
                if not self.optimizing:
                    self.auto_passes_left = 0
                    self.auto_phase = "record"
                else:
                    self.auto_opt_running = True
                    done = self.auto_passes_total - self.auto_passes_left + 1
                    self.set_workspace_lines([f"Auto N={self.auto_current_n}", f"Optimize {done}/{self.auto_passes_total}"])
                return
            if self.auto_opt_running and not self.optimizing:
                self.auto_opt_running = False
                self.auto_passes_left -= 1
                return

        if self.auto_phase == "record":
            self.record_output()
            best_s = self.opt_best_s if self.opt_best_s is not None else float(self.stats.get("s", 0.0))
            note = self.last_record or ""
            log_path, log_fallback = self.append_automation_log(
                self.auto_current_n, self.auto_passes_total, best_s, note=note
            )
            existing = list(self.workspace_lines)
            if log_path is None:
                existing.append("auto log: failed")
            else:
                suffix = " (fallback)" if log_fallback else ""
                existing.append(f"auto log: {log_path.name}{suffix}")
            self.set_workspace_lines(existing)
            if self.auto_current_n >= self.auto_end_n:
                self.stop_automation("Automation finished.")
                return
            self.auto_current_n += 1
            self.auto_phase = "autoplace"
            self.auto_seed_done = 0
            self.auto_seed_best_layout = None
            self.auto_seed_best_s = None

    # ---------- geometry ----------
    def world_to_screen(self, x, y):
        cx, cy = self.view_rect.centerx, self.view_rect.centery
        sx = cx + (x - self.camera[0]) * self.scale
        sy = cy - (y - self.camera[1]) * self.scale
        return int(sx), int(sy)

    def screen_to_world(self, sx, sy):
        cx, cy = self.view_rect.centerx, self.view_rect.centery
        x = (sx - cx) / self.scale + self.camera[0]
        y = (cy - sy) / self.scale + self.camera[1]
        return x, y

    def add_tree(self, x, y, deg=0.0):
        self.trees.append(TreeInstance.from_pose(x, y, deg, base_poly=self.base_poly))
        self.selected = len(self.trees) - 1
        self.stats_dirty = True

    def remove_selected(self):
        if self.selected is None:
            return
        self.trees.pop(self.selected)
        if not self.trees:
            self.selected = None
        else:
            self.selected = min(self.selected, len(self.trees) - 1)
        self.stats_dirty = True

    def rotate_selected(self, ddeg):
        if self.selected is None:
            return
        t = self.trees[self.selected]
        t.set_pose(t.x, t.y, t.deg + ddeg)
        self.stats_dirty = True

    def move_selected(self, dx, dy):
        if self.selected is None:
            return
        t = self.trees[self.selected]
        t.set_pose(t.x + dx, t.y + dy, t.deg)
        self.stats_dirty = True

    def pick_tree(self, wx, wy):
        if not self.trees:
            return None
        best = None
        best_d2 = None
        for i, t in enumerate(self.trees):
            cx, cy = t.poly.centroid.coords[0]
            d2 = (cx - wx) ** 2 + (cy - wy) ** 2
            if best_d2 is None or d2 < best_d2:
                best = i
                best_d2 = d2
        return best

    def selected_overlap_state(self):
        if self.selected is None:
            return "ok"
        p = self.trees[self.selected].poly
        touch = False
        for i, t in enumerate(self.trees):
            if i == self.selected:
                continue
            if strict_overlap(p, t.poly):
                return "overlap"
            if p.touches(t.poly):
                touch = True
        return "touch" if touch else "ok"

    def snap_selected_to_contact(self):
        if self.selected is None:
            return
        dx, dy = self.last_drag_delta
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return
        t = self.trees[self.selected]
        mag = math.hypot(dx, dy)
        ux, uy = dx / mag, dy / mag

        step = 0.02
        lo = 0.0
        hi = None
        for k in range(1, 400):
            dist = k * step
            trial = TreeInstance.from_pose(t.x + ux * dist, t.y + uy * dist, t.deg, base_poly=self.base_poly)
            if self.is_tree_legal(trial, ignore_index=self.selected):
                lo = dist
            else:
                hi = dist
                break
        if hi is None:
            hi = lo

        for _ in range(30):
            mid = 0.5 * (lo + hi)
            trial = TreeInstance.from_pose(t.x + ux * mid, t.y + uy * mid, t.deg, base_poly=self.base_poly)
            if self.is_tree_legal(trial, ignore_index=self.selected):
                lo = mid
            else:
                hi = mid

        t.set_pose(t.x + ux * lo, t.y + uy * lo, t.deg)
        self.stats_dirty = True

    def in_square_bounds(self, poly):
        s_side = self.current_square_side()
        ox, oy = self.square_origin
        minx, miny, maxx, maxy = poly.bounds
        eps = 1e-6
        return (minx >= ox - eps) and (miny >= oy - eps) and (maxx <= ox + s_side + eps) and (maxy <= oy + s_side + eps)

    def is_tree_legal(self, tree, ignore_index=None):
        if not self.in_square_bounds(tree.poly):
            return False
        for i, t in enumerate(self.trees):
            if ignore_index is not None and i == ignore_index:
                continue
            if strict_overlap(tree.poly, t.poly):
                return False
        return True

    def compute_s_for_polys(self, polys):
        minx, miny, maxx, maxy = bbox_polys(polys)
        return max(maxx - minx, maxy - miny)

    def update_stats(self):
        if not self.trees:
            self.stats = {
                "n": 0,
                "s": 0.0,
                "term": 0.0,
                "union": 0.0,
                "empty": 0.0,
                "density": 0.0,
                "bbox": (0.0, 0.0, 0.0, 0.0),
            }
            self.union_poly = None
            self.contact_count = 0
            return

        polys = [t.poly for t in self.trees]
        minx, miny, maxx, maxy = bbox_polys(polys)
        width = maxx - minx
        height = maxy - miny
        s = max(width, height)
        if s <= 0:
            s = 0.0
        U = unary_union(polys)
        union_area = float(U.area)
        empty = max(0.0, s * s - union_area)
        term = (s * s) / len(polys) if len(polys) > 0 else 0.0
        density = union_area / (s * s) if s > 0 else 0.0

        contact = 0
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polys[i].touches(polys[j]):
                    contact += 1

        self.stats = {
            "n": len(polys),
            "s": s,
            "term": term,
            "union": union_area,
            "empty": empty,
            "density": density,
            "bbox": (minx, miny, maxx, maxy),
        }
        self.square_origin[0] = minx
        self.square_origin[1] = miny
        self.union_poly = U
        self.contact_count = contact

        if self.opt_best_s is None or s < self.opt_best_s:
            self.opt_best_s = s
            self.opt_best_layout = [{"x": t.x, "y": t.y, "deg": t.deg} for t in self.trees]
            self.opt_best_stats = dict(self.stats)
            self.opt_last_improve = time.time()

    def reset_best_tracking(self):
        self.opt_best_s = None
        self.opt_best_layout = []
        self.opt_best_stats = {}

    def update_selection_stats(self):
        if not self.sel_start or not self.sel_end:
            self.sel_stats = {}
            return
        x0, y0 = self.sel_start
        x1, y1 = self.sel_end
        minx, maxx = min(x0, x1), max(x0, x1)
        miny, maxy = min(y0, y1), max(y0, y1)
        w = maxx - minx
        h = maxy - miny
        if w <= 0 or h <= 0:
            self.sel_stats = {}
            return
        rect = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
        area_rect = w * h
        if self.union_poly is None:
            occ = 0.0
        else:
            occ = float(self.union_poly.intersection(rect).area)
        empty = max(0.0, area_rect - occ)
        self.sel_stats = {
            "area": area_rect,
            "occ": occ,
            "empty": empty,
            "ratio": empty / area_rect if area_rect > 0 else 0.0,
        }

    def current_square_side(self):
        s = self.stats.get("s", 0.0)
        minx, miny, maxx, maxy = self.base_poly.bounds
        base_size = max(maxx - minx, maxy - miny)
        target = self.s_target if self.s_target is not None else 0.0
        return max(s, target, base_size, 1.0)

    # ---------- shapes ----------
    def make_triangle_cover(self, poly):
        apex = (0.0, 0.8)
        base_left = (-0.4375, -0.2)
        base_right = (0.4375, -0.2)
        return Polygon([apex, base_left, base_right])

    def make_parallelogram_from_triangle(self, tri):
        pts = list(tri.exterior.coords)[:-1]
        if len(pts) != 3:
            return Polygon(pts)
        apex = max(pts, key=lambda p: (p[1], -abs(p[0])))
        base_pts = [p for p in pts if p != apex]
        if len(base_pts) != 2:
            return Polygon(pts)
        base_pts.sort(key=lambda p: p[0])
        a, b = base_pts
        ax, ay = a
        bx, by = b
        cx, cy = apex
        d = (bx + (cx - ax), by + (cy - ay))
        return Polygon([a, b, d, apex])

    def set_shape(self, name):
        if name == "triangle":
            self.shape_name = "triangle"
            self.base_poly = self.tri_poly
        elif name == "parallelogram":
            self.shape_name = "parallelogram"
            self.base_poly = self.para_poly
        else:
            self.shape_name = "tree"
            self.base_poly = BASE_POLY
        self.area_bound = float(self.base_poly.area)
        self.rebuild_shapes()
        self.reset_best_tracking()
        self.set_status(f"Shape set to {self.shape_name}.")

    def rebuild_shapes(self):
        new_trees = []
        for t in self.trees:
            new_trees.append(TreeInstance.from_pose(t.x, t.y, t.deg, base_poly=self.base_poly))
        self.trees = new_trees
        if self.trees:
            self.selected = 0
        self.stats_dirty = True

    # ---------- optimization ----------
    def start_optimize(self, preserve_best=False):
        if len(self.trees) < 1:
            return
        if len(self.trees) == 1 and self.shape_name == "tree":
            self.place_singleton_optimal()
            self.set_status("Optimizer skipped for N=1.")
            return
        self.update_stats()
        self.stats_dirty = False
        now = time.time()
        self.optimizing = True
        self.opt_start_time = now
        self.opt_end_time = now + self.opt_duration_sec
        self.opt_start_layout = [{"x": t.x, "y": t.y, "deg": t.deg} for t in self.trees]
        self.opt_start_s = float(self.stats.get("s", 0.0))
        current_s = self.stats.get("s", None)
        if preserve_best and self.opt_best_layout:
            if current_s is not None and (self.opt_best_s is None or current_s < self.opt_best_s):
                self.opt_best_s = current_s
                self.opt_best_layout = [{"x": t.x, "y": t.y, "deg": t.deg} for t in self.trees]
                self.opt_best_stats = dict(self.stats)
        else:
            self.opt_best_s = current_s
            self.opt_best_layout = [{"x": t.x, "y": t.y, "deg": t.deg} for t in self.trees]
            self.opt_best_stats = dict(self.stats)
        self.opt_last_improve = now
        self.set_status("Optimizer running.")

    def start_multi_optimize(self):
        if self.auto_active:
            self.set_status("Stop automation before Auto O.")
            return
        if self.optimizing:
            self.set_status("Optimizer already running.")
            return
        if not self.auto_pass_override:
            self.auto_pass_menu_open = True
            self.set_status("Set Auto passes first.")
            self.set_workspace_lines(["Auto O: set Auto passes first."])
            return
        passes = max(1, int(self.auto_pass_override))
        self.multi_opt_active = True
        self.multi_opt_total = passes
        self.multi_opt_left = passes
        self.set_workspace_lines([f"Auto O passes: {passes}"])
        self.start_optimize()
        if self.optimizing:
            self.multi_opt_left -= 1

    def stop_optimize(self):
        self.optimizing = False
        self.set_status("Optimizer stopped.")

    def optimize_step(self):
        if not self.optimizing or not self.trees:
            return
        now = time.time()
        if now >= self.opt_end_time:
            self.optimizing = False
            if self.opt_best_layout:
                self.apply_layout(self.opt_best_layout)
            self.set_status("Optimizer finished (snapped to best).")
            return

        elapsed = max(0.0, now - self.opt_start_time)
        frac = min(1.0, elapsed / max(1e-6, self.opt_duration_sec))
        move_scale = 0.12 * (1.0 - 0.7 * frac)
        rot_scale = 9.0 * (1.0 - 0.6 * frac)
        accept_temp = 0.004 * (1.0 - 0.8 * frac)

        for _ in range(self.opt_iters_per_frame):
            idx = random.randrange(len(self.trees))
            t = self.trees[idx]
            dx = random.uniform(-move_scale, move_scale)
            dy = random.uniform(-move_scale, move_scale)
            ddeg = random.uniform(-rot_scale, rot_scale)

            trial = TreeInstance.from_pose(t.x + dx, t.y + dy, t.deg + ddeg, base_poly=self.base_poly)
            if not self.is_tree_legal(trial, ignore_index=idx):
                continue

            polys = []
            for i, other in enumerate(self.trees):
                polys.append(trial.poly if i == idx else other.poly)
            s_new = self.compute_s_for_polys(polys)

            s_old = self.stats.get("s", None)
            if s_old is None:
                self.update_stats()
                s_old = self.stats.get("s", s_new)

            accept = False
            if s_new <= s_old + 1e-9:
                accept = True
            else:
                delta = s_new - s_old
                if accept_temp > 0.0:
                    prob = math.exp(-delta / accept_temp)
                    if random.random() < prob:
                        accept = True
            if accept:
                t.set_pose(trial.x, trial.y, trial.deg)
                self.stats_dirty = True

        if self.stats_dirty:
            self.update_stats()
            self.stats_dirty = False

    # ---------- placement ----------
    def apply_layout(self, layout):
        self.trees = [TreeInstance.from_pose(d["x"], d["y"], d["deg"], base_poly=self.base_poly) for d in layout]
        self.selected = 0 if self.trees else None
        self.reset_best_tracking()
        self.stats_dirty = True
        self.align_to_square_origin()

    def random_place_n(self, n):
        self.trees = []
        self.selected = None
        self.reset_best_tracking()
        self.stats_dirty = True

        n = max(1, min(200, int(n)))
        self.target_n = n
        self.input_text = str(self.target_n)

        if n == 1 and self.shape_name == "tree":
            self.place_singleton_optimal()
            return

        if self.place_from_grid(n):
            self.set_status("Placed from grid table.")
            return
        base = math.sqrt(n) * 0.9
        s_guess = max(1.0, base)
        self.s_target = s_guess
        max_restarts = 12

        for _ in range(max_restarts):
            self.trees = []
            success = True
            for _k in range(n):
                placed = False
                for _try in range(500):
                    x = random.uniform(0.0, s_guess)
                    y = random.uniform(0.0, s_guess)
                    deg = random.uniform(0.0, 360.0)
                    cand = TreeInstance.from_pose(x, y, deg, base_poly=self.base_poly)
                    if self.is_tree_legal(cand):
                        self.trees.append(cand)
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            if success:
                break
            s_guess *= 1.18

        if self.trees:
            self.selected = 0
        self.stats_dirty = True
        self.align_to_square_origin()
        self.s_target = None
        self.set_status(f"Random placed N={n}.")

    def generate_triangle_grid(self, n):
        n = max(1, min(400, int(n)))
        bw = self.base_poly.bounds[2] - self.base_poly.bounds[0]
        bh = self.base_poly.bounds[3] - self.base_poly.bounds[1]
        if bw <= 0 or bh <= 0:
            return

        step_x = bw * 1.05
        step_y = bh * 1.05
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = max(1, int(math.ceil(n / cols)))
        self.s_target = max(cols * 2 * step_x, rows * 2 * step_y, 1.0)
        layout = []
        for r in range(rows * 2):
            if len(layout) >= n:
                break
            y = r * step_y
            x_offset = 0.0 if (r % 2 == 0) else 0.5 * step_x
            for c in range(cols * 2):
                if len(layout) >= n:
                    break
                x = x_offset + c * step_x
                deg = 0.0 if (r + c) % 2 == 0 else 180.0
                cand = TreeInstance.from_pose(x, y, deg, base_poly=self.base_poly)
                if self.is_tree_legal(cand, ignore_index=None):
                    layout.append(cand)

        self.trees = layout[:n]
        self.selected = 0 if self.trees else None
        self.reset_best_tracking()
        self.stats_dirty = True
        self.update_stats()
        self.align_to_square_origin()
        self.s_target = None
        self.target_n = n
        self.input_text = str(self.target_n)
        self.set_status(f"Generated grid N={n}.")

    def generate_optimal_layout(self):
        if self.shape_name == "triangle":
            self.generate_triangle_grid(self.target_n)
        else:
            self.random_place_n(self.target_n)
        self.align_to_square_origin()

    def place_singleton_optimal(self):
        if self.shape_name != "tree":
            return False
        self.trees = [TreeInstance.from_pose(0.0, 0.0, 45.0, base_poly=self.base_poly)]
        self.selected = 0
        self.s_target = None
        self.square_origin = [0.0, 0.0]
        self.reset_best_tracking()
        self.stats_dirty = True
        self.align_to_square_origin()
        self.set_status("Placed optimal N=1 (deg 45).")
        return True

    def auto_place(self, n):
        n = max(1, min(400, int(n)))
        if n == 1 and self.shape_name == "tree":
            self.place_singleton_optimal()
            return
        bw = self.base_poly.bounds[2] - self.base_poly.bounds[0]
        bh = self.base_poly.bounds[3] - self.base_poly.bounds[1]
        if bw <= 0 or bh <= 0:
            return
        step_x = bw * 1.05
        step_y = bh * 1.05
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = max(1, int(math.ceil(n / cols)))
        self.s_target = max(cols * step_x, rows * step_y, 1.0)

        layout = []
        slots = []
        for r in range(rows):
            y = r * step_y
            x_offset = 0.0 if (r % 2 == 0) else 0.5 * step_x
            for c in range(cols):
                x = x_offset + c * step_x
                deg = 0.0 if (r + c) % 2 == 0 else 180.0
                slots.append((x, y, deg))

        random.shuffle(slots)
        shifts = [0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75]
        for x, y, deg in slots:
            if len(layout) >= n:
                break
            base_cand = TreeInstance.from_pose(x, y, deg, base_poly=self.base_poly)
            if self.is_tree_legal(base_cand):
                layout.append(base_cand)
                continue
            ok = False
            for dx in shifts:
                for dy in shifts:
                    cand = TreeInstance.from_pose(x + dx * bw, y + dy * bh, deg, base_poly=self.base_poly)
                    legal = True
                    for placed in layout:
                        if strict_overlap(cand.poly, placed.poly):
                            legal = False
                            break
                    if legal:
                        layout.append(cand)
                        ok = True
                        break
                if ok:
                    break

        if len(layout) < n:
            tries = 0
            while len(layout) < n and tries < 2000:
                tries += 1
                x = random.uniform(0.0, self.s_target)
                y = random.uniform(0.0, self.s_target)
                deg = random.uniform(0.0, 360.0)
                cand = TreeInstance.from_pose(x, y, deg, base_poly=self.base_poly)
                if self.is_tree_legal(cand):
                    layout.append(cand)
        self.trees = layout[:n]
        self.selected = 0 if self.trees else None
        self.reset_best_tracking()
        self.stats_dirty = True
        self.update_stats()
        self.align_to_square_origin()
        self.s_target = None
        self.target_n = n
        self.input_text = str(self.target_n)
        self.set_status(f"Auto placed N={n}.")

    # ---------- IO ----------
    def safe_write_text(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
            return path, False
        except OSError:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            try:
                fallback.write_text(text, encoding="utf-8")
                return fallback, True
            except OSError:
                return None, True

    def layout_to_dicts(self, layout):
        items = []
        for item in layout:
            if isinstance(item, TreeInstance):
                x, y, deg = item.x, item.y, item.deg
            elif isinstance(item, dict):
                x, y, deg = item.get("x", 0.0), item.get("y", 0.0), item.get("deg", 0.0)
            else:
                x, y, deg = item
            items.append({"x": x, "y": y, "deg": deg})
        return items

    def build_submission_rows(self, layout, n_override=None):
        n = n_override if n_override is not None else len(layout)
        rows = []
        for idx, item in enumerate(layout):
            if isinstance(item, TreeInstance):
                x, y, deg = item.x, item.y, item.deg
            elif isinstance(item, dict):
                x, y, deg = item.get("x", 0.0), item.get("y", 0.0), item.get("deg", 0.0)
            else:
                x, y, deg = item
            tid = format_id(n, idx)
            rows.append(
                f"{tid},{format_submission_value(x)},{format_submission_value(y)},{format_submission_value(deg)}"
            )
        return rows

    def write_submission_csv(self, path, layout, n_override=None):
        rows = self.build_submission_rows(layout, n_override=n_override)
        text = "id,x,y,deg\n" + "\n".join(rows)
        return self.safe_write_text(path, text)

    def append_submission_csv(self, path, layout, n_override=None):
        rows = self.build_submission_rows(layout, n_override=n_override)
        if not rows:
            return None, True
        header = "id,x,y,deg"
        body = "\n".join(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not path.exists():
                return self.safe_write_text(path, f"{header}\n{body}")
            try:
                if path.stat().st_size == 0:
                    return self.safe_write_text(path, f"{header}\n{body}")
            except OSError:
                pass
            need_newline = False
            try:
                with path.open("rb") as f:
                    f.seek(-1, os.SEEK_END)
                    need_newline = f.read(1) != b"\n"
            except OSError:
                need_newline = True
            with path.open("a", encoding="utf-8") as f:
                if need_newline:
                    f.write("\n")
                f.write(body)
            return path, False
        except OSError:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            self.safe_write_text(fallback, f"{header}\n{body}")
            return fallback, True

    def load_layout_from_csv(self, path):
        try:
            with path.open("r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
        except OSError:
            return []
        if len(lines) <= 1:
            return []
        header = lines[0].strip().split(",")
        has_s = len(header) >= 5 and header[1].strip().lower() == "s"
        x_idx = 2 if has_s else 1
        y_idx = 3 if has_s else 2
        deg_idx = 4 if has_s else 3
        layout = []
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) <= deg_idx:
                continue
            x = parse_submission_value(parts[x_idx])
            y = parse_submission_value(parts[y_idx])
            deg = parse_submission_value(parts[deg_idx])
            layout.append((x, y, deg))
        return layout

    def load_layout_from_master(self, path, n):
        try:
            with Path(path).open("r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
        except OSError:
            return []
        if len(lines) <= 1:
            return []
        header = lines[0].strip().split(",")
        if len(header) < 4:
            return []
        id_idx = 0
        x_idx = 1
        y_idx = 2
        deg_idx = 3
        layout = []
        prefix = f"{int(n):03d}_"
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) <= deg_idx:
                continue
            if not parts[id_idx].startswith(prefix):
                continue
            x = parse_submission_value(parts[x_idx])
            y = parse_submission_value(parts[y_idx])
            deg = parse_submission_value(parts[deg_idx])
            layout.append((x, y, deg))
        return layout

    def find_latest_run_for_n(self, n):
        if not self.outputs_runs.exists():
            return None
        pattern = f"{n:03d}__*.csv"
        candidates = list(self.outputs_runs.glob(pattern))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def load_layout_for_n(self, n):
        n = int(n)
        if n < 1 or n > self.submission_max_n:
            self.set_status(f"N must be 1..{self.submission_max_n} for load.")
            return
        layout = None
        source = None
        if LOAD_MASTER_PATH:
            layout = self.load_layout_from_master(LOAD_MASTER_PATH, n)
            if layout:
                source = "comparison"
        if layout is None:
            best_path = self.outputs_best / f"{n:03d}.csv"
            if best_path.exists():
                layout = self.load_layout_from_csv(best_path)
                if layout:
                    source = "best"
        if layout is None:
            run_path = self.find_latest_run_for_n(n)
            if run_path is not None:
                layout = self.load_layout_from_csv(run_path)
                if layout:
                    source = "run"
        if not layout:
            self.set_status(f"No layout found for N={n}.")
            self.set_workspace_lines([f"Load N={n}: not found."])
            return
        if self.shape_name != "tree":
            self.set_shape("tree")
        layout_dicts = self.layout_to_dicts(layout)
        self.apply_layout(layout_dicts)
        self.target_n = n
        self.input_text = str(n)
        if self.shape_name == "tree":
            self.update_stats()
            s_now = float(self.stats.get("s", 0.0))
            if not self.canonical_csv.exists():
                self.initialize_canonical_csv()
            self.update_canonical_from_layout(self.trees, s_now)
        label = source if source is not None else "unknown"
        self.set_status(f"Loaded N={n} from {label}.")
        self.set_workspace_lines([f"Loaded N={n} from {label}."])

    def export_layout(self):
        layout = self.layout_to_dicts(self.trees)
        json_text = json.dumps(layout, indent=2)
        json_path, _ = self.safe_write_text(self.out_json, json_text)
        csv_path, csv_fallback = self.append_submission_csv(self.out_csv, self.trees)
        if csv_path is None or json_path is None:
            self.set_status("Export failed.")
            return
        self.last_export = csv_path.name
        if csv_fallback:
            self.set_status(f"Exported {csv_path.name} (append fallback).")
        else:
            self.set_status(f"Exported {csv_path.name} (appended).")

    def import_layout(self):
        if not self.out_json.exists():
            self.set_status("layout.json not found.")
            return
        with self.out_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.trees = [TreeInstance.from_pose(d["x"], d["y"], d["deg"], base_poly=self.base_poly) for d in data]
        self.selected = 0 if self.trees else None
        self.reset_best_tracking()
        self.stats_dirty = True
        self.target_n = len(self.trees)
        self.input_text = str(self.target_n)
        self.set_status(f"Imported {self.out_json.name}.")

    def save_snapshot_layout(self, layout, s_val=None):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = self.snapshots_dir / f"snapshot_{stamp}.json"
        csv_path = self.snapshots_dir / f"snapshot_{stamp}.csv"
        log_path = self.snapshots_dir / f"snapshot_{stamp}.txt"

        json_text = json.dumps(self.layout_to_dicts(layout), indent=2)
        self.safe_write_text(json_path, json_text)
        self.write_submission_csv(csv_path, layout)

        if s_val is None:
            s_val = self.compute_s_from_layout(layout, base_poly=BASE_POLY) or 0.0
        lines = [
            f"time={time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"n={len(layout)}",
            f"s={s_val:.9f}",
        ]
        self.safe_write_text(log_path, "\n".join(lines))
        self.last_snapshot = f"snapshots/{csv_path.name}"
        return csv_path

    def save_snapshot(self):
        if not self.trees:
            self.set_status("Nothing to snapshot.")
            return
        if self.layout_has_overlap(self.trees):
            self.set_status("Overlap detected. Snapshot canceled.")
            return
        snap_path = self.save_snapshot_layout(self.trees)
        if snap_path is not None:
            self.set_status(f"Snapshot saved: {snap_path.name}.")

    def ensure_outputs_dirs(self):
        self.outputs_base.mkdir(parents=True, exist_ok=True)
        self.outputs_best.mkdir(parents=True, exist_ok=True)
        self.outputs_runs.mkdir(parents=True, exist_ok=True)

    def export_current_csv(self, path, layout=None, n_override=None):
        layout = self.trees if layout is None else layout
        return self.write_submission_csv(path, layout, n_override=n_override)

    def compute_s_from_layout(self, layout, base_poly=None):
        base = self.base_poly if base_poly is None else base_poly
        polys = []
        for item in layout:
            if isinstance(item, TreeInstance):
                x, y, deg = item.x, item.y, item.deg
            elif isinstance(item, dict):
                x, y, deg = item.get("x", 0.0), item.get("y", 0.0), item.get("deg", 0.0)
            else:
                x, y, deg = item
            inst = TreeInstance.from_pose(x, y, deg, base_poly=base)
            polys.append(inst.poly)
        if not polys:
            return None
        return self.compute_s_for_polys(polys)

    def layout_has_overlap(self, layout, base_poly=None):
        base = self.base_poly if base_poly is None else base_poly
        polys = []
        for item in layout:
            if isinstance(item, TreeInstance):
                poly = item.poly
            elif isinstance(item, dict):
                poly = TreeInstance.from_pose(
                    item.get("x", 0.0),
                    item.get("y", 0.0),
                    item.get("deg", 0.0),
                    base_poly=base,
                ).poly
            else:
                x, y, deg = item
                poly = TreeInstance.from_pose(x, y, deg, base_poly=base).poly
            polys.append(poly)
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if strict_overlap(polys[i], polys[j]):
                    return True
        return False

    def compute_s_from_csv(self, path):
        try:
            with path.open("r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
            if not lines or len(lines) <= 1:
                return None
            header = lines[0].strip().split(",")
            has_s = len(header) >= 5 and header[1].strip().lower() == "s"
            x_idx = 2 if has_s else 1
            y_idx = 3 if has_s else 2
            deg_idx = 4 if has_s else 3
            polys = []
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) <= deg_idx:
                    continue
                x = parse_submission_value(parts[x_idx])
                y = parse_submission_value(parts[y_idx])
                deg = parse_submission_value(parts[deg_idx])
                inst = TreeInstance.from_pose(x, y, deg, base_poly=self.base_poly)
                polys.append(inst.poly)
            if not polys:
                return None
            return self.compute_s_for_polys(polys)
        except Exception:
            return None

    def record_output(self):
        if not self.trees and not self.opt_best_layout:
            self.set_status("Nothing to record.")
            return
        if self.shape_name != "tree":
            self.set_status("Record is only available in tree mode.")
            self.set_workspace_lines(["Record canceled: shape is not tree."])
            return
        layout = self.opt_best_layout if self.opt_best_layout else self.trees
        if self.layout_has_overlap(layout, base_poly=BASE_POLY):
            self.set_status("Overlap detected. Record canceled.")
            self.set_workspace_lines(["Record canceled: overlap detected."])
            return
        n = len(layout)
        s_after = self.opt_best_s
        if s_after is None:
            s_after = float(self.compute_s_from_layout(layout, base_poly=BASE_POLY) or 0.0)

        workspace = [f"Record N={n}  S={s_after:.4f}"]

        if not self.canonical_csv.exists():
            self.initialize_canonical_csv()

        layout_text = json.dumps(self.layout_to_dicts(layout), indent=2)
        json_path, _ = self.safe_write_text(self.out_json, layout_text)
        csv_path, csv_fallback = self.append_submission_csv(self.out_csv, layout)
        if csv_path is None:
            workspace.append("layout_kaggle.csv: write failed (locked).")
        else:
            suffix = " (fallback)" if csv_fallback else " (append)"
            workspace.append(f"layout_kaggle.csv: {csv_path.name}{suffix}")

        self.ensure_outputs_dirs()
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        eps = 1e-6
        best_path = self.outputs_best / f"{n:03d}.csv"
        if best_path.exists():
            prev_s = self.compute_s_from_csv(best_path)
            s_before = prev_s if prev_s is not None else float("inf")
        else:
            s_before = float("inf")

        run_path = self.outputs_runs / f"{n:03d}__{ts}__s{s_after:.6f}.csv"
        run_written, _ = self.export_current_csv(run_path, layout=layout, n_override=n)
        if run_written is None:
            workspace.append("best: write failed.")
        else:
            run_path = run_written
            improved = 0
            if s_after < s_before - eps:
                best_text = run_path.read_text(encoding="utf-8")
                self.safe_write_text(best_path, best_text)
                improved = 1
                workspace.append("best: updated")
            else:
                workspace.append("best: not improved")

            ledger_exists = self.outputs_ledger.exists()
            with self.outputs_ledger.open("a", encoding="utf-8") as f:
                if not ledger_exists:
                    f.write("timestamp,N,s_before,s_after,improved,export_filename\n")
                f.write(f"{ts},{n},{s_before:.9f},{s_after:.9f},{improved},{run_path}\n")
            self.last_record = f"runs/{run_path.name}"

        snap_path = self.save_snapshot_layout(layout, s_val=s_after)
        if snap_path is not None:
            workspace.append(f"snapshot: {snap_path.name}")

        canonical_note = self.update_canonical_from_layout(layout, s_after)
        if canonical_note:
            workspace.append(canonical_note)

        self.set_workspace_lines(workspace)
        self.set_status("Record complete.")

    # ---------- canonical submission ----------
    def canonical_layout_is_zero(self, layout):
        for item in layout:
            if isinstance(item, dict):
                x, y, deg = item.get("x", 0.0), item.get("y", 0.0), item.get("deg", 0.0)
            else:
                x, y, deg = item
            if abs(x) > 1e-9 or abs(y) > 1e-9 or abs(deg) > 1e-9:
                return False
        return True

    def load_canonical_csv(self):
        self.canonical_layouts = {}
        self.canonical_s = {}
        self.canonical_loaded = False
        if not self.canonical_csv.exists():
            return
        try:
            with self.canonical_csv.open("r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
        except OSError:
            return
        if len(lines) <= 1:
            return
        header = lines[0].strip().split(",")
        has_s = len(header) >= 5 and header[1].strip().lower() == "s"
        x_idx = 2 if has_s else 1
        y_idx = 3 if has_s else 2
        deg_idx = 4 if has_s else 3
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) <= deg_idx:
                continue
            parsed = parse_id(parts[0])
            if not parsed:
                continue
            n, idx = parsed
            if n < 1 or n > self.submission_max_n or idx < 0 or idx >= n:
                continue
            layout = self.canonical_layouts.get(n)
            if layout is None:
                layout = [None] * n
                self.canonical_layouts[n] = layout
            x = parse_submission_value(parts[x_idx])
            y = parse_submission_value(parts[y_idx])
            deg = parse_submission_value(parts[deg_idx])
            layout[idx] = (x, y, deg)

        for n in range(1, self.submission_max_n + 1):
            layout = self.canonical_layouts.get(n)
            if layout is None:
                layout = [(0.0, 0.0, 0.0)] * n
                self.canonical_layouts[n] = layout
            else:
                for i, val in enumerate(layout):
                    if val is None:
                        layout[i] = (0.0, 0.0, 0.0)
        self.canonical_loaded = True

    def build_canonical_text(self):
        rows = ["id,x,y,deg"]
        for n in range(1, self.submission_max_n + 1):
            layout = self.canonical_layouts.get(n)
            if layout is None or len(layout) != n:
                layout = [(0.0, 0.0, 0.0)] * n
                self.canonical_layouts[n] = layout
            for idx, item in enumerate(layout):
                if isinstance(item, dict):
                    x, y, deg = item.get("x", 0.0), item.get("y", 0.0), item.get("deg", 0.0)
                else:
                    x, y, deg = item
                rows.append(
                    f"{format_id(n, idx)},{format_submission_value(x)},{format_submission_value(y)},"
                    f"{format_submission_value(deg)}"
                )
        return "\n".join(rows)

    def write_canonical_csv(self):
        text = self.build_canonical_text()
        return self.safe_write_text(self.canonical_csv, text)

    def initialize_canonical_csv(self):
        self.canonical_layouts = {}
        for n in range(1, self.submission_max_n + 1):
            self.canonical_layouts[n] = [(0.0, 0.0, 0.0)] * n
        self.canonical_s = {}
        canonical_path, fallback = self.write_canonical_csv()
        if canonical_path is None:
            self.canonical_loaded = False
            self.set_status("Canonical init failed.")
            return
        self.canonical_loaded = True
        if fallback:
            self.set_status(f"Canonical initialized (locked): {canonical_path.name}.")
        else:
            self.set_status("Canonical initialized.")

    def get_canonical_s(self, n):
        if not self.canonical_loaded:
            return None
        if n < 1 or n > self.submission_max_n:
            return None
        if n in self.canonical_s:
            return self.canonical_s[n]
        layout = self.canonical_layouts.get(n)
        if not layout:
            return None
        if self.canonical_layout_is_zero(layout):
            self.canonical_s[n] = 0.0
            return 0.0
        s_val = self.compute_s_from_layout(layout, base_poly=BASE_POLY)
        if s_val is None:
            return None
        self.canonical_s[n] = float(s_val)
        return self.canonical_s[n]

    def import_best_to_canonical(self):
        if not self.canonical_csv.exists():
            self.set_status("Canonical CSV missing. Init CSV first.")
            return
        if not self.canonical_loaded:
            self.load_canonical_csv()
        n = self.target_n
        if n < 1 or n > self.submission_max_n:
            self.set_status(f"N must be 1..{self.submission_max_n} for canonical.")
            return
        best_path = self.outputs_best / f"{n:03d}.csv"
        if not best_path.exists():
            self.set_status(f"No recorded best for N={n}. Log best first.")
            return
        layout = self.load_layout_from_csv(best_path)
        if len(layout) != n:
            self.set_status(f"Best layout mismatch for N={n}.")
            return
        self.canonical_layouts[n] = layout
        s_val = self.compute_s_from_layout(layout, base_poly=BASE_POLY)
        if s_val is not None:
            self.canonical_s[n] = float(s_val)
        canonical_path, fallback = self.write_canonical_csv()
        if canonical_path is None:
            self.set_status("Canonical update failed.")
            return
        self.canonical_loaded = True
        if fallback:
            self.set_status(f"Canonical updated (locked): {canonical_path.name}.")
        else:
            self.set_status("Canonical updated.")

    def update_canonical_from_layout(self, layout, s_val=None):
        if not self.canonical_csv.exists():
            return "canonical: missing"
        if not self.canonical_loaded:
            self.load_canonical_csv()
        if self.layout_has_overlap(layout, base_poly=BASE_POLY):
            return "canonical: overlap (skipped)"
        n = len(layout)
        if n < 1 or n > self.submission_max_n:
            return f"canonical: N out of range"
        if s_val is None:
            s_val = self.compute_s_from_layout(layout, base_poly=BASE_POLY)
        if s_val is None:
            return "canonical: update failed"
        s_old = self.get_canonical_s(n)
        if s_old is not None and s_old > 0.0 and s_val >= s_old - 1e-9:
            return "canonical: not improved"
        new_layout = []
        for item in layout:
            if isinstance(item, TreeInstance):
                new_layout.append((item.x, item.y, item.deg))
            elif isinstance(item, dict):
                new_layout.append((item.get("x", 0.0), item.get("y", 0.0), item.get("deg", 0.0)))
            else:
                new_layout.append(item)
        self.canonical_layouts[n] = new_layout
        self.canonical_s[n] = float(s_val)
        canonical_path, fallback = self.write_canonical_csv()
        if canonical_path is None:
            return "canonical: write failed"
        self.canonical_loaded = True
        if fallback:
            return "canonical: updated (locked)"
        return "canonical: updated"

    def update_canonical_from_current(self):
        if not self.trees and not self.opt_best_layout:
            self.set_status("Nothing to update.")
            return
        layout = self.opt_best_layout if self.opt_best_layout else self.trees
        note = self.update_canonical_from_layout(layout)
        self.set_status(note or "Canonical updated.")

    # ---------- grid seeding ----------
    def find_grid_path(self):
        root = self.repo_root
        candidates = [
            root / "v6" / "canonical_v6.xlsx",
            root / "v72" / "v6" / "canonical_v6.xlsx",
            root / "agent" / "v72" / "v6" / "canonical_v6.xlsx",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def load_grid_table(self):
        self.grid_table = {}
        self.grid_loaded = False
        if pd is None:
            return
        if self.grid_path is None or not self.grid_path.exists():
            return
        try:
            df = pd.read_excel(self.grid_path, sheet_name="Parallelogram_targets")
            for _, row in df.iterrows():
                m = int(row.get("N_parallelogram", 0))
                self.grid_table[m] = {
                    "s_star": float(row.get("s_star", 0.0)),
                    "C": int(row.get("C_cols", 0)),
                    "R": int(row.get("R_rows", 0)),
                    "w_used": float(row.get("w_used", self.base_poly.bounds[2] - self.base_poly.bounds[0])),
                    "h_used": float(row.get("h_used", self.base_poly.bounds[3] - self.base_poly.bounds[1])),
                }
            self.grid_loaded = bool(self.grid_table)
        except Exception:
            self.grid_table = {}
            self.grid_loaded = False

    def place_from_grid(self, n_triangles):
        if self.shape_name not in ("triangle", "parallelogram"):
            return False
        if not self.grid_table:
            return False
        if self.shape_name == "triangle":
            m = math.ceil(n_triangles / 2)
            n_place = n_triangles
            base_poly = self.base_poly
            orient_fn = lambda r, c: (0.0 if (r + c) % 2 == 0 else 180.0)
        else:
            m = n_triangles
            n_place = m
            base_poly = self.para_poly
            orient_fn = lambda r, c: 0.0
        entry = self.grid_table.get(m)
        if not entry:
            return False
        C = max(1, entry.get("C", 1))
        R = max(1, entry.get("R", 1))
        w = entry.get("w_used", self.base_poly.bounds[2] - self.base_poly.bounds[0])
        h = entry.get("h_used", self.base_poly.bounds[3] - self.base_poly.bounds[1])
        self.trees = []
        idx = 0
        for r in range(R):
            y = r * h
            x_offset = 0.0 if (r % 2 == 0) else 0.5 * w
            for c in range(C):
                if idx >= n_place:
                    break
                x = x_offset + c * w
                deg = orient_fn(r, c)
                self.trees.append(TreeInstance.from_pose(x, y, deg, base_poly=base_poly))
                idx += 1
            if idx >= n_place:
                break
        if not self.trees:
            return False
        minx, miny, _, _ = bbox_polys([t.poly for t in self.trees])
        if minx != 0.0 or miny != 0.0:
            for t in self.trees:
                t.set_pose(t.x - minx, t.y - miny, t.deg)
        self.selected = 0
        self.stats_dirty = True
        self.s_target = entry.get("s_star", None)
        self.align_to_square_origin()
        return True

    # ---------- view ----------
    def align_to_square_origin(self):
        if not self.trees:
            return
        ox, oy = self.square_origin
        minx, miny, maxx, maxy = bbox_polys([t.poly for t in self.trees])
        dx, dy = ox - minx, oy - miny
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return
        for t in self.trees:
            t.set_pose(t.x + dx, t.y + dy, t.deg)
        self.stats_dirty = True

    def frame_view(self):
        if not self.trees:
            self.camera = [0.0, 0.0]
            self.scale = self.default_scale
            return
        minx, miny, maxx, maxy = bbox_polys([t.poly for t in self.trees])
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
        self.camera = [cx, cy]
        w = maxx - minx
        h = maxy - miny
        if w <= 0 or h <= 0:
            return
        scale_x = (self.view_rect.w * 0.72) / w
        scale_y = (self.view_rect.h * 0.72) / h
        self.scale = clamp(min(scale_x, scale_y), self.zoom_limits[0], self.zoom_limits[1])

    def reset_view(self):
        self.camera = [0.0, 0.0]
        self.scale = self.default_scale
        self.set_status("View reset.")

    # ---------- UI drawing ----------
    def draw_button(self, rect, label, active=False, disabled=False):
        mx, my = pygame.mouse.get_pos()
        hover = rect.collidepoint(mx, my)
        fill = self.theme["panel_inner"]
        border = self.theme["panel_border"]
        text = self.theme["text"]

        if disabled:
            fill = scale_color(fill, 0.75)
            text = self.theme["text_muted"]
        elif active:
            fill = scale_color(self.theme["accent_dim"], 1.05)
            border = self.theme["accent"]
        elif hover:
            fill = scale_color(fill, 1.15)

        pygame.draw.rect(self.screen, fill, rect)
        pygame.draw.rect(self.screen, border, rect, 1)

        font = self.font_small if rect.w < 90 or len(label) <= 4 else self.font_ui
        surf = font.render(label, True, text)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def draw_top_bar(self, fps):
        pygame.draw.rect(self.screen, self.theme["panel"], self.top_rect)
        pygame.draw.line(self.screen, self.theme["panel_border"], (0, self.top_rect.bottom - 1), (self.size[0], self.top_rect.bottom - 1))

        title = self.font_title.render(f"Tree Packing Studio {VERSION_TAG}", True, self.theme["text"])
        self.screen.blit(title, (self.margin, self.top_rect.y + 18))
        placed_n = len(self.trees)
        meta = f"Placed: {placed_n}  |  Target: {self.target_n}"
        meta_surf = self.font_small.render(meta, True, self.theme["text_muted"])
        self.screen.blit(meta_surf, (self.margin + title.get_width() + 16, self.top_rect.y + 22))

        self.draw_button(self.buttons["frame"], self.button_labels["frame"])
        self.draw_button(self.buttons["reset_view"], self.button_labels["reset_view"])

        status_parts = []
        if self.optimizing:
            remaining = max(0.0, self.opt_end_time - time.time())
            status_parts.append(f"Optimizing {remaining:.1f}s")
        else:
            status_parts.append("Idle")
        if self.opt_best_s is not None:
            status_parts.append(f"Best {self.opt_best_s:.4f}")
        status_parts.append(f"FPS {fps:.0f}")
        status_text = "  |  ".join(status_parts)
        status_surf = self.font_small.render(status_text, True, self.theme["accent_alt"])
        status_right = getattr(self, "top_action_left", self.size[0] - self.margin)
        status_x = max(self.margin, status_right - 12 - status_surf.get_width())
        self.screen.blit(status_surf, (status_x, self.top_rect.y + 22))

    def draw_bottom_bar(self):
        pygame.draw.rect(self.screen, self.theme["panel"], self.bottom_rect)
        pygame.draw.line(self.screen, self.theme["panel_border"], (0, self.bottom_rect.y), (self.size[0], self.bottom_rect.y))

        for name in ["generate", "auto", "opt", "auto_opt", "record"]:
            active = name == "opt" and self.optimizing
            self.draw_button(self.buttons[name], self.button_labels[name], active=active)

        pygame.draw.rect(self.screen, self.theme["panel_alt"], self.n_panel_rect)
        pygame.draw.rect(self.screen, self.theme["panel_border"], self.n_panel_rect, 1)
        placed_label = self.font_small.render("Placed", True, self.theme["text_muted"])
        self.screen.blit(placed_label, (self.n_panel_rect.x + 12, self.n_panel_rect.y + 2))

        pygame.draw.rect(self.screen, self.theme["panel_inner"], self.n_value_rect)
        pygame.draw.rect(self.screen, self.theme["panel_border"], self.n_value_rect, 1)
        placed_text = self.font_ui.render(str(len(self.trees)), True, self.theme["text"])
        self.screen.blit(placed_text, placed_text.get_rect(center=self.n_value_rect.center))

        placed_n = len(self.trees)
        best_n = placed_n if placed_n > 0 else self.target_n
        canonical_s = self.get_canonical_s(best_n)
        best_text = "-" if canonical_s is None else f"{canonical_s:.4f}"
        target_text = f"Target N (1..400)  |  High Score S: {best_text}"
        target_label = self.font_small.render(target_text, True, self.theme["text_muted"])
        self.screen.blit(target_label, (self.n_panel_rect.x + 12, self.n_value_rect.bottom + 4))

        input_border = self.theme["accent"] if self.input_active else self.theme["panel_border"]
        pygame.draw.rect(self.screen, self.theme["panel_inner"], self.n_input_rect)
        pygame.draw.rect(self.screen, input_border, self.n_input_rect, 1)
        text_color = self.theme["text"] if self.input_active else self.theme["text_muted"]
        surf = self.font_ui.render(self.input_text, True, text_color)
        self.screen.blit(surf, (self.n_input_rect.x + 10, self.n_input_rect.y + 6))

        for name in ["n_minus", "n_plus", "load_n"]:
            self.draw_button(self.buttons[name], self.button_labels[name])

        auto_label = self.font_small.render("Auto passes (dropdown)", True, self.theme["text_muted"])
        self.screen.blit(auto_label, self.auto_pass_label_pos)
        auto_border = self.theme["accent"] if self.auto_pass_menu_open else self.theme["panel_border"]
        pygame.draw.rect(self.screen, self.theme["panel_inner"], self.auto_pass_rect)
        pygame.draw.rect(self.screen, auto_border, self.auto_pass_rect, 1)
        auto_text = "auto" if self.auto_pass_override is None else str(self.auto_pass_override)
        auto_color = self.theme["text"]
        auto_surf = self.font_ui.render(auto_text, True, auto_color)
        self.screen.blit(auto_surf, (self.auto_pass_rect.x + 10, self.auto_pass_rect.y + 4))
        if self.auto_pass_menu_open:
            pygame.draw.rect(self.screen, self.theme["panel_inner"], self.auto_pass_menu_rect)
            pygame.draw.rect(self.screen, self.theme["panel_border"], self.auto_pass_menu_rect, 1)
            opt_y = self.auto_pass_menu_rect.y
            for opt in self.auto_pass_options:
                label = "auto" if opt is None else str(opt)
                opt_rect = pygame.Rect(
                    self.auto_pass_menu_rect.x,
                    opt_y,
                    self.auto_pass_menu_rect.w,
                    20,
                )
                pygame.draw.rect(self.screen, self.theme["panel_inner"], opt_rect)
                surf = self.font_small.render(label, True, self.theme["text"])
                self.screen.blit(surf, (opt_rect.x + 8, opt_rect.y + 2))
                opt_y += 20

        csv_label = self.font_small.render("Load CSV (comparison)", True, self.theme["text_muted"])
        self.screen.blit(csv_label, self.csv_label_pos)
        csv_border = self.theme["accent"] if self.csv_name_active else self.theme["panel_border"]
        pygame.draw.rect(self.screen, self.theme["panel_inner"], self.csv_name_rect)
        pygame.draw.rect(self.screen, csv_border, self.csv_name_rect, 1)
        csv_color = self.theme["text"] if self.csv_name_active else self.theme["text_muted"]
        csv_text = self.csv_name_text or ""
        csv_surf = self.font_small.render(csv_text, True, csv_color)
        self.screen.blit(csv_surf, (self.csv_name_rect.x + 8, self.csv_name_rect.y + 3))
        self.draw_button(self.buttons["csv_set"], self.button_labels["csv_set"])

        pygame.draw.rect(self.screen, self.theme["panel_alt"], self.view_panel_rect)
        pygame.draw.rect(self.screen, self.theme["panel_border"], self.view_panel_rect, 1)
        view_label = self.font_small.render("Workspace", True, self.theme["text_muted"])
        self.screen.blit(view_label, (self.view_panel_rect.x + 12, self.view_panel_rect.y + 6))
        info_y = self.view_panel_rect.y + 26
        for line in (self.workspace_lines or ["Ready."]):
            surf = self.font_small.render(line, True, self.theme["text"])
            self.screen.blit(surf, (self.view_panel_rect.x + 12, info_y))
            info_y += 18

        status_msg = "Ready."
        if self.status_msg and (time.time() - self.status_time) < 6.0:
            status_msg = self.status_msg
        status_surf = self.font_small.render(status_msg, True, self.theme["text_muted"])
        self.screen.blit(status_surf, (self.status_rect.x + 6, self.status_rect.y + 8))

    def draw_right_panel(self, fps):
        pygame.draw.rect(self.screen, self.theme["panel_alt"], self.right_rect)
        pygame.draw.rect(self.screen, self.theme["panel_border"], self.right_rect, 1)

        x = self.right_rect.x + 16
        y = self.right_rect.y + 16
        w = self.right_rect.w - 32

        def draw_card(y_pos, title, lines):
            pad = 10
            line_h = 18
            header_h = 18
            height = pad * 2 + header_h + line_h * len(lines)
            rect = pygame.Rect(x, y_pos, w, height)
            pygame.draw.rect(self.screen, self.theme["panel"], rect)
            pygame.draw.rect(self.screen, self.theme["panel_border"], rect, 1)
            title_surf = self.font_small.render(title.upper(), True, self.theme["accent"])
            self.screen.blit(title_surf, (rect.x + pad, rect.y + pad))
            ty = rect.y + pad + header_h
            for line in lines:
                surf = self.font_small.render(line, True, self.theme["text"])
                self.screen.blit(surf, (rect.x + pad, ty))
                ty += line_h
            return rect.bottom + 12

        st = self.stats
        bbox = st.get("bbox", (0, 0, 0, 0))
        stats_lines = [
            f"N: {st.get('n', 0)}",
            f"S (side): {st.get('s', 0.0):.4f}",
            f"Term (S^2/N): {st.get('term', 0.0):.4f}",
            f"Union (area): {st.get('union', 0.0):.4f}",
            f"Empty (area): {st.get('empty', 0.0):.4f}",
            f"Density (U/S^2): {st.get('density', 0.0):.3f}",
            f"Contacts: {self.contact_count}",
            f"BBox: {bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}",
        ]
        y = draw_card(y, "Stats", stats_lines)

        sel_lines = []
        if self.sel_stats:
            sel_lines = [
                f"Area: {self.sel_stats.get('area', 0.0):.4f}",
                f"Occ: {self.sel_stats.get('occ', 0.0):.4f}",
                f"Empty: {self.sel_stats.get('empty', 0.0):.4f}",
                f"Ratio: {self.sel_stats.get('ratio', 0.0):.3f}",
            ]
        else:
            sel_lines = ["No selection."]
        y = draw_card(y, "Selection", sel_lines)

        output_lines = [
            f"Record: {self.last_record or '-'}",
            f"Live CSV: {self.out_csv.name}",
            f"Canonical: {self.canonical_csv.name if self.canonical_loaded else 'missing'}",
        ]
        y = draw_card(y, "Output", output_lines)

        placed_n = len(self.trees)
        best_n = placed_n if placed_n > 0 else self.target_n
        canonical_s = self.get_canonical_s(best_n)
        canonical_line = "High Score S: -"
        if canonical_s is not None:
            canonical_line = f"High Score S: {canonical_s:.4f}"
        live_s = self.stats.get("s", None)
        live_line = "Live S: -" if live_s is None else f"Live S: {live_s:.4f}"
        best_line = "Current Best S: -" if self.opt_best_s is None else f"Current Best S: {self.opt_best_s:.4f}"
        live_term = None
        if live_s is not None and placed_n > 0:
            live_term = (live_s * live_s) / placed_n
        best_term = None
        if self.opt_best_s is not None and placed_n > 0:
            best_term = (self.opt_best_s * self.opt_best_s) / placed_n
        high_term = None
        if canonical_s is not None and placed_n > 0:
            high_term = (canonical_s * canonical_s) / placed_n
        live_term_line = "Live S^2/N: -" if live_term is None else f"Live S^2/N: {live_term:.4f}"
        best_term_line = "Best S^2/N: -" if best_term is None else f"Best S^2/N: {best_term:.4f}"
        high_term_line = "High S^2/N: -" if high_term is None else f"High S^2/N: {high_term:.4f}"
        session_lines = [
            f"Placed N: {placed_n}",
            f"Optimizer: {'on' if self.optimizing else 'off'}",
            live_line,
            best_line,
            canonical_line,
            live_term_line,
            best_term_line,
            high_term_line,
            f"FPS: {fps:.1f}",
        ]
        draw_card(y, "Session", session_lines)

    def draw_view(self):
        pygame.draw.rect(self.screen, self.theme["view_bg"], self.view_rect)
        self.screen.blit(self.view_grid, self.view_rect.topleft)
        pygame.draw.rect(self.screen, self.theme["panel_border"], self.view_rect, 1)

        clip_prev = self.screen.get_clip()
        self.screen.set_clip(self.view_rect)

        if self.trees:
            s_draw = self.current_square_side()
            ox, oy = self.square_origin
            sq = Polygon([(ox, oy), (ox + s_draw, oy), (ox + s_draw, oy + s_draw), (ox, oy + s_draw)])
            pts = polygon_to_screen(sq, self.world_to_screen)
            pygame.draw.lines(self.screen, self.theme["accent"], True, pts, 2)

        state = self.selected_overlap_state()
        for i, t in enumerate(self.trees):
            pts = polygon_to_screen(t.poly, self.world_to_screen)
            shadow_pts = [(x + 3, y + 3) for x, y in pts]
            pygame.draw.polygon(self.screen, self.theme["shadow"], shadow_pts)

            if i == self.selected:
                if state == "overlap":
                    color = self.theme["danger"]
                elif state == "touch":
                    color = self.theme["accent_alt"]
                else:
                    color = self.theme["ok"]
            else:
                color = self.theme["accent"]
            pygame.draw.polygon(self.screen, color, pts)
            pygame.draw.polygon(self.screen, (10, 12, 16), pts, 1)

        if self.selecting and self.sel_start and self.sel_end:
            x0, y0 = self.sel_start
            x1, y1 = self.sel_end
            p0 = self.world_to_screen(x0, y0)
            p1 = self.world_to_screen(x1, y1)
            rect = pygame.Rect(min(p0[0], p1[0]), min(p0[1], p1[1]), abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))
            pygame.draw.rect(self.screen, self.theme["accent_alt"], rect, 1)

        self.screen.set_clip(clip_prev)

    def draw_help(self):
        if not self.show_help:
            return
        help_lines = [
            "LMB drag: move  |  RMB drag: select  |  Wheel: rotate  |  Shift/Ctrl: fine/coarse rotate",
            "Ctrl+Wheel: zoom to cursor  |  +/-: zoom",
            "G: generate  |  P: auto place  |  O: optimize  |  T: auto optimize",
            "L: record instance (log + snapshot + canonical + append CSV)",
            "K: load N from comparison  |  Set CSV: type name + Set",
            "F9: toggle automation (2..200)",
            "Alt+Arrows: nudge selected  |  [ / ]: rotate selected",
            "M: pan mode  |  A: add tree  |  Tab: cycle  |  Del: remove  |  C: snap last drag",
            "Arrows: pan  |  F: frame  |  H: toggle help",
        ]
        pad = 10
        line_h = 18
        width = max(self.font_small.size(line)[0] for line in help_lines) + pad * 2
        height = line_h * len(help_lines) + pad * 2
        x = self.view_rect.x + 12
        y = self.view_rect.bottom - height - 12
        box = pygame.Surface((width, height), pygame.SRCALPHA)
        box.fill((12, 14, 20, 200))
        self.screen.blit(box, (x, y))
        for idx, line in enumerate(help_lines):
            surf = self.font_small.render(line, True, self.theme["text_muted"])
            self.screen.blit(surf, (x + pad, y + pad + idx * line_h))

    def draw(self, fps):
        self.screen.blit(self.bg_surface, (0, 0))
        self.draw_view()
        self.draw_top_bar(fps)
        self.draw_bottom_bar()
        self.draw_right_panel(fps)
        self.draw_help()
        pygame.display.flip()

    # ---------- events ----------
    def commit_n_input(self):
        text = self.input_text.strip()
        if text:
            try:
                self.target_n = clamp(int(text), 1, 400)
            except ValueError:
                self.target_n = clamp(self.target_n, 1, 400)
        self.input_text = str(self.target_n)
        self.input_active = False
        self.input_replace = False

    def commit_auto_pass_input(self):
        text = self.auto_pass_text.strip()
        if not text:
            self.auto_pass_override = None
        else:
            try:
                value = max(1, int(text))
            except ValueError:
                value = None
            self.auto_pass_override = value
        self.auto_pass_text = "" if self.auto_pass_override is None else str(self.auto_pass_override)
        self.auto_pass_active = False
        self.auto_pass_replace = False

    def commit_csv_name_input(self):
        text = self.csv_name_text.strip()
        if not text:
            self.csv_name_text = Path(LOAD_MASTER_PATH).name
            self.csv_name_active = False
            self.csv_name_replace = False
            return
        self.csv_name_text = text
        self.csv_name_active = False
        self.csv_name_replace = False

    def set_master_csv(self):
        candidate = (self.repo_root / "comparison" / self.csv_name_text).resolve()
        if not candidate.exists():
            self.set_status(f"CSV not found: {self.csv_name_text}")
            self.set_workspace_lines([f"CSV not found: {self.csv_name_text}"])
            return
        global LOAD_MASTER_PATH
        LOAD_MASTER_PATH = str(candidate)
        self.set_status(f"CSV set: {candidate.name}")
        self.set_workspace_lines([f"CSV set: {candidate.name}"])
        self.auto_pass_text = ""
        self.auto_pass_active = False
        self.auto_pass_replace = False
        self.auto_pass_override = None

    def bump_n(self, delta):
        self.target_n = clamp(self.target_n + delta, 1, 400)
        self.input_text = str(self.target_n)

    def nudge_selected(self, dx, dy):
        if self.selected is None:
            return
        t = self.trees[self.selected]
        trial = TreeInstance.from_pose(t.x + dx, t.y + dy, t.deg, base_poly=self.base_poly)
        if self.is_tree_legal(trial, ignore_index=self.selected):
            t.set_pose(trial.x, trial.y, trial.deg)
            self.stats_dirty = True

    def rotate_selected(self, ddeg):
        idx = self.selected
        if idx is None and self.mouse_world is not None:
            wx, wy = self.mouse_world
            idx = self.pick_tree(wx, wy)
            if idx is not None:
                self.selected = idx
        if idx is None:
            return
        t = self.trees[idx]
        trial = TreeInstance.from_pose(t.x, t.y, t.deg + ddeg, base_poly=self.base_poly)
        if self.is_tree_legal(trial, ignore_index=idx):
            t.set_pose(trial.x, trial.y, trial.deg)
            self.stats_dirty = True

    def zoom_at(self, sx, sy, factor):
        wx, wy = self.screen_to_world(sx, sy)
        new_scale = clamp(self.scale * factor, self.zoom_limits[0], self.zoom_limits[1])
        if abs(new_scale - self.scale) < 1e-9:
            return
        self.scale = new_scale
        # Keep world point under cursor fixed.
        cx, cy = self.view_rect.centerx, self.view_rect.centery
        self.camera[0] = wx - (sx - cx) / self.scale
        self.camera[1] = wy + (sy - cy) / self.scale

    def handle_ui_click(self, mx, my):
        if self.buttons["generate"].collidepoint(mx, my):
            self.generate_optimal_layout()
            return True
        if self.buttons["auto"].collidepoint(mx, my):
            self.auto_place(self.target_n)
            return True
        if self.buttons["opt"].collidepoint(mx, my):
            if self.optimizing:
                self.stop_optimize()
            else:
                self.start_optimize()
            return True
        if self.buttons["auto_opt"].collidepoint(mx, my):
            self.start_multi_optimize()
            return True
        if self.buttons["record"].collidepoint(mx, my):
            self.record_output()
            return True
        if self.buttons["frame"].collidepoint(mx, my):
            self.frame_view()
            return True
        if self.buttons["reset_view"].collidepoint(mx, my):
            self.reset_view()
            return True

        if self.buttons["n_minus"].collidepoint(mx, my):
            self.bump_n(-1)
            return True
        if self.buttons["n_plus"].collidepoint(mx, my):
            self.bump_n(1)
            return True
        if self.buttons["load_n"].collidepoint(mx, my):
            self.load_layout_for_n(self.target_n)
            return True
        if self.auto_pass_rect.collidepoint(mx, my):
            self.auto_pass_menu_open = not self.auto_pass_menu_open
            self.input_active = False
            self.auto_pass_active = False
            return True
        if self.auto_pass_menu_open and self.auto_pass_menu_rect.collidepoint(mx, my):
            rel_y = my - self.auto_pass_menu_rect.y
            idx = int(rel_y // 20)
            if 0 <= idx < len(self.auto_pass_options):
                self.auto_pass_override = self.auto_pass_options[idx]
                label = "auto" if self.auto_pass_override is None else str(self.auto_pass_override)
                self.set_status(f"Auto passes set: {label}")
                self.set_workspace_lines([f"Auto passes set: {label}"])
            self.auto_pass_menu_open = False
            return True
        if self.csv_name_rect.collidepoint(mx, my):
            self.csv_name_active = True
            self.csv_name_replace = True
            self.input_active = False
            return True
        if self.buttons["csv_set"].collidepoint(mx, my):
            self.commit_csv_name_input()
            self.set_master_csv()
            return True
        if self.n_input_rect.collidepoint(mx, my):
            self.input_active = True
            self.input_text = str(self.target_n)
            self.input_replace = True
            return True

        return False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False

                if self.input_active:
                    if event.key == pygame.K_RETURN:
                        self.commit_n_input()
                    elif event.key == pygame.K_BACKSPACE:
                        self.input_text = self.input_text[:-1]
                        self.input_replace = False
                    elif event.key == pygame.K_ESCAPE:
                        self.input_active = False
                        self.input_text = str(self.target_n)
                        self.input_replace = False
                    elif event.unicode.isdigit() and len(self.input_text) < 4:
                        if self.input_replace:
                            self.input_text = event.unicode
                            self.input_replace = False
                        else:
                            self.input_text += event.unicode
                    continue
                if self.csv_name_active:
                    if event.key == pygame.K_RETURN:
                        self.commit_csv_name_input()
                        self.set_master_csv()
                    elif event.key == pygame.K_BACKSPACE:
                        self.csv_name_text = self.csv_name_text[:-1]
                        self.csv_name_replace = False
                    elif event.key == pygame.K_ESCAPE:
                        self.csv_name_active = False
                        self.csv_name_text = Path(LOAD_MASTER_PATH).name
                        self.csv_name_replace = False
                    elif event.unicode and len(self.csv_name_text) < 80:
                        if self.csv_name_replace:
                            self.csv_name_text = event.unicode
                            self.csv_name_replace = False
                        else:
                            self.csv_name_text += event.unicode
                    continue
                if self.auto_pass_active:
                    if event.key == pygame.K_ESCAPE:
                        self.auto_pass_active = False
                    continue

                if event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_g:
                    self.generate_optimal_layout()
                elif event.key == pygame.K_p:
                    self.auto_place(self.target_n)
                elif event.key == pygame.K_o:
                    if self.optimizing:
                        self.stop_optimize()
                    else:
                        self.start_optimize()
                elif event.key == pygame.K_t:
                    self.start_multi_optimize()
                elif event.key == pygame.K_l:
                    self.record_output()
                elif event.key == pygame.K_k:
                    self.load_layout_for_n(self.target_n)
                elif event.key == pygame.K_m:
                    self.pan_mode = not self.pan_mode
                    self.set_status(f"Pan mode: {'on' if self.pan_mode else 'off'}")
                elif event.key == pygame.K_F9:
                    if self.auto_active:
                        self.stop_automation("Automation stopped.")
                    else:
                        self.start_automation()
                elif event.key == pygame.K_f:
                    self.frame_view()
                elif event.key == pygame.K_0:
                    self.reset_view()
                elif event.key == pygame.K_LEFTBRACKET:
                    mods = pygame.key.get_mods()
                    step = self.rotate_step
                    if mods & pygame.KMOD_SHIFT:
                        step = self.rotate_step_fine
                    if mods & pygame.KMOD_CTRL:
                        step = self.rotate_step_ultra
                    self.rotate_selected(-step)
                elif event.key == pygame.K_RIGHTBRACKET:
                    mods = pygame.key.get_mods()
                    step = self.rotate_step
                    if mods & pygame.KMOD_SHIFT:
                        step = self.rotate_step_fine
                    if mods & pygame.KMOD_CTRL:
                        step = self.rotate_step_ultra
                    self.rotate_selected(step)
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_ALT:
                        step = self.nudge_step
                        if mods & pygame.KMOD_SHIFT:
                            step = self.nudge_step_fine
                        if mods & pygame.KMOD_CTRL:
                            step = self.nudge_step_ultra
                        dx = dy = 0.0
                        if event.key == pygame.K_LEFT:
                            dx = -step
                        elif event.key == pygame.K_RIGHT:
                            dx = step
                        elif event.key == pygame.K_UP:
                            dy = step
                        elif event.key == pygame.K_DOWN:
                            dy = -step
                        self.nudge_selected(dx, dy)
                    else:
                        if event.key == pygame.K_LEFT:
                            self.camera[0] -= 0.1
                        elif event.key == pygame.K_RIGHT:
                            self.camera[0] += 0.1
                        elif event.key == pygame.K_UP:
                            self.camera[1] += 0.1
                        elif event.key == pygame.K_DOWN:
                            self.camera[1] -= 0.1
                elif event.key == pygame.K_a:
                    self.add_tree(self.mouse_world[0], self.mouse_world[1])
                    self.set_status("Added tree at cursor.")
                elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                    self.remove_selected()
                elif event.key == pygame.K_TAB:
                    if self.trees:
                        self.selected = 0 if self.selected is None else (self.selected + 1) % len(self.trees)
                elif event.key == pygame.K_c:
                    self.snap_selected_to_contact()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    cx, cy = self.view_rect.centerx, self.view_rect.centery
                    self.zoom_at(cx, cy, self.zoom_factor)
                elif event.key == pygame.K_MINUS:
                    cx, cy = self.view_rect.centerx, self.view_rect.centery
                    self.zoom_at(cx, cy, 1.0 / self.zoom_factor)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    if self.pan_mode and self.view_rect.collidepoint(mx, my):
                        self.panning = True
                        self.pan_last_screen = (mx, my)
                        continue
                    if self.input_active and not self.n_input_rect.collidepoint(mx, my):
                        self.input_active = False
                        self.input_text = str(self.target_n)
                        self.input_replace = False
                    if self.csv_name_active and not self.csv_name_rect.collidepoint(mx, my):
                        self.commit_csv_name_input()
                    if self.auto_pass_menu_open and not (self.auto_pass_rect.collidepoint(mx, my) or self.auto_pass_menu_rect.collidepoint(mx, my)):
                        self.auto_pass_menu_open = False

                    if self.handle_ui_click(mx, my):
                        continue

                    if self.view_rect.collidepoint(mx, my):
                        wx, wy = self.screen_to_world(mx, my)
                        self.mouse_world = (wx, wy)
                        mods = pygame.key.get_mods()
                        self.selected = self.pick_tree(wx, wy)
                        if self.selected is not None:
                            self.dragging = True
                            self.last_mouse_world = (wx, wy)
                        elif mods & pygame.KMOD_SHIFT:
                            self.selecting = True
                            self.sel_start = (wx, wy)
                            self.sel_end = (wx, wy)
                        else:
                            self.selected = None
                elif event.button == 3:
                    mx, my = event.pos
                    if self.view_rect.collidepoint(mx, my):
                        self.selecting = True
                        self.sel_start = self.screen_to_world(mx, my)
                        self.sel_end = self.sel_start
                elif event.button in (4, 5):
                    mods = pygame.key.get_mods()
                    direction = 1 if event.button == 4 else -1
                    if mods & pygame.KMOD_CTRL:
                        factor = self.zoom_factor if direction > 0 else 1.0 / self.zoom_factor
                        self.zoom_at(*event.pos, factor)
                    else:
                        step = 3.0 * direction
                        if mods & pygame.KMOD_SHIFT:
                            step = 1.0 * direction
                        self.rotate_selected(step)

            if event.type == pygame.MOUSEWHEEL:
                mods = pygame.key.get_mods()
                if mods & pygame.KMOD_CTRL:
                    factor = self.zoom_factor if event.y > 0 else 1.0 / self.zoom_factor
                    mx, my = pygame.mouse.get_pos()
                    self.zoom_at(mx, my, factor)
                else:
                    step = 3.0 if event.y > 0 else -3.0
                    if mods & pygame.KMOD_SHIFT:
                        step = 1.0 if event.y > 0 else -1.0
                    self.rotate_selected(step)

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
                    self.last_mouse_world = None
                    self.panning = False
                    self.pan_last_screen = None
                    if self.selecting:
                        self.selecting = False
                        self.update_selection_stats()
                elif event.button == 3:
                    self.selecting = False
                    self.update_selection_stats()

            if event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if self.view_rect.collidepoint(mx, my):
                    wx, wy = self.screen_to_world(mx, my)
                    self.mouse_world = (wx, wy)
                    if self.panning and self.pan_last_screen is not None:
                        lx, ly = self.pan_last_screen
                        dx = mx - lx
                        dy = my - ly
                        self.camera[0] -= dx / self.scale
                        self.camera[1] += dy / self.scale
                        self.pan_last_screen = (mx, my)
                        continue
                    if self.dragging and self.selected is not None and self.last_mouse_world is not None:
                        lx, ly = self.last_mouse_world
                        dx, dy = wx - lx, wy - ly
                        self.move_selected(dx, dy)
                        self.last_drag_delta = (dx, dy)
                        self.last_mouse_world = (wx, wy)
                    if self.selecting:
                        self.sel_end = (wx, wy)
                        self.update_selection_stats()

        return True

    # ---------- loop ----------
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            now = time.time()
            if self.stats_dirty and (now - self.stats_last_time) >= self.stats_update_interval:
                self.update_stats()
                self.stats_last_time = now
                self.stats_dirty = False
                if self.selecting or self.sel_stats:
                    self.update_selection_stats()

            if self.optimizing:
                self.optimize_step()
            if self.multi_opt_active and not self.optimizing:
                if self.multi_opt_left <= 0:
                    self.multi_opt_active = False
                    self.set_status("Auto O complete.")
                else:
                    self.start_optimize(preserve_best=True)
                    if self.optimizing:
                        self.multi_opt_left -= 1

            if self.auto_active:
                self.automation_step()

            fps = self.clock.get_fps()
            self.draw(fps)
            self.clock.tick(60)


def main():
    app = TreeSandboxApp()
    app.run()


if __name__ == "__main__":
    main()
