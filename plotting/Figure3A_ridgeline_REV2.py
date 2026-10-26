# -*- coding: utf-8 -*-
"""
Figure 3A (REV1+) — Ridgeline probability plot (legacy style) using unified table only.

Per threshold (one figure):
  - X axis: probability in [0,1]
      * apo probability  (pocket_p_apo)  in blue
      * holo probability (pocket_p_holo) in orange
  - Y axis: evidence ratio axis binned and ALWAYS centered at 0.0 (balance point)
      * default: log2((evidence_n_holo + HOLO_PLUS_ONE) / evidence_n_apo)
      * option: log10

Improvements vs prior REV1:
  1) Ratio bins anchored so 0.0 is exactly the CENTER of a bin (balance persists across thresholds).
  2) Optional auto-range trimming by minimum bin counts (off by default).
  3) Shorter & wider aspect ratio; vertical compactness now also changes figure height.
  4) Right-side winshare hairline + counts have independent spacing (no collision).

Inputs:
  analysis_unified_levels.csv (or a .zip containing it)

Required columns:
  threshold, level, pocket_class,
  evidence_n_apo, evidence_n_holo,
  pocket_p_apo, pocket_p_holo

Outputs:
  PDF + PNG per threshold (SVG optional)
"""

import argparse
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.colors as mcolors


# ----------------------------
# USER SETTINGS (Spyder)
# ----------------------------
# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg
from common import figspec


INPUT_TABLE  = str(cfg.ANALYSIS_UNIFIED_LEVELS_CSV)
OUTPUT_DIR   = str(cfg.PLOTS_DIR / "3A_ridgeline")
POSTER_OUTPUT_DIR = str(cfg.PRESENTATION_DIR)
OUT_BASENAME = "Figure3A_ridgeline_REV2"

THRESHOLDS_TO_PLOT = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050", "ST_030"]
#THRESHOLDS_TO_PLOT = ["UNCLUSTERED", "ST_100", "ST_050"]

# Filtering / ratio construction
HOLO_PLUS_ONE = 1.0
REQUIRE_EVIDENCE_APO_MIN = 1.0

# Ratio axis for binning
RATIO_AXIS = "log2"          # "log2" or "log10"
RIDGE_BINS = 22              # number of ratio bins (ridges)

# Ensure there is a BIN centered exactly at ANCHOR_CENTER (requires odd bin count).
FORCE_CENTER_BIN = True

# (1) Anchor so that 0.0 is a BIN CENTER (balance point)
ANCHOR_CENTER = 0.0          # keep at 0.0 unless you truly want a different equilibrium

# (2) Optional auto-range trimming by minimum bin counts
AUTO_TRIM_RANGE_BY_MINCOUNT = False
TRIM_MIN_COUNT = 60          # used only if AUTO_TRIM_RANGE_BY_MINCOUNT=True

SHOW_PANEL_LETTER = False
PANEL_LETTER = "a"      # lowercase per the Nature figure guide; --panel_letter overrides
PANEL_LETTER_FS = 12
TRANSPARENT_BACKGROUND = False

# On-panel annotation of the clustering threshold represented by the ridgeline
# (the panel otherwise carries no indication of which UBS threshold is shown).
SHOW_THRESHOLD_ANNOTATION = True
THRESHOLD_ANNOTATION_FS = 8.5

# Ridgeline density / smoothing toggle
RIDGE_DENSITY_MODE = "hist"  # "hist" or "kde"
RIDGE_SMOOTH_SIGMA = 1.2     # "hist" only; set 0 for no smoothing
RIDGE_KDE_BW_SCALE = 1.0     # "kde" only

# Probability gradient
USE_GRADIENT_FILL = False      # set False for flat fills
GRADIENT_LIGHTEN  = 0.60      # higher = subtler gradient (e.g., 0.75–0.90)

# Probability axis
PROB_X_MIN, PROB_X_MAX = 0.0, 1.0
N_POINTS = 201

# (3) Vertical compactness controls (all matter)
VERTICAL_SPACING = 0.22      # smaller => more compact stack (default 0.22)
MAX_HALF_HEIGHT = 0.18       # ridge amplitude (smaller => more compact)
# Right-hand strip carrying the winshare hairline and the per-bin count, in millimetres. The
# widest count is five digits at 6 pt (~9 mm) and the hairline takes a quarter of the strip.
STRIP_MM = 15.0
STRIP_GAP_MM = 1.5
Y_LABEL_EVERY = 1            # show every N-th y tick label (e.g., 2 reduces clutter)
TOP_CLEARANCE = 0.18   # extra space above the top ridge (in y-data units)
BOTTOM_CLEARANCE = 0.08


# Figure sizing (shorter + wider like legacy)
FIG_WIDTH = 3.46             # Single column width for NAR
FIG_SIZE_OVERRIDE = None     # (width, height) in inches; set by --fig_size, wins over the above
EXACT_CANVAS = False         # save the full canvas (no tight crop) so --fig_size is the size on
                             # disk and composed parts tile without rescaling
FIG_HEIGHT_MODE = "auto"     
FIG_HEIGHT_FIXED = 3.4       
FIG_HEIGHT_PER_RIDGE = 0.08  # Scaled down to restore aspect ratio
FIG_HEIGHT_BASE = 1.20       # Scaled down from 2.20

ALPHA = 0.55

# (4) Right-side decorations: winshare + bin counts (non-colliding)
SHOW_WINSHARE = True
SHOW_BIN_COUNTS = True
COUNT_MONO_FONT = True
WINSHARE_HEIGHT_SCALE = 0.92


# Export
SAVE_FORMATS = ["pdf", "png", "svg"]
DPI = 600


# ----------------------------
# Shared style block (Figure 3 house style)
# ----------------------------
TITLE_FS = figspec.FS_TITLE     # 7.0 -- see common/figspec.py
LAB_FS   = figspec.FS_TITLE     # 7.0
TICK_FS  = figspec.FS_TICK      # 6.0
SMALL_FS = figspec.FS_NOTE      # 6.0

AX_LW    = 0.8
TICK_W   = 0.8
TICK_LEN = 3.5

#TITLE_FS = 9.2
#TICK_FS  = 7.0
#LAB_FS   = 8.2
#SMALL_FS = 6.2

APO_COLOR  = figspec.APO   # single source: common/figspec.py
HOLO_COLOR = figspec.HOLO

FONT_PREFERENCE = ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"]

import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_PREFERENCE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "axes.linewidth": AX_LW,
    "xtick.major.width": TICK_W,
    "ytick.major.width": TICK_W,
    "xtick.major.size": TICK_LEN,
    "ytick.major.size": TICK_LEN,

    "axes.titlepad": 4,
})


# ----------------------------
# Helpers (I/O, validation)
# ----------------------------
def ensure_dir(p: str | None):
    if p:
        os.makedirs(p, exist_ok=True)
    return p

def parse_args():
    p = argparse.ArgumentParser(description="Plot Figure 3A ridgeline panel.")
    p.add_argument("--input_table", default=INPUT_TABLE)
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--out_basename", default=OUT_BASENAME)
    p.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated thresholds to plot, e.g. ST_100. Defaults to THRESHOLDS_TO_PLOT.",
    )
    p.add_argument("--exact_canvas", action="store_true",
                   help="save the FULL canvas rather than a tight crop, so the file is exactly "
                        "--fig_size and tiles into a composed figure without rescaling.")
    p.add_argument("--fig_size", default=None, metavar="W,H",
                   help="canvas in inches, overriding FIG_WIDTH and the auto height. Use when "
                        "this panel is being composed into a multi-panel figure and has to "
                        "arrive at its final size rather than be rescaled afterwards.")
    p.add_argument("--poster_output_dir", action="store_true", help="Write to the configured poster plots directory.")
    p.add_argument("--show_panel_letter", action="store_true")
    p.add_argument("--panel_letter", default=None, metavar="L",
                   help="the panel letter to draw (implies --show_panel_letter). Set it when "
                        "this panel is one of several being composed and is not the first.")
    p.add_argument("--hide_panel_letter", "--hide_panel_letters", dest="hide_panel_letter", action="store_true")
    p.add_argument("--transparent_background", action="store_true")
    p.add_argument("--hide_threshold_annotation", action="store_true",
                   help="drop the 'UBS - ST 100%' tag inside the plot area. Redundant once the "
                        "manuscript states the threshold before the figure, and it is the only "
                        "thing standing between the top ridge and the plot border.")
    return p.parse_args()

def configure_from_args(args):
    global FIG_SIZE_OVERRIDE, INPUT_TABLE, OUTPUT_DIR, OUT_BASENAME, THRESHOLDS_TO_PLOT, PANEL_LETTER
    global EXACT_CANVAS
    global SHOW_PANEL_LETTER, TRANSPARENT_BACKGROUND, SHOW_THRESHOLD_ANNOTATION
    INPUT_TABLE = args.input_table
    OUTPUT_DIR = POSTER_OUTPUT_DIR if args.poster_output_dir else args.output_dir
    OUT_BASENAME = args.out_basename
    if args.thresholds:
        THRESHOLDS_TO_PLOT = [t.strip() for t in args.thresholds.split(",") if t.strip()]
    EXACT_CANVAS = args.exact_canvas
    if args.hide_threshold_annotation:
        SHOW_THRESHOLD_ANNOTATION = False
    if args.fig_size:
        try:
            w, h = (float(v) for v in args.fig_size.split(","))
        except ValueError:
            raise SystemExit("--fig_size takes W,H in inches")
        FIG_SIZE_OVERRIDE = (w, h)
    if args.panel_letter:
        PANEL_LETTER = args.panel_letter
        SHOW_PANEL_LETTER = True
    if args.show_panel_letter:
        SHOW_PANEL_LETTER = True
    if args.hide_panel_letter:
        SHOW_PANEL_LETTER = False
    TRANSPARENT_BACKGROUND = args.transparent_background
    # Poster SVGs outline their glyphs so they render identically anywhere; the manuscript build
    # keeps live text ("none"), which needs the viewer to actually have the font stack installed.
    plt.rcParams["svg.fonttype"] = "path" if TRANSPARENT_BACKGROUND else "none"

def apply_background(fig):
    if TRANSPARENT_BACKGROUND:
        fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        for ax in fig.axes:
            if getattr(ax, "_transparent_background_axis", False):
                ax.set_facecolor("none")
            else:
                ax.set_facecolor("white")
    else:
        fig.patch.set_facecolor("white")

def save_figure(fig, basepath: str | None, formats: list[str], *, dpi: int = 150, close: bool = True):
    if not basepath:
        plt.show()
        return
    ensure_dir(os.path.dirname(basepath))
    stem = os.path.splitext(basepath)[0]
    apply_background(fig)
    save_kwargs = {
        "bbox_inches": (None if EXACT_CANVAS else "tight"),
        "pad_inches": 0.02,
        "facecolor": fig.get_facecolor(),
    }
    # poster output is png + svg only; a poster build never consumes the pdf
    if TRANSPARENT_BACKGROUND:
        formats = [e for e in formats if e.lower() != "pdf"]
    for ext in formats:
        path = f"{stem}.{ext}"
        try:
            fig.savefig(path, dpi=dpi, **save_kwargs)
        except PermissionError as exc:
            print(f"[warn] could not save {path}: {exc}")
    if close:
        plt.close(fig)

def load_unified(path: str) -> pd.DataFrame:
    p = str(path)
    if p.lower().endswith(".zip"):
        with zipfile.ZipFile(p) as z:
            name = "analysis_unified_levels.csv"
            if name not in z.namelist():
                csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    raise SystemExit("Zip contains no CSV.")
                name = csvs[0]
            with z.open(name) as f:
                return pd.read_csv(f, low_memory=False)
    return pd.read_csv(p, low_memory=False)

def require_cols(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        print("Available columns:", list(df.columns))
        raise SystemExit("Stopping due to missing columns.")

def format_row_label(thr: str) -> str:
    if thr == "UNCLUSTERED":
        return "PDB pockets"
    if thr.startswith("ST_") and thr[3:].isdigit():
        return f"UBS - ST {thr[3:]}%"
    return f"UBS - {thr}"


# ----------------------------
# Unified-table filtering per threshold
# ----------------------------
def subset_threshold_unified(df: pd.DataFrame, thr: str) -> pd.DataFrame:
    lvl = "entry" if thr == "UNCLUSTERED" else "ubs"
    sub = df[(df["threshold"].astype(str) == str(thr)) &
             (df["level"].astype(str).str.strip().str.lower() == lvl)].copy()

    sub["pocket_class"] = sub["pocket_class"].astype(str).str.strip().str.lower()
    sub = sub[sub["pocket_class"].isin(["apo", "holo"])].copy()

    for c in ["evidence_n_apo", "evidence_n_holo", "pocket_p_apo", "pocket_p_holo"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    sub = sub[np.isfinite(sub["pocket_p_apo"]) & np.isfinite(sub["pocket_p_holo"])].copy()
    sub = sub[np.isfinite(sub["evidence_n_apo"]) & np.isfinite(sub["evidence_n_holo"])].copy()
    sub = sub[sub["evidence_n_apo"] >= float(REQUIRE_EVIDENCE_APO_MIN)].copy()

    sub["evidence_n_holo_adj"] = sub["evidence_n_holo"] + float(HOLO_PLUS_ONE)
    sub = sub[sub["evidence_n_holo_adj"] > 0].copy()

    return sub


# ----------------------------
# Density & smoothing
# ----------------------------
def _gauss_smooth_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    rad = max(1, int(round(3 * sigma)))
    xk = np.arange(-rad, rad + 1, dtype=float)
    k = np.exp(-(xk**2) / (2.0 * sigma * sigma))
    k /= (k.sum() if k.sum() > 0 else 1.0)
    return np.convolve(arr, k, mode="same")

def _density_curve(vals: np.ndarray, x_min: float, x_max: float, n_points: int,
                   mode: str = "hist", smooth_sigma: float = 1.2, bw_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    xg = np.linspace(float(x_min), float(x_max), int(n_points))
    if v.size == 0:
        return xg, np.zeros_like(xg)

    mode = (mode or "hist").lower()
    if mode == "kde":
        n = v.size
        if n <= 1:
            bw = 0.05
        else:
            s = float(np.nanstd(v, ddof=1))
            iqr = float(np.nanpercentile(v, 75) - np.nanpercentile(v, 25))
            sigma = min(s, iqr/1.349) if np.isfinite(iqr) and iqr > 0 else s
            if (not np.isfinite(sigma)) or sigma <= 0:
                sigma = max(1e-3, float(np.nanstd(v)))
            bw = 0.9 * sigma * (n ** (-1/5))
        bw = max(float(bw) * float(bw_scale), np.finfo(float).eps)

        u = (xg[:, None] - v[None, :]) / bw
        dens = np.exp(-0.5 * u * u).sum(axis=1) / (n * bw * np.sqrt(2.0*np.pi))
        m = float(np.nanmax(dens)) if dens.size else 0.0
        dens = dens / (m if m > 0 else 1.0)
        return xg, dens

    hist, edges = np.histogram(v, bins=int(n_points) - 1, range=(float(x_min), float(x_max)), density=True)
    if float(smooth_sigma) > 0:
        hist = _gauss_smooth_1d(hist, float(smooth_sigma))
    m = float(np.nanmax(hist)) if hist.size else 0.0
    dens = hist / (m if m > 0 else 1.0)
    xc = 0.5 * (edges[:-1] + edges[1:])
    return xc, dens


# ----------------------------
# Ratio axis + binning (anchored at 0 center)
# ----------------------------
def _compute_ratio_axis(a_apo: np.ndarray, a_holo_adj: np.ndarray, *, mode: str = "log2") -> tuple[np.ndarray, str]:
    mode = (mode or "log2").lower()
    if mode == "log10":
        r = np.log10(a_holo_adj / a_apo)
        return r.astype(float), "log₁₀(evidence_n_holo_adj / evidence_n_apo)"
    r = np.log2(a_holo_adj / a_apo)
    return r.astype(float), "log₂(evidence_n_holo_adj / evidence_n_apo)"

def _anchored_edges_center0(rvals: np.ndarray, bins: int, center: float = 0.0,
                            trim_by_count: bool = False, trim_min_count: int = 50) -> np.ndarray:
    """
    Build equal-width bin edges such that `center` is exactly the CENTER of one bin.

    Practical rule:
      - If FORCE_CENTER_BIN is True, we ensure an ODD number of bins so one bin can be centered at `center`.
      - If bins provided is even, we silently use (bins + 1). You can later trim/skip a top/bottom bin if desired.
    """
    r = np.asarray(rvals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        b = max(int(bins), 3)
        if b % 2 == 0:
            b += 1
        return np.linspace(center - 1, center + 1, b + 1)

    b = max(int(bins), 3)
    if bool(FORCE_CENTER_BIN) and (b % 2 == 0):
        b += 1  # make odd so we can have a bin centered at `center`

    rmin, rmax = float(np.min(r)), float(np.max(r))
    # use the max absolute deviation from center to cover both tails
    span = max(abs(rmin - center), abs(rmax - center))
    span = max(span, 1e-6)

    bw = (2.0 * span) / b

    # Construct edges so that the MIDDLE bin is centered at `center`
    # For odd b, middle index is mid = b//2, and its center is:
    #   edges[mid] + bw/2 == center  => edges[mid] = center - bw/2
    mid = b // 2
    left_edge_at_mid = center - 0.5 * bw
    edges0 = left_edge_at_mid - mid * bw
    edges = edges0 + np.arange(b + 1, dtype=float) * bw

    if not trim_by_count:
        return edges

    # Trim sparse tails (still keeps alignment, because we slice whole bins)
    didx = np.digitize(r, edges, right=True) - 1
    didx = np.clip(didx, 0, b - 1)
    counts = np.bincount(didx, minlength=b)
    good = counts >= int(trim_min_count)
    if not np.any(good):
        return edges

    # expand from center bin to include contiguous good region
    lo = mid
    hi = mid
    while lo - 1 >= 0 and good[lo - 1]:
        lo -= 1
    while hi + 1 < b and good[hi + 1]:
        hi += 1

    # if center bin not good, keep at least 3 bins around it
    if not good[mid]:
        lo = max(0, mid - 1)
        hi = min(b - 1, mid + 1)

    return edges[lo:(hi + 2)]


# ----------------------------
# Ridgeline drawing (legacy-style "same top") (legacy-style "same top")
# ----------------------------
def _make_linear_cmap(base_hex: str, lighten: float = 0.75):
    base = np.array(mcolors.to_rgb(base_hex), dtype=float)
    light = 1.0 - (1.0 - base) * (1.0 - float(lighten))
    return mcolors.LinearSegmentedColormap.from_list("grad_"+base_hex, [light, base], N=256)

def _draw_half_ridge(ax, xv, y0, ytop, base_color, *, alpha, edgecolor, edgewidth,
                     x_min, x_max, max_half_height, z):
    x_poly = np.concatenate([xv, xv[::-1]])
    y_poly = np.concatenate([np.full_like(xv, y0), ytop[::-1]])
    path = Path(np.c_[x_poly, y_poly])

    if USE_GRADIENT_FILL:
        cmap = _make_linear_cmap(base_color, lighten=GRADIENT_LIGHTEN)
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        im = ax.imshow(
            grad,
            extent=(x_min, x_max, y0, y0 + float(max_half_height)),
            origin="lower",
            aspect="auto",
            cmap=cmap,
            alpha=alpha,
            zorder=z + 0.001,
        )
        im.set_clip_path(PathPatch(path, transform=ax.transData))
    else:
        # flat fill (no gradient)
        ax.add_patch(Polygon(
            np.c_[x_poly, y_poly], closed=True,
            facecolor=base_color, edgecolor="none",
            alpha=alpha, zorder=z + 0.001
        ))

    ax.add_patch(Polygon(
        np.c_[x_poly, y_poly], closed=True,
        facecolor="none",
        edgecolor=edgecolor, linewidth=edgewidth,
        alpha=1.0, zorder=z + 0.002
    ))

def _draw_split_violin_ridge_same_top(
    ax, *,
    y0: float,
    vals_left: np.ndarray,
    vals_right: np.ndarray,
    x_min: float,
    x_max: float,
    n_points: int,
    max_half_height: float,
    color_left: str,
    color_right: str,
    alpha: float,
    dens_mode: str,
    smooth_sigma: float,
    bw_scale: float,
    z: float,
):
    xl, dl = _density_curve(vals_left,  x_min, x_max, n_points, mode=dens_mode, smooth_sigma=smooth_sigma, bw_scale=bw_scale)
    xr, dr = _density_curve(vals_right, x_min, x_max, n_points, mode=dens_mode, smooth_sigma=smooth_sigma, bw_scale=bw_scale)

    yl = y0 + (dl * float(max_half_height))
    yr = y0 + (dr * float(max_half_height))

    _draw_half_ridge(ax, xl, y0, yl, color_left,  alpha=alpha, edgecolor="#4A4A4A", edgewidth=0.6,
                     x_min=x_min, x_max=x_max, max_half_height=max_half_height, z=z+0.10)
    _draw_half_ridge(ax, xr, y0, yr, color_right, alpha=alpha, edgecolor="#4A4A4A", edgewidth=0.6,
                     x_min=x_min, x_max=x_max, max_half_height=max_half_height, z=z+0.20)

    if np.isfinite(vals_left).any():
        med = float(np.nanmedian(vals_left[np.isfinite(vals_left)]))
        ax.plot([med, med], [y0, y0 + 0.055], lw=1.05, color=color_left, alpha=0.95, zorder=z + 1.0)
    if np.isfinite(vals_right).any():
        med = float(np.nanmedian(vals_right[np.isfinite(vals_right)]))
        ax.plot([med, med], [y0, y0 + 0.055], lw=1.05, color=color_right, alpha=0.95, zorder=z + 1.1)


def plot_ridgeline_fullrange_same_top(df: pd.DataFrame, *, title: str, save_base: str | None,
                                      thr_label: str | None = None):
    a_apo = pd.to_numeric(df["evidence_n_apo"], errors="coerce").to_numpy(float)
    a_holo = pd.to_numeric(df["evidence_n_holo_adj"], errors="coerce").to_numpy(float)
    p_apo = np.clip(pd.to_numeric(df["pocket_p_apo"], errors="coerce").to_numpy(float), 0.0, 1.0)
    p_holo = np.clip(pd.to_numeric(df["pocket_p_holo"], errors="coerce").to_numpy(float), 0.0, 1.0)

    ratio, ratio_label = _compute_ratio_axis(a_apo, a_holo, mode=RATIO_AXIS)
    m = np.isfinite(ratio) & np.isfinite(p_apo) & np.isfinite(p_holo)
    if not np.any(m):
        print("Ridgeline: nothing to plot after finite filtering.")
        return

    ratio = ratio[m]
    p_apo = p_apo[m]
    p_holo = p_holo[m]

    edges = _anchored_edges_center0(
        ratio, RIDGE_BINS, center=float(ANCHOR_CENTER),
        trim_by_count=bool(AUTO_TRIM_RANGE_BY_MINCOUNT),
        trim_min_count=int(TRIM_MIN_COUNT),
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    nb = len(centers)

    didx = np.digitize(ratio, edges, right=True) - 1
    didx = np.clip(didx, 0, nb - 1)

    per_bin = []
    for i, c in enumerate(centers):
        inb = (didx == i)
        n_total = int(np.sum(inb))
        if n_total == 0:
            continue
        per_bin.append((float(c), p_apo[inb], p_holo[inb], n_total, i))

    if not per_bin:
        print("Ridgeline: all bins empty.")
        return

    per_bin.sort(key=lambda t: t[0])
    n = len(per_bin)

    step = max(0.04, float(VERTICAL_SPACING))
    y_positions = np.arange(n, dtype=float) * step

    y_labels = []
    for j, (c, *_rest) in enumerate(per_bin):
        if (int(Y_LABEL_EVERY) > 1) and (j % int(Y_LABEL_EVERY) != 0):
            y_labels.append("")
        else:
            y_labels.append(f"{c:.1f}")

    if str(FIG_HEIGHT_MODE).lower() == "fixed":
        fig_h = float(FIG_HEIGHT_FIXED)
    else:
        fig_h = float(FIG_HEIGHT_BASE) + float(FIG_HEIGHT_PER_RIDGE) * n
        fig_h = max(3.0, min(fig_h, 7.2))

    fig_w = float(FIG_WIDTH)
    if FIG_SIZE_OVERRIDE:
        fig_w, fig_h = FIG_SIZE_OVERRIDE
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.set_axisbelow(True)
    ax.grid(True, axis="x", linestyle="--", alpha=0.28)
    ax.grid(False, axis="y")

    z0 = 10.0
    for j, (c, vA, vH, n_total, bin_i) in enumerate(per_bin):
        y0 = y_positions[j]
        z = z0 + j * 0.35
        _draw_split_violin_ridge_same_top(
            ax,
            y0=y0,
            vals_left=vA,
            vals_right=vH,
            x_min=float(PROB_X_MIN),
            x_max=float(PROB_X_MAX),
            n_points=int(N_POINTS),
            max_half_height=float(MAX_HALF_HEIGHT),
            color_left=APO_COLOR,
            color_right=HOLO_COLOR,
            alpha=float(ALPHA),
            dens_mode=RIDGE_DENSITY_MODE,
            smooth_sigma=float(RIDGE_SMOOTH_SIGMA),
            bw_scale=float(RIDGE_KDE_BW_SCALE),
            z=z,
        )

    winners_apo = (p_apo >= p_holo)
    winners_holo = ~winners_apo

    per_bin_counts = []
    for (c, vA, vH, n_total, bin_i) in per_bin:
        inb = (didx == int(bin_i))
        na = int(np.sum(inb & winners_apo))
        nh = int(np.sum(inb & winners_holo))
        per_bin_counts.append((n_total, na, nh))

    # ----------------------------
    # Axis limits / ticks / labels (set BEFORE right-strip axis so sharey is stable)
    # ----------------------------
    ax.set_xlim(float(PROB_X_MIN) - 0.02, float(PROB_X_MAX) + 0.02)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=TICK_FS)

    ax.set_xlabel("Pocket probability", fontsize=LAB_FS)
    ax.set_ylabel("Evidence ratio (log₂ holo/apo)", fontsize=LAB_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)

    # Final Y limits (must be set BEFORE creating the right-strip axis, otherwise sharey can misbehave)
    y_min = float(np.min(y_positions)) if y_positions.size else 0.0
    y_max = float(np.max(y_positions)) + float(MAX_HALF_HEIGHT) if y_positions.size else float(MAX_HALF_HEIGHT)
    ax.set_ylim(y_min - float(BOTTOM_CLEARANCE), y_max + float(TOP_CLEARANCE))

    # ----------------------------
    # Layout
    # ----------------------------
    # Millimetres once the canvas is the final canvas. As fractions (left/right 0.15, top/bottom
    # 0.15) the ridge block's WIDTH was 70% of whatever canvas it was given, so its proportions
    # were a consequence of the canvas rather than a choice -- at 180 x 76 mm that is a 126 x 53
    # mm block, 2.4:1, the widest panel in the manuscript. Fixed margins let the composer set the
    # block's aspect by choosing the canvas width.
    #   left   y tick numbers (-8.9) + the rotated axis name
    #   right  the winshare/count strip plus its gap (STRIP_MM + STRIP_GAP_MM) and a hair over
    #   top    the two-entry key, which sits above the axes
    if EXACT_CANVAS:
        fig.subplots_adjust(**figspec.margins(fig_w, fig_h, left=11.0,
                                              right=STRIP_GAP_MM + STRIP_MM + 1.0,
                                              top=7.0, bottom=9.0))
    else:
        fig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

    # ----------------------------
    # Right-side decorations: use a dedicated "strip" axis
    # ----------------------------
    ax_pos = ax.get_position()
    if EXACT_CANVAS:
        gap = STRIP_GAP_MM / (fig_w / figspec.MM)
        strip_w = STRIP_MM / (fig_w / figspec.MM)
    else:
        gap = 0.01       # 1 or 2% gap
        strip_w = 0.12   # 14% width. (0.82 + 0.02 + 0.14 = 0.98 total figure width)
    
    # Place axR strictly to the right of the main axes
    axR = fig.add_axes([ax_pos.x1 + gap, ax_pos.y0, strip_w, ax_pos.height], sharey=ax)
    axR._transparent_background_axis = True

    # Main axis: Bulletproof the left side so ticks/labels stay ON, but right side is OFF
    ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)

    # Strip axis cosmetics
    axR.set_xlim(0, 1) # Internal coordinates for axR (0 is left edge, 1 is right edge)
    axR.set_xticks([])
    
    # Safely hide ticks on the strip WITHOUT deleting them from the shared main axis
    axR.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    
    for spine in axR.spines.values():
        spine.set_visible(False)
    axR.set_facecolor("none")

    # Draw winshare bars + counts in the strip axis coordinates
    if SHOW_WINSHARE:
        for y0, (_n_total, na, nh) in zip(y_positions, per_bin_counts):
            tot = na + nh
            if tot <= 0:
                continue

            apo_frac = na / tot
            holo_frac = nh / tot

            hair_h = float(MAX_HALF_HEIGHT) * float(WINSHARE_HEIGHT_SCALE)

            axR.add_patch(Rectangle(
                (0.0, y0), 0.25, hair_h * apo_frac,  # Increased from 0.15 to 0.25
                facecolor=APO_COLOR, edgecolor="none",
                alpha=float(ALPHA), zorder=2.2
            ))
            axR.add_patch(Rectangle(
                (0.0, y0 + hair_h * apo_frac), 0.25, hair_h * holo_frac,
                facecolor=HOLO_COLOR, edgecolor="none",
                alpha=float(ALPHA), zorder=2.21
            ))

    if SHOW_BIN_COUNTS:
        for y0, (n_total, _na, _nh) in zip(y_positions, per_bin_counts):
            axR.text(
                0.95, # Anchored to the far right of the strip (95% mark)
                y0 + float(MAX_HALF_HEIGHT) * 0.02,
                f"{n_total:d}",
                ha="right", va="center",
                fontsize=SMALL_FS,
                color="#444444",
                alpha=0.92,
                family=("DejaVu Sans Mono" if COUNT_MONO_FONT else None),
                zorder=90,
            )


    handles = [
        # P(apo) / P(holo), as the text writes them -- from figspec so the key cannot drift again
        Line2D([], [], lw=6, color=APO_COLOR,  alpha=float(ALPHA), label=figspec.P_APO),
        Line2D([], [], lw=6, color=HOLO_COLOR, alpha=float(ALPHA), label=figspec.P_HOLO),
    ]

    # Legend: cleanly anchored ABOVE the plot border
    leg = ax.legend(
        handles=handles,
        loc="lower center",         # Anchor the BOTTOM of the legend...
        bbox_to_anchor=(0.5, 1.04), # ...just 4% above the top border of the plot
        ncol=2,
        frameon=True,
        framealpha=0.65,
        fontsize=TICK_FS,
        borderaxespad=0.2,
        handlelength=2.0,
        columnspacing=1.2,
    )

    # Style legend frame to match 3B
    frame = leg.get_frame()
    if TRANSPARENT_BACKGROUND:
        frame.set_alpha(None)
        frame.set_facecolor((1.0, 1.0, 1.0, 0.0))
    else:
        frame.set_alpha(0.65)
        frame.set_facecolor("white")
    frame.set_edgecolor("0.85")
    frame.set_linewidth(0.6)

    # On-panel threshold annotation (top-left, inside the axes above the sparse top ridge)
    if SHOW_THRESHOLD_ANNOTATION and thr_label:
        ax.text(
            0.025, 0.985, thr_label,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=THRESHOLD_ANNOTATION_FS, fontweight="bold", color="#222222",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="0.75", linewidth=0.6, alpha=0.85),
            zorder=200,
        )

    # Panel letter (match 3B)
    if SHOW_PANEL_LETTER and PANEL_LETTER:
        # 0.5 mm in from the top-left corner of the canvas, at the manuscript's letter size. The
        # published call put it at a fixed figure fraction (0.115) in 12 pt: on the composed v4
        # figure that is half again the size of every other panel letter, and the fraction lands
        # somewhere different on every canvas width -- inside the axes on a narrow one.
        fig.text(0.5 / (fig_w / figspec.MM), 1 - 0.5 / (fig_h / figspec.MM), PANEL_LETTER,
                 ha="left", va="top", fontsize=figspec.FS_LETTER, fontweight="bold")

    save_figure(fig, save_base, SAVE_FORMATS, dpi=int(DPI))


# ----------------------------
# Main
# ----------------------------
def main():
    configure_from_args(parse_args())
    df = load_unified(INPUT_TABLE)
    require_cols(df, [
        "threshold", "level", "pocket_class",
        "evidence_n_apo", "evidence_n_holo",
        "pocket_p_apo", "pocket_p_holo",
    ])

    ensure_dir(OUTPUT_DIR)

    for thr in THRESHOLDS_TO_PLOT:
        sub = subset_threshold_unified(df, thr)
        n = len(sub)
        if n == 0:
            print(f"[{thr}] no rows after filtering; skipping.")
            continue

        title = f"{format_row_label(thr)} — probability ridgeline (N={n:,})"
        out_base = os.path.join(OUTPUT_DIR, f"{OUT_BASENAME}_{thr}")

        plot_ridgeline_fullrange_same_top(sub, title=title, save_base=out_base,
                                          thr_label=format_row_label(thr))
        print(f"[{thr}] saved -> {out_base}.{{{','.join(SAVE_FORMATS)}}}")

if __name__ == "__main__":
    main()
