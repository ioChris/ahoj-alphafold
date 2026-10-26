# -*- coding: utf-8 -*-
"""
Figure 1 (REV4) — Pocket-level multi-threshold grid (double-column, full-width)

Goal
- Summarize how **AlphaFold (AF) pocket state** (AF-Apo vs AF-Holo) relates to
  **experimental evidence** in AHoJ/PDB (Exp-Apo vs Exp-Holo), and how this changes
  across UBS clustering thresholds.

Columns
1) AF state counts (AF-Apo vs AF-Holo): neutral fill + colored outline; annotate count + %
2) AF state composition vs shared minimum evidence (common_min) in fixed log2 bins:
   100% stacked bars (AF-Apo vs AF-Holo) with bin N annotated
3) Experimental evidence within each AF state (log y):
   4 lanes: (AF-Apo: Exp-Apo, Exp-Holo+1) and (AF-Holo: Exp-Apo, Exp-Holo+1)
   Scatter is downsampled per lane; box/IQR/median computed from all points.
4) AF state prevalence vs evidence cutoff k:
   For each k, keep entries with common_min >= k and plot prevalence of AF-Apo/AF-Holo (left y),
   and N kept survivors (right y).

Spyder-friendly
- Edit INPUT_CSV / OUTPUT_DIR / OUT_BASENAME below; no CLI parsing.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext, FuncFormatter, MaxNLocator
from matplotlib.legend_handler import HandlerBase

# ----------------------------
# USER SETTINGS (edit in Spyder)
# ----------------------------
# NOTE: For this rendered REV4 in the sandbox, defaults point to /mnt/data.
# In Spyder, replace INPUT_CSV/OUTPUT_DIR with your local paths.
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


INPUT_CSV    = str(cfg.ANALYSIS_UNIFIED_LEVELS_CSV)
OUTPUT_DIR   = str(cfg.PLOTS_DIR)
POSTER_OUTPUT_DIR = str(cfg.PRESENTATION_DIR)

OUT_BASENAME = "Figure1_main_REV4"

# Thresholds to plot (in order). If None, auto-detect and sort.
THRESHOLDS_ORDER = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050"]
# THRESHOLDS_ORDER = None
SHOW_PANEL_LETTERS = True
TRANSPARENT_BACKGROUND = False

HOLO_PLUS_ONE = 1.0  # pocket-level adjustment: Exp-Holo(+1)

# Figure typography (print-safe)
TITLE_FS    = figspec.FS_TITLE      # 7.0 -- see common/figspec.py
TICK_FS     = figspec.FS_TICK       # 6.0
LAB_FS      = figspec.FS_TITLE      # 7.0
LAB_S_FS    = figspec.FS_NOTE       # 6.0
NOTE_COLOR  = "0.25"   # grey used for annotations

# Colors for AF state (consistent across figure)
#APO_COLOR  = "#1f77b4"
#HOLO_COLOR = "#ff7f0e"

APO_COLOR  = figspec.APO   # single source: common/figspec.py
HOLO_COLOR = figspec.HOLO


# Neutral fills / outlines
BAR_FACE = (0.92, 0.92, 0.92, 1.0)
BAR_LW   = 1.7

# ----------------------------
# Column widths (narrow Col1, wider Col4)
# ----------------------------
#WIDTH_RATIOS = [0.78, 1.05, 1.28, 1.22]
#WIDTH_RATIOS = [0.68, 0.9, 0.8, 1.65]
# 7 columns: Col1, SP12, Col2, SP23, Col3, SP34, Col4
# SP12: tiny gutter so Col2 y-ticks don't render over Col1
# SP23/SP34: bigger gutters for y-labels of Col3 and Col4
#WIDTH_RATIOS = [0.68, 0.06, 0.90, 0.24, 0.82, 0.26, 1.65]
WIDTH_RATIOS = [0.68, 0.075, 0.90, 0.24, 0.82, 0.26, 1.65]
SINGLE_ROW_WIDTH_RATIOS = [0.68, 0.22, 0.90, 0.24, 0.82, 0.26, 1.65]

# Data columns, 1-based, in manuscript order. The grid interleaves a spacer between neighbours, so
# the full figure is 7 grid columns for 4 data columns. COLUMNS selects which data columns render;
# --columns takes the same 1-based numbers. Panel letters follow the SELECTION, not the original
# position, so `--columns 2,4` letters them a,b -- the manuscript v4 figures reuse single columns of
# this grid and need their own letter sequence.
COLUMNS = [1, 2, 3, 4]
FIG_SIZE_OVERRIDE = None    # (width, height) in inches; None = derive from the column selection
EXACT_CANVAS = False        # save the full canvas (no tight crop) so --fig_size is the
                            # size on disk and composed parts tile without rescaling
_COL_W_IDX     = {1: 0, 2: 2, 3: 4, 4: 6}   # index into WIDTH_RATIOS for the column itself
_COL_SPACER_IDX = {2: 1, 3: 3, 4: 5}        # index of the spacer that sits to the column's LEFT
PANEL_LETTERS = "abcdefgh"
YLAB_GAP_MM = 7.5           # gap between a column's axes and its shared y label
COUNT_YLAB_GAP_MM = None    # the "Pocket count" label's own gap (--count_ylab_gap); None = YLAB_GAP_MM.
                            # Column 2's "17.5k" ticks are wider than the other columns' numerals,
                            # and at 7.5 mm the label touched them (author, 2026-10-07)
LETTERS_AT_YLABEL = False   # --letters_at_ylabel: each panel letter's left edge on its y label's
                            # left edge, LETTER_DY_MM above the axes -- up and left of the plot, as in
                            # the other v4 figures, rather than over the axes' left corner
LETTER_DY_MM = 1.2
SHOW_ROW_LABELS = True      # the per-threshold label down the left edge; drop it when this strip
                            # is composed beside another that already names the threshold
LETTER_START = 0            # index into PANEL_LETTERS for this render's FIRST panel; set by
                            # --letter_start so a strip composed as the 2nd/3rd block of a
                            # figure carries the right letters instead of restarting at "a"

# Column 2 bins (fixed, 5 bins)
#LOG2_EDGES  = [1, 2, 4, 8, 16, float("inf")]
#LOG2_LABELS = ["1", "2-3", "4-7", "8-15", ">15"] #  ">=16"]
COL2_BINS = {
    "min": {"edges":  [1, 2, 4, 8, 16, float("inf")],
            "labels": ["1", "2-3", "4-7", "8-15", ">15"],},
    # avg tends to be larger than min -> push the top edge up so bins stay informative
    "avg": {"edges":  [1, 4, 8, 16, 64, float("inf")],
            "labels": ["1-3", "4-7", "8-15", "16-63", ">63"],},}
# Column 2 binning mode:
# "min" = current behavior: common_min = min(Exp-Apo, Exp-Holo(+1))
# "avg" = new behavior: shared_avg = 0.5*(Exp-Apo + Exp-Holo(+1))
COL2_BIN_MODE = "min"   # change to "avg" to test the alternative
# Column 2 "overall" pair (--col2_overall; manuscript v4 Figure 2a, 2026-10-07): the class totals as a
# sixth group behind a dotted divider, the idiom of v4 Figures 3a and 4a/b. The totals are ~4-5x the
# largest bin, so on the bins' axis the bins would flatten to a fifth of their height; the pair gets
# its OWN scale, read on a right-hand axis, with no gridlines across it, and prints its counts.
# Off by default so the four-column Figure 1 grid renders as before.
COL2_OVERALL = False
COL2_OVERALL_W = 1.30   # col2 width factor: the axis gains 1.7 x units (divider + pair) on 5.2
COL2_OVERALL_SP = 0.10  # extra width ratio for the spacer right of col2 (the right-hand ticks)

# Column 3 scatter/box styling
DOT_COLOR   = "0.10" # doesn't apply if COLOR_DOTS
DOT_ALPHA   = 0.22   # lower than the greyscale value: two tinted clouds now overlap
DOT_SIZE    = 12 # 14
MAX_SCATTER = 650
JITTER      = 0.018 # 0.017 # 0.030
COLOR_DOTS_AF_CLASS  = False
COLOR_DOTS_EVIDENCE  = True   # tint each lane's cloud by the evidence channel it holds

# Box outline + whiskers + caps (IQR). Was 0.70 (light grey): at 600 dpi the IQR box and the 5-95%
# whiskers all but vanished behind the tinted scatter, so the medians floated without a box
# (author, skeleton review 2026-09-10, item 7). Darker, same weight.
STAT_COLOR  = "0.35"
STAT_LW     = 1.2
WHISKER_LW  = 1.0
#BOX_FACE    = "0.86"
#BOX_FACE    = "0.96" # very light grey
BOX_FACE = "white"
BOX_ALPHA   = 0.65 # 0.60
MEDIAN_LW   = 3.0
RECT_W      = 0.14 # 0.13 # IQR box width, was 0.22
# Column 3 horizontal compactness (scale all x positions around the mid-point)
COL3_X_COMPRESS = 0.90  # 0.60   # 1.00 = current; try 0.92, 0.90, 0.88
COL3_DX = 0.10          # half-spacing within group (gap = 2*COL3_DX). Old gap 0.30 => DX=0.15
COL3_GROUP_SHIFT = 0.37 # 0.33   # try 0.05–0.20 (higher values reduce the AF-Apo ↔ AF-Holo gap)
BORDER_PAD_RAW = 0.10   # was 0.50 (because 0.5 is 0.35 left of 0.85)
# Column 3 legend style:
# "vertical" = rotated key in the mid-gap (your current one)
# "box"      = standard 2-row legend box in the first row only
COL3_LEGEND_STYLE = "box"   # "vertical" or "box"

if COL3_LEGEND_STYLE == "box":
    COL3_GROUP_SHIFT = COL3_GROUP_SHIFT + 0.09

# Column 3 evidence-channel colours. These WERE two greys, on the reasoning that reusing the AF
# class colours would make a reader take a lane for a class. The panel then carried no colour at
# all while every other apo/holo distinction in the manuscript is coloured, which cost more in
# readability than the ambiguity cost. Resolved by letting HUE mean apo/holo throughout and ROLE
# separate the two levels:
#   * lanes  = the evidence channel -> apo/holo hue, named n_apo / n_holo in the key
#   * groups = the AF class         -> apo/holo hue on the x tick labels, bold
# The median bars keep their white notch, which now marks "this is evidence, not a class".
EV_APO_MED_COLOR  = figspec.APO
EV_HOLO_MED_COLOR = figspec.HOLO


# Column 4 settings
K_MAX = None  # auto from data (capped below)
K_CAP = 500   # safety cap for plot x-range

# Column 4's survivor curves sit on the RIGHT axis and count retained pockets; the left axis
# already carries AF class prevalence in the apo/holo pair. These must therefore stay NEUTRAL --
# they used to borrow EV_*_MED_COLOR, which was grey until column 3 took the class colours, at
# which point panel b acquired a blue and an orange dashed line meaning something else entirely.
SURV_COMMON_COLOR = "0.18"
SURV_APO_COLOR    = "0.25"   # was EV_APO_MED_COLOR when that constant was a grey
SURV_HOLO_COLOR   = "0.55"   # ditto EV_HOLO_MED_COLOR
SURV_COMMON_LS = "--"
SURV_APO_LS    = "-."
SURV_HOLO_LS   = ":"


# ----------------------------
# Font preference (tries in order)
# ----------------------------
FONT_PREFERENCE = ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"]

# ----------------------------
# Print-safe rcParams (once)
# ----------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_PREFERENCE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "axes.titlepad": 4,
})

# ----------------------------
# Terminology (avoid Apo/Holo ambiguity)
# ----------------------------
AF_APO_LAB   = "AF-Apo"   # class suffix dropped: matches the legend and the other figures
AF_HOLO_LAB  = "AF-Holo"  # ditto
# Settled 2026-08-24: the experimental counts are named n_apo / n_holo with real subscripts,
# everywhere in the manuscript and its figures. "Exp-Apo"/"Exp-Holo" read as another class label
# next to AF-Apo/AF-Holo, when they are counts of structures; the subscripted form says so.
EXP_APO_LAB  = figspec.N_APO
EXP_HOLO_LAB = figspec.N_HOLO + f" + {HOLO_PLUS_ONE:g}"
COMMON_MIN_LAB = r"min($n_\mathrm{apo}$, $n_\mathrm{holo}$)"


# ----------------------------
# Helpers
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Plot Figure 1 main panel grid.")
    p.add_argument("--input_csv", default=INPUT_CSV)
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--out_basename", default=OUT_BASENAME)
    p.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated thresholds to plot, e.g. ST_100. Defaults to THRESHOLDS_ORDER.",
    )
    p.add_argument(
        "--columns",
        default=None,
        help="Comma-separated data columns to render, 1-based: 1=AF class counts, 2=class "
             "composition in shared-evidence bins, 3=experimental evidence per class, "
             "4=prevalence vs shared-evidence cutoff k. Defaults to all four. Panel letters "
             "re-sequence from 'a' over the selection.",
    )
    p.add_argument("--fig_size", default=None, metavar="W,H",
                   help="canvas in inches, overriding the size derived from the column "
                        "selection. A narrow subset (one column) otherwise inherits a canvas "
                        "too small for its tick labels.")
    p.add_argument("--exact_canvas", action="store_true",
                   help="save the FULL canvas rather than a tight crop, so the file is "
                        "exactly --fig_size. Use for parts that will be tiled into a "
                        "composed figure: a tight crop trims by however much outboard "
                        "label each panel happens to carry, which is what forces a "
                        "composer to rescale rows to make them line up.")
    p.add_argument("--poster_output_dir", action="store_true", help="Write to the configured poster plots directory.")
    p.add_argument("--hide_row_labels", action="store_true",
                   help="drop the threshold label down the left edge. Use when this strip is one "
                        "block of a composed figure whose other blocks already name it.")
    p.add_argument("--letter_start", default=None, metavar="L",
                   help="letter for this render's first panel (e.g. 'c'), so a strip that is "
                        "composed after another block continues the sequence. Implies letters "
                        "are drawn even for a single column.")
    p.add_argument("--hide_panel_letters", action="store_true")
    p.add_argument("--transparent_background", action="store_true")
    p.add_argument("--count_ylab_gap", type=float, default=None, metavar="MM",
                   help="gap in mm between the axes and the 'Pocket count' label (default: the "
                        "shared YLAB_GAP_MM); the left margin grows to match")
    p.add_argument("--letters_at_ylabel", action="store_true",
                   help="put each panel letter up and left of its plot, aligned with its y label, "
                        "at a fixed height above the axes (manuscript v4 Figure 2)")
    p.add_argument("--col2_overall", action="store_true",
                   help="add an 'overall' AF-Apo / AF-Holo pair to column 2, behind a dotted "
                        "divider and on its own right-hand scale (manuscript v4 Figure 2a)")
    return p.parse_args()


def configure_from_args(args):
    global INPUT_CSV, OUTPUT_DIR, OUT_BASENAME, THRESHOLDS_ORDER, COLUMNS, FIG_SIZE_OVERRIDE, LETTER_START, EXACT_CANVAS
    global SHOW_ROW_LABELS, COL2_OVERALL, COUNT_YLAB_GAP_MM, LETTERS_AT_YLABEL
    global SHOW_PANEL_LETTERS, TRANSPARENT_BACKGROUND
    INPUT_CSV = args.input_csv
    OUTPUT_DIR = POSTER_OUTPUT_DIR if args.poster_output_dir else args.output_dir
    OUT_BASENAME = args.out_basename
    if args.thresholds:
        THRESHOLDS_ORDER = [t.strip() for t in args.thresholds.split(",") if t.strip()]
    if args.columns:
        sel = []
        for tok in args.columns.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok not in ("1", "2", "3", "4"):
                raise SystemExit(f"--columns takes 1-4, got {tok!r}")
            if int(tok) not in sel:
                sel.append(int(tok))
        if not sel:
            raise SystemExit("--columns selected nothing")
        COLUMNS = sorted(sel)
    if args.fig_size:
        try:
            w, h = (float(v) for v in args.fig_size.split(","))
        except ValueError:
            raise SystemExit("--fig_size takes W,H in inches")
        FIG_SIZE_OVERRIDE = (w, h)
    SHOW_PANEL_LETTERS = not args.hide_panel_letters
    EXACT_CANVAS = args.exact_canvas
    SHOW_ROW_LABELS = not args.hide_row_labels
    if args.letter_start:
        L = args.letter_start.strip().lower()
        if L not in PANEL_LETTERS:
            raise SystemExit(f"--letter_start must be one of {PANEL_LETTERS!r}, got {L!r}")
        LETTER_START = PANEL_LETTERS.index(L)
    TRANSPARENT_BACKGROUND = args.transparent_background
    COL2_OVERALL = args.col2_overall
    COUNT_YLAB_GAP_MM = args.count_ylab_gap
    LETTERS_AT_YLABEL = args.letters_at_ylabel


def output_paths():
    return (
        os.path.join(OUTPUT_DIR, OUT_BASENAME + ".png"),
        os.path.join(OUTPUT_DIR, OUT_BASENAME + ".pdf"),
        os.path.join(OUTPUT_DIR, OUT_BASENAME + ".svg"),
    )


def save_outputs(fig):
    output_png, output_pdf, output_svg = output_paths()
    if TRANSPARENT_BACKGROUND:
        fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        for ax in fig.axes:
            ax.set_facecolor("white")
    else:
        fig.patch.set_facecolor("white")
    save_kwargs = {
        "bbox_inches": (None if EXACT_CANVAS else "tight"),
        "pad_inches": 0.03,
        "facecolor": fig.get_facecolor(),
    }
    # poster output is png + svg only; a poster build never consumes the pdf
    targets = (output_svg, output_png) if TRANSPARENT_BACKGROUND else (output_pdf, output_svg, output_png)
    for path in targets:
        try:
            fig.savefig(path, dpi=600, **save_kwargs)
        except PermissionError as exc:
            print(f"[warn] could not save {path}: {exc}")
    return output_png, output_pdf, output_svg


def kfmt(x, pos=None):
    """Format large values as k/M for compact axes."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    x = float(x)
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3:
        k = x/1e3
        return f"{k:.0f}k" if abs(k-round(k)) < 1e-6 else f"{k:.1f}k"
    return f"{x:.0f}"


def threshold_sort_key(t):
    t = str(t)
    if t == "UNCLUSTERED":
        return (0, 0, t)
    if t.startswith("ST_") and t[3:].isdigit():
        return (1, -int(t[3:]), t)  # ST_100 before ST_070
    if t.startswith("ST_LJ"):
        return (2, 0, t)
    return (9, 0, t)


def format_row_label(thr):
    if thr == "UNCLUSTERED":
        return  "Unclustered"# "Unclustered (entries)" | "PDB pockets"
    if thr.startswith("ST_") and thr[3:].isdigit():
        if thr[3:].startswith("0"):
            return f"UBS - ST {thr[4:]}%"
        return f"UBS - ST {thr[3:]}%"
        #return f"Clustered UBS (ST {thr[3:]}%)"
    return f"Clustered UBS ({thr})"


def load_df(path):
    usecols = ["threshold", "level", "pocket_class", "evidence_n_apo", "evidence_n_holo", "site_size_unit", "members"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    df["threshold"] = df["threshold"].astype(str)
    df["level"] = df["level"].astype(str).str.lower()
    df["pocket_class"] = df["pocket_class"].astype(str).str.strip().str.lower()

    df["evidence_n_apo"]  = pd.to_numeric(df["evidence_n_apo"], errors="coerce")
    df["evidence_n_holo"] = pd.to_numeric(df["evidence_n_holo"], errors="coerce")
    df["site_size_unit"] = pd.to_numeric(df["site_size_unit"], errors="coerce")
    df["members"] = pd.to_numeric(df["members"], errors="coerce")

    return df


def subset_threshold(df, thr, keep_all_classes=False):
    """UNCLUSTERED => level=entry; others => level=ubs; keep AF {apo,holo}; require evidence_n_apo>=1."""
    lvl = "entry" if thr == "UNCLUSTERED" else "ubs"
    sub = df[(df["threshold"] == thr) & (df["level"] == lvl)].copy()

    if not keep_all_classes:
        sub = sub[sub["pocket_class"].isin(["apo", "holo"])].copy()

    sub = sub[np.isfinite(sub["evidence_n_apo"]) & np.isfinite(sub["evidence_n_holo"])].copy()
    sub = sub[sub["evidence_n_apo"] >= 1].copy()

    sub["evidence_n_holo_adj"] = sub["evidence_n_holo"] + float(HOLO_PLUS_ONE)
    sub = sub[sub["evidence_n_holo_adj"] >= 1].copy()

    sub["common_min"] = np.minimum(sub["evidence_n_apo"].to_numpy(float),
                                   sub["evidence_n_holo_adj"].to_numpy(float))
    return sub


def draw_boxscatter_overlay(ax, v, x0, median_color, rng, dot_color=None):
#def draw_boxscatter_overlay(ax, v, x0, median_color, rng):
    """Scatter (downsampled) + grey box + colored median + IQR/whisker overlay. Scatter is rasterized for vector exports."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return

    if v.size > MAX_SCATTER:
        idx = rng.choice(v.size, size=MAX_SCATTER, replace=False)
        v_s = v[idx]
    else:
        v_s = v

    q25, med, q75 = np.percentile(v, [25, 50, 75])
    wlo, whi      = np.percentile(v, [5, 95])

    xs = x0 + rng.normal(0, JITTER, size=v_s.size)
    #ax.scatter(xs, v_s, s=DOT_SIZE, c=DOT_COLOR, alpha=DOT_ALPHA, linewidths=0, rasterized=True)
    c = DOT_COLOR if dot_color is None else dot_color
    ax.scatter(xs, v_s, s=DOT_SIZE, c=c, alpha=DOT_ALPHA, linewidths=0, rasterized=True)

    #ax.add_patch(Rectangle((x0 - RECT_W/2, q25), RECT_W, q75 - q25,
    #                       facecolor=BOX_FACE, edgecolor=STAT_COLOR, lw=STAT_LW, alpha=BOX_ALPHA))
    # 1) Fill only (alpha), NO edge
    ax.add_patch(Rectangle(
        (x0 - RECT_W/2, q25), RECT_W, q75 - q25,
        facecolor=BOX_FACE, edgecolor="none", alpha=BOX_ALPHA, zorder=2
    ))
    # 2) Outer edge only (alpha)
    ax.add_patch(Rectangle(
        (x0 - RECT_W/2, q25), RECT_W, q75 - q25,
        facecolor="none", edgecolor=STAT_COLOR, lw=STAT_LW, alpha=BOX_ALPHA, zorder=3
    ))

    ax.plot([x0, x0], [wlo, q25], color=STAT_COLOR, lw=WHISKER_LW)
    ax.plot([x0, x0], [q75, whi], color=STAT_COLOR, lw=WHISKER_LW)

    ax.plot([x0 - RECT_W/2, x0 + RECT_W/2], [med, med], color=median_color, lw=MEDIAN_LW, solid_capstyle="butt")
    # white notch in the median bar (distinguish "PDB evidence" from AF-class colors)
    NOTCH_H = 0.08  # in log10 decades; tweak 0.06–0.10
    y1 = med * (10 ** (-NOTCH_H/2))
    y2 = med * (10 ** ( NOTCH_H/2))
    ax.plot([x0, x0], [y1, y2], color="white", lw=1.2, solid_capstyle="butt", zorder=7)

    # Re-draw IQR outline on top (crisper than the alpha-filled edge)
    #ax.add_patch(Rectangle((x0 - RECT_W/2, q25), RECT_W, q75 - q25,
    #                       facecolor="none", edgecolor=STAT_COLOR, lw=STAT_LW, zorder=5, alpha=1.0))

    # 3) Crisp inner outline (solid), drawn last
    INSET = 0.00  # set to 0.01 if you want it visibly "inside"
    ax.add_patch(Rectangle(
        (x0 - RECT_W/2 + INSET, q25), RECT_W - 2*INSET, q75 - q25,
        facecolor="none", edgecolor=STAT_COLOR, lw=STAT_LW, alpha=1.0, zorder=6 # lw=STAT_LW-0.7
    ))


def prevalence_vs_k_fast(sub, kmax):
    """Vectorized prevalence curves for common_min >= k."""
    cm = sub["common_min"].to_numpy(float)
    cls = sub["pocket_class"].to_numpy(str)

    k = np.arange(1, kmax + 1)

    order = np.argsort(cm)[::-1]
    cm_s = cm[order]
    cls_s = cls[order]

    isA = (cls_s == "apo").astype(int)
    isH = (cls_s == "holo").astype(int)

    cut = np.searchsorted(-cm_s, -k, side="right")  # number with cm_s >= k
    n_kept = cut

    cA = np.cumsum(isA)
    cH = np.cumsum(isH)

    keptA = np.where(n_kept > 0, cA[n_kept - 1], 0)
    keptH = np.where(n_kept > 0, cH[n_kept - 1], 0)

    pA = np.where(n_kept > 0, keptA / n_kept * 100.0, np.nan)
    pH = np.where(n_kept > 0, keptH / n_kept * 100.0, np.nan)

    return pd.DataFrame({"k": k, "p_apo": pA, "p_holo": pH, "n_kept": n_kept})


def survivors_vs_k_fast(values, kmax):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        k = np.arange(1, kmax + 1)
        return k, np.zeros_like(k, dtype=int)

    v = np.sort(v)[::-1]  # descending
    k = np.arange(1, kmax + 1)
    n_kept = np.searchsorted(-v, -k, side="right")
    return k, n_kept

class SquareNotch:
    def __init__(self, facecolor):
        self.facecolor = facecolor

class HandlerSquareNotch(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        fc = orig_handle.facecolor
        # colored square
        sq = Rectangle((xdescent, ydescent), width, height,
                       facecolor=fc, edgecolor=fc, transform=trans)
        # inner white vertical notch
        xmid = xdescent + 0.5 * width
        notch = Line2D([xmid, xmid],
                       [ydescent + 0.2 * height, ydescent + 0.8 * height],
                       color="white", lw=1.0, transform=trans, solid_capstyle="butt")
        return [sq, notch]

# ----------------------------
# Plot panels
# ----------------------------
def plot_col1_counts(ax, sub, sub_all=None):
    counts = sub["pocket_class"].value_counts().reindex(["apo", "holo"]).fillna(0).astype(int)
    total = int(counts.sum())
    n_tie = 0
    if sub_all is not None and "pocket_class" in sub_all.columns:
        n_tie = int((sub_all["pocket_class"] == "tie").sum())


    x = np.arange(2)
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="0.88", lw=0.7, zorder=0)

    # bars (white fill, colored outlines)
    ax.bar(x[0], counts["apo"],  color=APO_COLOR, edgecolor=APO_COLOR,  linewidth=BAR_LW, zorder=2)
    ax.bar(x[1], counts["holo"], color=HOLO_COLOR, edgecolor=HOLO_COLOR, linewidth=BAR_LW, zorder=2)

    ax.set_xticks(x)
    #ax.set_xticklabels([AF_APO_LAB, AF_HOLO_LAB])  # let global tick_params control size
    ax.set_xticklabels([AF_APO_LAB, AF_HOLO_LAB], fontsize=LAB_FS) # TICK_FS or LAB_FS (larger)
    ax.set_ylabel("Count", fontsize=LAB_FS)
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))

    ymax = max(1, int(counts.max()))
    ax.set_ylim(0, ymax * 1.25)

    # Compute mean, weighted mean and median of used sites
    #mean_size = float(np.nanmean(sub["site_size_unit"])) if "site_size_unit" in sub.columns else np.nan
    #median_size = float(np.nanmedian(sub["site_size_unit"])) if "site_size_unit" in sub.columns else np.nan
    #size_mean_str = f"avg. size={mean_size:.1f} aa" if np.isfinite(mean_size) else "mean size=NA"
    #size_median_str = f"median size={median_size:.1f} aa" if np.isfinite(median_size) else "median size=NA"

    # --- member-weighted mean site size (each UBS weighted by how many entries it represents) ---
    #if ("site_size_unit" in sub.columns) and ("members" in sub.columns):
    #    x = sub["site_size_unit"].to_numpy(float)
    #    w = sub["members"].to_numpy(float)
    #    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    #    wmean_size = float(np.sum(x[m] * w[m]) / np.sum(w[m])) if np.any(m) else np.nan
    #else:
    #    wmean_size = np.nan
    #size_wmean_str = f"avg. size*={wmean_size:.1f} aa" if np.isfinite(wmean_size) else "avg. size*=NA" # "avg. size (wtd)=NA"

    tie_str = f"excluded ties={n_tie:,}" if n_tie else "excluded ties=0"
    #ax.text(0.02, 0.98, f"N={total:,}\n{size_wmean_str}\n{tie_str}",
    ax.text(0.02, 0.98, f"N={total:,}\n{tie_str}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=LAB_S_FS, color=NOTE_COLOR, linespacing=1.05)

    # place value+percent just *under* the top of each bar if possible
    for i, cls in enumerate(["apo", "holo"]):
        c = int(counts[cls])
        pct = (c / total * 100.0) if total else 0.0

        # y position logic: inside bar near top when tall; else above
        inside_margin = 0.04 * ymax   # vertical inset from bar top
        if c >= 2 * inside_margin:
            y = c - inside_margin
            va = "top"
        else:
            y = c + 0.02 * ymax
            va = "bottom"

        #ax.text(i, y, f"{c:,}\n({pct:.1f}%)",
        ax.text(i, y, f"{c:,}\n{pct:.1f}%",
                ha="center", va=va, fontsize=7.0, color=NOTE_COLOR, zorder=3)


def plot_col2_bins_grouped_counts(ax, sub, show_key=False):

    if COL2_BIN_MODE == "avg":
        bin_val = 0.5 * (sub["evidence_n_apo"].to_numpy(float) + sub["evidence_n_holo_adj"].to_numpy(float))
    else:
        bin_val = sub["common_min"].to_numpy(float)

    edges  = COL2_BINS[COL2_BIN_MODE]["edges"]
    labels = COL2_BINS[COL2_BIN_MODE]["labels"]

    bins = pd.cut(bin_val, bins=edges, right=False, labels=labels, include_lowest=True)

    tmp = sub.copy()
    tmp["bin"] = bins.astype(str)
    tmp["bin_value"] = bin_val
    
    tab = tmp.pivot_table(index="bin", columns="pocket_class", values="bin_value", aggfunc="size", fill_value=0)
    tab = tab.reindex(labels, fill_value=0)

    nA = tab.get("apo", 0).to_numpy(int)
    nH = tab.get("holo", 0).to_numpy(int)

    x = np.arange(len(labels))
    w = 0.32    #w = 0.36
    gap = 0.11

    ax.set_axisbelow(True)
    if not COL2_OVERALL:
        ax.grid(axis="y", which="major", color="0.88", lw=0.7, zorder=0)

    ax.bar(x - (w/2 + gap/2), nA, width=w, color=APO_COLOR, edgecolor=APO_COLOR, linewidth=BAR_LW, zorder=2) # color=BAR_FACE
    ax.bar(x + (w/2 + gap/2), nH, width=w, color=HOLO_COLOR, edgecolor=HOLO_COLOR, linewidth=BAR_LW, zorder=2) # color=BAR_FACE

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FS)
    ax.set_ylabel("Count", fontsize=LAB_FS)
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))

    ymax = max(1, int(max(nA.max(), nH.max())))
    ax.set_ylim(0, ymax * 1.25)

    if COL2_OVERALL:
        draw_col2_overall(ax, sub, nA, nH, w, gap, labels, bins)

    if show_key:
        handles = [
            Line2D([0], [0], marker="s", linestyle="None", markersize=5.0,
                   markerfacecolor=APO_COLOR, markeredgecolor=APO_COLOR, label="AF-Apo"), # "Apo classified"
            Line2D([0], [0], marker="s", linestyle="None", markersize=5.0,
                   markerfacecolor=HOLO_COLOR, markeredgecolor=HOLO_COLOR, label="AF-Holo"), # "Holo classified"
        ]
        ax.legend(handles=handles,
                  loc="upper left",
                  bbox_to_anchor=(0.02, 0.98),
                  frameon=False,
                  ncol=1,                 # <-- forces two rows
                  handletextpad=0.30,
                  borderaxespad=0.0,
                  prop={"size": 5.6})


def col2_overall_geometry(nb):
    """x of the overall pair's centre, the divider, and the axis limits, in bin units."""
    return nb + 0.6, nb - 0.15, -0.6, nb + 1.3


def col2_bins_centre():
    """Centre of the binned part of column 2 as an axes fraction, so the axis title sits under
    the bins it names rather than under the bins plus the overall pair (author, 2026-10-07)."""
    _, x_div, x_lo, x_hi = col2_overall_geometry(len(COL2_BINS[COL2_BIN_MODE]["labels"]))
    return (0.5 * (x_lo + x_div) - x_lo) / (x_hi - x_lo)


def draw_col2_overall(ax, sub, nA, nH, w, gap, labels, bins):
    """The 'overall' pair of column 2: class totals behind a dotted divider, on their own scale.

    Same geometry as v4 Figure 3a's overall group (centre nb + 0.6, divider nb - 0.15, hatched
    bars, bold 'overall' tick). The bins' gridlines stop at the divider, because they are on the
    bins' scale; the pair is read on the right-hand axis and by its printed counts.
    """
    cls = sub["pocket_class"].astype(str).str.lower()
    oA, oH = int((cls == "apo").sum()), int((cls == "holo").sum())
    n_unbinned = int(pd.isna(bins).sum())
    print(f"  col2 overall: AF-Apo {oA:,} / AF-Holo {oH:,} pockets "
          f"(binned {int(nA.sum()):,} / {int(nH.sum()):,}; {n_unbinned:,} in no bin)")

    nb = len(labels)
    x_ov, x_div, x_lo, x_hi = col2_overall_geometry(nb)
    top = ax.get_ylim()[1]
    # bins' units per overall pocket: the taller of the pair at 74% of the axis, about where the
    # tallest bin sits (80%), so neither group looks smaller; the rest is room for its rotated count
    k = top / (1.35 * max(oA, oH))
    ax.set_xlim(x_lo, x_hi)
    for y in ax.get_yticks():
        if 0 < y < top:
            ax.plot([x_lo, x_div], [y, y], color="0.88", lw=0.7, zorder=0)
    ax.axvline(x_div, color="0.75", lw=0.9, ls=":", zorder=1)
    for v, off, col in [(oA, -(w/2 + gap/2), APO_COLOR), (oH, +(w/2 + gap/2), HOLO_COLOR)]:
        ax.bar([x_ov + off], [v * k], width=w, color=col, edgecolor="0.3", linewidth=0.6,
               hatch="///", zorder=2)
        ax.annotate(f"{v / 1000:.1f}k", (x_ov + off, v * k), textcoords="offset points",
                    xytext=(0, 2), ha="center", va="bottom", rotation=90,
                    fontsize=LAB_S_FS, color=NOTE_COLOR)
    sec = ax.secondary_yaxis("right", functions=(lambda y: y / k, lambda v: v * k))
    sec.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10], integer=True))
    sec.yaxis.set_major_formatter(FuncFormatter(kfmt))
    sec.tick_params(labelsize=TICK_FS, pad=1)
    ax.set_xticks(list(range(nb)) + [x_ov])
    ax.set_xticklabels(list(labels) + ["overall"], fontsize=TICK_FS)
    ax.get_xticklabels()[-1].set_fontweight("bold")


def plot_col2_bins_stacked(ax, sub):
    #cm = sub["common_min"].to_numpy(float)
    #bins = pd.cut(cm, bins=LOG2_EDGES, right=False, labels=LOG2_LABELS, include_lowest=True)
    if COL2_BIN_MODE == "avg":
        bin_val = 0.5 * (sub["evidence_n_apo"].to_numpy(float) + sub["evidence_n_holo_adj"].to_numpy(float))
    else:
        bin_val = sub["common_min"].to_numpy(float)
    
    bins = pd.cut(bin_val, bins=LOG2_EDGES, right=False, labels=LOG2_LABELS, include_lowest=True)

    tmp = sub.copy()
    tmp["bin"] = bins.astype(str)

    tab = tmp.pivot_table(index="bin", columns="pocket_class", values="common_min", aggfunc="size", fill_value=0)
    tab = tab.reindex(LOG2_LABELS, fill_value=0)

    n_bin = tab.sum(axis=1).to_numpy(float)
    fracA = np.where(n_bin > 0, tab.get("apo", 0).to_numpy(float) / n_bin * 100.0, 0.0)
    fracH = np.where(n_bin > 0, tab.get("holo", 0).to_numpy(float) / n_bin * 100.0, 0.0)

    x = np.arange(len(LOG2_LABELS))
    ax.bar(x, fracA, color=BAR_FACE, edgecolor=APO_COLOR, linewidth=BAR_LW)
    ax.bar(x, fracH, bottom=fracA, color=BAR_FACE, edgecolor=HOLO_COLOR, linewidth=BAR_LW)

    ax.set_xticks(x)
    ax.set_xticklabels(LOG2_LABELS, fontsize=TICK_FS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of pockets", fontsize=LAB_FS)
    ax.grid(axis="y", color="0.88", lw=0.7)

    for xi, nb in zip(x, n_bin.astype(int)):
        if nb > 0:
            ax.text(xi, 102, f"N={nb:,}", ha="center", va="bottom", fontsize=6.6, color=NOTE_COLOR)


def plot_col3_evidence(ax, sub, rng, ylims=None, show_key=False):
    left  = sub[sub["pocket_class"] == "apo"]
    right = sub[sub["pocket_class"] == "holo"]

    APO_TEXT = "Apo data"
    HOLO_TEXT = "Holo data"

    # --- Column 3 x-geometry (preserve proportions, shrink overall width) ---
    x_mid = 1.50  # center of the whole Column 3 layout (midpoint of 0.5..2.5)
    def sx(x):    # scale x around x_mid
        return x_mid + (x - x_mid) * COL3_X_COMPRESS

    C_APO  = 1.00
    C_HOLO = 2.00 - COL3_GROUP_SHIFT   # keep your existing inter-group control    

    # Lanes encode EXPERIMENTAL channel (Exp-Apo vs Exp-Holo), not AF class colors
    lanes = [
        (left["evidence_n_apo"],       sx(C_APO  - COL3_DX), EV_APO_MED_COLOR),  # APO_COLOR  | EV_APO_MED_COLOR
        (left["evidence_n_holo_adj"],  sx(C_APO  + COL3_DX), EV_HOLO_MED_COLOR), # HOLO_COLOR | EV_HOLO_MED_COLOR
        (right["evidence_n_apo"],      sx(C_HOLO - COL3_DX), EV_APO_MED_COLOR),  # APO_COLOR  | EV_APO_MED_COLOR
        (right["evidence_n_holo_adj"], sx(C_HOLO + COL3_DX), EV_HOLO_MED_COLOR), # HOLO_COLOR | EV_HOLO_MED_COLOR
    ]

    # Dot color policy:
    # - AF class: by group (idx 0–1 = AF-apo, idx 2–3 = AF-holo)
    # - Evidence: by lane parity (idx even = Exp-Apo, idx odd = Exp-Holo)
    for idx, (v, x0, med_col) in enumerate(lanes):
        # idx 0–1 are AF-apo group, idx 2–3 are AF-holo group
        if COLOR_DOTS_EVIDENCE:
            dc = med_col
        elif COLOR_DOTS_AF_CLASS:
            dc = APO_COLOR if (idx < 2) else HOLO_COLOR
        else:
            dc = None
        draw_boxscatter_overlay(ax, v.to_numpy(float), x0, med_col, rng, dot_color=dc)
    #for v, x0, col in lanes:
    #    draw_boxscatter_overlay(ax, v.to_numpy(float), x0, col, rng)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    ax.set_axisbelow(True)                      # grid behind points/boxes
    ax.grid(axis="y", which="major", color="0.88", lw=0.7)
    ax.grid(False, axis="y", which="minor")     # no minor y-gridlines

    ax.set_xlim(sx(0.85 - BORDER_PAD_RAW), sx((2.15 - COL3_GROUP_SHIFT) + BORDER_PAD_RAW))

    # ticks at the group centers
    ax.set_xticks([sx(C_APO), sx(C_HOLO)])
    #ax.set_xticklabels([AF_APO_LAB, AF_HOLO_LAB])  # let global tick_params control size
    ax.set_xticklabels([AF_APO_LAB, AF_HOLO_LAB], fontsize=LAB_FS) # TICK_FS
    # the AF CLASS is the grouping, so it is named in the class colours; the lanes inside each
    # group are the evidence channel and carry the same two hues one level down
    for _t, _c in zip(ax.get_xticklabels(), (APO_COLOR, HOLO_COLOR)):
        _t.set_color(_c)
        _t.set_fontweight("bold")
    #ax.set_ylabel("Evidence count (log)", fontsize=LAB_FS)

    if ylims is not None:
        ax.set_ylim(*ylims)

    # Lift bottom so y=1 points aren't clipped by the axis spine
    y0, y1 = ax.get_ylim()
    ax.set_ylim(0.50, y1) # lower → shift higher (original=0.60)

    if show_key and (COL3_LEGEND_STYLE == "vertical"):

        # x in the middle gap, in DATA coords -> convert to axes coords so y stacking is stable
        x_center_data = sx((C_APO + C_HOLO) / 2.0)
        x_center_axes = ax.transAxes.inverted().transform(ax.transData.transform((x_center_data, 1.0)))[0]

        # vertical stack in AXES coords (bottom -> top)
        y0 = 0.07
        dy = 0.24

        # --- background box behind the vertical key (axes coords) ---
        pad_x = 0.020
        pad_y = 0.035

        box_w = 0.070                     # width in axes fraction; tweak if needed
        #box_h = (3*dy) + 2*pad_y          # covers 4 items from y0..y0+3*dy
        box_h = (3.6*dy) + 2*pad_y   # extra headroom for rotated text

        ax.add_patch(Rectangle(
            (x_center_axes - box_w/2 - pad_x, y0 - pad_y),
            box_w + 2*pad_x, box_h,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="0.90",
            linewidth=0.6,
            alpha=0.95,
            zorder=45,
            clip_on=False
        ))

        items = [
            ("s", APO_COLOR,  y0 + 0*dy, None),        # apo marker
            ("t", NOTE_COLOR, y0 + 1*dy, APO_TEXT),   # apo text
            ("s", HOLO_COLOR, y0 + 2*dy, None),        # holo marker
            ("t", NOTE_COLOR, y0 + 3*dy, HOLO_TEXT),  # holo text
        ]

        for kind, col, ya, txt in items:
            if kind == "s":
                ax.scatter([x_center_axes], [ya],
                           transform=ax.transAxes,
                           s=28, marker="s",
                           c=[col], edgecolors=[col],
                           linewidths=0.0, zorder=50, clip_on=False)
            else:
                ax.text(x_center_axes, ya, txt,
                        transform=ax.transAxes,
                        rotation=90, ha="center", va="center",
                        fontsize=5.6, color=col, zorder=50, clip_on=False)

    if show_key and (COL3_LEGEND_STYLE == "box"):
        #handles = [SquareNotch(APO_COLOR), SquareNotch(HOLO_COLOR)]
        handles = [SquareNotch(EV_APO_MED_COLOR), SquareNotch(EV_HOLO_MED_COLOR)]
        labels  = [figspec.N_APO, figspec.N_HOLO]
        #labels  = ["Apo evidence", "Holo evidence"] "PDB apo states", "PDB holo states" "Exp-Apo (PDB)", "Exp-Holo (PDB)"

        leg = ax.legend(handles=handles, labels=labels,
                        handler_map={SquareNotch: HandlerSquareNotch()},
                        loc="upper left", bbox_to_anchor=(0.02, 0.98),
                        frameon=True, ncol=1,
                        handlelength=1.0, handleheight=0.9,
                        handletextpad=0.45, borderaxespad=0.0,
                        prop={"size": 5.6})

        # semi-transparent white box (like your Column 4 legend)
        leg.get_frame().set_alpha(0.65)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("0.85")
        leg.get_frame().set_linewidth(0.6)


def plot_col4_prevalence(ax, sub, show_ylabel=False):
    cm_max = float(np.nanmax(sub["common_min"])) if len(sub) else 1.0
    kmax = int(cm_max) if K_MAX is None else int(K_MAX)
    kmax = max(1, min(kmax, K_CAP))

    pv = prevalence_vs_k_fast(sub, kmax=kmax)

    ax.plot(pv["k"], pv["p_apo"], color=APO_COLOR, lw=2.0)
    ax.plot(pv["k"], pv["p_holo"], color=HOLO_COLOR, lw=2.0)

    ax.set_ylim(0, 100)
    ax.set_xlim(-5, kmax + 5)
    ax.grid(axis="y", color="0.88", lw=0.7)

    if show_ylabel:
        ax.set_ylabel("AF state prevalence (%)", fontsize=LAB_FS)
    else:
        ax.set_ylabel("")

    axr = ax.twinx()

    # Survivors (right axis): encode "evidence criteria" in neutral greys (consistent with Col3)
    # - min common: darkest + dashed
    axr.plot(pv["k"], pv["n_kept"], color=SURV_COMMON_COLOR, lw=1.4, ls=SURV_COMMON_LS, label="min shared")
    # - min apo: Exp-Apo evidence threshold (SURV_APO_COLOR, neutral) + dash-dot
    k, nA = survivors_vs_k_fast(sub["evidence_n_apo"].to_numpy(float), kmax)
    axr.plot(k, nA, color=SURV_APO_COLOR, lw=1.2, ls=SURV_APO_LS, label="min Exp-Apo")
    # - min holo: Exp-Holo(+1) evidence threshold (SURV_HOLO_COLOR, neutral) + dotted
    k, nH = survivors_vs_k_fast(sub["evidence_n_holo_adj"].to_numpy(float), kmax)
    axr.plot(k, nH, color=SURV_HOLO_COLOR, lw=1.2, ls=SURV_HOLO_LS, label="min Exp-Holo")

    axr.set_ylabel("") # axr.set_ylabel("N kept", fontsize=LAB_FS, labelpad=2)
    axr.yaxis.set_major_formatter(FuncFormatter(kfmt))
    axr.tick_params(axis="y", labelsize=TICK_FS, pad=1)


    if show_ylabel:
        # --- Legend 1 (upper left): solid prevalence lines ---
        p0 = Line2D([0],[0], color=APO_COLOR,  lw=2.0, label="AF-Apo")
        p1 = Line2D([0],[0], color=HOLO_COLOR, lw=2.0, label="AF-Holo")

        leg_pre = ax.legend(
            handles=[p0, p1],
            loc="upper left", bbox_to_anchor=(0.04, 0.98),
            frameon=True, fontsize=5.6, handlelength=2.2,
            borderaxespad=0.0
        )
        leg_pre.get_frame().set_alpha(0.65)
        leg_pre.get_frame().set_edgecolor("0.8")
        leg_pre.get_frame().set_linewidth(0.6)
        leg_pre.get_frame().set_facecolor("white")

        # Keep this legend when we add a second one
        ax.add_artist(leg_pre)

        # --- Legend 2 (upper right): dashed survivors (N kept) ---
        h0 = Line2D([0],[0], color=SURV_COMMON_COLOR, lw=1.4, ls=SURV_COMMON_LS, label="min shared")   # min shared   | N kept (min common) | min shared (min[Exp-Apo, Exp-Holo (+1)])
        h1 = Line2D([0],[0], color=SURV_APO_COLOR,  lw=1.2, ls=SURV_APO_LS,    label=figspec.N_APO + r" $\geq k$")  # min PDB apo  | N kept (min apo)  | Exp-Apo ≥ k
        h2 = Line2D([0],[0], color=SURV_HOLO_COLOR, lw=1.2, ls=SURV_HOLO_LS,   label=figspec.N_HOLO + r" $\geq k$") # min PDB holo | N kept (min holo) | Exp-Holo ≥ k

        # no legend title: it read "Retained pockets", which is the twin axis's own name.
        # With that axis labelled again the title is the same words twice in one panel.
        leg_n = ax.legend(
            handles=[h0, h1, h2],
            loc="upper right", bbox_to_anchor=(0.98, 0.98),
            frameon=True, fontsize=5.6, handlelength=2.2,
            borderaxespad=0.0
        )
        leg_n.get_frame().set_alpha(0.65)
        leg_n.get_frame().set_edgecolor("0.8")
        leg_n.get_frame().set_linewidth(0.6)
        leg_n.get_frame().set_facecolor("white")


def build_global_ylims(subs):
    mins, maxs = [], []
    for sub in subs:
        if sub is None or len(sub) == 0:
            continue
        v = np.concatenate([sub["evidence_n_apo"].to_numpy(float), sub["evidence_n_holo_adj"].to_numpy(float)])
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            mins.append(np.min(v))
            maxs.append(np.max(v))
    if not mins:
        return (1, 10)
    ymin = max(0.9, min(mins))
    ymax = max(maxs)
    ymin = 10 ** np.floor(np.log10(ymin))
    ymax = 10 ** np.ceil(np.log10(ymax))         # keep decade snap for consistency
    ymax = ymax / 2.25                           # shrink headroom (try 1.2–1.4)
    return (ymin, ymax)


def main():
    configure_from_args(parse_args())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_df(INPUT_CSV)

    if THRESHOLDS_ORDER is None:
        thr_list = sorted(df["threshold"].dropna().unique().tolist(), key=threshold_sort_key)
    else:
        thr_list = [t for t in THRESHOLDS_ORDER if t in set(df["threshold"].unique())]

    if not thr_list:
        raise SystemExit("No matching thresholds found to plot.")

    subs_all  = [subset_threshold(df, thr, keep_all_classes=True)  for thr in thr_list]
    subs_used = [subset_threshold(df, thr, keep_all_classes=False) for thr in thr_list]

    ylims = build_global_ylims(subs_used)

    # Grid = the selected data columns, interleaved with the spacer that originally sat to the LEFT
    # of each. Full selection reproduces the published 7-column layout exactly:
    #   [0]=Col1, [1]=SP12, [2]=Col2, [3]=SP23, [4]=Col3, [5]=SP34, [6]=Col4
    sel = list(COLUMNS)
    nrows, ncols = len(thr_list), 2 * len(sel) - 1
    col_ax = {c: 2 * j for j, c in enumerate(sel)}          # data column -> grid index
    spacer_ax = {c: 2 * j - 1 for j, c in enumerate(sel) if j}   # data column -> its left spacer

    base = list(SINGLE_ROW_WIDTH_RATIOS if nrows == 1 else WIDTH_RATIOS)
    if COL2_OVERALL:
        # the overall pair widens column 2's axis by COL2_OVERALL_W at the same width per bin, and
        # its right-hand ticks need room in the spacer of whichever selected column follows it
        base[_COL_W_IDX[2]] *= COL2_OVERALL_W
        nxt = next((c for c in COLUMNS if c > 2), None)
        if nxt is not None:
            base[_COL_SPACER_IDX[nxt]] += COL2_OVERALL_SP
    width_ratios = []
    for j, c in enumerate(sel):
        if j:
            width_ratios.append(base[_COL_SPACER_IDX[c]])
        width_ratios.append(base[_COL_W_IDX[c]])

    full_w = 7.15 if nrows == 1 else 7.0  # ~double-column width for the complete grid
    fig_w = full_w * (sum(width_ratios) / sum(base))
    fig_h = 1.40 * nrows + 0.5
    if FIG_SIZE_OVERRIDE:
        fig_w, fig_h = FIG_SIZE_OVERRIDE
    # margins below were hand-tuned on the full canvas; hold them at constant INCHES so a
    # narrower subset does not silently crush the row labels into the y-axis
    m = full_w / fig_w          # 1.0 for the full grid, so it reproduces the published figure
    # left margin in mm: shared y label + (optionally) the rotated threshold label
    count_gap = YLAB_GAP_MM if COUNT_YLAB_GAP_MM is None else COUNT_YLAB_GAP_MM
    left_gap = count_gap if sel[0] in (1, 2) else YLAB_GAP_MM
    L_MM = (left_gap + 8.5) if SHOW_ROW_LABELS else (left_gap + 4.5)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
        gridspec_kw={"width_ratios": width_ratios},
    )

    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)
    """
    col_titles = {
        0: "AF class counts",
        2: "AF class counts in\nshared evidence bins",
        4: "Experimental evidence\nwithin AF class",
        6: "AF class prevalence (%)\nvs shared-evidence cutoff",
    }
    col_letters = {
        0: "A",
        2: "B",
        4: "C",
        6: "D",
    }
    for j, t in col_letters.items():
        axes[0, j].set_title(t, fontsize=TITLE_FS)
    """

    rng = np.random.default_rng(0)

    # -0.70 was tuned against Col1's width; hold the row label at the same distance in inches
    # whatever the leftmost selected column happens to be
    row_lab_x = -0.70 * base[_COL_W_IDX[1]] / base[_COL_W_IDX[sel[0]]]

    for i, (thr, sub, sub_all) in enumerate(zip(thr_list, subs_used, subs_all)):
        if SHOW_ROW_LABELS:
            axes[i, 0].text(row_lab_x, 0.5, format_row_label(thr),
                            transform=axes[i, 0].transAxes,
                            rotation=90, ha="center", va="center",
                            fontsize=LAB_FS, color="0.10", fontweight="bold")

        for c in sel:
            ax = axes[i, col_ax[c]]
            if c == 1:
                plot_col1_counts(ax, sub, sub_all=sub_all)
            elif c == 2:
                plot_col2_bins_grouped_counts(ax, sub, show_key=(i == 0))
                ax.tick_params(axis="y", pad=1)
            elif c == 3:
                plot_col3_evidence(ax, sub, rng=rng, ylims=ylims, show_key=(i == 0))
            elif c == 4:
                plot_col4_prevalence(ax, sub, show_ylabel=(i == 0))

        for j in range(ncols):
            ax = axes[i, j]

            # spacers carry no data; switch them off and skip
            if j % 2 == 1:
                ax.set_axis_off()
                continue

            # Keep y ticks consistent everywhere
            ax.tick_params(axis="both", labelsize=TICK_FS)

            # Let x ticks be per-panel (so Col1 can keep its intended size)
            #ax.tick_params(axis="y", labelsize=TICK_FS)
            #if j == 0:     ax.tick_params(axis="x", labelsize=LAB_FS)   # Col1: AF-Apo/Holo labels (multi-line)
            #else:          ax.tick_params(axis="x", labelsize=TICK_FS)  # others

            # hide x tick labels for non-bottom rows (but keep the bottom-row xlabels)
            if i != nrows - 1:
                ax.set_xticklabels([])
            # bottom-row x-axis labels (only where needed)
            if i == nrows - 1:
                if 2 in col_ax:
                    axes[i, col_ax[2]].set_xlabel(f"Shared evidence bins ({COL2_BIN_MODE})",
                                                  fontsize=LAB_FS, labelpad=2)
                    if COL2_OVERALL:     # set_xlabel re-centres the title, so this comes after it
                        axes[i, col_ax[2]].xaxis.label.set_x(col2_bins_centre())
                if 4 in col_ax:
                    axes[i, col_ax[4]].set_xlabel("Shared evidence cutoff k",
                                                  fontsize=LAB_FS, labelpad=2)
            ax.set_ylabel("")

    # subplots_adjust FIRST: the shared labels below are positioned from get_position(), and
    # querying that before the adjust returns the DEFAULT gridspec box. The published figure
    # did exactly that and compensated with hand-tuned nudges; now that the margins are real
    # millimetres the nudges are gone, so the positions have to be the real ones.
    # millimetres, not fractions: the same fraction is a different amount of ink on a 46 mm
    # canvas than on a 90 mm one, and the x tick row plus axis title need a fixed ~9 mm.
    # The right margin only has to clear the twin-axis label, which only column 4 carries.
    fig.subplots_adjust(**figspec.margins(fig_w, fig_h, left=L_MM,
                                          right=(13.0 if 4 in col_ax else 1.5),
                                          top=4.5, bottom=9.0),
                        wspace=0.10, hspace=0.10)

    # ---- Shared Y labels ----
    # Each label sits in the spacer to the LEFT of its column; the leftmost selected column has no
    # spacer, so it falls back to the outer margin (which is what Col1 always did).
    # NB positions are queried BEFORE subplots_adjust, as in the published figure -- the nudges
    # below were tuned against that ordering, so do not hoist the adjust call above this block.
    mid_y = 0.5 * (axes[0, 0].get_position().y1 + axes[-1, 0].get_position().y0)


    ylab_txt = {}

    def shared_ylabel(col, text, push=None, gap_mm=YLAB_GAP_MM):
        """Place the label a fixed distance left of its column's axes.

        `push` is accepted and ignored. It used to be a nudge expressed as a fraction of the
        spacer's width, which stopped meaning anything once the canvas became the final canvas:
        at 180 mm with two columns the spacer is wide enough that the published 0.98 push put the
        label a centimetre INSIDE the neighbouring panel.
        """
        x = axes[0, col_ax[col]].get_position().x0 - gap_mm / (fig_w / figspec.MM)
        ylab_txt[col] = fig.text(x, mid_y, text, rotation=90, va="center", ha="center",
                                 fontsize=LAB_FS)

    # Col1 and Col2 both count pockets, so the label goes on whichever of them is leftmost
    count_col = next((c for c in sel if c in (1, 2)), None)
    if count_col is not None:
        shared_ylabel(count_col, "Pocket count", push=0.0, gap_mm=count_gap)
    if 3 in col_ax:
        shared_ylabel(3, "Evidence per pocket (log)", push=0.58)
    if 4 in col_ax:
        shared_ylabel(4, "AF class prevalence (%)", push=0.98)

    # ---- Shared right-side Y label for Col4's twin axis ----
    if 4 in col_ax:
        pos_c4 = axes[0, col_ax[4]].get_position()
        mid_y4 = 0.5 * (pos_c4.y1 + axes[-1, col_ax[4]].get_position().y0)
        # Millimetres, like every other margin here. This used to be `x1 + 0.145 * m`, a
        # figure FRACTION: on the 180 mm v4 canvas that is x = 1.07, i.e. off the sheet. Under
        # bbox_inches="tight" the crop pulled it back and it survived; under --exact_canvas
        # nothing pulls it back, so panel b simply lost its right-hand axis name.
        # The 13 mm right margin holds the twin tick numbers (~6.5 mm) and then this label.
        fig.text(
            pos_c4.x1 + 9.0 / (fig_w / figspec.MM), mid_y4, "Retained pockets",
            rotation=270, va="center", ha="center",  # right-axis label reads top->bottom (clockwise)
            fontsize=LAB_FS
        )

    # With spacer columns, you can reduce global wspace; spacers control the key gaps.
    # The left margin scales so the hand-tuned value keeps its width in INCHES on a narrower canvas.

    # ---- Panel letters, anchored to axes (robust to subplots_adjust + tight bbox) ----
    PANEL_FS = TITLE_FS + 2
    panel_y = 1.03  # push above axes frame (try 1.02–1.06)
    panel_x = 0.00  # left edge of axes

    if SHOW_PANEL_LETTERS and (len(sel) > 1 or LETTER_START) and LETTERS_AT_YLABEL:
        # left edge of the panel's own y label (or, without one, of its y tick labels), read back
        # from the drawn extent so it follows whatever the label and ticks turn out to be
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        for k, c in enumerate(sel):
            ax0 = axes[0, col_ax[c]]
            if c in ylab_txt:
                x0 = ylab_txt[c].get_window_extent().x0
            else:
                ext = [t.get_window_extent().x0 for t in ax0.get_yticklabels() if t.get_text()]
                x0 = min(ext) if ext else ax0.get_window_extent().x0
            fig.text(inv.transform((x0, 0))[0],
                     ax0.get_position().y1 + LETTER_DY_MM / (fig_h / figspec.MM),
                     PANEL_LETTERS[LETTER_START + k], ha="left", va="bottom",
                     fontsize=PANEL_FS, fontweight="bold")
    elif SHOW_PANEL_LETTERS and (len(sel) > 1 or LETTER_START):
        for k, c in enumerate(sel):
            ax0 = axes[0, col_ax[c]]
            ax0.text(
                panel_x, panel_y, PANEL_LETTERS[LETTER_START + k],
                transform=ax0.transAxes,
                ha="left", va="bottom",
                fontsize=PANEL_FS, fontweight="bold",
                clip_on=False
            )

    # Slightly larger pad_inches helps twin-axis tick labels + bottom x-label survive "tight"
    output_png, output_pdf, output_svg = save_outputs(fig)
    plt.close(fig)

    print("Saved:")
    if not TRANSPARENT_BACKGROUND:      # poster mode writes no pdf
        print(" -", output_pdf)
    print(" -", output_svg)
    print(" -", output_png)


if __name__ == "__main__":
    main()
