# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 13:34:09 2026

@author: Christos P. Feidakis
"""

# -*- coding: utf-8 -*-
"""
Figure 2 (REV1) — Unified-table-only grid:
Rows = thresholds, Cols = (1) AF class counts vs evidence ratio, (2) evidence trendlines vs ratio, (3) pLDDT vs ratio hexbin

Inputs:
  analysis_unified_levels.csv (or a .zip containing it)

Required columns:
  threshold, level, pocket_class, evidence_n_apo, evidence_n_holo, model_pocket_plddt

Filtering:
  - UNCLUSTERED => level == "entry"
  - others      => level == "ubs"
  - pocket_class in {"apo","holo"}
  - evidence_n_apo >= 1
  - evidence_n_holo_adj = evidence_n_holo + HOLO_PLUS_ONE
  - ratio = evidence_n_holo_adj / evidence_n_apo
  - x = log10(ratio)
"""

import argparse
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon
from matplotlib.ticker import LogLocator, LogFormatterMathtext, FuncFormatter
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib.ticker import FuncFormatter

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
OUTPUT_DIR   = str(cfg.PLOTS_DIR)
POSTER_OUTPUT_DIR = str(cfg.PRESENTATION_DIR)
OUT_BASENAME = "Figure2_main_REV1"

# Threshold rows to plot (order matters)
THRESHOLDS_TO_PLOT = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050"]
# Data columns, 1-based: 1 = classification prevalence vs ratio, 2 = pocket counts vs ratio,
# 3 = pLDDT hexbin vs ratio. COLUMNS selects which render; --columns takes the same numbers.
# Panel letters re-sequence from "a" over the SELECTION -- the v4 figures reuse single columns.
COLUMNS = [1, 2, 3]
FIG_SIZE_OVERRIDE = None    # (width, height) in inches; None = derive from the column selection
EXACT_CANVAS = False        # save the full canvas (no tight crop) so --fig_size is the
                            # size on disk and composed parts tile without rescaling
COL_WIDTH_RATIOS = [1.05, 1.05, 0.90]   # per data column, indexed 1-based via COLUMNS
PANEL_LETTERS = "abcdefgh"
YLAB_GAP_MM = 7.5           # gap between a column's axes and its shared y label
SHOW_ROW_LABELS = True      # the per-threshold label down the left edge; drop it when this
                            # strip is composed under another that already names the threshold
LETTER_START = 0            # index into PANEL_LETTERS for this render's FIRST panel; set by
                            # --letter_start so a strip composed as the 2nd/3rd block of a
                            # figure carries the right letters instead of restarting at "a"
SHOW_PANEL_LETTERS = True
LETTERS_AT_YLABEL = False   # --letters_at_ylabel: each panel letter's left edge on its y label's
                            # left edge, LETTER_DY_MM above the axes (manuscript v4 Figure 2, as in
                            # figure1_main_REV4.py) rather than over the axes' left corner
LETTER_DY_MM = 1.2
SHOW_DIRECTION_CUE = True   # coloured "More apo" / "More holo" tags ON the ratio axis, in
                            # place of the +-1.5 numerals (see apply_ratio_tags)
TRANSPARENT_BACKGROUND = False

HOLO_PLUS_ONE = 1.0

# X axis range requested
X_MIN, X_MAX = -3.0, 3.0

# Left vertical labels position
ROW_LABEL_X = -0.42

# Binning for Columns 1/2 (fixed edges for reproducibility)
# (0.1 step gives 60 bins across [-3,3])
BIN_W = 0.10
X_EDGES = np.arange(X_MIN, X_MAX + BIN_W, BIN_W)
X_CENTERS = (X_EDGES[:-1] + X_EDGES[1:]) / 2.0

# Column 1 y-scale (counts)
#COL1_LOG_Y = True

# Column 2 options
COL2_LOG_Y = True       # Plot Y as log or regular count
SHOW_IQR_RIBBON = True  # Set False for cleaner print
EVIDENCE_TREND_STAT = "median"  # "median" or "mean"

# Column 3 hexbin
GRIDSIZE = 45
MINCNT = 1


# Typography / style (match Figure 1 spirit)
TITLE_FS = figspec.FS_TITLE     # 7.0 -- see common/figspec.py
TICK_FS  = figspec.FS_TICK      # 6.0
LAB_FS   = figspec.FS_TITLE     # 7.0
SMALL_FS = figspec.FS_NOTE      # 6.0

APO_COLOR  = figspec.APO   # single source: common/figspec.py
HOLO_COLOR = figspec.HOLO
ALL_COLOR  = "0.55"

FONT_PREFERENCE = ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"]

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
# Helpers
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Plot Figure 2 main panel grid.")
    p.add_argument("--input_table", default=INPUT_TABLE)
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--out_basename", default=OUT_BASENAME)
    p.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated thresholds to plot, e.g. ST_100. Defaults to THRESHOLDS_TO_PLOT.",
    )
    p.add_argument(
        "--columns",
        default=None,
        help="Comma-separated data columns to render, 1-based: 1=classification prevalence vs "
             "evidence ratio, 2=pocket counts vs ratio, 3=pLDDT hexbin vs ratio. Defaults to "
             "all three. Panel letters re-sequence from 'a' over the selection.",
    )
    p.add_argument("--fig_size", default=None, metavar="W,H",
                   help="canvas in inches, overriding the size derived from the column "
                        "selection.")
    p.add_argument("--exact_canvas", action="store_true",
                   help="save the FULL canvas rather than a tight crop, so the file is "
                        "exactly --fig_size. Use for parts that will be tiled into a "
                        "composed figure: a tight crop trims by however much outboard "
                        "label each panel happens to carry, which is what forces a "
                        "composer to rescale rows to make them line up.")
    p.add_argument("--poster_output_dir", action="store_true", help="Write to the configured poster plots directory.")
    p.add_argument("--hide_row_labels", action="store_true",
                   help="drop the threshold/N label down the left edge. Use when this strip is "
                        "one block of a composed figure whose other blocks already name it.")
    p.add_argument("--letter_start", default=None, metavar="L",
                   help="letter for this render's first panel (e.g. 'c'), so a strip that is "
                        "composed after another block continues the sequence. Implies letters "
                        "are drawn even for a single column.")
    p.add_argument("--hide_panel_letters", action="store_true")
    p.add_argument("--letters_at_ylabel", action="store_true",
                   help="put each panel letter up and left of its plot, aligned with its y label, "
                        "at a fixed height above the axes (manuscript v4 Figure 2)")
    p.add_argument("--transparent_background", action="store_true")
    return p.parse_args()


def configure_from_args(args):
    global INPUT_TABLE, OUTPUT_DIR, OUT_BASENAME, THRESHOLDS_TO_PLOT, COLUMNS, FIG_SIZE_OVERRIDE, LETTER_START, EXACT_CANVAS
    global SHOW_ROW_LABELS
    global SHOW_PANEL_LETTERS, TRANSPARENT_BACKGROUND, LETTERS_AT_YLABEL
    INPUT_TABLE = args.input_table
    OUTPUT_DIR = POSTER_OUTPUT_DIR if args.poster_output_dir else args.output_dir
    OUT_BASENAME = args.out_basename
    if args.thresholds:
        THRESHOLDS_TO_PLOT = [t.strip() for t in args.thresholds.split(",") if t.strip()]
    if args.columns:
        sel = []
        for tok in args.columns.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok not in ("1", "2", "3"):
                raise SystemExit(f"--columns takes 1-3, got {tok!r}")
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
    LETTERS_AT_YLABEL = args.letters_at_ylabel
    EXACT_CANVAS = args.exact_canvas
    SHOW_ROW_LABELS = not args.hide_row_labels
    if args.letter_start:
        L = args.letter_start.strip().lower()
        if L not in PANEL_LETTERS:
            raise SystemExit(f"--letter_start must be one of {PANEL_LETTERS!r}, got {L!r}")
        LETTER_START = PANEL_LETTERS.index(L)
    TRANSPARENT_BACKGROUND = args.transparent_background


def output_paths():
    return (
        os.path.join(OUTPUT_DIR, OUT_BASENAME + ".png"),
        os.path.join(OUTPUT_DIR, OUT_BASENAME + ".pdf"),
        os.path.join(OUTPUT_DIR, OUT_BASENAME + ".svg"),
    )


def save_outputs(fig, *, pad_inches=0.02):
    output_png, output_pdf, output_svg = output_paths()
    if TRANSPARENT_BACKGROUND:
        fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        for ax in fig.axes:
            ax.set_facecolor("white")
    else:
        fig.patch.set_facecolor("white")
    save_kwargs = {
        "bbox_inches": (None if EXACT_CANVAS else "tight"),
        "pad_inches": pad_inches,
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
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    x = float(x)
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3:
        k = x/1e3
        return f"{k:.0f}k" if abs(k-round(k)) < 1e-6 else f"{k:.1f}k"
    return f"{x:.0f}"

def format_row_label(thr):
    if thr == "UNCLUSTERED":
        return "Unclustered" # "PDB pockets"
    if thr.startswith("ST_") and thr[3:].isdigit():
        return f"UBS - ST {thr[3:]}%"
    return f"UBS - {thr}"

def load_unified(path):
    # Accept .csv or .zip
    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            # prefer canonical name if present, else first csv
            name = "analysis_unified_levels.csv"
            if name not in z.namelist():
                csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    raise SystemExit("Zip contains no CSV.")
                name = csvs[0]
            with z.open(name) as f:
                df = pd.read_csv(f, low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    return df

def require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        print("Available columns:", list(df.columns))
        raise SystemExit("Stopping due to missing columns.")

def subset_threshold(df, thr):
    lvl = "entry" if thr == "UNCLUSTERED" else "ubs"
    sub = df[(df["threshold"].astype(str) == str(thr)) & (df["level"].astype(str).str.lower() == lvl)].copy()

    # normalize
    sub["pocket_class"] = sub["pocket_class"].astype(str).str.strip().str.lower()

    # keep apo/holo only
    sub = sub[sub["pocket_class"].isin(["apo", "holo"])].copy()

    # numerics
    sub["evidence_n_apo"]  = pd.to_numeric(sub["evidence_n_apo"], errors="coerce")
    sub["evidence_n_holo"] = pd.to_numeric(sub["evidence_n_holo"], errors="coerce")
    sub["model_pocket_plddt"] = pd.to_numeric(sub["model_pocket_plddt"], errors="coerce")

    # filters
    sub = sub[np.isfinite(sub["evidence_n_apo"]) & np.isfinite(sub["evidence_n_holo"])].copy()
    sub = sub[sub["evidence_n_apo"] >= 1].copy()

    sub["evidence_n_holo_adj"] = sub["evidence_n_holo"] + float(HOLO_PLUS_ONE)
    sub = sub[sub["evidence_n_holo_adj"] > 0].copy()

    # ratio + log10
    ratio = sub["evidence_n_holo_adj"] / sub["evidence_n_apo"]
    sub["log10_ratio"] = np.log10(ratio.where(ratio > 0))
    sub = sub[np.isfinite(sub["log10_ratio"])].copy()

    # x-range restriction (requested)
    sub = sub[(sub["log10_ratio"] >= X_MIN) & (sub["log10_ratio"] <= X_MAX)].copy()

    return sub

def binned_counts(x, edges):
    # returns counts aligned to bins
    c, _ = np.histogram(x, bins=edges)
    return c.astype(int)

def binned_stat(y, x, edges, stat="median"):
    # returns center-wise stat and IQR (optional)
    out = np.full(len(edges) - 1, np.nan, float)
    q25 = np.full(len(edges) - 1, np.nan, float)
    q75 = np.full(len(edges) - 1, np.nan, float)

    bin_id = np.digitize(x, edges) - 1
    for b in range(len(out)):
        m = bin_id == b
        v = y[m]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        if stat == "mean":
            out[b] = float(np.mean(v))
        else:
            out[b] = float(np.median(v))
        q25[b], q75[b] = np.percentile(v, [25, 75])
    return out, q25, q75


# ----------------------------
# Plot functions
# ----------------------------
def plot_col1(ax, sub, show_legend=False):
    # Prevalence (%) of AF apo/holo per ratio bin
    x = sub["log10_ratio"].to_numpy(float)
    cls = sub["pocket_class"].to_numpy(str)

    # bin assignment
    bin_id = np.digitize(x, X_EDGES) - 1
    nbins = len(X_EDGES) - 1

    nA = np.zeros(nbins, dtype=float)
    nH = np.zeros(nbins, dtype=float)
    nT = np.zeros(nbins, dtype=float)

    for b in range(nbins):
        m = bin_id == b
        if not np.any(m):
            continue
        c = cls[m]
        nA[b] = np.sum(c == "apo")
        nH[b] = np.sum(c == "holo")
        nT[b] = nA[b] + nH[b]

    with np.errstate(divide="ignore", invalid="ignore"):
        pA = 100.0 * (nA / nT)
        pH = 100.0 * (nH / nT)

    # Plot
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", which="major", color="0.90", lw=0.7)
    ax.grid(True, axis="x", which="major", color="0.95", lw=0.6)

    ax.plot(X_CENTERS, pA, color=APO_COLOR, lw=2.0, label="Apo %")
    ax.plot(X_CENTERS, pH, color=HOLO_COLOR, lw=2.0, label="Holo %")

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, 100)
    #ax.set_ylabel("Prevalence (%)", fontsize=LAB_FS)

    # in-panel N
    #n = len(sub)
    #ax.text(0.02, 0.98, f"N={n:,}", transform=ax.transAxes, ha="left", va="top", fontsize=SMALL_FS, color="0.25")

    if show_legend:
        handles = [
            Line2D([0],[0], color=APO_COLOR,  lw=2.0, label="AF-Apo"),
            Line2D([0],[0], color=HOLO_COLOR, lw=2.0, label="AF-Holo"),
        ]
        leg = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.02),
        #leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, 0.98),
        #leg = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98),
                        frameon=True, fontsize=6.0, handlelength=2.2, borderaxespad=0.0)
        leg.get_frame().set_alpha(0.65)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("0.85")
        leg.get_frame().set_linewidth(0.6)

def plot_col1_original(ax, sub, show_legend=False):
    # median (or mean) evidence_n_apo and evidence_n_holo_adj vs log10_ratio bins
    x = sub["log10_ratio"].to_numpy(float)
    yA = sub["evidence_n_apo"].to_numpy(float)
    yH = sub["evidence_n_holo_adj"].to_numpy(float)

    stat = EVIDENCE_TREND_STAT.lower().strip()
    sA, q25A, q75A = binned_stat(yA, x, X_EDGES, stat=stat)
    sH, q25H, q75H = binned_stat(yH, x, X_EDGES, stat=stat)

    ax.set_axisbelow(True)
    ax.grid(True, axis="y", which="major", color="0.90", lw=0.7)
    ax.grid(True, axis="x", which="major", color="0.95", lw=0.6)

    if SHOW_IQR_RIBBON:
        ax.fill_between(X_CENTERS, q25A, q75A, alpha=0.15, linewidth=0)
        ax.fill_between(X_CENTERS, q25H, q75H, alpha=0.15, linewidth=0)

    ax.plot(X_CENTERS, sA, color=APO_COLOR,  lw=2.0, label="median(apo)" if stat=="median" else "mean(apo)")
    ax.plot(X_CENTERS, sH, color=HOLO_COLOR, lw=2.0, label="median(holo+1)" if stat=="median" else "mean(holo+1)")

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylabel("Evidence", fontsize=LAB_FS)
    # Optional: tidy ticks for linear scale
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))

    # Optional: percentile-based ylim to avoid a few extreme bins dominating
    vals = np.concatenate([sA[np.isfinite(sA)], sH[np.isfinite(sH)]])
    if vals.size:
        ymax = np.nanpercentile(vals, 98)  # tweak 95–99 as desired
        ax.set_ylim(0, ymax * 1.05)

    if show_legend:
        handles = [
            Line2D([0],[0], color=APO_COLOR,  lw=2.0, label="Exp Apo"),
            Line2D([0],[0], color=HOLO_COLOR, lw=2.0, label=f"Exp Holo (+{HOLO_PLUS_ONE:g})"),
        ]
        leg = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), # top right
                        frameon=True, fontsize=6.0, handlelength=2.2, borderaxespad=0.0)
        leg.get_frame().set_alpha(0.65)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("0.85")
        leg.get_frame().set_linewidth(0.6)

def plot_col2(ax, sub, show_legend=False):
    # step hist counts per AF class + optional "all"
    xA = sub.loc[sub["pocket_class"] == "apo", "log10_ratio"].to_numpy(float)
    xH = sub.loc[sub["pocket_class"] == "holo", "log10_ratio"].to_numpy(float)
    xAll = sub["log10_ratio"].to_numpy(float)

    cA = binned_counts(xA, X_EDGES)
    cH = binned_counts(xH, X_EDGES)
    cAll = binned_counts(xAll, X_EDGES)

    ax.set_axisbelow(True)
    ax.grid(True, axis="y", which="major", color="0.90", lw=0.7)
    ax.grid(True, axis="x", which="major", color="0.95", lw=0.6)

    #ax.step(X_CENTERS, cAll, where="mid", color=ALL_COLOR, lw=1.6, label="All")
    ax.step(X_CENTERS, cA,   where="mid", color=APO_COLOR, lw=2.0, label="Apo")
    ax.step(X_CENTERS, cH,   where="mid", color=HOLO_COLOR, lw=2.0, label="Holo")

    if COL2_LOG_Y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.set_ylim(1, max(2, int(max(cAll.max(), cA.max(), cH.max())) * 2))
    else:
        ax.set_yscale("linear")
        ax.yaxis.set_major_formatter(FuncFormatter(kfmt))

    ax.set_xlim(X_MIN, X_MAX)
    #ax.set_ylabel("Counts (log)" if COL2_LOG_Y else "Counts", fontsize=LAB_FS)

    # in-panel N
    #n = len(sub)
    #ax.text(0.02, 0.98, f"N={n:,}", transform=ax.transAxes, ha="left", va="top", fontsize=SMALL_FS, color="0.25")

    if show_legend:
        handles = [
            #Line2D([0],[0], color=ALL_COLOR,  lw=1.6, label="All"),
            Line2D([0],[0], color=APO_COLOR,  lw=2.0, label="AF-Apo"),
            Line2D([0],[0], color=HOLO_COLOR, lw=2.0, label="AF-Holo"),
        ]
        leg = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.02),
        #leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.98),
        #leg = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98),
                        frameon=True, fontsize=6.0, handlelength=2.2, borderaxespad=0.0)#, ncol=2
        leg.get_frame().set_alpha(0.65)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("0.85")
        leg.get_frame().set_linewidth(0.6)


def plot_col3(ax, sub, norm=None, y_max=None, y_min=None):
    # hexbin pLDDT vs ratio
    d = sub[np.isfinite(sub["model_pocket_plddt"])].copy()
    if d.empty:
        ax.text(0.5, 0.5, "No pLDDT data", transform=ax.transAxes,
                ha="center", va="center", fontsize=SMALL_FS, color="0.35")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(0, 102)
        ax.set_ylabel("pLDDT", fontsize=LAB_FS)
        return None

    # If norm not provided, estimate per-row vmax from a coarse 2D histogram
    if norm is None:
        H, _, _ = np.histogram2d(
            d["log10_ratio"].to_numpy(float),
            d["model_pocket_plddt"].to_numpy(float),
            bins=[60, 50],
            range=[[X_MIN, X_MAX], [0, 100]]
        )
        vmax = max(1, int(np.nanmax(H)))
        norm = LogNorm(vmin=1, vmax=vmax)

    #Y_HEX_MAX = 100  # keep hexbin grid in canonical 0–100 pLDDT space
    #Y_VIEW_MAX = 102 # add a little headroom for the axis border
    # --- Use data-driven y-extent (matches legacy hexbin script) ---
    if y_min is None:
        y_min = float(np.nanmin(d["model_pocket_plddt"].to_numpy(float)))
    if y_max is None:
        y_max = float(np.nanmax(d["model_pocket_plddt"].to_numpy(float)))

    # visual headroom (does NOT affect binning)
    Y_VIEW_MAX = 100 # y_max + 2.0
    Y_VIEW_MIN = 20 # y_min - 4.0

    hb = ax.hexbin(
        d["log10_ratio"].to_numpy(float),
        d["model_pocket_plddt"].to_numpy(float),
        gridsize=GRIDSIZE,
        mincnt=MINCNT,
        linewidths=0,
        edgecolors="none",
        extent=(X_MIN, X_MAX, y_min, y_max),   # <-- KEY FIX
        #extent=(X_MIN, X_MAX, 0, Y_HEX_MAX),
        #extent=(X_MIN, X_MAX, 0, 102),
        norm=norm
    )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_VIEW_MIN, Y_VIEW_MAX)        # <-- headroom only here

    #ax.set_ylabel("Pocket pLDDT", fontsize=LAB_FS)
    ax.grid(True, alpha=0.12)
    return hb


# ----------------------------
# Main
# ----------------------------
def apply_ratio_tags(ax):
    """Directional evidence axis: numerals at -3 / 0 / 3, and coloured More apo / More holo tags
    where the +-1.5 numerals were.

    This replaces the grey cue ROW the published figure drew under the axis -- two arrows, two
    grey words and a balance triangle per column, all positioned in figure fractions that stopped
    meaning anything once panels were drawn at their true millimetre size. The same information
    is carried by the axis itself, in the class colours, which is what the LBS figure does.
    The arrow is stacked under its word so each tag is only as wide as the word.
    """
    ax.set_xticks([X_MIN, 0.5 * X_MIN, 0.0, 0.5 * X_MAX, X_MAX])
    # mathtext arrows render regardless of the sans font (Helvetica Neue lacks U+2192)
    ax.set_xticklabels([f"{X_MIN:g}", "More apo\n" + r"$\leftarrow$", "0",
                        "More holo\n" + r"$\rightarrow$", f"{X_MAX:g}"])
    specs = [("0.15", "normal", TICK_FS), (APO_COLOR, "bold", 6.5), ("0.15", "normal", TICK_FS),
             (HOLO_COLOR, "bold", 6.5), ("0.15", "normal", TICK_FS)]
    for t, (c, w, fs) in zip(ax.get_xticklabels(), specs):
        t.set_color(c)
        t.set_fontweight(w)
        t.set_fontsize(fs)
        t.set_linespacing(0.95)


def main():
    configure_from_args(parse_args())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_unified(INPUT_TABLE)
    require_cols(df, ["threshold","level","pocket_class","evidence_n_apo","evidence_n_holo","model_pocket_plddt"])

    # build per-threshold subsets
    subs = []
    kept_thresholds = []
    for thr in THRESHOLDS_TO_PLOT:
        if str(thr) not in set(df["threshold"].astype(str).unique()):
            print(f"[warn] threshold not found in table: {thr}")
            continue
        sub = subset_threshold(df, thr)
        if sub.empty:
            print(f"[warn] no rows after filters for: {thr}")
            continue
        subs.append(sub)
        kept_thresholds.append(thr)

    if not subs:
        raise SystemExit("No thresholds produced data after filtering.")

    nrows = len(subs)
    sel = list(COLUMNS)
    ncols = len(sel)
    col_ax = {c: j for j, c in enumerate(sel)}      # data column -> grid index
    width_ratios = [COL_WIDTH_RATIOS[c - 1] for c in sel]

    # global y-range across the plotted subsets (after filters)
    y_max_global = max(float(np.nanmax(s["model_pocket_plddt"].to_numpy(float))) for s in subs)
    y_min_global = min(float(np.nanmin(s["model_pocket_plddt"].to_numpy(float))) for s in subs)

    # double-column width for the full grid; a subset keeps the same per-column width
    fig_w = 7.0 * sum(width_ratios) / sum(COL_WIDTH_RATIOS)
    if FIG_SIZE_OVERRIDE:
        fig_w = FIG_SIZE_OVERRIDE[0]
    # the margins and label offsets below were tuned on the 7.0 in canvas; hold them at constant
    # INCHES so a narrower column subset does not push the y labels onto the tick numbers
    m = 7.0 / fig_w
    if nrows == 1:
        fig_h = 2.05
        subplot_adjust = dict(**figspec.margins(fig_w, fig_h, left=(12.0 if SHOW_ROW_LABELS else 9.0),
                                                right=(13.0 if 3 in col_ax else 1.5),
                                                top=4.5, bottom=11.0),
                              wspace=0.30, hspace=0.12)
        shared_label_fs = TICK_FS
    else:
        EXTRA_BOTTOM_IN = 0.50   # ~ one line + breathing room, tune 0.25–0.5
        fig_h = 1.35 * nrows + 0.55 + EXTRA_BOTTOM_IN
        subplot_adjust = dict(**figspec.margins(fig_w, fig_h, left=(12.0 if SHOW_ROW_LABELS else 9.0),
                                                right=(13.0 if 3 in col_ax else 1.5),
                                                top=4.5, bottom=11.0),
                              wspace=0.30, hspace=0.12)
        shared_label_fs = LAB_FS
    if FIG_SIZE_OVERRIDE:
        fig_h = FIG_SIZE_OVERRIDE[1]
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
        gridspec_kw={"width_ratios": width_ratios} # Col 3 is deliberately narrower
    )
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    # Column titles
    #axes[0,0].text(0.02, 1.05, "A", transform=axes[0,0].transAxes, ha="left", va="bottom", fontsize=TITLE_FS+1, fontweight="bold")
    #axes[0,1].text(0.02, 1.05, "B", transform=axes[0,1].transAxes, ha="left", va="bottom", fontsize=TITLE_FS+1, fontweight="bold")
    #axes[0,2].text(0.02, 1.05, "C", transform=axes[0,2].transAxes, ha="left", va="bottom", fontsize=TITLE_FS+1, fontweight="bold")

    #axes[0,0].set_title("Classification prevalence vs ratio", fontsize=TITLE_FS)
    #axes[0,1].set_title("AF class counts vs evidence ratio", fontsize=TITLE_FS)
    #axes[0,2].set_title("pLDDT vs ratio (hexbin)", fontsize=TITLE_FS)

    # Make a shared LogNorm for all hexbins (based on global max count)
    # Do a quick pre-pass to estimate max bin counts
    maxcount = 1
    for sub in subs:
        d = sub[np.isfinite(sub["model_pocket_plddt"])][["log10_ratio","model_pocket_plddt"]]
        if d.empty:
            continue
        # approximate peak count via coarse hexbin on a temp axes-free histogram
        # (fast enough; we just need a stable norm)
        H, _, _ = np.histogram2d(
            d["log10_ratio"].to_numpy(float),
            d["model_pocket_plddt"].to_numpy(float),
            bins=[60, 50],
            range=[[X_MIN, X_MAX], [0, 100]]
        )
        maxcount = max(maxcount, int(np.nanmax(H)))

    shared_norm = LogNorm(vmin=1, vmax=maxcount)

    hbs = []
    for i, (thr, sub) in enumerate(zip(kept_thresholds, subs)):
        # row label on the left of col1
        row_label = format_row_label(thr)
        n = len(sub)

        if SHOW_ROW_LABELS:
            axes[i,0].text(ROW_LABEL_X, 0.52, row_label,
                           transform=axes[i,0].transAxes,
                           rotation=90, ha="center", va="center",
                           fontsize=LAB_FS, color="0.10", fontweight="bold")

            axes[i,0].text(ROW_LABEL_X + 0.07, 0.52, f"N={n:,}",
                           transform=axes[i,0].transAxes,
                           rotation=90, ha="center", va="center",
                           fontsize=SMALL_FS, color="0.25")

        if 1 in col_ax:
            plot_col1(axes[i, col_ax[1]], sub, show_legend=(i==0))
        if 2 in col_ax:
            plot_col2(axes[i, col_ax[2]], sub, show_legend=(i==0))   # Counts now in col 2
        if 3 in col_ax:
            hb = plot_col3(axes[i, col_ax[3]], sub, norm=shared_norm,
                           y_max=y_max_global, y_min=y_min_global)
            if hb is not None:
                hbs.append(hb)

        #if hb is not None:
            #divider = make_axes_locatable(axes[i,2])
            #cax = divider.append_axes("right", size="3.5%", pad=0.08)
            #cbar = fig.colorbar(hb, cax=cax)
            #cbar.ax.tick_params(labelsize=TICK_FS)
            #cbar.set_label("count", fontsize=LAB_FS)

        for j in range(ncols):
            ax = axes[i,j]
            ax.tick_params(axis="both", labelsize=TICK_FS)

            ax.set_xticks([-3, -1.5, 0, 1.5, 3])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
            #ax.set_xticks(np.arange(X_MIN, X_MAX + 1, 1))
            # optional minor ticks:
            # ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            # ax.tick_params(axis="x", which="minor", length=2.5)

            if i != nrows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("")  # no per-column xlabel
                if SHOW_DIRECTION_CUE:
                    apply_ratio_tags(ax)
                #ax.set_xlabel("Evidence ratio - log₁₀(holo / apo)", fontsize=LAB_FS)
                #ax.set_xlabel("log₁₀(holo / apo) evidence ratio", fontsize=LAB_FS)
                #ax.set_xlabel("Holo-to-apo evidence ratio (log₁₀)", fontsize=LAB_FS)

    # --- Dedicated spanning colorbar axis (prevents overlap & ensures full height) ---
    # Make room on the right for the colorbar
    fig.subplots_adjust(**subplot_adjust)

    if SHOW_PANEL_LETTERS and (ncols > 1 or LETTER_START) and not LETTERS_AT_YLABEL:
        # Add panel letters, re-sequenced over the selected columns
        pad_y = 0.01  # vertical offset above the top axes
        y_top = axes[0,0].get_position().y1 + pad_y
        for k, c in enumerate(sel):
            fig.text(axes[0, col_ax[c]].get_position().x0, y_top, PANEL_LETTERS[LETTER_START + k],
                     ha="left", va="bottom", fontsize=TITLE_FS+2, fontweight="bold")

    # Add a single Y axis label per column
    mid_y = 0.5*(axes[0,0].get_position().y1 + axes[-1,0].get_position().y0)
    YLABELS = {1: "Classification prevalence (%)",                       # orig: "Prevalence (%)"
               2: "Pocket count (log)" if COL2_LOG_Y else "Pocket count",  # orig: "Counts (log)"
               3: "Pocket mean pLDDT"}                                   # orig: "Pocket pLDDT"
    ylab_txt = {}
    for c in sel:
        # a fixed gap in mm: the old 0.048 figure-fraction offset shrank with the canvas and put
        # the label on top of the tick numbers once panels were drawn at their true size
        ylab_txt[c] = fig.text(axes[0, col_ax[c]].get_position().x0 - YLAB_GAP_MM / (fig_w / figspec.MM),
                               mid_y, YLABELS[c], rotation=90, va="center", ha="center",
                               fontsize=shared_label_fs)

    if SHOW_PANEL_LETTERS and (ncols > 1 or LETTER_START) and LETTERS_AT_YLABEL:
        # left edge of each column's y label, read back from the drawn extent
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        for k, c in enumerate(sel):
            x0 = inv.transform((ylab_txt[c].get_window_extent().x0, 0))[0]
            fig.text(x0, axes[0, col_ax[c]].get_position().y1 + LETTER_DY_MM / (fig_h / figspec.MM),
                     PANEL_LETTERS[LETTER_START + k], ha="left", va="bottom",
                     fontsize=TITLE_FS+2, fontweight="bold")

    # Add a shared X label centered under all columns
    left = axes[-1,0].get_position().x0
    right = axes[-1,-1].get_position().x1
    mid_x = 0.5 * (left + right)
    bottom = axes[-1,0].get_position().y0
    # millimetres, not a figure fraction: 0.162 is 7.5 mm on the 46 mm canvas this was tuned
    # on and 9.2 mm on a 57 mm one, which walks the label off the bottom of an exact canvas.
    # The tags above it are two 6.5 pt lines, so ~7.5 mm of drop clears them either way.
    LABEL_Y = bottom - 7.5 / (fig_h / figspec.MM)
    #fig.text(mid_x, LABEL_Y, "Evidence ratio per pocket - log₁₀(holo / apo)", ha="center", va="top", fontsize=LAB_FS)
    fig.text(mid_x, LABEL_Y, "Evidence ratio (log₁₀)", ha="center", va="top", fontsize=LAB_FS)

    # The directional cue used to be drawn HERE, as a row of its own below the axis: a balance
    # triangle per column plus one pair of grey arrows and words under the central column, all
    # placed in figure fractions. It now lives on the axis itself (apply_ratio_tags), in the
    # class colours and under every column rather than one, which retires the triangle and the
    # grey tags together. Kept behind `if False` rather than deleted so the geometry is on record.
    if False:
        # Offsets are FIGURE fractions, so they do not transfer between canvas heights: the
        # single-row figure is 2.05 in tall against 6.45 in for the 4-row one, which shrinks the
        # same fraction to a third of the physical gap and drops the cue row on top of the
        # -1.5 / 1.5 tick labels. The 1-row canvas also leaves LESS absolute room under the axes
        # (0.41 in vs 0.58 in), so these are re-tuned rather than simply rescaled: the cue must
        # clear the ticks and still sit above the axis label at LABEL_Y (0.12).
        if nrows == 1:
            TRI_DY, TRI_H, CUE_TEXT_DY, CUE_ARROW_DY = 0.115, 0.041, 0.082, 0.115
        else:
            TRI_DY, TRI_H, CUE_TEXT_DY, CUE_ARROW_DY = 0.050, 0.013, 0.034, 0.050
        TRI_HALF_W   = 0.0050  # triangle half base-width (figure fraction; width is unchanged)
        ARROW_INSET  = 0.010   # arrow tip inset from the central-column edges
        ARROW_GAP    = 0.022   # half-gap kept open around the column centre (for the triangle)
        TRI_COLOR    = "0.15"
        CUE_COLOR    = "0.30"
        ARROW_COLOR  = "0.45"

        # balance triangles (drawn as polygons, font-independent): ratio = 0 sits at
        # each column's horizontal centre, placed on the arrow row clear of the tick labels
        for j in range(ncols):
            pos = axes[-1, j].get_position()
            xc = 0.5 * (pos.x0 + pos.x1)
            yc = bottom - TRI_DY
            pts = [(xc, yc + 0.5 * TRI_H),
                   (xc - TRI_HALF_W, yc - 0.5 * TRI_H),
                   (xc + TRI_HALF_W, yc - 0.5 * TRI_H)]
            fig.add_artist(Polygon(pts, closed=True, transform=fig.transFigure,
                                   facecolor=TRI_COLOR, edgecolor="none", zorder=20))

        # directional arrows + labels under the central column
        # (ncols-1)//2 keeps the published triptych's CENTRE column and, for an even subset,
        # picks the left one -- spanning the cue across the row instead reads as a single
        # shared axis and collides with both panels' tick labels
        midj = (ncols - 1) // 2
        mpos = axes[-1, midj].get_position()
        cx = 0.5 * (mpos.x0 + mpos.x1)
        l_end = mpos.x0 + ARROW_INSET
        r_end = mpos.x1 - ARROW_INSET
        ay = bottom - CUE_ARROW_DY
        ty = bottom - CUE_TEXT_DY
        arrow_kw = dict(xycoords=fig.transFigure, textcoords=fig.transFigure,
                        arrowprops=dict(arrowstyle="->", color=ARROW_COLOR, lw=0.9),
                        annotation_clip=False)
        axes[-1, midj].annotate("", xy=(l_end, ay), xytext=(cx - ARROW_GAP, ay), **arrow_kw)
        axes[-1, midj].annotate("", xy=(r_end, ay), xytext=(cx + ARROW_GAP, ay), **arrow_kw)
        fig.text(0.5 * (l_end + cx - ARROW_GAP), ty, "More apo",
                 ha="center", va="top", fontsize=SMALL_FS, color=CUE_COLOR)
        fig.text(0.5 * (cx + ARROW_GAP + r_end), ty, "More holo",
                 ha="center", va="top", fontsize=SMALL_FS, color=CUE_COLOR)

    # The colorbar belongs to the pLDDT hexbin column; skip it when that column is not drawn
    if 3 in col_ax and hbs:
        # Add colorbar axis that spans (almost) the full hexbin-column stack
        top = axes[0, col_ax[3]].get_position().y1
        bot = axes[-1, col_ax[3]].get_position().y0
        # [left, bottom, width, height] in figure fraction coordinates
        #cax = fig.add_axes([0.945, 0.10, 0.015, 0.80])    
        #cax = fig.add_axes([0.945, 0.16, 0.015, 0.68]) # shorter bar
        #cax = fig.add_axes([0.945, 0.10, 0.015, 0.80]) # spans almost all rows
        cax = fig.add_axes([1.0 - 0.055 * m, bot, 0.015 * m, (top - bot)]) # entire height

        cbar = fig.colorbar(hbs[0], cax=cax)
        cbar_label = "Pocket count" if nrows == 1 else "Pocket count (shared log scale)"
        # right-hand label reads top->bottom (clockwise), the agreed convention for right axes
        # and colorbars; rotation=270 flips the baseline so it needs more pad than the default 90
        cbar.set_label(cbar_label, fontsize=shared_label_fs, rotation=270,
                       labelpad=(8 if nrows == 1 else 10))
        cbar.ax.tick_params(labelsize=TICK_FS)

    output_png, output_pdf, output_svg = save_outputs(fig, pad_inches=(0.05 if nrows == 1 else 0.02))
    plt.close(fig)

    print("Saved:")
    if not TRANSPARENT_BACKGROUND:      # poster mode writes no pdf
        print(" -", output_pdf)
    print(" -", output_svg)
    print(" -", output_png)

if __name__ == "__main__":
    main()
