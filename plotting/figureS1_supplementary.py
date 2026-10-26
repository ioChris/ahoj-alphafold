# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 12:33:53 2026

@author: Christos P. Feidakis
"""

# -*- coding: utf-8 -*-
"""
Supplementary Figure (REV3) — pLDDT versus evidence ratio, hexbin grid from the unified table
Rows = thresholds, Cols = (1) pocket density with the binned-median pLDDT line,
                          (2) total-evidence hexbin median, (3) minimum-evidence hexbin median

REV3 (2026-09-15, manuscript v4 SI Figure S2): the former single-threshold density panel
(SI Figure S5, `figure2_main_REV1.py --columns 3 --thresholds ST_100`) merges in here as the
FIRST column, drawn at every threshold, and finally with the binned-median line its caption had
always promised; the apo- and holo-evidence columns (REV2 c-d) drop out -- they supported one
caption sentence no main-text claim used. REV2's output file is left in place.

Baseline style/filters mimic Figure 2 main (REV1).

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
  - restrict x to [-3, 3]

Hexbin content:
  Col1: pocket count per hex (log colour), plus the median pLDDT per 0.25-wide ratio bin
  Col2: median over the hex of evidence_total = evidence_n_apo + evidence_n_holo_adj
  Col3: median over the hex of evidence_min   = min(evidence_n_apo, evidence_n_holo_adj)

Color scaling:
  - Shared across rows within each column (comparability across thresholds)
  - Col1: LogNorm from 1 to the largest hex count in the grid
  - Col2-3: LogNorm with percentile clipping (2–98%) per column
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, FixedLocator, FixedFormatter

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

INPUT_TABLE  = str(cfg.ANALYSIS_UNIFIED_LEVELS_CSV)
OUTPUT_DIR   = str(cfg.SI_PLOTS_DIR)
OUT_BASENAME = "FigureS1_EvidenceHexbin_REV3"

THRESHOLDS_TO_PLOT = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050"]

HOLO_PLUS_ONE = 1.0

# X axis range (requested)
X_MIN, X_MAX = -3.0, 3.0

# Hexbin
GRIDSIZE = 45
MINCNT   = 1

# Binned-median line (column 1): bin width on the log10-ratio axis, minimum pockets per bin
MEDIAN_BIN_W = 0.25
MEDIAN_MIN_N = 30
MEDIAN_COLOR = "#262626"

# Row label x position (left margin)
ROW_LABEL_X = -0.45 # -0.40 # -0.34 # -0.42

# Typography / style (match Figure 2 spirit)
TITLE_FS = 9.2
TICK_FS  = 7.0
LAB_FS   = 8.2
SMALL_FS = 6.2

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

OUTPUT_PNG = os.path.join(OUTPUT_DIR, OUT_BASENAME + ".png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, OUT_BASENAME + ".pdf")
OUTPUT_SVG = os.path.join(OUTPUT_DIR, OUT_BASENAME + ".svg")


# ----------------------------
# Helpers
# ----------------------------
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
    sub = df[(df["threshold"].astype(str) == str(thr)) &
             (df["level"].astype(str).str.lower() == lvl)].copy()

    # normalize class
    sub["pocket_class"] = sub["pocket_class"].astype(str).str.strip().str.lower()
    sub = sub[sub["pocket_class"].isin(["apo", "holo"])].copy()

    # numerics
    sub["evidence_n_apo"]  = pd.to_numeric(sub["evidence_n_apo"], errors="coerce")
    sub["evidence_n_holo"] = pd.to_numeric(sub["evidence_n_holo"], errors="coerce")
    sub["model_pocket_plddt"] = pd.to_numeric(sub["model_pocket_plddt"], errors="coerce")

    # required finite evidence fields
    sub = sub[np.isfinite(sub["evidence_n_apo"]) & np.isfinite(sub["evidence_n_holo"])].copy()

    # require apo evidence >= 1
    sub = sub[sub["evidence_n_apo"] >= 1].copy()

    # holo + 1 adjustment (requested)
    sub["evidence_n_holo_adj"] = sub["evidence_n_holo"] + float(HOLO_PLUS_ONE)
    sub = sub[sub["evidence_n_holo_adj"] > 0].copy()

    # ratio + log10
    ratio = sub["evidence_n_holo_adj"] / sub["evidence_n_apo"]
    sub["log10_ratio"] = np.log10(ratio.where(ratio > 0))
    sub = sub[np.isfinite(sub["log10_ratio"])].copy()

    # x-range restriction
    sub = sub[(sub["log10_ratio"] >= X_MIN) & (sub["log10_ratio"] <= X_MAX)].copy()

    return sub

def _safe_percentile(vals, q):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.nanpercentile(vals, q))

def _nice_log_ticks(vmin, vmax):
    """
    Return human-friendly log ticks using the 1-2-5 series within [vmin, vmax].
    """
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= 0:
        return [1]

    lo = int(np.floor(np.log10(max(vmin, 1e-12))))
    hi = int(np.ceil(np.log10(vmax)))
    candidates = []

    for exp in range(lo, hi + 1):
        base = 10 ** exp
        for m in (1, 2, 5):
            tick = m * base
            if vmin <= tick <= vmax:
                candidates.append(tick)

    if not candidates:
        candidates = [vmin, vmax] if vmin != vmax else [vmin]

    # avoid overcrowding if range is wide
    if len(candidates) > 8:
        step = int(np.ceil(len(candidates) / 8))
        candidates = candidates[::step]
        if candidates[-1] != max(candidates):
            pass

    return candidates

def apply_log_ticks_to_colorbar(cb, norm):
    ticks = _nice_log_ticks(norm.vmin, norm.vmax)
    labels = [f"{t:g}" for t in ticks]
    cb.locator = FixedLocator(ticks)
    cb.formatter = FixedFormatter(labels)
    cb.update_ticks()

def build_column_norms(subs, col_values_dict, clip=(2, 98), min_v=1e-6):
    """
    col_values_dict: dict {colkey: list_of_arrays}, gathered across all thresholds
    Returns dict {colkey: LogNorm}
    """
    norms = {}
    for k, arrays in col_values_dict.items():
        if not arrays:
            norms[k] = LogNorm(vmin=1, vmax=2)
            continue
        vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a is not None and len(a)])
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]  # for LogNorm
        if vals.size == 0:
            norms[k] = LogNorm(vmin=1, vmax=2)
            continue

        vmin = _safe_percentile(vals, clip[0])
        vmax = _safe_percentile(vals, clip[1])

        # enforce sane bounds for LogNorm
        if not np.isfinite(vmin):
            vmin = np.nanmin(vals)
        if not np.isfinite(vmax):
            vmax = np.nanmax(vals)

        vmin = max(min_v, vmin)
        if vmax <= vmin:
            vmax = vmin * 10.0

        norms[k] = LogNorm(vmin=vmin, vmax=vmax)
    return norms

def _finish_axes(ax, title=None):
    ax.set_xlim(X_MIN, X_MAX)
    # headroom for viewing (does not affect binning)
    ax.set_ylim(20, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FS)
    ax.grid(True, alpha=0.12)

def plot_hex_evidence(ax, d, C, norm, y_min, y_max, title=None):
    if d.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=SMALL_FS, color="0.35")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(0, 102)
        return None

    hb = ax.hexbin(
        d["log10_ratio"].to_numpy(float),
        d["model_pocket_plddt"].to_numpy(float),
        C=C,
        reduce_C_function=np.median,  # robust for heavy tails
        gridsize=GRIDSIZE,
        mincnt=MINCNT,
        linewidths=0,
        edgecolors="none",
        extent=(X_MIN, X_MAX, y_min, y_max),   # data-driven extent to avoid top sparse artifact
        norm=norm
    )
    _finish_axes(ax, title)
    return hb

def plot_hex_density(ax, d, y_min, y_max, title=None):
    """Column 1: pocket count per hex (norm attached later, once the grid's maximum is known)
    and the median pLDDT per ratio bin drawn over it."""
    if d.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=SMALL_FS, color="0.35")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(0, 102)
        return None

    x = d["log10_ratio"].to_numpy(float)
    y = d["model_pocket_plddt"].to_numpy(float)
    hb = ax.hexbin(
        x, y,
        gridsize=GRIDSIZE,
        mincnt=MINCNT,
        linewidths=0,
        edgecolors="none",
        extent=(X_MIN, X_MAX, y_min, y_max),
    )

    edges = np.arange(X_MIN, X_MAX + 1e-9, MEDIAN_BIN_W)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(x, edges) - 1
    med = np.full(len(centers), np.nan)
    for b in range(len(centers)):
        v = y[which == b]
        if v.size >= MEDIAN_MIN_N:
            med[b] = float(np.median(v))
    ax.plot(centers, med, color=MEDIAN_COLOR, lw=1.1, zorder=5,
            path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
    _finish_axes(ax, title)
    return hb


# ----------------------------
# Main
# ----------------------------
def main():
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

    # global y-range across plotted subsets (after filters)
    y_max_global = max(float(np.nanmax(s["model_pocket_plddt"].to_numpy(float))) for s in subs)
    y_min_global = min(float(np.nanmin(s["model_pocket_plddt"].to_numpy(float))) for s in subs)

    # Precompute evidence vectors for global scaling of columns 2 and 3
    col_vals = {"total": [], "min": []}

    for s in subs:
        d = s[np.isfinite(s["model_pocket_plddt"])].copy()
        if d.empty:
            continue

        eA = d["evidence_n_apo"].to_numpy(float)
        eH = d["evidence_n_holo_adj"].to_numpy(float)

        col_vals["total"].append(eA + eH)
        col_vals["min"].append(np.minimum(eA, eH))

    norms = build_column_norms(subs, col_vals, clip=(2, 98), min_v=1e-3)

    # Layout
    nrows = len(subs)
    ncols = 3

    fig_w = 6.2
    fig_h = 1.35 * nrows + 0.60

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]}
    )
    if nrows == 1:
        axes = np.array([axes])

    # Column titles
    col_titles = [
        "Pocket density\nhexbin count\nline: median pLDDT",
        "Total evidence\n(apo + holo + 1)\nhexbin median",
        "Minimum evidence\nmin(apo, holo + 1)\nhexbin median",
    ]
    PANEL_FS = 10.5
    PANEL_X = -0.10   # in axes fraction
    PANEL_Y = 1.10
    panel_letters = ["a", "b", "c"]
    for j in range(ncols):
        axes[0, j].set_title(col_titles[j], fontsize=TITLE_FS)
        axes[0, j].text(
            PANEL_X, PANEL_Y, panel_letters[j],
            transform=axes[0, j].transAxes,
            ha="left", va="bottom",
            fontsize=PANEL_FS, fontweight="bold", color="black"
        )

    # Plot
    hbs = [[None]*ncols for _ in range(nrows)]
    for i, (thr, sub) in enumerate(zip(kept_thresholds, subs)):
        row_label = format_row_label(thr)
        n = len(sub)

        # left row label (on col1)
        axes[i, 0].text(
            ROW_LABEL_X, 0.52, row_label,
            transform=axes[i, 0].transAxes,
            rotation=90, ha="center", va="center",
            fontsize=LAB_FS, color="0.10", fontweight="bold"
        )
        axes[i, 0].text(
            ROW_LABEL_X + 0.07, 0.52, f"N={n:,}",
            transform=axes[i, 0].transAxes,
            rotation=90, ha="center", va="center",
            fontsize=SMALL_FS, color="0.25"
        )

        d = sub[np.isfinite(sub["model_pocket_plddt"])].copy()
        if d.empty:
            # still set axes ranges/ticks consistently
            for j in range(ncols):
                ax = axes[i, j]
                ax.set_xlim(X_MIN, X_MAX)
                ax.set_ylim(20, 100)
            continue

        eA = d["evidence_n_apo"].to_numpy(float)
        eH = d["evidence_n_holo_adj"].to_numpy(float)

        C2 = eA + eH
        C3 = np.minimum(eA, eH)

        hbs[i][0] = plot_hex_density(axes[i, 0], d, y_min_global, y_max_global)
        hbs[i][1] = plot_hex_evidence(axes[i, 1], d, C2, norms["total"], y_min_global, y_max_global)
        hbs[i][2] = plot_hex_evidence(axes[i, 2], d, C3, norms["min"],   y_min_global, y_max_global)

        # Tick handling
        for j in range(ncols):
            ax = axes[i, j]
            ax.tick_params(axis="both", labelsize=TICK_FS)

            ax.set_xticks([-3, -1.5, 0, 1.5, 3])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))

            # only first column keeps y tick labels
            if j > 0:
                ax.tick_params(axis="y", labelleft=False, left=False)

            # remove repeated x tick labels except bottom row
            if i != nrows - 1:
                ax.set_xticklabels([])

            ax.set_xlabel("")
            ax.set_ylabel("")

    # Column 1 shares one log colour scale across rows: 1 .. the largest hex count in the grid
    maxcount = max((float(hbs[i][0].get_array().max()) for i in range(nrows) if hbs[i][0] is not None),
                   default=2.0)
    norms["count"] = LogNorm(vmin=1, vmax=max(2.0, maxcount))
    for i in range(nrows):
        if hbs[i][0] is not None:
            hbs[i][0].set_norm(norms["count"])

    # Spacing + colorbars
    fig.subplots_adjust(left=0.105, right=0.805, top=0.92, bottom=0.11, wspace=0.10, hspace=0.12)

    # Determine vertical span of the grid
    top = axes[0, 0].get_position().y1
    bot = axes[-1, 0].get_position().y0
    height = top - bot

    # Representative mappables
    mappable_count = next((hbs[i][0] for i in range(nrows) if hbs[i][0] is not None), None)
    mappable_total = next((hbs[i][1] for i in range(nrows) if hbs[i][1] is not None), None)
    mappable_min   = next((hbs[i][2] for i in range(nrows) if hbs[i][2] is not None), None)

    CBAR_LABEL_FS = 7.0
    CBAR_TICK_FS  = 6.6

    # --- Colorbar geometry controls (easy tweaks) ---
    CBAR_X0   = 0.826   # start x (figure fraction)
    CBAR_W    = 0.018
    CBAR_GAP  = 0.027

    cbar_specs = [
        (mappable_count,  [CBAR_X0 + 0*(CBAR_W + CBAR_GAP), bot, CBAR_W, height], "pocket count", norms["count"], "a"),
        (mappable_total,  [CBAR_X0 + 1*(CBAR_W + CBAR_GAP), bot, CBAR_W, height], "median total evidence", norms["total"], "b"),
        (mappable_min,    [CBAR_X0 + 2*(CBAR_W + CBAR_GAP), bot, CBAR_W, height], "median minimum evidence", norms["min"], "c"),
    ]

    for mappable, rect, label, norm, cbar_tag in cbar_specs:
        if mappable is None:
            continue

        cax = fig.add_axes(rect)
        cb = fig.colorbar(mappable, cax=cax)
        apply_log_ticks_to_colorbar(cb, norm)

        cb.ax.tick_params(labelsize=CBAR_TICK_FS, pad=1.2)

        cb.ax.yaxis.set_ticks_position("right")
        cb.ax.yaxis.set_label_position("right")

        for ticklab in cb.ax.get_yticklabels():
            ticklab.set_rotation(270)
            ticklab.set_va("center")
            ticklab.set_ha("center")

        # Draw label directly on the colorbar, centered vertically
        cb.ax.text(
            0.52, 0.5, label,
            transform=cb.ax.transAxes,
            rotation=270,
            ha="center",
            va="center",
            fontsize=CBAR_LABEL_FS,
            color="black",
            zorder=10
        )

        cb.ax.text(
            0.5, 1.02, cbar_tag,
            transform=cb.ax.transAxes,
            ha="center", va="bottom",
            fontsize=CBAR_LABEL_FS, fontweight="bold", color="black"
        )

    # One shared x-axis label for the full figure
    fig.supxlabel("Evidence ratio  log₁₀(holo / apo)", fontsize=LAB_FS, y=0.03)

    # One shared y-axis label for the full figure
    fig.supylabel("Pocket pLDDT", fontsize=LAB_FS, x=0.030)

    # Save
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.02)
    # optional SVG (can be heavy with hexbins)
    fig.savefig(OUTPUT_SVG, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("Saved:")
    print(" -", OUTPUT_PDF)
    print(" -", OUTPUT_PNG)
    print(" -", OUTPUT_SVG)

if __name__ == "__main__":
    main()
