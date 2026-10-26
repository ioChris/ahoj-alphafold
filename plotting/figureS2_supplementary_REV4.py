# -*- coding: utf-8 -*-
"""
Figure S2 (REV8) — Ratio-only, multi-threshold evidence + class summary plots

Combined figure with 3 panels:
  A) Bubble grid (bin counts as bubble size; evidence as color)
  B) Heatmaps (counts, median evidence_total, fraction holo, fraction apo)
  C) Stacked bars (apo/holo counts per ratio bin) + evidence trend (log)

Key design goals:
  - Consistent x-range/ticks across all panels: X_LABEL_RATIO in [-3, 3]
  - ST_LJ aggregation: if "ST_LJ" is included in THRESHOLDS_TO_PLOT,
    all thresholds starting with "ST_LJ" are merged and shown as "ST_LJ*"
  - Publication-friendly layout for NAR double-column width
  - Spyder-friendly globals (no CLI)

Author: Chris Feidakis (updated with ChatGPT)
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
OUT_PREFIX   = "FigureS2_RatioSummaries_REV4"

THRESHOLDS_TO_PLOT = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050", "ST_030", "ST_LJ"]
HOLO_PLUS_ONE = 1.0

# ----------------------------
# OUTPUT CONTROLS (Spyder)
# ----------------------------
MAKE_COMBINED_FIGURE = True
SAVE_INDIVIDUAL_PANELS = False

SAVE_PNG = True
SAVE_PDF = False
SAVE_SVG = False

DPI_PNG = 600
BBOX_TIGHT = True
PAD_INCHES = 0.05

# ----------------------------
# FIGURE SIZE (double-column)
# ----------------------------
FIG_W_IN = 7.2  # ~ double-column
# Height auto-scales with thresholds; override if you want:
FIG_H_IN = None

# ----------------------------
# AXIS / BINNING
# ----------------------------
X_MIN, X_MAX = -3.0, 3.0
XTICKS = [-3, -1.5, 0, 1.5, 3]

NBINS = 18
BIN_EDGES = np.linspace(X_MIN, X_MAX, NBINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

EVID_AGG = "median"  # "median" (robust) or "mean"

X_LABEL_RATIO = "Evidence ratio - log10(holo/apo)"

# ----------------------------
# COLORBAR WIDTH
# ----------------------------
CB_WIDTH = "11.0%"


plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# ----------------------------
# Helpers
# ----------------------------
def load_unified(path):
    """Load CSV or CSV inside ZIP."""
    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = "analysis_unified_levels.csv"
            if name not in z.namelist():
                csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    raise SystemExit("Zip contains no CSV.")
                name = csvs[0]
            with z.open(name) as f:
                return pd.read_csv(f, low_memory=False)
    return pd.read_csv(path, low_memory=False)

def require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        print("Available columns:", list(df.columns))
        raise SystemExit("Stopping due to missing columns.")

def threshold_short(thr):
    if thr == "UNCLUSTERED":
        #return "Unclustered"
        return "Unclust." # "Raw"
    return str(thr)

def nice_k(n):
    n = float(n)
    if n >= 1_000_000:
        s = f"{n/1_000_000:.1f}M"
        return s.rstrip("0").rstrip(".")
    if n >= 10_000:
        return f"{int(round(n/1000.0))}k"
    if n >= 1000:
        s = f"{n/1000.0:.1f}k"
        return s.rstrip("0").rstrip(".")
    return str(int(n))

def agg_series(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if EVID_AGG == "mean":
        return float(np.mean(x))
    return float(np.median(x))

def subset_threshold(df, thr):
    """Apply Figure 2-style filters and compute ratio + bins."""
    lvl = "entry" if thr == "UNCLUSTERED" else "ubs"
    sub = df[(df["threshold"].astype(str) == str(thr)) &
             (df["level"].astype(str).str.lower() == lvl)].copy()

    sub["pocket_class"] = sub["pocket_class"].astype(str).str.strip().str.lower()
    sub = sub[sub["pocket_class"].isin(["apo", "holo"])].copy()

    sub["evidence_n_apo"]  = pd.to_numeric(sub["evidence_n_apo"],  errors="coerce")
    sub["evidence_n_holo"] = pd.to_numeric(sub["evidence_n_holo"], errors="coerce")
    sub = sub[np.isfinite(sub["evidence_n_apo"]) & np.isfinite(sub["evidence_n_holo"])].copy()

    sub = sub[sub["evidence_n_apo"] >= 1].copy()
    sub["evidence_n_holo_adj"] = sub["evidence_n_holo"] + float(HOLO_PLUS_ONE)
    sub = sub[sub["evidence_n_holo_adj"] > 0].copy()

    ratio = sub["evidence_n_holo_adj"] / sub["evidence_n_apo"]
    sub["log10_ratio"] = np.log10(ratio.where(ratio > 0))
    sub = sub[np.isfinite(sub["log10_ratio"])].copy()

    sub = sub[(sub["log10_ratio"] >= X_MIN) & (sub["log10_ratio"] <= X_MAX)].copy()

    sub["evidence_total"] = sub["evidence_n_apo"] + sub["evidence_n_holo_adj"]
    sub["ratio_bin"] = pd.cut(sub["log10_ratio"], bins=BIN_EDGES, include_lowest=True, labels=False)
    return sub

def build_binned_tables(subs):
    n_thr = len(subs)
    n_bins = len(BIN_CENTERS)

    N = np.zeros((n_thr, n_bins), float)
    evid = np.full((n_thr, n_bins), np.nan, float)
    frac_holo = np.full((n_thr, n_bins), np.nan, float)

    for i, sub in enumerate(subs):
        for b in range(n_bins):
            d = sub[sub["ratio_bin"] == b]
            if d.empty:
                continue
            N[i, b] = len(d)
            evid[i, b] = agg_series(d["evidence_total"].values)
            frac_holo[i, b] = float((d["pocket_class"].values == "holo").mean())

    frac_apo = 1.0 - frac_holo
    return N, evid, frac_holo, frac_apo

def save_fig(fig, outbase):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bbox = "tight" if BBOX_TIGHT else None

    if SAVE_PDF:
        fig.savefig(os.path.join(OUTPUT_DIR, outbase + ".pdf"),
                    bbox_inches=bbox, pad_inches=PAD_INCHES)
    if SAVE_SVG:
        fig.savefig(os.path.join(OUTPUT_DIR, outbase + ".svg"),
                    bbox_inches=bbox, pad_inches=PAD_INCHES)
    if SAVE_PNG:
        fig.savefig(os.path.join(OUTPUT_DIR, outbase + ".png"),
                    dpi=DPI_PNG, bbox_inches=bbox, pad_inches=PAD_INCHES)
    plt.close(fig)

# ----------------------------
# Plotting utilities
# ----------------------------
def k_formatter(x, pos):
    x = float(x)
    if abs(x) >= 1000:
        return f"{x/1000:.0f}k"
    return f"{x:.0f}"


def heatmap_xticks():
    idxs = [int(np.argmin(np.abs(BIN_CENTERS - t))) for t in XTICKS]
    uniq = []
    for i in idxs:
        if i not in uniq:
            uniq.append(i)
    return uniq, [str(t) for t in XTICKS]


def add_cbar_outside(fig, ax, mappable, label, ticks=None, size="3.8%", pad=0.10, fontsize=8):
    """Append a full-height colorbar to the right of ax, with an external (compact) label."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = fig.colorbar(mappable, cax=cax)
    cb.ax.tick_params(labelsize=fontsize)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.set_label(label, rotation=90, labelpad=7, fontsize=fontsize)
    cb.ax.yaxis.set_label_position("right")
    return cb


def add_cbar_inside(fig, ax, mappable, text, ticks=None, size="6.0%", pad=0.12, fontsize=7.5):
    """Append a full-height colorbar to the right of ax, with label rendered inside the bar."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = fig.colorbar(mappable, cax=cax)
    # ADDED: labelrotation=270 flips the numbers sideways
    cb.ax.tick_params(labelsize=fontsize, labelrotation=270, pad=1)
    if ticks is not None:
        cb.set_ticks(ticks)

    if text:
        cb.ax.text(0.5, 0.5, text, transform=cb.ax.transAxes,
                   rotation=270, va="center", ha="center", fontsize=fontsize, color="black")

    # ADDED: This invisible label acts as a force-field, pushing Panel C to the right!
    cb.ax.set_ylabel(" ", labelpad=18) 

    return cb


def plot_panel_A(ax, fig, kept, subs, N, evid):
    n_thr = len(kept)
    y_spacing = 1.15 
    y = np.arange(n_thr) * y_spacing

    X, Y = np.meshgrid(BIN_CENTERS, y)
    x = X.ravel()
    yy = Y.ravel()
    n = N.ravel()
    c = evid.ravel()

    mask = np.isfinite(n) & (n > 0) & np.isfinite(c) & (c > 0)
    x = x[mask]; yy = yy[mask]; n = n[mask]; c = c[mask]

    size_scale = 5.0 # was 7.0
    s = size_scale * np.sqrt(n)

    vmin = np.nanpercentile(c, 2) if c.size else 1
    vmax = np.nanpercentile(c, 98) if c.size else 2
    norm = LogNorm(vmin=max(1e-3, vmin), vmax=max(vmin * 1.2, vmax))

    sc = ax.scatter(x, yy, s=s, c=c, norm=norm, alpha=1.0, linewidths=0, edgecolors="none")

    row_ns = [len(su) for su in subs]
    labels = [f"{threshold_short(t)}\nN={nice_k(n0)}" for t, n0 in zip(kept, row_ns)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    # Inside plot_panel_A:
    ax.set_xlim(X_MIN, X_MAX) # Removed the +/- 0.25 margin
    ax.set_xticks(XTICKS)
    ax.set_xlabel(X_LABEL_RATIO)

    # REPLACED: Swapped y.max and y.min to invert the axis so Unclustered is at the top!
    margin = y_spacing * 0.60
    ax.set_ylim(y.max() + margin, y.min() - margin) 

    ax.grid(True, alpha=0.12)
    ax.tick_params(axis="both", labelsize=9)

    ax.text(-0.14, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=12, ha="left", va="bottom")
    ax.text(-0.02, 1.04, "Bubble grid", transform=ax.transAxes, fontsize=12, ha="left", va="bottom")

    add_cbar_inside(fig, ax, sc, "median evidence_total", size=CB_WIDTH, pad=0.12)#, fontsize=8)

    # REPLACED: Compact Legend
    if n.size:
        q = np.nanpercentile(n, [25, 70, 95])
        #q = np.nanpercentile(n, [40, 70, 92]) # original
        legend_vals = []
        for v in q:
            if v < 50: legend_vals.append(int(round(v / 5) * 5))
            elif v < 500: legend_vals.append(int(round(v / 25) * 25))
            elif v < 5000: legend_vals.append(int(round(v / 250) * 250))
            else: legend_vals.append(int(round(v / 1000) * 1000))
        legend_vals = sorted(set([max(1, int(v)) for v in legend_vals]))
        while len(legend_vals) < 3: legend_vals.append(legend_vals[-1] * 2)
        legend_vals = legend_vals[:3]

        legend_scale = 0.40
        handles = [
            ax.scatter([], [], s=legend_scale * size_scale * np.sqrt(v),
                       facecolors="none", edgecolors="0.35", linewidths=0.9)
            for v in legend_vals
        ]

        # Shifted legend LEFT (x from 0.28 to 0.15) and UP (y from -0.28 to -0.18)
        ax.legend(
            handles, [nice_k(v) for v in legend_vals],
            loc="upper left",
            bbox_to_anchor=(0.15, -0.18), 
            ncol=3,
            frameon=False,
            fontsize=8,
            columnspacing=1.5,
            handletextpad=0.8,
            borderaxespad=0.0
        )
        # Shifted text LEFT (x from 0.00 to -0.13) and UP (y from -0.34 to -0.20)
        # to perfectly match the new legend position
        ax.text(0.00, -0.20, "N pockets\nper bin", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, linespacing=1.2)


def plot_panel_B(ax_list, fig, kept, N, evid, frac_holo, frac_apo):
    
    # Positioned at 1.45: High enough to clear the heatmap title, 
    # but low enough to avoid Panel A's legend
    ax_list[0].text(-0.16, 1.45, "b", transform=ax_list[0].transAxes, fontweight="bold",
                    fontsize=12, ha="left", va="bottom")
    ax_list[0].text(-0.04, 1.45, "Heatmaps across ratio bins", transform=ax_list[0].transAxes,
                    fontsize=12, ha="left", va="bottom")

    thr_labels = [threshold_short(t) for t in kept]
    xt_idx, xt_lab = heatmap_xticks()
    
    cb_pad = 0.12
    extent = [X_MIN, X_MAX, len(kept) - 0.5, -0.5]

    # 1) counts
    N_plot = N.copy()
    N_plot[N_plot < 1] = np.nan
    im = ax_list[0].imshow(
        N_plot, origin="upper", aspect="auto", extent=extent,
        norm=LogNorm(vmin=max(1, np.nanmin(N_plot)), vmax=np.nanmax(N_plot))
    )
    # RESTORED: Nice, tight padding (pad=4) for the heatmap title!
    ax_list[0].set_title("Pocket count per ratio bin", fontsize=10, fontweight="normal", pad=4)
    ax_list[0].set_yticks(range(len(thr_labels)))
    ax_list[0].set_yticklabels(thr_labels)
    add_cbar_inside(fig, ax_list[0], im, "N pockets", size=CB_WIDTH, pad=cb_pad)

    # 2) evidence
    vals = evid[np.isfinite(evid) & (evid > 0)]
    vmin = np.nanpercentile(vals, 2) if vals.size else 1
    vmax = np.nanpercentile(vals, 98) if vals.size else 2
    im = ax_list[1].imshow(
        evid, origin="upper", aspect="auto", extent=extent, # <-- CHANGED TO square_aspect
        norm=LogNorm(vmin=max(1e-3, vmin), vmax=max(vmin * 1.2, vmax))
    )
    ax_list[1].set_title("Median evidence_total per ratio bin", fontsize=10, fontweight="normal", pad=6)
    ax_list[1].set_yticks(range(len(thr_labels)))
    ax_list[1].set_yticklabels(thr_labels)
    add_cbar_inside(fig, ax_list[1], im, "median\nevidence_total", size=CB_WIDTH, pad=cb_pad)

    # 3) frac holo
    im = ax_list[2].imshow(
        frac_holo, origin="upper", aspect="auto", extent=extent, vmin=0, vmax=1 # <-- CHANGED TO square_aspect
    )
    ax_list[2].set_title("Fraction holo per ratio bin", fontsize=10, fontweight="normal", pad=6)
    ax_list[2].set_yticks(range(len(thr_labels)))
    ax_list[2].set_yticklabels(thr_labels)
    add_cbar_inside(fig, ax_list[2], im, "fraction holo", ticks=[0, 0.5, 1.0], size=CB_WIDTH, pad=cb_pad)

    # 4) frac apo
    im = ax_list[3].imshow(
        frac_apo, origin="upper", aspect="auto", extent=extent, vmin=0, vmax=1 # <-- CHANGED TO square_aspect
    )
    ax_list[3].set_title("Fraction apo per ratio bin", fontsize=10, fontweight="normal", pad=6)
    ax_list[3].set_yticks(range(len(thr_labels)))
    ax_list[3].set_yticklabels(thr_labels)
    ax_list[3].set_xlabel(X_LABEL_RATIO)
    add_cbar_inside(fig, ax_list[3], im, "fraction apo", ticks=[0, 0.5, 1.0], size=CB_WIDTH, pad=cb_pad)

    # Exact matching bounds to Panel A
    for ax in ax_list:
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_xticks(XTICKS)
        ax.tick_params(axis="both", labelsize=9)
        
    for ax in ax_list[:-1]:
        ax.set_xticklabels([])


def plot_panel_C(ax_list, fig, kept, subs):
    ax_list[0].text(-0.14, 1.15, "c", transform=ax_list[0].transAxes, fontweight="bold",
                    fontsize=12, ha="left", va="bottom")
    ax_list[0].text(-0.02, 1.15, "Pocket counts + evidence trend", transform=ax_list[0].transAxes,
                    fontsize=12, ha="left", va="bottom")

    n_thr = len(ax_list)
    ax2_list = [] # ADDED: List to store our right-side axes

    for i, (ax, sub, thr) in enumerate(zip(ax_list, subs, kept)):
        apo_counts = np.zeros(len(BIN_CENTERS), float)
        holo_counts = np.zeros(len(BIN_CENTERS), float)
        evid_med = np.full(len(BIN_CENTERS), np.nan, float)

        for b in range(len(BIN_CENTERS)):
            d = sub[sub["ratio_bin"] == b]
            if d.empty:
                continue
            apo_counts[b] = (d["pocket_class"].values == "apo").sum()
            holo_counts[b] = (d["pocket_class"].values == "holo").sum()
            evid_med[b] = agg_series(d["evidence_total"].values)

        width = (BIN_EDGES[1] - BIN_EDGES[0]) * 0.92
        ax.bar(BIN_CENTERS, apo_counts, width=width, label="apo")
        ax.bar(BIN_CENTERS, holo_counts, width=width, bottom=apo_counts, label="holo")

        ax.set_xlim(X_MIN, X_MAX)
        ax.set_xticks(XTICKS)
        ax.grid(True, alpha=0.12)

        ax.yaxis.set_major_formatter(FuncFormatter(k_formatter))
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)

        ax.text(
            0.02, 0.06, threshold_short(thr),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="0.5", alpha=0.75, linewidth=0.6)
        )

        ax2 = ax.twinx()
        ax2_list.append(ax2) # ADDED: Save this axis
        ax2.plot(BIN_CENTERS, evid_med, marker="o", linewidth=1.0, markersize=2.5, color="0.25")
        ax2.set_yscale("log")
        
        # CHANGED: direction="in" pushes the tick marks inside.
        # pad=-24 pulls the text labels inside the plot area.
        #ax2.tick_params(axis="y", direction="in", labelsize=9, labelright=True, right=True, pad=-4)
        ax2.tick_params(axis="y", which="both", direction="in", labelsize=9, labelright=True, right=True, pad=-4)
        
        # ADDED: Right-align the labels so they sit perfectly flush against the inside spine
        for label in ax2.get_yticklabels():
            label.set_ha("right")
            
        ax2.set_ylabel("")

        if i != n_thr - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(X_LABEL_RATIO)

        if i == 0:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(0.15, 0.99), # Shifted to the left (closer to 0.0)
                ncol=1,
                fontsize=9,
                frameon=True,
                framealpha=0.85,
                edgecolor="0.6",
                facecolor="white"
            )

    # REPLACED: Use official set_ylabel so constrained_layout sees them
    #mid_idx = n_thr // 2
    #ax_list[mid_idx].set_ylabel("N pockets", fontsize=10, labelpad=10)
    #ax2_list[mid_idx].set_ylabel("median evidence_total (log)", fontsize=10, labelpad=12)
    
    # ---------------------------------------------------------
    # SPANNING AXIS: For perfectly centered Y-labels
    # ---------------------------------------------------------
    # 1. Grab the gridspec for the entire right column
    gs = ax_list[0].get_subplotspec().get_gridspec()
    
    # 2. Add an invisible axis that spans all the rows in this column
    ax_span = fig.add_subplot(gs[:, 0], frameon=False)
    ax_span.tick_params(axis="both", which="both", bottom=False, top=False, 
                        left=False, right=False, labelbottom=False, labelleft=False)
    
    # 3. Add the left label to the spanning axis (defaults to 90 degrees)
    ax_span.set_ylabel("N pockets", fontsize=10, labelpad=15) 
    
    # 4. (ax2_span definition stays the same) ...
    ax2_span = ax_span.twinx()
    ax2_span.set_frame_on(False)
    ax2_span.tick_params(axis="both", which="both", bottom=False, top=False, 
                         left=False, right=False, labelbottom=False, labelright=False)
    
    # 5. Add the right label with 270-degree rotation!
    # Note: You might need to slightly increase labelpad (e.g., to 15 or 20) 
    # because the baseline flip changes how Matplotlib measures the gap.
    ax2_span.set_ylabel("median evidence_total (log)", fontsize=10, rotation=270, labelpad=18)


# ----------------------------
# Combined figure builder
# ----------------------------

def make_combined_figure(subs, kept, N, evid, frac_holo, frac_apo):
    n_thr = len(kept)

    # Let the figure grow slightly taller to accommodate the new spaces
    if FIG_H_IN is None:
        fig_h = 7.5 + 0.40 * n_thr 
    else:
        fig_h = float(FIG_H_IN)

    fig = plt.figure(figsize=(FIG_W_IN, fig_h), layout="constrained")

    outer = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[1.15, 1.25],
        wspace=0.25 
    )

    # CHANGED: nrows=6. Added a 0.35 height spacer row at index 1!
    gsLeft = outer[0, 0].subgridspec(nrows=6, ncols=1, height_ratios=[1.8, 0.13, 0.4, 0.4, 0.4, 0.4])

    # A (top-left) - Row 0
    axA = fig.add_subplot(gsLeft[0, 0])
    plot_panel_A(axA, fig, kept, subs, N, evid)

    # B (bottom-left): 4 heatmaps - Start at Row 2! (Row 1 is skipped)
    axB = [fig.add_subplot(gsLeft[i, 0]) for i in range(2, 6)]
    plot_panel_B(axB, fig, kept, N, evid, frac_holo, frac_apo)

    # C (right): stacked bars
    gsC = outer[0, 1].subgridspec(nrows=n_thr, ncols=1)
    axC = [fig.add_subplot(gsC[i, 0]) for i in range(n_thr)]
    plot_panel_C(axC, fig, kept, subs)

    return fig

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_unified(INPUT_TABLE)
    require_cols(df, ["threshold", "level", "pocket_class", "evidence_n_apo", "evidence_n_holo"])

    # ST_LJ aggregation triggered by THRESHOLDS_TO_PLOT containing "ST_LJ"
    ST_LJ_LABEL = "ST_LJ*"
    thresholds_requested = list(THRESHOLDS_TO_PLOT)

    if "ST_LJ" in thresholds_requested:
        thresholds_requested = [ST_LJ_LABEL if t == "ST_LJ" else t for t in thresholds_requested]
        df = df.copy()
        mask = df["threshold"].astype(str).str.startswith("ST_LJ")
        df.loc[mask, "threshold"] = ST_LJ_LABEL

    subs, kept = [], []
    for thr in thresholds_requested:
        if str(thr) not in set(df["threshold"].astype(str).unique()):
            print(f"[warn] threshold not in table: {thr}")
            continue
        sub = subset_threshold(df, thr)
        if sub.empty:
            print(f"[warn] no rows after filters for: {thr}")
            continue
        subs.append(sub)
        kept.append(thr)

    if not subs:
        raise SystemExit("No data after filtering for selected thresholds.")

    N, evid, frac_holo, frac_apo = build_binned_tables(subs)

    if MAKE_COMBINED_FIGURE:
        fig = make_combined_figure(subs, kept, N, evid, frac_holo, frac_apo)
        save_fig(fig, OUT_PREFIX + "_combined")

    if SAVE_INDIVIDUAL_PANELS:
        # Optional: add separate panel exports if you need them later.
        pass

    print("Done. Saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
