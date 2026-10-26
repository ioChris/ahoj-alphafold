# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:25:59 2026

@author: Christos P. Feidakis
"""
# -*- coding: utf-8 -*-
"""
Evidence ratio vs separation (REV2) — Unified-table-only analysis

Adds Plot C:
  Hexbin of pLDDT vs separation, with binned-median trend, small multiples by threshold.

All other behavior same as REV1.
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr, pearsonr


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
OUTPUT_DIR   = str(cfg.PLOTS_DIR / "ratio_vs_separation")
OUT_BASENAME = "Ratio_vs_Separation_pLDDT_REV2"

THRESHOLDS_TO_PLOT = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050", "ST_030", "ST_LJ"]

HOLO_PLUS_ONE = 1.0

# Which overlap metrics to plot as separation (one figure per metric)
OVERLAP_COLS_TO_PLOT = ["pocket_overlap_pair", "pocket_overlap_mode", "pocket_overlap_overall"]

# (Optional) human-friendly suffixes for output names
OVERLAP_SUFFIX = {
    "pocket_overlap_pair": "sep_from_overlap_pair",
    "pocket_overlap_mode": "sep_from_overlap_mode",
    "pocket_overlap_overall": "sep_from_overlap_overall",
}


# X range + ticks for ratio panel
X_MIN, X_MAX = -3.0, 3.0
X_TICKS = [-3, -1.5, 0, 1.5, 3]

# Ratio-binning (Plot A + trend overlays)
BIN_W = 0.10
X_EDGES = np.arange(X_MIN, X_MAX + BIN_W, BIN_W)
X_CENTERS = (X_EDGES[:-1] + X_EDGES[1:]) / 2.0
MIN_PER_BIN_FOR_STAT = 25

# pLDDT axis + binning (Plot C)
PLDDT_MIN, PLDDT_MAX = 20.0, 100.0
PLDDT_TICKS = [20, 40, 60, 80, 100]   # or [20, 60, 100] if you want it simpler
#PLDDT_MIN, PLDDT_MAX = 40.0, 100.0
#PLDDT_TICKS = [40, 60, 80, 100]   # or [20, 60, 100] if you want it simpler
PLDDT_BIN_W = 2.0

PLDDT_EDGES = np.arange(PLDDT_MIN, PLDDT_MAX + PLDDT_BIN_W, PLDDT_BIN_W)
PLDDT_CENTERS = (PLDDT_EDGES[:-1] + PLDDT_EDGES[1:]) / 2.0
MIN_PER_BIN_FOR_STAT_PLDDT = 50

# Plot B/C hexbin options
GRIDSIZE = 45
MINCNT = 1

# Layout
FIG_W_IN = 7.6  # slightly wider for 3 columns
ROW_LABEL_X = -0.40

# Typography / style (aligned with Fig2 spirit)
TITLE_FS = 9.2
TICK_FS  = 7.0
LAB_FS   = 8.2
SMALL_FS = 6.2

TREND_COLOR = "0.10"
HIST_COLOR  = "0.65"

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
OUT_FILTER_CSV = os.path.join(OUTPUT_DIR, OUT_BASENAME + "_filtering_summary.csv")
OUT_CORR_CSV   = os.path.join(OUTPUT_DIR, OUT_BASENAME + "_correlations.csv")


# ----------------------------
# Helpers
# ----------------------------
def load_unified(path):
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

def find_plddt_col(df):
    # Prefer what you've used before in this project, fall back to common variants
    candidates = [
        "model_pocket_plddt",
        "model_pocket_plddt_avg",
        "model_pocket_pLDDT_avg",
        "model_pocket_pLDDT",
        "pocket_plddt",
        "pocket_pLDDT",
        "plddt",
        "pLDDT",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # also try case-insensitive match
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None

def format_row_label(thr):
    if thr == "UNCLUSTERED":
        return "PDB pockets"
    if thr.startswith("ST_") and thr[3:].isdigit():
        return f"UBS - ST {thr[3:]}%"
    return f"UBS - {thr}"

def subset_threshold(df, thr, holo_plus_one, x_min, x_max, overlap_col, plddt_col=None):
    lvl = "entry" if thr == "UNCLUSTERED" else "ubs"
    sub = df[(df["threshold"].astype(str) == str(thr)) &
             (df["level"].astype(str).str.lower() == lvl)].copy()

    # normalize class, keep apo/holo only
    sub["pocket_class"] = sub["pocket_class"].astype(str).str.strip().str.lower()
    sub = sub[sub["pocket_class"].isin(["apo", "holo"])].copy()

    # numeric coercion
    for c in ["evidence_n_apo", "evidence_n_holo", overlap_col]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    if plddt_col is not None and plddt_col in sub.columns:
        sub[plddt_col] = pd.to_numeric(sub[plddt_col], errors="coerce")

    # drop NaNs / infs early
    finite_mask = (np.isfinite(sub["evidence_n_apo"]) &
                   np.isfinite(sub["evidence_n_holo"]) &
                   np.isfinite(sub[overlap_col]))
    sub = sub[finite_mask].copy()

    # enforce evidence constraints
    sub = sub[sub["evidence_n_apo"] >= 1].copy()
    sub["evidence_n_holo_adj"] = sub["evidence_n_holo"] + float(holo_plus_one)
    sub = sub[sub["evidence_n_holo_adj"] > 0].copy()

    # ratio + log10
    ratio = sub["evidence_n_holo_adj"] / sub["evidence_n_apo"]
    sub["log10_ratio"] = np.log10(ratio.where(ratio > 0))
    sub = sub[np.isfinite(sub["log10_ratio"])].copy()

    # x-range restriction
    sub = sub[(sub["log10_ratio"] >= x_min) & (sub["log10_ratio"] <= x_max)].copy()

    # overlap -> separation sanity
    sub = sub[(sub[overlap_col] >= 0) & (sub[overlap_col] <= 1)].copy()
    sub["separation"] = 1.0 - sub[overlap_col]

    return sub

def binned_stats(y, x, edges, min_n=1):
    nb = len(edges) - 1
    med = np.full(nb, np.nan, float)
    q25 = np.full(nb, np.nan, float)
    q75 = np.full(nb, np.nan, float)
    nbin = np.zeros(nb, dtype=int)

    bid = np.digitize(x, edges) - 1
    for b in range(nb):
        v = y[bid == b]
        v = v[np.isfinite(v)]
        nbin[b] = int(v.size)
        if v.size < min_n:
            continue
        med[b] = float(np.median(v))
        q25[b], q75[b] = np.percentile(v, [25, 75])
    return med, q25, q75, nbin

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


# ----------------------------
# Plotters
# ----------------------------
def plot_binned_trend(ax, sub, show_xlabel=False):
    x = sub["log10_ratio"].to_numpy(float)
    y = sub["separation"].to_numpy(float)

    med, q25, q75, nbin = binned_stats(y, x, X_EDGES, min_n=MIN_PER_BIN_FOR_STAT)

    # background histogram on secondary axis
    ax2 = ax.twinx()
    ax2.bar(X_CENTERS, nbin, width=(X_EDGES[1]-X_EDGES[0]) * 0.98,
            color=HIST_COLOR, alpha=0.18, linewidth=0, zorder=0)
    ax2.set_ylim(0, max(1, int(np.nanmax(nbin) * 1.15)))
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax.set_axisbelow(True)
    ax.grid(True, axis="y", which="major", color="0.90", lw=0.7)
    ax.grid(True, axis="x", which="major", color="0.95", lw=0.6)

    ax.fill_between(X_CENTERS, q25, q75, color=TREND_COLOR, alpha=0.10, linewidth=0, zorder=2)
    ax.plot(X_CENTERS, med, color=TREND_COLOR, lw=2.0, zorder=3)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(X_TICKS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    if not show_xlabel:
        ax.set_xticklabels([])

    ax.tick_params(axis="both", labelsize=TICK_FS)

def plot_hexbin_ratio(ax, sub, shared_norm=None, show_xlabel=False):
    x = sub["log10_ratio"].to_numpy(float)
    y = sub["separation"].to_numpy(float)

    hb = ax.hexbin(
        x, y,
        gridsize=GRIDSIZE,
        mincnt=MINCNT,
        linewidths=0,
        edgecolors="none",
        extent=(X_MIN, X_MAX, 0.0, 1.0),
        norm=shared_norm
    )

    med, _, _, _ = binned_stats(y, x, X_EDGES, min_n=MIN_PER_BIN_FOR_STAT)
    ax.plot(X_CENTERS, med, color=TREND_COLOR, lw=2.0, zorder=3)

    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.12)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(X_TICKS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    if not show_xlabel:
        ax.set_xticklabels([])

    ax.tick_params(axis="both", labelsize=TICK_FS)
    return hb

def plot_hexbin_plddt(ax, sub, plddt_col, shared_norm=None, show_xlabel=False):
    x = sub[plddt_col].to_numpy(float)
    y = sub["separation"].to_numpy(float)

    # Filter finite + sane pLDDT
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    m2 = (x >= PLDDT_MIN) & (x <= PLDDT_MAX)
    x = x[m2]
    y = y[m2]

    hb = ax.hexbin(
        x, y,
        gridsize=GRIDSIZE,
        mincnt=MINCNT,
        linewidths=0,
        edgecolors="none",
        extent=(PLDDT_MIN, PLDDT_MAX, 0.0, 1.0),
        norm=shared_norm
    )

    # binned median trend over pLDDT
    med, _, _, _ = binned_stats(y, x, PLDDT_EDGES, min_n=MIN_PER_BIN_FOR_STAT_PLDDT)
    ax.plot(PLDDT_CENTERS, med, color=TREND_COLOR, lw=2.0, zorder=3)

    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.12)

    ax.set_xlim(PLDDT_MIN, PLDDT_MAX)
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(PLDDT_TICKS)
    if not show_xlabel:
        ax.set_xticklabels([])

    ax.tick_params(axis="both", labelsize=TICK_FS)
    return hb, int(x.size)


# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_unified(INPUT_TABLE)
    base_required = ["threshold", "level", "pocket_class", "evidence_n_apo", "evidence_n_holo"]
    require_cols(df, base_required)
    
    # Ensure overlap columns exist
    for oc in OVERLAP_COLS_TO_PLOT:
        if oc not in df.columns:
            print(f"[warn] overlap column not found: {oc}")
            print("Available columns:", list(df.columns))
            raise SystemExit("Stopping due to missing overlap column.")


    plddt_col = find_plddt_col(df)
    if plddt_col is None:
        print("[warn] No pLDDT column found. Plot C will be skipped.")
    else:
        print(f"[info] Using pLDDT column: {plddt_col}")

    for overlap_col in OVERLAP_COLS_TO_PLOT:
        suffix = OVERLAP_SUFFIX.get(overlap_col, overlap_col)
        out_basename = f"{OUT_BASENAME}_{suffix}"
    
        OUTPUT_PNG = os.path.join(OUTPUT_DIR, out_basename + ".png")
        OUTPUT_PDF = os.path.join(OUTPUT_DIR, out_basename + ".pdf")
        OUT_FILTER_CSV = os.path.join(OUTPUT_DIR, out_basename + "_filtering_summary.csv")
        OUT_CORR_CSV   = os.path.join(OUTPUT_DIR, out_basename + "_correlations.csv")
    
        # ---- then run the existing pipeline, but:
        # 1) call subset_threshold(..., overlap_col=overlap_col, ...)
        # 2) (optional) update labels to mention overlap_col

    
        # Build per-threshold subsets + filtering summary
        subs = []
        kept_thr = []
        filter_rows = []
    
        available_thr = set(df["threshold"].astype(str).unique())
    
        for thr in THRESHOLDS_TO_PLOT:
            if str(thr) not in available_thr:
                print(f"[warn] threshold not found in table: {thr}")
                continue
    
            raw_lvl = "entry" if thr == "UNCLUSTERED" else "ubs"
            raw = df[(df["threshold"].astype(str) == str(thr)) &
                     (df["level"].astype(str).str.lower() == raw_lvl)].copy()
            n_raw = len(raw)
    
            sub = subset_threshold(df, thr, HOLO_PLUS_ONE, X_MIN, X_MAX, overlap_col=overlap_col, plddt_col=plddt_col)
            n_kept = len(sub)
    
            filter_rows.append({
                "threshold": thr,
                "level_used": raw_lvl,
                "n_raw_level": int(n_raw),
                "n_kept_after_filters": int(n_kept),
            })
    
            if sub.empty:
                print(f"[warn] no rows after filters for: {thr}")
                continue
    
            subs.append(sub)
            kept_thr.append(thr)
    
        if not subs:
            raise SystemExit("No thresholds produced data after filtering.")
    
        filt_df = pd.DataFrame(filter_rows)
        print("\nFiltering summary (per threshold):")
        print(filt_df.to_string(index=False))
        filt_df.to_csv(OUT_FILTER_CSV, index=False)
    
        # Correlations (ratio vs separation) per threshold + pooled
        corr_rows = []
        for thr, sub in zip(kept_thr, subs):
            x = sub["log10_ratio"].to_numpy(float)
            y = sub["separation"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            n = int(x.size)
    
            sr = spearmanr(x, y, nan_policy="omit")
            pr = pearsonr(x, y)
    
            corr_rows.append({
                "threshold": thr,
                "N": n,
                "spearman_rho": float(sr.statistic),
                "spearman_p": float(sr.pvalue),
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
            })
    
        pool = pd.concat([s.assign(threshold=t) for s, t in zip(subs, kept_thr)], ignore_index=True)
        x = pool["log10_ratio"].to_numpy(float)
        y = pool["separation"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        n = int(x.size)
        sr = spearmanr(x, y, nan_policy="omit")
        pr = pearsonr(x, y)
        corr_rows.append({
            "threshold": "POOLED",
            "N": n,
            "spearman_rho": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
        })
    
        corr_df = pd.DataFrame(corr_rows)
        print("\nCorrelation summary (ratio vs separation):")
        print(corr_df.to_string(index=False))
        corr_df.to_csv(OUT_CORR_CSV, index=False)
    
        # Shared LogNorm for hexbins
        # Ratio-space (Plot B)
        maxcount_ratio = 1
        for sub in subs:
            H, _, _ = np.histogram2d(
                sub["log10_ratio"].to_numpy(float),
                sub["separation"].to_numpy(float),
                bins=[60, 50],
                range=[[X_MIN, X_MAX], [0.0, 1.0]]
            )
            maxcount_ratio = max(maxcount_ratio, int(np.nanmax(H)))
        shared_norm_ratio = LogNorm(vmin=1, vmax=maxcount_ratio)
    
        # pLDDT-space (Plot C) - separate norm so the scale is meaningful
        shared_norm_plddt = None
        if plddt_col is not None:
            maxcount_plddt = 1
            for sub in subs:
                xpl = pd.to_numeric(sub[plddt_col], errors="coerce").to_numpy(float)
                y = sub["separation"].to_numpy(float)
                m = np.isfinite(xpl) & np.isfinite(y)
                xpl, y = xpl[m], y[m]
                m2 = (xpl >= PLDDT_MIN) & (xpl <= PLDDT_MAX)
                xpl, y = xpl[m2], y[m2]
                if xpl.size == 0:
                    continue
                H, _, _ = np.histogram2d(
                    xpl, y,
                    bins=[50, 50],
                    range=[[PLDDT_MIN, PLDDT_MAX], [0.0, 1.0]]
                )
                maxcount_plddt = max(maxcount_plddt, int(np.nanmax(H)))
            shared_norm_plddt = LogNorm(vmin=1, vmax=maxcount_plddt)
    
        # Figure layout: rows=thresholds, cols=2 or 3 depending on pLDDT availability
        nrows = len(subs)
        do_plddt = (plddt_col is not None)
        ncols = 3 if do_plddt else 2
    
        # Width ratios: A wider, B/C similar
        width_ratios = [1.05, 0.95, 0.95] if do_plddt else [1.05, 0.95]
    
        fig_h = 1.35 * nrows + 0.75
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(FIG_W_IN if do_plddt else 7.0, fig_h),
            constrained_layout=False,
            gridspec_kw={"width_ratios": width_ratios}
        )
        fig.suptitle(f"Separation defined as 1 - {overlap_col}", fontsize=TITLE_FS, y=0.985)

        if nrows == 1:
            axes = np.array([axes])
    
        hb_ratio_list = []
        hb_plddt_list = []
    
        for i, (thr, sub) in enumerate(zip(kept_thr, subs)):
            row_label = format_row_label(thr)
            n = len(sub)
    
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
    
            show_xlabel = (i == nrows - 1)
    
            # A
            plot_binned_trend(axes[i, 0], sub, show_xlabel=show_xlabel)
            # B
            hb = plot_hexbin_ratio(axes[i, 1], sub, shared_norm=shared_norm_ratio, show_xlabel=show_xlabel)
            hb_ratio_list.append(hb)
    
            # C
            if do_plddt:
                hbC, nC = plot_hexbin_plddt(axes[i, 2], sub, plddt_col, shared_norm=shared_norm_plddt, show_xlabel=show_xlabel)
                hb_plddt_list.append(hbC)
    
        # Global spacing (room for y labels and colorbars)
        fig.subplots_adjust(left=0.10, right=0.92 if not do_plddt else 0.90,
                            top=0.93, bottom=0.10, wspace=0.28, hspace=0.12)
    
        # Panel letters
        pad_y = 0.01
        y_top = axes[0, 0].get_position().y1 + pad_y
        fig.text(axes[0, 0].get_position().x0, y_top, "A", ha="left", va="bottom",
                 fontsize=TITLE_FS + 2, fontweight="bold")
        fig.text(axes[0, 1].get_position().x0, y_top, "B", ha="left", va="bottom",
                 fontsize=TITLE_FS + 2, fontweight="bold")
        if do_plddt:
            fig.text(axes[0, 2].get_position().x0, y_top, "C", ha="left", va="bottom",
                     fontsize=TITLE_FS + 2, fontweight="bold")
    
        # Single Y label per row (shared)
        mid_y = 0.5 * (axes[0, 0].get_position().y1 + axes[-1, 0].get_position().y0)
        col0_left = axes[0, 0].get_position().x0
        fig.text(col0_left - 0.050, mid_y, f"Separation (1 - {overlap_col})", rotation=90,
                 va="center", ha="center", fontsize=LAB_FS)
    
        # Shared X label for ratio (A+B)
        left = axes[-1, 0].get_position().x0
        right = axes[-1, 1].get_position().x1
        mid_x = 0.5 * (left + right)
        bottom = axes[-1, 0].get_position().y0
        fig.text(mid_x, bottom - 0.070, "Evidence ratio per pocket - log₁₀(holo / apo)",
                 ha="center", va="top", fontsize=LAB_FS)
    
        # Shared X label for pLDDT (C)
        if do_plddt:
            leftC = axes[-1, 2].get_position().x0
            rightC = axes[-1, 2].get_position().x1
            midC = 0.5 * (leftC + rightC)
            fig.text(midC, bottom - 0.070, f"Pocket pLDDT ({plddt_col})",
                     ha="center", va="top", fontsize=LAB_FS)
    
        # Colorbar for Plot B (ratio hexbin)
        topB = axes[0, 1].get_position().y1
        botB = axes[-1, 1].get_position().y0
        # place it on the far right if no Plot C, else between B and C to avoid crowding
        if not do_plddt:
            caxB = fig.add_axes([0.94, botB, 0.015, (topB - botB)])
        else:
            # put B's colorbar just to the right of column B
            x1B = axes[0, 1].get_position().x1
            caxB = fig.add_axes([x1B + 0.010, botB, 0.012, (topB - botB)])
        cbarB = fig.colorbar(hb_ratio_list[0], cax=caxB)
        cbarB.set_label("Pocket count (ratio panel, log)", fontsize=LAB_FS)
        cbarB.ax.tick_params(labelsize=TICK_FS)
    
        # Colorbar for Plot C (pLDDT hexbin)
        if do_plddt and hb_plddt_list:
            topC = axes[0, 2].get_position().y1
            botC = axes[-1, 2].get_position().y0
            caxC = fig.add_axes([0.94, botC, 0.015, (topC - botC)])
            cbarC = fig.colorbar(hb_plddt_list[0], cax=caxC)
            cbarC.set_label("Pocket count (pLDDT panel, log)", fontsize=LAB_FS)
            cbarC.ax.tick_params(labelsize=TICK_FS)
    
        # Save
        fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    
        print("\nSaved:")
        print(" -", OUTPUT_PDF)
        print(" -", OUTPUT_PNG)
        print(" -", OUT_FILTER_CSV)
        print(" -", OUT_CORR_CSV)


if __name__ == "__main__":
    main()
