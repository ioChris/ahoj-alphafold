
# -*- coding: utf-8 -*-
"""
Figure S3 (REV12) — q10-based comparison of low-tail pLDDT
against evidence-ratio extremeness and total evidence, with
both ratio-derived and native-evidence panel variants.

Goal
----
Create a compact 4-panel multi-panel figure for manuscript iteration:
  A) pLDDT q10 vs absolute evidence ratio (ratio-defined bins)
  B) pLDDT q10 vs total evidence (ratio-defined bins)
  C) pLDDT q10 vs total evidence (native evidence bins)
  D) Raw predictor coupling: |ratio| vs total evidence (pooled raw filtered rows), shown as smoothed density contours

The script still computes q5 summaries and saves them to tables for manuscript support,
but q5 is not shown in the main figure.

Filtering / definitions
-----------------------
- UNCLUSTERED -> level == "entry"
- clustered thresholds -> level == "ubs"
- keep pocket_class in {"apo", "holo"}
- require evidence_n_apo >= 1
- apply holo+1 correction:
      evidence_n_holo_adj = evidence_n_holo + HOLO_PLUS_ONE
- ratio = evidence_n_holo_adj / evidence_n_apo
- signed ratio = log10(ratio), restricted to [-3, 3]
- absolute ratio = |signed ratio|
- ratio-defined bins are fixed bins on the signed-ratio axis
- native evidence bins are fixed bins on pooled log10(total evidence)
- q10/q5 are computed from model_pocket_plddt within each threshold/bin

Outputs
-------
- 3+2 combined figure (PNG + optional PDF/SVG)
- Ratio-bin summary CSV
- Native-evidence-bin summary CSV
- Ratio-derived pairwise Spearman table
- Native-evidence pairwise Spearman table
- Native q10-focused influence summary CSV
- Raw predictor-correlation CSV
- Panel source-data CSV

Spyder-friendly globals are defined near the top. No CLI required.
"""

import os
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr, pearsonr, rankdata
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# -----------------------------------------------------------------------------
# USER SETTINGS (Spyder)
# -----------------------------------------------------------------------------
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
OUTPUT_DIR   = str(cfg.SI_PLOTS_DIR / "S3")
OUT_PREFIX = "FigureS3_q10_ratio_evidence_REV12"

THRESHOLDS_TO_PLOT = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050", "ST_030", "ST_LJ"]
CONTOUR_THRESHOLDS = ["UNCLUSTERED", "ST_100", "ST_050"]

HOLO_PLUS_ONE = 1.0
NBINS = 18
NATIVE_EVIDENCE_NBINS = NBINS
#X_MIN, X_MAX = -5.0, 5.0
X_MIN, X_MAX = -3.0, 3.0

SAVE_PNG = True
SAVE_PDF = True
SAVE_SVG = True
DPI_PNG = 600

SAVE_RATIO_BIN_SUMMARY_CSV = True
SAVE_NATIVE_BIN_SUMMARY_CSV = True
SAVE_RATIO_PAIRWISE_CSV = True
SAVE_NATIVE_PAIRWISE_CSV = True
SAVE_Q10_PAIRWISE_COMPARISON_CSV = True
SAVE_NATIVE_INFLUENCE_CSV = True
SAVE_RAW_PREDICTOR_CSV = True
SAVE_PANEL_SOURCE_CSV = True

# Figure layout
FIG_W_IN = 7.10
FIG_H_IN = 5.55

# Scatter styling
SHOW_THRESHOLD_LINES = True
POINT_ALPHA = 0.86
POINT_STYLE = "bubble"   # "bubble" or "mark"
FIXED_MARKER_SIZE = 22.0

SHOW_BIN_SIZE_LEGEND = True
BIN_SIZE_LEGEND_VALUES = [50, 500, 5000, 20000]

# Typography / style
FONT_PREFERENCE = ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"]

TITLE_FS = 8.7
TICK_FS = 6.9
LAB_FS = 8.3
SMALL_FS = 5.9
PANEL_FS = 10.7
ANN_FS = 5.8

GRID_COLOR = "0.90"
RATIO_COLOR = "#4c78a8"
EVIDENCE_COLOR = "#555555"
HEXBIN_CMAP = "Blues"

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

BIN_EDGES = np.linspace(X_MIN, X_MAX, NBINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def load_unified(path):
    """Load CSV directly or CSV from inside a ZIP archive."""
    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            name = "analysis_unified_levels.csv"
            if name not in zf.namelist():
                csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    raise SystemExit("Zip contains no CSV file.")
                name = csvs[0]
            with zf.open(name) as fh:
                return pd.read_csv(fh, low_memory=False)
    return pd.read_csv(path, low_memory=False)

def require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        print("Available columns:", list(df.columns))
        raise SystemExit("Stopping due to missing columns.")

def format_row_label(thr):
    if thr == "UNCLUSTERED":
        return "Unclustered"
    if thr.startswith("ST_") and thr[3:].isdigit():
        return f"UBS - ST {thr[3:]}%"
    return f"UBS - {thr}"

def threshold_colors(thresholds):
    base = {
        "UNCLUSTERED": "#4c78a8",
        "ST_100": "#f58518",
        "ST_070": "#54a24b",
        "ST_050": "#e45756",
        "ST_030": "#72b7b2",
        "ST_LJ": "#b279a2",
        "ST_LJ*": "#b279a2",
    }
    return {thr: base.get(thr, "0.35") for thr in thresholds}

def nice_k(n):
    n = float(n)
    if n >= 1_000_000:
        s = f"{n/1_000_000:.1f}M"
        return s.rstrip("0").rstrip(".")
    if n >= 10_000:
        return f"{int(round(n / 1000.0))}k"
    if n >= 1000:
        s = f"{n / 1000.0:.1f}k"
        return s.rstrip("0").rstrip(".")
    return str(int(round(n)))

def kfmt(x, pos=None):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    x = float(x)
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        k = x / 1e3
        return f"{k:.0f}k" if abs(k - round(k)) < 1e-6 else f"{k:.1f}k"
    return f"{x:.0f}"

def pfmt(p):
    if not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"

def ranks(arr):
    arr = np.asarray(arr, float)
    out = np.full(arr.shape, np.nan, float)
    mask = np.isfinite(arr)
    out[mask] = rankdata(arr[mask])
    return out

def residualize(y, X):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta

def threshold_design(g, thresholds):
    cats = pd.Categorical(g["threshold"], categories=list(thresholds), ordered=True)
    return pd.get_dummies(cats, drop_first=True).to_numpy(float)

def partial_spearman(g, x_col, y_col, covar_cols=None, thresholds=None, add_threshold=True):
    """Partial Spearman via rank residualization."""
    use = g[np.isfinite(g[x_col]) & np.isfinite(g[y_col])].copy()
    covar_cols = covar_cols or []
    for c in covar_cols:
        use = use[np.isfinite(use[c])].copy()
    if len(use) < 4:
        return np.nan, np.nan, len(use)

    x = ranks(use[x_col].to_numpy(float))
    y = ranks(use[y_col].to_numpy(float))

    covars = []
    if add_threshold and thresholds is not None:
        D = threshold_design(use, thresholds)
        if D.size:
            covars.append(D)
    for c in covar_cols:
        covars.append(ranks(use[c].to_numpy(float))[:, None])

    if covars:
        X = np.column_stack(covars)
        xr = residualize(x, X)
        yr = residualize(y, X)
    else:
        xr, yr = x, y

    rho, pval = pearsonr(xr, yr)
    return float(rho), float(pval), int(len(use))

def rank_regression_r2(g, y_col, predictor_cols, thresholds=None, add_threshold=True):
    """Rank-based linear regression R² with optional threshold fixed effects."""
    use = g[np.isfinite(g[y_col])].copy()
    for c in predictor_cols:
        use = use[np.isfinite(use[c])].copy()
    if len(use) < 4:
        return np.nan, len(use)

    y = ranks(use[y_col].to_numpy(float))

    Xs = []
    if add_threshold and thresholds is not None:
        D = threshold_design(use, thresholds)
        if D.size:
            Xs.append(D)
    for c in predictor_cols:
        Xs.append(ranks(use[c].to_numpy(float))[:, None])

    if Xs:
        X = np.column_stack(Xs)
        X = np.column_stack([np.ones(len(use)), X])
    else:
        X = np.ones((len(use), 1))

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return float(r2), int(len(use))

def marker_sizes(n_values):
    """Map bin counts to scatter marker area using sqrt scaling."""
    n = np.asarray(n_values, float)
    return 14.0 + 0.62 * np.sqrt(np.maximum(n, 0.0))

def scatter_size_array(n_values):
    if str(POINT_STYLE).lower() == "mark":
        return np.full(len(np.asarray(n_values)), FIXED_MARKER_SIZE, dtype=float)
    return marker_sizes(n_values)

def st_lj_aggregate_if_requested(df, thresholds_requested):
    """
    If ST_LJ is requested and multiple ST_LJ-like thresholds exist,
    merge them into one label.
    """
    thresholds_requested = list(thresholds_requested)
    if "ST_LJ" not in thresholds_requested:
        return df, thresholds_requested

    thresh_vals = df["threshold"].astype(str)
    st_lj_like = sorted([t for t in thresh_vals.unique() if t.startswith("ST_LJ")])
    if len(st_lj_like) <= 1:
        return df, thresholds_requested

    df2 = df.copy()
    mask = df2["threshold"].astype(str).str.startswith("ST_LJ")
    df2.loc[mask, "threshold"] = "ST_LJ*"
    thresholds_requested = ["ST_LJ*" if t == "ST_LJ" else t for t in thresholds_requested]
    return df2, thresholds_requested

def subset_threshold(df, thr):
    """Apply Figure 2 / S1 / S2 filters and compute derived evidence variables."""
    lvl = "entry" if thr == "UNCLUSTERED" else "ubs"
    sub = df[(df["threshold"].astype(str) == str(thr)) &
             (df["level"].astype(str).str.lower() == lvl)].copy()

    sub["pocket_class"] = sub["pocket_class"].astype(str).str.strip().str.lower()
    sub = sub[sub["pocket_class"].isin(["apo", "holo"])].copy()

    for col in ["evidence_n_apo", "evidence_n_holo", "model_pocket_plddt"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    sub = sub[np.isfinite(sub["evidence_n_apo"]) & np.isfinite(sub["evidence_n_holo"])].copy()
    sub = sub[sub["evidence_n_apo"] >= 1].copy()

    sub["evidence_n_holo_adj"] = sub["evidence_n_holo"] + float(HOLO_PLUS_ONE)
    sub = sub[sub["evidence_n_holo_adj"] > 0].copy()

    ratio = sub["evidence_n_holo_adj"] / sub["evidence_n_apo"]
    sub["log10_ratio"] = np.log10(ratio.where(ratio > 0))
    sub = sub[np.isfinite(sub["log10_ratio"])].copy()
    sub = sub[(sub["log10_ratio"] >= X_MIN) & (sub["log10_ratio"] <= X_MAX)].copy()

    sub["log10_ratio_abs"] = np.abs(sub["log10_ratio"])
    sub["evidence_total"] = sub["evidence_n_apo"] + sub["evidence_n_holo_adj"]
    sub["ratio_bin"] = pd.cut(sub["log10_ratio"], bins=BIN_EDGES, include_lowest=True, labels=False)
    return sub

def summarize_ratio_bins(sub, thr):
    """Summarize the fixed ratio bins for one threshold."""
    rows = []
    for b, xc in enumerate(BIN_CENTERS):
        d = sub[sub["ratio_bin"] == b].copy()
        plddt = d["model_pocket_plddt"].to_numpy(float)
        plddt = plddt[np.isfinite(plddt)]

        rows.append({
            "threshold": thr,
            "ratio_bin": b,
            "x_center": float(xc),
            "n_total": int(len(d)),
            "n_apo": int((d["pocket_class"] == "apo").sum()),
            "n_holo": int((d["pocket_class"] == "holo").sum()),
            "ratio_abs_median": float(np.nanmedian(np.abs(d["log10_ratio"].to_numpy(float)))) if len(d) else np.nan,
            "evidence_total_median": float(np.nanmedian(d["evidence_total"].to_numpy(float))) if len(d) else np.nan,
            "plddt_q10": float(np.nanpercentile(plddt, 10)) if plddt.size else np.nan,
            "plddt_q5": float(np.nanpercentile(plddt, 5)) if plddt.size else np.nan,
            "plddt_median": float(np.nanmedian(plddt)) if plddt.size else np.nan,
        })
    return pd.DataFrame(rows)

def summarize_native_evidence_bins(subsets_by_threshold, thresholds, nbins=18):
    """
    Fully evidence-native binning.

    Bins are defined directly on log10(total evidence), pooled across thresholds,
    then q10/q5 and median |ratio| are computed within those evidence bins for each threshold.
    """
    pooled_vals = []
    for thr in thresholds:
        sub = subsets_by_threshold.get(thr)
        if sub is None or sub.empty:
            continue
        vals = pd.to_numeric(sub["evidence_total"], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            pooled_vals.append(vals)

    if not pooled_vals:
        return pd.DataFrame(columns=[
            "threshold", "evidence_bin", "n_total",
            "ratio_abs_median", "evidence_total_median", "plddt_q10", "plddt_q5", "plddt_median",
            "bin_left_log10", "bin_right_log10", "bin_center_log10"
        ])

    pooled_vals = np.concatenate(pooled_vals)
    pooled_log = np.log10(pooled_vals)

    lo = np.floor(pooled_log.min() * 2.0) / 2.0
    hi = np.ceil(pooled_log.max() * 2.0) / 2.0
    if hi <= lo:
        hi = lo + 1.0

    edges_log = np.linspace(lo, hi, nbins + 1)

    rows = []
    for thr in thresholds:
        sub = subsets_by_threshold.get(thr)
        if sub is None or sub.empty:
            continue

        d = sub.copy()
        d["evidence_total"] = pd.to_numeric(d["evidence_total"], errors="coerce")
        d["model_pocket_plddt"] = pd.to_numeric(d["model_pocket_plddt"], errors="coerce")
        d["log10_ratio_abs"] = pd.to_numeric(d["log10_ratio_abs"], errors="coerce")
        d = d[np.isfinite(d["evidence_total"]) & (d["evidence_total"] > 0)].copy()
        if d.empty:
            continue

        d["log10_evidence_total"] = np.log10(d["evidence_total"])
        d["evidence_bin"] = pd.cut(
            d["log10_evidence_total"],
            bins=edges_log,
            include_lowest=True,
            labels=False
        )

        for b in range(nbins):
            g = d[d["evidence_bin"] == b].copy()
            plddt = g["model_pocket_plddt"].to_numpy(float)
            plddt = plddt[np.isfinite(plddt)]
            evid = g["evidence_total"].to_numpy(float)
            evid = evid[np.isfinite(evid) & (evid > 0)]
            ratio_abs = g["log10_ratio_abs"].to_numpy(float)
            ratio_abs = ratio_abs[np.isfinite(ratio_abs)]

            rows.append({
                "threshold": thr,
                "evidence_bin": b,
                "n_total": int(len(g)),
                "ratio_abs_median": float(np.nanmedian(ratio_abs)) if ratio_abs.size else np.nan,
                "evidence_total_median": float(np.nanmedian(evid)) if evid.size else np.nan,
                "plddt_q10": float(np.nanpercentile(plddt, 10)) if plddt.size else np.nan,
                "plddt_q5": float(np.nanpercentile(plddt, 5)) if plddt.size else np.nan,
                "plddt_median": float(np.nanmedian(plddt)) if plddt.size else np.nan,
                "bin_left_log10": float(edges_log[b]),
                "bin_right_log10": float(edges_log[b + 1]),
                "bin_center_log10": float(0.5 * (edges_log[b] + edges_log[b + 1])),
            })

    return pd.DataFrame(rows)

def compute_pairwise_correlations(summary_df):
    rows = []
    combos = [
        ("ratio_abs_median", "plddt_q10", "Absolute ratio", "pLDDT q10"),
        ("evidence_total_median", "plddt_q10", "Total evidence", "pLDDT q10"),
        ("ratio_abs_median", "plddt_q5", "Absolute ratio", "pLDDT q5"),
        ("evidence_total_median", "plddt_q5", "Total evidence", "pLDDT q5"),
        ("ratio_abs_median", "evidence_total_median", "Absolute ratio", "Total evidence"),
    ]

    for x_col, y_col, x_label, y_label in combos:
        for thr, g in list(summary_df.groupby("threshold", sort=False)) + [("POOLED", summary_df.copy())]:
            use = g[(g["n_total"] > 0) & np.isfinite(g[x_col]) & np.isfinite(g[y_col])].copy()
            if len(use) >= 3:
                rho, pval = spearmanr(use[x_col], use[y_col])
            else:
                rho, pval = np.nan, np.nan
            rows.append({
                "threshold": thr,
                "x_metric": x_col,
                "y_metric": y_col,
                "x_label": x_label,
                "y_label": y_label,
                "n_bins_used": int(len(use)),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(pval) if np.isfinite(pval) else np.nan,
            })
    return pd.DataFrame(rows)

def compute_influence_q10(summary_df, thresholds):
    """q10-focused relative contribution summaries on a summary table."""
    rows = []
    predictor_corr_rows = []

    for thr, g in list(summary_df.groupby("threshold", sort=False)) + [("POOLED", summary_df.copy())]:
        use = g[(g["n_total"] > 0) &
                np.isfinite(g["ratio_abs_median"]) &
                np.isfinite(g["evidence_total_median"])].copy()
        if len(use) >= 3:
            rho, pval = spearmanr(use["ratio_abs_median"], use["evidence_total_median"])
        else:
            rho, pval = np.nan, np.nan
        predictor_corr_rows.append({
            "threshold": thr,
            "n_bins_used": int(len(use)),
            "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            "p_value": float(pval) if np.isfinite(pval) else np.nan,
        })

    pooled = summary_df[(summary_df["n_total"] > 0)].copy()
    y_col = "plddt_q10"
    pair_ratio = pooled[np.isfinite(pooled["ratio_abs_median"]) & np.isfinite(pooled[y_col])]
    pair_total = pooled[np.isfinite(pooled["evidence_total_median"]) & np.isfinite(pooled[y_col])]

    rho_ratio, p_ratio = spearmanr(pair_ratio["ratio_abs_median"], pair_ratio[y_col]) if len(pair_ratio) >= 3 else (np.nan, np.nan)
    rho_total, p_total = spearmanr(pair_total["evidence_total_median"], pair_total[y_col]) if len(pair_total) >= 3 else (np.nan, np.nan)

    part_ratio, part_ratio_p, n_part_ratio = partial_spearman(
        pooled, "ratio_abs_median", y_col,
        covar_cols=["evidence_total_median"],
        thresholds=thresholds, add_threshold=True
    )
    part_total, part_total_p, n_part_total = partial_spearman(
        pooled, "evidence_total_median", y_col,
        covar_cols=["ratio_abs_median"],
        thresholds=thresholds, add_threshold=True
    )

    r2_base, _ = rank_regression_r2(pooled, y_col, [], thresholds=thresholds, add_threshold=True)
    r2_ratio, _ = rank_regression_r2(pooled, y_col, ["ratio_abs_median"], thresholds=thresholds, add_threshold=True)
    r2_total, _ = rank_regression_r2(pooled, y_col, ["evidence_total_median"], thresholds=thresholds, add_threshold=True)
    r2_both, n_both = rank_regression_r2(pooled, y_col, ["ratio_abs_median", "evidence_total_median"], thresholds=thresholds, add_threshold=True)

    rows.append({
        "y_metric": y_col,
        "y_label": "pLDDT q10",
        "pairwise_rho_ratio_abs": float(rho_ratio) if np.isfinite(rho_ratio) else np.nan,
        "pairwise_p_ratio_abs": float(p_ratio) if np.isfinite(p_ratio) else np.nan,
        "pairwise_rho_total_evidence": float(rho_total) if np.isfinite(rho_total) else np.nan,
        "pairwise_p_total_evidence": float(p_total) if np.isfinite(p_total) else np.nan,
        "partial_rho_ratio_abs_given_total": part_ratio,
        "partial_p_ratio_abs_given_total": part_ratio_p,
        "partial_rho_total_given_ratio_abs": part_total,
        "partial_p_total_given_ratio_abs": part_total_p,
        "rank_r2_threshold_only": r2_base,
        "rank_r2_plus_ratio_abs": r2_ratio,
        "rank_r2_plus_total_evidence": r2_total,
        "rank_r2_plus_both": r2_both,
        "unique_delta_r2_ratio_abs": (r2_both - r2_total) if np.isfinite(r2_both) and np.isfinite(r2_total) else np.nan,
        "unique_delta_r2_total_evidence": (r2_both - r2_ratio) if np.isfinite(r2_both) and np.isfinite(r2_ratio) else np.nan,
        "n_bins_pairwise_ratio": int(len(pair_ratio)),
        "n_bins_pairwise_total": int(len(pair_total)),
        "n_bins_partial_ratio": int(n_part_ratio),
        "n_bins_partial_total": int(n_part_total),
        "n_bins_rank_models": int(n_both),
    })

    return pd.DataFrame(rows), pd.DataFrame(predictor_corr_rows)

def compute_raw_influence(raw_df, thresholds):
    """Raw filtered-data relative contribution summary using raw pLDDT."""
    pooled = raw_df[
        np.isfinite(raw_df["log10_ratio_abs"]) &
        np.isfinite(raw_df["evidence_total"]) &
        np.isfinite(raw_df["model_pocket_plddt"])
    ].copy()

    y_col = "model_pocket_plddt"
    pair_ratio = pooled[np.isfinite(pooled["log10_ratio_abs"]) & np.isfinite(pooled[y_col])]
    pair_total = pooled[np.isfinite(pooled["evidence_total"]) & np.isfinite(pooled[y_col])]

    rho_ratio, p_ratio = spearmanr(pair_ratio["log10_ratio_abs"], pair_ratio[y_col]) if len(pair_ratio) >= 3 else (np.nan, np.nan)
    rho_total, p_total = spearmanr(pair_total["evidence_total"], pair_total[y_col]) if len(pair_total) >= 3 else (np.nan, np.nan)

    part_ratio, part_ratio_p, n_part_ratio = partial_spearman(
        pooled, "log10_ratio_abs", y_col,
        covar_cols=["evidence_total"],
        thresholds=thresholds, add_threshold=True
    )
    part_total, part_total_p, n_part_total = partial_spearman(
        pooled, "evidence_total", y_col,
        covar_cols=["log10_ratio_abs"],
        thresholds=thresholds, add_threshold=True
    )

    r2_base, _ = rank_regression_r2(pooled, y_col, [], thresholds=thresholds, add_threshold=True)
    r2_ratio, _ = rank_regression_r2(pooled, y_col, ["log10_ratio_abs"], thresholds=thresholds, add_threshold=True)
    r2_total, _ = rank_regression_r2(pooled, y_col, ["evidence_total"], thresholds=thresholds, add_threshold=True)
    r2_both, n_both = rank_regression_r2(pooled, y_col, ["log10_ratio_abs", "evidence_total"], thresholds=thresholds, add_threshold=True)

    rows = [{
        "y_metric": y_col,
        "y_label": "raw pLDDT",
        "pairwise_rho_ratio_abs": float(rho_ratio) if np.isfinite(rho_ratio) else np.nan,
        "pairwise_p_ratio_abs": float(p_ratio) if np.isfinite(p_ratio) else np.nan,
        "pairwise_rho_total_evidence": float(rho_total) if np.isfinite(rho_total) else np.nan,
        "pairwise_p_total_evidence": float(p_total) if np.isfinite(p_total) else np.nan,
        "partial_rho_ratio_abs_given_total": part_ratio,
        "partial_p_ratio_abs_given_total": part_ratio_p,
        "partial_rho_total_given_ratio_abs": part_total,
        "partial_p_total_given_ratio_abs": part_total_p,
        "rank_r2_threshold_only": r2_base,
        "rank_r2_plus_ratio_abs": r2_ratio,
        "rank_r2_plus_total_evidence": r2_total,
        "rank_r2_plus_both": r2_both,
        "unique_delta_r2_ratio_abs": (r2_both - r2_total) if np.isfinite(r2_both) and np.isfinite(r2_total) else np.nan,
        "unique_delta_r2_total_evidence": (r2_both - r2_ratio) if np.isfinite(r2_both) and np.isfinite(r2_ratio) else np.nan,
        "n_points_pairwise_ratio": int(len(pair_ratio)),
        "n_points_pairwise_total": int(len(pair_total)),
        "n_points_partial_ratio": int(n_part_ratio),
        "n_points_partial_total": int(n_part_total),
        "n_points_rank_models": int(n_both),
    }]

    return pd.DataFrame(rows)

def compute_raw_predictor_correlations(raw_df):
    rows = []
    use_all = raw_df[np.isfinite(raw_df["log10_ratio_abs"]) & np.isfinite(raw_df["evidence_total"])]
    for thr, g in raw_df.groupby("threshold", sort=False):
        use = g[np.isfinite(g["log10_ratio_abs"]) & np.isfinite(g["evidence_total"])]
        if len(use) >= 3:
            rho, pval = spearmanr(use["log10_ratio_abs"], use["evidence_total"])
        else:
            rho, pval = np.nan, np.nan
        rows.append({
            "threshold": thr,
            "n_points_used": int(len(use)),
            "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            "p_value": float(pval) if np.isfinite(pval) else np.nan,
        })
    if len(use_all) >= 3:
        rho, pval = spearmanr(use_all["log10_ratio_abs"], use_all["evidence_total"])
    else:
        rho, pval = np.nan, np.nan
    rows.append({
        "threshold": "POOLED",
        "n_points_used": int(len(use_all)),
        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "p_value": float(pval) if np.isfinite(pval) else np.nan,
    })
    return pd.DataFrame(rows)

def nice_plddt_limits(values):
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 80.0, 100.0
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    lo = max(0.0, np.floor((lo - 1.8) / 5.0) * 5.0)
    hi = min(100.0, np.ceil((hi + 1.0) / 2.5) * 2.5)
    if hi - lo < 10.0:
        lo = max(0.0, lo - 2.5)
    return lo, hi

def add_corr_box(ax, rho, pval, n_used, extra_lines=None,
                 x=0.03, y=0.97, ha="left", va="top"):
    lines = [
        f"Spearman ρ = {rho:.2f}" if np.isfinite(rho) else "Spearman ρ = NA",
        f"p = {pfmt(pval)}",
        f"n = {n_used}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes,
            ha=ha, va=va, fontsize=SMALL_FS,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                      edgecolor="0.80", linewidth=0.6, alpha=0.92))

def add_bin_size_legend_bottom(ax):
    if str(POINT_STYLE).lower() == "mark":
        return
    size_handles = [
        plt.scatter([], [], s=marker_sizes([v])[0],
                    color="0.55", alpha=0.70,
                    edgecolor="white", linewidth=0.35,
                    label=nice_k(v))
        for v in BIN_SIZE_LEGEND_VALUES
    ]
    leg = ax.legend(
        handles=size_handles,
        title="Bin size (pockets/sites per bin)",
        loc="lower center",
        bbox_to_anchor=(0.50, 0.02),
        ncol=len(BIN_SIZE_LEGEND_VALUES),
        frameon=True,
        fontsize=SMALL_FS,
        title_fontsize=SMALL_FS + 0.05,
        handletextpad=0.7,
        columnspacing=0.9,
        borderaxespad=0.2,
    )
    leg.get_frame().set_alpha(0.88)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.80")

def plot_threshold_scatter(ax, source_df, x_col, y_col, pairwise_df,
                           thresholds, colors, x_label, y_label,
                           logx=False, show_ylabel=True, add_size_legend=False,
                           force_ylim=None):
    pooled_row = pairwise_df[(pairwise_df["threshold"] == "POOLED") &
                             (pairwise_df["x_metric"] == x_col) &
                             (pairwise_df["y_metric"] == y_col)]
    rho = pooled_row["spearman_rho"].iloc[0] if not pooled_row.empty else np.nan
    pval = pooled_row["p_value"].iloc[0] if not pooled_row.empty else np.nan
    n_used = pooled_row["n_bins_used"].iloc[0] if not pooled_row.empty else 0

    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID_COLOR, lw=0.7)
    ax.grid(True, which="minor", color="0.95", lw=0.5)

    for thr in thresholds:
        g = source_df[(source_df["threshold"] == thr) &
                      (source_df["n_total"] > 0) &
                      np.isfinite(source_df[x_col]) &
                      np.isfinite(source_df[y_col])].copy()
        if g.empty:
            continue
        g = g.sort_values(x_col)
        if SHOW_THRESHOLD_LINES and len(g) >= 2:
            ax.plot(g[x_col], g[y_col], color=colors[thr], lw=0.7, alpha=0.38, zorder=1)
        ax.scatter(
            g[x_col], g[y_col],
            s=scatter_size_array(g["n_total"].to_numpy(float)),
            color=colors[thr], alpha=POINT_ALPHA,
            edgecolor="white", linewidth=0.35, zorder=3
        )

    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))

    ax.set_xlabel(x_label, fontsize=LAB_FS)
    if show_ylabel:
        ax.set_ylabel(y_label, fontsize=LAB_FS)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)

    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)

    if x_col == "ratio_abs_median":
        ax.set_xlim(-0.02, 3.10)
        ax.set_xticks([0, 1, 2, 3])

    if force_ylim is not None:
        ax.set_ylim(*force_ylim)
    elif y_col.startswith("plddt_q"):
        lo, hi = nice_plddt_limits(source_df[y_col].to_numpy(float))
        ax.set_ylim(lo, hi)

    add_corr_box(ax, rho, pval, n_used)

    if add_size_legend and SHOW_BIN_SIZE_LEGEND:
        add_bin_size_legend_bottom(ax)

def draw_raw_predictor_density(ax, raw_df, raw_corr_df, threshold=None, show_ylabel=True, show_xlabel=True):
    if threshold is None:
        pooled = raw_df[np.isfinite(raw_df["log10_ratio_abs"]) & np.isfinite(raw_df["evidence_total"])].copy()
        prow = raw_corr_df[raw_corr_df["threshold"] == "POOLED"]
        extra_lines = ["density = pockets/sites"] # , "smoothed contours"
    else:
        pooled = raw_df[(raw_df["threshold"] == threshold) &
                        np.isfinite(raw_df["log10_ratio_abs"]) &
                        np.isfinite(raw_df["evidence_total"])].copy()
        prow = raw_corr_df[raw_corr_df["threshold"] == threshold]
        extra_lines = ["density = pockets/sites"] # , "smoothed contours"

    x = pooled["log10_ratio_abs"].to_numpy(float)
    y = pooled["evidence_total"].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[mask]
    y = y[mask]

    if len(x) >= 10:
        ylog = np.log10(y)
        x_edges = np.linspace(0.0, 3.0, 42)
        ylog_edges = np.linspace(np.floor(ylog.min()*2)/2, np.ceil(ylog.max()*2)/2, 34)
        H, xe, ye = np.histogram2d(x, ylog, bins=[x_edges, ylog_edges])
        Hs = gaussian_filter(H.T, sigma=1.0)

        x_cent = 0.5 * (xe[:-1] + xe[1:])
        ylog_cent = 0.5 * (ye[:-1] + ye[1:])
        y_cent = 10 ** ylog_cent
        Xg, Yg = np.meshgrid(x_cent, y_cent)

        positive = Hs[Hs > 0]
        if positive.size:
            levels = np.quantile(positive, [0.18, 0.38, 0.56, 0.72, 0.84, 0.92, 0.97])
            levels = np.unique(levels)
            if len(levels) >= 2:
                # extend="max" fills the densest core (density above the top
                # level, the 0.97 quantile) with the colormap's darkest color
                # instead of leaving it white/unfilled.
                ax.contourf(Xg, Yg, Hs, levels=levels, cmap=HEXBIN_CMAP, alpha=0.92,
                            antialiased=True, extend="max")
                ax.contour(Xg, Yg, Hs, levels=levels, colors="0.42", linewidths=0.38, alpha=0.54)
            else:
                ax.contourf(Xg, Yg, Hs, levels=7, cmap=HEXBIN_CMAP, alpha=0.92,
                            antialiased=True, extend="max")
        else:
            ax.scatter(x, y, s=2, alpha=0.04, color=RATIO_COLOR, edgecolors="none")
    else:
        ax.scatter(x, y, s=4, alpha=0.10, color=RATIO_COLOR, edgecolors="none")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID_COLOR, lw=0.7)
    ax.grid(True, which="minor", color="0.95", lw=0.5)
    ax.set_xlim(-0.02, 3.10)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
    if show_xlabel:
        ax.set_xlabel("Absolute evidence ratio, |log10(holo/apo)|", fontsize=LAB_FS)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Total evidence", fontsize=LAB_FS)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_box_aspect(0.86)
    ax.set_anchor("W")

    if not prow.empty:
        prow = prow.iloc[0]
        add_corr_box(
            ax, prow["spearman_rho"], prow["p_value"], prow["n_points_used"],
            extra_lines=extra_lines, x=0.97, y=0.05, ha="right", va="bottom"
        )
    else:
        add_corr_box(ax, np.nan, np.nan, 0, extra_lines=extra_lines,
                     x=0.97, y=0.05, ha="right", va="bottom")

def build_q10_pairwise_comparison(ratio_pairwise_df, native_pairwise_df):
    """
    Panel E comparison using q10 with each predictor in its own preferred bins:
      - absolute ratio from ratio-defined bins
      - total evidence from native evidence bins
    """
    def _get(df, x_metric):
        row = df[(df["threshold"] == "POOLED") &
                 (df["x_metric"] == x_metric) &
                 (df["y_metric"] == "plddt_q10")]
        if row.empty:
            return {"rho": np.nan, "p_value": np.nan, "n_units": 0}
        row = row.iloc[0]
        return {
            "rho": float(row["spearman_rho"]) if np.isfinite(row["spearman_rho"]) else np.nan,
            "p_value": float(row["p_value"]) if np.isfinite(row["p_value"]) else np.nan,
            "n_units": int(row["n_bins_used"]),
        }

    ratio_stats = _get(ratio_pairwise_df, "ratio_abs_median")
    native_stats = _get(native_pairwise_df, "evidence_total_median")

    return pd.DataFrame([
        {
            "predictor": "absolute_ratio",
            "predictor_label": "Absolute ratio",
            "binning_scheme": "ratio bins",
            "pairwise_rho": ratio_stats["rho"],
            "p_value": ratio_stats["p_value"],
            "n_units": ratio_stats["n_units"],
        },
        {
            "predictor": "total_evidence",
            "predictor_label": "Total evidence",
            "binning_scheme": "native evidence bins",
            "pairwise_rho": native_stats["rho"],
            "p_value": native_stats["p_value"],
            "n_units": native_stats["n_units"],
        },
    ])

def draw_pairwise_q10_legend(ax):
    ax.axis("off")
    handles = [
        Patch(facecolor=RATIO_COLOR, edgecolor=RATIO_COLOR, alpha=0.88,
              label="Absolute ratio (ratio bins)"),
        Patch(facecolor=EVIDENCE_COLOR, edgecolor=EVIDENCE_COLOR, alpha=0.88,
              label="Total evidence (native evidence bins)"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        ncol=1,
        frameon=True,
        fontsize=SMALL_FS,
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=0.9,
        labelspacing=0.55,
    )
    leg.get_frame().set_alpha(0.88)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.80")

def draw_pairwise_q10_panel(ax, comparison_df, xtick_label="pLDDT q10"):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", which="major", color=GRID_COLOR, lw=0.7)
    ax.axhline(0, color="0.55", lw=0.7, zorder=1)

    center = 0.0
    step = 0.22
    width = 0.18
    offs = np.array([-0.5, 0.5]) * step
    vals = comparison_df["pairwise_rho"].to_numpy(float)
    fills = [RATIO_COLOR, EVIDENCE_COLOR]

    vmax = max([abs(v) for v in vals if np.isfinite(v)] + [0.25])
    yhi = min(1.00, max(0.35, vmax + 0.12))

    for j, v in enumerate(vals):
        ax.bar(center + offs[j], v, width=width,
               color=fills[j], alpha=0.88,
               edgecolor=fills[j], linewidth=0.85, zorder=3)
        if np.isfinite(v):
            ax.text(center + offs[j], v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=ANN_FS)

    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(0.0, yhi)
    ax.set_xticks([center])
    ax.set_xticklabels([xtick_label], fontsize=LAB_FS)
    ax.set_ylabel("Pairwise association strength", fontsize=LAB_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)

def draw_contribution_legend(ax):
    ax.axis("off")
    handles = [
        Patch(facecolor=RATIO_COLOR, edgecolor=RATIO_COLOR, alpha=0.88, label="Pairwise ρ: absolute ratio"),
        Patch(facecolor=EVIDENCE_COLOR, edgecolor=EVIDENCE_COLOR, alpha=0.88, label="Pairwise ρ: total evidence"),
        Patch(facecolor="white", edgecolor=RATIO_COLOR, hatch="///", label="Partial ρ: ratio | evidence"),
        Patch(facecolor="white", edgecolor=EVIDENCE_COLOR, hatch="///", label="Partial ρ: evidence | ratio"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        ncol=1,
        frameon=True,
        fontsize=SMALL_FS,
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=0.9,
        labelspacing=0.45,
    )
    leg.get_frame().set_alpha(0.88)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.80")

def draw_info_box(ax, text):
    ax.axis("off")
    ax.text(0.0, 1.0, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=SMALL_FS,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                      edgecolor="0.82", linewidth=0.6, alpha=0.92))

def draw_contribution_panel(ax, influence_df, xtick_label="raw pLDDT"):
    row = influence_df.iloc[0]

    ax.set_axisbelow(True)
    ax.grid(True, axis="y", which="major", color=GRID_COLOR, lw=0.7)
    ax.axhline(0, color="0.55", lw=0.7, zorder=1)

    center = 0.0
    step = 0.19
    width = 0.16
    offs = np.array([-1.5, -0.5, 0.5, 1.5]) * step
    vals = [
        row["pairwise_rho_ratio_abs"],
        row["pairwise_rho_total_evidence"],
        row["partial_rho_ratio_abs_given_total"],
        row["partial_rho_total_given_ratio_abs"],
    ]
    fills = [RATIO_COLOR, EVIDENCE_COLOR, RATIO_COLOR, EVIDENCE_COLOR]
    alphas = [0.88, 0.88, 0.18, 0.18]
    hatches = [None, None, "///", "///"]

    vmax = max([abs(v) for v in vals if np.isfinite(v)] + [0.25])
    yhi = min(1.00, max(0.32, vmax + 0.12))
    ylo = -0.10 if any(np.isfinite(v) and v < 0 for v in vals) else 0.0

    for j, v in enumerate(vals):
        ax.bar(center + offs[j], v, width=width,
               color=fills[j], alpha=alphas[j],
               edgecolor=fills[j], linewidth=0.9, hatch=hatches[j], zorder=3)
        if np.isfinite(v):
            offset = 0.02 if v >= 0 else -0.025
            ax.text(center + offs[j], v + offset,
                    f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=ANN_FS)

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([center])
    ax.set_xticklabels([xtick_label], fontsize=LAB_FS)
    ax.set_ylabel("Association strength", fontsize=LAB_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)

def draw_panel_title(ax, letter, title):
    ax.axis("off")
    ax.text(0.00, 0.34, letter, transform=ax.transAxes,
            fontsize=PANEL_FS, fontweight="bold", ha="left", va="bottom")
    ax.text(0.085, 0.34, title, transform=ax.transAxes,
            fontsize=TITLE_FS, ha="left", va="bottom")


def format_contour_title_label(thr):
    thr = str(thr)
    if thr == "UNCLUSTERED":
        return "Unclustered"
    if thr.startswith("ST_") and thr[3:].isdigit():
        return f"UBS - ST {thr[3:]}"
    return format_row_label(thr).replace('%', '')

def make_figure(ratio_df, ratio_pairwise_df, native_df, native_pairwise_df,
                raw_predictor_corr_df, raw_df, thresholds, contour_thresholds):
    colors = threshold_colors(thresholds)

    q10_vals = np.concatenate([
        ratio_df["plddt_q10"].to_numpy(float),
        native_df["plddt_q10"].to_numpy(float),
    ])
    top_ylim = nice_plddt_limits(q10_vals)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=False)
    outer = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0, 1.0])

    agrid = outer[0, 0].subgridspec(2, 1, height_ratios=[0.18, 0.82], hspace=0.04)
    bgrid = outer[0, 1].subgridspec(2, 1, height_ratios=[0.18, 0.82], hspace=0.04)
    cgrid = outer[0, 2].subgridspec(2, 1, height_ratios=[0.18, 0.82], hspace=0.04)
    dgrid = outer[1, 0].subgridspec(2, 1, height_ratios=[0.18, 0.82], hspace=0.04)
    egrid = outer[1, 1].subgridspec(2, 1, height_ratios=[0.18, 0.82], hspace=0.04)
    fgrid = outer[1, 2].subgridspec(2, 1, height_ratios=[0.18, 0.82], hspace=0.04)

    axA_title = fig.add_subplot(agrid[0, 0]); axA = fig.add_subplot(agrid[1, 0])
    axB_title = fig.add_subplot(bgrid[0, 0]); axB = fig.add_subplot(bgrid[1, 0])
    axC_title = fig.add_subplot(cgrid[0, 0]); axC = fig.add_subplot(cgrid[1, 0])
    axD_title = fig.add_subplot(dgrid[0, 0]); axD = fig.add_subplot(dgrid[1, 0])
    axE_title = fig.add_subplot(egrid[0, 0]); axE = fig.add_subplot(egrid[1, 0])
    axF_title = fig.add_subplot(fgrid[0, 0]); axF = fig.add_subplot(fgrid[1, 0])

    thr_handles = [
        Line2D([0], [0], marker="o", color=colors[thr], lw=1.0,
               markersize=4.8, label=format_row_label(thr))
        for thr in thresholds if thr in set(ratio_df["threshold"])
    ]
    fig_leg = fig.legend(
        handles=thr_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.973),
        ncol=3,
        frameon=True,
        fontsize=SMALL_FS,
        columnspacing=1.0,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    fig_leg.get_frame().set_alpha(0.88)
    fig_leg.get_frame().set_facecolor("white")
    fig_leg.get_frame().set_edgecolor("0.80")

    plot_threshold_scatter(
        axA, ratio_df, "ratio_abs_median", "plddt_q10", ratio_pairwise_df, thresholds, colors,
        x_label="Absolute evidence ratio, |log10(holo/apo)|", y_label="pLDDT q10",
        logx=False, show_ylabel=True, add_size_legend=True, force_ylim=top_ylim
    )
    plot_threshold_scatter(
        axB, ratio_df, "evidence_total_median", "plddt_q10", ratio_pairwise_df, thresholds, colors,
        x_label="Median total evidence\n(ratio-derived bins)", y_label="pLDDT q10",

        logx=True, show_ylabel=False, add_size_legend=False, force_ylim=top_ylim
    )
    plot_threshold_scatter(
        axC, native_df, "evidence_total_median", "plddt_q10", native_pairwise_df, thresholds, colors,
        x_label="Median total evidence\n(native evidence bins)", y_label="pLDDT q10",

        logx=True, show_ylabel=False, add_size_legend=False, force_ylim=top_ylim
    )

    ct = list(contour_thresholds)
    while len(ct) < 3:
        ct.append(None)
    for ax, thr, show_ylabel, show_xlabel in [
        (axD, ct[0], True, False),
        (axE, ct[1], False, True),
        (axF, ct[2], False, False),
    ]:
        draw_raw_predictor_density(
            ax, raw_df, raw_predictor_corr_df,
            threshold=thr, show_ylabel=show_ylabel, show_xlabel=show_xlabel
        )

    draw_panel_title(axA_title, "A", "pLDDT q10 vs |ratio|")
    draw_panel_title(axB_title, "B", "pLDDT q10 vs total evidence")
    draw_panel_title(axC_title, "C", "pLDDT q10 vs total evidence")
    draw_panel_title(axD_title, "D", f"Predictor coupling — {format_contour_title_label(ct[0])}")
    draw_panel_title(axE_title, "E", f"Predictor coupling — {format_contour_title_label(ct[1])}")
    draw_panel_title(axF_title, "F", f"Predictor coupling — {format_contour_title_label(ct[2])}")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.915, bottom=0.084,
                        wspace=0.16, hspace=0.24)
    return fig

def save_figure(fig, outbase):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_PNG:
        fig.savefig(os.path.join(OUTPUT_DIR, outbase + ".png"), dpi=DPI_PNG,
                    bbox_inches="tight", pad_inches=0.03)
    if SAVE_PDF:
        fig.savefig(os.path.join(OUTPUT_DIR, outbase + ".pdf"),
                    bbox_inches="tight", pad_inches=0.03)
    if SAVE_SVG:
        fig.savefig(os.path.join(OUTPUT_DIR, outbase + ".svg"),
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_unified(INPUT_TABLE)
    require_cols(df, [
        "threshold", "level", "pocket_class",
        "evidence_n_apo", "evidence_n_holo", "model_pocket_plddt"
    ])

    df, thresholds = st_lj_aggregate_if_requested(df, THRESHOLDS_TO_PLOT)

    kept = []
    ratio_tables = []
    available_thresholds = set(df["threshold"].astype(str).unique())
    subsets_by_threshold = {}

    for thr in thresholds:
        if str(thr) not in available_thresholds:
            print(f"[warn] threshold not in table: {thr}")
            continue

        sub = subset_threshold(df, thr)
        subsets_by_threshold[thr] = sub.copy()

        if sub.empty:
            print(f"[warn] no rows after filters for: {thr}")
            continue

        kept.append(thr)
        ratio_tables.append(summarize_ratio_bins(sub, thr))

    if not kept:
        raise SystemExit("No data remain after filtering for selected thresholds.")

    ratio_df = pd.concat(ratio_tables, ignore_index=True)
    native_df = summarize_native_evidence_bins(subsets_by_threshold, kept, nbins=NATIVE_EVIDENCE_NBINS)
    raw_cols = ["threshold", "log10_ratio_abs", "evidence_total", "model_pocket_plddt"]
    raw_df = pd.concat([
        subsets_by_threshold[thr][["log10_ratio_abs", "evidence_total", "model_pocket_plddt"]].assign(threshold=thr)
        for thr in kept
    ], ignore_index=True)

    ratio_pairwise_df = compute_pairwise_correlations(ratio_df)
    native_pairwise_df = compute_pairwise_correlations(native_df)
    ratio_influence_df, ratio_predictor_corr_df = compute_influence_q10(ratio_df, kept)
    native_influence_df, native_predictor_corr_df = compute_influence_q10(native_df, kept)
    raw_predictor_corr_df = compute_raw_predictor_correlations(raw_df)

    panel_source_df = pd.concat([
        ratio_df.assign(source_mode="ratio_bins"),
        native_df.assign(source_mode="native_evidence_bins"),
    ], ignore_index=True, sort=False)

    fig = make_figure(
        ratio_df.loc[ratio_df["n_total"] > 0].copy(),
        ratio_pairwise_df,
        native_df.loc[native_df["n_total"] > 0].copy(),
        native_pairwise_df,
        raw_predictor_corr_df,
        raw_df,
        kept,
        contour_thresholds=CONTOUR_THRESHOLDS,
    )
    save_figure(fig, OUT_PREFIX + "_combined")

    if SAVE_RATIO_BIN_SUMMARY_CSV:
        ratio_df.to_csv(os.path.join(OUTPUT_DIR, OUT_PREFIX + "_ratio_bin_summary.csv"), index=False)
    if SAVE_NATIVE_BIN_SUMMARY_CSV:
        native_df.to_csv(os.path.join(OUTPUT_DIR, OUT_PREFIX + "_native_evidence_bin_summary.csv"), index=False)
    if SAVE_RATIO_PAIRWISE_CSV:
        ratio_pairwise_df.to_csv(os.path.join(OUTPUT_DIR, OUT_PREFIX + "_ratio_pairwise_spearman.csv"), index=False)
    if SAVE_NATIVE_PAIRWISE_CSV:
        native_pairwise_df.to_csv(os.path.join(OUTPUT_DIR, OUT_PREFIX + "_native_pairwise_spearman.csv"), index=False)
    if SAVE_NATIVE_INFLUENCE_CSV:
        ratio_influence_df.to_csv(os.path.join(OUTPUT_DIR, OUT_PREFIX + "_ratio_influence_q10.csv"), index=False)
        ratio_predictor_corr_df.to_csv(os.path.join(OUTPUT_DIR, OUT_PREFIX + "_ratio_predictor_collinearity.csv"), index=False)
        native_influence_df.to_csv(os.path.join(OUTPUT_DIR, OUT_PREFIX + "_native_influence_q10.csv"), index=False)
        native_predictor_corr_df.to_csv(os.path.join(OUTPUT_DIR, OUT_PREFIX + "_native_predictor_collinearity.csv"), index=False)
    if SAVE_RAW_PREDICTOR_CSV:
        raw_predictor_corr_df.to_csv(os.path.join(OUTPUT_DIR, OUT_PREFIX + "_raw_predictor_correlation.csv"), index=False)
    if SAVE_PANEL_SOURCE_CSV:
        panel_source_df.to_csv(os.path.join(OUTPUT_DIR, OUT_PREFIX + "_panel_source_data.csv"), index=False)

    print("Saved Figure S3 REV9 outputs to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()