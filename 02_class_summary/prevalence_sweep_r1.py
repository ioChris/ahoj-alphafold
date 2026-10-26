
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prevalence vs min average candidates sweep (apo vs holo)

Single-file tool:
- Core function: run_prevalence_sweep(...)
- CLI wrapper: see argparse at bottom
- Dependencies: pandas, matplotlib, numpy (numpy only for arange convenience)
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#from pathlib import Path
#base = Path.cwd()  # in Spyder this is your working dir
#parse_ahoj = (base.parent / "parse-ahoj").resolve() # Point to ../parse-ahoj
# Add once, give it priority so your local copy wins
#if str(parse_ahoj) not in sys.path:    sys.path.insert(0, str(parse_ahoj))  # 0 is fine; Spyder's path[0] is the cwd
#from stats_plot import set_plot_font

# add from AHoJ2 repo
#ahoj2 = (base.parent / "AHoJ2").resolve() # Point to ../AHoJ2
#if str(ahoj2) not in sys.path:    sys.path.insert(0, str(ahoj2))  # 0 is fine; Spyder's path[0] is the cwd
#from common import download_alphafold_model

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg



VALID_CLASSES = {"apo", "holo"}


def get_2level_cwd():
    npath = os.path.normpath(os.getcwd())   # Normalize the path string for the OS
    path0 = os.path.join(npath.split(os.sep)[0], os.sep, npath.split(os.sep)[1], npath.split(os.sep)[2])
    if os.path.exists(path0):
        memo = "Root path found >> " + path0
    else:
        memo = 'Error finding root path in working dir:' + npath
        path0 = None
    print(memo)
    return path0

def ensure_dir(p: str | None):
    if p:
        os.makedirs(p, exist_ok=True)
    return p

def save_figure(fig, basepath: str | None, formats: list[str], *, dpi: int = 150, close: bool = True):
    """Save `fig` to each format under basepath (without extension). If basepath is None, just show."""
    if not basepath:
        plt.show()
        return
    d = os.path.dirname(basepath)
    ensure_dir(d)
    stem = os.path.splitext(basepath)[0]
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", dpi=dpi)
    if close:
        plt.close(fig)



def _as_bool(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}

def _scale_sizes(values, *, min_size=30.0, max_size=500.0):
    """Map positive values to marker sizes in [min_size, max_size]."""
    v = np.asarray(values, dtype=float)
    v = np.where(np.isfinite(v), v, 0.0)
    if v.max() <= 0:
        return np.full_like(v, min_size, dtype=float)
    w = (v - v.min()) / (v.max() - v.min() + 1e-12)
    return min_size + w * (max_size - min_size)

def normalize_threshold_series(s: pd.Series) -> pd.Series:
    """
    Collapse any threshold value that starts with 'ST_LJ' to 'ST_LJ'.
    Handles composite strings like 'ST_LJ023:ST_LJ054' by mapping to 'ST_LJ'.
    """
    s = s.astype(str).str.strip()
    return s.where(~s.str.startswith("ST_LJ"), "ST_LJ")

def read_csv_or_zipped_csv(path: str, **read_csv_kwargs) -> pd.DataFrame:
    """
    Read a CSV from a normal path or from a .zip (with one CSV inside).
    If .zip has multiple CSVs, the first .csv entry is read.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    lower = path.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            # find the first CSV entry
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("The provided .zip does not contain any .csv file.")
            csv_name = csv_names[0]
            with zf.open(csv_name) as f:
                return pd.read_csv(f, **read_csv_kwargs)
    else:
        return pd.read_csv(path, **read_csv_kwargs)


def _coerce_numeric(s: pd.Series, nan_as_zero: bool) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    if nan_as_zero:
        return s2.fillna(0)
    return s2


def _iter_k_values(k_start: float, k_stop: float, k_step: float) -> Iterable[float]:
    """
    Inclusive sweep from k_start to k_stop with k_step. Supports floats.
    """
    if k_step == 0:
        raise ValueError("k_step must be non-zero.")
    # np.arange can miss the last value due to float error; pad a tiny epsilon
    eps = abs(k_step) * 1e-9 + 1e-12
    # Ensure we go the right direction
    if k_step > 0:
        stop = k_stop + eps
    else:
        stop = k_stop - eps
    return np.arange(k_start, stop, k_step)


def prevalence_by_min_avg(df: pd.DataFrame,
                          class_col: str,
                          avg_apo_col: str,
                          avg_holo_col: str,
                          *,
                          bin_width: float = 1.0,
                          nan_as_zero: bool = True,
                          grid_mode: str = "edges") -> pd.DataFrame:

    cls = df[class_col].astype(str).str.strip().str.lower()
    mask = cls.isin(VALID_CLASSES)
    if not mask.any():
        return pd.DataFrame(columns=["min_avg_bin","n_total","n_apo","n_holo","prev_apo_pct","prev_holo_pct"])

    a_apo  = _coerce_numeric(df.loc[mask, avg_apo_col],  nan_as_zero=nan_as_zero).to_numpy(float)
    a_holo = _coerce_numeric(df.loc[mask, avg_holo_col], nan_as_zero=nan_as_zero).to_numpy(float)
    lab    = cls.loc[mask].to_numpy()

    min_avg = np.minimum(a_apo, a_holo)
    maxv = float(np.nanmax(min_avg)) if min_avg.size else 0.0
    if bin_width <= 0:
        bin_width = 1.0
    edges = np.arange(0.0, maxv + bin_width + 1e-12, bin_width)

    # include zeros in first bin
    # include zeros in first bin
    idx = np.digitize(min_avg, edges, right=True) - 1
    valid = np.isfinite(min_avg) & (idx >= 0) & (idx < len(edges)-1)
    if not valid.any():
        return pd.DataFrame(columns=["min_avg_bin","n_total","n_apo","n_holo","prev_apo_pct","prev_holo_pct"])
    
    idx = idx[valid]; lv = lab[valid]
    n_total = np.bincount(idx, minlength=len(edges)-1)
    n_apo   = np.bincount(idx, weights=(lv == "apo").astype(int),  minlength=len(edges)-1)
    n_holo  = np.bincount(idx, weights=(lv == "holo").astype(int), minlength=len(edges)-1)

    
    if grid_mode == "centers":
        xcoord = (edges[:-1] + edges[1:]) / 2.0
    else:
        xcoord = edges[:-1].astype(float)
    #centers = (edges[:-1] + edges[1:]) / 2.0

    with np.errstate(invalid="ignore", divide="ignore"):
        prev_apo  = np.where(n_total > 0, 100.0 * n_apo  / n_total, np.nan)
        prev_holo = np.where(n_total > 0, 100.0 * n_holo / n_total, np.nan)

    return pd.DataFrame({
        "min_avg_bin": xcoord, #centers.astype(float),
        "n_total":     n_total.astype(int),
        "n_apo":       n_apo.astype(int),
        "n_holo":      n_holo.astype(int),
        "prev_apo_pct":  np.round(prev_apo,  3),
        "prev_holo_pct": np.round(prev_holo, 3),
    })

def _plot_bubble_prevalence_by_min_avg(df_min: pd.DataFrame,
                                       title: str,
                                       png_path: str | None,
                                       size_min: float = 30.0,
                                       size_max: float = 500.0):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    s_apo  = _scale_sizes(df_min["n_apo"],  min_size=size_min, max_size=size_max)
    s_holo = _scale_sizes(df_min["n_holo"], min_size=size_min, max_size=size_max)

    ax.scatter(df_min["min_avg_bin"], df_min["prev_apo_pct"],  s=s_apo,  label="prev_apo_pct")
    ax.plot   (df_min["min_avg_bin"], df_min["prev_apo_pct"],  marker="o", linewidth=1.0)
    ax.scatter(df_min["min_avg_bin"], df_min["prev_holo_pct"], s=s_holo, label="prev_holo_pct")
    ax.plot   (df_min["min_avg_bin"], df_min["prev_holo_pct"], marker="o", linewidth=1.0)

    ax.set_xlabel("min(avg_n_apo, avg_n_holo)")
    ax.set_ylabel("prevalence (%)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.9, title="Series")
    fig.tight_layout()
    if png_path:
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def _tail_prevalence_along_k(
    a_apo: np.ndarray,
    a_holo: np.ndarray,
    lab: np.ndarray,
    bin_width: float,
    grid_mode: str = "edges",   # NEW: "edges" or "centers"
):
    """
    Cumulative-tail prevalence along k: at each k, keep rows with (apo>=k & holo>=k).
    Returns k_grid (k), frac_apo, frac_holo in 0..1.
    """
    bw = bin_width if (bin_width and bin_width > 0) else 1.0
    k_min = np.minimum(a_apo, a_holo)
    if k_min.size == 0 or not np.isfinite(k_min).any():
        return None, None, None

    kmax = float(np.nanmax(k_min))
    edges = np.arange(0.0, kmax + bw + 1e-12, bw)
    if grid_mode == "centers":
        if edges.size < 2:
            return None, None, None
        k_grid = (edges[:-1] + edges[1:]) / 2.0
    else:
        # default: evaluate at the exact k edges (0, 1, 2, ...)
        if edges.size < 1:
            return None, None, None
        k_grid = edges.astype(float)

    frac_h = np.full(len(k_grid), np.nan, dtype=float)
    frac_a = np.full(len(k_grid), np.nan, dtype=float)
    finite = np.isfinite(a_apo) & np.isfinite(a_holo)

    for i, k in enumerate(k_grid):
        keep = finite & (a_apo >= k) & (a_holo >= k)
        n = int(keep.sum())
        if n == 0:
            continue
        h = int((lab[keep] == "holo").sum())
        frac_h[i] = h / n
        frac_a[i] = 1.0 - frac_h[i]

    return k_grid, frac_a, frac_h


def _compute_x_vals(lab: np.ndarray, a_apo: np.ndarray, a_holo: np.ndarray, x_mode: str, delta_dir: str = "holo_minus_apo"):
    """
    Return (x_vals, x_label) according to x_mode.
    x_mode: "min" (default), "class", "max", "common", "delta", or "ratio" (log2(avg_n_holo/avg_n_apo)).
    """
    x_mode = (x_mode or "min").lower()
    if x_mode == "class":
        x_vals = np.where(lab == "holo", a_holo, a_apo)
        x_label = "avg_n_class (apo→avg_n_apo, holo→avg_n_holo)"
    elif x_mode == "max":
        x_vals = np.maximum(a_apo, a_holo)
        x_label = "max(avg_n_apo, avg_n_holo)"
    elif x_mode == "common":
        x_vals  = np.minimum(a_apo, a_holo)
        x_label = "common min k (apo & holo ≥ k)"
    elif x_mode == "delta":
        if (delta_dir or "holo_minus_apo") == "holo_minus_apo":
            x_vals  = a_holo - a_apo
            x_label = "Δ = avg_n_holo - avg_n_apo"
        else:
            x_vals  = a_apo - a_holo
            x_label = "Δ = avg_n_apo - avg_n_holo"
    elif x_mode in ("ratio", "ratio_log2", "logratio"):
        # NEW: bin by log2 ratio of average counts (holo vs apo)
        x_vals  = np.log2(a_holo / a_apo)
        x_label = "log₂(avg_n_holo / avg_n_apo)"
    # inside _compute_x_vals(...)
    elif x_mode == "mean":
        x_vals  = 0.5 * (a_apo + a_holo)
        x_label = "mean(avg_n_apo, avg_n_holo)"

    else:
        x_vals = np.minimum(a_apo, a_holo)
        x_label = "min(avg_n_apo, avg_n_holo)"
    return x_vals, x_label



def _binned_prevalence_along_x(
    x_vals: np.ndarray,
    lab: np.ndarray,
    bin_width: float,
    grid_mode: str = "edges",  # "edges" or "centers"
    bin_origin: float = 0.0,   # NEW: anchor bins to this origin (default 0.0)
):
    """
    Bin by x_vals and compute apo/holo fractions per bin.
    Returns k_grid (bin coord), frac_apo, frac_holo. Fractions are 0..1.

    grid_mode:
      - "edges": evaluate at left bin edges
      - "centers": evaluate at mid-bin centers

    bin_origin:
      - controls where bin edges align (useful if you want a bin edge at exactly 0.0)
    """
    bw = float(bin_width) if (bin_width and bin_width > 0) else 1.0
    if x_vals.size == 0:
        return None, None, None

    finite_x = x_vals[np.isfinite(x_vals)]
    if finite_x.size == 0:
        return None, None, None

    minv = float(np.nanmin(finite_x))
    maxv = float(np.nanmax(finite_x))

    # Anchor edges so negatives are included and 0.0 can be an exact edge.
    start = bin_origin + bw * np.floor((minv - bin_origin) / bw)
    stop  = bin_origin + bw * np.ceil((maxv - bin_origin) / bw)
    # +bw to include the rightmost edge; tiny epsilon avoids FP issues
    edges = np.arange(start, stop + bw * 1e-12, bw)

    if edges.size < 3:  # need at least 2 bins
        return None, None, None

    # Bin indices for all x (now includes negatives)
    idx = np.digitize(x_vals, edges, right=True) - 1
    valid = np.isfinite(x_vals) & (idx >= 0) & (idx < (len(edges) - 1))
    if not valid.any():
        return None, None, None

    idx = idx[valid]
    lv  = lab[valid]

    n_tot = np.bincount(idx, minlength=len(edges) - 1)
    n_h   = np.bincount(idx, weights=(lv == "holo").astype(int), minlength=len(edges) - 1)
    n_a   = np.bincount(idx, weights=(lv == "apo").astype(int),  minlength=len(edges) - 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        frac_h = np.where(n_tot > 0, n_h / n_tot, np.nan)
        frac_a = np.where(n_tot > 0, n_a / n_tot, np.nan)

    if grid_mode == "centers":
        k_grid = 0.5 * (edges[:-1] + edges[1:])
    else:
        k_grid = edges[:-1].astype(float)

    return k_grid, frac_a, frac_h




def _plot_scatter_min_avg_class(df: pd.DataFrame,
                                class_col: str,
                                avg_apo_col: str,
                                avg_holo_col: str,
                                *,
                                nan_as_zero: bool = True,
                                title: str = "Scatter vs min/max/class(avg) — trend-mapped",
                                save_base: str | None = None, formats: list[str] = ("png",),
                                #png_path: str | None = None,
                                sample_n: int = 200_000,
                                jitter_y: float = 0.02,     # fraction of full scale
                                alpha: float = 0.15,
                                s: float = 6.0,
                                x_mode: str = "min",
                                y_mode: str = "trend",      # "trend" or "class"
                                overlay_trend: bool = True,
                                trend_bin_width: float = 1.0,
                                y_units: str = "percent",   # "fraction" or "percent"
                                y_pad: float = 0.025,       # extra pad beyond canonical range
                                x_lo: float | None = None,
                                x_hi: float | None = None,
                                delta_dir="holo_minus_apo",
                                x_symmetric_delta=True,
                                scatter_style=None,
                                rasterize=None,
                                grid_mode="centers", # centers | edges
                                ):
    """
    Scatter of UBS points vs x (chosen by x_mode):
      - x_mode="min":   x = min(avg_n_apo, avg_n_holo)
      - x_mode="class": x = avg_n_apo for apo, avg_n_holo for holo
      - x_mode="max":   x = max(avg_n_apo, avg_n_holo)

    y_mode:
      - "trend": point y follows local class prevalence along the chosen x
      - "class": point y is 0 (apo) or 1 (holo)

    y_units: "fraction" (0..1) or "percent" (0..100). Padding uses y_pad and respects jitter.
    """

    from matplotlib.lines import Line2D

    # filter to apo/holo
    cls = df[class_col].astype(str).str.strip().str.lower()
    mask = cls.isin(VALID_CLASSES)
    if not mask.any():
        return

    # numeric columns
    a_apo  = _coerce_numeric(df.loc[mask, avg_apo_col],  nan_as_zero=nan_as_zero).to_numpy(float)
    a_holo = _coerce_numeric(df.loc[mask, avg_holo_col], nan_as_zero=nan_as_zero).to_numpy(float)
    lab    = cls.loc[mask].to_numpy()

    # choose x by mode
    x_vals, x_label = _compute_x_vals(lab, a_apo, a_holo, x_mode=x_mode, delta_dir=delta_dir)

    # optional sampling + x-range filter — keep arrays aligned
    idx = np.arange(x_vals.shape[0])

    if sample_n and sample_n > 0 and x_vals.shape[0] > sample_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(x_vals.shape[0], size=sample_n, replace=False)

    if (x_lo is not None) or (x_hi is not None):
        lo = -np.inf if x_lo is None else float(x_lo)
        hi =  np.inf if x_hi is None else float(x_hi)
        keep = (x_vals[idx] >= lo) & (x_vals[idx] <= hi)
        idx = idx[keep]

    # apply selection consistently
    x_vals = x_vals[idx]
    lab    = lab[idx]
    a_apo  = a_apo[idx]
    a_holo = a_holo[idx]

    if x_vals.size == 0:
        return

    # y (base) in FRACTION
    if y_mode == "class":
        y_base_frac = (lab == "holo").astype(float)
        centers = frac_a = frac_h = None
    else:
        if x_mode == "common":
            # Use the cumulative-tail definition (both apo & holo >= k)
            centers, frac_a, frac_h = _tail_prevalence_along_k(a_apo, a_holo, lab, trend_bin_width, grid_mode=grid_mode)
        else:
            # Local (disjoint-bin) prevalence along whatever x you chose (min/class/max)
            centers, frac_a, frac_h = _binned_prevalence_along_x(x_vals, lab, trend_bin_width, grid_mode=grid_mode)

        if centers is None:
            y_base_frac = (lab == "holo").astype(float)
        else:
            # interpolate class-specific fraction to each point
            y_a = np.interp(x_vals, centers[np.isfinite(frac_a)], frac_a[np.isfinite(frac_a)],
                            left=np.nanmean(frac_a), right=np.nanmean(frac_a))
            y_h = np.interp(x_vals, centers[np.isfinite(frac_h)], frac_h[np.isfinite(frac_h)],
                            left=np.nanmean(frac_h), right=np.nanmean(frac_h))
            y_base_frac = np.where(lab == "holo", y_h, y_a)

    # --- Method note: explain how y was derived ---
    if y_mode == "class":
        # Dots at 0/1 bands; trend (if any) indicates prevalence
        if overlay_trend and (centers is not None):
            trend_note = "cumulative-tail (apo & holo ≥ k)" if x_mode == "common" else "binned along x"
            method_note = f"Y = class bands (0/1) • overlay: {trend_note}"
        else:
            method_note = "Y = class bands (0/1)"
    else:
        # Dots mapped to local prevalence by x
        method_note = "Cumulative-tail (apo & holo ≥ k)" if x_mode == "common" else "Binned along x"
        #method_note = "Y = prevalence • " + ("cumulative-tail (apo & holo ≥ k)" if x_mode == "common" else "binned along x")

    # scale + jitter (no clipping)
    scale = 100.0 if y_units == "percent" else 1.0
    y_scaled = y_base_frac * scale
    if jitter_y > 0:
        rng = np.random.default_rng(123)
        y_scaled = y_scaled + (rng.random(y_scaled.shape[0]) - 0.5) * 2.0 * (jitter_y * scale)

    # plot
    #fig, ax = plt.subplots(figsize=(7, 4.2))

    # masks and counts
    #m_apo, m_holo = (lab == "apo"), (lab == "holo")
    #n_apo  = int(m_apo.sum())
    #n_holo = int(m_holo.sum())
    #n_tot  = int(x_vals.shape[0])

    # scatter with count-aware legend labels
    #ax.scatter(x_vals[m_apo],  y_scaled[m_apo],  s=s, alpha=alpha, label=f"apo (n={n_apo:,})")
    #ax.scatter(x_vals[m_holo], y_scaled[m_holo], s=s, alpha=alpha, label=f"holo (n={n_holo:,})")

    fig, ax = plt.subplots(figsize=(7, 4.2))

    # masks and counts
    m_apo, m_holo = (lab == "apo"), (lab == "holo")
    n_apo  = int(m_apo.sum())
    n_holo = int(m_holo.sum())
    n_tot  = int(x_vals.shape[0])

    raster = bool(rasterize)

    if scatter_style == "default":
        # Point size and transparency from function arguments
        sc_apo  = ax.scatter(x_vals[m_apo],   y_scaled[m_apo],  s=s, alpha=alpha, label=f"apo (n={n_apo:,})",   rasterized=raster)
        sc_holo = ax.scatter(x_vals[m_holo], y_scaled[m_holo], s=s, alpha=alpha, label=f"holo (n={n_holo:,})", rasterized=raster)

    elif scatter_style == "points":
        # Tiny, very transparent points; fastest minimal change
        sc_apo  = ax.scatter(x_vals[m_apo],   y_scaled[m_apo],  s=4, alpha=0.05, label=f"apo (n={n_apo:,})",   rasterized=raster)
        sc_holo = ax.scatter(x_vals[m_holo], y_scaled[m_holo], s=4, alpha=0.05, label=f"holo (n={n_holo:,})", rasterized=raster)

    elif scatter_style == "jitter":
        # Add small vertical jitter so dense columns become legible (esp. in class mode at y=0/1)
        rng = np.random.default_rng(123)
        # jitter width scales with the y-range you're using; for class bands (0/1) 0.06 works well
        jitter = 0.06
        y_apo  = y_scaled[m_apo]  + rng.uniform(-jitter, jitter, n_apo)
        y_holo = y_scaled[m_holo] + rng.uniform(-jitter, jitter, n_holo)

        sc_apo  = ax.scatter(x_vals[m_apo],  y_apo,  s=3, alpha=0.06, label=f"apo (n={n_apo:,})",
                   linewidths=0, rasterized=raster)
        sc_holo = ax.scatter(x_vals[m_holo], y_holo, s=3, alpha=0.06, label=f"holo (n={n_holo:,})",
                   linewidths=0, rasterized=raster)

    elif scatter_style == "hexbin_facets":

        # Two-panel hexbin view with unified styling & shared color scale
        plt.close(fig)  # discard single-axes fig created earlier
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True, figsize=(7.5, 6.2),
            height_ratios=[1, 1], constrained_layout=True)   # helps keep cbar off the data area
        fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.09, hspace=0.03, wspace=0.02)

        # Masks & counts
        m_apo, m_holo = (lab == "apo"), (lab == "holo")
        n_apo  = int(m_apo.sum())
        n_holo = int(m_holo.sum())
        n_tot  = int(x_vals.shape[0])

        # Consistent ranges
        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
        if y_units == "percent":
            y_min, y_max = 0.0, 100.0
            yticks = np.arange(0, 101, 10)      # same ticks top/bottom
            ylab_top  = "apo prevalence (%)"
            ylab_bot  = "holo prevalence (%)"
            # use y_pad if given; otherwise a gentle default (1.5 percentage points)
            pad = max(float(y_pad), 2.5) # 1.5
            y_lo = 0.0 - pad
            y_hi = 100 + pad
        else:
            y_min, y_max = 0.0, 1.0
            yticks = np.linspace(0, 1, 11)
            ylab_top  = "apo prevalence (fraction)"
            ylab_bot  = "holo prevalence (fraction)"
            # for fraction, default pad ~0.03
            pad = max(float(y_pad), 1.03) # 0.03
            y_lo = 0.0 - pad
            y_hi = 100 + pad
        extent = (x_min, x_max, y_min, y_max)

        # --- aspect-aware hexbin density with pixel-based control ---
        """
        # Data spans (use displayed y span with padding so it matches the axes limits)
        x_span = (x_max - x_min)
        y_span = (y_hi  - y_lo)
        
        # Axes pixel size / aspect (as you already have)
        bbox   = ax_top.get_position()
        fig_w, fig_h = fig.get_size_inches(); dpi = fig.get_dpi()
        axes_w_px = bbox.width  * fig_w * dpi
        axes_h_px = bbox.height * fig_h * dpi
        axes_aspect = axes_h_px / axes_w_px  # pixels_y / pixels_x
        
        # 1) Tile size in pixels (tune "resolution")
        hex_px = 8                   # 8–12 is a good range
        
        # 2) Shape scale (tune "tall vs flat")
        shape_scale = 1.5            # >1.0 → flatter; <1.0 → taller. Try 0.95–1.15
        
        # Base column count from width in pixels
        nx = max(40, min(400, int(round(axes_w_px / hex_px))))
        
        # Derive row count to make hexes look regular, then apply shape_scale
        # 0.866 ≈ sqrt(3)/2 compensates flat-top hex geometry
        ny_float = nx * (y_span / x_span) * axes_aspect / 0.866
        ny = max(10, min(400, int(round(ny_float * shape_scale))))

        # ------------------------------------------------------------

        # Hexbins per class (same grid for both)
        hb_apo = ax_top.hexbin(
            x_vals[m_apo], y_scaled[m_apo],
            gridsize=(nx, ny), extent=extent, mincnt=1, linewidths=0,
            rasterized=raster
        )
        hb_holo = ax_bot.hexbin(
            x_vals[m_holo], y_scaled[m_holo],
            gridsize=(nx, ny), extent=extent, mincnt=1, linewidths=0,
            rasterized=raster
        )
        """

        hb_apo = ax_top.hexbin(x_vals[m_apo], y_scaled[m_apo],
            gridsize=(63, 21), extent=extent, mincnt=1, linewidths=0, rasterized=raster) # gridsize=(60, 30), (50, 20)
        hb_holo = ax_bot.hexbin(x_vals[m_holo], y_scaled[m_holo],
            gridsize=(63, 21), extent=extent, mincnt=1, linewidths=0, rasterized=raster) # gridsize=(60, 30), 

        # Shared color normalization so colors mean the same count in both panels
        from matplotlib.colors import LogNorm
        arr1, arr2 = hb_apo.get_array(), hb_holo.get_array()
        vmax = max(float(np.nanmax(arr1)) if arr1.size else 0.0, float(np.nanmax(arr2)) if arr2.size else 0.0, 1.0)
        shared_norm = LogNorm(vmin=1.0, vmax=vmax)
        hb_apo.set_norm(shared_norm); hb_holo.set_norm(shared_norm)

        # One shared colorbar on the right
        cbar = fig.colorbar(hb_holo, ax=[ax_top, ax_bot], orientation="vertical", pad=0.02)
        cbar.set_label("UBS count (shared scale)")

        # Unified y limits/ticks and identical gridlines on both panels
        for axp in (ax_top, ax_bot):
            axp.set_ylim(y_lo, y_hi)           # <- apply padded limits
            #axp.set_ylim(y_min, y_max)
            axp.set_yticks(yticks)
            axp.grid(True, which="both", axis="both", alpha=0.25)
            axp.set_axisbelow(True)

        # Per-panel y labels (distinct), shared x (ticks only on bottom)
        ax_top.set_ylabel(ylab_top,  fontsize=10)
        ax_bot.set_ylabel(ylab_bot,  fontsize=10)
        ax_top.tick_params(labelbottom=False)
        ax_bot.set_xlabel(x_label)

        # Compute trend locally (so we can overlay it here, honoring grid_mode)
        k_grid = frac_a = frac_h = None
        if overlay_trend:
            if x_mode == "common":
                k_grid, frac_a, frac_h = _tail_prevalence_along_k(
                    a_apo, a_holo, lab, trend_bin_width, grid_mode=grid_mode)
            else:
                k_grid, frac_a, frac_h = _binned_prevalence_along_x(
                    x_vals, lab, trend_bin_width, grid_mode=grid_mode)

        # Overlay trend on BOTH panels (same curve for reference)
        if overlay_trend and (k_grid is not None):
            fa = np.isfinite(frac_a) if frac_a is not None else None
            fh = np.isfinite(frac_h) if frac_h is not None else None
            if y_units == "percent":
                if fh is not None and fh.any():
                    ax_top.plot(k_grid[fh], frac_h[fh] * 100.0, linewidth=1.3, zorder=10)
                    ax_bot.plot(k_grid[fh], frac_h[fh] * 100.0, linewidth=1.3, zorder=10)
                if fa is not None and fa.any():
                    ax_top.plot(k_grid[fa], frac_a[fa] * 100.0, linewidth=1.3, zorder=10)
                    ax_bot.plot(k_grid[fa], frac_a[fa] * 100.0, linewidth=1.3, zorder=10)
            else:
                if fh is not None and fh.any():
                    ax_top.plot(k_grid[fh], frac_h[fh], linewidth=1.3, zorder=10)
                    ax_bot.plot(k_grid[fh], frac_h[fh], linewidth=1.3, zorder=10)
                if fa is not None and fa.any():
                    ax_top.plot(k_grid[fa], frac_a[fa], linewidth=1.3, zorder=10)
                    ax_bot.plot(k_grid[fa], frac_a[fa], linewidth=1.3, zorder=10)

        # NEW: let constrained_layout place the suptitle; then anchor the right note to the same y
        st = fig.suptitle(title, fontsize=12, x=0.01, ha="left")  # no explicit 'y'; CL will reserve space
        y_sup = st.get_position()[1]  # suptitle's y in figure coords

        fig.text(0.99, y_sup, f"{method_note} • Total n = {n_tot:,}",
            transform=fig.transFigure, ha="right", va="center", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))

        # Use bottom axes for downstream saving code that expects 'ax'
        ax = ax_bot

    else:
        raise ValueError(f"Unknown scatter_style: {scatter_style}")

    # For non-facet styles, keep your existing title and the outside Total n tag:
    if scatter_style in ("points", "jitter", "default"):
        ax.set_title(title)
    
        # optional thin trends along chosen x
        if overlay_trend and (centers is not None):
            if y_units == "percent":
                ax.plot(centers, frac_a * 100.0, linewidth=1.25, label="apo  trend")
                ax.plot(centers, frac_h * 100.0, linewidth=1.25, label="holo trend")
            else:
                ax.plot(centers, frac_a,        linewidth=1.25, label="apo  trend")
                ax.plot(centers, frac_h,        linewidth=1.25, label="holo trend")
    
        # axes
        pad = max(float(y_pad), 1.25 * float(jitter_y)) * scale
        if y_units == "percent":
            ax.set_ylabel("prevalence (%)")
            ax.set_ylim(0 - pad, 100 + pad)
            ax.set_yticks(list(range(0, 101, 10)))
        else:
            ax.set_ylabel("prevalence (fraction)")
            ax.set_ylim(0 - pad, 1 + pad)
            ax.set_yticks(np.linspace(0, 1, 6))
    
        # Delta-specific cosmetics
        if x_mode == "delta":
            ax.axvline(0.0, linestyle="--", alpha=0.4)
            # If user didn't fix xmin/xmax and symmetric mode is on, make x-lims symmetric around 0
            if (x_lo is None) and (x_hi is None) and x_symmetric_delta:
                lo, hi = ax.get_xlim()
                m = max(abs(lo), abs(hi))
                ax.set_xlim(-1.05*m, 1.05*m)
    
        ax.set_xlabel(x_label)
        ax.set_xmargin(0.03)
        ax.autoscale(enable=True, axis="x", tight=False)
        ax.grid(True, linestyle="--", alpha=0.35)

    #if scatter_style in ("points", "jitter", "default"):
        # Build opaque proxy handles for legend (so legend markers ignore the scatter alpha)
        c_apo  = sc_apo.get_facecolors()[0] if len(sc_apo.get_facecolors()) else (0,0,0,1)
        c_holo = sc_holo.get_facecolors()[0] if len(sc_holo.get_facecolors()) else (0,0,0,1)

        handles = [
            Line2D([], [], marker='o', linestyle='None', markersize=6,
                   markerfacecolor=(c_apo[0], c_apo[1], c_apo[2], 1.0),
                   markeredgecolor='none', label=f"apo (n={n_apo:,})"),
            Line2D([], [], marker='o', linestyle='None', markersize=6,
                   markerfacecolor=(c_holo[0], c_holo[1], c_holo[2], 1.0),
                   markeredgecolor='none', label=f"holo (n={n_holo:,})"),
        ]
        ax.legend(handles=handles, loc="best", framealpha=0.95, prop={"size": 9})


    #ax.legend(loc="best", framealpha=0.9)
        ax.set_title(title)


    # Place "Total n" outside the plot, top-right, just above the axes
        ax.text(
            0.995, 1.015, f"Total n = {n_tot:,}\n{method_note}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7, color='#333',
            bbox=dict(facecolor='white', edgecolor='none', pad=2),   # <— no 'alpha' key; alpha lives inside the RGBA facecolor
            clip_on=False, zorder=99)
    # box_kwargs = dict(facecolor=(1,1,1,0.80), edgecolor='none', pad=2)

    # Only use tight_layout for single-axes styles; hexbin uses constrained_layout
    #if scatter_style in ("points", "jitter", "default"):
        fig.tight_layout()

    save_figure(fig, save_base, formats)


def _plot_prevalence_heatmap_2d(df: pd.DataFrame,
                                class_col: str,
                                avg_apo_col: str,
                                avg_holo_col: str,
                                *,
                                bins_apo: int = 20,
                                bins_holo: int = 20,
                                nan_as_zero: bool = True,
                                title: str,
                                png_path: str | None):
    cls = df[class_col].astype(str).str.strip().str.lower()
    mask = cls.isin(VALID_CLASSES)
    if not mask.any():
        return

    x = _coerce_numeric(df.loc[mask, avg_apo_col],  nan_as_zero=nan_as_zero).to_numpy(float)
    y = _coerce_numeric(df.loc[mask, avg_holo_col], nan_as_zero=nan_as_zero).to_numpy(float)
    c = cls.loc[mask].to_numpy()

    # Histogram counts per class on a shared grid
    H_holo, xedges, yedges = np.histogram2d(x[c == "holo"], y[c == "holo"], bins=[bins_apo, bins_holo])
    H_apo,  _,      _      = np.histogram2d(x[c == "apo"],  y[c == "apo"],  bins=[xedges, yedges])
    H_tot = H_holo + H_apo
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_holo = np.where(H_tot > 0, H_holo / H_tot, np.nan)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    m = ax.pcolormesh(xedges, yedges, frac_holo.T, shading="auto")
    cb = fig.colorbar(m, ax=ax)
    cb.set_label("holo prevalence (fraction)")
    ax.set_xlabel(avg_apo_col)
    ax.set_ylabel(avg_holo_col)
    ax.set_title(title)
    fig.tight_layout()
    if png_path:
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def _plot_bubble_prevalence_counts(df, x_col, y_cols=("prev_apo_pct","prev_holo_pct"),
                                   size_cols=("n_apo","n_holo"), title="Prevalence vs k (bubble=size)",
                                   save_base: str | None = None, formats: list[str] = ("png",),
                                   size_min: float = 30.0, size_max: float = 500.0):
    """
    Single-axes bubble plot:
      - x = k
      - y = prevalence (%) for apo/holo
      - marker size encodes n_apo / n_holo per respective series
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    s_apo  = _scale_sizes(df[size_cols[0]], min_size=size_min, max_size=size_max)
    s_holo = _scale_sizes(df[size_cols[1]], min_size=size_min, max_size=size_max)
    ax.scatter(df[x_col], df[y_cols[0]], s=s_apo,  label=y_cols[0]); ax.plot(df[x_col], df[y_cols[0]], marker="o", linewidth=1.0)
    ax.scatter(df[x_col], df[y_cols[1]], s=s_holo, label=y_cols[1]); ax.plot(df[x_col], df[y_cols[1]], marker="o", linewidth=1.0)
    ax.set_xlabel(x_col); ax.set_ylabel("prevalence (%)"); ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35); ax.legend(loc="best", framealpha=0.9, title="Series")
    fig.tight_layout()
    save_figure(fig, save_base, formats)


def run_prevalence_sweep(
    df: pd.DataFrame,
    class_col: str,
    avg_apo_col: str,
    avg_holo_col: str,
    *,
    mode: str = "per_class",
    k_start: float = 0,
    k_stop: float = 10,
    k_step: float = 1,
    nan_as_zero: bool = True,
) -> pd.DataFrame:
    """
    Compute classification prevalence across thresholds k.

    Modes:
      - "per_class": at threshold k, keep rows where
         (class == "apo"  and avg_apo >= k) or (class == "holo" and avg_holo >= k).
      - "global": keep rows where (avg_apo >= k and avg_holo >= k) regardless of class.
    """
    if class_col not in df.columns:
        raise KeyError(f"Missing required column: {class_col}")
    if avg_apo_col not in df.columns:
        raise KeyError(f"Missing required column: {avg_apo_col}")
    if avg_holo_col not in df.columns:
        raise KeyError(f"Missing required column: {avg_holo_col}")

    # Normalize classification column to lowercase strings and filter to apo/holo
    class_ser = df[class_col].astype(str).str.strip().str.lower()
    valid_mask = class_ser.isin(VALID_CLASSES)
    df = df.loc[valid_mask].copy()
    if df.empty:
        # Return an empty tidy frame with the expected columns
        return pd.DataFrame(
            columns=["k", "n_total", "n_apo", "n_holo", "prev_apo_pct", "prev_holo_pct"]
        )

    # Coerce numeric columns
    df["_avg_apo"] = _coerce_numeric(df[avg_apo_col], nan_as_zero=nan_as_zero)
    df["_avg_holo"] = _coerce_numeric(df[avg_holo_col], nan_as_zero=nan_as_zero)
    df["_class"] = class_ser

    rows = []
    mode = str(mode).strip().lower()
    if mode not in {"per_class", "global"}:
        raise ValueError('mode must be "per_class" or "global"')

    for k in _iter_k_values(k_start, k_stop, k_step):
        if mode == "per_class":
            keep_mask = ((df["_class"] == "apo") & (df["_avg_apo"] >= k)) | \
                        ((df["_class"] == "holo") & (df["_avg_holo"] >= k))
        else:  # global
            keep_mask = (df["_avg_apo"] >= k) & (df["_avg_holo"] >= k)

        d = df.loc[keep_mask]

        n_total = int(len(d))
        if n_total == 0:
            n_apo = n_holo = 0
            prev_apo_pct = prev_holo_pct = float("nan")
        else:
            n_apo = int((d["_class"] == "apo").sum())
            n_holo = int((d["_class"] == "holo").sum())
            prev_apo_pct = 100.0 * n_apo / n_total
            prev_holo_pct = 100.0 * n_holo / n_total

        rows.append({
            "k": float(k),
            "n_total": n_total,
            "n_apo": n_apo,
            "n_holo": n_holo,
            "prev_apo_pct": round(prev_apo_pct, 3) if pd.notna(prev_apo_pct) else float("nan"),
            "prev_holo_pct": round(prev_holo_pct, 3) if pd.notna(prev_holo_pct) else float("nan"),
        })

    out = pd.DataFrame(rows, columns=["k", "n_total", "n_apo", "n_holo", "prev_apo_pct", "prev_holo_pct"])
    return out



def _plot_lines(df: pd.DataFrame, x_col: str, y_cols: Tuple[str,str], title: str, save_base: str | None, formats: list[str]):
    """
    Single-axes line plot for two series.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for y in y_cols: ax.plot(df[x_col], df[y], marker="o", label=y)
    ax.set_xlabel(x_col); ax.set_ylabel("UBS"); ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9); ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, save_base, formats)


def _print_preview(df: pd.DataFrame, max_rows: int = 6):
    """
    Print a small preview and first/last few rows.
    """
    if df.empty:
        print("Summary is empty.")
        return
    print("\n== Preview (head) ==")
    print(df.head(max_rows).to_string(index=False))
    print("\n== Preview (tail) ==")
    print(df.tail(max_rows).to_string(index=False))


def float_to_str(value, treat_none_as=0):
    if value:
        return str(value)
    else:
        if treat_none_as:
            return str(treat_none_as)
        else:
            return '0'


def _plot_prob_vs_avg_n(df, avg_apo_col, avg_holo_col,
                        p_apo_col, p_holo_col,
                        class_col: str,
                        title: str, save_base: str | None,
                        formats=("png",), nan_as_zero=True):
    """Correlation scatter of average n vs probability per class."""
    cls = df[class_col].astype(str).str.strip().str.lower()
    mask = cls.isin(VALID_CLASSES)
    if not mask.any():
        return

    a_apo = _coerce_numeric(df.loc[mask, avg_apo_col], nan_as_zero=nan_as_zero)
    a_holo = _coerce_numeric(df.loc[mask, avg_holo_col], nan_as_zero=nan_as_zero)
    p_apo = _coerce_numeric(df.loc[mask, p_apo_col], nan_as_zero=False)
    p_holo = _coerce_numeric(df.loc[mask, p_holo_col], nan_as_zero=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sc1 = ax[0].scatter(a_apo, p_apo, s=8, alpha=0.3, label="apo points", rasterized=True)
    sc2 = ax[1].scatter(a_holo, p_holo, s=8, alpha=0.3, label="holo points", rasterized=True)

    for i, lbl in enumerate(("apo", "holo")):
        ax[i].set_xlabel(f"avg_n_{lbl}")
        ax[i].set_ylabel(f"avg_P({lbl})_mean_pocket")
        ax[i].grid(True, linestyle="--", alpha=0.35)
        r = np.corrcoef(a_apo if i==0 else a_holo,
                        p_apo if i==0 else p_holo)[0, 1]
        ax[i].text(0.98, 0.05, f"r = {r:.3f}", ha="right", va="bottom",
                   transform=ax[i].transAxes, fontsize=9,
                   bbox=dict(facecolor="white", edgecolor="none", pad=2))

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, save_base, formats)


def _plot_prob_balance_heatmap(df, avg_apo_col, avg_holo_col,
                               p_apo_col, p_holo_col,
                               title, save_base=None, formats=("png",),
                               bins=40, nan_as_zero=True):
    """2D heatmap showing mean ΔP = P(holo)-P(apo) across (avg_n_apo, avg_n_holo)."""
    a_apo = _coerce_numeric(df[avg_apo_col], nan_as_zero=nan_as_zero)
    a_holo = _coerce_numeric(df[avg_holo_col], nan_as_zero=nan_as_zero)
    p_apo = _coerce_numeric(df[p_apo_col], nan_as_zero=False)
    p_holo = _coerce_numeric(df[p_holo_col], nan_as_zero=False)
    dP = p_holo - p_apo

    # grid average
    H_sum, xedges, yedges = np.histogram2d(a_apo, a_holo, bins=bins, weights=dP)
    H_count, _, _ = np.histogram2d(a_apo, a_holo, bins=[xedges, yedges])
    H_mean = np.divide(H_sum, H_count, where=H_count > 0)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.pcolormesh(xedges, yedges, H_mean.T, shading="auto", cmap="coolwarm", vmin=-1, vmax=1)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("ΔP = avg_P(holo) − avg_P(apo)")
    ax.set_xlabel(avg_apo_col)
    ax.set_ylabel(avg_holo_col)
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, save_base, formats)


from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

def _get_apo_holo_cmap(mid_frac: float = 0.10,
                       blue: str = "#4C72B0",   # apo
                       red:  str = "#C44E52",   # holo
                       #mid:  str = "#E9E9E9"):  # very light grey
                       #mid:  str = "#F1F1F1"):  # very light grey +
                       #mid:  str = "#F6F6F6"):  # very light grey ++
                       mid:  str = "#FFFFFF"):  # white
    """
    Diverging colormap with a pale-grey center band around 0.
    mid_frac controls how wide that grey band is around the center of the range.
    """
    mid_frac = max(0.0, min(0.45, float(mid_frac)))  # keep sane
    # positions in [0..1]; 0.5 is the center (Δ ≈ 0)
    a = 0.5 - mid_frac/2.0
    b = 0.5 + mid_frac/2.0
    # build smooth transition: blue → blue_near → grey → red_near → red
    colors = [
        (0.00, blue),
        (a,    blue),
        (0.50, mid),
        (b,    red),
        (1.00, red),
    ]
    cmap = LinearSegmentedColormap.from_list("apo_holo_greycenter", colors, N=256)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    return cmap, norm





def _plot_prob_vs_n_ratio(df,
                          avg_apo_col, avg_holo_col,
                          p_apo_col, p_holo_col,
                          title, save_base=None, formats=("png",),
                          nan_as_zero=True,
                          color_by_maxP: bool = True):  # NEW
    """Plot ΔP (holo−apo) vs log2 ratio of (avg_n_holo / avg_n_apo).
       Optionally color points by signed max probability (apo→blue, holo→red)."""
    a_apo  = _coerce_numeric(df[avg_apo_col],  nan_as_zero=nan_as_zero)
    a_holo = _coerce_numeric(df[avg_holo_col], nan_as_zero=nan_as_zero)
    p_apo  = _coerce_numeric(df[p_apo_col],    nan_as_zero=False)
    p_holo = _coerce_numeric(df[p_holo_col],   nan_as_zero=False)

    ratio = np.log2(a_holo / a_apo)
    dP    = p_holo - p_apo

    # mask finite
    mask = np.isfinite(ratio) & np.isfinite(dP) & np.isfinite(p_apo) & np.isfinite(p_holo)
    if not mask.any():
        return
    ratio, dP, p_apo, p_holo = ratio[mask], dP[mask], p_apo[mask], p_holo[mask]

    # correlation (Pearson on raw points)
    r_pearson = np.corrcoef(ratio, dP)[0, 1]

    # --- Color mapping (apo→blue, holo→red); lighten ties/near-ties ---
    """
    if color_by_maxP:
        pmax  = np.maximum(p_apo, p_holo)                 # 0..1 (confidence)
        sign  = np.where(p_holo >= p_apo, 1.0, -1.0)      # holo + / apo -
        cvals = np.clip(sign * pmax, -1.0, 1.0)           # −1..+1
    
        cmap, norm = _get_apo_holo_cmap(mid_frac=0.12)    # widen/lighten center as you like
        scatter_kwargs = dict(c=cvals, cmap=cmap, norm=norm)

    else:
        scatter_kwargs = dict(color="tab:gray")
    """

    # --- Color mapping (apo→blue, holo→red), intensity by max probability ---
    if color_by_maxP:
        pmax  = np.maximum(p_apo, p_holo)           # 0..1
        sign  = np.where(p_holo >= p_apo, 1.0, -1.0)
        cvals = np.clip(sign * pmax, -1.0, 1.0)     # −1..+1
        cmap  = "coolwarm"
        vmin, vmax = -1.0, 1.0
        cbar_label = "signed max pocket probability (apo→blue, holo→red)"
        scatter_kwargs = dict(c=cvals, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        scatter_kwargs = dict(color="tab:gray")
        cbar_label = None

    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    sc = ax.scatter(ratio, dP, s=8, alpha=0.35, rasterized=True, **scatter_kwargs)

    # zero lines
    ax.axhline(0, color="gray", lw=1, alpha=0.6)
    ax.axvline(0, color="gray", lw=1, alpha=0.6)

    # regression trend
    try:
        import scipy.stats as st
        slope, intercept, r_value, p_value, _ = st.linregress(ratio, dP)
        xline = np.linspace(ratio.min(), ratio.max(), 200)
        ax.plot(xline, intercept + slope * xline, color="black", lw=1.2)
        trend_note = f"r = {r_pearson:.3f}, p = {p_value:.2e}"
    except Exception:
        trend_note = f"r = {r_pearson:.3f}"

    ax.text(0.98, 0.05, trend_note, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))

    ax.set_xlabel("log₂(avg_n_holo / avg_n_apo)")
    ax.set_ylabel("ΔP = avg_P(holo) − avg_P(apo)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    # colorbar with helpful ticks
    if color_by_maxP:
        cb = fig.colorbar(sc, ax=ax)
        #cb.set_label("signed max pocket probability (apo→blue, holo→red)")
        cb.set_label(cbar_label)
        cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        cb.set_ticklabels(["apo 1.0", "apo 0.5", "tie", "holo 0.5", "holo 1.0"])

    fig.tight_layout()
    save_figure(fig, save_base, formats)



# 1) Add helper: _overlay_half_violins_ratio_delta(ax, ratio_vals, deltaP_vals, class_vals, ...)
from matplotlib.patches import Polygon

def _overlay_half_violins_ratio_delta(
    ax,
    ratio_vals,                      # np.ndarray OR pd.Series
    deltaP_vals,                     # np.ndarray OR pd.Series
    class_vals,                      # np.ndarray OR pd.Series (strings 'apo'/'holo')
    *,
    # layout & binning
    bins: int = 24,
    y_min: float = -1.0,
    y_max: float = 1.0,
    y_points: int = 201,
    # visuals
    max_violin_width: float = 0.35,
    color_apo: str = "#3B6AB8",
    color_holo: str = "#C43C3C",
    alpha: float = 0.60,
    zorder: float = 1.5,
    # smoothing
    smooth_sigma: float = 1.2,
    # placement
    placement: str = "flip_by_x",     # "centered" | "hinged_zero" | "flip_by_x"
    # tapering
    taper_to_axis: bool = True,
    taper_power: float = 1.0,
    preserve_area: bool = True,
    y_gap: float = 0.00
) -> None:
    """
    Overlay split half-violins per ratio bin on an existing Axes.

    Accepts numpy arrays or pandas Series. If Series are provided, they will be
    aligned by index (inner join) to avoid length mismatches.
    """

    # ---- Align inputs safely ----
    is_series = any(isinstance(x, pd.Series) for x in (ratio_vals, deltaP_vals, class_vals))
    if is_series:
        # Convert to Series with names, align on index
        r = ratio_vals if isinstance(ratio_vals, pd.Series) else pd.Series(ratio_vals)
        d = deltaP_vals if isinstance(deltaP_vals, pd.Series) else pd.Series(deltaP_vals)
        c = class_vals  if isinstance(class_vals,  pd.Series) else pd.Series(class_vals)
        dat = pd.concat({"ratio": r, "dP": d, "cls": c}, axis=1, join="inner")
        dat = dat.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio", "dP", "cls"])
        if dat.empty:
            return
        ratio = dat["ratio"].to_numpy(dtype=float)
        dP    = np.clip(dat["dP"].to_numpy(dtype=float), -1.0, 1.0)
        cls   = dat["cls"].astype(str).str.strip().str.lower().to_numpy()
    else:
        # NumPy inputs: require same length
        ratio_vals = np.asarray(ratio_vals)
        deltaP_vals = np.asarray(deltaP_vals)
        class_vals = np.asarray(class_vals)
        if not (len(ratio_vals) == len(deltaP_vals) == len(class_vals)):
            raise ValueError(
                "ratio_vals, deltaP_vals, class_vals must have the same length.\n"
                "Tip: pass pandas Series (with the same index) so the function can align them, "
                "or build a common boolean mask and apply it to all three before calling."
            )
        # finite mask and cleaning
        m = np.isfinite(ratio_vals) & np.isfinite(deltaP_vals) & (class_vals != "")
        if not np.any(m):
            return
        ratio = ratio_vals[m].astype(float)
        dP    = np.clip(deltaP_vals[m].astype(float), -1.0, 1.0)
        cls   = pd.Series(class_vals[m]).astype(str).str.strip().str.lower().to_numpy()

    # ---- x-bins ----
    x_edges = np.linspace(float(np.nanmin(ratio)), float(np.nanmax(ratio)), bins + 1)
    if not np.all(np.isfinite(x_edges)):
        return
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0

    # ---- y grids ----
    if placement in ("hinged_zero", "flip_by_x"):
        y_abs_grid = np.linspace(0.0, 1.0, y_points)
        y_spine_up =  y_abs_grid + y_gap
        y_spine_dn = -y_abs_grid - y_gap
    else:
        y_grid = np.linspace(y_min, y_max, y_points)

    # ---- smoothing ----
    def _gauss_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return arr
        rad = max(1, int(round(3 * sigma)))
        xk = np.arange(-rad, rad + 1)
        k = np.exp(-(xk**2) / (2.0 * sigma * sigma))
        s = k.sum()
        if s > 0:
            k /= s
        return np.convolve(arr, k, mode="same")

    # ---- per-bin drawing ----
    for i, xc in enumerate(x_centers):
        left, right = x_edges[i], x_edges[i + 1]
        in_bin = (ratio >= left) & (ratio < right)
        if not np.any(in_bin):
            continue

        in_apo  = in_bin & (cls == "apo")
        in_holo = in_bin & (cls == "holo")

        # side for flip_by_x: x<0 → above; x>=0 → below
        if placement == "flip_by_x":
            side_sign = +1.0 if (xc < 0) else -1.0

        # helper: compute width curve from |ΔP|
        def _width_from_abs_vals(values_abs01: np.ndarray) -> np.ndarray:
            hist, _ = np.histogram(values_abs01, bins=len(y_abs_grid) - 1, range=(0.0, 1.0), density=True)
            dens = _gauss_smooth(hist, smooth_sigma)
            maxd = np.nanmax(dens) if dens.size else 0.0
            if maxd > 0:
                dens = dens / maxd
            base_width = max_violin_width * dens
            if taper_to_axis:
                taper = (y_abs_grid[:-1] ** max(0.0, float(taper_power)))
                width = base_width * taper
                if preserve_area:
                    dy = np.diff(y_abs_grid)
                    area_base  = float(np.sum(base_width * dy))
                    area_taper = float(np.sum(width * dy))
                    if area_taper > 0:
                        width *= (area_base / area_taper)
            else:
                width = base_width
            return width

        # ---------- APO (left half) ----------
        if np.any(in_apo):
            if placement == "centered":
                hist, _ = np.histogram(dP[in_apo], bins=len(y_grid) - 1, range=(y_min, y_max), density=True)
                dens = _gauss_smooth(hist, smooth_sigma)
                maxd = np.nanmax(dens) if dens.size else 0.0
                if maxd > 0:
                    dens = dens / maxd
                width = max_violin_width * dens
                x_poly = np.concatenate([xc - width, np.full_like(width, xc)])
                y_poly = np.concatenate([y_grid[:-1], y_grid[-2::-1]])
            else:
                mags = np.abs(dP[in_apo])
                width = _width_from_abs_vals(mags)
                sgn = (+1.0 if placement == "hinged_zero" else side_sign)
                spine = y_spine_up if sgn > 0 else y_spine_dn
                x_poly = np.concatenate([xc - width, np.full_like(width, xc)])
                y_poly = np.concatenate([spine[:-1], spine[-2::-1]])
            ax.add_patch(Polygon(np.c_[x_poly, y_poly], closed=True,
                                 facecolor=color_apo, edgecolor="none",
                                 alpha=alpha, zorder=zorder))

        # ---------- HOLO (right half) ----------
        if np.any(in_holo):
            if placement == "centered":
                hist, _ = np.histogram(dP[in_holo], bins=len(y_grid) - 1, range=(y_min, y_max), density=True)
                dens = _gauss_smooth(hist, smooth_sigma)
                maxd = np.nanmax(dens) if dens.size else 0.0
                if maxd > 0:
                    dens = dens / maxd
                width = max_violin_width * dens
                x_poly = np.concatenate([np.full_like(width, xc), xc + width])
                y_poly = np.concatenate([y_grid[:-1], y_grid[-2::-1]])
            else:
                mags = np.abs(dP[in_holo])
                width = _width_from_abs_vals(mags)
                sgn = (-1.0 if placement == "hinged_zero" else side_sign)
                spine = y_spine_up if sgn > 0 else y_spine_dn
                x_poly = np.concatenate([np.full_like(width, xc), xc + width])
                y_poly = np.concatenate([spine[:-1], spine[-2::-1]])
            ax.add_patch(Polygon(np.c_[x_poly, y_poly], closed=True,
                                 facecolor=color_holo, edgecolor="none",
                                 alpha=alpha, zorder=zorder))


# 2) Patch _plot_binned_prob_vs_n_ratio to support optional violins
def _plot_binned_prob_vs_n_ratio(df, avg_apo_col, avg_holo_col,
                                 p_apo_col, p_holo_col,
                                 title, save_base=None, formats=("png",),
                                 bins=30, nan_as_zero=True,
                                 force_full_y=True,
                                 x_pad_frac=0.03, y_pad_frac=0.03,
                                 x_pad_abs=None, y_pad_abs=None,
                                 add_half_violins=True,          # NEW
                                 class_col="highest_p_mean_pocket",  # NEW
                                 violin_width_factor=0.45,        # NEW (fraction of bin width per half)
                                 violin_alpha=0.80, # 0.28
                                 show_avg_n_line=False,          # NEW ← secondary y-axis line
                                 return_summary=False,          # NEW ← return a tidy bin table
                                 ):

    a_apo  = _coerce_numeric(df[avg_apo_col],  nan_as_zero=nan_as_zero)
    a_holo = _coerce_numeric(df[avg_holo_col], nan_as_zero=nan_as_zero)
    p_apo  = _coerce_numeric(df[p_apo_col],    nan_as_zero=False)
    p_holo = _coerce_numeric(df[p_holo_col],   nan_as_zero=False)

    ratio = np.log2(a_holo / a_apo)
    dP    = np.clip(p_holo - p_apo, -1.0, 1.0)

    # NEW: per-UBS "average n"
    avg_n_ubs = 0.5 * (a_apo + a_holo)

    m = np.isfinite(ratio) & np.isfinite(dP) & np.isfinite(avg_n_ubs)  # ensure finite on all
    if not m.any():
        return
    ratio, dP, avg_n_ubs = ratio[m], dP[m], avg_n_ubs[m]

    #m = np.isfinite(ratio) & np.isfinite(dP)
    #if not m.any():
    #    return
    #ratio, dP = ratio[m], dP[m]

    # binning
    x_edges  = np.linspace(float(np.nanmin(ratio)), float(np.nanmax(ratio)), bins + 1)
    centers  = (x_edges[:-1] + x_edges[1:]) / 2
    digitized = np.digitize(ratio, x_edges) - 1

    means  = [np.nanmean(dP[digitized == i]) if np.any(digitized == i) else np.nan for i in range(len(centers))]
    counts = [np.sum(digitized == i) for i in range(len(centers))]
    
    # NEW: mean of average n per UBS in each bin
    avg_n_mean_per_bin = [
        np.nanmean(avg_n_ubs[digitized == i]) if np.any(digitized == i) else np.nan
        for i in range(len(centers))
    ]

    fig, ax = plt.subplots(figsize=(6.7, 4.3))

    # compute / keep x_edges, centers, etc. as you already do...
    bin_width = np.diff(x_edges).min() if len(x_edges) > 1 else 1.0
    max_half_width = violin_width_factor * bin_width

    if add_half_violins and (class_col in df.columns):
        # inside _plot_binned_prob_vs_n_ratio(...)
        # Suppose you’re inside _plot_binned_prob_vs_n_ratio and have df and class_col
        cls_series = df[class_col].astype(str)

        _overlay_half_violins_ratio_delta(
            ax,
            ratio_vals=pd.Series(ratio, index=df.index),     # align by df index
            deltaP_vals=pd.Series(dP,    index=df.index),
            class_vals=cls_series,                           # same index
            bins=bins,
            max_violin_width=max_half_width,
            placement="flip_by_x",
            taper_to_axis=True,
            preserve_area=True,
            alpha=violin_alpha,
            zorder=1.5
        )


    # then draw your mean dots (zorder higher)
    sc = ax.scatter(
        centers, means,
        s=_scale_sizes(counts, min_size=20, max_size=250),
        c=means, cmap="coolwarm", vmin=-1.0, vmax=1.0,
        alpha=0.95, rasterized=True, zorder=3
    )

    # NEW: secondary Y-axis showing mean "average n per UBS" per bin
    if show_avg_n_line:
        ax2 = ax.twinx()
        ax2.plot(centers, avg_n_mean_per_bin, marker="s", linewidth=1.2, alpha=0.9)
        ax2.set_ylabel("mean average n per UBS (per bin)")
        # make sure the second axis does not dominate the look
        ax2.tick_params(axis="y", labelsize=9)

    ax.axhline(0, color="gray", lw=1, alpha=0.6, zorder=2)
    ax.axvline(0, color="gray", lw=1, alpha=0.6, zorder=2)
    fig.colorbar(sc, ax=ax, label="mean ΔP (holo − apo)")

    ax.set_xlabel("log₂(avg_n_holo / avg_n_apo)")
    ax.set_ylabel("mean ΔP")
    ax.set_title(title)

    # padding (as you already have)
    x_min, x_max = float(x_edges[0]), float(x_edges[-1])
    x_span = max(x_max - x_min, 1e-9)
    ax.set_xlim(x_min - (x_pad_abs if x_pad_abs is not None else x_pad_frac * x_span),
                x_max + (x_pad_abs if x_pad_abs is not None else x_pad_frac * x_span))

    finite_means = np.asarray([m for m in means if np.isfinite(m)])
    if force_full_y:
        ax.set_ylim(-1.0 - (y_pad_abs if y_pad_abs is not None else y_pad_frac * 2.0),
                    1.0 + (y_pad_abs if y_pad_abs is not None else y_pad_frac * 2.0))
        ax.set_yticks(np.linspace(-1.0, 1.0, 9))
    else:
        if finite_means.size:
            lo, hi = float(finite_means.min()), float(finite_means.max())
            if lo == hi: lo, hi = lo - 0.02, hi + 0.02
        else:
            lo, hi = -1.0, 1.0
        span = hi - lo
        ax.set_ylim(lo - (y_pad_abs if y_pad_abs is not None else y_pad_frac * span),
                    hi + (y_pad_abs if y_pad_abs is not None else y_pad_frac * span))

    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    
    # Build a tidy summary table for downstream use
    summary = pd.DataFrame({
        "bin_left":   x_edges[:-1],
        "bin_right":  x_edges[1:],
        "bin_center": centers,
        "n_ubs":      counts,
        "mean_deltaP": means,
        "avg_n_mean":  avg_n_mean_per_bin,   # <-- the new metric you asked for
    })
    
    # Optional quick preview in logs (handy during tuning)
    # _print_preview(summary, max_rows=8)
    
    fig.tight_layout()
    save_figure(fig, save_base, formats)
    
    if return_summary:
        return summary

    save_figure(fig, save_base, formats)


def _diagnose_prob_ratio_balance(df,
                                 avg_apo_col, avg_holo_col,
                                 p_apo_col, p_holo_col,
                                 title, save_base=None, formats=("png",),
                                 bins=30, nan_as_zero=True):
    """Diagnostic plots: ΔP histogram, P(apo)+P(holo) check, and bin sample counts vs ΔP."""
    a_apo = _coerce_numeric(df[avg_apo_col], nan_as_zero=nan_as_zero)
    a_holo = _coerce_numeric(df[avg_holo_col], nan_as_zero=nan_as_zero)
    p_apo = _coerce_numeric(df[p_apo_col], nan_as_zero=False)
    p_holo = _coerce_numeric(df[p_holo_col], nan_as_zero=False)

    ratio = np.log2(a_holo / a_apo)
    dP = p_holo - p_apo
    Psum = p_apo + p_holo

    mask = np.isfinite(ratio) & np.isfinite(dP)
    ratio, dP, Psum = ratio[mask], dP[mask], Psum[mask]

    # ==== 1️⃣ Figure layout ====
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    fig.suptitle(title)
    plt.subplots_adjust(wspace=0.35)

    # ==== 2️⃣ ΔP histogram ====
    ax = axes[0]
    ax.hist(dP, bins=40, color="steelblue", alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("ΔP = P(holo) − P(apo)")
    ax.set_ylabel("Count")
    ax.set_title(f"ΔP distribution (min={np.nanmin(dP):.2f}, max={np.nanmax(dP):.2f})")

    # ==== 3️⃣ Complementarity check (Psum) ====
    ax = axes[1]
    ax.hist(Psum, bins=40, color="orange", alpha=0.8)
    ax.axvline(1.0, color="black", lw=1)
    mean_psum = np.nanmean(Psum)
    ax.set_xlabel("P(apo) + P(holo)")
    ax.set_title(f"Complementarity check (mean={mean_psum:.2f})")
    ax.grid(True, linestyle="--", alpha=0.3)

    # ==== 4️⃣ Bin means and sample counts ====
    bins_edges = np.linspace(np.nanmin(ratio), np.nanmax(ratio), bins + 1)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
    digitized = np.digitize(ratio, bins_edges) - 1

    means = [np.nanmean(dP[digitized == i]) for i in range(len(bin_centers))]
    counts = [np.sum(digitized == i) for i in range(len(bin_centers))]

    ax = axes[2]
    color = plt.cm.coolwarm((np.array(means) + 1) / 2)  # map -1→blue, +1→red
    ax.bar(bin_centers, counts, width=np.diff(bins_edges), color=color, alpha=0.8)
    ax.set_xlabel("log₂(avg_n_holo / avg_n_apo)")
    ax.set_ylabel("Bin count")
    ax.set_title("Bin occupancy (color = mean ΔP)")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, save_base, formats)


def filter_by_min_avg_n(df: pd.DataFrame,
                        avg_apo_col: str,
                        avg_holo_col: str,
                        *,
                        min_apo_avg_n: float = 0.0,
                        min_holo_avg_n: float = 0.0,
                        mode: str = "both",         # "both" (AND), "per_class", or "either"
                        class_col: str | None = None,
                        nan_as_zero: bool = False,  # if False, NaN fails the threshold check
                        verbose: bool = True,
                        return_mask: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """
    Filter rows by minimum average counts for apo/holo.

    Parameters
    ----------
    df : DataFrame
    avg_apo_col, avg_holo_col : column names with avg_n_apo / avg_n_holo
    min_apo_avg_n, min_holo_avg_n : thresholds to require for apo/holo
    mode :
        - "both": keep rows with (avg_apo >= min_apo_avg_n) AND (avg_holo >= min_holo_avg_n)
        - "per_class": keep apo rows if (avg_apo >= min_apo_avg_n); keep holo rows if (avg_holo >= min_holo_avg_n).
                       Requires class_col (values 'apo'/'holo').
        - "either": keep rows with (avg_apo >= min_apo_avg_n) OR (avg_holo >= min_holo_avg_n)
    class_col : classification column (needed for mode="per_class")
    nan_as_zero : if True, treats NaN as 0 before comparison; otherwise NaN fails the check
    verbose : print a small summary
    return_mask : if True, also return the boolean mask used

    Returns
    -------
    filtered_df  (and optionally mask)
    """
    # Coerce numeric using your existing helper if present; fallback if not.
    try:
        a_apo  = _coerce_numeric(df[avg_apo_col],  nan_as_zero=nan_as_zero)
        a_holo = _coerce_numeric(df[avg_holo_col], nan_as_zero=nan_as_zero)
    except NameError:
        a_apo  = pd.to_numeric(df[avg_apo_col],  errors="coerce")
        a_holo = pd.to_numeric(df[avg_holo_col], errors="coerce")
        if nan_as_zero:
            a_apo  = a_apo.fillna(0.0)
            a_holo = a_holo.fillna(0.0)

    # Base comparisons (NaN >= x is False when nan_as_zero=False)
    ok_apo  = a_apo  >= float(min_apo_avg_n)
    ok_holo = a_holo >= float(min_holo_avg_n)

    mode = str(mode).lower()
    if mode == "both":
        mask = ok_apo & ok_holo
    elif mode == "either":
        mask = ok_apo | ok_holo
    elif mode == "per_class":
        if class_col is None or class_col not in df.columns:
            raise ValueError("mode='per_class' requires a valid class_col with values 'apo'/'holo'.")
        cls = df[class_col].astype(str).str.strip().str.lower()
        mask = ( (cls == "apo")  & ok_apo ) | ( (cls == "holo") & ok_holo )
    else:
        raise ValueError("mode must be one of: 'both', 'either', 'per_class'")

    if verbose:
        n0 = len(df); n1 = int(mask.sum()); ndrop = n0 - n1
        # Reason breakdown (useful in 'both' mode)
        below_apo  = (~ok_apo) & ok_holo
        below_holo = ok_apo & (~ok_holo)
        below_both = (~ok_apo) & (~ok_holo)
        print(f"[filter_by_min_avg_n] mode={mode} | kept {n1:,} / {n0:,} (dropped {ndrop:,}) "
              f"| thresholds: apo≥{min_apo_avg_n}, holo≥{min_holo_avg_n}")
        if mode == "both":
            print(f"  - drop breakdown: below apo only={int(below_apo.sum()):,}, "
                  f"below holo only={int(below_holo.sum()):,}, "
                  f"below both={int(below_both.sum()):,}")

    out = df.loc[mask].copy()
    return (out, mask) if return_mask else out


# ==== Ridgeline helpers ====

def _gauss_smooth_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or arr.size == 0:
        return arr
    rad = max(1, int(round(3 * sigma)))
    xk = np.arange(-rad, rad + 1, dtype=float)
    k = np.exp(-(xk**2) / (2.0 * sigma * sigma))
    k /= k.sum()
    return np.convolve(arr, k, mode="same")


def _draw_split_violin_ridge(
    ax,
    *,
    y0: float,                       # vertical position of the ridge (e.g., the ratio-bin center)
    vals_left: np.ndarray,           # e.g., P(apo) in [0,1]
    vals_right: np.ndarray,          # e.g., P(holo) in [0,1]
    x_min: float = 0.0,
    x_max: float = 1.0,
    n_points: int = 201,
    max_half_height: float = 0.38,   # vertical amplitude for one half
    color_left: str = "#3B6AB8",     # apo (blue)
    color_right: str = "#C43C3C",    # holo (red)
    alpha: float = 0.60,
    edgecolor: str | None = None,
    smooth_sigma: float = 1.2,
):
    """
    Draw one ridgeline row at vertical position y0, with:
      - top half (upwards) for 'vals_right' (e.g., holo probabilities)
      - bottom half (downwards) for 'vals_left' (e.g., apo probabilities)
    Horizontal axis is probability in [x_min, x_max].
    """
    import numpy as np
    from matplotlib.patches import Polygon

    def _density(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            xg = np.linspace(x_min, x_max, n_points)
            return xg, np.zeros_like(xg)
        # histogram-as-density on a fixed grid, then smooth
        xg = np.linspace(x_min, x_max, n_points)
        hist, edges = np.histogram(vals, bins=n_points - 1, range=(x_min, x_max), density=True)
        dens = _gauss_smooth_1d(hist, smooth_sigma)
        # normalize to unit peak so half-height controls the amplitude
        if np.nanmax(dens) > 0:
            dens = dens / np.nanmax(dens)
        # put dens on x-grid (edges→centers)
        xc = 0.5 * (edges[:-1] + edges[1:])
        return xc, dens

    # left (apo)
    xl, dl = _density(vals_left)
    yl_top = y0 - (dl * max_half_height)
    # right (holo)
    xr, dr = _density(vals_right)
    yr_top = y0 + (dr * max_half_height)

    # Compose polygons: bottom→top, top→bottom to close.
    # APO (bottom half)
    if xl.size and np.any(dl > 0):
        x_poly = np.concatenate([xl, xl[::-1]])
        y_poly = np.concatenate([np.full_like(xl, y0), yl_top[::-1]])
        ax.add_patch(Polygon(np.c_[x_poly, y_poly], closed=True,
                             facecolor=color_left, edgecolor=edgecolor, alpha=alpha, zorder=1.5))
    # HOLO
    if xr.size and np.any(dr > 0):
        x_poly = np.concatenate([xr, xr[::-1]])
        y_poly = np.concatenate([np.full_like(xr, y0), yr_top[::-1]])
        ax.add_patch(Polygon(np.c_[x_poly, y_poly], closed=True,
                             facecolor=color_right, edgecolor=edgecolor, alpha=alpha, zorder=1.6))



def _draw_split_violin_ridge_same_top(
    ax,
    *,
    y0: float,
    vals_left: np.ndarray,
    vals_right: np.ndarray,
    x_min: float = 0.5,
    x_max: float = 1.0,
    n_points: int = 201,
    max_half_height: float = 0.38,
    color_left: str = "#4C78A8",
    color_right: str = "#F28E2B",
    alpha: float = 0.60,
    edgecolor: str | None = "#4A4A4A",
    edgewidth: float = 0.6,
    smooth_sigma: float = 1.2,
    # layering controls
    zbase: float = 10.0,
    zstep: float = 0.02,
    # gradient options
    use_gradient: bool = True,
    gradient_lighten: float = 0.75,  # 0..1, how much lighter the left side is
):
    """
    Draw a ridgeline row at vertical position y0 with *both* halves on the top side:
      - upward half for 'vals_left' (apo winners)
      - upward half for 'vals_right' (holo winners)
    Horizontal axis is probability in [x_min, x_max].

    Overlap is intentional; transparency separates the shapes visually.
    """    """
    Draw a ridgeline row at y0 with both halves on top (apo=left channel, holo=right channel).
    If use_gradient=True, fill is a left→right gradient clipped to each polygon.
    """
    from matplotlib.patches import Polygon, PathPatch
    from matplotlib.path import Path
    #import matplotlib.colors as mcolors

    def _gauss_smooth_1d(arr, sigma):
        if sigma <= 0 or arr.size == 0:
            return arr
        rad = max(1, int(round(3 * sigma)))
        xk = np.arange(-rad, rad + 1, dtype=float)
        k = np.exp(-(xk**2) / (2.0 * sigma * sigma))
        k /= k.sum()
        return np.convolve(arr, k, mode="same")

    def _density(vals: np.ndarray):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        xg = np.linspace(x_min, x_max, n_points)
        if vals.size == 0:
            return xg, np.zeros_like(xg)
        hist, edges = np.histogram(vals, bins=n_points - 1, range=(x_min, x_max), density=True)
        dens = _gauss_smooth_1d(hist, smooth_sigma)
        if np.nanmax(dens) > 0:
            dens = dens / np.nanmax(dens)
        xc = 0.5 * (edges[:-1] + edges[1:])
        return xc, dens

    # APO top half
    xl, dl = _density(vals_left)
    yl_top_left = y0 + (dl * max_half_height)
    # HOLO top half
    xr, dr = _density(vals_right)
    yl_top_right = y0 + (dr * max_half_height)

    def _draw_half(xv, ytop, base_color, zoff):
        if xv.size == 0 or not np.any(ytop > y0):
            return
        # Build polygon path
        x_poly = np.concatenate([xv, xv[::-1]])
        y_poly = np.concatenate([np.full_like(xv, y0), ytop[::-1]])
        path = Path(np.c_[x_poly, y_poly])
        # Gradient fill
        if use_gradient:
            cmap = _make_linear_cmap(base_color, lighten=gradient_lighten)
            # a 1x256 horizontal gradient stretched to [x_min, x_max] x [y0, y0+max_half_height]
            grad = np.linspace(0, 1, 256).reshape(1, -1)
            im = ax.imshow(
                grad,
                extent=(x_min, x_max, y0, y0 + max_half_height),
                origin="lower",
                aspect="auto",
                cmap=cmap,
                alpha=alpha,
                zorder=zbase + zoff + 0.001,
            )
            im.set_clip_path(PathPatch(path, transform=ax.transData))
        else:
            # Solid fill fallback
            ax.add_patch(Polygon(
                np.c_[x_poly, y_poly], closed=True,
                facecolor=base_color, edgecolor="none",
                alpha=alpha, zorder=zbase + zoff + 0.001
            ))
        # Outline on top
        ax.add_patch(Polygon(
            np.c_[x_poly, y_poly], closed=True,
            facecolor="none",
            edgecolor=edgecolor, linewidth=edgewidth,
            alpha=1.0, zorder=zbase + zoff + 0.002
        ))

    # Draw left (apo) and right (holo) halves with slight z offsets
    _draw_half(xl, yl_top_left, color_left, zoff=1 * zstep)
    _draw_half(xr, yl_top_right, color_right, zoff=2 * zstep)


def _apply_ridgeline_ypadding(ax, y_positions, max_half_height, pad_frac=0.08):
    """
    Ensure breathing room above and below the ridgeline stack.
    y_positions: array of ridge baselines (y0 values) in data coords
    max_half_height: the upward amplitude used for each ridge
    pad_frac: extra margin as a fraction of the stack height
    """
    import numpy as np
    if len(y_positions) == 0:
        return
    y0_min = float(np.min(y_positions))
    y0_max = float(np.max(y_positions))
    # ridges extend upward by max_half_height
    content_min = y0_min
    content_max = y0_max + float(max_half_height)
    span = max(1e-9, content_max - content_min)
    pad = span * float(pad_frac)
    ax.set_ylim(content_min - pad, content_max + pad)


def _plot_ridgeline_prob_vs_n_ratio(
    df: pd.DataFrame,
    avg_apo_col: str,
    avg_holo_col: str,
    p_apo_col: str,
    p_holo_col: str,
    *,
    title: str,
    save_base: str | None = None,
    formats: tuple[str, ...] = ("png",),
    bins: int = 20,
    nan_as_zero: bool = True,
    overlap: float = 0.35,                # fraction of vertical spacing to overlap consecutive ridges
    max_half_height: float = 0.42,        # vertical amplitude of each half within a ridge
    color_apo: str = "#4C78A8",    # NEW default (blue)
    color_holo: str = "#F28E2B",   # NEW default (orange)
    alpha: float = 0.55,           # slightly clearer overlap
    #color_apo: str = "#4C72B0",
    #color_holo: str = "#C44E52",
    #lpha: float = 0.60,
    edgecolor: str | None = "#4A4A4A",   # ← fine dark grey outline
    edgewidth: float = 0.6,              # ← thin line
    smooth_sigma: float = 1.2,
    show_medians: bool = True,
    # NEW:
    show_winshare: bool = True,
    winshare_x: float = 1.025,       # x position of the hairline
    winshare_height_scale: float = 0.90,  # fraction of max_half_height for hairline height
    winshare_width: float = 0.006,   # small rectangle width
    winshare_edgecolor: str | None = "#4A4A4A",   #str | None = None,
):
    """
    Ridgeline of half-violins per log2(avg_n_holo/avg_n_apo) bin.

    X axis: winning pocket probability in [0.5, 1.0] (low probs omitted).
    For each UBS:
      - if P_apo >= P_holo and P_apo >= 0.5 → include P_apo in APO distribution
      - if P_holo >  P_apo and P_holo >= 0.5 → include P_holo in HOLO distribution
    Both halves are drawn upwards (same-side) with transparency.

    Y axis: binned log2 ratio centers.
    """

    from matplotlib.patches import Rectangle

    a_apo  = _coerce_numeric(df[avg_apo_col],  nan_as_zero=nan_as_zero)
    a_holo = _coerce_numeric(df[avg_holo_col], nan_as_zero=nan_as_zero)
    p_apo  = _coerce_numeric(df[p_apo_col],    nan_as_zero=False)
    p_holo = _coerce_numeric(df[p_holo_col],   nan_as_zero=False)

    ratio = np.log2(a_holo / a_apo)

    m = np.isfinite(ratio) & np.isfinite(p_apo) & np.isfinite(p_holo)
    if not m.any():
        return

    ratio  = ratio[m]
    p_apo  = p_apo[m].to_numpy(float)
    p_holo = p_holo[m].to_numpy(float)

    # Clamp to [0.5, 1.0] later; first decide winners
    apo_winners_mask  = (p_apo >= p_holo) & (p_apo >= 0.5)
    holo_winners_mask = (p_holo >  p_apo) & (p_holo >= 0.5)

    # Bin along ratio
    x_edges   = np.linspace(float(np.nanmin(ratio)), float(np.nanmax(ratio)), max(int(bins), 2) + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    didx = np.digitize(ratio, x_edges, right=True) - 1

    per_bin = []
    for i, c in enumerate(x_centers):
        in_bin = (didx == i)
        # Collect only the *winning* values, clipped to [0.5, 1.0]
        v_apo  = np.clip(p_apo[in_bin & apo_winners_mask],  0.5, 1.0)
        v_holo = np.clip(p_holo[in_bin & holo_winners_mask], 0.5, 1.0)
        per_bin.append((c, v_apo, v_holo))

    # Sort by center so y axis reads in ascending ratio
    per_bin.sort(key=lambda t: t[0])
    n = len(per_bin)
    if n == 0:
        return

    # Layout with controlled overlap
    base_step = 1.0
    step = base_step * (1.0 - float(overlap))
    y_positions = np.arange(n, dtype=float) * step
    y_labels    = [f"{c:.2f}" for (c, _, _) in per_bin]

    fig, ax = plt.subplots(figsize=(7.6, 6.0))

    for y, (center, v_apo, v_holo) in zip(y_positions, per_bin):
        _draw_split_violin_ridge_same_top(
            ax,
            y0=y,
            vals_left=v_apo,    # apo winners
            vals_right=v_holo,  # holo winners
            x_min=0.5, x_max=1.0, n_points=201,
            max_half_height=max_half_height,
            color_left=color_apo, color_right=color_holo,
            alpha=alpha, smooth_sigma=smooth_sigma,
            edgecolor=edgecolor, edgewidth=edgewidth,        # ← explicit
            use_gradient=False, gradient_lighten=0,

        )
        if show_medians and v_apo.size:
            med = float(np.nanmedian(v_apo))
            ax.plot([med, med], [y, y + 0.06], lw=1.2, color=color_apo, alpha=0.95, zorder=3)
        if show_medians and v_holo.size:
            med = float(np.nanmedian(v_holo))
            ax.plot([med, med], [y, y + 0.06], lw=1.2, color=color_holo, alpha=0.95, zorder=3)

        # --- Win-share hairline (apo/holo share among winners in this bin) ---
        if show_winshare:
            na, nh = int(v_apo.size), int(v_holo.size)
            tot = na + nh
            if tot > 0:
                apo_frac  = na / tot
                holo_frac = nh / tot
                hair_h = max_half_height * float(winshare_height_scale)
                y0_h   = y  # base at ridge baseline

                # APO segment (bottom part of the hairline)
                ax.add_patch(Rectangle(
                    (winshare_x, y0_h),
                    winshare_width,
                    hair_h * apo_frac,
                    facecolor=color_apo,
                    edgecolor=winshare_edgecolor,
                    alpha=alpha,
                    zorder=2.2,
                )) if apo_frac > 0 else None

                # HOLO segment (stacked above APO)
                ax.add_patch(Rectangle(
                    (winshare_x, y0_h + hair_h * apo_frac),
                    winshare_width,
                    hair_h * holo_frac,
                    facecolor=color_holo,
                    edgecolor=winshare_edgecolor,
                    alpha=alpha,
                    zorder=2.21,
                )) if holo_frac > 0 else None

    # Cosmetics
    xmax_hair = max(1.04, float(winshare_x) + float(winshare_width) + 0.005)
    ax.set_xlim(0.48, xmax_hair)
    ax.set_xlabel("Winning pocket probability (≥ 0.5)   |   win-share hairline at right")

    #ax.set_xlim(0.48, 1.02)
    #ax.set_xlabel("Winning pocket probability (≥ 0.5)")
    ax.set_ylabel("log₂(avg_n_holo / avg_n_apo) — binned")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_title(title)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], lw=6, color=color_apo,  alpha=alpha, label="apo probability"),
        Line2D([], [], lw=6, color=color_holo, alpha=alpha, label="holo probability"),
    ]

    # Place legend in figure coords (top-right), so it overlays the title line area
    plt.legend(handles=handles,
                     loc="upper right",
                     bbox_to_anchor=(0.985, 0.985),   # near the top-right corner of the figure
                     bbox_transform=plt.gcf().transFigure,
                     framealpha=0.95, fontsize=9, borderaxespad=0.0)

    fig.tight_layout()
    save_figure(fig, save_base, formats)


def _plot_ridgeline_prob_fullrange_same_top(
    df: pd.DataFrame,
    avg_apo_col: str,
    avg_holo_col: str,
    p_apo_col: str,
    p_holo_col: str,
    *,
    title: str,
    save_base: str | None = None,
    formats: tuple[str, ...] = ("png",),
    bins: int = 20,
    nan_as_zero: bool = True,
    overlap: float = 0.35,
    max_half_height: float = 0.42,
    color_apo: str = "#4C78A8",    # NEW default (blue)
    color_holo: str = "#F28E2B",   # NEW default (orange)
    alpha: float = 0.55,
    #color_apo: str = "#4C72B0",
    #color_holo: str = "#C44E52",
    #alpha: float = 0.60,
    edgecolor: str | None = "#4A4A4A",   # ← fine dark grey outline
    edgewidth: float = 0.6,              # ← thin line
    smooth_sigma: float = 1.2,
    show_medians: bool = True,
    show_winshare: bool = True,
    winshare_x: float = 1.020, # 1.025,
    winshare_height_scale: float = 0.90,
    winshare_width: float = 0.012, #0.006,
    winshare_edgecolor: str | None = None,
    # legend layout
    #legend_inside: bool = True,
    legend_ncol: int = 2,  # apo left, holo right
    # right-side per-bin count labels
    show_bin_counts: bool = True,
    bin_count_mode: str = "total",      # "total" or "winners"
    bin_count_fmt: str = "n={n}",
    bin_count_color: str = "#444444",
    bin_count_size: float = 9.0,
    bin_count_offset: float = 0.010, # 0.010,       # push labels to the right of the current x_max
    # count label vertical placement
    bin_count_y_offset_frac: float = 0.26,          # fraction of max_half_height to nudge labels up
    align_counts_to_hairline_center: bool = False,  # if True, center labels on the hairline bar
    # NEW:
    winshare_minprob: float = 0.5,                  # only count winners with prob >= this
    winshare_mode: str = "winners",                 # "winners" (default) or "counts"
    # Anchor bin edges and drop bins with few samples
    center_bins_at: float | None = 0.0,             # e.g., 0.0 to force a bin centered at 1:1
    min_bin_count: int = 0,                         # drop bins with < this many samples
    min_bin_count_mode: str = "total",              # "total" or "class"


):
    """
    Ridgeline of half-violins per log2(avg_n_holo/avg_n_apo) bin, using the FULL probability range [0,1].
    Both halves (apo & holo) are drawn upwards with transparency.
    The hairline shows per-bin apo/holo COUNT share (not 'winners' here).
    """
    from matplotlib.patches import Rectangle

    a_apo  = _coerce_numeric(df[avg_apo_col],  nan_as_zero=nan_as_zero)
    a_holo = _coerce_numeric(df[avg_holo_col], nan_as_zero=nan_as_zero)
    p_apo  = _coerce_numeric(df[p_apo_col],    nan_as_zero=False)
    p_holo = _coerce_numeric(df[p_holo_col],   nan_as_zero=False)

    # --- choose ratio axis (log2 or linear) ---
    ratio = np.log2(a_holo / a_apo)
    m = np.isfinite(ratio) & np.isfinite(p_apo) & np.isfinite(p_holo)
    if not m.any():
        return
    ratio  = ratio[m].to_numpy(float)
    p_apo  = np.clip(p_apo[m].to_numpy(float),  0.0, 1.0)
    p_holo = np.clip(p_holo[m].to_numpy(float), 0.0, 1.0)

    # Bin along ratio
    rmin, rmax = float(np.nanmin(ratio)), float(np.nanmax(ratio))
    x_edges = _make_anchored_edges(rmin, rmax, max(int(bins), 2), center_bins_at)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    didx = np.digitize(ratio, x_edges, right=True) - 1
    #x_edges   = np.linspace(float(np.nanmin(ratio)), float(np.nanmax(ratio)), max(int(bins), 2) + 1)

    per_bin = []
    for i, c in enumerate(x_centers):
        in_bin = (didx == i)
        v_apo  = p_apo[in_bin]
        v_holo = p_holo[in_bin]
    
        # counts for labels
        n_total = int(np.sum(in_bin))
        if show_bin_counts and bin_count_mode.lower() == "winners":
            apo_wins  = (v_apo >= v_holo) & (v_apo >= float(winshare_minprob))
            holo_wins = (v_holo >  v_apo) & (v_holo >= float(winshare_minprob))
            n_label = int(np.sum(apo_wins) + np.sum(holo_wins))
        else:
            n_label = n_total
    
        # stash counts with the bin
        per_bin.append((c, v_apo, v_holo, n_label))

    # --- sort bins ---
    per_bin.sort(key=lambda t: t[0])
    n = len(per_bin)
    if n == 0:
        return

    # Prune sparse bins
    if min_bin_count > 0:
        per_bin = [(c, v_apo, v_holo, _) for (c, v_apo, v_holo, _) in per_bin
                   if (len(v_apo) + len(v_holo)) >= int(min_bin_count)]
    n = len(per_bin)
    if n == 0:
        return

    base_step = 1.0
    step = base_step * (1.0 - float(overlap))
    y_positions = np.arange(n, dtype=float) * step
    y_labels    = [f"{c:.2f}" for (c, _, _, _) in per_bin]
    #y_labels    = [f"{c:.2f}" for (c, _, _) in per_bin]

    fig, ax = plt.subplots(figsize=(7.6, 6.0))

    for y, (center, v_apo, v_holo, _) in zip(y_positions, per_bin):
        _draw_split_violin_ridge_same_top(
            ax,
            y0=y,
            vals_left=v_apo,
            vals_right=v_holo,
            x_min=0.0, x_max=1.0, n_points=201,
            max_half_height=max_half_height,
            color_left=color_apo, color_right=color_holo,
            alpha=alpha, smooth_sigma=smooth_sigma,
            edgecolor=edgecolor, edgewidth=edgewidth,        # ← explicit
            use_gradient=False, gradient_lighten=0,
        )
        if show_medians and v_apo.size:
            med = float(np.nanmedian(v_apo))
            ax.plot([med, med], [y, y + 0.06], lw=1.2, color=color_apo, alpha=0.95, zorder=3)
        if show_medians and v_holo.size:
            med = float(np.nanmedian(v_holo))
            ax.plot([med, med], [y, y + 0.06], lw=1.2, color=color_holo, alpha=0.95, zorder=3)

        # Per-bin COUNT share hairline (apo/holo)
        # Per-bin win-share hairline (apo vs holo)
        if show_winshare:
            if winshare_mode == "winners":
                # Count only *winners* among aligned pairs in this bin
                # Winner logic: apo wins when v_apo >= v_holo (and >= minprob),
                #               holo wins when v_holo >  v_apo (and >= minprob).
                apo_wins  = (v_apo >= v_holo) & (v_apo >= float(winshare_minprob))
                holo_wins = (v_holo >  v_apo) & (v_holo >= float(winshare_minprob))
                na, nh = int(np.sum(apo_wins)), int(np.sum(holo_wins))
            else:
                # Fallback: raw counts (previous behavior)
                na, nh = int(v_apo.size), int(v_holo.size)

            tot = na + nh
            if tot > 0:
                apo_frac  = na / tot
                holo_frac = nh / tot
                hair_h = max_half_height * float(winshare_height_scale)
                y0_h   = y

                if apo_frac > 0:
                    ax.add_patch(Rectangle(
                        (winshare_x, y0_h),
                        winshare_width,
                        hair_h * apo_frac,
                        facecolor=color_apo,
                        edgecolor=winshare_edgecolor,
                        alpha=alpha,
                        zorder=2.2,
                    ))
                if holo_frac > 0:
                    ax.add_patch(Rectangle(
                        (winshare_x, y0_h + hair_h * apo_frac),
                        winshare_width,
                        hair_h * holo_frac,
                        facecolor=color_holo,
                        edgecolor=winshare_edgecolor,
                        alpha=alpha,
                        zorder=2.21,

                    ))

    # 1. set the x-limits first (hairline included)
    xmax_hair = max(1.04, float(winshare_x) + float(winshare_width) + 0.015)  # previously + 0.005
    #ax.set_xlim(0.48, xmax_hair)
    ax.set_xlim(-0.02, xmax_hair)

    # ----- right-side per-bin counts -----
    if show_bin_counts and len(per_bin) > 0:
        # 2. now compute text anchor based on the *final* limits
        # place labels just to the right of the current x max (hairline)
        x_left, x_right = ax.get_xlim()
        x_span = max(x_right - x_left, 1e-9)
        x_text = x_right + x_span * float(bin_count_offset)

        # monospaced for alignment
        txt_kw = dict(
            ha="left", va="center",
            fontsize=bin_count_size, color=bin_count_color,
            family="DejaVu Sans Mono",
            alpha=0.9, zorder=90,
        )

        # compute width hint so numbers align nicely
        max_n = max(row[3] for row in per_bin)
        width_hint = len(str(max_n))

        for y, (center, v_apo, v_holo, n_lbl) in zip(y_positions, per_bin):
            # default: small upward nudge
            y_text = y + max_half_height * float(bin_count_y_offset_frac)
        
            # optional: align to hairline center (uses the same height scale as the hairline)
            if align_counts_to_hairline_center:
                hair_h = max_half_height * float(winshare_height_scale)
                y_text = y + hair_h * 0.5
            ax.text(x_text, y_text, f"{n_lbl:>{width_hint}d}", **txt_kw)

        #for y, (_, _, _, n_lbl) in zip(y_positions, per_bin):
        #    ax.text(x_text, y, f"{n_lbl:>{width_hint}d}", **txt_kw)

        # extend xlim to include labels comfortably
        ax.set_xlim(x_left, x_text + x_span * 0.06)


    # Add y padding
    #_apply_ridgeline_ypadding(ax, y_positions, max_half_height, pad_frac=0.05)
    # add a bit of extra headroom for the title, keep bottom fixed
    ymin, ymax = ax.get_ylim()
    top_pad = (ymax - ymin) * 0.04   # tweak 0.02–0.08 as needed
    ax.set_ylim(ymin, ymax + top_pad)



    ax.set_xlabel("Pocket probability (apo & holo; full range 0–1)   |   per-bin count-share hairline at right")
    ax.set_ylabel("log₂(avg_n_holo / avg_n_apo) — binned")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    #ax.set_title(title, loc="left")
    ax.set_title(title)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], lw=6, color=color_apo,  alpha=alpha, label="apo probability"),
        Line2D([], [], lw=6, color=color_holo, alpha=alpha, label="holo probability"),
    ]

    # Legend inside
    #ax.legend(handles=handles, loc="upper right", framealpha=0.95, fontsize=9)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),   # centered at the top inside axes
        ncol=int(legend_ncol), framealpha=0.95, fontsize=9, borderaxespad=0.4, borderpad=0.7, handlelength=1.8, columnspacing=1.2,)

    # Place legend in figure coords (top-right), so it overlays the title line area
    #plt.legend(handles=handles,
    #                 loc="upper right",
    #                 bbox_to_anchor=(0.985, 0.985),   # near the top-right corner of the figure
    #                 bbox_transform=plt.gcf().transFigure,
    #                 framealpha=0.95, fontsize=9, borderaxespad=0.0)

    fig.tight_layout()

    save_figure(fig, save_base, formats)


def _summarize_ridgeline_bins(
    df: pd.DataFrame,
    avg_apo_col: str,
    avg_holo_col: str,
    p_apo_col: str,
    p_holo_col: str,
    *,
    bins: int = 20,
    nan_as_zero: bool = True,
    winshare_minprob: float = 0.5,       # threshold to count winners
    winshare_mode: str = "winners",      # "winners" (default) or "counts"
    save_base: str | None = None,        # if provided, write CSV next to plots
    csv_suffix: str = "binsummary",
    center_bins_at: float | None = 0.0,   # e.g., 0.0 to force a bin centered at 1:1
) -> pd.DataFrame:
    """
    Build an Excel-friendly per-bin table for the ridgeline binning over log2(avg_n_holo/avg_n_apo).

    Returns a DataFrame with one row per bin plus an OVERALL row:
      - bin_index, bin_left_edge, bin_right_edge, ratio_center_log2
      - n_rows (UBSs in bin)
      - n_apo_winners, n_holo_winners, share_apo_winners, share_holo_winners
      - median/mean of P(apo) and P(holo) in the bin

    winshare_mode:
      - "winners": counts only the side that *wins* per UBS if its prob >= winshare_minprob
      - "counts":  raw counts (apo/holo arrays) ignoring winner logic
    """
    a_apo  = _coerce_numeric(df[avg_apo_col],  nan_as_zero=nan_as_zero)
    a_holo = _coerce_numeric(df[avg_holo_col], nan_as_zero=nan_as_zero)
    p_apo  = _coerce_numeric(df[p_apo_col],    nan_as_zero=False)
    p_holo = _coerce_numeric(df[p_holo_col],   nan_as_zero=False)

    ratio = np.log2(a_holo / a_apo)
    m = np.isfinite(ratio) & np.isfinite(p_apo) & np.isfinite(p_holo)
    if not m.any():
        empty = pd.DataFrame(columns=[
            "bin_index","bin_left_edge","bin_right_edge","ratio_center_log2",
            "n_rows","n_apo_winners","n_holo_winners","share_apo_winners","share_holo_winners",
            "median_p_apo","median_p_holo","mean_p_apo","mean_p_holo",
            "mean_avg_n_apo","mean_avg_n_holo","mean_avg_n_overall"
        ])
        if save_base:
            csv_path = f"{save_base}_{csv_suffix}.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            empty.to_csv(csv_path, index=False)
        return empty

    ratio  = ratio[m].to_numpy(float)
    p_apo  = np.clip(p_apo[m].to_numpy(float),  0.0, 1.0)
    p_holo = np.clip(p_holo[m].to_numpy(float), 0.0, 1.0)

    # Build equal-width bins over observed ratio range
    a_apo  = a_apo[m].to_numpy(float)
    a_holo = a_holo[m].to_numpy(float)

    # Bin along ratio
    rmin, rmax = float(np.nanmin(ratio)), float(np.nanmax(ratio))
    x_edges = _make_anchored_edges(rmin, rmax, max(int(bins), 2), center_bins_at)
    #x_edges   = np.linspace(float(np.nanmin(ratio)), float(np.nanmax(ratio)), max(int(bins), 2) + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    didx = np.digitize(ratio, x_edges, right=True) - 1

    rows = []
    for i, (left, right, center) in enumerate(zip(x_edges[:-1], x_edges[1:], x_centers)):
        in_bin = (didx == i)
        n_rows = int(np.sum(in_bin))
        if n_rows == 0:
            rows.append({
                "bin_index": i,
                "bin_left_edge": float(left),
                "bin_right_edge": float(right),
                "ratio_center_log2": float(center),
                "n_rows": 0,
                "n_apo_winners": 0,
                "n_holo_winners": 0,
                "share_apo_winners": np.nan,
                "share_holo_winners": np.nan,
                "median_p_apo": np.nan,
                "median_p_holo": np.nan,
                "mean_p_apo": np.nan,
                "mean_p_holo": np.nan,
                "mean_avg_n_apo": np.nan,
                "mean_avg_n_holo": np.nan,
                "mean_avg_n_overall": np.nan,
            })
            continue

        v_apo  = p_apo[in_bin]
        v_holo = p_holo[in_bin]

        # Winner-aware (aligned) counting
        if winshare_mode == "winners":
            apo_wins  = (v_apo >= v_holo) & (v_apo >= float(winshare_minprob))
            holo_wins = (v_holo >  v_apo) & (v_holo >= float(winshare_minprob))
            n_apo_winners  = int(np.sum(apo_wins))
            n_holo_winners = int(np.sum(holo_wins))
        else:
            n_apo_winners  = int(v_apo.size)
            n_holo_winners = int(v_holo.size)

        tot_winners = n_apo_winners + n_holo_winners
        share_apo = (n_apo_winners / tot_winners) if tot_winners > 0 else np.nan
        share_holo = (n_holo_winners / tot_winners) if tot_winners > 0 else np.nan

        # Mean average n per UBS in this bin
        mean_avg_n_apo_bin  = float(np.nanmean(a_apo[in_bin])) if n_rows else np.nan
        mean_avg_n_holo_bin = float(np.nanmean(a_holo[in_bin])) if n_rows else np.nan
        mean_avg_n_overall_bin = float(0.5 * (mean_avg_n_apo_bin + mean_avg_n_holo_bin)) if n_rows else np.nan
        rows.append({
            "bin_index": i,
            "bin_left_edge": float(left),
            "bin_right_edge": float(right),
            "ratio_center_log2": float(center),
            "n_rows": n_rows,
            "n_apo_winners": n_apo_winners,
            "n_holo_winners": n_holo_winners,
            "share_apo_winners": share_apo,
            "share_holo_winners": share_holo,
            "median_p_apo": float(np.nanmedian(v_apo)) if v_apo.size else np.nan,
            "median_p_holo": float(np.nanmedian(v_holo)) if v_holo.size else np.nan,
            "mean_p_apo": float(np.nanmean(v_apo)) if v_apo.size else np.nan,
            "mean_p_holo": float(np.nanmean(v_holo)) if v_holo.size else np.nan,
            "mean_avg_n_apo": mean_avg_n_apo_bin,
            "mean_avg_n_holo": mean_avg_n_holo_bin,
            "mean_avg_n_overall": mean_avg_n_overall_bin,
        })

    bin_df = pd.DataFrame(rows)

    # OVERALL row
    # Overall averages of average n across all UBSs considered
    overall_mean_avg_n_apo  = float(np.nanmean(a_apo)) if a_apo.size else np.nan
    overall_mean_avg_n_holo = float(np.nanmean(a_holo)) if a_holo.size else np.nan
    overall_mean_avg_n_overall = float(0.5 * (overall_mean_avg_n_apo + overall_mean_avg_n_holo)) if a_apo.size and a_holo.size else np.nan
    overall = {
        "bin_index": "OVERALL",
        "bin_left_edge": np.nan,
        "bin_right_edge": np.nan,
        "ratio_center_log2": np.nan,
        "n_rows": int(bin_df["n_rows"].sum()),
        "n_apo_winners": int(bin_df["n_apo_winners"].sum()),
        "n_holo_winners": int(bin_df["n_holo_winners"].sum()),
        "share_apo_winners": (
            bin_df["n_apo_winners"].sum() / (bin_df["n_apo_winners"].sum() + bin_df["n_holo_winners"].sum())
            if (bin_df["n_apo_winners"].sum() + bin_df["n_holo_winners"].sum()) > 0 else np.nan
        ),
        "share_holo_winners": (
            bin_df["n_holo_winners"].sum() / (bin_df["n_apo_winners"].sum() + bin_df["n_holo_winners"].sum())
            if (bin_df["n_apo_winners"].sum() + bin_df["n_holo_winners"].sum()) > 0 else np.nan
        ),
        "median_p_apo": float(np.nanmedian(bin_df["median_p_apo"])) if bin_df["median_p_apo"].notna().any() else np.nan,
        "median_p_holo": float(np.nanmedian(bin_df["median_p_holo"])) if bin_df["median_p_holo"].notna().any() else np.nan,
        "mean_p_apo": float(np.nanmean(bin_df["mean_p_apo"])) if bin_df["mean_p_apo"].notna().any() else np.nan,
        "mean_p_holo": float(np.nanmean(bin_df["mean_p_holo"])) if bin_df["mean_p_holo"].notna().any() else np.nan,
        "mean_avg_n_apo": overall_mean_avg_n_apo,
        "mean_avg_n_holo": overall_mean_avg_n_holo,
        "mean_avg_n_overall": overall_mean_avg_n_overall,
    }
    bin_df = pd.concat([bin_df, pd.DataFrame([overall])], ignore_index=True)

    if save_base:
        csv_path = f"{save_base}_{csv_suffix}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        bin_df.to_csv(csv_path, index=False)

    return bin_df


def _plot_scatter_ratio_unbinned_probs(
    df: pd.DataFrame,
    avg_apo_col: str,
    avg_holo_col: str,
    p_apo_col: str,
    p_holo_col: str,
    *,
    title: str,
    save_base: str | None = None,
    formats: tuple[str, ...] = ("png",),
    color_apo: str = "#4C78A8",
    color_holo: str = "#F28E2B",
    alpha_pts: float = 0.45,
    size_pts: float = 10.0,
    x_jitter: float = 0.0,        # optional small jitter along x to break ties
    draw_vline_zero: bool = True,
):
    """
    Unbinned scatter: each UBS contributes two points at the same x (ratio),
    with y = 100*P(apo) and y = 100*P(holo).

    x = log2((avg_n_holo+1)/(avg_n_apo+1))
    y in [0, 100], one point per class.
    """
    a_apo  = _coerce_numeric(df[avg_apo_col],  nan_as_zero=True)
    a_holo = _coerce_numeric(df[avg_holo_col], nan_as_zero=True)
    p_apo  = _coerce_numeric(df[p_apo_col],    nan_as_zero=False)
    p_holo = _coerce_numeric(df[p_holo_col],   nan_as_zero=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log2(a_holo / a_apo)

    p_apo  = np.clip(p_apo,  0.0, 1.0)
    p_holo = np.clip(p_holo, 0.0, 1.0)

    m = np.isfinite(ratio) & np.isfinite(p_apo) & np.isfinite(p_holo)
    if not np.any(m):
        return

    ratio  = ratio[m].astype(float)
    y_apo  = (100.0 * p_apo[m]).astype(float)
    y_holo = (100.0 * p_holo[m]).astype(float)

    # Optional tiny jitter to reduce overplot (does not change values)
    if x_jitter and x_jitter > 0:
        rng = np.random.default_rng(12345)
        jitter = rng.normal(0.0, x_jitter, size=ratio.size)
    else:
        jitter = 0.0

    x_apo  = ratio + jitter
    x_holo = ratio + jitter

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.scatter(x_apo,  y_apo,  s=size_pts, alpha=alpha_pts, c=color_apo,  edgecolors="none", label="P(apo)",  zorder=1.1)
    ax.scatter(x_holo, y_holo, s=size_pts, alpha=alpha_pts, c=color_holo, edgecolors="none", label="P(holo)", zorder=1.2)

    # Cosmetics
    ax.set_xlabel("log₂(avg_n_holo / avg_n_apo)")
    ax.set_ylabel("probability (%)")
    ax.set_ylim(-2, 102)
    if draw_vline_zero:
        ax.axvline(0.0, color="#888888", lw=1.0, ls="--", alpha=0.7, zorder=0.9)
    ax.grid(True, axis="both", linestyle="--", alpha=0.30)
    ax.set_title(title, loc="left")

    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    fig.tight_layout()
    save_figure(fig, save_base, formats)


def _plot_ridgeline_prob_singleclass(
    df: pd.DataFrame,
    avg_apo_col: str,
    avg_holo_col: str,
    p_apo_col: str,
    p_holo_col: str,
    *,
    class_choice: str = "apo",          # "apo" or "holo"
    probability_mode: str = "winners",  # "winners" (default) or "full"
    min_winner_prob: float = 0.5,       # only used if probability_mode == "winners"
    title: str = "",
    save_base: str | None = None,
    formats: tuple[str, ...] = ("png",),
    bins: int = 20,
    drop_edge_bins: int = 0,            # number of bins to trim from each side (default 0)
    nan_as_zero: bool = True,
    # layout
    vertical_spacing: float = 0.65,     # 0..1 (smaller => tighter ridge packing)
    max_half_height: float = 0.42,      # keep the per-ridge amplitude independent of spacing
    # style
    color_apo: str = "#4C78A8",
    color_holo: str = "#F28E2B",
    alpha: float = 0.55,
    edgecolor: str | None = "#4A4A4A",
    edgewidth: float = 0.6,
    smooth_sigma: float = 1.2,
    show_median: bool = False,
    x_mode: str = "auto",               # "auto" => [0.5,1.0] for winners, [0,1] for full
    ridge_draw_priority: str = "top",  # "top" → highest (topmost) ridge drawn last and on top; "bottom" → default
    use_gradient=False,
    gradient_lighten=0.6,  # less bright at low x
    # right-side per-bin count labels
    show_bin_counts: bool = True,
    bin_count_mode: str = "total",     # "total" or "class"
    bin_count_fmt: str = "n={n}",      # e.g., "{n} UBS" or "n={n}"
    bin_count_color: str = "#444444",
    bin_count_size: float = 9.0,
    bin_count_offset: float = 0.03,    # fraction of x-span to push text to the right of x_max
    # Anchor bin edges and drop bins with few samples
    center_bins_at: float | None = 0.0,   # e.g., 0.0 to force a bin centered at 1:1
    min_bin_count: int = 0,               # drop bins with < this many samples
    min_bin_count_mode: str = "total",    # "total" or "class"

):
    """
    Ridgeline of a single class (apo or holo) per log2(avg_n_holo/avg_n_apo) bin.
    Each ridge draws one upward half-violin for the chosen class.

    probability_mode:
      - "winners": include the chosen class' probabilities only where it wins and >= min_winner_prob
      - "full":    include the chosen class' probabilities irrespective of winning side (clipped to [0,1])

    vertical_spacing controls how tightly ridges are stacked (independent of their height).
    """
    # --- prep ---
    cls = class_choice.strip().lower()
    if cls not in {"apo", "holo"}:
        raise ValueError("class_choice must be 'apo' or 'holo'")

    a_apo  = _coerce_numeric(df[avg_apo_col],  nan_as_zero=nan_as_zero)
    a_holo = _coerce_numeric(df[avg_holo_col], nan_as_zero=nan_as_zero)
    p_apo  = _coerce_numeric(df[p_apo_col],    nan_as_zero=False)
    p_holo = _coerce_numeric(df[p_holo_col],   nan_as_zero=False)

    ratio = np.log2(a_holo / a_apo)
    m = np.isfinite(ratio) & np.isfinite(p_apo) & np.isfinite(p_holo)
    if not m.any():
        return

    ratio  = ratio[m].to_numpy(float)
    p_apo  = np.clip(p_apo[m].to_numpy(float),  0.0, 1.0)
    p_holo = np.clip(p_holo[m].to_numpy(float), 0.0, 1.0)

    # --- bin on ratio ---
    rmin, rmax = float(np.nanmin(ratio)), float(np.nanmax(ratio))
    x_edges = _make_anchored_edges(rmin, rmax, max(int(bins), 2), center_bins_at)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    didx = np.digitize(ratio, x_edges, right=True) - 1
    #x_edges   = np.linspace(float(np.nanmin(ratio)), float(np.nanmax(ratio)), max(int(bins), 2)+1)

    if probability_mode == "winners":
        apo_wins  = (p_apo >= p_holo) & (p_apo >= float(min_winner_prob))
        holo_wins = (p_holo >  p_apo) & (p_holo >= float(min_winner_prob))
        if cls == "apo":
            values = np.where(apo_wins, p_apo, np.nan)
            x_min, x_max = 0.5, 1.0 if (x_mode == "auto") else (0.0, 1.0)
        else:
            values = np.where(holo_wins, p_holo, np.nan)
            x_min, x_max = 0.5, 1.0 if (x_mode == "auto") else (0.0, 1.0)
    else:
        # full: take all probs for that class, regardless of “winner”
        values = p_apo if cls == "apo" else p_holo
        x_min, x_max = (0.0, 1.0) if (x_mode == "auto") else (0.5, 1.0)

    # collect per bin
    per_bin_vals = []   # list of tuples: (center, v_values, n_total, n_class)
    for i, c in enumerate(x_centers):
        in_bin = (didx == i)
        n_total = int(np.sum(in_bin))  # all UBS landing in this ratio bin
        v = values[in_bin]
        v = v[np.isfinite(v)]
        n_class = int(v.size)          # values used for this ridge (depends on probability_mode/class_choice)
        per_bin_vals.append((c, v, n_total, n_class))
    #    in_bin = (didx == i)
    #    v = values[in_bin]
    #    v = v[np.isfinite(v)]
    #    per_bin_vals.append((c, v))

    # Optionally drop first and last few bins to zoom in
    if drop_edge_bins > 0 and len(per_bin_vals) > 2 * drop_edge_bins:
        per_bin_vals = per_bin_vals[drop_edge_bins: -drop_edge_bins]

    # --- sort bins ---
    per_bin_vals.sort(key=lambda t: t[0])
    n = len(per_bin_vals)
    if n == 0:
        return

    # --- prune sparse bins ---
    if min_bin_count > 0:
        use_class = (min_bin_count_mode.lower() == "class")
        per_bin_vals = [row for row in per_bin_vals
            if (row[3] if use_class else row[2]) >= int(min_bin_count)]
    n = len(per_bin_vals)
    if n == 0:
        return

    # --- vertical layout with explicit spacing ---
    step = max(0.05, float(vertical_spacing))   # keep sane
    y_positions = np.arange(n, dtype=float) * step
    y_labels    = [f"{row[0]:.2f}" for row in per_bin_vals]
    #y_labels    = [f"{c:.2f}" for (c, _) in per_bin_vals]

    # Determine draw order:
    # per_bin_vals is sorted by increasing ratio center (lowest at index 0).
    # y_positions increases with index, so the highest ridge is the last one.
    if ridge_draw_priority.lower() == "top":
        draw_indices = list(range(n-1, -1, -1))  # draw topmost last → appears on top
    else:
        draw_indices = list(range(n))            # original order (bottom-first)

    # z-order ladder: ensure later-drawn ridges can sit above earlier ones if needed
    # We still provide a gentle increasing zbase so even if order changes, layering is deterministic.
    z_start = 10.0
    z_delta = 0.20
    zstep = 0.02

    fig, ax = plt.subplots(figsize=(7.6, 6.0))

    for k, i in enumerate(draw_indices):
        y = y_positions[i]
        center, v, n_total, n_class = per_bin_vals[i]
        #center, v = per_bin_vals[i]
        zbase = z_start + k * z_delta  # each subsequent ridge sits slightly higher in z-order

        if cls == "apo":
            """
            _draw_split_violin_ridge_same_top(
                ax, y0=y, vals_left=v, vals_right=np.array([]),
                x_min=x_min, x_max=x_max, n_points=201,
                max_half_height=max_half_height,
                color_left=color_apo, color_right=color_holo,
                alpha=alpha, smooth_sigma=smooth_sigma,
                edgecolor=edgecolor, edgewidth=edgewidth,
                zbase=zbase, zstep=0.02, use_gradient=use_gradient,
                gradient_lighten=gradient_lighten,
            )
            """
            _draw_single_upward_violin_ridge(
                ax, y0=y, values=v,
                x_min=x_min, x_max=x_max,    # or 0.0..1.0 for full-range mode
                height=max_half_height,
                color=color_apo, alpha=alpha, edgecolor=edgecolor, edgewidth=edgewidth,
                mode="kde", bw_scale=1.0  # <- switch to KDE
            )

        else:
            _draw_split_violin_ridge_same_top(
                ax, y0=y, vals_left=np.array([]), vals_right=v,
                x_min=x_min, x_max=x_max, n_points=201,
                max_half_height=max_half_height,
                color_left=color_apo, color_right=color_holo,
                alpha=alpha, smooth_sigma=smooth_sigma,
                edgecolor=edgecolor, edgewidth=edgewidth,
                zbase=zbase, zstep=0.02, use_gradient=use_gradient,
                gradient_lighten=gradient_lighten,
            )

        if show_median and v.size:
            med = float(np.nanmedian(v))
            ax.plot([med, med], [y, y + 0.06],
                    lw=1.2,
                    color=(color_apo if cls == "apo" else color_holo),
                    alpha=0.95,
                    zorder=zbase + 3*zstep)  # keep medians above the fill

    # ----- right-side per-bin count labels -----
    # ----- lightweight right-side bin counts (publication friendly) -----
    if show_bin_counts and len(per_bin_vals) > 0:
        def _pick_n(n_total, n_class):
            return n_class if bin_count_mode.lower() == "class" else n_total
        # Decide horizontal placement: small shift to the right of the ridge end
        x_span = max(x_max - x_min, 1e-9)
        x_text = x_max + x_span * float(bin_count_offset)
        # Determine max count for alignment and label width
        all_counts = [_pick_n(r[2], r[3]) for r in per_bin_vals]
        max_n = max(all_counts) if all_counts else 0
        width_hint = len(str(max_n))
        # Style
        txt_kw = dict(
            ha="left", va="center",
            fontsize=9.5, color="#444444",
            family="DejaVu Sans Mono",
            alpha=0.9, zorder=90,)
        # Optional header
        #header_text = "UBS\nper bin" if bin_count_mode.lower() == "total" else "winners"
        #ax.text(
        #    x_text, y_positions[-1] + max_half_height * 1.5,
        #   header_text,
        #    ha="left", va="bottom",
        #    fontsize=10, color="#222222", fontweight="bold", alpha=0.9,)
        # Draw each label aligned vertically with its ridge
        for y, row in zip(y_positions, per_bin_vals):
            _, _, n_total, n_class = row
            n = _pick_n(n_total, n_class)
            label = f"{n:>{width_hint}d}"
            ax.text(x_text, y, label, **txt_kw)
        # Extend x-limits so the labels are fully visible
        ax.set_xlim(x_min - 0.02, x_text + x_span * 0.08)
    """
    if show_bin_counts and len(per_bin_vals) > 0:
        # decide which number to show
        def _pick_n(n_total, n_class):
            if bin_count_mode.lower() == "class":
                return n_class
            return n_total
        # place the text just to the right of x_max
        x_span = (max(x_max, x_min) - min(x_min, x_max))
        x_text = x_max + x_span * float(bin_count_offset)
        for y, (_, _, n_total, n_class) in zip(y_positions, per_bin_vals):
            n = _pick_n(n_total, n_class)
            ax.text(
                x_text, y + max_half_height * 0.02,  # tiny lift so it doesn't sit exactly on the baseline
                bin_count_fmt.format(n=n),
                ha="left", va="center",
                color=bin_count_color, fontsize=bin_count_size,
                zorder=100.0,  # above fills and outlines
            )
        # expand x-limits to include the labels comfortably
        ax.set_xlim(x_min - 0.02, x_text + x_span * 0.08)
    """

    # --- ensure a bit of vertical padding so top ridges don't touch plot border ---
    # ensure vertical padding (gradient fills don't autoscale y)
    _apply_ridgeline_ypadding(ax, y_positions, max_half_height, pad_frac=0.02)

    #ymin, ymax = ax.get_ylim()
    #pad = (ymax - ymin) * 0.05          # 5 % of range as margin
    #ax.set_ylim(ymin - pad, ymax + pad)

    # axes and cosmetics
    ax.set_xlim(x_min - 0.02, max(x_max, x_min) + 0.02)
    label_x = "Winning pocket probability (>= 0.5)" if (probability_mode == "winners") else "Pocket probability (full range 0–1)"
    ax.set_xlabel(label_x)
    ax.set_ylabel("log2(avg_n_holo / avg_n_apo) — binned")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.grid(True, axis="x", linestyle="--", alpha=0.30)

    head = f"{'apo' if cls=='apo' else 'holo'} only"
    tail = "winners" if (probability_mode == "winners") else "full"
    ax.set_title(title or f"Ridgeline ({head}, {tail})")

    from matplotlib.lines import Line2D
    handle = Line2D([], [], lw=6,
                    color=(color_apo if cls == "apo" else color_holo),
                    alpha=alpha,
                    label=f"{cls} probability")
    ax.legend(handles=[handle], loc="best", framealpha=0.95, fontsize=9)
    #ax.legend(handles=[handle], loc="upper right", framealpha=0.95, fontsize=9)

    fig.tight_layout()
    save_figure(fig, save_base, formats)


def _draw_single_upward_violin_ridge(
    ax, *, y0: float, values: np.ndarray,
    x_min: float = 0.5, x_max: float = 1.0, n_points: int = 201,
    height: float = 0.40, color: str = "#4C78A8",
    alpha: float = 0.55, edgecolor: str | None = "#4A4A4A", edgewidth: float = 0.6,
    smooth_sigma: float = 1.2,       # used only in mode="hist"
    mode: str = "kde",               # "kde" (Silverman) or "hist"
    bw_scale: float = 1.0,           # KDE bandwidth multiplier
    zorder: float = 10.0,
):
    """Draw one upward half-violin at baseline y0 using either KDE(Silverman) or hist+Gaussian smooth."""
    import numpy as np
    from matplotlib.patches import Polygon

    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    xg = np.linspace(x_min, x_max, n_points)

    if vals.size == 0:
        return

    if mode == "kde":
        # --- Silverman bandwidth + Gaussian KDE ---
        def _silverman_bw(v):
            v = v[np.isfinite(v)]
            n = len(v)
            if n <= 1: return 1.0
            s = np.nanstd(v, ddof=1)
            iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
            sigma = min(s, iqr/1.349) if np.isfinite(iqr) and iqr > 0 else s
            if not np.isfinite(sigma) or sigma == 0: sigma = 1.0
            return 0.9 * sigma * n ** (-1/5)

        def _kde_1d(v, grid, bw):
            v = v[np.isfinite(v)]
            n = len(v)
            if n == 0: return np.zeros_like(grid)
            bw = max(bw, np.finfo(float).eps)
            u = (grid[:, None] - v[None, :]) / bw
            return np.exp(-0.5 * u * u).sum(axis=1) / (n * bw * np.sqrt(2.0*np.pi))

        bw = _silverman_bw(vals) * float(bw_scale)
        dens = _kde_1d(vals, xg, bw)
        dens = dens / (np.nanmax(dens) if np.nanmax(dens) > 0 else 1.0)

    else:
        # --- histogram + Gaussian smoothing (previous behavior) ---
        hist, edges = np.histogram(vals, bins=n_points-1, range=(x_min, x_max), density=True)
        if smooth_sigma > 0:
            rad = max(1, int(round(3*smooth_sigma)))
            kx = np.arange(-rad, rad+1, dtype=float)
            k = np.exp(-(kx**2)/(2*smooth_sigma**2)); k /= k.sum()
            hist = np.convolve(hist, k, mode="same")
        dens = hist / (np.nanmax(hist) if np.nanmax(hist) > 0 else 1.0)
        xg = 0.5*(edges[:-1]+edges[1:])

    yt = y0 + height * dens
    poly_x = np.concatenate([xg, xg[::-1]])
    poly_y = np.concatenate([np.full_like(xg, y0), yt[::-1]])
    ax.add_patch(Polygon(np.c_[poly_x, poly_y], closed=True,
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=edgewidth, alpha=alpha, zorder=zorder))


def _make_linear_cmap(base_hex: str, lighten: float = 0.75):
    """
    Build a left→right colormap that goes from a lightened version of base_hex (left, low x)
    to the original base_hex (right, high x). 'lighten' is the fraction toward white (0..1).
    """
    import matplotlib.colors as mcolors
    base = np.array(mcolors.to_rgb(base_hex), dtype=float)
    light = 1.0 - (1.0 - base) * (1.0 - float(lighten))
    return mcolors.LinearSegmentedColormap.from_list("grad_"+base_hex, [light, base], N=256)


def _compute_ratio_axis(a_apo: np.ndarray, a_holo: np.ndarray, *,
                        mode: str = "log2", eps: float = 1.0) -> tuple[np.ndarray, str]:
    """
    Return (ratio_values, y_axis_label) for ridge binning.
    mode:
      - "log2": log2((holo+eps)/(apo+eps))  → symmetric around 0 (1:1)
      - "linear": (holo+eps)/(apo+eps)      → raw ratio, equilibrium at 1.0
    """
    mode = (mode or "log2").lower()
    if mode == "linear":
        r = (a_holo + eps) / (a_apo + eps)
        return r.astype(float), "ratio (avg_n_holo / avg_n_apo)"
    # default: log2
    r = np.log2((a_holo + eps) / (a_apo + eps))
    return r.astype(float), "log₂(avg_n_holo / avg_n_apo)"


def _make_anchored_edges(minv: float, maxv: float, bins: int, center_at: float | None):
    """
    Equal-width edges over [minv, maxv], but shifted so that `center_at`
    is exactly one of the bin *centers* (if provided).
    We may extend a tiny bit beyond [minv, maxv] to keep `bins` constant.
    """
    bins = max(int(bins), 2)
    minv = float(minv); maxv = float(maxv)
    if not np.isfinite(minv) or not np.isfinite(maxv) or maxv <= minv:
        return np.linspace(minv, maxv + 1e-12, bins + 1)

    if center_at is None:
        return np.linspace(minv, maxv, bins + 1)

    bw = (maxv - minv) / bins
    c0 = minv + 0.5 * bw                      # first center before shifting
    delta = (center_at - c0) % bw             # shift needed to land a center on center_at
    if delta > 0.5 * bw:                      # wrap to [-bw/2, bw/2]
        delta -= bw

    centers = c0 + delta + np.arange(bins) * bw
    edges = np.empty(bins + 1, dtype=float)
    edges[0] = centers[0] - 0.5 * bw
    edges[1:] = centers + 0.5 * bw
    return edges



# ===============================================================
# SECTION 8 — Sample Size vs Ratio vs Probability Relationships
# Triggered by: --plots size_to_ratio_plots
# ===============================================================

def _prepare_size_ratio_prob_data(df,
                                  avg_apo_col: str,
                                  avg_holo_col: str,
                                  p_apo_col: str,
                                  p_holo_col: str,
                                  *,
                                  nan_as_zero: bool = True) -> pd.DataFrame:
    """Create a small frame with avg_n_mean, ratio_log2 and ΔP using your column names."""
    out = pd.DataFrame(index=df.index).copy()
    a_apo  = _coerce_numeric(df[avg_apo_col],  nan_as_zero=nan_as_zero)
    a_holo = _coerce_numeric(df[avg_holo_col], nan_as_zero=nan_as_zero)
    p_apo  = _coerce_numeric(df[p_apo_col],    nan_as_zero=False)
    p_holo = _coerce_numeric(df[p_holo_col],   nan_as_zero=False)

    out["avg_n_mean"] = 0.5 * (a_apo + a_holo)
    # keep same ratio convention as elsewhere in your file
    out["ratio_log2"] = np.log2(a_holo / a_apo)
    out["deltaP"]     = (p_holo - p_apo)
    # keep only finite rows
    m = np.isfinite(out["avg_n_mean"]) & np.isfinite(out["ratio_log2"]) & np.isfinite(out["deltaP"])
    return out.loc[m].copy()


def _plot_scatter_size_ratio_prob(df_core: pd.DataFrame, *, title: str, save_base: str | None, formats=("png",)):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df_core["avg_n_mean"], df_core["ratio_log2"],
                    c=df_core["deltaP"], cmap="coolwarm", alpha=0.7, edgecolor="none", rasterized=True)
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label("ΔP = avg_P(holo) − avg_P(apo)")
    ax.set_xlabel("Average sample size (apo+holo)/2")
    ax.set_ylabel("log₂(avg_n_holo / avg_n_apo)")
    ax.set_title(title); ax.grid(True, alpha=0.4)
    save_figure(fig, save_base, formats)


def _plot_heatmap_size_ratio_prob(df_core: pd.DataFrame, *, title: str, save_base: str | None, formats=("png",), bins=(40, 40)):
    # mean ΔP per (avg_n_mean, ratio_log2) bin
    H_sum, xedges, yedges = np.histogram2d(df_core["avg_n_mean"], df_core["ratio_log2"], bins=bins, weights=df_core["deltaP"])
    H_cnt, _, _ = np.histogram2d(df_core["avg_n_mean"], df_core["ratio_log2"], bins=(xedges, yedges))
    H_mean = np.divide(H_sum, H_cnt, out=np.zeros_like(H_sum), where=H_cnt > 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(H_mean.T, origin="lower",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap="coolwarm", aspect="auto")
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("Mean ΔP per bin")
    ax.set_xlabel("Average sample size (apo+holo)/2")
    ax.set_ylabel("log₂(avg_n_holo / avg_n_apo)")
    ax.set_title(title)
    save_figure(fig, save_base, formats)


def _plot_ratio_vs_prob_by_size(df_core: pd.DataFrame, *, title: str, save_base: str | None, formats=("png",)):

    import seaborn as sns
    d = df_core.copy()
    d["size_quartile"] = pd.qcut(d["avg_n_mean"], q=4, labels=["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=d, x="ratio_log2", y="deltaP", hue="size_quartile",
                    palette="viridis", alpha=0.7, s=25, ax=ax, edgecolor=None)
    ax.axhline(0, color="grey", lw=1, ls="--"); ax.axvline(0, color="grey", lw=1, ls="--")
    ax.set_xlabel("log₂(avg_n_holo / avg_n_apo)")
    ax.set_ylabel("ΔP = avg_P(holo) − avg_P(apo)")
    ax.set_title(title); ax.grid(True, alpha=0.4)
    ax.legend(title="Average sample size quartile", framealpha=0.95)
    save_figure(fig, save_base, formats)


def _plot_hexbin_size_ratio_prob(
    df_core: pd.DataFrame, *,
    title: str,
    save_base: str | None,
    formats=("png",),
    gridsize=(63, 21),
    mincnt=5,
    delta_scale="full",        # "full": ΔP in [-1,1]; "half": ΔP/2 in [-0.5,0.5]
    extent: tuple[float, float, float, float] | None = None,  # (x_min,x_max,y_min,y_max)
    show_counts: bool = False, # NEW: default False → only the left panel
):
    """
    Hexbin of avg_n_mean (x) vs ratio_log2 (y), colored by mean ΔP with a fixed scale.
    If show_counts is True, adds a second panel with counts; otherwise only ΔP panel is drawn.
    """
    x = df_core["avg_n_mean"].to_numpy()
    y = df_core["ratio_log2"].to_numpy()
    dP = df_core["deltaP"].to_numpy()

    if delta_scale == "half":
        z = 0.5 * dP
        vmin, vmax = -0.5, 0.5
        cbar_label = "Signed max prob offset (≈ ΔP/2)"
    else:
        z = dP
        vmin, vmax = -1.0, 1.0
        cbar_label = "ΔP = avg_P(holo) − avg_P(apo)"

    if show_counts:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7.8, 6.0))

    # --- Mean ΔP per hex (left / only panel) ---
    hb = ax.hexbin(
        x, y, C=z, reduce_C_function=np.mean,
        gridsize=gridsize, mincnt=mincnt,
        cmap="coolwarm", linewidths=0, vmin=vmin, vmax=vmax,
        extent=extent
    )
    c1 = fig.colorbar(hb, ax=ax)
    c1.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel("Average sample size (apo+holo)/2")
    ax.set_ylabel("log₂(avg_n_holo / avg_n_apo)")
    ax.grid(True, alpha=0.25, linestyle=":")
    # zero-ratio reference
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.35)

    # --- Optional counts panel (right) ---
    if show_counts:
        ax2 = axes[1]
        hb2 = ax2.hexbin(
            x, y, gridsize=gridsize, mincnt=mincnt,
            cmap="viridis", linewidths=0, extent=extent
        )
        c2 = fig.colorbar(hb2, ax=ax2)
        c2.set_label("Count per hex")
        ax2.set_title("Counts")
        ax2.set_xlabel("Average sample size (apo+holo)/2")
        ax2.grid(True, alpha=0.25, linestyle=":")
        # zero-ratio reference
        ax2.axhline(0.0, color="k", lw=0.8, alpha=0.35)

    fig.tight_layout()
    save_figure(fig, save_base, formats)


def _plot_ridgeline_ratio_prob_overlay_quartiles(
    df_core: pd.DataFrame, *,
    title: str,
    save_base: str | None,
    formats=("png",),
    n_ratio_bins=12,
    min_n_per_quartile=40,
    enforce_all_quartiles=True,          # strict (require all 4 quartiles per bin)
    use_quartile_overlap_range=True,     # strict: intersect Q1..Q4; relaxed: use full range
    range_clip_pct=(0.5, 99.5),          # used when use_quartile_overlap_range=False
    overlap_clip_pct=(1.0, 99.0),        # used when use_quartile_overlap_range=True
    bw_scale=1.0,
    ridge_height=0.9, # 0.85,
    y_step=1.0,
    x_range=(-1.0, 1.0),                 # ΔP = P(holo) − P(apo) domain
    center_bins_at: float | None = 0.0,  # anchor a ratio bin at 0.0 (1:1)
    ridge_draw_priority="top",           # paint bottom→top so top rows are on top
    quartile_draw_order="ascending",     # draw Q1→Q4 (Q4 ends up visually on top)
    alpha=0.70,
    palette: str = "blues",              # "blues" (light→dark) or "greys"
):
    """
    Overlay ridgeline of ΔP across ratio bins; each row overlays FOUR KDE ridges
    corresponding to the size quartiles (Q1..Q4). Uses df_core with columns:
      - 'avg_n_mean'   : (avg_n_apo + avg_n_holo)/2
      - 'ratio_log2'   : log2(avg_n_holo / avg_n_apo)
      - 'deltaP'       : P(holo) - P(apo)
    Relies on helpers: _make_anchored_edges(...) and save_figure(...).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # ---- helpers ----
    def _silverman_bw(vals: np.ndarray) -> float:
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        if n <= 1:
            return 1.0
        s = np.nanstd(vals, ddof=1)
        iqr = np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)
        sigma = min(s, iqr / 1.349) if np.isfinite(iqr) and iqr > 0 else s
        if not np.isfinite(sigma) or sigma == 0:
            sigma = 1.0
        return 0.9 * sigma * n ** (-1/5)

    def _kde_1d(vals: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        if n == 0:
            return np.zeros_like(grid)
        bw = max(bw, np.finfo(float).eps)
        u = (grid[:, None] - vals[None, :]) / bw
        return np.exp(-0.5 * u * u).sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))

    def _quartile_palette(kind: str):
        if kind == "greys":
            # Q1..Q4 light→dark
            return ["#d9d9d9", "#bdbdbd", "#969696", "#525252"]
        # default blues (Q1..Q4 light→dark)
        return ["#c6dbef", "#9ecae1", "#6baed6", "#2171b5"]

    def _center(bin_interval):
        return 0.5 * (bin_interval.left + bin_interval.right) if hasattr(bin_interval, "left") else np.nan

    # ---- data prep & quartiles ----
    d = df_core.copy()
    d = d[np.isfinite(d["avg_n_mean"]) & np.isfinite(d["ratio_log2"]) & np.isfinite(d["deltaP"])]
    d["size_quartile"] = pd.qcut(
        d["avg_n_mean"], q=4,
        labels=["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"]
    )
    q_labels = list(d["size_quartile"].cat.categories)
    colors = _quartile_palette(palette)

    # ---- choose ratio range for binning ----
    r_all = d["ratio_log2"].to_numpy(float)
    r_all = r_all[np.isfinite(r_all)]
    if use_quartile_overlap_range:
        p_lo, p_hi = overlap_clip_pct
        q_ranges = []
        for qlab in q_labels:
            r_q = d.loc[d["size_quartile"] == qlab, "ratio_log2"].to_numpy(float)
            r_q = r_q[np.isfinite(r_q)]
            if r_q.size == 0:
                print(f"Overlay ridgeline: no ratio data for {qlab}; skipping.")
                return
            lo = np.nanpercentile(r_q, p_lo)
            hi = np.nanpercentile(r_q, p_hi)
            q_ranges.append((lo, hi))
        rmin_use = max(lo for lo, _ in q_ranges)
        rmax_use = min(hi for _, hi in q_ranges)
        if not np.isfinite(rmin_use) or not np.isfinite(rmax_use) or rmin_use >= rmax_use:
            print("Overlay ridgeline: empty quartile-overlap range; adjust overlap_clip_pct or disable overlap.")
            return
    else:
        p_lo, p_hi = range_clip_pct
        rmin_use = float(np.nanpercentile(r_all, p_lo))
        rmax_use = float(np.nanpercentile(r_all, p_hi))
        if not np.isfinite(rmin_use) or not np.isfinite(rmax_use) or rmin_use >= rmax_use:
            rmin_use, rmax_use = np.nanmin(r_all), np.nanmax(r_all)

    # ---- re-bin ratio within chosen range; anchor at 0.0 if requested ----
    edges = _make_anchored_edges(rmin_use, rmax_use, max(int(n_ratio_bins), 2), center_bins_at)
    d = d[(d["ratio_log2"] >= edges[0]) & (d["ratio_log2"] <= edges[-1])]
    d["ratio_bin"] = pd.cut(d["ratio_log2"], bins=edges, include_lowest=True)

    # ---- choose bins to draw (strict vs relaxed) ----
    bin_levels = list(d["ratio_bin"].cat.categories)  # all bins from cut, including empty
    bins_chosen = []
    for b in bin_levels:
        g = d[d["ratio_bin"] == b]
        if enforce_all_quartiles:
            ok = all(int((g["size_quartile"] == qlab).sum()) >= int(min_n_per_quartile)
                     for qlab in q_labels)
            if ok:
                bins_chosen.append(b)
        else:
            bins_chosen.append(b)

    if not bins_chosen:
        print("Overlay ridgeline: no bins after enforcement; try relaxed mode or lower min_n_per_quartile.")
        return

    # ---- order bins (bottom → top) and map to Y properly ----
    bins_sorted = sorted(bins_chosen, key=_center)  # negative centers first → bottom; positive → top
    y_positions = np.arange(len(bins_sorted), dtype=float) * y_step
    y_map = {b: y_positions[i] for i, b in enumerate(bins_sorted)}

    # paint order (so top rows don’t get cut)
    if str(ridge_draw_priority).lower() == "top":
        ordered_bins = bins_sorted[:]       # paint bottom→top; top rows last (on top)
    else:
        ordered_bins = bins_sorted[::-1]    # paint top→bottom

    # ---- ΔP grid for KDEs ----
    x = np.linspace(x_range[0], x_range[1], 512)

    # ---- draw ----
    fig, ax = plt.subplots(figsize=(10, 8))
    # faint row guides
    for j in range(len(bins_sorted) + 1):
        ax.axhline(j * y_step, color="k", lw=0.6, alpha=0.25, zorder=0)

    any_drawn = False
    q_iter = q_labels if quartile_draw_order.lower() == "ascending" else q_labels[::-1]

    for paint_rank, b in enumerate(ordered_bins):
        g = d[d["ratio_bin"] == b]
        baseline = y_map[b]
        row_z = 5 + paint_rank * 2

        # draw quartiles over the same baseline (overlay); skip missing when relaxed
        for qlab in q_iter:
            vals = g.loc[g["size_quartile"] == qlab, "deltaP"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if enforce_all_quartiles and vals.size < int(min_n_per_quartile):
                continue
            if not enforce_all_quartiles and vals.size == 0:
                continue

            bw = _silverman_bw(vals) * bw_scale
            yk = _kde_1d(vals, x, bw)
            if yk.max() > 0:
                yk = (yk / yk.max()) * ridge_height

            # thin white underlay to reduce cutting artifacts
            ax.fill_between(x, baseline, baseline + yk,
                            color="white", alpha=1.0, linewidth=0, zorder=row_z - 1)
            # main colored ridge
            qi = q_labels.index(qlab)
            ax.fill_between(x, baseline, baseline + yk,
                            facecolor=colors[qi], edgecolor="black",
                            linewidth=0.8, alpha=alpha, rasterized=True, zorder=row_z)
            any_drawn = True

        # compact in-axes count near right edge
        n_row = len(g)
        x_left, x_right = x_range
        x_text = x_right - 0.03 * (x_right - x_left)
        ax.text(x_text, baseline + ridge_height * 0.5, f"n={n_row:,}",
                fontsize=8.5, ha="right", va="center",
                color="#444444", family="DejaVu Sans Mono", zorder=row_z + 1)

    if not any_drawn:
        print("Overlay ridgeline: nothing drawn; relax constraints or check data.")
        plt.close(fig)
        return

    # ---- axes & legend ----
    ax.set_xlim(x_range)
    ax.set_xlabel("ΔP = P(holo) − P(apo)")
    ax.set_yticks([y_map[b] for b in bins_sorted])
    ax.set_yticklabels([f"{_center(b):.2f}" for b in bins_sorted])
    ax.set_ylabel("log₂(avg_n_holo / avg_n_apo) — binned (negative bottom, positive top)")
    ax.grid(axis="x", alpha=0.25, linestyle=":", zorder=1)
    ax.set_title(title)

    handles = [Patch(facecolor=colors[i], edgecolor="black", label=q_labels[i], alpha=alpha) for i in range(4)]
    ax.legend(handles=handles, title="Average sample size quartile",
              framealpha=0.95, loc="upper left")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + ridge_height * 0.25)

    fig.tight_layout()
    save_figure(fig, save_base, formats)




































def main(argv=None):

    # Set your local default input (CSV or .zip with one CSV inside)
    infile_default  = str(cfg.UBS_AGG_CLASSIFICATIONS_CSV)

    p = argparse.ArgumentParser(
        description="Prevalence vs min average candidates sweep (apo vs holo)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O & saving
    p.add_argument("--in_path",                 default=infile_default,                     help="Input CSV or .zip with one CSV")
    p.add_argument("--out_summary",             default="prevalence_sweep_summary",         help="Base filename for the combined summary CSV, to be suffixed by the used threshold")
    p.add_argument("--save_formats", nargs="*", default=["png"],                            help="Plot formats to save (e.g., png svg pdf)")
    p.add_argument("--plots_dir",               default=str(cfg.WORK_DIR / "prevalence_sweep_output" / "plots"),           help="Directory to save plots; empty disables saving")
    p.add_argument("--text_dir",                default=str(cfg.WORK_DIR / "prevalence_sweep_output" / "text"),            help="Directory to save CSV outputs; if empty, use the directory part of --out_summary")

    # CSV Columns
    p.add_argument("--class_col",               default="highest_p_mean_pocket",            help="Column with verbose apo/holo classification") # majority_vote_chain | highest_p_mean_pocket
    p.add_argument("--avg_apo_col",             default="avg_n_apo",                        help="Column with average apo candidates")
    p.add_argument("--avg_holo_col",            default="avg_n_holo",                       help="Column with average holo candidates")
    p.add_argument("--threshold_col",           default="threshold",                        help="Column that identifies the UBS threshold")
    p.add_argument("--p_apo_pocket_col",        default="avg_P(apo)_mean_pocket",           help="Pocket-level apo probability column (mean over pocket)")
    p.add_argument("--p_holo_pocket_col",       default="avg_P(holo)_mean_pocket",          help="Pocket-level holo probability column (mean over pocket)")

    # Add holo +1 
    p.add_argument("--holo_plus_one",           default=1,                                  help="Increase n_holo or avg_n_holo to reflect the query being added to the pocket metrics")

    # CSV probability filter    
    p.add_argument("--require_pocket_probs",    default=True,                               help="Drop rows where pocket probability columns are missing/empty (NaN).")
    p.add_argument("--require_pocket_probs_mode", choices=["both","either"], default="both", help="both=require both P(apo) and P(holo); either=require at least one.")

    p.add_argument("--filter_ambiguous",                                default=False,      help="If true, drop rows with max pocket P < min_prob_mean_pocket (true/false)")
    p.add_argument("--min_prob_mean_pocket",    type=float,             default=0.60,       help="Minimum of max(apo,holo) pocket probability to keep a UBS")

    # CSV avg n filter
    p.add_argument("--min_apo_avg_n",           type=int,               default=1,          help="Discard rows/UBSs with less than N entries in the avg_n_apo column")
    p.add_argument("--min_holo_avg_n",          type=int,               default=1,          help="Discard rows/UBSs with less than N entries in the avg_n_holo column")

    # Threshold groups
    p.add_argument("--group_lj",                        default=True,                       help="Collapse ST_LJ* into ST_LJ (true/false)")
    p.add_argument("--threshold_include",   nargs="*",  default="ST_050", metavar="THR",    help="Exact threshold labels to include (post-normalization); empty = all")
    #p.add_argument("--threshold_include", nargs="*", default=['ST_030', 'ST_050', 'ST_070', 'ST_100', 'ST_LJ'], metavar="THRESHOLD",
    #               help="List of thresholds to include; omit to include all (e.g., ST_030 ST_050 ST_070 ST_100 ST_LJ)")

    # Which plots to render (list)
    p.add_argument("--plots", nargs="*", default=["counts_vs_k","prevalence_vs_k","prevalence_bubble_vs_k","scatter_avg_n","heatmap_2d","prob_correlation","ridgeline_prob_vs_n_ratio","scatter_ratio_unbinned","size_to_ratio_plots"],
                   choices=["counts_vs_k","prevalence_vs_k","prevalence_bubble_vs_k","scatter_avg_n","heatmap_2d","prob_correlation","ridgeline_prob_vs_n_ratio","scatter_ratio_unbinned","size_to_ratio_plots"], help="Which plots to generate")

    # Trendline
    # Sweep (k)
    p.add_argument("--mode",        choices=["per_class","global"],     default="per_class",    help="Filtering mode for sweep")
    p.add_argument("--k_start",                 type=float,             default=0.0,            help="Start k (inclusive)")
    p.add_argument("--k_stop",                  type=float,             default=500.0,          help="Stop k (inclusive)")
    p.add_argument("--k_step",                  type=float,             default=1.0,            help="k step")
    p.add_argument("--nan_as_zero",                                     default=False,          help="Treat NaN in averages as 0 (true/false)")

    # Bubble vs k (if requested)
    p.add_argument("--bubble_size_min",         type=float,             default=10.0,           help="Min marker size for bubble plot")
    p.add_argument("--bubble_size_max",         type=float,             default=500.0,          help="Max marker size for bubble plot")

    # Scatter
    # x/y mode
    p.add_argument("--scatter_x_mode",  nargs="*",  choices=["min","max","class","common","delta","ratio","mean"], default=["min","max","class","common","delta","ratio","mean"], help="X value for scatter: min, class, max, common mutual min (tail), or delta filter")
    p.add_argument("--scatter_y_mode",              choices=["class","trend"],                       default="trend",   help="Y mapping: trend=local prevalence, class=0/1 bands")
    p.add_argument("--scatter_y_units",             choices=["fraction","percent"],                  default="percent", help="Y units")
    # Ratio unbinned
    p.add_argument("--scatter_ratio_unbinned_jitter", type=float, default=0.0, help="Horizontal jitter (std dev) for unbinned ratio scatter")
    p.add_argument("--scatter_ratio_unbinned_size",   type=float, default=10.0, help="Marker size for unbinned ratio scatter")

    # Delta
    p.add_argument("--delta_direction",             choices=["holo_minus_apo","apo_minus_holo"],     default="holo_minus_apo",   help="Delta = holo-apo or apo-holo")
    p.add_argument("--scatter_x_symmetric",                                                          default=False,              help="When x_mode=If delta and no xmin/xmax provided, make x-lims symmetric around 0 (true/false)")

    p.add_argument("--scatter_bin_width",       type=float,             default=1.0,        help="Bin width for trend mapping/lines")
    p.add_argument("--scatter_sample",          type=int,               default=0,          help="Max points to plot (0=all)")

    # Style
    p.add_argument("--scatter_alpha",           type=float,             default=0.15,       help="Point alpha")
    p.add_argument("--scatter_jitter",          type=float,             default=0.02,       help="Vertical jitter half-span")
    p.add_argument("--scatter_size",            type=float,             default=6.0,        help="Point size")
    p.add_argument("--scatter_y_pad",           type=float,             default=0.04,       help="Extra vertical pad beyond [0..1] / [0..100] (default=0.04")
    p.add_argument("--scatter_xmin",            type=float,             default=None,       help="Min x to show/filter (None=auto)")
    p.add_argument("--scatter_xmax",            type=float,             default=None,       help="Max x to show/filter (None=auto)")

    p.add_argument("--scatter_style",   choices=["default", "points", "jitter", "hexbin_facets"],  default="default",   help="How to render dense scatter: raw points, jittered bands, or per-class hexbin facets")

    p.add_argument("--rasterize",                                       default=True,       help="Rasterize heavy artists to keep vector outputs small")
    p.add_argument("--scatter_overlay_trend",                           default=False,      help="Overlay the prevalence trend line on the scatter")
    p.add_argument("--trend_grid", choices=["edges","centers"],         default="edges",    help="Grid for tail/binned trend evaluation (edges include k=0; centers use mid-bin)")

    # Heatmap (2D prevalence of holo over apo/holo grid)
    p.add_argument("--heatmap_bins_apo",  type=int, default=40, help="Number of bins along avg_apo axis for 2D heatmap")
    p.add_argument("--heatmap_bins_holo", type=int, default=40, help="Number of bins along avg_holo axis for 2D heatmap")

    # Ridgeline plot
    p.add_argument("--ridgeline_bins",          type=int,   default=24,         help="Number of ratio bins (ridgeline)")
    p.add_argument("--ridgeline_overlap",       type=float, default=0.35,       help="Overlap between ridges (0..0.8)")
    p.add_argument("--ridgeline_half_height",   type=float, default=0.42,       help="Vertical amplitude for each half")
    p.add_argument("--ridgeline_alpha",         type=float, default=0.60,       help="Violin alpha")
    
    p.add_argument("--ridgeline_center_at",     type=float, default=0.0,        help="Center ridgeline ratio bins at this log2 value (e.g., 0.0 for 1:1). Use None to disable.")
    p.add_argument("--ridgeline_min_bin_count", type=int,   default=20,          help="Drop ridges with fewer than this many samples (total apo+holo).")
    #p.add_argument("--ridgeline_min_bin_count_mode", choices=["total","class"], default="total", help="Count mode for singleclass pruning: total UBSs in bin vs only class points used.")

    p.add_argument("--drop_edge_bins",          type=int,   default=0,          help="Number of bins to trim from each side (default 0)")
    p.add_argument("--single_ridge_half_height",type=float, default=0.42,       help="Vertical amplitude for each half")
    p.add_argument("--single_ridge_alpha",      type=float, default=0.9,        help="Violin alpha")
    p.add_argument("--ridgeline_smooth_sigma",  type=float, default=1.2,        help="Smoothing of histogram density")
    p.add_argument("--single_group_prob_mode",  type=str,   default="full",     help="winners (shows only probabilities over 0.5) or full (probabilities 0-1)")




    # Spyder/IPython-friendly parsing (ignore extra flags)
    args, _unknown = p.parse_known_args(argv)

    # Parse arguments (defaults are absolute paths from common/config; used as-is)
    plots_dir = args.plots_dir if args.plots_dir else None
    text_dir  = args.text_dir if args.text_dir else None
    os.makedirs(text_dir, exist_ok=True)

    save_formats  = [s.lower().lstrip(".") for s in (args.save_formats or ["png"])]
    plots_wanted  = set(args.plots)
    threshold_col = args.threshold_col


    # Prepare Threshold include filter
    include_set = None
    if args.threshold_include.strip():
        include_set = {t.strip() for t in args.threshold_include.split(",") if t.strip()}


    def base(thr: str, name: str) -> str | None:
        if not plots_dir: return None
        return os.path.join(plots_dir, str(thr), name)


    # Read CSV
    try:
        df = read_csv_or_zipped_csv(args.in_path)
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(2)

    # Validate columns
    missing = [c for c in (args.class_col, args.avg_apo_col, args.avg_holo_col) if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}", file=sys.stderr)
        print("Available columns:", list(df.columns), file=sys.stderr)
        sys.exit(2)

    # --- Add +1 to holo counts to include the query holo pocket (AHoJ) / per-entry correction (UBS) ---
    # This is the SINGLE pseudocount application (decision 2: holo-only +1, re-adding the query holo
    # that upstream evidence counts exclude). It mutates the shared avg_n_holo column once, so every
    # downstream consumer — including the evidence ratio — sees holo already +1'd. The log2 evidence
    # ratios are therefore computed as log2(a_holo / a_apo) (NOT (a_holo+1)/(a_apo+1)); re-adding +1
    # here would double-count holo and spuriously pad apo. See Figure3A_ridgeline_REV2 / 04_p2rank.
    if args.holo_plus_one == 1:
        _h = pd.to_numeric(df[args.avg_holo_col], errors="coerce")
        if args.nan_as_zero:
            _h = _h.fillna(0.0)
        df[args.avg_holo_col] = _h + 1.0

    # Filter 1 (NaN classification): require pocket probabilities to be present (means classification actually ran)
    if _as_bool(args.require_pocket_probs):
        # Ensure columns exist
        missing_p = [c for c in (args.p_apo_pocket_col, args.p_holo_pocket_col) if c not in df.columns]
        if missing_p:
            print(f"[require_pocket_probs] WARNING: missing columns {missing_p}; skipping this filter.")
        else:
            p_apo  = pd.to_numeric(df[args.p_apo_pocket_col],  errors="coerce")
            p_holo = pd.to_numeric(df[args.p_holo_pocket_col], errors="coerce")

            if args.require_pocket_probs_mode == "either":
                m_prob = np.isfinite(p_apo) | np.isfinite(p_holo)
            else:
                m_prob = np.isfinite(p_apo) & np.isfinite(p_holo)

            before = len(df)
            df = df.loc[m_prob].copy()
            print(f"[require_pocket_probs] kept {len(df):,} / {before:,} (dropped {before-len(df):,})")
    #sys.exit(1)

    # Filter 2 (avg_n_apo, avg_n_holo)
    df = filter_by_min_avg_n(df, avg_apo_col=args.avg_apo_col, avg_holo_col=args.avg_holo_col,
       min_apo_avg_n=args.min_apo_avg_n, min_holo_avg_n=args.min_holo_avg_n, mode="both", nan_as_zero=args.nan_as_zero, verbose=True)


    # Run sweep
    if threshold_col in df.columns:
        df_thr = df.copy()

        # Optionally collapse ST_LJ* into ST_LJ
        if args.group_lj:
            collapsed = df_thr[threshold_col].astype(str)
            lj_mask = collapsed.str.startswith("ST_LJ", na=False)
            df_thr.loc[lj_mask, threshold_col] = "ST_LJ"

        # Optionally include only selected thresholds
        if include_set is not None:
            df_thr = df_thr[df_thr[threshold_col].isin(include_set)].copy()

        # --- NEW: drop ambiguous pocket-level classifications (optional) ---
        if args.filter_ambiguous:
            # be conservative: treat NaN as 0.0 (unknown -> weak)
            p_apo  = pd.to_numeric(df_thr.get(args.p_apo_pocket_col),  errors="coerce").fillna(0.0) # Converts NaN to 0.0!
            p_holo = pd.to_numeric(df_thr.get(args.p_holo_pocket_col), errors="coerce").fillna(0.0) # Converts NaN to 0.0!
            p_max  = pd.concat([p_apo, p_holo], axis=1).max(axis=1)

            before = len(df_thr)
            df_thr = df_thr.loc[p_max >= float(args.min_prob_mean_pocket)].copy()
            after  = len(df_thr)
            dropped = before - after
            print(f"Filtered ambiguous UBS by pocket P: kept {after:,} / {before:,} "
                  f"(dropped {dropped:,} with max(P_apo,P_holo) < {args.min_prob_mean_pocket:.2f})")
        # --- END NEW ---

        if df_thr.empty:
            print("No rows left after threshold filtering.", file=sys.stderr)
            sys.exit(2)

        # Set plot font
        plt.rcdefaults()
        #set_plot_font(font_name="Helvetica Neue")


        # Iterate thresholds
        summaries = []
        unique_thresholds = list(pd.unique(df_thr[threshold_col]))
        print(f"Found {len(unique_thresholds)} threshold groups in {threshold_col}: {unique_thresholds}")

        for thr in unique_thresholds:
            sub = df_thr[df_thr[threshold_col] == thr].copy()

            summary_t = run_prevalence_sweep(
                sub,
                class_col=args.class_col,
                avg_apo_col=args.avg_apo_col,
                avg_holo_col=args.avg_holo_col,
                mode=args.mode,
                k_start=args.k_start,
                k_stop=args.k_stop,
                k_step=args.k_step,
                nan_as_zero=args.nan_as_zero,
            )
            summary_t.insert(0, threshold_col, thr)
            summaries.append(summary_t)

            # Per-threshold outputs
            thr_slug = str(thr)  # safe if values are like ST_030, ST_050, ST_070, ST_100, ST_LJ
            out_csv_thr = os.path.join(text_dir, f"{args.out_summary}_{thr_slug}.csv")

            summary_t.to_csv(out_csv_thr, index=False)
            print(f"[{thr}] Saved summary to: {out_csv_thr}")

            # Brief report for this threshold
            for _, r in summary_t[:40].iterrows():
            #for _, r in summary_t.head(50).iterrows():
            #for _, r in summary_t.iterrows():
                k = r["k"]; n = int(r["n_total"])
                apo = r["prev_apo_pct"]; holo = r["prev_holo_pct"]
                apo_str = f"{apo:.1f}%" if pd.notna(apo) else "nan"
                holo_str = f"{holo:.1f}%" if pd.notna(holo) else "nan"
                print(f"[{thr}] k={k:g} -> apo={apo_str}, holo={holo_str} (n={n:,})")

            # 1) prevalence_vs_k (line)
            if "prevalence_vs_k" in plots_wanted:
                _plot_lines(summary_t, "k", ("prev_apo_pct","prev_holo_pct"),
                            title=f"Classification prevalence (%) vs k — {thr}",
                            save_base=base(thr, "prevalence_vs_k"), formats=save_formats)

            # 2) counts_vs_k
            if "counts_vs_k" in plots_wanted:
                _plot_lines(summary_t, "k", ("n_apo","n_holo"),
                            title=f"Counts vs k — {thr}",
                            save_base=base(thr, "counts_vs_k"), formats=save_formats)

            # 3) prevalence_bubble_vs_k
            if "prevalence_bubble_vs_k" in plots_wanted:
                _plot_bubble_prevalence_counts(
                    summary_t, "k",
                    y_cols=("prev_apo_pct","prev_holo_pct"),
                    size_cols=("n_apo","n_holo"),
                    title=f"Prevalence vs k (bubble=size) — {thr}",
                    save_base=base(thr, "prevalence_bubble_vs_k"),
                    formats=save_formats,
                    size_min=args.bubble_size_min, size_max=args.bubble_size_max)

            # 4) scatter_avg_n
            if "scatter_avg_n" in plots_wanted:

                for scatter_x_mode in args.scatter_x_mode:
                    plotname = f"prevalence_scatter_avg_n_xmode{scatter_x_mode}_ymode{args.scatter_y_mode}_n{float_to_str(args.scatter_xmin)}-{float_to_str(args.scatter_xmax, 'max')}"

                    _plot_scatter_min_avg_class(
                        sub,
                        class_col=args.class_col, avg_apo_col=args.avg_apo_col, avg_holo_col=args.avg_holo_col,
                        nan_as_zero=args.nan_as_zero,
                        title=f"Scatter vs {scatter_x_mode}(avg) — {thr}",
                        save_base=base(thr, plotname),
                        formats=save_formats,
                        sample_n=args.scatter_sample, jitter_y=args.scatter_jitter,
                        alpha=args.scatter_alpha, s=args.scatter_size,
                        x_mode=scatter_x_mode,
                        y_mode=args.scatter_y_mode, overlay_trend=args.scatter_overlay_trend,
                        trend_bin_width=args.scatter_bin_width, y_units=args.scatter_y_units,
                        y_pad=args.scatter_y_pad, x_lo=args.scatter_xmin, x_hi=args.scatter_xmax,
                        # NEW:
                        delta_dir=args.delta_direction,
                        x_symmetric_delta=args.scatter_x_symmetric,
                        scatter_style=args.scatter_style,
                        rasterize=args.rasterize,
                        grid_mode=args.trend_grid
                        )

            if "scatter_ratio_unbinned" in plots_wanted:
                _plot_scatter_ratio_unbinned_probs(
                    sub,
                    args.avg_apo_col, args.avg_holo_col,
                    args.p_apo_pocket_col, args.p_holo_pocket_col,
                    title=f"Scatter vs ratio(avg) — probabilities — {thr}",
                    save_base=base(thr, "prevalence_scatter_ratio_unbinned_probs"),
                    formats=save_formats,
                    x_jitter=getattr(args, "scatter_ratio_unbinned_jitter", 0.0),
                )

            # 5) heatmap_2d
            if "heatmap_2d" in plots_wanted:
                png_path = base(thr, f"prevalence_heatmap_2d_bins{args.heatmap_bins_apo}x{args.heatmap_bins_holo}") + ".png"

                _plot_prevalence_heatmap_2d(
                    sub,
                    class_col=args.class_col,
                    avg_apo_col=args.avg_apo_col,
                    avg_holo_col=args.avg_holo_col,
                    bins_apo=args.heatmap_bins_apo,
                    bins_holo=args.heatmap_bins_holo,
                    nan_as_zero=args.nan_as_zero,
                    title=f"2D holo prevalence across (avg_n_apo, avg_n_holo) — {thr}",
                    png_path=png_path
                )

                png_path = os.path.join(plots_dir, "ALL", f"prevalence_heatmap_2d_bins_ALL_{args.heatmap_bins_apo}x{args.heatmap_bins_holo}.png") if plots_dir else None
                #png_path = base(thr, f"prevalence_heatmap_2d_bins_ALL_{args.heatmap_bins_apo}x{args.heatmap_bins_holo}") + ".png"
                _plot_prevalence_heatmap_2d(
                    df,
                    class_col=args.class_col,
                    avg_apo_col=args.avg_apo_col,
                    avg_holo_col=args.avg_holo_col,
                    bins_apo=args.heatmap_bins_apo,
                    bins_holo=args.heatmap_bins_holo,
                    nan_as_zero=args.nan_as_zero,
                    title="2D holo prevalence across (avg_n_apo, avg_n_holo) — ALL",
                    png_path=png_path
                )

            # 6) probability correlation plots
            if "prob_correlation" in plots_wanted:
                _plot_prob_vs_avg_n(
                    sub, args.avg_apo_col, args.avg_holo_col,
                    args.p_apo_pocket_col, args.p_holo_pocket_col,
                    class_col=args.class_col,
                    title=f"Probability vs average n — {thr}",
                    save_base=base(thr, "prob_vs_avg_n"),
                    formats=save_formats,
                    nan_as_zero=args.nan_as_zero,
                )
                _plot_prob_balance_heatmap(
                    sub,
                    args.avg_apo_col, args.avg_holo_col,
                    args.p_apo_pocket_col, args.p_holo_pocket_col,
                    title=f"ΔP (holo−apo) vs avg_n_apo/holo — {thr}",
                    save_base=base(thr, "prob_balance_heatmap"),
                    formats=save_formats,
                )

            # 6b) ratio-based probability bias

                # Optional - filter down dataframe to only keep entries with minimum number of average enties per class
                #sub = filter_by_min_avg_n(sub, avg_apo_col=args.avg_apo_col, avg_holo_col=args.avg_holo_col,
                #   min_apo_avg_n=args.min_apo_avg_n, min_holo_avg_n=args.min_holo_avg_n, mode="both", nan_as_zero=args.nan_as_zero, verbose=True)

                _plot_prob_vs_n_ratio(
                    sub,
                    args.avg_apo_col, args.avg_holo_col,
                    args.p_apo_pocket_col, args.p_holo_pocket_col,
                    title=f"ΔP vs log₂(avg_n_holo / avg_n_apo) — {thr}",
                    save_base=base(thr, "prob_vs_n_ratio"),
                    formats=save_formats,
                    color_by_maxP=True
                )

                _plot_binned_prob_vs_n_ratio(
                    sub, args.avg_apo_col, args.avg_holo_col,
                    args.p_apo_pocket_col, args.p_holo_pocket_col,
                    title=f"Binned ΔP vs log₂(avg_n_holo/avg_n_apo) — {thr}",
                    save_base=base(thr, "prob_vs_n_ratio_binned_fullY"),
                    formats=save_formats,
                    force_full_y=True,
                    x_pad_frac=0.05,   # 5% padding on X
                    y_pad_frac=0.04    # 4% padding on Y (around [-1,1] if force_full_y)
                )

                _diagnose_prob_ratio_balance(
                    sub,
                    args.avg_apo_col, args.avg_holo_col,
                    args.p_apo_pocket_col, args.p_holo_pocket_col,
                    title=f"ΔP diagnostic — {thr}",
                    save_base=base(thr, "prob_ratio_diagnostics"),
                    formats=save_formats,
                )

            # 7) Ridgeline split or full violins of probability vs n-ratio bins
            if "ridgeline_prob_vs_n_ratio" in plots_wanted:
                #"""
                _plot_ridgeline_prob_vs_n_ratio(
                    sub,
                    args.avg_apo_col, args.avg_holo_col,
                    args.p_apo_pocket_col, args.p_holo_pocket_col,
                    title=f"Ridgeline: P(apo)/P(holo) per ratio bin — {thr}",
                    save_base=base(thr, f"ridgeline_prob_vs_n_ratio_bins{args.ridgeline_bins}"),
                    formats=save_formats,
                    bins=args.ridgeline_bins,
                    overlap=args.ridgeline_overlap,
                    max_half_height=args.ridgeline_half_height,
                    alpha=args.ridgeline_alpha,
                    smooth_sigma=args.ridgeline_smooth_sigma,
                )
                #"""

                _plot_ridgeline_prob_fullrange_same_top(
                    sub,
                    args.avg_apo_col, args.avg_holo_col,
                    args.p_apo_pocket_col, args.p_holo_pocket_col,
                    title=f"Ridgeline (full range): P(apo)/P(holo) per ratio bin — {thr}",
                    save_base=base(thr, f"ridgeline_prob_fullrange_bins{args.ridgeline_bins}"),
                    formats=save_formats,
                    bins=args.ridgeline_bins,
                    overlap=args.ridgeline_overlap,
                    max_half_height=args.ridgeline_half_height,
                    alpha=args.ridgeline_alpha,
                    smooth_sigma=args.ridgeline_smooth_sigma,
                    center_bins_at=args.ridgeline_center_at, min_bin_count=args.ridgeline_min_bin_count
                )

                # inside your per-threshold loop:
                _summarize_ridgeline_bins(
                    sub,
                    args.avg_apo_col, args.avg_holo_col,
                    args.p_apo_pocket_col, args.p_holo_pocket_col,
                    bins=args.ridgeline_bins,
                    nan_as_zero=args.nan_as_zero,   # whatever you use elsewhere
                    winshare_minprob=0.5,
                    winshare_mode="winners",
                    save_base=base(thr, f"ridgeline_prob_fullrange_bins{args.ridgeline_bins}"),
                    csv_suffix="binsummary",
                    center_bins_at=args.ridgeline_center_at,
                )

                # APO only, winners/full view
                _plot_ridgeline_prob_singleclass(
                    sub,
                    args.avg_apo_col, args.avg_holo_col,
                    args.p_apo_pocket_col, args.p_holo_pocket_col,
                    class_choice="apo",
                    probability_mode=args.single_group_prob_mode,
                    min_winner_prob=0.5,
                    title=f"Ridgeline: apo only — {str(args.single_group_prob_mode)} — {thr}",
                    save_base=base(thr, f"ridgeline_single_apo_{str(args.single_group_prob_mode)}_bins{args.ridgeline_bins}"),
                    formats=save_formats,
                    bins=args.ridgeline_bins,
                    drop_edge_bins=args.drop_edge_bins,
                    vertical_spacing=0.30, # 0.60,                 # tighter stacking
                    ridge_draw_priority="top",       # ← makes top ridges render on top
                    max_half_height=args.ridgeline_half_height,
                    alpha=args.single_ridge_alpha,
                    smooth_sigma=args.ridgeline_smooth_sigma,
                    use_gradient=True,
                    gradient_lighten=0.8,  # less bright at low x
                    center_bins_at=args.ridgeline_center_at, min_bin_count=args.ridgeline_min_bin_count
                )

                # HOLO only, winners/full view
                _plot_ridgeline_prob_singleclass(
                    sub,
                    args.avg_apo_col, args.avg_holo_col,
                    args.p_apo_pocket_col, args.p_holo_pocket_col,
                    class_choice="holo",
                    probability_mode=args.single_group_prob_mode,
                    min_winner_prob=0.5,
                    title=f"Ridgeline: holo only — {str(args.single_group_prob_mode)} — {thr}",
                    save_base=base(thr, f"ridgeline_single_holo_{str(args.single_group_prob_mode)}_bins{args.ridgeline_bins}"),
                    formats=save_formats,
                    bins=args.ridgeline_bins,
                    drop_edge_bins=args.drop_edge_bins,
                    vertical_spacing=0.30, # 0.60,
                    ridge_draw_priority="top",       # ← makes top ridges render on top
                    max_half_height=args.ridgeline_half_height,
                    alpha=args.single_ridge_alpha,
                    smooth_sigma=args.ridgeline_smooth_sigma,
                    use_gradient=True,
                    gradient_lighten=0.8,  # less bright at low x
                    center_bins_at=args.ridgeline_center_at, min_bin_count=args.ridgeline_min_bin_count
                )

            # 8) Plots for size–ratio–probability relationship
            if "size_to_ratio_plots" in plots_wanted:
                print("Generating sample size ↔ ratio ↔ probability plots...")
            
                df_core = _prepare_size_ratio_prob_data(
                    sub,
                    avg_apo_col=args.avg_apo_col,
                    avg_holo_col=args.avg_holo_col,
                    p_apo_col=args.p_apo_pocket_col,
                    p_holo_col=args.p_holo_pocket_col,
                    nan_as_zero=args.nan_as_zero,
                )

                _plot_scatter_size_ratio_prob(
                    df_core,
                    title=f"Sample Size vs Ratio colored by ΔP — {thr}",
                    save_base=base(thr, "size_ratio_scatter"),
                    formats=save_formats,
                )

                _plot_heatmap_size_ratio_prob(
                    df_core,
                    title=f"Mean ΔP across (size, ratio) bins — {thr}",
                    save_base=base(thr, "size_ratio_heatmap"),
                    formats=save_formats,
                    bins=(40, 40),
                )

                _plot_ratio_vs_prob_by_size(
                    df_core,
                    title=f"ΔP vs Ratio by Sample Size quartiles — {thr}",
                    save_base=base(thr, "ratio_vs_prob_by_size"),
                    formats=save_formats,
                )

                # Hexbin with fixed scale
                _plot_hexbin_size_ratio_prob(
                    df_core,
                    title=f"Hexbin: size × ratio colored by mean ΔP — {thr}",
                    save_base=base(thr, "size_ratio_hexbin"),
                    formats=save_formats,
                    gridsize=(63, 21),
                    mincnt=5,
                    delta_scale="full",          # or "half"
                    extent=None,                 # or (x_min, x_max, y_min, y_max) for consistency
                    show_counts=False,           # << only left graph
                )

                # Size-aware ridgeline (small multiples)
                _plot_ridgeline_ratio_prob_overlay_quartiles(
                    df_core,
                    title=f"ΔP ridgeline — overlay quartiles (strict) — {thr}",
                    save_base=base(thr, "ratio_prob_ridgeline_overlay_quartiles_strict"),
                    formats=save_formats,
                    n_ratio_bins=12,
                    min_n_per_quartile=40,
                    enforce_all_quartiles=True,
                    use_quartile_overlap_range=True,   # intersect Q1..Q4
                    overlap_clip_pct=(1.0, 99.0),
                    center_bins_at=0.0,
                )

                _plot_ridgeline_ratio_prob_overlay_quartiles(
                    df_core,
                    title=f"ΔP ridgeline — overlay quartiles (relaxed) — {thr}",
                    save_base=base(thr, "ratio_prob_ridgeline_overlay_quartiles_relaxed"),
                    formats=save_formats,
                    n_ratio_bins=12,
                    min_n_per_quartile=10,             # ignored when enforce_all_quartiles=False (only used to skip empty quartiles)
                    enforce_all_quartiles=False,
                    use_quartile_overlap_range=False,  # <- full data range
                    range_clip_pct=(0.5, 99.5),
                    center_bins_at=0.0,
                )






        # Combined summary across thresholds
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(os.path.join(text_dir, args.out_summary + ".csv"), index=False)
        #combined.to_csv(args.out_summary, index=False)
        print(f"Saved combined summary to: {args.out_summary}")

        # Filtered dataframe
        sub.to_csv(os.path.join(text_dir, f"sub_df_{thr}.csv"), index=False)
        df_core.to_csv(os.path.join(text_dir, f"df_core_{thr}.csv"), index=False)

        # Small preview
        _print_preview(combined, max_rows=8)

    else:
        # Fallback: no threshold column; run once on all rows
        summary = run_prevalence_sweep(
            df,
            class_col=args.class_col,
            avg_apo_col=args.avg_apo_col,
            avg_holo_col=args.avg_holo_col,
            mode=args.mode,
            k_start=args.k_start,
            k_stop=args.k_stop,
            k_step=args.k_step,
            nan_as_zero=args.nan_as_zero,
        )

        summary.to_csv(os.path.join(text_dir, args.out_summary + ".csv"), index=False)
        #summary.to_csv(args.out_summary, index=False)
        print(f"Saved summary to: {args.out_summary}")

        for _, r in summary.iterrows():
            k = r["k"]; n = int(r["n_total"])
            apo = r["prev_apo_pct"]; holo = r["prev_holo_pct"]
            apo_str = f"{apo:.1f}%" if pd.notna(apo) else "nan"
            holo_str = f"{holo:.1f}%" if pd.notna(holo) else "nan"
            print(f"k={k:g} -> apo={apo_str}, holo={holo_str} (n={n:,})")

        _print_preview(summary, max_rows=8)

    return 0



if __name__ == "__main__":
    main([])
    #raise SystemExit(main())
