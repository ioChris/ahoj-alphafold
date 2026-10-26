# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:43:28 2025

@author: Christos P. Feidakis
"""


## stats\_api.py (orchestration; no plotting)


# stats_api.py
from __future__ import annotations
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np


from stats_core import (
    StatsConfig, summarize_groups, DPClassifyConfig, dp_classify_new_point,
    coverage_ks_uniformity, coverage_entropy, dispersion_nnd
    )



### Functions

def compute_job_summaries(df: pd.DataFrame, metrics: List[str], by: List[str], *, cfg: StatsConfig | None = None) -> pd.DataFrame:
    cfg = StatsConfig() if cfg is None else cfg
    return summarize_groups(df, by=by, metrics=metrics, cfg=cfg)


def classify_point_overlap(df: pd.DataFrame, metrics: List[str], new_point_df: pd.DataFrame, *, cfg: DPClassifyConfig | None = None) -> Dict:
    return dp_classify_new_point(df, metrics, new_point_df, cfg)


def make_statistics_result(per_metric: Dict, overall: Dict, variables_block: Dict) -> Dict:
    return {
        "per_metric": per_metric,
        "overall": overall,
        "variables_block": variables_block,
    }


def _compute_range_theor(series: pd.Series, metric: str) -> float:
    bounds = {
        "pocket_rms": (0, None),
        "pocket_dist": (0, None),
        "rmsd": (0, None),
        "tm_score": (None, 1),
        "tm_score_i": (None, 1),
    }
    lo_b, hi_b = bounds.get(metric, (None, None))
    obs_min, obs_max = float(series.min()), float(series.max())
    used_min = obs_min if lo_b is None else lo_b
    used_max = obs_max if hi_b is None else hi_b
    return round(used_max - used_min, 3)


def _basic_stats(series: pd.Series,
                 metric: str,
                 *,
                 fixed_points_all: Tuple[float, float] | None = None,
                 ) -> Dict:
    """

    """

    s = pd.to_numeric(series, errors="coerce").dropna()
    n = int(s.size)
    if n == 0:
        return {
            "mean": np.nan, "median": np.nan, "mode": np.nan, "min": np.nan, "max": np.nan,
            "std": np.nan, "variance": np.nan, "iqr": np.nan, "mad": np.nan,
            "q1": np.nan, "q3": np.nan, "skewness": np.nan, "kurtosis": np.nan,
            "range": np.nan, "range_theor": np.nan,
            "uniformity_metrics": {
                #"KS_uniformity": {"ks_statistic": np.nan, "p_value": np.nan},
                "KS_uniformity": { # previously "KS_uniformity_dual"
                    "within_group": {"ks_statistic": np.nan, "p_value": np.nan},
                    "within_all":   {"ks_statistic": np.nan, "p_value": np.nan},
                },
                "entropy": np.nan,
                "nearest_neighbor": {"mean_nnd": np.nan, "std_nnd": np.nan},
            },
        }

    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    median = s.median()
    data_min, data_max = s.min(), s.max()
    mode_vals = s.mode()
    mode_val = mode_vals.iloc[0] if not mode_vals.empty else np.nan
    mad = float(np.median(np.abs(s - median)))

    # --- KS: compute both flavors explicitly ---
    ks_dual          = coverage_ks_uniformity(s, fixed_points_all=fixed_points_all)  # Dual: All- and Group-range
    #ks_within_all   = coverage_ks_uniformity(s, fixed_points_all=fixed_points_all)  # All-range
    #ks_within_group = coverage_ks_uniformity(s, fixed_points_all=None)              # Group-range

    return {
        "mean": round(float(s.mean()), 3),
        "median": round(float(median), 3),
        "mode": round(float(mode_val), 3) if np.isfinite(mode_val) else np.nan,
        "min": round(float(data_min), 3),
        "max": round(float(data_max), 3),
        "std": round(float(s.std(ddof=1)), 3) if n > 1 else np.nan,
        "variance": round(float(s.var(ddof=1)), 3) if n > 1 else np.nan,
        "iqr": round(float(q3 - q1), 3),
        "mad": round(mad, 3),
        "q1": round(float(q1), 3),
        "q3": round(float(q3), 3),
        "skewness": round(float(s.skew()), 3) if n > 2 else np.nan,
        "kurtosis": round(float(s.kurtosis()), 3) if n > 3 else np.nan,
        "range": round(float(data_max - data_min), 3),
        "range_theor": _compute_range_theor(s, metric), # generic naive approach: round(float((data_max - data_min)), 3),
        "uniformity_metrics": {     # New: always present for Apo/Holo/All (for All, both entries coincide)
            #"KS_uniformity": ks_within_all,
            "KS_uniformity": ks_dual
                #{ # previously "KS_uniformity_dual"
                #"within_all":   ks_within_all,
                #"within_group": ks_within_group,
            #}
            ,
            # Entropy vs All-range for comparability; if None, falls back to group-range
            "entropy": coverage_entropy(s, bins=10, fixed_range=fixed_points_all),
            "nearest_neighbor": dispersion_nnd(s),
        },
    }


def _cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = float(a.std(ddof=1)), float(b.std(ddof=1))
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (float(a.mean()) - float(b.mean())) / pooled


def _overlap_interval(min1, max1, min2, max2) -> float:
    inter = max(0.0, min(max1, max2) - max(min1, min2))
    r1, r2 = (max1 - min1), (max2 - min2)
    return np.nan if min(r1, r2) == 0 else inter / min(r1, r2)


def _robust_separation(med_a, iqr_a, med_h, iqr_h) -> float:
    avg_iqr = (iqr_a + iqr_h) / 2
    return np.nan if avg_iqr == 0 else abs(med_a - med_h) / avg_iqr


import math
import numbers


def _round_scalar(x, nd=3):
    # Only round real numbers; preserve NaN/None/bools/strings
    if isinstance(x, numbers.Real) and not isinstance(x, bool):
        if isinstance(x, (float, np.floating, int, np.integer)):
            if isinstance(x, (float, np.floating)):
                if math.isfinite(x):
                    return round(float(x), nd)
                return x  # keep inf/nan as-is
            return int(x)  # ints unchanged
    return x


def _round_nested(obj, nd=3, pval_nd=None):
    """
    Recursively round floats in nested dict/list structures.
    If pval_nd is set, keys named 'p_value' get that precision.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if pval_nd is not None and k == "p_value":
                out[k] = _round_nested(v, pval_nd, pval_nd)
            else:
                out[k] = _round_nested(v, nd, pval_nd)
        return out
    elif isinstance(obj, list):
        return [_round_nested(v, nd, pval_nd) for v in obj]
    else:
        return _round_scalar(obj, nd)


def compute_absolute_statistics_v2(apo_df: pd.DataFrame,
                                   holo_df: pd.DataFrame,
                                   metrics: List[str],
                                   *,
                                   round_to=3, pvalue_round_to=6) -> Dict[str, Dict]:
    """
    Backward-compatible replacement for compute_absolute_statistics.
    Returns the same nested dict keys: 'Apo', 'Holo', 'All',
    plus 'diff_means', 'cohen_d', 'overlap_coef', 'robust_sep'.
    """
    all_df = pd.concat([apo_df, holo_df], ignore_index=True)
    out: Dict[str, Dict] = {}
    for m in metrics:
        a = pd.to_numeric(apo_df[m], errors="coerce").dropna()
        h = pd.to_numeric(holo_df[m], errors="coerce").dropna()
        al = pd.to_numeric(all_df[m], errors="coerce").dropna()
        fixed_all = (float(al.min()), float(al.max()))

        stats_apo  = _basic_stats(a, m, fixed_points_all=fixed_all)
        stats_holo = _basic_stats(h, m, fixed_points_all=fixed_all)
        stats_all  = _basic_stats(al, m, fixed_points_all=fixed_all)

        _mean_apo = stats_apo["mean"]
        _mean_holo = stats_holo["mean"]
        diff_means = round(abs(_mean_apo - _mean_holo), round_to) if np.all(np.isfinite([_mean_apo, _mean_holo])) else np.nan
        cd = _cohen_d(a, h)
        cd = round(float(cd), round_to) if np.isfinite(cd) else None
        overlap_coef = _overlap_interval(stats_apo["min"], stats_apo["max"], stats_holo["min"], stats_holo["max"]) if all(
            np.isfinite([stats_apo["min"], stats_apo["max"], stats_holo["min"], stats_holo["max"]])
        ) else np.nan
        robust_sep = _robust_separation(stats_apo["median"], stats_apo["iqr"], stats_holo["median"], stats_holo["iqr"]) if all(
            np.isfinite([stats_apo["median"], stats_apo["iqr"], stats_holo["median"], stats_holo["iqr"]])
        ) else np.nan

        out[m] = {
            "Apo": stats_apo,
            "Holo": stats_holo,
            "All": stats_all,
            "diff_means": diff_means,
            "cohen_d": cd,
            "overlap_coef": round(float(overlap_coef), round_to) if np.isfinite(overlap_coef) else np.nan,
            "robust_sep": round(float(robust_sep), round_to) if np.isfinite(robust_sep) else np.nan,
        }

        # <<< NEW: enforce consistent rounding everywhere >>>
        out[m] = _round_nested(out[m], nd=round_to, pval_nd=pvalue_round_to)

    return out
