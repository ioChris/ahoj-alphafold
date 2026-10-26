# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:32:46 2025

@author: Christos P. Feidakis
"""



## stats\_core.py (compute-only)



# stats_core.py
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Optional, Dict
import numpy as np
import pandas as pd

from run_types import APO_LABELS, HOLO_LABELS   # the pooled label sets (AHOJ-001, AHOJ-020)

# DP-GMM
import warnings
from sklearn.exceptions import ConvergenceWarning

# stats_non_kde.py
##from __future__ import annotations
##from dataclasses import dataclass
#from typing import Dict#, List, Tuple
##import numpy as np
#from sklearn.mixture import BayesianGaussianMixture

@dataclass(frozen=True)
class StatsConfig:
    """Central configuration for statistics and dispersion/coverage.

    Attributes
    ----------
    ci_method : {"percentile", "normal"}
        Method for confidence intervals.
    alpha : float
        Significance level (e.g., 0.05 for 95% CI).
    ddof : int
        Delta degrees of freedom for std (sample=1).
    min_n : int
        Minimum n required to compute dispersion/CI metrics.
    winsor_limits : tuple[float, float] | None
        Lower/upper tail proportions for winsorization, e.g., (0.01, 0.01).
    round_to : int
        Decimal places for rounded public outputs.
    nan_policy : {"omit", "propagate"}
        If "omit", drop NaNs for compute; else return NaN when any NaN is present.
    """
    ci_method: str = "percentile"
    alpha: float = 0.05
    ddof: int = 1
    min_n: int = 2
    winsor_limits: Tuple[float, float] | None = None  # e.g., (0.01, 0.01)
    round_to: int = 3
    nan_policy: str = "omit"
    # New
    #winsor_limits: tuple[float, float] | None = None  # e.g., (0.01, 0.01)
    # scaling for uniformity metrics
    scaling_mode: str = "all"  # "all" | "group"
    # KS
    ks_engine: str = "scipy"   # "scipy" | "approx"
    ks_mode: str = "dual"        # "all" | "group" | "dual"
    # Entropy
    entropy_mode: str = "discrete"  # "discrete" | "differential"
    bins_min: int = 5
    bins_max: int = 50

    def to_dict(self) -> dict:
        return {
            "ci_method": self.ci_method,
            "alpha": self.alpha,
            "ddof": self.ddof,
            "min_n": self.min_n,
            "winsor_limits": None if self.winsor_limits is None else list(self.winsor_limits),
            "round_to": self.round_to,
            "nan_policy": self.nan_policy,
        }


# ------------------------- Safe numerics -------------------------

def ensure_numeric_1d(arr: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(arr), dtype=float)
    return a[np.isfinite(a)]


def _apply_nan_policy(a: np.ndarray, nan_policy: str) -> np.ndarray | float:
    if nan_policy == "omit":
        return a[np.isfinite(a)]
    if np.any(~np.isfinite(a)):
        return np.nan
    return a


def winsorize_safe(a: np.ndarray, limits: Tuple[float, float] | None) -> np.ndarray:
    if limits is None:
        return a
    lo, hi = limits
    if lo < 0 or hi < 0 or (lo + hi) >= 1:
        return a  # ignore invalid
    lo_q, hi_q = np.nanquantile(a, [lo, 1 - hi])
    return np.clip(a, lo_q, hi_q)


def nanmean_safe(arr: Sequence[float], round_to: int | None = None, *, cfg: StatsConfig | None = None) -> float | np.nan:
    a = ensure_numeric_1d(arr)
    if cfg is not None:
        a = _apply_nan_policy(a, cfg.nan_policy)
        if isinstance(a, float):
            return np.nan
        a = winsorize_safe(a, cfg.winsor_limits)
    if a.size == 0:
        return np.nan
    out = float(np.nanmean(a))
    return round(out, round_to) if round_to is not None else out


def nanstd_safe(arr: Sequence[float], ddof: int = 1, round_to: int | None = None, *, cfg: StatsConfig | None = None) -> float | np.nan:
    a = ensure_numeric_1d(arr)
    if cfg is not None:
        a = _apply_nan_policy(a, cfg.nan_policy)
        if isinstance(a, float):
            return np.nan
        a = winsorize_safe(a, cfg.winsor_limits)
    if a.size <= ddof:
        return np.nan
    out = float(np.nanstd(a, ddof=ddof))
    return round(out, round_to) if round_to is not None else out


def nanquantile_safe(arr: Sequence[float], q: float, round_to: int | None = None, *, cfg: StatsConfig | None = None) -> float | np.nan:
    a = ensure_numeric_1d(arr)
    if cfg is not None:
        a = _apply_nan_policy(a, cfg.nan_policy)
        if isinstance(a, float):
            return np.nan
        a = winsorize_safe(a, cfg.winsor_limits)
    if a.size == 0:
        return np.nan
    out = float(np.nanquantile(a, q))
    return round(out, round_to) if round_to is not None else out


def ci_interval(arr: Sequence[float], method: str, alpha: float, cfg: StatsConfig) -> Tuple[float, float]:
    a = ensure_numeric_1d(arr)
    a = _apply_nan_policy(a, cfg.nan_policy)
    if isinstance(a, float) or a.size < cfg.min_n:
        return (np.nan, np.nan)
    a = winsorize_safe(a, cfg.winsor_limits)
    if method == "percentile":
        lo, hi = np.nanpercentile(a, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
        return (float(lo), float(hi))
    # Normal approx
    m = float(np.nanmean(a))
    s = float(np.nanstd(a, ddof=cfg.ddof))
    from math import sqrt
    z = 1.959963984540054  # approx for 95%; could compute from scipy.stats.norm if available
    half = z * s / sqrt(a.size)
    return (m - half, m + half)


# ---------------- Coverage / dispersion metrics -----------------

def coverage_cus_1(series: pd.Series,
                 intervals: Sequence[int] = (5, 10, 20),
                 thresholds: Sequence[float] = (0.01, 0.05, 0.10),
                 fixed_points: Tuple[float, float] | None = None,
                 round_to: int = 3) -> dict:
    """Compute Coverage Uniformity Score (CUS) over multiple interval counts.

    Preserves existing semantics for small-n / zero-range cases used in your
    current helper module. Returns nested dict keyed by `intervals_{n}`.
    """
    data = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    results: dict = {}

    # Small-n behavior: mimic current code
    unique_vals = np.unique(data)
    if unique_vals.size < 2:
        for n_intervals in intervals:
            interval_results = {f"{int(t*100)}%": (1.0 / len(thresholds)) for t in thresholds}
            results[f"intervals_{n_intervals}"] = interval_results
        return results

    # Range selection
    if fixed_points is not None:
        data_min, data_max = fixed_points
    else:
        data_min, data_max = float(np.nanmin(data)), float(np.nanmax(data))
    range_span = data_max - data_min

    if range_span == 0:
        for n_intervals in intervals:
            results[f"intervals_{n_intervals}"] = {f"{int(t*100)}%": np.nan for t in thresholds}
        return results

    for n_intervals in intervals:
        # Bin edges and uniform target count per bin
        edges = np.linspace(data_min, data_max, n_intervals + 1)
        counts, _ = np.histogram(data, bins=edges)
        target = data.size / n_intervals
        # Compute deviations from perfect uniformity
        diffs = np.abs(counts - target) / max(target, 1e-12)
        # Aggregate by thresholds: fraction of bins within each tolerance
        interval_results = {}
        for t in thresholds:
            within = (diffs <= t).mean() if counts.size else np.nan
            interval_results[f"{int(t*100)}%"] = round(float(within), round_to) if not np.isnan(within) else np.nan
        results[f"intervals_{n_intervals}"] = interval_results
    return results


def coverage_cus(series: pd.Series,
                 intervals: Sequence[int] = (5, 10, 20),
                 thresholds: Sequence[float] = (0.01, 0.05, 0.10),
                 fixed_points=None,
                 round_to: int = 3) -> dict:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    results = {}

    # Old behavior: <2 unique -> 1/n_intervals for every threshold
    unique_vals = np.unique(x)
    if unique_vals.size < 2:
        for n_intervals in intervals:
            results[f"intervals_{n_intervals}"] = {
                f"{int(t*100)}%": round(1 / n_intervals, round_to) for t in thresholds
            }
        return results

    # Range
    if fixed_points is not None:
        lo, hi = fixed_points
    else:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    span = hi - lo
    if span <= 0:
        for n_intervals in intervals:
            results[f"intervals_{n_intervals}"] = {f"{int(t*100)}%": np.nan for t in thresholds}
        return results

    # Old behavior: grid points + nearest absolute distance normalized by span
    for n_intervals in intervals:
        pts = np.linspace(lo, hi, n_intervals)
        # min distance from each grid point to nearest data point
        min_dists = [np.min(np.abs(x - p)) / span for p in pts]
        interval_results = {}
        for t in thresholds:
            cov = np.mean([d <= t for d in min_dists])
            interval_results[f"{int(t*100)}%"] = round(float(cov), round_to)
        results[f"intervals_{n_intervals}"] = interval_results
    return results


def _ks_against_uniform(u: np.ndarray, engine: str = "scipy") -> dict:
    """
    Run KS test against U(0,1) on already-rescaled u in [0,1].
    Returns {'ks_statistic': float, 'p_value': float}.
    """
    if u.size == 0:
        return {"ks_statistic": np.nan, "p_value": np.nan}

    if engine == "scipy":
        try:
            from scipy.stats import kstest
            ks_stat, p = kstest(u, "uniform")
            return {"ks_statistic": float(ks_stat), "p_value": float(p)}
        except Exception:
            pass  # fall through to approx

    # Approximate KS (Massey) with same keys
    u_sorted = np.sort(np.clip(u, 0.0, 1.0))
    n = u_sorted.size
    cdf = np.arange(1, n + 1) / n
    d_plus  = np.max(cdf - u_sorted)
    d_minus = np.max(u_sorted - (np.arange(0, n) / n))
    ks = float(max(d_plus, d_minus))
    from math import sqrt, exp
    lam = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * ks
    p_approx = 2 * sum(((-1)**(k-1)) * exp(-2 * (lam**2) * (k**2)) for k in range(1, 6))
    p_approx = max(min(p_approx, 1.0), 0.0)
    return {"ks_statistic": ks, "p_value": p_approx}


def coverage_ks_uniformity(
    series: pd.Series,
    fixed_points_all: tuple[float, float] | None = None,
    *,
    engine: str = "scipy",
    clip: bool = True,
    return_extras: bool = False,
) -> dict:
    """
    Return KS uniformity in two senses:
      - within_group: scale by this group's [min,max]
      - within_all:   scale by the provided All-range [min,max]; if None, falls back to group
    """
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if x.size == 0:
        base = {"ks_statistic": np.nan, "p_value": np.nan}
        dual = {"within_group": base, "within_all": base}
        return dual if not return_extras else {**dual, "n": 0}

    lo_g, hi_g = float(np.nanmin(x)), float(np.nanmax(x))
    span_g = hi_g - lo_g
    if not np.isfinite(span_g) or span_g <= 0:
        base = {"ks_statistic": np.nan, "p_value": np.nan}
        dual = {"within_group": base, "within_all": base}
        return dual if not return_extras else {**dual, "n": int(x.size), "lo_g": lo_g, "hi_g": hi_g}

    # group-scaled U
    u_g = (x - lo_g) / span_g
    if clip:
        u_g = np.clip(u_g, 0.0, 1.0)

    # all-scaled U (or fallback to group)
    if fixed_points_all is None:
        lo_a, hi_a = lo_g, hi_g
    else:
        lo_a, hi_a = map(float, fixed_points_all)
    span_a = hi_a - lo_a
    if not np.isfinite(span_a) or span_a <= 0:
        u_a = u_g
    else:
        u_a = (x - lo_a) / span_a
        if clip:
            u_a = np.clip(u_a, 0.0, 1.0)

    def _ks(u: np.ndarray) -> dict:
        if u.size == 0:
            return {"ks_statistic": np.nan, "p_value": np.nan}
        if engine == "scipy":
            try:
                from scipy.stats import kstest
                stat, p = kstest(u, "uniform")
                return {"ks_statistic": float(stat), "p_value": float(p)}
            except Exception:
                pass
        # fallback
        u_sorted = np.sort(u)
        n = u_sorted.size
        cdf = np.arange(1, n + 1) / n
        d_plus = np.max(cdf - u_sorted)
        d_minus = np.max(u_sorted - (np.arange(0, n) / n))
        ks = float(max(d_plus, d_minus))
        from math import sqrt, exp
        lam = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * ks
        p_approx = 2 * sum(((-1)**(k-1)) * exp(-2 * (lam**2) * (k**2)) for k in range(1, 6))
        p_approx = max(min(p_approx, 1.0), 0.0)
        return {"ks_statistic": ks, "p_value": p_approx}

    res = {
        "within_group": _ks(u_g),
        "within_all":   _ks(u_a),
    }
    if return_extras:
        res.update({"n": int(x.size), "lo_g": lo_g, "hi_g": hi_g, "lo_all": lo_a, "hi_all": hi_a})
    return res


def coverage_entropy(series: pd.Series, bins: int = 10, fixed_range=None) -> float:
    """Shannon entropy of binned density; returns NaN for degenerate cases."""

    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if x.size == 0:
        return np.nan

    if fixed_range is None:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    else:
        lo, hi = fixed_range
    if hi <= lo:
        return np.nan

    counts, _ = np.histogram(x, bins=bins, range=(lo, hi), density=False)
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts[counts > 0].astype(float) / float(total)
    try:
        from scipy.stats import entropy as sp_entropy
        return float(sp_entropy(p, base=2))
    except Exception:
        return float(-(p * (np.log(p) / np.log(2))).sum())


def dispersion_nnd(series: pd.Series) -> dict:
    """Nearest-neighbor distance mean/std in 1D; NaN when <2 points.

    Matches old compute_nnd_stats semantics: an N-length vector of
    per-point nearest neighbor distances, not the (N-1) adjacent gaps.
    """
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    n = x.size
    if n < 2:
        return {"mean_nnd": np.nan, "std_nnd": np.nan}

    x_sorted = np.sort(x)
    gaps = np.diff(x_sorted)
    if n == 2:
        nnd = np.array([gaps[0], gaps[0]])                # both points see the same neighbor
    else:
        left  = np.r_[np.inf, gaps]
        right = np.r_[gaps, np.inf]
        nnd = np.minimum(left, right)                      # per-point nearest neighbor

    from stats_core import nanstd_safe  # already defined in this module
    mean_nnd = float(np.mean(nnd))
    std_nnd = nanstd_safe(nnd, ddof=1)                     # guard DoF; no warnings
    return {"mean_nnd": mean_nnd, "std_nnd": std_nnd}


def dispersion_stats(series: pd.Series, cfg: StatsConfig) -> dict:
    """Return mean, std, iqr, cov, ci_low, ci_high with safe handling.
    n < cfg.min_n -> all return NaN.
    """
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if x.size < cfg.min_n:
        return {
            "mean": np.nan,
            "std": np.nan,
            "iqr": np.nan,
            "cov": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": int(x.size),
        }
    x = winsorize_safe(x, cfg.winsor_limits)
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = float(q3 - q1)
    mean = float(np.nanmean(x))
    std = float(np.nanstd(x, ddof=cfg.ddof))
    cov = float(std / mean) if mean != 0 else np.nan
    lo, hi = ci_interval(x, cfg.ci_method, cfg.alpha, cfg)
    def r(v):
        return round(v, cfg.round_to) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else v
    return {"mean": r(mean), "std": r(std), "iqr": r(iqr), "cov": r(cov), "ci_low": r(lo), "ci_high": r(hi), "n": int(x.size)}


def summarize_groups(df: pd.DataFrame, by: List[str], metrics: List[str], cfg: StatsConfig) -> pd.DataFrame:
    """Vectorized multi-metric aggregation over groupby(by). No plotting.

    Returns a DataFrame with columns like:
    <metric>_mean, <metric>_std, <metric>_iqr, <metric>_cov, <metric>_ci_low, <metric>_ci_high, <metric>_n
    preserving the naming style you already rely on downstream.
    """
    df = df.copy()
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    g = df.groupby(by, dropna=False, sort=False)

    # Precompute per-metric dicts, then assemble
    rows = []
    for keys, part in g:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(by, keys)}
        for m in metrics:
            d = dispersion_stats(part[m], cfg)
            for k, v in d.items():
                row[f"{m}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)



### Dirichlet-based functions
## stats\_non\_kde.py (compute-only, groups DP overlap together with non-KDE stats)

# stats_non_kde.py
#from __future__ import annotations
#from dataclasses import dataclass
#from typing import Dict#, List, Tuple
#import numpy as np
from sklearn.mixture import BayesianGaussianMixture


@dataclass(frozen=True)
class DPClassifyConfig:
    grid_points: int = 1000
    n_components: int = 10
    random_state: int = 0
    round_to: int = 3
    init_params: str = "kmeans"   # "kmeans" or "random" to avoid KMeans warnings
    reg_covar: float = 1e-6

    # NEW (robustness + convergence control)
    max_iter: int = 1000               # more chances to converge
    tol: float = 1e-4                  # reasonable stopping criterion
    n_init: int = 3                    # multiple inits improve stability
    jitter: float = 0.0                # e.g., 1e-8 * data_range (applied only if needed)
    std_floor: float = 1e-6            # floor for degenerate variance fallback
    suppress_warnings: bool = True     # silence sklearn ConvergenceWarning

    def to_dict(self) -> dict:
        return {
            "grid_points": self.grid_points,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "round_to": self.round_to,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "n_init": self.n_init,
            "init_params": self.init_params,
            "reg_covar": self.reg_covar,
        }


def _fit_bgm(x: np.ndarray, n_components: int, random_state: int) -> BayesianGaussianMixture:
    bgm = BayesianGaussianMixture(
        n_components=max(1, min(n_components, x.shape[0])),
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        random_state=random_state,
    )
    return bgm.fit(x)


def _fit_bgm_safe(x: np.ndarray, cfg: DPClassifyConfig) -> Optional[BayesianGaussianMixture]:
    """
    Safely fit a 1D BGM without emitting ConvergenceWarning.
    Returns a fitted model or None if data are degenerate (fallback will be used).
    """
    x = np.asarray(x, float).reshape(-1, 1)
    n = x.shape[0]
    if n < 3:
        return None

    # Detect degeneracy
    rng = float(np.nanmax(x) - np.nanmin(x)) if n else 0.0
    std = float(np.nanstd(x)) if n else 0.0
    degenerate = (not np.isfinite(std)) or (std == 0.0)

    X = x.copy()
    if degenerate:
        # Optionally apply a tiny jitter only if there is range ~ 0
        if cfg.jitter and rng == 0.0:
            X = X + (cfg.jitter if np.isscalar(cfg.jitter) else 0.0)
        else:
            # Let caller use a parametric fallback instead of forcing a noisy fit
            return None

    n_components = max(1, min(cfg.n_components, n // 2 if n >= 6 else 1))
    params = dict(
        n_components=n_components,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        init_params=cfg.init_params,
        reg_covar=cfg.reg_covar,
        random_state=cfg.random_state,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        n_init=cfg.n_init,
    )

    if cfg.suppress_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return BayesianGaussianMixture(**params).fit(X)
    else:
        return BayesianGaussianMixture(**params).fit(X)


def _density(bgm: BayesianGaussianMixture, grid: np.ndarray) -> np.ndarray:
    return np.exp(bgm.score_samples(grid))


def _integral_overlap(x: np.ndarray, f_a: np.ndarray, f_h: np.ndarray) -> float:
    # Trapezoidal integral of min densities over 1D grid
    m = np.minimum(f_a, f_h)
    return float(np.trapz(m, x))


def dp_classify_with_overlap(
    apo_values: np.ndarray,
    holo_values: np.ndarray,
    new_point: float,
    cfg: DPClassifyConfig,
    return_curves: bool = False,
) -> Tuple[str, float, float, float]:
    """
    Return (classification, score, impact, overlap) using robust PDFs.
    If return_curves=True, append a 5th element: {"x": xs, "apo": f_a, "holo": f_h} for the exact
    densities used to classify (or None for the degenerate single-/no-group short-circuits), so the
    DP plot can draw precisely the curve that drove the classification (AHOJ-013 fidelity reuse).
    Uses evaluate_group_pdf() (BGM if possible, otherwise Normal fallback),
    so no direct BGM fitting happens here (avoids ConvergenceWarning).

    * classification: "Apo" if f_A(new_point) > f_H(new_point) else "Holo".
    * score: (f_H - f_A) / (f_A + f_H)  (keeps your existing polarity)
    * impact: |f_H - f_A|
    * overlap: ∫ min(f_A, f_H) dx on a padded grid spanning observed ranges.
    """
    # Clean inputs: keep only finite values
    a = ensure_numeric_1d(apo_values)
    h = ensure_numeric_1d(holo_values)

    # If no data at all, or new_point non-finite → cannot classify
    if (a.size == 0 and h.size == 0) or not np.isfinite(new_point):
        return ("Insufficient Data", np.nan, np.nan, np.nan) + ((None,) if return_curves else ())

    # If only one side has data, short-circuit without trying to fit the other
    if a.size == 0:
        return ("Holo", 1.0, np.nan, 0.0) + ((None,) if return_curves else ())
    if h.size == 0:
        return ("Apo", -1.0, np.nan, 0.0) + ((None,) if return_curves else ())

    # Determine grid bounds from observed ranges (pad a bit)
    lo = np.nanmin([a.min(), h.min()])
    hi = np.nanmax([a.max(), h.max()])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        # Small symmetric window around combined mean if degenerate
        mu = np.nanmean(np.concatenate([a, h]))
        width = 1.0
        lo, hi = mu - width, mu + width

    pad = 0.10 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, cfg.grid_points)

    # Robust group densities (BGM if viable, Normal fallback otherwise)
    f_a = evaluate_group_pdf(a, xs, cfg)
    f_h = evaluate_group_pdf(h, xs, cfg)

    # Density values at the classification point (via interpolation)
    z = float(new_point)
    fa_z = float(np.interp(z, xs, f_a, left=f_a[0], right=f_a[-1]))
    fh_z = float(np.interp(z, xs, f_h, left=f_h[0], right=f_h[-1]))

    # Classification and scalar summaries
    classification = "Apo" if fa_z > fh_z else "Holo"
    denom = fa_z + fh_z
    score = (fh_z - fa_z) / denom if denom != 0 else 0.0
    impact = abs(fh_z - fa_z)
    overlap = float(np.trapz(np.minimum(f_a, f_h), xs))

    # Round like before (respect cfg.round_to)
    r = (
        lambda v: round(v, cfg.round_to)
        if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
        else v
    )
    base = (classification, r(score), r(impact), r(overlap))
    if return_curves:
        return base + ({"x": xs, "apo": f_a, "holo": f_h},)
    return base


def dp_classify_new_point(df, metrics: List[str], new_point_df, cfg: DPClassifyConfig | None = None,
                          return_curves: bool = False) -> Dict:
    """
    Classify a new point across multiple metrics using DP overlap.
    Now robust to NaNs in source data and the new point row.

    Parameters
    ----------
    df : DataFrame
        Must contain columns for each metric and 'apoholo_assignment'; rows in run_types.APO_LABELS
        form the apo set and rows in run_types.HOLO_LABELS the holo set.
    metrics : list[str]
        Metric names to evaluate.
    new_point_df : DataFrame
        Single-row DF with the same metric columns representing the classification point.
    cfg : DPClassifyConfig

    Returns
    -------
    dict with per-metric results (classification, score, impact, overlap) and an
    overall block: avg_score, avg_impact, avg_overlap, overall_classification.
    """
    if cfg is None:
        cfg = DPClassifyConfig()

    res_per_metric: Dict[str, Dict] = {}
    curves_per_metric: Dict[str, Optional[dict]] = {}
    scores, impacts, overlaps = [], [], []

    mask_apo = df["apoholo_assignment"].isin(APO_LABELS)  # AHOJ-001 (A**), AHOJ-020 (A-) count as apo
    mask_holo = df["apoholo_assignment"].isin(HOLO_LABELS)

    for m in metrics:
        # Coerce to numeric and drop NaNs before modeling
        apo_vals = pd.to_numeric(df.loc[mask_apo, m], errors="coerce").dropna().to_numpy()
        holo_vals = pd.to_numeric(df.loc[mask_holo, m], errors="coerce").dropna().to_numpy()

        # Robustly extract the new value
        new_raw = new_point_df[m].iloc[0] if (m in new_point_df.columns and len(new_point_df) == 1) else np.nan
        try:
            new_val = float(new_raw)
        except Exception:
            new_val = np.nan

        if return_curves:
            cls, score, impact, ov, curves = dp_classify_with_overlap(
                apo_vals, holo_vals, new_val, cfg, return_curves=True)
            curves_per_metric[m] = curves
        else:
            cls, score, impact, ov = dp_classify_with_overlap(apo_vals, holo_vals, new_val, cfg)
        res_per_metric[m] = {"classification": cls, "score": score, "impact": impact, "overlap": ov}

        if isinstance(score, (int, float)) and np.isfinite(score):   scores.append(score)
        if isinstance(impact, (int, float)) and np.isfinite(impact): impacts.append(impact)
        if isinstance(ov, (int, float)) and np.isfinite(ov):         overlaps.append(ov)

    def avg_or_nan(x: List[float]):
        return float(np.mean(x)) if len(x) else np.nan

    avg_score = avg_or_nan(scores)
    avg_impact = avg_or_nan(impacts)
    avg_overlap = avg_or_nan(overlaps)

    if not np.isfinite(avg_score):
        overall_cls = "Insufficient Data"
    elif avg_score > 0:
        overall_cls = "Holo"
    elif avg_score < 0:
        overall_cls = "Apo"
    else:
        overall_cls = "Ambiguous"

    r = lambda v: round(v, cfg.round_to) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else v
    result = {
        "per_metric": res_per_metric,
        "overall": {
            "avg_score": r(avg_score),
            "avg_impact": r(avg_impact),
            "avg_overlap": r(avg_overlap),
            "overall_classification": overall_cls,
        },
    }
    if return_curves:
        return result, curves_per_metric
    return result



### Z-score, Robust statistics, Composite distance based point classifications


def classify_point_modified_z(df_point, comparisons, metrics):
    """
    Classify a single data point (provided as a one-row DataFrame) based on its proximity 
    to the Apo and Holo distributions for the specified metrics, using a modified z-score
    that is normalized by the overall range of the metric (combined Apo and Holo range).
    
    For each metric, the function computes:
      modified_z_apo = |value - Apo_mean| / overall_range
      modified_z_holo = |value - Holo_mean| / overall_range
    where overall_range = max(Apo_max, Holo_max) - min(Apo_min, Holo_min)
    
    The distances are then aggregated across metrics and the point is assigned to the group
    with the lower average modified z-score.
    
    Parameters:
    - df_point: a pandas DataFrame containing a single row of metric values.
    - comparisons: dictionary containing descriptive statistics for each metric for Apo and Holo groups.
    - metrics: list of metric column names.
    
    Returns:
    - A dictionary containing:
         - classification: the assigned group ("Apo" or "Holo")
         - avg_distance_apo: aggregated modified z-score to the Apo distribution.
         - avg_distance_holo: aggregated modified z-score to the Holo distribution.
         - score: the absolute difference between the aggregated distances.
         - per_metric: dictionary with detailed modified z-scores for each metric.
    """
    total_distance_apo = 0.0
    total_distance_holo = 0.0
    per_metric_details = {}
    
    if df_point is None or df_point.empty:
        return dict()
    
    for metric in metrics:
        
        if df_point is not None and not df_point.empty and metric in df_point.columns:

            # Extract the data point value (assuming a single-row DataFrame)
            value = df_point[metric].iloc[0]
            
            # Get the Apo and Holo statistics for the metric.
            apo_stats = comparisons[metric]['Apo']
            holo_stats = comparisons[metric]['Holo']
            apo_mean = apo_stats['mean']
            holo_mean = holo_stats['mean']
            
            # Compute the overall range from both groups.
            overall_min = min(apo_stats['min'], holo_stats['min'])
            overall_max = max(apo_stats['max'], holo_stats['max'])
            overall_range = overall_max - overall_min
            if overall_range == 0:
                overall_range = 1.0  # Avoid division by zero.
            
            # Compute modified z-scores for each group.
            z_apo = abs(value - apo_mean) / overall_range
            z_holo = abs(value - holo_mean) / overall_range
            
            per_metric_details[metric] = {
                'value': value,
                'z_apo': round(z_apo, 3),
                'z_holo': round(z_holo, 3)
            }
            
            total_distance_apo += z_apo
            total_distance_holo += z_holo
        
    # Average the distances across all metrics.
    avg_distance_apo = total_distance_apo / len(metrics)
    avg_distance_holo = total_distance_holo / len(metrics)
    
    # Classify based on the lower average modified z-score.
    classification = "Apo" if avg_distance_apo < avg_distance_holo else "Holo"
    score = abs(avg_distance_apo - avg_distance_holo)
    
    result = {
        'classification': classification,
        'avg_distance_apo': round(avg_distance_apo, 3),
        'avg_distance_holo': round(avg_distance_holo, 3),
        'score': round(score, 3),
        'per_metric': per_metric_details
    }
    
    return result


def robust_distance(value, stats, overall_range):
    """
    Calculate a robust distance between a value and a group's stats,
    using both the mean and median, normalized by the overall range 
    (across both groups).

    Parameters:
    - value: the value from the new data point.
    - stats: dictionary containing 'mean' and 'median' for a group.
    - overall_range: the overall range for the metric, defined as 
                     max(apo_max, holo_max) - min(apo_min, holo_min).
                     
    Returns:
    - Robust distance as a float.
    """
    # Avoid division by zero.
    if overall_range == 0:
        overall_range = 1.0
    return (abs(value - stats['mean']) + abs(value - stats['median'])) / (2 * overall_range)


def classify_point_robust(df_point, comparisons, metrics):
    """
    Classify a single data point (in a one-row DataFrame) using a robust distance metric
    that combines the mean and median differences, normalized by the overall range of the metric.

    For each metric, the function calculates the robust distance for the Apo and Holo groups
    using the overall range (from both groups) as the normalization factor.
    The distances are then averaged across metrics, and the group with the lower average distance
    is considered the better match.

    Parameters:
    - df_point: a pandas DataFrame containing a single row of metric values.
    - comparisons: a dictionary containing group statistics for each metric.
    - metrics: list of metric column names.

    Returns:
    - A dictionary containing:
         - classification: the assigned group ("Apo" or "Holo")
         - avg_distance_apo: the average robust distance (normalized) to the Apo distribution.
         - avg_distance_holo: the average robust distance (normalized) to the Holo distribution.
         - score: the absolute difference between the aggregated distances.
         - per_metric: detailed robust distances for each metric.
    """
    total_distance_apo = 0.0
    total_distance_holo = 0.0
    per_metric_details = {}

    if df_point is None or df_point.empty:
        return dict()

    for metric in metrics:
        # Get the value from the data point.
        value = df_point[metric].iloc[0]
        # Get group stats from the comparisons dictionary.
        stats_apo = comparisons[metric]['Apo']
        stats_holo = comparisons[metric]['Holo']

        # Compute the overall range for this metric from both groups.
        overall_range = max(stats_apo['max'], stats_holo['max']) - min(stats_apo['min'], stats_holo['min'])
        if overall_range == 0:
            overall_range = 1.0  # Avoid division by zero.

        # Compute robust distances for each group using the overall range.
        dist_apo = robust_distance(value, stats_apo, overall_range)
        dist_holo = robust_distance(value, stats_holo, overall_range)

        per_metric_details[metric] = {
            'value': value,
            'robust_distance_apo': round(dist_apo, 3),
            'robust_distance_holo': round(dist_holo, 3)
        }

        total_distance_apo += dist_apo
        total_distance_holo += dist_holo

    # Average distances over all metrics.
    avg_distance_apo = total_distance_apo / len(metrics)
    avg_distance_holo = total_distance_holo / len(metrics)

    # Classify the point by the group with the lower average robust distance.
    classification = "Apo" if avg_distance_apo < avg_distance_holo else "Holo"
    score = abs(avg_distance_apo - avg_distance_holo)
    # Get the score distance as a percentage of the separation of the two groups
    #percentage = point_score_percentage(score, group_separation_score_avg)

    return {
        'classification': classification,
        'avg_distance_apo': round(avg_distance_apo, 3),
        'avg_distance_holo': round(avg_distance_holo, 3),
        'score': round(score, 3),
        'per_metric': per_metric_details
    }


def composite_distance(value, stats, w_center=0.5, w_iqr=0.3, w_extremes=0.2, normalize=True, norm_factor=None):
    """
    Calculate a composite distance between a value and a group's summary statistics,
    incorporating differences from the mean, median, min, max, and IQR.
    
    If normalize is True, the raw composite distance is divided by a normalization factor.
    Here, norm_factor is intended to be the overall range of the metric across both groups.
    
    Parameters:
    - value: The value from the new data point.
    - stats: A dictionary containing 'mean', 'median', 'min', 'max', 'q1', 'q3'.
    - w_center: Weight for the center differences (mean and median).
    - w_iqr: Weight for the IQR deviation.
    - w_extremes: Weight for the extreme differences (min and max).
    - normalize: Boolean indicating whether to normalize.
    - norm_factor: The normalization factor to use. If None, defaults to the group's own range.
    
    Returns:
    - The (optionally normalized) composite distance as a float.
    """
    diff_mean = abs(value - stats['mean'])
    diff_median = abs(value - stats['median'])
    diff_center = (diff_mean + diff_median) / 2.0
    
    # For extremes:
    # Original: average absolute difference from min and max.
    diff_extremes_original = (abs(value - stats['min']) + abs(value - stats['max'])) / 2.0
    # Center-based: difference from the midrange.
    extremes_center = (stats['min'] + stats['max']) / 2.0
    diff_extremes_center = abs(value - extremes_center)
    # Combined extremes: average of both methods.
    diff_extremes = (diff_extremes_original + diff_extremes_center) / 2.0

    # For IQR:
    # Original: if outside the IQR, take the distance to the nearest quartile; else 0.
    if value < stats['q1']:
        diff_iqr_original = stats['q1'] - value
    elif value > stats['q3']:
        diff_iqr_original = value - stats['q3']
    else:
        diff_iqr_original = 0.0
    # Center-based: distance from the center of the IQR.
    iqr_center = (stats['q1'] + stats['q3']) / 2.0
    diff_iqr_center = abs(value - iqr_center)
    # Combined IQR: average of both methods.
    diff_iqr = (diff_iqr_original + diff_iqr_center) / 2.0

    raw_score = w_center * diff_center + w_extremes * diff_extremes + w_iqr * diff_iqr

    if normalize:
        # Use the provided norm_factor if available, otherwise fallback to the group's own range.
        if norm_factor is None:
            norm_factor = stats['max'] - stats['min']
        if norm_factor == 0:
            norm_factor = 1.0  # Avoid division by zero.
        return raw_score / norm_factor
    else:
        return raw_score


def assign_classification(avg_distance_apo, avg_distance_holo):
    """
    Assign classification and score based on the average distances.
    
    If one group is missing (i.e. its avg_distance is NaN), the classification is 
    assigned to the available group and the score is set to that group's distance.
    Otherwise, the classification is based on the lower average distance and the 
    score is the absolute difference.
    """
    if np.isnan(avg_distance_apo) and not np.isnan(avg_distance_holo):
        classification, score = "Holo", avg_distance_holo
    elif np.isnan(avg_distance_holo) and not np.isnan(avg_distance_apo):
        classification, score = "Apo", avg_distance_apo
    elif np.isnan(avg_distance_apo) and np.isnan(avg_distance_holo):
        classification, score = None, np.nan
    else:
        classification = "Apo" if avg_distance_apo < avg_distance_holo else "Holo"
        score = abs(avg_distance_apo - avg_distance_holo)
    return classification, score


def classify_point_composite(df_point, comparisons, metrics, w_center=0.5, w_iqr=0.3, w_extremes=0.2):
    """
    Classify a single data point (a one-row DataFrame) using a composite distance metric.
    
    For each metric, this function calculates the composite distance to the Apo, Holo,
    and All groups (using differences from the mean, median, min, max, and IQR). 
    The composite distances are normalized using the overall range of the respective group.
    
    If one group is missing (e.g. no Apo data), its composite distance is set to np.nan
    so that it does not erroneously count as an absolute match.
    
    Parameters:
      - df_point: A pandas DataFrame containing a single row of metric values.
      - comparisons: A dictionary containing group statistics for each metric
                     with sub-dictionaries for 'Apo', 'Holo', and 'All' (the combined group).
                     Each group's stats should include 'mean', 'median', 'min', 'max', 'q1', and 'q3'.
      - metrics: List of metric column names.
      - w_center: Weight for the center differences.
      - w_iqr: Weight for the IQR deviation.
      - w_extremes: Weight for the extreme differences.
    
    Returns:
      A dictionary containing:
         - classification: The assigned group ("Apo" or "Holo") based on the lower average distance.
         - avg_distance_apo: The average composite distance to the Apo distribution.
         - avg_distance_holo: The average composite distance to the Holo distribution.
         - avg_distance_all: The average composite distance to the All distribution.
         - score: The absolute difference between the averaged Apo and Holo distances.
         - per_metric: Detailed composite distances for each metric (including 'composite_distance_all').
    """

    total_distance_apo_list = []
    total_distance_holo_list = []
    total_distance_all_list = []
    per_metric_details = {}

    if df_point is None or df_point.empty:
        return dict()

    for metric in metrics:

        # robustly get the value; NaN if missing/non-numeric
        value = pd.to_numeric(df_point.get(metric, pd.Series([np.nan])).iloc[0], errors="coerce")
        #value = df_point[metric].iloc[0]

        if not np.isfinite(value):
            per_metric_details[metric] = {
                'value': None,
                'composite_distance_apo': None,
                'composite_distance_holo': None,
                'composite_distance_all': None,
            }
            total_distance_apo_list.append(np.nan)
            total_distance_holo_list.append(np.nan)
            total_distance_all_list.append(np.nan)
            continue

        stats_apo = comparisons[metric]['Apo']
        stats_holo = comparisons[metric]['Holo']
        stats_all = comparisons[metric]['All']

        # Determine overall range for Apo/Holo.
        if np.isnan(stats_apo['mean']):
            # Only Holo available.
            overall_range = stats_holo['max'] - stats_holo['min']
            dist_apo = np.nan
            dist_holo = composite_distance(value, stats_holo, w_center, w_iqr, w_extremes,
                                           normalize=True, norm_factor=overall_range)
        elif np.isnan(stats_holo['mean']):
            # Only Apo available.
            overall_range = stats_apo['max'] - stats_apo['min']
            dist_apo = composite_distance(value, stats_apo, w_center, w_iqr, w_extremes,
                                          normalize=True, norm_factor=overall_range)
            dist_holo = np.nan
        else:
            overall_range = max(stats_apo['max'], stats_holo['max']) - min(stats_apo['min'], stats_holo['min'])
            if overall_range == 0:
                overall_range = 1.0
            dist_apo = composite_distance(value, stats_apo, w_center, w_iqr, w_extremes,
                                          normalize=True, norm_factor=overall_range)
            dist_holo = composite_distance(value, stats_holo, w_center, w_iqr, w_extremes,
                                           normalize=True, norm_factor=overall_range)

        # Compute distance for the "All" group using its own range.
        overall_range_all = stats_all['max'] - stats_all['min']
        if overall_range_all == 0:
            overall_range_all = 1.0
        dist_all = composite_distance(value, stats_all, w_center, w_iqr, w_extremes,
                                      normalize=True, norm_factor=overall_range_all)

        per_metric_details[metric] = {
            'value': value,
            'composite_distance_apo': round(dist_apo, 3) if not np.isnan(dist_apo) else None,
            'composite_distance_holo': round(dist_holo, 3) if not np.isnan(dist_holo) else None,
            'composite_distance_all': round(dist_all, 3)
        }

        total_distance_apo_list.append(dist_apo)
        total_distance_holo_list.append(dist_holo)
        total_distance_all_list.append(dist_all)

    # Compute averages using np.nanmean so that np.nan values are ignored.
    avg_distance_apo = np.nan if np.all(np.isnan(total_distance_apo_list)) else np.nanmean(total_distance_apo_list)
    avg_distance_holo = np.nan if np.all(np.isnan(total_distance_holo_list)) else np.nanmean(total_distance_holo_list)
    avg_distance_all = np.nan if np.all(np.isnan(total_distance_all_list)) else np.nanmean(total_distance_all_list)

    classification, score = assign_classification(avg_distance_apo, avg_distance_holo)

    return {
        'classification': classification,
        'avg_distance_apo': round(avg_distance_apo, 3) if not np.isnan(avg_distance_apo) else None,
        'avg_distance_holo': round(avg_distance_holo, 3) if not np.isnan(avg_distance_holo) else None,
        'avg_distance_all': round(avg_distance_all, 3) if not np.isnan(avg_distance_all) else None,
        'score': round(score, 3) if not np.isnan(score) else None,
        'per_metric': per_metric_details
        }


## Group separation

def group_separation_score(stats_apo, stats_holo, weights=None, normalize=True):
    """
    Compute a separation score between the Apo and Holo groups for a single metric
    based on summary statistics, with an option to normalize by the overall range.

    The score is calculated as a weighted sum of:
      - Center differences: average absolute difference of the means and medians.
      - Extreme differences: average absolute difference of the minimums and maximums.
      - IQR differences: average absolute difference of the Q1 and Q3 values.
      
    If normalize is True, the raw score is divided by:
    
      overall_range = max(Apo_max, Holo_max) - min(Apo_min, Holo_min)
      
    Parameters:
    - stats_apo: Dictionary with keys 'mean', 'median', 'min', 'max', 'q1', 'q3' for the Apo group.
    - stats_holo: Dictionary with the same keys for the Holo group.
    - w_center: Weight for the central differences.
    - w_iqr: Weight for the IQR differences.
    - w_extremes: Weight for the extreme differences.
    - normalize: Whether to normalize the result by the overall range.
    
    Returns:
    - A separation score (float); a higher value indicates greater separation.
    """
    # If weights are provided, normalize them so that their sum is 1.
    if weights is not None:
        w_center = weights.get('center', 0)
        w_extremes = weights.get('extremes', 0)
        w_iqr = weights.get('iqr', 0)
        total_weight = w_center + w_extremes + w_iqr
        if total_weight > 0:
            w_center /= total_weight
            w_extremes /= total_weight
            w_iqr /= total_weight
        else:
            weights = None  # fallback if weights sum to 0
    
    diff_center = (abs(stats_apo['mean'] - stats_holo['mean']) + abs(stats_apo['median'] - stats_holo['median'])) / 2.0
    diff_extremes = (abs(stats_apo['min'] - stats_holo['min']) + abs(stats_apo['max'] - stats_holo['max'])) / 2.0
    diff_iqr = (abs(stats_apo['q1'] - stats_holo['q1']) + abs(stats_apo['q3'] - stats_holo['q3'])) / 2.0
    
    raw_score = w_center * diff_center + w_extremes * diff_extremes + w_iqr * diff_iqr
    
    if normalize:
        overall_range = max(stats_apo['max'], stats_holo['max']) - min(stats_apo['min'], stats_holo['min'])
        if overall_range == 0:
            overall_range = 1.0  # Avoid division by zero.
        return raw_score / overall_range
    else:
        return raw_score


def overall_group_separation(comparisons, metrics, weights=None, normalize=True):
    """
    Compute an overall separation score between the Apo and Holo groups across a list of metrics.
    
    For each metric, this function computes a normalized separation score using the weighted differences
    of the summary statistics. It then returns both a per-metric dictionary and an average separation score.
    
    Parameters:
    - comparisons: Dictionary with per-metric group statistics.
    - metrics: List of metric names.
    - w_center, w_extremes, w_iqr: Weights for the corresponding components.
    - normalize: Whether to normalize each metric's separation score by its overall range.
    
    Returns:
    - A dictionary with:
         'per_metric': {metric: separation score, ...}
         'average_separation': average separation score across all metrics.
    """
    separation_scores = {}
    total_score = 0.0
    
    for metric in metrics:
        stats_apo = comparisons[metric]['Apo']
        stats_holo = comparisons[metric]['Holo']
        score = group_separation_score(stats_apo, stats_holo, weights, normalize)
        separation_scores[metric] = round(score, 3)
        total_score += score
        
    avg_separation = total_score / len(metrics)
    return {
        'per_metric': separation_scores,
        'average_separation': round(avg_separation, 3)
    }


def overall_group_overlap(comparisons, metrics, weights=None):
    """
    Compute the overall average overlap of the Apo and Holo groups across the given metrics.
    
    If a weights dictionary is provided with keys 'center', 'extremes', and 'iqr', then for each metric:
      - Extreme overlap is calculated as:
            extreme_overlap = (max(0, min(apo_max, holo_max) - max(apo_min, holo_min))) / 
                              (max(apo_max, holo_max) - min(apo_min, holo_min))
      - Center overlap is calculated using the centers (average of mean and median):
            center_overlap = max(0, 1 - (abs(center_apo - center_holo) / overall_range))
      - IQR overlap is calculated as the overlap between the interquartile ranges:
            iqr_overlap = (max(0, min(apo_q3, holo_q3) - max(apo_q1, holo_q1))) / (max(apo_q3, holo_q3) - min(apo_q1, holo_q1))
            
    The final weighted overlap for a metric is:
    
        weighted_overlap = (w_extremes * extreme_overlap + w_center * center_overlap + w_iqr * iqr_overlap)
                           / (w_extremes + w_center + w_iqr)
                           
    If no weights are provided, the function uses only the extreme overlap as in the simple min–max method.
    
    Parameters:
    - comparisons: Dictionary with per-metric group statistics, which should
                   include for each group: 'min', 'max', 'mean', 'median', 'q1', 'q3'.
    - metrics: List of metric names.
    - weights: Optional dictionary with keys 'center', 'extremes', 'iqr'. If provided, these weights
               are used to compute a weighted overlap.
    
    Returns:
    - A dictionary with:
         'per_metric': {metric: overlap_ratio, ...}
         'average_overlap': average overlap ratio across all metrics.
    """
    per_metric_overlap = {}
    total_overlap = 0.0
    count = 0

    # If weights are provided, normalize them so that their sum is 1.
    if weights is not None:
        w_center = weights.get('center', 0)
        w_extremes = weights.get('extremes', 0)
        w_iqr = weights.get('iqr', 0)
        total_weight = w_center + w_extremes + w_iqr
        if total_weight > 0:
            w_center /= total_weight
            w_extremes /= total_weight
            w_iqr /= total_weight
        else:
            weights = None  # fallback if weights sum to 0

    for metric in metrics:
        stats_apo = comparisons[metric]['Apo']
        stats_holo = comparisons[metric]['Holo']
        
        # Check if one group's extreme values are missing; if so, set overlap to 0.
        if np.isnan(stats_apo['min']) or np.isnan(stats_holo['min']):
            overlap_ratio = np.nan  # missing group → ignore in averaging # overlap_ratio = 0.0
        else:
            # Overall extreme range for the metric
            overall_min = min(stats_apo['min'], stats_holo['min'])
            overall_max = max(stats_apo['max'], stats_holo['max'])
            overall_range = overall_max - overall_min
            
            # Extreme overlap (min–max based)
            if overall_range == 0:
                extreme_overlap = 1.0
            else:
                intersection = max(0, min(stats_apo['max'], stats_holo['max']) - max(stats_apo['min'], stats_holo['min']))
                extreme_overlap = intersection / overall_range

            if weights is None:
                # Use only the extreme overlap
                overlap_ratio = extreme_overlap
            else:
                # Center overlap: using average of mean and median for each group.
                center_apo = (stats_apo['mean'] + stats_apo['median']) / 2.0
                center_holo = (stats_holo['mean'] + stats_holo['median']) / 2.0
                if overall_range == 0:
                    center_overlap = 1.0
                else:
                    center_diff = abs(center_apo - center_holo)
                    center_overlap = max(0, 1 - (center_diff / overall_range))
                
                # IQR overlap: compute the overall IQR range from Q1 and Q3.
                overall_q1 = min(stats_apo['q1'], stats_holo['q1'])
                overall_q3 = max(stats_apo['q3'], stats_holo['q3'])
                overall_iqr_range = overall_q3 - overall_q1
                if overall_iqr_range == 0:
                    iqr_overlap = 1.0
                else:
                    intersection_iqr = max(0, min(stats_apo['q3'], stats_holo['q3']) - max(stats_apo['q1'], stats_holo['q1']))
                    iqr_overlap = intersection_iqr / overall_iqr_range
                
                # Weighted average of the three overlap components.
                overlap_ratio = w_extremes * extreme_overlap + w_center * center_overlap + w_iqr * iqr_overlap

        per_metric_overlap[metric] = round(overlap_ratio, 3) if np.isfinite(overlap_ratio) else np.nan
        if np.isfinite(overlap_ratio):
            total_overlap += overlap_ratio
            count += 1
        #per_metric_overlap[metric] = round(overlap_ratio, 3)
        #total_overlap += overlap_ratio
        #count += 1
    
    average_overlap = total_overlap / count if count > 0 else np.nan
    #average_overlap = total_overlap / count if count > 0 else 0.0
    return {
         'per_metric': per_metric_overlap,
         'average_overlap': round(average_overlap, 3) if np.isfinite(average_overlap) else np.nan
     }


def group_separation_score_robust(stats_apo, stats_holo, weights=None, normalize=True):
    """
    Compute a robust separation score between the Apo and Holo groups for a single metric
    using summary statistics and normalizing by the overall range.

    Parameters:
    - stats_apo: Dictionary with keys 'mean', 'median', 'min', 'max', 'q1', 'q3' for the Apo group.
    - stats_holo: Dictionary with the same keys for the Holo group.
    - w_center: Weight for the center differences (mean and median).
    - w_extremes: Weight for the extreme differences (min and max).
    - w_iqr: Weight for the IQR differences (q1 and q3).
    - normalize: If True, divide the raw score by the overall range.

    Returns:
    - A separation score (float), where a higher value indicates greater separation.
    """
    # Normalize weights if provided.
    if weights is not None:
        w_center = weights.get('center', 0)
        w_extremes = weights.get('extremes', 0)
        w_iqr = weights.get('iqr', 0)
        total_weight = w_center + w_extremes + w_iqr
        if total_weight > 0:
            w_center /= total_weight
            w_extremes /= total_weight
            w_iqr /= total_weight
        else:
            weights = None

    # If one group's values are missing, use the available group's stats for range computation.
    if np.isnan(stats_apo['min']) or np.isnan(stats_apo['max']):
        overall_min = stats_holo['min']
        overall_max = stats_holo['max']
    elif np.isnan(stats_holo['min']) or np.isnan(stats_holo['max']):
        overall_min = stats_apo['min']
        overall_max = stats_apo['max']
    else:
        overall_min = min(stats_apo['min'], stats_holo['min'])
        overall_max = max(stats_apo['max'], stats_holo['max'])
    overall_range = overall_max - overall_min
    if overall_range == 0:
        overall_range = 1.0  # Avoid division by zero.
    
    # Compute differences.
    diff_center = (abs(stats_apo['mean'] - stats_holo['mean']) + abs(stats_apo['median'] - stats_holo['median'])) / 2.0
    diff_extremes = (abs(stats_apo['min'] - stats_holo['min']) + abs(stats_apo['max'] - stats_holo['max'])) / 2.0
    diff_iqr = (abs(stats_apo['q1'] - stats_holo['q1']) + abs(stats_apo['q3'] - stats_holo['q3'])) / 2.0
    
    raw_score = w_center * diff_center + w_extremes * diff_extremes + w_iqr * diff_iqr
    if normalize:
        return raw_score / overall_range
    else:
        return raw_score


def overall_group_separation_robust(comparisons, metrics, weights=None, normalize=True):
    """
    Compute an overall robust separation score between the Apo and Holo groups across a list of metrics.
    
    For each metric, this function uses group_separation_score_robust to compute a normalized separation score.
    It returns both a per-metric dictionary and an average separation score.
    
    Parameters:
    - comparisons: Dictionary with per-metric group statistics.
    - metrics: List of metric names.
    - w_center, w_extremes, w_iqr: Weights for the corresponding components.
    - normalize: Whether to normalize each metric's separation score by its overall range.
    
    Returns:
    - A dictionary with:
         'per_metric': {metric: separation score, ...}
         'average_separation': average separation score across all metrics.
    """
    separation_scores = {}
    total_score = 0.0
    
    for metric in metrics:
        stats_apo = comparisons[metric]['Apo']
        stats_holo = comparisons[metric]['Holo']
        score = group_separation_score_robust(stats_apo, stats_holo, weights, normalize)
        separation_scores[metric] = round(score, 3)
        total_score += score
        
    avg_separation = total_score / len(metrics)
    return {
        'per_metric': separation_scores,
        'average_separation': round(avg_separation, 3)
    }


def overall_group_overlap_robust(comparisons, metrics, weights=None):
    """
    Compute the overall robust average overlap of the Apo and Holo groups across the given metrics.
    
    If a weights dictionary is provided with keys 'center', 'extremes', and 'iqr', then for each metric:
      - Extreme overlap is defined as:
            extreme_overlap = (max(0, min(apo_max, holo_max) - max(apo_min, holo_min))) / 
                              overall_range
      - Center overlap is defined using the centers (average of mean and median):
            center_overlap = max(0, 1 - (abs(center_apo - center_holo) / overall_range))
      - IQR overlap is defined as the overlap between the interquartile ranges:
            iqr_overlap = (max(0, min(apo_q3, holo_q3) - max(apo_q1, holo_q1))) / overall_iqr_range
    The final weighted overlap for a metric is the weighted average of these components.
    
    If no weights are provided, the function uses only the extreme overlap as in the simple min–max method.
    
    Parameters:
    - comparisons: Dictionary with per-metric group statistics including
                   'min', 'max', 'mean', 'median', 'q1', 'q3'.
    - metrics: List of metric names.
    - weights: Optional dictionary with keys 'center', 'extremes', and 'iqr'. If provided,
               these weights (normalized to sum to 1) are used to compute a weighted overlap.
    
    Returns:
    - A dictionary with:
         'per_metric': {metric: overlap ratio, ...}
         'average_overlap': average overlap ratio across all metrics.
    """
    per_metric_overlap = {}
    total_overlap = 0.0
    count = 0

    # Normalize weights if provided.
    if weights is not None:
        w_center = weights.get('center', 0)
        w_extremes = weights.get('extremes', 0)
        w_iqr = weights.get('iqr', 0)
        total_weight = w_center + w_extremes + w_iqr
        if total_weight > 0:
            w_center /= total_weight
            w_extremes /= total_weight
            w_iqr /= total_weight
        else:
            weights = None

    for metric in metrics:
        stats_apo = comparisons[metric]['Apo']
        stats_holo = comparisons[metric]['Holo']

        # Determine overall extreme range.
        # if either group has no extremes → cannot define overlap robustly
        if np.isnan(stats_apo['min']) or np.isnan(stats_apo['max']) \
           or np.isnan(stats_holo['min']) or np.isnan(stats_holo['max']):
            overlap_ratio = np.nan
            per_metric_overlap[metric] = np.nan
            # do not accumulate into total/count
            continue
        overall_min = min(stats_apo['min'], stats_holo['min'])
        overall_max = max(stats_apo['max'], stats_holo['max'])

        #if np.isnan(stats_apo['min']) or np.isnan(stats_holo['min']):
        #    overall_min = stats_apo['min'] if not np.isnan(stats_apo['min']) else stats_holo['min']
        #    overall_max = stats_apo['max'] if not np.isnan(stats_apo['max']) else stats_holo['max']
        #else:
        #    overall_min = min(stats_apo['min'], stats_holo['min'])
        #    overall_max = max(stats_apo['max'], stats_holo['max'])

        overall_range = overall_max - overall_min
        if overall_range == 0:
            overall_range = 1.0

        # Extreme overlap based on min and max.
        intersection = max(0, min(stats_apo['max'], stats_holo['max']) - max(stats_apo['min'], stats_holo['min']))
        extreme_overlap = intersection / overall_range

        if weights is None:
            overlap_ratio = extreme_overlap
        else:
            # Center overlap: based on centers (average of mean and median).
            center_apo = (stats_apo['mean'] + stats_apo['median']) / 2.0
            center_holo = (stats_holo['mean'] + stats_holo['median']) / 2.0
            center_diff = abs(center_apo - center_holo)
            center_overlap = max(0, 1 - (center_diff / overall_range))

            # IQR overlap: based on interquartile ranges.
            overall_q1 = min(stats_apo['q1'], stats_holo['q1'])
            overall_q3 = max(stats_apo['q3'], stats_holo['q3'])
            overall_iqr_range = overall_q3 - overall_q1
            if overall_iqr_range == 0:
                overall_iqr_range = 1.0
            intersection_iqr = max(0, min(stats_apo['q3'], stats_holo['q3']) - max(stats_apo['q1'], stats_holo['q1']))
            iqr_overlap = intersection_iqr / overall_iqr_range
    
            # Weighted combination.
            overlap_ratio = w_extremes * extreme_overlap + w_center * center_overlap + w_iqr * iqr_overlap

        per_metric_overlap[metric] = round(overlap_ratio, 3) if np.isfinite(overlap_ratio) else np.nan
        if np.isfinite(overlap_ratio):
            total_overlap += overlap_ratio
            count += 1
        #per_metric_overlap[metric] = round(overlap_ratio, 3)
        #total_overlap += overlap_ratio
        #count += 1

    average_overlap = total_overlap / count if count > 0 else np.nan
    #average_overlap = total_overlap / count if count > 0 else 0.0
    return {
        'per_metric': per_metric_overlap,
        'average_overlap': round(average_overlap, 3) if np.isfinite(average_overlap) else np.nan
    }


def calculate_group_separation_and_overlap(comparisons, metrics, overlap_weights=None,
                                           w_center=0.4, w_extremes=0.4, w_iqr=0.2, normalize=True):
    """
    Compute both the robust separation and overlap measures between the Apo and Holo groups for each metric.
    
    For each metric:
      - Separation is computed as a weighted sum of the differences in:
            * Center: average difference of means and medians.
            * Extremes: average difference of minimums and maximums.
            * IQR: average difference of q1 and q3.
        The raw separation is then normalized by the overall range:
            overall_range = max(Apo_max, Holo_max) - min(Apo_min, Holo_min)
      
      - Overlap is computed as follows:
          * Extreme overlap is defined as:
                extreme_overlap = max(0, min(Apo_max, Holo_max) - max(Apo_min, Holo_min)) / overall_range
          * If overlap_weights is provided (with keys 'center', 'extremes', and 'iqr'),
            then also compute:
                center_overlap = max(0, 1 - (|center_Apo - center_Holo| / overall_range))
                iqr_overlap = max(0, min(Apo_q3, Holo_q3) - max(Apo_q1, Holo_q1)) / (max(Apo_q3, Holo_q3) - min(Apo_q1, Holo_q1))
            However, if the IQR range is degenerate (i.e. both groups have q1 and q3 equal to 0),
            then we set the iqr component weight to 0 and renormalize the remaining weights.
          * If no overlap_weights are provided, use only the extreme overlap.
    
    Parameters:
      comparisons: Dictionary with per-metric group statistics. For each metric,
                   there should be sub-dictionaries for 'Apo' and 'Holo' containing:
                   'mean', 'median', 'min', 'max', 'q1', 'q3'.
      metrics: List of metric names.
      overlap_weights: Optional dictionary with keys 'center', 'extremes', and 'iqr'.
      w_center, w_extremes, w_iqr: Weights for the separation components.
      normalize: Whether to normalize the separation score by the overall range.
    
    Returns:
      A dictionary with:
         'per_metric': { metric: { 'separation': <value>, 'overlap': <value> }, ... }
         'average_separation': average separation score across all metrics,
         'average_overlap': average overlap score across all metrics.
    """
    per_metric_results = {}
    total_separation = 0.0
    total_overlap = 0.0
    count = 0
    
    for metric in metrics:
        stats_apo = comparisons[metric]['Apo']
        stats_holo = comparisons[metric]['Holo']
        
        # Compute overall extreme range.
        overall_min = min(stats_apo['min'], stats_holo['min'])
        overall_max = max(stats_apo['max'], stats_holo['max'])
        overall_range = overall_max - overall_min
        if overall_range == 0:
            overall_range = 1.0
        
        # Separation calculation.
        diff_center = (abs(stats_apo['mean'] - stats_holo['mean']) +
                       abs(stats_apo['median'] - stats_holo['median'])) / 2.0
        diff_extremes = (abs(stats_apo['min'] - stats_holo['min']) +
                         abs(stats_apo['max'] - stats_holo['max'])) / 2.0
        diff_iqr = (abs(stats_apo['q1'] - stats_holo['q1']) +
                    abs(stats_apo['q3'] - stats_holo['q3'])) / 2.0
        raw_separation = w_center * diff_center + w_extremes * diff_extremes + w_iqr * diff_iqr
        separation = raw_separation / overall_range if normalize else raw_separation
        
        # Overlap calculation.
        # Compute extreme overlap.
        extreme_intersection = max(0, min(stats_apo['max'], stats_holo['max']) - 
                                      max(stats_apo['min'], stats_holo['min']))
        extreme_overlap = extreme_intersection / overall_range
        
        if overlap_weights is None:
            overlap = extreme_overlap
        else:
            # Compute center overlap.
            center_apo = (stats_apo['mean'] + stats_apo['median']) / 2.0
            center_holo = (stats_holo['mean'] + stats_holo['median']) / 2.0
            center_diff = abs(center_apo - center_holo)
            center_overlap = max(0, 1 - (center_diff / overall_range))
            
            # Compute IQR overlap.
            overall_q1 = min(stats_apo['q1'], stats_holo['q1'])
            overall_q3 = max(stats_apo['q3'], stats_holo['q3'])
            overall_iqr_range = overall_q3 - overall_q1
            # If IQR is degenerate, omit the IQR component.
            if overall_iqr_range == 0:
                iqr_overlap = extreme_overlap  # or simply set to 0
                effective_w_iqr = 0.0
            else:
                iqr_intersection = max(0, min(stats_apo['q3'], stats_holo['q3']) - 
                                           max(stats_apo['q1'], stats_holo['q1']))
                iqr_overlap = iqr_intersection / overall_iqr_range
                effective_w_iqr = overlap_weights.get('iqr', 0)
            
            # Normalize the weights.
            w_c = overlap_weights.get('center', 0)
            w_e = overlap_weights.get('extremes', 0)
            total_w = w_c + w_e + effective_w_iqr
            if total_w > 0:
                w_c /= total_w
                w_e /= total_w
                effective_w_iqr /= total_w
            else:
                # Fallback if total weight is zero.
                w_c, w_e, effective_w_iqr = 0, 1, 0
            
            overlap = w_e * extreme_overlap + w_c * center_overlap + effective_w_iqr * iqr_overlap
        
        per_metric_results[metric] = {
            'separation': round(separation, 3),
            'overlap': round(overlap, 3)
        }
        
        total_separation += separation
        total_overlap += overlap
        count += 1
        
    average_separation = total_separation / count if count > 0 else 0.0
    average_overlap = total_overlap / count if count > 0 else 0.0
    
    return {
        'per_metric': per_metric_results,
        'average_separation': round(average_separation, 3),
        'average_overlap': round(average_overlap, 3)
    }


## Find closest candidates
def find_closest_candidates(
    df_candidates: pd.DataFrame,
    df_model: pd.DataFrame,
    metrics: list[str],
    *,
    group_col: str = "apoholo_assignment",
    structure_cols: list[str] = ["structure", "chains3", "pocket", "#resis3"],
    ligand_cols: tuple[str, str] = ("query_ligs", "nonquery_ligs"),
    top_n: int = 5,
    scale: str = "range",       # "range" | "none" | "z_all" | "z_group" (extend easily)
    decimals: int = 5,          # round floats in JSON output
) -> dict:
    """
    Rank candidate rows by proximity to the model point, per metric and overall.

    Robust to missing/invalid model rows and per-metric NaNs.
    """
    # -------------------- Guards for model row --------------------
    if df_model is None or len(df_model) == 0:
        # No model row → return a well-formed empty result so the batch doesn't crash
        empty_per_metric = {m: [] for m in metrics}
        return {
            "params": {
                "metrics": metrics,
                "scale": scale,
                "top_n": int(top_n),
                "decimals": int(decimals),
                "ligand_cols": list(ligand_cols),
                "model_row": None,
                "note": "No model row (role=='M') found; skipped candidate ranking for this entry.",
            },
            "per_metric": {"all": empty_per_metric, "apo": empty_per_metric.copy(), "holo": empty_per_metric.copy()},
            "overall": {"all": [], "apo": [], "holo": []},
        }

    if len(df_model) > 1:
        # Be permissive: take the first model row
        dm = df_model.iloc[0].copy()
    else:
        dm = df_model.iloc[0].copy()

    # Map A/A*/A** → Apo, H/H* → Holo
    def _gs(x):
        x = str(x)
        return "Apo" if x.startswith("A") else ("Holo" if x.startswith("H") else "Other")

    # Ligand collector
    qcol, ncol = ligand_cols
    def _collect_ligs(row) -> str:
        vals = []
        for col in (qcol, ncol):
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                vals.append(str(row[col]).strip())
        if not vals:
            return ""
        parts = []
        for v in vals:
            parts.extend(p.strip() for p in str(v).split("-") if p.strip())
        seen, merged = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                merged.append(p)
        return "-".join(merged)

    # Prep
    df = df_candidates.copy()
    df["group_simple"] = df[group_col].map(_gs)

    # Choose scaling per metric
    scales = {}
    if scale in ("range", "none"):
        for m in metrics:
            vals = pd.to_numeric(df[m], errors="coerce")
            if scale == "range":
                rng = float(vals.max() - vals.min())
                scales[m] = rng if rng > 0 else 1.0
            else:
                scales[m] = 1.0
    elif scale in ("z_all", "z_group"):
        for m in metrics:
            scales[m] = 1.0
    else:
        raise ValueError("scale must be 'range', 'none', 'z_all', or 'z_group'.")

    # Pre-extract model values per metric (numeric or NaN)
    model_vals: dict[str, float] = {}
    for m in metrics:
        try:
            model_vals[m] = float(pd.to_numeric(pd.Series([dm.get(m, np.nan)]), errors="coerce").iloc[0])
        except Exception:
            model_vals[m] = np.nan

    # Per-metric distances + overall (compute in full precision)
    dist_cols = []
    for m in metrics:
        mv = model_vals.get(m, np.nan)
        if not np.isfinite(mv):
            # No valid model value for this metric → mark distances as NaN to skip
            df[f"d_{m}"] = np.nan
        else:
            vals = pd.to_numeric(df[m], errors="coerce")
            df[f"d_{m}"] = (vals - mv).abs() / (scales[m] if scales[m] > 0 else 1.0)
        dist_cols.append(f"d_{m}")

    # Overall distance ignores NaNs per row
    df["d_overall"] = df[dist_cols].mean(axis=1, skipna=True)

    # Helpers to assemble records
    def _rec_subframe(sub: pd.DataFrame, metric: str) -> list[dict]:
        col_d = f"d_{metric}"
        if col_d not in sub.columns:
            return []
        cols = [*structure_cols, group_col, metric, col_d, qcol, ncol]
        cols = [c for c in cols if c in sub.columns]
        # nsmallest automatically drops NaNs; if all NaN → empty list
        out = []
        for _, r in sub.nsmallest(top_n, col_d)[cols].iterrows():
            rec = {k: r[k] for k in structure_cols if k in r}
            rec[group_col] = r[group_col]
            if metric in r and pd.notna(r[metric]):
                rec[metric] = round(float(r[metric]), decimals)
            if col_d in r and pd.notna(r[col_d]):
                rec["distance_norm"] = round(float(r[col_d]), decimals)
            rec["ligands_joined"] = _collect_ligs(r)
            out.append(rec)
        return out

    def _overall_records(sub: pd.DataFrame) -> list[dict]:
        if "d_overall" not in sub.columns:
            return []
        cols = [*structure_cols, group_col, "d_overall", qcol, ncol, *metrics]
        cols = [c for c in cols if c in sub.columns]
        out = []
        for _, r in sub.nsmallest(top_n, "d_overall")[cols].iterrows():
            rec = {k: r[k] for k in structure_cols if k in r}
            rec[group_col] = r[group_col]
            if pd.notna(r.get("d_overall", np.nan)):
                rec["d_overall"] = round(float(r["d_overall"]), decimals)
            for m in metrics:
                if m in r and pd.notna(r[m]):
                    rec[m] = round(float(r[m]), decimals)
            rec["ligands_joined"] = _collect_ligs(r)
            out.append(rec)
        return out

    # Build blocks
    per_metric_all = {m: _rec_subframe(df, m) for m in metrics}
    per_metric_apo = {m: _rec_subframe(df[df["group_simple"] == "Apo"], m) for m in metrics}
    per_metric_holo = {m: _rec_subframe(df[df["group_simple"] == "Holo"], m) for m in metrics}

    overall_all = _overall_records(df)
    overall_apo = _overall_records(df[df["group_simple"] == "Apo"])
    overall_holo = _overall_records(df[df["group_simple"] == "Holo"])

    # Model values snapshot (rounded) for JSON
    model_snapshot = {k: dm[k] for k in (structure_cols + metrics) if k in df_model.columns}
    for k, v in list(model_snapshot.items()):
        if isinstance(v, (float, int)) and k in metrics and np.isfinite(v):
            model_snapshot[k] = round(float(v), decimals)

    return {
        "params": {
            "metrics": metrics,
            "scale": scale,
            "top_n": int(top_n),
            "decimals": int(decimals),
            "ligand_cols": list(ligand_cols),
            "model_row": model_snapshot,
        },
        "per_metric": {"all": per_metric_all, "apo": per_metric_apo, "holo": per_metric_holo},
        "overall": {"all": overall_all, "apo": overall_apo, "holo": overall_holo},
    }



## Utils


def point_score_percentage(classification_score, overall_separation_score):
    """
    Calculate the point score as a percentage of the group separation score.
    
    Parameters:
    - classification_score: The absolute difference in composite distances (point score)
      between the Apo and Holo groups, as computed by classify_point_composite.
    - overall_separation_score: The average separation score between the groups,
      as computed by overall_group_separation.
    
    Returns:
    - A percentage value indicating how large the point's score is relative to the
      inherent separation of the groups. If overall_separation_score is 0, returns None.
    """
    if overall_separation_score == 0:
        return None  # Avoid division by zero.
    elif classification_score is None:
        return float('NaN') # 0.0
    return (classification_score / overall_separation_score) * 100


### DP functions

# --- DP Gaussian mixture helpers for plotting (safe & reusable) ---

def fit_bgm_safe0(values, cfg: "DPClassifyConfig" | None = None):
    """
    Fit a BayesianGaussianMixture robustly for 1D data.
    - Drops NaNs/inf
    - Caps components by number of unique values
    - Uses 'random' init and small reg_covar to avoid convergence issues
    Returns a fitted model or None if there is insufficient data (n < 2).
    """
    if cfg is None:
        cfg = DPClassifyConfig()

    x = ensure_numeric_1d(values)
    if x.size < 2:
        return None  # nothing meaningful to fit

    # cap components by unique values
    uniq = np.unique(x)
    ncomp = max(1, min(cfg.n_components, uniq.size))

    try:
        from sklearn.mixture import BayesianGaussianMixture
    except Exception:
        return None  # degrade gracefully if sklearn unavailable

    bgm = BayesianGaussianMixture(
        n_components=ncomp,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        init_params=cfg.init_params,   # e.g., "random" to avoid kmeans init warnings
        reg_covar=cfg.reg_covar,       # stabilize near-singular covariances
        max_iter=1000,
        tol=1e-4,
        random_state=cfg.random_state,
    ).fit(x.reshape(-1, 1))
    return bgm


def fit_bgm_safe(values, cfg: "DPClassifyConfig" | None = None):
    """
    Fit a BayesianGaussianMixture robustly for 1D data.
    Drops NaNs/inf, caps components by unique values, and suppresses sklearn ConvergenceWarning.
    Returns a fitted model or None if there is insufficient data (n < 2).
    """
    if cfg is None:
        cfg = DPClassifyConfig()

    x = ensure_numeric_1d(values)
    if x.size < 2:
        return None  # nothing meaningful to fit

    uniq = np.unique(x)
    # cap components both by unique values and by n//2 for tiny n
    ncap = max(1, min(cfg.n_components, len(uniq), (x.size // 2 if x.size >= 6 else 1)))

    try:
        from sklearn.mixture import BayesianGaussianMixture
    except Exception:
        return None  # degrade gracefully if sklearn unavailable

    bgm = BayesianGaussianMixture(
        n_components=ncap,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        init_params=cfg.init_params,   # "kmeans" or "random"
        reg_covar=cfg.reg_covar,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        n_init=cfg.n_init,
        random_state=cfg.random_state,
    )

    if cfg.suppress_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return bgm.fit(x.reshape(-1, 1))
    else:
        return bgm.fit(x.reshape(-1, 1))


def make_bgm_grid(apo_values, holo_values=None, grid_points: int = 1000):
    """
    Build a plotting grid from the union of (apo,holo) values with 10% padding.
    Returns np.ndarray of shape (grid_points,) or None if no data at all.
    """
    a = ensure_numeric_1d(apo_values)
    h = ensure_numeric_1d(holo_values) if holo_values is not None else np.array([], dtype=float)
    x = np.concatenate([a, h]) if h.size else a
    if x.size == 0:
        return None
    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        pad = 1.0  # arbitrary small pad if degenerate
        lo, hi = lo - pad, hi + pad
    else:
        pad = 0.10 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, int(grid_points))


def evaluate_bgm_pdf(bgm, xs: np.ndarray) -> np.ndarray | None:
    """
    Evaluate the fitted BGM density on xs (1D).
    Returns array of same length as xs, or None if bgm/xs invalid.
    """
    if bgm is None or xs is None or xs.size == 0:
        return None
    X = xs.reshape(-1, 1)
    try:
        return np.exp(bgm.score_samples(X))
    except Exception:
        return None


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, float)
    if sigma <= 0 or not np.isfinite(sigma):
        return np.zeros_like(x)
    c = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    z = (x - mu) / sigma
    return c * np.exp(-0.5 * z * z)


def evaluate_group_pdf(values: np.ndarray, xs: np.ndarray, cfg: DPClassifyConfig) -> np.ndarray:
    """
    Evaluate group density on xs using BGM if possible, otherwise
    fall back to a single Normal with robust sigma floor.
    """
    values = np.asarray(values, float)
    xs = np.asarray(xs, float)
    mask = np.isfinite(values)
    v = values[mask]
    if v.size == 0 or xs.size == 0:
        return np.zeros_like(xs)

    model = _fit_bgm_safe(v, cfg)
    if model is not None:
        return np.exp(model.score_samples(xs.reshape(-1, 1)))

    # Fallback: robust Normal approx
    mu = float(np.nanmedian(v)) if np.isfinite(np.nanmedian(v)) else float(np.nanmean(v))
    sigma = float(np.nanstd(v))
    if not np.isfinite(sigma) or sigma < cfg.std_floor:
        # MAD-based sigma with floor
        mad = float(np.nanmedian(np.abs(v - mu)))
        sigma = max(cfg.std_floor, 1.4826 * mad if np.isfinite(mad) and mad > 0 else cfg.std_floor)
    return _normal_pdf(xs, mu, sigma)

