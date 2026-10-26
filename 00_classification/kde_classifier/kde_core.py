# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 17:20:29 2025

@author: Christos P. Feidakis
"""


# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple #, List
import numpy as np
from scipy.stats import gaussian_kde, norm

# ---------- Dataclasses ----------

@dataclass(frozen=True)
class KDEConfig:
    bw: float = 0.01                 # absolute floor
    bw_mode: str = "pooled"          # "pooled" | "separate"
    bw_scale: float = 0.75           # temper pooled width
    variance_eps: float = 1e-12
    kde_extend: bool = True
    margin_cap_scale: float = 1.0  # scales how far tail-chasing may go
    #kde_cut: float = 2.0
    grid_points: int = 1000
    missing_group_policy: str = "nan"   # "nan" | "assume_separated"

    def to_dict(self) -> dict:
        """JSON-ready snapshot of KDE hyperparameters actually used."""
        return {
            "bw": self.bw,
            "bw_mode": self.bw_mode,
            "bw_scale": self.bw_scale,
            "variance_eps": self.variance_eps,
            "kde_extend": int(bool(self.kde_extend)),  # keep your saved style (0/1)
            "margin_cap_scale": self.margin_cap_scale,
            "grid_points": self.grid_points,
            "missing_group_policy": self.missing_group_policy,
        }

@dataclass(frozen=True)
class Priors:
    mode: str = "equal"              # "equal" | "empirical" | "fixed_alpha" | "power" | "mix"
    alpha: float = 10.0
    power: float = 0.5
    odds_cap: float = 5.0

    def to_dict(self) -> dict:
        """JSON-ready snapshot of Bayesian prior controls actually used."""
        return {
            "prior_mode": self.mode,
            "alpha_prior": self.alpha,
            "prior_power": self.power,
            "prior_odds_cap": self.odds_cap,
        }

@dataclass(frozen=True)
class MetricData:
    name: str
    apo: np.ndarray
    holo: np.ndarray
    combo: np.ndarray
    model_value: float | np.nan

@dataclass(frozen=True)
class DensityCurves:
    x: np.ndarray
    apo: np.ndarray
    holo: np.ndarray
    all: np.ndarray
    bw_apo: float
    bw_holo: float
    bw_all: float
    bw_mode: str

@dataclass(frozen=True)
class KDEResult:
    curves: DensityCurves
    max_apo: float
    max_holo: float
    max_all: float
    overlap: float | np.nan
    mode_overlap: float | np.nan

@dataclass(frozen=True)
class MetricClassResult:
    classification: str
    P_apo: float | np.nan
    P_holo: float | np.nan
    score: float
    density_sum: float
    norm_density_strongest: float
    norm_density_total: float
    norm_density_combined: float
    weight: float
    model_value: float | np.nan
    likelihood_apo: float
    likelihood_holo: float
    prior_apo: float
    prior_holo: float

@dataclass(frozen=True)
class OverallResult:
    avg_P_apo: float | np.nan
    avg_P_holo: float | np.nan
    overall_score: float
    overall_overlap: float | np.nan
    overall_mode_overlap: float | np.nan
    avg_norm_density_strongest: float | np.nan
    avg_norm_density_total: float | np.nan
    avg_norm_density_combined: float | np.nan
    overall_classification: str
    weights_sum: float | np.nan

# ---------- Utils (warning-safe) ----------

def ensure_numeric_1d(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float).ravel()
    return a[np.isfinite(a)]

def nanmean_safe(arr, round_to: Optional[int] = None):
    a = np.asarray(arr, float)
    mask = np.isfinite(a)
    if not mask.any(): 
        return np.nan
    v = float(np.mean(a[mask]))
    return round(v, round_to) if round_to is not None and np.isfinite(v) else v

def weighted_mean_safe(values, weights, round_to: Optional[int] = None):
    v = np.asarray(values, float); w = np.asarray(weights, float)
    mask = np.isfinite(v) & np.isfinite(w)
    if not mask.any(): 
        return np.nan
    s = np.sum(w[mask])
    if s <= 0: 
        return np.nan
    out = float(np.sum(v[mask]*w[mask]) / s)
    return round(out, round_to) if round_to is not None and np.isfinite(out) else out

def cohen_d_safe(a, b):
    a = ensure_numeric_1d(a); b = ensure_numeric_1d(b)
    if a.size < 2 or b.size < 2: 
        return np.nan
    s1 = np.std(a, ddof=1); s2 = np.std(b, ddof=1)
    if not np.isfinite(s1) or not np.isfinite(s2): 
        return np.nan
    pooled = np.sqrt(((a.size-1)*s1**2 + (b.size-1)*s2**2) / (a.size + b.size - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan

def _trapz(y, x):  # NaN-safe integral
    return float(np.trapz(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), x))

def compute_overall_overlap(dA, dH, x, renormalize=True, missing_group_policy="nan"):
    A = np.maximum(np.nan_to_num(np.asarray(dA,float), nan=0.0), 0.0)
    H = np.maximum(np.nan_to_num(np.asarray(dH,float), nan=0.0), 0.0)
    x = np.asarray(x,float)
    IA, IH = float(np.trapz(A,x)), float(np.trapz(H,x))
    apo_miss, holo_miss = (IA<=0 or not np.isfinite(IA)), (IH<=0 or not np.isfinite(IH))
    if apo_miss and holo_miss: 
        return np.nan
    if apo_miss or holo_miss:
        return 0.0 if missing_group_policy=="assume_separated" else np.nan
    if renormalize:
        if IA>0: A /= IA
        if IH>0: H /= IH
    return float(np.trapz(np.minimum(A,H), x))

def _fwhm_on_grid(density, x):
    d = np.asarray(density,float); x = np.asarray(x,float)
    if d.size==0 or not np.isfinite(d).any(): return np.nan
    m = float(np.nanmax(d)); 
    if m<=0 or not np.isfinite(m): return np.nan
    half = 0.5*m
    mask = (d>=half) & np.isfinite(d)
    if not mask.any(): return np.nan
    idx = np.where(mask)[0]
    return float(x[idx[-1]] - x[idx[0]])

def compute_mode_overlap(dA, dH, x, window_mode="point", window_param=1.0, common_bandwidth=None,
                         missing_group_policy="nan", round_to=3):
    # Shared with your prior logic: defaults to "point" behavior.
    dA = np.asarray(dA,float); dH = np.asarray(dH,float); x = np.asarray(x,float)
    mA = float(np.nanmax(dA)) if dA.size else 0.0
    mH = float(np.nanmax(dH)) if dH.size else 0.0
    if (mA<=0) or (mH<=0):
        return 0.0 if missing_group_policy=="assume_separated" else np.nan
    xA, xH = x[int(np.nanargmax(dA))], x[int(np.nanargmax(dH))]
    if window_mode == "point" or (isinstance(window_param,(int,float)) and window_param==0):
        densH_xA = float(np.interp(xA, x, dH))
        densA_xH = float(np.interp(xH, x, dA))
        rA = densH_xA / mH if mH>0 else np.nan
        rH = densA_xH / mA if mA>0 else np.nan
        val = 0.5*(rA+rH) if np.isfinite(rA) and np.isfinite(rH) else np.nan
        return float(np.round(val, round_to)) if np.isfinite(val) else np.nan
    # windowed modes
    if window_mode == "abs":
        wA = wH = float(window_param)
    elif window_mode == "bw":
        if not (isinstance(common_bandwidth,(int,float)) and common_bandwidth>0):
            raise ValueError("common_bandwidth must be positive for window_mode='bw'")
        k = float(window_param) if np.isfinite(window_param) and window_param>0 else 1.0
        wA = wH = 2.0 * k * float(common_bandwidth)
    elif window_mode == "fwhm":
        k = float(window_param) if np.isfinite(window_param) and window_param>0 else 1.0
        fA, fH = _fwhm_on_grid(dA,x), _fwhm_on_grid(dH,x)
        R = (x[-1]-x[0]) * 0.02
        if not np.isfinite(fA) or fA<=0: fA = R
        if not np.isfinite(fH) or fH<=0: fH = R
        wA, wH = k*fA, k*fH
    else:
        raise ValueError("window_mode must be 'point'|'abs'|'bw'|'fwhm'")
    def mask(center,width):
        if not np.isfinite(width) or width<=0:  # fallback to single cell
            m = np.zeros_like(x,dtype=bool)
            m[np.argmin(np.abs(x-center))] = True
            return m
        half = 0.5*width
        return (x>=center-half) & (x<=center+half)
    mA_mask, mH_mask = mask(xA,wA), mask(xH,wH)
    dA_A, dH_A = dA.copy(), dH.copy()
    dA_H, dH_H = dA.copy(), dH.copy()
    dA_A[~mA_mask]=0.0; dH_A[~mA_mask]=0.0
    dA_H[~mH_mask]=0.0; dH_H[~mH_mask]=0.0
    numA = _trapz(np.minimum(dA_A,dH_A), x); denA = _trapz(dA_A, x)
    numH = _trapz(np.minimum(dA_H,dH_H), x); denH = _trapz(dH_H, x)
    oA = (numA/denA) if denA>0 else np.nan
    oH = (numH/denH) if denH>0 else np.nan
    val = 0.5*(oA+oH) if np.isfinite(oA) and np.isfinite(oH) else np.nan
    return float(np.round(val, round_to)) if np.isfinite(val) else np.nan

# ---------- KDE building ----------

def _silverman(vals, default_bw=0.1):
    v = ensure_numeric_1d(vals)
    if v.size < 2: return default_bw
    s = np.std(v, ddof=1); n = v.size
    return 1.06 * s * n**(-1/5)

def _grid_and_bw(combo, model_value, cfg: KDEConfig):
    combo = ensure_numeric_1d(combo)
    if combo.size:
        x_min, x_max = float(np.nanmin(combo)), float(np.nanmax(combo))
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            x_min, x_max = 0.0, 1.0
        rng = max(1e-12, x_max - x_min)
        bw_pooled = _silverman(combo, default_bw=cfg.bw)
        bw_floor = 0.01 * rng
        bw_eff = max(bw_pooled*max(cfg.bw_scale,1e-6), bw_floor, cfg.bw)


        # grid margins
        DEFAULT_BASE_CUT = 2.0  # preserves old default behavior

        if cfg.kde_extend:
            # cheap combined pdf for tail checks
            def pdf(xx):
                if combo.size>=2 and np.nanstd(combo)>cfg.variance_eps:
                    kde_pre = gaussian_kde(combo)
                    return float(kde_pre([xx])[0])
                center = float(np.nanmean(combo))
                return float(norm.pdf(xx, loc=center, scale=bw_eff))

            # heuristic margin growth
            # seed with the old default base margin
            base_margin = DEFAULT_BASE_CUT * bw_eff
            margin = max(base_margin, 0.05 * rng)
            step = max(0.5 * bw_eff, rng * 0.01, 1e-9)

            # allow farther tail chasing with margin_cap_scale
            max_margin = (10.0 * cfg.margin_cap_scale) * bw_eff

            # estimate peak
            if combo.size>=2 and np.nanstd(combo)>cfg.variance_eps:
                kde_pre = gaussian_kde(combo)
                peak = float(np.max(kde_pre(combo)))
            else:
                center = float(np.nanmean(combo)) if combo.size else 0.0
                peak = float(norm.pdf(center, center, bw_eff))

            tail_rel = 1e-4
            while margin < max_margin and pdf(x_min - margin) > tail_rel * peak:
                margin += step
            while margin < max_margin and pdf(x_max + margin) > tail_rel * peak:
                margin += step
        else:
            margin = DEFAULT_BASE_CUT * bw_eff # when not extending, use the old default fixed margin 2.0 * bw_eff
            #margin = (cfg.kde_cut * bw_eff) if cfg.kde_cut else (0.05*rng if rng>0 else 1.0)

        x = np.linspace(x_min - margin, x_max + margin, cfg.grid_points)

    else:
        x = np.linspace(0, 1, cfg.grid_points)
        bw_eff = cfg.bw

    if model_value is not None and np.isfinite(model_value):
        x = np.unique(np.append(x, model_value))
    return x, bw_eff

def _density(values, x, bw_eff, cfg: KDEConfig, bw_mode_for_class: Optional[str]=None):
    v = ensure_numeric_1d(values)
    n = v.size
    if n == 0:
        return None, np.zeros_like(x), bw_eff
    if n == 1 or float(np.nanstd(v, ddof=1) if n>=2 else np.nan) <= cfg.variance_eps:
        center = float(np.nanmean(v))
        dens = norm.pdf(x, loc=center, scale=bw_eff)
        return None, dens, bw_eff
    s = float(np.nanstd(v, ddof=1))
    if cfg.bw_mode == "pooled":
        factor = max(1e-9, bw_eff / max(s, 1e-12))   # gaussian_kde’s factor * s = kernel std
        kde = gaussian_kde(v, bw_method=factor)
        return kde, kde(x), bw_eff
    elif cfg.bw_mode == "separate":
        kde_tmp = gaussian_kde(v)                     # Scott
        factor_scott = float(kde_tmp.factor)
        lo = 0.7 * (bw_eff / max(s,1e-12))
        hi = 1.4 * (bw_eff / max(s,1e-12))
        factor = float(np.clip(factor_scott, lo, hi))
        kde = gaussian_kde(v, bw_method=factor)
        return kde, kde(x), float(factor * s)
    else:
        raise ValueError("bw_mode must be 'pooled' or 'separate'")

def build_kde(metric: MetricData, cfg: KDEConfig) -> KDEResult:
    x, bw_eff = _grid_and_bw(metric.combo, metric.model_value, cfg)
    kdeA, dA, bwA = _density(metric.apo,  x, bw_eff, cfg)
    kdeH, dH, bwH = _density(metric.holo, x, bw_eff, cfg)
    kdeAll, dAll, bwAll = _density(metric.combo, x, bw_eff, cfg)

    maxA = float(np.nanmax(dA)) if dA.size else 0.0
    maxH = float(np.nanmax(dH)) if dH.size else 0.0
    maxAll = float(np.nanmax(dAll)) if dAll.size else 0.0

    overlap = compute_overall_overlap(dA, dH, x, renormalize=True, missing_group_policy=cfg.missing_group_policy)
    mode_overlap = compute_mode_overlap(dA, dH, x, window_mode="point", missing_group_policy=cfg.missing_group_policy)

    curves = DensityCurves(x=x, apo=dA, holo=dH, all=dAll,
                           bw_apo=(bwA if cfg.bw_mode=="separate" else bw_eff),
                           bw_holo=(bwH if cfg.bw_mode=="separate" else bw_eff),
                           bw_all=(bwAll if cfg.bw_mode=="separate" else bw_eff),
                           bw_mode=cfg.bw_mode)

    return KDEResult(curves=curves, max_apo=maxA, max_holo=maxH, max_all=maxAll,
                     overlap=float(np.round(overlap,3)) if np.isfinite(overlap) else np.nan,
                     mode_overlap=mode_overlap)

# ---------- Per-metric classification ----------

def _priors_for_sizes(nA: int, nH: int, pri: Priors) -> Tuple[float, float]:
    """
    Compute (prior_apo, prior_holo) from group sizes and prior controls.

    Modes:
      - "equal":       (0.5, 0.5)
      - "empirical":   Dirichlet-smoothed counts with alpha (same as 'fixed_alpha'):
                        priorA = (nA + alpha) / (N + 2*alpha)
      - "fixed_alpha": same as 'empirical' (alias)
      - "power":       priorA from tempered/capped odds:
                        odds = ((nH + alpha) / (nA + alpha))
                        if odds_cap: odds = min(odds, odds_cap)
                        odds = odds ** power
                        priorA = 1 / (1 + odds)
      - "mix":         convex mix of equal and empirical priors; λ fixed at 0.5:
                        priorA = 0.5*(0.5) + 0.5*((nA/N) if N>0 else 0.5)
    """
    # Protect against weird negatives and division by zero
    nA = max(0, int(nA))
    nH = max(0, int(nH))
    N = nA + nH
    alpha = float(pri.alpha)
    power = float(pri.power)
    odds_cap = float(pri.odds_cap) if pri.odds_cap is not None else None

    mode = (pri.mode or "equal").lower()

    if mode == "equal":
        priorA = 0.5

    elif mode in ("empirical", "fixed_alpha"):
        if N > 0:
            priorA = (nA + alpha) / (N + 2.0 * alpha)
        else:
            priorA = 0.5

    elif mode == "power":
        # Odds of H vs A with alpha smoothing
        denom = max(nA + alpha, 1e-12)
        odds = (nH + alpha) / denom
        if odds_cap is not None and np.isfinite(odds_cap) and odds_cap > 0:
            odds = min(odds, odds_cap)
        # Temper the odds by 'power' (γ)
        if np.isfinite(power) and power != 1.0:
            odds = odds ** power
        priorA = 1.0 / (1.0 + odds)

    elif mode == "mix":
        # Simple 50-50 blend between equal and empirical (unsmoothed)
        lam = 0.5
        emp = (nA / N) if N > 0 else 0.5
        priorA = (1 - lam) * 0.5 + lam * emp

    else:
        # Fallback to equal if unknown mode
        priorA = 0.5

    priorA = float(np.clip(priorA, 0.0, 1.0))
    priorH = float(1.0 - priorA)
    return priorA, priorH


def classify_metric(metric: MetricData, kde: KDEResult, priors: Priors, density_threshold: float = 1e-3) -> MetricClassResult:
    # Evaluate likelihoods at model value (if present)
    mv = metric.model_value
    if mv is None or not np.isfinite(mv):
        # no model → neutral, but keep structure
        return MetricClassResult(
            classification="Undefined", P_apo=np.nan, P_holo=np.nan, score=np.nan, # score=0.0,
            density_sum=np.nan, norm_density_strongest=np.nan,
            norm_density_total=np.nan, norm_density_combined=np.nan,
            weight=np.nan, model_value=np.nan,
            likelihood_apo=np.nan, likelihood_holo=np.nan,
            prior_apo=np.nan, prior_holo=np.nan
        )

    dA = float(np.interp(mv, kde.curves.x, kde.curves.apo))
    dH = float(np.interp(mv, kde.curves.x, kde.curves.holo))
    dAll = float(np.interp(mv, kde.curves.x, kde.curves.all))
    maxA, maxH, maxAll = kde.max_apo, kde.max_holo, kde.max_all

    # Priors
    nA, nH = metric.apo.size, metric.holo.size
    priorA, priorH = _priors_for_sizes(nA, nH, priors)

    # Posterior up to proportionality
    postA_raw = dA * priorA
    postH_raw = dH * priorH
    denom = postA_raw + postH_raw
    if denom > 0:
        P_apo_norm = postA_raw / denom
        P_holo_norm = postH_raw / denom
    else:
        P_apo_norm = np.nan
        P_holo_norm = np.nan

    # label
    if not (np.isfinite(P_apo_norm) and np.isfinite(P_holo_norm)):
        label = "Undefined"
    elif np.isclose(P_apo_norm, P_holo_norm, rtol=0.0, atol=1e-12):
        label = "Tie" #"Ambiguous"
    else:
        label = "Apo" if P_apo_norm > P_holo_norm else "Holo"
    # Issue #7: an undefined point (denom == 0 → both posteriors NaN) must carry NO score, so it is
    # EXCLUDED from the overall score-average (weighted_mean_safe masks NaN) exactly as its NaN
    # P(apo)/P(holo) are excluded from avg_P(apo)/avg_P(holo). A 0.0 fallback (a real "tie" value)
    # instead polluted overall_score, so the confidence-bar marker (overall_score) disagreed with the
    # overall Probability (avg_P). Mirrors the already-fixed no-model-value branch above (score=np.nan).
    score = round(P_holo_norm - P_apo_norm, 3) if np.isfinite(P_apo_norm) and np.isfinite(P_holo_norm) else np.nan

    # normalized densities. Issue #7 (extended): an Undefined point (denom == 0) carries NO valid
    # model-point information, so ALL its model-point fields are NaN — matching the no-model-value
    # branch above and the NaN P(apo)/P(holo)/score. This is also what makes the two "Undefined"
    # branches internally consistent (that branch already returns NaN densities). The effect is that
    # an undefined metric is excluded UNIFORMLY from every overall model-point aggregate
    # (avg_score AND avg_normalized_density_*), so the Overall confidence bars and the query-summary
    # reflect only the metrics that actually participated. The apo/holo distribution overlap and
    # mode_overlap live in KDEResult and stay valid for every metric, so they are still aggregated.
    min_d = 1e-12
    undefined = not (np.isfinite(P_apo_norm) and np.isfinite(P_holo_norm))
    if undefined:
        density_sum = strongest = total_norm = combined_norm = weight = np.nan
    else:
        normA = (dA/maxA)  if maxA  > min_d else 0.0
        normH = (dH/maxH)  if maxH  > min_d else 0.0
        strongest = round(max(normA, normH), 3)
        total_norm = round((dA + dH) / maxAll, 3) if maxAll > 0 else 0.0
        combined_norm = round(dAll / maxAll, 3) if maxAll > 0 else 0.0
        density_sum = round(dA + dH, 3)
        weight = round(float(min(1.0, (dA + dH) / max(density_threshold, min_d))), 3)

    return MetricClassResult(
        classification=label,
        P_apo=round(P_apo_norm,3) if np.isfinite(P_apo_norm) else np.nan,
        P_holo=round(P_holo_norm,3) if np.isfinite(P_holo_norm) else np.nan,
        score=score,
        density_sum=density_sum,
        norm_density_strongest=strongest,
        norm_density_total=total_norm,
        norm_density_combined=combined_norm,
        weight=weight,
        model_value=mv,
        likelihood_apo=round(dA,6),
        likelihood_holo=round(dH,6),
        prior_apo=round(priorA,6),
        prior_holo=round(priorH,6)
    )

# ---------- Overall aggregation ----------

def aggregate_overall(metric_results: Dict[str, MetricClassResult],
                      kde_results: Dict[str, KDEResult],
                      weights: Optional[Dict[str,float]] = None) -> OverallResult:
    # collect vectors
    names = list(metric_results.keys())
    w = np.array([ (weights.get(n,1.0) if weights else 1.0) for n in names ], float)

    P_a   = np.array([ metric_results[n].P_apo  for n in names ], float)
    P_h   = np.array([ metric_results[n].P_holo for n in names ], float)
    sc    = np.array([ metric_results[n].score  for n in names ], float)
    nds   = np.array([ metric_results[n].norm_density_strongest for n in names ], float)
    ndt   = np.array([ metric_results[n].norm_density_total     for n in names ], float)
    ndc   = np.array([ metric_results[n].norm_density_combined  for n in names ], float)
    ov    = np.array([ kde_results[n].overlap       for n in names ], float)
    mov   = np.array([ kde_results[n].mode_overlap  for n in names ], float)

    avg_Pa = weighted_mean_safe(P_a, w, 3)
    avg_Ph = weighted_mean_safe(P_h, w, 3)
    avg_sc = weighted_mean_safe(sc, w, 3)

    avg_nds = weighted_mean_safe(nds, w, 3)
    avg_ndt = weighted_mean_safe(ndt, w, 3)
    avg_ndc = weighted_mean_safe(ndc, w, 3)
    avg_ov  = weighted_mean_safe(ov,  w, 3)
    avg_mov = weighted_mean_safe(mov, w, 3)

    if np.isfinite(avg_Pa) and np.isfinite(avg_Ph):
        if np.isclose(avg_Pa, avg_Ph, rtol=0.0, atol=1e-12):
            overall_label = "Ambiguous"
        else:
            overall_label = "Apo" if avg_Pa > avg_Ph else "Holo"
    else:
        overall_label = "Undefined"

    return OverallResult(
        avg_P_apo=avg_Pa, avg_P_holo=avg_Ph, overall_score=avg_sc,
        overall_overlap=avg_ov, overall_mode_overlap=avg_mov,
        avg_norm_density_strongest=avg_nds,
        avg_norm_density_total=avg_ndt,
        avg_norm_density_combined=avg_ndc,
        overall_classification=overall_label,
        weights_sum=float(np.sum(w)) if w.size else np.nan
    )

# ---------- Final packer ----------

def make_classification_result(metric_results: Dict[str, MetricClassResult],
                               kde_results: Dict[str, KDEResult],
                               overall: OverallResult,
                               variables_block: Dict) -> Dict:
    out = {}
    for m in metric_results:
        mr = metric_results[m]; kr = kde_results[m]
        out[m] = {
            "classification": mr.classification,
            "P(apo)": mr.P_apo, "P(holo)": mr.P_holo, "score": mr.score,
            "overlap": kr.overlap, "mode_overlap": kr.mode_overlap,
            "normalized_density_strongest": mr.norm_density_strongest,
            "normalized_density_total": mr.norm_density_total,
            "normalized_density_combined": mr.norm_density_combined,
            "weight": mr.weight,
            # keep your extended bookkeeping
            "model_value": mr.model_value,
            "likelihood_apo": mr.likelihood_apo,
            "likelihood_holo": mr.likelihood_holo,
            "prior_apo": mr.prior_apo,
            "prior_holo": mr.prior_holo,
        }
    out["overall"] = {
        "avg_P(apo)": overall.avg_P_apo,
        "avg_P(holo)": overall.avg_P_holo,
        "overall_score": overall.overall_score,
        "overall_overlap": overall.overall_overlap,
        "overall_mode_overlap": overall.overall_mode_overlap,
        "avg_normalized_density_strongest": overall.avg_norm_density_strongest,
        "avg_normalized_density_total": overall.avg_norm_density_total,
        "avg_normalized_density_combined": overall.avg_norm_density_combined,
        "overall_classification": overall.overall_classification,
        "weights_sum": overall.weights_sum,
    }
    out["variables"] = variables_block
    return out

# ---------- Pipeline helpers ----------

def prepare_metric_data(df_c, metric: str, apo_group, holo_group, model_point) -> MetricData:
    apo_vals  = apo_group[metric].dropna().values
    holo_vals = holo_group[metric].dropna().values
    if apo_vals.size and holo_vals.size: 
        combo = np.concatenate([apo_vals, holo_vals])
    elif apo_vals.size: 
        combo = apo_vals
    else:
        combo = holo_vals
    mv = np.nan
    if model_point is not None and not model_point.empty and metric in model_point.columns:
        mv = model_point[metric].values[0]
        mv = float(mv) if np.isfinite(mv) else np.nan
    return MetricData(name=metric, apo=apo_vals, holo=holo_vals, combo=combo, model_value=mv)

