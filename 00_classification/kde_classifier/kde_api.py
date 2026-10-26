# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 17:49:08 2025

@author: Christos P. Feidakis
"""

# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from kde_core import (
    KDEConfig, Priors, prepare_metric_data, build_kde, classify_metric,
    aggregate_overall, make_classification_result, KDEResult, MetricClassResult
)

def classify_point_kde_r3(
    df_c, apo_group, holo_group, model_point,
    classification_metrics: List[str],
    *,
    density_threshold: float = 1e-3,
    bw: float = 0.01,
    kde_extend: bool = False,
    margin_cap_scale: float = 1.0, # Old: kde_cut: float = 2.0,
    bw_mode: str = "separate",     # "pooled" | "separate"
    prior_mode: str = "equal",
    alpha_prior: float = 10.0,
    prior_power: float = 0.5,
    prior_odds_cap: float = 5.0,
    bw_scale: float = 0.75
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:

    #density_threshold: float = 1e-3,
    #bw: float | None = None,
    #bw_mode: str | None = None,
    #bw_scale: float | None = None,
    #kde_extend: bool | None = None,

    #prior_mode: str | None = None,
    #alpha_prior: float | None = None,
    #prior_power: float | None = None,
    #prior_odds_cap: float | None = None,

    """
    Backward-compatible wrapper building the entire pipeline and returning:
      classification_results_dict, kde_data_dict, apo_values_dict, holo_values_dict, all_values_dict
    Keys and values are preserved to match your current consumers.
    """
    cfg = KDEConfig(bw=bw, bw_mode=bw_mode, bw_scale=bw_scale,
                    kde_extend=bool(kde_extend), margin_cap_scale=margin_cap_scale,
                    missing_group_policy="nan")
    pri = Priors(mode=prior_mode, alpha=alpha_prior, power=prior_power, odds_cap=prior_odds_cap)

    kde_data: Dict[str, dict] = {}
    metric_results: Dict[str, MetricClassResult] = {}
    kde_results: Dict[str, KDEResult] = {}
    apo_values_dict: Dict[str, np.ndarray] = {}
    holo_values_dict: Dict[str, np.ndarray] = {}
    all_values_dict: Dict[str, np.ndarray] = {}

    for metric in classification_metrics:
        # 1) data prep
        md = prepare_metric_data(df_c, metric, apo_group, holo_group, model_point)
        apo_values_dict[metric]  = md.apo
        holo_values_dict[metric] = md.holo
        all_values_dict[metric]  = md.combo

        # 2) kde build
        kde = build_kde(md, cfg)
        kde_results[metric] = kde
        kde_data[metric] = {
            "x_values": kde.curves.x,
            "density_apo":  kde.curves.apo,
            "density_holo": kde.curves.holo,
            "density_all":  kde.curves.all,
            "bw_apo":  kde.curves.bw_apo,
            "bw_holo": kde.curves.bw_holo,
            "bw_all":  kde.curves.bw_all,
            "bandwidth": kde.curves.bw_all,
            "bw_mode": cfg.bw_mode,
        }

        # 3) per-metric classify
        mr = classify_metric(md, kde, pri, density_threshold=density_threshold)
        metric_results[metric] = mr

    # 4) overall aggregation
    overall = aggregate_overall(metric_results, kde_results, weights=None)

    # 5) final pack
    variables_block = {
    "density_threshold": density_threshold,  # lives at API level
    **cfg.to_dict(),
    **pri.to_dict(),
    }

    classification_results = make_classification_result(metric_results, kde_results, overall, variables_block)

    return classification_results, kde_data, apo_values_dict, holo_values_dict, all_values_dict
