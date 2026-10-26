# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 18:00:00 2025

@author: Christos P. Feidakis
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
#from scipy.stats import gaussian_kde
#import math
#import json
#from typing import Dict
#from scipy.stats import norm

from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_rgba



# Create a custom colormap from white (0) to orange (1)
CMAP_SEP = LinearSegmentedColormap.from_list("white_to_orange", ["white","orange"])

# Define all available color schemes in a single dictionary:
VIOLIN_COLOR_SCHEMES = {
    "basic1": {
        "apo": {"color": "blue", "alpha": 0.42},
        "holo": {"color": "red", "alpha": 0.45},
        "points_alpha": 0.6,
    },
    "seaborn1": {
        "apo": {"color": "#2020DF", "alpha": 0.5},
        "holo": {"color": "#DF2020", "alpha": 0.5},
        "points_alpha": 0.37, # was 0.6
    },
    "colorbrewer1": {
        "apo": {"color": (0.13, 0.59, 0.95), "alpha": 0.6},     # #2196f3
        "holo": {"color": (0.89, 0.11, 0.15), "alpha": 0.6},    # #e41a1c
        "points_alpha": 0.6,
    },
    "natpub1": {
        "apo": {"color": (0.274, 0.510, 0.706), "alpha": 0.6},  # #4682B4
        "holo": {"color": (0.863, 0.078, 0.235), "alpha": 0.6}, # #DB143C
        "points_alpha": 0.6,
    },
    "mixnat_colorbrewer": {
        "apo": {"color": (0.13, 0.59, 0.95), "alpha": 0.7},     # #2196f3
        "holo": {"color": (0.863, 0.078, 0.235), "alpha": 0.7}, # #dc143c
        "points_alpha": 0.6,
    },
    "okabeito1": {
        "apo": {"color": (0.0, 0.45, 0.70), "alpha": 0.7},      # #1f78b4, #0073B3
        "holo": {"color": (0.85, 0.33, 0.10), "alpha": 0.7},    # #e31a1c, #D9541A
        "points_alpha": 0.6,
    },
    "okabeito_mixnat": {
        "apo": {"color": (0.0, 0.45, 0.70), "alpha": 0.7},      # #0073B3
        "holo": {"color": (0.863, 0.078, 0.235), "alpha": 0.7}, # #DB143C
        "points_alpha": 0.6,
    },
    "sankey_clr2": {
        "apo": {"color": (0.212, 0.361, 0.522), "alpha": 0.6},  # #365C85 muted navy blue
        "holo": {"color": (0.639, 0.278, 0.278), "alpha": 0.6}, # #A34747 muted brick red
        "points_alpha": 0.6, # was 0.6
    },
    "sankey_clr2_a2": {
        "apo": {"color": (0.212, 0.361, 0.522), "alpha": 0.3},  # #365C85 muted navy blue
        "holo": {"color": (0.639, 0.278, 0.278), "alpha": 0.3}, # #A34747 muted brick red
        "points_alpha": 0.6, # was 0.6
    },
    "nat_pdf": {
        "apo": {"color": (0.086, 0.188, 0.357), "alpha": 0.5},  # #16305B Dark blue (nat PDF)
        "holo": {"color": (0.451, 0.114, 0.090), "alpha": 0.5}, # #731D17 Dark red (nat PDF)
        "points_alpha": 0.4,
    },
    "nat_pdf_a2": {
        "apo": {"color": (0.086, 0.188, 0.357), "alpha": 0.3},  # #16305B Dark blue (nat PDF)
        "holo": {"color": (0.451, 0.114, 0.090), "alpha": 0.3}, # #731D17 Dark red (nat PDF)
        "points_alpha": 0.4,
    },
    "curves_legacy": {
        "apo":  {"color": "blue",  "alpha": 0.30},      # "alpha": 0.42
        "holo": {"color": "red",   "alpha": 0.30},      # "alpha": 0.45
        "all":  {"color": "#888888", "alpha": 0.6},     # combined PDF
        "model":{"color": "green", "alpha": 1.0},
        "points_alpha": 0.6,
    },
    "blue_orange1": {         # DP default look-alike
        # Option 1: tab* names (matches old DP)
        "apo":  {"color": "tab:blue",   "alpha": 1.0}, # "#1f77b4"
        "holo": {"color": "tab:orange", "alpha": 1.0}, # "#ff7f0e"
        "all":  {"color": "#888888",    "alpha": 0.6},
        "model":{"color": "green",      "alpha": 1.0}, # was 0.6
        "points_alpha": 0.35, # was 0.6
    },
}



def flattened_violin_dot_color(ax, color, alpha, *, alpha_boost=0.2, gamma_correct=False, page_bg=(1,1,1,1)):
    """
    Return a solid RGB that looks like `color` with `alpha` over the axes background.

    If the axes facecolor itself has transparency, we fold it over `page_bg` (default white).
    Set gamma_correct=True to blend in linear RGB (slightly more accurate).
    """
    sr, sg, sb, _ = to_rgba(color)
    a = float(alpha) + float(alpha_boost)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)  # clamp

    br, bg, bb, ba = ax.get_facecolor()
    # If the axes background is translucent, flatten it over a white "page"
    if ba < 1.0:
        br = br*ba + page_bg[0]*(1-ba)
        bg = bg*ba + page_bg[1]*(1-ba)
        bb = bb*ba + page_bg[2]*(1-ba)

    if gamma_correct:
        to_lin   = lambda c: c**2.2
        to_gamma = lambda c: c**(1/2.2)
        srL, sgL, sbL = map(to_lin, (sr, sg, sb))
        brL, bgL, bbL = map(to_lin, (br, bg, bb))
        r = to_gamma(srL*a + brL*(1-a))
        g = to_gamma(sgL*a + bgL*(1-a))
        b = to_gamma(sbL*a + bbL*(1-a))
    else:
        r = sr*a + br*(1-a)
        g = sg*a + bg*(1-a)
        b = sb*a + bb*(1-a)

    # final clamp
    r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
    g = 0.0 if g < 0.0 else (1.0 if g > 1.0 else g)
    b = 0.0 if b < 0.0 else (1.0 if b > 1.0 else b)
    return (r, g, b)


def _resolve_palette(scheme_name: str):
    """
    Returns a dict with colors for apo/holo and optional 'all' and 'model' keys.
    Provides robust fallbacks so older schemes still work.
    """
    scheme = VIOLIN_COLOR_SCHEMES.get(scheme_name, VIOLIN_COLOR_SCHEMES["seaborn1"])
    apo   = scheme.get("apo",  {"color": "blue", "alpha": 0.5})
    holo  = scheme.get("holo", {"color": "red",  "alpha": 0.5})
    all_  = scheme.get("all",  {"color": "#888888", "alpha": 0.6})
    model = scheme.get("model",{"color": "green", "alpha": 1.0})
    pts_a = scheme.get("points_alpha", 0.6)
    return apo, holo, all_, model, pts_a



def plot_kde_curves(
    kde_data,
    #df_c,
    apo_group,
    holo_group,
    classification_metrics,
    model_point,
    metrics_dict: dict = dict(),
    show_all: int = 0,
    supertitle: int | str = 1,
    show_plot: int = 1,
    save_dir: str = None,
    save_formats: list = list(),
    filename_suffix: str = "",
    transparency: int = None,
    separate_metrics: int = 0,
    *,
    # NEW — unify look & feel with violins:
    color_scheme: str = "seaborn1",
    set_title: int = 1,
    metrics_names_and_units_dict: dict | None = None,
    show_legend: bool = True,
    show_rug: bool = True,
):
    """
    Visualize the precomputed KDE probability density functions for the selected metrics,
    and mark the model point with a green dashed line.
    """

    # Normalize the label dict and palette
    if metrics_names_and_units_dict is None:
        metrics_names_and_units_dict = {m: metrics_dict.get(m, m) for m in classification_metrics}

    apo_style, holo_style, all_style, model_style, points_alpha = _resolve_palette(color_scheme)

    # Helper to plot a single metric
    def _plot_one_metric(metric, ax, model_point):
        if metric in kde_data:
            data = kde_data[metric]
            x_values = data["x_values"]
            density_apo = data["density_apo"]
            density_holo = data["density_holo"]

            # Plot combined KDE if available (soft grey)
            if show_all == 1:
                density_all = data.get("density_all", None)
                if density_all is not None:
                    ax.plot(x_values, density_all,
                            color=all_style["color"], alpha=all_style.get("alpha", 0.6),
                            label="All")
            ax.plot(x_values, density_apo,  label="Apo",  color=apo_style["color"])
            ax.plot(x_values, density_holo, label="Holo", color=holo_style["color"])
            ax.fill_between(x_values, density_apo,alpha=apo_style.get("alpha", 0.4),  color=apo_style["color"])
            ax.fill_between(x_values, density_holo,alpha=holo_style.get("alpha", 0.4), color=holo_style["color"])

            # Plot model value if available
            if model_point is not None and not model_point.empty:
                if metric in model_point.columns and not model_point[metric].isna().all():
                    model_value = model_point[metric].values[0]
                    if not np.isnan(model_value):
                        ax.axvline(model_value, color=model_style["color"], linestyle="--", linewidth=2, label="AF model") # <- prev color="green"
                        ax.text(model_value, ax.get_ylim()[1]*0.9, f"{model_value:.3f}", color=model_style["color"],  # <- prev color="green"
                                rotation=90, verticalalignment="center", fontsize=9)

            title_txt = metrics_names_and_units_dict.get(metric, metrics_dict.get(metric, metric))
            if set_title == 1:
                ax.set_title(f"{title_txt}", fontsize=11, fontweight='bold' ) # (f"{metrics_dict.get(metric, metric)}", fontsize=11, fontweight='bold')

            ax.set_ylabel("Density") # <- Previous "Density Estimate"

            # --- OPTIONAL: apo/holo rug markers beneath the curves ---
            if show_rug:
                # pull numeric values for this metric from the separate group dataframes
                x_apo = (pd.to_numeric(apo_group[metric], errors="coerce").dropna().to_numpy()
                    if isinstance(apo_group, pd.DataFrame) and metric in apo_group.columns
                    else np.array([], dtype=float))
                x_holo = (pd.to_numeric(holo_group[metric], errors="coerce").dropna().to_numpy()
                    if isinstance(holo_group, pd.DataFrame) and metric in holo_group.columns
                    else np.array([], dtype=float))
                if x_apo.size or x_holo.size:
                    # ensure the x-limits include both the curve grid and all raw points
                    xmin = np.nanmin([np.min(x_values), np.min(x_apo) if x_apo.size else np.inf, np.min(x_holo) if x_holo.size else np.inf])
                    xmax = np.nanmax([np.max(x_values), np.max(x_apo) if x_apo.size else -np.inf, np.max(x_holo) if x_holo.size else -np.inf])
                    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                        ax.set_xlim(xmin, xmax)
                    # draw as tiny vertical ticks; don't add legend entries to avoid clutter
                    if x_apo.size:
                        ax.scatter(x_apo, np.full_like(x_apo, ax.get_ylim()[0]), color=apo_style["color"], s=10, alpha=points_alpha, marker="|", clip_on=False, zorder=99)
                    if x_holo.size:
                        ax.scatter(x_holo, np.full_like(x_holo, ax.get_ylim()[0]), color=holo_style["color"], s=10, alpha=points_alpha, marker="|", clip_on=False, zorder=99)

            # Finish rugs section
            if show_legend:
                ax.legend(loc='best', framealpha=0.95, prop={'size': 9}) # ax.legend()
        else:
            ax.set_title(f"No KDE data available for {metric}")

    if not separate_metrics:
        num_metrics = len(classification_metrics)
        fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(7, num_metrics * 3))
        if num_metrics == 1:
            axes = [axes]
        for ax, metric in zip(axes, classification_metrics):
            _plot_one_metric(metric, ax, model_point)

        if supertitle == 1:
            plt.suptitle("AlphaFold-DB Model Classification", fontsize=14, fontweight='normal', y=0.99)
        elif isinstance(supertitle, str) and supertitle:
            plt.suptitle(supertitle, fontsize=14, fontweight='normal', y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        if transparency:
            fig.patch.set_facecolor('none')
            for ax in axes:
                ax.set_facecolor('white')

        plot_filename = f"kde_curve{filename_suffix}"
        save_plot(fig, save_formats, save_dir, plot_filename, show_plot)

    else:
        # Split figures: one figure per metric
        for metric in classification_metrics:
            fig, ax = plt.subplots(figsize=(7, 3))
            _plot_one_metric(metric, ax, model_point)
            if supertitle == 1:
                plt.suptitle("AlphaFold-DB Model Classification", fontsize=14, fontweight='normal', y=0.99)
            #plt.tight_layout(rect=[0, 0, 1, 0.98])

            if transparency:
                fig.patch.set_facecolor('none')
                ax.set_facecolor('white')

            # Save/show
            plot_filename = f"kde_curve_{metric}"
            save_plot(fig, save_formats, save_dir, plot_filename, show_plot)


def save_plot(fig, save_formats: list, save_dir, plot_filename, show_plot):

    if save_dir and save_formats:
        os.makedirs(save_dir, exist_ok=True)

        for saving_format in save_formats:
            save_path = os.path.join(save_dir, f"{plot_filename}.{saving_format}")
            plt.savefig(save_path, dpi='figure', format=saving_format, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(fig)


def mask_density(x_values, density, data, bw, cut=2):
    data = np.asarray(data)
    if len(data) == 0 or np.all(np.isnan(data)):
        return np.zeros_like(x_values)
    # Robustly compute min/max (ignore NaNs)
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    left = data_min - cut * bw
    right = data_max + cut * bw
    mask = (x_values >= left) & (x_values <= right)
    return np.where(mask, density, 0)



def plot_kde_violins(kde_data,metrics,save_dir,apo_values_dict,holo_values_dict,
    all_values_dict,model_df=None,query_df=None,save_formats=['svg'],show_plot=1,
    supertitle=None,set_title=None,filename_suffix='',transparency=0,metrics_names_and_units_dict=None,
    margin_ratio=0.04,violin_width=0.80,normalize_all=0,boxplots=1, #,extend_kde=None
    color_scheme='seaborn1', separate_metrics=0, gridline_intervals: list = None,
    metric_bounds=None,show_legend: bool = True, violin_cut=2.5, axes_map=None
    ):

    """
    Plot stacked and overlapping violin plots for given metrics using precomputed or computed KDE densities.

    For each metric, the function creates a two-layer violin plot:
      - The top layer ("All", y=1) displays a violin plot representing the density of all combined points,
        overlaid with individual apo and holo data points, plus optional markers for query and model values.
      - The bottom layer ("Apo/Holo", y=0) displays two overlapping full violins for apo and holo groups
        using their respective densities.

    Densities are provided as precomputed grids (from KDE or normal distributions). Optionally, densities
    can be recomputed on demand using `extend_kde=1`, with grid extension for smoother tails.

    Arguments:
        kde_data (dict): Dictionary containing density grids and bandwidths for each metric. For each metric, keys should include:
            - 'x_values': the x grid for evaluating densities
            - 'density_all', 'density_apo', 'density_holo': arrays of density values for all, apo, holo groups
            - 'bw_all', 'bw_apo', 'bw_holo': bandwidths used for all, apo, holo KDEs
        metrics (list): List of metric names to plot.
        save_dir (str): Directory to save the generated plots.
        apo_values_dict (dict): Dict mapping metric to apo data points (arrays).
        holo_values_dict (dict): Dict mapping metric to holo data points (arrays).
        all_values_dict (dict): Dict mapping metric to all data points (arrays).
        model_df (DataFrame, optional): DataFrame with AlphaFold (or other reference) model values. Default: None.
        query_df (DataFrame, optional): DataFrame with query/reference point values. Default: None.
        save_formats (list): List of file formats for saving (e.g. ['svg', 'png']). Default: ['svg'].
        show_plot (int): If True (1), displays the plot window. Default: 1.
        supertitle (str, optional): Supertitle for the entire figure. Default: None.
        set_title (int, optional): Title for each metric plot. Default: None.
        filename_suffix (str): String to append to output filename. Default: ''.
        transparency (int): If True (1), uses transparent background for figure. Default: 0.
        metrics_names_and_units_dict (dict, optional): Mapping of metric name to label/unit. Default: None (uses metric names).
        margin_ratio (float): Relative margin to add to x-axis limits. Default: 0.15.
        violin_width (float): Relative width of violins. Default: 0.80.
        normalize_all (0,1): Normalize the height of all violins (All, Apo, Holo) by the same factor (actual global max). 0='All' has its own max.
        extend_kde (int, optional): If 1, recompute KDEs with extended grids for smoother tails. Default: None.
        boxplots (int, optional): If 1, render a boxplot within Apo and Holo violins. Default: 1.
        gridline_intervals:
            None → no gridlines.
            int or List[int] → draw evenly spaced gridlines across the data range (true_data_min..true_data_max) for each provided number.
            "auto" → draw “nice” gridlines at tick positions computed from the axis range (x_plot_min..x_plot_max) using MaxNLocator.

    Details:
        - The function trims density arrays to nonzero regions for visually clean violins (no tails).
        - Density normalization is applied for comparability.
        - The legend indicates sample sizes for each group and marks model/query/reference points.
        - All plotting is done via Matplotlib for flexibility and publication-ready output.
        - If extend_kde is used, KDEs are recomputed for each group using the current data arrays.

    Returns:
        None. (Plots are saved to disk and optionally displayed.)

    Example:
        plot_overlapping_kde_violins(
            kde_data=kde_data_r3,
            metrics=['pocket_rms', 'volume'],
            save_dir='./plots',
            apo_values_dict=apo_dict,
            holo_values_dict=holo_dict,
            all_values_dict=all_dict,
            model_df=df_model,
            query_df=df_query,
            save_formats=['svg', 'png'],
            violin_width=0.85
        )
    """
    #os.makedirs(save_dir, exist_ok=True)

    if not metrics_names_and_units_dict:
        metrics_names_and_units_dict = {m: m for m in metrics}

    num_metrics = len(metrics)
    if not separate_metrics:
        fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(7, num_metrics * 3)) # from violins
        #fig, axes = plt.subplots(num_metrics, 1, figsize=(9, num_metrics * 3.6))
        if num_metrics == 1:
            axes = [axes]

    for idx, metric in enumerate(metrics):
        #ax = axes[idx]
        ax = None
        if axes_map is not None:                       # draw into caller-provided axes (composed figures)
            ax = axes_map.get(metric)
            if ax is None:
                continue
            fig = ax.figure
        elif separate_metrics:
            fig, ax = plt.subplots(figsize=(7, 3))#3.5))
        else:
            ax = axes[idx]

        if metric not in kde_data:
            ax.set_title(f"No KDE data for {metric}")
            continue

        #x_values_apo, x_values_holo, x_values_all = kde_data[metric]["x_values_apo"], kde_data[metric]["x_values_holo"], kde_data[metric]["x_values_all"]
        x_values_all = x_values_apo = x_values_holo = kde_data[metric]['x_values']
        #x_values = kde_data[metric]['x_values']
        dens_all = kde_data[metric].get('density_all', None)
        dens_apo = kde_data[metric]['density_apo']
        dens_holo = kde_data[metric]['density_holo']

        bw_all = kde_data[metric].get('bandwidth') # TODO test
        #bw_all = kde_data[metric].get('bw_all')
        bw_apo = kde_data[metric]['bw_apo']
        bw_holo = kde_data[metric]['bw_holo']

        # Gather all data points for xlim
        apo_points = np.array(apo_values_dict[metric])
        holo_points = np.array(holo_values_dict[metric])
        all_points = np.array(all_values_dict[metric])
        n_apo = len(apo_points)
        n_holo = len(holo_points)
        n_all = len(all_points)

        # Mask/fade tail density when NOT extending KDE (i.e., using wide grid)
        dens_apo = mask_density(x_values_apo, dens_apo, apo_points, bw_apo, cut=violin_cut)
        dens_holo = mask_density(x_values_holo, dens_holo, holo_points, bw_holo, cut=violin_cut)
        if dens_all is not None:
            dens_all = mask_density(x_values_all, dens_all, all_points, bw_all, cut=violin_cut)

        # after mask_density(...)
        peak = float(np.nanmax(dens_apo)) if dens_apo.size else 0.0
        if peak > 0:
            dens_apo[dens_apo < 1e-4 * peak] = 0.0

        peak = float(np.nanmax(dens_holo)) if dens_holo.size else 0.0
        if peak > 0:
            dens_holo[dens_holo < 1e-4 * peak] = 0.0

        if dens_all is not None:
            peak = float(np.nanmax(dens_all))
            if peak > 0:
                dens_all[dens_all < 1e-4 * peak] = 0.0

        # Normalize violins
        if normalize_all == 1:
            # Normalize everything by the global max density
            # Use the max of whichever arrays are present
            present = []
            if dens_all  is not None: present.append(np.nanmax(dens_all))
            if dens_apo  is not None: present.append(np.nanmax(dens_apo))
            if dens_holo is not None: present.append(np.nanmax(dens_holo))
            norm_factor = np.nanmax(present) if present else 1.0
            if dens_all  is not None: dens_all  = dens_all  / norm_factor
            if dens_apo  is not None: dens_apo  = dens_apo  / norm_factor
            if dens_holo is not None: dens_holo = dens_holo / norm_factor
        else:
            # Normalize Apo/Holo for comparability
            norm_factor_apo_holo = np.nanmax([np.nanmax(dens_apo), np.nanmax(dens_holo)])
            dens_apo = dens_apo / norm_factor_apo_holo if norm_factor_apo_holo else dens_apo
            dens_holo = dens_holo / norm_factor_apo_holo if norm_factor_apo_holo else dens_holo
            # Normalize "All" to its own max (full height at y=1)
            if dens_all is not None and np.nanmax(dens_all):
                dens_all = dens_all / np.nanmax(dens_all)

        # Set colors
        colors = VIOLIN_COLOR_SCHEMES[color_scheme]
        apo_color, apo_clr_trans = colors["apo"]["color"], colors["apo"]["alpha"]
        holo_color, holo_clr_trans = colors["holo"]["color"], colors["holo"]["alpha"]
        points_trans = colors["points_alpha"]

        # Add AF-model and Query points to x grid
        x_extras = []
        if model_df is not None and metric in model_df.columns and not model_df[metric].isna().all():
            mv = model_df[metric].values[0]
            if np.isfinite(mv):
                x_extras.append(mv)
        if query_df is not None and metric in query_df.columns and not query_df[metric].isna().all():
            qv = query_df[metric].values[0]
            if np.isfinite(qv):
                x_extras.append(qv)

        # Set x min/max
        x_data_min = np.nanmin(np.concatenate([x_values_all, apo_points, holo_points, all_points] + ([np.array(x_extras)] if x_extras else [])))
        x_data_max = np.nanmax(np.concatenate([x_values_all, apo_points, holo_points, all_points] + ([np.array(x_extras)] if x_extras else [])))

        # Keep smooth tails but decouple axis limits from the extended grid (violins extend beyond Y axes)
        #x_data_min = np.nanmin(np.concatenate([apo_points, holo_points, all_points] + ([np.array(x_extras)] if x_extras else [])))
        #x_data_max = np.nanmax(np.concatenate([apo_points, holo_points, all_points] + ([np.array(x_extras)] if x_extras else [])))

        data_range = x_data_max - x_data_min if x_data_max > x_data_min else 1.0
        margin = margin_ratio * data_range if data_range > 0 else 1
        x_plot_min = x_data_min - margin
        x_plot_max = x_data_max + margin

        # --- Vertical gridlines (optional) ---
        true_data_min = np.nanmin(np.concatenate([apo_points, holo_points, all_points] + ([np.array(x_extras)] if x_extras else [])))
        true_data_max = np.nanmax(np.concatenate([apo_points, holo_points, all_points] + ([np.array(x_extras)] if x_extras else [])))

        if gridline_intervals is not None:
            # NEW: "auto" → use a tick locator on the axis range (inclusive of margins)
            if isinstance(gridline_intervals, str) and gridline_intervals.lower() == "auto":
                # Compute nice tick positions from the final x-limits
                locator = MaxNLocator(nbins="auto", min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])
                ticks = locator.tick_values(x_plot_min, x_plot_max)
                # Clip to the visible region (avoid drawing outside)
                ticks = ticks[(ticks >= x_plot_min) & (ticks <= x_plot_max)]
                for gl in ticks:
                    ax.axvline(gl, color="purple", linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)

            else:
                # Existing behavior: evenly spaced lines over *data* range
                if isinstance(gridline_intervals, int):
                    gridline_intervals = [gridline_intervals]
                for n_intervals in gridline_intervals:
                    if n_intervals and n_intervals > 1:
                        gridlines = np.linspace(true_data_min, true_data_max, n_intervals)
                        for gl in gridlines:
                            ax.axvline(gl, color="purple", linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)

        def draw_violin(ax, x, density, y_center, width, color, alpha, zorder=2, linewidth=1.5):
            mask = (density > 1e-10)
            if not np.any(mask):
                return
            x = x[mask]
            d = density[mask]
            dx = d * (width / 2)
            verts = np.concatenate([
                np.column_stack([x, y_center + dx]),
                np.column_stack([x[::-1], y_center - dx[::-1]])
            ])
            poly = PolyCollection([verts], facecolor=color, alpha=alpha, edgecolor=color, linewidth=linewidth, zorder=zorder)
            ax.add_collection(poly)
            return poly

        # -- All violin at y=1 (classic full)
        draw_violin(ax, x_values_all, dens_all, y_center=1, width=violin_width, color="gray", alpha=0.2, zorder=1) # width=0.92,alpha=0.17

        # Overlay apo/holo points at y=1 (All violin)
        jitter_width = 0.1
        # Compose a list of (label, points, color)
        point_groups = []
        if len(apo_points) > 0:
            point_groups.append(('apo', apo_points, apo_color))
        if len(holo_points) > 0:
            point_groups.append(('holo', holo_points, holo_color))
        point_groups_sorted = sorted(point_groups, key=lambda x: len(x[1]), reverse=True)
        for label, points, color in point_groups_sorted:
            y_jitter = 1 + np.random.uniform(-jitter_width, jitter_width, size=points.shape)
            ax.scatter(points, y_jitter, color=color, s=16, alpha=points_trans, lw=0, zorder=5)

        # Mark Query at y=1
        if query_df is not None and metric in query_df.columns and not query_df[metric].isna().all():
            qv = query_df[metric].values[0]
            if np.isfinite(qv):
                ax.scatter(qv, 1, marker='o', s=25, facecolors='none', edgecolors='black', linewidths=1, zorder=6)

        # Mark Model at y=1
        if model_df is not None and metric in model_df.columns and not model_df[metric].isna().all():
            mv = model_df[metric].values[0]
            if np.isfinite(mv):
                ax.axvline(mv, color="green", linestyle="--", lw=2, zorder=3)

        # -- Apo/Holo violins at y=0 (both full, overlapping), no points
        draw_violin(ax, x_values_apo, dens_apo, y_center=0, width=violin_width, color=apo_color, alpha=apo_clr_trans, zorder=2)
        draw_violin(ax, x_values_holo, dens_holo, y_center=0, width=violin_width, color=holo_color, alpha=holo_clr_trans, zorder=2)

        # --- Add boxplots for Apo and Holo at y=0 ---
        if boxplots == 1:

            box_color = '#2c2c2c' # '#242424' dark grey, '#2c2c2c' less dark grey, '#36454F' Charcoal, 'black'

            def add_boxplot(ax, data, x_offset, zorder=20, width=0.08, alpha=0.5, median_color='white', med_alpha=0.9,
                            # NEW:
                            median_dot=True,                # turn the dot on/off
                            median_dot_size=16,             # px^2 24, 20, 16, 10
                            median_dot_color=None,          # defaults to group color if provided by caller
                            median_dot_edgecolor="white",
                            median_dot_lw=0.9
                            ):

                if len(data) >= 2:
                    #return
                    box = ax.boxplot(
                        data,
                        vert=False,
                        positions=[0 + x_offset],
                        widths=width,
                        patch_artist=True,
                        showcaps=True,
                        showfliers=False,
                        boxprops=dict(facecolor=box_color, color=box_color, linewidth=1.5, zorder=zorder, alpha=alpha),  # Solid black, transparent
                        whiskerprops=dict(color=box_color, linewidth=2.5, zorder=zorder, alpha=alpha),
                        capprops=dict(color=box_color, linewidth=0, zorder=zorder, alpha=alpha),  # No vertical cap lines
                        medianprops=dict(color='none', linewidth=0, zorder=zorder+1, alpha=0),
                        #medianprops=dict(color='none', linewidth=1.5, zorder=zorder+1, alpha=alpha),
                        manage_ticks=False,
                        zorder=zorder)

                    # For all components to be the same alpha, loop over them:
                    for patch in box['boxes']:
                        patch.set_facecolor(box_color)
                        patch.set_alpha(alpha)
                        patch.set_edgecolor('none') # 'black'
                        patch.set_linewidth(0) # 1.5
                    #for median in box['medians']:
                        #median.set_color('white')
                        #median.set_linewidth(1.5)
                        #median.set_alpha(alpha) # prev 0.9, now standard alpha
                    for whisker in box['whiskers']:
                        whisker.set_alpha(alpha)
                    for cap in box['caps']:
                        cap.set_alpha(alpha)

                # --- NEW: tiny colored median dot (on top or instead of the thin white line) ---
                if median_dot and len(data):
                    med = float(np.nanmedian(np.asarray(data, dtype=float)))
                    if np.isfinite(med):
                        ax.scatter(
                            med, 0 + x_offset,
                            s=median_dot_size, marker='o',
                            alpha=med_alpha,
                            facecolor=(median_dot_color if median_dot_color is not None else median_color),
                            edgecolor=median_dot_edgecolor, linewidth=median_dot_lw,
                            zorder=zorder+2, clip_on=False
                        )

            apo_dot_col  = flattened_violin_dot_color(ax, apo_color,  apo_clr_trans)
            holo_dot_col = flattened_violin_dot_color(ax, holo_color, holo_clr_trans)
            add_boxplot(ax, apo_points, x_offset=0, zorder=3, width=0.07, alpha=1, median_color=apo_dot_col, med_alpha=0.9)
            add_boxplot(ax, holo_points, x_offset=0, zorder=3, width=0.07, alpha=1, median_color=holo_dot_col, med_alpha=0.9)

        # Y axis ticks/labels
        ax.set_ylabel("Density")
        ax.set_yticks([1, 0])
        ax.set_yticklabels(['All', 'Apo/\nHolo'])
        ax.set_xlim(x_plot_min, x_plot_max)
        margin = violin_width / 2
        buffer = 0.08
        ax.set_ylim(0 - margin - buffer, 1 + margin + buffer)
        if set_title == 1:
            ax.set_title(metrics_names_and_units_dict.get(metric, metric), fontsize=11, fontweight='bold')

        # Legend with sample sizes, squares
        handles = [
            Line2D([], [], marker='s', color='gray', markersize=9, linestyle='None', alpha=0.2, label=f"All (n={n_all})"),
            Line2D([], [], marker='s', color=apo_color, markersize=9, linestyle='None', alpha=apo_clr_trans, label=f"Apo (n={n_apo})"),
            Line2D([], [], marker='s', color=holo_color, markersize=9, linestyle='None', alpha=holo_clr_trans, label=f"Holo (n={n_holo})"),
            #Line2D([], [], marker='o', color='black', markersize=5, linestyle='None', markerfacecolor='white', markeredgecolor='black', label="Query"),
            #Line2D([], [], color='green', linestyle="--", lw=2, label="AF model"),
        ]

        # Add query and model legend only if present
        if (query_df is not None and metric in query_df.columns
            and query_df[metric].notna().any()
            and np.isfinite(query_df[metric].values[0])):
            handles.append(Line2D([], [], marker='o', color='black', markersize=5, linestyle='None', markerfacecolor='white', markeredgecolor='black', label="Query"))
        if (model_df is not None and metric in model_df.columns
            and model_df[metric].notna().any()
            and np.isfinite(model_df[metric].values[0])):
            handles.append(Line2D([], [], color='green', linestyle="--", lw=2, label="AF model"))

        if show_legend:
            leg = ax.legend(handles=handles, loc='best', framealpha=0.95, prop={'size': 9})
            leg.set_zorder(999)

        if separate_metrics and axes_map is None:
            if transparency:
                fig.patch.set_facecolor('none')
                ax.set_facecolor('white')

            # Save/show
            plot_filename = f"kde_violin_{metric}"
            save_plot(fig, save_formats, save_dir, plot_filename, show_plot)

    if axes_map is not None:
        return          # caller owns the figure; nothing to save

    if not separate_metrics:
        if supertitle:
            plt.suptitle(supertitle, fontsize=14, fontweight='normal', y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        if transparency:
            fig.patch.set_facecolor('none')
            for ax in axes:
                ax.set_facecolor('white')

        #fig.text(0.99, 0.01, f"Priors: {prior_mode} (α={alpha_prior})", ha="right", va="bottom", fontsize=8, color="#444")

        # Save/show & close
        plot_filename = f"kde_violin{filename_suffix}"
        save_plot(fig, save_formats, save_dir, plot_filename, show_plot)


# --- Unified labeling helpers -------------------------------------------------
#
# SINGLE SOURCE OF TRUTH for the confidence-bar verbal bands (AHOJ-016).
# The authoritative, citable write-up lives in docs/confidence_bar_spec.md; the
# constants below are what the code actually reads, and the documented table is
# derived from them by band_spec_table() and pinned by
# tests/test_confidence_bar_bands.py. Do NOT hard-code these thresholds anywhere
# else -- read from here. The JS renderer
# (analysis/interactive_confbars_prototype/confbars.js) keeps a mirror that the
# same regression test holds numerically identical to these values.

# There are TWO native scales, each with its own inclusive-upper-edge table:
#   * densities / overlaps  -> quantities already on [0,1]      (UNIT_BAND_EDGES)
#   * apo/holo probability  -> winning posterior on [0.5, 1.0]  (PROB_BAND_EDGES)
# Both use the same five VERBAL_BANDS. The probability edges are defined DIRECTLY
# on the probability scale as round, interpretable numbers (the same approach the
# IPCC / ICD 203 verbal-probability scales use) -- there is NO sigmoid remap.
# (History: the earlier revision remapped the probability onto [0,1] with a
# sigmoid of gain k=1.64; that magic constant and its derived ugly edges were
# replaced by these direct edges. See KNOWN_ISSUES.md AHOJ-016.)

# The five verbal quality labels, lowest -> highest confidence.
VERBAL_BANDS = ("Very low", "Low", "Moderate", "High", "Very high")

# Density/overlap quantities: inclusive upper edges on the [0,1] scale:
#   u <= 0.20 -> Very low ; 0.20 < u <= 0.30 -> Low ; 0.30 < u <= 0.60 -> Moderate
#   ; 0.60 < u <= 0.85 -> High ; u > 0.85 -> Very high.
UNIT_BAND_EDGES = (0.20, 0.30, 0.60, 0.85)

# Apo/holo winning probability p = max(P(apo), P(holo)) in [0.5, 1.0]: inclusive
# upper edges defined DIRECTLY on the probability scale (no remap):
#   p <= 0.55 -> Very low ; 0.55 < p <= 0.60 -> Low ; 0.60 < p <= 0.70 -> Moderate
#   ; 0.70 < p <= 0.85 -> High ; p > 0.85 -> Very high.
PROB_BAND_EDGES = (0.55, 0.60, 0.70, 0.85)


def _unitize(value, *, src_min=0.0, src_max=1.0):
    """Map value from [src_min, src_max] -> [0,1] with NaN-safe clamps.

    Densities and overlaps already live on [0,1] and are labeled directly through
    this (identity) map; see docs/confidence_bar_spec.md and _verbal_from_unit.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    # avoid zero division, clamp input, and scale
    rng = float(src_max - src_min) if src_max != src_min else 1.0
    v = (float(value) - src_min) / rng
    if np.isnan(v):
        return np.nan
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def band_spec_table():
    """Derive the confidence-bar band table from the single source of truth
    (VERBAL_BANDS / UNIT_BAND_EDGES / PROB_BAND_EDGES). One dict row per verbal
    band with keys:

        label          verbal descriptor
        u_lo, u_hi     density/overlap band edges on [0,1]            (u_hi inclusive)
        p_lo, p_hi     apo/holo winning-probability band on [0.5,1]   (p_hi inclusive)

    This is the machine-readable form of docs/confidence_bar_spec.md; the doc and
    tests/test_confidence_bar_bands.py read it so the published numbers cannot
    drift from the code.
    """
    u_lo = (0.0,) + tuple(UNIT_BAND_EDGES)
    u_hi = tuple(UNIT_BAND_EDGES) + (1.0,)
    p_lo = (0.5,) + tuple(PROB_BAND_EDGES)
    p_hi = tuple(PROB_BAND_EDGES) + (1.0,)
    rows = []
    for i, label in enumerate(VERBAL_BANDS):
        rows.append({
            "label": label,
            "u_lo": u_lo[i], "u_hi": u_hi[i],
            "p_lo": p_lo[i], "p_hi": p_hi[i],
        })
    return rows


def _verbal_from_value(value, edges):
    """Return the verbal band for `value` given inclusive upper `edges` (ascending):
    value <= edges[0] -> VERBAL_BANDS[0], ..., value > edges[-1] -> VERBAL_BANDS[-1].
    Shared by the density/overlap scale (edges=UNIT_BAND_EDGES on [0,1]) and the
    apo/holo probability scale (edges=PROB_BAND_EDGES on [0.5,1]). NaN/None -> "n/a".
    """
    if value is None or np.isnan(value):
        return "n/a"
    for edge, label in zip(edges, VERBAL_BANDS):
        if value <= edge:
            return label
    return VERBAL_BANDS[-1]


def _verbal_from_unit(u):
    """Verbal band for a unitized [0,1] density/overlap score (reads UNIT_BAND_EDGES;
    see _verbal_from_value). u may be NaN."""
    return _verbal_from_value(u, UNIT_BAND_EDGES)


def verbose_probability_label(p_apo, p_holo, classification):
    if p_apo is None or p_holo is None or np.isnan(p_apo) or np.isnan(p_holo):
        return "Unclassified"

    abs_score = max(float(p_apo), float(p_holo))  # winning posterior in [0.5, 1.0]
    descriptor = _verbal_from_value(abs_score, PROB_BAND_EDGES)

    return f"Classification: {classification} - Probability: {descriptor} ({abs_score:.2f})"


def verbose_scoring_label(score):
    """
    Create a verbose label for a metric scaled on [0,1] (densities / overlaps).
    The numeric shown in parentheses is the raw [0,1] value at 2 dp; the verbal
    band comes from _verbal_from_unit (see docs/confidence_bar_spec.md).
    """
    abs_score = abs(score) if score is not None else None
    u = _unitize(abs_score, src_min=0.0, src_max=1.0)
    descriptor = _verbal_from_unit(u)
    if abs_score is None or (isinstance(abs_score, float) and np.isnan(abs_score)):
        return "n/a"
    return f"{descriptor} ({abs_score:.2f})"


def add_gradient(ax, min_val, max_val, cmap, bar_height,
                 n_points=256, cutoff=None, alpha=1.0):
    """
    Draw a 1D horizontal gradient on `ax` from min_val→max_val.
    If cutoff is not None, pixels with x > cutoff get alpha=0.
    """
    # 1D array of positions
    xs = np.linspace(min_val, max_val, n_points)
    # normalize to [0,1] for colormap lookup
    norm = (xs - min_val) / (max_val - min_val)
    colors = cmap(norm)           # shape (n_points, 4)
    # apply uniform alpha, then mask
    colors[:, -1] = alpha
    # Only mask when cutoff is a real number
    if cutoff is not None and not (isinstance(cutoff, float) and np.isnan(cutoff)):
        colors[:, -1] = np.where(xs <= cutoff, alpha, 0.0)
    ax.imshow(colors.reshape(1, n_points, 4),
              aspect='auto', extent=(min_val, max_val, 0, bar_height),
              origin='lower')


def draw_gradient_bar(ax, value, min_val, max_val, cmap, label,
                      bar_height=0.6, n_points=256, stop_at_value=False,
                      y1label=None, y2label=None):
    cutoff = value if stop_at_value else None

    # Draw the gradient (white→green etc)
    add_gradient(ax, min_val, max_val, cmap, bar_height, n_points=n_points, cutoff=cutoff)

    # If requested, draw a vertical line at `value`
    if stop_at_value:
        ax.axvline(x=value, color='grey', linewidth=1, alpha=0.5, clip_on=False)

    # Verbose label
    label_text = verbose_scoring_label(value)
    ax.set_xlabel(f"{label}: {label_text}")

    # Styling
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(0, bar_height)
    ax.set_xticks([])

    if y1label:
        ax.set_ylabel(y1label, rotation=90, labelpad=15, va='center')
    if y2label:
        ax2 = ax.twinx()
        ax2.set_ylabel(y2label, rotation=270, labelpad=15, va='center')

    ax.set_yticks([])


def draw_gradient_score_bar(ax, value, p_apo, p_holo, classification,
                                    min_val=-1.0, max_val=1.0,
                                    cmap=plt.get_cmap("bwr"),
                                    label=None, bar_height=0.6,
                                    marker_width=0.02, n_points=256):
    # Draw the colored bar
    add_gradient(ax, min_val, max_val, cmap, bar_height,
                 n_points=n_points, alpha=0.7)

    if not np.isnan(value):
        # Draw the hollow rectangle at `value`
        clipped = max(min_val, min(value, max_val))
        rect = patches.Rectangle(
            (clipped - marker_width/2, 0),
            marker_width, bar_height,
            facecolor="none", edgecolor="black",
            linewidth=3, clip_on=False
        )
        ax.add_patch(rect)

    # Labels & annotations (unchanged)
    prob_label = verbose_probability_label(p_apo, p_holo, classification)
    ax.set_xlabel(f"{prob_label}", fontweight='bold')
    #ax.set_xlabel(f"{label}: {prob_label}", fontweight='bold')
    ax.text(0.01, -0.10, rf"$P_{{\mathrm{{APO}}}}$: {p_apo:.2f}",transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.text(0.99, -0.10, rf"$P_{{\mathrm{{HOLO}}}}$: {p_holo:.2f}",transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.text(0.01, 1.15, "APO", transform=ax.transAxes,ha="left", va='bottom', fontsize=8.5, fontweight='bold')
    ax.text(0.99, 1.15, "HOLO", transform=ax.transAxes,ha="right", va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])


def plot_metric_summary(metric_name, data, CMAP_SEP, metrics_names_dict=dict(),
                        save_dir=None, save_formats=[], show_plot=None, supertitle=0, filename_suffix='', transparency=None):
    """
    Create a single figure for one metric, with four color-gradient bars:
      1) Score (–1 to +1)  [bwr]
      2) Separation from overlap = (1 – overlap) [white→darkgreen]
      3) Separation from mode overlap = (1 – mode_overlap) [white→darkgreen]
      4) Normalized density (0 to 1) [white→darkgreen]
    """
    import matplotlib.lines as mlines

    p_apo, p_holo = data.get("P(apo)", np.nan), data.get("P(holo)", np.nan)
    classification = data.get("classification", 'n/a')
    score = data.get("score", 0.0)
    overlap = data.get("overlap", np.nan)
    mode_overlap = data.get("mode_overlap", np.nan)
    normalized_density_strongest = data.get("normalized_density_strongest", 0.0)
    #normalized_density_total = data.get("normalized_density_total", 0.0)
    normalized_density_combined = data.get("normalized_density_combined", 0.0)

    # Transform Overlap to Separation (scale 0 - 1)
    separation = 1 - overlap if not np.isnan(overlap) else np.nan # <- Previously else 0.0
    separation_mode = 1 - mode_overlap if not np.isnan(mode_overlap) else np.nan # <- Previously else 0.0

    fig, axes = plt.subplots(nrows=5, figsize=(5, 3.5))
    #fig, axes = plt.subplots(nrows=5, figsize=(6, 3.5))
    #fig, axes = plt.subplots(nrows=6, figsize=(6, 3.5))
    #fig, axes = plt.subplots(nrows=4, figsize=(6, 3)) #figsize=(6, 8), sharex=False
    metric_name_raw = metric_name
    if metric_name != 'Overall':
        metric_name = metrics_names_dict.get(metric_name, metric_name)
    else:
        metric_name_raw = f"{metric_name_raw}{filename_suffix}"
    if supertitle == 1:
        fig.suptitle(f"{metric_name}", fontsize=14, fontweight='bold', y=0.92)

    # 1) Score: range –1 to +1, colormap: 'bwr'
    draw_gradient_score_bar(axes[0], value=score, p_apo=p_apo, p_holo=p_holo, classification=classification,
                    min_val=-1, max_val=1, cmap=plt.get_cmap("bwr"), label="Classification Probability")

    # 2) Normalized density: range 0 to 1
    draw_gradient_bar(axes[1], value=normalized_density_strongest, min_val=0, max_val=1, 
                      cmap=CMAP_SEP, label="Density at Classification Point (top group)", stop_at_value=True)

    # 3) Normalized density combined: range 0 to 1
    draw_gradient_bar(axes[2], value=normalized_density_combined, min_val=0, max_val=1, 
                      cmap=CMAP_SEP, label="Density at Classification Point (combined)", stop_at_value=True)

    # 4) Total Separation (from 'overlap'): range 0 to 1
    draw_gradient_bar(axes[3], value=separation, min_val=0, max_val=1, 
                      cmap=CMAP_SEP, label="Total Density Separation", stop_at_value=True)

    # 5) Peak Separation (from 'mode_overlap'): range 0 to 1
    draw_gradient_bar(axes[4], value=separation_mode, min_val=0, max_val=1, 
                      cmap=CMAP_SEP, label="Peak Density Separation", stop_at_value=True)

    # 5) Normalized density total: range 0 to 1
    #draw_gradient_bar(axes[4], value=normalized_density_total, min_val=0, max_val=1, cmap=CMAP_SEP, label="Density at Classification Point (total)", stop_at_value=True)

    plt.tight_layout(rect=[0, 0, 1, 1]) # [0, 0, 1, 0.96], [0, 0, 1, 0.99], [0, 0, 1, 0.80]

    # ── Begin bracket‐drawing block ──
    fig = plt.gcf()

    # ── BRACKET FOR “Classification bars” (covers bars 0, 1, 2) ──

    # 1) Find top of bar‐row 0 and bottom of bar‐row 2
    bbox0 = axes[0].get_position()
    bbox2 = axes[2].get_position()
    y_top0 = bbox0.y0 + bbox0.height
    y_bot2 = bbox2.y0

    # 2) Place bracket just to the RIGHT of bars 0–2
    max_x1_012 = max(ax_.get_position().x1 for ax_ in axes[0:3])
    x_br0 = max_x1_012 + 0.03  # shift 1% of figure width right

    # 3) Draw the vertical stem of the bracket
    line0 = mlines.Line2D(
        [x_br0, x_br0],
        [y_bot2, y_top0],
        transform=fig.transFigure,
        color="black", linewidth=1.5,
    )
    fig.add_artist(line0)

    # 4) Draw the horizontal “caps” extending leftward from that stem
    cap_len = 0.015
    # bottom cap (y = y_bot2)
    bot_cap0 = mlines.Line2D(
        [x_br0, x_br0 - cap_len],
        [y_bot2, y_bot2],
        transform=fig.transFigure,
        color="black", linewidth=1.5,
    )
    fig.add_artist(bot_cap0)
    # top cap (y = y_top0)
    top_cap0 = mlines.Line2D(
        [x_br0, x_br0 - cap_len],
        [y_top0, y_top0],
        transform=fig.transFigure,
        color="black", linewidth=1.5,
    )
    fig.add_artist(top_cap0)

    # 5) Place the vertical text label “Classification bars” just to the right of the stem
    y_mid0 = 0.5 * (y_top0 + y_bot2)
    fig.text(
        x_br0 + 0.005,  # slightly right of the stem
        y_mid0,
        "Classification", # Classification, CLASSIFICATION
        transform=fig.transFigure,
        rotation=270,
        va="center",
        ha="left",
        fontsize=9.8, # 9, 8.5
        fontweight="bold"
    )

    # ── BRACKET FOR “General” (covers bars 3, 4) ──
    # 1) Find top of bar‐row 3 and bottom of bar‐row 4
    bbox3 = axes[3].get_position()
    bbox4 = axes[4].get_position()
    y_top3 = bbox3.y0 + bbox3.height
    y_bot4 = bbox4.y0

    # 2) Place bracket just to the RIGHT of bars 3–4
    max_x1_34 = max(ax_.get_position().x1 for ax_ in axes[3:5])
    x_br1 = max_x1_34 + 0.03

    # 3) Draw the vertical stem for the second bracket
    line1 = mlines.Line2D(
        [x_br1, x_br1],
        [y_bot4, y_top3],
        transform=fig.transFigure,
        color="black", linewidth=1.5,
    )
    fig.add_artist(line1)

    # 4) Draw the horizontal “caps” extending leftward
    bot_cap1 = mlines.Line2D(
        [x_br1, x_br1 - cap_len],
        [y_bot4, y_bot4],
        transform=fig.transFigure,
        color="black", linewidth=1.5,
    )
    fig.add_artist(bot_cap1)
    top_cap1 = mlines.Line2D(
        [x_br1, x_br1 - cap_len],
        [y_top3, y_top3],
        transform=fig.transFigure,
        color="black", linewidth=1.5,
    )
    fig.add_artist(top_cap1)

    # 5) Place the vertical text label “General” just to the right of the stem
    y_mid1 = 0.5 * (y_top3 + y_bot4)
    fig.text(
        x_br1 + 0.005,
        y_mid1,
        "General",
        transform=fig.transFigure,
        rotation=270,
        va="center",
        ha="left",
        fontsize=9.8,
        fontweight="bold"
    )

    # 6) Ensure the right edge isn’t clipped
    fig.axes[0].get_xlim()  # just to trigger an autoscale  # cur_xlim =

    # ── End bracket‐drawing block ──

    # Handle transparency
    if transparency:
        fig.patch.set_facecolor('none')  # Make the overall figure background transparent
        for ax in axes:
            ax.set_facecolor('white')    # Set each individual plot background to white

    plot_filename = f"confidence_bars_{metric_name_raw}"
    save_plot(fig, save_formats, save_dir, plot_filename, show_plot)

    #for saving_format in save_formats:
    #    save_path = os.path.join(save_dir, f"{plot_filename}.{saving_format}")
    #    plt.savefig(save_path, dpi='figure', format=saving_format, bbox_inches="tight")
    #if show_plot:
    #    plt.show()
    #plt.close(fig)


def confidence_bars_kde(results_dict, metrics_names_dict=None,
                     save_dir=None, save_formats=[], show_plot=None, supertitle=0, filename_suffix='', transparency=None):
    """
    Loop over the metrics in 'results_dict' and create a separate figure for each metric.
    This version includes overall metrics by remapping their keys.
    """

    for metric_name, data in results_dict.items():
        if metric_name == "variables":
            continue
        elif metric_name == "overall":
            # Remap overall keys to those expected by plot_metric_summary:
            overall_data = {
                "classification": data.get("overall_classification", np.nan),
                "P(apo)": data.get("avg_P(apo)", np.nan),
                "P(holo)": data.get("avg_P(holo)", np.nan),
                "score": data.get("overall_score", 0.0),
                "overlap": data.get("overall_overlap", np.nan),
                "mode_overlap": data.get("overall_mode_overlap", np.nan),
                "normalized_density_strongest": data.get("avg_normalized_density_strongest", 0.0),
                "normalized_density_total": data.get("avg_normalized_density_total", 0.0),
                "normalized_density_combined": data.get("avg_normalized_density_combined", 0.0)
            }
            # Plot the overall metrics with the label "Overall"
            plot_metric_summary("Overall", overall_data, CMAP_SEP, metrics_names_dict, save_dir, save_formats, show_plot, supertitle, filename_suffix, transparency)
        else:
            plot_metric_summary(metric_name, data, CMAP_SEP, metrics_names_dict, save_dir, save_formats, show_plot, supertitle, filename_suffix, transparency)

