# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:43:38 2025

@author: Christos P. Feidakis
"""

## stats\_plot.py (plotting-only, optional)

# stats_plot.py
from __future__ import annotations
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

#import seaborn as sns

plt.ion()
#plt.xkcd() # graphic style

# These are stubs that only consume numbers/curves; put no math here.


def plot_dp_density_curves(x: np.ndarray, f_a: np.ndarray, f_h: np.ndarray, *, title: str | None = None):
    fig, ax = plt.subplots()
    ax.plot(x, f_a, label="Apo")
    ax.plot(x, f_h, label="Holo")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    if title:
        ax.set_title(title)
    ax.legend()
    return fig, ax


#### transfered functions


def is_font_installed(font_name):
    """Returns True if font_name is installed on this system (.ttf or .otf)."""
    try:
        font_paths = set(fm.findSystemFonts(fontext='ttf')) | set(fm.findSystemFonts(fontext='otf'))
        available_fonts = set()
        for fp in font_paths:
            try:
                available_fonts.add(fm.FontProperties(fname=fp).get_name())
            except Exception:
                continue
        return font_name in available_fonts
    except Exception:
        return False


def set_plot_font(font_name: str = 'Helvetica', font_size: float = 10, svg_fonttype: str = 'none', fontfile: str = None):
    # 'Arial', 'Helvetica', 'Helvetica Neue', font_size = 9.5
    """
    Resets current Matplotlib rc parameters and sets default font style and size for plots.
    svg.fonttype: supported values are ['none', 'path']
    'none' (default): Fonts referenced by name (not embedded).
    'path': Glyphs are converted to SVG paths—what you want!
    'svgfont': Embeds a subset of the font as an SVG font (not supported in all browsers).

    Note(rdk): Helvetica is not available nor easily installed on linux

    Returns
    -------
    None.

    """
    # Set plot font & style
    plt.rcdefaults()  # Resets rcParams to their default settings
    #plt.xkcd() # Set handwriting style for all plots
    # Show Warning if font is not installed
    if not is_font_installed(font_name):
        print(f"\nWARNING: Font '{font_name}' is not installed. Fallback font will be used.")

    #import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = [font_name]
    mpl.rcParams['font.family'] = 'sans-serif'

    #mpl.rcParams['font.family'] = font_name
    #mpl.rcParams['font.sans-serif'] = [font_name]
    mpl.rcParams['font.size'] = font_size
    #mpl.rcParams['font.weight'] = 400 # 400 = normal, 700 = bold
    mpl.rcParams['svg.fonttype'] = svg_fonttype

    # Test
    #t = plt.text(0.5, 0.5, "Hello QWERTY\nThe big brown fox\njumps over the lazy wolf", fontsize=14, fontweight='bold')
    #print("Font actually used:", t.get_fontname())



def _group_series(df, group_col, metric, order=("Apo", "Holo", "Other")):
    data = []
    for g in order:
        vals = pd.to_numeric(df.loc[df[group_col] == g, metric], errors="coerce").dropna().to_numpy()
        data.append(vals)
    return data

def _jitter(n, width=0.12, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    return rng.uniform(-width, width, size=n)


def visualize_metric_distributions(df_candidates, df_new, metrics, group_col="apoholo_assignment", 
                                   apo_label="Apo", holo_label="Holo", comparisons=None):
    """
    For each metric, create a plot that shows the distribution of values for the aggregated Apo and Holo groups
    (combining sub-categories such as A and A*, or H and H*) and overlays the new data point value.
    This function also overlays the actual data points for further insight.
    
    Parameters:
    - df_candidates: DataFrame containing candidate (reference) data with group labels.
    - df_new: DataFrame (typically one row) with the new data point.
    - metrics: list of metric column names.
    - group_col: name of the column holding the group labels.
    - apo_label/holo_label: labels for the groups.
    - comparisons: (optional) dictionary with pre-computed stats (e.g., medians) to annotate.
    """

    # aggregate A/A* → Apo, H/H* → Holo
    df_candidates = df_candidates.copy()
    df_candidates[group_col] = df_candidates[group_col].apply(
        lambda x: "Apo" if str(x).startswith("A")
        else ("Holo" if str(x).startswith("H") else "Other")
    )

    order = ("Apo", "Holo")  # same order as before
    colors = {"Apo": "blue", "Holo": "red", "Other": "gray"}

    num = len(metrics)
    fig, axes = plt.subplots(nrows=num, ncols=1, figsize=(8, 4 * num))
    if num == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data = _group_series(df_candidates, group_col, metric, order=order)
        positions = np.arange(1, len(order) + 1)

        bp = ax.boxplot(
            data, positions=positions, widths=0.6, patch_artist=True, manage_ticks=False
        )
        # style boxes
        for patch, g in zip(bp["boxes"], order):
            patch.set_facecolor(colors[g]); patch.set_alpha(0.5); patch.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(1.5)

        # overlay jittered points
        for pos, g, vals in zip(positions, order, data):
            if vals.size:
                xj = pos + _jitter(vals.size, width=0.12)
                ax.scatter(xj, vals, s=18, c="black", alpha=0.5, linewidths=0)

        # new datapoint line
        if metric in df_new.columns and len(df_new) >= 1:
            new_val = float(pd.to_numeric(df_new[metric].iloc[0], errors="coerce"))
            if np.isfinite(new_val):
                ax.axhline(new_val, linestyle="--", color="red", label="New Data Point")

        # optional median annotations
        if comparisons is not None and metric in comparisons:
            for pos, g in zip(positions, order):
                if g in comparisons[metric] and "median" in comparisons[metric][g]:
                    m = comparisons[metric][g]["median"]
                    if m is not None and np.isfinite(m):
                        ax.text(pos, m, f"median={m:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(positions)
        ax.set_xticklabels(order)
        ax.set_title(metric)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_normalized_distances(classification_result, metrics):
    """
    Plot a bar chart of the per-metric normalized distances (z-scores) 
    for the new data point relative to the Apo and Holo groups.

    Parameters:
    - classification_result: dictionary output from classify_point containing per_metric z-scores.
    - metrics: list of metric names.
    """
    #import numpy as np

    z_apo_scores = [classification_result['per_metric'][m]['z_apo'] for m in metrics]
    z_holo_scores = [classification_result['per_metric'][m]['z_holo'] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, z_apo_scores, width, label="Apo", color="blue", alpha=0.5) # bars1 = 
    ax.bar(x + width/2, z_holo_scores, width, label="Holo", color="red", alpha=0.5) # bars2 = 

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Normalized distance (z-score)")
    ax.set_title("Normalized distances of new data point to Apo and Holo distributions")
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_robust_distances(classification_result, metrics):
    """
    Plot a bar chart of the per-metric normalized distances (z-scores) 
    for the new data point relative to the Apo and Holo groups.
    
    Parameters:
    - classification_result: dictionary output from classify_point containing per_metric z-scores.
    - metrics: list of metric names.
    """
    #import numpy as np
    
    z_apo_scores = [classification_result['per_metric'][m]['robust_distance_apo'] for m in metrics]
    z_holo_scores = [classification_result['per_metric'][m]['robust_distance_holo'] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, z_apo_scores, width, label="Apo", color="blue", alpha=0.5) # bars1 = 
    ax.bar(x + width/2, z_holo_scores, width, label="Holo", color="red", alpha=0.5) # bars2 = 

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Normalized distance (robust)")
    ax.set_title("Normalized distances of new data point to Apo and Holo distributions")
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def visualize_metric_distributions_composite(df_candidates, df_new, metrics, 
                                             group_col="apoholo_assignment", 
                                             classification_composite_pocket=None,
                                             save_dir=None, save_formats=[],
                                             apo_label="Apo", holo_label="Holo"):
    """
    For each metric, create a plot that shows the distribution of candidate values for the 
    aggregated Apo and Holo groups (merging sub-categories like A/A* or H/H*) and overlays 
    the new data point (from df_new). Additionally, if provided, the composite distances 
    (from the classification result) are annotated on the plot.
    
    Parameters:
    - df_candidates: DataFrame with candidate data including group labels.
    - df_new: DataFrame (typically one row) with the new data point.
    - metrics: list of metric column names.
    - group_col: column name with group labels.
    - classification_composite_pocket: (optional) dictionary from classify_point_composite.
    - save_dir: str, the directory where the SVG file will be saved.
    - save_formats: list of filetypes as str (ex. ['svg']), for saving the figure (plt.savefig(.., format=..)).
    - apo_label/holo_label: labels for the groups.
    """
    # Create an aggregated group column so that any value starting with "A" becomes "Apo" 
    # and any value starting with "H" becomes "Holo"
    if save_formats is None:
        save_formats = []

    df_candidates = df_candidates.copy()
    df_candidates["group_simple"] = df_candidates[group_col].apply(
        lambda x: "Apo" if str(x).startswith("A")
        else ("Holo" if str(x).startswith("H") else "Other")
    )

    order = ("Apo", "Holo")
    colors = {"Apo": "blue", "Holo": "red", "Other": "gray"}

    num = len(metrics)
    fig, axes = plt.subplots(nrows=num, ncols=1, figsize=(8, 4 * num))
    if num == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data = _group_series(df_candidates, "group_simple", metric, order=order)
        positions = np.arange(1, len(order) + 1)

        bp = ax.boxplot(
            data, positions=positions, widths=0.6, patch_artist=True, manage_ticks=False
        )
        for patch, g in zip(bp["boxes"], order):
            patch.set_facecolor(colors[g]); patch.set_alpha(0.5); patch.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(1.5)

        for pos, g, vals in zip(positions, order, data):
            if vals.size:
                xj = pos + _jitter(vals.size, width=0.12)
                ax.scatter(xj, vals, s=18, c="black", alpha=0.5, linewidths=0)

        # new datapoint line
        if metric in df_new.columns and len(df_new) >= 1:
            new_val = float(pd.to_numeric(df_new[metric].iloc[0], errors="coerce"))
            if np.isfinite(new_val):
                ax.axhline(new_val, linestyle="--", color="red", label="New Data Point")

        # annotate composite distances if provided
        if classification_composite_pocket is not None:
            pm = classification_composite_pocket.get("per_metric", {})
            if metric in pm:
                for pos, g in zip(positions, order):
                    key = f"composite_distance_{g.lower()}"
                    if key in pm[metric]:
                        v = pm[metric][key]
                        if v is not None and np.isfinite(v):
                            ax.text(pos, ax.get_ylim()[1], f"{key}: {v:.3f}",
                                    ha="center", va="top", fontsize=9)

        ax.set_xticks(positions)
        ax.set_xticklabels(order)
        ax.set_title(metric)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    plt.tight_layout()
    if save_dir and len(save_formats):
        import os
        os.makedirs(save_dir, exist_ok=True)
        for fmt in save_formats:
            fig.savefig(os.path.join(save_dir, f"metric_distributions_composite.{fmt}"), format=fmt, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_composite_distances(classification_composite_pocket, metrics, save_dir=None, save_formats=[], 
                             apo_label="Apo", holo_label="Holo", include_all=False):
    """
    Create a bar chart of the per-metric composite distances for the new data point relative 
    to the Apo and Holo groups, and optionally the All group.
    
    If a composite distance is None (e.g. if a group is missing), it is replaced with 0 for plotting.
    
    Parameters:
      - classification_composite_pocket: dictionary output from classify_point_composite.
      - metrics: list of metric names.
      - save_dir: str, the directory where the file(s) will be saved.
      - save_formats: list of file formats as strings (e.g. ['svg']).
      - apo_label/holo_label: labels for the groups.
      - include_all: bool, if True the composite distance for the "All" group will also be plotted.
    """
    x = np.arange(len(metrics))

    # Replace None values with 0.
    def safe_value(val):
        return val if val is not None else 0

    if include_all:
        composite_apo = [ safe_value(classification_composite_pocket['per_metric'][m].get('composite_distance_apo')) for m in metrics ]
        composite_holo = [ safe_value(classification_composite_pocket['per_metric'][m].get('composite_distance_holo')) for m in metrics ]
        composite_all = [ safe_value(classification_composite_pocket['per_metric'][m].get('composite_distance_all')) for m in metrics ]
        width = 0.25  # narrower bar width for three bars per metric

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, composite_apo, width, label=apo_label, color="blue", alpha=0.5) # bars1 = 
        ax.bar(x, composite_holo, width, label=holo_label, color="red", alpha=0.5) # bars2 = 
        ax.bar(x + width, composite_all, width, label="All", color="gray", alpha=0.5) # bars3 = 
    else:
        composite_apo = [ safe_value(classification_composite_pocket['per_metric'][m].get('composite_distance_apo')) for m in metrics ]
        composite_holo = [ safe_value(classification_composite_pocket['per_metric'][m].get('composite_distance_holo')) for m in metrics ]
        width = 0.35  # bar width

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, composite_apo, width, label=apo_label, color="blue", alpha=0.5) # bars1 = 
        ax.bar(x + width/2, composite_holo, width, label=holo_label, color="red", alpha=0.5) # bars2 = 

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Composite Distance")
    ax.set_title("Composite Distances of New Data Point to Each Group (per metric)")
    ax.legend()

    plt.tight_layout()

    # Save the figure if a directory and formats are provided.
    if save_dir is not None:
        plot_filename = "composite_distances_per_metric"
        for saving_format in save_formats:
            save_path = os.path.join(save_dir, f"{plot_filename}.{saving_format}")
            plt.savefig(save_path, dpi='figure', format=saving_format)
    plt.show()
    plt.close(fig)


def plot_all_metrics_with_all_violin_overlap(df, metrics, save_dir, model_df=None, query_df=None, save_formats=[],
                                             show_plot=None, supertitle=None, filename_suffix='', transparency=None,
                                             gridline_intervals=None, metrics_names_and_units_dict=None):
    """
    Creates a figure with one subplot per metric.
    For each metric, two horizontal layers are used:
      - Layer "0" (y=0): Underlying Apo/Holo violins (drawn first).
      - Layer "1" (y=1): The combined All violin (gray) with overlaid colored data points.
    A unified legend is added that includes sample sizes for All, Apo, and Holo as well as reference lines.
    The y-axis is labeled "Density".

    Parameters:
      - df: pandas.DataFrame containing the data.
      - metrics: list of metric column names.
      - save_dir: directory where the figures will be saved.
      - model_df, query_df: single-row DataFrames with reference values (optional).
      - save_formats: list of strings, file extensions to save (e.g., ['svg', 'png']).
      - show_plot: if True, display the plot.
    """
    import seaborn as sns
    import pandas as pd

    # Dictionary of metric labels
    if not metrics_names_and_units_dict:
        metrics_names_and_units_dict = {
            "pocket_rms": "Pocket all-atom RMSD (Å)",
            "volume": "Volume (Å³)",
            "sasa": "Solvent Accessible Surface Area (Å²)",
            "sasa_avg": "Solvent Accessible Surface Area per-residue avg. (Å²)",
            "asa": "Molecular Surface Area (Å²)",
            "asa_avg": "Molecular Surface Area per-residue avg. (Å²)",
            "RoG": "Radius of Gyration (Å)",
            "pocket_dist": "Pocket Center Distance After Chain Superposition (Å)",
            "bfac_t": "B-factor (Å²) & model pLDDT",
            "bfac_avg": "B-factor per-atom avg. (Å²) & model pLDDT per-atom avg.",
            "tm_score": "Chain TM-score",
            "tm_score_i": "Chain TM-score inverse",
            "rmsd": "Chain Cα RMSD (Å)",
            "#solvent": "Pocket Water Molecules (#)"
        }

    os.makedirs(save_dir, exist_ok=True)
    num_metrics = len(metrics)
    fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(7, num_metrics * 3))
    #fig, axes = plt.subplots(num_metrics, 1, figsize=(7, 3 * num_metrics))
    if num_metrics == 1:
        axes = [axes]

    # Precompute masks for Apo and Holo groups
    apo_mask = df['apoholo_assignment'].str.startswith('A', na=False)
    holo_mask = df['apoholo_assignment'].str.startswith('H', na=False)

    for ax, metric in zip(axes, metrics):
        if metric not in df.columns:
            continue
        x_label = metrics_names_and_units_dict.get(metric, metric)
        valid_mask = df[metric].notna()

        # Subset data
        apo_series = df.loc[apo_mask & valid_mask, metric]
        holo_series = df.loc[holo_mask & valid_mask, metric]
        all_series = df.loc[valid_mask, metric]

        # (optional) Add jitter values to avoid zero variance issues
        #apo_series = add_jitter_to_values(apo_series)
        #holo_series = add_jitter_to_values(holo_series)
        #all_series = add_jitter_to_values(all_series)

        # Compute sample sizes directly
        n_all, n_apo, n_holo = all_series.count(), apo_series.count(), holo_series.count()

        # Common grid for density estimation
        x_min, x_max = all_series.min(), all_series.max()
        np.linspace(x_min, x_max, 200)

        # Optional vertical interval gridlines
        if gridline_intervals is not None:
            for n_intervals in gridline_intervals:
                gridlines = np.linspace(x_min, x_max, n_intervals)
                for gl in gridlines:
                    ax.axvline(gl, color='purple', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)

        # Create DataFrames with a new "layer" indicator:
        # Layer "0" for the underlying Apo/Holo violins; layer "1" for the All violin and data points.
        df_all = all_series.to_frame().assign(group="All", layer="0")
        df_apo = apo_series.to_frame().assign(group="Apo", layer="1")
        df_holo = holo_series.to_frame().assign(group="Holo", layer="1")
        df_split = pd.concat([df_apo, df_holo], ignore_index=True)
        # For the data points, assign them to layer "1" so they appear on top.
        df_points = df_split.copy()
        df_points["layer"] = "0"

        # First, plot the All violin (layer "1") on top using a constant color.
        sns.violinplot(
            data=df_all,
            x=metric,
            y="layer",
            order=["0"],
            color="gray",
            alpha=0.2, # 0.2
            linewidth=1.5,
            inner=None,
            density_norm="width",
            ax=ax
        )

        # Then, plot the underlying Apo/Holo violins (layer "0")        
        #n0 = len(ax.collections)
        sns.violinplot(
            data=df_split,
            x=metric,
            y="layer",
            order=["1"],
            hue="group",
            split=False,
            dodge=False,
            linewidth=1.5,
            inner="box",
            palette={"Apo": "blue", "Holo": "red"},
            alpha=0.5, # 0.4
            density_norm="width",
            ax=ax
        )
        #n1 = len(ax.collections)
        for coll in ax.collections:#[n0:n1]:
            if isinstance(coll, mpl.collections.PolyCollection):
                fc = coll.get_facecolors()
                if fc.size > 0:
                    coll.set_edgecolor(fc[0])

        # Overlay individual data points (from Apo/Holo) on the All violin.
        sns.stripplot(
            data=df_points,
            x=metric,
            y="layer",
            order=["0"],
            hue="group",
            dodge=False,
            jitter=True,#0.1,
            alpha=0.5, # 0.4
            palette={"Apo": "blue", "Holo": "red"},
            size=4,
            ax=ax
        )
        # Remove duplicate legends from hue mappings.
        #if ax.get_legend() is not None:
        #    ax.legend_.remove()

        # Set title and axis labels.
        ax.set_title(x_label, fontsize=11, fontweight='bold')
        ax.xaxis.label.set_visible(False) #ax.set_xlabel("")
        ax.set_ylabel("Density")
        # Set y-ticks so that layer 0 is labeled "Apo /\nHolo" and layer 1 is "All"
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["All", "Apo/\nHolo"])

        # Add vertical reference lines if provided.
        if model_df is not None and not model_df.empty and metric in model_df.columns:
            ax.axvline(model_df[metric].values[0], color="green", linestyle="--") #, alpha=0.7
        #if query_df is not None and not query_df.empty and metric in query_df.columns:
        #    ax.axvline(query_df[metric].values[0], color="black", linestyle="-", alpha=0.2)
        if query_df is not None and not query_df.empty and metric in query_df.columns:
            # plot a hollow circle at y=0 (the "All" layer)
            ax.scatter(query_df[metric].values[0], 0,marker='o',s=25,facecolors='none',edgecolors='black',linewidths=1,zorder=5)


        # Also apply horizontal gridlines for consistency (optional if vertical ones are also drawn)
        # Define custom y-ticks (evenly spaced)
        if gridline_intervals is not None:
            ax.set_yticks(np.arange(-0.5, 1.6, 0.25)) # Places gridlines at: -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5
            ax.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.5, zorder=0)
            ax.set_axisbelow(True)

        # Create a unified legend that includes sample sizes and reference markers.
        unified_handles = [
            Line2D([], [], marker='s', color='gray', markersize=8, linestyle='None', alpha=0.2,
                   label=f"All (n={n_all})"),
            Line2D([], [], marker='s', color='blue', markersize=8, linestyle='None', alpha=0.4,
                   label=f"Apo (n={n_apo})"),
            Line2D([], [], marker='s', color='red', markersize=8, linestyle='None', alpha=0.4,
                   label=f"Holo (n={n_holo})"),
            Line2D([], [], marker='o', markersize=5,markerfacecolor='white', markeredgecolor='black',linestyle='None', label="Query"),
            #Line2D([], [], color='black', linestyle="-", alpha=0.2, label="Query")
            Line2D([], [], color='green', linestyle="--", alpha=0.9, label="AF model")
        ]
        ax.legend(handles=unified_handles, loc='best', framealpha=0.95, prop={'size': 9})


    if supertitle: # "Distribution of Pocket Alternative States by Metric"
        plt.suptitle(supertitle, fontsize=14, fontweight='normal', y=0.99)
    #plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Handle transparency
    if transparency:
        fig.patch.set_facecolor('none')  # Make the overall figure background transparent
        for ax in axes:
            ax.set_facecolor('white')    # Set each individual plot background to white

    plot_filename = f"metric_distribution_violins{filename_suffix}"
    for saving_format in save_formats:
        #sformat, sresolution = saving_format.split('_')
        save_path = os.path.join(save_dir, f"{plot_filename}.{saving_format}")
        plt.savefig(save_path, dpi='figure', format=saving_format)#, transparent=True)

    if show_plot:
        plt.show()
    plt.close()


def plot_overall_metrics(overall_dict):
    """
    Create bar plots to display the overall aggregated metrics: average score, impact, and overlap.
    
    Parameters:
      overall_dict (dict): Dictionary with keys 'avg_score', 'avg_impact', 'avg_overlap',
                           and 'overall_classification'.
    """
    avg_score = overall_dict.get('avg_score', None)
    avg_impact = overall_dict.get('avg_impact', None)
    avg_overlap = overall_dict.get('avg_overlap', None)
    overall_class = overall_dict.get('overall_classification', "N/A")
    
    # Check if overall metrics are available
    if avg_score is None or avg_impact is None or avg_overlap is None:
        print("No overall metrics available to plot.")
        return
    
    # Create subplots for each overall metric
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Score ranges between -1 and 1, so set the y-axis accordingly.
    axs[0].bar(['Avg Score'], [avg_score], color='purple')
    axs[0].set_ylim(-1, 1)
    axs[0].set_title("Average Score")
    
    axs[1].bar(['Avg Impact'], [avg_impact], color='orange')
    axs[1].set_title("Average Impact")
    # Impact is not strictly normalized; we let the scale be automatic.
    
    axs[2].bar(['Avg Overlap'], [avg_overlap], color='green')
    axs[2].set_ylim(0, 1)
    axs[2].set_title("Average Overlap")
    
    plt.suptitle(f"Overall Aggregated Metrics\nOverall Classification: {overall_class}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


from stats_core import (
    DPClassifyConfig,
    fit_bgm_safe,
    make_bgm_grid,
    evaluate_bgm_pdf,
    ensure_numeric_1d,   # if you need to clean raw arrays for markers
)


def plot_density_estimation_per_metric(
    ax,
    metric_name: str,
    apo_values,
    holo_values,
    *,
    cfg: DPClassifyConfig | None = None,
    color_apo="tab:blue",
    alpha_apo=0.6,
    color_holo="tab:orange",
    alpha_holo=0.6,
    alpha_points=0.6,
    grid_points: int | None = None,
    show_points: bool = True,
    precomputed=None,               # AHOJ-013: {"x","apo","holo"} from the DP classifier, or None
    #show_legend: bool = False,      # NEW
    #set_title: int = 0,             # optional polish
):
    """
    Plot DP-GMM density estimates for a single metric on a given axes.
    - Robust to empty/degenerate groups
    - Uses safe DP settings (random init, reg_covar, unique-cap)
    - Plots whichever groups are available
    - If `precomputed` densities are supplied, draws exactly those (the curve used to classify)
      instead of refitting the BGM (AHOJ-013 fidelity reuse + avoids a redundant fit).
    """
    if cfg is None:
        cfg = DPClassifyConfig()
    if grid_points is None:
        grid_points = cfg.grid_points

    if precomputed is not None and precomputed.get("x") is not None:
        # Reuse the classifier's exact densities (no refit) so the plot matches the DP classification.
        xs = np.asarray(precomputed["x"], dtype=float)
        fa = precomputed.get("apo")
        fh = precomputed.get("holo")
        fa = np.asarray(fa, dtype=float) if fa is not None else None
        fh = np.asarray(fh, dtype=float) if fh is not None else None
    else:
        # Fit models safely (returns None if insufficient data)
        bgm_apo  = fit_bgm_safe(apo_values, cfg)
        bgm_holo = fit_bgm_safe(holo_values, cfg)

        # Make a common grid based on whatever data exists
        xs = make_bgm_grid(apo_values, holo_values, grid_points=grid_points)
        if xs is None:
            ax.text(0.5, 0.5, f"{metric_name}\nNo data", ha="center", va="center", transform=ax.transAxes)
            #if set_title:
            #    ax.set_title(metric_name)
            return

        # Evaluate PDFs (None if a model wasn't fitted)
        fa = evaluate_bgm_pdf(bgm_apo, xs)
        fh = evaluate_bgm_pdf(bgm_holo, xs)

    # Draw densities if available
    if fa is not None:
        ax.plot(xs, fa, color=color_apo, alpha = alpha_apo, label="Apo", lw=2, zorder=1)
    if fh is not None:
        ax.plot(xs, fh, color=color_holo, alpha = alpha_holo, label="Holo", lw=2, zorder=1)

    # Optionally show raw points as rugs/scatters
    if show_points:
        a = ensure_numeric_1d(apo_values)
        h = ensure_numeric_1d(holo_values)
        if a.size:
            ax.scatter(a, np.full_like(a, ax.get_ylim()[0]), color=color_apo, s=10, alpha=alpha_points, marker="|", zorder=3)
        if h.size:
            ax.scatter(h, np.full_like(h, ax.get_ylim()[0]), color=color_holo, s=10, alpha=alpha_points, marker="|", zorder=3)

    #if set_title:
    #    ax.set_title(metric_name)
    #if show_legend:
    #    ax.legend(loc="best")


from kde_plot import _resolve_palette, save_plot


def plot_density_estimation_panel(
    apo_df: pd.DataFrame,
    holo_df: pd.DataFrame,
    model_point_df: pd.DataFrame | None,
    metrics: list[str],
    *,
    cfg: DPClassifyConfig | None = None,
    figsize=(7, 2.5),
    show_points: bool = True,
    # NEW unified knobs
    save_dir: str | None = None,
    save_formats: list[str] | None = None,
    filename_suffix: str = "",
    transparency: int | None = 0,
    color_scheme: str = "blue_orange1", # "curves_legacy",
    metrics_names_and_units_dict: dict[str, str] | None = None,
    supertitle: int | str = 0,
    show_plot: int = 1,
    show_legend: bool = True,
    set_title: int = 1,
    separate_metrics: int = 0,
    precomputed_curves: dict | None = None,   # AHOJ-013: {metric: {"x","apo","holo"}} from the classifier
):
    """
    Render DP-GMM density curves for each metric.

    Back-compat: if you omit the *NEW* kwargs, it behaves like before.
    If separate_metrics==0 -> one stacked figure; else one figure per metric.
    If precomputed_curves is given, each metric's curve is drawn from the classifier's exact
    density instead of refitting the BGM (AHOJ-013).
    """
    if save_formats is None:
        save_formats = []
    if cfg is None:
        cfg = DPClassifyConfig()

    # Configure colors from palette
    apo_style, holo_style, all_style, model_style, points_alpha = _resolve_palette(color_scheme)
    apo_c, apo_a = apo_style['color'], apo_style.get("alpha", 0.6)
    hol_c, hol_a = holo_style["color"], holo_style.get("alpha", 0.6)
    model_c = model_style.get("color", "green")

    # Labels
    if metrics_names_and_units_dict is None:
        metrics_names_and_units_dict = {m: m for m in metrics}

    def _plot_one(ax, metric):
        a = pd.to_numeric(apo_df[metric],  errors="coerce").dropna()  if metric in apo_df  else pd.Series([], dtype=float)
        h = pd.to_numeric(holo_df[metric], errors="coerce").dropna()  if metric in holo_df else pd.Series([], dtype=float)

        # Draw DP densities (no area fill; keep current DP style)
        plot_density_estimation_per_metric(
            ax, metric, a, h, cfg=cfg,
            color_apo=apo_c, alpha_apo=apo_a, color_holo=hol_c, alpha_holo=hol_a,
            show_points=show_points, alpha_points=points_alpha,
            precomputed=(precomputed_curves or {}).get(metric),
        )

        # Optional AF model vertical line (like KDE)
        if model_point_df is not None and metric in model_point_df.columns and len(model_point_df) == 1:
            v = pd.to_numeric(model_point_df[metric], errors="coerce").iloc[0]
            if np.isfinite(v):
                ax.axvline(v, color=model_c, linestyle="--", linewidth=2, alpha=0.85, zorder=2, label="AF model")

        # Title & legend harmonized with KDE look
        if set_title:
            ax.set_title(metrics_names_and_units_dict.get(metric, metric), fontsize=11, fontweight="bold")
        ax.set_ylabel("Density")
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc="best", framealpha=0.95, prop={"size": 9})

    if not separate_metrics:
        n = len(metrics)
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1]*n), squeeze=False, constrained_layout=True)
        for i, m in enumerate(metrics):
            _plot_one(axes[i, 0], m)

        if supertitle == 1:
            fig.suptitle("AlphaFold-DB Model Classification", fontsize=14, fontweight="normal", y=0.99)
        elif isinstance(supertitle, str) and supertitle:
            fig.suptitle(supertitle, fontsize=14, fontweight="normal", y=0.99)

        if transparency:
            fig.patch.set_facecolor("none")
            for r in axes:
                r[0].set_facecolor("white")

        save_plot(fig, save_formats, save_dir, f"dp_curve{filename_suffix}", show_plot=bool(show_plot))

        return fig, axes
    else:
        # Per-metric figures
        out = []
        for m in metrics:
            fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
            _plot_one(ax, m)
            if supertitle == 1:
                fig.suptitle("AlphaFold-DB Model Classification", fontsize=14, fontweight="normal", y=0.99)
            if transparency:
                fig.patch.set_facecolor("none"); ax.set_facecolor("white")
            save_plot(fig, save_formats, save_dir, f"dp_curve_{m}", show_plot=bool(show_plot))
            out.append((fig, ax))
        return out


# --- NEW convenience wrapper with a KDE-like signature ---
def plot_dp_curves(
    apo_group: pd.DataFrame,
    holo_group: pd.DataFrame,
    classification_metrics: list[str],
    model_point: pd.DataFrame | None,
    *,
    supertitle: int | str = 0,
    show_plot: int = 1,
    save_dir: str | None = None,
    save_formats: list[str] | None = None,
    filename_suffix: str = "",
    transparency: int | None = 0,
    separate_metrics: int = 0,
    color_scheme: str = "blue_orange1",
    set_title: int = 1,
    metrics_names_and_units_dict: dict | None = None,
    show_legend: bool = True,
    show_points: bool = True,
    cfg: DPClassifyConfig | None = None,
    precomputed_curves: dict | None = None,   # AHOJ-013: classifier densities {metric: {"x","apo","holo"}}
):
    """
    Unified DP visualizer mirroring kde_plot.plot_kde_curves parameters (where applicable).
    Keeps DP style (no fill). Uses the shared color palette.
    """
    return plot_density_estimation_panel(
        apo_group, holo_group, model_point, classification_metrics,
        cfg=cfg,
        show_points=show_points,
        save_dir=save_dir, save_formats=save_formats or [],
        filename_suffix=filename_suffix,
        transparency=transparency or 0,
        color_scheme=color_scheme,
        metrics_names_and_units_dict=metrics_names_and_units_dict,
        supertitle=supertitle,
        show_plot=show_plot,
        show_legend=show_legend,
        set_title=set_title,
        separate_metrics=separate_metrics,
        precomputed_curves=precomputed_curves,
    )
