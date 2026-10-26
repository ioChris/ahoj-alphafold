# -*- coding: utf-8 -*-
"""
figure_pocket_example.py
========================

Build a **publication-quality, composed** pocket-classification figure for a single
AHoJ-DB example entry (the manuscript "Figure 1" style), from the per-entry outputs
written by `plot_example_entries.py` (which mirrors a run config, e.g. full_run_3).

Arrangement (mirrors the previous manuscript figure `manuscript/figures/fig0/drawing_v5`):
    A  (left column)   KDE violins for the 4 pocket metrics (RMSD, SASA, Volume, RoG)
    B  (right column)  the 4 per-metric confidence-bar groups (row-aligned with A)
    C  (bottom-left)   the overall pocket confidence-bar group
    D  (bottom-right)  RESERVED placeholder for the 3 PyMOL structure snapshots --
                       only the panel letter "D" is rendered; drop the structures in
                       manually. The reserved box keeps A/B/C alignment intact.

**Fixed physical size (the key design goal).** The figure is rendered onto a fixed
CANVAS_W_MM x CANVAS_H_MM canvas (Nature Communications double-column width = 180 mm)
with fonts specified in **final points** and **no tight-bbox cropping**, so the exported
SVG is exactly 180 mm wide and drops into the manuscript layout **unscaled** (1:1), fonts
already at print size. Change CANVAS_* / FS_* to retarget another journal's guidelines.

Fidelity: the confidence bars reuse the pipeline's own primitives
(`kde_plot.draw_gradient_score_bar` / `draw_gradient_bar` / `CMAP_SEP` and the
`verbose_*` / `PROB_BAND_EDGES` label logic), so bar colors, value lines and verbal
labels are identical to the DB / webserver -- only the layout/scale differ.

Spyder-friendly: edit the SETTINGS block and press Run.
"""

import os
import sys
import json
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.lines as mlines

# --- locate parse-ahoj repo so pipeline imports resolve (this file lives in it) --------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Repo-wide path anchors (common/config.py of the ahoj-alphafold repo; set them with environment
# variables or common/paths.ini). The loop finds the repo root however deeply this file is nested.
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402
from kde_plot import draw_gradient_score_bar, draw_gradient_bar, CMAP_SEP  # noqa: E402
from kde_api import classify_point_kde_r3  # noqa: E402  (the pipeline's KDE classifier)

# ======================================================================================
# SETTINGS  (edit these)
# ======================================================================================
ENTRY = "2yiw-A-I46-1354"
# Run config whose KDE parameters (bw_scale, bw_mode, kde_extend, ...) are reused so the
# Panel A violins are the SAME KDE that drove the classification (must match the run used
# by plot_example_entries.py). Only the KDE block is read.
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "run_config_full_run_3.json")
# Root where plot_example_entries.py wrote the per-entry outputs (mirrored <batch>/<entry>).
EXAMPLE_ROOT = str(cfg.EXAMPLE_ENTRIES_DIR)       # AHOJ_EXAMPLE_ENTRIES_DIR
STATS_DIR = None                  # None -> auto-locate EXAMPLE_ROOT/*/<ENTRY>/statistics
OUTPUT_DIR = None                 # None -> alongside the entry (<entry>/figure)
OUT_BASENAME = f"Figure_pocket_{ENTRY}"
SAVE_FORMATS = ["svg", "pdf", "png"]

# --- Fixed physical canvas (Nature Communications double column = 180 mm; max height 240 mm)
CANVAS_W_MM = 180.0
CANVAS_H_MM = 234.0                # violins are width-limited by col0, so extra height only adds letterbox; keep compact
DPI = 600                          # only affects the PNG raster; SVG/PDF are vector at exact mm

# --- Typography, in FINAL points (what appears at 180 mm, unscaled) --------------------
# Nature Communications: sans-serif, min 5 pt at final size (we are at final size -> the FS_*
# values below ARE the printed pt). We use Helvetica Neue, loaded from the OTFs in fonts/
# (so it renders even where it isn't installed), matching the classification run. The fonts are
# proprietary and not distributed: put HelveticaNeue-*.otf/.ttf in fonts/ (or point AHOJ_FONT_DIR
# at a folder holding them), or install them system-wide. Without them the next FONT_FAMILY entry
# that is installed is used.
FONT_DIR = os.environ.get("AHOJ_FONT_DIR", os.path.join(REPO_ROOT, "fonts"))
FONT_FAMILY = ["Helvetica Neue", "TeX Gyre Heros", "Nimbus Sans", "Arial", "DejaVu Sans"]
FS_BASE = 6.3       # confidence-bar descriptor labels
FS_TITLE = 7.6      # panel-metric titles (bold)
FS_TICK = 6.8       # violin x-axis tick numbers (enlarged: y-axis clutter removed)
FS_LEGEND = 6.8     # top-violin legend (enlarged; box tightened -> see _add_full_legend)
FS_NTEXT = 6.2      # compact "All N . Apo N . Holo N" count text (panels 2-4)
FS_BRACKET = 6.5    # Classification / General bracket labels (was 5.6; clearly above the 5 pt floor)
FS_CORNER = 6.5     # APO/HOLO poles + P_APO/P_HOLO on the score bar (note: the P_x subscripts render ~0.7x)
FS_LETTER = 11.0    # panel letters A / B / C / D (bold; clearly larger than the 7.6 pt titles)

# --- Colors (match the DB violin scheme) ----------------------------------------------
APO_COLOR = "#2020DF"
HOLO_COLOR = "#DF2020"
ALL_COLOR = "0.62"
MODEL_COLOR = "green"       # AF-DB model dashed line (matches plot_kde_violins' hardcoded green)
QUERY_EDGE = "black"
IQR_COLOR = "0.15"          # boxplot IQR bar/whiskers
NTEXT_COLOR = "0.33"        # single gray for the compact count text (Nat Comms: color coding is set by the top legend)
LEG_ALPHA_ALL = 0.20        # legend swatch alphas MATCH the violin fills (seaborn1: All grey 0.2, ...
LEG_ALPHA_APOHOLO = 0.50    # ... Apo/Holo 0.5) so the legend keys aren't over-saturated vs the plot

# --- Violin y-axis: drop the "Density" label + the All / Apo/Holo row ticks. The two rows
# (top = pooled "All" with raw points; bottom = separated Apo/Holo densities) are established
# by the top-panel legend's color coding, and dropping them enlarges the violins + frees the
# left margin for larger fonts. Flip these to True to restore the DB/webserver y-axis.
SHOW_VIOLIN_YLABEL = False
SHOW_VIOLIN_YTICKS = False

# --- Layout: STRICT 5x2 grid (figure fractions; tuned so nothing clips in the fixed canvas)
# Rows 0-3: col0 = violin[metric], col1 = bar-group[metric] (side-by-side, row-aligned per
# metric). Row 4 (slightly taller): col0 = Panel C overall bars, col1 = Panel D (letter only).
# Margins balanced so left (panel letters) and right (Panel B brackets+labels) whitespace match
# and the grid is centered; wider span than before -> larger cells. Nat Comms sets only outer
# limits (<=180x240 mm, >=5 pt), no internal-margin rule, so internal balance is aesthetic.
MARGIN = dict(left=0.053, right=0.938, top=0.968, bottom=0.035)
GRID_WIDTH_RATIOS = [1.08, 1.10]    # near-equal; col1 a hair wider for the long bar descriptors
GRID_HEIGHT_RATIOS = [1, 1, 1, 1, 1.25]  # row 4 (C|D) taller -> C reads as the enlarged summary
GRID_HSPACE = 0.40                  # tightened -> taller rows -> larger (height-limited) violins; still clears xlabels
GRID_WSPACE = 0.22                  # between col0 and col1: room for Panel C's bracket+label
BARGROUP_HSPACE = 1.15              # between the 5 bars inside a group (room for descriptor xlabels)
BARGROUP_HRATIOS = [1.45, 1, 1, 1, 1]  # score bar taller (APO/HOLO above + P labels below)
BAR_LABELPAD = 2.0                  # pull gradient-bar descriptor labels close under their bar
VIOLIN_BOX_ASPECT = 0.44            # violin plot-box height/width -> FIXED; 0.44 fills the cell width (webserver-like)
C_TOP_PAD = 0.18                    # empty pad above Panel C inside its row-4 cell -> lowers C + its title clear of RoG
# Legend / count-text anchor within the top / non-top violin axes (exposed for per-entry tuning:
# some entries' data may sit where these overlap -> move to a clearer corner without code edits).
LEGEND_LOC = "center right"         # full legend, top violin (band between the two violin rows)
NTEXT_LOC = "upper right"           # compact All/Apo/Holo count text, panels 2-4
BRACKET_DX = 0.016                  # gap from bar right edge to the bracket stem (was touching the bars)
LETTER_DX = 0.034                   # panel letter offset left of its cell's top-left corner
LETTER_DY = 0.004                   # panel letter offset above its cell top
DRAW_BRACKETS = True
DRAW_D_GUIDE = True                 # faint dashed box + hint text in the reserved D area
# Publication-only shortening of the (crowded) score-bar label to free horizontal space.
# The DB/webserver label is untouched. Set to [] to keep the full "Classification:" text.
SCORE_LABEL_REPLACEMENTS = [("Classification:", "Class:")]

# Pocket metrics shown, in order, with display names + units.
METRICS = ["pocket_rms", "sasa", "volume", "RoG"]
METRIC_META = {
    "pocket_rms": {"title": "Pocket all-atom RMSD (Å)"},
    "sasa":       {"title": "Solvent Accessible Surface Area (Å²)"},
    "volume":     {"title": "Volume (Å³)"},
    "RoG":        {"title": "Radius of Gyration (Å)"},
}
from run_types import APO_LABELS, HOLO_LABELS  # noqa: E402  (the pipeline's pooled label sets)


# ======================================================================================
# Data loading
# ======================================================================================
def resolve_stats_dir(entry=ENTRY, example_root=EXAMPLE_ROOT, stats_dir=STATS_DIR):
    if stats_dir:
        return stats_dir
    hits = glob.glob(os.path.join(example_root, "*", entry, "statistics"))
    if not hits:
        raise FileNotFoundError(
            f"No statistics dir for {entry} under {example_root}. "
            f"Run plot_example_entries.py for this entry first.")
    return hits[0]


def load_entry(stats_dir):
    with open(os.path.join(stats_dir, "classification_kde_pocket.json"), "r", encoding="utf-8") as fh:
        results = json.load(fh)
    df = pd.read_csv(os.path.join(stats_dir, "global_results_plots.csv"))
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return results, df


def normalize_query_df(query_df, zero_metrics, one_metrics):
    """Query reference point: fill the given metric columns' NaNs with 0 / 1
    (matches batch pipeline; e.g. query-vs-self pocket_rms -> 0). Missing columns are skipped."""
    q = query_df.copy()
    cols = [c for c in (zero_metrics + one_metrics) if c in q.columns]
    if cols:
        q.loc[:, cols] = q[cols].replace('', np.nan).astype(float)
    for m in zero_metrics:
        if m in q.columns:
            q.loc[:, m] = q[m].fillna(0.0)
    for m in one_metrics:
        if m in q.columns:
            q.loc[:, m] = q[m].fillna(1.0)
    return q


def kde_kwargs_from_config(config_path=CONFIG_PATH):
    """Read only the KDE parameters from a run config so the violins use the SAME KDE
    as that run's classification (falls back to the full_run_3 defaults)."""
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        cfg = {}
    return dict(
        kde_extend=bool(cfg.get("kde_extend", True)),
        margin_cap_scale=float(cfg.get("margin_cap_scale", 1.0)),
        bw_mode=cfg.get("bw_mode", "pooled"),
        bw_scale=float(cfg.get("bw_scale", 0.75)),
        prior_mode=cfg.get("prior_mode", "equal"),
    )


def compute_pocket_kde(df, kde_kwargs):
    """Reproduce the pipeline's per-metric KDE (kde_data) + apo/holo/all point arrays for the
    pocket metrics, from the filtered df (global_results_plots.csv). Mirrors
    do_kde_plots_and_classification's setup so the violins are the exact classification KDE."""
    df_q = df[df["role"] == "Q"]
    df_af = df[df["role"] == "M"]
    cand = df[df["role"] == "C"]
    df_c = pd.concat([df_q, cand], ignore_index=True)                    # add_query_row=1
    apo_df = df_c[df_c["apoholo_assignment"].isin(APO_LABELS)]
    holo_df = df_c[df_c["apoholo_assignment"].isin(HOLO_LABELS)]
    df_q_norm = normalize_query_df(df_q, ["pocket_rms", "pocket_dist", "rmsd"], ["tm_score", "tm_score_i"])
    _, kde_data, apo_v, holo_v, all_v = classify_point_kde_r3(
        df_c, apo_df, holo_df, df_af, METRICS, **kde_kwargs)
    return dict(kde_data=kde_data, apo=apo_v, holo=holo_v, allv=all_v, model_df=df_af, query_df=df_q_norm)


# ======================================================================================
# Panel A -- draw the ACTUAL pipeline violins (kde_plot.plot_kde_violins) into our own axes
# via its axes_map hook, then post-process fonts / marker sizes for the small fixed-size
# panels. This guarantees the violins are byte-identical to the DB/webserver kde_violin_*.svg
# (x-margins, boxplots, median dots, spacing, colors) -- only fonts/markers are rescaled.
# ======================================================================================
VIOLIN_POINT_SCALE = 0.34    # shrink scatter/median/query markers for the small panels
VIOLIN_LINE_SCALE = 0.55     # shrink violin edge + boxplot line widths


def draw_violin_panel(ax, metric, kde, is_top=False):
    from kde_plot import plot_kde_violins
    np.random.seed(0)        # deterministic point jitter (plot_kde_violins uses np.random)
    # We build the legend ourselves (below) for full control over its size/box, so the
    # pipeline is asked NOT to draw one (show_legend=False) in the axes_map path.
    plot_kde_violins(
        kde["kde_data"], [metric], None,
        kde["apo"], kde["holo"], kde["allv"],
        model_df=kde["model_df"], query_df=kde["query_df"],
        save_formats=[], show_plot=0, separate_metrics=1, set_title=1,
        metrics_names_and_units_dict={metric: METRIC_META[metric]["title"]},
        color_scheme="seaborn1", violin_width=0.80, boxplots=1,
        show_legend=False, axes_map={metric: ax},
    )
    _restyle_violin_ax(ax)
    n_all = int(np.size(kde["allv"][metric]))
    n_apo = int(np.size(kde["apo"][metric]))
    n_holo = int(np.size(kde["holo"][metric]))
    if is_top:
        # Full legend on the TOP violin only -- this is where the color coding is established.
        has_query = _metric_present(kde["query_df"], metric)
        has_model = _metric_present(kde["model_df"], metric)
        _add_full_legend(ax, n_all, n_apo, n_holo, has_query=has_query, has_model=has_model)
    else:
        # Panels 2-4: just the compact one-row count text (gray; colors already established).
        _add_compact_n_text(ax, n_all, n_apo, n_holo)


def _metric_present(df, metric):
    """True if df has a finite value for metric (mirror of plot_kde_violins' query/model test)."""
    try:
        return (df is not None and metric in df.columns
                and df[metric].notna().any() and np.isfinite(df[metric].values[0]))
    except Exception:
        return False


def _add_full_legend(ax, n_all, n_apo, n_holo, has_query=True, has_model=True):
    """Compact-but-legible legend for the top violin: larger font, tight box, placed at the
    center-right (in the empty band between the two violin rows) so it clears the data points.
    Swatch alphas MATCH the violins (All grey 0.2 / Apo blue 0.5 / Holo red 0.5) so they don't
    read as over-saturated. Establishes the color coding + the Query and AF-model markers."""
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_rgba
    ms = FS_LEGEND * 0.95                       # marker size (pt) scaled to the (enlarged) font
    handles = [
        Line2D([], [], marker="s", markersize=ms, markerfacecolor=to_rgba(ALL_COLOR, LEG_ALPHA_ALL),
               markeredgecolor="none", linestyle="None", label=f"All (n={n_all})"),
        Line2D([], [], marker="s", markersize=ms, markerfacecolor=to_rgba(APO_COLOR, LEG_ALPHA_APOHOLO),
               markeredgecolor="none", linestyle="None", label=f"Apo (n={n_apo})"),
        Line2D([], [], marker="s", markersize=ms, markerfacecolor=to_rgba(HOLO_COLOR, LEG_ALPHA_APOHOLO),
               markeredgecolor="none", linestyle="None", label=f"Holo (n={n_holo})"),
    ]
    if has_query:
        handles.append(Line2D([], [], marker="o", markersize=ms * 0.82, markerfacecolor="white",
                              markeredgecolor=QUERY_EDGE, markeredgewidth=0.7, linestyle="None", label="Query"))
    if has_model:
        handles.append(Line2D([], [], color=MODEL_COLOR, linestyle="--", linewidth=1.1, label="AF model"))
    leg = ax.legend(handles=handles, loc=LEGEND_LOC, fontsize=FS_LEGEND, frameon=True,
                    framealpha=0.9, borderpad=0.3, labelspacing=0.22, handlelength=1.1,
                    handletextpad=0.4, borderaxespad=0.35)
    leg.set_zorder(999)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor("0.7")


def _add_compact_n_text(ax, n_all, n_apo, n_holo):
    """Compact one-row 'All N . Apo N . Holo N' count text for the non-first violin panels.
    Single gray (Nat Comms style; the color coding is established by the top-panel legend).
    Anchored top-right, no frame."""
    from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox

    def _t(txt, bold=True):
        return TextArea(txt, textprops=dict(color=NTEXT_COLOR, fontsize=FS_NTEXT,
                                            fontweight=("bold" if bold else "normal")))

    row = HPacker(children=[_t(f"All {n_all}"), _t("  ·  ", bold=False),
                            _t(f"Apo {n_apo}"), _t("  ·  ", bold=False),
                            _t(f"Holo {n_holo}")],
                  align="center", pad=0, sep=0)
    box = AnchoredOffsetbox(loc=NTEXT_LOC, child=row, pad=0.15, borderpad=0.3, frameon=False)
    box.set_zorder(999)
    ax.add_artist(box)


def _restyle_violin_ax(ax):
    from matplotlib.collections import PathCollection
    ax.title.set_fontsize(FS_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK, length=2.2, pad=1.5)
    # Drop the "Density" y-label + the All / Apo/Holo row ticks (color coding comes from the
    # top legend; frees space + enlarges the violins). Keep the axes box (all 4 spines).
    if not SHOW_VIOLIN_YLABEL:
        ax.set_ylabel("")
    else:
        ax.yaxis.label.set_size(FS_TICK)
    if not SHOW_VIOLIN_YTICKS:
        ax.set_yticks([])
    else:
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(FS_TICK)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
    # shrink scatter markers (points / median dots / query) for the small panels
    for c in ax.collections:
        if isinstance(c, PathCollection):
            sizes = np.asarray(c.get_sizes(), float)
            if sizes.size:
                c.set_sizes(sizes * VIOLIN_POINT_SCALE)
    # thin down boxplot/whisker/model lines and violin edges
    for ln in ax.lines:
        ln.set_linewidth(ln.get_linewidth() * VIOLIN_LINE_SCALE)
    for coll in ax.collections:
        try:
            coll.set_linewidth(np.asarray(coll.get_linewidth(), float) * VIOLIN_LINE_SCALE)
        except Exception:
            pass
    # Keep the violin's aspect ratio fixed so it letterboxes within a resized grid cell
    # (the confidence bars, by contrast, stretch freely to fill their cells).
    ax.set_box_aspect(VIOLIN_BOX_ASPECT)


# ======================================================================================
# Panels B / C -- one 5-bar confidence-bar group (reusing kde_plot primitives)
# ======================================================================================
def _metric_bar_values(data, is_overall):
    if is_overall:
        return dict(classification=data.get("overall_classification", "n/a"),
                    p_apo=data.get("avg_P(apo)", np.nan), p_holo=data.get("avg_P(holo)", np.nan),
                    score=data.get("overall_score", 0.0),
                    nds=data.get("avg_normalized_density_strongest", 0.0),
                    ndc=data.get("avg_normalized_density_combined", 0.0),
                    overlap=data.get("overall_overlap", np.nan), mode_overlap=data.get("overall_mode_overlap", np.nan))
    return dict(classification=data.get("classification", "n/a"),
                p_apo=data.get("P(apo)", np.nan), p_holo=data.get("P(holo)", np.nan),
                score=data.get("score", 0.0),
                nds=data.get("normalized_density_strongest", 0.0),
                ndc=data.get("normalized_density_combined", 0.0),
                overlap=data.get("overlap", np.nan), mode_overlap=data.get("mode_overlap", np.nan))


def draw_bar_group(fig, subspec, data, is_overall=False):
    """Draw the 5 confidence bars into a 5-row sub-gridspec; returns the 5 axes."""
    v = _metric_bar_values(data, is_overall)
    sep = 1 - v["overlap"] if not np.isnan(v["overlap"]) else np.nan
    sep_mode = 1 - v["mode_overlap"] if not np.isnan(v["mode_overlap"]) else np.nan

    inner = GridSpecFromSubplotSpec(5, 1, subplot_spec=subspec, hspace=BARGROUP_HSPACE,
                                    height_ratios=BARGROUP_HRATIOS)
    axes = [fig.add_subplot(inner[i]) for i in range(5)]
    draw_gradient_score_bar(axes[0], value=v["score"], p_apo=v["p_apo"], p_holo=v["p_holo"],
                            classification=v["classification"], min_val=-1, max_val=1, cmap=plt.get_cmap("bwr"))
    draw_gradient_bar(axes[1], value=v["nds"], min_val=0, max_val=1, cmap=CMAP_SEP,
                      label="Density at Classification Point (top group)", stop_at_value=True)
    draw_gradient_bar(axes[2], value=v["ndc"], min_val=0, max_val=1, cmap=CMAP_SEP,
                      label="Density at Classification Point (combined)", stop_at_value=True)
    draw_gradient_bar(axes[3], value=sep, min_val=0, max_val=1, cmap=CMAP_SEP,
                      label="Total Density Separation", stop_at_value=True)
    draw_gradient_bar(axes[4], value=sep_mode, min_val=0, max_val=1, cmap=CMAP_SEP,
                      label="Peak Density Separation", stop_at_value=True)

    for ax in axes:
        ax.xaxis.label.set_size(FS_BASE)
    for ax in axes[1:]:            # gradient bars: hug the descriptor label just under its bar
        ax.xaxis.labelpad = BAR_LABELPAD
    for t in axes[0].texts:        # APO/HOLO poles + P_APO/P_HOLO -> final size
        t.set_fontsize(FS_CORNER)
    # the score bar is the most crowded (poles + centered verbal summary) -> optionally
    # shorten its "Classification:" prefix to free horizontal space (publication only;
    # the DB/webserver label is untouched).
    for old, new in SCORE_LABEL_REPLACEMENTS:
        axes[0].set_xlabel(axes[0].get_xlabel().replace(old, new))
    return axes


def _bracket(fig, axes, idx0, idx1, text, cap=0.007, dx=BRACKET_DX, label_dx=0.011):
    """Vertical bracket spanning axes[idx0..idx1], drawn just to the RIGHT of the bars with
    its label centered on the bracket's vertical midpoint (rotation_mode='anchor')."""
    top = max(axes[i].get_position().y1 for i in range(idx0, idx1 + 1))
    bot = min(axes[i].get_position().y0 for i in range(idx0, idx1 + 1))
    x = max(axes[i].get_position().x1 for i in range(idx0, idx1 + 1)) + dx
    for seg in ([[x, x], [bot, top]], [[x, x - cap], [bot, bot]], [[x, x - cap], [top, top]]):
        fig.add_artist(mlines.Line2D(seg[0], seg[1], transform=fig.transFigure, color="black", lw=1.0))
    fig.text(x + label_dx, 0.5 * (bot + top), text, transform=fig.transFigure, rotation=270,
             rotation_mode="anchor", va="center", ha="center",
             fontsize=FS_BRACKET, fontweight="bold")


def _titles_and_brackets(fig, axes, title=None):
    if title:
        sb = axes[0].get_position()
        fig.text(0.5 * (sb.x0 + sb.x1), sb.y1 + 0.15 * sb.height + 0.014, title,
                 ha="center", va="bottom", fontsize=FS_TITLE, fontweight="bold")
    if DRAW_BRACKETS:
        _bracket(fig, axes, 0, 2, "Classification")
        _bracket(fig, axes, 3, 4, "General")


# ======================================================================================
# Compose  (fixed 180 x 230 mm canvas, final-pt fonts, no tight-bbox)
# ======================================================================================
def _register_bundled_fonts(font_dir=FONT_DIR):
    """Register the bundled Helvetica Neue OTFs (fonts/HelveticaNeue-*) so the figure renders
    in Helvetica Neue even where it isn't installed. Bold resolves to HelveticaNeue-Bold.otf."""
    import matplotlib.font_manager as fm
    registered = []
    for fpath in sorted(glob.glob(os.path.join(font_dir, "HelveticaNeue*.otf")) +
                        glob.glob(os.path.join(font_dir, "HelveticaNeue*.ttf"))):
        try:
            fm.fontManager.addfont(fpath)
            registered.append(os.path.basename(fpath))
        except Exception:
            pass
    return registered


def build_figure(stats_dir=None, output_dir=None, out_basename=OUT_BASENAME):
    stats_dir = resolve_stats_dir(stats_dir=stats_dir or STATS_DIR)
    if output_dir is None:
        output_dir = OUTPUT_DIR or os.path.join(os.path.dirname(stats_dir), "figure")
    print(f"[stats] {stats_dir}")

    _register_bundled_fonts()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT_FAMILY,
        "font.size": FS_BASE, "axes.labelsize": FS_BASE,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    from matplotlib.font_manager import FontProperties, findfont
    print(f"[font] {os.path.basename(findfont(FontProperties(family=FONT_FAMILY)))}")
    results, df = load_entry(stats_dir)
    kde = compute_pocket_kde(df, kde_kwargs_from_config(CONFIG_PATH))

    MM = 1.0 / 25.4
    fig = plt.figure(figsize=(CANVAS_W_MM * MM, CANVAS_H_MM * MM))

    # STRICT 5x2 grid: rows 0-3 = [violin | bar-group] per metric (row-aligned); row 4 (taller)
    # = [Panel C overall bars | Panel D reserved]. One grid -> A and B align exactly per metric.
    gs = GridSpec(5, 2, height_ratios=GRID_HEIGHT_RATIOS, width_ratios=GRID_WIDTH_RATIOS,
                  hspace=GRID_HSPACE, wspace=GRID_WSPACE, **MARGIN)

    groups = []
    for i, metric in enumerate(METRICS):
        axA = fig.add_subplot(gs[i, 0])
        draw_violin_panel(axA, metric, kde, is_top=(i == 0))
        axesB = draw_bar_group(fig, gs[i, 1], results[metric], is_overall=False)
        groups.append((axesB, None))   # metric title is on the row's violin panel (A) already

    # row 4: C (left, enlarged overall bar group) | D (right, reserved placeholder).
    # Inset C into the LOWER part of its cell (empty C_TOP_PAD above) so the group and its
    # "Overall..." title drop clear of the RoG violin above + the panel letter.
    c_cell = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[4, 0], height_ratios=[C_TOP_PAD, 1.0], hspace=0.0)
    axesC = draw_bar_group(fig, c_cell[1], results["overall"], is_overall=True)
    groups.append((axesC, "Overall pocket classification"))

    axD = fig.add_subplot(gs[4, 1]); axD.set_xticks([]); axD.set_yticks([])
    if DRAW_D_GUIDE:
        for s in axD.spines.values():
            s.set_linestyle((0, (4, 4))); s.set_color("0.75"); s.set_linewidth(0.7)
        axD.text(0.5, 0.5, "Panel D — structure snapshots\n(placed manually)",
                 transform=axD.transAxes, ha="center", va="center", fontsize=FS_LEGEND, color="0.6")
    else:
        axD.set_axis_off()

    # titles / brackets / panel letters need the final (drawn) positions
    fig.canvas.draw()
    for axes, title in groups:
        _titles_and_brackets(fig, axes, title)

    # panel letters at each quadrant's top-left (cell positions are stable regardless of the
    # violin letterboxing, since they come from the gridspec, not the resized axes).
    for cell, letter in [(gs[0, 0], "A"), (gs[0, 1], "B"), (gs[4, 0], "C"), (gs[4, 1], "D")]:
        pos = cell.get_position(fig)
        fig.text(max(0.006, pos.x0 - LETTER_DX), pos.y1 + LETTER_DY, letter,
                 fontsize=FS_LETTER, fontweight="bold", va="bottom", ha="left")

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for fmt in SAVE_FORMATS:
        p = os.path.join(output_dir, f"{out_basename}.{fmt}")
        fig.savefig(p, dpi=DPI, facecolor="white")   # NO bbox_inches -> exact CANVAS size
        paths.append(p)
    plt.close(fig)
    print(f"Saved (exact {CANVAS_W_MM:.0f} x {CANVAS_H_MM:.0f} mm, drop in unscaled):")
    for p in paths:
        print(" -", p)
    return paths


if __name__ == "__main__":
    build_figure()
