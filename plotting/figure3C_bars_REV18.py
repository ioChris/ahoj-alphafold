# -*- coding: utf-8 -*-
"""
Figure 3C (REV14) — 3 metric panels with threshold trends + tick/label cleanup

Changes vs REV13:
  - Show "Apo/Holo" y tick labels ONLY on the first (leftmost) panel.
  - Show threshold labels ONLY once per bar row: on the rightmost panel.
  - Show x label ("Within-class (%)") ONLY once for the whole figure (centered at bottom).
  - Keeps horizontal 100% stacked bars with KDE-derived bins.

Spyder-friendly globals. No CLI.
"""

import argparse
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D

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
from common import figspec


INPUT_TABLE  = str(cfg.ANALYSIS_UNIFIED_LEVELS_CSV)
OUTPUT_DIR   = str(cfg.PLOTS_DIR / "3C_bars")
POSTER_OUTPUT_DIR = str(cfg.PRESENTATION_DIR)
OUT_BASENAME = "Figure3C_metric_panels_REV17"

THRESHOLDS_TO_PLOT = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050"]  # order matters (top->bottom within group)

MODE = "pocket"                 # "pocket" or "chain"
NORMALIZE_WITHIN_CLASS = True   # keep True to make each bar sum to 100%

# Segment labels inside bars
SHOW_PERCENT_LABELS = True
# Hide in-bar labels for segments narrower than this. 24 dropped every segment under a quarter
# of the bar -- at ST_100 that silenced five of the nine numbers, including all three of the
# density row's "low" segments. The smallest segment drawn is 9%, which at 180 mm is 4.7 mm
# against a 3.2 mm label, so nothing overflows at 0.
MIN_LABEL_PCT = 0

SHOW_CLASS_LEGEND = True    # redundant once the lanes are named on the y axis
SHOW_ALL_GROUP = False          # add an "All" lane (every classified pocket) between apo and holo
ALL_COLOR = figspec.ALL         # the neutral grey shared with the detection figure

IMPACT_LEGEND_STYLE = "grayscale"   # "grayscale" or "by_class"

# Geometry (compact; intended as subpanel in a larger double-width figure)
FIG_W = 3.46                    # Single column width for NAR
ROW_H = 0.20                    
BAR_HEIGHT = 0.70               
GROUP_GAP = 0.90                
SINGLE_ROW_GAP_SCALE = 0.30     # gap shrink factor when each group holds a single threshold
FIG_SIZE_OVERRIDE = None        # (width, height) in inches; set by --fig_size

# X ticks / grid
XTICKS_MODE = "ends"            
SHOW_X_GRID = False             

# Subplot spacing (Strict compartmentalization)
SUBPLOT_LEFT = 0.30             # Increased heavily to fit the "UNCLUSTERED" labels
SUBPLOT_RIGHT = 0.96            # Push right edge closer to boundary
SUBPLOT_TOP = 0.90
SUBPLOT_BOTTOM = 0.24           # Raised to give the shared X-label and legend dedicated room
WSPACE = 0.15                   # Tighter gap between the 3 columns

SHOW_PANEL_LETTER = False
PANEL_LETTER = "b"      # lowercase per the Nature figure guide; --panel_letter overrides
PANEL_LETTER_FS = 12
TRANSPARENT_BACKGROUND = False

# Threshold labels
RIGHT_LABEL_X = 101.0           
THRESH_LABEL_X = -15.0          # Pushed further left so they don't touch the first axis
THRESH_LABEL_COLOR = "0.45"
THRESH_SIDE = "left"

# Colors
USE_NEUTRAL_COLORS = False
#APO_COLOR  = "#1f77b4"
#HOLO_COLOR = "#ff7f0e"
NEUTRAL_A  = "0.35"
NEUTRAL_H  = "0.55"

# Legends (figure-level)
SHOW_LEGENDS = True

LEGEND_LABEL_APO  = "AF-Apo"
LEGEND_LABEL_HOLO = "AF-Holo"

# Legend placement in figure coordinates (0.0 to 1.0)
# Align left legend with the START of the plots (SUBPLOT_LEFT)
LEGEND_GROUP_X = 0.10               # Shifted far left into the blank space below threshold names     
# Align right legend with the END of the plots (SUBPLOT_RIGHT)
LEGEND_BINS_X = SUBPLOT_RIGHT       

SUPXLABEL_Y = 0.14                  # Lowered slightly
LEGEND_Y0 = 0.02                    # Legends sit flush at the very bottom

# ----------------------------
# Bins (from KDE verbal logic)
# ----------------------------
DENSSEP_LOW_MAX = 0.30
DENSSEP_MOD_MAX = 0.60
PROB_LOW_MAX = 0.5933
PROB_MOD_MAX = 0.6996
BIN_LABELS = ["Low", "Moderate", "High"]

# ----------------------------
# Shared style block (Figure 3 house style)
# ----------------------------
TITLE_FS = 9.0
LAB_FS   = 8.0
TICK_FS  = 7.0
SMALL_FS = 6.2

AX_LW    = 0.8
TICK_W   = 0.8
TICK_LEN = 3.5

APO_COLOR  = "#1f77b4"
HOLO_COLOR = "#ff7f0e"

FONT_PREFERENCE = ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"]

import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_PREFERENCE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "axes.linewidth": AX_LW,
    "xtick.major.width": TICK_W,
    "ytick.major.width": TICK_W,
    "xtick.major.size": TICK_LEN,
    "ytick.major.size": TICK_LEN,

    "axes.titlepad": 4,
})

# ----------------------------
# Helpers
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Plot Figure 3C metric panels.")
    p.add_argument("--input_table", default=INPUT_TABLE)
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--out_basename", default=OUT_BASENAME)
    p.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated thresholds to plot, e.g. ST_100. Defaults to THRESHOLDS_TO_PLOT.",
    )
    p.add_argument("--poster_output_dir", action="store_true", help="Write to the configured poster plots directory.")
    p.add_argument("--fig_size", default=None, metavar="W,H",
                   help="canvas in inches. The default height is locked to Figure 3A for the "
                        "4-threshold composition; a single-threshold render has far fewer lanes "
                        "and wants a shorter canvas.")
    p.add_argument("--show_all_group", action="store_true",
                   help="add an 'All' lane (every classified pocket) between the apo and holo "
                        "lanes, matching the AF-Apo / All / AF-Holo order of the detection figure.")
    p.add_argument("--show_panel_letter", action="store_true")
    p.add_argument("--hide_class_legend", action="store_true",
                   help="drop the AF-Apo/All/AF-Holo key. Redundant when --show_all_group puts "
                        "those names on the y axis, and it costs a whole legend row.")
    p.add_argument("--panel_letter", default=None, metavar="L",
                   help="the panel letter to draw (implies --show_panel_letter). Set it when this "
                        "panel is one of several being composed and is not the first.")
    p.add_argument("--hide_panel_letter", "--hide_panel_letters", dest="hide_panel_letter", action="store_true")
    p.add_argument("--transparent_background", action="store_true")
    return p.parse_args()

def configure_from_args(args):
    global INPUT_TABLE, OUTPUT_DIR, OUT_BASENAME, THRESHOLDS_TO_PLOT
    global SHOW_PANEL_LETTER, TRANSPARENT_BACKGROUND, SHOW_ALL_GROUP, FIG_SIZE_OVERRIDE
    global SHOW_CLASS_LEGEND
    global PANEL_LETTER
    INPUT_TABLE = args.input_table
    OUTPUT_DIR = POSTER_OUTPUT_DIR if args.poster_output_dir else args.output_dir
    OUT_BASENAME = args.out_basename
    if args.thresholds:
        THRESHOLDS_TO_PLOT = [t.strip() for t in args.thresholds.split(",") if t.strip()]
    SHOW_ALL_GROUP = args.show_all_group
    SHOW_CLASS_LEGEND = not args.hide_class_legend
    if args.panel_letter:
        PANEL_LETTER = args.panel_letter
        SHOW_PANEL_LETTER = True
    if args.fig_size:
        try:
            w, h = (float(v) for v in args.fig_size.split(","))
        except ValueError:
            raise SystemExit("--fig_size takes W,H in inches")
        FIG_SIZE_OVERRIDE = (w, h)
    if args.show_panel_letter:
        SHOW_PANEL_LETTER = True
    if args.hide_panel_letter:
        SHOW_PANEL_LETTER = False
    TRANSPARENT_BACKGROUND = args.transparent_background
    # Poster SVGs outline their glyphs so they render identically anywhere; the manuscript build
    # keeps live text ("none"), which needs the viewer to actually have the font stack installed.
    plt.rcParams["svg.fonttype"] = "path" if TRANSPARENT_BACKGROUND else "none"

def output_paths():
    return (
        os.path.join(OUTPUT_DIR, OUT_BASENAME + ".pdf"),
        os.path.join(OUTPUT_DIR, OUT_BASENAME + ".png"),
        os.path.join(OUTPUT_DIR, OUT_BASENAME + ".svg"),
    )

def apply_background(fig):
    if TRANSPARENT_BACKGROUND:
        fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        for ax in fig.axes:
            ax.set_facecolor("white")
    else:
        fig.patch.set_facecolor("white")

def save_outputs(fig):
    output_pdf, output_png, output_svg = output_paths()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    apply_background(fig)
    save_specs = [
        (output_pdf, {}),
        (output_png, {"dpi": 600}),
        (output_svg, {"format": "svg"}),
    ]
    # --fig_size means this panel is a PART: it is drawn at the millimetre size it will occupy
    # and placed, never resized. A tight crop trims by however much outboard label it happens to
    # carry -- here 1.7 mm of width -- so the composer would leave a white sliver beside it.
    save_kwargs = {
        "bbox_inches": (None if FIG_SIZE_OVERRIDE else "tight"),
        "pad_inches": 0.02,
        "facecolor": fig.get_facecolor(),
    }
    for path, extra in save_specs:
        try:
            fig.savefig(path, **save_kwargs, **extra)
        except PermissionError as exc:
            print(f"[warn] could not save {path}: {exc}")
    return output_pdf, output_png, output_svg

def load_unified(path):
    path = str(path)
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = "analysis_unified_levels.csv"
            if name not in z.namelist():
                csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    raise SystemExit("Zip contains no CSV.")
                name = csvs[0]
            with z.open(name) as f:
                return pd.read_csv(f, low_memory=False)
    return pd.read_csv(path, low_memory=False)

def require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        print("Available columns:", list(df.columns))
        raise SystemExit("Stopping due to missing columns.")

def subset_threshold(df, thr):
    lvl = "entry" if str(thr) == "UNCLUSTERED" else "ubs"
    return df[(df["threshold"].astype(str) == str(thr)) &
              (df["level"].astype(str).str.lower() == lvl)].copy()

def norm_class(s):
    return s.astype(str).str.strip().str.lower()

def bin3(v, low_max, mod_max):
    vv = pd.to_numeric(v, errors="coerce")
    vv = np.asarray(vv, dtype=float)
    out = np.full(vv.shape[0], np.nan, dtype=float)
    m = np.isfinite(vv)
    out[m & (vv <= low_max)] = 0
    out[m & (vv > low_max) & (vv <= mod_max)] = 1
    out[m & (vv > mod_max)] = 2
    return out

def percent(counts, denom):
    if not NORMALIZE_WITHIN_CLASS:
        return counts.astype(float)
    if denom <= 0:
        return np.zeros_like(counts, dtype=float)
    return 100.0 * counts.astype(float) / float(denom)

def shade(color, factor_to_white):
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * float(factor_to_white))

def impact_colors(base_color):
    return (shade(base_color, 0.70), shade(base_color, 0.35), mcolors.to_rgb(base_color))

def format_row_label(thr):
    if thr == "UNCLUSTERED":
        return  "Unclustered"# "Unclustered (entries)" | "PDB pockets"
    if thr.startswith("ST_") and thr[3:].isdigit():
        if thr[3:].startswith("0"):
            return f"ST {thr[4:]}"   # f"UBS - ST {thr[4:]}%"
        return f"ST {thr[3:]}"       # f"UBS - ST {thr[3:]}%"
        #return f"Clustered UBS (ST {thr[3:]}%)"
    return f"Clustered UBS ({thr})"
# ----------------------------
# Metric-specific computation for one (threshold, class, metric)
# ----------------------------
def compute_bins_for(df_thr, mode, metric_key, klass):
    cls_col = f"{mode}_class"
    pA_col  = f"{mode}_p_apo"
    pH_col  = f"{mode}_p_holo"

    d = df_thr.copy()
    d[cls_col] = norm_class(d[cls_col])
    d = d[d[cls_col].isin(["apo", "holo"])].copy()

    # require classification inputs present
    d[pA_col] = pd.to_numeric(d[pA_col], errors="coerce")
    d[pH_col] = pd.to_numeric(d[pH_col], errors="coerce")
    d = d[np.isfinite(d[pA_col]) & np.isfinite(d[pH_col])].copy()

    # select class -- "all" keeps every classified pocket (apo + holo; ties are already gone)
    if klass != "all":
        d = d[d[cls_col] == klass].copy()

    if metric_key == "probability":
        strength = np.maximum(d[pA_col].to_numpy(dtype=float), d[pH_col].to_numpy(dtype=float))
        bins = bin3(strength, PROB_LOW_MAX, PROB_MOD_MAX)

    elif metric_key == "density":
        col = f"{mode}_density_pair"
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d[np.isfinite(d[col])].copy()
        bins = bin3(d[col], DENSSEP_LOW_MAX, DENSSEP_MOD_MAX)

    elif metric_key == "separation":
        col = f"{mode}_overlap_pair"
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d[np.isfinite(d[col])].copy()
        sep = 1.0 - d[col].to_numpy(dtype=float)
        bins = bin3(sep, DENSSEP_LOW_MAX, DENSSEP_MOD_MAX)

    else:
        raise ValueError(metric_key)

    mk = np.isfinite(bins)
    n = int(np.sum(mk))
    if n == 0:
        return np.zeros(3, dtype=float), 0
    bc = np.bincount(bins[mk].astype(int), minlength=3)[:3].astype(int)
    return percent(bc, n), n

# ----------------------------
# Plotting
# ----------------------------
def draw_metric_panel(ax, df, thresholds, mode, metric_key, metric_title, colors,
                      show_class_ticks=False, show_threshold_labels=False, show_xlabel=False):
    cA, cH = colors
    a_low, a_mod, a_high = impact_colors(cA)
    h_low, h_mod, h_high = impact_colors(cH)

    nT = len(thresholds)
    # Group lanes down the y axis. With SHOW_ALL_GROUP off this is exactly the published
    # apo-then-holo layout; the "All" lane is inserted BETWEEN them, matching the AF-Apo / All /
    # AF-Holo order used by the detection figure so the two are read the same way.
    groups = ["all", "apo", "holo"] if SHOW_ALL_GROUP else ["apo", "holo"]
    gcolors = {"apo": (a_low, a_mod, a_high),
               "holo": (h_low, h_mod, h_high),
               "all": impact_colors(ALL_COLOR)}
    # GROUP_GAP was tuned against 4-threshold groups, where the gap is a small fraction of the
    # group's depth. With one threshold per group the same constant opens a gap nearly twice a
    # bar's height, so shrink it in proportion to how deep the groups actually are.
    gap = GROUP_GAP if nT > 1 else GROUP_GAP * SINGLE_ROW_GAP_SCALE
    ypos = {g: np.arange(nT, dtype=float) + gi * (nT + gap)
            for gi, g in enumerate(groups)}

    for i, thr in enumerate(thresholds):
        sub = subset_threshold(df, thr)
        for g in groups:
            segs, _n = compute_bins_for(sub, mode, metric_key, g)
            y = ypos[g][i]
            left = 0.0
            for seg, col in zip(segs, gcolors[g]):
                ax.barh(y, seg, height=BAR_HEIGHT, left=left, color=col, edgecolor="none")
                if SHOW_PERCENT_LABELS and seg >= MIN_LABEL_PCT:
                    # bare numerals: the axis is already titled "Proportion of class (%)" and
                    # its ticks are a bare 0 and 100, so a per-segment sign states the unit a
                    # tenth time. Figure 3's bar-top values follow the same rule.
                    ax.text(left + seg/2, y, f"{seg:.0f}", ha="center", va="center",
                            fontsize=figspec.FS_FLOOR, color="0.25")
                left += float(seg)
            if show_threshold_labels:
                # THRESH_SIDE controls which side the threshold labels appear on: 'left' or 'right'
                label_x = THRESH_LABEL_X if THRESH_SIDE == "left" else RIGHT_LABEL_X
                ha = "right" if THRESH_SIDE == "left" else "left"
                thr_lab = format_row_label(thr)     # main label (black)
                n_lab   = f"n={_n:,}"               # secondary label (gray)
                DY_PTS = 3.6
                up = mtransforms.ScaledTranslation(0, +DY_PTS/72.0, ax.figure.dpi_scale_trans)
                dn = mtransforms.ScaledTranslation(0, -DY_PTS/72.0, ax.figure.dpi_scale_trans)
                ax.text(label_x, y, thr_lab, transform=ax.transData + up, ha=ha, va="center",
                        fontsize=SMALL_FS, color=THRESH_LABEL_COLOR, clip_on=False)
                ax.text(label_x, y, n_lab, transform=ax.transData + dn, ha=ha, va="center",
                        fontsize=SMALL_FS, color=THRESH_LABEL_COLOR, clip_on=False)

    # Y ticks: show Apo/Holo only on first panel
    if show_class_ticks:
        mids = [float(np.mean([ypos[g][0], ypos[g][-1]])) if nT else 0.0 for g in groups]
        ax.set_yticks(mids)
        ax.set_yticklabels([{"apo": figspec.LAB_APO, "holo": figspec.LAB_HOLO,
                             "all": figspec.LAB_ALL}[g] for g in groups],
                           fontsize=TICK_FS)
    else:
        ax.set_yticks([])

    ax.set_xlim(0, 100)
    #ax.set_title(metric_title, fontsize=TITLE_FS, pad=4)
    ax.set_title(metric_title, fontsize=LAB_FS, pad=4)

    if SHOW_X_GRID:
        ax.grid(True, axis="x", which="major", color="0.92", lw=0.7)
    else:
        ax.grid(False)

    if XTICKS_MODE == "ends":
        ax.set_xticks([0, 100])
        # Force tick labels to the edges to avoid collision between panels
        ax.tick_params(axis='x', which='major', pad=3)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_horizontalalignment('center')

        # Manually shift the '100' and '0' text slightly inwards
        ticks = ax.xaxis.get_major_ticks()
        if len(ticks) >= 2:
            ticks[0].label1.set_horizontalalignment('left')   # Push '0' rightwards
            ticks[-1].label1.set_horizontalalignment('right') # Push '100' leftwards
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    if show_xlabel:
        ax.set_xlabel("Within-class (%)", fontsize=LAB_FS)
    else:
        ax.set_xlabel("")

def add_figure_legends(fig, colors):
    cA, cH = colors
    class_handles = [
        Line2D([0],[0], marker="s", linestyle="None", markersize=6,
               markerfacecolor=cA, markeredgecolor="none", label=LEGEND_LABEL_APO),
        Line2D([0],[0], marker="s", linestyle="None", markersize=6,
               markerfacecolor=cH, markeredgecolor="none", label=LEGEND_LABEL_HOLO),
    ]
    if SHOW_ALL_GROUP:
        class_handles.insert(1, Line2D([0],[0], marker="s", linestyle="None", markersize=6,
                                       markerfacecolor=ALL_COLOR, markeredgecolor="none",
                                       label="All"))

    # --- Impact legend (either grayscale or class-tinted) ---
    if IMPACT_LEGEND_STYLE == "grayscale":
        bin_handles = [
            mpatches.Patch(facecolor=shade("0.15", 0.70), edgecolor="none", label="Low"),
            mpatches.Patch(facecolor=shade("0.15", 0.35), edgecolor="none", label="Moderate"),
            mpatches.Patch(facecolor=mcolors.to_rgb("0.15"), edgecolor="none", label="High"),
        ]
    else:
        # class-tinted bins: Low/Moderate/High for Apo + Low/Moderate/High for Holo
        a_low, a_mod, a_high = impact_colors(APO_COLOR if not USE_NEUTRAL_COLORS else NEUTRAL_A)
        h_low, h_mod, h_high = impact_colors(HOLO_COLOR if not USE_NEUTRAL_COLORS else NEUTRAL_H)
    
        bin_handles = [
            mpatches.Patch(facecolor=a_low,  edgecolor="none", label="Apo Low"),
            mpatches.Patch(facecolor=a_mod,  edgecolor="none", label="Apo Moderate"),
            mpatches.Patch(facecolor=a_high, edgecolor="none", label="Apo High"),
            mpatches.Patch(facecolor=h_low,  edgecolor="none", label="Holo Low"),
            mpatches.Patch(facecolor=h_mod,  edgecolor="none", label="Holo Moderate"),
            mpatches.Patch(facecolor=h_high, edgecolor="none", label="Holo High"),
        ]

    if SHOW_CLASS_LEGEND:
        _add_class_legend(fig, class_handles)
    _add_impact_legend(fig, bin_handles)
    return


def _add_class_legend(fig, class_handles):
    leg_group = fig.legend(handles=class_handles, loc="lower left", frameon=True, fontsize=7.0,
                           bbox_to_anchor=((SUBPLOT_LEFT if LEGEND_GROUP_X is None else LEGEND_GROUP_X), LEGEND_Y0), borderpad=0.3,
                           handletextpad=0.4, labelspacing=0.3)
    leg_group_frame = leg_group.get_frame()
    if TRANSPARENT_BACKGROUND:
        leg_group_frame.set_alpha(None)
        leg_group_frame.set_facecolor((1.0, 1.0, 1.0, 0.0))
    else:
        leg_group_frame.set_alpha(0.65)
        leg_group_frame.set_facecolor("white")
    leg_group.get_frame().set_edgecolor("0.85")
    leg_group.get_frame().set_linewidth(0.6)



def _add_impact_legend(fig, bin_handles):
    leg_bins = fig.legend(handles=bin_handles, loc="lower right", frameon=True, fontsize=7.0,
                          ncol=3, bbox_to_anchor=((SUBPLOT_RIGHT if LEGEND_BINS_X is None else LEGEND_BINS_X), LEGEND_Y0),
                          borderpad=0.3, handletextpad=0.4, columnspacing=0.8)
    leg_bins_frame = leg_bins.get_frame()
    if TRANSPARENT_BACKGROUND:
        leg_bins_frame.set_alpha(None)
        leg_bins_frame.set_facecolor((1.0, 1.0, 1.0, 0.0))
    else:
        leg_bins_frame.set_alpha(0.65)
        leg_bins_frame.set_facecolor("white")
    leg_bins.get_frame().set_edgecolor("0.85")
    leg_bins.get_frame().set_linewidth(0.6)

def main():
    configure_from_args(parse_args())
    df = load_unified(INPUT_TABLE)
    mode = MODE.strip().lower()

    require_cols(df, ["threshold", "level",
                      f"{mode}_class",
                      f"{mode}_p_apo", f"{mode}_p_holo",
                      f"{mode}_density_pair",
                      f"{mode}_overlap_pair"])

    present = set(df["threshold"].astype(str).unique())
    thresholds = [t for t in THRESHOLDS_TO_PLOT if str(t) in present]
    for t in THRESHOLDS_TO_PLOT:
        if str(t) not in present:
            print(f"[warn] threshold not found in table: {t}")
    if not thresholds:
        raise SystemExit("No requested thresholds found in table.")

    colors = (NEUTRAL_A, NEUTRAL_H) if USE_NEUTRAL_COLORS else (APO_COLOR, HOLO_COLOR)

    # Figure height scales with #thresholds (bars per panel = 2*nT)
    nT = len(thresholds)
    bars_per_panel = 2 * nT
    
    # Lock height to match Figure 3A exactly, preventing vertical stretching
    fig_h = 3.4 

    fig_w = FIG_W
    if FIG_SIZE_OVERRIDE:
        fig_w, fig_h = FIG_SIZE_OVERRIDE
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), constrained_layout=False, sharex=True)

    metrics = [
        ("probability", "Probability"),
        ("density", "Density"),
        ("separation", "Separation"),
    ]

    for i, (ax, (k, title)) in enumerate(zip(axes, metrics)):
        draw_metric_panel(
            ax, df, thresholds, mode, k, title, colors,
            #show_class_ticks=(i == 0),
            # one threshold per group -> the repeated threshold label carries nothing and the
            # class is what distinguishes the lanes, so the two labellings swap over
            show_class_ticks=(i == 0 and nT == 1),
            show_threshold_labels=(i == 0 and nT > 1),
            show_xlabel=False
        )

    # Centre of the plotting area. Under --fig_size the margins come from figspec.margins() in
    # millimetres, NOT from SUBPLOT_LEFT/RIGHT, so centring on those constants put the shared x
    # label at 0.63 of the canvas against a true centre of 0.54 -- the visible rightward tilt.
    _lm_mm = (16.0 if SHOW_ALL_GROUP else 22.0)
    if FIG_SIZE_OVERRIDE:
        _w_mm = fig_w / figspec.MM
        plot_center_x = 0.5 * (_lm_mm / _w_mm + (1.0 - 2.0 / _w_mm))
    else:
        plot_center_x = SUBPLOT_LEFT + (SUBPLOT_RIGHT - SUBPLOT_LEFT) / 2.0

    # Shared x-label placed exactly beneath the middle panel
    _sup_y = (5.0 / (fig_h / figspec.MM)) if FIG_SIZE_OVERRIDE else SUPXLABEL_Y
    fig.text(plot_center_x, _sup_y, "Proportion of class (%)", 
             ha="center", va="center", fontsize=LAB_FS)

    #if not SHOW_PANEL_LETTER:
    #    fig.suptitle(f"Figure 3C — threshold trends by metric  •  MODE={mode}", fontsize=TITLE_FS, y=0.99)

    if SHOW_PANEL_LETTER and PANEL_LETTER:
        # at the figure's left edge, not at SUBPLOT_LEFT -- that is where column 1 starts, so the
        # letter landed on top of the first panel's title
        fig.text(0.004, 0.995, PANEL_LETTER, ha="left", va="top",
                 fontsize=figspec.FS_LETTER, fontweight="bold")

    if SHOW_LEGENDS:
        add_figure_legends(fig, colors)

    # More bottom room: legends + shared xlabel
    #fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.16, wspace=0.30)
    if FIG_SIZE_OVERRIDE:
        # millimetres once the canvas is the final canvas: SUBPLOT_BOTTOM=0.24 is 20 mm on the
        # 3.4 in default and 10 mm on a 43 mm-tall strip, and the shared x label plus the legend
        # row need a fixed amount of space, not a fixed fraction
        fig.subplots_adjust(**figspec.margins(fig_w, fig_h, left=_lm_mm, right=2.0,
                                              top=4.5, bottom=13.0), wspace=WSPACE)
    else:
        fig.subplots_adjust(left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT, top=SUBPLOT_TOP,
                            bottom=SUBPLOT_BOTTOM, wspace=WSPACE)

    out_pdf, out_png, out_svg = save_outputs(fig)
    plt.close(fig)

    print("Saved:")
    print(" -", out_pdf)
    print(" -", out_svg)
    print(" -", out_png)

if __name__ == "__main__":
    main()
