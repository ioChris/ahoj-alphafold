# -*- coding: utf-8 -*-
"""
Build the MERGED multi-predictor + docking main figure (run_3, harmonized TNS).

This is the durable reconstruction of the previously scratchpad-only
`make_combined_figures.py` (which was lost between sessions), extended per the
2026-07-14 review to merge the old Figure 6 (predictor+docking success / gradient
/ consensus contour) and Figure 7 (winning-probability verbal bins) into ONE
full-page, six-panel figure.

Panels (top -> bottom):
  A   Winning-probability verbal bins, per class (AF-Apo / All / AF-Holo), x3 LBS predictors
  B   Per-predictor detectability CONTOURS (p_holo x log2-ratio), single SHARED colorbar
  C   Detection / dockability success by AlphaFold pocket class (Apo / All / Holo), 5 methods
  D   Consensus success vs p_holo (class colour-switch at 0.5)
  E   Consensus success vs evidence balance (log2 ratio)
  F   Consensus detectability contour

General fixes applied throughout:
  * all contour colorbars are plain RECTANGLES (no triangular extend caps)
  * all contour + line panels in the bottom block are SQUARE (set_box_aspect)
  * colorbar labels read CLOCKWISE (rotation=270, top->bottom) per the manuscript
    right-axis convention
  * unified x-axis label on the two ratio panels (E and F)

Outputs: `--layout v4` (manuscript v4 Figure 3) -> <cfg.MANUSCRIPT_FIG_DIR>/Figure3_detection_and_docking.{png,pdf,svg};
`--layout v3` (the July merged figure) -> <cfg.MANUSCRIPT_FIG_DIR>/combined_predictor_docking/main_lbs_docking_combined.*;
`--out_dir` overrides either. Inputs (they follow AHOJ_AF_MODEL_SET): the harmonized per-pocket tables of
06_lbs_generality (cfg.lbs_tool_dir('p2rank' / 'fpocket' / 'ifsitepred')) and the docking pocket table
<cfg.DOCKING_RUN3_DIR>/analysis/pocket_level_table.csv.

Usage (from the repository root; the manuscript figure is the AFDB v4 re-run):
    AHOJ_AF_MODEL_SET=afdbv4 python plotting/manuscript_v4/make_combined_figure.py --layout v4 --overall bars
"""
import argparse
import os
import sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import BoundaryNorm
from scipy.ndimage import gaussian_filter, distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv4_paths import FIG_DIR, cfg, figspec  # noqa: E402

# --------------------------------------------------------------------------- paths
# every input comes from common/config.py and follows AHOJ_AF_MODEL_SET (afdbv4 -> the _afdbv4 tables)
OUT = os.path.join(str(FIG_DIR), "combined_predictor_docking")   # --layout v3 default; v4 writes FIG_DIR


def poster_dir():
    """Poster/talk export directory: the SAME one the public repo's plotting/ scripts write to, so
    the article and poster variants of every figure sit side by side. Single source of truth is the
    public repo's `common/config.py` anchor `presentation_dir` (set in common/paths.ini). Resolved
    lazily so a machine without that anchor configured can still build the manuscript figure."""
    cfg.require("PRESENTATION_DIR")
    return str(cfg.PRESENTATION_DIR)


# --------------------------------------------------- render state (set once by main())
# Defaults reproduce the manuscript figure exactly; only the CLI changes them.
OUT_DIR = OUT                          # where savef() writes
FORMATS = ("png", "pdf", "svg")        # poster exports drop the pdf (see main())
TRANSPARENT_BACKGROUND = False         # poster mode: transparent outside the axes, white plot area
FONT_SCALE = 1.0                       # uniform multiplier on every point size in this module
# y tick NUMBERS across a multi-column row. Tick MARKS always stay -- dropping them too would
# detach the numbers from the grid. 'first' = numbers on the leftmost column only (the columns
# share a scale, so repeating them is redundant ink); 'all' = every column; 'none' = suppress
# everywhere, for exporting the 2nd..nth panel of a row you are assembling by hand.
YTICK_LABELS = "first"
# SVG text handling. 'none' (matplotlib default here) writes <text> elements that merely NAME the
# font stack -- nothing is embedded, so a viewer without Helvetica Neue substitutes a font whose
# glyph widths differ from the ones matplotlib used to place the text, and labels/legends collide.
# 'path' outlines every glyph, so the file renders identically anywhere at the cost of the text no
# longer being editable. Manuscript keeps 'none' (publishers want live text); posters use 'path'.
SVG_TEXT = "none"
WSPACE = 0.26                   # horizontal gap between row columns, as a fraction of panel width
OVERALL = "none"                # add each method's overall rate to a bins plot: none|lines|bars
GAP_Y = None                    # fixed y for the +gap annotations (None = per-method, as published)
# Bar-top values in the binned panels. The published 5.3 pt is below the manuscript's 6 pt floor,
# and three values per bin at 6 pt do not fit side by side across a four-column 180 mm row --
# upright they run into each other, so they are set on their side.
BAR_LABEL_FS = figspec.FS_FLOOR
BAR_LABEL_ROT = 90
POOL_BINS = False               # collapse the 5 winning-p bins to 3 (very low+low / moderate / high+very high)
XTICK_STYLE = "full"            # 'full' or 'abbrev' bin labels
XTICK_ROT = 0                   # bin-label rotation; a narrow column needs ~40 deg
# Type is specified in POINTS, so how large a title looks depends on the panel it sits on. The
# stock individual-export canvases give a wider panel than a column inside a row, so the same 11 pt
# title covers less of it -- placed side by side at equal panel width in PowerPoint, an individual
# export reads ~20-30%% quieter than a row column. MATCH_ROW sizes an individual export so its AXES
# width equals one row column, making the two interchangeable on a poster.
MATCH_ROW = False
# When a set of individual exports is destined to be laid out as one row by hand, each panel needs
# the label set it would have had INSIDE the row -- y label + y tick numbers on the first column
# only, x label under the middle one -- or the assembled row carries five copies of every label.
AS_ROW_PARTS = False
# axes fraction of the canvas for each standalone layout (mirrors the subplots_adjust calls below)
_AXFRAC = {"bins": 0.97 - 0.16, "contour": 0.82 - 0.15, "contour_nocbar": 0.97 - 0.18}
_ROW_PANEL = {"bins": 2.3839, "contour": 2.0407, "contour_nocbar": 2.0407}


def _show_yticks(i):
    """Does column i of a row keep its y tick numbers?"""
    return {"all": True, "none": False}.get(YTICK_LABELS, i == 0)

# --------------------------------------------------------------------------- style
APO, HOLO, ALLC = figspec.APO, figspec.HOLO, figspec.ALL
CMAP_PRED = "viridis"
PRED_VLIM = (0.20, 0.80)          # shared colour scale for the 3 predictors + consensus
# unified ratio x-label (panels B, E, F). Two framings; build_main() selects one.
# Exact metric log2[(holo+1)/apo] + "positive = holo-leaning" defined in the caption.
RATIO_LABEL_PLAIN = r"Evidence ratio, $\log_2$(holo/apo)"   # opt1' (numeral ticks)
RATIO_LABEL_TAGS  = r"Evidence ratio ($\log_2$)"            # paired with directional tags (which own direction)
XLAB_RATIO = RATIO_LABEL_PLAIN                              # mutable; build_main() sets per mode

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": figspec.FONT_STACK,
    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.titlepad": 5, "figure.dpi": 130,
})
# v4 draws this figure at 180 mm, its true page width, so these ARE the sizes that reach the
# page. The old 11/9.8/8.6/7.6/13 pt were chosen for a 10.8 in canvas that production would
# have had to downscale, which would have taken every one of them below the 6 pt floor.
TITLE_FS = figspec.FS_TITLE
LAB_FS = figspec.FS_TITLE
TICK_FS = figspec.FS_TICK
SMALL_FS = figspec.FS_NOTE
LET_FS = figspec.FS_LETTER


def _fs(sz):
    """Every point size in this module is routed through here so --font_scale rescales the figure
    uniformly -- the named constants above AND the one-off literals on individual annotations. A
    standalone poster panel is a different physical size from its slot in the full-page figure, so
    its text usually wants rescaling with it."""
    return sz * FONT_SCALE

def load(p): return pd.read_csv(p, low_memory=False)

# --------------------------------------------------------------- load / build tables
tabs = {
 "P2Rank":      load(os.path.join(str(cfg.lbs_tool_dir("p2rank")), "lbs_p2rank_st100_merged_per_pocket.csv")),
 "fpocket":     load(os.path.join(str(cfg.lbs_tool_dir("fpocket")), "lbs_fpocket_st100_merged_per_pocket.csv")),
 "IF-SitePred": load(os.path.join(str(cfg.lbs_tool_dir("ifsitepred")), "lbs_ifsitepred_st100_merged_per_pocket.csv")),
}
PRED = ["P2Rank", "fpocket", "IF-SitePred"]
base = tabs["P2Rank"][["site_uid", "pocket_class", "pocket_p_holo",
                       "log2_ratio_holo_to_apo_avg_plus1", "winning_p", "prob_level_db"]].copy()
base = base.rename(columns={"log2_ratio_holo_to_apo_avg_plus1": "log2_ratio"})
for k in PRED:
    base = base.merge(tabs[k][["site_uid", "success_tns"]].rename(columns={"success_tns": k}),
                      on="site_uid", how="left")
base["consensus"] = base[PRED].mean(axis=1)
base["cls"] = base["pocket_class"].str.lower()

dk = load(os.path.join(str(cfg.DOCKING_RUN3_DIR), "analysis", "pocket_level_table.csv"))
dk1 = dk[dk["ligand_group"] == 1].dropna(subset=["is_holo"]).copy()
dk1["dockable_bin"] = (dk1["frac_dockable"] > 0).astype(int)
# join the winning-probability bin and the evidence ratio from the predictor table: the docking
# pocket table carries neither, and both are needed to draw docking on the Panel A / Panel B axes
dk1 = dk1.merge(base[["site_uid", "log2_ratio", "prob_level_db"]], on="site_uid", how="left")
dk1["cls"] = np.where(dk1["is_holo"].eq(1), "holo", "apo")   # docking has no tie class

# pocket counts (computed, never hard-coded): the 3 predictors share an identical
# scored set; docking runs on its own dockable-cognate subset (different denominator)
N_PRED = int(base["P2Rank"].notna().sum())   # 86,946 (== fpocket == IF-SitePred)
N_DOCK = int(len(dk1))                        # docking G1 dockable-cognate pockets

# --------------------------------------------------------------- the five readouts, poster order
# Each entry records WHERE its numbers come from, because the denominators differ: the 3 predictors
# and their consensus are scored on the same 86,946 pockets, docking on its own 60,689 dockable-
# cognate subset. Routing every plot through method_frame() means a method's bar plot and its
# contour can never drift apart, and that the docking n is never quietly reported as the predictor n.
METHOD_ORDER = ["P2Rank", "fpocket", "IF-SitePred", "consensus", "docking"]
METHOD_LABEL = {"consensus": "Consensus", "docking": "Docking"}
METHOD_ALIAS = {"p2rank": "P2Rank", "fpocket": "fpocket", "ifsitepred": "IF-SitePred",
                "if-sitepred": "IF-SitePred", "consensus": "consensus", "docking": "docking"}
DOCK_VLIM_OWN = (0.30, 0.90)     # docking's own scale (manuscript convention: different readout)
CMAP_DOCK = "magma"
DOCKING_SCALE = "shared"         # 'shared' = the predictors' viridis scale; 'own' = the above
# The shared range is a contrast-vs-clipping trade: contourf clips the fill to vlim, so a cell
# above the top is painted the top colour and its gradient is lost, while a range wider than the
# data wastes colour. Pooled over the five surfaces: 0.20-0.80 clips 5.4% but uses 100% of the bar;
# 0.15-0.85 clips 1.3% for 96%; 0.10-0.90 clips 0.3% for 84%; 0.00-1.00 clips nothing but uses only
# 67%. The manuscript ships 0.20-0.80; --contour_vlim overrides it without touching that default.
SHARED_VLIM = PRED_VLIM


def method_frame(m):
    """(frame, success column, n) for one method. Docking carries its own denominator."""
    if m == "docking":
        return dk1, "dockable_bin", N_DOCK
    return base, m, N_PRED


def method_scale(m):
    """(vlim, cmap) for a method's contour. Docking is a different quantity on a different
    denominator, so the manuscript gives it its own magma scale; a poster row of five surfaces
    usually wants them all on ONE scale instead (--docking_scale). Sharing clips 8.4% of docking's
    surface cells, less than the 14.0% IF-SitePred already clips on that same shared scale."""
    if m == "docking" and DOCKING_SCALE == "own":
        return DOCK_VLIM_OWN, CMAP_DOCK
    return SHARED_VLIM, CMAP_PRED


LEVELS5 = ["very_low", "low", "moderate", "high", "very_high"]
PRETTY = {"very_low": "very low", "low": "low", "moderate": "moderate", "high": "high", "very_high": "very high"}
ABBREV = {"very low": "v. low", "low": "low", "moderate": "mod.", "high": "high",
          "very high": "v. high", "overall": "overall"}
# Pooled 3-bin grouping: trades gradient resolution for room. The x labels are the binding
# constraint once an 'overall' group is added -- at 8.6 pt "moderate" alone needs more width than
# a sixth of a column, and wrapping it cannot help because it is a single word.
POOLED = [("low", ["very_low", "low"]), ("moderate", ["moderate"]), ("high", ["high", "very_high"])]


def bin_spec():
    """[(label, [prob_level_db members])] for the current binning."""
    if POOL_BINS:
        return list(POOLED)
    return [(PRETTY[l], [l]) for l in LEVELS5]


def xtick_text(label):
    return ABBREV.get(label, label) if XTICK_STYLE == "abbrev" else label

def apply_background(fig):
    """Poster mode = transparent OUTSIDE the axes, white INSIDE the plot area. The plot area has to
    stay white: the contour panels draw their iso-lines and iso-labels in white and the bar/line
    panels use a near-white grid, all of which would vanish on a coloured poster board."""
    if TRANSPARENT_BACKGROUND:
        fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        for ax in fig.axes:
            ax.set_facecolor("white")
    else:
        fig.patch.set_facecolor("white")


def savef(fig, name, dpi=300, exact=False):
    """`exact` saves the FULL canvas. The v4 figure is drawn at 180 mm with millimetre margins,
    so a tight crop would hand the manuscript a 171 mm image to be scaled back up to 180 -- the
    post-hoc rescaling the whole v4 pass exists to remove."""
    apply_background(fig)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=dpi,
                    bbox_inches=(None if exact else "tight"),
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote {}.{{{}}}".format(name, ",".join(FORMATS)))

def _gap_y(rates, pad=8.5):
    """Height for a '+N pp' gap annotation. Default: clear of the tallest bar it spans, which is
    what the published Panel C does per method -- so the annotations sit at DIFFERENT heights when
    several methods are shown side by side. Passing --gap_y locks them to one line instead."""
    if GAP_Y is not None:
        return GAP_Y
    return max(max(r["apo"], r["holo"]) for r in rates) + pad


def center_row_xlabel(fig, axes, text, fontsize):
    """Move a row's shared x label to the centre of ALL its columns.

    `ax.set_xlabel` centres on the axes it is set on, so a four-column row labelled on column 2
    or 3 sits most of a column off centre -- row a was labelled on axes[2] and row b on axes[1],
    landing at 0.60 and 0.34 of the canvas against a true row centre near 0.46. The label is
    still SET on an axes first, so matplotlib computes its vertical position below the tick row;
    only the x is overridden, and it is read back from the drawn extent so it stays correct
    whatever the tick labels turn out to be.
    """
    fig.canvas.draw()
    src = next(ax for ax in axes if ax.get_xlabel())
    bb = src.xaxis.label.get_window_extent()
    inv = fig.transFigure.inverted()
    y0 = inv.transform((bb.x0, bb.y0))[1]
    y1 = inv.transform((bb.x1, bb.y1))[1]
    src.set_xlabel("")
    cx = 0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1)
    fig.text(cx, 0.5 * (y0 + y1), text, ha="center", va="center", fontsize=fontsize)


def rate(df, col):
    s = df.dropna(subset=[col])
    return dict(all=100*s[col].mean(),
                apo=100*s[s.cls.eq("apo")][col].mean(),
                holo=100*s[s.cls.eq("holo")][col].mean())

def letter(ax, s, dx=-0.16, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=_fs(LET_FS), fontweight="bold")

# =================================================================== A: winning-p bins x3
def draw_winning_bins(axes, put_letter=True, methods=None):
    """Success vs winning-probability bin, split by class, one subplot per method.

    `methods` defaults to the three predictors (the manuscript Panel A). Passing consensus and/or
    docking extends the same panel to five columns for poster use -- every column keeps the 0-100
    y-axis, so the columns stay directly comparable however they are later arranged."""
    methods = list(methods or PRED)
    spec = bin_spec()
    nb = len(spec)
    w = 0.27
    for j, (ax, m) in enumerate(zip(axes, methods)):
        df, scol, n_m = method_frame(m)
        xpos = np.arange(nb)
        series = [("apo", APO, -w), ("all", ALLC, 0.0), ("holo", HOLO, +w)]
        for cl, col, off in series:
            r = []
            for _lab, members in spec:
                sel = df.prob_level_db.isin(members)
                s = df[sel] if cl == "all" else df[sel & (df.cls == cl)]
                r.append(100*s[scol].mean() if len(s) else np.nan)
            lab = {"apo": "AF-Apo", "all": "All", "holo": "AF-Holo"}[cl]
            bars = ax.bar(xpos + off, r, w, color=col, edgecolor="0.3", lw=.6, label=lab)
            for bar, v in zip(bars, r):
                # the middle ('All') bar carries no numeral: the three bars of a bin are 0.27
                # data units apart, ~1.2 mm at four columns across 180 mm, against a 2.1 mm
                # numeral. Two fit, three do not, and the 'All' value is the one the section
                # does not quote -- the class pair and the overall group are what it argues from.
                if np.isfinite(v) and cl != "all":
                    ax.text(bar.get_x()+bar.get_width()/2, v+1.2, f"{v:.0f}", ha="center",
                            va="bottom", fontsize=_fs(BAR_LABEL_FS), rotation=BAR_LABEL_ROT,
                            color="0.25")
        ax.set_title(METHOD_LABEL.get(m, m), fontsize=_fs(TITLE_FS))
        ax.set_xticks(xpos)
        ax.set_xticklabels([xtick_text(lab) for lab, _ in spec], rotation=XTICK_ROT,
                           ha=("right" if XTICK_ROT else "center"),
                           rotation_mode=("anchor" if XTICK_ROT else None))
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#ececec", lw=.7, ls="--"); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=_fs(TICK_FS))
        if not _show_yticks(j):
            ax.tick_params(labelleft=False)
        # pocket count (same for all three predictors), top-RIGHT so it never competes with
        # the legend (which lives top-left of the first subplot)
        # white ground, no frame: the dotted divider fencing off the 'overall' group runs
        # straight under this label, and at print size the two read as one smudge
        ax.text(0.97, 0.98, f"n = {n_m:,}", transform=ax.transAxes, ha="right", va="top",
                fontsize=_fs(6.6), color="0.45", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
        # --- each method's OVERALL rate, so a bin can be read against its own method average
        if OVERALL == "lines":
            # dashed reference line per class + its value at the right edge. Same idiom as the
            # internal Panel A grid in 06_lbs_generality/analysis/plot_fig6a_quality_gates.py.
            # Costs a sliver of right margin instead of a whole extra bar group.
            ov = rate(df, scol)
            for cl, col in [("apo", APO), ("all", ALLC), ("holo", HOLO)]:
                if not np.isfinite(ov[cl]):
                    continue
                ax.plot([-0.5, nb - 0.58], [ov[cl]]*2, color=col, lw=1.0, ls=(0, (4, 2)),
                        alpha=.85, zorder=0, clip_on=False, solid_capstyle="butt")
                ax.text(nb - 0.48, ov[cl], f"{ov[cl]:.0f}", ha="left", va="center",
                        fontsize=_fs(5.8), color=col, fontweight="bold")
            ax.set_xlim(-0.55, nb + 0.05)
            ax.annotate(f"+{ov['holo']-ov['apo']:.0f}", (nb - 0.25, _gap_y([ov])), ha="center",
                        fontsize=_fs(7.2), fontweight="bold", color="0.3")
        elif OVERALL == "bars":
            # a sixth group carrying the marginal rate, fenced off from the binned ones
            ov = rate(df, scol)
            for cl, col, off in [("apo", APO, -w), ("all", ALLC, 0.0), ("holo", HOLO, +w)]:
                ax.bar([nb + 0.6 + off], [ov[cl]], w, color=col, edgecolor="0.3", lw=.6,
                       hatch="///", zorder=2)
                if cl == "all":
                    continue
                ax.text(nb + 0.6 + off, ov[cl] + 1.2, f"{ov[cl]:.0f}", ha="center", va="bottom",
                        fontsize=_fs(BAR_LABEL_FS), rotation=BAR_LABEL_ROT, color="0.25")
            ax.axvline(nb - 0.15, color="0.8", lw=.9, ls=":")
            ax.annotate(f"+{ov['holo']-ov['apo']:.0f}", (nb + 0.6, _gap_y([ov])), ha="center",
                        fontsize=_fs(7.2), fontweight="bold", color="0.3")
            ax.set_xlim(-0.6, nb + 1.3)
            ax.set_xticks(list(xpos) + [nb + 0.6])
            ax.set_xticklabels([xtick_text(lab) for lab, _ in spec] + [xtick_text("overall")],
                               rotation=XTICK_ROT,
                               ha=("right" if XTICK_ROT else "center"),
                               rotation_mode=("anchor" if XTICK_ROT else None))
            # tick_params ran earlier in this loop, so restate the size or these labels would keep
            # matplotlib's default rather than the panel's TICK_FS
            ax.tick_params(axis="x", labelsize=_fs(TICK_FS))
            ax.get_xticklabels()[-1].set_fontweight("bold")
    axes[0].set_ylabel("Success rate (%)", fontsize=_fs(LAB_FS))
    # x-label under the middle column, whatever the column count (1, 3 or 5)
    axes[len(axes)//2].set_xlabel("Winning-probability bin", fontsize=_fs(LAB_FS), labelpad=8)
    # legend on the first subplot, top-left
    axes[0].legend(frameon=False, fontsize=_fs(SMALL_FS), loc="upper left",
                   handlelength=1.1, labelspacing=.3, borderaxespad=.2)
    if put_letter:
        letter(axes[0], "a", dx=-0.30, dy=1.05)

# =================================================================== contour surface
def _surface(xv, yv, sv, xlim=(-4, 4), ylim=(0, 1), xb=46, yb=38, sigma=1.25):
    m = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(sv)
    xv, yv, sv = np.clip(xv[m], *xlim), yv[m], sv[m]
    xe = np.linspace(*xlim, xb+1); ye = np.linspace(*ylim, yb+1)
    ssum, _, _ = np.histogram2d(yv, xv, bins=[ye, xe], weights=sv)
    cnt, _, _ = np.histogram2d(yv, xv, bins=[ye, xe])
    valid = cnt > 0
    filled = np.where(valid, np.divide(ssum, cnt, out=np.zeros_like(ssum), where=valid), 0.0)
    sn = gaussian_filter(filled*valid, sigma=sigma); sd = gaussian_filter(valid.astype(float), sigma=sigma)
    R = np.divide(sn, sd, out=np.full_like(sn, np.nan), where=sd > 0.015)
    nanm = ~np.isfinite(R)
    if nanm.any():
        idx = distance_transform_edt(nanm, return_distances=False, return_indices=True)
        R = R[tuple(idx)]
    xc = (xe[:-1]+xe[1:])/2; yc = (ye[:-1]+ye[1:])/2
    return np.meshgrid(xc, yc), R

def draw_contour(ax, sv, vlim, cmap=CMAP_PRED, title=None, xv=None, yv=None,
                 cbar=True, cax=None, cbar_label="Success rate (%)", square=True,
                 show_xlabel=True, show_ylabel=True, xlim_full=False, note=None,
                 show_yticklabels=True):
    """One contour panel. Rectangular colorbar (data clipped to vlim -> gap-free fill,
    no triangular extend caps). Colorbar label reads clockwise (rotation=270).
    Tick NUMBERS are always shown; show_xlabel/show_ylabel toggle only the axis-label text,
    so Panel B's three surfaces keep readable ticks but state each axis name once."""
    xv = base.log2_ratio.to_numpy(float) if xv is None else xv
    yv = base.pocket_p_holo.to_numpy(float) if yv is None else yv
    (XX, YY), R = _surface(xv, yv, sv)
    vmin, vmax = vlim
    levels = np.round(np.arange(vmin, vmax+1e-9, 0.05), 3)
    Rf = np.clip(R, vmin+1e-9, vmax-1e-9)                       # keep fill inside range -> rectangular bar
    cf = ax.contourf(XX, YY, Rf, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)  # no 'extend'
    cline = [v for v in np.round(np.arange(vmin, vmax+1e-9, 0.1), 3) if vmin < v < vmax]
    cs = ax.contour(XX, YY, R, levels=cline, colors="white", linewidths=.7, alpha=.9)
    ax.clabel(cs, inline=True, fmt=lambda v: f"{int(round(v*100))}", fontsize=_fs(6.4))
    ax.axvline(0, color="white", lw=.8, ls=":", alpha=.6)
    ax.axhline(0.5, color="white", lw=.8, ls=":", alpha=.6)
    if show_xlabel:
        ax.set_xlabel(XLAB_RATIO, fontsize=_fs(LAB_FS))
    if show_ylabel:
        ax.set_ylabel(r"$p_\mathrm{holo}$", fontsize=_fs(LAB_FS))
    if title: ax.set_title(title, fontsize=_fs(TITLE_FS))
    if note:                      # e.g. the pocket count, for a surface exported on its own
        ax.text(1.0, 1.015, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=_fs(6.6), color="0.45")
    ax.tick_params(labelsize=_fs(TICK_FS))
    if not show_yticklabels:
        ax.tick_params(labelleft=False)
    if xlim_full:
        ax.set_xlim(-4, 4)          # show the full -4..4 range so -4/+4 ticks render (tag mode)
    if square: ax.set_box_aspect(1)
    if cbar:
        _add_cbar(ax.figure, cf, vmin, vmax, cbar_label, cax=cax, ref_ax=ax)
    return cf

def apply_ratio_tags(ax, two_line=False, tag_fs=7.4):
    """Directional evidence-axis: numerals at -4/0/4, colored 'More apo' (<-) / 'More holo' (->)
    tags replacing the +-2 numerals, sitting below the axis.

    `two_line` stacks the arrow under the word. At four columns across 180 mm a one-line tag is
    ~13 mm wide against a ~4 mm data unit, so it reaches the -4 numeral on one side and the 0 on
    the other; stacked, it is the width of the word alone and clears both. That width is the
    reason the v4 layout dropped these tags in the first place."""
    ax.set_xticks([-4, -2, 0, 2, 4])
    # mathtext arrows render regardless of the sans font (Helvetica Neue lacks U+2192)
    if two_line:
        apo_lab = "More apo\n" + r"$\leftarrow$"
        holo_lab = "More holo\n" + r"$\rightarrow$"
    else:
        apo_lab = r"$\leftarrow$ More apo"
        holo_lab = r"More holo $\rightarrow$"
    ax.set_xticklabels(["−4", apo_lab, "0", holo_lab, "4"])
    specs = [("0.15", "normal", TICK_FS), (APO, "bold", tag_fs), ("0.15", "normal", TICK_FS),
             (HOLO, "bold", tag_fs), ("0.15", "normal", TICK_FS)]
    for t, (c, w, fs) in zip(ax.get_xticklabels(), specs):
        t.set_color(c); t.set_fontweight(w); t.set_fontsize(_fs(fs))
        if two_line:
            t.set_linespacing(0.95)

def _add_cbar(fig, cf, vmin, vmax, label, cax=None, ref_ax=None):
    ticks = np.round(np.arange(vmin, vmax+1e-9, 0.1), 3)
    if cax is not None:
        cb = fig.colorbar(cf, cax=cax, ticks=ticks)
    else:
        cb = fig.colorbar(cf, ax=ref_ax, fraction=0.046, pad=0.03, ticks=ticks)
    cb.set_label(label, fontsize=_fs(LAB_FS), rotation=270, labelpad=14)   # clockwise / inward
    cb.ax.set_yticklabels([f"{int(round(t*100))}" for t in ticks])
    cb.ax.tick_params(labelsize=_fs(TICK_FS))
    return cb


def match_cbar(fig, cax, ref_ax, gap=0.014, width=0.013):
    """Snap a colorbar axes to the right edge and exact height of its reference panel. Needs a draw
    first: set_box_aspect(1) only settles the panel's true position at draw time."""
    fig.canvas.draw()
    p = ref_ax.get_position()
    cax.set_position([p.x1 + gap, p.y0, width, p.height])

# =================================================================== C: success by class
def draw_success_by_class(ax, put_letter=True):
    methods = ["P2Rank", "fpocket", "IF-SitePred", "consensus", "docking"]
    disp = {"consensus": "Consensus", "docking": "Docking"}
    x = np.arange(len(methods)); w = 0.26
    def vals(m):
        if m == "docking":
            s = dk1
            return (100*s[s.is_holo.eq(0)].dockable_bin.mean(),
                    100*s.dockable_bin.mean(),
                    100*s[s.is_holo.eq(1)].dockable_bin.mean())
        r = rate(base, m); return (r["apo"], r["all"], r["holo"])
    apov = [vals(m)[0] for m in methods]
    allv = [vals(m)[1] for m in methods]
    holv = [vals(m)[2] for m in methods]
    # order: Apo, All, Holo
    b_apo = ax.bar(x - w, apov, w, color=APO,  edgecolor="0.3", lw=.6, label="AF-Apo")
    b_all = ax.bar(x,     allv, w, color=ALLC, edgecolor="0.3", lw=.6, label="All")
    b_hol = ax.bar(x + w, holv, w, color=HOLO, edgecolor="0.3", lw=.6, label="AF-Holo")
    for bars, vv in [(b_apo, apov), (b_all, allv), (b_hol, holv)]:
        for bar, v in zip(bars, vv):
            ax.text(bar.get_x()+bar.get_width()/2, v+1.3, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=_fs(6.6), color="0.2")
    for xi in range(len(methods)):
        g = holv[xi]-apov[xi]
        ax.annotate(f"+{g:.0f}", (xi, _gap_y([{"apo": apov[xi], "holo": holv[xi]}])),
                    ha="center", fontsize=_fs(7.2), fontweight="bold", color="0.3")
    ax.set_xticks(x); ax.set_xticklabels([disp.get(m, m) for m in methods], fontsize=_fs(TICK_FS))
    ax.set_ylabel("Success rate (%)", fontsize=_fs(LAB_FS)); ax.set_ylim(0, 100)
    ax.axvline(3.5, color="0.8", lw=.9, ls=":")
    # docking runs on its own (different) denominator -> state its n here; the predictors'
    # shared n = 86,946 is carried in Panel A (not restated to avoid redundancy)
    ax.text(4.0, 94, "different denominator", ha="center", fontsize=_fs(6.2), color="0.55", style="italic")
    ax.text(4.0, 86, f"n = {N_DOCK:,}", ha="center", fontsize=_fs(6.8), color="0.4")
    ax.grid(axis="y", color="#ececec", lw=.7, ls="--"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=_fs(SMALL_FS), ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13),
              handlelength=1.2, columnspacing=1.3)
    ax.tick_params(labelsize=_fs(TICK_FS))
    if put_letter:
        letter(ax, "c", dx=-0.065, dy=1.10)

# =================================================================== D: gradient p_holo
def _binned(xv, yv, edges, minn=25):
    idx = np.digitize(xv, edges) - 1
    out = np.full(len(edges)-1, np.nan)
    for b in range(len(out)):
        m = idx == b
        if m.sum() >= minn: out[b] = np.nanmean(yv[m])
    return out

def draw_gradient_pholo(ax, put_letter=True):
    xb = np.linspace(0, 1, 26); xc = (xb[:-1]+xb[1:])/2
    ph = base.pocket_p_holo.to_numpy(float)
    cons = _binned(ph, base.consensus.to_numpy(float)*100, xb)
    L = xc <= 0.5; Rr = xc >= 0.5
    ax.plot(xc[L], cons[L], color=APO, lw=3, solid_capstyle="round")
    ax.plot(xc[Rr], cons[Rr], color=HOLO, lw=3, solid_capstyle="round")
    for k, ls in zip(PRED, [":", "--", "-."]):
        ax.plot(xc, _binned(ph, base[k].to_numpy(float)*100, xb), color="0.58", lw=1.0, ls=ls, label=k)
    ax.axvline(0.5, color="0.7", lw=.9, ls=":")
    # class tags moved to the TOP of the plot area (clears the per-predictor legend at lower-right)
    ax.text(0.25, 92, "AF-Apo", color=APO, ha="center", fontsize=_fs(SMALL_FS), fontweight="bold")
    ax.text(0.75, 92, "AF-Holo", color=HOLO, ha="center", fontsize=_fs(SMALL_FS), fontweight="bold")
    ax.set_xlabel(r"$p_\mathrm{holo}$", fontsize=_fs(LAB_FS))
    ax.set_ylabel("Consensus success (%)", fontsize=_fs(LAB_FS))
    ax.set_ylim(0, 100); ax.set_xlim(0, 1)
    ax.grid(color="#ececec", lw=.7, ls="--"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=_fs(6.2), loc="lower right", title="per predictor", title_fontsize=_fs(6.2))
    ax.tick_params(labelsize=_fs(TICK_FS))
    ax.set_box_aspect(1)
    if put_letter:
        letter(ax, "d", dx=-0.22, dy=1.02)

# =================================================================== E: gradient ratio
def draw_gradient_ratio(ax, put_letter=True):
    rb = np.linspace(-4, 4, 33); rc = (rb[:-1]+rb[1:])/2
    for cl, col in [("apo", APO), ("holo", HOLO)]:
        sub = base[base.cls.eq(cl)]
        ax.plot(rc, _binned(sub.log2_ratio.to_numpy(float), sub.consensus.to_numpy(float)*100, rb),
                color=col, lw=2.6, label=f"AF-{cl.capitalize()}")
    ax.plot(rc, _binned(base.log2_ratio.to_numpy(float), base.consensus.to_numpy(float)*100, rb),
            color="0.3", lw=1.3, ls="--", label="All")
    ax.axvline(0, color="0.7", lw=.9, ls=":")
    ax.set_xlim(-4, 4); ax.set_ylim(0, 100)
    ax.set_xlabel(XLAB_RATIO, fontsize=_fs(LAB_FS))               # unified with panel F
    ax.set_ylabel("Consensus success (%)", fontsize=_fs(LAB_FS))
    ax.grid(color="#ececec", lw=.7, ls="--"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=_fs(SMALL_FS), loc="lower right")
    ax.tick_params(labelsize=_fs(TICK_FS))
    ax.set_box_aspect(1)
    if put_letter:
        letter(ax, "e", dx=-0.22, dy=1.02)

# =================================================================== assemble main figure
def build_main(tags=False, outname="main_lbs_docking_combined"):
    global XLAB_RATIO
    XLAB_RATIO = RATIO_LABEL_TAGS if tags else RATIO_LABEL_PLAIN
    fig = plt.figure(figsize=(10.8, 11.5))
    # uniform vertical spacing between all panel rows (single hspace, no spacer rows)
    gs = GridSpec(4, 4, figure=fig,
                  width_ratios=[1, 1, 1, 0.06],
                  height_ratios=[0.78, 1.02, 0.74, 1.02],
                  hspace=0.44, wspace=0.26,
                  left=0.07, right=0.94, top=0.965, bottom=0.05)

    # --- Row A : winning-p bins
    axA = [fig.add_subplot(gs[0, i]) for i in range(3)]
    draw_winning_bins(axA)

    # --- Row B : per-predictor contours + single shared colorbar.
    # ticks on all three; y-label once (P2Rank/left), x-label once (fpocket/middle)
    axB = [fig.add_subplot(gs[1, i]) for i in range(3)]
    cf = None
    for i, (ax, m) in enumerate(zip(axB, PRED)):
        cf = draw_contour(ax, base[m].to_numpy(float), PRED_VLIM, title=None, cbar=False,
                          square=True, show_xlabel=(i == 1), show_ylabel=(i == 0), xlim_full=tags)
    caxB = fig.add_axes([0.9, 0.55, 0.012, 0.16])            # placeholder; repositioned below
    _add_cbar(fig, cf, *PRED_VLIM, "Success rate (%)", cax=caxB)
    letter(axB[0], "b", dx=-0.30, dy=1.05)

    # --- Row C : success by class (spans the 3 data columns)
    axC = fig.add_subplot(gs[2, 0:3])
    draw_success_by_class(axC)

    # --- Row D/E/F
    axD = fig.add_subplot(gs[3, 0]); draw_gradient_pholo(axD)
    axE = fig.add_subplot(gs[3, 1]); draw_gradient_ratio(axE)
    axF = fig.add_subplot(gs[3, 2])
    cfF = draw_contour(axF, base.consensus.to_numpy(float), PRED_VLIM, title="Consensus detectability",
                       cbar=False, square=True, xlim_full=tags)
    caxF = fig.add_axes([0.9, 0.08, 0.012, 0.16])
    _add_cbar(fig, cfF, *PRED_VLIM, "Consensus success (%)", cax=caxF)
    letter(axF, "f", dx=-0.22, dy=1.02)

    # --- directional evidence-axis tags: all three Panel B contours (consistent across the row),
    #     plus Panels E and F. Axis label stays once (under Panel B center + on E, F).
    if tags:
        for ax in axB:
            apply_ratio_tags(ax)
        apply_ratio_tags(axE)
        apply_ratio_tags(axF)

    # --- match the two shared colorbars to the height/right-edge of their reference squares
    match_cbar(fig, caxB, axB[2])
    match_cbar(fig, caxF, axF)

    savef(fig, outname)


# =================================================================== assemble the v4 main figure
def build_main_v4(tags=False, outname="Figure3_detection_and_docking"):
    """Manuscript v4 Figure 3 -- "does the label predict anything?".

    Drawn at 180 mm, the true page width, so nothing is downscaled afterwards.

    Layout, settled with the author 2026-08-24 -- the consensus and docking surfaces are added by
    WIDENING rows rather than adding a fourth, because the figure was already over the page height
    and a fourth row would have pushed every type size below the 6 pt floor at print size:

        a   winning-probability bins, 4 columns: P2Rank | fpocket | IF-SitePred | consensus
        b   detectability contours, the SAME 4 methods, column-aligned under row a
        c   consensus success vs p_holo
        d   consensus success vs evidence ratio
        e   docking dockability bins
        f   docking detectability contour

    Row a folds in what used to be Panel C (each method's overall rate by class) via OVERALL.
    Docking keeps its own panels rather than a column in rows a/b: its class-assigned pockets are a
    NESTED subset of the predictors', and a shared row would read as one pooled denominator.
    """
    global XLAB_RATIO, XTICK_ROT, GAP_Y
    XLAB_RATIO = RATIO_LABEL_TAGS       # the directional tags own the direction; see below
    XTICK_ROT = 40
    if GAP_Y is None:
        # One line for every '+N pp' across the figure. Floating each above its own tallest bar
        # left them at five different heights, which reads as five different quantities. It only
        # has to clear the 'overall' group it sits over (69 at the tallest, ~77 with its
        # rotated value) and stay under the n counter, whose box reaches down to ~90.
        GAP_Y = 79.0

    # ROW GEOMETRY, in millimetres. `hspace` is a single number for the whole grid, so a uniform
    # 0.62 put the SAME 21 mm between rows a and b as between b and row 3 -- and rows a and b are
    # the same four predictors, column-aligned, while row 3 is four different panels. Two spacer
    # ROWS with hspace=0 buy per-gap control, so the pairing can be shown by proximity.
    #
    # The a-b gap cannot shrink much: it holds row a's 40-degree bin labels (~8 mm) plus its axis
    # title (~5 mm), so ~13 mm of it is ink. At 17 mm row a's "Winning-probability bin" sat level
    # with row b's letter and read as b's title (author, 2026-10-07); 22 mm gives that title about
    # the white that row b's own axis title has above row 3's letters (~5 mm). What makes the
    # pairing read is still that the b-to-row-3 gap carries more: row b's two-line directional
    # tags, its axis title, and row 3's own titles and letters.
    ROW_A, GAP_AB, ROW_B, GAP_B3, ROW_C = 33.0, 22.0, 33.0, 19.0, 30.5
    TOP_MM, BOTTOM_MM = 5.0, 15.0     # bottom: panel e has 40-degree labels AND a title under them
    W_IN = figspec.W2 * figspec.MM
    H_IN = (TOP_MM + ROW_A + GAP_AB + ROW_B + GAP_B3 + ROW_C + BOTTOM_MM) * figspec.MM
    fig = plt.figure(figsize=(W_IN, H_IN))
    # 4 data columns + a narrow colorbar column. wspace was 0.42, which spent 9.6 mm of every
    # gap on tick numbers that the columns no longer repeat; at 0.20 rows a and b gain ~16% of
    # their column width, which is what the bar-top values and the directional tags needed.
    # Row 3 does NOT want that gap, so it gets its own sub-grid below: panel f is the one
    # bottom-row panel that still carries a y axis, and its neighbour e no longer has tick
    # numbers holding a gap open for the label to sit in.
    gs = GridSpec(5, 5, figure=fig,
                  width_ratios=[1, 1, 1, 1, 0.05],
                  height_ratios=[ROW_A, GAP_AB, ROW_B, GAP_B3, ROW_C],
                  hspace=0.0, wspace=0.20,
                  **{k: v for k, v in figspec.margins(
                      W_IN, H_IN, left=12.0, right=13.5,
                      top=TOP_MM, bottom=BOTTOM_MM).items()})

    METH_A = PRED + ["consensus"]

    # --- Row a : winning-p bins, one column per method, each with its own overall group
    axA = [fig.add_subplot(gs[0, i]) for i in range(4)]
    draw_winning_bins(axA, methods=METH_A)

    # --- Row b : the same four methods as contours, column-aligned under row a
    axB = [fig.add_subplot(gs[2, i]) for i in range(4)]
    cf = None
    for i, (ax, m) in enumerate(zip(axB, METH_A)):
        # y tick NUMBERS on the first column only -- the four surfaces share one p_holo scale,
        # and repeating it four times is the width the directional tags below need
        cf = draw_contour(ax, base[m].to_numpy(float), PRED_VLIM, title=None, cbar=False,
                          square=True, show_xlabel=(i == 1), show_ylabel=(i == 0),
                          xlim_full=True, show_yticklabels=_show_yticks(i))
    caxB = fig.add_axes([0.9, 0.55, 0.012, 0.16])            # placeholder; repositioned below
    _add_cbar(fig, cf, *PRED_VLIM, "Success rate (%)", cax=caxB)
    letter(axB[0], "b", dx=-0.30, dy=1.05)

    # --- Row c : the consensus gradients, then docking's own two panels.
    # All four share the 0-100% success axis, so it is named ONCE, on c, and the method is
    # carried by the column titles instead -- the same idiom rows a and b already use.
    # Row 3 carries four DIFFERENT panels rather than four columns of one series, so it is not
    # column-aligned to rows a and b -- it spans the same width and gives the space back as gaps.
    gs3 = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[4, 0:4], wspace=0.30)
    axC = fig.add_subplot(gs3[0, 0]); draw_gradient_pholo(axC, put_letter=False)
    axC.set_ylabel("Success rate (%)", fontsize=_fs(LAB_FS))
    axC.set_title("Consensus", fontsize=_fs(TITLE_FS))
    letter(axC, "c", dx=-0.30, dy=1.10)
    axD = fig.add_subplot(gs3[0, 1]); draw_gradient_ratio(axD, put_letter=False)
    axD.set_ylabel(""); axD.tick_params(labelleft=False)
    axD.set_title("Consensus", fontsize=_fs(TITLE_FS))
    letter(axD, "d", dx=-0.12, dy=1.10)
    axE = fig.add_subplot(gs3[0, 2]); draw_winning_bins([axE], put_letter=False, methods=["docking"])
    # Lifted and tightened. As drawn by draw_winning_bins the key started 1.4% below the axis top
    # with 0.3 label spacing and reached down to y = 73.3, which is exactly where the rotated
    # "56 | 61" values of the very-low bin end -- they touched. Flush at the top with 0.18 spacing
    # it ends at 76.4, about 1 mm clear. It stays ONE column: at two columns it is 22 mm wide on a
    # 30 mm panel and runs into the n counter at the top right.
    #
    # The vertical and horizontal insets have to be set SEPARATELY. borderaxespad/borderpad are
    # isotropic, so zeroing them to buy the lift also put the key hard against the left spine --
    # which, unlike the top, is a drawn line. So: zero padding, and the horizontal inset comes
    # from bbox_to_anchor instead, in points converted through the panel's own drawn width so it
    # matches what panel a gets from its borderaxespad .2 + borderpad .4 at the same type size.
    fig.canvas.draw()          # box_aspect only settles the panel's true width at draw time
    _pad_pt = _fs(SMALL_FS) * 0.6
    _pad_ax = (_pad_pt / 72.0) / (axE.get_window_extent().width / fig.dpi)
    axE.legend(frameon=False, fontsize=_fs(SMALL_FS), loc="upper left",
               bbox_to_anchor=(_pad_ax, 1.0), handlelength=1.0, handletextpad=0.5,
               labelspacing=0.18, borderpad=0.0, borderaxespad=0.0)
    # square, like c, d and f: without a box aspect this panel filled its whole grid cell while
    # the three around it were constrained to squares, which is the elongation in the rev-2 read
    axE.set_ylabel(""); axE.tick_params(labelleft=False); axE.set_box_aspect(1)
    letter(axE, "e", dx=-0.12, dy=1.10)
    axF = fig.add_subplot(gs3[0, 3])
    # docking's surface comes from its OWN frame and denominator (method_frame), and is drawn on
    # the predictors' shared scale so row b and this panel are directly comparable
    dfF, scolF, nF = method_frame("docking")
    draw_contour(axF, dfF[scolF].to_numpy(float), PRED_VLIM, title="Docking", cbar=False,
                 square=True, xv=dfF.log2_ratio.to_numpy(float),
                 yv=dfF.pocket_p_holo.to_numpy(float), xlim_full=True,
                 show_ylabel=True)    # its own y axis: c's label is three columns away
    axF.yaxis.labelpad = 2.0
    letter(axF, "f", dx=-0.30, dy=1.10)

    # Directional tags on every ratio axis: the coloured More apo / More holo pair replacing the
    # +-2 numerals. Restored in rev 3 -- the width that made them overlap came from four columns
    # of repeated y tick numbers and a 0.42 wspace, both of which are gone, and the tags are
    # stacked over two lines so each is only as wide as its word.
    for ax in axB + [axD, axF]:
        apply_ratio_tags(ax, two_line=True, tag_fs=6.5)

    # x labels centred on the row rather than on whichever column carries them
    center_row_xlabel(fig, axA, "Winning-probability bin", _fs(LAB_FS))
    center_row_xlabel(fig, axB, XLAB_RATIO, _fs(LAB_FS))

    match_cbar(fig, caxB, axB[3])
    figspec.check_size(fig, outname)
    savef(fig, outname, dpi=600, exact=True)


# =================================================================== single panels (poster / talk)
# Default canvas per panel, in inches. Derived from each panel's true footprint inside the full-page
# figure (a data column is ~2.57 in wide, the square panels ~2.27 in) plus room for its labels, so a
# standalone panel keeps the manuscript figure's proportions. Every save is bbox_inches="tight", so
# these numbers set the aspect and the breathing room -- the final crop follows the artists.
PANEL_SIZE = {"A": (9.6, 2.9), "B": (9.0, 3.4), "C": (8.8, 2.9),
              "D": (3.5, 3.3), "E": (3.5, 3.3), "F": (4.2, 3.3)}
PANELS = tuple(sorted(PANEL_SIZE))
# panel-letter offsets re-tuned for a standalone canvas: the build_main() values are calibrated for
# the 4x4 grid and would put the letter off the edge of a single panel
PANEL_LETTER_DX = {"A": -0.10, "B": -0.10, "C": -0.055, "D": -0.20, "E": -0.20, "F": -0.20}


def build_panel(which, tags=True, letters=False, size=None, outname=None, dpi=300):
    """Render ONE panel of the merged figure as its own file.

    The panel is drawn by the SAME draw_* function build_main() uses, so it is the same plot as in
    the manuscript figure -- only the canvas, the panel letter and (in poster mode) the background
    differ. Two deliberate departures from the merged layout:
      * Panel B gains per-predictor TITLES. In the merged figure its three surfaces are identified
        by the Panel A column headings directly above them; standing alone they would be unlabelled.
      * Panel letters default OFF -- a lone poster panel is not part of a lettered composite.
    """
    global XLAB_RATIO
    XLAB_RATIO = RATIO_LABEL_TAGS if tags else RATIO_LABEL_PLAIN
    which = which.upper()
    if which not in PANEL_SIZE:
        raise ValueError("unknown panel {!r}; choose from {}".format(which, list(PANELS)))
    fig = plt.figure(figsize=tuple(size or PANEL_SIZE[which]))
    dx = PANEL_LETTER_DX[which]

    if which == "A":
        gs = GridSpec(1, 3, figure=fig, wspace=0.26,
                      left=0.075, right=0.985, top=0.90, bottom=0.20)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        draw_winning_bins(axes, put_letter=False)
        if letters:
            letter(axes[0], "a", dx=dx, dy=1.05)

    elif which == "B":
        gs = GridSpec(1, 3, figure=fig, wspace=0.26,
                      left=0.075, right=0.87, top=0.90, bottom=0.22)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cf = None
        for i, (ax, m) in enumerate(zip(axes, PRED)):
            cf = draw_contour(ax, base[m].to_numpy(float), SHARED_VLIM, title=m, cbar=False,
                              square=True, show_xlabel=(i == 1), show_ylabel=(i == 0),
                              xlim_full=tags, show_yticklabels=_show_yticks(i))
        cax = fig.add_axes([0.9, 0.25, 0.013, 0.5])           # placeholder; matched below
        _add_cbar(fig, cf, *SHARED_VLIM, "Success rate (%)", cax=cax)
        if tags:
            for ax in axes:
                apply_ratio_tags(ax)
        if letters:
            letter(axes[0], "b", dx=dx, dy=1.05)
        match_cbar(fig, cax, axes[2])

    elif which == "C":
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.13)
        draw_success_by_class(ax, put_letter=False)
        if letters:
            letter(ax, "c", dx=dx, dy=1.10)

    elif which == "D":
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.14)
        draw_gradient_pholo(ax, put_letter=False)
        if letters:
            letter(ax, "d", dx=dx, dy=1.02)

    elif which == "E":
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.17)
        draw_gradient_ratio(ax, put_letter=False)
        if tags:
            apply_ratio_tags(ax)
        if letters:
            letter(ax, "e", dx=dx, dy=1.02)

    else:                                                      # F
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.15, right=0.82, top=0.92, bottom=0.17)
        cf = draw_contour(ax, base.consensus.to_numpy(float), SHARED_VLIM,
                          title="Consensus detectability", cbar=False, square=True, xlim_full=tags)
        cax = fig.add_axes([0.86, 0.25, 0.013, 0.5])          # placeholder; matched below
        _add_cbar(fig, cf, *SHARED_VLIM, "Consensus success (%)", cax=cax)
        if tags:
            apply_ratio_tags(ax)
        if letters:
            letter(ax, "f", dx=dx, dy=1.02)
        match_cbar(fig, cax, ax)

    savef(fig, outname or ("fig6_panel" + which), dpi=dpi)




# ============================================================ per-method exports (poster assembly)
# Panels A and B of the manuscript figure carry three columns (the three predictors). For a poster
# the same two readouts are useful for FIVE methods -- the predictors, their consensus, and docking
# on its own denominator. These builders emit each cell as its own file so it can be arranged
# freely (PowerPoint, Illustrator, ...), while guaranteeing the thing that makes such an assembly
# legitimate: identical scales. Every bins plot is fixed to 0-100%, and every contour shares the
# vlim / cmap / levels / axis limits decided by method_scale().
BINS_SIZE = (3.6, 2.9)          # one winning-p bar plot
CONTOUR_SIZE = (4.2, 3.3)       # one contour + its colorbar
CONTOUR_SIZE_NOCBAR = (3.5, 3.3)


def set_ratio_mode(tags):
    """Select the evidence-axis label. The directional tags own the direction, so in tag mode the
    label drops to a bare 'Evidence ratio (log2)'. build_main() does this for the merged figure;
    every standalone builder has to do it too or it inherits whatever ran last."""
    global XLAB_RATIO
    XLAB_RATIO = RATIO_LABEL_TAGS if tags else RATIO_LABEL_PLAIN


def row_part_labels(i, n):
    """(show_ylabel, show_yticklabels, show_xlabel) for column i of an n-column row."""
    if not AS_ROW_PARTS:
        return True, _show_yticks(0), True
    return i == 0, _show_yticks(i), i == n // 2


def row_matched_size(kind, height):
    """Canvas whose axes width equals one column of the corresponding row."""
    return (_ROW_PANEL[kind] / _AXFRAC[kind], height)


def resolve_method(name):
    key = METHOD_ALIAS.get(str(name).strip().lower())
    if key is None:
        raise ValueError("unknown method {!r}; choose from {}".format(name, list(METHOD_ALIAS)))
    return key


def build_bins(method, tags=True, size=None, outname=None, dpi=300, pos=None, npos=1):
    """One method's winning-probability bin plot, on its own canvas (Panel A content)."""
    m = resolve_method(method)
    set_ratio_mode(tags)
    default = row_matched_size("bins", BINS_SIZE[1]) if MATCH_ROW else BINS_SIZE
    fig = plt.figure(figsize=tuple(size or default))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.19)
    ax = fig.add_subplot(111)
    show_yl, show_yt, show_xl = row_part_labels(0 if pos is None else pos, npos)
    draw_winning_bins([ax], put_letter=False, methods=[m])
    if not show_yl:
        ax.set_ylabel("")
    if not show_yt:
        ax.tick_params(labelleft=False)
    if not show_xl:
        ax.set_xlabel("")
    savef(fig, outname or ("fig6_bins_" + m.lower().replace("-", "")), dpi=dpi)


def build_contour(method, tags=True, size=None, outname=None, dpi=300, cbar=True,
                  pos=None, npos=1):
    """One method's detectability surface, on its own canvas (Panel B / F content)."""
    m = resolve_method(method)
    set_ratio_mode(tags)
    df, scol, n_m = method_frame(m)
    vlim, cmap = method_scale(m)
    if MATCH_ROW:
        kind = "contour" if cbar else "contour_nocbar"
        default = row_matched_size(kind, (CONTOUR_SIZE if cbar else CONTOUR_SIZE_NOCBAR)[1])
    else:
        default = CONTOUR_SIZE if cbar else CONTOUR_SIZE_NOCBAR
    fig = plt.figure(figsize=tuple(size or default))
    if cbar:
        fig.subplots_adjust(left=0.15, right=0.82, top=0.90, bottom=0.17)
    else:
        fig.subplots_adjust(left=0.18, right=0.97, top=0.90, bottom=0.17)
    ax = fig.add_subplot(111)
    show_yl, show_yt, show_xl = row_part_labels(0 if pos is None else pos, npos)
    cf = draw_contour(ax, df[scol].to_numpy(float), vlim, cmap=cmap,
                      title=METHOD_LABEL.get(m, m), xv=df.log2_ratio.to_numpy(float),
                      yv=df.pocket_p_holo.to_numpy(float), cbar=False, square=True,
                      xlim_full=tags, note=f"n = {n_m:,}", show_yticklabels=show_yt,
                      show_ylabel=show_yl, show_xlabel=show_xl)
    if cbar:
        cax = fig.add_axes([0.86, 0.25, 0.013, 0.5])          # placeholder; matched below
        _add_cbar(fig, cf, *vlim, "Success rate (%)", cax=cax)
    if tags:
        apply_ratio_tags(ax)
    if cbar:
        match_cbar(fig, cax, ax)
    savef(fig, outname or ("fig6_contour_" + m.lower().replace("-", "")), dpi=dpi)


def _row_width(n, panel_w, axfrac):
    """Canvas width that keeps each column `panel_w` inches wide at the current WSPACE. Shrinking
    the gap therefore narrows the whole block instead of fattening the panels -- which is the point
    when the saved width is going to poster margin."""
    return panel_w * (n + (n - 1) * WSPACE) / axfrac


def build_bins_row(methods=None, tags=True, size=None, outname=None, dpi=300):
    """All five winning-p bin plots as ONE figure -- the alignment done for you."""
    ms = [resolve_method(m) for m in (methods or METHOD_ORDER)]
    set_ratio_mode(tags)
    w, h = size or (_row_width(len(ms), 2.3839, 0.935), 3.0)
    fig = plt.figure(figsize=(w, h))
    gs = GridSpec(1, len(ms), figure=fig, wspace=WSPACE,
                  left=0.055, right=0.99, top=0.90, bottom=0.19)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(ms))]
    draw_winning_bins(axes, put_letter=False, methods=ms)
    savef(fig, outname or "fig6_bins_row", dpi=dpi)


def build_contour_row(methods=None, tags=True, size=None, outname=None, dpi=300, cbar=True):
    """All five detectability surfaces as ONE figure. Methods sharing a scale share a colorbar; a
    method on its own scale (docking under --docking_scale own) gets its own, so the row can never
    imply a comparability that is not there."""
    ms = [resolve_method(m) for m in (methods or METHOD_ORDER)]
    set_ratio_mode(tags)
    # one colorbar per distinct scale -- reserve the right margin BEFORE laying the row out, or a
    # second bar has nowhere to go and lands on top of the last panel
    scales = []
    for m in ms:
        if method_scale(m) not in scales:
            scales.append(method_scale(m))
    right = 0.99 if not cbar else 0.88 - 0.07 * (len(scales) - 1)
    axfrac = right - 0.05
    w, h = size or (_row_width(len(ms), 2.0407, axfrac), 3.5)
    fig = plt.figure(figsize=(w, h))
    gs = GridSpec(1, len(ms), figure=fig, wspace=WSPACE,
                  left=0.05, right=right, top=0.90, bottom=0.22)
    axes, cfs = [], {}
    mid = len(ms) // 2
    for i, m in enumerate(ms):
        ax = fig.add_subplot(gs[0, i])
        df, scol, n_m = method_frame(m)
        vlim, cmap = method_scale(m)
        cfs[m] = draw_contour(ax, df[scol].to_numpy(float), vlim, cmap=cmap,
                              title=METHOD_LABEL.get(m, m), xv=df.log2_ratio.to_numpy(float),
                              yv=df.pocket_p_holo.to_numpy(float), cbar=False, square=True,
                              show_xlabel=(i == mid), show_ylabel=(i == 0), xlim_full=tags,
                              note=f"n = {n_m:,}", show_yticklabels=_show_yticks(i))
        if tags:
            apply_ratio_tags(ax)
        axes.append(ax)

    # One colorbar per distinct scale. They all sit to the right of the WHOLE row, stacked
    # outward -- anchoring a shared bar beside its own last member would drop it on the panel after
    # it whenever that panel is on a different scale.
    groups = {}
    for m, ax in zip(ms, axes):
        groups.setdefault(method_scale(m), []).append(m)
    if not cbar:
        groups = {}
    fig.canvas.draw()
    last = axes[-1].get_position()
    for k, ((vlim, cmap), members) in enumerate(groups.items()):
        # only label it 'Dockability' when the bar serves docking ALONE; a bar shared with the
        # detectors is a success-rate bar
        if all(mm == "docking" for mm in members):
            lab = "Dockability (%)"
        elif "docking" in members:      # one bar spanning both readouts: name both
            lab = "Detection / dockability (%)"
        else:
            lab = "Success rate (%)"
        cax = fig.add_axes([last.x1 + 0.014 + 0.062 * k, last.y0, 0.013, last.height])
        _add_cbar(fig, cfs[members[-1]], vlim[0], vlim[1], lab, cax=cax)
    savef(fig, outname or "fig6_contour_row", dpi=dpi)


def build_cbar(methods=None, tags=True, size=None, outname=None, dpi=300, height=3.5):
    """The row's colorbar(s) on their own canvas, for placing beside a --no_cbar row by hand.

    Reproduces the banded bar exactly: same vlim, same 0.05 levels, same 10-point ticks, same
    clockwise label -- built from a BoundaryNorm rather than from a contour set, so it needs no
    data and cannot drift from what draw_contour() paints."""
    ms = [resolve_method(m) for m in (methods or METHOD_ORDER)]
    groups = {}
    for m in ms:
        groups.setdefault(method_scale(m), []).append(m)
    n = len(groups)
    fig = plt.figure(figsize=tuple(size or (0.62 * n + 0.35, height)))
    for k, ((vlim, cmap), members) in enumerate(groups.items()):
        vmin, vmax = vlim
        levels = np.round(np.arange(vmin, vmax + 1e-9, 0.05), 3)
        sm = plt.cm.ScalarMappable(norm=BoundaryNorm(levels, plt.get_cmap(cmap).N), cmap=cmap)
        if all(mm == "docking" for mm in members):
            lab = "Dockability (%)"
        elif "docking" in members:
            lab = "Detection / dockability (%)"
        else:
            lab = "Success rate (%)"
        cax = fig.add_axes([0.06 + k / n, 0.06, 0.16 / n, 0.88])
        _add_cbar(fig, sm, vmin, vmax, lab, cax=cax)
    savef(fig, outname or "fig6_contour_cbar", dpi=dpi)


# =================================================================== CLI
def main():
    p = argparse.ArgumentParser(
        description="Build the merged 6-panel Figure 6, or single panels of it for a poster/talk.")
    p.add_argument("--panels", default="all",
                   help="'all' (default) = the merged manuscript figure. Or a comma-separated list "
                        "of: " + ",".join(PANELS) + " (a panel of the manuscript figure); "
                        "bins:<method> / contour:<method> (one method's Panel A / Panel B cell as "
                        "its own file, method in " + ",".join(METHOD_ORDER) + " or 'all'); "
                        "bins:row / contour:row (all five methods pre-aligned in one figure); "
                        "cbar (the shared colorbar alone, to pair with a --no_cbar row).")
    p.add_argument("--docking_scale", choices=("shared", "own"), default="shared",
                   help="colour scale for the docking surface: 'shared' (default) puts it on the "
                        "predictors' viridis 0.20-0.80 so a row of five is directly comparable; "
                        "'own' keeps the manuscript's magma 0.30-0.90, which signals that docking "
                        "is a different readout on a different denominator.")
    p.add_argument("--contour_vlim", default=None, metavar="LO,HI",
                   help="shared contour colour range as fractions, default 0.20,0.80 (the "
                        "manuscript scale). Widening trades colour contrast for less clipping: "
                        "0.15,0.85 clips 1.3%% of cells vs 5.4%%, keeping 96%% of the bar in use; "
                        "0.00,1.00 clips nothing but leaves a third of the bar unused.")
    p.add_argument("--ytick_labels", choices=("first", "all", "none"), default="first",
                   help="y tick NUMBERS across a row (tick marks always stay): 'first' (default) "
                        "puts them on the leftmost column only, since the columns share a scale; "
                        "'all' repeats them; 'none' suppresses them, for exporting the 2nd..nth "
                        "panel of a row you are assembling by hand.")
    p.add_argument("--wspace", type=float, default=0.26, metavar="F",
                   help="horizontal gap between row columns as a fraction of one column's width "
                        "(default 0.26). The canvas narrows to match, so the saved width becomes "
                        "poster margin. The gap is only ~17%% of the block, so a 30%% cut (0.18) "
                        "buys ~5%% total width; 0.05 buys ~14%%.")
    p.add_argument("--overall", choices=("none", "lines", "bars"), default="none",
                   help="add each method's OVERALL rate to its bins plot, folding Panel C's "
                        "content in: 'lines' = dashed per-class reference lines with right-edge "
                        "values; 'bars' = a sixth hatched 'overall' group behind a divider. "
                        "NEITHER widens the canvas -- the extra x range is absorbed by the "
                        "existing axes, so bars just render ~17%% narrower. Default none.")
    p.add_argument("--pool_bins", action="store_true",
                   help="collapse the 5 winning-probability bins to 3 (very low+low / moderate / "
                        "high+very high). Buys the x-label room that an 'overall' group costs, at "
                        "the price of gradient resolution.")
    p.add_argument("--xtick_style", choices=("full", "abbrev"), default="full",
                   help="bin labels: 'full' (very low ... very high) or 'abbrev' (v. low, mod., "
                        "v. high), which frees horizontal room without shrinking the type.")
    p.add_argument("--gap_y", type=float, default=None, metavar="Y",
                   help="lock every '+N pp' gap annotation to this y value instead of floating "
                        "above each method's own tallest bar (which leaves them at differing "
                        "heights across a row). Try 76.")
    p.add_argument("--svg_text", choices=("path", "none"), default=None,
                   help="how SVG carries text. 'path' outlines the glyphs so the file renders "
                        "identically anywhere (no font dependency, no metric-drift overlaps) but "
                        "is no longer text-editable; 'none' keeps live text that requires the "
                        "viewer to have the fonts. Default: 'path' for poster output, 'none' for "
                        "the manuscript build.")
    p.add_argument("--no_cbar", action="store_true",
                   help="omit the colorbar from contour exports AND from contour:row. Pair with "
                        "the 'cbar' panel token, which emits the bar on its own canvas, to place "
                        "it by hand.")
    p.add_argument("--transparent_background", action="store_true",
                   help="poster mode: transparent outside the axes, white inside the plot area.")
    p.add_argument("--poster_output_dir", action="store_true",
                   help="write to the configured poster/talk plots directory (the public repo's "
                        "common/paths.ini 'presentation_dir').")
    p.add_argument("--out_dir", default=None, help="explicit output directory (wins over the above).")
    p.add_argument("--out_basename", default=None,
                   help="output file name without extension. Only honoured when a single figure is "
                        "requested (the merged figure, or exactly one panel).")
    p.add_argument("--formats", default=None,
                   help="comma-separated output formats. Default png,pdf,svg for the merged "
                        "manuscript figure and png,svg for poster output (panels or transparent).")
    p.add_argument("--font_scale", type=float, default=1.0,
                   help="uniform multiplier on every point size, e.g. 1.25 for poster viewing.")
    p.add_argument("--as_row_parts", action="store_true",
                   help="label a set of individual bins:/contour: exports as if they were the "
                        "consecutive columns of one row: y label and y tick numbers on the first "
                        "only, x label under the middle one. Use when assembling a row by hand, "
                        "so the result does not carry five copies of every axis label.")
    p.add_argument("--match_row", action="store_true",
                   help="size individual bins:/contour: exports so their panel matches one column "
                        "of the corresponding row. Without this, an individual export's panel is "
                        "wider, so its 11 pt title reads ~20-30%% smaller than a row column's when "
                        "both are placed at the same width. Use it when mixing the two on a poster.")
    p.add_argument("--panel_size", default=None, metavar="W,H",
                   help="canvas in inches for standalone panels, overriding the per-panel default.")
    p.add_argument("--panel_letters", action="store_true",
                   help="draw the A-F panel letter on standalone panels (off by default).")
    p.add_argument("--no_tags", action="store_true",
                   help="plain numeric evidence axis instead of the directional More apo/More holo tags.")
    p.add_argument("--layout", choices=("v3", "v4"), default="v3",
                   help="which merged figure to build: 'v3' (default) is the published 6-panel "
                        "Figure 6; 'v4' is manuscript v4 Figure 3 -- Panel C folded into the bins "
                        "panel, docking as its own panel, the two consensus/docking contours moved "
                        "to the SI. Ignored when --panels selects individual panels.")
    p.add_argument("--dpi", type=int, default=300)
    a = p.parse_args()

    global OUT_DIR, FORMATS, TRANSPARENT_BACKGROUND, FONT_SCALE, DOCKING_SCALE, SHARED_VLIM
    global YTICK_LABELS, WSPACE, OVERALL, GAP_Y, SVG_TEXT, POOL_BINS, XTICK_STYLE, MATCH_ROW
    global AS_ROW_PARTS
    DOCKING_SCALE = a.docking_scale
    YTICK_LABELS = a.ytick_labels
    WSPACE = a.wspace
    OVERALL = a.overall
    GAP_Y = a.gap_y
    POOL_BINS = a.pool_bins
    XTICK_STYLE = a.xtick_style
    MATCH_ROW = a.match_row
    AS_ROW_PARTS = a.as_row_parts
    if a.contour_vlim:
        try:
            lo, hi = (float(v) for v in a.contour_vlim.split(","))
        except ValueError:
            p.error("--contour_vlim takes 'LO,HI' as fractions, e.g. 0.15,0.85")
        if not 0.0 <= lo < hi <= 1.0:
            p.error("--contour_vlim needs 0 <= LO < HI <= 1")
        SHARED_VLIM = (lo, hi)

    # jobs = the render list. Each is ('panel'|'bins'|'contour'|'bins_row'|'contour_row', arg).
    jobs = []
    if a.panels.strip().lower() != "all":
        for tok in (t.strip() for t in a.panels.split(",") if t.strip()):
            kind, _, arg = tok.partition(":")
            kind = kind.strip().lower()
            arg = arg.strip()
            try:
                if kind == "cbar" and not arg:               # the shared colorbar, standalone
                    jobs.append(("cbar", None))
                elif not arg:                                # a bare manuscript panel letter
                    if kind.upper() not in PANEL_SIZE:
                        raise ValueError("unknown panel {!r}".format(tok))
                    jobs.append(("panel", kind.upper()))
                elif kind not in ("bins", "contour"):
                    raise ValueError("unknown token {!r}".format(tok))
                elif arg.lower() == "row":
                    jobs.append((kind + "_row", None))
                elif arg.lower() == "all":
                    jobs.extend((kind, m) for m in METHOD_ORDER)
                else:
                    jobs.append((kind, resolve_method(arg)))
            except ValueError as exc:
                p.error("{}. Expected one of {}, 'cbar', or bins:<m> / contour:<m> with "
                        "<m> in {} (or 'all' / 'row'), or 'all'."
                        .format(exc, list(PANELS), METHOD_ORDER))

    if a.out_dir:
        OUT_DIR = a.out_dir
    elif a.poster_output_dir:
        OUT_DIR = poster_dir()
    elif a.layout == "v4" and a.panels.strip().lower() == "all":
        # the v4 figure set lives in one folder (cfg.MANUSCRIPT_FIG_DIR); without this the
        # documented build command wrote the v3 figure's directory
        OUT_DIR = str(FIG_DIR)
    TRANSPARENT_BACKGROUND = a.transparent_background
    FONT_SCALE = a.font_scale
    if a.formats:
        FORMATS = tuple(f.strip().lstrip(".").lower() for f in a.formats.split(",") if f.strip())
    elif jobs or a.transparent_background:
        FORMATS = ("png", "svg")             # poster output: raster + vector, no pdf

    size = None
    if a.panel_size:
        try:
            w, h = (float(v) for v in a.panel_size.split(","))
        except ValueError:
            p.error("--panel_size takes 'W,H' in inches, e.g. 4.5,3.2")
        size = (w, h)

    # poster SVGs outline their text by default; the manuscript build keeps it live
    SVG_TEXT = a.svg_text or ("path" if (jobs or a.transparent_background) else "none")
    mpl.rcParams["svg.fonttype"] = SVG_TEXT

    tags = not a.no_tags
    if not jobs:
        # FINAL manuscript figure: directional-tag evidence axis (all three Panel B contours),
        # "Evidence ratio (log2)".
        if a.layout == "v4":
            build_main_v4(tags=tags, outname=a.out_basename or "Figure3_detection_and_docking")
        else:
            build_main(tags=tags, outname=a.out_basename or "main_lbs_docking_combined")
    else:
        only = a.out_basename if len(jobs) == 1 else None
        # when the per-method exports are to be assembled into a row, each needs to know which
        # column it will occupy; index within this run's own sequence of that kind
        order = {}
        for kind, arg in jobs:
            if kind in ("bins", "contour"):
                order.setdefault(kind, []).append(arg)
        for kind, arg in jobs:
            if kind == "panel":
                build_panel(arg, tags=tags, letters=a.panel_letters, size=size, dpi=a.dpi,
                            outname=only)
            elif kind in ("bins", "contour"):
                seq = order[kind]
                fn = build_bins if kind == "bins" else build_contour
                kw = dict(tags=tags, size=size, dpi=a.dpi, outname=only,
                          pos=seq.index(arg), npos=len(seq))
                if kind == "contour":
                    kw["cbar"] = not a.no_cbar
                fn(arg, **kw)
            elif kind == "bins_row":
                build_bins_row(tags=tags, size=size, dpi=a.dpi, outname=only)
            elif kind == "cbar":
                build_cbar(tags=tags, size=size, dpi=a.dpi, outname=only)
            else:
                build_contour_row(tags=tags, size=size, dpi=a.dpi, outname=only,
                                  cbar=not a.no_cbar)
    print("OUT:", OUT_DIR)


if __name__ == "__main__":
    main()
