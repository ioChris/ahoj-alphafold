"""Figure 1 data panels — natcomm_v4_rev1.

Builds the two DATA panels (c = attrition flow, d = class composition) as
standalone, correctly-sized vector files for assembly against the artwork
panels (a = entry anatomy, b = classification schematic, from the graphical
abstract), plus a combined reference mockup of the whole figure.

    python make_figure1_panels.py

Outputs (same directory):
    panel_c_flow.{svg,pdf,png}
    panel_d_composition.{svg,pdf,png}
    figure1_reference_mockup.{svg,png}

Design contract
---------------
* Single typeface: Helvetica Neue (weights 100-900, roman + italic, verified
  after the 2026-08 font re-install). Italics are available and safe to use.
  BUT the family has no U+2192 arrow glyph at any weight or style, so every
  arrow here is drawn as a patch, never typeset.
* Type floor 6 pt, axis titles 7 pt, panel letters 8 pt bold lowercase
  (Nature: "8-pt bold, upright, lowercase a, b, c").
* SVG is written with svg.fonttype='none', so text stays editable text.
* Every number is sourced — see NUMBER SOURCES at the bottom. Nothing is
  transcribed from prose; re-run the checks there before submission.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import os

MM = 1 / 25.4

# ----------------------------------------------------------------- type spec
FONT = "Helvetica Neue"
FS_LETTER = 8      # panel letter, bold
FS_TITLE = 7       # axis titles
FS_BODY = 6.5      # bar values, row labels
FS_TICK = 6        # tick labels
FS_NOTE = 6        # annotations -- 6 pt is the floor, do not go below

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": FS_BODY,
    "axes.linewidth": 0.6,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,          # embed as TrueType, keep text selectable
})

APO, HOLO, AF = "#1f77b4", "#ff7f0e", "#2ca02c"
GREY, LGREY, EDGE = "#5a5a5a", "#d9d9d9", "#9aa5b1"
APOCUT = "#a01b895"[:7] or "#9c27b0"
APOCUT = "#8e1f6b"                # colour that links the two apo-reference cuts

# ------------------------------------------------------------------- numbers
# (kept, removed_before_this_bar, label, is_apo_reference_cut)
FLOW = [
    (516794, None, None, False),
    (464869, 51925, "no comparator structures", True),
    (438143, 26726, "no AlphaFold model", False),
    (435314, 2829, "no classification", False),
    (238738, 196576, "no apo state", True),
    (237922, 816, "NMR / low resolution", False),
]
APO_CUT_TOTAL = 50598 + 196576        # 247,174 -- pockets with no apo reference
DB_TOTAL = 516794
ANALYSIS = dict(pockets=237922, unps=12255)
UBS100_SITES = 105713

LADDER = [  # label, criterion, apo_n, holo_n, apo_pct, holo_pct
    ("Unclustered", "individual pockets", 90246, 146615, 38.1, 61.9),
    ("UBS  ST 100%", "identical residue sets", 41555, 63791, 39.4, 60.6),
    ("UBS  ST 70%", "Jaccard ≥ 0.70", 21479, 22982, 48.3, 51.7),
    ("UBS  ST 50%", "Jaccard ≥ 0.50", 17257, 16202, 51.6, 48.4),
    ("UBS  ST 30%", "Jaccard ≥ 0.30", 15249, 13513, 53.0, 47.0),
]

MINUS = "−"


def _arrow(ax, p0, p1, lw=0.9, color=GREY, ms=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, shrinkA=1, shrinkB=1, zorder=4))


# =============================================================== panel c
def draw_panel_c(ax):
    """Attrition from the database to the analysis set, plus the UBS merge."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ymax, rowh = 0.912, 0.100
    maxc = FLOW[0][0]

    for i, (kept, removed, label, is_apo) in enumerate(FLOW):
        y = ymax - i * rowh
        if removed is not None:
            col = APOCUT if is_apo else GREY
            ax.text(0.995, y + rowh * 0.50, f"{MINUS}{removed:,}   {label}",
                    fontsize=FS_NOTE, ha="right", va="center", color=col,
                    fontweight="bold" if is_apo else "normal")
        wfrac = 0.40 * kept / maxc + 0.08
        endpoint = i in (0, len(FLOW) - 1)
        ax.add_patch(Rectangle((0.045, y - rowh * 0.26), wfrac, rowh * 0.42,
                               fc="#c6ced6" if endpoint else LGREY, ec=EDGE, lw=0.5))
        ax.text(0.045 + wfrac + 0.018, y - rowh * 0.05, f"{kept:,}",
                fontsize=FS_BODY, va="center",
                fontweight="bold" if endpoint else "normal")

    ax.text(0.045, ymax + rowh * 0.60, "AHoJ-DB query pockets",
            fontsize=FS_NOTE, va="center", color=GREY)

    yend = ymax - (len(FLOW) - 1) * rowh
    ax.text(0.045, yend - rowh * 0.85,
            f"analysis set   {ANALYSIS['pockets']:,} pockets  ·  "
            f"{ANALYSIS['unps']:,} UniProts",
            fontsize=FS_BODY, va="center", fontweight="bold")

    # the two apo-reference cuts, linked by colour rather than by a bracket
    pct = 100 * APO_CUT_TOTAL / DB_TOTAL
    ax.add_patch(Rectangle((0.045, 0.268), 0.020, 0.020, fc=APOCUT, ec="none"))
    ax.text(0.077, 0.278, f"the two apo-reference cuts remove {APO_CUT_TOTAL:,}",
            fontsize=FS_NOTE, va="center", color=APOCUT)
    ax.text(0.077, 0.232, f"pockets in total — {pct:.0f}% of the database",
            fontsize=FS_NOTE, va="center", color=APOCUT)

    # UBS merge
    ax.plot([0.045, 0.995], [0.192, 0.192], color=EDGE, lw=0.6, ls=(0, (3, 2)))
    ax.text(0.045, 0.152, "redundancy reduction — unified binding sites",
            fontsize=FS_BODY, va="center", fontweight="bold")
    cy = 0.082
    for k, dx in enumerate((0.0, 0.024, 0.048)):
        ax.add_patch(Circle((0.075 + dx, cy), 0.024, fc=HOLO if k % 2 else APO,
                            alpha=0.45, ec="none", zorder=2))
    _arrow(ax, (0.170, cy), (0.230, cy), lw=0.8)
    ax.add_patch(Circle((0.272, cy), 0.028, fc="#8e9aa6", alpha=0.75, ec="none"))
    ax.text(0.345, cy, "pockets with identical\nresidue sets merge", fontsize=FS_NOTE,
            va="center", color=GREY, linespacing=1.35)
    # count transition on its own line, arrows drawn as shapes (Helvetica Neue
    # has no U+2192 glyph)
    ax.text(0.045, 0.018, f"{ANALYSIS['pockets']:,} pockets", fontsize=FS_BODY,
            va="center")
    _arrow(ax, (0.330, 0.018), (0.400, 0.018), lw=0.8)
    ax.text(0.430, 0.018, f"{UBS100_SITES:,} sites", fontsize=FS_BODY, va="center",
            fontweight="bold")
    ax.text(0.680, 0.018, "at ST 100%", fontsize=FS_NOTE, va="center", color=GREY)


# =============================================================== panel d
def draw_panel_d(ax):
    """Class composition across redundancy thresholds."""
    y = np.arange(len(LADDER))[::-1]
    for i, (lab, crit, a, h, ap, hp) in enumerate(LADDER):
        yy = y[i]
        ax.barh(yy, ap, color=APO, height=0.62, zorder=3)
        ax.barh(yy, hp, left=ap, color=HOLO, height=0.62, zorder=3)
        ax.text(ap / 2, yy, f"{ap:.1f}", ha="center", va="center", fontsize=FS_BODY,
                color="white", fontweight="bold", zorder=4)
        ax.text(ap + hp / 2, yy, f"{hp:.1f}", ha="center", va="center",
                fontsize=FS_BODY, color="white", fontweight="bold", zorder=4)
        ax.text(103, yy, f"{a + h:,}", fontsize=FS_TICK, va="center", color=GREY)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}\n{c}" for l, c, *_ in LADDER], fontsize=FS_TICK,
                       linespacing=1.4)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("class composition (%)", fontsize=FS_TITLE, labelpad=2)
    ax.tick_params(labelsize=FS_TICK, length=2, width=0.5, pad=2)
    ax.axvline(50, color="white", lw=0.6, zorder=5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    hi = y[1]
    ax.add_patch(Rectangle((-0.5, hi - 0.46), 101, 0.92, fill=False, ec="black",
                           lw=0.9, zorder=6, clip_on=False))
    ax.get_yticklabels()[list(y).index(hi)].set_fontweight("bold")
    ax.text(0, -0.20, "boxed row = the unit used throughout the main text",
            fontsize=FS_NOTE, va="top", transform=ax.transAxes)
    ax.text(0, len(LADDER) - 0.30, "AF-Apo", color=APO, fontsize=FS_BODY,
            fontweight="bold")
    ax.text(27, len(LADDER) - 0.30, "AF-Holo", color=HOLO, fontsize=FS_BODY,
            fontweight="bold")
    ax.text(103, len(LADDER) - 0.30, "sites", fontsize=FS_TICK, color=GREY)


# =============================================================== exports
def export_panel_c(path, w_mm=80, h_mm=100):
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    fig.patch.set_facecolor("white")
    draw_panel_c(fig.add_axes([0.02, 0.02, 0.96, 0.945]))
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)


def export_panel_d(path, w_mm=88, h_mm=58):
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    fig.patch.set_facecolor("white")
    draw_panel_d(fig.add_axes([0.29, 0.20, 0.58, 0.72]))
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)


def export_reference(path, w_mm=180, h_mm=178):
    """Whole-figure reference: a/b as labelled placeholders, c/d for real."""
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    fig.patch.set_facecolor("white")

    for rect, name, note in (
        ([0.050, 0.615, 0.295, 0.330], "a",
         "entry anatomy\n\nquery holo pocket\n+ apo states / holo states\n+ AlphaFold model\n\n"
         "AHoJ-DB scale strip\n516,794 pockets · 121,227 PDB\n37,661 ligands"),
        ([0.395, 0.615, 0.585, 0.330], "b",
         "classification schematic  (from the graphical abstract)\n\n"
         "pocket  ·  4 metrics  ·  apo/holo KDE densities  ·  P(apo), P(holo)  ·  class\n"
         "(chained left to right; draw the arrows as shapes, not as glyphs)\n\n"
         "+ one real apo/holo distribution with the model marker,\nto define density and separation"),
    ):
        ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, fc="#fafbfc", ec=EDGE, lw=0.7,
                               ls=(0, (4, 3))))
        ax.text(0.5, 0.55, note, ha="center", va="center", fontsize=FS_NOTE,
                color=GREY, linespacing=1.7)
        ax.text(0.5, 0.06, "ARTWORK — assemble by hand", ha="center", va="center",
                fontsize=FS_NOTE, color=EDGE, fontweight="bold")

    cax = fig.add_axes([0.050, 0.030, 0.330, 0.520]); draw_panel_c(cax)
    draw_panel_d(fig.add_axes([0.610, 0.140, 0.300, 0.330]))

    for x, yy, s in ((0.010, 0.972, "a"), (0.355, 0.972, "b"),
                     (0.010, 0.568, "c"), (0.500, 0.568, "d")):
        fig.text(x, yy, s, fontsize=FS_LETTER, fontweight="bold", va="top")
    fig.text(0.610, 0.535, "classification outcome, and the effect of\nredundancy reduction",
             fontsize=FS_NOTE, color=GREY, va="top", linespacing=1.5)

    for ext in ("svg", "png"):
        fig.savefig(f"{path}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)


def export_template_fallback(path, w_mm=180, h_mm=178):
    """An empty assembly canvas emitted through matplotlib's SVG writer.

    Exists because a hand-written SVG carrying Inkscape-specific markup
    (sodipodi:namedview, hand-rolled guides, a non-numeric inkscape:version)
    crashed Inkscape 1.4.40 on open, while matplotlib's plain output loads
    fine. This file has NO Inkscape-specific attributes at all, so it cannot
    trip that path -- at the cost of carrying no layer structure.

    Slots and letters match figure1_template.svg exactly:
        panel c  X 4   Y 76  W 80  H 100
        panel d  X 88  Y 97  W 88  H 58
    """
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm); ax.set_ylim(h_mm, 0)      # y down, so mm match Inkscape
    ax.axis("off")

    slots = [("a", 4, 4, 52, 66, "entry anatomy  (artwork)"),
             ("b", 60, 4, 116, 66, "classification schematic  (artwork)"),
             ("c", 4, 76, 80, 100, "place panel_c_flow.svg"),
             ("d", 88, 97, 88, 58, "place panel_d_composition.svg")]
    for letter_, x, y, w, h, label in slots:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#c8d0d8", lw=0.25,
                               ls=(0, (5.6, 4))))
        ax.text(x + w / 2, y + h / 2, f"{letter_}   {label}", ha="center",
                va="center", fontsize=FS_NOTE, color="#aab4bd")
    for letter_, lx, ly in (("a", 1.5, 6.5), ("b", 57.5, 6.5),
                            ("c", 1.5, 78.5), ("d", 85.5, 78.5)):
        ax.text(lx, ly, letter_, fontsize=FS_LETTER, fontweight="bold", va="baseline")
    ax.text(88, 86, "classification outcome, and the effect", fontsize=FS_NOTE,
            color=GREY, va="baseline")
    ax.text(88, 89.2, "of redundancy reduction", fontsize=FS_NOTE, color=GREY,
            va="baseline")
    ax.text(4, 172, "Helvetica Neue only  ·  panel letter 8 pt bold lowercase  ·  "
                    "axis titles 7 pt  ·  floor 6 pt  ·  arrows as paths, not glyphs",
            fontsize=5.6, color="#8e9aa6", va="baseline")
    fig.savefig(f"{path}.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    export_panel_c(os.path.join(here, "panel_c_flow"))
    export_panel_d(os.path.join(here, "panel_d_composition"))
    export_reference(os.path.join(here, "figure1_reference_mockup"))
    assembly = os.path.join(os.path.dirname(here), "assembly")
    export_template_fallback(os.path.join(assembly, "figure1_template_fallback"))
    print("wrote panels to", here)
    print("wrote fallback template to", assembly)

# ---------------------------------------------------------- NUMBER SOURCES
# All from ahoj-alphafold `common.config` anchors; re-derive, never transcribe.
#
# 516,794 attempted / 51,925 discarded / 464,869 classified
#     CLASSIFICATION_MERGED_DIR/{Success,Discarded}_jobs.txt  (+ merged_manifest.json)
#     Error_jobs.txt and "File Not Found_jobs.txt" are EMPTY in both source runs.
#     Discard reasons: 50,598 "apo=0, holo=0, total=0" (all of them, verified) and
#     1,327 "Empty dataframe".
# 121,227 PDB entries / 37,661 ligands
#     parsed from the 516,794 attempted job names (pdb-chain-lig-index).
# 26,726 / 2,829 / 196,576 / 816 / 237,922 / 19,783 / 12,255
#     FILTERED_SURVIVORS_DIR/filter_report_summary.txt  (sequential order:
#     has_model -> pocket probs -> n_apo>=1 -> resolution; MIN_N_APO=1,
#     MAX_QUERY_RESOLUTION=20.0). The 816 = 688 with no resolution + 128 >20 A.
# 247,174 apo-reference total = 50,598 + 196,576. NOTE: valid ONLY in this
#     sequential order. Marginal n_apo==0 over all 464,869 is 207,916, and the
#     conditions overlap (9,659 no-model rows also have n_apo==0), so the steps
#     cannot be reordered without recomputing all of them.
# ladder apo/holo per threshold, and 105,713 ST_100 sites incl. 367 ties
#     ANALYSIS_UNIFIED_LEVELS_CSV, level=='ubs' (ties excluded from the split).
# 29,463 UniProts (panel a, artwork) -- QUERY_SUMMARIES_CLASS_CSV, a Jun-2025
#     snapshot covering 514,527 of 516,794 (99.6%). RECOMPUTE or drop.
