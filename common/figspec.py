# -*- coding: utf-8 -*-
"""
One figure specification for the whole manuscript: canvas sizes, type sizes, colours.

WHY THIS EXISTS
---------------
Two problems it fixes, both of which had already reached rendered figures:

1. **Post-hoc scaling.** Panels were rendered at whatever size their generator happened to use and
   then resized by the composer to make a row fit. Every panel then shrank or grew by a different
   factor, so nominally equal type ended up at different sizes in the same figure, and the panel
   letters visibly disagreed. Nothing here is ever rescaled after rendering: a panel is drawn at the
   millimetre size it will occupy on the page, and the composer only places it.

2. **Divergent colour.** `figure1_main_REV4` drew apo/holo in a muted pair while
   `figure2_main_REV1`, the ridgeline and the sankey used the tab10 pair. Those generators supply
   different ROWS OF THE SAME FIGURE, so manuscript Figure 2 carried two different blues for apo.

Import this rather than redefining either.

    from common.figspec import MM, W2, APO, HOLO, FS, apply_rc

USAGE
-----
Sizes are millimetres, because that is the unit the journal specifies and the unit a page has.
`MM` converts to inches for matplotlib:

    fig = plt.figure(figsize=(W2 * MM, 120 * MM))       # 180 mm wide, 120 mm tall

TYPE
----
Nature's own rule is the panel letter: 8 pt, bold, lowercase. The rest is a house scale inside the
usual 5-7 pt range, with a hard floor at 6 pt. Because panels are drawn at final size, a point is a
point -- these are the sizes that reach the page.
"""

MM = 1 / 25.4          # millimetres -> inches, for figsize

# ------------------------------------------------------------------ canvas, millimetres
W1 = 88.0              # single column
W1_5 = 120.0           # 1.5 column, allowed but rarely worth it
W2 = 180.0             # double column / full width
H_MAX = 185.0          # absolute maximum height for a display item
H_SAFE = 170.0         # working maximum: leaves the caption off the figure canvas

# ------------------------------------------------------------------ type, points
FS_LETTER = 8.0        # panel letter: bold, lowercase, upright. Nature's explicit rule.
FS_TITLE = 7.0         # panel titles and axis titles
FS_BODY = 6.5          # in-panel values, row labels, legend text
FS_TICK = 6.0          # tick labels
FS_NOTE = 6.0          # annotations, footnotes
FS_FLOOR = 6.0         # nothing smaller than this reaches the page

FS = dict(letter=FS_LETTER, title=FS_TITLE, body=FS_BODY, tick=FS_TICK, note=FS_NOTE)

# ------------------------------------------------------------------ colour
# The muted pair, which is what the detection figure and the classification grids already used.
# The tab10 pair it replaces was brighter and read as a different mapping in the reader's first
# panel -- the specific problem the Figure 1 rebuild called out.
APO = "#2C7FB8"
HOLO = "#F28E2B"
ALL = "#8a8f98"        # the neutral "All"/"Overall" series
MODEL = "#3B7A57"      # the AlphaFold model, where it appears as its own series

# pale tints of the same hues, for card fills and shaded bands
FILL_APO = "#E8F1F9"
FILL_HOLO = "#FDF1E3"
FILL_ALL = "#F4F6F8"
FILL_MODEL = "#EAF6EA"

INK = "#262626"        # primary text
MUTE = "#6E6E6E"       # secondary text and annotations
FAINT = "#ECECEC"      # grid lines
RULE = "#9AA5B1"       # light rules and box edges

# ------------------------------------------------------------------ shared wording
# Settled 2026-08-24: ONE name for this axis everywhere. The ridgeline said "Evidence balance"
# while the neighbouring panels of the same figure said "Evidence ratio".
RATIO_NAME = "Evidence ratio"
RATIO_LAB_LOG2 = r"Evidence ratio, $\log_2$"
RATIO_LAB_LOG10 = r"Evidence ratio, $\log_{10}$"
# n_apo / n_holo read better than "Exp-Apo"/"Exp-Holo" and need real subscripts
N_APO = r"$n_\mathrm{apo}$"
N_HOLO = r"$n_\mathrm{holo}$"
LAB_APO = "AF-Apo"
LAB_HOLO = "AF-Holo"
LAB_ALL = "Overall"    # was "All"; "Overall" reads as a summary rather than another bin
# The per-pocket probabilities are written P(apo) / P(holo) in the text; the ridgeline key said
# "apo probability" / "holo probability" (skeleton review 2026-09-10, item 6). One spelling everywhere.
P_APO = "P(apo)"
P_HOLO = "P(holo)"

FONT_STACK = ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"]


def apply_rc(rcparams=None, base=None):
    """Set the manuscript's rcParams. Call once, at import time, in every figure generator."""
    import matplotlib as mpl
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": base or FS_BODY,
        "axes.linewidth": 0.6,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_BODY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "legend.frameon": False,
        "svg.fonttype": "none",    # live text; the publisher wants editable type
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if rcparams:
        rc.update(rcparams)
    mpl.rcParams.update(rc)
    return rc


def letter(ax, s, dx=-0.14, dy=1.02, **kw):
    """Draw a panel letter in the house style: 8 pt, bold, lowercase, axes-relative."""
    return ax.text(dx, dy, s, transform=ax.transAxes, ha="left", va="bottom",
                   fontsize=FS_LETTER, fontweight="bold", **kw)


def check_size(fig, name="figure"):
    """Warn if a figure will not fit the page. Called at save time, not left to the eye."""
    w_mm, h_mm = (v / MM for v in fig.get_size_inches())
    msgs = []
    if w_mm > W2 + 0.5:
        msgs.append(f"width {w_mm:.1f} mm > {W2:.0f} mm")
    if h_mm > H_MAX + 0.5:
        msgs.append(f"height {h_mm:.1f} mm > {H_MAX:.0f} mm")
    if msgs:
        print(f"  ! {name}: " + "; ".join(msgs) + " -- will be downscaled by production, "
              "which shrinks every type size below spec")
    else:
        print(f"  {name}: {w_mm:.1f} x {h_mm:.1f} mm (fits)")
    return not msgs


def margins(fig_w_in, fig_h_in, left=0.0, right=0.0, top=0.0, bottom=0.0):
    """Millimetre margins -> the fractions `subplots_adjust` wants.

    Margins expressed as fractions are the reason panels lost their axis labels the moment the
    canvas was fixed at its true size: a `bottom=0.085` tuned on a 90 mm-tall canvas is 7.6 mm
    there and 3.9 mm on a 46 mm one, which is less than a tick row plus an axis title. Ink needs
    a fixed number of millimetres regardless of how tall the panel is, so specify it that way.

        fig.subplots_adjust(**margins(w_in, h_in, left=14, right=2, top=4, bottom=9))
    """
    w_mm, h_mm = fig_w_in / MM, fig_h_in / MM
    return dict(left=left / w_mm, right=1.0 - right / w_mm,
                top=1.0 - top / h_mm, bottom=bottom / h_mm)
