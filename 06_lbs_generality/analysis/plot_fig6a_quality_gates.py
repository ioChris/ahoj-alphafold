# -*- coding: utf-8 -*-
"""
Figures for the Fig 6A density / separation quality-gating analysis.
====================================================================

Consumes the tables written by `analyze_fig6a_quality_gates.py`.

**SI deliverable** (`SI_fig6a_density_separation_gates`), 3 panels:
  A  dD ladder -- how the divergence responds to WHERE the density / separation cut is placed.
     Density strengthens the trend at every cut; separation never does. Answers "is moderate+
     a cherry-pick?" (no: the ladder is flat from low+ to high+, and moderate+ is the project's
     own `confident_probs` constant).
  B  The apo/holo decomposition at the very-high-probability end vs the density cut: AF-Apo
     success falls monotonically while AF-Holo is nearly flat -- i.e. the gate acts through the
     apo arm. This is the panel that licenses the main-text sentence.
  C  Which metric actually carries the divergence: the apo/holo gap against each metric's OWN
     verbal bins. winning_p spans ~34 pp; separation ~13; density is FLAT. Density carries no
     divergence itself -- consistent with it being undirected support -- which is exactly why it
     works as an orthogonal FILTER rather than as a competing axis.

Internal reference figures (not for the SI):
  `fig6a_gates_panelA_grid`  -- the literal Panel A redrawn under each gate (4 gates x 3 predictors)
  `fig6a_gates_summary`      -- gap-vs-bin curves per gate + the dD forest + gate cost

Convention: SI figures live under `<project>/manuscript/figures/SI/` with an `SI_` prefix and a
DESCRIPTIVE name (no figure number -- the final `Figure Sx` is assigned at SI finalization).
`--manuscript_dir <dir>` copies the SI figure + caption there.

Inputs  : work_dir/lbs_generality/analysis_v1/tables/fig6a_gates_*.csv
Outputs : work_dir/lbs_generality/analysis_v1/plots/*.{png,pdf,svg}

Base env (pandas + numpy + matplotlib):
    python 06_lbs_generality/analysis/plot_fig6a_quality_gates.py \
        [--manuscript_dir "<...>/manuscript/figures/SI"]
"""
from __future__ import annotations

import argparse
import shutil
import sys as _sys
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb
import matplotlib.patheffects as pe

for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# manuscript palette (AF-Apo / AF-Holo / collective) + per-gate accents
APO, HOLO, ALLC = "#2C7FB8", "#F28E2B", "#8a8f98"
DENS_C, SEP_C, BOTH_C = "#1B7837", "#762A83", "#B8860B"
GATE_COL = {"all": "0.25", "dens_mod+": DENS_C, "sep_mod+": SEP_C, "both_mod+": BOTH_C,
            "dens_high+": DENS_C, "sep_high+": SEP_C, "both_high+": BOTH_C}
GATE_LAB = {"all": "ungated", "dens_mod+": "density mod+", "sep_mod+": "separation mod+",
            "both_mod+": "both mod+", "dens_high+": "density high+", "sep_high+": "separation high+",
            "both_high+": "both high+"}
GATES = ["all", "dens_mod+", "sep_mod+", "both_mod+"]
GATES_HIGH = ["all", "dens_high+", "sep_high+", "both_high+"]
PRED = ["P2Rank", "fpocket", "IF-SitePred"]
METHODS = PRED + ["consensus"]
LEVELS5 = ["very_low", "low", "moderate", "high", "very_high"]
RUNGS = ["low+", "moderate+", "high+", "very_high+"]
# shared by panels d, e and f: one baseline across the whole bottom row. The negative headroom seats
# the n= row (and, in f, the descriptor labels below the axis); the top clears f's contrast brackets.
ROW2_YLIM = (-7, 60)

# Winning-probability series drawn in panels a/b. Two families, selected by --wp_series:
#   CUMULATIVE (nested: very_high < high+ < moderate+) -- solid for the strictest, progressively
#     lighter/dashed as the subset widens. This is the default (very_high + moderate+ = 4 lines).
#   DELIMITED (bin_*: each pocket in exactly one) -- 5 per class = 10 lines, distinguished by a
#     lightness ramp on the class colour rather than by dash pattern, which does not scale to 5.
# Entries: (subset_name, linestyle, alpha, label, shade) where shade blends the class colour toward
# white (0 = full colour, 1 = white).
WP_SERIES_ALL = {
    "very_high": ("wp_very_high", "-", 1.0, "very-high winning p", 0.0),
    "high+": ("wp_high+", (0, (5, 1)), 0.72, "high+ winning p", 0.0),
    "moderate+": ("wp_moderate+", "--", 0.45, "moderate+ winning p", 0.0),
    "bin_very_low": ("wp_bin_very_low", "-", 1.0, "very low", 0.62),
    "bin_low": ("wp_bin_low", "-", 1.0, "low", 0.46),
    "bin_moderate": ("wp_bin_moderate", "-", 1.0, "moderate", 0.31),
    "bin_high": ("wp_bin_high", "-", 1.0, "high", 0.15),
    "bin_very_high": ("wp_bin_very_high", "-", 1.0, "very high", 0.0),
}
WP_SERIES = [WP_SERIES_ALL[k] for k in ("very_high", "moderate+")]   # rebound in main()


def _shade(hex_color: str, f: float) -> tuple:
    """Blend a hex colour toward white by fraction f (0 = unchanged, 1 = white)."""
    c = np.array(to_rgb(hex_color))
    return tuple(c + (np.ones(3) - c) * float(f))
PRETTY = {"very_low": "very low", "low": "low", "moderate": "moderate",
          "high": "high", "very_high": "very high"}

# poster/talk rendering; rebound in main() from --transparent_background
TRANSPARENT_BACKGROUND = False

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
TITLE_FS, LAB_FS, TICK_FS, SMALL_FS, LET_FS = 10.5, 9.2, 8.0, 7.2, 12

CAPTION = """\
Supplementary Figure Sx | The confidence-dependent divergence in ligand-binding-site detection
requires evidence density beneath it; separation raises detection for both classes and narrows it.

Harmonized heavy-atom DCA/TNS detection success for the three LBS predictors on the shared set of
n = 86,946 AlphaFold ST_100 pockets (34,900 AF-Apo / 51,751 AF-Holo / 295 tie). "Consensus" is the
per-pocket mean of the three predictors (a derived summary, not an independent fourth method).
**The unit of analysis is the UBS pocket throughout**: every rate, gap and n shown is over pockets.
95% CIs are cluster bootstraps that RESAMPLE PROTEINS (10,419 UniProts, ~8.3 pockets each, B = 1,000)
rather than pockets, because pockets within a protein are correlated; this widens the intervals
appropriately and leaves every point estimate unchanged. Pocket density and separation are the two
undirected support descriptors carried throughout: density = mean of the peak-normalized mode
(strongest-group) and combined (all-points) KDE density at the model's value; separation =
1 - overlap between the apo and holo reference distributions.

Throughout, "pp" = PERCENTAGE POINTS: a gap of 37 pp means AF-Holo pockets are detected 37 percentage
points more often than AF-Apo ones (e.g. 73% versus 36%). Columns pair the two descriptors:
(a, d) density, (b, e) separation. Support bins are DELIMITED (each pocket in exactly one bin), not
cumulative cut-offs.

(a, b) Which arm moves. AF-Apo and AF-Holo detection success across the bins of each support
descriptor, for pockets called with very-high winning probability (solid) and for the broader
moderate+ subset (dashed). The two descriptors behave oppositely -- and in ways a gap alone would
hide. DENSITY (a): AF-Apo detection falls from 47.6% at very-low density to 31.1% at very-high
(-16.5 pp) while AF-Holo is flat (72.4 -> 73.2%) -- the gap widens through the AF-Apo arm alone.
SEPARATION (b): BOTH classes improve as separation rises -- AF-Apo 28.0 -> 48.0% and AF-Holo
67.9 -> 80.8% -- so the gap NARROWS (39.9 -> 32.9 pp) even though absolute detection is highest
there. Well-separated apo/holo distributions mark pockets with a pronounced conformational
difference: distinctive, and easier for every predictor, AF-Apo most of all. The large gap at LOW
separation is therefore not a sign of good performance; it is the signature of poor AF-Apo detection
(28.0%) against a roughly unchanged AF-Holo baseline.
(c) The marginals: the apo/holo gap against each descriptor's own verbal bins, pooled over the
others. Only winning probability -- the sole DIRECTIONAL descriptor, defined as the normalized
posterior ratio dH*priorH / (dA*priorA + dH*priorH) at the model's value -- carries a class gradient
(3.7 -> 37.3 pp). Separation shows a partial echo (10.7 -> 23.2 pp) inherited from its correlation
with winning probability (rho = +0.45), and density is flat (17.9 -> 16.5 pp). This panel is a
SPECIFICITY CHECK on the construction, not a comparison of predictive power: density and separation
are undirected by construction and are not expected to rank classes.
(d, e) How much the apo/holo gap WIDENS across the winning-probability axis -- the Figure 6a tendency
itself -- recomputed within each support bin. Coloured line = consensus with its 95% CI; grey lines =
the individual predictors; n = pockets per bin. DENSITY (d): the tendency is essentially absent in
the very-low-density bin (consensus 10.2 pp [-0.27, 21.20], CI includes zero) and full strength
everywhere else (33.9 / 34.8 / 38.6 / 36.9) -- a threshold at the bottom of the range, not a gradient
across it. (Both ends contribute: that bin's confident end is far weaker, 24.9 vs 42.5 pp in the
high-density bin, and its unconfident end is anomalously high at 14.7 vs ~3 pp elsewhere; that cell
holds only 206 AF-Apo / 242 AF-Holo pockets and is correspondingly noisy, so the robust half of the
statement is the weak confident end.) SEPARATION (e): no such ordering (40.1 / 30.6 / 36.8 / 26.2 /
28.5), declining at its confident end.
(f) The headline contrast. Classification confidence translates into a detection divergence only
where there is evidence density beneath it; a confident class call in a near-empty region of the
conformational distribution carries little real experimental support, and for those pockets the
apo/holo detection gap largely disappears.

[INTERNAL NOTE -- not for publication]
* Panel letters are LOWER-CASE bold per Nature spec ("Separate panels in multi-panelled figures
  should be labelled with 8-pt bold, upright (not italic) and lowercase a, b, c"). The main figures
  currently use UPPER-CASE and need re-rendering.
* A 2D gap HEATMAP (winning-p bin x support bin) was tried for a/b and REJECTED. Plotting only the
  gap hides the two arms, and a reader -- correctly noting that nearly every tile was orange -- read
  large-gap tiles as "high performance", the opposite of the truth at low separation. Panels (a)/(b)
  exist precisely to prevent that reading. The grid survives as `fig6a_gates_gap_grid.csv`.
* Low density = DIFFUSE evidence, NOT little evidence: the very-high-probability AF-Apo pockets that
  a density gate would drop are structure-RICH (n_apo_avg 63.5 vs 42.1 for kept). Never describe them
  as "low-evidence" or "sparse".
* Never quote the POOLED Spearman(winning_p, density) = -0.141 -- it is a class-averaging dilution of
  -0.267 (apo) / -0.094 (holo). At very-high winning p, density's association with success flips sign
  by class (apo -0.119 [-0.146, -0.090]; holo +0.050 [+0.032, +0.068]).
* REDUNDANCY, if the figure must shrink: (f) is just (d)'s two extreme bins -- cut it first. (c) is
  the marginal of the (a)/(b) family. (a) and (b) are NOT droppable: they are the only panels showing
  WHICH class moves.
* The cumulative GATE ladder lives only as a table (`fig6a_gates_threshold_ladder.csv`): consensus
  dD = +2.03 / +2.98 / +4.44 / +3.26 pp for density at low+ / moderate+ / high+ / very_high+ and
  -1.51 / -0.72 / -5.63 / -5.07 for separation; retention 96 / 91 / 60 / 16% and 87 / 72 / 26 / 5%.
  It is the "is moderate+ a cherry-pick?" answer (no: rung-to-rung contrasts are all non-significant,
  and moderate+ is the pre-existing project constant `confident_probs`) plus the cost accounting. Its
  pocket-survival counts are covered better by the joint-tier Sankey (SI Fig S5), which already splits
  both classes by winning-p / density / separation tier at ST_100.
* Multiplicity: of the 12 gate x method comparisons in that table, only the consensus density result
  survives Westfall-Young step-down correction (p = 0.014). The (d) interaction is far larger and does
  not depend on that correction.
* The separation GATE shifts headline marginals (overall success up to +3.4 pp) and must NEVER be
  adopted as a default filter. A density gate does not (<= 0.41 pp on any reported cell).
"""


def apply_background(fig):
    """Poster mode = transparent OUTSIDE the axes, white INSIDE the plot area (the same switch the
    plotting/ scripts carry). The plot area has to stay white: these panels use a near-white grid
    and white-backed value labels, which would vanish on a coloured poster board."""
    if TRANSPARENT_BACKGROUND:
        fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        for ax in fig.axes:
            ax.set_facecolor("white")
    else:
        fig.patch.set_facecolor("white")


def savef(fig, outdir: _Path, name: str, dpi=300, svg=False):
    apply_background(fig)
    # poster exports are png + svg: a poster build never consumes the pdf
    exts = ("png", "svg") if TRANSPARENT_BACKGROUND else \
        (("png", "pdf", "svg") if svg else ("png", "pdf"))
    for ext in exts:
        fig.savefig(outdir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", name, "(%s)" % ",".join(exts))


# ============================================================== SI figure
def fig_si(alt: pd.DataFrame, dsb: pd.DataFrame, db: pd.DataFrame, hc: pd.DataFrame,
           outdir: _Path, name: str):
    """Panel letters are LOWER-CASE BOLD per Nature spec ("Separate panels in multi-panelled
    figures should be labelled with 8-pt bold, upright (not italic) and lowercase a, b, c").

    Layout is a 2x3 with the two supports as COLUMNS and the two questions as ROWS:
        a  density -> apo/holo success        b  separation -> apo/holo success
        d  density -> gap widening            e  separation -> gap widening
    plus c (the marginals / specificity check) and f (the headline contrast).

    Design history, so it is not re-litigated:
      * A 2D gap heatmap was tried for a/b and REJECTED. Plotting only the GAP hides the apo and holo
        arms, and a reader (correctly noting that nearly every tile was orange) read large-gap tiles as
        "high performance" -- the opposite of the truth at low separation, where the gap is large
        precisely BECAUSE AF-Apo detection is poor (28.0% vs 67.9% holo). Showing the two arms
        separately makes that unmissable. The grid survives as a table (fig6a_gates_gap_grid.csv).
      * The cumulative gate ladder is likewise a TABLE only (fig6a_gates_threshold_ladder.csv).
      * "pp" = percentage points throughout; spelled out on the c/d/e/f axes and in the caption.
    """
    fig = plt.figure(figsize=(12.6, 8.0))
    gs = GridSpec(2, 3, figure=fig, wspace=0.34, hspace=0.58,
                  left=0.06, right=0.985, top=0.90, bottom=0.09)

    def letter(ax, s, dx=-0.24):
        ax.text(dx, 1.06, s, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=LET_FS, fontweight="bold")

    x = np.arange(5)

    # --- a, b: the two arms, per support. THE panels: they show WHICH class moves.
    for k, (prop, col, lt) in enumerate([("density", DENS_C, "a"), ("separation", SEP_C, "b")]):
        ax = fig.add_subplot(gs[0, k])
        for wp, ls, alpha, lab, shade in WP_SERIES:
            t = db[(db.property == prop) & (db.wp_subset == wp) & (db.method == "consensus")] \
                .set_index("bin").reindex(LEVELS5)
            for cl, ccol in [("apo", APO), ("holo", HOLO)]:
                ax.plot(x, t[f"rate_{cl}"], marker="o", ms=3.4 if shade else 4.2,
                        lw=1.7 if shade else 2.2, ls=ls, color=_shade(ccol, shade), alpha=alpha,
                        label=f"AF-{cl.capitalize()} \u00b7 {lab}")
        # annotate the strictest apo arm's endpoints -- the whole story is in that line. Skipped in
        # the delimited variant, where 5 apo lines converge at the left edge and labels would collide.
        if not any(s[4] for s in WP_SERIES):
            t = db[(db.property == prop) & (db.wp_subset == "wp_very_high")
                   & (db.method == "consensus")].set_index("bin").reindex(LEVELS5)
            for xi, lv in [(0, "very_low"), (4, "very_high")]:
                ax.annotate(f'{t.loc[lv, "rate_apo"]:.0f}', (xi, t.loc[lv, "rate_apo"]),
                            xytext=(6 if xi == 0 else -16, -12), textcoords="offset points",
                            fontsize=6.8, color=APO, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY[l] for l in LEVELS5], fontsize=6.4, rotation=30, ha="right")
        ax.set_xlabel(f"{prop} bin (delimited)", fontsize=LAB_FS)
        ax.set_ylim(20, 90)
        ax.set_title(prop, fontsize=TITLE_FS, color=col, fontweight="bold")
        ax.grid(color="#ececec", lw=.7, ls="--")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=TICK_FS)
        if k == 0:
            ax.set_ylabel("Detection success (%)", fontsize=LAB_FS)
            if len(WP_SERIES) > 3:
                # delimited variant: 10 entries -> two columns (apo | holo), outside the axes, with
                # the winning-p bin as the legend title since that is what the shading encodes
                ax.legend(frameon=False, fontsize=5.0, loc="upper left", bbox_to_anchor=(0.0, -0.30),
                          ncol=2, columnspacing=0.9, handlelength=1.6, labelspacing=0.25,
                          title="winning-probability bin (shade)", title_fontsize=5.4)
            else:
                ax.legend(frameon=False, fontsize=5.8, loc="lower left")
        letter(ax, lt)

    # --- c: the marginals (specificity check -- NOT a horse race; see caption)
    axC = fig.add_subplot(gs[0, 2])
    for ax_name, col in [("winning_p", "#C0392B"), ("separation", SEP_C), ("density", DENS_C)]:
        t = alt[(alt.axis == ax_name) & (alt.method == "consensus")].set_index("bin").reindex(LEVELS5)
        axC.plot(x, t["gap"].to_numpy(float), marker="o", ms=4, lw=2.0, color=col,
                 label={"winning_p": "winning probability (directional)",
                        "separation": "separation (undirected)",
                        "density": "density (undirected)"}[ax_name])
    axC.set_xticks(x)
    axC.set_xticklabels([PRETTY[l] for l in LEVELS5], fontsize=6.4, rotation=30, ha="right")
    axC.set_ylim(0, 42)
    axC.set_ylabel("AF-Holo \u2212 AF-Apo gap\n(percentage points, pp)", fontsize=LAB_FS)
    axC.set_xlabel("verbal bin of that descriptor", fontsize=LAB_FS)
    axC.set_title("only the directional descriptor\ncarries a class gradient", fontsize=9.2)
    axC.legend(frameon=False, fontsize=5.6, loc="upper left")
    axC.grid(color="#ececec", lw=.7, ls="--")
    axC.set_axisbelow(True)
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)
    axC.tick_params(labelsize=TICK_FS)
    letter(axC, "c")

    # --- d, e: gap widening within each support bin, per predictor + n.
    # Restored as two separate panels per review: the per-predictor spread and the cell counts were
    # wanted; only the LABEL was unclear (that winning_p is the axis being contrasted).
    for k, (prop, col, lt) in enumerate([("density", DENS_C, "d"), ("separation", SEP_C, "e")]):
        ax = fig.add_subplot(gs[1, k])
        t = dsb[(dsb.property == prop) & (dsb.method == "consensus")].set_index("bin").reindex(LEVELS5)
        ax.fill_between(x, t["D_lo"], t["D_hi"], color=col, alpha=.18, lw=0)
        ax.plot(x, t["D"], marker="o", ms=6, lw=2.6, color=col, label="consensus")
        for m, ls, mk in zip(PRED, [":", "--", "-."], ["s", "^", "v"]):
            tm = dsb[(dsb.property == prop) & (dsb.method == m)].set_index("bin").reindex(LEVELS5)
            ax.plot(x, tm["D"], ls=ls, marker=mk, ms=3.4, lw=0.9, color="0.62", label=m)
        ax.axhline(0, color="0.35", lw=1.0)
        for xi, lv in zip(x, LEVELS5):
            ax.text(xi, -4.8, f'n={int(t.loc[lv, "n_bin"]):,}', ha="center", fontsize=5.4, color="0.5")
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY[l] for l in LEVELS5], fontsize=6.4, rotation=30, ha="right")
        ax.set_xlabel(f"{prop} bin (delimited)", fontsize=LAB_FS)
        ax.set_ylim(*ROW2_YLIM)   # shared with panel f so the whole row sits on one baseline
        ax.set_title(f"gap widening within each {prop} bin", fontsize=9.2, color=col, fontweight="bold")
        ax.grid(color="#ececec", lw=.7, ls="--")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=TICK_FS)
        if k == 0:
            # spell the estimand out on the axis -- "Divergence D" alone was not readable
            ax.set_ylabel("Gap widening across winning p\n(very high \u2212 very low, pp)",
                          fontsize=LAB_FS)
            # center-right, not lower-right: the n= row owns the bottom of the axes
            ax.legend(frameon=False, fontsize=5.6, loc="center right", borderaxespad=0.4)
        letter(ax, lt)

    # --- f: the headline contrast, BOTH descriptors, bottom vs top support bin.
    # UNPOOLED by design: pooling {very_low+low} vs {high+very_high} HALVES the contrast (density
    # +28.4 -> +14.2, separation -13.9 -> -7.6) because density's effect is a threshold at very_low
    # only, and it barely buys precision (bottom-group n 2.2x, CI width -6%) because the bootstrap is
    # clustered by UniProt. Tested and recorded in fig6a_gates_headline_contrasts.csv.
    axF = fig.add_subplot(gs[1, 2])
    groups = [("density", "very_low", "#c9d9cf"), ("density", "high", DENS_C),
              ("separation", "very_low", "#d9cfe0"), ("separation", "high", SEP_C)]
    xs = [0, 1, 2.4, 3.4]
    for (prop, lv, ccol), xi in zip(groups, xs):
        t = dsb[(dsb.property == prop) & (dsb.method == "consensus")].set_index("bin")
        v = t.loc[lv, "D"]
        err = np.array([[v - t.loc[lv, "D_lo"]], [t.loc[lv, "D_hi"] - v]])
        axF.bar([xi], [v], 0.72, yerr=err, capsize=3.5, color=ccol, edgecolor="0.3", lw=.7,
                error_kw=dict(lw=1.0, ecolor="0.35"))
        axF.text(xi, t.loc[lv, "D_hi"] + 1.5, f"{v:.1f}", ha="center", fontsize=7.6,
                 fontweight="bold", color="0.2")
        axF.text(xi, -4.8, f'n={int(t.loc[lv, "n_bin"]):,}', ha="center", fontsize=5.4, color="0.5")
    # contrast brackets
    hcc = hc[(hc.scheme == "unpooled") & (hc.method == "consensus")].set_index("property")
    for (x0, x1), prop, col in [((0, 1), "density", DENS_C), ((2.4, 3.4), "separation", SEP_C)]:
        c = hcc.loc[prop, "contrast"]
        sig = (hcc.loc[prop, "contrast_lo"] > 0) or (hcc.loc[prop, "contrast_hi"] < 0)
        # sits clear of the tallest CI whisker + its value label
        y = 56.2
        axF.plot([x0, x0, x1, x1], [y - 1.6, y, y, y - 1.6], lw=.9, color="0.4")
        axF.text((x0 + x1) / 2, y + 0.9, f"{c:+.1f}" + ("" if sig else " n.s."), ha="center",
                 fontsize=7.0, fontweight="bold" if sig else "normal", color=col if sig else "0.45")
    axF.axhline(0, color="0.35", lw=1.0)
    axF.set_xticks(xs)
    axF.set_xticklabels(["very low", "high", "very low", "high"], fontsize=6.8)
    for xt, lab, col in [(0.5, "density", DENS_C), (2.9, "separation", SEP_C)]:
        axF.text(xt, -0.155, lab, transform=axF.get_xaxis_transform(), ha="center", va="top",
                 fontsize=8.0, color=col, fontweight="bold")
    axF.set_ylabel("Gap widening across winning p\n(very high \u2212 very low, pp)", fontsize=LAB_FS)
    axF.set_ylim(*ROW2_YLIM)
    # the negative space exists only to seat the n= row and the descriptor labels -- no ticks or
    # gridlines there, they would run straight through the labels
    axF.set_yticks([0, 10, 20, 30, 40, 50])
    axF.set_xlim(-0.7, 4.1)
    axF.set_title("the confidence axis diverges where\nevidence supports it \u2014 not where states differ",
                  fontsize=9.0)
    axF.grid(axis="y", color="#ececec", lw=.7, ls="--")
    axF.set_axisbelow(True)
    for s in ("top", "right"):          # keep the bottom spine: d and e have one
        axF.spines[s].set_visible(False)
    axF.tick_params(labelsize=TICK_FS, length=0)
    letter(axF, "f")

    savef(fig, outdir, name, svg=True)


# ============================================================== internal: Panel A grid
def _overall(t: pd.DataFrame, cl: str) -> float:
    """Marginal success rate over the whole gated set, from the per-bin table.

    Exact, not an approximation: each bin's rate is a mean over that bin's pockets, so the
    pocket-count-weighted mean of the bins is the mean over all of them. NaN bins (no pockets of
    that class) are dropped by the mask rather than poisoning the sum.
    """
    r = t[f"rate_{cl}"].to_numpy(float)
    n = t[("n_bin" if cl == "all" else f"n_{cl}")].to_numpy(float)
    ok = np.isfinite(r) & np.isfinite(n) & (n > 0)
    return float(np.sum(r[ok] * n[ok]) / np.sum(n[ok])) if ok.any() else np.nan


def fig_panelA_grid(br: pd.DataFrame, comp: pd.DataFrame, outdir: _Path,
                    gates=None, name="fig6a_gates_panelA_grid", family_lab="moderate+"):
    gates = gates or GATES
    fig, axes = plt.subplots(len(gates), len(PRED), figsize=(12.2, 12.6), sharey=True)
    w = 0.27
    xpos = np.arange(len(LEVELS5))
    for i, gate in enumerate(gates):
        n_gate = int(comp[comp.gate == gate]["n"].iloc[0])
        frac = float(comp[comp.gate == gate]["frac_kept"].iloc[0])
        for j, m in enumerate(PRED):
            ax = axes[i, j]
            t = br[(br.gate == gate) & (br.method == m)].set_index("bin").reindex(LEVELS5)

            # overall (bin-marginal) rate per class: dashed reference line + value at the right edge.
            # Drawn UNDER the bars so it reads as a baseline, and it shows at a glance which bins
            # beat the predictor's own average -- the bars alone only give the shape, not the level.
            ov = {cl: _overall(t, cl) for cl in ("apo", "all", "holo")}
            for cl, col in [("apo", APO), ("all", ALLC), ("holo", HOLO)]:
                if not np.isfinite(ov[cl]):
                    continue
                # a segment, not axhline: it stops short of the right margin so the value label
                # sits on clean white instead of having the line drawn through it
                ax.plot([-0.5, 4.42], [ov[cl]] * 2, color=col, lw=1.0, ls=(0, (4, 2)), alpha=.85,
                        zorder=0, clip_on=False, solid_capstyle="butt")
                ax.text(4.52, ov[cl], f"{ov[cl]:.0f}", ha="left", va="center", fontsize=5.6,
                        color=col, fontweight="bold")

            for cl, col, off, lab in [("apo", APO, -w, "AF-Apo"), ("all", ALLC, 0.0, "All"),
                                      ("holo", HOLO, +w, "AF-Holo")]:
                vals = t[f"rate_{cl}"].to_numpy(float)
                bars = ax.bar(xpos + off, vals, w, color=col, edgecolor="0.3", lw=.5, label=lab,
                              zorder=2)
                for bar, v in zip(bars, vals):
                    if np.isfinite(v):
                        # white halo: a per-bin value can land on top of a reference line, and the
                        # dashes would otherwise cut straight through the digits
                        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.0f}",
                                ha="center", va="bottom", fontsize=5.0, color="0.25", zorder=4,
                                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
            D = t.loc["very_high", "gap"] - t.loc["very_low", "gap"]
            ax.text(0.02, 0.965, f"D = {D:.1f} pp", transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.0, fontweight="bold", color="0.15")
            ax.text(0.02, 0.905,
                    f"overall {ov['apo']:.0f} / {ov['all']:.0f} / {ov['holo']:.0f}"
                    f"  (gap {ov['holo'] - ov['apo']:+.0f})",
                    transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color="0.45")
            ax.set_xticks(xpos)
            ax.set_xticklabels([PRETTY[l] for l in LEVELS5], fontsize=6.0)
            ax.set_xlim(-0.55, 5.0)      # right margin seats the overall-rate value labels
            ax.set_ylim(0, 100)
            ax.grid(axis="y", color="#ececec", lw=.7, ls="--"); ax.set_axisbelow(True)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=TICK_FS)
            if i == 0: ax.set_title(m, fontsize=TITLE_FS)
            if j == 0:
                ax.set_ylabel("Success rate (%)", fontsize=LAB_FS)
                ax.text(-0.30, 0.5, f"{GATE_LAB[gate]}\nn = {n_gate:,} ({100*frac:.0f}%)",
                        transform=ax.transAxes, ha="center", va="center", rotation=90,
                        fontsize=8.6, fontweight="bold", color=GATE_COL[gate])
            if i == len(GATES) - 1 and j == 1:
                ax.set_xlabel("Winning-probability bin", fontsize=LAB_FS, labelpad=6)
    axes[0, 0].legend(frameon=False, fontsize=SMALL_FS, loc="upper right",
                      handlelength=1.1, labelspacing=.3, borderaxespad=.2)
    fig.suptitle(f"[INTERNAL] Figure 6 Panel A under {family_lab} quality gates — success vs "
                 f"winning probability\nD = gap(very high) − gap(very low);  dashed lines + "
                 f"right-edge values = that panel's OVERALL rate per class", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0.02, 0, 1, 0.965])
    savef(fig, outdir, name, dpi=200)


# ============================================================== internal: summary
def fig_summary(br: pd.DataFrame, dv: pd.DataFrame, comp: pd.DataFrame, outdir: _Path):
    fig = plt.figure(figsize=(12.2, 7.0))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.85], hspace=0.45, wspace=0.28,
                  left=0.06, right=0.985, top=0.89, bottom=0.09)
    xpos = np.arange(len(LEVELS5))
    for j, m in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, j])
        for gate in GATES:
            t = br[(br.gate == gate) & (br.method == m)].set_index("bin").reindex(LEVELS5)
            ax.plot(xpos, t["gap"].to_numpy(float), marker="o", ms=4, color=GATE_COL[gate],
                    lw=2.0 if gate == "all" else 1.5, ls="-" if gate == "all" else "--",
                    label=GATE_LAB[gate])
        ax.set_xticks(xpos)
        ax.set_xticklabels([PRETTY[l] for l in LEVELS5], fontsize=6.2, rotation=30, ha="right")
        ax.set_title(m, fontsize=TITLE_FS); ax.set_ylim(0, 50)
        ax.grid(color="#ececec", lw=.7, ls="--"); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=TICK_FS)
        if j == 0:
            ax.set_ylabel("AF-Holo − AF-Apo gap (pp)", fontsize=LAB_FS)
            ax.legend(frameon=False, fontsize=6.4, loc="upper left")
        if j == 1:
            ax.set_xlabel("Winning-probability bin", fontsize=LAB_FS)

    axf = fig.add_subplot(gs[1, 0:2])
    order = [(g, m) for g in ["dens_mod+", "sep_mod+", "both_mod+"] for m in METHODS]
    ys = np.arange(len(order))[::-1]
    for y, (g, m) in zip(ys, order):
        r = dv[(dv.gate == g) & (dv.method == m)].iloc[0]
        sig = (r.dD_lo > 0) or (r.dD_hi < 0)
        axf.plot([r.dD_lo, r.dD_hi], [y, y], color=GATE_COL[g], lw=2.0, alpha=.85, solid_capstyle="round")
        axf.plot(r.dD_vs_ungated, y, "o", ms=6.5, color=GATE_COL[g],
                 mec="0.15" if sig else "white", mew=1.2 if sig else 0.8, zorder=3)
        axf.text(11.6, y, f"{r.dD_vs_ungated:+.1f} [{r.dD_lo:+.1f}, {r.dD_hi:+.1f}]",
                 va="center", fontsize=6.6, color="0.25", family="monospace")
    axf.axvline(0, color="0.35", lw=1.0)
    axf.set_yticks(ys); axf.set_yticklabels([f"{GATE_LAB[g]} · {m}" for g, m in order], fontsize=7.0)
    axf.set_xlim(-8, 18.5); axf.set_xticks([-6, -3, 0, 3, 6, 9])
    axf.set_xlabel("Δ divergence vs ungated (pp)   —   right = gate STRENGTHENS the tendency", fontsize=LAB_FS)
    axf.grid(axis="x", color="#ececec", lw=.7, ls="--"); axf.set_axisbelow(True)
    for s in ("top", "right"): axf.spines[s].set_visible(False)
    axf.tick_params(labelsize=TICK_FS)
    axf.set_title("Does the gate strengthen the winning-p divergence?  (paired cluster bootstrap, 95% CI)",
                  fontsize=9.0, loc="left")

    axc = fig.add_subplot(gs[1, 2])
    gg = [g for g in GATES if g != "all"]
    kept = [100 * float(comp[comp.gate == g]["frac_kept"].iloc[0]) for g in gg]
    axc.bar(np.arange(len(gg)), kept, 0.6, color=[GATE_COL[g] for g in gg], edgecolor="0.3", lw=.6)
    for i, v in enumerate(kept):
        axc.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7.4, color="0.25")
    axc.set_xticks(np.arange(len(gg)))
    axc.set_xticklabels([GATE_LAB[g] for g in gg], fontsize=6.8, rotation=20, ha="right")
    axc.set_ylim(0, 105); axc.set_ylabel("Pockets retained (%)", fontsize=LAB_FS)
    axc.set_title("Cost of the gate", fontsize=9.0)
    axc.grid(axis="y", color="#ececec", lw=.7, ls="--"); axc.set_axisbelow(True)
    for s in ("top", "right"): axc.spines[s].set_visible(False)
    axc.tick_params(labelsize=TICK_FS)

    axr = fig.add_subplot(gs[1, 3])
    rr = ["retain_" + l for l in LEVELS5]
    for g in gg:
        axr.plot(np.arange(5), comp[comp.gate == g][rr].iloc[0].to_numpy(float), marker="o", ms=4,
                 lw=1.8, color=GATE_COL[g], label=GATE_LAB[g])
    axr.set_xticks(np.arange(5))
    axr.set_xticklabels([PRETTY[l] for l in LEVELS5], fontsize=6.2, rotation=30, ha="right")
    axr.set_ylim(0, 105)
    axr.set_ylabel("Retention (%)", fontsize=LAB_FS)
    axr.set_xlabel("Winning-probability bin", fontsize=7.6)
    axr.set_title("Retention along the plotted axis", fontsize=8.6)
    axr.legend(frameon=False, fontsize=6.2, loc="lower left")
    axr.grid(color="#ececec", lw=.7, ls="--"); axr.set_axisbelow(True)
    for s in ("top", "right"): axr.spines[s].set_visible(False)
    axr.tick_params(labelsize=TICK_FS)

    fig.suptitle("[INTERNAL] Density / separation gating of the Figure 6A winning-probability divergence",
                 fontsize=12.5, y=0.965)
    savef(fig, outdir, "fig6a_gates_summary", dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=None, help="default: work_dir/lbs_generality/analysis_v1/tables")
    ap.add_argument("--outdir", default=None, help="default: work_dir/lbs_generality/analysis_v1/plots")
    ap.add_argument("--manuscript_dir", default=None, help="also copy the SI figure + caption here")
    ap.add_argument("--manuscript_name", default="SI_fig6a_density_separation_gates")
    ap.add_argument("--transparent_background", action="store_true",
                    help="poster mode: transparent outside the axes, white inside the plot area. "
                         "Exports png + svg only (no pdf).")
    ap.add_argument("--poster_output_dir", action="store_true",
                    help="write to the configured poster/talk plots directory (paths.ini "
                         "'presentation_dir') instead of the analysis plots directory.")
    ap.add_argument("--wp_series", default="very_high,moderate+",
                    help="comma-separated winning-p subsets to draw in panels a/b, in draw order. "
                         "Options: very_high, high+, moderate+ (they are nested). "
                         "Default 'very_high,moderate+'; pass 'very_high,high+,moderate+' for the "
                         "3-series variant.")
    a = ap.parse_args()

    global WP_SERIES, TRANSPARENT_BACKGROUND
    TRANSPARENT_BACKGROUND = a.transparent_background
    try:
        WP_SERIES = [WP_SERIES_ALL[k.strip()] for k in a.wp_series.split(",") if k.strip()]
    except KeyError as e:
        ap.error(f"unknown --wp_series entry {e}; choose from {list(WP_SERIES_ALL)}")

    cfg.require("WORK_DIR")
    tdir = _Path(a.tables) if a.tables else (cfg.LBS_ANALYSIS_DIR / "tables")
    if a.outdir:
        outdir = _Path(a.outdir)
    elif a.poster_output_dir:
        cfg.require("PRESENTATION_DIR")
        outdir = _Path(cfg.PRESENTATION_DIR)
    else:
        outdir = cfg.LBS_ANALYSIS_DIR / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    br = pd.read_csv(tdir / "fig6a_gates_bin_rates.csv")
    dv = pd.read_csv(tdir / "fig6a_gates_divergence.csv")
    comp = pd.read_csv(tdir / "fig6a_gates_composition.csv")
    ld = pd.read_csv(tdir / "fig6a_gates_threshold_ladder.csv")
    alt = pd.read_csv(tdir / "fig6a_gates_altaxis_bin_rates.csv")
    dsb = pd.read_csv(tdir / "fig6a_gates_divergence_by_support_bin.csv")
    db = pd.read_csv(tdir / "fig6a_gates_delimited_bins.csv")

    hc = pd.read_csv(tdir / "fig6a_gates_headline_contrasts.csv")
    fig_si(alt, dsb, db, hc, outdir, a.manuscript_name)
    (outdir / f"{a.manuscript_name}_caption.txt").write_text(CAPTION, encoding="utf-8")
    fig_panelA_grid(br, comp, outdir)                                   # manuscript gates (moderate+)
    fig_panelA_grid(br, comp, outdir, gates=GATES_HIGH, family_lab="high+",
                    name="fig6a_gates_panelA_grid_high")                # stricter-cut counterpart
    fig_summary(br, dv, comp, outdir)

    if a.manuscript_dir:
        mdir = _Path(a.manuscript_dir)
        mdir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):           # poster mode produces no pdf
            src = outdir / f"{a.manuscript_name}.{ext}"
            if src.is_file():
                shutil.copy2(src, mdir / src.name)
        shutil.copy2(outdir / f"{a.manuscript_name}_caption.txt", mdir / f"{a.manuscript_name}_caption.txt")
        print("copied SI figure + caption ->", mdir)
    print("wrote ->", outdir)


if __name__ == "__main__":
    main()
