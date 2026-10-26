# -*- coding: utf-8 -*-
"""
Figures for the ratio-vs-class 2x2 quadrant read.
=================================================

Consumes the tables written by `analyze_ratio_vs_class_quadrants.py`.

**SI deliverable** (`SI_ratio_vs_class_quadrants`), 3 panels:
  a  the ask: each method's contour surface collapsed onto the four quadrants of
     (evidence balance) x (classification), with the pooled success rate and n in every cell.
     Q1/Q3 are the concordant corners, Q2/Q4 the discordant ones.
  b  the two main effects side by side, per method: how many pp of success does moving from
     apo-leaning to holo-leaning EVIDENCE buy, versus moving from AF-Apo to AF-Holo CLASS.
     Open markers = the same effect after standardizing to a common pocket-size distribution.
  c  the headline: PREF = E_ratio - E_class = Q4 - Q2. Positive => the evidence balance is the
     stronger facilitator. Shown raw and size-standardized, with cluster-bootstrap CIs.

Internal reference figures (not for the SI):
  `quadrants_tiles`              -- panel a alone, enlarged (slides / the synthesis HTML)
  `quadrants_scheme_robustness`  -- PREF across the three cut schemes x both adjustments, plus the
                                    pocket-size composition that drives the raw/adjusted gap
  `quadrants_tiles_size_std`     -- panel a on directly standardized rates
  `quadrants_tiles_size_resid`   -- panel a on residual (indirect) size-adjusted rates

Convention: SI figures live under `<project>/manuscript/figures/SI/` with an `SI_` prefix and a
DESCRIPTIVE name (no figure number -- the final `Figure Sx` is assigned at SI finalization).
`--manuscript_dir <dir>` copies the SI figure + caption there. Panel letters are lower-case bold
per Nature spec.

Inputs  : work_dir/lbs_generality/analysis_v1/tables/quadrants_*.csv
Outputs : work_dir/lbs_generality/analysis_v1/plots/*.{png,pdf,svg}

Base env (pandas + numpy + matplotlib):
    python 06_lbs_generality/analysis/plot_ratio_vs_class_quadrants.py \
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

PRED = ["P2Rank", "fpocket", "IF-SitePred"]
LBS_METHODS = PRED + ["consensus"]
METHODS = LBS_METHODS + ["docking"]
DISP = {"consensus": "Consensus", "docking": "Docking"}
SCHEMES = ["sign", "median", "quartile"]
SCHEME_LAB = {"sign": "sign cut\n(ratio 0, $p_\\mathrm{holo}$ 0.5)", "median": "median cut\n(equal marginals)",
              "quartile": "quartile cut\n(matched extremes)"}
# The two size adjustments, in the words the caption uses. They estimate the same thing by
# different routes -- `size_std` reweights each quadrant to the population's size distribution,
# `size_resid` subtracts what size alone predicts -- and they agree to ~1-2 pp, which is why both
# are drawn: agreement between two estimators is the evidence that the shrinkage is real.
ADJ_LAB = {"raw": "raw", "size_std": "size-standardized (direct)",
           "size_resid": "size-adjusted (residual)"}

# palette: reuse the ratio / p_holo colours of the sibling ratio-vs-class analysis so the two
# figures read as one family (ratio = purple, classification = blue)
RATIO_C, CLASS_C = "#7B3294", "#2C7FB8"
APO, HOLO, ALLC = "#2C7FB8", "#F28E2B", "#8a8f98"
# contour-figure convention: the detectors share one scale (comparable); docking, a different
# readout on a different denominator, keeps its own scale and its own colormap
PRED_CMAP, PRED_VLIM = "viridis", (20.0, 80.0)
DOCK_CMAP, DOCK_VLIM = "magma", (30.0, 90.0)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
TITLE_FS, LAB_FS, TICK_FS, SMALL_FS, LET_FS = 10.5, 9.2, 8.0, 7.2, 12

CAPTION = """\
Supplementary Figure Sx | Holo-leaning experimental evidence is a stronger facilitator of pocket
detection than the apo/holo class label itself, and about half of that advantage is pocket size.

Each method's success surface over evidence balance (x) and classification posterior (y) -- the
surface plotted as contours in Figure 6 -- collapsed onto the four quadrants defined by the two
substantive cut points, log2((n_holo+1)/n_apo) = 0 and p_holo = 0.5 (the class boundary; 99.97%
agreement with `pocket_class`). (a) Pooled success rate and pocket count per quadrant. Q1 and Q3 are
the concordant corners (evidence and class agree); Q2 and Q4 are discordant. The three LBS
predictors are scored on the identical set of n = 86,946 AlphaFold ST_100 pockets under the
harmonized heavy-atom DCA/TNS criterion; "Consensus" is the per-pocket mean of the three (a derived
summary, not an independent fourth method) and shares their colour scale. Docking is a DIFFERENT
readout (fraction of cognate ligands docked into the site > 0) on its own nested G1 population
(n = 60,893) and therefore carries its own scale -- its rates are not comparable in level to the
detectors'. (b) The two main effects per method: E_ratio = 1/2[(Q1-Q2)+(Q4-Q3)] and
E_class = 1/2[(Q1-Q4)+(Q2-Q3)], in percentage points of success. (c) Their difference,
PREF = E_ratio - E_class, which by the algebra of a 2x2 equals Q4 - Q2, the contrast between the two
discordant corners: positive values mean the evidence balance is the stronger facilitator. Filled
markers/bars are raw rates; open markers hold pocket size fixed, by two independent routes --
circles by direct standardization to the method's own union-Rg decile distribution, squares by
subtracting what size alone predicts (success minus E[success | union-Rg decile]). The two agree to
1-2 pp throughout, so the shrinkage is a property of the data and not of either estimator. NO
pockets are discarded by either adjustment: both reweight rather than subset. 95% CIs are cluster
bootstraps that RESAMPLE PROTEINS (UniProt clusters, B = 1,000) rather than pockets, because pockets
within a protein are correlated.

The evidence balance wins in every detector, raw (PREF +17 to +22 pp) and size-standardized (+7 to
+10 pp); for docking the raw advantage (+15 pp) is abolished once size is held fixed, i.e. for
ligand placement the two facilitators are of comparable strength. The halving under standardization
is composition, not noise: holo-leaning sites are systematically larger (mean union Rg 7.7 A in Q1
and Q4 vs 6.2-6.7 A in Q2 and Q3), and size dominates detection.

[INTERNAL NOTE -- not for the caption] The 2x2 is descriptive and DICHOTOMIZED; the inferential
ranking of the two as continuous predictors is `analyze_lbs_ratio_vs_class.py` (size-controlled
partial Spearman + nested-CV AUC), and it agrees at zero order (ratio >= p_holo >= is_holo in all
four methods).

A 2x2 compares one step along x with one step along y, so the answer is conditional on where the
axes are cut -- and the sign cut hands the ratio the LONGER LEVER (it splits a wide, heavy-tailed
axis at 0, while p_holo is split near its own median). Two corrections follow, and neither is
optional when quoting this figure:
  * PREF's point estimate stays positive for all four detectors under all three cut schemes and
    both adjustments -- but at the MATCHED-EXTREME (quartile) cut WITH size held fixed the CIs span
    zero for P2Rank (+3.1 [-5.6, +11.2]), IF-SitePred (+3.6 [-5.3, +11.6]) and the consensus
    (+5.7 [-0.6, +12.2]); only fpocket stays clearly positive (+10.4 [+1.9, +19.7]). So the robust
    claim is "the ratio wins at the sign cut, raw and size-adjusted". **Once the axes are given
    equal lever arms AND size is removed, the two facilitators are statistically indistinguishable
    for two of the three predictors** -- state that, do not bury it.
  * The quartile cut is the only place a real sample-size cost appears: 24,136 of 86,946 pockets
    (28%), because the discordant extreme corners are genuinely rare (Q2 1,198, Q4 1,825). The size
    STANDARDIZATION itself drops nobody -- it reweights (Kish ESS 92 / 88 / 67 / 95% of n for
    Q1..Q4; Q3 pays most, being the quadrant least like the population in size).

Docking is the one method whose PREF sign flips (quartile + size_std goes to -5.5 pp), which is why
docking's statement is "comparable strength", not "class wins". The direction of travel matches the
quality-gated result: docking and IF-SitePred were also the two methods where the binary class
label overtook the ratio under size control at high classification confidence.
"""


def savef(fig, outdir: _Path, name: str, dpi=300, svg=True):
    exts = ("png", "pdf", "svg") if svg else ("png", "pdf")
    for ext in exts:
        fig.savefig(outdir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name, "(%s)" % ",".join(exts))


def _letter(ax, s, dx=-0.16, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=LET_FS, fontweight="bold")


def _row(qt: pd.DataFrame, method: str, adjust: str, scheme: str = "sign") -> pd.Series:
    r = qt[(qt.method == method) & (qt.adjust == adjust) & (qt.scheme == scheme)]
    if len(r) != 1:
        raise KeyError(f"expected 1 row for {method}/{adjust}/{scheme}, got {len(r)}")
    return r.iloc[0]


# ------------------------------------------------------------------- panel a: the tiles
def draw_tile(ax, r: pd.Series, method: str, cmap: str, vlim: tuple[float, float],
              show_n: bool = True, ylabels: bool = True):
    """One 2x2. Layout matches the contour it summarizes: holo class on TOP, holo evidence RIGHT."""
    grid = np.array([[r.rate_Q2, r.rate_Q1],
                     [r.rate_Q3, r.rate_Q4]], float)
    ns = np.array([[r.n_Q2, r.n_Q1], [r.n_Q3, r.n_Q4]], float)
    tags = np.array([["Q2", "Q1"], ["Q3", "Q4"]])
    im = ax.imshow(grid, cmap=cmap, vmin=vlim[0], vmax=vlim[1], origin="upper", aspect="auto")
    cm = plt.get_cmap(cmap)
    for i in range(2):
        for j in range(2):
            rgb = cm((grid[i, j] - vlim[0]) / (vlim[1] - vlim[0]))[:3]
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            c = "white" if lum < 0.55 else "0.12"
            ax.text(j, i - 0.16, f"{grid[i, j]:.0f}%", ha="center", va="center",
                    fontsize=15, fontweight="bold", color=c)
            if show_n:
                ax.text(j, i + 0.17, f"n = {ns[i, j]:,.0f}", ha="center", va="center",
                        fontsize=SMALL_FS, color=c)
            ax.text(j - 0.45, i - 0.42, tags[i, j], ha="left", va="center",
                    fontsize=SMALL_FS, color=c, alpha=0.85)
    # the discordant corners -- the two cells whose difference IS the headline
    for (i, j) in [(0, 0), (1, 1)]:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=2.0,
                                   ec="white", ls=(0, (3, 2)), alpha=0.9))
    ax.set_xticks([0, 1], ["apo-leaning", "holo-leaning"], fontsize=TICK_FS)
    ax.set_yticks([0, 1], ["AF-Holo", "AF-Apo"] if ylabels else ["", ""], fontsize=TICK_FS)
    if ylabels:
        ax.set_ylabel("classification", fontsize=LAB_FS, labelpad=1)
    ax.set_title(DISP.get(method, method), fontsize=TITLE_FS, pad=4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def panel_a(fig, spec, qt: pd.DataFrame, adjust: str = "raw", standalone: bool = False):
    gs = GridSpecFromSubplotSpec(2, 5, subplot_spec=spec, height_ratios=[1.0, 0.085],
                                 hspace=0.62, wspace=0.16)
    axes = []
    for i, m in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, i])
        cmap, vlim = (DOCK_CMAP, DOCK_VLIM) if m == "docking" else (PRED_CMAP, PRED_VLIM)
        im = draw_tile(ax, _row(qt, m, adjust), m, cmap, vlim, ylabels=(i == 0))
        axes.append((ax, im, m))
    # a shared bar for the four comparable detectors, a separate one for docking
    cax1 = fig.add_subplot(gs[1, 0:4])
    cb1 = fig.colorbar(axes[0][1], cax=cax1, orientation="horizontal")
    cb1.set_label("harmonized DCA/TNS success (%)", fontsize=SMALL_FS, labelpad=2)
    cax2 = fig.add_subplot(gs[1, 4])
    cb2 = fig.colorbar(axes[-1][1], cax=cax2, orientation="horizontal")
    cb2.set_label("dockable (%)", fontsize=SMALL_FS, labelpad=2)
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=TICK_FS - 0.6, length=2)
        cb.outline.set_visible(False)
    # x-axis meaning, written once under the row rather than five times
    axes[2][0].set_xlabel("evidence balance", fontsize=LAB_FS, labelpad=2)
    axes[-1][0].text(1.06, 0.5, "different readout\n& denominator", transform=axes[-1][0].transAxes,
                     rotation=90, ha="center", va="center", fontsize=SMALL_FS - 0.6,
                     color="0.45", style="italic")
    if not standalone:
        _letter(axes[0][0], "a", dx=-0.30, dy=1.14)
    note = ("dashed outline = the two DISCORDANT quadrants; their difference (Q4 − Q2) is "
            + ("the preference index PREF" if standalone else "panel c"))
    cax1.text(0.0, -2.3, note, transform=cax1.transAxes, ha="left", va="top",
              fontsize=SMALL_FS - 0.6, color="0.45", style="italic")
    return axes


# --------------------------------------------------------- panel b: the two main effects
def panel_b(ax, qt: pd.DataFrame):
    y = np.arange(len(METHODS))[::-1].astype(float)
    h = 0.34
    for off, key, col, lab in [(+h / 2, "E_ratio", RATIO_C, "evidence balance"),
                               (-h / 2, "E_class", CLASS_C, "classification")]:
        vals = [_row(qt, m, "raw")[key] for m in METHODS]
        lo = [_row(qt, m, "raw").get(f"{key}_lo", np.nan) for m in METHODS]
        hi = [_row(qt, m, "raw").get(f"{key}_hi", np.nan) for m in METHODS]
        ax.barh(y + off, vals, h, color=col, edgecolor="0.25", lw=.5, label=lab, zorder=2)
        ax.errorbar(vals, y + off, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                    fmt="none", ecolor="0.2", elinewidth=.9, capsize=2, zorder=3)
        # the two size adjustments, drawn as open markers on the raw bar: they agree closely,
        # which is the point -- the shrinkage is not an artifact of either estimator
        for adj_key, mk in [("size_std", "o"), ("size_resid", "s")]:
            adj = [_row(qt, m, adj_key)[key] for m in METHODS]
            ax.scatter(adj, y + off, s=26, marker=mk, facecolors="white", edgecolors="0.15",
                       lw=1.0, zorder=4,
                       label=(ADJ_LAB[adj_key] if key == "E_ratio" else None))
            for v, a_, yy in zip(vals, adj, y + off):
                ax.annotate("", xy=(a_, yy), xytext=(v, yy),
                            arrowprops=dict(arrowstyle="-", color="0.35", lw=.7, ls=":"), zorder=3)
    ax.set_yticks(y, [DISP.get(m, m) for m in METHODS], fontsize=TICK_FS)
    ax.set_xlabel("main effect on success (pp)", fontsize=LAB_FS)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.30)          # headroom: the legend must clear the bars
    ax.axvline(0, color="0.4", lw=.8)
    ax.grid(axis="x", color="#ececec", lw=.7, ls="--")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=TICK_FS, length=0)
    ax.legend(frameon=False, fontsize=SMALL_FS, loc="lower right", handlelength=1.3)
    ax.set_title("How much each facilitator buys", fontsize=TITLE_FS, pad=4)
    _letter(ax, "b", dx=-0.30)


# ------------------------------------------------------------- panel c: the PREF forest
def panel_c(ax, qt: pd.DataFrame):
    y = np.arange(len(METHODS))[::-1].astype(float)
    for off, adj, face, mk in [(+0.22, "raw", "0.15", "o"),
                               (0.0, "size_std", "white", "o"),
                               (-0.22, "size_resid", "white", "s")]:
        v = np.array([_row(qt, m, adj).PREF for m in METHODS])
        lo = np.array([_row(qt, m, adj).get("PREF_lo", np.nan) for m in METHODS])
        hi = np.array([_row(qt, m, adj).get("PREF_hi", np.nan) for m in METHODS])
        ax.errorbar(v, y + off, xerr=[v - lo, hi - v], fmt=mk, ms=6,
                    mfc=face, mec="0.15", mew=1.1, ecolor="0.3", elinewidth=1.1, capsize=2.5,
                    label=ADJ_LAB[adj], zorder=3)
    ax.axvline(0, color="0.3", lw=1.0)
    lim = ax.get_xlim()
    pad = 0.06 * (lim[1] - lim[0])
    ax.set_xlim(min(lim[0], -3) - pad, lim[1] + pad)
    ax.axvspan(ax.get_xlim()[0], 0, color=CLASS_C, alpha=0.06, zorder=0)
    ax.axvspan(0, ax.get_xlim()[1], color=RATIO_C, alpha=0.06, zorder=0)
    # NB: no arrow glyphs -- Helvetica Neue has none, and the fallback swap is visible in print
    ax.text(0.99, 1.015, "evidence balance stronger", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=SMALL_FS, color=RATIO_C, fontweight="bold")
    ax.text(0.01, 1.015, "classification stronger", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=SMALL_FS, color=CLASS_C, fontweight="bold")
    ax.set_yticks(y, [DISP.get(m, m) for m in METHODS], fontsize=TICK_FS)
    ax.set_xlabel("PREF = $E_\\mathrm{ratio}-E_\\mathrm{class}$ = Q4 − Q2  (pp)", fontsize=LAB_FS)
    ax.grid(axis="x", color="#ececec", lw=.7, ls="--")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=TICK_FS, length=0)
    ax.legend(frameon=False, fontsize=SMALL_FS, loc="lower right", handlelength=1.0)
    ax.set_title("Which facilitator is stronger", fontsize=TITLE_FS, pad=12)
    _letter(ax, "c", dx=-0.26)


# --------------------------------------------------------------------------- SI figure
def fig_si(qt: pd.DataFrame, outdir: _Path, name: str):
    fig = plt.figure(figsize=(11.6, 7.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.92], hspace=0.50, wspace=0.34)
    panel_a(fig, gs[0, :], qt)
    panel_b(fig.add_subplot(gs[1, 0]), qt)
    panel_c(fig.add_subplot(gs[1, 1]), qt)
    savef(fig, outdir, name)


# ------------------------------------------------------------- internal reference figures
def fig_tiles_only(qt: pd.DataFrame, outdir: _Path, adjust: str = "raw",
                   name: str = "quadrants_tiles"):
    fig = plt.figure(figsize=(12.2, 3.4))
    gs = GridSpec(1, 1, figure=fig)
    panel_a(fig, gs[0, 0], qt, adjust=adjust, standalone=True)
    ttl = ("Pooled success by quadrant of evidence balance x classification"
           + ("" if adjust == "raw" else f" - {ADJ_LAB[adjust]}"))
    fig.suptitle(ttl, fontsize=TITLE_FS + 1, y=1.10)
    savef(fig, outdir, name, svg=True)


def fig_scheme_robustness(qt: pd.DataFrame, comp: pd.DataFrame, outdir: _Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax = axes[0]
    y = np.arange(len(METHODS))[::-1].astype(float)
    mk = {"sign": "o", "median": "s", "quartile": "^"}
    for si, s in enumerate(SCHEMES):
        for ai, (adj, face) in enumerate([("raw", "0.15"), ("size_std", "white"),
                                          ("size_resid", "0.62")]):
            off = 0.30 - si * 0.20 - ai * 0.055
            v = [_row(qt, m, adj, s).PREF for m in METHODS]
            ax.scatter(v, y + off, marker=mk[s], s=34, facecolors=face, edgecolors="0.15", lw=1.0,
                       label=f"{s} / {adj.replace('size_', '')}", zorder=3)
    ax.axvline(0, color="0.3", lw=1.0)
    ax.set_yticks(y, [DISP.get(m, m) for m in METHODS], fontsize=TICK_FS)
    ax.set_xlabel("PREF (pp); > 0: evidence balance stronger", fontsize=LAB_FS)
    ax.grid(axis="x", color="#ececec", lw=.7, ls="--")
    ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, fontsize=SMALL_FS - 1.2, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.20))          # below the axis: the panel has no free corner
    ax.set_title("Cut-scheme sensitivity", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS, length=0)

    ax = axes[1]
    c = comp[(comp.scheme == "sign") & (comp.population == "lbs")].sort_values("quadrant")
    x = np.arange(len(c))
    # colour by the EVIDENCE side, not the class side: that is the axis pocket size tracks
    cols = {"Q1": RATIO_C, "Q4": RATIO_C, "Q2": "#c9cdd4", "Q3": "#c9cdd4"}
    ax.bar(x, c.mean_union_rg, 0.62, color=[cols[q] for q in c.quadrant], edgecolor="0.25", lw=.6)
    ax.text(0.5, 0.02, "purple = holo-leaning evidence (Q1, Q4)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=SMALL_FS - 0.4, color="0.35", style="italic")
    for xi, (rg, nres) in enumerate(zip(c.mean_union_rg, c.mean_union_residues)):
        ax.text(xi, rg + 0.06, f"{rg:.2f} Å\n{nres:.1f} res", ha="center", va="bottom",
                fontsize=SMALL_FS - 0.4, color="0.25")
    ax.set_xticks(x, c.quadrant, fontsize=TICK_FS)
    ax.set_ylabel("mean union $R_g$ (Å)", fontsize=LAB_FS)
    ax.set_ylim(0, max(c.mean_union_rg) * 1.28)
    ax.set_title("Why the raw effect shrinks: quadrant size composition", fontsize=TITLE_FS)
    ax.grid(axis="y", color="#ececec", lw=.7, ls="--")
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=TICK_FS)
    fig.suptitle("[INTERNAL] Robustness of the quadrant preference read", fontsize=TITLE_FS + 1.5,
                 y=1.03)
    fig.tight_layout()
    savef(fig, outdir, "quadrants_scheme_robustness", dpi=200, svg=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=None,
                    help="default: work_dir/lbs_generality/analysis_v1/tables")
    ap.add_argument("--outdir", default=None,
                    help="default: work_dir/lbs_generality/analysis_v1/plots")
    ap.add_argument("--manuscript_dir", default=None, help="also copy the SI figure + caption here")
    ap.add_argument("--manuscript_name", default="SI_ratio_vs_class_quadrants")
    a = ap.parse_args()

    cfg.require("WORK_DIR")
    tdir = _Path(a.tables) if a.tables else (cfg.LBS_ANALYSIS_DIR / "tables")
    outdir = _Path(a.outdir) if a.outdir else (cfg.LBS_ANALYSIS_DIR / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    qt = pd.read_csv(tdir / "quadrants_summary.csv")
    comp = pd.read_csv(tdir / "quadrants_composition.csv")

    fig_si(qt, outdir, a.manuscript_name)
    (outdir / f"{a.manuscript_name}_caption.txt").write_text(CAPTION, encoding="utf-8")
    fig_tiles_only(qt, outdir)
    for adj in ("size_std", "size_resid"):
        fig_tiles_only(qt, outdir, adjust=adj, name=f"quadrants_tiles_{adj}")
    fig_scheme_robustness(qt, comp, outdir)

    if a.manuscript_dir:
        mdir = _Path(a.manuscript_dir)
        mdir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            src = outdir / f"{a.manuscript_name}.{ext}"
            if src.is_file():
                shutil.copy2(src, mdir / src.name)
        shutil.copy2(outdir / f"{a.manuscript_name}_caption.txt",
                     mdir / f"{a.manuscript_name}_caption.txt")
        print("copied SI figure + caption ->", mdir)
    print("wrote ->", outdir)


if __name__ == "__main__":
    main()
