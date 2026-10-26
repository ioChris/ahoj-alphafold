# -*- coding: utf-8 -*-
"""
Manuscript v4 **Figure 5** -- "when can you trust it?": pLDDT tracks pocket FLEXIBILITY, not pocket
STATE.

    a  pocket pLDDT vs conformational diversity (mean pocket RMSD across the experimental conformers)
    b  pocket pLDDT vs |evidence ratio|, on the same y axis -- the flat comparator
    c  rho(pLDDT, diversity) within AlphaFold's own confidence bands, with the published disorder
       cut-offs marked

WHY PANEL C CARRIES THE RESULT
------------------------------
pLDDT is already an established disorder predictor and the binary cut-offs are published: DeepMind's
own guidance treats long stretches below 50 as disorder rather than structure; AlphaFold-disorder
scores residues as 1 - pLDDT; CAID-style MCC-optimised cuts land near 68 for AlphaFold2. **Every one
of those sits below 70.** The association measured here is STRONGEST ABOVE 90 -- more than twenty
points into the confidently ordered regime, and in the band with the least pLDDT variance, so
restriction of range works against it there rather than for it. Without panel c this figure restates
a known result at a smaller scale; with it, the claim is that conformational flexibility is encoded
in AlphaFold's confidence across scales, from disordered chains down to ordered but mobile pockets.

WHAT IS PLOTTED, AND WHAT IS DELIBERATELY NOT
---------------------------------------------
Two diversity measures are reported, never a composite: MEAN POCKET RMSD (the strongest, and the
direct analogue of Saldano's chain-level measure) and the CV OF RoG (the scale-free one). SD of
volume and of SASA are excluded on purpose -- their spread correlates with pocket size at rho +0.92
and +0.61, so they are size proxies wearing a flexibility label. A composite of all four is weaker
than either component; it would look more rigorous and carry less information.

The size control travels with the result, printed to the console and meant for the caption: the
partial correlation given pocket size is -0.479 against a zero-order -0.494, so this is not the
pocket-size artefact that the separation analysis turned out to be.

Set: the 237,922 unclustered analysis pockets (filtered_survivors.csv), joined to the run_3 model
metrics and per-pocket experimental spread.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

for _root in Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402
from common import figspec  # noqa: E402

MERGED = Path(cfg.CLASSIFICATION_MERGED_DIR)
SURVIVORS = Path(cfg.FILTERED_SURVIVORS_CSV)
QMETRICS = MERGED / "query_metrics_do_kde_plots_and_classification.csv"

# AlphaFold's own confidence bands, used as the x categories of panel c
BANDS = [("< 50", -np.inf, 50), ("50-70", 50, 70), ("70-90", 70, 90), ("> 90", 90, np.inf)]
# published disorder cut-offs, all of which sit BELOW the band where our effect is strongest
DISORDER_CUTS = [(50, "DeepMind\ndisorder guidance"), (68, "CAID-style MCC\noptimum (AF2)")]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "figure.dpi": 130,
})
TITLE_FS = figspec.FS_TITLE
LAB_FS = figspec.FS_TITLE
TICK_FS = figspec.FS_TICK
SMALL_FS = figspec.FS_NOTE
LET_FS = figspec.FS_LETTER
# green and purple, not green and brown: the two are told apart with red-green colour blindness too
# (Chloé Weiler, 2026-10-12). The ratio bars are open (pale fill, purple outline) against solid green,
# so they also differ in lightness, a cue that needs no colour vision at all; a white hatch did this
# before. Neither is the blue or orange that mean AF-Apo and AF-Holo elsewhere in the paper.
DIV_C, RATIO_C, RATIO_FILL = "#3B7A57", "#7B4F9E", "#E4D9EE"


def load():
    keep = ["job_name", "n_apo", "n_holo", "model_pocket_pLDDT_avg", "model_pocket_len"]
    d = pd.read_csv(SURVIVORS, low_memory=False, usecols=lambda c: c in keep)
    q = pd.read_csv(QMETRICS, low_memory=False,
                    usecols=["job_name", "pocket_rms_mean", "RoG_mean", "RoG_std_dev"])
    d = d.merge(q, on="job_name", how="left")
    d["plddt"] = pd.to_numeric(d["model_pocket_pLDDT_avg"], errors="coerce")
    d["diversity"] = pd.to_numeric(d["pocket_rms_mean"], errors="coerce")
    rog_m = pd.to_numeric(d["RoG_mean"], errors="coerce")
    d["cv_rog"] = pd.to_numeric(d["RoG_std_dev"], errors="coerce") / rog_m.where(rog_m > 0)
    # the evidence ratio exactly as the manuscript defines it
    n_apo = pd.to_numeric(d["n_apo"], errors="coerce")
    d["ratio"] = np.log2((pd.to_numeric(d["n_holo"], errors="coerce") + 1) / n_apo.where(n_apo > 0))
    d["abs_ratio"] = d["ratio"].abs()
    # NB not "size": DataFrame.size is an attribute, so d.size would silently be the
    # element count rather than this column
    d["pocket_size"] = pd.to_numeric(d["model_pocket_len"], errors="coerce")
    return d


def rho(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 100:
        return np.nan, 0
    r, _ = spearmanr(a[m], b[m])
    return r, int(m.sum())


def partial_rho(x, y, z):
    """Spearman partial correlation of x and y given z, via ranks + residualisation."""
    m = x.notna() & y.notna() & z.notna()
    rx, ry, rz = (v[m].rank().to_numpy(float) for v in (x, y, z))
    def resid(v):
        A = np.c_[np.ones_like(rz), rz]
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    r, _ = spearmanr(resid(rx), resid(ry))
    return r, int(m.sum())


def hexpanel(ax, x, y, xlabel, color, xlim=None, logx=False):
    m = x.notna() & y.notna()
    xs, ys = x[m].to_numpy(float), y[m].to_numpy(float)
    if xlim:
        k = (xs >= xlim[0]) & (xs <= xlim[1])
        xs, ys = xs[k], ys[k]
    hb = ax.hexbin(xs, ys, gridsize=58, bins="log", mincnt=1, cmap="Greys", linewidths=0)
    # median of y within x bins, so the trend is legible over the density
    edges = np.linspace(xs.min(), xs.max(), 31)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, len(edges) - 2)
    cen, med = [], []
    for b in range(len(edges) - 1):
        s = idx == b
        if s.sum() >= 100:
            cen.append((edges[b] + edges[b + 1]) / 2)
            med.append(np.median(ys[s]))
    ax.plot(cen, med, color=color, lw=2.6, solid_capstyle="round")
    ax.set_xlabel(xlabel, fontsize=LAB_FS)
    ax.tick_params(labelsize=TICK_FS)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return hb


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_basename", default="Figure5_plddt_reliability")
    a = ap.parse_args()

    d = load()
    print(f"analysis pockets: {len(d):,}  with pLDDT: {d.plddt.notna().sum():,}")

    r_div, n_div = rho(d.diversity, d.plddt)
    r_cv, n_cv = rho(d.cv_rog, d.plddt)
    r_rat, n_rat = rho(d.abs_ratio, d.plddt)
    pr_div, _ = partial_rho(d.diversity, d.plddt, d.pocket_size)
    pr_cv, _ = partial_rho(d.cv_rog, d.plddt, d.pocket_size)
    pr_rat, _ = partial_rho(d.abs_ratio, d.plddt, d.pocket_size)
    pr_ratdiv, _ = partial_rho(d.abs_ratio, d.plddt, d.diversity)
    print("\n=== numbers for the caption (re-derive from here, never from a note) ===")
    print(f"  rho(pLDDT, mean pocket RMSD) = {r_div:+.3f}   n={n_div:,}   partial|size {pr_div:+.3f}")
    print(f"  rho(pLDDT, CV RoG)           = {r_cv:+.3f}   n={n_cv:,}   partial|size {pr_cv:+.3f}")
    print(f"  rho(pLDDT, |evidence ratio|) = {r_rat:+.3f}   n={n_rat:,}   partial|size {pr_rat:+.3f}")
    print(f"  rho(pLDDT, |ratio|) controlling for diversity = {pr_ratdiv:+.3f}")

    rows = []
    for lab, lo, hi in BANDS:
        s = d[(d.plddt > lo) & (d.plddt <= hi)]
        rd, nd = rho(s.diversity, s.plddt)
        rr, _ = rho(s.abs_ratio, s.plddt)
        # n is the count the diversity rho was computed ON. Every analysis pocket has a mean
        # pocket RMSD, so today it equals the band size (the 1,835-pocket gap is in CV RoG,
        # n=236,087); taking it from rho() keeps the annotation right if that ever changes
        rows.append({"band": lab, "n_band": len(s), "n": nd, "plddt_sd": s.plddt.std(),
                     "rho_diversity": rd, "rho_abs_ratio": rr})
    bands = pd.DataFrame(rows)
    print("\n=== panel c: within AlphaFold's own confidence bands ===")
    print(bands.round(3).to_string(index=False))

    # ---------------------------------------------------------------- figure
    W_MM, H_MM = figspec.W2, 62.0
    fig, axes = plt.subplots(1, 3, figsize=(W_MM * figspec.MM, H_MM * figspec.MM))
    hb = hexpanel(axes[0], d.diversity, d.plddt, "Mean pocket RMSD across\nconformers (Å)",
                  DIV_C, xlim=(0, np.nanpercentile(d.diversity, 99)))
    axes[0].set_ylabel("Pocket mean pLDDT", fontsize=LAB_FS)
    axes[0].set_title("Conformational diversity", fontsize=TITLE_FS)
    axes[0].text(0.96, 0.06, f"$\\rho$ = {r_div:+.2f}", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=LAB_FS)
    hb_b = hexpanel(axes[1], d.abs_ratio, d.plddt, "|Evidence ratio| (log₂)", RATIO_C,
                    xlim=(0, np.nanpercentile(d.abs_ratio, 99)))
    # one colour scale for a and b, so the bar in a reads for both: each on its own, b's densest
    # hexagon (~2,000 pockets) was as black as a's (~4,000)
    norm = LogNorm(1, max(h.get_array().max() for h in (hb, hb_b)))
    for h in (hb, hb_b):
        h.set_norm(norm)
    axes[1].set_title("Evidence ratio", fontsize=TITLE_FS)
    axes[1].text(0.96, 0.06, f"$\\rho$ = {r_rat:+.2f}", transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=LAB_FS)
    for ax in axes[:2]:
        ax.set_ylim(40, 100)
    axes[1].tick_params(labelleft=False)

    # panel c -- the bands
    ax = axes[2]
    x = np.arange(len(bands))
    ax.bar(x - 0.19, bands.rho_diversity, 0.36, color=DIV_C, label="diversity")
    ax.bar(x + 0.19, bands.rho_abs_ratio, 0.36, color=RATIO_FILL, label="|evidence ratio|",
           edgecolor=RATIO_C, linewidth=1.1)
    ax.axhline(0, color="0.4", lw=0.9)
    # every published disorder cut-off lies inside the two LEFT bands; our effect peaks on the right
    ax.axvspan(-0.5, 1.5, color="#f3f0e8", zorder=0)
    # two lines: on one line this ran the width of both shaded bands and sat on the 50-70 bar
    ax.text(0.5, -0.45, "published disorder\ncut-offs", ha="center", va="center",
            fontsize=figspec.FS_FLOOR, color="0.45", linespacing=1.15)
    # n at a fixed row along the foot: on the bars these collided with each other and with the
    # shaded span's caption, and they sat on different sides depending on the sign
    for xi, nb in enumerate(bands.n):
        ax.text(xi, -0.70, f"n={nb:,}", ha="center", va="bottom",
                fontsize=SMALL_FS - 1.2, color="0.45")
    ax.set_xticks(x)
    ax.set_xticklabels(bands.band, fontsize=TICK_FS)
    ax.set_xlabel("AlphaFold confidence band", fontsize=LAB_FS)
    ax.set_ylabel("ρ with pocket pLDDT", fontsize=LAB_FS)
    ax.set_ylim(-0.72, 0.42)
    ax.set_xlim(-0.5, len(bands) - 0.5)
    # inside the axes: anchored above them it ran into the panel title. The upper-right
    # corner is empty here because the right-hand bars are negative.
    ax.legend(frameon=False, fontsize=figspec.FS_FLOOR, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), handlelength=1.0, handleheight=0.85,
              borderaxespad=0.15, labelspacing=0.28, handletextpad=0.5)
    ax.grid(axis="y", color="#ececec", lw=.7, ls="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_title("Strongest where AlphaFold\nis most confident", fontsize=TITLE_FS)

    for ax, L in zip(axes, "abc"):
        ax.text(-0.22, 1.06, L, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=LET_FS, fontweight="bold")

    # inset into panel a's sparse lower-right corner. As a figure-level axes between a and b it
    # landed on top of panel b's tick numbers.
    # fully inside the axes: as an inset reaching to 0.94 its label overhung the plot area
    # pushed right: at 0.40 the bar and its label sat over the dense left flank of the cloud
    cax = axes[0].inset_axes([0.56, 0.30, 0.40, 0.035])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    # every decade in range: on a bar this short the default locator labels every other one
    # (10^0, 10^2), leaving the densest band (10^3 up to vmax ~4,000) past the last label
    cb.set_ticks(10.0 ** np.arange(0, np.floor(np.log10(hb.norm.vmax)) + 1))
    cb.set_label("pockets per hexagon", fontsize=figspec.FS_FLOOR, labelpad=1.5)
    cb.ax.tick_params(labelsize=figspec.FS_FLOOR, length=1.8, pad=1)

    fig.subplots_adjust(**figspec.margins(W_MM * figspec.MM, H_MM * figspec.MM,
                                          left=12.0, right=3.0, top=8.0, bottom=14.0),
                        wspace=0.40)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{a.out_basename}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    bands.to_csv(out / f"{a.out_basename}_bands.csv", index=False)
    print("\nwrote", out / f"{a.out_basename}.png")


if __name__ == "__main__":
    main()
