# -*- coding: utf-8 -*-
"""
The ratio x class success SURFACES -- raw and size-adjusted, side by side.
=========================================================================

This is the contour panel of the combined predictor + docking figure, and the continuous
counterpart of `plot_ratio_vs_class_quadrants.py` (which collapses the same surfaces to 2x2 tiles).
Both read the same per-pocket tables; keep their conclusions in sync.

    x = evidence balance   log2((n_holo_avg + 1) / n_apo_avg)
    y = p_holo             the classifier's holo posterior
    z = success            harmonized DCA/TNS (detectors) or "any cognate ligand docked" (docking)

Two surfaces per method
-----------------------
**raw** -- the observable. This is what someone actually meets when they point a pocket predictor
or a docking run at an AlphaFold model, and it is the one the main figure should carry.

**size-adjusted** -- success net of what pocket size alone predicts:

    value(cell) = grand_mean + mean_over_cell( s_i - E[s | size_bin(i)] )

i.e. INDIRECT (residual) standardization. Pocket size dominates detection here (rho +0.59 with
union_rg, +0.64 with the union residue count) and holo-leaning sites are systematically bigger, so
the raw surface confounds "more holo evidence" with "bigger pocket". The adjusted surface asks
whether the evidence balance still buys anything once that is removed. It does -- but the surface
also changes SHAPE, from near-vertical contours to diagonal ones, which is the substantive result.

Why INDIRECT and not the direct standardization used for the quadrant tiles: at this grid
resolution direct standardization runs out of pockets. The median (grid cell x size decile) holds
4 pockets and 52% hold fewer than 5; even with 3 strata only 81% of cells have every stratum
occupied. Indirect standardization needs only the GLOBAL size->success curve, so it is defined in
every cell. The two agree to 1-2 pp where both are computable (quadrant level; see
`analyze_ratio_vs_class_quadrants.py`, which reports both).

Reading the adjusted panels
---------------------------
The adjustment removes most of the DYNAMIC RANGE (consensus Q1-Q3 span 36.7 -> 13.7 pp) while the
per-point noise falls only slightly, so signal-to-noise drops ~17 -> ~8.5. Consequence for the
figure: an adjusted surface plotted on the raw panel's colour scale is a flat wash, and one shared
scale across the adjusted row is nearly as bad, because the adjustment leaves the five methods with
quite different residual spreads (23 to 35 pp).

**Every panel therefore carries its own colour bar, directly beneath it and column-aligned with its
twin in the other row.** What those bars mean differs by row, and that difference is deliberate:

  row a (raw)          the four detectors share ONE range (20-80%), so their bars are identical and
                       their colours ARE comparable across methods -- the project's contour
                       convention, which the main figure depends on. Docking keeps its own range
                       and colormap: different readout, different denominator.
  row b (adjusted)     each panel is stretched to its OWN 2-98 percentile range, which restores the
                       gradient in every panel at the cost of cross-method colour comparability.
                       `--adjusted_scale shared` gives the conservative variant instead.

The differing tick ranges under row b, against the identical ones under row a, make the distinction
self-documenting. **Never let a reader compare panels by colour across rows, or (in row b) across
methods** -- the contour LABELS carry absolute % and are what to compare.

Deliverables
------------
  `SI_ratio_class_surfaces`     the side-by-side: 5 methods x {raw, size-adjusted}   <- the SI figure
  `surface_<method>`            per-method standalone raw contour (main-figure source panels)
  `surface_<method>_adjusted`   per-method standalone adjusted contour

Inputs  : work_dir/lbs_generality/p2rank/lbs_p2rank_st100_merged_per_pocket.csv
          work_dir/lbs_generality/{fpocket,ifsitepred}/full/lbs_*_st100_merged_per_pocket.csv
          work_dir/docking/run3/analysis/pocket_level_table.csv          (ligand_group == 1)
Outputs : work_dir/lbs_generality/analysis_v1/plots/*.{png,pdf,svg}

Base env (pandas + numpy + scipy + matplotlib):
    python 06_lbs_generality/analysis/plot_ratio_class_surfaces.py \
        [--size_col union_rg] [--manuscript_dir "<...>/manuscript/figures/SI"]
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
from scipy.ndimage import distance_transform_edt, gaussian_filter

for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
# one loader for both figures: the surfaces and the tiles must never drift apart
from analyze_ratio_vs_class_quadrants import (LBS_METHODS, METHODS, SIZE_COL_DEFAULT,  # noqa: E402
                                              SIZE_COLS, load_long)

DISP = {"consensus": "Consensus", "docking": "Docking"}
# detectors share one scale (comparable readout); docking is a different quantity on a different
# denominator and keeps its own colormap, exactly as in the quadrant tiles
PRED_CMAP, PRED_VLIM = "viridis", (0.20, 0.80)
DOCK_CMAP, DOCK_VLIM = "magma", (0.30, 0.90)
XLIM, YLIM = (-4.0, 4.0), (0.0, 1.0)
XBINS, YBINS, SIGMA = 46, 38, 1.25
N_SIZE_BINS = 20            # finer than the quadrant tiles: the global curve can afford it

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
TITLE_FS, LAB_FS, TICK_FS, SMALL_FS, LET_FS = 10.5, 9.2, 8.0, 7.2, 12

CAPTION = """\
Supplementary Figure Sx | Removing pocket size flattens the success surface and turns it from
evidence-dominated to jointly graded, without abolishing the evidence gradient.

Success over evidence balance (x) and classification posterior (y) for the three LBS predictors,
their consensus, and docking. Top row: raw rates -- the observable, and the surface carried by the
main figure. Bottom row: the same data with pocket size removed, plotted as success net of what size
alone predicts, grand mean + mean(success - E[success | union-Rg decile]); no pockets are discarded,
the adjustment reweights rather than subsets. The three LBS predictors are scored on the identical
set of n = 86,946 AlphaFold ST_100 pockets (harmonized heavy-atom DCA/TNS); "Consensus" is their
per-pocket mean, a derived summary rather than an independent fourth method; docking is a different
readout (any cognate ligand docked) on its own nested population (n = 60,893). Dotted lines mark the
two cut points used for the quadrant summary (evidence balance 0, p_holo 0.5). Surfaces are mean
success per 46 x 38 cell, Gaussian-smoothed (sigma = 1.25 cells); cells with no support are filled
from the nearest supported cell for display only.

**Every panel carries its own colour bar, and the two rows use them differently** (stated on the
bars). In the top row the four detectors share one range, so their colours are comparable across
methods. In the bottom row each panel is stretched to its own range: the adjustment removes most of
the dynamic range -- the consensus corner-to-corner span falls from 36.7 to 13.7 percentage points,
and the residual spreads differ between methods -- so any shared range flattens some panels into a
wash. Compare panels across rows, and across methods within the bottom row, by the contour labels,
which are absolute percentages, never by colour.

Raw, the contours run close to vertical: the evidence balance dominates. Size-adjusted, they run
diagonally -- both axes grade success and neither dominates -- while the left-to-right gradient
survives, i.e. holo-leaning experimental evidence still marks a more detectable pocket after its
larger size is accounted for.

[INTERNAL NOTE -- not for the caption] The shape change is the whole reason this figure exists, and
it is consistent with the quadrant analysis: at the matched-extreme (quartile) cut with size held
fixed, the difference between the two facilitators is statistically indistinguishable from zero for
P2Rank, IF-SitePred and the consensus. Do NOT describe the adjusted surface as "the ratio still
wins" -- at equal lever arms it does not, for two of the three predictors. Direct standardization
is NOT usable at this grid resolution (median 4 pockets per cell x size decile); that is why the
tiles use it and the surfaces use the residual form. Their quadrant-level agreement (1-2 pp) is
what licenses mixing the two across the two figures.
"""


# --------------------------------------------------------------------------- estimation
def size_adjusted(df: pd.DataFrame, col: str, size_col: str, k: int = N_SIZE_BINS) -> np.ndarray:
    """success - E[success | size stratum] + grand mean.

    Strata are quantiles of `size_col` within this table. `union_n_residues_found` is integer and
    ties collapse quantile edges, so duplicate edges are dropped rather than left empty. Rows with
    no size are returned as NaN and drop out of the surface.
    """
    s = df[col].to_numpy(float)
    v = df[size_col].to_numpy(float)
    out = np.full(len(s), np.nan)
    ok = np.isfinite(v) & np.isfinite(s)
    if not ok.any():
        return out
    edges = np.unique(np.nanpercentile(v[ok], np.linspace(0, 100, k + 1)[1:-1]))
    z = np.searchsorted(edges, v, side="right")
    exp = np.array([s[ok & (z == j)].mean() if (ok & (z == j)).any() else np.nan
                    for j in range(int(z[ok].max()) + 1)])
    out[ok] = s[ok] - exp[z[ok]] + s[ok].mean()
    return out


def surface(xv, yv, sv):
    """Smoothed mean-success surface + the raw per-cell counts (for the support diagnostics)."""
    m = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(sv)
    xv, yv, sv = np.clip(xv[m], *XLIM), yv[m], sv[m]
    xe = np.linspace(*XLIM, XBINS + 1)
    ye = np.linspace(*YLIM, YBINS + 1)
    ssum, _, _ = np.histogram2d(yv, xv, bins=[ye, xe], weights=sv)
    cnt, _, _ = np.histogram2d(yv, xv, bins=[ye, xe])
    valid = cnt > 0
    filled = np.where(valid, np.divide(ssum, cnt, out=np.zeros_like(ssum), where=valid), 0.0)
    num = gaussian_filter(filled * valid, sigma=SIGMA)
    den = gaussian_filter(valid.astype(float), sigma=SIGMA)
    rate = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0.015)
    # fill the remaining unsupported cells from the nearest supported one -- DISPLAY ONLY, so the
    # panel has no white holes; every such cell is outside the data's support and must not be read
    nan = ~np.isfinite(rate)
    if nan.any():
        rate = rate[tuple(distance_transform_edt(nan, return_distances=False, return_indices=True))]
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    return np.meshgrid(xc, yc), rate, cnt


def auto_vlim(R: np.ndarray, step: float = 0.02, lo_pct=2, hi_pct=98) -> tuple[float, float]:
    """Colour limits rounded outward to a clean multiple -- adjusted surfaces need their own."""
    lo = np.floor(np.nanpercentile(R, lo_pct) / step) * step
    hi = np.ceil(np.nanpercentile(R, hi_pct) / step) * step
    return round(float(lo), 3), round(float(hi), 3)


# --------------------------------------------------------------------------- drawing
def draw_surface(ax, XX, YY, R, vlim, cmap, *, title="", show_y=True, show_x=True,
                 label_step=None):
    lo, hi = vlim
    step = (hi - lo) / 12.0
    label_step = label_step or max(step * 2, 0.02)
    cf = ax.contourf(XX, YY, R, levels=np.linspace(lo, hi, 13), cmap=cmap,
                     vmin=lo, vmax=hi, extend="both")
    cl = [v for v in np.round(np.arange(lo, hi + 1e-9, label_step), 3) if lo < v < hi]
    if cl:
        cs = ax.contour(XX, YY, R, levels=cl, colors="white", linewidths=.7, alpha=.9)
        ax.clabel(cs, inline=True, fmt=lambda v: f"{int(round(v * 100))}", fontsize=6.3)
    ax.axvline(0, color="white", lw=1.0, ls=":", alpha=.75)
    ax.axhline(0.5, color="white", lw=1.0, ls=":", alpha=.75)
    if title:
        ax.set_title(title, fontsize=TITLE_FS, pad=4)
    ax.set_ylabel(r"$p_\mathrm{holo}$", fontsize=LAB_FS) if show_y else ax.set_yticklabels([])
    if show_x:
        ax.set_xlabel(r"$\log_2$ (holo+1)/apo", fontsize=LAB_FS)
    ax.tick_params(labelsize=TICK_FS)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return cf


def savef(fig, outdir: _Path, name: str, dpi=300):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name, "(png,pdf,svg)")


def build(long: pd.DataFrame, size_col: str):
    """Per method: the raw and adjusted surfaces plus their support counts."""
    out = {}
    for m in METHODS:
        df = long[long["method"] == m]
        x = df["log2_ratio"].to_numpy(float)
        y = df["pocket_p_holo"].to_numpy(float)
        (XX, YY), R_raw, cnt = surface(x, y, df["success"].to_numpy(float))
        (_, _), R_adj, _ = surface(x, y, size_adjusted(df, "success", size_col))
        out[m] = dict(XX=XX, YY=YY, raw=R_raw, adj=R_adj, cnt=cnt, n=len(df))
    return out


def fig_side_by_side(S: dict, outdir: _Path, name: str, size_col: str,
                     adjusted_scale: str = "per_panel"):
    """The SI deliverable: 5 methods across, raw over size-adjusted.

    EVERY PANEL CARRIES ITS OWN COLOUR BAR, directly beneath it and the same width, so the ten bars
    line up in five columns and each panel's scale is unambiguous.

    The two rows differ in what those scales mean, deliberately:

      row a (raw)  -- the four detectors share ONE range (20-80%), so their bars are identical and
                      their colours ARE comparable across methods. That is the project's contour
                      convention and the main figure depends on it. Docking keeps its own range and
                      colormap because it is a different readout on a different denominator.

      row b (size-adjusted, `adjusted_scale='per_panel'`) -- each panel is stretched to its OWN
                      2-98 percentile range. The adjustment leaves the five methods with quite
                      different residual spreads (23 to 35 pp), so one shared range washes the
                      smaller ones out; per-panel scales restore the gradient in every panel at the
                      cost of cross-method colour comparability. The contour LABELS stay absolute,
                      so nothing is lost that a reader cannot recover -- and the differing tick
                      ranges under row b vs the identical ones under row a make the distinction
                      self-documenting.

    Pass `adjusted_scale='shared'` for the conservative variant where row b also shares one range
    across the detectors (comparable, but visibly flatter).
    """
    fig = plt.figure(figsize=(15.2, 9.0))
    gs = GridSpec(4, 5, figure=fig, height_ratios=[1.0, 0.055, 1.0, 0.055],
                  hspace=0.32, wspace=0.16)
    shared_adj = auto_vlim(np.concatenate([S[m]["adj"].ravel() for m in LBS_METHODS]))
    first_cax = {}                      # col-0 bar per row, so the row note can be hung off it
    for ri, mode in enumerate(["raw", "adj"]):
        row = ri * 2
        for col, m in enumerate(METHODS):
            d = S[m]
            dock = m == "docking"
            cmap = DOCK_CMAP if dock else PRED_CMAP
            if mode == "raw":
                vlim = DOCK_VLIM if dock else PRED_VLIM
            elif dock or adjusted_scale == "per_panel":
                vlim = auto_vlim(d["adj"])                    # this panel's own range
            else:
                vlim = shared_adj
            ax = fig.add_subplot(gs[row, col])
            cf = draw_surface(ax, d["XX"], d["YY"], d[mode], vlim, cmap,
                              title=(f"{DISP.get(m, m)}\nn = {d['n']:,}" if row == 0 else ""),
                              show_y=(col == 0), show_x=True)
            if col == 0:
                ax.text(-0.36, 0.5, "raw" if mode == "raw" else "size-adjusted",
                        transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=LET_FS, fontweight="bold")
                ax.text(-0.30, 1.16 if row == 0 else 1.05, "a" if mode == "raw" else "b",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=LET_FS, fontweight="bold")
            # this panel's own bar, directly beneath it and column-aligned with its twin
            cax = fig.add_subplot(gs[row + 1, col])
            ticks = np.linspace(vlim[0], vlim[1], 5)
            cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=ticks)
            cb.ax.set_xticklabels([f"{int(round(t * 100))}" for t in ticks])
            cb.ax.tick_params(labelsize=TICK_FS - 1.2, length=2, pad=1.5)
            cb.outline.set_visible(False)
            if col == 0 or dock:
                if dock:
                    qual = "own scale"
                elif mode == "raw" or adjusted_scale == "shared":
                    qual = "shared across detectors"
                else:
                    qual = "each panel its own scale"
                cb.set_label(f"{'dockable' if dock else 'success'} (%) · {qual}",
                             fontsize=SMALL_FS - 0.9, labelpad=1)
                if col == 0:
                    first_cax[mode] = cax
    per_panel = adjusted_scale == "per_panel"
    fig.text(0.5, first_cax["adj"].get_position().y0 - 0.055,
             # kept to ~3 short lines: one long line forces bbox_inches="tight" to widen the whole
             # figure and squashes the panels
             f"size adjustment: success − E[success | {size_col} decile] — no pockets discarded.\n"
             "Row a: the four detectors share one range, so their colours are comparable.  "
             + ("Row b: each panel is stretched to its own range, because the adjustment leaves\n"
                "the methods with different residual spreads.  " if per_panel else
                "Row b: detectors share one range.\n")
             + "Across rows — and in b across methods — compare the contour labels "
               "(absolute percentages), not the colours.",
             ha="center", va="top", fontsize=SMALL_FS - 0.4, color="0.45", style="italic")
    savef(fig, outdir, name)


def fig_standalone(S: dict, outdir: _Path):
    """Per-method single panels -- the source panels for the main figure and for slides."""
    for m in METHODS:
        d = S[m]
        dock = m == "docking"
        for mode, suffix in [("raw", ""), ("adj", "_adjusted")]:
            vlim = ((DOCK_VLIM if dock else PRED_VLIM) if mode == "raw"
                    else auto_vlim(d["adj"]))
            fig, ax = plt.subplots(figsize=(5.4, 4.3))
            cf = draw_surface(ax, d["XX"], d["YY"], d[mode], vlim,
                              DOCK_CMAP if dock else PRED_CMAP,
                              title=f"{DISP.get(m, m)}{' — size-adjusted' if mode == 'adj' else ''}"
                                    f"  (n = {d['n']:,})")
            ticks = np.linspace(vlim[0], vlim[1], 7)
            cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, ticks=ticks)
            cb.ax.set_yticklabels([f"{int(round(t*100))}" for t in ticks])
            cb.set_label("dockable (%)" if dock else "success (%)", fontsize=LAB_FS)
            cb.ax.tick_params(labelsize=TICK_FS)
            savef(fig, outdir, f"surface_{m.lower().replace('-', '')}{suffix}")


def support_table(S: dict, min_n: int = 10) -> pd.DataFrame:
    """Grid support + how much dynamic range the adjustment removes -- the numbers in the caption.

    Range is the 5-95 percentile spread over WELL-SUPPORTED cells only (>= `min_n` pockets). Plain
    min-max over the whole surface is meaningless here: the sparse corners are nearest-filled for
    display and their extremes are noise, which makes the adjustment look like it removed almost
    nothing (~90% "retained") when the supported surface actually loses well over half its spread.
    """
    rows = []
    for m in METHODS:
        d = S[m]
        occ = d["cnt"][d["cnt"] > 0]
        good = d["cnt"] >= min_n
        rec = {"method": m, "n_pockets": d["n"],
               "cells_occupied": int((d["cnt"] > 0).sum()), "cells_total": d["cnt"].size,
               "cells_ge_min_n": int(good.sum()), "min_n": min_n,
               "median_n_per_cell": float(np.median(occ)),
               "pct_cells_lt10": 100 * float((occ < 10).mean())}
        for mode in ("raw", "adj"):
            v = d[mode][good]
            lo, hi = np.nanpercentile(v, [5, 95])
            rec[f"{mode}_range_pp"] = 100 * float(hi - lo)
        rows.append(rec)
    t = pd.DataFrame(rows)
    t["range_retained_pct"] = 100 * t.adj_range_pp / t.raw_range_pp
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=None,
                    help="default: work_dir/lbs_generality/analysis_v1/plots")
    ap.add_argument("--size_col", default=SIZE_COL_DEFAULT, choices=SIZE_COLS,
                    help="size variable the adjustment conditions on (default: union_rg)")
    ap.add_argument("--manuscript_dir", default=None, help="also copy the SI figure + caption here")
    ap.add_argument("--manuscript_name", default="SI_ratio_class_surfaces")
    ap.add_argument("--no_standalone", action="store_true",
                    help="skip the per-method single panels")
    ap.add_argument("--adjusted_scale", default="per_panel", choices=("per_panel", "shared"),
                    help="row b colour scaling: 'per_panel' stretches each method to its own range "
                         "(gradients legible, colours not comparable across methods); 'shared' puts "
                         "the four detectors on one range (comparable, visibly flatter)")
    a = ap.parse_args()

    cfg.require("WORK_DIR")
    outdir = _Path(a.outdir) if a.outdir else (cfg.LBS_ANALYSIS_DIR / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    long = load_long()
    S = build(long, a.size_col)
    sup = support_table(S)
    pd.set_option("display.width", 200)
    print(sup.round(1).to_string(index=False))
    sup.to_csv(outdir.parent / "tables" / f"surfaces_support_{a.size_col}.csv", index=False)

    fig_side_by_side(S, outdir, a.manuscript_name, a.size_col, adjusted_scale=a.adjusted_scale)
    # the conservative counterpart, for the record: row b on one shared detector range
    if a.adjusted_scale == "per_panel":
        fig_side_by_side(S, outdir, f"{a.manuscript_name}_shared_scale", a.size_col,
                         adjusted_scale="shared")
    (outdir / f"{a.manuscript_name}_caption.txt").write_text(CAPTION, encoding="utf-8")
    if not a.no_standalone:
        fig_standalone(S, outdir)

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
