# -*- coding: utf-8 -*-
"""
Figure S3 alternative — evidence ratio vs the FRACTION of pockets below a pLDDT cutoff.

Motivation
----------
The published panel A summarizes each bin with pLDDT q10 (a percentile). That is
hard to read ("what is a q10 of 87.6?") and invites the "low-confidence" wording
the data does not support: pooled q10 = 87.6 sits inside AlphaFold's *confident*
(70-90) band, and only 0.93% of pockets fall below 70.

Flipping the statistic — "what share of this bin's pockets fall below cutoff X?" —
measures the same thing with no percentile, on a natural axis (a percentage), at a
cutoff readers already know (90 = AF's "very high" boundary).

Panels
------
  A) % of pockets below pLDDT 90 vs |evidence ratio|   (pooled rho ~ -0.80)
  B) % of pockets below pLDDT 70 vs |evidence ratio|   (pooled rho ~  0.07 — nothing)
  C) Pooled rho as a function of the cutoff — where the signal lives

Sign: rho is NEGATIVE by construction. More extreme evidence ratio -> FEWER
low-pLDDT pockets, i.e. the same direction of effect as the rising q10 in REV13.

B is the honest control: at AF's actual low-confidence boundary (70) there is no
association at all. The signal is in the 85-95 range, not in a low-confidence
population.

Why cutoff 90 is the one to quote (run5, 2026-07-15)
----------------------------------------------------
It is nearly as strong as the published q10 AND far more robust to the protein
pseudo-replication that inflates q10. Collapsing each protein to one value:

    metric              per-pocket   per-protein   protein-clustered bootstrap
    pLDDT q10              0.801        0.540       0.664  [0.542, 0.771]
    % below 90            -0.795       -0.743      -0.685 [-0.778, -0.569]

q10 loses 0.26 to dedup; % below 90 loses 0.05, and for UNCLUSTERED it actually
strengthens (-0.701 -> -0.762). Deeper cutoffs are NOT robust (87.5: -0.767 ->
-0.484; 85: -0.733 -> -0.467) — 90 sits above the thin part of the tail, so the
fraction is large (~13% of pockets) and stably estimated.

90 is also externally motivated — it is AlphaFold's own published "very high"
boundary, not a threshold we tuned — even though it happens to be the sweep's
argmax too.

Run with the `plotting` conda env.
"""
import sys as _sys
import os
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from common import config as cfg

X_MIN, X_MAX = -3.0, 3.0
HOLO_PLUS_ONE = 1.0
NBINS = 18
EDGES = np.linspace(X_MIN, X_MAX, NBINS + 1)
THRESHOLDS = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050", "ST_030", "ST_LJ"]
CUTOFF_SWEEP = [95, 92.5, 90, 87.5, 85, 80, 75, 70, 60, 50]
HEADLINE_CUTOFF = 90

COLORS = {"UNCLUSTERED": "#4c78a8", "ST_100": "#f58518", "ST_070": "#54a24b",
          "ST_050": "#e45756", "ST_030": "#72b7b2", "ST_LJ": "#b279a2"}
LABEL = {"UNCLUSTERED": "Unclustered", "ST_100": "UBS - ST 100%", "ST_070": "UBS - ST 70%",
         "ST_050": "UBS - ST 50%", "ST_030": "UBS - ST 30%", "ST_LJ": "UBS - ST_LJ"}

TITLE_FS, TICK_FS, LAB_FS, SMALL_FS = 9.5, 7.5, 9.0, 6.4
GRID_COLOR = "0.90"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8,
})


def subset_threshold(df, thr):
    lvl = "entry" if thr == "UNCLUSTERED" else "ubs"
    s = df[(df["threshold"].astype(str) == str(thr)) &
           (df["level"].astype(str).str.lower() == lvl)].copy()
    s["pocket_class"] = s["pocket_class"].astype(str).str.strip().str.lower()
    s = s[s["pocket_class"].isin(["apo", "holo"])]
    for c in ["evidence_n_apo", "evidence_n_holo", "model_pocket_plddt"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s[np.isfinite(s["evidence_n_apo"]) & np.isfinite(s["evidence_n_holo"])]
    s = s[s["evidence_n_apo"] >= 1].copy()
    s["evidence_n_holo_adj"] = s["evidence_n_holo"] + HOLO_PLUS_ONE
    ratio = s["evidence_n_holo_adj"] / s["evidence_n_apo"]
    s["log10_ratio"] = np.log10(ratio.where(ratio > 0))
    s = s[np.isfinite(s["log10_ratio"])]
    s = s[(s["log10_ratio"] >= X_MIN) & (s["log10_ratio"] <= X_MAX)].copy()
    return s[np.isfinite(s["model_pocket_plddt"])].copy()


def frac_bins(sub, cutoffs):
    """Per ratio bin: median |ratio|, n, and % of pockets below each cutoff."""
    b = pd.cut(sub["log10_ratio"], bins=EDGES, include_lowest=True, labels=False)
    rows = []
    for i in range(NBINS):
        d = sub[b == i]
        if len(d) < 2:
            continue
        pl = d["model_pocket_plddt"].to_numpy(float)
        row = {"n": len(d), "x": float(np.median(np.abs(d["log10_ratio"])))}
        for c in cutoffs:
            row[f"pct_below_{c}"] = float((pl < c).mean() * 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def msize(n):
    return 14.0 + 0.62 * np.sqrt(np.maximum(np.asarray(n, float), 0.0))


def draw_frac_panel(ax, summ, pooled, cutoff, letter, title, show_ylabel=True):
    col = f"pct_below_{cutoff}"
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID_COLOR, lw=0.7)
    for t in THRESHOLDS:
        g = summ[t].sort_values("x")
        ax.plot(g["x"], g[col], color=COLORS[t], lw=0.7, alpha=0.38, zorder=1)
        ax.scatter(g["x"], g[col], s=msize(g["n"]), color=COLORS[t], alpha=0.86,
                   edgecolor="white", linewidth=0.35, zorder=3)
    rho, p = spearmanr(pooled["x"], pooled[col])
    ax.text(0.97, 0.97, f"Pooled ρ = {rho:.2f}\np = {p:.1e}\nn = {len(pooled)} bins",
            transform=ax.transAxes, ha="right", va="top", fontsize=SMALL_FS + 0.6,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.80", lw=0.6))
    ax.set_xlim(-0.02, 3.10)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Absolute evidence ratio, |log10(holo/apo)|", fontsize=LAB_FS)
    if show_ylabel:
        ax.set_ylabel(f"% of pockets with pLDDT < {cutoff}", fontsize=LAB_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_title(f"{letter}   {title}", fontsize=TITLE_FS, loc="left", fontweight="bold")


def main():
    df = pd.read_csv(cfg.ANALYSIS_UNIFIED_LEVELS_CSV, low_memory=False)
    df.loc[df["threshold"].astype(str).str.startswith("ST_LJ"), "threshold"] = "ST_LJ"
    subs = {t: subset_threshold(df, t) for t in THRESHOLDS}

    cutoffs = sorted(set(CUTOFF_SWEEP + [HEADLINE_CUTOFF, 70]))
    summ = {t: frac_bins(subs[t], cutoffs) for t in THRESHOLDS}
    pooled = pd.concat(summ.values(), ignore_index=True)

    fig = plt.figure(figsize=(11.4, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.30)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    draw_frac_panel(axA, summ, pooled, 90, "A", "Below AF's “very high” line (90)")
    draw_frac_panel(axB, summ, pooled, 70, "B", "Below AF's low-confidence line (70)")

    axA.legend(handles=[Line2D([0], [0], marker="o", color=COLORS[t], lw=1.0,
                               markersize=4.2, label=LABEL[t]) for t in THRESHOLDS],
               loc="lower left", fontsize=SMALL_FS - 0.5, frameon=True, ncol=2)

    # ---- C: rho vs cutoff ----
    axC.set_axisbelow(True)
    axC.grid(True, which="major", color=GRID_COLOR, lw=0.7)
    for t in THRESHOLDS:
        rs = [spearmanr(summ[t]["x"], summ[t][f"pct_below_{c}"])[0] for c in CUTOFF_SWEEP]
        axC.plot(CUTOFF_SWEEP, rs, color=COLORS[t], lw=0.9, alpha=0.45, marker="o", ms=2.4)
    pooled_rs = [spearmanr(pooled["x"], pooled[f"pct_below_{c}"])[0] for c in CUTOFF_SWEEP]
    axC.plot(CUTOFF_SWEEP, pooled_rs, color="black", lw=1.9, marker="o", ms=4.0,
             label="Pooled", zorder=5)
    axC.axhline(0, color="0.55", lw=0.7)
    axC.axvline(HEADLINE_CUTOFF, color="0.45", ls="--", lw=0.9)
    axC.text(HEADLINE_CUTOFF - 1.1, -0.92, "cutoff 90", fontsize=SMALL_FS,
             color="0.35", rotation=90, va="bottom")
    axC.axvline(70, color="#c44", ls="--", lw=0.9)
    axC.text(70 - 1.1, -0.92, "cutoff 70", fontsize=SMALL_FS, color="#c44",
             rotation=90, va="bottom")
    axC.invert_xaxis()
    axC.set_xlabel("pLDDT cutoff", fontsize=LAB_FS)
    axC.set_ylabel("Pooled Spearman ρ vs |ratio|", fontsize=LAB_FS)
    axC.set_ylim(-1.0, 0.35)
    axC.tick_params(labelsize=TICK_FS)
    axC.set_title("C   Where the signal lives", fontsize=TITLE_FS, loc="left", fontweight="bold")
    leg = axC.legend(loc="lower right", fontsize=SMALL_FS + 0.4, frameon=True)
    leg.get_frame().set_edgecolor("0.80")

    outdir = cfg.SI_PLOTS_DIR / "S3"
    os.makedirs(outdir, exist_ok=True)
    base = str(outdir / "FigureS3_ALT_fraction_below_cutoff")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base}.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    tbl = pd.DataFrame({"cutoff": CUTOFF_SWEEP, "pooled_rho": pooled_rs})
    tbl.to_csv(base + "_cutoff_sweep.csv", index=False)
    pooled.to_csv(base + "_bin_source_data.csv", index=False)

    print("wrote:", base + ".png")
    print("\nPooled Spearman rho vs |ratio|, by pLDDT cutoff:")
    for c, r in zip(CUTOFF_SWEEP, pooled_rs):
        print(f"   pLDDT < {c:<5} : rho = {r:+.3f}")
    print("\n% of pockets below cutoff, pooled over all thresholds/bins:")
    for c in [90, 70]:
        print(f"   < {c}: mean across bins = {pooled[f'pct_below_{c}'].mean():.2f}%  "
              f"(range {pooled[f'pct_below_{c}'].min():.2f} - {pooled[f'pct_below_{c}'].max():.2f}%)")


if __name__ == "__main__":
    main()
