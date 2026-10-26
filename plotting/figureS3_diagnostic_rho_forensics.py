# -*- coding: utf-8 -*-
"""
Figure S3 diagnostic — evidence base for the bin-level rho caveat. NOT an SI figure.

Answers, in one 6-panel sheet, what "rho = 0.86 is a bin-level correlation" does
and does not license:
  A/B  pLDDT q10 (quoted) vs q5 — deeper is NOT sharper (0.80 -> 0.67 pooled)
  C    rho vs tail depth — signal is tail-specific, dies at the median (0.08)
  D    rho vs bin count — 0.62 (90 bins) to 0.89 (6 bins); the number is partly
       a binning choice
  E    pockets vs distinct proteins per bin — extreme bins are a few proteins
  F    rho with per-pocket bins vs one value per protein (pooled 0.80 -> 0.54)

Companion to figureS3_q10_ratio_evidence_REV13.py (the actual SI figure).
Run with the `plotting` conda env. Writes one PNG next to the S3 outputs.
"""
import sys as _sys
import os
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from common import config as cfg

X_MIN, X_MAX = -3.0, 3.0
THRESHOLDS = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050", "ST_030", "ST_LJ"]
COLORS = {"UNCLUSTERED": "#4c78a8", "ST_100": "#f58518", "ST_070": "#54a24b",
          "ST_050": "#e45756", "ST_030": "#72b7b2", "ST_LJ": "#b279a2"}
LABEL = {"UNCLUSTERED": "Unclustered", "ST_100": "UBS - ST 100%", "ST_070": "UBS - ST 70%",
         "ST_050": "UBS - ST 50%", "ST_030": "UBS - ST 30%", "ST_LJ": "UBS - ST_LJ"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42, "axes.linewidth": 0.8})

df = pd.read_csv(cfg.ANALYSIS_UNIFIED_LEVELS_CSV, low_memory=False)
df.loc[df.threshold.astype(str).str.startswith("ST_LJ"), "threshold"] = "ST_LJ"


def subset_threshold(df, thr):
    lvl = "entry" if thr == "UNCLUSTERED" else "ubs"
    s = df[(df.threshold.astype(str) == thr) & (df.level.astype(str).str.lower() == lvl)].copy()
    s["pocket_class"] = s.pocket_class.astype(str).str.strip().str.lower()
    s = s[s.pocket_class.isin(["apo", "holo"])]
    for c in ["evidence_n_apo", "evidence_n_holo", "model_pocket_plddt"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s[np.isfinite(s.evidence_n_apo) & np.isfinite(s.evidence_n_holo)]
    s = s[s.evidence_n_apo >= 1].copy()
    s["adj"] = s.evidence_n_holo + 1.0
    s["lr"] = np.log10(s.adj / s.evidence_n_apo)
    s = s[np.isfinite(s.lr)]
    s = s[(s.lr >= X_MIN) & (s.lr <= X_MAX)].copy()
    return s[np.isfinite(s.model_pocket_plddt)].copy()


subs = {t: subset_threshold(df, t) for t in THRESHOLDS}
E18 = np.linspace(X_MIN, X_MAX, 19)
DEPTHS = [1, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 40, 50]


def bin_tbl(s, edges, qs, dedup=False):
    b = pd.cut(s.lr, bins=edges, include_lowest=True, labels=False)
    rows = []
    for i in range(len(edges) - 1):
        d = s[b == i]
        if len(d) == 0:
            continue
        pl = (d.groupby("uniprots_str").model_pocket_plddt.median().to_numpy(float)
              if dedup else d.model_pocket_plddt.to_numpy(float))
        pl = pl[np.isfinite(pl)]
        if pl.size < 2:
            continue
        r = {"n": len(d), "n_prot": d.uniprots_str.nunique(), "n_units": pl.size,
             "x": float(np.median(np.abs(d.lr))),
             "x_signed": float(0.5 * (edges[i] + edges[i + 1]))}
        for q in qs:
            r[f"q{q}"] = float(np.percentile(pl, q))
        rows.append(r)
    return pd.DataFrame(rows)


summ = {t: bin_tbl(subs[t], E18, DEPTHS) for t in THRESHOLDS}
pooled18 = pd.concat(summ.values(), ignore_index=True)
msize = lambda n: 14.0 + 0.62 * np.sqrt(np.maximum(np.asarray(n, float), 0.0))

fig = plt.figure(figsize=(13.6, 7.9))
gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.27)
axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
axD, axE, axF = (fig.add_subplot(gs[1, i]) for i in range(3))

# ---- A / B ------------------------------------------------------------------
for ax, q, title in [(axA, "q10", "A   pLDDT q10 vs |ratio|  (as published)"),
                     (axB, "q5", "B   pLDDT q5 vs |ratio|  (requested)")]:
    ax.set_axisbelow(True); ax.grid(True, color="0.90", lw=0.7)
    for t in THRESHOLDS:
        g = summ[t].sort_values("x")
        ax.plot(g["x"], g[q], color=COLORS[t], lw=0.7, alpha=0.38, zorder=1)
        ax.scatter(g["x"], g[q], s=msize(g["n"]), color=COLORS[t], alpha=0.86,
                   edgecolor="white", linewidth=0.35, zorder=3)
    r, p = spearmanr(pooled18["x"], pooled18[q])
    ax.text(0.03, 0.97, f"POOLED $\\rho$ = {r:.2f}\np = {p:.1e}\nn = {len(pooled18)} bins",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", lw=0.6))
    ax.set_xlim(-0.02, 3.10); ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Absolute evidence ratio, |log10(holo/apo)|", fontsize=9)
    ax.set_ylabel(f"pLDDT {q}", fontsize=9)
    ax.set_title(title, fontsize=9.5, loc="left", fontweight="bold")
    ax.tick_params(labelsize=7.5)
ylo = min(axA.get_ylim()[0], axB.get_ylim()[0]); yhi = max(axA.get_ylim()[1], axB.get_ylim()[1])
axA.set_ylim(ylo, yhi); axB.set_ylim(ylo, yhi)
axA.legend(handles=[Line2D([0], [0], marker="o", color=COLORS[t], lw=1.0, markersize=4.5,
                           label=LABEL[t]) for t in THRESHOLDS],
           loc="lower right", fontsize=6.0, frameon=True, ncol=2)
w = summ["UNCLUSTERED"].nsmallest(1, "q5").iloc[0]
axB.annotate("347 pockets from ONE protein\n(Q9H2K2); 277 at identical\npLDDT 67.93 — q5 lands in the spike",
             xy=(w["x"], w["q5"]), xytext=(0.30, 0.18), textcoords="axes fraction", fontsize=6.4,
             arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"), color="0.2",
             bbox=dict(boxstyle="round,pad=0.25", fc="#fff6f6", ec="#c44", lw=0.6))

# ---- C : tail depth ---------------------------------------------------------
axC.set_axisbelow(True); axC.grid(True, color="0.90", lw=0.7)
for t in THRESHOLDS:
    axC.plot(DEPTHS, [spearmanr(summ[t]["x"], summ[t][f"q{q}"])[0] for q in DEPTHS],
             color=COLORS[t], lw=0.9, alpha=0.45, marker="o", ms=2.5)
pooled_rhos = [spearmanr(pooled18["x"], pooled18[f"q{q}"])[0] for q in DEPTHS]
axC.plot(DEPTHS, pooled_rhos, color="black", lw=2.0, marker="o", ms=4.5, label="POOLED", zorder=5)
for xv, c, lab in [(10, "0.45", "q10 (published)"), (5, "#c44", "q5 (requested)")]:
    axC.axvline(xv, color=c, ls="--", lw=0.9)
    axC.text(xv + 0.4, -0.12, lab, fontsize=6.4, color=c, rotation=90, va="bottom")
axC.axhline(0, color="0.55", lw=0.7)
axC.set_xlabel("Tail depth — pLDDT percentile used as bin summary", fontsize=9)
axC.set_ylabel("Bin-level Spearman $\\rho$ vs |ratio|", fontsize=9)
axC.set_title("C   Signal is tail-specific; median shows nothing", fontsize=9.5,
              loc="left", fontweight="bold")
axC.tick_params(labelsize=7.5); axC.legend(fontsize=6.5, loc="upper right"); axC.set_ylim(-0.2, 1.0)

# ---- D : binning sensitivity ------------------------------------------------
axD.set_axisbelow(True); axD.grid(True, color="0.90", lw=0.7)
NB = [4, 6, 9, 12, 18, 24, 30, 40, 60, 90]
for q, col in [("q10", "0.15"), ("q5", "#c44")]:
    ys = []
    for nb in NB:
        e = np.linspace(X_MIN, X_MAX, nb + 1)
        allb = pd.concat([bin_tbl(subs[t], e, (5, 10)) for t in THRESHOLDS], ignore_index=True)
        ys.append(spearmanr(allb["x"], allb[q])[0])
    axD.plot(NB, ys, marker="o", ms=4, lw=1.6, color=col, label=q)
axD.axvline(18, color="0.45", ls="--", lw=0.9)
axD.text(18.7, 0.05, "18 bins (published)", fontsize=6.4, color="0.35", rotation=90, va="bottom")
axD.axhline(0.86, color="#2a7", ls=":", lw=1.3)
axD.text(92, 0.868, "0.86 as cited", fontsize=6.6, color="#2a7", ha="right", va="bottom")
axD.set_xscale("log"); axD.set_xticks(NB); axD.set_xticklabels([str(n) for n in NB], fontsize=7)
axD.set_xlabel("Number of ratio bins (equal width over [-3, 3])", fontsize=9)
axD.set_ylabel("Pooled bin-level Spearman $\\rho$", fontsize=9)
axD.set_title("D   The same q10, binned differently", fontsize=9.5, loc="left", fontweight="bold")
axD.tick_params(axis="y", labelsize=7.5); axD.legend(fontsize=7, loc="lower left")
axD.set_ylim(0.0, 1.0)

# ---- E : pseudo-replication -------------------------------------------------
axE.set_axisbelow(True); axE.grid(True, color="0.90", lw=0.7)
u = summ["UNCLUSTERED"].sort_values("x_signed")
axE.plot(u["x_signed"], u["n"], marker="o", ms=4, lw=1.5, color="#4c78a8", label="pockets in bin")
axE.plot(u["x_signed"], u["n_prot"], marker="s", ms=4, lw=1.5, color="#c44",
         label="distinct proteins in bin")
axE.set_yscale("log")
axE.set_xlabel("Signed evidence ratio, log10(holo/apo)", fontsize=9)
axE.set_ylabel("Count in bin (log)", fontsize=9)
axE.set_title("E   Extreme bins are a few proteins, repeated", fontsize=9.5,
              loc="left", fontweight="bold")
axE.tick_params(labelsize=7.5); axE.legend(fontsize=7, loc="lower center")
axE.set_xlim(-3.1, 3.1); axE.set_xticks([-3, -2, -1, 0, 1, 2, 3])
axE.annotate("586 pockets\nfrom 6 proteins\n(one = 82%)",
             xy=(u["x_signed"].iloc[-1], u["n_prot"].iloc[-1]),
             xytext=(0.60, 0.62), textcoords="axes fraction", fontsize=6.4,
             arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"), color="0.2",
             bbox=dict(boxstyle="round,pad=0.25", fc="#fff6f6", ec="#c44", lw=0.6))
axE.annotate("15 pockets\nfrom 9 proteins",
             xy=(u["x_signed"].iloc[0], u["n_prot"].iloc[0]),
             xytext=(0.06, 0.62), textcoords="axes fraction", fontsize=6.4,
             arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"), color="0.2",
             bbox=dict(boxstyle="round,pad=0.25", fc="#f6f9ff", ec="#4c78a8", lw=0.6))

# ---- F : dedup effect -------------------------------------------------------
axF.set_axisbelow(True); axF.grid(True, axis="y", color="0.90", lw=0.7)
rows = []
for t in THRESHOLDS:
    a = summ[t]
    b_ = bin_tbl(subs[t], E18, (5, 10), dedup=True)
    rows.append((LABEL[t], spearmanr(a["x"], a["q10"])[0], spearmanr(b_["x"], b_["q10"])[0]))
A = pooled18
B = pd.concat([bin_tbl(subs[t], E18, (5, 10), dedup=True) for t in THRESHOLDS], ignore_index=True)
rows.append(("POOLED", spearmanr(A["x"], A["q10"])[0], spearmanr(B["x"], B["q10"])[0]))
xs = np.arange(len(rows)); wdt = 0.36
axF.bar(xs - wdt / 2, [r[1] for r in rows], wdt, color="0.35", label="per-pocket bins (published)")
axF.bar(xs + wdt / 2, [r[2] for r in rows], wdt, color="#c44", alpha=0.85,
        label="one value per protein")
for i, r in enumerate(rows):
    axF.text(i - wdt / 2, r[1] + 0.015, f"{r[1]:.2f}", ha="center", fontsize=5.8)
    axF.text(i + wdt / 2, r[2] + 0.015, f"{r[2]:.2f}", ha="center", fontsize=5.8)
axF.axhline(0.86, color="#2a7", ls=":", lw=1.3)
axF.text(len(rows) - 0.4, 0.868, "0.86 as cited", fontsize=6.6, color="#2a7", ha="right", va="bottom")
axF.set_xticks(xs); axF.set_xticklabels([r[0] for r in rows], fontsize=6.3, rotation=30, ha="right")
axF.set_ylabel("Bin-level Spearman $\\rho$ (q10)", fontsize=9)
axF.set_title("F   Collapsing each protein to one value", fontsize=9.5, loc="left", fontweight="bold")
axF.tick_params(axis="y", labelsize=7.5); axF.legend(fontsize=6.5, loc="lower left")
axF.set_ylim(0, 1.0)

fig.suptitle("Figure S3 diagnostic — what the bin-level $\\rho$ = 0.86 does and does not mean",
             fontsize=12, fontweight="bold", y=0.985)
outdir = cfg.SI_PLOTS_DIR / "S3"
os.makedirs(outdir, exist_ok=True)
base = str(outdir / "FigureS3_DIAG_q5_vs_q10_tail_depth")
fig.savefig(base + ".png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("wrote:", base + ".png")
