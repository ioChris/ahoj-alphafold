# -*- coding: utf-8 -*-
"""
Supplementary figure — the AF-Holo P2Rank detection advantage is not a pocket-size artefact.

For each pocket-size proxy (site atoms, site residues, radius of gyration) the classified ST_100
pockets are split into size quintiles; within each quintile P2Rank's success rate is computed
separately for AF-Apo and AF-Holo. The AF-Holo advantage persists in every size stratum, so the
class effect is not merely a restatement of pocket size.

SUCCESS CRITERION (`CRITERION`, or `--criterion` on the command line)
  harmonized  (default since 2026-10-06) the manuscript's tool-agnostic DCA4-TNS: distance from the
              predicted pocket centre to the nearest heavy atom of the site's union residues on the
              model <= 4 A, pocket ranked within the top n. Column `success_tns` of
              cfg.lbs_tool_dir("p2rank") / lbs_p2rank_st100_merged_per_pocket.csv, the table SI
              Table S2 and Figure 3 read (AFDB v4: AF-Apo 47.3%, AF-Holo 63.5%).
  native      P2Rank's own evaluation (SAS points as site atoms): `success_tns` of
              cfg.P2RANK_MERGED_PER_POCKET_CSV (AFDB v4: 66.3% / 84.8%). It inflates AlphaFold
              detection and is not comparable across tools (manuscript Methods), so it is kept only
              as a labelled reference. Until 2026-10-06 the SI figure was drawn on this criterion
              while its caption said DCA4-TNS.

SIZE PROXIES are read from the P2Rank per-pocket table under both criteria (joined on
unit_uid = site_uid), so only the outcome changes between them:
  ligand_atoms   NOT a ligand: P2Rank's sites file `#atoms`, the atom count of the site's own
                 residues (8.0 atoms per residue; plan_v4/O6_O8_DECISIONS.md §O8). Labelled
                 "Site atoms" (it was "Ligand atoms" until 2026-10-06).
  site_residues  residues of the site (equal to the union residue count on the model)
  rg             union radius of gyration, the size covariate used everywhere else

Reports, per size proxy:
  - apo/holo success rate and counts per quintile, and the holo-apo delta
  - first-order partial Spearman rho(is_holo, success | size)
  - Cochran-Mantel-Haenszel test (holo vs apo success, stratified on size quintile)
and, once, the gap standardized to the union-Rg deciles with a cluster-bootstrap CI (the
manuscript's estimator; on the harmonized criterion it should agree with SI Table S10c's P2Rank row).

Drawn at its true size (180 mm, common/figspec.py); panel letters a-c.
Outputs: PNG/PDF/SVG + a tidy CSV of the per-quintile deltas and stats, in cfg.P2RANK_PLOTS_DIR /
cfg.P2RANK_TABLES_DIR; run with AHOJ_AF_MODEL_SET=afdbv4 for the manuscript's AFDB v4 set.
Spyder-friendly: settings are module globals; the command line only overrides CRITERION.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, chi2

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg
from common import figspec as fs

# ----------------------------------------------------------------------------- settings
CRITERION = "harmonized"          # "harmonized" (DCA4-TNS, the manuscript's) | "native" (P2Rank's own)

SIZE_SRC = str(cfg.P2RANK_MERGED_PER_POCKET_CSV)   # size proxies, and the native outcome
CRITERIA = {
    # name: (outcome table, outcome column, output prefix, y-axis label)
    "harmonized": (str(cfg.lbs_tool_dir("p2rank") / "lbs_p2rank_st100_merged_per_pocket.csv"),
                   "success_tns", "p2rank_dca4tns_size_controlled_class",
                   "P2Rank DCA4-TNS success (%)"),
    "native": (SIZE_SRC, "success_tns", "p2rank_tns_size_controlled_class",
               "P2Rank native TNS success (%)"),
}
OUT_PLOT_DIR = str(cfg.P2RANK_PLOTS_DIR)
OUT_TABLE_DIR = str(cfg.P2RANK_TABLES_DIR)

CLASS_COL = "pocket_class"
# (column, axis title, tick format of the quintile median)
SIZE_PROXIES = [
    ("ligand_atoms", "Site atoms", "{:.0f}"),
    ("site_residues", "Site residues", "{:.0f}"),
    ("rg", "Radius of gyration (Å)", "{:.1f}"),
]
NQ = 5  # quintiles

SAVE_PNG, SAVE_PDF, SAVE_SVG = True, True, True
DPI_PNG = 600

FIG_W, FIG_H = fs.W2, 62.0        # mm, the size it occupies on the page
fs.apply_rc()

# ----------------------------------------------------------------------------- helpers
def partial_spearman_xyz(x, y, z):
    """First-order partial Spearman corr of x,y controlling z."""
    rxy = spearmanr(x, y).correlation
    rxz = spearmanr(x, z).correlation
    ryz = spearmanr(y, z).correlation
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return np.nan if denom == 0 else (rxy - rxz * ryz) / denom

def cmh_test(a, b, c, d):
    """Cochran-Mantel-Haenszel chi-square (with continuity correction).
    Per stratum 2x2:  [[a(holo,success), b(holo,fail)], [c(apo,success), d(apo,fail)]]."""
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))  # float: avoid int32 overflow
    n = a + b + c + d
    num = np.sum(a - (a + b) * (a + c) / n)
    den = np.sum((a + b) * (c + d) * (a + c) * (b + d) / (n**2 * (n - 1)))
    stat = (abs(num) - 0.5)**2 / den
    return stat, chi2.sf(stat, 1)

def size_std_gap(d, K=10, B=400, seed=20260908):
    """Holo - apo gap (points) by direct adjustment to the union-Rg decile distribution of the whole
    set, with a cluster bootstrap over UniProt -- the manuscript's estimator (Methods; Figure 4,
    SI Table S10c). Printed as a cross-check, not drawn."""
    zb = pd.qcut(d["rg"], K, labels=False, duplicates="drop").to_numpy()
    ok = ~np.isnan(zb)
    zb, y, ish = zb[ok].astype(int), d["success"].to_numpy(float)[ok], d["is_holo"].to_numpy()[ok]
    cl = pd.factorize(d["uniprot"].to_numpy()[ok])[0]
    nC = cl.max() + 1
    lab = (cl * K + zb) * 2 + ish
    s = np.bincount(lab, weights=y, minlength=nC * K * 2).reshape(nC, K * 2)
    n = np.bincount(lab, minlength=nC * K * 2).astype(float).reshape(nC, K * 2)
    w = np.bincount(zb, minlength=K) / len(zb)

    def gap(cw):
        S, N = (cw @ s).reshape(K, 2), (cw @ n).reshape(K, 2)
        both = (N > 0).all(axis=1)
        mu = S[both] / N[both]
        return 100 * np.sum(w[both] * (mu[:, 1] - mu[:, 0])) / w[both].sum()

    rng = np.random.default_rng(seed)
    bs = np.array([gap(np.bincount(rng.integers(0, nC, nC), minlength=nC).astype(float))
                   for _ in range(B)])
    return gap(np.ones(nC)), np.percentile(bs, 2.5), np.percentile(bs, 97.5), nC

def load(criterion):
    """Classified pockets (AF-Apo / AF-Holo; ties dropped) with the outcome and the size proxies."""
    src, col, _, _ = CRITERIA[criterion]
    size_cols = [c for c, _, _ in SIZE_PROXIES]
    sz = pd.read_csv(SIZE_SRC, low_memory=False, usecols=["unit_uid", CLASS_COL, col] + size_cols)
    sz["unit_uid"] = sz["unit_uid"].astype(str).str.strip()
    if criterion == "native":
        df = sz.rename(columns={col: "success"})
    else:
        oc = pd.read_csv(src, low_memory=False, usecols=["site_uid", CLASS_COL, col])
        oc["site_uid"] = oc["site_uid"].astype(str).str.strip()
        df = oc.rename(columns={col: "success"}).merge(
            sz[["unit_uid"] + size_cols], left_on="site_uid", right_on="unit_uid",
            how="left", validate="one_to_one")
        n_miss = int(df[size_cols].isna().any(axis=1).sum())
        print(f"{criterion}: {len(oc):,} sites, {n_miss:,} without size proxies in {SIZE_SRC}")
    df = df[df[CLASS_COL].astype(str).str.lower().isin(["apo", "holo"])].copy()
    df["uniprot"] = df["unit_uid"].str.split(":").str[0]
    df["is_holo"] = (df[CLASS_COL].str.lower() == "holo").astype(int)
    df["success"] = pd.to_numeric(df["success"], errors="coerce")
    df = df[np.isfinite(df["success"])].copy()
    df["success"] = (df["success"] > 0).astype(int)
    return df

# ----------------------------------------------------------------------------- main
def main(criterion=CRITERION):
    _, _, out_prefix, ylab = CRITERIA[criterion]
    os.makedirs(OUT_PLOT_DIR, exist_ok=True)
    os.makedirs(OUT_TABLE_DIR, exist_ok=True)

    df = load(criterion)
    n_apo_all, n_holo_all = int((df.is_holo == 0).sum()), int((df.is_holo == 1).sum())
    overall_apo = 100 * df.loc[df.is_holo == 0, "success"].mean()
    overall_holo = 100 * df.loc[df.is_holo == 1, "success"].mean()

    rows = []
    fig, axes = plt.subplots(1, len(SIZE_PROXIES), figsize=(FIG_W * fs.MM, FIG_H * fs.MM), sharey=True)
    fig.subplots_adjust(**fs.margins(FIG_W * fs.MM, FIG_H * fs.MM, left=11, right=2, top=9, bottom=9),
                        wspace=0.10)

    for ax, letter, (col, label, tick_fmt) in zip(axes, "abc", SIZE_PROXIES):
        d = df[np.isfinite(pd.to_numeric(df[col], errors="coerce"))].copy()
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d["qbin"] = pd.qcut(d[col].rank(method="first"), NQ, labels=False)

        xs = np.arange(1, NQ + 1)
        apo_rate, holo_rate, medians = [], [], []
        A, B, C, D = [], [], [], []
        for q in range(NQ):
            g = d[d.qbin == q]
            ga, gh = g[g.is_holo == 0], g[g.is_holo == 1]
            apo_rate.append(100 * ga["success"].mean())
            holo_rate.append(100 * gh["success"].mean())
            medians.append(float(g[col].median()))
            A.append(int(gh["success"].sum())); B.append(int((1 - gh["success"]).sum()))
            C.append(int(ga["success"].sum())); D.append(int((1 - ga["success"]).sum()))
            rows.append({"criterion": criterion, "size_proxy": col, "quintile": q + 1,
                         "size_min": float(g[col].min()), "size_median": medians[-1],
                         "size_max": float(g[col].max()),
                         "n_apo": len(ga), "n_holo": len(gh),
                         "apo_success_pct": apo_rate[-1], "holo_success_pct": holo_rate[-1],
                         "holo_minus_apo_pct": holo_rate[-1] - apo_rate[-1]})

        pr = partial_spearman_xyz(d.is_holo.values, d["success"].values, d[col].values)
        cmh_stat, cmh_p = cmh_test(A, B, C, D)
        for r in rows:
            if r["size_proxy"] == col:
                r["n_pockets"] = len(d)
                r["partial_rho_holo_given_size"] = pr
                r["cmh_chi2"] = cmh_stat; r["cmh_p"] = cmh_p

        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color=fs.FAINT, lw=0.5)
        ax.plot(xs, apo_rate, "-o", color=fs.APO, ms=3.0, lw=1.0, label=fs.LAB_APO)
        ax.plot(xs, holo_rate, "-o", color=fs.HOLO, ms=3.0, lw=1.0, label=fs.LAB_HOLO)
        for x, ar, hr in zip(xs, apo_rate, holo_rate):   # holo - apo, in points, above the upper point
            ax.annotate(f"{hr - ar:+.0f}", (x, max(ar, hr)), textcoords="offset points",
                        xytext=(0, 3.5), ha="center", va="bottom", fontsize=fs.FS_NOTE, color=fs.MUTE)
        ax.set_xlim(0.5, NQ + 0.5); ax.set_xticks(xs)
        ax.set_xticklabels([tick_fmt.format(m) for m in medians])
        ax.set_ylim(0, 105); ax.set_yticks(range(0, 101, 20))
        ax.set_xlabel(f"{label}, quintile median", fontsize=fs.FS_TITLE)
        ax.xaxis.set_label_coords(0.5, -0.165)           # baseline-aligned: the Å would drop c's label
        ax.xaxis.label.set_verticalalignment("baseline")
        if cmh_p < 1e-300:
            p_txt = "p < 10$^{-300}$"
        else:
            e = int(np.floor(np.log10(cmh_p)))
            p_txt = f"p = {cmh_p / 10**e:.0f} × 10$^{{{e}}}$"
        ax.text(0.97, 0.04, f"partial ρ = {pr:.2f}\nCMH {p_txt}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=fs.FS_NOTE, color=fs.INK, linespacing=1.25)
        fs.letter(ax, letter, dx=-0.13 if letter == "a" else -0.05, dy=1.03)

    axes[0].set_ylabel(ylab, fontsize=fs.FS_TITLE)
    handles = [Line2D([0], [0], color=fs.APO, marker="o", lw=1.0, ms=3.0,
                      label=f"{fs.LAB_APO} ({n_apo_all:,} pockets; overall {overall_apo:.1f}%)"),
               Line2D([0], [0], color=fs.HOLO, marker="o", lw=1.0, ms=3.0,
                      label=f"{fs.LAB_HOLO} ({n_holo_all:,} pockets; overall {overall_holo:.1f}%)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               fontsize=fs.FS_BODY, handlelength=2.0, columnspacing=2.5, borderaxespad=0.3)
    fs.check_size(fig, out_prefix)

    base = os.path.join(OUT_PLOT_DIR, out_prefix)
    if SAVE_PNG: fig.savefig(base + ".png", dpi=DPI_PNG)          # true size: no bbox_inches="tight"
    if SAVE_PDF: fig.savefig(base + ".pdf")
    if SAVE_SVG: fig.savefig(base + ".svg")
    plt.close(fig)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_TABLE_DIR, out_prefix + "_deltas.csv"), index=False)
    print(f"Criterion: {criterion} ({CRITERIA[criterion][0]} :: {CRITERIA[criterion][1]})")
    print("Pockets: %d AF-Apo, %d AF-Holo" % (n_apo_all, n_holo_all))
    print("Overall: AF-Apo %.2f%%  AF-Holo %.2f%%  (delta %.2f points)"
          % (overall_apo, overall_holo, overall_holo - overall_apo))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(out.to_string(index=False))
    summ = out.groupby("size_proxy", sort=False).agg(
        partial_rho=("partial_rho_holo_given_size", "first"), cmh_chi2=("cmh_chi2", "first"),
        cmh_p=("cmh_p", "first"), delta_min=("holo_minus_apo_pct", "min"),
        delta_max=("holo_minus_apo_pct", "max"))
    print("\nPer proxy (delta = holo - apo, points):")
    print(summ.to_string(float_format=lambda v: f"{v:.4g}"))
    print("partial rho range: %.3f to %.3f" % (summ.partial_rho.min(), summ.partial_rho.max()))
    g, lo, hi, nC = size_std_gap(df)
    print("Size-standardized gap (union-Rg deciles, direct adjustment; cluster bootstrap over "
          "%d UniProt, B = 400): %+.2f [%+.2f, %+.2f] points; raw %+.2f"
          % (nC, g, lo, hi, overall_holo - overall_apo))
    print("\nSaved:", base + ".png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--criterion", choices=sorted(CRITERIA), default=CRITERION,
                    help="success criterion (default: %(default)s)")
    main(ap.parse_known_args()[0].criterion)
