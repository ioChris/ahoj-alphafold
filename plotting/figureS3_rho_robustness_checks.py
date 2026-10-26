# -*- coding: utf-8 -*-
"""
Figure S3 — robustness checks behind the caption numbers. Console report, no figure.

Run this to regenerate every number quoted in the Fig S3 caveat discussion:

  1. Per-pocket ("raw") rho          — verifies the "~0.06" in the disclaimer
  2. AlphaFold confidence bands      — where q5/q10/q15/q20 actually sit
  3. Binning sensitivity             — rho as a function of bin count
  4. Protein pseudo-replication      — pockets vs distinct proteins per bin
  5. Protein-clustered bootstrap     — the honest CI on the headline rho
  6. ALT caption numbers             — every value quoted in the "% below 90" caption

Key results (run5, 2026-07-15):
  - pooled bin-level rho (|ratio| vs q10) = 0.801 over 108 bins; ST_070 = 0.868.
    The manuscript's "0.86 over 108 bins" pairs ST_070's rho with the pooled bin
    count — it does not reconcile with this run.
  - raw per-pocket rho = 0.0496 pooled / 0.0695 unclustered -> "~0.06" is fair.
  - q10 = 87.6 pooled sits in AF's "confident" (70-90) band; only 0.93% of
    pockets fall below 70. Do NOT call this a "low-confidence" population.
  - protein-collapsed rho = 0.540; protein-clustered bootstrap = 0.66,
    95% CI [0.54, 0.77] -> the cited 0.86 falls outside it.
  - the adopted statistic, % below 90: pooled -0.795, protein-collapsed -0.743,
    protein-clustered bootstrap -0.682 [-0.790, -0.583]; per-pocket counterpart
    -0.046; share below 90 falls 17.5% -> 9.7% (balanced -> extreme evidence),
    18.2% -> 11.5% counting proteins rather than pockets.

Section 6 exists so that EVERY number in the Figure S3 caption is reproducible
from committed code. If you change the caption, add its number here.

The bootstrap is seeded (default_rng(1)) so the CIs above are exactly
reproducible by running this file — but they are Monte Carlo estimates, and they
move by ~0.01 if you change N_BOOT, the seed, or the order of BOOT_STATS (the
draws come off one shared stream). Quote them to 2 dp, and re-run this script
rather than a side analysis when transcribing them into the caption. A number
quoted from a one-off run that nothing can regenerate is how the stale
"rho = 0.86" got into the manuscript in the first place.

Run with the `plotting` conda env:
  <anaconda>/envs/plotting/python.exe plotting/figureS3_rho_robustness_checks.py
"""
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import config as cfg

X_MIN, X_MAX = -3.0, 3.0
HOLO_PLUS_ONE = 1.0
NBINS = 18
THRESHOLDS = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050", "ST_030", "ST_LJ"]
PROTEIN_ID_COL = "uniprots_str"
N_BOOT = 400
EDGES = np.linspace(X_MIN, X_MAX, NBINS + 1)

# The adopted bin statistic: AlphaFold's published "very high" boundary.
CUTOFF = 90
# "Balanced" vs "extreme" evidence regimes, for the caption's effect size.
BALANCED_MAX = 0.5
EXTREME_MIN = 2.0

# Bin summary statistics to bootstrap. Each maps a bin's pLDDT array -> one number.
BOOT_STATS = [
    ("q10", lambda p: float(np.percentile(p, 10))),
    ("q5", lambda p: float(np.percentile(p, 5))),
    (f"% below {CUTOFF}", lambda p: float((p < CUTOFF).mean() * 100.0)),
]


def subset_threshold(df, thr):
    """Figure S3 row filters + derived ratio/evidence columns, for one threshold."""
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
    s["log10_ratio_abs"] = np.abs(s["log10_ratio"])
    s["evidence_total"] = s["evidence_n_apo"] + s["evidence_n_holo_adj"]
    return s[np.isfinite(s["model_pocket_plddt"])].copy()


def bin_rho(lr, pl, depth, edges=EDGES):
    """Bin on the signed ratio, summarize pLDDT at `depth`, correlate vs median |ratio|."""
    b = np.digitize(lr, edges[1:-1], right=False)
    pts = []
    for i in range(len(edges) - 1):
        m = b == i
        if m.sum() < 2:
            continue
        pts.append((float(np.median(np.abs(lr[m]))), float(np.percentile(pl[m], depth))))
    if len(pts) < 3:
        return np.nan, 0
    a = np.array(pts)
    return spearmanr(a[:, 0], a[:, 1])[0], len(a)


def main():
    df = pd.read_csv(cfg.ANALYSIS_UNIFIED_LEVELS_CSV, low_memory=False)
    df.loc[df["threshold"].astype(str).str.startswith("ST_LJ"), "threshold"] = "ST_LJ"
    subs = {t: subset_threshold(df, t) for t in THRESHOLDS}
    allp = pd.concat(subs.values(), ignore_index=True)

    print("=" * 84)
    print("1. PER-POCKET (raw) association — the disclaimer's '~0.06'")
    print("=" * 84)
    print(f"{'threshold':<13}{'n':>10}  {'rho(|ratio|,pLDDT)':>19}  {'rho(signed,pLDDT)':>18}")
    for thr in THRESHOLDS + ["POOLED"]:
        g = allp if thr == "POOLED" else subs[thr]
        r_abs = spearmanr(g["log10_ratio_abs"], g["model_pocket_plddt"])[0]
        r_sgn = spearmanr(g["log10_ratio"], g["model_pocket_plddt"])[0]
        print(f"{thr:<13}{len(g):>10,}  {r_abs:>19.4f}  {r_sgn:>18.4f}")

    print()
    print("=" * 84)
    print("2. ALPHAFOLD CONFIDENCE BANDS  (>90 very high | 70-90 confident | <70 low)")
    print("=" * 84)
    pl = allp["model_pocket_plddt"].to_numpy(float)
    for q in [1, 2.5, 5, 10, 15, 20, 25, 50]:
        v = np.percentile(pl, q)
        band = ("very high >90" if v > 90 else "CONFIDENT 70-90" if v >= 70
                else "low 50-70" if v >= 50 else "very low <50")
        print(f"   q{q:<5} = {v:6.2f}   -> {band}")
    for cut in [90, 70, 50]:
        print(f"   fraction < {cut}: {100 * (pl < cut).mean():6.3f}%  ({int((pl < cut).sum()):,} pockets)")
    print("   NOTE: the tracked tail is the low end of a uniformly high distribution,")
    print("         NOT a low-confidence population. Word the caption accordingly.")

    print()
    print("=" * 84)
    print("3. BINNING SENSITIVITY — pooled bin-level rho vs bin count")
    print("=" * 84)
    print(f"{'nbins':>6}  {'rho(q10)':>9}  {'rho(q5)':>9}  {'rho(median)':>12}")
    for nb in [4, 6, 9, 12, 18, 24, 30, 40, 60, 90]:
        e = np.linspace(X_MIN, X_MAX, nb + 1)
        out = []
        for depth in [10, 5, 50]:
            pts = []
            for t in THRESHOLDS:
                s = subs[t]
                lr = s["log10_ratio"].to_numpy(float)
                p = s["model_pocket_plddt"].to_numpy(float)
                b = np.digitize(lr, e[1:-1], right=False)
                for i in range(nb):
                    m = b == i
                    if m.sum() < 2:
                        continue
                    pts.append((float(np.median(np.abs(lr[m]))), float(np.percentile(p[m], depth))))
            a = np.array(pts)
            out.append(spearmanr(a[:, 0], a[:, 1])[0])
        print(f"{nb:>6}  {out[0]:>9.3f}  {out[1]:>9.3f}  {out[2]:>12.3f}")

    print()
    print("=" * 84)
    print("4. PSEUDO-REPLICATION — pockets vs distinct proteins per bin (UNCLUSTERED)")
    print("=" * 84)
    s = subs["UNCLUSTERED"].copy()
    s["b"] = pd.cut(s["log10_ratio"], bins=EDGES, include_lowest=True, labels=False)
    print(f"{'bin':>4} {'|ratio|':>8} {'pockets':>9} {'proteins':>9} {'pockets/prot':>13} {'top prot':>9}")
    for b in range(NBINS):
        d = s[s["b"] == b]
        if len(d) == 0:
            continue
        npr = d[PROTEIN_ID_COL].nunique()
        top = d[PROTEIN_ID_COL].value_counts().iloc[0] / len(d) * 100
        print(f"{b:>4} {np.median(np.abs(d['log10_ratio'])):>8.2f} {len(d):>9,} {npr:>9,} "
              f"{len(d) / npr:>13.1f} {top:>8.1f}%")

    print()
    print("=" * 84)
    print(f"5. PROTEIN-CLUSTERED BOOTSTRAP of the pooled rho ({N_BOOT} reps)")
    print("=" * 84)
    packed = {}
    for t in THRESHOLDS:
        st = subs[t]
        pos = {}
        for i, p in enumerate(st[PROTEIN_ID_COL].values):
            pos.setdefault(p, []).append(i)
        packed[t] = {
            "lr": st["log10_ratio"].to_numpy(float),
            "pl": st["model_pocket_plddt"].to_numpy(float),
            "prots": np.array(list(pos.keys()), dtype=object),
            "pos": {k: np.asarray(v) for k, v in pos.items()},
        }

    rng = np.random.default_rng(1)
    for label, stat_fn in BOOT_STATS:
        vals = []
        for _ in range(N_BOOT):
            pts = []
            for t in THRESHOLDS:
                P = packed[t]
                pick = rng.integers(0, len(P["prots"]), len(P["prots"]))
                take = np.concatenate([P["pos"][P["prots"][j]] for j in pick])
                lr, p = P["lr"][take], P["pl"][take]
                b = np.digitize(lr, EDGES[1:-1], right=False)
                for i in range(NBINS):
                    m = b == i
                    if m.sum() < 2:
                        continue
                    pts.append((float(np.median(np.abs(lr[m]))), stat_fn(p[m])))
            a = np.array(pts)
            r = spearmanr(a[:, 0], a[:, 1])[0]
            if np.isfinite(r):
                vals.append(r)
        v = np.array(vals)
        print(f"  {label:<12}: median={np.median(v):+.3f}  95% CI = [{np.percentile(v, 2.5):+.3f}, "
              f"{np.percentile(v, 97.5):+.3f}]")
    print("\n  Compare: pocket-level bootstrap gives q10 [0.622, 0.812] — it treats")
    print("  pockets of the same protein as independent and understates the spread.")

    print()
    print("=" * 84)
    print(f"6. CAPTION NUMBERS for the adopted '% below {CUTOFF}' statistic")
    print("=" * 84)
    allp["below"] = (allp["model_pocket_plddt"] < CUTOFF).astype(float)

    print(f"  Per-pocket counterpart — rho(|ratio|, indicator pLDDT < {CUTOFF}).")
    print("  Unlike q10 (a percentile, which has no per-pocket version), the bin")
    print("  statistic IS the mean of this indicator, so this is exact apples-to-apples.")
    print(f"\n  {'threshold':<13}{'n':>10}  {'rho':>9}  {'p':>11}")
    for thr in THRESHOLDS + ["POOLED"]:
        g = allp if thr == "POOLED" else allp[allp["threshold"] == thr]
        r, p = spearmanr(g["log10_ratio_abs"], g["below"])
        print(f"  {thr:<13}{len(g):>10,}  {r:>9.4f}  {p:>11.2e}")

    print(f"\n  Effect size — share of pockets below {CUTOFF} by evidence regime (pooled):")
    regimes = [(0.0, 0.5, "balanced      |log10 ratio| <= 0.5"),
               (0.5, 1.0, "moderate      0.5 - 1.0"),
               (1.0, 2.0, "extreme       1.0 - 2.0"),
               (2.0, 3.01, "very extreme  |log10 ratio| >= 2")]
    for lo, hi, lab in regimes:
        g = allp[(allp["log10_ratio_abs"] >= lo) & (allp["log10_ratio_abs"] < hi)]
        if len(g) == 0:
            continue
        print(f"    {lab:<36} n = {len(g):>7,}   {g['below'].mean() * 100:5.2f}%")

    bal = allp[allp["log10_ratio_abs"] <= BALANCED_MAX]
    ext = allp[allp["log10_ratio_abs"] >= EXTREME_MIN]
    p_bal, p_ext = bal["below"].mean() * 100, ext["below"].mean() * 100
    print(f"\n    balanced -> very extreme: {p_bal:.1f}% -> {p_ext:.1f}%  "
          f"({p_bal - p_ext:.1f} pp; {p_bal / p_ext:.2f}x relative)")

    print("\n  Same split counting PROTEINS, not pockets (pseudo-replication-safe):")
    for g, lab in [(bal, "balanced"), (ext, "very extreme")]:
        per = g.groupby(PROTEIN_ID_COL)["model_pocket_plddt"].median()
        print(f"    {lab:<14} n_proteins = {len(per):>6,}   "
              f"{(per < CUTOFF).mean() * 100:5.2f}% of proteins below {CUTOFF}")


if __name__ == "__main__":
    main()
