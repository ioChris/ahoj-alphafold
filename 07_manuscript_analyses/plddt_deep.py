# -*- coding: utf-8 -*-
"""pLDDT vs conformational diversity: (a) does it hold in the low-pLDDT tail?
(b) does it hold across ALL pocket metrics, on a comparable scale?

Backs the Methods' choice of diversity measures ("pLDDT and conformational diversity": the coefficient
of variation of the radius of gyration as the scale-free alternative, rho = -0.37 with pocket pLDDT,
because the spread of volume and SASA is largely a pocket-size proxy) and the corresponding sentence under
Supplementary Table S6.

    python plddt_deep.py [frontier_per_pocket.csv]

Input: the per-pocket table of frontier_at_scale.py (default P.OUT/frontier_per_pocket.csv). Prints only.
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

d = pd.read_csv(P.out(sys.argv[1] if len(sys.argv) > 1 else "frontier_per_pocket.csv"), low_memory=False)
d = d[d["model_pocket_pLDDT_avg"].notna()].copy()
d["ratio"] = np.log2((d["n_holo"]+1)/d["n_apo"].where(d["n_apo"]>0))
d["abs_ratio"] = d["ratio"].abs()
y = "model_pocket_pLDDT_avg"
print(f"pockets: {len(d):,}")

# ---------- (b) all metrics on a comparable scale -----------------------------
# Spearman is already scale-free. To COMBINE metrics we need a dimensionless spread:
# CV = SD/mean and rIQR = IQR/mean. Both are unit-free; pocket_rms is already an absolute distance.
SPREADS = {}
for met in ["volume", "sasa", "RoG", "pocket_rms"]:
    mu = d[f"{met}_mean"].replace(0, np.nan)
    d[f"cv_{met}"] = d[f"{met}_std_dev"] / mu
    d[f"riqr_{met}"] = d[f"{met}_iqr"] / mu
    SPREADS[met] = (f"{met}_std_dev", f"{met}_iqr", f"cv_{met}", f"riqr_{met}")

print("\n=== (b) rho with pLDDT, per metric, four spread definitions ===")
rows = []
for met, cols in SPREADS.items():
    rec = {"metric": met}
    for c, lab in zip(cols, ["SD", "IQR", "CV = SD/mean", "IQR/mean"]):
        ok = d[c].notna() & np.isfinite(d[c])
        rec[lab] = spearmanr(d.loc[ok, c], d.loc[ok, y])[0]
    rows.append(rec)
sp = pd.DataFrame(rows).set_index("metric")
pd.set_option("display.width", 200)
print(sp.round(3).to_string())

# composite: average percentile rank of the four metrics' relative spread
comp_cols = [f"cv_{m}" for m in ["volume", "sasa", "RoG", "pocket_rms"]]
ranks = d[comp_cols].rank(pct=True)
d["diversity_composite"] = ranks.mean(axis=1)
ok = d["diversity_composite"].notna()
print(f"\ncomposite (mean percentile rank of the 4 CVs):  rho = {spearmanr(d.loc[ok,'diversity_composite'], d.loc[ok,y])[0]:+.3f}")
print(f"best single metric (pocket_rms mean RMSD):      rho = {spearmanr(d['pocket_rms_mean'], d[y])[0]:+.3f}")
print(f"|evidence ratio|:                               rho = {spearmanr(d['abs_ratio'], d[y])[0]:+.3f}")

# comparable EFFECT SIZE: pLDDT points per 1 SD of the (z-scored) predictor
print("\n=== comparable effect size: change in pLDDT per 1 SD of predictor (OLS on z-scores) ===")
for lab, c in [("mean pocket RMSD", "pocket_rms_mean"), ("CV RoG", "cv_RoG"), ("CV SASA", "cv_sasa"),
               ("CV volume", "cv_volume"), ("CV pocket_rms", "cv_pocket_rms"),
               ("diversity composite", "diversity_composite"), ("|evidence ratio|", "abs_ratio")]:
    s = d[[c, y]].replace([np.inf, -np.inf], np.nan).dropna()
    z = (s[c] - s[c].mean()) / s[c].std()
    beta = np.polyfit(z, s[y], 1)[0]
    print(f"  {lab:<22s} {beta:+.2f} pLDDT points per SD   (n={len(s):,})")

# ---------- (a) the low-pLDDT tail -------------------------------------------
print("\n=== (a) within AlphaFold's own confidence bands ===")
d["band"] = pd.cut(d[y], [0, 50, 70, 90, 100],
                   labels=["very low (<50)", "low (50-70)", "confident (70-90)", "very high (>90)"])
out = []
for b, s in d.groupby("band", observed=True):
    if len(s) < 200: 
        out.append({"band": b, "n": len(s), "note": "too few"}); continue
    out.append({"band": b, "n": len(s),
                "pLDDT_sd": s[y].std(),
                "rho_diversity": spearmanr(s["pocket_rms_mean"], s[y])[0],
                "rho_composite": spearmanr(s["diversity_composite"], s[y])[0],
                "rho_abs_ratio": spearmanr(s["abs_ratio"], s[y])[0]})
print(pd.DataFrame(out).round(3).to_string(index=False))
print("\nNOTE: within-band rho is attenuated by restriction of range (pLDDT SD shrinks inside a band);")
print("compare bands to each other, not to the pooled value.")

# median diversity per band -- the unrestricted view of the same question
print("\n=== median conformational diversity per confidence band (no range restriction) ===")
print(d.groupby("band", observed=True).agg(
    n=(y, "size"), median_pocket_rms=("pocket_rms_mean", "median"),
    median_cv_RoG=("cv_RoG", "median"), median_abs_ratio=("abs_ratio", "median")).round(3).to_string())
