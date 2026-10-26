# -*- coding: utf-8 -*-
"""Saldano says pLDDT anti-correlates with conformational diversity; we say it tracks the evidence
ratio. Which does it track MORE? Head-to-head on the analysis set.

Backs the first table of Supplementary Table S6 (Spearman rho of pocket pLDDT with each diversity and
evidence variable over the 237,922 analysis pockets) and the partial correlations under it (the evidence
ratio given diversity, +0.007; diversity given the ratio, -0.490).

    python plddt_headtohead.py [frontier_per_pocket.csv [plddt_headtohead.csv]]

Input: the per-pocket table of frontier_at_scale.py (default P.OUT/frontier_per_pocket.csv). The CSV goes
to P.OUT.
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

d = pd.read_csv(P.out(sys.argv[1] if len(sys.argv) > 1 else "frontier_per_pocket.csv"), low_memory=False)
d = d[d["model_pocket_pLDDT_avg"].notna()].copy()
print(f"pockets with pLDDT: {len(d):,}")

# evidence ratio, exactly as the manuscript defines it
d["ratio"] = np.log2((d["n_holo"] + 1) / d["n_apo"].where(d["n_apo"] > 0))
d["abs_ratio"] = d["ratio"].abs()

DIVERSITY = {
    "pocket_rms_max": "max pocket RMSD across conformers",
    "pocket_rms_mean": "mean pocket RMSD",
    "pocket_rms_iqr": "IQR pocket RMSD",
    "volume_std_dev": "SD volume", "volume_iqr": "IQR volume",
    "sasa_std_dev": "SD SASA", "sasa_iqr": "IQR SASA",
    "RoG_std_dev": "SD RoG", "RoG_iqr": "IQR RoG",
    "local_flexibility": "local flexibility", "global_flexibility": "global flexibility",
}
EVIDENCE = {"ratio": "evidence ratio log2((holo+1)/apo)", "abs_ratio": "|evidence ratio|",
            "n_all": "n conformers", "n_apo": "n apo", "n_holo": "n holo"}

y = d["model_pocket_pLDDT_avg"]
rows = []
for grp, cols in [("conformational diversity", DIVERSITY), ("experimental evidence", EVIDENCE)]:
    for c, lab in cols.items():
        if c not in d.columns: continue
        ok = d[c].notna() & y.notna()
        if ok.sum() < 100: continue
        rho, p = spearmanr(d.loc[ok, c], y[ok])
        rows.append({"group": grp, "variable": lab, "col": c, "n": int(ok.sum()),
                     "rho_vs_pLDDT": rho, "p": p})
res = pd.DataFrame(rows).sort_values("rho_vs_pLDDT")
pd.set_option("display.width", 200, "display.max_columns", 20)
print("\n=== Spearman rho with pocket pLDDT (analysis set) ===")
print(res[["group", "variable", "n", "rho_vs_pLDDT"]].round(3).to_string(index=False))

# head-to-head on the common subset, plus a partial correlation
best_div = res[res.group == "conformational diversity"].iloc[0]
print(f"\nstrongest diversity term: {best_div['variable']} (rho {best_div['rho_vs_pLDDT']:.3f})")
sub = d[[best_div["col"], "abs_ratio", "ratio", "model_pocket_pLDDT_avg"]].dropna()
print(f"common subset n = {len(sub):,}")
for c, lab in [(best_div["col"], best_div["variable"]), ("abs_ratio", "|evidence ratio|"), ("ratio", "evidence ratio")]:
    rho, _ = spearmanr(sub[c], sub["model_pocket_pLDDT_avg"])
    print(f"  rho(pLDDT, {lab:<32s}) = {rho:+.3f}")

# partial: does the evidence ratio survive controlling for diversity, and vice versa?
r = sub.rank()
def partial(a, b, c):
    rab, rac, rbc = np.corrcoef(r[a], r[b])[0,1], np.corrcoef(r[a], r[c])[0,1], np.corrcoef(r[b], r[c])[0,1]
    return (rab - rac*rbc) / np.sqrt((1-rac**2)*(1-rbc**2))
print(f"\n  partial rho(pLDDT, |ratio| | diversity) = {partial('model_pocket_pLDDT_avg','abs_ratio',best_div['col']):+.3f}")
print(f"  partial rho(pLDDT, diversity | |ratio|) = {partial('model_pocket_pLDDT_avg',best_div['col'],'abs_ratio'):+.3f}")
out_csv = P.out(sys.argv[2] if len(sys.argv) > 2 else "plddt_headtohead.csv")
res.to_csv(out_csv, index=False); print("\n[OK] ->", out_csv)
