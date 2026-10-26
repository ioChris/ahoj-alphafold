# -*- coding: utf-8 -*-
"""Is `pocket_density_pair` just a soft version of the single-structure-frontier placement?

density = mean of peak-normalized KDE heights (combined + strongest) evaluated AT THE MODEL'S VALUE
       -> conceptually "does the model sit where the experimental conformers are"
SSF     = hard containment (inside [min,max]) + centrality (z = |model-mean|/SD) per metric

Backs the density-centrality correlation of Supplementary Table S5's note and the Results ("Density
falls as the model moves away from that spread", rho = -0.68 between pocket_density_pair and the mean
per-metric centrality |model - mean| / SD).

    python density_vs_ssf.py [frontier_per_pocket.csv [analysis_unified_levels.csv]]

Inputs: the per-pocket table of frontier_at_scale.py (default P.OUT/frontier_per_pocket.csv) and
cfg.ANALYSIS_UNIFIED_LEVELS_CSV. Prints only.
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

d = pd.read_csv(P.out(sys.argv[1]) if len(sys.argv) > 1 else P.out("frontier_per_pocket.csv"), low_memory=False)
uni = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else P.cfg.ANALYSIS_UNIFIED_LEVELS_CSV,
                  usecols=["threshold","rep_job","pocket_density_pair",
                                        "pocket_density_combined","pocket_density_strongest",
                                        "pocket_overlap_pair","pocket_class"], low_memory=False)
uni = uni[uni.threshold=="UNCLUSTERED"].rename(columns={"rep_job":"job_name"})
d = d.merge(uni.drop(columns=["threshold"]), on="job_name", how="inner")
print(f"joined: {len(d):,}")

METS = ["volume","sasa","RoG"]
# aggregate SSF statistics
d["n_inside"] = d[[f"inside_{m}" for m in METS]].sum(axis=1)          # 0-3 metrics inside range
d["mean_z"]   = d[[f"z_{m}" for m in METS]].mean(axis=1)              # mean centrality
d["mean_centrality"] = d[[f"u_{m}" for m in METS]].apply(lambda r: (r-0.5).abs().mean(), axis=1)
d["n_frontier"] = d[[f"frontier_{m}" for m in METS]].sum(axis=1)

print("\n=== correlation of density with the SSF statistics ===")
for lab, col, note in [
    ("n metrics inside [min,max] (0-3)", "n_inside", "hard containment"),
    ("mean z = |model-mean|/SD",         "mean_z",   "centrality; LOWER = more central"),
    ("mean |u-0.5|",                     "mean_centrality", "LOWER = more central"),
    ("n metrics on the frontier (0-3)",  "n_frontier", "distance-ratio statistic"),
]:
    s = d[["pocket_density_pair", col]].replace([np.inf,-np.inf],np.nan).dropna()
    print(f"  rho(density, {lab:<34s}) = {spearmanr(s.pocket_density_pair, s[col])[0]:+.3f}   [{note}]")

print("\n=== and against its own components ===")
for c in ["pocket_density_combined","pocket_density_strongest"]:
    s=d[[c,"mean_z"]].replace([np.inf,-np.inf],np.nan).dropna()
    print(f"  rho({c:<28s}, mean_z) = {spearmanr(s[c],s.mean_z)[0]:+.3f}")

print("\n=== density tier x SSF outcome ===")
sys.path.insert(0, str(P.REPO / "04_p2rank" / "analysis"))
from confidence_labeling import support_level
d["dens_tier"] = d["pocket_density_pair"].map(lambda v: support_level(v))
tab = d.groupby("dens_tier", observed=True).agg(
    n=("job_name","size"),
    median_n_conformers=("n_all","median"),
    pct_inside_all3=("n_inside", lambda s: 100*(s==3).mean()),
    mean_n_inside=("n_inside","mean"),
    median_mean_z=("mean_z","median"),
    pct_frontier_all3=("n_frontier", lambda s: 100*(s==3).mean()))
order=["very_low","low","moderate","high","very_high"]
print(tab.reindex([o for o in order if o in tab.index]).round(2).to_string())

print("\n=== does the association depend on how many conformers there are? ===")
d["nbin"]=pd.cut(d["n_all"],[0,5,10,25,50,100,10**6],labels=["1-5","6-10","11-25","26-50","51-100",">100"])
rows=[]
for b,s in d.groupby("nbin",observed=True):
    s2=s[["pocket_density_pair","n_inside","mean_z"]].replace([np.inf,-np.inf],np.nan).dropna()
    if len(s2)<200: continue
    rows.append({"n_conformers":b,"n":len(s2),
                 "rho(density, n_inside)":spearmanr(s2.pocket_density_pair,s2.n_inside)[0],
                 "rho(density, mean_z)":spearmanr(s2.pocket_density_pair,s2.mean_z)[0]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
