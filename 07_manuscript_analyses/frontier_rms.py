# -*- coding: utf-8 -*-
"""Add pocket_rms (the 4th classifier metric) to the single-structure-frontier measurement.

SEMANTIC CAVEAT, kept front and centre: pocket_rms is a DISTANCE FROM THE QUERY, and the query is a
holo structure. Holo candidates therefore sit near 0 by construction and apo candidates further out,
so this is the axis on which apo and holo are separated *by design*. It is not a fourth independent
instance of the volume/SASA/RoG measurement; report it separately and test how redundant it is with
the class label itself.

Backs the note under Supplementary Table S5: the pocket-RMSD frontier is nearly independent of the class
label (rho = +0.06 with AF-Holo; frontier rate 62.7% for AF-Apo against 68.4% for AF-Holo).

    python frontier_rms.py [model_metrics.csv [frontier_per_pocket.csv [apoholo_stats.csv [frontier_rms.csv]]]]

Defaults: the model-row harvest under cfg.AHOJ_DB_DIR/Summaries/model_metrics/full_run_3, the per-pocket
table of frontier_at_scale.py in P.OUT, cfg.APOHOLO_METRIC_STATS_CSV, and P.OUT/frontier_rms.csv.
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

_args = sys.argv[1:5] + [None] * (4 - len(sys.argv[1:5]))
harvest = _args[0] or (P.cfg.AHOJ_DB_DIR / "Summaries/model_metrics/full_run_3"
                       / "run_23082026_180200_do_collect_model_pLDDT" / "query_summaries_do_collect_model_pLDDT.csv")
per_pocket = P.out(_args[1] or "frontier_per_pocket.csv")
stats_csv = _args[2] or P.cfg.APOHOLO_METRIC_STATS_CSV
out_csv = P.out(_args[3] or "frontier_rms.csv")

h = pd.read_csv(harvest, usecols=["job_name", "model_pocket_rms", "model_pocket_pLDDT_avg"], low_memory=False)
h = h[h.model_pocket_rms.notna()]
print(f"harvested model_pocket_rms: {len(h):,}")

d = pd.read_csv(per_pocket, low_memory=False)
st = pd.read_csv(stats_csv, usecols=["job_name", "pocket_rms_apo_mean", "pocket_rms_holo_mean"], low_memory=False)
d = d.merge(h[["job_name", "model_pocket_rms"]], on="job_name", how="left").merge(st, on="job_name", how="left")
print(f"analysis set: {len(d):,} | with model_pocket_rms: {int(d.model_pocket_rms.notna().sum()):,}")

lo, hi, mu, sd = d.pocket_rms_min, d.pocket_rms_max, d.pocket_rms_mean, d.pocket_rms_std_dev
v = d.model_pocket_rms
rng = (hi - lo).where(lambda x: x > 0)
u = (v - lo) / rng
inside = (u >= 0) & (u <= 1)
z = (v - mu).abs() / sd.where(sd > 0)
a, hh = d.pocket_rms_apo_mean, d.pocket_rms_holo_mean
d_states = (hh - a).abs()
d_model = pd.concat([(v - a).abs(), (v - hh).abs()], axis=1).min(axis=1)
both = d_states.notna() & d_model.notna() & (d_states > 0)
frontier = (d_model < d_states).where(both)

print("\n=== pocket_rms, alongside the other three ===")
print(f"  n usable range        : {int(rng.notna().sum()):,}")
print(f"  inside [min,max]      : {100*inside.sum()/rng.notna().sum():.1f}%")
print(f"  mean +/- 2 SD         : {100*((v>=mu-2*sd)&(v<=mu+2*sd)).sum()/sd.notna().sum():.1f}%")
print(f"  mean +/- 1 SD         : {100*((v>=mu-sd)&(v<=mu+sd)).sum()/sd.notna().sum():.1f}%")
print(f"  z < 1                 : {100*(z<1).sum()/z.notna().sum():.1f}%   (median z {np.nanmedian(z):.2f})")
print(f"  FRONTIER              : {100*frontier[both].sum()/both.sum():.1f}%   (n={int(both.sum()):,})")
print(f"  median d_model {np.nanmedian(d_model[both]):.3f} vs median d_states {np.nanmedian(d_states[both]):.3f}")

d["frontier_pocket_rms"], d["inside_pocket_rms"], d["z_pocket_rms"] = frontier, inside, z

print("\n=== is it redundant with the class label? (the thing to check) ===")
cls = d["pocket_class"].astype(str).str.lower()
for lab, mask in [("AF-Apo", cls == "apo"), ("AF-Holo", cls == "holo")]:
    s = frontier[both & mask]
    print(f"  frontier rate among {lab:<8}: {100*s.mean():.1f}%  (n={len(s):,})")
sub = d[both].copy()
print(f"  rho(frontier_rms, is_holo)   = {spearmanr(sub.frontier_pocket_rms.astype(float), (sub.pocket_class.str.lower()=='holo').astype(float))[0]:+.3f}")
for m in ["volume", "sasa", "RoG"]:
    s2 = d[[f"frontier_{m}", "frontier_pocket_rms"]].dropna()
    print(f"  rho(frontier_rms, frontier_{m:<6}) = {spearmanr(s2[f'frontier_{m}'].astype(float), s2.frontier_pocket_rms.astype(float))[0]:+.3f}")
s3 = d[["z_pocket_rms", "pocket_density_pair"]].replace([np.inf,-np.inf],np.nan).dropna() if "pocket_density_pair" in d.columns else None
if s3 is not None and len(s3):
    print(f"  rho(z_rms, density)          = {spearmanr(s3.z_pocket_rms, s3.pocket_density_pair)[0]:+.3f}")

d[["job_name","model_pocket_rms","inside_pocket_rms","z_pocket_rms","frontier_pocket_rms"]].to_csv(out_csv, index=False)
print("\n[OK] ->", out_csv)
