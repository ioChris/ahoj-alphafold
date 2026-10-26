# -*- coding: utf-8 -*-
"""Quantifying the single-structure frontier at proteome scale.

Builds the per-pocket table that the frontier and pLDDT analyses read (frontier_final.py, frontier_rms.py,
density_vs_ssf.py, plddt_headtohead.py, plddt_deep.py) and the volume / SASA / RoG summary behind
Supplementary Table S5 (frontier_final.py adds pocket RMSD and the final denominators).

    python frontier_at_scale.py [frontier_per_pocket.csv [frontier_summary.csv]]     (both written to P.OUT)

Inputs: cfg.ANALYSIS_UNIFIED_LEVELS_CSV (the UNCLUSTERED analysis set), the full_run_3 classification
summaries and metrics (cfg.CLASSIFICATION_MERGED_DIR) and the apo/holo metric means
(cfg.APOHOLO_METRIC_STATS_CSV).

Lane 2023 and Karelina 2023 argue by example that an AF model can be about as close to an
experimental structure as two experimental structures of the same protein are to each other.
We can measure that on the whole analysis set.

Per pocket, per metric (volume / SASA / RoG -- the three classification metrics for which the AF
model has its own value):
  placement    u = (model - min) / (max - min) over the experimental conformers
  inside       0 <= u <= 1
  frontier     d_model = min(|model - apo_mean|, |model - holo_mean|)
               d_states = |holo_mean - apo_mean|
               FRONTIER when d_model < d_states, i.e. the model sits closer to one experimental
               state than the two states sit to each other.
  centroid     z = |model - all_mean| / std_dev  (z < 1 = closer to the centre than a typical conformer)
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import sys
from pathlib import Path
import numpy as np, pandas as pd

DB = P.cfg.AHOJ_DB_DIR
MERGED = P.cfg.CLASSIFICATION_MERGED_DIR
STATS = P.cfg.APOHOLO_METRIC_STATS_CSV
UNI = P.cfg.ANALYSIS_UNIFIED_LEVELS_CSV

METRICS = [("volume", "model_pocket_volume"), ("sasa", "model_pocket_sasa"), ("RoG", "model_pocket_RoG")]

# ---- analysis set -----------------------------------------------------------
uni = pd.read_csv(UNI, usecols=["threshold", "rep_job", "pocket_class"], low_memory=False)
aset = uni[uni.threshold == "UNCLUSTERED"][["rep_job", "pocket_class"]].rename(columns={"rep_job": "job_name"})
print(f"analysis set (UNCLUSTERED): {len(aset):,}")

# ---- model values + counts --------------------------------------------------
s = pd.read_csv(MERGED / "query_summaries_do_kde_plots_and_classification.csv",
                usecols=["job_name", "n_apo", "n_holo", "n_all", "model_pocket_pLDDT_avg",
                         "model_pocket_volume", "model_pocket_sasa", "model_pocket_RoG"], low_memory=False)
# ---- experimental spread ----------------------------------------------------
mc = ["job_name", "local_flexibility", "global_flexibility"]
for met in ["volume", "sasa", "RoG", "pocket_rms"]:
    mc += [f"{met}_{k}" for k in ("min", "max", "mean", "std_dev", "iqr")]
q = pd.read_csv(MERGED / "query_metrics_do_kde_plots_and_classification.csv", usecols=mc, low_memory=False)
# ---- apo/holo means ---------------------------------------------------------
sc = ["job_name"] + [f"{m}_{w}_mean" for m in ["volume", "sasa", "RoG"] for w in ("apo", "holo")]
st = pd.read_csv(STATS, usecols=sc, low_memory=False)

d = aset.merge(s, on="job_name", how="left").merge(q, on="job_name", how="left").merge(st, on="job_name", how="left")
d = d[d["model_pocket_volume"].notna()].copy()
print(f"with an AF model: {len(d):,}\n")

rows = []
for met, mod in METRICS:
    lo, hi, mu, sd = d[f"{met}_min"], d[f"{met}_max"], d[f"{met}_mean"], d[f"{met}_std_dev"]
    rng = (hi - lo).where(lambda x: x > 0)
    u = (d[mod] - lo) / rng
    inside = (u >= 0) & (u <= 1)
    z = (d[mod] - mu).abs() / sd.where(sd > 0)
    a, h = d[f"{met}_apo_mean"], d[f"{met}_holo_mean"]
    d_states = (h - a).abs()
    d_model = pd.concat([(d[mod] - a).abs(), (d[mod] - h).abs()], axis=1).min(axis=1)
    frontier = d_model < d_states
    both = d_states.notna() & d_model.notna() & (d_states > 0)
    d[f"u_{met}"], d[f"inside_{met}"], d[f"z_{met}"] = u, inside, z
    d[f"frontier_{met}"] = frontier.where(both)
    rows.append({
        "metric": met,
        "n_usable_range": int(rng.notna().sum()),
        "inside_%": 100 * inside.sum() / rng.notna().sum(),
        "median_u": float(np.nanmedian(u[inside])),
        "n_z": int(z.notna().sum()),
        "z<1_%": 100 * (z < 1).sum() / z.notna().sum(),
        "median_z": float(np.nanmedian(z)),
        "n_frontier": int(both.sum()),
        "frontier_%": 100 * frontier[both].sum() / both.sum(),
        "median_d_model": float(np.nanmedian(d_model[both])),
        "median_d_states": float(np.nanmedian(d_states[both])),
    })

res = pd.DataFrame(rows)
pd.set_option("display.width", 220, "display.max_columns", 30)
print("=== single-structure frontier, analysis set (UNCLUSTERED) ===")
print(res.round(2).to_string(index=False))

# conditioning on how many conformers the range is built from
print("\n=== 'inside the range' depends on how many conformers there are ===")
d["nbin"] = pd.cut(d["n_all"], [0, 2, 5, 10, 25, 50, 100, 10**6],
                   labels=["1-2", "3-5", "6-10", "11-25", "26-50", "51-100", ">100"])
tab = d.groupby("nbin", observed=True).agg(
    n=("job_name", "size"),
    inside_volume=("inside_volume", "mean"), inside_sasa=("inside_sasa", "mean"),
    inside_RoG=("inside_RoG", "mean"), frontier_volume=("frontier_volume", "mean"),
    frontier_sasa=("frontier_sasa", "mean"), frontier_RoG=("frontier_RoG", "mean"))
for c in tab.columns[1:]:
    tab[c] = 100 * tab[c]
print(tab.round(1).to_string())

# all three metrics at once
d["inside_all3"] = d[[f"inside_{m}" for m, _ in METRICS]].all(axis=1)
d["frontier_all3"] = d[[f"frontier_{m}" for m, _ in METRICS]].all(axis=1)
d["frontier_any"] = d[[f"frontier_{m}" for m, _ in METRICS]].any(axis=1)
print(f"\ninside the range on ALL THREE metrics : {100*d['inside_all3'].mean():.1f}%")
print(f"frontier on ALL THREE metrics        : {100*d['frontier_all3'].mean():.1f}%")
print(f"frontier on AT LEAST ONE metric      : {100*d['frontier_any'].mean():.1f}%")

out_pockets = P.out(sys.argv[1] if len(sys.argv) > 1 else "frontier_per_pocket.csv")
out_summary = P.out(sys.argv[2] if len(sys.argv) > 2 else "frontier_summary.csv")
d.to_csv(out_pockets, index=False)
res.to_csv(out_summary, index=False)
print("\n[OK] ->", out_pockets, "|", out_summary)
