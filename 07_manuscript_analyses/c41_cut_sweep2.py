# -*- coding: utf-8 -*-
"""Docking cut sensitivity on the manuscript's estimand (SI Table S4).

Manuscript "success" = pocket_frac_dockable = per-pocket fraction of cognate ligands whose top pose
is in-pocket AND scores <= -6 kcal/mol, averaged over pockets within class. G1, class-assigned pockets
only (61,012 on the v4 run: 22,611 AF-Apo, 38,401 AF-Holo; 60,689 on the July v6 run). The affinity cut is
swept from -4 to -9 kcal/mol (SI Table S4), with the threshold-free Cliff's delta of the pocket median affinity.

    python c41_cut_sweep2.py [docking_run3_merged.csv] [out.csv]
        defaults: <work_dir>/docking/run3<_afdbv4>/analysis/docking_run3_merged.csv and
        <manuscript_analyses_dir>/c41_cut_sweep<_afdbv4>.csv, the suffix following AHOJ_AF_MODEL_SET
        (AHOJ_AF_MODEL_SET=afdbv4 gives SI Table S4's c41_cut_sweep_afdbv4.csv).
"""
import os
import sys
import numpy as np, pandas as pd

import ma_paths as P

if len(sys.argv) < 2:
    sys.argv.append(os.path.join(P.WD, "docking", f"run3{P.MODEL_SFX}", "analysis", "docking_run3_merged.csv"))
if len(sys.argv) < 3:
    sys.argv.append(P.out(f"c41_cut_sweep{P.MODEL_SFX}.csv"))
d = pd.read_csv(sys.argv[1], low_memory=False)
d = d[(d["ligand_group"] == 1) & (d["status"].astype(str).str.lower() == "ok")].copy()
d["aff"] = pd.to_numeric(d["affinity_top"], errors="coerce")
d["inpk"] = d["pose_in_pocket"].astype(str).str.lower().isin(["true", "1", "yes"])
d["cls"] = d["pocket_class"].astype(str).str.lower()
d = d[d["cls"].isin(["apo", "holo"])]
print("G1 class-assigned jobs:", len(d), "| pockets:", d["site_uid"].nunique())

rows = []
for cut in [-4, -5, -5.5, -6, -6.5, -7, -8, -9]:
    d["_ok"] = d["inpk"] & (d["aff"] <= cut)
    pk = d.groupby(["site_uid", "cls"])["_ok"].mean().reset_index()      # per-pocket FRACTION
    r = pk.groupby("cls")["_ok"].agg(["mean", "size"])
    a, h = 100 * r.loc["apo", "mean"], 100 * r.loc["holo", "mean"]
    rows.append({"cut": cut, "n_apo": int(r.loc["apo", "size"]), "n_holo": int(r.loc["holo", "size"]),
                 "apo_%": a, "holo_%": h, "gap_pp": h - a, "ratio": h / a})
sw = pd.DataFrame(rows)
print("\n=== mean per-pocket dockable fraction, by affinity cut ===")
print(sw.round(2).to_string(index=False))
print("\n(at -6 the manuscript reports 49% apo / 69% holo, +20 pp)")

print("\n=== threshold-free: no cut anywhere ===")
pk = d[d["inpk"]].groupby(["site_uid", "cls"])["aff"].median().reset_index()
for cl in ("apo", "holo"):
    s = pk[pk["cls"] == cl]["aff"]
    print(f"  {cl:4s} n={len(s):6,}  median pocket affinity {s.median():.3f}")
try:
    from scipy.stats import mannwhitneyu
    a = pk[pk.cls == "apo"]["aff"]; h = pk[pk.cls == "holo"]["aff"]
    u, p = mannwhitneyu(h, a, alternative="less")
    print(f"  MannWhitneyU(holo<apo) p={p:.3g}  Cliff's delta={(2.0*u)/(len(h)*len(a))-1.0:.3f}")
except Exception as e:
    print("  scipy:", e)
sw.to_csv(sys.argv[2], index=False); print("\n[OK] ->", sys.argv[2])
