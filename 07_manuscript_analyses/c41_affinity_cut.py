# -*- coding: utf-8 -*-
"""Is <= -6 kcal/mol a defensible docking cut? Pose correctness against the cut (SI Table S4; Methods,
Molecular docking).

No docking into the experimental holo structures was run, so the cut cannot be calibrated against them. What
exists instead: for every cognate (G1) job we know whether the pose docked into the AF model actually
REPRODUCES the experimental pose (symmetry-corrected in-place RMSD <= 2 A). So we can ask directly whether the
-6 cut separates correct poses from incorrect ones. On the v4 run this gives SI Table S4's "On 65,414
good-frame jobs, 72.2% of poses that reproduce the crystallographic pose (RMSD <= 2 A, n = 5,489) and 61.3% of
those that do not (n = 59,925) score <= -6 kcal/mol" and the Methods' "27.8% of poses that reproduce the
crystallographic pose within 2 A score worse than -6 kcal/mol" (the "good frame" block).

    python c41_affinity_cut.py [redock_rmsd.csv]
        default input: <work_dir>/docking/run3<_afdbv4>/reference_poses/redock_rmsd.csv, the suffix following
        AHOJ_AF_MODEL_SET (set it to afdbv4 for the manuscript's run); prints only, writes nothing.
"""
import os
import sys
import numpy as np, pandas as pd

import ma_paths as P

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    P.WD, "docking", f"run3{P.MODEL_SFX}", "reference_poses", "redock_rmsd.csv")
d = pd.read_csv(src)
d["aff"] = pd.to_numeric(d["affinity_top"], errors="coerce")
d["rmsd"] = pd.to_numeric(d["redock_rmsd_top"], errors="coerce")
d["sup"] = pd.to_numeric(d["superpose_rmsd"], errors="coerce")

print("=== data quality: implausible affinities ===")
for lo, hi in [(-20, 0), (-25, 5)]:
    inr = d["aff"].between(lo, hi)
    print(f"  within [{lo},{hi}]: {inr.sum():,} ({100*inr.mean():.2f}%)   outside: {(~inr).sum():,}")
print("  aff <= -20 :", int((d['aff'] <= -20).sum()), " | aff >= 0 :", int((d['aff'] >= 0).sum()))

ok = d[d["rmsd_status"] == "ok"].copy()
print(f"\nrmsd_status ok: {len(ok):,}")
# 'good frame' = reliable superposition, matching redock_summary.json's n_good_frame
gf = ok[ok["sup"] <= 2.0].copy()
print(f"good frame (superpose_rmsd <= 2 A): {len(gf):,}")

plaus = gf[gf["aff"].between(-20, 0)].copy()
print(f"good frame AND plausible affinity in [-20,0]: {len(plaus):,}")

for name, sub in [("good frame", gf), ("good frame + plausible aff", plaus)]:
    print(f"\n=== {name} (n={len(sub):,}) ===")
    corr = sub[sub["rmsd"] <= 2.0]; wrong = sub[sub["rmsd"] > 2.0]
    print(f"  pose CORRECT (rmsd<=2A): n={len(corr):,} ({100*len(corr)/len(sub):.1f}%)")
    print(f"     median affinity {corr['aff'].median():.2f} | frac <= -6: {100*(corr['aff']<=-6).mean():.1f}%"
          f" | frac FAILING the cut (> -6): {100*(corr['aff']>-6).mean():.1f}%")
    print(f"  pose WRONG   (rmsd> 2A): n={len(wrong):,}")
    print(f"     median affinity {wrong['aff'].median():.2f} | frac <= -6: {100*(wrong['aff']<=-6).mean():.1f}%")
    # discrimination
    tp = ((sub['aff']<=-6) & (sub['rmsd']<=2)).sum(); fp = ((sub['aff']<=-6) & (sub['rmsd']>2)).sum()
    fn = ((sub['aff']>-6) & (sub['rmsd']<=2)).sum(); tn = ((sub['aff']>-6) & (sub['rmsd']>2)).sum()
    print(f"  cut as a filter for pose correctness: sensitivity {100*tp/(tp+fn):.1f}%, "
          f"specificity {100*tn/(tn+fp):.1f}%, precision {100*tp/(tp+fp):.1f}%")

# where does the cut sit in the affinity distribution of CORRECT poses?
corr = plaus[plaus["rmsd"] <= 2.0]["aff"]
print("\n=== affinity percentiles among pose-correct dockings (plausible range) ===")
for q in [1, 5, 10, 25, 50, 75, 90]:
    print(f"  p{q:<3d} = {np.percentile(corr, q):7.2f}")
print(f"\n  -> {100*(corr > -6).mean():.1f}% of CORRECTLY REPRODUCED poses score worse than -6 kcal/mol")
