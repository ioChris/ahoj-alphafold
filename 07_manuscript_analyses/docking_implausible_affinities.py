# -*- coding: utf-8 -*-
"""Implausible docking affinities: how many, one rule, and what the rule moves (Methods, Molecular docking).

A small share of Uni-Dock jobs return a top affinity outside any physical range (down to -345 and up to
+300 kcal/mol). Uni-Dock's own paper treats scores outside -15 to 0 kcal/mol as outliers; on the July (v6) run
2.3% of the jobs lay outside [-20, 0). This script recomputes the share on the v4 docking run (the manuscript's),
characterizes the tail, and recomputes the three docking estimands of the main text under each candidate rule.
It gives the Methods sentence "2.4% of the jobs with a result (1,645 of 69,738) fall outside that range, 1,198
with a positive score (large ligands, median 40 heavy atoms against 24) and 447 below -15 kcal/mol ... moves each
class's dockable fraction by less than one point, their gap by less than 0.2 points, Cliff's delta by 0.006 and
the within-ligand share by 0.3 points", and the published Cliff's delta (-0.27) and within-ligand share (59.2% of
3,422). The rules:

    published   no rule: every job with a result counts as returned
    fail        jobs outside the plausible range are scored as failures (not dockable) and dropped from the
                affinity statistics
    drop        jobs outside the plausible range are dropped from everything (denominators shrink)

for both ranges, Uni-Dock's [-15, 0] and the mark's [-20, 0). The estimands are the mean per-pocket dockable
fraction by class (G1, class-assigned pockets; main text "69% against 50%"), Cliff's delta of the pocket
median affinity (holo against apo; main text -0.27) and the within-ligand pairing (share of CCD ligands
docked into both classes whose holo median is better; main text 59.2% of 3,422). Definitions follow
05_docking/runner/analyze_docking_pilot.py in the public repo (status == ok, G1; dockable = pose in pocket
and affinity_top <= -6; the pocket's median affinity over its jobs; the CCD's median affinity per class).

Inputs (read-only)
    <work_dir>/docking/<run>/analysis/docking_run3_merged.csv   run = run3_afdbv4 (default) or run3 (July, v6)

Usage
    python docking_implausible_affinities.py                 # the v4 run (AHOJ_AF_MODEL_SET=afdbv4 is implied)
    python docking_implausible_affinities.py --run run3      # July's v6 run (1,612 of 69,395 outside [-20, 0))
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

import ma_paths as P

WORK = Path(P.WD)
TAU = -6.0
RANGES = {"unidock [-15, 0]": (-15.0, 0.0, True), "mark [-20, 0)": (-20.0, 0.0, False)}


def plausible(aff, rng):
    lo, hi, hi_inclusive = rng
    return (aff >= lo) & ((aff <= hi) if hi_inclusive else (aff < hi))


def cliffs_delta(h, a):
    """Cliff's delta of h against a (+1 when every h exceeds every a), via the Mann-Whitney U statistic."""
    u = sstats.mannwhitneyu(h, a, alternative="two-sided").statistic
    return 2.0 * u / (len(h) * len(a)) - 1.0


def estimands(g):
    """g: G1 ok jobs with columns site_uid, ligand_id, is_holo (0/1), aff, dockable (0/1 or NaN)."""
    pk = g.groupby("site_uid").agg(is_holo=("is_holo", "first"), median_aff=("aff", "median"),
                                   frac_dockable=("dockable", "mean"))
    apo, holo = pk[pk.is_holo == 0], pk[pk.is_holo == 1]
    out = {"n_pockets_apo": len(apo), "n_pockets_holo": len(holo),
           "dockable_apo": 100 * apo.frac_dockable.mean(), "dockable_holo": 100 * holo.frac_dockable.mean()}
    out["gap"] = out["dockable_holo"] - out["dockable_apo"]
    ma, mh = apo.median_aff.dropna(), holo.median_aff.dropna()
    out["delta"] = cliffs_delta(mh.to_numpy(), ma.to_numpy())
    med = g.dropna(subset=["aff"]).groupby(["ligand_id", "is_holo"])["aff"].median().unstack("is_holo")
    paired = med.dropna(subset=[0, 1])
    diff = paired[1] - paired[0]
    out["wl_n"] = int(len(paired))
    out["wl_pct_holo_better"] = 100 * float((diff < 0).mean())
    out["wl_median_diff"] = float(diff.median())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="run3_afdbv4", choices=["run3_afdbv4", "run3"])
    a = ap.parse_args()
    src = WORK / "docking" / a.run / "analysis" / "docking_run3_merged.csv"
    cols = ["job_id", "site_uid", "ligand_group", "ligand_id", "status", "affinity_top", "pose_in_pocket",
            "pocket_class", "ligand_n_heavy_atoms"]
    m = pd.read_csv(src, low_memory=False, usecols=cols)
    m["aff"] = pd.to_numeric(m.affinity_top, errors="coerce")
    ok = m[m.status.astype(str).str.lower() == "ok"].copy()
    g1 = ok[ok.ligand_group == 1].copy()
    print(f"=== {a.run}: {src.name} ===")
    print(f"  jobs {len(m):,}; status ok {len(ok):,} (error {int((m.status != 'ok').sum()):,}); "
          f"G1 ok {len(g1):,} on {g1.site_uid.nunique():,} pockets; G2 ok {int((ok.ligand_group == 2).sum()):,}")
    print(f"  affinity_top over ok jobs: min {ok.aff.min():.1f}, max {ok.aff.max():.1f} kcal/mol")

    print("\n=== the tail (G1 jobs with a result) ===")
    for name, rng in RANGES.items():
        bad = ~plausible(g1.aff, rng)
        lo = g1.aff < rng[0]
        hi = (g1.aff > rng[1]) if rng[2] else (g1.aff >= rng[1])
        print(f"  outside {name}: {int(bad.sum()):,} of {len(g1):,} ({100 * bad.mean():.2f}%): "
              f"below {rng[0]:g}: {int(lo.sum()):,}; at or above 0: {int(hi.sum()):,}")
    hv = g1[g1.aff >= 0]
    neg = g1[g1.aff <= -20]
    mid = g1[(g1.aff > -20) & (g1.aff < 0)]
    print(f"  ligand heavy atoms, median: score >= 0: {hv.ligand_n_heavy_atoms.median():.0f}; "
          f"score <= -20: {neg.ligand_n_heavy_atoms.median():.0f}; the rest: {mid.ligand_n_heavy_atoms.median():.0f}")
    print(f"  pose in pocket, share: score >= 0: {100 * hv.pose_in_pocket.astype(str).str.lower().isin(['true', '1']).mean():.1f}%; "
          f"score <= -20: {100 * neg.pose_in_pocket.astype(str).str.lower().isin(['true', '1']).mean():.1f}%; "
          f"the rest: {100 * mid.pose_in_pocket.astype(str).str.lower().isin(['true', '1']).mean():.1f}%")
    cls = g1.pocket_class.astype(str).str.lower()
    for lab, sel in (("AF-Apo", cls == "apo"), ("AF-Holo", cls == "holo")):
        s = g1[sel]
        print(f"  {lab}: jobs {len(s):,}; score >= 0: {100 * (s.aff >= 0).mean():.2f}%; score <= -20: {100 * (s.aff <= -20).mean():.2f}%")
    print(f"  pockets with at least one such job (either range, G1): "
          f"{g1[~plausible(g1.aff, RANGES['mark [-20, 0)'])].site_uid.nunique():,} of {g1.site_uid.nunique():,}")

    g1["is_holo"] = cls.map({"apo": 0, "holo": 1})
    g1["inpk"] = g1.pose_in_pocket.astype(str).str.lower().isin(["true", "1", "yes"])
    g1["dockable"] = (g1.inpk & (g1.aff <= TAU)).astype(float)
    base = g1.dropna(subset=["is_holo"]).copy()
    base["is_holo"] = base.is_holo.astype(int)

    print("\n=== the three estimands under each rule (G1, class-assigned pockets) ===")
    print(f"  {'rule':34s} {'pockets apo/holo':>18s} {'dockable apo':>12s} {'holo':>7s} {'gap':>6s} "
          f"{'delta':>7s} {'within-ligand n':>15s} {'holo better':>11s}")
    rows = []

    def show(label, g):
        e = estimands(g)
        rows.append((label, e))
        print(f"  {label:34s} {e['n_pockets_apo']:>8,}/{e['n_pockets_holo']:<9,} {e['dockable_apo']:>12.2f} "
              f"{e['dockable_holo']:>7.2f} {e['gap']:>6.2f} {e['delta']:>7.3f} {e['wl_n']:>15,} {e['wl_pct_holo_better']:>10.1f}%")

    show("published (no rule)", base)
    for name, rng in RANGES.items():
        pl = plausible(base.aff, rng)
        f = base.copy()
        f.loc[~pl, "dockable"] = 0.0
        f.loc[~pl, "aff"] = np.nan
        show(f"fail: {name}", f)
        show(f"drop: {name}", base[pl])
    pub = rows[0][1]
    print("\n=== what the rule moves (against published) ===")
    for label, e in rows[1:]:
        print(f"  {label:34s} gap {e['gap'] - pub['gap']:+.2f} pp; delta {e['delta'] - pub['delta']:+.3f}; "
              f"within-ligand {e['wl_pct_holo_better'] - pub['wl_pct_holo_better']:+.1f} points, n {e['wl_n'] - pub['wl_n']:+,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
