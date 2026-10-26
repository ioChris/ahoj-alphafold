# -*- coding: utf-8 -*-
"""Does the class gap widen when the low-density pockets are left out?

Backs the last sentence of the Supplementary Table S3 caption (consensus +8.4 to +8.1 points, docking +10.2 to
+10.0, without the very-low and low density tiers) and the main-text statement that leaving the low-density
pockets out leaves the class gap unchanged.

The main text says a confident class probability at very low density is not to be trusted (the divergence of
AF-Apo and AF-Holo detection with winning probability collapses there; Supplementary Figure S10). If the labels in that
tier were noise, the pooled AF-Holo - AF-Apo gap would be diluted by them, and dropping them would widen it.
This script tests that directly.

Frames: the Figure 4 detection set (separation_shape.load_lbs(), P2Rank / fpocket / IF-SitePred DCA4-TNS and
their per-pocket consensus) and its docking subset (separation_shape.load_docking(): mean per-pocket dockable
fraction, G1, status ok). Density tier = pocket_density_pair cut at 0.20 / 0.30 / 0.60 / 0.85 (Methods).

Estimator: Figure 4's (separation_shape.Gap): the size-standardized gap by direct adjustment to the union-Rg
deciles of the WHOLE task frame, so every subset is adjusted to one common size distribution; 400 cluster
resamples over UniProt (seed 20260908). Every subset of a task is evaluated on the same resamples, so
"subset minus all" is a paired difference with its own interval.

Sections
  1  pooled gap: all / without very-low density / without very-low and low / the excluded pockets themselves
  2  the same within the confident calls (winning-probability tier high or very high), where Figure S10's
     warning applies: these are the calls the text says not to trust at low density
  3  the separation gradient (Figure 4d) with and without the low-density pockets
  3b the excluded pockets by separation tier (the cell behind docking's steeper span)
  4  a continuous density floor: drop the bottom q% by density, q = 0 ... 50
  5  sections 1 and 3 under the size-balanced labels (Supplementary Note 3 and Table S10;
     detection_rank_cut.two_label_frame)

Usage:  AHOJ_AF_MODEL_SET=afdbv4 python density_exclusion_gaps.py > density_exclusion_gaps_afdbv4.out.txt
Writes density_exclusion_gaps{_afdbv4}.csv to P.OUT. Reads, never writes, everything else; needs
density_separation_gaps{_afdbv4}.csv in P.OUT (run density_separation_gaps.py first).
Gate: the per-density-tier detection gaps equal SI Table S3's density column (density_separation_gaps{_afdbv4}.csv);
the script exits non-zero if any differs by more than 0.05 pt. (Table S3's docking column is built from the
docking table's own frame, not the detection-set subset used here, so it is not gated.)
"""
from __future__ import annotations

import ma_paths as P  # roots; puts the classification code on sys.path

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import separation_shape as S                     # noqa: E402  Figure 4's frames and estimator

SFX = S.MODEL_SFX
TIERS = S.TIERS
TASKS = S.PRED + ["consensus", "docking"]
OUT = P.out(f"density_exclusion_gaps{SFX}.csv")
REF_S3 = P.out(f"density_separation_gaps{SFX}.csv")
CUTS = [0.20, 0.30, 0.60, 0.85]
SUBSETS = [("all", None), ("no_vlow", {"very_low"}), ("no_vlow_low", {"very_low", "low"}),
           ("only_vlow", "only:very_low"), ("only_vlow_low", "only:very_low,low")]
FLOORS = [0, 5, 10, 20, 30, 40, 50]
ROWS: list[dict] = []


def dens_tier(v):
    v = pd.to_numeric(v, errors="coerce")
    t = pd.cut(v, [-np.inf] + CUTS + [np.inf], labels=TIERS, right=True)
    return t.astype(str).where(v.notna(), "")


def mask_for(d, spec):
    if spec is None:
        return np.ones(len(d), bool)
    if isinstance(spec, str):
        keep = set(spec.split(":", 1)[1].split(","))
        return d["dens"].isin(keep).to_numpy()
    return (~d["dens"].isin(spec)).to_numpy()


def eval_masks(g, masks):
    """Point and (B,) bootstrap gap for each boolean mask over g.d, all on g's shared resamples."""
    out = {}
    for name, m in masks.items():
        cell = np.where(m, 0, -1)
        out[name] = (g.point(cell, 1)[0], g.boot(cell, 1)[:, 0], int(m.sum()),
                     int((m & g.ish).sum()), int((m & ~g.ish).sum()))
    return out


def fmt(p, b):
    lo, hi = S.ci(b)
    return f"{p:+6.1f} [{lo:+5.1f}, {hi:+5.1f}]"


def rec(**k):
    ROWS.append(k)


def frames(label_col=None):
    lbs = S.load_lbs()
    if label_col is not None:
        import detection_rank_cut as DRC        # the size-balanced site labels, as SI Table S10 builds them
        lbs = DRC.two_label_frame(lbs)
        lbs["cls"] = lbs[label_col]
    lbs["dens"] = dens_tier(lbs["pocket_density_pair"])
    dk = S.load_docking(lbs)
    return lbs, dk


def section1_2(lbs, dk, labels):
    print(f"\n{'=' * 100}\n1-2. POOLED GAP WITH AND WITHOUT THE LOW-DENSITY POCKETS  ({labels} labels)\n{'=' * 100}")
    print("  gap = size-standardized AF-Holo - AF-Apo, pp [95% cluster-bootstrap CI];  diff = subset - all, paired")
    for scope in ("all calls", "confident calls"):
        print(f"\n  -- {scope}" + (" (winning-probability tier high / very high)" if scope != "all calls" else ""))
        print(f"  {'task':12s} {'subset':15s} {'n':>7s} {'apo':>6s} {'holo':>6s}   {'gap':>22s}   {'diff vs all':>22s}")
        for t in TASKS:
            src = dk if t == "docking" else lbs
            if scope != "all calls":
                src = src[src["prob"].isin(["high", "very_high"])]
            g = S.Gap(src, t)
            res = eval_masks(g, {n: mask_for(g.d, s) for n, s in SUBSETS})
            p0, b0 = res["all"][:2]
            for n, (p, b, nn, nh, na) in res.items():
                dstr = "" if n == "all" else fmt(p - p0, b - b0)
                print(f"  {t:12s} {n:15s} {nn:7,d} {na:6,d} {nh:6,d}   {fmt(p, b):>22s}   {dstr:>22s}"
                      f"   {100 * nn / res['all'][2]:5.1f}% of the frame")
                lo, hi = S.ci(b)
                dlo, dhi = S.ci(b - b0)
                rec(labels=labels, section="pooled", scope=scope, task=t, subset=n, n=nn, n_apo=na, n_holo=nh,
                    gap=p, lo=lo, hi=hi, diff_vs_all=(p - p0) if n != "all" else 0.0,
                    diff_lo=dlo if n != "all" else np.nan, diff_hi=dhi if n != "all" else np.nan)
            print()


def section3(lbs, dk, labels):
    print(f"\n{'=' * 100}\n3. SEPARATION GRADIENT (Figure 4d) WITH AND WITHOUT THE LOW-DENSITY POCKETS  ({labels})\n{'=' * 100}")
    print(f"  {'task':12s} {'subset':12s}" + "".join(f"{s:>8s}" for s in S.SHORT)
          + "   span v.high - v.low         span change vs all")
    for t in TASKS:
        src = dk if t == "docking" else lbs
        g = S.Gap(src, t)
        sep = S.codes_from(g.d["sep"], TIERS)
        spans = {}
        for n, spec in SUBSETS[:3]:
            m = mask_for(g.d, spec)
            cell = np.where(m, sep, -1)
            p, b = g.point(cell, 5), g.boot(cell, 5)
            sp, sb = p[4] - p[0], b[:, 4] - b[:, 0]
            spans[n] = (sp, sb)
            chg = "" if n == "all" else fmt(sp - spans["all"][0], sb - spans["all"][1])
            print(f"  {t:12s} {n:12s}" + "".join(f"{v:8.1f}" for v in p) + f"   {fmt(sp, sb)}   {chg}")
            for i, tier in enumerate(TIERS):
                lo, hi = S.ci(b[:, i])
                rec(labels=labels, section="by_separation", scope="all calls", task=t, subset=n, sep_tier=tier,
                    n=int(((cell == i)).sum()), gap=p[i], lo=lo, hi=hi)
            lo, hi = S.ci(sb)
            rec(labels=labels, section="separation_span", scope="all calls", task=t, subset=n, gap=sp, lo=lo, hi=hi,
                diff_vs_all=sp - spans["all"][0],
                diff_lo=S.ci(sb - spans["all"][1])[0] if n != "all" else np.nan,
                diff_hi=S.ci(sb - spans["all"][1])[1] if n != "all" else np.nan)
        print()


def section3b(lbs, dk, labels):
    """The excluded pockets themselves, by separation tier: which cell moves a span when they are dropped."""
    print(f"\n  3b. the very-low + low density pockets by separation tier ({labels})")
    for t in ("consensus", "docking"):
        src = dk if t == "docking" else lbs
        g = S.Gap(src, t)
        sep = S.codes_from(g.d["sep"], TIERS)
        low = g.d["dens"].isin(["very_low", "low"]).to_numpy()
        cell = np.where(low, sep, -1)
        p, b = g.point(cell, 5), g.boot(cell, 5)
        n, nall = S.Gap.counts(cell, 5), S.Gap.counts(sep, 5)
        for i, tier in enumerate(TIERS):
            lo, hi = S.ci(b[:, i])
            print(f"    {t:10s} {tier:10s} n {n[i]:5,d} ({100 * n[i] / nall[i]:4.1f}% of the tier)   gap {fmt(p[i], b[:, i])}")
            rec(labels=labels, section="low_density_by_separation", scope="all calls", task=t, subset="only_vlow_low",
                sep_tier=tier, n=int(n[i]), gap=p[i], lo=lo, hi=hi)


def section4(lbs, dk, labels):
    print(f"\n{'=' * 100}\n4. CONTINUOUS DENSITY FLOOR: drop the bottom q% of pockets by density  ({labels})\n{'=' * 100}")
    print(f"  {'task':12s}" + "".join(f"{('q=' + str(q)):>24s}" for q in FLOORS))
    for t in TASKS:
        src = dk if t == "docking" else lbs
        g = S.Gap(src, t)
        dv = pd.to_numeric(g.d["pocket_density_pair"], errors="coerce").to_numpy()
        masks = {q: (dv > np.nanpercentile(dv, q)) if q else np.ones(len(dv), bool) for q in FLOORS}
        res = eval_masks(g, masks)
        p0, b0 = res[0][:2]
        print(f"  {t:12s}" + "".join(f"{fmt(res[q][0], res[q][1]):>24s}" for q in FLOORS))
        print(f"  {'  diff':12s}" + "".join(f"{'':>24s}" if q == 0 else f"{fmt(res[q][0] - p0, res[q][1] - b0):>24s}"
                                             for q in FLOORS))
        for q in FLOORS:
            p, b, nn, nh, na = res[q]
            lo, hi = S.ci(b)
            dlo, dhi = S.ci(b - b0) if q else (np.nan, np.nan)
            rec(labels=labels, section="density_floor", scope="all calls", task=t, subset=f"floor_q{q}", n=nn,
                n_apo=na, n_holo=nh, gap=p, lo=lo, hi=hi, diff_vs_all=p - p0, diff_lo=dlo, diff_hi=dhi,
                density_floor=float(np.nanpercentile(dv, q)) if q else np.nan)


def gate(lbs):
    """Per-density-tier gaps on this frame against SI Table S3's source (the same estimator, per tier)."""
    ref = pd.read_csv(REF_S3)
    ref = ref[ref.metric == "density"]
    worst = 0.0
    print("\n  gate: per-density-tier gap vs SI Table S3 source (detection tasks)")
    for t in S.PRED + ["consensus"]:
        g = S.Gap(lbs, t)
        # Table S3's script standardizes within each tier to the task's whole-frame deciles, as Gap does
        cell = S.codes_from(g.d["dens"], TIERS)
        p = g.point(cell, 5)
        r = ref[ref.task == t].set_index("tier").reindex(TIERS)["gap_std"].to_numpy()
        n_ref = ref[ref.task == t].set_index("tier").reindex(TIERS)["n"].to_numpy()
        n_here = S.Gap.counts(cell, 5)
        dev = np.nanmax(np.abs(p - r))
        worst = max(worst, dev)
        print(f"    {t:12s} here " + " ".join(f"{v:5.1f}" for v in p) + "   S3 " + " ".join(f"{v:5.1f}" for v in r)
              + f"   max|d| {dev:.2f}   n here {n_here.sum():,} / S3 {int(n_ref.sum()):,}")
    return worst


def main():
    print(f"model set: {'AFDB v4 re-run' if SFX else 'July (v6)'}")
    lbs, dk = frames()
    print(f"detection pockets {len(lbs):,} ({lbs.site_uniprot.nunique():,} UniProt); docking pockets {len(dk):,}")
    print("density tiers (detection): " + ", ".join(f"{t} {100 * (lbs.dens == t).mean():.1f}%" for t in TIERS))
    print("density tiers (docking):   " + ", ".join(f"{t} {100 * (dk.dens == t).mean():.1f}%" for t in TIERS))
    for cl in ("apo", "holo"):
        s = lbs[lbs.cls == cl]
        print(f"  AF-{cl.capitalize():4s} share in very low + low density: {100 * s.dens.isin(['very_low', 'low']).mean():.1f}%")
    worst = gate(lbs)
    section1_2(lbs, dk, "released")
    section3(lbs, dk, "released")
    section3b(lbs, dk, "released")
    section4(lbs, dk, "released")

    bl, bd = frames("cls_bal")
    print(f"\nsize-balanced frame: detection {len(bl):,}, docking {len(bd):,}")
    section1_2(bl, bd, "size-balanced")
    section3(bl, bd, "size-balanced")

    pd.DataFrame(ROWS).to_csv(OUT, index=False)
    print(f"\n[OK] {len(ROWS)} rows -> {OUT}")
    print(f"gate max|d| vs Table S3 density column: {worst:.2f} pt")
    if worst > 0.05:
        sys.exit("gate failed: the detection frame or estimator differs from SI Table S3's")


if __name__ == "__main__":
    main()
