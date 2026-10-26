# -*- coding: utf-8 -*-
"""Which predictor shows the most dramatic class effect -- and on which quantity?

Two readings of the same three detectors appear to disagree:

  * Figure 3 (raw rates, whole set): IF-SitePred carries the LARGEST AF-Holo - AF-Apo gap
    (+21.8 pp) and fpocket the SMALLEST (+13.1 pp). Read as a level, fpocket is the most robust
    to the label.
  * Figure 4d / SEPARATION_SHAPE (size-standardized, by separation tier): fpocket has the WIDEST
    separation span (+19.3 pp, from a zero gap at very-low separation), so it reads as the most
    label-sensitive -- attributed to it being the most purely geometric of the three.

Candidate sources of the disagreement, each testable on ONE row set with ONE cluster bootstrap so
that every tool-vs-tool and raw-vs-standardized contrast is paired and carries its own interval:

  (1) estimand   -- a LEVEL of the gap (Figure 3) vs its SLOPE along a moderator (Figure 4);
  (2) adjustment -- raw vs union_rg-decile size standardization;
  (3) axis       -- Figure 3a stratifies on winning probability, Figure 4d on separation;
  (4) composition-- the overall gap = within-tier gaps + a class-tier association term (AF-holo
                    pockets sit at higher separation, where every class is detected better);
  (5) scale      -- absolute pp vs relative (log odds ratio), since fpocket's apo baseline is higher;
  (6) arm        -- whether the widening is an apo collapse or a holo gain;
  (7) row set    -- Figure 3 quotes n = 86,946, the separation analysis n = 86,651.

Estimators are Figure 4's own (`separation_shape.Gap`): size-standardized mean difference by
direct adjustment to the union_rg decile distribution, cluster bootstrap over site_uniprot,
B = 400. The raw versions skip the decile adjustment and nothing else. Docking is reported as a
reference on its own row set and is NOT paired with the detectors.

This analysis ran on the July (v6) detection tables and none of its own numbers is quoted in the
manuscript; it is in this folder because detection_rank_cut.py (Supplementary Table S12) imports
its `log_or` and `rank_share` helpers.

    python tool_gap_reconciliation.py      -> <manuscript_analyses_dir>/tool_gap_reconciliation.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import ma_paths as P

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from separation_shape import (load_lbs, load_docking, Gap, ci, codes_from, rank01,  # noqa: E402
                              W, PRED, TIERS, SHORT, K, B)

TOOLS = PRED + ["consensus"]
PAIRS = (("fpocket", "IF-SitePred"), ("fpocket", "P2Rank"), ("IF-SitePred", "P2Rank"))
ROWS = []          # long-form record of every printed estimate, for the CSV


def rec(section, quantity, axis, adjust, tool, est, bs, **extra):
    lo, hi = ci(bs)
    ROWS.append(dict(section=section, quantity=quantity, axis=axis, adjust=adjust, tool=tool,
                     estimate=float(est), lo=float(lo), hi=float(hi), **extra))
    return lo, hi


def line(label, est, bs, unit="pp", width=44, tail=""):
    lo, hi = ci(bs)
    star = " *" if np.isfinite(lo) and (lo > 0 or hi < 0) else ""
    print(f"  {label:<{width}s} {est:+7.2f} {unit} [{lo:+7.2f}, {hi:+7.2f}]{star}{tail}")


def rank_share(point_by_tool, boot_by_tool, tool, largest=True):
    """Share of resamples in which `tool` has the largest (or smallest) value among the three
    predictors -- the ordering's own stability, not three separate intervals to eyeball."""
    M = np.column_stack([boot_by_tool[t] for t in PRED])
    j = PRED.index(tool)
    pick = M.argmax(1) if largest else M.argmin(1)
    return 100.0 * np.mean(pick == j)


def log_or(rates):
    """rates: (..., 2) in percent, [apo, holo] -> log odds ratio holo vs apo."""
    p = np.clip(np.asarray(rates, float) / 100.0, 1e-6, 1 - 1e-6)
    return np.log(p[..., 1] / (1 - p[..., 1])) - np.log(p[..., 0] / (1 - p[..., 0]))


# =========================================================================== shared bootstrap
def build_gaps(d, tools):
    """One Gap per outcome, all on identical rows, sharing ONE set of bootstrap resamples."""
    gaps = {t: Gap(d, t) for t in tools}
    g0 = gaps[tools[0]]
    for t in tools[1:]:
        assert gaps[t].d.index.equals(g0.d.index), f"{t} is not on the common row set"
        assert np.array_equal(gaps[t].codes, g0.codes)
    ix = g0._indices()
    for t in tools[1:]:
        gaps[t]._boot_ix = ix
    return gaps


def series(g, cell, ncell):
    """Point + bootstrap for the gap, both adjustments, and the two arms, on one resample set."""
    out = {}
    out["gap_std"] = (g.point(cell, ncell), g.boot(cell, ncell))
    rt, brt = g.rates(cell, ncell), g.boot_rates(cell, ncell)
    rw, brw = g.rates(cell, ncell, raw=True), g.boot_rates(cell, ncell, raw=True)
    out["gap_raw"] = (rw[:, 1] - rw[:, 0], brw[:, :, 1] - brw[:, :, 0])
    out["arms_std"] = (rt, brt)
    out["arms_raw"] = (rw, brw)
    out["lor_raw"] = (log_or(rw), log_or(brw))
    out["lor_std"] = (log_or(rt), log_or(brt))
    out["n"] = Gap.counts(cell, ncell)
    return out


# =========================================================================== 0. row set
def section0(d):
    print("=" * 96)
    print("0. ROW SET -- is Figure 3's n = 86,946 the same population as the separation analysis?")
    print("=" * 96)
    cols = ["site_uid", "pocket_class", "success_tns"]
    base = pd.read_csv(W / "lbs_generality/p2rank/lbs_p2rank_st100_merged_per_pocket.csv",
                       usecols=cols, low_memory=False).rename(columns={"success_tns": "P2Rank"})
    for tool, name in (("fpocket", "fpocket"), ("ifsitepred", "IF-SitePred")):
        t = pd.read_csv(W / f"lbs_generality/{tool}/full/lbs_{tool}_st100_merged_per_pocket.csv",
                        usecols=["site_uid", "success_tns"], low_memory=False)
        base = base.merge(t.rename(columns={"success_tns": name}), on="site_uid", how="left")
    base = base[base["P2Rank"].notna()].copy()
    base["cls"] = base["pocket_class"].astype(str).str.lower()
    print(f"  Figure 3 base (P2Rank success not null): {len(base):,} rows; "
          f"class counts: {base['cls'].value_counts().to_dict()}")
    print(f"  separation-analysis set (load_lbs):      {len(d):,} rows; "
          f"class counts: {d['cls'].value_counts().to_dict()}")
    inter = len(set(base.loc[base.cls.isin(['apo', 'holo']), 'site_uid']) & set(d.site_uid))
    print(f"  apo/holo rows of the Figure 3 base also in load_lbs: {inter:,}")
    print("\n  raw overall rates, Figure 3 base vs load_lbs set  (apo / holo / gap, pp):")
    for t in PRED:
        r1 = [100 * base.loc[base.cls == c, t].mean() for c in ("apo", "holo")]
        r2 = [100 * d.loc[d.cls == c, t].mean() for c in ("apo", "holo")]
        print(f"    {t:12s} Fig3 {r1[0]:5.1f} / {r1[1]:5.1f} / {r1[1]-r1[0]:+5.1f}"
              f"     load_lbs {r2[0]:5.1f} / {r2[1]:5.1f} / {r2[1]-r2[0]:+5.1f}")
    print("  (RESULTS.md quotes IF-SitePred 46.2/68.1 +21.8, P2Rank 46.8/63.1 +16.3, "
          "fpocket 49.9/62.9 +13.1)")


# =========================================================================== 1. level
def section1(gaps, gdock):
    print("\n" + "=" * 96)
    print("1. THE LEVEL -- overall class gap, four ways (Figure 3's quantity, then its variants)")
    print("=" * 96)
    res = {}
    for t in TOOLS:
        g = gaps[t]
        res[t] = series(g, np.zeros(len(g.d), int), 1)
    dres = series(gdock, np.zeros(len(gdock.d), int), 1)

    for key, lab, unit in (("gap_raw", "raw pp gap (Figure 3)", "pp"),
                           ("gap_std", "size-standardized pp gap", "pp"),
                           ("lor_raw", "raw log odds ratio", "ln"),
                           ("lor_std", "size-standardized log odds ratio", "ln")):
        print(f"\n  --- {lab} ---")
        pts, bss = {}, {}
        for t in TOOLS:
            p, b = res[t][key]
            pts[t], bss[t] = float(p[0]), b[:, 0]
            rec("1_level", key, "overall", key.split("_")[1], t, pts[t], bss[t])
            arms = res[t]["arms_raw" if key.endswith("raw") else "arms_std"][0][0]
            line(t, pts[t], bss[t], unit, tail=f"   apo {arms[0]:5.1f}  holo {arms[1]:5.1f}")
        p, b = dres[key]
        line("docking (own row set, reference)", float(p[0]), b[:, 0], unit)
        rec("1_level", key, "overall", key.split("_")[1], "docking", float(p[0]), b[:, 0])
        for a, b_ in PAIRS:
            dif = bss[a] - bss[b_]
            rec("1_level", key + "_contrast", "overall", key.split("_")[1], f"{a}-{b_}",
                pts[a] - pts[b_], dif)
            line(f"  {a} - {b_}", pts[a] - pts[b_], dif, unit)
        print(f"  P(fpocket has the SMALLEST gap of the three) = "
              f"{rank_share(pts, bss, 'fpocket', largest=False):.1f}%   "
              f"P(IF-SitePred the LARGEST) = {rank_share(pts, bss, 'IF-SitePred'):.1f}%")
    return res


# =========================================================================== 2/3. slope
def slope_section(gaps, gdock, axis, title, tag):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    res = {}
    for t in TOOLS:
        g = gaps[t]
        res[t] = series(g, codes_from(g.d[axis], TIERS), 5)
    dres = series(gdock, codes_from(gdock.d[axis], TIERS), 5)
    n = res[TOOLS[0]]["n"]
    print("  n per tier: " + "  ".join(f"{s}={v:,}" for s, v in zip(SHORT, n)))

    for key, lab in (("gap_raw", "RAW gap by tier"), ("gap_std", "SIZE-STANDARDIZED gap by tier")):
        adj = key.split("_")[1]
        print(f"\n  --- {lab} (pp) ---")
        print(f"  {'':14s}" + "".join(f"{s:>8s}" for s in SHORT) + "      span (v.high - v.low)")
        pts, bss = {}, {}
        for t in TOOLS + ["docking"]:
            p, b = (dres if t == "docking" else res[t])[key]
            span, bspan = p[4] - p[0], b[:, 4] - b[:, 0]
            lo, hi = ci(bspan)
            star = " *" if (lo > 0 or hi < 0) else ""
            print(f"  {t:14s}" + "".join(f"{v:8.1f}" for v in p)
                  + f"   {span:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]{star}")
            for i, tier in enumerate(TIERS):
                rec(tag, key, tier, adj, t, p[i], b[:, i])
            rec(tag, key + "_span", "v.high-v.low", adj, t, span, bspan)
            if t != "docking":
                pts[t], bss[t] = span, bspan
        for a, b_ in PAIRS:
            line(f"  span {a} - {b_}", pts[a] - pts[b_], bss[a] - bss[b_])
            rec(tag, key + "_span_contrast", "v.high-v.low", adj, f"{a}-{b_}",
                pts[a] - pts[b_], bss[a] - bss[b_])
        print(f"  P(fpocket has the WIDEST span of the three) = "
              f"{rank_share(pts, bss, 'fpocket'):.1f}%")
        # is fpocket's span made at the floor or at the ceiling? tier-wise paired contrasts
        print(f"  fpocket - IF-SitePred, tier by tier ({adj}):")
        pa, ba = res["fpocket"][key]
        pb, bb = res["IF-SitePred"][key]
        for i, s in enumerate(SHORT):
            line(f"    at {s}", pa[i] - pb[i], ba[:, i] - bb[:, i], width=20)
            rec(tag, key + "_tier_contrast", TIERS[i], adj, "fpocket-IF-SitePred",
                pa[i] - pb[i], ba[:, i] - bb[:, i])

    print("\n  --- does standardization CHANGE any tool's span?  (std span - raw span, paired) ---")
    for t in TOOLS + ["docking"]:
        r = dres if t == "docking" else res[t]
        ps, bs_ = r["gap_std"]
        pr, br = r["gap_raw"]
        d_pt = (ps[4] - ps[0]) - (pr[4] - pr[0])
        d_bs = (bs_[:, 4] - bs_[:, 0]) - (br[:, 4] - br[:, 0])
        line(t, d_pt, d_bs)
        rec(tag, "span_std_minus_raw", "v.high-v.low", "std-raw", t, d_pt, d_bs)

    print("\n  --- relative scale: log odds ratio by tier, and its span ---")
    for key in ("lor_raw", "lor_std"):
        adj = key.split("_")[1]
        print(f"  {adj:5s}{'':9s}" + "".join(f"{s:>8s}" for s in SHORT) + "      span")
        pts, bss = {}, {}
        for t in TOOLS:
            p, b = res[t][key]
            span, bspan = p[4] - p[0], b[:, 4] - b[:, 0]
            lo, hi = ci(bspan)
            print(f"  {t:14s}" + "".join(f"{v:8.2f}" for v in p)
                  + f"   {span:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]")
            pts[t], bss[t] = span, bspan
            rec(tag, key + "_span", "v.high-v.low", adj, t, span, bspan)
        print(f"  P(fpocket widest span, log-OR, {adj}) = {rank_share(pts, bss, 'fpocket'):.1f}%")
    return res, dres


# =========================================================================== 4. arms
def section_arms(res, axis_label, tag):
    print("\n" + "=" * 96)
    print(f"4. WHICH ARM -- change in each class's rate across {axis_label} (v.high - v.low)")
    print("=" * 96)
    for key, lab in (("arms_raw", "RAW"), ("arms_std", "SIZE-STANDARDIZED")):
        adj = key.split("_")[1]
        print(f"\n  --- {lab} ---")
        print(f"  {'':14s}{'apo v.low':>10s}{'apo v.high':>11s}{'d apo':>22s}"
              f"{'holo v.low':>13s}{'holo v.high':>12s}{'d holo':>22s}")
        pa, ph, ba, bh = {}, {}, {}, {}
        for t in TOOLS:
            r, b = res[t][key]
            pa[t], ph[t] = r[4, 0] - r[0, 0], r[4, 1] - r[0, 1]
            ba[t], bh[t] = b[:, 4, 0] - b[:, 0, 0], b[:, 4, 1] - b[:, 0, 1]
            la, ha = ci(ba[t])
            lh, hh = ci(bh[t])
            print(f"  {t:14s}{r[0,0]:10.1f}{r[4,0]:11.1f}   {pa[t]:+6.1f} [{la:+5.1f},{ha:+5.1f}]"
                  f"{r[0,1]:13.1f}{r[4,1]:12.1f}   {ph[t]:+6.1f} [{lh:+5.1f},{hh:+5.1f}]")
            rec(tag, "apo_arm_change", "v.high-v.low", adj, t, pa[t], ba[t])
            rec(tag, "holo_arm_change", "v.high-v.low", adj, t, ph[t], bh[t])
        for a, b_ in PAIRS:
            line(f"  apo-arm change {a} - {b_}", pa[a] - pa[b_], ba[a] - ba[b_])
            rec(tag, "apo_arm_change_contrast", "v.high-v.low", adj, f"{a}-{b_}",
                pa[a] - pa[b_], ba[a] - ba[b_])
        print(f"  P(fpocket has the largest apo-arm FALL of the three) = "
              f"{rank_share(pa, ba, 'fpocket', largest=False):.1f}%")


# =========================================================================== 5. composition
def section_composition(gaps, axis, axis_label, tag):
    """Kitagawa split of the overall RAW gap:
         gap = sum_t wbar_t (H_t - A_t)  +  sum_t (h_t - a_t) (A_t + H_t)/2
       with a_t, h_t the tier shares of each class, wbar = (a+h)/2. The first term is the
       within-tier gap; the second is what the class-tier association contributes on its own."""
    print("\n" + "=" * 96)
    print(f"5. COMPOSITION -- how much of the overall RAW gap is within-{axis_label} gap, and how")
    print(f"   much is the AF-holo excess in high-{axis_label} tiers (where every class does better)?")
    print("=" * 96)
    g0 = gaps[TOOLS[0]]
    tier = codes_from(g0.d[axis], TIERS)
    ish = g0.ish
    Y = np.column_stack([gaps[t].y for t in TOOLS])

    def split(ix=None):
        tt, hh, YY = (tier, ish, Y) if ix is None else (tier[ix], ish[ix], Y[ix])
        m = tt >= 0
        tt, hh, YY = tt[m], hh[m], YY[m]
        na = np.bincount(tt[~hh], minlength=5).astype(float)
        nh = np.bincount(tt[hh], minlength=5).astype(float)
        a, h = na / na.sum(), nh / nh.sum()
        out = np.empty((len(TOOLS), 3))
        for j in range(len(TOOLS)):
            sa = np.bincount(tt[~hh], weights=YY[~hh, j], minlength=5)
            sh = np.bincount(tt[hh], weights=YY[hh, j], minlength=5)
            A = np.where(na > 0, sa / np.where(na > 0, na, 1), 0.0)
            H = np.where(nh > 0, sh / np.where(nh > 0, nh, 1), 0.0)
            within = ((a + h) / 2 * (H - A)).sum()
            comp = ((h - a) * (A + H) / 2).sum()
            out[j] = 100.0 * np.array([within + comp, within, comp])
        return out, a, h

    pt, a, h = split()
    bs = np.stack([split(ix)[0] for ix in g0._indices()])
    print("  tier shares  " + "".join(f"{s:>8s}" for s in SHORT))
    print("    AF-apo     " + "".join(f"{v:8.3f}" for v in a))
    print("    AF-holo    " + "".join(f"{v:8.3f}" for v in h))
    print(f"\n  {'':14s}{'overall gap':>14s}{'within-tier':>24s}{'composition':>24s}{'comp share':>12s}")
    for j, t in enumerate(TOOLS):
        l1, h1 = ci(bs[:, j, 1])
        l2, h2 = ci(bs[:, j, 2])
        print(f"  {t:14s}{pt[j,0]:14.1f}   {pt[j,1]:+6.1f} [{l1:+5.1f},{h1:+5.1f}]"
              f"   {pt[j,2]:+6.1f} [{l2:+5.1f},{h2:+5.1f}]{100*pt[j,2]/pt[j,0]:11.0f}%")
        rec(tag, "overall_raw_gap", "overall", "raw", t, pt[j, 0], bs[:, j, 0])
        rec(tag, "within_tier_gap", "overall", "raw", t, pt[j, 1], bs[:, j, 1])
        rec(tag, "composition_term", "overall", "raw", t, pt[j, 2], bs[:, j, 2])
    print("\n  within-tier gap, paired contrasts:")
    for a_, b_ in PAIRS:
        ia, ib = TOOLS.index(a_), TOOLS.index(b_)
        line(f"  {a_} - {b_}", pt[ia, 1] - pt[ib, 1], bs[:, ia, 1] - bs[:, ib, 1])
        rec(tag, "within_tier_gap_contrast", "overall", "raw", f"{a_}-{b_}",
            pt[ia, 1] - pt[ib, 1], bs[:, ia, 1] - bs[:, ib, 1])
    print("  composition term, paired contrasts:")
    for a_, b_ in PAIRS:
        ia, ib = TOOLS.index(a_), TOOLS.index(b_)
        line(f"  {a_} - {b_}", pt[ia, 2] - pt[ib, 2], bs[:, ia, 2] - bs[:, ib, 2])
        rec(tag, "composition_term_contrast", "overall", "raw", f"{a_}-{b_}",
            pt[ia, 2] - pt[ib, 2], bs[:, ia, 2] - bs[:, ib, 2])


# =========================================================================== 6. continuous
def section_continuous(gaps, xcol, axis_label, tag):
    """y ~ decile FE x class + x + H*x  (x rank-uniform, centred), all tools as multiple RHS on
    the SAME resample. 100 * coef(H*x) = pp of gap widening across the full range of x. The raw
    version drops the decile fixed effects. No binning, so it is the un-tiered check on the span."""
    print("\n" + "=" * 96)
    print(f"6. CONTINUOUS SLOPE -- gap widening across the full {axis_label} range, no tiers")
    print("=" * 96)
    g0 = gaps[TOOLS[0]]
    d = g0.d
    Y = np.column_stack([gaps[t].y for t in TOOLS])
    h = g0.ish.astype(float)
    x = rank01(d[xcol]) - 0.5
    ok = np.isfinite(x)
    Dm = np.zeros((len(d), K))
    Dm[np.arange(len(d)), g0.zb] = 1.0
    X_std = np.column_stack([Dm, Dm * h[:, None], x, h * x])
    X_raw = np.column_stack([np.ones(len(d)), h, x, h * x])

    def fit(X, ix=None):
        A, b, o = (X, Y, ok) if ix is None else (X[ix], Y[ix], ok[ix])
        return 100.0 * np.linalg.lstsq(A[o], b[o], rcond=None)[0][-1]

    for X, adj in ((X_raw, "raw"), (X_std, "std")):
        pt = fit(X)
        bs = np.stack([fit(X, ix) for ix in g0._indices()])
        print(f"\n  --- {adj} ---")
        pts, bss = {}, {}
        for j, t in enumerate(TOOLS):
            pts[t], bss[t] = pt[j], bs[:, j]
            line(t, pt[j], bs[:, j])
            rec(tag, "continuous_slope", "full-range", adj, t, pt[j], bs[:, j])
        for a, b_ in PAIRS:
            ia, ib = TOOLS.index(a), TOOLS.index(b_)
            line(f"  {a} - {b_}", pt[ia] - pt[ib], bs[:, ia] - bs[:, ib])
            rec(tag, "continuous_slope_contrast", "full-range", adj, f"{a}-{b_}",
                pt[ia] - pt[ib], bs[:, ia] - bs[:, ib])
        print(f"  P(fpocket steepest of the three) = {rank_share(pts, bss, 'fpocket'):.1f}%")


# =========================================================================== 7. ledger
def section_ledger():
    print("\n" + "=" * 96)
    print("7. LEDGER -- the ordering of the three detectors on every quantity above")
    print("=" * 96)
    df = pd.DataFrame(ROWS)
    keep = df[df["tool"].isin(PRED) & ~df["quantity"].str.contains("contrast")]
    keys = [("1_level", "gap_raw", "overall"), ("1_level", "gap_std", "overall"),
            ("1_level", "lor_raw", "overall"), ("1_level", "lor_std", "overall"),
            ("2_separation", "gap_raw_span", "v.high-v.low"),
            ("2_separation", "gap_std_span", "v.high-v.low"),
            ("2_separation", "lor_raw_span", "v.high-v.low"),
            ("2_separation", "lor_std_span", "v.high-v.low"),
            ("6_sep_continuous", "continuous_slope", "full-range"),
            ("2_separation", "gap_raw", "very_low"), ("2_separation", "gap_std", "very_low"),
            ("2_separation", "gap_raw", "very_high"), ("2_separation", "gap_std", "very_high"),
            ("4_sep_arms", "apo_arm_change", "v.high-v.low"),
            ("4_sep_arms", "holo_arm_change", "v.high-v.low"),
            ("5_sep_composition", "within_tier_gap", "overall"),
            ("5_sep_composition", "composition_term", "overall"),
            ("3_confidence", "gap_raw_span", "v.high-v.low"),
            ("3_confidence", "gap_std_span", "v.high-v.low"),
            ("6_conf_continuous", "continuous_slope", "full-range"),
            ("3_confidence", "gap_raw", "very_high"), ("3_confidence", "gap_std", "very_high")]
    print(f"  {'section':18s}{'quantity':22s}{'axis':13s}{'adj':5s}"
          + "".join(f"{t:>13s}" for t in PRED) + "   order (largest first)")
    for sec, q, ax in keys:
        sub = keep[(keep.section == sec) & (keep.quantity == q) & (keep.axis == ax)]
        for adj, s2 in sub.groupby("adjust", sort=False):
            vals = s2.set_index("tool")["estimate"].reindex(PRED)
            order = " > ".join(vals.sort_values(ascending=False).index)
            print(f"  {sec:18s}{q:22s}{ax:13s}{adj:5s}"
                  + "".join(f"{v:13.2f}" for v in vals) + f"   {order}")


def main():
    d = load_lbs()
    print(f"LBS ST_100 pockets on the common row set: {len(d):,}   "
          f"UniProt clusters: {d.site_uniprot.nunique():,}   B = {B}")
    section0(d)
    gaps = build_gaps(d, TOOLS)
    dk = load_docking(d)
    gdock = Gap(dk, "docking")
    print(f"  docking reference set: {len(gdock.d):,} pockets, {dk.site_uniprot.nunique():,} clusters")

    section1(gaps, gdock)
    res_sep, _ = slope_section(gaps, gdock, "sep",
                               "2. THE SLOPE -- gap by SEPARATION tier (Figure 4d's axis)",
                               "2_separation")
    res_conf, _ = slope_section(gaps, gdock, "prob",
                                "3. THE OTHER AXIS -- gap by WINNING-PROBABILITY tier (Figure 3a's axis)",
                                "3_confidence")
    section_arms(res_sep, "separation", "4_sep_arms")
    section_composition(gaps, "sep", "separation", "5_sep_composition")
    section_composition(gaps, "prob", "confidence", "5_conf_composition")
    section_continuous(gaps, "pocket_separation_pair", "separation", "6_sep_continuous")
    section_continuous(gaps, "winning_p", "winning-probability", "6_conf_continuous")
    section_ledger()
    out = Path(P.out("tool_gap_reconciliation.csv"))
    pd.DataFrame(ROWS).to_csv(out, index=False, float_format="%.4f")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
