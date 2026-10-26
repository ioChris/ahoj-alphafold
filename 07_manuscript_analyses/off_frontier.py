# -*- coding: utf-8 -*-
"""The MIRROR of the low-separation case: what happens when the AF pocket is NEITHER apo-like
NOR holo-like -- i.e. it sits where NEITHER experimental class has probability mass.

Backs Supplementary Note 2 and Supplementary Table S8 (the distance to the nearer state, "models far
from both states"), and the main-text numbers that quote them (rho = -0.74 with density; the
size-standardized consensus and docking penalties of the top decile; rho with pLDDT and winning
probability). Usage, on the manuscript's AFDB v4 set:

    AHOJ_AF_MODEL_SET=afdbv4 python off_frontier.py [off_frontier_afdbv4.csv [cache.parquet]] \
        > off_frontier_afdbv4.out.txt

The CSV (quintile and top-decile tables) goes to P.OUT under the name given (default
off_frontier{_afdbv4}.csv); the optional parquet caches the per-job construction block.

Two geometries are easily conflated and this script keeps them apart:

  LOW SEPARATION  the apo and holo ensembles OVERLAP, so the model sits in a region BOTH classes
                  occupy. separation_shape.py shows the class GAP collapses there (Figure 4d,
                  Supplementary Table S3).

  OFF-MANIFOLD    the model sits where NEITHER class has mass. `pocket_density_pair` is the
                  existing handle (rho with separation = -0.011, genuinely orthogonal) and the
                  class gap is FLAT across its tiers (Supplementary Table S3).
                  What was never asked is whether the LEVEL moves -- and it does.

WHAT IS MEASURED, and why not the two obvious candidates
-------------------------------------------------------
Per pocket and per classifier metric m in {volume, SASA, RoG, pocket_rms}, with the model's own
value v and the experimental apo/holo class means mu_a, mu_h over that pocket's conformers:

  d_model  = min(|v - mu_a|, |v - mu_h|)          distance to the NEARER class mode
  sd_w     = within-class SD, recovered from the pooled SD by the two-group decomposition of
             separation_shape.add_shape():  Var_w = Var_tot - (n_a n_h / n^2)(mu_h - mu_a)^2
  offmode  = mean over the four metrics of  d_model / sd_w

That is "how many within-class SDs from the nearer mode", i.e. the direct operationalization of
NEITHER apo-like NOR holo-like. Two alternatives were tested and BOTH are contaminated:

  * `frontier` (d_model < d_states, the single-structure frontier of Supplementary Table S5,
    frontier_final.py) is HALF SEPARATION -- rho(n_frontier, separation) = +0.502 -- because
    d_states is small by construction when the two states coincide. It cannot be the mirror of the
    separation axis; it is partly the same axis.
  * `inside [min,max]` is HALF SAMPLING DEPTH -- rho(n_inside, n_conformers) = +0.513 -- as the
    sampling-depth stratification of the marginal rate already showed. Its downstream association
    collapses once conformer count is held fixed.

Estimator throughout is Figure 4's own, imported not reimplemented: size-standardized mean
difference (holo - apo) by direct adjustment to the `union_rg` decile distribution, cluster
bootstrap over `site_uniprot`, B = 400, every cell of a table on the SAME resample so contrasts
carry their own interval. The MARGINAL success level is the same direct adjustment carried one step
further -- Gap.rates() gives the size-standardized apo and holo rates, and they are combined at the
task's own fixed class mix, so it is a class-and-size standardized rate, not a new estimator.

Raw is reported beside standardized everywhere, per the settled constraint: here standardization
HALVES the level effect and removes essentially all of the apparent gap effect.
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from separation_shape import MODEL_SFX, Gap, W, ci, load_docking, load_lbs, qcodes

DB = P.cfg.AHOJ_DB_DIR
MERGED = P.cfg.CLASSIFICATION_MERGED_DIR
STATS = P.cfg.APOHOLO_METRIC_STATS_CSV
MODEL = (DB / "Summaries/model_metrics/full_run_3/run_23082026_180200_do_collect_model_pLDDT"
            / "query_summaries_do_collect_model_pLDDT.csv")
UNI = W / "unified_output/run5/analysis_unified_levels.csv"

MET = ["volume", "sasa", "RoG", "pocket_rms"]
MODCOL = {"volume": "model_pocket_volume", "sasa": "model_pocket_sasa",
          "RoG": "model_pocket_RoG", "pocket_rms": "model_pocket_rms"}
PRED = ["P2Rank", "fpocket", "IF-SitePred"]
QLAB5 = ["Q1 on-manifold", "Q2", "Q3", "Q4", "Q5 OFF-manifold"]


def off_cut(d):
    """The single threshold that defines "off-manifold" as a CATEGORY, everywhere in this script.

    Taken once on the whole LBS set and held fixed. It must not be re-derived per task or per
    subset: `qcodes(frame["offmode"], 5) == 4` gives the 80th percentile OF THAT FRAME, which is
    1.447 on the 86,651-pocket detection set and 1.310 on the 60,689-pocket docking set -- so the
    same words would name two different populations (1,495 docking pockets differ) and the sec.3
    and sec.4 tables would not be comparable. Quintiles as an ORDERED AXIS (sec.1, sec.2) are a
    different matter and stay within-frame: "Q5 of this set" is self-describing, "off-manifold" is
    not.
    """
    return float(pd.to_numeric(d["offmode"], errors="coerce").quantile(0.80))


# ============================================================================== construction
def build_offmanifold(cache=None):
    """Per-job off-manifold / frontier / inside-band statistics on the 237,922 analysis pockets.

    Reproduces Supplementary Table S5 exactly (frontier 66.3 / 65.7 / 61.9 / 54.8%, inside
    84.6 / 87.7 / 84.0 / 76.4%; frontier_final.py) as a check that the source join is the same one,
    then adds the off-manifold columns that table does not have.
    """
    if cache and Path(cache).exists():
        # Print the SAME block on the cache-hit path. Everything it reports is recoverable from
        # the cached columns, and the stdout has to be invocation-independent: the quoted numbers
        # are checked against this stdout, so a cached run that stayed silent here would make six
        # quoted figures look missing. A check that false-fails is worse than no check.
        #
        # The guard is not optional. Adding `has_model` to the cached frame for the line above
        # silently invalidated every cache written before it, and an unguarded read turned that
        # into a KeyError on the documented fast path -- hitting hardest whoever had been running
        # this most. A cache is a derived artefact: when its schema no longer matches, rebuild it
        # rather than trusting it or dying on it.
        cached = pd.read_parquet(cache)
        need = _required_columns()
        missing = need - set(cached.columns)
        if not missing:
            return _report_construction(cached)
        print(f"  [cache] {Path(cache).name} predates the current schema "
              f"({len(missing)} column(s) missing, e.g. {sorted(missing)[0]}) -- rebuilding it.")
    uni = pd.read_csv(UNI, usecols=["threshold", "rep_job"], low_memory=False)
    aset = uni[uni.threshold == "UNCLUSTERED"][["rep_job"]].rename(columns={"rep_job": "job_name"})
    s = pd.read_csv(MERGED / "query_summaries_do_kde_plots_and_classification.csv", low_memory=False,
                    usecols=["job_name", "n_apo", "n_holo", "n_all", "model_pocket_pLDDT_avg",
                             "model_pocket_volume", "model_pocket_sasa", "model_pocket_RoG"])
    mc = ["job_name"]
    for m in MET:
        mc += [f"{m}_{k}" for k in ("min", "max", "mean", "std_dev")]
    q = pd.read_csv(MERGED / "query_metrics_do_kde_plots_and_classification.csv",
                    usecols=mc, low_memory=False)
    st = pd.read_csv(STATS, low_memory=False,
                     usecols=["job_name"] + [f"{m}_{w}_mean" for m in MET for w in ("apo", "holo")])
    rm = pd.read_csv(MODEL, usecols=["job_name", "model_pocket_rms"], low_memory=False)
    d = (aset.merge(s, on="job_name", how="left").merge(q, on="job_name", how="left")
              .merge(st, on="job_name", how="left").merge(rm, on="job_name", how="left"))
    d["has_model"] = d[MODCOL["volume"]].notna()

    na, nh = d.n_apo.astype(float), d.n_holo.astype(float)
    n = na + nh
    keep = ["job_name", "n_apo", "n_holo", "n_all", "model_pocket_pLDDT_avg", "has_model"]
    for m in MET:
        v = d[MODCOL[m]]
        lo, hi, mu, sd = d[f"{m}_min"], d[f"{m}_max"], d[f"{m}_mean"], d[f"{m}_std_dev"]
        a, h = d[f"{m}_apo_mean"], d[f"{m}_holo_mean"]
        rng = (hi - lo).where(lambda x: x > 0)
        ds = (h - a).abs()
        dm = pd.concat([(v - a).abs(), (v - h).abs()], axis=1).min(axis=1)
        okF = v.notna() & ds.notna() & dm.notna() & (ds > 0)
        # within-class SD by the two-group decomposition (separation_shape.add_shape); the pooled SD
        # contains the between-class term and must not be used as a within-class scale.
        sdw = np.sqrt((sd ** 2 - (na * nh / n ** 2) * (h - a) ** 2).clip(lower=0))
        d[f"F_{m}"] = (dm < ds).where(okF)                            # frontier indicator
        d[f"IN_{m}"] = ((v >= lo) & (v <= hi)).where(v.notna() & rng.notna())
        d[f"NZ_{m}"] = (dm / sd.where(sd > 0)).where(v.notna() & dm.notna())     # pooled-SD version
        d[f"NZW_{m}"] = (dm / sdw.where(sdw > 0)).where(v.notna() & dm.notna())  # within-SD version
        d[f"OUT_{m}"] = (np.maximum(lo - v, v - hi) / rng).clip(lower=0).where(v.notna() & rng.notna())
        keep += [f"{p}_{m}" for p in ("F", "IN", "NZ", "NZW", "OUT")]
    out = d[keep]
    if cache:
        out.to_parquet(cache, index=False)
    return _report_construction(out)


def _required_columns():
    """Exactly what a cached frame must carry for the rest of the script to run off it."""
    cols = {"job_name", "n_apo", "n_holo", "n_all", "model_pocket_pLDDT_avg", "has_model"}
    return cols | {f"{p}_{m}" for p in ("F", "IN", "NZ", "NZW", "OUT") for m in MET}


def _report_construction(d):
    """The construction block, from the columns alone, so a cached run prints what a cold run does.

    Returns its argument so it can wrap either return path.
    """
    print(f"analysis set (UNCLUSTERED): {len(d):,}  |  with an AF model: "
          f"{int(d['has_model'].sum()):,}")
    print("  per-metric reproduction of FRONTIER_AND_PLDDT sec.1:")
    for m in MET:
        print(f"    {m:11s} frontier {100*d[f'F_{m}'].mean():5.1f}%   inside [min,max] "
              f"{100*d[f'IN_{m}'].mean():5.1f}%   median d/sd_w {d[f'NZW_{m}'].median():.2f}")
    return d


def load(cache=None):
    """LBS ST_100 set exactly as `separation_shape.load_lbs()` builds it, plus the off-manifold
    block joined on `rep_job` -- which is a `job_name`, so the join is 1:1 and lossless."""
    d = load_lbs()
    # follows AHOJ_AF_MODEL_SET, like separation_shape.load_lbs() above it; same path when unset
    ex = pd.read_csv(W / f"lbs_generality/p2rank{MODEL_SFX}/lbs_p2rank_st100_merged_per_pocket.csv",
                     usecols=["site_uid", "rep_job", "model_pocket_plddt", "density_level_db"],
                     low_memory=False)
    d = d.merge(ex, on="site_uid", how="left")
    assert d.rep_job.nunique() == len(d), "rep_job is not 1:1 with the ST_100 pocket"
    d = d.merge(build_offmanifold(cache).rename(columns={"job_name": "rep_job"}),
                on="rep_job", how="left")

    def finite_mean(cols):
        b = d[cols]
        return b.where(np.isfinite(b)).mean(axis=1)

    d["offmode"] = finite_mean([f"NZW_{m}" for m in MET])
    d["offmode_pooled"] = finite_mean([f"NZ_{m}" for m in MET])
    d["offmode_3"] = finite_mean([f"NZW_{m}" for m in ("sasa", "RoG", "pocket_rms")])
    d["excursion"] = d[[f"OUT_{m}" for m in MET]].mean(axis=1)
    Fc, INc = [f"F_{m}" for m in MET], [f"IN_{m}" for m in MET]
    d["n_front"] = d[Fc].sum(axis=1).where(d[Fc].notna().all(axis=1))
    d["n_in"] = d[INc].sum(axis=1).where(d[INc].notna().all(axis=1))
    d["dens"] = pd.to_numeric(d["pocket_density_pair"], errors="coerce")
    d["n_all"] = pd.to_numeric(d["n_all"], errors="coerce")
    d["plddt"] = pd.to_numeric(d["model_pocket_plddt"], errors="coerce")
    return d


# ============================================================================== small helpers
def rho(d, a, b):
    s = d[[a, b]].apply(pd.to_numeric, errors="coerce")
    s = s[np.isfinite(s).all(axis=1)]
    return (np.nan, 0) if len(s) < 100 else (spearmanr(s[a], s[b])[0], len(s))


def int_codes(s):
    """A 0..k count column used directly as cell codes; missing becomes -1, which `Gap` drops."""
    v = pd.to_numeric(s, errors="coerce").to_numpy(float)
    return np.where(np.isfinite(v), v, -1).astype(int)


def resid_on(x, z):
    """Rank-residualise x on z. Used the same way separation_shape.q1 residualised its splitter
    on size: to ask whether the axis still orders the outcome once the confounder is removed."""
    y = pd.Series(np.asarray(x, float)).rank(pct=True)
    w = pd.Series(np.asarray(z, float)).rank(pct=True)
    ok = y.notna() & w.notna()
    b = np.polyfit(w[ok], y[ok], 1)
    out = pd.Series(np.nan, index=y.index)
    out[ok] = y[ok] - (b[0] * w[ok] + b[1])
    return out.to_numpy()


class Cells:
    """One Gap, one cell partition, everything the tables below need on ONE set of resamples.

    `marg` is the size-standardized apo and holo rates recombined at the task's own fixed class
    mix. Cells differ in class composition, so an unweighted cell mean would confound the level
    with the mix; fixing the weights makes the level comparable across cells. This is Figure 4's
    direct adjustment carried one step further, not a second estimator.
    """

    def __init__(self, g, cell, ncell, labels):
        self.g, self.cell, self.n, self.labels = g, cell, ncell, labels
        self.w_apo = float((g.d["cls"] == "apo").mean())
        self.gap = g.point(cell, ncell)
        self.bgap = g.boot(cell, ncell)
        self.cnt = g.counts(cell, ncell)
        self.rate = g.rates(cell, ncell)
        self.brate = g.boot_rates(cell, ncell)
        self.rate_raw = g.rates(cell, ncell, raw=True)
        self.brate_raw = g.boot_rates(cell, ncell, raw=True)

    def _m(self, r):
        return self.w_apo * r[..., 0] + (1 - self.w_apo) * r[..., 1]

    def marg(self, raw=False):
        return self._m(self.rate_raw if raw else self.rate)

    def bmarg(self, raw=False):
        return self._m(self.brate_raw if raw else self.brate)

    def gap_raw(self):
        return self.rate_raw[:, 1] - self.rate_raw[:, 0]

    def bgap_raw(self):
        return self.brate_raw[:, :, 1] - self.brate_raw[:, :, 0]

    def contrast(self, i, j, raw=False):
        """(marginal, apo, holo, gap), each as (point, lo, hi), for cell i minus cell j."""
        r = self.rate_raw if raw else self.rate
        br = self.brate_raw if raw else self.brate
        gp = self.gap_raw() if raw else self.gap
        bg = self.bgap_raw() if raw else self.bgap
        m, bm = self.marg(raw), self.bmarg(raw)
        out = []
        for pt, bs in ((m[i] - m[j], bm[:, i] - bm[:, j]),
                       (r[i, 0] - r[j, 0], br[:, i, 0] - br[:, j, 0]),
                       (r[i, 1] - r[j, 1], br[:, i, 1] - br[:, j, 1]),
                       (gp[i] - gp[j], bg[:, i] - bg[:, j])):
            lo, hi = ci(bs)
            out.append((pt, lo, hi))
        return out

    def show(self, title, raw=False):
        r = self.rate_raw if raw else self.rate
        gp = self.gap_raw() if raw else self.gap
        bg = self.bgap_raw() if raw else self.bgap
        m = self.marg(raw)
        print(f"\n  {title}{'  [RAW, unadjusted]' if raw else ''}")
        print(f"    {'cell':<32s}{'n':>8s}{'marginal':>10s}{'AF-apo':>9s}{'AF-holo':>9s}"
              f"{'gap':>8s}   95% CI on gap")
        for i, l in enumerate(self.labels):
            lo, hi = ci(bg[:, i])
            print(f"    {l:<32s}{self.cnt[i]:8,d}{m[i]:10.1f}{r[i,0]:9.1f}{r[i,1]:9.1f}"
                  f"{gp[i]:8.1f}   [{lo:+6.1f}, {hi:+6.1f}]")

    def show_contrast(self, label, i, j, raw=False, indent=4):
        names = ["marginal", "AF-apo", "AF-holo", "gap"]
        s = "  ".join(f"{n} {p:+5.1f} [{lo:+5.1f},{hi:+5.1f}]"
                      for n, (p, lo, hi) in zip(names, self.contrast(i, j, raw)))
        print(" " * indent + f"{label:<40s} {s}")


def cells_of(frame, ycol, series, ncell, labels):
    g = Gap(frame, ycol)
    return Cells(g, qcodes(g.d[series], ncell), ncell, labels)


def tail_cells(frame, ycol, series, hot_low=False):
    """Top (or bottom) decile of `series` against the other nine."""
    g = Gap(frame, ycol)
    q = qcodes(g.d[series], 10)
    hot = 0 if hot_low else 9
    return Cells(g, np.where(q == hot, 1, np.where(q >= 0, 0, -1)), 2, ["rest", "tail decile"])


# ============================================================================== sec.0
def s0_construction(d):
    print("\n" + "=" * 100)
    print("0.  WHICH STATISTIC -- and why neither `frontier` nor `inside the range` can be it")
    print("=" * 100)
    print(f"\nLBS ST_100 pockets {len(d):,}   UniProt clusters {d.site_uniprot.nunique():,}")

    print("\n=== 0.1 measurable for 91% of pockets, and the 9% loss is severely non-random ===")
    d["has_off"] = d["offmode"].notna() & np.isfinite(d["offmode"])
    print(d.groupby("has_off").agg(
        n=("site_uid", "size"), median_rg=("union_rg", "median"),
        median_n_conformers=("n_all", "median"), median_sep=("pocket_separation_pair", "median"),
        median_density=("dens", "median"), median_wp=("winning_p", "median"),
        median_plddt=("plddt", "median"), consensus=("consensus", "mean"),
        frac_holo=("cls", lambda s: (s == "holo").mean())).round(3).to_string())
    print(f"  clusters retained: {d[d.has_off].site_uniprot.nunique():,} of "
          f"{d.site_uniprot.nunique():,}")
    print("  READ: the 9% without the statistic have a MEDIAN OF 4 CONFORMERS against 43, and are")
    print("  the worst-performing pockets in the set (0.330 vs 0.603). Off-manifold is only")
    print("  definable where an experimental manifold exists, so all of the below is conditional")
    print("  on that -- it is a descriptor for evidenced sites, not a screen for unevidenced ones.")

    print("\n=== 0.2 the candidate handles against everything they could be confounded with ===")
    cols = ["pocket_separation_pair", "dens", "n_all", "union_rg", "plddt", "winning_p", "consensus"]
    print(f"  {'':26s}" + "".join(f"{h:>13s}" for h in
          ["separation", "density", "n_conform", "union_rg", "pLDDT", "winning_p", "consensus"]))
    for v, lab in [("offmode", "offmode (primary)"), ("offmode_pooled", "offmode, pooled SD"),
                   ("excursion", "excursion, SD-free"), ("dens", "density_pair (existing)"),
                   ("n_front", "n frontier /4"), ("n_in", "n inside band /4")]:
        cells = ["      n/a  " if v == w else f"{rho(d, v, w)[0]:+13.3f}" for w in cols]
        print(f"  {lab:26s}" + "".join(cells))
    print("\n  READ, and this answers 'is the frontier statistic the better operationalization?':")
    print("   * `n_front` carries rho = +0.502 with SEPARATION. The frontier test is d_model <")
    print("     d_states, and d_states is small BY CONSTRUCTION where the ensembles coincide, so")
    print("     half of it is the very axis this analysis is supposed to be orthogonal to. (This")
    print("     is the internal-consistency check FRONTIER_AND_PLDDT.md left open.)")
    print("   * `n_in` carries rho = +0.513 with CONFORMER COUNT -- the sampling-depth confound")
    print("     that document flagged for the marginal rate, now inside the analysis variable.")
    print("   * `offmode` and `density` are both orthogonal to separation and to pLDDT and")
    print("     correlate -0.74 with each other: the same object, on different scales.")

    print("\n=== 0.3 how big is this population? ===")
    m = d["has_off"]
    for thr in (1, 2, 3, 5):
        s = d["offmode"] > thr
        print(f"  more than {thr} within-class SD from the nearer mode: {int(s.sum()):,} "
              f"({100*s.mean():5.1f}% of all pockets, {100*s[m].mean():5.1f}% of measurable ones)"
              f"   median n_conformers {d.loc[s,'n_all'].median():5.0f}   consensus "
              f"{d.loc[s,'consensus'].mean():.3f}")
    ni = d["n_in"]
    print(f"  outside the experimental [min,max] on ALL FOUR metrics: {int((ni==0).sum()):,} "
          f"({100*(ni==0).mean():.1f}% of all, {100*(ni[ni.notna()]==0).mean():.1f}% of complete cases)")
    print(f"  outside it on AT LEAST ONE metric:                      {int((ni<4).sum()):,} "
          f"({100*(ni<4).mean():.1f}% of all, {100*(ni[ni.notna()]<4).mean():.1f}% of complete cases)")

    print("\n=== 0.4 the frontier rate DOES fall at low separation, as predicted ===")
    print("  (FRONTIER_AND_PLDDT.md left this as an open internal-consistency check: at very low")
    print("   separation d_states is small by construction, so the frontier rate should fall there.)")
    order = ["very_low", "low", "moderate", "high", "very_high"]
    tt = (100 * d.groupby("sep")[[f"F_{m}" for m in MET]].mean()).reindex(order)
    tt.columns = [f"frontier% {m}" for m in MET]
    tt.insert(0, "mean n_front /4", d.groupby("sep")["n_front"].mean().reindex(order))
    tt.insert(0, "n", d.groupby("sep").size().reindex(order))
    print(tt.round(1).to_string())
    print("  It falls, on every metric. The frontier statistic is therefore not an independent")
    print("  second axis -- it is partly the separation axis, and must not be used as its mirror.")


# ============================================================================== sec.1
def s1_downstream(d, dk):
    print("\n" + "=" * 100)
    print("1.  THE DOWNSTREAM COST -- a LEVEL effect in every task, and no GAP effect in any")
    print("=" * 100)
    tasks = [(d, "consensus", "consensus"), (d, "P2Rank", "P2Rank"), (d, "fpocket", "fpocket"),
             (d, "IF-SitePred", "IF-SitePred"), (dk, "docking", "docking")]
    res = {}
    for frame, y, lab in tasks:
        c = cells_of(frame, y, "offmode", 5, QLAB5)
        res[lab] = c
        c.show(f"{lab}: quintiles of off-manifold distance   (n = {len(c.g.d):,})")
        c.show_contrast("Q5 - Q1, size-standardized", 4, 0)
        c.show_contrast("Q5 - Q1, RAW (unadjusted)", 4, 0, raw=True)

    print("\n=== 1.0 the whole-set baseline, so the tail can be read against something ===")
    for frame, y, lab in tasks:
        g = Gap(frame, y)
        c = Cells(g, np.zeros(len(g.d), dtype=int), 1, ["whole set"])
        lo, hi = ci(c.bgap[:, 0])
        print(f"  {lab:12s} n {c.cnt[0]:7,d}   marginal {c.marg()[0]:5.1f}   AF-apo {c.rate[0,0]:5.1f}"
              f"   AF-holo {c.rate[0,1]:5.1f}   gap {c.gap[0]:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]")
    print("  The class gap in the off-manifold tail must be compared against THIS, not against zero:")
    print("  the question 'does the label still mean anything out there' is answered by whether the")
    print("  tail's gap differs from the set's, and it does not.")

    print("\n=== 1.1 the headline contrast: the top decile against the other 90% ===")
    print("  A decile because the effect is a TAIL effect -- the marginal is flat across Q1-Q4 and")
    print("  drops in Q5, so a quintile contrast understates the tail and overstates the middle.")
    tail = {}
    for frame, y, lab in tasks:
        c = tail_cells(frame, y, "offmode")
        tail[lab] = c
        c.show_contrast(f"{lab}  (n {c.cnt[1]:,} vs {c.cnt[0]:,})", 1, 0)
    print("\n  READ: five outcomes, five negative marginals, five intervals excluding zero, and BOTH")
    print("  arms move DOWN in every one. Unlike the enhancer result in SEPARATION_SHAPE sec.5 this")
    print("  is not one tool wearing a consensus average.")

    print("\n=== 1.2 the linearity control that killed the enhancer claim, run again here ===")
    per = [res[p].contrast(4, 0)[3][0] for p in PRED]
    print("  per-predictor gap Q5-Q1: " + ", ".join(f"{p} {v:+.2f}" for p, v in zip(PRED, per)))
    print(f"  their mean {np.mean(per):+.2f}  vs consensus {res['consensus'].contrast(4,0)[3][0]:+.2f}"
          "  -- identical, because consensus is the mean of three binaries and the estimator is")
    print("  linear. So a consensus-only gap interval would again be an averaging artefact. Here it")
    print("  does not matter: the gap effect is null or marginal in EVERY tool including consensus,")
    print("  and it is the LEVEL effect that replicates tool by tool.")
    return res, tail


# ============================================================================== sec.2
def s2_traps(d, dk):
    print("\n" + "=" * 100)
    print("2.  THE TWO TRAPS -- 'outside the range' is sampling depth, `frontier` is separation")
    print("=" * 100)
    for frame, y, lab in [(d, "consensus", "consensus"), (dk, "docking", "docking")]:
        g = Gap(frame, y)
        c = Cells(g, int_codes(g.d["n_in"]), 5,
                  ["0 (outside all four)", "1", "2", "3", "4 (inside all four)"])
        c.show(f"{lab}: how many of the four metrics fall inside the experimental [min, max]")
        c.show_contrast("inside-4 minus inside-0", 4, 0)
        c2 = Cells(g, qcodes(resid_on(g.d["n_in"].to_numpy(float), g.d["n_all"].to_numpy()), 5), 5,
                   [f"Q{i+1}" for i in range(5)])
        c2.show_contrast("SAME, splitter residualised on n_conformers", 4, 0)
        cf = Cells(g, int_codes(g.d["n_front"]), 5,
                   ["0 (frontier on none)", "1", "2", "3", "4 (frontier on all)"])
        cf.show(f"{lab}: how many of the four metrics are on the single-structure frontier")
        cf.show_contrast("frontier-4 minus frontier-0", 4, 0)
    print("\n  READ: 'outside the experimental range' looks like a 6-8 pp marginal cost and, for")
    print("  docking, an 8 pp gap effect. Residualise the splitter on conformer count and the gap")
    print("  effect vanishes entirely and the marginal drops to 1-3 pp. The frontier count moves the")
    print("  LEVEL by nothing at all, because its separation content moves the two arms in opposite")
    print("  directions and they cancel. Neither is usable; the off-manifold distance is.")

    print("\n=== 2.1 how much of the quintile effect is evidence depth, on the quintile layout ===")
    print("  (Sec.2.2's grid does this for the top decile; the quintile version is the one that")
    print("   pairs with the Sec.1 table, and conformer count is run as its own axis for scale.)")
    for frame, y, lab in [(d, "consensus", "consensus"), (dk, "docking", "docking")]:
        g = Gap(frame, y)
        raw_q = Cells(g, qcodes(g.d["offmode"], 5), 5, QLAB5)
        raw_q.show_contrast(f"{lab}: offmode Q5-Q1, as reported", 4, 0)
        r = resid_on(pd.to_numeric(g.d["offmode"], errors="coerce").to_numpy(),
                     g.d["n_all"].to_numpy())
        Cells(g, qcodes(r, 5), 5, QLAB5).show_contrast(
            f"{lab}: SAME, residualised on n_conformers", 4, 0)
        Cells(g, qcodes(g.d["n_all"], 5), 5, [f"Q{i+1}" for i in range(5)]).show_contrast(
            f"{lab}: n_conformers Q5-Q1, the confounder itself", 4, 0)

    print("\n=== 2.2 the existing handle, `pocket_density_pair`, on both layouts ===")
    for frame, y, lab in [(d, "consensus", "consensus"), (dk, "docking", "docking")]:
        c = cells_of(frame, y, "dens", 5,
                     ["Q1 LOWEST density", "Q2", "Q3", "Q4", "Q5 highest density"])
        c.show(f"{lab}: quintiles of pocket_density_pair")
        c.show_contrast("Q5 - Q1, size-standardized", 4, 0)
        g = Gap(frame, y)
        lv = g.d["density_level_db"].astype(str).str.strip().str.lower()
        order = ["very_low", "low", "moderate", "high", "very_high"]
        cd = Cells(g, lv.map({v: i for i, v in enumerate(order)}).fillna(-1).astype(int).to_numpy(),
                   5, ["v.low", "low", "mod", "high", "v.high"])
        cd.show(f"{lab}: the PUBLISHED density tiers (`density_level_db`), for comparability with"
                f" DENSITY_SEPARATION.md")
        cd.show_contrast("v.high - v.low, size-standardized", 4, 0)
    print("\n  READ: DENSITY_SEPARATION.md found the class gap flat across density tiers and that")
    print("  holds here. What is new is the LEVEL: density orders it only weakly for detection")
    print("  (compare off-manifold's Q5-Q1 of -5.9 pp) and better for docking. Density is a")
    print("  saturating bounded score; the SD-distance version resolves the tail that density does")
    print("  not, which is where the whole effect lives.")

    print("\n=== 2.3 robustness grid for the top-decile contrast (all size-standardized) ===")
    for frame, y, lab in [(d, "consensus", "consensus"), (dk, "docking", "docking")]:
        print(f"\n  [{lab}]")
        for name, fr, var, rz, low in [
                ("offmode (primary, within-class SD)", frame, "offmode", None, False),
                ("offmode, pooled SD", frame, "offmode_pooled", None, False),
                ("offmode, 3 metrics (drops volume)", frame, "offmode_3", None, False),
                ("offmode | residualised on n_conformers", frame, "offmode", "n_all", False),
                ("excursion beyond [min,max], SD-free", frame, "excursion", None, False),
                ("offmode | n_conformers >= 10", frame[frame.n_all >= 10], "offmode", None, False),
                ("offmode | n_conformers >= 25", frame[frame.n_all >= 25], "offmode", None, False),
                ("offmode | model pLDDT >= 90", frame[frame.plddt >= 90], "offmode", None, False),
                ("density_pair, LOWEST decile", frame, "dens", None, True)]:
            g = Gap(fr, y)
            x = pd.to_numeric(g.d[var], errors="coerce").replace([np.inf, -np.inf], np.nan)
            x = resid_on(x.to_numpy(), g.d[rz].to_numpy()) if rz else x.to_numpy()
            q = qcodes(x, 10)
            hot = 0 if low else 9
            c = Cells(g, np.where(q == hot, 1, np.where(q >= 0, 0, -1)), 2, ["rest", "tail"])
            c.show_contrast(f"{name}  ({c.cnt[1]:,})", 1, 0)


# ============================================================================== sec.3
def s3_two_geometries(d, dk, thr_off):
    print("\n" + "=" * 100)
    print("3.  THE DOUBLE DISSOCIATION -- separation moves the GAP, off-manifold moves the LEVEL")
    print("=" * 100)
    for frame, y, lab in [(d, "consensus", "consensus"), (dk, "docking", "docking")]:
        g = Gap(frame, y)
        lo_sep = g.d["sep"].isin(["very_low", "low"]).to_numpy()
        off = pd.to_numeric(g.d["offmode"], errors="coerce").to_numpy()
        cell = np.where(np.isfinite(off),
                        lo_sep.astype(int) * 2 + (off > thr_off).astype(int), -1)
        c = Cells(g, cell, 4, ["sep mod/high, on-manifold", "sep mod/high, OFF-manifold",
                               "sep LOW, on-manifold", "sep LOW, OFF-manifold"])
        c.show(f"{lab}: separation x off-manifold, 2 x 2")
        c.show_contrast("off-manifold effect | sep mod/high", 1, 0)
        c.show_contrast("off-manifold effect | sep LOW", 3, 2)
        c.show_contrast("low-separation effect | on-manifold", 2, 0)
        c.show_contrast("low-separation effect | OFF-manifold", 3, 1)
    print("\n  READ: the two axes have orthogonal SIGNATURES, not merely orthogonal values. Moving")
    print("  off-manifold costs absolute success in BOTH separation strata and barely touches the")
    print("  gap. Dropping to low separation costs ~0 pp of absolute success in BOTH manifold strata")
    print("  and cuts the gap. Neither substitutes for the other and neither can be inferred from")
    print("  the other -- which is what makes them two descriptors rather than one.")


# ============================================================================== sec.4
def s4_ambiguity(d, dk, thr_off):
    print("\n" + "=" * 100)
    print("4.  WHAT `winning_p` NEAR 0.5 MEANS -- 'in the overlap' or 'outside both'?")
    print("=" * 100)
    amb = d["winning_p"] < 0.60
    print(f"\n  ambiguous stratum, winning_p < 0.60: {int(amb.sum()):,} pockets "
          f"({100*amb.mean():.1f}% of the set)")
    print("  `winning_p` cannot tell the two apart on its own -- it is p(holo) folded at 0.5, so it")
    print("  returns the same value for a model between two coincident ensembles and for a model")
    print("  outside both. Off-manifold distance and separation can, and they are near-orthogonal")
    print(f"  (rho {rho(d,'offmode','pocket_separation_pair')[0]:+.3f}), so the 2x2 is populated.")
    # "off-manifold" MUST mean the same thing here as in sec.3, or the two 2x2s are not comparable
    # and the two cannot both be quoted. Sec.3 uses the top quintile of the whole set, so the cut is
    # taken on the WHOLE set and applied as a fixed threshold inside the stratum -- not re-derived
    # within the subset (which would be a stricter cut on a skewed sub-population) and not the
    # median (which is a different statistic altogether).
    thr_sep = float(pd.to_numeric(d["pocket_separation_pair"], errors="coerce")
                    [d["sep"].isin(["very_low", "low"])].max())
    print(f"\n  cuts, both taken on the WHOLE set and held fixed: off-manifold = offmode > "
          f"{thr_off:.3f} (its 80th\n  percentile, i.e. sec.3's Q5), low separation = the very_low "
          f"+ low tiers (separation <= {thr_sep:.2f}).")
    for frame, y, lab in [(d, "consensus", "consensus"), (dk, "docking", "docking")]:
        g = Gap(frame[frame["winning_p"] < 0.60], y)
        lo_sep = g.d["sep"].isin(["very_low", "low"]).to_numpy()
        off = pd.to_numeric(g.d["offmode"], errors="coerce").to_numpy()
        cell = np.where(np.isfinite(off), lo_sep.astype(int) * 2 + (off > thr_off).astype(int), -1)
        c = Cells(g, cell, 4, ["sep mod/high, near a mode", "sep mod/high, OFF both modes",
                               "sep LOW (overlap), near a mode", "sep LOW (overlap), OFF both"])
        print(f"    [{lab}] the 2x2 covers {int((cell >= 0).sum()):,} of the {len(g.d):,} "
              f"ambiguous pockets -- the rest have no offmode_z.")
        c.show(f"{lab}: inside the ambiguous stratum only")
        c.show_contrast("off-manifold effect | sep mod/high", 1, 0)
        c.show_contrast("off-manifold effect | sep LOW", 3, 2)
        c.show_contrast("low-separation effect | near a mode", 2, 0)
        c.show_contrast("low-separation effect | OFF-manifold", 3, 1)
    print("\n  READ: the class GAP is uniformly small in all four cells -- an ambiguous label is an")
    print("  uninformative label whatever produced the ambiguity, so the two failure modes are")
    print("  indistinguishable on that axis. They are NOT indistinguishable on the level: sitting")
    print("  outside both modes costs absolute success, sitting in the overlap costs none.")


# ============================================================================== sec.5
def s5_plddt(d):
    print("\n" + "=" * 100)
    print("5.  IS OFF-MANIFOLD JUST A LOW-CONFIDENCE MODEL?  No -- they are orthogonal")
    print("=" * 100)
    print()
    for v, lab in [("offmode", "off-manifold distance"), ("excursion", "excursion beyond band"),
                   ("dens", "density_pair"), ("n_in", "n inside band /4"),
                   ("n_front", "n frontier /4")]:
        r, n = rho(d, v, "plddt")
        print(f"  rho({lab:24s}, pocket pLDDT) = {r:+.3f}   n = {n:,}")
    q = qcodes(d["offmode"], 5)
    mus = [d.loc[q == i, "plddt"].mean() for i in range(5)]
    print("\n  mean pocket pLDDT across off-manifold quintiles: "
          + " -> ".join(f"{m:.2f}" for m in mus) + f"   (span {max(mus)-min(mus):.1f})")
    print("  For scale, FRONTIER_AND_PLDDT.md sec.2 records a span of 1.1 pLDDT points across the")
    print("  whole |evidence ratio| axis -- which it calls negligible -- against 5.8 across")
    print("  conformational diversity. Off-manifold placement is in the negligible class.")
    d["plddt_band"] = pd.cut(d["plddt"], [0, 50, 70, 90, 101],
                             labels=["<50", "50-70", "70-90", ">90"])
    print("\n  the one place the two do meet -- the very low pLDDT tail:")
    print(d.groupby("plddt_band", observed=True).agg(
        n=("site_uid", "size"), median_offmode=("offmode", "median"),
        median_density=("dens", "median"), consensus=("consensus", "mean")).round(3).to_string())
    print("  Below pLDDT 70 the model IS far off-manifold, but that band is under 1% of the set.")
    print("  Across the 99% above it, confidence carries no information about off-manifold")
    print("  placement -- the third orthogonality in this paper after pLDDT vs state and density vs")
    print("  separation, and the reason a cheap confidence filter cannot substitute for the geometry.")


# ============================================================================== sec.6
def s6_evidence_depth(d, dk):
    print("\n" + "=" * 100)
    print("6.  OFF-MANIFOLD x EVIDENCE DEPTH -- the AF-apo penalty GROWS where evidence is richest")
    print("=" * 100)
    print("\n  Two readings of an off-manifold model: (a) nobody has resolved that state, so the")
    print("  manifold is merely undersampled; (b) the protein does not visit it. They differ in what")
    print("  they predict for well-sampled sites -- under (a) the penalty should attenuate with")
    print("  evidence depth, under (b) it should not.")
    for frame, y, lab in [(d, "consensus", "consensus"), (dk, "docking", "docking")]:
        print(f"\n  [{lab}] top-decile off-manifold vs the other 90%, within conformer-count strata")
        for thr, name in [(0, "all sites"), (10, "n_conformers >= 10"),
                          (25, "n_conformers >= 25"), (50, "n_conformers >= 50")]:
            c = tail_cells(frame[frame.n_all >= thr], y, "offmode")
            c.show_contrast(f"{name}  ({c.cnt[1]:,} vs {c.cnt[0]:,})", 1, 0)
    print("\n  READ: the MARGINAL penalty is roughly flat, but the AF-Apo arm's share of it grows")
    print("  monotonically with evidence depth, and the gap grows with it. Treat as EXPLORATORY --")
    print("  it is a three-way contrast on nested subsets with no multiplicity correction. It is")
    print("  reported because it is the only place in this analysis where the two readings above")
    print("  make different predictions, and because it names the candidate set for a dynamics")
    print("  follow-up (see the separate MD idea note).")


# ============================================================================== sec.7
def s7_brief_figures(d, dk):
    """Quoted figures that no earlier section prints.

    Every number which could reach a caption must be regenerable from this script. Everything below
    was quoted in the write-up and was, at one point, produced only by a scratch script that no
    longer exists. Kept in one block rather than scattered, so the provenance audit is a single diff.
    """
    print("\n" + "=" * 100)
    print("7.  FIGURES QUOTED IN OFF_MANIFOLD.md THAT NO EARLIER SECTION PRINTS")
    print("=" * 100)

    print("\n=== 7.1 the linearity control on the LEVEL, not just the gap (sec.1.2 does the gap) ===")
    per = []
    for y in ["P2Rank", "fpocket", "IF-SitePred"]:
        g = Gap(d, y)
        c = Cells(g, qcodes(g.d["offmode"], 5), 5, QLAB5)
        per.append(c.marg()[4] - c.marg()[0])
    gc = Cells(Gap(d, "consensus"), qcodes(Gap(d, "consensus").d["offmode"], 5), 5, QLAB5)
    con = gc.marg()[4] - gc.marg()[0]
    print("  per-predictor marginal Q5-Q1: "
          + ", ".join(f"{p} {v:+.1f}" for p, v in zip(PRED, per)))
    print(f"  their mean {np.mean(per):+.2f}  vs consensus {con:+.2f}  -- the same identity as the")
    print("  gap version, which is why the LEVEL claim rests on the per-tool rows and not on this.")

    print("\n=== 7.2 the decile series behind 'it is a tail effect' ===")
    for frame, y, lab in [(d, "consensus", "consensus"), (dk, "docking", "docking")]:
        g = Gap(frame, y)
        c = Cells(g, qcodes(g.d["offmode"], 10), 10, [f"D{i+1}" for i in range(10)])
        print(f"  [{lab:9s}] marginal " + " ".join(f"{v:5.1f}" for v in c.marg()))
        print(f"  {'':12s} gap      " + " ".join(f"{v:5.1f}" for v in c.gap))

    print("\n=== 7.3 the containment distribution, as percentages ===")
    INc = [f"IN_{m}" for m in MET]
    cc = d[INc].notna().all(axis=1)
    ki = d.loc[cc, INc].sum(axis=1)
    print(f"  metrics inside [min,max], of 4 (complete cases n = {int(cc.sum()):,}): "
          + "  ".join(f"{k} -> {100*(ki == k).mean():.1f}%" for k in range(5)))

    print("\n=== 7.4 loose ends the brief cites in prose ===")
    r, _ = rho(d, "offmode", "offmode_pooled")
    print(f"  rho(offmode, its pooled-SD variant)          = {r:+.3f}  "
          "(same ordering, different separation contamination)")
    dens = pd.to_numeric(d["dens"], errors="coerce")
    print(f"  pocket_density_pair: median {dens.median():.2f},  "
          f"{100*(dens <= 0.05).mean():.1f}% at or below 0.05,  {100*(dens == 0).mean():.2f}% at 0")
    print("  ^ this is why density cannot resolve the tail: it is bounded, two-decimal and piles up")
    print("    well away from its own floor, where offmode_z runs to 490.")
    ok = d["offmode"].notna() & np.isfinite(d["offmode"])
    for lab, m in (("with the statistic", ok), ("without it", ~ok)):
        print(f"  selection loss, {lab:19s}: {100*(d.loc[m, 'cls'] == 'holo').mean():.1f}% AF-Holo")

    print("\n=== 7.5 the cohort where 'nobody has resolved that state' is NOT available ===")
    print("  Far off-manifold AND heavily sampled experimentally AND confidently modelled. Quoted in")
    print("  the brief's interpretation section and in the MD follow-up note; not a paper number.")
    for thr, na, pl in ((3, 20, 90), (2, 10, 90)):
        s = d[(d.offmode > thr) & (d.n_all >= na) & (d.plddt >= pl)]
        print(f"    offmode_z > {thr}, n_conformers >= {na:2d}, pLDDT >= {pl}: {len(s):5,d} pockets, "
              f"{s.site_uniprot.nunique():4,d} UniProts, median n_conformers {s.n_all.median():3.0f}, "
              f"consensus {s.consensus.mean():.3f}, {100*(s.cls == 'apo').mean():.0f}% AF-Apo")


# ============================================================================== sec.8
def s8_which_task(d, dk, thr_off):
    """Is docking hurt MORE than detection, and do the three detectors differ from each other?

    sec.1 gives -5.9 for consensus and -9.9 for docking, but on DIFFERENT row sets (86,651 vs
    60,689), so it cannot support a comparison -- the two numbers differ partly because the docking
    set is a non-random subset. Everything here is on ONE row set with shared resamples, which is
    what lets a task-vs-task difference carry its own interval.
    """
    print("\n" + "=" * 100)
    print("8.  WHICH TASK IS HURT MORE? -- one row set, shared resamples, so the difference has a CI")
    print("=" * 100)

    def off_cells(frame, ycol):
        g = Gap(frame, ycol)
        off = pd.to_numeric(g.d["offmode"], errors="coerce").to_numpy()
        cell = np.where(np.isfinite(off), (off > thr_off).astype(int), -1)
        return Cells(g, cell, 2, ["on-manifold", "OFF-manifold"])

    print("\n  detection vs docking, both on the docking row set (both outcomes non-null):")
    cc, cd = off_cells(dk, "consensus"), off_cells(dk, "docking")
    # Print rows AND clusters for the same object, twice, rather than one of each. Pairing a
    # filtered row count with an unfiltered cluster count is the error this line used to make, and
    # it is exactly what the provenance checker cannot see: both numbers are real, they just do not
    # describe the same set.
    ok = np.isfinite(pd.to_numeric(dk["offmode"], errors="coerce"))
    print(f"    Gap frame          : {len(dk):,} docking pockets / "
          f"{dk.site_uniprot.nunique():,} clusters (what the bootstrap resamples)")
    print(f"    entering the cells : {int(cc.cnt.sum()):,} pockets / "
          f"{dk.loc[ok, 'site_uniprot'].nunique():,} clusters (offmode_z defined)")
    print("    The 1,759 clusters that contribute no rows are zero-variance bootstrap units; "
          "pre-filtering\n    the frame instead changes the interval by <0.1 pp, so the choice "
          "does not matter here.")
    for lab, c in (("consensus", cc), ("docking", cd)):
        c.show_contrast(f"{lab:10s} OFF - on", 1, 0)
    dif = ((cd.bmarg()[:, 1] - cd.bmarg()[:, 0]) - (cc.bmarg()[:, 1] - cc.bmarg()[:, 0]))
    pt = (cd.marg()[1] - cd.marg()[0]) - (cc.marg()[1] - cc.marg()[0])
    lo, hi = ci(dif)
    print(f"    PAIRED docking - consensus  {pt:+5.1f} [{lo:+5.1f}, {hi:+5.1f}] pp"
          f"   ({100*(dif < 0).mean():.0f}% of resamples negative)")
    print("    Reads mechanistically: detection needs the site to be RECOGNISABLE, docking needs")
    print("    the cavity to actually ADMIT the ligand. Do not over-read a barely-clearing interval.")

    print("\n  the three detectors against each other, same row set (the full LBS set), same cut:")
    cs = {p_: off_cells(d, p_) for p_ in PRED}
    for p_, c in cs.items():
        c.show_contrast(f"{p_:12s} OFF - on", 1, 0)
    print("    pairwise differences of those contrasts, on shared resamples:")
    for a, b in (("fpocket", "IF-SitePred"), ("fpocket", "P2Rank"), ("P2Rank", "IF-SitePred")):
        da = cs[a].bmarg()[:, 1] - cs[a].bmarg()[:, 0]
        db = cs[b].bmarg()[:, 1] - cs[b].bmarg()[:, 0]
        pt = ((cs[a].marg()[1] - cs[a].marg()[0]) - (cs[b].marg()[1] - cs[b].marg()[0]))
        lo, hi = ci(da - db)
        star = "" if lo < 0 < hi else "   *"
        print(f"      {a:12s} - {b:12s} {pt:+5.1f} [{lo:+5.1f}, {hi:+5.1f}] pp{star}")
    print("    READ: every detector pair crosses zero. The unanimity in sec.1 is therefore stronger")
    print("    than 'all five point the same way' -- the three detectors are statistically")
    print("    INDISTINGUISHABLE on this axis, and the only task difference that separates is that")
    print("    docking is hit harder. Contrast sec.1's SEPARATION result, where the tools disagree")
    print("    about which arm moves.")


# ============================================================================== sec.9
def s9_within_confidence(d, dk, thr_off):
    """Does the penalty survive inside every confidence tier, and is the class gap preserved?

    This is the direct evidence for the main-text sentence's claim that the classifier's own
    confidence gives no warning. sec.5 shows pLDDT is orthogonal; this shows `winning_p` is too,
    in the only way that matters -- the penalty is still there at every level of it.
    """
    print("\n" + "=" * 100)
    print("9.  THE PENALTY INSIDE EVERY CONFIDENCE TIER -- and the class gap is undisturbed")
    print("=" * 100)
    tiers = ["very_low", "low", "moderate", "high", "very_high"]
    for frame, y, lab in [(d, "consensus", "consensus"), (dk, "docking", "docking")]:
        print(f"\n  [{lab}] off-manifold minus on-manifold, within each `prob_level_db` tier")
        print(f"    {'tier':<12s}{'n on':>9s}{'n OFF':>8s}{'marginal OFF - on':>26s}"
              f"{'gap on':>9s}{'gap OFF':>9s}")
        for t in tiers:
            sub = frame[frame["prob"] == t]
            g = Gap(sub, y)
            off = pd.to_numeric(g.d["offmode"], errors="coerce").to_numpy()
            cell = np.where(np.isfinite(off), (off > thr_off).astype(int), -1)
            c = Cells(g, cell, 2, ["on", "OFF"])
            m, lo, hi = c.contrast(1, 0)[0]
            print(f"    {t:<12s}{c.cnt[0]:9,d}{c.cnt[1]:8,d}"
                  f"{m:+10.1f} [{lo:+5.1f}, {hi:+5.1f}]{c.gap[0]:9.1f}{c.gap[1]:9.1f}")
    print("\n  READ: the level penalty survives in every tier of the classifier's own confidence,")
    print("  and the class gap keeps its gradient inside both manifold strata -- the classifier is")
    print("  doing its job normally out there, it simply has nothing to say about whether the model")
    print("  landed on the experimental manifold. That is the evidence behind the main-text clause")
    print("  'neither model confidence nor the winning probability gives any warning'.")
    print("  ! The consensus penalty LOOKS like it shrinks across tiers. The endpoints overlap")
    print("    heavily and no trend test was run. Do not write it as a trend.")


def main():
    cache = P.out(sys.argv[2]) if len(sys.argv) > 2 else None
    d = load(cache)
    dk = load_docking(d)
    print(f"\ndocking subset: {len(dk):,} pockets, {dk.site_uniprot.nunique():,} clusters")
    s0_construction(d)
    res, tail = s1_downstream(d, dk)
    s2_traps(d, dk)
    thr_off = off_cut(d)
    print("\noff-manifold category cut, fixed on the whole set for sec.3 and sec.4: "
          f"offmode_z > {thr_off:.3f}")
    s3_two_geometries(d, dk, thr_off)
    s4_ambiguity(d, dk, thr_off)
    s5_plddt(d)
    s6_evidence_depth(d, dk)
    s7_brief_figures(d, dk)
    s8_which_task(d, dk, thr_off)
    s9_within_confidence(d, dk, thr_off)

    out_csv = P.out(sys.argv[1] if len(sys.argv) > 1 else f"off_frontier{MODEL_SFX}.csv")
    if out_csv:
        rows = []
        for table, book in (("offmode_quintile", res), ("offmode_top_decile", tail)):
            for lab, c in book.items():
                for i, l in enumerate(c.labels):
                    rows.append({"table": table, "task": lab, "cell": l, "n": int(c.cnt[i]),
                                 "marginal_std": c.marg()[i], "rate_apo_std": c.rate[i, 0],
                                 "rate_holo_std": c.rate[i, 1], "gap_std": c.gap[i],
                                 "marginal_raw": c.marg(raw=True)[i],
                                 "rate_apo_raw": c.rate_raw[i, 0],
                                 "rate_holo_raw": c.rate_raw[i, 1], "gap_raw": c.gap_raw()[i]})
        pd.DataFrame(rows).round(3).to_csv(out_csv, index=False)
        print(f"\n[OK] -> {out_csv}")


if __name__ == "__main__":
    main()
