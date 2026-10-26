# -*- coding: utf-8 -*-
"""Is low separation TWO things, and does separation ENHANCE a confident label?

Figure 4's estimator and frames, and four questions asked with them. The module is also the shared
library of the supplementary analyses: `load_lbs()` (the 87,138-pocket ST_100 detection set of
Figure 4 with the P2Rank / fpocket / IF-SitePred DCA4-TNS outcomes and their per-pocket consensus),
`load_docking()` (its docking subset, mean per-pocket dockable fraction) and `Gap` (the
size-standardized AF-Holo - AF-Apo gap with cluster-bootstrap intervals) are imported by
off_frontier.py, density_exclusion_gaps.py and others. Its own output backs the main-text statement
that at very low separation no single predictor's gap differs from zero (Figure 4d; Q4 below) and
the arm-level rates behind "the gap widens because both classes move" (Q3). An earlier analysis had
established that Figure 4d's separation gradient flattens inside a fixed winning-probability tier
while the confidence gradient does not; two questions were left open, and this script answers both
with Figure 4d's own estimator.

Q1  `pocket_separation_pair` = 1 - distributional overlap. Overlap conflates the distance between
    the two class means with the spread inside each class: two ensembles can overlap heavily and
    still have distant members, if both are broad. So a "very low separation" pocket may be a
    RIGID COINCIDENT one (both ensembles narrow and on top of each other) or a BROAD OVERLAPPING
    one (both ensembles wide, sampling the same space). Those are the geometric signatures of two
    different stories. If the class gap differs between them, overlap-separation is the wrong axis
    and a normalized effect size -- between over within, a Cohen's d analogue -- is the right one.

Q2  The author's hypothesis is that separation is AF-pocket agnostic: the two states can be far
    apart while THIS pocket is still an uncertain call. That predicts an INTERACTION -- the gap
    largest where the states separate AND the label is confident -- which is a different claim
    from the conditional main effect already tested, and it has never been tested.

WHAT THE DESCRIPTOR COLUMNS ACTUALLY ARE (verified against the generator,
`04_p2rank/analysis/analyze_p2rank_descriptor_correlations.py` in this repository):

  * `{m}_apo_holo_avg_delta` = `{m}_holo_avg` MINUS `{m}_apo_avg`. Verified numerically below.
  * `pocket_rms` is a distance FROM THE AHoJ QUERY STRUCTURE, which carries a ligand and is
    therefore one of the holo structures. So the apo/holo means are RADIAL coordinates around a
    holo anchor: holo sits near it, apo further out, and the delta is predominantly negative.
    |delta| is a legitimate between-class distance; the class ranges are radial spreads and so are
    a LOWER BOUND on the true ensemble diameter (two apo structures 2 A from the anchor can be
    4 A apart). Every "within" number below inherits that attenuation.
  * `pocket_rms_std_dev` is pooled over apo AND holo, so it is NOT a within-class spread -- it
    contains the between-class term by construction. Using it raw is the trap this script exists
    to avoid. The within-class part is recovered by the two-group variance decomposition
        Var_total = Var_within + (n_a n_h / n^2) * delta^2,
    checked below (the n-weighted class mean reproduces `pocket_rms_mean` to ~0.002 A).

Estimator throughout is Figure 4's: size-standardized mean difference (holo - apo) by direct
adjustment to the union_rg decile distribution, cluster-bootstrapped over `site_uniprot`. Every
cell of a table is computed on the SAME bootstrap resample, so differences, spans and
differences-in-differences carry their own interval instead of four separate ones to eyeball.

Usage (the manuscript's numbers are the AFDB v4 set):
    AHOJ_AF_MODEL_SET=afdbv4 python separation_shape.py > separation_shape_afdbv4.out.txt
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

W = P.WD
DESC = W / "P2Rank/ubs_run_4/analysis_v1/tables/p2rank_st100_descriptor_features_per_ubs.csv"
# AHOJ_AF_MODEL_SET=afdbv4 reads the detection and docking tables of the re-run on the AlphaFold DB v4
# models (the manuscript's set); unset, the July tables computed on v6 files. This script writes nothing.
MODEL_SFX = P.MODEL_SFX
DOCK = W / f"docking/run3{MODEL_SFX}/analysis/docking_run3_merged.csv"
S = "_mean_across_jobs"

PRED = ["P2Rank", "fpocket", "IF-SitePred"]
TIERS = ["very_low", "low", "moderate", "high", "very_high"]
SHORT = ["v.low", "low", "mod", "high", "v.high"]
K = 10
B = 400
SEED = 20260908


# =============================================================================== data
def load_lbs():
    """The Figure 4 detection set: P2Rank / fpocket / IF-SitePred DCA4-TNS per ST_100 pocket, their
    consensus, apo/holo pockets with a size and a UniProt, in a defined separation and probability tier."""
    cols = ["site_uid", "site_uniprot", "pocket_class", "success_tns", "union_rg",
            "separation_level_db", "prob_level_db", "winning_p", "pocket_p_holo",
            "pocket_separation_pair", "pocket_density_pair", "unit_uid",
            "n_apo_avg", "n_holo_avg"]
    p = pd.read_csv(W / f"lbs_generality/p2rank{MODEL_SFX}/lbs_p2rank_st100_merged_per_pocket.csv",
                    usecols=cols, low_memory=False).rename(columns={"success_tns": "P2Rank"})
    for tool, name in (("fpocket", "fpocket"), ("ifsitepred", "IF-SitePred")):
        t = pd.read_csv(W / f"lbs_generality/{tool}/full{MODEL_SFX}/lbs_{tool}_st100_merged_per_pocket.csv",
                        usecols=["site_uid", "success_tns"], low_memory=False)
        p = p.merge(t.rename(columns={"success_tns": name}), on="site_uid", how="left")
    p = p[p["P2Rank"].notna()].copy()
    p["consensus"] = p[PRED].mean(axis=1)
    p["cls"] = p["pocket_class"].astype(str).str.lower()
    p["sep"] = p["separation_level_db"].astype(str).str.strip().str.lower()
    p["prob"] = p["prob_level_db"].astype(str).str.strip().str.lower()
    p = p[p["cls"].isin(["apo", "holo"]) & p["union_rg"].notna() & p["site_uniprot"].notna()]
    return p[p["sep"].isin(TIERS) & p["prob"].isin(TIERS)].copy()


def add_shape(d, verbose=True):
    """Join the descriptor block and build the between / within / effect-size trio."""
    cols = ["unit_uid", "apo_pockets" + S, "holo_pockets" + S,
            "pocket_rms_mean" + S, "pocket_rms_std_dev" + S,
            "pocket_rms_coefficient_of_variation" + S,
            "pocket_rms_apo_avg" + S, "pocket_rms_holo_avg" + S,
            "pocket_rms_apo_holo_avg_delta" + S,
            "pocket_rms_apo_min" + S, "pocket_rms_apo_max" + S,
            "pocket_rms_holo_min" + S, "pocket_rms_holo_max" + S,
            "pocket_rms_apo_range" + S, "pocket_rms_holo_range" + S,
            "RoG_coefficient_of_variation" + S, "volume_std_dev" + S]
    D = pd.read_csv(DESC, usecols=cols, low_memory=False)
    d = d.merge(D, on="unit_uid", how="left")

    ma, mh = d["pocket_rms_apo_avg" + S], d["pocket_rms_holo_avg" + S]
    dl = d["pocket_rms_apo_holo_avg_delta" + S]
    na, nh = d["apo_pockets" + S], d["holo_pockets" + S]
    sd = d["pocket_rms_std_dev" + S]

    if verbose:
        ok = dl.notna() & ma.notna() & mh.notna()
        print("=== 0. what the descriptor columns are (verified, not assumed) ===")
        print(f"  descriptor join: {d['pocket_rms_mean' + S].notna().sum():,} of {len(d):,} "
              f"pockets ({100*d['pocket_rms_mean'+S].notna().mean():.1f}%)")
        print(f"  delta vs (holo_avg - apo_avg): max|diff| = "
              f"{np.nanmax(np.abs(dl[ok]-(mh-ma)[ok])):.3f} A")
        print(f"  delta vs (apo_avg - holo_avg): max|diff| = "
              f"{np.nanmax(np.abs(dl[ok]-(ma-mh)[ok])):.3f} A")
        print(f"  => sign convention is HOLO MINUS APO.  median {dl.median():+.3f} A, "
              f"mean {dl.mean():+.3f} A, {100*(dl<0).mean():.1f}% negative")
        print(f"  radial anchor is a holo structure: median pocket_rms to apo {ma.median():.3f} A, "
              f"to holo {mh.median():.3f} A")
        pooled = (na * ma + nh * mh) / (na + nh)
        v = pooled.notna() & d["pocket_rms_mean" + S].notna()
        print(f"  decomposition sanity: median |n-weighted class mean - pocket_rms_mean| = "
              f"{np.nanmedian(np.abs(pooled[v]-d['pocket_rms_mean'+S][v])):.4f} A "
              f"(std_dev really is pooled over both classes)")

    n = na + nh
    var_btw = (na * nh / n**2) * dl**2
    var_win = sd**2 - var_btw
    if verbose:
        nn = var_win.notna().sum()
        print(f"  Var_within < 0 in {100*(var_win < 0).sum()/nn:.2f}% of the {nn:,} rows where it "
              f"is computable (aggregation noise; clipped to 0)")

    d["between"] = dl.abs()
    d["within_sd"] = np.sqrt(var_win.clip(lower=0))
    d["within_naive"] = sd                      # the trap: pooled, contains the between term
    d["dprime"] = d["between"] / d["within_sd"].replace(0, np.nan)
    d["dprime_naive"] = d["between"] / sd.replace(0, np.nan)
    # radial reach: how far the nearest apo structure gets inside the holo shell
    d["reach"] = d["pocket_rms_holo_max" + S] - d["pocket_rms_apo_min" + S]
    d["reach_z"] = d["reach"] / d["within_sd"].replace(0, np.nan)
    d["n_struct"] = n
    d["has_shape"] = d["dprime"].notna() & np.isfinite(d["dprime"])
    return d


def load_docking(lbs):
    dk = pd.read_csv(DOCK, low_memory=False,
                     usecols=["site_uid", "ligand_group", "status", "affinity_top",
                              "pose_in_pocket"])
    dk = dk[(dk.ligand_group == 1) & (dk.status.astype(str).str.lower() == "ok")].copy()
    aff = pd.to_numeric(dk["affinity_top"], errors="coerce")
    inpk = dk["pose_in_pocket"].astype(str).str.lower().isin(["true", "1", "yes"])
    dk["_ok"] = (inpk & (aff <= -6)).astype(float)
    pk = dk.groupby("site_uid", as_index=False).agg(docking=("_ok", "mean"))
    return lbs.merge(pk, on="site_uid", how="inner")


# ================================================= Figure 4's estimator, many cells at once
class Gap:
    """Size-standardized holo - apo gap (pp) for an arbitrary cell partition.

    Deciles and the standard population are fixed on the WHOLE frame handed in, so every cell is
    adjusted to one common size distribution. `boot` evaluates all cells on each resample, which
    is what lets a contrast between cells carry its own interval.
    """

    def __init__(self, d, ycol, seed=SEED):
        d = d[d[ycol].notna() & d["union_rg"].notna() & d["site_uniprot"].notna()].copy()
        zb = pd.qcut(pd.to_numeric(d["union_rg"], errors="coerce"), K, labels=False,
                     duplicates="drop")
        d = d[zb.notna()].copy()
        self.d = d
        self.zb = zb[zb.notna()].astype(int).to_numpy()
        self.y = d[ycol].to_numpy(float)
        self.ish = (d["cls"].to_numpy() == "holo")
        self.codes = pd.factorize(d["site_uniprot"])[0]
        w = np.bincount(self.zb, minlength=K).astype(float)
        self.w = w / w.sum()
        self.seed = seed
        self._boot_ix = None

    def _gaps(self, cell, ncell, ix=None):
        y, ish, zb, c = self.y, self.ish, self.zb, cell
        if ix is not None:
            y, ish, zb, c = y[ix], ish[ix], zb[ix], cell[ix]
        m = c >= 0
        lab = (c[m] * K + zb[m]) * 2 + ish[m]
        L = ncell * K * 2
        s = np.bincount(lab, weights=y[m], minlength=L).reshape(ncell, K, 2)
        n = np.bincount(lab, minlength=L).astype(float).reshape(ncell, K, 2)
        ok = (n[:, :, 0] > 0) & (n[:, :, 1] > 0)
        mu = np.where(n > 0, s / np.where(n > 0, n, 1.0), 0.0)
        diff = np.where(ok, mu[:, :, 1] - mu[:, :, 0], 0.0)
        ww = np.where(ok, self.w[None, :], 0.0)
        den = ww.sum(1)
        return np.where(den > 0, 100.0 * (diff * ww).sum(1) / np.where(den > 0, den, 1.0), np.nan)

    def point(self, cell, ncell):
        return self._gaps(cell, ncell)

    def rates(self, cell, ncell, ix=None, raw=False):
        """Size-standardized apo and holo RATES (pp) per cell -- the LEVELS behind the gap.

        Needed twice: to tell a real interaction from a ceiling effect, and to ask which ARM of
        the contrast moves. `raw=True` skips the decile adjustment, which is the comparison that
        shows how much of an apparent arm-level trend is pocket size.
        """
        y, ish, zb, c = self.y, self.ish, self.zb, cell
        if ix is not None:
            y, ish, zb, c = y[ix], ish[ix], zb[ix], cell[ix]
        m = c >= 0
        if raw:
            lab = c[m] * 2 + ish[m]
            s = np.bincount(lab, weights=y[m], minlength=ncell * 2).reshape(ncell, 2)
            n = np.bincount(lab, minlength=ncell * 2).astype(float).reshape(ncell, 2)
            return np.where(n > 0, 100.0 * s / np.where(n > 0, n, 1.0), np.nan)
        lab = (c[m] * K + zb[m]) * 2 + ish[m]
        L = ncell * K * 2
        s = np.bincount(lab, weights=y[m], minlength=L).reshape(ncell, K, 2)
        n = np.bincount(lab, minlength=L).astype(float).reshape(ncell, K, 2)
        mu = np.where(n > 0, s / np.where(n > 0, n, 1.0), 0.0)
        out = np.full((ncell, 2), np.nan)
        for c_ in range(2):
            ok = n[:, :, c_] > 0
            ww = np.where(ok, self.w[None, :], 0.0)
            den = ww.sum(1)
            num = (np.where(ok, mu[:, :, c_], 0.0) * ww).sum(1)
            out[:, c_] = np.where(den > 0, 100.0 * num / np.where(den > 0, den, 1.0), np.nan)
        return out

    def boot_rates(self, cell, ncell, raw=False):
        """(B, ncell, 2) rates -- same resamples as `boot`, so an arm-level change and the gap
        change are computed on one bootstrap and can be compared directly."""
        return np.stack([self.rates(cell, ncell, ix, raw=raw) for ix in self._indices()])

    @staticmethod
    def counts(cell, ncell):
        c = cell[cell >= 0]
        return np.bincount(c, minlength=ncell)

    def _indices(self):
        if self._boot_ix is None:
            rng = np.random.default_rng(self.seed)
            order = np.argsort(self.codes, kind="stable")
            sc = self.codes[order]
            nC = sc.max() + 1
            st = np.searchsorted(sc, np.arange(nC), "left")
            ln = np.searchsorted(sc, np.arange(nC), "right") - st
            out = []
            for _ in range(B):
                pick = rng.integers(0, nC, nC)
                lp, sp = ln[pick], st[pick]
                off = np.zeros(len(lp), dtype=np.int64)
                np.cumsum(lp[:-1], out=off[1:])
                out.append(order[np.arange(int(lp.sum())) - np.repeat(off, lp)
                                 + np.repeat(sp, lp)])
            self._boot_ix = out
        return self._boot_ix

    def boot(self, cell, ncell):
        """(B, ncell) gaps -- one resample per row, shared across cells."""
        return np.vstack([self._gaps(cell, ncell, ix) for ix in self._indices()])


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (np.nan, np.nan) if len(v) < 20 else (np.percentile(v, 2.5), np.percentile(v, 97.5))


def show(label, g, bs, n=None, width=34):
    lo, hi = ci(bs)
    tail = f"   n = {n:,}" if n is not None else ""
    print(f"  {label:<{width}s} {g:+6.1f} pp  [{lo:+6.1f}, {hi:+6.1f}]{tail}")


def codes_from(series, order):
    m = {v: i for i, v in enumerate(order)}
    return series.map(lambda v: m.get(v, -1)).to_numpy()


def qcodes(x, q):
    """Equal-count bin codes, -1 where the value is missing. Ranks first so that a variable with
    heavy ties (reach, reach_frac) still yields exactly `q` non-empty bins -- `pd.qcut` silently
    drops duplicate edges and then the top bin comes back empty."""
    s = pd.to_numeric(pd.Series(np.asarray(x, dtype=float)), errors="coerce")
    r = s.rank(method="first", pct=True)
    b = np.ceil(r * q) - 1.0
    return pd.Series(b).clip(0, q - 1).fillna(-1).astype(int).to_numpy()


def rank01(x):
    """Rank-uniform [0, 1] transform; NaN preserved. Makes an interaction coefficient read as
    'pp of gap across the full range of the variable'."""
    s = pd.to_numeric(x, errors="coerce")
    r = s.rank(pct=True)
    return r.to_numpy()


# ============================================================================ Q1
def q1(d):
    print("\n" + "=" * 92)
    print("QUESTION 1 -- is low separation two different things?")
    print("=" * 92)

    sh = d[d["has_shape"]].copy()
    print("\n=== 1.0 coverage of the shape variables, and whether the loss is random ===")
    d["has_desc"] = d["pocket_rms_mean" + S].notna()
    print(f"  descriptor join at all:              "
          f"{int(d['has_desc'].sum()):,} of {len(d):,} ({100*d['has_desc'].mean():.1f}%)")
    print(f"  usable for the between/within split: {len(sh):,} of {len(d):,} "
          f"({100*len(sh)/len(d):.1f}%), {sh.site_uniprot.nunique():,} UniProt clusters")
    print("  the second loss needs BOTH classes present in the AHoJ jobs, so it is not random:")
    print(d.groupby("has_desc").agg(
        n=("site_uid", "size"), median_rg=("union_rg", "median"),
        median_sep=("pocket_separation_pair", "median"), median_wp=("winning_p", "median"),
        median_n_apo=("n_apo_avg", "median"), median_n_holo=("n_holo_avg", "median"),
        consensus=("consensus", "mean"),
        frac_holo=("cls", lambda s: (s == "holo").mean())).round(3).to_string())
    tab = d.groupby("has_shape").agg(
        n=("site_uid", "size"), median_rg=("union_rg", "median"),
        median_sep=("pocket_separation_pair", "median"), median_wp=("winning_p", "median"),
        median_n_apo=("n_apo_avg", "median"), median_n_holo=("n_holo_avg", "median"),
        consensus=("consensus", "mean"), frac_holo=("cls", lambda s: (s == "holo").mean()))
    print(tab.round(3).to_string())
    keep = d.groupby("sep")["has_shape"].mean().reindex(TIERS)
    print("  retained fraction by separation tier: "
          + "  ".join(f"{t}={v:.2f}" for t, v in keep.items()))

    g_all = Gap(d, "consensus")
    c_all = codes_from(g_all.d["sep"], TIERS)
    p_all = g_all.point(c_all, 5)
    g_sh = Gap(sh, "consensus")
    c_sh = codes_from(g_sh.d["sep"], TIERS)
    p_sh = g_sh.point(c_sh, 5)
    print("\n  does Figure 4d's separation gradient survive on the retained subset?")
    print(f"    full set  {len(g_all.d):,}: " + "  ".join(f"{v:+5.1f}" for v in p_all))
    print(f"    retained  {len(g_sh.d):,}: " + "  ".join(f"{v:+5.1f}" for v in p_sh))

    # ---------------------------------------------------------------- 1.1 the artefact
    print("\n=== 1.1 does within-class spread really rise with separation? ===")
    print("    (it does on the POOLED sd, which contains the between term -- and that is the trap)")
    hdr = f"  {'':22s}" + "".join(f"{s:>9s}" for s in SHORT)
    print(hdr)
    for lab, col in (("pooled sd (trap)", "within_naive"), ("within-class sd", "within_sd"),
                     ("between |delta|", "between"), ("d' = between/within", "dprime")):
        med = sh.groupby("sep")[col].median().reindex(TIERS)
        print(f"  {lab:22s}" + "".join(f"{v:9.3f}" for v in med))
    for lab, col in (("within-class sd p10", "within_sd"), ("within-class sd p90", "within_sd")):
        q = 0.10 if "p10" in lab else 0.90
        v = sh.groupby("sep")[col].quantile(q).reindex(TIERS)
        print(f"  {lab:22s}" + "".join(f"{x:9.3f}" for x in v))
    rho = lambda a, b: spearmanr(sh[a], sh[b], nan_policy="omit").statistic
    print(f"\n  rho(separation, pooled sd)   = {rho('pocket_separation_pair', 'within_naive'):+.3f}"
          f"   <- reads as 'mobility rises with separation'")
    print(f"  rho(separation, within sd)   = {rho('pocket_separation_pair', 'within_sd'):+.3f}"
          f"   <- it does not; the pooled rise IS the between term")
    print(f"  rho(separation, between)     = {rho('pocket_separation_pair', 'between'):+.3f}")
    print(f"  rho(separation, d')          = {rho('pocket_separation_pair', 'dprime'):+.3f}")

    print("\n=== 1.2 the size trap (SUCCESS_VS_DIVERSITY: SD volume is a size proxy) ===")
    for col, lab in (("pocket_separation_pair", "separation"), ("between", "between |delta|"),
                     ("within_sd", "within-class sd"), ("within_naive", "pooled sd (trap)"),
                     ("dprime", "d' = between/within"),
                     ("volume_std_dev" + S, "SD volume (known proxy)"),
                     ("RoG_coefficient_of_variation" + S, "CV RoG (scale-free)"),
                     ("n_struct", "n structures")):
        print(f"  rho(union_rg, {lab:24s}) = {rho('union_rg', col):+.3f}"
              f"     rho(winning_p, .) = {rho('winning_p', col):+.3f}"
              f"     rho(n_struct, .) = {rho('n_struct', col):+.3f}")
    print("    n structures matters because a 'broad' ensemble could just be a well-sampled one;")
    print("    it is also the evidence axis, so it must not be doing the work.")

    # ------------------------------------------- 1.3 split the low-separation stratum
    print("\n=== 1.3 splitting LOW separation on within-class spread ===")
    print("    narrow-coincident = low between, low within   (rigid, ensembles on top of each other)")
    print("    broad-overlapping = low between, high within  (mobile, ensembles sampling one space)")
    for stratum, name in ((["very_low"], "very_low only"),
                          (["very_low", "low"], "very_low + low")):
        s = sh[sh["sep"].isin(stratum)].copy()
        cut = s["within_sd"].median()
        g = Gap(s, "consensus")
        broad = (g.d["within_sd"] > cut).to_numpy()
        cell = np.where(broad, 1, 0)
        pt = g.point(cell, 2)
        bs = g.boot(cell, 2)
        n = Gap.counts(cell, 2)
        print(f"\n  --- {name}  (n = {len(g.d):,}, within-sd median = {cut:.3f} A) ---")
        show("narrow-coincident", pt[0], bs[:, 0], n[0])
        show("broad-overlapping", pt[1], bs[:, 1], n[1])
        show("difference (broad - narrow)", pt[1] - pt[0], bs[:, 1] - bs[:, 0])
        # tertiles for shape, and a size-residualised version of the splitter
        cell3 = qcodes(g.d["within_sd"], 3)
        p3, b3 = g.point(cell3, 3), g.boot(cell3, 3)
        n3 = Gap.counts(cell3, 3)
        print("    within-sd tertiles: " + "  ".join(
            f"T{i+1} {p3[i]:+.1f} (n={n3[i]:,})" for i in range(3)))
        show("    T3 - T1", p3[2] - p3[0], b3[:, 2] - b3[:, 0], width=30)
        rr = g.d["within_sd"].rank(pct=True) - g.d["union_rg"].rank(pct=True)
        cellr = np.where(rr > rr.median(), 1, 0)
        pr, br = g.point(cellr, 2), g.boot(cellr, 2)
        show("    size-residualised split", pr[1] - pr[0], br[:, 1] - br[:, 0], width=30)

    # -------------------------------------------------- 1.4 does d' order the gap better?
    print("\n=== 1.4 does the normalized effect size order the gap better than overlap-separation? ===")
    g = Gap(sh, "consensus")
    print(f"  {'quintiles of':26s}" + "".join(f"{i:>9s}" for i in
                                              ["Q1", "Q2", "Q3", "Q4", "Q5"]) + "     Q5 - Q1")
    spans = {}
    for col, lab in (("pocket_separation_pair", "separation (Fig 4d axis)"),
                     ("dprime", "d' = between/within"),
                     ("between", "between |delta| alone"),
                     ("within_sd", "within-class sd alone"),
                     ("dprime_naive", "|delta| / pooled sd (trap)"),
                     ("winning_p", "winning_p (reference)")):
        cell = qcodes(g.d[col], 5)
        pt, bs = g.point(cell, 5), g.boot(cell, 5)
        lo, hi = ci(bs[:, 4] - bs[:, 0])
        spans[col] = bs[:, 4] - bs[:, 0]
        print(f"  {lab:26s}" + "".join(f"{v:9.1f}" for v in pt)
              + f"   {pt[4]-pt[0]:+6.1f} [{lo:+.1f}, {hi:+.1f}]")
    dif = spans["dprime"] - spans["pocket_separation_pair"]
    lo, hi = ci(dif)
    print(f"\n  span(d') - span(separation) = {np.mean(dif):+.1f} pp  [{lo:+.1f}, {hi:+.1f}]"
          f"   (bootstrap mean of the paired difference)")
    print(f"  share of resamples where d' spans wider: {100*np.mean(dif > 0):.1f}%")

    print("\n  cross-tabulated -- tertiles of separation (rows) x tertiles of d' (cols):")
    ts = qcodes(g.d["pocket_separation_pair"], 3)
    td = qcodes(g.d["dprime"], 3)
    cell = np.where((ts >= 0) & (td >= 0), ts * 3 + td, -1)
    pt, n = g.point(cell, 9), Gap.counts(cell, 9)
    print(f"    {'sep tert':10s}" + "".join(f"{c:>9s}" for c in ["d' T1", "d' T2", "d' T3"]))
    for i in range(3):
        print(f"    {'T'+str(i+1):10s}" + "".join(f"{pt[i*3+j]:9.1f}" for j in range(3))
              + "    n = " + " ".join(f"{n[i*3+j]:,}" for j in range(3)))

    # --------------------------------------------- 1.5 conformational-selection probe
    print("\n=== 1.5 direct conformational-selection probe (radial reach) ===")
    print("    anchor is a holo structure, so 'apo reaches into holo territory' means the CLOSEST")
    print("    apo structure sits inside the shell the holo ensemble occupies:")
    print("      reach = pocket_rms_holo_max - pocket_rms_apo_min   (> 0 = apo reaches in)")
    r = sh["reach"]
    print(f"    reach > 0 in {100*(r > 0).mean():.1f}% of pockets; median {r.median():+.3f} A")
    print("    by separation tier, median reach: "
          + "  ".join(f"{t}={v:+.2f}" for t, v in sh.groupby('sep')['reach'].median()
                      .reindex(TIERS).items()))
    for col, lab in (("reach", "reach (A)"), ("reach_z", "reach / within sd")):
        cell = qcodes(g.d[col], 5)
        pt, bs = g.point(cell, 5), g.boot(cell, 5)
        lo, hi = ci(bs[:, 4] - bs[:, 0])
        print(f"    gap by {lab:18s} quintile: " + "  ".join(f"{v:+6.1f}" for v in pt)
              + f"    Q5-Q1 {pt[4]-pt[0]:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    print(f"    rho(reach, separation) = "
          f"{spearmanr(sh.reach, sh.pocket_separation_pair, nan_policy='omit').statistic:+.3f}"
          f"   rho(reach, d') = "
          f"{spearmanr(sh.reach, sh.dprime, nan_policy='omit').statistic:+.3f}"
          f"   rho(reach, union_rg) = "
          f"{spearmanr(sh.reach, sh.union_rg, nan_policy='omit').statistic:+.3f}")

    print("\n=== 1.5b between x within, 3 x 3 -- which term carries the gap? ===")
    print(f"    the two terms are NOT independent: rho(between, within sd) = "
          f"{spearmanr(sh.between, sh.within_sd, nan_policy='omit').statistic:+.3f}, so a marginal")
    print("    'within' effect can be borrowed from 'between'. Cross-classifying is the only way")
    print("    to see which one is doing the work.")
    tb = qcodes(g.d["between"], 3)
    tw = qcodes(g.d["within_sd"], 3)
    cell = np.where((tb >= 0) & (tw >= 0), tb * 3 + tw, -1)
    pt, n9 = g.point(cell, 9), Gap.counts(cell, 9)
    bs9 = g.boot(cell, 9)
    print(f"    {'between':12s}" + "".join(f"{c:>11s}" for c in
                                           ["within T1", "within T2", "within T3"])
          + "     T3 - T1 (within)")
    for i in range(3):
        lo, hi = ci(bs9[:, i * 3 + 2] - bs9[:, i * 3])
        print(f"    {'T'+str(i+1):12s}" + "".join(f"{pt[i*3+j]:11.1f}" for j in range(3))
              + f"   {pt[i*3+2]-pt[i*3]:+6.1f} [{lo:+.1f}, {hi:+.1f}]"
              + "    n = " + " ".join(f"{n9[i*3+j]:,}" for j in range(3)))
    print(f"    {'T3-T1 (between)':12s}" + "".join(
        f"{pt[6+j]-pt[j]:11.1f}" for j in range(3)))

    # ------------------------------------------ 1.6 within LOW separation only, by d' pieces
    print("\n=== 1.6 inside the low-separation stratum, which piece moves the gap? ===")
    low = sh[sh["sep"].isin(["very_low", "low"])]
    gl = Gap(low, "consensus")
    for col, lab in (("within_sd", "within-class sd"), ("between", "between |delta|"),
                     ("dprime", "d'"), ("reach_z", "reach / within sd"),
                     ("winning_p", "winning_p")):
        cell = qcodes(gl.d[col], 5)
        pt, bs = gl.point(cell, 5), gl.boot(cell, 5)
        lo, hi = ci(bs[:, 4] - bs[:, 0])
        print(f"  {lab:20s}" + "".join(f"{v:8.1f}" for v in pt)
              + f"    Q5-Q1 {pt[4]-pt[0]:+6.1f} [{lo:+.1f}, {hi:+.1f}]")
    return sh


# ============================================================================ Q2
def did_from(bs5x5, wts=None):
    """Contrasts read off a (B, 25) matrix of cell gaps indexed sep*5 + prob."""
    g = bs5x5.reshape(-1, 5, 5)
    corner = (g[:, 4, 4] - g[:, 0, 4]) - (g[:, 4, 0] - g[:, 0, 0])
    # linear-trend interaction: weighted LS of gap on (sep index, prob index, product)
    si, pi = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    X = np.column_stack([np.ones(25), si.ravel(), pi.ravel(), (si * pi).ravel()])
    w = np.ones(25) if wts is None else np.asarray(wts, float)
    y = bs5x5
    ok = np.isfinite(y).all(0)
    Xw = X[ok] * np.sqrt(w[ok])[:, None]
    beta = np.linalg.lstsq(Xw, (y[:, ok] * np.sqrt(w[ok])[None, :]).T, rcond=None)[0]
    return corner, beta[3] * 16.0     # scaled to span the full 4-step x 4-step grid


def ols_boot(d, ycol, xcol, label, n_boot=B, seed=SEED):
    """Continuous three-way test.

    y ~ decile FE (fully interacted with class, so the class contrast is size-standardized within
    decile) + H*(1 + x + c + x*c) + x + c + x*c,  with x and c rank-uniform on [0, 1].
    The coefficient on H*x*c is the difference-in-differences in continuous form: extra pp of
    class gap when BOTH x and c move across their full range.
    """
    dd = d[d[ycol].notna() & d["union_rg"].notna() & d[xcol].notna() & d["winning_p"].notna()].copy()
    zb = pd.qcut(pd.to_numeric(dd["union_rg"], errors="coerce"), K, labels=False, duplicates="drop")
    dd = dd[zb.notna()].copy()
    zb = zb[zb.notna()].astype(int).to_numpy()
    y = dd[ycol].to_numpy(float)
    h = (dd["cls"].to_numpy() == "holo").astype(float)
    x = rank01(dd[xcol]) - 0.5
    c = rank01(dd["winning_p"]) - 0.5
    Dm = np.zeros((len(dd), K))
    Dm[np.arange(len(dd)), zb] = 1.0
    X = np.column_stack([Dm, Dm * h[:, None], x, c, x * c,
                         h * x, h * c, h * x * c])
    names = ["H*x", "H*c", "H*x*c"]
    codes = pd.factorize(dd["site_uniprot"])[0]

    wdec = np.bincount(zb, minlength=K).astype(float)
    wdec /= wdec.sum()

    def fit(ix=None, full=False):
        A, b = (X, y) if ix is None else (X[ix], y[ix])
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
        return beta * 100.0 if full else beta[-3:] * 100.0

    pt = fit()
    beta_full = fit(full=True)
    centre = float(beta_full[K:2 * K] @ wdec)
    rng = np.random.default_rng(seed)
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    nC = sc.max() + 1
    st = np.searchsorted(sc, np.arange(nC), "left")
    ln = np.searchsorted(sc, np.arange(nC), "right") - st
    out = np.empty((n_boot, 3))
    for b_ in range(n_boot):
        pick = rng.integers(0, nC, nC)
        lp, sp = ln[pick], st[pick]
        off = np.zeros(len(lp), dtype=np.int64)
        np.cumsum(lp[:-1], out=off[1:])
        ix = order[np.arange(int(lp.sum())) - np.repeat(off, lp) + np.repeat(sp, lp)]
        out[b_] = fit(ix)
    print(f"  {label}   n = {len(dd):,}   size-standardized gap at median x, median c: "
          f"{centre:+.1f} pp")
    for j, nm in enumerate(names):
        lo, hi = ci(out[:, j])
        star = "  *" if (lo > 0 or hi < 0) else ""
        print(f"    {nm:8s} {pt[j]:+7.2f} pp  [{lo:+7.2f}, {hi:+7.2f}]{star}")
    return pt, out


def q2(d, sh):
    print("\n" + "=" * 92)
    print("QUESTION 2 -- does separation ENHANCE a confident label? (an interaction, not a "
          "main effect)")
    print("=" * 92)

    g = Gap(d, "consensus")
    cs = codes_from(g.d["sep"], TIERS)
    cp = codes_from(g.d["prob"], TIERS)
    cell = np.where((cs >= 0) & (cp >= 0), cs * 5 + cp, -1)
    pt, bs = g.point(cell, 25), g.boot(cell, 25)
    n = Gap.counts(cell, 25)

    print("\n=== 2.1 the 5x5 grid it all comes from (rows = separation, cols = confidence) ===")
    print(f"  {'':12s}" + "".join(f"{s:>9s}" for s in SHORT) + "      sep effect (v.high - v.low)")
    for i in range(5):
        row = "".join(f"{pt[i*5+j]:9.1f}" for j in range(5))
        print(f"  {TIERS[i]:12s}{row}")
    print(f"  {'n min/max':12s}{n.min():9,}{n.max():>9,}")
    print("\n  separation effect (v.high - v.low sep) WITHIN each confidence tier:")
    for j in range(5):
        dif = pt[4 * 5 + j] - pt[0 * 5 + j]
        lo, hi = ci(bs[:, 4 * 5 + j] - bs[:, 0 * 5 + j])
        print(f"    confidence {TIERS[j]:11s} {dif:+6.1f} pp  [{lo:+6.1f}, {hi:+6.1f}]"
              f"   n = {n[4*5+j]:,} / {n[0*5+j]:,}")

    print("\n=== 2.2 formal difference-in-differences, one CI on the DiD itself ===")
    corner, trend = did_from(bs, wts=n)
    corner_pt = (pt[24] - pt[4]) - (pt[20] - pt[0])
    trend_pt = did_from(pt[None, :], wts=n)[1][0]
    lo, hi = ci(corner)
    print(f"  corner DiD  (sep v.high-v.low) x (conf v.high-v.low)   {corner_pt:+6.1f} pp "
          f"[{lo:+6.1f}, {hi:+6.1f}]")
    lo, hi = ci(trend)
    print(f"  linear-trend DiD across all 25 cells, n-weighted        {trend_pt:+6.1f} pp "
          f"[{lo:+6.1f}, {hi:+6.1f}]")
    print(f"  share of resamples with a positive trend DiD: {100*np.mean(trend > 0):.1f}%")

    print("\n=== 2.3 continuous three-way, size-adjusted (no tiering, no thin cells) ===")
    ols_boot(d, "consensus", "pocket_separation_pair", "separation x winning_p x class")
    ols_boot(sh, "consensus", "dprime", "d'        x winning_p x class  (shape subset)")
    ols_boot(sh, "consensus", "pocket_separation_pair",
             "separation x winning_p x class  (same subset, for comparison)")
    ols_boot(sh, "consensus", "between",
             "between |delta| alone  x winning_p x class  (is normalizing what helps?)")
    ols_boot(sh, "consensus", "within_sd",
             "within-class sd alone  x winning_p x class  (placebo)")
    print("    NB the placebo does NOT come back null, because rho(between, within) = +0.51 and")
    print("    this model holds neither fixed. Section 1.5b, which cross-classifies them, is the")
    print("    test that discriminates; these rows only establish that all four orderings carry")
    print("    the same interaction, not that each is a separate effect.")

    print("\n=== 2.4 the author's expectation, stated as a test ===")
    print("    'among CONFIDENT pockets, the gap at low separation should be below the gap at high'")
    top = d[d["prob"] == "very_high"]
    gt = Gap(top, "consensus")
    ct = codes_from(gt.d["sep"], TIERS)
    ptt, bst = gt.point(ct, 5), gt.boot(ct, 5)
    nt = Gap.counts(ct, 5)
    for i, t in enumerate(TIERS):
        show(f"conf v.high, sep {t}", ptt[i], bst[:, i], nt[i])
    lo, hi = ci(bst[:, 4] - bst[:, 0])
    print(f"  {'v.high - v.low separation':<34s} {ptt[4]-ptt[0]:+6.1f} pp  [{lo:+6.1f}, {hi:+6.1f}]"
          f"   ({100*np.mean(bst[:, 4] - bst[:, 0] > 0):.0f}% of resamples positive)")
    # and against the marginal, which is what "reliably below" would have to beat
    gm = Gap(d, "consensus")
    cm = codes_from(gm.d["prob"], TIERS)
    pm = gm.point(cm, 5)
    print(f"  marginal gap in the top confidence tier (all separations): {pm[4]:+.1f} pp")

    print("\n=== 2.5 per predictor -- is any interaction a consensus artefact? ===")
    dk = load_docking(d)
    print(f"  docking set: {len(dk):,} pockets, {dk.site_uniprot.nunique():,} clusters")
    print(f"  {'task':14s}{'corner DiD':>26s}{'trend DiD':>26s}")
    for task, frame in [(t, d) for t in PRED + ["consensus"]] + [("docking", dk)]:
        gg = Gap(frame, task)
        a = codes_from(gg.d["sep"], TIERS)
        b_ = codes_from(gg.d["prob"], TIERS)
        cl = np.where((a >= 0) & (b_ >= 0), a * 5 + b_, -1)
        p_, bo = gg.point(cl, 25), gg.boot(cl, 25)
        nn = Gap.counts(cl, 25)
        cor, tre = did_from(bo, wts=nn)
        cor_pt = (p_[24] - p_[4]) - (p_[20] - p_[0])
        tre_pt = did_from(p_[None, :], wts=nn)[1][0]
        l1, h1 = ci(cor)
        l2, h2 = ci(tre)
        print(f"  {task:14s}{cor_pt:+8.1f} [{l1:+6.1f},{h1:+6.1f}]"
              f"{tre_pt:+8.1f} [{l2:+6.1f},{h2:+6.1f}]")
    print("\n  the same question with the POWERED test -- continuous three-way, per predictor")
    print("  (the 5x5 DiD above spends most of its n on two cells of 114 and 303):")
    for task, frame in [(t, d) for t in PRED + ["consensus"]] + [("docking", dk)]:
        ols_boot(frame, task, "pocket_separation_pair", f"{task:12s} sep x winning_p x class")

    print("\n  and the raw per-predictor separation gradient inside the top confidence tier:")
    print(f"  {'':14s}" + "".join(f"{s:>9s}" for s in SHORT))
    for task in PRED + ["consensus"]:
        gg = Gap(top, task)
        cl = codes_from(gg.d["sep"], TIERS)
        p_ = gg.point(cl, 5)
        print(f"  {task:14s}" + "".join(f"{v:9.1f}" for v in p_))

    print("\n=== 2.5b is the interaction a ceiling effect? the levels behind the four corners ===")
    print("    (a gap cannot widen where the holo rate is already at 100, so the LEVELS decide")
    print("     whether a difference-in-differences is a real modulation or an arithmetic one)")
    print(f"  {'task':12s}{'corner':28s}{'apo':>8s}{'holo':>8s}{'gap':>8s}")
    for task in ["consensus", "fpocket", "IF-SitePred", "P2Rank"]:
        gg = Gap(d, task)
        a = codes_from(gg.d["sep"], TIERS)
        b_ = codes_from(gg.d["prob"], TIERS)
        corners = {(0, 0): "sep v.low  x conf v.low", (4, 0): "sep v.high x conf v.low",
                   (0, 4): "sep v.low  x conf v.high", (4, 4): "sep v.high x conf v.high"}
        cl = np.full(len(gg.d), -1)
        for i, (k, _) in enumerate(corners.items()):
            cl[(a == k[0]) & (b_ == k[1])] = i
        rt = gg.rates(cl, 4)
        for i, lab in enumerate(corners.values()):
            print(f"  {task if i == 0 else '':12s}{lab:28s}{rt[i,0]:8.1f}{rt[i,1]:8.1f}"
                  f"{rt[i,1]-rt[i,0]:+8.1f}")

    print("\n  class mix inside the 5x5 (fraction holo) -- conditioning stays non-degenerate:")
    mix = pd.crosstab(d["sep"], d["prob"], values=(d["cls"] == "holo"),
                      aggfunc="mean").reindex(index=TIERS, columns=TIERS)
    print("    " + mix.round(2).to_string().replace("\n", "\n    "))

    print("\n=== 2.6 mediation -- how much of separation's gradient does confidence account for? ===")
    print("    total  = marginal gap(sep v.high) - gap(sep v.low)")
    print("    direct = the same contrast averaged over confidence tiers (n-weighted)")
    cs_ = codes_from(gm.d["sep"], TIERS)
    pm_s, bm_s = gm.point(cs_, 5), gm.boot(cs_, 5)
    total_pt = pm_s[4] - pm_s[0]
    total_bs = bm_s[:, 4] - bm_s[:, 0]
    wts = np.array([n[4 * 5 + j] + n[0 * 5 + j] for j in range(5)], float)
    wts /= wts.sum()
    direct_pt = float(np.nansum([wts[j] * (pt[4 * 5 + j] - pt[0 * 5 + j]) for j in range(5)]))
    direct_bs = np.nansum(np.stack([wts[j] * (bs[:, 4 * 5 + j] - bs[:, 0 * 5 + j])
                                    for j in range(5)], 1), 1)
    show("total (marginal) separation effect", total_pt, total_bs)
    show("direct (confidence held fixed)", direct_pt, direct_bs)
    med = 100.0 * (1.0 - direct_bs / total_bs)
    lo, hi = ci(med)
    print(f"  {'proportion mediated':<34s} {100*(1-direct_pt/total_pt):+6.1f} %   "
          f"[{lo:+6.1f}, {hi:+6.1f}]")

    print("\n  same decomposition with d' in place of overlap-separation (shape subset):")
    gs = Gap(sh, "consensus")
    cd = qcodes(gs.d["dprime"], 5)
    cq = codes_from(gs.d["prob"], TIERS)
    pd_, bd_ = gs.point(cd, 5), gs.boot(cd, 5)
    cell2 = np.where((cd >= 0) & (cq >= 0), cd * 5 + cq, -1)
    p2, b2 = gs.point(cell2, 25), gs.boot(cell2, 25)
    n2 = Gap.counts(cell2, 25)
    tot_pt, tot_bs = pd_[4] - pd_[0], bd_[:, 4] - bd_[:, 0]
    w2 = np.array([n2[4 * 5 + j] + n2[0 * 5 + j] for j in range(5)], float)
    w2 /= w2.sum()
    dir_pt = float(np.nansum([w2[j] * (p2[4 * 5 + j] - p2[0 * 5 + j]) for j in range(5)]))
    dir_bs = np.nansum(np.stack([w2[j] * (b2[:, 4 * 5 + j] - b2[:, 0 * 5 + j])
                                 for j in range(5)], 1), 1)
    show("total (marginal) d' effect", tot_pt, tot_bs)
    show("direct (confidence held fixed)", dir_pt, dir_bs)
    med2 = 100.0 * (1.0 - dir_bs / tot_bs)
    lo, hi = ci(med2)
    print(f"  {'proportion mediated':<34s} {100*(1-dir_pt/tot_pt):+6.1f} %   "
          f"[{lo:+6.1f}, {hi:+6.1f}]")
    print("\n  gap by d' quintile WITHIN each confidence tier (rows = confidence):")
    print(f"  {'':12s}" + "".join(f"{'Q'+str(i+1):>9s}" for i in range(5)))
    for j, t in enumerate(TIERS):
        row = [p2[i * 5 + j] for i in range(5)]
        print(f"  {t:12s}" + "".join(f"{v:9.1f}" for v in row))


# ============================================================================ Q3
def q3(d):
    """Which ARM of the contrast moves along the separation gradient?

    The 'cleaner signal' reading of Figure 4d -- that discrete, non-overlapping ensembles sharpen
    both classes, pushing apo down and holo up -- makes a specific, checkable prediction: a
    SYMMETRIC fan-out. Both arms must move, in opposite directions. One arm moving is a different
    story and needs different words.

    Reported standardized AND raw, because the raw apo trend across separation bins is the known
    pocket-size artefact (the raw AF-Apo gain across the tiers is pocket size and is abolished by
    standardization) and the contrast between the two rows is the cleanest way to show that it has
    been handled.
    """
    print("\n" + "=" * 92)
    print("QUESTION 3 -- which arm moves? the fan-out test")
    print("=" * 92)
    print("    prediction of the 'cleaner signal' reading: apo DOWN and holo UP, symmetrically.")
    dk = load_docking(d)
    for task, frame in [(t, d) for t in ["consensus"] + PRED] + [("docking", dk)]:
        g = Gap(frame, task)
        cell = codes_from(g.d["sep"], TIERS)
        rt = g.rates(cell, 5)
        rw = g.rates(cell, 5, raw=True)
        br = g.boot_rates(cell, 5)
        gp = g.point(cell, 5)
        bg = g.boot(cell, 5)
        n = Gap.counts(cell, 5)
        print(f"\n  --- {task}  (n = {len(g.d):,}) ---")
        print(f"  {'':22s}" + "".join(f"{s:>8s}" for s in SHORT) + "     v.high - v.low")
        for j, lab in ((0, "AF-apo, standardized"), (1, "AF-holo, standardized")):
            lo, hi = ci(br[:, 4, j] - br[:, 0, j])
            print(f"  {lab:22s}" + "".join(f"{rt[i, j]:8.1f}" for i in range(5))
                  + f"   {rt[4, j]-rt[0, j]:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")
        lo, hi = ci(bg[:, 4] - bg[:, 0])
        print(f"  {'gap':22s}" + "".join(f"{gp[i]:8.1f}" for i in range(5))
              + f"   {gp[4]-gp[0]:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")
        brw = g.boot_rates(cell, 5, raw=True)
        for j, lab in ((0, "AF-apo, RAW"), (1, "AF-holo, RAW")):
            lo, hi = ci(brw[:, 4, j] - brw[:, 0, j])
            print(f"  {lab:22s}" + "".join(f"{rw[i, j]:8.1f}" for i in range(5))
                  + f"   {rw[4, j]-rw[0, j]:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")
        gr = rw[:, 1] - rw[:, 0]
        bgr = brw[:, :, 1] - brw[:, :, 0]
        lo, hi = ci(bgr[:, 4] - bgr[:, 0])
        print(f"  {'gap, RAW':22s}" + "".join(f"{gr[i]:8.1f}" for i in range(5))
              + f"   {gr[4]-gr[0]:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")
        print(f"  {'n':22s}" + "".join(f"{v:8,}" for v in n))


# ============================================================================ Q4
def q4(d):
    """How much of each tool's class gap is NOT made by the conformational difference?

    The lowest separation tier is the natural place to ask. There the apo and holo ensembles
    largely coincide, so whatever gap remains cannot be attributed to the two states being
    conformationally different -- it is whatever else the AF class label is tracking for that tool.

    Motivating claim (author, verified externally): P2Rank was trained exclusively on holo
    structures, and IF-SitePred's training reflects a holo-skewed PDB. If a learned holo prior were
    what produces a residual gap at low separation, the residual should order with how holo-trained
    each tool is. Docking is the control that decides it: Vina plus a pose/affinity cut has no
    learned class prior at all.

    All tools are evaluated on ONE row set (load_lbs already yields complete cases across the
    three predictors), so the cluster codes -- and therefore the bootstrap resamples -- are shared
    and a tool-vs-tool difference gets its own interval.
    """
    print("\n" + "=" * 92)
    print("QUESTION 4 -- how much of each tool's gap survives where the states coincide?")
    print("=" * 92)
    assert d[PRED].notna().all(axis=1).all(), "tools are not on a common row set"
    dk = load_docking(d)
    res = {}
    for tool in PRED + ["consensus"]:
        g = Gap(d, tool)
        cell = codes_from(g.d["sep"], TIERS)
        res[tool] = (g.point(cell, 5), g.boot(cell, 5))
    gd = Gap(dk, "docking")
    cd = codes_from(gd.d["sep"], TIERS)
    res["docking"] = (gd.point(cd, 5), gd.boot(cd, 5))

    print("\n  gap in the LOWEST separation tier -- the part not attributable to conformation:")
    for t, (p, b) in res.items():
        lo, hi = ci(b[:, 0])
        frac = 100.0 * p[0] / p[4] if p[4] else np.nan
        print(f"  {t:12s} v.low {p[0]:+6.1f} pp [{lo:+6.1f}, {hi:+6.1f}]"
              f"   v.high {p[4]:+6.1f}   already present at v.low: {frac:5.1f}%")

    print("\n  tool vs tool at v.low separation (shared resamples, identical pockets):")
    for a, b_ in (("IF-SitePred", "fpocket"), ("IF-SitePred", "P2Rank"), ("P2Rank", "fpocket")):
        dif = res[a][1][:, 0] - res[b_][1][:, 0]
        lo, hi = ci(dif)
        star = "  *" if (lo > 0 or hi < 0) else ""
        print(f"    {a:12s} - {b_:12s} {res[a][0][0]-res[b_][0][0]:+6.1f} pp "
              f"[{lo:+6.1f}, {hi:+6.1f}]{star}")

    print("\n  full series, for reference:")
    print(f"  {'':12s}" + "".join(f"{s:>8s}" for s in SHORT) + "        span")
    for t, (p, b) in res.items():
        lo, hi = ci(b[:, 4] - b[:, 0])
        print(f"  {t:12s}" + "".join(f"{v:8.1f}" for v in p)
              + f"   {p[4]-p[0]:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")
    print("\n  READ: only fpocket's gap vanishes where the ensembles coincide, and only the")
    print("  extreme pair (IF-SitePred vs fpocket) separates. Docking carries the LARGEST")
    print("  residual with no learned class prior whatever, so the residual in the trained")
    print("  tools cannot be attributed to training bias on this evidence.")


def main():
    d = add_shape(load_lbs())
    print(f"\nLBS ST_100 pockets: {len(d):,}   UniProt clusters: {d.site_uniprot.nunique():,}")
    sh = q1(d)
    q2(d, sh)
    q3(d)
    q4(d)


if __name__ == "__main__":
    main()
