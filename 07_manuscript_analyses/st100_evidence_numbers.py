# -*- coding: utf-8 -*-
"""ST_100 evidence, class and detection-set numbers quoted in the Results, Methods and Figure 2 legend:
whether the evidence RATIO or the evidence COUNTS set where a pocket falls (item 1), class counts and
evidence asymmetry (item 4), n_apo / n_holo per class (item 7, what Figure 2c draws), the filter chain
from the 105,713 ST_100 sites to the 86,946 scored detection pockets (item 8), and separation tiers by
class (item 9). balanced_classification.py reuses load_st100 and partial_spearman.

    python st100_evidence_numbers.py all            # everything, in item order
    python st100_evidence_numbers.py item1|item4|item7|item8|item9

Data: ST_100 rows of analysis_unified_levels.csv (UBS level; evidence = member-average n_apo /
n_holo, the basis Figure 2 plots), ties excluded unless stated. Item 8 also reads the detection-set
tables (afdb_st_100_dataset*.csv, the three LBS per-pocket tables).

Item 1 -- does the evidence RATIO set placement, or do the COUNTS?
    ratio  = log10((n_holo + 1) / n_apo)                (the Figure 2d/e axis)
    counts = log10 n_apo, log10 (n_holo + 1), log10 min(n_apo, n_holo + 1), log10 (n_apo + n_holo + 1)
  (a) Spearman rho of each with P(holo);  (b) partial Spearman of each count given the ratio, and of
  the ratio given both counts;  (c) AF-Holo prevalence in a ratio-bin x total-count-quartile grid,
  with the spread across count quartiles at fixed ratio against the spread across ratio bins at fixed
  count;  (d) logistic AUC for class: ratio alone, counts alone, ratio + total.
Item 7 -- medians / IQR of n_apo and n_holo per class (what Figure 2c draws).
Item 9 -- separation tiers (1 - pocket_overlap_pair; Figure 4e edges 0.30 / 0.60) by class.
Item 4 -- class counts and evidence asymmetry at ST_100 (the ensemble spread is in
           ensemble_containment.py summarise).
Item 8 -- the filter chain from 105,713 ST_100 sites to the 86,946 scored detection pockets.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import ma_paths as P  # noqa: E402

WD = str(P.WD)
UNI = os.path.join(WD, r"unified_output\run5\analysis_unified_levels.csv")
DS = os.path.join(WD, r"afdb_dataset\st100_run3\afdb_st_100_dataset.csv")
DSG = os.path.join(WD, r"afdb_dataset\st100_run3\afdb_st_100_dataset_with_geometry.csv")
LBS = os.path.join(WD, "lbs_generality")
LBS_TABLES = {
    "p2rank": os.path.join(LBS, r"p2rank\lbs_p2rank_st100_merged_per_pocket.csv"),
    "fpocket": os.path.join(LBS, r"fpocket\full\lbs_fpocket_st100_merged_per_pocket.csv"),
    "ifsitepred": os.path.join(LBS, r"ifsitepred\full\lbs_ifsitepred_st100_merged_per_pocket.csv"),
}

SEP_EDGES = (0.30, 0.60)          # figure3C_bars_REV18: DENSSEP_LOW_MAX / DENSSEP_MOD_MAX
PROB_EDGES = (0.5933, 0.6996)     # figure3C_bars_REV18: PROB_LOW_MAX / PROB_MOD_MAX
TIERS = ["low", "moderate", "high"]

pd.set_option("display.width", 250, "display.max_columns", 30)


def load_st100(classified=True):
    cols = ["threshold", "unit_uid", "uniprots_key", "n_uniprots", "n_chains", "is_interface", "members",
            "pocket_class", "pocket_p_apo", "pocket_p_holo", "evidence_n_apo", "evidence_n_holo",
            "pocket_density_pair", "pocket_overlap_pair", "site_size_unit"]
    d = pd.read_csv(UNI, usecols=cols, low_memory=False)
    d = d[d.threshold == "ST_100"].copy()
    if classified:
        d = d[d.pocket_class.isin(["apo", "holo"])].copy()
        d["holo"] = (d.pocket_class == "holo").astype(int)
        d["win"] = d[["pocket_p_apo", "pocket_p_holo"]].max(axis=1)
        d["nh1"] = d.evidence_n_holo + 1.0
        d["ratio"] = np.log10(d.nh1 / d.evidence_n_apo)
        d["l_apo"] = np.log10(d.evidence_n_apo)
        d["l_nh1"] = np.log10(d.nh1)
        d["l_min"] = np.log10(np.minimum(d.evidence_n_apo, d.nh1))
        d["l_tot"] = np.log10(d.evidence_n_apo + d.nh1)
    return d


def _rank(x):
    return pd.Series(x).rank().to_numpy(float)


def partial_spearman(y, x, controls):
    """Partial rank correlation: residualise the ranks of y and x on the ranks of the controls."""
    ry, rx = _rank(y), _rank(x)
    C = np.column_stack([np.ones(len(ry))] + [_rank(c) for c in controls])
    beta_y, *_ = np.linalg.lstsq(C, ry, rcond=None)
    beta_x, *_ = np.linalg.lstsq(C, rx, rcond=None)
    ey, ex = ry - C @ beta_y, rx - C @ beta_x
    return float(np.corrcoef(ey, ex)[0, 1])


def item1():
    d = load_st100()
    print("=" * 100)
    print(f"ITEM 1 -- ratio vs counts as what sets placement. ST_100 classified sites n = {len(d):,}")
    print("\n(a) Spearman rho with P(holo) [and with the class indicator]:")
    preds = [("ratio log10((n_holo+1)/n_apo)", "ratio"), ("log10 n_apo", "l_apo"), ("log10 (n_holo+1)", "l_nh1"),
             ("log10 min(n_apo, n_holo+1)", "l_min"), ("log10 (n_apo + n_holo + 1)", "l_tot")]
    for lab, c in preds:
        r1 = spearmanr(d[c], d.pocket_p_holo).correlation
        r2 = spearmanr(d[c], d.holo).correlation
        print(f"    {lab:32s} rho_P(holo) = {r1:+.3f}   rho_class = {r2:+.3f}")

    print("\n(b) partial Spearman with P(holo):")
    for lab, c in preds[1:]:
        pr = partial_spearman(d.pocket_p_holo, d[c], [d.ratio])
        print(f"    {lab:32s} | ratio            = {pr:+.3f}")
    pr = partial_spearman(d.pocket_p_holo, d.ratio, [d.l_apo, d.l_nh1])
    print(f"    {'ratio':32s} | log n_apo, log (n_holo+1) = {pr:+.3f}")
    pr = partial_spearman(d.pocket_p_holo, d.ratio, [d.l_tot])
    print(f"    {'ratio':32s} | log total        = {pr:+.3f}")
    pr = partial_spearman(d.pocket_p_holo, d.ratio, [d.l_min])
    print(f"    {'ratio':32s} | log min          = {pr:+.3f}")

    print("\n(c) AF-Holo prevalence (%) in ratio bins x quartiles of total evidence (n_apo + n_holo + 1):")
    edges = [-np.inf, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, np.inf]
    labels = ["<-1", "-1..-0.5", "-0.5..-0.25", "-0.25..0", "0..0.25", "0.25..0.5", "0.5..1", "1..1.5", ">1.5"]
    d["rbin"] = pd.cut(d.ratio, edges, labels=labels)
    d["tq"] = pd.qcut(d.l_tot, 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])
    g = d.pivot_table(index="rbin", columns="tq", values="holo", aggfunc="mean", observed=False) * 100
    n = d.pivot_table(index="rbin", columns="tq", values="holo", aggfunc="size", observed=False)
    g["all"] = d.groupby("rbin", observed=False).holo.mean() * 100
    g["n"] = d.groupby("rbin", observed=False).size()
    print(g.round(1).to_string())
    print("    cell n:")
    print(n.to_string())
    within = (g.iloc[:, :4].max(axis=1) - g.iloc[:, :4].min(axis=1))
    across = (g.iloc[:, :4].max(axis=0) - g.iloc[:, :4].min(axis=0))
    print(f"    spread across COUNT quartiles at fixed ratio bin: median {within.median():.1f} pp, max {within.max():.1f} pp")
    print(f"    spread across RATIO bins at fixed count quartile: median {across.median():.1f} pp, min {across.min():.1f} pp")
    # monotone direction of the count effect within ratio bins
    slopes = []
    for b, sub in d.groupby("rbin", observed=False):
        if len(sub) > 200:
            slopes.append((b, spearmanr(sub.l_tot, sub.pocket_p_holo).correlation, len(sub)))
    print("    Spearman(total evidence, P(holo)) inside each ratio bin: " +
          ", ".join(f"{b} {r:+.2f} (n={k:,})" for b, r, k in slopes))
    # the same grid at fixed MIN (shared evidence) instead of total
    d["mq"] = pd.qcut(d.l_min.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    g2 = d.pivot_table(index="rbin", columns="mq", values="holo", aggfunc="mean", observed=False) * 100
    within2 = (g2.max(axis=1) - g2.min(axis=1))
    print(f"    same grid on quartiles of min(n_apo, n_holo+1): spread across count quartiles at fixed ratio "
          f"median {within2.median():.1f} pp, max {within2.max():.1f} pp")

    print("\n(d) logistic regression for the class (5-fold CV AUC; standardised inputs):")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        y = d.holo.to_numpy()
        for lab, cols in [("ratio", ["ratio"]), ("log n_apo + log (n_holo+1)", ["l_apo", "l_nh1"]),
                          ("log total + log min", ["l_tot", "l_min"]), ("ratio + log total", ["ratio", "l_tot"]),
                          ("ratio + log min", ["ratio", "l_min"])]:
            X = d[cols].to_numpy()
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
            auc = cross_val_score(m, X, y, cv=5, scoring="roc_auc")
            m.fit(X, y)
            coefs = ", ".join(f"{c} {b:+.2f}" for c, b in zip(cols, m[-1].coef_[0]))
            print(f"    {lab:28s} AUC = {auc.mean():.3f}   std. coef: {coefs}")
    except ImportError:
        print("    sklearn not available")


def item7():
    d = load_st100()
    print("=" * 100)
    print(f"ITEM 7 -- evidence per class (Figure 2c; member-average basis; ties excluded). n = {len(d):,}")
    for cl in ["apo", "holo"]:
        x = d[d.pocket_class == cl]
        for lab, v in [("n_apo", x.evidence_n_apo), ("n_holo", x.evidence_n_holo), ("n_holo + 1", x.nh1)]:
            q = v.quantile([.25, .5, .75]).tolist()
            print(f"    AF-{cl:4s} {lab:10s} n = {len(x):,}  median {q[1]:.1f}  IQR [{q[0]:.1f}, {q[2]:.1f}]  mean {v.mean():.1f}")
        print(f"    AF-{cl:4s} share with n_apo > n_holo + 1: {100 * (x.evidence_n_apo > x.nh1).mean():.1f}%; "
              f"n_apo < n_holo + 1: {100 * (x.evidence_n_apo < x.nh1).mean():.1f}%; "
              f"median ratio log10((n_holo+1)/n_apo) {x.ratio.median():+.2f}")
    a, h = d[d.pocket_class == "apo"], d[d.pocket_class == "holo"]
    print(f"    between classes: median n_apo AF-Apo / AF-Holo = {a.evidence_n_apo.median():.0f} / {h.evidence_n_apo.median():.0f}; "
          f"median n_holo AF-Apo / AF-Holo = {a.evidence_n_holo.median():.0f} / {h.evidence_n_holo.median():.0f}")
    print(f"    Mann-Whitney AUC (prob. a random AF-Apo site has more n_apo than a random AF-Holo site): "
          f"{_auc(a.evidence_n_apo, h.evidence_n_apo):.3f}; for n_holo (AF-Holo > AF-Apo): {_auc(h.evidence_n_holo, a.evidence_n_holo):.3f}")


def _auc(x, y):
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(u / (len(x) * len(y)))


def item9():
    d = load_st100()
    d["sep"] = 1.0 - d.pocket_overlap_pair
    d["sep_tier"] = pd.cut(d.sep, [-np.inf, SEP_EDGES[0], SEP_EDGES[1], np.inf], labels=TIERS)
    d["dens_tier"] = pd.cut(d.pocket_density_pair, [-np.inf, SEP_EDGES[0], SEP_EDGES[1], np.inf], labels=TIERS)
    d["prob_tier"] = pd.cut(d.win, [-np.inf, PROB_EDGES[0], PROB_EDGES[1], np.inf], labels=TIERS)
    print("=" * 100)
    print(f"ITEM 9 -- Figure 4e tiers by class (ST_100, ties excluded). n = {len(d):,}; "
          f"separation missing: {int(d.sep.isna().sum())}")
    for key, name in [("prob_tier", "winning probability"), ("dens_tier", "density"), ("sep_tier", "separation")]:
        t = pd.crosstab(d[key], d.pocket_class)
        t["AF-Holo share %"] = 100 * t.holo / (t.apo + t.holo)
        comp = pd.crosstab(d.pocket_class, d[key], normalize="index") * 100
        print(f"\n  {name} tiers (edges {PROB_EDGES if key == 'prob_tier' else SEP_EDGES}):")
        print("    composition of each tier:")
        print(t.round(1).to_string())
        print("    tier distribution within each class (% of class):")
        print(comp.round(1).to_string())
    a, h = d[d.pocket_class == "apo"], d[d.pocket_class == "holo"]
    print(f"\n  median separation AF-Apo {a.sep.median():.3f} vs AF-Holo {h.sep.median():.3f}; "
          f"Spearman(separation, P(holo)) = {spearmanr(d.sep, d.pocket_p_holo).correlation:+.3f}; "
          f"Spearman(separation, winning p) = {spearmanr(d.sep, d.win).correlation:+.3f}; "
          f"Spearman(density, P(holo)) = {spearmanr(d.pocket_density_pair, d.pocket_p_holo).correlation:+.3f}")
    print(f"  AF-Holo share: low-separation tier {100 * (d[d.sep_tier == 'low'].holo.mean()):.1f}%, "
          f"high-separation tier {100 * (d[d.sep_tier == 'high'].holo.mean()):.1f}%, overall {100 * d.holo.mean():.1f}%")


def item4():
    d = load_st100()
    print("=" * 100)
    print(f"ITEM 4 (site level) -- ST_100 classified sites n = {len(d):,}")
    a, h = d[d.pocket_class == "apo"], d[d.pocket_class == "holo"]
    print(f"  AF-Apo {len(a):,} vs AF-Holo {len(h):,} sites ({100 * len(a) / len(d):.1f}% / {100 * len(h) / len(d):.1f}%)")
    for lab, fa, fh in [("n_apo", a.evidence_n_apo, h.evidence_n_apo), ("n_holo", a.evidence_n_holo, h.evidence_n_holo),
                        ("n_apo + n_holo", a.evidence_n_apo + a.evidence_n_holo, h.evidence_n_apo + h.evidence_n_holo)]:
        print(f"  median {lab:14s} AF-Apo {fa.median():.0f}  AF-Holo {fh.median():.0f}   (means {fa.mean():.1f} / {fh.mean():.1f})")
    print(f"  pooled over all classified sites: median n_apo {d.evidence_n_apo.median():.0f}, median n_holo {d.evidence_n_holo.median():.0f}; "
          f"sites with n_apo > n_holo: {100 * (d.evidence_n_apo > d.evidence_n_holo).mean():.1f}%; "
          f"n_holo > n_apo: {100 * (d.evidence_n_holo > d.evidence_n_apo).mean():.1f}%")
    print(f"  total apo structures (member-average summed over sites) {d.evidence_n_apo.sum():,.0f} vs holo {d.evidence_n_holo.sum():,.0f}")


def item8():
    print("=" * 100)
    print("ITEM 8 -- from ST_100 sites to the detection set")
    u = load_st100(classified=False)
    n0 = len(u)
    print(f"  ST_100 unified binding sites (Table 1; incl. {int((u.pocket_class == 'tie').sum())} ties): {n0:,}")
    f1 = u[u.n_uniprots.eq(1)]
    print(f"  single UniProt accession (n_uniprots == 1):        {len(f1):,}   (drops {n0 - len(f1):,})")
    f2 = f1[f1.n_chains.eq(1)]
    print(f"  single chain (n_chains == 1):                       {len(f2):,}   (drops {len(f1) - len(f2):,})")
    f3 = f2[~f2.is_interface.astype(bool)]
    print(f"  not an interface site (is_interface == False):      {len(f3):,}   (drops {len(f2) - len(f3):,})")
    print(f"  [the three conditions jointly drop {n0 - len(f3):,} sites; is_interface alone flags "
          f"{int(u.is_interface.astype(bool).sum()):,}, multi-chain alone {int((u.n_chains > 1).sum()):,}, "
          f"multi-UniProt alone {int((u.n_uniprots > 1).sum()):,}]")
    ds = pd.read_csv(DS, usecols=["site_uid", "multimeric_status_all_unique", "pocket_class"], low_memory=False)
    print(f"  monomeric in every member entry (multimeric_status_all_unique == 1; dataset builder): {len(ds):,}   "
          f"(drops {len(f3) - len(ds):,})")
    missing = set(f3.unit_uid) - set(ds.site_uid)
    print(f"    [check: {len(missing):,} of the {len(f3):,} unified survivors are absent from the dataset; "
          f"dataset rows not in the unified survivors: {len(set(ds.site_uid) - set(f3.unit_uid)):,}]")
    g = pd.read_csv(DSG, usecols=["site_uid", "pocket_class", "union_center_geom_x", "union_chain_resi", "afdb_filename",
                                  "union_n_residues_found"], low_memory=False)
    ok = g.union_center_geom_x.notna() & g.union_chain_resi.fillna("").astype(str).str.strip().ne("")
    print(f"  AlphaFold model found and pocket geometry computed:  {int(ok.sum()):,}   (drops {int((~ok).sum()):,}; "
          f"{int(g.afdb_filename.isna().sum()):,} without an AFDB model file, "
          f"{int((g.afdb_filename.notna() & ~ok).sum()):,} with a model but no mappable pocket residues)")
    gg = g[ok]
    vc = gg.pocket_class.astype(str).str.lower().value_counts()
    print(f"    class composition of the {len(gg):,} scored pockets: " + ", ".join(f"{k} {v:,}" for k, v in vc.items()))
    # the three LBS tables
    m = None
    for tool, path in LBS_TABLES.items():
        t = pd.read_csv(path, usecols=["site_uid", "pocket_class", "success_tns"], low_memory=False)
        t = t.rename(columns={"success_tns": f"tns_{tool}"})
        print(f"    {tool:10s} per-pocket table rows {len(t):,}; scored (success_tns not null) {int(t[f'tns_{tool}'].notna().sum()):,}")
        m = t if m is None else m.merge(t[["site_uid", f"tns_{tool}"]], on="site_uid", how="inner")
    scored = m[m[[c for c in m.columns if c.startswith("tns_")]].notna().all(axis=1)]
    cls = scored.pocket_class.astype(str).str.lower()
    print(f"  scored by all three predictors: {len(scored):,}; with a definite class (apo/holo): "
          f"{int(cls.isin(['apo', 'holo']).sum()):,} ({int((cls == 'apo').sum()):,} AF-Apo, {int((cls == 'holo').sum()):,} AF-Holo); "
          f"ties {int((~cls.isin(['apo', 'holo'])).sum()):,}")
    acc = pd.read_csv(LBS_TABLES["p2rank"], usecols=["site_uid", "site_uniprot"]).merge(scored[["site_uid"]])
    acc_cls = acc.merge(scored.loc[cls.isin(["apo", "holo"]), ["site_uid"]])
    print(f"  UniProt accessions in the scored set: {acc.site_uniprot.nunique():,}; "
          f"among the classified pockets: {acc_cls.site_uniprot.nunique():,}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"item1": item1, "item4": item4, "item7": item7, "item8": item8, "item9": item9}
    for k in ([what] if what != "all" else ["item1", "item4", "item7", "item8", "item9"]):
        fns[k]()
