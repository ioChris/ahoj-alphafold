# -*- coding: utf-8 -*-
"""CryptoBench as an external cryptic-site label: do the AlphaFold-side pocket descriptors recognise a cryptic site?
(Supplementary Table S13, and the main-text statements that cite it: the model shows the ligand-bound pocket at 72%
of CryptoBench's cryptic sites.)

CryptoBench (Škrhák et al. 2025, Bioinformatics 41:btae745; built on AHoJ-DB by four of our authors) pairs an apo
structure with holo structures of the same protein and calls the pocket cryptic when the apo-holo pocket RMSD exceeds
2 Å (dataset.json, 5,493 pairs on 1,107 apo structures, 1,361 pockets after clustering); its auxiliary file lists the
non-cryptic pockets of the same apo structures (pocket RMSD under 2 Å; noncryptic-pockets.json, 14,493 pairs). Every
pair's holo structure, chain, ligand and residue number is an AHoJ-DB query, so a pair maps to one of our analysis
entries by job name (pdb-chain-ligand-resnum). The label is external to our classifier (it never looks at the model)
but not to the database, and the non-cryptic pockets sit on the same apo structures, which makes them a control that
holds the protein and its record fixed.

Questions (unit: the mapped entry, cluster bootstrap over the UniProt key; a site-level repeat for detection):
  1. Which pocket descriptors separate CryptoBench's cryptic from its non-cryptic pockets (AUC toward cryptic)?
     Pocket pLDDT is the user's lead: it tracks conformational diversity (Figure 5), and a cryptic site is one whose
     record holds a large change. Diversity itself is the yardstick (its single-pair form is close to the label).
  2. Is pLDDT's signal more than diversity? AUC within diversity deciles; and within one apo structure, is the
     cryptic pocket modelled with lower pLDDT than the non-cryptic pockets of the same model?
  3. Which state does the model take at a cryptic site: the AF-Holo share (released and size-balanced labels, at a
     fixed evidence ratio), and does the model's own pocket sit within 2 Å of the ligand-bound pocket (CryptoBench's
     own criterion applied to the model)?
  4. Separation at cryptic sites, and the stringent ensemble form (every apo conformer > 2 Å).
  5. Detection on the model at cryptic sites (ST_100, AFDB v4 re-run): the AF-Holo - AF-Apo gap, size-standardized,
     at cryptic against non-cryptic sites, both labels; docking likewise. CryptoBench's own precedent is P2Rank's
     recall on experimental apo against holo forms of the same sites (0.62 against 0.84).

    python cryptobench_external_label.py summarise > cryptobench_external_label.out.txt
        -> <manuscript_analyses_dir>/cryptobench_external_label_summary.csv and cryptobench_external_label_entries.csv
           (the mapped pairs with their descriptors)
Inputs: the CryptoBench dataset folder (cryptobench_dir: the OSF deposit pz4a9, folder cryptobench-dataset), the
unified table, the query summaries and metrics, the harvested model metrics, balanced_entries.csv
(balanced_classification.py), <manuscript_analyses_dir>/ensemble_containment_entries.csv (regenerable:
ensemble_containment.py collect) and the AFDB v4 detection and docking tables through separation_shape.py.
AHOJ_AF_MODEL_SET defaults to afdbv4 here.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("AHOJ_AF_MODEL_SET", "afdbv4")
import ma_paths as P  # noqa: E402  (after the switch: common/config.py reads it at import)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import balanced_classification as bc  # noqa: E402
import separation_shape as S  # noqa: E402

CB = str(P.anchor("cryptobench_dir", "AHOJ_CRYPTOBENCH_DIR"))
CB_CRYPTIC = os.path.join(CB, "dataset.json")
CB_NONCRYPTIC = os.path.join(CB, "auxiliary-data", "non-cryptic-pockets", "noncryptic-pockets.json")
METRICS_CSV = os.path.join(str(P.cfg.CLASSIFICATION_MERGED_DIR), "query_metrics_do_kde_plots_and_classification.csv")
MODEL_CSV = os.path.join(str(P.cfg.AHOJ_DB_DIR), "Summaries", "model_metrics", "full_run_3",
                         "run_23082026_180200_do_collect_model_pLDDT", "query_summaries_do_collect_model_pLDDT.csv")
SUMMARY_CSV = P.out("cryptobench_external_label_summary.csv")
ENTRIES_OUT = P.out("cryptobench_external_label_entries.csv")
B = 400
SEED = 20261006
TIERS = ["very low", "low", "moderate", "high", "very high"]
RATIO_BINS = [-np.inf, -1, 0, 1, 2, 3, np.inf]
RATIO_LABELS = ["< -1", "-1 to 0", "0 to 1", "1 to 2", "2 to 3", "> 3"]
ROWS = []


def rec(**k):
    ROWS.append(k)


def ec_entries_path():
    """ensemble_containment_entries.csv is regenerable (ensemble_containment.py collect) and read from P.OUT."""
    cands = [P.out("ensemble_containment_entries.csv")]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError("ensemble_containment_entries.csv: run `ensemble_containment.py collect`; tried " + "; ".join(cands))


# ------------------------------------------------------------------------------------------------ inputs

def load_cryptobench():
    rows = []
    for path, label in ((CB_CRYPTIC, "cryptic"), (CB_NONCRYPTIC, "non-cryptic")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for apo, lst in data.items():
            for h in lst:
                sel = h["holo_pocket_selection"]
                rows.append(dict(
                    label=label, apo_pdb=apo.lower(), apo_chain=h["apo_chain"], uniprot_id=h.get("uniprot_id", ""),
                    holo_pdb=h["holo_pdb_id"].lower(), holo_chain=h["holo_chain"], ligand=h["ligand"],
                    ligand_index=str(h["ligand_index"]), ligand_chain=h["ligand_chain"],
                    prmsd=h.get("pRMSD", np.nan), is_main=bool(h.get("is_main_holo_structure", False)),
                    n_res=len(sel), multi_chain=len({x.split("_")[0] for x in sel}) > 1,
                    apo_sel=";".join(sorted(h["apo_pocket_selection"]))))
    d = pd.DataFrame(rows)
    d["job_lc"] = d.holo_pdb + "-" + d.ligand_chain + "-" + d.ligand + "-" + d.ligand_index
    d["job_hc"] = d.holo_pdb + "-" + d.holo_chain + "-" + d.ligand + "-" + d.ligand_index
    return d


def load_entries():
    """One row per analysis entry (237,922) with the descriptors the questions need."""
    u = pd.read_csv(bc.UNI, usecols=["threshold", "rep_job", "pocket_class", "pocket_p_holo", "pocket_overlap_pair",
                                      "pocket_density_pair", "model_pocket_plddt", "uniprots_key", "is_interface"],
                    low_memory=False)
    u = u[u.threshold == "UNCLUSTERED"].drop(columns="threshold").rename(columns={"rep_job": "job_name"})
    u["sep"] = 1 - pd.to_numeric(u.pocket_overlap_pair, errors="coerce")
    u["tier"] = pd.cut(u.sep, bins=bc.TIER_EDGES, labels=TIERS)
    u["rel_cls"] = u.pocket_class.astype(str).str.lower()
    s = pd.read_csv(bc.SUMMARY, usecols=["job_name", "n_apo", "n_holo", "query_RoG", "#resis3", "query_resolution",
                                          "model_pocket_len"], low_memory=False).drop_duplicates("job_name")
    s = s.rename(columns={"#resis3": "n_resis"})
    q = pd.read_csv(METRICS_CSV, usecols=["job_name", "pocket_rms_mean", "pocket_rms_max", "local_flexibility",
                                           "global_flexibility"], low_memory=False).drop_duplicates("job_name")
    q = q.rename(columns={"pocket_rms_mean": "div_mean", "pocket_rms_max": "div_max"})
    m = pd.read_csv(MODEL_CSV, usecols=["job_name", "model_pocket_rms"], low_memory=False).drop_duplicates("job_name")
    b = bc._entries(bc.ENTRIES_CSV)[["job_name", "bal_p_holo", "bal_cls"]]
    e = pd.read_csv(ec_entries_path(), usecols=["job_name", "pocket_rms_a_min", "pocket_rms_a_max", "pocket_rms_a_mean"],
                    low_memory=False)
    d = u.merge(s, on="job_name", how="left").merge(q, on="job_name", how="left").merge(m, on="job_name", how="left")
    d = d.merge(b, on="job_name", how="left").merge(e, on="job_name", how="left")
    d["ratio"] = np.log2((d.n_holo + 1) / d.n_apo.where(d.n_apo > 0))
    d["ratio_bin"] = pd.cut(d.ratio, bins=RATIO_BINS, labels=RATIO_LABELS)
    d["model_rms"] = pd.to_numeric(d.model_pocket_rms, errors="coerce")
    d["stringent"] = d.pocket_rms_a_min > 2.0
    return d


def site_members():
    u = pd.read_csv(bc.UNI, usecols=["threshold", "unit_uid", "job_names"], low_memory=False)
    u = u[u.threshold == "ST_100"]
    ex = u[["unit_uid"]].assign(job_name=u.job_names.str.split(";")).explode("job_name")
    return ex


# ------------------------------------------------------------------------------------------------ statistics

def cluster_boot_indices(codes, reps=B, seed=SEED):
    rng = np.random.default_rng(seed)
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    nC = sc.max() + 1
    st = np.searchsorted(sc, np.arange(nC), "left")
    ln = np.searchsorted(sc, np.arange(nC), "right") - st
    out = []
    for _ in range(reps):
        pick = rng.integers(0, nC, nC)
        lp, sp = ln[pick], st[pick]
        off = np.zeros(len(lp), dtype=np.int64)
        np.cumsum(lp[:-1], out=off[1:])
        out.append(order[np.arange(int(lp.sum())) - np.repeat(off, lp) + np.repeat(sp, lp)])
    return out


def auc(x, y):
    """P(x_pos > x_neg) with ties at one half; y is boolean (positive = cryptic)."""
    x = np.asarray(x, float)
    y = np.asarray(y, bool)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_stat(fn, ix_list):
    v = np.array([fn(ix) for ix in ix_list], float)
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)


def fmt(p, lo, hi, d=3, signed=False):
    f = f"{{:+.{d}f}}" if signed else f"{{:.{d}f}}"
    return f"{f.format(p)} [{f.format(lo)}, {f.format(hi)}]"


def header(t):
    print(f"\n{'=' * 110}\n{t}\n{'=' * 110}")


# ------------------------------------------------------------------------------------------------ analyses

def mapping(cb, ent):
    header("0. MAPPING CryptoBench -> analysis entries")
    jobs = set(ent.job_name)
    cb["job_name"] = np.where(cb.job_lc.isin(jobs), cb.job_lc, np.where(cb.job_hc.isin(jobs), cb.job_hc, ""))
    cb["mapped"] = cb.job_name != ""
    for lab, g in cb.groupby("label"):
        pk = g.groupby(["apo_pdb", "apo_chain", "apo_sel"]).ngroups
        pk_m = g[g.mapped].groupby(["apo_pdb", "apo_chain", "apo_sel"]).ngroups
        print(f"  {lab:12s} pairs {len(g):6,}  mapped {int(g.mapped.sum()):6,} ({100 * g.mapped.mean():.1f}%)   "
              f"apo structures {g.apo_pdb.nunique():5,} -> {g[g.mapped].apo_pdb.nunique():5,}   "
              f"distinct apo pockets {pk:6,} -> {pk_m:6,}   UniProts {g.uniprot_id.nunique():5,} -> "
              f"{g[g.mapped].uniprot_id.nunique():5,}   multi-chain pairs {int(g.multi_chain.sum()):5,} "
              f"(mapped {100 * g[g.multi_chain].mapped.mean():.1f}%)")
        rec(section="0 mapping", group=lab, metric="pairs", value=len(g))
        rec(section="0 mapping", group=lab, metric="pairs mapped", value=int(g.mapped.sum()))
        rec(section="0 mapping", group=lab, metric="pairs mapped %", value=100 * g.mapped.mean())
        rec(section="0 mapping", group=lab, metric="apo structures mapped", value=g[g.mapped].apo_pdb.nunique())
        rec(section="0 mapping", group=lab, metric="distinct apo pockets", value=pk)
        rec(section="0 mapping", group=lab, metric="distinct apo pockets mapped", value=pk_m)
        rec(section="0 mapping", group=lab, metric="uniprots mapped", value=g[g.mapped].uniprot_id.nunique())
    if "cryptic" in set(cb.label):
        main = cb[(cb.label == "cryptic") & cb.is_main]
        print(f"  main holo structure of each cryptic apo structure: {len(main):,} pairs, mapped {int(main.mapped.sum()):,}")
        rec(section="0 mapping", group="cryptic", metric="main pairs mapped", value=int(main.mapped.sum()))
    m = cb[cb.mapped]
    both = set(m[m.label == "cryptic"].job_name) & set(m[m.label == "non-cryptic"].job_name)
    print(f"  entries carrying both labels (cryptic against one apo structure, non-cryptic against another): {len(both):,}; dropped")
    rec(section="0 mapping", group="both", metric="entries with both labels", value=len(both))
    m = m[~m.job_name.isin(both)]
    # one row per entry and label; keep the CryptoBench pair fields of the main holo structure where there is one
    m = m.sort_values(["is_main"], ascending=False).drop_duplicates(["job_name", "label"])
    lab = m.merge(ent, on="job_name", how="left")
    lab["cryptic"] = lab.label == "cryptic"
    lab["cluster"] = pd.factorize(lab.uniprots_key.fillna(lab.uniprot_id))[0]
    print(f"  labelled entries: {len(lab):,} (cryptic {int(lab.cryptic.sum()):,}, non-cryptic {int((~lab.cryptic).sum()):,}); "
          f"UniProt clusters {lab.cluster.nunique():,}; interface entries {100 * lab.is_interface.astype(bool).mean():.1f}%")
    rec(section="0 mapping", group="entries", metric="labelled entries", value=len(lab))
    rec(section="0 mapping", group="entries", metric="cryptic entries", value=int(lab.cryptic.sum()))
    rec(section="0 mapping", group="entries", metric="non-cryptic entries", value=int((~lab.cryptic).sum()))
    rec(section="0 mapping", group="entries", metric="uniprot clusters", value=lab.cluster.nunique())
    unm = cb[(cb.label == "cryptic") & ~cb.mapped]
    print(f"  unmapped cryptic pairs: {len(unm):,}; of them multi-chain {int(unm.multi_chain.sum()):,}")
    return lab


DESCRIPTORS = [
    ("model_pocket_plddt", "pocket pLDDT (model)"),
    ("div_mean", "diversity: mean pocket RMSD of all conformers to the query"),
    ("pocket_rms_a_mean", "mean pocket RMSD of the apo conformers"),
    ("pocket_rms_a_max", "max apo pocket RMSD (single-pair crypticity)"),
    ("pocket_rms_a_min", "min apo pocket RMSD (stringent crypticity)"),
    ("sep", "separation (1 - pair overlap)"),
    ("pocket_density_pair", "density"),
    ("pocket_p_holo", "P(holo), released"),
    ("bal_p_holo", "P(holo), size-balanced"),
    ("ratio", "evidence ratio log2((n_holo+1)/n_apo)"),
    ("n_apo", "n apo"),
    ("n_holo", "n holo"),
    ("query_RoG", "pocket radius of gyration (query)"),
    ("n_resis", "pocket residues (query)"),
    ("model_rms", "model pocket RMSD to the query holo pocket"),
    ("local_flexibility", "local flexibility"),
    ("query_resolution", "query resolution"),
]


def descriptors(lab, ent):
    header("1. DESCRIPTORS: cryptic against non-cryptic entries (AUC toward cryptic; 95% UniProt-cluster bootstrap)")
    ix = cluster_boot_indices(lab.cluster.to_numpy())
    y = lab.cryptic.to_numpy()
    print(f"  {'descriptor':60s} {'n':>7s} {'med cryptic':>12s} {'med non-cr':>11s} {'med all set':>12s}   AUC [95% CI]")
    for col, name in DESCRIPTORS:
        x = pd.to_numeric(lab[col], errors="coerce").to_numpy(float)
        ok = ~np.isnan(x)
        p = auc(x, y)
        lo, hi = boot_stat(lambda i: auc(x[i], y[i]), ix)
        mc, mn = np.nanmedian(x[y]), np.nanmedian(x[~y])
        ma = pd.to_numeric(ent[col], errors="coerce").median()
        print(f"  {name:60s} {int(ok.sum()):7,} {mc:12.3f} {mn:11.3f} {ma:12.3f}   {fmt(p, lo, hi)}")
        rec(section="1 descriptors", group=col, metric="AUC", value=p, lo=lo, hi=hi, n=int(ok.sum()))
        rec(section="1 descriptors", group=col, metric="median cryptic", value=mc)
        rec(section="1 descriptors", group=col, metric="median non-cryptic", value=mn)
        rec(section="1 descriptors", group=col, metric="median analysis set", value=ma)
    # one row per CryptoBench pocket (its main holo structure where flagged), so a pocket with many holo
    # structures does not weigh more than one with a single pair
    pk = lab.sort_values("is_main", ascending=False).drop_duplicates(["apo_pdb", "apo_chain", "label", "apo_sel"])
    ixp = cluster_boot_indices(pk.cluster.to_numpy())
    yp = pk.cryptic.to_numpy()
    print(f"\n  one row per CryptoBench pocket: {len(pk):,} (cryptic {int(yp.sum()):,}, non-cryptic {int((~yp).sum()):,})")
    for col, name in DESCRIPTORS[:6] + [DESCRIPTORS[7], DESCRIPTORS[8], DESCRIPTORS[14]]:
        x = pd.to_numeric(pk[col], errors="coerce").to_numpy(float)
        p = auc(x, yp)
        lo, hi = boot_stat(lambda i: auc(x[i], yp[i]), ixp)
        print(f"  {name:60s} {int((~np.isnan(x)).sum()):7,} {np.nanmedian(x[yp]):12.3f} {np.nanmedian(x[~yp]):11.3f} {'':12s}   {fmt(p, lo, hi)}")
        rec(section="1 descriptors per pocket", group=col, metric="AUC", value=p, lo=lo, hi=hi, n=int((~np.isnan(x)).sum()))
    rec(section="1 descriptors per pocket", group="pockets", metric="n", value=len(pk))
    # operating points: what a cut-off on each descriptor would catch (sensitivity = cryptic entries flagged,
    # false-positive rate = non-cryptic entries flagged)
    print("\n  operating points (per pair): sensitivity / false-positive rate of a cut-off, in %")
    for col, name, cuts, below in (("model_pocket_plddt", "pLDDT below", (85, 90, 95, 97), True),
                                   ("div_mean", "diversity above (Å)", (0.75, 1.0, 1.5, 2.0), False),
                                   ("sep", "separation above", (0.3, 0.45, 0.6, 0.85), False)):
        x = pd.to_numeric(lab[col], errors="coerce").to_numpy(float)
        for c in cuts:
            flag = (x < c) if below else (x > c)
            se, fp = 100 * flag[y].mean(), 100 * flag[~y].mean()
            print(f"    {name:22s} {c:5}: sensitivity {se:5.1f}  false-positive {fp:5.1f}")
            rec(section="1 operating points", group=f"{col}|{c}", metric="sensitivity %", value=se)
            rec(section="1 operating points", group=f"{col}|{c}", metric="false-positive %", value=fp)
    # pLDDT bands
    print("\n  pocket pLDDT bands (share of entries), cryptic / non-cryptic / analysis set:")
    edges = [0, 50, 70, 90, 101]
    names = ["< 50", "50-70", "70-90", ">= 90"]
    for lo_, hi_, nm in zip(edges[:-1], edges[1:], names):
        f = lambda s: 100 * ((s >= lo_) & (s < hi_)).mean()  # noqa: E731
        vc, vn, va = f(lab.model_pocket_plddt[y]), f(lab.model_pocket_plddt[~y]), f(ent.model_pocket_plddt)
        print(f"    {nm:6s} {vc:5.1f} / {vn:5.1f} / {va:5.1f}")
        rec(section="1 plddt bands", group=nm, metric="cryptic %", value=vc)
        rec(section="1 plddt bands", group=nm, metric="non-cryptic %", value=vn)
        rec(section="1 plddt bands", group=nm, metric="analysis set %", value=va)


def plddt_beyond_diversity(lab):
    header("2. IS pLDDT MORE THAN DIVERSITY?")
    ix = cluster_boot_indices(lab.cluster.to_numpy())
    y = lab.cryptic.to_numpy()
    pl = lab.model_pocket_plddt.to_numpy(float)
    for col, name in (("div_mean", "diversity (mean pocket RMSD, all conformers)"),
                      ("pocket_rms_a_max", "max apo pocket RMSD"), ("query_RoG", "pocket size (RoG)"),
                      ("n_apo", "n apo")):
        x = pd.to_numeric(lab[col], errors="coerce").to_numpy(float)
        dec = pd.qcut(pd.Series(x), 10, labels=False, duplicates="drop").to_numpy()

        def strat(i, dec=dec):
            num = den = 0.0
            for k in np.unique(dec[i][~np.isnan(dec[i])]):
                m = i[dec[i] == k]
                n1, n0 = y[m].sum(), (~y[m]).sum()
                a = auc(pl[m], y[m])
                if n1 and n0 and not np.isnan(a):
                    num += a * n1 * n0
                    den += n1 * n0
            return num / den if den else np.nan
        p = strat(np.arange(len(lab)))
        lo, hi = boot_stat(strat, ix)
        print(f"  pLDDT AUC within deciles of {name:48s} {fmt(p, lo, hi)}")
        rec(section="2 plddt stratified", group=col, metric="stratified AUC", value=p, lo=lo, hi=hi)
    # the reverse: diversity's AUC within pLDDT deciles
    dv = lab.div_mean.to_numpy(float)
    dec = pd.qcut(pd.Series(pl), 10, labels=False, duplicates="drop").to_numpy()

    def strat2(i):
        num = den = 0.0
        for k in np.unique(dec[i][~np.isnan(dec[i])]):
            m = i[dec[i] == k]
            n1, n0 = y[m].sum(), (~y[m]).sum()
            a = auc(dv[m], y[m])
            if n1 and n0 and not np.isnan(a):
                num += a * n1 * n0
                den += n1 * n0
        return num / den if den else np.nan
    p = strat2(np.arange(len(lab)))
    lo, hi = boot_stat(strat2, ix)
    print(f"  diversity AUC within deciles of pLDDT{'':40s} {fmt(p, lo, hi)}")
    rec(section="2 plddt stratified", group="div_mean|plddt", metric="stratified AUC", value=p, lo=lo, hi=hi)
    # Spearman of pLDDT with diversity inside the labelled frame, each label
    from scipy.stats import spearmanr
    for nm, msk in (("cryptic", y), ("non-cryptic", ~y)):
        ok = msk & ~np.isnan(pl) & ~np.isnan(dv)
        r = spearmanr(pl[ok], dv[ok]).correlation
        print(f"  Spearman pLDDT vs diversity within {nm:12s} entries: {r:+.3f} (n {int(ok.sum()):,})")
        rec(section="2 plddt stratified", group=nm, metric="spearman plddt diversity", value=r)

    # within one apo structure: cryptic pocket against the non-cryptic pockets of the same model
    print("\n  Within one apo structure (same protein, same model): mean pLDDT of its cryptic pocket(s) minus that of its "
          "non-cryptic pocket(s)")
    g = lab.groupby(["apo_pdb", "apo_chain", "label"]).agg(plddt=("model_pocket_plddt", "mean"), cluster=("cluster", "first"),
                                                            div=("div_mean", "mean"), rog=("query_RoG", "mean")).reset_index()
    w = g.pivot_table(index=["apo_pdb", "apo_chain"], columns="label", values=["plddt", "div", "rog"], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.dropna(subset=["plddt_cryptic", "plddt_non-cryptic"]).reset_index()
    w = w.merge(g[["apo_pdb", "apo_chain", "cluster"]].drop_duplicates(["apo_pdb", "apo_chain"]), on=["apo_pdb", "apo_chain"])
    d = (w["plddt_cryptic"] - w["plddt_non-cryptic"]).to_numpy()
    dd = (w["div_cryptic"] - w["div_non-cryptic"]).to_numpy()
    dr = (w["rog_cryptic"] - w["rog_non-cryptic"]).to_numpy()
    ixw = cluster_boot_indices(w.cluster.to_numpy())
    med = np.median(d)
    lo, hi = boot_stat(lambda i: np.median(d[i]), ixw)
    sh = 100 * (d < 0).mean()
    slo, shi = boot_stat(lambda i: 100 * (d[i] < 0).mean(), ixw)
    print(f"    apo structures with both labels: {len(w):,} ({w.cluster.nunique():,} UniProt clusters)")
    print(f"    median difference (cryptic - non-cryptic) {fmt(med, lo, hi, 2, True)} pLDDT; cryptic lower in {fmt(sh, slo, shi, 1)}%")
    print(f"    the same structures: diversity difference median {np.median(dd):+.2f} Å, pocket RoG difference median {np.median(dr):+.2f} Å")
    rec(section="2 within structure", group="plddt", metric="n apo structures", value=len(w))
    rec(section="2 within structure", group="plddt", metric="median difference", value=med, lo=lo, hi=hi)
    rec(section="2 within structure", group="plddt", metric="cryptic lower %", value=sh, lo=slo, hi=shi)
    rec(section="2 within structure", group="diversity", metric="median difference", value=float(np.median(dd)))
    rec(section="2 within structure", group="rog", metric="median difference", value=float(np.median(dr)))
    # the same restricted to confident models (both pockets >= 90)
    hi90 = (w.plddt_cryptic >= 90) & (w["plddt_non-cryptic"] >= 90)
    if hi90.sum() > 20:
        d9 = d[hi90.to_numpy()]
        print(f"    both pockets >= 90: n {int(hi90.sum()):,}, median difference {np.median(d9):+.2f}, cryptic lower in {100 * (d9 < 0).mean():.1f}%")
        rec(section="2 within structure", group="plddt>=90", metric="n apo structures", value=int(hi90.sum()))
        rec(section="2 within structure", group="plddt>=90", metric="median difference", value=float(np.median(d9)))
        rec(section="2 within structure", group="plddt>=90", metric="cryptic lower %", value=100 * (d9 < 0).mean())


def state(lab, ent):
    header("3. WHICH STATE DOES THE MODEL TAKE AT A CRYPTIC SITE?")
    ix = cluster_boot_indices(lab.cluster.to_numpy())
    y = lab.cryptic.to_numpy()
    for col, nm in (("rel_cls", "released"), ("bal_cls", "size-balanced")):
        c = lab[col].astype(str).to_numpy()
        allset = ent[col].astype(str)
        print(f"\n  AF-Holo share among definite calls, {nm} label (ties excluded):")
        for lbl, msk in (("cryptic", y), ("non-cryptic", ~y)):
            def f(i, msk=msk):
                m = msk[i] & np.isin(c[i], ["apo", "holo"])
                return 100 * (c[i][m] == "holo").mean()
            p = f(np.arange(len(lab)))
            lo, hi = boot_stat(f, ix)
            n = int((msk & np.isin(c, ["apo", "holo"])).sum())
            print(f"    {lbl:12s} n {n:6,}  {fmt(p, lo, hi, 1)}%")
            rec(section="3 class", group=f"{nm}|{lbl}", metric="AF-Holo %", value=p, lo=lo, hi=hi, n=n)
        a = allset[allset.isin(["apo", "holo"])]
        print(f"    {'analysis set':12s} n {len(a):6,}  {100 * (a == 'holo').mean():.1f}%")
        rec(section="3 class", group=f"{nm}|analysis set", metric="AF-Holo %", value=100 * (a == "holo").mean(), n=len(a))
        print(f"    by evidence ratio bin (log2), cryptic / non-cryptic AF-Holo % (n):")
        for rb in RATIO_LABELS:
            out = []
            for lbl, msk in (("cryptic", y), ("non-cryptic", ~y)):
                m = msk & (lab.ratio_bin.astype(str).to_numpy() == rb) & np.isin(c, ["apo", "holo"])
                v = 100 * (c[m] == "holo").mean() if m.sum() else np.nan
                out.append((v, int(m.sum())))
                rec(section="3 class by ratio", group=f"{nm}|{lbl}|{rb}", metric="AF-Holo %", value=v, n=int(m.sum()))
            print(f"      {rb:8s} {out[0][0]:5.1f} ({out[0][1]:5,}) / {out[1][0]:5.1f} ({out[1][1]:5,})")
    # the model's own pocket against the ligand-bound pocket: CryptoBench's 2 A rule applied to the model
    print("\n  The model's pocket RMSD to the query (ligand-bound) pocket; 'formed' = within 2 Å, CryptoBench's rule:")
    mr = lab.model_rms.to_numpy(float)
    for lbl, msk in (("cryptic", y), ("non-cryptic", ~y)):
        ok = msk & ~np.isnan(mr)
        def f(i, msk=msk):
            m = msk[i] & ~np.isnan(mr[i])
            return 100 * (mr[i][m] <= 2.0).mean()
        p = f(np.arange(len(lab)))
        lo, hi = boot_stat(f, ix)
        print(f"    {lbl:12s} n {int(ok.sum()):6,}  median {np.median(mr[ok]):.2f} Å   formed {fmt(p, lo, hi, 1)}%")
        rec(section="3 model formed", group=lbl, metric="formed %", value=p, lo=lo, hi=hi, n=int(ok.sum()))
        rec(section="3 model formed", group=lbl, metric="median model RMSD", value=float(np.median(mr[ok])))
        for col, nm in (("rel_cls", "released"), ("bal_cls", "balanced")):
            c = lab[col].astype(str).to_numpy()
            for cl in ("apo", "holo"):
                m = ok & (c == cl)
                v = 100 * (mr[m] <= 2.0).mean() if m.sum() else np.nan
                print(f"        {nm:9s} AF-{cl.capitalize():5s} n {int(m.sum()):6,}  median {np.median(mr[m]) if m.sum() else np.nan:.2f} Å   formed {v:.1f}%")
                rec(section="3 model formed", group=f"{lbl}|{nm}|{cl}", metric="formed %", value=v, n=int(m.sum()))
    a = ent.model_rms.dropna()
    print(f"    {'analysis set':12s} n {len(a):6,}  median {a.median():.2f} Å   formed {100 * (a <= 2).mean():.1f}%")
    rec(section="3 model formed", group="analysis set", metric="formed %", value=100 * (a <= 2).mean(), n=len(a))
    # the model's distance against the apo structure's: is the model nearer the holo than CryptoBench's apo was?
    cr = lab[y & ~np.isnan(mr) & lab.prmsd.notna()]
    from scipy.stats import spearmanr
    r = spearmanr(cr.model_rms, cr.prmsd).correlation
    nearer = 100 * (cr.model_rms < cr.prmsd).mean()
    print(f"    cryptic pairs with both: model RMSD vs CryptoBench's apo-holo pRMSD, Spearman {r:+.3f}; model nearer the holo "
          f"pocket than the apo structure in {nearer:.1f}% (n {len(cr):,})")
    rec(section="3 model formed", group="cryptic", metric="spearman model rms vs prmsd", value=r, n=len(cr))
    rec(section="3 model formed", group="cryptic", metric="model nearer than apo %", value=nearer, n=len(cr))


def separation(lab, ent):
    header("4. SEPARATION AT CRYPTIC SITES")
    y = lab.cryptic.to_numpy()
    print(f"  {'tier':10s} {'cryptic %':>10s} {'non-cr %':>10s} {'all set %':>10s}")
    for t in TIERS:
        vc = 100 * (lab.tier[y].astype(str) == t).mean()
        vn = 100 * (lab.tier[~y].astype(str) == t).mean()
        va = 100 * (ent.tier.astype(str) == t).mean()
        print(f"  {t:10s} {vc:10.1f} {vn:10.1f} {va:10.1f}")
        rec(section="4 separation tiers", group=t, metric="cryptic %", value=vc)
        rec(section="4 separation tiers", group=t, metric="non-cryptic %", value=vn)
        rec(section="4 separation tiers", group=t, metric="analysis set %", value=va)
    hi = lambda s: 100 * s.astype(str).isin(["high", "very high"]).mean()  # noqa: E731
    print(f"  high + very high: cryptic {hi(lab.tier[y]):.1f}%, non-cryptic {hi(lab.tier[~y]):.1f}%, analysis set {hi(ent.tier):.1f}%")
    rec(section="4 separation tiers", group="high+very high", metric="cryptic %", value=hi(lab.tier[y]))
    rec(section="4 separation tiers", group="high+very high", metric="non-cryptic %", value=hi(lab.tier[~y]))
    rec(section="4 separation tiers", group="high+very high", metric="analysis set %", value=hi(ent.tier))
    for lbl, msk in (("cryptic", y), ("non-cryptic", ~y)):
        s = lab.stringent[msk]
        print(f"  stringent (every apo conformer > 2 Å from the query): {lbl} {100 * s.mean():.1f}% (n {len(s):,})")
        rec(section="4 stringent", group=lbl, metric="stringent %", value=100 * s.mean(), n=len(s))
    print(f"  stringent, analysis set: {100 * ent.stringent.mean():.1f}%")
    rec(section="4 stringent", group="analysis set", metric="stringent %", value=100 * ent.stringent.mean())
    print("  (a non-cryptic pair has by definition an apo conformer within 2 Å, so its stringent share is 0 by construction)")
    # the cryptic pockets split by the ensemble criterion
    print("\n  Cryptic entries, stringent (every apo > 2 Å) against the rest (some apo conformer within 2 Å of the query):")
    print(f"  {'subset':12s} {'n':>6s} {'AF-Holo rel %':>14s} {'AF-Holo bal %':>14s} {'formed %':>9s} {'med pLDDT':>10s} {'med sep':>8s} {'med n_apo':>10s}")
    cr = lab[y]
    for nm, msk in (("stringent", cr.stringent), ("not", ~cr.stringent)):
        s = cr[msk]
        rel = s.rel_cls[s.rel_cls.isin(["apo", "holo"])]
        bal = s.bal_cls[s.bal_cls.isin(["apo", "holo"])]
        formed = 100 * (s.model_rms <= 2).mean()
        print(f"  {nm:12s} {len(s):6,} {100 * (rel == 'holo').mean():14.1f} {100 * (bal == 'holo').mean():14.1f} {formed:9.1f} "
              f"{s.model_pocket_plddt.median():10.2f} {s.sep.median():8.3f} {s.n_apo.median():10.0f}")
        rec(section="4 cryptic stringent split", group=nm, metric="n", value=len(s))
        rec(section="4 cryptic stringent split", group=nm, metric="AF-Holo rel %", value=100 * (rel == "holo").mean())
        rec(section="4 cryptic stringent split", group=nm, metric="AF-Holo bal %", value=100 * (bal == "holo").mean())
        rec(section="4 cryptic stringent split", group=nm, metric="formed %", value=formed)
        rec(section="4 cryptic stringent split", group=nm, metric="median plddt", value=float(s.model_pocket_plddt.median()))
        rec(section="4 cryptic stringent split", group=nm, metric="median n_apo", value=float(s.n_apo.median()))


def site_labels(lab):
    """ST_100 site label from the mapped members: cryptic if any member is a cryptic pair and none a non-cryptic one."""
    ex = site_members()
    m = ex.merge(lab[["job_name", "label"]], on="job_name", how="inner")
    g = m.groupby("unit_uid").label.agg(lambda s: set(s))
    lab_site = g.map(lambda s: "cryptic" if s == {"cryptic"} else ("non-cryptic" if s == {"non-cryptic"} else "mixed"))
    return lab_site.rename("cb_label").reset_index()


def balanced_site_labels(lbs):
    d = bc._entries(bc.ENTRIES_CSV)
    ex = site_members().merge(d[["job_name", "bal_p_holo"]], on="job_name", how="left")
    agg = ex.groupby("unit_uid").agg(n_mem=("job_name", "size"), n_ok=("bal_p_holo", "count"), bal_p=("bal_p_holo", "mean")).reset_index()
    agg["bal_cls"] = np.where(agg.n_ok == agg.n_mem, bc._cls(agg.bal_p), "")
    return lbs.merge(agg[["unit_uid", "bal_cls"]], on="unit_uid", how="left")


def detection(lab):
    header("5. DETECTION AND DOCKING ON THE MODEL AT CRYPTIC SITES (ST_100, AFDB v4 re-run)")
    lbs = S.load_lbs()
    lbs = balanced_site_labels(lbs)
    sl = site_labels(lab)
    lbs = lbs.merge(sl, on="unit_uid", how="left")
    lbs["cb_label"] = lbs.cb_label.fillna("unlabelled")
    dk = S.load_docking(lbs)
    vc = lbs.cb_label.value_counts()
    print(f"  detection pockets {len(lbs):,}: cryptic {vc.get('cryptic', 0):,}, non-cryptic {vc.get('non-cryptic', 0):,}, "
          f"mixed {vc.get('mixed', 0):,} (excluded), unlabelled {vc.get('unlabelled', 0):,}; docking pockets {len(dk):,} "
          f"(cryptic {int((dk.cb_label == 'cryptic').sum()):,}, non-cryptic {int((dk.cb_label == 'non-cryptic').sum()):,})")
    for k, v in vc.items():
        rec(section="5 sites", group=k, metric="detection pockets", value=int(v))
    rec(section="5 sites", group="cryptic", metric="docking pockets", value=int((dk.cb_label == "cryptic").sum()))
    rec(section="5 sites", group="non-cryptic", metric="docking pockets", value=int((dk.cb_label == "non-cryptic").sum()))
    cells = {"non-cryptic": 0, "cryptic": 1, "unlabelled": 2}
    tasks = S.PRED + ["consensus", "docking"]
    for label_col, nm in (("cls", "released"), ("bal_cls", "size-balanced")):
        print(f"\n  -- {nm} labels: size-standardized AF-Apo / AF-Holo rates and the gap (pp) [95% cluster-bootstrap CI]")
        print(f"  {'task':12s} {'sites':12s} {'n_apo':>6s} {'n_holo':>6s} {'apo':>6s} {'holo':>6s}   {'gap':>22s}   {'raw apo':>7s} {'raw holo':>8s}")
        for t in tasks:
            src = dk if t == "docking" else lbs
            src = src.copy()
            src["cls"] = src[label_col].astype(str)
            src = src[src.cls.isin(["apo", "holo"])]
            g = S.Gap(src, t)
            cell = g.d.cb_label.map(cells).fillna(-1).astype(int).to_numpy()
            pt = g.point(cell, 3)
            bs = g.boot(cell, 3)
            rt = g.rates(cell, 3)
            raw = g.rates(cell, 3, raw=True)
            cnt = g.d.groupby(["cb_label", "cls"]).size()
            for nm2, ci_ in cells.items():
                lo, hi = np.nanpercentile(bs[:, ci_], [2.5, 97.5])
                na, nh = cnt.get((nm2, "apo"), 0), cnt.get((nm2, "holo"), 0)
                print(f"  {t:12s} {nm2:12s} {na:6,} {nh:6,} {rt[ci_, 0]:6.1f} {rt[ci_, 1]:6.1f}   {fmt(pt[ci_], lo, hi, 1, True):>22s}   "
                      f"{raw[ci_, 0]:7.1f} {raw[ci_, 1]:8.1f}")
                rec(section="5 detection", group=f"{nm}|{t}|{nm2}", metric="gap", value=pt[ci_], lo=lo, hi=hi, n=int(na + nh))
                rec(section="5 detection", group=f"{nm}|{t}|{nm2}", metric="apo rate std", value=rt[ci_, 0], n=int(na))
                rec(section="5 detection", group=f"{nm}|{t}|{nm2}", metric="holo rate std", value=rt[ci_, 1], n=int(nh))
                rec(section="5 detection", group=f"{nm}|{t}|{nm2}", metric="apo rate raw", value=raw[ci_, 0], n=int(na))
                rec(section="5 detection", group=f"{nm}|{t}|{nm2}", metric="holo rate raw", value=raw[ci_, 1], n=int(nh))
            dlt = bs[:, 1] - bs[:, 0]
            lo, hi = np.nanpercentile(dlt, [2.5, 97.5])
            print(f"  {'':12s} {'cryptic - non-cryptic gap':26s} {fmt(pt[1] - pt[0], lo, hi, 1, True)}")
            rec(section="5 detection", group=f"{nm}|{t}|diff", metric="gap diff", value=pt[1] - pt[0], lo=lo, hi=hi)
            # the level: pooled detection at cryptic vs non-cryptic sites, size-standardized over both classes
            br = g.boot_rates(cell, 3)
            cnt_a = np.array([cnt.get((k, "apo"), 0) for k in cells])
            cnt_h = np.array([cnt.get((k, "holo"), 0) for k in cells])
            w_h = cnt_h / np.maximum(cnt_a + cnt_h, 1)
            lvl = rt[:, 0] * (1 - w_h) + rt[:, 1] * w_h
            blvl = br[:, :, 0] * (1 - w_h) + br[:, :, 1] * w_h
            dl = blvl[:, 1] - blvl[:, 0]
            lo, hi = np.nanpercentile(dl, [2.5, 97.5])
            print(f"  {'':12s} {'level (class mix kept)':26s} cryptic {lvl[1]:.1f}, non-cryptic {lvl[0]:.1f}, unlabelled {lvl[2]:.1f}; "
                  f"diff {fmt(lvl[1] - lvl[0], lo, hi, 1, True)}")
            rec(section="5 detection", group=f"{nm}|{t}|level", metric="cryptic level", value=lvl[1])
            rec(section="5 detection", group=f"{nm}|{t}|level", metric="non-cryptic level", value=lvl[0])
            rec(section="5 detection", group=f"{nm}|{t}|level", metric="level diff", value=lvl[1] - lvl[0], lo=lo, hi=hi)
    # P2Rank at cryptic sites, by class, raw: the counterpart of CryptoBench's apo/holo recall (0.62 / 0.84)
    s = lbs[(lbs.cb_label == "cryptic") & lbs.cls.isin(["apo", "holo"])]
    print(f"\n  P2Rank top-n success at cryptic sites on the model, raw: AF-Apo {100 * s[s.cls == 'apo'].P2Rank.mean():.1f}% "
          f"(n {int((s.cls == 'apo').sum()):,}), AF-Holo {100 * s[s.cls == 'holo'].P2Rank.mean():.1f}% (n {int((s.cls == 'holo').sum()):,})")
    # size: are cryptic sites larger?
    for nm2 in ("cryptic", "non-cryptic", "unlabelled"):
        g2 = lbs[lbs.cb_label == nm2]
        print(f"  {nm2:12s} median union RoG {g2.union_rg.median():.2f} Å; AF-Holo share {100 * (g2.cls == 'holo').mean():.1f}%; "
              f"separation high+ {100 * g2.sep.isin(['high', 'very_high']).mean():.1f}%")
        rec(section="5 sites", group=nm2, metric="median union rg", value=float(g2.union_rg.median()))
        rec(section="5 sites", group=nm2, metric="AF-Holo %", value=100 * (g2.cls == "holo").mean())


def summarise():
    cb = load_cryptobench()
    ent = load_entries()
    print(f"CryptoBench pairs {len(cb):,}; analysis entries {len(ent):,} (model set {os.environ['AHOJ_AF_MODEL_SET']})")
    lab = mapping(cb, ent)
    keep = ["job_name", "label", "apo_pdb", "apo_chain", "uniprot_id", "prmsd", "is_main", "n_res", "multi_chain", "cluster"] + \
           [c for c, _ in DESCRIPTORS] + ["rel_cls", "bal_cls", "tier", "stringent", "is_interface"]
    lab[keep].to_csv(ENTRIES_OUT, index=False)
    descriptors(lab, ent)
    plddt_beyond_diversity(lab)
    state(lab, ent)
    separation(lab, ent)
    detection(lab)
    pd.DataFrame(ROWS).to_csv(SUMMARY_CSV, index=False)
    print(f"\nwrote {SUMMARY_CSV} ({len(ROWS)} rows) and {ENTRIES_OUT}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summarise"
    if cmd == "summarise":
        summarise()
    else:
        raise SystemExit(f"unknown command {cmd!r}")
