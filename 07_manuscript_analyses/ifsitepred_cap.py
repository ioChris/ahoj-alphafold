# -*- coding: utf-8 -*-
"""How much does IF-SitePred's three-site output cap move its detection results? (Supplementary Table S12, the
paragraph on IF-SitePred's three-site cap: subset_rates.csv.)

IF-SitePred ranks its DBSCAN clusters by size and writes at most three (ifsitepred_raw_full_afdbv4.csv: ranks
1-3 only). Detection is scored as DCA4-TNS: a site counts as found when a predicted centre within 4 A of its
reference atoms is ranked within n, n = `n_unp_pockets`, the number of ST_100 sites of the accession. For a
site whose protein has n > 3, P2Rank and fpocket can be credited at ranks 4..n; IF-SitePred cannot.

Units. The manuscript scores SITES (87,434 detection pockets on 10,501 AlphaFold models), not proteins. Here a
protein is one model (one accession). Its N is reported three ways:
  N_ref  n_unp_pockets, the TNS rank cut (all ST_100 sites of the accession, interface sites included)
  N_det  detection sites on the model (the sites that are actually scored)
  n_eff  distinct reference regions (detection sites merged when their residue sets overlap by >= 0.5 of the
         smaller; detection_rank_cut.py's `neff50`)
ST_100 sites overlap heavily, so one predicted centre can hit many of them: 3 / N_det is NOT an upper bound on a
protein's site recall. The bound that holds in spirit is 3 regions, i.e. region recall <= 3 / n_eff (a centre
can still touch two adjacent regions, so this is checked, not assumed).

"Datasets / splits". There are no train/test splits here. The strata are the pocket class (AF-Apo / AF-Holo)
and the separation tier; proteins are split by the classes of their sites (only AF-Apo, only AF-Holo, mixed).

Section 4 compares, per tool, the pooled TNS rate, the raw class gap and the size-standardized class gap
(separation_shape.Gap, Figure 4's estimator: union-Rg deciles, cluster bootstrap over UniProt, B = 400,
seed 20260908) on
  all        the 87,138-pocket analysis set (published criterion)
  N_ref<=3   sites whose rank cut is at most 3, so the cap cannot bind for any tool
  n_eff<=3   sites whose protein has at most 3 distinct regions (three slots can cover every region)
  N_ref>3    the complement of N_ref<=3
  n_eff>3    the complement of n_eff<=3: the proteins where three slots cannot cover every region
and one criterion change, on all sites and on n_eff>3:
  cap3       every tool scored at rank <= min(n, 3), i.e. P2Rank and fpocket held to IF-SitePred's budget.
             What cap3 does to P2Rank's and fpocket's class gap is the best available estimate of what the
             cap does to IF-SitePred's (its clusters beyond the third were never written, so its uncapped
             result cannot be computed from the existing outputs).
Every row shares one set of resamples, so tool differences and subset differences are paired.

Inputs (read only): the detection base (06_lbs_generality/analysis/analyze_fig6a_quality_gates.load_base), the three scored
per-pocket tables (dca_rank, n_unp_pockets, predicted_pockets_total), detection_rank_cut.py's cached per-site
table <manuscript_analyses_dir>/rank_cut_afdbv4/site_ranks.csv (n_sites, neff50, candidates per tool), the ST_100
geometry table and receptors.csv (for the region labels). Gates: TNS recomputed from dca_rank and n equals the
released success for every tool and site; the analysis set is 87,138 pockets (35,124 / 52,014); the region
labels reproduce neff50 for every site; the size-standardized TNS gaps per separation tier equal
density_separation_gaps_afdbv4.csv (SI Table S3).

Usage (base conda python with scipy):

    $env:AHOJ_AF_MODEL_SET = "afdbv4"        # PowerShell; bash: export AHOJ_AF_MODEL_SET=afdbv4
    python ifsitepred_cap.py > <manuscript_analyses_dir>/ifsitepred_cap_afdbv4/out.txt

Writes into <manuscript_analyses_dir>/ifsitepred_cap_afdbv4/: proteins.csv (one row per model), subset_rates.csv
(section 4), counts.csv (sections 1-3). Nothing else is written. Also reads
<manuscript_analyses_dir>/density_separation_gaps_afdbv4.csv (density_separation_gaps.py) for its Table S3 gate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import ma_paths as P

HERE = Path(__file__).resolve().parent
if os.environ.get("AHOJ_AF_MODEL_SET", "").strip().lower() != "afdbv4":
    sys.exit("set AHOJ_AF_MODEL_SET=afdbv4 (the published detection set)")
PUBLIC = P.REPO
sys.path[:0] = [str(HERE), str(PUBLIC), str(PUBLIC / "06_lbs_generality" / "analysis")]

cfg = P.cfg                                          # common/config.py
import analyze_fig6a_quality_gates as G              # noqa: E402
import separation_shape as S                         # noqa: E402  Figure 4's estimator (import reads nothing)

OUTDIR = Path(P.out("ifsitepred_cap_afdbv4"))
RANKS_CSV = Path(P.OUT) / "rank_cut_afdbv4" / "site_ranks.csv"      # detection_rank_cut.py ranks
REF_S3 = Path(P.out("density_separation_gaps_afdbv4.csv"))
TOOLS = [("p2rank", "P2Rank"), ("fpocket", "fpocket"), ("ifsitepred", "IF-SitePred")]
PRED = [n for _, n in TOOLS]
CAP = 3


def pct(x):
    x = np.asarray(x, float)
    return 100.0 * np.nanmean(x) if np.isfinite(x).any() else np.nan


def header(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


# ============================================================================================== load
def region_labels(models, res, thr=0.5):
    """Per site, its distinct-region label on the model (detection_rank_cut._regions, returning labels)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    lab = {}
    for fn, us in models:
        sets = [res[u] for u in us]
        if len(us) == 1:
            lab[us[0]] = 0
            continue
        allres = sorted(set().union(*sets))
        idx = {r: i for i, r in enumerate(allres)}
        M = np.zeros((len(us), len(allres)), dtype=np.float32)
        for i, s in enumerate(sets):
            M[i, [idx[r] for r in s]] = 1.0
        sz = M.sum(1)
        oc = (M @ M.T) / np.minimum.outer(sz, sz).clip(min=1.0)
        comp = connected_components(csr_matrix(oc >= thr), directed=False)[1]
        lab.update(zip(us, comp.tolist()))
    return lab


def load():
    base = G.load_base()
    rk = pd.read_csv(RANKS_CSV).set_index("site_uid").reindex(base["site_uid"])
    if rk["n_sites"].isna().any():
        raise AssertionError("site_ranks.csv does not cover the detection base")
    base["model"] = rk["model"].to_numpy()
    base["N_det"] = rk["n_sites"].to_numpy(float)
    base["n_eff"] = rk["neff50"].to_numpy(float)
    n_ref = None
    for key, name in TOOLS:
        t = pd.read_csv(G._tool_table(key), usecols=["site_uid", "dca_rank", "n_unp_pockets",
                                                     "predicted_pockets_total"]).set_index("site_uid")
        t = t.reindex(base["site_uid"])
        rank = pd.to_numeric(t["dca_rank"], errors="coerce").to_numpy(float)
        n = pd.to_numeric(t["n_unp_pockets"], errors="coerce").to_numpy(float)
        if not np.array_equal(rk[f"{name}|r1"].to_numpy(float), rank):
            raise AssertionError(f"{name}: site_ranks.csv rank differs from dca_rank")
        pub = base[name].to_numpy(float)
        ok = np.isfinite(pub)
        tns = (rank >= 1) & (rank <= n)
        if not np.array_equal(tns[ok].astype(float), pub[ok]):
            raise AssertionError(f"{name}: rank <= n does not reproduce the released success_tns")
        base[f"{name}|rank"] = rank
        base[f"{name}|tns"] = np.where(ok, tns, np.nan)
        base[f"{name}|any"] = np.where(ok, rank >= 1, np.nan)
        base[f"{name}|cap3"] = np.where(ok, (rank >= 1) & (rank <= np.minimum(n, CAP)), np.nan)
        base[f"{name}|ncand"] = rk[f"{name}|ncand"].to_numpy(float)
        if n_ref is None:
            n_ref = n
        elif not np.array_equal(n, n_ref):
            raise AssertionError("n_unp_pockets differs between the tools' tables")
    base["N_ref"] = n_ref
    npred = pd.read_csv(G._tool_table("ifsitepred"), usecols=["site_uid", "predicted_pockets_total"]
                        ).set_index("site_uid").reindex(base["site_uid"])["predicted_pockets_total"]
    if not np.array_equal(npred.fillna(0).to_numpy(float), base["IF-SitePred|ncand"].to_numpy(float)):
        raise AssertionError("IF-SitePred candidate count differs between site_ranks.csv and the scored table")
    if base["IF-SitePred|ncand"].max() > CAP:
        raise AssertionError("IF-SitePred returned more than three candidates somewhere")
    print(f"  detection base {len(base):,} sites on {base['model'].nunique():,} models; TNS = (rank <= n) reproduces "
          f"the released success for all three tools")

    geo = pd.read_csv(cfg.AFDB_ST100_DATASET_GEOM_CSV, usecols=["site_uid", "union_chain_resi"], low_memory=False)
    res = {u: frozenset(str(s).split()) for u, s in zip(geo["site_uid"], geo["union_chain_resi"].fillna(""))}
    rec = pd.read_csv(cfg.LBS_RECEPTORS_CSV)
    lab = region_labels([(fn, u.split()) for fn, u in zip(rec["afdb_filename"], rec["site_uids"])], res)
    base["region"] = base["site_uid"].map(lab)
    chk = base.groupby("model")["region"].transform("nunique")
    if not np.array_equal(chk.to_numpy(float), base["n_eff"].to_numpy(float)):
        raise AssertionError("region labels do not reproduce neff50")
    print("  region labels (overlap >= 0.5) reproduce neff50 for every site")

    p = base.copy()
    p["sep"] = p["separation_level_db"].astype(str).str.strip().str.lower()
    p["prob"] = p["prob_level_db"].astype(str).str.strip().str.lower()
    p = p[p["cls"].isin(["apo", "holo"]) & p["union_rg"].notna() & p["site_uniprot"].notna()
          & p["sep"].isin(S.TIERS) & p["prob"].isin(S.TIERS)].reset_index(drop=True)
    na, nh = int((p.cls == "apo").sum()), int((p.cls == "holo").sum())
    if (len(p), na, nh) != (87138, 35124, 52014):
        raise AssertionError(f"analysis set is {len(p)} ({na} / {nh}), expected 87,138 (35,124 / 52,014)")
    print(f"  analysis set {len(p):,} pockets ({na:,} AF-Apo, {nh:,} AF-Holo) on {p['site_uniprot'].nunique():,} accessions")
    return base, p


# ======================================================================================= sections 1-3
def proteins(base):
    g = base.groupby("model", sort=True)
    if (g["N_ref"].nunique() > 1).any():
        raise AssertionError("n_unp_pockets varies within a model")
    pr = g.agg(uniprot=("site_uniprot", "first"), N_ref=("N_ref", "first"), N_det=("N_det", "first"),
               n_eff=("n_eff", "first"), n_apo=("cls", lambda s: int((s == "apo").sum())),
               n_holo=("cls", lambda s: int((s == "holo").sum())), ncand_if=("IF-SitePred|ncand", "first"))
    for name in PRED:
        pr[f"found_{name}"] = g[f"{name}|tns"].sum()
        pr[f"recall_{name}"] = pr[f"found_{name}"] / pr["N_det"]
        reg = base[base[f"{name}|tns"] == 1].groupby("model")["region"].nunique()
        pr[f"regions_found_{name}"] = reg.reindex(pr.index).fillna(0).astype(int)
        pr[f"region_recall_{name}"] = pr[f"regions_found_{name}"] / pr["n_eff"]
    pr["bound_sites"] = np.minimum(1.0, CAP / pr["N_det"])
    pr["bound_regions"] = np.minimum(1.0, CAP / pr["n_eff"])
    pr["split"] = np.select([(pr.n_apo > 0) & (pr.n_holo == 0), (pr.n_holo > 0) & (pr.n_apo == 0),
                             (pr.n_apo > 0) & (pr.n_holo > 0)], ["only AF-Apo", "only AF-Holo", "mixed"],
                            "no class (ties)")
    return pr.reset_index()


def sections_1_3(base, p, pr, rows):
    def rec(section, quantity, stratum, value, denom=None):
        rows.append({"section": section, "quantity": quantity, "stratum": stratum, "value": value, "of": denom})

    header("1. PROTEINS AND SITES WITH N > 3")
    print(f"  proteins (models): {len(pr):,}. N_ref median {pr.N_ref.median():.0f}, N_det median "
          f"{pr.N_det.median():.0f}, n_eff median {pr.n_eff.median():.0f}")
    print(f"  {'stratum':18s}{'proteins':>10s}{'N_ref>3':>16s}{'N_det>3':>16s}{'n_eff>3':>16s}")
    for s in ["all", "only AF-Apo", "only AF-Holo", "mixed", "no class (ties)"]:
        q = pr if s == "all" else pr[pr.split == s]
        if not len(q):
            continue
        cells = []
        for c in ("N_ref", "N_det", "n_eff"):
            k = int((q[c] > CAP).sum())
            cells.append(f"{k:,} ({100 * k / len(q):.1f}%)")
            rec(1, f"proteins_{c}_gt3", s, k, len(q))
        print(f"  {s:18s}{len(q):>10,}" + "".join(f"{c:>16s}" for c in cells))
    print("\n  sites of the analysis set whose protein has N_ref > 3 / n_eff > 3, by class and separation tier:")
    print(f"  {'tier':11s}{'AF-Apo N_ref>3':>16s}{'AF-Holo':>9s}{'AF-Apo n_eff>3':>16s}{'AF-Holo':>9s}{'sites':>9s}")
    for t in ["all"] + S.TIERS:
        q = p if t == "all" else p[p.sep == t]
        a, h = q[q.cls == "apo"], q[q.cls == "holo"]
        vals = [pct(a.N_ref > CAP), pct(h.N_ref > CAP), pct(a.n_eff > CAP), pct(h.n_eff > CAP)]
        for lab, v in zip(["apo_Nref_gt3", "holo_Nref_gt3", "apo_neff_gt3", "holo_neff_gt3"], vals):
            rec(1, f"sites_pct_{lab}", t, v, len(q))
        print(f"  {t:11s}{vals[0]:>15.1f}%{vals[1]:>8.1f}%{vals[2]:>15.1f}%{vals[3]:>8.1f}%{len(q):>9,}")

    header("2. IF-SITEPRED RECALL AGAINST THE 3/N CEILING (proteins with N > 3)")
    for c, rc, bd, what in (("N_det", "recall", "bound_sites", "site recall (found / N_det), ceiling 3 / N_det"),
                            ("n_eff", "region_recall", "bound_regions",
                             "region recall (regions found / n_eff), ceiling 3 / n_eff")):
        q = pr[pr[c] > CAP]
        print(f"\n  {what}; {len(q):,} proteins with {c} > 3")
        print(f"    {'tool':12s}{'mean ceiling':>13s}{'mean recall':>13s}{'at ceiling':>12s}{'above it':>10s}"
              f"{'recall / ceiling':>18s}")
        for name in PRED:
            r, b = q[f"{rc}_{name}"], q[bd]
            at = pct(np.isclose(r, b)) if name == "IF-SitePred" else np.nan
            above = pct(r > b + 1e-12)
            print(f"    {name:12s}{100 * b.mean():>12.1f}%{100 * r.mean():>12.1f}%"
                  f"{(f'{at:.1f}%' if np.isfinite(at) else '--'):>12s}{above:>9.1f}%{r.mean() / b.mean():>18.2f}")
            rec(2, f"{rc}_mean_{name}", f"{c}>3", 100 * r.mean(), len(q))
            rec(2, f"{rc}_above_ceiling_pct_{name}", f"{c}>3", above, len(q))
            if name == "IF-SitePred":
                rec(2, f"{rc}_at_ceiling_pct_{name}", f"{c}>3", at, len(q))
        rec(2, f"{bd}_mean", f"{c}>3", 100 * q[bd].mean(), len(q))
    print("\n  (P2Rank and fpocket are held to no 3-slot ceiling; their row shows where they sit against it.)")

    header("3. PROTEINS WHERE IF-SITEPRED RETURNED FEWER THAN 3 SITES, OR FEWER THAN N")
    nc = pr.ncand_if
    print("  candidates returned per protein: " + ", ".join(
        f"{k}: {int((nc == k).sum()):,}" for k in range(CAP + 1)) + f" (of {len(pr):,})")
    for k in range(CAP + 1):
        rec(3, f"proteins_ncand_eq{k}", "all", int((nc == k).sum()), len(pr))
    for c in ("N_ref", "N_det", "n_eff"):
        k1, k2 = int((nc < CAP).sum()), int((nc < pr[c]).sum())
        k3 = int(((nc < pr[c]) & (nc == CAP)).sum())
        print(f"  fewer than 3: {k1:,} ({100 * k1 / len(pr):.1f}%);  fewer than {c}: {k2:,} ({100 * k2 / len(pr):.1f}%), "
              f"of which {k3:,} returned the full 3")
        rec(3, f"proteins_ncand_lt_{c}", "all", k2, len(pr))
    rec(3, "proteins_ncand_lt3", "all", int((nc < CAP).sum()), len(pr))
    q = p.assign(nc=p["IF-SitePred|ncand"])
    for cl in ("apo", "holo"):
        m = q.cls == cl
        print(f"  analysis-set AF-{cl.capitalize()} sites on a protein with < 3 IF-SitePred candidates: "
              f"{pct(q.nc[m] < CAP):.1f}% (none at all: {pct(q.nc[m] == 0):.1f}%)")
        rec(3, f"sites_pct_ncand_lt3_{cl}", "analysis set", pct(q.nc[m] < CAP), int(m.sum()))
        rec(3, f"sites_pct_ncand_eq0_{cl}", "analysis set", pct(q.nc[m] == 0), int(m.sum()))
    lt = q[q.nc < CAP]
    print(f"  IF-SitePred TNS on those sites: AF-Apo {pct(lt.loc[lt.cls == 'apo', 'IF-SitePred|tns']):.1f}%, "
          f"AF-Holo {pct(lt.loc[lt.cls == 'holo', 'IF-SitePred|tns']):.1f}% (n = {len(lt):,})")


# ========================================================================================== section 4
def section_4(p, rows_out):
    header("4. SUCCESS RATE AND CLASS GAP BY SUBSET AND CRITERION (the 87,138-pocket analysis set)")
    subsets = [("all", np.ones(len(p), bool), "tns"), ("N_ref<=3", (p.N_ref <= CAP).to_numpy(), "tns"),
               ("n_eff<=3", (p.n_eff <= CAP).to_numpy(), "tns"), ("N_ref>3", (p.N_ref > CAP).to_numpy(), "tns"),
               ("n_eff>3", (p.n_eff > CAP).to_numpy(), "tns"), ("all, cap3", np.ones(len(p), bool), "cap3"),
               ("n_eff>3, cap3", (p.n_eff > CAP).to_numpy(), "cap3")]
    # one Gap per (tool, criterion) on the same frame, so the resamples coincide (same codes, same seed)
    gaps = {(n, c): S.Gap(p.assign(y=p[f"{n}|{c}"]), "y") for n in PRED for c in ("tns", "cap3")}
    g0 = gaps[(PRED[0], "tns")]
    if len(g0.d) != len(p):
        raise AssertionError("Gap dropped rows of the analysis set")
    for g in gaps.values():
        if not np.array_equal(g.codes, g0.codes):
            raise AssertionError("resamples would differ between tools")
    ix_all = g0._indices()

    # gate: Table S3, size-standardized TNS gap by separation tier
    ref = pd.read_csv(REF_S3)
    ref = ref[ref.metric == "separation"].set_index(["task", "tier"])["gap_std"]
    cell = pd.Categorical(p["sep"], categories=S.TIERS).codes.astype(int)
    mx = 0.0
    for n in PRED:
        pt = gaps[(n, "tns")].point(cell, len(S.TIERS))
        mx = max(mx, max(abs(pt[i] - ref[(n, t)]) for i, t in enumerate(S.TIERS)))
    if mx > 1e-9:
        raise AssertionError(f"size-standardized gaps by tier differ from {REF_S3.name} (max {mx:.2e})")
    print(f"  [gate] {REF_S3.name}: size-standardized TNS gaps by separation tier reproduced (max |diff| {mx:.1e})")

    y = {(n, c): p[f"{n}|{c}"].to_numpy(float) for n in PRED for c in ("tns", "cap3")}
    ish = (p.cls == "holo").to_numpy()

    def raw(v, m, h):
        return (100 * v[m].mean(), 100 * v[m & ~h].mean(), 100 * v[m & h].mean())

    out = {}
    for lab, m, crit in subsets:
        cellm = np.where(m, 0, -1)
        for n in PRED:
            v = y[(n, crit)]
            pt = raw(v, m, ish)
            bt = np.array([raw(v[ix], m[ix], ish[ix]) for ix in ix_all])
            std = gaps[(n, crit)].point(cellm, 1)[0]
            std_b = gaps[(n, crit)].boot(cellm, 1)[:, 0]
            out[(lab, n)] = dict(pooled=pt[0], apo=pt[1], holo=pt[2], gap=pt[2] - pt[1], std=std,
                                 b_pooled=bt[:, 0], b_gap=bt[:, 2] - bt[:, 1], b_std=std_b,
                                 n=int(m.sum()), n_apo=int((m & ~ish).sum()), n_holo=int((m & ish).sum()),
                                 prot=p.loc[m, "site_uniprot"].nunique())

    def ci(b):
        lo, hi = S.ci(b)
        return f"[{lo:+.1f}, {hi:+.1f}]"

    for lab, m, crit in subsets:
        o0 = out[(lab, PRED[0])]
        print(f"\n  {lab}: {o0['n']:,} pockets ({o0['n_apo']:,} AF-Apo / {o0['n_holo']:,} AF-Holo), "
              f"{o0['prot']:,} accessions" + ("; every tool at rank <= min(n, 3)" if crit == "cap3" else ""))
        print(f"    {'tool':12s}{'pooled':>8s}{'AF-Apo':>8s}{'AF-Holo':>9s}{'raw gap':>22s}{'size-std gap':>22s}"
              f"{'IF-SitePred minus tool: pooled':>32s}{'size-std gap':>22s}")
        ranks = sorted(PRED, key=lambda n: -out[(lab, n)]["pooled"])
        for n in PRED:
            o = out[(lab, n)]
            d = out[(lab, "IF-SitePred")]
            diff = "" if n == "IF-SitePred" else \
                f"{d['pooled'] - o['pooled']:>+18.1f} {ci(d['b_pooled'] - o['b_pooled']):>13s}" \
                f"{d['std'] - o['std']:>+9.1f} {ci(d['b_std'] - o['b_std']):>12s}"
            print(f"    {n:12s}{o['pooled']:>8.1f}{o['apo']:>8.1f}{o['holo']:>9.1f}"
                  f"{o['gap']:>+9.1f} {ci(o['b_gap']):>12s}{o['std']:>+9.1f} {ci(o['b_std']):>12s}{diff}")
            lo, hi = S.ci(o["b_std"])
            rows_out.append({"subset": lab, "criterion": crit, "tool": n, "n_pockets": o["n"],
                             "n_apo": o["n_apo"], "n_holo": o["n_holo"], "n_accessions": o["prot"],
                             "rate_pooled": o["pooled"], "rate_apo": o["apo"], "rate_holo": o["holo"],
                             "gap_raw": o["gap"], "gap_raw_lo": S.ci(o["b_gap"])[0], "gap_raw_hi": S.ci(o["b_gap"])[1],
                             "gap_std": o["std"], "gap_std_lo": lo, "gap_std_hi": hi,
                             "ifsitepred_minus_tool_pooled": d["pooled"] - o["pooled"],
                             "ifsitepred_minus_tool_gap_raw": d["gap"] - o["gap"],
                             "ifsitepred_minus_tool_gap_raw_lo": S.ci(d["b_gap"] - o["b_gap"])[0],
                             "ifsitepred_minus_tool_gap_raw_hi": S.ci(d["b_gap"] - o["b_gap"])[1],
                             "ifsitepred_minus_tool_gap_std": d["std"] - o["std"],
                             "ifsitepred_minus_tool_gap_std_lo": S.ci(d["b_std"] - o["b_std"])[0],
                             "ifsitepred_minus_tool_gap_std_hi": S.ci(d["b_std"] - o["b_std"])[1],
                             "rank_by_pooled_rate": ranks.index(n) + 1,
                             "rank_by_gap_std": sorted(PRED, key=lambda k: -out[(lab, k)]["std"]).index(n) + 1})
        d = out[(lab, "IF-SitePred")]
        print("    IF-SitePred minus tool, raw gap: " + ";  ".join(
            f"{n} {d['gap'] - out[(lab, n)]['gap']:+.1f} {ci(d['b_gap'] - out[(lab, n)]['b_gap'])}" for n in PRED[:2]))
        print("    ranking by pooled rate: " + " > ".join(ranks) + ";  by size-std gap: " +
              " > ".join(sorted(PRED, key=lambda k: -out[(lab, k)]["std"])))

    print("\n  what ranks 4..n add for P2Rank and fpocket (TNS minus cap3, points; a negative gap change means the")
    print("  3-slot budget WIDENS that tool's class gap):")
    for sa, sc in (("all", "all, cap3"), ("n_eff>3", "n_eff>3, cap3")):
        for n in PRED[:2]:
            a, c3 = out[(sa, n)], out[(sc, n)]
            print(f"    {sa:8s} {n:10s} pooled {a['pooled'] - c3['pooled']:+5.1f}  AF-Apo {a['apo'] - c3['apo']:+5.1f}  "
                  f"AF-Holo {a['holo'] - c3['holo']:+5.1f}  size-std gap {a['std'] - c3['std']:+.1f} "
                  f"{ci(a['b_std'] - c3['b_std'])}")
    print("  IF-SitePred's any-rank rate equals its TNS rate wherever N_ref >= 3; it differs only on "
          f"{int((p.N_ref < CAP).sum()):,} sites with N_ref < 3, "
          f"where {int(((p['IF-SitePred|rank'] > p.N_ref) & (p.N_ref < CAP)).sum()):,} hits sit beyond rank n")


def main():
    OUTDIR.mkdir(exist_ok=True)
    print(f"ifsitepred_cap.py   [model set 'afdbv4']")
    header("LOAD")
    base, p = load()
    pr = proteins(base)
    rows = []
    sections_1_3(base, p, pr, rows)
    rates = []
    section_4(p, rates)
    pr.to_csv(OUTDIR / "proteins.csv", index=False)
    pd.DataFrame(rows).to_csv(OUTDIR / "counts.csv", index=False)
    pd.DataFrame(rates).to_csv(OUTDIR / "subset_rates.csv", index=False)
    print(f"\n  [OK] -> {OUTDIR.name}/proteins.csv, counts.csv, subset_rates.csv")


if __name__ == "__main__":
    main()
