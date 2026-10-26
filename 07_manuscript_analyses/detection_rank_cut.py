# -*- coding: utf-8 -*-
"""Which detection criterion can carry a methodological comparison between the three predictors?
(Supplementary Table S12, upper part: detection success under criteria that count each binding site once.)

Detection success in the manuscript is DCA4-TNS: a predicted centre within 4 A of a reference pocket atom,
ranked within n, where n is the number of ST_100 reference sites of the protein (`n_unp_pockets`; the scorer
is 06_lbs_generality/score/compute_lbs_success.py). Two fpocket statements rest on it:

  A. Gap level against gradient. fpocket has the smallest pooled class gap but the steepest separation
     gradient under DCA4-TNS (Supplementary Table S3).
  B. Study effort. fpocket is the one task in Supplementary Table S11 whose success follows the total amount of
     experimental evidence at fixed ratio and size (partial rho 0.198).

Every rank criterion carries a budget, and a budget can bias a class contrast in two opposite ways:
  * too loose: n counts overlapping ST_100 sites (carbonic anhydrase II: 594 sites, 19 distinct regions), so the
    TNS budget grows with study effort and differs by class for reasons that are not the pocket's;
  * too tight: a fixed top-k lets the protein's OTHER true sites use up the k slots, which penalizes proteins
    with many sites (these sit at low separation) -- a fixed top-3 masks performance there.
So the criteria here are, per tool:
  tns   rank <= n (published)
  teff  rank <= n_eff, n_eff = distinct reference regions of the model (ST_100 sites merged when their residue
        sets overlap by >= 0.5 of the smaller; 0.25 and 0.75 as sensitivity): Chen/Krivak top-n with n counted
        once per site
  cf1, cf3  competition-free top-1 / top-3: the hit counts if fewer than 1 / 3 SPURIOUS candidates (hitting no
        reference site of the model) rank above it; other true sites never use up the budget
  top1, top3  fixed cuts, kept as the comparison the first version of this analysis leaned on
  any   any rank: detection alone, no budget (the tool's ranking plays no part)
`dca_rank` is the rank, in the tool's own order, of the first candidate within 4 A of the reference atoms. The
per-candidate hits against every reference site of the model are rebuilt from the normalized predictions and
the ground-truth atom store (mode `ranks`); the rebuilt first-hit rank must equal `dca_rank` for every site, and
DCA4-TNS recomputed from it must equal the released `success_tns` exactly.

Modes
  ranks      builds the per-site table (n_eff; per tool first-hit rank, candidates, spurious and other-site
             candidates above the hit) into <manuscript_analyses_dir>/rank_cut_afdbv4/site_ranks.csv. ~2 min.
             The other modes build it if it is missing (ifsitepred_cap.py reads it too).
  mechanism  what n and n_eff are, how they move with class, separation and study effort, how often each tool's
             cut can bind, how each tool ranks its hits, and the raw class gaps per criterion. About a minute.
  tiers      (ii) Figure 4's estimator (separation_shape.Gap: size-standardized by union-Rg deciles, cluster
             bootstrap over UniProt, B = 400, seed 20260908) per predictor and criterion by separation tier,
             on the 87,091 pockets that carry both the released and the size-balanced label, with one set of
             resamples for every predictor, criterion and label, so every contrast is paired. Gates: the
             TNS rows reproduce balanced_downstream_gaps.csv (Supplementary Table S10c-d) exactly, and on
             the 87,138-pocket set the per-tier point estimates of density_separation_gaps (Table S3).
             About five minutes.
  s11        (i) SI Table S11 under each criterion, on evidence_amount_vs_ratio.py's own loaders, gates,
             common mask, one-rank-pass partial Spearman and paired cluster bootstrap (B = 400, seed
             20260922). Every criterion's outcome is ranked in the same pass, so the criteria share resamples
             and their differences are paired. Docking has no rank cut and enters every criterion unchanged.
             Gate: the TNS rows reproduce evidence_amount_vs_ratio_afdbv4.csv exactly. About 30 minutes.
  all        ranks (if missing), mechanism, tiers, s11.

Usage (base conda python; the `plotting` env lacks the pandas extras):

    $env:AHOJ_AF_MODEL_SET = "afdbv4"        # PowerShell; bash: export AHOJ_AF_MODEL_SET=afdbv4
    python detection_rank_cut.py all > detection_rank_cut_afdbv4.out.txt

Writes detection_rank_cut_{mechanism,tiers,s11}_afdbv4.csv into <manuscript_analyses_dir> (no suffix on the July
v6 set); Supplementary Table S12 is drawn from the tiers and s11 tables. Reads from <manuscript_analyses_dir> the
gate tables of the other analyses: density_separation_gaps_afdbv4.csv, evidence_amount_vs_ratio_afdbv4.csv and,
under afdbv4, afdb_v4_realign/balanced_sites_csv/balanced_downstream_gaps.csv (balanced_classification.py `sites
--csv-dir` run with the switch; without it, balanced_downstream_gaps.csv).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import ma_paths as P

HERE = Path(__file__).resolve().parent
MODEL_SET = os.environ.get("AHOJ_AF_MODEL_SET", "").strip().lower()
if MODEL_SET not in ("", "afdbv4"):
    sys.exit(f"AHOJ_AF_MODEL_SET={MODEL_SET!r}: expected unset or 'afdbv4'")
SFX = "_afdbv4" if MODEL_SET == "afdbv4" else ""
sys.path.insert(0, str(HERE))

import evidence_amount_vs_ratio as E                  # noqa: E402  SI Table S11's machinery
import separation_shape as S                          # noqa: E402  Figure 4's estimator
import balanced_classification as BC                  # noqa: E402  the size-balanced site labels
from tool_gap_reconciliation import log_or, rank_share  # noqa: E402

qg = E.qg
G = E._load("fig6a_quality_gates", "06_lbs_generality/analysis/analyze_fig6a_quality_gates.py")
cfg = P.cfg                                           # common/config.py

if cfg.MODEL_SET != MODEL_SET or S.MODEL_SFX != SFX:
    sys.exit("the model-set switch disagrees between the public repo and separation_shape.py")

CRITS = ["tns", "teff", "cf1", "cf3", "top1", "top3", "any"]
CRIT_LABEL = {"tns": "DCA4-TNS (rank <= n, published)", "teff": "top-n_eff (n = distinct regions)",
              "cf1": "competition-free top-1", "cf3": "competition-free top-3", "top1": "fixed top-1",
              "top3": "fixed top-3", "any": "any rank (detection only)"}
SHORT = {"tns": "DCA4-TNS", "teff": "top-n_eff", "cf1": "cf top-1", "cf3": "cf top-3", "top1": "top-1",
         "top3": "top-3", "any": "any rank"}
NEFF_THR = {"neff25": 0.25, "neff50": 0.5, "neff75": 0.75}
TOOLS = [("p2rank", "P2Rank"), ("fpocket", "fpocket"), ("ifsitepred", "IF-SitePred")]
PRED = list(S.PRED)
TASKS = PRED + ["consensus"]
PAIRS = [("fpocket", "IF-SitePred"), ("fpocket", "P2Rank"), ("IF-SitePred", "P2Rank")]
LABELS = [("released", "cls_rel"), ("size-balanced", "cls_bal")]

WORK = Path(P.WD)
REF_S10 = (Path(P.OUT) / "afdb_v4_realign" / "balanced_sites_csv" / "balanced_downstream_gaps.csv"
           if SFX else Path(P.out("balanced_downstream_gaps.csv")))
REF_S3 = Path(P.out(f"density_separation_gaps{SFX}.csv"))
REF_S11 = Path(P.out(f"evidence_amount_vs_ratio{SFX}.csv"))
OUT = {m: Path(P.out(f"detection_rank_cut_{m}{SFX}.csv")) for m in ("mechanism", "tiers", "s11")}
RANKS_CSV = Path(P.OUT) / f"rank_cut{SFX}" / "site_ranks.csv"


# ============================================================================================ helpers
def cut_outcomes(rank, n, neff, fpa, defined):
    """Success under each criterion from the rank of the first hit, the protein's n and n_eff, and the number
    of spurious candidates ranked above the hit; NaN where TNS is not scorable."""
    ok = rank >= 1
    outs = {"tns": ok & (rank <= n), "teff": ok & (rank <= neff), "cf1": ok & (fpa <= 0), "cf3": ok & (fpa <= 2),
            "top1": ok & (rank <= 1), "top3": ok & (rank <= 3), "any": ok}
    return {k: np.where(defined, v, np.nan).astype(float) for k, v in outs.items()}


# ====================================================================================== site ranks
def _regions(sets, thr):
    """Connected components of a model's sites, two sites joined when their residue sets overlap by at least
    `thr` of the smaller one (overlap coefficient)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    k = len(sets)
    if k == 1:
        return 1
    allres = sorted(set().union(*sets))
    idx = {r: i for i, r in enumerate(allres)}
    M = np.zeros((k, len(allres)), dtype=np.float32)
    for i, s in enumerate(sets):
        M[i, [idx[r] for r in s]] = 1.0
    inter = M @ M.T
    sz = M.sum(1)
    oc = inter / np.minimum.outer(sz, sz).clip(min=1.0)
    return connected_components(csr_matrix(oc >= thr), directed=False)[0]


def site_ranks(rebuild=False):
    """Per site: n_eff at three overlap thresholds, and per tool the first-hit rank, the number of candidates,
    and how many candidates rank above the hit that hit no reference site of the model (spurious) or another
    one (other-site). Cached in <manuscript_analyses_dir>; regenerable in about two minutes."""
    if RANKS_CSV.is_file() and not rebuild:
        return pd.read_csv(RANKS_CSV)
    from scipy.spatial import cKDTree
    header("RANKS -- per-candidate hits against every reference site of each model")
    geo = pd.read_csv(cfg.AFDB_ST100_DATASET_GEOM_CSV, usecols=["site_uid", "union_chain_resi"], low_memory=False)
    res = {u: frozenset(str(s).split()) for u, s in zip(geo["site_uid"], geo["union_chain_resi"].fillna(""))}
    rec = pd.read_csv(cfg.LBS_RECEPTORS_CSV)
    models = [(fn, uids.split()) for fn, uids in zip(rec["afdb_filename"], rec["site_uids"])]
    rows = {}
    for fn, us in models:
        sets = [res[u] for u in us]
        comp = {c: _regions(sets, thr) for c, thr in NEFF_THR.items()}
        for u in us:
            rows[u] = {"site_uid": u, "model": fn, "n_sites": len(us), **comp}
    print(f"  {len(models):,} models, {len(rows):,} sites; residue sets from {Path(cfg.AFDB_ST100_DATASET_GEOM_CSV).name}")
    z = np.load(cfg.LBS_GROUND_TRUTH_NPZ, allow_pickle=True)
    xyz, starts, counts = z["coords"], z["starts"], z["counts"]
    gt = {str(u): xyz[int(s):int(s) + int(c)] for u, s, c in zip(z["site_uids"], starts, counts)}
    for key, name in TOOLS:
        t0 = time.time()
        pr = pd.read_csv(cfg.lbs_tool_dir(key) / "predictions.csv",
                         usecols=["site_uid", "pocket_rank", "center_x", "center_y", "center_z"])
        groups = {u: grp.sort_values("pocket_rank") for u, grp in pr.groupby("site_uid", sort=False)}
        ref = pd.read_csv(G._tool_table(key), usecols=["site_uid", "dca_rank"]).set_index("site_uid")["dca_rank"]
        n_lists, n_same = 0, 0
        for fn, us in models:
            lists = [groups.get(u) for u in us]
            present = [c for c in lists if c is not None]
            if not present:
                for u in us:
                    rows[u].update({f"{name}|r1": -1, f"{name}|ncand": 0, f"{name}|fpa": 0, f"{name}|ota": 0})
                continue
            cand = present[0]
            C = cand[["center_x", "center_y", "center_z"]].to_numpy(float)
            ranks = cand["pocket_rank"].to_numpy()
            for c in present[1:]:
                n_lists += 1
                n_same += int(len(c) == len(cand) and np.allclose(
                    c[["center_x", "center_y", "center_z"]].to_numpy(float), C, equal_nan=True))
            ok = np.isfinite(C).all(1)
            tree, ridx = cKDTree(C[ok]), np.flatnonzero(ok)
            hits, anyhit = {}, np.zeros(len(C), bool)
            for u in us:
                h = np.zeros(len(C), bool)
                a = gt.get(u)
                if a is not None and len(a):
                    for lst in tree.query_ball_point(a, r=4.0):
                        h[ridx[lst]] = True
                hits[u] = h
                anyhit |= h
            for u in us:
                h = hits[u]
                if not h.any():
                    rows[u].update({f"{name}|r1": -1, f"{name}|ncand": len(C), f"{name}|fpa": 0, f"{name}|ota": 0})
                    continue
                r1 = ranks[h].min()
                above = ranks < r1
                rows[u].update({f"{name}|r1": int(r1), f"{name}|ncand": len(C),
                                f"{name}|fpa": int((above & ~anyhit).sum()), f"{name}|ota": int((above & anyhit).sum())})
        if n_same != n_lists:
            raise AssertionError(f"{name}: {n_lists - n_same} site lists differ from their model's first list")
        r1 = pd.Series({u: r[f"{name}|r1"] for u, r in rows.items()})
        mism = int((r1 != ref.reindex(r1.index)).sum())
        if mism:
            raise AssertionError(f"{name}: rebuilt first-hit rank differs from dca_rank for {mism} sites")
        print(f"  {name}: candidate lists identical across a model's sites ({n_lists:,} checked); rebuilt first-hit "
              f"rank equals the released dca_rank for all {len(r1):,} sites ({time.time() - t0:.0f} s)")
    df = pd.DataFrame(list(rows.values()))
    RANKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RANKS_CSV, index=False)
    print(f"  [OK] -> <manuscript_analyses_dir>/{RANKS_CSV.relative_to(Path(P.OUT)).as_posix()}")
    return df


def ci(b):
    return S.ci(b)


def tri(pt, b, dp=1):
    """`+x.x [+lo, +hi]`: the triple format the provenance checker matches."""
    lo, hi = ci(b)
    return f"{pt:+.{dp}f} [{lo:+.{dp}f}, {hi:+.{dp}f}]"


def pct(x):
    x = np.asarray(x, dtype=float)
    return 100.0 * float(np.nanmean(x)) if x.size else np.nan


def header(text):
    print("\n" + "=" * 110)
    print(text)
    print("=" * 110)


# ======================================================================================= data frames
def detection_base():
    """The public repo's detection base (analyze_fig6a_quality_gates.load_base), with each tool's rank of
    first hit, candidate count, spurious and other-site candidates above the hit, and the protein's n and n_eff
    attached, and every criterion per tool."""
    base = G.load_base()
    rk = site_ranks().set_index("site_uid").reindex(base["site_uid"])
    if rk["n_sites"].isna().any():
        raise AssertionError("site_ranks.csv does not cover the detection base")
    for c in NEFF_THR:
        base[c] = rk[c].to_numpy(float)
    base["n_sites"] = rk["n_sites"].to_numpy(float)
    n_ref = None
    for key, name in TOOLS:
        t = pd.read_csv(G._tool_table(key), low_memory=False,
                        usecols=["site_uid", "dca_rank", "n_unp_pockets", "predicted_pockets_total"])
        if not t["site_uid"].is_unique:
            raise AssertionError(f"{key}: duplicate site_uid")
        t = t.set_index("site_uid").reindex(base["site_uid"])
        rank = pd.to_numeric(t["dca_rank"], errors="coerce").to_numpy(float)
        n = pd.to_numeric(t["n_unp_pockets"], errors="coerce").to_numpy(float)
        if not np.array_equal(rk[f"{name}|r1"].to_numpy(float), rank):
            raise AssertionError(f"{name}: site_ranks.csv first-hit rank differs from dca_rank")
        fpa = rk[f"{name}|fpa"].to_numpy(float)
        pub = base[name].to_numpy(float)
        defined = np.isfinite(pub)
        outs = cut_outcomes(rank, n, base["neff50"].to_numpy(float), fpa, defined)
        if not np.array_equal(outs["tns"][defined], pub[defined]):
            raise AssertionError(f"{name}: rank <= n does not reproduce the released success_tns")
        for k, v in outs.items():
            base[f"{name}|{k}"] = v
        for c in ("neff25", "neff75"):          # the n_eff sensitivity
            base[f"{name}|teff{c[-2:]}"] = np.where(defined, (rank >= 1) & (rank <= base[c].to_numpy(float)),
                                                    np.nan).astype(float)
        hit = rank >= 1
        base[f"{name}|rank"] = rank
        base[f"{name}|fpa"] = np.where(hit, fpa, np.nan)
        base[f"{name}|ota"] = np.where(hit, rk[f"{name}|ota"].to_numpy(float), np.nan)
        # ranking given detection: the hit is ahead of every spurious candidate / of all but two
        base[f"{name}|rk1"] = np.where(hit, (fpa <= 0).astype(float), np.nan)
        base[f"{name}|rk3"] = np.where(hit, (fpa <= 2).astype(float), np.nan)
        base[f"{name}|npred"] = pd.to_numeric(t["predicted_pockets_total"], errors="coerce").to_numpy(float)
        if n_ref is None:
            n_ref = n
        elif not np.array_equal(n_ref, n):
            raise AssertionError("n_unp_pockets differs between the tools' tables")
    base["n_ref"] = n_ref
    for k in ("teff25", "teff75"):
        base[f"consensus|{k}"] = base[[f"{p}|{k}" for p in PRED]].mean(axis=1)
    for k in CRITS:
        base[f"consensus|{k}"] = base[[f"{p}|{k}" for p in PRED]].mean(axis=1)
    if not np.allclose(base["consensus|tns"], base["consensus"]):
        raise AssertionError("consensus of the recomputed TNS differs from load_base()'s")
    tot = pd.to_numeric(base["n_apo_avg"], errors="coerce") + pd.to_numeric(base["n_holo_avg"], errors="coerce")
    base["log10_total"] = np.log10(tot.where(tot > 0))
    return base


def lbs_frame(base):
    """load_base() restricted exactly as separation_shape.load_lbs() restricts it (classified, sized,
    tiered), in the same row order -- asserted, because the cluster bootstrap depends on the order."""
    p = base.copy()
    p["sep"] = p["separation_level_db"].astype(str).str.strip().str.lower()
    p["prob"] = p["prob_level_db"].astype(str).str.strip().str.lower()
    p = p[p["cls"].isin(["apo", "holo"]) & p["union_rg"].notna() & p["site_uniprot"].notna()]
    p = p[p["sep"].isin(S.TIERS) & p["prob"].isin(S.TIERS)].copy()
    ref = S.load_lbs()
    if not np.array_equal(p["site_uid"].to_numpy(), ref["site_uid"].to_numpy()):
        raise AssertionError("row set or order differs from separation_shape.load_lbs()")
    for c in TASKS:
        if not np.allclose(p[c].to_numpy(float), ref[c].to_numpy(float), equal_nan=True):
            raise AssertionError(f"{c}: TNS differs from separation_shape.load_lbs()")
    if not np.array_equal(p["site_uid"].to_numpy(), ref["unit_uid"].to_numpy()):
        raise AssertionError("site_uid is not the unit_uid the balanced labels join on")
    p["unit_uid"] = p["site_uid"]
    return p.reset_index(drop=True)


def two_label_frame(lbs):
    """balanced_classification.py `sites` step 3 on this frame: the ST_100 size-balanced class is the member
    mean of the per-entry balanced P(holo); the pockets that tie under it are dropped. Row order is the
    order of separation_shape.load_lbs(), as there, so the resamples are the published ones."""
    d = BC._entries(BC.ENTRIES_CSV)
    u = pd.read_csv(BC.UNI, usecols=["threshold", "unit_uid", "job_names"], low_memory=False)
    u = u[u["threshold"] == "ST_100"]
    ex = u[["unit_uid", "job_names"]].assign(job_name=u["job_names"].str.split(";")).explode("job_name")
    ex = ex.drop(columns="job_names").merge(d[["job_name", "full_p_holo", "bal_p_holo"]], on="job_name",
                                            how="left")
    agg = ex.groupby("unit_uid", sort=False).agg(
        n_mem=("job_name", "size"), n_ok=("bal_p_holo", "count"), full_p=("full_p_holo", "mean"),
        bal_p=("bal_p_holo", "mean")).reset_index()
    agg["covered"] = agg["n_ok"] == agg["n_mem"]
    agg["bal_cls"] = np.where(agg["covered"], BC._cls(agg["bal_p"]), "")
    x = lbs.merge(agg[["unit_uid", "full_p", "bal_cls", "covered"]], on="unit_uid", how="left")
    fid = (x["full_p"].round(3) - x["pocket_p_holo"]).abs() < 0.0015
    keep = x["bal_cls"].isin(["apo", "holo"])
    print(f"  two-label frame: {len(x):,} detection pockets; site P(holo) rebuilt from the members equals the "
          f"released value for {pct(fid):.2f}%; definite under the size-balanced label {int(keep.sum()):,} "
          f"(balanced ties {int((x['bal_cls'] == 'tie').sum()):,})")
    x = x[keep].copy()
    x["cls_rel"], x["cls_bal"] = x["cls"], x["bal_cls"]
    return x.reset_index(drop=True)


# ========================================================================================= mechanism
def mechanism(base, lbs):
    header("MECHANISM -- what the rank cut n is, and where it binds")
    rows = []

    def rec(quantity, tool, cls, stratum, value):
        rows.append({"quantity": quantity, "tool": tool, "cls": cls, "stratum": stratum, "value": value})

    print(f"  scored detection pockets (load_base): {len(base):,}; classified and tiered (the analysis set): "
          f"{len(lbs):,} ({int((lbs.cls == 'apo').sum()):,} AF-Apo, {int((lbs.cls == 'holo').sum()):,} AF-Holo), "
          f"{lbs.site_uniprot.nunique():,} UniProt accessions")
    n = lbs["n_ref"]
    print("\n  n = number of ST_100 reference sites of the accession (the TNS rank cut):")
    for cl in ("apo", "holo"):
        m = lbs.cls == cl
        print(f"    AF-{cl.capitalize():4s}: median {n[m].median():.0f}, mean {n[m].mean():.2f}, "
              f"n = 1 for {pct(n[m] == 1):.1f}%")
        rec("n_median", "all", cl, "all", float(n[m].median()))
        rec("n_mean", "all", cl, "all", float(n[m].mean()))
        rec("n_eq1_pct", "all", cl, "all", pct(n[m] == 1))
    r_tot = spearmanr(lbs["log10_total"], n, nan_policy="omit").correlation
    r_sep = spearmanr(lbs["pocket_separation_pair"], n, nan_policy="omit").correlation
    ratio = np.log2((lbs["n_holo_avg"] + 1.0) / lbs["n_apo_avg"].replace(0, np.nan))
    r_rat = spearmanr(ratio, n, nan_policy="omit").correlation
    r_rt = spearmanr(ratio, lbs["log10_total"], nan_policy="omit").correlation
    print(f"  Spearman rho(log10 total evidence, n) = {r_tot:+.3f}; rho(separation, n) = {r_sep:+.3f}; "
          f"rho(evidence ratio, n) = {r_rat:+.3f}; rho(evidence ratio, log10 total) = {r_rt:+.3f}")
    rec("rho_total_n", "all", "all", "all", r_tot)
    rec("rho_sep_n", "all", "all", "all", r_sep)
    rec("rho_ratio_n", "all", "all", "all", r_rat)
    rec("rho_ratio_total", "all", "all", "all", r_rt)

    print("\n  median n by separation tier and class:")
    print(f"    {'tier':10s}{'AF-Apo':>8s}{'AF-Holo':>9s}{'pockets':>10s}")
    for t in S.TIERS:
        s = lbs[lbs.sep == t]
        a, h = s.loc[s.cls == "apo", "n_ref"].median(), s.loc[s.cls == "holo", "n_ref"].median()
        print(f"    {t:10s}{a:>8.0f}{h:>9.0f}{len(s):>10,}")
        rec("n_median", "all", "apo", t, float(a))
        rec("n_median", "all", "holo", t, float(h))

    print("\n  n_eff = distinct reference regions of the model (a model's detection sites merged when their residue sets"
          "\n  overlap by at least the threshold, as a share of the smaller set):")
    for c, thr in NEFF_THR.items():
        bits = []
        for cl in ("apo", "holo"):
            m = lbs.cls == cl
            bits.append(f"AF-{cl.capitalize()} median {lbs.loc[m, c].median():.0f} (mean {lbs.loc[m, c].mean():.2f})")
            rec(f"{c}_median", "all", cl, "all", float(lbs.loc[m, c].median()))
            rec(f"{c}_mean", "all", cl, "all", float(lbs.loc[m, c].mean()))
        print(f"    overlap >= {thr:.2f}: " + "; ".join(bits))
    ns = lbs["n_sites"]
    print(f"    detection sites on the same model (the part of n inside the detection set): median AF-Apo "
          f"{ns[lbs.cls == 'apo'].median():.0f}, AF-Holo {ns[lbs.cls == 'holo'].median():.0f}")
    r_te = spearmanr(lbs["log10_total"], lbs["neff50"], nan_policy="omit").correlation
    print(f"  Spearman rho(log10 total evidence, n_eff) = {r_te:+.3f} (n: {r_tot:+.3f})")
    rec("rho_total_neff", "all", "all", "all", r_te)
    print("  median n_eff (overlap >= 0.5) by separation tier, AF-Apo / AF-Holo: " + ", ".join(
        f"{t} {lbs.loc[(lbs.sep == t) & (lbs.cls == 'apo'), 'neff50'].median():.0f} / "
        f"{lbs.loc[(lbs.sep == t) & (lbs.cls == 'holo'), 'neff50'].median():.0f}" for t in S.TIERS))
    for t in S.TIERS:
        for cl in ("apo", "holo"):
            rec("neff50_median", "all", cl, t, float(lbs.loc[(lbs.sep == t) & (lbs.cls == cl), "neff50"].median()))
    top = lbs.drop_duplicates("site_uniprot").nlargest(3, "n_sites")
    print("  the most redundant proteins: " + "; ".join(
        f"{r.site_uniprot} n = {r.n_ref:.0f}, detection sites {r.n_sites:.0f}, n_eff {r.neff25:.0f} / {r.neff50:.0f} / "
        f"{r.neff75:.0f} at overlap 0.25 / 0.5 / 0.75" for r in top.itertuples()))

    print("\n  candidates returned and how often the cut can bind (a hit can sit beyond rank n only when the tool"
          "\n  returned more than n candidates):")
    print(f"    {'tool':12s}{'median candidates':>19s}{'max':>6s}{'> n cand., AF-Apo':>19s}{'AF-Holo':>9s}"
          f"{'hits beyond n, AF-Apo':>23s}{'AF-Holo':>9s}")
    for _, name in TOOLS:
        npred, rank = lbs[f"{name}|npred"], lbs[f"{name}|rank"]
        hit = rank >= 1
        bits = []
        for cl in ("apo", "holo"):
            m = lbs.cls == cl
            bind = pct((npred > n)[m])
            beyond = pct((rank > n)[m & hit])
            bits += [bind, beyond]
            rec("cut_can_bind_pct", name, cl, "all", bind)
            rec("hits_beyond_n_pct", name, cl, "all", beyond)
        rec("npred_median", name, "all", "all", float(npred.median()))
        print(f"    {name:12s}{npred.median():>19.0f}{npred.max():>6.0f}{bits[0]:>18.1f}%{bits[2]:>8.1f}%"
              f"{bits[1]:>22.1f}%{bits[3]:>8.1f}%")

    print("\n  fpocket by separation tier (raw rates, %): any-rank hit rate, TNS rate, and the share of any-rank hits"
          "\n  that the TNS cut removes (rank > n)")
    print(f"    {'tier':10s}{'any, apo':>10s}{'holo':>7s}{'TNS, apo':>10s}{'holo':>7s}{'removed, apo':>14s}{'holo':>7s}")
    for t in S.TIERS:
        s = lbs[lbs.sep == t]
        vals = []
        for cl in ("apo", "holo"):
            ss = s[s.cls == cl]
            anyr, tns = pct(ss["fpocket|any"]), pct(ss["fpocket|tns"])
            rem = pct((ss["fpocket|rank"] > ss["n_ref"])[ss["fpocket|rank"] >= 1])
            vals.append((anyr, tns, rem))
            rec("fpocket_any_pct", "fpocket", cl, t, anyr)
            rec("fpocket_tns_pct", "fpocket", cl, t, tns)
            rec("fpocket_removed_pct", "fpocket", cl, t, rem)
        (aa, ta, ra), (ah, th, rh) = vals
        print(f"    {t:10s}{aa:>10.1f}{ah:>7.1f}{ta:>10.1f}{th:>7.1f}{ra:>14.1f}{rh:>7.1f}")

    print("\n  how each tool ranks the hits it has (detected pockets only), by separation tier, AF-Apo | AF-Holo:"
          "\n  median first-hit rank, mean spurious candidates above the hit, mean other-site candidates above it, and"
          "\n  the share of hits ahead of every spurious candidate (rk1) and of all but two (rk3)")
    for _, name in TOOLS:
        print(f"    {name}")
        for t in S.TIERS:
            line = f"      {t:10s}"
            for cl in ("apo", "holo"):
                s = lbs[(lbs.sep == t) & (lbs.cls == cl) & (lbs[f"{name}|rank"] >= 1)]
                vals = {"r1_median": s[f"{name}|rank"].median(), "fpa_mean": s[f"{name}|fpa"].mean(),
                        "ota_mean": s[f"{name}|ota"].mean(), "rk1_pct": pct(s[f"{name}|rk1"]),
                        "rk3_pct": pct(s[f"{name}|rk3"])}
                for q, v in vals.items():
                    rec(q, name, cl, t, float(v))
                line += (f"  rank {vals['r1_median']:3.0f}  spurious {vals['fpa_mean']:5.2f}  other-site "
                         f"{vals['ota_mean']:4.2f}  rk1 {vals['rk1_pct']:5.1f}%  rk3 {vals['rk3_pct']:5.1f}%  |")
            print(line)

    print("\n  study effort: quartiles of log10(n_apo + n_holo) -- median n and n_eff, and per tool the rates (%) at any"
          "\n  rank, DCA4-TNS, top-n_eff and competition-free top-3")
    q = pd.qcut(lbs["log10_total"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    head = "".join(f"{nm + ': any':>18s}{'TNS':>6s}{'n_eff':>6s}{'cf3':>6s}" for _, nm in TOOLS)
    print(f"    {'quartile':9s}{'n':>5s}{'n_eff':>6s}{head}")
    for lab in ["Q1", "Q2", "Q3", "Q4"]:
        s = lbs[q == lab]
        line = f"    {lab:9s}{s['n_ref'].median():>5.0f}{s['neff50'].median():>6.0f}"
        rec("n_median", "all", "all", f"effort {lab}", float(s["n_ref"].median()))
        rec("neff50_median", "all", "all", f"effort {lab}", float(s["neff50"].median()))
        for _, nm in TOOLS:
            vals = [pct(s[f"{nm}|{k}"]) for k in ("any", "tns", "teff", "cf3")]
            line += f"{vals[0]:>18.1f}{vals[1]:>6.1f}{vals[2]:>6.1f}{vals[3]:>6.1f}"
            for k, v in zip(("any", "tns", "teff", "cf3"), vals):
                rec(f"{k}_pct", nm, "all", f"effort {lab}", v)
        print(line)

    print("\n  raw detection rates (%) and class gap (points) under each criterion, the analysis set:")
    print(f"    {'tool':12s}" + "".join(f"{SHORT[k]:>24s}" for k in CRITS))
    for tool in TASKS:
        line = f"    {tool:12s}"
        for k in CRITS:
            a = pct(lbs.loc[lbs.cls == "apo", f"{tool}|{k}"])
            h = pct(lbs.loc[lbs.cls == "holo", f"{tool}|{k}"])
            line += f"{a:>9.1f} / {h:4.1f} ({h - a:+5.1f})"
            rec("raw_rate_apo", tool, "apo", k, a)
            rec("raw_rate_holo", tool, "holo", k, h)
        print(line)
    print("\n  Figure 3a's fan -- raw detection (%) across the five winning-probability bins, AF-Apo then AF-Holo;"
          "\n  median n by bin: " + ", ".join(
              f"{t} {lbs.loc[(lbs.prob == t) & (lbs.cls == 'apo'), 'n_ref'].median():.0f} / "
              f"{lbs.loc[(lbs.prob == t) & (lbs.cls == 'holo'), 'n_ref'].median():.0f}" for t in S.TIERS))
    for tool in TASKS:
        for k in CRITS:
            t = lbs.groupby(["prob", "cls"])[f"{tool}|{k}"].mean().unstack().reindex(S.TIERS) * 100
            print(f"    {tool:12s}{SHORT[k]:10s} AF-Apo " + " ".join(f"{v:5.1f}" for v in t["apo"])
                  + "   AF-Holo " + " ".join(f"{v:5.1f}" for v in t["holo"]))
            for cl in ("apo", "holo"):
                for tier, v in t[cl].items():
                    rec("fig3a_raw_pct", tool, cl, f"{k}: {tier}", float(v))
    pd.DataFrame(rows).to_csv(OUT["mechanism"], index=False)
    print(f"\n  [OK] -> {OUT['mechanism'].name}")


# ============================================================================================ tiers
def gap_summary(g, code, zero):
    """Figure 4's quantities for one outcome on the shared resamples: raw and size-standardized pooled gap,
    the size-standardized gap and arm rates by tier."""
    rw = g.rates(zero, 1, raw=True)[0]
    brw = g.boot_rates(zero, 1, raw=True)[:, 0, :]
    return {"raw": (rw, brw),
            "std": (g.point(zero, 1)[0], g.boot(zero, 1)[:, 0]),
            "tier": (g.point(code, 5), g.boot(code, 5)),
            "rates": (g.rates(code, 5), g.boot_rates(code, 5)),
            "n": S.Gap.counts(code, 5)}


def continuous_slope(frames, cols, ix_list):
    """tool_gap_reconciliation.py section 6, size-standardized: y ~ decile FE x class + x + H*x with x the
    rank-uniform separation, centred; 100 * coef(H*x) = points of gap widening across the whole range. All
    outcomes as multiple right-hand sides on the same rows and the same resamples."""
    g0 = frames[cols[0]]
    d = g0.d
    Y = np.column_stack([frames[c].y for c in cols])
    h = g0.ish.astype(float)
    x = S.rank01(d["pocket_separation_pair"]) - 0.5
    ok = np.isfinite(x)
    Dm = np.zeros((len(d), S.K))
    Dm[np.arange(len(d)), g0.zb] = 1.0
    X = np.column_stack([Dm, Dm * h[:, None], x, h * x])

    def fit(ix=None):
        A, b, o = (X, Y, ok) if ix is None else (X[ix], Y[ix], ok[ix])
        return 100.0 * np.linalg.lstsq(A[o], b[o], rcond=None)[0][-1]

    return fit(), np.stack([fit(ix) for ix in ix_list])


def tiers(lbs):
    header("TIERS -- size-standardized class gap by separation tier under each criterion (Figure 4's estimator)")
    rows = []

    # ---- gate 1: Table S3's per-tier point estimates on the 87,138-pocket set
    ref3 = pd.read_csv(REF_S3)
    worst = 0.0
    for task in TASKS:
        g = S.Gap(lbs[["site_uniprot", "union_rg", "cls", "sep", f"{task}|tns"]], f"{task}|tns")
        code = S.codes_from(g.d["sep"], S.TIERS)
        p = g.point(code, 5)
        r = ref3[(ref3.task == task) & (ref3.metric == "separation")].set_index("tier").reindex(S.TIERS)
        worst = max(worst, float(np.max(np.abs(p - r["gap_std"].to_numpy()))))
        print(f"  Table S3 reproduction, {task:12s} TNS by tier: " + " / ".join(f"{v:.1f}" for v in p))
    if worst > 1e-9:
        raise AssertionError(f"Table S3 not reproduced (max |diff| {worst:.2e})")
    print(f"  [gate] {REF_S3.name}: per-tier size-standardized TNS gaps reproduced (max |diff| {worst:.1e})")

    # ---- the two-label frame and one set of resamples for everything
    x = two_label_frame(lbs)
    keep_cols = ["site_uniprot", "union_rg", "sep", "pocket_separation_pair", "cls_rel", "cls_bal"]
    slim = x[keep_cols + [f"{t}|{k}" for t in TASKS for k in CRITS]]
    ix, codes0 = None, None
    res = {}
    t0 = time.time()
    for lab, col in LABELS:
        frames = {}
        for task in TASKS:
            for k in CRITS:
                ycol = f"{task}|{k}"
                g = S.Gap(slim.assign(cls=slim[col])[["site_uniprot", "union_rg", "cls", "sep",
                                                       "pocket_separation_pair", ycol]], ycol)
                if ix is None:
                    ix, codes0 = g._indices(), g.codes
                    print(f"  one set of {len(ix)} cluster resamples over {codes0.max() + 1:,} UniProt clusters, "
                          f"{len(g.d):,} pockets, shared by every predictor, criterion and label")
                elif not np.array_equal(g.codes, codes0):
                    raise AssertionError(f"{ycol}: not on the common row set")
                g._boot_ix = ix
                code = S.codes_from(g.d["sep"], S.TIERS)
                res[(lab, task, k)] = gap_summary(g, code, np.zeros(len(g.d), dtype=int))
                frames[ycol] = g
        if lab == "released":
            cols = [f"{t}|{k}" for k in CRITS for t in TASKS]
            slope_pt, slope_bs = continuous_slope(frames, cols, ix)
        del frames
    print(f"  (bootstrap done in {time.time() - t0:.0f} s)")

    # ---- gate 2: the TNS rows are the published Table S10c-d / l.609-610 numbers
    ref = pd.read_csv(REF_S10)
    worst, nchk = 0.0, 0
    for lab, _ in LABELS:
        rl = "released" if lab == "released" else "balanced"
        for task in TASKS:
            r = res[(lab, task, "tns")]
            mine = {("raw", "all"): (r["raw"][0][1] - r["raw"][0][0], r["raw"][1][:, 1] - r["raw"][1][:, 0]),
                    ("size_std", "all"): r["std"]}
            for i, t in enumerate(S.TIERS):
                mine[("size_std", f"released separation: {t}")] = (r["tier"][0][i], r["tier"][1][:, i])
            mine[("size_std_span", "released separation")] = (r["tier"][0][4] - r["tier"][0][0],
                                                               r["tier"][1][:, 4] - r["tier"][1][:, 0])
            for (est, stratum), (pt, bs) in mine.items():
                q = ref[(ref.task == task) & (ref.labels == rl) & (ref.estimate == est) & (ref.stratum == stratum)]
                if len(q) != 1:
                    raise AssertionError(f"reference row missing: {task} {rl} {est} {stratum}")
                lo, hi = ci(bs)
                worst = max(worst, abs(pt - q.gap.iloc[0]), abs(lo - q.lo.iloc[0]), abs(hi - q.hi.iloc[0]))
                nchk += 1
    if worst > 1e-9:
        raise AssertionError(f"balanced_downstream_gaps.csv not reproduced (max |diff| {worst:.2e})")
    print(f"  [gate] {REF_S10.name} ({'v4 re-run, work_dir' if SFX else 'July set'}): {nchk} published TNS "
          f"estimates with their intervals reproduced (max |diff| {worst:.1e})")

    # ---- per label, criterion and task: the block Figure 4 / Table S10d prints
    for lab, _ in LABELS:
        for k in CRITS:
            print(f"\n  ----- {lab} labels, {CRIT_LABEL[k]} -----")
            for task in TASKS:
                r = res[(lab, task, k)]
                (rw, brw), (sp, sb), (tp, tb), (rt, brt) = r["raw"], r["std"], r["tier"], r["rates"]
                print(f"  {task}: raw AF-Apo {rw[0]:.1f}, AF-Holo {rw[1]:.1f}, gap "
                      f"{tri(rw[1] - rw[0], brw[:, 1] - brw[:, 0])}; size-standardized {tri(sp, sb)}")
                rows.append(dict(labels=lab, criterion=k, task=task, estimate="raw", stratum="all",
                                 n=int(r["n"].sum()), gap=rw[1] - rw[0], lo=ci(brw[:, 1] - brw[:, 0])[0],
                                 hi=ci(brw[:, 1] - brw[:, 0])[1], rate_apo=rw[0], rate_holo=rw[1]))
                rows.append(dict(labels=lab, criterion=k, task=task, estimate="size_std", stratum="all",
                                 n=int(r["n"].sum()), gap=sp, lo=ci(sb)[0], hi=ci(sb)[1]))
                for i, t in enumerate(S.TIERS):
                    print(f"    {t:10s} n = {r['n'][i]:>6,}   {tri(tp[i], tb[:, i]):>22s}   std rates apo / holo "
                          f"{rt[i, 0]:.1f} / {rt[i, 1]:.1f}")
                    rows.append(dict(labels=lab, criterion=k, task=task, estimate="size_std", stratum=t,
                                     n=int(r["n"][i]), gap=tp[i], lo=ci(tb[:, i])[0], hi=ci(tb[:, i])[1],
                                     rate_apo=rt[i, 0], rate_holo=rt[i, 1]))
                span, bspan = tp[4] - tp[0], tb[:, 4] - tb[:, 0]
                arm_a, barm_a = rt[4, 0] - rt[0, 0], brt[:, 4, 0] - brt[:, 0, 0]
                arm_h, barm_h = rt[4, 1] - rt[0, 1], brt[:, 4, 1] - brt[:, 0, 1]
                print(f"    very high minus very low: gap {tri(span, bspan)}; AF-Apo arm {tri(arm_a, barm_a)}, "
                      f"AF-Holo arm {tri(arm_h, barm_h)}")
                for est, pt, bs in (("span", span, bspan), ("arm_apo", arm_a, barm_a), ("arm_holo", arm_h, barm_h)):
                    rows.append(dict(labels=lab, criterion=k, task=task, estimate=est, stratum="very_high-very_low",
                                     n=int(r["n"].sum()), gap=pt, lo=ci(bs)[0], hi=ci(bs)[1]))

    # ---- ledger and paired contrasts
    for lab, _ in LABELS:
        header(f"LEDGER, {lab} labels -- pooled size-standardized gap (level) against its separation span (gradient)")
        print(f"  {'criterion':11s}{'task':13s}{'pooled gap':>22s}{'span (v.high - v.low)':>25s}"
              f"{'AF-Apo arm':>24s}{'AF-Holo arm':>24s}")
        for k in CRITS:
            for task in TASKS:
                r = res[(lab, task, k)]
                (sp, sb), (tp, tb), (rt, brt) = r["std"], r["tier"], r["rates"]
                print(f"  {SHORT[k]:11s}{task:13s}{tri(sp, sb):>22s}"
                      f"{tri(tp[4] - tp[0], tb[:, 4] - tb[:, 0]):>25s}"
                      f"{tri(rt[4, 0] - rt[0, 0], brt[:, 4, 0] - brt[:, 0, 0]):>24s}"
                      f"{tri(rt[4, 1] - rt[0, 1], brt[:, 4, 1] - brt[:, 0, 1]):>24s}")
        print(f"\n  paired contrasts between predictors ({lab} labels)")
        for k in CRITS:
            span = {t: (res[(lab, t, k)]["tier"][0][4] - res[(lab, t, k)]["tier"][0][0],
                        res[(lab, t, k)]["tier"][1][:, 4] - res[(lab, t, k)]["tier"][1][:, 0]) for t in PRED}
            pooled = {t: res[(lab, t, k)]["std"] for t in PRED}
            vlow = {t: (res[(lab, t, k)]["tier"][0][0], res[(lab, t, k)]["tier"][1][:, 0]) for t in PRED}
            arm = {t: (res[(lab, t, k)]["rates"][0][4, 0] - res[(lab, t, k)]["rates"][0][0, 0],
                       res[(lab, t, k)]["rates"][1][:, 4, 0] - res[(lab, t, k)]["rates"][1][:, 0, 0]) for t in PRED}
            print(f"    {CRIT_LABEL[k]}:")
            for a, b in PAIRS:
                print(f"      {a} - {b}: span {tri(span[a][0] - span[b][0], span[a][1] - span[b][1])}, "
                      f"pooled gap {tri(pooled[a][0] - pooled[b][0], pooled[a][1] - pooled[b][1])}, "
                      f"very-low gap {tri(vlow[a][0] - vlow[b][0], vlow[a][1] - vlow[b][1])}, "
                      f"AF-Apo arm {tri(arm[a][0] - arm[b][0], arm[a][1] - arm[b][1])}")
                for est, dd in (("span", span), ("size_std", pooled), ("very_low", vlow), ("arm_apo", arm)):
                    bs = dd[a][1] - dd[b][1]
                    rows.append(dict(labels=lab, criterion=k, task=f"{a} - {b}", estimate=f"contrast_{est}",
                                     stratum="paired", gap=dd[a][0] - dd[b][0], lo=ci(bs)[0], hi=ci(bs)[1]))
            ps = rank_share({t: span[t][0] for t in PRED}, {t: span[t][1] for t in PRED}, "fpocket")
            pl = rank_share({t: pooled[t][0] for t in PRED}, {t: pooled[t][1] for t in PRED}, "fpocket",
                            largest=False)
            print(f"      share of resamples in which fpocket has the widest span {ps:.1f}%, the smallest pooled "
                  f"gap {pl:.1f}%")
            rows.append(dict(labels=lab, criterion=k, task="fpocket", estimate="p_widest_span", stratum="paired",
                             gap=ps))
            rows.append(dict(labels=lab, criterion=k, task="fpocket", estimate="p_smallest_pooled",
                             stratum="paired", gap=pl))
        print(f"\n  each criterion against the published TNS, same predictor ({lab} labels): span and pooled gap")
        for task in TASKS:
            r0 = res[(lab, task, "tns")]
            s0, b0 = r0["tier"][0][4] - r0["tier"][0][0], r0["tier"][1][:, 4] - r0["tier"][1][:, 0]
            bits = []
            for k in CRITS[1:]:
                r = res[(lab, task, k)]
                s1, b1 = r["tier"][0][4] - r["tier"][0][0], r["tier"][1][:, 4] - r["tier"][1][:, 0]
                bits.append(f"{CRIT_LABEL[k]} span {tri(s1 - s0, b1 - b0)}, pooled "
                            f"{tri(r['std'][0] - r0['std'][0], r['std'][1] - r0['std'][1])}")
                rows.append(dict(labels=lab, criterion=k, task=task, estimate="span_minus_tns", stratum="paired",
                                 gap=s1 - s0, lo=ci(b1 - b0)[0], hi=ci(b1 - b0)[1]))
            print(f"    {task:12s}" + "; ".join(bits))

    # ---- scale-free and un-tiered checks, released labels
    header("CHECKS ON THE SPAN, released labels -- log odds ratio from the standardized rates, and the continuous slope")
    print(f"  {'criterion':11s}{'task':13s}{'log-OR span':>26s}{'continuous slope (pp)':>28s}")
    cols = [f"{t}|{k}" for k in CRITS for t in TASKS]
    for k in CRITS:
        for task in TASKS:
            r = res[("released", task, k)]
            lo_, blo = log_or(r["rates"][0]), log_or(r["rates"][1])
            j = cols.index(f"{task}|{k}")
            print(f"  {SHORT[k]:11s}{task:13s}"
                  f"{tri(lo_[4] - lo_[0], blo[:, 4] - blo[:, 0], dp=2):>26s}"
                  f"{tri(slope_pt[j], slope_bs[:, j]):>28s}")
            rows.append(dict(labels="released", criterion=k, task=task, estimate="log_or_span",
                             stratum="very_high-very_low", gap=lo_[4] - lo_[0], lo=ci(blo[:, 4] - blo[:, 0])[0],
                             hi=ci(blo[:, 4] - blo[:, 0])[1]))
            rows.append(dict(labels="released", criterion=k, task=task, estimate="continuous_slope",
                             stratum="full range", gap=slope_pt[j], lo=ci(slope_bs[:, j])[0],
                             hi=ci(slope_bs[:, j])[1]))
        jf, ji, jp = (cols.index(f"{t}|{k}") for t in ("fpocket", "IF-SitePred", "P2Rank"))
        print(f"  {'':11s}continuous slope, fpocket - IF-SitePred "
              f"{tri(slope_pt[jf] - slope_pt[ji], slope_bs[:, jf] - slope_bs[:, ji])}, fpocket - P2Rank "
              f"{tri(slope_pt[jf] - slope_pt[jp], slope_bs[:, jf] - slope_bs[:, jp])}")

    # ---- how much top-n_eff depends on what counts as one region
    header("SENSITIVITY -- the residue-overlap threshold that merges ST_100 sites into one region (top-n_eff),\n"
           "released labels, the shared resamples")
    xr = x.assign(cls=x["cls_rel"])

    def shared_gap(frame, ycol):
        g = S.Gap(frame[["site_uniprot", "union_rg", "cls", "sep", ycol]], ycol)
        if not np.array_equal(g.codes, codes0):
            raise AssertionError(f"{ycol}: not on the common row set")
        g._boot_ix = ix
        return g

    for task in TASKS:
        bits = []
        for k, thr in (("teff25", "0.25"), ("teff", "0.5"), ("teff75", "0.75")):
            g = shared_gap(xr, f"{task}|{k}")
            code = S.codes_from(g.d["sep"], S.TIERS)
            p, b = g.point(code, 5), g.boot(code, 5)
            rt, brt = g.rates(code, 5), g.boot_rates(code, 5)
            bits.append(f"overlap {thr}: span {tri(p[4] - p[0], b[:, 4] - b[:, 0])}, AF-Apo arm "
                        f"{tri(rt[4, 0] - rt[0, 0], brt[:, 4, 0] - brt[:, 0, 0])}")
            rows.append(dict(labels="released", criterion=k, task=task, estimate="span", stratum="very_high-very_low",
                             gap=p[4] - p[0], lo=ci(b[:, 4] - b[:, 0])[0], hi=ci(b[:, 4] - b[:, 0])[1]))
            rows.append(dict(labels="released", criterion=k, task=task, estimate="arm_apo",
                             stratum="very_high-very_low", gap=rt[4, 0] - rt[0, 0],
                             lo=ci(brt[:, 4, 0] - brt[:, 0, 0])[0], hi=ci(brt[:, 4, 0] - brt[:, 0, 0])[1]))
        print(f"  {task:12s}" + "; ".join(bits))

    # ---- detection against ranking: where each predictor's class signal enters
    header("WHERE THE CLASS SIGNAL ENTERS -- detection (any rank, no budget) against ranking given detection,\n"
           "released labels")
    print("  ranking given detection: among the pockets a tool detects at any rank, the share whose hit ranks ahead of\n"
          "  every spurious candidate (rk1) or of all but two (rk3). Size-standardized on that subset's own union-Rg\n"
          "  deciles, with its own cluster resamples, so these rows are not paired with the ones above.")
    for task in PRED:
        det = res[("released", task, "any")]
        (sp, sb), (tp, tb), (rt_, brt_) = det["std"], det["tier"], det["rates"]
        print(f"\n  {task}: detection, pooled {tri(sp, sb)}, tiers " + " / ".join(f"{v:+.1f}" for v in tp)
              + f", span {tri(tp[4] - tp[0], tb[:, 4] - tb[:, 0])}, AF-Apo arm "
              f"{tri(rt_[4, 0] - rt_[0, 0], brt_[:, 4, 0] - brt_[:, 0, 0])}")
        sub = xr[xr[f"{task}|rank"] >= 1]
        for k in ("rk1", "rk3"):
            if task == "IF-SitePred" and k == "rk3":
                print("    rk3: IF-SitePred returns at most three sites, so every hit meets it")
                continue
            g = S.Gap(sub[["site_uniprot", "union_rg", "cls", "sep", f"{task}|{k}"]], f"{task}|{k}")
            code = S.codes_from(g.d["sep"], S.TIERS)
            zero = np.zeros(len(g.d), dtype=int)
            r = gap_summary(g, code, zero)
            (rw, brw), (sp, sb), (tp, tb), (rt_, brt_) = r["raw"], r["std"], r["tier"], r["rates"]
            print(f"    {k} ({len(g.d):,} detected): raw AF-Apo {rw[0]:.1f}, AF-Holo {rw[1]:.1f}; size-standardized "
                  f"pooled {tri(sp, sb)}; tiers " + " / ".join(f"{v:+.1f}" for v in tp)
                  + f"; span {tri(tp[4] - tp[0], tb[:, 4] - tb[:, 0])}; std rates very low {rt_[0, 0]:.1f} / "
                  f"{rt_[0, 1]:.1f}, very high {rt_[4, 0]:.1f} / {rt_[4, 1]:.1f}; AF-Apo arm "
                  f"{tri(rt_[4, 0] - rt_[0, 0], brt_[:, 4, 0] - brt_[:, 0, 0])}")
            for est, pt, bs in (("size_std", sp, sb), ("span", tp[4] - tp[0], tb[:, 4] - tb[:, 0]),
                                ("arm_apo", rt_[4, 0] - rt_[0, 0], brt_[:, 4, 0] - brt_[:, 0, 0])):
                rows.append(dict(labels="released", criterion=k, task=task, estimate=f"given_detection_{est}",
                                 stratum="detected", n=len(g.d), gap=pt, lo=ci(bs)[0], hi=ci(bs)[1]))
            for i, t in enumerate(S.TIERS):
                rows.append(dict(labels="released", criterion=k, task=task, estimate="given_detection_size_std",
                                 stratum=t, n=int(r["n"][i]), gap=tp[i], lo=ci(tb[:, i])[0], hi=ci(tb[:, i])[1],
                                 rate_apo=rt_[i, 0], rate_holo=rt_[i, 1]))

    # ---- the ranking signature compared between tools on ONE row set: pockets all three detect
    both = xr[(xr["P2Rank|rank"] >= 1) & (xr["fpocket|rank"] >= 1) & (xr["IF-SitePred|rank"] >= 1)]
    print(f"\n  ranking given detection, paired: the {len(both):,} pockets that all three predictors detect at any rank"
          f"\n  ({both.site_uniprot.nunique():,} UniProt clusters), rk1, one set of resamples for the three tools")
    gg, ixb = {}, None
    for task in PRED:
        g = S.Gap(both[["site_uniprot", "union_rg", "cls", "sep", f"{task}|rk1"]], f"{task}|rk1")
        if ixb is None:
            ixb, codesb = g._indices(), g.codes
        elif not np.array_equal(g.codes, codesb):
            raise AssertionError("common-detection subset differs between tools")
        g._boot_ix = ixb
        gg[task] = g
    codeb = S.codes_from(gg[PRED[0]].d["sep"], S.TIERS)
    zerob = np.zeros(len(codeb), dtype=int)
    rb = {t: gap_summary(g, codeb, zerob) for t, g in gg.items()}
    for task in PRED:
        (sp, sb), (tp, tb), (rt_, brt_) = rb[task]["std"], rb[task]["tier"], rb[task]["rates"]
        print(f"    {task:12s} pooled {tri(sp, sb)}; tiers " + " / ".join(f"{v:+.1f}" for v in tp)
              + f"; span {tri(tp[4] - tp[0], tb[:, 4] - tb[:, 0])}; AF-Apo arm "
              f"{tri(rt_[4, 0] - rt_[0, 0], brt_[:, 4, 0] - brt_[:, 0, 0])}")
        rows.append(dict(labels="released", criterion="rk1", task=task, estimate="common_detected_span",
                         stratum="very_high-very_low", n=len(both), gap=tp[4] - tp[0],
                         lo=ci(tb[:, 4] - tb[:, 0])[0], hi=ci(tb[:, 4] - tb[:, 0])[1]))
    for a, b in PAIRS:
        ta, tb_ = rb[a]["tier"], rb[b]["tier"]
        sa, sb_ = ta[0][4] - ta[0][0], tb_[0][4] - tb_[0][0]
        bsd = (ta[1][:, 4] - ta[1][:, 0]) - (tb_[1][:, 4] - tb_[1][:, 0])
        aa = rb[a]["rates"][0][4, 0] - rb[a]["rates"][0][0, 0]
        ab = rb[b]["rates"][0][4, 0] - rb[b]["rates"][0][0, 0]
        bsa = ((rb[a]["rates"][1][:, 4, 0] - rb[a]["rates"][1][:, 0, 0])
               - (rb[b]["rates"][1][:, 4, 0] - rb[b]["rates"][1][:, 0, 0]))
        print(f"    {a} - {b}: span {tri(sa - sb_, bsd)}, AF-Apo arm {tri(aa - ab, bsa)}")
        rows.append(dict(labels="released", criterion="rk1", task=f"{a} - {b}", estimate="common_detected_span_contrast",
                         stratum="paired", n=len(both), gap=sa - sb_, lo=ci(bsd)[0], hi=ci(bsd)[1]))
    pd.DataFrame(rows).to_csv(OUT["tiers"], index=False)
    print(f"\n  [OK] -> {OUT['tiers'].name}")


# ============================================================================================== s11
def _groups(clus):
    """evidence_amount_vs_ratio.run() builds {cluster: np.flatnonzero(clus == c)}; this is the same map,
    in the same order, without 10k passes over the column."""
    uniq, codes = np.unique(clus, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    st = np.searchsorted(cs, np.arange(len(uniq)), "left")
    en = np.searchsorted(cs, np.arange(len(uniq)), "right")
    groups = [order[a:b] for a, b in zip(st, en)]
    for i in np.linspace(0, len(uniq) - 1, 25).astype(int):
        if not np.array_equal(groups[i], np.flatnonzero(clus == uniq[i])):
            raise AssertionError("cluster index map differs from evidence_amount_vs_ratio.run()'s")
    return uniq, groups


def s11():
    header("TABLE S11 UNDER EACH CRITERION -- success against the apo/holo predictors with study effort controlled")
    ref = pd.read_csv(REF_S11) if REF_S11.is_file() else None
    rng = np.random.default_rng(E.SEED)      # consumed exactly as evidence_amount_vs_ratio.run() consumes it
    B = E.B
    rows, boots = [], {}
    rk = site_ranks().set_index("site_uid")
    s11_name = {"fpocket": "fpocket", "ifsitepred": "IF-SitePred", "p2rank": "P2Rank"}
    t0 = time.time()
    for tool in E.METHODS:
        md = E.method_data(tool)
        g = E.frame(md)
        size = list(md.ctrl_cols)
        y0 = md.primary().col
        lbs = tool != "docking"
        if lbs:
            rank = pd.to_numeric(g["dca_rank"], errors="coerce").to_numpy(float)
            n = pd.to_numeric(g["n_unp_pockets"], errors="coerce").to_numpy(float)
            j = rk.reindex(g["site_uid"])
            nm = s11_name[tool]
            if not np.array_equal(j[f"{nm}|r1"].to_numpy(float), rank):
                raise AssertionError(f"{tool}: site_ranks.csv does not match the S7 loader's dca_rank")
            pub = g[y0].to_numpy(float)
            defined = np.isfinite(pub)
            outs = cut_outcomes(rank, n, j["neff50"].to_numpy(float), j[f"{nm}|fpa"].to_numpy(float), defined)
            if not np.array_equal(outs["tns"][defined], pub[defined]):
                raise AssertionError(f"{tool}: rank <= n does not reproduce success_tns in the S7 loader")
            for k, v in outs.items():
                g[f"cut_{k}"] = v
            g["log10_n_ref"] = np.log10(np.where(n >= 1, n, np.nan))
            ycols, extra = [f"cut_{k}" for k in CRITS], ["log10_n_ref"]
        else:
            ycols, extra = [y0], []
        print(f"\n--- {tool}  (outcome {'rank-cut success' if lbs else y0}; size control {size}) ---")
        for gname, levels in E.GATES:
            gm = qg.gate_mask(g, levels, None)
            need = [y0, qg.RATIO, "p_holo", "is_holo", E.TOTAL] + size       # the published common mask
            sub = g[gm & g[need].notna().all(axis=1)]
            if lbs and not sub[ycols + extra].notna().all().all():
                raise AssertionError(f"{tool}/{gname}: a fixed-cut outcome is missing on the published rows")
            union = [qg.RATIO, "p_holo", "is_holo"] + size + [E.TOTAL] + extra
            M = np.column_stack([sub[c].to_numpy(float) for c in ycols + union])
            at = {c: len(ycols) + i for i, c in enumerate(union)}
            P, Sz, T, R = [at[c] for c in qg.PREDICTORS], [at[c] for c in size], at[E.TOTAL], at[qg.RATIO]
            plan = []
            for j, yc in enumerate(ycols):
                plan += [(j, "zero", 3, [j] + P), (j, "ctrl", 3, [j] + P + Sz),
                         (j, "ctrl+tot", 3, [j] + P + Sz + [T]), (j, "mirror", 1, [j, T] + Sz + [R])]
                if lbs:
                    plan.append((j, "mirror|n", 1, [j, T] + Sz + [R, at["log10_n_ref"]]))

            def rhos(Mx):
                Vr = qg._rank_cols(Mx)
                return [qg._partial_rhos_from_ranks(Vr[:, cols], npd, len(cols) - 1 - npd)
                        for _j, _r, npd, cols in plan]

            pts = rhos(M)
            # the published regimes must come out of this path unchanged (evidence_amount_vs_ratio's own check)
            j_tns = 0
            for pi, (j, rname, _npd, _cols) in enumerate(plan):
                if j == j_tns and rname in ("zero", "ctrl", "ctrl+tot"):
                    ctrl = {"zero": [], "ctrl": size, "ctrl+tot": size + [E.TOTAL]}[rname]
                    refr, _n = E.rhos_for(sub, y0, qg.PREDICTORS, ctrl)
                    if not np.allclose(refr, pts[pi], atol=1e-9, equal_nan=True):
                        raise AssertionError(f"fast path disagrees with qg.partial_rhos for {tool}/{gname}/{rname}")
            uniq, groups = _groups(sub["_clu"].to_numpy())
            bs = [[] for _ in plan]
            for _ in range(B):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                take = np.concatenate([groups[i] for i in np.searchsorted(uniq, pick)])
                for pi, v in enumerate(rhos(M[take])):
                    bs[pi].append(v)
            print(f"  {gname:<11} n={len(sub):,}  clusters={len(uniq):,}   ({time.time() - t0:.0f} s)")
            for pi, (j, rname, _npd, _cols) in enumerate(plan):
                crits_here = [CRITS[j]] if lbs else CRITS    # docking's one outcome stands for every criterion
                arr = np.asarray(bs[pi], float)
                preds = qg.PREDICTORS if rname in ("zero", "ctrl", "ctrl+tot") else [E.TOTAL]
                for c in crits_here:
                    for q, p in enumerate(preds):
                        b = arr[:, q]
                        b = b[np.isfinite(b)]
                        lo, hi = ((np.nan, np.nan) if b.size < 10 else
                                  (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))))
                        rows.append({"model_set": SFX or "v6", "criterion": c, "method": tool, "gate": gname,
                                     "n": len(sub), "clusters": len(uniq), "regime": rname, "predictor": p,
                                     "rho": float(pts[pi][q]), "lo": lo, "hi": hi})
                        boots[(c, tool, gname, rname, p)] = arr[:, q]
    df = pd.DataFrame(rows)
    df.to_csv(OUT["s11"], index=False)
    print(f"\n  [OK] -> {OUT['s11'].name}   ({time.time() - t0:.0f} s)")

    # ---- gate: the TNS rows are SI Table S11
    if ref is not None:
        m = df[df.criterion == "tns"].merge(ref, on=["method", "gate", "regime", "predictor"], suffixes=("", "_ref"))
        if len(m) != len(ref):
            raise AssertionError(f"Table S11 rows: {len(m)} matched of {len(ref)}")
        worst = float(np.nanmax(np.abs(m[["rho", "lo", "hi"]].to_numpy() - m[["rho_ref", "lo_ref", "hi_ref"]].to_numpy())))
        if worst > 1e-12 or not (m.n == m.n_ref).all() or not (m.clusters == m.clusters_ref).all():
            raise AssertionError(f"Table S11 not reproduced (max |diff| {worst:.2e})")
        print(f"  [gate] {REF_S11.name}: all {len(ref)} published rows reproduced, point and interval "
              f"(max |diff| {worst:.1e})")
    else:
        print(f"  (no {REF_S11.name}: the TNS gate is skipped for this model set)")

    def row(c, tool, gname, rname, p):
        return df[(df.criterion == c) & (df.method == tool) & (df.gate == gname) & (df.regime == rname)
                  & (df.predictor == p)].iloc[0]

    def rt(r):
        return f"{r.rho:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"

    header("PER TASK -- the amount's own effect (mirror: success ~ total evidence | size, ratio), by criterion")
    for gname, _ in E.GATES:
        print(f"\n  gate {gname}")
        for tool in E.METHODS:
            for c in CRITS:
                r = row(c, tool, gname, "mirror", E.TOTAL)
                line = f"    {tool:11s}{CRIT_LABEL[c]:34s} mirror {rt(r)}"
                if tool != "docking":
                    rn = row(c, tool, gname, "mirror|n", E.TOTAL)
                    line += f"   with n also held fixed {rt(rn)}"
                    if c != "tns":
                        dd = boots[(c, tool, gname, "mirror", E.TOTAL)] - boots[("tns", tool, gname, "mirror", E.TOTAL)]
                        lo, hi = ci(dd)
                        line += f"   minus TNS {r.rho - row('tns', tool, gname, 'mirror', E.TOTAL).rho:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                print(line)

    header("PER TASK -- the evidence ratio before and after the total is controlled, and its rivals, by criterion")
    for tool in E.METHODS[:3]:
        print(f"\n  {tool}, all pockets")
        for c in CRITS:
            r1, r2 = row(c, tool, "all", "ctrl", qg.RATIO), row(c, tool, "all", "ctrl+tot", qg.RATIO)
            ph, ih = row(c, tool, "all", "ctrl+tot", "p_holo"), row(c, tool, "all", "ctrl+tot", "is_holo")
            print(f"    {CRIT_LABEL[c]:34s} ratio | size {r1.rho:+.3f} -> | size + total {rt(r2)};  "
                  f"p_holo {ph.rho:+.3f}, class {ih.rho:+.3f}")

    header("PER TASK -- SI Figure S7's comparison (ratio, p_holo, class) by gate and criterion, size-controlled and with\n"
           "the total also controlled; the last two columns are the paired differences p_holo - ratio and class - ratio")
    for tool in E.METHODS:
        for rname in ("ctrl", "ctrl+tot"):
            print(f"\n  {tool}, regime {rname}")
            for gname, _ in E.GATES:
                for c in (CRITS if tool != "docking" else CRITS[:1]):
                    rr = [row(c, tool, gname, rname, p) for p in qg.PREDICTORS]
                    br = [boots[(c, tool, gname, rname, p)] for p in qg.PREDICTORS]
                    d1, d2 = br[1] - br[0], br[2] - br[0]
                    (l1, h1), (l2, h2) = ci(d1), ci(d2)
                    lab = CRIT_LABEL[c] if tool != "docking" else "(no rank cut)"
                    print(f"    {gname:<11}{lab:34s} ratio {rt(rr[0])}  p_holo {rt(rr[1])}  class {rt(rr[2])}"
                          f"   p_holo - ratio {rr[1].rho - rr[0].rho:+.3f} [{l1:+.3f}, {h1:+.3f}]"
                          f"   class - ratio {rr[2].rho - rr[0].rho:+.3f} [{l2:+.3f}, {h2:+.3f}]")

    header("POOLED over the four tasks (Fisher-z mean), by criterion -- the rows of SI Table S11")
    for c in CRITS:
        print(f"\n  {CRIT_LABEL[c]}")
        for gname, _ in E.GATES:
            for rname in ("zero", "ctrl", "ctrl+tot"):
                bits = []
                for p in qg.PREDICTORS:
                    v = df[(df.criterion == c) & (df.gate == gname) & (df.regime == rname) & (df.predictor == p)].rho
                    bits.append(f"{p} {np.tanh(np.arctanh(v.to_numpy()).mean()):+.3f}")
                tail = ""
                if rname == "ctrl+tot":
                    v = df[(df.criterion == c) & (df.gate == gname) & (df.regime == "mirror")].rho
                    tail = f"   total evidence, ratio fixed {np.tanh(np.arctanh(v.to_numpy()).mean()):+.3f}"
                print(f"    {gname:<11} {rname:<9} " + "   ".join(bits) + tail)
    return df


# ============================================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["ranks", "mechanism", "tiers", "s11", "all"])
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    print(f"detection_rank_cut.py {a.mode}   [model set {MODEL_SET or 'July/v6'!r}]")
    if a.mode in ("ranks", "all"):          # `all` rebuilds it, so its gates are in the printout the checker reads
        site_ranks(rebuild=True)
        if a.mode == "ranks":
            return
    if a.mode in ("mechanism", "tiers", "all"):
        base = detection_base()
        lbs = lbs_frame(base)
        print(f"  detection base: rank <= n reproduces the released success_tns for all three tools; "
              f"{len(lbs):,} pockets equal separation_shape.load_lbs() row for row")
        if a.mode in ("mechanism", "all"):
            mechanism(base, lbs)
        if a.mode in ("tiers", "all"):
            tiers(lbs)
    if a.mode in ("s11", "all"):
        s11()


if __name__ == "__main__":
    main()
