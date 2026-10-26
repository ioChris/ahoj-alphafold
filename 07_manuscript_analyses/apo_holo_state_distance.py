# -*- coding: utf-8 -*-
"""How far apart are the apo and holo states of a binding site, and where does the AlphaFold pocket sit
relative to the distances between their MEMBERS?  Backs Supplementary Table S7, "Distance between the states"
(d', trimmed ranges, sampling depth, separation, and the model against the conformers).

Three questions:
  1. Robustness. SI Table S7's range statistics use min-max ranges. Do they hold when each class's range is
     trimmed to its central 99 / 97 / 95%?
  2. Distance. How far apart are the two state centres, in units of the within-state spread (d')? How does
     that depend on sampling depth and on the released separation descriptor?
  3. The member-level frontier. SI Table S5 compares the model with the CLASS MEANS. Here the model's distance
     to its nearer state is compared with (a) that state's members among themselves, (b) the apo<->holo
     member distances, (c) all structures pairwise (Lane's comparison); and the model is asked whether it is
     closer to its nearer state than the OTHER state's members are, i.e. whether it takes a side.

Data: per-structure metric values live only in the per-job AHoJ archives. `collect` reads them with the
classifier's own curation by importing ensemble_containment.py (same functions, same parameters; its
n_apo / n_holo fidelity against the classification summary is re-checked here), joins the model's values
(query_summaries for volume / SASA / RoG; the harvested model_pocket_rms file for pocket RMSD) and computes
every per-entry statistic in the worker. `summarise` prints every number Table S7 quotes and writes the
tidy summary CSV.

Volume. The classifier's v2c rule (`volume_nan_to_zero`, 00_classification/kde_classifier/batch_job_functions.py) turns a missing PyVOL
volume into 0.0 when the pocket has residues; the published class means keep those zeros (stats table
`volume_apo_mean` = per-entry mean with zeros, max |diff| 0.0005), and the model's own volume is 0 in 37%
of pockets. `volume` below keeps the zeros (the published convention); `volume_nz` drops them from both
classes and treats a zero model volume as missing (sensitivity).

Metrics: volume, volume_nz, sasa, RoG, pocket_rms. pocket_rms is a distance from the query (holo) structure,
so for that metric every value is a radial coordinate around a holo anchor.

    python apo_holo_state_distance.py collect [--limit N] [--procs 10]  -> apo_holo_state_distance_entries.parquet
    python apo_holo_state_distance.py summarise  > apo_holo_state_distance.out.txt
                                                                         -> apo_holo_state_distance_summary.csv
All files are in the manuscript-analyses folder (ma_paths.OUT). The per-entry table (~238k rows) is
regenerable; the summary CSV and the stdout are the published record.
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import argparse
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ensemble_containment as ec  # noqa: E402  (curation, paths, archive reader)

DB = ec.DB  # .../AHOJ_DB/v2c
MODEL = os.path.join(DB, "Summaries", "model_metrics", "full_run_3", "run_23082026_180200_do_collect_model_pLDDT",
                     "query_summaries_do_collect_model_pLDDT.csv")
STATS = os.path.join(DB, "Summaries", "apoholo_metric_stats", "full_run_3",
                     "query_summaries_do_collect_apoholo_metric_stats.csv")
ENTRIES = P.out("apo_holo_state_distance_entries.parquet")
SUMMARY_CSV = P.out("apo_holo_state_distance_summary.csv")
FRONTIER_TXT = P.out("frontier_final_table.txt")           # frontier_final.py (SI Table S5)
CONTAIN_CSV = P.out("ensemble_containment_summary.csv")    # ensemble_containment.py summarise

SRC = ["volume", "sasa", "RoG", "pocket_rms"]                 # archive columns
OUTM = ["volume", "volume_nz", "sasa", "RoG", "pocket_rms"]   # reported metrics
MODCOL = {"volume": "model_pocket_volume", "volume_nz": "model_pocket_volume", "sasa": "model_pocket_sasa",
          "RoG": "model_pocket_RoG", "pocket_rms": "model_pocket_rms"}
LABEL = {"volume": "volume (zeros kept)", "volume_nz": "volume (zeros dropped)", "sasa": "SASA",
         "RoG": "radius of gyration", "pocket_rms": "pocket RMSD"}
QS = (100, 99, 97, 95)
CAP = 300          # members per class for the pairwise statistics
TIER_EDGES = [-1, .2, .3, .6, .85, 2]
TIER_LAB = ["very low", "low", "moderate", "high", "very high"]
NMIN_EDGES = [1, 2, 5, 10, 25, 10 ** 9]
NMIN_LAB = ["2", "3-5", "6-10", "11-25", ">25"]


# ----------------------------------------------------------------------------------------------- worker

def _upper(M):
    i, j = np.triu_indices(M.shape[0], k=1)
    return M[i, j]


def entry_stats(A, H, m, rng):
    """A, H: non-null per-structure values of one metric; m: the model's value (NaN allowed)."""
    na, nh = len(A), len(H)
    o = {"n_a": na, "n_h": nh}
    for tag, x in (("a", A), ("h", H)):
        if x.size:
            o[f"{tag}_mean"], o[f"{tag}_min"], o[f"{tag}_max"] = float(x.mean()), float(x.min()), float(x.max())
            o[f"{tag}_q50"] = float(np.median(x))
        if x.size >= 2:
            o[f"{tag}_sd"] = float(x.std(ddof=1))
    has_m = m is not None and np.isfinite(m)
    o["has_model"] = bool(has_m)
    if has_m:
        o["m"] = float(m)
    # --- SI Table S5 reproduction block: class-mean frontier and pooled bands (needs >= 1 of each class)
    if has_m and na >= 1 and nh >= 1:
        ds = abs(H.mean() - A.mean())
        if ds > 0:
            o["frontier_mean"] = bool(min(abs(m - A.mean()), abs(m - H.mean())) < ds)
            o["t_pos"] = float((m - A.mean()) / (H.mean() - A.mean()))
    X = np.concatenate([A, H])
    if has_m and X.size >= 2:
        lo, hi = X.min(), X.max()
        if hi > lo:
            o["inside_minmax"] = bool(lo <= m <= hi)
        mu, sd = X.mean(), X.std(ddof=1)
        if sd > 0:
            o["inside_1sd"] = bool(abs(m - mu) <= sd)
            o["inside_2sd"] = bool(abs(m - mu) <= 2 * sd)
    if na < 2 or nh < 2:
        return o
    # --- range statistics at four central coverages (both classes >= 2)
    for q in QS:
        lo = (100 - q) / 200
        aL, aU = np.quantile(A, [lo, 1 - lo])
        hL, hU = np.quantile(H, [lo, 1 - lo])
        a_in = float(np.mean((A >= hL) & (A <= hU)))
        h_in = float(np.mean((H >= aL) & (H <= aU)))
        o[f"a_in_h_{q}"], o[f"h_in_a_{q}"] = a_in, h_in
        o[f"interp_{q}"] = bool(a_in > 0 and h_in > 0)
        o[f"disjoint_{q}"] = bool(aU < hL or hU < aL)
        inter = max(0.0, min(aU, hU) - max(aL, hL))
        union = max(aU, hU) - min(aL, hL)
        o[f"jac_{q}"] = float(inter / union) if union > 0 else np.nan
        At, Ht = A[(A >= aL) & (A <= aU)], H[(H >= hL) & (H <= hU)]
        if len(At) >= 2 and len(Ht) >= 2:
            w = np.sqrt((At.std(ddof=1) ** 2 + Ht.std(ddof=1) ** 2) / 2)
            o[f"dprime_{q}"] = float(abs(Ht.mean() - At.mean()) / w) if w > 0 else np.nan
    iqa = np.subtract(*np.percentile(A, [75, 25]))
    iqh = np.subtract(*np.percentile(H, [75, 25]))
    wr = np.sqrt((iqa ** 2 + iqh ** 2) / 2) / 1.349
    o["dprime_robust"] = float(abs(np.median(H) - np.median(A)) / wr) if wr > 0 else np.nan
    w_true = np.sqrt((A.std(ddof=1) ** 2 + H.std(ddof=1) ** 2) / 2)
    o["within_sd"] = float(w_true)
    o["between"] = float(abs(H.mean() - A.mean()))
    # the classifier's own effect size (parse-ahoj kde_core.cohen_d_safe): pooled SD weighted by n - 1,
    # so the better-sampled class's spread dominates the denominator; kept for the fidelity check
    pooled = np.sqrt(((na - 1) * A.std(ddof=1) ** 2 + (nh - 1) * H.std(ddof=1) ** 2) / (na + nh - 2))
    o["cohen_d"] = float(abs(A.mean() - H.mean()) / pooled) if pooled > 0 else np.nan
    if not has_m:
        return o
    # --- member-level statistics (both classes >= 2, model present)
    As = A if na <= CAP else rng.choice(A, CAP, replace=False)
    Hs = H if nh <= CAP else rng.choice(H, CAP, replace=False)
    dmA, dmH = np.abs(m - As), np.abs(m - Hs)
    C = np.abs(As[:, None] - Hs[None, :])                     # apo x holo member distances
    WA = _upper(np.abs(As[:, None] - As[None, :]))
    WH = _upper(np.abs(Hs[:, None] - Hs[None, :]))
    Xs = np.concatenate([As, Hs])
    WX = _upper(np.abs(Xs[:, None] - Xs[None, :]))
    mA, mH = float(dmA.mean()), float(dmH.mean())
    near_A = mA < mH
    o["near"] = "A" if near_A else "H"
    dm_near = min(mA, mH)
    W_near = WA.mean() if near_A else WH.mean()
    o["r_within"] = float(dm_near / W_near) if W_near > 0 else np.nan
    o["r_cross"] = float(dm_near / C.mean()) if C.mean() > 0 else np.nan
    o["r_pooled"] = float(np.concatenate([dmA, dmH]).mean() / WX.mean()) if WX.mean() > 0 else np.nan
    dHA = C.mean(axis=0)   # each holo member's mean distance to the apo members
    dAH = C.mean(axis=1)   # each apo member's mean distance to the holo members
    o["pct_holo_farther_from_apo"] = float(np.mean(dHA > mA))
    o["pct_apo_farther_from_holo"] = float(np.mean(dAH > mH))
    sgn = 1.0 if H.mean() > A.mean() else -1.0
    o["apo_beyond"] = float(np.mean(sgn * A > sgn * m))      # apo conformers more holo-ward than the model
    o["holo_beyond"] = float(np.mean(sgn * H < sgn * m))     # holo conformers more apo-ward than the model
    if w_true > 0:
        o["z_near"] = float(min(abs(m - A.mean()), abs(m - H.mean())) / w_true)
    return o


def worker(job):
    from parsing_functions_DBv2 import sort_dataframe_v2c, relabel_apoholo_by_ions
    from parse_AHoJ3_DBv2b import replace_zero_with_nan
    job_name, tgz, mvals = job
    a = ec._ARGS
    try:
        df = ec._read_global_results(tgz)
    except Exception as e:  # noqa: BLE001
        return {"job_name": job_name, "status": f"error:{type(e).__name__}"}
    if df is None or df.empty:
        return {"job_name": job_name, "status": "not_found"}
    # --- the classifier's curation, verbatim from ensemble_containment.process
    if int(a.relabel_apoholo) == 1:
        df, _ = relabel_apoholo_by_ions(df, builtin="total_le1", save_dir=None, verbose=False)
    df = replace_zero_with_nan(df, a.columns_nozero)
    s = pd.to_numeric(df["#resis3"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")
    df["volume"] = np.where(s.notna() & (s != 0) & v.isna(), 0.0, v)
    df_f = sort_dataframe_v2c(df, sorting=0, full_pockets_aligned=a.full_pockets_aligned,
                              bndgrsds_prcnt_seqres=a.bndgrsds_prcnt_seqres,
                              bndgrsds_prcnt_obs=a.bndgrsds_prcnt_obs, resolution_soft=a.resolution_soft,
                              rog_diff_threshold_deltaQ=a.rog_diff_thrs_query,
                              rog_diff_threshold_mean=a.rog_diff_thrs_mean, pocket_distance=a.pocket_dist,
                              pocket_len_resis3=a.pocket_len_resis3,
                              drop_no_resolution=a.drop_no_resolution)
    df_c = df_f[df_f["role"] == "C"]
    apo = df_c[df_c["apoholo_assignment"].isin(["A", "A*", "A**"])]
    holo = df_c[df_c["apoholo_assignment"].isin(["H", "H*"])]
    rec = {"job_name": job_name, "status": "ok", "n_apo": int(len(apo)), "n_holo": int(len(holo))}
    rng = np.random.default_rng(zlib.crc32(job_name.encode("utf-8")))
    for m in SRC:
        if m not in df_c.columns:
            continue
        A = pd.to_numeric(apo[m], errors="coerce").dropna().to_numpy(float)
        H = pd.to_numeric(holo[m], errors="coerce").dropna().to_numpy(float)
        mv = mvals.get(m, np.nan)
        for k, val in entry_stats(A, H, mv, rng).items():
            rec[f"{m}_{k}"] = val
        if m == "volume":
            rec["volume_n_a0"], rec["volume_n_h0"] = int((A == 0).sum()), int((H == 0).sum())
            rec["volume_model_zero"] = bool(np.isfinite(mv) and mv == 0)
            mv_nz = np.nan if (not np.isfinite(mv) or mv == 0) else mv
            for k, val in entry_stats(A[A > 0], H[H > 0], mv_nz, rng).items():
                rec[f"volume_nz_{k}"] = val
    return rec


# ----------------------------------------------------------------------------------------------- collect

def job_list(limit=None):
    uni = pd.read_csv(ec.UNI, usecols=["threshold", "rep_job", "query_path"], low_memory=False)
    uni = uni[uni.threshold == "UNCLUSTERED"].rename(columns={"rep_job": "job_name"})
    qs = pd.read_csv(ec.SUMMARY, usecols=["job_name", "model_pocket_volume", "model_pocket_sasa",
                                          "model_pocket_RoG"], low_memory=False)
    mr = pd.read_csv(MODEL, usecols=["job_name", "model_pocket_rms"], low_memory=False)
    uni = uni.merge(qs, on="job_name", how="left").merge(mr, on="job_name", how="left")
    jobs = []
    for r in uni.itertuples(index=False):
        batch = str(r.query_path).replace("\\", "/").split("/")[-2]
        mvals = {"volume": float(r.model_pocket_volume), "sasa": float(r.model_pocket_sasa),
                 "RoG": float(r.model_pocket_RoG), "pocket_rms": float(r.model_pocket_rms)}
        jobs.append((r.job_name, os.path.join(ec.ROOT, batch, f"{r.job_name}.tgz"), mvals))
    if limit:
        jobs = jobs[:limit]
    return jobs


def collect(limit, procs, out=ENTRIES):
    jobs = job_list(limit)
    print(f"analysis-set entries: {len(jobs):,}  procs={procs}", flush=True)
    t0 = time.time()
    rows = []
    with Pool(procs, initializer=ec._init) as pool:
        for i, rec in enumerate(pool.imap_unordered(worker, jobs, chunksize=32), 1):
            rows.append(rec)
            if i % 10000 == 0:
                print(f"  {i:,} done  {time.time() - t0:7.0f} s", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(out, index=False)
    print(f"wrote {out}  rows={len(d):,}  cols={d.shape[1]}  {time.time() - t0:.0f} s")
    print(d.status.value_counts().to_string())


# ----------------------------------------------------------------------------------------------- summarise

ROWS = []


def rec(section, metric, group, stat, value, n=None):
    ROWS.append({"section": section, "metric": metric, "group": group, "statistic": stat,
                 "value": value, "n": n})


def pct(x):
    x = pd.Series(x).dropna()
    return 100 * float(x.astype(float).mean()) if len(x) else np.nan


def load_entries(entries=ENTRIES):
    d = pd.read_parquet(entries)
    d = d[d.status == "ok"].copy()
    uni = pd.read_csv(ec.UNI, usecols=["threshold", "rep_job", "pocket_class", "pocket_overlap_pair",
                                       "pocket_density_pair"], low_memory=False)
    uni = uni[uni.threshold == "UNCLUSTERED"].rename(columns={"rep_job": "job_name"})
    uni["separation"] = 1 - uni.pocket_overlap_pair
    d = d.merge(uni[["job_name", "pocket_class", "separation", "pocket_density_pair"]], on="job_name", how="left")
    d["tier"] = pd.cut(d.separation, TIER_EDGES, labels=TIER_LAB)
    return d


def _tier_table(s, cols, title, labels):
    g = s.groupby("tier", observed=True)
    t = pd.DataFrame({lab: g[c].apply(pct) if c not in ("n",) else g.size() for c, lab in zip(cols, labels)})
    t.insert(0, "n", g.size())
    print(f"  {title}\n" + t.round(1).to_string())
    return t


def summarise(entries=ENTRIES):
    d = load_entries(entries)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"entries ok: {len(d):,}")

    # ===================================================================================== 0. fidelity
    print("\n" + "=" * 100 + "\n0. FIDELITY")
    s = pd.read_csv(ec.SUMMARY, usecols=["job_name", "n_apo", "n_holo"], low_memory=False)
    mm = d.merge(s, on="job_name", how="left", suffixes=("", "_cls"))
    both = (mm.n_apo == mm.n_apo_cls) & (mm.n_holo == mm.n_holo_cls)
    print(f"  n_apo / n_holo match the classification summary for {100 * both.mean():.2f}% of {len(mm):,} entries")
    rec("fidelity", "all", "all", "n_apo_n_holo_match_pct", 100 * both.mean(), len(mm))
    # SI Table S7 (ensemble_containment_summary.csv): pooled containment at full range
    cs = pd.read_csv(CONTAIN_CSV)
    print("  SI Table S7 reproduction (pooled over structures, full min-max ranges): this run | committed")
    for m in SRC:
        sub = d[(d[f"{m}_n_a"] >= 2) & (d[f"{m}_n_h"] >= 2)]
        a_in = 100 * np.average(sub[f"{m}_a_in_h_100"], weights=sub[f"{m}_n_a"])
        h_in = 100 * np.average(sub[f"{m}_h_in_a_100"], weights=sub[f"{m}_n_h"])
        ca = cs[(cs.metric == m) & (cs.statistic == "apo state inside holo min-max")].pooled_pct.iloc[0]
        ch = cs[(cs.metric == m) & (cs.statistic == "holo state inside apo min-max")].pooled_pct.iloc[0]
        inter = pct(sub[f"{m}_interp_100"])
        print(f"    {m:10s} apo-in-holo {a_in:5.1f} | {ca:5.1f}   holo-in-apo {h_in:5.1f} | {ch:5.1f}   "
              f"interpenetrate {inter:5.1f}  (n = {len(sub):,}; S7 denominators differ: n_holo >= 2 alone for apo-in-holo)")
        rec("fidelity", m, "both>=2", "apo_in_holo_pooled_pct", a_in, len(sub))
        rec("fidelity", m, "both>=2", "holo_in_apo_pooled_pct", h_in, len(sub))
        rec("fidelity", m, "both>=2", "interpenetrate_pct", inter, len(sub))
    # SI Table S5 (frontier_final_table.txt)
    print("  SI Table S5 reproduction (class-mean frontier; pooled bands): this run | committed")
    ft = {}
    if os.path.exists(FRONTIER_TXT):
        for line in open(FRONTIER_TXT, encoding="utf-8"):
            p = line.split()
            if len(p) >= 9 and p[0] in ("volume", "sasa", "RoG", "pocket_rms"):
                ft[p[0]] = dict(frontier=float(p[1]), n=int(p[2]), inside=float(p[3]), pm2=float(p[5]), pm1=float(p[6]))
    for m in SRC:
        fr = d[f"{m}_frontier_mean"].dropna()
        inm, i1, i2 = d[f"{m}_inside_minmax"].dropna(), d[f"{m}_inside_1sd"].dropna(), d[f"{m}_inside_2sd"].dropna()
        c = ft.get(m, {})
        print(f"    {m:10s} frontier {100 * fr.astype(float).mean():5.1f} (n {len(fr):,}) | {c.get('frontier', np.nan):5.1f} (n {c.get('n', 0):,})"
              f"   inside min-max {100 * inm.astype(float).mean():5.1f} | {c.get('inside', np.nan):5.1f}"
              f"   +-1 SD {100 * i1.astype(float).mean():5.1f} | {c.get('pm1', np.nan):5.1f}"
              f"   +-2 SD {100 * i2.astype(float).mean():5.1f} | {c.get('pm2', np.nan):5.1f}")
        rec("fidelity", m, "model", "frontier_mean_pct", 100 * fr.astype(float).mean(), len(fr))
        rec("fidelity", m, "model", "inside_minmax_pct", 100 * inm.astype(float).mean(), len(inm))
        rec("fidelity", m, "model", "inside_1sd_pct", 100 * i1.astype(float).mean(), len(i1))
    # the classifier's own Cohen's d (stats table) against d' here
    st = pd.read_csv(STATS, usecols=["job_name"] + [f"{m}_cohen_d" for m in SRC], low_memory=False)
    st = st.rename(columns={f"{m}_cohen_d": f"{m}_cohen_d_stats" for m in SRC})
    ms = d.merge(st, on="job_name", how="left")
    print("  the stats table's own |Cohen's d| (n-weighted pooled SD) re-derived here, and against d' (RMS of the class SDs):")
    from scipy.stats import spearmanr
    for m in SRC:
        mine, theirs = ms[f"{m}_cohen_d"], ms[f"{m}_cohen_d_stats"].abs()
        ok = mine.notna() & theirs.notna() & np.isfinite(mine) & np.isfinite(theirs)
        diff = (mine[ok] - theirs[ok]).abs()
        x = ms[f"{m}_dprime_100"]
        ok2 = x.notna() & theirs.notna() & np.isfinite(x) & np.isfinite(theirs)
        rho = spearmanr(x[ok2], theirs[ok2]).correlation
        print(f"    {m:10s} |cohen_d| re-derived vs stats table: max |diff| {diff.max():.4f}, within 0.001 for {100 * (diff < 1e-3).mean():.2f}% (n {int(ok.sum()):,}); "
              f"d' vs |cohen_d|: Spearman rho {rho:+.3f}, medians {x[ok2].median():.3f} | {theirs[ok2].median():.3f} "
              f"(difference of medians {abs(x[ok2].median() - theirs[ok2].median()):.3f})")
        rec("fidelity", m, "stats_table", "cohen_d_within_0.001_pct", 100 * (diff < 1e-3).mean(), int(ok.sum()))
        rec("fidelity", m, "stats_table", "rho_dprime_vs_cohen_d", rho, int(ok2.sum()))
        rec("fidelity", m, "stats_table", "cohen_d_median", theirs[ok2].median(), int(ok2.sum()))

    # ===================================================================================== 1. volume zeros
    print("\n" + "=" * 100 + "\n1. VOLUME ZEROS (parse-ahoj's v2c rule: missing PyVOL volume with pocket residues -> 0.0)")
    any_a0, any_h0 = d.volume_n_a0 > 0, d.volume_n_h0 > 0
    all_a0 = (d.volume_n_a0 == d.volume_n_a) & (d.volume_n_a > 0)
    all_h0 = (d.volume_n_h0 == d.volume_n_h) & (d.volume_n_h > 0)
    tot0 = d.volume_n_a0.sum() + d.volume_n_h0.sum()
    tot = d.volume_n_a.sum() + d.volume_n_h.sum()
    print(f"  entries with a zero volume among apo conformers {pct(any_a0):.1f}%, among holo {pct(any_h0):.1f}%; "
          f"apo class all zero {pct(all_a0):.1f}%, holo class all zero {pct(all_h0):.1f}%")
    print(f"  structures with zero volume: {100 * tot0 / tot:.1f}% of {tot:,} apo + holo conformers; "
          f"model volume zero in {pct(d.volume_model_zero):.1f}% of pockets")
    rec("volume_zeros", "volume", "entries", "any_zero_apo_pct", pct(any_a0), len(d))
    rec("volume_zeros", "volume", "entries", "any_zero_holo_pct", pct(any_h0), len(d))
    rec("volume_zeros", "volume", "entries", "all_zero_apo_pct", pct(all_a0), len(d))
    rec("volume_zeros", "volume", "entries", "all_zero_holo_pct", pct(all_h0), len(d))
    rec("volume_zeros", "volume", "structures", "zero_share_pct", 100 * tot0 / tot, int(tot))
    rec("volume_zeros", "volume", "model", "zero_pct", pct(d.volume_model_zero), len(d))
    maj = pd.Series(np.select([all_a0 & all_h0, all_a0, all_h0], ["both classes all zero", "apo all zero", "holo all zero"],
                              "a measured volume in both"), index=d.index)
    ct = pd.crosstab(maj, d.volume_model_zero, normalize="index") * 100
    print("  model volume zero (%) by the classes' zero pattern:\n" + ct.round(1).to_string())
    ct2 = pd.crosstab(maj, d.pocket_class, normalize="index") * 100
    print("  AlphaFold class (%) by the classes' zero pattern:\n" + ct2.round(1).to_string())
    print(f"  class-mean frontier for volume: zeros kept {pct(d.volume_frontier_mean):.1f}% (n {d.volume_frontier_mean.notna().sum():,}); "
          f"zeros dropped {pct(d.volume_nz_frontier_mean):.1f}% (n {d.volume_nz_frontier_mean.notna().sum():,})")
    rec("volume_zeros", "volume", "model", "frontier_mean_pct", pct(d.volume_frontier_mean), int(d.volume_frontier_mean.notna().sum()))
    rec("volume_zeros", "volume_nz", "model", "frontier_mean_pct", pct(d.volume_nz_frontier_mean), int(d.volume_nz_frontier_mean.notna().sum()))

    # ===================================================================================== 2. robustness
    print("\n" + "=" * 100 + "\n2. RANGE STATISTICS UNDER TRIMMED CENTRAL INTERVALS (entries with >= 2 apo and >= 2 holo; % of entries)")
    for m in OUTM:
        s = d[(d[f"{m}_n_a"] >= 2) & (d[f"{m}_n_h"] >= 2)]
        c = s.dropna(subset=[f"{m}_dprime_{q}" for q in QS] + [f"{m}_dprime_robust"])
        print(f"\n--- {LABEL[m]}  n = {len(s):,}; common denominator for d' n = {len(c):,} ---")
        t = {}
        for q in QS:
            col = {"interpenetrate %": pct(s[f"{m}_interp_{q}"]),
                   "disjoint %": pct(s[f"{m}_disjoint_{q}"]),
                   "apo in holo interval, pooled %": 100 * np.average(s[f"{m}_a_in_h_{q}"], weights=s[f"{m}_n_a"]),
                   "holo in apo interval, pooled %": 100 * np.average(s[f"{m}_h_in_a_{q}"], weights=s[f"{m}_n_h"]),
                   "interval Jaccard, median": s[f"{m}_jac_{q}"].median(),
                   "d' after trimming, median (common n)": c[f"{m}_dprime_{q}"].median(),
                   "share d' < 1 % (common n)": pct(c[f"{m}_dprime_{q}"] < 1),
                   "share d' >= 2 % (common n)": pct(c[f"{m}_dprime_{q}"] >= 2)}
            t[f"central {q}%"] = col
            for k, v in col.items():
                rec("robustness", m, f"central_{q}", k, v, len(c) if "common" in k else len(s))
        print(pd.DataFrame(t).round(2).to_string())
        sh_i = t["central 100%"]["interpenetrate %"] - t["central 95%"]["interpenetrate %"]
        sh_d = t["central 95%"]["disjoint %"] - t["central 100%"]["disjoint %"]
        sh_p = t["central 95%"]["d' after trimming, median (common n)"] - t["central 100%"]["d' after trimming, median (common n)"]
        print(f"  shift full -> central 95%: interpenetrate -{sh_i:.2f} points; disjoint +{sh_d:.2f} points; median d' +{sh_p:.2f}")
        rec("robustness", m, "shift_100_to_95", "interpenetrate_points", -sh_i, len(s))
        rec("robustness", m, "shift_100_to_95", "disjoint_points", sh_d, len(s))
        rec("robustness", m, "shift_100_to_95", "dprime_median", sh_p, len(c))
        rob = c[f"{m}_dprime_robust"]
        print(f"  robust d' (median difference / IQR-based within scale), common n: median {rob.median():.2f}; "
              f"< 1: {pct(rob < 1):.1f}%; >= 2: {pct(rob >= 2):.1f}%")
        rec("robustness", m, "common", "dprime_robust_median", rob.median(), len(c))
        rec("robustness", m, "common", "dprime_robust_lt1_pct", pct(rob < 1), len(c))
        rec("robustness", m, "common", "dprime_robust_ge2_pct", pct(rob >= 2), len(c))

    # ===================================================================================== 3. d'
    print("\n" + "=" * 100 + "\n3. DISTANCE BETWEEN THE STATE CENTRES, d' = |mean_holo - mean_apo| / within-state SD (full range)")
    from scipy.stats import spearmanr as _sp
    for m in OUTM:
        s = d[(d[f"{m}_n_a"] >= 2) & (d[f"{m}_n_h"] >= 2)].copy()
        dp = s[f"{m}_dprime_100"]
        ok = dp.notna() & np.isfinite(dp)
        n_undef = int((~ok).sum())
        s, dp = s[ok], dp[ok]
        q = dp.quantile([.1, .25, .5, .75, .9])
        print(f"\n--- {LABEL[m]}  n = {len(s):,} (d' undefined, no within-state spread, for {n_undef:,} of the entries with >= 2 of each state) ---")
        rec("dprime", m, "all", "n_undefined", n_undef, len(s) + n_undef)
        print("  d' quantiles: p10 {:.2f} p25 {:.2f} median {:.2f} p75 {:.2f} p90 {:.2f}".format(*q))
        print(f"  share d' < 0.5: {pct(dp < .5):.1f}%  < 1: {pct(dp < 1):.1f}%  < 2: {pct(dp < 2):.1f}%  >= 2: {pct(dp >= 2):.1f}%  >= 3: {pct(dp >= 3):.1f}%")
        print(f"  natural units: median |mean_holo - mean_apo| {s[f'{m}_between'].median():.4g}; median within-state SD {s[f'{m}_within_sd'].median():.4g}; "
              f"median apo range {(s[f'{m}_a_max'] - s[f'{m}_a_min']).median():.4g}, holo range {(s[f'{m}_h_max'] - s[f'{m}_h_min']).median():.4g}")
        for k, v in (("p10", q.iloc[0]), ("p25", q.iloc[1]), ("median", q.iloc[2]), ("p75", q.iloc[3]), ("p90", q.iloc[4]),
                     ("lt0.5_pct", pct(dp < .5)), ("lt1_pct", pct(dp < 1)), ("lt2_pct", pct(dp < 2)), ("ge2_pct", pct(dp >= 2)), ("ge3_pct", pct(dp >= 3)),
                     ("between_median", s[f"{m}_between"].median()), ("within_sd_median", s[f"{m}_within_sd"].median())):
            rec("dprime", m, "all", k, v, len(s))
        s["nmin"] = pd.cut(np.minimum(s[f"{m}_n_a"], s[f"{m}_n_h"]), NMIN_EDGES, labels=NMIN_LAB)
        g = s.groupby("nmin", observed=True)
        t = pd.DataFrame({"n": g.size(), "median d'": g[f"{m}_dprime_100"].median(),
                          "d' < 1 %": g[f"{m}_dprime_100"].apply(lambda x: pct(x < 1)),
                          "disjoint %": g[f"{m}_disjoint_100"].apply(pct),
                          "interpenetrate %": g[f"{m}_interp_100"].apply(pct)})
        print("  by min(n_apo, n_holo):\n" + t.round(2).to_string())
        for idx, r in t.iterrows():
            rec("dprime", m, f"nmin_{idx}", "median", r["median d'"], int(r["n"]))
            rec("dprime", m, f"nmin_{idx}", "lt1_pct", r["d' < 1 %"], int(r["n"]))
            rec("dprime", m, f"nmin_{idx}", "disjoint_pct", r["disjoint %"], int(r["n"]))
        g = s.groupby("tier", observed=True)
        t = pd.DataFrame({"n": g.size(), "median d'": g[f"{m}_dprime_100"].median(),
                          "d' < 1 %": g[f"{m}_dprime_100"].apply(lambda x: pct(x < 1)),
                          "d' >= 2 %": g[f"{m}_dprime_100"].apply(lambda x: pct(x >= 2))})
        print("  by separation tier:\n" + t.round(2).to_string())
        for idx, r in t.iterrows():
            rec("dprime", m, f"tier_{idx}", "median", r["median d'"], int(r["n"]))
        oks = s.separation.notna()
        rho = _sp(dp[oks], s.separation[oks]).correlation
        rho_j = _sp(s[f"{m}_jac_100"][oks], s.separation[oks]).correlation
        print(f"  Spearman rho(d', separation) = {rho:+.3f}; rho(range Jaccard, separation) = {rho_j:+.3f}")
        rec("dprime", m, "all", "rho_dprime_separation", rho, int(oks.sum()))
        rec("dprime", m, "all", "rho_jaccard_separation", rho_j, int(oks.sum()))
    # conjunctions across the four published metrics
    cols = [f"{m}_dprime_100" for m in SRC]
    D = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n  all four published metrics at once (n = {len(D):,}): d' < 1 on all four {pct((D < 1).all(axis=1)):.1f}%; "
          f"d' >= 2 on all four {pct((D >= 2).all(axis=1)):.1f}%; d' >= 2 on at least one {pct((D >= 2).any(axis=1)):.1f}%; "
          f"median of the four-metric mean d' {D.mean(axis=1).median():.2f}")
    rec("dprime", "all4", "complete", "lt1_all_pct", pct((D < 1).all(axis=1)), len(D))
    rec("dprime", "all4", "complete", "ge2_all_pct", pct((D >= 2).all(axis=1)), len(D))
    rec("dprime", "all4", "complete", "ge2_any_pct", pct((D >= 2).any(axis=1)), len(D))
    rec("dprime", "all4", "complete", "mean_dprime_median", D.mean(axis=1).median(), len(D))

    # ===================================================================================== 4. member-level frontier
    print("\n" + "=" * 100 + "\n4. THE MODEL AGAINST MEMBER-TO-MEMBER DISTANCES (>= 2 of each state, model present)")
    for m in OUTM:
        s = d[(d[f"{m}_n_a"] >= 2) & (d[f"{m}_n_h"] >= 2) & (d[f"{m}_has_model"] == True)].copy()  # noqa: E712
        s = s[s[f"{m}_r_cross"].notna()]
        print(f"\n--- {LABEL[m]}  n = {len(s):,} ---")
        print(f"  class-mean frontier (SI Table S5 criterion) on this set: {pct(s[f'{m}_frontier_mean']):.1f}%")
        rec("member_frontier", m, "all", "frontier_mean_pct", pct(s[f"{m}_frontier_mean"]), len(s))
        print("  model -> nearer state, relative to ...                         median ratio   model is the closer one")
        for col, lab in (("r_pooled", "all structures pairwise (Lane's comparison, pooled)"),
                         ("r_within", "that state's members among themselves (within-state)"),
                         ("r_cross", "apo <-> holo member distances (cross-state)")):
            x = s[f"{m}_{col}"]
            print(f"    {lab:58s} {x.median():6.2f}        {pct(x < 1):5.1f}%")
            rec("member_frontier", m, "all", f"{col}_median", x.median(), len(s))
            rec("member_frontier", m, "all", f"{col}_lt1_pct", pct(x < 1), len(s))
        # side-taking
        near_H = s[f"{m}_near"] == "H"
        pa, ph = s[f"{m}_pct_holo_farther_from_apo"], s[f"{m}_pct_apo_farther_from_holo"]
        s["closer_near"] = np.where(near_H, ph >= .5, pa >= .5)
        s["closer_near_all"] = np.where(near_H, ph == 1, pa == 1)
        s["closer_apo"], s["closer_holo"] = pa >= .5, ph >= .5
        s["dbin"] = pd.cut(s[f"{m}_dprime_100"], [-1, 1, 2, 1e9], labels=["d' < 1", "1-2", "d' >= 2"])
        s["nmin"] = pd.cut(np.minimum(s[f"{m}_n_a"], s[f"{m}_n_h"]), NMIN_EDGES, labels=NMIN_LAB)
        s["tcat"] = pd.cut(s[f"{m}_t_pos"], [-1e18, 0, 1, 1e18], labels=["beyond apo", "between centres", "beyond holo"])
        print(f"  model closer to APO than the median HOLO conformer is: {pct(s.closer_apo):.1f}% (than every holo conformer {pct(pa == 1):.1f}%); "
              f"closer to HOLO than the median APO conformer is: {pct(s.closer_holo):.1f}% (than every apo conformer {pct(ph == 1):.1f}%)")
        print(f"  model closer to its NEARER state than the median opposite-state conformer: {pct(s.closer_near):.1f}%; "
              f"than EVERY opposite-state conformer: {pct(s.closer_near_all):.1f}%")
        rec("side", m, "all", "closer_apo_pct", pct(s.closer_apo), len(s))
        rec("side", m, "all", "closer_holo_pct", pct(s.closer_holo), len(s))
        rec("side", m, "all", "closer_near_pct", pct(s.closer_near), len(s))
        rec("side", m, "all", "closer_near_all_pct", pct(s.closer_near_all), len(s))
        for gcol, gname in (("tier", "separation tier"), ("dbin", "d'"), ("nmin", "min(n_apo, n_holo)")):
            g = s.groupby(gcol, observed=True)
            t = pd.DataFrame({"n": g.size(),
                              "within < 1 %": g[f"{m}_r_within"].apply(lambda x: pct(x < 1)),
                              "cross < 1 %": g[f"{m}_r_cross"].apply(lambda x: pct(x < 1)),
                              "pooled < 1 %": g[f"{m}_r_pooled"].apply(lambda x: pct(x < 1)),
                              "closer_near %": g["closer_near"].apply(pct),
                              "closer_near_all %": g["closer_near_all"].apply(pct),
                              "class-mean frontier %": g[f"{m}_frontier_mean"].apply(pct)})
            print(f"  by {gname}:\n" + t.round(1).to_string())
            for idx, r in t.iterrows():
                for c in t.columns[1:]:
                    rec("side", m, f"{gcol}_{idx}", c, r[c], int(r["n"]))
        print("  position on the apo -> holo axis (0 = apo centre, 1 = holo centre): median t = {:.2f}; ".format(s[f"{m}_t_pos"].median())
              + ", ".join(f"{k} {100 * v:.1f}%" for k, v in s.tcat.value_counts(normalize=True).items()))
        rec("side", m, "all", "t_pos_median", s[f"{m}_t_pos"].median(), len(s))
        for k, v in s.tcat.value_counts(normalize=True).items():
            rec("side", m, "all", f"tcat_{k}_pct", 100 * v, len(s))
        ct = 100 * pd.crosstab(s.dbin, s.tcat, normalize="index")
        print("    by d':\n" + ct.round(1).to_string())
        for idx, r in ct.iterrows():
            for c in ct.columns:
                rec("side", m, f"dbin_{idx}", f"tcat_{c}_pct", r[c])
        ct = 100 * pd.crosstab(s.pocket_class, s.tcat, normalize="index")
        print("    by AlphaFold class:\n" + ct.round(1).to_string())
        for idx, r in ct.iterrows():
            for c in ct.columns:
                rec("side", m, f"class_{idx}", f"tcat_{c}_pct", r[c])
        hi = s[s.dbin == "d' >= 2"]
        ct = 100 * pd.crosstab(hi.pocket_class, hi.tcat, normalize="index")
        print(f"    by AlphaFold class, d' >= 2 only (n = {len(hi):,}; AF-Apo {int((hi.pocket_class == 'apo').sum()):,}, AF-Holo {int((hi.pocket_class == 'holo').sum()):,}):\n" + ct.round(1).to_string())
        for idx, r in ct.iterrows():
            for c in ct.columns:
                rec("side", m, f"class_{idx}_dge2", f"tcat_{c}_pct", r[c])
        g = s.groupby("dbin", observed=True)
        t = pd.DataFrame({"n": g.size(), "apo conformers more holo-ward than the model (median share)": g[f"{m}_apo_beyond"].median(),
                          "holo conformers more apo-ward than the model (median share)": g[f"{m}_holo_beyond"].median()})
        print("  member-wise depth by d':\n" + t.round(2).to_string())
        for idx, r in t.iterrows():
            rec("side", m, f"dbin_{idx}", "apo_beyond_median", r.iloc[1], int(r["n"]))
            rec("side", m, f"dbin_{idx}", "holo_beyond_median", r.iloc[2], int(r["n"]))
        # the within-SD distance to the nearer centre against the released density descriptor
        okz = s[f"{m}_z_near"].notna() & s.pocket_density_pair.notna()
        rho = _sp(s[f"{m}_z_near"][okz], s.pocket_density_pair[okz]).correlation
        print(f"  Spearman rho(distance to the nearer centre in within-state SD, density) = {rho:+.3f} (n {int(okz.sum()):,})")
        rec("side", m, "all", "rho_znear_density", rho, int(okz.sum()))

    pd.DataFrame(ROWS).to_csv(SUMMARY_CSV, index=False)
    print(f"\nwrote {SUMMARY_CSV}  rows={len(ROWS):,}")


# ----------------------------------------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--procs", type=int, default=10)
    c.add_argument("--out", default=ENTRIES)
    s_ = sub.add_parser("summarise")
    s_.add_argument("--entries", default=ENTRIES)
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(args.limit, args.procs, args.out)
    else:
        summarise(args.entries)
