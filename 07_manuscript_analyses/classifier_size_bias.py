# -*- coding: utf-8 -*-
"""Does the released classifier's P(holo) follow the apo/holo COUNT imbalance by itself?
Backs Supplementary Note 3 ("How large it is without AlphaFold", "How large it is on real sites and
models", and the draw-noise re-draw read by balanced_classification.py sites for Table S10).

Three measurements, in increasing order of realism, run in the order shown at the bottom of this
docstring.

  simulate    Pure null, using the classifier's own kde_core. Four independent metrics are drawn for
              the apo set, the holo set and the "model" from ONE distribution (identical), or with
              the two classes shifted apart by d SD and the model drawn from one of them (true_apo /
              true_holo). Nothing about AlphaFold is involved, so any movement of the call with the
              class sizes is made by the estimator.

  subsample   Real entries, real AlphaFold models. For sites with at least 20 apo and 20 holo
              structures, the model and the site are held FIXED and only the reference-set sizes are
              subsampled. Any movement of P(holo) is again the estimator, because the model that is
              being classified never changes.

  downstream  Site level. Re-classify every member pocket of sampled ST_100 detection sites with the
              size-balanced comparison (balanced_classification.py explains it), aggregate to the
              site the way the pipeline does (mean member P(holo), larger mean wins), and compare
              the detection and docking class gaps under the released and the balanced labels, raw,
              size-standardized, and standardized for size and evidence ratio together.

Size standardization is direct adjustment to the set's union_rg decile mix, as in
density_separation_gaps.py, so the gaps are comparable to Supplementary Table S3. "Size" there means POCKET size;
the imbalance measured here is in the number of reference STRUCTURES, a different axis.

    python classifier_size_bias.py simulate   [--reps 400]                  -> size_bias_simulation.csv
    python classifier_size_bias.py subsample  [--sites 1500] [--procs 22]   -> size_bias_subsample.csv
    python classifier_size_bias.py downstream [--sites 16000] [--procs 22]  -> size_bias_downstream.csv

    python classifier_size_bias.py simulate > out.txt
    python classifier_size_bias.py subsample >> out.txt
    python classifier_size_bias.py downstream >> out.txt
    python balanced_classification.py summarise >> out.txt
    python balanced_classification.py sites >> out.txt

`downstream` samples sites and re-classifies their members on the fly. The whole-set, site-level
version of the same comparison, with Figure 4's intervals, is `balanced_classification.py sites`,
which reads the per-member re-draw `downstream` leaves (downstream_members.csv) to measure draw noise.
All files go to the manuscript-analyses folder (ma_paths.OUT).
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import balanced_classification as B  # noqa: E402  (paths, curation, classifier settings, _balanced)

W = B.WORK
LEVELS = ["very_low", "low", "moderate", "high", "very_high"]
SIM_CSV = P.out("size_bias_simulation.csv")
SUB_CSV = P.out("size_bias_subsample.csv")
DOWN_CSV = P.out("size_bias_downstream.csv")
CONFIGS = [(1, 20), (2, 20), (5, 20), (20, 20), (20, 5), (20, 2), (20, 1)]
SUB_REPS = 6


# --------------------------------------------------------------------------------------- simulate
def _sim_once(n_apo, n_holo, d, model_from, rng, n_metrics=4):
    from kde_core import KDEConfig, Priors, MetricData, build_kde, classify_metric
    cfg = KDEConfig(bw=0.01, bw_mode="pooled", bw_scale=0.75, kde_extend=True,
                    margin_cap_scale=1.0, missing_group_policy="nan")
    pri = Priors(mode="equal")
    ph = []
    for _ in range(n_metrics):
        a = rng.normal(0, 1, n_apo)
        h = rng.normal(d, 1, n_holo)
        m = rng.normal(0 if model_from == "apo" else d, 1)
        md = MetricData(name="m", apo=a, holo=h, combo=np.concatenate([a, h]), model_value=float(m))
        r = classify_metric(md, build_kde(md, cfg), pri)
        if np.isfinite(r.P_holo):
            ph.append(r.P_holo)
    return float(np.mean(ph)) if ph else np.nan


def simulate(reps):
    rng = np.random.default_rng(42)
    pairs = [(1, 50), (2, 40), (3, 30), (5, 25), (5, 10), (10, 10), (10, 5), (25, 5), (30, 3), (40, 2), (50, 1)]
    rows = []
    print("=== simulate: the estimator alone, no AlphaFold ===")
    for scenario, d, src in [("identical", 0.0, "apo"), ("true_apo", 1.5, "apo"), ("true_holo", 1.5, "holo"),
                             ("true_apo", 3.0, "apo"), ("true_holo", 3.0, "holo")]:
        print(f"\n{scenario}, class separation d = {d} SD")
        print(f"{'n_apo':>6} {'n_holo':>7} {'log10 ratio':>12} {'mean P(holo)':>13} {'called holo %':>14}")
        for na, nh in pairs:
            v = np.array([_sim_once(na, nh, d, src, rng) for _ in range(reps)])
            v = v[np.isfinite(v)]
            rec = {"scenario": scenario, "d": d, "n_apo": na, "n_holo": nh,
                   "log10_ratio": np.log10(nh / na), "mean_p_holo": v.mean(),
                   "pct_called_holo": 100 * (v > 0.5).mean(), "reps": len(v)}
            rows.append(rec)
            print(f"{na:>6} {nh:>7} {rec['log10_ratio']:>12.2f} {rec['mean_p_holo']:>13.3f} "
                  f"{rec['pct_called_holo']:>14.1f}")
    pd.DataFrame(rows).to_csv(SIM_CSV, index=False)
    print(f"\nwrote {SIM_CSV}")


# -------------------------------------------------------------------------------------- subsample
def _work_sub(job):
    job_name, tgz, seed = job
    try:
        loaded = B._load(tgz)
        if loaded is None:
            return None
        df_c, apo, holo, df_af = loaded
        if df_af.empty:
            return None
        rng = np.random.default_rng(seed)
        m = df_af.iloc[:1]
        rec = {"job_name": job_name, "n_apo_pool": len(apo), "n_holo_pool": len(holo),
               "p_full": B._one(B._classify(df_c, apo, holo, m))["p_holo"]}
        for na, nh in CONFIGS:
            if len(apo) < na or len(holo) < nh:
                continue
            vals = []
            for _ in range(SUB_REPS):
                a = apo.iloc[rng.choice(len(apo), na, replace=False)]
                h = holo.iloc[rng.choice(len(holo), nh, replace=False)]
                vals.append(B._one(B._classify(pd.concat([a, h]), a, h, m))["p_holo"])
            rec[f"p_{na}_{nh}"] = float(np.nanmean(vals))
        return rec
    except Exception as e:  # noqa: BLE001
        return {"job_name": job_name, "error": f"{type(e).__name__}: {e}"}


def subsample(n_sites, procs):
    s = pd.read_csv(B.SUMMARY, usecols=["job_name", "query_path", "n_apo", "n_holo"], low_memory=False)
    uni = pd.read_csv(B.UNI, usecols=["threshold", "rep_job"], low_memory=False)
    s = s[s.job_name.isin(set(uni.loc[uni.threshold == "UNCLUSTERED", "rep_job"]))]
    pool = s[(s.n_apo >= 20) & (s.n_holo >= 20)]
    sel = pool.sample(n=min(n_sites, len(pool)), random_state=7)
    jobs = [(j, os.path.join(B.ROOT, str(q).replace("\\", "/").split("/")[-2], f"{j}.tgz"), 60_000 + i)
            for i, (j, q) in enumerate(zip(sel.job_name, sel.query_path))]
    print(f"=== subsample: same site, same model, only the reference counts change ===")
    print(f"well-sampled sites (>=20 apo and >=20 holo): {len(pool):,}; sampled {len(jobs):,}")
    t0 = time.time()
    with Pool(procs, initializer=B._init) as p:
        rows = [r for r in p.imap_unordered(_work_sub, jobs, chunksize=8) if r]
    d = pd.DataFrame(rows)
    d = d[d.get("error").isna()] if "error" in d else d
    d.to_csv(SUB_CSV, index=False)
    print(f"sites classified: {len(d):,} in {(time.time() - t0) / 60:.1f} min")
    cols = [c for c in d.columns if c.startswith("p_") and c.count("_") == 2]
    print(f"\n{'n_apo:n_holo':>13} {'log10 ratio':>12} {'mean P(holo)':>13} {'called holo %':>14}")
    for c in cols:
        na, nh = map(int, c.split("_")[1:])
        print(f"{na:>6}:{nh:<6} {np.log10(nh / na):>12.2f} {d[c].mean():>13.3f} {100 * (d[c] > 0.5).mean():>14.1f}")
    swing = 100 * ((d["p_1_20"] > 0.5).mean() - (d["p_20_1"] > 0.5).mean())
    flip = 100 * ((d["p_1_20"] > 0.5) != (d["p_20_1"] > 0.5)).mean()
    print(f"\nholo-call swing between 1:20 and 20:1 on the SAME sites and models: {swing:.1f} pp")
    print(f"sites whose call flips: 1:20 vs 20:1 {flip:.1f}%; 2:20 vs 20:2 "
          f"{100 * ((d['p_2_20'] > 0.5) != (d['p_20_2'] > 0.5)).mean():.1f}%; 5:20 vs 20:5 "
          f"{100 * ((d['p_5_20'] > 0.5) != (d['p_20_5'] > 0.5)).mean():.1f}%")
    print(f"wrote {SUB_CSV}")


# ------------------------------------------------------------------------------------- downstream
def _load_sites():
    keep = ["site_uid", "unit_uid", "success_tns", "pocket_p_holo", "pocket_class", "union_rg",
            "separation_level_db", "log2_ratio_holo_to_apo_avg_plus1"]
    p = pd.read_csv(os.path.join(W, "lbs_generality/p2rank/lbs_p2rank_st100_merged_per_pocket.csv"),
                    usecols=keep, low_memory=False).rename(
        columns={"success_tns": "P2Rank", "log2_ratio_holo_to_apo_avg_plus1": "ratio"})
    for tool, name in [("fpocket", "fpocket"), ("ifsitepred", "IF-SitePred")]:
        t = pd.read_csv(os.path.join(W, f"lbs_generality/{tool}/full/lbs_{tool}_st100_merged_per_pocket.csv"),
                        usecols=["site_uid", "success_tns"], low_memory=False)
        p = p.merge(t.rename(columns={"success_tns": name}), on="site_uid", how="left")
    p = p[p[["P2Rank", "fpocket", "IF-SitePred"]].notna().all(axis=1)].copy()
    p["consensus"] = p[["P2Rank", "fpocket", "IF-SitePred"]].mean(axis=1)
    u = pd.read_csv(B.UNI, usecols=["threshold", "unit_uid", "job_names", "members"], low_memory=False)
    p = p.merge(u[u.threshold == "ST_100"].drop(columns="threshold"), on="unit_uid", how="left")
    return p[p.pocket_class.isin(["apo", "holo"])].copy()


def _work_member(job):
    job_name, tgz, seed = job
    try:
        loaded = B._load(tgz)
        if loaded is None:
            return {"job_name": job_name, "error": "no archive"}
        df_c, apo, holo, df_af = loaded
        if df_af.empty or len(apo) < 1 or len(holo) < 1:
            return {"job_name": job_name, "error": "no model / empty class"}
        m = df_af.iloc[:1]
        bal, _ = B._balanced(apo, holo, m, np.random.default_rng(seed))
        return {"job_name": job_name, "p_full": B._one(B._classify(df_c, apo, holo, m))["p_holo"],
                "p_bal": bal.get("bal_p_holo", np.nan)}
    except Exception as e:  # noqa: BLE001
        return {"job_name": job_name, "error": f"{type(e).__name__}: {e}"}


def _gap(d, cls_col, val, strata, w):
    a, h = d[d[cls_col] == "apo"], d[d[cls_col] == "holo"]
    def std(g):
        per = g.groupby(strata, observed=True)[val].mean()
        ww = w.reindex(per.index)
        ok = ww.notna()
        return 100 * float((per[ok] * ww[ok]).sum() / ww[ok].sum())
    return std(h) - std(a)


def downstream(n_sites, procs):
    s = _load_sites()
    pool = s[s.members <= 10]
    sel = pool.sample(n=min(n_sites, len(pool)), random_state=3)
    jobs = sorted({j for js in sel.job_names for j in str(js).split(";")})
    summ = pd.read_csv(B.SUMMARY, usecols=["job_name", "query_path"], low_memory=False)
    summ = summ[summ.job_name.isin(jobs)]
    todo = [(j, os.path.join(B.ROOT, str(q).replace("\\", "/").split("/")[-2], f"{j}.tgz"), 70_000 + i)
            for i, (j, q) in enumerate(zip(summ.job_name, summ.query_path))]
    print("=== downstream: released vs size-balanced labels on the same sites ===")
    print(f"classified detection sites {len(s):,}; with <= 10 member pockets {len(pool):,} "
          f"({100 * len(pool) / len(s):.1f}%); sampled {len(sel):,} sites, {len(todo):,} member pockets")
    t0 = time.time()
    with Pool(procs, initializer=B._init) as p:
        rows = list(p.imap_unordered(_work_member, todo, chunksize=16))
    e = pd.DataFrame(rows)
    e = e[e.get("error").isna()] if "error" in e else e
    print(f"member pockets classified: {len(e):,} in {(time.time() - t0) / 60:.1f} min")
    # an independent re-draw of these pockets (different seeds from `collect`):
    # balanced_classification.py sites reads it back to measure draw noise
    os.makedirs(B.OUT_DIR, exist_ok=True)
    e.to_csv(B.DOWN_MEMBERS, index=False)
    mp = sel[["site_uid", "job_names"]].assign(job_name=sel.job_names.str.split(";")).explode("job_name")
    agg = mp.merge(e, on="job_name", how="left").groupby("site_uid").agg(
        p_full=("p_full", "mean"), p_bal=("p_bal", "mean"), n_ok=("p_full", "count"), n_mem=("job_name", "size"))
    d = sel.merge(agg, on="site_uid", how="left")
    d = d[d.n_ok == d.n_mem].copy()
    print(f"sites with every member classified: {len(d):,}; site P(holo) reproduces the released value for "
          f"{100 * (np.abs(d.p_full - d.pocket_p_holo) < 0.0015).mean():.1f}%")
    d["cls_orig"] = d.pocket_class
    d["cls_bal"] = np.where(d.p_bal > 0.5, "holo", np.where(d.p_bal < 0.5, "apo", "tie"))
    print(f"AF-Holo share: released {100 * (d.cls_orig == 'holo').mean():.1f}%, "
          f"balanced {100 * (d.cls_bal == 'holo').mean():.1f}%; labels changed "
          f"{100 * (d.cls_orig != d.cls_bal).mean():.1f}%")
    d["_zb"] = pd.qcut(d.union_rg, 10, labels=False, duplicates="drop")
    d["_rb"] = pd.cut(d.ratio, [-np.inf, -1, 0, 1, 2, 3, 4, np.inf])
    w_rg = d._zb.value_counts(normalize=True).sort_index()
    w_rgr = d.groupby(["_zb", "_rb"], observed=True).size()
    w_rgr = w_rgr / w_rgr.sum()
    rows = []
    for lab, tag in [("cls_orig", "released"), ("cls_bal", "balanced")]:
        rows.append({"labels": tag, "task": "consensus detection", "tier": "all",
                     "raw": 100 * (d[d[lab] == "holo"].consensus.mean() - d[d[lab] == "apo"].consensus.mean()),
                     "size_std": _gap(d, lab, "consensus", ["_zb"], w_rg),
                     "size_ratio_std": _gap(d, lab, "consensus", ["_zb", "_rb"], w_rgr), "n": len(d)})
        for lv in LEVELS:
            g = d[d.separation_level_db == lv]
            if len(g) > 50:
                rows.append({"labels": tag, "task": "consensus detection", "tier": lv, "raw": np.nan,
                             "size_std": _gap(g, lab, "consensus", ["_zb"], w_rg),
                             "size_ratio_std": np.nan, "n": len(g)})
    dk = pd.read_csv(os.path.join(W, "docking/run3/analysis/docking_run3_merged.csv"),
                     usecols=["site_uid", "ligand_group", "status", "affinity_top", "pose_in_pocket"],
                     low_memory=False)
    dk = dk[(dk.ligand_group == 1) & (dk.status.astype(str).str.lower() == "ok")].copy()
    dk["ok"] = (dk.pose_in_pocket.astype(str).str.lower().isin(["true", "1", "yes"])
                & (pd.to_numeric(dk.affinity_top, errors="coerce") <= -6)).astype(float)
    t = d.merge(dk.groupby("site_uid", as_index=False).ok.mean().rename(columns={"ok": "dockable"}),
                on="site_uid", how="inner")
    t["_zb"] = pd.qcut(t.union_rg, 10, labels=False, duplicates="drop")
    w_t = t._zb.value_counts(normalize=True).sort_index()
    for lab, tag in [("cls_orig", "released"), ("cls_bal", "balanced")]:
        rows.append({"labels": tag, "task": "docking", "tier": "all",
                     "raw": 100 * (t[t[lab] == "holo"].dockable.mean() - t[t[lab] == "apo"].dockable.mean()),
                     "size_std": _gap(t, lab, "dockable", ["_zb"], w_t), "size_ratio_std": np.nan, "n": len(t)})
        for lv in LEVELS:
            g = t[t.separation_level_db == lv]
            if len(g) > 50:
                rows.append({"labels": tag, "task": "docking", "tier": lv, "raw": np.nan,
                             "size_std": _gap(g, lab, "dockable", ["_zb"], w_t),
                             "size_ratio_std": np.nan, "n": len(g)})
    r = pd.DataFrame(rows)
    r.to_csv(DOWN_CSV, index=False)
    pd.set_option("display.width", 200)
    print("\nAF-Holo minus AF-Apo, percentage points")
    print(r.round(1).to_string(index=False))
    print(f"wrote {DOWN_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["simulate", "subsample", "downstream"])
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--sites", type=int, default=None)
    ap.add_argument("--procs", type=int, default=22)
    a = ap.parse_args()
    if a.mode == "simulate":
        simulate(a.reps)
    elif a.mode == "subsample":
        subsample(a.sites or 1500, a.procs)
    else:
        downstream(a.sites or 16000, a.procs)
