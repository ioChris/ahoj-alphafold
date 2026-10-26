# -*- coding: utf-8 -*-
"""Does the AlphaFold pocket sit nearer the peak of its site's experimental distribution than the site's own
conformers do?  Backs Supplementary Table S7, "Typicality" (the model's density rank among its conformers).

The question: the last Results section says the model lies inside the experimental spread
about as often as one of its own conformers does. Would the model's position argue for "data-majority influence"
(the model drawn toward where most of the record sits) rather than "memorization" (recall of one particular
structure)? A copy of one deposited conformer would sit like a typical member, so containment cannot tell the two
apart. What can is the density: an averaging influence puts the model nearer the peak than a typical member.

The published density descriptors (kde_core.classify_metric in 00_classification/kde_classifier/), per metric:
  combined  = height of the pooled apo + holo KDE at the model's value / that KDE's peak on the grid
  strongest = max(apo KDE height / apo peak, holo KDE height / holo peak)
each averaged over the defined metrics (aggregate_overall); the manuscript's "density" is their mean
(`pocket_density_pair`). 1 means the model sits at the peak. The pooled KDE counts every structure, so the
majority state dominates it.

Here each conformer is scored the same way, LEAVING ITSELF OUT: its own kernel is removed from the pooled KDE
(and from its own class's KDE) before the height at its value and the peak are taken. The model is not in the
sample, so the left-out conformer is the fair comparison; the in-sample form is kept to show the difference.
The bandwidth is the full sample's (one value per metric, as the classifier sets it); removing one point would
change it by O(1/n). The model's percentile among its conformers (mid-rank for ties) is uniform across pockets if
the model is a typical member, and piles up near 1 if it sits nearer the peak than its conformers do.

Data and curation: the per-structure values come from the per-job AHoJ archives, read by
balanced_classification._load (the classifier's own relabel, zero-to-NaN, volume rule, sort_dataframe_v2c,
query row added, full_run_3 parameters). The KDEs are built by the classifier's own build_kde with the released
settings (pooled bandwidth, 0.75 scale, extended grid, equal priors), and the model's recomputed densities and
P(holo) are checked against the released classification summary.

Targets: up to CAP structures per entry (seeded), the same structures on every metric, so a conformer's
four-metric average is taken over the same metrics as the model's.

    python density_percentile.py collect [--limit N] [--procs 22]   -> density_percentile_entries.parquet
    python density_percentile.py summarise [--state-distance PATH] > density_percentile.out.txt
                                                                         -> density_percentile_summary.csv
(--state-distance: the per-entry parquet of apo_holo_state_distance.py collect, for the position block.)
All files are in the manuscript-analyses folder (ma_paths.OUT). The per-entry table (~238k rows) is regenerable;
the summary CSV and the stdout are the published record.
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import argparse
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import balanced_classification as bc  # noqa: E402  (paths, curation, job list)

ENTRIES = P.out("density_percentile_entries.parquet")
# the model's own volume and pocket pLDDT (the file apo_holo_state_distance.py reads for the model values)
MODEL = os.path.join(bc.DB, "Summaries", "model_metrics", "full_run_3", "run_23082026_180200_do_collect_model_pLDDT",
                     "query_summaries_do_collect_model_pLDDT.csv")
# per-entry table of apo_holo_state_distance.py collect (position t and distance to the nearer centre per metric);
# regenerable. Optional: the position block is skipped when it is absent.
STATE_DIST = P.out("apo_holo_state_distance_entries.parquet")
SUMMARY_CSV = P.out("density_percentile_summary.csv")
METRICS = ["pocket_rms", "sasa", "volume", "RoG"]      # the classifier's order (balanced_classification.METRICS)
LABEL = {"pocket_rms": "pocket RMSD", "sasa": "SASA", "volume": "volume", "RoG": "radius of gyration"}
CAP = 300
TIE = 1e-9
DESC = ["comb", "str", "pair"]
DLAB = {"comb": "combined (pooled apo + holo)", "str": "strongest (own class)", "pair": "density (the published pair)"}
TIER_EDGES = [-np.inf, 0.20, 0.30, 0.60, 0.85, np.inf]
TIER_LAB = ["very low", "low", "moderate", "high", "very high"]
N_EDGES = [0, 5, 10, 25, 100, 300, np.inf]
N_LAB = ["2-5", "6-10", "11-25", "26-100", "101-300", "> 300"]
R_EDGES = [-np.inf, 0.5, 1, 2, 3, np.inf]
R_LAB = ["< 0.5", "0.5-1", "1-2", "2-3", ">= 3"]
PL_EDGES = [-np.inf, 70, 90, 95, np.inf]
PL_LAB = ["< 70", "70-90", "90-95", ">= 95"]
Z_EDGES = [0, 0.25, 0.5, 1, 2, np.inf]
Z_LAB = ["< 0.25", "0.25-0.5", "0.5-1", "1-2", ">= 2"]

_K = None


def _init():
    global _K
    bc._init()
    from kde_core import KDEConfig, Priors, build_kde, classify_metric, prepare_metric_data
    _K = dict(cfg=KDEConfig(bw=0.01, bw_mode="pooled", bw_scale=0.75, kde_extend=True, margin_cap_scale=1.0,
                            missing_group_policy="nan"),
              pri=Priors(mode="equal"), build=build_kde, classify=classify_metric, prep=prepare_metric_data)


# ----------------------------------------------------------------------------------------------- worker

def _phi(d, h):
    return np.exp(-0.5 * (d / h) ** 2) / (h * np.sqrt(2 * np.pi))


def _pct(c, m):
    """Mid-rank percentile of m among the finite values c."""
    c = c[np.isfinite(c)]
    if not c.size or not np.isfinite(m):
        return np.nan
    return float((np.sum(c < m - TIE) + 0.5 * np.sum(np.abs(c - m) <= TIE)) / c.size)


def _loo(vals_t, pool, curve, h, loo, empty=0.0):
    """Normalized height of a KDE at each target value. `pool` is the KDE's own sample, `curve` its values on the
    grid. loo[i] removes target i's kernel first (it is a member of this sample); a target that was the sample's
    only member gets `empty` (0 for a class KDE, which the classifier treats as an empty class; NaN for the pooled
    KDE, which then has no left-out form)."""
    n = pool.size
    f_t = _phi(vals_t[:, None] - pool[None, :], h).mean(axis=1)            # exact height at each target
    peak = np.full(vals_t.size, float(np.nanmax(curve)))
    out = np.where(peak > 1e-12, f_t / np.where(peak > 1e-12, peak, 1.0), 0.0)
    rm = loo & (n >= 2)
    if rm.any():
        x = _K["x"]
        kt = _phi(x[None, :] - vals_t[rm][:, None], h)                     # each target's own kernel on the grid
        c_loo = (n * curve[None, :] - kt) / (n - 1)
        pk = c_loo.max(axis=1)
        f_l = (n * f_t[rm] - _phi(0.0, h)) / (n - 1)
        out[rm] = np.where(pk > 1e-12, f_l / np.where(pk > 1e-12, pk, 1.0), 0.0)
    out[loo & (n < 2)] = empty
    return out


def _work(job):
    job_name, tgz, seed = job
    warnings.simplefilter("ignore", RuntimeWarning)      # nanmean / nanmedian of all-NaN slices
    try:
        loaded = bc._load(tgz)
        if loaded is None:
            return {"job_name": job_name, "error": "no global_results"}
        df_c, apo, holo, df_af = loaded
        if df_af.empty or len(apo) < 1 or len(holo) < 1:
            return {"job_name": job_name, "error": "no model / empty class"}
        mp = df_af.iloc[:1]
        rng = np.random.default_rng(seed)
        ref = pd.concat([apo.assign(_c="A"), holo.assign(_c="H")], ignore_index=True)
        tix = np.sort(rng.choice(len(ref), CAP, replace=False)) if len(ref) > CAP else np.arange(len(ref))
        T = ref.iloc[tix]
        is_a = (T["_c"] == "A").to_numpy()
        rec = {"job_name": job_name, "n_apo": len(apo), "n_holo_pool": len(holo), "n_targets": len(T)}
        M = {d: {} for d in DESC}                  # model, per metric
        C = {d: {} for d in DESC}                  # targets (left out), per metric
        CI = {"comb": {}}                          # targets in-sample (combined only)
        p_holo = {}
        for met in METRICS:
            md = _K["prep"](df_c, met, apo, holo, mp)
            kde = _K["build"](md, _K["cfg"])
            mr = _K["classify"](md, kde, _K["pri"], density_threshold=1e-3)
            cv, h = kde.curves, float(kde.curves.bw_all)
            _K["x"] = cv.x
            defined = np.isfinite(mr.P_holo)
            rec[f"pub_comb_{met}"] = mr.norm_density_combined
            if defined:
                p_holo[met] = mr.P_holo
                mv = md.model_value
                dA, dH, dAll = (float(np.interp(mv, cv.x, c)) for c in (cv.apo, cv.holo, cv.all))
                nA = dA / kde.max_apo if kde.max_apo > 1e-12 else 0.0
                nH = dH / kde.max_holo if kde.max_holo > 1e-12 else 0.0
                M["comb"][met] = dAll / kde.max_all if kde.max_all > 0 else 0.0
                M["str"][met] = max(nA, nH)
                M["pair"][met] = 0.5 * (M["comb"][met] + M["str"][met])
            v = pd.to_numeric(T[met], errors="coerce").to_numpy(float)
            ok = np.isfinite(v)
            rec[f"nt_{met}"] = int(ok.sum())
            comb, strg, comb_in = (np.full(len(T), np.nan) for _ in range(3))
            if ok.any() and md.combo.size:
                vt, la = v[ok], is_a[ok]
                comb[ok] = _loo(vt, md.combo, cv.all, h, np.ones(vt.size, bool), empty=np.nan)
                comb_in[ok] = _loo(vt, md.combo, cv.all, h, np.zeros(vt.size, bool))
                nA_t = np.zeros(vt.size)
                nH_t = np.zeros(vt.size)
                if md.apo.size:
                    nA_t = _loo(vt, md.apo, cv.apo, h, la)
                if md.holo.size:
                    nH_t = _loo(vt, md.holo, cv.holo, h, ~la)
                strg[ok] = np.maximum(nA_t, nH_t)
            C["comb"][met], C["str"][met], CI["comb"][met] = comb, strg, comb_in
            C["pair"][met] = 0.5 * (comb + strg)
            for d in DESC:
                m = M[d].get(met, np.nan)
                c = C[d][met]
                rec[f"m_{d}_{met}"] = m
                rec[f"pct_{d}_{met}"] = _pct(c, m)
                rec[f"cmed_{d}_{met}"] = float(np.nanmedian(c)) if np.isfinite(c).any() else np.nan
            rec[f"pctin_comb_{met}"] = _pct(comb_in, M["comb"].get(met, np.nan))
            rec[f"cmedin_comb_{met}"] = float(np.nanmedian(comb_in)) if np.isfinite(comb_in).any() else np.nan
        rec["p_holo"] = float(np.mean(list(p_holo.values()))) if p_holo else np.nan
        mets = [mt for mt in METRICS if mt in M["comb"]]   # the metrics that are defined for the model
        rec["n_def"] = len(mets)
        if mets:
            for d in DESC:
                m = float(np.mean([M[d][mt] for mt in mets]))
                c = np.nanmean(np.vstack([C[d][mt] for mt in mets]), axis=0) if len(T) else np.array([])
                rec[f"m_{d}_avg"] = m
                rec[f"pct_{d}_avg"] = _pct(c, m)
                rec[f"cmed_{d}_avg"] = float(np.nanmedian(c)) if np.isfinite(c).any() else np.nan
                rec[f"c09_{d}_avg"] = float(np.mean(c[np.isfinite(c)] >= 0.9)) if np.isfinite(c).any() else np.nan
            ci = np.nanmean(np.vstack([CI["comb"][mt] for mt in mets]), axis=0)
            rec["pctin_comb_avg"] = _pct(ci, rec["m_comb_avg"])
            rec["cmedin_comb_avg"] = float(np.nanmedian(ci)) if np.isfinite(ci).any() else np.nan
        return rec
    except Exception as e:  # noqa: BLE001
        return {"job_name": job_name, "error": f"{type(e).__name__}: {e}"}


def collect(limit, procs):
    jobs = bc._jobs(limit)
    print(f"entries: {len(jobs):,}  procs={procs}  -> {ENTRIES}", flush=True)
    t0 = time.time()
    rows = []
    with Pool(procs, initializer=_init) as pool:
        for i, rec in enumerate(pool.imap_unordered(_work, jobs, chunksize=16), 1):
            rows.append(rec)
            if i % 10000 == 0:
                el = time.time() - t0
                print(f"  {i:,} / {len(jobs):,}  {el / 60:5.1f} min elapsed, "
                      f"{(el / i) * (len(jobs) - i) / 60:5.1f} min left", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(ENTRIES, index=False)
    print(f"wrote {ENTRIES}  rows={len(d):,}  {(time.time() - t0) / 60:.1f} min", flush=True)
    if "error" in d:
        print(d["error"].dropna().value_counts().head().to_string(), flush=True)


# ----------------------------------------------------------------------------------------------- summary

def _load_entries(path):
    d = pd.read_parquet(path)
    if "error" in d:
        print(f"entries with an error: {d['error'].notna().sum():,}")
        d = d[d["error"].isna()].copy()
    rel = pd.read_csv(bc.SUMMARY, usecols=["job_name", "n_apo", "n_holo", "pocket_avg_P(holo)",
                                           "pocket_avg_normalized_density_combined",
                                           "pocket_avg_normalized_density_strongest"], low_memory=False)
    uni = pd.read_csv(bc.UNI, usecols=["threshold", "rep_job", "pocket_class", "pocket_overlap_pair",
                                       "pocket_density_pair"], low_memory=False)
    uni = uni[uni.threshold == "UNCLUSTERED"].drop(columns="threshold").rename(columns={"rep_job": "job_name"})
    d = d.merge(rel.rename(columns={"n_apo": "rel_n_apo"}), on="job_name", how="left")
    d = d.merge(uni, on="job_name", how="left")
    mod = pd.read_csv(MODEL, usecols=["job_name", "model_pocket_volume", "model_pocket_pLDDT_avg"], low_memory=False)
    d = d.merge(mod, on="job_name", how="left")
    d["plddt_bin"] = pd.cut(d.model_pocket_pLDDT_avg, PL_EDGES, labels=PL_LAB, right=False)
    d["sep_tier"] = pd.cut(1 - d.pocket_overlap_pair, TIER_EDGES, labels=TIER_LAB)
    d["n_bin"] = pd.cut(d.n_apo + d.n_holo_pool, N_EDGES, labels=N_LAB)
    d["ratio_bin"] = pd.cut(np.abs(np.log2((d.n_holo + 1) / d.rel_n_apo)), R_EDGES, labels=R_LAB, right=False)
    return d


def _dist(x):
    x = x.dropna()
    return dict(n=len(x), median=x.median(), mean=x.mean(), above_half=100 * (x > 0.5).mean(),
                top10=100 * (x >= 0.9).mean(), bottom10=100 * (x <= 0.1).mean())


def summarise(path, state_dist=STATE_DIST):
    d = _load_entries(path)
    rows = []
    print(f"entries: {len(d):,}\n")
    # ---- fidelity
    print("== Fidelity against the released classification summary ==")
    for lab, mine, pub, tol in [("P(holo)", "p_holo", "pocket_avg_P(holo)", 0.0015),
                                ("combined density", "m_comb_avg", "pocket_avg_normalized_density_combined", 0.0015),
                                ("strongest density", "m_str_avg", "pocket_avg_normalized_density_strongest", 0.0015),
                                ("density (pair, unified table)", "m_pair_avg", "pocket_density_pair", 0.0015)]:
        ok = d[[mine, pub]].dropna()
        good = (ok[mine] - ok[pub]).abs() <= tol
        print(f"  {lab:32s} within {tol} for {100 * good.mean():.2f}% of {len(ok):,}  "
              f"(max |diff| {(ok[mine] - ok[pub]).abs().max():.4f})")
        rows.append(dict(block="fidelity", key=lab, value=100 * good.mean(), n=len(ok)))
    for met in METRICS:
        ok = d[[f"m_comb_{met}", f"pub_comb_{met}"]].dropna()
        good = (ok.iloc[:, 0] - ok.iloc[:, 1]).abs() <= 0.0006
        print(f"  per-metric combined, {LABEL[met]:18s} within 0.0006 for {100 * good.mean():.2f}% of {len(ok):,}")
    nt = d.n_targets
    print(f"  targets per entry: median {nt.median():.0f}, capped at {CAP} for {100 * (nt == CAP).mean():.1f}% "
          f"of entries; entries with >= 1 target: {(nt > 0).sum():,}")

    # ---- the model's own densities and the conformers'
    print("\n== The model's density and its conformers' (left out), four-metric averages ==")
    print(f"  {'descriptor':32s} {'model median':>12s} {'conformer median':>17s} {'in-sample':>10s} "
          f"{'model >= 0.9':>13s} {'conformers >= 0.9':>18s}")
    for dd in DESC:
        m, c = d[f"m_{dd}_avg"], d[f"cmed_{dd}_avg"]
        ins = d["cmedin_comb_avg"].median() if dd == "comb" else np.nan
        print(f"  {DLAB[dd]:32s} {m.median():12.3f} {c.median():17.3f} {ins:10.3f} "
              f"{100 * (m >= 0.9).mean():12.1f}% {100 * d[f'c09_{dd}_avg'].mean():17.1f}%")
        rows += [dict(block="levels", key=f"{dd}_model_median", value=m.median(), n=m.notna().sum()),
                 dict(block="levels", key=f"{dd}_conformer_median", value=c.median(), n=c.notna().sum()),
                 dict(block="levels", key=f"{dd}_model_ge09", value=100 * (m >= 0.9).mean(), n=m.notna().sum()),
                 dict(block="levels", key=f"{dd}_conformer_ge09", value=100 * d[f"c09_{dd}_avg"].mean(),
                      n=d[f"c09_{dd}_avg"].notna().sum())]
        diff = (m - c).dropna()
        print(f"  {'':32s} model minus its median conformer: median {diff.median():+.3f}; "
              f"model higher in {100 * (diff > TIE).mean():.1f}%, lower in {100 * (diff < -TIE).mean():.1f}%")
        rows.append(dict(block="levels", key=f"{dd}_model_minus_conformer_median", value=diff.median(), n=len(diff)))

    # ---- percentiles: the test
    print("\n== The model's percentile among its conformers (uniform if it is a typical member: "
          "median 0.50, 50% above half, 10% in the top and bottom tenths) ==")
    print(f"  {'':40s} {'n':>8s} {'median':>7s} {'mean':>6s} {'> 0.5':>7s} {'>= 0.9':>7s} {'<= 0.1':>7s}")

    def line(lab, x, key):
        s = _dist(x)
        print(f"  {lab:40s} {s['n']:8,d} {s['median']:7.3f} {s['mean']:6.3f} {s['above_half']:6.1f}% "
              f"{s['top10']:6.1f}% {s['bottom10']:6.1f}%")
        rows.append(dict(block="pct", key=key, value=s["median"], n=s["n"], mean=s["mean"],
                         above_half=s["above_half"], top10=s["top10"], bottom10=s["bottom10"]))

    for dd in DESC:
        line(f"{DLAB[dd]}, four-metric average", d[f"pct_{dd}_avg"], f"{dd}_avg")
    line("combined, in-sample conformers", d["pctin_comb_avg"], "comb_avg_insample")
    x = d["pct_comb_avg"].dropna()
    print(f"  {'combined: model below every scored conformer':40s} {100 * (x < TIE).mean():.1f}% of entries")
    rows.append(dict(block="pct", key="comb_avg_below_all", value=100 * (x < TIE).mean(), n=len(x)))
    for dd in ("comb", "str"):
        for met in METRICS:
            line(f"{dd}, {LABEL[met]}", d[f"pct_{dd}_{met}"], f"{dd}_{met}")

    # ---- strata
    for col, lab in [("pocket_class", "class"), ("sep_tier", "separation tier"),
                     ("n_bin", "structures per entry (apo + holo)"), ("ratio_bin", "|log2 evidence ratio|")]:
        print(f"\n== By {lab}: combined and published-density percentiles (four-metric average) ==")
        print(f"  {'':12s} {'n':>8s} {'comb median':>11s} {'> 0.5':>7s} {'>= 0.9':>7s} {'pair median':>11s} "
              f"{'> 0.5':>7s} {'model comb':>10s} {'conf comb':>9s}")
        for k, g in d.groupby(col, observed=True):
            a, b = _dist(g.pct_comb_avg), _dist(g.pct_pair_avg)
            print(f"  {str(k):12s} {a['n']:8,d} {a['median']:11.3f} {a['above_half']:6.1f}% {a['top10']:6.1f}% "
                  f"{b['median']:11.3f} {b['above_half']:6.1f}% {g.m_comb_avg.median():10.3f} "
                  f"{g.cmed_comb_avg.median():9.3f}")
            rows.append(dict(block=f"by_{col}", key=str(k), value=a["median"], n=a["n"], above_half=a["above_half"],
                             top10=a["top10"], pair_median=b["median"], pair_above_half=b["above_half"]))

    # ---- volume: the zero spike
    print("\n== Volume: models without a detectable cavity (volume 0) sit on the zero spike with their conformers ==")
    v = d.pct_comb_volume
    print(f"  volume percentile exactly 0.5 (all tied) in {100 * (np.abs(v - 0.5) < TIE).mean():.1f}% of entries")
    for lab, g in (("model volume 0", d[d.model_pocket_volume == 0]), ("model volume > 0", d[d.model_pocket_volume > 0])):
        x = _dist(g.pct_comb_volume)
        print(f"  {lab:18s} n={x['n']:,}  median percentile {x['median']:.3f}  > 0.5 {x['above_half']:.1f}%  "
              f"model density median {g.m_comb_volume.median():.3f}  conformer median {g.cmed_comb_volume.median():.3f}")
        rows.append(dict(block="volume_zero", key=lab, value=x["median"], n=x["n"], above_half=x["above_half"]))

    # ---- pocket pLDDT
    print("\n== By pocket pLDDT: combined and published-density percentiles (four-metric average) ==")
    for k, g in d.groupby("plddt_bin", observed=True):
        a, b = _dist(g.pct_comb_avg), _dist(g.pct_pair_avg)
        print(f"  {str(k):8s} n={a['n']:>8,d}  comb median {a['median']:.3f}  > 0.5 {a['above_half']:.1f}%  "
              f"pair median {b['median']:.3f}  > 0.5 {b['above_half']:.1f}%")
        rows.append(dict(block="by_plddt", key=str(k), value=a["median"], n=a["n"], above_half=a["above_half"],
                         pair_median=b["median"], pair_above_half=b["above_half"]))

    # ---- where the low-density models sit (needs the state-distance table)
    if os.path.exists(state_dist):
        cols = ["job_name"] + [f"{m}_{k}" for m in ("sasa", "RoG", "pocket_rms") for k in ("t_pos", "dprime_100", "z_near")]
        sd = pd.read_parquet(state_dist, columns=cols)
        e = d.merge(sd, on="job_name", how="left")
        print(f"\n== Where the model sits, from {os.path.basename(state_dist)} "
              f"(n = {sd.job_name.isin(d.job_name).sum():,}): per-metric combined percentile by the model's distance "
              "to its nearer state centre (within-state SD) ==")
        for m in ("sasa", "RoG", "pocket_rms"):
            zb = pd.cut(e[f"{m}_z_near"], Z_EDGES, labels=Z_LAB, right=False)
            g = pd.DataFrame({"p": e[f"pct_comb_{m}"], "zb": zb}).dropna()
            tot = len(g)
            parts = []
            for k, h in g.groupby("zb", observed=True):
                below_all = 100 * (h.p < TIE).mean()
                parts.append(f"{k}: {100 * len(h) / tot:.1f}% of pockets, median {h.p.median():.3f}, "
                             f"below every conformer {below_all:.1f}%")
                rows.append(dict(block=f"by_znear_{m}", key=str(k), value=h.p.median(), n=len(h),
                                 share=100 * len(h) / tot, below_all=below_all))
            print(f"  {LABEL[m]:18s} (n {tot:,}) " + "; ".join(parts))
        print("\n  ... and by position on the apo->holo axis (t < 0 beyond the apo centre, 0-1 between, > 1 beyond "
              "holo), by d' bin")
        for m in ("sasa", "RoG", "pocket_rms"):
            t, dp = e[f"{m}_t_pos"], e[f"{m}_dprime_100"]
            pos = pd.cut(t, [-np.inf, 0, 1, np.inf], labels=["beyond apo", "between", "beyond holo"])
            dpb = pd.cut(dp, [-np.inf, 1, 2, np.inf], labels=["d' < 1", "d' 1-2", "d' >= 2"], right=False)
            g = pd.DataFrame({"p": e[f"pct_comb_{m}"], "pos": pos, "dpb": dpb}).dropna()
            for k, h in g.groupby("dpb", observed=True):
                parts = [f"{q} {x.p.median():.3f} (n {len(x):,})" for q, x in h.groupby("pos", observed=True)]
                print(f"  {LABEL[m]:18s} {k:8s} " + "; ".join(parts))
                for q, x in h.groupby("pos", observed=True):
                    rows.append(dict(block=f"by_pos_{m}", key=f"{k} | {q}", value=x.p.median(), n=len(x)))
    else:
        print(f"\n(position block skipped: {state_dist} not found; run apo_holo_state_distance.py collect)")

    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f"\nwrote {SUMMARY_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["collect", "summarise"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--procs", type=int, default=22)
    ap.add_argument("--entries", default=ENTRIES)
    ap.add_argument("--state-distance", default=STATE_DIST,
                    help="apo_holo_state_distance.py's per-entry parquet (position block; optional)")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    if a.mode == "collect":
        if a.limit:
            ENTRIES = P.out(f"density_percentile_entries_limit{a.limit}.parquet")
        collect(a.limit, a.procs)
    else:
        summarise(a.entries, a.state_distance)
