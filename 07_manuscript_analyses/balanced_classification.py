# -*- coding: utf-8 -*-
"""Size-balanced re-classification of every analysis-set pocket, and its known-state validation.
Backs Supplementary Note 3, Table S9 (known-state validation, `summarise`) and Table S10 (released and
size-balanced labels side by side, `sites`), and the size-balanced labels used in Methods and main Results.

WHY. The released classifier compares an apo and a holo kernel density built with ONE pooled
bandwidth and equal priors. Unit-area normalisation removes the class sizes as a prior, but not as a
shape: a class with one or two structures is a few narrow spikes, so a model that does not sit on one
of them scores higher under the better-sampled class. The consequence is that P(holo) moves with the
apo/holo COUNT imbalance itself, independently of AlphaFold (Supplementary Note 3;
classifier_size_bias.py measures it three ways).

WHAT. Same classifier, same settings, but each comparison draws k = min(n_apo, n_holo_pool, K_CAP)
structures from EACH class, repeated R times, and averages the outputs. Every conformer is still used
across draws; only the asymmetry goes away.

    python balanced_classification.py collect  [--limit N] [--procs P]   -> per-entry CSV
    python balanced_classification.py validate [--n N] [--procs P]       -> known-state CSV
    python balanced_classification.py query_row --n 2000 [--procs P]     -> query-row check
    python balanced_classification.py summarise                          -> entry level + known state
    python balanced_classification.py sites                              -> site level (see below)

The query row carries no pocket RMSD (RMSD is measured from it), so a balanced draw that picks it gives
that one metric k - 1 holo values; `query_row` measures what that does against plain draw noise.

Curation reproduces the classifier's do_kde_plots_and_classification (00_classification/kde_classifier/)
for full_run_3 exactly: relabel ions
-> zero-to-NaN -> volume rule -> sort_dataframe_v2c -> role C, query row prepended, apo A/A*/A**,
holo H/H*. Fidelity check: the recomputed full-set P(holo) must equal the released
pocket_avg_P(holo); `collect` prints the match rate and it must be 100%, and `summarise` repeats it.

`summarise` reads the two per-entry CSVs: coverage and fidelity, the AF-Holo share and the Spearman
with the evidence ratio under both labels, and the known-state validation (balanced accuracy overall,
by separation tier and by evidence-ratio bin, with 95% intervals from a cluster bootstrap over UniProt
accession, B = 400, one set of resamples shared by both classifiers so their difference carries its
own interval). Tiers are the entry's RELEASED separation, the descriptor a user of AHoJ-DB sees.

`sites` aggregates the per-entry balanced P(holo) to every clustering level exactly as the pipeline
does (mean of the member pockets' probabilities, larger mean wins; NOT the vote columns) and re-asks
the paper's site-level questions under both labels, with no re-classification: (1) AF-Holo share at
each level; (2) st100_evidence_numbers.py item1, the evidence-ratio test at ST_100, on the member-average
counts it uses; (3) Figure 4's estimator (separation_shape.Gap: size-standardized, cluster bootstrap)
for consensus and per-tool detection and for docking, overall and by separation tier, released and
balanced labels on ONE row set so the paired difference carries its own interval, plus the gap
standardized for pocket size and evidence ratio together; (4) draw noise, when
classifier_size_bias.py downstream has left its independent re-draw of member pockets (downstream_members.csv).
The released arm of every table is the published label, so each block doubles as a fidelity check
against the number already in the paper.

A class under the balanced comparison is holo above P(holo) = 0.5, apo below, tie at 0.5; the
released label is the published one (P(holo) against P(apo), stored to three decimals), which differs
from the 0.5 rule for 87 of 237,922 entries, all at P(holo) = 0.500.

All files go to the manuscript-analyses folder (ma_paths.OUT). The per-entry CSVs are large and
regenerable; the small summary CSVs are the published tables.
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import argparse
import copy
import json
import os
import sys
import time
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import pandas as pd

import ahoj_job  # job archives under the names before AHoJ 2.6.0

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = str(P.WD)
UNI = str(P.cfg.ANALYSIS_UNIFIED_LEVELS_CSV)
DB = str(P.cfg.AHOJ_DB_DIR)
MERGED = str(P.cfg.CLASSIFICATION_MERGED_DIR)
SUMMARY = str(P.cfg.CLASSIFICATION_SUMMARIES_CSV)
PARAMS = os.path.join(DB, "Summaries", "apoholo_metric_stats", "full_run_3", "parameters.json")
ROOT = str(P.cfg.AHOJ_OUTPUT_FIX_DIR)

OUT_DIR = str(P.OUT)
ENTRIES_CSV = os.path.join(OUT_DIR, "balanced_entries.csv")
VALID_CSV = os.path.join(OUT_DIR, "balanced_known_state_entry_out.csv")   # held-out entry leaves both classes
VALID_ONE_CLASS_CSV = os.path.join(OUT_DIR, "balanced_known_state.csv")   # first design: its own class only
QUERY_ROW_CSV = os.path.join(OUT_DIR, "balanced_query_row.csv")
DOWN_MEMBERS = os.path.join(OUT_DIR, "downstream_members.csv")   # classifier_size_bias.py downstream
CSV_DIR = OUT_DIR   # the published summary CSVs; --csv-dir points a test run elsewhere

METRICS = ["pocket_rms", "sasa", "volume", "RoG"]
KW = dict(kde_extend=True, margin_cap_scale=1.0, bw_mode="pooled", bw_scale=0.75, prior_mode="equal")
K_CAP = 20          # draws are capped so a huge class does not dominate the runtime
REPS = 10           # draws per entry; P(holo) is the mean over draws
TIERS = ["very low", "low", "moderate", "high", "very high"]
TIER_EDGES = [-np.inf, 0.20, 0.30, 0.60, 0.85, np.inf]   # the database's own cuts (support_level)
LEVELS = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050", "ST_030"]
# known-state call rates by the log10 evidence ratio left after the hold-out
RATIO_EDGES = [-np.inf, -0.5, 0.0, 0.5, 1.0, 1.5, np.inf]
RATIO_LABELS = ["< -0.5", "-0.5 to 0", "0 to 0.5", "0.5 to 1", "1 to 1.5", "> 1.5"]
# pocket size x evidence ratio strata for the jointly standardized gap (log2 ratio, as in
# classifier_size_bias.downstream)
RATIO_L2_EDGES = [-np.inf, -1, 0, 1, 2, 3, 4, np.inf]
MIN_EVIDENCE_EDGES = [0, 1, 2, 4, 8, 16, np.inf]
MIN_EVIDENCE_LABELS = ["1", "2", "3-4", "5-8", "9-16", "> 16"]

_ARGS = None


def _init():
    global _ARGS
    with open(PARAMS, encoding="utf-8") as fh:
        _ARGS = SimpleNamespace(**json.load(fh))


def _load(tgz):
    """-> (df_c, apo, holo, model) with the classifier's own curation, or None."""
    from parsing_functions_DBv2 import sort_dataframe_v2c, relabel_apoholo_by_ions
    from parse_AHoJ3_DBv2b import replace_zero_with_nan
    a = _ARGS
    df = ahoj_job.read_global_results(tgz)
    if df is None:
        return None
    if int(a.relabel_apoholo) == 1:
        df, _ = relabel_apoholo_by_ions(df, builtin="total_le1", save_dir=None, verbose=False)
    df = replace_zero_with_nan(df, a.columns_nozero)
    s = pd.to_numeric(df["#resis3"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")
    df["volume"] = np.where(s.notna() & (s != 0) & v.isna(), 0.0, v)
    df_q, df_af = df[df["role"] == "Q"], df[df["role"] == "M"]
    df_f = sort_dataframe_v2c(df, sorting=0, full_pockets_aligned=a.full_pockets_aligned,
                              bndgrsds_prcnt_seqres=a.bndgrsds_prcnt_seqres,
                              bndgrsds_prcnt_obs=a.bndgrsds_prcnt_obs, resolution_soft=a.resolution_soft,
                              rog_diff_threshold_deltaQ=a.rog_diff_thrs_query,
                              rog_diff_threshold_mean=a.rog_diff_thrs_mean, pocket_distance=a.pocket_dist,
                              pocket_len_resis3=a.pocket_len_resis3,
                              drop_no_resolution=a.drop_no_resolution)
    df_c = pd.concat([df_q, df_f[df_f["role"] == "C"]], ignore_index=True)  # add_query_row = 1
    apo = df_c[df_c["apoholo_assignment"].isin(["A", "A*", "A**"])]
    holo = df_c[df_c["apoholo_assignment"].isin(["H", "H*"])]
    return df_c, apo, holo, df_af


def _classify(df_c, apo, holo, point):
    """Released classifier on the given reference sets -> the 'overall' block."""
    from kde_api import classify_point_kde_r3
    return classify_point_kde_r3(df_c, apo, holo, point, METRICS, **KW)[0]["overall"]


def _one(o):
    return dict(p_holo=o["avg_P(holo)"], overlap=o["overall_overlap"], mode_overlap=o["overall_mode_overlap"],
                dens_strong=o["avg_normalized_density_strongest"], dens_comb=o["avg_normalized_density_combined"],
                cls=o["overall_classification"])


def _balanced(apo, holo, point, rng, reps=REPS, k=None):
    """k drawn from EACH class, reps times; every output averaged over draws. `k` is normally
    min(n_apo, n_holo_pool, K_CAP); `query_row` fixes it so its three variants compare like with like."""
    k = min(len(apo), len(holo), K_CAP) if k is None else k
    if k < 1:
        return {}, 0
    acc = []
    for _ in range(reps):
        a = apo.iloc[rng.choice(len(apo), k, replace=False)]
        h = holo.iloc[rng.choice(len(holo), k, replace=False)]
        acc.append(_one(_classify(pd.concat([a, h]), a, h, point)))
    out = {f"bal_{f}": float(np.nanmean([r[f] for r in acc]))
           for f in ("p_holo", "overlap", "mode_overlap", "dens_strong", "dens_comb")}
    return out, k


def _burn(rng, n_a, n_h, reps=REPS):
    """Advance `rng` exactly as `_balanced` does on sets of these sizes, without classifying."""
    k = min(n_a, n_h, K_CAP)
    for _ in range(reps if k >= 1 else 0):
        rng.choice(n_a, k, replace=False)
        rng.choice(n_h, k, replace=False)


def _work_collect(job):
    job_name, tgz, seed = job
    try:
        loaded = _load(tgz)
        if loaded is None:
            return {"job_name": job_name, "error": "no global_results"}
        df_c, apo, holo, df_af = loaded
        if df_af.empty or len(apo) < 1 or len(holo) < 1:
            return {"job_name": job_name, "error": "no model / empty class"}
        m = df_af.iloc[:1]
        rec = {"job_name": job_name, "n_apo": len(apo), "n_holo_pool": len(holo)}
        rec.update({f"full_{k}": v for k, v in _one(_classify(df_c, apo, holo, m)).items()})
        bal, k = _balanced(apo, holo, m, np.random.default_rng(seed))
        rec.update(bal)
        rec["k"] = k
        return rec
    except Exception as e:  # noqa: BLE001
        return {"job_name": job_name, "error": f"{type(e).__name__}: {e}"}


def _work_validate(job):
    """Hold out one PDB ENTRY of known state and classify one of its structures, with the released
    classifier and with the balanced one. Every chain of that entry leaves BOTH reference sets: a
    crystal often carries liganded and unliganded copies of the same site, and a sibling left in the
    other class is a near-duplicate an AlphaFold model never has. The held-out holo entry is never the
    query's own entry; `apo_sib` / `holo_sib` count the siblings removed from the other class
    (`summarise` prints how often there are any), and a test whose removal empties the other class is
    dropped and flagged (`apo_dropped` / `holo_dropped`). k and the evidence ratio come from the reduced
    sets. The first run of this mode, 2026-09-16, removed the entry from its own class only; its output
    is kept as VALID_ONE_CLASS_CSV. The random stream is consumed exactly as that run consumed it (the
    apo test's balanced draws come from a copy, and `_burn` advances the main stream), so both designs
    hold out the same entries and agree exactly wherever an entry has no siblings; `summarise` reports
    the difference on the tests valid in both. Removing whole entries does not remove near-duplicates
    deposited as separate entries (re-refinements, repeat depositions), which still flatter both
    classifiers."""
    job_name, tgz, seed = job
    try:
        loaded = _load(tgz)
        if loaded is None:
            return {"job_name": job_name, "error": "no global_results"}
        df_c, apo, holo, df_af = loaded
        if df_af.empty or len(apo) < 1 or len(holo) < 1:
            return {"job_name": job_name, "error": "no model / empty class"}
        rng = np.random.default_rng(seed)
        rec = {"job_name": job_name, "n_apo": len(apo), "n_holo_pool": len(holo)}
        ent_a = apo["structure"].astype(str).str.lower().str[:4]
        ent_h = holo["structure"].astype(str).str.lower().str[:4]
        ua = ent_a.unique()
        if len(ua) >= 2:
            e = ua[int(rng.integers(len(ua)))]
            sel = ent_a == e
            pt = apo[sel].iloc[[int(rng.integers(int(sel.sum())))]]
            rest, other = apo[~sel], holo[ent_h != e]
            rec["apo_entry"], rec["apo_sib"] = e, int(len(holo) - len(other))
            # an apo chain from the query's own crystal takes the query row out of the holo set too
            rec["apo_query_out"] = int(((ent_h == e) & (holo["role"] == "Q")).any())
            draw = copy.deepcopy(rng)
            # the main stream moves exactly as in the first design, so the held-out holo entry drawn
            # below is the same one; the balanced draws for this test come from the copy
            _burn(rng, len(rest), len(holo))
            if len(other) >= 1:   # both reference sets reduced; k and the ratio come from them
                rec["apo_full"] = _one(_classify(pd.concat([rest, other]), rest, other, pt))["p_holo"]
                rec["apo_bal"] = _balanced(rest, other, pt, draw)[0].get("bal_p_holo", np.nan)
                rec["apo_ratio"] = np.log10(len(other) / max(len(rest), 1))
            else:                 # the entry held the site's only holo structures: no test
                rec["apo_dropped"] = 1
        cand = [e for e in ent_h.unique() if e != job_name[:4].lower()]
        if cand:
            e = cand[int(rng.integers(len(cand)))]
            sel = (ent_h == e) & (holo["role"] == "C")
            if sel.any() and (~(ent_h == e)).sum() >= 1:
                pt = holo[sel].iloc[[int(rng.integers(int(sel.sum())))]]
                rest, other = holo[ent_h != e], apo[ent_a != e]
                rec["holo_entry"], rec["holo_sib"] = e, int(len(apo) - len(other))
                if len(other) >= 1:
                    rec["holo_full"] = _one(_classify(pd.concat([other, rest]), other, rest, pt))["p_holo"]
                    rec["holo_bal"] = _balanced(other, rest, pt, rng)[0].get("bal_p_holo", np.nan)
                    rec["holo_ratio"] = np.log10(len(rest) / max(len(other), 1))
                else:             # the entry held the site's only apo structures: no test
                    rec["holo_dropped"] = 1
        return rec
    except Exception as e:  # noqa: BLE001
        return {"job_name": job_name, "error": f"{type(e).__name__}: {e}"}


def _work_query_row(job):
    """The query row sits in the holo pool with sasa, volume and RoG but no pocket RMSD (RMSD is measured
    from it), so when a balanced draw picks it, pocket RMSD sees k - 1 holo values against k apo. Same
    entry, same model: balanced P(holo) with the query row eligible (as in `collect`), with it excluded,
    and an independent re-draw of the first as the draw-noise baseline, all three at ONE k, the largest
    the excluded variant allows. Where the query row is the site's only holo structure the excluded
    variant does not exist; those entries are flagged (`query_only`), not dropped silently."""
    job_name, tgz, seed = job
    try:
        loaded = _load(tgz)
        if loaded is None:
            return {"job_name": job_name, "error": "no global_results"}
        df_c, apo, holo, df_af = loaded
        if df_af.empty or len(apo) < 1 or len(holo) < 1:
            return {"job_name": job_name, "error": "no model / empty class"}
        q = holo["role"] == "Q"
        rms = pd.to_numeric(holo.loc[q, "pocket_rms"], errors="coerce")
        rec = {"job_name": job_name, "n_apo": len(apo), "n_holo_pool": len(holo), "query_rows": int(q.sum()),
               "query_rms_missing": bool(rms.isna().all()), "k_released_rule": min(len(apo), len(holo), K_CAP)}
        if (~q).sum() < 1:
            rec["query_only"] = 1
            return rec
        m = df_af.iloc[:1]
        k = min(len(apo), int((~q).sum()), K_CAP)
        rec["k"] = k
        for tag, pool, s in (("with", holo, seed), ("without", holo[~q], seed), ("with_redraw", holo, seed + 1)):
            rec[f"p_{tag}"] = _balanced(apo, pool, m, np.random.default_rng(s), k=k)[0].get("bal_p_holo", np.nan)
        return rec
    except Exception as e:  # noqa: BLE001
        return {"job_name": job_name, "error": f"{type(e).__name__}: {e}"}


def _jobs(limit=None, sample=None, seed=0):
    uni = pd.read_csv(UNI, usecols=["threshold", "rep_job"], low_memory=False)
    aset = set(uni.loc[uni.threshold == "UNCLUSTERED", "rep_job"])
    s = pd.read_csv(SUMMARY, usecols=["job_name", "query_path"], low_memory=False)
    s = s[s.job_name.isin(aset)]
    if sample:
        s = s.sample(n=min(sample, len(s)), random_state=seed)
    if limit:
        s = s.iloc[:limit]
    return [(j, os.path.join(ROOT, str(q).replace("\\", "/").split("/")[-2], f"{j}.tgz"), 90_000 + i)
            for i, (j, q) in enumerate(zip(s.job_name, s.query_path))]


def _run(fn, jobs, out, procs):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"entries: {len(jobs):,}  procs={procs}  -> {out}", flush=True)
    t0 = time.time()
    rows = []
    with Pool(procs, initializer=_init) as pool:
        for i, rec in enumerate(pool.imap_unordered(fn, jobs, chunksize=16), 1):
            rows.append(rec)
            if i % 10000 == 0:
                el = time.time() - t0
                print(f"  {i:,} / {len(jobs):,}  {el / 60:5.1f} min elapsed, "
                      f"{(el / i) * (len(jobs) - i) / 60:5.1f} min left", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(d):,}  {(time.time() - t0) / 60:.1f} min", flush=True)
    if "error" in d:
        print(d["error"].dropna().value_counts().head().to_string(), flush=True)
    return d


def collect(limit, procs):
    d = _run(_work_collect, _jobs(limit), ENTRIES_CSV, procs)
    ok = d[d.get("error").isna()] if "error" in d else d
    rel = pd.read_csv(SUMMARY, usecols=["job_name", "pocket_avg_P(holo)"], low_memory=False)
    m = ok.merge(rel, on="job_name", how="left")
    good = (m.full_p_holo - m["pocket_avg_P(holo)"]).abs() < 0.0015
    print(f"FIDELITY: recomputed full-set P(holo) equals the released value for "
          f"{100 * good.mean():.2f}% of {len(m):,} entries (must be 100%)")


def validate(n, procs):
    _run(_work_validate, _jobs(sample=n, seed=7), VALID_CSV, procs)


def query_row(n, procs):
    d = _run(_work_query_row, _jobs(sample=n, seed=13), QUERY_ROW_CSV, procs)
    d = d[d["error"].isna()] if "error" in d else d
    _query_row_report(d)


def _query_row_report(d):
    qo = d.get("query_only", pd.Series(np.nan, index=d.index)).fillna(0) > 0
    print(f"\n=== query row in the balanced draws: {len(d):,} entries ===")
    print(f"  query row present in the holo pool for {_pct(d.query_rows == 1):.1f}%; its pocket RMSD is missing "
          f"for {_pct(d.query_rms_missing):.1f}%")
    print(f"  the query row is the site's ONLY holo structure for {int(qo.sum()):,} entries "
          f"({_pct(qo):.1f}%): there the balanced draw always uses it, pocket RMSD has no holo value at all, and "
          f"no excluded variant exists, so the asymmetry is unmeasurable there rather than small")
    ok = d[["p_with", "p_without", "p_with_redraw"]].notna().all(axis=1) & ~qo
    x = d[ok]
    print(f"  compared at one k per entry (the largest the excluded variant allows) on {len(x):,} entries; "
          f"that k is below the collect rule's for {_pct(x.k < x.k_released_rule):.1f}% of them")
    for lab, a, b in (("query row eligible vs excluded", "p_with", "p_without"),
                      ("two independent draws, query row eligible (baseline)", "p_with", "p_with_redraw")):
        dd = x[a] - x[b]
        flip = _cls(x[a]) != _cls(x[b])
        print(f"  {lab}: mean difference {dd.mean():+.4f}, median |difference| {dd.abs().median():.4f}, "
              f"class flips {_pct(flip):.1f}% of {len(x):,}")
    kb = pd.cut(x.k, [0, 2, 5, K_CAP], labels=["k 1-2", "k 3-5", "k 6-20"])
    for b, g in x.groupby(kb, observed=True):
        print(f"    {b}: n = {len(g):,}; eligible minus excluded {(g.p_with - g.p_without).mean():+.4f}, flips "
              f"{_pct(_cls(g.p_with) != _cls(g.p_without)):.1f}%; draw-noise flips "
              f"{_pct(_cls(g.p_with) != _cls(g.p_with_redraw)):.1f}%")


def _tier(sep):
    return pd.cut(sep, TIER_EDGES, labels=TIERS)


# ===================================================================================== summaries
ARMS = (("full", "released"), ("bal", "balanced"))


def _cls(p):
    """Class from P(holo) alone (the balanced comparison records no P(apo)): holo above 0.5, apo
    below, tie at 0.5. Rounding to 6 decimals absorbs the float noise of averaging draws."""
    p = np.round(np.asarray(p, dtype=float), 6)
    return np.where(np.isnan(p), "", np.where(p > 0.5, "holo", np.where(p < 0.5, "apo", "tie")))


def _pct(x):
    x = np.asarray(x, dtype=float)
    return 100.0 * float(x.mean()) if len(x) else np.nan


def _shape():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import separation_shape as S
    return S


def _boot_indices(codes):
    """Cluster-bootstrap resamples over `codes`, drawn by Figure 4's own resampler and seed
    (separation_shape.Gap._indices), so these intervals are made the way the paper's are."""
    S = _shape()
    return S.Gap._indices(SimpleNamespace(codes=np.asarray(codes), seed=S.SEED, _boot_ix=None))


def _ci_txt(pt, bs, signed=False):
    lo, hi = _shape().ci(bs)
    f = "+.1f" if signed else ".1f"
    return f"{pt:{f}} [{lo:{f}}, {hi:{f}}]"


def _entries(path):
    d = pd.read_csv(path, low_memory=False)
    bad = d["error"].notna() if "error" in d else pd.Series(False, index=d.index)
    print(f"entries in {os.path.basename(path)}: {len(d):,}; not classifiable: {int(bad.sum()):,}")
    if bad.any():
        print(d.loc[bad, "error"].value_counts().head().to_string())
    d = d[~bad].copy()
    d["bal_cls"] = _cls(d.bal_p_holo)
    for tag, _ in ARMS:
        d[f"{tag}_sep"] = 1 - (d[f"{tag}_overlap"] + d[f"{tag}_mode_overlap"]) / 2
    return d


def _released():
    """Published per-entry values: P(holo) and counts from the query summaries; the class and the
    UniProt key from the UNCLUSTERED rows of the unified table (the label the paper reports)."""
    rel = pd.read_csv(SUMMARY, usecols=["job_name", "n_apo", "n_holo", "pocket_avg_P(holo)"], low_memory=False)
    rel = rel.drop_duplicates("job_name").rename(columns={
        "n_apo": "rel_n_apo", "n_holo": "rel_n_holo", "pocket_avg_P(holo)": "rel_p_holo"})
    u = pd.read_csv(UNI, usecols=["threshold", "rep_job", "pocket_class", "uniprots_key"], low_memory=False)
    u = u[u.threshold == "UNCLUSTERED"].drop(columns="threshold").rename(
        columns={"rep_job": "job_name", "pocket_class": "rel_cls"})
    return rel.merge(u, on="job_name", how="inner")


def _accuracy(a, h, cell, ncell, ix=None):
    """Per cell: % of known-apo tests called apo, % of known-holo tests called holo, their mean."""
    if ix is not None:
        a, h, cell = a[ix], h[ix], cell[ix]
    out = np.full((ncell, 3), np.nan)
    for j, (x, hit) in enumerate(((a, a < 0.5), (h, h > 0.5))):
        ok = np.isfinite(x) & (cell >= 0)
        n = np.bincount(cell[ok], minlength=ncell).astype(float)
        s = np.bincount(cell[ok], weights=hit[ok].astype(float), minlength=ncell)
        out[:, j] = np.where(n > 0, 100.0 * s / np.where(n > 0, n, 1.0), np.nan)
    out[:, 2] = out[:, :2].mean(axis=1)
    return out


def _called_holo(x, cell, ncell, ix=None):
    if ix is not None:
        x, cell = x[ix], cell[ix]
    ok = np.isfinite(x) & (cell >= 0)
    n = np.bincount(cell[ok], minlength=ncell).astype(float)
    s = np.bincount(cell[ok], weights=(x[ok] > 0.5).astype(float), minlength=ncell)
    return np.where(n > 0, 100.0 * s / np.where(n > 0, n, 1.0), np.nan)


def summarise(entries=ENTRIES_CSV, valid=VALID_CSV, one_class=VALID_ONE_CLASS_CSV):
    from scipy.stats import spearmanr
    print("=" * 100)
    print("balanced_classification.py summarise -- released vs size-balanced, entry level")
    print("=" * 100)
    d = _entries(entries).merge(_released(), on="job_name", how="left")

    print("\n=== A. coverage and fidelity ===")
    fid = (d.full_p_holo - d.rel_p_holo).abs() < 0.0015
    print(f"FIDELITY: recomputed full-set P(holo) equals the released value for {100 * fid.mean():.2f}% "
          f"of {len(d):,} entries (must be 100%)")
    full_cls = d.full_cls.astype(str).str.lower().map({"holo": "holo", "apo": "apo", "ambiguous": "tie"})
    print(f"  released class reproduced for {_pct(full_cls == d.rel_cls):.2f}%; n_apo equal for "
          f"{_pct(d.n_apo == d.rel_n_apo):.2f}%; n_holo_pool = n_holo + 1 for "
          f"{_pct(d.n_holo_pool == d.rel_n_holo + 1):.2f}%")
    rule = pd.Series(_cls(d.rel_p_holo), index=d.index)
    off = rule != d.rel_cls
    print(f"  published label vs the 0.5 rule on the published P(holo): differ for {int(off.sum()):,} entries; "
          f"all of them at P(holo) = 0.500: {bool((d.loc[off, 'rel_p_holo'] == 0.5).all())}")
    d["ratio"] = np.log10(d.n_holo_pool / d.n_apo)
    d["min_ev"] = np.minimum(d.n_apo, d.n_holo_pool)

    print("\n=== B. released vs size-balanced, all analysis-set entries ===")
    rel_def, bal_def = d.rel_cls.isin(["apo", "holo"]), d.bal_cls.isin(["apo", "holo"])
    both = rel_def & bal_def
    ch = both & (d.rel_cls != d.bal_cls)
    rho = {"released": spearmanr(d.rel_p_holo, d.ratio)[0], "balanced": spearmanr(d.bal_p_holo, d.ratio)[0]}
    share = {"released": _pct(d.rel_cls[rel_def] == "holo"), "balanced": _pct(d.bal_cls[bal_def] == "holo")}
    cls = {"released": d.rel_cls, "balanced": d.bal_cls}
    sep = {"released": d.full_sep, "balanced": d.bal_sep}
    rows = []
    print(f"  {'':44s}{'released':>14s}{'size-balanced':>16s}")
    for q, fn, fmt in (("AF-Holo pockets", lambda a: int((cls[a] == "holo").sum()), ",d"),
                       ("AF-Apo pockets", lambda a: int((cls[a] == "apo").sum()), ",d"),
                       ("ties", lambda a: int((cls[a] == "tie").sum()), ",d"),
                       ("AF-Holo share, ties excluded (%)", lambda a: share[a], ".1f"),
                       ("Spearman P(holo) vs log10 evidence ratio", lambda a: rho[a], "+.3f"),
                       ("median separation", lambda a: float(sep[a].median()), ".3f")):
        r, b = fn("released"), fn("balanced")
        rows.append({"quantity": q, "released": r, "balanced": b})
        print(f"  {q:44s}{format(r, fmt):>14s}{format(b, fmt):>16s}")
    chg = 100.0 * ch.sum() / both.sum()
    rows.append({"quantity": "labels changed (% of entries definite under both)", "released": np.nan,
                 "balanced": chg})
    h2a, a2h = int((ch & (d.rel_cls == "holo")).sum()), int((ch & (d.rel_cls == "apo")).sum())
    print(f"  labels changed: {chg:.1f}% of {int(both.sum()):,} entries definite under both "
          f"(holo -> apo {h2a:,}; apo -> holo {a2h:,}): a net shift toward AF-Apo of "
          f"{100 * (h2a - a2h) / both.sum():.1f}% of entries plus a two-way part of {200 * a2h / both.sum():.1f}% "
          f"(compare the disagreement between two independent balanced draws, `sites` section 4)")
    print(f"  the holo reference set is the larger for {_pct(d.n_holo_pool > d.n_apo):.1f}% of entries, the apo set "
          f"for {_pct(d.n_holo_pool < d.n_apo):.1f}%; median n_apo {d.n_apo.median():.0f}, median n_holo + 1 "
          f"{d.n_holo_pool.median():.0f}")
    print(f"  draws: k = min(n_apo, n_holo + 1, {K_CAP}) from each class, {REPS} draws; median k "
          f"{d.k.median():.0f}; k = 1 for {_pct(d.k == 1):.1f}% of entries, k <= 2 for {_pct(d.k <= 2):.1f}%, "
          f"k = {K_CAP} for {_pct(d.k == K_CAP):.1f}%")
    pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, "balanced_classification_summary.csv"), index=False)

    print("\n=== C. AF-Holo share (%, ties excluded) by released separation tier ===")
    d["tier"] = _tier(d.full_sep)
    rows = []
    for t, g in d.groupby("tier", observed=True):
        rows.append({"tier": t, "n": len(g),
                     "released": _pct(g.rel_cls[g.rel_cls.isin(["apo", "holo"])] == "holo"),
                     "balanced": _pct(g.bal_cls[g.bal_cls.isin(["apo", "holo"])] == "holo"),
                     "median_k": float(g.k.median()), "k_le_2_pct": _pct(g.k <= 2),
                     "rho_ratio_released": spearmanr(g.rel_p_holo, g.ratio)[0],
                     "rho_ratio_balanced": spearmanr(g.bal_p_holo, g.ratio)[0]})
        r = rows[-1]
        print(f"  {t:10s} n = {r['n']:>7,}   released {r['released']:5.1f}   size-balanced {r['balanced']:5.1f}"
              f"   median k {r['median_k']:4.1f}   k <= 2: {r['k_le_2_pct']:5.1f}%   AlphaFold rho with the ratio "
              f"{r['rho_ratio_released']:+.3f} / {r['rho_ratio_balanced']:+.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, "balanced_by_separation_tier.csv"), index=False)

    print("\n=== D. separation is itself count-sensitive ===")
    lm = np.log10(d.min_ev)
    print(f"  Spearman(separation, log10 min(n_apo, n_holo + 1)): released {spearmanr(d.full_sep, lm)[0]:+.3f}; "
          f"size-balanced {spearmanr(d.bal_sep, lm)[0]:+.3f}")
    print(f"  Spearman(released separation, log10 evidence ratio): {spearmanr(d.full_sep, d.ratio)[0]:+.3f} over all "
          f"entries, {spearmanr(d.full_sep[d.ratio > 0], d.ratio[d.ratio > 0])[0]:+.3f} where the holo set is the larger")
    d["mbin"] = pd.cut(d.min_ev, MIN_EVIDENCE_EDGES, labels=MIN_EVIDENCE_LABELS)
    for b, g in d.groupby("mbin", observed=True):
        print(f"  shared evidence {b:>5s}: n = {len(g):>7,}   median separation released {g.full_sep.median():.3f}, "
              f"size-balanced {g.bal_sep.median():.3f}   very-low tier released {_pct(g.full_sep <= 0.20):4.1f}%, "
              f"size-balanced {_pct(g.bal_sep <= 0.20):4.1f}%")

    if os.path.exists(QUERY_ROW_CSV):
        qr = pd.read_csv(QUERY_ROW_CSV, low_memory=False)
        _query_row_report(qr[qr["error"].isna()] if "error" in qr else qr)

    if not os.path.exists(valid):
        print(f"\n(no known-state file at {valid})")
        return
    v = pd.read_csv(valid, low_memory=False)
    nv = len(v)
    bad = v["error"].notna() if "error" in v else pd.Series(False, index=v.index)
    # rows arrive in pool order; sort so the cluster codes, and so the resamples, are reproducible
    v = v[~bad].merge(d[["job_name", "full_sep", "uniprots_key"]], on="job_name", how="left")
    v = v.sort_values("job_name", kind="stable").reset_index(drop=True)
    v["tier"] = _tier(v.full_sep)
    has_a = v.apo_full.notna() & v.apo_bal.notna()
    has_h = v.holo_full.notna() & v.holo_bal.notna()
    codes = pd.factorize(v.uniprots_key.fillna(v.job_name))[0]
    idx = _boot_indices(codes)
    print("\n=== E. known-state validation: one PDB entry of known state held out per sampled site ===")
    print(f"  {os.path.basename(valid)}: entries sampled {nv:,}; not classifiable {int(bad.sum()):,}; "
          f"known-apo tests {int(has_a.sum()):,} (sites with >= 2 apo PDB entries); known-holo tests "
          f"{int(has_h.sum()):,}; UniProt clusters {len(np.unique(codes)):,}; cluster bootstrap B = {len(idx)}")
    if "apo_sib" in v:
        print(f"  the held-out entry also had chains in the OTHER class (removed there too) for "
              f"{_pct(v.loc[has_a, 'apo_sib'] > 0):.1f}% of apo tests and {_pct(v.loc[has_h, 'holo_sib'] > 0):.1f}% "
              f"of holo tests")
        qout = v.get("apo_query_out", pd.Series(np.nan, index=v.index)).fillna(0) > 0
        print(f"  apo tests whose held-out entry is the query's own crystal (the query row left the holo set "
              f"with it): {int((qout & has_a).sum()):,} of {int(has_a.sum()):,}")
        drop_a = v.get("apo_dropped", pd.Series(np.nan, index=v.index)).fillna(0) > 0
        drop_h = v.get("holo_dropped", pd.Series(np.nan, index=v.index)).fillna(0) > 0
        print(f"  tests dropped because removing the entry emptied the other class: apo {int(drop_a.sum()):,} "
              f"of {int((drop_a | has_a).sum()):,}, holo {int(drop_h.sum()):,} of {int((drop_h | has_h).sum()):,}; "
              f"by released separation tier (apo / holo): " + ", ".join(
                  f"{t} {int((drop_a & (v.tier == t)).sum()):,} / {int((drop_h & (v.tier == t)).sum()):,}"
                  for t in TIERS))
    arm = {tag: (np.where(has_a, v[f"apo_{tag}"], np.nan), np.where(has_h, v[f"holo_{tag}"], np.nan))
           for tag, _ in ARMS}

    def table(cell, ncell):
        res = {}
        for tag, _ in ARMS:
            a, h = arm[tag]
            res[tag] = (_accuracy(a, h, cell, ncell), np.stack([_accuracy(a, h, cell, ncell, ix) for ix in idx]))
        return res

    zero = np.zeros(len(v), dtype=int)
    res = table(zero, 1)
    rows = []
    print(f"  {'classifier':12s}{'known apo called apo (%)':>28s}{'known holo called holo (%)':>30s}"
          f"{'balanced accuracy (%)':>26s}")
    for tag, name in ARMS:
        pt, bs = res[tag]
        print(f"  {name:12s}" + "".join(f"{_ci_txt(pt[0, j], bs[:, 0, j]):>28s}" for j in range(3)))
        row = {"classifier": name}
        for j, key in enumerate(("apo_recall", "holo_recall", "balanced_accuracy")):
            lo, hi = _shape().ci(bs[:, 0, j])
            row.update({key: pt[0, j], f"{key}_lo": lo, f"{key}_hi": hi})
        rows.append(row)
    dpt = res["bal"][0][0] - res["full"][0][0]
    dbs = res["bal"][1][:, 0] - res["full"][1][:, 0]
    print(f"  {'difference':12s}" + "".join(f"{_ci_txt(dpt[j], dbs[:, j], signed=True):>28s}" for j in range(3))
          + "   (size-balanced minus released, same resamples)")
    for kind, col, mask in (("known apo", "apo", has_a), ("known holo", "holo", has_h)):
        r1 = spearmanr(v.loc[mask, f"{col}_full"], v.loc[mask, f"{col}_ratio"])[0]
        r2 = spearmanr(v.loc[mask, f"{col}_bal"], v.loc[mask, f"{col}_ratio"])[0]
        print(f"  Spearman(P(holo), log10 evidence ratio after the hold-out), {kind}: released {r1:+.3f}, "
              f"size-balanced {r2:+.3f}")
        for r in rows:
            r[f"rho_ratio_{col}"] = r1 if r["classifier"] == "released" else r2
        # the ratio trend inside separation tiers (separation and the ratio are correlated)
        per = []
        for t in TIERS:
            m2 = mask & (v.tier == t)
            if m2.sum() > 30:
                per.append(f"{t} {spearmanr(v.loc[m2, f'{col}_full'], v.loc[m2, f'{col}_ratio'])[0]:+.3f} / "
                           f"{spearmanr(v.loc[m2, f'{col}_bal'], v.loc[m2, f'{col}_ratio'])[0]:+.3f}")
        print(f"    within released separation tiers (released / size-balanced): " + "; ".join(per))
    if "holo_entry" in v:
        # PanDDA-style group depositions (hundreds of near-identical entries per site) sit in the 5P-5S and
        # 7G-7H ID blocks; the entry-level hold-out leaves their siblings in the pool. A heuristic, by ID.
        grp = v.holo_entry.astype(str).str.lower().str[:2].isin(["5p", "5q", "5r", "5s", "7g", "7h"])
        print(f"  known-holo tests whose held-out entry has a group-deposition ID (5P-5S, 7G-7H): "
              f"{int((grp & has_h).sum()):,} of {int(has_h.sum()):,}")
        for lab, m2 in (("without them", has_h & ~grp), ("them only", has_h & grp)):
            if m2.sum() > 30:
                print(f"    {lab}: n = {int(m2.sum()):,}; Spearman with the ratio released "
                      f"{spearmanr(v.loc[m2, 'holo_full'], v.loc[m2, 'holo_ratio'])[0]:+.3f}, size-balanced "
                      f"{spearmanr(v.loc[m2, 'holo_bal'], v.loc[m2, 'holo_ratio'])[0]:+.3f}; called AF-Holo "
                      f"released {_pct(v.loc[m2, 'holo_full'] > 0.5):.1f}%, size-balanced "
                      f"{_pct(v.loc[m2, 'holo_bal'] > 0.5):.1f}%")
    pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, "balanced_known_state_summary.csv"), index=False)

    if one_class and os.path.exists(one_class) and os.path.abspath(one_class) != os.path.abspath(valid):
        cols = ["apo_full", "apo_bal", "holo_full", "holo_bal"]
        o = pd.read_csv(one_class, low_memory=False)
        o = o[o["error"].isna()] if "error" in o else o
        w = v.merge(o[["job_name"] + cols].rename(columns={c: f"{c}_one" for c in cols}), on="job_name", how="inner")
        wa = w[["apo_full", "apo_bal", "apo_full_one", "apo_bal_one"]].notna().all(axis=1).to_numpy()
        wh = w[["holo_full", "holo_bal", "holo_full_one", "holo_bal_one"]].notna().all(axis=1).to_numpy()
        widx = _boot_indices(pd.factorize(w.uniprots_key.fillna(w.job_name))[0])
        z = np.zeros(len(w), dtype=int)
        print(f"\n  design check, {os.path.basename(one_class)} vs {os.path.basename(valid)} on the {len(w):,} "
              f"entries in both (same sample, same seeds; apo tests {int(wa.sum()):,}, holo tests {int(wh.sum()):,}):"
              f" balanced accuracy (%) with the held-out entry removed from its own class only, and from both")
        for tag, name in ARMS:
            res2 = {}
            for sfx in ("_one", ""):
                a = np.where(wa, w[f"apo_{tag}{sfx}"], np.nan)
                h = np.where(wh, w[f"holo_{tag}{sfx}"], np.nan)
                res2[sfx] = (_accuracy(a, h, z, 1)[0], np.stack([_accuracy(a, h, z, 1, ix)[0] for ix in widx]))
            (p0, b0), (p1, b1) = res2["_one"], res2[""]
            print(f"    {name:9s} own class only {_ci_txt(p0[2], b0[:, 2])}; both classes {_ci_txt(p1[2], b1[:, 2])}; "
                  f"difference {_ci_txt(p1[2] - p0[2], b1[:, 2] - b0[:, 2], signed=True)}   (recall apo "
                  f"{p0[0]:.1f} -> {p1[0]:.1f}, holo {p0[1]:.1f} -> {p1[1]:.1f})")

    print("\n  balanced accuracy (%) by the site's released separation tier")
    tcode = v.tier.cat.codes.to_numpy().astype(int)
    res = table(tcode, 5)
    rows = []
    print(f"  {'tier':10s}{'apo tests':>10s}{'holo tests':>11s}{'released':>22s}{'size-balanced':>22s}"
          f"{'difference':>22s}   recall apo / holo, released | size-balanced")
    for i, t in enumerate(TIERS):
        na, nh = int((has_a & (tcode == i)).sum()), int((has_h & (tcode == i)).sum())
        (pr, br), (pb, bb) = res["full"], res["bal"]
        print(f"  {t:10s}{na:>10,}{nh:>11,}{_ci_txt(pr[i, 2], br[:, i, 2]):>22s}"
              f"{_ci_txt(pb[i, 2], bb[:, i, 2]):>22s}"
              f"{_ci_txt(pb[i, 2] - pr[i, 2], bb[:, i, 2] - br[:, i, 2], signed=True):>22s}"
              f"   {pr[i, 0]:.1f} / {pr[i, 1]:.1f} | {pb[i, 0]:.1f} / {pb[i, 1]:.1f}")
        for tag, name in ARMS:
            p, b = res[tag]
            lo, hi = _shape().ci(b[:, i, 2])
            rows.append({"tier": t, "classifier": name, "n_apo_tests": na, "n_holo_tests": nh,
                         "apo_recall": p[i, 0], "holo_recall": p[i, 1], "balanced_accuracy": p[i, 2],
                         "balanced_accuracy_lo": lo, "balanced_accuracy_hi": hi})
    for tag, name in ARMS:
        p, b = res[tag]
        print(f"  {name:>14s}: very high minus very low "
              f"{_ci_txt(p[4, 2] - p[0, 2], b[:, 4, 2] - b[:, 0, 2], signed=True)} points")
    pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, "balanced_known_state_by_tier.csv"), index=False)

    print("\n  share called AF-Holo (%) by the log10 evidence ratio left after the hold-out")
    rows = []
    for kind, col, mask in (("known apo", "apo", has_a), ("known holo", "holo", has_h)):
        rb = pd.cut(v[f"{col}_ratio"], RATIO_EDGES, labels=RATIO_LABELS).cat.codes.to_numpy().astype(int)
        rb = np.where(mask, rb, -1)
        nb = np.bincount(rb[rb >= 0], minlength=len(RATIO_LABELS))
        out = {}
        for tag, _ in ARMS:
            x = v[f"{col}_{tag}"].to_numpy(float)
            out[tag] = (_called_holo(x, rb, len(RATIO_LABELS)),
                        np.stack([_called_holo(x, rb, len(RATIO_LABELS), ix) for ix in idx]))
        print(f"  {kind}:")
        for i, lab in enumerate(RATIO_LABELS):
            (pr, br), (pb, bb) = out["full"], out["bal"]
            print(f"    ratio {lab:>10s}  n = {nb[i]:>6,}   released {_ci_txt(pr[i], br[:, i]):>20s}   "
                  f"size-balanced {_ci_txt(pb[i], bb[:, i]):>20s}")
            for tag, name in ARMS:
                p, b = out[tag]
                lo, hi = _shape().ci(b[:, i])
                rows.append({"test": kind, "ratio_bin": lab, "n": int(nb[i]), "classifier": name,
                             "pct_called_holo": p[i], "lo": lo, "hi": hi})
        for tag, name in ARMS:
            p, b = out[tag]
            print(f"    {name:>14s}: most holo-rich minus most apo-rich bin "
                  f"{_ci_txt(p[-1] - p[0], b[:, -1] - b[:, 0], signed=True)} points")
    pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, "balanced_known_state_by_ratio.csv"), index=False)


# ======================================================================================== sites
def _joint(S, g, col, edges=RATIO_L2_EDGES):
    """Figure 4's gap with the standard population on union-Rg deciles CROSSED with evidence-ratio
    bins -- the direct adjustment of classifier_size_bias.downstream, with an interval. Gap reads the
    stratum count from separation_shape.K when it is called, so K is swapped for the call and
    restored; the resamples are the parent's."""
    nr = len(edges) - 1
    rb = pd.cut(g.d[col], edges, labels=False).to_numpy()
    ok = np.isfinite(rb)
    g._indices()
    h = copy.copy(g)
    h.zb = g.zb * nr + np.where(ok, rb, 0).astype(int)
    w = np.bincount(h.zb[ok], minlength=S.K * nr).astype(float)
    h.w = w / w.sum()
    cell = np.where(ok, 0, -1)
    k0 = S.K
    S.K = k0 * nr
    try:
        return h.point(cell, 1)[0], h.boot(cell, 1)[:, 0]
    finally:
        S.K = k0


def _gap_block(S, task, fr):
    """Released and balanced labels on ONE row set with ONE set of resamples."""
    g_r = S.Gap(fr.assign(cls=fr.cls_rel), task)
    g_b = S.Gap(fr.assign(cls=fr.cls_bal), task)
    assert len(g_r.d) == len(g_b.d) and (g_r.codes == g_b.codes).all()
    g_b._boot_ix = g_r._indices()
    n = len(g_r.d)
    gs = (("released", g_r), ("balanced", g_b))
    rows = []
    print(f"\n  --- {task}: n = {n:,} pockets, {g_r.d.site_uniprot.nunique():,} UniProt clusters; AF-Holo "
          f"released {int(g_r.ish.sum()):,}, size-balanced {int(g_b.ish.sum()):,} ---")
    zero = np.zeros(n, dtype=int)
    pts, bss = {}, {}
    for name, g in gs:
        rt = g.rates(zero, 1, raw=True)[0]
        br = g.boot_rates(zero, 1, raw=True)[:, 0, :]
        print(f"  raw, {name:9s}  AF-Apo {rt[0]:5.1f}  AF-Holo {rt[1]:5.1f}  gap "
              f"{_ci_txt(rt[1] - rt[0], br[:, 1] - br[:, 0], signed=True)} pp")
        lo, hi = S.ci(br[:, 1] - br[:, 0])
        rows.append({"task": task, "labels": name, "estimate": "raw", "stratum": "all", "n": n,
                     "gap": rt[1] - rt[0], "lo": lo, "hi": hi, "rate_apo": rt[0], "rate_holo": rt[1]})
        pts[name], bss[name] = g.point(zero, 1)[0], g.boot(zero, 1)[:, 0]
    for name, _ in gs:
        S.show(f"size-standardized, {name}", pts[name], bss[name], width=46)
        lo, hi = S.ci(bss[name])
        rows.append({"task": task, "labels": name, "estimate": "size_std", "stratum": "all", "n": n,
                     "gap": pts[name], "lo": lo, "hi": hi})
    S.show("size-standardized, balanced minus released", pts["balanced"] - pts["released"],
           bss["balanced"] - bss["released"], width=46)
    for name, g in gs:
        p, b = _joint(S, g, "ratio_l2")
        S.show(f"size x evidence-ratio standardized, {name}", p, b, width=46)
        lo, hi = S.ci(b)
        rows.append({"task": task, "labels": name, "estimate": "size_ratio_std", "stratum": "all", "n": n,
                     "gap": p, "lo": lo, "hi": hi})

    for axis, col in (("released separation", "sep"), ("size-balanced separation", "sep_bal")):
        c = S.codes_from(g_r.d[col], S.TIERS)
        n5 = S.Gap.counts(c, 5)
        res = {name: (g.point(c, 5), g.boot(c, 5), g.rates(c, 5)) for name, g in gs}
        print(f"  size-standardized gap by {axis} tier (pp){'':4s}{'released label':>24s}"
              f"{'size-balanced label':>24s}{'difference':>24s}   std. rates apo / holo, released | balanced")
        for i, t in enumerate(S.TIERS):
            (pr, br, rr), (pb, bb, rb_) = res["released"], res["balanced"]
            print(f"    {t:10s} n = {n5[i]:>7,}{'':14s}{_ci_txt(pr[i], br[:, i], True):>24s}"
                  f"{_ci_txt(pb[i], bb[:, i], True):>24s}{_ci_txt(pb[i] - pr[i], bb[:, i] - br[:, i], True):>24s}"
                  f"   {rr[i, 0]:.1f} / {rr[i, 1]:.1f} | {rb_[i, 0]:.1f} / {rb_[i, 1]:.1f}")
            for name, _ in gs:
                p, b, r = res[name]
                lo, hi = S.ci(b[:, i])
                rows.append({"task": task, "labels": name, "estimate": "size_std", "stratum": f"{axis}: {t}",
                             "n": int(n5[i]), "gap": p[i], "lo": lo, "hi": hi,
                             "rate_apo": r[i, 0], "rate_holo": r[i, 1]})
        (pr, br, _), (pb, bb, _) = res["released"], res["balanced"]
        print(f"    {'very high minus very low':34s}{_ci_txt(pr[4] - pr[0], br[:, 4] - br[:, 0], True):>24s}"
              f"{_ci_txt(pb[4] - pb[0], bb[:, 4] - bb[:, 0], True):>24s}"
              f"{_ci_txt((pb[4] - pb[0]) - (pr[4] - pr[0]), (bb[:, 4] - bb[:, 0]) - (br[:, 4] - br[:, 0]), True):>24s}")
        for name, _ in gs:
            p, b, _ = res[name]
            lo, hi = S.ci(b[:, 4] - b[:, 0])
            rows.append({"task": task, "labels": name, "estimate": "size_std_span", "stratum": axis,
                         "n": n, "gap": p[4] - p[0], "lo": lo, "hi": hi})
    return rows


def sites(entries=ENTRIES_CSV, redraw=DOWN_MEMBERS):
    from scipy.stats import spearmanr
    S = _shape()
    from st100_evidence_numbers import load_st100, partial_spearman
    print("=" * 100)
    print("balanced_classification.py sites -- released vs size-balanced labels at the site level")
    print("=" * 100)
    d = _entries(entries)
    u = pd.read_csv(UNI, usecols=["threshold", "unit_uid", "job_names", "pocket_class", "pocket_p_holo"],
                    low_memory=False)
    u = u[u.threshold.isin(LEVELS)].copy()
    ex = u[["threshold", "unit_uid", "job_names"]].assign(job_name=u.job_names.str.split(";")).explode("job_name")
    ex = ex.drop(columns="job_names").merge(
        d[["job_name", "full_p_holo", "bal_p_holo", "bal_overlap", "bal_mode_overlap"]], on="job_name", how="left")
    ex["bal_op"] = (ex.bal_overlap + ex.bal_mode_overlap) / 2
    agg = ex.groupby(["threshold", "unit_uid"], sort=False).agg(
        n_mem=("job_name", "size"), n_ok=("bal_p_holo", "count"), full_p=("full_p_holo", "mean"),
        bal_p=("bal_p_holo", "mean"), bal_op=("bal_op", "mean")).reset_index()
    s = u.drop(columns="job_names").merge(agg, on=["threshold", "unit_uid"], how="left")
    s["covered"] = s.n_ok == s.n_mem
    s["bal_cls"] = np.where(s.covered, _cls(s.bal_p), "")
    s["bal_sep"] = 1 - s.bal_op

    print("\n=== 1. AF-Holo share at every clustering level (a site's P(holo) is the mean over its members; "
          "ties excluded) ===")
    print(f"  {'level':12s}{'sites':>9s}{'rebuilt = released':>20s}{'released holo / apo':>22s}{'share':>7s}"
          f"{'size-balanced holo / apo':>26s}{'share':>7s}{'ties':>6s}{'changed':>9s}")
    rows = []
    for lv in LEVELS:
        g = s[s.threshold == lv]
        fid = _pct(((g.full_p.round(3) - g.pocket_p_holo).abs() < 0.0015)[g.covered])
        rd, bd = g.pocket_class.isin(["apo", "holo"]), g.bal_cls.isin(["apo", "holo"])
        r = {"level": lv, "sites": len(g), "sites_with_unclassified_member": int((~g.covered).sum()),
             "rebuilt_equals_released_pct": fid,
             "released_holo": int((g.pocket_class == "holo").sum()), "released_apo": int((g.pocket_class == "apo").sum()),
             "released_share": _pct(g.pocket_class[rd] == "holo"),
             "balanced_holo": int((g.bal_cls == "holo").sum()), "balanced_apo": int((g.bal_cls == "apo").sum()),
             "balanced_ties": int((g.bal_cls == "tie").sum()), "balanced_share": _pct(g.bal_cls[bd] == "holo"),
             "changed_pct": _pct((g.pocket_class != g.bal_cls)[rd & bd])}
        rows.append(r)
        print(f"  {lv:12s}{r['sites']:>9,}{r['rebuilt_equals_released_pct']:>19.2f}%"
              f"{r['released_holo']:>12,} / {r['released_apo']:<7,}{r['released_share']:>7.1f}"
              f"{r['balanced_holo']:>16,} / {r['balanced_apo']:<7,}{r['balanced_share']:>7.1f}"
              f"{r['balanced_ties']:>6,}{r['changed_pct']:>8.1f}%")
    print(f"  sites with a member the balanced run could not classify: "
          + ", ".join(f"{r['level']} {r['sites_with_unclassified_member']:,}" for r in rows))
    pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, "balanced_prevalence_by_level.csv"), index=False)

    print("\n=== 2. what sets where a pocket falls? review_items.py item1 under both labels (ST_100) ===")
    e = load_st100()
    n0 = len(e)
    kmin = ex[ex.threshold == "ST_100"].merge(d[["job_name", "k"]], on="job_name").groupby("unit_uid").k.min()
    e = e.merge(s.loc[s.threshold == "ST_100", ["unit_uid", "bal_p", "bal_cls"]], on="unit_uid", how="left")
    e = e[e.bal_cls.isin(["apo", "holo"])].copy()
    e["bal_holo"] = (e.bal_cls == "holo").astype(int)
    e["kmin"] = e.unit_uid.map(kmin)
    print(f"  classified ST_100 sites {n0:,}; also definite under the size-balanced label {len(e):,}. Evidence = "
          f"member-average n_apo / n_holo, ratio = log10((n_holo + 1)/n_apo), as in item1.")
    print("  AF-Holo share (%) and median |P(holo) - 0.5| by the smallest k among a site's members "
          "(k = 1: each draw compares single structures):")
    for b, g in e.groupby(pd.cut(e.kmin, [0, 1, 2, 5, 10, K_CAP], labels=["1", "2", "3-5", "6-10", "11-20"]),
                          observed=True):
        print(f"      k {b:>5s}: n = {len(g):>7,}   released {_pct(g.holo):5.1f}  size-balanced {_pct(g.bal_holo):5.1f}"
              f"   |P - 0.5| released {(g.pocket_p_holo - 0.5).abs().median():.3f}, size-balanced "
              f"{(g.bal_p - 0.5).abs().median():.3f}")
    arms = (("released", "pocket_p_holo", "holo"), ("size-balanced", "bal_p", "bal_holo"))
    preds = [("ratio log10((n_holo+1)/n_apo)", "ratio"), ("log10 n_apo", "l_apo"), ("log10 (n_holo+1)", "l_nh1"),
             ("log10 min(n_apo, n_holo+1)", "l_min"), ("log10 (n_apo + n_holo + 1)", "l_tot")]
    rows = []
    print(f"\n  (a) Spearman rho with P(holo){'':6s}{'released':>12s}{'size-balanced':>16s}")
    for lab, c in preds:
        r = [spearmanr(e[c], e[p]).correlation for _, p, _ in arms]
        rows.append({"test": "spearman", "predictor": lab, "released": r[0], "balanced": r[1]})
        print(f"      {lab:36s}{r[0]:>+8.3f}{r[1]:>+16.3f}")
    codes = pd.factorize(e.uniprots_key)[0]
    idx = _boot_indices(codes)
    x, p1, p2 = e.ratio.to_numpy(), e.pocket_p_holo.to_numpy(), e.bal_p.to_numpy()
    bs = np.array([[spearmanr(x[ix], p1[ix]).correlation, spearmanr(x[ix], p2[ix]).correlation] for ix in idx])
    lo1, hi1 = S.ci(100 * bs[:, 0])
    lo2, hi2 = S.ci(100 * bs[:, 1])
    lo3, hi3 = S.ci(100 * (bs[:, 1] - bs[:, 0]))
    print(f"      ratio, x100 with 95% CI (cluster bootstrap over {codes.max() + 1:,} UniProt keys): released "
          f"{100 * rows[0]['released']:.1f} [{lo1:.1f}, {hi1:.1f}], size-balanced {100 * rows[0]['balanced']:.1f} "
          f"[{lo2:.1f}, {hi2:.1f}], difference {100 * (rows[0]['balanced'] - rows[0]['released']):.1f} "
          f"[{lo3:.1f}, {hi3:.1f}]")
    print(f"      the same intervals as rho: released {lo1 / 100:.4f} to {hi1 / 100:.4f}; size-balanced "
          f"{lo2 / 100:.4f} to {hi2 / 100:.4f}")
    print(f"\n  (b) partial Spearman with P(holo){'':2s}{'released':>12s}{'size-balanced':>16s}")
    for lab, c in preds[1:]:
        r = [partial_spearman(e[p], e[c], [e.ratio]) for _, p, _ in arms]
        rows.append({"test": "partial | ratio", "predictor": lab, "released": r[0], "balanced": r[1]})
        print(f"      {lab + ' | ratio':36s}{r[0]:>+8.3f}{r[1]:>+16.3f}")
    for lab, ctrl in (("ratio | log n_apo, log (n_holo+1)", ["l_apo", "l_nh1"]), ("ratio | log total", ["l_tot"]),
                      ("ratio | log min", ["l_min"])):
        r = [partial_spearman(e[p], e.ratio, [e[k] for k in ctrl]) for _, p, _ in arms]
        rows.append({"test": "partial", "predictor": lab, "released": r[0], "balanced": r[1]})
        print(f"      {lab:36s}{r[0]:>+8.3f}{r[1]:>+16.3f}")

    print("\n  (c) AF-Holo prevalence (%) across ratio bins, and the spread across quartiles of total evidence "
          "(n_apo + n_holo + 1) inside each")
    edges = [-np.inf, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, np.inf]
    labels = ["<-1", "-1..-0.5", "-0.5..-0.25", "-0.25..0", "0..0.25", "0.25..0.5", "0.5..1", "1..1.5", ">1.5"]
    e["rbin"] = pd.cut(e.ratio, edges, labels=labels)
    e["tq"] = pd.qcut(e.l_tot, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    byr = e.groupby("rbin", observed=False).agg(n=("holo", "size"), released=("holo", "mean"),
                                                 balanced=("bal_holo", "mean"))
    print("      ratio bin     " + "".join(f"{b:>12s}" for b in labels))
    print("      n             " + "".join(f"{v:>12,}" for v in byr.n))
    print("      released      " + "".join(f"{100 * v:>12.1f}" for v in byr.released))
    print("      size-balanced " + "".join(f"{100 * v:>12.1f}" for v in byr.balanced))
    cell_n = e.pivot_table(index="rbin", columns="tq", values="holo", aggfunc="size", observed=False)
    for name, pcol, ccol in arms:
        g = e.pivot_table(index="rbin", columns="tq", values=ccol, aggfunc="mean", observed=False) * 100
        within = g.max(axis=1) - g.min(axis=1)
        across = g.max(axis=0) - g.min(axis=0)
        print(f"      {name}: AF-Holo % by ratio bin (rows) x quartile of total evidence (columns); "
              f"cell n {int(cell_n.min().min()):,} to {int(cell_n.max().max()):,}")
        print("        " + g.round(1).to_string().replace("\n", "\n        "))
        slopes = [spearmanr(sub.l_tot, sub[pcol]).correlation
                  for _, sub in e.groupby("rbin", observed=False) if len(sub) > 200]
        print(f"      {name:14s} spread across COUNT quartiles at fixed ratio: median {within.median():.1f} pp, "
              f"max {within.max():.1f} pp; across RATIO bins at fixed count: median {across.median():.1f} pp, "
              f"min {across.min():.1f} pp")
        print(f"      {'':14s} Spearman(total evidence, P(holo)) inside the ratio bins: "
              + ", ".join(f"{v:+.2f}" for v in slopes))
        rows += [{"test": "grid", "predictor": f"{name}: within-ratio spread median", "released": np.nan,
                  "balanced": within.median()},
                 {"test": "grid", "predictor": f"{name}: across-ratio spread median", "released": np.nan,
                  "balanced": across.median()}]

    print(f"\n  (d) logistic regression for the class, 5-fold CV AUC (standardized inputs)"
          f"{'':2s}{'released':>10s}{'size-balanced':>16s}")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    for lab, cols in [("ratio", ["ratio"]), ("log n_apo + log (n_holo+1)", ["l_apo", "l_nh1"]),
                      ("log total + log min", ["l_tot", "l_min"]), ("ratio + log total", ["ratio", "l_tot"]),
                      ("ratio + log min", ["ratio", "l_min"])]:
        aucs = []
        for _, _, ccol in arms:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
            aucs.append(cross_val_score(m, e[cols].to_numpy(), e[ccol].to_numpy(), cv=5, scoring="roc_auc").mean())
        rows.append({"test": "cv_auc", "predictor": lab, "released": aucs[0], "balanced": aucs[1]})
        print(f"      {lab:70s}{aucs[0]:>10.3f}{aucs[1]:>16.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, "balanced_evidence_test.csv"), index=False)

    print("\n=== 3. does the label predict anything? Figure 4's estimator (separation_shape.Gap) under both labels ===")
    lbs = S.load_lbs()
    st = s.loc[s.threshold == "ST_100", ["unit_uid", "full_p", "bal_p", "bal_cls", "bal_sep", "covered"]]
    x = lbs.merge(st, on="unit_uid", how="left")
    fid = (x.full_p.round(3) - x.pocket_p_holo).abs() < 0.0015
    keep = x.bal_cls.isin(["apo", "holo"])
    uncovered = ~x.covered.fillna(False).astype(bool)
    print(f"  detection pockets (separation_shape.load_lbs): {len(x):,}; site P(holo) rebuilt from the members "
          f"equals the released value for {_pct(fid):.2f}%; definite under the size-balanced label "
          f"{int(keep.sum()):,} (balanced ties {int((x.bal_cls == 'tie').sum()):,}, with an unclassifiable "
          f"member {int(uncovered.sum()):,})")
    x = x[keep].copy()
    x["cls_rel"], x["cls_bal"] = x.cls, x.bal_cls
    x["sep_bal"] = pd.cut(x.bal_sep, TIER_EDGES, labels=S.TIERS).astype(object)
    x["ratio_l2"] = np.log2((x.n_holo_avg + 1) / x.n_apo_avg)
    ch = x.cls_rel != x.cls_bal
    print(f"  on this set: AF-Holo share released {_pct(x.cls_rel == 'holo'):.1f}%, size-balanced "
          f"{_pct(x.cls_bal == 'holo'):.1f}%; labels changed {_pct(ch):.1f}% (holo -> apo "
          f"{int((ch & (x.cls_rel == 'holo')).sum()):,}, apo -> holo {int((ch & (x.cls_rel == 'apo')).sum()):,}); "
          f"sites moving tier when separation is size-balanced {_pct(x.sep != x.sep_bal):.1f}%")
    dk = S.load_docking(x)
    rows = []
    for task, fr in [(t, x) for t in ["consensus"] + S.PRED] + [("docking", dk)]:
        rows += _gap_block(S, task, fr)
    pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, "balanced_downstream_gaps.csv"), index=False)

    if not os.path.exists(redraw):
        print(f"\n(no independent re-draw at {redraw}; run classifier_size_bias.py downstream first)")
        return
    print("\n=== 4. draw noise: classifier_size_bias.py downstream re-drew its member pockets independently ===")
    m = pd.read_csv(redraw, low_memory=False)
    m = m[m.p_bal.notna()]
    j = m.merge(d[["job_name", "bal_p_holo"]], on="job_name", how="inner")
    dp = (j.p_bal - j.bal_p_holo).abs()
    agree = _pct(_cls(j.p_bal) == _cls(j.bal_p_holo))
    print(f"  entries re-drawn {len(j):,}: size-balanced label agrees for {agree:.1f}% (disagrees for "
          f"{100 - agree:.1f}%); |difference in P(holo)| median {dp.median():.3f}, 95th percentile "
          f"{dp.quantile(0.95):.3f}")
    e2 = ex[ex.threshold == "ST_100"].merge(m[["job_name", "p_bal"]], on="job_name", how="left")
    a2 = e2.groupby("unit_uid").agg(n_mem=("job_name", "size"), n2=("p_bal", "count"), p2=("p_bal", "mean"))
    a2 = a2[a2.n2 == a2.n_mem].reset_index()
    y = x.merge(a2[["unit_uid", "p2"]], on="unit_uid", how="inner")
    y["cls_bal2"] = _cls(y.p2)
    y = y[y.cls_bal2.isin(["apo", "holo"])].copy()
    site_agree = _pct(y.cls_bal == y.cls_bal2)
    print(f"  detection sites whose members were all re-drawn: {len(y):,}; site label agrees for "
          f"{site_agree:.1f}% (disagrees for {100 - site_agree:.1f}%)")
    g1 = S.Gap(y.assign(cls=y.cls_bal), "consensus")
    g2 = S.Gap(y.assign(cls=y.cls_bal2), "consensus")
    g0 = S.Gap(y.assign(cls=y.cls_rel), "consensus")
    g2._boot_ix = g0._boot_ix = g1._indices()
    zero = np.zeros(len(g1.d), dtype=int)
    p1, b1 = g1.point(zero, 1)[0], g1.boot(zero, 1)[:, 0]
    p2_, b2 = g2.point(zero, 1)[0], g2.boot(zero, 1)[:, 0]
    S.show("consensus gap, released label", g0.point(zero, 1)[0], g0.boot(zero, 1)[:, 0], len(g1.d), width=46)
    S.show("consensus gap, size-balanced (collect draws)", p1, b1, width=46)
    S.show("consensus gap, size-balanced (independent draws)", p2_, b2, width=46)
    S.show("independent minus collect draws", p2_ - p1, b2 - b1, width=46)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["collect", "validate", "query_row", "summarise", "sites"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--procs", type=int, default=22)
    ap.add_argument("--entries", default=ENTRIES_CSV, help="per-entry CSV (summarise, sites)")
    ap.add_argument("--valid", default=VALID_CSV, help="known-state CSV (summarise)")
    ap.add_argument("--valid-one-class", default=VALID_ONE_CLASS_CSV,
                    help="first-design known-state CSV, compared on the same entries (summarise)")
    ap.add_argument("--redraw", default=DOWN_MEMBERS, help="independent member re-draw (sites)")
    ap.add_argument("--csv-dir", default=OUT_DIR, help="where the summary CSVs go (default: ma_paths.OUT)")
    a = ap.parse_args()
    CSV_DIR = a.csv_dir
    pd.set_option("display.width", 250)
    if a.mode == "collect":
        collect(a.limit, a.procs)
    elif a.mode == "validate":
        validate(a.n, a.procs)
    elif a.mode == "query_row":
        query_row(a.n, a.procs)
    elif a.mode == "summarise":
        summarise(a.entries, a.valid, a.valid_one_class)
    else:
        sites(a.entries, a.redraw)
