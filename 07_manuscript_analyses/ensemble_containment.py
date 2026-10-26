# -*- coding: utf-8 -*-
"""Ensemble containment: how often does an experimental apo state fall inside the holo ensemble of
the same binding site, and vice versa?  Backs Supplementary Table S7 (containment, within-entry spread
and conformers per ligand state).

Motivation: the Introduction's warrant for comparing AlphaFold against the whole apo/holo ENSEMBLE
rather than a single reference structure. If one apo conformer routinely sits inside the range of the
holo conformers of the same site, then a single-reference apo-vs-holo comparison is ambiguous by
construction, independently of anything AlphaFold does.

Data: the per-structure metric values are not in any summary table -- they live in the per-job
AHoJ archives (global_results.csv inside <root>/<batch>/<job>.tgz). This script reproduces the
classifier's own curation (relabel_apoholo_by_ions -> replace_zero_with_nan -> volume rule ->
sort_dataframe_v2c -> role C -> apo/holo split) by importing the same functions from the
classification code (00_classification/kde_classifier/), with the parameters of the full_run_3
classification. Fidelity check:
the per-entry n_apo / n_holo obtained here are compared against the classification summary's own
n_apo / n_holo; the match rate is printed and must be ~100%.

Metrics: the four classifier metrics -- volume, sasa, RoG, pocket_rms. pocket_rms is measured
against the query (holo) structure, so it is a distance from a holo anchor for both classes.

Per entry and metric, with A = apo values and H = holo values (non-null):
    apo_in_holo_range   #A inside [min H, max H]          (needs |H| >= 2)
    apo_in_holo_sd      #A inside mean(H) +- 1 SD(H)      (needs |H| >= 2, SD > 0; SD ddof=1)
    holo_in_apo_range   #H inside [min A, max A]          (needs |A| >= 2)
    holo_in_apo_sd      #H inside mean(A) +- 1 SD(A)      (needs |A| >= 2, SD > 0)
Denominators are stored alongside so the summary can pool over structures or average over entries.
Also recorded per entry (apo pockets: more evidence per pocket but fewer; holo pockets: larger
conformational space, fewer conformers per state): per class and metric the mean,
SD (ddof=1), min and max; the number of distinct holo ligand states (set of ligand codes bound in the
pocket); and the number of distinct PDB entries per class.

    python ensemble_containment.py collect [--limit N] [--procs 12]   -> ensemble_containment_entries.csv
    python ensemble_containment.py summarise                           -> prints the Table S7 numbers
                                                                          -> ensemble_containment_summary.csv

Both files go to the manuscript-analyses folder (ma_paths.OUT). The per-entry CSV (~238k rows) is
regenerable; the summary CSV is the published table.
"""
import ma_paths as P  # noqa: F401  (roots; puts the classification code on sys.path)

import argparse
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
UNI = str(P.cfg.ANALYSIS_UNIFIED_LEVELS_CSV)
DB = str(P.cfg.AHOJ_DB_DIR)
MERGED = str(P.cfg.CLASSIFICATION_MERGED_DIR)
SUMMARY = str(P.cfg.CLASSIFICATION_SUMMARIES_CSV)
PARAMS = os.path.join(DB, "Summaries", "apoholo_metric_stats", "full_run_3", "parameters.json")
ROOT = str(P.cfg.AHOJ_OUTPUT_FIX_DIR)

ENTRIES_CSV = P.out("ensemble_containment_entries.csv")
SUMMARY_CSV = P.out("ensemble_containment_summary.csv")

METRICS = ["volume", "sasa", "RoG", "pocket_rms"]
NOZERO = ["RoG", "bfac_t", "bfac_avg", "resolution"]

_ARGS = None


def _load_args():
    with open(PARAMS, encoding="utf-8") as fh:
        p = json.load(fh)
    return SimpleNamespace(
        full_pockets_aligned=p["full_pockets_aligned"],
        bndgrsds_prcnt_seqres=p["bndgrsds_prcnt_seqres"],
        bndgrsds_prcnt_obs=p["bndgrsds_prcnt_obs"],
        resolution_soft=p["resolution_soft"],
        rog_diff_thrs_query=p["rog_diff_thrs_query"],
        rog_diff_thrs_mean=p["rog_diff_thrs_mean"],
        pocket_dist=p["pocket_dist"],
        pocket_len_resis3=p["pocket_len_resis3"],
        drop_no_resolution=p["drop_no_resolution"],
        relabel_apoholo=p["relabel_apoholo"],
        columns_nozero=p["columns_nozero"],
    )


def _init():
    global _ARGS
    _ARGS = _load_args()


def _read_global_results(tgz):
    return ahoj_job.read_global_results(tgz)


def _containment(a, h):
    """a, h: 1-D float arrays (non-null). Returns dict of numerators / denominators, plus the
    per-class summary (mean, SD, min, max) that item 4 needs for the within-entry spread."""
    out = {}
    na, nh = a.size, h.size
    out["n_a"], out["n_h"] = na, nh
    for tag, x in (("a", a), ("h", h)):
        if x.size:
            out[f"{tag}_mean"], out[f"{tag}_min"], out[f"{tag}_max"] = float(x.mean()), float(x.min()), float(x.max())
        if x.size >= 2:
            out[f"{tag}_sd"] = float(x.std(ddof=1))
    # apo inside holo
    if nh >= 2:
        out["a_in_h_range_den"] = na
        out["a_in_h_range"] = int(((a >= h.min()) & (a <= h.max())).sum())
        sd = h.std(ddof=1)
        if sd > 0:
            mu = h.mean()
            out["a_in_h_sd_den"] = na
            out["a_in_h_sd"] = int(((a >= mu - sd) & (a <= mu + sd)).sum())
    if na >= 2:
        out["h_in_a_range_den"] = nh
        out["h_in_a_range"] = int(((h >= a.min()) & (h <= a.max())).sum())
        sd = a.std(ddof=1)
        if sd > 0:
            mu = a.mean()
            out["h_in_a_sd_den"] = nh
            out["h_in_a_sd"] = int(((h >= mu - sd) & (h <= mu + sd)).sum())
    return out


def _lig_state(row):
    """Ligand state of one holo conformer: the set of ligand codes bound in the pocket."""
    codes = set()
    for col in ("query_ligs", "nonquery_ligs"):
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            codes.update(v.strip().split("-"))
    return frozenset(codes)


def process(job):
    """job: (job_name, tgz_path). Returns a flat dict or None."""
    from parsing_functions_DBv2 import sort_dataframe_v2c, relabel_apoholo_by_ions
    from parse_AHoJ3_DBv2b import replace_zero_with_nan
    job_name, tgz = job
    a = _ARGS
    try:
        df = _read_global_results(tgz)
    except Exception as e:  # noqa: BLE001
        return {"job_name": job_name, "status": f"error:{type(e).__name__}"}
    if df is None or df.empty:
        return {"job_name": job_name, "status": "not_found"}
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
    rec = {"job_name": job_name, "status": "ok", "n_apo": int(len(apo)), "n_holo": int(len(holo)),
           # item 4: how many distinct ligand states the holo conformers cover (a "state" is the set
           # of ligand codes in the pocket), and how many distinct PDB entries each class spans
           "n_holo_lig_states": int(holo.apply(_lig_state, axis=1).nunique()) if len(holo) else 0,
           "n_holo_pdb": int(holo["structure"].nunique()), "n_apo_pdb": int(apo["structure"].nunique())}
    for m in METRICS:
        if m not in df_c.columns:
            continue
        av = pd.to_numeric(apo[m], errors="coerce").dropna().to_numpy(float)
        hv = pd.to_numeric(holo[m], errors="coerce").dropna().to_numpy(float)
        for k, val in _containment(av, hv).items():
            rec[f"{m}_{k}"] = val
    return rec


def job_list(limit=None):
    uni = pd.read_csv(UNI, usecols=["threshold", "rep_job", "query_path"], low_memory=False)
    uni = uni[uni.threshold == "UNCLUSTERED"]
    jobs = []
    for jn, qp in zip(uni.rep_job, uni.query_path):
        batch = str(qp).replace("\\", "/").split("/")[-2]
        jobs.append((jn, os.path.join(ROOT, batch, f"{jn}.tgz")))
    if limit:
        jobs = jobs[:limit]
    return jobs


def collect(limit, procs, out=ENTRIES_CSV):
    jobs = job_list(limit)
    print(f"analysis-set entries: {len(jobs):,}  procs={procs}")
    t0 = time.time()
    rows = []
    with Pool(procs, initializer=_init) as pool:
        for i, rec in enumerate(pool.imap_unordered(process, jobs, chunksize=64), 1):
            rows.append(rec)
            if i % 10000 == 0:
                print(f"  {i:,} done  {time.time() - t0:7.0f} s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(d):,}  {time.time() - t0:.0f} s")
    print(d.status.value_counts().to_string())


def summarise(entries=ENTRIES_CSV):
    d = pd.read_csv(entries, low_memory=False)
    print(f"entries read: {len(d):,}; status counts:\n{d.status.value_counts().to_string()}")
    d = d[d.status == "ok"].copy()

    # --- fidelity: our curation vs the classifier's own n_apo / n_holo
    s = pd.read_csv(SUMMARY, usecols=["job_name", "n_apo", "n_holo"], low_memory=False)
    m = d.merge(s, on="job_name", how="left", suffixes=("", "_cls"))
    ok_a = (m.n_apo == m.n_apo_cls)
    ok_h = (m.n_holo == m.n_holo_cls)
    print(f"\nfidelity vs classification summary (n = {len(m):,}): n_apo match {100 * ok_a.mean():.2f}%, "
          f"n_holo match {100 * ok_h.mean():.2f}%, both {100 * (ok_a & ok_h).mean():.2f}%")

    print(f"\nanalysis-set entries with n_apo >= 2: {int((d.n_apo >= 2).sum()):,} "
          f"({100 * (d.n_apo >= 2).mean():.1f}%); n_holo >= 2: {int((d.n_holo >= 2).sum()):,} "
          f"({100 * (d.n_holo >= 2).mean():.1f}%); both >= 2: {int(((d.n_apo >= 2) & (d.n_holo >= 2)).sum()):,} "
          f"({100 * ((d.n_apo >= 2) & (d.n_holo >= 2)).mean():.1f}%)")
    print(f"median n_apo {d.n_apo.median():.0f}, median n_holo {d.n_holo.median():.0f}")

    rows = []
    for m_ in METRICS:
        for direction, num, den in [("apo state inside holo min-max", f"{m_}_a_in_h_range", f"{m_}_a_in_h_range_den"),
                                    ("apo state inside holo mean+-1SD", f"{m_}_a_in_h_sd", f"{m_}_a_in_h_sd_den"),
                                    ("holo state inside apo min-max", f"{m_}_h_in_a_range", f"{m_}_h_in_a_range_den"),
                                    ("holo state inside apo mean+-1SD", f"{m_}_h_in_a_sd", f"{m_}_h_in_a_sd_den")]:
            sub = d[d[den].notna() & (d[den] > 0)]
            frac_entry = sub[num] / sub[den]
            rows.append({
                "metric": m_, "statistic": direction,
                "n_entries": len(sub),
                "n_structures": int(sub[den].sum()),
                "pooled_pct": 100 * sub[num].sum() / sub[den].sum(),
                "entry_mean_pct": 100 * frac_entry.mean(),
                "entry_median_pct": 100 * frac_entry.median(),
                "pct_entries_all_inside": 100 * (frac_entry == 1).mean(),
                "pct_entries_none_inside": 100 * (frac_entry == 0).mean(),
                "pct_entries_any_inside": 100 * (frac_entry > 0).mean(),
            })
    t = pd.DataFrame(rows)
    t.to_csv(SUMMARY_CSV, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print("\n=== containment per metric (analysis-set entries; denominators as defined in the docstring) ===")
    print(t.round(1).to_string(index=False))

    # --- the two-sided statement: at least one apo conformer inside the holo range AND vice versa
    print("\n=== entries where the ranges interpenetrate (both >= 2 states per class) ===")
    for m_ in METRICS:
        sub = d[(d[f"{m_}_a_in_h_range_den"] > 0) & (d[f"{m_}_h_in_a_range_den"] > 0)]
        both = (sub[f"{m_}_a_in_h_range"] > 0) & (sub[f"{m_}_h_in_a_range"] > 0)
        either = (sub[f"{m_}_a_in_h_range"] > 0) | (sub[f"{m_}_h_in_a_range"] > 0)
        print(f"{m_:10s} n={len(sub):,}: apo-in-holo AND holo-in-apo {100 * both.mean():.1f}% | "
              f"either {100 * either.mean():.1f}% | disjoint ranges {100 * (~either).mean():.1f}%")

    # --- all four metrics at once, complete cases
    cols_a = [f"{m_}_a_in_h_range" for m_ in METRICS]
    dens_a = [f"{m_}_a_in_h_range_den" for m_ in METRICS]
    ok = d[dens_a].gt(0).all(axis=1)
    sub = d[ok]
    any_all4 = (sub[cols_a].gt(0)).all(axis=1)
    print(f"\ncomplete cases (n_holo >= 2, all four metrics) n={len(sub):,}: at least one apo state inside "
          f"the holo min-max on ALL FOUR metrics {100 * any_all4.mean():.1f}%; on at least one metric "
          f"{100 * (sub[cols_a].gt(0)).any(axis=1).mean():.1f}%")
    frac4 = pd.concat([sub[c] / sub[dn] for c, dn in zip(cols_a, dens_a)], axis=1)
    print(f"  per-entry fraction of apo states inside holo min-max, mean over entries, per metric: "
          + ", ".join(f"{m_} {100 * frac4[i].mean():.1f}%" for i, m_ in enumerate(METRICS)))

    item4_ensemble(d)


def item4_ensemble(d):
    """Within-entry spread (Table S7): do holo conformers span a larger conformational space than apo
    conformers of the same site, and with fewer conformers per (ligand) state?"""
    uni = pd.read_csv(UNI, usecols=["threshold", "rep_job", "pocket_class"], low_memory=False)
    uni = uni[uni.threshold == "UNCLUSTERED"].rename(columns={"rep_job": "job_name"})
    d = d.merge(uni[["job_name", "pocket_class"]], on="job_name", how="left")
    print("\n" + "=" * 100)
    print("ITEM 4 (ensemble half) -- within-entry spread per class and conformers per state")
    both = d[(d.n_apo >= 3) & (d.n_holo >= 3)].copy()
    print(f"entries with >= 3 apo AND >= 3 holo conformers: {len(both):,} of {len(d):,} "
          f"({100 * len(both) / len(d):.1f}%)")
    print("\n  spread of each class's conformers within the entry (SD, ddof=1; CV = SD / mean for the three "
          "size metrics; pocket_rms is a distance from the query so its SD is compared directly):")
    for m_ in METRICS:
        a_sd, h_sd = both[f"{m_}_a_sd"], both[f"{m_}_h_sd"]
        ok = a_sd.notna() & h_sd.notna()
        if m_ != "pocket_rms":
            a_cv = a_sd / both[f"{m_}_a_mean"].where(both[f"{m_}_a_mean"] > 0)
            h_cv = h_sd / both[f"{m_}_h_mean"].where(both[f"{m_}_h_mean"] > 0)
            ok = ok & a_cv.notna() & h_cv.notna() & np.isfinite(a_cv) & np.isfinite(h_cv)
            x, y, lab = a_cv[ok], h_cv[ok], "CV"
        else:
            x, y, lab = a_sd[ok], h_sd[ok], "SD"
        print(f"    {m_:10s} n={int(ok.sum()):,}  median {lab} apo {x.median():.3f} vs holo {y.median():.3f}; "
              f"holo spread > apo spread in {100 * (y > x).mean():.1f}% of entries; "
              f"median holo/apo spread ratio {np.median((y / x.where(x > 0)).dropna()):.2f}")
        # range as an alternative to SD
        a_rng = both[f"{m_}_a_max"] - both[f"{m_}_a_min"]
        h_rng = both[f"{m_}_h_max"] - both[f"{m_}_h_min"]
        okr = a_rng.notna() & h_rng.notna()
        print(f"    {'':10s} min-max range: holo > apo in {100 * (h_rng[okr] > a_rng[okr]).mean():.1f}% of entries "
              f"(median range apo {a_rng[okr].median():.2f} vs holo {h_rng[okr].median():.2f})")
    # but holo sets are larger; compare at matched size: subsample? -- report the spread gap in
    # entries where n_apo and n_holo are within a factor of two of each other
    bal = both[(both.n_holo <= 2 * both.n_apo) & (both.n_apo <= 2 * both.n_holo)]
    print(f"\n  restricted to entries with n_apo and n_holo within a factor of two (n = {len(bal):,}), "
          "holo spread > apo spread in: " +
          ", ".join(f"{m_} {100 * ((bal[f'{m_}_h_sd'] > bal[f'{m_}_a_sd'])[bal[f'{m_}_h_sd'].notna() & bal[f'{m_}_a_sd'].notna()]).mean():.1f}%"
                    for m_ in METRICS))

    print("\n  conformers per state (a holo 'state' = distinct set of ligand codes in the pocket; apo has one state):")
    ok = d.n_holo >= 1
    cps_h = d.loc[ok, "n_holo"] / d.loc[ok, "n_holo_lig_states"].where(d.loc[ok, "n_holo_lig_states"] > 0)
    print(f"    all entries (n = {int(ok.sum()):,}): median holo ligand states per entry {d.loc[ok, 'n_holo_lig_states'].median():.0f} "
          f"(mean {d.loc[ok, 'n_holo_lig_states'].mean():.1f}); median holo conformers per state {cps_h.median():.1f} "
          f"vs median apo conformers (one state) {d.loc[ok, 'n_apo'].median():.0f}")
    one = d.n_holo_lig_states == 1
    print(f"    entries whose holo conformers all carry the same ligand set: {100 * one[ok].mean():.1f}%")
    print(f"    distinct PDB entries per class: median apo {d.n_apo_pdb.median():.0f}, holo {d.n_holo_pdb.median():.0f}; "
          f"conformers per PDB entry: apo {(d.n_apo / d.n_apo_pdb.where(d.n_apo_pdb > 0)).median():.2f}, "
          f"holo {(d.n_holo / d.n_holo_pdb.where(d.n_holo_pdb > 0)).median():.2f}")
    print("\n  by AlphaFold class of the entry (UNCLUSTERED level):")
    for cl in ["apo", "holo"]:
        s = d[d.pocket_class == cl]
        cps = s.n_holo / s.n_holo_lig_states.where(s.n_holo_lig_states > 0)
        sb = both[both.pocket_class == cl]
        vol = (sb.volume_h_sd / sb.volume_h_mean.where(sb.volume_h_mean > 0)) > (sb.volume_a_sd / sb.volume_a_mean.where(sb.volume_a_mean > 0))
        print(f"    AF-{cl:4s} n = {len(s):,}: median n_apo {s.n_apo.median():.0f}, n_holo {s.n_holo.median():.0f}, "
              f"holo ligand states {s.n_holo_lig_states.median():.0f}, holo conformers per state {cps.median():.1f}; "
              f"holo volume-CV > apo volume-CV in {100 * vol[vol.notna()].mean():.1f}% of the {int(vol.notna().sum()):,} entries with >= 3 of each")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["collect", "summarise"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", default=ENTRIES_CSV)
    a = ap.parse_args()
    if a.mode == "collect":
        collect(a.limit, a.procs, a.out)
    else:
        summarise(a.out)
