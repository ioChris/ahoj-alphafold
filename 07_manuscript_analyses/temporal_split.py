# -*- coding: utf-8 -*-
"""Temporal split: does the AlphaFold pocket follow what AlphaFold could have seen, or the protein?
Backs Supplementary Note 5 (release eras, tests A-D) and the size-balanced draw coverage quoted in
Supplementary Note 3 (section E of `summarise`).

Every experimental conformer of a site is put in one of three eras by its PDB RELEASE date (the convention of
Škrinjar et al. 2025, bioRxiv 10.1101/2025.02.03.636309, which split by release date, not deposition date):
  E1  released on or before 2018-04-30: AlphaFold2's training cut-off (Jumper et al. 2021, Methods), so seen in
      training and also available as a template;
  E2  released 2018-05-01 to 2021-02-15: not in training, but available to the database models as templates
      (PDB70 of 10 Feb 2021 and PDB SEQRES of 15 Feb 2021; Tunyasuvunakool et al. 2021, Data availability; the
      latest template in any v2c v4 file was deposited 2021-01-20);
  E3  released after 2021-02-15: seen by neither route.
(Škrinjar's own cut-off, 2021-09-30, is that of AlphaFold3 and the co-folding methods, not of these models.)

Three tests:
  A. The balance. Which era's apo/holo balance predicts the model's class (size-balanced P(holo) of
     balanced_classification.py, and the released P(holo))? The eras differ in size, and a larger count is a less
     noisy ratio, so the larger era is thinned to the smaller one's size before the comparison.
  B. The resemblance. Within the model's own state, is the model more typical of the E1 conformers than of the E3
     ones? Typicality = the share of a set's members whose mean distance to the other members exceeds the model's
     mean distance to the members (0.5 for a typical member, high = central). The positive control is the same
     statistic for the conformers themselves: an E3 conformer scored against the E1 set and, left out, against the
     rest of E3 (and vice versa), which shows how strongly real structures cluster by era.
  C. The templates. The v4 files list each model's PDB templates (ID and chain). Where a template is one of the
     site's own conformers: does its state go with the model's class, and is it unusually close to the model?

    python temporal_split.py dates        -> temporal_split_release_eras.csv   (RCSB search API, release dates)
    python temporal_split.py templates    -> temporal_split_templates.csv      (the local v4/v3 AFDB store)
    python temporal_split.py collect [--limit N] [--procs 22]  -> temporal_split_entries.parquet
    python temporal_split.py summarise > temporal_split.out.txt  -> temporal_split_summary.csv
All files are in the manuscript-analyses folder (ma_paths.OUT). The two input tables and the per-entry table are
regenerable; the summary CSV and the stdout are the published record. `summarise` also reads
balanced_classification.py's per-entry CSV (balanced_entries.csv).
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import argparse
import gzip
import json
import os
import shlex
import sys
import time
import urllib.request
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import balanced_classification as bc  # noqa: E402  (paths, curation, job list, size-balanced labels)

TRAIN_CUT = "2018-04-30"
TPL_CUT = "2021-02-15"
ERA_LAB = {1: "E1 (<= 2018-04-30, training)", 2: "E2 (2018-05 to 2021-02-15, templates only)",
           3: "E3 (after 2021-02-15, unseen)"}
DATES_CSV = P.out("temporal_split_release_eras.csv")
TPL_CSV = P.out("temporal_split_templates.csv")
ENTRIES = P.out("temporal_split_entries.parquet")
SUMMARY_CSV = P.out("temporal_split_summary.csv")
AFDB = str(P.anchor("afdb_store_dir", "AHOJ_AFDB_STORE_DIR"))     # AFDB model store with v4/ and v3/ subfolders
RCSB = "https://search.rcsb.org/rcsbsearch/v2/query"
# PDB entries.idx (deposition dates; the batch proxy of test A)
ENTRIES_IDX = os.path.join(str(P.anchor("pdb_resolutions_dir", "AHOJ_PDB_RESOLUTIONS_DIR")), "entries.idx")
METRICS = ["pocket_rms", "sasa", "volume", "RoG"]
MAIN3 = ["pocket_rms", "sasa", "RoG"]          # volume ties on its zero spike; reported separately
LABEL = {"pocket_rms": "pocket RMSD", "sasa": "SASA", "volume": "volume", "RoG": "radius of gyration"}
CAP = 300          # members per (state, era) set
R_CTRL = 10        # pseudo-models per era per set for the control
NMIN = 3           # members per (state, era) set for test B
TIE = 1e-9


# ----------------------------------------------------------------------------------------------- inputs

def _rcsb_ids(lo, hi):
    q = {"query": {"type": "terminal", "service": "text",
                   "parameters": {"attribute": "rcsb_accession_info.initial_release_date", "operator": "range",
                                  "value": {"from": lo, "to": hi, "include_lower": True, "include_upper": True}}},
         "return_type": "entry", "request_options": {"return_all_hits": True}}
    req = urllib.request.Request(RCSB, data=json.dumps(q).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                if r.status == 204:
                    return []
                return [x["identifier"] for x in json.load(r).get("result_set", [])]
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1} for {lo}..{hi}: {e}", flush=True)
            time.sleep(10)
    raise RuntimeError(f"RCSB query failed for {lo}..{hi}")


def dates():
    """Release year and era of every current PDB entry, from date-range queries (no per-ID lookups)."""
    rows = []
    for y in range(1970, int(time.strftime("%Y")) + 1):
        if y == 2018:
            spans = [(f"{y}-01-01T00:00:00Z", f"{TRAIN_CUT}T23:59:59Z", 1),
                     ("2018-05-01T00:00:00Z", f"{y}-12-31T23:59:59Z", 2)]
        elif y == 2021:
            spans = [(f"{y}-01-01T00:00:00Z", f"{TPL_CUT}T23:59:59Z", 2),
                     ("2021-02-16T00:00:00Z", f"{y}-12-31T23:59:59Z", 3)]
        else:
            spans = [(f"{y}-01-01T00:00:00Z", f"{y}-12-31T23:59:59Z", 1 if y < 2018 else (2 if y < 2021 else 3))]
        for lo, hi, era in spans:
            ids = _rcsb_ids(lo, hi)
            rows += [(i.lower(), y, era) for i in ids]
            print(f"  {lo[:10]} .. {hi[:10]}  era {era}: {len(ids):,}", flush=True)
    d = pd.DataFrame(rows, columns=["pdb_id", "release_year", "era"]).drop_duplicates("pdb_id")
    d.to_csv(DATES_CSV, index=False)
    print(f"wrote {DATES_CSV}: {len(d):,} entries (queried {time.strftime('%Y-%m-%d')})")
    print(d.era.value_counts().sort_index().to_string())


def _loop(txt, cat):
    """Rows of an mmCIF loop as dicts (quoted values kept whole)."""
    i = txt.find(f"loop_\n{cat}.")
    if i < 0:
        return []
    block = txt[i + 6:].split("\n#", 1)[0].strip().splitlines()
    cols = [ln.strip()[len(cat) + 1:] for ln in block if ln.startswith(cat + ".")]
    data = [ln for ln in block if not ln.startswith(cat + ".")]
    out = []
    for ln in data:
        vals = shlex.split(ln)
        if len(vals) == len(cols):
            out.append(dict(zip(cols, vals)))
    return out


def templates():
    rows, nfile, nnone = [], 0, 0
    for sub, tag in (("v4", "model_v4"), ("v3", "model_v3")):
        root = os.path.join(AFDB, sub)
        if not os.path.isdir(root):
            continue
        for f in sorted(os.listdir(root)):
            if not f.endswith(f"-{tag}.cif.gz"):
                continue
            nfile += 1
            acc = f.split("-")[1]
            with gzip.open(os.path.join(root, f), "rt") as fh:
                txt = fh.read()
            ref = {r["template_id"]: r["db_accession_code"] for r in _loop(txt, "_ma_template_ref_db_details")
                   if r.get("db_name") == "PDB"}
            det = _loop(txt, "_ma_template_details")
            if not det:
                nnone += 1
            for r in det:
                if r.get("template_id") in ref:
                    rows.append((acc, sub, r["template_id"], ref[r["template_id"]].lower(),
                                 r.get("template_auth_asym_id", "")))
    d = pd.DataFrame(rows, columns=["acc", "version", "template_id", "pdb_id", "chain"])
    d.to_csv(TPL_CSV, index=False)
    print(f"wrote {TPL_CSV}: {len(d):,} template rows from {nfile:,} model files ({nnone:,} without templates); "
          f"{d.acc.nunique():,} accessions with at least one")


# ----------------------------------------------------------------------------------------------- worker

_G = {}


def _init():
    bc._init()
    e = pd.read_csv(DATES_CSV, dtype={"pdb_id": str})
    _G["era"] = dict(zip(e.pdb_id, e.era.astype(int)))
    _G["year"] = dict(zip(e.pdb_id, e.release_year.astype(int)))
    t = pd.read_csv(TPL_CSV, dtype=str, keep_default_na=False)
    _G["tpl"] = {a: set(zip(g.pdb_id, g.chain)) for a, g in t.groupby("acc")}
    dep = {}
    with open(ENTRIES_IDX, encoding="latin-1") as fh:
        for line in fh:
            q = line.split("\t")
            if len(q) > 2 and len(q[0]) == 4:
                dep[q[0].lower()] = q[2]
    _G["dep"] = dep


def _members(x, rng):
    x = x[np.isfinite(x)]
    if x.size > CAP:
        x = rng.choice(x, CAP, replace=False)
    return x


def _side(y, med_a, med_b):
    """1 if y is nearer med_a than med_b, 0 if farther, 0.5 on a tie."""
    da, db = abs(y - med_a), abs(y - med_b)
    return 1.0 if da < db - TIE else (0.0 if da > db + TIE else 0.5)


def _work(job):
    job_name, tgz, seed = job
    warnings.simplefilter("ignore", RuntimeWarning)
    try:
        loaded = bc._load(tgz)
        if loaded is None:
            return {"job_name": job_name, "error": "no global_results"}
        df_c, apo, holo, df_af = loaded
        if df_af.empty or len(apo) < 1 or len(holo) < 1:
            return {"job_name": job_name, "error": "no model / empty class"}
        rng = np.random.default_rng(seed)
        acc = str(df_af["UNPs"].iloc[0]).split("-")[0].split(",")[0].strip()
        ref = pd.concat([apo.assign(_s="apo"), holo.assign(_s="holo")], ignore_index=True)
        sid = ref["structure"].astype(str).str.lower()
        ref["_era"] = [_G["era"].get(x, 0) for x in sid]                 # 0 = not in the current PDB
        ref["_year"] = [_G["year"].get(x, 0) for x in sid]
        ref["_dep"] = [_G["dep"].get(x, x) for x in sid]                # unknown date -> its own event
        ref["_x"] = ref["exp_method"].astype(str).str.upper().str.contains("X-RAY")
        tpls = _G["tpl"].get(acc, set())
        tpl_ids = {p for p, _ in tpls}
        ch = ref["chains3"].fillna(ref["chains2"]).astype(str).str.split("-")
        ref["_tpl"] = [s in tpl_ids and any((s, c) in tpls for c in cs) for s, cs in zip(sid, ch)]
        rec = {"job_name": job_name, "acc": acc, "n_tpl_model": len(tpls),
               "n_tpl_pdb_in_set": int(sid[sid.isin(tpl_ids)].nunique())}
        for s in ("apo", "holo"):
            for e in (0, 1, 2, 3):
                g = ref[(ref._s == s) & (ref._era == e)]
                rec[f"n_{s}_{e}"] = len(g)
                rec[f"nx_{s}_{e}"] = int(g._x.sum())
                rec[f"ev_{s}_{e}"] = int(g._dep.nunique())
            t = ref[(ref._s == s) & ref._tpl]
            rec[f"n_tpl_{s}"] = len(t)
            rec[f"tpl_eras_{s}"] = ",".join(str(v) for v in sorted(set(t._era)))
        # ---- test B: where the seen and unseen conformers of a state differ, which side is the model on?
        # 13 = E1 against E3; 23 = E2 against E3; 1E / 1L = E1 against the E3 conformers released in 2021-2022 /
        # in 2023 or later (AlphaFold-assisted structure solution grew after the July 2021 release)
        yr = ref["_year"].to_numpy()
        for var, sub_mask in (("all", np.ones(len(ref), bool)), ("x", ref._x.to_numpy())):
            for s in ("apo", "holo"):
                base = sub_mask & (ref._s == s).to_numpy()
                eras = ref._era.to_numpy()
                masks = {1: base & (eras == 1), 2: base & (eras == 2), 3: base & (eras == 3),
                         "3e": base & (eras == 3) & (yr <= 2022), "3l": base & (eras == 3) & (yr >= 2023)}
                for met in METRICS:
                    m = pd.to_numeric(df_af[met], errors="coerce").iloc[0]
                    if not np.isfinite(m):
                        continue
                    vals = pd.to_numeric(ref[met], errors="coerce").to_numpy(float)
                    X = {e: _members(vals[mk], rng) for e, mk in masks.items()}
                    for tag, ea, eb in (("13", 1, 3), ("23", 2, 3), ("1E", 1, "3e"), ("1L", 1, "3l")):
                        if tag != "13" and var == "x":
                            continue
                        A_, B_ = X[ea], X[eb]
                        if A_.size < NMIN or B_.size < NMIN:
                            continue
                        w = np.sqrt((A_.var(ddof=1) + B_.var(ddof=1)) / 2)
                        ma, mb = np.median(A_), np.median(B_)
                        key = f"B{tag}_{var}_{s}_{met}"
                        rec[f"{key}_na"], rec[f"{key}_nb"] = A_.size, B_.size
                        rec[f"{key}_d"] = float(abs(ma - mb) / w) if w > 0 else 0.0
                        rec[f"{key}_m"] = _side(m, ma, mb)
                        ia = rng.choice(A_.size, min(R_CTRL, A_.size), replace=False)
                        ib = rng.choice(B_.size, min(R_CTRL, B_.size), replace=False)
                        rec[f"{key}_ca"] = float(np.mean([_side(A_[i], np.median(np.delete(A_, i)), mb) for i in ia]))
                        rec[f"{key}_cb"] = float(np.mean([_side(B_[i], ma, np.median(np.delete(B_, i))) for i in ib]))
        # ---- test C2: how close the model's own templates are to it, against their own centrality
        if ref._tpl.any():
            rc, rz = [], []
            for met in MAIN3:
                m = pd.to_numeric(df_af[met], errors="coerce").iloc[0]
                if not np.isfinite(m):
                    continue
                vals = pd.to_numeric(ref[met], errors="coerce").to_numpy(float)
                for s in ("apo", "holo"):
                    ok = np.flatnonzero(np.isfinite(vals) & (ref._s == s).to_numpy())
                    if ok.size < 3:
                        continue
                    is_t = ref._tpl.to_numpy()[ok]
                    tpos, opos = ok[is_t], ok[~is_t]
                    if not tpos.size:
                        continue
                    if ok.size > CAP:
                        keep = max(CAP - tpos.size, 2)
                        opos = rng.choice(opos, min(keep, opos.size), replace=False)
                    idx = np.concatenate([tpos, opos])
                    x = vals[idx]
                    D = np.abs(x[:, None] - x[None, :]).sum(axis=1) / (x.size - 1)
                    dm = np.abs(x - m)
                    for j in range(tpos.size):
                        rc.append(float((np.sum(dm < dm[j] - TIE) + 0.5 * (np.sum(np.abs(dm - dm[j]) <= TIE) - 1))
                                        / (x.size - 1)))
                        rz.append(float((np.sum(D < D[j] - TIE) + 0.5 * (np.sum(np.abs(D - D[j]) <= TIE) - 1))
                                        / (x.size - 1)))
            if rc:
                rec["C_rank_closest"] = float(np.nanmean(rc))
                rec["C_rank_central"] = float(np.nanmean(rz))
        return rec
    except Exception as e:  # noqa: BLE001
        return {"job_name": job_name, "error": f"{type(e).__name__}: {e}"}


def collect(limit, procs):
    jobs = bc._jobs(limit)
    out = ENTRIES if not limit else P.out(f"temporal_split_entries_limit{limit}.parquet")
    print(f"entries: {len(jobs):,}  procs={procs}  -> {out}", flush=True)
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
    d.to_parquet(out, index=False)
    print(f"wrote {out}  rows={len(d):,}  {(time.time() - t0) / 60:.1f} min", flush=True)
    if "error" in d:
        print(d["error"].dropna().value_counts().head().to_string(), flush=True)


# ----------------------------------------------------------------------------------------------- summary

def _ratio(h, a):
    return np.log2((h + 0.5) / (a + 0.5))


def _rank(x):
    return pd.Series(x).rank().to_numpy()


def _spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(_rank(x[ok]), _rank(y[ok]))[0, 1]), int(ok.sum())


def _partial(y, x, z):
    """Spearman partial correlation of y and x given z (ranks residualized on z)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    ry, rx, rz = _rank(y[ok]), _rank(x[ok]), _rank(z[ok])
    A = np.column_stack([np.ones(ok.sum()), rz])
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    return float(np.corrcoef(ey, ex)[0, 1])


def _thin(a, h, n, rng):
    """Draw n of the a + h structures without replacement -> (a', h')."""
    hh = rng.hypergeometric(h, a, n)
    return n - hh, hh


def _load(path):
    d = pd.read_parquet(path)
    if "error" in d:
        print(f"entries with an error: {d['error'].notna().sum():,}")
        d = d[d["error"].isna()].copy()
    b = pd.read_csv(bc.ENTRIES_CSV, usecols=["job_name", "bal_p_holo", "k", "n_apo", "n_holo_pool"]).rename(
        columns={"k": "bal_k", "n_apo": "bal_n_apo", "n_holo_pool": "bal_n_holo_pool"})
    rel = pd.read_csv(bc.SUMMARY, usecols=["job_name", "pocket_avg_P(holo)", "n_apo", "n_holo"], low_memory=False)
    uni = pd.read_csv(bc.UNI, usecols=["threshold", "rep_job", "pocket_class"], low_memory=False)
    uni = uni[uni.threshold == "UNCLUSTERED"].drop(columns="threshold").rename(columns={"rep_job": "job_name"})
    d = d.merge(b, on="job_name", how="left").merge(rel, on="job_name", how="left").merge(uni, on="job_name", how="left")
    d["p_rel"] = d["pocket_avg_P(holo)"]
    d["bal_holo"] = np.where(d.bal_p_holo > 0.5, 1.0, np.where(d.bal_p_holo < 0.5, 0.0, np.nan))
    d["rel_holo"] = np.where(d.pocket_class == "holo", 1.0, np.where(d.pocket_class == "apo", 0.0, np.nan))
    d["ratio_all"] = np.log2((d.n_holo + 1) / d.n_apo)
    for e in (1, 2, 3):
        d[f"a{e}"], d[f"h{e}"] = d[f"n_apo_{e}"], d[f"n_holo_{e}"]
        d[f"t{e}"] = d[f"a{e}"] + d[f"h{e}"]
        d[f"r{e}"] = _ratio(d[f"h{e}"], d[f"a{e}"])
    return d


def summarise(path):
    d = _load(path)
    rows = []
    print(f"entries: {len(d):,}\n")

    # ---- 0. the record by era
    print("== 0. The record by release era (curated reference conformers, query structure included) ==")
    tot = {e: int(d[f"n_apo_{e}"].sum() + d[f"n_holo_{e}"].sum()) for e in (0, 1, 2, 3)}
    allc = sum(tot.values())
    for e in (1, 2, 3, 0):
        lab = ERA_LAB.get(e, "not in the current PDB")
        sa, sh = int(d[f"n_apo_{e}"].sum()), int(d[f"n_holo_{e}"].sum())
        print(f"  {lab:46s} {tot[e]:>11,} conformers ({100 * tot[e] / allc:5.1f}%), apo {sa:,}, holo {sh:,} "
              f"(holo share {100 * sh / max(sa + sh, 1):.1f}%)")
        rows.append(dict(block="record", key=f"era{e}", value=tot[e], share=100 * tot[e] / allc))
    for e in (1, 2, 3):
        print(f"  entries with any {ERA_LAB[e][:2]} conformer: {(d[f't{e}'] > 0).mean() * 100:.1f}%; with at least one "
              f"apo and one holo: {((d[f'a{e}'] > 0) & (d[f'h{e}'] > 0)).mean() * 100:.1f}%")
    both13 = (d.t1 > 0) & (d.t3 > 0)
    print(f"  entries with E1 and E3 conformers: {both13.sum():,} ({100 * both13.mean():.1f}%)")

    # ---- A. the balance
    print("\n== A. Which era's apo/holo balance goes with the model's class? Spearman rho with P(holo) ==")
    print("   (ratio = log2((holo + 0.5)/(apo + 0.5)) within the era; 'thinned' = the larger era subsampled to the "
          "smaller one's size, seed 0; 'events' = distinct deposition dates instead of structures, which collapses "
          "group depositions; [ ] = 95% bootstrap interval over proteins, 200 draws)")
    for unit, pa, ph in (("structures", "n_apo", "n_holo"), ("events", "ev_apo", "ev_holo")):
        rng = np.random.default_rng(0)
        cnt = {e: (d[f"{pa}_{e}"].to_numpy().astype(int), d[f"{ph}_{e}"].to_numpy().astype(int)) for e in (1, 2, 3)}
        tt = {e: cnt[e][0] + cnt[e][1] for e in (1, 2, 3)}
        sel = (tt[1] >= 4) & (tt[3] >= 4)
        A = d[sel].copy()
        n = np.minimum(tt[1][sel], tt[3][sel])
        thinned = {}
        for e in (1, 3):
            a, h = cnt[e][0][sel].copy(), cnt[e][1][sel].copy()
            for i in np.flatnonzero(a + h > n):
                a[i], h[i] = _thin(a[i], h[i], n[i], rng)
            thinned[e] = _ratio(h, a)
        A["r1t"], A["r3t"] = thinned[1], thinned[3]
        print(f"\n  -- {unit} -- entries with at least 4 in E1 and in E3: {len(A):,} ({A.acc.nunique():,} proteins); "
              f"median E1 / E3 {np.median(tt[1][sel]):.0f} / {np.median(tt[3][sel]):.0f}; common size after thinning, "
              f"median {np.median(n):.0f}")
        for ylab, ycol in (("size-balanced P(holo)", "bal_p_holo"), ("released P(holo)", "p_rel")):
            y = A[ycol].to_numpy()
            r1v, r3v = A.r1t.to_numpy(), A.r3t.to_numpy()

            def stats(ix, y=y, r1v=r1v, r3v=r3v):
                return (_spearman(r1v[ix], y[ix])[0], _spearman(r3v[ix], y[ix])[0],
                        _partial(y[ix], r1v[ix], r3v[ix]), _partial(y[ix], r3v[ix], r1v[ix]))
            pt = stats(np.arange(len(A)))
            bs = _boot(A.acc.to_numpy(), stats)
            ci = [f"{pt[k]:+.3f} [{bs[k][0]:+.3f}, {bs[k][1]:+.3f}]" for k in range(4)]
            print(f"  {ylab:22s} n={len(A):,}: E1 {ci[0]}; E3 {ci[1]}; partial E1|E3 {ci[2]}; E3|E1 {ci[3]}")
            rows.append(dict(block=f"A_{unit}", key=ylab, n=len(A), r1t=pt[0], r3t=pt[1], p1=pt[2], p3=pt[3],
                             r1t_lo=bs[0][0], r1t_hi=bs[0][1], r3t_lo=bs[1][0], r3t_hi=bs[1][1],
                             p1_lo=bs[2][0], p1_hi=bs[2][1], p3_lo=bs[3][0], p3_hi=bs[3][1]))
        r13 = _spearman(A.r1t.to_numpy(), A.r3t.to_numpy())[0]
        print(f"  rho(E1 ratio, E3 ratio), thinned: {r13:+.3f}")
        rows.append(dict(block=f"A_{unit}", key="r13", value=r13))
        print("  Discordant eras (thinned; |ratio| >= 1 in opposite directions): share of AF-Holo calls")
        for lab, msk in (("E1 holo-rich, E3 apo-rich", (A.r1t >= 1) & (A.r3t <= -1)),
                         ("E1 apo-rich, E3 holo-rich", (A.r1t <= -1) & (A.r3t >= 1)),
                         ("both holo-rich", (A.r1t >= 1) & (A.r3t >= 1)),
                         ("both apo-rich", (A.r1t <= -1) & (A.r3t <= -1))):
            g = A[msk]
            sb, sr = 100 * g.bal_holo.mean(), 100 * g.rel_holo.mean()
            print(f"   {lab:26s} n={len(g):>6,}  size-balanced {sb:5.1f}%  released {sr:5.1f}%")
            rows.append(dict(block=f"A_discordant_{unit}", key=lab, n=len(g), bal=sb, rel=sr))
        if unit == "structures":
            sel3 = (tt[1] >= 4) & (tt[2] >= 4) & (tt[3] >= 4)
            nn = np.minimum(np.minimum(tt[1][sel3], tt[2][sel3]), tt[3][sel3])
            th = {}
            for e in (1, 2, 3):
                a, h = cnt[e][0][sel3].copy(), cnt[e][1][sel3].copy()
                for i in np.flatnonzero(a + h > nn):
                    a[i], h[i] = _thin(a[i], h[i], nn[i], rng)
                th[e] = _ratio(h, a)
            y = d.bal_p_holo.to_numpy()[sel3]
            rr = {e: _spearman(th[e], y)[0] for e in (1, 2, 3)}
            print(f"  three eras, all thinned to the smallest (n={sel3.sum():,}, median size {np.median(nn):.0f}): "
                  f"size-balanced P(holo) vs E1 {rr[1]:+.3f}, E2 {rr[2]:+.3f}, E3 {rr[3]:+.3f}")
            rows.append(dict(block="A3", key="bal", n=int(sel3.sum()), r1=rr[1], r2=rr[2], r3=rr[3]))

    # ---- B. the side
    print("\n== B. Where the seen (E1) and unseen (E3) conformers of the model's own state differ, which side is the "
          "model on? ==")
    print("   share of pockets with the model nearer the E1 median than the E3 median (0.5 = tie or no preference). "
          "Controls, each held out of its own era: E1 conformers (a seen structure) and E3 conformers (an unseen one). "
          "f = (model - E3 conformer)/(E1 conformer - E3 conformer): 1 = sides like a seen structure, 0 = like an "
          "unseen one; [ ] = 95% bootstrap over proteins. d = |median E1 - median E3| / within-era SD.")
    for tag, la, lb, variants in (("13", "E1", "E3", (("all", "all methods"), ("x", "X-ray only"))),
                                  ("23", "E2", "E3", (("all", "all methods"),)),
                                  ("1E", "E1", "E3 2021-22", (("all", "all methods"),)),
                                  ("1L", "E1", "E3 2023+", (("all", "all methods"),))):
        for var, vlab in variants:
            print(f"\n  -- {la} against {lb}, {vlab} --")
            print(f"  {'metric':20s} {'d >=':>5s} {'n':>8s} {'model':>7s} {la + ' conf':>8s} {lb + ' conf':>8s}  f")
            for met in METRICS + ["three"]:
                mets = MAIN3 if met == "three" else [met]
                cols = {k: np.concatenate([_own_state(d, var, mt, k, tag) for mt in mets]) for k in ("m", "d", "ca", "cb")}
                accs = np.concatenate([d.acc.to_numpy() for _ in mets])
                for thr in (0.0, 0.5, 1.0):
                    ok = np.isfinite(cols["m"]) & (cols["d"] >= thr)
                    if ok.sum() < 50:
                        continue
                    m_, a_, b_ = cols["m"][ok], cols["ca"][ok], cols["cb"][ok]

                    def fstat(ix, m_=m_, a_=a_, b_=b_):
                        return ((np.mean(m_[ix]) - np.mean(b_[ix])) / (np.mean(a_[ix]) - np.mean(b_[ix])),)
                    f = fstat(np.arange(ok.sum()))[0]
                    lo, hi = _boot(accs[ok], fstat)[0]
                    lab = "three (pooled)" if met == "three" else LABEL[met]
                    print(f"  {lab:20s} {thr:5.1f} {ok.sum():8,d} {np.mean(m_):7.3f} {np.mean(a_):8.3f} {np.mean(b_):8.3f}  "
                          f"{f:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                    rows.append(dict(block=f"B{tag}_{var}", key=f"{met} | d>={thr}", n=int(ok.sum()), model=np.mean(m_),
                                     ca=np.mean(a_), cb=np.mean(b_), f=f, f_lo=lo, f_hi=hi))

    # ---- D. proteins AlphaFold could not have seen
    print("\n== D. Does the class follow the evidence ratio where none of the site's structures could have been seen? ==")
    print("   AF-Holo share (size-balanced label) by the full-record evidence ratio (log2), by what was available; "
          "rho = Spearman of size-balanced P(holo) with the ratio; [ ] = 95% bootstrap over proteins")
    avail = np.select([d.t1 > 0, (d.t1 == 0) & (d.t2 > 0), (d.t1 == 0) & (d.t2 == 0) & (d.t3 > 0)],
                      ["some structure in training (E1)", "none in training, some as template (E2)",
                       "none in training or templates (E3 only)"], "other")
    d["rbin2"] = pd.cut(d.ratio_all, [-np.inf, -1, 0, 1, 2, 3, np.inf])
    for lab in ("some structure in training (E1)", "none in training, some as template (E2)",
                "none in training or templates (E3 only)"):
        g = d[(avail == lab) & d.bal_p_holo.notna()]
        y, x = g.bal_p_holo.to_numpy(), g.ratio_all.to_numpy()

        def st(ix, y=y, x=x):
            return (_spearman(x[ix], y[ix])[0],)
        rho = st(np.arange(len(g)))[0]
        lo, hi = _boot(g.acc.to_numpy(), st)[0]
        by = g.groupby("rbin2", observed=True).bal_holo.agg(["size", "mean"])
        parts = [f"{str(k)} {100 * r['mean']:.1f}% (n {int(r['size']):,})" for k, r in by.iterrows()]
        print(f"  {lab:42s} n={len(g):>7,} ({g.acc.nunique():,} proteins)  rho {rho:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print("     " + "; ".join(parts))
        rows.append(dict(block="D", key=lab, n=len(g), proteins=g.acc.nunique(), rho=rho, lo=lo, hi=hi))
        for k, r in by.iterrows():
            rows.append(dict(block="D_bins", key=f"{lab} | {k}", n=int(r["size"]), value=100 * r["mean"]))
        if lab.startswith("none in training or"):
            nt = g.n_tpl_pdb_in_set.gt(0).mean() * 100
            print(f"     (of these, {nt:.1f}% have a template PDB entry among the site's conformers, which cannot happen "
                  f"for a structure released after the template library: a check on the era assignment)")
            rows.append(dict(block="D_check", key="E3-only with a template in the set", value=nt, n=len(g)))

    # ---- C. templates
    print("\n== C. The model's own templates ==")
    has = d.n_tpl_model > 0
    print(f"  entries whose model lists PDB templates: {100 * has.mean():.1f}%; with a template PDB entry among the "
          f"site's conformers: {100 * (d.n_tpl_pdb_in_set > 0).mean():.1f}%; with a chain-matched template conformer: "
          f"{100 * ((d.n_tpl_apo + d.n_tpl_holo) > 0).mean():.1f}%")
    d["tpl_state"] = np.select([(d.n_tpl_holo > 0) & (d.n_tpl_apo == 0), (d.n_tpl_apo > 0) & (d.n_tpl_holo == 0),
                                (d.n_tpl_apo > 0) & (d.n_tpl_holo > 0)], ["holo", "apo", "both"], "none at the site")
    d["rbin"] = pd.cut(d.ratio_all, [-np.inf, -1, 0, 1, 2, 3, np.inf])
    print("  AF-Holo share (size-balanced label) by the state of the model's template at this site, within evidence-ratio "
          "bins (log2):")
    tab = d[has].groupby(["rbin", "tpl_state"], observed=True).bal_holo.agg(["size", "mean"]).unstack("tpl_state")
    for rb in tab.index:
        parts = []
        for stt in ("apo", "holo", "both", "none at the site"):
            if ("size", stt) in tab.columns and np.isfinite(tab.loc[rb, ("size", stt)]):
                nn_ = int(tab.loc[rb, ("size", stt)])
                parts.append(f"{stt} {100 * tab.loc[rb, ('mean', stt)]:.1f}% (n {nn_:,})")
                rows.append(dict(block="C_state", key=f"{rb} | {stt}", n=nn_, value=100 * tab.loc[rb, ("mean", stt)]))
        if ("mean", "apo") in tab.columns and ("mean", "holo") in tab.columns:
            dif = 100 * (tab.loc[rb, ("mean", "holo")] - tab.loc[rb, ("mean", "apo")])
            parts.append(f"holo - apo {dif:+.1f} points")
            rows.append(dict(block="C_state_diff", key=str(rb), value=dif))
        print(f"   ratio {str(rb):14s} " + "; ".join(parts))
    for stt in ("apo", "holo", "both", "none at the site"):
        g = d[has & (d.tpl_state == stt)]
        print(f"  overall, template {stt:17s} n={len(g):>7,}  AF-Holo {100 * g.bal_holo.mean():5.1f}% (size-balanced), "
              f"{100 * g.rel_holo.mean():5.1f}% (released)")
        rows.append(dict(block="C_state_all", key=stt, n=len(g), bal=100 * g.bal_holo.mean(), rel=100 * g.rel_holo.mean()))
    te = (d.tpl_eras_apo.fillna("") + "," + d.tpl_eras_holo.fillna("")).str.split(",").apply(
        lambda v: frozenset(x for x in v if x))
    for elab, eset in (("E1 only (seen in training)", frozenset({"1"})), ("E2 only (templates only)", frozenset({"2"}))):
        for stt in ("apo", "holo"):
            g = d[has & (d.tpl_state == stt) & (te == eset)]
            print(f"  templates at the site all {elab:27s}, template {stt:4s}: n={len(g):>6,}  AF-Holo "
                  f"{100 * g.bal_holo.mean():5.1f}% (size-balanced)")
            rows.append(dict(block="C_state_era", key=f"{elab} | {stt}", n=len(g), bal=100 * g.bal_holo.mean()))
    c = d.dropna(subset=["C_rank_closest"])
    print(f"  template conformers' distance rank to the model among their state's conformers (0 = closest; 0.5 by "
          f"chance), n={len(c):,} entries: median {c.C_rank_closest.median():.3f}, mean {c.C_rank_closest.mean():.3f}; "
          f"their own centrality rank: median {c.C_rank_central.median():.3f}, mean {c.C_rank_central.mean():.3f}")
    rows.append(dict(block="C_rank", key="closest", n=len(c), value=c.C_rank_closest.median(), mean=c.C_rank_closest.mean()))
    rows.append(dict(block="C_rank", key="central", n=len(c), value=c.C_rank_central.median(), mean=c.C_rank_central.mean()))
    # ---- E. coverage: the sampling caps and the outcome label's draws
    print("\n== E. Coverage ==")
    for s_ in ("apo", "holo"):
        na, nb = d.get(f"B13_all_{s_}_sasa_na"), d.get(f"B13_all_{s_}_sasa_nb")
        ok = na.notna()
        print(f"  test B, {s_} sets (SASA, all methods; n={ok.sum():,} entries): E1 set at the {CAP}-member cap in "
              f"{100 * (na[ok] == CAP).mean():.1f}%, E3 set in {100 * (nb[ok] == CAP).mean():.1f}%")
        rows.append(dict(block="E_cap", key=s_, n=int(ok.sum()), e1=100 * (na[ok] == CAP).mean(),
                         e3=100 * (nb[ok] == CAP).mean()))
    k = d.bal_k.to_numpy(float)
    nl = np.maximum(d.bal_n_apo, d.bal_n_holo_pool).to_numpy(float)
    ns = np.minimum(d.bal_n_apo, d.bal_n_holo_pool).to_numpy(float)
    reps = bc.REPS
    cov_l = np.where(nl > k, 1 - (1 - k / nl) ** reps, 1.0)
    cov_s = np.where(ns > k, 1 - (1 - k / ns) ** reps, 1.0)
    pooled = float(((nl * cov_l + ns * cov_s).sum()) / (nl + ns).sum())
    print(f"  size-balanced label: k = min(n_apo, n_holo + 1, {bc.K_CAP}) per class, {reps} draws. Median k "
          f"{np.median(k):.0f}; k = {bc.K_CAP} for {100 * (k == bc.K_CAP).mean():.1f}% of entries, k = 1 for "
          f"{100 * (k == 1).mean():.1f}%")
    print(f"  expected share of the larger class drawn at least once: median {100 * np.median(cov_l):.1f}%, below half "
          f"in {100 * (cov_l < 0.5).mean():.1f}% of entries; of all conformers, pooled over entries, "
          f"{100 * pooled:.1f}% contribute to at least one draw")
    rows.append(dict(block="E_draws", key="k", value=float(np.median(k)), k20=100 * (k == bc.K_CAP).mean(),
                     k1=100 * (k == 1).mean(), cov_l_median=100 * np.median(cov_l),
                     cov_l_below_half=100 * (cov_l < 0.5).mean(), pooled=100 * pooled))

    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f"\nwrote {SUMMARY_CSV}")


def _own_state(d, var, met, k, tag="13"):
    """The column for the model's own (size-balanced) state."""
    a = d.get(f"B{tag}_{var}_apo_{met}_{k}", pd.Series(np.nan, index=d.index))
    h = d.get(f"B{tag}_{var}_holo_{met}_{k}", pd.Series(np.nan, index=d.index))
    return np.where(d.bal_holo == 1, h, np.where(d.bal_holo == 0, a, np.nan)).astype(float)


def _boot(clusters, stat, reps=200, seed=1):
    """95% percentile interval of each output of stat(index array), resampling clusters (proteins)."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(u) + 1))
    members = [order[bounds[c]:bounds[c + 1]] for c in range(len(u))]
    out = []
    for _ in range(reps):
        pick = rng.integers(0, len(u), len(u))
        out.append(stat(np.concatenate([members[c] for c in pick])))
    out = np.array(out)
    return [(float(np.nanpercentile(out[:, k], 2.5)), float(np.nanpercentile(out[:, k], 97.5))) for k in range(out.shape[1])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["dates", "templates", "collect", "summarise"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--procs", type=int, default=22)
    ap.add_argument("--entries", default=ENTRIES)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    if a.mode == "dates":
        dates()
    elif a.mode == "templates":
        templates()
    elif a.mode == "collect":
        collect(a.limit, a.procs)
    else:
        summarise(a.entries)
