# -*- coding: utf-8 -*-
"""How many AlphaFold DB models changed between v4 (AHoJ-DB v2c) and v6 (detection, docking), and what that
means for the v6 model-row refresh.

Every number of the analysis is printed by `report`. It backs the Methods and Supplementary Note 4 statement that
version 6 re-predicted, to a close approximation, the Swiss-Prot and reference-proteome models and kept the
coordinates and per-residue confidence of the rest (`atoms`: exact atom-by-atom identity of every v2c model's v4
and v6 files), and it writes the per-accession model identity (identical / moved / different_atoms) that
afdb_v4_realign.py stratifies Supplementary Note 4 by (afdb_v4_v6_accessions.csv) and that
alphafill_model_version.py joins (v4_v6_atoms.csv). Paths, accession sets, the v6 headers, the archive scan and
the atom reader are those of afdb_v4_sources.py (imported); nothing here writes into the v4 store, the v2c
archives or the v6 folder. Caches go to <manuscript_analyses_dir>/afdb_v4_v6_diff/, the three summary tables to
<manuscript_analyses_dir>.

    python afdb_v4_v6_diff.py atoms   [--procs 12]     -> cache: every v2c model's v4 file against its local v6 file
                                                          (exact identity, sequence, global and pLDDT deviations); the v4
                                                          sequence alone where no v6 file is local
    python afdb_v4_v6_diff.py api     [--threads 6]    -> cache: AFDB API facts (model date, sequence) for the accessions
                                                          with no local v6 file, plus a sample of local ones as a check
    python afdb_v4_v6_diff.py scan    [--procs 10]     -> cache: the AF pocket selection(s) of every analysis entry's
                                                          archive (model accession and residue list)
    python afdb_v4_v6_diff.py pockets [--procs 12]     -> cache: pocket-level movement v4 -> v6 for every distinct
                                                          (model, pocket) of the analysis entries and ST_100 sites
    python afdb_v4_v6_diff.py worklist                 -> cache: per-entry (237,922) and per-site (105,713) worklists;
                                                          the summary tables afdb_v4_v6_{diff_summary,tiers,accessions}.csv
    python afdb_v4_v6_diff.py report  > out.txt        -> every quoted number (reads the caches; the accession routes
                                                          of afdb_v4_sources.labelled() read two EBI FTP listings)

PROBES. `api` reads AFDB API metadata (one small JSON per accession). It downloads no model file. The accessions
without a local v6 file are not fetched here; their count and size are printed for the author to decide.

"Identical" means every polymer atom (chain, residue number and name, atom name) has the same coordinates and
B-factor (pLDDT) in the v6 file as in the v4 file. "Moved" means the same residues and atoms with other
coordinates or pLDDT. "Different atoms" means the residue or atom set differs (a sequence change).
Pocket movement is measured after a superposition on the pocket's own atoms (Kabsch), so it is the change of
the pocket's internal geometry, which is what the model-row metrics (RoG, SASA, volume, pocket RMSD) see.
"""
import argparse
import ast
import collections
import os
import re
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ma_paths as P  # noqa: E402
import afdb_v4_sources as S  # noqa: E402  (paths, accession sets, v6 headers, archive scan, atom reader)

OUT_DIR = os.path.join(str(P.OUT), "afdb_v4_v6_diff")
ATOMS_CSV = os.path.join(OUT_DIR, "v4_v6_atoms.csv")                 # one row per v2c accession
API_CSV = os.path.join(OUT_DIR, "afdb_api_nonlocal.csv")             # accessions with no local v6 file (+ a local sample)
SCAN_CSV = os.path.join(OUT_DIR, "entry_model_pockets.csv")          # one row per AF selection of every analysis entry
POCKETS_CSV = os.path.join(OUT_DIR, "pocket_movement.csv")           # one row per distinct (acc, pocket)
ENTRY_CSV = os.path.join(OUT_DIR, "entry_worklist.csv")              # 237,922 analysis entries
SITE_CSV = os.path.join(OUT_DIR, "st100_site_worklist.csv")          # 105,713 ST_100 sites
AFDS = os.path.join(S.WORK, "afdb_dataset", "st100_run3", "afdb_st_100_dataset.csv")   # detection-side ST_100 sites (union sets)
REFRESH_CSV = os.path.join(S.WORK, "af_centric_rerun", "model_row_refresh_v6.csv")   # the modelv6 prototype's 1,201 rows
PILOT_STRUCT = os.path.join(S.PILOT_WD, "structures")                # cached query crystals of the AF-centric pilot
SRC_STRUCT = os.path.join(S.CACHE, "structures")                     # and of afdb_v4_sources.py validate

SUMMARY_CSV = P.out("afdb_v4_v6_diff_summary.csv")                   # committed: the counts of the summary tables
TIERS_CSV = P.out("afdb_v4_v6_tiers.csv")                            # committed: totals and cost per tolerance tier
ACC_CSV = P.out("afdb_v4_v6_accessions.csv")                         # committed: compact per-accession table (local pairs)

TIERS = [("identical", 0.0), ("lt_0.1", 0.1), ("0.1_0.5", 0.5), ("0.5_1", 1.0), ("ge_1", np.inf)]
SEED = 20260921


# =============================================================================== helpers
def _tier(rms, identical=False, seq_changed=False):
    if seq_changed:
        return "seq_changed"
    if identical or (rms is not None and not np.isnan(rms) and rms == 0.0):
        return "identical"
    if rms is None or np.isnan(rms):
        return "unknown"
    for name, hi in TIERS[1:]:
        if rms < hi:
            return name
    return "ge_1"


TIER_ORDER = ["identical", "lt_0.1", "0.1_0.5", "0.5_1", "ge_1", "seq_changed", "unknown"]


def _kabsch_rmsd(P, Q):
    """RMSD after the optimal superposition of P onto Q (same order)."""
    if len(P) < 3:
        return float("nan")
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum(1).mean()))


def _rog(X, m):
    c = (X * m[:, None]).sum(0) / m.sum()
    return float(np.sqrt((m * ((X - c) ** 2).sum(1)).sum() / m.sum()))


def _model(path):
    """Polymer atoms of a model file: dict (resnum) -> resname, and arrays per atom key."""
    import gemmi
    st = gemmi.read_structure(path)
    st.remove_ligands_and_waters()
    ch = st[0][0]
    res = {r.seqid.num: r.name for r in ch}
    atoms = {(r.seqid.num, a.name): (a.pos.x, a.pos.y, a.pos.z, a.b_iso, a.element.weight) for r in ch for a in r}
    return res, atoms


def _store_path(acc, version):
    return os.path.join(S.STORE, f"v{version}", f"AF-{acc}-F1-model_v{version}.cif.gz")


def _v6_path(acc):
    p = os.path.join(S.AFDB_V6, f"AF-{acc}-F1-model_v6.cif.gz")
    return p if os.path.isfile(p) else None


def _seq(res):
    import gemmi
    return gemmi.one_letter_code([res[k] for k in sorted(res)])


# =============================================================================== atoms
def _atoms_one(job):
    acc, version = job
    rec = {"acc": acc, "v4_version": version}
    p4 = _store_path(acc, version)
    if not os.path.isfile(p4):
        rec["status"] = "no_store_file"
        return rec
    res4, at4 = _model(p4)
    rec.update(n_res_v4=len(res4), n_atoms_v4=len(at4), seq_v4=_seq(res4),
               plddt_v4=float(np.mean([v[3] for k, v in at4.items() if k[1] == "CA"])))
    p6 = _v6_path(acc)
    if p6 is None:
        rec["status"] = "no_local_v6"
        return rec
    res6, at6 = _model(p6)
    rec.update(status="compared", n_res_v6=len(res6), n_atoms_v6=len(at6), seq_v6=_seq(res6),
               plddt_v6=float(np.mean([v[3] for k, v in at6.items() if k[1] == "CA"])))
    rec["seq_same"] = res4 == res6
    rec["atomset_same"] = set(at4) == set(at6)
    common = sorted(set(at4) & set(at6))
    rec["n_atoms_common"] = len(common)
    if not common:
        rec.update(coords_equal=False, plddt_equal=False, identical=False, category="different_atoms")
        return rec
    X4 = np.array([at4[k][:3] for k in common])
    X6 = np.array([at6[k][:3] for k in common])
    B4 = np.array([at4[k][3] for k in common])
    B6 = np.array([at6[k][3] for k in common])
    rec["coords_equal"] = bool(np.array_equal(X4, X6))
    rec["plddt_equal"] = bool(np.array_equal(B4, B6))
    rec["identical"] = rec["atomset_same"] and rec["coords_equal"] and rec["plddt_equal"]
    rec["max_raw_dev"] = float(np.abs(X4 - X6).max())
    ca = [i for i, k in enumerate(common) if k[1] == "CA"]
    rec["ca_rmsd_global"] = _kabsch_rmsd(X4[ca], X6[ca])
    rec["aa_rmsd_global"] = _kabsch_rmsd(X4, X6)
    rec["mean_abs_dplddt"] = float(np.abs(B4[ca] - B6[ca]).mean()) if ca else float("nan")
    rec["max_abs_dplddt"] = float(np.abs(B4[ca] - B6[ca]).max()) if ca else float("nan")
    rec["category"] = "identical" if rec["identical"] else ("moved" if rec["atomset_same"] else "different_atoms")
    return rec


def cmd_atoms(a):
    sets = pd.read_csv(S.SETS_CSV)
    scan = pd.read_csv(S.SCAN_CSV, usecols=["af_models", "af_mat_models", "n_af_sel"], low_memory=False)
    models = S.job_models(scan)
    v3 = {m.group(1) for s in models for m in S.MODEL_RE.finditer(s) if m.group(3) == "3"}
    jobs = [(acc, 3 if acc in v3 else 4) for acc in sets.acc[sets.in_v2c]]
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = pool.map(_atoms_one, jobs, chunksize=32)
    df = pd.DataFrame(rows)
    h = pd.read_csv(S.V6HDR_CSV)[["acc", "deposited", "first_rev"]]
    df = df.merge(h, on="acc", how="left").merge(sets, on="acc", how="left")
    S._save(df, ATOMS_CSV)
    print(f"compared {int((df.status == 'compared').sum()):,} v4/v6 pairs and read {int((df.status == 'no_local_v6').sum()):,} "
          f"v4-only models in {time.time() - t0:.0f} s; wrote {ATOMS_CSV}")
    summarise_atoms(df)


def summarise_atoms(df=None):
    if df is None:
        df = pd.read_csv(ATOMS_CSV, low_memory=False)
    c = df[df.status == "compared"].copy()
    print("\n=== every local v6 file against its v4 store file (exact identity) ===")
    print(f"  v2c accessions {len(df):,} (store files: v4 {int((df.v4_version == 4).sum()):,}, v3 {int((df.v4_version == 3).sum())}); "
          f"with a local v6 file {len(c):,}; v4 only {int((df.status == 'no_local_v6').sum()):,}; "
          f"no store file {int((df.status == 'no_store_file').sum())}")
    cat = c.category.value_counts()
    print(f"  identical {int(cat.get('identical', 0)):,}; moved (same residues and atoms, other coordinates or pLDDT) "
          f"{int(cat.get('moved', 0)):,}; different atoms (sequence or atom set changed) {int(cat.get('different_atoms', 0)):,}")
    print(f"  the earlier split (store verify): identical {int(c.identical.eq(True).sum()):,} ({S._pct(c.identical.eq(True).sum(), len(c)):.1f}%), "
          f"not identical {int(c.identical.eq(False).sum()):,} ({S._pct(c.identical.eq(False).sum(), len(c)):.1f}%)")
    mv = c[c.category == "moved"]
    ce, pe = mv.coords_equal.eq(True), mv.plddt_equal.eq(True)
    print(f"  among moved: coordinates equal but pLDDT differs {int((ce & ~pe).sum())}; "
          f"pLDDT equal but coordinates differ {int((~ce & pe).sum())}; both differ {int((~ce & ~pe).sum()):,}")
    sq = c.seq_same.eq(False)
    print(f"  sequence changed (residue names by number) {int(sq.sum())}; of those with the same residue count "
          f"{int((sq & (c.n_res_v4 == c.n_res_v6)).sum())}, with another residue count "
          f"{int((sq & (c.n_res_v4 != c.n_res_v6)).sum())}")
    print("  by v6 header date:")
    for d, g in c.groupby("deposited"):
        cc = g.category.value_counts()
        print(f"    {d}: {len(g):,} files; identical {int(cc.get('identical', 0)):,}, moved {int(cc.get('moved', 0)):,}, "
              f"different atoms {int(cc.get('different_atoms', 0)):,}")
    for name, col in (("analysis set", "in_analysis"), ("detection set", "in_detection"), ("detection sites", "in_detection_sites")):
        g = c[c[col].eq(True)]
        cc = g.category.value_counts()
        nch = len(g) - int(cc.get('identical', 0))
        print(f"  {name}: {len(g):,} local pairs; identical {int(cc.get('identical', 0)):,} ({S._pct(cc.get('identical', 0), len(g)):.1f}%), "
              f"moved {int(cc.get('moved', 0)):,}, different atoms {int(cc.get('different_atoms', 0)):,}; changed {nch:,} ({S._pct(nch, len(g)):.1f}%)")
    ch = c[c.category != "identical"]
    print(f"  changed models ({len(ch):,}): global C-alpha RMSD after superposition median {ch.ca_rmsd_global.median():.2f} A, "
          f"IQR {ch.ca_rmsd_global.quantile(.25):.2f}-{ch.ca_rmsd_global.quantile(.75):.2f}, p90 {ch.ca_rmsd_global.quantile(.9):.2f}; "
          f"below 0.5 A {int((ch.ca_rmsd_global < 0.5).sum()):,}, 0.5-1 {int(((ch.ca_rmsd_global >= 0.5) & (ch.ca_rmsd_global < 1)).sum()):,}, "
          f"1-2 {int(((ch.ca_rmsd_global >= 1) & (ch.ca_rmsd_global < 2)).sum()):,}, 2 A or more {int((ch.ca_rmsd_global >= 2).sum()):,}")
    print(f"  changed models: global pLDDT v6 - v4 median {(ch.plddt_v6 - ch.plddt_v4).median():+.2f}, mean {(ch.plddt_v6 - ch.plddt_v4).mean():+.2f}; "
          f"per-residue |dpLDDT| mean of means {ch.mean_abs_dplddt.mean():.2f}; models with global pLDDT up {int((ch.plddt_v6 > ch.plddt_v4).sum()):,}, "
          f"down {int((ch.plddt_v6 < ch.plddt_v4).sum()):,}")
    odd = c[(c.deposited == "2022-06-01") & (c.category != "identical")]
    print(f"  dated 2022-06-01 yet not identical: {len(odd)} ({', '.join(sorted(odd.acc))})")


# =============================================================================== api
def _api_one(acc):
    import requests
    rec = {"acc": acc}
    for attempt in range(4):
        try:
            r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=60)
        except Exception as e:  # noqa: BLE001
            rec["api_status"] = f"err:{type(e).__name__}"
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        rec["api_status"] = r.status_code
        if r.status_code != 200:
            return rec
        j = [x for x in r.json() if x.get("entryId") == f"AF-{acc}-F1"]
        if not j:
            rec["api_status"] = "no_F1"
            return rec
        j = j[0]
        rec.update(api_created=str(j.get("modelCreatedDate", ""))[:10], api_latest=j.get("latestVersion"),
                   api_versions="-".join(map(str, j.get("allVersions") or [])),
                   api_seq_version=str(j.get("sequenceVersionDate", ""))[:10], api_seq=j.get("sequence"),
                   api_seq_len=len(j.get("sequence") or ""), api_reviewed=j.get("isReviewed"),
                   api_ref_proteome=j.get("isReferenceProteome"), api_plddt=j.get("globalMetricValue"))
        return rec
    rec.setdefault("api_status", "failed")
    return rec


def cmd_api(a):
    at = pd.read_csv(ATOMS_CSV, usecols=["acc", "status"], low_memory=False)
    nonlocal_ = at.loc[at.status != "compared", "acc"].tolist()
    local_sample = at.loc[at.status == "compared", "acc"].sample(n=a.check_local, random_state=SEED).tolist()
    done = pd.read_csv(API_CSV) if os.path.isfile(API_CSV) and not a.redo else pd.DataFrame(columns=["acc", "api_status"])
    done = done[done.api_status.astype(str).isin(["200", "404", "no_F1"])]
    todo = [x for x in nonlocal_ + local_sample if x not in set(done.acc)]
    print(f"AFDB API: {len(nonlocal_):,} accessions without a local v6 file + {len(local_sample)} local ones as a check; "
          f"{len(todo):,} to fetch ({len(done):,} cached)", flush=True)
    t0, rows = time.time(), []
    with ThreadPoolExecutor(a.threads) as ex:
        for i, r in enumerate(ex.map(_api_one, todo), 1):
            rows.append(r)
            if i % 2000 == 0:
                print(f"  {i:,} ({time.time() - t0:.0f} s)", flush=True)
                S._save(pd.concat([done, pd.DataFrame(rows)], ignore_index=True), API_CSV)
    df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("acc", keep="last")
    df["probed"] = time.strftime("%Y-%m-%d")
    df["is_local_check"] = df.acc.isin(local_sample)
    S._save(df, API_CSV)
    print(f"wrote {API_CSV} in {time.time() - t0:.0f} s")
    summarise_api(df)


def summarise_api(api=None):
    if api is None:
        api = pd.read_csv(API_CSV, low_memory=False)
    at = pd.read_csv(ATOMS_CSV, low_memory=False)
    api["api_status"] = api.api_status.astype(str)
    chk = api[api.is_local_check.eq(True)].merge(at[["acc", "deposited", "category", "seq_v6"]], on="acc")
    print(f"\n=== AFDB API (probed {api.probed.dropna().iloc[0] if api.probed.notna().any() else '?'}) ===")
    print(f"  check on {len(chk)} accessions with a local v6 file: modelCreatedDate equals the file's header date in "
          f"{int((chk.api_created == chk.deposited).sum())}; API sequence equals the v6 file's in "
          f"{int((chk.api_seq == chk.seq_v6).sum())}")
    nl = api[~api.is_local_check.eq(True)].merge(at[["acc", "seq_v4", "in_analysis", "in_detection", "n_v2c_jobs"]], on="acc", how="left")
    st = nl.api_status.value_counts().to_dict()
    print(f"  {len(nl):,} accessions without a local v6 file: API status {st}")
    ok = nl[nl.api_status == "200"]
    print(f"  served (200) {len(ok):,}: model made {ok.api_created.value_counts().to_dict()}")
    print(f"  no v6 entry (404 or no F1) {int((nl.api_status != '200').sum()):,}: in the analysis set "
          f"{int((nl[nl.api_status != '200'].in_analysis.eq(True)).sum())}, in the detection set "
          f"{int((nl[nl.api_status != '200'].in_detection.eq(True)).sum())}")
    ok = ok.assign(seq_same=ok.api_seq == ok.seq_v4)
    print(f"  served, sequence equals the v4 file's {int(ok.seq_same.sum()):,}; sequence changed {int((~ok.seq_same).sum()):,} "
          f"(of which model dated 2025 {int((~ok.seq_same & ok.api_created.str.startswith('2025')).sum()):,})")
    for d, g in ok.groupby("api_created"):
        print(f"    made {d}: {len(g):,}; sequence changed {int((~g.seq_same).sum())}")


# =============================================================================== scan
_RESI = re.compile(r"resi ([0-9+\-]+)")


def _expand(resi):
    out = []
    for tok in resi.split("+"):
        if not tok:
            continue
        if "-" in tok[1:]:
            lo, hi = tok.split("-", 1)
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    return out


def _scan_one(path):
    job = os.path.basename(path)[:-4]
    rows = []
    try:
        with tarfile.open(path, "r|gz") as tar:
            for m in tar:
                if m.name == "./pocket_selections.csv":
                    txt = tar.extractfile(m).read().decode("utf-8", "replace")
                    for ln in txt.splitlines():
                        if ln.startswith("AF_"):
                            label, expr = ln.split(",", 1)
                            mm = S.MODEL_RE.search(expr)
                            rr = _RESI.search(expr)
                            rows.append({"job_name": job, "af_as": label, "acc": mm.group(1) if mm else None,
                                         "version": int(mm.group(3)) if mm else None,
                                         "resi": "+".join(map(str, sorted(set(_expand(rr.group(1)))))) if rr else ""})
                    break
    except Exception as e:  # noqa: BLE001
        rows.append({"job_name": job, "af_as": f"error:{type(e).__name__}", "acc": None, "version": None, "resi": ""})
    if not rows:
        rows.append({"job_name": job, "af_as": "no_pocket_selections", "acc": None, "version": None, "resi": ""})
    return rows


def _entry_jobs():
    uni = pd.read_csv(S.UNI, usecols=["threshold", "rep_job"], low_memory=False)
    return uni.loc[uni.threshold == "UNCLUSTERED", "rep_job"].tolist()


def _paths(jobs):
    s = pd.read_csv(S.SUMMARY, usecols=["job_name", "query_path"], low_memory=False)
    s = s[s.job_name.isin(set(jobs))]
    return {j: os.path.join(S.ROOT, str(qp).replace("\\", "/").split("/")[-2], f"{j}.tgz") for j, qp in zip(s.job_name, s.query_path)}


def cmd_scan(a):
    jobs = _entry_jobs()
    paths = _paths(jobs)
    todo = [paths[j] for j in jobs if j in paths]
    print(f"analysis entries {len(jobs):,}; archives located {len(todo):,}", flush=True)
    t0, rows = time.time(), []
    with Pool(a.procs) as pool:
        for i, rs in enumerate(pool.imap_unordered(_scan_one, todo, chunksize=64), 1):
            rows += rs
            if i % 50000 == 0:
                print(f"  {i:,} ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["job_name", "af_as"])
    S._save(df, SCAN_CSV)
    print(f"wrote {SCAN_CSV} ({len(df):,} AF selections) in {time.time() - t0:.0f} s")


# =============================================================================== pockets
def _pocket_sets():
    """Distinct (acc, resi) pockets: every analysis entry's AF selection(s), and the ST_100 sites' unions."""
    sc = pd.read_csv(SCAN_CSV, dtype=str).fillna("")
    sc = sc[(sc.acc != "") & (sc.resi != "")]
    ent = sc[["acc", "resi"]].drop_duplicates()
    ent["source"] = "entry"
    # ST_100 site pocket := union of the members' AF selections of the site's accession(s)
    uni = pd.read_csv(S.UNI, usecols=["threshold", "unit_uid", "job_names", "uniprots_key"], low_memory=False)
    st = uni[uni.threshold == "ST_100"]
    by_job = collections.defaultdict(list)
    for r in sc.itertuples():
        by_job[r.job_name].append((r.acc, r.resi))
    rows = []
    for r in st.itertuples():
        acc_res = collections.defaultdict(set)
        for j in str(r.job_names).split(";"):
            for acc, resi in by_job.get(j, []):
                acc_res[acc].update(_expand(resi))
        for acc, res in acc_res.items():
            rows.append({"unit_uid": r.unit_uid, "acc": acc, "resi": "+".join(map(str, sorted(res)))})
    site = pd.DataFrame(rows)
    S._save(site, os.path.join(OUT_DIR, "st100_site_pockets.csv"))
    u = site[["acc", "resi"]].drop_duplicates()
    u["source"] = "site"
    both = pd.concat([ent, u]).drop_duplicates(["acc", "resi"])
    return both, site


def _pockets_acc(job):
    acc, resis = job
    p4, p6 = _store_path(acc, 4), _v6_path(acc)
    if not os.path.isfile(p4) or p6 is None:
        return [{"acc": acc, "resi": r, "status": "no_pair"} for r in resis]
    res4, at4 = _model(p4)
    res6, at6 = _model(p6)
    out = []
    for resi in resis:
        rs = _expand(resi)
        rec = {"acc": acc, "resi": resi, "n_resi": len(rs), "n_found_v4": sum(r in res4 for r in rs),
               "n_found_v6": sum(r in res6 for r in rs), "status": "ok"}
        rec["pocket_seq_same"] = all(res4.get(r) == res6.get(r) for r in rs)
        keys = sorted(k for k in at4 if k[0] in set(rs) and k in at6)
        if len(keys) < 3:
            rec["status"] = "too_few_atoms"
            out.append(rec)
            continue
        X4 = np.array([at4[k][:3] for k in keys])
        X6 = np.array([at6[k][:3] for k in keys])
        B4 = np.array([at4[k][3] for k in keys])
        B6 = np.array([at6[k][3] for k in keys])
        M4 = np.array([at4[k][4] for k in keys])
        ca = [i for i, k in enumerate(keys) if k[1] == "CA"]
        rec["pocket_identical"] = bool(np.array_equal(X4, X6) and np.array_equal(B4, B6)
                                       and set(k for k in at4 if k[0] in set(rs)) == set(k for k in at6 if k[0] in set(rs)))
        rec["pocket_ca_rmsd"] = _kabsch_rmsd(X4[ca], X6[ca]) if len(ca) >= 3 else float("nan")
        rec["pocket_aa_rmsd"] = _kabsch_rmsd(X4, X6)
        rec["pocket_rog_v4"], rec["pocket_rog_v6"] = _rog(X4, M4), _rog(X6, M4)
        rec["pocket_plddt_v4"] = float(B4[ca].mean()) if ca else float("nan")
        rec["pocket_plddt_v6"] = float(B6[ca].mean()) if ca else float("nan")
        rec["pocket_max_dplddt"] = float(np.abs(B4[ca] - B6[ca]).max()) if ca else float("nan")
        out.append(rec)
    return out


def cmd_pockets(a):
    both, site = _pocket_sets()
    at = pd.read_csv(ATOMS_CSV, usecols=["acc", "status", "category"], low_memory=False)
    local = set(at.loc[at.status == "compared", "acc"])
    ident = set(at.loc[at.category == "identical", "acc"])
    todo = both[both.acc.isin(local - ident)]
    skip = both[both.acc.isin(ident)]
    print(f"distinct (model, pocket) pairs {len(both):,}: on a changed local model {len(todo):,}, on an identical model "
          f"{len(skip):,} (trivially unchanged), model not local {int((~both.acc.isin(local)).sum()):,}", flush=True)
    jobs = [(acc, g.resi.tolist()) for acc, g in todo.groupby("acc")]
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = [r for rs in pool.imap_unordered(_pockets_acc, jobs, chunksize=8) for r in rs]
    df = pd.DataFrame(rows)
    triv = skip.assign(status="ok", pocket_identical=True, pocket_seq_same=True, pocket_ca_rmsd=0.0, pocket_aa_rmsd=0.0,
                       pocket_max_dplddt=0.0)[["acc", "resi", "status", "pocket_identical", "pocket_seq_same", "pocket_ca_rmsd",
                                               "pocket_aa_rmsd", "pocket_max_dplddt"]]
    df = pd.concat([df, triv], ignore_index=True)
    S._save(df, POCKETS_CSV)
    print(f"measured {len(rows):,} pockets in {time.time() - t0:.0f} s; wrote {POCKETS_CSV}")
    summarise_pockets(df)


def summarise_pockets(df=None):
    if df is None:
        df = pd.read_csv(POCKETS_CSV, low_memory=False)
    at = pd.read_csv(ATOMS_CSV, usecols=["acc", "category", "deposited"], low_memory=False)
    d = df[df.status == "ok"].merge(at, on="acc", how="left")
    ch = d[d.category != "identical"].copy()
    ch["tier"] = [_tier(r, seq_changed=not s) for r, s in zip(ch.pocket_aa_rmsd, ch.pocket_seq_same.eq(True))]
    ch["tier_ca"] = [_tier(r, seq_changed=not s) for r, s in zip(ch.pocket_ca_rmsd, ch.pocket_seq_same.eq(True))]
    print("\n=== pocket-level movement on changed models (superposition on the pocket's own atoms) ===")
    if os.path.isfile(SCAN_CSV):
        sc = pd.read_csv(SCAN_CSV, dtype=str).fillna("")
        print(f"  AF selections of the analysis entries {len(sc):,} in {sc.job_name.nunique():,} archives")
        sp = pd.read_csv(os.path.join(OUT_DIR, "st100_site_pockets.csv"), dtype=str).fillna("")
        pairs = pd.concat([sc[(sc.acc != "") & (sc.resi != "")][["acc", "resi"]], sp[["acc", "resi"]]]).drop_duplicates()
        pairs = pairs.merge(at, on="acc", how="left")
        n_ident = int((pairs.category == "identical").sum())
        n_local = int(pairs.category.notna().sum())
        print(f"  distinct (model, pocket) pairs {len(pairs):,}: on a changed local model {n_local - n_ident:,}, on an identical model "
              f"{n_ident:,} (trivially unchanged), model not local {len(pairs) - n_local:,}")
    print(f"  distinct (model, pocket) pairs measured on changed local models {len(ch):,} over {ch.acc.nunique():,} models; "
          f"pockets whose residue names changed {int((~ch.pocket_seq_same.eq(True)).sum()):,}; pocket atoms identical "
          f"{int(ch.pocket_identical.eq(True).sum()):,}")
    ok = ch[ch.pocket_seq_same.eq(True)]
    print(f"  pocket all-atom RMSD: median {ok.pocket_aa_rmsd.median():.3f} A, IQR {ok.pocket_aa_rmsd.quantile(.25):.3f}-"
          f"{ok.pocket_aa_rmsd.quantile(.75):.3f}, p90 {ok.pocket_aa_rmsd.quantile(.9):.2f}, max {ok.pocket_aa_rmsd.max():.1f}")
    print(f"  pocket C-alpha RMSD: median {ok.pocket_ca_rmsd.median():.3f} A, IQR {ok.pocket_ca_rmsd.quantile(.25):.3f}-"
          f"{ok.pocket_ca_rmsd.quantile(.75):.3f}, p90 {ok.pocket_ca_rmsd.quantile(.9):.2f}")
    dr = (ok.pocket_rog_v6 - ok.pocket_rog_v4).abs()
    dp = (ok.pocket_plddt_v6 - ok.pocket_plddt_v4)
    print(f"  pocket RoG |v6 - v4|: median {dr.median():.3f} A, p90 {dr.quantile(.9):.3f}; within 0.01 A {int((dr <= 0.01).sum()):,} "
          f"({S._pct((dr <= 0.01).sum(), len(ok)):.1f}%); within 0.05 A {int((dr <= 0.05).sum()):,} ({S._pct((dr <= 0.05).sum(), len(ok)):.1f}%)")
    print(f"  pocket mean pLDDT v6 - v4: median {dp.median():+.2f}, IQR {dp.quantile(.25):+.2f} to {dp.quantile(.75):+.2f}; "
          f"|d| > 2 in {int((dp.abs() > 2).sum()):,} ({S._pct((dp.abs() > 2).sum(), len(ok)):.1f}%), > 5 in {int((dp.abs() > 5).sum()):,}")
    for lab, col in (("all-atom", "tier"), ("C-alpha", "tier_ca")):
        vc = ch[col].value_counts()
        print(f"  tiers by pocket {lab} RMSD: " + ", ".join(f"{t} {int(vc.get(t, 0)):,} ({S._pct(vc.get(t, 0), len(ch)):.1f}%)"
                                                             for t in TIER_ORDER if vc.get(t, 0)))
    _check_union()
    # calibration against the modelv6 prototype: does pocket movement predict a class change?
    m = _calibration(d)
    if m is not None:
        print(f"  calibration on the modelv6 prototype's {len(m):,} classified rows (AF_CENTRIC_RERUN.md), by pocket all-atom RMSD tier:")
        for t in TIER_ORDER:
            g = m[m.tier == t]
            if len(g):
                print(f"    {t:<10} rows {len(g):>4}; entry class changed {int(g.flip.sum()):>3} ({S._pct(g.flip.sum(), len(g)):.1f}%); "
                      f"|dP(holo)| median {g.dP.median():.3f}, p90 {g.dP.quantile(.9):.3f}; RoG changed by > 0.01 in "
                      f"{int(((g.v6_RoG - g.v4_RoG).abs() > 0.011).sum())}, SASA by > 1% in "
                      f"{int(((g.v6_sasa - g.v4_sasa).abs() > 0.01 * g.v4_sasa.abs()).sum())}, volume by > 10% in "
                      f"{int(((g.v6_volume - g.v4_volume).abs() > 0.1 * g.v4_volume.abs().clip(lower=1)).sum())}")


def _calibration(d=None):
    """The modelv6 prototype's classified rows joined to their pocket's movement tier."""
    if not (os.path.isfile(REFRESH_CSV) and os.path.isfile(SCAN_CSV) and os.path.isfile(POCKETS_CSV)):
        return None
    if d is None:
        d = pd.read_csv(POCKETS_CSV, low_memory=False)
        d = d[d.status == "ok"]
    r = pd.read_csv(REFRESH_CSV)
    r = r[(r.status == "ok") & r.cls_v4.notna() & r.cls_v6.notna()]
    sc = pd.read_csv(SCAN_CSV, dtype=str).fillna("")
    sc = sc[sc.job_name.isin(set(r.job_name))].drop_duplicates("job_name")
    m = r.merge(sc[["job_name", "acc", "resi"]], on=["job_name", "acc"], how="inner").merge(
        d[["acc", "resi", "pocket_aa_rmsd", "pocket_seq_same", "pocket_identical"]], on=["acc", "resi"], how="left")
    m = m[m.pocket_aa_rmsd.notna()].copy()
    m["tier"] = [_tier(x, i is True, s is False) for x, i, s in zip(m.pocket_aa_rmsd, m.pocket_identical.eq(True), m.pocket_seq_same.eq(True))]
    m["flip"] = m.cls_v4 != m.cls_v6
    m["dP"] = (m.p_holo_v6 - m.p_holo_v4).abs()
    return m


def _check_union():
    """The site pockets built here (union of the members' AF selections) against the paper's ST_100 union residue sets."""
    p = os.path.join(OUT_DIR, "st100_site_pockets.csv")
    if not (os.path.isfile(p) and os.path.isfile(AFDS)):
        return
    sp = pd.read_csv(p, dtype=str).fillna("")
    a = pd.read_csv(AFDS, usecols=["site_uid", "site_uniprot", "union_residues"], low_memory=False)
    m = a.merge(sp, left_on=["site_uid", "site_uniprot"], right_on=["unit_uid", "acc"], how="left")
    eq = n = 0
    for r in m.itertuples():
        if not isinstance(r.resi, str) or not r.resi:
            continue
        n += 1
        u = {int(x.split("_")[-1]) for x in ast.literal_eval(r.union_residues)}
        eq += u == set(map(int, r.resi.split("+")))
    print(f"  site pockets (union of the members' AF selections) equal the paper's ST_100 union residue set for {eq:,} of "
          f"{len(a):,} detection-table sites ({n:,} matched)")


# =============================================================================== worklist
def _provisional():
    """Per accession without a local v6 file: the API's model date (exact-ish: the same date the file header carries),
    else the UniProt rule of afdb_v4_sources.predict_repredicted."""
    at = pd.read_csv(ATOMS_CSV, low_memory=False)
    lab = None
    try:
        v, _ = S.labelled()
        lab = v[["acc", "pred_repred", "reviewed", "route"]]
    except Exception as e:  # noqa: BLE001
        print(f"UniProt rule unavailable offline ({type(e).__name__}); provisional labels from the API only")
    api = pd.read_csv(API_CSV, low_memory=False) if os.path.isfile(API_CSV) else pd.DataFrame(columns=["acc"])
    api = api[~api.is_local_check.eq(True)] if "is_local_check" in api else api
    p = at[at.status != "compared"][["acc", "seq_v4", "v4_version"]].merge(api, on="acc", how="left")
    if lab is not None:
        p = p.merge(lab, on="acc", how="left")
    p["api_status"] = p.api_status.astype(str)
    served = p.api_status == "200"
    p["v6_missing"] = p.api_status.isin(["404", "no_F1"])
    p["prov_source"] = np.select([p.v6_missing, served, p.get("pred_repred", pd.Series(index=p.index)).notna()],
                                 ["api_absent", "api_date", "uniprot_rule"], "none")
    remade_api = served & p.api_created.astype(str).str.startswith("2025")
    carried_api = served & p.api_created.astype(str).str.startswith("2022")
    seq_changed = served & (p.api_seq != p.seq_v4)
    p["prov_status"] = np.select(
        [p.v6_missing, seq_changed, remade_api, carried_api,
         (p.prov_source == "uniprot_rule") & p.get("pred_repred", pd.Series(False, index=p.index)).eq(True),
         (p.prov_source == "uniprot_rule")],
        ["missing_v6", "changed_prov", "changed_prov", "identical_prov", "changed_prov", "identical_prov"], "unknown")
    p["prov_seq_changed"] = seq_changed
    return p


def _load_tables():
    at = pd.read_csv(ATOMS_CSV, low_memory=False)
    pk = pd.read_csv(POCKETS_CSV, low_memory=False)
    pk = pk[pk.status == "ok"]
    prov = _provisional()
    return at, pk, prov


def _model_status(acc, at_cat, prov_status):
    if acc in at_cat:
        return "identical" if at_cat[acc] == "identical" else "changed"
    return prov_status.get(acc, "unknown")


def cmd_worklist(a):
    at, pk, prov = _load_tables()
    at_cat = dict(zip(at.loc[at.status == "compared", "acc"], at.loc[at.status == "compared", "category"]))
    prov_status = dict(zip(prov.acc, prov.prov_status))
    prov_source = dict(zip(prov.acc, prov.prov_source))
    pk_key = {(r.acc, r.resi): r for r in pk.itertuples(index=False)}

    def pocket_tier(acc, resi):
        r = pk_key.get((acc, resi))
        if r is None:
            return "unknown", np.nan, np.nan
        return _tier(r.pocket_aa_rmsd, r.pocket_identical is True or r.pocket_identical == True,  # noqa: E712
                     r.pocket_seq_same is False or r.pocket_seq_same == False), r.pocket_aa_rmsd, r.pocket_ca_rmsd  # noqa: E712

    # ---- entries
    sc = pd.read_csv(SCAN_CSV, dtype=str).fillna("")
    uni = pd.read_csv(S.UNI, usecols=["threshold", "rep_job", "unit_uid", "is_interface", "pocket_class", "pocket_p_holo",
                                      "uniprots_key"], low_memory=False)
    ent = uni[uni.threshold == "UNCLUSTERED"].rename(columns={"rep_job": "job_name"})
    groups = {j: g for j, g in sc.groupby("job_name")}
    rows = []
    for r in ent.itertuples(index=False):
        g = groups.get(r.job_name)
        sels = [(x.acc, x.resi) for x in g.itertuples()] if g is not None else []
        sels = [s for s in sels if s[0]]
        accs = sorted({s[0] for s in sels})
        stat = [_model_status(acc, at_cat, prov_status) for acc in accs]
        tiers, aa, ca = [], [], []
        for acc, resi in sels:
            if acc in at_cat and at_cat[acc] != "identical" and resi:
                t, x, y = pocket_tier(acc, resi)
                tiers.append(t)
                aa.append(x)
                ca.append(y)
        # the entry's model status: worst over its models
        order = ["missing_v6", "changed", "changed_prov", "unknown", "identical_prov", "identical"]
        ms = min(stat, key=order.index) if stat else "no_model"
        worst = max(tiers, key=TIER_ORDER.index) if tiers else ("identical" if ms == "identical" else "")
        rows.append({"job_name": r.job_name, "query_pdb": r.job_name[:4], "n_models": len(sels), "accs": ";".join(accs),
                     "interface_model": len(sels) > 1, "model_status": ms,
                     "prov_source": ";".join(sorted({prov_source.get(x, "") for x in accs if x not in at_cat})),
                     "pocket_tier": worst, "pocket_aa_rmsd_max": max([x for x in aa if not np.isnan(x)], default=np.nan),
                     "pocket_ca_rmsd_max": max([x for x in ca if not np.isnan(x)], default=np.nan),
                     "pocket_class": r.pocket_class, "pocket_p_holo": r.pocket_p_holo})
    e = pd.DataFrame(rows)
    S._save(e, ENTRY_CSV)
    print(f"wrote {ENTRY_CSV} ({len(e):,} entries)")

    # ---- ST_100 sites
    sp = pd.read_csv(os.path.join(OUT_DIR, "st100_site_pockets.csv"), dtype=str).fillna("")
    st = uni[uni.threshold == "ST_100"]
    lbs = pd.read_csv(S.LBS, usecols=["unit_uid", "success_tns", "site_uniprot"], low_memory=False)
    det = set(lbs.loc[lbs.success_tns.notna(), "unit_uid"])
    det_all = set(lbs.unit_uid)
    sg = {u: g for u, g in sp.groupby("unit_uid")}
    rows = []
    for r in st.itertuples(index=False):
        g = sg.get(r.unit_uid)
        sels = [(x.acc, x.resi) for x in g.itertuples()] if g is not None else []
        accs = sorted({s[0] for s in sels})
        stat = [_model_status(acc, at_cat, prov_status) for acc in accs]
        tiers, aa = [], []
        for acc, resi in sels:
            if acc in at_cat and at_cat[acc] != "identical":
                t, x, _ = pocket_tier(acc, resi)
                tiers.append(t)
                aa.append(x)
        order = ["missing_v6", "changed", "changed_prov", "unknown", "identical_prov", "identical"]
        ms = min(stat, key=order.index) if stat else "no_model"
        worst = max(tiers, key=TIER_ORDER.index) if tiers else ("identical" if ms == "identical" else "")
        rows.append({"unit_uid": r.unit_uid, "uniprots_key": r.uniprots_key, "is_interface": r.is_interface,
                     "n_models": len(accs), "model_status": ms, "pocket_tier": worst,
                     "pocket_aa_rmsd_max": max([x for x in aa if not np.isnan(x)], default=np.nan),
                     "in_detection_table": r.unit_uid in det_all, "detection_site": r.unit_uid in det,
                     "pocket_class": r.pocket_class, "pocket_p_holo": r.pocket_p_holo})
    s_ = pd.DataFrame(rows)
    S._save(s_, SITE_CSV)
    print(f"wrote {SITE_CSV} ({len(s_):,} sites)")
    summarise_worklist(e, s_, at, prov)


def _needs(df, tol_tier):
    """Entries/sites that need a refresh when pockets moved less than the tier's bound are tolerated.
    tol_tier = 'identical': every changed model; 'lt_0.1': tolerate < 0.1 A; ...; provisional and unknown always count."""
    tol_idx = TIER_ORDER.index(tol_tier)
    tolerated = set(TIER_ORDER[1:tol_idx + 1]) if tol_idx > 0 else set()
    changed_local = df.model_status.eq("changed")
    within = changed_local & df.pocket_tier.isin(tolerated)
    return (changed_local & ~within) | df.model_status.isin(["changed_prov", "unknown"])


def summarise_worklist(e=None, s_=None, at=None, prov=None):
    if e is None:
        e = pd.read_csv(ENTRY_CSV, low_memory=False)
        s_ = pd.read_csv(SITE_CSV, low_memory=False)
        at, _, prov = _load_tables()
    summary = []

    def add(table, key, value):
        summary.append({"table": table, "key": key, "value": value})

    print("\n=== worklist: analysis entries (237,922) ===")
    print(f"  entries {len(e):,}; with an AF selection {int((e.n_models > 0).sum()):,}; interface models (several AF chains) "
          f"{int(e.interface_model.sum()):,} ({S._pct(e.interface_model.sum(), len(e)):.1f}%)")
    vc = e.model_status.value_counts()
    labels = {"identical": "model identical (no refresh)", "changed": "model changed (local v6 differs from v4)",
              "missing_v6": "model missing in v6", "identical_prov": "v6 not local; provisionally identical (API date 2022 or UniProt rule)",
              "changed_prov": "v6 not local; provisionally re-made (API date 2025, sequence changed, or UniProt rule)",
              "unknown": "v6 not local; no label", "no_model": "no AF selection in the archive"}
    for k in ["identical", "changed", "missing_v6", "identical_prov", "changed_prov", "unknown", "no_model"]:
        if vc.get(k, 0):
            print(f"    {labels[k]:<90} {int(vc[k]):>8,} ({S._pct(vc[k], len(e)):.1f}%)")
            add("entries_by_model_status", k, int(vc[k]))
    ch = e[e.model_status == "changed"]
    vt = ch.pocket_tier.value_counts()
    print(f"  changed local models, by the entry's pocket movement (worst pocket, all-atom RMSD after pocket superposition):")
    for t in TIER_ORDER:
        if vt.get(t, 0):
            print(f"    {t:<12} {int(vt[t]):>8,} ({S._pct(vt[t], len(ch)):.1f}% of changed; {S._pct(vt[t], len(e)):.1f}% of entries)")
            add("changed_entries_by_pocket_tier", t, int(vt[t]))
    print(f"  provisional labels' source for entries without a local v6 model: {e.loc[e.model_status.isin(['identical_prov', 'changed_prov', 'unknown']), 'prov_source'].value_counts().to_dict()}")
    print("  entries to refresh under each tolerance (provisionally re-made and unlabelled models always counted):")
    tiers_rows = []
    cal = _calibration()
    flip_rate = {t: g.flip.mean() for t, g in cal.groupby("tier")} if cal is not None else {}
    r = pd.read_csv(REFRESH_CSV)
    t = r.t_refresh_s.dropna()
    t_med, t_mean = float(t.median()), float(t.mean())
    cached = set()
    for root in (PILOT_STRUCT, SRC_STRUCT):
        if os.path.isdir(root):
            for d, _, fs in os.walk(root):
                cached |= {f[:4].lower() for f in fs if f.endswith(".cif.gz")}
    sizes = [os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(PILOT_STRUCT) for f in fs if f.endswith(".cif.gz")] \
        if os.path.isdir(PILOT_STRUCT) else []
    mean_size = float(np.mean(sizes)) if sizes else float("nan")
    print(f"  refresh cost per monomer model row (modelv6 prototype, {len(t):,} rows): median {t_med:.2f} s, mean {t_mean:.2f} s; "
          f"query crystals cached locally {len(cached):,} PDB entries (mean {mean_size / 1e6:.2f} MB gzip)")
    for tol in ["identical", "lt_0.1", "0.1_0.5", "0.5_1"]:
        need = _needs(e, tol)
        n = int(need.sum())
        sub = e[need]
        n_if = int(sub.interface_model.sum())
        n_mono = n - n_if
        pdbs = set(sub.query_pdb.str.lower())
        fetch = pdbs - cached
        # interface rows: one US-align + metrics per chain, so about n_models times a monomer row (assumption)
        rows_eq = n_mono + int(sub.loc[sub.interface_model, "n_models"].sum())
        core_h_mean = rows_eq * t_mean / 3600
        core_h_med = rows_eq * t_med / 3600
        lab = {"identical": "any change (tolerance 0)", "lt_0.1": "pocket moved < 0.1 A tolerated", "0.1_0.5": "< 0.5 A tolerated",
               "0.5_1": "< 1 A tolerated"}[tol]
        print(f"    {lab:<34} entries {n:>8,} ({S._pct(n, len(e)):.1f}%): monomer {n_mono:,}, interface {n_if:,}; "
              f"model rows {rows_eq:,}; {core_h_mean:,.0f} core-h at the mean ({core_h_med:,.0f} at the median), "
              f"{core_h_mean / 22:,.1f} h on 22 processes; distinct query PDB entries {len(pdbs):,}, to fetch {len(fetch):,} "
              f"(about {len(fetch) * mean_size / 1e9:,.1f} GB)")
        skipped = e[e.model_status.eq("changed") & ~need]
        miss = sum(len(g) * flip_rate.get(t, 0) for t, g in skipped.groupby("pocket_tier"))
        if len(skipped):
            print(f"      skipped changed entries {len(skipped):,}; at the calibration's per-tier flip rates about {miss:,.0f} of them "
                  f"would change class if refreshed ({S._pct(miss, len(skipped)):.1f}%)")
        tiers_rows.append({"tolerance": tol, "entries": n, "entries_pct": round(S._pct(n, len(e)), 1), "monomer": n_mono,
                           "skipped_changed": len(skipped), "expected_flips_skipped": round(miss),
                           "interface": n_if, "model_rows": rows_eq, "core_h_mean": round(core_h_mean, 1),
                           "core_h_median": round(core_h_med, 1), "hours_22proc": round(core_h_mean / 22, 1),
                           "query_pdb_entries": len(pdbs), "query_pdb_to_fetch": len(fetch),
                           "fetch_gb": round(len(fetch) * mean_size / 1e9, 2)})
    print(f"  distinct query PDB entries over all {len(e):,} entries {e.query_pdb.str.lower().nunique():,}")
    lower = _needs(e, "identical") & ~e.model_status.isin(["changed_prov", "unknown"])
    print(f"  of which already known from local files (changed local model): {int(lower.sum()):,}; the provisional part "
          f"{int(e.model_status.isin(['changed_prov', 'unknown']).sum()):,} entries")
    for row in tiers_rows:
        row["scope"] = "entries"
    # ---- sites
    print("\n=== worklist: ST_100 sites ===")
    for lab, sub in (("all ST_100 sites (unified table)", s_), ("detection sites (86,946)", s_[s_.detection_site]),
                     ("detection-table sites with no v6 model (488)", s_[s_.in_detection_table & ~s_.detection_site])):
        vc = sub.model_status.value_counts()
        print(f"  {lab}: {len(sub):,}; " + ", ".join(f"{k} {int(v):,} ({S._pct(v, len(sub)):.1f}%)" for k, v in vc.items()))
        for k, v in vc.items():
            add(f"sites_by_model_status[{lab.split(' (')[0]}]", k, int(v))
    print(f"  ST_100 sites with a provisional label {int(s_.model_status.isin(['changed_prov', 'identical_prov']).sum()):,}")
    det = s_[s_.detection_site]
    vt = det[det.model_status == "changed"].pocket_tier.value_counts()
    print(f"  detection sites with a changed model, by pocket tier (union pocket): "
          + ", ".join(f"{t} {int(vt.get(t, 0)):,}" for t in TIER_ORDER if vt.get(t, 0)))
    for t in TIER_ORDER:
        if vt.get(t, 0):
            add("changed_detection_sites_by_pocket_tier", t, int(vt[t]))
    for tol in ["identical", "lt_0.1", "0.1_0.5", "0.5_1"]:
        need = _needs(det, tol)
        print(f"    detection sites to refresh, tolerance {tol:<10} {int(need.sum()):>7,} ({S._pct(need.sum(), len(det)):.1f}%)")
        tiers_rows.append({"scope": "detection_sites", "tolerance": tol, "entries": int(need.sum()),
                           "entries_pct": round(S._pct(need.sum(), len(det)), 1)})
    print(f"  interface sites among detection sites {int(det.is_interface.eq(True).sum()):,}")
    S._save(pd.DataFrame(tiers_rows), TIERS_CSV)
    S._save(pd.DataFrame(summary), SUMMARY_CSV)
    # compact per-accession table (local pairs) for the repo
    c = at[at.status == "compared"].copy()
    keep = ["acc", "deposited", "category", "seq_same", "n_res_v4", "n_res_v6", "ca_rmsd_global", "aa_rmsd_global",
            "plddt_v4", "plddt_v6", "in_analysis", "in_detection", "in_detection_sites", "n_v2c_jobs"]
    c = c[keep].copy()
    for k in ("ca_rmsd_global", "aa_rmsd_global", "plddt_v4", "plddt_v6"):
        c[k] = c[k].round(3)
    S._save(c.sort_values("acc"), ACC_CSV)
    print(f"  wrote {SUMMARY_CSV}, {TIERS_CSV}, {ACC_CSV}")


# =============================================================================== reconcile
def summarise_reconcile():
    at = pd.read_csv(ATOMS_CSV, low_memory=False)
    prov = _provisional()
    c = at[at.status == "compared"]
    n_all = len(at)
    print("\n=== reconciling 'about 500 models changed' ===")
    # (a) sequence changed
    seq_local = int(c.seq_same.eq(False).sum())
    seq_prov = int(prov.prov_seq_changed.sum())
    served_prov = int((prov.api_status == "200").sum())
    print(f"  (a) sequence changed, v4 file vs served v6: local pairs {seq_local} of {len(c):,}; accessions without a local file "
          f"{seq_prov} of {served_prov:,} served; total {seq_local + seq_prov} of {len(c) + served_prov:,} with a v6 model "
          f"({S._pct(seq_local + seq_prov, len(c) + served_prov):.2f}%)")
    print(f"      the AFDB changelog's 276,539 changed of 241,070,489 (0.115%) scaled to {n_all:,} accessions gives about "
          f"{n_all * 276539 / 241070489:.0f}")
    # (b) absent from v6
    miss = prov[prov.v6_missing]
    print(f"  (b) no v6 model served: {len(miss)} accessions of {n_all:,}; "
          f"the detection table's sites without a v6 model: 488 on 82 accessions (its own column)")
    sets = pd.read_csv(S.SETS_CSV)
    m2 = miss.merge(sets, on="acc", how="left")
    print(f"      by set: analysis {int(m2.in_analysis.eq(True).sum())}, detection {int(m2.in_detection.eq(True).sum())}, "
          f"released summary {int(m2.in_summary.eq(True).sum())}; v2c jobs on them {int(m2.n_v2c_jobs.fillna(0).sum()):,}")
    # (c) v4 update-shard members (a v3 -> v4 coordinate change; sampled by byte range by afdb_v4_sources.py tars)
    if os.path.isfile(S.TARS_CSV):
        t = pd.read_csv(S.TARS_CSV)
        k, n = int(t.shard.notna().sum()), len(t)
        lo, hi = S._wilson(k, n)
        print(f"  (c) v4 update-shard members (coordinates changed v3 -> v4, not v4 -> v6): {k} of {n} probed accessions "
              f"({S._pct(k, n):.1f}% [{lo:.0f}, {hi:.0f}]) -> about {n_all * k / n:,.0f} [{n_all * lo / 100:,.0f}, {n_all * hi / 100:,.0f}] of {n_all:,}")
    # (d) the Swiss-Prot/proteome gap and (e) obsolete accessions, from labelled()
    if "route" in prov:
        v, _ = S.labelled()
        vc = v.route.value_counts()
        print(f"  (d) re-made but in no EBI v4 tar ('neither'): {int(vc.get('neither', 0)):,} of {n_all:,} (local v6 headers "
              f"{int((v[v.label_source == 'v6_header'].route == 'neither').sum()):,})")
        print(f"  (e) obsolete (no UniProt entry, no local v6): {int(vc.get('obsolete', 0)):,}; of them the API serves a v6 model for "
              f"{int(prov[prov.acc.isin(v.loc[v.route == 'obsolete', 'acc']) & (prov.api_status == '200')].shape[0]):,}")
    # (f) pocket moved beyond a tolerance: accession-level, worst pocket
    pk = pd.read_csv(POCKETS_CSV, low_memory=False)
    pk = pk[pk.status == "ok"].merge(c[["acc", "category"]], on="acc")
    ch = pk[pk.category != "identical"]
    worst = ch.groupby("acc").pocket_aa_rmsd.max()
    seqch = ch.groupby("acc").pocket_seq_same.apply(lambda s: (~s.eq(True)).any())
    print(f"  (f) local changed models by their worst pocket: pocket all-atom RMSD >= 1 A {int((worst >= 1).sum()):,}, >= 0.5 A "
          f"{int((worst >= 0.5).sum()):,}, >= 0.1 A {int((worst >= 0.1).sum()):,}, any change {int((worst > 0).sum()):,} of {len(worst):,}; "
          f"pocket residue names changed {int(seqch.sum())}")
    # (g) class flips (the modelv6 prototype)
    r = pd.read_csv(REFRESH_CSV)
    r = r[(r.status == "ok") & r.in_random & r.cls_v4.notna() & r.cls_v6.notna()]
    rate = (r.cls_v4 != r.cls_v6).mean()
    print(f"  (g) entry class flips under the refresh: {int((r.cls_v4 != r.cls_v6).sum())} of {len(r)} random entries ({100 * rate:.1f}%) "
          f"-> about {237922 * rate:,.0f} of 237,922 entries; pilot sites 9 of 100 -> about {86946 * 0.09:,.0f} of 86,946")
    # (h) earlier round numbers a colleague may have heard
    n_loc_ch = int((c.category != "identical").sum())
    n_prov_ch = int((prov.prov_status == "changed_prov").sum())
    n_prov_id = int((prov.prov_status == "identical_prov").sum())
    print(f"  (i) changed models over the whole v2c set: exact (local pairs) {n_loc_ch:,} of {len(c):,}; provisional (API date 2025 "
          f"or sequence changed) {n_prov_ch:,} of {n_prov_ch + n_prov_id:,} served without a local file; together "
          f"{n_loc_ch + n_prov_ch:,} of {len(c) + n_prov_ch + n_prov_id:,} accessions with a v6 model "
          f"({S._pct(n_loc_ch + n_prov_ch, len(c) + n_prov_ch + n_prov_id):.1f}%); unchanged {int((c.category == 'identical').sum()) + n_prov_id:,}")
    print(f"  (h) other numbers in circulation: 'about 400 uncovered' (27,615 - 27,212 estimated anonymous+AlphaFill coverage, "
          f"AFDB_V4_SOURCES.md); 488 detection sites on 82 accessions without a v6 model (AF_CENTRIC_RERUN.md)")


# =============================================================================== report
def cmd_report(a):
    print(f"afdb_v4_v6_diff.py report, {time.strftime('%Y-%m-%d')} (caches only; no network)")
    summarise_atoms()
    if os.path.isfile(API_CSV):
        summarise_api()
    summarise_pockets()
    summarise_worklist()
    summarise_reconcile()
    # the accessions without a local v6 file, should the author want them fetched
    at = pd.read_csv(ATOMS_CSV, low_memory=False)
    prov = _provisional()
    nl = prov[prov.api_status == "200"].merge(at[["acc", "n_atoms_v4", "in_analysis", "in_detection"]], on="acc", how="left")
    v6sz = [os.path.getsize(os.path.join(S.AFDB_V6, f)) for f in os.listdir(S.AFDB_V6)[:2000] if f.endswith(".cif.gz")]
    mean_v6 = float(np.mean(v6sz))
    print(f"\n=== not fetched: the v6 files that are not local ===")
    print(f"  served v6 models without a local file {len(nl):,} (analysis set {int(nl.in_analysis.eq(True).sum()):,}); at the local mean of "
          f"{mean_v6 / 1e3:.0f} kB gzip about {len(nl) * mean_v6 / 1e9:.2f} GB; of them provisionally re-made "
          f"{int((nl.prov_status == 'changed_prov').sum()):,}, provisionally carried over {int((nl.prov_status == 'identical_prov').sum()):,}")
    print("  Fetching them is the author's call (AFDB_V4_SOURCES.md header rule); nothing was downloaded here.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("atoms")
    q.add_argument("--procs", type=int, default=12)
    q = sub.add_parser("api")
    q.add_argument("--threads", type=int, default=6)
    q.add_argument("--check-local", type=int, default=300)
    q.add_argument("--redo", action="store_true")
    q = sub.add_parser("scan")
    q.add_argument("--procs", type=int, default=10)
    q = sub.add_parser("pockets")
    q.add_argument("--procs", type=int, default=12)
    sub.add_parser("worklist")
    sub.add_parser("report")
    a = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    {"atoms": cmd_atoms, "api": cmd_api, "scan": cmd_scan, "pockets": cmd_pockets, "worklist": cmd_worklist,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
