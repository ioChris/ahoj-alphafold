# -*- coding: utf-8 -*-
"""Where can the AlphaFold DB v4 model files that AHoJ-DB v2c used still be obtained? And the v2c model store.

Every arm of the manuscript ran on the AFDB version 4 models that AHoJ-DB v2c used (Methods, Study design;
Data availability; Supplementary Note 4), while AFDB now serves version 6 per accession. This script finds where
the version 4 files can still be obtained (the Google Cloud public dataset gs://public-datasets-deepmind-alphafold-v4
and the EBI FTP v4 archives, as the Data availability statement says), which version 6 models are carried over
unchanged and which were re-predicted (Swiss-Prot and reference-proteome entries; Methods and Supplementary
Note 4), and builds and verifies the local model store whose files and manifest of accession, model version and
checksum (afdb_v2c_store_manifest.csv, written by `store verify`) are deposited with the data. Its paths,
accession sets, v6 headers, archive scan and atom reader are also imported by afdb_v4_v6_diff.py and
afdb_v4_realign.py. Every number of the analysis is printed by `report` (or, for the one-off scans, by the
subcommand that made the cache).

    python afdb_v4_sources.py v2c_scan [--procs 8]    -> cache CSV: the AF model(s) named in every v2c job archive
    python afdb_v4_sources.py sets                    -> cache CSV: accession sets (v2c, released summary, analysis, detection)
    python afdb_v4_sources.py uniprot                 -> cache CSV: current UniProt facts per accession (REST, batched)
    python afdb_v4_sources.py v6hdr [--procs 8]       -> cache CSV: the headers of the local v6 files (model date, first release)
    python afdb_v4_sources.py alphafill               -> cache CSV: the local AlphaFill databank's filled_v4 models
    python afdb_v4_sources.py gcs [--threads 16]      -> cache CSV: the v4 Google Cloud bucket listed per accession
    python afdb_v4_sources.py tars [--n 160]          -> cache CSV: EBI FTP v4 tar membership by byte-range search (sample)
    python afdb_v4_sources.py validate [--n 40]       -> cache CSV: FTP v4 files vs local v4 files, archived US-align frames
                                                         and the v2c model-row metrics
    python afdb_v4_sources.py store list              -> cache CSV: every bucket object of the v2c model store, as listed
    python afdb_v4_sources.py store fetch             -> <afdb_store_dir>/v4 and /v3: the objects copied with the user's own
                                                         gcloud login (stored gzip bytes, checked against the listing)
    python afdb_v4_sources.py store verify            -> the store checked; its manifest written beside it
    python afdb_v4_sources.py validate --arm store --n 200   -> the store against v2c jobs (archived frames, model rows)
    python afdb_v4_sources.py mirrors                 -> third-party copies (Foldcomp, AlphaFill, PLINDER, ...) probed
    python afdb_v4_sources.py report > out.txt        -> every quoted number; the HTTP probes are re-run live and dated
                                                         (--offline skips the dated live-probe block)

WHAT COUNTS AS A PROBE. Listings, HEAD requests, API metadata, and HTTP byte ranges only. Single tar
members are pulled out of the EBI tars by binary search over byte offsets (the tars are sorted by member
name and uncompressed), so no tar is downloaded. The EBI server honours at most 200 ranges per request and
answers 201 or more with the WHOLE file (a 37 GB 200 response), so `_ranges` never asks for more than 200
and every ranged GET is streamed and abandoned unless the status is 206.

THE ONE BULK COPY. `store fetch` copies the v2c models (with their confidence and PAE JSON, 2026-09-17) from
the Google Cloud buckets through the user's own `gcloud` login, set up beforehand; this script never creates or
configures an account.

Roots: the regenerable cache CSVs go to <manuscript_analyses_dir>/afdb_v4_sources/, the committed choices
(afdb_v4_tar_sample.csv, afdb_v4_validation_jobs.csv, afdb_v4_store_validation_jobs.csv) to <manuscript_analyses_dir>;
extracted model files and query crystals go to <afdb_v4_sources_dir>; the model store is <afdb_store_dir> (v4/, v3/
and the v6/ files AFDB serves today); <webserver_cache_dir> holds the AHoJ webserver's AF/ and structures/ caches and
<af_centric_pilot_dir> the AF-centric re-run pilot (its workdir/ query crystals and US-align build). The v2c job
archives are cfg.AHOJ_OUTPUT_FIX_DIR.
"""
import argparse
import base64
import collections
import gzip
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd

import ma_paths as P
import ahoj_job  # job archives under the names before AHoJ 2.6.0

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = str(P.WD)
DB = str(P.cfg.AHOJ_DB_DIR)
SUMMARY = str(P.cfg.CLASSIFICATION_SUMMARIES_CSV)
LBS = os.path.join(WORK, "lbs_generality", "p2rank", "lbs_p2rank_st100_merged_per_pocket.csv")
UNI = str(P.cfg.ANALYSIS_UNIFIED_LEVELS_CSV)
ROOT = str(P.cfg.AHOJ_OUTPUT_FIX_DIR)                                       # 518,883 per-job .tgz
WEBSERVER = str(P.anchor("webserver_cache_dir", "AHOJ_WEBSERVER_CACHE_DIR"))  # the AHoJ webserver's AF/ and structures/
V4_LOCAL = os.path.join(WEBSERVER, "AF")                                    # the 14 local v4 models
AFDB_V6 = os.path.join(str(P.anchor("afdb_store_dir", "AHOJ_AFDB_STORE_DIR")), "v6")
CACHE = str(P.anchor("afdb_v4_sources_dir", "AHOJ_AFDB_V4_SOURCES_DIR"))     # extracted v4 files, query crystals
PILOT = str(P.anchor("af_centric_pilot_dir", "AHOJ_AF_CENTRIC_PILOT_DIR"))    # the AF-centric re-run pilot
PILOT_WD = os.path.join(PILOT, "workdir")                                   # read-only: cached query crystals
USALIGN = os.path.join(PILOT, "code_5c3a191", "alignment_executables", "USalign_20240319.exe")
PY311 = os.environ.get("AHOJ_AF_PY311", sys.executable)                     # the python with PyMOL, PyVOL (and msms)

OUT_DIR = os.path.join(str(P.OUT), "afdb_v4_sources")
SCAN_CSV = os.path.join(OUT_DIR, "v2c_model_rows.csv")
SETS_CSV = os.path.join(OUT_DIR, "accession_sets.csv")
UNIPROT_CSV = os.path.join(OUT_DIR, "uniprot_facts.csv")
GCS_CSV = os.path.join(OUT_DIR, "gcs_v4_listing.csv")
TARS_CSV = os.path.join(OUT_DIR, "ftp_tar_membership.csv")
SHARDS_CSV = os.path.join(OUT_DIR, "ftp_updated_shard_index.csv")
VALID_CSV = os.path.join(OUT_DIR, "validation.csv")
SAMPLE_CSV = P.out("afdb_v4_tar_sample.csv")                               # committed: the sampled accessions
VALID_JOBS_CSV = P.out("afdb_v4_validation_jobs.csv")                      # committed: the validated v2c jobs

FTP = "https://ftp.ebi.ac.uk/pub/databases/alphafold/"
BUCKET = "public-datasets-deepmind-alphafold-v4"
GCS_API = "https://storage.googleapis.com/storage/v1/b/"
SEED = 20260916
MODEL_RE = re.compile(r"AF-([A-Z0-9]+(?:-\d+)?)-F(\d+)-model_v(\d+)")


def _save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _pct(k, n):
    return 100.0 * k / n if n else float("nan")


def _wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# =============================================================================== v2c_scan
def _scan_one(path):
    """AF model(s) of one v2c job: the AF_ lines of pocket_selections.csv, and the archived AF matrices."""
    rec = {"job_name": os.path.basename(path)[:-4], "batch": os.path.basename(os.path.dirname(path)),
           "status": "ok", "n_af_sel": 0, "af_models": "", "af_as": "", "n_af_mat": 0, "af_mat_models": ""}
    try:
        mats, sel = [], None
        with tarfile.open(path, "r|gz") as tar:
            for m in tar:
                if m.name.startswith("./matrices/aln_AF-"):
                    mats.append(m.name)
                elif m.name == "./pocket_selections.csv":        # written after every matrix; near the end
                    sel = tar.extractfile(m).read().decode("utf-8", "replace")
                    break
        if sel is None:
            # a job that stopped before writing its pocket tables still archived the model's US-align
            # matrix (aln_AF-<acc>-F1-model-v4_to_<pdb><chain>.txt), and the released summary counts its model row
            rec["status"] = "no_pocket_selections"
        lines = [ln for ln in (sel or "").splitlines() if ln.startswith("AF_")]
        mat_models = [re.sub(r"-model-v(\d+)_to_.*$", r"-model_v\1", m.split("/aln_", 1)[1]) for m in mats]
        rec.update(n_af_sel=len(lines), n_af_mat=len(mats), af_mat_models=";".join(mat_models),
                   af_models=";".join(ln.split(",", 1)[1].split(" and ")[0].strip() for ln in lines),
                   af_as=";".join(ln.split(",", 1)[0] for ln in lines))
    except Exception as e:  # noqa: BLE001
        rec["status"] = f"error:{type(e).__name__}"
    return rec


def cmd_v2c_scan(a):
    paths = []
    for batch in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, batch)
        if os.path.isdir(d):
            paths += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".tgz")]
    print(f"v2c archives: {len(paths):,} in {len(os.listdir(ROOT))} batch folders", flush=True)
    t0, rows = time.time(), []
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_scan_one, paths, chunksize=64), 1):
            rows.append(r)
            if i % 50000 == 0:
                print(f"  {i:,} ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows).sort_values("job_name")
    _save(df, SCAN_CSV)
    print(f"wrote {SCAN_CSV} in {time.time() - t0:.0f} s")
    summarise_scan(df)


def _accs(models):
    return [m.group(1) for m in MODEL_RE.finditer(str(models))]


def job_models(df):
    """The AF model file names of each job: pocket_selections.csv, else the archived AF matrices."""
    sel = df.af_models.fillna("").astype(str)
    mat = df.af_mat_models.fillna("").astype(str)
    return sel.where(df.n_af_sel > 0, mat)


def summarise_scan(df):
    print("\n=== v2c job archives: the AlphaFold model rows ===")
    print(f"  archives scanned {len(df):,}; status {df.status.str.split(':').str[0].value_counts().to_dict()}")
    print(f"  AF selections per archive: {df.n_af_sel.value_counts().sort_index().to_dict()}")
    df = df.assign(models=job_models(df))
    sel = df[df.n_af_sel > 0]
    print(f"  AF selections equal AF matrices in {int((sel.n_af_sel == sel.n_af_mat).sum()):,} of {len(sel):,}")
    nosel = df[(df.n_af_sel == 0) & (df.n_af_mat > 0)]
    print(f"  archives without pocket_selections.csv whose model is named only by its matrix: {len(nosel):,}")
    has = df[df.models != ""]
    vers = collections.Counter(m.group(3) for s in has.models for m in MODEL_RE.finditer(s))
    frags = collections.Counter(m.group(2) for s in has.models for m in MODEL_RE.finditer(s))
    print(f"  archives with an AF model {len(has):,}; model files named: version {dict(vers)}, fragment {dict(frags)}")
    v3 = has[has.models.str.contains("model_v3")]
    print(f"  archives naming a v3 model: {len(v3)} ({len({a for s in v3.models for a in _accs(s)})} accessions: "
          f"{', '.join(sorted({a for s in v3.models for a in _accs(s)}))})")
    accs = collections.Counter(x for s in has.models for x in set(_accs(s)))
    multi = has[has.models.str.count(";") > 0]
    hetero = multi[multi.models.map(lambda s: len(set(_accs(s))) > 1)]
    print(f"  distinct model accessions {len(accs):,}; interface jobs (several AF chains) {len(multi):,}, "
          f"of which with several accessions {len(hetero):,}")


# =============================================================================== sets
def cmd_sets(a):
    scan = pd.read_csv(SCAN_CSV, low_memory=False)
    scan["models"] = job_models(scan)
    scan = scan[scan.models != ""].copy()
    scan["accs"] = scan.models.map(lambda s: sorted(set(_accs(s))))
    job_accs = dict(zip(scan.job_name, scan.accs))

    s = pd.read_csv(SUMMARY, usecols=["job_name", "has_model", "UNPs"], low_memory=False)
    sm = s[s.has_model == 1]
    miss = sm[~sm.job_name.isin(job_accs)]
    print("=== accession sets ===")
    print(f"  released summary: {len(s):,} jobs, {len(sm):,} with a model row; {len(miss):,} of those with no "
          f"AF model named in the archive")
    # the summary's UNPs column against the archive: the model accessions are a subset of the query UNPs
    unp = sm[sm.job_name.isin(job_accs)]
    sub = sum(set(job_accs[j]) <= set(str(u).split("-")) | set(re.findall(r"[A-Z0-9]{6,10}", str(u)))
              for j, u in zip(unp.job_name, unp.UNPs))
    print(f"  archive model accessions contained in the summary's UNPs column: {sub:,} of {len(unp):,} jobs")

    uni = pd.read_csv(UNI, usecols=["threshold", "job_names", "rep_job"], low_memory=False)
    ent = uni[uni.threshold == "UNCLUSTERED"]
    ent_jobs = {j for js in ent.job_names.astype(str) for j in js.split(";")}
    lbs = pd.read_csv(LBS, usecols=["site_uniprot", "success_tns"], low_memory=False)
    det_all = set(lbs.site_uniprot)
    det_sites = set(lbs.loc[lbs.success_tns.notna(), "site_uniprot"])

    count = collections.Counter(x for xs in scan.accs for x in xs)
    in_sum = {x for j in sm.job_name if j in job_accs for x in job_accs[j]}
    in_ent = {x for j in ent_jobs if j in job_accs for x in job_accs[j]}
    universe = sorted(set(count) | det_all)
    df = pd.DataFrame({"acc": universe})
    df["n_v2c_jobs"] = df.acc.map(count).fillna(0).astype(int)
    df["in_v2c"] = df.n_v2c_jobs > 0
    df["in_summary"] = df.acc.isin(in_sum)
    df["in_analysis"] = df.acc.isin(in_ent)
    df["in_detection"] = df.acc.isin(det_all)
    df["in_detection_sites"] = df.acc.isin(det_sites)
    _save(df, SETS_CSV)
    summarise_sets(df, len(ent), len(ent_jobs))


SETS = [("v2c", "in_v2c", "every AF model named in a v2c job archive"),
        ("summary", "in_summary", "model rows of the released full_run_3 summary"),
        ("analysis", "in_analysis", "model rows of the 237,922 analysis entries (run5 UNCLUSTERED)"),
        ("detection", "in_detection", "the ST_100 detection table (all sites)"),
        ("detection_sites", "in_detection_sites", "detection sites with a v6 model")]


def summarise_sets(df, n_ent=None, n_ent_jobs=None):
    if n_ent is not None:
        print(f"  analysis entries (run5 UNCLUSTERED): {n_ent:,} rows, {n_ent_jobs:,} member jobs")
    for name, col, what in SETS:
        print(f"  set {name:<16} {int(df[col].sum()):>7,} accessions  ({what})")
    print(f"  detection accessions with no AF model in any v2c archive: "
          f"{int((df.in_detection & ~df.in_v2c).sum())}")


# =============================================================================== uniprot
# Current UniProt facts, used only to PREDICT membership of the EBI tars (whose member lists are not
# published) and to stratify the sample that measures it. The AFDB FAQ gives the v4-era rules: Swiss-Prot
# from UniProt 2021_04, the proteomes from the 2021_02/2021_04 "one sequence per gene" reference proteomes,
# 16-2,700 residues (1,280 for the rest of UniProt), no non-standard residues. For a reviewed entry,
# UniProt's "date of creation" is the date it entered Swiss-Prot (A0A009IHW8: TrEMBL 2014, Swiss-Prot
# 2020-02-26, created 2020-02-26).
UNIPROT_FIELDS = ("accession,reviewed,organism_id,length,date_created,date_sequence_modified,"
                  "sequence_version,xref_proteomes,sequence")
SP_2021_04 = "2021-09-29"          # UniProt release 2021_04: the Swiss-Prot of the EBI v4 tar
SP_2025_03 = "2025-06-18"          # UniProt release 2025_03: the AFDB 2025 synchronisation
NONSTD = re.compile(r"[XBZUOJ]")


def cmd_uniprot(a):
    import requests
    accs = pd.read_csv(SETS_CSV).acc.tolist()
    s = requests.Session()
    rows, n_chunk = [], 400
    for i in range(0, len(accs), n_chunk):
        chunk = accs[i:i + n_chunk]
        for attempt in range(5):
            r = s.get("https://rest.uniprot.org/uniprotkb/accessions",
                      params={"accessions": ",".join(chunk), "fields": UNIPROT_FIELDS, "format": "tsv"}, timeout=300)
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        release = r.headers.get("x-uniprot-release")
        t = pd.read_csv(io.StringIO(r.text), sep="\t", dtype=str)
        rows.append(t)
        if (i // n_chunk) % 20 == 0:
            print(f"  {i + len(chunk):,} / {len(accs):,}", flush=True)
    u = pd.concat(rows, ignore_index=True)
    u = u.rename(columns={"Entry": "acc", "Reviewed": "reviewed", "Organism (ID)": "taxid", "Length": "length",
                          "Date of creation": "created", "Date of last sequence modification": "seq_modified",
                          "Sequence version": "seq_version", "Proteomes": "proteomes", "Sequence": "sequence"})
    u["nonstd"] = u.sequence.fillna("").str.contains(NONSTD)
    u = u.drop(columns=["sequence"]).drop_duplicates("acc")
    u["uniprot_release"] = release
    _save(u, UNIPROT_CSV)
    print(f"UniProt {release}: {len(u):,} of {len(accs):,} accessions active; wrote {UNIPROT_CSV}")


def _ftp_v4_proteomes():
    """taxid -> v4 proteome tar name, from the live FTP listing."""
    import requests
    html = requests.get(FTP + "v4/", timeout=60).text
    names = sorted(set(re.findall(r'href="(UP\d+_\d+_[A-Z0-9]+_v4\.tar)"', html)))
    return {n.split("_")[1]: n for n in names}


def _v6_proteome_ups():
    import requests
    md = requests.get(FTP + "download_metadata.json", timeout=60).json()
    return {x["reference_proteome"] for x in md if x.get("reference_proteome")}


def predict(u, prot):
    """Predicted EBI-tar membership from today's UniProt facts (validated on the sample).

    Swiss-Prot tar: reviewed by UniProt 2021_04 (for a reviewed entry `created` is its Swiss-Prot date).
    Proteome tars: an entry of one of the 47 v4 reference proteomes that already existed in 2021_04."""
    rev = u.reviewed.eq("reviewed")
    created = u.created.fillna("9999")
    L = pd.to_numeric(u.length, errors="coerce")
    unchanged = u.seq_modified.fillna("9999") <= SP_2021_04          # length/residues as in 2021_04
    nonstd = u.nonstd.astype(str).str.lower().eq("true")               # read back from CSV as text
    ok_len = L.between(16, 2700) & ~nonstd
    sp = rev & (created <= SP_2021_04) & (ok_len | ~unchanged)
    up = u.proteomes.fillna("").str.extract(r"(UP\d+)")[0]
    prot_ups = {n.split("_")[0] for n in prot.values()}
    pr = up.isin(prot_ups) & (created <= SP_2021_04) & (ok_len | ~unchanged)
    return sp, pr


def predict_repredicted(u):
    """Predicted 2025 re-prediction (v6 != v4): in Swiss-Prot or in an AFDB v6 proteome at UniProt 2025_03."""
    rev25 = u.reviewed.eq("reviewed") & (u.created.fillna("9999") <= SP_2025_03)
    up = u.proteomes.fillna("").str.extract(r"(UP\d+)")[0]
    return rev25 | up.isin(_v6_proteome_ups())


# =============================================================================== v6hdr
# A v6 mmCIF records its own history: `_pdbx_database_status.recvd_initial_deposition_date` is the date the
# served model was made (2022-06-01 for a model carried over from the July-2022 UniProt release, 2025-08-01
# for a 2025 re-prediction), and the first `_pdbx_audit_revision_history` row is the release in which the
# entry first appeared (1 = July 2021 model organisms, 2 = Dec 2021 Swiss-Prot / Jan 2022 global health,
# 3 = July 2022 rest of UniProt). The EBI v4 tars hold exactly the v1/v2-era sets.
V6HDR_CSV = os.path.join(OUT_DIR, "local_v6_headers.csv")


def _v6hdr_one(path):
    rec = {"file": os.path.basename(path), "deposited": None, "first_rev": None, "first_rev_date": None, "n_rev": 0}
    state = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("_pdbx_database_status.recvd_initial_deposition_date"):
                rec["deposited"] = ln.split()[-1]
            elif ln.startswith("_pdbx_audit_revision_history.revision_date"):
                state = "rev"
                continue
            elif state == "rev":
                if ln.startswith('"Structure model"'):
                    f = ln.split()
                    rec["n_rev"] += 1
                    if rec["first_rev"] is None:
                        rec["first_rev"], rec["first_rev_date"] = int(f[-2]), f[-1]
                    continue
                state = None
            if rec["deposited"] and rec["first_rev"] is not None and state is None:
                break
            if ln.startswith("_atom_site."):
                break
    m = MODEL_RE.match(rec["file"])
    rec["acc"], rec["frag"] = (m.group(1), int(m.group(2))) if m else (None, None)
    return rec


def cmd_v6hdr(a):
    paths = [os.path.join(AFDB_V6, f) for f in os.listdir(AFDB_V6) if f.endswith(".cif.gz")]
    with Pool(a.procs) as pool:
        df = pd.DataFrame(pool.map(_v6hdr_one, paths, chunksize=64))
    _save(df, V6HDR_CSV)
    print(f"read {len(df):,} local v6 headers; wrote {V6HDR_CSV}")


# =============================================================================== alphafill
# The author's local AlphaFill 2.1.1 databank (alphafill.eu bulk tar, extracted): each filled model carries
# the AFDB v4 protein unchanged (its JSON names the source file, /DATA/afdb-v4/<xx>/AF-<acc>-F1-model_v4.cif.gz)
# and appends the transplanted ligands as further entities.
ALPHAFILL = str(P.anchor("alphafill_databank_dir", "AHOJ_ALPHAFILL_DATABANK_DIR"))
AFILL_CSV = os.path.join(OUT_DIR, "alphafill_local.csv")


def afill_path(acc, ext=".cif.gz"):
    return os.path.join(ALPHAFILL, acc[:2], f"AF-{acc}-F1-filled_v4{ext}")


def _afill_one(acc):
    rec = {"acc": acc, "afill": os.path.isfile(afill_path(acc))}
    js = afill_path(acc, ".cif.json")
    if os.path.isfile(js):
        with open(js, "rb") as fh:
            head = fh.read(400).decode("utf-8", "replace")
        for k in ("alphafill_version", "date", "file"):
            m = re.search(rf'"{k}":"([^"]*)"', head)
            rec[f"afill_{k}"] = m.group(1).replace("\\/", "/") if m else None
    return rec


def cmd_alphafill(a):
    accs = pd.read_csv(SETS_CSV).acc.tolist()
    with ThreadPoolExecutor(a.threads) as ex:
        df = pd.DataFrame(list(ex.map(_afill_one, accs)))
    _save(df, AFILL_CSV)
    print(f"checked {len(df):,} accessions against {ALPHAFILL}: {int(df.afill.sum()):,} filled_v4 models; "
          f"wrote {AFILL_CSV}")


def protein_atoms(path):
    """(chain, resnum, resname, atom) -> (x, y, z, B) for the polymer atoms of a model file."""
    import gemmi
    st = gemmi.read_structure(path)
    st.remove_ligands_and_waters()
    return {(c.name, r.seqid.num, r.name, t.name): (t.pos.x, t.pos.y, t.pos.z, t.b_iso)
            for c in st[0] for r in c for t in r}


# =============================================================================== gcs
def _gcs_list(session, prefix, fields="items(name,size,md5Hash,updated),nextPageToken"):
    items, token = [], None
    while True:
        params = {"prefix": prefix, "fields": fields, "maxResults": 1000}
        if token:
            params["pageToken"] = token
        for attempt in range(6):
            try:
                r = session.get(GCS_API + BUCKET + "/o", params=params, timeout=60)
                if r.status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2 ** attempt)
        j = r.json()
        items += j.get("items", [])
        token = j.get("nextPageToken")
        if not token:
            return items


def _gcs_one(acc):
    import requests
    global _GCS_SESSION
    if "_GCS_SESSION" not in globals():
        _GCS_SESSION = requests.Session()
    items = _gcs_list(_GCS_SESSION, f"AF-{acc}-F")
    models = {m.group(2): i for i in items if (m := MODEL_RE.match(i["name"])) and m.group(1) == acc
              and i["name"].endswith(".cif")}
    f1 = models.get("1", {})
    return {"acc": acc, "n_frag": len(models), "f1_size": f1.get("size"), "f1_md5": f1.get("md5Hash"),
            "f1_updated": f1.get("updated"),
            "n_conf": sum(i["name"].endswith("confidence_v4.json") for i in items),
            "n_pae": sum(i["name"].endswith("predicted_aligned_error_v4.json") for i in items)}


def cmd_gcs(a):
    accs = pd.read_csv(SETS_CSV).acc.tolist()
    t0, rows = time.time(), []
    with ThreadPoolExecutor(a.threads) as ex:
        for i, r in enumerate(ex.map(_gcs_one, accs), 1):
            rows.append(r)
            if i % 10000 == 0:
                print(f"  {i:,} ({time.time() - t0:.0f} s)", flush=True)
    _save(pd.DataFrame(rows), GCS_CSV)
    print(f"listed {len(rows):,} accessions in {time.time() - t0:.0f} s; wrote {GCS_CSV}")


# =============================================================================== store
# The v2c model store, copied from Google Cloud with the author's gcloud login. The objects are stored
# gzip-encoded (Content-Encoding: gzip) and the listing's md5Hash is the md5 of the stored bytes, so they are
# copied as stored (--do-not-decompress; the default decompresses), checked against the listing and saved as
# AF-<acc>-F1-<kind>_v<N>.<ext>.gz: for the model, the name AHoJ2's cache (<work_dir>/AF/) and the webapp's
# cache (<db_workdir>/alphafold/<last two characters>/) expect; for the JSON, the EBI tars' member names.
# The three v3 models v2c named come from the "previous versions" bucket into a v3\ folder. Bucket copies of
# the local v4 files that v2c does not name go to a check folder, never into the store.
STORE = str(P.anchor("afdb_store_dir", "AHOJ_AFDB_STORE_DIR"))             # v4/ and v3/ beside v6/
STORE_MANIFEST = os.path.join(STORE, "afdb_v2c_store_manifest.csv")
STORE_CSV = os.path.join(OUT_DIR, "store_objects.csv")             # every object to copy, as listed
STORE_VERIFY_CSV = os.path.join(OUT_DIR, "store_verify.csv")       # every object, as found on disk
STORE_CHECK_CSV = os.path.join(OUT_DIR, "store_checks.csv")        # content comparisons with other copies
STORE_FETCH_JSON = os.path.join(OUT_DIR, "store_fetch_stats.json")
FETCH_DIR = os.path.join(CACHE, "gcs_fetch")                        # staging, gcloud manifests and log
GCS_CHECK = os.path.join(CACHE, "gcs_check")
GCLOUD = os.environ.get("AHOJ_AF_GCLOUD") or shutil.which("gcloud") or "gcloud"
BUCKETS = {4: BUCKET, 3: "public-datasets-deepmind-alphafold"}
KINDS = {"model": "model_v{v}.cif", "confidence": "confidence_v{v}.json", "pae": "predicted_aligned_error_v{v}.json"}


def _b64md5(data):
    return base64.b64encode(hashlib.md5(data).digest()).decode()


def _store_path(r):
    return os.path.join(STORE if r.purpose == "store" else GCS_CHECK, r.local)


def _read_store_csv():
    obj = pd.read_csv(STORE_CSV, dtype={"generation": str, "md5": str, "crc32c": str})
    obj["path"] = [_store_path(r) for r in obj.itertuples()]
    return obj


def _store_pairs():
    """(acc, fragment, version) of every AF model the v2c archives name -> number of archives."""
    scan = pd.read_csv(SCAN_CSV, usecols=["n_af_sel", "af_models", "af_mat_models"], low_memory=False)
    n = collections.Counter()
    for s in job_models(scan):
        for key in {(m.group(1), int(m.group(2)), int(m.group(3))) for m in MODEL_RE.finditer(s)}:
            n[key] += 1
    return n


def _gcloud(args, log, stdin=None):
    """Run the author's gcloud (its stored login), appending the output to `log`; -> exit code."""
    with open(log, "ab") as fh:
        fh.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} gcloud {' '.join(args)}\n".encode())
        fh.flush()
        fin = open(stdin, "rb") if stdin else subprocess.DEVNULL
        try:
            return subprocess.run([GCLOUD, *args], stdin=fin, stdout=fh, stderr=subprocess.STDOUT).returncode
        finally:
            if stdin:
                fin.close()


def _gcloud_list(bucket, prefix):
    """Object metadata under a prefix, listed with the author's login (the v3 bucket lists nothing anonymously)."""
    r = subprocess.run([GCLOUD, "storage", "ls", "--json", f"gs://{bucket}/{prefix}*"], capture_output=True,
                       text=True)
    if r.returncode:
        if "matched no objects" in r.stderr:
            return []
        raise RuntimeError(r.stderr.strip()[-300:])
    return [x["metadata"] for x in json.loads(r.stdout) if x.get("type") == "cloud_object"]


def _gcloud_version():
    r = subprocess.run([GCLOUD, "version", "--format=json"], capture_output=True, text=True)
    return json.loads(r.stdout).get("Google Cloud SDK") if r.returncode == 0 else None


def cmd_store_list(a):
    import requests
    pairs = _store_pairs()
    sets = pd.read_csv(SETS_CSV)
    accs = {k[0] for k in pairs}
    assert accs == set(sets.loc[sets.in_v2c, "acc"]), "the scan and the accession sets disagree"
    local = {MODEL_RE.match(f).group(1) for f in os.listdir(V4_LOCAL) if f.endswith("model_v4.cif.gz")}
    ftp = {MODEL_RE.match(f).group(1) for f in os.listdir(os.path.join(CACHE, "v4_ftp")) if f.endswith("model_v4.cif.gz")}
    want = [(acc, frag, ver, "store") for acc, frag, ver in sorted(pairs)] + \
           [(acc, 1, 4, "check") for acc in sorted((local | ftp) - accs)]
    fields = "items(name,size,md5Hash,crc32c,generation,updated,contentEncoding),nextPageToken"
    s = requests.Session()

    def one(w):
        acc, frag, ver, purpose = w
        prefix = f"AF-{acc}-F{frag}-"
        items = _gcs_list(s, prefix, fields) if ver == 4 else _gcloud_list(BUCKETS[ver], prefix)
        by = {i["name"]: i for i in items}
        rows = []
        for kind, pat in KINDS.items():
            if purpose == "check" and kind != "model":
                continue
            obj = f"{prefix}{pat.format(v=ver)}"
            i = by.get(obj, {})
            rows.append({"acc": acc, "version": ver, "kind": kind, "purpose": purpose, "bucket": BUCKETS[ver],
                         "object": obj, "size": i.get("size"), "md5": i.get("md5Hash"), "crc32c": i.get("crc32c"),
                         "generation": i.get("generation"), "updated": i.get("updated"),
                         "encoding": i.get("contentEncoding"),
                         "local": (f"v{ver}/" if purpose == "store" else "") + obj + ".gz"})
        return rows
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        rows = [r for rs in ex.map(one, want) for r in rs]
    _save(pd.DataFrame(rows), STORE_CSV)
    print(f"listed {len(want):,} accessions in {time.time() - t0:.0f} s; wrote {STORE_CSV}")
    summarise_store_list(pairs)


def _gb(b):
    return f"{b / 1e9:.2f} GB" if b >= 1e7 else f"{b / 1e3:,.0f} kB"


def summarise_store_list(pairs=None):
    obj = _read_store_csv()
    st = obj[obj.purpose == "store"]
    print(f"\n=== the v2c model store: what to copy (listed "
          f"{time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(STORE_CSV)))}) ===")
    if pairs is not None:
        by_ver = collections.Counter(k[2] for k in pairs)
        print(f"  model files named by the v2c archives: {len(pairs):,} ({', '.join(f'v{v} {n:,}' for v, n in sorted(by_ver.items(), reverse=True))}); "
              f"fragments {sorted({k[1] for k in pairs})}; accessions named with two versions "
              f"{len(pairs) - len({k[0] for k in pairs})}")
    for (ver, bucket), g in st.groupby(["version", "bucket"]):
        m = g[g.kind == "model"]
        print(f"  gs://{bucket}: {m.acc.nunique():,} accessions; " + "; ".join(
            f"{kind} listed {int(k['size'].notna().sum()):,} of {len(k):,}, {_gb(k['size'].sum())}"
            for kind, k in g.groupby("kind", sort=False)) + f"; together {_gb(g['size'].sum())}; "
            f"encodings {g.encoding.fillna('none').value_counts().to_dict()}; object dates "
            f"{str(g.updated.min())[:10]} to {str(g.updated.max())[:10]}")
    tot = st["size"].sum()
    print(f"  the store: {int(st['size'].notna().sum()):,} objects, {tot / 1e9:.2f} GB as stored; at a list price of "
          f"$0.12 per GiB of internet egress ${0.12 * tot / 2 ** 30:.2f} (models alone "
          f"${0.12 * st.loc[st.kind == 'model', 'size'].sum() / 2 ** 30:.2f})")
    miss = st[st["size"].isna()]
    print(f"  objects expected but not listed: {len(miss)}" + (f" ({', '.join(miss.object[:6])})" if len(miss) else ""))
    if os.path.isfile(GCS_CSV):
        g = pd.read_csv(GCS_CSV, dtype={"f1_md5": str}).set_index("acc")
        m = st[(st.kind == "model") & (st.version == 4)].set_index("acc")
        both = m.index.intersection(g.index)
        same = (m.loc[both, "size"] == g.loc[both, "f1_size"]) & (m.loc[both, "md5"] == g.loc[both, "f1_md5"])
        print(f"  v4 models: size and md5 equal to gcs_v4_listing.csv (listed "
              f"{time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(GCS_CSV)))}) for {int(same.sum()):,} of {len(m):,}")
    ck = obj[obj.purpose == "check"]
    if len(ck):
        print(f"  check copies (local v4 files outside v2c, never in the store): {len(ck)} "
              f"({', '.join(ck.acc)}), {ck['size'].sum() / 1e3:.0f} kB")


def _admit(r, stage):
    """Check a staged object against the listing and move it into place; -> None, or why not."""
    src = os.path.join(stage, r.object)
    if not os.path.isfile(src):
        return "not copied"
    with open(src, "rb") as fh:
        data = fh.read()
    if len(data) != int(r.size) or _b64md5(data) != r.md5:
        os.remove(src)                  # our own bad copy: the next pass fetches it again
        return f"size {len(data)} and md5 {_b64md5(data)} against the listing's {int(r.size)} and {r.md5}"
    os.makedirs(os.path.dirname(r.path), exist_ok=True)
    if data[:2] == b"\x1f\x8b":
        os.replace(src, r.path)
    else:                               # stored without gzip encoding: gzip it here
        with gzip.open(r.path + ".part", "wb") as fh:
            fh.write(data)
        os.replace(r.path + ".part", r.path)
        os.remove(src)
    return None


def cmd_store_fetch(a):
    obj = _read_store_csv()
    obj = obj[obj["size"].notna()]
    if a.limit:                         # a pilot: a seeded random set of accessions
        pick = sorted(set(obj.acc))
        random.Random(SEED).shuffle(pick)
        obj = obj[obj.acc.isin(set(pick[:a.limit]))]
    os.makedirs(FETCH_DIR, exist_ok=True)
    stats = json.load(open(STORE_FETCH_JSON)) if os.path.isfile(STORE_FETCH_JSON) else {"runs": []}
    run = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "gcloud": _gcloud_version(), "limit": a.limit,
           "workers": a.workers, "objects": len(obj), "passes": []}
    for n_pass in range(1, a.passes + 1):
        todo = obj[~obj.path.map(os.path.isfile)]
        print(f"pass {n_pass}: {len(todo):,} of {len(obj):,} objects to copy ({todo['size'].sum() / 1e9:.2f} GB)",
              flush=True)
        if todo.empty:
            break
        stamp, t0, fails, results = time.strftime("%Y%m%d-%H%M%S"), time.time(), [], collections.Counter()
        for bucket, g in todo.groupby("bucket"):
            stage = os.path.join(FETCH_DIR, "incoming", bucket)
            os.makedirs(stage, exist_ok=True)
            # gcloud resolves a URL list from stdin one object at a time (about 10 per second, whatever its
            # process and thread counts), so the list is split over parallel gcloud runs
            parts = []
            for k in range(min(a.workers, len(g))):
                base = os.path.join(FETCH_DIR, f"{stamp}_{bucket}_{k:02d}")
                with open(base + "_urls.txt", "w", newline="\n") as fh:
                    fh.writelines(f"gs://{bucket}/{o}\n" for o in g.object.iloc[k::a.workers])
                parts.append(base)
            with ThreadPoolExecutor(len(parts)) as ex:
                rcs = list(ex.map(lambda b: _gcloud(["storage", "cp", "--do-not-decompress", "--no-clobber",
                                                     "--read-paths-from-stdin", "--manifest-path", b + "_manifest.csv",
                                                     stage], b + ".log", stdin=b + "_urls.txt"), parts))
            detail, res_b = {}, collections.Counter()
            for b in parts:                     # gcloud's own per-object record: Source, ..., Result, Description
                if os.path.isfile(b + "_manifest.csv"):
                    m = pd.read_csv(b + "_manifest.csv", dtype=str).rename(columns=str.lower)
                    res_b.update(m["result"].str.lower().value_counts().to_dict())
                    bad = m[m["result"].str.lower() == "error"]
                    detail.update({str(src).rsplit("/", 1)[-1]: str(d)[:200]
                                   for src, d in zip(bad["source"], bad.get("description", bad["result"]))})
            results.update(res_b)
            with ThreadPoolExecutor(8) as ex:
                why = list(ex.map(lambda r: _admit(r, stage), g.itertuples()))
            fails += [{"pass": stamp, "bucket": bucket, "object": r.object, "reason": w,
                       "gcloud": detail.get(r.object, "")} for r, w in zip(g.itertuples(), why) if w]
            print(f"  gs://{bucket}: {len(g):,} requested over {len(parts)} gcloud runs (exit codes "
                  f"{sorted(set(rcs))}), manifest {dict(res_b)}; admitted {sum(w is None for w in why):,}, "
                  f"failed {sum(w is not None for w in why):,} ({time.time() - t0:.0f} s)", flush=True)
        if fails:
            f = os.path.join(FETCH_DIR, "fetch_failures.csv")
            pd.DataFrame(fails).to_csv(f, mode="a", header=not os.path.isfile(f), index=False)
        run["passes"].append({"stamp": stamp, "requested": len(todo), "bytes": float(todo["size"].sum()),
                              "seconds": round(time.time() - t0), "gcloud_results": dict(results),
                              "failed": len(fails)})
    run["in_place"] = int(obj.path.map(os.path.isfile).sum())
    run["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["runs"].append(run)
    with open(STORE_FETCH_JSON, "w") as fh:
        json.dump(stats, fh, indent=1)
    print(f"in place {run['in_place']:,} of {len(obj):,}; passes {[(p['requested'], p['failed'], p['seconds']) for p in run['passes']]}")


def _verify_one(job):
    path, size, md5, kind = job
    rec = {"present": os.path.isfile(path)}
    if not rec["present"]:
        return rec
    with open(path, "rb") as fh:
        data = fh.read()
    rec.update(stored_size=len(data), size_ok=len(data) == size, md5_ok=_b64md5(data) == md5,
               md5_hex=hashlib.md5(data).hexdigest(),
               copied=time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(path))))
    try:
        content = gzip.decompress(data)     # checks the gzip CRC and length too
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"gzip: {type(e).__name__}"
        return rec
    rec.update(content_size=len(content), sha256=hashlib.sha256(content).hexdigest())
    if kind == "model":
        rec["parses"] = content.startswith(b"data_AF-") and b"\n_atom_site." in content
    else:
        try:
            json.loads(content)
            rec["parses"] = True
        except ValueError:
            rec["parses"] = False
    return rec


def _global_plddt(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("_ma_qa_metric_global.metric_value"):
                return float(ln.split()[-1])
            if ln.startswith("_atom_site."):
                return None


def _atoms_same(pair):
    """Atoms (chain, residue number and name, atom name) and their coordinates and B-factors, exactly."""
    a0, a1 = protein_atoms(pair[0]), protein_atoms(pair[1])
    common = [k for k in a0 if k in a1]
    d = np.array([np.linalg.norm(np.subtract(a0[k][:3], a1[k][:3])) for k in common]) if common else np.array([np.nan])
    return {"identical": set(a0) == set(a1) and all(a0[k] == a1[k] for k in a0), "n_atoms": len(a0),
            "n_atoms_other": len(a1), "mean_dev": float(d.mean()), "max_dev": float(d.max()),
            "plddt_global": _global_plddt(pair[0]), "plddt_global_other": _global_plddt(pair[1])}


def cmd_store_verify(a):
    obj = _read_store_csv()
    t0 = time.time()
    jobs = [(r.path, r.size, r.md5, r.kind) for r in obj.itertuples()]
    with Pool(a.procs) as pool:
        res = pd.DataFrame(pool.map(_verify_one, jobs, chunksize=64))
    v = pd.concat([obj.reset_index(drop=True), res], axis=1)
    st = v[v.purpose == "store"]
    folders = sorted({os.path.dirname(p) for p in st.path})
    known = set(st.path.map(os.path.normcase))
    extra = [os.path.join(d, f) for d in folders for f in os.listdir(d) if os.path.normcase(os.path.join(d, f)) not in known]
    staged = sum(len(fs) for _, _, fs in os.walk(os.path.join(FETCH_DIR, "incoming")))
    _save(v.drop(columns=["path"]).assign(extra_files=len(extra), staged_left=staged), STORE_VERIFY_CSV)
    man = st.assign(file=st.local, source="gs://" + st.bucket + "/" + st.object, object_updated=st.updated.str[:10])
    man = man[["acc", "version", "kind", "file", "stored_size", "md5", "md5_hex", "content_size", "sha256", "source",
               "generation", "object_updated", "copied"]].rename(columns={"md5": "md5_base64"})
    _save(man, STORE_MANIFEST)
    print(f"verified {len(v):,} objects in {time.time() - t0:.0f} s; wrote {STORE_VERIFY_CSV} and {STORE_MANIFEST}")

    # the content against the other copies: the local v4 files, the EBI tar members, and the served v6 files
    model = v[(v.kind == "model") & (v.version == 4) & v.present.eq(True)].set_index("acc")
    rows = []
    for tag, folder in (("local_v4", V4_LOCAL), ("ftp_v4", os.path.join(CACHE, "v4_ftp"))):
        for f in sorted(os.listdir(folder)):
            if not f.endswith("model_v4.cif.gz"):
                continue
            acc = MODEL_RE.match(f).group(1)
            with open(os.path.join(folder, f), "rb") as fh:
                other = fh.read()
            rec = {"check": tag, "acc": acc, "in_store": acc in model.index and model.at[acc, "purpose"] == "store"}
            if acc in model.index:
                with open(model.at[acc, "path"], "rb") as fh:
                    mine = fh.read()
                rec.update(identical=gzip.decompress(mine) == gzip.decompress(other), bytes_identical=mine == other,
                           gz_size_equal=len(mine) == len(other))
            rows.append(rec)
    h = pd.read_csv(V6HDR_CSV)
    h = h[(h.frag == 1) & h.acc.isin(model.index[model.purpose == "store"])]
    pairs = [(model.at[r.acc, "path"], os.path.join(AFDB_V6, r.file)) for r in h.itertuples()]
    with Pool(a.procs) as pool:
        same = pool.map(_atoms_same, pairs, chunksize=16)
    rows += [{"check": "v6_atoms", "acc": r.acc, "in_store": True, "deposited": r.deposited, **s}
             for r, s in zip(h.itertuples(), same)]
    # a v6 file dated 2022-06-01 that is not the v4 model: did v4 itself change it (an EBI update-shard member)?
    odd = [r for r in rows if r["check"] == "v6_atoms" and str(r["deposited"]).startswith("2022") and not r["identical"]]
    if odd:
        try:
            import requests
            shards = Shards(requests.Session())
            for r in odd:
                sh, o, _ = shards.find(f"AF-{r['acc']}-F1-model_v4.cif.gz")
                r["in_update_shard"] = o is not None
            for t in shards.tars.values():
                t.save()
        except Exception as e:  # noqa: BLE001  (offline: leave the column empty)
            print(f"update-shard probe skipped: {type(e).__name__}")
    _save(pd.DataFrame(rows), STORE_CHECK_CSV)
    print(f"content checks {len(rows):,} in {time.time() - t0:.0f} s; wrote {STORE_CHECK_CSV}")
    summarise_store()


def summarise_store(v=None):
    """The store as copied and checked; `v` (from labelled()) adds the validated jobs' routes."""
    summarise_store_list()
    if os.path.isfile(STORE_FETCH_JSON):
        stats = json.load(open(STORE_FETCH_JSON))
        for run in stats["runs"]:
            p = run["passes"]
            pilot = f", pilot of {run['limit']} accessions" if run["limit"] else ""
            print(f"  copy run {run['started']} to {run['finished']} (gcloud {run['gcloud']}{pilot}, "
                  f"{run.get('workers', 1)} parallel gcloud runs per bucket): "
                  f"{run['objects']:,} objects; passes " + "; ".join(
                      f"{q['requested']:,} requested, {q['bytes'] / 1e9:.2f} GB, {q['seconds']:,} s, gcloud "
                      f"{q['gcloud_results']}, failed {q['failed']}" for q in p) + f"; in place {run['in_place']:,}")
    if os.path.isfile(STORE_VERIFY_CSV):
        d = pd.read_csv(STORE_VERIFY_CSV, low_memory=False)
        st = d[d.purpose == "store"]
        print(f"  on disk ({STORE}): present {int(st.present.sum()):,} of {len(st):,}; files not in the listing "
              f"{int(d.extra_files.iloc[0])}; staged files left {int(d.staged_left.iloc[0])}; "
              f"{st.stored_size.sum() / 1e9:.2f} GB as stored, {st.content_size.sum() / 1e9:.2f} GB decompressed")
        for (ver, kind), g in st.groupby(["version", "kind"], sort=False):
            print(f"    v{ver} {kind:<10} {len(g):>6,}: size and md5 equal to the listing "
                  f"{int((g.size_ok.eq(True) & g.md5_ok.eq(True)).sum()):,}; gzip intact {int(g.sha256.notna().sum()):,}; "
                  f"{'mmCIF with atoms' if kind == 'model' else 'valid JSON'} {int(g.parses.eq(True).sum()):,}")
        good = st.assign(good=st.present.eq(True) & st.size_ok.eq(True) & st.md5_ok.eq(True))
        acc_ok = good.groupby(["version", "acc"]).good.all()
        print(f"  accessions with every object in place and equal to the listing: "
              + ", ".join(f"v{ver} {int(s.sum()):,} of {len(s):,}" for ver, s in acc_ok.groupby(level=0)))
        print(f"  copied on {sorted(st.copied.dropna().unique())}; distinct decompressed models "
              f"{st.loc[st.kind == 'model', 'sha256'].nunique():,}; files per folder "
              f"{st.local.str.split('/').str[0].value_counts().to_dict()}")
        t = st[(st.acc == "P12931") & (st.kind == "model")]
        if len(t):
            print(f"  the test file, P12931's model: {int(t.stored_size.iloc[0]):,} B as stored (gzip), "
                  f"{int(t.content_size.iloc[0]):,} B decompressed")
        ck = d[d.purpose == "check"]
        if len(ck):
            print(f"  check copies: size and md5 equal to the listing {int((ck.size_ok.eq(True) & ck.md5_ok.eq(True)).sum())} "
                  f"of {len(ck)}")
    if os.path.isfile(STORE_CHECK_CSV):
        c = pd.read_csv(STORE_CHECK_CSV)
        for tag, what in (("local_v4", "the local v4 files (webserver\\AF)"),
                          ("ftp_v4", "the v4 members pulled from the EBI tars")):
            g = c[c.check == tag]
            src = ""
            if tag == "ftp_v4" and os.path.isfile(TARS_CSV):
                t = g.merge(pd.read_csv(TARS_CSV), on="acc", how="left")
                shard = t.swissprot_off.isna() & t.proteome_off.isna()
                src = (f" (update-shard members among them {int((shard & t.bytes_identical.eq(True)).sum())} of "
                       f"{int(shard.sum())}; Swiss-Prot or proteome members {int((~shard & t.bytes_identical.eq(True)).sum())})")
            print(f"  decompressed content equal to {what}: {int(g.identical.eq(True).sum())} of {len(g)} "
                  f"({int(g.in_store.sum())} from the store, {int((~g.in_store.astype(bool)).sum())} from check copies); "
                  f"gzip bytes identical {int(g.bytes_identical.eq(True).sum())}{src}, gzip size equal "
                  f"{int(g.gz_size_equal.eq(True).sum())}")
        g = c[c.check == "v6_atoms"]
        if len(g):
            print("  every atom's coordinates and B-factor equal to the local v6 file, by the v6 model's date: " + "; ".join(
                f"made {dep} {int(x.identical.sum()):,} of {len(x):,}" for dep, x in g.groupby("deposited")))
            odd = g[g.deposited.astype(str).str.startswith("2022") & ~g.identical.astype(bool)]
            sets = pd.read_csv(SETS_CSV).set_index("acc")
            for r in odd.to_dict("records"):
                print(f"    made 2022-06-01 but not the v4 model: {r['acc']} (atoms {int(r['n_atoms']):,}, the same set: "
                      f"{int(r['n_atoms']) == int(r['n_atoms_other'])}); atoms moved {r['mean_dev']:.1f} A on average, "
                      f"{r['max_dev']:.1f} A at most; global pLDDT v4 {r['plddt_global']:.2f}, v6 "
                      f"{r['plddt_global_other']:.2f}; member of a v4 update shard: {r.get('in_update_shard')}; in the sets "
                      f"{[n for n, col, _ in SETS if sets.at[r['acc'], col]]}")
    if os.path.isfile(STORE_VALID_CSV):
        j = pd.read_csv(STORE_VALID_CSV)
        ok = j[j.status == "ok"]
        same = ok[[c for c in ok if c.startswith("same_")]].astype(bool)
        if os.path.isfile(STORE_VALID_JOBS_CSV):
            with open(STORE_VALID_JOBS_CSV) as fh:
                head = fh.readline().strip()
            if head.startswith("#"):
                print(f"  the jobs, {head[2:]}")
        print(f"  v2c jobs on the store files: {len(j)} jobs, one per accession ({j.acc.nunique()} accessions; "
              f"versions {j.version.value_counts().to_dict()}; query crystal {j.crystal.value_counts().to_dict()}); "
              f"ran {len(ok)} ({j.status.value_counts().to_dict()}); archived US-align report reproduced verbatim "
              f"{int(ok.frame_text_identical.astype(bool).sum())}, all five model-row values equal "
              f"{int(same.all(1).sum())} ({', '.join(f'{c[5:]} {int(same[c].sum())}' for c in same)})")
        for i, r in ok[~ok.frame_text_identical.astype(bool)].iterrows():
            print(f"    report not verbatim: {r.acc} {r.job_name}; largest difference in the matrix "
                  f"{r.frame_max_dev:.0e} (US-align prints 10 decimals); all five values equal {bool(same.loc[i].all())}")
        if v is not None:
            r = ok.merge(v[["acc", "route", "label_source"]], on="acc", how="left")
            r["all5"] = same.all(1).values
            print("    by route: " + "; ".join(
                f"{k} {int(g2.all5.sum())} of {len(g2)}" for k, g2 in r.groupby(r.route.fillna("v3 (no label)"))))


# =============================================================================== EBI tars by byte range
# Member order. The tars are sorted by accession, then fragment, but NOT byte-wise within an accession:
# the update shards store model, confidence, PAE (model_v4.cif.gz before confidence_v4.json.gz), the
# proteome tars cif before pdb. Comparing whole names byte-wise lets a probe that lands on an accession's
# own confidence file skip its model file (a false 'absent'), so names are compared by this key.
_RANK = {"model_v4.cif.gz": 0, "model_v4.pdb.gz": 1, "confidence_v4.json.gz": 2,
         "predicted_aligned_error_v4.json.gz": 3}
HDR_CACHE = os.path.join(CACHE, "tar_headers")


def member_key(name):
    m = re.match(r"AF-(.+?)-F(\d+)-(.+)$", name)
    return (m.group(1), int(m.group(2)), _RANK.get(m.group(3), 9)) if m else (name, 0, 0)


class RemoteTar:
    """A sorted, uncompressed tar on an HTTP server, searched by byte ranges.

    Headers are found with multi-range requests for the 'ustar' magic of every 512-byte block in a window
    (about 109 bytes per block on the wire instead of 512), then confirmed by reading the block, whose
    checksum tarfile verifies. Confirmed headers and scan results persist in HDR_CACHE, so a re-run
    repeats no read it has already made."""
    MAX_RANGES = 200                   # the EBI server's limit; 201 ranges -> 200 OK with the whole file

    def __init__(self, session, url, walk=1 << 20):
        self.s, self.url, self.walk = session, url, walk
        self.path = os.path.join(HDR_CACHE, re.sub(r"[^\w.-]", "_", url.split("/alphafold/", 1)[-1]) + ".json")
        saved = json.load(open(self.path)) if os.path.isfile(self.path) else {}
        if saved.get("total"):
            self.total = saved["total"]
        else:
            h = session.head(url, timeout=60)
            h.raise_for_status()
            self.total = int(h.headers["Content-Length"])
        self.hdr = {int(k): (tarfile.TarInfo(v[0]) if v else None) for k, v in saved.get("hdr", {}).items()}
        for k, v in saved.get("hdr", {}).items():
            if v:
                self.hdr[int(k)].size = v[1]
        self.cache = {int(k): v for k, v in saved.get("scan", {}).items()}   # probe -> header offset
        self.requests = self.bytes = self.order_violations = 0

    def save(self):
        os.makedirs(HDR_CACHE, exist_ok=True)
        with open(self.path + ".part", "w") as fh:
            json.dump({"total": self.total, "scan": self.cache,
                       "hdr": {k: ([t.name, t.size] if t is not None else None) for k, t in self.hdr.items()}}, fh)
        os.replace(self.path + ".part", self.path)

    def _get(self, spec, limit):
        with self.s.get(self.url, headers={"Range": "bytes=" + spec}, stream=True, timeout=120) as r:
            if r.status_code != 206:           # never read a non-ranged body
                raise RuntimeError(f"{self.url}: status {r.status_code} to a ranged request")
            body = r.raw.read(limit + 1)
            ctype = r.headers.get("Content-Type", "")
        if len(body) > limit:
            raise RuntimeError("ranged response larger than expected")
        self.requests += 1
        self.bytes += len(body)
        return ctype, body

    def block(self, off):
        """TarInfo of the header at `off`, or None."""
        if off in self.hdr:
            return self.hdr[off]
        if off + 512 > self.total:
            return None
        _, b = self._get(f"{off}-{off + 511}", 512)
        ti = None
        if b[257:262] == b"ustar":
            try:
                ti = tarfile.TarInfo.frombuf(b, "utf-8", "surrogateescape")
            except Exception:  # noqa: BLE001
                ti = None
        self.hdr[off] = ti
        return ti

    def _magic(self, offs):
        spec = ",".join(f"{o + 257}-{o + 261}" for o in offs)
        ctype, body = self._get(spec, 200 * len(offs) + 4096)
        if "multipart/byteranges" not in ctype:
            return {offs[0]: body[:5]}
        out = {}
        for part in body.split(b"\r\n--")[1:]:
            head, _, data = part.partition(b"\r\n\r\n")
            m = re.search(rb"[Cc]ontent-[Rr]ange: bytes (\d+)-(\d+)/", head)
            if m:
                a0, a1 = int(m.group(1)), int(m.group(2))
                out[a0 - 257] = data[:a1 - a0 + 1]
        return out

    def first_at_or_after(self, off):
        """(offset, TarInfo) of the first header at a block offset >= off; (total, None) past the end."""
        off = (off + 511) // 512 * 512
        if off in self.cache:
            o = self.cache[off]
            return o, (self.hdr.get(o) if o < self.total else None)
        start, n = off, 64
        while start < self.total:
            offs = [o for o in range(start, min(start + n * 512, self.total), 512)]
            got = self._magic(offs)
            for o in offs:
                if got.get(o) == b"ustar":
                    ti = self.block(o)
                    if ti is not None:
                        self.cache[off] = o
                        return o, ti
            start = offs[-1] + 512
            n = self.MAX_RANGES
        self.cache[off] = self.total
        return self.total, None

    @staticmethod
    def next_off(o, ti):
        return o + 512 + (ti.size + 511) // 512 * 512

    def find(self, target):
        """-> (offset, TarInfo) if member `target` exists, else (None, (previous name, next name))."""
        tk = member_key(target)
        lo, hi = 0, self.total
        while hi - lo > self.walk:
            mid = (lo + hi) // 2
            o, ti = self.first_at_or_after(mid)
            if ti is None or member_key(ti.name) > tk:
                hi = mid
            elif ti.name == target:
                return o, ti
            else:
                lo = self.next_off(o, ti)
        o, ti = self.first_at_or_after(lo)
        prev = None
        while ti is not None and o < hi + 512:
            k = member_key(ti.name)
            if prev is not None and k <= member_key(prev):
                self.order_violations += 1
            if ti.name == target:
                return o, ti
            if k > tk:
                return None, (prev, ti.name)
            prev = ti.name
            o = self.next_off(o, ti)
            ti = self.block(o) if o < self.total else None
        return None, (prev, ti.name if ti is not None else None)

    def member(self, o, ti):
        _, b = self._get(f"{o + 512}-{o + 511 + ti.size}", ti.size)
        return b


class Shards:
    """v4/updated_entries: 921 globally sorted shards; the first member of each shard indexes them."""

    def __init__(self, session):
        self.s = session
        html = session.get(FTP + "v4/updated_entries/", timeout=60).text
        self.names = sorted(set(re.findall(r'href="(v4_updated_entries-\d+\.tar)"', html)),
                            key=lambda x: int(re.search(r"-(\d+)\.tar", x).group(1)))
        if os.path.isfile(SHARDS_CSV):
            idx = pd.read_csv(SHARDS_CSV)
        else:
            rows = []
            for n in self.names:
                t = RemoteTar(session, FTP + "v4/updated_entries/" + n, walk=256 << 10)
                ti = t.block(0)
                rows.append({"shard": n, "first": ti.name, "bytes": t.total})
            idx = pd.DataFrame(rows)
            _save(idx, SHARDS_CSV)
        self.index = idx
        self.tars = {}
        self.requests = len(idx)
        self.bytes = 512 * len(idx)

    def find(self, target):
        k = int(np.searchsorted(self.index["first"].values, target, side="right")) - 1
        if k < 0:
            return None, None, None
        n = self.index.shard.iloc[k]
        if n not in self.tars:
            self.tars[n] = RemoteTar(self.s, FTP + "v4/updated_entries/" + n, walk=256 << 10)
        o, ti = self.tars[n].find(target)
        return n, o, ti

    def stats(self):
        return (self.requests + sum(t.requests for t in self.tars.values()),
                self.bytes + sum(t.bytes for t in self.tars.values()))


def labelled():
    """Every accession of the sets with two labels and the route they imply.

    tar      predicted from UniProt: the EBI v4 tars hold it (Swiss-Prot 2021_04 or a v4 reference proteome)
    carried  the served v6 model is the v4 model: exact from the local v6 header (model made 2022-06-01), else
             predicted from UniProt (not in Swiss-Prot or an AFDB proteome at 2025_03)
    route    'tar' | 'v6_same' | 'neither' (re-predicted in 2025 but not in a tar) | 'obsolete' (no UniProt
             entry and no local v6 file)"""
    sets = pd.read_csv(SETS_CSV)
    u = pd.read_csv(UNIPROT_CSV, dtype=str)
    sp, pr = predict(u, _ftp_v4_proteomes())
    u["pred_sp"], u["pred_prot"] = sp.values, pr.values
    u["pred_repred"] = predict_repredicted(u).values
    h = pd.read_csv(V6HDR_CSV)
    h = h[h.frag == 1][["acc", "deposited", "first_rev"]]
    v = sets.merge(u, on="acc", how="left").merge(h, on="acc", how="left")
    for c in ("pred_sp", "pred_prot", "pred_repred"):
        v[c] = v[c].eq(True)
    v["tar"] = v.pred_sp | v.pred_prot
    hdr, has_u = v.deposited.notna(), v.reviewed.notna()
    v["label_source"] = np.select([hdr, has_u], ["v6_header", "uniprot"], "none")
    v["carried"] = np.where(hdr, v.deposited.astype(str).str.startswith("2022"), has_u & ~v.pred_repred)
    v["route"] = np.select([v.tar, v.carried, ~hdr & ~has_u], ["tar", "v6_same", "obsolete"], "neither")
    return v, u


def _api_facts(session, acc):
    try:
        r = session.get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=60)
        if r.status_code != 200:
            return {"api_status": r.status_code}
        j = [x for x in r.json() if x.get("entryId") == f"AF-{acc}-F1"]
        if not j:
            return {"api_status": "no_F1"}
        j = j[0]
        return {"api_status": 200, "api_latest": j.get("latestVersion"),
                "api_first": min(j.get("allVersions") or [0]),
                "api_versions": "-".join(map(str, j.get("allVersions") or [])),
                "api_created": str(j.get("modelCreatedDate", ""))[:10], "api_reviewed": j.get("isReviewed")}
    except Exception as e:  # noqa: BLE001
        return {"api_status": f"err:{type(e).__name__}"}


def _redo_shards(a):
    """Repeat only the update-shard searches (the first run compared names byte-wise; see member_key)."""
    import requests
    df = pd.read_csv(TARS_CSV)
    stats_path = TARS_CSV.replace(".csv", "_stats.json")
    stats = json.load(open(stats_path))
    s = requests.Session()
    shards = Shards(s)
    shards.requests = shards.bytes = 0                      # the index is cached; count only new reads
    todo = df.swissprot_off.isna() & df.proteome_off.isna()
    before = int(df.loc[todo, "shard_off"].notna().sum())
    for i in df.index[todo]:
        sh, o, ti = shards.find(f"AF-{df.at[i, 'acc']}-F1-model_v4.cif.gz")
        df.at[i, "shard"] = sh if o is not None else None
        df.at[i, "shard_off"] = o
        df.at[i, "shard_size"] = ti.size if o is not None else None
    for t in shards.tars.values():
        t.save()
    _save(df, TARS_CSV)
    nreq, nbytes = shards.stats()
    viol = sum(t.order_violations for t in shards.tars.values())
    stats.update({"shards_redone": stats.get("shards_redone", 0) + 1, "requests": stats["requests"] + nreq,
                  "bytes": stats["bytes"] + nbytes, "order_violations_first_run": stats["order_violations"],
                  "order_violations": viol, "shard_hits_first_run": before,
                  "shards_opened": len(shards.tars)})
    with open(stats_path, "w") as fh:
        json.dump(stats, fh)
    print(f"update shards searched again for {int(todo.sum())} accessions held by neither tar family: hits "
          f"{int(df.loc[todo, 'shard_off'].notna().sum())} (first run {before}); {nreq:,} requests, "
          f"{nbytes / 1e6:.1f} MB; order violations {viol} (first run, byte-wise names: {stats['order_violations_first_run']})")


def cmd_tars(a):
    import requests
    if a.redo_shards:
        return _redo_shards(a)
    v, u = labelled()
    v = v[v.in_v2c]
    prot = _ftp_v4_proteomes()
    if os.path.isfile(SAMPLE_CSV) and not a.resample:
        samp = pd.read_csv(SAMPLE_CSV)
    else:
        # strata: predicted route x where the carried-over label comes from (v6 header | UniProt | none)
        strata = list(v.groupby(["route", "label_source"]))
        k = a.n // len(strata)
        samp = pd.concat([g.sample(n=min(k, len(g)), random_state=SEED) for _, g in strata])
        samp = samp[["acc", "route", "label_source", "taxid"]].sort_values("acc")
        samp.to_csv(SAMPLE_CSV, index=False)
    local = sorted(re.match(r"AF-(\S+)-F1-model_v4", f).group(1) for f in os.listdir(V4_LOCAL) if "_v4." in f)
    todo = sorted(set(samp.acc) | set(local))
    tax = dict(zip(u.acc, u.taxid))
    s = requests.Session()
    sp_tar = RemoteTar(s, FTP + "v4/swissprot_cif_v4.tar")
    ptars = {}
    shards = Shards(s)
    rows = []
    for acc in todo:
        name = f"AF-{acc}-F1-model_v4.cif.gz"
        r = {"acc": acc, "in_sample": acc in set(samp.acc), "local_v4": acc in local}
        r.update(_api_facts(s, acc))
        o, ti = sp_tar.find(name)
        r["swissprot_off"] = o
        r["swissprot_size"] = ti.size if o is not None else None
        t = prot.get(str(tax.get(acc)))
        r["proteome_tar"] = t
        if t:
            if t not in ptars:
                ptars[t] = RemoteTar(s, FTP + "v4/" + t)
            o2, ti2 = ptars[t].find(name)
            r["proteome_off"] = o2
            r["proteome_size"] = ti2.size if o2 is not None else None
        # the update shards matter only for what the two tar families miss (saves transfer)
        if r["swissprot_off"] is None and r.get("proteome_off") is None:
            sh, o3, ti3 = shards.find(name)
            r["shard"] = sh if o3 is not None else None
            r["shard_off"] = o3
            r["shard_size"] = ti3.size if o3 is not None else None
        rows.append(r)
        print(f"  {acc:<11} swissprot {r['swissprot_off'] is not None!s:<5} proteome {r.get('proteome_off') is not None!s:<5} "
              f"shard {r.get('shard_off') is not None!s:<5}", flush=True)
    df = pd.DataFrame(rows)
    _save(df, TARS_CSV)
    for t in [sp_tar, *ptars.values(), *shards.tars.values()]:
        t.save()
    nreq = sp_tar.requests + sum(t.requests for t in ptars.values()) + shards.stats()[0]
    nbytes = sp_tar.bytes + sum(t.bytes for t in ptars.values()) + shards.stats()[1]
    viol = sp_tar.order_violations + sum(t.order_violations for t in ptars.values()) + \
        sum(t.order_violations for t in shards.tars.values())
    stats = {"date": time.strftime("%Y-%m-%d"), "accessions": len(todo), "proteome_tars": len(ptars),
             "shards_opened": len(shards.tars), "requests": nreq, "bytes": nbytes, "order_violations": viol}
    with open(TARS_CSV.replace(".csv", "_stats.json"), "w") as fh:
        json.dump(stats, fh)
    print(f"searched {len(todo)} accessions in the Swiss-Prot tar, {len(ptars)} proteome tars and the update shards: "
          f"{nreq:,} ranged requests, {nbytes / 1e6:.1f} MB transferred, sort-order violations {viol}")


# =============================================================================== validate
# The v2.1.0 model-row recipe, as reproduced by the authors' AF-centric re-run prototype (af_centric_rerun.py,
# not part of this repository): RoG mass-weighted over all atoms but D; SASA get_area with dot_solvent 1;
# PyVOL min_volume 200; pocket RMSD cmd.align(cycles=0) against the archived query pocket; the model placed
# by US-align 20240319 -fast on C-alpha-only PDB files written without TER records.
_PM = None


def _pymol():
    global _PM
    if _PM is None:
        os.environ["PATH"] = os.path.join(os.path.dirname(PY311), "bin") + os.pathsep + os.environ["PATH"]  # msms
        import pymol
        pymol.finish_launching(["pymol", "-qc"])
        from pymol import cmd
        fetch = os.path.join(CACHE, "pymol_fetch")      # chemical components PyMOL downloads (e.g. UNK.cif), not the cwd
        os.makedirs(fetch, exist_ok=True)
        cmd.set("fetch_path", fetch)
        _PM = cmd
    return _PM


def tgz_read(tgz, member):
    with tarfile.open(tgz, "r:gz") as tar:
        for cand in (f"./{member}", member):
            try:
                f = tar.extractfile(cand)
            except KeyError:
                continue
            if f is not None:
                return f.read()
    return None


def _rog_v21(cmd, sele):
    atoms = cmd.get_model(sele + " and not elem D").atom
    if not atoms:
        return np.nan, np.nan
    x = np.array([t.coord for t in atoms])
    m = np.array([t.get_mass() for t in atoms])
    c = (x * m[:, None]).sum(0) / m.sum()
    rg = float(np.sqrt((m * ((x - c) ** 2).sum(1)).sum() / m.sum()))
    return round(rg, 2), round(float(sum(t.b for t in cmd.get_model(sele).atom)), 2)


def _pyvol_volume(sele):
    import contextlib
    if "float_" not in np.__dict__:          # PyVOL 1.7.8 still calls np.float_, removed in NumPy 2
        np.float_ = np.float64
    from pyvol.pymol_interface import pymol_pocket_cmdline
    tmp, cwd, buf = tempfile.mkdtemp(), os.getcwd(), io.StringIO()
    try:
        os.chdir(tmp)
        with contextlib.redirect_stdout(buf):
            pymol_pocket_cmdline(protein=sele, min_volume=200)
    except Exception:  # noqa: BLE001  (v2.1.0: except -> 0 for the model)
        return 0.0
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    m = re.search(r"Volume: (\d+\.?\d*) A\^3", buf.getvalue())
    return float(m.group(1)) if m else None


def _usalign_report(cmd, mobile, target):
    """v2.1.0: US-align 20240319 -fast on CA-only PDB files written without TER records -> matrix-file text."""
    tmp = tempfile.mkdtemp()
    try:
        mf, tf, mx = (os.path.join(tmp, n) for n in ("mobile.pdb", "target.pdb", "m.txt"))
        v = cmd.get_setting_boolean("pdb_use_ter_records")
        cmd.set("pdb_use_ter_records", 0)
        cmd.save(mf, f"({mobile}) and (not hetatm) and name CA and alt +A", state=1)
        cmd.save(tf, f"({target}) and (not hetatm) and name CA and alt +A", state=1)
        if v:
            cmd.set("pdb_use_ter_records", 1)
        out = subprocess.run([USALIGN, mf, tf, "-m", mx, "-fast"], capture_output=True, text=True, timeout=600)
        # v2.1.0 archived US-align's report followed by the -m matrix block, in one file
        return out.stdout + open(mx).read() if os.path.isfile(mx) else None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _report_core(txt):
    """The parts of a US-align matrix file that depend on the coordinates (not on temp-file names or CPU time)."""
    keep = []
    for ln in txt.splitlines():
        if ln.startswith(("Name of Structure", "#Total CPU time")):
            continue
        keep.append(ln.rstrip())
    return "\n".join(keep).strip()


def _matrix(txt):
    m, row = [], 0
    for ln in txt.splitlines():
        if 4 >= row > 0:
            if row >= 2:
                v = list(map(float, ln.split()))
                m.extend(v[2:5])
                m.append(v[1])
            row += 1
        elif ln.lower().startswith("------ the rotation matrix"):
            row = 1
    return m + [0, 0, 0, 1] if len(m) == 12 else None


CRYSTAL_CACHES = (os.path.join(PILOT_WD, "structures"), os.path.join(WEBSERVER, "structures"),
                  os.path.join(CACHE, "structures"))
_DOWNLOADS = {"n": 0}


def _cached_crystal(pdb):
    return next((p for p in (os.path.join(c, pdb[1:3], f"{pdb}.cif.gz") for c in CRYSTAL_CACHES)
                 if os.path.isfile(p)), None)


def _struct_path(pdb, max_downloads=30):
    p = _cached_crystal(pdb)
    if p:
        return p
    if _DOWNLOADS["n"] >= max_downloads:
        _DOWNLOADS.setdefault("why", []).append((pdb, "cap"))
        return None
    _DOWNLOADS["n"] += 1
    p = os.path.join(CACHE, "structures", pdb[1:3], f"{pdb}.cif.gz")
    import requests
    with requests.get(f"https://files.rcsb.org/download/{pdb}.cif.gz", stream=True, timeout=300) as r:
        if r.status_code != 200 or int(r.headers.get("Content-Length", 0)) > 10_000_000:
            _DOWNLOADS.setdefault("why", []).append((pdb, f"status {r.status_code}"))
            return None
        data = r.raw.read(10_000_001)
    if len(data) > 10_000_000:                  # no Content-Length (chunked): never keep a truncated file
        _DOWNLOADS.setdefault("why", []).append((pdb, "over 10 MB"))
        return None
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(data)
    return p


def _validate_job(cmd, job, tgz, model_path, strip=False):
    """strip: drop every non-protein atom after loading (AlphaFill models carry transplanted ligands)."""
    rec = {"job_name": job}
    sel = tgz_read(tgz, "pocket_selections.csv").decode("utf-8", "replace").splitlines()
    gr = ahoj_job.read_global_results(tgz, low_memory=False)
    m = gr[gr.role == "M"]
    q_line = next(ln for ln in sel if not ln.startswith("AF_"))
    af_line = next(ln for ln in sel if ln.startswith("AF_"))
    pdb, q_expr = q_line.split(".", 1)[0], q_line.split(",", 1)[1]
    resi = re.search(r"resi ([0-9+]+)", af_line).group(1)
    q_chain = str(m.chains3.iloc[0])
    mat_name = str(m.aln_matrix.iloc[0]).split()[0]
    saved = tgz_read(tgz, f"matrices/{mat_name}").decode("utf-8", "replace")
    qp = _struct_path(pdb)
    if qp is None:
        return {**rec, "status": "query_download_failed"}
    cmd.reinitialize()
    cmd.load(qp, pdb)
    cmd.select("qp", q_expr)
    cmd.load(model_path, "m4")
    if strip:
        cmd.remove("m4 and not polymer.protein")
    cmd.alter("m4 and chain A", f"chain='{q_chain}'")
    new = _usalign_report(cmd, "m4", f"{pdb} and chain {q_chain}")
    rec["frame_text_identical"] = new is not None and _report_core(new) == _report_core(saved)
    mat_new, mat_old = _matrix(new or ""), _matrix(saved)
    rec["frame_max_dev"] = (float(np.abs(np.array(mat_new) - np.array(mat_old)).max())
                            if mat_new and mat_old else np.nan)
    cmd.transform_object("m4", mat_old, state=0, homogenous=1)       # the archived frame
    cmd.select("mp", f"m4 and chain {q_chain} and resi {resi}")
    rg, bsum = _rog_v21(cmd, "mp")
    cmd.set("dot_solvent", 0)
    cmd.flag("ignore", "hetatm and not polymer")
    cmd.set("dot_solvent", 1)
    sasa = round(cmd.get_area("mp"), 2)
    sc = cmd.align("mp", "qp", cycles=0, transform=0)
    vol = _pyvol_volume("mp")
    got = {"RoG": rg, "bfac_t": bsum, "sasa": sasa, "pocket_rms": round(sc[0], 2), "volume": vol}
    for k, v in got.items():
        ref = pd.to_numeric(m[k], errors="coerce").iloc[0]
        rec[f"v2c_{k}"], rec[f"ftp_{k}"] = ref, v
        rec[f"same_{k}"] = bool(v is not None and np.isclose(float(v), float(ref), atol=0.0051))
    rec["status"] = "ok"
    return rec


STORE_VALID_CSV = os.path.join(OUT_DIR, "validation_store.csv")
STORE_VALID_JOBS_CSV = P.out("afdb_v4_store_validation_jobs.csv")              # committed: the jobs of arm 'store'


def _validate_store(a):
    """Arm 'store': v2c jobs re-measured on the recovered files, one single-model job per accession.

    The accessions are taken in seeded random order, the three v3 models first. For each, a job whose query
    crystal is already on disk is preferred; at most --max-downloads accessions may take a job whose crystal
    must be fetched (RCSB, under 10 MB), and further accessions without a cached crystal are passed over."""
    obj = _read_store_csv()
    store = {(r.acc, int(r.version)): r.path for r in obj[(obj.purpose == "store") & (obj.kind == "model")].itertuples()}
    scan = pd.read_csv(SCAN_CSV, usecols=["job_name", "batch", "n_af_sel", "n_af_mat", "af_models"], low_memory=False)
    single = scan[(scan.n_af_sel == 1) & (scan.n_af_mat == 1)].copy()
    mm = single.af_models.str.extract(MODEL_RE)
    single["acc"], single["version"] = mm[0], mm[2].astype(int)
    single = single.set_index("job_name", drop=False)
    if os.path.isfile(STORE_VALID_JOBS_CSV) and not a.rechoose:
        chosen = pd.read_csv(STORE_VALID_JOBS_CSV, comment="#")
    else:
        rng = random.Random(SEED)
        pdbs = single.job_name.str[:4]
        cached = {p: _cached_crystal(p) is not None for p in set(pdbs)}
        single["cached"] = pdbs.map(cached)
        v3 = sorted(set(single.loc[single.version == 3, "acc"]))
        rest = sorted(set(single.acc) - set(v3))
        rng.shuffle(rest)
        groups = dict(tuple(single.groupby("acc")))
        rows, fetched, passed = [], 0, 0
        for acc in v3 + rest:
            if len(rows) >= a.n:
                break
            g = groups[acc]
            c = g[g.cached]
            if len(c):
                job, how = c.iloc[rng.randrange(len(c))], "cached"
            elif fetched < a.max_downloads:
                job, how = g.iloc[rng.randrange(len(g))], "download"
                fetched += 1
            else:
                passed += 1
                continue
            rows.append({"acc": acc, "version": int(job.version), "job_name": job.job_name, "crystal": how})
        chosen = pd.DataFrame(rows)
        acc_c = single.groupby("acc").cached.any()
        # the choice depends on this machine's crystal caches (and fills them), so its facts are kept with it
        head = (f"# chosen {time.strftime('%Y-%m-%d')}: single-model v2c jobs {len(single):,} over "
                f"{single.acc.nunique():,} accessions; accessions with a job whose query crystal was on disk "
                f"{int(acc_c.sum()):,}; crystal to fetch {fetched}; passed over for lack of a cached crystal {passed:,}")
        with open(STORE_VALID_JOBS_CSV, "w", newline="") as fh:
            fh.write(head + "\n")
            chosen.to_csv(fh, index=False, lineterminator="\n")
        print(head[2:], flush=True)
    done = pd.read_csv(STORE_VALID_CSV) if os.path.isfile(STORE_VALID_CSV) and not a.rechoose else pd.DataFrame()
    rows = done[done.status == "ok"].to_dict("records") if len(done) else []
    skip = {r["job_name"] for r in rows}
    cmd = _pymol()
    t0 = time.time()
    for i, r in enumerate(chosen.itertuples(), 1):
        if r.job_name in skip:
            continue
        job = single.loc[[r.job_name]].iloc[0]
        tgz = os.path.join(ROOT, str(job.batch).zfill(3), f"{r.job_name}.tgz")
        try:
            rec = _validate_job(cmd, r.job_name, tgz, store[(r.acc, int(r.version))])
        except Exception as e:  # noqa: BLE001
            rec = {"job_name": r.job_name, "status": f"error:{type(e).__name__}:{str(e)[:100]}"}
        rec.update({"check": "v2c_job", "arm": "store", "acc": r.acc, "version": r.version, "crystal": r.crystal})
        rows.append(rec)
        print(f"  {i:>4} {r.acc:<11} v{r.version} {r.job_name:<22} {rec.get('status')} frame {rec.get('frame_text_identical')} "
              + " ".join(f"{k}={rec.get(f'same_{k}')}" for k in ("RoG", "bfac_t", "sasa", "pocket_rms", "volume"))
              + f"  ({time.time() - t0:.0f} s)", flush=True)
        if i % 10 == 0:
            _save(pd.DataFrame(rows), STORE_VALID_CSV)
    _save(pd.DataFrame(rows), STORE_VALID_CSV)
    print(f"store arm: {len(rows)} jobs recorded; query crystals downloaded {_DOWNLOADS['n']}, not obtained "
          f"{collections.Counter(w for _, w in _DOWNLOADS.get('why', []))}")
    summarise_store()


def cmd_validate(a):
    if a.arm == "store":
        return _validate_store(a)
    import requests
    tars = pd.read_csv(TARS_CSV)
    for c in ("proteome_tar", "proteome_off", "shard", "shard_off"):
        if c not in tars:
            tars[c] = np.nan
    scan = pd.read_csv(SCAN_CSV, low_memory=False)
    single = scan[(scan.n_af_sel == 1) & (scan.n_af_mat == 1)].copy()
    single["acc"] = single.af_models.map(lambda s: _accs(s)[0])
    found = tars[tars.swissprot_off.notna() | tars.proteome_off.notna() | tars.shard_off.notna()]
    s = requests.Session()
    os.makedirs(os.path.join(CACHE, "v4_ftp"), exist_ok=True)
    rows, remotes = [], {}

    def fetch(r):
        """v4 member bytes from whichever EBI tar holds it (offsets from the `tars` search)."""
        name = f"AF-{r.acc}-F1-model_v4.cif.gz"
        path = os.path.join(CACHE, "v4_ftp", name)
        if os.path.isfile(path):
            return path, open(path, "rb").read()
        if pd.notna(r.swissprot_off):
            url, off = FTP + "v4/swissprot_cif_v4.tar", int(r.swissprot_off)
        elif pd.notna(r.proteome_off):
            url, off = FTP + "v4/" + r.proteome_tar, int(r.proteome_off)
        else:
            url, off = FTP + "v4/updated_entries/" + r.shard, int(r.shard_off)
        if url not in remotes:
            remotes[url] = RemoteTar(s, url)
        t = remotes[url]
        ti = t.block(off)
        assert ti.name == name, (ti.name, name)
        data = t.member(off, ti)
        with open(path, "wb") as fh:
            fh.write(data)
        return path, data

    # (1) the local v4 files: decompressed content, and size against the bucket listing; a local v4 file
    #     that no tar holds is compared atom by atom with the served v6 file instead
    gcs = pd.read_csv(GCS_CSV) if os.path.isfile(GCS_CSV) else pd.DataFrame(columns=["acc", "f1_size"])
    extra = [x for x in tars.loc[tars.local_v4, "acc"] if x not in set(gcs.acc)]   # local copies outside the sets
    gcs = pd.concat([gcs, pd.DataFrame([_gcs_one(x) for x in extra])], ignore_index=True).set_index("acc")
    for r in found[found.local_v4].itertuples():
        path, data = fetch(r)
        loc = gzip.decompress(open(os.path.join(V4_LOCAL, f"AF-{r.acc}-F1-model_v4.cif.gz"), "rb").read())
        rows.append({"check": "local_v4", "acc": r.acc, "identical": gzip.decompress(data) == loc,
                     "gz_size_equals_bucket": gcs is not None and r.acc in gcs.index
                     and int(gcs.loc[r.acc, "f1_size"]) == len(data)})
    import gemmi

    def atoms(p):
        st = gemmi.read_structure(p)
        return {(c.name, res.seqid.num, res.name, at.name): (at.pos.x, at.pos.y, at.pos.z, at.b_iso, at.occ)
                for c in st[0] for res in c for at in res}
    for r in tars[tars.local_v4 & ~tars.acc.isin(found.acc)].itertuples():
        p6 = next((p for p in (os.path.join(AFDB_V6, f"AF-{r.acc}-F1-model_v6.cif.gz"),
                               os.path.join(V4_LOCAL, f"AF-{r.acc}-F1-model_v6.cif.gz")) if os.path.isfile(p)), None)
        a4 = atoms(os.path.join(V4_LOCAL, f"AF-{r.acc}-F1-model_v4.cif.gz"))
        a6 = atoms(p6) if p6 else {}
        same = bool(a6) and set(a4) == set(a6) and all(
            np.allclose(a4[k][:4], a6[k][:4], atol=0, rtol=0) for k in a4)
        rows.append({"check": "local_v4_vs_v6", "acc": r.acc, "identical": same, "api_created": r.api_created,
                     "n_atoms": len(a4)})
    def v6_file(acc):
        """The served v6 model: the local detection-set copy, else a per-model download (< 10 MB)."""
        for p in (os.path.join(AFDB_V6, f"AF-{acc}-F1-model_v6.cif.gz"),
                  os.path.join(CACHE, "v6", f"AF-{acc}-F1-model_v6.cif.gz")):
            if os.path.isfile(p):
                return p
        with s.get(f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.cif", stream=True, timeout=300) as r:
            if r.status_code != 200:
                return None
            body = r.raw.read(10_000_001, decode_content=True)
        if len(body) > 10_000_000:
            return None
        p = os.path.join(CACHE, "v6", f"AF-{acc}-F1-model_v6.cif.gz")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with gzip.open(p, "wb") as fh:
            fh.write(body)
        return p

    # (2) sampled v2c jobs, one per accession: the archived US-align report and the model-row metrics,
    #     arm "ftp_v4"        2021-era models, v4 file pulled from the EBI tar       -> expect identical
    #     arm "v6_carried"    2022-era models (v6 made 2022-06-01), served v6 file   -> expect identical
    #     arm "v6_repredicted" control: 2021-era models, served v6 file (made 2025)  -> expect different
    rng = random.Random(SEED)
    cmd = _pymol()
    tars["found"] = tars.swissprot_off.notna() | tars.proteome_off.notna() | tars.shard_off.notna()
    tars["afill"] = tars.acc.map(lambda x: os.path.isfile(afill_path(x)))
    tars["carried"] = tars.api_created.astype(str).str.startswith("2022") & (tars.api_latest == 6)

    # (1b) AlphaFill's protein against the v4 tar member, or against the served v6 file where it is carried over
    for r in tars[tars.in_sample & tars.afill & (tars.found | tars.carried)].itertuples():
        if r.found:
            path, _ = fetch(r)
            ref, what = path, "ftp_v4"
        else:
            ref, what = v6_file(r.acc), "v6_carried"
            if ref is None:
                continue
        a0, a1 = protein_atoms(afill_path(r.acc)), protein_atoms(ref)
        same = set(a0) == set(a1) and all(a0[k] == a1[k] for k in a0)
        rows.append({"check": "alphafill_atoms", "arm": what, "acc": r.acc, "identical": same,
                     "n_atoms": len(a0), "api_created": r.api_created})

    samp = tars[tars.in_sample & tars.acc.isin(single.acc)].copy()
    # accessions with a job whose query crystal is already on disk first (keeps downloads to a few)
    has_cached = single.assign(c=single.job_name.str[:4].map(lambda x: _cached_crystal(x) is not None))
    samp["cached"] = samp.acc.isin(set(has_cached.loc[has_cached.c, "acc"]))
    samp = samp.sort_values(["cached", "acc"], ascending=[False, True])
    arms = {"ftp_v4": samp[samp.found],
            "v6_carried": samp[~samp.found & samp.carried],
            "v6_repredicted": samp[samp.found & samp.api_created.astype(str).str.startswith("2025")],
            "alphafill": samp[samp.afill]}
    limits = {"ftp_v4": a.n, "v6_carried": a.n, "v6_repredicted": 10, "alphafill": a.n}
    # the jobs are chosen once (preferring a query crystal already on disk, which depends on the machine's
    # caches) and kept beside this script, so a re-run measures the same jobs
    fixed = pd.read_csv(VALID_JOBS_CSV) if os.path.isfile(VALID_JOBS_CSV) and not a.rechoose else None
    chosen = []
    for arm, sub in arms.items():
        done = 0
        if fixed is not None:
            pick = fixed[fixed.arm == arm]
            sub = sub.set_index("acc").loc[[x for x in pick.acc if x in set(sub.acc)]].reset_index()
            want = dict(zip(pick.acc, pick.job_name))
        for r in sub.itertuples():
            if done >= limits[arm]:
                break
            jobs = single[single.acc == r.acc]
            if fixed is not None:
                job = jobs[jobs.job_name == want[r.acc]].iloc[0]
            else:
                cached = jobs[jobs.job_name.str[:4].map(lambda x: _cached_crystal(x) is not None)]
                jobs = cached if len(cached) else jobs          # prefer a query crystal already on disk
                job = jobs.iloc[rng.randrange(len(jobs))]
            chosen.append({"arm": arm, "acc": r.acc, "job_name": job.job_name})
            data = None
            if arm == "ftp_v4":
                path, data = fetch(r)
            elif arm == "alphafill":
                path = afill_path(r.acc)
            else:
                path = v6_file(r.acc)
                if path is None:
                    continue
            tgz = os.path.join(ROOT, str(job.batch).zfill(3), f"{job.job_name}.tgz")
            try:
                rec = _validate_job(cmd, job.job_name, tgz, path, strip=arm == "alphafill")
            except Exception as e:  # noqa: BLE001
                rec = {"job_name": job.job_name, "status": f"error:{type(e).__name__}:{str(e)[:100]}"}
            rec.update({"check": "v2c_job", "arm": arm, "acc": r.acc, "api_created": r.api_created,
                        "source": ("swissprot" if pd.notna(r.swissprot_off) else
                                   "proteome" if pd.notna(r.proteome_off) else "shard") if arm == "ftp_v4"
                        else ("alphafill" if arm == "alphafill" else "v6"),
                        "gz_size_equals_bucket": data is not None and gcs is not None and r.acc in gcs.index
                        and int(gcs.loc[r.acc, "f1_size"]) == len(data)})
            rows.append(rec)
            done += 1
            print(f"  {arm:<15} {r.acc:<11} {job.job_name:<22} {rec.get('status')} frame "
                  f"{rec.get('frame_text_identical')} "
                  + " ".join(f"{k}={rec.get(f'same_{k}')}" for k in ("RoG", "bfac_t", "sasa", "pocket_rms", "volume")),
                  flush=True)
    _save(pd.DataFrame(rows), VALID_CSV)
    if fixed is None:
        pd.DataFrame(chosen).to_csv(VALID_JOBS_CSV, index=False)
    mb = sum(t.bytes for t in remotes.values()) / 1e6
    print(f"validation rows {len(rows)}; {mb:.1f} MB transferred for the tar extractions; query crystals "
          f"downloaded {_DOWNLOADS['n']}, not obtained {collections.Counter(w for _, w in _DOWNLOADS.get('why', []))}")


# =============================================================================== mirrors
# Third-party copies. Foldcomp (Steinegger lab, anonymous S3) stores every v4 model lossily; its Swiss-Prot
# database is keyed in name order, so one entry is found by binary search over the .lookup and .index text
# files with byte ranges. Decompression needs the `foldcomp` package, kept out of py311: point
# AHOJ_AF_FOLDCOMP_PY at a python that has it (pip install foldcomp, in a venv on a short path).
FOLDCOMP = "https://steineggerlab.s3.amazonaws.com/foldcomp/"
FOLDCOMP_PY = os.environ.get("AHOJ_AF_FOLDCOMP_PY", "")
MIRRORS_CSV = os.path.join(OUT_DIR, "foldcomp_fidelity.csv")


class RemoteText:
    """A sorted text file on an HTTP server, searched line by line with byte ranges."""

    def __init__(self, session, url):
        self.s, self.url = session, url
        self.size = int(session.head(url, timeout=60).headers["Content-Length"])
        self.requests = self.bytes = 0

    def _get(self, a, b):
        with self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, stream=True, timeout=120) as r:
            if r.status_code != 206:
                raise RuntimeError(f"{self.url}: status {r.status_code} to a ranged request")
            body = r.raw.read(b - a + 2)
        self.requests += 1
        self.bytes += len(body)
        return body

    def line_at(self, pos, chunk=512):
        """(start, text) of the first whole line starting at or after pos."""
        if pos == 0:
            data, start = self._get(0, min(self.size - 1, chunk)), 0
        else:
            data = self._get(pos - 1, min(self.size - 1, pos - 1 + chunk))
            nl = data.find(b"\n")
            while nl < 0:
                data += self._get(pos - 1 + len(data), min(self.size - 1, pos - 1 + len(data) + chunk))
                nl = data.find(b"\n")
            start, data = pos + nl, data[nl + 1:]
        if start >= self.size:
            return None, None
        end = data.find(b"\n")
        while end < 0 and start + len(data) < self.size:
            data += self._get(start + len(data), min(self.size - 1, start + len(data) + chunk))
            end = data.find(b"\n")
        return start, data[:end if end >= 0 else len(data)].decode()

    def find(self, key, keyfunc):
        lo, hi = 0, self.size
        while lo < hi:
            mid = (lo + hi) // 2
            _, line = self.line_at(mid)
            if line is None or keyfunc(line) >= key:
                hi = mid
            else:
                lo = mid + 1
        _, line = self.line_at(lo)
        return line if line is not None and keyfunc(line) == key else None


def _foldcomp_entry(session, db, name):
    lk = RemoteText(session, FOLDCOMP + db + ".lookup")
    line = lk.find(name, lambda l: l.split("\t")[1])
    if line is None:
        return None, lk.requests, lk.bytes
    key = int(line.split("\t")[0])
    ix = RemoteText(session, FOLDCOMP + db + ".index")
    iline = ix.find(key, lambda l: int(l.split("\t")[0]))
    _, off, ln = iline.split("\t")
    with session.get(FOLDCOMP + db, headers={"Range": f"bytes={int(off)}-{int(off) + int(ln) - 1}"},
                     stream=True, timeout=120) as r:
        if r.status_code != 206:
            raise RuntimeError(f"foldcomp data: status {r.status_code}")
        blob = r.raw.read(int(ln) + 1)
    return blob, lk.requests + ix.requests + 1, lk.bytes + ix.bytes + len(blob)


def _zip_names(session, url, max_cd=5_000_000):
    """Member names of a remote zip, from its central directory (read by byte range)."""
    import struct
    size = int(session.head(url, timeout=60).headers["Content-Length"])

    def rng(a, b):
        with session.get(url, headers={"Range": f"bytes={a}-{b}"}, stream=True, timeout=120) as r:
            if r.status_code != 206:
                raise RuntimeError(f"{url}: status {r.status_code}")
            return r.raw.read(b - a + 1)
    tail = rng(max(0, size - 70000), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    _, _, _, _, n_total, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    j = tail.rfind(b"PK\x06\x06")
    if j >= 0:
        (_, _, _, _, _, _, _, n_total, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", tail[j:j + 56])
    if cd_size > max_cd:
        return size, n_total, cd_size, None
    cd, p, names = rng(cd_off, cd_off + cd_size - 1), 0, []
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        fnl, exl, cml = struct.unpack("<HHH", cd[p + 28:p + 34])
        names.append(cd[p + 46:p + 46 + fnl].decode("utf-8", "replace"))
        p += 46 + fnl + exl + cml
    return size, n_total, cd_size, names


def cmd_mirrors(a):
    import requests
    s = requests.Session()
    print(f"=== third-party copies, probed {time.strftime('%Y-%m-%d')} ===")
    # --- Foldcomp: what is there
    for db in ("afdb_uniprot_v4", "afdb_swissprot_v4", "afdb_rep_v4", "afdb_uniprot_v6", "afdb_swissprot"):
        sizes = []
        for suf in ("", ".index", ".lookup"):
            h = s.head(FOLDCOMP + db + suf, timeout=60)
            sizes.append(int(h.headers["Content-Length"]) if h.status_code == 200 else None)
        print(f"  Foldcomp {db}: data {sizes[0] / 1e9 if sizes[0] else float('nan'):.1f} GB, index "
              f"{(sizes[1] or 0) / 1e9:.2f} GB, lookup {(sizes[2] or 0) / 1e9:.2f} GB")
    lst = s.get("https://steineggerlab.s3.amazonaws.com/", params={"list-type": 2, "prefix": "foldseek/archive/afdb_v4/"},
                timeout=60).text
    fs = re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", lst)
    print("  Foldseek archive afdb_v4 (anonymous S3 listing): "
          + ", ".join(f"{k.rsplit('/', 1)[-1]} {int(v) / 1e9:.1f} GB" for k, v in fs if k.endswith(".tar.gz")))
    ver = [k for k, _ in fs if k.endswith(".version")]
    if ver:
        txt = s.get("https://steineggerlab.s3.amazonaws.com/" + ver[0], timeout=60).text.strip()
        print(f"  {ver[0].rsplit('/', 1)[-1]} reads {txt!r}; current Foldseek AFDB download: "
              f"{s.get('https://foldseek.steineggerlab.workers.dev/afdb.version', timeout=60).text.strip()!r}")
    # --- Foldcomp fidelity: the Swiss-Prot v4 database against the v4 files pulled from the EBI tars
    rows = []
    v4dir = os.path.join(CACHE, "v4_ftp")
    in_sp = set(pd.read_csv(TARS_CSV).dropna(subset=["swissprot_off"]).acc)      # Swiss-Prot tar members
    have = sorted(f for f in os.listdir(v4dir) if f.endswith("_v4.cif.gz") and f.split("-")[1] in in_sp)[: a.n]
    if not os.path.isfile(FOLDCOMP_PY):
        print(f"  foldcomp python not found ({FOLDCOMP_PY}); fidelity check skipped")
    for f in have:
        acc = f.split("-")[1]
        blob, nreq, nb = _foldcomp_entry(s, "afdb_swissprot_v4", f"AF-{acc}-F1-model_v4")
        rec = {"acc": acc, "requests": nreq, "bytes": nb, "in_db": blob is not None}
        if blob is not None and os.path.isfile(FOLDCOMP_PY):
            tmp = tempfile.mkdtemp()
            try:
                fcz, pdb = os.path.join(tmp, "e.fcz"), os.path.join(tmp, "e.pdb")
                open(fcz, "wb").write(blob)
                subprocess.run([FOLDCOMP_PY, "-c", "import foldcomp,sys; n,p=foldcomp.decompress(open(sys.argv[1],'rb').read());"
                                "open(sys.argv[2],'w').write(p)", fcz, pdb], check=True, capture_output=True, timeout=120)
                a4, af = protein_atoms(os.path.join(v4dir, f)), protein_atoms(pdb)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            k4 = {(k[1], k[2], k[3]): v for k, v in a4.items()}
            kf = {(k[1], k[2], k[3]): v for k, v in af.items()}
            common = sorted(set(k4) & set(kf))
            d = np.array([np.linalg.norm(np.subtract(k4[k][:3], kf[k][:3])) for k in common])
            bb = np.array([k[2] in ("N", "CA", "C") for k in common])
            ca = [k for k in common if k[2] == "CA"]
            dp = np.array([abs(k4[k][3] - kf[k][3]) for k in ca])
            rec.update(n_v4=len(k4), n_fc=len(kf), n_common=len(common), rmsd_all=float(np.sqrt((d ** 2).mean())),
                       max_dev=float(d.max()), rmsd_bb=float(np.sqrt((d[bb] ** 2).mean())),
                       plddt_mean_abs=float(dp.mean()), plddt_max_abs=float(dp.max()))
        rows.append(rec)
    fd = pd.DataFrame(rows)
    _save(fd, MIRRORS_CSV)
    ok = fd[fd["n_common"].notna()] if "n_common" in fd else fd.iloc[0:0]
    print(f"  Foldcomp afdb_swissprot_v4 against {len(fd)} v4 files from the EBI tars: in the database "
          f"{int(fd.in_db.sum())}; lookup by byte range {fd.requests.median():.0f} requests and "
          f"{fd.bytes.median() / 1e3:.0f} kB per entry (median)")
    if len(ok):
        print(f"    atoms v4/Foldcomp equal in {int((ok.n_v4 == ok.n_fc).sum())} of {len(ok)} (all common in "
              f"{int((ok.n_common == ok.n_v4).sum())}); all-atom RMSD median {ok.rmsd_all.median():.3f} A "
              f"(range {ok.rmsd_all.min():.3f}-{ok.rmsd_all.max():.3f}); backbone N/CA/C RMSD median "
              f"{ok.rmsd_bb.median():.3f} A (max {ok.rmsd_bb.max():.3f}); largest single-atom deviation "
              f"{ok.max_dev.max():.2f} A; per-residue pLDDT |d| mean {ok.plddt_mean_abs.mean():.3f}, max "
              f"{ok.plddt_max_abs.max():.2f}")
        w = ok.loc[ok.rmsd_all.idxmax()]
        print(f"    worst model {w.acc} ({int(w.n_v4):,} atoms; largest tested {int(ok.n_v4.max()):,}): all-atom "
              f"{w.rmsd_all:.3f} A, backbone {w.rmsd_bb:.3f} A; rank correlation of RMSD with atom count "
              f"{ok.n_v4.rank().corr(ok.rmsd_all.rank()):+.2f} over the {len(ok)}")
    # --- AlphaFill as served today
    for ident in ("AF-P62993-F1-filled_v4", "AF-P62993-F1-filled_v6", "AF-P62993-F1-filled_v3"):
        with s.get(f"https://alphafill.eu/v1/aff/{ident}", stream=True, timeout=120) as r:
            code = r.status_code
        print(f"  alphafill.eu /v1/aff/{ident}: HTTP {code}")
    with s.get("https://alphafill.eu/v1/aff/P62993/json", stream=True, timeout=120) as r:
        head = r.raw.read(4000, decode_content=True).decode("utf-8", "replace")
    m = re.search(r'"file"\s*:\s*"([^"]+)"', head)
    v = re.search(r'"alphafill_version"\s*:\s*"([^"]+)"', head)
    print(f"  alphafill.eu P62993 JSON: alphafill {v.group(1) if v else '?'}, built from "
          f"{m.group(1).replace(chr(92) + '/', '/') if m else '?'}")
    # --- 3D-Beacons
    j = s.get("https://www.ebi.ac.uk/pdbe/pdbe-kb/3dbeacons/api/v2/uniprot/summary/P62993.json", timeout=120).json()
    urls = sorted({x["summary"]["model_url"] for x in j.get("structures", [])
                   if x["summary"].get("provider") == "AlphaFold DB"})
    print(f"  3D-Beacons P62993, AlphaFold DB model URLs: {[u.rsplit('/', 1)[-1] for u in urls]}")
    # --- PLINDER predicted monomers, and our coverage
    size, n, cd, names = _zip_names(s, "https://storage.googleapis.com/plinder/2024-06/v2/linked_structures/pred.zip")
    pl = {nm.rsplit("/", 1)[-1].split("_")[0] for nm in (names or []) if nm.endswith(".cif")}
    sets = pd.read_csv(SETS_CSV)
    print(f"  PLINDER 2024-06/v2 linked_structures/pred.zip: {size / 1e9:.2f} GB, {n:,} zip entries, "
          f"{len(pl):,} predicted monomers; covers " + ", ".join(
              f"{name} {int(sets.loc[sets[col], 'acc'].isin(pl).sum()):,} of {int(sets[col].sum()):,}"
              for name, col, _ in SETS[:1] + SETS[3:4]))
    size2 = int(s.head("https://pinderdata.org/2024-02/pdbs.zip", timeout=60).headers["Content-Length"])
    print(f"  PINDER 2024-02 pdbs.zip: {size2 / 1e9:.1f} GB (af__<UniProt>.pdb members among the PDB entries)")
    # --- Wayback Machine captures of the EBI v4 tars
    cdx = s.get("http://web.archive.org/cdx/search/cdx", timeout=180,
                params={"url": "ftp.ebi.ac.uk/pub/databases/alphafold/v4/", "matchType": "prefix",
                        "filter": "statuscode:200", "fl": "timestamp,original", "output": "json"}).json()
    tars = {}
    for ts, orig in cdx[1:]:
        if orig.endswith("_v4.tar"):
            tars.setdefault(orig.rsplit("/", 1)[-1], []).append(ts)
    prot = [t for t in tars if t.startswith("UP")]
    gone = sorted(set(prot) - set(_ftp_v4_proteomes().values()))
    print(f"  Wayback Machine, ftp.ebi.ac.uk/.../alphafold/v4/ captured with status 200: {len(prot)} proteome tars, "
          f"swissprot_cif_v4.tar {'yes' if 'swissprot_cif_v4.tar' in tars else 'no'}, mane_overlap_v4.tar "
          f"{'yes' if 'mane_overlap_v4.tar' in tars else 'no'}; capture dates "
          f"{min(min(v) for v in tars.values())[:8]}-{max(max(v) for v in tars.values())[:8]}; captured but no longer "
          f"on the EBI FTP: {gone}")


# =============================================================================== report
def _live_probes():
    import requests
    s = requests.Session()
    print(f"=== live probes, {time.strftime('%Y-%m-%d')} ===")
    # --- per-model URLs and the API
    local = sorted(re.match(r"AF-(\S+)-F1-model_v4", f).group(1) for f in os.listdir(V4_LOCAL) if "_v4." in f)
    st = collections.Counter()
    for acc in local:
        for suf in ("model_v4.cif", "model_v4.pdb", "model_v6.cif"):
            r = s.head(f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-{suf}", timeout=60, allow_redirects=True)
            st[(suf, r.status_code)] += 1
    print(f"  per-model files for the {len(local)} accessions with a local v4 copy: "
          + ", ".join(f"{k[0]} {k[1]}: {n}" for k, n in sorted(st.items())))
    acc = "P12931"
    old = {v: s.head(f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{v}.cif", timeout=60).status_code
           for v in (1, 2, 3, 4, 5, 6)}
    r = s.get(f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.cif", timeout=60)
    code = re.search(r"<Code>(\w+)</Code>", r.text)
    print(f"  {acc} model_v1..v6.cif status {old}; the v4 404 is a Cloud Storage '{code.group(1) if code else '?'}' "
          f"(x-goog/UploadServer backend: {'X-GUploader-UploadID' in r.headers})")
    a0 = s.get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=60).json()
    a4 = s.get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", params={"version": 4}, timeout=60).json()
    f1 = [x for x in a0 if x.get("entryId") == f"AF-{acc}-F1"][0]
    spec = s.get("https://alphafold.ebi.ac.uk/api/openapi.json", timeout=60).json()
    pp = [p.get("name") for p in spec["paths"]["/prediction/{qualifier}"]["get"].get("parameters", [])]
    print(f"  API {acc}: latestVersion {f1['latestVersion']}, allVersions {f1['allVersions']}, modelCreatedDate "
          f"{f1['modelCreatedDate'][:10]}, cifUrl ends {f1['cifUrl'].rsplit('-', 1)[-1]}; '?version=4' returns the "
          f"same payload: {a0 == a4}; /prediction parameters {pp}")
    # --- EBI FTP
    html = s.get(FTP + "v4/", timeout=60).text
    tars4 = sorted(set(re.findall(r'href="([^"]+_v4\.tar)"', html)))
    prot = [t for t in tars4 if t.startswith("UP")]
    heads = {t: s.head(FTP + "v4/" + t, timeout=60) for t in tars4}
    size = {t: int(h.headers["Content-Length"]) for t, h in heads.items()}
    dates = collections.Counter(h.headers.get("Last-Modified", "")[5:16] for h in heads.values())
    print(f"  EBI FTP v4/: {len(prot)} proteome tars ({sum(size[t] for t in prot) / 1e9:.1f} GB), "
          f"swissprot_cif_v4.tar {size['swissprot_cif_v4.tar']:,} B ({size['swissprot_cif_v4.tar'] / 1e9:.1f} GB), "
          f"swissprot_pdb_v4.tar {size['swissprot_pdb_v4.tar'] / 1e9:.1f} GB, mane_overlap_v4.tar "
          f"{size['mane_overlap_v4.tar'] / 1e9:.1f} GB; Last-Modified {dict(dates)}")
    sh = pd.read_csv(SHARDS_CSV) if os.path.isfile(SHARDS_CSV) else None
    if sh is not None:
        print(f"  EBI FTP v4/updated_entries/: {len(sh)} shards, {sh.bytes.sum() / 1e9:.0f} GB, first members "
              f"{sh['first'].iloc[0]} ... {sh['first'].iloc[-1]} (globally sorted: {sh['first'].is_monotonic_increasing})")
    t = RemoteTar(s, FTP + "v4/swissprot_cif_v4.tar")
    t.hdr.pop(0, None)                                       # re-read: the cache keeps no mtime
    ti = t.block(0)
    print(f"  swissprot_cif_v4.tar first member {ti.name}, member mtime {time.strftime('%Y-%m-%d', time.gmtime(ti.mtime))}")
    # byte identity with the bucket: the first member of update shard 0, and the Swiss-Prot tar's first member
    for url in (FTP + "v4/updated_entries/v4_updated_entries-0.tar", FTP + "v4/swissprot_cif_v4.tar"):
        tt = RemoteTar(s, url)
        tt.hdr.pop(0, None)
        m0 = tt.block(0)
        data = tt.member(0, m0)
        obj = m0.name[:-3]
        item = [i for i in _gcs_list(s, obj, "items(name,size,md5Hash)") if i["name"] == obj][0]
        md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
        print(f"  {url.rsplit('/', 1)[-1]} member {m0.name}: {len(data):,} B, bucket object {int(item['size']):,} B; "
              f"md5 equal {md5 == item['md5Hash']}; gzip mtime {time.strftime('%Y-%m-%d', time.gmtime(int.from_bytes(data[4:8], 'little')))}")
    with s.get(FTP + "v4/swissprot_cif_v4.tar", stream=True, timeout=60,
               headers={"Range": "bytes=" + ",".join(f"{o + 257}-{o + 261}" for o in range(0, 512 * 201, 512))}) as r:
        over = r.status_code
    with s.get(FTP + "v4/swissprot_cif_v4.tar", stream=True, timeout=60,
               headers={"Range": "bytes=" + ",".join(f"{o + 257}-{o + 261}" for o in range(0, 512 * 200, 512))}) as r:
        at = r.status_code
    print(f"  multi-range requests: 200 ranges -> {at}; 201 ranges -> {over} (the whole file; abandoned unread)")
    cl = s.get(FTP + "CHANGELOG.txt", timeout=60).text
    block = cl.split("## ")[1]
    keep = [ln.strip("* ").strip() for ln in block.splitlines() if re.search(r"structures|Added:|Removed:|Changed:|Unchanged:", ln)]
    print(f"  CHANGELOG {block.splitlines()[0].strip()}: " + "; ".join(keep))
    upd = re.search(r"The list of ([\d,]+) accessions with updated coordinates", cl)
    print(f"  CHANGELOG 2022-11-01 (v4): {upd.group(1) if upd else '?'} accessions with updated coordinates, "
          f"in v4/updated_entries/")
    # --- Google Cloud
    perms = ["storage.objects.get", "storage.objects.list", "storage.buckets.get"]
    for b in (BUCKET, "public-datasets-deepmind-alphafold"):
        r = s.get(GCS_API + b + "/iam/testPermissions", params={"permissions": perms}, timeout=60).json()
        print(f"  gs://{b}: anonymous permissions {r.get('permissions', [])}")
    meta = s.get(GCS_API + BUCKET, timeout=60).json()
    print(f"  gs://{BUCKET}: created {meta['timeCreated'][:10]}, updated {meta['updated'][:10]}, location "
          f"{meta['location']}, uniform access {meta['iamConfiguration']['uniformBucketLevelAccess']['enabled']}, "
          f"requester-pays flag present: {'billing' in meta}")
    r = s.get(f"https://storage.googleapis.com/{BUCKET}/AF-{acc}-F1-model_v4.cif", timeout=60)
    print(f"  anonymous object download: {r.status_code} "
          f"({re.search(r'<Details>(.{0,60})', r.text).group(1) if '<Details>' in r.text else r.text[:60]})")
    top = s.get(GCS_API + BUCKET + "/o", params={"startOffset": "AG", "delimiter": "/",
                                                  "fields": "prefixes,items(name,size)"}, timeout=60).json()
    print(f"  bucket top level besides AF-*: prefixes {top.get('prefixes')}, objects "
          + ", ".join(f"{i['name']} {int(i['size']) / 1e9:.1f} GB" for i in top.get("items", [])))
    for b in ("v3", "v5", "v6"):
        code = s.get(GCS_API + f"public-datasets-deepmind-alphafold-{b}", timeout=60).status_code
        print(f"  gs://public-datasets-deepmind-alphafold-{b}: HTTP {code}")
    c = s.get("https://api.github.com/repos/google-deepmind/alphafold/commits",
              params={"path": "afdb/README.md", "per_page": 1}, timeout=60).json()
    readme = s.get("https://raw.githubusercontent.com/google-deepmind/alphafold/main/afdb/README.md", timeout=60).text
    need = re.search(r"Google Cloud account is required for the download", readme.replace("\n", " "))
    shards = re.search(r"([\d,]+)\s+sharded proteome tar files", readme.replace("\n", " "))
    print(f"  DeepMind afdb/README.md (last commit {c[0]['commit']['committer']['date'][:10]}): states a Google Cloud "
          f"account is required: {bool(need)}; still calls v4 the latest: {'latest version (v4)' in readme}; "
          f"proteome shards in the bucket: {shards.group(1) if shards else '?'}")


def _run_provenance(scan, n=60):
    """Where v2.1.0 kept its models: the job log's 'AlphaFold directory', from a seeded sample of archives."""
    rows = []
    for r in scan.sample(n=n, random_state=SEED).itertuples():
        tgz = os.path.join(ROOT, str(r.batch).zfill(3), f"{r.job_name}.tgz")
        rec = {}
        with tarfile.open(tgz, "r|gz") as tar:
            for m in tar:
                if m.name == "./stdout_log":
                    log_ = tar.extractfile(m).read().decode("utf-8", "replace")
                    d = re.search(r"AlphaFold directory:\s*(\S+)", log_)
                    rec["dir"] = d.group(1) if d else None
                    rec["fresh"] = len(re.findall(r"model_v\d\.cif\.gz downloaded and compressed", log_))
                    rec["cached"] = len(re.findall(r"model_v\d\.cif\.gz already downloaded", log_))
                elif m.name == "./script_parameters.json":
                    j = json.loads(tar.extractfile(m).read())
                    rec.update(version=j.get("script_version"), afdb=j.get("afdb_version"), date=j.get("date", "")[:10])
                    break
        rows.append(rec)
    d = pd.DataFrame(rows)
    print(f"  job logs of {len(d)} random archives: AlphaFold directory {d['dir'].value_counts().to_dict()}; "
          f"model fetched in the job {int((d.fresh > 0).sum())}, already in the cache {int((d.cached > 0).sum())}; "
          f"AHoJ {d.version.value_counts().to_dict()}, afdb_version {d.afdb.value_counts().to_dict()}, "
          f"run dates {d.date.min()} to {d.date.max()}")


def cmd_report(a):
    if not a.offline:
        _live_probes()
    print()
    scan = pd.read_csv(SCAN_CSV, low_memory=False)
    summarise_scan(scan)
    _run_provenance(scan)
    print()
    sets = pd.read_csv(SETS_CSV)
    print("=== accession sets ===")
    summarise_sets(sets)

    # ---- the v6 headers: which served models are carried over from v4
    h = pd.read_csv(V6HDR_CSV)
    print(f"\n=== local v6 models ({len(h):,} files in AFDB_structures\\v6) ===")
    print(f"  model made (recvd_initial_deposition_date): {h.deposited.value_counts().to_dict()}")
    xt = pd.crosstab(h.first_rev, h.deposited)
    for fr, row in xt.iterrows():
        print(f"  first released in v{fr}: " + ", ".join(f"made {k} {int(n):,}" for k, n in row.items() if n))
    ref = os.path.join(WORK, "af_centric_rerun", "model_row_refresh_v6.csv")
    if os.path.isfile(ref):
        r = pd.read_csv(ref)
        r = r[r.status == "ok"].merge(h[["acc", "deposited"]], on="acc", how="left")
        r["same"] = r.frame_dev.fillna(1) < 1e-9
        xt = pd.crosstab(r.deposited.fillna("no local v6"), r.same)
        print(f"  AF_CENTRIC_RERUN.md model-row refresh ({len(r):,} rows): v6 US-align frame bit-identical to the "
              f"archived v4 frame, by the v6 model's date: "
              + "; ".join(f"{d} {int(row.get(True, 0))} of {int(row.sum())}" for d, row in xt.iterrows()))
    ref = os.path.join(WORK, "af_centric_rerun", "afdb_v4_v6_model_pocket_check.csv")
    if os.path.isfile(ref):
        p = pd.read_csv(ref)
        p = p[p.status == "ok"].merge(h[["acc", "deposited"]], on="acc", how="left")
        p["same"] = ((p.rog_v6 - p.rog_v4).abs() <= 0.011) & \
                    ((p.bfac_t_v6 - p.bfac_t_v4).abs() / p.bfac_t_v4.abs().clip(lower=1e-9) <= 1e-3)
        xt = pd.crosstab(p.deposited.fillna("no local v6"), p.same)
        print(f"  AF_CENTRIC_RERUN.md 600-pocket check (RoG to 0.01 A and pLDDT sum to 0.1%, {len(p)} rows): "
              + "; ".join(f"{d} passes {int(row.get(True, 0))} of {int(row.sum())}" for d, row in xt.iterrows())
              + f"; bit-identical models {int(p.deposited.astype(str).str.startswith('2022').sum())} of {len(p)} "
              f"({_pct(p.deposited.astype(str).str.startswith('2022').sum(), len(p)):.1f}%)")

    # ---- the two labels per set, and the UniProt rules against the exact labels
    v, u = labelled()
    ex = v[v.label_source == "v6_header"]
    rp_true = ~ex.deposited.astype(str).str.startswith("2022")
    rp_pred = ex.pred_repred
    print("\n=== which models are served unchanged as v6, and which the EBI tars should hold ===")
    print(f"  UniProt {u.uniprot_release.iloc[0]}: {len(u):,} of {len(v):,} accessions active")
    print(f"  re-prediction rule (in Swiss-Prot or an AFDB v6 proteome at UniProt 2025_03) against {len(ex):,} local "
          f"v6 headers: re-made and predicted {int((rp_true & rp_pred).sum()):,}, re-made but missed "
          f"{int((rp_true & ~rp_pred).sum()):,}, carried over and predicted {int((~rp_true & ~rp_pred).sum()):,}, "
          f"carried over but called re-made {int((~rp_true & rp_pred).sum()):,}; agreement "
          f"{_pct(int((rp_true == rp_pred).sum()), len(ex)):.1f}%")
    print(f"  of the same {len(ex):,}: predicted in a tar {int(ex.tar.sum()):,} (v6 re-made "
          f"{int((ex.tar & rp_true).sum()):,}); v6 re-made but predicted in no tar {int((~ex.tar & rp_true).sum()):,}")
    for name, col, _ in SETS:
        sub = v[v[col]]
        r = sub.groupby(["route", "label_source"]).size().to_dict()
        print(f"  {name:<16} {len(sub):>7,}: " + ", ".join(f"{k[0]} ({k[1]}) {n:,}" for k, n in sorted(r.items())))
    ds = v[v.in_detection_sites]
    print(f"  detection sites' models: served v6 is the v4 model {int(ds.carried.sum()):,} "
          f"({_pct(ds.carried.sum(), len(ds)):.1f}%), re-made in 2025 {int((~ds.carried).sum()):,} "
          f"({_pct((~ds.carried).sum(), len(ds)):.1f}%) of {len(ds):,}")
    # the v6 model-row refresh (the AF-centric re-run prototype) only has work where the model was re-made
    uni = pd.read_csv(UNI, usecols=["threshold", "job_names"], low_memory=False)
    ent = uni.loc[uni.threshold == "UNCLUSTERED", "job_names"].astype(str)
    scan = pd.read_csv(SCAN_CSV, usecols=["job_name", "n_af_sel", "af_models", "af_mat_models"], low_memory=False)
    scan["models"] = job_models(scan)
    job_accs = dict(zip(scan.job_name, scan.models.map(lambda s: set(_accs(s)))))
    carried = set(v.loc[v.carried, "acc"])
    accs = ent.map(lambda j: job_accs.get(j, set()))
    keep = accs.map(lambda s: bool(s) and s <= carried)
    print(f"  analysis entries whose every model is served unchanged as v6 (no refresh needed): {int(keep.sum()):,} of "
          f"{len(ent):,} ({_pct(keep.sum(), len(ent)):.1f}%); entries with a re-made or unlabelled model "
          f"{int((~keep).sum()):,}")
    if os.path.isfile(STORE_CHECK_CSV):
        # the v4 store allows the exact test: the local v6 file's atoms against the v4 file's (store verify)
        g = pd.read_csv(STORE_CHECK_CSV)
        g = g[g.check == "v6_atoms"]
        tested = set(g.acc)
        carried_x = (carried - tested) | set(g.loc[g.identical.astype(bool), "acc"])
        ds_x = ds.acc.isin(carried_x)
        keep_x = accs.map(lambda s: bool(s) and s <= carried_x)
        print(f"  with the exact test instead of the date ({len(tested):,} accessions with a local v6 file; carried-over "
              f"labels withdrawn {len(carried - carried_x)}, added {len(carried_x - carried)}): detection sites' models "
              f"carried over {int(ds_x.sum()):,} ({_pct(ds_x.sum(), len(ds)):.1f}%); analysis entries needing no refresh "
              f"{int(keep_x.sum()):,} ({_pct(keep_x.sum(), len(ent)):.1f}%)")

    # ---- the Google Cloud v4 bucket, listed per accession
    if os.path.isfile(GCS_CSV):
        g = pd.read_csv(GCS_CSV)
        v = v.merge(g, on="acc", how="left")
        print("\n=== gs://public-datasets-deepmind-alphafold-v4, listed per accession ===")
        for name, col, _ in SETS:
            sub = v[v[col]]
            has = sub.n_frag.fillna(0) > 0
            miss = sorted(sub.loc[~has, "acc"])
            print(f"  {name:<16} F1 model_v4.cif listed for {int(has.sum()):,} of {len(sub):,} "
                  f"({_pct(has.sum(), len(sub)):.2f}%); {pd.to_numeric(sub.f1_size, errors='coerce').sum() / 1e9:.2f} GB "
                  f"gzip; missing: {', '.join(miss[:8])}{' ...' if len(miss) > 8 else ''}")

    # ---- the author's AlphaFill 2.x databank
    if os.path.isfile(AFILL_CSV):
        af = pd.read_csv(AFILL_CSV)
        v = v.merge(af[["acc", "afill"]], on="acc", how="left")
        v["afill"] = v.afill.eq(True)
        src = af.afill_file.dropna().str.extract(r"^(/DATA/[^/]+)")[0].value_counts().to_dict()
        print(f"\n=== local AlphaFill databank ({ALPHAFILL}) ===")
        root = os.path.dirname(ALPHAFILL)
        tar_, log_ = os.path.join(root, "alphafill.tar"), os.path.join(root, "alphafill.log")
        if os.path.isfile(tar_) and os.path.isfile(log_):
            with open(log_, "rb") as fh:
                n_models = sum(1 for ln in fh if ln.rstrip().endswith(b"-filled_v4.cif.gz"))
            print(f"  alphafill.tar {os.path.getsize(tar_) / 1e9:.1f} GB; its listing (alphafill.log) names "
                  f"{n_models:,} filled_v4 models")
        print(f"  versions {af.afill_alphafill_version.value_counts().to_dict()}; source file named in the JSON "
              f"head: {src}")
        for name, col, _ in SETS:
            sub = v[v[col]]
            print(f"  {name:<16} filled_v4 model present for {int(sub.afill.sum()):,} of {len(sub):,} "
                  f"({_pct(sub.afill.sum(), len(sub)):.1f}%); by route "
                  + ", ".join(f"{k} {int(g2.afill.sum()):,}/{len(g2):,}" for k, g2 in sub.groupby("route")))

    # ---- the EBI tars, sampled
    rates = {}
    if os.path.isfile(TARS_CSV):
        t = pd.read_csv(TARS_CSV)
        for c in ("proteome_tar", "proteome_off", "shard", "shard_off"):
            if c not in t:
                t[c] = np.nan
        st = json.load(open(TARS_CSV.replace(".csv", "_stats.json")))
        samp = pd.read_csv(SAMPLE_CSV)
        t = t.merge(samp[["acc", "route", "label_source"]], on="acc", how="left")
        t = t.merge(v[["acc", "carried", "afill"]], on="acc", how="left")
        t["in_tar"] = t.swissprot_off.notna() | t.proteome_off.notna()
        t["shard_only"] = t.shard_off.notna() & ~t.in_tar
        made = t.api_created.astype(str)
        t["api_carried"] = made.str.startswith("2022") & (t.api_latest == 6)
        # the label a re-backing would rely on: the v6 header where one exists, else the API
        t["same"] = np.where(t.label_source == "v6_header", t.carried.eq(True), t.api_carried)
        t["anon"] = t.in_tar | t.shard_only | t.same
        print(f"\n=== EBI FTP v4 tars, byte-range search ({st['date']}) ===")
        print(f"  {st['accessions']} accessions ({int(t.in_sample.sum())} sampled + the local v4 copies); "
              f"{st['requests']:,} ranged requests, {st['bytes'] / 1e6:.1f} MB ({st['bytes'] / st['accessions'] / 1e6:.1f} MB "
              f"per accession), sort-order violations {st['order_violations']}, update shards opened {st['shards_opened']}")
        if "order_violations_first_run" in st:
            print(f"  the first run compared whole names byte-wise and met {st['order_violations_first_run']} order "
                  f"violations inside the update shards; the repeated shard search found the same "
                  f"{st['shard_hits_first_run']} hits")
        sm = t[t.in_sample]
        for key, g2 in sm.groupby(["route", "label_source"]):
            k, c, n, an = int(g2.in_tar.sum()), int(g2.same.sum()), len(g2), int(g2.anon.sum())
            rates[key] = (an, n)
            lo, hi = _wilson(k, n)
            lo2, hi2 = _wilson(an, n)
            api = g2.api_created.astype(str).str[:4].value_counts().to_dict()
            print(f"  {key[0]} ({key[1]}), n={n}: in a tar {k} ({_pct(k, n):.0f}% [{lo:.0f}, {hi:.0f}]; Swiss-Prot "
                  f"{int(g2.swissprot_off.notna().sum())}, proteome {int(g2.proteome_off.notna().sum())}), update shard "
                  f"only {int(g2.shard_only.sum())}; v6 is the v4 model {c}; any anonymous source {an} "
                  f"({_pct(an, n):.0f}% [{lo2:.0f}, {hi2:.0f}]); API v6 made {api}, no AFDB entry "
                  f"{int((g2.api_status.astype(str) != '200').sum())}")
        pt, ft = sm.route.eq("tar"), sm.in_tar
        print(f"  tar prediction on the sample: predicted and held {int((pt & ft).sum())}, predicted but absent "
              f"{int((pt & ~ft).sum())}, not predicted but held {int((~pt & ft).sum())}, neither {int((~pt & ~ft).sum())}")
        remade = sm.api_created.astype(str).str.startswith("2025")
        print(f"  sampled accessions held by a tar whose v6 was re-made: {int((ft & remade).sum())} of {int(ft.sum())}; "
              f"API-carried among those held by no tar: {int((~ft & sm.api_carried).sum())} of {int((~ft).sum())}")
        loc = t[t.local_v4]
        print(f"  local v4 copies: in a tar {int(loc.in_tar.sum())} of {len(loc)}; the others' v6 made "
              f"{sorted(loc.loc[~loc.in_tar, 'api_created'].astype(str).unique())}")

    # ---- validation
    if os.path.isfile(VALID_CSV):
        d = pd.read_csv(VALID_CSV)
        print("\n=== validation ===")
        lv = d[d.check == "local_v4"]
        print(f"  local v4 files vs the tar member: decompressed content identical {int(lv.identical.sum())} of "
              f"{len(lv)}; gzip size equals the bucket listing {int(lv.gz_size_equals_bucket.sum())} of {len(lv)}")
        l6 = d[d.check == "local_v4_vs_v6"]
        if len(l6):
            print(f"  local v4 files held by no tar vs the served v6 file: every atom's coordinates and B-factor "
                  f"identical {int(l6.identical.sum())} of {len(l6)} (v6 made "
                  f"{sorted(l6.api_created.astype(str).unique())})")
        aa = d[d.check == "alphafill_atoms"]
        for arm, g2 in aa.groupby("arm"):
            print(f"  AlphaFill protein atoms vs the {arm} file: coordinates and B-factors identical "
                  f"{int(g2.identical.astype(bool).sum())} of {len(g2)}")
        j = d[d.check == "v2c_job"]
        for arm, g2 in j.groupby("arm"):
            ok = g2[g2.status == "ok"]
            same = ok[[c for c in ok if c.startswith("same_")]].astype(bool)
            print(f"  {arm:<15} {len(g2)} jobs ({len(ok)} ran; sources {ok.source.value_counts().to_dict()}): "
                  f"archived US-align report reproduced verbatim {int(ok.frame_text_identical.astype(bool).sum())}, "
                  f"all five model-row values equal {int(same.all(1).sum())} "
                  f"({', '.join(f'{c[5:]} {int(same[c].sum())}' for c in same)})")

    # ---- every v4 model of a set: which source
    if rates and os.path.isfile(GCS_CSV) and "afill" in v:
        print("\n=== every v4 model of a set: where it can come from ===")
        for name, col, _ in SETS:
            sub = v[v[col]]
            est = est_af = 0.0
            parts = []
            for key, g2 in sub.groupby(["route", "label_source"]):
                k, n = rates.get(key, (0, 0))
                exact = key == ("v6_same", "v6_header")      # the header says the served v6 is the v4 model
                p = 1.0 if exact else (k / n if n else 0.0)
                est += len(g2) * p
                est_af += g2.afill.sum() + (~g2.afill).sum() * p
                parts.append(f"{key[0]}/{key[1]} {len(g2):,} x {'1 (exact)' if exact else f'{k}/{n}'}")
            gone = int((sub.n_frag.fillna(0) == 0).sum())
            print(f"  {name:<16} {len(sub):>7,}: " + "; ".join(parts)
                  + f" -> anonymous (EBI tars or served v6) about {est:,.0f} ({_pct(est, len(sub)):.1f}%); with the "
                  f"local AlphaFill models about {est_af:,.0f} ({_pct(est_af, len(sub)):.1f}%); in no v4 listing {gone}")
        import requests
        s = requests.Session()
        by_up = {n.split("_")[0]: n for n in _ftp_v4_proteomes().values()}
        need = v[v.in_v2c & v.pred_prot & ~v.pred_sp]
        ups = need.proteomes.fillna("").str.extract(r"(UP\d+)")[0].value_counts()
        sizes = {up: int(s.head(FTP + "v4/" + by_up[up], timeout=60).headers["Content-Length"]) for up in ups.index}
        sp_size = int(s.head(FTP + "v4/swissprot_cif_v4.tar", timeout=60).headers["Content-Length"])
        print(f"  bulk route for the v2c set: swissprot_cif_v4.tar {sp_size / 1e9:.1f} GB for "
              f"{int((v.in_v2c & v.pred_sp).sum()):,} predicted members, plus {len(sizes)} proteome tars "
              f"({sum(sizes.values()) / 1e9:.1f} GB) for {len(need):,} more; largest "
              + ", ".join(f"{by_up[up].split('_')[2]} {ups[up]:,}" for up in list(ups.index)[:4]))
        gall = pd.to_numeric(v.loc[v.in_v2c, "f1_size"], errors="coerce").sum()
        print(f"  the bucket route for the whole v2c set: {int((v.in_v2c & (v.n_frag.fillna(0) > 0)).sum()):,} F1 files, "
              f"{gall / 1e9:.2f} GB gzip; at a list price of $0.12 per GiB of internet egress "
              f"${0.12 * gall / 2 ** 30:.2f}")

    # ---- the v2c model store, copied from the bucket (store list / fetch / verify; validate --arm store)
    if os.path.isfile(STORE_CSV):
        summarise_store(v)


# =============================================================================== main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("v2c_scan")
    q.add_argument("--procs", type=int, default=8)
    sub.add_parser("sets")
    sub.add_parser("uniprot")
    q = sub.add_parser("alphafill")
    q.add_argument("--threads", type=int, default=16)
    q = sub.add_parser("v6hdr")
    q.add_argument("--procs", type=int, default=8)
    q = sub.add_parser("gcs")
    q.add_argument("--threads", type=int, default=16)
    q = sub.add_parser("tars")
    q.add_argument("--n", type=int, default=126)
    q.add_argument("--resample", action="store_true")
    q.add_argument("--redo-shards", action="store_true", help="repeat only the update-shard searches")
    q = sub.add_parser("store", help="the v2c model store: list the bucket objects, copy them (the author's gcloud "
                                     "login), verify the copies")
    q.add_argument("action", choices=["list", "fetch", "verify"])
    q.add_argument("--threads", type=int, default=16, help="list: listing threads")
    q.add_argument("--passes", type=int, default=3, help="fetch: copy passes, each retrying what the last missed")
    q.add_argument("--limit", type=int, default=0, help="fetch: only this many accessions (a pilot)")
    q.add_argument("--workers", type=int, default=12, help="fetch: parallel gcloud runs per bucket")
    q.add_argument("--procs", type=int, default=8, help="verify: worker processes")
    q = sub.add_parser("validate")
    q.add_argument("--arm", choices=["sources", "store"], default="sources",
                   help="sources: the tar, v6 and AlphaFill arms; store: the recovered store")
    q.add_argument("--n", type=int, default=20)
    q.add_argument("--max-downloads", type=int, default=20,
                   help="store arm: accessions that may take a job whose query crystal must be fetched")
    q.add_argument("--rechoose", action="store_true", help="choose the validation jobs afresh")
    q = sub.add_parser("mirrors")
    q.add_argument("--n", type=int, default=16, help="v4 files to test against Foldcomp")
    q = sub.add_parser("report")
    q.add_argument("--offline", action="store_true", help="skip the live HTTP probes")
    a = p.parse_args()
    store = {"list": cmd_store_list, "fetch": cmd_store_fetch, "verify": cmd_store_verify}
    {"v2c_scan": cmd_v2c_scan, "sets": cmd_sets, "uniprot": cmd_uniprot, "alphafill": cmd_alphafill,
     "v6hdr": cmd_v6hdr, "gcs": cmd_gcs, "store": lambda a: store[a.action](a),
     "tars": cmd_tars, "validate": cmd_validate, "mirrors": cmd_mirrors, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
