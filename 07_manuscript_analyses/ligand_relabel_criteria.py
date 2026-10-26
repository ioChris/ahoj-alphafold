# -*- coding: utf-8 -*-
"""Which single-atom ligand exclusion list did the published classification actually use?
Backs Supplementary Data 1 (the 106- and 123-code lists) and the Methods section "Apo/holo relabelling of
candidate structures" (relabelled candidates, entries touched, the query-ion rule, the heavy_le1 counterfactual).

Two built-in lists live side by side in the classification code
(00_classification/kde_classifier/parsing_functions_DBv2.py, under the comment
"Hard-coded list of excluded single-atom ligands (heavy atoms or total atoms)"):

    EXCLUDED_LIGANDS_TOTAL_LE1   ligands with <= 1 atom in total          (106 codes)
    EXCLUDED_LIGANDS_HEAVY_LE1   ligands with <= 1 non-hydrogen atom      (123 codes)

TOTAL_LE1 is a strict subset of HEAVY_LE1; the 17 extra codes in HEAVY_LE1 are polyatomic species
with a single heavy atom (water and its isotopologues, hydroxide, ammonia/ammonium, methyl/methylene,
H2S and a few one-heavy-atom fragments).

`relabel_apoholo_by_ions(...)` takes `builtin=` and DEFAULTS to "heavy_le1", which is the trap: the
default is not what ran.  Every caller in the batch pipeline passes `builtin="total_le1"` explicitly,
and has done so since the relabel was introduced.  This script states that provenance from the
repository itself and then measures what the choice is worth on the published data.

    python ligand_relabel_criteria.py lists              -> the two lists, their difference, the call sites
    python ligand_relabel_criteria.py collect [--limit N] [--procs 20]
                                                         -> ligand_relabel_criteria_entries.csv
    python ligand_relabel_criteria.py transitions [--procs 22]
                                                         -> every old -> new candidate-label move, whole set
    python ligand_relabel_criteria.py summarise          -> every number the Methods quote

`collect` re-runs the published relabel on the per-job AHoJ archives of the 237,922 analysis-set
entries, three ways -- no relabel, total_le1 (what ran), heavy_le1 (the road not taken) -- and
reproduces the classifier's own curation around each, so the n_apo / n_holo it reports are the counts
that fed the KDEs.  Fidelity against the released classification summary is printed by `summarise`
and must be ~100% for the total_le1 arm.

All files go to the manuscript-analyses folder (ma_paths.OUT). The lists and the call sites are read from
AHOJ_PARSE_AHOJ when it is set (a parse-ahoj checkout, whose git log dates the call sites), else from the
copy of the classification code in this repository.
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import pandas as pd

import ahoj_job  # job archives under the names before AHoJ 2.6.0

HERE = os.path.dirname(os.path.abspath(__file__))
PARSE_AHOJ = os.environ.get("AHOJ_PARSE_AHOJ", str(P.CLASSIFIER_DIR))
UNI = str(P.cfg.ANALYSIS_UNIFIED_LEVELS_CSV)
DB = str(P.cfg.AHOJ_DB_DIR)
MERGED = str(P.cfg.CLASSIFICATION_MERGED_DIR)
SUMMARY = str(P.cfg.CLASSIFICATION_SUMMARIES_CSV)
PARAMS = os.path.join(DB, "Summaries", "apoholo_metric_stats", "full_run_3", "parameters.json")
ROOT = str(P.cfg.AHOJ_OUTPUT_FIX_DIR)

SRC = os.path.join(PARSE_AHOJ, "parsing_functions_DBv2.py")
ENTRIES_CSV = P.out("ligand_relabel_criteria_entries.csv")
SUMMARY_CSV = P.out("ligand_relabel_criteria_summary.csv")
CODES_CSV = P.out("ligand_relabel_criteria_codes.csv")

APO_LABELS = ["A", "A*", "A**"]
HOLO_LABELS = ["H", "H*"]
VARIANTS = ["raw", "total", "heavy"]          # raw = AHoJ's own labels, no relabel
_SPLIT = re.compile(r"[-\s;,]+")

_ARGS = None


# ---------------------------------------------------------------- the lists and their provenance

def read_lists():
    """Parse the two built-in lists straight out of the methodology repo, so this script cannot
    drift from it silently."""
    src = open(SRC, encoding="utf-8").read()
    out = {}
    for name in ("EXCLUDED_LIGANDS_TOTAL_LE1", "EXCLUDED_LIGANDS_HEAVY_LE1"):
        m = re.search(name + r"\s*=\s*\[(.*?)\]", src, re.S)
        if m is None:
            raise RuntimeError(f"{name} not found in {SRC}")
        out[name] = sorted({t.strip().strip("'\"").upper() for t in m.group(1).split(",") if t.strip()})
    return out


def call_sites():
    """Every place in the methodology repo that calls the relabel, with the builtin it passes."""
    rows = []
    for root, dirs, files in os.walk(PARSE_AHOJ):
        dirs[:] = [d for d in dirs if d not in {".git", ".venv", "__pycache__", ".local_tests"}]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            try:
                lines = open(p, encoding="utf-8").read().splitlines()
            except Exception:  # noqa: BLE001
                continue
            for i, line in enumerate(lines, 1):
                if "relabel_apoholo_by_ions(" not in line or line.lstrip().startswith(("#", "def ", "from ", "import ")):
                    continue
                chunk = " ".join(lines[i - 1:i + 3])
                m = re.search(r"builtin\s*=\s*[\"']([a-z0-9_]+)[\"']", chunk)
                rows.append({"file": os.path.relpath(p, PARSE_AHOJ).replace("\\", "/"), "line": i,
                             "builtin": m.group(1) if m else "(default: heavy_le1)",
                             "code": line.strip()})
    return pd.DataFrame(rows).sort_values(["file", "line"])


def first_commit_of_call_site():
    """When did the pipeline first pin builtin='total_le1'?  Answered by the methodology repo's log."""
    try:
        out = subprocess.run(
            ["git", "-C", PARSE_AHOJ, "log", "--reverse", "--format=%h|%ad|%s", "--date=short",
             "-S", 'relabel_apoholo_by_ions(df, builtin="total_le1"', "--", "batch_job_functions.py"],
            capture_output=True, text=True, check=False)
        return [l for l in out.stdout.splitlines() if l.strip()]
    except Exception as e:  # noqa: BLE001
        return [f"(git unavailable: {e})"]


def ahoj_run_provenance(n=5):
    """The AHoJ settings the published archives were produced with. They decide what can reach the ligand
    columns at all -- in particular water, which AHoJ reports in its own solvent columns."""
    keys = ("script_version", "water_as_ligand", "nonstd_rsds_as_lig", "d_aa_as_lig", "lig_scan_radius",
            "afdb_version")
    seen, shards = {k: set() for k in keys}, sorted(os.listdir(ROOT))[:n]
    for sh in shards:
        d = os.path.join(ROOT, sh)
        files = [f for f in os.listdir(d) if f.endswith(".tgz")][:1]
        for fn in files:
            try:
                with tarfile.open(os.path.join(d, fn), "r:gz") as tar:
                    names = tar.getnames()
                    m = "./script_parameters.json" if "./script_parameters.json" in names else "script_parameters.json"
                    p = json.load(io.TextIOWrapper(tar.extractfile(m)))
            except Exception:  # noqa: BLE001
                continue
            for k in keys:
                seen[k].add(str(p.get(k)))
    return seen


def cmd_lists(_a):
    L = read_lists()
    t, h = set(L["EXCLUDED_LIGANDS_TOTAL_LE1"]), set(L["EXCLUDED_LIGANDS_HEAVY_LE1"])
    print(f"source: {SRC}")
    print(f"\nEXCLUDED_LIGANDS_TOTAL_LE1  n = {len(t)}   (<= 1 atom in total)")
    print(f"EXCLUDED_LIGANDS_HEAVY_LE1  n = {len(h)}   (<= 1 non-hydrogen atom)")
    print(f"\nTOTAL_LE1 is a subset of HEAVY_LE1: {t.issubset(h)}")
    print(f"codes only in HEAVY_LE1 (n = {len(h - t)}): {', '.join(sorted(h - t))}")
    print(f"codes only in TOTAL_LE1 (n = {len(t - h)}): {', '.join(sorted(t - h)) or '(none)'}")
    print("\n=== the published list, in full (TOTAL_LE1) ===")
    print(", ".join(sorted(t)))
    print("\n=== call sites in the methodology repo ===")
    pd.set_option("display.width", 250, "display.max_colwidth", 110)
    print(call_sites()[["file", "line", "builtin"]].to_string(index=False))
    print("\n=== when the pipeline first pinned builtin='total_le1' ===")
    for line in first_commit_of_call_site():
        print("  " + line)
    print("\nThe run parameters archived with the published run record only relabel_apoholo = 1;")
    print("the list identity is fixed in code, which is why it has to be read from the repository.")
    print("\n=== the AHoJ settings behind the published archives (script_parameters.json) ===")
    for k, v in ahoj_run_provenance().items():
        print(f"  {k:<20} {', '.join(sorted(v)) or '(unread)'}")
    print("  water_as_ligand 0 plus AHoJ's `hetatm and not solvent and not polymer` ligand selection is why")
    print("  HOH and DOD never appear in query_ligs / nonquery_ligs (see `summarise`).")


# ------------------------------------------------------------------------------ the measurement

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
    sys.path.insert(0, PARSE_AHOJ)
    _ARGS = _load_args()


def _read_global_results(tgz):
    return ahoj_job.read_global_results(tgz)


def _parse_ligs(val):
    if pd.isna(val):
        return []
    s = str(val).strip().upper()
    if s in {"", "NAN", "NONE", "."}:
        return []
    return [p for p in _SPLIT.split(s) if p]


def _curate(df, a):
    """The published curation that follows the relabel: zero -> NaN, the volume rule, then
    sort_dataframe_v2c with the full_run_3 thresholds."""
    from parsing_functions_DBv2 import sort_dataframe_v2c
    from parse_AHoJ3_DBv2b import replace_zero_with_nan
    d = replace_zero_with_nan(df, a.columns_nozero)
    s = pd.to_numeric(d["#resis3"], errors="coerce")
    v = pd.to_numeric(d["volume"], errors="coerce")
    d["volume"] = np.where(s.notna() & (s != 0) & v.isna(), 0.0, v)
    d = sort_dataframe_v2c(d, sorting=0, full_pockets_aligned=a.full_pockets_aligned,
                           bndgrsds_prcnt_seqres=a.bndgrsds_prcnt_seqres,
                           bndgrsds_prcnt_obs=a.bndgrsds_prcnt_obs, resolution_soft=a.resolution_soft,
                           rog_diff_threshold_deltaQ=a.rog_diff_thrs_query,
                           rog_diff_threshold_mean=a.rog_diff_thrs_mean, pocket_distance=a.pocket_dist,
                           pocket_len_resis3=a.pocket_len_resis3,
                           drop_no_resolution=a.drop_no_resolution)
    return d[d["role"] == "C"]


def process(job):
    """job: (job_name, tgz_path).  Returns one flat dict per analysis-set entry."""
    from parsing_functions_DBv2 import relabel_apoholo_by_ions
    job_name, tgz = job
    a = _ARGS
    try:
        df = _read_global_results(tgz)
    except Exception as e:  # noqa: BLE001
        return {"job_name": job_name, "status": f"error:{type(e).__name__}"}
    if df is None or df.empty:
        return {"job_name": job_name, "status": "not_found"}

    L = _LISTS
    heavy_only = L["heavy_only"]
    rec = {"job_name": job_name, "status": "ok"}

    cand = df["role"].astype(str) == "C"
    rec["n_cand_rows"] = int(cand.sum())
    old = df["apoholo_assignment"].astype(str).str.upper()

    # the query-ion escape hatch: if ANY query row's own ligand is on the list, no filtering
    # is applied anywhere in this job (query_ion_disables_filter, the default)
    q = df[df["role"].astype(str) == "Q"]
    for tag, S in (("total", L["total"]), ("heavy", L["heavy"])):
        rec[f"q_disabled_{tag}"] = bool(
            not q.empty and q["query_ligs"].apply(lambda v: any(x in S for x in _parse_ligs(v))).any())

    # how often do the 17 HEAVY_LE1-only codes appear at all, in the columns the filter reads?
    hits = Counter()
    for col in ("query_ligs", "nonquery_ligs"):
        for v in df.loc[cand, col]:
            for code in _parse_ligs(v):
                if code in heavy_only:
                    hits[code] += 1
    rec["heavy_only_hits"] = int(sum(hits.values()))
    rec["heavy_only_codes"] = ";".join(f"{k}:{v}" for k, v in sorted(hits.items())) if hits else ""

    labels = {}
    for tag, builtin in (("total", "total_le1"), ("heavy", "heavy_le1")):
        d, _ = relabel_apoholo_by_ions(df.copy(), builtin=builtin, save_dir=None, verbose=False)
        labels[tag] = d["apoholo_assignment"].astype(str).str.upper()
        rec[f"n_changed_{tag}"] = int((cand & (labels[tag] != old)).sum())
        rec[f"n_demoted_{tag}"] = int((cand & old.isin(HOLO_LABELS) & (labels[tag] == "A**")).sum())
        rec[f"n_promoted_{tag}"] = int((cand & old.isin(["A", "A*"]) & labels[tag].isin(HOLO_LABELS)).sum())
        try:
            c = _curate(d, a)
            rec[f"n_apo_{tag}"] = int(c["apoholo_assignment"].astype(str).isin(APO_LABELS).sum())
            rec[f"n_holo_{tag}"] = int(c["apoholo_assignment"].astype(str).isin(HOLO_LABELS).sum())
        except Exception as e:  # noqa: BLE001
            rec[f"n_apo_{tag}"] = rec[f"n_holo_{tag}"] = -1
            rec["status"] = f"curate_error:{type(e).__name__}"
    rec["n_heavy_vs_total"] = int((cand & (labels["heavy"] != labels["total"])).sum())

    # the counterfactual the analysis set turns on: no relabel at all
    try:
        c = _curate(df.copy(), a)
        rec["n_apo_raw"] = int(c["apoholo_assignment"].astype(str).isin(APO_LABELS).sum())
        rec["n_holo_raw"] = int(c["apoholo_assignment"].astype(str).isin(HOLO_LABELS).sum())
    except Exception as e:  # noqa: BLE001
        rec["n_apo_raw"] = rec["n_holo_raw"] = -1
        rec["status"] = f"curate_error_raw:{type(e).__name__}"
    return rec


_LISTS = None


def _init_full():
    global _ARGS, _LISTS
    sys.path.insert(0, PARSE_AHOJ)
    _ARGS = _load_args()
    L = read_lists()
    t, h = set(L["EXCLUDED_LIGANDS_TOTAL_LE1"]), set(L["EXCLUDED_LIGANDS_HEAVY_LE1"])
    _LISTS = {"total": t, "heavy": h, "heavy_only": h - t}


def transitions_one(job):
    """Every old -> new candidate-label transition of one entry, under the published list.  Cheap:
    no curation pass, so this runs over the whole analysis set in a few minutes."""
    from parsing_functions_DBv2 import relabel_apoholo_by_ions
    job_name, tgz = job
    try:
        df = _read_global_results(tgz)
    except Exception:  # noqa: BLE001
        return Counter({("_error", "_error"): 1})
    if df is None or df.empty:
        return Counter()
    old = df["apoholo_assignment"].astype(str).str.upper()
    d, _ = relabel_apoholo_by_ions(df.copy(), builtin="total_le1", save_dir=None, verbose=False)
    new = d["apoholo_assignment"].astype(str).str.upper()
    c = df["role"].astype(str) == "C"
    return Counter(zip(old[c], new[c]))


def transitions(procs, limit=None):
    jobs = job_list(limit)
    print(f"entries: {len(jobs):,}  procs={procs}")
    t0 = time.time()
    tot = Counter()
    with Pool(procs, initializer=_init_full) as pool:
        for i, c in enumerate(pool.imap_unordered(transitions_one, jobs, chunksize=64), 1):
            tot.update(c)
            if i % 25000 == 0:
                print(f"  {i:,} done  {time.time() - t0:6.0f} s", flush=True)
    rows = int(sum(tot.values()))
    changed = {k: v for k, v in tot.items() if k[0] != k[1]}
    print(f"\ncandidate rows: {rows:,}; transitions that changed the label: {sum(changed.values()):,}")
    apo, holo = set(APO_LABELS), set(HOLO_LABELS)
    buckets = Counter()
    for (o, n), v in changed.items():
        if o in holo and n in apo:
            buckets["holo -> apo"] += v
        elif o in apo and n in holo:
            buckets["apo -> holo"] += v
        elif o in holo and n in holo:
            buckets["within holo (H <-> H*)"] += v
        elif o in apo and n in apo:
            buckets["within apo"] += v
        else:
            buckets[f"other ({o} -> {n})"] += v
    print("\nby class movement:")
    for k, v in buckets.most_common():
        print(f"  {k:<26} {v:>10,}  ({100 * v / rows:.3f}% of candidate rows)")
    print("\nevery transition:")
    out = []
    for (o, n), v in sorted(tot.items(), key=lambda kv: -kv[1]):
        mark = "" if o == n else "  <- changed"
        print(f"  {o:>4} -> {n:<4} {v:>10,}{mark}")
        out.append({"old": o, "new": n, "candidate_rows": v, "changed": o != n})
    pd.DataFrame(out).to_csv(P.out("ligand_relabel_criteria_transitions.csv"), index=False)
    print(f"\nwrote ligand_relabel_criteria_transitions.csv  ({time.time() - t0:.0f} s)")


def job_list(limit=None, seed=42):
    uni = pd.read_csv(UNI, usecols=["threshold", "rep_job", "query_path"], low_memory=False)
    uni = uni[uni.threshold == "UNCLUSTERED"]
    jobs = []
    for jn, qp in zip(uni.rep_job, uni.query_path):
        batch = str(qp).replace("\\", "/").split("/")[-2]
        jobs.append((jn, os.path.join(ROOT, batch, f"{jn}.tgz")))
    if limit and limit < len(jobs):
        rng = np.random.default_rng(seed)          # a random sample, not the head: the head is
        idx = rng.choice(len(jobs), limit, replace=False)   # alphabetical and therefore biased
        jobs = [jobs[i] for i in sorted(idx)]
    return jobs


def collect(limit, procs, out=ENTRIES_CSV):
    jobs = job_list(limit)
    print(f"entries to process: {len(jobs):,}  procs={procs}  (analysis set = 237,922)")
    t0 = time.time()
    rows = []
    with Pool(procs, initializer=_init_full) as pool:
        for i, rec in enumerate(pool.imap_unordered(process, jobs, chunksize=32), 1):
            rows.append(rec)
            if i % 5000 == 0:
                print(f"  {i:,} done  {time.time() - t0:7.0f} s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(d):,}  {time.time() - t0:.0f} s")
    print(d.status.value_counts().to_string())


def summarise(entries=ENTRIES_CSV):
    L = read_lists()
    t, h = set(L["EXCLUDED_LIGANDS_TOTAL_LE1"]), set(L["EXCLUDED_LIGANDS_HEAVY_LE1"])
    print(f"lists: TOTAL_LE1 n={len(t)}, HEAVY_LE1 n={len(h)}, HEAVY-only n={len(h - t)}: "
          f"{', '.join(sorted(h - t))}")

    d = pd.read_csv(entries, low_memory=False)
    n_all = len(d)
    print(f"\nentries read: {n_all:,}\n{d.status.value_counts().to_string()}")
    d = d[d.status == "ok"].copy()
    sampled = n_all < 200000
    print(f"ok entries: {len(d):,}" + ("   [SAMPLE -- percentages generalise, counts do not]" if sampled else ""))

    # --- fidelity of the total_le1 arm against the released classification summary
    if os.path.exists(SUMMARY):
        s = pd.read_csv(SUMMARY, usecols=["job_name", "n_apo", "n_holo"], low_memory=False)
        m = d.merge(s, on="job_name", how="inner")
        ok = (m.n_apo_total == m.n_apo) & (m.n_holo_total == m.n_holo)
        okh = (m.n_apo_heavy == m.n_apo) & (m.n_holo_heavy == m.n_holo)
        print(f"\nfidelity vs the released classification (n = {len(m):,}):")
        print(f"  total_le1 reproduces n_apo AND n_holo: {100 * ok.mean():.3f}%   <- what ran")
        print(f"  heavy_le1 reproduces n_apo AND n_holo: {100 * okh.mean():.3f}%")
    else:
        print(f"\n[classification summary not found at {SUMMARY}; fidelity check skipped]")

    # --- what the relabel does, on the published list
    tot_rows = int(d.n_cand_rows.sum())
    print(f"\n=== the published relabel (TOTAL_LE1), over {len(d):,} analysis-set entries ===")
    print(f"candidate rows before curation: {tot_rows:,}")
    for k, lab in (("n_changed_total", "candidate rows whose label the relabel changed"),
                   ("n_demoted_total", "  of which demoted holo -> A** (ligand set is excluded-only)"),
                   ("n_promoted_total", "  of which promoted apo -> H/H* (ligand column non-empty)")):
        v = int(d[k].sum())
        print(f"{lab}: {v:,} ({100 * v / tot_rows:.2f}% of candidate rows)")
    ent = int((d.n_changed_total > 0).sum())
    print(f"entries with at least one relabelled candidate: {ent:,} ({100 * ent / len(d):.1f}%)")
    qd = int(d.q_disabled_total.sum())
    print(f"entries where the query's own ligand is on the list, so filtering is disabled job-wide: "
          f"{qd:,} ({100 * qd / len(d):.1f}%)")

    # --- the counterfactual: without the relabel, would the entry still be in the analysis set?
    lost = d[(d.n_apo_raw == 0) & (d.n_apo_total >= 1)]
    print(f"\nentries whose apo evidence exists only because of the relabel (n_apo would be 0 without it): "
          f"{len(lost):,} ({100 * len(lost) / len(d):.1f}%)  <- these entries are in the analysis set "
          f"because the relabel put them there")
    ch = d[(d.n_apo_raw != d.n_apo_total) | (d.n_holo_raw != d.n_holo_total)]
    print(f"entries whose curated n_apo / n_holo differ with vs without the relabel: {len(ch):,} "
          f"({100 * len(ch) / len(d):.1f}%)")

    # --- heavy_le1: would it have mattered?
    print(f"\n=== heavy_le1 vs total_le1 (the 17 extra codes) ===")
    hh = int(d.heavy_only_hits.sum())
    print(f"candidate-row occurrences of any HEAVY-only code in query_ligs / nonquery_ligs: {hh:,}")
    hv = int(d.n_heavy_vs_total.sum())
    print(f"candidate rows labelled differently by the two lists: {hv:,}")
    diff = d[(d.n_apo_heavy != d.n_apo_total) | (d.n_holo_heavy != d.n_holo_total)]
    print(f"entries whose curated n_apo / n_holo would differ: {len(diff):,} "
          f"({100 * len(diff) / len(d):.2f}%)")
    codes = Counter()
    for s_ in d.heavy_only_codes.dropna():
        for tok in str(s_).split(";"):
            if ":" in tok:
                k, v = tok.split(":")
                codes[k] += int(v)
    if codes:
        cd = pd.DataFrame(sorted(codes.items(), key=lambda kv: -kv[1]), columns=["code", "candidate_rows"])
        cd.to_csv(CODES_CSV, index=False)
        print(f"\nHEAVY-only codes actually seen (written to {os.path.basename(CODES_CSV)}):")
        print(cd.to_string(index=False))
    else:
        print("\nNo HEAVY-only code occurs in the ligand columns at all.  AHoJ 2.1.0 selects pocket")
        print("ligands with `hetatm and not solvent and not polymer` and reports water separately in")
        print("its own `#solvent` / `solvent` columns (water_as_ligand = 0 in the published run), so")
        print("HOH, DOD, OH, NH3, CH3 and the rest never reach the filter.  On this data the two")
        print("lists are interchangeable; the published choice is TOTAL_LE1 either way.")

    out = pd.DataFrame([{
        "entries_ok": len(d),
        "candidate_rows": tot_rows,
        "relabelled_rows": int(d.n_changed_total.sum()),
        "demoted_holo_to_Astarstar": int(d.n_demoted_total.sum()),
        "promoted_apo_to_holo": int(d.n_promoted_total.sum()),
        "entries_with_a_relabel": ent,
        "entries_query_ion_disabled": qd,
        "entries_apo_only_via_relabel": len(lost),
        "entries_counts_changed_by_relabel": len(ch),
        "heavy_only_code_occurrences": hh,
        "rows_differing_heavy_vs_total": hv,
        "entries_differing_heavy_vs_total": len(diff),
        "list_total_le1_n": len(t), "list_heavy_le1_n": len(h), "heavy_only_n": len(h - t),
        "is_sample": sampled,
    }])
    out.to_csv(SUMMARY_CSV, index=False)
    print(f"\nwrote {SUMMARY_CSV}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lists").set_defaults(fn=cmd_lists)
    c = sub.add_parser("collect")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--procs", type=int, default=20)
    c.set_defaults(fn=lambda a: collect(a.limit, a.procs))
    tr = sub.add_parser("transitions")
    tr.add_argument("--procs", type=int, default=22)
    tr.add_argument("--limit", type=int, default=None)
    tr.set_defaults(fn=lambda a: transitions(a.procs, a.limit))
    s = sub.add_parser("summarise")
    s.add_argument("--entries", default=ENTRIES_CSV)
    s.set_defaults(fn=lambda a: summarise(a.entries))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
