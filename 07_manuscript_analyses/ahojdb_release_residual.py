# -*- coding: utf-8 -*-
"""The processed and the released AHoJ-DB v2c sets, and how they overlap (Methods, Study design).

The Methods quote two sets of numbers for AHoJ-DB v2c: the release statistics the webserver publishes
(db_stats.json: 515,463 entries, 121,029 PDB entries, 37,620 ligands, 29,463 UniProt accessions) and the
set the classification run processed (516,794 query pockets, 121,227 PDB entries, 37,662 ligands, counted
over the run_3 job lists as Figure 1 does). Removing the 1,327 jobs that finalized without a table ("Empty
dataframe") from the processed set leaves 516,794 - 1,327 = 515,467 against 515,463 released (+4), and likewise
one PDB entry. This script recomputes every term from the job lists, the release tar's listing and db_stats.json,
and (with --api, or from its cache) checks the two sides against the webserver's entry endpoint. That gives the
Methods sentence "the two sets overlap in 514,529 entries, 99.8% of the release (1,327 processed pockets returned
no result table and are not served, a further 938 are not served, and 934 served entries were not in the run's
job list)".

Inputs (read-only)
    <ahoj_db_dir>/Products/KDEs_and_classification/full_run_3/plots_summary/full_run_3_merged/
        Success_jobs.txt, Discarded_jobs.txt           the run_3 job lists (job: reason)
    <ahoj_db_dir>/Database/ahoj_db_v2c_tar.log         `tar -cvf ... --verify` listing of the release archive (UTF-16)
    https://dev.apoholo.cz/api/db/archive/v2c/db_stats.json, cached in <manuscript_analyses_dir> as
        ahojdb_v2c_db_stats.json (the cache is used when the fetch fails)
    --api: https://dev.apoholo.cz/api/db/entry/<key> (HTTP 200 = released, 404 = not released), results cached
        in <manuscript_analyses_dir>/ahojdb_release_residual_api.json; without --api that cache is read if present

Usage
    python ahojdb_release_residual.py            # local terms (job lists, tar listing, db_stats cache or fetch),
                                                 # plus the endpoint statuses from the cache if it is there
    python ahojdb_release_residual.py --api      # also probe the entry endpoint for the empty-table jobs and the
                                                 # tar-only archives (a few thousand light GETs; cached)
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import ma_paths as P

HERE = Path(__file__).resolve().parent
AHOJ_DB_DIR = Path(P.cfg.AHOJ_DB_DIR)
MERGED = AHOJ_DB_DIR / "Products" / "KDEs_and_classification" / "full_run_3" / "plots_summary" / "full_run_3_merged"
TAR_LOG = AHOJ_DB_DIR / "Database" / "ahoj_db_v2c_tar.log"
DB_STATS_URL = "https://dev.apoholo.cz/api/db/archive/v2c/db_stats.json"
ENTRY_URL = "https://dev.apoholo.cz/api/db/entry/{key}"
DB_STATS_CACHE = Path(P.out("ahojdb_v2c_db_stats.json"))
API_CACHE = Path(P.out("ahojdb_release_residual_api.json"))

EMPTY = "Empty dataframe"
LOWCAND = "Low candidates after filtering"


def read_jobs(path):
    """job -> reason (empty string for Success_jobs.txt), in file order."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            key, _, reason = line.partition(":")
            out[key.strip()] = reason.strip()
    return out


def pdb_lig(job):
    """Figure 1's parsing of a job name <pdb>-<chain>-<ligand>-<resnum>: PDB id and ligand code."""
    p = job.split("-")
    return p[0].lower(), "-".join(p[2:-1]).upper()


def read_tar_keys(path):
    """Entry keys (archive basenames without .tgz) listed in the release tar log."""
    keys = []
    with open(path, encoding="utf-16", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.endswith(".tgz"):
                keys.append(line.split("/")[-1][:-4])
    return keys


def fetch_db_stats():
    try:
        import urllib.request
        with urllib.request.urlopen(DB_STATS_URL, timeout=30) as r:
            stats = json.load(r)
        DB_STATS_CACHE.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        return stats, "fetched"
    except Exception as e:  # noqa: BLE001
        if DB_STATS_CACHE.exists():
            return json.loads(DB_STATS_CACHE.read_text(encoding="utf-8")), f"cache ({e.__class__.__name__})"
        raise


def probe(keys, label, cache):
    """HTTP status of /api/db/entry/<key> for every key; cached by key."""
    import urllib.error
    import urllib.request
    todo = [k for k in keys if k not in cache]
    print(f"  probing {label}: {len(keys):,} keys, {len(todo):,} not cached")
    t0 = time.time()
    for i, k in enumerate(todo, 1):
        try:
            with urllib.request.urlopen(ENTRY_URL.format(key=k), timeout=30) as r:
                cache[k] = r.status
        except urllib.error.HTTPError as e:
            cache[k] = e.code
        except Exception as e:  # noqa: BLE001
            cache[k] = f"error: {e.__class__.__name__}"
        if i % 200 == 0:
            print(f"    {i:,}/{len(todo):,} ({time.time() - t0:.0f} s)")
            API_CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8")
    API_CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8")
    return Counter(cache[k] for k in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--api", action="store_true", help="probe the webserver's entry endpoint (cached)")
    a = ap.parse_args()

    succ = read_jobs(MERGED / "Success_jobs.txt")
    disc = read_jobs(MERGED / "Discarded_jobs.txt")
    processed = dict(succ)
    processed.update(disc)
    reasons = Counter(r.split(":")[0] for r in disc.values())
    empty = {k for k, r in disc.items() if r.startswith(EMPTY)}
    lowc = {k for k, r in disc.items() if r.startswith(LOWCAND)}
    print("=== the processed set (run_3 job lists) ===")
    print(f"  success {len(succ):,} + discarded {len(disc):,} = processed {len(processed):,}")
    for r, n in reasons.most_common():
        print(f"    discarded, {r}: {n:,}")
    assert len(empty) + len(lowc) == len(disc), "unexpected discard reason"

    def counts(keys):
        pdbs, ligs = set(), set()
        for k in keys:
            p, l = pdb_lig(k)
            pdbs.add(p)
            ligs.add(l)
        return len(keys), len(pdbs), len(ligs)

    n_p, pdb_p, lig_p = counts(processed)
    kept = set(processed) - empty
    n_k, pdb_k, lig_k = counts(kept)
    print(f"  processed: {n_p:,} pockets, {pdb_p:,} PDB entries, {lig_p:,} ligands (Figure 1's parsing)")
    print(f"  minus the {len(empty):,} empty-table jobs: {n_k:,} pockets, {pdb_k:,} PDB entries, {lig_k:,} ligands")
    pdb_only_empty = pdb_p - pdb_k
    lig_only_empty = lig_p - lig_k
    print(f"    (PDB entries whose every job is empty-table: {pdb_only_empty:,}; ligands likewise: {lig_only_empty:,})")

    stats, how = fetch_db_stats()
    n_r, pdb_r, lig_r, unp_r = (stats["num_entries"], stats["num_unique_pdb_ids"],
                                stats["num_unique_target_ligands"], stats["num_unique_uniprot_ids"])
    print(f"\n=== the released statistics (db_stats.json, {how}) ===")
    print(f"  {n_r:,} entries, {pdb_r:,} PDB entries, {lig_r:,} ligands, {unp_r:,} UniProt accessions")
    print("\n=== residual: (processed minus empty-table) minus released ===")
    print(f"  entries     {n_k:,} - {n_r:,} = {n_k - n_r:+d}")
    print(f"  PDB entries {pdb_k:,} - {pdb_r:,} = {pdb_k - pdb_r:+d}")
    print(f"  ligands     {lig_k:,} - {lig_r:,} = {lig_k - lig_r:+d}")
    print("  (the UniProt count has no job-list equivalent: accessions are assigned per entry at release)")

    tar = read_tar_keys(TAR_LOG)
    tar_set = set(tar)
    print(f"\n=== the release archive (ahoj_db_v2c.tar listing) ===")
    print(f"  {len(tar):,} archives ({len(tar_set):,} distinct keys)")
    print(f"  processed keys in the tar: {len(set(processed) & tar_set):,} of {len(processed):,}; "
          f"processed keys NOT in the tar: {len(set(processed) - tar_set):,}")
    tar_only = sorted(tar_set - set(processed))
    print(f"  tar archives not in the processed set: {len(tar_only):,}")
    print(f"  tar - empty-table jobs = {len(tar_set - empty):,}; released {n_r:,}; "
          f"difference {len(tar_set - empty) - n_r:+,} (archives the webserver did not load as entries, "
          f"beyond the empty-table ones)")
    missing_from_processed_case = {k.lower() for k in tar_only} & {k.lower() for k in processed}
    print(f"  tar-only keys that match a processed key up to case: {len(missing_from_processed_case):,}")

    cache = json.loads(API_CACHE.read_text(encoding="utf-8")) if API_CACHE.exists() else {}
    if a.api:
        print("\n=== webserver entry endpoint (HTTP status per key; 200 = released, 404 = not released) ===")
        c_e = probe(sorted(empty), "empty-table jobs", cache)
        c_t = probe(tar_only, "tar-only archives", cache)
    elif cache:
        print(f"\n=== webserver entry endpoint, from the cache {API_CACHE.name} (re-probe with --api) ===")
        c_e = Counter(cache.get(k, "not probed") for k in empty)
        c_t = Counter(cache.get(k, "not probed") for k in tar_only)
    else:
        print("\n(no API probe: run with --api to check the two sides against the webserver)")
        return 0
    print(f"  empty-table jobs: {dict(c_e)}")
    print(f"  tar-only archives: {dict(c_t)}")
    released_tar_only = sorted(k for k in tar_only if cache.get(k) == 200)
    served_empty = sum(1 for k in empty if cache.get(k) == 200)
    print(f"  tar-only archives that ARE released: {len(released_tar_only):,} "
          f"(e.g. {', '.join(released_tar_only[:4])}); empty-table jobs released: {served_empty:,}")
    # The webserver serves entries from this tar only, so released = (processed and served) + (tar-only and served).
    served_processed = n_r - len(released_tar_only)
    unserved_processed = len(processed) - served_processed
    print("\n=== the relation of the two sets (released entries come from the tar, and only from it) ===")
    print(f"  released {n_r:,} = processed-and-served {served_processed:,} + tar-only-and-served {len(released_tar_only):,}")
    print(f"  processed {len(processed):,} = served {served_processed:,} + not served {unserved_processed:,} "
          f"(of which empty-table {len(empty) - served_empty:,}, other {unserved_processed - len(empty) + served_empty:,})")
    print(f"  overlap: {served_processed:,} entries = {100 * served_processed / n_r:.2f}% of the release, "
          f"{100 * served_processed / len(processed):.2f}% of the processed set")
    print(f"  the +4 of the Methods is therefore a net of two one-sided differences "
          f"({unserved_processed - len(empty) + served_empty:,} processed-not-served beyond the empty-table jobs, "
          f"{len(released_tar_only):,} served-not-processed), not four attributable entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
