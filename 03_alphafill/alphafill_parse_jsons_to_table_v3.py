
"""
AlphaFill vs UBS parser
-----------------------------------------
Parses AlphaFill JSONs for your UniProts and compares transplants against
AHoJ-DB ligands implied by each UBS row ("job_names").

Outputs two CSVs:
  1) alphafill_summary_per_uniprot.csv
  2) alphafill_per_ubs_detailed.csv

"Present" means AlphaFill transplanted a ligand whose CCD matches one of the
CCDs seen among the UBS's AHoJ-DB entries. "Missing" means no such CCD appears
in the JSON for that UniProt.

How to use in Spyder:
  1) Run the downloader first to get JSONs.
  2) Edit the CONFIG block below (paths).
  3) Press F5 to run.

Implementation notes:
  * We rely on AlphaFill's JSON metadata. The documented endpoint is
    https://alphafill.eu/v1/aff/${entryid}/json
  * The JSON schema may evolve; we therefore read TCS using flexible keys
    (e.g., 'tcs', 'TCS', 'transplantClashScore'). Likewise for compound IDs.
  * Per-UBS assignment is done by matching CCDs from your AHoJ entries to
    the set of transplanted CCDs for that UniProt. If a UniProt has several
    UBSs with the same CCD in play, this is an intentional, CCD-level match.

AlphaFill JSON -> flat table parser (Step 1, v2)
-----------------------------------------------------------------
Fixes:
  * Uses top-level 'hits' with nested 'transplants' (correct AlphaFill schema).
  * Builds reconstructed AHoJ keys: <pdb_id>_<pdb_auth_asym_id>_<compound_id>_<pdb_auth_seq_id>.
  * Writes per-UniProt JSON files listing all reconstructed keys + metrics.
  * Keeps CSV outputs + run log as before.
Outputs in OUTPUT_DIR:
  - alphafill_parsed.csv
  - alphafill_uniprots_summary.csv
  - per_uniprot_json/<uniprot>.json (new)
Log in LOGGING_DIR:
  - _parse_log.csv

AlphaFill JSON -> flat table parser (Step 1, v3)
-----------------------------------------------------------------
Captures structure-level and ligand-level chain IDs + mapping context.
"""

from __future__ import annotations

import os, json, pathlib
from typing import Any, Dict, List, Tuple
import pandas as pd


def get_cwd_prefix(levels=3):    # levels=3 -> root + 2 dirs; levels=2 -> root + 1 dir, etc.
    from pathlib import Path
    return str(Path(*Path.cwd().parts[:levels]))


# ===================== CONFIG =====================
# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# Input
SOURCE_CSV_PATH  = str(cfg.ANALYSIS_UNIFIED_LEVELS_CSV)
AF_JSON_DIR      = str(cfg.ALPHAFILL_JSONS_DIR)                  # Directory with <uniprot>.json files

# Output
OUTPUT_DIR       = str(cfg.ALPHAFILL_PARSED_ZIP.parent)  # config-driven version dir (run_3: v3; was hardcoded "v2")
LOGGING_DIR      = str(cfg.ALPHAFILL_PARSED_ZIP.parent)  # Where logs will be saved

SINGLE_UNIPROT = True
UNP_COLUMN     = "uniprots_str"     # "uniprots_str" or "sites_uniprot" (old file)
UNP_SEP        = " "                # " " or "_"
LEVEL_FILTER   = "entry"            # "entry" or "ubs"

# ==================================================

def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def expected_uniprots_from_csv(
    csv_path: str,
    uniprot_col: str = "uniprots_str",  # was "site_uniprots"
    separator: str | None = " ",        # was "_"
    level_filter: str | None = "entry", # "entry" or "ubs"
    skip_multi_uniprot_entries: bool = False,
) -> List[str]:
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path, low_memory=False)

    if level_filter is not None:
        if "level" not in df.columns:
            return []
        df = df[df["level"].astype(str).str.lower() == str(level_filter).lower()].copy()

    if uniprot_col not in df.columns:
        return []

    def split_tokens(s: str) -> List[str]:
        s = "" if pd.isna(s) else str(s).strip()
        if not s:
            return []
        if separator is None:
            return s.split()
        return [tok.strip() for tok in s.split(separator) if tok.strip()]

    if skip_multi_uniprot_entries:
        df = df[df[uniprot_col].apply(lambda x: len(split_tokens(x)) == 1)].copy()

    seen, ordered = set(), []
    for v in df[uniprot_col]:
        for up in split_tokens(v):
            if up not in seen:
                seen.add(up)
                ordered.append(up)

    return ordered

def list_json_uniprots(json_dir: str) -> List[str]:
    if not os.path.isdir(json_dir): return []
    return [os.path.splitext(nm)[0] for nm in os.listdir(json_dir) if nm.lower().endswith(".json")]

def parse_one_json(uniprot: str, json_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    per_up: List[Dict[str, Any]] = []

    summary = {
        "site_uniprot": uniprot,
        "json_path": os.path.abspath(json_path),
        "json_exists": os.path.exists(json_path),
        "json_valid": False,
        "parse_ok": False,
        "parse_status": "",
        "has_hits_field": False,
        "hits_is_null": False,
        "n_hits": 0,
        "n_hits_with_transplants": 0,
        "has_transplants": False,
        "n_transplants": 0,
        "n_unique_compounds": 0,
        "unique_compounds": "",
        "error": "",
    }

    if not os.path.exists(json_path):
        summary["parse_status"] = "json_missing"
        summary["error"] = "json_missing"
        return rows, summary, per_up

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        try:
            with open(json_path, "rb") as f:
                raw = f.read()
            data = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception as e2:
            summary["parse_status"] = "json_parse_error"
            summary["error"] = f"json_parse_error:{type(e2).__name__}:{e2}"
            return rows, summary, per_up

    summary["json_valid"] = True

    if not isinstance(data, dict):
        summary["parse_ok"] = True
        summary["parse_status"] = "json_not_object"
        return rows, summary, per_up

    summary["has_hits_field"] = "hits" in data
    hits_raw = data.get("hits", None)

    if "hits" not in data:
        summary["parse_ok"] = True
        summary["parse_status"] = "hits_missing"
        return rows, summary, per_up

    if hits_raw is None:
        summary["parse_ok"] = True
        summary["hits_is_null"] = True
        summary["parse_status"] = "hits_null"
        return rows, summary, per_up

    if not isinstance(hits_raw, list):
        summary["parse_ok"] = True
        summary["parse_status"] = "hits_not_list"
        return rows, summary, per_up

    summary["n_hits"] = len(hits_raw)

    if len(hits_raw) == 0:
        summary["parse_ok"] = True
        summary["parse_status"] = "hits_empty"
        return rows, summary, per_up

    compounds_seen = set()

    for hit in hits_raw:
        if not isinstance(hit, dict):
            continue

        pdb_id = str(hit.get("pdb_id") or hit.get("pdb") or "")
        mapping_global_rmsd = hit.get("global_rmsd", None)
        structure_pdb_asym_id = hit.get("pdb_asym_id", None)

        alignment = hit.get("alignment", {}) if isinstance(hit.get("alignment"), dict) else {}
        align_identity = alignment.get("identity", None)
        align_len = alignment.get("length", None)
        align_af_start = alignment.get("af_start", None)
        align_pdb_start = alignment.get("pdb_start", None)

        transplants = hit.get("transplants", [])
        if not isinstance(transplants, list) or len(transplants) == 0:
            continue

        summary["n_hits_with_transplants"] += 1

        for tr in transplants:
            if not isinstance(tr, dict):
                continue

            compound_id = str(tr.get("compound_id") or "").upper()
            analogue_id = str(tr.get("analogue_id") or "").upper()
            ligand_asym_id = tr.get("asym_id", None)
            ligand_pdb_asym_id = tr.get("pdb_asym_id", None)
            ligand_pdb_auth_asym_id = tr.get("pdb_auth_asym_id", None)
            ligand_pdb_auth_seq_id = tr.get("pdb_auth_seq_id", None)

            clash = tr.get("clash", {}) if isinstance(tr.get("clash"), dict) else {}
            clash_count = clash.get("clash_count", None)
            clash_score = clash.get("score", None)
            local_rmsd = tr.get("local_rmsd", None)

            pae = tr.get("pae", {}) if isinstance(tr.get("pae"), dict) else {}
            pae_mean = pae.get("mean", None)
            pae_median = pae.get("median", None)
            pae_std = pae.get("stddev", None)

            reconstructed_key = None
            if pdb_id and ligand_pdb_auth_asym_id and compound_id and ligand_pdb_auth_seq_id:
                reconstructed_key = f"{pdb_id}_{ligand_pdb_auth_asym_id}_{compound_id}_{ligand_pdb_auth_seq_id}"

            row = {
                "site_uniprot": uniprot,
                "pdb_id": pdb_id,
                "compound_id": compound_id,
                "analogue_id": analogue_id,
                "structure_pdb_asym_id": structure_pdb_asym_id,
                "mapping_global_rmsd": mapping_global_rmsd,
                "mapping_align_identity": align_identity,
                "mapping_align_length": align_len,
                "mapping_align_af_start": align_af_start,
                "mapping_align_pdb_start": align_pdb_start,
                "ligand_asym_id": ligand_asym_id,
                "ligand_pdb_asym_id": ligand_pdb_asym_id,
                "ligand_pdb_auth_asym_id": ligand_pdb_auth_asym_id,
                "ligand_pdb_auth_seq_id": ligand_pdb_auth_seq_id,
                "reconstructed_key": reconstructed_key,
                "clash_count": clash_count,
                "clash_score": clash_score,
                "local_rmsd": local_rmsd,
                "pae_mean": pae_mean,
                "pae_median": pae_median,
                "pae_stddev": pae_std,
            }
            rows.append(row)

            per_up.append({
                "reconstructed_key": reconstructed_key,
                "pdb_id": pdb_id,
                "structure_pdb_asym_id": structure_pdb_asym_id,
                "mapping_global_rmsd": mapping_global_rmsd,
                "ligand_asym_id": ligand_asym_id,
                "ligand_pdb_asym_id": ligand_pdb_asym_id,
                "ligand_pdb_auth_asym_id": ligand_pdb_auth_asym_id,
                "ligand_pdb_auth_seq_id": ligand_pdb_auth_seq_id,
                "compound_id": compound_id,
                "analogue_id": analogue_id,
                "clash_count": clash_count,
                "clash_score": clash_score,
                "local_rmsd": local_rmsd,
                "pae_mean": pae_mean,
                "pae_median": pae_median,
                "pae_stddev": pae_std,
            })

            if compound_id:
                compounds_seen.add(compound_id)

    summary["parse_ok"] = True
    summary["has_transplants"] = len(rows) > 0
    summary["n_transplants"] = len(rows)
    summary["n_unique_compounds"] = len(compounds_seen)
    summary["unique_compounds"] = ";".join(sorted(compounds_seen)) if compounds_seen else ""

    if len(rows) > 0:
        summary["parse_status"] = "ok"
    else:
        summary["parse_status"] = "hits_present_but_no_transplants"

    return rows, summary, per_up


def main():
    ensure_dir(OUTPUT_DIR); ensure_dir(LOGGING_DIR)
    per_up_dir = os.path.join(OUTPUT_DIR, "per_uniprot_json"); ensure_dir(per_up_dir)

    expected = expected_uniprots_from_csv(
        SOURCE_CSV_PATH,
        uniprot_col=UNP_COLUMN,
        separator=UNP_SEP,
        level_filter=LEVEL_FILTER,
        skip_multi_uniprot_entries=SINGLE_UNIPROT,
    )

    if not expected:
        print("[WARN] Expected UniProt list from SOURCE_CSV_PATH is empty. Coverage output may be misleading.")

    json_present = sorted(set(list_json_uniprots(AF_JSON_DIR)))
    universe = json_present

    if not universe:
        print("[WARN] No JSON files found in AF_JSON_DIR."); return

    expected_set = set(expected)
    json_set = set(json_present)
    coverage_union = sorted(expected_set | json_set)

    coverage_out = os.path.join(OUTPUT_DIR, "alphafill_json_coverage.csv")
    parsed_out   = os.path.join(OUTPUT_DIR, "alphafill_parsed.csv")
    summary_out  = os.path.join(OUTPUT_DIR, "alphafill_uniprots_summary.csv")
    log_csv_path = os.path.join(LOGGING_DIR, "_parse_log.csv")

    parsed_rows_all: List[Dict[str, Any]] = []; summaries_all: List[Dict[str, Any]] = []; log_rows: List[Dict[str, Any]] = []

    for i, up in enumerate(universe, 1):
        json_path = os.path.join(AF_JSON_DIR, f"{up}.json")
        rows, summary, per_up = parse_one_json(up, json_path)
        summary["in_expected_from_csv"] = up in expected_set
        summary["json_listed_in_dir"] = up in json_set

        parsed_rows_all.extend(rows); summaries_all.append(summary)
        with open(os.path.join(per_up_dir, f"{up}.json"), "w", encoding="utf-8") as f: json.dump(per_up, f, ensure_ascii=False, indent=2)

        log_rows.append({
            "site_uniprot": up,
            "json_exists": summary.get("json_exists"),
            "json_valid": summary.get("json_valid"),
            "parse_ok": summary.get("parse_ok"),
            "parse_status": summary.get("parse_status"),
            "has_hits_field": summary.get("has_hits_field"),
            "hits_is_null": summary.get("hits_is_null"),
            "n_hits": summary.get("n_hits"),
            "n_hits_with_transplants": summary.get("n_hits_with_transplants"),
            "has_transplants": summary.get("has_transplants"),
            "n_transplants": summary.get("n_transplants"),
            "n_unique_compounds": summary.get("n_unique_compounds"),
            "error": summary.get("error", ""),
        })

        if (i % 50) == 0 or summary.get("parse_status") != "ok":
            print(
                f"[{i}/{len(universe)}] {up}: "
                f"exists={summary['json_exists']} "
                f"valid={summary['json_valid']} "
                f"status={summary['parse_status']} "
                f"hits={summary['n_hits']} "
                f"transplants={summary['n_transplants']} "
                f"err={summary['error']}"
            )

    coverage_rows = []
    for up in coverage_union:
        in_expected = up in expected_set
        in_json_dir = up in json_set

        if in_expected and in_json_dir:
            coverage_status = "expected_and_present"
        elif in_expected and not in_json_dir:
            coverage_status = "expected_missing_json"
        else:
            coverage_status = "present_not_expected"

        coverage_rows.append({
            "site_uniprot": up,
            "in_expected_from_csv": in_expected,
            "json_listed_in_dir": in_json_dir,
            "coverage_status": coverage_status,
        })

    pd.DataFrame(coverage_rows).to_csv(coverage_out, index=False)

    df_parsed = pd.DataFrame(parsed_rows_all) if parsed_rows_all else pd.DataFrame(columns=[
        "site_uniprot","pdb_id","compound_id","analogue_id",
        "structure_pdb_asym_id",#"structure_af_asym_id",
        "mapping_global_rmsd",
        "mapping_align_identity","mapping_align_length","mapping_align_af_start","mapping_align_pdb_start",
        "ligand_asym_id","ligand_pdb_asym_id","ligand_pdb_auth_asym_id","ligand_pdb_auth_seq_id",
        "reconstructed_key","clash_count","clash_score","local_rmsd","pae_mean","pae_median","pae_stddev"
    ])
    df_parsed.to_csv(parsed_out, index=False)
    pd.DataFrame(summaries_all).to_csv(summary_out, index=False)
    pd.DataFrame(log_rows).to_csv(log_csv_path, index=False)

    print("Saved:"); print(" -", parsed_out); print(" -", summary_out); print(" -", log_csv_path); print(" - per-UniProt JSONs in", per_up_dir)
    print(" -", coverage_out)

if __name__ == "__main__":
    main()
