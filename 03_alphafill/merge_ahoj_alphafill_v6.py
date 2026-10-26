
#!/usr/bin/env python3
"""
Merge analysis_unified_levels with AlphaFill parsed transplants using JOB-TO-JOB matching.

Key merge rule
--------------
- Matching is performed ONLY on normalized job keys:
    * AHoJ/unified jobs come from 'job_names' (semicolon-separated) and 'rep_job'
    * AlphaFill jobs come from 'reconstructed_key'
- UniProt is NOT used for matching.
- AlphaFill UniProt is kept only as reference / QA metadata.

Outputs
-------
1) merged_ahoj_alphafill_summary.csv
   One row per original row in the unified CSV.
   Keeps all original unified columns and appends compact AlphaFill summary columns.

2) merged_ahoj_alphafill_long.csv
   One row per matched (unified row × matched job × AlphaFill transplant).
   Keeps all original unified columns and appends a SELECTIVE set of AlphaFill columns
   and merge bookkeeping columns.

3) merge_coverage_by_level_threshold.csv
   Coverage summary by (level, threshold).

4) merge_quick_report.txt
   Human-readable report with bidirectional AHoJ <-> AlphaFill job coverage.

5) merged_ahoj_alphafill.jsonl.gz   [optional, debug]
   Gzipped JSONL, one object per unified row, with matched AlphaFill rows.

Notes
-----
- Inputs may be plain .csv or compressed (.zip/.gz); pandas infers compression by extension.
- The script is Spyder-friendly: configure paths in the USER CONFIG block and run.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterable
from collections import OrderedDict

import numpy as np
import pandas as pd


# =========================
# USER CONFIG (Spyder)
# =========================

USE_CLI = False

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

UNIFIED_CSV_PATH = cfg.ANALYSIS_UNIFIED_LEVELS_ZIP
AFILL_CSV_PATH   = cfg.ALPHAFILL_PARSED_ZIP
OUTPUT_DIR       = cfg.MERGE_AHOJ_ALPHAFILL_DIR

OUT_SUMMARY_CSV  = "merged_ahoj_alphafill_summary.csv"
OUT_LONG_CSV     = "merged_ahoj_alphafill_long.csv"
OUT_COVERAGE_CSV = "merge_coverage_by_level_threshold.csv"
OUT_REPORT_TXT   = "merge_quick_report.txt"
OUT_JSONL_GZ     = "merged_ahoj_alphafill.jsonl.gz"

WRITE_LONG_CSV = True
WRITE_SUMMARY_CSV = True
WRITE_COVERAGE_CSV = True
WRITE_REPORT = True
WRITE_JSONL_GZ = False  # debug-only; can still be large

# If not None, only process the first N unified rows (debug/testing)
ROW_LIMIT = None

# Input column names
JOB_NAMES_COL = "job_names"
REP_JOB_COL = "rep_job"
UNIPROTS_COL = "uniprots_str"
LEVEL_COL = "level"
THRESHOLD_COL = "threshold"
UNIT_UID_COL = "unit_uid"

AFILL_KEY_COL = "reconstructed_key"
AFILL_UNIPROT_COL = "site_uniprot"

# Selective AlphaFill columns to keep in LONG output
AFILL_LONG_COLS = [
    "reconstructed_key",
    "site_uniprot",
    "pdb_id",
    "compound_id",
    "analogue_id",
    "structure_pdb_asym_id",
    "mapping_global_rmsd",
    "mapping_align_identity",
    "mapping_align_length",
    "ligand_pdb_auth_asym_id",
    "ligand_pdb_auth_seq_id",
    "clash_count",
    "clash_score",
    "local_rmsd",
    "pae_mean",
    "pae_median",
]

# Numeric AlphaFill metrics to aggregate into SUMMARY output
AFILL_NUMERIC_METRICS = [
    "clash_count",
    "clash_score",
    "local_rmsd",
    "mapping_global_rmsd",
    "mapping_align_identity",
    "pae_mean",
    "pae_median",
]

ROUND_DECIMALS = 3


# ----------------------------
# CLI (optional)
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge unified AHoJ table with AlphaFill using job-key matching.")
    p.add_argument("--unified_csv", type=Path, default=UNIFIED_CSV_PATH)
    p.add_argument("--alphafill_csv", type=Path, default=AFILL_CSV_PATH)
    p.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--row_limit", type=int, default=ROW_LIMIT)
    p.add_argument("--write_jsonl_gz", action="store_true", default=WRITE_JSONL_GZ)
    return p.parse_args()


# ----------------------------
# Generic helpers
# ----------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer", low_memory=False)

def _to_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)

def normalize_job_key(key: Any) -> str:
    """
    Normalize a job-like key to AHoJ style:
      - convert '_' to '-'
      - strip surrounding whitespace
      - collapse duplicate '-'
      - uppercase
    Example: 4b11_c_7i1_999 -> 4B11-C-7I1-999
    """
    s = _to_str(key).strip()
    if not s:
        return ""
    s = s.replace("_", "-")
    while "--" in s:
        s = s.replace("--", "-")
    return s.upper()

def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    od = OrderedDict()
    for x in items:
        if x and x not in od:
            od[x] = None
    return list(od.keys())

def split_jobs(job_names: Any, rep_job: Any = None) -> list[str]:
    """
    Parse semicolon-separated AHoJ jobs from job_names.
    Add rep_job if present and not already included.
    Returns normalized jobs, deduplicated, order-preserved.
    """
    jobs = []
    s = _to_str(job_names).strip()
    if s:
        jobs.extend(normalize_job_key(p) for p in s.split(";") if p and str(p).strip())
    rep = normalize_job_key(rep_job)
    if rep:
        jobs.append(rep)
    return dedupe_preserve_order(jobs)

def split_uniprots(s: Any, sep: str | None = " ") -> list[str]:
    ss = _to_str(s).strip()
    if not ss:
        return []
    if sep is None:
        toks = ss.split()
    else:
        toks = [t.strip() for t in ss.split(sep)]
    return [t.upper() for t in toks if t]

def join_sorted_unique(values: Iterable[Any], sep: str = " ") -> str:
    vals = sorted({str(v) for v in values if pd.notna(v) and str(v) != ""})
    return sep.join(vals)

def ordered_join(values: Iterable[str], sep: str = ";") -> str:
    vals = dedupe_preserve_order([str(v) for v in values if pd.notna(v) and str(v) != ""])
    return sep.join(vals)

def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return list(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

def round_or_nan(x: Any, ndigits: int = ROUND_DECIMALS):
    if pd.isna(x):
        return np.nan
    return round(float(x), ndigits)


# ----------------------------
# Unified-table compatibility helpers
# ----------------------------

def add_support_compat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lightweight compatibility columns that may help downstream analyses.
    Uses the new unified schema as source of truth.
    """
    df = df.copy()

    if "winning_class" not in df.columns and "pocket_class" in df.columns:
        df["winning_class"] = df["pocket_class"]

    if "winning_p" not in df.columns and {"pocket_p_apo", "pocket_p_holo"}.issubset(df.columns):
        p_apo = pd.to_numeric(df["pocket_p_apo"], errors="coerce")
        p_holo = pd.to_numeric(df["pocket_p_holo"], errors="coerce")
        df["winning_p"] = pd.concat([p_apo, p_holo], axis=1).max(axis=1)

    if "pair_density_mean_pocket" not in df.columns and "pocket_density_pair" in df.columns:
        df["pair_density_mean_pocket"] = pd.to_numeric(df["pocket_density_pair"], errors="coerce")

    if "mean_separation" not in df.columns and "pocket_overlap_pair" in df.columns:
        overlap = pd.to_numeric(df["pocket_overlap_pair"], errors="coerce")
        df["mean_separation"] = 1.0 - overlap

    return df


# ----------------------------
# Core preparation
# ----------------------------

def prepare_unified_df(df: pd.DataFrame, row_limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      base_df    : unified table with added merge bookkeeping columns
      jobs_df    : one row per (merge_row_id, AHoJ job)
    """
    df = df.copy()
    if row_limit is not None:
        df = df.iloc[:row_limit].copy()

    df = df.reset_index(drop=False).rename(columns={"index": "source_row_index"})
    df.insert(0, "merge_row_id", np.arange(len(df), dtype=int))

    df = add_support_compat_columns(df)

    # Parse jobs per unified row and compute row-level job coverage later
    job_rows: list[dict[str, Any]] = []
    all_ahoj_jobs: set[str] = set()

    # Precompute normalized rep jobs
    rep_jobs_norm = []
    for row in df.itertuples(index=False):
        jobs = split_jobs(getattr(row, JOB_NAMES_COL, ""), getattr(row, REP_JOB_COL, ""))
        rep_norm = normalize_job_key(getattr(row, REP_JOB_COL, ""))
        rep_jobs_norm.append(rep_norm)

        for order_idx, job in enumerate(jobs, start=1):
            job_rows.append({
                "merge_row_id": getattr(row, "merge_row_id"),
                "ahoj_job": job,
                "ahoj_job_order": order_idx,
                "ahoj_job_is_rep": bool(rep_norm and job == rep_norm),
            })
            all_ahoj_jobs.add(job)

    df["rep_job_norm"] = rep_jobs_norm
    jobs_df = pd.DataFrame(job_rows)
    if jobs_df.empty:
        jobs_df = pd.DataFrame(columns=["merge_row_id", "ahoj_job", "ahoj_job_order", "ahoj_job_is_rep"])

    return df, jobs_df


def prepare_alphafill_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns needed for long output / summary aggregation,
    and add normalized AlphaFill job key.
    """
    keep_cols = [c for c in AFILL_LONG_COLS if c in df.columns]
    required = []
    if AFILL_KEY_COL in df.columns and AFILL_KEY_COL not in keep_cols:
        required.append(AFILL_KEY_COL)
    if AFILL_UNIPROT_COL in df.columns and AFILL_UNIPROT_COL not in keep_cols:
        keep_cols.append(AFILL_UNIPROT_COL)

    af = df[keep_cols + required].copy()
    af = af.reset_index(drop=False).rename(columns={"index": "afill_row_index"})
    af["afill_job"] = af[AFILL_KEY_COL].map(normalize_job_key) if AFILL_KEY_COL in af.columns else ""

    # Prefix selective AlphaFill columns for long output to avoid collisions
    rename_map = {}
    for c in keep_cols:
        rename_map[c] = f"afill_{c}"
    af = af.rename(columns=rename_map)

    return af


# ----------------------------
# Summary builders
# ----------------------------

def add_row_job_coverage(base_df: pd.DataFrame, jobs_df: pd.DataFrame, af_job_set: set[str]) -> pd.DataFrame:
    """
    Add row-level job coverage columns to the unified/base table.
    """
    base_df = base_df.copy()

    # Fast row-wise coverage using the exploded jobs table
    jobs_df = jobs_df.copy()
    if jobs_df.empty:
        base_df["afill_n_jobs_total"] = 0
        base_df["afill_n_jobs_matched"] = 0
        base_df["afill_n_jobs_unmatched"] = 0
        base_df["afill_frac_jobs_matched"] = np.nan
        base_df["afill_rep_job_matched"] = False
        base_df["afill_matched_jobs_str"] = ""
        return base_df

    jobs_df["job_matched"] = jobs_df["ahoj_job"].isin(af_job_set)
    jobs_df["rep_job_matched"] = jobs_df["ahoj_job_is_rep"] & jobs_df["job_matched"]

    jobs_df = jobs_df.sort_values(["merge_row_id", "ahoj_job_order"]).copy()
    grp = jobs_df.groupby("merge_row_id", sort=False)

    coverage = grp.agg(
        afill_n_jobs_total=("ahoj_job", "size"),
        afill_n_jobs_matched=("job_matched", "sum"),
        afill_rep_job_matched=("rep_job_matched", "any"),
    ).reset_index()

    coverage["afill_n_jobs_unmatched"] = coverage["afill_n_jobs_total"] - coverage["afill_n_jobs_matched"]
    coverage["afill_frac_jobs_matched"] = coverage["afill_n_jobs_matched"] / coverage["afill_n_jobs_total"]

    matched_jobs = (
        jobs_df.loc[jobs_df["job_matched"], ["merge_row_id", "ahoj_job"]]
        .groupby("merge_row_id", sort=False)["ahoj_job"]
        .agg(lambda s: ordered_join(s, sep=";"))
        .rename("afill_matched_jobs_str")
    )

    coverage = coverage.merge(matched_jobs.reset_index(), on="merge_row_id", how="left")

    base_df = base_df.merge(coverage, on="merge_row_id", how="left")

    # Rows with no jobs (rare) -> fill sensible defaults
    for c in ["afill_n_jobs_total", "afill_n_jobs_matched", "afill_n_jobs_unmatched"]:
        base_df[c] = base_df[c].fillna(0).astype(int)
    base_df["afill_frac_jobs_matched"] = base_df["afill_frac_jobs_matched"].astype(float)
    base_df["afill_rep_job_matched"] = base_df["afill_rep_job_matched"].fillna(False).astype(bool)
    base_df["afill_matched_jobs_str"] = base_df["afill_matched_jobs_str"].fillna("")

    return base_df


def build_long_df(base_df: pd.DataFrame, jobs_df: pd.DataFrame, af_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return matched long table:
      one row per (unified row × matched job × AlphaFill row)
    Keeps all original unified columns plus selective prefixed AlphaFill columns.
    """
    if jobs_df.empty or af_df.empty:
        return pd.DataFrame(columns=list(base_df.columns) + ["ahoj_job", "ahoj_job_order", "ahoj_job_is_rep"])

    # Inner join -> matched jobs only
    long_matches = jobs_df.merge(af_df, left_on="ahoj_job", right_on="afill_job", how="inner")

    # Add compact bookkeeping columns
    long_matches["afill_matched_job"] = long_matches["ahoj_job"]
    long_matches["afill_matched_job_is_rep"] = long_matches["ahoj_job_is_rep"]

    # Merge all original unified/base columns
    long_df = base_df.merge(long_matches, on="merge_row_id", how="inner")

    # Cross-UniProt annotation (reference only, not used for matching)
    if UNIPROTS_COL in long_df.columns and f"afill_{AFILL_UNIPROT_COL}" in long_df.columns:
        def _cross_up(row) -> bool:
            base_ups = set(split_uniprots(row.get(UNIPROTS_COL, "")))
            af_up = _to_str(row.get(f"afill_{AFILL_UNIPROT_COL}", "")).upper().strip()
            return bool(af_up) and (af_up not in base_ups)
        long_df["afill_cross_uniprot"] = long_df.apply(_cross_up, axis=1)
    else:
        long_df["afill_cross_uniprot"] = False

    return long_df


def add_alphafill_aggregates_to_summary(base_df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate matched AlphaFill rows per unified row and append the result to summary/base_df.
    """
    base_df = base_df.copy()

    if long_df.empty:
        base_df["afill_n_transplants_matched"] = 0
        base_df["afill_n_uniprots"] = 0
        base_df["afill_uniprots_str"] = ""
        base_df["afill_cross_uniprot"] = False
        for metric in AFILL_NUMERIC_METRICS:
            base_df[f"afill_{metric}_mean"] = np.nan
            base_df[f"afill_{metric}_median"] = np.nan
        return base_df

    grp = long_df.groupby("merge_row_id", sort=False)

    agg = grp.size().rename("afill_n_transplants_matched").to_frame()

    # Distinct AlphaFill UniProts seen for the matched rows
    af_up_col = f"afill_{AFILL_UNIPROT_COL}"
    if af_up_col in long_df.columns:
        up_str = grp[af_up_col].agg(lambda s: join_sorted_unique(s, sep=" ")).rename("afill_uniprots_str")
        agg = agg.join(up_str, how="left")
        agg["afill_n_uniprots"] = agg["afill_uniprots_str"].fillna("").map(lambda s: 0 if not s else len(s.split()))
    else:
        agg["afill_uniprots_str"] = ""
        agg["afill_n_uniprots"] = 0

    # Numeric metric aggregates
    for metric in AFILL_NUMERIC_METRICS:
        col = f"afill_{metric}"
        if col not in long_df.columns:
            agg[f"afill_{metric}_mean"] = np.nan
            agg[f"afill_{metric}_median"] = np.nan
            continue
        ser = pd.to_numeric(long_df[col], errors="coerce")
        tmp = pd.DataFrame({"merge_row_id": long_df["merge_row_id"], col: ser})
        g2 = tmp.groupby("merge_row_id", sort=False)[col]
        agg[f"afill_{metric}_mean"] = g2.mean().round(ROUND_DECIMALS)
        agg[f"afill_{metric}_median"] = g2.median().round(ROUND_DECIMALS)

    summary_df = base_df.merge(agg.reset_index(), on="merge_row_id", how="left")

    # Fill match-dependent defaults
    summary_df["afill_n_transplants_matched"] = summary_df["afill_n_transplants_matched"].fillna(0).astype(int)
    summary_df["afill_n_uniprots"] = summary_df["afill_n_uniprots"].fillna(0).astype(int)
    summary_df["afill_uniprots_str"] = summary_df["afill_uniprots_str"].fillna("")

    # Cross-UniProt annotation at row level
    base_ups_list = summary_df[UNIPROTS_COL].fillna("").astype(str).tolist() if UNIPROTS_COL in summary_df.columns else [""] * len(summary_df)
    af_ups_list = summary_df["afill_uniprots_str"].fillna("").astype(str).tolist()

    cross_flags = []
    for base_ups_s, af_ups_s in zip(base_ups_list, af_ups_list):
        base_ups = set(split_uniprots(base_ups_s, sep=" "))
        af_ups = set(split_uniprots(af_ups_s, sep=" "))
        cross_flags.append(bool(af_ups - base_ups))

    summary_df["afill_cross_uniprot"] = cross_flags

    return summary_df


# ----------------------------
# Coverage reporting
# ----------------------------

def build_coverage_by_level_threshold(summary_df: pd.DataFrame) -> pd.DataFrame:
    level_col = LEVEL_COL if LEVEL_COL in summary_df.columns else None
    th_col = THRESHOLD_COL if THRESHOLD_COL in summary_df.columns else None

    group_cols = []
    if level_col:
        group_cols.append(level_col)
    if th_col:
        group_cols.append(th_col)

    if not group_cols:
        return pd.DataFrame()

    work = summary_df.copy()
    work["rows_any_match"] = work["afill_n_jobs_matched"] > 0
    work["rows_all_jobs_matched"] = work["afill_n_jobs_total"].gt(0) & work["afill_n_jobs_unmatched"].eq(0)
    work["rows_partial_jobs_matched"] = work["afill_n_jobs_matched"].gt(0) & work["afill_n_jobs_unmatched"].gt(0)
    work["rows_no_match"] = work["afill_n_jobs_matched"].eq(0)
    work["afill_has_cross_uniprot"] = work["afill_cross_uniprot"].fillna(False).astype(bool)

    cov = work.groupby(group_cols, dropna=False, sort=False).agg(
        n_rows=("merge_row_id", "size"),
        rows_any_match=("rows_any_match", "sum"),
        rows_no_match=("rows_no_match", "sum"),
        rows_all_jobs_matched=("rows_all_jobs_matched", "sum"),
        rows_partial_jobs_matched=("rows_partial_jobs_matched", "sum"),
        sum_jobs_total=("afill_n_jobs_total", "sum"),
        sum_jobs_matched=("afill_n_jobs_matched", "sum"),
        sum_jobs_unmatched=("afill_n_jobs_unmatched", "sum"),
        sum_transplants_matched=("afill_n_transplants_matched", "sum"),
        rows_cross_uniprot=("afill_has_cross_uniprot", "sum"),
    ).reset_index()

    cov["frac_rows_any_match"] = (cov["rows_any_match"] / cov["n_rows"]).round(ROUND_DECIMALS)
    cov["frac_jobs_matched"] = np.where(
        cov["sum_jobs_total"] > 0,
        (cov["sum_jobs_matched"] / cov["sum_jobs_total"]).round(ROUND_DECIMALS),
        np.nan,
    )
    return cov


def write_report(
    report_path: Path,
    summary_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    all_ahoj_jobs: set[str],
    all_af_jobs: set[str],
) -> None:
    matched_jobs = all_ahoj_jobs & all_af_jobs
    ahoj_only_jobs = all_ahoj_jobs - all_af_jobs
    af_only_jobs = all_af_jobs - all_ahoj_jobs

    n_rows = len(summary_df)
    rows_any_match = int((summary_df["afill_n_jobs_matched"] > 0).sum())
    rows_no_match = int((summary_df["afill_n_jobs_matched"] == 0).sum())
    rows_all_jobs = int(((summary_df["afill_n_jobs_total"] > 0) & (summary_df["afill_n_jobs_unmatched"] == 0)).sum())
    rows_partial = int(((summary_df["afill_n_jobs_matched"] > 0) & (summary_df["afill_n_jobs_unmatched"] > 0)).sum())
    rows_cross_up = int(summary_df["afill_cross_uniprot"].fillna(False).sum())

    lines = []
    lines.append("AHoJ <-> AlphaFill merge report")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Merge rule")
    lines.append("----------")
    lines.append("Job-to-job matching only; UniProt is reference/QA metadata only.")
    lines.append("AlphaFill reconstructed_key is normalized from '_' to '-' before matching.")
    lines.append("")
    lines.append("Overall row coverage")
    lines.append("--------------------")
    lines.append(f"Unified rows total              : {n_rows}")
    lines.append(f"Unified rows with any AF match  : {rows_any_match}")
    lines.append(f"Unified rows with no AF match   : {rows_no_match}")
    lines.append(f"Rows with all jobs matched      : {rows_all_jobs}")
    lines.append(f"Rows with partial job coverage  : {rows_partial}")
    lines.append(f"Rows with cross-UniProt AF hits : {rows_cross_up}")
    lines.append("")
    lines.append("Overall job coverage")
    lines.append("--------------------")
    lines.append(f"Distinct AHoJ jobs total        : {len(all_ahoj_jobs)}")
    lines.append(f"Distinct AF jobs total          : {len(all_af_jobs)}")
    lines.append(f"Distinct jobs in both           : {len(matched_jobs)}")
    lines.append(f"AHoJ jobs missing in AF         : {len(ahoj_only_jobs)}")
    lines.append(f"AF jobs missing in AHoJ         : {len(af_only_jobs)}")
    lines.append("")
    lines.append("By level / threshold")
    lines.append("--------------------")
    if coverage_df.empty:
        lines.append("(coverage table not available)")
    else:
        lines.append(coverage_df.to_string(index=False))
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_jsonl_gz(jsonl_gz_path: Path, summary_df: pd.DataFrame, long_df: pd.DataFrame) -> None:
    """
    Debug-only artifact.
    One object per unified row with compact matched AlphaFill records.
    """
    selected_long_cols = [
        "merge_row_id",
        "ahoj_job",
        "ahoj_job_order",
        "ahoj_job_is_rep",
        "afill_row_index",
        "afill_job",
        "afill_matched_job",
        "afill_matched_job_is_rep",
        "afill_cross_uniprot",
    ] + [c for c in long_df.columns if c.startswith("afill_") and c not in {"afill_cross_uniprot", "afill_job", "afill_matched_job", "afill_matched_job_is_rep"}]

    long_compact = long_df[[c for c in selected_long_cols if c in long_df.columns]].copy()

    grouped = {}
    if not long_compact.empty:
        for rid, grp in long_compact.groupby("merge_row_id", sort=False):
            grouped[int(rid)] = grp.drop(columns=["merge_row_id"], errors="ignore").to_dict(orient="records")

    with gzip.open(jsonl_gz_path, "wt", encoding="utf-8") as fh:
        for row in summary_df.to_dict(orient="records"):
            rid = int(row["merge_row_id"])
            rec = {
                "summary": row,
                "matches": grouped.get(rid, []),
            }
            fh.write(json.dumps(rec, default=_json_default, ensure_ascii=False) + "\n")


# ----------------------------
# Main
# ----------------------------

def main(
    unified_csv: Path,
    alphafill_csv: Path,
    output_dir: Path,
    row_limit: int | None = None,
    write_jsonl_gz_flag: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading unified table...")
    unified_df = _load_csv(unified_csv)
    print(f"  Unified rows loaded: {len(unified_df):,}")

    print("[2/7] Loading AlphaFill parsed table...")
    afill_raw_df = _load_csv(alphafill_csv)
    print(f"  AlphaFill rows loaded: {len(afill_raw_df):,}")

    print("[3/7] Preparing tables...")
    base_df, jobs_df = prepare_unified_df(unified_df, row_limit=row_limit)
    af_df = prepare_alphafill_df(afill_raw_df)

    all_ahoj_jobs = set(jobs_df["ahoj_job"].dropna().astype(str)) if not jobs_df.empty else set()
    all_af_jobs = {x for x in af_df["afill_job"].dropna().astype(str) if x} if not af_df.empty else set()

    print(f"  Unified rows in run : {len(base_df):,}")
    print(f"  Distinct AHoJ jobs  : {len(all_ahoj_jobs):,}")
    print(f"  Distinct AF jobs    : {len(all_af_jobs):,}")

    print("[4/7] Computing row-level job coverage...")
    summary_seed_df = add_row_job_coverage(base_df, jobs_df, all_af_jobs)

    print("[5/7] Building matched long table...")
    long_df = build_long_df(base_df, jobs_df, af_df)
    print(f"  Long matched rows   : {len(long_df):,}")

    print("[6/7] Aggregating AlphaFill metrics to summary...")
    summary_df = add_alphafill_aggregates_to_summary(summary_seed_df, long_df)
    coverage_df = build_coverage_by_level_threshold(summary_df)

    print("[7/7] Writing outputs...")
    if WRITE_SUMMARY_CSV:
        summary_path = output_dir / OUT_SUMMARY_CSV
        summary_df.to_csv(summary_path, index=False)
        print(f"  Wrote summary CSV   : {summary_path}")

    if WRITE_LONG_CSV:
        long_path = output_dir / OUT_LONG_CSV
        long_df.to_csv(long_path, index=False)
        print(f"  Wrote long CSV      : {long_path}")

    if WRITE_COVERAGE_CSV and not coverage_df.empty:
        cov_path = output_dir / OUT_COVERAGE_CSV
        coverage_df.to_csv(cov_path, index=False)
        print(f"  Wrote coverage CSV  : {cov_path}")

    if WRITE_REPORT:
        report_path = output_dir / OUT_REPORT_TXT
        write_report(report_path, summary_df, coverage_df, all_ahoj_jobs, all_af_jobs)
        print(f"  Wrote report TXT    : {report_path}")

    if write_jsonl_gz_flag:
        jsonl_gz_path = output_dir / OUT_JSONL_GZ
        write_jsonl_gz(jsonl_gz_path, summary_df, long_df)
        print(f"  Wrote JSONL.GZ      : {jsonl_gz_path}")

    print("Done.")


if __name__ == "__main__":
    if USE_CLI:
        args = parse_args()
        main(
            unified_csv=args.unified_csv,
            alphafill_csv=args.alphafill_csv,
            output_dir=args.output_dir,
            row_limit=args.row_limit,
            write_jsonl_gz_flag=args.write_jsonl_gz,
        )
    else:
        main(
            unified_csv=UNIFIED_CSV_PATH,
            alphafill_csv=AFILL_CSV_PATH,
            output_dir=OUTPUT_DIR,
            row_limit=ROW_LIMIT,
            write_jsonl_gz_flag=WRITE_JSONL_GZ,
        )
