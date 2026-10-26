# -*- coding: utf-8 -*-
"""
AHoJ / AFDB — analysis-set inclusion filter (survivor filter)
=============================================================

Horizontal filtering of the per-query apo/holo classification summary
(``query_summaries_do_kde_plots_and_classification.csv``) down to the
**analysis set** consumed by the rest of this repo (stages 01+).

This is an ANALYSIS decision, not a classification-method one: the thresholds
below (>=1 experimental apo, usable pocket classification, query resolution
<= 20 Angstrom, has_model==1) are the study's analysis-set inclusion criteria —
they set the manuscript's headline N. The apo/holo *classification* itself is a
feature of AHoJ-DB and lives in a separate methodology repository (see the
README in this directory); this script only selects which classified pockets
enter the downstream analyses. It was relocated here (from the methodology repo,
where it was ``filter_KDE_classifications.py``) so the analysis-set definition
lives with the analysis code.

What it does
------------
- Loads an input CSV OR a ZIP containing a single CSV (or a chosen one).
- Applies a fixed set of threshold filters (in a fixed order).
- Saves:
    - filtered_survivors.csv (rows that pass all enabled filters)
    - filtered_out_XX_<reason>.csv (rows removed by each step)
    - filtered_out_all.csv (all removed rows with a reason column)
    - annotated_with_filter_flags.csv (optional; original rows + pass/fail flags + fail reasons)
    - filter_report.csv (+ a short .txt summary)
    - manifest.json (thresholds + run metadata)
    - run.log

Config wiring
-------------
- INPUT  defaults to ``cfg.CLASSIFICATION_SUMMARIES_CSV`` (the merged run_3
  classification summary, under the AHoJ-DB derived tree).
- OUTPUT is written to a fresh ``filtering_<timestamp>/`` directory next to the
  input (i.e. inside ``cfg.CLASSIFICATION_MERGED_DIR``, the derived tree). After a
  run, point ``cfg.FILTERED_SURVIVORS_DIR`` at the produced timestamped directory
  so stages 01+ consume it. The canonical run_3 output is
  ``filtering_20260704_152000`` (237,922 survivors / 12,255 UniProts).
- Override either via the ``INPUT_PATH`` / ``OUTDIR_OVERRIDE`` globals below
  (Spyder-friendly) or the ``AHOJ_AF_FILTER_INPUT`` / ``AHOJ_AF_FILTER_OUTDIR``
  environment variables.

Notes
-----
- Unique UniProt counting is done by splitting the "UNPs" column on whitespace,
  because entries may contain multiple UNPs joined by " " (interface sites).
- Filtering is sequential (order defined below) BUT the script also computes
  "independent" per-filter fail counts on the original dataset (in the report),
  so you can see each filter's impact regardless of order.
"""

#from __future__ import annotations

from pathlib import Path
import datetime as dt
import hashlib
import json
import logging
import os
import zipfile
import pandas as pd
from typing import Dict, List, Optional, Tuple

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg


# =============================================================================
# USER SETTINGS (edit these in Spyder)
# =============================================================================

# Input can be a .csv OR a .zip containing a csv. Defaults to the configured
# merged classification summary; override here or via AHOJ_AF_FILTER_INPUT.
INPUT_PATH = os.environ.get("AHOJ_AF_FILTER_INPUT") or str(cfg.CLASSIFICATION_SUMMARIES_CSV)

# If INPUT_PATH is a zip with multiple CSVs, set the csv filename inside the zip.
# If None, the script will auto-pick the single CSV in the zip.
CSV_IN_ZIP_NAME: Optional[str] = None

# Output directory. By default a fresh "<OUTDIR_BASENAME_PREFIX><timestamp>" dir is
# created next to INPUT_PATH (inside the derived tree). Set OUTDIR_OVERRIDE (or the
# AHOJ_AF_FILTER_OUTDIR env var) to write somewhere specific instead.
OUTDIR_BASENAME_PREFIX = "filtering_"
OUTDIR_OVERRIDE: Optional[str] = os.environ.get("AHOJ_AF_FILTER_OUTDIR") or None

# Core thresholds
MIN_N_APO: Optional[int] = 1
MIN_N_HOLO: Optional[int] = None # e.g. 1

REQUIRE_POCKET_CLASSIFICATION: bool = True
REQUIRE_CHAIN_CLASSIFICATION: bool = False

# AHOJ-018: full_run_3+ query_summaries carry a `has_model` flag (0 = no-model conformational-only
# entry with an "N/A (no model)" verdict and NaN probabilities). Drop these for the apo/holo analysis.
# Gated on column presence, so pre-run_3 CSVs (no such column) are unaffected. Note: these rows are
# also removed by REQUIRE_POCKET_CLASSIFICATION (NaN pocket probs) — this makes the intent explicit and
# gives them their own clean removed bucket.
REQUIRE_HAS_MODEL: bool = True
COL_HAS_MODEL = "has_model"

MAX_QUERY_RESOLUTION: Optional[float] = 20.0  # e.g. 5.0; if set, require resolution <= this AND non-empty
COL_QUERY_RESOLUTION = "query_resolution"

# Optional numeric thresholds on max probability
# Example: 0.60 means max(P_apo, P_holo) must be >= 0.60
MIN_POCKET_MAX_PROB: Optional[float] = None   # e.g. 0.60
MIN_CHAIN_MAX_PROB: Optional[float] = None    # e.g. 0.60

# Save annotated full dataset with pass/fail flags + fail reasons?
SAVE_ANNOTATED_ALL_ROWS: bool = True

# Column names (edit if your CSV schema changes)
COL_N_APO = "n_apo"
COL_N_HOLO = "n_holo"
COL_UNPS = "UNPs"

POCKET_PROB_COLS = ("pocket_avg_P(apo)", "pocket_avg_P(holo)")
CHAIN_PROB_COLS = ("chain_avg_P(apo)", "chain_avg_P(holo)")

# CSV writing options
CSV_SEP = ","
CSV_FLOAT_FORMAT = None  # e.g. "%.6f" if you want consistent formatting


# =============================================================================
# HELPERS
# =============================================================================

def _now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns in input CSV:\n"
            f"  Missing: {missing}\n"
            f"  Present: {list(df.columns)}"
        )


def unique_unps_count(unps_series: pd.Series) -> int:
    """
    Count unique UniProt accessions by splitting each cell on whitespace.
    """
    s = set()
    for val in unps_series.astype(str):
        for tok in val.split():
            if tok:
                s.add(tok)
    return len(s)


def read_input_dataframe(input_path: Path, csv_in_zip_name: Optional[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns (df, meta) where meta includes source details.
    """
    meta: Dict = {
        "input_path": str(input_path),
        "input_is_zip": input_path.suffix.lower() == ".zip",
    }

    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as zf:
            csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]

            if csv_in_zip_name is not None:
                if csv_in_zip_name not in zf.namelist():
                    raise FileNotFoundError(
                        f"CSV_IN_ZIP_NAME='{csv_in_zip_name}' not found in zip. "
                        f"Available CSVs: {csv_members}"
                    )
                chosen = csv_in_zip_name
            else:
                if len(csv_members) != 1:
                    raise ValueError(
                        "ZIP contains multiple CSVs. Set CSV_IN_ZIP_NAME to one of:\n"
                        f"{csv_members}"
                    )
                chosen = csv_members[0]

            meta["zip_csv_name"] = chosen

            with zf.open(chosen) as f:
                df = pd.read_csv(f, sep=CSV_SEP, low_memory=False)

    else:
        df = pd.read_csv(input_path, sep=CSV_SEP, low_memory=False)

    meta["rows"] = int(df.shape[0])
    meta["cols"] = int(df.shape[1])
    return df, meta


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    """
    In-place: ensure selected columns are numeric (non-numeric becomes NaN).
    Useful if a downstream CSV saved floats as strings.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def build_filter_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Build per-filter boolean masks on the ORIGINAL df.
    Each mask is True for rows that PASS that single criterion.
    """
    masks: Dict[str, pd.Series] = {}

    if REQUIRE_HAS_MODEL and COL_HAS_MODEL in df.columns:
        masks["has_model"] = pd.to_numeric(df[COL_HAS_MODEL], errors="coerce") == 1
    else:
        masks["has_model"] = pd.Series(True, index=df.index)

    if MIN_N_APO is not None:
        masks["min_n_apo"] = df[COL_N_APO] >= MIN_N_APO
    else:
        masks["min_n_apo"] = pd.Series(True, index=df.index)

    if MIN_N_HOLO is not None:
        masks["min_n_holo"] = df[COL_N_HOLO] >= MIN_N_HOLO
    else:
        masks["min_n_holo"] = pd.Series(True, index=df.index)

    if MAX_QUERY_RESOLUTION is not None:
        masks["max_query_resolution"] = (df[COL_QUERY_RESOLUTION].notna() & (df[COL_QUERY_RESOLUTION] <= MAX_QUERY_RESOLUTION))
    else:
        masks["max_query_resolution"] = pd.Series(True, index=df.index)

    if REQUIRE_POCKET_CLASSIFICATION:
        masks["has_pocket_probs"] = df[list(POCKET_PROB_COLS)].notna().all(axis=1)
    else:
        masks["has_pocket_probs"] = pd.Series(True, index=df.index)

    if REQUIRE_CHAIN_CLASSIFICATION:
        masks["has_chain_probs"] = df[list(CHAIN_PROB_COLS)].notna().all(axis=1)
    else:
        masks["has_chain_probs"] = pd.Series(True, index=df.index)

    if MIN_POCKET_MAX_PROB is not None:
        masks["min_pocket_max_prob"] = df[list(POCKET_PROB_COLS)].max(axis=1) >= MIN_POCKET_MAX_PROB
    else:
        masks["min_pocket_max_prob"] = pd.Series(True, index=df.index)

    if MIN_CHAIN_MAX_PROB is not None:
        masks["min_chain_max_prob"] = df[list(CHAIN_PROB_COLS)].max(axis=1) >= MIN_CHAIN_MAX_PROB
    else:
        masks["min_chain_max_prob"] = pd.Series(True, index=df.index)

    return masks


def add_filter_flag_columns(df: pd.DataFrame, masks: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Returns a new dataframe with pass_* columns and fail reason strings.
    """
    out = df.copy()
    for k, m in masks.items():
        out[f"pass_{k}"] = m

    pass_all = pd.Series(True, index=df.index)
    for m in masks.values():
        pass_all = pass_all & m
    out["pass_all_filters"] = pass_all

    # Build fail reasons (order matters, matches sequential filtering order)
    ordered_keys = [
        "has_model",
        "has_pocket_probs",
        "has_chain_probs",
        "min_n_apo",
        "min_n_holo",
        "max_query_resolution",
        "min_pocket_max_prob",
        "min_chain_max_prob",
    ]

    def _fail_reasons_row(i) -> str:
        reasons = [k for k in ordered_keys if not bool(out.loc[i, f"pass_{k}"])]
        return "|".join(reasons) if reasons else ""

    # Efficient-ish: apply over index (acceptable for ~0.5M rows)
    out["filter_fail_reasons"] = [ _fail_reasons_row(i) for i in out.index ]
    out["filter_primary_fail_reason"] = out["filter_fail_reasons"].str.split("|").str[0].fillna("")
    return out


def setup_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run.log"

    logger = logging.getLogger()

    # IMPORTANT for Spyder / repeated runs: close old handlers to release file locks
    for h in list(logger.handlers):
        try:
            h.flush()
            h.close()
        finally:
            logger.removeHandler(h)

    logger.setLevel(logging.INFO)


    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, sep=CSV_SEP, float_format=CSV_FLOAT_FORMAT)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    input_path = Path(INPUT_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"INPUT_PATH not found: {input_path}")

    if OUTDIR_OVERRIDE:
        outdir = Path(OUTDIR_OVERRIDE)
    else:
        outdir = input_path.parent / f"{OUTDIR_BASENAME_PREFIX}{_now_stamp()}"
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir)

    logging.info("Starting filtering run")
    logging.info(f"Input: {input_path}")
    logging.info(f"Output dir: {outdir}")

    # Load
    df, meta = read_input_dataframe(input_path, CSV_IN_ZIP_NAME)

    required_cols = [
        COL_N_APO, COL_N_HOLO, COL_UNPS,
        POCKET_PROB_COLS[0], POCKET_PROB_COLS[1],
        CHAIN_PROB_COLS[0], CHAIN_PROB_COLS[1],
    ]
    ensure_columns(df, required_cols)

    # Coerce numeric (robustness)
    coerce_numeric(df, [COL_N_APO, COL_N_HOLO, *POCKET_PROB_COLS, *CHAIN_PROB_COLS, COL_QUERY_RESOLUTION])

    logging.info(f"Loaded dataframe: {df.shape[0]} rows x {df.shape[1]} cols")

    # Build masks on original df (independent)
    masks = build_filter_masks(df)

    # Optional annotated full dataset
    if SAVE_ANNOTATED_ALL_ROWS:
        logging.info("Creating annotated dataset with filter flags...")
        df_annot = add_filter_flag_columns(df, masks)
        annotated_path = outdir / "annotated_with_filter_flags.csv"
        save_csv(df_annot, annotated_path)
        logging.info(f"Saved: {annotated_path.name}")

    # Sequential filtering (for disjoint per-step removed groups)
    ordered_steps = [
        ("00_no_model", "has_model"),
        ("01_missing_pocket_probs", "has_pocket_probs"),
        ("02_missing_chain_probs", "has_chain_probs"),
        ("03_min_n_apo", "min_n_apo"),
        ("04_min_n_holo", "min_n_holo"),
        ("05_max_query_resolution", "max_query_resolution"),
        ("06_low_pocket_max_prob", "min_pocket_max_prob"),
        ("07_low_chain_max_prob", "min_chain_max_prob"),
    ]

    df_current = df.copy()
    removed_parts: List[pd.DataFrame] = []

    report_rows = []

    # Summary before filtering
    total_rows = int(df_current.shape[0])
    total_unique_unps = unique_unps_count(df_current[COL_UNPS])
    logging.info(f"Initial rows: {total_rows} | unique UNPs: {total_unique_unps}")

    # Independent (original) per-filter stats
    independent_stats = {}
    for key, m in masks.items():
        fail = ~m
        independent_stats[key] = {
            "orig_fail_rows": int(fail.sum()),
            "orig_fail_unique_unps": unique_unps_count(df.loc[fail, COL_UNPS]),
            "orig_pass_rows": int(m.sum()),
            "orig_pass_unique_unps": unique_unps_count(df.loc[m, COL_UNPS]),
        }

    for step_label, mask_key in ordered_steps:
        step_mask = masks[mask_key].reindex(df_current.index)

        kept = df_current.loc[step_mask].copy()
        removed = df_current.loc[~step_mask].copy()

        removed_rows = int(removed.shape[0])
        kept_rows = int(kept.shape[0])

        removed_unique = unique_unps_count(removed[COL_UNPS]) if removed_rows else 0
        kept_unique = unique_unps_count(kept[COL_UNPS]) if kept_rows else 0

        logging.info(
            f"{step_label}: removed {removed_rows} (unique UNPs {removed_unique}) "
            f"| kept {kept_rows} (unique UNPs {kept_unique})"
        )

        # Save removed group for this step
        if removed_rows > 0:
            removed.insert(0, "filtered_out_reason", mask_key)
            out_path = outdir / f"filtered_out_{step_label}.csv"
            save_csv(removed, out_path)
            removed_parts.append(removed)

        # Append report row
        report_rows.append({
            "step": step_label,
            "criterion_key": mask_key,
            "sequential_removed_rows": removed_rows,
            "sequential_removed_unique_unps": removed_unique,
            "sequential_kept_rows": kept_rows,
            "sequential_kept_unique_unps": kept_unique,
            # independent stats on the original df (order-independent):
            **independent_stats.get(mask_key, {}),
        })

        df_current = kept

    # Save survivors
    survivors_path = outdir / "filtered_survivors.csv"
    save_csv(df_current, survivors_path)
    logging.info(f"Saved: {survivors_path.name} ({df_current.shape[0]} rows)")

    # Save all removed combined
    if removed_parts:
        df_removed_all = pd.concat(removed_parts, axis=0, ignore_index=True)
        removed_all_path = outdir / "filtered_out_all.csv"
        save_csv(df_removed_all, removed_all_path)
        logging.info(f"Saved: {removed_all_path.name} ({df_removed_all.shape[0]} rows)")
    else:
        logging.info("No rows were filtered out; skipping filtered_out_all.csv")

    # Report
    report_df = pd.DataFrame(report_rows)
    report_csv_path = outdir / "filter_report.csv"
    save_csv(report_df, report_csv_path)
    logging.info(f"Saved: {report_csv_path.name}")

    # Short text summary
    summary_txt = outdir / "filter_report_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("Filtering summary\n")
        f.write("=================\n\n")
        f.write(f"Input: {meta.get('input_path')}\n")
        if meta.get("input_is_zip"):
            f.write(f"ZIP member: {meta.get('zip_csv_name')}\n")
        f.write(f"Initial rows: {total_rows}\n")
        f.write(f"Initial unique UNPs: {total_unique_unps}\n\n")
        f.write(f"Survivors rows: {int(df_current.shape[0])}\n")
        f.write(f"Survivors unique UNPs: {unique_unps_count(df_current[COL_UNPS])}\n\n")
        f.write("Per-step sequential removals:\n")
        for r in report_rows:
            f.write(
                f"  {r['step']} ({r['criterion_key']}): "
                f"removed {r['sequential_removed_rows']} "
                f"(unique UNPs {r['sequential_removed_unique_unps']}), "
                f"kept {r['sequential_kept_rows']}\n"
            )
    logging.info(f"Saved: {summary_txt.name}")

    # Manifest
    manifest = {
        "run_timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "input": meta,
        "input_sha256": sha256_file(input_path),
        "thresholds": {
            "REQUIRE_HAS_MODEL": REQUIRE_HAS_MODEL,
            "MIN_N_APO": MIN_N_APO,
            "MIN_N_HOLO": MIN_N_HOLO,
            "MAX_QUERY_RESOLUTION": MAX_QUERY_RESOLUTION,
            "REQUIRE_POCKET_CLASSIFICATION": REQUIRE_POCKET_CLASSIFICATION,
            "REQUIRE_CHAIN_CLASSIFICATION": REQUIRE_CHAIN_CLASSIFICATION,
            "MIN_POCKET_MAX_PROB": MIN_POCKET_MAX_PROB,
            "MIN_CHAIN_MAX_PROB": MIN_CHAIN_MAX_PROB,
        },
        "columns": {
            "COL_N_APO": COL_N_APO,
            "COL_N_HOLO": COL_N_HOLO,
            "COL_QUERY_RESOLUTION": COL_QUERY_RESOLUTION,
            "COL_UNPS": COL_UNPS,
            "POCKET_PROB_COLS": POCKET_PROB_COLS,
            "CHAIN_PROB_COLS": CHAIN_PROB_COLS,
        },
        "outputs": {
            "outdir": str(outdir),
            "filtered_survivors": str(survivors_path),
            "filter_report_csv": str(report_csv_path),
        },
        "results": {
            "initial_rows": total_rows,
            "initial_unique_unps": total_unique_unps,
            "survivor_rows": int(df_current.shape[0]),
            "survivor_unique_unps": unique_unps_count(df_current[COL_UNPS]),
        },
        "pandas_version": pd.__version__,
    }

    # Try to save a copy of the running script for reproducibility (works if __file__ is defined)
    try:
        script_path = Path(__file__)
        script_copy_path = outdir / script_path.name
        if script_path.exists():
            script_copy_path.write_text(script_path.read_text(encoding="utf-8"), encoding="utf-8")
            manifest["outputs"]["script_copy"] = str(script_copy_path)
            logging.info(f"Saved script copy: {script_copy_path.name}")
    except Exception as e:
        logging.info(f"Could not save script copy (this is normal in some Spyder contexts): {e}")

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info(f"Saved: {manifest_path.name}")

    logging.info("Done.")
    logging.shutdown()


if __name__ == "__main__":
    main()
