# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 18:36:27 2025

@author: Christos P. Feidakis
"""

#!/usr/bin/env python

#!/usr/bin/env python

import os
from pathlib import Path

import pandas as pd

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# === USER CONFIG (edit these paths in Spyder) ================================
# LEGACY prep util (pre-afdb_dataset pipeline). Paths are anchored under the
# configured work_dir; adjust the sub-path to point at your ST_LJ centers file.
INPUT_CSV = str(cfg.WORK_DIR / "afdb_ubs_sets" / "ubs_centers" / "LJ" / "sub_df_ST_LJ_ubs_filtered_annotated_with_centers_nUNP0.csv")
OUTPUT_DIR = str(cfg.WORK_DIR / "afdb_ubs_sets" / "ubs_centers" / "LJ" / "p2rank_datasets")

# ============================================================================

COMMON_COLS = [
    "site_uniprots",
    "site_uid",
    "site_recipe",
    "threshold",
    "afdb_filename",
]

UNION_COLS = [
    "union_chain_resi",
    "union_center_x",
    "union_center_y",
    "union_center_z",
]

INTERSECTION_COLS = [
    "intersection_chain_resi",
    "intersection_center_x",
    "intersection_center_y",
    "intersection_center_z",
]

# Rename maps so that both outputs have the SAME column names
UNION_RENAME = {
    "union_chain_resi": "chain_resi",
    "union_center_x": "center_x",
    "union_center_y": "center_y",
    "union_center_z": "center_z",
}

INTERSECTION_RENAME = {
    "intersection_chain_resi": "chain_resi",
    "intersection_center_x": "center_x",
    "intersection_center_y": "center_y",
    "intersection_center_z": "center_z",
}


def main() -> None:
    input_path = Path(INPUT_CSV)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read source CSV
    df = pd.read_csv(input_path)

    # Count unique UniProt IDs in site_uniprots (for monomers only!)
    n_unique_uniprots = df["site_uniprots"].nunique()

    # Check that required columns exist
    missing_union = [c for c in COMMON_COLS + UNION_COLS if c not in df.columns]
    missing_intersection = [c for c in COMMON_COLS + INTERSECTION_COLS if c not in df.columns]

    if missing_union:
        raise ValueError(f"The following columns required for the UNION dataset are missing: {missing_union}")
    if missing_intersection:
        raise ValueError(
            f"The following columns required for the INTERSECTION dataset are missing: {missing_intersection}"
        )

    # Build union dataframe and normalize column names
    df_union = df[COMMON_COLS + UNION_COLS].rename(columns=UNION_RENAME)

    # Build intersection dataframe and normalize column names
    df_intersection = df[COMMON_COLS + INTERSECTION_COLS].rename(columns=INTERSECTION_RENAME)

    # Derive output filenames from input name
    base_name = input_path.stem  # e.g. "sub_df_ST_LJ_ubs_filtered_annotated_with_centers_nUNP0"
    union_out = out_dir / f"{base_name}_union_centers.csv"
    intersection_out = out_dir / f"{base_name}_intersection_centers.csv"

    # Save CSVs
    df_union.to_csv(union_out, index=False)
    df_intersection.to_csv(intersection_out, index=False)

    # --- Small report --------------------------------------------------------
    report_lines = [
        f"Source file: {input_path}",
        f"Output directory: {out_dir}",
        "",
        f"Original dataset:     {df.shape[0]} rows x {df.shape[1]} columns",
        f"Unique UniProt IDs in 'site_uniprots': {n_unique_uniprots}",
        f"Union dataset:        {df_union.shape[0]} rows x {df_union.shape[1]} columns",
        f"Intersection dataset: {df_intersection.shape[0]} rows x {df_intersection.shape[1]} columns",
        "",
        f"Common columns kept: {', '.join(COMMON_COLS)}",
        f"Union-specific columns (renamed to): "
        f"{', '.join([UNION_RENAME.get(c, c) for c in UNION_COLS])}",
        f"Intersection-specific columns (renamed to): "
        f"{', '.join([INTERSECTION_RENAME.get(c, c) for c in INTERSECTION_COLS])}",
        "",
        f"Union CSV:        {union_out.name}",
        f"Intersection CSV: {intersection_out.name}",
    ]

    report_text = "\n".join(report_lines)
    report_path = out_dir / f"{base_name}_dataset_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    # Also print to console
    print(report_text)


if __name__ == "__main__":
    main()

