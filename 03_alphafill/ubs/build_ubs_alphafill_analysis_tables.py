# -*- coding: utf-8 -*-
"""
Build clean UBS-level AlphaFill analysis tables from the unified AlphaFill merge.

Inputs
------
- merged_ahoj_alphafill_summary.csv
- merged_ahoj_alphafill_long.csv

Outputs
-------
- ubs_alphafill_all_summary.csv
- ubs_alphafill_matched_summary.csv
- ubs_alphafill_unmatched_summary.csv
- ubs_alphafill_unmatched_same_uniprot_summary.csv
- ubs_alphafill_matched_long.csv
- ubs_alphafill_coverage_by_threshold.csv
- ubs_alphafill_manifest.json

Design choices
--------------
- The authoritative merge source is the unified AlphaFill merge, not the old
  UBS-only merge.
- This script works at `level == "ubs"` only.
- It keeps a curated set of classification/support columns plus the AlphaFill
  fit metrics needed downstream.
- It adds QA columns distinguishing:
    * any AlphaFill match
    * same-UniProt AlphaFill match
    * cross-UniProt-only match
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
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


DEFAULT_SUMMARY = cfg.MERGED_AHOJ_ALPHAFILL_SUMMARY_CSV
DEFAULT_LONG = cfg.MERGED_AHOJ_ALPHAFILL_LONG_CSV
DEFAULT_OUTPUT_DIR = cfg.ALPHAFILL_UNIFIED_DIR

SUMMARY_KEEP = [
    "merge_row_id",
    "source_row_index",
    "level",
    "threshold",
    "threshold_detail",
    "unit_uid",
    "unit_recipe",
    "uniprots_str",
    "uniprots_key",
    "is_interface",
    "n_uniprots",
    "n_chains",
    "members",
    "members_total",
    "rep_job",
    "job_names",
    "site_size_unit",
    "site_size_intersection",
    "evidence_basis",
    "evidence_n_apo",
    "evidence_n_holo",
    "n_apo_avg",
    "n_holo_avg",
    "model_pocket_plddt",
    "pocket_class",
    "pocket_p_apo",
    "pocket_p_holo",
    "pocket_score",
    "pocket_density_combined",
    "pocket_density_pair",
    "pocket_density_strongest",
    "pocket_overlap_mode",
    "pocket_overlap_overall",
    "pocket_overlap_pair",
    "pocket_separation_pair",
    "afill_n_jobs_total",
    "afill_n_jobs_matched",
    "afill_n_jobs_unmatched",
    "afill_frac_jobs_matched",
    "afill_rep_job_matched",
    "afill_matched_jobs_str",
    "afill_n_transplants_matched",
    "afill_uniprots_str",
    "afill_n_uniprots",
    "afill_clash_count_mean",
    "afill_clash_count_median",
    "afill_clash_score_mean",
    "afill_clash_score_median",
    "afill_local_rmsd_mean",
    "afill_local_rmsd_median",
    "afill_mapping_global_rmsd_mean",
    "afill_mapping_global_rmsd_median",
    "afill_mapping_align_identity_mean",
    "afill_mapping_align_identity_median",
    "afill_pae_mean_mean",
    "afill_pae_mean_median",
    "afill_pae_median_mean",
    "afill_pae_median_median",
    "afill_cross_uniprot",
    "afill_n_long_rows",
    "afill_n_same_uniprot_rows",
    "afill_n_cross_uniprot_rows",
    "afill_has_match_any",
    "afill_has_same_uniprot_match",
    "afill_has_only_cross_uniprot_match",
]

LONG_KEEP = [
    "merge_row_id",
    "source_row_index",
    "level",
    "threshold",
    "threshold_detail",
    "unit_uid",
    "unit_recipe",
    "uniprots_str",
    "uniprots_key",
    "is_interface",
    "n_uniprots",
    "n_chains",
    "members",
    "members_total",
    "rep_job",
    "job_names",
    "site_size_unit",
    "site_size_intersection",
    "evidence_basis",
    "evidence_n_apo",
    "evidence_n_holo",
    "n_apo_avg",
    "n_holo_avg",
    "model_pocket_plddt",
    "pocket_class",
    "pocket_p_apo",
    "pocket_p_holo",
    "pocket_score",
    "pocket_density_combined",
    "pocket_density_pair",
    "pocket_density_strongest",
    "pocket_overlap_mode",
    "pocket_overlap_overall",
    "pocket_overlap_pair",
    "pocket_separation_pair",
    "afill_matched_job",
    "afill_reconstructed_key",
    "afill_site_uniprot",
    "afill_compound_id",
    "afill_analogue_id",
    "afill_structure_pdb_asym_id",
    "afill_ligand_pdb_auth_asym_id",
    "afill_ligand_pdb_auth_seq_id",
    "afill_mapping_global_rmsd",
    "afill_mapping_align_identity",
    "afill_mapping_align_length",
    "afill_clash_count",
    "afill_clash_score",
    "afill_local_rmsd",
    "afill_pae_mean",
    "afill_pae_median",
    "afill_cross_uniprot",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build UBS-level AlphaFill analysis tables from the unified merge outputs.")
    p.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--long_csv", type=Path, default=DEFAULT_LONG)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--thresholds",
        nargs="*",
        default=None,
        help="Optional subset of thresholds, e.g. ST_100 ST_LJ. Default keeps all UBS thresholds.",
    )
    p.add_argument("--exclude_interface", action="store_true", help="Drop interface UBS rows.")
    p.add_argument("--single_uniprot_only", action="store_true", help="Keep only UBS rows with n_uniprots == 1.")
    p.add_argument("--single_chain_only", action="store_true", help="Keep only UBS rows with n_chains == 1.")
    return p.parse_args()


def keep_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[c for c in columns if c in df.columns]].copy()


def add_pocket_separation_pair(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pocket_separation_pair" not in df.columns and "pocket_overlap_pair" in df.columns:
        overlap = pd.to_numeric(df["pocket_overlap_pair"], errors="coerce")
        df["pocket_separation_pair"] = 1.0 - overlap
    return df


def filter_ubs_rows(
    df: pd.DataFrame,
    thresholds: list[str] | None,
    exclude_interface: bool,
    single_uniprot_only: bool,
    single_chain_only: bool,
) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[out["level"].astype(str).str.lower() == "ubs"].copy()

    if thresholds:
        out = out.loc[out["threshold"].astype(str).isin(thresholds)].copy()
    if exclude_interface and "is_interface" in out.columns:
        out = out.loc[~out["is_interface"].fillna(False).astype(bool)].copy()
    if single_uniprot_only and "n_uniprots" in out.columns:
        out = out.loc[pd.to_numeric(out["n_uniprots"], errors="coerce").eq(1)].copy()
    if single_chain_only and "n_chains" in out.columns:
        out = out.loc[pd.to_numeric(out["n_chains"], errors="coerce").eq(1)].copy()

    return out


def compute_match_qa(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                "merge_row_id",
                "afill_n_long_rows",
                "afill_n_same_uniprot_rows",
                "afill_n_cross_uniprot_rows",
                "afill_has_match_any",
                "afill_has_same_uniprot_match",
                "afill_has_only_cross_uniprot_match",
            ]
        )

    cross = long_df["afill_cross_uniprot"].fillna(False).astype(bool)
    tmp = pd.DataFrame(
        {
            "merge_row_id": long_df["merge_row_id"],
            "is_same_uniprot": ~cross,
            "is_cross_uniprot": cross,
        }
    )

    qa = tmp.groupby("merge_row_id", sort=False).agg(
        afill_n_long_rows=("merge_row_id", "size"),
        afill_n_same_uniprot_rows=("is_same_uniprot", "sum"),
        afill_n_cross_uniprot_rows=("is_cross_uniprot", "sum"),
    ).reset_index()
    qa["afill_has_match_any"] = qa["afill_n_long_rows"].gt(0)
    qa["afill_has_same_uniprot_match"] = qa["afill_n_same_uniprot_rows"].gt(0)
    qa["afill_has_only_cross_uniprot_match"] = qa["afill_has_match_any"] & ~qa["afill_has_same_uniprot_match"]
    return qa


def build_coverage(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    work = summary_df.copy()
    work["n_rows"] = 1
    work["n_matched_any"] = work["afill_has_match_any"].fillna(False).astype(int)
    work["n_unmatched_any"] = (~work["afill_has_match_any"].fillna(False)).astype(int)
    work["n_matched_same_uniprot"] = work["afill_has_same_uniprot_match"].fillna(False).astype(int)
    work["n_only_cross_uniprot"] = work["afill_has_only_cross_uniprot_match"].fillna(False).astype(int)

    def _ordered_join(values: pd.Series, max_items: int = 8) -> str:
        seen = []
        for v in values.fillna("").astype(str):
            if v and v not in seen:
                seen.append(v)
        if len(seen) <= max_items:
            return ";".join(seen)
        return ";".join(seen[:max_items]) + ";..."

    cov = work.groupby(["threshold"], dropna=False).agg(
        n_threshold_detail_values=("threshold_detail", lambda s: int(pd.Series(s).fillna("").astype(str).replace("", np.nan).dropna().nunique())),
        threshold_detail_examples=("threshold_detail", _ordered_join),
        n_ubs_rows=("n_rows", "sum"),
        n_matched_any=("n_matched_any", "sum"),
        n_unmatched_any=("n_unmatched_any", "sum"),
        n_matched_same_uniprot=("n_matched_same_uniprot", "sum"),
        n_only_cross_uniprot=("n_only_cross_uniprot", "sum"),
        sum_long_rows=("afill_n_long_rows", "sum"),
        sum_same_uniprot_long_rows=("afill_n_same_uniprot_rows", "sum"),
        sum_cross_uniprot_long_rows=("afill_n_cross_uniprot_rows", "sum"),
    ).reset_index()

    cov["frac_matched_any"] = cov["n_matched_any"] / cov["n_ubs_rows"]
    cov["frac_matched_same_uniprot"] = cov["n_matched_same_uniprot"] / cov["n_ubs_rows"]
    cov["frac_only_cross_uniprot"] = cov["n_only_cross_uniprot"] / cov["n_ubs_rows"]
    cov["frac_same_uniprot_long_rows"] = np.where(
        cov["sum_long_rows"] > 0,
        cov["sum_same_uniprot_long_rows"] / cov["sum_long_rows"],
        np.nan,
    )
    cov["frac_cross_uniprot_long_rows"] = np.where(
        cov["sum_long_rows"] > 0,
        cov["sum_cross_uniprot_long_rows"] / cov["sum_long_rows"],
        np.nan,
    )

    return cov


def build_tables(
    summary_csv: Path,
    long_csv: Path,
    thresholds: list[str] | None,
    exclude_interface: bool,
    single_uniprot_only: bool,
    single_chain_only: bool,
) -> tuple[dict[str, pd.DataFrame], dict]:
    summary_df = pd.read_csv(summary_csv, low_memory=False)
    long_df = pd.read_csv(long_csv, low_memory=False)

    summary_df = add_pocket_separation_pair(summary_df)
    long_df = add_pocket_separation_pair(long_df)

    summary_df = filter_ubs_rows(
        summary_df,
        thresholds=thresholds,
        exclude_interface=exclude_interface,
        single_uniprot_only=single_uniprot_only,
        single_chain_only=single_chain_only,
    )
    long_df = filter_ubs_rows(
        long_df,
        thresholds=thresholds,
        exclude_interface=exclude_interface,
        single_uniprot_only=single_uniprot_only,
        single_chain_only=single_chain_only,
    )

    qa_df = compute_match_qa(long_df)
    summary_df = summary_df.merge(qa_df, on="merge_row_id", how="left")

    for col in ["afill_n_long_rows", "afill_n_same_uniprot_rows", "afill_n_cross_uniprot_rows"]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].fillna(0).astype(int)
    for col in ["afill_has_match_any", "afill_has_same_uniprot_match", "afill_has_only_cross_uniprot_match"]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].eq(True)

    matched_ids_any = set(summary_df.loc[summary_df["afill_has_match_any"], "merge_row_id"].tolist())
    matched_ids_same = set(summary_df.loc[summary_df["afill_has_same_uniprot_match"], "merge_row_id"].tolist())

    matched_long = keep_existing(long_df, LONG_KEEP).sort_values(["threshold", "unit_uid", "merge_row_id"]).reset_index(drop=True)
    all_summary = keep_existing(summary_df, SUMMARY_KEEP).sort_values(["threshold", "unit_uid", "merge_row_id"]).reset_index(drop=True)
    matched_summary = keep_existing(summary_df.loc[summary_df["merge_row_id"].isin(matched_ids_any)], SUMMARY_KEEP).sort_values(
        ["threshold", "unit_uid", "merge_row_id"]
    ).reset_index(drop=True)
    unmatched_summary = keep_existing(summary_df.loc[~summary_df["merge_row_id"].isin(matched_ids_any)], SUMMARY_KEEP).sort_values(
        ["threshold", "unit_uid", "merge_row_id"]
    ).reset_index(drop=True)
    unmatched_same_uniprot_summary = keep_existing(
        summary_df.loc[~summary_df["merge_row_id"].isin(matched_ids_same)],
        SUMMARY_KEEP,
    ).sort_values(["threshold", "unit_uid", "merge_row_id"]).reset_index(drop=True)

    coverage_df = build_coverage(summary_df)

    manifest = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "summary_csv": str(summary_csv),
            "long_csv": str(long_csv),
        },
        "filters": {
            "level": "ubs",
            "thresholds": thresholds,
            "exclude_interface": exclude_interface,
            "single_uniprot_only": single_uniprot_only,
            "single_chain_only": single_chain_only,
        },
        "counts": {
            "all_summary_rows": int(len(all_summary)),
            "matched_summary_rows": int(len(matched_summary)),
            "unmatched_summary_rows": int(len(unmatched_summary)),
            "unmatched_same_uniprot_summary_rows": int(len(unmatched_same_uniprot_summary)),
            "matched_long_rows": int(len(matched_long)),
            "unique_ubs_rows": int(all_summary["unit_uid"].nunique()) if "unit_uid" in all_summary.columns else None,
        },
    }

    return {
        "ubs_alphafill_all_summary.csv": all_summary,
        "ubs_alphafill_matched_summary.csv": matched_summary,
        "ubs_alphafill_unmatched_summary.csv": unmatched_summary,
        "ubs_alphafill_unmatched_same_uniprot_summary.csv": unmatched_same_uniprot_summary,
        "ubs_alphafill_matched_long.csv": matched_long,
        "ubs_alphafill_coverage_by_threshold.csv": coverage_df,
    }, manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables, manifest = build_tables(
        summary_csv=args.summary_csv,
        long_csv=args.long_csv,
        thresholds=args.thresholds,
        exclude_interface=args.exclude_interface,
        single_uniprot_only=args.single_uniprot_only,
        single_chain_only=args.single_chain_only,
    )

    for filename, df in tables.items():
        out_path = args.output_dir / filename
        df.to_csv(out_path, index=False)
        print(f"[SAVE] {out_path}")

    manifest_path = args.output_dir / "ubs_alphafill_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[SAVE] {manifest_path}")


if __name__ == "__main__":
    main()
