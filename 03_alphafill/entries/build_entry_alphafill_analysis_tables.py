# -*- coding: utf-8 -*-
"""
Build clean entry-level AlphaFill analysis tables from the unified AlphaFill merge.

Inputs
------
- merged_ahoj_alphafill_summary.csv
- merged_ahoj_alphafill_long.csv

Outputs
-------
- entry_alphafill_all_summary.csv
- entry_alphafill_matched_summary.csv
- entry_alphafill_unmatched_summary.csv
- entry_alphafill_unmatched_same_uniprot_summary.csv
- entry_alphafill_matched_long.csv
- entry_alphafill_coverage.csv
- entry_alphafill_manifest.json

Design choices
--------------
- The authoritative merge source is the unified AlphaFill merge.
- This script works at `level == "entry"` only.
- It keeps the unclustered entry rows (`threshold == "UNCLUSTERED"`).
- It mirrors the same AlphaFill QA columns used in the UBS branch:
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
DEFAULT_OUTPUT_DIR = cfg.ALPHAFILL_ENTRIES_DIR

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
    p = argparse.ArgumentParser(description="Build entry-level AlphaFill analysis tables from the unified merge outputs.")
    p.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--long_csv", type=Path, default=DEFAULT_LONG)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--thresholds",
        nargs="*",
        default=["UNCLUSTERED"],
        help="Optional subset of thresholds. Default keeps only UNCLUSTERED entry rows.",
    )
    p.add_argument("--exclude_interface", action="store_true", help="Drop interface entry rows.")
    p.add_argument("--single_uniprot_only", action="store_true", help="Keep only entry rows with n_uniprots == 1.")
    p.add_argument("--single_chain_only", action="store_true", help="Keep only entry rows with n_chains == 1.")
    return p.parse_args()


def keep_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[c for c in columns if c in df.columns]].copy()


def add_pocket_separation_pair(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pocket_separation_pair" not in df.columns and "pocket_overlap_pair" in df.columns:
        overlap = pd.to_numeric(df["pocket_overlap_pair"], errors="coerce")
        df["pocket_separation_pair"] = 1.0 - overlap
    return df


def filter_entry_rows(
    df: pd.DataFrame,
    thresholds: list[str] | None,
    exclude_interface: bool,
    single_uniprot_only: bool,
    single_chain_only: bool,
) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[out["level"].astype(str).str.lower() == "entry"].copy()
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


def build_outputs(
    summary_csv: Path,
    long_csv: Path,
    thresholds: list[str] | None,
    exclude_interface: bool,
    single_uniprot_only: bool,
    single_chain_only: bool,
) -> tuple[dict[str, pd.DataFrame], dict]:
    summary_df = pd.read_csv(summary_csv)
    long_df = pd.read_csv(long_csv)
    summary_df = add_pocket_separation_pair(summary_df)
    long_df = add_pocket_separation_pair(long_df)

    summary_df = filter_entry_rows(
        summary_df,
        thresholds=thresholds,
        exclude_interface=exclude_interface,
        single_uniprot_only=single_uniprot_only,
        single_chain_only=single_chain_only,
    )
    long_df = filter_entry_rows(
        long_df,
        thresholds=thresholds,
        exclude_interface=exclude_interface,
        single_uniprot_only=single_uniprot_only,
        single_chain_only=single_chain_only,
    )

    qa = compute_match_qa(long_df)
    summary_df = summary_df.merge(qa, on="merge_row_id", how="left")
    for col in [
        "afill_n_long_rows",
        "afill_n_same_uniprot_rows",
        "afill_n_cross_uniprot_rows",
        "afill_has_match_any",
        "afill_has_same_uniprot_match",
        "afill_has_only_cross_uniprot_match",
    ]:
        if col not in summary_df.columns:
            continue
        if summary_df[col].dtype == bool:
            summary_df[col] = summary_df[col].fillna(False)
        else:
            summary_df[col] = summary_df[col].fillna(0)

    matched_ids_any = set(qa.loc[qa["afill_has_match_any"], "merge_row_id"].tolist())
    matched_ids_same = set(qa.loc[qa["afill_has_same_uniprot_match"], "merge_row_id"].tolist())

    matched_long = keep_existing(long_df, LONG_KEEP).sort_values(["threshold", "unit_uid", "merge_row_id"]).reset_index(drop=True)
    all_summary = keep_existing(summary_df, SUMMARY_KEEP).sort_values(["threshold", "unit_uid", "merge_row_id"]).reset_index(drop=True)
    matched_summary = keep_existing(summary_df.loc[summary_df["merge_row_id"].isin(matched_ids_any)], SUMMARY_KEEP).sort_values(
        ["threshold", "unit_uid", "merge_row_id"]
    ).reset_index(drop=True)
    unmatched_summary = keep_existing(summary_df.loc[~summary_df["merge_row_id"].isin(matched_ids_any)], SUMMARY_KEEP).sort_values(
        ["threshold", "unit_uid", "merge_row_id"]
    ).reset_index(drop=True)
    unmatched_same_uniprot_summary = keep_existing(summary_df.loc[~summary_df["merge_row_id"].isin(matched_ids_same)], SUMMARY_KEEP).sort_values(
        ["threshold", "unit_uid", "merge_row_id"]
    ).reset_index(drop=True)

    coverage_rows = []
    for threshold, sub in summary_df.groupby("threshold", dropna=False, sort=False):
        n_rows = len(sub)
        n_matched_any = int(sub["merge_row_id"].isin(matched_ids_any).sum())
        n_matched_same = int(sub["merge_row_id"].isin(matched_ids_same).sum())
        n_only_cross = int((sub["merge_row_id"].isin(matched_ids_any) & ~sub["merge_row_id"].isin(matched_ids_same)).sum())
        coverage_rows.append(
            {
                "threshold": threshold,
                "n_rows": n_rows,
                "n_entry_rows": n_rows,
                "n_matched_any": n_matched_any,
                "n_unmatched_any": n_rows - n_matched_any,
                "n_matched_same_uniprot": n_matched_same,
                "n_only_cross_uniprot": n_only_cross,
                "frac_matched_any": (n_matched_any / n_rows) if n_rows else np.nan,
                "frac_matched_same_uniprot": (n_matched_same / n_rows) if n_rows else np.nan,
                "frac_only_cross_uniprot": (n_only_cross / n_rows) if n_rows else np.nan,
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "level": "entry",
        "thresholds": thresholds,
        "exclude_interface": exclude_interface,
        "single_uniprot_only": single_uniprot_only,
        "single_chain_only": single_chain_only,
        "sources": {
            "summary_csv": str(summary_csv),
            "long_csv": str(long_csv),
        },
        "counts": {
            "all_summary_rows": int(len(all_summary)),
            "matched_summary_rows": int(len(matched_summary)),
            "unmatched_summary_rows": int(len(unmatched_summary)),
            "unmatched_same_uniprot_summary_rows": int(len(unmatched_same_uniprot_summary)),
            "matched_long_rows": int(len(matched_long)),
            "unique_entry_rows": int(all_summary["unit_uid"].nunique()) if "unit_uid" in all_summary.columns else None,
        },
    }

    return {
        "entry_alphafill_all_summary.csv": all_summary,
        "entry_alphafill_matched_summary.csv": matched_summary,
        "entry_alphafill_unmatched_summary.csv": unmatched_summary,
        "entry_alphafill_unmatched_same_uniprot_summary.csv": unmatched_same_uniprot_summary,
        "entry_alphafill_matched_long.csv": matched_long,
        "entry_alphafill_coverage.csv": coverage_df,
    }, manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs, manifest = build_outputs(
        summary_csv=args.summary_csv,
        long_csv=args.long_csv,
        thresholds=args.thresholds,
        exclude_interface=args.exclude_interface,
        single_uniprot_only=args.single_uniprot_only,
        single_chain_only=args.single_chain_only,
    )
    for filename, df in outputs.items():
        path = args.output_dir / filename
        df.to_csv(path, index=False)
        print(f"[SAVE] {path}")
    manifest_path = args.output_dir / "entry_alphafill_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[SAVE] {manifest_path}")


if __name__ == "__main__":
    main()
