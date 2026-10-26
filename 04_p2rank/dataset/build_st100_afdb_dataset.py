# -*- coding: utf-8 -*-
"""
Build an AFDB-focused ST_100 dataset for downstream pocket prediction work.

Inputs:
  1) Unified analysis table (`analysis_unified_levels.csv/.zip`)
  2) Aggregated UBS table (`ubs_allthresholds_aggregated_classifications_filtered.csv/.zip`)
  3) UniProt-specific UBS root (`UBS_root_2/<UniProt>/ST_100.tsv`)

The output keeps:
  - pocket-level classification/confidence metrics from the unified table
  - residue lists and site membership from the aggregated UBS table
  - per-UniProt pocket counts from the threshold files in UBS_root_2

Design choices:
  - Target only clustered UBS rows (`level == "ubs"`)
  - Default threshold: ST_100
  - Keep only single-UniProt, single-chain, non-interface rows
  - Keep only monomeric rows (`multimeric_status_all_unique == 1`)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterable

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


DEFAULT_UNIFIED = cfg.ANALYSIS_UNIFIED_LEVELS_ZIP
DEFAULT_UBS = cfg.UBS_AGG_FILTERED_ZIP
DEFAULT_UBS_ROOT = cfg.UBS_ROOT_DIR
DEFAULT_OUTPUT_DIR = cfg.AFDB_DATASET_ST100_DIR
DEFAULT_COUNT_THRESHOLDS = ["ST_030", "ST_050", "ST_070", "ST_100", "ST_LJ"]

UNIFIED_COLS = [
    "level",
    "threshold",
    "threshold_detail",
    "unit_uid",
    "unit_recipe",
    "uniprots_str",
    "uniprots_key",
    "n_uniprots",
    "n_chains",
    "is_interface",
    "members",
    "members_total",
    "job_names",
    "rep_job",
    "rep_job_overlap",
    "rep_job_resis3",
    "resis3_unit",
    "site_size_intersection",
    "site_size_unit",
    "pocket_class",
    "pocket_density_combined",
    "pocket_density_pair",
    "pocket_density_strongest",
    "pocket_overlap_mode",
    "pocket_overlap_overall",
    "pocket_overlap_pair",
    "pocket_p_apo",
    "pocket_p_holo",
    "pocket_score",
    "model_pocket_plddt",
    "n_apo_avg",
    "n_holo_avg",
]

UBS_COLS = [
    "site_uid",
    "site_recipe",
    "site_uniprots",
    "threshold",
    "threshold_detail",
    "is_interface",
    "n_uniprots",
    "n_entries",
    "n_entries_total",
    "max_n_chains3",
    "max_n_resis3",
    "multimeric_status_all_unique",
    "n_union",
    "n_intersection",
    "union_residues",
    "intersection_residues",
    "representative_job",
    "representative_job_overlap",
    "representative_job_resis3",
    "job_names",
]

OUTPUT_COLS = [
    "threshold",
    "threshold_detail",
    "site_uniprot",
    "site_uid",
    "site_recipe",
    "n_unp_pockets",
    "n_unp_pockets_multichain",
    "n_unp_pockets_ST_030",
    "n_unp_pockets_multichain_ST_030",
    "n_unp_pockets_ST_050",
    "n_unp_pockets_multichain_ST_050",
    "n_unp_pockets_ST_070",
    "n_unp_pockets_multichain_ST_070",
    "n_unp_pockets_ST_100",
    "n_unp_pockets_multichain_ST_100",
    "n_unp_pockets_ST_LJ",
    "n_unp_pockets_multichain_ST_LJ",
    "members",
    "members_total",
    "n_entries",
    "n_entries_total",
    "n_uniprots",
    "n_chains",
    "max_n_chains3",
    "multimeric_status_all_unique",
    "rep_job",
    "rep_job_overlap",
    "rep_job_resis3",
    "job_names",
    "resis3_unit",
    "site_size_unit",
    "site_size_intersection",
    "n_union",
    "n_intersection",
    "union_residues",
    "intersection_residues",
    "pocket_class",
    "pocket_density_combined",
    "pocket_density_pair",
    "pocket_density_strongest",
    "pocket_overlap_mode",
    "pocket_overlap_overall",
    "pocket_overlap_pair",
    "pocket_separation_pair",
    "pocket_p_apo",
    "pocket_p_holo",
    "pocket_score",
    "model_pocket_plddt",
    "n_apo_avg",
    "n_holo_avg",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an ST_100 AFDB pocket dataset.")
    p.add_argument("--unified", type=Path, default=DEFAULT_UNIFIED)
    p.add_argument("--ubs", type=Path, default=DEFAULT_UBS)
    p.add_argument("--ubs_root", type=Path, default=DEFAULT_UBS_ROOT)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--threshold", default="ST_100")
    p.add_argument(
        "--count_thresholds",
        default=",".join(DEFAULT_COUNT_THRESHOLDS),
        help="Comma-separated thresholds to count per UniProt. Use ST_LJ to count the UniProt-specific ST_LJ*.tsv file.",
    )
    p.add_argument("--outfile_name", default=None)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes_iter(byte_iter: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for chunk in byte_iter:
        h.update(chunk)
    return h.hexdigest()


def read_csv_auto(path: Path, usecols: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    meta = {"input_path": str(path), "member_name": None, "sha256": None}

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            member = members[0] if members else zf.namelist()[0]
            meta["member_name"] = member

            def _iter_bytes():
                with zf.open(member) as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        yield chunk

            meta["sha256"] = sha256_bytes_iter(_iter_bytes())
            with zf.open(member) as fh:
                df = pd.read_csv(fh, usecols=usecols, low_memory=False)
        return df, meta

    meta["sha256"] = sha256_file(path)
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    return df, meta


def parse_single_token(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    tokens = [t.strip() for t in str(value).split() if t.strip()]
    return tokens[0] if len(tokens) == 1 else ""


def is_true_singleton_flag(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    if any(sep in text for sep in (";", ",", " ")):
        return False
    try:
        return int(float(text)) == 1
    except Exception:
        return text == "1"


def is_false_like(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return not bool(value)
    text = str(value).strip().lower()
    return text in {"false", "0", "no", "n", ""}


def parse_count_thresholds(raw_value: str) -> list[str]:
    out = []
    for token in str(raw_value).split(","):
        token = token.strip()
        if token and token not in out:
            out.append(token)
    return out


def normalize_threshold_key(raw: str) -> str:
    text = str(raw).strip()
    return "ST_LJ" if text.startswith("ST_LJ") else text


def count_ubs_rows_from_tsv(tsv_path: Path) -> tuple[int, int]:
    df = pd.read_csv(tsv_path, sep="\t")
    total = len(df)
    monomer = int(df["multimeric_status_all"].apply(is_true_singleton_flag).sum()) if "multimeric_status_all" in df.columns else 0
    return (monomer, total)


def collect_threshold_counts_for_uniprot(
    uniprot: str,
    ubs_root: Path,
    thresholds: list[str],
) -> dict[str, tuple[int, int]]:
    unp_dir = ubs_root / uniprot
    out: dict[str, tuple[int, int]] = {}

    if not unp_dir.exists():
        return {threshold: (0, 0) for threshold in thresholds}

    for threshold in thresholds:
        if threshold == "ST_LJ":
            lj_hits = sorted(unp_dir.glob("ST_LJ*.tsv"))
            out[threshold] = count_ubs_rows_from_tsv(lj_hits[0]) if lj_hits else (0, 0)
            continue

        tsv_path = unp_dir / f"{threshold}.tsv"
        out[threshold] = count_ubs_rows_from_tsv(tsv_path) if tsv_path.exists() else (0, 0)

    return out


def build_dataset(
    unified_path: Path,
    ubs_path: Path,
    ubs_root: Path,
    threshold: str,
    count_thresholds: list[str],
) -> tuple[pd.DataFrame, dict]:
    df_unified, meta_unified = read_csv_auto(unified_path, usecols=UNIFIED_COLS)
    df_ubs, meta_ubs = read_csv_auto(ubs_path, usecols=UBS_COLS)

    df_unified = df_unified.loc[
        (df_unified["level"] == "ubs") &
        (df_unified["threshold"] == threshold)
    ].copy()
    df_ubs = df_ubs.loc[df_ubs["threshold"] == threshold].copy()

    df_unified["site_uniprot"] = df_unified["uniprots_str"].apply(parse_single_token)
    df_ubs["site_uniprot"] = df_ubs["site_uniprots"].apply(parse_single_token)

    df = df_unified.merge(
        df_ubs,
        how="left",
        left_on="unit_uid",
        right_on="site_uid",
        suffixes=("_unified", "_ubs"),
        validate="one_to_one",
    )

    if df["site_uid"].isna().any():
        missing = int(df["site_uid"].isna().sum())
        raise RuntimeError(f"Merge failed for {missing} unified rows with missing aggregated UBS matches.")

    df["threshold"] = df["threshold_unified"]
    df["site_uniprot"] = df["site_uniprot_unified"].fillna(df["site_uniprot_ubs"])
    df["site_uid"] = df["unit_uid"]
    df["site_recipe"] = df["unit_recipe"]
    df["threshold_detail"] = df["threshold_detail_unified"].fillna(df["threshold_detail_ubs"])
    df["n_entries"] = pd.to_numeric(df["n_entries"], errors="coerce")
    df["n_entries_total"] = pd.to_numeric(df["n_entries_total"], errors="coerce")
    df["n_uniprots"] = pd.to_numeric(df["n_uniprots_unified"], errors="coerce")
    df["n_chains"] = pd.to_numeric(df["n_chains"], errors="coerce")
    df["max_n_chains3"] = pd.to_numeric(df["max_n_chains3"], errors="coerce")

    df = df.loc[
        df["site_uniprot"].ne("") &
        df["n_uniprots"].eq(1) &
        df["n_chains"].eq(1) &
        df["is_interface_unified"].apply(is_false_like) &
        df["multimeric_status_all_unique"].apply(is_true_singleton_flag)
    ].copy()

    pocket_counts: dict[str, dict[str, tuple[int, int]]] = {}
    for uniprot in sorted(df["site_uniprot"].dropna().unique()):
        pocket_counts[uniprot] = collect_threshold_counts_for_uniprot(uniprot, ubs_root, count_thresholds)

    threshold_key = normalize_threshold_key(threshold)
    df["n_unp_pockets"] = df["site_uniprot"].map(
        lambda u: pocket_counts.get(u, {}).get(threshold_key, (0, 0))[0]
    )
    df["n_unp_pockets_multichain"] = df["site_uniprot"].map(
        lambda u: pocket_counts.get(u, {}).get(threshold_key, (0, 0))[1]
    )
    for extra_threshold in count_thresholds:
        df[f"n_unp_pockets_{extra_threshold}"] = df["site_uniprot"].map(
            lambda u, t=extra_threshold: pocket_counts.get(u, {}).get(t, (0, 0))[0]
        )
        df[f"n_unp_pockets_multichain_{extra_threshold}"] = df["site_uniprot"].map(
            lambda u, t=extra_threshold: pocket_counts.get(u, {}).get(t, (0, 0))[1]
        )
    df["pocket_separation_pair"] = 1.0 - pd.to_numeric(df["pocket_overlap_pair"], errors="coerce")

    rename_map = {
        "site_uniprot": "site_uniprot",
        "members": "members",
        "members_total": "members_total",
        "job_names_unified": "job_names",
        "rep_job": "rep_job",
        "rep_job_overlap": "rep_job_overlap",
        "rep_job_resis3": "rep_job_resis3",
        "resis3_unit": "resis3_unit",
        "site_size_unit": "site_size_unit",
        "site_size_intersection": "site_size_intersection",
        "pocket_class": "pocket_class",
        "pocket_density_combined": "pocket_density_combined",
        "pocket_density_pair": "pocket_density_pair",
        "pocket_density_strongest": "pocket_density_strongest",
        "pocket_overlap_mode": "pocket_overlap_mode",
        "pocket_overlap_overall": "pocket_overlap_overall",
        "pocket_overlap_pair": "pocket_overlap_pair",
        "pocket_p_apo": "pocket_p_apo",
        "pocket_p_holo": "pocket_p_holo",
        "pocket_score": "pocket_score",
        "model_pocket_plddt": "model_pocket_plddt",
        "n_apo_avg": "n_apo_avg",
        "n_holo_avg": "n_holo_avg",
    }
    df = df.rename(columns=rename_map)

    manifest = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "unified": meta_unified,
            "ubs": meta_ubs,
            "ubs_root": str(ubs_root),
        },
        "filtering": {
            "threshold": threshold,
            "level": "ubs",
            "n_uniprots": 1,
            "n_chains": 1,
            "is_interface": False,
            "multimeric_status_all_unique": 1,
        },
        "count_thresholds": count_thresholds,
        "counts": {
            "rows_output": int(len(df)),
            "unique_uniprots_output": int(df["site_uniprot"].nunique()),
        },
    }

    return df[OUTPUT_COLS].sort_values(["site_uniprot", "site_uid"]).reset_index(drop=True), manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, manifest = build_dataset(
        unified_path=args.unified,
        ubs_path=args.ubs,
        ubs_root=args.ubs_root,
        threshold=args.threshold,
        count_thresholds=parse_count_thresholds(args.count_thresholds),
    )

    stem = f"afdb_{args.threshold.lower()}_dataset"
    out_csv = args.output_dir / (args.outfile_name or f"{stem}.csv")
    out_json = args.output_dir / f"{Path(out_csv).stem}_manifest.json"

    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {out_json}")
    print(f"[ROWS] {len(df)}")
    print(f"[UNIPROTS] {df['site_uniprot'].nunique()}")


if __name__ == "__main__":
    main()
