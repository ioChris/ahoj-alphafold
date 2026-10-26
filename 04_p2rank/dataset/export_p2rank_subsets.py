# -*- coding: utf-8 -*-
"""
Create P2Rank-ready subset CSVs from the AFDB geometry dataset.

For now this exports two geometric-center variants:
  - union residue set
  - intersection residue set

Each output keeps the shared annotation columns, renames the chosen residue list
to `chain_resi`, renames the chosen center columns to `center_x/y/z`, and
drops the alternative geometry/residue-set columns.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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


DEFAULT_INPUT = cfg.AFDB_ST100_DATASET_GEOM_CSV
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "p2rank_datasets"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export P2Rank-ready subset CSVs from the AFDB geometry dataset.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def infer_threshold_slug(df: pd.DataFrame) -> str:
    if "threshold" not in df.columns or df.empty:
        return "threshold_unknown"
    token = str(df["threshold"].dropna().iloc[0]).strip().lower()
    return token.replace("*", "star")


def build_subset(df: pd.DataFrame, residue_prefix: str, center_kind: str) -> pd.DataFrame:
    center_cols = [f"{residue_prefix}_center_{center_kind}_x", f"{residue_prefix}_center_{center_kind}_y", f"{residue_prefix}_center_{center_kind}_z"]
    chain_col = f"{residue_prefix}_chain_resi"
    rg_col = f"{residue_prefix}_rg"
    atoms_col = f"{residue_prefix}_n_atoms"
    found_col = f"{residue_prefix}_n_residues_found"

    out = df.copy()
    out = out.loc[
        out[chain_col].fillna("").astype(str).str.strip().ne("") &
        out[center_cols].notna().all(axis=1)
    ].copy()

    drop_cols = []
    for prefix in ("union", "intersection"):
        for suffix in (
            "chain_resi",
            "center_geom_x", "center_geom_y", "center_geom_z",
            "center_com_x", "center_com_y", "center_com_z",
            "rg",
            "n_atoms",
            "n_residues_found",
        ):
            col = f"{prefix}_{suffix}"
            if col in out.columns:
                drop_cols.append(col)

    out = out.drop(columns=drop_cols)
    out["chain_resi"] = df.loc[out.index, chain_col].values
    out["center_x"] = df.loc[out.index, center_cols[0]].values
    out["center_y"] = df.loc[out.index, center_cols[1]].values
    out["center_z"] = df.loc[out.index, center_cols[2]].values
    out["rg"] = df.loc[out.index, rg_col].values
    out["n_atoms"] = df.loc[out.index, atoms_col].values
    out["n_residues_found"] = df.loc[out.index, found_col].values

    return out.reset_index(drop=True)


def export_subset(
    df: pd.DataFrame,
    input_csv: Path,
    output_dir: Path,
    threshold_slug: str,
    residue_prefix: str,
    center_kind: str,
) -> dict:
    subset_dir = output_dir / f"{threshold_slug}_{residue_prefix}_{center_kind}"
    subset_dir.mkdir(parents=True, exist_ok=True)

    subset_df = build_subset(df, residue_prefix=residue_prefix, center_kind=center_kind)
    out_csv = subset_dir / f"{threshold_slug}_p2rank_{residue_prefix}_{center_kind}.csv"
    out_json = subset_dir / f"{threshold_slug}_p2rank_{residue_prefix}_{center_kind}_manifest.json"

    subset_df.to_csv(out_csv, index=False)
    manifest = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input_csv": str(input_csv),
        "output_csv": str(out_csv),
        "residue_prefix": residue_prefix,
        "center_kind": center_kind,
        "rows_output": int(len(subset_df)),
        "unique_uniprots_output": int(subset_df["site_uniprot"].nunique()) if "site_uniprot" in subset_df.columns else None,
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, low_memory=False)
    threshold_slug = infer_threshold_slug(df)

    manifests = []
    for residue_prefix in ("union", "intersection"):
        manifests.append(
            export_subset(
                df=df,
                input_csv=args.input_csv,
                output_dir=args.output_dir,
                threshold_slug=threshold_slug,
                residue_prefix=residue_prefix,
                center_kind="geom",
            )
        )

    summary_path = args.output_dir / f"{threshold_slug}_p2rank_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(manifests, fh, indent=2)

    for manifest in manifests:
        print(f"[SAVE] {manifest['output_csv']}")
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
