from __future__ import annotations

import argparse
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


WORK_DIR = cfg.WORK_DIR
DEFAULT_P2RANK_TABLE = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1" / "tables" / "p2rank_st100_merged_per_pocket.csv"
DEFAULT_ALPHAFILL_TABLE = WORK_DIR / "alphafill_unified" / "run1" / "analysis_v1" / "per_ubs_summary_same_uniprot_ge_0p90.csv"
DEFAULT_OUTPUT_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1" / "controlled_pairs_v1"

FULL_LEVEL_ORDER = ["very_low", "low", "moderate", "high", "very_high"]
VERBAL_MAP = {
    "winning_p": "prob_level_db",
    "pocket_separation_pair": "separation_level_db",
    "pocket_density_pair": "density_level_db",
}
QUANTILE_LABELS = [f"Q{i}" for i in range(1, 6)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controlled-pair analysis of P2Rank success across AHoJ/AlphaFill metrics.")
    p.add_argument("--p2rank_table", type=Path, default=DEFAULT_P2RANK_TABLE)
    p.add_argument("--alphafill_table", type=Path, default=DEFAULT_ALPHAFILL_TABLE)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--success_col", default="success_tns")
    p.add_argument("--focus_metric", default="pocket_separation_pair")
    p.add_argument("--focus_mode", choices=["auto", "verbal", "quantile"], default="auto")
    p.add_argument("--control_metric", default="log2_ratio_holo_to_apo_avg_plus1")
    p.add_argument("--control_mode", choices=["auto", "verbal", "quantile"], default="auto")
    p.add_argument("--quantile_bins", type=int, default=5)
    p.add_argument("--threshold", default="ST_100")
    p.add_argument("--match_scope", default="same_uniprot")
    p.add_argument("--identity_tag", default="ge_0p90")
    return p.parse_args()


def load_merged(p2rank_table: Path, alphafill_table: Path, *, threshold: str) -> pd.DataFrame:
    p2 = pd.read_csv(p2rank_table, low_memory=False)
    p2 = p2[p2["threshold"] == threshold].copy()
    af_cols = [
        "unit_uid",
        "threshold",
        "matched",
        "analysis_match_scope",
        "analysis_identity_threshold",
        "avg_clash",
        "avg_clash_count",
        "avg_local_rmsd",
        "avg_global_rmsd",
        "avg_pae_mean",
        "avg_pae_median",
        "mean_identity",
        "max_identity",
        "fit_success_low_clash",
        "unmatched_reason",
    ]
    af = pd.read_csv(alphafill_table, usecols=af_cols, low_memory=False)
    af = af[af["threshold"] == threshold].drop_duplicates(subset=["unit_uid"]).copy()
    merged = p2.merge(af, on=["unit_uid", "threshold"], how="left", validate="one_to_one")
    merged["matched"] = merged["matched"].fillna(False)
    return merged


def resolve_mode(metric: str, mode: str) -> str:
    if mode != "auto":
        return mode
    return "verbal" if metric in VERBAL_MAP else "quantile"


def _format_bin_value(v: float, metric: str) -> str:
    if pd.isna(v):
        return "NA"
    if metric in {"ligand_atoms", "n_holo_avg", "n_apo_avg", "predicted_pockets_total"}:
        return str(int(round(float(v))))
    return f"{float(v):.2f}".rstrip("0").rstrip(".")


def add_binned_column(
    df: pd.DataFrame,
    metric: str,
    mode: str,
    n_bins: int,
    out_col: str,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    out = df.copy()
    if mode == "verbal":
        verbal_col = VERBAL_MAP[metric]
        out[out_col] = out[verbal_col].astype(str)
        label_map = {label: label for label in FULL_LEVEL_ORDER}
        return out, FULL_LEVEL_ORDER, label_map
    vals = pd.to_numeric(out[metric], errors="coerce")
    valid = vals.notna()
    if valid.sum() == 0:
        out[out_col] = np.nan
        empty_labels = {label: label for label in QUANTILE_LABELS[:n_bins]}
        return out, QUANTILE_LABELS[:n_bins], empty_labels
    try:
        bins = pd.qcut(vals[valid], q=n_bins, labels=QUANTILE_LABELS[:n_bins], duplicates="drop")
    except ValueError:
        out[out_col] = np.nan
        empty_labels = {label: label for label in QUANTILE_LABELS[:n_bins]}
        return out, QUANTILE_LABELS[:n_bins], empty_labels
    out[out_col] = pd.Series(index=out.index, dtype="object")
    out.loc[valid, out_col] = bins.astype(str)
    present = set(out.loc[valid, out_col].dropna().astype(str).tolist())
    order = [label for label in QUANTILE_LABELS[:n_bins] if label in present]
    intervals = pd.qcut(vals[valid], q=n_bins, duplicates="drop")
    cats = intervals.cat.categories
    display_labels: dict[str, str] = {}
    for label, interval in zip(QUANTILE_LABELS[: len(cats)], cats):
        left = _format_bin_value(interval.left, metric)
        right = _format_bin_value(interval.right, metric)
        display_labels[label] = f"{label} [{left}, {right}]"
    return out, order, display_labels


def summarize_controlled(df: pd.DataFrame, *, success_col: str, focus_bin_col: str, control_bin_col: str) -> pd.DataFrame:
    rows = []
    for dataset in ["all", "alphafill_matched"]:
        base = df.copy() if dataset == "all" else df[df["matched"] == True].copy()
        for pocket_class in ["apo", "holo"]:
            class_df = base[base["pocket_class"] == pocket_class].copy()
            for control_bin, cgrp in class_df.groupby(control_bin_col, dropna=False):
                for focus_bin, grp in cgrp.groupby(focus_bin_col, dropna=False):
                    success = pd.to_numeric(grp[success_col], errors="coerce")
                    rows.append(
                        {
                            "dataset": dataset,
                            "pocket_class": pocket_class,
                            "control_bin": str(control_bin),
                            "focus_bin": str(focus_bin),
                            "n": int(success.notna().sum()),
                            "success_rate": float(success.mean()),
                            "success_percent": float(success.mean() * 100.0),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_control_means(df: pd.DataFrame, *, success_col: str, focus_bin_col: str, control_bin_col: str, focus_metric: str, control_metric: str) -> pd.DataFrame:
    rows = []
    for dataset in ["all", "alphafill_matched"]:
        base = df.copy() if dataset == "all" else df[df["matched"] == True].copy()
        for pocket_class in ["apo", "holo"]:
            class_df = base[base["pocket_class"] == pocket_class].copy()
            for control_bin, cgrp in class_df.groupby(control_bin_col, dropna=False):
                for focus_bin, grp in cgrp.groupby(focus_bin_col, dropna=False):
                    rows.append(
                        {
                            "dataset": dataset,
                            "pocket_class": pocket_class,
                            "control_bin": str(control_bin),
                            "focus_bin": str(focus_bin),
                            "n": int(len(grp)),
                            "success_percent": float(pd.to_numeric(grp[success_col], errors="coerce").mean() * 100.0),
                            "focus_metric_mean": float(pd.to_numeric(grp[focus_metric], errors="coerce").mean()),
                            "control_metric_mean": float(pd.to_numeric(grp[control_metric], errors="coerce").mean()),
                        }
                    )
    return pd.DataFrame(rows)


def write_manifest(
    output_dir: Path,
    df: pd.DataFrame,
    args: argparse.Namespace,
    focus_order: list[str],
    control_order: list[str],
    focus_display_labels: dict[str, str],
    control_display_labels: dict[str, str],
) -> None:
    manifest = {
        "p2rank_table": str(args.p2rank_table),
        "alphafill_table": str(args.alphafill_table),
        "threshold": args.threshold,
        "match_scope": args.match_scope,
        "identity_tag": args.identity_tag,
        "success_col": args.success_col,
        "focus_metric": args.focus_metric,
        "focus_mode": args.focus_mode,
        "control_metric": args.control_metric,
        "control_mode": args.control_mode,
        "n_total": int(len(df)),
        "n_alphafill_matched": int(df["matched"].fillna(False).sum()),
        "focus_order": focus_order,
        "control_order": control_order,
        "focus_display_labels": focus_display_labels,
        "control_display_labels": control_display_labels,
    }
    with (output_dir / "tables" / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def main() -> None:
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merged(args.p2rank_table, args.alphafill_table, threshold=args.threshold)
    focus_mode = resolve_mode(args.focus_metric, args.focus_mode)
    control_mode = resolve_mode(args.control_metric, args.control_mode)
    merged, focus_order, focus_display_labels = add_binned_column(
        merged, args.focus_metric, focus_mode, args.quantile_bins, "focus_bin"
    )
    merged, control_order, control_display_labels = add_binned_column(
        merged, args.control_metric, control_mode, args.quantile_bins, "control_bin"
    )

    valid = merged["focus_bin"].notna() & merged["control_bin"].notna() & pd.to_numeric(merged[args.success_col], errors="coerce").notna()
    merged_valid = merged[valid].copy()

    summary = summarize_controlled(
        merged_valid,
        success_col=args.success_col,
        focus_bin_col="focus_bin",
        control_bin_col="control_bin",
    )
    means = summarize_control_means(
        merged_valid,
        success_col=args.success_col,
        focus_bin_col="focus_bin",
        control_bin_col="control_bin",
        focus_metric=args.focus_metric,
        control_metric=args.control_metric,
    )

    merged_valid.to_csv(tables_dir / "controlled_pair_merged.csv", index=False)
    summary.to_csv(tables_dir / "controlled_pair_success_summary.csv", index=False)
    means.to_csv(tables_dir / "controlled_pair_mean_summary.csv", index=False)
    write_manifest(
        args.output_dir,
        merged_valid,
        args,
        focus_order,
        control_order,
        focus_display_labels,
        control_display_labels,
    )
    print(f"[SAVE] {args.output_dir}")


if __name__ == "__main__":
    main()
