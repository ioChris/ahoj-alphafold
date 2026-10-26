from __future__ import annotations

import argparse
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


WORK_DIR = cfg.WORK_DIR
DEFAULT_P2RANK_TABLE = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1" / "tables" / "p2rank_st100_merged_per_pocket.csv"
DEFAULT_ALPHAFILL_TABLE = WORK_DIR / "alphafill_unified" / "run1" / "analysis_v1" / "per_ubs_summary_same_uniprot_ge_0p90.csv"
DEFAULT_OUTPUT_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1" / "alphafill_matched_v1"

FULL_LEVEL_ORDER = ["very_low", "low", "moderate", "high", "very_high"]
ALPHAFILL_METRICS = ["avg_clash", "avg_clash_count", "avg_local_rmsd", "avg_global_rmsd"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze P2Rank success against matched AlphaFill ST_100 metrics.")
    p.add_argument("--p2rank_table", type=Path, default=DEFAULT_P2RANK_TABLE)
    p.add_argument("--alphafill_table", type=Path, default=DEFAULT_ALPHAFILL_TABLE)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--success_col", default="success_tns")
    p.add_argument("--threshold", default="ST_100")
    p.add_argument("--match_scope", default="same_uniprot")
    p.add_argument("--identity_tag", default="ge_0p90")
    p.add_argument("--quantile_bins", type=int, default=5)
    return p.parse_args()


def load_inputs(p2rank_table: Path, alphafill_table: Path, *, threshold: str) -> pd.DataFrame:
    p2_cols = [
        "unit_uid",
        "threshold",
        "pocket_class",
        "winning_p",
        "pocket_separation_pair",
        "prob_level_db",
        "separation_level_db",
        "success_top1",
        "success_topn",
        "success_tns",
        "success_tnm",
        "success_dca4_sites",
        "success_dca_4A",
        "success_dca_5A",
        "success_dca_6A",
        "success_dca_8A",
        "success_dca_10A",
    ]
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
    p2 = pd.read_csv(p2rank_table, usecols=p2_cols, low_memory=False)
    p2 = p2[p2["threshold"] == threshold].copy()
    af = pd.read_csv(alphafill_table, usecols=af_cols, low_memory=False)
    af = af[af["threshold"] == threshold].copy()
    merged = p2.merge(af.drop_duplicates(subset=["unit_uid"]), on=["unit_uid", "threshold"], how="inner", validate="one_to_one")
    matched = merged[merged["matched"] == True].copy()
    return matched


def verbal_separation_summary(df: pd.DataFrame, success_col: str) -> pd.DataFrame:
    rows = []
    for pocket_class in ["apo", "holo"]:
        sub = df[df["pocket_class"] == pocket_class].copy()
        for sep in FULL_LEVEL_ORDER:
            grp = sub[sub["separation_level_db"] == sep].copy()
            if grp.empty:
                continue
            rows.append(
                {
                    "pocket_class": pocket_class,
                    "separation_level_db": sep,
                    "n": int(len(grp)),
                    "success_rate": float(pd.to_numeric(grp[success_col], errors="coerce").mean()),
                    "success_percent": float(pd.to_numeric(grp[success_col], errors="coerce").mean() * 100.0),
                }
            )
    return pd.DataFrame(rows)


def metric_quantile_summary(df: pd.DataFrame, success_col: str, n_bins: int) -> pd.DataFrame:
    rows = []
    for metric in ALPHAFILL_METRICS:
        vals = pd.to_numeric(df[metric], errors="coerce")
        valid = vals.notna() & pd.to_numeric(df[success_col], errors="coerce").notna()
        if valid.sum() < n_bins * 20:
            continue
        try:
            bins = pd.qcut(vals[valid], q=n_bins, labels=[f"Q{i}" for i in range(1, n_bins + 1)], duplicates="drop")
        except ValueError:
            continue
        tmp = df.loc[valid, ["pocket_class", success_col]].copy()
        tmp["metric"] = metric
        tmp["bin"] = bins.astype(str)
        tmp["metric_value"] = vals[valid]
        for pocket_class in ["apo", "holo"]:
            class_df = tmp[tmp["pocket_class"] == pocket_class].copy()
            for bin_label, grp in class_df.groupby("bin", dropna=False):
                rows.append(
                    {
                        "metric": metric,
                        "bin": str(bin_label),
                        "pocket_class": pocket_class,
                        "n": int(len(grp)),
                        "success_rate": float(pd.to_numeric(grp[success_col], errors="coerce").mean()),
                        "success_percent": float(pd.to_numeric(grp[success_col], errors="coerce").mean() * 100.0),
                        "metric_mean": float(pd.to_numeric(grp["metric_value"], errors="coerce").mean()),
                    }
                )
    return pd.DataFrame(rows)


def metric_separation_combo_summary(df: pd.DataFrame, success_col: str, n_bins: int) -> pd.DataFrame:
    rows = []
    for metric in ALPHAFILL_METRICS:
        vals = pd.to_numeric(df[metric], errors="coerce")
        valid = vals.notna() & pd.to_numeric(df[success_col], errors="coerce").notna()
        if valid.sum() < n_bins * 30:
            continue
        try:
            bins = pd.qcut(vals[valid], q=n_bins, labels=[f"Q{i}" for i in range(1, n_bins + 1)], duplicates="drop")
        except ValueError:
            continue
        tmp = df.loc[valid, ["pocket_class", "separation_level_db", success_col]].copy()
        tmp["metric"] = metric
        tmp["metric_bin"] = bins.astype(str)
        for pocket_class in ["apo", "holo"]:
            class_df = tmp[tmp["pocket_class"] == pocket_class].copy()
            for sep in FULL_LEVEL_ORDER:
                sep_df = class_df[class_df["separation_level_db"] == sep].copy()
                for bin_label, grp in sep_df.groupby("metric_bin", dropna=False):
                    rows.append(
                        {
                            "metric": metric,
                            "pocket_class": pocket_class,
                            "separation_level_db": sep,
                            "metric_bin": str(bin_label),
                            "n": int(len(grp)),
                            "success_rate": float(pd.to_numeric(grp[success_col], errors="coerce").mean()),
                            "success_percent": float(pd.to_numeric(grp[success_col], errors="coerce").mean() * 100.0),
                        }
                    )
    return pd.DataFrame(rows)


def write_manifest(output_dir: Path, df: pd.DataFrame, success_col: str, args: argparse.Namespace) -> None:
    manifest = {
        "p2rank_table": str(args.p2rank_table),
        "alphafill_table": str(args.alphafill_table),
        "threshold": args.threshold,
        "match_scope": args.match_scope,
        "identity_tag": args.identity_tag,
        "success_col": success_col,
        "n_matched_rows": int(len(df)),
        "n_unique_unit_uid": int(df["unit_uid"].nunique()),
        "overall_success_percent": float(pd.to_numeric(df[success_col], errors="coerce").mean() * 100.0),
        "class_counts": df["pocket_class"].value_counts(dropna=False).to_dict(),
    }
    with (output_dir / "tables" / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def main() -> None:
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    matched = load_inputs(args.p2rank_table, args.alphafill_table, threshold=args.threshold)
    sep_summary = verbal_separation_summary(matched, args.success_col)
    metric_quantiles = metric_quantile_summary(matched, args.success_col, args.quantile_bins)
    combo_summary = metric_separation_combo_summary(matched, args.success_col, args.quantile_bins)

    matched.to_csv(tables_dir / "p2rank_alphafill_matched_merged.csv", index=False)
    sep_summary.to_csv(tables_dir / "p2rank_alphafill_matched_success_by_separation.csv", index=False)
    metric_quantiles.to_csv(tables_dir / "p2rank_alphafill_matched_success_by_metric_quantiles.csv", index=False)
    combo_summary.to_csv(tables_dir / "p2rank_alphafill_matched_success_by_metric_and_separation.csv", index=False)
    write_manifest(args.output_dir, matched, args.success_col, args)
    print(f"[SAVE] {args.output_dir}")


if __name__ == "__main__":
    main()
