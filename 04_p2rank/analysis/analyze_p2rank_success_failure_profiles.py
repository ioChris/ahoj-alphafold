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
DEFAULT_OUTPUT_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1" / "success_failure_profiles_v1"

FULL_LEVEL_ORDER = ["very_low", "low", "moderate", "high", "very_high"]
VERBAL_METRICS = {
    "separation_level_db": FULL_LEVEL_ORDER,
    "density_level_db": FULL_LEVEL_ORDER,
    "prob_level_db": FULL_LEVEL_ORDER,
}
CONTINUOUS_METRICS = [
    "pocket_overlap_pair",
    "log2_ratio_holo_to_apo_avg_plus1",
    "pocket_separation_pair",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze success-vs-failure P2Rank pocket profiles.")
    p.add_argument("--p2rank_table", type=Path, default=DEFAULT_P2RANK_TABLE)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--success_col", default="success_tns")
    p.add_argument("--size_metric", default="ligand_atoms")
    p.add_argument("--size_bins", type=int, default=5)
    p.add_argument("--profile_bins", type=int, default=5)
    return p.parse_args()


def load_table(p2rank_table: Path) -> pd.DataFrame:
    df = pd.read_csv(p2rank_table, low_memory=False)
    return df[df["threshold"] == "ST_100"].copy()


def add_size_bins(df: pd.DataFrame, size_metric: str, n_bins: int) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    vals = pd.to_numeric(out[size_metric], errors="coerce")
    valid = vals.notna()
    out["size_bin"] = pd.Series(index=out.index, dtype="object")
    bins = pd.qcut(vals[valid], q=n_bins, labels=[f"Q{i}" for i in range(1, n_bins + 1)], duplicates="drop")
    out.loc[valid, "size_bin"] = bins.astype(str)
    order = [f"Q{i}" for i in range(1, n_bins + 1) if f"Q{i}" in set(out["size_bin"].dropna())]
    return out, order


def verbal_profile_summary(df: pd.DataFrame, success_col: str) -> pd.DataFrame:
    rows = []
    out = df.copy()
    if success_col == "success_topn" and "dca4rank" in out.columns:
        rank = pd.to_numeric(out["dca4rank"], errors="coerce")
        out["outcome"] = rank.map(lambda v: "success" if pd.notna(v) and v >= 1 else "failure")
    else:
        out["outcome"] = out[success_col].map(lambda v: "success" if bool(v) else "failure")
    for metric, order in VERBAL_METRICS.items():
        for pocket_class in ["apo", "holo"]:
            for outcome in ["success", "failure"]:
                sub = out[(out["pocket_class"] == pocket_class) & (out["outcome"] == outcome)].copy()
                total = len(sub)
                if total == 0:
                    continue
                counts = sub[metric].astype(str).value_counts(dropna=False)
                for bin_label in order:
                    n = int(counts.get(bin_label, 0))
                    rows.append(
                        {
                            "metric": metric,
                            "pocket_class": pocket_class,
                            "outcome": outcome,
                            "bin": bin_label,
                            "n": n,
                            "share_within_group": float(n / total),
                        }
                    )
    return pd.DataFrame(rows)


def quantile_profile_summary(df: pd.DataFrame, success_col: str, n_bins: int) -> pd.DataFrame:
    rows = []
    out = df.copy()
    if success_col == "success_topn" and "dca4rank" in out.columns:
        rank = pd.to_numeric(out["dca4rank"], errors="coerce")
        out["outcome"] = rank.map(lambda v: "success" if pd.notna(v) and v >= 1 else "failure")
    else:
        out["outcome"] = out[success_col].map(lambda v: "success" if bool(v) else "failure")
    for metric in CONTINUOUS_METRICS:
        vals = pd.to_numeric(out[metric], errors="coerce")
        valid = vals.notna()
        if valid.sum() < n_bins * 20:
            continue
        try:
            bins = pd.qcut(vals[valid], q=n_bins, labels=[f"Q{i}" for i in range(1, n_bins + 1)], duplicates="drop")
        except ValueError:
            continue
        tmp = out.loc[valid, ["pocket_class", "outcome"]].copy()
        tmp["metric"] = metric
        tmp["bin"] = bins.astype(str)
        order = [f"Q{i}" for i in range(1, n_bins + 1) if f"Q{i}" in set(tmp["bin"].dropna())]
        for pocket_class in ["apo", "holo"]:
            for outcome in ["success", "failure"]:
                sub = tmp[(tmp["pocket_class"] == pocket_class) & (tmp["outcome"] == outcome)].copy()
                total = len(sub)
                if total == 0:
                    continue
                counts = sub["bin"].value_counts(dropna=False)
                for bin_label in order:
                    n = int(counts.get(bin_label, 0))
                    rows.append(
                        {
                            "metric": metric,
                            "pocket_class": pocket_class,
                            "outcome": outcome,
                            "bin": bin_label,
                            "n": n,
                            "share_within_group": float(n / total),
                        }
                    )
    return pd.DataFrame(rows)


def size_stratified_success_summary(df: pd.DataFrame, success_col: str, n_bins: int) -> pd.DataFrame:
    rows = []
    working = df.copy()
    continuous = {
        "pocket_overlap_pair": "quantile",
        "log2_ratio_holo_to_apo_avg_plus1": "quantile",
    }
    for metric, kind in {**{m: "verbal" for m in VERBAL_METRICS}, **continuous}.items():
        temp = working.copy()
        if kind == "verbal":
            temp["metric_bin"] = temp[metric].astype(str)
            order = VERBAL_METRICS[metric]
        else:
            vals = pd.to_numeric(temp[metric], errors="coerce")
            valid = vals.notna()
            if valid.sum() < n_bins * 20:
                continue
            temp["metric_bin"] = pd.Series(index=temp.index, dtype="object")
            bins = pd.qcut(vals[valid], q=n_bins, labels=[f"Q{i}" for i in range(1, n_bins + 1)], duplicates="drop")
            temp.loc[valid, "metric_bin"] = bins.astype(str)
            order = [f"Q{i}" for i in range(1, n_bins + 1) if f"Q{i}" in set(temp["metric_bin"].dropna())]
        for pocket_class in ["apo", "holo"]:
            class_df = temp[temp["pocket_class"] == pocket_class].copy()
            for size_bin, sgrp in class_df.groupby("size_bin", dropna=False):
                for metric_bin, grp in sgrp.groupby("metric_bin", dropna=False):
                    success = pd.to_numeric(grp[success_col], errors="coerce")
                    rows.append(
                        {
                            "metric": metric,
                            "metric_kind": kind,
                            "pocket_class": pocket_class,
                            "size_bin": str(size_bin),
                            "metric_bin": str(metric_bin),
                            "n": int(success.notna().sum()),
                            "success_rate": float(success.mean()),
                            "success_percent": float(success.mean() * 100.0),
                        }
                    )
    return pd.DataFrame(rows)


def write_manifest(output_dir: Path, df: pd.DataFrame, args: argparse.Namespace, size_order: list[str]) -> None:
    manifest = {
        "p2rank_table": str(args.p2rank_table),
        "success_col": args.success_col,
        "size_metric": args.size_metric,
        "size_bins": args.size_bins,
        "profile_bins": args.profile_bins,
        "n_rows": int(len(df)),
        "size_order": size_order,
        "class_counts": df["pocket_class"].value_counts(dropna=False).to_dict(),
        "success_percent": float(pd.to_numeric(df[args.success_col], errors="coerce").mean() * 100.0),
    }
    with (output_dir / "tables" / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def main() -> None:
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.p2rank_table)
    df, size_order = add_size_bins(df, args.size_metric, args.size_bins)
    valid = df["size_bin"].notna() & pd.to_numeric(df[args.success_col], errors="coerce").notna()
    df = df[valid].copy()

    verbal = verbal_profile_summary(df, args.success_col)
    quantile = quantile_profile_summary(df, args.success_col, args.profile_bins)
    size_strat = size_stratified_success_summary(df, args.success_col, args.profile_bins)

    df.to_csv(tables_dir / "p2rank_success_failure_merged.csv", index=False)
    verbal.to_csv(tables_dir / "p2rank_success_failure_verbal_profiles.csv", index=False)
    quantile.to_csv(tables_dir / "p2rank_success_failure_quantile_profiles.csv", index=False)
    size_strat.to_csv(tables_dir / "p2rank_success_failure_size_stratified.csv", index=False)
    write_manifest(args.output_dir, df, args, size_order)
    print(f"[SAVE] {args.output_dir}")


if __name__ == "__main__":
    main()
