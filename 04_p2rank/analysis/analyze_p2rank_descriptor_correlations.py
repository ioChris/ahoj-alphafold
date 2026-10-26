from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
AHOJ_DB = cfg.AHOJ_DB_DIR

DEFAULT_P2RANK_MERGED = (
    WORK_DIR
    / "P2Rank"
    / "ubs_run_4"
    / "analysis_v1"
    / "tables"
    / "p2rank_st100_merged_per_pocket.csv"
)
DEFAULT_UNION_GEOM = (
    WORK_DIR
    / "afdb_dataset"
    / "st100"
    / "p2rank_datasets"
    / "st_100_union_geom"
    / "st_100_p2rank_union_geom.csv"
)
DEFAULT_DESCRIPTOR_SUMMARY = (
    AHOJ_DB
    / "Summaries"
    / "parsing_v3c5_DBv2b_561batches_511845jobs_80seqres_Noneobs_4.5res_0fullpcktaln_sums0"
    / "query_summaries_rog_DQ5.csv"
)
DEFAULT_RANGE_DIR = (
    AHOJ_DB
    / "Summaries"
    / "parsing_v3c3_DBv2b_561batches_518883jobs_80seqres_Noneobs_4.5res_0fullpcktaln"
)
DEFAULT_OUT_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1"

METRICS = ["pocket_rms", "volume", "sasa", "RoG"]
SUMMARY_SUFFIXES = [
    "mean",
    "std_dev",
    "iqr",
    "coefficient_of_variation",
    "confidence_interval_low",
    "confidence_interval_high",
]
SUCCESS_COLS = [
    "success_top1",
    "success_topn",
    "success_dca4_sites",
    "success_tns",
    "success_tnm",
    "success_dca_4A",
    "success_dca_5A",
    "success_dca_6A",
    "success_dca_8A",
    "success_dca_10A",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Join AHoJ pocket descriptor variability metrics to P2Rank ST_100 UBS success results."
    )
    p.add_argument("--p2rank_merged", type=Path, default=DEFAULT_P2RANK_MERGED)
    p.add_argument("--union_geom", type=Path, default=DEFAULT_UNION_GEOM)
    p.add_argument("--descriptor_summary", type=Path, default=DEFAULT_DESCRIPTOR_SUMMARY)
    p.add_argument("--range_dir", type=Path, default=DEFAULT_RANGE_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--threshold", default="ST_100")
    p.add_argument("--primary_success", default="success_tns")
    return p.parse_args()


def ahoj_query_to_job_name(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", "-", regex=True)
        .str.replace(r"-+$", "", regex=True)
    )


def split_jobs(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def build_unit_job_map(union_geom: pd.DataFrame, unit_ids: set[str]) -> tuple[pd.DataFrame, set[str]]:
    geom = union_geom[union_geom["site_uid"].isin(unit_ids)].copy()
    rows: list[dict[str, str]] = []
    for row in geom[["site_uid", "job_names"]].itertuples(index=False):
        for job in split_jobs(row.job_names):
            rows.append({"unit_uid": row.site_uid, "job_name": job})
    unit_jobs = pd.DataFrame(rows).drop_duplicates()
    return unit_jobs, set(unit_jobs["job_name"])


def load_descriptor_summary(path: Path, needed_jobs: set[str]) -> pd.DataFrame:
    cols = ["ahoj_query", "local_flexibility", "global_flexibility"]
    for metric in METRICS:
        for suffix in SUMMARY_SUFFIXES:
            col = f"{metric}_{suffix}"
            if col != "RoG_coefficient_of_variation":
                cols.append(col)
            else:
                cols.append(col)

    df = pd.read_csv(path, usecols=lambda c: c in set(cols), low_memory=False)
    df["job_name"] = ahoj_query_to_job_name(df["ahoj_query"])
    df = df[df["job_name"].isin(needed_jobs)].copy()
    df = df.drop(columns=["ahoj_query"])
    numeric_cols = [c for c in df.columns if c != "job_name"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(subset=["job_name"])


def _load_class_range(path: Path, prefix: str, needed_jobs: set[str]) -> pd.DataFrame:
    usecols = ["job_id", "pockets"]
    for metric in METRICS:
        usecols.extend([f"{metric}_min", f"{metric}_max", f"{metric}_avg"])
    df = pd.read_csv(path, usecols=lambda c: c in set(usecols), low_memory=False)
    df = df[df["job_id"].isin(needed_jobs)].copy()
    df["pockets"] = pd.to_numeric(df["pockets"], errors="coerce")
    for metric in METRICS:
        for stat_name in ["min", "max", "avg"]:
            col = f"{metric}_{stat_name}"
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{metric}_range"] = df[f"{metric}_max"] - df[f"{metric}_min"]
    rename = {"job_id": "job_name", "pockets": f"{prefix}_pockets"}
    for metric in METRICS:
        for stat_name in ["min", "max", "avg", "range"]:
            rename[f"{metric}_{stat_name}"] = f"{metric}_{prefix}_{stat_name}"
    return df.rename(columns=rename).drop_duplicates(subset=["job_name"])


def load_range_features(range_dir: Path, needed_jobs: set[str]) -> pd.DataFrame:
    apo = _load_class_range(range_dir / "apo_summaries.csv", "apo", needed_jobs)
    holo = _load_class_range(range_dir / "holo_summaries.csv", "holo", needed_jobs)
    merged = apo.merge(holo, on="job_name", how="outer")
    for metric in METRICS:
        merged[f"{metric}_all_min"] = merged[
            [f"{metric}_apo_min", f"{metric}_holo_min"]
        ].min(axis=1, skipna=True)
        merged[f"{metric}_all_max"] = merged[
            [f"{metric}_apo_max", f"{metric}_holo_max"]
        ].max(axis=1, skipna=True)
        merged[f"{metric}_all_range"] = (
            merged[f"{metric}_all_max"] - merged[f"{metric}_all_min"]
        )
        merged[f"{metric}_apo_holo_avg_delta"] = (
            merged[f"{metric}_holo_avg"] - merged[f"{metric}_apo_avg"]
        )
    return merged


def aggregate_to_ubs(unit_jobs: pd.DataFrame, job_features: pd.DataFrame) -> pd.DataFrame:
    joined = unit_jobs.merge(job_features, on="job_name", how="left")
    feature_cols = [c for c in joined.columns if c not in {"unit_uid", "job_name"}]
    numeric_cols = []
    for col in feature_cols:
        joined[col] = pd.to_numeric(joined[col], errors="coerce")
        if joined[col].notna().any():
            numeric_cols.append(col)

    agg_ops: dict[str, list[str]] = {}
    for col in numeric_cols:
        if col.endswith("_range") or col.endswith("_std_dev") or col.endswith("_iqr"):
            agg_ops[col] = ["mean", "max", "median"]
        else:
            agg_ops[col] = ["mean"]

    agg = joined.groupby("unit_uid", dropna=False).agg(agg_ops)
    agg.columns = [f"{col}_{op}_across_jobs" for col, op in agg.columns]
    agg = agg.reset_index()

    coverage = (
        joined.assign(has_descriptor=joined[numeric_cols].notna().any(axis=1) if numeric_cols else False)
        .groupby("unit_uid")
        .agg(
            descriptor_jobs=("job_name", "nunique"),
            descriptor_jobs_with_features=("has_descriptor", "sum"),
        )
        .reset_index()
    )
    agg = agg.merge(coverage, on="unit_uid", how="left")
    agg["descriptor_job_coverage"] = (
        agg["descriptor_jobs_with_features"] / agg["descriptor_jobs"].replace(0, np.nan)
    )
    return agg


def correlation_table(df: pd.DataFrame, feature_cols: list[str], success_cols: list[str]) -> pd.DataFrame:
    rows = []
    for success_col in success_cols:
        if success_col not in df.columns:
            continue
        y = pd.to_numeric(df[success_col], errors="coerce")
        for feature in feature_cols:
            x = pd.to_numeric(df[feature], errors="coerce")
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
                continue
            pearson = stats.pearsonr(x[mask], y[mask])
            spearman = stats.spearmanr(x[mask], y[mask])
            rows.append(
                {
                    "success_col": success_col,
                    "feature": feature,
                    "n": n,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "feature_mean": float(x[mask].mean()),
                    "success_rate": float(y[mask].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["success_col", "spearman_rho"], key=lambda s: s.abs() if s.name == "spearman_rho" else s
    )


def quantile_success_table(
    df: pd.DataFrame, feature_cols: list[str], success_col: str, n_bins: int = 10
) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[success_col], errors="coerce")
    for feature in feature_cols:
        x = pd.to_numeric(df[feature], errors="coerce")
        sub = pd.DataFrame({"x": x, "success": y}).dropna()
        if len(sub) < n_bins or sub["x"].nunique() < 2:
            continue
        try:
            sub["bin"] = pd.qcut(sub["x"], q=n_bins, duplicates="drop")
        except ValueError:
            continue
        grouped = sub.groupby("bin", observed=True)
        for idx, (bin_label, grp) in enumerate(grouped, start=1):
            rows.append(
                {
                    "success_col": success_col,
                    "feature": feature,
                    "bin_index": idx,
                    "bin_label": str(bin_label),
                    "n": int(len(grp)),
                    "feature_min": float(grp["x"].min()),
                    "feature_max": float(grp["x"].max()),
                    "feature_mean": float(grp["x"].mean()),
                    "success_rate": float(grp["success"].mean()),
                }
            )
    return pd.DataFrame(rows)


def nice_label(feature: str) -> str:
    label = feature.replace("_mean_across_jobs", "")
    label = label.replace("_max_across_jobs", " max")
    label = label.replace("_median_across_jobs", " median")
    label = label.replace("_", " ")
    return label


def plot_top_correlations(corr: pd.DataFrame, success_col: str, out_path: Path, top_n: int = 24) -> None:
    sub = corr[corr["success_col"] == success_col].copy()
    if sub.empty:
        return
    sub = sub.reindex(sub["spearman_rho"].abs().sort_values(ascending=False).index).head(top_n)
    sub = sub.sort_values("spearman_rho")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(sub))))
    colors = np.where(sub["spearman_rho"] >= 0, "#2a9d8f", "#c44536")
    ax.barh([nice_label(x) for x in sub["feature"]], sub["spearman_rho"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Spearman rho with P2Rank success")
    ax.set_title(f"Descriptor correlations ({success_col})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_range_quantiles(qtab: pd.DataFrame, success_col: str, out_path: Path) -> None:
    features = [
        f"{metric}_all_range_mean_across_jobs"
        for metric in METRICS
        if f"{metric}_all_range_mean_across_jobs" in set(qtab["feature"])
    ]
    if not features:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    axes = axes.flatten()
    for ax, feature in zip(axes, features):
        sub = qtab[(qtab["success_col"] == success_col) & (qtab["feature"] == feature)].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        ax.plot(sub["bin_index"], sub["success_rate"] * 100.0, marker="o", color="#2f6f8f")
        ax.set_title(nice_label(feature), fontsize=10)
        ax.set_xlabel("Descriptor range decile")
        ax.set_ylabel("Success rate (%)")
        ax.grid(alpha=0.25)
    for ax in axes[len(features) :]:
        ax.set_axis_off()
    fig.suptitle(f"P2Rank success across descriptor range bins ({success_col})", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_range_correlation_bars(corr: pd.DataFrame, success_col: str, out_path: Path) -> None:
    features = [f"{metric}_all_range_mean_across_jobs" for metric in METRICS]
    sub = corr[(corr["success_col"] == success_col) & (corr["feature"].isin(features))].copy()
    if sub.empty:
        return
    order = {feature: i for i, feature in enumerate(features)}
    sub["order"] = sub["feature"].map(order)
    sub = sub.sort_values("order")
    labels = ["Pocket RMSD", "Volume", "SASA", "RoG"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, sub["spearman_rho"], color=["#6a4c93", "#2a9d8f", "#e9a03b", "#3d405b"])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Spearman rho with P2Rank success")
    ax.set_title(f"Descriptor range correlations ({success_col})")
    ax.grid(axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table_dir = args.out_dir / "tables"
    plot_dir = args.out_dir / "plots" / "descriptor_metrics"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    p2rank = pd.read_csv(args.p2rank_merged, low_memory=False)
    unit_ids = set(p2rank["unit_uid"].astype(str))
    union_geom = pd.read_csv(
        args.union_geom,
        usecols=["threshold", "site_uid", "job_names"],
        low_memory=False,
    )
    union_geom = union_geom[union_geom["threshold"] == args.threshold].copy()
    unit_jobs, needed_jobs = build_unit_job_map(union_geom, unit_ids)

    descriptor_summary = load_descriptor_summary(args.descriptor_summary, needed_jobs)
    range_features = load_range_features(args.range_dir, needed_jobs)
    job_features = descriptor_summary.merge(range_features, on="job_name", how="outer")

    ubs_features = aggregate_to_ubs(unit_jobs, job_features)
    merged = p2rank.merge(ubs_features, on="unit_uid", how="left", validate="one_to_one")

    feature_cols = [
        c
        for c in ubs_features.columns
        if c
        not in {
            "unit_uid",
            "descriptor_jobs",
            "descriptor_jobs_with_features",
            "descriptor_job_coverage",
        }
    ]
    success_cols = [c for c in SUCCESS_COLS if c in merged.columns]
    corr = correlation_table(merged, feature_cols, success_cols)

    range_features_for_bins = [
        f"{metric}_all_range_mean_across_jobs"
        for metric in METRICS
        if f"{metric}_all_range_mean_across_jobs" in merged.columns
    ]
    qtab = quantile_success_table(
        merged,
        range_features_for_bins,
        args.primary_success,
        n_bins=10,
    )

    feature_out = table_dir / "p2rank_st100_descriptor_features_per_ubs.csv"
    merged_out = table_dir / "p2rank_st100_merged_with_descriptor_features.csv"
    corr_out = table_dir / "p2rank_descriptor_correlations.csv"
    qtab_out = table_dir / "p2rank_descriptor_quantile_success.csv"

    ubs_features.to_csv(feature_out, index=False)
    merged.to_csv(merged_out, index=False)
    corr.to_csv(corr_out, index=False)
    qtab.to_csv(qtab_out, index=False)

    plot_top_correlations(
        corr,
        args.primary_success,
        plot_dir / f"descriptor_top_spearman_{args.primary_success}.png",
    )
    plot_range_quantiles(
        qtab,
        args.primary_success,
        plot_dir / f"descriptor_range_deciles_{args.primary_success}.png",
    )
    plot_range_correlation_bars(
        corr,
        args.primary_success,
        plot_dir / f"descriptor_range_spearman_{args.primary_success}.png",
    )

    manifest = {
        "threshold": args.threshold,
        "primary_success": args.primary_success,
        "inputs": {
            "p2rank_merged": str(args.p2rank_merged),
            "union_geom": str(args.union_geom),
            "descriptor_summary": str(args.descriptor_summary),
            "range_dir": str(args.range_dir),
        },
        "outputs": {
            "descriptor_features": str(feature_out),
            "merged_with_descriptor_features": str(merged_out),
            "correlations": str(corr_out),
            "quantile_success": str(qtab_out),
            "plot_dir": str(plot_dir),
        },
        "n_p2rank_rows": int(len(p2rank)),
        "n_unit_jobs": int(len(unit_jobs)),
        "n_needed_jobs": int(len(needed_jobs)),
        "n_descriptor_summary_jobs_matched": int(descriptor_summary["job_name"].nunique()),
        "n_range_jobs_matched": int(range_features["job_name"].nunique()),
        "n_ubs_with_any_descriptor": int(
            merged["descriptor_job_coverage"].fillna(0).gt(0).sum()
        ),
        "mean_descriptor_job_coverage": float(
            merged["descriptor_job_coverage"].mean(skipna=True)
        ),
        "metrics": METRICS,
        "note": (
            "Exact all_range features are computed from apo_summaries/holo_summaries "
            "as max(apo/holo max) minus min(apo/holo min), then aggregated across "
            "member jobs per UBS. Summary std/IQR/CV features come from query_summaries_rog_DQ5."
        ),
    }
    manifest_out = table_dir / "p2rank_descriptor_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
