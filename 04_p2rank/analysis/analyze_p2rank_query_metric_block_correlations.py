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
DEFAULT_QUERY_SUMMARY = (
    AHOJ_DB
    / "Summaries"
    / "parsing_v3c5_DBv2b_561batches_511845jobs_80seqres_Noneobs_4.5res_0fullpcktaln_sums0"
    / "query_summaries_rog_DQ5.csv"
)
DEFAULT_OUT_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1"

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
METRIC_PREFIXES = [
    "pocket_rms",
    "tm_score_i",
    "tm_score",
    "volume",
    "sasa",
    "RoG",
    "rmsd",
]
AGGREGATIONS = ["mean", "median", "max"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Correlate the full AHoJ query-summary metric block with P2Rank "
            "ST_100 UBS prediction success."
        )
    )
    p.add_argument("--p2rank_merged", type=Path, default=DEFAULT_P2RANK_MERGED)
    p.add_argument("--union_geom", type=Path, default=DEFAULT_UNION_GEOM)
    p.add_argument("--query_summary", type=Path, default=DEFAULT_QUERY_SUMMARY)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--threshold", default="ST_100")
    p.add_argument("--metric_start", default="local_flexibility")
    p.add_argument("--metric_end", default="rmsd_coefficient_of_variation")
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


def metric_block_columns(path: Path, start_col: str, end_col: str) -> list[str]:
    cols = list(pd.read_csv(path, nrows=0).columns)
    try:
        start_idx = cols.index(start_col)
        end_idx = cols.index(end_col)
    except ValueError as exc:
        raise SystemExit(f"Could not find metric block boundary column: {exc}") from exc
    if end_idx < start_idx:
        raise SystemExit(f"Metric block end {end_col!r} appears before start {start_col!r}")
    return cols[start_idx : end_idx + 1]


def build_unit_job_map(union_geom: pd.DataFrame, unit_ids: set[str]) -> tuple[pd.DataFrame, set[str]]:
    geom = union_geom[union_geom["site_uid"].isin(unit_ids)].copy()
    rows: list[dict[str, str]] = []
    for row in geom[["site_uid", "job_names"]].itertuples(index=False):
        for job in split_jobs(row.job_names):
            rows.append({"unit_uid": row.site_uid, "job_name": job})
    unit_jobs = pd.DataFrame(rows).drop_duplicates()
    return unit_jobs, set(unit_jobs["job_name"])


def load_query_metrics(path: Path, metric_cols: list[str], needed_jobs: set[str]) -> pd.DataFrame:
    wanted = set(metric_cols) | {"ahoj_query"}
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    df["job_name"] = ahoj_query_to_job_name(df["ahoj_query"])
    df = df[df["job_name"].isin(needed_jobs)].copy()
    df = df.drop(columns=["ahoj_query"])
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(subset=["job_name"])


def add_ci_width_metrics(df: pd.DataFrame, metric_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    derived_cols: list[str] = []
    for prefix in METRIC_PREFIXES:
        low = f"{prefix}_confidence_interval_low"
        high = f"{prefix}_confidence_interval_high"
        out = f"{prefix}_confidence_interval_width"
        if low in df.columns and high in df.columns:
            df[out] = df[high] - df[low]
            derived_cols.append(out)
    return df, metric_cols + derived_cols


def aggregate_to_ubs(
    unit_jobs: pd.DataFrame,
    job_features: pd.DataFrame,
    metric_cols: list[str],
) -> pd.DataFrame:
    joined = unit_jobs.merge(job_features, on="job_name", how="left")
    agg_ops = {col: AGGREGATIONS for col in metric_cols if col in joined.columns}
    agg = joined.groupby("unit_uid", dropna=False).agg(agg_ops)
    agg.columns = [f"{col}_{op}_across_jobs" for col, op in agg.columns]
    agg = agg.reset_index()

    has_any_metric = joined[metric_cols].notna().any(axis=1)
    coverage = (
        joined.assign(has_metric=has_any_metric)
        .groupby("unit_uid")
        .agg(
            metric_jobs=("job_name", "nunique"),
            metric_jobs_with_values=("has_metric", "sum"),
        )
        .reset_index()
    )
    agg = agg.merge(coverage, on="unit_uid", how="left")
    agg["metric_job_coverage"] = (
        agg["metric_jobs_with_values"] / agg["metric_jobs"].replace(0, np.nan)
    )
    return agg


def source_column_from_feature(feature: str) -> tuple[str, str]:
    for agg in AGGREGATIONS:
        suffix = f"_{agg}_across_jobs"
        if feature.endswith(suffix):
            return feature[: -len(suffix)], agg
    return feature, "raw"


def describe_source_column(source_col: str) -> tuple[str, str]:
    if source_col == "local_flexibility":
        return "flexibility", "local_flexibility"
    if source_col == "global_flexibility":
        return "flexibility", "global_flexibility"
    for prefix in METRIC_PREFIXES:
        metric_prefix = f"{prefix}_"
        if source_col.startswith(metric_prefix):
            return prefix, source_col[len(metric_prefix) :]
    return "other", source_col


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
            source_col, across_jobs_agg = source_column_from_feature(feature)
            metric, subscore = describe_source_column(source_col)
            rows.append(
                {
                    "success_col": success_col,
                    "feature": feature,
                    "source_column": source_col,
                    "source_metric": metric,
                    "subscore": subscore,
                    "across_jobs_agg": across_jobs_agg,
                    "n": n,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "abs_spearman_rho": float(abs(spearman.statistic)),
                    "feature_mean": float(x[mask].mean()),
                    "success_rate": float(y[mask].mean()),
                }
            )
    corr = pd.DataFrame(rows)
    if corr.empty:
        return corr
    return corr.sort_values(
        ["success_col", "abs_spearman_rho", "feature"],
        ascending=[True, False, True],
    )


def best_by(corr: pd.DataFrame, primary_success: str, group_cols: list[str]) -> pd.DataFrame:
    sub = corr[corr["success_col"] == primary_success].copy()
    if sub.empty:
        return sub
    idx = sub.groupby(group_cols, dropna=False)["abs_spearman_rho"].idxmax()
    return sub.loc[idx].sort_values("abs_spearman_rho", ascending=False)


def mean_agg_matrix(corr: pd.DataFrame, primary_success: str) -> pd.DataFrame:
    sub = corr[
        (corr["success_col"] == primary_success)
        & (corr["across_jobs_agg"] == "mean")
        & (corr["source_metric"] != "flexibility")
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(
        index="source_metric",
        columns="subscore",
        values="spearman_rho",
        aggfunc="first",
    ).reset_index()


def nice_label(feature: str) -> str:
    source_col, agg = source_column_from_feature(feature)
    label = source_col.replace("_", " ")
    if agg != "mean":
        label = f"{label} ({agg})"
    return label


def plot_top_correlations(corr: pd.DataFrame, primary_success: str, out_path: Path, top_n: int = 30) -> None:
    sub = corr[corr["success_col"] == primary_success].copy()
    if sub.empty:
        return
    sub = sub.head(top_n).sort_values("spearman_rho")
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(sub))))
    colors = np.where(sub["spearman_rho"] >= 0, "#2a9d8f", "#c44536")
    ax.barh([nice_label(x) for x in sub["feature"]], sub["spearman_rho"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Spearman rho with P2Rank success")
    ax.set_title(f"Query-summary metric block correlations ({primary_success})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_metric_subscore_heatmap(matrix: pd.DataFrame, out_path: Path) -> None:
    if matrix.empty:
        return
    mat = matrix.set_index("source_metric")
    preferred_cols = [
        "mean",
        "std_dev",
        "iqr",
        "coefficient_of_variation",
        "confidence_interval_width",
        "confidence_interval_low",
        "confidence_interval_high",
    ]
    cols = [c for c in preferred_cols if c in mat.columns]
    cols.extend([c for c in mat.columns if c not in cols])
    mat = mat[cols]

    fig, ax = plt.subplots(figsize=(1.5 * len(mat.columns) + 2.2, 0.55 * len(mat) + 2.2))
    im = ax.imshow(mat.values.astype(float), cmap="coolwarm", vmin=-0.6, vmax=0.6)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels([c.replace("_", " ") for c in mat.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Spearman rho by metric and subscore (mean across UBS jobs)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table_dir = args.out_dir / "tables"
    plot_dir = args.out_dir / "plots" / "query_metric_block"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = metric_block_columns(args.query_summary, args.metric_start, args.metric_end)

    p2rank = pd.read_csv(args.p2rank_merged, low_memory=False)
    unit_ids = set(p2rank["unit_uid"].astype(str))
    union_geom = pd.read_csv(
        args.union_geom,
        usecols=["threshold", "site_uid", "job_names"],
        low_memory=False,
    )
    union_geom = union_geom[union_geom["threshold"] == args.threshold].copy()
    unit_jobs, needed_jobs = build_unit_job_map(union_geom, unit_ids)

    query_metrics = load_query_metrics(args.query_summary, metric_cols, needed_jobs)
    query_metrics, metric_cols_with_derived = add_ci_width_metrics(query_metrics, metric_cols)
    ubs_features = aggregate_to_ubs(unit_jobs, query_metrics, metric_cols_with_derived)

    success_cols = [c for c in SUCCESS_COLS if c in p2rank.columns]
    feature_cols = [
        c
        for c in ubs_features.columns
        if c not in {"unit_uid", "metric_jobs", "metric_jobs_with_values", "metric_job_coverage"}
    ]
    merged_cols = ["unit_uid", *success_cols]
    merged = p2rank[merged_cols].merge(ubs_features, on="unit_uid", how="left", validate="one_to_one")
    corr = correlation_table(merged, feature_cols, success_cols)

    best_metric = best_by(corr, args.primary_success, ["source_metric"])
    best_subscore = best_by(corr, args.primary_success, ["subscore"])
    best_metric_subscore = best_by(corr, args.primary_success, ["source_metric", "subscore"])
    matrix = mean_agg_matrix(corr, args.primary_success)

    features_out = table_dir / "p2rank_st100_query_metric_block_features_per_ubs.csv"
    merged_out = table_dir / "p2rank_st100_merged_with_query_metric_block_features.csv"
    corr_out = table_dir / "p2rank_query_metric_block_correlations.csv"
    best_metric_out = table_dir / "p2rank_query_metric_block_primary_best_by_metric.csv"
    best_subscore_out = table_dir / "p2rank_query_metric_block_primary_best_by_subscore.csv"
    best_metric_subscore_out = table_dir / "p2rank_query_metric_block_primary_best_by_metric_subscore.csv"
    matrix_out = table_dir / "p2rank_query_metric_block_primary_mean_spearman_matrix.csv"

    ubs_features.to_csv(features_out, index=False)
    merged.to_csv(merged_out, index=False)
    corr.to_csv(corr_out, index=False)
    best_metric.to_csv(best_metric_out, index=False)
    best_subscore.to_csv(best_subscore_out, index=False)
    best_metric_subscore.to_csv(best_metric_subscore_out, index=False)
    matrix.to_csv(matrix_out, index=False)

    plot_top_correlations(
        corr,
        args.primary_success,
        plot_dir / f"query_metric_block_top_spearman_{args.primary_success}.png",
    )
    plot_metric_subscore_heatmap(
        matrix,
        plot_dir / f"query_metric_block_metric_subscore_heatmap_{args.primary_success}.png",
    )

    manifest = {
        "threshold": args.threshold,
        "primary_success": args.primary_success,
        "metric_block": {
            "query_summary": str(args.query_summary),
            "start_column": args.metric_start,
            "end_column": args.metric_end,
            "n_raw_metric_columns": len(metric_cols),
            "raw_metric_columns": metric_cols,
            "derived_columns": [
                c for c in metric_cols_with_derived if c not in set(metric_cols)
            ],
            "across_jobs_aggregations": AGGREGATIONS,
        },
        "inputs": {
            "p2rank_merged": str(args.p2rank_merged),
            "union_geom": str(args.union_geom),
        },
        "outputs": {
            "features_per_ubs": str(features_out),
            "merged_with_features": str(merged_out),
            "correlations": str(corr_out),
            "best_by_metric": str(best_metric_out),
            "best_by_subscore": str(best_subscore_out),
            "best_by_metric_subscore": str(best_metric_subscore_out),
            "mean_spearman_matrix": str(matrix_out),
            "plot_dir": str(plot_dir),
        },
        "n_p2rank_rows": int(len(p2rank)),
        "n_unit_jobs": int(len(unit_jobs)),
        "n_needed_jobs": int(len(needed_jobs)),
        "n_query_summary_jobs_matched": int(query_metrics["job_name"].nunique()),
        "n_ubs_with_any_metric": int(merged["metric_job_coverage"].fillna(0).gt(0).sum()),
        "mean_metric_job_coverage": float(merged["metric_job_coverage"].mean(skipna=True)),
        "note": (
            "The source file does not include min/max/range columns in this metric block. "
            "confidence_interval_width is derived as high-low for interpretability."
        ),
    }
    manifest_out = table_dir / "p2rank_query_metric_block_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
