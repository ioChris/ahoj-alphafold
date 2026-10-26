from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_p2rank_ubs_run4 import continuous_correlations, pairwise_composite_correlations
from write_p2rank_variable_rank_report import df_to_md, nice_feature

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
DEFAULT_ANALYSIS_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1"
DEFAULT_EVAL_DIR = (
    cfg.P2RANK_PRED_DIR
    / "ubs_run_4"
    / "UBS2"
    / "eval_predict_st_100_p2rank_intersection_geom_CENTER_explicit"
)
DEFAULT_REPORT_PATH = DEFAULT_ANALYSIS_DIR / "validation" / "p2rank_findings_variable_distance.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write markdown findings for variable-distance P2Rank DCA metrics.")
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--p2rank_eval_dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--output_md", type=Path, default=DEFAULT_REPORT_PATH)
    p.add_argument("--top_corr_n", type=int, default=12)
    p.add_argument("--top_pair_n", type=int, default=10)
    return p.parse_args()


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace(" ", "_") for c in out.columns]
    return out


def _load_base(analysis_dir: Path, eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pd.read_csv(analysis_dir / "tables" / "p2rank_st100_merged_per_pocket.csv", low_memory=False)
    ranks = _clean_columns(pd.read_csv(eval_dir / "cases" / "ranks.csv", low_memory=False))
    return merged, ranks


def _dynamic_cutoff(rg: pd.Series) -> tuple[pd.Series, float]:
    rg = pd.to_numeric(rg, errors="coerce")
    rg_ref = float(rg.dropna().median())
    cutoff = 4.0 * np.sqrt(rg_ref / rg)
    cutoff = cutoff.clip(lower=2.0, upper=6.0)
    return cutoff, rg_ref


def _add_variable_distance_metrics(merged: pd.DataFrame, ranks: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    df = merged.copy()
    df["dca4rank"] = pd.to_numeric(df["dca4rank"], errors="coerce")
    df["ligands_per_protein"] = pd.to_numeric(df["ligands_per_protein"], errors="coerce")
    df["n_unp_pockets"] = pd.to_numeric(df["n_unp_pockets"], errors="coerce")
    df["n_unp_pockets_multichain"] = pd.to_numeric(df["n_unp_pockets_multichain"], errors="coerce")
    df["rg"] = pd.to_numeric(df["rg"], errors="coerce")

    valid_sites_n = df["ligands_per_protein"].notna() & (df["ligands_per_protein"] >= 1)
    df["success_dca4_sites"] = np.where(
        valid_sites_n,
        (df["dca4rank"] > 0) & (df["dca4rank"] <= df["ligands_per_protein"]),
        np.nan,
    )

    cutoff, rg_ref = _dynamic_cutoff(df["rg"])
    df["dca_var_cutoff"] = cutoff
    df["dca_var_cutoff_rounded"] = cutoff.round().clip(lower=2, upper=12).astype("Int64")

    rank_cols = {i: f"DCA({i}.0)" for i in range(2, 13)}
    ranks = ranks.copy()
    ranks["ligand"] = ranks["ligand"].astype(str).str.strip()
    keep_cols = ["ligand"] + list(rank_cols.values())
    ranks = ranks[keep_cols].copy()
    for col in rank_cols.values():
        ranks[col] = pd.to_numeric(ranks[col], errors="coerce")
    df = df.merge(ranks, left_on="unit_uid", right_on="ligand", how="left")

    var_rank = pd.Series(np.nan, index=df.index, dtype=float)
    for i, col in rank_cols.items():
        m = df["dca_var_cutoff_rounded"] == i
        if m.any():
            var_rank.loc[m] = pd.to_numeric(df.loc[m, col], errors="coerce")
    df["dca_var_rank"] = var_rank

    valid_rank = df["dca_var_rank"].notna()
    df["success_dca_var_top1"] = np.where(valid_rank, df["dca_var_rank"] == 1, np.nan)
    df["success_dca_var_topn"] = np.where(valid_rank, df["dca_var_rank"] > 0, np.nan)
    valid_sites = valid_rank & df["ligands_per_protein"].notna() & (df["ligands_per_protein"] >= 1)
    df["success_dca_var_sites"] = np.where(
        valid_sites,
        (df["dca_var_rank"] > 0) & (df["dca_var_rank"] <= df["ligands_per_protein"]),
        np.nan,
    )
    valid_tns = valid_rank & df["n_unp_pockets"].notna() & (df["n_unp_pockets"] >= 1)
    df["success_dca_var_tns"] = np.where(
        valid_tns,
        (df["dca_var_rank"] > 0) & (df["dca_var_rank"] <= df["n_unp_pockets"]),
        np.nan,
    )
    valid_tnm = valid_rank & df["n_unp_pockets_multichain"].notna() & (df["n_unp_pockets_multichain"] >= 1)
    df["success_dca_var_tnm"] = np.where(
        valid_tnm,
        (df["dca_var_rank"] > 0) & (df["dca_var_rank"] <= df["n_unp_pockets_multichain"]),
        np.nan,
    )
    return df, rg_ref


def _summary_rows(df: pd.DataFrame, metric_map: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for metric, label in metric_map:
        row = {"Metric": label}
        for cls, out_label in [("all", "All"), ("apo", "AF-Apo"), ("holo", "AF-Holo")]:
            if cls == "all":
                s = pd.to_numeric(df[metric], errors="coerce")
            else:
                s = pd.to_numeric(df.loc[df["pocket_class"] == cls, metric], errors="coerce")
            row[out_label] = f"{float(s.mean() * 100.0):.1f}%" if s.notna().any() else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _corr_table(df: pd.DataFrame, success_col: str, top_corr_n: int) -> pd.DataFrame:
    corr = continuous_correlations(df, success_col=success_col)
    priority_features = {
        "ligand_atoms",
        "site_residues",
        "rg",
        "site_radius",
        "pocket_p_holo",
        "pocket_p_apo",
        "winning_p",
        "pocket_density_pair",
        "pocket_separation_pair",
        "pocket_overlap_pair",
        "pocket_overlap_overall",
        "log2_ratio_holo_to_apo_avg_plus1",
        "n_holo_avg_plus1",
        "best_assigned_score",
        "best_assigned_proba",
        "n_unp_pockets",
        "n_unp_pockets_multichain",
        "dca_var_cutoff",
    }
    show = corr[corr["feature"].isin(priority_features)].copy()
    show["abs_r"] = pd.to_numeric(show["spearman_r_success"], errors="coerce").abs()
    show = show.sort_values("abs_r", ascending=False).head(top_corr_n).copy()
    show = show[["feature", "spearman_r_success", "pearson_r_success"]].rename(
        columns={"feature": "Feature", "spearman_r_success": "Spearman rho", "pearson_r_success": "Pearson r"}
    )
    show["Feature"] = show["Feature"].map(nice_feature)
    show["Spearman rho"] = show["Spearman rho"].map(lambda v: f"{v:+.3f}")
    show["Pearson r"] = show["Pearson r"].map(lambda v: f"{v:+.3f}")
    return show


def _pair_table(df: pd.DataFrame, success_col: str, top_pair_n: int) -> pd.DataFrame:
    features = [
        "pocket_p_holo",
        "pocket_p_apo",
        "winning_p",
        "pocket_separation_pair",
        "pocket_overlap_pair",
        "pocket_density_pair",
        "log2_ratio_holo_to_apo_avg_plus1",
        "n_holo_avg",
        "n_apo_avg",
        "ratio_n_holo_to_apo_avg",
        "predicted_pockets_total",
        "model_pocket_plddt",
        "dca_var_cutoff",
    ]
    pair_df = pairwise_composite_correlations(df, success_col=success_col, features=features, min_n=200)
    pair_df = pair_df[
        (pair_df["feature_x"] != "predicted_pockets_total")
        & (pair_df["feature_y"] != "predicted_pockets_total")
        & (pair_df["feature_x"] != "ratio_n_holo_to_apo_avg")
        & (pair_df["feature_y"] != "ratio_n_holo_to_apo_avg")
    ].copy()
    pair_df["Feature pair"] = pair_df["feature_x"].map(nice_feature) + " + " + pair_df["feature_y"].map(nice_feature)
    pair_df = pair_df.sort_values("spearman_r_composite", ascending=False).head(top_pair_n).copy()
    pair_df = pair_df[["Feature pair", "spearman_r_composite", "delta_vs_best_single_abs_r"]].rename(
        columns={
            "spearman_r_composite": "Composite rho",
            "delta_vs_best_single_abs_r": "Delta vs best single abs(rho)",
        }
    )
    pair_df["Composite rho"] = pair_df["Composite rho"].map(lambda v: f"{v:+.3f}")
    pair_df["Delta vs best single abs(rho)"] = pair_df["Delta vs best single abs(rho)"].map(lambda v: f"{v:+.3f}")
    return pair_df


def _size_quantile_comparison(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    vals = pd.to_numeric(df["rg"], errors="coerce")
    valid = vals.notna()
    labels = [f"Q{i}" for i in range(1, n_bins + 1)]
    bins = pd.qcut(vals[valid], q=n_bins, labels=labels, duplicates="drop")
    tmp = df.loc[valid, ["rg", "success_topn", "success_tns", "success_dca_var_topn", "success_dca_var_tns"]].copy()
    tmp["rg_bin"] = bins.astype(str)
    rows = []
    for bin_label, grp in tmp.groupby("rg_bin", sort=False):
        rows.append(
            {
                "rg quantile": bin_label,
                "n": int(len(grp)),
                "mean rg": f"{float(pd.to_numeric(grp['rg'], errors='coerce').mean()):.2f}",
                "Fixed Top-N": f"{float(pd.to_numeric(grp['success_topn'], errors='coerce').mean() * 100.0):.1f}%",
                "Var Top-N": f"{float(pd.to_numeric(grp['success_dca_var_topn'], errors='coerce').mean() * 100.0):.1f}%",
                "Fixed TNS": f"{float(pd.to_numeric(grp['success_tns'], errors='coerce').mean() * 100.0):.1f}%",
                "Var TNS": f"{float(pd.to_numeric(grp['success_dca_var_tns'], errors='coerce').mean() * 100.0):.1f}%",
            }
        )
    return pd.DataFrame(rows)


def build_report(df: pd.DataFrame, rg_ref: float, top_corr_n: int, top_pair_n: int) -> str:
    n_sites = len(df)
    n_proteins = df["protein_file"].nunique()

    fixed = _summary_rows(
        df,
        [
            ("success_top1", "Top-1"),
            ("success_topn", "Top-N"),
            ("success_dca4_sites", "DCA4-#sites"),
            ("success_tns", "TNS"),
            ("success_tnm", "TNM"),
        ],
    )
    variable = _summary_rows(
        df,
        [
            ("success_dca_var_top1", "DCA var Top-1"),
            ("success_dca_var_topn", "DCA var Top-N"),
            ("success_dca_var_sites", "DCA var #sites"),
            ("success_dca_var_tns", "DCA var TNS"),
            ("success_dca_var_tnm", "DCA var TNM"),
        ],
    )
    corr_var_tns = _corr_table(df, "success_dca_var_tns", top_corr_n)
    pair_var_tns = _pair_table(df, "success_dca_var_tns", top_pair_n)
    size_compare = _size_quantile_comparison(df)

    lines = [
        "# UBS2 Prediction Analysis Findings (Variable-Distance DCA)",
        "",
        "Dataset: `eval_predict_st_100_p2rank_intersection_geom_CENTER_explicit`",
        f"{n_sites:,} observed sites across {n_proteins:,} proteins (AlphaFold models).",
        "",
        "## Variable-Distance Rule",
        "",
        r"Per-pocket cutoff: `DCA_var(rg) = clip(4.0 * sqrt(median_rg / rg), 2.0, 6.0)`",
        f"- Median `rg` in this ST_100 set: `{rg_ref:.3f}`",
        "- The cutoff is stricter for larger pockets and looser for smaller pockets.",
        "- The continuous cutoff is rounded to the nearest available DCA threshold in `ranks.csv` and then clipped to the `2-6 Å` working range.",
        "",
        "## Fixed-Distance Reference Metrics",
        "",
        df_to_md(fixed),
        "",
        "## Variable-Distance Metrics",
        "",
        df_to_md(variable),
        "",
        "## Pocket-Size Quantile Comparison",
        "",
        "This comparison uses `rg` quintiles to test whether the variable-distance rule reduces the large-pocket success advantage.",
        "",
        df_to_md(size_compare),
        "",
        "## Feature Correlations for DCA var TNS",
        "",
        df_to_md(corr_var_tns),
        "",
        "## Feature Pairs for DCA var TNS",
        "",
        df_to_md(pair_var_tns),
        "",
        "## Notes",
        "",
        "- `DCA var #sites`, `DCA var TNS`, and `DCA var TNM` reuse the same rank tolerances as before, but apply them to the variable-distance DCA rank.",
        "- The main candidate within this family is `DCA var TNS`, since it combines the variable-distance cutoff with the AHoJ-derived true single-chain pocket count.",
        "- `DCA var` is exploratory here; it is not part of the standard P2Rank evaluator output.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    merged, ranks = _load_base(args.analysis_dir, args.p2rank_eval_dir)
    df, rg_ref = _add_variable_distance_metrics(merged, ranks)
    report = build_report(df, rg_ref, args.top_corr_n, args.top_pair_n)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"[SAVE] {args.output_md}")


if __name__ == "__main__":
    main()
