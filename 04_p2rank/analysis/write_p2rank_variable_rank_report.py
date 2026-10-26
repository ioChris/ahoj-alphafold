from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_p2rank_ubs_run4 import continuous_correlations, pairwise_composite_correlations

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
DEFAULT_REPORT_PATH = DEFAULT_ANALYSIS_DIR / "validation" / "p2rank_findings_variable_rank.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write markdown findings for variable-rank P2Rank metrics (TNS/TNM).")
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--output_md", type=Path, default=DEFAULT_REPORT_PATH)
    p.add_argument("--top_corr_n", type=int, default=12)
    p.add_argument("--top_pair_n", type=int, default=10)
    return p.parse_args()


def nice_feature(name: str) -> str:
    mapping = {
        "ligand_atoms": "#atoms",
        "site_residues": "#residues",
        "site_radius": "site_radius",
        "rg": "rg",
        "pocket_p_holo": "p_holo",
        "pocket_p_apo": "p_apo",
        "winning_p": "winning_p",
        "pocket_density_pair": "density_pair",
        "pocket_separation_pair": "separation_pair",
        "pocket_overlap_pair": "overlap_pair",
        "pocket_overlap_overall": "overlap_overall",
        "pocket_overlap_mode": "overlap_mode",
        "model_pocket_plddt": "pLDDT",
        "n_apo_avg": "n_apo_avg",
        "n_holo_avg": "n_holo_avg",
        "n_holo_avg_plus1": "n_holo_avg + 1",
        "log2_ratio_holo_to_apo_avg_plus1": "log2((n_holo_avg+1)/n_apo_avg)",
        "best_assigned_score": "best_assigned_score",
        "best_assigned_proba": "best_assigned_proba",
        "n_unp_pockets": "TNS",
        "n_unp_pockets_multichain": "TNM",
    }
    return mapping.get(name, name)


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    return df.to_markdown(index=False)


def load_inputs(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    tables = analysis_dir / "tables"
    merged = pd.read_csv(tables / "p2rank_st100_merged_per_pocket.csv", low_memory=False)
    merged["dca4rank"] = pd.to_numeric(merged["dca4rank"], errors="coerce")
    merged["ligands_per_protein"] = pd.to_numeric(merged["ligands_per_protein"], errors="coerce")
    valid_sites_n = merged["ligands_per_protein"].notna() & (merged["ligands_per_protein"] >= 1)
    merged["success_dca4_sites"] = pd.Series(pd.NA, index=merged.index, dtype="boolean")
    merged.loc[valid_sites_n, "success_dca4_sites"] = (
        (merged.loc[valid_sites_n, "dca4rank"] > 0)
        & (merged.loc[valid_sites_n, "dca4rank"] <= merged.loc[valid_sites_n, "ligands_per_protein"])
    ).astype(bool)

    class_summary = pd.read_csv(tables / "p2rank_class_success_by_cutoff.csv", low_memory=False)
    if "success_dca4_sites" not in set(class_summary["success_metric"].astype(str)):
        rows = []
        for pocket_class in ["apo", "holo"]:
            grp = merged[merged["pocket_class"] == pocket_class].copy()
            success = pd.to_numeric(grp["success_dca4_sites"], errors="coerce")
            rows.append(
                {
                    "success_metric": "success_dca4_sites",
                    "pocket_class": pocket_class,
                    "n": int(success.notna().sum()),
                    "success_rate": float(success.mean()),
                    "success_percent": float(success.mean() * 100.0),
                }
            )
        success_all = pd.to_numeric(merged["success_dca4_sites"], errors="coerce")
        rows.append(
            {
                "success_metric": "success_dca4_sites",
                "pocket_class": "all",
                "n": int(success_all.notna().sum()),
                "success_rate": float(success_all.mean()),
                "success_percent": float(success_all.mean() * 100.0),
            }
        )
        class_summary = pd.concat([class_summary, pd.DataFrame(rows)], ignore_index=True)

    pair_features = [
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
    ]

    corr_dca4_sites_path = tables / "p2rank_dca4_sites_continuous_correlations.csv"
    if corr_dca4_sites_path.exists():
        corr_dca4_sites = pd.read_csv(corr_dca4_sites_path, low_memory=False)
    else:
        corr_dca4_sites = continuous_correlations(merged, success_col="success_dca4_sites")
        corr_dca4_sites.to_csv(corr_dca4_sites_path, index=False)

    pair_dca4_sites_path = tables / "p2rank_dca4_sites_pairwise_composite_correlations.csv"
    if pair_dca4_sites_path.exists():
        pair_dca4_sites = pd.read_csv(pair_dca4_sites_path, low_memory=False)
    else:
        pair_dca4_sites = pairwise_composite_correlations(
            merged,
            success_col="success_dca4_sites",
            features=pair_features,
            min_n=200,
        )
        pair_dca4_sites.to_csv(pair_dca4_sites_path, index=False)

    return {
        "merged": merged,
        "class_summary": class_summary,
        "corr_topn": pd.read_csv(tables / "p2rank_topn_continuous_correlations.csv", low_memory=False),
        "corr_dca4_sites": corr_dca4_sites,
        "corr_tns": pd.read_csv(tables / "p2rank_tns_continuous_correlations.csv", low_memory=False),
        "corr_tnm": pd.read_csv(tables / "p2rank_tnm_continuous_correlations.csv", low_memory=False),
        "pair_topn": pd.read_csv(tables / "p2rank_topn_pairwise_composite_correlations.csv", low_memory=False),
        "pair_dca4_sites": pair_dca4_sites,
        "pair_tns": pd.read_csv(tables / "p2rank_tns_pairwise_composite_correlations.csv", low_memory=False),
        "pair_tnm": pd.read_csv(tables / "p2rank_tnm_pairwise_composite_correlations.csv", low_memory=False),
    }


def _summary_table(class_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("success_top1", "Top-1"),
        ("success_topn", "Top-N"),
        ("success_dca4_sites", "DCA4-#sites"),
        ("success_tns", "TNS"),
        ("success_tnm", "TNM"),
    ]
    rows = []
    for metric, label in metrics:
        sub = class_summary[class_summary["success_metric"] == metric].copy()
        if sub.empty:
            continue
        row = {"Metric": label}
        for cls, out_label in [("all", "All"), ("apo", "AF-Apo"), ("holo", "AF-Holo")]:
            m = sub[sub["pocket_class"] == cls]
            if m.empty:
                row[out_label] = ""
            else:
                row[out_label] = f"{float(m['success_percent'].iloc[0]):.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def _top_corr_table(corr: pd.DataFrame, top_corr_n: int) -> pd.DataFrame:
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
    }
    show = corr.copy()
    show = show[pd.to_numeric(show["spearman_r_success"], errors="coerce").notna()].copy()
    show["abs_r"] = pd.to_numeric(show["spearman_r_success"], errors="coerce").abs()
    show = show[show["feature"].isin(priority_features)].sort_values("abs_r", ascending=False).head(top_corr_n).copy()
    show = show[["feature", "spearman_r_success", "pearson_r_success"]].rename(
        columns={
            "feature": "Feature",
            "spearman_r_success": "Spearman rho",
            "pearson_r_success": "Pearson r",
        }
    )
    show["Feature"] = show["Feature"].map(nice_feature)
    show["Spearman rho"] = show["Spearman rho"].map(lambda v: f"{v:+.3f}")
    show["Pearson r"] = show["Pearson r"].map(lambda v: f"{v:+.3f}")
    return show


def _top_pair_table(pair_df: pd.DataFrame, top_pair_n: int) -> pd.DataFrame:
    show = pair_df.copy()
    show = show[
        (show["feature_x"] != "predicted_pockets_total")
        & (show["feature_y"] != "predicted_pockets_total")
        & (show["feature_x"] != "ratio_n_holo_to_apo_avg")
        & (show["feature_y"] != "ratio_n_holo_to_apo_avg")
    ].copy()
    show["Feature pair"] = show["feature_x"].map(nice_feature) + " + " + show["feature_y"].map(nice_feature)
    show = show.sort_values("spearman_r_composite", ascending=False).head(top_pair_n).copy()
    show = show[["Feature pair", "spearman_r_composite", "delta_vs_best_single_abs_r"]].rename(
        columns={
            "spearman_r_composite": "Composite rho",
            "delta_vs_best_single_abs_r": "Delta vs best single abs(rho)",
        }
    )
    show["Composite rho"] = show["Composite rho"].map(lambda v: f"{v:+.3f}")
    show["Delta vs best single abs(rho)"] = show["Delta vs best single abs(rho)"].map(lambda v: f"{v:+.3f}")
    return show


def build_report(data: dict[str, pd.DataFrame], top_corr_n: int, top_pair_n: int) -> str:
    merged = data["merged"].copy()
    class_summary = data["class_summary"].copy()
    n_sites = len(merged)
    n_proteins = merged["protein_file"].nunique()

    summary = _summary_table(class_summary)
    corr_topn = _top_corr_table(data["corr_topn"], top_corr_n)
    corr_dca4_sites = _top_corr_table(data["corr_dca4_sites"], top_corr_n)
    corr_tns = _top_corr_table(data["corr_tns"], top_corr_n)
    corr_tnm = _top_corr_table(data["corr_tnm"], top_corr_n)
    pair_topn = _top_pair_table(data["pair_topn"], top_pair_n)
    pair_dca4_sites = _top_pair_table(data["pair_dca4_sites"], top_pair_n)
    pair_tns = _top_pair_table(data["pair_tns"], top_pair_n)
    pair_tnm = _top_pair_table(data["pair_tnm"], top_pair_n)

    lines = [
        "# UBS2 Prediction Analysis Findings (Variable-Rank Metrics)",
        "",
        "Dataset: `eval_predict_st_100_p2rank_intersection_geom_CENTER_explicit`",
        f"{n_sites:,} observed sites across {n_proteins:,} proteins (AlphaFold models).",
        "",
        "## Metric Definitions",
        "",
        "- `Top-1`: observed site found at DCA(4.0) with `dca4rank == 1`.",
        "- `Top-N`: observed site found at DCA(4.0) with `dca4rank >= 1`.",
        "- `DCA4-#sites`: observed site found within the top `#sites` DCA(4.0) ranks, where `#sites` comes from the P2Rank observed-sites case file.",
        "- `TNS`: observed site found within the top `n_unp_pockets` predicted pockets for that protein.",
        "- `TNM`: observed site found within the top `n_unp_pockets_multichain` predicted pockets for that protein.",
        "- `DCA4-#sites`, `TNS`, and `TNM` are all variable-rank metrics, but they use different notions of the expected pocket count.",
        "",
        "## Success Rate Comparison",
        "",
        df_to_md(summary),
        "",
        "## Feature Correlations for Top-N",
        "",
        df_to_md(corr_topn),
        "",
        "## Feature Correlations for DCA4-#sites",
        "",
        df_to_md(corr_dca4_sites),
        "",
        "## Feature Correlations for TNS",
        "",
        df_to_md(corr_tns),
        "",
        "## Feature Correlations for TNM",
        "",
        df_to_md(corr_tnm),
        "",
        "## Feature Pairs for Top-N",
        "",
        df_to_md(pair_topn),
        "",
        "## Feature Pairs for DCA4-#sites",
        "",
        df_to_md(pair_dca4_sites),
        "",
        "## Feature Pairs for TNS",
        "",
        df_to_md(pair_tns),
        "",
        "## Feature Pairs for TNM",
        "",
        df_to_md(pair_tnm),
        "",
        "## Notes",
        "",
        "- Correlation tables are ranked by Spearman rho against the corresponding success metric.",
        "- Pearson r is shown as secondary context only.",
        "- `DCA4-#sites` is based on P2Rank's own observed `#sites` field, while `TNS` and `TNM` come from the ST_100 union-geometry pocket dataset.",
        "- `TNS` is the single-chain true-pocket-count metric, while `TNM` is its multichain counterpart.",
        "- `log2((n_holo_avg+1)/n_apo_avg)` is reconstructed from the AHoJ evidence columns.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(load_inputs(args.analysis_dir), args.top_corr_n, args.top_pair_n)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"[SAVE] {args.output_md}")


if __name__ == "__main__":
    main()
