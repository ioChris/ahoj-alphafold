from __future__ import annotations

import argparse
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
DEFAULT_ANALYSIS_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1"
DEFAULT_REPORT_PATH = (
    cfg.P2RANK_PRED_DIR
    / "ubs_run_4"
    / "UBS2"
    / "eval_predict_st_100_p2rank_intersection_geom_CENTER_explicit"
    / "analyze"
    / "findings_enriched.md"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write enriched markdown findings for the ST_100 P2Rank run.")
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--output_md", type=Path, default=DEFAULT_REPORT_PATH)
    p.add_argument("--top_corr_n", type=int, default=18)
    p.add_argument("--top_pair_n", type=int, default=15)
    return p.parse_args()


def nice_feature(name: str) -> str:
    mapping = {
        "ligand_atoms": "#atoms",
        "site_residues": "#residues",
        "site_radius": "site_radius",
        "rg": "radius of gyration (rg)",
        "site_radius": "site_radius",
        "pocket_p_holo": "pocket_p_holo",
        "pocket_p_apo": "pocket_p_apo",
        "winning_p": "winning_p",
        "pocket_separation_pair": "pocket_separation_pair",
        "pocket_overlap_pair": "pocket_overlap_pair",
        "pocket_overlap_overall": "pocket_overlap_overall",
        "pocket_overlap_mode": "pocket_overlap_mode",
        "pocket_density_pair": "pocket_density_pair",
        "model_pocket_plddt": "model_pocket_plddt",
        "n_holo_avg": "n_holo_avg",
        "n_holo_avg_plus1": "n_holo_avg + 1",
        "n_apo_avg": "n_apo_avg",
        "log2_ratio_holo_to_apo_avg_plus1": "log2((n_holo_avg+1)/n_apo_avg)",
        "best_assigned_score": "best_assigned_score",
        "best_assigned_proba": "best_assigned_proba",
    }
    return mapping.get(name, name)


def load_inputs(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    tables = analysis_dir / "tables"
    return {
        "merged": pd.read_csv(tables / "p2rank_st100_merged_per_pocket.csv", low_memory=False),
        "class_summary": pd.read_csv(tables / "p2rank_class_success_by_cutoff.csv", low_memory=False),
        "corr_tns": pd.read_csv(tables / "p2rank_tns_continuous_correlations.csv", low_memory=False),
        "pair_comp": pd.read_csv(tables / "p2rank_tns_pairwise_composite_correlations.csv", low_memory=False),
    }


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = [str(c) for c in df.columns]
        rows_raw = [[str(row[c]) for c in df.columns] for _, row in df.iterrows()]
        widths = []
        for i, col in enumerate(cols):
            cell_width = max((len(r[i]) for r in rows_raw), default=0)
            widths.append(max(len(col), cell_width))
        header = "| " + " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols)) + " |"
        sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
        rows = [
            "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(cols))) + " |"
            for vals in rows_raw
        ]
        return "\n".join([header, sep] + rows)


def _format_interval_label(interval_str: str) -> str:
    text = str(interval_str).strip()
    if "," not in text:
        return text
    text = text.strip("()[]")
    left, right = [p.strip() for p in text.split(",", 1)]
    try:
        left_v = float(left)
        right_v = float(right)
        if abs(left_v) < 5e-3:
            left_v = 0.0
        if abs(right_v) < 5e-3:
            right_v = 0.0
        return f"{left_v:.2f} - {right_v:.2f}"
    except ValueError:
        return text


def build_report(data: dict[str, pd.DataFrame], top_corr_n: int, top_pair_n: int) -> str:
    merged = data["merged"].copy()
    class_summary = data["class_summary"].copy()
    corr = data["corr_tns"].copy()
    pair_comp = data["pair_comp"].copy()

    n_sites = len(merged)
    n_proteins = merged["protein_file"].nunique()

    topn = class_summary.loc[class_summary["success_metric"] == "success_tns"].copy()
    top1 = class_summary.loc[class_summary["success_metric"] == "success_top1"].copy()
    overall_topn = float(topn.loc[topn["pocket_class"] == "all", "success_percent"].iloc[0])
    overall_top1 = float(top1.loc[top1["pocket_class"] == "all", "success_percent"].iloc[0])

    p_holo_vals = pd.to_numeric(merged["pocket_p_holo"], errors="coerce")
    merged["p_holo_quintile"] = pd.qcut(p_holo_vals, q=5, duplicates="drop")
    p_holo_q = (
        merged.dropna(subset=["p_holo_quintile"])
        .groupby("p_holo_quintile", observed=False)
        .agg(
            n=("unit_uid", "size"),
            found_pct=("success_tns", lambda s: pd.to_numeric(s, errors="coerce").mean() * 100.0),
            top1_pct=("success_top1", lambda s: pd.to_numeric(s, errors="coerce").mean() * 100.0),
        )
        .reset_index()
    )
    p_holo_q["p_holo quintile"] = p_holo_q["p_holo_quintile"].astype(str).map(_format_interval_label)
    p_holo_q = p_holo_q[["p_holo quintile", "n", "found_pct", "top1_pct"]].rename(
        columns={"found_pct": "Found %", "top1_pct": "Top-1 %"}
    )
    p_holo_q["n"] = p_holo_q["n"].map(lambda v: f"{int(v):,}")
    p_holo_q["Found %"] = p_holo_q["Found %"].map(lambda v: f"{v:.1f}")
    p_holo_q["Top-1 %"] = p_holo_q["Top-1 %"].map(lambda v: f"{v:.1f}")

    by_class = (
        merged.groupby("pocket_class", dropna=False)
        .agg(
            n=("unit_uid", "size"),
            found_pct=("success_tns", lambda s: pd.to_numeric(s, errors="coerce").mean() * 100.0),
            top1_pct=("success_top1", lambda s: pd.to_numeric(s, errors="coerce").mean() * 100.0),
            mean_p_holo=("pocket_p_holo", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
    )
    by_class = by_class[by_class["pocket_class"].isin(["holo", "apo", "tie"])].copy()
    by_class["n"] = by_class["n"].map(lambda v: f"{int(v):,}")
    by_class["found_pct"] = by_class["found_pct"].map(lambda v: f"{v:.1f}")
    by_class["top1_pct"] = by_class["top1_pct"].map(lambda v: f"{v:.1f}")
    by_class["mean_p_holo"] = by_class["mean_p_holo"].map(lambda v: f"{v:.3f}")
    by_class = by_class.rename(
        columns={
            "pocket_class": "Pocket class",
            "found_pct": "Found %",
            "top1_pct": "Top-1 %",
            "mean_p_holo": "Mean p_holo",
        }
    )

    corr = corr.copy()
    corr = corr[pd.to_numeric(corr["spearman_r_success"], errors="coerce").notna()].copy()
    corr["abs_r"] = corr["spearman_r_success"].abs()
    priority_features = {
        "ligand_atoms",
        "site_residues",
        "site_radius",
        "rg",
        "pocket_p_holo",
        "pocket_p_apo",
        "log2_ratio_holo_to_apo_avg_plus1",
        "n_holo_avg",
        "n_holo_avg_plus1",
        "n_apo_avg",
        "pocket_separation_pair",
        "pocket_overlap_overall",
        "pocket_overlap_pair",
        "pocket_overlap_mode",
        "pocket_density_pair",
        "winning_p",
        "model_pocket_plddt",
    }
    corr_show = corr[corr["feature"].isin(priority_features)].sort_values("abs_r", ascending=False).head(top_corr_n).copy()
    corr_show = corr_show[
        ["feature", "spearman_r_success", "pearson_r_success", "spearman_r_distance"]
    ].rename(
        columns={
            "feature": "Feature",
            "spearman_r_success": "Spearman rho vs Found",
            "pearson_r_success": "Pearson r vs Found",
            "spearman_r_distance": "rho vs closestPocketDist",
        }
    )
    corr_show["Feature"] = corr_show["Feature"].map(nice_feature)
    corr_show["Spearman rho vs Found"] = corr_show["Spearman rho vs Found"].map(lambda v: f"{v:+.3f}")
    corr_show["Pearson r vs Found"] = corr_show["Pearson r vs Found"].map(lambda v: f"{v:+.3f}")
    corr_show["rho vs closestPocketDist"] = corr_show["rho vs closestPocketDist"].map(lambda v: f"{v:+.3f}")

    pair_show = pair_comp.copy()
    pair_show = pair_show[
        (pair_show["feature_x"] != "predicted_pockets_total")
        & (pair_show["feature_y"] != "predicted_pockets_total")
        & (pair_show["feature_x"] != "ratio_n_holo_to_apo_avg")
        & (pair_show["feature_y"] != "ratio_n_holo_to_apo_avg")
    ].copy()
    pair_show["pair"] = pair_show["feature_x"].map(nice_feature) + " + " + pair_show["feature_y"].map(nice_feature)
    pair_show = pair_show.sort_values("spearman_r_composite", ascending=False).head(top_pair_n)
    pair_show = pair_show[
        ["pair", "spearman_r_composite", "delta_vs_best_single_abs_r"]
    ].rename(
        columns={
            "pair": "Feature pair",
            "spearman_r_composite": "Composite rho",
            "delta_vs_best_single_abs_r": "Delta vs best single abs(rho)",
        }
    )
    pair_show["Composite rho"] = pair_show["Composite rho"].map(lambda v: f"{v:+.3f}")
    pair_show["Delta vs best single abs(rho)"] = pair_show["Delta vs best single abs(rho)"].map(lambda v: f"{v:+.3f}")

    strongest = corr[corr["feature"].isin(priority_features)].sort_values("spearman_r_success", ascending=False, key=lambda s: s.abs()).head(8)
    strongest_names = ", ".join(nice_feature(v) for v in strongest["feature"].tolist())

    lines = [
        "# UBS2 Prediction Analysis Findings (Enriched)",
        "",
        "Dataset: `eval_predict_st_100_p2rank_intersection_geom_CENTER_explicit`",
        f"{n_sites:,} observed sites across {n_proteins:,} proteins (AlphaFold models).",
        "",
        "## Overall Performance",
        "",
        f"- **{overall_topn:.1f}% of sites found** under the **TNS DCA4** criterion, **{overall_top1:.1f}% as top-1 prediction**",
        "- This report uses the ST_100 true-single-chain-pocket count (`TNS`) as the DCA(4.0) rank tolerance.",
        "",
        "## pocket_p_holo is a Strong Predictor",
        "",
        "- Clear monotonic relationship with success rate:",
        "",
        df_to_md(p_holo_q),
        "",
        "## Holo vs Apo is the Dominant Factor",
        "",
        df_to_md(by_class),
        "",
        "## Feature Correlations with Success (TNS DCA4)",
        "",
        f"Most informative features in the current run include: {strongest_names}.",
        "",
        df_to_md(corr_show),
        "",
        "Notes:",
        "- Rows are ranked only by Spearman `rho vs Found`, so this section stays comparable to the earlier `findings.md` style while also showing Pearson `r`.",
        "- `rho vs closestPocketDist` is included only as secondary context and does not affect ranking.",
        "- `rg` comes from the ST_100 union-geometry pocket dataset and behaves as another strong pocket-size / extent descriptor.",
        "- `log2((n_holo_avg+1)/n_apo_avg)` was reconstructed from the AHoJ evidence columns.",
        "- `n_holo_avg + 1` is shown explicitly for readability, but it has the same rank correlation as `n_holo_avg` because it is just a constant shift.",
        "",
        "## Feature Combinations (pairs of features)",
        "",
        "Pairs are ranked by the Spearman rho of a rank-based composite score against found/not-found status.",
        "",
        df_to_md(pair_show),
        "",
        "## Key Takeaways",
        "",
        "1. Site size remains the dominant axis of predictability, with `#atoms`, `#residues`, and now also `rg` among the strongest correlates.",
        "2. Holo character is the next major axis, captured by `pocket_p_holo` and the holo/apo evidence-ratio terms.",
        "3. Separation and overlap still matter, but they are weaker than the size and holo-likeness signals in the raw uncontrolled ranking.",
        "4. The best feature pairs usually combine holo-enrichment with pocket-context variables such as predicted pocket count or separation/overlap.",
        "",
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
