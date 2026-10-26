from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from confidence_labeling import probability_level, support_level

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
DEFAULT_P2RANK_EVAL_DIR = (
    cfg.P2RANK_PRED_DIR
    / "ubs_run_4"
    / "UBS2"
    / "eval_predict_st_100_p2rank_intersection_geom_CENTER_explicit"
)
DEFAULT_UNIFIED_TABLE = WORK_DIR / "unified_output" / "run4" / "analysis_unified_levels.csv"
DEFAULT_SEPARATION_TABLE = (
    WORK_DIR / "alphafill_unified" / "run1" / "analysis_v1" / "per_ubs_summary_any_ge_0p90.csv"
)
DEFAULT_UNION_GEOM_TABLE = (
    WORK_DIR / "afdb_dataset" / "st100" / "p2rank_datasets" / "st_100_union_geom" / "st_100_p2rank_union_geom.csv"
)
DEFAULT_OUTPUT_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1"

CUTOFFS = [4.0, 5.0, 6.0, 8.0, 10.0]
SUCCESS_COLS = ["success_top1", "success_topn", "success_dca4_sites", "success_tns", "success_tnm"] + [f"success_dca_{int(c)}A" for c in CUTOFFS]
FULL_LEVEL_ORDER = ["very_low", "low", "moderate", "high", "very_high"]
PRIMARY_SUCCESS_COL = "success_tns"
PRIMARY_TAG = "tns"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze P2Rank ST_100 UBS predictions against AHoJ pocket metrics.")
    p.add_argument("--p2rank_eval_dir", type=Path, default=DEFAULT_P2RANK_EVAL_DIR)
    p.add_argument("--unified_table", type=Path, default=DEFAULT_UNIFIED_TABLE)
    p.add_argument("--separation_table", type=Path, default=DEFAULT_SEPARATION_TABLE)
    p.add_argument("--union_geom_table", type=Path, default=DEFAULT_UNION_GEOM_TABLE)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--threshold", default="ST_100")
    p.add_argument("--level", default="ubs")
    p.add_argument("--holo_plus_one", type=float, default=1.0)
    return p.parse_args()


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace(" ", "_") for c in out.columns]
    return out


def load_p2rank_cases(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases_dir = eval_dir / "cases"
    observed = _clean_columns(pd.read_csv(cases_dir / "observed_sites.csv"))
    predicted = _clean_columns(pd.read_csv(cases_dir / "predicted_pockets.csv"))
    proteins = _clean_columns(pd.read_csv(cases_dir / "proteins.csv"))
    summary = _clean_columns(pd.read_csv(eval_dir / "summary.csv"))
    success_rates = pd.read_csv(eval_dir / "success_rates.csv", header=0)
    return observed, predicted, proteins, summary, success_rates


def load_ahoj_metrics(
    unified_table: Path,
    separation_table: Path,
    union_geom_table: Path,
    *,
    threshold: str,
    level: str,
    holo_plus_one: float,
) -> pd.DataFrame:
    usecols = [
        "threshold",
        "level",
        "unit_uid",
        "rep_job",
        "uniprots_str",
        "members",
        "members_total",
        "site_size_unit",
        "site_size_intersection",
        "model_pocket_plddt",
        "pocket_class",
        "pocket_p_apo",
        "pocket_p_holo",
        "pocket_score",
        "pocket_density_pair",
        "pocket_overlap_pair",
        "pocket_overlap_overall",
        "pocket_overlap_mode",
        "ratio_n_holo_to_apo_avg",
        "n_apo_avg",
        "n_holo_avg",
        "evidence_n_apo",
        "evidence_n_holo",
        "evidence_basis",
    ]
    unified = pd.read_csv(unified_table, usecols=usecols, low_memory=False)
    unified = unified[(unified["threshold"] == threshold) & (unified["level"] == level)].copy()

    sep = pd.read_csv(
        separation_table,
        usecols=["threshold", "unit_uid", "pocket_separation_pair"],
        low_memory=False,
    )
    sep = sep[sep["threshold"] == threshold].copy()
    sep = sep.drop_duplicates(subset=["unit_uid"])

    union_geom = pd.read_csv(
        union_geom_table,
        usecols=[
            "threshold",
            "site_uid",
            "rg",
            "n_atoms",
            "n_residues_found",
            "n_unp_pockets",
            "n_unp_pockets_multichain",
        ],
        low_memory=False,
    )
    union_geom = union_geom[union_geom["threshold"] == threshold].copy()
    union_geom = union_geom.rename(
        columns={
            "site_uid": "unit_uid",
            "n_atoms": "site_union_atoms",
            "n_residues_found": "site_union_residues",
        }
    ).drop_duplicates(subset=["unit_uid"])

    merged = unified.merge(sep[["unit_uid", "pocket_separation_pair"]], on="unit_uid", how="left")
    merged = merged.merge(
        union_geom[
            [
                "unit_uid",
                "rg",
                "site_union_atoms",
                "site_union_residues",
                "n_unp_pockets",
                "n_unp_pockets_multichain",
            ]
        ],
        on="unit_uid",
        how="left",
    )
    merged["winning_p"] = merged[["pocket_p_apo", "pocket_p_holo"]].max(axis=1)
    merged["n_holo_avg_plus1"] = merged["n_holo_avg"] + float(holo_plus_one)
    merged["prob_level_db"] = merged["winning_p"].map(lambda v: probability_level(v, merge=False))
    merged["prob_level"] = merged["winning_p"].map(lambda v: probability_level(v, merge=True))
    merged["density_level_db"] = merged["pocket_density_pair"].map(lambda v: support_level(v, merge=False))
    merged["density_level"] = merged["pocket_density_pair"].map(lambda v: support_level(v, merge=True))
    merged["separation_level_db"] = merged["pocket_separation_pair"].map(lambda v: support_level(v, merge=False))
    merged["separation_level"] = merged["pocket_separation_pair"].map(lambda v: support_level(v, merge=True))
    merged["ratio_holo_to_apo_avg_plus1"] = (merged["n_holo_avg"] + float(holo_plus_one)) / merged["n_apo_avg"].replace(0, np.nan)
    merged["log2_ratio_holo_to_apo_avg_plus1"] = np.log2(merged["ratio_holo_to_apo_avg_plus1"])
    return merged.drop_duplicates(subset=["unit_uid"]).copy()


def aggregate_predicted_pockets(predicted: pd.DataFrame) -> pd.DataFrame:
    pred = predicted.copy()
    pred["ligand"] = pred["ligand"].fillna("").astype(str).str.strip()
    pred = pred[pred["ligand"] != ""].copy()
    pred["rank"] = pd.to_numeric(pred["rank"], errors="coerce")
    pred["score"] = pd.to_numeric(pred["score"], errors="coerce")
    pred["probaTP"] = pd.to_numeric(pred["probaTP"], errors="coerce")
    pred["zScoreTP"] = pd.to_numeric(pred["zScoreTP"], errors="coerce")
    pred = pred.sort_values(["ligand", "rank", "score"], ascending=[True, True, False])

    rows = []
    for ligand, grp in pred.groupby("ligand", sort=False):
        best = grp.iloc[0]
        rows.append(
            {
                "unit_uid": ligand,
                "assigned_pockets_n": int(len(grp)),
                "best_assigned_rank": pd.to_numeric(best["rank"], errors="coerce"),
                "best_assigned_score": pd.to_numeric(best["score"], errors="coerce"),
                "best_assigned_proba": pd.to_numeric(best["probaTP"], errors="coerce"),
                "best_assigned_zscore": pd.to_numeric(best["zScoreTP"], errors="coerce"),
                "assigned_pockets_str": " ".join(grp["pocket"].astype(str).tolist()[:8]),
            }
        )
    return pd.DataFrame(rows)


def build_merged_table(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    proteins: pd.DataFrame,
    ahoj: pd.DataFrame,
) -> pd.DataFrame:
    obs = observed.copy()
    obs["unit_uid"] = obs["site"].astype(str).str.strip()
    obs["protein_file"] = obs["file"].astype(str).str.strip()
    obs["dca4rank"] = pd.to_numeric(obs["dca4rank"], errors="coerce")
    obs["closest_pocket_dist"] = pd.to_numeric(obs["closestPocketDist"], errors="coerce")
    obs["protein_dist"] = pd.to_numeric(obs["proteinDist"], errors="coerce")
    obs["center_to_protein_dist"] = pd.to_numeric(obs["centerToProteinDist"], errors="coerce")
    obs["sas_dist"] = pd.to_numeric(obs["sasDist"], errors="coerce")
    obs["contact_protein_atoms"] = pd.to_numeric(obs["#contactProteinAtoms"], errors="coerce")
    obs["ligand_atoms"] = pd.to_numeric(obs["#atoms"], errors="coerce")
    obs["ligands_per_protein"] = pd.to_numeric(obs["#sites"], errors="coerce")
    obs["site_residues"] = pd.to_numeric(obs["#residues"], errors="coerce")
    obs["site_radius"] = pd.to_numeric(obs["site_radius"], errors="coerce")
    obs["success_top1"] = obs["dca4rank"] == 1
    obs["success_topn"] = obs["dca4rank"] > 0
    valid_sites_n = obs["ligands_per_protein"].notna() & (obs["ligands_per_protein"] >= 1)
    obs["success_dca4_sites"] = np.where(
        valid_sites_n,
        (obs["dca4rank"] > 0) & (obs["dca4rank"] <= obs["ligands_per_protein"]),
        np.nan,
    )
    for cutoff in CUTOFFS:
        obs[f"success_dca_{int(cutoff)}A"] = obs["closest_pocket_dist"] <= cutoff

    prot = proteins.copy()
    prot["protein_file"] = prot["name"].astype(str).str.strip()
    prot["predicted_pockets_total"] = pd.to_numeric(prot["#pockets"], errors="coerce")
    prot["ligands_total"] = pd.to_numeric(prot["#ligands"], errors="coerce")
    prot = prot[["protein_file", "predicted_pockets_total", "ligands_total", "#proteinAtoms", "#atoms"]].rename(
        columns={"#proteinAtoms": "protein_atoms", "#atoms": "structure_atoms"}
    )

    pred_agg = aggregate_predicted_pockets(predicted)
    drop_obs_metric_cols = [
        "pocket_class",
        "pocket_density_pair",
        "pocket_overlap_pair",
        "pocket_separation_pair",
        "pocket_p_apo",
        "pocket_p_holo",
        "pocket_score",
        "model_pocket_plddt",
        "n_apo_avg",
        "n_holo_avg",
    ]
    obs = obs.drop(columns=[c for c in drop_obs_metric_cols if c in obs.columns])
    merged = obs.merge(pred_agg, on="unit_uid", how="left")
    merged = merged.merge(prot, on="protein_file", how="left")
    merged = merged.merge(ahoj, on="unit_uid", how="left", validate="many_to_one")
    merged["n_unp_pockets"] = pd.to_numeric(merged["n_unp_pockets"], errors="coerce")
    merged["n_unp_pockets_multichain"] = pd.to_numeric(merged["n_unp_pockets_multichain"], errors="coerce")
    valid_tns = merged["n_unp_pockets"].notna() & (merged["n_unp_pockets"] >= 1)
    valid_tnm = merged["n_unp_pockets_multichain"].notna() & (merged["n_unp_pockets_multichain"] >= 1)
    merged["success_tns"] = np.where(
        valid_tns,
        (merged["dca4rank"] > 0) & (merged["dca4rank"] <= merged["n_unp_pockets"]),
        np.nan,
    )
    merged["success_tnm"] = np.where(
        valid_tnm,
        (merged["dca4rank"] > 0) & (merged["dca4rank"] <= merged["n_unp_pockets_multichain"]),
        np.nan,
    )
    merged["pocket_class"] = merged["pocket_class"].fillna("unknown").astype(str).str.lower()
    return merged


def tidy_success_rates(success_rates: pd.DataFrame) -> pd.DataFrame:
    cols = list(success_rates.columns)
    tolerance_cols = cols[1:]
    tidy = success_rates.rename(columns={cols[0]: "metric"}).melt(
        id_vars="metric",
        value_vars=tolerance_cols,
        var_name="tolerance",
        value_name="success_percent",
    )
    tidy["tolerance"] = tidy["tolerance"].astype(str).str.strip("[]")
    tidy["success_percent"] = pd.to_numeric(tidy["success_percent"], errors="coerce")
    return tidy


def class_summary_by_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for success_col in SUCCESS_COLS:
        for pocket_class, grp in df.groupby("pocket_class", dropna=False):
            if pocket_class not in {"apo", "holo"}:
                continue
            success = pd.to_numeric(grp[success_col], errors="coerce")
            rows.append(
                {
                    "success_metric": success_col,
                    "pocket_class": pocket_class,
                    "n": int(success.notna().sum()),
                    "success_rate": float(success.mean()),
                    "success_percent": float(success.mean() * 100.0),
                }
            )
        rows.append(
            {
                "success_metric": success_col,
                "pocket_class": "all",
                "n": int(df[success_col].notna().sum()),
                "success_rate": float(pd.to_numeric(df[success_col], errors="coerce").mean()),
                "success_percent": float(pd.to_numeric(df[success_col], errors="coerce").mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def bin_summary(df: pd.DataFrame, success_col: str = "success_dca_4A") -> pd.DataFrame:
    rows = []
    for bin_type in ["prob_level_db", "separation_level_db", "density_level_db"]:
        for pocket_class, class_df in df.groupby("pocket_class", dropna=False):
            if pocket_class not in {"apo", "holo"}:
                continue
            for bin_label, grp in class_df.groupby(bin_type, dropna=False):
                success = pd.to_numeric(grp[success_col], errors="coerce")
                rows.append(
                    {
                        "bin_type": bin_type,
                        "bin_label": str(bin_label),
                        "pocket_class": pocket_class,
                        "success_metric": success_col,
                        "n": int(success.notna().sum()),
                        "success_rate": float(success.mean()),
                        "success_percent": float(success.mean() * 100.0),
                        "mean_closest_pocket_dist": float(pd.to_numeric(grp["closest_pocket_dist"], errors="coerce").mean()),
                    }
                )
    return pd.DataFrame(rows)


def quantile_summary(
    df: pd.DataFrame,
    success_col: str = "success_dca_4A",
    features: list[str] | None = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    if features is None:
        features = [
            "winning_p",
            "pocket_density_pair",
            "pocket_separation_pair",
            "pocket_overlap_pair",
            "log2_ratio_holo_to_apo_avg_plus1",
        ]
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        vals = pd.to_numeric(df[feature], errors="coerce")
        valid = vals.notna()
        if valid.sum() < n_bins * 10:
            continue
        try:
            bins = pd.qcut(vals[valid], q=n_bins, labels=[f"q{i}" for i in range(1, n_bins + 1)], duplicates="drop")
        except ValueError:
            continue
        tmp = df.loc[valid, ["pocket_class", success_col]].copy()
        tmp["feature"] = feature
        tmp["bin"] = bins.astype(str)
        tmp["feature_value"] = vals[valid]
        for pocket_class, class_df in tmp.groupby("pocket_class", dropna=False):
            if pocket_class not in {"apo", "holo"}:
                continue
            for bin_label, grp in class_df.groupby("bin", dropna=False):
                success = pd.to_numeric(grp[success_col], errors="coerce")
                rows.append(
                    {
                        "feature": feature,
                        "bin": str(bin_label),
                        "pocket_class": pocket_class,
                        "success_metric": success_col,
                        "n": int(success.notna().sum()),
                        "success_rate": float(success.mean()),
                        "feature_mean": float(pd.to_numeric(grp["feature_value"], errors="coerce").mean()),
                    }
                )
    return pd.DataFrame(rows)


def continuous_correlations(
    df: pd.DataFrame,
    *,
    success_col: str = "success_dca_4A",
    features: list[str] | None = None,
) -> pd.DataFrame:
    if features is None:
        features = [
            "winning_p",
            "pocket_p_apo",
            "pocket_p_holo",
            "pocket_density_pair",
            "pocket_separation_pair",
            "pocket_overlap_pair",
            "pocket_overlap_overall",
            "pocket_overlap_mode",
            "ratio_n_holo_to_apo_avg",
            "log2_ratio_holo_to_apo_avg_plus1",
            "n_apo_avg",
            "n_holo_avg",
            "n_holo_avg_plus1",
            "model_pocket_plddt",
            "ligand_atoms",
            "rg",
            "site_residues",
            "site_radius",
            "predicted_pockets_total",
            "best_assigned_score",
            "best_assigned_proba",
        ]
    rows = []
    success = pd.to_numeric(df[success_col], errors="coerce").astype(float)
    closest = pd.to_numeric(df["closest_pocket_dist"], errors="coerce")
    for feature in features:
        if feature not in df.columns:
            continue
        vals = pd.to_numeric(df[feature], errors="coerce")
        mask = vals.notna() & success.notna()
        if mask.sum() < 25:
            continue
        try:
            pear_r, pear_p = pearsonr(vals[mask], success[mask])
        except Exception:
            pear_r, pear_p = np.nan, np.nan
        try:
            spear_r, spear_p = spearmanr(vals[mask], success[mask], nan_policy="omit")
        except Exception:
            spear_r, spear_p = np.nan, np.nan
        dist_mask = vals.notna() & closest.notna()
        try:
            dist_spear_r, dist_spear_p = spearmanr(vals[dist_mask], closest[dist_mask], nan_policy="omit")
        except Exception:
            dist_spear_r, dist_spear_p = np.nan, np.nan
        rows.append(
            {
                "feature": feature,
                "success_metric": success_col,
                "n_success": int(mask.sum()),
                "pearson_r_success": float(pear_r) if pd.notna(pear_r) else np.nan,
                "pearson_p_success": float(pear_p) if pd.notna(pear_p) else np.nan,
                "spearman_r_success": float(spear_r) if pd.notna(spear_r) else np.nan,
                "spearman_p_success": float(spear_p) if pd.notna(spear_p) else np.nan,
                "n_distance": int(dist_mask.sum()),
                "spearman_r_distance": float(dist_spear_r) if pd.notna(dist_spear_r) else np.nan,
                "spearman_p_distance": float(dist_spear_p) if pd.notna(dist_spear_p) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_r_success", ascending=False, key=lambda s: s.abs())


def combined_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pocket_class in ["apo", "holo"]:
        class_df = df[df["pocket_class"] == pocket_class].copy()
        for prob_level in FULL_LEVEL_ORDER:
            for sep_level in FULL_LEVEL_ORDER:
                grp = class_df[
                    (class_df["prob_level_db"] == prob_level)
                    & (class_df["separation_level_db"] == sep_level)
                ].copy()
                n = len(grp)
                if n == 0:
                    continue
                row = {
                    "pocket_class": pocket_class,
                    "prob_level_db": prob_level,
                    "separation_level_db": sep_level,
                    "n": int(n),
                    "closest_pocket_dist_mean": float(pd.to_numeric(grp["closest_pocket_dist"], errors="coerce").mean()),
                    "closest_pocket_dist_median": float(pd.to_numeric(grp["closest_pocket_dist"], errors="coerce").median()),
                    "winning_p_mean": float(pd.to_numeric(grp["winning_p"], errors="coerce").mean()),
                    "pocket_separation_pair_mean": float(pd.to_numeric(grp["pocket_separation_pair"], errors="coerce").mean()),
                    "pocket_density_pair_mean": float(pd.to_numeric(grp["pocket_density_pair"], errors="coerce").mean()),
                    "log2_ratio_holo_to_apo_avg_plus1_mean": float(
                        pd.to_numeric(grp["log2_ratio_holo_to_apo_avg_plus1"], errors="coerce").mean()
                    ),
                }
                for success_col in SUCCESS_COLS:
                    row[f"{success_col}_rate"] = float(pd.to_numeric(grp[success_col], errors="coerce").mean())
                    row[f"{success_col}_percent"] = float(pd.to_numeric(grp[success_col], errors="coerce").mean() * 100.0)
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["pocket_class", "prob_level_db", "separation_level_db"],
        key=lambda s: s.map({v: i for i, v in enumerate(FULL_LEVEL_ORDER)}) if s.name in {"prob_level_db", "separation_level_db"} else s,
    )


def pairwise_combo_rankings(
    df: pd.DataFrame,
    *,
    success_col: str,
    features: list[str],
    n_bins: int = 5,
    min_joint_n: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = {}
    success = pd.to_numeric(df[success_col], errors="coerce")
    for feature in features:
        if feature not in df.columns:
            continue
        vals = pd.to_numeric(df[feature], errors="coerce")
        mask = vals.notna() & success.notna()
        if mask.sum() < n_bins * 20:
            continue
        try:
            r, _ = spearmanr(vals[mask], success[mask], nan_policy="omit")
        except Exception:
            r = np.nan
        if pd.notna(r):
            corr[feature] = float(r)

    ranking_rows = []
    grid_rows = []
    for feat_x, feat_y in combinations([f for f in features if f in corr], 2):
        x = pd.to_numeric(df[feat_x], errors="coerce")
        y = pd.to_numeric(df[feat_y], errors="coerce")
        mask = x.notna() & y.notna() & success.notna()
        if mask.sum() < n_bins * n_bins * 10:
            continue
        try:
            xbins = pd.qcut(x[mask], q=n_bins, labels=[f"Q{i}" for i in range(1, n_bins + 1)], duplicates="drop")
            ybins = pd.qcut(y[mask], q=n_bins, labels=[f"Q{i}" for i in range(1, n_bins + 1)], duplicates="drop")
        except ValueError:
            continue
        tmp = pd.DataFrame(
            {
                "x_bin": xbins.astype(str),
                "y_bin": ybins.astype(str),
                "success": success[mask].astype(float).to_numpy(),
            }
        )
        agg = tmp.groupby(["x_bin", "y_bin"], observed=False).agg(n=("success", "size"), success_rate=("success", "mean")).reset_index()
        agg["feature_x"] = feat_x
        agg["feature_y"] = feat_y
        grid_rows.extend(agg.to_dict(orient="records"))

        fav_x = f"Q{n_bins}" if corr[feat_x] >= 0 else "Q1"
        unfav_x = "Q1" if corr[feat_x] >= 0 else f"Q{n_bins}"
        fav_y = f"Q{n_bins}" if corr[feat_y] >= 0 else "Q1"
        unfav_y = "Q1" if corr[feat_y] >= 0 else f"Q{n_bins}"

        joint_fav = agg[(agg["x_bin"] == fav_x) & (agg["y_bin"] == fav_y)]
        joint_unfav = agg[(agg["x_bin"] == unfav_x) & (agg["y_bin"] == unfav_y)]
        if joint_fav.empty or joint_unfav.empty:
            continue
        n_fav = int(joint_fav["n"].iloc[0])
        n_unfav = int(joint_unfav["n"].iloc[0])
        if n_fav < min_joint_n or n_unfav < min_joint_n:
            continue
        s_fav = float(joint_fav["success_rate"].iloc[0] * 100.0)
        s_unfav = float(joint_unfav["success_rate"].iloc[0] * 100.0)
        ranking_rows.append(
            {
                "success_metric": success_col,
                "feature_x": feat_x,
                "feature_y": feat_y,
                "spearman_x": corr[feat_x],
                "spearman_y": corr[feat_y],
                "favorable_x_bin": fav_x,
                "favorable_y_bin": fav_y,
                "unfavorable_x_bin": unfav_x,
                "unfavorable_y_bin": unfav_y,
                "n_joint_favorable": n_fav,
                "n_joint_unfavorable": n_unfav,
                "joint_favorable_success_percent": s_fav,
                "joint_unfavorable_success_percent": s_unfav,
                "delta_success_percent": s_fav - s_unfav,
            }
        )
    ranking_df = pd.DataFrame(ranking_rows).sort_values("delta_success_percent", ascending=False)
    grid_df = pd.DataFrame(grid_rows)
    return ranking_df, grid_df


def pairwise_composite_correlations(
    df: pd.DataFrame,
    *,
    success_col: str,
    features: list[str],
    min_n: int = 200,
) -> pd.DataFrame:
    success = pd.to_numeric(df[success_col], errors="coerce").astype(float)
    single_corr = {}
    for feature in features:
        if feature not in df.columns:
            continue
        vals = pd.to_numeric(df[feature], errors="coerce")
        mask = vals.notna() & success.notna()
        if mask.sum() < min_n:
            continue
        try:
            r, p = spearmanr(vals[mask], success[mask], nan_policy="omit")
        except Exception:
            r, p = np.nan, np.nan
        if pd.notna(r):
            single_corr[feature] = (float(r), float(p) if pd.notna(p) else np.nan)

    rows = []
    for feat_x, feat_y in combinations([f for f in features if f in single_corr], 2):
        x = pd.to_numeric(df[feat_x], errors="coerce")
        y = pd.to_numeric(df[feat_y], errors="coerce")
        mask = x.notna() & y.notna() & success.notna()
        n = int(mask.sum())
        if n < min_n:
            continue
        xr = x[mask].rank(pct=True)
        yr = y[mask].rank(pct=True)
        x_oriented = xr if single_corr[feat_x][0] >= 0 else (1.0 - xr)
        y_oriented = yr if single_corr[feat_y][0] >= 0 else (1.0 - yr)
        composite = 0.5 * (x_oriented + y_oriented)
        try:
            comp_r, comp_p = spearmanr(composite, success[mask], nan_policy="omit")
        except Exception:
            comp_r, comp_p = np.nan, np.nan
        if pd.isna(comp_r):
            continue
        best_single = max(abs(single_corr[feat_x][0]), abs(single_corr[feat_y][0]))
        rows.append(
            {
                "success_metric": success_col,
                "feature_x": feat_x,
                "feature_y": feat_y,
                "n": n,
                "spearman_r_composite": float(comp_r),
                "spearman_p_composite": float(comp_p) if pd.notna(comp_p) else np.nan,
                "spearman_r_x": float(single_corr[feat_x][0]),
                "spearman_r_y": float(single_corr[feat_y][0]),
                "best_single_abs_r": float(best_single),
                "delta_vs_best_single_abs_r": float(abs(comp_r) - best_single),
            }
        )
    return pd.DataFrame(rows).sort_values(["spearman_r_composite", "delta_vs_best_single_abs_r"], ascending=False, key=lambda s: s.abs() if s.name == "spearman_r_composite" else s)


def write_manifest(
    output_dir: Path,
    *,
    eval_dir: Path,
    union_geom_table: Path,
    threshold: str,
    level: str,
    holo_plus_one: float,
    merged: pd.DataFrame,
    success_rates_tidy: pd.DataFrame,
) -> None:
    manifest = {
        "p2rank_eval_dir": str(eval_dir),
        "union_geom_table": str(union_geom_table),
        "threshold": threshold,
        "level": level,
        "holo_plus_one": float(holo_plus_one),
        "n_rows_merged": int(len(merged)),
        "n_unique_unit_uid": int(merged["unit_uid"].nunique()),
        "overall_success_top1_percent": float(merged["success_top1"].mean() * 100.0),
        "overall_success_topn_percent": float(merged["success_topn"].mean() * 100.0),
        "overall_success_dca4_sites_percent": float(pd.to_numeric(merged["success_dca4_sites"], errors="coerce").mean() * 100.0),
        "overall_success_tns_percent": float(pd.to_numeric(merged["success_tns"], errors="coerce").mean() * 100.0),
        "overall_success_tnm_percent": float(pd.to_numeric(merged["success_tnm"], errors="coerce").mean() * 100.0),
        "overall_success_dca4_percent": float(merged["success_dca_4A"].mean() * 100.0),
        "reported_dca4_tol0_percent": float(
            success_rates_tidy.loc[
                (success_rates_tidy["metric"] == "DCA(4.0)") & (success_rates_tidy["tolerance"] == "0"),
                "success_percent",
            ].iloc[0]
        ),
    }
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    with (output_dir / "tables" / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def main() -> None:
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    observed, predicted, proteins, summary, success_rates = load_p2rank_cases(args.p2rank_eval_dir)
    ahoj = load_ahoj_metrics(
        args.unified_table,
        args.separation_table,
        args.union_geom_table,
        threshold=args.threshold,
        level=args.level,
        holo_plus_one=args.holo_plus_one,
    )
    merged = build_merged_table(observed, predicted, proteins, ahoj)
    success_rates_tidy = tidy_success_rates(success_rates)
    class_summary = class_summary_by_cutoff(merged)
    dca4_bin_summary = bin_summary(merged, success_col="success_dca_4A")
    dca4_quantile_summary = quantile_summary(merged, success_col="success_dca_4A")
    correlations = continuous_correlations(merged, success_col="success_dca_4A")
    combined_summary = combined_group_summary(merged)
    primary_bin_summary = bin_summary(merged, success_col=PRIMARY_SUCCESS_COL)
    primary_quantile_summary = quantile_summary(merged, success_col=PRIMARY_SUCCESS_COL)
    primary_correlations = continuous_correlations(merged, success_col=PRIMARY_SUCCESS_COL)
    dca4_sites_correlations = continuous_correlations(merged, success_col="success_dca4_sites")
    tns_correlations = continuous_correlations(merged, success_col="success_tns")
    tnm_correlations = continuous_correlations(merged, success_col="success_tnm")
    pair_features = [
        "pocket_p_holo",
        "pocket_p_apo",
        "winning_p",
        "pocket_separation_pair",
        "pocket_overlap_pair",
        "pocket_density_pair",
        "log2_ratio_holo_to_apo_avg_plus1",
        "ligand_atoms",
        "n_holo_avg",
    ]
    primary_pair_rankings, primary_pair_grids = pairwise_combo_rankings(
        merged,
        success_col=PRIMARY_SUCCESS_COL,
        features=pair_features,
        n_bins=5,
        min_joint_n=100,
    )
    primary_pair_corr_features = [
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
    primary_pair_composite_corr = pairwise_composite_correlations(
        merged,
        success_col=PRIMARY_SUCCESS_COL,
        features=primary_pair_corr_features,
        min_n=200,
    )
    dca4_sites_pair_composite_corr = pairwise_composite_correlations(
        merged,
        success_col="success_dca4_sites",
        features=primary_pair_corr_features,
        min_n=200,
    )
    tns_pair_composite_corr = pairwise_composite_correlations(
        merged,
        success_col="success_tns",
        features=primary_pair_corr_features,
        min_n=200,
    )
    tnm_pair_composite_corr = pairwise_composite_correlations(
        merged,
        success_col="success_tnm",
        features=primary_pair_corr_features,
        min_n=200,
    )

    try:
        merged.to_csv(tables_dir / "p2rank_st100_merged_per_pocket.csv", index=False)
    except PermissionError:
        print("[WARN] Could not overwrite p2rank_st100_merged_per_pocket.csv (file may be open); continuing with downstream tables.")
    summary.to_csv(tables_dir / "p2rank_summary.csv", index=False)
    success_rates_tidy.to_csv(tables_dir / "p2rank_success_rates_reported_tidy.csv", index=False)
    class_summary.to_csv(tables_dir / "p2rank_class_success_by_cutoff.csv", index=False)
    dca4_bin_summary.to_csv(tables_dir / "p2rank_dca4_success_by_verbal_bins.csv", index=False)
    dca4_quantile_summary.to_csv(tables_dir / "p2rank_dca4_success_by_quantiles.csv", index=False)
    correlations.to_csv(tables_dir / "p2rank_dca4_continuous_correlations.csv", index=False)
    combined_summary.to_csv(tables_dir / "p2rank_dca4_combined_prob_sep_groups.csv", index=False)
    primary_bin_summary.to_csv(tables_dir / f"p2rank_{PRIMARY_TAG}_success_by_verbal_bins.csv", index=False)
    primary_quantile_summary.to_csv(tables_dir / f"p2rank_{PRIMARY_TAG}_success_by_quantiles.csv", index=False)
    primary_correlations.to_csv(tables_dir / f"p2rank_{PRIMARY_TAG}_continuous_correlations.csv", index=False)
    dca4_sites_correlations.to_csv(tables_dir / "p2rank_dca4_sites_continuous_correlations.csv", index=False)
    tns_correlations.to_csv(tables_dir / "p2rank_tns_continuous_correlations.csv", index=False)
    tnm_correlations.to_csv(tables_dir / "p2rank_tnm_continuous_correlations.csv", index=False)
    primary_pair_rankings.to_csv(tables_dir / f"p2rank_{PRIMARY_TAG}_pairwise_combo_rankings.csv", index=False)
    primary_pair_grids.to_csv(tables_dir / f"p2rank_{PRIMARY_TAG}_pairwise_combo_heatmap_data.csv", index=False)
    primary_pair_composite_corr.to_csv(tables_dir / f"p2rank_{PRIMARY_TAG}_pairwise_composite_correlations.csv", index=False)
    dca4_sites_pair_composite_corr.to_csv(tables_dir / "p2rank_dca4_sites_pairwise_composite_correlations.csv", index=False)
    tns_pair_composite_corr.to_csv(tables_dir / "p2rank_tns_pairwise_composite_correlations.csv", index=False)
    tnm_pair_composite_corr.to_csv(tables_dir / "p2rank_tnm_pairwise_composite_correlations.csv", index=False)
    write_manifest(
        args.output_dir,
        eval_dir=args.p2rank_eval_dir,
        union_geom_table=args.union_geom_table,
        threshold=args.threshold,
        level=args.level,
        holo_plus_one=args.holo_plus_one,
        merged=merged,
        success_rates_tidy=success_rates_tidy,
    )
    print(f"[SAVE] {args.output_dir}")


if __name__ == "__main__":
    main()
