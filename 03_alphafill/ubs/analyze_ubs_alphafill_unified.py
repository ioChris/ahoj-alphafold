# -*- coding: utf-8 -*-
"""
Analyze UBS-level AlphaFill fit versus AlphaFold/AHoJ pocket classification support.

This script is the clean second step after
`build_ubs_alphafill_analysis_tables.py`.

Workflow
--------
1. Start from UBS-level summary rows and matched long AlphaFill rows.
2. Optionally restrict AlphaFill rows to same-UniProt matches only.
3. Apply identity threshold to AlphaFill rows.
4. Aggregate surviving transplants back to one row per UBS.
5. Combine matched and unmatched UBSs into a per-threshold analysis table.

Outputs
-------
- per_ubs_summary_<scope>_<identity>.csv
- bin_summary_<scope>_<identity>.csv
- class_summary_<scope>_<identity>.csv
- correlation_summary_<scope>.csv
- threshold_identity_overview_<scope>.csv
- analysis_manifest_<scope>.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

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


DEFAULT_ALL_SUMMARY = cfg.ALPHAFILL_UNIFIED_DIR / "ubs_alphafill_all_summary.csv"
DEFAULT_MATCHED_LONG = cfg.ALPHAFILL_UNIFIED_DIR / "ubs_alphafill_matched_long.csv"
DEFAULT_OUTPUT_DIR = cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR

FIT_NUMERIC_COLS = [
    "afill_clash_count",
    "afill_clash_score",
    "afill_local_rmsd",
    "afill_mapping_global_rmsd",
    "afill_mapping_align_identity",
    "afill_pae_mean",
    "afill_pae_median",
]

SUPPORT_COLS = [
    "pocket_p_apo",
    "pocket_p_holo",
    "pocket_score",
    "pocket_density_pair",
    "pocket_separation_pair",
    "model_pocket_plddt",
    "n_apo_avg",
    "n_holo_avg",
]

X_CORR_COLS = [
    "avg_clash",
    "median_clash",
    "avg_clash_count",
    "avg_local_rmsd",
    "median_local_rmsd",
    "avg_global_rmsd",
    "avg_pae_mean",
    "avg_pae_median",
    "mean_identity",
    "max_identity",
    "min_identity",
]

Y_CORR_COLS = [
    "pocket_p_apo",
    "pocket_p_holo",
    "winning_p",
    "pocket_score",
    "pocket_density_combined",
    "pocket_density_pair",
    "pocket_density_strongest",
    "pocket_overlap_mode",
    "pocket_overlap_overall",
    "pocket_overlap_pair",
    "pocket_separation_pair",
    "model_pocket_plddt",
    "n_apo_avg",
    "n_holo_avg",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze UBS-level AlphaFill fit versus pocket classification support.")
    p.add_argument("--all_summary_csv", type=Path, default=DEFAULT_ALL_SUMMARY)
    p.add_argument("--matched_long_csv", type=Path, default=DEFAULT_MATCHED_LONG)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--thresholds", nargs="*", default=None, help="Optional subset of thresholds to analyze.")
    p.add_argument(
        "--identity_thresholds",
        nargs="*",
        type=float,
        default=[0.3, 0.5, 0.7, 0.9],
        help="Identity thresholds to sweep.",
    )
    p.add_argument("--include_any_identity", action="store_true", help="Also run an 'any' mode without identity filtering.")
    p.add_argument(
        "--match_scope",
        choices=["same_uniprot", "any"],
        default="same_uniprot",
        help="Which AlphaFill matches to consider. Default uses same-UniProt matches only.",
    )
    p.add_argument("--low_clash_cutoff", type=float, default=0.5, help="Low-clash success cutoff.")
    p.add_argument("--min_transplants_per_ubs", type=int, default=1, help="Minimum surviving transplants per matched UBS.")
    p.add_argument(
        "--prob_label_mode",
        choices=["linear", "sigmoid"],
        default="sigmoid",
        help="Database-facing probability verbalization mode. Default matches confidence plots.",
    )
    p.add_argument(
        "--prob_gain",
        type=float,
        default=1.64,
        help="Sigmoid gain for probability verbalization when --prob_label_mode=sigmoid.",
    )
    return p.parse_args()


def safe_tag_identity(thr: float | None) -> str:
    if thr is None:
        return "any_identity"
    return f"ge_{thr:.2f}".replace(".", "p")


def ordered_unique_join(values: Iterable[object], sep: str = ";") -> str:
    seen: list[str] = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s not in seen:
            seen.append(s)
    return sep.join(seen)


def keep_thresholds(df: pd.DataFrame, thresholds: list[str] | None) -> pd.DataFrame:
    if not thresholds:
        return df.copy()
    return df.loc[df["threshold"].astype(str).isin(thresholds)].copy()


def add_support_levels(df: pd.DataFrame, *, prob_label_mode: str = "sigmoid", prob_gain: float = 1.64) -> pd.DataFrame:
    out = df.copy()

    p_apo = pd.to_numeric(out.get("pocket_p_apo"), errors="coerce")
    p_holo = pd.to_numeric(out.get("pocket_p_holo"), errors="coerce")
    out["winning_p"] = pd.concat([p_apo, p_holo], axis=1).max(axis=1)

    out["prob_level_db"] = out["winning_p"].map(lambda v: probability_level(v, mode=prob_label_mode, k=prob_gain, merge=False))
    out["prob_level"] = out["winning_p"].map(lambda v: probability_level(v, mode=prob_label_mode, k=prob_gain, merge=True))
    out["density_level_db"] = out["pocket_density_pair"].map(lambda v: support_level(v, merge=False))
    out["density_level"] = out["pocket_density_pair"].map(lambda v: support_level(v, merge=True))
    out["separation_level_db"] = out["pocket_separation_pair"].map(lambda v: support_level(v, merge=False))
    out["separation_level"] = out["pocket_separation_pair"].map(lambda v: support_level(v, merge=True))
    return out


def select_scope(long_df: pd.DataFrame, scope: str) -> pd.DataFrame:
    out = long_df.copy()
    if scope == "same_uniprot":
        out = out.loc[~out["afill_cross_uniprot"].fillna(False).astype(bool)].copy()
    return out


def aggregate_matches(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["merge_row_id"])

    df = long_df.copy()
    for col in FIT_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg = df.groupby("merge_row_id", sort=False).agg(
        n_transplants=("merge_row_id", "size"),
        n_distinct_compounds=("afill_compound_id", lambda s: int(pd.Series(s).dropna().astype(str).nunique())),
        compounds_str=("afill_compound_id", ordered_unique_join),
        analogues_str=("afill_analogue_id", ordered_unique_join),
        avg_clash=("afill_clash_score", "mean"),
        median_clash=("afill_clash_score", "median"),
        avg_clash_count=("afill_clash_count", "mean"),
        avg_local_rmsd=("afill_local_rmsd", "mean"),
        median_local_rmsd=("afill_local_rmsd", "median"),
        avg_global_rmsd=("afill_mapping_global_rmsd", "mean"),
        median_global_rmsd=("afill_mapping_global_rmsd", "median"),
        avg_pae_mean=("afill_pae_mean", "mean"),
        avg_pae_median=("afill_pae_median", "mean"),
        mean_identity=("afill_mapping_align_identity", "mean"),
        max_identity=("afill_mapping_align_identity", "max"),
        min_identity=("afill_mapping_align_identity", "min"),
    ).reset_index()
    return agg


def build_per_ubs_summary(
    all_summary_df: pd.DataFrame,
    long_df: pd.DataFrame,
    scope: str,
    identity_threshold: float | None,
    low_clash_cutoff: float,
    min_transplants_per_ubs: int,
    prob_label_mode: str = "sigmoid",
    prob_gain: float = 1.64,
) -> pd.DataFrame:
    base = all_summary_df.copy()
    long_scope = select_scope(long_df, scope)

    prefilter_counts = long_scope.groupby("merge_row_id", sort=False).size().rename("scope_prefilter_n_long_rows").reset_index()
    base = base.merge(prefilter_counts, on="merge_row_id", how="left")
    base["scope_prefilter_n_long_rows"] = base["scope_prefilter_n_long_rows"].fillna(0).astype(int)
    base["scope_has_prefilter_match"] = base["scope_prefilter_n_long_rows"].gt(0)

    if identity_threshold is not None:
        idv = pd.to_numeric(long_scope["afill_mapping_align_identity"], errors="coerce")
        long_scope = long_scope.loc[idv >= identity_threshold].copy()

    agg = aggregate_matches(long_scope)
    if min_transplants_per_ubs > 1 and not agg.empty:
        agg = agg.loc[agg["n_transplants"] >= min_transplants_per_ubs].copy()

    out = base.merge(agg, on="merge_row_id", how="left")
    out["matched"] = out["n_transplants"].fillna(0).gt(0)

    for col in ["n_transplants", "n_distinct_compounds"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)
    for col in ["compounds_str", "analogues_str"]:
        if col in out.columns:
            out[col] = out[col].fillna("")

    out["fit_success_low_clash"] = out["matched"] & pd.to_numeric(out["avg_clash"], errors="coerce").le(low_clash_cutoff)

    out["unmatched_reason"] = np.where(
        out["matched"],
        "matched",
        np.where(out["scope_has_prefilter_match"], "filtered_out", "no_prefilter_match"),
    )
    out["analysis_match_scope"] = scope
    out["analysis_identity_threshold"] = "any" if identity_threshold is None else identity_threshold
    out["analysis_low_clash_cutoff"] = low_clash_cutoff

    return add_support_levels(out, prob_label_mode=prob_label_mode, prob_gain=prob_gain)


def build_bin_summary(per_ubs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    by = ["threshold", "pocket_class", "prob_level", "density_level", "separation_level"]
    for keys, sub in per_ubs_df.groupby(by, dropna=False):
        total = len(sub)
        matched = int(sub["matched"].sum())
        unmatched = total - matched
        msub = sub.loc[sub["matched"]].copy()
        rows.append(
            {
                "threshold": keys[0],
                "pocket_class": keys[1],
                "prob_level": keys[2],
                "density_level": keys[3],
                "separation_level": keys[4],
                "n_ubs": total,
                "matched_ubs": matched,
                "unmatched_ubs": unmatched,
                "match_rate": matched / total if total else np.nan,
                "success_rate_low_clash": float(msub["fit_success_low_clash"].mean()) if len(msub) else np.nan,
                "median_avg_clash": pd.to_numeric(msub["avg_clash"], errors="coerce").median(),
                "median_avg_local_rmsd": pd.to_numeric(msub["avg_local_rmsd"], errors="coerce").median(),
                "median_avg_global_rmsd": pd.to_numeric(msub["avg_global_rmsd"], errors="coerce").median(),
                "median_max_identity": pd.to_numeric(msub["max_identity"], errors="coerce").median(),
                "median_winning_p": pd.to_numeric(sub["winning_p"], errors="coerce").median(),
                "median_pocket_density_pair": pd.to_numeric(sub["pocket_density_pair"], errors="coerce").median(),
                "median_pocket_separation_pair": pd.to_numeric(sub["pocket_separation_pair"], errors="coerce").median(),
            }
        )
    return pd.DataFrame(rows)


def build_class_summary(per_ubs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    by = ["threshold", "pocket_class"]
    for keys, sub in per_ubs_df.groupby(by, dropna=False):
        matched = sub.loc[sub["matched"]].copy()
        rows.append(
            {
                "threshold": keys[0],
                "pocket_class": keys[1],
                "n_ubs": int(len(sub)),
                "n_matched": int(sub["matched"].sum()),
                "match_rate": float(sub["matched"].mean()) if len(sub) else np.nan,
                "median_winning_p": pd.to_numeric(sub["winning_p"], errors="coerce").median(),
                "median_density_pair": pd.to_numeric(sub["pocket_density_pair"], errors="coerce").median(),
                "median_separation_pair": pd.to_numeric(sub["pocket_separation_pair"], errors="coerce").median(),
                "median_avg_clash": pd.to_numeric(matched["avg_clash"], errors="coerce").median(),
                "median_avg_local_rmsd": pd.to_numeric(matched["avg_local_rmsd"], errors="coerce").median(),
                "median_avg_global_rmsd": pd.to_numeric(matched["avg_global_rmsd"], errors="coerce").median(),
                "median_max_identity": pd.to_numeric(matched["max_identity"], errors="coerce").median(),
            }
        )
    return pd.DataFrame(rows)


def compute_correlations(df: pd.DataFrame, x_cols: list[str], y_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for threshold, sub in df.groupby("threshold", dropna=False):
        for x in x_cols:
            if x not in sub.columns:
                continue
            for y in y_cols:
                if y not in sub.columns:
                    continue
                xvals = pd.to_numeric(sub[x], errors="coerce")
                yvals = pd.to_numeric(sub[y], errors="coerce")
                mask = xvals.notna() & yvals.notna()
                n = int(mask.sum())
                pearson = xvals[mask].corr(yvals[mask], method="pearson") if n >= 3 else np.nan
                spearman = xvals[mask].corr(yvals[mask], method="spearman") if n >= 3 else np.nan
                rows.append(
                    {
                        "threshold": threshold,
                        "x": x,
                        "y": y,
                        "pearson": pearson,
                        "spearman": spearman,
                        "n": n,
                    }
                )
    return pd.DataFrame(rows)


def build_threshold_identity_overview(per_ubs_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for tag, df in per_ubs_dfs.items():
        identity_label = df["analysis_identity_threshold"].iloc[0] if not df.empty else tag
        scope = df["analysis_match_scope"].iloc[0] if not df.empty else ""
        for threshold, sub in df.groupby("threshold", dropna=False):
            rows.append(
                {
                    "identity_tag": tag,
                    "identity_threshold": identity_label,
                    "match_scope": scope,
                    "threshold": threshold,
                    "n_ubs": int(len(sub)),
                    "n_matched": int(sub["matched"].sum()),
                    "n_unmatched": int((~sub["matched"]).sum()),
                    "n_unmatched_no_prefilter_match": int((sub["unmatched_reason"] == "no_prefilter_match").sum()),
                    "n_unmatched_filtered_out": int((sub["unmatched_reason"] == "filtered_out").sum()),
                    "match_rate": float(sub["matched"].mean()) if len(sub) else np.nan,
                    "success_rate_low_clash": float(sub.loc[sub["matched"], "fit_success_low_clash"].mean()) if int(sub["matched"].sum()) else np.nan,
                    "median_avg_clash_matched": pd.to_numeric(sub.loc[sub["matched"], "avg_clash"], errors="coerce").median(),
                    "median_avg_local_rmsd_matched": pd.to_numeric(sub.loc[sub["matched"], "avg_local_rmsd"], errors="coerce").median(),
                    "median_max_identity_matched": pd.to_numeric(sub.loc[sub["matched"], "max_identity"], errors="coerce").median(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summary = pd.read_csv(args.all_summary_csv, low_memory=False)
    matched_long = pd.read_csv(args.matched_long_csv, low_memory=False)

    all_summary = keep_thresholds(all_summary, args.thresholds)
    matched_long = keep_thresholds(matched_long, args.thresholds)

    identity_runs: list[float | None] = list(args.identity_thresholds)
    if args.include_any_identity:
        identity_runs = [None] + identity_runs

    per_ubs_tables: dict[str, pd.DataFrame] = {}
    correlation_rows: list[pd.DataFrame] = []

    for thr in identity_runs:
        tag = safe_tag_identity(thr)
        per_ubs = build_per_ubs_summary(
            all_summary_df=all_summary,
            long_df=matched_long,
            scope=args.match_scope,
            identity_threshold=thr,
            low_clash_cutoff=args.low_clash_cutoff,
            min_transplants_per_ubs=args.min_transplants_per_ubs,
            prob_label_mode=args.prob_label_mode,
            prob_gain=args.prob_gain,
        )
        per_ubs_tables[tag] = per_ubs

        per_ubs_path = args.output_dir / f"per_ubs_summary_{args.match_scope}_{tag}.csv"
        bin_path = args.output_dir / f"bin_summary_{args.match_scope}_{tag}.csv"
        class_path = args.output_dir / f"class_summary_{args.match_scope}_{tag}.csv"

        per_ubs.to_csv(per_ubs_path, index=False)
        build_bin_summary(per_ubs).to_csv(bin_path, index=False)
        build_class_summary(per_ubs).to_csv(class_path, index=False)
        print(f"[SAVE] {per_ubs_path}")
        print(f"[SAVE] {bin_path}")
        print(f"[SAVE] {class_path}")

        corr_df = compute_correlations(per_ubs.loc[per_ubs["matched"]].copy(), X_CORR_COLS, Y_CORR_COLS)
        corr_df.insert(0, "identity_tag", tag)
        corr_df.insert(1, "identity_threshold", "any" if thr is None else thr)
        corr_df.insert(2, "match_scope", args.match_scope)
        correlation_rows.append(corr_df)

    corr_all = pd.concat(correlation_rows, ignore_index=True) if correlation_rows else pd.DataFrame()
    overview_df = build_threshold_identity_overview(per_ubs_tables)

    corr_path = args.output_dir / f"correlation_summary_{args.match_scope}.csv"
    overview_path = args.output_dir / f"threshold_identity_overview_{args.match_scope}.csv"
    corr_all.to_csv(corr_path, index=False)
    overview_df.to_csv(overview_path, index=False)
    print(f"[SAVE] {corr_path}")
    print(f"[SAVE] {overview_path}")

    manifest = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "all_summary_csv": str(args.all_summary_csv),
            "matched_long_csv": str(args.matched_long_csv),
        },
        "filters": {
            "thresholds": args.thresholds,
            "match_scope": args.match_scope,
            "identity_thresholds": ["any" if x is None else x for x in identity_runs],
            "low_clash_cutoff": args.low_clash_cutoff,
            "min_transplants_per_ubs": args.min_transplants_per_ubs,
            "prob_label_mode": args.prob_label_mode,
            "prob_gain": args.prob_gain,
        },
        "outputs": {
            "n_identity_runs": len(identity_runs),
            "correlation_rows": int(len(corr_all)),
            "overview_rows": int(len(overview_df)),
        },
    }
    manifest_path = args.output_dir / f"analysis_manifest_{args.match_scope}.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[SAVE] {manifest_path}")


if __name__ == "__main__":
    main()
