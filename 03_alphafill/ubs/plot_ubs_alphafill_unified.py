# -*- coding: utf-8 -*-
"""
Plot compact, high-signal figures from the rebuilt UBS AlphaFill analysis tables.

Inputs
------
- threshold_identity_overview_<scope>.csv
- class_summary_<scope>_<identity>.csv
- bin_summary_<scope>_<identity>.csv

Outputs
-------
- overview heatmaps/line plots across thresholds and identity levels
- per-identity class match-rate plots
- focused ST_100 class/separation heatmaps
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence_labeling import FULL_LEVEL_ORDER, MERGED_LEVEL_ORDER, pretty_level

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg


DEFAULT_ANALYSIS_DIR = cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR

THRESHOLD_ORDER = ["ST_030", "ST_050", "ST_070", "ST_100", "ST_LJ"]
CLASS_ORDER = ["apo", "holo", "tie"]
LEVEL_ORDER = ["low", "moderate", "high", "very_high", "unknown"]
SUPPORT_LEVEL_ORDER = ["low", "moderate", "high", "unknown"]
CLASS_COLORS = {
    "apo": "#2E5EAA",
    "holo": "#C24A3A",
    "tie": "#7B6D8D",
    "unknown": "#777777",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot compact AlphaFill-unified analysis figures.")
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--match_scope", default="same_uniprot")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--focus_threshold", default="ST_100")
    p.add_argument("--focus_identity_tag", default="ge_0p30")
    p.add_argument(
        "--level_mode",
        choices=["merged", "full"],
        default="merged",
        help="Use merged bins or full database-facing verbal threshold bins in level-based plots.",
    )
    return p.parse_args()


def safe_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def pivot_ordered(df: pd.DataFrame, index: str, columns: str, values: str, index_order: list[str], col_order: list[str]) -> pd.DataFrame:
    piv = df.pivot(index=index, columns=columns, values=values)
    piv = piv.reindex(index=[x for x in index_order if x in piv.index] + [x for x in piv.index if x not in index_order])
    piv = piv.reindex(columns=[x for x in col_order if x in piv.columns] + [x for x in piv.columns if x not in col_order])
    return piv


def draw_heatmap(ax, piv: pd.DataFrame, title: str, cmap: str, fmt: str = ".2f", vmin=None, vmax=None) -> None:
    data = piv.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(list(piv.columns), rotation=30, ha="right")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(list(piv.index))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=8, color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def level_config(level_mode: str) -> tuple[str, list[str]]:
    if level_mode == "full":
        return "separation_level_db", [x for x in FULL_LEVEL_ORDER if x != "unknown"]
    return "separation_level", [x for x in MERGED_LEVEL_ORDER if x != "unknown"]


def prettify_levels(labels: list[object]) -> list[str]:
    return [pretty_level(x) for x in labels]


def pearson_text(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, n
    return float(xv[mask].corr(yv[mask], method="pearson")), n


def plot_overview_heatmaps(overview_df: pd.DataFrame, outdir: Path, scope: str) -> None:
    id_order = list(overview_df["identity_tag"].drop_duplicates())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, value_col, title, cmap in [
        (axes[0], "match_rate", "Match Rate", "YlGnBu"),
        (axes[1], "success_rate_low_clash", "Low-Clash Success Rate", "YlOrRd"),
        (axes[2], "median_avg_clash_matched", "Median Avg Clash", "viridis_r"),
    ]:
        piv = pivot_ordered(
            overview_df,
            index="threshold",
            columns="identity_tag",
            values=value_col,
            index_order=THRESHOLD_ORDER,
            col_order=id_order,
        )
        draw_heatmap(ax, piv, title=title, cmap=cmap)
    out = outdir / f"overview_heatmaps_{scope}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overview_lines(overview_df: pd.DataFrame, outdir: Path, scope: str) -> None:
    tags = list(overview_df["identity_tag"].drop_duplicates())
    x = np.arange(len(tags))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    specs = [
        ("match_rate", "Match Rate"),
        ("success_rate_low_clash", "Low-Clash Success Rate"),
        ("median_avg_clash_matched", "Median Avg Clash"),
    ]
    for ax, (metric, ylabel) in zip(axes, specs):
        for thr in THRESHOLD_ORDER:
            sub = overview_df.loc[overview_df["threshold"] == thr].copy()
            if sub.empty:
                continue
            sub = sub.set_index("identity_tag").reindex(tags).reset_index()
            ax.plot(x, pd.to_numeric(sub[metric], errors="coerce"), marker="o", label=thr)
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    out = outdir / f"overview_lines_{scope}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_match_rates(class_df: pd.DataFrame, outdir: Path, scope: str, identity_tag: str) -> None:
    piv = pivot_ordered(
        class_df,
        index="threshold",
        columns="pocket_class",
        values="match_rate",
        index_order=THRESHOLD_ORDER,
        col_order=CLASS_ORDER,
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    x = np.arange(len(piv.index))
    width = 0.24
    for idx, cls in enumerate(piv.columns):
        ax.bar(x + (idx - (len(piv.columns) - 1) / 2) * width, piv[cls].to_numpy(dtype=float), width=width, label=cls)
    ax.set_xticks(x)
    ax.set_xticklabels(list(piv.index), rotation=25, ha="right")
    ax.set_ylabel("Match Rate")
    ax.set_title(f"Match Rate by Class — {identity_tag}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    out = outdir / f"class_match_rate_{scope}_{identity_tag}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def reduce_focus_df(per_ubs_df: pd.DataFrame, focus_threshold: str, sep_col: str) -> pd.DataFrame:
    sub = per_ubs_df.loc[per_ubs_df["threshold"] == focus_threshold].copy()
    if sub.empty:
        return sub

    rows = []
    by = ["pocket_class", sep_col]
    for keys, grp in sub.groupby(by, dropna=False):
        total = len(grp)
        matched = grp.loc[grp["matched"]].copy()
        rows.append(
            {
                "pocket_class": keys[0],
                "separation_level_plot": keys[1],
                "n_ubs": int(total),
                "match_rate": float(grp["matched"].mean()) if total else np.nan,
                "success_rate_low_clash": float(matched["fit_success_low_clash"].mean()) if len(matched) else np.nan,
                "median_avg_clash": pd.to_numeric(matched["avg_clash"], errors="coerce").median(),
            }
        )
    return pd.DataFrame(rows)


def plot_focus_heatmaps(
    per_ubs_df: pd.DataFrame,
    outdir: Path,
    scope: str,
    focus_threshold: str,
    identity_tag: str,
    level_mode: str,
) -> None:
    sep_col, sep_order = level_config(level_mode)
    reduced = reduce_focus_df(per_ubs_df, focus_threshold=focus_threshold, sep_col=sep_col)
    if reduced.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, value_col, title, cmap in [
        (axes[0], "match_rate", f"{focus_threshold} Match Rate", "YlGnBu"),
        (axes[1], "success_rate_low_clash", f"{focus_threshold} Low-Clash Success", "YlOrRd"),
        (axes[2], "median_avg_clash", f"{focus_threshold} Median Avg Clash", "viridis_r"),
    ]:
        piv = pivot_ordered(
            reduced,
            index="pocket_class",
            columns="separation_level_plot",
            values=value_col,
            index_order=CLASS_ORDER,
            col_order=sep_order,
        )
        piv.columns = prettify_levels(list(piv.columns))
        draw_heatmap(ax, piv, title=title, cmap=cmap)
    out = outdir / f"{focus_threshold.lower()}_focus_heatmaps_{scope}_{identity_tag}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    piv_counts = pivot_ordered(
        reduced,
        index="pocket_class",
        columns="separation_level_plot",
        values="n_ubs",
        index_order=CLASS_ORDER,
        col_order=sep_order,
    )
    piv_counts.columns = prettify_levels(list(piv_counts.columns))
    fig2, ax2 = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
    draw_heatmap(ax2, piv_counts, title=f"{focus_threshold} UBS Counts", cmap="Greys", fmt=".0f")
    out2 = outdir / f"{focus_threshold.lower()}_focus_counts_{scope}_{identity_tag}.png"
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig2)


def plot_focus_correlation_heatmap(corr_df: pd.DataFrame, outdir: Path, scope: str, focus_threshold: str, identity_tag: str) -> None:
    sub = corr_df.loc[
        (corr_df["threshold"] == focus_threshold) &
        (corr_df["identity_tag"] == identity_tag)
    ].copy()
    if sub.empty:
        return

    x_order = [
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
    y_order = [
        "pocket_p_apo",
        "pocket_p_holo",
        "winning_p",
        "pocket_score",
        "pocket_density_combined",
        "pocket_density_pair",
        "pocket_density_strongest",
        "pocket_overlap_pair",
        "pocket_separation_pair",
        "model_pocket_plddt",
        "n_apo_avg",
        "n_holo_avg",
    ]

    piv = sub.pivot(index="x", columns="y", values="pearson")
    piv = piv.reindex(index=[x for x in x_order if x in piv.index], columns=[y for y in y_order if y in piv.columns])
    fig, ax = plt.subplots(figsize=(11.5, 7.0), constrained_layout=True)
    draw_heatmap(ax, piv, title=f"{focus_threshold} Pearson Correlations ({identity_tag})", cmap="coolwarm", fmt=".2f", vmin=-0.35, vmax=0.35)
    out = outdir / f"{focus_threshold.lower()}_correlation_heatmap_{scope}_{identity_tag}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_focus_scatter_grid(per_ubs_df: pd.DataFrame, outdir: Path, scope: str, focus_threshold: str, identity_tag: str, max_points: int = 12000) -> None:
    sub = per_ubs_df.loc[per_ubs_df["matched"]].copy()
    sub = sub.loc[sub["threshold"] == focus_threshold].copy()
    if sub.empty:
        return
    if len(sub) > max_points:
        sub = sub.sample(n=max_points, random_state=0).copy()

    x_metrics = [
        ("avg_clash", "Avg Clash"),
        ("avg_local_rmsd", "Avg Local RMSD"),
        ("avg_pae_mean", "Avg PAE Mean"),
    ]
    y_metrics = [
        ("pocket_p_apo", "Pocket P(apo)"),
        ("pocket_p_holo", "Pocket P(holo)"),
        ("pocket_separation_pair", "Pocket Separation"),
    ]

    fig, axes = plt.subplots(len(x_metrics), len(y_metrics), figsize=(12.5, 11.0), constrained_layout=True)
    for i, (xcol, xlabel) in enumerate(x_metrics):
        for j, (ycol, ylabel) in enumerate(y_metrics):
            ax = axes[i, j]
            for cls in CLASS_ORDER:
                part = sub.loc[sub["pocket_class"] == cls].copy()
                if part.empty:
                    continue
                ax.scatter(
                    pd.to_numeric(part[xcol], errors="coerce"),
                    pd.to_numeric(part[ycol], errors="coerce"),
                    s=12,
                    alpha=0.30,
                    label=cls if i == 0 and j == 0 else None,
                    color=CLASS_COLORS.get(cls, "#777777"),
                )
            r, n = pearson_text(sub[xcol], sub[ycol])
            label = f"r={r:.2f}, n={n}" if np.isfinite(r) else f"r=NA, n={n}"
            ax.text(0.03, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
            if i == len(x_metrics) - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.grid(alpha=0.20)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    out = outdir / f"{focus_threshold.lower()}_scatter_grid_{scope}_{identity_tag}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        outdir = args.output_dir
    elif args.level_mode == "merged":
        outdir = args.analysis_dir / f"plots_{args.match_scope}"
    else:
        outdir = args.analysis_dir / f"plots_{args.match_scope}_{args.level_mode}"
    outdir.mkdir(parents=True, exist_ok=True)

    overview_path = args.analysis_dir / f"threshold_identity_overview_{args.match_scope}.csv"
    class_path = args.analysis_dir / f"class_summary_{args.match_scope}_{args.focus_identity_tag}.csv"
    corr_path = args.analysis_dir / f"correlation_summary_{args.match_scope}.csv"
    per_ubs_path = args.analysis_dir / f"per_ubs_summary_{args.match_scope}_{args.focus_identity_tag}.csv"

    overview_df = read_csv(overview_path)
    class_df = read_csv(class_path)
    corr_df = read_csv(corr_path)
    per_ubs_df = read_csv(per_ubs_path)

    plot_overview_heatmaps(overview_df, outdir=outdir, scope=args.match_scope)
    plot_overview_lines(overview_df, outdir=outdir, scope=args.match_scope)
    plot_class_match_rates(class_df, outdir=outdir, scope=args.match_scope, identity_tag=args.focus_identity_tag)
    plot_focus_heatmaps(
        per_ubs_df,
        outdir=outdir,
        scope=args.match_scope,
        focus_threshold=args.focus_threshold,
        identity_tag=args.focus_identity_tag,
        level_mode=args.level_mode,
    )
    plot_focus_correlation_heatmap(
        corr_df,
        outdir=outdir,
        scope=args.match_scope,
        focus_threshold=args.focus_threshold,
        identity_tag=args.focus_identity_tag,
    )
    plot_focus_scatter_grid(
        per_ubs_df,
        outdir=outdir,
        scope=args.match_scope,
        focus_threshold=args.focus_threshold,
        identity_tag=args.focus_identity_tag,
    )

    print(f"[SAVE] {outdir}")


if __name__ == "__main__":
    main()
