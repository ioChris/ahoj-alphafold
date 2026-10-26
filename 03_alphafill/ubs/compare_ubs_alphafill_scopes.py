# -*- coding: utf-8 -*-
"""
Compare rebuilt UBS AlphaFill analysis outputs across match scopes.

This script complements `plot_ubs_alphafill_unified.py` by producing direct
same-UniProt vs any-match comparisons from the unified analysis tables.

Inputs
------
- threshold_identity_overview_same_uniprot.csv
- threshold_identity_overview_any.csv
- correlation_summary_same_uniprot.csv
- correlation_summary_any.csv
- per_ubs_summary_same_uniprot_<identity>.csv
- per_ubs_summary_any_<identity>.csv

Outputs
-------
- overview comparison lines for a focus identity threshold
- delta heatmaps (any - same_uniprot) across thresholds and identity levels
- clash vs apo/holo/separation scatter comparisons for a focus threshold
- top-correlation comparison plot and CSV for a focus threshold/identity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


DEFAULT_ANALYSIS_DIR = cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR

THRESHOLD_ORDER = ["ST_030", "ST_050", "ST_070", "ST_100", "ST_LJ"]
METRIC_SPECS = [
    ("match_rate", "Match Rate"),
    ("success_rate_low_clash", "Low-Clash Success Rate"),
    ("median_avg_clash_matched", "Median Avg Clash"),
]
CLASS_COLORS = {
    "apo": "#2E5EAA",
    "holo": "#C24A3A",
    "tie": "#7B6D8D",
    "unknown": "#777777",
}
SCOPE_COLORS = {
    "same_uniprot": "#2E5EAA",
    "any": "#C24A3A",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare AlphaFill unified analysis outputs across match scopes.")
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--focus_threshold", default="ST_100")
    p.add_argument("--focus_identity_tag", default="ge_0p30")
    p.add_argument("--top_n", type=int, default=12)
    p.add_argument("--max_points", type=int, default=12000)
    return p.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


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


def pearson_text(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, n
    return float(xv[mask].corr(yv[mask], method="pearson")), n


def load_overview(analysis_dir: Path) -> pd.DataFrame:
    parts = []
    for scope in ["same_uniprot", "any"]:
        path = analysis_dir / f"threshold_identity_overview_{scope}.csv"
        df = read_csv(path)
        df["match_scope"] = scope
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_correlations(analysis_dir: Path) -> pd.DataFrame:
    parts = []
    for scope in ["same_uniprot", "any"]:
        path = analysis_dir / f"correlation_summary_{scope}.csv"
        df = read_csv(path)
        df["match_scope"] = scope
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_per_ubs(analysis_dir: Path, identity_tag: str) -> pd.DataFrame:
    parts = []
    for scope in ["same_uniprot", "any"]:
        path = analysis_dir / f"per_ubs_summary_{scope}_{identity_tag}.csv"
        df = read_csv(path)
        df["match_scope"] = scope
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def plot_focus_identity_scope_lines(overview_df: pd.DataFrame, outdir: Path, identity_tag: str) -> None:
    sub = overview_df.loc[overview_df["identity_tag"] == identity_tag].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    x = np.arange(len(THRESHOLD_ORDER))
    for ax, (metric, ylabel) in zip(axes, METRIC_SPECS):
        for scope in ["same_uniprot", "any"]:
            part = sub.loc[sub["match_scope"] == scope].copy()
            part = part.set_index("threshold").reindex(THRESHOLD_ORDER).reset_index()
            ax.plot(
                x,
                pd.to_numeric(part[metric], errors="coerce"),
                marker="o",
                linewidth=2.0,
                label=scope,
                color=SCOPE_COLORS[scope],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(THRESHOLD_ORDER, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} ({identity_tag})")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    out = outdir / f"scope_comparison_lines_{identity_tag}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmaps(overview_df: pd.DataFrame, outdir: Path) -> None:
    pivoted = {}
    id_order = list(overview_df["identity_tag"].drop_duplicates())
    for metric, _ in METRIC_SPECS:
        piv = overview_df.pivot_table(
            index=["threshold", "identity_tag"],
            columns="match_scope",
            values=metric,
            aggfunc="first",
        ).reset_index()
        piv["delta_any_minus_same"] = pd.to_numeric(piv.get("any"), errors="coerce") - pd.to_numeric(
            piv.get("same_uniprot"), errors="coerce"
        )
        delta = piv.pivot(index="threshold", columns="identity_tag", values="delta_any_minus_same")
        delta = delta.reindex(index=[x for x in THRESHOLD_ORDER if x in delta.index], columns=id_order)
        pivoted[metric] = delta

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    for ax, (metric, title) in zip(axes, METRIC_SPECS):
        vmax = float(np.nanmax(np.abs(pivoted[metric].to_numpy(dtype=float)))) if not pivoted[metric].empty else 0.1
        vmax = max(vmax, 0.01)
        draw_heatmap(
            ax,
            pivoted[metric],
            title=f"any - same_uniprot: {title}",
            cmap="coolwarm",
            fmt=".2f",
            vmin=-vmax,
            vmax=vmax,
        )
    out = outdir / "scope_delta_heatmaps.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_clash_vs_support_comparison(
    per_ubs_df: pd.DataFrame,
    outdir: Path,
    focus_threshold: str,
    identity_tag: str,
    max_points: int,
) -> None:
    sub = per_ubs_df.loc[per_ubs_df["matched"]].copy()
    sub = sub.loc[sub["threshold"] == focus_threshold].copy()
    if sub.empty:
        return

    y_metrics = [
        ("pocket_p_apo", "Pocket P(apo)"),
        ("pocket_p_holo", "Pocket P(holo)"),
        ("pocket_separation_pair", "Pocket Separation (pair)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.2), constrained_layout=True, sharex="col")

    for row, scope in enumerate(["same_uniprot", "any"]):
        part = sub.loc[sub["match_scope"] == scope].copy()
        if len(part) > max_points:
            part = part.sample(n=max_points, random_state=0).copy()
        for col, (ycol, ylabel) in enumerate(y_metrics):
            ax = axes[row, col]
            for cls, grp in part.groupby("pocket_class", dropna=False):
                ax.scatter(
                    pd.to_numeric(grp["avg_clash"], errors="coerce"),
                    pd.to_numeric(grp[ycol], errors="coerce"),
                    s=12,
                    alpha=0.25,
                    color=CLASS_COLORS.get(str(cls), "#777777"),
                    label=cls if (row == 0 and col == 0) else None,
                )
            r, n = pearson_text(part["avg_clash"], part[ycol])
            label = f"r={r:.2f}, n={n}" if np.isfinite(r) else f"r=NA, n={n}"
            ax.text(
                0.03,
                0.97,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )
            if row == 1:
                ax.set_xlabel("Avg Clash Score")
            if col == 0:
                ax.set_ylabel(f"{scope}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            ax.grid(alpha=0.20)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    out = outdir / f"{focus_threshold.lower()}_clash_vs_support_scope_comparison_{identity_tag}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_top_correlation_table(corr_df: pd.DataFrame, focus_threshold: str, identity_tag: str, top_n: int) -> pd.DataFrame:
    sub = corr_df.loc[
        (corr_df["threshold"] == focus_threshold) &
        (corr_df["identity_tag"] == identity_tag)
    ].copy()
    if sub.empty:
        return sub
    sub["abs_pearson"] = pd.to_numeric(sub["pearson"], errors="coerce").abs()
    sub = sub.sort_values(["match_scope", "abs_pearson"], ascending=[True, False]).copy()
    top = (
        sub.groupby("match_scope", as_index=False, group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    cols = [
        "match_scope",
        "threshold",
        "identity_tag",
        "x",
        "y",
        "pearson",
        "spearman",
        "n",
        "abs_pearson",
    ]
    return top.loc[:, cols]


def plot_top_correlation_comparison(top_df: pd.DataFrame, outdir: Path, focus_threshold: str, identity_tag: str) -> None:
    if top_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.2), constrained_layout=True, sharex=True)
    for ax, scope in zip(axes, ["same_uniprot", "any"]):
        sub = top_df.loc[top_df["match_scope"] == scope].copy()
        labels = (sub["x"].astype(str) + " vs " + sub["y"].astype(str)).tolist()
        y = np.arange(len(sub))
        vals = pd.to_numeric(sub["pearson"], errors="coerce").to_numpy(dtype=float)
        colors = ["#2E8B57" if v >= 0 else "#B85450" for v in vals]
        ax.barh(y, vals, color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(scope)
        ax.set_xlabel("Pearson r")
        ax.grid(axis="x", alpha=0.25)
        lim = max(0.15, float(np.nanmax(np.abs(vals))) * 1.15 if len(vals) else 0.15)
        ax.set_xlim(-lim, lim)
    fig.suptitle(f"Top {len(top_df) // 2} Correlations: {focus_threshold} ({identity_tag})")
    out = outdir / f"{focus_threshold.lower()}_top_correlations_scope_comparison_{identity_tag}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = args.output_dir or (args.analysis_dir / "plots_compare")
    outdir.mkdir(parents=True, exist_ok=True)

    overview_df = load_overview(args.analysis_dir)
    corr_df = load_correlations(args.analysis_dir)
    per_ubs_df = load_per_ubs(args.analysis_dir, args.focus_identity_tag)

    plot_focus_identity_scope_lines(overview_df, outdir=outdir, identity_tag=args.focus_identity_tag)
    plot_delta_heatmaps(overview_df, outdir=outdir)
    plot_clash_vs_support_comparison(
        per_ubs_df,
        outdir=outdir,
        focus_threshold=args.focus_threshold,
        identity_tag=args.focus_identity_tag,
        max_points=args.max_points,
    )

    top_df = build_top_correlation_table(
        corr_df,
        focus_threshold=args.focus_threshold,
        identity_tag=args.focus_identity_tag,
        top_n=args.top_n,
    )
    if not top_df.empty:
        out_csv = outdir / f"{args.focus_threshold.lower()}_top_correlations_scope_comparison_{args.focus_identity_tag}.csv"
        top_df.to_csv(out_csv, index=False)
        plot_top_correlation_comparison(
            top_df,
            outdir=outdir,
            focus_threshold=args.focus_threshold,
            identity_tag=args.focus_identity_tag,
        )

    print(f"[SAVE] {outdir}")


if __name__ == "__main__":
    main()
